import argparse
import datetime
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Iterable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from pretraining.data_pipeline import (
    BinaryDatasetConfig,
    BinaryDatasetMode,
    DataPolicy,
    resolve_binary_block_size,
    resolve_binary_dataloader_workers,
)
from pretraining.dataset_registry import dataset_hf_cache_root, dataset_label_for_basename
from pretraining.run_types import (
    Branch,
    BranchRunConfig,
    InitMode,
    LossMode,
    ModePlannerConfig,
    RunSweep,
    SSLProfile,
    TrainingMode,
    is_contrastive_loss,
)
from pretraining.training_modes import plan_runs_for_spec
from pretraining.stradavit_trainer import guarded_print, run_ssl_training
from utils.model_utils import cleanup_after_run
from utils.strada_datasets import DEFAULT_MEGA_CHUNK_BYTES, RECORD_BYTES


PROGRESS_FILE = os.getenv("STRADA_PROGRESS_FILE") or os.path.join(os.path.dirname(__file__), "strada_sweep_progress.json")
METRICS_PLAN_FILE = os.getenv("STRADA_METRICS_PLAN_FILE") or os.path.join(os.path.dirname(__file__), "strada_metrics_plan.json")
FATAL_ERROR_FILE = os.getenv("STRADA_FATAL_ERROR_FILE") or "fatal_error.log"

MAE_MODEL_NAME = "facebook/vit-mae-base"
DINOV2_MODEL_NAME = "facebook/dinov2-base"
DINOV2_REG_MODEL_NAME = "facebook/dinov2-with-registers-base"

DATASET_BIN_PATHS = {
    BinaryDatasetMode.DEFAULT: "/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_v5.bin",
    BinaryDatasetMode.CURATED: "/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_curated.bin",
}

VISIBLE_MAE_LOSSES = (LossMode.L2, LossMode.L2_L1, LossMode.L2_BL1)
VISIBLE_CONTRASTIVE_LOSSES = (LossMode.HCL, LossMode.SOFT_HCL, LossMode.SIMCLR)


@dataclass(frozen=True)
class GuidedConfig:
    branch: Branch
    training_modes: tuple[TrainingMode, ...]
    mae_loss_modes: tuple[LossMode, ...]
    contrastive_loss_modes: tuple[LossMode, ...]
    use_roi_crop: bool
    enable_augmentations: bool
    registers: tuple[int, ...]
    contrastive_views: tuple[int, ...]
    dataset_bin_path: str
    dataset_label: str
    dry_plan: bool
    dry_plan_json: bool


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_branch(raw: str) -> Branch:
    key = raw.strip().lower()
    if key == "dino":
        key = Branch.DINOV2.value
    try:
        return Branch(key)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("branch must be one of: mae, dinov2, dino") from exc


def _parse_training_modes(raw: str) -> tuple[TrainingMode, ...]:
    try:
        values = tuple(TrainingMode(item) for item in _csv(raw))
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in TrainingMode)
        raise argparse.ArgumentTypeError(f"training modes must be from: {allowed}") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one training mode is required")
    return values


def _parse_loss_modes(raw: str, allowed: Iterable[LossMode], label: str) -> tuple[LossMode, ...]:
    allowed = tuple(allowed)
    by_value = {mode.value: mode for mode in allowed}
    values: list[LossMode] = []
    for item in _csv(raw):
        try:
            values.append(by_value[item])
        except KeyError as exc:
            allowed_text = ", ".join(mode.value for mode in allowed)
            raise argparse.ArgumentTypeError(f"{label} loss modes must be from: {allowed_text}") from exc
    return tuple(values)


def _parse_on_off(raw: str) -> bool:
    key = raw.strip().lower()
    if key in ("on", "true", "1", "yes"):
        return True
    if key in ("off", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("value must be on or off")


def _parse_views(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in _csv(raw):
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"contrastive view count must be an integer, got {item!r}") from exc
        if value < 2:
            raise argparse.ArgumentTypeError("contrastive view count must be >= 2")
        values.append(value)
    if not values:
        values.append(2)
    return tuple(values)


def _parse_registers(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in _csv(raw):
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"register count must be 0 or 4, got {item!r}") from exc
        if value not in (0, 4):
            raise argparse.ArgumentTypeError("register count must be 0 or 4")
        values.append(value)
    if not values:
        values.append(0)
    return tuple(values)


def _parse_dataset_bin_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return DATASET_BIN_PATHS[BinaryDatasetMode.DEFAULT]
    if not value.endswith(".bin"):
        raise argparse.ArgumentTypeError("dataset bin path must point to a .bin file")
    return value


def parse_args() -> GuidedConfig:
    parser = argparse.ArgumentParser(description="Guided Strada SSL training entrypoint.")
    parser.add_argument("--branch", type=_parse_branch, required=True)
    parser.add_argument("--training-modes", type=_parse_training_modes, required=True)
    parser.add_argument(
        "--mae-loss-modes",
        default="",
        type=lambda raw: _parse_loss_modes(raw, VISIBLE_MAE_LOSSES, "MAE"),
    )
    parser.add_argument(
        "--contrastive-loss-modes",
        default="",
        type=lambda raw: _parse_loss_modes(raw, VISIBLE_CONTRASTIVE_LOSSES, "contrastive"),
    )
    parser.add_argument("--roi-crop", type=_parse_on_off, default=True)
    parser.add_argument("--augmentations", type=_parse_on_off, default=True)
    parser.add_argument("--registers", type=_parse_registers, default=(0,))
    parser.add_argument("--contrastive-views", type=_parse_views, default=(2,))
    parser.add_argument("--dataset-bin-path", type=_parse_dataset_bin_path, default=DATASET_BIN_PATHS[BinaryDatasetMode.DEFAULT])
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--dry-plan-json", action="store_true")
    ns = parser.parse_args()
    dataset_label = str(ns.dataset_label or "").strip()
    if not dataset_label:
        dataset_label = dataset_label_for_basename(os.path.basename(ns.dataset_bin_path))[0]
    return GuidedConfig(
        branch=ns.branch,
        training_modes=ns.training_modes,
        mae_loss_modes=ns.mae_loss_modes,
        contrastive_loss_modes=ns.contrastive_loss_modes,
        use_roi_crop=bool(ns.roi_crop),
        enable_augmentations=bool(ns.augmentations),
        registers=tuple(ns.registers),
        contrastive_views=tuple(ns.contrastive_views),
        dataset_bin_path=ns.dataset_bin_path,
        dataset_label=dataset_label,
        dry_plan=bool(ns.dry_plan),
        dry_plan_json=bool(ns.dry_plan_json),
    )


def validate_config(cfg: GuidedConfig) -> None:
    if cfg.branch == Branch.DINOV2:
        invalid = [mode.value for mode in cfg.training_modes if mode != TrainingMode.CONTRASTIVE_ONLY]
        if invalid:
            raise ValueError(f"DINO only supports contrastive_only; got {invalid}.")
        if cfg.mae_loss_modes:
            raise ValueError("DINO does not use MAE loss modes.")
    for mode in cfg.training_modes:
        if mode == TrainingMode.MAE_ONLY and not cfg.mae_loss_modes:
            raise ValueError("mae_only requires at least one MAE loss mode.")
        if mode == TrainingMode.CONTRASTIVE_ONLY and not cfg.contrastive_loss_modes:
            raise ValueError("contrastive_only requires at least one contrastive loss mode.")
        if mode == TrainingMode.MAE_THEN_CONTRASTIVE:
            if not cfg.mae_loss_modes or not cfg.contrastive_loss_modes:
                raise ValueError("mae_then_contrastive requires MAE and contrastive loss modes.")
        if mode in (TrainingMode.CONTRASTIVE_ONLY, TrainingMode.MAE_THEN_CONTRASTIVE):
            if any(views < 2 for views in cfg.contrastive_views):
                raise ValueError("contrastive view counts must be >= 2.")


def build_branch_config(cfg: GuidedConfig, registers: int) -> BranchRunConfig:
    dataset_mode = BinaryDatasetMode.CURATED if cfg.dataset_label.strip().lower() == BinaryDatasetMode.CURATED.value else BinaryDatasetMode.DEFAULT
    dataset_config = BinaryDatasetConfig(
        mode=dataset_mode,
        bin_path_override=cfg.dataset_bin_path,
        label=cfg.dataset_label,
    )
    data_policy = DataPolicy(
        use_roi_crop=cfg.use_roi_crop,
        enable_augmentations=cfg.enable_augmentations,
    )
    if cfg.branch == Branch.MAE:
        return BranchRunConfig(
            branch=Branch.MAE,
            model_name=MAE_MODEL_NAME,
            train_profile=SSLProfile.CONTINUED,
            init_mode=InitMode.FACEBOOK_HF,
            mae_learning_rate=5e-4,
            contrastive_learning_rate=5e-4,
            training_modes=cfg.training_modes,
            mae_only_loss_modes=cfg.mae_loss_modes,
            contrastive_loss_modes=cfg.contrastive_loss_modes,
            patch_sizes=(16,),
            masks=(0.75,),
            n_registers_list=(registers,),
            dataset_config=dataset_config,
            data_policy=data_policy,
            use_dino_encoder=False,
            drop_path_rate=0.1,
        )

    return BranchRunConfig(
        branch=Branch.DINOV2,
        model_name=DINOV2_REG_MODEL_NAME if registers == 4 else DINOV2_MODEL_NAME,
        train_profile=SSLProfile.CONTINUED,
        init_mode=InitMode.DINOV2_HF,
        mae_learning_rate=5e-4,
        contrastive_learning_rate=1e-4,
        training_modes=cfg.training_modes,
        mae_only_loss_modes=(),
        contrastive_loss_modes=cfg.contrastive_loss_modes,
        patch_sizes=(14,),
        masks=(0.0,),
        n_registers_list=(registers,),
        dataset_config=dataset_config,
        data_policy=data_policy,
    )


def _branch_bs_map(branch: Branch, views: int) -> dict[LossMode, tuple[int, int]]:
    if branch == Branch.MAE:
        return {
            LossMode.L2: (256, 256),
            LossMode.L2_L1: (256, 256),
            LossMode.L2_BL1: (256, 256),
            LossMode.SIMCLR: (64 if views == 2 else 48, 64 if views == 2 else 48),
            LossMode.SOFT_HCL: (64 if views == 2 else 48, 64 if views == 2 else 48),
            LossMode.HCL: (64 if views == 2 else 48, 64 if views == 2 else 48),
        }
    return {
        LossMode.SIMCLR: (20 if views == 2 else 16, 20 if views == 2 else 16),
        LossMode.SOFT_HCL: (20 if views == 2 else 16, 20 if views == 2 else 16),
        LossMode.HCL: (20 if views == 2 else 16, 20 if views == 2 else 16),
    }


def _base_init_token(branch_cfg: BranchRunConfig) -> str:
    if branch_cfg.init_mode == InitMode.FACEBOOK_HF:
        return InitMode.FACEBOOK_HF.value
    if branch_cfg.init_mode == InitMode.DINOV2_HF:
        return InitMode.DINOV2_HF.value
    if branch_cfg.init_mode == InitMode.SCRATCH_HF:
        return InitMode.SCRATCH_HF.value
    raise ValueError(f"Unsupported branch init mode for direct stages: {branch_cfg.init_mode}")


def build_planned_runs(cfg: GuidedConfig):
    loss_weights = {"l2": 1.0, "l1": 0.1, "bl1": 0.1}
    hf_cache_path = dataset_hf_cache_root(cfg.dataset_bin_path)
    planned = []
    first_views = cfg.contrastive_views[0]

    for registers in cfg.registers:
        branch_cfg = build_branch_config(cfg, registers)
        for views in cfg.contrastive_views:
            if views == first_views:
                training_modes = branch_cfg.training_modes
            else:
                training_modes = tuple(
                    mode
                    for mode in branch_cfg.training_modes
                    if mode in (TrainingMode.CONTRASTIVE_ONLY, TrainingMode.MAE_THEN_CONTRASTIVE)
                )
            if not training_modes:
                continue
            sweep = RunSweep(
                model_name=branch_cfg.model_name,
                profile=branch_cfg.train_profile,
                training_modes=training_modes,
                mae_only_loss_modes=branch_cfg.mae_only_loss_modes,
                contrastive_only_loss_modes=branch_cfg.contrastive_loss_modes,
                two_phase_mae_loss_modes=branch_cfg.mae_only_loss_modes,
                two_phase_contrastive_loss_modes=branch_cfg.contrastive_loss_modes,
                patch_sizes=branch_cfg.patch_sizes or (16,),
                masks=branch_cfg.masks or (0.0,),
                n_views=views,
                projector_hidden_dim=2048,
                projector_out_dim=128,
                preview_samples=128,
                preview_repeats=10,
                n_registers_list=branch_cfg.n_registers_list,
                dataset_config=branch_cfg.dataset_config,
                data_policy=branch_cfg.data_policy,
            )
            planner_config = ModePlannerConfig(
                branch=branch_cfg.branch,
                hf_cache_path=hf_cache_path,
                run_tag=None,
                dataloader_num_workers=_env_int("STRADA_DATALOADER_NUM_WORKERS", 8),
                seed=42,
                loss_weights=loss_weights,
                use_dino_encoder=bool(branch_cfg.use_dino_encoder),
                drop_path_rate=branch_cfg.drop_path_rate,
                base_init_token=_base_init_token(branch_cfg),
                learning_rate_mae=float(branch_cfg.mae_learning_rate),
                learning_rate_contrastive=float(branch_cfg.contrastive_learning_rate),
                warmup_ratio_mae=0.15,
                warmup_ratio_contrastive=0.15,
                num_train_epochs_mae=35,
                num_train_epochs_contrastive=35,
                branch_bs_map=_branch_bs_map(branch_cfg.branch, views),
                soft_hcl_alpha=0.5,
                soft_hcl_tau_h=0.15,
                simclr_debias_tau_plus=0.10,
                hcl_beta=1.0,
                hcl_tau_plus=None,
                contrastive_diagnostics_enabled=True,
                heavy_contrastive_diagnostics=True,
            )
            for spec in sweep.iter_specs():
                for planned_run in plan_runs_for_spec(spec, planner_config):
                    planned.append((spec, planned_run))

    return tuple(planned)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def checkpoint_exists(ckpt_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin"))
        or os.path.exists(os.path.join(ckpt_dir, "model.safetensors"))
    )


def write_progress(*, total: int, current: int, status: str, spec, planned_run, error: str | None = None) -> None:
    payload = {
        "total": int(total),
        "current": int(current),
        "status": status,
        "branch": planned_run.run_config.branch.value,
        "training_mode": spec.training_mode.value,
        "stage": planned_run.stage_name,
        "loss_modes": [loss.value for loss in spec.loss_modes],
        "registers": int(spec.n_registers),
        "contrastive_views": int(spec.n_views),
        "run_root": planned_run.run_root,
        "started_at": getattr(write_progress, "_started_at", None),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if error:
        payload["error"] = error
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_metrics_plan(planned_runs) -> None:
    stages = []
    total = len(planned_runs)
    for idx, (spec, planned_run) in enumerate(planned_runs, start=1):
        run_config = planned_run.run_config
        stages.append(
            {
                "index": idx,
                "total": total,
                "branch": run_config.branch.value,
                "training_mode": spec.training_mode.value,
                "stage": planned_run.stage_name,
                "loss_modes": [run_config.loss_mode.value],
                "recipe_loss_modes": [loss.value for loss in spec.loss_modes],
                "registers": int(spec.n_registers),
                "contrastive_views": int(spec.n_views),
                "run_root": planned_run.run_root,
                "logs_dir": os.path.join(planned_run.run_root, "logs"),
                "planned_epochs": int(run_config.num_train_epochs),
                "checkpoint_dir": planned_run.checkpoint_dir,
            }
        )
    payload = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total": total,
        "stages": stages,
    }
    os.makedirs(os.path.dirname(METRICS_PLAN_FILE), exist_ok=True)
    with open(METRICS_PLAN_FILE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def print_dry_plan(planned_runs) -> None:
    guarded_print(f"[DryPlan] planned_stages={len(planned_runs)}")
    for idx, (spec, planned_run) in enumerate(planned_runs, start=1):
        losses = ",".join(loss.value for loss in spec.loss_modes)
        guarded_print(
            f"[DryPlan] {idx}/{len(planned_runs)} "
            f"branch={planned_run.run_config.branch.value} "
            f"mode={spec.training_mode.value} stage={planned_run.stage_name} "
            f"losses={losses} regs={spec.n_registers} views={spec.n_views} "
            f"run_root={planned_run.run_root}"
        )


def dry_plan_payload(planned_runs) -> dict:
    stages = []
    total = len(planned_runs)
    for idx, (spec, planned_run) in enumerate(planned_runs, start=1):
        run_config = planned_run.run_config
        dataset_bin_path = run_config.dataset_config.bin_path
        effective_block_size = int(run_config.dataset_config.block_size)
        effective_workers = int(run_config.dataloader_num_workers)
        try:
            effective_block_size, _, _ = resolve_binary_block_size(
                dataset_bin_path,
                int(run_config.dataset_config.block_size),
            )
        except Exception:
            pass
        try:
            effective_workers, _, _ = resolve_binary_dataloader_workers(
                dataset_bin_path,
                int(run_config.dataloader_num_workers),
            )
        except Exception:
            pass
        mega_chunk_samples = max(1, int(DEFAULT_MEGA_CHUNK_BYTES // int(RECORD_BYTES)))
        mega_chunk_count = 0
        try:
            dataset_bytes = os.path.getsize(dataset_bin_path)
            dataset_samples = dataset_bytes // int(RECORD_BYTES)
            mega_chunk_count = int(math.ceil(dataset_samples / mega_chunk_samples)) if dataset_samples > 0 else 0
        except Exception:
            pass
        epochs_per_mega_chunk = int(run_config.num_train_epochs)
        internal_trainer_epochs = int(max(1, mega_chunk_count) * max(1, epochs_per_mega_chunk))
        stages.append(
            {
                "index": idx,
                "total": total,
                "branch": run_config.branch.value,
                "training_mode": spec.training_mode.value,
                "stage": planned_run.stage_name,
                "loss_modes": [run_config.loss_mode.value],
                "recipe_loss_modes": [loss.value for loss in spec.loss_modes],
                "registers": int(spec.n_registers),
                "contrastive_views": int(spec.n_views),
                "run_root": planned_run.run_root,
                "logs_dir": os.path.join(planned_run.run_root, "logs"),
                "planned_epochs": int(run_config.num_train_epochs),
                "configured_full_file_epochs": int(run_config.num_train_epochs),
                "internal_trainer_epochs": int(internal_trainer_epochs),
                "epoch_semantics": "mega_chunk_local",
                "checkpoint_dir": planned_run.checkpoint_dir,
                "dataset_bin_path": dataset_bin_path,
                "dataset_label": str(run_config.dataset_config.label or ""),
                "dataset_block_size": int(effective_block_size),
                "dataset_configured_block_size": int(run_config.dataset_config.block_size),
                "dataloader_workers_per_rank": int(effective_workers),
                "dataloader_configured_workers_per_rank": int(run_config.dataloader_num_workers),
                "mega_chunk_bytes": int(DEFAULT_MEGA_CHUNK_BYTES),
                "mega_chunk_samples": int(mega_chunk_samples),
                "mega_chunk_count": int(mega_chunk_count),
                "mega_chunk_order_policy": "rotate",
                "mega_chunk_order_period_epochs": 1,
                "epochs_per_mega_chunk": int(epochs_per_mega_chunk),
            }
        )
    return {"total": total, "stages": stages}


def main() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    cfg = parse_args()
    validate_config(cfg)
    planned_runs = build_planned_runs(cfg)
    if not planned_runs:
        raise ValueError("No runs were planned from the selected guided configuration.")
    if cfg.dry_plan_json:
        print(json.dumps(dry_plan_payload(planned_runs), separators=(",", ":")))
        return
    if cfg.dry_plan:
        print_dry_plan(planned_runs)
        return

    write_metrics_plan(planned_runs)
    write_progress._started_at = datetime.datetime.utcnow().isoformat() + "Z"
    total = len(planned_runs)
    for idx, (spec, planned_run) in enumerate(planned_runs, start=1):
        losses = ",".join(loss.value for loss in spec.loss_modes)
        try:
            if planned_run.skip_if_checkpoint_exists and checkpoint_exists(planned_run.checkpoint_dir):
                guarded_print(
                    f"[SweepProgress] {idx}/{total} skipped {spec.training_mode.value} "
                    f"{planned_run.stage_name} losses={losses} checkpoint={planned_run.checkpoint_dir}"
                )
                write_progress(total=total, current=idx, status="skipped", spec=spec, planned_run=planned_run)
                continue

            guarded_print(
                f"[SweepProgress] {idx}/{total} starting {spec.training_mode.value} "
                f"{planned_run.stage_name} losses={losses}"
            )
            write_progress(total=total, current=idx, status="running", spec=spec, planned_run=planned_run)
            if planned_run.cleanup_before_run:
                cleanup_after_run(logger=guarded_print, scoped_objects=[])
            run_ssl_training(planned_run.run_config)
            guarded_print(
                f"[SweepProgress] {idx}/{total} completed {spec.training_mode.value} "
                f"{planned_run.stage_name} losses={losses}"
            )
            write_progress(total=total, current=idx, status="completed", spec=spec, planned_run=planned_run)
        except Exception as exc:
            write_progress(
                total=total,
                current=idx,
                status="failed",
                spec=spec,
                planned_run=planned_run,
                error=str(exc),
            )
            raise

    cleanup_after_run(logger=guarded_print, scoped_objects=[])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            with open(FATAL_ERROR_FILE, "a") as f:
                f.write(traceback.format_exc())
        finally:
            cleanup_after_run(logger=guarded_print, scoped_objects=[])
        raise
