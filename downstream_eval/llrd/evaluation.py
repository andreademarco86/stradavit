import hashlib
import inspect
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoConfig, AutoImageProcessor

import downstream_eval.llrd.settings as settings
from pretraining.data_pipeline import basic_flips
from utils.strada_datasets import LoTSSDR2HortonDataset, MiraBestDatasetV2, RadioGalaxyDataset
from downstream_eval.llrd.cache import save_cache
from downstream_eval.llrd.settings import (
    ANALYSIS_SAVE_COMBINED_LP_FT,
    BATCH_SIZE,
    DATASET_MODE,
    DATA_INDEX_CACHE_DIR,
    DATA_INDEX_WORKERS,
    DEBUG_LAYER_DECAY,
    EVAL_MODE,
    FINETUNE_EPOCHS,
    FINETUNE_WEIGHT_DECAY,
    GLOBAL_SEED,
    LOTSS_HORTON_DATA_DIR,
    LR_CONFIGS,
    MAX_GRAD_NORM,
    MIRABEST_DATA_DIR,
    MODEL_ENTRIES,
    NUM_FOLDS,
    RGZ_DATA_DIR,
    SUBSAMPLE_FRACTION,
    USE_CLASS_WEIGHTED_LOSS,
    WARMUP_RATIO_FT,
    WARMUP_RATIO_LP,
    DatasetMode,
    EvalMode,
    _cfg_label,
    _cfg_tag,
    _fmt_lr,
)
from downstream_eval.llrd.models import (
    _extract_image_size_from_processor,
    _pretrained_candidate_dirs,
    _resolve_pretrained_config_source,
    _weighted_f1_from_cm,
    infer_n_registers,
    load_hf_dinov2_config,
    load_hf_dinov2_config_dict,
)
from downstream_eval.llrd.runtime import PROJECT_ROOT, _is_main_process, configure_runtime_environment, guarded_print
from downstream_eval.llrd.trainer_runtime import (
    apply_linear_probe_runtime,
    build_batch_plan,
    build_classifier_model,
    build_trainer,
    build_training_args,
    log_batch_plan,
    should_find_unused_parameters,
)

if TYPE_CHECKING:
    from downstream_eval.llrd.config import LLRDConfig

OUTPUT_DIR = settings.SCRIPT_DIR
SELECTED_DATASET_KEYS: set[str] | None = None
DETERMINISTIC_EVAL = True
PROGRESS_FILE = os.getenv("STRADA_PROGRESS_FILE", "")
_PROGRESS = {
    "total": 0,
    "completed": 0,
    "skipped_cached": 0,
    "status": "pending",
    "started_at": None,
}

PROCESSOR_FALLBACKS = {
    "vit-mae-base": "facebook/vit-mae-base",
    "vit-mae-scratch-hf": "facebook/vit-mae-base",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}


@dataclass(frozen=True)
class EvaluationResultRecord:
    alias: str
    macro_f1: float
    macro_f1_std: float
    weighted_f1: float
    weighted_f1_std: float
    per_class_f1_mean: np.ndarray
    per_class_f1_std: np.ndarray
    cm_total: np.ndarray
    macro_f1_curve_mean: np.ndarray
    macro_f1_curve_std: np.ndarray
    weighted_f1_curve_mean: np.ndarray
    weighted_f1_curve_std: np.ndarray

    def as_legacy_tuple(self):
        return (
            self.alias,
            self.macro_f1,
            self.macro_f1_std,
            self.weighted_f1,
            self.weighted_f1_std,
            self.per_class_f1_mean,
            self.per_class_f1_std,
            self.cm_total,
            self.macro_f1_curve_mean,
            self.macro_f1_curve_std,
            self.weighted_f1_curve_mean,
            self.weighted_f1_curve_std,
        )


def _eval_modes_to_run(eval_mode: EvalMode) -> list[EvalMode]:
    if eval_mode == EvalMode.ALL:
        return [EvalMode.LINEAR_PROBE, EvalMode.FINETUNE]
    return [eval_mode]


def _write_progress(**updates) -> None:
    if not PROGRESS_FILE:
        return
    _PROGRESS.update(updates)
    _PROGRESS["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        tmp = f"{PROGRESS_FILE}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(_PROGRESS, f, indent=2)
            f.write("\n")
        os.replace(tmp, PROGRESS_FILE)
    except Exception:
        pass


def _reset_global_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transforms(mean, std):
    """Build the shared downstream train and evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
        transforms.Lambda(basic_flips),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def _stable_eval_seed(*parts: object) -> int:
    payload = "|".join([str(GLOBAL_SEED), *[str(part) for part in parts]])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def _dataset_runs() -> list[tuple[str, object, str]]:
    runs: list[tuple[str, object, str]] = []
    selected = SELECTED_DATASET_KEYS
    if (selected is None and DATASET_MODE in (DatasetMode.RGZ, DatasetMode.ALL)) or (selected is not None and "rgz" in selected):
        runs.append(("RGZ", RadioGalaxyDataset, RGZ_DATA_DIR))
    if (selected is None and DATASET_MODE in (DatasetMode.MIRABEST, DatasetMode.ALL)) or (selected is not None and "mirabest" in selected):
        runs.append(("MiraBest", MiraBestDatasetV2, MIRABEST_DATA_DIR))
    if (selected is None and DATASET_MODE in (DatasetMode.LOTSS_HORTON, DatasetMode.ALL)) or (selected is not None and "lotss_horton" in selected):
        runs.append(("LoTSS-Horton", LoTSSDR2HortonDataset, LOTSS_HORTON_DATA_DIR))
    return runs


def _encoder_family_for_tag(model_tag: str) -> str:
    if model_tag in ("vit-mae-base", "vit-mae-scratch-hf"):
        return "mae"
    if model_tag in ("dinov2-small", "dinov2-base"):
        return "dinov2"
    return "generic"


def _resolve_checkpoint_dir(raw: str) -> str | None:
    if not isinstance(raw, str) or raw.strip() == "":
        return None

    def _has_weights(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(
            os.path.join(path, "pytorch_model.bin")
        )

    def _latest_checkpoint(path: str) -> str | None:
        try:
            names = os.listdir(path)
        except OSError:
            return None
        candidates = []
        for name in names:
            if not name.startswith("checkpoint-"):
                continue
            step = name.removeprefix("checkpoint-")
            if not step.isdigit():
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full) and _has_weights(full):
                candidates.append((int(step), full))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return os.path.abspath(candidates[-1][1])

    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(os.path.join(PROJECT_ROOT, raw))
        candidates.append(os.path.join(os.path.dirname(__file__), raw))
    for cand in candidates:
        try:
            if os.path.isdir(cand):
                if _has_weights(cand):
                    return os.path.abspath(cand)
                latest = _latest_checkpoint(cand)
                if latest is not None:
                    return latest
                if os.path.basename(cand.rstrip(os.sep)) == "checkpoints":
                    return os.path.abspath(cand)
                return os.path.abspath(cand)
        except Exception:
            continue
    return None


def _looks_like_local_checkpoint(raw: str) -> bool:
    return any(tok in raw for tok in ("checkpoint-", "/results/", "runs/", "="))


def _set_module_value(name: str, value) -> None:
    globals()[name] = value
    setattr(settings, name, value)


def configure_from_config(config: "LLRDConfig") -> None:
    import downstream_eval.llrd.reporting as reporting

    run = config.run
    work_dir = os.path.abspath(run.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    _set_module_value("WORK_DIR", work_dir)

    output_dir = os.path.abspath(run.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _set_module_value("OUTPUT_DIR", output_dir)

    dataset_by_key = {dataset.key: dataset for dataset in config.datasets}
    selected_dataset_keys = {dataset.key for dataset in config.selected_datasets()}
    _set_module_value("EVAL_MODE", EvalMode(run.eval_mode))
    _set_module_value("DATASET_MODE", DatasetMode(run.dataset_mode))
    _set_module_value("SELECTED_DATASET_KEYS", selected_dataset_keys)
    _set_module_value("USE_CLASS_WEIGHTED_LOSS", bool(run.use_class_weighted_loss))
    _set_module_value("GLOBAL_SEED", int(run.global_seed))
    _set_module_value("DETERMINISTIC_EVAL", bool(run.deterministic_eval))
    _set_module_value("BATCH_SIZE", int(run.batch_size))
    _set_module_value("SUBSAMPLE_FRACTION", run.subsample_fraction)
    _set_module_value("NUM_FOLDS", int(run.num_folds))
    _set_module_value("FINETUNE_EPOCHS", int(run.finetune_epochs))
    _set_module_value("FINETUNE_WEIGHT_DECAY", float(run.finetune_weight_decay))
    _set_module_value("WARMUP_RATIO_FT", float(run.warmup_ratio_ft))
    _set_module_value("WARMUP_RATIO_LP", float(run.warmup_ratio_lp))
    _set_module_value("MAX_GRAD_NORM", float(run.max_grad_norm))
    _set_module_value("DEBUG_LAYER_DECAY", int(run.debug_layer_decay))
    _set_module_value("LR_CONFIGS", [dict(item) for item in run.lr_configs])
    _set_module_value("MODEL_ENTRIES", [(m.alias, m.encoder, m.model_id) for m in config.selected_models()])
    _set_module_value("CACHE_JSON_PATH", config.cache_path())

    if "rgz" in dataset_by_key:
        _set_module_value("RGZ_DATA_DIR", dataset_by_key["rgz"].root)
        _set_module_value("DATA_INDEX_CACHE_DIR", dataset_by_key["rgz"].index_cache_dir)
        _set_module_value("DATA_INDEX_WORKERS", int(dataset_by_key["rgz"].index_workers))
    if "mirabest" in dataset_by_key:
        _set_module_value("MIRABEST_DATA_DIR", dataset_by_key["mirabest"].root)
    if "lotss_horton" in dataset_by_key:
        _set_module_value("LOTSS_HORTON_DATA_DIR", dataset_by_key["lotss_horton"].root)

    analysis = config.analysis
    _set_module_value(
        "ANALYSIS_MODELS",
        [
            {
                "lookup_alias": model.alias,
            }
            for model in config.selected_models()
            if model.analysis
        ],
    )
    _set_module_value(
        "ANALYSIS_DATASET_ORDER",
        [
            (dataset.label, dataset.analysis_label)
            for dataset in sorted(config.selected_datasets(), key=lambda item: int(item.analysis_order))
        ],
    )
    _set_module_value("ANALYSIS_COMPACT_TABLE_LAYOUT", bool(analysis.compact_table_layout))
    _set_module_value("ANALYSIS_SAVE_COMBINED_LP_FT", bool(analysis.save_combined_lp_ft))
    _set_module_value("ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS", bool(analysis.save_individual_mode_plots))
    _set_module_value("ANALYSIS_CM_CMAP_NAME", analysis.cm_cmap_name)
    _set_module_value("ANALYSIS_CM_NORMALIZATION", analysis.cm_normalization)
    _set_module_value("ANALYSIS_HEADER_FONT_SIZE", int(analysis.header_font_size))
    _set_module_value(
        "ANALYSIS_CLASS_ORDER",
        {dataset.label: list(dataset.class_order) for dataset in config.selected_datasets()},
    )

    for name in (
        "SCRIPT_DIR",
        "LR_CONFIGS",
        "ANALYSIS_MODELS",
        "ANALYSIS_DATASET_ORDER",
        "ANALYSIS_COMPACT_TABLE_LAYOUT",
        "ANALYSIS_SAVE_COMBINED_LP_FT",
        "ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS",
        "ANALYSIS_CM_CMAP_NAME",
        "ANALYSIS_CM_NORMALIZATION",
        "ANALYSIS_HEADER_FONT_SIZE",
        "ANALYSIS_CLASS_ORDER",
    ):
        setattr(reporting, name, globals().get(name, getattr(settings, name, None)))
    reporting.SCRIPT_DIR = output_dir


def run_llrd_evaluation(config: "LLRDConfig") -> None:
    configure_runtime_environment(
        config.run.hf_cache_path,
        deterministic_eval=bool(config.run.deterministic_eval),
    )
    settings.apply_matplotlib_style()
    configure_from_config(config)

    from downstream_eval.llrd.cache import load_cache

    cache = load_cache()
    eval_modes_to_run = _eval_modes_to_run(EVAL_MODE)
    total = len(eval_modes_to_run) * len(_dataset_runs()) * len(MODEL_ENTRIES) * max(1, len(LR_CONFIGS))
    _write_progress(
        total=total,
        completed=0,
        skipped_cached=0,
        status="running",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    try:
        for eval_mode_sel in eval_modes_to_run:
            _run_eval_mode(cache, eval_mode_sel)

        _write_progress(status="finished")
    except Exception as exc:
        _write_progress(status="failed", error=str(exc))
        raise

def evaluate_on_dataset(dataset_label, DatasetCls, data_dir, cache, eval_mode_sel: EvalMode):
    # Reproducibility (per dataset so splits are consistent)
    _reset_global_rng(GLOBAL_SEED)

    os.makedirs(DATA_INDEX_CACHE_DIR, exist_ok=True)
    data_dir_abs = os.path.abspath(os.path.expanduser(data_dir))
    data_dir_tag = hashlib.md5(data_dir_abs.encode("utf-8")).hexdigest()[:10]
    index_cache_path = os.path.join(DATA_INDEX_CACHE_DIR, f"{dataset_label.lower()}_{data_dir_tag}.pkl")

    def _dataset_accepts_kwarg(name: str) -> bool:
        try:
            return name in inspect.signature(DatasetCls.__init__).parameters
        except Exception:
            return False

    def _make_dataset(transform, image_size, shared_index=None):
        kwargs = {"root": data_dir, "transform": transform, "image_size": image_size}
        if shared_index is not None and _dataset_accepts_kwarg("samples"):
            kwargs.update(shared_index)
        elif _dataset_accepts_kwarg("index_cache_path"):
            kwargs["index_cache_path"] = index_cache_path
            if _dataset_accepts_kwarg("index_num_workers"):
                kwargs["index_num_workers"] = DATA_INDEX_WORKERS
        return DatasetCls(**kwargs)

    # Set up dataset-level cache
    dataset_cache = cache.setdefault(dataset_label, {})
    models_cache = dataset_cache.setdefault("models", {})

    guarded_print(f"\n=== Dataset: {dataset_label} ===")

    # Cached dataset metadata (may be enough to avoid dataset prep)
    class_names = dataset_cache.get("class_names", None)
    num_classes = dataset_cache.get("num_classes", None)
    class_weights = None
    base_ds = None
    base_labels = None
    subset_indices = None
    split_indices = None
    dataset_prepared = False

    def _ensure_dataset_prepared():
        nonlocal class_names, num_classes, class_weights, base_ds, base_labels
        nonlocal subset_indices, split_indices, dataset_prepared
        if dataset_prepared:
            return

        # Build base dataset for stratified split (no transform, fixed size)
        base_ds = _make_dataset(transform=None, image_size=224, shared_index=None)
        # Extract class labels for stratified splitting.
        # Prefer a 'samples' attribute (as in RadioGalaxyDataset),
        # otherwise fall back to 'targets' (as in MiraBestDataset),
        # and finally a slow DataLoader-based fallback.
        if hasattr(base_ds, "samples"):
            base_labels = np.array([lbl for _, lbl in base_ds.samples], dtype=np.int64)
        elif hasattr(base_ds, "targets"):
            base_labels = np.array(base_ds.targets, dtype=np.int64)
        else:
            # Generic fallback: iterate once over the dataset to collect labels
            tmp_loader = DataLoader(base_ds, batch_size=64, shuffle=False)
            labels_list = []
            for batch in tmp_loader:
                labels_list.extend(batch["labels"].numpy().tolist())
            base_labels = np.array(labels_list, dtype=np.int64)

        # Recover class names/num_classes from cache or base dataset if available
        if class_names is None and hasattr(base_ds, "classes"):
            class_names = list(base_ds.classes)
            num_classes = len(class_names)

        # Compute per-class weights (balanced by frequency) for loss, if enabled
        classes = np.unique(base_labels)
        if USE_CLASS_WEIGHTED_LOSS:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=base_labels,
            ).astype(np.float32)
            guarded_print(
                f"  → Class weights (balanced CE): "
                f"{dict(zip(classes.tolist(), class_weights.tolist()))}"
            )

        # Optionally subsample a fraction of the dataset, equally per class
        if SUBSAMPLE_FRACTION is None or SUBSAMPLE_FRACTION >= 1.0:
            subset_indices = np.arange(len(base_ds))
        else:
            cls_counts = [np.sum(base_labels == c) for c in classes]
            max_count = max(cls_counts)
            target_per_class = max(1, int(round(SUBSAMPLE_FRACTION * max_count)))

            rng = np.random.RandomState(GLOBAL_SEED)
            subset_idx_list = []
            for c in classes:
                cls_idx = np.where(base_labels == c)[0]
                if len(cls_idx) <= target_per_class:
                    subset_idx_list.append(cls_idx)
                else:
                    subset_idx_list.append(
                        rng.choice(cls_idx, size=target_per_class, replace=False)
                    )
            subset_indices = np.concatenate(subset_idx_list)
            subset_indices = np.sort(subset_indices)

        guarded_print(
            f"Using {len(subset_indices)} samples out of {len(base_ds)} "
            f"({len(subset_indices) / len(base_ds):.1%}) for embedding/probe tests."
        )

        # Prepare stratified k-fold splits on the (possibly) subsampled set
        base_labels_sub = base_labels[subset_indices]
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
        split_indices = list(skf.split(subset_indices, base_labels_sub))
        dataset_prepared = True

    results = []
    sweep_results = {}

    # Helper to pack results for JSON serialization (fine-tuning only)
    def _pack_record(model_id,
                     lr_backbone,
                     head_mult,
                     head_lr,
                     layer_decay,
                     macro_f1,
                     macro_f1_std,
                     weighted_f1,
                     weighted_f1_std,
                     per_class_f1_mean,
                     per_class_f1_std,
                     cm_total,
                     macro_f1_curve_mean,
                     macro_f1_curve_std,
                     weighted_f1_curve_mean,
                     weighted_f1_curve_std):
        return {
            "model_id": model_id,
            "lr_backbone": float(lr_backbone),
            "head_lr_mult": float(head_mult),
            "head_lr": float(head_lr),
            "layer_decay": float(layer_decay),
            "macro_f1": float(macro_f1),
            "macro_f1_std": float(macro_f1_std),
            "weighted_f1": float(weighted_f1),
            "weighted_f1_std": float(weighted_f1_std),
            "per_class_f1_mean": per_class_f1_mean.tolist(),
            "per_class_f1_std": per_class_f1_std.tolist(),
            "cm_total": cm_total.tolist(),
            "macro_f1_curve_mean": macro_f1_curve_mean.tolist(),
            "macro_f1_curve_std": macro_f1_curve_std.tolist(),
            "weighted_f1_curve_mean": weighted_f1_curve_mean.tolist(),
            "weighted_f1_curve_std": weighted_f1_curve_std.tolist(),
        }

    eval_mode = eval_mode_sel.value
    mode_key = eval_mode
    mode_label = "FT" if eval_mode == "finetune" else "LP"

    for alias, model_tag, model_id in MODEL_ENTRIES:
        guarded_print(f"\n{'=' * 60}")
        guarded_print(f"Processing ({dataset_label}): {alias}")
        guarded_print(f"  Tag:   {model_tag}")
        guarded_print(f"  Model: {model_id}")
        _write_progress(current_dataset=dataset_label, current_alias=alias, current_eval_mode=eval_mode)

        # Per-alias cache (to hold separate entries for finetune / linear_probe)
        model_cache = models_cache.setdefault(alias, {})

        # === STEP 6: Run the selected evaluation mode (full FT or linear probe) ===
        guarded_print(f"  → Mode: {mode_label} ({eval_mode})")
        config_results = {}
        all_cached = True

        for cfg in LR_CONFIGS:
            cfg_tag = _cfg_tag(cfg)
            cfg_label = _cfg_label(cfg)
            cache_key = f"{mode_key}|{cfg_tag}"
            alias_with_mode = f"{alias} [{mode_label}] cfg={cfg_label}"

            cached_record = model_cache.get(cache_key, None)
            if cached_record is not None and "macro_f1_curve_mean" in cached_record:
                guarded_print(f"    → Using cached results ({cfg_label}); skipping evaluation")
                _write_progress(skipped_cached=int(_PROGRESS.get("skipped_cached", 0)) + 1)

                macro_f1 = float(cached_record["macro_f1"])
                macro_f1_std = float(cached_record["macro_f1_std"])
                weighted_f1 = float(cached_record.get("weighted_f1", float("nan")))
                weighted_f1_std = float(cached_record.get("weighted_f1_std", 0.0))
                per_class_f1_mean = np.array(cached_record["per_class_f1_mean"], dtype=np.float32)
                per_class_f1_std = np.array(cached_record["per_class_f1_std"], dtype=np.float32)
                cm_total = np.array(cached_record["cm_total"], dtype=np.int64)
                macro_f1_curve_mean = np.array(cached_record["macro_f1_curve_mean"], dtype=np.float32)
                macro_f1_curve_std = np.array(cached_record["macro_f1_curve_std"], dtype=np.float32)
                weighted_f1_curve_mean = np.array(
                    cached_record.get("weighted_f1_curve_mean", []), dtype=np.float32
                )
                weighted_f1_curve_std = np.array(
                    cached_record.get("weighted_f1_curve_std", []), dtype=np.float32
                )
                if not np.isfinite(weighted_f1):
                    weighted_f1 = _weighted_f1_from_cm(cm_total)
                    weighted_f1_std = 0.0
                if weighted_f1_curve_mean.size == 0:
                    weighted_f1_curve_mean = np.array([weighted_f1], dtype=np.float32)
                    weighted_f1_curve_std = np.zeros_like(weighted_f1_curve_mean)

                if class_names is None or num_classes is None:
                    num_classes = int(per_class_f1_mean.shape[0])
                    class_names = [f"class_{i}" for i in range(num_classes)]

                results.append(
                    (
                        alias_with_mode,
                        macro_f1,
                        macro_f1_std,
                        weighted_f1,
                        weighted_f1_std,
                        per_class_f1_mean,
                        per_class_f1_std,
                        cm_total,
                        macro_f1_curve_mean,
                        macro_f1_curve_std,
                        weighted_f1_curve_mean,
                        weighted_f1_curve_std,
                    )
                )

                config_results[cfg_tag] = (
                    macro_f1,
                    macro_f1_std,
                    weighted_f1,
                    weighted_f1_std,
                )
                continue

            all_cached = False

        if all_cached:
            sweep_results.setdefault(alias, {}).update(config_results)
            guarded_print(f"✓ Completed (cached): {alias} on {dataset_label}")
            continue

        raw_model_id = model_id
        resolved_ckpt = _resolve_checkpoint_dir(raw_model_id)
        if resolved_ckpt is not None:
            model_id = resolved_ckpt
            if raw_model_id != model_id:
                guarded_print(f"  → Resolved checkpoint dir: {model_id}")
        else:
            model_id = raw_model_id
            if _looks_like_local_checkpoint(raw_model_id):
                guarded_print(f"  ✗ Checkpoint dir not found on disk: {raw_model_id}")
                guarded_print("    → Skipping this model and continuing.\n")
                continue
        model_is_local = resolved_ckpt is not None

        _ensure_dataset_prepared()

        encoder_family = _encoder_family_for_tag(model_tag)
        regs_count = infer_n_registers(model_id, encoder_family)
        if encoder_family == "mae":
            guarded_print(f"  → Registers detected: n_registers={regs_count}")

        # === STEP 2: Load processor and verify normalization stats ===
        mean, std = None, None
        proc = None
        proc_ids = (
            [model_id] + _pretrained_candidate_dirs(model_id) + [PROCESSOR_FALLBACKS.get(model_tag)]
            if model_is_local
            else [model_id, PROCESSOR_FALLBACKS.get(model_tag)]
        )
        seen = set()
        proc_source = None
        for proc_id in proc_ids:
            if not proc_id or proc_id in seen:
                continue
            seen.add(proc_id)
            try:
                try:
                    proc = AutoImageProcessor.from_pretrained(proc_id, use_fast=True)
                except Exception:
                    proc = AutoImageProcessor.from_pretrained(proc_id, use_fast=False)
                mean, std = proc.image_mean, proc.image_std
                proc_img_size = _extract_image_size_from_processor(proc)
                proc_source = proc_id
                if proc_id == model_id:
                    guarded_print(f"  ✓ Loaded processor: mean={mean}, std={std}")
                else:
                    guarded_print(f"  ✓ Loaded processor fallback ({proc_id}): mean={mean}, std={std}")
                if proc_img_size is not None:
                    guarded_print(f"  → Processor image size hint: {proc_img_size}x{proc_img_size}")
                break
            except Exception as e:
                if proc_id == model_id:
                    guarded_print(f"  ✗ Failed to load processor directly: {e}")
        if mean is None or std is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            guarded_print("  → Using default ImageNet mean/std (fallback, processor missing)")

        # === STEP 3: Create transforms with verified normalization ===
        train_transform, eval_transform = build_image_transforms(mean, std)

        # === STEP 4: Choose optimal image size for this model ===
        img_size = None
        cfg = None
        raw_dino_cfg = {}
        cfg_source = None
        cfg_candidates = [_resolve_pretrained_config_source(model_id)] if model_is_local else [model_id]
        for cfg_id in cfg_candidates:
            try:
                if encoder_family == "dinov2":
                    raw_dino_cfg = load_hf_dinov2_config_dict(cfg_id)
                    cfg = load_hf_dinov2_config(cfg_id)
                    cfg_source = cfg_id
                else:
                    cfg = AutoConfig.from_pretrained(cfg_id)
                    cfg_source = cfg_id
                break
            except Exception:
                continue
        if cfg is None and model_is_local:
            try:
                if encoder_family == "dinov2":
                    raw_dino_cfg = load_hf_dinov2_config_dict(model_id)
                    cfg = load_hf_dinov2_config(model_id)
                    cfg_source = model_id
                else:
                    cfg = AutoConfig.from_pretrained(model_id)
                    cfg_source = model_id
            except Exception:
                cfg = None
                cfg_source = None
        proc_img_size = _extract_image_size_from_processor(proc)
        proc_is_exact = (proc_source == model_id)
        if encoder_family == "dinov2":
            raw_cfg_img_size = None
            if isinstance(raw_dino_cfg, dict):
                try:
                    raw_cfg_img_size = raw_dino_cfg.get("image_size")
                except Exception:
                    raw_cfg_img_size = None
            guarded_print(
                f"  [DINO-ConfigDebug] proc_source={proc_source!r} exact={proc_is_exact} "
                f"proc_img_size={proc_img_size!r}"
            )
            guarded_print(
                f"  [DINO-ConfigDebug] cfg_source={cfg_source!r} raw_model_type="
                f"{(raw_dino_cfg.get('model_type') if isinstance(raw_dino_cfg, dict) else None)!r} "
                f"raw_image_size={raw_cfg_img_size!r}"
            )
            if cfg is not None:
                guarded_print(
                    f"  [DINO-ConfigDebug] cfg_class={cfg.__class__.__name__} "
                    f"cfg_model_type={getattr(cfg, 'model_type', None)!r} "
                    f"cfg_image_size={getattr(cfg, 'image_size', None)!r}"
                )
            if isinstance(raw_dino_cfg, dict) and raw_dino_cfg.get("image_size") is not None:
                try:
                    img_size = int(raw_dino_cfg["image_size"])
                except Exception:
                    img_size = None
            if img_size is None and cfg is not None:
                img_size = getattr(cfg, "image_size", None)
            if img_size is not None:
                if isinstance(raw_dino_cfg, dict) and raw_dino_cfg.get("image_size") is not None:
                    guarded_print(f"  → Using image size from raw config.json: {img_size}x{img_size}")
                else:
                    guarded_print(f"  → Using image size from config: {img_size}x{img_size}")
            elif proc_img_size is not None and proc_is_exact:
                img_size = proc_img_size
                guarded_print(f"  → Using image size from processor: {img_size}x{img_size}")
            else:
                img_size = 518
                guarded_print(f"  → Using image size: {img_size}x{img_size} (DINO native)")
        else:
            if proc_img_size is not None and proc_is_exact:
                img_size = proc_img_size
                guarded_print(f"  → Using image size from processor: {img_size}x{img_size}")
            else:
                if cfg is not None:
                    img_size = getattr(cfg, "image_size", None)
                if img_size is not None:
                    guarded_print(f"  → Using image size from config: {img_size}x{img_size}")
                else:
                    img_size = 224
                    guarded_print(f"  → Using image size: {img_size}x{img_size} (ViT standard)")

        # === STEP 5: Create datasets with model-specific preprocessing ===
        shared_index = None
        if hasattr(base_ds, "samples") and hasattr(base_ds, "classes") and hasattr(base_ds, "class_to_idx"):
            shared_index = {
                "samples": getattr(base_ds, "samples"),
                "classes": list(getattr(base_ds, "classes")),
                "class_to_idx": dict(getattr(base_ds, "class_to_idx")),
            }

        train_ds_full = _make_dataset(transform=train_transform, image_size=img_size, shared_index=shared_index)
        eval_ds_full = _make_dataset(transform=eval_transform, image_size=img_size, shared_index=shared_index)
        ds_cur_sub = Subset(eval_ds_full, subset_indices)

        if class_names is None:
            class_names = eval_ds_full.classes
            num_classes = len(class_names)
            guarded_print(f"  → Classes ({dataset_label}): {class_names}")

        for cfg in LR_CONFIGS:
            cfg_tag = _cfg_tag(cfg)
            cfg_label = _cfg_label(cfg)
            cache_key = f"{mode_key}|{cfg_tag}"
            alias_with_mode = f"{alias} [{mode_label}] cfg={cfg_label}"

            cached_record = model_cache.get(cache_key, None)
            if cached_record is not None and "macro_f1_curve_mean" in cached_record:
                continue

            head_lr = cfg["bb_lr"] * cfg["head_mult"]
            lr_for_trainer = head_lr if eval_mode == "linear_probe" else cfg["bb_lr"]
            guarded_print(
                f"    → Config: base_lr={_fmt_lr(cfg['bb_lr'])}, "
                f"head_lr={_fmt_lr(head_lr)} (x{int(cfg['head_mult'])}), "
                f"decay={cfg['layer_decay']:.2f}"
            )
            guarded_print(f"    → Trainer LR ({mode_label}): {_fmt_lr(lr_for_trainer)}")
            guarded_print(f"    → Running k-fold {eval_mode} ({NUM_FOLDS} folds)")
            macro_f1_folds = []
            weighted_f1_folds = []
            per_class_f1_folds = []
            cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
            macro_f1_curve_folds = []
            weighted_f1_curve_folds = []

            try:
                for fold_id, (train_idx, test_idx) in enumerate(split_indices):
                    guarded_print(f"    → Fold {fold_id + 1}/{NUM_FOLDS}")
                    fold_seed = GLOBAL_SEED
                    if DETERMINISTIC_EVAL:
                        fold_seed = _stable_eval_seed(dataset_label, cfg_tag, fold_id)
                    guarded_print(f"      → Fold seed: {fold_seed}")
                    _reset_global_rng(fold_seed)

                    train_indices = subset_indices[train_idx]
                    test_indices = subset_indices[test_idx]

                    train_ds = Subset(train_ds_full, train_indices)
                    test_ds = Subset(eval_ds_full, test_indices)

                    model = build_classifier_model(
                        encoder_family=encoder_family,
                        checkpoint_path=model_id,
                        num_classes=num_classes,
                        class_weights=class_weights,
                        n_registers=regs_count,
                    )
                    if eval_mode_sel == EvalMode.LINEAR_PROBE:
                        model = apply_linear_probe_runtime(model, num_classes)

                    output_dir = os.path.join(
                        WORK_DIR,
                        "trainer_runs",
                        dataset_label,
                        alias.replace("/", "_"),
                        f"{mode_label}_{cfg_tag}",
                        f"fold{fold_id + 1}",
                    )

                    batch_plan = build_batch_plan(BATCH_SIZE, encoder_family)
                    log_batch_plan(batch_plan, BATCH_SIZE)
                    ddp_unused = should_find_unused_parameters(encoder_family, regs_count)
                    warmup_ratio = WARMUP_RATIO_LP if eval_mode == "linear_probe" else WARMUP_RATIO_FT
                    training_args = build_training_args(
                        output_dir=output_dir,
                        batch_plan=batch_plan,
                        num_epochs=FINETUNE_EPOCHS,
                        learning_rate=lr_for_trainer,
                        weight_decay=FINETUNE_WEIGHT_DECAY,
                        warmup_ratio=warmup_ratio,
                        seed=fold_seed,
                        max_grad_norm=MAX_GRAD_NORM,
                        ddp_find_unused_parameters=ddp_unused,
                        deterministic_eval=DETERMINISTIC_EVAL,
                    )
                    trainer = build_trainer(
                        eval_mode=eval_mode_sel,
                        model=model,
                        training_args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=test_ds,
                        lr_config=cfg,
                        debug_layer_decay=DEBUG_LAYER_DECAY,
                    )

                    trainer.train()

                    # Extract macro-F1 curve from Trainer log history
                    log_history = trainer.state.log_history
                    fold_macro_f1_curve = [
                        log["eval_macro_f1"]
                        for log in log_history
                        if "eval_macro_f1" in log
                    ]
                    fold_weighted_f1_curve = [
                        log["eval_weighted_f1"]
                        for log in log_history
                        if "eval_weighted_f1" in log
                    ]
                    if not fold_macro_f1_curve or not fold_weighted_f1_curve:
                        # Fallback: run a single eval
                        eval_metrics = trainer.evaluate()
                        if not fold_macro_f1_curve:
                            fold_macro_f1_curve = [float(eval_metrics.get("eval_macro_f1", 0.0))]
                        if not fold_weighted_f1_curve:
                            fold_weighted_f1_curve = [float(eval_metrics.get("eval_weighted_f1", 0.0))]

                    # Use Trainer.predict to get final predictions for confusion matrix
                    preds_output = trainer.predict(test_ds)
                    logits = preds_output.predictions
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    preds = np.argmax(logits, axis=-1)
                    true_labels = preds_output.label_ids

                    fold_macro_f1_final = float(
                        f1_score(true_labels, preds, average="macro", zero_division=0)
                    )
                    fold_weighted_f1_final = float(
                        f1_score(true_labels, preds, average="weighted", zero_division=0)
                    )

                    macro_f1_folds.append(fold_macro_f1_final)
                    macro_f1_curve_folds.append(fold_macro_f1_curve)
                    weighted_f1_folds.append(fold_weighted_f1_final)
                    weighted_f1_curve_folds.append(fold_weighted_f1_curve)

                    cm = confusion_matrix(true_labels, preds, labels=list(range(num_classes)))
                    cm_total += cm

                    per_class_f1 = []
                    for cls_idx in range(num_classes):
                        tp = cm[cls_idx, cls_idx]
                        fp = cm[:, cls_idx].sum() - tp
                        fn = cm[cls_idx, :].sum() - tp
                        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
                        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
                        f1_i = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
                        per_class_f1.append(f1_i)
                    per_class_f1_folds.append(per_class_f1)

                    guarded_print(
                        f"      Fold {fold_id + 1} final F1 ({mode_label}): "
                        f"macro={fold_macro_f1_final:.3f}, weighted={fold_weighted_f1_final:.3f}"
                    )
                    # Remove Trainer output directory to avoid cluttering the filesystem
                    if _is_main_process():
                        shutil.rmtree(output_dir, ignore_errors=True)

            except Exception as e:
                guarded_print(f"  ✗ Error while processing {alias_with_mode}: {e}")
                guarded_print("    → Skipping this mode and continuing.\n")
                continue

            # Aggregate across folds for this mode
            macro_f1 = float(np.mean(macro_f1_folds))
            macro_f1_std = float(np.std(macro_f1_folds))
            weighted_f1 = float(np.mean(weighted_f1_folds)) if weighted_f1_folds else 0.0
            weighted_f1_std = float(np.std(weighted_f1_folds)) if weighted_f1_folds else 0.0
            per_class_f1_folds = np.array(per_class_f1_folds)
            per_class_f1_mean = per_class_f1_folds.mean(axis=0)
            per_class_f1_std = per_class_f1_folds.std(axis=0)

            # Align F1 curves by padding shorter ones (if any) to the max length
            max_len = max(len(curve) for curve in macro_f1_curve_folds)
            curves_aligned = []
            for curve in macro_f1_curve_folds:
                if len(curve) < max_len:
                    pad_val = curve[-1]
                    curve_ext = list(curve) + [pad_val] * (max_len - len(curve))
                else:
                    curve_ext = curve
                curves_aligned.append(curve_ext)
            macro_f1_curve_folds_arr = np.array(curves_aligned, dtype=np.float32)
            macro_f1_curve_mean = macro_f1_curve_folds_arr.mean(axis=0)
            macro_f1_curve_std = macro_f1_curve_folds_arr.std(axis=0)
            weighted_f1_curve_mean = np.array([], dtype=np.float32)
            weighted_f1_curve_std = np.array([], dtype=np.float32)
            if weighted_f1_curve_folds:
                max_len_w = max(len(curve) for curve in weighted_f1_curve_folds)
                curves_aligned_w = []
                for curve in weighted_f1_curve_folds:
                    if len(curve) < max_len_w:
                        pad_val = curve[-1]
                        curve_ext = list(curve) + [pad_val] * (max_len_w - len(curve))
                    else:
                        curve_ext = curve
                    curves_aligned_w.append(curve_ext)
                weighted_f1_curve_folds_arr = np.array(curves_aligned_w, dtype=np.float32)
                weighted_f1_curve_mean = weighted_f1_curve_folds_arr.mean(axis=0)
                weighted_f1_curve_std = weighted_f1_curve_folds_arr.std(axis=0)

            results.append(
                (
                    alias_with_mode,
                    macro_f1,
                    macro_f1_std,
                    weighted_f1,
                    weighted_f1_std,
                    per_class_f1_mean,
                    per_class_f1_std,
                    cm_total,
                    macro_f1_curve_mean,
                    macro_f1_curve_std,
                    weighted_f1_curve_mean,
                    weighted_f1_curve_std,
                )
            )

            model_cache[cache_key] = _pack_record(
                model_id,
                cfg["bb_lr"],
                cfg["head_mult"],
                head_lr,
                cfg["layer_decay"],
                macro_f1,
                macro_f1_std,
                weighted_f1,
                weighted_f1_std,
                per_class_f1_mean,
                per_class_f1_std,
                cm_total,
                macro_f1_curve_mean,
                macro_f1_curve_std,
                weighted_f1_curve_mean,
                weighted_f1_curve_std,
            )

            config_results[cfg_tag] = (
                macro_f1,
                macro_f1_std,
                weighted_f1,
                weighted_f1_std,
            )

            guarded_print(f"✓ Completed: {alias_with_mode} on {dataset_label}")
            _write_progress(completed=int(_PROGRESS.get("completed", 0)) + 1)

        if config_results:
            sweep_results.setdefault(alias, {}).update(config_results)

    # Persist dataset-level metadata back into the cache
    if class_names is not None:
        dataset_cache["class_names"] = list(class_names)
        dataset_cache["num_classes"] = int(len(class_names))

    return {
        "label": dataset_label,
        "results": results,
        "sweep": sweep_results,
        "class_names": class_names,
        "num_classes": num_classes,
    }

def _run_eval_mode(cache: dict, eval_mode_sel: EvalMode) -> None:
    sections = [
        evaluate_on_dataset(label, dataset_cls, root, cache, eval_mode_sel)
        for label, dataset_cls, root in _dataset_runs()
    ]
    if not sections:
        raise RuntimeError("No datasets selected; check DATASET_MODE.")

    n_models = len(MODEL_ENTRIES)
    if n_models == 0:
        guarded_print("No models specified in MODEL_ENTRIES; nothing to do.")
        return

    if not _is_main_process():
        return

    save_cache(cache)

    # Avoid IDE display proxy errors; files are already saved to disk.
    plt.close("all")
