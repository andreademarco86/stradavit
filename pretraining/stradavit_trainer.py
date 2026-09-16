import sys
sys.path.append('../')
# --- make local packages importable regardless of CWD ---
import os, sys, random, datetime
from dataclasses import dataclass
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----- rank-aware printing helpers -----
def _is_main_process():
    return os.getenv("RANK", "0") in ("0", "") or os.getenv("LOCAL_RANK", "0") in ("0", "")

def guarded_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)

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

guarded_print(f"[PathDebug] cwd={os.getcwd()}")
guarded_print(f"[PathDebug] added to sys.path: {PROJECT_ROOT}")
# --------------------------------------------------------

os.environ.update({
    "NCCL_TIMEOUT": "3600",
    "NCCL_LAUNCH_TIMEOUT": "3600",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
})
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
# Prefer TF32 on Ampere/Ada for faster GEMMs while retaining fp32 API numerics
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
)
from pretraining.vit_utils import (
    infer_num_prefix_tokens,
    pool_patch_tokens,
)
from pretraining.data_pipeline import (
    BinaryDatasetConfig,
    BinaryDatasetMode,
    DataInputConfig,
    DataPolicy,
    NonEmptyResamplingDataset,
    SSLDataModule,
    preview_augmentations_from_dataset,
    resolve_binary_dataloader_workers,
)
from pretraining.dataset_registry import dataset_hf_cache_root
from pretraining.model_factory import ModelInputConfig, SSLModelFactory
from pretraining.model_factory import load_dinov2_config
from pretraining.run_types import (
    BatchSchedulePlan,
    Branch,
    BranchRunConfig,
    InitMode,
    LossMode,
    ModePlannerConfig,
    PROFILE_SETTINGS,
    RunSweep,
    SSLProfile,
    TrainingMode,
    TrainingRunConfig,
    is_contrastive_loss,
)
from pretraining.training_modes import plan_runs_for_spec, proj_token
from utils.model_utils import cleanup_after_run, set_all_seeds, get_device, _DevicePrefetch
from utils.strada_datasets import (
    BinaryFileDataset,
    BlockShuffleDistributedSampler,
    DEFAULT_MEGA_CHUNK_BYTES,
    DEFAULT_EPOCHS_PER_MEGA_CHUNK,
    RECORD_BYTES,
)
from utils.manifest_utils import (
    derive_arch_tok_from_config,
    phase_tag_from_output_dir,
    short_arch_name,
)
import math
from trainer_callbacks import (
    ReconVizCallback,
    MetricsPlotCallback,
    ProcessorSaveCallback,
    TemperatureScheduleCallback,
)

# -------------------------------------------------------------------------
# --- Helpers: worker seeding ---
def seed_worker(worker_id: int):
    """Different RNG stream per worker for Python, NumPy, and PyTorch."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _find_binary_dataset(dataset):
    """Return the underlying BinaryFileDataset even when it is wrapped."""
    current = dataset
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, BinaryFileDataset):
            return current
        seen.add(id(current))
        current = getattr(current, "base", None)
    return None


def _binary_mega_chunk_summary(binary_dataset, configured_full_file_epochs: int | None = None) -> dict[str, int | str]:
    n_samples = int(len(binary_dataset))
    mega_chunk_samples = max(1, int(DEFAULT_MEGA_CHUNK_BYTES // int(RECORD_BYTES)))
    mega_chunk_count = int(math.ceil(n_samples / mega_chunk_samples)) if n_samples > 0 else 0
    epochs_per_mega_chunk = int(configured_full_file_epochs or DEFAULT_EPOCHS_PER_MEGA_CHUNK)
    internal_trainer_epochs = int(max(1, mega_chunk_count) * max(1, epochs_per_mega_chunk))
    return {
        "mega_chunk_bytes": int(DEFAULT_MEGA_CHUNK_BYTES),
        "mega_chunk_samples": int(mega_chunk_samples),
        "mega_chunk_count": int(mega_chunk_count),
        "mega_chunk_order_policy": "rotate",
        "mega_chunk_order_period_epochs": 1,
        "epochs_per_mega_chunk": int(epochs_per_mega_chunk),
        "configured_full_file_epochs": int(epochs_per_mega_chunk),
        "internal_trainer_epochs": int(internal_trainer_epochs),
        "epoch_semantics": "mega_chunk_local",
    }


def _prepare_rank_sharded_binary_dataloader(accelerator, dataloader):
    """Use Accelerate's DataLoader wrapper without re-sharding an already rank-local loader."""
    try:
        from accelerate.data_loader import prepare_data_loader
    except Exception as exc:
        raise RuntimeError(
            "Could not import accelerate.data_loader.prepare_data_loader. "
            "The binary DataLoader is already rank-sharded by BlockShuffleDistributedSampler; "
            "plain accelerator.prepare(dataloader) double-shards it on this runtime."
        ) from exc

    import inspect

    candidate_kwargs = {
        "device": getattr(accelerator, "device", None),
        "num_processes": 1,
        "process_index": 0,
        "split_batches": False,
        "put_on_device": True,
        "rng_types": getattr(accelerator, "rng_types", None),
    }
    signature = inspect.signature(prepare_data_loader)
    supported_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters and value is not None
    }
    return prepare_data_loader(dataloader, **supported_kwargs)


@dataclass(frozen=True)
class ProfileRuntimeState:
    output_dir: str
    device: torch.device
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temp_warm: float
    temp_main: float
    adam_betas: tuple[float, float]
    apply_norm: bool


@dataclass
class RuntimeBundle:
    config: object
    model: torch.nn.Module
    init_tok: str
    processor: object
    transform: object
    mean: list[float]
    std: list[float]
    train_dataset: torch.utils.data.Dataset
    preview_dataset: torch.utils.data.Dataset | None
    data_collator: object


@dataclass
class TrainerRuntimeState:
    trainer: "MAETrainer"
    training_args: TrainingArguments
    batch_plan: BatchSchedulePlan
    gradient_accum_steps: int
    global_bs_imgs: int
    global_bs_views: int
    world_size: int
    save_every_epochs: int

# --- Custom Trainer with auxiliary L1 loss ---
class MAETrainer(Trainer):
    def __init__(self, *args,
                 loss_mode: LossMode = LossMode.L2,
                 loss_weights: dict | None = None,
                 simclr_hardness_tau: float = 0.15,
                 simclr_hardness_alpha: float = 0.5,
                 hcl_beta: float = 1.0,
                 hcl_tau_plus: float | None = None,
                 simclr_debias_tau_plus: float = 0.10,
                 longrun_no_decay: bool = False,
                 contrastive_diagnostics_enabled: bool = True,
                 heavy_contrastive_diagnostics: bool = True,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.loss_mode = loss_mode
        # Accept a dict of weights; fall back to individual args for backward compatibility
        self.loss_weights = loss_weights
        self._use_cls_token = False  # always use mean‑pooled patch tokens for Contrastive features
        self.simclr_hardness_tau = float(simclr_hardness_tau)
        self.simclr_hardness_alpha = float(simclr_hardness_alpha)
        self.hcl_beta = float(hcl_beta)
        self.hcl_tau_plus = None if hcl_tau_plus is None else float(hcl_tau_plus)
        self.simclr_debias_tau_plus = float(simclr_debias_tau_plus)
        self.longrun_no_decay = bool(longrun_no_decay)
        self.contrastive_diagnostics_enabled = bool(contrastive_diagnostics_enabled)
        self.heavy_contrastive_diagnostics = bool(heavy_contrastive_diagnostics)

        if is_contrastive_loss(self.loss_mode):
            self.temperature = 0.1
            self.force_constant_temperature = None
        else:
            self.temperature = 0.1
            self.force_constant_temperature = None

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if not self.longrun_no_decay:
            return super().create_optimizer()

        no_decay_module_types = (
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
        )
        no_decay_module_params = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, no_decay_module_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    no_decay_module_params.add(full_name)

        token_name_markers = (
            "pos_embed",
            "position_embedding",
            "position_embeddings",
            "positional_embedding",
            "positional_embeddings",
            "cls_token",
            "reg_token",
            "reg_tokens",
            "mask_token",
            "register_token",
            "register_tokens",
        )

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                name.endswith(".bias")
                or name in no_decay_module_params
                or any(marker in name for marker in token_name_markers)
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        decay_tensors = len(decay_params)
        no_decay_tensors = len(no_decay_params)
        decay_params_count = sum(p.numel() for p in decay_params)
        no_decay_params_count = sum(p.numel() for p in no_decay_params)
        guarded_print(
            f"[OptGroups] decay_tensors={decay_tensors} params={decay_params_count} | "
            f"no_decay_tensors={no_decay_tensors} params={no_decay_params_count} | "
            f"weight_decay={self.args.weight_decay}"
        )
        if float(self.args.weight_decay) <= 0.0:
            guarded_print("[OptGroups][Warn] weight_decay is non-positive; decay group has no effect.")
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _load_rng_state(self, resume_from_checkpoint: str | None) -> None:
        """
        HF Trainer resumes RNG state via torch.load(...). With PyTorch 2.6+, torch.load defaults to
        weights_only=True which can reject numpy objects stored in rng_state*.pth. Our checkpoints
        are trusted (self-generated), so we explicitly load with weights_only=False and gracefully
        skip RNG restore if files are missing/corrupt.
        """
        if not resume_from_checkpoint:
            return

        # Match HF naming: either a shared rng_state.pth or per-rank rng_state_{process_index}.pth
        try:
            process_index = int(getattr(self.args, "process_index", os.getenv("RANK", "0")))
        except Exception:
            process_index = int(os.getenv("RANK", "0"))

        candidates = [
            os.path.join(resume_from_checkpoint, f"rng_state_{process_index}.pth"),
            os.path.join(resume_from_checkpoint, "rng_state.pth"),
        ]
        rng_file = next((p for p in candidates if os.path.isfile(p)), None)
        if rng_file is None:
            guarded_print(f"[Resume][RNG] no rng_state*.pth found under {resume_from_checkpoint}; continuing without RNG restore")
            return

        try:
            try:
                checkpoint_rng_state = torch.load(rng_file, map_location="cpu", weights_only=False)
            except TypeError:
                # Older PyTorch without weights_only kwarg.
                checkpoint_rng_state = torch.load(rng_file, map_location="cpu")
        except Exception as e:
            guarded_print(f"[Resume][RNG] failed to load {rng_file}: {e}; continuing without RNG restore")
            return

        if not isinstance(checkpoint_rng_state, dict):
            guarded_print(f"[Resume][RNG] unexpected rng state type {type(checkpoint_rng_state)} from {rng_file}; continuing without RNG restore")
            return

        try:
            import random as _random
            import numpy as _np

            if "python" in checkpoint_rng_state:
                _random.setstate(checkpoint_rng_state["python"])
            if "numpy" in checkpoint_rng_state:
                _np.random.set_state(checkpoint_rng_state["numpy"])
            if "cpu" in checkpoint_rng_state:
                torch.random.set_rng_state(checkpoint_rng_state["cpu"])
            if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
                try:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
                except Exception:
                    # Some versions expose only set_rng_state_all.
                    try:
                        torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])
                    except Exception:
                        pass
            guarded_print(f"[Resume][RNG] restored RNG state from {rng_file}")
        except Exception as e:
            guarded_print(f"[Resume][RNG] failed to apply RNG state from {rng_file}: {e}; continuing without RNG restore")
            return

    def get_train_dataloader(self):
        dataset = self.train_dataset
        sampler = self._get_train_sampler()
        binary_dataset = _find_binary_dataset(dataset)

        binary_num_replicas = 1
        binary_rank = 0

        # For binary memmaps, the sampler owns DDP rank sharding. We still use
        # Accelerate's DataLoader wrapper, but with num_processes=1 so the
        # already rank-local stream is not sharded a second time.
        if binary_dataset is not None:
            try:
                binary_num_replicas = int(getattr(self.accelerator, "num_processes", 1))
                binary_rank = int(getattr(self.accelerator, "process_index", 0))
            except Exception:
                binary_num_replicas = int(os.getenv("WORLD_SIZE", "1"))
                binary_rank = int(os.getenv("RANK", "0"))

            sampler = BlockShuffleDistributedSampler(
                dataset=dataset,
                block_size=getattr(binary_dataset, "block_size", 8192),
                num_replicas=binary_num_replicas,
                rank=binary_rank,
                shuffle=True,
                seed=int(getattr(self.args, "seed", 42)),
                drop_last=bool(self.args.dataloader_drop_last or is_contrastive_loss(self.loss_mode)),
                num_workers=int(self.args.dataloader_num_workers),
                interleave_workers=True,
                warm_ahead_blocks=2,
                readahead_chunk_bytes=512 << 20,
                warm_start_blocks=4,
                warm_bytes_at_init=16 << 30,
                warm_local_rank0_only=True,
                mega_chunk_bytes=DEFAULT_MEGA_CHUNK_BYTES,
                mega_chunk_order="rotate",
                mega_chunk_order_period_epochs=1,
                epochs_per_mega_chunk=int(getattr(self.args, "configured_full_file_epochs", 1)),
                binary_dataset=binary_dataset,
            )

        g = torch.Generator()
        try:
            g.manual_seed(int(self.args.seed))
        except Exception:
            g.manual_seed(42)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=(self.args.dataloader_drop_last or is_contrastive_loss(self.loss_mode)),
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            collate_fn=self.data_collator,
            worker_init_fn=seed_worker,
            generator=g,
        )
        try:
            guarded_print(
                f"[DataLoader] bs_per_device={self.args.per_device_train_batch_size} | drop_last={self.args.dataloader_drop_last or is_contrastive_loss(self.loss_mode)} | workers={self.args.dataloader_num_workers} | prefetch={self.args.dataloader_prefetch_factor} | persistent={self.args.dataloader_persistent_workers}"
            )
            if binary_dataset is not None:
                mega = sampler.mega_chunk_summary() if hasattr(sampler, "mega_chunk_summary") else _binary_mega_chunk_summary(
                    binary_dataset,
                    configured_full_file_epochs=int(getattr(self.args, "configured_full_file_epochs", 1)),
                )
                active = sampler.active_chunk_summary(
                    trace_blocks=(8 if os.getenv("STRADA_DATALOADER_TRACE", "").strip().lower() in {"1", "true", "yes", "on"} else 0)
                ) if hasattr(sampler, "active_chunk_summary") else {}
                sampler_len = len(sampler)
                dataloader_len = len(dataloader)
                guarded_print(
                    f"[DataLoader] block_sampler=on | block_size={getattr(binary_dataset, 'block_size', 'unknown')} "
                    f"| sharding=rank_aware_sampler | accelerator_prepare=used_no_reshard | warm_local_rank0_only=True "
                    f"| interleave_workers=True"
                )
                guarded_print(
                    f"[DataLoader] file={len(binary_dataset) * RECORD_BYTES / (1024 ** 4):.2f} TiB "
                    f"| mega_chunk={int(mega['mega_chunk_bytes']) / (1024 ** 3):.0f} GiB "
                    f"| mega_chunks={mega['mega_chunk_count']} "
                    f"| mega_order={mega['mega_chunk_order_policy']} "
                    f"| mega_period_cycles={mega['mega_chunk_order_period_epochs']} "
                    f"| epochs_per_mega_chunk={mega['epochs_per_mega_chunk']} "
                    f"| epoch_semantics={mega['epoch_semantics']} "
                    f"| workers/rank={self.args.dataloader_num_workers}"
                )
                guarded_print(
                    f"[DataLoader] world_size={binary_num_replicas} | rank={binary_rank} "
                    f"| sampler_len={sampler_len} | dataloader_len={dataloader_len} "
                    f"| active_chunk={active.get('chunk_id')} "
                    f"| chunk_range=[{active.get('chunk_start')},{active.get('chunk_end')}) "
                    f"| rank_pos=[{active.get('rank_start_pos')},{active.get('rank_end_pos')}) "
                    f"| local_epoch={active.get('local_epoch')}"
                )
                if os.getenv("STRADA_DATALOADER_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    print(
                        f"[DataLoaderTrace] rank={binary_rank}/{binary_num_replicas} "
                        f"active_chunk={active.get('chunk_id')} first_block_ids={active.get('first_block_ids', [])}",
                        flush=True,
                    )
        except Exception:
            pass
        if binary_dataset is not None:
            sampler_len = len(sampler)
            batch_size = int(self.args.per_device_train_batch_size)
            drop_last = bool(self.args.dataloader_drop_last or is_contrastive_loss(self.loss_mode))
            expected_len = sampler_len // batch_size if drop_last else int(math.ceil(sampler_len / batch_size))
            before_prepare_len = len(dataloader)
            if before_prepare_len != expected_len:
                raise RuntimeError(
                    "Binary DataLoader length mismatch before training: "
                    f"actual_len={before_prepare_len}, expected_len={expected_len}, "
                    f"sampler_len={sampler_len}, batch_size={batch_size}, "
                    f"drop_last={drop_last}, world_size={binary_num_replicas}, rank={binary_rank}. "
                    "This usually means the rank-aware binary sampler is being sharded again."
                )
            prepared = _prepare_rank_sharded_binary_dataloader(self.accelerator, dataloader)
            after_prepare_len = len(prepared)
            if after_prepare_len != expected_len:
                raise RuntimeError(
                    "Binary DataLoader length changed after no-reshard Accelerate preparation: "
                    f"before_prepare_len={before_prepare_len}, after_prepare_len={after_prepare_len}, "
                    f"expected_len={expected_len}, sampler_len={sampler_len}, "
                    f"batch_size={batch_size}, drop_last={drop_last}, "
                    f"world_size={binary_num_replicas}, rank={binary_rank}. "
                    "The binary sampler owns rank sharding; prepared length must remain unchanged."
                )
            guarded_print(
                f"[DataLoader] dataloader_len_after_prepare={after_prepare_len} "
                f"| length_guard=passed | accelerator_prepare=used_no_reshard"
            )
            return prepared
        return self.accelerator.prepare(dataloader)

    def log(self, logs, *args, **kwargs):
        """
        Extend HF Trainer logs with:
          - `grad_norm_preclip` / `grad_norm_postclip` (post is a proxy using max_grad_norm)
          - sub-losses from the last `compute_loss` call

        Note: HF/accelerate report `grad_norm` as the *pre-clip* norm. We derive a post-clip proxy as
        `min(pre, max_grad_norm)` and log the implied clip coefficient.
        """
        try:
            if isinstance(logs, dict):
                # 1) grad norm pre/post clip
                if "grad_norm" in logs:
                    try:
                        pre = float(logs["grad_norm"])
                    except Exception:
                        pre = None
                    if pre is not None:
                        logs["grad_norm_preclip"] = float(pre)
                        try:
                            mx = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
                        except Exception:
                            mx = 0.0
                        if mx > 0.0:
                            logs["grad_norm_postclip"] = float(min(pre, mx))
                            logs["grad_norm_clip_coef"] = float(min(1.0, mx / max(pre, 1e-12)))
                        else:
                            logs["grad_norm_postclip"] = float(pre)
                            logs["grad_norm_clip_coef"] = 1.0

                # 2) sub-losses from compute_loss
                if hasattr(self, "_current_sub_losses") and isinstance(getattr(self, "_current_sub_losses"), dict):
                    logs.update(self._current_sub_losses)
        except Exception:
            pass

        return super().log(logs, *args, **kwargs)

    def _compute_infonce_loss(
        self,
        z_query: torch.Tensor,
        z_keys: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_weights: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unified InfoNCE with optional negative reweighting and valid-mask.

        Args:
            z_query: (N, D) normalized query embeddings.
            z_keys:  (M, D) normalized key embeddings.
            pos_mask: (N, M) boolean mask of positives for each query.
            neg_weights: (N, M) optional weights for negatives (zeros elsewhere).
            valid_mask: (N, M) optional boolean mask of entries allowed in the softmax denominator
                        (e.g., to exclude self-pairs/diagonal). If None, all entries are valid.

        Returns:
            Scalar loss tensor (mean over valid rows).
        """
        # Similarity logits with temperature
        logits = (z_query.float() @ z_keys.float().T) / float(self.temperature)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if valid_mask is not None:
            # Mask invalid entries out of the softmax with -inf
            logits = logits.masked_fill(~valid_mask, float('-inf'))

        # Exponentiate for numerator/denominator composition
        exp_logits = torch.exp(logits)
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)

        pos_mask_f = pos_mask.float()
        pos_sum = (exp_logits * pos_mask_f).sum(dim=1)

        if neg_weights is None:
            # Uniform negatives: sum over all valid non-positives
            if valid_mask is not None:
                valid_f = valid_mask.float()
                # Denominator includes positives + negatives over valid entries
                denom = (exp_logits * valid_f).sum(dim=1)
            else:
                denom = exp_logits.sum(dim=1)
        else:
            # Weighted negatives: only count negatives (neg_weights should already be zero elsewhere)
            neg_sum = (exp_logits * neg_weights).sum(dim=1)
            # Denominator = positives + weighted negatives
            denom = (pos_sum + neg_sum)

        # Numerical safety
        pos_sum = pos_sum.clamp_min(1e-8)
        denom = denom.clamp_min(1e-8)

        # Multi-positive InfoNCE (sum over positives in numerator)
        log_pos = torch.log(pos_sum) - torch.log(denom)

        # Only keep rows that actually have a positive
        valid_rows = (pos_mask_f.sum(dim=1) > 0)
        if valid_rows.any():
            return -(log_pos[valid_rows]).mean()
        else:
            return logits.new_tensor(0.0)

    def _compute_hcl_loss(
        self,
        z_query: torch.Tensor,
        z_keys: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        beta: float,
        tau_plus: float,
    ) -> torch.Tensor:
        """Hard negative reweighted InfoNCE (Robinson et al., 2021).

        Implements the "hard sampling objective" (importance-weighted negatives) in the paper:
        weights are proportional to negative mass, i.e. `w_k ∝ neg_k`, and the resulting negative
        term becomes `Σ_k w_k * neg_k` (hard negatives contribute more).

        Notes:
        - Assumes embeddings are L2-normalized (as in SimCLR); then no extra clipping is needed.
        - `tau_plus` can be set to 0.0 to recover a non-debiased hard-negative objective.
        """
        import math as _math

        beta = float(max(0.0, beta))
        tau_plus = float(max(0.0, min(float(tau_plus), 0.999)))
        temp = float(self.temperature)

        logits = (z_query.float() @ z_keys.float().T) / max(temp, 1e-8)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = logits.masked_fill(~valid_mask, float("-inf"))

        exp_logits = torch.exp(logits)
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)

        pos_f = pos_mask.float()
        valid_f = valid_mask.float()
        neg_mask = valid_mask & (~pos_mask)
        neg_f = neg_mask.float()

        pos_sum = (exp_logits * pos_f).sum(dim=1)
        all_sum = (exp_logits * valid_f).sum(dim=1)
        neg_sum = (all_sum - pos_sum).clamp_min(0.0)

        # Counts
        n_pos = pos_f.sum(dim=1).clamp_min(1.0)
        n_neg = neg_f.sum(dim=1).clamp_min(1.0)
        pos_mean = pos_sum / n_pos

        # Negative masses (vector per row, zeros outside negatives)
        neg_mass = exp_logits * neg_f
        neg_mean = (neg_sum / n_neg).clamp_min(1e-8)

        # Paper-style reweighting: reweight_k = (beta * neg_k) / mean(neg)
        reweight = (beta * neg_mass) / neg_mean.unsqueeze(1)
        neg_reweighted_sum = (reweight * neg_mass).sum(dim=1)

        denom_scale = max(1e-8, 1.0 - tau_plus)
        Ng = (neg_reweighted_sum - (tau_plus * n_neg * pos_mean)) / denom_scale

        # Lower-bound from the paper (scalar, not scaled by n_neg)
        Ng = torch.clamp(Ng, min=float(_math.exp(-1.0 / max(temp, 1e-8))))

        pos_sum = pos_sum.clamp_min(1e-8)
        denom = (pos_sum + Ng).clamp_min(1e-8)
        log_pos = torch.log(pos_sum) - torch.log(denom)

        valid_rows = (pos_f.sum(dim=1) > 0)
        if valid_rows.any():
            return -(log_pos[valid_rows]).mean()
        return logits.new_tensor(0.0)

    def _compute_dcl_loss(
        self,
        z_query: torch.Tensor,
        z_keys: torch.Tensor,
        pos_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        tau_plus: float,
    ) -> torch.Tensor:
        """Debiased Contrastive Learning (DCL) style correction (Chuang et al., 2020).

        This is intended for regimes with substantial false-negative probability. It uses a
        class-prior-like knob `tau_plus` to reduce the effective negative mass.

        Args:
            z_query: (N, D) normalized query embeddings.
            z_keys:  (M, D) normalized key embeddings.
            pos_mask: (N, M) boolean positives mask.
            valid_mask: (N, M) boolean mask for entries allowed in denominator (e.g., excludes diagonal).
            tau_plus: prior in [0, 1). Roughly, probability a sampled "negative" is actually positive.
        """
        import math as _math

        # Clamp tau_plus to sane range
        tau_plus = float(max(0.0, min(float(tau_plus), 0.999)))
        temp = float(self.temperature)

        logits = (z_query.float() @ z_keys.float().T) / max(temp, 1e-8)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = logits.masked_fill(~valid_mask, float("-inf"))

        exp_logits = torch.exp(logits)
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)

        pos_f = pos_mask.float()
        valid_f = valid_mask.float()

        pos_sum = (exp_logits * pos_f).sum(dim=1)
        all_sum = (exp_logits * valid_f).sum(dim=1)
        neg_sum = (all_sum - pos_sum).clamp_min(0.0)

        # Counts (float for math)
        n_pos = pos_f.sum(dim=1).clamp_min(1.0)
        neg_mask = valid_mask & (~pos_mask)
        n_neg = neg_mask.float().sum(dim=1).clamp_min(1.0)

        # Use mean positive mass when multiple positives exist (multi-view)
        pos_mean = pos_sum / n_pos

        # Debiased effective negative mass (Ng) (common implementation multiplies by n_neg)
        denom_scale = max(1e-8, 1.0 - tau_plus)
        Ng = (neg_sum - (tau_plus * n_neg * pos_mean)) / denom_scale

        # Clamp to avoid negative/degenerate denominators
        min_Ng = n_neg * float(_math.exp(-1.0 / max(temp, 1e-8)))
        Ng = torch.clamp(Ng, min=min_Ng)

        pos_sum = pos_sum.clamp_min(1e-8)
        denom = (pos_sum + Ng).clamp_min(1e-8)
        log_pos = torch.log(pos_sum) - torch.log(denom)

        valid_rows = (pos_mask.float().sum(dim=1) > 0)
        if valid_rows.any():
            return -(log_pos[valid_rows]).mean()
        return logits.new_tensor(0.0)

    # ---- Shared helpers for SIMCLR ----
    def _gather_with_grad(self, t: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            try:
                from torch.distributed.nn.functional import all_gather as dist_all_gather
            except Exception as exc:
                raise RuntimeError(
                    "Distributed contrastive training requires a grad-preserving all_gather implementation, "
                    "but torch.distributed.nn.functional.all_gather is unavailable on this runtime."
                ) from exc
            gathered = dist_all_gather(t)
            world_size = dist.get_world_size()
            if not isinstance(gathered, (list, tuple)) or len(gathered) != world_size:
                raise RuntimeError(
                    f"Grad-preserving all_gather returned {type(gathered).__name__} with len="
                    f"{len(gathered) if isinstance(gathered, (list, tuple)) else 'n/a'}; expected {world_size} tensors."
                )
            out = torch.cat(tuple(gathered), dim=0)
            expected_rows = world_size * int(t.size(0))
            if out.size(0) != expected_rows:
                raise RuntimeError(
                    f"Grad-preserving all_gather returned shape {tuple(out.shape)}; "
                    f"expected first dimension {expected_rows}."
                )
            return out
        return t

    def _gather_simclr_embeddings(self, z: torch.Tensor):
        """All-gather embeddings across ranks, returning concatenated tensor and local block offsets."""
        N_local = z.size(0)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            z_all = self._gather_with_grad(z)
            block_start = rank * N_local
            block_end = block_start + N_local
        else:
            z_all = z
            block_start, block_end = 0, N_local
        return z_all, block_start, block_end

    def _build_simclr_masks(
        self,
        z_all: torch.Tensor,
        n_local: int,
        block_start: int,
        block_end: int,
        B: int,
    ):
        """Construct diagonal exclusion mask and positive mask for SimCLR-style losses."""
        device = z_all.device
        diag_mask = torch.zeros((n_local, z_all.size(0)), dtype=torch.bool, device=device)
        diag_mask[torch.arange(n_local, device=device), torch.arange(block_start, block_end, device=device)] = True

        idx = torch.arange(n_local, device=device)
        orig_ids = idx % B
        local_pos = (orig_ids[:, None] == orig_ids[None, :]) & (~torch.eye(n_local, device=device, dtype=torch.bool))

        pos_mask = torch.zeros_like(diag_mask)
        pos_mask[:, block_start:block_end] = local_pos
        return diag_mask, pos_mask

    def _simclr_common_forward(self, model_wrapped, pixel_values: torch.Tensor, inputs):
        """Shared encoder/projector forward + mask construction for SimCLR variants."""
        x = pixel_values
        n_views = int(inputs.get("n_views", 2))
        assert x.dim() == 4 and x.size(0) % n_views == 0, \
            f"Expect (B*n_views, C, H, W); got {tuple(x.shape)} with n_views={n_views}"

        B = x.size(0) // n_views

        enc = model_wrapped.vit
        proj = model_wrapped.projector_head

        x_fp32 = x.float()
        out = enc(x_fp32, output_hidden_states=False)
        reps = None
        # Prefer the backbone's native pooled representation when available (e.g. DINOv2),
        # and fall back to pooled patch tokens otherwise (e.g. MAE-style encoders).
        pooled = getattr(out, "pooler_output", None)
        if isinstance(pooled, torch.Tensor) and pooled.dim() == 2 and pooled.size(0) == x_fp32.size(0):
            reps = pooled
            self._embed_rep_last_used = "backbone pooler_output"
        else:
            hs = out.last_hidden_state
            reps = pool_patch_tokens(hs, use_cls_token=self._use_cls_token, model_or_config=enc)
            self._embed_rep_last_used = "pooled patch tokens"

        z = proj(reps)
        z = torch.nn.functional.normalize(z, dim=1, eps=1e-6)
        if not torch.isfinite(z).all():
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        z_all, block_start, block_end = self._gather_simclr_embeddings(z)
        diag_mask, pos_mask = self._build_simclr_masks(
            z_all=z_all,
            n_local=z.size(0),
            block_start=block_start,
            block_end=block_end,
            B=B,
        )

        return {
            "z": z,
            "z_all": z_all,
            "reps": reps,
            "B": B,
            "n_views": n_views,
            "diag_mask": diag_mask,
            "pos_mask": pos_mask,
        }

    def _should_run_contrastive_diagnostics(self) -> bool:
        if not self.contrastive_diagnostics_enabled:
            return False
        state = getattr(self, "state", None)
        step = int(getattr(state, "global_step", 0)) if state is not None else 0
        log_every = max(1, int(getattr(self.args, "logging_steps", 100)))
        return ((step + 1) % log_every) == 0

    def _collect_contrastive_similarity_diagnostics(
        self,
        *,
        z: torch.Tensor,
        z_all: torch.Tensor,
        loss: torch.Tensor,
        B: int,
        n_views: int,
    ) -> dict[str, float]:
        try:
            import math as _math

            keys_global = int(z_all.size(0))
            keff_est = float(_math.exp(float(loss.detach().cpu().item())))
            if n_views >= 2 and B > 0:
                pos_sims = (z[:B] * z[B:2 * B]).sum(dim=1)
                pos_sim_mean = float(pos_sims.mean().detach().cpu().item())
            else:
                pos_sim_mean = float("nan")

            sample_k = min(512, keys_global)
            if sample_k > 0 and B > 0:
                sample_idx = torch.randint(0, keys_global, (sample_k,), device=z.device)
                neg_logits_sample = (z[:B] @ z_all[sample_idx].T)
                neg_sim_mean = float(neg_logits_sample.mean().detach().cpu().item()) if neg_logits_sample.numel() > 0 else float("nan")
                neg_sim_std = float(neg_logits_sample.std().detach().cpu().item()) if neg_logits_sample.numel() > 1 else float("nan")
            else:
                neg_sim_mean = float("nan")
                neg_sim_std = float("nan")

            return {
                "clr_pos_sim_mean": pos_sim_mean,
                "clr_neg_sim_mean": neg_sim_mean,
                "clr_neg_sim_std": neg_sim_std,
                "clr_keff_est": keff_est,
            }
        except Exception:
            return {}

    def _collect_contrastive_geometry_diagnostics(
        self,
        *,
        reps: torch.Tensor,
        B: int,
    ) -> dict[str, float]:
        if not self.heavy_contrastive_diagnostics or B <= 1:
            return {}
        try:
            reps_first = reps[:B].detach()
            reps_g = self._gather_with_grad(reps_first)
            if reps_g.size(0) <= 1:
                return {
                    "clr_cos_std": float("nan"),
                    "clr_cos_mean_abs": float("nan"),
                    "clr_eff_rank": float("nan"),
                }

            u = torch.nn.functional.normalize(reps_g, dim=1)
            C = u @ u.T
            mask = ~torch.eye(C.size(0), dtype=torch.bool, device=C.device)
            vals = C[mask]
            if vals.numel() > 0:
                clr_cos_std = vals.std().item()
                clr_cos_mean_abs = vals.abs().mean().item()
            else:
                clr_cos_std = float("nan")
                clr_cos_mean_abs = float("nan")

            S = torch.linalg.svdvals(u)
            num = S.sum() ** 2
            den = (S.pow(2).sum()).clamp_min(1e-8)
            clr_eff_rank = float((num / den).item())
            return {
                "clr_cos_std": clr_cos_std,
                "clr_cos_mean_abs": clr_cos_mean_abs,
                "clr_eff_rank": clr_eff_rank,
            }
        except Exception:
            return {}

    def _update_contrastive_diagnostics(
        self,
        *,
        z: torch.Tensor,
        z_all: torch.Tensor,
        reps: torch.Tensor,
        loss: torch.Tensor,
        B: int,
        n_views: int,
    ) -> None:
        if not self._should_run_contrastive_diagnostics():
            return
        self._current_sub_losses.update(
            self._collect_contrastive_similarity_diagnostics(
                z=z,
                z_all=z_all,
                loss=loss,
                B=B,
                n_views=n_views,
            )
        )
        self._current_sub_losses.update(
            self._collect_contrastive_geometry_diagnostics(
                reps=reps,
                B=B,
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_wrapped = model.module if hasattr(model, "module") else model

        def _need(name: str):
            if self.loss_weights is None or name not in self.loss_weights:
                raise ValueError(f"Missing required loss weight '{name}' for loss_mode={self.loss_mode.value}")
            return self.loss_weights[name]

        # ---- Input preprocessing ----
        pixel_values = inputs["pixel_values"]
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # Expand grayscale to 3 channels if needed
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.expand(-1, 3, -1, -1).contiguous()

        # Run MAE forward only for L2 / L2_L1. Do NOT call MAE in contrastive losses (decoder dropped)
        outputs = None
        feat_var_val = None
        if self.loss_mode in (LossMode.L2, LossMode.L2_L1, LossMode.L2_BL1):
            outputs = model_wrapped(pixel_values=pixel_values, output_hidden_states=True)

            # --- Feature variance metric (MAE encoder; pooled-token per-dim variance, mean) ---
            try:
                hs = outputs.hidden_states
                if isinstance(hs, (list, tuple)) and len(hs) > 0:
                    last = hs[-1]  # expected (B, T, D) for ViT-style encoders
                    # Require at least two samples to define a variance across the batch
                    if hasattr(last, "size") and last.size(0) > 1:
                        # Reuse the default pooling policy (CLS vs patch tokens)
                        feat = pool_patch_tokens(last, use_cls_token=self._use_cls_token, model_or_config=model_wrapped)  # (B, D)
                        var_per_dim = feat.var(dim=0, unbiased=False)
                        feat_var_val = float(var_per_dim.mean().detach().cpu().item())
            except Exception:
                pass

        if self.loss_mode == LossMode.L2:
            l2_w = _need("l2")
            loss = l2_w * outputs.loss
            self._current_sub_losses = {
                "l2_loss": float((l2_w * outputs.loss).detach().cpu().item()),
            }
            if feat_var_val is not None:
                self._current_sub_losses["feat_var"] = feat_var_val

        elif self.loss_mode == LossMode.L2_L1:
            # Base L2 from model (masked-patch MSE)
            l2 = outputs.loss
            # Reconstruct pixels; measure vanilla residuals in the same normalized space as pixel_values
            recon = model_wrapped.unpatchify(outputs.logits)
            resid = recon - pixel_values
            l1 = resid.abs().mean()

            l2_w = _need("l2")
            l1_w = _need("l1")
            # Combine: keep MAE's internal masked MSE (l2) and add *unweighted* L1 term
            loss = l2_w * l2 + l1_w * l1

            self._current_sub_losses = {
                "l2_loss": float((l2_w * l2).detach().cpu().item()),
                "l1_loss": float((l1_w * l1).detach().cpu().item()),
            }
            if feat_var_val is not None:
                self._current_sub_losses["feat_var"] = feat_var_val

        elif self.loss_mode == LossMode.L2_BL1:
            # Base L2 from model (masked-patch MSE)
            l2 = outputs.loss

            # Reconstruct pixels; residuals are in the same (possibly normalised) space as pixel_values
            recon = model_wrapped.unpatchify(outputs.logits)
            resid = recon - pixel_values
            abs_resid = resid.abs()

            # Brightness proxy from the input itself (|x|); emphasise brighter pixels
            with torch.no_grad():
                brightness = pixel_values.abs()
                # Normalise brightness so that mean weight is ~1, to keep scale comparable
                eps = 1e-6
                mean_b = brightness.mean().clamp_min(eps)
                w = brightness / mean_b

            bl1 = (abs_resid * w).mean()

            l2_w = _need("l2")
            bl1_w = _need("bl1")
            loss = l2_w * l2 + bl1_w * bl1

            self._current_sub_losses = {
                "l2_loss": float((l2_w * l2).detach().cpu().item()),
                "bl1_loss": float((bl1_w * bl1).detach().cpu().item()),
            }
            if feat_var_val is not None:
                self._current_sub_losses["feat_var"] = feat_var_val

        elif self.loss_mode == LossMode.SIMCLR:
            simclr = self._simclr_common_forward(model_wrapped, pixel_values, inputs)
            z = simclr["z"]
            z_all = simclr["z_all"]
            pos_mask = simclr["pos_mask"]
            diag_mask = simclr["diag_mask"]
            B = simclr["B"]
            n_views = simclr["n_views"]
            reps = simclr["reps"]

            loss = self._compute_infonce_loss(
                z_query=z,
                z_keys=z_all,
                pos_mask=pos_mask,
                neg_weights=None,
                valid_mask=~diag_mask,
            )
            self._current_sub_losses = {
                "clr_loss": float(loss.detach().cpu().item()),
                "n_views": int(n_views),
            }
            if B > 1:
                feat_var_val = float(reps[:B].var(dim=0, unbiased=False).mean().detach().cpu().item())
                self._current_sub_losses["feat_var"] = feat_var_val
            self._update_contrastive_diagnostics(
                z=z,
                z_all=z_all,
                reps=reps,
                loss=loss,
                B=B,
                n_views=n_views,
            )

        elif self.loss_mode == LossMode.SOFT_HCL:
            simclr = self._simclr_common_forward(model_wrapped, pixel_values, inputs)
            z = simclr["z"]
            z_all = simclr["z_all"]
            pos_mask = simclr["pos_mask"]
            diag_mask = simclr["diag_mask"]
            B = simclr["B"]
            n_views = simclr["n_views"]
            reps = simclr["reps"]

            sim = z @ z_all.T
            sim = sim.clamp(-1.0, 1.0)

            neg_mask = (~pos_mask) & (~diag_mask)
            neg_mask_f = neg_mask.float()

            # --- legacy hardness-aware negative weighting (uniform + hard distribution) ---
            with torch.no_grad():
                neg_sim = sim.detach().clone()
                neg_sim[~neg_mask] = float("-inf")

                tau_h = float(getattr(self, "simclr_hardness_tau", 0.15))
                h_raw = torch.exp(neg_sim / tau_h)
                h_raw[~neg_mask] = 0.0

                h_sum = h_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
                w_hard = h_raw / h_sum

                num_negs = neg_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                w_uniform = neg_mask_f / num_negs

                alpha = float(getattr(self, "simclr_hardness_alpha", 0.5))
                w_final = (1.0 - alpha) * w_uniform + alpha * w_hard
                w_final[~neg_mask] = 0.0
                # Scale weights to sum to the number of negatives per row
                n_neg = neg_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                w_final = w_final * n_neg

            loss = self._compute_infonce_loss(
                z_query=z,
                z_keys=z_all,
                pos_mask=pos_mask,
                neg_weights=w_final,   # already zeros outside negatives
                valid_mask=~diag_mask,
            )
            self._current_sub_losses = {
                "soft_hcl_loss": float(loss.detach().cpu().item()),
                "n_views": int(n_views),
                "soft_hcl_tau": float(getattr(self, "simclr_hardness_tau", 0.15)),
                "soft_hcl_alpha": float(getattr(self, "simclr_hardness_alpha", 0.5)),
            }
            if B > 1:
                feat_var_val = float(reps[:B].var(dim=0, unbiased=False).mean().detach().cpu().item())
                self._current_sub_losses["feat_var"] = feat_var_val
            self._update_contrastive_diagnostics(
                z=z,
                z_all=z_all,
                reps=reps,
                loss=loss,
                B=B,
                n_views=n_views,
            )

        elif self.loss_mode == LossMode.HCL:
            simclr = self._simclr_common_forward(model_wrapped, pixel_values, inputs)
            z = simclr["z"]
            z_all = simclr["z_all"]
            pos_mask = simclr["pos_mask"]
            diag_mask = simclr["diag_mask"]
            B = simclr["B"]
            n_views = simclr["n_views"]
            reps = simclr["reps"]

            beta = float(getattr(self, "hcl_beta", 1.0))
            tau_plus = getattr(self, "hcl_tau_plus", None)
            if tau_plus is None:
                tau_plus = 0.0
            loss = self._compute_hcl_loss(
                z_query=z,
                z_keys=z_all,
                pos_mask=pos_mask,
                valid_mask=~diag_mask,
                beta=beta,
                tau_plus=float(tau_plus),
            )
            self._current_sub_losses = {
                "hcl_loss": float(loss.detach().cpu().item()),
                "n_views": int(n_views),
                "hcl_beta": float(beta),
                "hcl_tau_plus": float(tau_plus),
            }
            if B > 1:
                feat_var_val = float(reps[:B].var(dim=0, unbiased=False).mean().detach().cpu().item())
                self._current_sub_losses["feat_var"] = feat_var_val
            self._update_contrastive_diagnostics(
                z=z,
                z_all=z_all,
                reps=reps,
                loss=loss,
                B=B,
                n_views=n_views,
            )

        elif self.loss_mode == LossMode.SIMCLR_DEBIASED:
            simclr = self._simclr_common_forward(model_wrapped, pixel_values, inputs)
            z = simclr["z"]
            z_all = simclr["z_all"]
            pos_mask = simclr["pos_mask"]
            diag_mask = simclr["diag_mask"]
            B = simclr["B"]
            n_views = simclr["n_views"]
            reps = simclr["reps"]

            tau_plus = float(getattr(self, "simclr_debias_tau_plus", 0.10))
            loss = self._compute_dcl_loss(
                z_query=z,
                z_keys=z_all,
                pos_mask=pos_mask,
                valid_mask=~diag_mask,
                tau_plus=tau_plus,
            )
            self._current_sub_losses = {
                "clr_debiased_loss": float(loss.detach().cpu().item()),
                "n_views": int(n_views),
                "clr_tau_plus": float(tau_plus),
            }
            if B > 1:
                feat_var_val = float(reps[:B].var(dim=0, unbiased=False).mean().detach().cpu().item())
                self._current_sub_losses["feat_var"] = feat_var_val
            self._update_contrastive_diagnostics(
                z=z,
                z_all=z_all,
                reps=reps,
                loss=loss,
                B=B,
                n_views=n_views,
            )

        if return_outputs:
            # For reconstruction modes, return MAE outputs; contrastive modes return None
            return loss, outputs
        else:
            return loss

class FlattenCollator:
    def __init__(self, expected_n_views: int | None = None):
        self.expected_n_views = expected_n_views

    def __call__(self, examples):
        first = examples[0]["pixel_values"]

        # Single-view case
        if not isinstance(first, (list, tuple)):
            if self.expected_n_views not in (None, 1):
                raise ValueError(
                    f"FlattenCollator expected {self.expected_n_views} views but got single-view tensors"
                )
            return {"pixel_values": torch.stack([ex["pixel_values"] for ex in examples], dim=0)}

        # Multi-view case
        n_views = len(first)
        for ex in examples:
            pv = ex.get("pixel_values")
            if not isinstance(pv, (list, tuple)) or len(pv) != n_views:
                raise ValueError("Inconsistent number of views across samples in batch")
        if self.expected_n_views is not None and n_views != int(self.expected_n_views):
            raise ValueError(f"Transform produced n_views={n_views} but expected {self.expected_n_views}")

        grouped = []
        for v in range(n_views):
            for ex in examples:
                grouped.append(ex["pixel_values"][v])
        return {"pixel_values": torch.stack(grouped, dim=0), "n_views": n_views}

def prepare_run_environment(hf_cache_path: str, seed: int) -> None:
    """Set HF cache dirs and global seeds for a run."""
    os.environ["HF_HOME"] = os.path.join(hf_cache_path, "HF_HOME")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_path, "HF_CACHE")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    set_all_seeds(seed)

def ensure_output_structure(output_dir: str) -> None:
    """Create the standard output subfolders for a run."""
    for sub in ("results", "logs", "checkpoints", "recon_viz"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

def derive_batch_and_schedule(
        per_device_train_batch_size: int,
        n_views: int,
        loss_mode: LossMode,
        train_dataset_len: int,
        num_train_epochs: int,
        target_global_bs_imgs: int,
        warmup_ratio: float,
        save_every_epochs: int,
) -> BatchSchedulePlan:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    base_step_imgs = per_device_train_batch_size * world_size
    if base_step_imgs < 1:
        raise ValueError("per_device_train_batch_size and WORLD_SIZE must be positive.")

    a_floor = max(1, target_global_bs_imgs // base_step_imgs)
    a_ceil  = max(1, math.ceil(target_global_bs_imgs / base_step_imgs))
    g_floor = base_step_imgs * a_floor
    g_ceil  = base_step_imgs * a_ceil

    gradient_accum_steps, global_bs_imgs = min(
        ((a_floor, g_floor), (a_ceil, g_ceil)),
        key=lambda t: (abs(t[1] - target_global_bs_imgs), t[0])
    )

    views_factor = (n_views if is_contrastive_loss(loss_mode) else 1)
    global_bs_views = global_bs_imgs * views_factor

    dataloader_drop_last = bool(is_contrastive_loss(loss_mode))
    if dataloader_drop_last:
        # Mirror DistributedSampler(..., drop_last=True): each rank sees a truncated shard.
        shard_size = train_dataset_len // max(1, world_size)
        # Mirror DataLoader(..., drop_last=True): incomplete final batch is dropped.
        microsteps_per_epoch = shard_size // per_device_train_batch_size
    else:
        # Mirror DistributedSampler(..., drop_last=False): each rank is padded to ceil(N / world_size).
        shard_size = int(math.ceil(train_dataset_len / max(1, world_size)))
        # Mirror DataLoader(..., drop_last=False): incomplete final batch is kept.
        microsteps_per_epoch = int(math.ceil(shard_size / per_device_train_batch_size))

    microsteps_per_epoch = max(1, microsteps_per_epoch)
    steps_per_epoch_updates = max(1, microsteps_per_epoch // gradient_accum_steps)

    save_steps = steps_per_epoch_updates * save_every_epochs
    warmup_epochs_effective = max(1, int(round(num_train_epochs * warmup_ratio)))
    warmup_steps = int(max(1, steps_per_epoch_updates) * warmup_epochs_effective)

    return BatchSchedulePlan(
        gradient_accum_steps=gradient_accum_steps,
        global_bs_imgs=global_bs_imgs,
        global_bs_views=global_bs_views,
        microsteps_per_epoch=microsteps_per_epoch,
        steps_per_epoch_updates=steps_per_epoch_updates,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        shard_size=shard_size,
        warmup_epochs_effective=warmup_epochs_effective,
        world_size=world_size,
    )


def resolve_schedule_from_train_dataloader(
    *,
    trainer: Trainer,
    batch_plan: BatchSchedulePlan,
    num_train_epochs: int,
    warmup_ratio: float,
    save_every_epochs: int,
) -> BatchSchedulePlan:
    train_dataloader = trainer.get_train_dataloader()
    try:
        microsteps_per_epoch = int(len(train_dataloader))
    except TypeError as exc:
        raise RuntimeError(
            "Train dataloader length is undefined; cannot derive save/warmup schedule from the actual loader."
        ) from exc
    if microsteps_per_epoch < 1:
        raise RuntimeError("Train dataloader is empty; cannot derive a valid training schedule.")

    train_sampler = getattr(train_dataloader, "sampler", None)
    try:
        shard_size = int(len(train_sampler))
    except Exception:
        shard_size = int(microsteps_per_epoch * max(1, trainer.args.per_device_train_batch_size))

    gradient_accum_steps = max(1, int(batch_plan.gradient_accum_steps))
    steps_per_epoch_updates = max(1, int(math.ceil(microsteps_per_epoch / gradient_accum_steps)))
    warmup_epochs_effective = max(1, int(round(num_train_epochs * warmup_ratio)))
    warmup_steps = int(max(1, steps_per_epoch_updates) * warmup_epochs_effective)
    save_steps = int(max(1, steps_per_epoch_updates) * max(1, int(save_every_epochs)))

    return BatchSchedulePlan(
        gradient_accum_steps=batch_plan.gradient_accum_steps,
        global_bs_imgs=batch_plan.global_bs_imgs,
        global_bs_views=batch_plan.global_bs_views,
        microsteps_per_epoch=microsteps_per_epoch,
        steps_per_epoch_updates=steps_per_epoch_updates,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        shard_size=shard_size,
        warmup_epochs_effective=warmup_epochs_effective,
        world_size=batch_plan.world_size,
    )

def log_profile(profile: SSLProfile, learning_rate: float, weight_decay: float, warmup_ratio: float, temp_warm: float, temp_main: float, num_train_epochs: int) -> None:
    guarded_print(f"[Profile] profile={profile.value} | lr={learning_rate} wd={weight_decay} warmup_ratio={warmup_ratio} temp=({temp_warm}->{temp_main}) epochs={num_train_epochs}")

def log_norm_choice(profile: SSLProfile, apply_norm: bool) -> None:
    guarded_print(f"[InitMode] {profile.value} | [Norm] apply_norm={apply_norm}")

def log_batching_info(per_device_train_batch_size: int, world_size: int, target_global_bs_imgs: int, gradient_accum_steps: int, global_bs_imgs: int, global_bs_views: int, views_factor: int) -> None:
    guarded_print(f"[BatchSizing] per_gpu={per_device_train_batch_size}, gpus={world_size}, base_step={per_device_train_batch_size * world_size}, target={target_global_bs_imgs} -> accum={gradient_accum_steps}, effective_global={global_bs_imgs}")
    guarded_print(f"Global BS (images) = {per_device_train_batch_size} × {world_size} GPUs × {gradient_accum_steps} accum = {global_bs_imgs}")
    guarded_print(f"Global BS (views)  = {global_bs_imgs} images × {views_factor} views = {global_bs_views}")

def log_checkpointing_info(shard_size: int, microsteps_per_epoch: int, gradient_accum_steps: int, steps_per_epoch_updates: int, save_steps: int, save_every_epochs: int) -> None:
    guarded_print(f"[Checkpointing] shard={shard_size}, microsteps/epoch={microsteps_per_epoch}, "
                  f"accum={gradient_accum_steps} -> update_steps/epoch={steps_per_epoch_updates}; "
                  f"save_steps={save_steps} (every {save_every_epochs} epochs)")

def log_warmup_info(num_train_epochs: int, warmup_ratio: float, warmup_epochs_effective: int, warmup_steps: int) -> None:
    guarded_print(f"[LR] num_epochs={num_train_epochs}, warmup_ratio={warmup_ratio:.3f} -> warmup_epochs={warmup_epochs_effective}, warmup_steps={warmup_steps}")

def _resolve_profile_runtime(cfg: TrainingRunConfig) -> ProfileRuntimeState:
    output_dir = os.path.abspath(cfg.output_dir)
    guarded_print(f"[RunDir] run_root={output_dir} results_dir={os.path.join(output_dir, 'results')}")

    profile_settings = PROFILE_SETTINGS[cfg.profile].with_overrides(
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
    )
    log_profile(
        cfg.profile,
        profile_settings.learning_rate,
        profile_settings.weight_decay,
        profile_settings.warmup_ratio,
        profile_settings.temp_warm,
        profile_settings.temp_main,
        cfg.num_train_epochs,
    )

    apply_norm = bool(cfg.profile != SSLProfile.SCRATCH)
    log_norm_choice(cfg.profile, apply_norm)
    guarded_print(f"[Registers] n_registers={cfg.n_registers}")
    guarded_print(f"[Dataset] mode={cfg.dataset_config.mode.value} bin_path={cfg.dataset_config.bin_path}")
    guarded_print(
        f"[DataPolicy] use_roi_crop={cfg.data_policy.use_roi_crop} "
        f"enable_augmentations={cfg.data_policy.enable_augmentations}"
    )

    ensure_output_structure(output_dir)
    device = get_device()
    return ProfileRuntimeState(
        output_dir=output_dir,
        device=device,
        learning_rate=profile_settings.learning_rate,
        weight_decay=profile_settings.weight_decay,
        warmup_ratio=profile_settings.warmup_ratio,
        temp_warm=profile_settings.temp_warm,
        temp_main=profile_settings.temp_main,
        adam_betas=profile_settings.adam_betas,
        apply_norm=apply_norm,
    )


def _build_runtime_bundle(cfg: TrainingRunConfig, profile_state: ProfileRuntimeState) -> RuntimeBundle:
    model_factory = SSLModelFactory(
        ModelInputConfig(
            branch=cfg.branch,
            profile=cfg.profile,
            model_name_or_path=cfg.model_name_or_path,
            mask_ratio=cfg.mask_ratio,
            patch_size=cfg.patch_size,
            n_registers=cfg.n_registers,
            init_from_checkpoint=cfg.init_from_checkpoint,
            use_dino_encoder=cfg.use_dino_encoder,
            drop_path_rate=cfg.drop_path_rate,
            projector_hidden_dim=cfg.projector_hidden_dim,
            projector_out_dim=cfg.projector_out_dim,
        ),
        log_fn=guarded_print,
    )
    model_bundle = model_factory.build(loss_mode=cfg.loss_mode)
    if cfg.expected_init_token is not None and str(model_bundle.init_token) != str(cfg.expected_init_token):
        raise ValueError(
            f"Resolved init token '{model_bundle.init_token}' does not match expected '{cfg.expected_init_token}' "
            f"for output_dir='{profile_state.output_dir}'."
        )

    data_module = SSLDataModule(
        DataInputConfig(dataset=cfg.dataset_config, policy=cfg.data_policy),
        log_fn=guarded_print,
    )
    data_bundle = data_module.build_bundle(
        model_config=model_bundle.config,
        model_name_or_path=cfg.model_name_or_path,
        apply_norm=profile_state.apply_norm,
        loss_mode=cfg.loss_mode,
        n_views=cfg.n_views,
        patch_size=cfg.patch_size,
        build_preview_dataset=bool(cfg.aug_preview_samples and _is_main_process()),
    )
    guarded_print(f"[Dataset] train_len={len(data_bundle.train_dataset)}")
    if data_bundle.preview_dataset is not None:
        guarded_print(f"[Dataset] preview_len={len(data_bundle.preview_dataset)}")

    if cfg.aug_preview_samples and _is_main_process() and data_bundle.preview_dataset is not None:
        try:
            if len(data_bundle.preview_dataset) > 0:
                preview_augmentations_from_dataset(
                    dataset=data_bundle.preview_dataset,
                    transform=data_bundle.transform,
                    mean=data_bundle.mean,
                    std=data_bundle.std,
                    apply_norm=profile_state.apply_norm,
                    num_images=int(cfg.aug_preview_samples),
                    repeats_per_image=max(1, int(cfg.aug_preview_repeats)),
                    out_dir=os.path.join(profile_state.output_dir, "viz"),
                    filename=None,
                    log_fn=guarded_print,
                )
        except Exception as exc:
            guarded_print(f"[AugPreview] failed: {exc}")

    data_collator = None
    if is_contrastive_loss(cfg.loss_mode):
        data_collator = FlattenCollator(expected_n_views=int(cfg.n_views))
    guarded_print("Data collator set up.")

    return RuntimeBundle(
        config=model_bundle.config,
        model=model_bundle.model,
        init_tok=model_bundle.init_token,
        processor=data_bundle.processor,
        transform=data_bundle.transform,
        mean=data_bundle.mean,
        std=data_bundle.std,
        train_dataset=data_bundle.train_dataset,
        preview_dataset=data_bundle.preview_dataset,
        data_collator=data_collator,
    )


def _build_trainer_and_schedule(
    cfg: TrainingRunConfig,
    profile_state: ProfileRuntimeState,
    runtime_bundle: RuntimeBundle,
) -> TrainerRuntimeState:
    target_global_bs_imgs = 2048
    save_every_epochs = 5

    binary_dataset = _find_binary_dataset(runtime_bundle.train_dataset)
    mega_summary = _binary_mega_chunk_summary(binary_dataset, cfg.num_train_epochs) if binary_dataset is not None else None
    internal_num_train_epochs = int(cfg.num_train_epochs)
    if mega_summary is not None:
        internal_num_train_epochs = int(max(1, mega_summary["internal_trainer_epochs"]))

    batch_plan = derive_batch_and_schedule(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        n_views=cfg.n_views,
        loss_mode=cfg.loss_mode,
        train_dataset_len=len(runtime_bundle.train_dataset),
        num_train_epochs=cfg.num_train_epochs,
        target_global_bs_imgs=target_global_bs_imgs,
        warmup_ratio=profile_state.warmup_ratio,
        save_every_epochs=save_every_epochs,
    )
    # For binary mega-chunk runs, cfg.num_train_epochs remains the number of
    # full-file-equivalent passes. HF Trainer epochs are expanded so each
    # internal epoch is one pass over the active mega-chunk.

    gradient_accum_steps = batch_plan.gradient_accum_steps
    global_bs_imgs = batch_plan.global_bs_imgs
    global_bs_views = batch_plan.global_bs_views
    world_size = batch_plan.world_size
    views_factor = (cfg.n_views if is_contrastive_loss(cfg.loss_mode) else 1)
    log_batching_info(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        world_size=world_size,
        target_global_bs_imgs=target_global_bs_imgs,
        gradient_accum_steps=gradient_accum_steps,
        global_bs_imgs=global_bs_imgs,
        global_bs_views=global_bs_views,
        views_factor=views_factor,
    )

    max_grad_norm = 3.0 if is_contrastive_loss(cfg.loss_mode) else 1.0
    use_bf16 = bool(
        profile_state.device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(profile_state.device.type == "cuda" and not use_bf16)
    dataloader_num_workers = int(cfg.dataloader_num_workers)
    if binary_dataset is not None:
        try:
            resolved_workers, worker_policy, file_bytes = resolve_binary_dataloader_workers(
                str(getattr(binary_dataset, "path")),
                dataloader_num_workers,
            )
            if int(resolved_workers) != dataloader_num_workers or worker_policy != "configured":
                guarded_print(
                    f"[DataLoader] workers_per_rank={resolved_workers} "
                    f"(configured={dataloader_num_workers}, policy={worker_policy}, "
                    f"file={file_bytes / (1024 ** 4):.2f} TiB)"
                )
            dataloader_num_workers = int(resolved_workers)
        except Exception as exc:
            guarded_print(
                f"[DataLoader][Warn] failed to resolve binary worker policy: {exc}; "
                f"using configured={dataloader_num_workers}"
            )
        if mega_summary is not None:
            guarded_print(
                "[DataLoader] mega_chunk_local_epoching=on "
                f"configured_full_file_epochs={mega_summary['configured_full_file_epochs']} "
                f"mega_chunks={mega_summary['mega_chunk_count']} "
                f"epochs_per_mega_chunk={mega_summary['epochs_per_mega_chunk']} "
                f"internal_trainer_epochs={mega_summary['internal_trainer_epochs']} "
                f"epoch_semantics={mega_summary['epoch_semantics']}"
            )

    training_args = TrainingArguments(
        output_dir=os.path.join(profile_state.output_dir, "results"),
        save_strategy="steps",
        save_steps=batch_plan.save_steps,
        eval_strategy="no",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accum_steps,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        num_train_epochs=internal_num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=batch_plan.warmup_steps,
        learning_rate=profile_state.learning_rate,
        optim="adamw_torch_fused" if (profile_state.device.type == "cuda") else "adamw_torch",
        max_grad_norm=float(max_grad_norm),
        weight_decay=profile_state.weight_decay,
        logging_dir=os.path.join(profile_state.output_dir, "logs"),
        logging_steps=100,
        report_to=["tensorboard"],
        disable_tqdm=False,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_on_each_node=False,
        dataloader_drop_last=is_contrastive_loss(cfg.loss_mode),
        adam_beta1=profile_state.adam_betas[0],
        adam_beta2=profile_state.adam_betas[1],
    )
    if mega_summary is not None:
        setattr(training_args, "configured_full_file_epochs", int(cfg.num_train_epochs))
        setattr(training_args, "epochs_per_mega_chunk", int(mega_summary["epochs_per_mega_chunk"]))
        setattr(training_args, "mega_chunk_count", int(mega_summary["mega_chunk_count"]))
        setattr(training_args, "epoch_semantics", str(mega_summary["epoch_semantics"]))

    trainer = MAETrainer(
        model=runtime_bundle.model,
        args=training_args,
        train_dataset=NonEmptyResamplingDataset(runtime_bundle.train_dataset, max_resamples=5),
        data_collator=runtime_bundle.data_collator if runtime_bundle.data_collator is not None else None,
        loss_weights=cfg.loss_weights,
        loss_mode=cfg.loss_mode,
        longrun_no_decay=cfg.longrun_no_decay,
        simclr_hardness_tau=float(cfg.soft_hcl_tau_h),
        simclr_hardness_alpha=float(cfg.soft_hcl_alpha),
        hcl_beta=float(cfg.hcl_beta),
        hcl_tau_plus=(None if cfg.hcl_tau_plus is None else float(cfg.hcl_tau_plus)),
        simclr_debias_tau_plus=(0.10 if cfg.simclr_debias_tau_plus is None else float(cfg.simclr_debias_tau_plus)),
        contrastive_diagnostics_enabled=cfg.contrastive_diagnostics_enabled,
        heavy_contrastive_diagnostics=cfg.heavy_contrastive_diagnostics,
    )

    log_checkpointing_info(
        shard_size=batch_plan.shard_size,
        microsteps_per_epoch=batch_plan.microsteps_per_epoch,
        gradient_accum_steps=gradient_accum_steps,
        steps_per_epoch_updates=batch_plan.steps_per_epoch_updates,
        save_steps=batch_plan.save_steps,
        save_every_epochs=save_every_epochs,
    )
    log_warmup_info(
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=profile_state.warmup_ratio,
        warmup_epochs_effective=batch_plan.warmup_epochs_effective,
        warmup_steps=batch_plan.warmup_steps,
    )

    return TrainerRuntimeState(
        trainer=trainer,
        training_args=training_args,
        batch_plan=batch_plan,
        gradient_accum_steps=gradient_accum_steps,
        global_bs_imgs=global_bs_imgs,
        global_bs_views=global_bs_views,
        world_size=world_size,
        save_every_epochs=save_every_epochs,
    )


def _build_simclr_manifest(cfg: TrainingRunConfig) -> dict[str, object] | None:
    if cfg.loss_mode == LossMode.SIMCLR:
        return {"impl": "simclr"}
    if cfg.loss_mode == LossMode.SIMCLR_DEBIASED:
        tau_plus = 0.10 if cfg.simclr_debias_tau_plus is None else float(cfg.simclr_debias_tau_plus)
        return {"impl": "dcl", "tau_plus": float(tau_plus)}
    if cfg.loss_mode == LossMode.SOFT_HCL:
        return {"impl": "soft_hcl", "alpha": float(cfg.soft_hcl_alpha), "tau_h": float(cfg.soft_hcl_tau_h)}
    if cfg.loss_mode == LossMode.HCL:
        tau_plus = 0.0 if cfg.hcl_tau_plus is None else float(cfg.hcl_tau_plus)
        return {"impl": "hcl", "beta": float(cfg.hcl_beta), "tau_plus": float(tau_plus)}
    return None


def _write_run_manifest(
    cfg: TrainingRunConfig,
    profile_state: ProfileRuntimeState,
    runtime_bundle: RuntimeBundle,
    trainer_state: TrainerRuntimeState,
) -> None:
    try:
        import json

        if cfg.branch == Branch.DINOV2:
            arch_tok = short_arch_name(str(cfg.model_name_or_path).split("/")[-1])
        else:
            arch_tok = derive_arch_tok_from_config(runtime_bundle.config, cfg.model_name_or_path)
        proj_tok = proj_token(
            cfg.loss_mode,
            projector_hidden_dim=cfg.projector_hidden_dim,
            projector_out_dim=cfg.projector_out_dim,
        )
        is_contrastive = is_contrastive_loss(cfg.loss_mode)
        mask_tok = 0.0 if is_contrastive else float(cfg.mask_ratio)
        phase_tag = phase_tag_from_output_dir(profile_state.output_dir)
        binary_dataset = _find_binary_dataset(runtime_bundle.train_dataset)
        dataset_manifest = cfg.dataset_config.manifest_fields()
        if binary_dataset is not None:
            dataset_manifest["dataset_configured_block_size"] = int(cfg.dataset_config.block_size)
            dataset_manifest["dataset_block_size"] = int(getattr(binary_dataset, "block_size", cfg.dataset_config.block_size))
            dataset_manifest.update(_binary_mega_chunk_summary(binary_dataset, cfg.num_train_epochs))

        try:
            ta_json = trainer_state.training_args.to_json_string()
        except Exception:
            ta_json = None

        manifest = {
            "training_args_json": ta_json,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "init": {
                "mode": runtime_bundle.init_tok,
                "mae_model_name_or_path": (
                    cfg.model_name_or_path
                    if (
                        cfg.branch == Branch.MAE
                        and cfg.profile == SSLProfile.CONTINUED
                        and cfg.init_from_checkpoint is None
                    )
                    else None
                ),
                "dinov2_model_name_or_path": (
                    cfg.model_name_or_path
                    if (cfg.branch == Branch.DINOV2 and cfg.init_from_checkpoint is None)
                    else None
                ),
                "checkpoint_dir": (cfg.init_from_checkpoint if cfg.init_from_checkpoint is not None else None),
            },
            "paths": {
                "run_root": profile_state.output_dir,
                "results_dir": trainer_state.training_args.output_dir,
                "logs_dir": os.path.join(profile_state.output_dir, "logs"),
                "checkpoints_dir": os.path.join(profile_state.output_dir, "checkpoints"),
            },
            "tokens": {
                "arch": arch_tok,
                "branch": cfg.branch.value,
                "init": runtime_bundle.init_tok,
                "training_mode": (str(cfg.mode_name) if cfg.mode_name is not None else None),
                "stage": (str(cfg.stage_name) if cfg.stage_name is not None else None),
                "prephase": (str(cfg.stage_tokens.get("prephase")) if cfg.stage_tokens.get("prephase") is not None else None),
                "loss": cfg.loss_mode.value.lower(),
                "proj": proj_tok,
                "views": int(cfg.n_views if is_contrastive else 1),
                "img_size": int(getattr(runtime_bundle.config, "image_size", 224) or 224),
                "patch": int(cfg.patch_size),
                "mask": float(mask_tok),
                "regs": int(cfg.n_registers),
                **dataset_manifest,
                "dino": bool(getattr(runtime_bundle.config, "use_dino_encoder", False)),
                "drop_path": float(getattr(runtime_bundle.config, "drop_path_rate", 0.0) or 0.0),
                **cfg.data_policy.manifest_fields(),
                "run_tag": (str(cfg.run_tag) if cfg.run_tag else None),
            },
            "training": {
                "num_train_epochs": int(cfg.num_train_epochs),
                "configured_full_file_epochs": int(cfg.num_train_epochs),
                "internal_trainer_epochs": int(trainer_state.training_args.num_train_epochs),
                "epoch_semantics": str(getattr(trainer_state.training_args, "epoch_semantics", "full_dataset")),
                "mega_chunk_count": int(getattr(trainer_state.training_args, "mega_chunk_count", 1)),
                "epochs_per_mega_chunk": int(getattr(trainer_state.training_args, "epochs_per_mega_chunk", cfg.num_train_epochs)),
                "learning_rate": float(profile_state.learning_rate),
                "warmup_ratio": float(profile_state.warmup_ratio),
                "warmup_steps": int(trainer_state.batch_plan.warmup_steps),
                "warmup_epochs_effective": int(trainer_state.batch_plan.warmup_epochs_effective),
                "contrastive_diagnostics_enabled": bool(cfg.contrastive_diagnostics_enabled),
                "heavy_contrastive_diagnostics": bool(cfg.heavy_contrastive_diagnostics),
            },
            "optimizer": {
                "name": trainer_state.training_args.optim,
                "learning_rate": float(profile_state.learning_rate),
                "weight_decay": float(profile_state.weight_decay),
                "lr_scheduler": str(trainer_state.training_args.lr_scheduler_type),
                "betas": [float(profile_state.adam_betas[0]), float(profile_state.adam_betas[1])],
                "warmup": {
                    "ratio": float(profile_state.warmup_ratio),
                    "steps": int(trainer_state.batch_plan.warmup_steps),
                },
                "temperature": {
                    "warm": float(profile_state.temp_warm),
                    "main": float(profile_state.temp_main),
                },
            },
            "data": {
                **dataset_manifest,
                "apply_norm": bool(profile_state.apply_norm),
            },
            "data_policy": cfg.data_policy.manifest_fields(),
            "execution": {
                "mode": (str(cfg.mode_name) if cfg.mode_name is not None else None),
                "stage": (str(cfg.stage_name) if cfg.stage_name is not None else None),
                "stage_tokens": dict(cfg.stage_tokens),
                "mode_options": dict(cfg.mode_options),
                "checkpoint_policy": str(cfg.checkpoint_policy_name),
                "export_policy": str(cfg.export_policy_name),
            },
            "batching": {
                "per_device_train_batch_size": int(cfg.per_device_train_batch_size),
                "world_size": int(trainer_state.world_size),
                "gradient_accumulation_steps": int(trainer_state.gradient_accum_steps),
                "global_batch_images": int(trainer_state.global_bs_imgs),
                "global_batch_views": int(trainer_state.global_bs_views),
                "shard_size": int(trainer_state.batch_plan.shard_size),
                "microsteps_per_epoch": int(trainer_state.batch_plan.microsteps_per_epoch),
                "steps_per_epoch_updates": int(trainer_state.batch_plan.steps_per_epoch_updates),
                "save_steps": int(trainer_state.batch_plan.save_steps),
                "dataloader_num_workers_per_rank": int(trainer_state.training_args.dataloader_num_workers),
                "dataloader_prefetch_factor": int(trainer_state.training_args.dataloader_prefetch_factor or 0),
            },
            "seed": int(cfg.seed),
        }
        simclr_cfg_manifest = _build_simclr_manifest(cfg)
        if simclr_cfg_manifest is not None:
            manifest["optimizer"]["simclr"] = simclr_cfg_manifest
        if phase_tag is not None:
            manifest["phase"] = phase_tag

        os.makedirs(profile_state.output_dir, exist_ok=True)
        with open(os.path.join(profile_state.output_dir, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        try:
            with open(os.path.join(trainer_state.training_args.output_dir, "run_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
        guarded_print("[Manifest] wrote run_manifest.json")
    except Exception as exc:
        guarded_print(f"[Manifest] failed to write: {exc}")


def _attach_trainer_callbacks(
    cfg: TrainingRunConfig,
    profile_state: ProfileRuntimeState,
    runtime_bundle: RuntimeBundle,
    trainer_state: TrainerRuntimeState,
) -> None:
    trainer = trainer_state.trainer

    if is_contrastive_loss(cfg.loss_mode):
        try:
            trainer.temperature = float(profile_state.temp_warm)
            guarded_print(f"[TempSchedule] init -> temperature={trainer.temperature}")
            trainer.add_callback(
                TemperatureScheduleCallback(
                    trainer,
                    warmup_epochs=trainer_state.batch_plan.warmup_epochs_effective,
                    temp_warm=profile_state.temp_warm,
                    temp_main=profile_state.temp_main,
                )
            )
            guarded_print(
                f"[TempSchedule] added (warmup_epochs={trainer_state.batch_plan.warmup_epochs_effective}, "
                f"temp={profile_state.temp_warm}->{profile_state.temp_main})"
            )
        except Exception as exc:
            guarded_print(f"[TempSchedule] failed to add: {exc}")
    guarded_print(f"[Suffix] profile={cfg.profile.value}")

    try:
        trainer.add_callback(ProcessorSaveCallback(runtime_bundle.processor))
    except Exception as exc:
        guarded_print(f"[ProcessorSave] Failed to attach callback: {exc}")

    if is_contrastive_loss(cfg.loss_mode):
        guarded_print(
            f"[EmbedRep] {cfg.loss_mode.value}: using backbone pooler_output when available; "
            f"fallback = {'CLS' if getattr(trainer, '_use_cls_token', False) else 'pooled patch tokens'}."
        )
    else:
        guarded_print(
            f"[EmbedRep] {cfg.loss_mode.value}: using "
            f"{'CLS' if getattr(trainer, '_use_cls_token', False) else 'pooled patch tokens'} for encoder features."
        )

    trainer._prepare_inputs = _DevicePrefetch(profile_state.device)

    try:
        trainer.add_callback(MetricsPlotCallback(out_dir=profile_state.output_dir))
    except Exception as exc:
        guarded_print(f"[MetricsPlot] Failed to attach callback: {exc}")

    if cfg.viz_enabled and cfg.loss_mode in (LossMode.L2, LossMode.L2_L1, LossMode.L2_BL1):
        group_images = 6
        trainer.add_callback(
            ReconVizCallback(
                dataset=runtime_bundle.train_dataset,
                processor=runtime_bundle.processor,
                out_dir=os.path.join(profile_state.output_dir, "recon_viz"),
                num_samples=group_images * 6,
                group_size=group_images,
                err_gain=1.0,
                vary_samples=False,
            )
        )


def _find_latest_checkpoint_dir(root_dir: str, max_step_exclusive: int | None = None) -> str | None:
    try:
        entries = os.listdir(root_dir)
    except Exception:
        return None
    best_step = None
    best_path = None
    for name in entries:
        if not name.startswith("checkpoint-"):
            continue
        step_str = name.split("checkpoint-")[-1]
        try:
            step = int(step_str)
        except Exception:
            continue
        if max_step_exclusive is not None and step >= int(max_step_exclusive):
            continue
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


def _run_train_resume_and_export(
    cfg: TrainingRunConfig,
    profile_state: ProfileRuntimeState,
    runtime_bundle: RuntimeBundle,
    trainer_state: TrainerRuntimeState,
) -> None:
    if str(cfg.checkpoint_policy_name) != "standard":
        raise NotImplementedError(
            f"Unsupported checkpoint_policy_name={cfg.checkpoint_policy_name!r} for active runner."
        )
    trainer = trainer_state.trainer
    training_args = trainer_state.training_args
    steps_per_epoch_updates = trainer_state.batch_plan.steps_per_epoch_updates
    max_steps = int(steps_per_epoch_updates) * int(cfg.num_train_epochs)
    last_ckpt = _find_latest_checkpoint_dir(training_args.output_dir, max_step_exclusive=max_steps)

    if last_ckpt is not None:
        try:
            abs_results_dir = os.path.abspath(training_args.output_dir)
            ckpts = []
            for name in os.listdir(training_args.output_dir):
                if name.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, name)):
                    ckpts.append(name)
            ckpts_sorted = sorted(ckpts, key=lambda s: int(s.split("checkpoint-")[-1]) if s.split("checkpoint-")[-1].isdigit() else -1)
            guarded_print(f"[Resume] results_dir={abs_results_dir} | checkpoints_found={ckpts_sorted[-5:]}")
        except Exception:
            pass
        try:
            import numpy as _np
            import torch.serialization as _ts
            _ts.add_safe_globals([
                _np._core.multiarray._reconstruct,
                _np.ndarray,
                _np.dtype,
            ])
        except Exception:
            pass
        try:
            step = int(os.path.basename(last_ckpt).split("checkpoint-")[-1])
            epoch = float(step) / float(max(1, steps_per_epoch_updates))
            guarded_print(
                f"[Resume] Resuming from {last_ckpt} (step={step}, approx_epoch={epoch:.2f}/{float(cfg.num_train_epochs):.2f}, "
                f"steps_per_epoch={steps_per_epoch_updates})"
            )
        except Exception:
            guarded_print(f"[Resume] Resuming from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        guarded_print("[Resume] No valid checkpoint found; starting fresh")
        trainer.train()

    if str(cfg.export_policy_name) != "standard":
        raise NotImplementedError(
            f"Unsupported export_policy_name={cfg.export_policy_name!r} for active runner."
        )
    ckpt_dir = os.path.join(profile_state.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save_model(ckpt_dir)
    runtime_bundle.processor.save_pretrained(ckpt_dir)
    try:
        runtime_bundle.config.save_pretrained(profile_state.output_dir)
    except Exception:
        pass
    cleanup_after_run(logger=guarded_print, scoped_objects=[trainer])


def run_ssl_training(run_config: TrainingRunConfig):
    cfg = run_config
    prepare_run_environment(cfg.hf_cache_path, cfg.seed)
    profile_state = _resolve_profile_runtime(cfg)
    runtime_bundle = _build_runtime_bundle(cfg, profile_state)
    trainer_state = _build_trainer_and_schedule(cfg, profile_state, runtime_bundle)
    _write_run_manifest(cfg, profile_state, runtime_bundle, trainer_state)
    _attach_trainer_callbacks(cfg, profile_state, runtime_bundle, trainer_state)
    _run_train_resume_and_export(cfg, profile_state, runtime_bundle, trainer_state)


# Execution
if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    DATASET_BIN_PATHS = {
        BinaryDatasetMode.DEFAULT: "/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_v5.bin",
        BinaryDatasetMode.CURATED: "/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_curated.bin",
    }
    HF_CACHE_PATH = dataset_hf_cache_root(DATASET_BIN_PATHS[BinaryDatasetMode.DEFAULT])
    # --- Branch selection ---
    # Branch.MAE:
    #   ViT-MAE family runs (MAE-only, contrastive-only from HF MAE init, or MAE->contrastive).
    # Branch.DINOV2:
    #   Direct contrastive continued-pretraining from a HF DINOv2-family checkpoint.
    BRANCH = Branch.MAE

    # Shared objective sets.
    # MAE_LOSS_MODES_TO_RUN = [LossMode.L2, LossMode.L2_L1, LossMode.L2_BL1]
    MAE_LOSS_MODES_TO_RUN = [LossMode.L2_L1]
    # CONTRASTIVE_LOSS_MODES_TO_RUN = [LossMode.HCL, LossMode.SOFT_HCL, LossMode.SIMCLR]
    CONTRASTIVE_LOSS_MODES_TO_RUN = [LossMode.SOFT_HCL]
    DATASET_CONFIG = BinaryDatasetConfig(
        mode=BinaryDatasetMode.DEFAULT,
        bin_path_override=DATASET_BIN_PATHS[BinaryDatasetMode.DEFAULT],
        label="default",
    )
    DATA_POLICY = DataPolicy(
        use_roi_crop=True,
        enable_augmentations=True,
    )

    BRANCH_CONFIGS: dict[Branch, BranchRunConfig] = {
        Branch.MAE: BranchRunConfig(
            branch=Branch.MAE,
            model_name="facebook/vit-mae-base",
            train_profile=SSLProfile.CONTINUED,
            init_mode=InitMode.FACEBOOK_HF,
            mae_learning_rate=5e-4,
            contrastive_learning_rate=5e-4,
            training_modes=[
                # TrainingMode.MAE_ONLY,
                # TrainingMode.CONTRASTIVE_ONLY,
                TrainingMode.MAE_THEN_CONTRASTIVE,
            ],
            mae_only_loss_modes=list(MAE_LOSS_MODES_TO_RUN),
            contrastive_loss_modes=list(CONTRASTIVE_LOSS_MODES_TO_RUN),
            patch_sizes=[16],
            masks=[0.75],
            n_registers_list=[4], # [0, 4]
            dataset_config=DATASET_CONFIG,
            data_policy=DATA_POLICY,
            use_dino_encoder=False,
            drop_path_rate=0.1,
        ),
        Branch.DINOV2: BranchRunConfig(
            branch=Branch.DINOV2,
            # Change this to any HF DINOv2-family repo; register-token metadata is inferred.
            # model_name="facebook/dinov2-with-registers-base",
            model_name="facebook/dinov2-base",
            train_profile=SSLProfile.CONTINUED,
            init_mode=InitMode.DINOV2_HF,
            mae_learning_rate=5e-4,
            contrastive_learning_rate=1e-4,
            training_modes=[TrainingMode.CONTRASTIVE_ONLY],
            mae_only_loss_modes=[],
            contrastive_loss_modes=list(CONTRASTIVE_LOSS_MODES_TO_RUN),
            dataset_config=DATASET_CONFIG,
            data_policy=DATA_POLICY,
        ),
    }
    branch_cfg = BRANCH_CONFIGS[BRANCH]

    if branch_cfg.branch == Branch.DINOV2:
        invalid_modes = [mode.value for mode in branch_cfg.training_modes if mode != TrainingMode.CONTRASTIVE_ONLY]
        if invalid_modes:
            raise ValueError(f"Branch.DINOV2 only supports contrastive-only runs; got {invalid_modes}.")
        if branch_cfg.init_mode != InitMode.DINOV2_HF:
            raise ValueError("Branch.DINOV2 must use init_mode=InitMode.DINOV2_HF.")
    elif branch_cfg.init_mode == InitMode.DINOV2_HF:
        raise ValueError("InitMode.DINOV2_HF is only valid for Branch.DINOV2.")
    elif branch_cfg.init_mode == InitMode.MAE_PHASE1:
        raise ValueError("InitMode.MAE_PHASE1 is only used internally for phase-2 checkpoint naming, not as a branch init.")

    # One-time augmentation preview at the very start (per run). 0 disables it.
    PREVIEW_AUGS_SAMPLES = 128  # e.g., 128 rows showing: original + CONTRASTIVE_VIEWS, 0 to disable
    PREVIEW_AUGS_REPEATS = 10  # draw the augmentation pipeline this many times per image in the preview
    CONTRASTIVE_VIEWS = 2  # number of SimCLR views per image
    PROJECTOR_HIDDEN_DIM = 2048
    PROJECTOR_OUT_DIM = 128 # was 128 in all previous tests, now opening up to 256 (no real change going to 256)
    MAE_BRANCH_BS_MAP = {
        LossMode.L2: (256, 256),
        LossMode.L2_L1: (256, 256),
        LossMode.L2_BL1: (256, 256),
        LossMode.SIMCLR: (64 if CONTRASTIVE_VIEWS == 2 else 48,
                          64 if CONTRASTIVE_VIEWS == 2 else 48),
        LossMode.SOFT_HCL: (64 if CONTRASTIVE_VIEWS == 2 else 48,
                            64 if CONTRASTIVE_VIEWS == 2 else 48),
        LossMode.HCL: (64 if CONTRASTIVE_VIEWS == 2 else 48,
                       64 if CONTRASTIVE_VIEWS == 2 else 48),
        LossMode.SIMCLR_DEBIASED: (64 if CONTRASTIVE_VIEWS == 2 else 48,
                                   64 if CONTRASTIVE_VIEWS == 2 else 48),
    }
    DINOV2_BRANCH_BS_MAP = {
        # Native DINO contrastive continued-pretraining is typically a bit heavier per sample.
        LossMode.SIMCLR: (20 if CONTRASTIVE_VIEWS == 2 else 16,
                          20 if CONTRASTIVE_VIEWS == 2 else 16),
        LossMode.SOFT_HCL: (20 if CONTRASTIVE_VIEWS == 2 else 16,
                            20 if CONTRASTIVE_VIEWS == 2 else 16),
        LossMode.HCL: (20 if CONTRASTIVE_VIEWS == 2 else 16,
                       20 if CONTRASTIVE_VIEWS == 2 else 16),
        LossMode.SIMCLR_DEBIASED: (20 if CONTRASTIVE_VIEWS == 2 else 16,
                                   20 if CONTRASTIVE_VIEWS == 2 else 16),
    }
    # --- Run schedule ---
    # Default schedule matches previous runs (35/35). Enable LONGRUN to train longer (e.g., 50 or 100 epochs).
    LONGRUN = False
    LONGRUN_EPOCHS_PHASE1 = 50
    LONGRUN_EPOCHS_PHASE2 = 125
    LONGRUN_WARMUP_RATIO_PHASE1 = 0.10
    LONGRUN_WARMUP_RATIO_PHASE2 = 0.05

    if LONGRUN:
        RUN_TAG = "longrun"
        NUM_TRAIN_EPOCHS_MAE = int(LONGRUN_EPOCHS_PHASE1)
        NUM_TRAIN_EPOCHS_CONTRASTIVE = int(LONGRUN_EPOCHS_PHASE2)
        WARMUP_RATIO_MAE = float(LONGRUN_WARMUP_RATIO_PHASE1)
        WARMUP_RATIO_CONTRASTIVE = float(LONGRUN_WARMUP_RATIO_PHASE2)
    else:
        RUN_TAG = None
        NUM_TRAIN_EPOCHS_MAE = 35
        NUM_TRAIN_EPOCHS_CONTRASTIVE = 35
        WARMUP_RATIO_MAE = 0.15
        WARMUP_RATIO_CONTRASTIVE = 0.15

    LEARNING_RATE_MAE = float(branch_cfg.mae_learning_rate)
    LEARNING_RATE_CONTRASTIVE = float(branch_cfg.contrastive_learning_rate)

    # Contrastive loss knobs:
    # - LossMode.SIMCLR: vanilla SimCLR / InfoNCE (uniform negatives).
    # - LossMode.SIMCLR_DEBIASED: DCL-style debiasing (tau_plus).
    # - LossMode.SOFT_HCL: your legacy "soft" hard-negative mix (alpha, tau_h).
    # - LossMode.HCL: Robinson et al. hard-negative objective (beta, optional tau_plus).
    #
    SIMCLR_DEBIASED_TAU_PLUS = 0.10
    SOFT_HCL_ALPHA = 0.5
    SOFT_HCL_TAU_H = 0.15
    HCL_BETA = 1.0
    HCL_TAU_PLUS = None  # None -> no debias correction in HCL
    CONTRASTIVE_DIAGNOSTICS_ENABLED = True
    HEAVY_CONTRASTIVE_DIAGNOSTICS = True

    loss_weights = {
        "l2": 1.0,
        "l1": 0.1,
        "bl1": 0.1,
    }

    TWO_PHASE_MAE_LOSS_MODES = list(branch_cfg.mae_only_loss_modes)
    effective_patch_sizes = branch_cfg.patch_sizes
    effective_masks = branch_cfg.masks
    if branch_cfg.branch == Branch.MAE:
        branch_bs_map = MAE_BRANCH_BS_MAP
    elif branch_cfg.branch == Branch.DINOV2:
        branch_bs_map = DINOV2_BRANCH_BS_MAP
    else:
        branch_bs_map = {}

    def _checkpoint_exists(ckpt_dir: str) -> bool:
        return (
            os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin"))
            or os.path.exists(os.path.join(ckpt_dir, "model.safetensors"))
        )

    if branch_cfg.init_mode == InitMode.FACEBOOK_HF:
        base_init_tok = InitMode.FACEBOOK_HF.value
    elif branch_cfg.init_mode == InitMode.DINOV2_HF:
        base_init_tok = InitMode.DINOV2_HF.value
    elif branch_cfg.init_mode == InitMode.SCRATCH_HF:
        base_init_tok = InitMode.SCRATCH_HF.value
    else:
        raise ValueError(f"Unsupported branch init mode for direct stages: {branch_cfg.init_mode}")

    mae_use_dino_encoder = bool(branch_cfg.use_dino_encoder)
    effective_n_registers_list = branch_cfg.n_registers_list
    if branch_cfg.branch == Branch.DINOV2:
        probe_cfg = load_dinov2_config(
            model_name_or_path=branch_cfg.model_name,
        )
        inferred_patch_size = int(getattr(probe_cfg, "patch_size", 0) or 0)
        if inferred_patch_size <= 0:
            raise ValueError(
                f"Branch.DINOV2 could not infer a valid patch_size from '{branch_cfg.model_name}'."
            )
        effective_patch_sizes = [inferred_patch_size]
        effective_masks = [0.0]
        inferred_dino_registers = max(0, int(infer_num_prefix_tokens(probe_cfg)) - 1)
        effective_n_registers_list = [inferred_dino_registers]
        guarded_print(
            f"[BranchConfig] DINO branch inferred patch_size={inferred_patch_size}, "
            f"n_registers={inferred_dino_registers} from '{branch_cfg.model_name}'."
        )
    else:
        if not effective_patch_sizes or not effective_masks:
            raise ValueError(f"Branch.MAE requires explicit patch_sizes and masks; got {branch_cfg}.")

    sweep = RunSweep(
        model_name=branch_cfg.model_name,
        profile=branch_cfg.train_profile,
        training_modes=branch_cfg.training_modes,
        mae_only_loss_modes=branch_cfg.mae_only_loss_modes,
        contrastive_only_loss_modes=branch_cfg.contrastive_loss_modes,
        two_phase_mae_loss_modes=TWO_PHASE_MAE_LOSS_MODES,
        two_phase_contrastive_loss_modes=branch_cfg.contrastive_loss_modes,
        patch_sizes=effective_patch_sizes,
        masks=effective_masks,
        n_views=CONTRASTIVE_VIEWS,
        projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
        projector_out_dim=PROJECTOR_OUT_DIM,
        preview_samples=PREVIEW_AUGS_SAMPLES,
        preview_repeats=PREVIEW_AUGS_REPEATS,
        n_registers_list=effective_n_registers_list,
        dataset_config=branch_cfg.dataset_config,
        data_policy=branch_cfg.data_policy,
    )

    planner_config = ModePlannerConfig(
        branch=branch_cfg.branch,
        hf_cache_path=HF_CACHE_PATH,
        run_tag=RUN_TAG,
        dataloader_num_workers=_env_int("STRADA_DATALOADER_NUM_WORKERS", 8),
        seed=42,
        loss_weights=loss_weights,
        use_dino_encoder=mae_use_dino_encoder,
        drop_path_rate=branch_cfg.drop_path_rate,
        base_init_token=base_init_tok,
        learning_rate_mae=LEARNING_RATE_MAE,
        learning_rate_contrastive=LEARNING_RATE_CONTRASTIVE,
        warmup_ratio_mae=WARMUP_RATIO_MAE,
        warmup_ratio_contrastive=WARMUP_RATIO_CONTRASTIVE,
        num_train_epochs_mae=NUM_TRAIN_EPOCHS_MAE,
        num_train_epochs_contrastive=NUM_TRAIN_EPOCHS_CONTRASTIVE,
        branch_bs_map=branch_bs_map,
        soft_hcl_alpha=SOFT_HCL_ALPHA,
        soft_hcl_tau_h=SOFT_HCL_TAU_H,
        simclr_debias_tau_plus=SIMCLR_DEBIASED_TAU_PLUS,
        hcl_beta=HCL_BETA,
        hcl_tau_plus=HCL_TAU_PLUS,
        contrastive_diagnostics_enabled=CONTRASTIVE_DIAGNOSTICS_ENABLED,
        heavy_contrastive_diagnostics=HEAVY_CONTRASTIVE_DIAGNOSTICS,
    )

    for spec in sweep.iter_specs():
        try:
            planned_runs = plan_runs_for_spec(spec, planner_config)
            for planned_run in planned_runs:
                if planned_run.skip_if_checkpoint_exists and _checkpoint_exists(planned_run.checkpoint_dir):
                    if planned_run.skip_message:
                        guarded_print(planned_run.skip_message)
                    else:
                        guarded_print(
                            f"[{planned_run.log_prefix}] Skipping run "
                            f"(existing checkpoint found at {planned_run.checkpoint_dir})"
                        )
                    continue
                if planned_run.cleanup_before_run:
                    cleanup_after_run(logger=guarded_print, scoped_objects=[])
                guarded_print(planned_run.start_message)
                run_ssl_training(planned_run.run_config)

            cleanup_after_run(logger=guarded_print, scoped_objects=[])

        except Exception:
            import traceback
            traceback.print_exc()
            fatal_error_file = os.getenv("STRADA_FATAL_ERROR_FILE") or "fatal_error.log"
            with open(fatal_error_file, "a") as f:
                f.write(traceback.format_exc())
            cleanup_after_run(logger=guarded_print, scoped_objects=[])
            raise
