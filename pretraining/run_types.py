import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum

from pretraining.data_pipeline import BinaryDatasetConfig, DataPolicy


class LossMode(Enum):
    L2 = "l2"
    L2_L1 = "l2_l1"
    L2_BL1 = "l2_bl1"
    SIMCLR = "simclr"
    SOFT_HCL = "soft_hcl"
    HCL = "hcl"
    SIMCLR_DEBIASED = "simclr_debiased"


class SSLProfile(Enum):
    CONTINUED = "continued"
    SCRATCH = "scratch"


class TrainingMode(Enum):
    MAE_ONLY = "mae_only"
    CONTRASTIVE_ONLY = "contrastive_only"
    MAE_THEN_CONTRASTIVE = "mae_then_contrastive"


class Branch(Enum):
    MAE = "mae"
    DINOV2 = "dinov2"


class InitMode(Enum):
    FACEBOOK_HF = "facebook-hf"
    DINOV2_HF = "dinov2-hf"
    MAE_PHASE1 = "mae-phase1"
    SCRATCH_HF = "scratch-hf"


RECONSTRUCTION_LOSS_MODES = (LossMode.L2, LossMode.L2_L1, LossMode.L2_BL1)
CONTRASTIVE_LOSS_MODES = (
    LossMode.SIMCLR,
    LossMode.SOFT_HCL,
    LossMode.HCL,
    LossMode.SIMCLR_DEBIASED,
)


def is_reconstruction_loss(loss_mode: LossMode) -> bool:
    return loss_mode in RECONSTRUCTION_LOSS_MODES


def is_contrastive_loss(loss_mode: LossMode) -> bool:
    return loss_mode in CONTRASTIVE_LOSS_MODES


@dataclass(frozen=True)
class BranchRunConfig:
    branch: Branch
    model_name: str
    train_profile: SSLProfile
    init_mode: InitMode
    mae_learning_rate: float
    contrastive_learning_rate: float
    training_modes: tuple[TrainingMode, ...]
    mae_only_loss_modes: tuple[LossMode, ...]
    contrastive_loss_modes: tuple[LossMode, ...]
    patch_sizes: tuple[int, ...] | None = None
    masks: tuple[float, ...] | None = None
    n_registers_list: tuple[int, ...] | None = None
    dataset_config: BinaryDatasetConfig = field(default_factory=BinaryDatasetConfig)
    data_policy: DataPolicy = field(default_factory=DataPolicy)
    use_dino_encoder: bool = False
    drop_path_rate: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_modes", tuple(self.training_modes))
        object.__setattr__(self, "mae_only_loss_modes", tuple(self.mae_only_loss_modes))
        object.__setattr__(self, "contrastive_loss_modes", tuple(self.contrastive_loss_modes))
        if self.patch_sizes is not None:
            object.__setattr__(self, "patch_sizes", tuple(self.patch_sizes))
        if self.masks is not None:
            object.__setattr__(self, "masks", tuple(self.masks))
        if self.n_registers_list is not None:
            object.__setattr__(self, "n_registers_list", tuple(self.n_registers_list))


@dataclass(frozen=True)
class RunSpec:
    training_mode: TrainingMode
    loss_modes: tuple[LossMode, ...]
    patch_size: int
    mask: float
    n_views: int
    profile: SSLProfile
    model_name: str
    projector_hidden_dim: int
    projector_out_dim: int
    preview_samples: int
    preview_repeats: int
    n_registers: int
    dataset_config: BinaryDatasetConfig
    data_policy: DataPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss_modes", tuple(self.loss_modes))
        if self.training_mode in (TrainingMode.MAE_ONLY, TrainingMode.CONTRASTIVE_ONLY):
            if len(self.loss_modes) != 1:
                raise ValueError(f"{self.training_mode.value} requires exactly one loss mode.")
        elif self.training_mode == TrainingMode.MAE_THEN_CONTRASTIVE:
            if len(self.loss_modes) != 2:
                raise ValueError("mae_then_contrastive requires exactly two loss modes.")


@dataclass(frozen=True)
class RunSweep:
    model_name: str
    profile: SSLProfile
    training_modes: tuple[TrainingMode, ...]
    mae_only_loss_modes: tuple[LossMode, ...]
    contrastive_only_loss_modes: tuple[LossMode, ...]
    two_phase_mae_loss_modes: tuple[LossMode, ...]
    two_phase_contrastive_loss_modes: tuple[LossMode, ...]
    patch_sizes: tuple[int, ...]
    masks: tuple[float, ...]
    n_views: int = 2
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 128
    preview_samples: int = 128
    preview_repeats: int = 10
    dataset_config: BinaryDatasetConfig = field(default_factory=BinaryDatasetConfig)
    data_policy: DataPolicy = field(default_factory=DataPolicy)
    n_registers: int = 0
    n_registers_list: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_modes", tuple(self.training_modes))
        object.__setattr__(self, "mae_only_loss_modes", tuple(self.mae_only_loss_modes))
        object.__setattr__(self, "contrastive_only_loss_modes", tuple(self.contrastive_only_loss_modes))
        object.__setattr__(self, "two_phase_mae_loss_modes", tuple(self.two_phase_mae_loss_modes))
        object.__setattr__(self, "two_phase_contrastive_loss_modes", tuple(self.two_phase_contrastive_loss_modes))
        object.__setattr__(self, "patch_sizes", tuple(self.patch_sizes))
        object.__setattr__(self, "masks", tuple(self.masks))
        if self.n_registers_list is not None:
            object.__setattr__(self, "n_registers_list", tuple(self.n_registers_list))

    def iter_specs(self) -> Iterator[RunSpec]:
        regs_list = self.n_registers_list if self.n_registers_list is not None else [self.n_registers]
        common = {
            "n_views": int(self.n_views),
            "profile": self.profile,
            "model_name": self.model_name,
            "projector_hidden_dim": int(self.projector_hidden_dim),
            "projector_out_dim": int(self.projector_out_dim),
            "preview_samples": self.preview_samples,
            "preview_repeats": self.preview_repeats,
            "dataset_config": self.dataset_config,
            "data_policy": self.data_policy,
        }
        for training_mode in self.training_modes:
            if training_mode == TrainingMode.MAE_ONLY:
                for loss_mode, patch_size, mask, n_registers in itertools.product(
                    self.mae_only_loss_modes, self.patch_sizes, self.masks, regs_list
                ):
                    if not is_reconstruction_loss(loss_mode):
                        raise ValueError(f"MAE_ONLY requires reconstruction losses; got {loss_mode.value}")
                    yield RunSpec(
                        training_mode=training_mode,
                        loss_modes=(loss_mode,),
                        patch_size=int(patch_size),
                        mask=float(mask),
                        n_registers=int(n_registers),
                        **common,
                    )
            elif training_mode == TrainingMode.CONTRASTIVE_ONLY:
                for loss_mode, patch_size, mask, n_registers in itertools.product(
                    self.contrastive_only_loss_modes, self.patch_sizes, self.masks, regs_list
                ):
                    if not is_contrastive_loss(loss_mode):
                        raise ValueError(f"CONTRASTIVE_ONLY requires contrastive losses; got {loss_mode.value}")
                    yield RunSpec(
                        training_mode=training_mode,
                        loss_modes=(loss_mode,),
                        patch_size=int(patch_size),
                        mask=float(mask),
                        n_registers=int(n_registers),
                        **common,
                    )
            elif training_mode == TrainingMode.MAE_THEN_CONTRASTIVE:
                for mae_loss, contrastive_loss, patch_size, mask, n_registers in itertools.product(
                    self.two_phase_mae_loss_modes,
                    self.two_phase_contrastive_loss_modes,
                    self.patch_sizes,
                    self.masks,
                    regs_list,
                ):
                    if not is_reconstruction_loss(mae_loss):
                        raise ValueError(
                            f"MAE_THEN_CONTRASTIVE phase1 requires reconstruction losses; got {mae_loss.value}"
                        )
                    if not is_contrastive_loss(contrastive_loss):
                        raise ValueError(
                            f"MAE_THEN_CONTRASTIVE phase2 requires contrastive losses; got {contrastive_loss.value}"
                        )
                    yield RunSpec(
                        training_mode=training_mode,
                        loss_modes=(mae_loss, contrastive_loss),
                        patch_size=int(patch_size),
                        mask=float(mask),
                        n_registers=int(n_registers),
                        **common,
                    )
            else:
                raise ValueError(f"Unsupported training_mode={training_mode}")


@dataclass(frozen=True)
class ProfileSettings:
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temp_warm: float
    temp_main: float
    adam_betas: tuple[float, float] = (0.9, 0.95)

    def with_overrides(
        self,
        *,
        learning_rate: float | None = None,
        warmup_ratio: float | None = None,
    ) -> "ProfileSettings":
        return ProfileSettings(
            learning_rate=self.learning_rate if learning_rate is None else float(learning_rate),
            weight_decay=float(self.weight_decay),
            warmup_ratio=self.warmup_ratio if warmup_ratio is None else float(warmup_ratio),
            temp_warm=float(self.temp_warm),
            temp_main=float(self.temp_main),
            adam_betas=tuple(self.adam_betas),
        )


PROFILE_SETTINGS: Mapping[SSLProfile, ProfileSettings] = {
    SSLProfile.CONTINUED: ProfileSettings(
        learning_rate=5e-4,
        weight_decay=0.05,
        warmup_ratio=0.15,
        temp_warm=0.20,
        temp_main=0.10,
        adam_betas=(0.9, 0.95),
    ),
    SSLProfile.SCRATCH: ProfileSettings(
        learning_rate=5e-4,
        weight_decay=0.05,
        warmup_ratio=0.15,
        temp_warm=0.20,
        temp_main=0.10,
        adam_betas=(0.9, 0.95),
    ),
}


@dataclass(frozen=True)
class BatchSchedulePlan:
    gradient_accum_steps: int
    global_bs_imgs: int
    global_bs_views: int
    microsteps_per_epoch: int
    steps_per_epoch_updates: int
    save_steps: int
    warmup_steps: int
    shard_size: int
    warmup_epochs_effective: int
    world_size: int


@dataclass(frozen=True)
class ModePlannerConfig:
    branch: Branch
    hf_cache_path: str
    run_tag: str | None
    dataloader_num_workers: int
    seed: int
    loss_weights: Mapping[str, float]
    use_dino_encoder: bool
    drop_path_rate: float | None
    base_init_token: str
    learning_rate_mae: float
    learning_rate_contrastive: float
    warmup_ratio_mae: float
    warmup_ratio_contrastive: float
    num_train_epochs_mae: int
    num_train_epochs_contrastive: int
    branch_bs_map: Mapping[LossMode, tuple[int, int]]
    soft_hcl_alpha: float = 0.5
    soft_hcl_tau_h: float = 0.15
    simclr_debias_tau_plus: float | None = None
    hcl_beta: float = 1.0
    hcl_tau_plus: float | None = None
    contrastive_diagnostics_enabled: bool = True
    heavy_contrastive_diagnostics: bool = True


@dataclass(frozen=True)
class TrainingRunConfig:
    dataset_config: BinaryDatasetConfig
    hf_cache_path: str
    output_dir: str
    model_name_or_path: str
    mask_ratio: float
    loss_weights: Mapping[str, float]
    per_device_train_batch_size: int = 640
    num_train_epochs: int = 50
    dataloader_num_workers: int = 4
    viz_enabled: bool = True
    loss_mode: LossMode = LossMode.L2
    seed: int = 42
    patch_size: int = 16
    n_registers: int = 0
    data_policy: DataPolicy = field(default_factory=DataPolicy)
    use_dino_encoder: bool | None = None
    drop_path_rate: float | None = None
    n_views: int = 2
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    longrun_no_decay: bool = False
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 128
    soft_hcl_alpha: float = 0.5
    soft_hcl_tau_h: float = 0.15
    simclr_debias_tau_plus: float | None = None
    hcl_beta: float = 1.0
    hcl_tau_plus: float | None = None
    run_tag: str | None = None
    mode_name: str | None = None
    stage_name: str | None = None
    stage_tokens: Mapping[str, str] = field(default_factory=dict)
    mode_options: Mapping[str, object] = field(default_factory=dict)
    init_from_checkpoint: str | None = None
    expected_init_token: str | None = None
    checkpoint_policy_name: str = "standard"
    export_policy_name: str = "standard"
    aug_preview_samples: int = 0
    aug_preview_repeats: int = 1
    contrastive_diagnostics_enabled: bool = True
    heavy_contrastive_diagnostics: bool = True
    profile: SSLProfile = SSLProfile.SCRATCH
    branch: Branch = Branch.MAE

    def __post_init__(self) -> None:
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be positive.")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be positive.")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if self.n_registers < 0:
            raise ValueError("n_registers must be non-negative.")
        if self.n_views < 1:
            raise ValueError("n_views must be positive.")
        if self.projector_hidden_dim < 1:
            raise ValueError("projector_hidden_dim must be positive.")
        if self.projector_out_dim < 1:
            raise ValueError("projector_out_dim must be positive.")
        if not 0.0 <= float(self.mask_ratio) <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        if self.learning_rate is not None and float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive when provided.")
        if self.warmup_ratio is not None and not 0.0 <= float(self.warmup_ratio) <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1] when provided.")
        if float(self.soft_hcl_alpha) < 0.0:
            raise ValueError("soft_hcl_alpha must be non-negative.")
        if float(self.soft_hcl_tau_h) <= 0.0:
            raise ValueError("soft_hcl_tau_h must be positive.")
        if self.simclr_debias_tau_plus is not None and float(self.simclr_debias_tau_plus) < 0.0:
            raise ValueError("simclr_debias_tau_plus must be non-negative when provided.")
        if float(self.hcl_beta) < 0.0:
            raise ValueError("hcl_beta must be non-negative.")
        if self.hcl_tau_plus is not None and float(self.hcl_tau_plus) < 0.0:
            raise ValueError("hcl_tau_plus must be non-negative when provided.")
        if is_contrastive_loss(self.loss_mode) and self.n_views < 2:
            raise ValueError("Contrastive loss modes require n_views >= 2.")
        if self.heavy_contrastive_diagnostics and not self.contrastive_diagnostics_enabled:
            raise ValueError(
                "heavy_contrastive_diagnostics requires contrastive_diagnostics_enabled=True."
            )
        if self.branch == Branch.DINOV2 and not is_contrastive_loss(self.loss_mode):
            raise ValueError("Branch.DINOV2 only supports contrastive loss modes.")
        if not str(self.checkpoint_policy_name).strip():
            raise ValueError("checkpoint_policy_name must be non-empty.")
        if not str(self.export_policy_name).strip():
            raise ValueError("export_policy_name must be non-empty.")


@dataclass(frozen=True)
class PlannedRun:
    training_mode: TrainingMode
    stage_name: str
    log_prefix: str
    start_message: str
    run_root: str
    checkpoint_dir: str
    run_config: TrainingRunConfig
    skip_if_checkpoint_exists: bool = True
    cleanup_before_run: bool = False
    skip_message: str | None = None
