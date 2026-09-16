import os

from pretraining.data_pipeline import BinaryDatasetConfig, DataPolicy
from pretraining.run_types import (
    Branch,
    InitMode,
    LossMode,
    ModePlannerConfig,
    PlannedRun,
    RunSpec,
    TrainingMode,
    TrainingRunConfig,
    is_contrastive_loss,
)
from utils.manifest_utils import fmt_lr_token, short_arch_name

MODEL_RUNS_DIR = "Models"


def proj_token(
    loss_mode: LossMode,
    projector_hidden_dim: int | None = None,
    projector_out_dim: int | None = None,
    projector_layers: int | None = 2,
) -> str:
    """Standardized projector token: 'mlp-L{layers}-H{hidden}-O{out}[+pred]'."""
    try:
        key = getattr(loss_mode, "value", str(loss_mode)).lower()
    except Exception:
        key = str(loss_mode)
    if key in ("simclr", "simclr_debiased", "soft_hcl", "hcl"):
        layers = 2 if projector_layers is None else int(projector_layers)
        hidden = 2048 if projector_hidden_dim is None else int(projector_hidden_dim)
        out = 128 if projector_out_dim is None else int(projector_out_dim)
        has_pred = False
    else:
        return "none"
    tok = f"mlp-L{layers}-H{hidden}-O{out}"
    if has_pred:
        tok += "+pred"
    return tok


def build_run_components(
    *,
    branch: Branch,
    mode_name: str,
    stage_name: str,
    init_tok: str,
    loss_mode: LossMode,
    learning_rate: float,
    projector_hidden_dim: int,
    projector_out_dim: int,
    run_tag: str | None,
    n_registers: int,
    dataset_config: BinaryDatasetConfig,
    data_policy: DataPolicy,
    use_dino_encoder: bool,
    n_views: int,
    stage_tokens: dict[str, str] | None = None,
) -> list[str]:
    components = [
        f"mode={mode_name}",
        f"stage={stage_name}",
        f"init={init_tok}",
        f"loss={loss_mode.value.lower()}",
        f"lr={fmt_lr_token(learning_rate)}",
        f"proj={proj_token(loss_mode, projector_hidden_dim=projector_hidden_dim, projector_out_dim=projector_out_dim)}",
    ]
    if stage_tokens:
        components.extend(f"{key}={value}" for key, value in stage_tokens.items())
    components.extend(dataset_config.run_tokens())
    components.extend(data_policy.run_tokens())
    if run_tag:
        components.append(str(run_tag))
    if n_registers and int(n_registers) > 0:
        components.append(f"regs={int(n_registers)}")
        if branch == Branch.MAE and not use_dino_encoder:
            components.append("nodinoenc")
    if is_contrastive_loss(loss_mode):
        components.append(f"views={int(n_views)}")
    return components


def build_run_root(
    *,
    branch: Branch,
    arch_tok: str,
    components: list[str],
    patch_size: int,
    mask: float,
) -> str:
    os.makedirs(MODEL_RUNS_DIR, exist_ok=True)
    base_name = f"{arch_tok}-{'--'.join(components)}"
    base = os.path.join(".", MODEL_RUNS_DIR, base_name)
    if branch == Branch.DINOV2:
        return base
    return f"{base}--patch={patch_size}--mask={mask}"


def _checkpoint_dir(run_root: str) -> str:
    return os.path.join(run_root, "checkpoints")


def _build_stage_run_config(
    *,
    ctx: ModePlannerConfig,
    spec: RunSpec,
    output_dir: str,
    loss_mode: LossMode,
    per_device_train_batch_size: int,
    num_train_epochs: int,
    viz_enabled: bool,
    learning_rate: float,
    warmup_ratio: float,
    mode_name: str,
    stage_name: str,
    stage_tokens: dict[str, str] | None,
    init_from_checkpoint: str | None,
    expected_init_token: str | None,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        dataset_config=spec.dataset_config,
        hf_cache_path=ctx.hf_cache_path,
        model_name_or_path=spec.model_name,
        mask_ratio=spec.mask,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        dataloader_num_workers=ctx.dataloader_num_workers,
        num_train_epochs=num_train_epochs,
        loss_weights=ctx.loss_weights,
        viz_enabled=viz_enabled,
        loss_mode=loss_mode,
        seed=ctx.seed,
        patch_size=spec.patch_size,
        n_registers=spec.n_registers,
        data_policy=spec.data_policy,
        use_dino_encoder=ctx.use_dino_encoder,
        drop_path_rate=ctx.drop_path_rate,
        n_views=spec.n_views,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        projector_hidden_dim=spec.projector_hidden_dim,
        projector_out_dim=spec.projector_out_dim,
        soft_hcl_alpha=ctx.soft_hcl_alpha,
        soft_hcl_tau_h=ctx.soft_hcl_tau_h,
        simclr_debias_tau_plus=ctx.simclr_debias_tau_plus,
        hcl_beta=ctx.hcl_beta,
        hcl_tau_plus=ctx.hcl_tau_plus,
        run_tag=ctx.run_tag,
        mode_name=mode_name,
        stage_name=stage_name,
        stage_tokens=(dict(stage_tokens) if stage_tokens else {}),
        mode_options={},
        init_from_checkpoint=init_from_checkpoint,
        expected_init_token=expected_init_token,
        checkpoint_policy_name="standard",
        export_policy_name="standard",
        aug_preview_samples=spec.preview_samples,
        aug_preview_repeats=spec.preview_repeats,
        contrastive_diagnostics_enabled=ctx.contrastive_diagnostics_enabled,
        heavy_contrastive_diagnostics=ctx.heavy_contrastive_diagnostics,
        profile=spec.profile,
        branch=ctx.branch,
    )


def _plan_mae_only(spec: RunSpec, ctx: ModePlannerConfig) -> tuple[PlannedRun, ...]:
    mae_loss_mode = spec.loss_modes[0]
    arch_tok = short_arch_name(spec.model_name.split("/")[-1])
    mode_name = spec.training_mode.value
    stage_name = "main"
    stage_tokens: dict[str, str] = {}
    run_root = build_run_root(
        branch=ctx.branch,
        arch_tok=arch_tok,
        components=build_run_components(
            branch=ctx.branch,
            mode_name=mode_name,
            stage_name=stage_name,
            init_tok=ctx.base_init_token,
            loss_mode=mae_loss_mode,
            learning_rate=ctx.learning_rate_mae,
            projector_hidden_dim=spec.projector_hidden_dim,
            projector_out_dim=spec.projector_out_dim,
            run_tag=ctx.run_tag,
            n_registers=spec.n_registers,
            dataset_config=spec.dataset_config,
            data_policy=spec.data_policy,
            use_dino_encoder=ctx.use_dino_encoder,
            n_views=spec.n_views,
            stage_tokens=stage_tokens,
        ),
        patch_size=spec.patch_size,
        mask=spec.mask,
    )
    bs_mae, _ = ctx.branch_bs_map.get(mae_loss_mode, (640, 640))
    run_config = _build_stage_run_config(
        ctx=ctx,
        spec=spec,
        output_dir=run_root,
        loss_mode=mae_loss_mode,
        per_device_train_batch_size=bs_mae,
        num_train_epochs=ctx.num_train_epochs_mae,
        viz_enabled=True,
        learning_rate=ctx.learning_rate_mae,
        warmup_ratio=ctx.warmup_ratio_mae,
        mode_name=mode_name,
        stage_name=stage_name,
        stage_tokens=stage_tokens,
        init_from_checkpoint=None,
        expected_init_token=ctx.base_init_token,
    )
    return (
        PlannedRun(
            training_mode=spec.training_mode,
            stage_name=stage_name,
            log_prefix="MAE_ONLY",
            start_message=(
                f"[MAE_ONLY] Starting MAE run with mask_ratio={spec.mask} and patch size={spec.patch_size} "
                f"(mode={mae_loss_mode.value})..."
            ),
            run_root=run_root,
            checkpoint_dir=_checkpoint_dir(run_root),
            run_config=run_config,
            skip_if_checkpoint_exists=True,
            cleanup_before_run=False,
            skip_message=None,
        ),
    )


def _plan_contrastive_only(spec: RunSpec, ctx: ModePlannerConfig) -> tuple[PlannedRun, ...]:
    contrastive_loss_mode = spec.loss_modes[0]
    arch_tok = short_arch_name(spec.model_name.split("/")[-1])
    mode_name = spec.training_mode.value
    stage_name = "main"
    stage_tokens = {"prephase": "none"}
    run_root = build_run_root(
        branch=ctx.branch,
        arch_tok=arch_tok,
        components=build_run_components(
            branch=ctx.branch,
            mode_name=mode_name,
            stage_name=stage_name,
            init_tok=ctx.base_init_token,
            loss_mode=contrastive_loss_mode,
            learning_rate=ctx.learning_rate_contrastive,
            projector_hidden_dim=spec.projector_hidden_dim,
            projector_out_dim=spec.projector_out_dim,
            run_tag=ctx.run_tag,
            n_registers=spec.n_registers,
            dataset_config=spec.dataset_config,
            data_policy=spec.data_policy,
            use_dino_encoder=ctx.use_dino_encoder,
            n_views=spec.n_views,
            stage_tokens=stage_tokens,
        ),
        patch_size=spec.patch_size,
        mask=(0.0 if ctx.branch == Branch.MAE else spec.mask),
    )
    _, bs_contrastive = ctx.branch_bs_map.get(contrastive_loss_mode, (64, 64))
    run_config = _build_stage_run_config(
        ctx=ctx,
        spec=spec,
        output_dir=run_root,
        loss_mode=contrastive_loss_mode,
        per_device_train_batch_size=bs_contrastive,
        num_train_epochs=ctx.num_train_epochs_contrastive,
        viz_enabled=False,
        learning_rate=ctx.learning_rate_contrastive,
        warmup_ratio=ctx.warmup_ratio_contrastive,
        mode_name=mode_name,
        stage_name=stage_name,
        stage_tokens=stage_tokens,
        init_from_checkpoint=None,
        expected_init_token=ctx.base_init_token,
    )
    return (
        PlannedRun(
            training_mode=spec.training_mode,
            stage_name=stage_name,
            log_prefix="CONTRASTIVE_ONLY",
            start_message=(
                f"[CONTRASTIVE_ONLY] Starting contrastive run (mode={contrastive_loss_mode.value}) "
                "directly from HF init..."
            ),
            run_root=run_root,
            checkpoint_dir=_checkpoint_dir(run_root),
            run_config=run_config,
            skip_if_checkpoint_exists=True,
            cleanup_before_run=True,
            skip_message=None,
        ),
    )


def _plan_mae_then_contrastive(spec: RunSpec, ctx: ModePlannerConfig) -> tuple[PlannedRun, ...]:
    mae_loss_mode, contrastive_loss_mode = spec.loss_modes
    arch_tok = short_arch_name(spec.model_name.split("/")[-1])
    mode_name = spec.training_mode.value

    phase1_stage_tokens: dict[str, str] = {}
    phase1_run_root = build_run_root(
        branch=ctx.branch,
        arch_tok=arch_tok,
        components=build_run_components(
            branch=ctx.branch,
            mode_name=mode_name,
            stage_name="phase1",
            init_tok=ctx.base_init_token,
            loss_mode=mae_loss_mode,
            learning_rate=ctx.learning_rate_mae,
            projector_hidden_dim=spec.projector_hidden_dim,
            projector_out_dim=spec.projector_out_dim,
            run_tag=ctx.run_tag,
            n_registers=spec.n_registers,
            dataset_config=spec.dataset_config,
            data_policy=spec.data_policy,
            use_dino_encoder=ctx.use_dino_encoder,
            n_views=spec.n_views,
            stage_tokens=phase1_stage_tokens,
        ),
        patch_size=spec.patch_size,
        mask=spec.mask,
    )
    bs_mae, _ = ctx.branch_bs_map.get(mae_loss_mode, (640, 640))
    phase1_run_config = _build_stage_run_config(
        ctx=ctx,
        spec=spec,
        output_dir=phase1_run_root,
        loss_mode=mae_loss_mode,
        per_device_train_batch_size=bs_mae,
        num_train_epochs=ctx.num_train_epochs_mae,
        viz_enabled=True,
        learning_rate=ctx.learning_rate_mae,
        warmup_ratio=ctx.warmup_ratio_mae,
        mode_name=mode_name,
        stage_name="phase1",
        stage_tokens=phase1_stage_tokens,
        init_from_checkpoint=None,
        expected_init_token=ctx.base_init_token,
    )

    phase2_stage_tokens = {
        "prephase": mae_loss_mode.value.lower(),
        "phase1_loss": mae_loss_mode.value.lower(),
        "phase1_mask": str(spec.mask),
    }
    phase2_run_root = build_run_root(
        branch=ctx.branch,
        arch_tok=arch_tok,
        components=build_run_components(
            branch=ctx.branch,
            mode_name=mode_name,
            stage_name="phase2",
            init_tok=InitMode.MAE_PHASE1.value,
            loss_mode=contrastive_loss_mode,
            learning_rate=ctx.learning_rate_contrastive,
            projector_hidden_dim=spec.projector_hidden_dim,
            projector_out_dim=spec.projector_out_dim,
            run_tag=ctx.run_tag,
            n_registers=spec.n_registers,
            dataset_config=spec.dataset_config,
            data_policy=spec.data_policy,
            use_dino_encoder=ctx.use_dino_encoder,
            n_views=spec.n_views,
            stage_tokens=phase2_stage_tokens,
        ),
        patch_size=spec.patch_size,
        mask=(0.0 if ctx.branch == Branch.MAE else spec.mask),
    )
    _, bs_contrastive = ctx.branch_bs_map.get(contrastive_loss_mode, (64, 64))
    phase2_run_config = _build_stage_run_config(
        ctx=ctx,
        spec=spec,
        output_dir=phase2_run_root,
        loss_mode=contrastive_loss_mode,
        per_device_train_batch_size=bs_contrastive,
        num_train_epochs=ctx.num_train_epochs_contrastive,
        viz_enabled=False,
        learning_rate=ctx.learning_rate_contrastive,
        warmup_ratio=ctx.warmup_ratio_contrastive,
        mode_name=mode_name,
        stage_name="phase2",
        stage_tokens=phase2_stage_tokens,
        init_from_checkpoint=_checkpoint_dir(phase1_run_root),
        expected_init_token=InitMode.MAE_PHASE1.value,
    )

    return (
        PlannedRun(
            training_mode=spec.training_mode,
            stage_name="phase1",
            log_prefix="MAE_THEN_CONTRASTIVE][PHASE1",
            start_message=(
                f"[MAE_THEN_CONTRASTIVE][PHASE1] Starting MAE run with mask_ratio={spec.mask} "
                f"and patch size={spec.patch_size} (mode={mae_loss_mode.value})..."
            ),
            run_root=phase1_run_root,
            checkpoint_dir=_checkpoint_dir(phase1_run_root),
            run_config=phase1_run_config,
            skip_if_checkpoint_exists=True,
            cleanup_before_run=False,
            skip_message=(
                f"[MAE_THEN_CONTRASTIVE][PHASE1] Skipping MAE run "
                f"(existing checkpoint found at {_checkpoint_dir(phase1_run_root)})"
            ),
        ),
        PlannedRun(
            training_mode=spec.training_mode,
            stage_name="phase2",
            log_prefix="MAE_THEN_CONTRASTIVE][PHASE2",
            start_message=(
                f"[MAE_THEN_CONTRASTIVE][PHASE2] Starting contrastive run "
                f"(mode={contrastive_loss_mode.value}) from MAE checkpoint..."
            ),
            run_root=phase2_run_root,
            checkpoint_dir=_checkpoint_dir(phase2_run_root),
            run_config=phase2_run_config,
            skip_if_checkpoint_exists=False,
            cleanup_before_run=True,
            skip_message=None,
        ),
    )


def plan_runs_for_spec(spec: RunSpec, ctx: ModePlannerConfig) -> tuple[PlannedRun, ...]:
    if spec.training_mode == TrainingMode.MAE_ONLY:
        return _plan_mae_only(spec, ctx)
    if spec.training_mode == TrainingMode.CONTRASTIVE_ONLY:
        return _plan_contrastive_only(spec, ctx)
    if spec.training_mode == TrainingMode.MAE_THEN_CONTRASTIVE:
        return _plan_mae_then_contrastive(spec, ctx)
    raise ValueError(f"Unsupported training_mode={spec.training_mode}")
