import math
import os
import types
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig, ViTMAEConfig, ViTMAEForPreTraining

from pretraining.run_types import Branch, InitMode, LossMode, SSLProfile, is_contrastive_loss
from pretraining.vit_mae_registers import ViTMAEWithRegistersForPreTraining
from pretraining.vit_utils import (
    Dinov2ForContrastive,
    MlpProjector,
    infer_num_prefix_tokens,
    load_hf_dinov2_config,
)
from utils.model_utils import count_parameters


@dataclass(frozen=True)
class ModelInputConfig:
    branch: Branch
    profile: SSLProfile
    model_name_or_path: str
    mask_ratio: float
    patch_size: int
    n_registers: int = 0
    init_from_checkpoint: str | None = None
    use_dino_encoder: bool | None = None
    drop_path_rate: float | None = None
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 128


@dataclass
class ModelBundle:
    config: PretrainedConfig
    model: torch.nn.Module
    init_token: str


def disable_mae_random_masking_for_encoder(model: ViTMAEForPreTraining) -> None:
    """
    Force HF ViT-MAE encoder masking/shuffle to become a no-op for contrastive runs.
    """
    vit = getattr(model, "vit", None)
    if vit is None or not hasattr(vit, "embeddings"):
        return
    emb = vit.embeddings

    def _random_masking_noop(self, x: torch.Tensor, noise: torch.Tensor | None = None):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() != 3:
            B = x.size(0) if x.dim() > 0 else 1
            L = x.size(1) if x.dim() > 1 else 1
            mask = x.new_zeros(B, L)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            return x, mask, ids_restore
        B, L, _ = x.shape
        mask = x.new_zeros(B, L)
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return x, mask, ids_restore

    try:
        emb.random_masking = types.MethodType(_random_masking_noop, emb)
    except Exception:
        pass


def load_vitmae_config(
    model_name_or_path: str,
    mask_ratio: float,
    patch_size: int,
    *,
    n_registers: int = 0,
    drop_path_rate: float | None = None,
    use_dino_encoder: bool | None = None,
    log_fn=print,
) -> ViTMAEConfig:
    try:
        config = ViTMAEConfig.from_pretrained(model_name_or_path)
    except Exception:
        config = ViTMAEConfig()
    config.mask_ratio = mask_ratio
    config.norm_pix_loss = False
    config.n_registers = int(n_registers)

    if use_dino_encoder is None and int(n_registers) > 0:
        use_dino_encoder = True
    if use_dino_encoder is not None:
        try:
            config.use_dino_encoder = bool(use_dino_encoder)
        except Exception:
            pass
    if bool(getattr(config, "use_dino_encoder", False)):
        try:
            config.layerscale_value = 1e-5
        except Exception as exc:
            log_fn(f"[Config] unable to set layerscale_value: {exc}")
        try:
            config.layer_norm_eps = 1e-6
        except Exception as exc:
            log_fn(f"[Config] unable to set layer_norm_eps: {exc}")

    if drop_path_rate is not None:
        try:
            config.drop_path_rate = float(drop_path_rate)
        except Exception as exc:
            log_fn(f"[Config] unable to set drop_path_rate: {exc}")

    try:
        ps = int(patch_size)
        if ps > 0:
            img_sz = int(getattr(config, "image_size", 224) or 224)
            if img_sz % ps != 0:
                new_img_sz = int(math.ceil(img_sz / ps) * ps)
                log_fn(
                    f"[PatchCompat] adjusting image_size {img_sz} -> {new_img_sz} "
                    f"to be divisible by patch_size={ps}"
                )
                config.image_size = new_img_sz
            if getattr(config, "patch_size", None) != ps:
                log_fn(f"[PatchCompat] setting config.patch_size={ps}")
                config.patch_size = ps
    except Exception as exc:
        log_fn(f"[PatchCompat] failed to set patch_size: {exc}")

    try:
        config.attn_implementation = "sdpa"
        log_fn(f"[Config] attn_implementation = {getattr(config, 'attn_implementation', None)}")
    except Exception as exc:
        log_fn(f"[Config] failed to set attn_implementation: {exc}")

    log_fn(f"[Config] norm_pix_loss = {getattr(config, 'norm_pix_loss', None)}")
    return config


def load_dinov2_config(
    model_name_or_path: str,
    *,
    patch_size: int | None = None,
    log_fn=print,
) -> PretrainedConfig:
    try:
        config = load_hf_dinov2_config(model_name_or_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load DINO config for '{model_name_or_path}': {exc}") from exc

    if patch_size is not None:
        try:
            ps = int(patch_size)
            if ps > 0 and getattr(config, "patch_size", None) != ps:
                log_fn(f"[PatchCompat] setting config.patch_size={ps}")
                config.patch_size = ps
        except Exception as exc:
            log_fn(f"[PatchCompat] failed to set DINO patch_size: {exc}")

    log_fn(f"[Config] loaded DINO config from '{model_name_or_path}'")
    return config


def _validate_hf_mae_init_config(
    *,
    model_name_or_path: str,
    config: ViTMAEConfig,
    use_dino_encoder: bool,
) -> None:
    if use_dino_encoder:
        raise RuntimeError(
            "Continued HF MAE init is not supported when config.use_dino_encoder=True. "
            "Use scratch init or an explicit checkpoint instead."
        )

    source_config = ViTMAEConfig.from_pretrained(model_name_or_path)
    for field_name in (
        "patch_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    ):
        expected = getattr(config, field_name, None)
        actual = getattr(source_config, field_name, None)
        if expected != actual:
            raise RuntimeError(
                f"HF init config mismatch for '{field_name}': expected={expected!r}, got={actual!r}."
            )


def _validate_phase1_checkpoint_config(
    *,
    checkpoint_dir: str,
    config: ViTMAEConfig,
    log_fn=print,
) -> None:
    try:
        checkpoint_config = ViTMAEConfig.from_pretrained(checkpoint_dir)
    except Exception as exc:
        log_fn(f"[Init] Unable to validate checkpoint config in '{checkpoint_dir}': {exc}")
        return

    for field_name in (
        "patch_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "image_size",
    ):
        expected = getattr(config, field_name, None)
        actual = getattr(checkpoint_config, field_name, None)
        if expected != actual:
            raise RuntimeError(
                f"Checkpoint config mismatch for '{field_name}': expected={expected!r}, got={actual!r}."
            )

    expected_registers = int(getattr(config, "n_registers", 0) or 0)
    actual_registers = int(getattr(checkpoint_config, "n_registers", 0) or 0)
    if expected_registers != actual_registers:
        raise RuntimeError(
            f"Checkpoint config mismatch for 'n_registers': expected={expected_registers}, got={actual_registers}."
        )

    expected_dino_encoder = bool(getattr(config, "use_dino_encoder", False))
    actual_dino_encoder = bool(getattr(checkpoint_config, "use_dino_encoder", False))
    if expected_dino_encoder != actual_dino_encoder:
        raise RuntimeError(
            "Checkpoint config mismatch for 'use_dino_encoder': "
            f"expected={expected_dino_encoder}, got={actual_dino_encoder}."
        )


def _load_hf_mae_weights_into_register_model(
    *,
    model: torch.nn.Module,
    model_name_or_path: str,
    log_fn=print,
) -> None:
    source_model = ViTMAEForPreTraining.from_pretrained(model_name_or_path)
    try:
        state = source_model.state_dict()
    finally:
        del source_model

    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "vit.register_tokens",
        "vit.patch_embed_norm.",
    )
    invalid_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "HF MAE weights are not fully compatible with the register-aware model. "
            f"missing={invalid_missing[:10]}{'...' if len(invalid_missing) > 10 else ''} "
            f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    log_fn(f"[Init] Loaded HF MAE weights from '{model_name_or_path}' into register-aware model.")


def load_or_init_mae_model(
    *,
    profile: SSLProfile,
    config: ViTMAEConfig,
    model_name_or_path: str,
    init_from_checkpoint: str | None,
    loss_mode: LossMode,
    n_registers: int = 0,
    projector_hidden_dim: int = 2048,
    projector_out_dim: int = 128,
    log_fn=print,
) -> tuple[torch.nn.Module, str]:
    use_dino_encoder = bool(getattr(config, "use_dino_encoder", False))
    use_registers = (int(n_registers) > 0) or use_dino_encoder
    model_cls = ViTMAEWithRegistersForPreTraining if use_registers else ViTMAEForPreTraining

    if profile == SSLProfile.CONTINUED and init_from_checkpoint is None:
        try:
            _validate_hf_mae_init_config(
                model_name_or_path=model_name_or_path,
                config=config,
                use_dino_encoder=use_dino_encoder,
            )
            if use_registers:
                model = model_cls(config, n_registers=n_registers)
                _load_hf_mae_weights_into_register_model(
                    model=model,
                    model_name_or_path=model_name_or_path,
                    log_fn=log_fn,
                )
            else:
                model = model_cls.from_pretrained(model_name_or_path, config=config)
            init_tok = InitMode.FACEBOOK_HF.value
            log_fn(f"[Init] Continued pretraining from HF MAE weights '{model_name_or_path}'.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HF MAE weights '{model_name_or_path}' for continued pretraining."
            ) from exc
    else:
        model = model_cls(config, n_registers=n_registers) if use_registers else model_cls(config)
        if init_from_checkpoint is not None:
            try:
                _validate_phase1_checkpoint_config(
                    checkpoint_dir=init_from_checkpoint,
                    config=config,
                    log_fn=log_fn,
                )
                bin_path = os.path.join(init_from_checkpoint, "pytorch_model.bin")
                safe_path = os.path.join(init_from_checkpoint, "model.safetensors")

                if os.path.exists(bin_path):
                    state = torch.load(bin_path, map_location="cpu")
                    src_path = bin_path
                elif os.path.exists(safe_path):
                    from safetensors.torch import load_file as safe_load

                    state = safe_load(safe_path)
                    src_path = safe_path
                else:
                    raise FileNotFoundError(
                        f"No model weights found in '{init_from_checkpoint}' "
                        f"(expected 'pytorch_model.bin' or 'model.safetensors')."
                    )

                missing, unexpected = model.load_state_dict(state, strict=False)
                log_fn(f"[Init] Loaded MAE weights from {src_path}")
                if missing or unexpected:
                    raise RuntimeError(
                        "Checkpoint state_dict is not fully compatible with the target model. "
                        f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
                        f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
                    )
                init_tok = InitMode.MAE_PHASE1.value
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MAE checkpoint from '{init_from_checkpoint}'."
                ) from exc
        else:
            log_fn("[Init] ViTMAEWithRegistersForPreTraining random init (scratch)")
            init_tok = InitMode.SCRATCH_HF.value

    if is_contrastive_loss(loss_mode):
        config.mask_ratio = 0.0
        disable_mae_random_masking_for_encoder(model)
        log_fn("[Masking] Contrastive mode -> disabled MAE random_masking for encoder (no mask, no shuffle).")
        log_fn("[Config] contrastive mode -> forcing mask_ratio=0.0 for encoder")

    feat_dim = model.vit.config.hidden_size
    if is_contrastive_loss(loss_mode) and not hasattr(model, "projector_head"):
        model.projector_head = MlpProjector(
            feat_dim,
            hidden_dim=int(projector_hidden_dim),
            out_dim=int(projector_out_dim),
        )
        log_fn(
            f"[Head] Contrastive: using 2-layer projector "
            f"({int(projector_hidden_dim)} -> {int(projector_out_dim)})."
        )

    if is_contrastive_loss(loss_mode):
        for attr in [
            "decoder",
            "decoder_embed",
            "decoder_blocks",
            "decoder_pred",
            "decoder_norm",
            "decoder_pos_embed",
            "mask_token",
        ]:
            if hasattr(model, attr):
                try:
                    mod = getattr(model, attr)
                    if isinstance(mod, torch.nn.Module):
                        mod.to(torch.device("cpu"))
                    setattr(model, attr, None)
                except Exception:
                    setattr(model, attr, None)
        log_fn("[Model] Dropped MAE decoder for encoder-only contrastive mode.")

    try:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1
    if is_contrastive_loss(loss_mode) and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        log_fn("[BN] Using SyncBatchNorm across GPUs")
    else:
        log_fn("[BN] Using local BatchNorm")

    if int(os.getenv("RANK", "0")) == 0:
        count_parameters(model)

    return model, init_tok


def load_or_init_dinov2_contrastive_model(
    *,
    model_name_or_path: str,
    projector_hidden_dim: int = 2048,
    projector_out_dim: int = 128,
    log_fn=print,
) -> tuple[torch.nn.Module, str]:
    model = Dinov2ForContrastive.from_pretrained(
        model_name_or_path,
        projector_hidden_dim=int(projector_hidden_dim),
        projector_out_dim=int(projector_out_dim),
    )
    log_fn(f"[Init] Loaded HF DINOv2 weights from '{model_name_or_path}' for contrastive continued pretraining.")
    log_fn(
        f"[Head] DINO projector: using 2-layer projector without output BatchNorm "
        f"({int(projector_hidden_dim)} -> {int(projector_out_dim)})."
    )
    if int(os.getenv("RANK", "0")) == 0:
        count_parameters(model)
    return model, InitMode.DINOV2_HF.value


class SSLModelFactory:
    """Single owner for branch-specific config loading and model initialization."""

    def __init__(self, input_config: ModelInputConfig, *, log_fn=print):
        self.input_config = input_config
        self.log_fn = log_fn

    def build(self, *, loss_mode: LossMode) -> ModelBundle:
        cfg = self.input_config

        if cfg.branch == Branch.MAE:
            config = load_vitmae_config(
                cfg.model_name_or_path,
                cfg.mask_ratio,
                cfg.patch_size,
                n_registers=cfg.n_registers,
                drop_path_rate=cfg.drop_path_rate,
                use_dino_encoder=cfg.use_dino_encoder,
                log_fn=self.log_fn,
            )
            model, init_tok = load_or_init_mae_model(
                profile=cfg.profile,
                config=config,
                model_name_or_path=cfg.model_name_or_path,
                init_from_checkpoint=cfg.init_from_checkpoint,
                loss_mode=loss_mode,
                n_registers=cfg.n_registers,
                projector_hidden_dim=cfg.projector_hidden_dim,
                projector_out_dim=cfg.projector_out_dim,
                log_fn=self.log_fn,
            )
            return ModelBundle(config=config, model=model, init_token=init_tok)

        if cfg.branch == Branch.DINOV2:
            if not is_contrastive_loss(loss_mode):
                raise ValueError(
                    f"Branch.DINOV2 only supports contrastive losses; got {loss_mode.value}."
                )
            if cfg.init_from_checkpoint is not None:
                raise ValueError("Branch.DINOV2 currently supports direct HF init only (no MAE checkpoint init).")
            config = load_dinov2_config(
                cfg.model_name_or_path,
                patch_size=cfg.patch_size,
                log_fn=self.log_fn,
            )
            actual_dino_registers = max(0, int(infer_num_prefix_tokens(config)) - 1)
            if int(cfg.n_registers) != actual_dino_registers:
                raise ValueError(
                    f"Branch.DINOV2 register metadata mismatch for '{cfg.model_name_or_path}': "
                    f"requested n_registers={cfg.n_registers}, but config implies {actual_dino_registers}."
                )
            model, init_tok = load_or_init_dinov2_contrastive_model(
                model_name_or_path=cfg.model_name_or_path,
                projector_hidden_dim=cfg.projector_hidden_dim,
                projector_out_dim=cfg.projector_out_dim,
                log_fn=self.log_fn,
            )
            return ModelBundle(config=config, model=model, init_token=init_tok)

        raise ValueError(f"Unsupported branch={cfg.branch}")
