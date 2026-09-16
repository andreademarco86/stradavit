# --- Standard and 3rd-party imports ---
import torch
import os
import importlib
# Prefer TF32 on Ampere/Ada for faster GEMMs while retaining fp32 API numerics
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # or "medium"
except Exception:
    pass

import torch.nn.functional as F

from transformers import (
    ViTMAEConfig,
    PreTrainedModel,
    AutoConfig,
)

# For loading .safetensors files
from safetensors.torch import load_file as safe_load_file

def infer_num_prefix_tokens(model_or_config) -> int:
    """Infer how many non-patch prefix tokens a ViT-family encoder uses.

    Returns at least 1 for CLS. If the config exposes register-token counts,
    they are added on top.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    n_regs = 0
    for attr in ("num_register_tokens", "n_registers", "num_registers"):
        try:
            val = int(getattr(cfg, attr, 0) or 0)
        except Exception:
            val = 0
        n_regs = max(n_regs, val)
    return 1 + max(0, n_regs)


def pool_patch_tokens(
    hs: torch.Tensor,
    *,
    use_cls_token: bool = False,
    model_or_config=None,
) -> torch.Tensor:
    """Pool ViT-family hidden states into one embedding.

    - Supports variable prefix-token counts (CLS + optional registers).
    - Defaults to mean-pooling patch tokens after all prefix tokens.
    - Can optionally return the CLS token directly.
    """
    if hs is None:
        raise ValueError("pool_patch_tokens: `hs` must not be None")

    if not isinstance(hs, torch.Tensor):
        hs = torch.as_tensor(hs)

    if hs.dim() != 3:
        if hs.dim() >= 2:
            return hs.view(hs.size(0), -1)
        return hs

    batch_size, token_count, hidden_dim = hs.shape
    if token_count == 0:
        return hs.new_zeros(batch_size, hidden_dim)

    if use_cls_token:
        return hs[:, 0, :]

    prefix_tokens = 1
    if model_or_config is not None:
        try:
            prefix_tokens = int(max(1, infer_num_prefix_tokens(model_or_config)))
        except Exception:
            prefix_tokens = 1

    if token_count > prefix_tokens:
        return hs[:, prefix_tokens:, :].mean(dim=1)

    return hs[:, 0, :]


def _safe_pretrained_config_dict(model_name_or_path: str) -> dict:
    """Fetch config dict without requiring AutoConfig model_type mapping support."""
    try:
        from transformers.configuration_utils import PretrainedConfig  # type: ignore
        cfg_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
        if isinstance(cfg_dict, dict):
            return cfg_dict
    except Exception:
        pass
    return {}


def load_hf_dinov2_config_dict(model_name_or_path: str) -> dict:
    """Return the raw config dict for a DINO checkpoint/repo when available."""
    cfg_dict = _safe_pretrained_config_dict(model_name_or_path)
    return cfg_dict if isinstance(cfg_dict, dict) else {}


def _hydrate_config_from_dict(config_cls, cfg_dict: dict):
    """Instantiate a config class from a raw config dict, preserving source values."""
    if not isinstance(cfg_dict, dict) or not cfg_dict:
        return None
    try:
        return config_cls.from_dict(cfg_dict)
    except Exception:
        try:
            return config_cls(**cfg_dict)
        except Exception:
            return None


def is_dinov2_with_registers_config(model_or_config) -> bool:
    """Return True when a DINO config/checkpoint implies native register tokens."""
    if isinstance(model_or_config, str):
        model_ref = model_or_config.lower()
        if "with-registers" in model_ref or "with_registers" in model_ref:
            return True
        cfg_dict = _safe_pretrained_config_dict(model_or_config)
        model_type = str(cfg_dict.get("model_type", "") or "").lower()
        archs = cfg_dict.get("architectures", None) or []
        nreg_candidates = [
            cfg_dict.get("num_register_tokens", 0),
            cfg_dict.get("n_registers", 0),
            cfg_dict.get("num_registers", 0),
        ]
    else:
        cfg = getattr(model_or_config, "config", model_or_config)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        archs = getattr(cfg, "architectures", None) or []
        nreg_candidates = []
        for attr in ("num_register_tokens", "n_registers", "num_registers"):
            try:
                nreg_candidates.append(getattr(cfg, attr, 0))
            except Exception:
                nreg_candidates.append(0)

    if "with_registers" in model_type or "with-registers" in model_type:
        return True

    try:
        arch_names = [str(a).lower() for a in archs]
    except Exception:
        arch_names = []
    if any("withregisters" in a or "with_registers" in a for a in arch_names):
        return True

    for val in nreg_candidates:
        try:
            if int(val or 0) > 0:
                return True
        except Exception:
            continue
    return False


def load_hf_dinov2_config(model_name_or_path: str):
    """Load the exact HF DINO config class for plain vs. register-bearing repos."""
    cfg_dict = _safe_pretrained_config_dict(model_name_or_path)

    use_registers = is_dinov2_with_registers_config(model_name_or_path)

    if use_registers:
        try:
            module = importlib.import_module(
                "transformers.models.dinov2_with_registers.configuration_dinov2_with_registers"
            )
            Dinov2WithRegistersConfig = getattr(module, "Dinov2WithRegistersConfig")
            cfg = _hydrate_config_from_dict(Dinov2WithRegistersConfig, cfg_dict)
            if cfg is not None:
                return cfg
            return Dinov2WithRegistersConfig.from_pretrained(model_name_or_path)
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 with-registers config detected, but the required HF config "
                "class transformers.models.dinov2_with_registers.configuration_dinov2_with_registers."
                "Dinov2WithRegistersConfig could not be imported or loaded. Install the "
                "pinned project requirements before running evaluation."
            ) from exc

    try:
        module = importlib.import_module("transformers.models.dinov2.configuration_dinov2")
        Dinov2Config = getattr(module, "Dinov2Config")
        cfg = _hydrate_config_from_dict(Dinov2Config, cfg_dict)
        if cfg is not None:
            return cfg
        return Dinov2Config.from_pretrained(model_name_or_path)
    except Exception:
        pass

    raise ValueError(
        f"Could not load the required HF DINOv2 config class for '{model_name_or_path}'. "
        "Install the pinned project requirements before running evaluation."
    )


def get_hf_dinov2_model_cls(model_or_config):
    """Resolve the exact HF DINO backbone class for plain vs. register-bearing repos."""
    use_registers = is_dinov2_with_registers_config(model_or_config)
    if use_registers:
        module_name = "transformers.models.dinov2_with_registers.modeling_dinov2_with_registers"
        class_name = "Dinov2WithRegistersModel"
    else:
        module_name = "transformers.models.dinov2.modeling_dinov2"
        class_name = "Dinov2Model"
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import the required HF backbone class {module_name}.{class_name}. "
            "Install the pinned project requirements before running evaluation."
        ) from exc


def get_hf_dinov2_image_classification_cls(model_or_config):
    """Resolve the exact HF DINO image-classification class for plain vs. registers."""
    use_registers = is_dinov2_with_registers_config(model_or_config)
    if use_registers:
        module_name = "transformers.models.dinov2_with_registers.modeling_dinov2_with_registers"
        class_name = "Dinov2WithRegistersForImageClassification"
    else:
        module_name = "transformers.models.dinov2.modeling_dinov2"
        class_name = "Dinov2ForImageClassification"
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import the required HF image-classification class {module_name}.{class_name}. "
            "Install the pinned project requirements before running evaluation."
        ) from exc


def load_hf_dinov2_backbone(model_name_or_path: str):
    """Load the appropriate native HF DINO backbone class for a checkpoint/repo."""
    cfg = load_hf_dinov2_config(model_name_or_path)
    model_cls = get_hf_dinov2_model_cls(cfg)
    return model_cls.from_pretrained(model_name_or_path)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """1D sin-cos positional embedding from positions (M,) -> (M, D).

    Direct torch port of MAE's util/pos_embed.get_1d_sincos_pos_embed_from_grid.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for sin-cos embedding"

    # frequency spectrum
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)

    # positions
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """2D sin-cos embedding from a (2, 1, H, W) grid tensor.

    Mirrors MAE's get_2d_sincos_pos_embed_from_grid.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos embedding"

    # use half of dimensions to encode each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> torch.Tensor:
    """2D sin-cos position embedding as in MAE.

    Returns a tensor of shape (grid_size*grid_size + cls_token, embed_dim).
    If cls_token is True, the first row is zeros (CLS), matching MAE/timm
    when extra_tokens == 1.
    """
    # base grid, MAE-style: np.meshgrid(grid_w, grid_h) with indexing="xy"
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    # gw, gh have shape (H, W); gw holds w-coordinates, gh holds h-coordinates
    gw, gh = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack([gw, gh], dim=0)           # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)  # (2, 1, H, W) as in MAE

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D)

    if cls_token:
        cls = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls, pos_embed], dim=0)  # (1 + H*W, D)

    return pos_embed

class ViTMLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, in_features)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTAttention(torch.nn.Module):
    """Multi-head self-attention with fused QKV projection, HF-style."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection to mirror HF ViT/ViTMAE
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim)
        self.proj_drop = torch.nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Fused QKV projection: (B, N, 3 * C) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Memory-efficient scaled dot-product attention (PyTorch 2 SDPA)
        drop_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
        )

        # Merge heads
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTBlock(torch.nn.Module):
    """Pre-LN transformer block, close to HF ViT/ViTMAE behaviour."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = ViTAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = ViTMLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual block
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CustomViTMAEOutput:
    """Lightweight output container mirroring the fields we actually use from ViTMAEForPreTrainingOutput.

    Attributes
    ----------
    loss : torch.Tensor | None
        Masked-patch MSE loss.
    logits : torch.Tensor
        Decoder patch predictions of shape (B, N, patch_dim).
    mask : torch.Tensor
        Boolean mask over patches, True for masked positions, shape (B, N).
    ids_restore : torch.Tensor
        Indices to restore full sequence order, shape (B, N).
    hidden_states : tuple[torch.Tensor] | None
        Optional encoder hidden states (we use the last one for diagnostics).
    """
    def __init__(self, loss, logits, mask, ids_restore, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.mask = mask
        self.ids_restore = ids_restore
        self.hidden_states = hidden_states

class CustomViTEncoderOutput:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states

class CustomViTEncoderWrapper(torch.nn.Module):
    """Thin wrapper that exposes the encoder-only forward used by SIMCLR as `model.vit`.

    This keeps SimCLR code like:

        enc = model_wrapped.vit
        out = enc(pixel_values, output_hidden_states=True)

    working on top of the MAE backbone, without creating a recursive module graph.
    """
    def __init__(self, mae_model: "CustomViTMAEForPreTraining"):
        super().__init__()
        # Do NOT store mae_model as a submodule; that would create a cycle:
        # mae_model -> vit -> mae_model -> ...
        # Instead, store a bound helper to the encoder-only path and the config.
        self._encode_mae_mask_fn = mae_model._encode_mae_masked
        # Expose config so existing code can query vit.config.hidden_size
        self.config = mae_model.config

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False) -> CustomViTEncoderOutput:
        tokens, hs = self._encode_mae_mask_fn(pixel_values, output_hidden_states=output_hidden_states)
        return CustomViTEncoderOutput(last_hidden_state=tokens, hidden_states=hs)

class CustomViTMAEForPreTraining(PreTrainedModel):
    """Editable MAE-style ViT model.

    Design choices:
      - Conv patch embedding (kernel=stride=patch_size).
      - CLS token in the encoder, following HF ViTMAE.
      - Random per-sample masking in the encoder (keep ratio = 1 - mask_ratio).
      - Transformer decoder that reconstructs all patches.

    The goal is to be behaviorally equivalent to ViTMAEForPreTraining for our usage
    (same inputs/outputs and loss semantics), while being easy to extend later
    with DINOv2-like architectural tweaks.
    """
    # Hugging Face integration: tell PreTrainedModel which config class we use
    config_class = ViTMAEConfig
    base_model_prefix = "vit"

    def __init__(self, config: "ViTMAEConfig") -> None:
        # Initialise as a HF PreTrainedModel so that save_pretrained / from_pretrained work.
        super().__init__(config)
        self.config = config

        img_size = int(getattr(config, "image_size", 224) or 224)
        patch_size = int(getattr(config, "patch_size", 16) or 16)
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_channels = int(getattr(config, "num_channels", 3) or 3)
        self.embed_dim = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_layers = int(config.num_hidden_layers)
        self.mlp_ratio = float(
            getattr(config, "intermediate_size", 4 * self.embed_dim) / self.embed_dim
        )

        # --- Patch embedding (conv2d patchify) ---
        self.patch_embed = torch.nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches_h = img_size // patch_size
        num_patches_w = img_size // patch_size
        self.num_patches = num_patches_h * num_patches_w

        # CLS token + absolute positional embeddings for encoder (CLS + patches)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # Fixed 2D sin-cos positional embeddings (filled in post_init)
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, 1 + self.num_patches, self.embed_dim),
            persistent=False,
        )

        # --- Encoder: ViT stack (no CLS token) ---
        dropout = float(getattr(self.config, "hidden_dropout_prob", 0.0))
        attn_dropout = float(getattr(self.config, "attention_probs_dropout_prob", 0.0))
        ln_eps = float(getattr(self.config, "layer_norm_eps", 1e-6))
        qkv_bias = bool(getattr(self.config, "qkv_bias", True))

        # HF-style ViT encoder blocks (pre-LN, qkv_bias, config-driven dropout)
        self.encoder = torch.nn.ModuleList(
            [
                ViTBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm_eps=ln_eps,
                )
                for _ in range(self.num_layers)
            ]
        )

        # ✅ ADD THIS LINE:
        self.layernorm = torch.nn.LayerNorm(self.embed_dim, eps=ln_eps)

        # --- Decoder configuration ---
        decoder_dim = int(getattr(config, "decoder_hidden_size", self.embed_dim))
        decoder_depth = int(getattr(config, "decoder_num_hidden_layers", 4))
        decoder_heads = int(getattr(config, "decoder_num_attention_heads", self.num_heads))
        self.decoder_mlp_ratio = float(getattr(self.config, "decoder_mlp_ratio", self.mlp_ratio))

        self.decoder_embed = torch.nn.Linear(self.embed_dim, decoder_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # Fixed decoder 2D sin-cos positional embeddings (CLS + patches, filled in post_init)
        self.register_buffer(
            "decoder_pos_embed",
            torch.zeros(1, 1 + self.num_patches, decoder_dim),
            persistent=False,
        )

        self.decoder = torch.nn.ModuleList(
            [
                ViTBlock(
                    embed_dim=decoder_dim,
                    num_heads=decoder_heads,
                    mlp_ratio=self.decoder_mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm_eps=ln_eps,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = torch.nn.LayerNorm(decoder_dim, eps=ln_eps)

        patch_dim = self.num_channels * (patch_size ** 2)
        self.decoder_pred = torch.nn.Linear(decoder_dim, patch_dim)

        # MAE hyperparameters
        self.mask_ratio = float(getattr(config, "mask_ratio", 0.75))
        self.norm_pix_loss = bool(getattr(config, "norm_pix_loss", False))

        # Expose an encoder-only wrapper for SIMCLR / contrastive modes
        self.vit = CustomViTEncoderWrapper(self)
        # Let PreTrainedModel handle weight initialisation via our override
        self.post_init()

    def init_weights(self) -> None:
        """Initialize weights following Hugging Face ViT/ViTMAE style.

        HF ViTMAE uses a plain normal_ with std = config.initializer_range for
        Linear and Conv2d (and Embedding via the generic PreTrainedModel logic).
        """
        std = float(getattr(self.config, "initializer_range", 0.02))

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, torch.nn.Conv2d):
                # Match HF: no special case for patch_embed, use normal_ everywhere
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        # Tokens: follow HF convention (normal_ with same std)
        if hasattr(self, "mask_token") and self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, mean=0.0, std=std)
        if hasattr(self, "cls_token") and self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, mean=0.0, std=std)

    def post_init(self) -> None:
        """Run default HF initialisation, then build fixed sin-cos position embeddings."""
        super().post_init()
        self._init_sincos_pos_embeddings()

    def _init_sincos_pos_embeddings(self) -> None:
        # Compute grid size from image and patch dimensions
        grid_size = self.img_size // self.patch_size

        # Encoder pos embedding: (1, 1 + N, D)
        enc_pos = get_2d_sincos_pos_embed(self.embed_dim, grid_size, cls_token=True)
        enc_pos = enc_pos.unsqueeze(0).to(self.pos_embed.dtype)
        self.pos_embed.copy_(enc_pos)

        # Decoder pos embedding: (1, 1 + N, D_dec) including CLS
        dec_dim = self.decoder_embed.out_features
        dec_pos = get_2d_sincos_pos_embed(dec_dim, grid_size, cls_token=True)
        dec_pos = dec_pos.unsqueeze(0).to(self.decoder_pos_embed.dtype)
        self.decoder_pos_embed.copy_(dec_pos)

    # --- Patchify / unpatchify helpers (MAE-compatible) ---
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H == W == self.img_size, f"Expected square {self.img_size}x{self.img_size}, got {H}x{W}"
        assert H % p == 0 and W % p == 0
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) -> (B, C, H, W)."""
        B, N, D = x.shape
        p = self.patch_size
        C = self.num_channels
        h = w = int(N ** 0.5)
        assert h * w == N, "Number of patches must be a square"
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(B, C, h * p, w * p)
        return imgs

    def _random_mask(
        self,
        B: int,
        L: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample random masking following MAE.

        Returns
        -------
        mask : (B, L) bool
            True for masked positions.
        ids_keep : (B, L_keep) long
            Indices of visible patches.
        ids_restore : (B, L) long
            Indices to restore original order.
        """
        noise = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        L_keep = int(L * (1.0 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :L_keep]

        mask = torch.ones(B, L, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return mask, ids_keep, ids_restore

    def _encode_mae_masked(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        """Encoder-only forward that mirrors HF ViTMAE encoder semantics.

        Used by CustomViTEncoderWrapper so that `model.vit(x)` behaves like
        the underlying HF ViTMAE encoder:

          * Conv patch embedding
          * Add fixed 2D sin-cos positional embeddings to patches
          * Apply the same random masking as in the MAE forward
          * Prepend CLS token (unmasked) before the encoder
          * Run through all encoder blocks + final LayerNorm

        Returns:
            tokens: (B, 1 + N_vis, D) — CLS + visible patch tokens
            hidden_states: optional tuple of intermediate encoder states
        """
        B, C, H, W = pixel_values.shape

        # Patch embedding
        x = self.patch_embed(pixel_values)           # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)             # (B, N, D)
        N_tokens = x.size(1)
        assert N_tokens == self.num_patches, "Patch count mismatch; check image_size/patch_size"

        # Add positional embeddings to patches (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking on patches only (same as MAE forward)
        mask, ids_keep, _ = self._random_mask(B, N_tokens, pixel_values.device)
        x_vis = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )  # (B, N_vis, D)

        # Prepend CLS token (unmasked) before encoder
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)          # (B, 1, D)
        x_enc_in = torch.cat([cls_tokens, x_vis], dim=1)  # (B, 1 + N_vis, D)

        # Transformer encoder + final LayerNorm
        if output_hidden_states:
            hs_list = []
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
                hs_list.append(x_enc)
            enc_out = x_enc
            hidden_states = tuple(hs_list)
        else:
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
            enc_out = x_enc
            hidden_states = None

        enc_out = self.layernorm(enc_out)
        return enc_out, hidden_states

    def _encode_full_image(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        """Encoder-only forward on the full image (no MAE masking).

        This is the plain ViT-style encoder path, suitable for downstream
        probing/UMAP/classification: all patch tokens are kept, a CLS token is
        prepended, and the sequence is passed through the transformer stack
        followed by the final LayerNorm.

        Returns
        -------
        enc_out : (B, 1 + N, D)
            CLS + all patch tokens after encoder + LayerNorm.
        hidden_states : tuple[Tensor] | None
            Optional per-layer encoder outputs, if requested.
        """
        B, C, H, W = pixel_values.shape

        # Patch embedding
        x = self.patch_embed(pixel_values)           # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)             # (B, N, D)
        N_tokens = x.size(1)
        assert N_tokens == self.num_patches, "Patch count mismatch; check image_size/patch_size"

        # Add positional embeddings to all patches (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        # Prepend CLS token (unmasked) before encoder
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)          # (B, 1, D)
        x_enc_in = torch.cat([cls_tokens, x], dim=1)      # (B, 1 + N, D)

        # Transformer encoder + final LayerNorm
        if output_hidden_states:
            hs_list = []
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
                hs_list.append(x_enc)
            enc_out = x_enc
            hidden_states = tuple(hs_list)
        else:
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
            enc_out = x_enc
            hidden_states = None

        enc_out = self.layernorm(enc_out)
        return enc_out, hidden_states

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> CustomViTMAEOutput:
        """MAE forward pass.

        Args
        ----
        pixel_values : (B, C, H, W)
            Input images (already normalized as per processor).
        output_hidden_states : bool
            If True, returns encoder hidden states (we keep the last one).
        """
        B, C, H, W = pixel_values.shape

        # Patch embed
        x = self.patch_embed(pixel_values)           # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)             # (B, N, D) patch tokens
        N_tokens = x.size(1)
        assert N_tokens == self.num_patches, "Patch count mismatch; check image_size/patch_size"

        # Add positional embeddings to patches (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking on patches only
        mask, ids_keep, ids_restore = self._random_mask(B, N_tokens, pixel_values.device)
        x_vis = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

        # Append CLS token (unmasked) before encoder
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_enc_in = torch.cat([cls_tokens, x_vis], dim=1)

        # Encoder: iterate over ViTBlocks
        if output_hidden_states:
            hs_list = []
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
                hs_list.append(x_enc)
            enc_out = x_enc
            hidden_states = tuple(hs_list)
        else:
            x_enc = x_enc_in
            for blk in self.encoder:
                x_enc = blk(x_enc)
            enc_out = x_enc
            hidden_states = None

        # ✅ ADD THIS LINE:
        enc_out = self.layernorm(enc_out)

        # Decoder: keep CLS token, mask/restore patches only
        # enc_out: (B, 1 + L_vis, D)
        x_dec = self.decoder_embed(enc_out)         # (B, 1 + L_vis, D_dec)
        B, L_vis_plus_cls, D_dec = x_dec.shape

        # Separate CLS and visible patch tokens
        x_dec_cls = x_dec[:, :1, :]                 # (B, 1, D_dec)
        x_dec_patches = x_dec[:, 1:, :]             # (B, L_vis, D_dec)
        L_vis = x_dec_patches.size(1)
        L = N_tokens                                # total number of patches
        L_mask = L - L_vis

        # Append mask tokens for missing patches (patch dimension only)
        if L_mask > 0:
            mask_tokens = self.mask_token.expand(B, L_mask, -1)
            x_patches = torch.cat([x_dec_patches, mask_tokens], dim=1)  # (B, L, D_dec) before restore
        else:
            x_patches = x_dec_patches

        # Restore original patch order (CLS is always kept at position 0)
        x_patches = torch.gather(
            x_patches, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec),
        )

        # Concatenate CLS back and add positional embeddings (CLS + patches)
        x_full = torch.cat([x_dec_cls, x_patches], dim=1)    # (B, 1 + L, D_dec)
        x_full = x_full + self.decoder_pos_embed

        for blk in self.decoder:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        # Drop CLS token for reconstruction prediction
        logits = self.decoder_pred(x_full[:, 1:, :])         # (B, L, patch_dim)

        # MAE loss: masked-patch MSE
        with torch.no_grad():
            target = self.patchify(pixel_values)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True, unbiased=False)
                target = (target - mean) / (var + 1e-6).sqrt()

        if mask.any():
            loss = (logits[mask] - target[mask]).pow(2).mean()
        else:
            loss = logits.new_tensor(0.0)

        return CustomViTMAEOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=hidden_states,
        )

class CustomViTMAEModel(torch.nn.Module):
    """Encoder-only view of CustomViTMAEForPreTraining, analogous to HF ViTMAEModel.

    By default this uses the full-image, **unmasked** encoder path suitable
    for downstream probing (UMAP, linear probes, SimCLR heads, etc.).

    If you want to mirror MAE-style masked encoder semantics (CLS + visible
    patches only), pass use_mae_masking=True.
    """

    def __init__(
        self,
        mae_model: "CustomViTMAEForPreTraining",
        use_mae_masking: bool = False,
    ) -> None:
        super().__init__()
        self.mae_model = mae_model
        self.use_mae_masking = use_mae_masking
        # Expose config so downstream code can query hidden_size, num_hidden_layers, etc.
        self.config = mae_model.config

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: "torch.device | str | None" = None,
        use_mae_masking: bool = False,
    ) -> "CustomViTMAEModel":
        """Instantiate a CustomViTMAEModel from a STRADAVIT checkpoint directory.

        This bypasses Hugging Face's PreTrainedModel.from_pretrained logic and
        instead:
          * loads the config via AutoConfig.from_pretrained(model_name_or_path)
          * builds a CustomViTMAEForPreTraining(cfg)
          * loads weights from model.safetensors or pytorch_model.bin
          * wraps the full MAE in CustomViTMAEModel
        """
        # 1) Load config (must match what was used at training time)
        cfg = AutoConfig.from_pretrained(model_name_or_path)

        # 2) Instantiate full MAE model (encoder + decoder)
        mae_full = CustomViTMAEForPreTraining(cfg)

        # 3) Load state dict from safetensors or PyTorch bin
        st_path = os.path.join(model_name_or_path, "model.safetensors")
        pt_path = os.path.join(model_name_or_path, "pytorch_model.bin")

        if os.path.isfile(st_path):
            print(f"[CustomViTMAEModel.from_pretrained] Loading weights from {st_path} (safetensors)")
            state_dict = safe_load_file(st_path)
        elif os.path.isfile(pt_path):
            print(f"[CustomViTMAEModel.from_pretrained] Loading weights from {pt_path} (pytorch_model.bin)")
            state_dict = torch.load(pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found in {model_name_or_path}"
            )

        missing, unexpected = mae_full.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[CustomViTMAEModel.from_pretrained] missing keys: {missing}")
        if unexpected:
            print(f"[CustomViTMAEModel.from_pretrained] unexpected keys: {unexpected}")

        if device is not None:
            mae_full = mae_full.to(device)

        # 4) Wrap in encoder-only model with the requested masking behaviour
        return cls(mae_full, use_mae_masking=use_mae_masking)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> CustomViTEncoderOutput:
        if self.use_mae_masking:
            # CLS + visible patches only, MAE-style
            tokens, hs = self.mae_model._encode_mae_masked(
                pixel_values,
                output_hidden_states=output_hidden_states,
            )
        else:
            # Full image, no masking (plain ViT encoder)
            tokens, hs = self.mae_model._encode_full_image(
                pixel_values,
                output_hidden_states=output_hidden_states,
            )
        return CustomViTEncoderOutput(last_hidden_state=tokens, hidden_states=hs)

class MlpProjector(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 192,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
        self.ln1 = torch.nn.LayerNorm(hidden_dim, eps=1e-6)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.use_batchnorm = bool(use_batchnorm)
        self.bn = (
            torch.nn.BatchNorm1d(out_dim, affine=False, eps=1e-4)
            if self.use_batchnorm
            else torch.nn.Identity()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        return self.bn(x)
        # # ensure BN computes/outputs in fp32 for stable statistics under AMP
        # x_fp32 = x.float()
        # x_bn = self.bn(x_fp32)
        # return x_bn.to(x.dtype)


class Dinov2ForContrastive(torch.nn.Module):
    """Thin contrastive wrapper around a native Hugging Face DINO checkpoint.

    Exposes:
      - `.vit` as the encoder backbone used by the trainer
      - `.projector_head` as the SimCLR/HCL projector
      - `.config` for manifesting and token-readout helpers
    """

    def __init__(self, vit: torch.nn.Module, projector_hidden_dim: int = 2048, projector_out_dim: int = 128):
        super().__init__()
        self.vit = vit
        self.config = getattr(vit, "config", None)
        if self.config is None:
            raise ValueError("Dinov2ForContrastive requires the backbone to expose `.config`.")
        hidden_size = int(getattr(self.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("Dinov2ForContrastive could not infer hidden_size from the DINO config.")
        self.projector_head = MlpProjector(
            hidden_size,
            hidden_dim=int(projector_hidden_dim),
            out_dim=int(projector_out_dim),
            use_batchnorm=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 128,
    ) -> "Dinov2ForContrastive":
        vit = load_hf_dinov2_backbone(model_name_or_path)
        return cls(
            vit=vit,
            projector_hidden_dim=int(projector_hidden_dim),
            projector_out_dim=int(projector_out_dim),
        )
