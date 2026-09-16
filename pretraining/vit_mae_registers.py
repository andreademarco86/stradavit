import inspect
import torch
import torch.nn as nn
from typing import Optional

from transformers import ViTMAEModel, ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput

# =============================================================================
# Register-aware ViT-MAE: architectural upgrades vs. vanilla ViTMAEModel
# -----------------------------------------------------------------------------
# This wrapper extends the HF ViT-MAE encoder with the following:
#   - Register tokens: inserted after CLS, participate in all encoder blocks,
#     stripped before the decoder; controlled by config.n_registers (0 = off).
#   - Optional patch-embed LayerNorm: apply LayerNorm to token embeddings
#     (use_patch_embed_layer_norm=True) before registers are inserted.
#   - Optional DINOv2 encoder blocks: if config.use_dino_encoder=True, route
#     the embedded tokens through HF's Dinov2Encoder (LayerScale/drop-path/etc),
#     while preserving ViT-MAE masking and decoder behavior.
# These changes are opt-in via the config (primarily when n_registers>0) and
# preserve HF save/load semantics; set n_registers=0 to recover vanilla behavior.
# =============================================================================


class ViTMAEWithRegistersModel(ViTMAEModel):
    """
    ViT-MAE encoder that supports register tokens (inserted after CLS).

    - Registers are part of the encoder stream but removed before the decoder.
    - Masking/shuffling stays unchanged: only patch tokens are masked.
    - With n_registers=0 behavior matches the standard ViTMAEModel.
    """

    def __init__(self, config, n_registers: Optional[int] = None):
        super().__init__(config)
        self.n_registers = int(
            n_registers
            if n_registers is not None
            else getattr(config, "n_registers", 0)
            or 0
        )
        # Persist on config so save_pretrained exports it
        self.config.n_registers = self.n_registers
        self.use_patch_embed_layer_norm = bool(
            getattr(config, "use_patch_embed_layer_norm", False)
        )
        self.layer_norm_eps = getattr(config, "layer_norm_eps", 1e-6)
        self.patch_embed_norm = (
            nn.LayerNorm(self.config.hidden_size, eps=self.layer_norm_eps)
            if self.use_patch_embed_layer_norm
            else None
        )

        self.use_dino_encoder = bool(getattr(config, "use_dino_encoder", False))
        self.hf_dinov2_encoder = None
        self.hf_dinov2_norm = None
        if self.use_dino_encoder:
            self._init_dino_encoder()
            # Avoid carrying a full, unused ViT-MAE encoder stack alongside Dinov2Encoder.
            # Keeping the attribute (as Identity) is safer than deleting it, while removing params.
            self.encoder = nn.Identity()

        if self.n_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.n_registers, self.config.hidden_size)
            )
            nn.init.trunc_normal_(self.register_tokens, std=self.config.initializer_range)
        else:
            self.register_tokens = None

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # HF embeddings handle masking/shuffling and return ids_restore
        if not hasattr(self, "_embedding_forward_params"):
            self._embedding_forward_params = set(inspect.signature(self.embeddings.forward).parameters)
        embedding_kwargs = {"pixel_values": pixel_values}
        if bool_masked_pos is not None and "bool_masked_pos" in self._embedding_forward_params:
            embedding_kwargs["bool_masked_pos"] = bool_masked_pos
        if interpolate_pos_encoding is not None and "interpolate_pos_encoding" in self._embedding_forward_params:
            embedding_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding
        if noise is not None and "noise" in self._embedding_forward_params:
            embedding_kwargs["noise"] = noise
        embedding_output, mask, ids_restore = self.embeddings(**embedding_kwargs)  # [batch, 1 + N_vis, hidden]

        if self.patch_embed_norm is not None:
            embedding_output = self.patch_embed_norm(embedding_output)

        if self.n_registers > 0:
            cls_tok, patch_tokens = embedding_output[:, :1, :], embedding_output[:, 1:, :]
            regs = self.register_tokens.expand(embedding_output.size(0), -1, -1)
            encoder_input = torch.cat([cls_tok, regs, patch_tokens], dim=1)
        else:
            encoder_input = embedding_output

        if self.use_dino_encoder:
            sequence_output, hidden_states, attentions = self._forward_dino_encoder(
                encoder_input,
                output_hidden_states=bool(output_hidden_states),
                output_attentions=bool(output_attentions),
            )
        else:
            encoder_outputs = self.encoder(
                encoder_input,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            sequence_output = encoder_outputs[0]
            hidden_states = encoder_outputs.hidden_states
            attentions = encoder_outputs.attentions

        if self.n_registers > 0:
            # Strip registers before decoder: keep CLS + visible patches
            sequence_output = torch.cat(
                [sequence_output[:, :1, :], sequence_output[:, 1 + self.n_registers :, :]],
                dim=1,
            )
            if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
                hidden_states = [
                    torch.cat([h[:, :1, :], h[:, 1 + self.n_registers :, :]], dim=1) for h in hidden_states
                ]

        if not return_dict:
            return tuple(
                v
                for v in (
                    sequence_output,
                    hidden_states if output_hidden_states else None,
                    attentions if output_attentions else None,
                    mask,
                    ids_restore,
                )
                if v is not None
            )

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states,
            attentions=attentions,
            mask=mask,
            ids_restore=ids_restore,
        )

    def _init_dino_encoder(self) -> None:
        self._init_hf_dinov2_encoder()

    def _init_hf_dinov2_encoder(self) -> None:
        # Lazy import so this file remains usable even if a runtime lacks DINOv2.
        from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

        cfg = Dinov2Config()
        # Copy only attributes that exist on this transformers version's config.
        for k, v in {
            "image_size": int(getattr(self.config, "image_size", 224) or 224),
            "patch_size": int(getattr(self.config, "patch_size", 16) or 16),
            "hidden_size": int(getattr(self.config, "hidden_size", 0) or 0),
            "num_hidden_layers": int(getattr(self.config, "num_hidden_layers", 0) or 0),
            "num_attention_heads": int(getattr(self.config, "num_attention_heads", 0) or 0),
            "mlp_ratio": float(getattr(self.config, "mlp_ratio", 4.0) or 4.0),
            "qkv_bias": bool(getattr(self.config, "qkv_bias", True)),
            "hidden_dropout_prob": float(getattr(self.config, "hidden_dropout_prob", 0.0) or 0.0),
            "attention_probs_dropout_prob": float(getattr(self.config, "attention_probs_dropout_prob", 0.0) or 0.0),
            "drop_path_rate": float(getattr(self.config, "drop_path_rate", 0.0) or 0.0),
            "layer_norm_eps": float(getattr(self.config, "layer_norm_eps", 1e-6) or 1e-6),
            "layerscale_value": getattr(self.config, "layerscale_value", None),
        }.items():
            if v is None:
                continue
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

        if getattr(cfg, "hidden_size", 0) <= 0:
            raise ValueError("Dinov2Config.hidden_size must be set from ViTMAEConfig.hidden_size")
        if getattr(cfg, "num_hidden_layers", 0) <= 0:
            raise ValueError("Dinov2Config.num_hidden_layers must be set from ViTMAEConfig.num_hidden_layers")
        if getattr(cfg, "num_attention_heads", 0) <= 0:
            raise ValueError("Dinov2Config.num_attention_heads must be set from ViTMAEConfig.num_attention_heads")

        self.hf_dinov2_encoder = Dinov2Encoder(cfg)
        self.hf_dinov2_norm = nn.LayerNorm(cfg.hidden_size, eps=float(getattr(cfg, "layer_norm_eps", 1e-6)))

    def _forward_dino_encoder(
        self,
        x: torch.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
    ):
        if self.hf_dinov2_encoder is None or self.hf_dinov2_norm is None:
            raise RuntimeError(
                "HF DINOv2 encoder is not initialized. "
                "Ensure `transformers` includes `dinov2` and that `config.use_dino_encoder=True` only when available."
            )

        enc_out = self.hf_dinov2_encoder(
            x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        x = self.hf_dinov2_norm(enc_out.last_hidden_state)

        hidden_states = None
        if output_hidden_states:
            hs = enc_out.hidden_states
            if hs is None:
                hidden_states = (x,)
            else:
                hidden_states = tuple(hs) + (x,)
        attentions = enc_out.attentions if output_attentions else None
        return x, hidden_states, attentions


class ViTMAEWithRegistersForPreTraining(ViTMAEForPreTraining):
    """
    Pretraining wrapper that swaps the encoder for ViTMAEWithRegistersModel.
    Decoder and loss stay unchanged.
    """

    def __init__(self, config, n_registers: Optional[int] = None):
        super().__init__(config)
        # Replace the default encoder with register-aware version
        self.vit = ViTMAEWithRegistersModel(config, n_registers=n_registers)
