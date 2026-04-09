from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class StradaViTOutput:
    embedding: torch.Tensor
    last_hidden_state: torch.Tensor | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None


def _pool_patch_mean(last_hidden_state: torch.Tensor) -> torch.Tensor:
    # Mirror `pretraining/ft_test_llrd.py`: mean over all non-CLS tokens.
    if last_hidden_state.dim() != 3 or last_hidden_state.size(1) < 2:
        raise ValueError(f"Expected (B, T, D) with CLS+patches, got {tuple(last_hidden_state.shape)}")
    return last_hidden_state[:, 1:, :].mean(dim=1)


class StradaViTModel(nn.Module):
    """
    Lightweight encoder-only wrapper that exposes a consistent embedding API for:
      - vanilla ViTMAE checkpoints (any patch size)
      - register-aware / Dinov2Encoder-backed MAE checkpoints

    Embedding policy matches `pretraining/ft_test_llrd.py`:
      embedding = mean over patch tokens (drop CLS).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.config = getattr(backbone, "config", None)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """
        Loads a backbone in a way that is compatible with our checkpoints:
          - If config indicates registers or Dinov2Encoder path, use `ViTMAEWithRegistersModel`.
          - Else use `ViTModel` to avoid MAE random masking/shuffling in downstream usage.
        """
        from transformers import ViTModel, ViTMAEConfig

        config = ViTMAEConfig.from_pretrained(checkpoint_path)
        use_dino_encoder = bool(getattr(config, "use_dino_encoder", False))
        n_registers = int(getattr(config, "n_registers", 0) or 0)

        if use_dino_encoder or n_registers > 0:
            from pretraining.vit_mae_registers import ViTMAEWithRegistersModel

            backbone = ViTMAEWithRegistersModel.from_pretrained(
                checkpoint_path,
                n_registers=n_registers,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        else:
            # ViTModel loads MAE weights with an expected "vit_mae -> vit" type conversion warning.
            backbone = ViTModel.from_pretrained(
                checkpoint_path,
                add_pooling_layer=False,
                **kwargs,
            )
        return cls(backbone=backbone)

    def _forward_backbone(self, pixel_values: torch.Tensor, **kwargs) -> Any:
        """
        Runs the backbone and returns its native outputs.
        For MAE-family backbones, we disable embeddings.random_masking to get a full-image encoding.
        """
        bb = self.backbone
        emb = getattr(bb, "embeddings", None)
        if emb is None or not hasattr(emb, "random_masking"):
            return bb(pixel_values=pixel_values, **kwargs)

        orig_random_masking = emb.random_masking

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
            device = x.device
            mask = x.new_zeros(B, L)
            ids_restore = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            return x, mask, ids_restore

        try:
            import types

            emb.random_masking = types.MethodType(_random_masking_noop, emb)
            return bb(pixel_values=pixel_values, **kwargs)
        finally:
            emb.random_masking = orig_random_masking

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = True,
        **kwargs,
    ) -> StradaViTOutput:
        outputs = self._forward_backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            # Some HF models may return a tuple.
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                last_hidden_state = outputs[0]
            else:
                raise ValueError("Backbone output does not include last_hidden_state")

        emb = _pool_patch_mean(last_hidden_state)
        out = StradaViTOutput(
            embedding=emb,
            last_hidden_state=last_hidden_state,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        return out


class StradaViTForImageClassification(nn.Module):
    """
    Simple classification head on top of `StradaViTModel` embeddings.

    Head policy:
      - LayerNorm (+ optional dropout) + Linear for all MAE-family variants.

    Rationale: consistent ViT fine-tuning protocol and batch-size agnostic normalization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_labels: int,
        class_weights: list[float] | None = None,
        head_norm: str = "ln",  # kept for backward compatibility; must be "ln" or "auto"
        n_registers: int | None = None,  # accepted for call-site compatibility; config remains source of truth
    ):
        super().__init__()
        self.backbone = StradaViTModel.from_pretrained(checkpoint_path)
        self.config = getattr(self.backbone, "config", None)
        self.num_labels = int(num_labels)

        hidden_size = None
        if self.config is not None:
            hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone config.")

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        cfg_n_regs = int(getattr(self.config, "n_registers", 0) or 0) if self.config is not None else 0
        cfg_use_dino = bool(getattr(self.config, "use_dino_encoder", False)) if self.config is not None else False
        if n_registers is not None and int(n_registers) != cfg_n_regs:
            raise ValueError(f"n_registers={int(n_registers)} does not match checkpoint config.n_registers={cfg_n_regs}.")

        if head_norm not in ("auto", "ln"):
            raise ValueError("head_norm must be one of {'ln','auto'} (BatchNorm is disabled).")
        # "auto" is retained for older call sites; it maps to LN unconditionally now.
        head_norm = "ln"

        dropout_prob = float(getattr(self.config, "classifier_dropout_prob", 0.0) or 0.0) if self.config is not None else 0.0
        ln_eps = float(getattr(self.config, "layer_norm_eps", 1e-6) or 1e-6) if self.config is not None else 1e-6

        self.norm = nn.LayerNorm(int(hidden_size), eps=ln_eps)
        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Linear(int(hidden_size), self.num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        out = self.backbone(pixel_values=pixel_values, **kwargs)
        x = out.embedding
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if getattr(self, "class_weights", None) is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Prefer HF's standard output container when available (Trainer-friendly),
        # but keep a dict fallback so this module can be imported without transformers installed.
        try:
            from transformers.modeling_outputs import ImageClassifierOutput  # type: ignore

            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=out.hidden_states,
                attentions=out.attentions,
            )
        except Exception:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": out.hidden_states,
                "attentions": out.attentions,
            }
