from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from pretraining.vit_utils import (
    get_hf_dinov2_image_classification_cls,
    get_hf_dinov2_model_cls,
    load_hf_dinov2_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old STRADA DINO readout with native HF DINO image-classification readout."
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint directory.")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def old_strada_readout(backbone, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = backbone(pixel_values=pixel_values)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled

    sequence_output = outputs.last_hidden_state
    config = getattr(backbone, "config", None)
    if getattr(config, "global_pool", False):
        return sequence_output[:, 1:, :].mean(dim=1)
    return sequence_output[:, 0, :]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_hf_dinov2_config(args.model)
    cfg.num_labels = int(args.num_labels)
    image_size = int(args.image_size or getattr(cfg, "image_size", 518))

    backbone_cls = get_hf_dinov2_model_cls(cfg)
    native_cls = get_hf_dinov2_image_classification_cls(cfg)

    backbone = backbone_cls.from_pretrained(args.model, config=cfg).eval()
    native = native_cls.from_pretrained(
        args.model,
        config=cfg,
        ignore_mismatched_sizes=True,
    ).eval()

    pixel_values = torch.randn(args.batch_size, 3, image_size, image_size)

    with torch.no_grad():
        old_features = old_strada_readout(backbone, pixel_values)

    classifier = getattr(native, "classifier", None)
    if classifier is None:
        raise RuntimeError(f"{native.__class__.__name__} does not expose `.classifier`.")

    print(f"model={args.model}")
    print(f"config_class={cfg.__class__.__module__}.{cfg.__class__.__name__}")
    print(f"config_model_type={getattr(cfg, 'model_type', None)!r}")
    print(f"num_register_tokens={getattr(cfg, 'num_register_tokens', None)!r}")
    print(f"backbone_class={backbone.__class__.__module__}.{backbone.__class__.__name__}")
    print(f"native_classifier_class={native.__class__.__module__}.{native.__class__.__name__}")
    print(f"old_feature_shape={tuple(old_features.shape)}")
    print(f"native_classifier_in_features={getattr(classifier, 'in_features', None)}")

    if old_features.shape[-1] != getattr(classifier, "in_features", None):
        print("READOUT_PARITY=NO")
        print("reason=old pooled feature width differs from native classifier input width")
        return

    probe_head = torch.nn.Linear(old_features.shape[-1], args.num_labels)
    probe_head.load_state_dict(classifier.state_dict())
    with torch.no_grad():
        old_logits = probe_head(old_features)
        native_logits = native(pixel_values=pixel_values).logits

    diff = (old_logits - native_logits).abs()
    print(f"max_abs_logit_diff={float(diff.max().item()):.10g}")
    print(f"mean_abs_logit_diff={float(diff.mean().item()):.10g}")
    print(f"READOUT_PARITY={'YES' if float(diff.max().item()) < 1e-6 else 'NO'}")


if __name__ == "__main__":
    main()
