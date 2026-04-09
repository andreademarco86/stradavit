"""
Example: load a shipped StradaViT checkpoint and extract embeddings.

This mirrors the embedding policy:
  - Use the ViT encoder's `last_hidden_state`
  - Mean-pool patch tokens (drop CLS): `hs[:, 1:, :].mean(dim=1)`

Expected checkpoint layout (from our training scripts):
  <RUN_ROOT>/checkpoints/
    - config.json
    - pytorch_model.bin (or model.safetensors)
    - preprocessor_config.json
    - (optional) tokenizer/feature extractor extras

Usage:
  python3 examples/use_shipped_stradavit_model.py \\
    --checkpoint /path/to/run/checkpoints \\
    --image /path/to/image.png
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import StradaViTModel


def load_model_and_processor(checkpoint_dir: str):
    """
    Loads a StradaViT checkpoint and the matching HF image processor.
    """
    from transformers import ViTImageProcessor, ViTMAEConfig

    config = ViTMAEConfig.from_pretrained(checkpoint_dir)
    processor = ViTImageProcessor.from_pretrained(checkpoint_dir)
    model = StradaViTModel.from_pretrained(checkpoint_dir)
    model.eval()
    return model, processor, config


def load_image(path: str):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to <run_root>/checkpoints (contains config + weights)")
    ap.add_argument("--image", required=True, help="Path to an image file (png/jpg/...)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"--checkpoint must be a directory: {ckpt}")

    device = torch.device(args.device)

    model, processor, config = load_model_and_processor(ckpt)
    model.to(device)

    img = load_image(args.image)
    inputs: dict[str, Any] = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        out = model(pixel_values=pixel_values)
        emb = out.embedding

    print(
        f"Loaded checkpoint: {ckpt}\n"
        f"  model_type={getattr(config, 'model_type', None)} use_dino_encoder={bool(getattr(config, 'use_dino_encoder', False))} "
        f"n_registers={int(getattr(config, 'n_registers', 0) or 0)}\n"
        f"  image_size={int(getattr(config, 'image_size', 0) or 0)} patch_size={int(getattr(config, 'patch_size', 0) or 0)}\n"
        f"Embedding shape: {tuple(emb.shape)} dtype={emb.dtype} device={emb.device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
