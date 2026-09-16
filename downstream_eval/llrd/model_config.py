from __future__ import annotations

from pathlib import PurePosixPath

from pretraining.dataset_registry import dataset_alias_slug


def infer_encoder(model_id: str) -> str:
    text = model_id.lower()
    if "dinov2-small" in text or "dinov2-s" in text:
        return "dinov2-small"
    if "dinov2" in text:
        return "dinov2-base"
    return "vit-mae-base"


def _checkpoint_suffix(model_id: str) -> str | None:
    name = PurePosixPath(model_id).name
    if name.startswith("checkpoint-"):
        return "ckpt" + name.removeprefix("checkpoint-")
    return None


def _run_folder(model_id: str) -> str:
    parts = PurePosixPath(model_id).parts
    if parts and parts[-1] == "checkpoints" and len(parts) >= 2:
        return parts[-2]
    if len(parts) >= 3 and parts[-2] == "results" and parts[-1].startswith("checkpoint-"):
        return parts[-3]
    return PurePosixPath(model_id).name


def _parse_run_folder(folder: str) -> tuple[str, dict[str, str], set[str]]:
    segments = folder.split("--")
    arch_and_first = segments[0] if segments else folder
    if "-mode=" in arch_and_first:
        arch, mode_value = arch_and_first.split("-mode=", 1)
        first_token = f"mode={mode_value}"
        segments = [first_token] + segments[1:]
    else:
        arch = arch_and_first.split("-init=", 1)[0]
    values: dict[str, str] = {}
    flags: set[str] = set()
    for segment in segments:
        if "=" in segment:
            key, value = segment.split("=", 1)
            values[key] = value
        else:
            flags.add(segment)
    return arch, values, flags


def derive_alias(model_id: str) -> str:
    if "/" in model_id and "--" not in model_id and not model_id.endswith((".bin", ".safetensors")):
        return PurePosixPath(model_id).name

    folder = _run_folder(model_id)
    if "-mode=" not in folder:
        raise ValueError(
            "Local model_id values must use the new training run folder format "
            "'<arch>-mode=...--stage=.../.../checkpoints'. "
            f"Got {model_id!r}."
        )
    arch, values, flags = _parse_run_folder(folder)
    if not values and not flags:
        return folder

    parts = [arch]
    for key in ("mode", "stage", "loss"):
        if key in values:
            parts.append(values[key])
    if "phase1_loss" in values:
        parts.append(f"from-{values['phase1_loss']}")
    elif "prephase" in values:
        parts.append(f"pre-{values['prephase']}")
    if "patch" in values:
        parts.append(f"p{values['patch']}")
    if "mask" in values and values.get("mask") not in ("0", "0.0"):
        parts.append(f"m{values['mask']}")
    if "regs" in values:
        parts.append(f"r{values['regs']}")
    if "views" in values:
        parts.append(f"v{values['views']}")
    if "data" in values:
        # Training data is part of a local run's identity. Without it, two
        # otherwise identical runs trained on different binaries collide in
        # the evaluation cache and generated job config.
        parts.append(f"data-{dataset_alias_slug(values['data'])}")
    if "nodinoenc" in flags:
        parts.append("nodinoenc")
    if "noroi" in flags:
        parts.append("noroi")
    if "noaug" in flags:
        parts.append("noaug")
    if "longrun" in flags:
        parts.append("long")
    ckpt = _checkpoint_suffix(model_id)
    if ckpt:
        parts.append(ckpt)
    return "/".join(parts)


def normalize_model_entry(item: dict) -> dict:
    if "model_id" not in item:
        raise ValueError(f"Model entry is missing model_id: {item}")
    missing = sorted({"enabled", "analysis"} - set(item))
    if missing:
        raise ValueError(
            "Model entries must explicitly set enabled and analysis. "
            f"Missing {missing} from {item.get('model_id')!r}."
        )
    handwritten = sorted({"alias", "encoder", "display_name"}.intersection(item))
    if handwritten:
        raise ValueError(
            "Model entries must omit alias, encoder, and display_name; aliases are derived from model_id. "
            f"Remove {handwritten} from {item.get('model_id')!r}."
        )
    model_id = str(item["model_id"])
    normalized = dict(item)
    normalized.setdefault("alias", derive_alias(model_id))
    normalized.setdefault("encoder", infer_encoder(model_id))
    return normalized
