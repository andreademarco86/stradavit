from __future__ import annotations

from pathlib import Path
import re


KNOWN_DATASET_ALIASES: dict[str, str] = {
    "strada_images_fp32_v5.bin": "default",
    "strada_images_fp32_curated.bin": "curated",
    "reduced.bin": "reduced",
    "crops_pad.bin": "default/cropped/padded",
    "crops_rescale.bin": "default/cropped/rescaled",
    "reduced_crops.bin": "reduced/cropped/rescaled",
}

_DATASET_IDENTIFIER_ALIASES: dict[str, str] = {
    "default": "default",
    "strada-images-fp32-v5": "default",
    "curated": "curated",
    "strada-images-fp32-curated": "curated",
    "reduced": "reduced",
    "crops-pad": "default/cropped/padded",
    "default-cropped-padded": "default/cropped/padded",
    "crops-rescale": "default/cropped/rescaled",
    "default-cropped-rescaled": "default/cropped/rescaled",
    "reduced-crops": "reduced/cropped/rescaled",
    "reduced-cropped-rescaled": "reduced/cropped/rescaled",
}


def _identifier_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def dataset_label_for_identifier(value: str) -> tuple[str, bool]:
    """Resolve filenames plus historical folder/manifest dataset tokens."""
    raw = str(value or "").strip()
    if not raw:
        return ("unnamed", False)

    for candidate in (raw, Path(raw).name, Path(raw).stem):
        label = _DATASET_IDENTIFIER_ALIASES.get(_identifier_key(candidate))
        if label:
            return (label, True)

    return dataset_label_for_basename(Path(raw).name)


def dataset_label_for_basename(basename: str) -> tuple[str, bool]:
    name = str(basename or "").strip()
    if not name:
        return ("unnamed", False)
    if name in KNOWN_DATASET_ALIASES:
        return (KNOWN_DATASET_ALIASES[name], True)
    stem = Path(name).stem
    cleaned = re.sub(r"[_-]+", " ", stem).strip()
    return (cleaned or stem or name, False)


def dataset_alias_slug(label: str, basename: str = "") -> str:
    source = str(label or "").strip() or str(basename or "").strip()
    if not source:
        return "dataset"
    alias, _ = dataset_label_for_identifier(source)
    slug = re.sub(r"[^a-z0-9]+", "-", alias).strip("-")
    return slug or "dataset"


def dataset_hf_cache_root(dataset_bin_path: str) -> str:
    path = Path(str(dataset_bin_path or "")).expanduser()
    parent = path.parent
    if parent.name.lower() == "ssl":
        return str(parent.parent)
    return str(parent)
