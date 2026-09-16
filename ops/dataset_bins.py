from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretraining.dataset_registry import dataset_label_for_basename


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def list_dataset_bins(dataset_dir: Path | str) -> dict[str, Any]:
    raw_dir = str(dataset_dir or "").strip()
    models: list[dict[str, Any]] = []
    errors: list[str] = []
    if not raw_dir:
        return {
            "dataset_bin_dir": "",
            "datasets": [],
            "errors": ["Dataset directory is not configured."],
            "updated_at": utc_now(),
        }
    dataset_dir = Path(raw_dir).expanduser()
    if not dataset_dir.exists():
        return {
            "dataset_bin_dir": str(dataset_dir),
            "datasets": [],
            "errors": [f"Dataset directory not found: {dataset_dir}"],
            "updated_at": utc_now(),
        }
    if not dataset_dir.is_dir():
        return {
            "dataset_bin_dir": str(dataset_dir),
            "datasets": [],
            "errors": [f"Dataset path is not a directory: {dataset_dir}"],
            "updated_at": utc_now(),
        }

    try:
        entries = list(dataset_dir.iterdir())
    except Exception as exc:
        return {
            "dataset_bin_dir": str(dataset_dir),
            "datasets": [],
            "errors": [f"Could not scan dataset directory: {exc}"],
            "updated_at": utc_now(),
        }

    for entry in sorted(entries, key=lambda item: item.name.lower()):
        if not entry.is_file() or entry.suffix.lower() != ".bin":
            continue
        label, known = dataset_label_for_basename(entry.name)
        models.append(
            {
                "path": str(entry),
                "basename": entry.name,
                "label": label,
                "known": bool(known),
            }
        )

    models.sort(key=lambda item: (0 if item["known"] else 1, str(item["label"]).lower(), str(item["basename"]).lower()))
    return {
        "dataset_bin_dir": str(dataset_dir),
        "datasets": models,
        "errors": errors,
        "updated_at": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    payload = list_dataset_bins(args.dataset_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
