from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.jobs_state import load_jobs


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rank_rows(
    cache: dict[str, Any],
    *,
    dataset_filter: set[str] | None = None,
    alias_filter: set[str] | None = None,
    mode_filter: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for dataset_label, dataset_payload in cache.items():
        if dataset_filter is not None and str(dataset_label) not in dataset_filter:
            continue
        if not isinstance(dataset_payload, dict):
            continue
        models = dataset_payload.get("models") if isinstance(dataset_payload.get("models"), dict) else {}
        rows = []
        for alias, configs in models.items():
            if alias_filter is not None and str(alias) not in alias_filter:
                continue
            if not isinstance(configs, dict):
                continue
            for key, rec in configs.items():
                if not isinstance(rec, dict) or "macro_f1" not in rec:
                    continue
                mode = str(key).split("|", 1)[0]
                if mode_filter is not None and mode not in mode_filter:
                    continue
                rows.append({
                    "alias": str(alias),
                    "eval_mode": mode,
                    "macro_f1": float(rec.get("macro_f1") or 0.0),
                    "macro_f1_std": float(rec.get("macro_f1_std") or 0.0),
                    "weighted_f1": float(rec.get("weighted_f1") or 0.0),
                    "weighted_f1_std": float(rec.get("weighted_f1_std") or 0.0),
                })
        rows.sort(key=lambda item: item["macro_f1"], reverse=True)
        tables[str(dataset_label)] = rows
    return tables


def summary(job_dir: Path) -> dict[str, Any]:
    job = _load_json(job_dir / "job.json")
    job_id = str(job.get("job_id") or job_dir.name)
    for candidate in load_jobs(PROJECT_ROOT / "pretraining"):
        if str(candidate.get("job_id") or "") == job_id:
            job = candidate
            break
    generated = job.get("generated_config") if isinstance(job.get("generated_config"), dict) else {}
    if not generated:
        generated = job.get("plan") if isinstance(job.get("plan"), dict) else {}
    output_dir = Path(str(generated.get("output_dir") or job_dir / "evaluation_outputs"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    raw_cache_path = str(generated.get("cache_json_path") or "")
    cache_path = Path(raw_cache_path) if raw_cache_path else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    cache = _load_json(cache_path) if cache_path is not None else {}
    dataset_filter = set(str(item) for item in generated.get("dataset_labels", []) or generated.get("datasets", []) or [])
    alias_filter = {
        str(item.get("alias") or "")
        for item in generated.get("models", [])
        if isinstance(item, dict) and str(item.get("alias") or "")
    }
    mode_filter = set(str(item) for item in generated.get("eval_modes", []) or [])
    tables_by_dataset = _rank_rows(
        cache,
        dataset_filter=dataset_filter or None,
        alias_filter=alias_filter or None,
        mode_filter=mode_filter or None,
    )
    return {
        "job": {
            "job_id": job.get("job_id") or job_dir.name,
            "status": job.get("status") or "",
            "cuda_devices": job.get("cuda_devices") or "",
        },
        "output_dir": str(output_dir),
        "cache_path": str(cache_path) if cache_path is not None else "",
        "tables": [{"title": dataset, "rows": rows} for dataset, rows in tables_by_dataset.items()],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default="")
    args = parser.parse_args()
    try:
        payload = summary(Path(args.job_dir).resolve())
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
