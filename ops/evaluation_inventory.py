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

from downstream_eval.llrd.config import load_config, validate_config
from downstream_eval.llrd.model_config import normalize_model_entry
from downstream_eval.llrd.settings import DEFAULT_CONFIG_PATH, load_mapping
from ops.models_inventory import load_json, title_from_manifest


def _has_checkpoint(path: Path) -> bool:
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        return True
    for child in path.glob("checkpoint-*"):
        if child.is_dir() and ((child / "model.safetensors").exists() or (child / "pytorch_model.bin").exists()):
            return True
    return False


def _cache_path(config: dict[str, Any]) -> Path:
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    raw = run.get("cache_json_path") or str(Path(str(run.get("output_dir") or "downstream_eval/llrd/outputs")) / "finetune_results_cache_llrd.json")
    path = Path(str(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_cache(config: dict[str, Any]) -> dict[str, Any]:
    path = _cache_path(config)
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _eval_modes(eval_mode: str) -> list[str]:
    return ["linear_probe", "finetune"] if eval_mode == "all" else [eval_mode]


def _coverage(cache: dict[str, Any], datasets: list[dict[str, Any]], aliases: list[str], eval_modes: list[str]) -> dict[str, Any]:
    by_alias: dict[str, Any] = {}
    for alias in aliases:
        cells = []
        complete = 0
        total = 0
        for dataset in datasets:
            label = str(dataset.get("label") or "")
            model_cache = cache.get(label, {}).get("models", {}).get(alias, {})
            for mode in eval_modes:
                total += 1
                has_mode = any(str(key).startswith(f"{mode}|") and isinstance(value, dict) and "macro_f1_curve_mean" in value for key, value in model_cache.items())
                complete += 1 if has_mode else 0
                cells.append({"dataset": label, "eval_mode": mode, "cached": bool(has_mode)})
        by_alias[alias] = {"complete": complete, "total": total, "cells": cells}
    return by_alias


def _baseline_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in config.get("models", []) or []:
        model_id = str(item.get("model_id") or "")
        if model_id.startswith("./Models/") or model_id.startswith("Models/") or "/Models/" in model_id:
            continue
        spec = normalize_model_entry({"model_id": model_id, "enabled": bool(item.get("enabled", True)), "analysis": bool(item.get("analysis", False))})
        rows.append({"alias": spec["alias"], "model_id": model_id, "enabled": bool(item.get("enabled", True)), "analysis": bool(item.get("analysis", False)), "source": "baseline"})
    return rows


def _trained_models(run_dir: Path) -> list[dict[str, Any]]:
    models_dir = run_dir / "Models"
    rows = []
    if not models_dir.is_dir():
        return rows
    for child in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not child.is_dir():
            continue
        checkpoint_dir = child / "checkpoints"
        if not checkpoint_dir.is_dir() or not _has_checkpoint(checkpoint_dir):
            continue
        model_id = f"pretraining/Models/{child.name}/checkpoints"
        try:
            spec = normalize_model_entry({"model_id": model_id, "enabled": True, "analysis": False})
        except ValueError:
            continue
        manifest = load_json(child / "run_manifest.json")
        rows.append(
            {
                "alias": title_from_manifest(manifest, child.name),
                "cache_alias": spec["alias"],
                "model_id": model_id,
                "enabled": True,
                "analysis": False,
                "source": "trained",
            }
        )
    return rows


def inventory(run_dir: Path, config_path: str | None = None) -> dict[str, Any]:
    config = load_mapping(config_path or DEFAULT_CONFIG_PATH)
    cfg = load_config(config_path)
    datasets = [
        {
            "key": dataset.key,
            "label": dataset.label,
            "enabled": dataset.enabled,
            "analysis_order": dataset.analysis_order,
            "analysis_label": dataset.analysis_label,
        }
        for dataset in cfg.datasets
    ]
    trained = _trained_models(run_dir)
    baselines = _baseline_models(config)
    aliases = [row.get("cache_alias") or row["alias"] for row in trained + baselines]
    coverage = _coverage(_load_cache(config), datasets, aliases, _eval_modes(cfg.run.eval_mode))
    return {
        "datasets": datasets,
        "trained_models": trained,
        "baseline_models": baselines,
        "cache_coverage": coverage,
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "cache_path": str(_cache_path(config)),
    }


def _selected_model_map(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = selection.get("models") if isinstance(selection.get("models"), list) else []
    return {str(row.get("model_id") or ""): row for row in rows if isinstance(row, dict) and str(row.get("model_id") or "")}


def write_config(run_dir: Path, output_path: Path, selection: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    config = load_mapping(config_path or DEFAULT_CONFIG_PATH)
    shared_cache_path = _cache_path(config)
    selected_datasets = set(str(item) for item in selection.get("datasets", []) or [])
    selected_models = _selected_model_map(selection)
    if not selected_datasets:
        raise ValueError("Select at least one dataset.")
    if not selected_models:
        raise ValueError("Select at least one model.")

    selected_dataset_rows: list[dict[str, Any]] = []
    for dataset in config.get("datasets", []) or []:
        dataset["enabled"] = str(dataset.get("key") or "") in selected_datasets
        if dataset["enabled"]:
            selected_dataset_rows.append(dataset)

    run = config.setdefault("run", {})
    run["eval_mode"] = str(selection.get("eval_mode") or "all")
    eval_modes = _eval_modes(str(run["eval_mode"]))
    normalized_rows = []
    for model_id, row in selected_models.items():
        run_model = bool(row.get("run", True))
        analysis_model = bool(row.get("analysis", False))
        spec = normalize_model_entry({"model_id": model_id, "enabled": run_model, "analysis": analysis_model})
        if analysis_model and not run_model:
            coverage = _coverage(_load_cache(config), selected_dataset_rows, [spec["alias"]], eval_modes)
            if not coverage.get(spec["alias"], {}).get("complete") == coverage.get(spec["alias"], {}).get("total"):
                analysis_model = False
        if run_model or analysis_model:
            normalized_rows.append({"model_id": model_id, "enabled": run_model, "analysis": analysis_model})
    config["models"] = normalized_rows
    if not config["models"]:
        raise ValueError("Select at least one model to run, or one analysis-only model with complete cached results.")
    run["dataset_mode"] = "all"
    run.pop("num_workers", None)
    work_dir = output_path.parent / "evaluation_work"
    run["work_dir"] = str(work_dir)
    run["output_dir"] = str(work_dir / "outputs")
    run["hf_cache_path"] = str(work_dir / "hf_cache")
    for dataset in config.get("datasets", []) or []:
        dataset["index_cache_dir"] = str(work_dir / "dataset_index_cache")
    if not run.get("cache_json_path") or not Path(str(run["cache_json_path"])).is_absolute():
        run["cache_json_path"] = str(shared_cache_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    cfg = load_config(str(output_path))
    validate_config(cfg)
    return {
        "config_path": str(output_path),
        "work_dir": str(run["work_dir"]),
        "output_dir": str(run["output_dir"]),
        "cache_json_path": str(run["cache_json_path"]),
        "datasets": [dataset.key for dataset in cfg.selected_datasets()],
        "dataset_labels": [dataset.label for dataset in cfg.selected_datasets()],
        "models": [{"alias": model.alias, "model_id": model.model_id, "analysis": model.analysis} for model in cfg.selected_models()],
        "eval_modes": cfg.selected_eval_modes(),
        "lr_config_count": len(cfg.run.lr_configs),
        "total_records": len(cfg.selected_datasets()) * len(cfg.selected_models()) * len(cfg.selected_eval_modes()) * max(1, len(cfg.run.lr_configs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--write-config", default="")
    parser.add_argument("--selection-json", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    try:
        if args.write_config:
            selection = json.loads(args.selection_json or "{}")
            payload = write_config(run_dir, Path(args.write_config), selection, args.config or None)
        else:
            payload = inventory(run_dir, args.config or None)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
