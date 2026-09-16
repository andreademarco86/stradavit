from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretraining.dataset_registry import dataset_label_for_identifier


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def iso_mtime(path: Path) -> str:
    try:
        return _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def safe_model_id(raw: str) -> str:
    model_id = str(raw or "").strip()
    if not model_id:
        raise ValueError("model id is required")
    if model_id in {".", ".."} or "/" in model_id or "\\" in model_id:
        raise ValueError("model id must be a single folder name")
    return model_id


def resolve_run_root(run_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    return (run_dir / path).resolve()


def current_training_root(run_dir: Path, live_pid: str) -> Path | None:
    return None


def active_roots_from_json(run_dir: Path, raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    items = payload.get("active_roots") if isinstance(payload, dict) else []
    roots: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return roots
    for item in items:
        if not isinstance(item, dict):
            continue
        root = resolve_run_root(run_dir, item.get("run_root"))
        if root:
            roots.append({"job_id": str(item.get("job_id") or ""), "run_root": root, "pid": str(item.get("pid") or "")})
    return roots


def has_final_checkpoint(model_dir: Path) -> bool:
    ckpt_dir = model_dir / "checkpoints"
    return (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists()


def label_value(value: Any, mapping: dict[str, str] | None = None) -> str:
    text = "" if value is None else str(value)
    if mapping and text in mapping:
        return mapping[text]
    return text.replace("_", " ").strip().title()


def token_from_folder(folder: str, key: str) -> str | None:
    marker = f"{key}="
    for part in folder.split("--"):
        if marker in part:
            return part.split(marker, 1)[1]
    return None


def training_dataset_label(manifest: dict[str, Any], folder: str) -> str:
    """Return the human-readable training dataset recorded for a model run."""
    tokens = manifest.get("tokens") if isinstance(manifest.get("tokens"), dict) else {}
    data = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}

    value = (
        data.get("dataset_label")
        or tokens.get("dataset_label")
        or token_from_folder(folder, "data")
        or data.get("dataset_mode")
        or tokens.get("dataset_mode")
    )
    if not value:
        return ""
    label, _ = dataset_label_for_identifier(str(value))
    return label_value(label)


def title_from_manifest(manifest: dict[str, Any], folder: str) -> str:
    tokens = manifest.get("tokens") if isinstance(manifest.get("tokens"), dict) else {}
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    data_policy = manifest.get("data_policy") if isinstance(manifest.get("data_policy"), dict) else {}

    branch = label_value(tokens.get("branch") or token_from_folder(folder, "branch"))
    mode = label_value(
        tokens.get("training_mode") or execution.get("mode") or token_from_folder(folder, "mode"),
        {
            "mae_only": "MAE Only",
            "contrastive_only": "Contrastive",
            "mae_then_contrastive": "MAE+Contrastive",
        },
    )
    stage = label_value(
        tokens.get("stage") or execution.get("stage") or token_from_folder(folder, "stage"),
        {"phase1": "Phase 1", "phase2": "Phase 2", "main": "Main"},
    )
    loss = label_value(tokens.get("loss") or token_from_folder(folder, "loss"))
    regs = tokens.get("regs") if tokens.get("regs") is not None else token_from_folder(folder, "regs")
    views = tokens.get("views") if tokens.get("views") is not None else token_from_folder(folder, "views")
    training_data = training_dataset_label(manifest, folder)

    roi = data_policy.get("roi_crop")
    aug = data_policy.get("augmentations")
    if roi is None:
        roi = tokens.get("roi_crop")
    if aug is None:
        aug = tokens.get("augmentations")
    if roi is None:
        roi = "noroi" not in folder.split("--")
    if aug is None:
        aug = "noaug" not in folder.split("--")
    roi_label = "ROI" if bool(roi) else "No ROI"
    aug_label = "Aug" if bool(aug) else "No Aug"

    left = " ".join(part for part in (branch.upper() if branch else "", mode, stage) if part)
    details = []
    if loss:
        details.append(loss)
    if regs is not None:
        details.append(f"regs={regs}")
    if views is not None:
        details.append(f"views={views}")
    details.extend([roi_label, aug_label])
    if training_data:
        details.append(f"Data: {training_data}")
    if left and details:
        return f"{left} | {' | '.join(details)}"
    return left or shorten_folder(folder)


def shorten_folder(folder: str, limit: int = 96) -> str:
    if len(folder) <= limit:
        return folder
    return f"{folder[:44]}...{folder[-44:]}"


def model_record(run_dir: Path, models_dir: Path, model_dir: Path, active_roots: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_json(model_dir / "run_manifest.json")
    active = next((item for item in active_roots if model_dir.resolve() == item["run_root"].resolve()), None)
    is_training = active is not None
    finished = has_final_checkpoint(model_dir)
    status = "Training" if is_training else ("Finished" if finished else "Cancelled")
    return {
        "id": model_dir.name,
        "title": title_from_manifest(manifest, model_dir.name),
        "status": status,
        "folder": model_dir.name,
        "path": str(model_dir),
        "updated_at": iso_mtime(model_dir),
        "updated_ts": model_dir.stat().st_mtime,
        "has_manifest": bool(manifest),
        "has_checkpoint": finished,
        "job_id": str(active.get("job_id") or "") if active else "",
    }


def inventory(run_dir: Path, live_pid: str, active_roots_json: str = "") -> dict[str, Any]:
    models_dir = run_dir / "Models"
    diagnostics = {
        "run_dir": str(run_dir),
        "models_dir": str(models_dir),
        "models_dir_exists": models_dir.exists(),
        "models_dir_is_dir": models_dir.is_dir(),
        "direct_child_count": 0,
    }
    if not models_dir.exists():
        return {"models": [], "errors": [], "diagnostics": diagnostics}
    active_roots = active_roots_from_json(run_dir, active_roots_json)
    fallback_training_root = current_training_root(run_dir, live_pid)
    if fallback_training_root:
        active_roots.append({"job_id": "", "run_root": fallback_training_root, "pid": str(live_pid or "")})
    models: list[dict[str, Any]] = []
    errors: list[str] = []
    children = list(models_dir.iterdir())
    diagnostics["direct_child_count"] = len(children)
    diagnostics["direct_children"] = [child.name for child in children[:20]]
    for child in children:
        if not child.is_dir():
            continue
        try:
            models.append(model_record(run_dir, models_dir, child, active_roots))
        except Exception as exc:
            errors.append(f"{child.name}: {exc}")
    models.sort(key=lambda row: float(row.get("updated_ts") or 0), reverse=True)
    for row in models:
        row.pop("updated_ts", None)
    return {"models": models, "errors": errors, "diagnostics": diagnostics}


def delete_model(run_dir: Path, live_pid: str, model_id: str, active_roots_json: str = "") -> dict[str, Any]:
    model_id = safe_model_id(model_id)
    models_dir = (run_dir / "Models").resolve()
    target = (models_dir / model_id).resolve()
    if not str(target).startswith(str(models_dir) + os.sep):
        raise ValueError("target must be inside Models")
    if not target.is_dir():
        raise ValueError(f"model folder does not exist: {model_id}")
    training_root = current_training_root(run_dir, live_pid)
    active_roots = active_roots_from_json(run_dir, active_roots_json)
    if training_root:
        active_roots.append({"job_id": "", "run_root": training_root, "pid": str(live_pid or "")})
    record = model_record(run_dir, models_dir, target, active_roots)
    if record["status"] == "Training":
        raise ValueError("cannot delete a model that is currently training")
    if record["status"] not in {"Finished", "Cancelled"}:
        raise ValueError(f"model status does not allow deletion: {record['status']}")
    shutil.rmtree(target)
    return {"deleted": True, "model": record}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--live-pid", default="")
    parser.add_argument("--delete", default="")
    parser.add_argument("--active-roots-json", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    try:
        if args.delete:
            payload = delete_model(run_dir, args.live_pid, args.delete, args.active_roots_json)
        else:
            payload = inventory(run_dir, args.live_pid, args.active_roots_json)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
