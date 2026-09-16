from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

EXCLUDED_METRICS = {
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}

PREFERRED_METRICS = [
    "loss",
    "l2_loss",
    "l1_loss",
    "bl1_loss",
    "grad_norm",
    "grad_norm_preclip",
    "grad_norm_postclip",
    "grad_norm_clip_coef",
    "learning_rate",
    "feat_var",
    "n_views",
    "clr_loss",
    "soft_hcl_loss",
    "hcl_loss",
    "clr_debiased_loss",
    "clr_tau_plus",
    "soft_hcl_tau",
    "soft_hcl_alpha",
    "hcl_beta",
    "hcl_tau_plus",
    "clr_pos_sim_mean",
    "clr_neg_sim_mean",
    "clr_neg_sim_std",
    "clr_keff_est",
    "clr_cos_std",
    "clr_cos_mean_abs",
    "clr_eff_rank",
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_run_path(run_dir: Path, raw: str | None) -> Path:
    value = str(raw or "").strip()
    if not value:
        return run_dir
    path = Path(value)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def normalize_metric_name(tag: str) -> str:
    name = str(tag).strip().replace("\\", "/")
    for prefix in ("train/", "eval/", "metrics/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("/", "_")


def metric_sort_key(name: str) -> tuple[int, int | str]:
    if name in PREFERRED_METRICS:
        return (0, PREFERRED_METRICS.index(name))
    return (1, name)


def planned_epochs_from_manifest(manifest: dict[str, Any], fallback: Any = None) -> float | None:
    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    value = training.get("num_train_epochs", fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def steps_per_epoch_from_manifest(manifest: dict[str, Any]) -> float | None:
    batching = manifest.get("batching") if isinstance(manifest.get("batching"), dict) else {}
    value = batching.get("steps_per_epoch_updates")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def trainer_epoch_scale_from_manifest(manifest: dict[str, Any]) -> float:
    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    if str(training.get("epoch_semantics") or "") != "mega_chunk_local":
        return 1.0

    sources = [
        training,
        manifest.get("data") if isinstance(manifest.get("data"), dict) else {},
        manifest.get("tokens") if isinstance(manifest.get("tokens"), dict) else {},
    ]
    for source in sources:
        try:
            mega_chunk_count = float(source.get("mega_chunk_count"))
        except (TypeError, ValueError, AttributeError):
            continue
        if mega_chunk_count > 0:
            return 1.0 / mega_chunk_count
    return 1.0


def warmup_epoch_from_manifest(manifest: dict[str, Any], planned_epochs: float | None) -> float | None:
    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    optimizer = manifest.get("optimizer") if isinstance(manifest.get("optimizer"), dict) else {}
    batching = manifest.get("batching") if isinstance(manifest.get("batching"), dict) else {}

    try:
        warmup_epochs = float(training.get("warmup_epochs_effective"))
        if warmup_epochs > 0:
            return warmup_epochs
    except (TypeError, ValueError):
        pass

    warmup_steps = None
    for source in (training, optimizer.get("warmup") if isinstance(optimizer.get("warmup"), dict) else {}):
        try:
            warmup_steps = float(source.get("warmup_steps") or source.get("steps"))
            break
        except (TypeError, ValueError):
            continue
    steps_per_epoch = steps_per_epoch_from_manifest(manifest)
    if warmup_steps is not None and steps_per_epoch:
        return warmup_steps / steps_per_epoch

    try:
        ratio = float(training.get("warmup_ratio"))
    except (TypeError, ValueError):
        try:
            ratio = float((optimizer.get("warmup") or {}).get("ratio"))
        except (TypeError, ValueError, AttributeError):
            ratio = 0.0
    if planned_epochs and ratio > 0:
        return float(planned_epochs) * ratio

    return None


def stage_title(stage: dict[str, Any], manifest: dict[str, Any]) -> str:
    index = stage.get("index")
    total = stage.get("total")
    tokens = manifest.get("tokens") if isinstance(manifest.get("tokens"), dict) else {}
    mode_raw = str(stage.get("training_mode") or tokens.get("training_mode") or "")
    mode = {
        "mae_only": "MAE",
        "contrastive_only": "Contrastive",
        "mae_then_contrastive": "MAE+Contrastive",
    }.get(mode_raw, mode_raw.replace("_", " "))
    phase_raw = str(stage.get("stage") or tokens.get("stage") or "")
    phase = {"phase1": "Phase 1", "phase2": "Phase 2", "main": "Main"}.get(phase_raw, phase_raw)
    loss_modes = stage.get("loss_modes")
    if isinstance(loss_modes, list) and loss_modes:
        loss = "+".join(str(item).upper() for item in loss_modes)
    else:
        loss = str(tokens.get("loss") or "").upper()
    bits = []
    if index and total:
        bits.append(f"{index}/{total}")
    if mode:
        bits.append(mode)
    if phase:
        bits.append(phase)
    if loss:
        bits.append(loss)
    data = manifest.get("data") if isinstance(manifest.get("data"), dict) else {}
    dataset_label = str(
        data.get("dataset_label")
        or tokens.get("dataset_label")
        or stage.get("dataset_label")
        or ""
    ).strip()
    if dataset_label:
        bits.append(f"Data: {dataset_label}")
    return " ".join(bits) or str(stage.get("run_root") or "Run")


def read_tensorboard_scalars(logs_dir: Path, manifest: dict[str, Any], max_points: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    event_files = sorted(glob.glob(str(logs_dir / "**" / "events.out.tfevents.*"), recursive=True))
    if not event_files:
        return {}, [f"No TensorBoard event files found in {logs_dir}"]

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        return {}, [f"TensorBoard EventAccumulator import failed: {exc}"]

    try:
        accumulator = event_accumulator.EventAccumulator(
            str(logs_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
    except Exception as exc:
        return {}, [f"TensorBoard reload failed for {logs_dir}: {exc}"]

    tags = list(accumulator.Tags().get("scalars", []) or [])
    if not tags:
        return {}, [f"No scalar tags found in {logs_dir}"]

    raw_series: dict[str, list[Any]] = {}
    for tag in tags:
        name = normalize_metric_name(tag)
        if name in EXCLUDED_METRICS or name in {"epoch", "step"}:
            continue
        try:
            raw_series[name] = list(accumulator.Scalars(tag))
        except Exception as exc:
            errors.append(f"Unable to read scalar {tag}: {exc}")

    trainer_epoch_scale = trainer_epoch_scale_from_manifest(manifest)
    epoch_by_step: dict[int, float] = {}
    for tag in tags:
        if normalize_metric_name(tag) != "epoch":
            continue
        try:
            for event in accumulator.Scalars(tag):
                epoch_by_step[int(event.step)] = float(event.value) * trainer_epoch_scale
        except Exception:
            pass

    steps_per_epoch = steps_per_epoch_from_manifest(manifest)
    metrics: dict[str, Any] = {}
    for name, events in raw_series.items():
        if not events:
            continue
        stride = max(1, len(events) // max(1, int(max_points)))
        selected = events[::stride]
        steps = [int(event.step) for event in selected]
        values = [float(event.value) for event in selected]
        epochs = []
        for step in steps:
            if step in epoch_by_step:
                epochs.append(epoch_by_step[step])
            elif steps_per_epoch:
                epochs.append(float(step) / steps_per_epoch)
            else:
                epochs.append(float(step))
        metrics[name] = {
            "steps": steps,
            "epochs": epochs,
            "values": values,
        }
    return metrics, errors


def plan_stages(run_dir: Path) -> list[dict[str, Any]]:
    plan = load_json(run_dir / "strada_metrics_plan.json")
    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    if stages:
        return [dict(stage) for stage in stages if isinstance(stage, dict)]

    # If run_dir is itself a model folder, derive a single-stage plan directly.
    if (run_dir / "run_manifest.json").exists():
        manifest = load_json(run_dir / "run_manifest.json")
        execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
        tokens = manifest.get("tokens") if isinstance(manifest.get("tokens"), dict) else {}
        loss_token = tokens.get("loss")
        loss_modes = [str(loss_token)] if loss_token else []
        return [
            {
                "index": 1,
                "total": 1,
                "training_mode": execution.get("mode"),
                "stage": execution.get("stage"),
                "loss_modes": loss_modes,
                "run_root": str(run_dir),
                "logs_dir": str(run_dir / "logs"),
            }
        ]

    session_path = run_dir / "strada_run_session.json"
    session_cutoff = session_path.stat().st_mtime if session_path.exists() else 0.0
    manifests = sorted(
        (path for path in (run_dir / "Models").glob("**/run_manifest.json") if path.stat().st_mtime >= session_cutoff),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if not manifests:
        return []
    latest = manifests[0]
    run_root = latest.parent
    manifest = load_json(latest)
    return [
        {
            "index": 1,
            "total": 1,
            "training_mode": (manifest.get("execution") or {}).get("mode") if isinstance(manifest.get("execution"), dict) else None,
            "stage": (manifest.get("execution") or {}).get("stage") if isinstance(manifest.get("execution"), dict) else None,
            "loss_modes": [((manifest.get("tokens") or {}).get("loss"))] if isinstance(manifest.get("tokens"), dict) else [],
            "run_root": str(run_root),
            "logs_dir": str(run_root / "logs"),
        }
    ]


def snapshot(run_dir: Path, max_points: int) -> dict[str, Any]:
    session = load_json(run_dir / "strada_run_session.json")
    stages_in = plan_stages(run_dir)
    stages: list[dict[str, Any]] = []
    errors: list[str] = []

    for fallback_index, stage in enumerate(stages_in, start=1):
        run_root = resolve_run_path(run_dir, stage.get("run_root"))
        logs_dir = resolve_run_path(run_dir, stage.get("logs_dir") or str(run_root / "logs"))
        manifest = load_json(run_root / "run_manifest.json")
        planned_epochs = planned_epochs_from_manifest(manifest, stage.get("planned_epochs"))
        warmup_epoch = warmup_epoch_from_manifest(manifest, planned_epochs)
        metrics, metric_errors = read_tensorboard_scalars(logs_dir, manifest, max_points=max_points)
        stage_errors = list(metric_errors)
        errors.extend(metric_errors)
        available = sorted(metrics.keys(), key=metric_sort_key)

        stages.append(
            {
                "index": int(stage.get("index") or fallback_index),
                "total": int(stage.get("total") or len(stages_in) or 1),
                "title": stage_title({**stage, "index": stage.get("index") or fallback_index, "total": stage.get("total") or len(stages_in) or 1}, manifest),
                "training_mode": stage.get("training_mode"),
                "stage": stage.get("stage"),
                "loss_modes": stage.get("loss_modes") or [],
                "run_root": str(run_root),
                "logs_dir": str(logs_dir),
                "planned_epochs": planned_epochs,
                "warmup_epoch": warmup_epoch,
                "available_metrics": available,
                "metrics": metrics,
                "errors": stage_errors,
            }
        )

    return {
        "session": session,
        "updated_at": utc_now(),
        "stage_count": len(stages),
        "stages": stages,
        "errors": errors[:20],
    }


def emit_frame(payload: dict[str, Any]) -> None:
    print("__FRAME__", flush=True)
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    print("__END__", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-points", type=int, default=1500)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    interval = max(1.0, float(args.interval))
    while True:
        try:
            payload = snapshot(run_dir, max_points=max(100, int(args.max_points)))
        except Exception as exc:
            payload = {
                "session": load_json(run_dir / "strada_run_session.json"),
                "updated_at": utc_now(),
                "stage_count": 0,
                "stages": [],
                "errors": [str(exc)],
            }
        emit_frame(payload)
        if args.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
