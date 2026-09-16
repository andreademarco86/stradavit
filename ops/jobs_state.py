from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def validate_job_id(job_id: str) -> str:
    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    if job_id in {".", ".."} or "/" in job_id or "\\" in job_id or not JOB_ID_RE.match(job_id):
        raise ValueError("job_id must be a safe single path segment")
    return job_id


def job_dir(run_dir: Path, job_id: str) -> Path:
    return run_dir.resolve().parent / "ops" / "jobs" / validate_job_id(job_id)


def jobs_root(run_dir: Path) -> Path:
    return run_dir.resolve().parent / "ops" / "jobs"


def _pid_cmdline_text(pid: str) -> str:
    path = Path("/proc").joinpath(pid, "cmdline")
    try:
        raw = path.read_bytes()
    except Exception:
        return ""
    if not raw:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip().lower()


def is_live_pid(pid: Any) -> bool:
    text = str(pid or "").strip()
    if not text.isdigit():
        return False
    proc_dir = Path("/proc").joinpath(text)
    if not proc_dir.exists():
        return False
    cmdline = _pid_cmdline_text(text)
    if not cmdline:
        return False
    markers = (
        "torchrun",
        "torch.distributed.run",
        "stradavit_trainer.py",
        "stradavit_trainer_configurable.py",
        "pretraining/stradavit_trainer",
        "downstream_eval/llrd/main.py",
        "downstream_eval.llrd",
    )
    return any(marker in cmdline for marker in markers)


def checkpoint_exists(run_root: Path) -> bool:
    ckpt_dir = run_root / "checkpoints"
    return (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists()


def parse_gpu_set(raw: str) -> set[str]:
    values: set[str] = set()
    for part in str(raw or "").split(","):
        item = part.strip()
        if item:
            values.add(item)
    return values


def planned_roots(plan: dict[str, Any]) -> list[str]:
    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    roots: list[str] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        root = stage.get("run_root")
        if root and str(root) not in roots:
            roots.append(str(root))
    return roots


def planned_paths(plan: dict[str, Any]) -> list[str]:
    paths = planned_roots(plan)
    for key in ("work_dir", "output_dir", "cache_json_path", "config_path"):
        value = plan.get(key)
        if value and str(value) not in paths:
            paths.append(str(value))
    return paths


def _pid_owner(pid: str) -> str:
    try:
        out = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
    return out or "unknown"


def _gpu_busy_processes() -> list[dict[str, str]]:
    try:
        map_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    uuid_to_index: dict[str, str] = {}
    for line in map_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            uuid_to_index[parts[1]] = parts[0]
    if not uuid_to_index:
        return []
    try:
        apps_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in apps_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid, gpu_uuid = parts[0], parts[1]
        gpu_id = uuid_to_index.get(gpu_uuid)
        if not pid or not gpu_id:
            continue
        rows.append({"pid": pid, "gpu_id": gpu_id, "owner": _pid_owner(pid)})
    return rows


def status_for(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "").lower()
    job_type = str(job.get("job_type") or "training").lower()
    if job_type == "evaluation":
        progress = load_json(Path(str(job.get("job_dir") or "")) / "strada_sweep_progress.json")
        progress_status = str(progress.get("status") or "").lower()
        if progress_status in {"finished", "failed", "cancelled"}:
            return progress_status
        total = int(progress.get("total") or 0)
        done = int(progress.get("completed") or 0) + int(progress.get("skipped_cached") or 0)
        if total > 0 and done >= total and progress_status not in {"failed", "cancelled"}:
            return "finished"
    if is_live_pid(job.get("pid")):
        return "running"
    if status in {"finished", "cancelled", "failed"}:
        return status
    if job_type == "evaluation" and status == "running" and job.get("pid"):
        return "cancelled"
    if job_type == "evaluation" and status == "planned":
        return "planned"
    roots = [Path(str(root)) for root in job.get("planned_run_roots", []) or [] if root]
    if roots and all(checkpoint_exists(root) for root in roots):
        return "finished"
    if job.get("pid"):
        return "cancelled"
    return status or "draft"


def load_jobs(run_dir: Path) -> list[dict[str, Any]]:
    root = jobs_root(run_dir)
    jobs: list[dict[str, Any]] = []
    if not root.is_dir():
        return jobs
    for path in root.glob("*/job.json"):
        job = load_json(path)
        if not job:
            continue
        job["job_id"] = str(job.get("job_id") or path.parent.name)
        job["job_dir"] = str(path.parent)
        if not str(job.get("pid") or "").strip():
            log_file = str(job.get("log_file") or "").strip()
            pid_path = Path(log_file + ".pid") if log_file else path.parent / "training.log.pid"
            if not pid_path.exists():
                pid_path = path.parent / "evaluation.log.pid"
            try:
                pid_text = pid_path.read_text().strip()
            except Exception:
                pid_text = ""
            if pid_text:
                job["pid"] = pid_text
        resolved_status = status_for(job)
        if str(job.get("status") or "") != resolved_status:
            job["status"] = resolved_status
            job["updated_at"] = utc_now()
            if resolved_status in {"finished", "cancelled", "failed"} and not is_live_pid(job.get("pid")):
                job["pid"] = ""
            try:
                write_json(path, job)
            except Exception:
                pass
        else:
            job["status"] = resolved_status
            if resolved_status in {"finished", "cancelled", "failed"} and job.get("pid") and not is_live_pid(job.get("pid")):
                job["pid"] = ""
                job["updated_at"] = utc_now()
                try:
                    write_json(path, job)
                except Exception:
                    pass
        jobs.append(job)
    jobs.sort(key=lambda item: str(item.get("updated_at") or item.get("started_at") or ""), reverse=True)
    return jobs


def active_roots(run_dir: Path) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for job in load_jobs(run_dir):
        if job.get("status") != "running":
            continue
        progress = load_json(Path(str(job.get("job_dir"))) / "strada_sweep_progress.json")
        root = progress.get("run_root") or job.get("active_run_root")
        if root:
            roots.append({"job_id": job.get("job_id"), "run_root": str(root), "pid": str(job.get("pid") or "")})
    return roots


def _read_cpu_registry(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return rows
    for line in lines:
        parts = line.split("|")
        if len(parts) < 7:
            continue
        pid, job_id, affinity, owner, project, cuda_devices, started_at = parts[:7]
        if not pid or not affinity:
            continue
        rows.append(
            {
                "pid": pid,
                "job_id": job_id,
                "cpu_affinity": affinity,
                "owner": owner,
                "project": project,
                "cuda_devices": cuda_devices,
                "started_at": started_at,
            }
        )
    return rows


def inventory(run_dir: Path, cpu_registry: Path | None = None) -> dict[str, Any]:
    jobs = load_jobs(run_dir)
    if cpu_registry is not None:
        affinity_by_job = {row["job_id"]: row for row in _read_cpu_registry(cpu_registry) if row.get("job_id")}
        affinity_by_pid = {row["pid"]: row for row in _read_cpu_registry(cpu_registry) if row.get("pid")}
        for job in jobs:
            row = affinity_by_job.get(str(job.get("job_id") or "")) or affinity_by_pid.get(str(job.get("pid") or ""))
            if not row:
                continue
            job.setdefault("cpu_affinity", row.get("cpu_affinity", ""))
            job.setdefault("cuda_devices", row.get("cuda_devices", ""))
    return {"jobs": jobs, "active_roots": active_roots(run_dir), "updated_at": utc_now()}


def parse_json_arg(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return {}
    return payload if isinstance(payload, dict) else {}


def validate_launch(
    run_dir: Path,
    job_id: str,
    cuda_devices: str,
    plan: dict[str, Any],
    guided_config: dict[str, Any],
    runtime_config: dict[str, Any],
    job_type: str = "training",
) -> dict[str, Any]:
    requested = parse_gpu_set(cuda_devices)
    conflicts: list[dict[str, Any]] = []
    for job in load_jobs(run_dir):
        if job.get("status") != "running":
            continue
        overlap = sorted(requested & parse_gpu_set(str(job.get("cuda_devices") or "")))
        if overlap:
            conflicts.append({"job_id": job.get("job_id"), "cuda_devices": job.get("cuda_devices"), "overlap": overlap})
    if conflicts:
        raise ValueError(f"GPU selection overlaps with running jobs: {conflicts}")

    allowed_pids = {str(job.get("pid") or "") for job in load_jobs(run_dir) if job.get("status") == "running"}
    allowed_pids.discard("")
    foreign_gpu_users: list[dict[str, str]] = []
    for row in _gpu_busy_processes():
        gpu_id = str(row.get("gpu_id") or "")
        pid = str(row.get("pid") or "")
        if gpu_id not in requested:
            continue
        if pid in allowed_pids:
            continue
        foreign_gpu_users.append({"gpu_id": gpu_id, "pid": pid, "owner": str(row.get("owner") or "unknown")})
    if foreign_gpu_users:
        foreign_gpu_users.sort(key=lambda item: (item.get("gpu_id", ""), item.get("pid", "")))
        raise ValueError(
            "Requested GPU(s) are already in use by non-portal processes: "
            + "; ".join(f"GPU {item['gpu_id']} pid={item['pid']} owner={item['owner']}" for item in foreign_gpu_users[:16])
        )

    if job_type == "training":
        existing = []
        for root in planned_roots(plan):
            root_path = Path(root)
            if root_path.exists():
                existing.append(str(root_path))
        if existing:
            raise ValueError("Planned model folder already exists: " + ", ".join(existing[:8]))

    payload = {
        "job_id": validate_job_id(job_id),
        "job_type": job_type,
        "status": "planned",
        "cuda_devices": cuda_devices,
        "planned_run_roots": planned_roots(plan),
        "planned_paths": planned_paths(plan),
        "plan": plan,
        "guided_config": guided_config,
        "runtime_config": runtime_config,
        "updated_at": utc_now(),
    }
    write_json(job_dir(run_dir, job_id) / "job.json", payload)
    return {"ok": True, "job": payload}


def update_job(run_dir: Path, job_id: str, **updates: Any) -> dict[str, Any]:
    path = job_dir(run_dir, job_id) / "job.json"
    payload = load_json(path)
    payload.update({key: value for key, value in updates.items() if value is not None})
    payload["job_id"] = validate_job_id(job_id)
    payload["updated_at"] = utc_now()
    write_json(path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "active_roots", "validate_launch", "affinity", "started", "stopped"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--pid", default="")
    parser.add_argument("--cuda-devices", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--job-dir", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--cpu-affinity-mode", default="")
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--requested-cpu-cores", default="")
    parser.add_argument("--dataloader-workers", default="")
    parser.add_argument("--cpu-registry", default="")
    parser.add_argument("--plan-json", default="")
    parser.add_argument("--guided-json", default="")
    parser.add_argument("--runtime-json", default="")
    parser.add_argument("--job-type", default="training")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    try:
        if args.command == "inventory":
            payload = inventory(run_dir, Path(args.cpu_registry) if args.cpu_registry else None)
        elif args.command == "active_roots":
            payload = {"active_roots": active_roots(run_dir)}
        elif args.command == "validate_launch":
            plan = parse_json_arg(args.plan_json)
            if not plan:
                raise ValueError("invalid --plan-json payload (expected JSON object from trainer dry plan)")
            payload = validate_launch(
                run_dir,
                args.job_id,
                args.cuda_devices,
                plan,
                parse_json_arg(args.guided_json),
                parse_json_arg(args.runtime_json),
                args.job_type or "training",
            )
        elif args.command == "affinity":
            payload = update_job(
                run_dir,
                args.job_id,
                cpu_affinity_mode=args.cpu_affinity_mode,
                cpu_affinity=args.cpu_affinity,
                requested_cpu_cores=args.requested_cpu_cores,
                dataloader_workers=args.dataloader_workers,
            )
        elif args.command == "started":
            payload = update_job(
                run_dir,
                args.job_id,
                status="running",
                pid=args.pid,
                cuda_devices=args.cuda_devices,
                branch=args.branch,
                log_file=args.log_file,
                job_dir=args.job_dir,
                cpu_affinity_mode=args.cpu_affinity_mode,
                cpu_affinity=args.cpu_affinity,
                requested_cpu_cores=args.requested_cpu_cores,
                dataloader_workers=args.dataloader_workers,
                started_at=utc_now(),
            )
        else:
            payload = update_job(run_dir, args.job_id, status=args.status or "cancelled", pid="", cpu_affinity="")
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
