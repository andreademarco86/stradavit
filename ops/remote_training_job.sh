#!/usr/bin/env bash
set -euo pipefail

: "${REPO_URL:?REPO_URL is required}"
: "${REMOTE_PROJECT_DIR:?REMOTE_PROJECT_DIR is required}"
: "${BRANCH:?BRANCH is required}"
: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=pytorchenv}"
: "${REQUIREMENTS_FILE:=requirements.txt}"
: "${UPGRADE_PIP:=1}"
: "${PRETRAINING_DIR:=pretraining}"
: "${CUDA_VISIBLE_DEVICES_VALUE:=0}"
: "${MASTER_PORT:=auto}"
: "${TRAINING_ENTRYPOINT:=stradavit_trainer.py}"
: "${LOG_FILE:=training.log}"
: "${CPU_AFFINITY_MODE:=off}"
: "${DATALOADER_NUM_WORKERS:=}"
: "${RAM_LIMIT_GB:=}"
: "${RAM_CHECK_INTERVAL_SECONDS:=10}"
: "${INSTALL_REQUIREMENTS_EVERY_RUN:=0}"
: "${BOOTSTRAP_ONLY:=0}"
: "${SKIP_SETUP_STEPS:=0}"
: "${EXTRA_TORCHRUN_ARGS:=}"
: "${EXTRA_TRAINING_ARGS:=}"
: "${JOB_ID:=}"
: "${GUIDED_CONFIG_JSON:=}"
: "${RUNTIME_CONFIG_JSON:=}"

CPU_REGISTRY="${TMPDIR:-/tmp}/strada_cpu_affinity.registry"
CPU_LOCK_DIR="${TMPDIR:-/tmp}/strada_cpu_affinity.lock"
LAUNCH_LOCK_DIR=""
LAUNCH_LOCK_HELD=0

collect_process_tree() {
  local root_pid="$1"
  ps -eo pid=,ppid= 2>/dev/null | awk -v root="${root_pid}" '
    {
      pid=$1;
      ppid=$2;
      live[pid]=1;
      children[ppid]=children[ppid] " " pid;
    }
    END {
      if (!(root in live)) exit;
      head=1; tail=1; queue[1]=root;
      while (head <= tail) {
        pid=queue[head++];
        if (seen[pid]++) continue;
        print pid;
        n=split(children[pid], kids, " ");
        for (i=1; i<=n; i++) {
          if (kids[i] != "") queue[++tail]=kids[i];
        }
      }
    }
  '
}

with_cpu_lock() {
  local waited=0
  until mkdir "${CPU_LOCK_DIR}" 2>/dev/null; do
    sleep 0.1
    waited=$((waited + 1))
    if ((waited > 100)); then
      echo "Timed out waiting for CPU affinity allocator lock." >&2
      return 1
    fi
  done
  trap 'rmdir "${CPU_LOCK_DIR}" 2>/dev/null || true' RETURN
  set +e
  "$@"
  local rc=$?
  set -e
  rmdir "${CPU_LOCK_DIR}" 2>/dev/null || true
  trap - RETURN
  return "${rc}"
}

acquire_launch_lock() {
  local waited=0
  [[ -n "${LAUNCH_LOCK_DIR}" ]] || return 0
  until mkdir "${LAUNCH_LOCK_DIR}" 2>/dev/null; do
    sleep 0.2
    waited=$((waited + 1))
    if ((waited > 300)); then
      echo "Timed out waiting for launch lock." >&2
      return 1
    fi
  done
  LAUNCH_LOCK_HELD=1
}

release_launch_lock() {
  if [[ "${LAUNCH_LOCK_HELD}" == "1" && -n "${LAUNCH_LOCK_DIR}" ]]; then
    rmdir "${LAUNCH_LOCK_DIR}" 2>/dev/null || true
    LAUNCH_LOCK_HELD=0
  fi
}

cleanup_on_exit() {
  release_launch_lock
}
trap cleanup_on_exit EXIT

cpu_count_for_job() {
  local gpu_count workers
  if [[ -z "${CUDA_VISIBLE_DEVICES_VALUE}" || "${CUDA_VISIBLE_DEVICES_VALUE}" == "-1" ]]; then
    gpu_count=1
  else
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES_VALUE}"
    gpu_count=0
    for device in "${visible_devices[@]}"; do
      device="${device//[[:space:]]/}"
      [[ -n "${device}" ]] && gpu_count=$((gpu_count + 1))
    done
    ((gpu_count > 0)) || gpu_count=1
  fi
  workers="${EFFECTIVE_DATALOADER_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-8}}"
  [[ "${workers}" =~ ^[0-9]+$ ]] || workers=8
  ((workers > 0)) || workers=8
  echo $((gpu_count * workers))
}

cleanup_cpu_registry_locked() {
  local tmp_file="${CPU_REGISTRY}.$$"
  local pid job_id affinity owner project cuda_devices started_at
  : > "${tmp_file}"
  if [[ -f "${CPU_REGISTRY}" ]]; then
    while IFS='|' read -r pid job_id affinity owner project cuda_devices started_at; do
      [[ -z "${pid}" || -z "${job_id}" || -z "${affinity}" || -z "${project}" ]] && continue
      if [[ -d "/proc/${pid}" ]]; then
        printf '%s|%s|%s|%s|%s|%s|%s\n' "${pid}" "${job_id}" "${affinity}" "${owner}" "${project}" "${cuda_devices:-}" "${started_at:-}" >> "${tmp_file}"
      fi
    done < "${CPU_REGISTRY}"
  fi
  mv "${tmp_file}" "${CPU_REGISTRY}"
}

allocate_cpu_affinity_locked() {
  local pid="$1"
  local needed="$2"
  local total_cpus start end cpu affinity owner started_at
  local entry_pid entry_job_id entry_affinity entry_owner entry_project entry_cuda entry_started
  local selected=()
  declare -A reserved=()

  cleanup_cpu_registry_locked
  total_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all 2>/dev/null || echo 1)"
  if ((needed > total_cpus)); then
    echo "Requested ${needed} CPU cores, but only ${total_cpus} are online." >&2
    return 1
  fi

  if [[ -f "${CPU_REGISTRY}" ]]; then
    while IFS='|' read -r entry_pid entry_job_id entry_affinity entry_owner entry_project entry_cuda entry_started; do
      [[ -z "${entry_pid}" || -z "${entry_job_id}" || -z "${entry_affinity}" || -z "${entry_project}" ]] && continue
      IFS=',' read -r -a parts <<< "${entry_affinity}"
      for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [[ -z "${part}" ]] && continue
        if [[ "${part}" == *-* ]]; then
          start="${part%-*}"
          end="${part#*-}"
          for ((cpu=start; cpu<=end; cpu++)); do
            reserved["${cpu}"]=1
          done
        else
          reserved["${part}"]=1
        fi
      done
    done < "${CPU_REGISTRY}"
  fi

  for ((cpu=0; cpu<total_cpus; cpu++)); do
    if [[ -z "${reserved[${cpu}]:-}" ]]; then
      selected+=("${cpu}")
      if ((${#selected[@]} == needed)); then
        break
      fi
    fi
  done

  if ((${#selected[@]} < needed)); then
    echo "Could not allocate requested CPU affinity." >&2
    echo "Requested cores: ${needed}" >&2
    echo "Available unreserved cores: ${#selected[@]}" >&2
    echo "Reserved cores: ${#reserved[@]}" >&2
    echo "Existing Strada reservations (pid|job_id|affinity|owner|project|cuda|started):" >&2
    cat "${CPU_REGISTRY}" >&2
    return 1
  fi

  affinity="$(printf '%s\n' "${selected[@]}" | awk '
    NR == 1 { start=$1; prev=$1; next }
    $1 == prev + 1 { prev=$1; next }
    {
      if (out) out=out ",";
      out = out (start == prev ? start : start "-" prev);
      start=$1; prev=$1;
    }
    END {
      if (NR > 0) {
        if (out) out=out ",";
        out = out (start == prev ? start : start "-" prev);
      }
      print out;
    }
  ')"
  owner="$(whoami 2>/dev/null || echo unknown)"
  started_at="$(date --iso-8601=seconds 2>/dev/null || date)"
  printf '%s|%s|%s|%s|%s|%s|%s\n' "${pid}" "${JOB_ID:-}" "${affinity}" "${owner}" "${REMOTE_PROJECT_DIR}" "${CUDA_VISIBLE_DEVICES_VALUE}" "${started_at}" >> "${CPU_REGISTRY}"
  echo "${affinity}"
}

release_cpu_affinity_locked() {
  local pid="$1"
  local job_id="${2:-}"
  local tmp_file="${CPU_REGISTRY}.$$"
  local entry_pid entry_job_id affinity owner project cuda_devices started_at
  : > "${tmp_file}"
  if [[ -f "${CPU_REGISTRY}" ]]; then
    while IFS='|' read -r entry_pid entry_job_id affinity owner project cuda_devices started_at; do
      [[ -z "${entry_pid}" || -z "${entry_job_id}" || -z "${affinity}" || -z "${project}" ]] && continue
      if [[ "${entry_pid}" != "${pid}" && ( -z "${job_id}" || "${entry_job_id}" != "${job_id}" ) && -d "/proc/${entry_pid}" ]]; then
        printf '%s|%s|%s|%s|%s|%s|%s\n' "${entry_pid}" "${entry_job_id}" "${affinity}" "${owner}" "${project}" "${cuda_devices:-}" "${started_at:-}" >> "${tmp_file}"
      fi
    done < "${CPU_REGISTRY}"
  fi
  mv "${tmp_file}" "${CPU_REGISTRY}"
}

replace_cpu_registry_pid_locked() {
  local old_pid="$1"
  local new_pid="$2"
  local tmp_file="${CPU_REGISTRY}.$$"
  local entry_pid entry_job_id affinity owner project cuda_devices started_at
  : > "${tmp_file}"
  if [[ -f "${CPU_REGISTRY}" ]]; then
    while IFS='|' read -r entry_pid entry_job_id affinity owner project cuda_devices started_at; do
      [[ -z "${entry_pid}" || -z "${entry_job_id}" || -z "${affinity}" || -z "${project}" ]] && continue
      if [[ "${entry_pid}" == "${old_pid}" ]]; then
        printf '%s|%s|%s|%s|%s|%s|%s\n' "${new_pid}" "${entry_job_id}" "${affinity}" "${owner}" "${project}" "${cuda_devices:-}" "${started_at:-}" >> "${tmp_file}"
      elif [[ -d "/proc/${entry_pid}" ]]; then
        printf '%s|%s|%s|%s|%s|%s|%s\n' "${entry_pid}" "${entry_job_id}" "${affinity}" "${owner}" "${project}" "${cuda_devices:-}" "${started_at:-}" >> "${tmp_file}"
      fi
    done < "${CPU_REGISTRY}"
  fi
  mv "${tmp_file}" "${CPU_REGISTRY}"
}

process_pss_kb() {
  local pid="$1"
  if [[ ! -r "/proc/${pid}/smaps_rollup" ]]; then
    return 1
  fi
  awk '/^Pss:/ {print int($2); found=1; exit} END {if (!found) exit 1}' "/proc/${pid}/smaps_rollup" 2>/dev/null
}

process_job_ram_kb() {
  local pid="$1"
  if [[ -r "/proc/${pid}/smaps_rollup" ]]; then
    awk '
      /^Pss_Anon:/ { print int($2); found=1; exit }
      END { if (!found) exit 1 }
    ' "/proc/${pid}/smaps_rollup" 2>/dev/null && return 0
  fi
  if [[ -r "/proc/${pid}/status" ]]; then
    awk '
      /^RssAnon:/ { print int($2); found=1; exit }
      END { if (!found) exit 1 }
    ' "/proc/${pid}/status" 2>/dev/null && return 0
  fi
  return 1
}

sum_process_tree_pss_kb() {
  local total=0
  local pid pss
  for pid in "$@"; do
    pss="$(process_pss_kb "${pid}")" || return 1
    total=$((total + pss))
  done
  echo "${total}"
}

sum_process_tree_job_ram_kb() {
  local total=0
  local pid mem
  for pid in "$@"; do
    mem="$(process_job_ram_kb "${pid}")" || return 1
    total=$((total + mem))
  done
  echo "${total}"
}

terminate_process_tree() {
  local root_pid="$1"
  mapfile -t pids < <(collect_process_tree "${root_pid}")
  if ((${#pids[@]} == 0)); then
    return 0
  fi

  kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  sleep 15
  mapfile -t pids < <(collect_process_tree "${root_pid}")
  if ((${#pids[@]})); then
    kill -KILL "${pids[@]}" >/dev/null 2>&1 || true
  fi
}

monitor_memory() {
  set +e
  local root_pid="$1"
  local limit_kb="$2"
  local interval_seconds="$3"
  local log_file="$4"
  local job_kb job_gb pss_kb pss_gb
  local pids=()

  trap '' HUP
  echo "[MemoryMonitor] pid=${root_pid} limit_kb=${limit_kb} interval=${interval_seconds}s mode=job_anon"
  while kill -0 "${root_pid}" >/dev/null 2>&1; do
    mapfile -t pids < <(collect_process_tree "${root_pid}")
    if ((${#pids[@]} == 0)); then
      break
    fi

    job_kb="$(sum_process_tree_job_ram_kb "${pids[@]}")" || {
      echo "[MemoryMonitor] job_ram_gb=unavailable processes=${#pids[@]} -- /proc memory fields are not readable; cannot enforce RAM limit this cycle"
      sleep "${interval_seconds}"
      continue
    }
    job_gb="$(awk -v kb="${job_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"

    if pss_kb="$(sum_process_tree_pss_kb "${pids[@]}")"; then
      pss_gb="$(awk -v kb="${pss_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    else
      pss_gb="unavailable"
    fi

    echo "[MemoryMonitor] job_ram_gb=${job_gb} process_tree_pss_gb=${pss_gb} processes=${#pids[@]}"
    if ((job_kb > limit_kb)); then
      echo "[MemoryMonitor] Job RAM exceeded limit. Terminating process tree rooted at ${root_pid}." | tee -a "${log_file}"
      terminate_process_tree "${root_pid}"
      exit 0
    fi
    sleep "${interval_seconds}"
  done
  echo "[MemoryMonitor] process exited"
}

derive_nproc_per_node() {
  local devices="$1"
  local count=0
  local device

  if [[ -z "${devices}" || "${devices}" == "-1" ]]; then
    echo "1"
    return 0
  fi

  IFS=',' read -r -a visible_devices <<< "${devices}"
  for device in "${visible_devices[@]}"; do
    device="${device//[[:space:]]/}"
    [[ -n "${device}" ]] && count=$((count + 1))
  done

  if ((count < 1)); then
    echo "1"
  else
    echo "${count}"
  fi
}

derive_master_port() {
  local configured_port="$1"

  if [[ -n "${configured_port}" && "${configured_port}" != "auto" ]]; then
    if [[ ! "${configured_port}" =~ ^[1-9][0-9]*$ ]]; then
      echo "MASTER_PORT must be a positive integer or auto, got ${configured_port}" >&2
      return 1
    fi
    echo "${configured_port}"
    return 0
  fi

  python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

if [[ "${SKIP_SETUP_STEPS}" == "1" ]]; then
  cd "${REMOTE_PROJECT_DIR}" 2>/dev/null || {
    echo "Remote repository is not present at ${REMOTE_PROJECT_DIR}." >&2
    echo "Run setup first to clone/setup the project on the node." >&2
    exit 1
  }
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Virtualenv is not ready at ${REMOTE_PROJECT_DIR}/${VENV_DIR}." >&2
    echo "Run setup before launching training." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
else
  parent_dir="$(dirname "${REMOTE_PROJECT_DIR}")"
  repo_dir="$(basename "${REMOTE_PROJECT_DIR}")"

  mkdir -p "${parent_dir}"
  cd "${parent_dir}"

  if [[ ! -d "${REMOTE_PROJECT_DIR}/.git" ]]; then
    echo "Cloning ${REPO_URL} into ${REMOTE_PROJECT_DIR}"
    git clone "${REPO_URL}" "${repo_dir}"
  fi

  cd "${REMOTE_PROJECT_DIR}"

  echo "Fetching latest refs"
  git fetch origin

  echo "Checking out ${BRANCH}"
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}"
  else
    git checkout -B "${BRANCH}" "origin/${BRANCH}"
  fi

  echo "Resetting ${BRANCH} to origin/${BRANCH}"
  git reset --hard "origin/${BRANCH}"

  created_venv="0"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if ! command -v virtualenv >/dev/null 2>&1; then
      echo "virtualenv is required on the cluster but was not found in PATH." >&2
      echo "Install it or load the cluster module that provides it, then rerun." >&2
      exit 1
    fi

    echo "Creating virtualenv at ${REMOTE_PROJECT_DIR}/${VENV_DIR}"
    virtualenv "${VENV_DIR}" --python="${PYTHON_BIN}"
    created_venv="1"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"

  if [[ "${UPGRADE_PIP}" == "1" ]]; then
    python -m pip install --upgrade pip
  fi

  if [[ "${created_venv}" == "1" || "${INSTALL_REQUIREMENTS_EVERY_RUN}" == "1" ]]; then
    echo "Installing requirements from ${REQUIREMENTS_FILE}"
    python -m pip install -r "${REQUIREMENTS_FILE}"
  fi

  if [[ "${BOOTSTRAP_ONLY}" == "1" ]]; then
    echo "Bootstrap complete. Repo is synced and virtualenv is ready."
    exit 0
  fi
fi

cd "${PRETRAINING_DIR}"
RUN_DIR="${PWD}"
JOBS_DIR="${REMOTE_PROJECT_DIR}/ops/jobs"
JOB_DIR=""
JOB_STATE_HELPER="${REMOTE_PROJECT_DIR}/ops/jobs_state.py"
if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID is required for ops-launched training jobs; run metadata must live under ${JOBS_DIR}/<job_id>." >&2
  exit 1
fi
if [[ ! "${JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ || "${JOB_ID}" == "." || "${JOB_ID}" == ".." ]]; then
  echo "JOB_ID must be a safe single path segment, got ${JOB_ID}" >&2
  exit 1
fi
JOB_DIR="${JOBS_DIR}/${JOB_ID}"
mkdir -p "${JOB_DIR}"
LOG_FILE="${JOB_DIR}/training.log"
LAUNCH_LOCK_DIR="${RUN_DIR}/.strada_launch.lock"
PROGRESS_FILE="${JOB_DIR}/strada_sweep_progress.json"
METRICS_PLAN_FILE="${JOB_DIR}/strada_metrics_plan.json"
SESSION_FILE="${JOB_DIR}/strada_run_session.json"
FATAL_ERROR_FILE="${JOB_DIR}/fatal_error.log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
NPROC_PER_NODE="$(derive_nproc_per_node "${CUDA_VISIBLE_DEVICES_VALUE}")"
RESOLVED_MASTER_PORT="$(derive_master_port "${MASTER_PORT}")"
rm -f "${PROGRESS_FILE}"
rm -f "${METRICS_PLAN_FILE}"
SESSION_ID="$(date +%Y%m%d_%H%M%S)-$$"

write_run_session() {
  local pid_value="${1:-}"
  SESSION_ID="${SESSION_ID}" \
  SESSION_STARTED_AT="${SESSION_STARTED_AT}" \
  SESSION_PID="${pid_value}" \
  SESSION_ENTRYPOINT="${TRAINING_ENTRYPOINT}" \
  SESSION_BRANCH="${BRANCH}" \
  SESSION_LOG_FILE="${LOG_FILE}" \
  SESSION_RUN_DIR="${JOB_DIR:-${PWD}}" \
  SESSION_FILE="${SESSION_FILE}" \
  python - <<'PY'
import json
import os

payload = {
    "session_id": os.environ.get("SESSION_ID", ""),
    "started_at": os.environ.get("SESSION_STARTED_AT", ""),
    "pid": os.environ.get("SESSION_PID", ""),
    "entrypoint": os.environ.get("SESSION_ENTRYPOINT", ""),
    "branch": os.environ.get("SESSION_BRANCH", ""),
    "log_file": os.environ.get("SESSION_LOG_FILE", ""),
    "run_dir": os.environ.get("SESSION_RUN_DIR", ""),
}
with open(os.environ.get("SESSION_FILE", "strada_run_session.json"), "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}
SESSION_STARTED_AT="$(date --iso-8601=seconds 2>/dev/null || date)"
write_run_session ""

if [[ -n "${DATALOADER_NUM_WORKERS}" ]]; then
  export STRADA_DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}"
fi
CPU_AFFINITY_MODE="$(echo "${CPU_AFFINITY_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${CPU_AFFINITY_MODE}" != "off" && "${CPU_AFFINITY_MODE}" != "auto" ]]; then
  echo "CPU_AFFINITY_MODE must be one of: off, auto. Got: ${CPU_AFFINITY_MODE}" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export STRADA_PROGRESS_FILE="${PROGRESS_FILE}"
export STRADA_METRICS_PLAN_FILE="${METRICS_PLAN_FILE}"
export STRADA_RUN_SESSION_FILE="${SESSION_FILE}"
export STRADA_FATAL_ERROR_FILE="${FATAL_ERROR_FILE}"

if [[ -f "${LOG_FILE}" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  archived_log="${LOG_FILE}.${timestamp}.bak"
  echo "Archiving existing log to ${archived_log}"
  mv "${LOG_FILE}" "${archived_log}"
fi

{
  echo "========== Strada Cluster Training =========="
  echo "Started at: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Host: $(hostname)"
  echo "Repository: ${REMOTE_PROJECT_DIR}"
  echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "Commit: $(git rev-parse HEAD)"
  echo "Commit summary: $(git log -1 --pretty=format:'%h %ci %s')"
  echo "Remote origin: $(git config --get remote.origin.url)"
  echo "Remote origin/${BRANCH}: $(git rev-parse "origin/${BRANCH}")"
  echo "Worktree status:"
  git status --short
  echo "============================================="
  echo
} > "${LOG_FILE}"

eval "torchrun_args=(${EXTRA_TORCHRUN_ARGS})"
eval "training_args=(${EXTRA_TRAINING_ARGS})"

acquire_launch_lock
dry_plan_raw="$(python "${TRAINING_ENTRYPOINT}" "${training_args[@]}" --dry-plan-json)"
dry_plan_json="$(python - <<'PY' "${dry_plan_raw}"
import json
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
text = str(raw).strip()
payload = None

if text:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = None

if payload is None and text:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = None

if payload is None:
    raise SystemExit(2)

print(json.dumps(payload, separators=(",", ":")))
PY
)" || {
  echo "Failed to parse dry-plan JSON from ${TRAINING_ENTRYPOINT} --dry-plan-json" >&2
  echo "Raw dry-plan output:" >&2
  printf '%s\n' "${dry_plan_raw}" >&2
  exit 1
}
EFFECTIVE_DATALOADER_NUM_WORKERS="$(python - <<'PY' "${dry_plan_json}" "${DATALOADER_NUM_WORKERS:-8}"
import json
import sys

try:
    plan = json.loads(sys.argv[1])
except Exception:
    plan = {}
configured = sys.argv[2] if len(sys.argv) > 2 else "8"
try:
    configured_int = int(configured)
except Exception:
    configured_int = 8

workers = configured_int
stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
for stage in stages:
    if not isinstance(stage, dict):
        continue
    value = stage.get("dataloader_workers_per_rank")
    try:
        workers = min(workers, int(value))
    except Exception:
        continue

print(max(1, int(workers)))
PY
)" || EFFECTIVE_DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export STRADA_DATALOADER_NUM_WORKERS="${EFFECTIVE_DATALOADER_NUM_WORKERS}"
python "${JOB_STATE_HELPER}" validate_launch \
  --run-dir "${RUN_DIR}" \
  --job-id "${JOB_ID}" \
  --cuda-devices "${CUDA_VISIBLE_DEVICES_VALUE}" \
  --plan-json "${dry_plan_json}" \
  --guided-json "${GUIDED_CONFIG_JSON}" \
  --runtime-json "${RUNTIME_CONFIG_JSON}"

echo "Starting training"
echo "JOB_ID=${JOB_ID}"
echo "Job directory: ${JOB_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${RESOLVED_MASTER_PORT}"
if [[ -n "${EFFECTIVE_DATALOADER_NUM_WORKERS}" ]]; then
  echo "DataLoader workers per rank: ${EFFECTIVE_DATALOADER_NUM_WORKERS}"
  if [[ -n "${DATALOADER_NUM_WORKERS}" && "${EFFECTIVE_DATALOADER_NUM_WORKERS}" != "${DATALOADER_NUM_WORKERS}" ]]; then
    echo "Configured DataLoader workers per rank: ${DATALOADER_NUM_WORKERS}"
  fi
fi
echo "CPU affinity mode: ${CPU_AFFINITY_MODE}"
if [[ -n "${RAM_LIMIT_GB}" ]]; then
  echo "RAM limit: ${RAM_LIMIT_GB} GB job memory"
fi
echo "Log file: ${LOG_FILE}"

command_prefix=()
assigned_cpu_affinity=""
rm -f "${LOG_FILE}.cpu_affinity"
if [[ "${CPU_AFFINITY_MODE}" == "auto" ]]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is unavailable; continuing without CPU affinity."
    CPU_AFFINITY_MODE="off"
  fi
fi

stdio_prefix=()
if command -v stdbuf >/dev/null 2>&1; then
  stdio_prefix=(stdbuf -oL -eL)
fi

ram_limit_kb=""
if [[ -n "${RAM_LIMIT_GB}" ]]; then
  if [[ ! "${RAM_LIMIT_GB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RAM_LIMIT_GB must be a positive integer number of GB, got ${RAM_LIMIT_GB}" >&2
    exit 1
  fi
  if [[ ! "${RAM_CHECK_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RAM_CHECK_INTERVAL_SECONDS must be a positive integer, got ${RAM_CHECK_INTERVAL_SECONDS}" >&2
    exit 1
  fi
  ram_limit_kb=$((RAM_LIMIT_GB * 1024 * 1024))
fi

if [[ "${CPU_AFFINITY_MODE}" == "auto" ]]; then
  cpu_slots="$(cpu_count_for_job)"
  echo "CPU affinity request: ${cpu_slots} cores ($(derive_nproc_per_node "${CUDA_VISIBLE_DEVICES_VALUE}") GPUs * ${EFFECTIVE_DATALOADER_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-8}} workers)"
  if assigned_cpu_affinity="$(with_cpu_lock allocate_cpu_affinity_locked "$$" "${cpu_slots}")"; then
    echo "Assigned CPU affinity: ${assigned_cpu_affinity}"
    if [[ -n "${JOB_ID}" ]]; then
      python "${JOB_STATE_HELPER}" affinity \
        --run-dir "${RUN_DIR}" \
        --job-id "${JOB_ID}" \
        --cpu-affinity-mode "${CPU_AFFINITY_MODE}" \
        --cpu-affinity "${assigned_cpu_affinity}" \
        --requested-cpu-cores "${cpu_slots}" \
        --dataloader-workers "${EFFECTIVE_DATALOADER_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-8}}" >/dev/null || true
    fi
    command_prefix=(taskset -c "${assigned_cpu_affinity}")
  else
    echo "CPU affinity allocation failed; continuing without CPU affinity."
    CPU_AFFINITY_MODE="off"
    assigned_cpu_affinity=""
    cpu_slots=""
  fi
fi
if [[ "${CPU_AFFINITY_MODE}" != "auto" ]]; then
  echo "CPU affinity assignment disabled; letting the OS scheduler place work."
fi

nohup "${command_prefix[@]}" \
  "${stdio_prefix[@]}" \
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${RESOLVED_MASTER_PORT}" \
    "${torchrun_args[@]}" \
    "${TRAINING_ENTRYPOINT}" \
    "${training_args[@]}" \
    >> "${LOG_FILE}" 2>&1 &

pid="$!"
echo "Started training PID ${pid}"
echo "${pid}" > "${LOG_FILE}.pid"
write_run_session "${pid}"
python "${JOB_STATE_HELPER}" started \
  --run-dir "${RUN_DIR}" \
  --job-id "${JOB_ID}" \
  --pid "${pid}" \
  --cuda-devices "${CUDA_VISIBLE_DEVICES_VALUE}" \
  --branch "${BRANCH}" \
  --log-file "${LOG_FILE}" \
  --job-dir "${JOB_DIR}" \
  --cpu-affinity-mode "${CPU_AFFINITY_MODE}" \
  --cpu-affinity "${assigned_cpu_affinity}" \
  --requested-cpu-cores "${cpu_slots:-}" \
  --dataloader-workers "${EFFECTIVE_DATALOADER_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-}}" >/dev/null || true
release_launch_lock
if [[ "${CPU_AFFINITY_MODE}" == "auto" ]]; then
  echo "${assigned_cpu_affinity}" > "${LOG_FILE}.cpu_affinity"
  with_cpu_lock replace_cpu_registry_pid_locked "$$" "${pid}" || {
    echo "Failed to register CPU affinity ${assigned_cpu_affinity} for PID ${pid}" >&2
    with_cpu_lock release_cpu_affinity_locked "$$" "${JOB_ID:-}" || true
    kill -TERM "${pid}" >/dev/null 2>&1 || true
    rm -f "${LOG_FILE}.pid" "${LOG_FILE}.cpu_affinity"
    exit 1
  }
fi

if [[ -n "${RAM_LIMIT_GB}" ]]; then
  monitor_log="${LOG_FILE}.memory_monitor.log"
  (monitor_memory "${pid}" "${ram_limit_kb}" "${RAM_CHECK_INTERVAL_SECONDS}" "${LOG_FILE}") > "${monitor_log}" 2>&1 &
  echo "$!" > "${LOG_FILE}.memory_monitor.pid"
  echo "Started memory monitor PID $!; log: ${monitor_log}"
fi
