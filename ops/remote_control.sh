#!/usr/bin/env bash
set -euo pipefail

: "${REMOTE_PROJECT_DIR:?REMOTE_PROJECT_DIR is required}"
: "${PRETRAINING_DIR:=pretraining}"
: "${LOG_FILE:=training.log}"
: "${CUDA_VISIBLE_DEVICES_VALUE:=0}"
: "${RAM_LIMIT_GB:=}"
: "${CONTROL_COMMAND:=status}"
: "${LOG_TAIL_LINES:=80}"
: "${CPU_SAMPLE_SECONDS:=0.2}"
: "${RESOURCE_STREAM_INTERVAL:=1}"
: "${METRICS_STREAM_INTERVAL:=5}"
: "${VENV_DIR:=pytorchenv}"
: "${JOB_ID:=}"
: "${MODEL_ID:=}"
: "${DATASET_BIN_DIR:=}"
: "${EVALUATION_ARTIFACT_PATH:=}"

RUN_DIR="${REMOTE_PROJECT_DIR}/${PRETRAINING_DIR}"
JOBS_DIR="${REMOTE_PROJECT_DIR}/ops/jobs"
JOB_DIR=""
if [[ -n "${JOB_ID}" ]]; then
  if [[ ! "${JOB_ID}" =~ ^[A-Za-z0-9_.-]+$ || "${JOB_ID}" == "." || "${JOB_ID}" == ".." ]]; then
    echo "JOB_ID must be a safe single path segment, got ${JOB_ID}" >&2
    exit 1
  fi
  JOB_DIR="${JOBS_DIR}/${JOB_ID}"
  LOG_FILE="${JOB_DIR}/training.log"
  if [[ -f "${JOB_DIR}/job.json" ]]; then
    job_info="$(python3 - "${JOB_DIR}/job.json" <<'PY' 2>/dev/null || true
import json
import sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(json.dumps({"cuda": data.get("cuda_devices") or "", "log_file": data.get("log_file") or ""}))
except Exception:
    pass
PY
)"
    job_cuda="$(python3 - "${job_info}" <<'PY' 2>/dev/null || true
import json
import sys
try:
    print(json.loads(sys.argv[1]).get("cuda") or "")
except Exception:
    pass
PY
)"
    job_log_file="$(python3 - "${job_info}" <<'PY' 2>/dev/null || true
import json
import sys
try:
    print(json.loads(sys.argv[1]).get("log_file") or "")
except Exception:
    pass
PY
)"
    if [[ -n "${job_cuda}" ]]; then
      CUDA_VISIBLE_DEVICES_VALUE="${job_cuda}"
    fi
    if [[ -n "${job_log_file}" ]]; then
      LOG_FILE="${job_log_file}"
    fi
  fi
else
  LOG_FILE="${JOBS_DIR}/_node/resources.log"
fi
PID_FILE="${LOG_FILE}.pid"
MONITOR_PID_FILE="${LOG_FILE}.memory_monitor.pid"
MONITOR_LOG="${LOG_FILE}.memory_monitor.log"
CPU_AFFINITY_FILE="${LOG_FILE}.cpu_affinity"
PROGRESS_FILE="${JOB_DIR}/strada_sweep_progress.json"
METRICS_RUN_DIR="${JOB_DIR}"
METRICS_HELPER="${REMOTE_PROJECT_DIR}/ops/metrics_stream.py"
MODELS_HELPER="${REMOTE_PROJECT_DIR}/ops/models_inventory.py"
EVALUATION_HELPER="${REMOTE_PROJECT_DIR}/ops/evaluation_inventory.py"
EVALUATION_RESULTS_HELPER="${REMOTE_PROJECT_DIR}/ops/evaluation_results.py"
JOBS_HELPER="${REMOTE_PROJECT_DIR}/ops/jobs_state.py"
DATASET_HELPER="${REMOTE_PROJECT_DIR}/ops/dataset_bins.py"
CPU_REGISTRY="${TMPDIR:-/tmp}/strada_cpu_affinity.registry"
CPU_LOCK_DIR="${TMPDIR:-/tmp}/strada_cpu_affinity.lock"

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

release_cpu_affinity_locked() {
  local pid="$1"
  local job_id="${2:-}"
  local tmp_file="${CPU_REGISTRY}.$$"
  local entry_pid entry_job_id affinity owner project cuda_devices started_at
  : > "${tmp_file}"
  if [[ -f "${CPU_REGISTRY}" ]]; then
    while IFS='|' read -r entry_pid entry_job_id affinity owner project cuda_devices started_at; do
      [[ -z "${entry_pid}" ]] && continue
      if [[ "${entry_pid}" != "${pid}" && ( -z "${job_id}" || "${entry_job_id}" != "${job_id}" ) && -d "/proc/${entry_pid}" ]]; then
        printf '%s|%s|%s|%s|%s|%s|%s\n' "${entry_pid}" "${entry_job_id}" "${affinity}" "${owner}" "${project}" "${cuda_devices:-}" "${started_at:-}" >> "${tmp_file}"
      fi
    done < "${CPU_REGISTRY}"
  fi
  mv "${tmp_file}" "${CPU_REGISTRY}"
}

read_training_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

read_assigned_cpu_affinity() {
  if [[ -f "${CPU_AFFINITY_FILE}" ]]; then
    tr -d '[:space:]' < "${CPU_AFFINITY_FILE}"
  fi
}

clear_job_state() {
  local pid="${1:-}"
  if [[ -n "${pid}" || -n "${JOB_ID}" ]]; then
    with_cpu_lock release_cpu_affinity_locked "${pid}" "${JOB_ID}" || true
  fi
  rm -f "${PID_FILE}" "${MONITOR_PID_FILE}" "${CPU_AFFINITY_FILE}"
}

live_training_pid() {
  local pid
  pid="$(read_training_pid || true)"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    printf '%s\n' "${pid}"
    return 0
  fi
  echo "Cleared stale PID file for finished process ${pid}." >&2
  clear_job_state "${pid}"
  return 1
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

process_rss_anon_kb() {
  local pid="$1"
  if [[ ! -r "/proc/${pid}/status" ]]; then
    return 1
  fi
  awk '
    /^RssAnon:/ { print int($2); found=1; exit }
    END { if (!found) exit 1 }
  ' "/proc/${pid}/status" 2>/dev/null
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

sum_process_tree_rss_anon_kb() {
  local total=0
  local pid mem
  for pid in "$@"; do
    mem="$(process_rss_anon_kb "${pid}")" || return 1
    total=$((total + mem))
  done
  echo "${total}"
}

terminate_process_tree() {
  local root_pid="$1"
  local pids=()
  mapfile -t pids < <(collect_process_tree "${root_pid}")
  if ((${#pids[@]} == 0)); then
    echo "No live process tree found for PID ${root_pid}"
    return 0
  fi

  echo "Sending TERM to: ${pids[*]}"
  kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  sleep 15

  mapfile -t pids < <(collect_process_tree "${root_pid}")
  if ((${#pids[@]})); then
    echo "Sending KILL to: ${pids[*]}"
    kill -KILL "${pids[@]}" >/dev/null 2>&1 || true
  fi
}

cleanup_cancelled_evaluation_work_dir() {
  if [[ -z "${JOB_DIR:-}" || ! -f "${JOB_DIR}/job.json" ]]; then
    return 0
  fi

  local python_bin work_dir
  python_bin="$(control_python_bin)"
  work_dir="$("${python_bin}" - "${JOB_DIR}/job.json" <<'PY' 2>/dev/null || true
import json
import pathlib
import sys

job_path = pathlib.Path(sys.argv[1]).resolve()
try:
    payload = json.loads(job_path.read_text())
except Exception:
    raise SystemExit(0)

if str(payload.get("job_type") or "training").lower() != "evaluation":
    raise SystemExit(0)

plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
work_dir = str(plan.get("work_dir") or "").strip()
if not work_dir:
    raise SystemExit(0)

path = pathlib.Path(work_dir).resolve()
job_dir = job_path.parent.resolve()

try:
    path.relative_to(job_dir)
except ValueError:
    raise SystemExit(0)

if path.name != "evaluation_work":
    raise SystemExit(0)

print(path)
PY
)"
  if [[ -n "${work_dir}" && -d "${work_dir}" ]]; then
    echo "Deleting cancelled evaluation work directory: ${work_dir}"
    rm -rf -- "${work_dir}"
  fi
}

expand_cpu_list() {
  local spec="$1"
  local part start end cpu

  [[ -z "${spec}" ]] && return 0
  IFS=',' read -r -a parts <<< "${spec}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == *-* ]]; then
      start="${part%-*}"
      end="${part#*-}"
      for ((cpu=start; cpu<=end; cpu++)); do
        echo "${cpu}"
      done
    else
      echo "${part}"
    fi
  done
}

print_cpu_raw() {
  local assigned_cpu_affinity
  assigned_cpu_affinity="$(read_assigned_cpu_affinity)"
  if [[ ! -r /proc/stat ]]; then
    return 0
  fi

  local cpus=()
  if [[ -n "${assigned_cpu_affinity}" ]]; then
    mapfile -t cpus < <(expand_cpu_list "${assigned_cpu_affinity}")
    if ((${#cpus[@]} == 0)); then
      return 0
    fi
    awk -v cpus="${cpus[*]}" '
      BEGIN {
        split(cpus, wanted, " ");
        for (i in wanted) want["cpu" wanted[i]]=1;
      }
      $1 in want {
        idle=$5+$6;
        total=0;
        for (i=2; i<=NF; i++) total+=$i;
        print $1, idle, total;
      }
    ' /proc/stat > /tmp/strada_cpu_sample_1.$$
  else
    awk '
      $1 ~ /^cpu[0-9]+$/ {
        idle=$5+$6;
        total=0;
        for (i=2; i<=NF; i++) total+=$i;
        print $1, idle, total;
      }
    ' /proc/stat > /tmp/strada_cpu_sample_1.$$
  fi

  sleep "${CPU_SAMPLE_SECONDS}"
  if [[ -n "${assigned_cpu_affinity}" ]]; then
    awk -v cpus="${cpus[*]}" '
      BEGIN {
        split(cpus, wanted, " ");
        for (i in wanted) want["cpu" wanted[i]]=1;
      }
      $1 in want {
        idle=$5+$6;
        total=0;
        for (i=2; i<=NF; i++) total+=$i;
        print $1, idle, total;
      }
    ' /proc/stat > /tmp/strada_cpu_sample_2.$$
  else
    awk '
      $1 ~ /^cpu[0-9]+$/ {
        idle=$5+$6;
        total=0;
        for (i=2; i<=NF; i++) total+=$i;
        print $1, idle, total;
      }
    ' /proc/stat > /tmp/strada_cpu_sample_2.$$
  fi

  awk '
    NR==FNR { idle[$1]=$2; total[$1]=$3; next }
    {
      idle_delta=$2-idle[$1];
      total_delta=$3-total[$1];
      util=0;
      if (total_delta > 0) util=(1 - idle_delta / total_delta) * 100;
      printf "%s %.1f\n", $1, util;
    }
  ' /tmp/strada_cpu_sample_1.$$ /tmp/strada_cpu_sample_2.$$
  rm -f /tmp/strada_cpu_sample_1.$$ /tmp/strada_cpu_sample_2.$$
}

print_ram() {
  local pid="$1"
  local pids=()
  local job_kb job_gb pss_kb pss_gb

  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Training process is not running."
    return 0
  fi

  mapfile -t pids < <(collect_process_tree "${pid}")
  if job_kb="$(sum_process_tree_job_ram_kb "${pids[@]}")"; then
    job_gb="$(awk -v kb="${job_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    echo "Job RAM: ${job_gb} GB across ${#pids[@]} processes"
  else
    echo "Job RAM: unavailable across ${#pids[@]} processes"
  fi

  if pss_kb="$(sum_process_tree_pss_kb "${pids[@]}")"; then
    pss_gb="$(awk -v kb="${pss_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    echo "Job PSS: ${pss_gb} GB across ${#pids[@]} processes"
  else
    echo "Job PSS: unavailable across ${#pids[@]} processes"
  fi
  if [[ -n "${RAM_LIMIT_GB}" ]]; then
    echo "Limit: ${RAM_LIMIT_GB} GB"
  fi
}

print_ram_raw() {
  local pid="$1"
  local pids=()
  local job_kb job_gb pss_kb pss_gb

  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "rss_gb=0.00"
    echo "job_ram_gb=0.00"
    echo "process_tree_pss_gb=0.00"
    echo "node_mem_pressure_gb=0.00"
    echo "system_mem_gb=0.00"
    echo "memory_available=1"
    echo "process_count=0"
    if [[ -n "${RAM_LIMIT_GB}" ]]; then
      echo "limit_gb=${RAM_LIMIT_GB}"
    fi
    return 0
  fi

  mapfile -t pids < <(collect_process_tree "${pid}")
  if job_kb="$(sum_process_tree_job_ram_kb "${pids[@]}")"; then
    job_gb="$(awk -v kb="${job_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    echo "job_ram_gb=${job_gb}"
    echo "system_mem_gb=${job_gb}"
  else
    echo "job_ram_gb=0.00"
    echo "system_mem_gb=0.00"
  fi
  if pss_kb="$(sum_process_tree_pss_kb "${pids[@]}")"; then
    pss_gb="$(awk -v kb="${pss_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    echo "rss_gb=${pss_gb}"
    echo "process_tree_pss_gb=${pss_gb}"
    echo "memory_available=1"
  else
    echo "rss_gb=0.00"
    echo "process_tree_pss_gb=0.00"
    echo "memory_available=0"
  fi
  echo "process_count=${#pids[@]}"
  if [[ -n "${RAM_LIMIT_GB}" ]]; then
    echo "limit_gb=${RAM_LIMIT_GB}"
  fi
}

print_ram_raw_fast() {
  local pid="$1"
  local pids=()
  local job_kb job_gb
  local mem_total_kb mem_available_kb mem_free_kb node_used_kb node_pressure_kb node_used_gb node_pressure_gb node_total_gb

  mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_free_kb="$(awk '/^MemFree:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  [[ "${mem_total_kb}" =~ ^[0-9]+$ ]] || mem_total_kb=0
  [[ "${mem_available_kb}" =~ ^[0-9]+$ ]] || mem_available_kb=0
  [[ "${mem_free_kb}" =~ ^[0-9]+$ ]] || mem_free_kb=0
  if ((mem_total_kb > mem_available_kb)); then
    node_used_kb=$((mem_total_kb - mem_available_kb))
  else
    node_used_kb=0
  fi
  if ((mem_total_kb > mem_free_kb)); then
    node_pressure_kb=$((mem_total_kb - mem_free_kb))
  else
    node_pressure_kb=0
  fi
  node_used_gb="$(awk -v kb="${node_used_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
  node_pressure_gb="$(awk -v kb="${node_pressure_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
  node_total_gb="$(awk -v kb="${mem_total_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"

  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    mapfile -t pids < <(collect_process_tree "${pid}")
    if job_kb="$(sum_process_tree_rss_anon_kb "${pids[@]}")"; then
      job_gb="$(awk -v kb="${job_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
      echo "rss_gb=${job_gb}"
      echo "job_ram_gb=${job_gb}"
      echo "process_tree_pss_gb=0.00"
      echo "memory_available=1"
    else
      echo "rss_gb=0.00"
      echo "job_ram_gb=0.00"
      echo "process_tree_pss_gb=0.00"
      echo "memory_available=0"
    fi
    echo "process_count=${#pids[@]}"
  else
    echo "rss_gb=0.00"
    echo "job_ram_gb=0.00"
    echo "process_tree_pss_gb=0.00"
    echo "memory_available=1"
    echo "process_count=0"
  fi
  echo "node_mem_gb=${node_used_gb}"
  echo "node_mem_pressure_gb=${node_pressure_gb}"
  echo "system_mem_gb=${node_total_gb}"
  if [[ -n "${RAM_LIMIT_GB}" ]]; then
    echo "limit_gb=${RAM_LIMIT_GB}"
  fi
}

print_gpu_raw() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  nvidia-smi \
    -i "${CUDA_VISIBLE_DEVICES_VALUE}" \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null || true
}

print_status() {
  local pid pids=() assigned_cpu_affinity
  pid="$(live_training_pid 2>/dev/null || true)"
  assigned_cpu_affinity="$(read_assigned_cpu_affinity)"

  echo "Host: $(hostname)"
  echo "Run directory: ${METRICS_RUN_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "CUDA devices: ${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "CPU affinity: ${assigned_cpu_affinity:-unset}"
  echo

  if [[ -d "${REMOTE_PROJECT_DIR}/.git" ]]; then
    cd "${REMOTE_PROJECT_DIR}"
    echo "Git: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
    echo "Commit: $(git log -1 --pretty=format:'%h %ci %s')"
    echo
  fi

  if [[ -z "${pid}" ]]; then
    echo "No PID file found: ${PID_FILE}"
  else
    echo "Training PID: ${pid} running"
    mapfile -t pids < <(collect_process_tree "${pid}")
    ps -o pid,ppid,stat,pcpu,pmem,rss,etime,cmd -p "$(IFS=,; echo "${pids[*]}")"
  fi

  echo
  print_ram "${pid}"

  if [[ -f "${MONITOR_LOG}" ]]; then
    echo
    echo "Memory monitor tail:"
    tail -n 8 "${MONITOR_LOG}"
  fi

  if [[ -f "${LOG_FILE}" ]]; then
    echo
    echo "Training log tail:"
    tail -n "${LOG_TAIL_LINES}" "${LOG_FILE}"
  fi
}

print_status_brief() {
  local pid pids=() assigned_cpu_affinity
  pid="$(live_training_pid 2>/dev/null || true)"
  assigned_cpu_affinity="$(read_assigned_cpu_affinity)"

  echo "Host: $(hostname)"
  echo "Run directory: ${METRICS_RUN_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "CUDA devices: ${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "CPU affinity: ${assigned_cpu_affinity:-unset}"
  echo

  if [[ -d "${REMOTE_PROJECT_DIR}/.git" ]]; then
    cd "${REMOTE_PROJECT_DIR}"
    echo "Git: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
    echo "Commit: $(git log -1 --pretty=format:'%h %ci %s')"
    echo
  fi

  if [[ -z "${pid}" ]]; then
    echo "No PID file found: ${PID_FILE}"
  else
    echo "Training PID: ${pid} running"
    mapfile -t pids < <(collect_process_tree "${pid}")
    echo "Process count: ${#pids[@]}"
  fi

  echo
  print_ram "${pid}"
}

print_status_fast() {
  local pid pids=() assigned_cpu_affinity
  pid="$(live_training_pid 2>/dev/null || true)"
  assigned_cpu_affinity="$(read_assigned_cpu_affinity)"

  echo "Host: $(hostname)"
  echo "Run directory: ${METRICS_RUN_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "CUDA devices: ${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "CPU affinity: ${assigned_cpu_affinity:-unset}"
  echo

  if [[ -z "${pid}" ]]; then
    echo "No PID file found: ${PID_FILE}"
  else
    echo "Training PID: ${pid} running"
    mapfile -t pids < <(collect_process_tree "${pid}")
    echo "Process count: ${#pids[@]}"
  fi
}

print_progress_json() {
  if [[ -n "${JOB_DIR}" && -f "${PROGRESS_FILE}" ]]; then
    cat "${PROGRESS_FILE}"
    echo
  else
    echo "{}"
  fi
}

control_python_bin() {
  local python_bin="${REMOTE_PROJECT_DIR}/${VENV_DIR}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="python3"
    else
      python_bin="python"
    fi
  fi
  printf '%s\n' "${python_bin}"
}

case "${CONTROL_COMMAND}" in
  status)
    print_status
    ;;
  status_brief)
    print_status_brief
    ;;
  gpu_raw)
    print_gpu_raw
    ;;
  cpu_raw)
    print_cpu_raw
    ;;
  ram_raw)
    print_ram_raw "$(live_training_pid 2>/dev/null || true)"
    ;;
  resources_raw)
    echo "__STATUS__"
    print_status_fast
    echo "__RAM__"
    print_ram_raw_fast "$(live_training_pid 2>/dev/null || true)"
    echo "__GPU__"
    print_gpu_raw
    echo "__CPU__"
    print_cpu_raw
    echo "__PROGRESS__"
    print_progress_json
    ;;
  resources_stream)
    while true; do
      echo "__FRAME__"
      echo "__STATUS__"
      print_status_fast
      echo "__RAM__"
      print_ram_raw_fast "$(live_training_pid 2>/dev/null || true)"
      echo "__GPU__"
      print_gpu_raw
      echo "__CPU__"
      print_cpu_raw
      echo "__PROGRESS__"
      print_progress_json
      echo "__END__"
      sleep "${RESOURCE_STREAM_INTERVAL}"
    done
    ;;
  tail_stream)
    mkdir -p "$(dirname "${LOG_FILE}")"
    touch "${LOG_FILE}"
    exec tail -n "${LOG_TAIL_LINES}" -F "${LOG_FILE}"
    ;;
  metrics_stream)
    python_bin="$(control_python_bin)"
    exec "${python_bin}" "${METRICS_HELPER}" --run-dir "${METRICS_RUN_DIR}" --interval "${METRICS_STREAM_INTERVAL}"
    ;;
  metrics_raw)
    python_bin="$(control_python_bin)"
    metrics_dir="${METRICS_RUN_DIR}"
    if [[ -n "${MODEL_ID}" ]]; then
      if [[ "${MODEL_ID}" == "." || "${MODEL_ID}" == ".." || "${MODEL_ID}" == */* || "${MODEL_ID}" == *\\* ]]; then
        echo "MODEL_ID must be a safe single folder name, got ${MODEL_ID}" >&2
        exit 1
      fi
      metrics_dir="${RUN_DIR}/Models/${MODEL_ID}"
      if [[ ! -d "${metrics_dir}" ]]; then
        echo "Model folder not found for MODEL_ID=${MODEL_ID}" >&2
        exit 1
      fi
    fi
    exec "${python_bin}" "${METRICS_HELPER}" --run-dir "${metrics_dir}" --once
    ;;
  jobs_inventory)
    python_bin="$(control_python_bin)"
    with_cpu_lock cleanup_cpu_registry_locked || true
    exec "${python_bin}" "${JOBS_HELPER}" inventory --run-dir "${RUN_DIR}" --cpu-registry "${CPU_REGISTRY}"
    ;;
  models_inventory)
    python_bin="$(control_python_bin)"
    active_roots_json="$("${python_bin}" "${JOBS_HELPER}" active_roots --run-dir "${RUN_DIR}" 2>/dev/null || echo '{}')"
    exec "${python_bin}" "${MODELS_HELPER}" --run-dir "${RUN_DIR}" --live-pid "$(live_training_pid 2>/dev/null || true)" --active-roots-json "${active_roots_json}"
    ;;
  evaluation_inventory)
    python_bin="$(control_python_bin)"
    exec "${python_bin}" "${EVALUATION_HELPER}" --run-dir "${RUN_DIR}"
    ;;
  evaluation_results)
    if [[ -z "${JOB_DIR:-}" ]]; then
      echo "JOB_ID is required for evaluation_results." >&2
      exit 1
    fi
    python_bin="$(control_python_bin)"
    exec "${python_bin}" "${EVALUATION_RESULTS_HELPER}" --job-dir "${JOB_DIR}"
    ;;
  dataset_bins)
    python_bin="$(control_python_bin)"
    exec "${python_bin}" "${DATASET_HELPER}" --dataset-dir "${DATASET_BIN_DIR}"
    ;;
  delete_model)
    if [[ -z "${MODEL_ID:-}" ]]; then
      echo "MODEL_ID is required for delete_model." >&2
      exit 1
    fi
    python_bin="$(control_python_bin)"
    active_roots_json="$("${python_bin}" "${JOBS_HELPER}" active_roots --run-dir "${RUN_DIR}" 2>/dev/null || echo '{}')"
    exec "${python_bin}" "${MODELS_HELPER}" --run-dir "${RUN_DIR}" --live-pid "$(live_training_pid 2>/dev/null || true)" --active-roots-json "${active_roots_json}" --delete "${MODEL_ID}"
    ;;
  stop)
    pid="$(read_training_pid || true)"
    if [[ -z "${pid}" ]]; then
      clear_job_state
      echo "No PID file found: ${PID_FILE}"
      exit 1
    fi
    terminate_process_tree "${pid}"
    clear_job_state "${pid}"
    if [[ -n "${JOB_ID}" ]]; then
      python_bin="$(control_python_bin)"
      "${python_bin}" "${JOBS_HELPER}" stopped --run-dir "${RUN_DIR}" --job-id "${JOB_ID}" --status cancelled >/dev/null || true
      cleanup_cancelled_evaluation_work_dir
    fi
    ;;
  *)
    echo "Unknown control command: ${CONTROL_COMMAND}" >&2
    exit 1
    ;;
esac
