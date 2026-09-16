#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-}"
CONFIG_FILE="${STRADA_OPS_CONFIG:-}"

usage() {
  cat >&2 <<EOF
Usage: $0 [setup|launch|evaluation_launch|branches|status|status_brief|stop|gpu_raw|cpu_raw|ram_raw|resources_raw|resources_stream|tail_stream|metrics_stream|metrics_raw|jobs_inventory|models_inventory|evaluation_inventory|evaluation_results|dataset_bins|delete_model]
EOF
}

if [[ -z "${COMMAND}" ]]; then
  usage
  exit 1
fi

case "${COMMAND}" in
  setup|launch|evaluation_launch|branches|status|status_brief|stop|gpu_raw|cpu_raw|ram_raw|resources_raw|resources_stream|tail_stream|metrics_stream|metrics_raw|jobs_inventory|models_inventory|evaluation_inventory|evaluation_results|dataset_bins|delete_model)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing STRADA_OPS_CONFIG runtime config." >&2
  echo "The Streamlit backend should provide STRADA_OPS_CONFIG." >&2
  exit 1
fi

# Defaults for optional Bash arrays before sourcing user config.
EXTRA_TORCHRUN_ARGS=()
EXTRA_TRAINING_ARGS=()

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

GATEWAY_USER="${GATEWAY_USER:-${CLUSTER_USER:-}}"

: "${GATEWAY_USER:?Set CLUSTER_USER or GATEWAY_USER in ${CONFIG_FILE}}"
: "${GATEWAY_HOST:?Set GATEWAY_HOST in ${CONFIG_FILE}}"
: "${TARGET_HOST:?Set TARGET_HOST in ${CONFIG_FILE}}"
: "${REMOTE_PROJECT_DIR:?Set REMOTE_PROJECT_DIR in ${CONFIG_FILE}}"

gateway="${GATEWAY_USER}@${GATEWAY_HOST}"
target="${TARGET_HOST}"

gateway_ssh() {
  ssh -tt "${gateway}" "$@"
}

base64_inline() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

build_training_env() {
  : "${REPO_URL:?Set REPO_URL in ${CONFIG_FILE}}"
  : "${BRANCH:?Set BRANCH in ${CONFIG_FILE}}"

  remote_env=(
    "REPO_URL=${REPO_URL}"
    "REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR}"
    "BRANCH=${BRANCH}"
    "PYTHON_BIN=${PYTHON_BIN:-python3}"
    "VENV_DIR=${VENV_DIR:-pytorchenv}"
    "REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements.txt}"
    "UPGRADE_PIP=${UPGRADE_PIP:-1}"
    "PRETRAINING_DIR=${PRETRAINING_DIR:-pretraining}"
    "DATASET_BIN_DIR=${DATASET_BIN_DIR:-}"
    "CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0}"
    "MASTER_PORT=${MASTER_PORT:-auto}"
    "TRAINING_ENTRYPOINT=${TRAINING_ENTRYPOINT:-stradavit_trainer.py}"
    "LOG_FILE=${LOG_FILE:-training.log}"
    "CPU_AFFINITY_MODE=${CPU_AFFINITY_MODE:-off}"
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-}"
    "RAM_LIMIT_GB=${RAM_LIMIT_GB:-}"
    "RAM_CHECK_INTERVAL_SECONDS=${RAM_CHECK_INTERVAL_SECONDS:-10}"
    "LOG_TAIL_LINES=${LOG_TAIL_LINES:-30}"
    "INSTALL_REQUIREMENTS_EVERY_RUN=${INSTALL_REQUIREMENTS_EVERY_RUN:-0}"
    "JOB_ID=${JOB_ID:-}"
    "GUIDED_CONFIG_JSON=${GUIDED_CONFIG_JSON:-}"
    "RUNTIME_CONFIG_JSON=${RUNTIME_CONFIG_JSON:-}"
    "BOOTSTRAP_ONLY=0"
  )

  extra_torchrun=""
  extra_training=""
  if ((${#EXTRA_TORCHRUN_ARGS[@]})); then
    printf -v extra_torchrun '%q ' "${EXTRA_TORCHRUN_ARGS[@]}"
  fi
  if ((${#EXTRA_TRAINING_ARGS[@]})); then
    printf -v extra_training '%q ' "${EXTRA_TRAINING_ARGS[@]}"
  fi
  remote_env+=("EXTRA_TORCHRUN_ARGS=${extra_torchrun}")
  remote_env+=("EXTRA_TRAINING_ARGS=${extra_training}")
}

build_control_env() {
  remote_env=(
    "CONTROL_COMMAND=${COMMAND}"
    "REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR}"
    "PRETRAINING_DIR=${PRETRAINING_DIR:-pretraining}"
    "DATASET_BIN_DIR=${DATASET_BIN_DIR:-}"
    "LOG_FILE=${LOG_FILE:-training.log}"
    "CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0}"
    "RAM_LIMIT_GB=${RAM_LIMIT_GB:-}"
    "LOG_TAIL_LINES=${LOG_TAIL_LINES:-80}"
    "RESOURCE_STREAM_INTERVAL=${RESOURCE_STREAM_INTERVAL:-1}"
    "METRICS_STREAM_INTERVAL=${METRICS_STREAM_INTERVAL:-5}"
    "VENV_DIR=${VENV_DIR:-pytorchenv}"
    "MODEL_ID=${MODEL_ID:-}"
    "JOB_ID=${JOB_ID:-}"
    "EVALUATION_ARTIFACT_PATH=${EVALUATION_ARTIFACT_PATH:-}"
  )
}

run_on_target() {
  local inner_command="$1"
  shift || true
  local entry key value export_script="" target_command=""
  local export_b64 command_b64 gateway_command
  for entry in "$@"; do
    key="${entry%%=*}"
    value="${entry#*=}"
    printf -v export_script '%sexport %s=%q\n' "${export_script}" "${key}" "${value}"
  done
  export_b64="$(base64_inline "${export_script}")"
  command_b64="$(base64_inline "${inner_command}")"
  printf -v target_command "eval \"\$(printf '%%s' '%s' | base64 -d)\"; exec bash -lc \"\$(printf '%%s' '%s' | base64 -d)\"" "${export_b64}" "${command_b64}"
  printf -v gateway_command 'ssh -tt %q %q' "${target}" "${target_command}"
  gateway_ssh "${gateway_command}"
}

if [[ "${COMMAND}" == "setup" ]]; then
  build_training_env
  remote_env+=("BOOTSTRAP_ONLY=1")
  bootstrap_launch='
set -euo pipefail
parent_dir="$(dirname "${REMOTE_PROJECT_DIR}")"
repo_dir="$(basename "${REMOTE_PROJECT_DIR}")"
mkdir -p "${parent_dir}"
cd "${parent_dir}"
if [[ ! -d "${REMOTE_PROJECT_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${repo_dir}"
fi
cd "${REMOTE_PROJECT_DIR}"
git fetch origin
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "${BRANCH}"
else
  git checkout -B "${BRANCH}" "origin/${BRANCH}"
fi
git reset --hard "origin/${BRANCH}"
exec bash ops/remote_training_job.sh
'
  printf 'Running remote setup on %s through %s\n' "${target}" "${gateway}"
  run_on_target "${bootstrap_launch}" "${remote_env[@]}"
  exit 0
fi

if [[ "${COMMAND}" == "launch" ]]; then
  build_training_env
  remote_env+=("SKIP_SETUP_STEPS=1")
  launch_entry='
set -euo pipefail
cd "${REMOTE_PROJECT_DIR}" 2>/dev/null || {
  echo "Remote repository is not present at ${REMOTE_PROJECT_DIR}."
  echo "Run setup first to clone/setup the project on the node."
  exit 1
}
exec bash ops/remote_training_job.sh
'
  printf 'Launching remote training on %s through %s\n' "${target}" "${gateway}"
  run_on_target "${launch_entry}" "${remote_env[@]}"
  exit 0
fi

if [[ "${COMMAND}" == "evaluation_launch" ]]; then
  build_training_env
  remote_env+=("SKIP_SETUP_STEPS=1")
  remote_env+=("CPU_AFFINITY_MODE=off")
  remote_env+=("EVALUATION_SELECTION_JSON=${EVALUATION_SELECTION_JSON:-}")
  eval_launch_entry='
set -euo pipefail
cd "${REMOTE_PROJECT_DIR}" 2>/dev/null || {
  echo "Remote repository is not present at ${REMOTE_PROJECT_DIR}."
  echo "Run setup first to clone/setup the project on the node."
  exit 1
}
exec bash ops/remote_evaluation_job.sh
'
  printf 'Launching remote evaluation on %s through %s\n' "${target}" "${gateway}"
  run_on_target "${eval_launch_entry}" "${remote_env[@]}"
  exit 0
fi

if [[ "${COMMAND}" == "branches" ]]; then
  build_training_env
  list_branches='
set -euo pipefail
git ls-remote --heads "${REPO_URL}" | awk "{print \$2}" | sed "s#refs/heads/##" | sort
'
  run_on_target "${list_branches}" "${remote_env[@]}"
  exit 0
fi

build_control_env
control_entry='
set -euo pipefail
cd "${REMOTE_PROJECT_DIR}" 2>/dev/null || {
  echo "Remote repository is not present at ${REMOTE_PROJECT_DIR}."
  echo "Run setup first to clone/setup the project on the node."
  exit 1
}
exec bash ops/remote_control.sh
'
run_on_target "${control_entry}" "${remote_env[@]}"
