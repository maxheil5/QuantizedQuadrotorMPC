#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/sitl_runtime_sitl_retrain_edmd_light_anchor.yaml}"
RUN_COUNT="${SITL_REPEAT_RUN_COUNT:-2}"
RUN_TIMEOUT_SECONDS="${SITL_RUN_TIMEOUT_SECONDS:-120}"
KILL_AFTER_SECONDS="${SITL_KILL_AFTER_SECONDS:-10}"
CLEANUP_WAIT_SECONDS="${SITL_CLEANUP_WAIT_SECONDS:-2}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

cleanup_sitl_processes() {
  bash "${ROOT_DIR}/scripts/cleanup_sitl_processes.sh"
}

for run_index in $(seq 1 "${RUN_COUNT}"); do
  echo "=== Running repeatability trial ${run_index}/${RUN_COUNT} with ${CONFIG_PATH} ==="
  cleanup_sitl_processes
  sleep "${CLEANUP_WAIT_SECONDS}"

  if timeout --signal=INT --kill-after="${KILL_AFTER_SECONDS}" "${RUN_TIMEOUT_SECONDS}" bash "${ROOT_DIR}/scripts/run_sitl_experiment.sh" "${CONFIG_PATH}"; then
    echo "=== Trial ${run_index} completed ==="
  else
    status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
      echo "=== Trial ${run_index} hit timeout/forced cleanup after ${RUN_TIMEOUT_SECONDS}s; continuing after cleanup ===" >&2
    else
      echo "=== Trial ${run_index} failed with status ${status}; stopping batch ===" >&2
      cleanup_sitl_processes
      exit "${status}"
    fi
  fi

  cleanup_sitl_processes
  sleep "${CLEANUP_WAIT_SECONDS}"
done
