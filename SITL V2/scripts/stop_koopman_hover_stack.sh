#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${1:-$(ls -td "${V2_ROOT}/results/runtime_logs/koopman_hover"/* 2>/dev/null | head -n 1 || true)}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "No hover run directory found." >&2
  exit 1
fi

PID_FILE="${RUN_DIR}/pids.env"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "PID file not found at ${PID_FILE}." >&2
  exit 1
fi

source "${PID_FILE}"

kill_if_started() {
  local started="$1"
  local pid="$2"
  local name="$3"

  if [[ "${started}" != "1" || -z "${pid}" ]]; then
    return 0
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Stopping ${name} (pid ${pid})."
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

kill_if_started "${koopman_started_by_script}" "${koopman_pid}" "koopman_mpc_node launch"
kill_if_started "${gazebo_started_by_script}" "${gazebo_pid}" "gazebo launch"
kill_if_started "${master_started_by_script}" "${roscore_pid}" "roscore"

echo "Stop request issued for ${RUN_DIR}"
