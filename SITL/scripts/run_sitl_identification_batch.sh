#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/sitl_runtime_identification.yaml"
PACKAGE_CONFIG_PATH="${ROOT_DIR}/src/quantized_quadrotor_sitl/config/runtime_identification.yaml"
RUN_TIMEOUT_SECONDS="${SITL_RUN_TIMEOUT_SECONDS:-180}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PACKAGE_CONFIG_PATH}" ]]; then
  echo "Missing package config: ${PACKAGE_CONFIG_PATH}" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(2141444 2141445 2141446 2141447)
fi

cleanup_sitl_processes() {
  pkill -f MicroXRCEAgent >/dev/null 2>&1 || true
  pkill -f px4 >/dev/null 2>&1 || true
  pkill -f "gz sim" >/dev/null 2>&1 || true
}

for seed in "${SEEDS[@]}"; do
  echo "=== Running SITL identification seed ${seed} ==="
  sed -i "s/^reference_seed: .*/reference_seed: ${seed}/" "${CONFIG_PATH}" "${PACKAGE_CONFIG_PATH}"
  cleanup_sitl_processes
  sleep 2
  if timeout --signal=INT --kill-after=15 "${RUN_TIMEOUT_SECONDS}" bash "${ROOT_DIR}/scripts/run_sitl_experiment.sh" "${CONFIG_PATH}"; then
    echo "=== Seed ${seed} completed ==="
  else
    status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
      echo "=== Seed ${seed} hit timeout/forced cleanup after ${RUN_TIMEOUT_SECONDS}s; continuing after cleanup ===" >&2
    else
      echo "=== Seed ${seed} failed with status ${status}; stopping batch ===" >&2
      cleanup_sitl_processes
      exit "${status}"
    fi
  fi
  cleanup_sitl_processes
  sleep 2
done
