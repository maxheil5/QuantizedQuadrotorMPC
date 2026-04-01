#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/sitl_runtime_identification.yaml"
PACKAGE_CONFIG_PATH="${ROOT_DIR}/src/quantized_quadrotor_sitl/config/runtime_identification.yaml"

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

for seed in "${SEEDS[@]}"; do
  echo "=== Running SITL identification seed ${seed} ==="
  sed -i "s/^reference_seed: .*/reference_seed: ${seed}/" "${CONFIG_PATH}" "${PACKAGE_CONFIG_PATH}"
  pkill -f MicroXRCEAgent >/dev/null 2>&1 || true
  pkill -f px4 >/dev/null 2>&1 || true
  pkill -f "gz sim" >/dev/null 2>&1 || true
  bash "${ROOT_DIR}/scripts/run_sitl_experiment.sh" "${CONFIG_PATH}"
done
