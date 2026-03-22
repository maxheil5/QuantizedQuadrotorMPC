#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"

python3 -m quantized_quadrotor_sitl.experiments.offline_sitl_retrain \
  --runs-root "${ROOT_DIR}/results/sitl" \
  --results-root "${ROOT_DIR}/results/offline" \
  "$@"
