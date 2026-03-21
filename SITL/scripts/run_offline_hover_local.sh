#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}/src/quantized_quadrotor_sitl:${PYTHONPATH:-}"
python3 -m quantized_quadrotor_sitl.experiments.offline_hover_local \
  --results-root "${ROOT_DIR}/results/offline"
