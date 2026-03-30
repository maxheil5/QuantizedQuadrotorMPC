#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}:${PACKAGE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "streamlit is missing in the active Python environment." >&2
  echo "Install it with: python -m pip install streamlit" >&2
  exit 1
fi

python -m streamlit run "${ROOT_DIR}/dashboard/app.py"
