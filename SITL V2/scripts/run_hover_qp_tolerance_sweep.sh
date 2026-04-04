#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_PATH="${2:-}"
DURATION_S="${3:-10}"
TARGET_RELATIVE_Z_M="${4:-1.0}"
QP_TOL_LIST=(${KOOPMAN_QP_TOL_LIST:-1e-8 1e-4})

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

for qp_tol in "${QP_TOL_LIST[@]}"; do
  echo "Running hover validation with KOOPMAN_QP_TOL=${qp_tol}"
  KOOPMAN_QP_TOL="${qp_tol}" \
  KOOPMAN_TARGET_MODE="relative" \
  KOOPMAN_TARGET_RELATIVE_Z="${TARGET_RELATIVE_Z_M}" \
  bash "${SCRIPT_DIR}/run_koopman_hover_validation.sh" "${WORKSPACE_ROOT}" "${MODEL_PATH}" "${DURATION_S}"
  python3 "${SCRIPT_DIR}/plot_hover_diagnostic.py"
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh"
done
