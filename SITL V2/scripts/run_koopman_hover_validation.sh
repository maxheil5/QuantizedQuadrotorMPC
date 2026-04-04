#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_PATH="${2:-}"
DURATION_S="${3:-10}"
TARGET_MODE="${KOOPMAN_TARGET_MODE:-relative}"
TARGET_Z_M="${4:-${KOOPMAN_TARGET_Z:-1.0}}"
TARGET_RELATIVE_Z_M="${4:-${KOOPMAN_TARGET_RELATIVE_Z:-1.0}}"
TARGET_YAW_RAD="${KOOPMAN_TARGET_YAW:-0.0}"
VALIDATION_OUTPUT_ROOT="${KOOPMAN_VALIDATION_OUTPUT_ROOT:-}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

if [[ -n "${MODEL_PATH}" ]]; then
  bash "${SCRIPT_DIR}/start_koopman_hover_stack.sh" "${WORKSPACE_ROOT}" "${MODEL_PATH}"
else
  bash "${SCRIPT_DIR}/start_koopman_hover_stack.sh" "${WORKSPACE_ROOT}"
fi

sleep 2

VALIDATOR_ARGS=(
  --duration "${DURATION_S}"
  --target-mode "${TARGET_MODE}"
  --target-yaw "${TARGET_YAW_RAD}"
)

if [[ -n "${VALIDATION_OUTPUT_ROOT}" ]]; then
  VALIDATOR_ARGS+=(--output-root "${VALIDATION_OUTPUT_ROOT}")
fi

if [[ "${TARGET_MODE}" == "relative" ]]; then
  echo "Validation target mode: relative (${TARGET_RELATIVE_Z_M} m above takeoff)"
  VALIDATOR_ARGS+=(--target-relative-z "${TARGET_RELATIVE_Z_M}")
else
  echo "Validation target mode: absolute (z=${TARGET_Z_M} m world)"
  VALIDATOR_ARGS+=(--target-z "${TARGET_Z_M}")
fi

python3 "${SCRIPT_DIR}/validate_hover_response.py" "${VALIDATOR_ARGS[@]}"

cat <<EOF

Stack is still running.
Stop it with:
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh"
EOF
