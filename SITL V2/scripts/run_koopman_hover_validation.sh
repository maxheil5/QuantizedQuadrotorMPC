#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_PATH="${2:-}"
DURATION_S="${3:-10}"
TARGET_Z_M="${4:-1.0}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

if [[ -n "${MODEL_PATH}" ]]; then
  bash "${SCRIPT_DIR}/start_koopman_hover_stack.sh" "${WORKSPACE_ROOT}" "${MODEL_PATH}"
else
  bash "${SCRIPT_DIR}/start_koopman_hover_stack.sh" "${WORKSPACE_ROOT}"
fi

sleep 2

python3 "${SCRIPT_DIR}/validate_hover_response.py" --duration "${DURATION_S}" --target-z "${TARGET_Z_M}"

cat <<EOF

Stack is still running.
Stop it with:
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh"
EOF
