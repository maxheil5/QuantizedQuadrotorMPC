#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
HOVER_VALIDATION_RUN_DIR="${2:-${KOOPMAN_HOVER_VALIDATION_RUN_DIR:-}}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

bash "${SCRIPT_DIR}/init_results_tree.sh"

echo "Running offline unquantized hover_5s."
python3 -m koopman_python.experiments.offline_learned_mpc --scenario-name hover_5s

echo "Running offline unquantized line_tracking."
python3 -m koopman_python.experiments.offline_learned_mpc --scenario-name line_tracking

echo "Running offline quantized wl=12 N=5 hover_5s."
python3 -m koopman_python.experiments.offline_quantized --scenario-name hover_5s --word-length 12 --realizations 5

echo "Running offline quantized wl=12 N=5 line_tracking."
python3 -m koopman_python.experiments.offline_quantized --scenario-name line_tracking --word-length 12 --realizations 5

echo "Packaging comparison table."
python3 "${SCRIPT_DIR}/generate_showable_results_summary.py"

if [[ -n "${HOVER_VALIDATION_RUN_DIR}" ]]; then
  echo "Generating hover diagnostic figures from ${HOVER_VALIDATION_RUN_DIR}."
  python3 "${SCRIPT_DIR}/plot_hover_diagnostic.py" --run-dir "${HOVER_VALIDATION_RUN_DIR}"
fi
