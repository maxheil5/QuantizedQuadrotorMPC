#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
DURATION_S="${2:-10}"
TARGET_RELATIVE_Z_M="${3:-1.0}"
WORD_LENGTHS=(${KOOPMAN_WORD_LENGTHS:-8 12 14 16})
REALIZATIONS="${KOOPMAN_REALIZATIONS:-5}"
REALIZATION_INDEX="${KOOPMAN_REALIZATION_INDEX:-0}"
BUILD_MODELS="${KOOPMAN_BUILD_QUANTIZED_MODELS:-false}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SWEEP_DIR="${V2_ROOT}/results/summary/hover_quantized_sweep/${STAMP}"
MANIFEST_PATH="${SWEEP_DIR}/manifest.csv"

mkdir -p "${SWEEP_DIR}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

printf '%s\n' 'word_length,realizations,realization_index,validation_run_dir,figure_dir,log_path' > "${MANIFEST_PATH}"

if [[ "${BUILD_MODELS}" == "true" ]]; then
  KOOPMAN_WORD_LENGTHS="${WORD_LENGTHS[*]}" \
  KOOPMAN_REALIZATIONS="${REALIZATIONS}" \
  KOOPMAN_SCENARIO="hover_5s" \
  bash "${SCRIPT_DIR}/build_quantized_hover_models.sh" "${WORKSPACE_ROOT}"
fi

for word_length in "${WORD_LENGTHS[@]}"; do
  echo "Running quantized hover validation: wl=${word_length} N=${REALIZATIONS} realization=${REALIZATION_INDEX}"
  log_path="${SWEEP_DIR}/wl_${word_length}.log"
  KOOPMAN_MODEL_VARIANT="quantized" \
  KOOPMAN_SCENARIO="hover_5s" \
  KOOPMAN_WORD_LENGTH="${word_length}" \
  KOOPMAN_REALIZATIONS="${REALIZATIONS}" \
  KOOPMAN_REALIZATION_INDEX="${REALIZATION_INDEX}" \
  KOOPMAN_TARGET_MODE="relative" \
  KOOPMAN_TARGET_RELATIVE_Z="${TARGET_RELATIVE_Z_M}" \
  bash "${SCRIPT_DIR}/run_koopman_hover_validation.sh" "${WORKSPACE_ROOT}" "" "${DURATION_S}" "${TARGET_RELATIVE_Z_M}" | tee "${log_path}"

  validation_run_dir="$(grep '^hover_validation_run_dir=' "${log_path}" | tail -n 1 | cut -d= -f2-)"
  if [[ -z "${validation_run_dir}" ]]; then
    echo "Could not determine hover_validation_run_dir for wl=${word_length}." >&2
    bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh" || true
    exit 1
  fi

  python3 "${SCRIPT_DIR}/plot_hover_diagnostic.py" --run-dir "${validation_run_dir}"
  printf '%s\n' "${word_length},${REALIZATIONS},${REALIZATION_INDEX},${validation_run_dir},${validation_run_dir}/figures,${log_path}" >> "${MANIFEST_PATH}"
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh"
done

python3 "${SCRIPT_DIR}/generate_quantized_hover_sweep_summary.py" --sweep-dir "${SWEEP_DIR}"
echo "quantized_hover_sweep_dir=${SWEEP_DIR}"
