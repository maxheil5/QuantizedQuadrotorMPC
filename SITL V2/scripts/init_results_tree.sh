#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${V2_ROOT}/results"

mkdir -p \
  "${RESULTS_ROOT}/baselines/linear_mpc" \
  "${RESULTS_ROOT}/baselines/nonlinear_mpc" \
  "${RESULTS_ROOT}/learned/unquantized" \
  "${RESULTS_ROOT}/learned/quantized" \
  "${RESULTS_ROOT}/summary"

touch "${RESULTS_ROOT}/baselines/linear_mpc/.gitkeep"
touch "${RESULTS_ROOT}/baselines/nonlinear_mpc/.gitkeep"
touch "${RESULTS_ROOT}/learned/unquantized/.gitkeep"
touch "${RESULTS_ROOT}/learned/quantized/.gitkeep"

if [[ ! -f "${RESULTS_ROOT}/summary/run_index.csv" ]]; then
  printf '%s\n' \
    'run_id,run_family,controller_variant,scenario,word_length,realizations,config_path,metrics_path,trajectory_path,control_path,timing_path,environment_path,figure_dir' \
    > "${RESULTS_ROOT}/summary/run_index.csv"
fi

if [[ ! -f "${RESULTS_ROOT}/summary/metrics_long.csv" ]]; then
  printf '%s\n' \
    'run_id,metric_name,metric_value,units' \
    > "${RESULTS_ROOT}/summary/metrics_long.csv"
fi

if [[ ! -f "${RESULTS_ROOT}/summary/metrics_wide.csv" ]]; then
  printf '%s\n' \
    'run_id,run_family,controller_variant,scenario,word_length,realizations,success,hover_rmse_m,tracking_rmse_m,max_error_m,solve_time_ms_mean' \
    > "${RESULTS_ROOT}/summary/metrics_wide.csv"
fi

echo "Results tree ready at ${RESULTS_ROOT}"

