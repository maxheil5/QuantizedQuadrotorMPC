#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
SCENARIO="${KOOPMAN_SCENARIO:-hover_5s}"
REALIZATIONS="${KOOPMAN_REALIZATIONS:-5}"
WORD_LENGTHS=(${KOOPMAN_WORD_LENGTHS:-8 12 14 16})
OUTPUT_ROOT="${KOOPMAN_OUTPUT_ROOT:-}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

for word_length in "${WORD_LENGTHS[@]}"; do
  echo "Building quantized hover model: scenario=${SCENARIO} wl=${word_length} N=${REALIZATIONS}"
  cmd=(
    python3 -m koopman_python.experiments.offline_quantized
    --scenario-name "${SCENARIO}"
    --word-length "${word_length}"
    --realizations "${REALIZATIONS}"
  )
  if [[ -n "${OUTPUT_ROOT}" ]]; then
    cmd+=(--output-root "${OUTPUT_ROOT}")
  fi
  "${cmd[@]}"
done
