#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/sitl_runtime_sitl_retrain_edmd_anchor_h8.yaml}"
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"
source /opt/ros/humble/setup.bash
if [[ -f "${ROOT_DIR}/install/setup.bash" ]]; then
  source "${ROOT_DIR}/install/setup.bash"
fi

print_run_output_summary() {
  local run_dir="$1"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Standard anchor trial finished, but the run directory could not be summarized." >&2
    return 1
  fi

  echo "" >&2
  echo "Standard anchor trial fully complete." >&2
  echo "Run directory: ${run_dir}" >&2
  echo "Stored files:" >&2
  python - "${run_dir}" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
    if path.is_file():
        print(f"  - {path.name}")
PY
  echo "It is safe to stop here once you see this summary." >&2
}

resolve_run_dir() {
  python - "${CONFIG_PATH}" "${ROOT_DIR}" <<'PY'
from pathlib import Path
import sys
from quantized_quadrotor_sitl.core.config import load_runtime_config

config_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
resolved_config = config_path if config_path.is_absolute() else (root_dir / config_path)
config = load_runtime_config(resolved_config)
results_dir = Path(config.results_dir)
if not results_dir.is_absolute():
    results_dir = root_dir / results_dir
run_dir = results_dir.resolve(strict=False) if results_dir.name == "latest" else results_dir.resolve(strict=False)
print(run_dir)
PY
}

RUN_STATUS=0
if ! SITL_PRINT_RUN_OUTPUTS=0 bash "${ROOT_DIR}/scripts/run_sitl_experiment.sh" "${CONFIG_PATH}"; then
  RUN_STATUS=$?
fi

if run_dir="$(resolve_run_dir)"; then
  if [[ -f "${run_dir}/runtime_log.csv" ]]; then
    if ! python -m quantized_quadrotor_sitl.experiments.sitl_postrun_analysis \
      --run-dir "${run_dir}" \
      --base-dir "${ROOT_DIR}"; then
      echo "WARNING: standard anchor post-run backfill failed for ${run_dir}" >&2
    fi
  else
    echo "WARNING: runtime_log.csv not found in ${run_dir}; skipping explicit post-run backfill." >&2
  fi
else
  echo "WARNING: failed to resolve run directory for ${CONFIG_PATH}; skipping explicit post-run backfill." >&2
fi

print_run_output_summary "${run_dir:-}" || true

exit "${RUN_STATUS}"
