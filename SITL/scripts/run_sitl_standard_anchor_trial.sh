#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/sitl_runtime_sitl_retrain_edmd_anchor_h8.yaml}"
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"
INTERRUPT_REQUESTED=0

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

handle_interrupt() {
  local signal="$1"
  if [[ "${INTERRUPT_REQUESTED}" == "1" ]]; then
    echo "Second ${signal} received; exiting standard anchor wrapper immediately." >&2
    trap - INT TERM
    exit 130
  fi
  INTERRUPT_REQUESTED=1
  echo "Received ${signal}; waiting for the SITL runner to finalize artifacts..." >&2
}

trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM

print_run_output_summary() {
  local run_dir="$1"
  local root_milestone_summary_path="${2:-}"
  local root_milestone_summary_updated="${3:-0}"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Standard anchor trial finished, but the run directory could not be summarized." >&2
    return 1
  fi

  echo "" >&2
  echo "Standard anchor trial fully complete." >&2
  echo "Run directory: ${run_dir}" >&2
  echo "Canonical run folder name: $(basename "${run_dir}")" >&2
  echo "Use this exact folder name for upload and analysis. Do not rename it." >&2
  echo "Stored files:" >&2
  python - "${run_dir}" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
    if path.is_file():
        print(f"  - {path.name}")
PY
  local required_files=(
    "runtime_log.csv"
    "run_metadata.json"
    "runtime_health_summary.json"
    "drift_summary.json"
    "drift_trace.csv"
    "control_audit_summary.json"
    "control_audit_trace.csv"
    "u2_root_cause_summary.json"
    "u2_root_cause_trace.csv"
  )
  local missing_required=()
  local required_file
  for required_file in "${required_files[@]}"; do
    if [[ ! -f "${run_dir}/${required_file}" ]]; then
      missing_required+=("${required_file}")
    fi
  done
  if (( ${#missing_required[@]} == 0 )); then
    echo "Required files: complete" >&2
  else
    echo "Missing required files: ${missing_required[*]}" >&2
  fi
  if [[ -n "${root_milestone_summary_path}" ]]; then
    echo "Root milestone summary: ${root_milestone_summary_path}" >&2
    if [[ "${root_milestone_summary_updated}" == "1" ]]; then
      echo "Root milestone summary row for this run: present" >&2
    else
      echo "WARNING: root milestone_summary.csv was not updated for this run." >&2
    fi
  else
    echo "WARNING: root milestone_summary.csv path could not be resolved for this run." >&2
  fi
  echo "Host cleanup before rerun after invalid-runtime results:" >&2
  echo "  1. Run one SITL job at a time." >&2
  echo "  2. Avoid parallel heavy jobs on the Ubuntu host." >&2
  echo "  3. Use a fresh shell with ROS and the venv sourced once." >&2
  echo "  4. Restart PX4/Gazebo cleanly with: bash ./scripts/cleanup_sitl_processes.sh" >&2
  echo "It is safe to stop here once you see this summary." >&2
}

resolve_run_dir() {
  python - "${CONFIG_PATH}" "${ROOT_DIR}" <<'PY'
from pathlib import Path
import json
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
metadata_path = run_dir / "run_metadata.json"
if metadata_path.exists():
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_run_dir = payload.get("run_dir")
    if metadata_run_dir:
        run_dir = Path(metadata_run_dir).resolve(strict=False)
print(run_dir)
PY
}

RUN_STATUS=0
POSTRUN_SUMMARY_JSON=""
ROOT_MILESTONE_SUMMARY_PATH=""
ROOT_MILESTONE_SUMMARY_UPDATED="0"
if ! SITL_PRINT_RUN_OUTPUTS=0 bash "${ROOT_DIR}/scripts/run_sitl_experiment.sh" "${CONFIG_PATH}"; then
  RUN_STATUS=$?
fi

if run_dir="$(resolve_run_dir)"; then
  if [[ -f "${run_dir}/runtime_log.csv" ]]; then
    POSTRUN_SUMMARY_JSON="$(mktemp)"
    if ! python -m quantized_quadrotor_sitl.experiments.sitl_postrun_analysis \
      --run-dir "${run_dir}" \
      --base-dir "${ROOT_DIR}" >"${POSTRUN_SUMMARY_JSON}"; then
      echo "WARNING: standard anchor post-run backfill failed for ${run_dir}" >&2
    elif mapfile -t postrun_summary_fields < <(
      python - "${POSTRUN_SUMMARY_JSON}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as stream:
    payload = json.load(stream)
print(payload.get("milestone_summary_path", ""))
print("1" if payload.get("milestone_summary_contains_run") else "0")
PY
    ); then
      ROOT_MILESTONE_SUMMARY_PATH="${postrun_summary_fields[0]:-}"
      ROOT_MILESTONE_SUMMARY_UPDATED="${postrun_summary_fields[1]:-0}"
    fi
    rm -f "${POSTRUN_SUMMARY_JSON}"
  else
    echo "WARNING: runtime_log.csv not found in ${run_dir}; skipping explicit post-run backfill." >&2
  fi
else
  echo "WARNING: failed to resolve run directory for ${CONFIG_PATH}; skipping explicit post-run backfill." >&2
fi

print_run_output_summary "${run_dir:-}" "${ROOT_MILESTONE_SUMMARY_PATH:-}" "${ROOT_MILESTONE_SUMMARY_UPDATED:-0}" || true

exit "${RUN_STATUS}"
