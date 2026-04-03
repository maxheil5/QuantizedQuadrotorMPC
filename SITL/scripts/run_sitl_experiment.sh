#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/sitl_runtime.yaml}"
CALLER_WORKDIR="$(pwd)"
PX4_DIR="${PX4_DIR:-${ROOT_DIR}/artifacts/external/PX4-Autopilot}"
AGENT_DIR="${AGENT_DIR:-${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-${ROOT_DIR}/configs/gazebo/quantized_koopman_quad.yaml}"
WORLD_FILE="${PX4_GZ_WORLD_FILE:-}"
WORLD_NAME="${PX4_GZ_WORLD:-default}"
SIM_MODEL_NAME="${PX4_SIM_MODEL:-gz_quantized_koopman_quad}"
HEADLESS="${HEADLESS:-0}"
GCS_HEARTBEAT_HOST="${GCS_HEARTBEAT_HOST:-127.0.0.1}"
GCS_HEARTBEAT_PORT="${GCS_HEARTBEAT_PORT:-18570}"
GCS_HEARTBEAT_RATE_HZ="${GCS_HEARTBEAT_RATE_HZ:-1.0}"
AUTO_DRIFT_ANALYSIS="${AUTO_DRIFT_ANALYSIS:-1}"
SITL_PRINT_RUN_OUTPUTS="${SITL_PRINT_RUN_OUTPUTS:-1}"
SIMULATION_STACK_STOPPED=0
PX4_PGID=""
INTERRUPT_REQUESTED=0
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"
GZ_MODELS_DIR="${ROOT_DIR}/artifacts/generated/gazebo_models"
GZ_WORLDS_DIR="${ROOT_DIR}/configs/gazebo/worlds"
PX4_BUNDLED_MODELS_DIR="${PX4_DIR}/Tools/simulation/gz/models"
PX4_BUNDLED_WORLDS_DIR="${PX4_DIR}/Tools/simulation/gz/worlds"
PX4_EXTERNAL_MODELS_DIR="${ROOT_DIR}/artifacts/external/PX4-gazebo-models/models"
LOCAL_CACHE_MODELS_DIR="${HOME}/.simulation-gazebo/models"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"
export GZ_TRANSPORT_LOCALHOST_ONLY="${GZ_TRANSPORT_LOCALHOST_ONLY:-1}"
export IGN_TRANSPORT_DISABLE_MULTICAST="${IGN_TRANSPORT_DISABLE_MULTICAST:-1}"
export IGN_IP="${IGN_IP:-127.0.0.1}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
source /opt/ros/humble/setup.bash
if [[ -f "${ROOT_DIR}/install/setup.bash" ]]; then
  source "${ROOT_DIR}/install/setup.bash"
fi

if [[ ! -x "${AGENT_DIR}/build/MicroXRCEAgent" ]]; then
  echo "Micro XRCE-DDS Agent is missing. Run ./scripts/setup_linux.sh first." >&2
  exit 1
fi

if [[ ! -d "${PX4_DIR}" ]]; then
  echo "PX4 checkout is missing. Run ./scripts/setup_linux.sh first." >&2
  exit 1
fi

if ! python -c "import pymavlink" >/dev/null 2>&1; then
  echo "pymavlink is missing in the active Python environment." >&2
  echo "Install it with: python -m pip install pymavlink" >&2
  exit 1
fi

if [[ -n "${WORLD_FILE}" && ! -f "${WORLD_FILE}" ]]; then
  echo "Gazebo world file is missing: ${WORLD_FILE}" >&2
  exit 1
fi

bash "${ROOT_DIR}/scripts/install_px4_gazebo_overlay.sh" "${OVERLAY_CONFIG}" >/dev/null
if [[ -n "${WORLD_FILE}" ]]; then
  mkdir -p "${PX4_BUNDLED_WORLDS_DIR}"
  cp "${WORLD_FILE}" "${PX4_BUNDLED_WORLDS_DIR}/${WORLD_NAME}.sdf"
fi
export GZ_SIM_RESOURCE_PATH="${LOCAL_CACHE_MODELS_DIR}:${GZ_MODELS_DIR}:${GZ_WORLDS_DIR}:${PX4_BUNDLED_MODELS_DIR}:${PX4_EXTERNAL_MODELS_DIR}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

"${AGENT_DIR}/build/MicroXRCEAgent" udp4 -p 8888 &
AGENT_PID=$!

resolve_process_group_id() {
  local pid="$1"
  ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true
}

resolve_results_dir_from_config() {
  python - "${CONFIG_PATH}" "${CALLER_WORKDIR}" <<'PY'
from pathlib import Path
import sys
from quantized_quadrotor_sitl.core.config import load_runtime_config

config_path = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
resolved_config = config_path if config_path.is_absolute() else (base_dir / config_path)
config = load_runtime_config(resolved_config)
results_dir = Path(config.results_dir)
if not results_dir.is_absolute():
    results_dir = base_dir / results_dir
print(results_dir)
PY
}

resolve_active_run_dir() {
  local resolved_results_dir="$1"
  python - "${resolved_results_dir}" <<'PY'
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
run_dir = results_dir.resolve(strict=False) if results_dir.name == "latest" else results_dir.resolve(strict=False)
print(run_dir)
PY
}

resolve_run_dir_from_metadata() {
  local metadata_path="$1"
  python - "${metadata_path}" <<'PY'
from pathlib import Path
import json
import sys

metadata_path = Path(sys.argv[1])
if not metadata_path.exists():
    raise SystemExit(1)
payload = json.loads(metadata_path.read_text(encoding="utf-8"))
run_dir = payload.get("run_dir")
if run_dir is None:
    raise SystemExit(1)
print(Path(run_dir).resolve(strict=False))
PY
}

resolve_canonical_run_dir() {
  local resolved_results_dir guessed_run_dir metadata_path canonical_run_dir
  if ! resolved_results_dir="$(resolve_results_dir_from_config)"; then
    return 1
  fi
  if ! guessed_run_dir="$(resolve_active_run_dir "${resolved_results_dir}")"; then
    return 1
  fi
  metadata_path="${guessed_run_dir}/run_metadata.json"
  if canonical_run_dir="$(resolve_run_dir_from_metadata "${metadata_path}" 2>/dev/null)"; then
    echo "${canonical_run_dir}"
    return 0
  fi
  echo "${guessed_run_dir}"
}

resolve_artifact_path_from_metadata() {
  local metadata_path="$1"
  python - "${metadata_path}" "${CALLER_WORKDIR}" <<'PY'
from pathlib import Path
import json
import sys

metadata_path = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
raw_path = payload.get("model_artifact")
if raw_path is None:
    raise SystemExit("run metadata does not include model_artifact")
artifact_path = Path(raw_path)
if not artifact_path.is_absolute():
    artifact_path = base_dir / artifact_path
print(artifact_path)
PY
}

require_running() {
  local pid="$1"
  local label="$2"
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "${label} failed to start or exited immediately." >&2
    exit 1
  fi
}

stop_px4_process_group() {
  local pgid="$1"
  local pid="$2"
  local wait_count=0
  if [[ -z "${pgid}" ]]; then
    return 0
  fi
  kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
  while [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" >/dev/null 2>&1 && (( wait_count < 20 )); do
    sleep 0.25
    wait_count=$((wait_count + 1))
  done
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
  fi
}

stop_simulation_stack() {
  if [[ "${SIMULATION_STACK_STOPPED:-0}" == "1" ]]; then
    return 0
  fi
  SIMULATION_STACK_STOPPED=1

  echo "Stopping PX4/Gazebo simulation stack..." >&2
  kill "${TELEMETRY_PID:-0}" >/dev/null 2>&1 || true
  stop_px4_process_group "${PX4_PGID:-}" "${PX4_PID:-0}"
  kill "${PX4_PID:-0}" >/dev/null 2>&1 || true
  kill "${GCS_HEARTBEAT_PID:-0}" >/dev/null 2>&1 || true
  kill "${AGENT_PID:-0}" >/dev/null 2>&1 || true
  wait "${TELEMETRY_PID:-0}" >/dev/null 2>&1 || true
  wait "${PX4_PID:-0}" >/dev/null 2>&1 || true
  wait "${GCS_HEARTBEAT_PID:-0}" >/dev/null 2>&1 || true
  wait "${AGENT_PID:-0}" >/dev/null 2>&1 || true
}

cleanup() {
  trap - EXIT INT TERM
  kill "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
  stop_simulation_stack
  wait "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

request_graceful_shutdown() {
  local signal="$1"
  if [[ "${INTERRUPT_REQUESTED}" == "1" ]]; then
    echo "Second ${signal} received; forcing immediate shutdown." >&2
    trap - EXIT INT TERM
    kill "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
    stop_simulation_stack
    exit 130
  fi

  INTERRUPT_REQUESTED=1
  echo "Received ${signal}; stopping the simulation and finalizing run artifacts. Do not press Ctrl-C again unless you want to abort cleanup." >&2
  kill "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
  stop_simulation_stack
}

trap 'request_graceful_shutdown INT' INT
trap 'request_graceful_shutdown TERM' TERM

maybe_generate_postrun_analyses() {
  if [[ "${AUTO_DRIFT_ANALYSIS}" != "1" ]]; then
    return 0
  fi

  local run_dir metadata_path artifact_path
  local postrun_cmd=()

  if ! run_dir="$(resolve_canonical_run_dir)"; then
    echo "WARNING: automatic post-run analysis failed while resolving run_dir." >&2
    echo "Config path: ${CONFIG_PATH}" >&2
    return 1
  fi
  metadata_path="${run_dir}/run_metadata.json"
  if ! artifact_path="$(resolve_artifact_path_from_metadata "${metadata_path}")"; then
    echo "WARNING: automatic post-run analysis failed while resolving artifact_path." >&2
    echo "Resolved run dir: ${run_dir}" >&2
    echo "Metadata path: ${metadata_path}" >&2
    return 1
  fi

  postrun_cmd=(
    python -m quantized_quadrotor_sitl.experiments.sitl_postrun_analysis
    --run-dir "${run_dir}"
    --artifact-path "${artifact_path}"
    --metadata-path "${metadata_path}"
  )

  if ! "${postrun_cmd[@]}"; then
    echo "WARNING: automatic post-run analysis failed; runtime_log.csv is still available." >&2
    echo "Resolved run dir: ${run_dir}" >&2
    echo "Resolved artifact path: ${artifact_path}" >&2
    echo "Failed command: ${postrun_cmd[*]}" >&2
    return 1
  fi
}

print_run_output_summary() {
  if [[ "${SITL_PRINT_RUN_OUTPUTS}" != "1" ]]; then
    return 0
  fi

  local run_dir
  if ! run_dir="$(resolve_canonical_run_dir)"; then
    echo "SITL cleanup finished, but the final run directory could not be resolved." >&2
    return 1
  fi
  if [[ ! -d "${run_dir}" ]]; then
    echo "SITL cleanup finished, but the run directory does not exist yet: ${run_dir}" >&2
    return 1
  fi

  echo "" >&2
  echo "SITL run fully complete." >&2
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
  echo "Host cleanup before rerun after invalid-runtime results:" >&2
  echo "  1. Run one SITL job at a time." >&2
  echo "  2. Avoid parallel heavy jobs on the Ubuntu host." >&2
  echo "  3. Use a fresh shell with ROS and the venv sourced once." >&2
  echo "  4. Restart PX4/Gazebo cleanly with: bash ./scripts/cleanup_sitl_processes.sh" >&2
  echo "It is safe to stop here once you see this summary." >&2
}

if [[ ! -x "${PX4_DIR}/build/px4_sitl_default/bin/px4" ]]; then
  pushd "${PX4_DIR}" >/dev/null
  make px4_sitl
  popd >/dev/null
fi

if [[ "${HEADLESS}" == "1" ]]; then
  echo "HEADLESS=1 is not wired for the default PX4 launcher path." >&2
fi

python -u -m quantized_quadrotor_sitl.tools.gcs_heartbeat \
  --host "${GCS_HEARTBEAT_HOST}" \
  --port "${GCS_HEARTBEAT_PORT}" \
  --rate-hz "${GCS_HEARTBEAT_RATE_HZ}" &
GCS_HEARTBEAT_PID=$!
sleep 1
require_running "${GCS_HEARTBEAT_PID}" "GCS heartbeat helper"

pushd "${PX4_DIR}" >/dev/null
if command -v setsid >/dev/null 2>&1; then
  setsid env \
    PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4001}" \
    PX4_GZ_WORLD="${WORLD_NAME}" \
    PX4_SIM_MODEL="${SIM_MODEL_NAME}" \
    ./build/px4_sitl_default/bin/px4 &
else
  env \
    PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4001}" \
    PX4_GZ_WORLD="${WORLD_NAME}" \
    PX4_SIM_MODEL="${SIM_MODEL_NAME}" \
    ./build/px4_sitl_default/bin/px4 &
fi
PX4_PID=$!
popd >/dev/null
sleep 1
PX4_PGID="$(resolve_process_group_id "${PX4_PID}")"

python -m quantized_quadrotor_sitl.ros.telemetry_adapter_node --ros-args -p config_path:="${CONFIG_PATH}" &
TELEMETRY_PID=$!

python -m quantized_quadrotor_sitl.ros.controller_node --ros-args -p config_path:="${CONFIG_PATH}" &
CONTROLLER_PID=$!

CONTROLLER_STATUS=0
if ! wait "${CONTROLLER_PID}"; then
  CONTROLLER_STATUS=$?
fi

stop_simulation_stack
maybe_generate_postrun_analyses || true
print_run_output_summary || true
exit "${CONTROLLER_STATUS}"
