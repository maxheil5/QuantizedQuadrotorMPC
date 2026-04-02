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
GZ_MODELS_DIR="${ROOT_DIR}/artifacts/generated/gazebo_models"
GZ_WORLDS_DIR="${ROOT_DIR}/configs/gazebo/worlds"
PX4_BUNDLED_MODELS_DIR="${PX4_DIR}/Tools/simulation/gz/models"
PX4_BUNDLED_WORLDS_DIR="${PX4_DIR}/Tools/simulation/gz/worlds"
PX4_EXTERNAL_MODELS_DIR="${ROOT_DIR}/artifacts/external/PX4-gazebo-models/models"
LOCAL_CACHE_MODELS_DIR="${HOME}/.simulation-gazebo/models"
PACKAGE_ROOT="${ROOT_DIR}/src/quantized_quadrotor_sitl"

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

cleanup() {
  trap - EXIT INT TERM
  kill "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
  kill "${TELEMETRY_PID:-0}" >/dev/null 2>&1 || true
  kill "${PX4_PID:-0}" >/dev/null 2>&1 || true
  kill "${GCS_HEARTBEAT_PID:-0}" >/dev/null 2>&1 || true
  kill "${AGENT_PID:-0}" >/dev/null 2>&1 || true
  wait "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
  wait "${TELEMETRY_PID:-0}" >/dev/null 2>&1 || true
  wait "${PX4_PID:-0}" >/dev/null 2>&1 || true
  wait "${GCS_HEARTBEAT_PID:-0}" >/dev/null 2>&1 || true
  wait "${AGENT_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

require_running() {
  local pid="$1"
  local label="$2"
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "${label} failed to start or exited immediately." >&2
    exit 1
  fi
}

maybe_generate_drift_analysis() {
  if [[ "${AUTO_DRIFT_ANALYSIS}" != "1" ]]; then
    return 0
  fi

  python - "${CONFIG_PATH}" "${CALLER_WORKDIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from quantized_quadrotor_sitl.core.config import load_runtime_config
from quantized_quadrotor_sitl.experiments.sitl_drift_analysis import analyze_runtime_drift


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


config_path = resolve_path(sys.argv[1], Path(sys.argv[2]))
config = load_runtime_config(config_path)

if config.controller_mode != "edmd_mpc":
    print("Skipping drift analysis for non-EDMD controller mode.")
    raise SystemExit(0)

artifact_path = resolve_path(config.model_artifact, Path(sys.argv[2]))
results_dir = resolve_path(config.results_dir, Path(sys.argv[2]))
run_dir = results_dir.resolve(strict=False) if results_dir.name == "latest" else results_dir
log_path = run_dir / "runtime_log.csv"

if not artifact_path.exists():
    print(f"Skipping drift analysis because artifact is missing: {artifact_path}")
    raise SystemExit(0)

if not log_path.exists():
    print(f"Skipping drift analysis because runtime log is missing: {log_path}")
    raise SystemExit(0)

summary = analyze_runtime_drift(log_path=log_path, artifact_path=artifact_path)
print(
    json.dumps(
        {
            "drift_summary_path": str(run_dir / "drift_summary.json"),
            "drift_trace_path": str(run_dir / "drift_trace.csv"),
            "selected_branch": summary.get("selected_branch"),
            "dominant_error_group": summary.get("dominant_error_group"),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
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
env \
  PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4001}" \
  PX4_GZ_WORLD="${WORLD_NAME}" \
  PX4_SIM_MODEL="${SIM_MODEL_NAME}" \
  ./build/px4_sitl_default/bin/px4 &
PX4_PID=$!
popd >/dev/null

python -m quantized_quadrotor_sitl.ros.telemetry_adapter_node --ros-args -p config_path:="${CONFIG_PATH}" &
TELEMETRY_PID=$!

python -m quantized_quadrotor_sitl.ros.controller_node --ros-args -p config_path:="${CONFIG_PATH}" &
CONTROLLER_PID=$!

CONTROLLER_STATUS=0
if ! wait "${CONTROLLER_PID}"; then
  CONTROLLER_STATUS=$?
fi

if ! maybe_generate_drift_analysis; then
  echo "WARNING: automatic drift analysis failed; runtime_log.csv is still available." >&2
fi
exit "${CONTROLLER_STATUS}"
