#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/sitl_runtime.yaml}"
PX4_DIR="${PX4_DIR:-${ROOT_DIR}/artifacts/external/PX4-Autopilot}"
AGENT_DIR="${AGENT_DIR:-${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent}"
OVERLAY_CONFIG="${OVERLAY_CONFIG:-${ROOT_DIR}/configs/gazebo/quantized_koopman_quad.yaml}"
WORLD_FILE="${PX4_GZ_WORLD_FILE:-}"
WORLD_NAME="${PX4_GZ_WORLD:-default}"
SIM_MODEL_NAME="${PX4_SIM_MODEL:-gz_quantized_koopman_quad}"
HEADLESS="${HEADLESS:-0}"
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
  kill "${AGENT_PID}" >/dev/null 2>&1 || true
  kill "${PX4_PID:-0}" >/dev/null 2>&1 || true
  kill "${TELEMETRY_PID:-0}" >/dev/null 2>&1 || true
  kill "${CONTROLLER_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -x "${PX4_DIR}/build/px4_sitl_default/bin/px4" ]]; then
  pushd "${PX4_DIR}" >/dev/null
  make px4_sitl
  popd >/dev/null
fi

if [[ "${HEADLESS}" == "1" ]]; then
  echo "HEADLESS=1 is not wired for the default PX4 launcher path." >&2
fi

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

wait "${CONTROLLER_PID}"
