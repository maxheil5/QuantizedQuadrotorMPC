#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_PATH="${2:-}"
LOWLEVEL_PKG_ROOT="${WORKSPACE_ROOT}/src/mav_control_rw/mav_lowlevel_attitude_controller"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${V2_ROOT}/results/runtime_logs/koopman_hover/${STAMP}"
PID_FILE="${RUN_DIR}/pids.env"
LATEST_MANAGED_RUN="$(ls -td "${V2_ROOT}/results/runtime_logs/koopman_hover"/* 2>/dev/null | head -n 1 || true)"

mkdir -p "${RUN_DIR}"

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_ROOT}/devel/setup.bash"

if [[ -z "${MODEL_PATH}" ]]; then
  latest_run="$(ls -td "${V2_ROOT}/results/learned/unquantized/hover_5s"/* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_run}" ]]; then
    MODEL_PATH="${latest_run}/model.npz"
  fi
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Could not find a valid model.npz. Pass it as the second argument." >&2
  exit 1
fi

wait_for_success() {
  local command="$1"
  local description="$2"
  local attempts="${3:-40}"
  local delay_seconds="${4:-0.5}"
  local i

  for ((i = 0; i < attempts; ++i)); do
    if eval "${command}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay_seconds}"
  done

  echo "Timed out waiting for ${description}." >&2
  return 1
}

wait_for_absence() {
  local command="$1"
  local description="$2"
  local attempts="${3:-20}"
  local delay_seconds="${4:-0.5}"
  local i

  for ((i = 0; i < attempts; ++i)); do
    if ! eval "${command}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay_seconds}"
  done

  echo "Timed out waiting for ${description} to stop." >&2
  return 1
}

master_started_by_script=0
gazebo_started_by_script=0
koopman_started_by_script=0
lowlevel_started_by_script=0
roscore_pid=""
gazebo_pid=""
koopman_pid=""
lowlevel_pid=""

ensure_lowlevel_firefly_yaml_links() {
  local firefly_yaml="${LOWLEVEL_PKG_ROOT}/resources/PID_attitude_firefly.yaml"
  local resource_yaml="${LOWLEVEL_PKG_ROOT}/resources/PID_attitude.yaml"
  local cfg_yaml="${LOWLEVEL_PKG_ROOT}/cfg/PID_attitude.yaml"

  if [[ ! -f "${firefly_yaml}" ]]; then
    echo "Warning: ${firefly_yaml} not found. Skipping low-level YAML link setup." >&2
    return 0
  fi

  mkdir -p "${LOWLEVEL_PKG_ROOT}/cfg"
  ln -sf "${firefly_yaml}" "${resource_yaml}"
  ln -sf "${firefly_yaml}" "${cfg_yaml}"
}

stop_existing_node_if_running() {
  local node_name="$1"
  local process_pattern="${2:-}"

  if rosnode list 2>/dev/null | grep -q "^${node_name}$"; then
    echo "Stopping existing ${node_name}."
    rosnode kill "${node_name}" >/dev/null 2>&1 || true
    if ! wait_for_absence "rosnode list | grep -q '^${node_name}$'" "${node_name}"; then
      if [[ -n "${process_pattern}" ]]; then
        echo "Force-stopping process pattern ${process_pattern}."
        pkill -f "${process_pattern}" >/dev/null 2>&1 || true
        sleep 1
        wait_for_absence "rosnode list | grep -q '^${node_name}$'" "${node_name}" || true
      fi
    fi
  fi
}

show_log_tail_and_exit() {
  local log_path="$1"
  local description="$2"

  echo "${description} failed. Recent log output from ${log_path}:" >&2
  if [[ -f "${log_path}" ]]; then
    tail -n 80 "${log_path}" >&2 || true
  else
    echo "  Log file not found." >&2
  fi
  exit 1
}

if [[ -n "${LATEST_MANAGED_RUN}" && "${LATEST_MANAGED_RUN}" != "${RUN_DIR}" && -f "${LATEST_MANAGED_RUN}/pids.env" ]]; then
  echo "Stopping previous managed hover stack: ${LATEST_MANAGED_RUN}"
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh" "${LATEST_MANAGED_RUN}" >/dev/null 2>&1 || true
  sleep 2
fi

if rosnode list >/dev/null 2>&1; then
  echo "Using existing ROS master."
else
  echo "Starting roscore."
  nohup roscore > "${RUN_DIR}/roscore.log" 2>&1 &
  roscore_pid="$!"
  master_started_by_script=1
  wait_for_success "rosnode list" "ROS master"
fi

if rosnode list 2>/dev/null | grep -q '^/gazebo$'; then
  echo "Reloading existing Gazebo Firefly stack."
  stop_existing_node_if_running "/firefly/koopman_mpc_node" "koopman_mpc_node.py"
  stop_existing_node_if_running "/firefly/mav_lowlevel_attitude_controller" "mav_pid_attitude_controller_node"
  stop_existing_node_if_running "/firefly/PID_attitude_controller" "PID_attitude_controller_node"
  stop_existing_node_if_running "/gazebo_gui" "gzclient"
  stop_existing_node_if_running "/gazebo" "gzserver"
  pkill -f gzserver >/dev/null 2>&1 || true
  pkill -f gzclient >/dev/null 2>&1 || true
  sleep 2
fi

echo "Starting Gazebo Firefly stack."
nohup roslaunch rotors_gazebo mav.launch mav_name:=firefly > "${RUN_DIR}/gazebo.log" 2>&1 &
gazebo_pid="$!"
gazebo_started_by_script=1
wait_for_success "rosservice list | grep -q '^/gazebo/unpause_physics$'" "Gazebo services"

echo "Unpausing Gazebo physics."
rosservice call /gazebo/unpause_physics "{}" > "${RUN_DIR}/unpause.log" 2>&1 || true
wait_for_success "rostopic list | grep -q '^/clock$'" "clock topic"
wait_for_success "rostopic list | grep -q '^/firefly/ground_truth/odometry$'" "ground-truth odometry topic"

stop_existing_node_if_running "/firefly/koopman_mpc_node" "koopman_mpc_node.py"

echo "Starting learned hover node."
nohup env ROS_NAMESPACE=firefly rosrun koopman_mpc_ros koopman_mpc_node.py __name:=koopman_mpc_node "_model_path:=${MODEL_PATH}" _parameter_profile:=rotors_firefly_linear_mpc_runtime _pred_horizon:=10 _qp_max_iter:=100 odometry:=ground_truth/odometry > "${RUN_DIR}/koopman_mpc_node.log" 2>&1 &
koopman_pid="$!"
koopman_started_by_script=1
if ! wait_for_success "rosnode list | grep -q '^/firefly/koopman_mpc_node$'" "koopman_mpc_node"; then
  show_log_tail_and_exit "${RUN_DIR}/koopman_mpc_node.log" "Starting koopman_mpc_node"
fi

ensure_lowlevel_firefly_yaml_links

stop_existing_node_if_running "/firefly/mav_lowlevel_attitude_controller" "mav_pid_attitude_controller_node"
stop_existing_node_if_running "/firefly/PID_attitude_controller" "PID_attitude_controller_node"

echo "Starting low-level attitude controller."
nohup roslaunch mav_lowlevel_attitude_controller mav_lowlevel_controller.launch > "${RUN_DIR}/mav_lowlevel_attitude_controller.log" 2>&1 &
lowlevel_pid="$!"
lowlevel_started_by_script=1
if ! wait_for_success "rosnode list | grep -q '^/firefly/mav_lowlevel_attitude_controller$'" "mav_lowlevel_attitude_controller"; then
  show_log_tail_and_exit "${RUN_DIR}/mav_lowlevel_attitude_controller.log" "Starting mav_lowlevel_attitude_controller"
fi

{
  echo "run_dir='${RUN_DIR}'"
  echo "workspace_root='${WORKSPACE_ROOT}'"
  echo "model_path='${MODEL_PATH}'"
  echo "master_started_by_script='${master_started_by_script}'"
  echo "gazebo_started_by_script='${gazebo_started_by_script}'"
  echo "koopman_started_by_script='${koopman_started_by_script}'"
  echo "lowlevel_started_by_script='${lowlevel_started_by_script}'"
  echo "roscore_pid='${roscore_pid}'"
  echo "gazebo_pid='${gazebo_pid}'"
  echo "koopman_pid='${koopman_pid}'"
  echo "lowlevel_pid='${lowlevel_pid}'"
} > "${PID_FILE}"

cat <<EOF
koopman_hover_stack_run_dir=${RUN_DIR}
Logs:
  ${RUN_DIR}/roscore.log
  ${RUN_DIR}/gazebo.log
  ${RUN_DIR}/koopman_mpc_node.log
  ${RUN_DIR}/mav_lowlevel_attitude_controller.log

Hover command:
  rostopic pub -1 /firefly/command/pose geometry_msgs/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 0.0, y: 0.0, z: 1.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

Raw output check:
  rostopic echo -n 5 /firefly/command/raw_body_wrench

Motor speed check:
  rostopic echo -n 5 /firefly/command/motor_speed

Stop command:
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh" "${RUN_DIR}"
EOF
