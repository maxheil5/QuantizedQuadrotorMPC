#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
MODEL_PATH="${2:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${V2_ROOT}/results/runtime_logs/koopman_hover/${STAMP}"
PID_FILE="${RUN_DIR}/pids.env"

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

master_started_by_script=0
gazebo_started_by_script=0
koopman_started_by_script=0
roscore_pid=""
gazebo_pid=""
koopman_pid=""

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
  echo "Using existing Gazebo node."
else
  echo "Starting Gazebo Firefly stack."
  nohup roslaunch rotors_gazebo mav.launch mav_name:=firefly > "${RUN_DIR}/gazebo.log" 2>&1 &
  gazebo_pid="$!"
  gazebo_started_by_script=1
  wait_for_success "rosservice list | grep -q '^/gazebo/unpause_physics$'" "Gazebo services"
fi

echo "Unpausing Gazebo physics."
rosservice call /gazebo/unpause_physics "{}" > "${RUN_DIR}/unpause.log" 2>&1 || true
wait_for_success "rostopic list | grep -q '^/clock$'" "clock topic"
wait_for_success "rostopic list | grep -q '^/firefly/ground_truth/odometry$'" "ground-truth odometry topic"

if rosnode list 2>/dev/null | grep -q '^/firefly/koopman_mpc_node$'; then
  echo "Stopping existing /firefly/koopman_mpc_node."
  rosnode kill /firefly/koopman_mpc_node >/dev/null 2>&1 || true
  sleep 1
fi

echo "Starting learned hover node."
nohup env ROS_NAMESPACE=firefly rosrun koopman_mpc_ros koopman_mpc_node.py __name:=koopman_mpc_node "_model_path:=${MODEL_PATH}" _parameter_profile:=rotors_firefly_linear_mpc_runtime _pred_horizon:=10 _qp_max_iter:=100 odometry:=ground_truth/odometry > "${RUN_DIR}/koopman_mpc_node.log" 2>&1 &
koopman_pid="$!"
koopman_started_by_script=1
wait_for_success "rosnode list | grep -q '^/firefly/koopman_mpc_node$'" "koopman_mpc_node"

{
  echo "run_dir='${RUN_DIR}'"
  echo "workspace_root='${WORKSPACE_ROOT}'"
  echo "model_path='${MODEL_PATH}'"
  echo "master_started_by_script='${master_started_by_script}'"
  echo "gazebo_started_by_script='${gazebo_started_by_script}'"
  echo "koopman_started_by_script='${koopman_started_by_script}'"
  echo "roscore_pid='${roscore_pid}'"
  echo "gazebo_pid='${gazebo_pid}'"
  echo "koopman_pid='${koopman_pid}'"
} > "${PID_FILE}"

cat <<EOF
koopman_hover_stack_run_dir=${RUN_DIR}
Logs:
  ${RUN_DIR}/roscore.log
  ${RUN_DIR}/gazebo.log
  ${RUN_DIR}/koopman_mpc_node.log

Hover command:
  rostopic pub -1 /firefly/command/pose geometry_msgs/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 0.0, y: 0.0, z: 1.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

Raw output check:
  rostopic echo -n 5 /firefly/command/raw_body_wrench

Stop command:
  bash "${SCRIPT_DIR}/stop_koopman_hover_stack.sh" "${RUN_DIR}"
EOF
