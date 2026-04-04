#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
CATKIN_SRC="${WORKSPACE_ROOT}/src"
BLACKLIST_PACKAGES="rotors_hil_interface"

clone_or_update() {
  local name="$1"
  local url="$2"
  local target="${CATKIN_SRC}/${name}"
  if [[ ! -d "${target}/.git" ]]; then
    git clone "${url}" "${target}"
  else
    git -C "${target}" fetch --all --tags
  fi
}

link_local_package() {
  local package_name="$1"
  local source_dir="${V2_ROOT}/src/${package_name}"
  local target_dir="${CATKIN_SRC}/${package_name}"
  mkdir -p "${CATKIN_SRC}"
  ln -sfn "${source_dir}" "${target_dir}"
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended to run on Ubuntu Linux."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  git curl wget build-essential cmake pkg-config \
  python3-dev python3-pip python3-venv python3-rosdep python3-wstool \
  python3-catkin-tools liblapacke-dev protobuf-compiler libgoogle-glog-dev \
  libeigen3-dev ros-noetic-desktop-full ros-noetic-joy \
  ros-noetic-octomap-ros ros-noetic-mavlink ros-noetic-mavros \
  ros-noetic-control-toolbox

sudo rosdep init || true
rosdep update

mkdir -p "${CATKIN_SRC}"

clone_or_update "rotors_simulator" "https://github.com/ethz-asl/rotors_simulator.git"
clone_or_update "mav_control_rw" "https://github.com/ethz-asl/mav_control_rw.git"
clone_or_update "mav_comm" "https://github.com/ethz-asl/mav_comm.git"
clone_or_update "eigen_catkin" "https://github.com/ethz-asl/eigen_catkin.git"
clone_or_update "catkin_simple" "https://github.com/catkin/catkin_simple.git"

link_local_package "koopman_python"
link_local_package "koopman_mpc_ros"

source /opt/ros/noetic/setup.bash

cd "${WORKSPACE_ROOT}"
catkin config --extend /opt/ros/noetic --blacklist ${BLACKLIST_PACKAGES} --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build

echo
echo "Workspace bootstrap complete."
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Repo root: ${REPO_ROOT}"
echo "Active V2 root: ${V2_ROOT}"
