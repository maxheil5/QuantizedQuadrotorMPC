#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOVE_FORTRESS_CONFLICTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-fortress-conflicts)
      REMOVE_FORTRESS_CONFLICTS=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f /etc/os-release ]]; then
  echo "This script targets Ubuntu 22.04." >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Expected Ubuntu 22.04, found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  curl \
  gnupg \
  lsb-release \
  ca-certificates \
  software-properties-common \
  git \
  build-essential \
  cmake \
  ninja-build \
  python3-venv \
  python3-pip \
  python3-colcon-common-extensions \
  python3-vcstool \
  python3-rosdep

if [[ ! -f /etc/apt/sources.list.d/ros2.list ]]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
fi

if [[ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]]; then
  sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable ${UBUNTU_CODENAME} main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
fi

sudo apt-get update
sudo apt-get install -y ros-humble-desktop gz-harmonic

if dpkg -l | grep -q '^ii  ros-humble-ros-gz'; then
  if [[ "${REMOVE_FORTRESS_CONFLICTS}" -eq 1 ]]; then
    sudo apt-get remove -y 'ros-humble-ros-gz*'
  else
    echo "Detected ros-humble-ros-gz packages. Re-run with --remove-fortress-conflicts to replace them with ros-humble-ros-gzharmonic." >&2
    exit 1
  fi
fi

sudo apt-get install -y ros-humble-ros-gzharmonic

sudo rosdep init >/dev/null 2>&1 || true
rosdep update

python3 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "numpy==1.24.4" \
  "scipy==1.10.1" \
  matplotlib \
  pandas \
  plotly \
  streamlit \
  pymavlink \
  pyyaml \
  osqp \
  pytest \
  catkin_pkg \
  empy \
  lark-parser

if [[ ! -d "${ROOT_DIR}/artifacts/external/PX4-Autopilot/.git" ]]; then
  git clone https://github.com/PX4/PX4-Autopilot.git "${ROOT_DIR}/artifacts/external/PX4-Autopilot"
fi
git -C "${ROOT_DIR}/artifacts/external/PX4-Autopilot" fetch --tags
git -C "${ROOT_DIR}/artifacts/external/PX4-Autopilot" checkout v1.16.0
git -C "${ROOT_DIR}/artifacts/external/PX4-Autopilot" submodule update --init --recursive

if [[ ! -d "${ROOT_DIR}/artifacts/external/PX4-gazebo-models/.git" ]]; then
  git clone https://github.com/PX4/PX4-gazebo-models.git "${ROOT_DIR}/artifacts/external/PX4-gazebo-models"
fi
git -C "${ROOT_DIR}/artifacts/external/PX4-gazebo-models" fetch
git -C "${ROOT_DIR}/artifacts/external/PX4-gazebo-models" checkout main

if [[ ! -d "${ROOT_DIR}/src/px4_msgs/.git" ]]; then
  git clone https://github.com/PX4/px4_msgs.git "${ROOT_DIR}/src/px4_msgs"
fi
git -C "${ROOT_DIR}/src/px4_msgs" fetch
git -C "${ROOT_DIR}/src/px4_msgs" checkout release/1.16

if [[ ! -d "${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent/.git" ]]; then
  git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent"
fi
cmake -S "${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent" -B "${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent/build"
cmake --build "${ROOT_DIR}/artifacts/external/Micro-XRCE-DDS-Agent/build" -j"$(nproc)"

bash "${ROOT_DIR}/artifacts/external/PX4-Autopilot/Tools/setup/ubuntu.sh" --no-sim-tools --no-nuttx
python -m pip install -e "${ROOT_DIR}/src/quantized_quadrotor_sitl"
bash "${ROOT_DIR}/scripts/install_px4_gazebo_overlay.sh"

source /opt/ros/humble/setup.bash
rosdep install --from-paths "${ROOT_DIR}/src" --ignore-src -r -y
colcon build --symlink-install --base-paths "${ROOT_DIR}/src"

echo "Setup complete."
