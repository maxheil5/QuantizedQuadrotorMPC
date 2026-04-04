#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${REPO_ROOT}/.." && pwd)}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${V2_ROOT}/docs/environment_snapshots/${STAMP}"
CATKIN_SRC="${WORKSPACE_ROOT}/src"

mkdir -p "${OUT_DIR}"

{
  echo "timestamp_utc=${STAMP}"
  echo "repo_root=${REPO_ROOT}"
  echo "workspace_root=${WORKSPACE_ROOT}"
} > "${OUT_DIR}/snapshot.env"

if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -a > "${OUT_DIR}/lsb_release.txt" 2>&1 || true
fi

uname -a > "${OUT_DIR}/uname.txt"
env | sort > "${OUT_DIR}/environment.txt"

if command -v rosversion >/dev/null 2>&1; then
  rosversion -d > "${OUT_DIR}/ros_distro.txt" 2>&1 || true
fi

if command -v gazebo >/dev/null 2>&1; then
  gazebo --version > "${OUT_DIR}/gazebo_version.txt" 2>&1 || true
fi

if command -v catkin >/dev/null 2>&1; then
  catkin config > "${OUT_DIR}/catkin_config.txt" 2>&1 || true
fi

if command -v python3 >/dev/null 2>&1; then
  python3 -m pip freeze > "${OUT_DIR}/pip_freeze.txt" 2>&1 || true
fi

apt-mark showmanual | sort > "${OUT_DIR}/apt_manual_packages.txt" 2>&1 || true
dpkg-query -W -f='${binary:Package}\t${Version}\n' | sort > "${OUT_DIR}/dpkg_packages.tsv" 2>&1 || true

{
  echo "rotors_simulator"
  git -C "${CATKIN_SRC}/rotors_simulator" rev-parse HEAD 2>/dev/null || true
  echo "mav_control_rw"
  git -C "${CATKIN_SRC}/mav_control_rw" rev-parse HEAD 2>/dev/null || true
  echo "mav_comm"
  git -C "${CATKIN_SRC}/mav_comm" rev-parse HEAD 2>/dev/null || true
  echo "eigen_catkin"
  git -C "${CATKIN_SRC}/eigen_catkin" rev-parse HEAD 2>/dev/null || true
  echo "catkin_simple"
  git -C "${CATKIN_SRC}/catkin_simple" rev-parse HEAD 2>/dev/null || true
} > "${OUT_DIR}/workspace_commits.txt"

echo "Environment snapshot written to ${OUT_DIR}"

