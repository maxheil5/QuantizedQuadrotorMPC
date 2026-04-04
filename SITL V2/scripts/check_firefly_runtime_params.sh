#!/usr/bin/env bash

set -euo pipefail

NODE_NS="${1:-/firefly/lee_position_controller_node}"
OUTPUT_PATH="${2:-}"

if ! command -v rosparam >/dev/null 2>&1; then
  echo "rosparam not found. Source /opt/ros/noetic/setup.bash and your workspace first." >&2
  exit 1
fi

if ! rosparam list >/dev/null 2>&1; then
  echo "ROS master is not reachable. Start Gazebo/controller launch files first." >&2
  exit 1
fi

if ! rosparam get "${NODE_NS}/mass" >/dev/null 2>&1; then
  echo "Runtime parameters for ${NODE_NS} are not available yet." >&2
  echo "Expected the controller stack to be running before this check." >&2
  exit 1
fi

emit_report() {
  echo "node_namespace: ${NODE_NS}"
  echo "checked_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mass:"
  rosparam get "${NODE_NS}/mass"
  echo "inertia:"
  rosparam get "${NODE_NS}/inertia"
  if rosparam get "${NODE_NS}/rotor_configuration" >/dev/null 2>&1; then
    echo "rotor_configuration:"
    rosparam get "${NODE_NS}/rotor_configuration"
  fi
}

if [[ -n "${OUTPUT_PATH}" ]]; then
  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  emit_report | tee "${OUTPUT_PATH}"
else
  emit_report
fi
