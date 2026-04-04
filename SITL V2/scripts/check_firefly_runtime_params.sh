#!/usr/bin/env bash

set -euo pipefail

ROOT_NS="${1:-/firefly}"
OUTPUT_PATH="${2:-}"

if ! command -v rosparam >/dev/null 2>&1; then
  echo "rosparam not found. Source /opt/ros/noetic/setup.bash and your workspace first." >&2
  exit 1
fi

if ! rosparam list >/dev/null 2>&1; then
  echo "ROS master is not reachable. Start Gazebo/controller launch files first." >&2
  exit 1
fi

mapfile -t MASS_KEYS < <(rosparam list | grep "^${ROOT_NS}/.*/mass$" || true)
mapfile -t INERTIA_KEYS < <(rosparam list | grep "^${ROOT_NS}/.*/inertia$" || true)
mapfile -t ROTOR_KEYS < <(rosparam list | grep "^${ROOT_NS}/.*/rotor_configuration$" || true)

if [[ ${#MASS_KEYS[@]} -eq 0 && ${#INERTIA_KEYS[@]} -eq 0 ]]; then
  echo "No runtime mass/inertia parameters were found under ${ROOT_NS}." >&2
  echo "Expected the Gazebo + controller stack to be running before this check." >&2
  echo "You can inspect candidates manually with: rosparam list | grep '^${ROOT_NS}/'" >&2
  exit 1
fi

emit_key_block() {
  local title="$1"
  shift
  local keys=("$@")
  echo "${title}:"
  if [[ ${#keys[@]} -eq 0 ]]; then
    echo "  []"
    return
  fi

  local key
  for key in "${keys[@]}"; do
    echo "  - key: ${key}"
    echo "    value:"
    rosparam get "${key}" | sed 's/^/      /'
  done
}

emit_report() {
  echo "root_namespace: ${ROOT_NS}"
  echo "checked_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  emit_key_block "mass_parameters" "${MASS_KEYS[@]}"
  emit_key_block "inertia_parameters" "${INERTIA_KEYS[@]}"
  emit_key_block "rotor_configuration_parameters" "${ROTOR_KEYS[@]}"
}

if [[ -n "${OUTPUT_PATH}" ]]; then
  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  emit_report | tee "${OUTPUT_PATH}"
else
  emit_report
fi
