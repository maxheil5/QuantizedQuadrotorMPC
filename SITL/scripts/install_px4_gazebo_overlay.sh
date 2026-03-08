#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_CONFIG="${1:-${ROOT_DIR}/configs/gazebo/quantized_koopman_quad.yaml}"
PX4_DIR="${PX4_DIR:-${ROOT_DIR}/artifacts/external/PX4-Autopilot}"
PX4_BUNDLED_MODELS_DIR="${PX4_DIR}/Tools/simulation/gz/models"
PX4_MODELS_DIR="${ROOT_DIR}/artifacts/external/PX4-gazebo-models/models"
LOCAL_CACHE_DIR="${HOME}/.simulation-gazebo/models"
OUTPUT_DIR="${ROOT_DIR}/artifacts/generated/gazebo_models"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}/src/quantized_quadrotor_sitl:${PYTHONPATH:-}"

python3 - <<'PY' "${OVERLAY_CONFIG}" "${PX4_BUNDLED_MODELS_DIR}" "${PX4_MODELS_DIR}" "${LOCAL_CACHE_DIR}" "${OUTPUT_DIR}"
from pathlib import Path
import sys

from quantized_quadrotor_sitl.utils.gazebo_overlay import install_overlay, load_overlay_config

config_path = Path(sys.argv[1])
px4_bundled_models_dir = Path(sys.argv[2])
px4_models_dir = Path(sys.argv[3])
local_cache_dir = Path(sys.argv[4])
output_dir = Path(sys.argv[5])

config = load_overlay_config(config_path)
candidate_dirs = [
    px4_bundled_models_dir / config.source_model_name,
    px4_models_dir / config.source_model_name,
    local_cache_dir / config.source_model_name,
]
source_dir = next((path for path in candidate_dirs if path.exists()), None)
if source_dir is None:
    raise FileNotFoundError(
        "Could not locate the source PX4 Gazebo model. Expected one of: "
        + ", ".join(str(path) for path in candidate_dirs)
    )
output_dir.mkdir(parents=True, exist_ok=True)
installed = install_overlay(source_dir, output_dir, config)
print(installed)
PY
