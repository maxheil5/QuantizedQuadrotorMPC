from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantized_quadrotor_sitl.core.types import ParsedStateSeries
from quantized_quadrotor_sitl.utils.state import parse_state18_history


DEFAULT_RUN_NAME = "3-21-26_1735_base_controller"
STATE_WIDTH = 18
CONTROL_WIDTH = 4
PX4_COLUMNS = [
    "px4_collective_command_newton",
    "px4_collective_normalized",
    "px4_thrust_body_z",
]


@dataclass(slots=True)
class DashboardRunData:
    run_name: str
    run_dir: Path
    log_path: Path
    metadata: dict[str, Any] | None
    frame: pd.DataFrame
    time_s: np.ndarray
    state_source: str
    control_source: str
    state_history: np.ndarray
    reference_history: np.ndarray
    control_history: np.ndarray
    parsed_state: ParsedStateSeries
    parsed_reference: ParsedStateSeries
    position_error: np.ndarray
    velocity_error: np.ndarray
    attitude_error: np.ndarray
    angular_velocity_error: np.ndarray
    position_error_norm: np.ndarray
    velocity_error_norm: np.ndarray
    lateral_deviation: np.ndarray
    px4_available: bool
    summary: dict[str, Any]


def default_results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "sitl"


def discover_run_names(results_root: Path | None = None) -> list[str]:
    root = Path(results_root) if results_root is not None else default_results_root()
    if not root.exists():
        return []
    run_names = [path.parent.name for path in root.glob("*/runtime_log.csv")]
    return sorted(run_names, reverse=True)


def default_run_name(run_names: list[str]) -> str | None:
    if not run_names:
        return None
    if DEFAULT_RUN_NAME in run_names:
        return DEFAULT_RUN_NAME
    return run_names[0]


def run_directory(run_name: str, results_root: Path | None = None) -> Path:
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / run_name


def log_path_for_run(run_name: str, results_root: Path | None = None) -> Path:
    return run_directory(run_name, results_root) / "runtime_log.csv"


def metadata_path_for_run(run_name: str, results_root: Path | None = None) -> Path:
    return run_directory(run_name, results_root) / "run_metadata.json"


def load_run_frame(run_name: str, results_root: Path | None = None) -> pd.DataFrame:
    path = log_path_for_run(run_name, results_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing runtime log for run '{run_name}': {path}")
    return pd.read_csv(path)


def load_run_metadata(run_name: str, results_root: Path | None = None) -> dict[str, Any] | None:
    path = metadata_path_for_run(run_name, results_root)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def available_sources(frame: pd.DataFrame, prefix: str, width: int) -> list[str]:
    sources: list[str] = []
    for source in ("raw", "used"):
        expected = [f"{prefix}_{source}_{idx}" for idx in range(width)]
        if all(column in frame.columns for column in expected):
            sources.append(source)
    return sources


def _extract_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=float).T


def _real_array(values: np.ndarray) -> np.ndarray:
    resolved = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(resolved):
        resolved = resolved.real
    return np.asarray(resolved, dtype=float)


def _summary_metrics(
    run_name: str,
    time_s: np.ndarray,
    state_history: np.ndarray,
    reference_history: np.ndarray,
    position_error_norm: np.ndarray,
    lateral_deviation: np.ndarray,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    position_rmse = float(np.sqrt(np.mean(np.sum((state_history[0:3, :] - reference_history[0:3, :]) ** 2, axis=0))))
    final_position_error = float(position_error_norm[-1]) if position_error_norm.size else 0.0
    final_time = float(time_s[-1]) if time_s.size else 0.0
    max_altitude = float(np.max(state_history[2, :])) if state_history.size else 0.0
    max_lateral_deviation = float(np.max(lateral_deviation)) if lateral_deviation.size else 0.0
    summary = {
        "run_name": run_name,
        "sample_count": int(time_s.size),
        "final_time_s": final_time,
        "position_rmse": position_rmse,
        "final_position_error": final_position_error,
        "max_altitude": max_altitude,
        "max_lateral_deviation": max_lateral_deviation,
    }
    if metadata:
        summary["controller_mode"] = metadata.get("controller_mode", "unknown")
        summary["reference_mode"] = metadata.get("reference_mode", "unknown")
        summary["quantization_mode"] = metadata.get("quantization_mode", "unknown")
    return summary


def prepare_run_data(
    run_name: str,
    state_source: str,
    control_source: str,
    results_root: Path | None = None,
) -> DashboardRunData:
    frame = load_run_frame(run_name, results_root)
    metadata = load_run_metadata(run_name, results_root)

    state_options = available_sources(frame, "state", STATE_WIDTH)
    control_options = available_sources(frame, "control", CONTROL_WIDTH)
    if state_source not in state_options:
        raise ValueError(f"state source '{state_source}' is unavailable for run '{run_name}'")
    if control_source not in control_options:
        raise ValueError(f"control source '{control_source}' is unavailable for run '{run_name}'")

    time_s = frame["experiment_time_s"].to_numpy(dtype=float)
    state_history = _extract_matrix(frame, [f"state_{state_source}_{idx}" for idx in range(STATE_WIDTH)])
    reference_history = _extract_matrix(frame, [f"reference_{idx}" for idx in range(STATE_WIDTH)])
    control_history = _extract_matrix(frame, [f"control_{control_source}_{idx}" for idx in range(CONTROL_WIDTH)])

    parsed_state = parse_state18_history(state_history)
    parsed_reference = parse_state18_history(reference_history)
    parsed_state.theta = _real_array(parsed_state.theta)
    parsed_reference.theta = _real_array(parsed_reference.theta)
    parsed_state.wb = _real_array(parsed_state.wb)
    parsed_reference.wb = _real_array(parsed_reference.wb)
    parsed_state.x = _real_array(parsed_state.x)
    parsed_reference.x = _real_array(parsed_reference.x)
    parsed_state.dx = _real_array(parsed_state.dx)
    parsed_reference.dx = _real_array(parsed_reference.dx)

    position_error = state_history[0:3, :] - reference_history[0:3, :]
    velocity_error = state_history[3:6, :] - reference_history[3:6, :]
    attitude_error = parsed_state.theta - parsed_reference.theta
    angular_velocity_error = parsed_state.wb - parsed_reference.wb
    position_error_norm = np.linalg.norm(position_error, axis=0)
    velocity_error_norm = np.linalg.norm(velocity_error, axis=0)
    lateral_center = reference_history[0:2, 0:1]
    lateral_deviation = np.linalg.norm(state_history[0:2, :] - lateral_center, axis=0)

    px4_available = all(column in frame.columns for column in PX4_COLUMNS)
    summary = _summary_metrics(
        run_name,
        time_s,
        state_history,
        reference_history,
        position_error_norm,
        lateral_deviation,
        metadata,
    )
    return DashboardRunData(
        run_name=run_name,
        run_dir=run_directory(run_name, results_root),
        log_path=log_path_for_run(run_name, results_root),
        metadata=metadata,
        frame=frame,
        time_s=time_s,
        state_source=state_source,
        control_source=control_source,
        state_history=state_history,
        reference_history=reference_history,
        control_history=control_history,
        parsed_state=parsed_state,
        parsed_reference=parsed_reference,
        position_error=position_error,
        velocity_error=velocity_error,
        attitude_error=attitude_error,
        angular_velocity_error=angular_velocity_error,
        position_error_norm=position_error_norm,
        velocity_error_norm=velocity_error_norm,
        lateral_deviation=lateral_deviation,
        px4_available=px4_available,
        summary=summary,
    )
