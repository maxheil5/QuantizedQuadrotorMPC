from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HOVER_GATE_PROFILES: dict[str, dict[str, float]] = {
    "light": {
        "z_max_min": 0.60,
        "final_altitude_error_max": 0.15,
        "max_lateral_radius_max": 0.50,
        "torque_near_limit_fraction_max": 0.25,
        "solver_mean_ms_max": 15.0,
    },
    "standard": {
        "final_altitude_error_max": 0.15,
        "max_lateral_radius_max": 0.30,
        "torque_near_limit_fraction_max": 0.20,
        "solver_mean_ms_max": 8.0,
        "tick_mean_ms_max": 12.0,
    },
}


def _column_history(rows: list[dict[str, str]], prefix: str, width: int) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_{idx}"]) for row in rows] for idx in range(width)],
        dtype=float,
    )


def compute_hover_gate_metrics(log_path: Path) -> dict[str, float]:
    resolved = Path(log_path)
    with resolved.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{resolved} does not contain any runtime samples")

    state_history = _column_history(rows, "state_used", 18)
    reference_history = _column_history(rows, "reference", 18)
    control_history = _column_history(rows, "control_used", 4)
    solver_ms = np.asarray([float(row["solver_ms"]) for row in rows], dtype=float)
    tick_dt_ms = np.asarray([float(row["tick_dt_ms"]) for row in rows], dtype=float)
    experiment_time_s = np.asarray([float(row["experiment_time_s"]) for row in rows], dtype=float)

    position_error = state_history[0:3, :] - reference_history[0:3, :]
    lateral_radius = np.linalg.norm(state_history[0:2, :], axis=0)
    final_altitude_error = abs(float(state_history[2, -1] - reference_history[2, -1]))

    metadata_path = resolved.parent / "run_metadata.json"
    torque_limits = np.array([1.0, 1.0, 0.6], dtype=float)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        scaling = metadata.get("vehicle_scaling", {})
        torque_limits = np.array(
            [
                float(scaling.get("max_body_torque_x_nm", 1.0)),
                float(scaling.get("max_body_torque_y_nm", 1.0)),
                float(scaling.get("max_body_torque_z_nm", 0.6)),
            ],
            dtype=float,
        )

    near_limit_fraction = np.mean(
        np.abs(control_history[1:4, :]) >= 0.98 * torque_limits.reshape(3, 1),
        axis=1,
    )
    return {
        "sample_count": float(state_history.shape[1]),
        "final_time_s": float(experiment_time_s[-1]),
        "z_max": float(np.max(state_history[2, :])),
        "final_altitude_error": final_altitude_error,
        "max_lateral_radius": float(np.max(lateral_radius)),
        "final_position_error": float(np.linalg.norm(position_error[:, -1])),
        "position_rmse": float(np.sqrt(np.mean(np.sum(position_error**2, axis=0)))),
        "solver_mean_ms": float(np.mean(solver_ms)),
        "tick_mean_ms": float(np.mean(tick_dt_ms)),
        "u1_near_limit_fraction": float(near_limit_fraction[0]),
        "u2_near_limit_fraction": float(near_limit_fraction[1]),
        "u3_near_limit_fraction": float(near_limit_fraction[2]),
    }


def load_drift_summary(log_path: Path, drift_summary_path: Path | None = None) -> dict[str, object] | None:
    summary_path = Path(drift_summary_path) if drift_summary_path is not None else Path(log_path).parent / "drift_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else None


def evaluate_hover_gates(log_path: Path, profile: str, drift_summary_path: Path | None = None) -> dict[str, object]:
    if profile not in HOVER_GATE_PROFILES:
        raise ValueError(f"unsupported hover-gate profile: {profile}")

    thresholds = HOVER_GATE_PROFILES[profile]
    metrics = compute_hover_gate_metrics(log_path)
    drift_summary = load_drift_summary(log_path, drift_summary_path)
    checks: dict[str, bool] = {}
    if "z_max_min" in thresholds:
        checks["z_max"] = metrics["z_max"] >= thresholds["z_max_min"]
    checks["final_altitude_error"] = metrics["final_altitude_error"] <= thresholds["final_altitude_error_max"]
    checks["max_lateral_radius"] = metrics["max_lateral_radius"] <= thresholds["max_lateral_radius_max"]
    checks["u1_near_limit_fraction"] = metrics["u1_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
    checks["u2_near_limit_fraction"] = metrics["u2_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
    checks["u3_near_limit_fraction"] = metrics["u3_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
    checks["solver_mean_ms"] = metrics["solver_mean_ms"] <= thresholds["solver_mean_ms_max"]
    if "tick_mean_ms_max" in thresholds:
        checks["tick_mean_ms"] = metrics["tick_mean_ms"] <= thresholds["tick_mean_ms_max"]
    return {
        "profile": profile,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "diagnostic_branch": None if drift_summary is None else drift_summary.get("selected_branch"),
        "dominant_drift_channel": None if drift_summary is None else drift_summary.get("dominant_error_group"),
    }
