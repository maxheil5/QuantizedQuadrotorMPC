from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .sitl_runtime_health import compute_runtime_health_summary


HOVER_GATE_PROFILES: dict[str, dict[str, float]] = {
    "light": {
        "z_max_min": 0.75,
        "final_altitude_error_max": 0.50,
        "max_lateral_radius_max": 3.0,
        "solver_mean_ms_max": 15.0,
        "post_4s_internal_bound_fraction_max": 0.15,
        "early_window_rmse_ratio_x_max": 7.0,
        "early_window_rmse_ratio_dx_max": 3.0,
        "early_window_rmse_ratio_wb_max": 3.0,
    },
    "standard": {
        "final_altitude_error_max": 0.15,
        "max_lateral_radius_max": 0.30,
        "torque_near_limit_fraction_max": 0.20,
        "solver_mean_ms_max": 8.0,
        "tick_mean_ms_max": 12.0,
    },
    "light_anchor_confirmation": {
        "runtime_sample_count_min": 440.0,
        "runtime_effective_control_rate_hz_min": 45.0,
        "runtime_tick_dt_ms_p90_max": 25.0,
        "runtime_solver_ms_p90_max": 18.0,
        "final_altitude_error_max": 0.15,
        "max_lateral_error_radius_max": 0.75,
        "final_position_error_max": 0.75,
        "position_rmse_max": 0.40,
        "solver_mean_ms_max": 12.0,
        "post_4s_internal_bound_fraction_u1_max": 0.05,
        "post_4s_internal_bound_fraction_u2_max": 0.05,
        "post_4s_internal_bound_fraction_u3_max": 0.05,
        "u2_first_used_mismatch_min_s": 8.0,
        "require_divergence_to_reach_end": 1.0,
    },
    "standard_anchor_h8_promotion": {
        "runtime_sample_count_min": 850.0,
        "runtime_effective_control_rate_hz_min": 85.0,
        "runtime_tick_dt_ms_p90_max": 14.0,
        "runtime_solver_ms_p90_max": 10.0,
        "final_altitude_error_max": 0.20,
        "max_lateral_error_radius_max": 1.50,
        "final_position_error_max": 1.50,
        "position_rmse_max": 0.75,
        "solver_mean_ms_max": 10.0,
        "post_4s_internal_bound_fraction_u1_max": 0.10,
        "post_4s_internal_bound_fraction_u2_max": 0.10,
        "post_4s_internal_bound_fraction_u3_max": 0.10,
        "require_divergence_to_reach_end": 1.0,
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
    lateral_error_radius = np.linalg.norm(position_error[0:2, :], axis=0)
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
        "max_lateral_error_radius": float(np.max(lateral_error_radius)),
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


def load_control_audit_summary(log_path: Path, control_audit_summary_path: Path | None = None) -> dict[str, object] | None:
    summary_path = (
        Path(control_audit_summary_path)
        if control_audit_summary_path is not None
        else Path(log_path).parent / "control_audit_summary.json"
    )
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else None


def load_runtime_health_summary(log_path: Path, runtime_health_summary_path: Path | None = None) -> dict[str, object]:
    summary_path = (
        Path(runtime_health_summary_path)
        if runtime_health_summary_path is not None
        else Path(log_path).parent / "runtime_health_summary.json"
    )
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            return payload
    return compute_runtime_health_summary(Path(log_path))


def evaluate_hover_gates(
    log_path: Path,
    profile: str,
    drift_summary_path: Path | None = None,
    control_audit_summary_path: Path | None = None,
    runtime_health_summary_path: Path | None = None,
) -> dict[str, object]:
    if profile not in HOVER_GATE_PROFILES:
        raise ValueError(f"unsupported hover-gate profile: {profile}")

    thresholds = HOVER_GATE_PROFILES[profile]
    metrics = compute_hover_gate_metrics(log_path)
    runtime_health_summary = load_runtime_health_summary(log_path, runtime_health_summary_path)
    drift_summary = load_drift_summary(log_path, drift_summary_path)
    control_audit_summary = load_control_audit_summary(log_path, control_audit_summary_path)
    checks: dict[str, bool] = {}
    runtime_checks: dict[str, bool] = {}
    if "runtime_sample_count_min" in thresholds:
        runtime_checks["runtime_sample_count"] = float(runtime_health_summary["sample_count"]) >= thresholds["runtime_sample_count_min"]
    if "runtime_effective_control_rate_hz_min" in thresholds:
        runtime_checks["runtime_effective_control_rate_hz"] = (
            float(runtime_health_summary["effective_control_rate_hz"]) >= thresholds["runtime_effective_control_rate_hz_min"]
        )
    if "runtime_tick_dt_ms_p90_max" in thresholds:
        runtime_checks["runtime_tick_dt_ms_p90"] = (
            float(runtime_health_summary["tick_dt_ms_p90"]) <= thresholds["runtime_tick_dt_ms_p90_max"]
        )
    if "runtime_solver_ms_p90_max" in thresholds:
        runtime_checks["runtime_solver_ms_p90"] = (
            float(runtime_health_summary["solver_ms_p90"]) <= thresholds["runtime_solver_ms_p90_max"]
        )
    if runtime_checks:
        runtime_checks["runtime_validity"] = runtime_health_summary.get("runtime_validity") == "valid_runtime"
        if not all(runtime_checks.values()):
            return {
                "profile": profile,
                "passed": False,
                "checks": runtime_checks,
                "metrics": metrics,
                "runtime_validity": runtime_health_summary.get("runtime_validity"),
                "runtime_failure_reason": runtime_health_summary.get("runtime_failure_reason"),
                "controller_quality_evaluated": False,
                "diagnostic_branch": None,
                "dominant_drift_channel": None,
                "early_window_rmse_ratio": None,
                "post_4s_internal_bound_fraction": None,
                "control_audit_mapping_status": None,
                "control_audit_dominant_mismatch_axis": None,
                "u2_first_used_mismatch_time_s": None,
            }
    if "z_max_min" in thresholds:
        checks["z_max"] = metrics["z_max"] >= thresholds["z_max_min"]
    if "final_altitude_error_max" in thresholds:
        checks["final_altitude_error"] = metrics["final_altitude_error"] <= thresholds["final_altitude_error_max"]
    if "max_lateral_radius_max" in thresholds:
        checks["max_lateral_radius"] = metrics["max_lateral_radius"] <= thresholds["max_lateral_radius_max"]
    if "max_lateral_error_radius_max" in thresholds:
        checks["max_lateral_error_radius"] = metrics["max_lateral_error_radius"] <= thresholds["max_lateral_error_radius_max"]
    if "final_position_error_max" in thresholds:
        checks["final_position_error"] = metrics["final_position_error"] <= thresholds["final_position_error_max"]
    if "position_rmse_max" in thresholds:
        checks["position_rmse"] = metrics["position_rmse"] <= thresholds["position_rmse_max"]
    if "torque_near_limit_fraction_max" in thresholds:
        checks["u1_near_limit_fraction"] = metrics["u1_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
        checks["u2_near_limit_fraction"] = metrics["u2_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
        checks["u3_near_limit_fraction"] = metrics["u3_near_limit_fraction"] <= thresholds["torque_near_limit_fraction_max"]
    checks["solver_mean_ms"] = metrics["solver_mean_ms"] <= thresholds["solver_mean_ms_max"]
    if "tick_mean_ms_max" in thresholds:
        checks["tick_mean_ms"] = metrics["tick_mean_ms"] <= thresholds["tick_mean_ms_max"]
    if drift_summary is not None:
        post_four = drift_summary.get("post_4s_internal_bound_fraction", {})
        if thresholds.get("require_divergence_to_reach_end"):
            divergence_time_s = drift_summary.get("divergence_time_s")
            if divergence_time_s is not None:
                checks["divergence_reaches_end"] = float(divergence_time_s) >= float(metrics["final_time_s"]) - 1.0e-6
        if isinstance(post_four, dict) and "post_4s_internal_bound_fraction_max" in thresholds:
            for key in ("u0", "u1", "u2", "u3"):
                if key in post_four:
                    checks[f"post_4s_internal_bound_fraction_{key}"] = float(post_four[key]) <= thresholds["post_4s_internal_bound_fraction_max"]
        if isinstance(post_four, dict):
            for key in ("u1", "u2", "u3"):
                threshold_key = f"post_4s_internal_bound_fraction_{key}_max"
                if threshold_key in thresholds and key in post_four:
                    checks[f"post_4s_internal_bound_fraction_{key}"] = float(post_four[key]) <= thresholds[threshold_key]
        early_ratios = drift_summary.get("early_window_rmse_ratio", {})
        if isinstance(early_ratios, dict):
            for key in ("x", "dx", "wb"):
                threshold_key = f"early_window_rmse_ratio_{key}_max"
                if threshold_key in thresholds and key in early_ratios:
                    checks[f"early_window_rmse_ratio_{key}"] = float(early_ratios[key]) <= thresholds[threshold_key]
    if control_audit_summary is not None and "u2_first_used_mismatch_min_s" in thresholds:
        mismatch_time = control_audit_summary.get("u2_first_used_mismatch_time_s")
        checks["u2_first_used_mismatch_time_s"] = mismatch_time is None or float(mismatch_time) >= thresholds["u2_first_used_mismatch_min_s"]
    return {
        "profile": profile,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "runtime_validity": runtime_health_summary.get("runtime_validity"),
        "runtime_failure_reason": runtime_health_summary.get("runtime_failure_reason"),
        "controller_quality_evaluated": True,
        "diagnostic_branch": None if drift_summary is None else drift_summary.get("selected_branch"),
        "dominant_drift_channel": None if drift_summary is None else drift_summary.get("dominant_error_group"),
        "early_window_rmse_ratio": None if drift_summary is None else drift_summary.get("early_window_rmse_ratio"),
        "post_4s_internal_bound_fraction": None if drift_summary is None else drift_summary.get("post_4s_internal_bound_fraction"),
        "control_audit_mapping_status": None if control_audit_summary is None else control_audit_summary.get("mapping_status"),
        "control_audit_dominant_mismatch_axis": None if control_audit_summary is None else control_audit_summary.get("dominant_mismatch_axis"),
        "u2_first_used_mismatch_time_s": None if control_audit_summary is None else control_audit_summary.get("u2_first_used_mismatch_time_s"),
    }
