from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.experiments.sitl_u2_root_cause import analyze_runtime_u2_root_cause


def _identity_state18(x_pos: float = 0.0, z_pos: float = 0.75) -> list[float]:
    state = [0.0] * 18
    state[0] = x_pos
    state[2] = z_pos
    state[6] = 1.0
    state[10] = 1.0
    state[14] = 1.0
    return state


def _write_runtime_log(path: Path, *, x_positions: list[float], z_positions: list[float], raw_u2: list[float], used_u2: list[float]) -> None:
    header = [
        "step",
        "timestamp_ns",
        "experiment_time_s",
        "reference_index",
        "tick_dt_ms",
        "solver_ms",
        "px4_collective_command_newton",
        "px4_collective_normalized",
        "px4_thrust_body_z",
        *[f"state_raw_{idx}" for idx in range(18)],
        *[f"state_used_{idx}" for idx in range(18)],
        *[f"control_raw_{idx}" for idx in range(4)],
        *[f"control_internal_{idx}" for idx in range(4)],
        *[f"control_used_{idx}" for idx in range(4)],
        *[f"reference_{idx}" for idx in range(18)],
    ]
    rows = []
    for step, (x_pos, z_pos, raw_u2_value, used_u2_value) in enumerate(zip(x_positions, z_positions, raw_u2, used_u2, strict=True)):
        state = _identity_state18(x_pos, z_pos)
        reference = _identity_state18(0.0, 0.75)
        raw_control = [43.0, 0.0, raw_u2_value, 0.0]
        used_control = [43.0, 0.0, used_u2_value, 0.0]
        row = {
            "step": step,
            "timestamp_ns": 1_000_000 * step,
            "experiment_time_s": 2.0 * step,
            "reference_index": step,
            "tick_dt_ms": 20.0,
            "solver_ms": 8.0,
            "px4_collective_command_newton": 43.0,
            "px4_collective_normalized": 43.0 / 62.0,
            "px4_thrust_body_z": -(43.0 / 62.0),
        }
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"state_used_{idx}"] = state[idx]
            row[f"reference_{idx}"] = reference[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = raw_control[idx]
            row[f"control_internal_{idx}"] = raw_control[idx]
            row[f"control_used_{idx}"] = used_control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path, artifact_path: str) -> None:
    payload = {
        "config_path": "/tmp/runtime.yaml",
        "controller_mode": "edmd_mpc",
        "control_rate_hz": 50.0,
        "pred_horizon": 8,
        "reference_mode": "takeoff_hold",
        "reference_seed": 2141444,
        "reference_duration_s": 10.0,
        "cost_state_mode": "decoded24_raw",
        "model_artifact": artifact_path,
        "quantization_mode": "none",
        "learned_bound_margin_fraction": 0.05,
        "vehicle_scaling": {
            "max_collective_thrust_newton": 62.0,
            "max_body_torque_x_nm": 1.0,
            "max_body_torque_y_nm": 1.0,
            "max_body_torque_z_nm": 0.6,
        },
        "baseline": {
            "position_gains_diag": [0.5, 0.5, 5.5],
            "velocity_gains_diag": [0.8, 0.8, 3.0],
            "attitude_gains_diag": [2.5, 2.5, 0.8],
            "angular_rate_gains_diag": [0.25, 0.25, 0.15],
            "z_integral_gain": 1.2,
            "z_integral_limit": 1.5,
            "max_tilt_deg": 12.0,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifact(path: Path) -> None:
    model = EDMDModel(
        A=np.eye(24, dtype=float),
        B=np.zeros((24, 4), dtype=float),
        C=np.eye(24, dtype=float),
        Z1=np.zeros((24, 1), dtype=float),
        Z2=np.zeros((24, 1), dtype=float),
        n_basis=0,
    )
    np.savez(
        path,
        A=model.A,
        B=model.B,
        C=model.C,
        Z1=model.Z1,
        Z2=model.Z2,
        n_basis=np.array([model.n_basis]),
        u_train_min=np.array([[0.0], [-1.0], [-1.0], [-0.6]], dtype=float),
        u_train_max=np.array([[62.0], [1.0], [1.0], [0.6]], dtype=float),
        u_train_mean=np.array([[43.0], [0.0], [0.0], [0.0]], dtype=float),
        u_train_std=np.array([[5.0], [1.0], [1.0], [1.0]], dtype=float),
        u_trim=np.array([[43.0], [0.0], [0.0], [0.0]], dtype=float),
        affine_enabled=np.array([0.0], dtype=float),
        residual_enabled=np.array([0.0], dtype=float),
    )


def _write_runtime_health(path: Path, *, runtime_validity: str) -> None:
    payload = {
        "runtime_validity": runtime_validity,
        "runtime_failure_reason": None if runtime_validity == "valid_runtime" else "sample_count < 440",
        "sample_count": 5,
        "effective_control_rate_hz": 50.0,
        "tick_dt_ms_p90": 20.0,
        "solver_ms_p90": 8.0,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_drift_summary(path: Path, *, divergence_time_s: float) -> None:
    payload = {
        "divergence_time_s": divergence_time_s,
        "selected_branch": "Branch A",
        "dominant_error_group": "dx",
        "post_4s_internal_bound_fraction": {"u0": 0.0, "u1": 0.0, "u2": 0.0, "u3": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_control_audit_summary(
    path: Path,
    *,
    u2_first_raw_mismatch_time_s: float | None,
    u2_first_used_mismatch_time_s: float | None,
    raw_pre_sign: float,
    used_pre_sign: float,
    raw_late_sign: float,
    used_late_sign: float,
    raw_late_mag: float,
    used_late_mag: float,
    late_pos_norm: float,
    late_vel_norm: float,
) -> None:
    payload = {
        "mapping_status": "model/runtime issue",
        "anchor_mapping_status": "learned-controller/runtime issue",
        "dominant_mismatch_axis": "u2",
        "u2_first_raw_mismatch_time_s": u2_first_raw_mismatch_time_s,
        "u2_first_used_mismatch_time_s": u2_first_used_mismatch_time_s,
        "raw_pre_divergence_sign_match_fraction_by_axis": {"u2": raw_pre_sign},
        "used_pre_divergence_sign_match_fraction_by_axis": {"u2": used_pre_sign},
        "u2_late_window_mean_raw_sign_match": raw_late_sign,
        "u2_late_window_mean_used_sign_match": used_late_sign,
        "u2_late_window_mean_raw_magnitude_ratio": raw_late_mag,
        "u2_late_window_mean_used_magnitude_ratio": used_late_mag,
        "u2_late_window_mean_lateral_position_residual_norm": late_pos_norm,
        "u2_late_window_mean_lateral_velocity_residual_norm": late_vel_norm,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_analyze_runtime_u2_root_cause_classifies_raw_late_sign_instability(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "4-3-26_bad"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        x_positions=[0.0, 0.1, 0.2, 0.4, 0.6],
        z_positions=[0.75, 0.80, 0.88, 0.95, 1.05],
        raw_u2=[0.02, 0.02, 0.08, 0.10, 0.12],
        used_u2=[0.02, 0.02, 0.09, 0.11, 0.13],
    )
    _write_run_metadata(run_dir / "run_metadata.json", str(artifact_path))
    _write_runtime_health(run_dir / "runtime_health_summary.json", runtime_validity="valid_runtime")
    _write_drift_summary(run_dir / "drift_summary.json", divergence_time_s=8.0)
    _write_control_audit_summary(
        run_dir / "control_audit_summary.json",
        u2_first_raw_mismatch_time_s=6.0,
        u2_first_used_mismatch_time_s=6.0,
        raw_pre_sign=0.86,
        used_pre_sign=0.91,
        raw_late_sign=0.50,
        used_late_sign=0.62,
        raw_late_mag=0.66,
        used_late_mag=0.72,
        late_pos_norm=0.70,
        late_vel_norm=0.55,
    )

    summary = analyze_runtime_u2_root_cause(run_dir / "runtime_log.csv", artifact_path)

    assert summary["runtime_validity"] == "valid_runtime"
    assert summary["u2_root_cause_classification"] == "raw_u2_late_sign_instability"
    assert (run_dir / "u2_root_cause_summary.json").exists()
    assert (run_dir / "u2_root_cause_trace.csv").exists()


def test_analyze_runtime_u2_root_cause_classifies_inconclusive_good_run(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "4-3-26_good"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        x_positions=[0.0, 0.02, 0.04, 0.06, 0.08],
        z_positions=[0.75, 0.75, 0.76, 0.76, 0.77],
        raw_u2=[0.02, 0.02, 0.03, 0.03, 0.04],
        used_u2=[0.02, 0.02, 0.03, 0.03, 0.04],
    )
    _write_run_metadata(run_dir / "run_metadata.json", str(artifact_path))
    _write_runtime_health(run_dir / "runtime_health_summary.json", runtime_validity="valid_runtime")
    _write_drift_summary(run_dir / "drift_summary.json", divergence_time_s=10.0)
    _write_control_audit_summary(
        run_dir / "control_audit_summary.json",
        u2_first_raw_mismatch_time_s=8.5,
        u2_first_used_mismatch_time_s=8.5,
        raw_pre_sign=0.97,
        used_pre_sign=0.98,
        raw_late_sign=0.94,
        used_late_sign=0.97,
        raw_late_mag=0.90,
        used_late_mag=0.95,
        late_pos_norm=0.20,
        late_vel_norm=0.10,
    )

    summary = analyze_runtime_u2_root_cause(run_dir / "runtime_log.csv", artifact_path)

    assert summary["runtime_validity"] == "valid_runtime"
    assert summary["u2_root_cause_classification"] == "inconclusive"


def test_analyze_runtime_u2_root_cause_short_circuits_invalid_runtime(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "4-3-26_invalid"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        x_positions=[0.0, 0.5, 1.0, 1.5, 2.0],
        z_positions=[0.75, 0.90, 1.10, 1.30, 1.60],
        raw_u2=[0.02, 0.20, 0.20, 0.20, 0.20],
        used_u2=[0.02, 0.20, 0.20, 0.20, 0.20],
    )
    _write_run_metadata(run_dir / "run_metadata.json", str(artifact_path))
    _write_runtime_health(run_dir / "runtime_health_summary.json", runtime_validity="timing_collapse")
    _write_drift_summary(run_dir / "drift_summary.json", divergence_time_s=2.0)
    _write_control_audit_summary(
        run_dir / "control_audit_summary.json",
        u2_first_raw_mismatch_time_s=1.5,
        u2_first_used_mismatch_time_s=1.5,
        raw_pre_sign=0.40,
        used_pre_sign=0.40,
        raw_late_sign=0.20,
        used_late_sign=0.20,
        raw_late_mag=0.20,
        used_late_mag=0.20,
        late_pos_norm=2.0,
        late_vel_norm=1.0,
    )

    summary = analyze_runtime_u2_root_cause(run_dir / "runtime_log.csv", artifact_path)

    assert summary["runtime_validity"] == "timing_collapse"
    assert summary["u2_root_cause_classification"] == "runtime_invalid"
