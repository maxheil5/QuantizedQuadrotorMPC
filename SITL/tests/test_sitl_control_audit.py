from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.experiments.sitl_control_audit import (
    analyze_runtime_control_audit,
    replay_baseline_control_history,
    select_anchor_mapping_status,
    select_control_audit_mapping_status,
)
from quantized_quadrotor_sitl.experiments.sitl_dataset import load_sitl_run_dataset
from quantized_quadrotor_sitl.telemetry.adapter import physical_control_to_px4_wrench, vehicle_odometry_to_state18


def _identity_state18(x_pos: float = 0.0, y_pos: float = 0.0, z_pos: float = 0.0) -> list[float]:
    state = [0.0] * 18
    state[0] = x_pos
    state[1] = y_pos
    state[2] = z_pos
    state[6] = 1.0
    state[10] = 1.0
    state[14] = 1.0
    return state


def _write_runtime_log(
    path: Path,
    states: list[list[float]],
    references: list[list[float]],
    controls: list[list[float]],
    raw_controls: list[list[float]] | None = None,
    used_controls: list[list[float]] | None = None,
    control_internal: list[list[float]] | None = None,
) -> None:
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
    for step, (state, reference, control) in enumerate(zip(states, references, controls, strict=True)):
        raw_control = control if raw_controls is None else raw_controls[step]
        used_control = control if used_controls is None else used_controls[step]
        internal = control if control_internal is None else control_internal[step]
        row = {
            "step": step,
            "timestamp_ns": 1_000_000 * step,
            "experiment_time_s": 0.5 * step,
            "reference_index": step,
            "tick_dt_ms": 20.0,
            "solver_ms": 5.0,
            "px4_collective_command_newton": used_control[0],
            "px4_collective_normalized": used_control[0] / 62.0,
            "px4_thrust_body_z": -(used_control[0] / 62.0),
        }
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"state_used_{idx}"] = state[idx]
            row[f"reference_{idx}"] = reference[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = raw_control[idx]
            row[f"control_internal_{idx}"] = internal[idx]
            row[f"control_used_{idx}"] = used_control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path) -> None:
    payload = {
        "controller_mode": "edmd_mpc",
        "reference_mode": "takeoff_hold",
        "reference_seed": 2141444,
        "reference_duration_s": 10.0,
        "model_artifact": "results/offline/test/edmd_unquantized.npz",
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
        x_train_min=np.zeros((18, 1), dtype=float),
        x_train_max=np.ones((18, 1), dtype=float),
        u_train_min=np.array([[0.0], [-1.0], [-1.0], [-0.6]], dtype=float),
        u_train_max=np.array([[62.0], [1.0], [1.0], [0.6]], dtype=float),
        u_train_mean=np.array([[43.0], [0.0], [0.0], [0.0]], dtype=float),
        u_train_std=np.array([[5.0], [1.0], [1.0], [1.0]], dtype=float),
        u_trim=np.array([[43.0], [0.0], [0.0], [0.0]], dtype=float),
        affine_enabled=np.array([0.0], dtype=float),
    )
    (path.parent / "metrics_summary.csv").write_text(
        "\n".join(
            [
                "split,run_name,rmse_x,rmse_dx,rmse_theta,rmse_wb",
                "eval,held_out,1.0,1.0,1.0,1.0",
            ]
        ),
        encoding="utf-8",
    )


def test_replay_baseline_control_history_reproduces_expected_lateral_signs(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_signs"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0, 0.0, 0.75), _identity_state18(0.0, 0.0, 0.75)],
        references=[_identity_state18(1.0, 1.0, 0.75), _identity_state18(1.0, 1.0, 0.75)],
        controls=[[43.0, 0.0, 0.0, 0.0], [43.0, 0.0, 0.0, 0.0]],
    )

    run = load_sitl_run_dataset(log_path, state_source="used", control_source="used")
    baseline_history, _ = replay_baseline_control_history(run)

    assert baseline_history[1, 0] < 0.0
    assert baseline_history[2, 0] > 0.0


def test_control_audit_ignores_near_zero_baseline_axes(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_hover"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_run_metadata(run_dir / "run_metadata.json")
    hover = _identity_state18(0.0, 0.0, 0.75)
    _write_runtime_log(
        log_path,
        states=[hover, hover, hover],
        references=[hover, hover, hover],
        controls=[[43.0, 0.01, -0.01, 0.01]] * 3,
    )

    summary = analyze_runtime_control_audit(log_path, artifact_path)

    assert summary["active_sample_count_by_axis"] == {"u1": 0, "u2": 0, "u3": 0}
    assert summary["mapping_status"] == "model/runtime issue"
    assert summary["anchor_mapping_status"] == "learned-controller/runtime issue"
    assert (run_dir / "control_audit_summary.json").exists()
    assert (run_dir / "control_audit_trace.csv").exists()


def test_control_audit_classifies_forced_sign_flip_as_mapping_bug(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_flip"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_run_metadata(run_dir / "run_metadata.json")
    states = [_identity_state18(0.0, 0.0, 0.75)] * 5
    references = [_identity_state18(1.0, 1.0, 0.75)] * 5
    controls = [[43.0, 0.20, -0.20, 0.05]] * 5
    _write_runtime_log(log_path, states=states, references=references, controls=controls)

    summary = analyze_runtime_control_audit(log_path, artifact_path)

    assert summary["mapping_status"] == "mapping/sign bug"
    assert select_control_audit_mapping_status(summary) == "mapping/sign bug"
    assert summary["pre_divergence_sign_match_fraction_by_axis"]["u1"] == pytest.approx(0.0)
    assert summary["pre_divergence_sign_match_fraction_by_axis"]["u2"] == pytest.approx(0.0)


def test_control_audit_classifies_anchor_induced_sign_issue_from_raw_vs_used_split(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_anchor_flip"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_run_metadata(run_dir / "run_metadata.json")
    states = [_identity_state18(0.0, 0.0, 0.75)] * 5
    references = [_identity_state18(1.0, 1.0, 0.75)] * 5
    raw_controls = [[43.0, -0.20, 0.20, 0.05]] * 5
    used_controls = [[43.0, -0.20, -0.20, 0.05]] * 5
    _write_runtime_log(
        log_path,
        states=states,
        references=references,
        controls=used_controls,
        raw_controls=raw_controls,
        used_controls=used_controls,
    )

    summary = analyze_runtime_control_audit(log_path, artifact_path)

    assert summary["raw_pre_divergence_sign_match_fraction_by_axis"]["u2"] == pytest.approx(1.0)
    assert summary["used_pre_divergence_sign_match_fraction_by_axis"]["u2"] == pytest.approx(0.0)
    assert summary["anchor_mapping_status"] == "anchor-induced sign issue"
    assert select_anchor_mapping_status(summary) == "anchor-induced sign issue"


def test_control_audit_classifies_raw_sign_bug_as_learned_controller_issue(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_raw_bug"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_run_metadata(run_dir / "run_metadata.json")
    states = [_identity_state18(0.0, 0.0, 0.75)] * 5
    references = [_identity_state18(1.0, 1.0, 0.75)] * 5
    raw_controls = [[43.0, -0.20, -0.20, 0.05]] * 5
    used_controls = [[43.0, -0.20, -0.20, 0.05]] * 5
    _write_runtime_log(
        log_path,
        states=states,
        references=references,
        controls=used_controls,
        raw_controls=raw_controls,
        used_controls=used_controls,
    )

    summary = analyze_runtime_control_audit(log_path, artifact_path)

    assert summary["raw_pre_divergence_sign_match_fraction_by_axis"]["u2"] == pytest.approx(0.0)
    assert summary["used_pre_divergence_sign_match_fraction_by_axis"]["u2"] == pytest.approx(0.0)
    assert summary["anchor_mapping_status"] == "learned-controller/runtime issue"
    assert select_anchor_mapping_status(summary) == "learned-controller/runtime issue"


def test_vehicle_odometry_to_state18_preserves_expected_enu_flu_conventions():
    state = vehicle_odometry_to_state18(
        position_ned=np.array([1.0, 2.0, -3.0], dtype=float),
        velocity_ned=np.array([4.0, 5.0, -6.0], dtype=float),
        quaternion_wxyz_ned_frd=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_velocity_frd=np.array([0.1, 0.2, 0.3], dtype=float),
    )

    assert np.allclose(state[0:3], np.array([2.0, 1.0, 3.0], dtype=float))
    assert np.allclose(state[3:6], np.array([5.0, 4.0, 6.0], dtype=float))
    expected_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    assert np.allclose(state[6:15].reshape(3, 3, order="F"), expected_rotation)
    assert np.allclose(state[15:18], np.array([0.1, -0.2, -0.3], dtype=float))


def test_physical_control_to_px4_wrench_flips_body_yz_signs_for_px4():
    thrust_body, normalized_moments, collective_command_newton, collective_normalized = physical_control_to_px4_wrench(
        np.array([31.0, 0.2, -0.3, 0.12], dtype=float),
        max_collective_thrust_newton=62.0,
        max_body_torque_nm=np.array([1.0, 1.0, 0.6], dtype=float),
    )

    assert np.allclose(thrust_body, np.array([0.0, 0.0, -0.5], dtype=float))
    assert np.allclose(normalized_moments, np.array([0.2, 0.3, -0.2], dtype=float))
    assert collective_command_newton == pytest.approx(31.0)
    assert collective_normalized == pytest.approx(0.5)


def test_load_sitl_run_dataset_preserves_control_column_order_with_all_histories(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_order"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0, 0.0, 0.75)] * 3,
        references=[_identity_state18(0.0, 0.0, 0.75)] * 3,
        controls=[[43.0, 0.11, -0.22, 0.33], [44.0, 0.12, -0.23, 0.34], [45.0, 0.13, -0.24, 0.35]],
        raw_controls=[[42.5, 0.10, -0.20, 0.30], [43.5, 0.11, -0.21, 0.31], [44.5, 0.12, -0.22, 0.32]],
        used_controls=[[43.0, 0.11, -0.22, 0.33], [44.0, 0.12, -0.23, 0.34], [45.0, 0.13, -0.24, 0.35]],
        control_internal=[[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]],
    )

    dataset = load_sitl_run_dataset(log_path, state_source="used", control_source="used")

    assert np.allclose(dataset.control_history[:, 0], np.array([43.0, 0.11, -0.22, 0.33], dtype=float))
    assert np.allclose(dataset.control_raw_history[:, 0], np.array([42.5, 0.10, -0.20, 0.30], dtype=float))
    assert np.allclose(dataset.control_used_history[:, 0], np.array([43.0, 0.11, -0.22, 0.33], dtype=float))
    assert np.allclose(dataset.control_internal_history[:, 0], np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
