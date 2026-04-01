from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.core.artifacts import load_edmd_artifact
from quantized_quadrotor_sitl.experiments.offline_sitl_retrain import run_sitl_retrain


def _write_runtime_log(path: Path, collective_values: list[float], x_values: list[float], z_values: list[float]) -> None:
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
        *[f"control_used_{idx}" for idx in range(4)],
        *[f"reference_{idx}" for idx in range(18)],
    ]
    rows = []
    for step, (collective, x_pos, z_pos) in enumerate(zip(collective_values, x_values, z_values, strict=True)):
        state = [0.0] * 18
        state[0] = x_pos
        state[2] = z_pos
        state[6] = 1.0
        state[10] = 1.0
        state[14] = 1.0
        control = [collective, 0.0, 0.0, 0.0]
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": 0.01 * step,
            "reference_index": step,
            "tick_dt_ms": 10.0,
            "solver_ms": 0.4,
            "px4_collective_command_newton": collective,
            "px4_collective_normalized": collective / 62.0,
            "px4_thrust_body_z": -(collective / 62.0),
        }
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"state_used_{idx}"] = state[idx]
            row[f"reference_{idx}"] = state[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = control[idx]
            row[f"control_used_{idx}"] = control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path) -> None:
    payload = {
        "controller_mode": "baseline_geometric",
        "reference_mode": "sitl_identification_v1",
        "reference_seed": 2141444,
        "reference_duration_s": 24.0,
        "model_artifact": "results/offline/sitl_baseline_v1/latest/edmd_unquantized.npz",
        "quantization_mode": "none",
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
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_run_sitl_retrain_writes_excitation_diagnostics_and_warning(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "low_excitation"
    run_dir.mkdir(parents=True)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        collective_values=[0.2, 0.3, 0.2, 0.4],
        x_values=[0.0, 0.01, 0.0, 0.01],
        z_values=[0.0, 0.02, 0.03, 0.01],
    )
    _write_run_metadata(run_dir / "run_metadata.json")

    output = run_sitl_retrain(
        train_runs=[run_dir / "runtime_log.csv"],
        eval_runs=[],
        results_root=tmp_path / "results" / "offline",
        state_source="raw",
        control_source="used",
        n_basis=1,
        tag="test",
    )

    summary = json.loads(output.summary_json.read_text())
    metrics_lines = output.metrics_csv.read_text().splitlines()
    model, metadata = load_edmd_artifact(output.artifact_paths[0])

    assert summary["warnings"]
    assert any("below 1 N" in warning for warning in summary["warnings"])
    assert "control_0_std" in metrics_lines[0]
    assert "state_z_range" in metrics_lines[0]
    assert model.B.shape[1] == 4
    assert "u_train_mean" in metadata
    assert "u_train_std" in metadata
    assert "u_trim" in metadata
    np.testing.assert_allclose(metadata["u_trim"], metadata["u_train_mean"])


def test_run_sitl_retrain_can_emit_affine_artifact_metadata(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "affine_excitation"
    run_dir.mkdir(parents=True)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        collective_values=[44.0, 48.0, 52.0, 50.0],
        x_values=[0.0, 0.2, 0.5, 0.9],
        z_values=[0.0, 0.1, 0.2, 0.3],
    )
    _write_run_metadata(run_dir / "run_metadata.json")

    output = run_sitl_retrain(
        train_runs=[run_dir / "runtime_log.csv"],
        eval_runs=[run_dir / "runtime_log.csv"],
        results_root=tmp_path / "results" / "offline",
        state_source="raw",
        control_source="used",
        n_basis=0,
        tag="affine",
        affine=True,
    )

    summary = json.loads(output.summary_json.read_text())
    model, metadata = load_edmd_artifact(output.artifact_paths[0])

    assert summary["affine_enabled"] is True
    assert len(summary["bias"]) == 24
    assert model.affine_enabled is True
    assert model.bias is not None
    assert "u_trim" in metadata


def test_run_sitl_retrain_can_emit_hover_residual_artifact_metadata(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "hover_residual_excitation"
    run_dir.mkdir(parents=True)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        collective_values=[44.0, 48.0, 52.0, 50.0],
        x_values=[0.0, 0.1, 0.2, 0.3],
        z_values=[0.0, 0.15, 0.30, 0.45],
    )
    _write_run_metadata(run_dir / "run_metadata.json")

    output = run_sitl_retrain(
        train_runs=[run_dir / "runtime_log.csv"],
        eval_runs=[run_dir / "runtime_log.csv"],
        results_root=tmp_path / "results" / "offline",
        state_source="raw",
        control_source="used",
        n_basis=0,
        tag="hover_residual",
        affine=True,
        hover_residual=True,
    )

    summary = json.loads(output.summary_json.read_text())
    _, metadata = load_edmd_artifact(output.artifact_paths[0])

    assert summary["hover_residual"] is True
    assert summary["residual_enabled"] is True
    assert summary["state_coordinates"] == "takeoff_hold_hover_local"
    assert summary["state_trim_mode"] == "per_run_takeoff_hold_final"
    assert len(summary["state_trim"]) == 18
    assert metadata["residual_enabled"] is True
    assert metadata["state_coordinates"] == "takeoff_hold_hover_local"
    assert metadata["state_trim_mode"] == "per_run_takeoff_hold_final"
    assert np.asarray(metadata["state_trim"], dtype=float).shape == (18, 1)


def test_load_edmd_artifact_supports_affine_bias_prediction(tmp_path: Path):
    artifact_path = tmp_path / "affine_edmd_unquantized.npz"
    bias = np.array([0.25, -0.5, 0.0, 1.0], dtype=float)
    np.savez(
        artifact_path,
        A=np.eye(4, dtype=float),
        B=np.zeros((4, 4), dtype=float),
        C=np.eye(4, dtype=float),
        Z1=np.zeros((4, 1), dtype=float),
        Z2=np.zeros((4, 1), dtype=float),
        n_basis=np.array([0]),
        bias=bias,
        affine_enabled=np.array([1.0], dtype=float),
        x_train_min=np.zeros((18, 1), dtype=float),
        x_train_max=np.ones((18, 1), dtype=float),
        u_train_min=np.zeros((4, 1), dtype=float),
        u_train_max=np.ones((4, 1), dtype=float),
    )

    model, _ = load_edmd_artifact(artifact_path)
    predicted = model.predict_next_lifted(np.zeros(4, dtype=float), np.zeros(4, dtype=float))

    assert model.affine_enabled is True
    np.testing.assert_allclose(predicted, bias)


def test_load_edmd_artifact_remains_backward_compatible_with_legacy_metadata(tmp_path: Path):
    artifact_path = tmp_path / "legacy_edmd_unquantized.npz"
    np.savez(
        artifact_path,
        A=np.eye(4, dtype=float),
        B=np.ones((4, 4), dtype=float),
        C=np.eye(4, dtype=float),
        Z1=np.zeros((4, 1), dtype=float),
        Z2=np.zeros((4, 1), dtype=float),
        n_basis=np.array([3]),
        x_train_min=np.zeros((18, 1), dtype=float),
        x_train_max=np.ones((18, 1), dtype=float),
        u_train_min=np.zeros((4, 1), dtype=float),
        u_train_max=np.ones((4, 1), dtype=float),
    )

    model, metadata = load_edmd_artifact(artifact_path)

    assert model.n_basis == 3
    assert model.affine_enabled is False
    assert model.bias is None
    assert "u_train_min" in metadata
    assert "u_train_max" in metadata
    assert "u_trim" not in metadata


def test_load_edmd_artifact_supports_residual_metadata(tmp_path: Path):
    artifact_path = tmp_path / "residual_edmd_unquantized.npz"
    np.savez(
        artifact_path,
        A=np.eye(4, dtype=float),
        B=np.eye(4, dtype=float),
        C=np.eye(4, dtype=float),
        Z1=np.zeros((4, 1), dtype=float),
        Z2=np.zeros((4, 1), dtype=float),
        n_basis=np.array([0]),
        residual_enabled=np.array([1.0], dtype=float),
        state_coordinates=np.array(["takeoff_hold_hover_local"]),
        state_trim_mode=np.array(["per_run_takeoff_hold_final"]),
        state_trim=np.zeros((18, 1), dtype=float),
        x_train_min=np.zeros((18, 1), dtype=float),
        x_train_max=np.ones((18, 1), dtype=float),
        u_train_min=np.zeros((4, 1), dtype=float),
        u_train_max=np.ones((4, 1), dtype=float),
    )

    _, metadata = load_edmd_artifact(artifact_path)

    assert metadata["residual_enabled"] is True
    assert metadata["state_coordinates"] == "takeoff_hold_hover_local"
    assert metadata["state_trim_mode"] == "per_run_takeoff_hold_final"
    assert np.asarray(metadata["state_trim"], dtype=float).shape == (18, 1)
