from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.experiments.sitl_drift_analysis import (
    analyze_runtime_drift,
    select_drift_branch_from_summaries,
)


def _identity_state18(x_pos: float = 0.0, z_pos: float = 0.0) -> list[float]:
    state = [0.0] * 18
    state[0] = x_pos
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
        *[f"control_used_{idx}" for idx in range(4)],
        *[f"reference_{idx}" for idx in range(18)],
    ]
    if control_internal is not None:
        header = [
            *header[:-22],
            *[f"control_internal_{idx}" for idx in range(4)],
            *header[-22:],
        ]
    rows = []
    for step, (state, reference, control) in enumerate(zip(states, references, controls, strict=True)):
        row = {
            "step": step,
            "timestamp_ns": 1_000_000 * step,
            "experiment_time_s": 0.5 * step,
            "reference_index": step,
            "tick_dt_ms": 20.0,
            "solver_ms": 5.0,
            "px4_collective_command_newton": control[0],
            "px4_collective_normalized": control[0] / 62.0,
            "px4_thrust_body_z": -(control[0] / 62.0),
        }
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"state_used_{idx}"] = state[idx]
            row[f"reference_{idx}"] = reference[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = control[idx]
            row[f"control_used_{idx}"] = control[idx]
            if control_internal is not None:
                row[f"control_internal_{idx}"] = control_internal[step][idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path, cost_state_mode: str = "decoded24_raw") -> None:
    path.write_text(
        json.dumps(
            {
                "cost_state_mode": cost_state_mode,
                "vehicle_scaling": {
                    "max_collective_thrust_newton": 62.0,
                    "max_body_torque_x_nm": 1.0,
                    "max_body_torque_y_nm": 1.0,
                    "max_body_torque_z_nm": 0.6,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_artifact(path: Path, model: EDMDModel, rmse_values: tuple[float, float, float, float]) -> None:
    payload: dict[str, np.ndarray] = {
        "A": model.A,
        "B": model.B,
        "C": model.C,
        "Z1": model.Z1,
        "Z2": model.Z2,
        "n_basis": np.array([model.n_basis]),
        "x_train_min": np.zeros((18, 1), dtype=float),
        "x_train_max": np.ones((18, 1), dtype=float),
        "u_train_min": np.array([[0.0], [-1.0], [-1.0], [-1.0]], dtype=float),
        "u_train_max": np.array([[4.0], [1.0], [1.0], [1.0]], dtype=float),
        "u_train_mean": np.array([[2.0], [0.0], [0.0], [0.0]], dtype=float),
        "u_train_std": np.array([[0.5], [1.0], [1.0], [1.0]], dtype=float),
        "u_trim": np.array([[2.0], [0.0], [0.0], [0.0]], dtype=float),
        "affine_enabled": np.array([1.0 if model.affine_enabled else 0.0], dtype=float),
    }
    if model.affine_enabled and model.bias is not None:
        payload["bias"] = np.asarray(model.bias, dtype=float)
    np.savez(path, **payload)
    metrics_path = path.parent / "metrics_summary.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "split,run_name,rmse_x,rmse_dx,rmse_theta,rmse_wb",
                f"eval,held_out,{rmse_values[0]},{rmse_values[1]},{rmse_values[2]},{rmse_values[3]}",
            ]
        ),
        encoding="utf-8",
    )


def _write_residual_artifact(path: Path, model: EDMDModel, rmse_values: tuple[float, float, float, float]) -> None:
    payload: dict[str, np.ndarray] = {
        "A": model.A,
        "B": model.B,
        "C": model.C,
        "Z1": model.Z1,
        "Z2": model.Z2,
        "n_basis": np.array([model.n_basis]),
        "x_train_min": np.zeros((18, 1), dtype=float),
        "x_train_max": np.ones((18, 1), dtype=float),
        "u_train_min": np.array([[0.0], [-1.0], [-1.0], [-1.0]], dtype=float),
        "u_train_max": np.array([[4.0], [1.0], [1.0], [1.0]], dtype=float),
        "u_train_mean": np.array([[2.0], [0.0], [0.0], [0.0]], dtype=float),
        "u_train_std": np.array([[0.5], [1.0], [1.0], [1.0]], dtype=float),
        "u_trim": np.array([[2.0], [0.0], [0.0], [0.0]], dtype=float),
        "affine_enabled": np.array([1.0 if model.affine_enabled else 0.0], dtype=float),
        "residual_enabled": np.array([1.0], dtype=float),
        "state_coordinates": np.array(["takeoff_hold_hover_local"]),
        "state_trim_mode": np.array(["per_run_takeoff_hold_final"]),
        "state_trim": np.zeros((18, 1), dtype=float),
    }
    if model.affine_enabled and model.bias is not None:
        payload["bias"] = np.asarray(model.bias, dtype=float)
    np.savez(path, **payload)
    metrics_path = path.parent / "metrics_summary.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "split,run_name,rmse_x,rmse_dx,rmse_theta,rmse_wb",
                f"eval,held_out,{rmse_values[0]},{rmse_values[1]},{rmse_values[2]},{rmse_values[3]}",
            ]
        ),
        encoding="utf-8",
    )


def _identity_model() -> EDMDModel:
    return EDMDModel(
        A=np.eye(24, dtype=float),
        B=np.zeros((24, 4), dtype=float),
        C=np.eye(24, dtype=float),
        Z1=np.zeros((24, 1), dtype=float),
        Z2=np.zeros((24, 1), dtype=float),
        n_basis=0,
    )


def test_analyze_runtime_drift_uses_trim_scaled_controls_for_one_step_prediction(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1640_1"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)

    model = _identity_model()
    model.A = np.zeros((24, 24), dtype=float)
    model.B[0, 0] = 1.0
    _write_artifact(artifact_path, model, rmse_values=(0.1, 0.1, 0.1, 0.1))
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0), _identity_state18(2.0)],
        references=[_identity_state18(0.0), _identity_state18(2.0)],
        controls=[[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
    )

    summary = analyze_runtime_drift(log_path, artifact_path)

    assert summary["early_window_rmse_ratio"]["x"] == pytest.approx(0.0)
    assert (run_dir / "drift_trace.csv").exists()
    assert (run_dir / "drift_summary.json").exists()


def test_analyze_runtime_drift_selects_branch_a_for_large_early_model_error(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1530"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)

    _write_artifact(artifact_path, _identity_model(), rmse_values=(0.01, 0.1, 0.1, 0.1))
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[
            _identity_state18(0.0),
            _identity_state18(1.0),
            _identity_state18(2.0),
            _identity_state18(3.0),
            _identity_state18(4.0),
        ],
        references=[
            _identity_state18(0.0),
            _identity_state18(1.0),
            _identity_state18(2.0),
            _identity_state18(3.0),
            _identity_state18(4.0),
        ],
        controls=[[2.0, 0.0, 0.0, 0.0]] * 5,
    )

    summary = analyze_runtime_drift(log_path, artifact_path)

    assert summary["selected_branch"] == "Branch A"
    assert summary["dominant_error_group"] == "x"
    assert summary["early_window_rmse_ratio"]["x"] > 4.0
    assert select_drift_branch_from_summaries([summary]) == "Branch A"


def test_analyze_runtime_drift_selects_branch_b_for_early_divergence_with_bound_activity(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1540"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)

    _write_artifact(artifact_path, _identity_model(), rmse_values=(1.0, 1.0, 1.0, 1.0))
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0)] * 4,
        references=[
            _identity_state18(0.0),
            _identity_state18(2.0),
            _identity_state18(2.0),
            _identity_state18(2.0),
        ],
        controls=[[4.0, 1.0, 1.0, 1.0]] * 4,
        control_internal=[[1.0, 1.0, 1.0, 1.0]] * 4,
    )

    summary = analyze_runtime_drift(log_path, artifact_path)

    assert summary["selected_branch"] == "Branch B"
    assert summary["divergence_time_s"] < 4.0
    assert summary["pre_divergence_internal_bound_fraction"]["u1"] > 0.1


def test_analyze_runtime_drift_reports_post_four_second_bound_activity_for_residual_artifacts(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1900"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)

    _write_residual_artifact(artifact_path, _identity_model(), rmse_values=(1.0, 1.0, 1.0, 1.0))
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0, 0.1 * idx) for idx in range(10)],
        references=[_identity_state18(0.0, 0.1 * idx) for idx in range(10)],
        controls=[[4.0, 1.0, 0.0, 0.0]] * 10,
        control_internal=[[0.0, 0.0, 0.0, 0.0]] * 8 + [[3.6, 0.9, 0.0, 0.0], [3.6, 0.9, 0.0, 0.0]],
    )

    summary = analyze_runtime_drift(log_path, artifact_path)

    assert summary["residual_enabled"] is True
    assert summary["post_4s_internal_bound_fraction"]["u0"] > 0.0
    assert summary["post_4s_internal_bound_fraction"]["u1"] > 0.0


def test_analyze_runtime_drift_reports_minimal_residual_cost_state_mode(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_minimal"
    run_dir.mkdir()
    log_path = run_dir / "runtime_log.csv"
    artifact_path = tmp_path / "results" / "offline" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)

    _write_artifact(artifact_path, _identity_model(), rmse_values=(1.0, 1.0, 1.0, 1.0))
    _write_run_metadata(run_dir / "run_metadata.json", cost_state_mode="minimal_residual")
    _write_runtime_log(
        log_path,
        states=[_identity_state18(0.0), _identity_state18(0.1)],
        references=[_identity_state18(0.0), _identity_state18(0.1)],
        controls=[[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
    )

    summary = analyze_runtime_drift(log_path, artifact_path)

    assert summary["cost_state_mode"] == "minimal_residual"
