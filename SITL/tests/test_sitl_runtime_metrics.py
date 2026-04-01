from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quantized_quadrotor_sitl.experiments.sitl_runtime_metrics import compute_hover_gate_metrics, evaluate_hover_gates


def _write_runtime_log(path: Path, z_values: list[float], x_values: list[float], u2_values: list[float], solver_ms: float, tick_dt_ms: float) -> None:
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
    for step, (z_pos, x_pos, u2) in enumerate(zip(z_values, x_values, u2_values, strict=True)):
        state = [0.0] * 18
        reference = [0.0] * 18
        control = [50.0, 0.0, u2, 0.0]
        state[0] = x_pos
        state[2] = z_pos
        reference[2] = 0.75
        state[6] = 1.0
        state[10] = 1.0
        state[14] = 1.0
        reference[6] = 1.0
        reference[10] = 1.0
        reference[14] = 1.0
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": 0.1 * step,
            "reference_index": step,
            "tick_dt_ms": tick_dt_ms,
            "solver_ms": solver_ms,
            "px4_collective_command_newton": 50.0,
            "px4_collective_normalized": 50.0 / 62.0,
            "px4_thrust_body_z": -(50.0 / 62.0),
        }
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"state_used_{idx}"] = state[idx]
            row[f"reference_{idx}"] = reference[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = control[idx]
            row[f"control_internal_{idx}"] = control[idx]
            row[f"control_used_{idx}"] = control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path) -> None:
    payload = {
        "vehicle_scaling": {
            "max_collective_thrust_newton": 62.0,
            "max_body_torque_x_nm": 1.0,
            "max_body_torque_y_nm": 1.0,
            "max_body_torque_z_nm": 0.6,
        }
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_compute_hover_gate_metrics_extracts_repeatable_validation_metrics(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1530"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", [0.0, 0.45, 0.74], [0.0, 0.05, 0.09], [0.0, 0.05, 0.10], 9.5, 10.5)
    _write_run_metadata(run_dir / "run_metadata.json")

    metrics = compute_hover_gate_metrics(run_dir / "runtime_log.csv")

    assert metrics["z_max"] == 0.74
    assert metrics["final_altitude_error"] == pytest.approx(0.01)
    assert metrics["solver_mean_ms"] == 9.5
    assert metrics["tick_mean_ms"] == 10.5
    assert metrics["max_lateral_radius"] > 0.0


def test_evaluate_hover_gates_reports_standard_profile_pass(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1540"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", [0.0, 0.62, 0.74], [0.0, 0.03, 0.04], [0.0, 0.05, 0.08], 7.5, 11.0)
    _write_run_metadata(run_dir / "run_metadata.json")

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="standard")

    assert evaluation["passed"] is True
    assert all(bool(value) for value in evaluation["checks"].values())
