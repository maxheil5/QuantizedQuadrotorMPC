from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.experiments.sitl_dataset import (
    build_sitl_edmd_snapshots,
    compute_sitl_run_diagnostics,
    excitation_warnings_from_diagnostics,
    load_sitl_run_dataset,
)


def _write_runtime_log(path: Path) -> None:
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
    for step in range(3):
        state = np.zeros(18, dtype=float)
        state[0] = 0.1 * step
        state[2] = 0.2 * step
        state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
        control = np.array([43.0 + step, 0.1 * step, -0.05 * step, 0.02 * step], dtype=float)
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": 0.01 * step,
            "reference_index": step,
            "tick_dt_ms": 10.0,
            "solver_ms": 0.4,
            "px4_collective_command_newton": control[0],
            "px4_collective_normalized": control[0] / 62.0,
            "px4_thrust_body_z": -(control[0] / 62.0),
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


def test_load_sitl_run_dataset_parses_state_and_control_history(tmp_path: Path):
    log_path = tmp_path / "runtime_log.csv"
    _write_runtime_log(log_path)
    _write_run_metadata(tmp_path / "run_metadata.json")

    dataset = load_sitl_run_dataset(log_path)

    assert dataset.sample_count == 3
    assert dataset.pair_count == 2
    assert dataset.state_history.shape == (18, 3)
    assert dataset.control_history.shape == (4, 3)
    assert np.allclose(dataset.tick_dt_ms, np.array([10.0, 10.0, 10.0]))
    assert dataset.run_metadata["reference_mode"] == "sitl_identification_v1"


def test_build_sitl_edmd_snapshots_stacks_without_crossing_run_boundaries(tmp_path: Path):
    log_a = tmp_path / "run_a.csv"
    log_b = tmp_path / "run_b.csv"
    _write_runtime_log(log_a)
    _write_runtime_log(log_b)

    run_a = load_sitl_run_dataset(log_a)
    run_b = load_sitl_run_dataset(log_b)
    x_history, u_history, x1_history, x2_history, u1_history = build_sitl_edmd_snapshots([run_a, run_b])

    assert x_history.shape == (18, 6)
    assert u_history.shape == (4, 6)
    assert x1_history.shape == (18, 4)
    assert x2_history.shape == (18, 4)
    assert u1_history.shape == (4, 4)


def test_compute_sitl_run_diagnostics_reports_control_and_state_ranges(tmp_path: Path):
    log_path = tmp_path / "runtime_log.csv"
    _write_runtime_log(log_path)
    _write_run_metadata(tmp_path / "run_metadata.json")

    dataset = load_sitl_run_dataset(log_path)
    diagnostics = compute_sitl_run_diagnostics(dataset)

    assert diagnostics["control_0_min"] == 43.0
    assert diagnostics["control_0_max"] == 45.0
    assert diagnostics["state_x_range"] == 0.2
    assert diagnostics["state_z_range"] == 0.4
    assert diagnostics["collective_below_1n_fraction"] == 0.0


def test_excitation_warning_triggers_for_low_excitation_dataset():
    diagnostics_rows = [
        {
            "run_name": "low_excitation",
            "control_0_std": 0.1,
            "control_1_std": 0.001,
            "control_2_std": 0.001,
            "control_3_std": 0.0005,
            "collective_below_1n_fraction": 0.8,
            "state_x_range": 0.01,
            "state_y_range": 0.02,
            "state_z_range": 0.05,
        }
    ]

    warnings = excitation_warnings_from_diagnostics(diagnostics_rows)

    assert warnings
    assert any("collective thrust std" in warning for warning in warnings)
    assert any("below 1 N" in warning for warning in warnings)
