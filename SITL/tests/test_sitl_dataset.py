from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.experiments.sitl_dataset import build_sitl_edmd_snapshots, load_sitl_run_dataset


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


def test_load_sitl_run_dataset_parses_state_and_control_history(tmp_path: Path):
    log_path = tmp_path / "runtime_log.csv"
    _write_runtime_log(log_path)

    dataset = load_sitl_run_dataset(log_path)

    assert dataset.sample_count == 3
    assert dataset.pair_count == 2
    assert dataset.state_history.shape == (18, 3)
    assert dataset.control_history.shape == (4, 3)
    assert np.allclose(dataset.tick_dt_ms, np.array([10.0, 10.0, 10.0]))


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
