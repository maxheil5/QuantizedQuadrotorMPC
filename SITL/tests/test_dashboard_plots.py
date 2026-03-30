from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.figure
import numpy as np


SITL_ROOT = Path(__file__).resolve().parents[1]
if str(SITL_ROOT) not in sys.path:
    sys.path.insert(0, str(SITL_ROOT))

from dashboard.data import prepare_run_data  # noqa: E402
from dashboard.plots import (  # noqa: E402
    plot_angular_velocity_errors,
    plot_angular_velocity_states,
    plot_attitude_errors,
    plot_attitude_states,
    plot_control_series,
    plot_position_errors,
    plot_position_states,
    plot_px4_diagnostics,
    plot_timing_series,
    plot_trajectory_3d,
    plot_velocity_errors,
    plot_velocity_states,
    plot_xy_trajectory,
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
    for step in range(6):
        state = np.zeros(18, dtype=float)
        reference = np.zeros(18, dtype=float)
        state[0] = 0.05 * step
        state[1] = 0.01 * step
        state[2] = 0.08 * step
        reference[0] = 0.04 * step
        reference[1] = 0.015 * step
        reference[2] = 0.07 * step
        state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
        reference[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
        control = np.array([44.0 + step, 0.1 * step, -0.05 * step, 0.02 * step], dtype=float)
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": 0.02 * step,
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
            row[f"reference_{idx}"] = reference[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = control[idx]
            row[f"control_used_{idx}"] = control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def test_dashboard_plot_builders_return_matplotlib_figures(tmp_path: Path):
    results_root = tmp_path / "results" / "sitl"
    run_dir = results_root / "plot_run"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv")

    data = prepare_run_data("plot_run", "raw", "used", results_root)

    figures = [
        plot_trajectory_3d(data),
        plot_xy_trajectory(data),
        plot_position_states(data),
        plot_velocity_states(data),
        plot_attitude_states(data),
        plot_angular_velocity_states(data),
        plot_position_errors(data),
        plot_velocity_errors(data),
        plot_attitude_errors(data),
        plot_angular_velocity_errors(data),
        plot_control_series(data),
        plot_timing_series(data),
    ]
    px4_figure = plot_px4_diagnostics(data)
    if px4_figure is not None:
        figures.append(px4_figure)

    assert figures
    assert all(isinstance(figure, matplotlib.figure.Figure) for figure in figures)
