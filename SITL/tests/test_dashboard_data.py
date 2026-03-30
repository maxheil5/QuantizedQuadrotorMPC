from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


SITL_ROOT = Path(__file__).resolve().parents[1]
if str(SITL_ROOT) not in sys.path:
    sys.path.insert(0, str(SITL_ROOT))

from dashboard.data import (  # noqa: E402
    available_sources,
    default_run_name,
    discover_run_names,
    load_run_metadata,
    prepare_run_data,
)


def _write_runtime_log(path: Path, include_used: bool = True, include_px4: bool = True) -> None:
    header = [
        "step",
        "timestamp_ns",
        "experiment_time_s",
        "reference_index",
        "tick_dt_ms",
        "solver_ms",
    ]
    if include_px4:
        header += [
            "px4_collective_command_newton",
            "px4_collective_normalized",
            "px4_thrust_body_z",
        ]
    header += [f"state_raw_{idx}" for idx in range(18)]
    if include_used:
        header += [f"state_used_{idx}" for idx in range(18)]
    header += [f"control_raw_{idx}" for idx in range(4)]
    if include_used:
        header += [f"control_used_{idx}" for idx in range(4)]
    header += [f"reference_{idx}" for idx in range(18)]

    rows = []
    for step in range(5):
        state = np.zeros(18, dtype=float)
        reference = np.zeros(18, dtype=float)
        state[0] = 0.05 * step
        state[1] = -0.02 * step
        state[2] = 0.1 * step
        reference[0] = 0.04 * step
        reference[1] = -0.01 * step
        reference[2] = 0.09 * step
        state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
        reference[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
        control = np.array([43.0 + step, 0.1 * step, -0.05 * step, 0.02 * step], dtype=float)
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": 0.01 * step,
            "reference_index": step,
            "tick_dt_ms": 10.0,
            "solver_ms": 0.4,
        }
        if include_px4:
            row["px4_collective_command_newton"] = control[0]
            row["px4_collective_normalized"] = control[0] / 62.0
            row["px4_thrust_body_z"] = -(control[0] / 62.0)
        for idx in range(18):
            row[f"state_raw_{idx}"] = state[idx]
            row[f"reference_{idx}"] = reference[idx]
            if include_used:
                row[f"state_used_{idx}"] = state[idx]
        for idx in range(4):
            row[f"control_raw_{idx}"] = control[idx]
            if include_used:
                row[f"control_used_{idx}"] = control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path) -> None:
    payload = {
        "controller_mode": "baseline_geometric",
        "reference_mode": "takeoff_hold",
        "quantization_mode": "none",
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_discover_run_names_and_default_selection(tmp_path: Path):
    results_root = tmp_path / "results" / "sitl"
    (results_root / "3-21-26_1735_base_controller").mkdir(parents=True)
    (results_root / "3-22-26_1920").mkdir(parents=True)
    _write_runtime_log(results_root / "3-21-26_1735_base_controller" / "runtime_log.csv")
    _write_runtime_log(results_root / "3-22-26_1920" / "runtime_log.csv")

    run_names = discover_run_names(results_root)

    assert run_names == ["3-22-26_1920", "3-21-26_1735_base_controller"]
    assert default_run_name(run_names) == "3-21-26_1735_base_controller"


def test_prepare_run_data_loads_metadata_and_derives_metrics(tmp_path: Path):
    results_root = tmp_path / "results" / "sitl"
    run_dir = results_root / "3-21-26_1735_base_controller"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv")
    _write_run_metadata(run_dir / "run_metadata.json")

    data = prepare_run_data("3-21-26_1735_base_controller", "raw", "raw", results_root)

    assert data.metadata is not None
    assert data.metadata["controller_mode"] == "baseline_geometric"
    assert data.time_s.shape == (5,)
    assert data.state_history.shape == (18, 5)
    assert data.reference_history.shape == (18, 5)
    assert data.control_history.shape == (4, 5)
    assert data.summary["sample_count"] == 5
    assert data.summary["position_rmse"] > 0.0
    assert data.px4_available is True


def test_prepare_run_data_handles_missing_metadata_and_missing_px4_columns(tmp_path: Path):
    results_root = tmp_path / "results" / "sitl"
    run_dir = results_root / "3-20-26_1600"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", include_used=False, include_px4=False)

    data = prepare_run_data("3-20-26_1600", "raw", "raw", results_root)

    assert load_run_metadata("3-20-26_1600", results_root) is None
    assert available_sources(data.frame, "state", 18) == ["raw"]
    assert available_sources(data.frame, "control", 4) == ["raw"]
    assert data.metadata is None
    assert data.px4_available is False
