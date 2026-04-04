from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.experiments.sitl_milestone_summary import (
    milestone_summary_contains_run,
    summarize_milestone_run,
    update_milestone_summary_csv,
)
from quantized_quadrotor_sitl.experiments.sitl_runtime_health import analyze_runtime_health


def _identity_state18(x_pos: float = 0.0, z_pos: float = 0.0) -> list[float]:
    state = [0.0] * 18
    state[0] = x_pos
    state[2] = z_pos
    state[6] = 1.0
    state[10] = 1.0
    state[14] = 1.0
    return state


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
        *[f"control_internal_{idx}" for idx in range(4)],
        *[f"control_used_{idx}" for idx in range(4)],
        *[f"reference_{idx}" for idx in range(18)],
    ]
    states = [_identity_state18(0.0, 0.0), _identity_state18(0.2, 0.0), _identity_state18(0.4, 0.0)]
    references = [_identity_state18(0.0, 0.0), _identity_state18(0.2, 0.0), _identity_state18(0.4, 0.0)]
    controls = [[43.0, 0.1, 0.0, 0.0], [43.0, 0.1, 0.0, 0.0], [43.0, 0.1, 0.0, 0.0]]
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
            row[f"control_internal_{idx}"] = control[idx]
            row[f"control_used_{idx}"] = control[idx]
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(path: Path, artifact_path: str) -> None:
    payload = {
        "config_path": "/tmp/postrun.yaml",
        "control_rate_hz": 50.0,
        "controller_mode": "edmd_mpc",
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


def test_update_milestone_summary_csv_writes_and_updates_run_rows(tmp_path: Path):
    results_root = tmp_path / "results" / "sitl"
    run_dir = results_root / "4-3-26_1700_01"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(run_dir / "runtime_log.csv")
    _write_run_metadata(run_dir / "run_metadata.json", str(artifact_path))
    analyze_runtime_health(log_path=run_dir / "runtime_log.csv", output_dir=run_dir, metadata_path=run_dir / "run_metadata.json")
    (run_dir / "drift_summary.json").write_text(
        '{"divergence_time_s": 1.0, "selected_branch": "Branch A", "dominant_error_group": "x", "post_4s_internal_bound_fraction": {"u0": 0.0, "u1": 0.0, "u2": 0.0, "u3": 0.0}}',
        encoding="utf-8",
    )
    (run_dir / "control_audit_summary.json").write_text(
        '{"mapping_status": "model/runtime issue", "dominant_mismatch_axis": "u2", "u2_first_used_mismatch_time_s": 8.5}',
        encoding="utf-8",
    )
    (run_dir / "u2_root_cause_summary.json").write_text(
        json.dumps(
            {
                "u2_first_raw_mismatch_time_s": 8.2,
                "u2_late_window_raw_sign_match": 0.95,
                "u2_late_window_used_sign_match": 0.97,
                "u2_root_cause_classification": "inconclusive",
            }
        ),
        encoding="utf-8",
    )

    row = summarize_milestone_run(run_dir)
    assert row["run_name"] == "4-3-26_1700_01"
    assert row["hover_gate_profile"] == "light_anchor_confirmation"
    assert row["u2_root_cause_classification"] == "inconclusive"

    summary_path = update_milestone_summary_csv(run_dir)
    assert summary_path == results_root / "milestone_summary.csv"
    assert (run_dir / "milestone_summary.csv").exists()
    assert milestone_summary_contains_run(summary_path, "4-3-26_1700_01") is True

    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["run_name"] == "4-3-26_1700_01"
    assert rows[0]["u2_root_cause_classification"] == "inconclusive"
    with (run_dir / "milestone_summary.csv").open("r", encoding="utf-8", newline="") as stream:
        snapshot_rows = list(csv.DictReader(stream))
    assert snapshot_rows == rows

    update_milestone_summary_csv(run_dir)
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
