from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.experiments.sitl_postrun_analysis import run_postrun_edmd_analyses


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


def _write_run_metadata(path: Path, artifact_path: str, cost_state_mode: str = "decoded24_raw") -> None:
    payload = {
        "controller_mode": "edmd_mpc",
        "reference_mode": "takeoff_hold",
        "reference_seed": 2141444,
        "reference_duration_s": 10.0,
        "cost_state_mode": cost_state_mode,
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


def test_run_postrun_edmd_analyses_writes_drift_and_control_sidecars(tmp_path: Path):
    base_dir = tmp_path
    run_dir = base_dir / "results" / "sitl" / "4-1-26_postrun"
    run_dir.mkdir(parents=True)
    artifact_path = base_dir / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(run_dir / "runtime_log.csv")
    _write_run_metadata(
        run_dir / "run_metadata.json",
        "results/offline/test/edmd_unquantized.npz",
        cost_state_mode="minimal_residual",
    )
    config_path = base_dir / "postrun.yaml"
    config_path.write_text(
        "\n".join(
            [
                "controller_mode: edmd_mpc",
                "model_artifact: results/offline/test/edmd_unquantized.npz",
                "results_dir: results/sitl/4-1-26_postrun",
                "mpc:",
                "  cost_state_mode: minimal_residual",
            ]
        ),
        encoding="utf-8",
    )

    summary = run_postrun_edmd_analyses(config_path=config_path, base_dir=base_dir)

    assert summary["skipped"] is False
    assert Path(summary["run_dir"]) == run_dir
    assert summary["cost_state_mode"] == "minimal_residual"
    assert (run_dir / "drift_summary.json").exists()
    assert (run_dir / "drift_trace.csv").exists()
    assert (run_dir / "control_audit_summary.json").exists()
    assert (run_dir / "control_audit_trace.csv").exists()


def test_run_postrun_edmd_analyses_writes_sidecars_from_run_dir(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "4-1-26_direct"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    _write_runtime_log(run_dir / "runtime_log.csv")
    _write_run_metadata(
        run_dir / "run_metadata.json",
        str(artifact_path),
        cost_state_mode="decoded24_raw",
    )

    summary = run_postrun_edmd_analyses(
        run_dir=run_dir,
        artifact_path=artifact_path,
    )

    assert summary["skipped"] is False
    assert Path(summary["run_dir"]) == run_dir
    assert (run_dir / "drift_summary.json").exists()
    assert (run_dir / "control_audit_summary.json").exists()


def test_run_postrun_edmd_analyses_writes_sidecars_from_log_path(tmp_path: Path):
    run_dir = tmp_path / "results" / "sitl" / "4-1-26_log_path"
    run_dir.mkdir(parents=True)
    artifact_path = tmp_path / "results" / "offline" / "test" / "edmd_unquantized.npz"
    artifact_path.parent.mkdir(parents=True)
    _write_artifact(artifact_path)
    log_path = run_dir / "runtime_log.csv"
    _write_runtime_log(log_path)
    _write_run_metadata(
        run_dir / "run_metadata.json",
        str(artifact_path),
        cost_state_mode="decoded24_raw",
    )

    summary = run_postrun_edmd_analyses(
        log_path=log_path,
        artifact_path=artifact_path,
    )

    assert summary["skipped"] is False
    assert Path(summary["run_dir"]) == run_dir
    assert (run_dir / "drift_summary.json").exists()
    assert (run_dir / "control_audit_summary.json").exists()


def test_run_sitl_experiment_script_invokes_postrun_with_run_dir():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sitl_experiment.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "--run-dir \"${run_dir}\"" in script_text
    assert "--artifact-path \"${artifact_path}\"" in script_text
    assert "--metadata-path \"${metadata_path}\"" in script_text
