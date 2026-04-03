from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quantized_quadrotor_sitl.experiments.sitl_runtime_health import compute_runtime_health_summary
from quantized_quadrotor_sitl.experiments.sitl_runtime_metrics import compute_hover_gate_metrics, evaluate_hover_gates


def _write_runtime_log(
    path: Path,
    z_values: list[float],
    x_values: list[float],
    u2_values: list[float],
    solver_ms: float | list[float],
    tick_dt_ms: float | list[float],
    *,
    experiment_dt_s: float = 0.1,
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
    solver_series = solver_ms if isinstance(solver_ms, list) else [solver_ms] * len(z_values)
    tick_series = tick_dt_ms if isinstance(tick_dt_ms, list) else [tick_dt_ms] * len(z_values)
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
            "experiment_time_s": experiment_dt_s * step,
            "reference_index": step,
            "tick_dt_ms": tick_series[step],
            "solver_ms": solver_series[step],
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


def _write_run_metadata(
    path: Path,
    *,
    control_rate_hz: float = 50.0,
    pred_horizon: int = 8,
    reference_duration_s: float = 10.0,
) -> None:
    payload = {
        "config_path": "/tmp/runtime.yaml",
        "control_rate_hz": control_rate_hz,
        "pred_horizon": pred_horizon,
        "reference_duration_s": reference_duration_s,
        "reference_seed": 2141444,
        "vehicle_scaling": {
            "max_collective_thrust_newton": 62.0,
            "max_body_torque_x_nm": 1.0,
            "max_body_torque_y_nm": 1.0,
            "max_body_torque_z_nm": 0.6,
        }
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _write_drift_summary(path: Path, *, divergence_time_s: float, post_four: dict[str, float] | None = None) -> None:
    payload = {
        "selected_branch": "Branch A",
        "dominant_error_group": "x",
        "divergence_time_s": divergence_time_s,
        "post_4s_internal_bound_fraction": post_four or {"u0": 0.0, "u1": 0.0, "u2": 0.0, "u3": 0.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_control_audit_summary(path: Path, *, u2_first_used_mismatch_time_s: float | None) -> None:
    payload = {
        "mapping_status": "model/runtime issue",
        "dominant_mismatch_axis": "u2",
        "u2_first_used_mismatch_time_s": u2_first_used_mismatch_time_s,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    assert metrics["max_lateral_error_radius"] == pytest.approx(0.09)


def test_compute_runtime_health_summary_classifies_valid_light_run(tmp_path: Path):
    run_dir = tmp_path / "4-3-26_1700_01"
    run_dir.mkdir(parents=True)
    samples = 501
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        [0.74 for _ in range(samples)],
        [0.001 * step for step in range(samples)],
        [0.02 for _ in range(samples)],
        11.0,
        20.0,
        experiment_dt_s=0.02,
    )
    _write_run_metadata(run_dir / "run_metadata.json", control_rate_hz=50.0, pred_horizon=8, reference_duration_s=10.0)

    summary = compute_runtime_health_summary(run_dir / "runtime_log.csv")

    assert summary["runtime_validity"] == "valid_runtime"
    assert summary["sample_count"] == samples
    assert summary["effective_control_rate_hz"] == pytest.approx(50.0)


def test_compute_runtime_health_summary_classifies_timing_collapse_run(tmp_path: Path):
    run_dir = tmp_path / "4-3-26_1730"
    run_dir.mkdir(parents=True)
    samples = 200
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        [0.74 for _ in range(samples)],
        [0.05 * step for step in range(samples)],
        [0.02 for _ in range(samples)],
        42.0,
        50.0,
        experiment_dt_s=0.05,
    )
    _write_run_metadata(run_dir / "run_metadata.json", control_rate_hz=50.0, pred_horizon=8, reference_duration_s=10.0)

    summary = compute_runtime_health_summary(run_dir / "runtime_log.csv")

    assert summary["runtime_validity"] == "timing_collapse"
    assert "sample_count" in str(summary["runtime_failure_reason"])


def test_evaluate_hover_gates_reports_standard_profile_pass(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1540"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", [0.0, 0.62, 0.74], [0.0, 0.03, 0.04], [0.0, 0.05, 0.08], 7.5, 11.0)
    _write_run_metadata(run_dir / "run_metadata.json")

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="standard")

    assert evaluation["passed"] is True
    assert all(bool(value) for value in evaluation["checks"].values())


def test_evaluate_hover_gates_includes_drift_sidecar_metadata_when_present(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1640_1"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", [0.0, 0.62, 0.74], [0.0, 0.03, 0.04], [0.0, 0.05, 0.08], 7.5, 11.0)
    _write_run_metadata(run_dir / "run_metadata.json")
    (run_dir / "drift_summary.json").write_text(
        json.dumps(
            {
                "selected_branch": "Branch A",
                "dominant_error_group": "x",
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="standard")

    assert evaluation["diagnostic_branch"] == "Branch A"
    assert evaluation["dominant_drift_channel"] == "x"


def test_evaluate_hover_gates_applies_light_residual_drift_checks(tmp_path: Path):
    run_dir = tmp_path / "4-1-26_1900"
    run_dir.mkdir(parents=True)
    _write_runtime_log(run_dir / "runtime_log.csv", [0.0, 0.76, 0.80], [0.0, 0.20, 0.25], [0.0, 0.02, 0.04], 12.0, 20.0)
    _write_run_metadata(run_dir / "run_metadata.json")
    (run_dir / "drift_summary.json").write_text(
        json.dumps(
            {
                "selected_branch": "Branch A",
                "dominant_error_group": "x",
                "early_window_rmse_ratio": {"x": 6.0, "dx": 2.5, "wb": 2.0},
                "post_4s_internal_bound_fraction": {"u0": 0.10, "u1": 0.0, "u2": 0.0, "u3": 0.0},
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="light")

    assert evaluation["passed"] is True
    assert evaluation["checks"]["early_window_rmse_ratio_x"] is True
    assert evaluation["checks"]["post_4s_internal_bound_fraction_u0"] is True


def test_evaluate_hover_gates_passes_light_anchor_confirmation_profile(tmp_path: Path):
    run_dir = tmp_path / "4-3-26_1700_01"
    run_dir.mkdir(parents=True)
    samples = 501
    z_values = [0.74 + 0.0001 * min(step, 10) for step in range(samples)]
    x_values = [0.0006 * step for step in range(samples)]
    u2_values = [0.02 for _ in range(samples)]
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        z_values,
        x_values,
        u2_values,
        11.5,
        20.0,
        experiment_dt_s=0.02,
    )
    _write_run_metadata(run_dir / "run_metadata.json", control_rate_hz=50.0, pred_horizon=8, reference_duration_s=10.0)
    _write_drift_summary(run_dir / "drift_summary.json", divergence_time_s=10.0)
    _write_control_audit_summary(run_dir / "control_audit_summary.json", u2_first_used_mismatch_time_s=8.4)

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="light_anchor_confirmation")

    assert evaluation["passed"] is True
    assert evaluation["runtime_validity"] == "valid_runtime"
    assert evaluation["controller_quality_evaluated"] is True
    assert evaluation["checks"]["max_lateral_error_radius"] is True
    assert evaluation["checks"]["final_position_error"] is True
    assert evaluation["checks"]["position_rmse"] is True
    assert evaluation["checks"]["divergence_reaches_end"] is True
    assert evaluation["checks"]["post_4s_internal_bound_fraction_u2"] is True
    assert evaluation["checks"]["u2_first_used_mismatch_time_s"] is True
    assert evaluation["u2_first_used_mismatch_time_s"] == pytest.approx(8.4)


def test_evaluate_hover_gates_blocks_standard_anchor_h8_promotion_when_u2_bounds_are_hot(tmp_path: Path):
    run_dir = tmp_path / "4-3-26_1800_01"
    run_dir.mkdir(parents=True)
    samples = 1001
    z_values = [0.76 for _ in range(samples)]
    x_values = [0.0008 * step for step in range(samples)]
    u2_values = [0.02 for _ in range(samples)]
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        z_values,
        x_values,
        u2_values,
        9.8,
        10.0,
        experiment_dt_s=0.01,
    )
    _write_run_metadata(run_dir / "run_metadata.json", control_rate_hz=100.0, pred_horizon=8, reference_duration_s=10.0)
    _write_drift_summary(
        run_dir / "drift_summary.json",
        divergence_time_s=10.0,
        post_four={"u0": 0.0, "u1": 0.0, "u2": 0.12, "u3": 0.0},
    )
    _write_control_audit_summary(run_dir / "control_audit_summary.json", u2_first_used_mismatch_time_s=8.8)

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="standard_anchor_h8_promotion")

    assert evaluation["passed"] is False
    assert evaluation["runtime_validity"] == "valid_runtime"
    assert evaluation["controller_quality_evaluated"] is True
    assert evaluation["checks"]["post_4s_internal_bound_fraction_u2"] is False
    assert evaluation["checks"]["divergence_reaches_end"] is True


def test_evaluate_hover_gates_stops_before_control_quality_when_runtime_is_invalid(tmp_path: Path):
    run_dir = tmp_path / "4-3-26_1730"
    run_dir.mkdir(parents=True)
    samples = 200
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        [0.74 for _ in range(samples)],
        [0.05 * step for step in range(samples)],
        [0.02 for _ in range(samples)],
        42.0,
        50.0,
        experiment_dt_s=0.05,
    )
    _write_run_metadata(run_dir / "run_metadata.json", control_rate_hz=50.0, pred_horizon=8, reference_duration_s=10.0)
    _write_drift_summary(
        run_dir / "drift_summary.json",
        divergence_time_s=2.6,
        post_four={"u0": 1.0, "u1": 1.0, "u2": 1.0, "u3": 1.0},
    )
    _write_control_audit_summary(run_dir / "control_audit_summary.json", u2_first_used_mismatch_time_s=1.6)

    evaluation = evaluate_hover_gates(run_dir / "runtime_log.csv", profile="light_anchor_confirmation")

    assert evaluation["passed"] is False
    assert evaluation["runtime_validity"] == "timing_collapse"
    assert evaluation["controller_quality_evaluated"] is False
    assert evaluation["checks"]["runtime_validity"] is False
    assert evaluation["diagnostic_branch"] is None
    assert evaluation["control_audit_mapping_status"] is None
