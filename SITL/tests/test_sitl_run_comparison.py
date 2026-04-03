from __future__ import annotations

import csv
import json
from pathlib import Path

from quantized_quadrotor_sitl.experiments.sitl_run_comparison import compare_sitl_runs, write_run_comparison_summary


def _write_runtime_log(
    path: Path,
    *,
    samples: int,
    experiment_dt_s: float,
    solver_ms: float,
    tick_dt_ms: float,
    final_x: float,
    final_z: float,
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
    rows: list[dict[str, float | int]] = []
    for step in range(samples):
        alpha = 0.0 if samples <= 1 else step / float(samples - 1)
        state = [0.0] * 18
        reference = [0.0] * 18
        control = [50.0, 0.0, 0.02, 0.0]
        state[0] = alpha * final_x
        state[2] = 0.75 + alpha * (final_z - 0.75)
        state[6] = 1.0
        state[10] = 1.0
        state[14] = 1.0
        reference[2] = 0.75
        reference[6] = 1.0
        reference[10] = 1.0
        reference[14] = 1.0
        row = {
            "step": step,
            "timestamp_ns": 1000 * step,
            "experiment_time_s": experiment_dt_s * step,
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


def _write_run_metadata(path: Path, *, control_rate_hz: float = 50.0, pred_horizon: int = 8) -> None:
    payload = {
        "config_path": "/tmp/runtime.yaml",
        "controller_mode": "edmd_mpc",
        "control_rate_hz": control_rate_hz,
        "pred_horizon": pred_horizon,
        "reference_duration_s": 10.0,
        "reference_seed": 2141444,
        "vehicle_scaling": {
            "max_collective_thrust_newton": 62.0,
            "max_body_torque_x_nm": 1.0,
            "max_body_torque_y_nm": 1.0,
            "max_body_torque_z_nm": 0.6,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_drift_summary(path: Path, *, divergence_time_s: float, u1: float, u2: float, u3: float) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_branch": "Branch A",
                "dominant_error_group": "dx",
                "divergence_time_s": divergence_time_s,
                "post_4s_internal_bound_fraction": {"u0": 0.0, "u1": u1, "u2": u2, "u3": u3},
            }
        ),
        encoding="utf-8",
    )


def _write_control_summary(
    path: Path,
    *,
    raw_first: float,
    used_first: float,
    raw_pre_sign: float,
    used_pre_sign: float,
    raw_pre_mag: float,
    used_pre_mag: float,
    raw_late_sign: float,
    used_late_sign: float,
    raw_late_mag: float,
    used_late_mag: float,
    late_pos_norm: float,
    late_vel_norm: float,
    anchor_frac: float,
    anchor_flips: int,
) -> None:
    payload = {
        "mapping_status": "model/runtime issue",
        "anchor_mapping_status": "learned-controller/runtime issue",
        "dominant_mismatch_axis": "u2",
        "u2_first_raw_mismatch_time_s": raw_first,
        "u2_first_used_mismatch_time_s": used_first,
        "raw_pre_divergence_sign_match_fraction_by_axis": {"u2": raw_pre_sign},
        "used_pre_divergence_sign_match_fraction_by_axis": {"u2": used_pre_sign},
        "raw_pre_divergence_mean_magnitude_ratio_by_axis": {"u2": raw_pre_mag},
        "used_pre_divergence_mean_magnitude_ratio_by_axis": {"u2": used_pre_mag},
        "u2_late_window_mean_raw_sign_match": raw_late_sign,
        "u2_late_window_mean_used_sign_match": used_late_sign,
        "u2_late_window_mean_raw_magnitude_ratio": raw_late_mag,
        "u2_late_window_mean_used_magnitude_ratio": used_late_mag,
        "u2_late_window_mean_lateral_position_residual_norm": late_pos_norm,
        "u2_late_window_mean_lateral_velocity_residual_norm": late_vel_norm,
        "anchor_intervention_fraction_by_axis": {"u2": anchor_frac},
        "anchor_sign_flip_count_by_axis": {"u2": anchor_flips},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    root: Path,
    name: str,
    *,
    samples: int,
    experiment_dt_s: float,
    solver_ms: float,
    tick_dt_ms: float,
    final_x: float,
    final_z: float,
    divergence_time_s: float,
    post_u2: float,
    raw_first: float,
    used_first: float,
    raw_pre_sign: float,
    used_pre_sign: float,
    raw_pre_mag: float,
    used_pre_mag: float,
    raw_late_sign: float,
    used_late_sign: float,
    raw_late_mag: float,
    used_late_mag: float,
    late_pos_norm: float,
    late_vel_norm: float,
    anchor_frac: float,
    anchor_flips: int,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    _write_runtime_log(
        run_dir / "runtime_log.csv",
        samples=samples,
        experiment_dt_s=experiment_dt_s,
        solver_ms=solver_ms,
        tick_dt_ms=tick_dt_ms,
        final_x=final_x,
        final_z=final_z,
    )
    _write_run_metadata(run_dir / "run_metadata.json")
    _write_drift_summary(run_dir / "drift_summary.json", divergence_time_s=divergence_time_s, u1=0.0, u2=post_u2, u3=0.0)
    _write_control_summary(
        run_dir / "control_audit_summary.json",
        raw_first=raw_first,
        used_first=used_first,
        raw_pre_sign=raw_pre_sign,
        used_pre_sign=used_pre_sign,
        raw_pre_mag=raw_pre_mag,
        used_pre_mag=used_pre_mag,
        raw_late_sign=raw_late_sign,
        used_late_sign=used_late_sign,
        raw_late_mag=raw_late_mag,
        used_late_mag=used_late_mag,
        late_pos_norm=late_pos_norm,
        late_vel_norm=late_vel_norm,
        anchor_frac=anchor_frac,
        anchor_flips=anchor_flips,
    )
    return run_dir


def test_compare_sitl_runs_classifies_raw_u2_late_sign_instability(tmp_path: Path):
    reference_run = _make_run(
        tmp_path,
        "4-3-26_1700_1",
        samples=501,
        experiment_dt_s=0.02,
        solver_ms=11.0,
        tick_dt_ms=20.0,
        final_x=0.28,
        final_z=0.80,
        divergence_time_s=10.0,
        post_u2=0.02,
        raw_first=8.3,
        used_first=8.5,
        raw_pre_sign=0.97,
        used_pre_sign=0.98,
        raw_pre_mag=0.92,
        used_pre_mag=0.96,
        raw_late_sign=0.94,
        used_late_sign=0.97,
        raw_late_mag=0.91,
        used_late_mag=0.95,
        late_pos_norm=0.22,
        late_vel_norm=0.18,
        anchor_frac=0.08,
        anchor_flips=2,
    )
    candidate_run = _make_run(
        tmp_path,
        "4-3-26_1910",
        samples=485,
        experiment_dt_s=0.020618523,
        solver_ms=10.95,
        tick_dt_ms=20.62,
        final_x=0.65,
        final_z=1.60,
        divergence_time_s=8.08,
        post_u2=0.17,
        raw_first=6.96,
        used_first=6.96,
        raw_pre_sign=0.86,
        used_pre_sign=0.92,
        raw_pre_mag=0.41,
        used_pre_mag=0.56,
        raw_late_sign=0.49,
        used_late_sign=0.63,
        raw_late_mag=0.66,
        used_late_mag=0.72,
        late_pos_norm=0.70,
        late_vel_norm=0.56,
        anchor_frac=0.25,
        anchor_flips=29,
    )

    comparison = compare_sitl_runs(reference_run, candidate_run)

    assert comparison["reference_run_name"] == "4-3-26_1700_1"
    assert comparison["candidate_run_name"] == "4-3-26_1910"
    assert comparison["candidate"]["runtime_validity"] == "valid_runtime"
    assert comparison["candidate"]["hover_gate_passed"] is False
    assert comparison["u2_regression_classification"] == "raw_u2_late_sign_instability"
    assert comparison["delta"]["u2_first_used_mismatch_time_s"] < 0.0
    assert comparison["delta"]["u2_late_window_raw_sign_match"] < 0.0
    assert comparison["delta"]["post_4s_internal_bound_fraction_u2"] > 0.0


def test_write_run_comparison_summary_uses_candidate_folder_by_default(tmp_path: Path):
    reference_run = _make_run(
        tmp_path,
        "4-3-26_1700_1",
        samples=501,
        experiment_dt_s=0.02,
        solver_ms=11.0,
        tick_dt_ms=20.0,
        final_x=0.28,
        final_z=0.80,
        divergence_time_s=10.0,
        post_u2=0.02,
        raw_first=8.3,
        used_first=8.5,
        raw_pre_sign=0.97,
        used_pre_sign=0.98,
        raw_pre_mag=0.92,
        used_pre_mag=0.96,
        raw_late_sign=0.94,
        used_late_sign=0.97,
        raw_late_mag=0.91,
        used_late_mag=0.95,
        late_pos_norm=0.22,
        late_vel_norm=0.18,
        anchor_frac=0.08,
        anchor_flips=2,
    )
    candidate_run = _make_run(
        tmp_path,
        "4-3-26_1910",
        samples=200,
        experiment_dt_s=0.05,
        solver_ms=42.0,
        tick_dt_ms=50.0,
        final_x=4.0,
        final_z=2.0,
        divergence_time_s=2.6,
        post_u2=1.0,
        raw_first=1.6,
        used_first=1.6,
        raw_pre_sign=0.4,
        used_pre_sign=0.4,
        raw_pre_mag=0.2,
        used_pre_mag=0.2,
        raw_late_sign=0.2,
        used_late_sign=0.2,
        raw_late_mag=0.2,
        used_late_mag=0.2,
        late_pos_norm=2.0,
        late_vel_norm=1.5,
        anchor_frac=0.4,
        anchor_flips=10,
    )

    output_path = write_run_comparison_summary(reference_run, candidate_run)

    assert output_path == candidate_run / "u2_comparison_to_4-3-26_1700_1.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["u2_regression_classification"] == "candidate_runtime_invalid"
    assert payload["recommended_next_step"].startswith("Treat the candidate run as invalid runtime")
