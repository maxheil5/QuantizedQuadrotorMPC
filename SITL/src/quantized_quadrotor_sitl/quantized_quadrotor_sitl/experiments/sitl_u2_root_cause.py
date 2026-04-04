from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..core.artifacts import load_edmd_artifact
from ..edmd.basis import lift_state
from ..utils.control_bounds import runtime_edmd_control_coordinates
from ..utils.io import write_csv, write_json
from ..utils.state import encode_state24_from_state18, hover_local_translation_rotated, takeoff_hold_trim_state18
from .sitl_control_audit import (
    _load_or_generate_drift_summary,
    _sign_analysis_for_history,
    analyze_runtime_control_audit,
    replay_baseline_control_history,
)
from .sitl_dataset import load_sitl_run_dataset, transform_sitl_run_dataset_to_hover_local_residual
from .sitl_drift_analysis import (
    _cost_state_mode_from_run,
    _decoded_groups,
    _group_error_norms,
    _learned_bound_margin_fraction_from_run,
    _vehicle_scaling_from_run,
)
from .sitl_runtime_metrics import load_control_audit_summary, load_runtime_health_summary


U2_LATE_SIGN_INSTABILITY_FIRST_MISMATCH_FRACTION_OF_DURATION = 0.8
U2_LATE_SIGN_INSTABILITY_LATE_SIGN_MATCH_MAX = 0.75
U2_UNDERCORRECTION_PRE_SIGN_MIN = 0.85
U2_UNDERCORRECTION_LATE_SIGN_MIN = 0.75
U2_UNDERCORRECTION_LATE_MAG_MAX = 0.75
U2_UNDERCORRECTION_LATE_POS_RESIDUAL_MIN = 0.4
STATE_MODEL_MISMATCH_PRE_SIGN_MIN = 0.85
STATE_MODEL_MISMATCH_LATE_SIGN_MIN = 0.75
STATE_MODEL_MISMATCH_ERROR_RATIO_MIN = 1.5


def _load_reference_index_history(log_path: Path) -> np.ndarray:
    with Path(log_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([int(row["reference_index"]) for row in rows], dtype=int)


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _group_norms(decoded_state: np.ndarray, cost_state_mode: str) -> dict[str, float]:
    groups = _decoded_groups(decoded_state, cost_state_mode)
    return {key: float(np.linalg.norm(np.asarray(value, dtype=float))) for key, value in groups.items()}


def _group_rmse_from_rows(rows: list[dict[str, float]], mask: np.ndarray) -> dict[str, float]:
    if not rows or not np.any(mask):
        return {key: 0.0 for key in ("x", "dx", "theta", "wb")}
    return {
        key: float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray([row[key] for row, keep in zip(rows, mask, strict=True) if keep], dtype=float)
                    )
                )
            )
        )
        for key in ("x", "dx", "theta", "wb")
    }


def _total_group_error(group_errors: dict[str, float]) -> float:
    return float(sum(float(group_errors[key]) for key in ("x", "dx", "theta", "wb")))


def _classify_u2_root_cause(
    runtime_validity: str | None,
    *,
    reference_duration_s: float,
    u2_first_raw_mismatch_time_s: float | None,
    u2_late_window_raw_sign_match: float | None,
    u2_late_window_raw_magnitude_ratio: float | None,
    u2_late_window_lateral_position_residual_norm: float | None,
    raw_pre_divergence_sign_match: float | None,
    one_step_prediction_rmse_before_raw_u2_mismatch: dict[str, float],
    one_step_prediction_rmse_after_raw_u2_mismatch: dict[str, float],
) -> str:
    if runtime_validity != "valid_runtime":
        return "runtime_invalid"

    late_sign_instability_first_mismatch_max_s = (
        U2_LATE_SIGN_INSTABILITY_FIRST_MISMATCH_FRACTION_OF_DURATION * max(float(reference_duration_s), 1.0e-9)
    )

    if (
        u2_first_raw_mismatch_time_s is not None
        and u2_first_raw_mismatch_time_s < late_sign_instability_first_mismatch_max_s
        and u2_late_window_raw_sign_match is not None
        and u2_late_window_raw_sign_match < U2_LATE_SIGN_INSTABILITY_LATE_SIGN_MATCH_MAX
    ):
        return "raw_u2_late_sign_instability"

    if (
        raw_pre_divergence_sign_match is not None
        and raw_pre_divergence_sign_match >= U2_UNDERCORRECTION_PRE_SIGN_MIN
        and u2_late_window_raw_sign_match is not None
        and u2_late_window_raw_sign_match >= U2_UNDERCORRECTION_LATE_SIGN_MIN
        and u2_late_window_raw_magnitude_ratio is not None
        and u2_late_window_raw_magnitude_ratio < U2_UNDERCORRECTION_LATE_MAG_MAX
        and u2_late_window_lateral_position_residual_norm is not None
        and u2_late_window_lateral_position_residual_norm >= U2_UNDERCORRECTION_LATE_POS_RESIDUAL_MIN
    ):
        return "raw_u2_under_correction"

    if (
        raw_pre_divergence_sign_match is not None
        and raw_pre_divergence_sign_match >= STATE_MODEL_MISMATCH_PRE_SIGN_MIN
        and u2_late_window_raw_sign_match is not None
        and u2_late_window_raw_sign_match >= STATE_MODEL_MISMATCH_LATE_SIGN_MIN
        and (
            (
                u2_first_raw_mismatch_time_s is not None
                and u2_first_raw_mismatch_time_s < late_sign_instability_first_mismatch_max_s
            )
            or (
                u2_late_window_lateral_position_residual_norm is not None
                and u2_late_window_lateral_position_residual_norm >= U2_UNDERCORRECTION_LATE_POS_RESIDUAL_MIN
            )
        )
    ):
        before_total = _total_group_error(one_step_prediction_rmse_before_raw_u2_mismatch)
        after_total = _total_group_error(one_step_prediction_rmse_after_raw_u2_mismatch)
        if before_total > 0.0 and after_total >= STATE_MODEL_MISMATCH_ERROR_RATIO_MIN * before_total:
            return "state_model_mismatch"

    return "inconclusive"


def analyze_runtime_u2_root_cause(
    log_path: Path,
    artifact_path: Path,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    resolved_log_path = Path(log_path)
    output_root = Path(output_dir) if output_dir is not None else resolved_log_path.parent
    run = load_sitl_run_dataset(resolved_log_path, state_source="used", control_source="used")
    model, artifact_metadata = load_edmd_artifact(artifact_path)
    runtime_health_summary = load_runtime_health_summary(resolved_log_path)
    reference_duration_s = float(runtime_health_summary.get("reference_duration_s", float(run.experiment_time_s[-1])) or float(run.experiment_time_s[-1]))
    drift_summary = _load_or_generate_drift_summary(resolved_log_path, artifact_path, output_root)
    control_summary = load_control_audit_summary(resolved_log_path)
    if control_summary is None:
        control_summary = analyze_runtime_control_audit(
            log_path=resolved_log_path,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            output_dir=output_root,
        )

    analysis_run = run
    if bool(artifact_metadata.get("residual_enabled", False)):
        runtime_trim = None
        state_coordinates = run.run_metadata.get("state_coordinates", {})
        rotate_translation = False
        if isinstance(state_coordinates, dict):
            if state_coordinates.get("runtime_state_trim"):
                runtime_trim = np.asarray(state_coordinates["runtime_state_trim"], dtype=float).reshape(18)
            rotate_translation = hover_local_translation_rotated(state_coordinates.get("state_coordinates"))
        if runtime_trim is None:
            runtime_trim = takeoff_hold_trim_state18(run.state_history[:, 0])
        analysis_run = transform_sitl_run_dataset_to_hover_local_residual(
            run,
            runtime_trim,
            rotate_translation=rotate_translation,
        )

    cost_state_mode = _cost_state_mode_from_run(run)
    reference_index_history = _load_reference_index_history(resolved_log_path)
    divergence_time_s = float(drift_summary["divergence_time_s"])
    pre_divergence_mask = np.asarray(run.experiment_time_s, dtype=float) <= divergence_time_s
    final_time_s = float(run.experiment_time_s[-1])
    late_window_start_s = max(0.0, final_time_s - 2.0)
    late_window_mask = np.asarray(run.experiment_time_s, dtype=float) >= late_window_start_s

    vehicle_scaling = _vehicle_scaling_from_run(run)
    coordinates = runtime_edmd_control_coordinates(
        vehicle_scaling,
        artifact_metadata,
        learned_bound_margin_fraction=_learned_bound_margin_fraction_from_run(run),
    )
    if run.control_internal_history is not None:
        control_internal_history = np.asarray(run.control_internal_history, dtype=float)
    else:
        control_internal_history = np.column_stack(
            [coordinates.physical_to_internal(run.control_raw_history[:, idx]) for idx in range(run.sample_count)]
        )

    baseline_control_history, _ = replay_baseline_control_history(run)
    raw_analysis = _sign_analysis_for_history(
        baseline_control_history,
        run.control_raw_history,
        run.experiment_time_s,
        pre_divergence_mask,
    )
    used_analysis = _sign_analysis_for_history(
        baseline_control_history,
        run.control_used_history,
        run.experiment_time_s,
        pre_divergence_mask,
    )

    lateral_position_residual = run.reference_history[0:2, :] - run.state_history[0:2, :]
    lateral_velocity_residual = run.reference_history[3:5, :] - run.state_history[3:5, :]
    lateral_position_residual_norm = np.linalg.norm(lateral_position_residual, axis=0)
    lateral_velocity_residual_norm = np.linalg.norm(lateral_velocity_residual, axis=0)

    prediction_error_rows: list[dict[str, float]] = []
    trace_rows: list[dict[str, object]] = []
    for idx in range(run.sample_count):
        baseline_u2 = float(baseline_control_history[2, idx])
        raw_u2 = float(run.control_raw_history[2, idx])
        used_u2 = float(run.control_used_history[2, idx])
        residual_state_decoded = encode_state24_from_state18(analysis_run.state_history[:, idx])
        residual_reference_decoded = encode_state24_from_state18(analysis_run.reference_history[:, idx])
        residual_state_norms = _group_norms(residual_state_decoded, cost_state_mode)
        residual_reference_norms = _group_norms(residual_reference_decoded, cost_state_mode)

        prediction_errors = {key: 0.0 for key in ("x", "dx", "theta", "wb")}
        if idx < analysis_run.pair_count:
            lifted_state = lift_state(analysis_run.state_history[:, idx], model.n_basis)
            predicted_lifted = model.predict_next_lifted(lifted_state, control_internal_history[:, idx])
            predicted_decoded = model.C @ predicted_lifted
            actual_next_decoded = encode_state24_from_state18(analysis_run.state_history[:, idx + 1])
            prediction_errors = _group_error_norms(predicted_decoded, actual_next_decoded, cost_state_mode)
        prediction_error_rows.append(prediction_errors)

        baseline_active = bool(raw_analysis["active_masks"]["u2"][idx])
        raw_overlap = bool(raw_analysis["overlap_masks"]["u2"][idx])
        used_overlap = bool(used_analysis["overlap_masks"]["u2"][idx])
        trace_rows.append(
            {
                "step": int(idx),
                "experiment_time_s": float(run.experiment_time_s[idx]),
                "reference_index": int(reference_index_history[idx]),
                "raw_u2": raw_u2,
                "used_u2": used_u2,
                "baseline_u2": baseline_u2,
                "raw_sign_match_u2": bool(raw_analysis["sign_match_masks"]["u2"][idx]) if raw_overlap else None,
                "used_sign_match_u2": bool(used_analysis["sign_match_masks"]["u2"][idx]) if used_overlap else None,
                "raw_magnitude_ratio_u2": (
                    float(raw_analysis["magnitude_ratio_by_axis"]["u2"][idx]) if baseline_active else None
                ),
                "used_magnitude_ratio_u2": (
                    float(used_analysis["magnitude_ratio_by_axis"]["u2"][idx]) if baseline_active else None
                ),
                "anchor_delta_u2": float(used_u2 - raw_u2),
                "lateral_position_residual_x": float(lateral_position_residual[0, idx]),
                "lateral_position_residual_y": float(lateral_position_residual[1, idx]),
                "lateral_position_residual_norm": float(lateral_position_residual_norm[idx]),
                "lateral_velocity_residual_x": float(lateral_velocity_residual[0, idx]),
                "lateral_velocity_residual_y": float(lateral_velocity_residual[1, idx]),
                "lateral_velocity_residual_norm": float(lateral_velocity_residual_norm[idx]),
                "state_group_x": residual_state_norms["x"],
                "state_group_dx": residual_state_norms["dx"],
                "state_group_theta": residual_state_norms["theta"],
                "state_group_wb": residual_state_norms["wb"],
                "reference_group_x": residual_reference_norms["x"],
                "reference_group_dx": residual_reference_norms["dx"],
                "reference_group_theta": residual_reference_norms["theta"],
                "reference_group_wb": residual_reference_norms["wb"],
                "pred_error_x": float(prediction_errors["x"]),
                "pred_error_dx": float(prediction_errors["dx"]),
                "pred_error_theta": float(prediction_errors["theta"]),
                "pred_error_wb": float(prediction_errors["wb"]),
            }
        )

    raw_mismatch_time = None
    if isinstance(control_summary, dict):
        raw_mismatch_time = control_summary.get("u2_first_raw_mismatch_time_s")
    raw_mismatch_time = None if raw_mismatch_time is None else float(raw_mismatch_time)
    pair_time_history = np.asarray(run.experiment_time_s[: analysis_run.pair_count], dtype=float)
    before_mask = np.ones(analysis_run.pair_count, dtype=bool) if raw_mismatch_time is None else pair_time_history < raw_mismatch_time
    after_mask = np.zeros(analysis_run.pair_count, dtype=bool) if raw_mismatch_time is None else pair_time_history >= raw_mismatch_time

    one_step_prediction_rmse_before_raw_u2_mismatch = _group_rmse_from_rows(prediction_error_rows[: analysis_run.pair_count], before_mask)
    one_step_prediction_rmse_after_raw_u2_mismatch = _group_rmse_from_rows(prediction_error_rows[: analysis_run.pair_count], after_mask)

    raw_pre_divergence_sign_match = None
    if isinstance(control_summary, dict):
        pre_divergence_sign_match = control_summary.get("raw_pre_divergence_sign_match_fraction_by_axis")
        if isinstance(pre_divergence_sign_match, dict):
            raw_pre_divergence_sign_match = float(pre_divergence_sign_match.get("u2")) if pre_divergence_sign_match.get("u2") is not None else None

    u2_root_cause_classification = _classify_u2_root_cause(
        runtime_health_summary.get("runtime_validity"),
        reference_duration_s=reference_duration_s,
        u2_first_raw_mismatch_time_s=raw_mismatch_time,
        u2_late_window_raw_sign_match=(
            None if control_summary is None else control_summary.get("u2_late_window_mean_raw_sign_match")
        ),
        u2_late_window_raw_magnitude_ratio=(
            None if control_summary is None else control_summary.get("u2_late_window_mean_raw_magnitude_ratio")
        ),
        u2_late_window_lateral_position_residual_norm=(
            None if control_summary is None else control_summary.get("u2_late_window_mean_lateral_position_residual_norm")
        ),
        raw_pre_divergence_sign_match=raw_pre_divergence_sign_match,
        one_step_prediction_rmse_before_raw_u2_mismatch=one_step_prediction_rmse_before_raw_u2_mismatch,
        one_step_prediction_rmse_after_raw_u2_mismatch=one_step_prediction_rmse_after_raw_u2_mismatch,
    )

    summary = {
        "artifact_path": str(artifact_path),
        "log_path": str(resolved_log_path),
        "metadata_path": str(metadata_path or resolved_log_path.with_name("run_metadata.json")),
        "run_name": run.run_name,
        "runtime_validity": runtime_health_summary.get("runtime_validity"),
        "u2_first_raw_mismatch_time_s": raw_mismatch_time,
        "u2_first_used_mismatch_time_s": None if control_summary is None else control_summary.get("u2_first_used_mismatch_time_s"),
        "u2_late_window_start_s": late_window_start_s,
        "u2_late_window_raw_sign_match": None if control_summary is None else control_summary.get("u2_late_window_mean_raw_sign_match"),
        "u2_late_window_used_sign_match": None if control_summary is None else control_summary.get("u2_late_window_mean_used_sign_match"),
        "u2_late_window_raw_magnitude_ratio": None if control_summary is None else control_summary.get("u2_late_window_mean_raw_magnitude_ratio"),
        "u2_late_window_used_magnitude_ratio": None if control_summary is None else control_summary.get("u2_late_window_mean_used_magnitude_ratio"),
        "u2_late_window_lateral_position_residual_norm": (
            None if control_summary is None else control_summary.get("u2_late_window_mean_lateral_position_residual_norm")
        ),
        "u2_late_window_lateral_velocity_residual_norm": (
            None if control_summary is None else control_summary.get("u2_late_window_mean_lateral_velocity_residual_norm")
        ),
        "one_step_prediction_rmse_before_raw_u2_mismatch": one_step_prediction_rmse_before_raw_u2_mismatch,
        "one_step_prediction_rmse_after_raw_u2_mismatch": one_step_prediction_rmse_after_raw_u2_mismatch,
        "u2_root_cause_classification": u2_root_cause_classification,
    }

    write_csv(output_root / "u2_root_cause_trace.csv", trace_rows)
    write_json(output_root / "u2_root_cause_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose late-run raw u2 failure modes for an EDMD SITL run.")
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze_runtime_u2_root_cause(
        log_path=args.log_path,
        artifact_path=args.artifact_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
    )
    output_dir = args.output_dir or args.log_path.parent
    print(
        json.dumps(
            {
                "u2_root_cause_summary_path": str(Path(output_dir) / "u2_root_cause_summary.json"),
                "u2_root_cause_trace_path": str(Path(output_dir) / "u2_root_cause_trace.csv"),
                "runtime_validity": summary["runtime_validity"],
                "u2_root_cause_classification": summary["u2_root_cause_classification"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
