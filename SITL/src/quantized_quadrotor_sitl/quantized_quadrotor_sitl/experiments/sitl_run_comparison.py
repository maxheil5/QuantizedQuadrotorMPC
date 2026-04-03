from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..utils.io import write_json
from .sitl_milestone_summary import milestone_profile_for_metadata
from .sitl_runtime_metrics import (
    evaluate_hover_gates,
    load_control_audit_summary,
    load_drift_summary,
    load_runtime_health_summary,
)


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _delta(candidate_value: float | None, reference_value: float | None) -> float | None:
    if candidate_value is None or reference_value is None:
        return None
    return candidate_value - reference_value


def _axis_float(summary: dict[str, object], field: str, axis: str) -> float | None:
    payload = summary.get(field)
    if not isinstance(payload, dict):
        return None
    return _float_or_none(payload.get(axis))


def _run_snapshot(run_dir: Path, *, profile: str | None = None) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve(strict=False)
    metadata = _load_optional_json(resolved_run_dir / "run_metadata.json")
    selected_profile = profile or milestone_profile_for_metadata(metadata)
    log_path = resolved_run_dir / "runtime_log.csv"
    evaluation = (
        evaluate_hover_gates(log_path, profile=selected_profile)
        if selected_profile != "unsupported"
        else {
            "passed": False,
            "checks": {},
            "metrics": {},
            "runtime_validity": None,
            "runtime_failure_reason": "unsupported_profile",
            "controller_quality_evaluated": False,
            "control_audit_mapping_status": None,
            "control_audit_dominant_mismatch_axis": None,
            "u2_first_used_mismatch_time_s": None,
        }
    )
    runtime_health = load_runtime_health_summary(log_path)
    drift_summary = load_drift_summary(log_path)
    control_summary = load_control_audit_summary(log_path)
    u2_root_cause_summary = _load_optional_json(resolved_run_dir / "u2_root_cause_summary.json")
    metrics = evaluation.get("metrics", {})
    drift_post_four = drift_summary.get("post_4s_internal_bound_fraction", {}) if drift_summary else {}

    return {
        "run_name": resolved_run_dir.name,
        "run_dir": str(resolved_run_dir),
        "profile": selected_profile,
        "runtime_validity": evaluation.get("runtime_validity"),
        "runtime_failure_reason": evaluation.get("runtime_failure_reason"),
        "hover_gate_passed": evaluation.get("passed"),
        "controller_quality_evaluated": evaluation.get("controller_quality_evaluated"),
        "u2_root_cause_classification": u2_root_cause_summary.get("u2_root_cause_classification"),
        "metadata": {
            "config_path": metadata.get("config_path"),
            "controller_mode": metadata.get("controller_mode"),
            "control_rate_hz": metadata.get("control_rate_hz"),
            "pred_horizon": metadata.get("pred_horizon"),
            "reference_seed": metadata.get("reference_seed"),
        },
        "metrics": {
            "final_time_s": _float_or_none(metrics.get("final_time_s")),
            "sample_count": _float_or_none(runtime_health.get("sample_count")),
            "effective_control_rate_hz": _float_or_none(runtime_health.get("effective_control_rate_hz")),
            "tick_dt_ms_p90": _float_or_none(runtime_health.get("tick_dt_ms_p90")),
            "solver_ms_p90": _float_or_none(runtime_health.get("solver_ms_p90")),
            "final_altitude_error": _float_or_none(metrics.get("final_altitude_error")),
            "max_lateral_error_radius": _float_or_none(metrics.get("max_lateral_error_radius")),
            "final_position_error": _float_or_none(metrics.get("final_position_error")),
            "position_rmse": _float_or_none(metrics.get("position_rmse")),
            "solver_mean_ms": _float_or_none(metrics.get("solver_mean_ms")),
            "divergence_time_s": _float_or_none(None if drift_summary is None else drift_summary.get("divergence_time_s")),
            "post_4s_internal_bound_fraction_u1": _float_or_none(
                drift_post_four.get("u1") if isinstance(drift_post_four, dict) else None
            ),
            "post_4s_internal_bound_fraction_u2": _float_or_none(
                drift_post_four.get("u2") if isinstance(drift_post_four, dict) else None
            ),
            "post_4s_internal_bound_fraction_u3": _float_or_none(
                drift_post_four.get("u3") if isinstance(drift_post_four, dict) else None
            ),
        },
        "u2": {
            "mapping_status": None if control_summary is None else control_summary.get("mapping_status"),
            "anchor_mapping_status": None if control_summary is None else control_summary.get("anchor_mapping_status"),
            "dominant_mismatch_axis": None if control_summary is None else control_summary.get("dominant_mismatch_axis"),
            "first_raw_mismatch_time_s": None if control_summary is None else _float_or_none(control_summary.get("u2_first_raw_mismatch_time_s")),
            "first_used_mismatch_time_s": None if control_summary is None else _float_or_none(control_summary.get("u2_first_used_mismatch_time_s")),
            "raw_pre_divergence_sign_match": None
            if control_summary is None
            else _axis_float(control_summary, "raw_pre_divergence_sign_match_fraction_by_axis", "u2"),
            "used_pre_divergence_sign_match": None
            if control_summary is None
            else _axis_float(control_summary, "used_pre_divergence_sign_match_fraction_by_axis", "u2"),
            "raw_pre_divergence_magnitude_ratio": None
            if control_summary is None
            else _axis_float(control_summary, "raw_pre_divergence_mean_magnitude_ratio_by_axis", "u2"),
            "used_pre_divergence_magnitude_ratio": None
            if control_summary is None
            else _axis_float(control_summary, "used_pre_divergence_mean_magnitude_ratio_by_axis", "u2"),
            "late_window_raw_sign_match": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_raw_sign_match")),
            "late_window_used_sign_match": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_used_sign_match")),
            "late_window_raw_magnitude_ratio": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_raw_magnitude_ratio")),
            "late_window_used_magnitude_ratio": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_used_magnitude_ratio")),
            "late_window_lateral_position_residual_norm": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_lateral_position_residual_norm")),
            "late_window_lateral_velocity_residual_norm": None
            if control_summary is None
            else _float_or_none(control_summary.get("u2_late_window_mean_lateral_velocity_residual_norm")),
            "anchor_intervention_fraction": None
            if control_summary is None
            else _axis_float(control_summary, "anchor_intervention_fraction_by_axis", "u2"),
            "anchor_sign_flip_count": None
            if control_summary is None
            else _axis_float(control_summary, "anchor_sign_flip_count_by_axis", "u2"),
            "one_step_prediction_rmse_before_raw_mismatch": (
                None
                if not u2_root_cause_summary
                else u2_root_cause_summary.get("one_step_prediction_rmse_before_raw_u2_mismatch")
            ),
            "one_step_prediction_rmse_after_raw_mismatch": (
                None
                if not u2_root_cause_summary
                else u2_root_cause_summary.get("one_step_prediction_rmse_after_raw_u2_mismatch")
            ),
        },
    }


def _classify_u2_regression(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate.get("runtime_validity") != "valid_runtime":
        return "candidate_runtime_invalid"

    reference_u2 = reference["u2"]
    candidate_u2 = candidate["u2"]

    raw_mismatch_delta = _delta(candidate_u2["first_raw_mismatch_time_s"], reference_u2["first_raw_mismatch_time_s"])
    used_mismatch_delta = _delta(candidate_u2["first_used_mismatch_time_s"], reference_u2["first_used_mismatch_time_s"])
    raw_sign_delta = _delta(candidate_u2["late_window_raw_sign_match"], reference_u2["late_window_raw_sign_match"])
    used_sign_delta = _delta(candidate_u2["late_window_used_sign_match"], reference_u2["late_window_used_sign_match"])
    raw_mag_delta = _delta(candidate_u2["late_window_raw_magnitude_ratio"], reference_u2["late_window_raw_magnitude_ratio"])

    if (
        raw_mismatch_delta is not None
        and raw_sign_delta is not None
        and raw_mismatch_delta <= -0.75
        and raw_sign_delta <= -0.15
    ):
        return "raw_u2_late_sign_instability"

    if (
        used_mismatch_delta is not None
        and used_sign_delta is not None
        and used_mismatch_delta <= -0.75
        and used_sign_delta <= -0.15
        and (raw_sign_delta is None or raw_sign_delta > -0.10)
    ):
        return "anchor_path_regression"

    if (
        raw_mag_delta is not None
        and raw_mag_delta <= -0.15
        and (raw_sign_delta is None or raw_sign_delta >= -0.10)
    ):
        return "raw_u2_under_correction"

    return "inconclusive"


def _recommended_next_step(classification: str) -> str:
    if classification in {"candidate_runtime_invalid", "runtime_invalid"}:
        return "Treat the candidate run as invalid runtime and rerun after host cleanup before interpreting MPC behavior."
    if classification == "raw_u2_late_sign_instability":
        return "Inspect the learned raw u2 state-to-control path and late-window residual alignment before changing anchor settings."
    if classification == "anchor_path_regression":
        return "Inspect anchor intervention and sign-flip behavior on u2 before retuning the learned controller."
    if classification == "raw_u2_under_correction":
        return "Inspect raw u2 magnitude scaling and late-window lateral residual growth before changing the anchor floor."
    return "Compare raw and used u2 traces around the first mismatch window before making controller changes."


def compare_sitl_runs(
    reference_run_dir: Path,
    candidate_run_dir: Path,
    *,
    reference_profile: str | None = None,
    candidate_profile: str | None = None,
) -> dict[str, Any]:
    reference = _run_snapshot(reference_run_dir, profile=reference_profile)
    candidate = _run_snapshot(candidate_run_dir, profile=candidate_profile)
    classification = str(candidate.get("u2_root_cause_classification") or "") or _classify_u2_regression(reference, candidate)
    if classification == "inconclusive":
        classification = _classify_u2_regression(reference, candidate)

    delta_summary = {
        "final_altitude_error": _delta(candidate["metrics"]["final_altitude_error"], reference["metrics"]["final_altitude_error"]),
        "max_lateral_error_radius": _delta(
            candidate["metrics"]["max_lateral_error_radius"],
            reference["metrics"]["max_lateral_error_radius"],
        ),
        "final_position_error": _delta(candidate["metrics"]["final_position_error"], reference["metrics"]["final_position_error"]),
        "position_rmse": _delta(candidate["metrics"]["position_rmse"], reference["metrics"]["position_rmse"]),
        "solver_mean_ms": _delta(candidate["metrics"]["solver_mean_ms"], reference["metrics"]["solver_mean_ms"]),
        "divergence_time_s": _delta(candidate["metrics"]["divergence_time_s"], reference["metrics"]["divergence_time_s"]),
        "post_4s_internal_bound_fraction_u2": _delta(
            candidate["metrics"]["post_4s_internal_bound_fraction_u2"],
            reference["metrics"]["post_4s_internal_bound_fraction_u2"],
        ),
        "u2_first_raw_mismatch_time_s": _delta(
            candidate["u2"]["first_raw_mismatch_time_s"],
            reference["u2"]["first_raw_mismatch_time_s"],
        ),
        "u2_first_used_mismatch_time_s": _delta(
            candidate["u2"]["first_used_mismatch_time_s"],
            reference["u2"]["first_used_mismatch_time_s"],
        ),
        "u2_late_window_raw_sign_match": _delta(
            candidate["u2"]["late_window_raw_sign_match"],
            reference["u2"]["late_window_raw_sign_match"],
        ),
        "u2_late_window_used_sign_match": _delta(
            candidate["u2"]["late_window_used_sign_match"],
            reference["u2"]["late_window_used_sign_match"],
        ),
        "u2_late_window_raw_magnitude_ratio": _delta(
            candidate["u2"]["late_window_raw_magnitude_ratio"],
            reference["u2"]["late_window_raw_magnitude_ratio"],
        ),
        "u2_late_window_used_magnitude_ratio": _delta(
            candidate["u2"]["late_window_used_magnitude_ratio"],
            reference["u2"]["late_window_used_magnitude_ratio"],
        ),
        "u2_late_window_lateral_position_residual_norm": _delta(
            candidate["u2"]["late_window_lateral_position_residual_norm"],
            reference["u2"]["late_window_lateral_position_residual_norm"],
        ),
        "u2_late_window_lateral_velocity_residual_norm": _delta(
            candidate["u2"]["late_window_lateral_velocity_residual_norm"],
            reference["u2"]["late_window_lateral_velocity_residual_norm"],
        ),
        "u2_raw_pre_divergence_sign_match": _delta(
            candidate["u2"]["raw_pre_divergence_sign_match"],
            reference["u2"]["raw_pre_divergence_sign_match"],
        ),
        "u2_used_pre_divergence_sign_match": _delta(
            candidate["u2"]["used_pre_divergence_sign_match"],
            reference["u2"]["used_pre_divergence_sign_match"],
        ),
        "u2_anchor_intervention_fraction": _delta(
            candidate["u2"]["anchor_intervention_fraction"],
            reference["u2"]["anchor_intervention_fraction"],
        ),
        "u2_anchor_sign_flip_count": _delta(
            candidate["u2"]["anchor_sign_flip_count"],
            reference["u2"]["anchor_sign_flip_count"],
        ),
        "u2_anchor_sign_match_boost": _delta(
            _delta(candidate["u2"]["late_window_used_sign_match"], candidate["u2"]["late_window_raw_sign_match"]),
            _delta(reference["u2"]["late_window_used_sign_match"], reference["u2"]["late_window_raw_sign_match"]),
        ),
    }

    return {
        "reference_run_name": reference["run_name"],
        "candidate_run_name": candidate["run_name"],
        "same_profile": reference["profile"] == candidate["profile"],
        "reference": reference,
        "candidate": candidate,
        "delta": delta_summary,
        "u2_regression_classification": classification,
        "recommended_next_step": _recommended_next_step(classification),
    }


def write_run_comparison_summary(
    reference_run_dir: Path,
    candidate_run_dir: Path,
    *,
    output_path: Path | None = None,
    reference_profile: str | None = None,
    candidate_profile: str | None = None,
) -> Path:
    comparison = compare_sitl_runs(
        reference_run_dir,
        candidate_run_dir,
        reference_profile=reference_profile,
        candidate_profile=candidate_profile,
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(candidate_run_dir).resolve(strict=False) / f"u2_comparison_to_{Path(reference_run_dir).name}.json"
    )
    write_json(destination, comparison)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SITL runs with a focus on late-window u2 behavior.")
    parser.add_argument("--reference-run-dir", required=True, type=Path)
    parser.add_argument("--candidate-run-dir", required=True, type=Path)
    parser.add_argument("--reference-profile", type=str, default=None)
    parser.add_argument("--candidate-profile", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    output_path = write_run_comparison_summary(
        args.reference_run_dir,
        args.candidate_run_dir,
        output_path=args.output_path,
        reference_profile=args.reference_profile,
        candidate_profile=args.candidate_profile,
    )
    comparison = _load_optional_json(output_path)
    print(
        json.dumps(
            {
                "comparison_summary_path": str(output_path),
                "reference_run_name": comparison.get("reference_run_name"),
                "candidate_run_name": comparison.get("candidate_run_name"),
                "u2_regression_classification": comparison.get("u2_regression_classification"),
                "recommended_next_step": comparison.get("recommended_next_step"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
