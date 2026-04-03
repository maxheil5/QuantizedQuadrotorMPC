from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..controllers import compute_baseline_control
from ..core.artifacts import load_edmd_artifact
from ..core.config import BaselineControllerConfig, VehicleScalingConfig
from ..dynamics.params import get_params
from ..utils.control_bounds import runtime_edmd_control_coordinates
from ..utils.io import write_csv, write_json
from .sitl_dataset import SITLRunDataset, load_sitl_run_dataset
from .sitl_drift_analysis import analyze_runtime_drift


ACTIVE_AXIS_THRESHOLDS = {
    "u1": 0.05,
    "u2": 0.05,
    "u3": 0.03,
}
SIGN_MATCH_MIN_FRACTION = 0.80
NEAR_BOUND_MARGIN_FRACTION = 0.05
ANCHOR_DECISION_AXIS = "u2"
ANCHOR_INDUCED_RAW_MIN = 0.90
ANCHOR_INDUCED_USED_MAX = 0.80


def _baseline_config_from_run(run: SITLRunDataset) -> BaselineControllerConfig:
    payload = run.run_metadata.get("baseline", {})
    if isinstance(payload, dict):
        return BaselineControllerConfig(**payload)
    return BaselineControllerConfig()


def _vehicle_scaling_from_run(run: SITLRunDataset) -> VehicleScalingConfig:
    payload = run.run_metadata.get("vehicle_scaling", {})
    if isinstance(payload, dict):
        return VehicleScalingConfig(**payload)
    return VehicleScalingConfig()


def _learned_bound_margin_fraction_from_run(run: SITLRunDataset) -> float:
    control_coordinates = run.run_metadata.get("control_coordinates", {})
    if isinstance(control_coordinates, dict) and "learned_bound_margin_fraction" in control_coordinates:
        return max(0.0, float(control_coordinates["learned_bound_margin_fraction"]))
    if "learned_bound_margin_fraction" in run.run_metadata:
        return max(0.0, float(run.run_metadata["learned_bound_margin_fraction"]))
    return 0.05


def _control_internal_history(run: SITLRunDataset, coordinates) -> np.ndarray:
    if run.control_internal_history is not None:
        return np.asarray(run.control_internal_history, dtype=float)
    return np.column_stack(
        [coordinates.physical_to_internal(run.control_history[:, idx]) for idx in range(run.sample_count)]
    )


def _fallback_tick_dt_s(run: SITLRunDataset) -> float:
    positive = np.asarray(run.tick_dt_ms, dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return 0.01
    return float(np.median(positive) / 1000.0)


def replay_baseline_control_history(
    run: SITLRunDataset,
    baseline_config: BaselineControllerConfig | None = None,
    scaling: VehicleScalingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = baseline_config or _baseline_config_from_run(run)
    vehicle_scaling = scaling or _vehicle_scaling_from_run(run)
    params = get_params()
    control_history = np.zeros((4, run.sample_count), dtype=float)
    z_integral_history = np.zeros(run.sample_count, dtype=float)
    z_error_integral = 0.0
    fallback_tick_dt_s = _fallback_tick_dt_s(run)

    for idx in range(run.sample_count):
        tick_dt_ms = float(run.tick_dt_ms[idx])
        tick_dt_s = fallback_tick_dt_s if tick_dt_ms <= 0.0 else (tick_dt_ms / 1000.0)
        z_error = float(run.reference_history[2, idx] - run.state_history[2, idx])
        z_error_integral += z_error * tick_dt_s
        z_error_integral = float(
            np.clip(
                z_error_integral,
                -config.z_integral_limit,
                config.z_integral_limit,
            )
        )
        _, control_used = compute_baseline_control(
            run.state_history[:, idx],
            run.reference_history[:, idx],
            z_error_integral,
            config,
            vehicle_scaling,
            params,
        )
        control_history[:, idx] = control_used
        z_integral_history[idx] = z_error_integral
    return control_history, z_integral_history


def _load_or_generate_drift_summary(
    log_path: Path,
    artifact_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    summary_path = output_dir / "drift_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return analyze_runtime_drift(log_path=log_path, artifact_path=artifact_path, output_dir=output_dir)


def _bound_activity_mask(
    control_internal_history: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span = np.maximum(np.asarray(upper_bounds, dtype=float) - np.asarray(lower_bounds, dtype=float), 1.0e-6)
    tolerance = NEAR_BOUND_MARGIN_FRACTION * span
    near_lower = control_internal_history <= (np.asarray(lower_bounds, dtype=float).reshape(-1, 1) + tolerance.reshape(-1, 1))
    near_upper = control_internal_history >= (np.asarray(upper_bounds, dtype=float).reshape(-1, 1) - tolerance.reshape(-1, 1))
    return near_lower, near_upper, np.logical_or(near_lower, near_upper)


def _fraction_or_one(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return 1.0
    return float(np.mean(values[mask]))


def _first_time(mask: np.ndarray, experiment_time_s: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return float(experiment_time_s[int(indices[0])])


def _mean_or_zero(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _masked_vector_mean(mask: np.ndarray, values: np.ndarray) -> list[float]:
    if not np.any(mask):
        return [0.0 for _ in range(values.shape[0])]
    return np.mean(values[:, mask], axis=1).astype(float).tolist()


def _dominant_mismatch_axis_from_fields(
    pre_div_match: dict[str, float],
    pre_div_overlap: dict[str, float],
    overlap_sample_count: dict[str, int],
) -> str:
    overlap_axes = [axis for axis, value in overlap_sample_count.items() if int(value) > 0]
    if overlap_axes:
        return min(overlap_axes, key=lambda axis: (float(pre_div_match[axis]), axis))
    return min(("u1", "u2", "u3"), key=lambda axis: (float(pre_div_overlap[axis]), axis))


def _dominant_mismatch_axis(summary: dict[str, object]) -> str:
    pre_div_match = summary["pre_divergence_sign_match_fraction_by_axis"]
    assert isinstance(pre_div_match, dict)
    pre_div_overlap = summary["pre_divergence_active_overlap_fraction_by_axis"]
    assert isinstance(pre_div_overlap, dict)
    overlap_counts = summary["pre_divergence_overlap_sample_count_by_axis"]
    assert isinstance(overlap_counts, dict)
    return _dominant_mismatch_axis_from_fields(pre_div_match, pre_div_overlap, overlap_counts)


def select_control_audit_mapping_status(summary: dict[str, object]) -> str:
    pre_div_match = summary["pre_divergence_sign_match_fraction_by_axis"]
    assert isinstance(pre_div_match, dict)
    overlap_counts = summary["pre_divergence_overlap_sample_count_by_axis"]
    assert isinstance(overlap_counts, dict)
    for axis in ("u1", "u2", "u3"):
        if int(overlap_counts[axis]) <= 0:
            continue
        if float(pre_div_match[axis]) < SIGN_MATCH_MIN_FRACTION:
            return "mapping/sign bug"
    return "model/runtime issue"


def select_anchor_mapping_status(summary: dict[str, object]) -> str:
    raw_match = summary["raw_pre_divergence_sign_match_fraction_by_axis"]
    assert isinstance(raw_match, dict)
    used_match = summary["used_pre_divergence_sign_match_fraction_by_axis"]
    assert isinstance(used_match, dict)
    if (
        float(raw_match[ANCHOR_DECISION_AXIS]) >= ANCHOR_INDUCED_RAW_MIN
        and float(used_match[ANCHOR_DECISION_AXIS]) < ANCHOR_INDUCED_USED_MAX
    ):
        return "anchor-induced sign issue"
    return "learned-controller/runtime issue"


def _sign_analysis_for_history(
    baseline_control_history: np.ndarray,
    learned_control_history: np.ndarray,
    experiment_time_s: np.ndarray,
    pre_divergence_mask: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    active_sign_match_fraction_by_axis: dict[str, float] = {}
    pre_divergence_sign_match_fraction_by_axis: dict[str, float] = {}
    active_overlap_fraction_by_axis: dict[str, float] = {}
    pre_divergence_active_overlap_fraction_by_axis: dict[str, float] = {}
    active_sample_count_by_axis: dict[str, int] = {}
    overlap_sample_count_by_axis: dict[str, int] = {}
    pre_divergence_active_sample_count_by_axis: dict[str, int] = {}
    pre_divergence_overlap_sample_count_by_axis: dict[str, int] = {}
    first_sign_mismatch_time_s_by_axis: dict[str, float | None] = {}
    mean_magnitude_ratio_by_axis: dict[str, float] = {}
    pre_divergence_mean_magnitude_ratio_by_axis: dict[str, float] = {}
    active_masks: dict[str, np.ndarray] = {}
    learned_active_masks: dict[str, np.ndarray] = {}
    overlap_masks: dict[str, np.ndarray] = {}
    sign_match_masks: dict[str, np.ndarray] = {}
    magnitude_ratio_by_axis: dict[str, np.ndarray] = {}

    for axis_idx in range(1, 4):
        axis_name = f"u{axis_idx}"
        threshold = ACTIVE_AXIS_THRESHOLDS[axis_name]
        baseline_axis = baseline_control_history[axis_idx, :]
        learned_axis = learned_control_history[axis_idx, :]
        active_masks[axis_name] = np.abs(baseline_axis) >= threshold
        learned_active_masks[axis_name] = np.abs(learned_axis) >= threshold
        overlap_masks[axis_name] = np.logical_and(active_masks[axis_name], learned_active_masks[axis_name])
        sign_match_masks[axis_name] = np.logical_and(
            overlap_masks[axis_name],
            np.sign(learned_axis) == np.sign(baseline_axis),
        )

        active_mask = active_masks[axis_name]
        overlap_mask = overlap_masks[axis_name]
        match_mask = sign_match_masks[axis_name]
        active_sample_count_by_axis[axis_name] = int(np.count_nonzero(active_mask))
        overlap_sample_count_by_axis[axis_name] = int(np.count_nonzero(overlap_mask))
        pre_active_mask = np.logical_and(active_mask, pre_divergence_mask)
        pre_overlap_mask = np.logical_and(overlap_mask, pre_divergence_mask)
        pre_divergence_active_sample_count_by_axis[axis_name] = int(np.count_nonzero(pre_active_mask))
        pre_divergence_overlap_sample_count_by_axis[axis_name] = int(np.count_nonzero(pre_overlap_mask))

        active_overlap_fraction_by_axis[axis_name] = _fraction_or_one(active_mask, overlap_mask)
        pre_divergence_active_overlap_fraction_by_axis[axis_name] = _fraction_or_one(pre_active_mask, pre_overlap_mask)
        active_sign_match_fraction_by_axis[axis_name] = _fraction_or_one(overlap_mask, match_mask)
        pre_divergence_sign_match_fraction_by_axis[axis_name] = _fraction_or_one(pre_overlap_mask, match_mask)

        magnitude_ratio = np.divide(
            np.abs(learned_axis),
            np.maximum(np.abs(baseline_axis), 1.0e-12),
        )
        magnitude_ratio_by_axis[axis_name] = magnitude_ratio
        mean_magnitude_ratio_by_axis[axis_name] = float(np.mean(magnitude_ratio[active_mask])) if np.any(active_mask) else 0.0
        pre_divergence_mean_magnitude_ratio_by_axis[axis_name] = (
            float(np.mean(magnitude_ratio[pre_active_mask])) if np.any(pre_active_mask) else 0.0
        )

        mismatch_mask = np.logical_and(overlap_mask, np.logical_not(match_mask))
        first_sign_mismatch_time_s_by_axis[axis_name] = _first_time(mismatch_mask, experiment_time_s)

    return {
        "active_sign_match_fraction_by_axis": active_sign_match_fraction_by_axis,
        "pre_divergence_sign_match_fraction_by_axis": pre_divergence_sign_match_fraction_by_axis,
        "active_overlap_fraction_by_axis": active_overlap_fraction_by_axis,
        "pre_divergence_active_overlap_fraction_by_axis": pre_divergence_active_overlap_fraction_by_axis,
        "active_sample_count_by_axis": active_sample_count_by_axis,
        "overlap_sample_count_by_axis": overlap_sample_count_by_axis,
        "pre_divergence_active_sample_count_by_axis": pre_divergence_active_sample_count_by_axis,
        "pre_divergence_overlap_sample_count_by_axis": pre_divergence_overlap_sample_count_by_axis,
        "first_sign_mismatch_time_s_by_axis": first_sign_mismatch_time_s_by_axis,
        "mean_magnitude_ratio_by_axis": mean_magnitude_ratio_by_axis,
        "pre_divergence_mean_magnitude_ratio_by_axis": pre_divergence_mean_magnitude_ratio_by_axis,
        "active_masks": active_masks,
        "learned_active_masks": learned_active_masks,
        "overlap_masks": overlap_masks,
        "sign_match_masks": sign_match_masks,
        "magnitude_ratio_by_axis": magnitude_ratio_by_axis,
    }


def _mismatch_snapshot(
    *,
    idx: int | None,
    experiment_time_s: np.ndarray,
    baseline_control_history: np.ndarray,
    raw_control_history: np.ndarray,
    used_control_history: np.ndarray,
    raw_analysis: dict[str, dict[str, float | int | None] | dict[str, np.ndarray]],
    used_analysis: dict[str, dict[str, float | int | None] | dict[str, np.ndarray]],
    lateral_position_residual: np.ndarray,
    lateral_position_residual_norm: np.ndarray,
    lateral_velocity_residual: np.ndarray,
    lateral_velocity_residual_norm: np.ndarray,
) -> dict[str, object] | None:
    if idx is None:
        return None
    axis_name = "u2"
    baseline_value = float(baseline_control_history[2, idx])
    raw_value = float(raw_control_history[2, idx])
    used_value = float(used_control_history[2, idx])
    baseline_active = bool(raw_analysis["active_masks"][axis_name][idx])
    raw_overlap = bool(raw_analysis["overlap_masks"][axis_name][idx])
    used_overlap = bool(used_analysis["overlap_masks"][axis_name][idx])
    raw_sign_match = bool(raw_analysis["sign_match_masks"][axis_name][idx]) if raw_overlap else None
    used_sign_match = bool(used_analysis["sign_match_masks"][axis_name][idx]) if used_overlap else None
    raw_magnitude_ratio = None if not baseline_active else float(abs(raw_value) / max(abs(baseline_value), 1.0e-12))
    used_magnitude_ratio = None if not baseline_active else float(abs(used_value) / max(abs(baseline_value), 1.0e-12))
    return {
        "experiment_time_s": float(experiment_time_s[idx]),
        "baseline_u2": baseline_value,
        "raw_u2": raw_value,
        "used_u2": used_value,
        "raw_sign_match": raw_sign_match,
        "used_sign_match": used_sign_match,
        "raw_magnitude_ratio": raw_magnitude_ratio,
        "used_magnitude_ratio": used_magnitude_ratio,
        "lateral_position_residual_xy": lateral_position_residual[:, idx].astype(float).tolist(),
        "lateral_position_residual_norm": float(lateral_position_residual_norm[idx]),
        "lateral_velocity_residual_xy": lateral_velocity_residual[:, idx].astype(float).tolist(),
        "lateral_velocity_residual_norm": float(lateral_velocity_residual_norm[idx]),
    }


def analyze_runtime_control_audit(
    log_path: Path,
    artifact_path: Path,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    run = load_sitl_run_dataset(log_path, state_source="used", control_source="used")
    _, artifact_metadata = load_edmd_artifact(artifact_path)
    vehicle_scaling = _vehicle_scaling_from_run(run)
    baseline_config = _baseline_config_from_run(run)
    output_root = output_dir or log_path.parent
    drift_summary = _load_or_generate_drift_summary(log_path, artifact_path, output_root)
    divergence_time_s = float(drift_summary["divergence_time_s"])

    coordinates = runtime_edmd_control_coordinates(
        vehicle_scaling,
        artifact_metadata,
        learned_bound_margin_fraction=_learned_bound_margin_fraction_from_run(run),
    )
    control_internal_history = _control_internal_history(run, coordinates)
    baseline_control_history, z_integral_history = replay_baseline_control_history(
        run,
        baseline_config=baseline_config,
        scaling=vehicle_scaling,
    )

    near_lower, near_upper, near_bound = _bound_activity_mask(
        control_internal_history,
        coordinates.internal_lower_bounds,
        coordinates.internal_upper_bounds,
    )

    pre_divergence_mask = np.asarray(run.experiment_time_s, dtype=float) <= divergence_time_s
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
    final_time_s = float(run.experiment_time_s[-1])
    late_window_start_s = max(0.0, final_time_s - 2.0)
    late_window_mask = np.asarray(run.experiment_time_s, dtype=float) >= late_window_start_s
    u2_raw_mismatch_mask = np.logical_and(
        raw_analysis["overlap_masks"]["u2"],
        np.logical_not(raw_analysis["sign_match_masks"]["u2"]),
    )
    u2_used_mismatch_mask = np.logical_and(
        used_analysis["overlap_masks"]["u2"],
        np.logical_not(used_analysis["sign_match_masks"]["u2"]),
    )
    u2_first_raw_mismatch_idx = int(np.flatnonzero(u2_raw_mismatch_mask)[0]) if np.any(u2_raw_mismatch_mask) else None
    u2_first_used_mismatch_idx = int(np.flatnonzero(u2_used_mismatch_mask)[0]) if np.any(u2_used_mismatch_mask) else None
    anchor_delta_history = run.control_used_history - run.control_raw_history
    anchor_intervention_masks: dict[str, np.ndarray] = {}
    anchor_sign_flip_masks: dict[str, np.ndarray] = {}
    trace_rows: list[dict[str, object]] = []

    for axis_idx in range(1, 4):
        axis_name = f"u{axis_idx}"
        raw_axis = run.control_raw_history[axis_idx, :]
        used_axis = run.control_used_history[axis_idx, :]
        intervention_mask = np.logical_not(np.isclose(raw_axis, used_axis, atol=1.0e-12, rtol=1.0e-9))
        sign_flip_mask = np.logical_and(
            intervention_mask,
            np.logical_and(np.sign(raw_axis) != 0.0, np.sign(used_axis) != 0.0),
        )
        sign_flip_mask = np.logical_and(sign_flip_mask, np.sign(raw_axis) != np.sign(used_axis))
        anchor_intervention_masks[axis_name] = intervention_mask
        anchor_sign_flip_masks[axis_name] = sign_flip_mask

    for idx in range(run.sample_count):
        row: dict[str, object] = {
            "step": int(idx),
            "experiment_time_s": float(run.experiment_time_s[idx]),
            "divergence_time_s": divergence_time_s,
            "baseline_z_error_integral": float(z_integral_history[idx]),
            "lateral_position_residual_x": float(lateral_position_residual[0, idx]),
            "lateral_position_residual_y": float(lateral_position_residual[1, idx]),
            "lateral_position_residual_norm": float(lateral_position_residual_norm[idx]),
            "lateral_velocity_residual_x": float(lateral_velocity_residual[0, idx]),
            "lateral_velocity_residual_y": float(lateral_velocity_residual[1, idx]),
            "lateral_velocity_residual_norm": float(lateral_velocity_residual_norm[idx]),
        }
        for control_idx in range(4):
            row[f"learned_control_{control_idx}"] = float(run.control_history[control_idx, idx])
            row[f"raw_control_{control_idx}"] = float(run.control_raw_history[control_idx, idx])
            row[f"used_control_{control_idx}"] = float(run.control_used_history[control_idx, idx])
            row[f"baseline_control_{control_idx}"] = float(baseline_control_history[control_idx, idx])
            row[f"control_internal_{control_idx}"] = float(control_internal_history[control_idx, idx])
            row[f"control_internal_near_lower_{control_idx}"] = bool(near_lower[control_idx, idx])
            row[f"control_internal_near_upper_{control_idx}"] = bool(near_upper[control_idx, idx])
            row[f"control_internal_near_bound_{control_idx}"] = bool(near_bound[control_idx, idx])
        for axis_idx in range(1, 4):
            axis_name = f"u{axis_idx}"
            threshold = ACTIVE_AXIS_THRESHOLDS[axis_name]
            baseline_value = float(baseline_control_history[axis_idx, idx])
            raw_value = float(run.control_raw_history[axis_idx, idx])
            used_value = float(run.control_used_history[axis_idx, idx])
            baseline_active = bool(raw_analysis["active_masks"][axis_name][idx])
            raw_active = bool(raw_analysis["learned_active_masks"][axis_name][idx])
            used_active = bool(used_analysis["learned_active_masks"][axis_name][idx])
            raw_overlap_active = bool(raw_analysis["overlap_masks"][axis_name][idx])
            used_overlap_active = bool(used_analysis["overlap_masks"][axis_name][idx])
            raw_sign_match = bool(raw_analysis["sign_match_masks"][axis_name][idx]) if raw_overlap_active else None
            used_sign_match = bool(used_analysis["sign_match_masks"][axis_name][idx]) if used_overlap_active else None
            raw_magnitude_ratio = None
            used_magnitude_ratio = None
            if baseline_active:
                raw_magnitude_ratio = float(abs(raw_value) / max(abs(baseline_value), 1.0e-12))
                used_magnitude_ratio = float(abs(used_value) / max(abs(baseline_value), 1.0e-12))
            row[f"{axis_name}_active_threshold_nm"] = threshold
            row[f"{axis_name}_baseline_active"] = baseline_active
            row[f"{axis_name}_raw_active"] = raw_active
            row[f"{axis_name}_used_active"] = used_active
            row[f"{axis_name}_raw_overlap_active"] = raw_overlap_active
            row[f"{axis_name}_used_overlap_active"] = used_overlap_active
            row[f"{axis_name}_raw_sign_match"] = raw_sign_match
            row[f"{axis_name}_used_sign_match"] = used_sign_match
            row[f"{axis_name}_raw_magnitude_ratio"] = raw_magnitude_ratio
            row[f"{axis_name}_used_magnitude_ratio"] = used_magnitude_ratio
            row[f"{axis_name}_anchor_applied"] = bool(anchor_intervention_masks[axis_name][idx])
            row[f"{axis_name}_anchor_delta"] = float(anchor_delta_history[axis_idx, idx])
            row[f"{axis_name}_anchor_sign_flip"] = bool(anchor_sign_flip_masks[axis_name][idx])
        trace_rows.append(row)

    first_bound_hit_time_s_by_axis: dict[str, float | None] = {}
    anchor_intervention_fraction_by_axis: dict[str, float] = {}
    first_anchor_intervention_time_s_by_axis: dict[str, float | None] = {}
    anchor_sign_flip_count_by_axis: dict[str, int] = {}

    for axis_name in ("u1", "u2", "u3"):
        anchor_intervention_fraction_by_axis[axis_name] = float(np.mean(anchor_intervention_masks[axis_name]))
        first_anchor_intervention_time_s_by_axis[axis_name] = _first_time(anchor_intervention_masks[axis_name], run.experiment_time_s)
        anchor_sign_flip_count_by_axis[axis_name] = int(np.count_nonzero(np.logical_and(anchor_sign_flip_masks[axis_name], pre_divergence_mask)))

    for axis_idx in range(4):
        axis_name = f"u{axis_idx}"
        first_bound_hit_time_s_by_axis[axis_name] = _first_time(near_bound[axis_idx, :], run.experiment_time_s)

    used_pre_divergence_sign_match_fraction_by_axis = used_analysis["pre_divergence_sign_match_fraction_by_axis"]
    used_pre_divergence_active_overlap_fraction_by_axis = used_analysis["pre_divergence_active_overlap_fraction_by_axis"]
    used_pre_divergence_overlap_sample_count_by_axis = used_analysis["pre_divergence_overlap_sample_count_by_axis"]
    u2_late_raw_overlap_mask = np.logical_and(late_window_mask, raw_analysis["overlap_masks"]["u2"])
    u2_late_used_overlap_mask = np.logical_and(late_window_mask, used_analysis["overlap_masks"]["u2"])
    u2_late_active_mask = np.logical_and(late_window_mask, raw_analysis["active_masks"]["u2"])
    summary: dict[str, object] = {
        "artifact_path": str(artifact_path),
        "log_path": str(log_path),
        "metadata_path": str(metadata_path or log_path.with_name("run_metadata.json")),
        "run_name": run.run_name,
        "cost_state_mode": str(run.run_metadata.get("cost_state_mode", "decoded24_raw")),
        "divergence_time_s": divergence_time_s,
        "active_axis_thresholds_nm": dict(ACTIVE_AXIS_THRESHOLDS),
        "active_sample_count_by_axis": used_analysis["active_sample_count_by_axis"],
        "overlap_sample_count_by_axis": used_analysis["overlap_sample_count_by_axis"],
        "pre_divergence_active_sample_count_by_axis": used_analysis["pre_divergence_active_sample_count_by_axis"],
        "pre_divergence_overlap_sample_count_by_axis": used_analysis["pre_divergence_overlap_sample_count_by_axis"],
        "active_overlap_fraction_by_axis": used_analysis["active_overlap_fraction_by_axis"],
        "pre_divergence_active_overlap_fraction_by_axis": used_analysis["pre_divergence_active_overlap_fraction_by_axis"],
        "active_sign_match_fraction_by_axis": used_analysis["active_sign_match_fraction_by_axis"],
        "pre_divergence_sign_match_fraction_by_axis": used_analysis["pre_divergence_sign_match_fraction_by_axis"],
        "mean_magnitude_ratio_by_axis": used_analysis["mean_magnitude_ratio_by_axis"],
        "pre_divergence_mean_magnitude_ratio_by_axis": used_analysis["pre_divergence_mean_magnitude_ratio_by_axis"],
        "first_sign_mismatch_time_s_by_axis": used_analysis["first_sign_mismatch_time_s_by_axis"],
        "first_bound_hit_time_s_by_axis": first_bound_hit_time_s_by_axis,
        "raw_active_sign_match_fraction_by_axis": raw_analysis["active_sign_match_fraction_by_axis"],
        "raw_pre_divergence_sign_match_fraction_by_axis": raw_analysis["pre_divergence_sign_match_fraction_by_axis"],
        "raw_mean_magnitude_ratio_by_axis": raw_analysis["mean_magnitude_ratio_by_axis"],
        "raw_pre_divergence_mean_magnitude_ratio_by_axis": raw_analysis["pre_divergence_mean_magnitude_ratio_by_axis"],
        "raw_first_sign_mismatch_time_s_by_axis": raw_analysis["first_sign_mismatch_time_s_by_axis"],
        "used_active_sign_match_fraction_by_axis": used_analysis["active_sign_match_fraction_by_axis"],
        "used_pre_divergence_sign_match_fraction_by_axis": used_analysis["pre_divergence_sign_match_fraction_by_axis"],
        "used_mean_magnitude_ratio_by_axis": used_analysis["mean_magnitude_ratio_by_axis"],
        "used_pre_divergence_mean_magnitude_ratio_by_axis": used_analysis["pre_divergence_mean_magnitude_ratio_by_axis"],
        "used_first_sign_mismatch_time_s_by_axis": used_analysis["first_sign_mismatch_time_s_by_axis"],
        "anchor_intervention_fraction_by_axis": anchor_intervention_fraction_by_axis,
        "first_anchor_intervention_time_s_by_axis": first_anchor_intervention_time_s_by_axis,
        "anchor_sign_flip_count_by_axis": anchor_sign_flip_count_by_axis,
        "u2_first_raw_mismatch_time_s": _first_time(u2_raw_mismatch_mask, run.experiment_time_s),
        "u2_first_used_mismatch_time_s": _first_time(u2_used_mismatch_mask, run.experiment_time_s),
        "u2_late_window_start_s": late_window_start_s,
        "u2_late_window_mean_raw_sign_match": _fraction_or_one(
            u2_late_raw_overlap_mask,
            raw_analysis["sign_match_masks"]["u2"],
        ),
        "u2_late_window_mean_used_sign_match": _fraction_or_one(
            u2_late_used_overlap_mask,
            used_analysis["sign_match_masks"]["u2"],
        ),
        "u2_late_window_mean_raw_magnitude_ratio": _mean_or_zero(
            u2_late_active_mask,
            raw_analysis["magnitude_ratio_by_axis"]["u2"],
        ),
        "u2_late_window_mean_used_magnitude_ratio": _mean_or_zero(
            u2_late_active_mask,
            used_analysis["magnitude_ratio_by_axis"]["u2"],
        ),
        "u2_first_raw_mismatch_snapshot": _mismatch_snapshot(
            idx=u2_first_raw_mismatch_idx,
            experiment_time_s=run.experiment_time_s,
            baseline_control_history=baseline_control_history,
            raw_control_history=run.control_raw_history,
            used_control_history=run.control_used_history,
            raw_analysis=raw_analysis,
            used_analysis=used_analysis,
            lateral_position_residual=lateral_position_residual,
            lateral_position_residual_norm=lateral_position_residual_norm,
            lateral_velocity_residual=lateral_velocity_residual,
            lateral_velocity_residual_norm=lateral_velocity_residual_norm,
        ),
        "u2_first_used_mismatch_snapshot": _mismatch_snapshot(
            idx=u2_first_used_mismatch_idx,
            experiment_time_s=run.experiment_time_s,
            baseline_control_history=baseline_control_history,
            raw_control_history=run.control_raw_history,
            used_control_history=run.control_used_history,
            raw_analysis=raw_analysis,
            used_analysis=used_analysis,
            lateral_position_residual=lateral_position_residual,
            lateral_position_residual_norm=lateral_position_residual_norm,
            lateral_velocity_residual=lateral_velocity_residual,
            lateral_velocity_residual_norm=lateral_velocity_residual_norm,
        ),
        "u2_late_window_mean_baseline_control": _mean_or_zero(late_window_mask, baseline_control_history[2, :]),
        "u2_late_window_mean_raw_control": _mean_or_zero(late_window_mask, run.control_raw_history[2, :]),
        "u2_late_window_mean_used_control": _mean_or_zero(late_window_mask, run.control_used_history[2, :]),
        "u2_late_window_mean_lateral_position_residual_xy": _masked_vector_mean(
            late_window_mask,
            lateral_position_residual,
        ),
        "u2_late_window_mean_lateral_position_residual_norm": _mean_or_zero(
            late_window_mask,
            lateral_position_residual_norm,
        ),
        "u2_late_window_mean_lateral_velocity_residual_xy": _masked_vector_mean(
            late_window_mask,
            lateral_velocity_residual,
        ),
        "u2_late_window_mean_lateral_velocity_residual_norm": _mean_or_zero(
            late_window_mask,
            lateral_velocity_residual_norm,
        ),
    }
    summary["dominant_mismatch_axis"] = _dominant_mismatch_axis(summary)
    summary["mapping_status"] = select_control_audit_mapping_status(summary)
    summary["anchor_mapping_status"] = select_anchor_mapping_status(summary)
    summary["raw_dominant_mismatch_axis"] = _dominant_mismatch_axis_from_fields(
        raw_analysis["pre_divergence_sign_match_fraction_by_axis"],
        raw_analysis["pre_divergence_active_overlap_fraction_by_axis"],
        raw_analysis["pre_divergence_overlap_sample_count_by_axis"],
    )
    summary["used_dominant_mismatch_axis"] = _dominant_mismatch_axis_from_fields(
        used_pre_divergence_sign_match_fraction_by_axis,
        used_pre_divergence_active_overlap_fraction_by_axis,
        used_pre_divergence_overlap_sample_count_by_axis,
    )

    trace_path = output_root / "control_audit_trace.csv"
    summary_path = output_root / "control_audit_summary.json"
    write_csv(trace_path, trace_rows)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze learned-vs-baseline moment-direction agreement for a SITL run.")
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze_runtime_control_audit(
        log_path=args.log_path,
        artifact_path=args.artifact_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
    )
    output_dir = args.output_dir or args.log_path.parent
    print(
        json.dumps(
            {
                "control_audit_summary_path": str(output_dir / "control_audit_summary.json"),
                "control_audit_trace_path": str(output_dir / "control_audit_trace.csv"),
                "cost_state_mode": summary["cost_state_mode"],
                "mapping_status": summary["mapping_status"],
                "anchor_mapping_status": summary["anchor_mapping_status"],
                "dominant_mismatch_axis": summary["dominant_mismatch_axis"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
