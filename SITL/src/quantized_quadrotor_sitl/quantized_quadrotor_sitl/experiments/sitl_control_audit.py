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


def _dominant_mismatch_axis(summary: dict[str, object]) -> str:
    pre_div_match = summary["pre_divergence_sign_match_fraction_by_axis"]
    assert isinstance(pre_div_match, dict)
    pre_div_overlap = summary["pre_divergence_active_overlap_fraction_by_axis"]
    assert isinstance(pre_div_overlap, dict)

    overlap_axes = [axis for axis, value in summary["pre_divergence_overlap_sample_count_by_axis"].items() if int(value) > 0]
    if overlap_axes:
        return min(overlap_axes, key=lambda axis: (float(pre_div_match[axis]), axis))
    return min(("u1", "u2", "u3"), key=lambda axis: (float(pre_div_overlap[axis]), axis))


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

    active_masks: dict[str, np.ndarray] = {}
    learned_active_masks: dict[str, np.ndarray] = {}
    overlap_masks: dict[str, np.ndarray] = {}
    sign_match_masks: dict[str, np.ndarray] = {}
    pre_divergence_mask = np.asarray(run.experiment_time_s, dtype=float) <= divergence_time_s
    trace_rows: list[dict[str, object]] = []

    for axis_idx in range(1, 4):
        axis_name = f"u{axis_idx}"
        threshold = ACTIVE_AXIS_THRESHOLDS[axis_name]
        baseline_axis = baseline_control_history[axis_idx, :]
        learned_axis = run.control_history[axis_idx, :]
        active_masks[axis_name] = np.abs(baseline_axis) >= threshold
        learned_active_masks[axis_name] = np.abs(learned_axis) >= threshold
        overlap_masks[axis_name] = np.logical_and(active_masks[axis_name], learned_active_masks[axis_name])
        sign_match_masks[axis_name] = np.logical_and(
            overlap_masks[axis_name],
            np.sign(learned_axis) == np.sign(baseline_axis),
        )

    for idx in range(run.sample_count):
        row: dict[str, object] = {
            "step": int(idx),
            "experiment_time_s": float(run.experiment_time_s[idx]),
            "divergence_time_s": divergence_time_s,
            "baseline_z_error_integral": float(z_integral_history[idx]),
        }
        for control_idx in range(4):
            row[f"learned_control_{control_idx}"] = float(run.control_history[control_idx, idx])
            row[f"baseline_control_{control_idx}"] = float(baseline_control_history[control_idx, idx])
            row[f"control_internal_{control_idx}"] = float(control_internal_history[control_idx, idx])
            row[f"control_internal_near_lower_{control_idx}"] = bool(near_lower[control_idx, idx])
            row[f"control_internal_near_upper_{control_idx}"] = bool(near_upper[control_idx, idx])
            row[f"control_internal_near_bound_{control_idx}"] = bool(near_bound[control_idx, idx])
        for axis_idx in range(1, 4):
            axis_name = f"u{axis_idx}"
            threshold = ACTIVE_AXIS_THRESHOLDS[axis_name]
            baseline_value = float(baseline_control_history[axis_idx, idx])
            learned_value = float(run.control_history[axis_idx, idx])
            baseline_active = bool(active_masks[axis_name][idx])
            learned_active = bool(learned_active_masks[axis_name][idx])
            overlap_active = bool(overlap_masks[axis_name][idx])
            sign_match = bool(sign_match_masks[axis_name][idx]) if overlap_active else None
            magnitude_ratio = None
            if baseline_active:
                magnitude_ratio = float(abs(learned_value) / max(abs(baseline_value), 1.0e-12))
            row[f"{axis_name}_active_threshold_nm"] = threshold
            row[f"{axis_name}_baseline_active"] = baseline_active
            row[f"{axis_name}_learned_active"] = learned_active
            row[f"{axis_name}_overlap_active"] = overlap_active
            row[f"{axis_name}_sign_match"] = sign_match
            row[f"{axis_name}_magnitude_ratio"] = magnitude_ratio
        trace_rows.append(row)

    active_sign_match_fraction_by_axis: dict[str, float] = {}
    pre_divergence_sign_match_fraction_by_axis: dict[str, float] = {}
    active_overlap_fraction_by_axis: dict[str, float] = {}
    pre_divergence_active_overlap_fraction_by_axis: dict[str, float] = {}
    active_sample_count_by_axis: dict[str, int] = {}
    overlap_sample_count_by_axis: dict[str, int] = {}
    pre_divergence_active_sample_count_by_axis: dict[str, int] = {}
    pre_divergence_overlap_sample_count_by_axis: dict[str, int] = {}
    first_sign_mismatch_time_s_by_axis: dict[str, float | None] = {}
    first_bound_hit_time_s_by_axis: dict[str, float | None] = {}
    mean_magnitude_ratio_by_axis: dict[str, float] = {}
    pre_divergence_mean_magnitude_ratio_by_axis: dict[str, float] = {}

    for axis_idx in range(1, 4):
        axis_name = f"u{axis_idx}"
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

        baseline_axis = baseline_control_history[axis_idx, :]
        learned_axis = run.control_history[axis_idx, :]
        magnitude_ratio = np.divide(
            np.abs(learned_axis),
            np.maximum(np.abs(baseline_axis), 1.0e-12),
        )
        mean_magnitude_ratio_by_axis[axis_name] = float(np.mean(magnitude_ratio[active_mask])) if np.any(active_mask) else 0.0
        pre_divergence_mean_magnitude_ratio_by_axis[axis_name] = (
            float(np.mean(magnitude_ratio[pre_active_mask])) if np.any(pre_active_mask) else 0.0
        )

        mismatch_mask = np.logical_and(overlap_mask, np.logical_not(match_mask))
        first_sign_mismatch_time_s_by_axis[axis_name] = _first_time(mismatch_mask, run.experiment_time_s)

    for axis_idx in range(4):
        axis_name = f"u{axis_idx}"
        first_bound_hit_time_s_by_axis[axis_name] = _first_time(near_bound[axis_idx, :], run.experiment_time_s)

    summary: dict[str, object] = {
        "artifact_path": str(artifact_path),
        "log_path": str(log_path),
        "metadata_path": str(metadata_path or log_path.with_name("run_metadata.json")),
        "run_name": run.run_name,
        "divergence_time_s": divergence_time_s,
        "active_axis_thresholds_nm": dict(ACTIVE_AXIS_THRESHOLDS),
        "active_sample_count_by_axis": active_sample_count_by_axis,
        "overlap_sample_count_by_axis": overlap_sample_count_by_axis,
        "pre_divergence_active_sample_count_by_axis": pre_divergence_active_sample_count_by_axis,
        "pre_divergence_overlap_sample_count_by_axis": pre_divergence_overlap_sample_count_by_axis,
        "active_overlap_fraction_by_axis": active_overlap_fraction_by_axis,
        "pre_divergence_active_overlap_fraction_by_axis": pre_divergence_active_overlap_fraction_by_axis,
        "active_sign_match_fraction_by_axis": active_sign_match_fraction_by_axis,
        "pre_divergence_sign_match_fraction_by_axis": pre_divergence_sign_match_fraction_by_axis,
        "mean_magnitude_ratio_by_axis": mean_magnitude_ratio_by_axis,
        "pre_divergence_mean_magnitude_ratio_by_axis": pre_divergence_mean_magnitude_ratio_by_axis,
        "first_sign_mismatch_time_s_by_axis": first_sign_mismatch_time_s_by_axis,
        "first_bound_hit_time_s_by_axis": first_bound_hit_time_s_by_axis,
    }
    summary["dominant_mismatch_axis"] = _dominant_mismatch_axis(summary)
    summary["mapping_status"] = select_control_audit_mapping_status(summary)

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
                "mapping_status": summary["mapping_status"],
                "dominant_mismatch_axis": summary["dominant_mismatch_axis"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
