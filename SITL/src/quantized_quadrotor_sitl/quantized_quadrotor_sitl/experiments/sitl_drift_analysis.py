from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import logm

from ..core.artifacts import load_edmd_artifact
from ..core.config import VehicleScalingConfig
from ..core.types import RMSEBreakdown
from ..edmd.basis import lift_state
from ..utils.control_bounds import runtime_edmd_control_coordinates
from ..utils.io import write_csv, write_json
from ..utils.linear_algebra import vee_map
from ..utils.metrics import normalized_rmse
from ..utils.state import takeoff_hold_trim_state18
from ..utils.state import decode_lifted_prefix, decoded24_to_cost_state, encode_state24_from_state18, hover_local_translation_rotated
from .sitl_dataset import SITLRunDataset, load_sitl_run_dataset, transform_sitl_run_dataset_to_hover_local_residual


EARLY_WINDOW_SECONDS = 2.0
REPLAY_HORIZON_STEPS = 10
BOUND_ACTIVITY_MARGIN_FRACTION = 0.05
BRANCH_A_THETA_RATIO_THRESHOLD = 3.0
BRANCH_A_WB_RATIO_THRESHOLD = 3.0
BRANCH_A_STATE_RATIO_THRESHOLD = 4.0
BRANCH_B_DIVERGENCE_TIME_SECONDS = 4.0
BRANCH_B_BOUND_FRACTION_THRESHOLD = 0.10
DIVERGENCE_LATERAL_RADIUS_THRESHOLD_M = 1.0
DIVERGENCE_ALTITUDE_ERROR_THRESHOLD_M = 0.5
DIVERGENCE_POSITION_ERROR_THRESHOLD_M = 1.0


def _cost_state_mode_from_run(run: SITLRunDataset) -> str:
    return str(run.run_metadata.get("cost_state_mode", "decoded24_raw"))


def _decoded_groups(decoded_state: np.ndarray, cost_state_mode: str) -> dict[str, np.ndarray]:
    if cost_state_mode == "minimal_residual":
        cost_state = decoded24_to_cost_state(decoded_state, cost_state_mode)
        return {
            "x": np.asarray(cost_state[0:3], dtype=float).reshape(3),
            "dx": np.asarray(cost_state[3:6], dtype=float).reshape(3),
            "theta": np.asarray(cost_state[6:9], dtype=float).reshape(3),
            "wb": np.asarray(cost_state[9:12], dtype=float).reshape(3),
        }
    x, dx, r_matrix, wb = decode_lifted_prefix(decoded_state)
    theta = vee_map(logm(r_matrix))
    return {
        "x": np.asarray(x, dtype=float).reshape(3),
        "dx": np.asarray(dx, dtype=float).reshape(3),
        "theta": np.asarray(theta, dtype=float).reshape(3),
        "wb": np.asarray(wb, dtype=float).reshape(3),
    }


def _group_error_norms(predicted_decoded: np.ndarray, reference_decoded: np.ndarray, cost_state_mode: str) -> dict[str, float]:
    predicted_groups = _decoded_groups(predicted_decoded, cost_state_mode)
    reference_groups = _decoded_groups(reference_decoded, cost_state_mode)
    return {
        key: float(np.linalg.norm(predicted_groups[key] - reference_groups[key]))
        for key in ("x", "dx", "theta", "wb")
    }


def _rmse_breakdown_for_cost_mode(
    decoded_prediction: np.ndarray,
    decoded_reference: np.ndarray,
    cost_state_mode: str,
) -> RMSEBreakdown:
    groups_pred: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "dx", "theta", "wb")}
    groups_ref: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "dx", "theta", "wb")}
    for idx in range(decoded_prediction.shape[1]):
        predicted = _decoded_groups(decoded_prediction[:, idx], cost_state_mode)
        reference = _decoded_groups(decoded_reference[:, idx], cost_state_mode)
        for key in groups_pred:
            groups_pred[key].append(predicted[key])
            groups_ref[key].append(reference[key])
    return RMSEBreakdown(
        x=normalized_rmse(np.column_stack(groups_pred["x"]), np.column_stack(groups_ref["x"])),
        dx=normalized_rmse(np.column_stack(groups_pred["dx"]), np.column_stack(groups_ref["dx"])),
        theta=normalized_rmse(np.column_stack(groups_pred["theta"]), np.column_stack(groups_ref["theta"])),
        wb=normalized_rmse(np.column_stack(groups_pred["wb"]), np.column_stack(groups_ref["wb"])),
    )


def _artifact_eval_rmse(artifact_path: Path) -> RMSEBreakdown:
    metrics_path = artifact_path.parent / "metrics_summary.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{metrics_path} does not contain any evaluation rows")

    eval_rows = [row for row in rows if row.get("split", "").strip() == "eval"]
    source_rows = eval_rows or rows
    return RMSEBreakdown(
        x=float(np.mean([float(row["rmse_x"]) for row in source_rows])),
        dx=float(np.mean([float(row["rmse_dx"]) for row in source_rows])),
        theta=float(np.mean([float(row["rmse_theta"]) for row in source_rows])),
        wb=float(np.mean([float(row["rmse_wb"]) for row in source_rows])),
    )


def _vehicle_scaling_from_run(run: SITLRunDataset) -> VehicleScalingConfig:
    scaling_payload = run.run_metadata.get("vehicle_scaling", {})
    if isinstance(scaling_payload, dict):
        return VehicleScalingConfig(**scaling_payload)
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


def _pre_divergence_bound_fraction(
    control_internal_history: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    divergence_index: int,
) -> dict[str, float]:
    end_index = max(1, min(divergence_index, control_internal_history.shape[1]))
    history = control_internal_history[:, :end_index]
    span = np.maximum(upper_bounds - lower_bounds, 1.0e-6)
    tolerance = BOUND_ACTIVITY_MARGIN_FRACTION * span
    near_lower = history <= (lower_bounds.reshape(-1, 1) + tolerance.reshape(-1, 1))
    near_upper = history >= (upper_bounds.reshape(-1, 1) - tolerance.reshape(-1, 1))
    fraction = np.mean(np.logical_or(near_lower, near_upper), axis=1)
    return {f"u{idx}": float(fraction[idx]) for idx in range(history.shape[0])}


def _post_time_bound_fraction(
    control_internal_history: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    experiment_time_s: np.ndarray,
    start_time_s: float,
) -> dict[str, float]:
    mask = np.asarray(experiment_time_s, dtype=float) >= float(start_time_s)
    if not np.any(mask):
        mask = np.ones_like(experiment_time_s, dtype=bool)
    history = control_internal_history[:, mask]
    span = np.maximum(upper_bounds - lower_bounds, 1.0e-6)
    tolerance = BOUND_ACTIVITY_MARGIN_FRACTION * span
    near_lower = history <= (lower_bounds.reshape(-1, 1) + tolerance.reshape(-1, 1))
    near_upper = history >= (upper_bounds.reshape(-1, 1) - tolerance.reshape(-1, 1))
    fraction = np.mean(np.logical_or(near_lower, near_upper), axis=1)
    return {f"u{idx}": float(fraction[idx]) for idx in range(history.shape[0])}


def _divergence_time(run: SITLRunDataset) -> tuple[float, int]:
    position_error = run.state_history[0:3, :] - run.reference_history[0:3, :]
    position_error_norm = np.linalg.norm(position_error, axis=0)
    lateral_radius = np.linalg.norm(run.state_history[0:2, :], axis=0)
    altitude_error = np.abs(run.state_history[2, :] - run.reference_history[2, :])
    mask = np.logical_or.reduce(
        [
            position_error_norm >= DIVERGENCE_POSITION_ERROR_THRESHOLD_M,
            lateral_radius >= DIVERGENCE_LATERAL_RADIUS_THRESHOLD_M,
            altitude_error >= DIVERGENCE_ALTITUDE_ERROR_THRESHOLD_M,
        ]
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return float(run.experiment_time_s[-1]), int(run.sample_count)
    divergence_index = int(indices[0])
    return float(run.experiment_time_s[divergence_index]), divergence_index


def select_drift_branch(summary: dict[str, object]) -> str:
    ratios = summary["early_window_rmse_ratio"]
    assert isinstance(ratios, dict)
    if float(ratios["theta"]) > BRANCH_A_THETA_RATIO_THRESHOLD or float(ratios["wb"]) > BRANCH_A_WB_RATIO_THRESHOLD:
        return "Branch A"
    if any(float(ratios[key]) > BRANCH_A_STATE_RATIO_THRESHOLD for key in ("x", "dx", "theta", "wb")):
        return "Branch A"

    pre_bound = summary["pre_divergence_internal_bound_fraction"]
    assert isinstance(pre_bound, dict)
    if float(summary["divergence_time_s"]) < BRANCH_B_DIVERGENCE_TIME_SECONDS:
        return "Branch B"
    if any(float(value) > BRANCH_B_BOUND_FRACTION_THRESHOLD for value in pre_bound.values()):
        return "Branch B"
    return "Branch B"


def select_drift_branch_from_summaries(summaries: list[dict[str, object]]) -> str:
    if any(select_drift_branch(summary) == "Branch A" for summary in summaries):
        return "Branch A"
    return "Branch B"


def analyze_runtime_drift(
    log_path: Path,
    artifact_path: Path,
    output_dir: Path | None = None,
    replay_horizon_steps: int = REPLAY_HORIZON_STEPS,
    early_window_s: float = EARLY_WINDOW_SECONDS,
) -> dict[str, object]:
    run = load_sitl_run_dataset(log_path, state_source="used", control_source="used")
    model, metadata = load_edmd_artifact(artifact_path)
    cost_state_mode = _cost_state_mode_from_run(run)
    residual_enabled = bool(metadata.get("residual_enabled", False))
    analysis_run = run
    if residual_enabled:
        runtime_trim = None
        state_coordinates = run.run_metadata.get("state_coordinates", {})
        if isinstance(state_coordinates, dict) and state_coordinates.get("runtime_state_trim"):
            runtime_trim = np.asarray(state_coordinates["runtime_state_trim"], dtype=float).reshape(18)
        rotate_translation = False
        if isinstance(state_coordinates, dict):
            rotate_translation = hover_local_translation_rotated(state_coordinates.get("state_coordinates"))
        elif "state_coordinates" in metadata:
            rotate_translation = hover_local_translation_rotated(metadata.get("state_coordinates"))
        if runtime_trim is None:
            runtime_trim = takeoff_hold_trim_state18(run.state_history[:, 0])
        analysis_run = transform_sitl_run_dataset_to_hover_local_residual(
            run,
            runtime_trim,
            rotate_translation=rotate_translation,
        )
    scaling = _vehicle_scaling_from_run(run)
    coordinates = runtime_edmd_control_coordinates(
        scaling,
        metadata,
        learned_bound_margin_fraction=_learned_bound_margin_fraction_from_run(run),
    )
    control_internal_history = _control_internal_history(run, coordinates)
    artifact_eval = _artifact_eval_rmse(artifact_path)

    one_step_predicted_columns: list[np.ndarray] = []
    one_step_reference_columns: list[np.ndarray] = []
    trace_rows: list[dict[str, object]] = []

    for idx in range(analysis_run.pair_count):
        lifted_state = lift_state(analysis_run.state_history[:, idx], model.n_basis)
        one_step_predicted = model.predict_next_lifted(lifted_state, control_internal_history[:, idx])
        one_step_decoded = model.C @ one_step_predicted
        one_step_reference = encode_state24_from_state18(analysis_run.state_history[:, idx + 1])
        one_step_predicted_columns.append(one_step_decoded)
        one_step_reference_columns.append(one_step_reference)

        replay_steps = min(replay_horizon_steps, analysis_run.pair_count - idx)
        replay_lifted = lift_state(analysis_run.state_history[:, idx], model.n_basis)
        for step in range(replay_steps):
            replay_lifted = model.predict_next_lifted(replay_lifted, control_internal_history[:, idx + step])
        replay_decoded = model.C @ replay_lifted
        replay_reference = encode_state24_from_state18(analysis_run.state_history[:, idx + replay_steps])

        one_step_errors = _group_error_norms(one_step_decoded, one_step_reference, cost_state_mode)
        replay_errors = _group_error_norms(replay_decoded, replay_reference, cost_state_mode)

        trace_row: dict[str, object] = {
            "step": idx,
            "experiment_time_s": float(analysis_run.experiment_time_s[idx + 1]),
            "replay_horizon_steps": int(replay_steps),
            **{f"one_step_error_{key}": value for key, value in one_step_errors.items()},
            **{f"replay_error_{key}": value for key, value in replay_errors.items()},
        }
        for control_idx in range(control_internal_history.shape[0]):
            control_value = float(control_internal_history[control_idx, idx])
            lower_bound = float(coordinates.internal_lower_bounds[control_idx])
            upper_bound = float(coordinates.internal_upper_bounds[control_idx])
            span = max(upper_bound - lower_bound, 1.0e-6)
            tolerance = BOUND_ACTIVITY_MARGIN_FRACTION * span
            trace_row[f"control_internal_{control_idx}"] = control_value
            trace_row[f"control_internal_near_lower_{control_idx}"] = control_value <= lower_bound + tolerance
            trace_row[f"control_internal_near_upper_{control_idx}"] = control_value >= upper_bound - tolerance
        trace_rows.append(trace_row)

    one_step_predicted_history = np.column_stack(one_step_predicted_columns)
    one_step_reference_history = np.column_stack(one_step_reference_columns)
    prediction_time_s = analysis_run.experiment_time_s[1:]
    early_mask = prediction_time_s <= float(early_window_s)
    if not np.any(early_mask):
        early_mask = np.ones_like(prediction_time_s, dtype=bool)
    early_rmse = _rmse_breakdown_for_cost_mode(
        one_step_predicted_history[:, early_mask],
        one_step_reference_history[:, early_mask],
        cost_state_mode,
    )
    early_window_rmse_ratio = {
        "x": float(early_rmse.x / max(artifact_eval.x, 1.0e-12)),
        "dx": float(early_rmse.dx / max(artifact_eval.dx, 1.0e-12)),
        "theta": float(early_rmse.theta / max(artifact_eval.theta, 1.0e-12)),
        "wb": float(early_rmse.wb / max(artifact_eval.wb, 1.0e-12)),
    }
    dominant_error_group = max(early_window_rmse_ratio, key=early_window_rmse_ratio.get)
    divergence_time_s, divergence_index = _divergence_time(run)
    pre_divergence_fraction = _pre_divergence_bound_fraction(
        control_internal_history,
        coordinates.internal_lower_bounds,
        coordinates.internal_upper_bounds,
        divergence_index,
    )
    post_four_second_fraction = _post_time_bound_fraction(
        control_internal_history,
        coordinates.internal_lower_bounds,
        coordinates.internal_upper_bounds,
        run.experiment_time_s,
        start_time_s=4.0,
    )

    summary: dict[str, object] = {
        "run_name": run.run_name,
        "log_path": str(Path(log_path)),
        "artifact_path": str(Path(artifact_path)),
        "cost_state_mode": cost_state_mode,
        "residual_enabled": residual_enabled,
        "replay_horizon_steps": int(replay_horizon_steps),
        "early_window_s": float(early_window_s),
        "artifact_eval_rmse": artifact_eval.as_dict(),
        "early_window_rmse": early_rmse.as_dict(),
        "early_window_rmse_ratio": early_window_rmse_ratio,
        "divergence_time_s": divergence_time_s,
        "pre_divergence_internal_bound_fraction": pre_divergence_fraction,
        "post_4s_internal_bound_fraction": post_four_second_fraction,
        "dominant_error_group": dominant_error_group,
        "thresholds": {
            "branch_a_theta_ratio": BRANCH_A_THETA_RATIO_THRESHOLD,
            "branch_a_wb_ratio": BRANCH_A_WB_RATIO_THRESHOLD,
            "branch_a_state_ratio": BRANCH_A_STATE_RATIO_THRESHOLD,
            "branch_b_divergence_time_s": BRANCH_B_DIVERGENCE_TIME_SECONDS,
            "branch_b_bound_fraction": BRANCH_B_BOUND_FRACTION_THRESHOLD,
            "bound_activity_margin_fraction": BOUND_ACTIVITY_MARGIN_FRACTION,
            "post_4s_internal_bound_fraction_max": 0.15,
        },
    }
    summary["selected_branch"] = select_drift_branch(summary)

    destination = Path(output_dir) if output_dir is not None else Path(log_path).parent
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "drift_trace.csv", trace_rows)
    write_json(destination / "drift_summary.json", summary)
    return summary


def analyze_runtime_drift_batch(
    log_paths: list[Path],
    artifact_path: Path,
    output_dir: Path | None = None,
    replay_horizon_steps: int = REPLAY_HORIZON_STEPS,
    early_window_s: float = EARLY_WINDOW_SECONDS,
) -> dict[str, object]:
    summaries = [
        analyze_runtime_drift(
            log_path=path,
            artifact_path=artifact_path,
            output_dir=(Path(output_dir) / Path(path).parent.name) if output_dir is not None else None,
            replay_horizon_steps=replay_horizon_steps,
            early_window_s=early_window_s,
        )
        for path in log_paths
    ]
    batch_summary = {
        "artifact_path": str(Path(artifact_path)),
        "selected_branch": select_drift_branch_from_summaries(summaries),
        "runs": {str(summary["run_name"]): summary for summary in summaries},
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "drift_batch_summary.json", batch_summary)
    return batch_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SITL EDMD drift against a fitted artifact.")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--log-path", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replay-horizon-steps", type=int, default=REPLAY_HORIZON_STEPS)
    parser.add_argument("--early-window-s", type=float, default=EARLY_WINDOW_SECONDS)
    args = parser.parse_args()

    if len(args.log_path) == 1:
        summary = analyze_runtime_drift(
            log_path=args.log_path[0],
            artifact_path=args.artifact_path,
            output_dir=args.output_dir,
            replay_horizon_steps=args.replay_horizon_steps,
            early_window_s=args.early_window_s,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    batch_summary = analyze_runtime_drift_batch(
        log_paths=args.log_path,
        artifact_path=args.artifact_path,
        output_dir=args.output_dir,
        replay_horizon_steps=args.replay_horizon_steps,
        early_window_s=args.early_window_s,
    )
    print(json.dumps(batch_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
