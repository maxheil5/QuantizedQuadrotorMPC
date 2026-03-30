from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.types import EDMDModel, RMSEBreakdown
from ..edmd.basis import lift_state
from ..utils.metrics import rmse
from ..utils.state import encode_state24_from_state18


@dataclass(slots=True)
class SITLRunDataset:
    run_name: str
    log_path: Path
    state_history: np.ndarray
    control_history: np.ndarray
    tick_dt_ms: np.ndarray
    solver_ms: np.ndarray
    run_metadata: dict[str, object]

    @property
    def sample_count(self) -> int:
        return int(self.state_history.shape[1])

    @property
    def pair_count(self) -> int:
        return max(0, self.sample_count - 1)


def _column_history(rows: list[dict[str, str]], prefix: str, width: int) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_{idx}"]) for row in rows] for idx in range(width)],
        dtype=float,
    )


def load_sitl_run_dataset(
    log_path: Path,
    state_source: str = "raw",
    control_source: str = "used",
) -> SITLRunDataset:
    resolved_path = Path(log_path)
    with resolved_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"{resolved_path} does not contain enough samples for identification")

    state_prefix = f"state_{state_source}"
    control_prefix = f"control_{control_source}"
    state_history = _column_history(rows, state_prefix, 18)
    control_history = _column_history(rows, control_prefix, 4)
    tick_dt_ms = np.asarray([float(row["tick_dt_ms"]) for row in rows], dtype=float)
    solver_ms = np.asarray([float(row["solver_ms"]) for row in rows], dtype=float)
    metadata_path = resolved_path.parent / "run_metadata.json"
    run_metadata: dict[str, object] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            run_metadata = json.load(stream)
    return SITLRunDataset(
        run_name=resolved_path.parent.name,
        log_path=resolved_path,
        state_history=state_history,
        control_history=control_history,
        tick_dt_ms=tick_dt_ms,
        solver_ms=solver_ms,
        run_metadata=run_metadata,
    )


def build_sitl_edmd_snapshots(
    runs: list[SITLRunDataset],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not runs:
        raise ValueError("at least one SITL run is required")

    x_history = np.hstack([run.state_history for run in runs])
    u_history = np.hstack([run.control_history for run in runs])
    x1_history = np.hstack([run.state_history[:, :-1] for run in runs])
    x2_history = np.hstack([run.state_history[:, 1:] for run in runs])
    u1_history = np.hstack([run.control_history[:, :-1] for run in runs])
    return x_history, u_history, x1_history, x2_history, u1_history


def evaluate_model_on_sitl_run(run: SITLRunDataset, model: EDMDModel) -> RMSEBreakdown:
    predicted_columns: list[np.ndarray] = []
    reference_columns: list[np.ndarray] = []
    for idx in range(run.pair_count):
        lifted_state = lift_state(run.state_history[:, idx], model.n_basis)
        predicted_lifted = model.A @ lifted_state + model.B @ run.control_history[:, idx]
        predicted_columns.append(model.C @ predicted_lifted)
        reference_columns.append(encode_state24_from_state18(run.state_history[:, idx + 1]))
    return rmse(np.column_stack(predicted_columns), np.column_stack(reference_columns))


def compute_sitl_run_diagnostics(run: SITLRunDataset) -> dict[str, float | str]:
    control_min = np.min(run.control_history, axis=1)
    control_max = np.max(run.control_history, axis=1)
    control_mean = np.mean(run.control_history, axis=1)
    control_std = np.std(run.control_history, axis=1)
    state_min = np.min(run.state_history[0:3, :], axis=1)
    state_max = np.max(run.state_history[0:3, :], axis=1)
    state_range = state_max - state_min

    scaling = run.run_metadata.get("vehicle_scaling", {})
    max_body_torque = np.asarray(
        [
            float(scaling.get("max_body_torque_x_nm", 1.0)),
            float(scaling.get("max_body_torque_y_nm", 1.0)),
            float(scaling.get("max_body_torque_z_nm", 0.6)),
        ],
        dtype=float,
    )
    collective_limit = float(scaling.get("max_collective_thrust_newton", 62.0))
    near_limit_thresholds = np.array(
        [
            collective_limit,
            max_body_torque[0],
            max_body_torque[1],
            max_body_torque[2],
        ],
        dtype=float,
    )
    near_limit_fraction = np.mean(np.abs(run.control_history) >= 0.98 * near_limit_thresholds.reshape(4, 1), axis=1)
    collective_below_1n_fraction = float(np.mean(run.control_history[0, :] < 1.0))

    diagnostics: dict[str, float | str] = {
        "run_name": run.run_name,
        "samples": float(run.sample_count),
        "pairs": float(run.pair_count),
        "tick_dt_ms_mean": float(np.mean(run.tick_dt_ms)),
        "tick_dt_ms_std": float(np.std(run.tick_dt_ms)),
        "solver_ms_mean": float(np.mean(run.solver_ms)),
        "solver_ms_std": float(np.std(run.solver_ms)),
        "collective_below_1n_fraction": collective_below_1n_fraction,
        "state_x_range": float(state_range[0]),
        "state_y_range": float(state_range[1]),
        "state_z_range": float(state_range[2]),
        "state_x_min": float(state_min[0]),
        "state_x_max": float(state_max[0]),
        "state_y_min": float(state_min[1]),
        "state_y_max": float(state_max[1]),
        "state_z_min": float(state_min[2]),
        "state_z_max": float(state_max[2]),
    }
    for idx in range(4):
        diagnostics[f"control_{idx}_min"] = float(control_min[idx])
        diagnostics[f"control_{idx}_max"] = float(control_max[idx])
        diagnostics[f"control_{idx}_mean"] = float(control_mean[idx])
        diagnostics[f"control_{idx}_std"] = float(control_std[idx])
        diagnostics[f"control_{idx}_near_limit_fraction"] = float(near_limit_fraction[idx])
    return diagnostics


def excitation_warnings_from_diagnostics(
    diagnostics_rows: list[dict[str, float | str]],
    collective_std_threshold: float = 1.0,
    body_moment_std_threshold: np.ndarray | None = None,
    z_range_threshold: float = 0.30,
    xy_range_threshold: float = 0.10,
    collective_below_1n_fraction_threshold: float = 0.25,
) -> list[str]:
    if not diagnostics_rows:
        return ["No identification runs were provided for SITL retraining."]

    moment_std_threshold = np.asarray(body_moment_std_threshold if body_moment_std_threshold is not None else [0.02, 0.02, 0.01], dtype=float)
    warnings: list[str] = []
    for row in diagnostics_rows:
        run_name = str(row["run_name"])
        if float(row["control_0_std"]) < collective_std_threshold:
            warnings.append(
                f"{run_name}: collective thrust std {float(row['control_0_std']):.3f} N is below the {collective_std_threshold:.2f} N excitation threshold."
            )
        if float(row["collective_below_1n_fraction"]) > collective_below_1n_fraction_threshold:
            warnings.append(
                f"{run_name}: collective thrust was below 1 N for {100.0 * float(row['collective_below_1n_fraction']):.1f}% of samples."
            )
        for idx in range(3):
            if float(row[f"control_{idx + 1}_std"]) < float(moment_std_threshold[idx]):
                warnings.append(
                    f"{run_name}: body-moment channel u{idx + 1} std {float(row[f'control_{idx + 1}_std']):.4f} is below the excitation threshold {float(moment_std_threshold[idx]):.4f}."
                )
        if float(row["state_z_range"]) < z_range_threshold:
            warnings.append(
                f"{run_name}: z excursion {float(row['state_z_range']):.3f} m is below the {z_range_threshold:.2f} m identification threshold."
            )
        if max(float(row["state_x_range"]), float(row["state_y_range"])) < xy_range_threshold:
            warnings.append(
                f"{run_name}: lateral excursion {max(float(row['state_x_range']), float(row['state_y_range'])):.3f} m is below the {xy_range_threshold:.2f} m identification threshold."
            )
    return warnings
