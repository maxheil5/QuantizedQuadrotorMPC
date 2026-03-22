from __future__ import annotations

import csv
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
    return SITLRunDataset(
        run_name=resolved_path.parent.name,
        log_path=resolved_path,
        state_history=state_history,
        control_history=control_history,
        tick_dt_ms=tick_dt_ms,
        solver_ms=solver_ms,
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
