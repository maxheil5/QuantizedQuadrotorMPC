from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.types import EDMDModel, ParsedStateSeries, RMSEBreakdown
from ..utils.metrics import rmse
from ..utils.state import parse_decoded_state_history
from .basis import lift_state


FloatArray = NDArray[np.float64]


def _predict_fixed_trajectory(
    initial_state: FloatArray,
    trajectory: FloatArray,
    control: FloatArray,
    dt: float,
    model: EDMDModel,
) -> tuple[RMSEBreakdown, FloatArray, FloatArray]:
    z0 = lift_state(initial_state, model.n_basis)
    n_prediction = 100
    z_pred = [z0]
    z = z0.copy()
    for idx in range(n_prediction):
        z = model.predict_next_lifted(z, control[:, idx])
        z_pred.append(z)
    z_pred_history = np.column_stack(z_pred)

    z_true_history = np.column_stack(
        [lift_state(trajectory[:, idx], model.n_basis) for idx in range(n_prediction + 1)]
    )
    x_ref = np.column_stack([model.C @ z_true_history[:, idx] for idx in range(z_true_history.shape[1])])
    x_pred = np.column_stack([model.C @ z_pred_history[:, idx] for idx in range(z_pred_history.shape[1])])
    return rmse(x_pred, x_ref), x_pred, x_ref


def eval_edmd_fixed_traj(
    initial_state: FloatArray,
    trajectory: FloatArray,
    control: FloatArray,
    dt: float,
    model: EDMDModel,
) -> tuple[RMSEBreakdown, ParsedStateSeries, ParsedStateSeries]:
    rmse_value, x_pred, x_ref = _predict_fixed_trajectory(initial_state, trajectory, control, dt, model)
    return rmse_value, parse_decoded_state_history(x_pred), parse_decoded_state_history(x_ref)


def eval_edmd(
    initial_state: FloatArray,
    trajectories: list[tuple[FloatArray, FloatArray]],
    dt: float,
    model: EDMDModel,
) -> tuple[dict[str, tuple[float, float]], list[RMSEBreakdown]]:
    all_scores: list[RMSEBreakdown] = []
    for trajectory, control in trajectories:
        all_scores.append(_predict_fixed_trajectory(initial_state, trajectory, control, dt, model)[0])

    def _mean_std(selector: str) -> tuple[float, float]:
        values = np.array([getattr(score, selector) for score in all_scores], dtype=float)
        return float(np.mean(values)), float(np.std(values))

    return {
        "x": _mean_std("x"),
        "dx": _mean_std("dx"),
        "theta": _mean_std("theta"),
        "wb": _mean_std("wb"),
    }, all_scores
