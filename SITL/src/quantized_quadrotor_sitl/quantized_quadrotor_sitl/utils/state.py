from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import logm

from ..core.types import ParsedStateSeries
from .linear_algebra import vee_map


FloatArray = NDArray[np.float64]


def rotation_from_state18(state: FloatArray) -> FloatArray:
    return np.asarray(state[6:15], dtype=float).reshape(3, 3, order="F")


def wb_from_state18(state: FloatArray) -> FloatArray:
    return np.asarray(state[15:18], dtype=float).reshape(3)


def takeoff_hold_trim_state18(state18: FloatArray, hover_altitude_delta_m: float = 0.75) -> FloatArray:
    state = np.asarray(state18, dtype=float).reshape(18)
    trim = state.copy()
    trim[0:3] = np.array([state[0], state[1], state[2] + hover_altitude_delta_m], dtype=float)
    trim[3:6] = 0.0
    trim[15:18] = 0.0
    return trim


def state18_to_hover_local_residual(state18: FloatArray, trim_state18: FloatArray) -> FloatArray:
    state = np.asarray(state18, dtype=float).reshape(18)
    trim = np.asarray(trim_state18, dtype=float).reshape(18)
    rotation = rotation_from_state18(state)
    trim_rotation = rotation_from_state18(trim)
    rotation_delta = trim_rotation.T @ rotation
    return np.concatenate(
        [
            state[0:3] - trim[0:3],
            state[3:6] - trim[3:6],
            rotation_delta.reshape(-1, order="F"),
            state[15:18] - trim[15:18],
        ]
    )


def state18_from_hover_local_residual(residual_state18: FloatArray, trim_state18: FloatArray) -> FloatArray:
    residual = np.asarray(residual_state18, dtype=float).reshape(18)
    trim = np.asarray(trim_state18, dtype=float).reshape(18)
    trim_rotation = rotation_from_state18(trim)
    rotation_delta = rotation_from_state18(residual)
    rotation = trim_rotation @ rotation_delta
    return np.concatenate(
        [
            residual[0:3] + trim[0:3],
            residual[3:6] + trim[3:6],
            rotation.reshape(-1, order="F"),
            residual[15:18] + trim[15:18],
        ]
    )


def state18_history_to_hover_local_residual(state_history: FloatArray, trim_state18: FloatArray) -> FloatArray:
    history = np.asarray(state_history, dtype=float)
    if history.ndim != 2 or history.shape[0] != 18:
        raise ValueError("state_history must have shape (18, N)")
    return np.column_stack(
        [state18_to_hover_local_residual(history[:, idx], trim_state18) for idx in range(history.shape[1])]
    )


def state18_history_from_hover_local_residual(
    residual_state_history: FloatArray,
    trim_state18: FloatArray,
) -> FloatArray:
    history = np.asarray(residual_state_history, dtype=float)
    if history.ndim != 2 or history.shape[0] != 18:
        raise ValueError("residual_state_history must have shape (18, N)")
    return np.column_stack(
        [state18_from_hover_local_residual(history[:, idx], trim_state18) for idx in range(history.shape[1])]
    )


def decode_lifted_prefix(decoded24: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    x = np.asarray(decoded24[0:3], dtype=float).reshape(3)
    dx = np.asarray(decoded24[3:6], dtype=float).reshape(3)
    r_stored = np.asarray(decoded24[6:15], dtype=float).reshape(3, 3, order="F")
    wb_hat_stored = np.asarray(decoded24[15:24], dtype=float).reshape(3, 3, order="F")
    r = r_stored.T
    wb = vee_map(wb_hat_stored.T)
    return x, dx, r, wb


def encode_state24_from_state18(state18: FloatArray) -> FloatArray:
    state = np.asarray(state18, dtype=float).reshape(-1)
    x = state[0:3]
    dx = state[3:6]
    r = state[6:15].reshape(3, 3, order="F")
    wb = state[15:18]
    wb_hat = np.array(
        [
            [0.0, -wb[2], wb[1]],
            [wb[2], 0.0, -wb[0]],
            [-wb[1], wb[0], 0.0],
        ],
        dtype=float,
    )
    return np.concatenate(
        [
            x,
            dx,
            r.T.reshape(-1, order="F"),
            wb_hat.T.reshape(-1, order="F"),
        ]
    )


def parse_decoded_state_history(decoded_history: FloatArray) -> ParsedStateSeries:
    x_series: list[FloatArray] = []
    dx_series: list[FloatArray] = []
    theta_series: list[FloatArray] = []
    wb_series: list[FloatArray] = []
    for idx in range(decoded_history.shape[1]):
        x, dx, r, wb = decode_lifted_prefix(decoded_history[:, idx])
        theta = vee_map(logm(r))
        x_series.append(x)
        dx_series.append(dx)
        theta_series.append(theta)
        wb_series.append(wb)
    return ParsedStateSeries(
        x=np.column_stack(x_series),
        dx=np.column_stack(dx_series),
        theta=np.column_stack(theta_series),
        wb=np.column_stack(wb_series),
    )


def parse_state18_history(state_history: FloatArray) -> ParsedStateSeries:
    x_series: list[FloatArray] = []
    dx_series: list[FloatArray] = []
    theta_series: list[FloatArray] = []
    wb_series: list[FloatArray] = []
    for idx in range(state_history.shape[1]):
        state = state_history[:, idx]
        x_series.append(state[0:3])
        dx_series.append(state[3:6])
        r = state[6:15].reshape(3, 3, order="F")
        theta_series.append(vee_map(logm(r)))
        wb_series.append(state[15:18])
    return ParsedStateSeries(
        x=np.column_stack(x_series),
        dx=np.column_stack(dx_series),
        theta=np.column_stack(theta_series),
        wb=np.column_stack(wb_series),
    )
