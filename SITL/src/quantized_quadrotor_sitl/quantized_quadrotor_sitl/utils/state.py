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
