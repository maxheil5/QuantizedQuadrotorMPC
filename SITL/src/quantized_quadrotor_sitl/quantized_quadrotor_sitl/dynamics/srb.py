from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.linear_algebra import hat_map
from .params import QuadrotorParams


FloatArray = NDArray[np.float64]


def dynamics_srb(_: float, state: FloatArray, control: FloatArray, params: QuadrotorParams) -> FloatArray:
    mass = params.mass
    inertia = params.J
    g = 9.81
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)

    x = np.asarray(state[0:3], dtype=float).reshape(3)
    dx = np.asarray(state[3:6], dtype=float).reshape(3)
    r = np.asarray(state[6:15], dtype=float).reshape(3, 3, order="F")
    wb = np.asarray(state[15:18], dtype=float).reshape(3)

    fb = float(control[0])
    mb = np.asarray(control[1:4], dtype=float).reshape(3)

    ddx = (fb / mass) * e3 - g * r.T @ e3
    dr = r @ hat_map(wb)
    dwb = np.linalg.solve(inertia, mb - hat_map(wb) @ inertia @ wb)

    return np.concatenate([dx, ddx, dr.reshape(-1, order="F"), dwb])

