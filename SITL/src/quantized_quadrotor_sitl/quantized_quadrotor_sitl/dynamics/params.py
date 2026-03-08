from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class QuadrotorParams:
    mass: float
    J: FloatArray
    g: float


def get_params() -> QuadrotorParams:
    return QuadrotorParams(
        mass=4.34,
        J=np.diag([0.0820, 0.0845, 0.1377]).astype(float),
        g=9.81,
    )

