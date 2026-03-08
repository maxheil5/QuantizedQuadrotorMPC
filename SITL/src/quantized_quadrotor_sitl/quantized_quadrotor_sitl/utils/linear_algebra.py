from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def hat_map(vector: FloatArray) -> FloatArray:
    v = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def vee_map(matrix: FloatArray) -> FloatArray:
    mat = np.real_if_close(np.asarray(matrix))
    return np.array(
        [
            -mat[1, 2],
            mat[0, 2],
            -mat[0, 1],
        ],
        dtype=float,
    )


def vectorize(matrix: FloatArray) -> FloatArray:
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")

