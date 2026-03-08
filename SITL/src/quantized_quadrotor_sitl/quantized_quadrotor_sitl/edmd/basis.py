from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.linear_algebra import hat_map, vectorize


FloatArray = NDArray[np.float64]


def get_basis(state: FloatArray, n_basis: int) -> FloatArray:
    r = np.asarray(state[6:15], dtype=float).reshape(3, 3, order="F")
    wb = np.asarray(state[15:18], dtype=float).reshape(3)
    wb_hat = hat_map(wb)

    basis_terms: list[FloatArray] = []
    z_matrix = r.copy()
    for _ in range(n_basis):
        z_matrix = z_matrix @ wb_hat
        basis_terms.append(vectorize(z_matrix))

    return np.concatenate(
        [
            vectorize(r.T),
            vectorize(wb_hat.T),
            *basis_terms,
        ]
    )


def lift_state(state: FloatArray, n_basis: int) -> FloatArray:
    return np.concatenate([state[0:3], state[3:6], get_basis(state, n_basis)])

