from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.types import EDMDModel
from .basis import lift_state


FloatArray = NDArray[np.float64]


def get_edmd(x1: FloatArray, x2: FloatArray, u1: FloatArray, n_basis: int) -> EDMDModel:
    z1_columns: list[FloatArray] = []
    z2_columns: list[FloatArray] = []
    for idx in range(x1.shape[1]):
        z1_columns.append(lift_state(x1[:, idx], n_basis))
        z2_columns.append(lift_state(x2[:, idx], n_basis))

    z1 = np.column_stack(z1_columns)
    z2 = np.column_stack(z2_columns)
    z1_aug = np.vstack([z1, u1])
    m = z1.shape[1]

    a_matrix = (z2 @ z1_aug.T) / m
    g_matrix = (z1_aug @ z1_aug.T) / m

    c_matrix = np.zeros((24, z1.shape[0]), dtype=float)
    c_matrix[:24, :24] = np.eye(24, dtype=float)

    k_matrix = a_matrix @ np.linalg.pinv(g_matrix)
    return EDMDModel(
        A=k_matrix[:, : z1.shape[0]],
        B=k_matrix[:, z1.shape[0] :],
        C=c_matrix,
        Z1=z1,
        Z2=z2,
        n_basis=n_basis,
    )
