from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag, logm
from scipy import sparse

from ..core.config import MPCConfig
from ..core.types import EDMDModel
from ..utils.linear_algebra import vee_map


FloatArray = NDArray[np.float64]


def get_qp(
    model: EDMDModel,
    lifted_state: FloatArray,
    lifted_reference: FloatArray,
    horizon: int,
    _: MPCConfig,
) -> tuple[FloatArray, FloatArray, sparse.csc_matrix, FloatArray]:
    a_matrix = model.A
    b_matrix = model.B
    c_matrix = model.C
    n_state = a_matrix.shape[1]
    n_input = b_matrix.shape[1]

    decoded = c_matrix @ lifted_state
    _ = decoded[0:3]
    _ = decoded[3:6]
    r_matrix = decoded[6:15].reshape(3, 3, order="F")
    _ = vee_map(logm(r_matrix.T))
    wb_hat = decoded[15:24].reshape(3, 3, order="F")
    _ = vee_map(wb_hat.T)

    qx = np.diag([1.0e4, 1.0e4, 1.0e2])
    qv = np.diag([1.0e2, 1.0e2, 1.0e2])
    qa = 1.0e2 * np.eye(9)
    qw = 1.0e2 * np.eye(9)
    q_i = np.zeros((lifted_state.shape[0], lifted_state.shape[0]), dtype=float)
    q_i[:24, :24] = block_diag(qx, qv, qa, qw)
    p_terminal = q_i.copy()
    r_i = np.diag([1.0e-6, 1.0, 1.0, 1.0])

    a_hat = np.zeros((n_state * horizon, n_state), dtype=float)
    for step in range(horizon):
        a_hat[step * n_state : (step + 1) * n_state, :] = np.linalg.matrix_power(a_matrix, step + 1)

    q_hat_blocks = [q_i.copy() for _ in range(horizon)]
    q_hat_blocks[-1] = p_terminal
    q_hat = block_diag(*q_hat_blocks)
    r_hat = block_diag(*[r_i for _ in range(horizon)])

    b_hat = np.zeros((n_state * horizon, n_input * horizon), dtype=float)
    for row in range(horizon):
        for col in range(row + 1):
            power = row - col
            block = np.linalg.matrix_power(a_matrix, power) @ b_matrix
            b_hat[
                row * n_state : (row + 1) * n_state,
                col * n_input : (col + 1) * n_input,
            ] = block

    a_ineq_i = np.kron(np.eye(4), np.array([[-1.0], [1.0]], dtype=float))
    a_ineq = block_diag(*[a_ineq_i for _ in range(horizon)])
    u_lb = -50.0
    u_ub = 50.0
    b_ineq_i = np.tile(np.array([-u_lb, u_ub], dtype=float), 4)
    b_ineq = np.concatenate([b_ineq_i for _ in range(horizon)])

    g_matrix = 2.0 * (r_hat + b_hat.T @ q_hat @ b_hat)
    y_vector = lifted_reference.reshape(-1, order="F")
    f_vector = 2.0 * b_hat.T @ q_hat @ (a_hat @ lifted_state - y_vector)

    return (
        f_vector,
        g_matrix,
        sparse.csc_matrix(a_ineq),
        b_ineq,
    )

