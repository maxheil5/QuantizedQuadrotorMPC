"""SE(3) lifting basis ported from MATLAB/edmd/get_basis.m."""

from __future__ import annotations

import numpy as np

from koopman_python.dynamics.srb import hat_map, unpack_state


def vectorize(matrix: np.ndarray) -> np.ndarray:
    """Match MATLAB/utils/vectorize.m column-wise flattening."""

    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def basis_dimension(n_basis: int) -> int:
    """Return the size of the MATLAB basis vector.

    MATLAB get_basis returns:
    - vec(R')
    - vec(wb_hat')
    - vec(R * wb_hat^k) for k = 1..n_basis
    """

    if n_basis < 0:
        raise ValueError("n_basis must be non-negative.")
    return 18 + 9 * n_basis


def lifted_state_dimension(n_basis: int) -> int:
    """Return the size of z = [x; dx; basis]."""

    return 6 + basis_dimension(n_basis)


def get_basis(state: np.ndarray, n_basis: int) -> np.ndarray:
    """Port of MATLAB/edmd/get_basis.m for a single SRB state.

    The input state follows the MATLAB SRB layout:
    X = [x(3), dx(3), R(:), wb(3)].
    """

    if n_basis < 0:
        raise ValueError("n_basis must be non-negative.")

    _position, _velocity, rotation, angular_velocity = unpack_state(state)
    wb_hat = hat_map(angular_velocity)

    lifted_blocks = [vectorize(rotation.T), vectorize(wb_hat.T)]

    z_term = np.asarray(rotation, dtype=float)
    for _ in range(n_basis):
        z_term = z_term @ wb_hat
        lifted_blocks.append(vectorize(z_term))

    return np.concatenate(lifted_blocks)


def lift_state(state: np.ndarray, n_basis: int) -> np.ndarray:
    """Construct z = [x; dx; basis] exactly as in the MATLAB EDMD code."""

    flat = np.asarray(state, dtype=float).reshape(18)
    return np.concatenate((flat[0:3], flat[3:6], get_basis(flat, n_basis)))


def lift_trajectory(states: np.ndarray, n_basis: int) -> np.ndarray:
    """Lift an 18 x T state trajectory into the MATLAB z-space."""

    state_matrix = np.asarray(states, dtype=float)
    if state_matrix.ndim != 2 or state_matrix.shape[0] != 18:
        raise ValueError("states must be an 18 x T matrix.")
    return np.column_stack([lift_state(state_matrix[:, i], n_basis) for i in range(state_matrix.shape[1])])
