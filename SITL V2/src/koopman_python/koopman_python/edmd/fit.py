"""EDMD fitting ported from MATLAB/edmd/get_EDMD.m."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from koopman_python.edmd.basis import lift_state, lifted_state_dimension


@dataclass(frozen=True)
class EdmdModel:
    """Compact container matching the MATLAB EDMD struct fields."""

    K: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Z1: np.ndarray
    Z2: np.ndarray
    n_basis: int


def _as_snapshot_matrix(data: np.ndarray | Iterable[np.ndarray], rows: int) -> np.ndarray:
    """Normalize snapshot inputs to a 2D matrix with one snapshot per column."""

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(rows, 1)
    if array.ndim != 2:
        raise ValueError("Snapshot data must be a 2D matrix or 1D column vector.")
    if array.shape[0] != rows:
        raise ValueError(f"Expected {rows} rows, got {array.shape[0]}.")
    return array


def build_lifted_snapshot_matrix(states: np.ndarray, n_basis: int) -> np.ndarray:
    """Lift a state snapshot matrix into the MATLAB EDMD z-space."""

    state_matrix = _as_snapshot_matrix(states, rows=18)
    lifted_columns = [lift_state(state_matrix[:, i], n_basis) for i in range(state_matrix.shape[1])]
    return np.column_stack(lifted_columns)


def build_state_observation_matrix(z_dim: int) -> np.ndarray:
    """Construct the MATLAB observation map X = C Z."""

    if z_dim < 24:
        raise ValueError("Lifted state dimension must be at least 24.")
    observation = np.zeros((24, z_dim), dtype=float)
    observation[0:24, 0:24] = np.eye(24, dtype=float)
    return observation


def fit_edmd(
    X1: np.ndarray | Iterable[np.ndarray],
    X2: np.ndarray | Iterable[np.ndarray],
    U1: np.ndarray | Iterable[np.ndarray],
    n_basis: int,
) -> EdmdModel:
    """Port of MATLAB/edmd/get_EDMD.m.

    Inputs follow the MATLAB convention:
    - X1, X2: state snapshot matrices with one state per column
    - U1: control snapshot matrix with one control vector per column
    """

    Z1 = build_lifted_snapshot_matrix(np.asarray(X1, dtype=float), n_basis)
    Z2 = build_lifted_snapshot_matrix(np.asarray(X2, dtype=float), n_basis)
    control_matrix = np.asarray(U1, dtype=float)
    if control_matrix.ndim == 1:
        control_matrix = control_matrix.reshape(-1, 1)
    if control_matrix.ndim != 2:
        raise ValueError("U1 must be a 2D control snapshot matrix or 1D column vector.")
    if Z1.shape[1] != Z2.shape[1] or Z1.shape[1] != control_matrix.shape[1]:
        raise ValueError("X1, X2, and U1 must contain the same number of snapshots.")

    Z1_aug = np.vstack((Z1, control_matrix))

    sample_count = Z1.shape[1]
    empirical_cross = (Z2 @ Z1_aug.T) / sample_count
    empirical_gram = (Z1_aug @ Z1_aug.T) / sample_count

    K = empirical_cross @ np.linalg.pinv(empirical_gram)
    A = K[:, : Z1.shape[0]]
    B = K[:, Z1.shape[0] :]
    C = build_state_observation_matrix(Z1.shape[0])

    return EdmdModel(K=K, A=A, B=B, C=C, Z1=Z1, Z2=Z2, n_basis=n_basis)


def expected_lifted_rows(n_basis: int) -> int:
    """Convenience helper for callers that need the z dimension."""

    return lifted_state_dimension(n_basis)
