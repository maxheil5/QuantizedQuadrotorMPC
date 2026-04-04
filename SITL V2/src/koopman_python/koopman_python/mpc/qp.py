"""Lifted-state MPC QP construction for the V2 EDMD controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from koopman_python.edmd.fit import EdmdModel


@dataclass(frozen=True)
class MpcWeights:
    """Cost weights matching the current MATLAB MPC defaults."""

    qx: np.ndarray
    qv: np.ndarray
    qa: np.ndarray
    qw: np.ndarray
    r: np.ndarray


@dataclass(frozen=True)
class QpProblem:
    """QP matrices and box bounds for the lifted MPC horizon."""

    F: np.ndarray
    G: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    lower_bound: float
    upper_bound: float
    A_hat: np.ndarray
    B_hat: np.ndarray
    Q_hat: np.ndarray
    R_hat: np.ndarray


def default_mpc_weights() -> MpcWeights:
    return MpcWeights(
        qx=np.diag([1e4, 1e4, 1e2]).astype(float),
        qv=np.diag([1e2, 1e2, 1e2]).astype(float),
        qa=(1e2 * np.eye(9)).astype(float),
        qw=(1e2 * np.eye(9)).astype(float),
        r=np.diag([1e-6, 1.0, 1.0, 1.0]).astype(float),
    )


def build_state_cost_matrix(z_dim: int, weights: MpcWeights | None = None) -> np.ndarray:
    """Build the MATLAB-style lifted-state stage cost."""

    w = default_mpc_weights() if weights is None else weights
    q_i = np.zeros((z_dim, z_dim), dtype=float)
    q_i[0:24, 0:24] = np.block(
        [
            [w.qx, np.zeros((3, 3)), np.zeros((3, 9)), np.zeros((3, 9))],
            [np.zeros((3, 3)), w.qv, np.zeros((3, 9)), np.zeros((3, 9))],
            [np.zeros((9, 3)), np.zeros((9, 3)), w.qa, np.zeros((9, 9))],
            [np.zeros((9, 3)), np.zeros((9, 3)), np.zeros((9, 9)), w.qw],
        ]
    )
    return q_i


def build_qp(
    model: EdmdModel,
    z_current: np.ndarray,
    z_reference: np.ndarray,
    horizon: int,
    *,
    lower_bound: float = -50.0,
    upper_bound: float = 50.0,
    weights: MpcWeights | None = None,
) -> QpProblem:
    """Port of MATLAB/mpc/get_QP.m.

    The returned problem matches the lifted-horizon cost:
    1/2 U^T G U + F^T U, subject to box constraints.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    A = np.asarray(model.A, dtype=float)
    B = np.asarray(model.B, dtype=float)
    z = np.asarray(z_current, dtype=float).reshape(-1)
    z_ref = np.asarray(z_reference, dtype=float)
    if z_ref.ndim != 2 or z_ref.shape[1] != horizon:
        raise ValueError("z_reference must be a z_dim x horizon matrix.")

    z_dim = A.shape[1]
    control_dim = B.shape[1]
    q_i = build_state_cost_matrix(z_dim, weights=weights)
    p = q_i.copy()
    r_i = (default_mpc_weights() if weights is None else weights).r

    A_hat = np.zeros((z_dim * horizon, z_dim), dtype=float)
    Q_hat = np.zeros((z_dim * horizon, z_dim * horizon), dtype=float)
    for i in range(horizon):
        A_hat[i * z_dim : (i + 1) * z_dim, :] = np.linalg.matrix_power(A, i + 1)
        Q_hat[i * z_dim : (i + 1) * z_dim, i * z_dim : (i + 1) * z_dim] = q_i
    Q_hat[-z_dim:, -z_dim:] = p

    B_hat = np.zeros((z_dim * horizon, control_dim * horizon), dtype=float)
    R_hat = np.zeros((control_dim * horizon, control_dim * horizon), dtype=float)
    for i in range(horizon):
        R_hat[i * control_dim : (i + 1) * control_dim, i * control_dim : (i + 1) * control_dim] = r_i
        for j in range(i + 1):
            power = i - j
            block = np.linalg.matrix_power(A, power) @ B
            row_slice = slice(i * z_dim, (i + 1) * z_dim)
            col_slice = slice(j * control_dim, (j + 1) * control_dim)
            B_hat[row_slice, col_slice] = block

    A_ineq_i = np.kron(np.eye(control_dim), np.array([[-1.0], [1.0]])).reshape(2 * control_dim, control_dim)
    A_ineq = np.zeros((2 * control_dim * horizon, control_dim * horizon), dtype=float)
    b_ineq = np.zeros((2 * control_dim * horizon,), dtype=float)
    b_block = np.tile(np.array([-lower_bound, upper_bound], dtype=float), control_dim)
    for i in range(horizon):
        row_slice = slice(i * 2 * control_dim, (i + 1) * 2 * control_dim)
        col_slice = slice(i * control_dim, (i + 1) * control_dim)
        A_ineq[row_slice, col_slice] = A_ineq_i
        b_ineq[row_slice] = b_block

    G = 2.0 * (R_hat + B_hat.T @ Q_hat @ B_hat)
    y = z_ref.reshape(-1, order="F")
    F = 2.0 * B_hat.T @ Q_hat @ (A_hat @ z - y)

    return QpProblem(
        F=F,
        G=G,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        A_hat=A_hat,
        B_hat=B_hat,
        Q_hat=Q_hat,
        R_hat=R_hat,
    )


def solve_box_qp(
    problem: QpProblem,
    *,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve the box-constrained QP with projected gradient descent."""

    G = np.asarray(problem.G, dtype=float)
    F = np.asarray(problem.F, dtype=float).reshape(-1)
    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square.")

    eigvals = np.linalg.eigvalsh(0.5 * (G + G.T))
    lipschitz = float(np.max(np.maximum(eigvals, 0.0)))
    step = 1.0 / max(lipschitz, 1e-9)

    u = np.zeros_like(F)
    lower = np.full_like(F, problem.lower_bound, dtype=float)
    upper = np.full_like(F, problem.upper_bound, dtype=float)

    for _ in range(max_iter):
        grad = G @ u + F
        candidate = np.clip(u - step * grad, lower, upper)
        if np.linalg.norm(candidate - u, ord=np.inf) <= tol:
            u = candidate
            break
        u = candidate
    return u
