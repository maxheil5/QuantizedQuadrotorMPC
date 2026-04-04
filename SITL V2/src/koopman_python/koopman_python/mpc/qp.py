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
    step_size: float


@dataclass(frozen=True)
class QpStructure:
    """Constant lifted-horizon matrices that do not depend on z or z_ref."""

    G: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    lower_bound: float
    upper_bound: float
    A_hat: np.ndarray
    B_hat: np.ndarray
    Q_hat: np.ndarray
    R_hat: np.ndarray
    step_size: float
    horizon: int
    z_dim: int
    control_dim: int


@dataclass(frozen=True)
class QpSolveResult:
    """Projected box-QP solve result."""

    solution: np.ndarray
    iterations: int
    converged: bool


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


def _compute_step_size(G: np.ndarray) -> float:
    G_sym = 0.5 * (G + G.T)
    eigvals = np.linalg.eigvalsh(G_sym)
    lipschitz = float(np.max(np.maximum(eigvals, 0.0)))
    return 1.0 / max(lipschitz, 1e-9)


def build_qp_structure(
    model: EdmdModel,
    horizon: int,
    *,
    lower_bound: float = -50.0,
    upper_bound: float = 50.0,
    weights: MpcWeights | None = None,
) -> QpStructure:
    """Precompute the constant lifted-horizon matrices for a fixed model and horizon."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    A = np.asarray(model.A, dtype=float)
    B = np.asarray(model.B, dtype=float)
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

    return QpStructure(
        G=G,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        A_hat=A_hat,
        B_hat=B_hat,
        Q_hat=Q_hat,
        R_hat=R_hat,
        step_size=_compute_step_size(G),
        horizon=horizon,
        z_dim=z_dim,
        control_dim=control_dim,
    )


def form_qp(
    structure: QpStructure,
    z_current: np.ndarray,
    z_reference: np.ndarray,
) -> QpProblem:
    """Form the per-step linear term for a cached lifted QP structure."""

    z = np.asarray(z_current, dtype=float).reshape(-1)
    z_ref = np.asarray(z_reference, dtype=float)
    if z_ref.ndim != 2 or z_ref.shape != (structure.z_dim, structure.horizon):
        raise ValueError("z_reference must match the cached structure dimensions.")

    y = z_ref.reshape(-1, order="F")
    F = 2.0 * structure.B_hat.T @ structure.Q_hat @ (structure.A_hat @ z - y)

    return QpProblem(
        F=F,
        G=structure.G,
        A_ineq=structure.A_ineq,
        b_ineq=structure.b_ineq,
        lower_bound=structure.lower_bound,
        upper_bound=structure.upper_bound,
        A_hat=structure.A_hat,
        B_hat=structure.B_hat,
        Q_hat=structure.Q_hat,
        R_hat=structure.R_hat,
        step_size=structure.step_size,
    )


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

    structure = build_qp_structure(
        model=model,
        horizon=horizon,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        weights=weights,
    )
    return form_qp(structure=structure, z_current=z_current, z_reference=z_reference)


def shift_warm_start(previous_solution: np.ndarray, control_dim: int) -> np.ndarray:
    """Shift the previous horizon solution forward by one control block."""

    solution = np.asarray(previous_solution, dtype=float).reshape(-1)
    if solution.size < control_dim:
        raise ValueError("previous_solution is too short for the requested control dimension.")
    if solution.size % control_dim != 0:
        raise ValueError("previous_solution length must be divisible by control_dim.")
    if solution.size == control_dim:
        return solution.copy()
    shifted = np.empty_like(solution)
    shifted[:-control_dim] = solution[control_dim:]
    shifted[-control_dim:] = solution[-control_dim:]
    return shifted


def solve_box_qp(
    problem: QpProblem,
    *,
    initial_guess: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> QpSolveResult:
    """Solve the box-constrained QP with projected gradient descent."""

    G = np.asarray(problem.G, dtype=float)
    F = np.asarray(problem.F, dtype=float).reshape(-1)
    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square.")

    step = float(problem.step_size)
    if not np.isfinite(step) or step <= 0.0:
        step = _compute_step_size(G)

    lower = np.full_like(F, problem.lower_bound, dtype=float)
    upper = np.full_like(F, problem.upper_bound, dtype=float)

    if initial_guess is None:
        u = np.zeros_like(F)
    else:
        guess = np.asarray(initial_guess, dtype=float).reshape(-1)
        if guess.shape != F.shape:
            raise ValueError("initial_guess must have the same shape as the QP decision vector.")
        u = np.clip(guess, lower, upper)

    converged = False
    iterations = max_iter
    for iteration in range(1, max_iter + 1):
        grad = G @ u + F
        candidate = np.clip(u - step * grad, lower, upper)
        if np.linalg.norm(candidate - u, ord=np.inf) <= tol:
            u = candidate
            converged = True
            iterations = iteration
            break
        u = candidate
    return QpSolveResult(solution=u, iterations=iterations, converged=converged)
