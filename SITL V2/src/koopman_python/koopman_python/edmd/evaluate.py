"""EDMD evaluation helpers ported from the MATLAB scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from koopman_python.edmd.basis import lift_state
from koopman_python.edmd.fit import EdmdModel


@dataclass(frozen=True)
class RmseMetrics:
    x: float
    dx: float
    theta: float
    wb: float


@dataclass(frozen=True)
class ObservedStateComponents:
    x: np.ndarray
    dx: np.ndarray
    theta: np.ndarray
    wb: np.ndarray


@dataclass(frozen=True)
class EdmdEvaluationResult:
    rmse: RmseMetrics
    Z_pred: np.ndarray
    Z_true: np.ndarray
    X_pred: np.ndarray
    X_true: np.ndarray
    pred_components: ObservedStateComponents
    true_components: ObservedStateComponents


def vee_map(matrix: np.ndarray) -> np.ndarray:
    """Port of MATLAB/utils/vee_map.m."""

    skew = np.asarray(matrix, dtype=float).reshape(3, 3)
    return np.array(
        [-skew[1, 2], skew[0, 2], -skew[0, 1]],
        dtype=float,
    )


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix to SO(3) using SVD."""

    candidate = np.asarray(matrix, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(candidate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _matrix_log_vector(rotation: np.ndarray) -> np.ndarray:
    """Return vee(log(R)) for a rotation matrix using a closed-form SO(3) map."""

    R = _project_to_rotation(rotation)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta <= 1e-8:
        return 0.5 * vee_map(R - R.T)

    if np.pi - theta <= 1e-5:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= np.finfo(float).eps:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = axis / axis_norm
        return theta * axis

    return (theta / (2.0 * np.sin(theta))) * vee_map(R - R.T)


def _decode_observed_column(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one 24-state observation column from C * z."""

    flat = np.asarray(observed, dtype=float).reshape(24)
    position = flat[0:3]
    velocity = flat[3:6]
    rotation = flat[6:15].reshape(3, 3, order="F").T
    wb_hat = flat[15:24].reshape(3, 3, order="F").T
    theta = _matrix_log_vector(rotation)
    wb = vee_map(wb_hat)
    return position, velocity, theta, wb


def _decode_observed_trajectory(observed_matrix: np.ndarray) -> ObservedStateComponents:
    """Decode all columns in a 24 x T observed state trajectory."""

    matrix = np.asarray(observed_matrix, dtype=float)
    positions = []
    velocities = []
    thetas = []
    angular_velocities = []
    for i in range(matrix.shape[1]):
        position, velocity, theta, wb = _decode_observed_column(matrix[:, i])
        positions.append(position)
        velocities.append(velocity)
        thetas.append(theta)
        angular_velocities.append(wb)
    return ObservedStateComponents(
        x=np.column_stack(positions),
        dx=np.column_stack(velocities),
        theta=np.column_stack(thetas),
        wb=np.column_stack(angular_velocities),
    )


def _normalized_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Match MATLAB rmse.m, with a safe zero-reference fallback."""

    predicted_flat = np.asarray(predicted, dtype=float).reshape(-1)
    reference_flat = np.asarray(reference, dtype=float).reshape(-1)
    rmse = np.sqrt(np.mean((reference_flat - predicted_flat) ** 2))
    reference_scale = np.sqrt(np.mean(reference_flat**2))
    if reference_scale <= np.finfo(float).eps:
        return float(rmse)
    return float(rmse / reference_scale)


def compute_rmse(predicted_observed: np.ndarray, reference_observed: np.ndarray) -> RmseMetrics:
    """Port the MATLAB state-wise normalized RMSE calculation."""

    predicted = _decode_observed_trajectory(predicted_observed)
    reference = _decode_observed_trajectory(reference_observed)
    return RmseMetrics(
        x=_normalized_rmse(predicted.x, reference.x),
        dx=_normalized_rmse(predicted.dx, reference.dx),
        theta=_normalized_rmse(predicted.theta, reference.theta),
        wb=_normalized_rmse(predicted.wb, reference.wb),
    )


def evaluate_edmd_fixed_trajectory(
    initial_state: np.ndarray,
    true_states: np.ndarray,
    controls: np.ndarray,
    model: EdmdModel,
) -> EdmdEvaluationResult:
    """Port of MATLAB/edmd/eval_EDMD_fixed_traj.m.

    - ``initial_state`` is the 18-state SRB initial condition.
    - ``true_states`` is an 18 x T matrix.
    - ``controls`` is a 4 x (T - 1) matrix.
    """

    X_true_states = np.asarray(true_states, dtype=float)
    U = np.asarray(controls, dtype=float)
    if X_true_states.ndim != 2 or X_true_states.shape[0] != 18:
        raise ValueError("true_states must be an 18 x T state matrix.")
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if U.ndim != 2 or U.shape[0] != 4:
        raise ValueError("controls must be a 4 x (T - 1) control matrix.")
    if X_true_states.shape[1] != U.shape[1] + 1:
        raise ValueError("true_states must contain exactly one more column than controls.")

    z = lift_state(np.asarray(initial_state, dtype=float), model.n_basis)
    Z_pred_columns = [z]
    for i in range(U.shape[1]):
        z = model.A @ z + model.B @ U[:, i]
        Z_pred_columns.append(z)
    Z_pred = np.column_stack(Z_pred_columns)

    Z_true = np.column_stack(
        [lift_state(X_true_states[:, i], model.n_basis) for i in range(X_true_states.shape[1])]
    )

    X_pred = model.C @ Z_pred
    X_true = model.C @ Z_true
    rmse = compute_rmse(X_pred, X_true)
    pred_components = _decode_observed_trajectory(X_pred)
    true_components = _decode_observed_trajectory(X_true)

    return EdmdEvaluationResult(
        rmse=rmse,
        Z_pred=Z_pred,
        Z_true=Z_true,
        X_pred=X_pred,
        X_true=X_true,
        pred_components=pred_components,
        true_components=true_components,
    )


def evaluate_edmd(
    initial_state: np.ndarray,
    true_states: np.ndarray,
    controls: np.ndarray,
    model: EdmdModel,
) -> EdmdEvaluationResult:
    """Convenience alias for the fixed-trajectory evaluation path."""

    return evaluate_edmd_fixed_trajectory(
        initial_state=initial_state,
        true_states=true_states,
        controls=controls,
        model=model,
    )
