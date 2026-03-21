from __future__ import annotations

import numpy as np

from ..core.config import BaselineControllerConfig, VehicleScalingConfig
from ..dynamics.params import QuadrotorParams
from ..utils.linear_algebra import vee_map


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    candidate = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(candidate))
    if norm > 1.0e-9:
        return candidate / norm
    fallback_vec = np.asarray(fallback, dtype=float).reshape(3)
    fallback_norm = float(np.linalg.norm(fallback_vec))
    if fallback_norm > 1.0e-9:
        return fallback_vec / fallback_norm
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    u_mat, _, vh_mat = np.linalg.svd(matrix)
    projected = u_mat @ vh_mat
    if np.linalg.det(projected) < 0.0:
        u_mat[:, -1] *= -1.0
        projected = u_mat @ vh_mat
    return projected


def _desired_rotation(total_acceleration_world: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
    body_z_desired = _normalize(total_acceleration_world, np.array([0.0, 0.0, 1.0], dtype=float))
    heading_hint = np.asarray(reference_rotation[:, 0], dtype=float).reshape(3)
    heading_hint[2] = 0.0
    body_y_desired = _normalize(np.cross(body_z_desired, heading_hint), np.array([0.0, 1.0, 0.0], dtype=float))
    body_x_desired = _normalize(np.cross(body_y_desired, body_z_desired), np.array([1.0, 0.0, 0.0], dtype=float))
    return np.column_stack([body_x_desired, body_y_desired, body_z_desired])


def _limit_tilt(total_acceleration_world: np.ndarray, max_tilt_deg: float) -> np.ndarray:
    limited = np.asarray(total_acceleration_world, dtype=float).reshape(3).copy()
    vertical_component = max(limited[2], 1.0e-3)
    max_tilt_rad = np.deg2rad(float(max_tilt_deg))
    max_horizontal = np.tan(max_tilt_rad) * vertical_component
    horizontal_norm = float(np.linalg.norm(limited[0:2]))
    if horizontal_norm > max_horizontal > 0.0:
        limited[0:2] *= max_horizontal / horizontal_norm
    return limited


def compute_baseline_control(
    state: np.ndarray,
    reference: np.ndarray,
    z_error_integral: float,
    config: BaselineControllerConfig,
    scaling: VehicleScalingConfig,
    params: QuadrotorParams,
) -> tuple[np.ndarray, np.ndarray]:
    state_vec = np.asarray(state, dtype=float).reshape(18)
    reference_vec = np.asarray(reference, dtype=float).reshape(18)

    position = state_vec[0:3]
    velocity = state_vec[3:6]
    rotation = _project_rotation(state_vec[6:15].reshape(3, 3, order="F"))
    angular_velocity = state_vec[15:18]

    reference_position = reference_vec[0:3]
    reference_velocity = reference_vec[3:6]
    reference_rotation = _project_rotation(reference_vec[6:15].reshape(3, 3, order="F"))

    position_error = reference_position - position
    velocity_error = reference_velocity - velocity

    total_acceleration_world = np.array([0.0, 0.0, params.g], dtype=float)
    total_acceleration_world += config.position_gains() * position_error
    total_acceleration_world += config.velocity_gains() * velocity_error
    total_acceleration_world[2] += float(config.z_integral_gain) * float(z_error_integral)
    total_acceleration_world = _limit_tilt(total_acceleration_world, config.max_tilt_deg)

    desired_rotation = _desired_rotation(total_acceleration_world, reference_rotation)
    attitude_error_matrix = 0.5 * (desired_rotation.T @ rotation - rotation.T @ desired_rotation)
    attitude_error = vee_map(attitude_error_matrix)

    body_moments = -config.attitude_gains() * attitude_error
    body_moments -= config.angular_rate_gains() * angular_velocity
    body_moments += np.cross(angular_velocity, params.J @ angular_velocity)

    body_z_world = rotation[:, 2]
    collective_newton = float(params.mass * np.dot(total_acceleration_world, body_z_world))

    control_raw = np.concatenate(([collective_newton], body_moments))
    control_used = np.clip(control_raw, scaling.control_lower_bounds(), scaling.control_upper_bounds())
    return control_raw, control_used
