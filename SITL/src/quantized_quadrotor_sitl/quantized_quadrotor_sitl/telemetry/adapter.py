from __future__ import annotations

import numpy as np


NED_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0]).astype(float)


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(4)
    return np.array(
        [
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def vehicle_odometry_to_state18(
    position_ned: np.ndarray,
    velocity_ned: np.ndarray,
    quaternion_wxyz_ned_frd: np.ndarray,
    angular_velocity_frd: np.ndarray,
) -> np.ndarray:
    position_enu = NED_TO_ENU @ np.asarray(position_ned, dtype=float).reshape(3)
    velocity_enu = NED_TO_ENU @ np.asarray(velocity_ned, dtype=float).reshape(3)
    rotation_ned_frd = quaternion_wxyz_to_rotation(quaternion_wxyz_ned_frd)
    rotation_enu_flu = NED_TO_ENU @ rotation_ned_frd @ FRD_TO_FLU
    angular_velocity_flu = FRD_TO_FLU @ np.asarray(angular_velocity_frd, dtype=float).reshape(3)
    return np.concatenate(
        [
            position_enu,
            velocity_enu,
            rotation_enu_flu.reshape(-1, order="F"),
            angular_velocity_flu,
        ]
    )


def physical_control_to_px4_wrench(
    control_flu: np.ndarray,
    max_collective_thrust_newton: float,
    max_body_torque_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    control = np.asarray(control_flu, dtype=float).reshape(4)
    collective = np.clip(control[0] / max_collective_thrust_newton, 0.0, 1.0)
    moments_frd = FRD_TO_FLU @ control[1:4]
    normalized_moments = np.divide(moments_frd, max_body_torque_nm, out=np.zeros(3), where=max_body_torque_nm != 0.0)
    normalized_moments = np.clip(normalized_moments, -1.0, 1.0)
    thrust_body = np.array([0.0, 0.0, collective], dtype=float)
    return thrust_body, normalized_moments
