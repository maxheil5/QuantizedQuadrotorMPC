"""Python port of MATLAB/dynamics/dynamics_SRB.m."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def hat_map(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix used in SO(3) kinematics."""

    wx, wy, wz = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
        dtype=float,
    )


def unpack_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the MATLAB state layout X = [x dx R wb]'."""

    flat = np.asarray(state, dtype=float).reshape(18)
    x = flat[0:3]
    dx = flat[3:6]
    rotation = flat[6:15].reshape(3, 3)
    wb = flat[15:18]
    return x, dx, rotation, wb


def pack_state(
    position: np.ndarray,
    velocity: np.ndarray,
    rotation: np.ndarray,
    angular_velocity: np.ndarray,
) -> np.ndarray:
    """Encode the MATLAB state layout X = [x dx R wb]'."""

    return np.concatenate(
        (
            np.asarray(position, dtype=float).reshape(3),
            np.asarray(velocity, dtype=float).reshape(3),
            np.asarray(rotation, dtype=float).reshape(9),
            np.asarray(angular_velocity, dtype=float).reshape(3),
        )
    )


def unpack_control(control: np.ndarray) -> Tuple[float, np.ndarray]:
    """Decode the MATLAB control layout U = [Fb Mb]'."""

    flat = np.asarray(control, dtype=float).reshape(4)
    return float(flat[0]), flat[1:4]


def dynamics_srb(
    _time_s: float,
    state: np.ndarray,
    control: np.ndarray,
    params: Dict[str, object],
) -> np.ndarray:
    """Port of MATLAB/dynamics/dynamics_SRB.m.

    The frame and state conventions intentionally match the MATLAB code:
    X = [x, dx, R(:), wb], U = [Fb, Mbx, Mby, Mbz].
    """

    mass = float(params["mass"])
    inertia = np.asarray(params["J"], dtype=float).reshape(3, 3)
    gravity = float(params.get("gravity", params.get("g", 9.81)))
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)

    _position, velocity, rotation, angular_velocity = unpack_state(state)
    net_force_body, moment_body = unpack_control(control)

    acceleration = (net_force_body / mass) * e3 - gravity * rotation.T @ e3
    d_rotation = rotation @ hat_map(angular_velocity)
    d_angular_velocity = np.linalg.solve(
        inertia,
        moment_body - hat_map(angular_velocity) @ inertia @ angular_velocity,
    )

    return pack_state(
        position=velocity,
        velocity=acceleration,
        rotation=d_rotation,
        angular_velocity=d_angular_velocity,
    )
