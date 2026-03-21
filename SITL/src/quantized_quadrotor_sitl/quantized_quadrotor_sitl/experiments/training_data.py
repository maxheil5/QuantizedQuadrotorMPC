from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..dynamics.params import get_params
from ..dynamics.srb import dynamics_srb


FloatArray = NDArray[np.float64]


def _control_covariance(flag: str) -> FloatArray:
    if flag == "train":
        return np.diag([10.0, 10.0, 10.0, 10.0])
    if flag == "val":
        return np.diag([20.0, 20.0, 20.0, 20.0])
    if flag == "mpc":
        return np.diag([30.0, 30.0, 30.0, 30.0])
    raise ValueError(f"unsupported trajectory flag: {flag}")


def _simulate_constant_controls(
    initial_state: FloatArray,
    controls: FloatArray,
    t_traj: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    params = get_params()

    x_history: list[FloatArray] = []
    x1_history: list[FloatArray] = []
    x2_history: list[FloatArray] = []
    u_history: list[FloatArray] = []
    u1_history: list[FloatArray] = []
    u2_history: list[FloatArray] = []

    for control in controls:
        solution = solve_ivp(
            lambda t, state: dynamics_srb(t, state, control, params),
            (float(t_traj[0]), float(t_traj[-1])),
            initial_state,
            method="RK45",
            t_eval=t_traj,
            rtol=1.0e-3,
            atol=1.0e-6,
        )
        trajectory = solution.y
        x_history.append(trajectory)
        x1_history.append(trajectory[:, :-1])
        x2_history.append(trajectory[:, 1:])
        u_history.append(np.repeat(control[:, None], t_traj.size, axis=1))
        u1_history.append(np.repeat(control[:, None], t_traj.size - 1, axis=1))
        u2_history.append(np.repeat(control[:, None], t_traj.size - 1, axis=1))

    return (
        np.hstack(x_history),
        np.hstack(u_history),
        np.hstack(x1_history),
        np.hstack(x2_history),
        np.hstack(u1_history),
        np.hstack(u2_history),
    )


def get_random_trajectories(
    initial_state: FloatArray,
    n_control: int,
    t_traj: FloatArray,
    flag: str,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    params = get_params()
    net_weight = params.mass * params.g

    controls = rng.multivariate_normal(
        mean=np.zeros(4, dtype=float),
        cov=_control_covariance(flag),
        size=n_control,
    )
    controls[:, 0] += net_weight
    return _simulate_constant_controls(initial_state, controls, t_traj)


def get_hover_local_trajectories(
    initial_state: FloatArray,
    n_control: int,
    t_traj: FloatArray,
    rng: np.random.Generator,
    collective_std_newton: float,
    collective_band_newton: float,
    body_moment_std_nm: FloatArray,
    body_moment_band_nm: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    params = get_params()
    hover_thrust = params.mass * params.g

    body_moment_std = np.asarray(body_moment_std_nm, dtype=float).reshape(3)
    body_moment_band = np.asarray(body_moment_band_nm, dtype=float).reshape(3)

    controls = np.zeros((n_control, 4), dtype=float)
    controls[:, 0] = hover_thrust + rng.normal(0.0, collective_std_newton, size=n_control)
    controls[:, 0] = np.clip(
        controls[:, 0],
        max(0.0, hover_thrust - collective_band_newton),
        hover_thrust + collective_band_newton,
    )
    controls[:, 1:] = rng.normal(0.0, body_moment_std, size=(n_control, 3))
    controls[:, 1:] = np.clip(controls[:, 1:], -body_moment_band, body_moment_band)

    return _simulate_constant_controls(initial_state, controls, t_traj)
