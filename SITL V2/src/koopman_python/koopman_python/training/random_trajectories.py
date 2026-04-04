"""Random trajectory generation ported from MATLAB/training/get_rnd_trajectories.m."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from koopman_python.dynamics.params import get_params
from koopman_python.dynamics.srb import dynamics_srb


TrajectoryMode = Literal["train", "val", "mpc"]


@dataclass(frozen=True)
class RandomTrajectoryBatch:
    """Container matching the MATLAB training-data outputs."""

    X: np.ndarray
    U: np.ndarray
    X1: np.ndarray
    X2: np.ndarray
    U1: np.ndarray
    U2: np.ndarray
    U_rnd: np.ndarray
    t_traj: np.ndarray
    mode: TrajectoryMode


def control_covariance(mode: TrajectoryMode) -> np.ndarray:
    """Return the MATLAB control covariance for the requested mode."""

    if mode == "train":
        variance = 10.0
    elif mode == "val":
        variance = 20.0
    elif mode == "mpc":
        variance = 30.0
    else:
        raise ValueError("mode must be one of: 'train', 'val', 'mpc'.")
    return np.diag([variance, variance, variance, variance]).astype(float)


def sample_random_controls(
    n_control: int,
    mode: TrajectoryMode,
    params: dict[str, object] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample constant control inputs following the MATLAB distributions."""

    if n_control <= 0:
        raise ValueError("n_control must be positive.")
    params = get_params() if params is None else params
    generator = np.random.default_rng() if rng is None else rng
    controls = generator.multivariate_normal(
        mean=np.zeros(4, dtype=float),
        cov=control_covariance(mode),
        size=n_control,
    )
    controls = np.asarray(controls, dtype=float).reshape(n_control, 4)
    net_weight = float(params["mass"]) * float(params.get("gravity", params.get("g", 9.81)))
    controls[:, 0] = controls[:, 0] + net_weight
    return controls


def rk4_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    params: dict[str, object],
    time_s: float = 0.0,
) -> np.ndarray:
    """Advance one SRB step with a constant control input."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    x = np.asarray(state, dtype=float).reshape(18)
    u = np.asarray(control, dtype=float).reshape(4)
    k1 = dynamics_srb(time_s, x, u, params)
    k2 = dynamics_srb(time_s + 0.5 * dt, x + 0.5 * dt * k1, u, params)
    k3 = dynamics_srb(time_s + 0.5 * dt, x + 0.5 * dt * k2, u, params)
    k4 = dynamics_srb(time_s + dt, x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_constant_control_trajectory(
    initial_state: np.ndarray,
    control: np.ndarray,
    t_traj: np.ndarray,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Simulate a constant-control SRB trajectory on the provided time grid."""

    params = get_params() if params is None else params
    time_grid = np.asarray(t_traj, dtype=float).reshape(-1)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("t_traj must be a 1D array with at least two samples.")
    if not np.all(np.diff(time_grid) > 0.0):
        raise ValueError("t_traj must be strictly increasing.")

    states = np.zeros((18, time_grid.size), dtype=float)
    states[:, 0] = np.asarray(initial_state, dtype=float).reshape(18)
    for i in range(time_grid.size - 1):
        dt = float(time_grid[i + 1] - time_grid[i])
        states[:, i + 1] = rk4_step(
            state=states[:, i],
            control=control,
            dt=dt,
            params=params,
            time_s=float(time_grid[i]),
        )
    return states


def get_random_trajectories(
    initial_state: np.ndarray,
    n_control: int,
    t_traj: np.ndarray,
    show_plot: bool = False,
    mode: TrajectoryMode = "train",
    params: dict[str, object] | None = None,
    rng: np.random.Generator | None = None,
) -> RandomTrajectoryBatch:
    """Port of MATLAB get_rnd_trajectories.m without the plotting path.

    Returns the same snapshot matrices used by the EDMD MATLAB code:
    - X, U: all full trajectories stacked across controls
    - X1, X2, U1, U2: one-step snapshot pairs
    """

    del show_plot  # plotting is intentionally omitted in the first Python port

    params = get_params() if params is None else params
    time_grid = np.asarray(t_traj, dtype=float).reshape(-1)
    random_controls = sample_random_controls(
        n_control=n_control,
        mode=mode,
        params=params,
        rng=rng,
    )

    trajectory_blocks = []
    x1_blocks = []
    x2_blocks = []
    for i in range(n_control):
        control = random_controls[i, :]
        states = simulate_constant_control_trajectory(
            initial_state=initial_state,
            control=control,
            t_traj=time_grid,
            params=params,
        )
        trajectory_blocks.append(states)
        x1_blocks.append(states[:, :-1])
        x2_blocks.append(states[:, 1:])

    X = np.column_stack(trajectory_blocks)
    X1 = np.column_stack(x1_blocks)
    X2 = np.column_stack(x2_blocks)

    repeated_controls = [
        np.repeat(random_controls[i, :].reshape(4, 1), time_grid.size, axis=1)
        for i in range(n_control)
    ]
    repeated_snapshot_controls = [
        np.repeat(random_controls[i, :].reshape(4, 1), time_grid.size - 1, axis=1)
        for i in range(n_control)
    ]

    U = np.column_stack(repeated_controls)
    U1 = np.column_stack(repeated_snapshot_controls)
    U2 = np.column_stack(repeated_snapshot_controls)

    return RandomTrajectoryBatch(
        X=X,
        U=U,
        X1=X1,
        X2=X2,
        U1=U1,
        U2=U2,
        U_rnd=random_controls.T,
        t_traj=time_grid,
        mode=mode,
    )
