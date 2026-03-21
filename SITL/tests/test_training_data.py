from __future__ import annotations

import numpy as np

from quantized_quadrotor_sitl.core.config import hover_local_v1_profile, initial_state
from quantized_quadrotor_sitl.dynamics.params import get_params
from quantized_quadrotor_sitl.experiments.training_data import get_hover_local_trajectories


def test_hover_local_trajectories_stay_within_local_control_band():
    config = hover_local_v1_profile()
    rng = np.random.default_rng(config.random_seed)
    t_traj = np.arange(0.0, config.train_traj_duration + config.dt, config.dt)
    _, u_history, _, _, u1_history, _ = get_hover_local_trajectories(
        initial_state(),
        8,
        t_traj,
        rng,
        collective_std_newton=config.collective_std_newton,
        collective_band_newton=config.collective_band_newton,
        body_moment_std_nm=config.body_moment_std(),
        body_moment_band_nm=config.body_moment_bounds(),
    )

    params = get_params()
    hover_thrust = params.mass * params.g
    collective_min = max(0.0, hover_thrust - config.collective_band_newton)
    collective_max = hover_thrust + config.collective_band_newton

    assert u_history.shape[0] == 4
    assert u1_history.shape[0] == 4
    assert np.min(u_history[0, :]) >= collective_min - 1.0e-9
    assert np.max(u_history[0, :]) <= collective_max + 1.0e-9
    assert np.max(np.abs(u_history[1, :])) <= config.body_moment_band_nm[0] + 1.0e-9
    assert np.max(np.abs(u_history[2, :])) <= config.body_moment_band_nm[1] + 1.0e-9
    assert np.max(np.abs(u_history[3, :])) <= config.body_moment_band_nm[2] + 1.0e-9
