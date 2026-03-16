from __future__ import annotations

import numpy as np

from .training_data import get_random_trajectories


def _reference_sample_count(reference_duration_s: float, sim_timestep: float) -> int:
    sample_count = int(np.floor(reference_duration_s / sim_timestep + 1.0e-9))
    return max(sample_count, 1)


def build_hover_step_reference(
    initial_state: np.ndarray,
    reference_duration_s: float,
    sim_timestep: float,
) -> np.ndarray:
    state0 = np.asarray(initial_state, dtype=float).reshape(18)
    sample_count = _reference_sample_count(reference_duration_s, sim_timestep)
    sample_times = np.arange(sample_count, dtype=float) * sim_timestep

    reference = np.repeat(state0[:, None], sample_count, axis=1)
    reference[3:6, :] = 0.0
    reference[15:18, :] = 0.0

    position = np.repeat(state0[0:3, None], sample_count, axis=1)
    x0, y0, z0 = state0[0:3]

    for idx, t_val in enumerate(sample_times):
        if t_val < 1.0:
            position[:, idx] = [x0, y0, z0]
        elif t_val < 3.0:
            alpha = (t_val - 1.0) / 2.0
            position[:, idx] = [x0, y0, z0 + 0.75 * alpha]
        elif t_val < 5.0:
            position[:, idx] = [x0, y0, z0 + 0.75]
        elif t_val < 7.0:
            alpha = (t_val - 5.0) / 2.0
            position[:, idx] = [x0 + 0.50 * alpha, y0, z0 + 0.75]
        else:
            position[:, idx] = [x0 + 0.50, y0, z0 + 0.75]

    reference[0:3, :] = position
    return reference


def build_paper_random_reference(
    initial_state: np.ndarray,
    reference_duration_s: float,
    sim_timestep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_count = _reference_sample_count(reference_duration_s, sim_timestep)
    sample_times = np.arange(sample_count + 1, dtype=float) * sim_timestep
    x_ref, _, _, _, _, _ = get_random_trajectories(initial_state, 1, sample_times, "mpc", rng)
    return x_ref[:, 1 : sample_count + 1]


def build_runtime_reference(
    initial_state: np.ndarray,
    reference_mode: str,
    reference_duration_s: float,
    sim_timestep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if reference_mode == "hover_step":
        return build_hover_step_reference(initial_state, reference_duration_s, sim_timestep)
    if reference_mode == "paper_random":
        return build_paper_random_reference(initial_state, reference_duration_s, sim_timestep, rng)
    raise ValueError(f"unsupported runtime reference mode: {reference_mode}")
