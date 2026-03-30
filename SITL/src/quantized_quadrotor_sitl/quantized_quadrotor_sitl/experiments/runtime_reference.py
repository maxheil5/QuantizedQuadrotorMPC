from __future__ import annotations

import numpy as np

from .training_data import get_random_trajectories


def _reference_sample_count(reference_duration_s: float, sim_timestep: float) -> int:
    sample_count = int(np.floor(reference_duration_s / sim_timestep + 1.0e-9))
    return max(sample_count, 1)


def _smooth_blend(start: np.ndarray, end: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    clamped = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)
    eased = 0.5 - 0.5 * np.cos(np.pi * clamped)
    start_vec = np.asarray(start, dtype=float).reshape(3, 1)
    end_vec = np.asarray(end, dtype=float).reshape(3, 1)
    return start_vec + (end_vec - start_vec) * eased.reshape(1, -1)


def _smooth_segment(position: np.ndarray, sample_times: np.ndarray, t0: float, t1: float, start: np.ndarray, end: np.ndarray) -> None:
    mask = (sample_times >= t0) & (sample_times < t1)
    if not np.any(mask):
        return
    alpha = (sample_times[mask] - t0) / max(t1 - t0, 1.0e-9)
    position[:, mask] = _smooth_blend(start, end, alpha)


def _constant_segment(position: np.ndarray, sample_times: np.ndarray, t0: float, t1: float, target: np.ndarray) -> None:
    mask = (sample_times >= t0) & (sample_times < t1)
    if np.any(mask):
        position[:, mask] = np.asarray(target, dtype=float).reshape(3, 1)


def build_sitl_identification_reference(
    initial_state: np.ndarray,
    reference_duration_s: float,
    sim_timestep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state0 = np.asarray(initial_state, dtype=float).reshape(18)
    sample_count = _reference_sample_count(reference_duration_s, sim_timestep)
    sample_times = np.arange(sample_count, dtype=float) * sim_timestep

    reference = np.repeat(state0[:, None], sample_count, axis=1)
    reference[3:6, :] = 0.0
    reference[15:18, :] = 0.0

    x0, y0, z0 = state0[0:3]
    hover_position = np.array([x0, y0, z0 + 0.75], dtype=float)
    position = np.repeat(state0[0:3, None], sample_count, axis=1)

    _constant_segment(position, sample_times, 0.0, 2.0, np.array([x0, y0, z0], dtype=float))
    _smooth_segment(position, sample_times, 2.0, 6.0, np.array([x0, y0, z0], dtype=float), hover_position)
    _constant_segment(position, sample_times, 6.0, 8.0, hover_position)

    z_targets = z0 + np.array([0.65, 0.85, 0.60, 0.80], dtype=float)
    z_targets = z_targets[rng.permutation(z_targets.size)]
    vertical_times = np.array([8.0, 8.8, 9.6, 10.4, 11.2, 12.0], dtype=float)
    vertical_points = [
        hover_position,
        np.array([x0, y0, z_targets[0]], dtype=float),
        np.array([x0, y0, z_targets[1]], dtype=float),
        np.array([x0, y0, z_targets[2]], dtype=float),
        np.array([x0, y0, z_targets[3]], dtype=float),
        hover_position,
    ]
    for idx in range(vertical_times.size - 1):
        _smooth_segment(
            position,
            sample_times,
            float(vertical_times[idx]),
            float(vertical_times[idx + 1]),
            vertical_points[idx],
            vertical_points[idx + 1],
        )

    x_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float)))
    x_times = np.array([12.0, 13.0, 14.0, 15.0, 16.0], dtype=float)
    x_points = [
        hover_position,
        np.array([x0 + 0.20 * x_sign, y0, z0 + 0.75], dtype=float),
        np.array([x0 - 0.20 * x_sign, y0, z0 + 0.75], dtype=float),
        np.array([x0 + 0.10 * x_sign, y0, z0 + 0.75], dtype=float),
        hover_position,
    ]
    for idx in range(x_times.size - 1):
        _smooth_segment(
            position,
            sample_times,
            float(x_times[idx]),
            float(x_times[idx + 1]),
            x_points[idx],
            x_points[idx + 1],
        )

    y_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float)))
    y_times = np.array([16.0, 17.0, 18.0, 19.0, 20.0], dtype=float)
    y_points = [
        hover_position,
        np.array([x0, y0 + 0.20 * y_sign, z0 + 0.75], dtype=float),
        np.array([x0, y0 - 0.20 * y_sign, z0 + 0.75], dtype=float),
        np.array([x0, y0 + 0.10 * y_sign, z0 + 0.75], dtype=float),
        hover_position,
    ]
    for idx in range(y_times.size - 1):
        _smooth_segment(
            position,
            sample_times,
            float(y_times[idx]),
            float(y_times[idx + 1]),
            y_points[idx],
            y_points[idx + 1],
        )

    figure_mask = sample_times >= 20.0
    if np.any(figure_mask):
        figure_time = sample_times[figure_mask] - 20.0
        phase = 2.0 * np.pi * figure_time / 4.0
        figure_position = np.vstack(
            [
                x0 + 0.15 * x_sign * np.sin(phase),
                y0 + 0.10 * y_sign * np.sin(2.0 * phase),
                z0 + 0.75 + 0.05 * np.sin(phase),
            ]
        )
        position[:, figure_mask] = figure_position

    reference[0:3, :] = position
    return reference


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


def build_takeoff_hold_reference(
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
        else:
            position[:, idx] = [x0, y0, z0 + 0.75]

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
    if reference_mode == "takeoff_hold":
        return build_takeoff_hold_reference(initial_state, reference_duration_s, sim_timestep)
    if reference_mode == "sitl_identification_v1":
        return build_sitl_identification_reference(initial_state, reference_duration_s, sim_timestep, rng)
    if reference_mode == "hover_step":
        return build_hover_step_reference(initial_state, reference_duration_s, sim_timestep)
    if reference_mode == "paper_random":
        return build_paper_random_reference(initial_state, reference_duration_s, sim_timestep, rng)
    raise ValueError(f"unsupported runtime reference mode: {reference_mode}")
