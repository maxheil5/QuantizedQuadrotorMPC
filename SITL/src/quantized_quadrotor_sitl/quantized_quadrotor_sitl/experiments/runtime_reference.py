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


def _yaw_rotation_matrix(yaw_angle: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw_angle))
    sin_yaw = float(np.sin(yaw_angle))
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _initial_heading_angle(initial_state: np.ndarray) -> float:
    rotation = np.asarray(initial_state[6:15], dtype=float).reshape(3, 3, order="F")
    heading = rotation[:, 0].copy()
    heading[2] = 0.0
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-9:
        return 0.0
    heading /= heading_norm
    return float(np.arctan2(heading[1], heading[0]))


def _fill_velocity_reference(reference: np.ndarray, position: np.ndarray, sim_timestep: float) -> None:
    if position.shape[1] <= 1:
        reference[3:6, :] = 0.0
        return
    edge_order = 2 if position.shape[1] > 2 else 1
    reference[3:6, :] = np.gradient(position, sim_timestep, axis=1, edge_order=edge_order)


def _fill_heading_reference(reference: np.ndarray, heading_angles: np.ndarray) -> None:
    headings = np.asarray(heading_angles, dtype=float).reshape(-1)
    for idx, heading_angle in enumerate(headings):
        reference[6:15, idx] = _yaw_rotation_matrix(float(heading_angle)).reshape(-1, order="F")


def _windowed_segment_profile(sample_times: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (sample_times >= t0) & (sample_times < t1)
    if not np.any(mask):
        empty = np.zeros(0, dtype=float)
        return mask, empty, empty
    tau = (sample_times[mask] - t0) / max(t1 - t0, 1.0e-9)
    envelope = np.sin(np.pi * tau) ** 2
    return mask, tau, envelope


def _rotate_planar_offsets(x_offset: np.ndarray, y_offset: np.ndarray, rotation_angle: float) -> tuple[np.ndarray, np.ndarray]:
    cos_angle = float(np.cos(rotation_angle))
    sin_angle = float(np.sin(rotation_angle))
    x_vec = np.asarray(x_offset, dtype=float)
    y_vec = np.asarray(y_offset, dtype=float)
    return cos_angle * x_vec - sin_angle * y_vec, sin_angle * x_vec + cos_angle * y_vec


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


def build_sitl_identification_reference_v2(
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
    heading_profile = np.full(sample_count, _initial_heading_angle(state0), dtype=float)

    _constant_segment(position, sample_times, 0.0, 2.0, np.array([x0, y0, z0], dtype=float))
    _smooth_segment(position, sample_times, 2.0, 6.0, np.array([x0, y0, z0], dtype=float), hover_position)
    _constant_segment(position, sample_times, 6.0, 8.0, hover_position)

    x_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float)))
    y_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float)))
    z_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float)))

    vertical_mask = (sample_times >= 8.0) & (sample_times < 12.0)
    if np.any(vertical_mask):
        vertical_time = sample_times[vertical_mask] - 8.0
        position[0, vertical_mask] = x0
        position[1, vertical_mask] = y0
        position[2, vertical_mask] = (
            z0
            + 0.75
            + z_sign * 0.10 * np.sin(2.0 * np.pi * vertical_time / 4.0)
            + 0.04 * np.sin(6.0 * np.pi * vertical_time / 4.0)
        )

    x_sweep_mask = (sample_times >= 12.0) & (sample_times < 16.0)
    if np.any(x_sweep_mask):
        x_time = sample_times[x_sweep_mask] - 12.0
        position[0, x_sweep_mask] = x0 + x_sign * 0.28 * np.sin(2.0 * np.pi * x_time / 4.0)
        position[1, x_sweep_mask] = y0 + y_sign * 0.06 * np.sin(4.0 * np.pi * x_time / 4.0)
        position[2, x_sweep_mask] = z0 + 0.75 + 0.04 * np.sin(4.0 * np.pi * x_time / 4.0)

    y_sweep_mask = (sample_times >= 16.0) & (sample_times < 20.0)
    if np.any(y_sweep_mask):
        y_time = sample_times[y_sweep_mask] - 16.0
        position[0, y_sweep_mask] = x0 + x_sign * 0.08 * np.sin(4.0 * np.pi * y_time / 4.0)
        position[1, y_sweep_mask] = y0 + y_sign * 0.24 * np.sin(2.0 * np.pi * y_time / 4.0)
        position[2, y_sweep_mask] = z0 + 0.75 + 0.05 * np.sin(2.0 * np.pi * y_time / 4.0)

    figure_mask = sample_times >= 20.0
    if np.any(figure_mask):
        figure_time = sample_times[figure_mask] - 20.0
        position[0, figure_mask] = x0 + x_sign * 0.30 * np.sin(2.0 * np.pi * figure_time / 4.0)
        position[1, figure_mask] = y0 + y_sign * 0.22 * np.sin(4.0 * np.pi * figure_time / 4.0)
        position[2, figure_mask] = z0 + 0.75 + 0.06 * np.sin(4.0 * np.pi * figure_time / 4.0)
        heading_profile[figure_mask] += np.deg2rad(12.0) * np.sin(2.0 * np.pi * figure_time / 4.0)

    reference[0:3, :] = position
    _fill_velocity_reference(reference, position, sim_timestep)
    _fill_heading_reference(reference, heading_profile)
    return reference


def build_sitl_identification_reference_v3(
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
    initial_heading = _initial_heading_angle(state0)
    hover_position = np.array([x0, y0, z0 + 0.75], dtype=float)
    position = np.repeat(state0[0:3, None], sample_count, axis=1)
    heading_profile = np.full(sample_count, initial_heading, dtype=float)

    _constant_segment(position, sample_times, 0.0, 2.0, np.array([x0, y0, z0], dtype=float))
    _smooth_segment(position, sample_times, 2.0, 5.5, np.array([x0, y0, z0], dtype=float), hover_position)
    _constant_segment(position, sample_times, 5.5, 7.0, hover_position)

    rotation_angle = float(rng.uniform(-np.pi / 3.0, np.pi / 3.0))
    phase_xy = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_z = float(rng.uniform(0.0, 2.0 * np.pi))
    heading_phase = float(rng.uniform(0.0, 2.0 * np.pi))

    mask, tau, envelope = _windowed_segment_profile(sample_times, 7.0, 11.0)
    if np.any(mask):
        x_offset_local = envelope * (
            0.12 * np.sin(6.0 * np.pi * tau + phase_xy) + 0.05 * np.sin(12.0 * np.pi * tau + 0.4)
        )
        y_offset_local = envelope * (
            0.10 * np.sin(8.0 * np.pi * tau + 0.5 * phase_xy) + 0.03 * np.sin(16.0 * np.pi * tau + 0.2)
        )
        x_offset_world, y_offset_world = _rotate_planar_offsets(x_offset_local, y_offset_local, rotation_angle)
        position[0, mask] = x0 + x_offset_world
        position[1, mask] = y0 + y_offset_world
        position[2, mask] = hover_position[2] + envelope * (
            0.14 * np.sin(2.0 * np.pi * tau + phase_z) + 0.05 * np.sin(6.0 * np.pi * tau + 0.1)
        )
        heading_profile[mask] = initial_heading + 0.12 * envelope * np.sin(4.0 * np.pi * tau + heading_phase)

    mask, tau, envelope = _windowed_segment_profile(sample_times, 11.0, 15.0)
    if np.any(mask):
        x_offset_local = envelope * (
            0.34 * np.sin(3.0 * np.pi * tau + phase_xy) + 0.08 * np.sin(9.0 * np.pi * tau)
        )
        y_offset_local = envelope * 0.18 * np.sin(6.0 * np.pi * tau + 0.4 + 0.5 * phase_xy)
        x_offset_world, y_offset_world = _rotate_planar_offsets(x_offset_local, y_offset_local, rotation_angle)
        position[0, mask] = x0 + x_offset_world
        position[1, mask] = y0 + y_offset_world
        position[2, mask] = hover_position[2] + envelope * 0.06 * np.sin(4.0 * np.pi * tau + phase_z)
        heading_profile[mask] = initial_heading + rotation_angle + 0.18 * envelope * np.sin(4.0 * np.pi * tau + heading_phase)

    mask, tau, envelope = _windowed_segment_profile(sample_times, 15.0, 19.0)
    if np.any(mask):
        x_offset_local = envelope * 0.20 * np.sin(6.0 * np.pi * tau + 0.3 + phase_xy)
        y_offset_local = envelope * (
            0.32 * np.sin(3.0 * np.pi * tau + 0.2 + 0.5 * phase_xy) + 0.08 * np.sin(9.0 * np.pi * tau + 0.1)
        )
        x_offset_world, y_offset_world = _rotate_planar_offsets(x_offset_local, y_offset_local, rotation_angle)
        position[0, mask] = x0 + x_offset_world
        position[1, mask] = y0 + y_offset_world
        position[2, mask] = hover_position[2] + envelope * 0.07 * np.sin(4.0 * np.pi * tau + phase_z + 0.5)
        heading_profile[mask] = initial_heading - rotation_angle + 0.22 * envelope * np.sin(4.0 * np.pi * tau + heading_phase + 0.6)

    mask, tau, envelope = _windowed_segment_profile(sample_times, 19.0, 24.0)
    if np.any(mask):
        x_offset_local = envelope * (
            0.30 * np.sin(4.0 * np.pi * tau + phase_xy) + 0.08 * np.sin(12.0 * np.pi * tau + 0.3)
        )
        y_offset_local = envelope * 0.24 * np.sin(8.0 * np.pi * tau + 0.5 * phase_xy)
        x_offset_world, y_offset_world = _rotate_planar_offsets(x_offset_local, y_offset_local, rotation_angle)
        position[0, mask] = x0 + x_offset_world
        position[1, mask] = y0 + y_offset_world
        position[2, mask] = hover_position[2] + envelope * (
            0.08 * np.sin(4.0 * np.pi * tau + phase_z) + 0.03 * np.sin(12.0 * np.pi * tau)
        )
        heading_profile[mask] = initial_heading + rotation_angle + 0.30 * envelope * np.sin(4.0 * np.pi * tau + heading_phase)

    reference[0:3, :] = position
    _fill_velocity_reference(reference, position, sim_timestep)
    _fill_heading_reference(reference, heading_profile)
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
    if reference_mode == "sitl_identification_v2":
        return build_sitl_identification_reference_v2(initial_state, reference_duration_s, sim_timestep, rng)
    if reference_mode == "sitl_identification_v3":
        return build_sitl_identification_reference_v3(initial_state, reference_duration_s, sim_timestep, rng)
    if reference_mode == "hover_step":
        return build_hover_step_reference(initial_state, reference_duration_s, sim_timestep)
    if reference_mode == "paper_random":
        return build_paper_random_reference(initial_state, reference_duration_s, sim_timestep, rng)
    raise ValueError(f"unsupported runtime reference mode: {reference_mode}")
