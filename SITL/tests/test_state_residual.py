from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.experiments.runtime_reference import build_takeoff_hold_reference
from quantized_quadrotor_sitl.utils.state import (
    HOVER_LOCAL_STATE_COORDINATES_ROTATED,
    hover_local_translation_rotated,
    state18_from_hover_local_residual,
    state18_history_to_hover_local_residual,
    state18_to_hover_local_residual,
    takeoff_hold_trim_state18,
)


def _rotation_z(theta_rad: float) -> np.ndarray:
    c_val = np.cos(theta_rad)
    s_val = np.sin(theta_rad)
    return np.array(
        [
            [c_val, -s_val, 0.0],
            [s_val, c_val, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _state18(position: np.ndarray, velocity: np.ndarray, rotation: np.ndarray, wb: np.ndarray) -> np.ndarray:
    return np.concatenate([position, velocity, rotation.reshape(-1, order="F"), wb])


def test_hover_local_residual_round_trip_matches_original_state():
    trim_state = _state18(
        position=np.array([1.0, -2.0, 0.75], dtype=float),
        velocity=np.zeros(3, dtype=float),
        rotation=_rotation_z(np.pi / 6.0),
        wb=np.zeros(3, dtype=float),
    )
    raw_state = _state18(
        position=np.array([1.2, -1.8, 1.05], dtype=float),
        velocity=np.array([0.1, -0.2, 0.05], dtype=float),
        rotation=_rotation_z(np.pi / 3.0),
        wb=np.array([0.02, -0.03, 0.04], dtype=float),
    )

    residual_state = state18_to_hover_local_residual(raw_state, trim_state)
    reconstructed_state = state18_from_hover_local_residual(residual_state, trim_state)

    npt.assert_allclose(reconstructed_state, raw_state, atol=1.0e-9)


def test_hover_local_residual_rotation_uses_trim_transpose_times_state():
    trim_rotation = _rotation_z(np.pi / 6.0)
    raw_rotation = _rotation_z(np.pi / 3.0)
    trim_state = _state18(
        position=np.zeros(3, dtype=float),
        velocity=np.zeros(3, dtype=float),
        rotation=trim_rotation,
        wb=np.zeros(3, dtype=float),
    )
    raw_state = _state18(
        position=np.zeros(3, dtype=float),
        velocity=np.zeros(3, dtype=float),
        rotation=raw_rotation,
        wb=np.zeros(3, dtype=float),
    )

    residual_state = state18_to_hover_local_residual(raw_state, trim_state)
    expected_rotation_delta = trim_rotation.T @ raw_rotation

    npt.assert_allclose(residual_state[6:15].reshape(3, 3, order="F"), expected_rotation_delta, atol=1.0e-9)


def test_takeoff_hold_reference_transforms_to_zero_residual_at_hover_trim():
    initial_state = _state18(
        position=np.array([0.5, -0.25, 0.1], dtype=float),
        velocity=np.array([0.2, 0.1, -0.1], dtype=float),
        rotation=_rotation_z(np.pi / 8.0),
        wb=np.array([0.03, -0.02, 0.01], dtype=float),
    )
    trim_state = takeoff_hold_trim_state18(initial_state)
    reference = build_takeoff_hold_reference(initial_state, reference_duration_s=10.0, sim_timestep=1.0e-3)

    residual_reference = state18_history_to_hover_local_residual(reference, trim_state)
    final_column = residual_reference[:, -1]

    npt.assert_allclose(final_column[0:3], np.zeros(3, dtype=float), atol=1.0e-9)
    npt.assert_allclose(final_column[3:6], np.zeros(3, dtype=float), atol=1.0e-9)
    npt.assert_allclose(final_column[6:15].reshape(3, 3, order="F"), np.eye(3, dtype=float), atol=1.0e-9)
    npt.assert_allclose(final_column[15:18], np.zeros(3, dtype=float), atol=1.0e-9)


def test_hover_local_residual_can_rotate_translation_into_trim_frame():
    trim_rotation = _rotation_z(np.pi / 2.0)
    trim_state = _state18(
        position=np.array([1.0, 2.0, 0.75], dtype=float),
        velocity=np.zeros(3, dtype=float),
        rotation=trim_rotation,
        wb=np.zeros(3, dtype=float),
    )
    raw_state = _state18(
        position=np.array([2.0, 2.0, 1.25], dtype=float),
        velocity=np.array([1.0, 0.0, 0.0], dtype=float),
        rotation=trim_rotation,
        wb=np.zeros(3, dtype=float),
    )

    residual_state = state18_to_hover_local_residual(raw_state, trim_state, rotate_translation=True)

    npt.assert_allclose(residual_state[0:3], np.array([0.0, -1.0, 0.5], dtype=float), atol=1.0e-9)
    npt.assert_allclose(residual_state[3:6], np.array([0.0, -1.0, 0.0], dtype=float), atol=1.0e-9)


def test_hover_local_translation_rotation_helper_detects_v2_coordinates():
    assert hover_local_translation_rotated(HOVER_LOCAL_STATE_COORDINATES_ROTATED) is True
    assert hover_local_translation_rotated("takeoff_hold_hover_local") is False
