from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.core.config import initial_state
from quantized_quadrotor_sitl.experiments.runtime_reference import (
    build_hover_step_reference,
    build_paper_random_reference,
    build_sitl_identification_reference,
    build_sitl_identification_reference_v2,
    build_sitl_identification_reference_v3,
    build_sitl_identification_reference_v4,
    build_sitl_identification_reference_v5,
    build_sitl_identification_reference_v6,
    build_takeoff_hold_reference,
    build_runtime_reference,
)


def test_takeoff_hold_reference_matches_expected_profile():
    state0 = initial_state()
    reference = build_takeoff_hold_reference(state0, reference_duration_s=10.0, sim_timestep=1.0e-3)

    assert reference.shape == (18, 10000)
    npt.assert_allclose(reference[0:3, 0], state0[0:3])
    npt.assert_allclose(reference[6:15, 0], state0[6:15])
    npt.assert_allclose(reference[3:6, 0], np.zeros(3))
    npt.assert_allclose(reference[15:18, 0], np.zeros(3))
    npt.assert_allclose(np.max(reference[2, :]), state0[2] + 0.75)
    npt.assert_allclose(reference[0, :], np.full(reference.shape[1], state0[0]))
    npt.assert_allclose(reference[1, :], np.full(reference.shape[1], state0[1]))


def test_hover_step_reference_matches_expected_profile():
    state0 = initial_state()
    reference = build_hover_step_reference(state0, reference_duration_s=10.0, sim_timestep=1.0e-3)

    assert reference.shape == (18, 10000)
    npt.assert_allclose(reference[0:3, 0], state0[0:3])
    npt.assert_allclose(reference[6:15, 0], state0[6:15])
    npt.assert_allclose(reference[3:6, 0], np.zeros(3))
    npt.assert_allclose(reference[15:18, 0], np.zeros(3))
    npt.assert_allclose(np.max(reference[2, :]), state0[2] + 0.75)
    npt.assert_allclose(np.max(reference[0, :]), state0[0] + 0.50)
    npt.assert_allclose(reference[1, :], np.full(reference.shape[1], state0[1]))


def test_runtime_reference_selector_preserves_mode_specific_builders():
    state0 = initial_state()

    takeoff_reference = build_runtime_reference(
        state0,
        reference_mode="takeoff_hold",
        reference_duration_s=10.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141444),
    )
    npt.assert_allclose(takeoff_reference, build_takeoff_hold_reference(state0, 10.0, 1.0e-3))

    hover_reference = build_runtime_reference(
        state0,
        reference_mode="hover_step",
        reference_duration_s=10.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141444),
    )
    npt.assert_allclose(hover_reference, build_hover_step_reference(state0, 10.0, 1.0e-3))

    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected_random_reference = build_runtime_reference(
        state0,
        reference_mode="paper_random",
        reference_duration_s=10.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected_random_reference = build_paper_random_reference(state0, 10.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected_random_reference, expected_random_reference)


def test_sitl_identification_reference_matches_bounds_and_translation_only_profile():
    state0 = initial_state()
    rng = np.random.default_rng(2141444)
    reference = build_sitl_identification_reference(state0, reference_duration_s=24.0, sim_timestep=1.0e-3, rng=rng)

    assert reference.shape == (18, 24000)
    npt.assert_allclose(reference[6:15, :], np.repeat(state0[6:15, None], reference.shape[1], axis=1))
    npt.assert_allclose(reference[15:18, :], np.zeros((3, reference.shape[1])))
    assert np.min(reference[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference[2, :]) <= state0[2] + 0.85 + 1.0e-9
    assert np.min(reference[0, :]) >= state0[0] - 0.20 - 1.0e-9
    assert np.max(reference[0, :]) <= state0[0] + 0.20 + 1.0e-9
    assert np.min(reference[1, :]) >= state0[1] - 0.20 - 1.0e-9
    assert np.max(reference[1, :]) <= state0[1] + 0.20 + 1.0e-9


def test_runtime_reference_selector_supports_sitl_identification_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v1",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)


def test_sitl_identification_reference_v2_matches_bounds_and_excites_heading_and_velocity():
    state0 = initial_state()
    rng = np.random.default_rng(2141444)
    reference = build_sitl_identification_reference_v2(state0, reference_duration_s=24.0, sim_timestep=1.0e-3, rng=rng)

    assert reference.shape == (18, 24000)
    assert np.min(reference[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference[2, :]) <= state0[2] + 0.90 + 1.0e-9
    assert np.min(reference[0, :]) >= state0[0] - 0.30 - 1.0e-9
    assert np.max(reference[0, :]) <= state0[0] + 0.30 + 1.0e-9
    assert np.min(reference[1, :]) >= state0[1] - 0.24 - 1.0e-9
    assert np.max(reference[1, :]) <= state0[1] + 0.24 + 1.0e-9
    assert np.max(np.abs(reference[3:6, :])) > 1.0e-3
    assert np.max(np.abs(reference[15:18, :])) <= 1.0e-9

    rotation_history = reference[6:15, :].reshape(3, 3, -1, order="F")
    determinants = np.linalg.det(np.moveaxis(rotation_history, 2, 0))
    npt.assert_allclose(determinants, np.ones_like(determinants), atol=1.0e-6)
    assert np.max(np.abs(reference[6:15, :] - np.repeat(state0[6:15, None], reference.shape[1], axis=1))) > 1.0e-3


def test_runtime_reference_selector_supports_sitl_identification_v2_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v2",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference_v2(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)


def test_sitl_identification_reference_v3_matches_bounds_and_excites_all_axes():
    state0 = initial_state()
    rng = np.random.default_rng(2141444)
    reference = build_sitl_identification_reference_v3(state0, reference_duration_s=24.0, sim_timestep=1.0e-3, rng=rng)

    assert reference.shape == (18, 24000)
    assert np.min(reference[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference[2, :]) <= state0[2] + 0.95 + 1.0e-9
    assert np.min(reference[0, :]) >= state0[0] - 0.50 - 1.0e-9
    assert np.max(reference[0, :]) <= state0[0] + 0.50 + 1.0e-9
    assert np.min(reference[1, :]) >= state0[1] - 0.50 - 1.0e-9
    assert np.max(reference[1, :]) <= state0[1] + 0.50 + 1.0e-9
    assert np.max(np.abs(reference[3:6, :])) > 0.25
    assert np.max(np.abs(reference[15:18, :])) <= 1.0e-9

    x_range = float(np.max(reference[0, :]) - np.min(reference[0, :]))
    y_range = float(np.max(reference[1, :]) - np.min(reference[1, :]))
    assert x_range > 0.45
    assert y_range > 0.35

    rotation_history = reference[6:15, :].reshape(3, 3, -1, order="F")
    determinants = np.linalg.det(np.moveaxis(rotation_history, 2, 0))
    npt.assert_allclose(determinants, np.ones_like(determinants), atol=1.0e-6)
    assert np.max(np.abs(reference[6:15, :] - np.repeat(state0[6:15, None], reference.shape[1], axis=1))) > 1.0e-3


def test_runtime_reference_selector_supports_sitl_identification_v3_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v3",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference_v3(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)


def test_sitl_identification_reference_v4_targets_forward_axis_excitation():
    state0 = initial_state()
    rng = np.random.default_rng(2141444)
    reference = build_sitl_identification_reference_v4(state0, reference_duration_s=24.0, sim_timestep=1.0e-3, rng=rng)

    assert reference.shape == (18, 24000)
    assert np.min(reference[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference[2, :]) <= state0[2] + 0.85 + 1.0e-9
    assert np.min(reference[0, :]) >= state0[0] - 0.55 - 1.0e-9
    assert np.max(reference[0, :]) <= state0[0] + 0.55 + 1.0e-9
    assert np.min(reference[1, :]) >= state0[1] - 0.25 - 1.0e-9
    assert np.max(reference[1, :]) <= state0[1] + 0.25 + 1.0e-9
    assert np.max(np.abs(reference[3:6, :])) > 0.30
    assert np.max(np.abs(reference[15:18, :])) <= 1.0e-9

    x_range = float(np.max(reference[0, :]) - np.min(reference[0, :]))
    y_range = float(np.max(reference[1, :]) - np.min(reference[1, :]))
    assert x_range > 0.55
    assert y_range > 0.12
    assert x_range > y_range

    rotation_history = reference[6:15, :].reshape(3, 3, -1, order="F")
    determinants = np.linalg.det(np.moveaxis(rotation_history, 2, 0))
    npt.assert_allclose(determinants, np.ones_like(determinants), atol=1.0e-6)


def test_runtime_reference_selector_supports_sitl_identification_v4_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v4",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference_v4(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)


def test_sitl_identification_reference_v5_is_deterministic_and_forward_dominant():
    state0 = initial_state()
    reference_a = build_sitl_identification_reference_v5(
        state0,
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141444),
    )
    reference_b = build_sitl_identification_reference_v5(
        state0,
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141447),
    )

    npt.assert_allclose(reference_a, reference_b)
    assert reference_a.shape == (18, 24000)
    assert np.min(reference_a[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference_a[2, :]) <= state0[2] + 0.85 + 1.0e-9
    assert np.min(reference_a[0, :]) >= state0[0] - 0.70 - 1.0e-9
    assert np.max(reference_a[0, :]) <= state0[0] + 0.70 + 1.0e-9
    assert np.min(reference_a[1, :]) >= state0[1] - 0.15 - 1.0e-9
    assert np.max(reference_a[1, :]) <= state0[1] + 0.15 + 1.0e-9
    assert np.max(np.abs(reference_a[3:6, :])) > 0.35
    assert np.max(np.abs(reference_a[15:18, :])) <= 1.0e-9

    x_range = float(np.max(reference_a[0, :]) - np.min(reference_a[0, :]))
    y_range = float(np.max(reference_a[1, :]) - np.min(reference_a[1, :]))
    assert x_range > 0.90
    assert x_range > 4.0 * y_range


def test_runtime_reference_selector_supports_sitl_identification_v5_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v5",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference_v5(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)


def test_sitl_identification_reference_v6_is_deterministic_and_balanced_across_axes():
    state0 = initial_state()
    reference_a = build_sitl_identification_reference_v6(
        state0,
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141444),
    )
    reference_b = build_sitl_identification_reference_v6(
        state0,
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=np.random.default_rng(2141447),
    )

    npt.assert_allclose(reference_a, reference_b)
    assert reference_a.shape == (18, 24000)
    assert np.min(reference_a[2, :]) >= state0[2] - 1.0e-9
    assert np.max(reference_a[2, :]) <= state0[2] + 0.85 + 1.0e-9
    assert np.min(reference_a[0, :]) >= state0[0] - 0.65 - 1.0e-9
    assert np.max(reference_a[0, :]) <= state0[0] + 0.65 + 1.0e-9
    assert np.min(reference_a[1, :]) >= state0[1] - 0.45 - 1.0e-9
    assert np.max(reference_a[1, :]) <= state0[1] + 0.45 + 1.0e-9
    assert np.max(np.abs(reference_a[3:6, :])) > 0.30
    assert np.max(np.abs(reference_a[15:18, :])) <= 1.0e-9

    x_range = float(np.max(reference_a[0, :]) - np.min(reference_a[0, :]))
    y_range = float(np.max(reference_a[1, :]) - np.min(reference_a[1, :]))
    assert x_range > 0.45
    assert y_range > 0.30

    rotation_history = reference_a[6:15, :].reshape(3, 3, -1, order="F")
    determinants = np.linalg.det(np.moveaxis(rotation_history, 2, 0))
    npt.assert_allclose(determinants, np.ones_like(determinants), atol=1.0e-6)
    assert np.max(np.abs(reference_a[6:15, :] - np.repeat(state0[6:15, None], reference_a.shape[1], axis=1))) > 1.0e-3


def test_runtime_reference_selector_supports_sitl_identification_v6_mode():
    state0 = initial_state()
    rng_selected = np.random.default_rng(2141444)
    rng_expected = np.random.default_rng(2141444)
    selected = build_runtime_reference(
        state0,
        reference_mode="sitl_identification_v6",
        reference_duration_s=24.0,
        sim_timestep=1.0e-3,
        rng=rng_selected,
    )
    expected = build_sitl_identification_reference_v6(state0, 24.0, 1.0e-3, rng_expected)
    npt.assert_allclose(selected, expected)
