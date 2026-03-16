from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.core.config import initial_state
from quantized_quadrotor_sitl.experiments.runtime_reference import (
    build_hover_step_reference,
    build_paper_random_reference,
    build_runtime_reference,
)


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
