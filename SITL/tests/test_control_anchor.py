from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.core.config import MomentAuthorityAnchorConfig
from quantized_quadrotor_sitl.utils.control_anchor import apply_moment_authority_anchor


def test_moment_authority_anchor_lifts_weak_moments_to_baseline_fraction():
    learned = np.array([50.0, -0.01, 0.02, 0.0], dtype=float)
    baseline = np.array([50.0, -0.40, 0.30, -0.12], dtype=float)
    lower = np.array([0.0, -1.0, -1.0, -0.6], dtype=float)
    upper = np.array([62.0, 1.0, 1.0, 0.6], dtype=float)
    config = MomentAuthorityAnchorConfig(enabled=True, minimum_baseline_fraction=0.25)

    anchored = apply_moment_authority_anchor(learned, baseline, lower, upper, config)

    npt.assert_allclose(anchored, np.array([50.0, -0.10, 0.075, -0.03], dtype=float))


def test_moment_authority_anchor_leaves_inactive_or_already_strong_axes_unchanged():
    learned = np.array([50.0, -0.08, 0.06, 0.04], dtype=float)
    baseline = np.array([50.0, -0.04, 0.20, 0.02], dtype=float)
    lower = np.array([0.0, -1.0, -1.0, -0.6], dtype=float)
    upper = np.array([62.0, 1.0, 1.0, 0.6], dtype=float)
    config = MomentAuthorityAnchorConfig(enabled=True, minimum_baseline_fraction=0.25)

    anchored = apply_moment_authority_anchor(learned, baseline, lower, upper, config)

    npt.assert_allclose(anchored, learned)


def test_moment_authority_anchor_respects_bounds_without_shrinking_existing_command():
    learned = np.array([50.0, 0.10, 0.02, 0.0], dtype=float)
    baseline = np.array([50.0, 2.00, 0.30, 0.0], dtype=float)
    lower = np.array([0.0, -1.0, -1.0, -0.6], dtype=float)
    upper = np.array([62.0, 0.30, 1.0, 0.6], dtype=float)
    config = MomentAuthorityAnchorConfig(enabled=True, minimum_baseline_fraction=0.50)

    anchored = apply_moment_authority_anchor(learned, baseline, lower, upper, config)

    npt.assert_allclose(anchored, np.array([50.0, 0.30, 0.15, 0.0], dtype=float))
