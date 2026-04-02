from __future__ import annotations

import numpy as np

from ..core.config import MomentAuthorityAnchorConfig


def apply_moment_authority_anchor(
    learned_control: np.ndarray,
    baseline_control: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: MomentAuthorityAnchorConfig,
) -> np.ndarray:
    anchored = np.asarray(learned_control, dtype=float).reshape(4).copy()
    baseline = np.asarray(baseline_control, dtype=float).reshape(4)
    lower = np.asarray(lower_bounds, dtype=float).reshape(4)
    upper = np.asarray(upper_bounds, dtype=float).reshape(4)

    if not config.enabled:
        return anchored

    minimum_fraction = max(0.0, float(config.minimum_baseline_fraction))
    thresholds = config.active_thresholds()
    for axis_offset, threshold in enumerate(thresholds, start=1):
        baseline_value = float(baseline[axis_offset])
        baseline_magnitude = abs(baseline_value)
        if baseline_magnitude < float(threshold):
            continue

        minimum_magnitude = minimum_fraction * baseline_magnitude
        if abs(anchored[axis_offset]) >= minimum_magnitude:
            continue

        baseline_sign = float(np.sign(baseline_value))
        if baseline_sign == 0.0:
            continue

        candidate = float(np.clip(baseline_sign * minimum_magnitude, lower[axis_offset], upper[axis_offset]))
        if abs(candidate) > abs(anchored[axis_offset]):
            anchored[axis_offset] = candidate
    return anchored
