from __future__ import annotations

import numpy as np

from ..core.config import VehicleScalingConfig


def runtime_edmd_control_bounds(
    scaling: VehicleScalingConfig,
    metadata: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = scaling.control_lower_bounds().copy()
    upper_bounds = scaling.control_upper_bounds().copy()

    if not metadata or "u_train_min" not in metadata:
        return lower_bounds, upper_bounds

    u_train_min = np.asarray(metadata["u_train_min"], dtype=float).reshape(-1)
    if u_train_min.size == 0:
        return lower_bounds, upper_bounds

    collective_floor = float(u_train_min[0])
    lower_bounds[0] = min(upper_bounds[0], max(lower_bounds[0], collective_floor))
    return lower_bounds, upper_bounds
