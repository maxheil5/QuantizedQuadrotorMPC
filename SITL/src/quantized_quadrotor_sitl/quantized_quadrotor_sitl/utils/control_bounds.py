from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import VehicleScalingConfig


@dataclass(slots=True)
class RuntimeControlCoordinates:
    normalized: bool
    trim: np.ndarray
    scale: np.ndarray
    physical_lower_bounds: np.ndarray
    physical_upper_bounds: np.ndarray
    internal_lower_bounds: np.ndarray
    internal_upper_bounds: np.ndarray

    def physical_to_internal(self, control_physical: np.ndarray) -> np.ndarray:
        control = np.asarray(control_physical, dtype=float).reshape(self.trim.shape)
        return (control - self.trim) / self.scale

    def internal_to_physical(self, control_internal: np.ndarray) -> np.ndarray:
        control = np.asarray(control_internal, dtype=float).reshape(self.trim.shape)
        return self.trim + self.scale * control


def _metadata_vector(
    metadata: dict[str, object] | None,
    key: str,
    width: int,
) -> np.ndarray | None:
    if not metadata or key not in metadata:
        return None
    values = np.asarray(metadata[key], dtype=float).reshape(-1)
    if values.size != width:
        return None
    return values


def _residual_enabled(metadata: dict[str, object] | None) -> bool:
    if not metadata:
        return False
    return bool(metadata.get("residual_enabled", False))


def runtime_edmd_control_bounds(
    scaling: VehicleScalingConfig,
    metadata: dict[str, object] | None,
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


def runtime_edmd_control_coordinates(
    scaling: VehicleScalingConfig,
    metadata: dict[str, object] | None,
) -> RuntimeControlCoordinates:
    vehicle_lower_bounds = scaling.control_lower_bounds()
    vehicle_upper_bounds = scaling.control_upper_bounds()
    width = vehicle_lower_bounds.size

    u_train_min = _metadata_vector(metadata, "u_train_min", width)
    u_train_max = _metadata_vector(metadata, "u_train_max", width)
    u_trim = _metadata_vector(metadata, "u_trim", width)
    u_train_std = _metadata_vector(metadata, "u_train_std", width)

    if u_train_min is None or u_train_max is None or u_trim is None or u_train_std is None:
        physical_lower_bounds, physical_upper_bounds = runtime_edmd_control_bounds(scaling, metadata)
        trim = np.zeros(width, dtype=float)
        scale = np.ones(width, dtype=float)
        return RuntimeControlCoordinates(
            normalized=False,
            trim=trim,
            scale=scale,
            physical_lower_bounds=physical_lower_bounds,
            physical_upper_bounds=physical_upper_bounds,
            internal_lower_bounds=physical_lower_bounds.copy(),
            internal_upper_bounds=physical_upper_bounds.copy(),
        )

    physical_lower_bounds = np.maximum(vehicle_lower_bounds, u_train_min)
    physical_upper_bounds = np.minimum(vehicle_upper_bounds, u_train_max)
    if _residual_enabled(metadata):
        span = np.maximum(physical_upper_bounds - physical_lower_bounds, 0.0)
        margin = 0.05 * span
        candidate_lower_bounds = physical_lower_bounds + margin
        candidate_upper_bounds = physical_upper_bounds - margin
        valid_mask = candidate_lower_bounds <= candidate_upper_bounds
        physical_lower_bounds = np.where(valid_mask, candidate_lower_bounds, physical_lower_bounds)
        physical_upper_bounds = np.where(valid_mask, candidate_upper_bounds, physical_upper_bounds)
    physical_upper_bounds = np.maximum(physical_upper_bounds, physical_lower_bounds)
    scale = np.maximum(np.abs(u_train_std), 1.0e-6)
    internal_lower_bounds = (physical_lower_bounds - u_trim) / scale
    internal_upper_bounds = (physical_upper_bounds - u_trim) / scale
    return RuntimeControlCoordinates(
        normalized=True,
        trim=u_trim,
        scale=scale,
        physical_lower_bounds=physical_lower_bounds,
        physical_upper_bounds=physical_upper_bounds,
        internal_lower_bounds=internal_lower_bounds,
        internal_upper_bounds=internal_upper_bounds,
    )
