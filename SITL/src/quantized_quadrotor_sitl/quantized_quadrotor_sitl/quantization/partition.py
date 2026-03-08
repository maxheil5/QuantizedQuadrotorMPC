from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def partition_range(u_min: FloatArray, u_max: FloatArray, bits: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    u_min = np.asarray(u_min, dtype=float).reshape(-1, 1)
    u_max = np.asarray(u_max, dtype=float).reshape(-1, 1)
    span = u_max - u_min
    total_bins = int(math.pow(2, bits))
    span_inflation = span / (2.0 * (total_bins - 1))
    epsilon = (span + span_inflation) / total_bins
    u_min_new = u_min - span_inflation / 2.0
    u_max_new = u_max + span_inflation / 2.0

    partition = np.vstack(
        [
            np.linspace(u_min_new[i, 0], u_max_new[i, 0], total_bins + 1)
            for i in range(u_min_new.shape[0])
        ]
    )
    mid_points = (partition[:, :-1] + partition[:, 1:]) / 2.0
    return epsilon, u_min_new, u_max_new, partition, mid_points


def quantize_scalar(u_min: float, u_max: float, value: float, bin_length: float, mid_points: FloatArray) -> float:
    if value >= u_max:
        return float(mid_points[-1])
    if value <= u_min:
        return float(mid_points[0])
    idx = int(np.ceil((value - u_min) / bin_length))
    return float(mid_points[idx - 1])


def quantize_custom(u_min: FloatArray, u_max: FloatArray, data: FloatArray, bin_length: FloatArray, mid_points: FloatArray) -> FloatArray:
    u_min = np.asarray(u_min, dtype=float).reshape(-1)
    u_max = np.asarray(u_max, dtype=float).reshape(-1)
    bin_length = np.asarray(bin_length, dtype=float).reshape(-1)
    data = np.asarray(data, dtype=float)
    mid_points = np.asarray(mid_points, dtype=float)
    quantized = np.zeros_like(data, dtype=float)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            quantized[row, col] = quantize_scalar(
                u_min[row],
                u_max[row],
                data[row, col],
                bin_length[row],
                mid_points[row, :],
            )
    return quantized

