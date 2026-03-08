from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .partition import quantize_custom


FloatArray = NDArray[np.float64]


def dither_signal(
    input_data: FloatArray,
    epsilon: FloatArray,
    min_new: FloatArray,
    max_new: FloatArray,
    mid_points: FloatArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    input_data = np.asarray(input_data, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float).reshape(-1, 1)
    dither = (epsilon / 2.0) * (rng.random(input_data.shape) - 0.5)
    noisy = input_data + dither
    quantized = quantize_custom(min_new, max_new, noisy, epsilon, mid_points)
    return quantized - dither, dither
