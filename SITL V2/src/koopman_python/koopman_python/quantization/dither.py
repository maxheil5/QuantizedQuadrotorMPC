"""Dither quantization ported from MATLAB/Dither_Func.m."""

from __future__ import annotations

import numpy as np

from koopman_python.quantization.partition import WordLengthPartition, quantize_with_partition


def dither_signal(
    input_data: np.ndarray,
    epsilon: WordLengthPartition | np.ndarray,
    min_new: np.ndarray | None = None,
    max_new: np.ndarray | None = None,
    partition: np.ndarray | None = None,
    mid_points: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Port of MATLAB/Dither_Func.m."""

    values = np.asarray(input_data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("input_data must be a 2D matrix or 1D column vector.")

    if isinstance(epsilon, WordLengthPartition):
        quantizer = epsilon
    else:
        if min_new is None or max_new is None or partition is None or mid_points is None:
            raise ValueError("Explicit quantizer arrays are required when epsilon is not a WordLengthPartition.")
        epsilon_array = np.asarray(epsilon, dtype=float)
        if epsilon_array.ndim == 1:
            epsilon_array = epsilon_array.reshape(-1, 1)
        quantizer = WordLengthPartition(
            epsilon=epsilon_array,
            min_new=np.asarray(min_new, dtype=float),
            max_new=np.asarray(max_new, dtype=float),
            partition=np.asarray(partition, dtype=float),
            mid_points=np.asarray(mid_points, dtype=float),
            word_length=int(np.log2(np.asarray(mid_points).shape[1])),
            total_bins=int(np.asarray(mid_points).shape[1]),
        )

    generator = np.random.default_rng() if rng is None else rng
    dither_noise = (quantizer.epsilon / 2.0) * (generator.random(values.shape) - 0.5)
    noisy_values = values + dither_noise
    quantized_noisy_values = quantize_with_partition(noisy_values, quantizer)
    return quantized_noisy_values - dither_noise
