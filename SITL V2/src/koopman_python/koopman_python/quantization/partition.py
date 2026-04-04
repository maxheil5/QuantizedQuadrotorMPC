"""Word-length partitioning ported from MATLAB/Partition.m."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WordLengthPartition:
    """Container for the SciTech quantizer partition."""

    epsilon: np.ndarray
    min_new: np.ndarray
    max_new: np.ndarray
    partition: np.ndarray
    mid_points: np.ndarray
    word_length: int
    total_bins: int


def _as_column_vector(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2 or array.shape[1] != 1:
        raise ValueError("Expected a scalar, 1D array, or Nx1 column vector.")
    return array


def partition_word_length(
    data_min: np.ndarray | list[float],
    data_max: np.ndarray | list[float],
    word_length: int,
) -> WordLengthPartition:
    """Port of MATLAB/Partition.m."""

    if word_length <= 0:
        raise ValueError("word_length must be positive.")

    data_min_column = _as_column_vector(data_min)
    data_max_column = _as_column_vector(data_max)
    if data_min_column.shape != data_max_column.shape:
        raise ValueError("data_min and data_max must have the same shape.")

    span = data_max_column - data_min_column
    total_bins = 2**int(word_length)
    span_inflation = span / (2.0 * (total_bins - 1))
    epsilon = (span + span_inflation) / total_bins
    min_new = data_min_column - 0.5 * span_inflation
    max_new = data_max_column + 0.5 * span_inflation

    partition = np.column_stack(
        [
            np.linspace(float(min_new[i, 0]), float(max_new[i, 0]), total_bins + 1)
            for i in range(min_new.shape[0])
        ]
    ).T
    mid_points = 0.5 * (partition[:, :-1] + partition[:, 1:])

    return WordLengthPartition(
        epsilon=epsilon,
        min_new=min_new,
        max_new=max_new,
        partition=partition,
        mid_points=mid_points,
        word_length=int(word_length),
        total_bins=total_bins,
    )


def quantize_with_partition(
    data: np.ndarray,
    partition: WordLengthPartition,
) -> np.ndarray:
    """Port of MATLAB/quantizeCustom.m + Quantization.m."""

    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("data must be a 2D matrix or 1D column vector.")
    if values.shape[0] != partition.min_new.shape[0]:
        raise ValueError("data row count must match the partition row count.")

    quantized = np.zeros_like(values, dtype=float)
    bin_length = partition.epsilon.reshape(-1)
    min_new = partition.min_new.reshape(-1)
    max_new = partition.max_new.reshape(-1)
    mid_points = partition.mid_points

    for row_index in range(values.shape[0]):
        if bin_length[row_index] <= 0.0:
            quantized[row_index, :] = mid_points[row_index, 0]
            continue

        row_values = values[row_index, :]
        indices = np.ceil((row_values - min_new[row_index]) / bin_length[row_index]).astype(int)
        indices = np.clip(indices, 1, partition.total_bins) - 1
        quantized[row_index, :] = mid_points[row_index, indices]

        above_mask = row_values >= max_new[row_index]
        below_mask = row_values <= min_new[row_index]
        if np.any(above_mask):
            quantized[row_index, above_mask] = mid_points[row_index, -1]
        if np.any(below_mask):
            quantized[row_index, below_mask] = mid_points[row_index, 0]

    return quantized
