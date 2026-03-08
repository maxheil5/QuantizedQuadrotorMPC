import numpy as np

from quantized_quadrotor_sitl.quantization.dither import dither_signal
from quantized_quadrotor_sitl.quantization.partition import partition_range, quantize_scalar


def test_partition_shape_matches_word_length():
    u_min = np.array([[0.0], [1.0]])
    u_max = np.array([[1.0], [3.0]])
    epsilon, _, _, partition, mid_points = partition_range(u_min, u_max, bits=4)
    assert epsilon.shape == (2, 1)
    assert partition.shape == (2, 17)
    assert mid_points.shape == (2, 16)


def test_quantize_scalar_uses_midpoint_bins():
    mid_points = np.array([0.25, 0.75])
    assert quantize_scalar(0.0, 1.0, 0.1, 0.5, mid_points) == 0.25
    assert quantize_scalar(0.0, 1.0, 0.9, 0.5, mid_points) == 0.75


def test_dither_signal_preserves_shape():
    rng = np.random.default_rng(7)
    data = np.array([[0.2, 0.4], [1.2, 1.4]])
    epsilon, min_new, max_new, _, mid_points = partition_range(
        np.min(data, axis=1, keepdims=True),
        np.max(data, axis=1, keepdims=True),
        bits=4,
    )
    quantized, dither = dither_signal(data, epsilon, min_new, max_new, mid_points, rng)
    assert quantized.shape == data.shape
    assert dither.shape == data.shape

