import numpy as np

from quantized_quadrotor_sitl.utils.linear_algebra import hat_map, vectorize, vee_map


def test_hat_and_vee_are_consistent():
    vector = np.array([1.2, -0.4, 2.0])
    assert np.allclose(vee_map(hat_map(vector)), vector)


def test_vectorize_is_column_major():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(vectorize(matrix), np.array([1.0, 3.0, 2.0, 4.0]))

