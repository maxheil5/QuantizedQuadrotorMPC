import numpy as np

from quantized_quadrotor_sitl.core.config import initial_state
from quantized_quadrotor_sitl.edmd.basis import get_basis, lift_state
from quantized_quadrotor_sitl.edmd.fit import get_edmd


def test_basis_dimension_matches_matlab_formula():
    basis = get_basis(initial_state(), n_basis=3)
    assert basis.shape == (45,)


def test_lifted_state_dimension_is_51():
    lifted = lift_state(initial_state(), n_basis=3)
    assert lifted.shape == (51,)


def test_edmd_output_shapes_match_matlab():
    state = initial_state()
    x1 = np.repeat(state[:, None], 2, axis=1)
    x2 = np.repeat(state[:, None], 2, axis=1)
    u1 = np.zeros((4, 2))
    model = get_edmd(x1, x2, u1, n_basis=3)
    assert model.A.shape == (51, 51)
    assert model.B.shape == (51, 4)
    assert model.C.shape == (24, 51)


def test_affine_edmd_exposes_bias_vector():
    state = initial_state()
    x1 = np.repeat(state[:, None], 3, axis=1)
    x2 = x1.copy()
    x2[0, :] += np.array([0.1, 0.2, 0.3], dtype=float)
    u1 = np.zeros((4, 3))
    model = get_edmd(x1, x2, u1, n_basis=0, affine=True)

    assert model.affine_enabled is True
    assert model.bias is not None
    assert model.bias.shape == (24,)
