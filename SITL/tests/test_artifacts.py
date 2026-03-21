from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.core.artifacts import coarsen_edmd_model
from quantized_quadrotor_sitl.core.types import EDMDModel


def test_coarsen_edmd_model_matches_zero_order_hold_accumulation():
    model = EDMDModel(
        A=np.array([[2.0]], dtype=float),
        B=np.array([[3.0]], dtype=float),
        C=np.array([[1.0]], dtype=float),
        Z1=np.zeros((1, 1), dtype=float),
        Z2=np.zeros((1, 1), dtype=float),
        n_basis=1,
    )

    coarsened = coarsen_edmd_model(model, step_multiple=3)

    npt.assert_allclose(coarsened.A, np.array([[8.0]], dtype=float))
    npt.assert_allclose(coarsened.B, np.array([[21.0]], dtype=float))
    npt.assert_allclose(coarsened.C, model.C)
    npt.assert_allclose(coarsened.Z1, model.Z1)
    npt.assert_allclose(coarsened.Z2, model.Z2)
