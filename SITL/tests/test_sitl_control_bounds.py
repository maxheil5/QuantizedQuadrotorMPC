from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quantized_quadrotor_sitl.core.config import MPCConfig, VehicleScalingConfig
from quantized_quadrotor_sitl.core.types import EDMDModel
from quantized_quadrotor_sitl.mpc.qp import get_qp
from quantized_quadrotor_sitl.telemetry.adapter import physical_control_to_px4_wrench


def test_vehicle_scaling_exposes_absolute_collective_control_bounds():
    scaling = VehicleScalingConfig()
    npt.assert_allclose(scaling.control_lower_bounds(), np.array([0.0, -4.0, -4.0, -2.5]))
    npt.assert_allclose(scaling.control_upper_bounds(), np.array([80.0, 4.0, 4.0, 2.5]))
    assert scaling.collective_command_newton(40.0) == 40.0
    assert scaling.collective_command_normalized(40.0) == 0.5


def test_qp_uses_asymmetric_collective_bounds_for_sitl():
    lifted_state = np.zeros(24, dtype=float)
    lifted_state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
    lifted_reference = np.zeros((24, 1), dtype=float)
    lifted_reference[6:15, 0] = np.eye(3, dtype=float).reshape(-1, order="F")

    model = EDMDModel(
        A=np.eye(24, dtype=float),
        B=np.zeros((24, 4), dtype=float),
        C=np.eye(24, dtype=float),
        Z1=np.zeros((24, 1), dtype=float),
        Z2=np.zeros((24, 1), dtype=float),
        n_basis=3,
    )
    scaling = VehicleScalingConfig()

    _, _, _, b_ineq = get_qp(
        model,
        lifted_state,
        lifted_reference,
        1,
        MPCConfig(),
        scaling.control_lower_bounds(),
        scaling.control_upper_bounds(),
    )

    npt.assert_allclose(b_ineq, np.array([0.0, 80.0, 4.0, 4.0, 4.0, 4.0, 2.5, 2.5]))


def test_px4_wrench_adapter_clamps_negative_collective_command():
    thrust_body, normalized_moments, collective_command_newton, collective_normalized = physical_control_to_px4_wrench(
        np.array([-50.0, 0.5, -0.25, 0.1], dtype=float),
        max_collective_thrust_newton=80.0,
        max_body_torque_nm=np.array([4.0, 4.0, 2.5], dtype=float),
    )

    npt.assert_allclose(thrust_body, np.array([0.0, 0.0, 0.0]))
    npt.assert_allclose(normalized_moments, np.array([0.125, 0.0625, -0.04]))
    assert collective_command_newton == 0.0
    assert collective_normalized == 0.0


def test_px4_wrench_adapter_normalizes_absolute_collective_command():
    thrust_body, normalized_moments, collective_command_newton, collective_normalized = physical_control_to_px4_wrench(
        np.array([40.0, -0.5, 0.25, -0.1], dtype=float),
        max_collective_thrust_newton=80.0,
        max_body_torque_nm=np.array([4.0, 4.0, 2.5], dtype=float),
    )

    npt.assert_allclose(thrust_body, np.array([0.0, 0.0, -0.5]))
    npt.assert_allclose(normalized_moments, np.array([-0.125, -0.0625, 0.04]))
    assert collective_command_newton == 40.0
    assert collective_normalized == 0.5


def test_qp_uses_configured_control_weights_in_hessian():
    lifted_state = np.zeros(24, dtype=float)
    lifted_state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
    lifted_reference = np.zeros((24, 1), dtype=float)
    lifted_reference[6:15, 0] = np.eye(3, dtype=float).reshape(-1, order="F")

    model = EDMDModel(
        A=np.eye(24, dtype=float),
        B=np.zeros((24, 4), dtype=float),
        C=np.eye(24, dtype=float),
        Z1=np.zeros((24, 1), dtype=float),
        Z2=np.zeros((24, 1), dtype=float),
        n_basis=3,
    )
    config = MPCConfig(control_weights_diag=[0.25, 2.0, 3.0, 4.0])

    _, g_matrix, _, _ = get_qp(
        model,
        lifted_state,
        lifted_reference,
        1,
        config,
        np.array([-1.0, -1.0, -1.0, -1.0], dtype=float),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
    )

    npt.assert_allclose(np.diag(g_matrix), np.array([0.5, 4.0, 6.0, 8.0]))
