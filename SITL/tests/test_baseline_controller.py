from __future__ import annotations

import numpy as np

from quantized_quadrotor_sitl.controllers import compute_baseline_control
from quantized_quadrotor_sitl.core.config import BaselineControllerConfig, VehicleScalingConfig
from quantized_quadrotor_sitl.dynamics.params import get_params


def _hover_state() -> np.ndarray:
    state = np.zeros(18, dtype=float)
    state[6:15] = np.eye(3, dtype=float).reshape(-1, order="F")
    return state


def test_baseline_controller_returns_hover_thrust_at_trim():
    state = _hover_state()
    reference = state.copy()

    control_raw, control_used = compute_baseline_control(
        state,
        reference,
        z_error_integral=0.0,
        config=BaselineControllerConfig(),
        scaling=VehicleScalingConfig(),
        params=get_params(),
    )

    hover_thrust = get_params().mass * get_params().g
    assert abs(control_raw[0] - hover_thrust) < 1.0e-9
    assert abs(control_used[0] - hover_thrust) < 1.0e-9
    assert np.allclose(control_used[1:4], np.zeros(3), atol=1.0e-9)


def test_baseline_controller_increases_collective_for_takeoff_error():
    state = _hover_state()
    reference = state.copy()
    reference[2] += 0.75

    control_raw, control_used = compute_baseline_control(
        state,
        reference,
        z_error_integral=0.0,
        config=BaselineControllerConfig(),
        scaling=VehicleScalingConfig(),
        params=get_params(),
    )

    hover_thrust = get_params().mass * get_params().g
    assert control_raw[0] > hover_thrust
    assert control_used[0] > hover_thrust


def test_baseline_controller_clips_moments_to_vehicle_scaling_limits():
    state = _hover_state()
    reference = state.copy()
    ninety_deg_yaw = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    reference[6:15] = ninety_deg_yaw.reshape(-1, order="F")
    scaling = VehicleScalingConfig(max_collective_thrust_newton=80.0, max_body_torque_x_nm=0.2, max_body_torque_y_nm=0.2, max_body_torque_z_nm=0.1)

    _, control_used = compute_baseline_control(
        state,
        reference,
        z_error_integral=0.0,
        config=BaselineControllerConfig(),
        scaling=scaling,
        params=get_params(),
    )

    assert np.all(control_used[1:4] <= scaling.control_upper_bounds()[1:4] + 1.0e-9)
    assert np.all(control_used[1:4] >= scaling.control_lower_bounds()[1:4] - 1.0e-9)
