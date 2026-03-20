from pathlib import Path

from quantized_quadrotor_sitl.core.config import load_runtime_config, matlab_v2_profile, paper_v2_profile


def test_matlab_profile_matches_v2_defaults():
    config = matlab_v2_profile()
    assert config.word_lengths == [4, 6, 8, 10, 12, 14]
    assert config.run_count == 5
    assert config.include_unquantized is True
    assert config.tracking_enabled is False


def test_paper_profile_matches_manuscript_study():
    config = paper_v2_profile()
    assert config.word_lengths == [4, 8, 12, 14, 16]
    assert config.run_count == 50
    assert config.tracking_enabled is True


def test_runtime_config_includes_estimator_topic_defaults():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime.yaml"))
    assert config.control_rate_hz == 100.0
    assert config.force_arm_in_sitl is True
    assert config.force_arm_magic == 21196.0
    assert config.reference_mode == "takeoff_hold"
    assert config.vehicle_scaling.hover_thrust_bias_newton == 40.0
    assert config.mpc.pred_horizon == 10
    assert config.mpc.sim_timestep == 1.0e-3
    assert config.mpc.position_error_weights_diag == [500.0, 500.0, 20000.0]
    assert config.mpc.velocity_error_weights_diag == [50.0, 50.0, 500.0]
    assert config.mpc.attitude_error_weight == 100.0
    assert config.mpc.angular_velocity_error_weight == 500.0
    assert config.mpc.control_weights_diag == [1.0e-6, 12.0, 12.0, 30.0]
    assert config.vehicle_odometry_topic == "/fmu/out/vehicle_odometry"
    assert config.vehicle_local_position_topic == "/fmu/out/vehicle_local_position"
    assert config.vehicle_attitude_topic == "/fmu/out/vehicle_attitude"
    assert config.vehicle_angular_velocity_topic == "/fmu/out/vehicle_angular_velocity"
    assert config.vehicle_status_topic == "/fmu/out/vehicle_status_v1"
