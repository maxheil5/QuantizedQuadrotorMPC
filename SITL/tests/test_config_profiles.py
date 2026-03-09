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
    assert config.vehicle_odometry_topic == "/fmu/out/vehicle_odometry"
    assert config.vehicle_local_position_topic == "/fmu/out/vehicle_local_position"
    assert config.vehicle_attitude_topic == "/fmu/out/vehicle_attitude"
    assert config.vehicle_angular_velocity_topic == "/fmu/out/vehicle_angular_velocity"
