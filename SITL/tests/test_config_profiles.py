from pathlib import Path

from quantized_quadrotor_sitl.core.config import hover_local_v1_profile, load_runtime_config, matlab_v2_profile, paper_v2_profile


RESIDUAL_ARTIFACT_PATH = "results/offline/sitl_baseline_v1/20260401T234823Z_sitl_id_v7_hovercentered_affine/edmd_unquantized.npz"


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


def test_hover_local_profile_matches_sitl_debug_defaults():
    config = hover_local_v1_profile()
    assert config.profile_name == "hover_local_v1"
    assert config.dt == 1.0e-3
    assert config.train_traj_duration == 0.15
    assert config.training_n_control == 250
    assert config.prediction_eval_n_control == 40
    assert config.n_basis == 3
    assert config.collective_std_newton == 2.5
    assert config.collective_band_newton == 8.0
    assert config.body_moment_std_nm == [0.10, 0.10, 0.05]
    assert config.body_moment_band_nm == [0.30, 0.30, 0.15]


def test_runtime_config_includes_estimator_topic_defaults():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime.yaml"))
    assert config.control_rate_hz == 100.0
    assert config.force_arm_in_sitl is True
    assert config.force_arm_magic == 21196.0
    assert config.learned_bound_margin_fraction == 0.05
    assert config.controller_mode == "baseline_geometric"
    assert config.reference_mode == "takeoff_hold"
    assert config.vehicle_scaling.max_collective_thrust_newton == 62.0
    assert config.vehicle_scaling.max_body_torque_x_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_y_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_z_nm == 0.6
    assert config.baseline.position_gains_diag == [0.5, 0.5, 5.5]
    assert config.baseline.velocity_gains_diag == [0.8, 0.8, 3.0]
    assert config.baseline.attitude_gains_diag == [2.5, 2.5, 0.8]
    assert config.baseline.angular_rate_gains_diag == [0.25, 0.25, 0.15]
    assert config.baseline.z_integral_gain == 1.2
    assert config.baseline.z_integral_limit == 1.5
    assert config.baseline.max_tilt_deg == 12.0
    assert config.mpc.pred_horizon == 10
    assert config.mpc.sim_timestep == 1.0e-3
    assert config.mpc.position_error_weights_diag == [250.0, 250.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [40.0, 40.0, 200.0]
    assert config.mpc.attitude_error_weight == 100.0
    assert config.mpc.angular_velocity_error_weight == 150.0
    assert config.mpc.control_weights_diag == [1.0e-5, 40.0, 40.0, 60.0]
    assert config.mpc.control_delta_weights_diag == [1.0, 6.0, 6.0, 8.0]
    assert config.vehicle_odometry_topic == "/fmu/out/vehicle_odometry"
    assert config.vehicle_local_position_topic == "/fmu/out/vehicle_local_position"
    assert config.vehicle_attitude_topic == "/fmu/out/vehicle_attitude"
    assert config.vehicle_angular_velocity_topic == "/fmu/out/vehicle_angular_velocity"
    assert config.vehicle_status_topic == "/fmu/out/vehicle_status_v1"


def test_sitl_retrained_edmd_runtime_config_preserves_baseline_scaling():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.vehicle_scaling.max_collective_thrust_newton == 62.0
    assert config.vehicle_scaling.max_body_torque_x_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_y_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_z_nm == 0.6
    assert config.mpc.pred_horizon == 10
    assert config.mpc.control_weights_diag == [1.0e-5, 40.0, 40.0, 60.0]
    assert config.mpc.control_delta_weights_diag == [1.0, 6.0, 6.0, 8.0]


def test_sitl_retrained_edmd_light_runtime_config_reduces_solver_load():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.vehicle_scaling.max_collective_thrust_newton == 62.0
    assert config.vehicle_scaling.max_body_torque_x_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_y_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_z_nm == 0.6
    assert config.mpc.pred_horizon == 8
    assert config.mpc.control_weights_diag == [1.0e-5, 40.0, 40.0, 60.0]
    assert config.mpc.control_delta_weights_diag == [1.0, 6.0, 6.0, 8.0]


def test_sitl_retrained_edmd_light_latest_runtime_config_tracks_latest_retrain_artifact():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light_latest.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == "results/offline/sitl_baseline_v1/latest/edmd_unquantized.npz"
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.learned_bound_margin_fraction == 0.05
    assert config.mpc.pred_horizon == 8
    assert config.mpc.position_error_weights_diag == [250.0, 250.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [40.0, 40.0, 200.0]
    assert config.mpc.control_weights_diag == [1.0e-5, 40.0, 40.0, 60.0]
    assert config.mpc.control_delta_weights_diag == [1.0, 6.0, 6.0, 8.0]


def test_sitl_retrained_edmd_light_tuned_runtime_config_pins_artifact_and_tunes_lateral_weights():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light_tuned.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.mpc.pred_horizon == 8
    assert config.mpc.position_error_weights_diag == [600.0, 600.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [120.0, 120.0, 200.0]
    assert config.mpc.control_weights_diag == [1.0e-5, 40.0, 40.0, 60.0]
    assert config.mpc.control_delta_weights_diag == [2.0, 10.0, 10.0, 12.0]


def test_sitl_retrained_edmd_light_conservative_runtime_config_prioritizes_bound_suppression():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light_conservative.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.mpc.pred_horizon == 8
    assert config.mpc.position_error_weights_diag == [350.0, 350.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [60.0, 60.0, 200.0]
    assert config.mpc.control_weights_diag == [1.0e-5, 60.0, 60.0, 90.0]
    assert config.mpc.control_delta_weights_diag == [4.0, 16.0, 16.0, 20.0]


def test_sitl_retrained_edmd_light_margin10_runtime_config_uses_bound_margin_with_small_regularization_bump():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light_margin10.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.learned_bound_margin_fraction == 0.10
    assert config.mpc.pred_horizon == 8
    assert config.mpc.position_error_weights_diag == [250.0, 250.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [40.0, 40.0, 200.0]
    assert config.mpc.control_weights_diag == [1.0e-5, 50.0, 50.0, 75.0]
    assert config.mpc.control_delta_weights_diag == [2.0, 10.0, 10.0, 12.0]


def test_sitl_retrained_edmd_light_regularized_runtime_config_keeps_baseline_tracking_with_small_effort_bump():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_sitl_retrain_edmd_light_regularized.yaml"))
    assert config.controller_mode == "edmd_mpc"
    assert config.model_artifact == RESIDUAL_ARTIFACT_PATH
    assert config.reference_mode == "takeoff_hold"
    assert config.control_rate_hz == 50.0
    assert config.learned_bound_margin_fraction == 0.05
    assert config.mpc.pred_horizon == 8
    assert config.mpc.position_error_weights_diag == [250.0, 250.0, 5000.0]
    assert config.mpc.velocity_error_weights_diag == [40.0, 40.0, 200.0]
    assert config.mpc.control_weights_diag == [1.0e-5, 50.0, 50.0, 75.0]
    assert config.mpc.control_delta_weights_diag == [2.0, 10.0, 10.0, 12.0]


def test_sitl_identification_runtime_config_preserves_known_good_baseline():
    config = load_runtime_config(Path("SITL/configs/sitl_runtime_identification.yaml"))
    assert config.controller_mode == "baseline_geometric"
    assert config.quantization_mode == "none"
    assert config.reference_mode == "sitl_identification_v7"
    assert config.reference_duration_s == 24.0
    assert config.vehicle_scaling.max_collective_thrust_newton == 62.0
    assert config.vehicle_scaling.max_body_torque_x_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_y_nm == 1.0
    assert config.vehicle_scaling.max_body_torque_z_nm == 0.6
    assert config.baseline.position_gains_diag == [0.5, 0.5, 5.5]
    assert config.baseline.velocity_gains_diag == [0.8, 0.8, 3.0]
    assert config.baseline.attitude_gains_diag == [2.5, 2.5, 0.8]
    assert config.baseline.angular_rate_gains_diag == [0.25, 0.25, 0.15]
    assert config.baseline.z_integral_gain == 1.2
    assert config.baseline.z_integral_limit == 1.5
