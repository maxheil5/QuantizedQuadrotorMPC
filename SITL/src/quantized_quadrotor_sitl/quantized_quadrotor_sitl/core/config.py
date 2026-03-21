from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class MPCConfig:
    pred_horizon: int = 10
    sim_timestep: float = 1.0e-3
    sim_duration: float = 1.2
    use_casadi: bool = False
    position_error_weights_diag: list[float] = field(default_factory=lambda: [1.0e4, 1.0e4, 1.0e2])
    velocity_error_weights_diag: list[float] = field(default_factory=lambda: [1.0e2, 1.0e2, 1.0e2])
    attitude_error_weight: float = 1.0e2
    angular_velocity_error_weight: float = 1.0e2
    control_weights_diag: list[float] = field(default_factory=lambda: [1.0e-6, 1.0, 1.0, 1.0])

    @property
    def max_iter(self) -> int:
        return int(np.floor(self.sim_duration / self.sim_timestep))

    def position_error_weights(self) -> FloatArray:
        return np.asarray(self.position_error_weights_diag, dtype=float).reshape(3)

    def velocity_error_weights(self) -> FloatArray:
        return np.asarray(self.velocity_error_weights_diag, dtype=float).reshape(3)

    def control_weights(self) -> FloatArray:
        return np.asarray(self.control_weights_diag, dtype=float).reshape(4)


@dataclass(slots=True)
class OfflineExperimentConfig:
    profile_name: str
    random_seed: int
    dt: float
    train_traj_duration: float
    training_n_control: int
    n_basis: int
    word_lengths: list[int]
    include_unquantized: bool
    run_count: int
    prediction_eval_n_control: int
    tracking_enabled: bool
    paper_tracking_word_lengths: list[int]
    output_root: Path
    mpc: MPCConfig
    reference_horizon_duration: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        return payload


@dataclass(slots=True)
class HoverLocalOfflineConfig:
    profile_name: str
    random_seed: int
    dt: float
    train_traj_duration: float
    training_n_control: int
    prediction_eval_n_control: int
    n_basis: int
    output_root: Path
    collective_std_newton: float = 2.5
    collective_band_newton: float = 8.0
    body_moment_std_nm: list[float] = field(default_factory=lambda: [0.10, 0.10, 0.05])
    body_moment_band_nm: list[float] = field(default_factory=lambda: [0.30, 0.30, 0.15])

    def collective_bounds(self, hover_thrust_newton: float) -> tuple[float, float]:
        return (
            float(max(0.0, hover_thrust_newton - self.collective_band_newton)),
            float(hover_thrust_newton + self.collective_band_newton),
        )

    def body_moment_std(self) -> FloatArray:
        return np.asarray(self.body_moment_std_nm, dtype=float).reshape(3)

    def body_moment_bounds(self) -> FloatArray:
        return np.asarray(self.body_moment_band_nm, dtype=float).reshape(3)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        return payload


@dataclass(slots=True)
class VehicleScalingConfig:
    max_collective_thrust_newton: float = 80.0
    max_body_torque_x_nm: float = 4.0
    max_body_torque_y_nm: float = 4.0
    max_body_torque_z_nm: float = 2.5

    def torque_scale(self) -> FloatArray:
        return np.array(
            [
                self.max_body_torque_x_nm,
                self.max_body_torque_y_nm,
                self.max_body_torque_z_nm,
            ],
            dtype=float,
        )

    def control_lower_bounds(self) -> FloatArray:
        return np.array(
            [
                0.0,
                -self.max_body_torque_x_nm,
                -self.max_body_torque_y_nm,
                -self.max_body_torque_z_nm,
            ],
            dtype=float,
        )

    def control_upper_bounds(self) -> FloatArray:
        return np.array(
            [
                self.max_collective_thrust_newton,
                self.max_body_torque_x_nm,
                self.max_body_torque_y_nm,
                self.max_body_torque_z_nm,
            ],
            dtype=float,
        )

    def collective_command_newton(self, collective_newton: float) -> float:
        return float(
            np.clip(
                collective_newton,
                0.0,
                self.max_collective_thrust_newton,
            )
        )

    def collective_command_normalized(self, collective_newton: float) -> float:
        if self.max_collective_thrust_newton <= 0.0:
            return 0.0
        return self.collective_command_newton(collective_newton) / self.max_collective_thrust_newton


@dataclass(slots=True)
class RuntimeConfig:
    control_rate_hz: float = 100.0
    offboard_warmup_cycles: int = 10
    arm_retry_cycles: int = 500
    force_arm_in_sitl: bool = True
    force_arm_magic: float = 21196.0
    model_artifact: str = "results/offline/paper_v2/latest/edmd_unquantized.npz"
    quantization_mode: str = "none"
    quantized_word_length: int = 12
    reference_mode: str = "takeoff_hold"
    reference_seed: int = 2141444
    reference_duration_s: float = 10.0
    state_topic: str = "/quantized_mpc/state18"
    state_log_topic: str = "/quantized_mpc/state18_quantized"
    control_debug_topic: str = "/quantized_mpc/control_debug"
    vehicle_odometry_topic: str = "/fmu/out/vehicle_odometry"
    vehicle_local_position_topic: str = "/fmu/out/vehicle_local_position"
    vehicle_attitude_topic: str = "/fmu/out/vehicle_attitude"
    vehicle_angular_velocity_topic: str = "/fmu/out/vehicle_angular_velocity"
    vehicle_status_topic: str = "/fmu/out/vehicle_status_v1"
    offboard_control_mode_topic: str = "/fmu/in/offboard_control_mode"
    vehicle_command_topic: str = "/fmu/in/vehicle_command"
    vehicle_thrust_setpoint_topic: str = "/fmu/in/vehicle_thrust_setpoint"
    vehicle_torque_setpoint_topic: str = "/fmu/in/vehicle_torque_setpoint"
    state_history_limit: int = 10000
    results_dir: str = "results/sitl/latest"
    vehicle_scaling: VehicleScalingConfig = field(default_factory=VehicleScalingConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)


def initial_state() -> FloatArray:
    x0 = np.zeros(3, dtype=float)
    dx0 = np.zeros(3, dtype=float)
    r0 = np.eye(3, dtype=float).reshape(-1, order="F")
    wb0 = np.array([0.1, 0.0, 0.0], dtype=float)
    return np.concatenate([x0, dx0, r0, wb0])


def matlab_v2_profile(output_root: Path | None = None) -> OfflineExperimentConfig:
    return OfflineExperimentConfig(
        profile_name="matlab_v2",
        random_seed=2141444,
        dt=1.0e-3,
        train_traj_duration=0.1,
        training_n_control=100,
        n_basis=3,
        word_lengths=[4, 6, 8, 10, 12, 14],
        include_unquantized=True,
        run_count=5,
        prediction_eval_n_control=1,
        tracking_enabled=False,
        paper_tracking_word_lengths=[4, 8, 12, 14, 16],
        output_root=output_root or Path("results/offline"),
        mpc=MPCConfig(),
    )


def paper_v2_profile(output_root: Path | None = None) -> OfflineExperimentConfig:
    return OfflineExperimentConfig(
        profile_name="paper_v2",
        random_seed=2141444,
        dt=1.0e-3,
        train_traj_duration=0.1,
        training_n_control=100,
        n_basis=3,
        word_lengths=[4, 8, 12, 14, 16],
        include_unquantized=False,
        run_count=50,
        prediction_eval_n_control=1,
        tracking_enabled=True,
        paper_tracking_word_lengths=[4, 8, 12, 14, 16],
        output_root=output_root or Path("results/offline"),
        mpc=MPCConfig(),
    )


def hover_local_v1_profile(output_root: Path | None = None) -> HoverLocalOfflineConfig:
    return HoverLocalOfflineConfig(
        profile_name="hover_local_v1",
        random_seed=2141444,
        dt=1.0e-3,
        train_traj_duration=0.15,
        training_n_control=250,
        prediction_eval_n_control=40,
        n_basis=3,
        output_root=output_root or Path("results/offline"),
    )


def load_runtime_config(path: Path) -> RuntimeConfig:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    vehicle_scaling = VehicleScalingConfig(**payload.pop("vehicle_scaling", {}))
    mpc = MPCConfig(**payload.pop("mpc", {}))
    return RuntimeConfig(vehicle_scaling=vehicle_scaling, mpc=mpc, **payload)
