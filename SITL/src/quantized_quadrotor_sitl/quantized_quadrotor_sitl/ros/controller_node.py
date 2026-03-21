from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from time import perf_counter

import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, VehicleThrustSetpoint, VehicleTorqueSetpoint
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

from ..core.artifacts import coarsen_edmd_model, load_edmd_artifact
from ..core.config import RuntimeConfig, load_runtime_config
from ..edmd.basis import lift_state
from ..experiments.runtime_reference import build_runtime_reference
from ..mpc.qp import get_qp
from ..mpc.simulate import solve_qp
from ..quantization.dither import dither_signal
from ..quantization.partition import partition_range
from ..telemetry.adapter import physical_control_to_px4_wrench
from ..utils.io import ensure_dir
from .offboard import offboard_control_mode_msg


class FlightPhase(str, Enum):
    WAITING_FOR_STATE = "waiting_for_state"
    WARMUP = "warmup"
    WAITING_FOR_ARM = "waiting_for_arm"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("koopman_mpc_controller_node")
        config_path = self.declare_parameter("config_path", "configs/sitl_runtime.yaml").value
        self.config: RuntimeConfig = load_runtime_config(self._resolve_path(config_path))
        base_model, self.metadata = load_edmd_artifact(self._resolve_path(self.config.model_artifact))
        self.mpc_timestep = self.config.mpc.effective_timestep()
        self.mpc_step_multiple = self.config.mpc.runtime_step_multiple()
        self.model = coarsen_edmd_model(base_model, self.mpc_step_multiple)
        if self.mpc_step_multiple > 1:
            self.get_logger().info(
                "Using coarsened EDMD model with "
                f"dt={self.mpc_timestep:.3f}s ({self.mpc_step_multiple} x {self.config.mpc.sim_timestep:.3f}s)"
            )
        self.rng = np.random.default_rng(self.config.reference_seed)
        self.latest_state: np.ndarray | None = None
        self.reference_lifted: np.ndarray | None = None
        self.reference_physical: np.ndarray | None = None
        self.reference_sample_count = 0
        self.step_index = 0
        self.warmup_cycles = 0
        self.arm_retry_counter = 0
        self.armed = False
        self.flight_phase = FlightPhase.WAITING_FOR_STATE
        self.experiment_start_ns: int | None = None
        self.last_active_tick_ns: int | None = None
        self.shutdown_requested = False
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.state_subscription = self.create_subscription(
            Float64MultiArray,
            self.config.state_topic,
            self._handle_state,
            10,
        )
        self.vehicle_status_subscription = self.create_subscription(
            VehicleStatus,
            self.config.vehicle_status_topic,
            self._handle_vehicle_status,
            px4_qos,
        )
        self.state_debug_publisher = self.create_publisher(Float64MultiArray, self.config.state_log_topic, 10)
        self.control_debug_publisher = self.create_publisher(Float64MultiArray, self.config.control_debug_topic, 10)
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            self.config.offboard_control_mode_topic,
            px4_qos,
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            self.config.vehicle_command_topic,
            px4_qos,
        )
        self.vehicle_thrust_publisher = self.create_publisher(
            VehicleThrustSetpoint,
            self.config.vehicle_thrust_setpoint_topic,
            px4_qos,
        )
        self.vehicle_torque_publisher = self.create_publisher(
            VehicleTorqueSetpoint,
            self.config.vehicle_torque_setpoint_topic,
            px4_qos,
        )

        results_dir = ensure_dir(self._resolve_path(self.config.results_dir))
        self.log_path = results_dir / "runtime_log.csv"
        self._log_stream = self.log_path.open("w", encoding="utf-8", newline="")
        self._log_writer = csv.writer(self._log_stream)
        self._log_writer.writerow(
            [
                "step",
                "timestamp_ns",
                "experiment_time_s",
                "reference_index",
                "tick_dt_ms",
                "solver_ms",
                "px4_collective_command_newton",
                "px4_collective_normalized",
                "px4_thrust_body_z",
                *[f"state_raw_{idx}" for idx in range(18)],
                *[f"state_used_{idx}" for idx in range(18)],
                *[f"control_raw_{idx}" for idx in range(4)],
                *[f"control_used_{idx}" for idx in range(4)],
                *[f"reference_{idx}" for idx in range(18)],
            ]
        )

        self.timer = self.create_timer(1.0 / self.config.control_rate_hz, self._control_tick)

    def destroy_node(self):
        self._log_stream.close()
        return super().destroy_node()

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _handle_state(self, msg: Float64MultiArray) -> None:
        self.latest_state = np.asarray(msg.data, dtype=float)
        if self.reference_lifted is None and self.latest_state.size == 18:
            self._initialize_reference(self.latest_state)

    def _handle_vehicle_status(self, msg: VehicleStatus) -> None:
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _initialize_reference(self, state0: np.ndarray) -> None:
        self.reference_physical = build_runtime_reference(
            state0,
            self.config.reference_mode,
            self.config.reference_duration_s,
            self.mpc_timestep,
            self.rng,
        )
        self.reference_sample_count = self.reference_physical.shape[1]
        self.reference_lifted = np.column_stack(
            [lift_state(self.reference_physical[:, idx], self.model.n_basis) for idx in range(self.reference_physical.shape[1])]
        )

    def _reference_window(self, reference_index: int) -> np.ndarray:
        assert self.reference_lifted is not None
        horizon = self.config.mpc.pred_horizon
        end_index = min(reference_index + horizon, self.reference_sample_count)
        window = self.reference_lifted[:, reference_index:end_index]
        if window.shape[1] == horizon:
            return window
        final_column = self.reference_lifted[:, -1:]
        pad_width = horizon - window.shape[1]
        return np.hstack([window, np.repeat(final_column, pad_width, axis=1)])

    def _quantize_vector(self, values: np.ndarray, min_key: str, max_key: str) -> np.ndarray:
        epsilon, min_new, max_new, _, mid_points = partition_range(
            self.metadata[min_key],
            self.metadata[max_key],
            self.config.quantized_word_length,
        )
        quantized, _ = dither_signal(values.reshape(-1, 1), epsilon, min_new, max_new, mid_points, self.rng)
        return quantized[:, 0]

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        if not rclpy.ok():
            return
        msg = VehicleCommand()
        msg.timestamp = self._timestamp_us()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        try:
            self.vehicle_command_publisher.publish(msg)
        except Exception:
            if rclpy.ok():
                raise

    def _publish_offboard_mode(self) -> None:
        if not rclpy.ok():
            return
        try:
            self.offboard_control_mode_publisher.publish(
                offboard_control_mode_msg(OffboardControlMode, self._timestamp_us())
            )
        except Exception:
            if rclpy.ok():
                raise

    def _request_offboard_arm(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        arm_param2 = self.config.force_arm_magic if self.config.force_arm_in_sitl else 0.0
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
            param2=arm_param2,
        )

    def _finish_run(self, phase: FlightPhase, message: str) -> None:
        if self.flight_phase in {FlightPhase.COMPLETED, FlightPhase.FAILED}:
            return
        self.flight_phase = phase
        self._log_stream.flush()
        if phase == FlightPhase.FAILED:
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)
        self.shutdown_requested = True

    def _publish_wrench(self, control_used: np.ndarray) -> tuple[float, float, float]:
        if not rclpy.ok():
            return 0.0, 0.0, 0.0
        thrust_body, normalized_moments, collective_command_newton, collective_normalized = physical_control_to_px4_wrench(
            control_used,
            self.config.vehicle_scaling.max_collective_thrust_newton,
            self.config.vehicle_scaling.torque_scale(),
        )

        thrust_msg = VehicleThrustSetpoint()
        thrust_msg.timestamp = self._timestamp_us()
        thrust_msg.xyz = thrust_body.tolist()
        try:
            self.vehicle_thrust_publisher.publish(thrust_msg)
        except Exception:
            if rclpy.ok():
                raise

        torque_msg = VehicleTorqueSetpoint()
        torque_msg.timestamp = self._timestamp_us()
        torque_msg.xyz = normalized_moments.tolist()
        try:
            self.vehicle_torque_publisher.publish(torque_msg)
        except Exception:
            if rclpy.ok():
                raise
        return collective_command_newton, collective_normalized, float(thrust_body[2])

    def _control_tick(self) -> None:
        if self.shutdown_requested:
            if rclpy.ok():
                rclpy.shutdown()
            return

        now_ns = self.get_clock().now().nanoseconds
        if self.latest_state is None or self.reference_lifted is None or self.reference_physical is None:
            self.flight_phase = FlightPhase.WAITING_FOR_STATE
            return

        if self.flight_phase == FlightPhase.WAITING_FOR_STATE:
            self.flight_phase = FlightPhase.WARMUP

        if self.flight_phase == FlightPhase.WARMUP:
            self._publish_offboard_mode()
            if self.warmup_cycles < self.config.offboard_warmup_cycles:
                self.warmup_cycles += 1
                return
            self.flight_phase = FlightPhase.WAITING_FOR_ARM

        if self.flight_phase == FlightPhase.WAITING_FOR_ARM:
            self._publish_offboard_mode()
            if self.armed:
                self.flight_phase = FlightPhase.ACTIVE
                self.experiment_start_ns = now_ns
                self.last_active_tick_ns = None
            else:
                retry_cycles = max(1, self.config.arm_retry_cycles)
                if self.arm_retry_counter % retry_cycles == 0:
                    self._request_offboard_arm()
                self.arm_retry_counter += 1
                return

        if self.flight_phase != FlightPhase.ACTIVE:
            return

        self._publish_offboard_mode()
        if not self.armed:
            self._finish_run(FlightPhase.FAILED, "Vehicle disarmed before the SITL reference completed.")
            return

        if self.experiment_start_ns is None:
            self.experiment_start_ns = now_ns

        experiment_time_s = max(0.0, (now_ns - self.experiment_start_ns) / 1.0e9)
        reference_index = int(np.floor(experiment_time_s / self.mpc_timestep + 1.0e-9))
        if reference_index >= self.reference_sample_count:
            self._finish_run(FlightPhase.COMPLETED, "Completed the SITL reference window.")
            return

        tick_dt_ms = 0.0 if self.last_active_tick_ns is None else (now_ns - self.last_active_tick_ns) / 1.0e6
        self.last_active_tick_ns = now_ns

        state_raw = self.latest_state.copy()
        state_used = state_raw.copy()
        if self.config.quantization_mode in {"state", "both"} and self.metadata:
            state_used = self._quantize_vector(state_used, "x_train_min", "x_train_max")

        lifted_state = lift_state(state_used, self.model.n_basis)
        lifted_reference = self._reference_window(reference_index)
        solver_start = perf_counter()
        f_vector, g_matrix, a_ineq, b_ineq = get_qp(
            self.model,
            lifted_state,
            lifted_reference,
            self.config.mpc.pred_horizon,
            self.config.mpc,
            self.config.vehicle_scaling.control_lower_bounds(),
            self.config.vehicle_scaling.control_upper_bounds(),
        )
        solution = solve_qp(f_vector, g_matrix, a_ineq, b_ineq)
        solver_ms = (perf_counter() - solver_start) * 1000.0

        control_raw = solution[:4]
        control_used = control_raw.copy()
        if self.config.quantization_mode in {"control", "both"} and self.metadata:
            control_used = self._quantize_vector(control_used, "u_train_min", "u_train_max")

        px4_collective_command_newton, px4_collective_normalized, px4_thrust_body_z = self._publish_wrench(control_used)

        state_debug = Float64MultiArray()
        state_debug.data = state_used.tolist()
        try:
            self.state_debug_publisher.publish(state_debug)
        except Exception:
            if rclpy.ok():
                raise

        control_debug = Float64MultiArray()
        control_debug.data = control_used.tolist()
        try:
            self.control_debug_publisher.publish(control_debug)
        except Exception:
            if rclpy.ok():
                raise

        reference_row = self.reference_physical[:, reference_index]
        self._log_writer.writerow(
            [
                self.step_index,
                now_ns,
                experiment_time_s,
                reference_index,
                tick_dt_ms,
                solver_ms,
                px4_collective_command_newton,
                px4_collective_normalized,
                px4_thrust_body_z,
                *state_raw.tolist(),
                *state_used.tolist(),
                *control_raw.tolist(),
                *control_used.tolist(),
                *reference_row.tolist(),
            ]
        )
        self._log_stream.flush()
        self.step_index += 1

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def main() -> None:
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
