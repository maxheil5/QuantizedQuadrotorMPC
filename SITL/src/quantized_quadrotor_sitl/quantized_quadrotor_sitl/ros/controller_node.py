from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, VehicleThrustSetpoint, VehicleTorqueSetpoint
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

from ..core.artifacts import load_edmd_artifact
from ..core.config import RuntimeConfig, load_runtime_config
from ..edmd.basis import lift_state
from ..experiments.training_data import get_random_trajectories
from ..mpc.qp import get_qp
from ..mpc.simulate import solve_qp
from ..quantization.dither import dither_signal
from ..quantization.partition import partition_range
from ..telemetry.adapter import physical_control_to_px4_wrench
from ..utils.io import ensure_dir
from .offboard import offboard_control_mode_msg


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("koopman_mpc_controller_node")
        config_path = self.declare_parameter("config_path", "configs/sitl_runtime.yaml").value
        self.config: RuntimeConfig = load_runtime_config(self._resolve_path(config_path))
        self.model, self.metadata = load_edmd_artifact(self._resolve_path(self.config.model_artifact))
        self.rng = np.random.default_rng(self.config.reference_seed)
        self.latest_state: np.ndarray | None = None
        self.reference_lifted: np.ndarray | None = None
        self.reference_physical: np.ndarray | None = None
        self.step_index = 0
        self.warmup_cycles = 0
        self.arm_retry_counter = 0
        self.armed = False
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
        t_ref = np.arange(0.0, self.config.reference_duration_s + self.config.mpc.sim_timestep, self.config.mpc.sim_timestep)
        x_ref, _, _, _, _, _ = get_random_trajectories(state0, 1, t_ref, "mpc", self.rng)
        self.reference_physical = x_ref[:, 1:]
        self.reference_lifted = np.column_stack(
            [lift_state(self.reference_physical[:, idx], self.model.n_basis) for idx in range(self.reference_physical.shape[1])]
        )

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

    def _publish_wrench(self, control_used: np.ndarray) -> None:
        if not rclpy.ok():
            return
        thrust_body, normalized_moments = physical_control_to_px4_wrench(
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

    def _control_tick(self) -> None:
        self._publish_offboard_mode()
        if self.latest_state is None or self.reference_lifted is None or self.reference_physical is None:
            return

        if self.warmup_cycles < self.config.offboard_warmup_cycles:
            self.warmup_cycles += 1
            return

        if not self.armed:
            retry_cycles = max(1, self.config.arm_retry_cycles)
            if self.arm_retry_counter % retry_cycles == 0:
                self._request_offboard_arm()
            self.arm_retry_counter += 1
            return

        if self.step_index + self.config.mpc.pred_horizon >= self.reference_lifted.shape[1]:
            return

        state_raw = self.latest_state.copy()
        state_used = state_raw.copy()
        if self.config.quantization_mode in {"state", "both"} and self.metadata:
            state_used = self._quantize_vector(state_used, "x_train_min", "x_train_max")

        lifted_state = lift_state(state_used, self.model.n_basis)
        lifted_reference = self.reference_lifted[:, self.step_index : self.step_index + self.config.mpc.pred_horizon]
        f_vector, g_matrix, a_ineq, b_ineq = get_qp(self.model, lifted_state, lifted_reference, self.config.mpc.pred_horizon, self.config.mpc)
        solution = solve_qp(f_vector, g_matrix, a_ineq, b_ineq)
        control_raw = solution[:4]
        control_used = control_raw.copy()
        if self.config.quantization_mode in {"control", "both"} and self.metadata:
            control_used = self._quantize_vector(control_used, "u_train_min", "u_train_max")

        self._publish_wrench(control_used)

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

        reference_row = self.reference_physical[:, self.step_index]
        self._log_writer.writerow(
            [
                self.step_index,
                self.get_clock().now().nanoseconds,
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
