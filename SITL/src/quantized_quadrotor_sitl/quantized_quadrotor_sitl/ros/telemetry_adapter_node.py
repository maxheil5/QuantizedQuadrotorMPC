from __future__ import annotations

import numpy as np
import rclpy
from px4_msgs.msg import VehicleAngularVelocity, VehicleAttitude, VehicleLocalPosition, VehicleOdometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

from ..core.config import RuntimeConfig, load_runtime_config
from ..telemetry.adapter import vehicle_odometry_to_state18


class TelemetryAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("telemetry_adapter_node")
        config_path = self.declare_parameter("config_path", "configs/sitl_runtime.yaml").value
        self.config: RuntimeConfig = load_runtime_config(self._resolve_path(config_path))
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(Float64MultiArray, self.config.state_topic, 10)
        self._position_ned: np.ndarray | None = None
        self._velocity_ned: np.ndarray | None = None
        self._quaternion_wxyz_ned_frd: np.ndarray | None = None
        self._angular_velocity_frd: np.ndarray | None = None
        self.odometry_subscription = self.create_subscription(
            VehicleOdometry,
            self.config.vehicle_odometry_topic,
            self._handle_vehicle_odometry,
            px4_qos,
        )
        self.local_position_subscription = self.create_subscription(
            VehicleLocalPosition,
            self.config.vehicle_local_position_topic,
            self._handle_vehicle_local_position,
            px4_qos,
        )
        self.attitude_subscription = self.create_subscription(
            VehicleAttitude,
            self.config.vehicle_attitude_topic,
            self._handle_vehicle_attitude,
            px4_qos,
        )
        self.angular_velocity_subscription = self.create_subscription(
            VehicleAngularVelocity,
            self.config.vehicle_angular_velocity_topic,
            self._handle_vehicle_angular_velocity,
            px4_qos,
        )

    def _resolve_path(self, config_path: str):
        from pathlib import Path

        path = Path(config_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _handle_vehicle_odometry(self, msg: VehicleOdometry) -> None:
        state = vehicle_odometry_to_state18(
            np.array(msg.position, dtype=float),
            np.array(msg.velocity, dtype=float),
            np.array(msg.q, dtype=float),
            np.array(msg.angular_velocity, dtype=float),
        )
        self._publish_state(state)

    def _handle_vehicle_local_position(self, msg: VehicleLocalPosition) -> None:
        self._position_ned = np.array([msg.x, msg.y, msg.z], dtype=float)
        self._velocity_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
        self._publish_estimator_state_if_ready()

    def _handle_vehicle_attitude(self, msg: VehicleAttitude) -> None:
        self._quaternion_wxyz_ned_frd = np.array(msg.q, dtype=float)
        self._publish_estimator_state_if_ready()

    def _handle_vehicle_angular_velocity(self, msg: VehicleAngularVelocity) -> None:
        self._angular_velocity_frd = np.array(msg.xyz, dtype=float)
        self._publish_estimator_state_if_ready()

    def _publish_estimator_state_if_ready(self) -> None:
        if (
            self._position_ned is None
            or self._velocity_ned is None
            or self._quaternion_wxyz_ned_frd is None
            or self._angular_velocity_frd is None
        ):
            return
        state = vehicle_odometry_to_state18(
            self._position_ned,
            self._velocity_ned,
            self._quaternion_wxyz_ned_frd,
            self._angular_velocity_frd,
        )
        self._publish_state(state)

    def _publish_state(self, state: np.ndarray) -> None:
        if not rclpy.ok():
            return
        payload = Float64MultiArray()
        payload.data = state.tolist()
        try:
            self.publisher.publish(payload)
        except Exception:
            if rclpy.ok():
                raise


def main() -> None:
    rclpy.init()
    node = TelemetryAdapterNode()
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
