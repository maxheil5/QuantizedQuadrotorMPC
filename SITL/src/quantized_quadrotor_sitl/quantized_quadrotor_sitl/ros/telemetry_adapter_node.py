from __future__ import annotations

import numpy as np
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ..core.config import RuntimeConfig, load_runtime_config
from ..telemetry.adapter import vehicle_odometry_to_state18


class TelemetryAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("telemetry_adapter_node")
        config_path = self.declare_parameter("config_path", "configs/sitl_runtime.yaml").value
        self.config: RuntimeConfig = load_runtime_config(self._resolve_path(config_path))
        self.publisher = self.create_publisher(Float64MultiArray, self.config.state_topic, 10)
        self.subscription = self.create_subscription(
            VehicleOdometry,
            self.config.vehicle_odometry_topic,
            self._handle_vehicle_odometry,
            10,
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
        payload = Float64MultiArray()
        payload.data = state.tolist()
        self.publisher.publish(payload)


def main() -> None:
    rclpy.init()
    node = TelemetryAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
