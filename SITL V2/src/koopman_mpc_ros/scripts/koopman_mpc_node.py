#!/usr/bin/env python3
"""ROS1 learned-controller node for the first hover-only SITL V2 bring-up."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from koopman_python.dynamics.srb import pack_state, unpack_state
from koopman_python.mpc import (
    LearnedMpcController,
    LearnedMpcRuntimeConfig,
    build_constant_reference_horizon,
    control_to_roll_pitch_yawrate_thrust,
    load_edmd_model,
)


def _quaternion_xyzw_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


@dataclass
class PoseReference:
    position_xyz: np.ndarray
    yaw_rad: float


class KoopmanMpcRosNode:
    def __init__(self) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from mav_msgs.msg import RollPitchYawrateThrust
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import MultiDOFJointTrajectory

        self.rospy = rospy
        self.RollPitchYawrateThrust = RollPitchYawrateThrust
        self.Float64MultiArray = Float64MultiArray

        model_path = Path(rospy.get_param("~model_path", "")).expanduser()
        if not model_path:
            raise SystemExit("~model_path must point to an offline learned model.npz file.")

        runtime_config = LearnedMpcRuntimeConfig(
            pred_horizon=int(rospy.get_param("~pred_horizon", 10)),
            control_lower_bound=float(rospy.get_param("~control_lower_bound", -50.0)),
            control_upper_bound=float(rospy.get_param("~control_upper_bound", 50.0)),
            qp_max_iter=int(rospy.get_param("~qp_max_iter", 100)),
            qp_tol=float(rospy.get_param("~qp_tol", 1e-8)),
            parameter_profile=str(rospy.get_param("~parameter_profile", "rotors_firefly_linear_mpc_runtime")),
            control_time_step=float(rospy.get_param("~control_time_step", 0.01)),
            max_roll_pitch_rad=float(rospy.get_param("~max_roll_pitch_rad", 0.35)),
        )
        self.controller = LearnedMpcController(
            model=load_edmd_model(model_path),
            config=runtime_config,
        )
        self.altitude_assist_kp = float(rospy.get_param("~altitude_assist_kp", 7.0))
        self.altitude_assist_kd = float(rospy.get_param("~altitude_assist_kd", 6.0))
        self.altitude_assist_ki = float(rospy.get_param("~altitude_assist_ki", 1.2))
        self.altitude_assist_max_delta_newton = float(
            rospy.get_param("~altitude_assist_max_delta_newton", 14.0)
        )
        self.altitude_assist_integral_max_delta_newton = float(
            rospy.get_param("~altitude_assist_integral_max_delta_newton", 3.0)
        )
        self.takeoff_altitude_error_threshold_m = float(
            rospy.get_param("~takeoff_altitude_error_threshold_m", 0.2)
        )
        self.takeoff_vertical_speed_threshold_mps = float(
            rospy.get_param("~takeoff_vertical_speed_threshold_mps", 0.5)
        )
        self.takeoff_boost_end_fraction = float(
            rospy.get_param("~takeoff_boost_end_fraction", 0.65)
        )
        self.takeoff_min_thrust_newton = float(
            rospy.get_param(
                "~takeoff_min_thrust_newton",
                max(float(self.controller.params["mass"]) * 9.81 + 1.5, 19.0),
            )
        )
        self.command_thrust_max_newton = float(
            rospy.get_param("~command_thrust_max_newton", 22.0)
        )
        self.command_thrust_slew_rate_newton_per_s = float(
            rospy.get_param("~command_thrust_slew_rate_newton_per_s", 30.0)
        )
        self.hover_xy_position_kp = float(rospy.get_param("~hover_xy_position_kp", 0.08))
        self.hover_xy_velocity_kd = float(rospy.get_param("~hover_xy_velocity_kd", 0.65))
        self.hover_xy_command_blend = float(rospy.get_param("~hover_xy_command_blend", 0.05))
        self.hover_xy_max_roll_pitch_rad = float(
            rospy.get_param("~hover_xy_max_roll_pitch_rad", 0.10)
        )
        self.altitude_error_integral = 0.0
        self.last_thrust_newton: float | None = None
        self.reference: PoseReference | None = None
        self.current_state: np.ndarray | None = None

        self.command_pub = rospy.Publisher("command/roll_pitch_yawrate_thrust", RollPitchYawrateThrust, queue_size=1)
        self.raw_control_pub = rospy.Publisher("command/raw_body_wrench", Float64MultiArray, queue_size=1)
        self.state_sub = rospy.Subscriber("odometry", Odometry, self._handle_odometry, queue_size=1)
        self.pose_sub = rospy.Subscriber("command/pose", PoseStamped, self._handle_pose, queue_size=1)
        self.traj_sub = rospy.Subscriber("command/trajectory", MultiDOFJointTrajectory, self._handle_trajectory, queue_size=1)
        publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 100.0))
        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / publish_rate_hz), self._timer_tick)

        rospy.loginfo("koopman_mpc_node ready with hover-first learned MPC runtime.")
        rospy.loginfo("Using model_path=%s", str(model_path))

    def _handle_odometry(self, msg) -> None:
        position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        velocity = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=float,
        )
        quaternion = msg.pose.pose.orientation
        rotation = _quaternion_xyzw_to_rotation_matrix(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        angular_velocity = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
            dtype=float,
        )
        self.current_state = pack_state(
            position=position,
            velocity=velocity,
            rotation=rotation,
            angular_velocity=angular_velocity,
        )

    def _handle_pose(self, msg) -> None:
        quaternion = msg.pose.orientation
        yaw = _yaw_from_quaternion_xyzw(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.altitude_error_integral = 0.0
        self.reference = PoseReference(
            position_xyz=np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=float,
            ),
            yaw_rad=yaw,
        )

    def _handle_trajectory(self, msg) -> None:
        if not msg.points or not msg.points[0].transforms:
            return
        transform = msg.points[0].transforms[0]
        quaternion = transform.rotation
        yaw = _yaw_from_quaternion_xyzw(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.altitude_error_integral = 0.0
        self.reference = PoseReference(
            position_xyz=np.array(
                [transform.translation.x, transform.translation.y, transform.translation.z],
                dtype=float,
            ),
            yaw_rad=yaw,
        )

    def _timer_tick(self, _event) -> None:
        if self.current_state is None or self.reference is None:
            return

        reference_horizon = build_constant_reference_horizon(
            position_xyz=self.reference.position_xyz,
            yaw_rad=self.reference.yaw_rad,
            horizon=self.controller.config.pred_horizon,
        )
        step = self.controller.solve(self.current_state, reference_horizon)
        command = control_to_roll_pitch_yawrate_thrust(
            current_state=self.current_state,
            control=step.control,
            parameter_profile=self.controller.config.parameter_profile,
            control_time_step=self.controller.config.control_time_step,
            max_roll_pitch_rad=self.controller.config.max_roll_pitch_rad,
        )
        position, velocity, _rotation, _angular_velocity = unpack_state(self.current_state)
        xy_error = self.reference.position_xyz[:2] - position[:2]
        xy_velocity_error = -velocity[:2]
        z_error = float(self.reference.position_xyz[2] - position[2])
        z_velocity = float(velocity[2])
        pitch_correction = float(
            np.clip(
                self.hover_xy_position_kp * xy_error[0] + self.hover_xy_velocity_kd * xy_velocity_error[0],
                -self.hover_xy_max_roll_pitch_rad,
                self.hover_xy_max_roll_pitch_rad,
            )
        )
        roll_correction = float(
            np.clip(
                -(self.hover_xy_position_kp * xy_error[1] + self.hover_xy_velocity_kd * xy_velocity_error[1]),
                -self.hover_xy_max_roll_pitch_rad,
                self.hover_xy_max_roll_pitch_rad,
            )
        )
        dt = self.controller.config.control_time_step
        self.altitude_error_integral = float(
            np.clip(
                self.altitude_error_integral + z_error * dt,
                -self.altitude_assist_integral_max_delta_newton / max(self.altitude_assist_ki, 1e-6),
                self.altitude_assist_integral_max_delta_newton / max(self.altitude_assist_ki, 1e-6),
            )
        )
        thrust_assist = float(
            np.clip(
                self.altitude_assist_kp * z_error
                - self.altitude_assist_kd * z_velocity
                + self.altitude_assist_ki * self.altitude_error_integral,
                -self.altitude_assist_max_delta_newton,
                self.altitude_assist_max_delta_newton,
            )
        )
        thrust_newton = command.thrust_newton + thrust_assist
        if (
            z_error >= self.takeoff_altitude_error_threshold_m
            and position[2] <= self.reference.position_xyz[2] * self.takeoff_boost_end_fraction
            and z_velocity <= self.takeoff_vertical_speed_threshold_mps
        ):
            thrust_newton = max(thrust_newton, self.takeoff_min_thrust_newton)
        thrust_newton = float(np.clip(thrust_newton, 0.0, self.command_thrust_max_newton))
        if self.last_thrust_newton is None:
            self.last_thrust_newton = thrust_newton
        else:
            max_thrust_step_newton = self.command_thrust_slew_rate_newton_per_s * dt
            thrust_newton = float(
                np.clip(
                    thrust_newton,
                    self.last_thrust_newton - max_thrust_step_newton,
                    self.last_thrust_newton + max_thrust_step_newton,
                )
            )
            self.last_thrust_newton = thrust_newton
        roll_rad = float(
            np.clip(
                self.hover_xy_command_blend * command.roll_rad
                + (1.0 - self.hover_xy_command_blend) * roll_correction,
                -self.hover_xy_max_roll_pitch_rad,
                self.hover_xy_max_roll_pitch_rad,
            )
        )
        pitch_rad = float(
            np.clip(
                self.hover_xy_command_blend * command.pitch_rad
                + (1.0 - self.hover_xy_command_blend) * pitch_correction,
                -self.hover_xy_max_roll_pitch_rad,
                self.hover_xy_max_roll_pitch_rad,
            )
        )

        msg = self.RollPitchYawrateThrust()
        msg.header.stamp = self.rospy.Time.now()
        msg.roll = roll_rad
        msg.pitch = pitch_rad
        msg.yaw_rate = command.yaw_rate_rad_s
        msg.thrust.x = 0.0
        msg.thrust.y = 0.0
        msg.thrust.z = thrust_newton
        self.command_pub.publish(msg)

        raw = self.Float64MultiArray()
        raw.data = [
            float(step.control[0]),
            float(step.control[1]),
            float(step.control[2]),
            float(step.control[3]),
            float(thrust_assist),
            float(thrust_newton),
            float(roll_correction),
            float(pitch_correction),
            float(roll_rad),
            float(pitch_rad),
            float(xy_error[0]),
            float(xy_error[1]),
            float(velocity[0]),
            float(velocity[1]),
            float(z_error),
            float(z_velocity),
            float(step.solve_time_ms),
            float(step.solve_iterations),
            1.0 if step.solve_converged else 0.0,
            float(self.altitude_error_integral),
        ]
        self.raw_control_pub.publish(raw)


def main() -> None:
    try:
        import rospy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"rospy is required for koopman_mpc_ros: {exc}")

    rospy.init_node("koopman_mpc_node")
    node = KoopmanMpcRosNode()
    rospy.spin()
    del node


if __name__ == "__main__":
    main()
