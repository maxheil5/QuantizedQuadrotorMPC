from __future__ import annotations

import csv
import os
from dataclasses import asdict
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

from ..controllers import compute_baseline_control
from ..core.artifacts import load_edmd_artifact
from ..core.config import RuntimeConfig, load_runtime_config
from ..dynamics.params import get_params
from ..edmd.basis import lift_state
from ..experiments.runtime_reference import build_runtime_reference
from ..mpc.qp import get_qp
from ..mpc.simulate import solve_qp
from ..quantization.dither import dither_signal
from ..quantization.partition import partition_range
from ..telemetry.adapter import physical_control_to_px4_wrench
from ..utils.control_anchor import apply_moment_authority_anchor
from ..utils.control_bounds import RuntimeControlCoordinates, runtime_edmd_control_coordinates
from ..utils.host import runtime_host_snapshot
from ..utils.io import create_sitl_results_directory, write_json
from ..utils.state import state18_history_to_hover_local_residual, state18_to_hover_local_residual, takeoff_hold_trim_state18
from ..utils.state import hover_local_translation_rotated
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
        self.config_path = self._resolve_path(config_path)
        self.config: RuntimeConfig = load_runtime_config(self.config_path)
        self.params = get_params()
        self.model = None
        self.metadata: dict[str, object] = {}
        self.control_coordinates: RuntimeControlCoordinates | None = None
        self.residual_enabled = False
        self.residual_rotate_translation = False
        self.runtime_state_trim: np.ndarray | None = None
        if self.config.controller_mode == "edmd_mpc":
            self.model, self.metadata = load_edmd_artifact(self._resolve_path(self.config.model_artifact))
            self.control_coordinates = runtime_edmd_control_coordinates(
                self.config.vehicle_scaling,
                self.metadata,
                learned_bound_margin_fraction=self.config.learned_bound_margin_fraction,
            )
            self.residual_enabled = bool(self.metadata.get("residual_enabled", False))
            self.residual_rotate_translation = hover_local_translation_rotated(self.metadata.get("state_coordinates"))
            self.get_logger().info(
                f"Using controller mode '{self.config.controller_mode}' with artifact {self.config.model_artifact}"
            )
            if self.model.affine_enabled:
                self.get_logger().info(
                    "Affine EDMD bias is enabled for runtime prediction."
                )
            if self.control_coordinates.normalized:
                self.get_logger().info(
                    "Using trim-centered normalized EDMD controls with "
                    f"u_trim={np.array2string(self.control_coordinates.trim, precision=3)}, "
                    f"u_train_std={np.array2string(self.control_coordinates.scale, precision=3)}."
                )
                self.get_logger().info(
                    "Effective physical control bounds "
                    f"{np.array2string(self.control_coordinates.physical_lower_bounds, precision=3)} to "
                    f"{np.array2string(self.control_coordinates.physical_upper_bounds, precision=3)}."
                )
                self.get_logger().info(
                    "Effective normalized control bounds "
                    f"{np.array2string(self.control_coordinates.internal_lower_bounds, precision=3)} to "
                    f"{np.array2string(self.control_coordinates.internal_upper_bounds, precision=3)}."
                )
                if self.residual_enabled:
                    self.get_logger().info(
                        f"Applying a {100.0 * self.config.learned_bound_margin_fraction:.1f}% inward margin to learned residual control bounds."
                    )
            if self.residual_enabled:
                self.get_logger().info(
                    "Using takeoff-hold hover-local residual state coordinates for runtime EDMD prediction."
                )
                if self.residual_rotate_translation:
                    self.get_logger().info(
                        "Residual translation states are rotated into the trim frame for this artifact."
                    )
            elif self.control_coordinates.physical_lower_bounds[0] > self.config.vehicle_scaling.control_lower_bounds()[0]:
                self.get_logger().info(
                    f"Using learned collective floor {self.control_coordinates.physical_lower_bounds[0]:.2f} N from artifact metadata."
                )
            if self.config.moment_authority_anchor.enabled:
                self.get_logger().info(
                    "Moment authority anchor enabled with "
                    f"minimum_baseline_fraction={self.config.moment_authority_anchor.minimum_baseline_fraction:.2f} "
                    f"and active_thresholds_nm={self.config.moment_authority_anchor.active_thresholds_nm}."
                )
        elif self.config.controller_mode == "baseline_geometric":
            self.get_logger().info(
                "Using controller mode 'baseline_geometric' for a SITL-only hover sanity check."
            )
        else:
            raise ValueError(f"unsupported controller mode: {self.config.controller_mode}")
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
        self.baseline_z_error_integral = 0.0
        self.previous_control_internal = np.zeros(4, dtype=float)
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

        run_seed_suffix = int(self.config.reference_seed) if self.config.reference_mode.startswith("sitl_identification") else None
        results_dir = create_sitl_results_directory(
            self._resolve_path(self.config.results_dir),
            seed_suffix=run_seed_suffix,
        )
        self.log_path = results_dir / "runtime_log.csv"
        self.metadata_path = results_dir / "run_metadata.json"
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
                *[f"control_internal_{idx}" for idx in range(4)],
                *[f"control_used_{idx}" for idx in range(4)],
                *[f"reference_{idx}" for idx in range(18)],
            ]
        )
        self._write_run_metadata()

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
            self.config.mpc.sim_timestep,
            self.rng,
        )
        self.reference_sample_count = self.reference_physical.shape[1]
        if self.config.controller_mode == "edmd_mpc":
            assert self.model is not None
            if self.residual_enabled:
                self.runtime_state_trim = takeoff_hold_trim_state18(state0)
                reference_model_state = state18_history_to_hover_local_residual(
                    self.reference_physical,
                    self.runtime_state_trim,
                    rotate_translation=self.residual_rotate_translation,
                )
            else:
                self.runtime_state_trim = None
                reference_model_state = self.reference_physical
            self.reference_lifted = np.column_stack(
                [lift_state(reference_model_state[:, idx], self.model.n_basis) for idx in range(reference_model_state.shape[1])]
            )
        else:
            self.reference_lifted = np.empty((0, 0), dtype=float)
        self._write_run_metadata(state0)

    def _write_run_metadata(self, initial_state: np.ndarray | None = None) -> None:
        payload: dict[str, object] = {
            "config_path": str(self.config_path),
            "controller_mode": self.config.controller_mode,
            "control_rate_hz": float(self.config.control_rate_hz),
            "reference_mode": self.config.reference_mode,
            "reference_seed": int(self.config.reference_seed),
            "reference_duration_s": float(self.config.reference_duration_s),
            "cost_state_mode": self.config.mpc.cost_state_mode,
            "pred_horizon": int(self.config.mpc.pred_horizon),
            "model_artifact": self.config.model_artifact,
            "model_affine_enabled": bool(self.model.affine_enabled) if self.model is not None else False,
            "model_residual_enabled": bool(self.residual_enabled),
            "quantization_mode": self.config.quantization_mode,
            "learned_bound_margin_fraction": float(self.config.learned_bound_margin_fraction),
            "baseline": asdict(self.config.baseline),
            "moment_authority_anchor": asdict(self.config.moment_authority_anchor),
            "vehicle_scaling": asdict(self.config.vehicle_scaling),
            "run_dir": str(self.log_path.parent),
            "log_path": str(self.log_path),
            "metadata_path": str(self.metadata_path),
            "host_snapshot": runtime_host_snapshot(headless=os.getenv("HEADLESS", "0") == "1"),
        }
        if self.control_coordinates is not None:
            payload["control_coordinates"] = {
                "normalized": bool(self.control_coordinates.normalized),
                "learned_bound_margin_fraction": float(self.config.learned_bound_margin_fraction),
                "trim": self.control_coordinates.trim.tolist(),
                "scale": self.control_coordinates.scale.tolist(),
                "physical_lower_bounds": self.control_coordinates.physical_lower_bounds.tolist(),
                "physical_upper_bounds": self.control_coordinates.physical_upper_bounds.tolist(),
                "internal_lower_bounds": self.control_coordinates.internal_lower_bounds.tolist(),
                "internal_upper_bounds": self.control_coordinates.internal_upper_bounds.tolist(),
            }
        if self.residual_enabled:
            payload["state_coordinates"] = {
                "residual_enabled": True,
                "state_coordinates": str(self.metadata.get("state_coordinates", "takeoff_hold_hover_local")),
                "state_trim_mode": str(self.metadata.get("state_trim_mode", "per_run_takeoff_hold_final")),
                "artifact_state_trim": []
                if "state_trim" not in self.metadata
                else np.asarray(self.metadata["state_trim"], dtype=float).reshape(18).tolist(),
                "runtime_state_trim": []
                if self.runtime_state_trim is None
                else np.asarray(self.runtime_state_trim, dtype=float).reshape(18).tolist(),
            }
        if initial_state is not None:
            payload["initial_state"] = np.asarray(initial_state, dtype=float).reshape(18).tolist()
        if self.reference_sample_count > 0:
            payload["reference_sample_count"] = int(self.reference_sample_count)
        write_json(self.metadata_path, payload)

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
        self.previous_control_internal = np.zeros(4, dtype=float)
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

    def _update_baseline_z_error_integral(self, state_used: np.ndarray, reference_row: np.ndarray, tick_dt_s: float) -> None:
        z_error = float(reference_row[2] - state_used[2])
        self.baseline_z_error_integral += z_error * tick_dt_s
        self.baseline_z_error_integral = float(
            np.clip(
                self.baseline_z_error_integral,
                -self.config.baseline.z_integral_limit,
                self.config.baseline.z_integral_limit,
            )
        )

    def _control_tick(self) -> None:
        if self.shutdown_requested:
            if rclpy.ok():
                rclpy.shutdown()
            return

        now_ns = self.get_clock().now().nanoseconds
        if self.latest_state is None or self.reference_physical is None:
            self.flight_phase = FlightPhase.WAITING_FOR_STATE
            return
        if self.config.controller_mode == "edmd_mpc" and self.reference_lifted is None:
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
                self.baseline_z_error_integral = 0.0
                self.previous_control_internal = np.zeros(4, dtype=float)
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
        reference_index = int(np.floor(experiment_time_s / self.config.mpc.sim_timestep + 1.0e-9))
        if reference_index >= self.reference_sample_count:
            self._finish_run(FlightPhase.COMPLETED, "Completed the SITL reference window.")
            return

        tick_dt_ms = 0.0 if self.last_active_tick_ns is None else (now_ns - self.last_active_tick_ns) / 1.0e6
        self.last_active_tick_ns = now_ns
        tick_dt_s = (1.0 / self.config.control_rate_hz) if tick_dt_ms <= 0.0 else (tick_dt_ms / 1000.0)

        state_raw = self.latest_state.copy()
        state_used = state_raw.copy()

        reference_row = self.reference_physical[:, reference_index]
        baseline_anchor_control_used: np.ndarray | None = None
        if self.config.controller_mode == "baseline_geometric" or self.config.moment_authority_anchor.enabled:
            self._update_baseline_z_error_integral(state_used, reference_row, tick_dt_s)
        if self.config.controller_mode == "edmd_mpc":
            assert self.model is not None
            assert self.control_coordinates is not None
            if self.residual_enabled:
                assert self.runtime_state_trim is not None
                model_state = state18_to_hover_local_residual(
                    state_used,
                    self.runtime_state_trim,
                    rotate_translation=self.residual_rotate_translation,
                )
                if self.config.quantization_mode in {"state", "both"} and self.metadata:
                    model_state = self._quantize_vector(model_state, "x_train_min", "x_train_max")
            else:
                if self.config.quantization_mode in {"state", "both"} and self.metadata:
                    state_used = self._quantize_vector(state_used, "x_train_min", "x_train_max")
                model_state = state_used
            lifted_state = lift_state(model_state, self.model.n_basis)
            lifted_reference = self._reference_window(reference_index)
            solver_start = perf_counter()
            f_vector, g_matrix, a_ineq, b_ineq = get_qp(
                self.model,
                lifted_state,
                lifted_reference,
                self.config.mpc.pred_horizon,
                self.config.mpc,
                self.control_coordinates.internal_lower_bounds,
                self.control_coordinates.internal_upper_bounds,
                previous_control=self.previous_control_internal,
            )
            solution = solve_qp(f_vector, g_matrix, a_ineq, b_ineq)
            solver_ms = (perf_counter() - solver_start) * 1000.0

            control_internal = solution[:4]
            control_raw = self.control_coordinates.internal_to_physical(control_internal)
            control_used = control_raw.copy()
            if self.config.quantization_mode in {"control", "both"} and self.metadata:
                control_used = self._quantize_vector(control_used, "u_train_min", "u_train_max")
            if self.config.moment_authority_anchor.enabled:
                _, baseline_anchor_control_used = compute_baseline_control(
                    state_used,
                    reference_row,
                    self.baseline_z_error_integral,
                    self.config.baseline,
                    self.config.vehicle_scaling,
                    self.params,
                )
                control_used = apply_moment_authority_anchor(
                    control_used,
                    baseline_anchor_control_used,
                    self.control_coordinates.physical_lower_bounds,
                    self.control_coordinates.physical_upper_bounds,
                    self.config.moment_authority_anchor,
                )
            self.previous_control_internal = np.clip(
                self.control_coordinates.physical_to_internal(control_used),
                self.control_coordinates.internal_lower_bounds,
                self.control_coordinates.internal_upper_bounds,
            )
        else:
            solver_start = perf_counter()
            control_raw, control_used = compute_baseline_control(
                state_used,
                reference_row,
                self.baseline_z_error_integral,
                self.config.baseline,
                self.config.vehicle_scaling,
                self.params,
            )
            solver_ms = (perf_counter() - solver_start) * 1000.0
            control_internal = control_raw.copy()

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
                *control_internal.tolist(),
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
