# ROS + PX4 + Gazebo Architecture

## Runtime Data Path

1. PX4 SITL runs the vehicle in Gazebo Harmonic.
2. PX4 publishes `VehicleOdometry` over the ROS 2 bridge.
3. `telemetry_adapter_node` converts PX4 NED/FRD telemetry to the MATLAB-facing state convention used by the parity layer.
4. `controller_node` loads an offline EDMD artifact produced by the parity runner.
5. The controller optionally quantizes the telemetry path, the control path, both, or neither.
6. The controller solves the lifted QP and publishes `VehicleThrustSetpoint` and `VehicleTorqueSetpoint`.
7. The controller logs raw state, quantized state, raw control, quantized control, and the active reference trajectory.

## Quantization Injection

- Offline identification quantization:
  - Applied to the training state snapshots and control snapshots before EDMD fitting.
  - Implemented from the same partition/dither/quantize/subtract sequence as MATLAB.
- Runtime telemetry quantization:
  - Applied after PX4 telemetry is converted into the MATLAB state convention.
- Runtime control quantization:
  - Applied after the QP solve and before mapping the wrench command into PX4 normalized thrust/torque setpoints.

## Coordinate Handling

- PX4 telemetry arrives in NED/FRD.
- The telemetry adapter converts it to ENU/FLU before forming the MATLAB-style state vector.
- The controller computes force/moment commands in the MATLAB-facing body frame.
- The runtime mapper converts those commands back into PX4 body-frame normalized thrust/torque setpoints.

## Artifact Flow

- Offline runner outputs `edmd_unquantized.npz` and `edmd_bits_XX_run_YY.npz`.
- Those artifacts include:
  - `A`, `B`, `C`, `Z1`, `Z2`, `n_basis`
  - training-data min/max ranges for state and control quantizers
- The SITL controller loads one artifact and uses the saved ranges for runtime quantization.

