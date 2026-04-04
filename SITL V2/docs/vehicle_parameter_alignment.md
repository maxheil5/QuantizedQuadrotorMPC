# Vehicle Parameter Alignment

This note locks down the first V2 decision that matters for the learned
controller path: preserve the MATLAB **structure**, but align the active
vehicle **physics constants** with the Gazebo model we actually fly.

## Decision

- Keep MATLAB as the algorithmic source of truth for SRB state and control
  conventions.
- Use the RotorS Firefly model as the default physical parameter source for V2.
- Preserve the original MATLAB parameter set explicitly for offline reference
  and thesis traceability.

## Why this matters

The MATLAB project was built around a much heavier quadrotor model than the
RotorS Firefly we are flying in Gazebo. If we port the learned controller math
 but keep stale PX4-era physical constants, we introduce a model mismatch before
 the learned EDMD pipeline even starts.

## Parameter profiles

### MATLAB reference profile

Source: `MATLAB/dynamics/get_params.m`

- mass: `4.34 kg`
- inertia diag: `[0.0820, 0.0845, 0.1377] kg m^2`
- gravity: `9.81 m/s^2`

### RotorS Firefly model profile

Source: `ethz-asl/rotors_simulator`, `rotors_description/urdf/firefly.xacro`

- mass: `1.5 kg`
- inertia diag: `[0.0347563, 0.0458929, 0.0977] kg m^2`
- arm length: `0.215 m`
- motor constant: `8.54858e-06`
- moment constant: `0.016`
- gravity: `9.81 m/s^2`

### RotorS verified linear-MPC runtime profile

Source: verified from the live `/firefly` ROS parameter tree on Ubuntu

- mass: `1.52 kg`
- inertia diag: `[0.034756, 0.045893, 0.0977] kg m^2`
- arm length: `0.2156 m`
- motor constant: `8.54858e-06`
- moment constant: `0.016`
- drag coefficients: `[0.01, 0.01, 0.0]`
- gravity: `9.81 m/s^2`

## Active V2 default

The active default for `koopman_python.dynamics.params.get_params()` is:

- `rotors_firefly_linear_mpc_runtime`

That is the safest default for the fresh learned-controller port because it
matches the live Gazebo + `mav_linear_mpc` baseline we already brought up on
Ubuntu.

## Immediate follow-up on Ubuntu

Before we start online learned-controller integration, verify which runtime
parameter profile is actually active in the controller stack. In the
`mav_control_rw` path, mass and inertia may live under different namespaces, so
query the whole `/firefly` tree instead of assuming a single controller node:

```bash
source /opt/ros/noetic/setup.bash
cd ~/thesis_v2_ws
source devel/setup.bash
bash ~/thesis_v2_ws/QuantizedQuadrotorMPC/"SITL V2"/scripts/check_firefly_runtime_params.sh \
  /firefly \
  ~/thesis_v2_ws/QuantizedQuadrotorMPC/"SITL V2"/results/summary/firefly_runtime_params.txt
```

We still keep the pure xacro and historical controller profiles explicit in V2,
but the verified runtime profile is now the default for learned-controller
development so we do not accidentally train against one vehicle and fly another.
