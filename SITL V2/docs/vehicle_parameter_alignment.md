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

### RotorS runtime controller profile

Source: Firefly runtime parameter dump from the RotorS Lee position controller

- mass: `1.56779 kg`
- inertia diag: `[0.0347563, 0.0458929, 0.0977] kg m^2`
- arm length: `0.215 m`
- motor constant: `8.54858e-06`
- moment constant: `0.016`
- gravity: `9.81 m/s^2`

## Active V2 default

The active default for `koopman_python.dynamics.params.get_params()` is:

- `rotors_firefly_xacro`

That is the safest default for the fresh learned-controller port because it
matches the Gazebo vehicle description directly.

## Immediate follow-up on Ubuntu

Before we start online learned-controller integration, verify which runtime
parameter profile is actually active in the controller stack:

```bash
source /opt/ros/noetic/setup.bash
cd ~/thesis_v2_ws
source devel/setup.bash
rosparam get /firefly/lee_position_controller_node/mass
rosparam get /firefly/lee_position_controller_node/inertia
```

If the runtime controller still uses `1.56779 kg`, we will keep both profiles
explicit in V2 and make the online learned-controller experiments choose one
deliberately instead of mixing them accidentally.
