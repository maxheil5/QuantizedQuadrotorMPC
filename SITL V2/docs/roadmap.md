# SITL V2 Roadmap

## Objective

Deliver a stable thesis result ladder in a fresh Gazebo-based stack:

1. Classical linear MPC
2. Classical nonlinear MPC
3. Learned EDMD-MPC
4. Quantized learned EDMD-MPC

The guiding principle is: start small, validate early, then scale.

## Phase 1: Stable Gazebo benchmark

- Use RotorS Firefly as the default vehicle.
- Bring up `mav_linear_mpc` first.
- Bring up `mav_nonlinear_mpc` second.
- Keep the low-level RotorS controller unchanged.
- First scenarios:
  - `hover_5s`
  - `line_tracking`
  - `circle_tracking`

Success gate:

- `hover_5s` succeeds 3 times in a row for linear MPC.
- `hover_5s` succeeds 3 times in a row for nonlinear MPC.
- `line_tracking` and `circle_tracking` succeed before learned-controller integration starts.

## Phase 2: Fresh Python learned-stack port

Reimplement the learned path from scratch under `SITL V2/src/koopman_python`.

Source-of-truth priority:

1. MATLAB code in this repo
2. `SITL V2/src/KoopmanMPC_Quadrotor`

Core modules to port:

- dynamics
- training trajectory generation
- EDMD basis functions
- EDMD fitting
- prediction evaluation
- lifted linear MPC QP construction
- closed-loop simulation
- dither quantization

## Phase 3: Offline learned validation

Validation ladder:

1. one unquantized offline learned run
2. one quantized offline run at word length `12`, `N = 5`
3. word lengths `8`, `14`, `16` at `N = 5`

Promotion gate:

- do not move online until offline plots and metrics are stable and interpretable

## Phase 4: Online learned controller

Add one ROS1 learned-controller package under `SITL V2/src/koopman_mpc_ros`.

Interface:

- subscribe: `odometry`
- subscribe: `command/pose`
- subscribe: `command/trajectory`
- publish: `command/roll_pitch_yawrate_thrust`

Scenario ladder:

1. `hover_5s`
2. `line_tracking`
3. `circle_tracking`

## Phase 5: Quantized learned controller

Main quantization path follows the MATLAB SciTech workflow:

- quantize and dither state training snapshots
- quantize and dither control training inputs
- fit the quantized EDMD model
- run learned MPC with that model

Reported word lengths:

- `8`
- `12`
- `14`
- `16`

Realization ladder:

- `N = 5`
- `N = 10`
- `N = 20`
- `N = 50`

Promotion rule:

- only move to the next `N` after the previous stage has stable metrics, usable plots, and no obvious pipeline bugs

## Thesis Result Packages

### Package A: Classical benchmark

- linear MPC hover
- nonlinear MPC hover
- line and circle tracking
- timing and error tables

### Package B: Learned EDMD-MPC offline

- prediction quality
- learned-MPC offline tracking
- comparison to classical references

### Package C: Learned EDMD-MPC online

- `hover_5s`
- `line_tracking`
- `circle_tracking`
- comparison to linear and nonlinear MPC

### Package D: Quantized learned EDMD-MPC, small validation

- word lengths `8, 12, 14, 16`
- `N = 5`
- unquantized reference

### Package E: Quantized learned EDMD-MPC, scaled thesis results

- same word lengths
- `N = 10`
- `N = 20`
- `N = 50`

