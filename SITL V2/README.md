# SITL V2

This workspace is the clean parallel SITL track for the thesis.

It keeps the current `SITL/` work untouched and uses a simpler
Gazebo + MPC stack that should be easier to stabilize and modify with a
learned EDMD-MPC and a SciTech-style dithered quantization workflow.

## What Is Active

- `SITL/` stays preserved as prior work and reference only.
- `SITL V2/` is the only active implementation path for thesis results.
- The target environment for V2 is:
  - Ubuntu `20.04.6 LTS`
  - ROS `Noetic`
  - Gazebo `11`
  - external SSD boot to preserve the current machine

## Quick Links

- Roadmap: [docs/roadmap.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/docs/roadmap.md)
- Ubuntu setup: [docs/ubuntu_external_ssd_setup.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/docs/ubuntu_external_ssd_setup.md)
- MATLAB-to-Python mapping: [docs/matlab_to_python_mapping.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/docs/matlab_to_python_mapping.md)
- Results layout: [docs/results_layout.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/docs/results_layout.md)
- Upstream lock: [upstream_sources.lock.json](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/upstream_sources.lock.json)

## Bootstrap Scripts

- Workspace bootstrap:
  - [scripts/bootstrap_noetic_workspace.sh](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/scripts/bootstrap_noetic_workspace.sh)
- Environment freeze:
  - [scripts/freeze_environment.sh](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/scripts/freeze_environment.sh)
- Results tree init:
  - [scripts/init_results_tree.sh](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/scripts/init_results_tree.sh)
- Firefly runtime parameter audit:
  - [scripts/check_firefly_runtime_params.sh](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/scripts/check_firefly_runtime_params.sh)

## Upstream repos

- `src/rotors_simulator`
  - Upstream: `https://github.com/ethz-asl/rotors_simulator`
  - Local commit at clone time: `cd813b7a`
- `src/mav_control_rw`
  - Upstream: `https://github.com/ethz-asl/mav_control_rw`
  - Local commit at clone time: `207058d`
- `src/KoopmanMPC_Quadrotor`
  - Upstream: `https://github.com/sriram-2502/KoopmanMPC_Quadrotor`
  - Local commit at clone time: `4511293`

## Why this stack

- `rotors_simulator` provides a widely used Gazebo multicopter simulator.
- `mav_control_rw` provides working linear and nonlinear MPC controllers
  for multirotors and is designed to run with RotorS.
- `KoopmanMPC_Quadrotor` provides a concrete EDMD/Koopman quadrotor MPC
  reference that we can port into the Gazebo baseline.
- This should be a cleaner place to combine a stable Gazebo stack with the
  thesis quantizer and a learned MPC than the current PX4-heavy path.

## Fresh Reimplementation Rule

- Do not reuse the old `SITL` EDMD, quantization, or QP Python modules.
- Reimplement the learned controller path fresh in Python under `SITL V2`.
- Use the MATLAB code in this repo as the primary source of truth.
- Use `KoopmanMPC_Quadrotor` only to fill gaps or resolve ambiguity.

## New V2 Packages

- Fresh math package:
  - [src/koopman_python](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/src/koopman_python)
- ROS1 wrapper scaffold:
  - [src/koopman_mpc_ros](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/src/koopman_mpc_ros)
- Results root:
  - [results](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL%20V2/results)

## Runtime Parameter Audit

After launching Firefly with a controller on Ubuntu, capture the active runtime
mass and inertia with:

```bash
source /opt/ros/noetic/setup.bash
cd ~/thesis_v2_ws
source devel/setup.bash
bash ~/thesis_v2_ws/QuantizedQuadrotorMPC/"SITL V2"/scripts/check_firefly_runtime_params.sh \
  /firefly/lee_position_controller_node \
  ~/thesis_v2_ws/QuantizedQuadrotorMPC/"SITL V2"/results/summary/firefly_runtime_params.txt
```

## Initial Thesis Result Ladder

1. Stable Gazebo benchmark with `mav_linear_mpc`
2. Stable Gazebo benchmark with `mav_nonlinear_mpc`
3. Offline learned unquantized EDMD-MPC
4. Online learned unquantized EDMD-MPC in Gazebo
5. Quantized learned EDMD-MPC with word lengths `8, 12, 14, 16`

## Quantization Schedule

- Word lengths:
  - `8`
  - `12`
  - `14`
  - `16`
- Unquantized reference case is always included.
- Realization schedule:
  - `N = 5`
  - `N = 10`
  - `N = 20`
  - `N = 50`
- We only scale to the next `N` after the previous stage produces stable,
  interpretable plots and metrics.
