# Quantized Quadrotor SITL

This workspace rebuilds the MATLAB quantized Koopman/EDMD + MPC workflow inside `SITL/` and keeps the controller math in Python so the same code can be used offline and in ROS 2 / PX4 SITL.

## Workspace Layout

- `docs/` contains the MATLAB mapping, architecture notes, and Ubuntu 22.04 setup details.
- `configs/` contains offline profile references and the default SITL runtime config.
- `scripts/` contains setup and launch entrypoints.
- `src/quantized_quadrotor_sitl/` contains the reusable parity package and ROS 2 nodes.
- `tests/` contains parity-critical math tests.
- `results/`, `logs/`, and `artifacts/` are reserved for generated outputs and external checkouts.

## MATLAB Source Mapping

Start with [docs/plan.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL/docs/plan.md) and [docs/matlab_mapping.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL/docs/matlab_mapping.md).

The important behavior split is:

- `MATLAB/main_Heil_FINAL_V2.m` defines the quantization experiment loop and remains the primary source of truth.
- `MATLAB/main.m` supplies the MATLAB MPC rollout that the V2 script leaves as tracking placeholders.
- The paper defines the intended 50-realization closed-loop study and the Gazebo/PX4 migration target.

## Setup

Ubuntu 22.04 + ROS 2 Humble + Gazebo Harmonic + PX4 setup is documented in [docs/setup_ubuntu_2204.md](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL/docs/setup_ubuntu_2204.md).

The automated entrypoint is:

```bash
cd SITL
./scripts/setup_linux.sh
```

## Offline Parity

Strict `main_Heil_FINAL_V2.m` parity:

```bash
cd SITL
./scripts/run_offline_parity.sh matlab_v2
```

Paper-style 50-realization closed-loop study:

```bash
cd SITL
./scripts/run_offline_parity.sh paper_v2
```

Outputs are written under `results/offline/<profile>/latest/`.

## Gazebo + PX4 SITL

After setup and after generating an offline model artifact, start the SITL stack with:

```bash
cd SITL
./scripts/run_sitl_experiment.sh configs/sitl_runtime.yaml
```

This launches:

1. Micro XRCE-DDS Agent
2. PX4 SITL in Gazebo Harmonic
3. ROS 2 telemetry adapter
4. ROS 2 quantized Koopman/EDMD + MPC controller node

## Quantization Injection Modes

Set `quantization_mode` in [configs/sitl_runtime.yaml](/Users/maxheil/Documents/Github/QuantizedQuadrotorMPC/SITL/configs/sitl_runtime.yaml):

- `none`
- `state`
- `control`
- `both`

The runtime quantizer uses the training-data ranges saved in the offline model artifact.

## Current Verification Status

The codebase is structured so it can be executed on Ubuntu 22.04 after dependencies are installed, but it was not possible to run the full offline or ROS/PX4/Gazebo stack in the current environment because NumPy/SciPy/matplotlib/pytest and the ROS 2/PX4/Gazebo toolchains are not installed here.

