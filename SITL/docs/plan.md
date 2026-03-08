# SITL Implementation Plan

## Scope

This `SITL/` project rebuilds the quantized Koopman/EDMD + MPC workflow from the MATLAB reference inside a Linux-targeted ROS 2 / Gazebo / PX4 stack without changing the underlying math. The implementation order follows the repository instructions:

1. Inspect MATLAB and manuscript.
2. Build an offline Python parity path first.
3. Layer Gazebo + ROS 2 + PX4 SITL on top of that parity layer.
4. Preserve MATLAB behavior before paper framing whenever they differ.

## Governing Sources

Priority order used in this project:

1. `MATLAB/main_Heil_FINAL_V2.m`
2. The MATLAB functions it calls
3. `MATLAB/main.m` for the existing MATLAB MPC rollout that `main_Heil_FINAL_V2.m` leaves as placeholders
4. The paper `Effect_of_Quantization_on_Data_Driven_Model_Predictive_Control_of_Quadcopters_Final_12-9-25_V2.pdf`

## MATLAB To SITL Mapping

| MATLAB source | SITL Python module | Notes |
| --- | --- | --- |
| `MATLAB/dynamics/get_params.m` | `quantized_quadrotor_sitl.dynamics.params` | Preserve mass, inertia, gravity constants. |
| `MATLAB/dynamics/dynamics_SRB.m` | `quantized_quadrotor_sitl.dynamics.srb` | Preserve the exact MATLAB state ordering and force/moment dynamics, including MATLAB’s rotation handling. |
| `MATLAB/training/get_rnd_trajectories.m` | `quantized_quadrotor_sitl.experiments.training_data` | Preserve per-flag covariance choices and constant-control trajectory generation. |
| `MATLAB/edmd/get_basis.m` | `quantized_quadrotor_sitl.edmd.basis` | Preserve `R'`, `wb_hat'`, and `R * wb_hat^i` lifting structure. |
| `MATLAB/edmd/get_EDMD.m` | `quantized_quadrotor_sitl.edmd.fit` | Preserve lifted snapshot assembly and least-squares matrix construction. |
| `MATLAB/edmd/eval_EDMD.m` | `quantized_quadrotor_sitl.edmd.evaluate` | Preserve open-loop prediction evaluation. |
| `MATLAB/edmd/eval_EDMD_fixed_traj.m` | `quantized_quadrotor_sitl.edmd.evaluate` | Preserve fixed-trajectory prediction RMSE flow. |
| `MATLAB/mpc/get_QP.m` | `quantized_quadrotor_sitl.mpc.qp` | Preserve lifted-state QP structure and weights. |
| `MATLAB/mpc/sim_MPC.m` | `quantized_quadrotor_sitl.mpc.simulate` | Preserve lifted MPC rollout using nonlinear SRB propagation. |
| `MATLAB/Partition.m` | `quantized_quadrotor_sitl.quantization.partition` | Preserve span inflation and midpoint quantizer construction. |
| `MATLAB/Dither_Func.m` | `quantized_quadrotor_sitl.quantization.dither` | Preserve additive dither, quantize, subtract-noise sequence. |
| `MATLAB/Quantization.m` | `quantized_quadrotor_sitl.quantization.scalar` | Preserve midpoint bin behavior and saturation behavior. |
| `MATLAB/quantizeCustom.m` | `quantized_quadrotor_sitl.quantization.array` | Preserve entrywise quantization semantics. |
| `MATLAB/utils/*.m` | `quantized_quadrotor_sitl.utils.*` and `plotting.*` | Preserve hat/vee/vectorization/RMSE/plot layout logic. |
| `MATLAB/main_Heil_FINAL_V2.m` | `quantized_quadrotor_sitl.experiments.offline_parity` | Primary offline experiment flow. |
| `MATLAB/main.m` | `quantized_quadrotor_sitl.experiments.paper_tracking` | Supplies the MATLAB MPC rollout used by the paper but not completed in `main_Heil_FINAL_V2.m`. |

## Known MATLAB vs Paper Mismatches

These mismatches are preserved and documented instead of silently normalized:

- `main_Heil_FINAL_V2.m` uses `run_number = 5`, while the paper reports `N = 50` dither realizations.
- `main_Heil_FINAL_V2.m` evaluates `word_length_array = 4:2:14` plus an unquantized `Inf` case, while the paper emphasizes `b = {4, 8, 12, 14, 16}`.
- `main_Heil_FINAL_V2.m` leaves `tracking_error_*` as placeholders, while `main.m` and the paper describe an actual MPC rollout and position-tracking metric.
- `MATLAB/Dither_Func.m` uses `(epsilon / 2) * (rand - 0.5)`, which is narrower than the full-width Schuchman dither interval described in the paper. The codebase preserves MATLAB behavior and documents the mismatch.
- The MATLAB dynamics and lifting code include rotation-transpose conventions that are internally consistent in the repository but not always written the same way as the paper equations. The SITL code keeps MATLAB behavior first and documents frame conversions explicitly.

## Deliverables

The `SITL/` tree will contain:

- `README.md` with runnable setup and experiment commands
- `docs/` with MATLAB mapping, ROS/PX4/Gazebo architecture, and setup notes
- `configs/` for experiment profiles, vehicle/setpoint scaling, and launch-time parameters
- `scripts/` for Ubuntu 22.04 setup and experiment entrypoints
- `src/` with the reusable parity package and ROS 2 nodes
- `tests/` for parity-critical math and experiment assembly
- `logs/`, `results/`, and `artifacts/` for generated runtime outputs

## Implementation Stages

### Stage 1: Offline parity

- Recreate the MATLAB trajectory generation, lifting, EDMD fit, dither quantization, RMSE, and MPC QP flow in Python.
- Support two experiment profiles:
  - `matlab_v2`: strict to `main_Heil_FINAL_V2.m`
  - `paper_v2`: same math, but with the paper’s 50-realization closed-loop tracking study
- Save plots and machine-readable metrics under `results/offline/`.

### Stage 2: SITL runtime

- Add a ROS 2 controller node that loads the same parity package and runs the quantized Koopman/EDMD + MPC controller online.
- Add a telemetry adapter that converts PX4 telemetry into the MATLAB state convention used by the parity layer.
- Add quantization injection modes for telemetry, control, both, or neither.
- Add logging and plotting pipelines that mirror the offline metrics and generate comparable trajectory plots from Gazebo/PX4 runs.

### Stage 3: Integration and documentation

- Provide Ubuntu 22.04 setup automation for ROS 2 Humble, Gazebo Harmonic, PX4 SITL, and ROS/PX4 message transport.
- Document package conflicts explicitly instead of changing the requested stack.
- Provide reproducible commands for offline and SITL experiments from inside `SITL/`.

## Verification Strategy

- Unit tests validate parity-critical formulas and shapes.
- Offline experiment commands validate end-to-end assembly once Python dependencies are present.
- ROS 2 / PX4 / Gazebo components are documented and structured for Ubuntu 22.04 execution, but cannot be fully executed in the current environment because ROS 2, PX4, Gazebo, and Python scientific packages are not installed here.
