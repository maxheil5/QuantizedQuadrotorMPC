# MATLAB To Fresh Python Mapping

This mapping defines the fresh V2 implementation targets. The existing
`SITL/` Python modules are reference-only and must not be imported into the
new V2 learned-controller path.

## Source-of-truth order

1. MATLAB files in this repo
2. `SITL V2/src/KoopmanMPC_Quadrotor`

## Mapping

| MATLAB source | V2 Python target | Notes |
| --- | --- | --- |
| `MATLAB/dynamics/get_params.m` | `koopman_python.dynamics.params` | Preserve physical constants and state conventions. |
| `MATLAB/dynamics/dynamics_SRB.m` | `koopman_python.dynamics.srb` | Preserve SRB dynamics and frame conventions. |
| `MATLAB/training/get_rnd_trajectories.m` | `koopman_python.training.random_trajectories` | Preserve random training data generation. |
| `MATLAB/main.m` trajectory path and `KoopmanMPC_Quadrotor/training/get_pid_trajectories.m` | `koopman_python.training.pid_trajectories` | Use for nominal trajectory-following training data. |
| `MATLAB/edmd/get_basis.m` and `KoopmanMPC_Quadrotor/edmd/get_basis.m` | `koopman_python.edmd.basis` | Preserve SE(3) lifting behavior. |
| `MATLAB/edmd/get_EDMD.m` and `KoopmanMPC_Quadrotor/edmd/get_EDMD.m` | `koopman_python.edmd.fit` | Preserve lifted snapshot assembly and least-squares fit. |
| `MATLAB/edmd/eval_EDMD.m` and `MATLAB/edmd/eval_EDMD_fixed_traj.m` | `koopman_python.edmd.evaluate` | Preserve one-step and rollout evaluation. |
| `MATLAB/mpc/get_QP.m` and `KoopmanMPC_Quadrotor/mpc/get_QP.m` | `koopman_python.mpc.qp` | Preserve lifted-state QP structure and costs. |
| `MATLAB/mpc/sim_MPC.m` and `KoopmanMPC_Quadrotor/mpc/sim_MPC.m` | `koopman_python.mpc.simulate` | Preserve closed-loop learned MPC rollout. |
| `MATLAB/Partition.m` | `koopman_python.quantization.partition` | Preserve word-length partitioning. |
| `MATLAB/Dither_Func.m` | `koopman_python.quantization.dither` | Preserve add-noise, quantize, subtract-noise sequence. |
| `MATLAB/Quantization.m` and `MATLAB/quantizeCustom.m` | `koopman_python.quantization` helpers | Preserve midpoint quantization semantics. |
| `MATLAB/main_Heil_FINAL_V2.m` | `koopman_python.experiments.offline_quantized` | Main SciTech-style quantized experiment path. |
| `MATLAB/main.m` | `koopman_python.experiments.offline_unquantized` | Main unquantized learned-MPC reference path. |

## Explicit non-goal

Do not port by copying from:

- `SITL/src/quantized_quadrotor_sitl/quantized_quadrotor_sitl/edmd`
- `SITL/src/quantized_quadrotor_sitl/quantized_quadrotor_sitl/mpc`
- `SITL/src/quantized_quadrotor_sitl/quantized_quadrotor_sitl/quantization`

Those modules can be inspected for context, but V2 must not depend on them.

## Parameter alignment rule

See [vehicle_parameter_alignment.md](./vehicle_parameter_alignment.md).

- Preserve the MATLAB SRB structure.
- Default the active V2 physical constants to the RotorS Firefly model.
- Keep the MATLAB parameter profile available for offline reference and thesis
  comparisons.
