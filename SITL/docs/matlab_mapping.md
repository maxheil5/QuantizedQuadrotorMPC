# MATLAB Mapping

## Entry Points

- `MATLAB/main_Heil_FINAL_V2.m`
  - Recreated by `quantized_quadrotor_sitl.experiments.offline_parity`
  - Preserves word-length sweep, EDMD refit from dithered data, and matrix/prediction metrics
- `MATLAB/main.m`
  - Reused for the closed-loop MPC rollout that the paper reports but `main_Heil_FINAL_V2.m` leaves as placeholders

## Function Mapping

| MATLAB | Python |
| --- | --- |
| `training/get_rnd_trajectories.m` | `experiments.training_data.get_random_trajectories` |
| `dynamics/get_params.m` | `dynamics.params.get_params` |
| `dynamics/dynamics_SRB.m` | `dynamics.srb.dynamics_srb` |
| `edmd/get_basis.m` | `edmd.basis.get_basis` |
| `edmd/get_EDMD.m` | `edmd.fit.get_edmd` |
| `edmd/eval_EDMD*.m` | `edmd.evaluate.*` |
| `mpc/get_QP.m` | `mpc.qp.get_qp` |
| `mpc/sim_MPC.m` | `mpc.simulate.sim_mpc` |
| `Partition.m` | `quantization.partition.partition_range` |
| `Dither_Func.m` | `quantization.dither.dither_signal` |
| `Quantization.m` | `quantization.partition.quantize_scalar` |
| `quantizeCustom.m` | `quantization.partition.quantize_custom` |
| `utils/*.m` | `utils.*` and `plotting.matlab_style` |

## State Conventions

The MATLAB dynamics evolve an 18-state physical vector:

```text
[x; dx; R(:); wb]
```

The EDMD prefix keeps the first 24 lifted coordinates as:

```text
[x; dx; vec(R'); vec(wb_hat')]
```

The Python code preserves that split exactly instead of normalizing it away.

## Important Mismatches Preserved

- `main_Heil_FINAL_V2.m` uses `run_number = 5`; the paper uses `N = 50`.
- `main_Heil_FINAL_V2.m` includes an unquantized `Inf` case; the paper focuses on selected finite word-lengths.
- `main_Heil_FINAL_V2.m` leaves tracking errors as placeholders; `main.m` contains the actual MPC rollout.
- `Dither_Func.m` uses `(epsilon / 2) * (rand - 0.5)` instead of the full-width uniform dither interval described in the paper. The SITL code preserves the MATLAB implementation and documents the mismatch.

