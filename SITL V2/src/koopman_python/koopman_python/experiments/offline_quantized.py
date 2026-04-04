"""Offline quantized EDMD-MPC experiment runner."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from koopman_python.dynamics.params import DEFAULT_PROFILE, get_params
from koopman_python.edmd.basis import lift_state, lift_trajectory
from koopman_python.edmd.evaluate import compute_rmse, evaluate_edmd_fixed_trajectory
from koopman_python.edmd.fit import EdmdModel, fit_edmd
from koopman_python.experiments.offline_learned_mpc import (
    CONTROLLER_VARIANT,
    RUN_FAMILY,
    SOLVER_BACKEND,
    _effective_scenario_name,
    _effective_sim_duration,
    _reference_duration,
    _tracking_metrics,
)
from koopman_python.experiments.offline_unquantized import (
    _append_csv_row,
    _control_rows,
    _default_initial_state,
    _git_commit,
    _repo_root,
    _time_grid,
    _v2_root,
    _write_csv,
    _write_json,
)
from koopman_python.experiments.reference_scenarios import (
    RANDOM_REFERENCE_SCENARIO,
    available_scenarios,
    generate_reference_states,
)
from koopman_python.mpc import MpcSimulationConfig, simulate_closed_loop
from koopman_python.quantization import dither_signal, partition_word_length
from koopman_python.training import get_random_trajectories, get_reference_seeded_random_trajectories


@dataclass(frozen=True)
class OfflineQuantizedConfig:
    word_length: int = 12
    realizations: int = 5
    training_n_control: int = 500
    training_dt: float = 1e-3
    training_t_span: float = 0.1
    training_source: str = "reference_seeded_random"
    training_seed_count: int = 25
    eval_mode: str = "val"
    n_basis: int = 3
    seed: int = 2141444
    parameter_profile: str = DEFAULT_PROFILE
    scenario_name: str = "line_tracking"
    pred_horizon: int = 10
    sim_time_step: float = 1e-3
    sim_duration: float = 5.0
    control_lower_bound: float = -50.0
    control_upper_bound: float = 50.0
    qp_max_iter: int = 100
    qp_tol: float = 1e-8
    reference_mode: str = "mpc"


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_offline_quantized")


def _flatten_metric_groups(prefix: str, metrics: dict[str, float]) -> list[dict[str, object]]:
    return [
        {
            "metric_name": f"{prefix}_{name}",
            "metric_value": value,
            "units": "normalized_rmse" if "rmse" in name else "",
        }
        for name, value in metrics.items()
    ]


def _rebuild_snapshot_pairs_from_trajectory_stack(
    trajectory_stack: np.ndarray,
    time_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_matrix = np.asarray(trajectory_stack, dtype=float)
    if state_matrix.ndim != 2:
        raise ValueError("trajectory_stack must be a 2D matrix.")
    trajectory_length = int(np.asarray(time_grid, dtype=float).size)
    if trajectory_length < 2:
        raise ValueError("time_grid must contain at least two samples.")
    if state_matrix.shape[1] % trajectory_length != 0:
        raise ValueError("trajectory_stack column count must be divisible by trajectory length.")

    n_trajectories = state_matrix.shape[1] // trajectory_length
    x1_blocks = []
    x2_blocks = []
    for trajectory_index in range(n_trajectories):
        lower = trajectory_index * trajectory_length
        upper = lower + trajectory_length
        trajectory = state_matrix[:, lower:upper]
        x1_blocks.append(trajectory[:, :-1])
        x2_blocks.append(trajectory[:, 1:])
    return np.column_stack(x1_blocks), np.column_stack(x2_blocks)


def _fit_training_batch(
    config: OfflineQuantizedConfig,
    params: dict[str, object],
    reference_states: np.ndarray,
    initial_state: np.ndarray,
    training_time_grid: np.ndarray,
    rng: np.random.Generator,
):
    if config.training_source == "single_initial_state_random":
        return get_random_trajectories(
            initial_state=initial_state,
            n_control=config.training_n_control,
            t_traj=training_time_grid,
            mode="train",
            params=params,
            rng=rng,
        )
    if config.training_source == "reference_seeded_random":
        seeded_reference = np.column_stack((initial_state.reshape(18, 1), reference_states))
        return get_reference_seeded_random_trajectories(
            reference_states=seeded_reference,
            n_control_total=config.training_n_control,
            t_traj=training_time_grid,
            num_seed_states=config.training_seed_count,
            mode="train",
            params=params,
            rng=rng,
        )
    raise ValueError(f"Unsupported training_source: {config.training_source}")


def _build_reference_data(
    config: OfflineQuantizedConfig,
    params: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    reference_time_grid = _time_grid(config.sim_time_step, _reference_duration(config))
    scenario_name = _effective_scenario_name(config)
    if scenario_name == RANDOM_REFERENCE_SCENARIO:
        initial_state = _default_initial_state()
        reference_batch = get_random_trajectories(
            initial_state=initial_state,
            n_control=1,
            t_traj=reference_time_grid,
            mode=config.reference_mode,
            params=params,
            rng=rng,
        )
        return initial_state, reference_batch.X[:, 1:]
    deterministic_reference = generate_reference_states(scenario_name, reference_time_grid)
    return deterministic_reference[:, 0], deterministic_reference[:, 1:]


def _model_difference(unquantized: EdmdModel, quantized: EdmdModel) -> tuple[float, float]:
    a_diff = float(np.linalg.norm(unquantized.A - quantized.A, ord="fro") / np.linalg.norm(unquantized.A, ord="fro"))
    b_diff = float(np.linalg.norm(unquantized.B - quantized.B, ord="fro") / np.linalg.norm(unquantized.B, ord="fro"))
    return a_diff, b_diff


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    if not metric_dicts:
        raise ValueError("metric_dicts must not be empty.")
    keys = metric_dicts[0].keys()
    values = {key: np.array([metrics[key] for metrics in metric_dicts], dtype=float) for key in keys}
    mean = {key: float(np.mean(array)) for key, array in values.items()}
    std = {key: float(np.std(array)) for key, array in values.items()}
    return mean, std


def run_offline_quantized_experiment(
    config: OfflineQuantizedConfig,
    output_root: Path | None = None,
) -> Path:
    """Run the SciTech-style quantized EDMD-MPC batch experiment."""

    v2_root = _v2_root()
    repo_root = _repo_root()
    results_root = output_root or (v2_root / "results")
    run_id = _timestamp_run_id()
    scenario_name = _effective_scenario_name(config)
    run_dir = results_root / "learned" / "quantized" / f"wl_{config.word_length}" / f"N_{config.realizations}" / scenario_name / run_id
    figure_dir = run_dir / "figures"
    realization_root = run_dir / "realizations"
    figure_dir.mkdir(parents=True, exist_ok=True)
    realization_root.mkdir(parents=True, exist_ok=True)

    params = get_params(config.parameter_profile)
    master_rng = np.random.default_rng(config.seed)
    initial_state, reference_states = _build_reference_data(config, params, master_rng)
    training_time_grid = _time_grid(config.training_dt, config.training_t_span)
    eval_time_grid = training_time_grid

    train_batch = _fit_training_batch(
        config=config,
        params=params,
        reference_states=reference_states,
        initial_state=initial_state,
        training_time_grid=training_time_grid,
        rng=master_rng,
    )
    fit_start = time.perf_counter()
    unquantized_model = fit_edmd(
        X1=train_batch.X1,
        X2=train_batch.X2,
        U1=train_batch.U1,
        n_basis=config.n_basis,
    )
    unquantized_fit_duration_s = time.perf_counter() - fit_start

    eval_batch = get_random_trajectories(
        initial_state=initial_state,
        n_control=1,
        t_traj=eval_time_grid,
        mode=config.eval_mode,
        params=params,
        rng=np.random.default_rng(config.seed + 10_000),
    )
    unquantized_eval = evaluate_edmd_fixed_trajectory(
        initial_state=initial_state,
        true_states=eval_batch.X[:, : eval_time_grid.size],
        controls=eval_batch.U1[:, : eval_time_grid.size - 1],
        model=unquantized_model,
    )

    lifted_reference = lift_trajectory(reference_states, config.n_basis)
    initial_lifted_state = lift_state(initial_state, config.n_basis)

    x_partition = partition_word_length(np.min(train_batch.X, axis=1), np.max(train_batch.X, axis=1), config.word_length)
    u_partition = partition_word_length(np.min(train_batch.U1, axis=1), np.max(train_batch.U1, axis=1), config.word_length)

    realization_rows = []
    prediction_metric_dicts = []
    tracking_metric_dicts = []
    matrix_metric_dicts = []
    solve_metric_dicts = []
    actual_state_trajectories = []
    control_trajectories = []
    solve_time_trajectories = []
    solve_iteration_trajectories = []
    solve_converged_trajectories = []
    solve_projected_step_inf_norm_trajectories = []
    solve_hit_iteration_cap_trajectories = []
    tracked_reference_template = None
    time_template = None
    successful_realizations = 0

    for realization_index in range(config.realizations):
        realization_rng = np.random.default_rng(config.seed + realization_index + 1)
        realization_dir = realization_root / f"realization_{realization_index:03d}"
        realization_dir.mkdir(parents=True, exist_ok=True)

        x_dithered = dither_signal(train_batch.X, x_partition, rng=realization_rng)
        x1_dithered, x2_dithered = _rebuild_snapshot_pairs_from_trajectory_stack(x_dithered, training_time_grid)
        u1_dithered = dither_signal(train_batch.U1, u_partition, rng=realization_rng)

        fit_start = time.perf_counter()
        quantized_model = fit_edmd(
            X1=x1_dithered,
            X2=x2_dithered,
            U1=u1_dithered,
            n_basis=config.n_basis,
        )
        fit_duration_s = time.perf_counter() - fit_start

        prediction_eval = evaluate_edmd_fixed_trajectory(
            initial_state=initial_state,
            true_states=eval_batch.X[:, : eval_time_grid.size],
            controls=eval_batch.U1[:, : eval_time_grid.size - 1],
            model=quantized_model,
        )

        sim_start = time.perf_counter()
        mpc_result = simulate_closed_loop(
            model=quantized_model,
            initial_lifted_state=initial_lifted_state,
            lifted_reference=lifted_reference,
            reference_states=reference_states,
            config=MpcSimulationConfig(
                pred_horizon=config.pred_horizon,
                sim_time_step=config.sim_time_step,
                sim_duration=_effective_sim_duration(config),
                control_lower_bound=config.control_lower_bound,
                control_upper_bound=config.control_upper_bound,
                qp_max_iter=config.qp_max_iter,
                qp_tol=config.qp_tol,
                parameter_profile=config.parameter_profile,
            ),
            params=params,
        )
        sim_duration_s = time.perf_counter() - sim_start

        actual_states = mpc_result.X.T
        tracked_reference_states = mpc_result.X_ref.T
        if tracked_reference_template is None:
            tracked_reference_template = tracked_reference_states
            time_template = np.asarray(mpc_result.t, dtype=float)
        tracking_metrics, _ = _tracking_metrics(actual_states, tracked_reference_states)
        a_diff, b_diff = _model_difference(unquantized_model, quantized_model)
        success = bool(np.all(np.isfinite(actual_states)) and np.all(np.isfinite(mpc_result.U)))
        if success:
            successful_realizations += 1

        prediction_metrics = asdict(prediction_eval.rmse)
        matrix_metrics = {"A_relative_difference": a_diff, "B_relative_difference": b_diff}
        solve_metrics = {
            "fit_duration_ms": fit_duration_s * 1000.0,
            "simulation_duration_ms": sim_duration_s * 1000.0,
            "solve_time_ms_mean": float(np.mean(mpc_result.solve_times_ms)),
            "solve_time_ms_max": float(np.max(mpc_result.solve_times_ms)),
            "solve_iterations_mean": float(np.mean(mpc_result.solve_iterations)),
            "solve_iterations_max": float(np.max(mpc_result.solve_iterations)),
            "solve_converged_fraction": float(np.mean(mpc_result.solve_converged.astype(float))),
            "solve_projected_step_inf_norm_mean": float(np.mean(mpc_result.solve_projected_step_inf_norms)),
            "solve_projected_step_inf_norm_max": float(np.max(mpc_result.solve_projected_step_inf_norms)),
            "solve_hit_iteration_cap_fraction": float(np.mean(mpc_result.solve_hit_iteration_cap.astype(float))),
        }

        prediction_metric_dicts.append(prediction_metrics)
        tracking_metric_dicts.append(tracking_metrics)
        matrix_metric_dicts.append(matrix_metrics)
        solve_metric_dicts.append(solve_metrics)
        actual_state_trajectories.append(actual_states)
        control_trajectories.append(mpc_result.U.T)
        solve_time_trajectories.append(np.asarray(mpc_result.solve_times_ms, dtype=float))
        solve_iteration_trajectories.append(np.asarray(mpc_result.solve_iterations, dtype=float))
        solve_converged_trajectories.append(np.asarray(mpc_result.solve_converged, dtype=float))
        solve_projected_step_inf_norm_trajectories.append(
            np.asarray(mpc_result.solve_projected_step_inf_norms, dtype=float)
        )
        solve_hit_iteration_cap_trajectories.append(
            np.asarray(mpc_result.solve_hit_iteration_cap, dtype=float)
        )

        trajectory_rows = []
        for step_index, t_s in enumerate(mpc_result.t):
            trajectory_rows.append(
                {
                    "t_s": float(t_s),
                    "x_ref": float(tracked_reference_states[0, step_index]),
                    "y_ref": float(tracked_reference_states[1, step_index]),
                    "z_ref": float(tracked_reference_states[2, step_index]),
                    "x_mpc": float(actual_states[0, step_index]),
                    "y_mpc": float(actual_states[1, step_index]),
                    "z_mpc": float(actual_states[2, step_index]),
                }
            )
        timing_rows = [
            {
                "step_index": int(step_index),
                "t_s": float(mpc_result.t[step_index]),
                "solve_time_ms": float(mpc_result.solve_times_ms[step_index]),
                "solve_iterations": int(mpc_result.solve_iterations[step_index]),
                "solve_converged": bool(mpc_result.solve_converged[step_index]),
                "solve_projected_step_inf_norm": float(mpc_result.solve_projected_step_inf_norms[step_index]),
                "solve_hit_iteration_cap": bool(mpc_result.solve_hit_iteration_cap[step_index]),
            }
            for step_index in range(mpc_result.solve_times_ms.size)
        ]

        _write_json(
            realization_dir / "metrics.json",
            {
                "realization_index": realization_index,
                "success": success,
                "prediction_rmse": prediction_metrics,
                "tracking_metrics": tracking_metrics,
                "matrix_metrics": matrix_metrics,
                **solve_metrics,
            },
        )
        _write_csv(realization_dir / "trajectory.csv", ["t_s", "x_ref", "y_ref", "z_ref", "x_mpc", "y_mpc", "z_mpc"], trajectory_rows)
        control_rows = _control_rows(mpc_result.t, mpc_result.U.T)
        _write_csv(realization_dir / "control.csv", list(control_rows[0].keys()), control_rows)
        _write_csv(
            realization_dir / "timing.csv",
            [
                "step_index",
                "t_s",
                "solve_time_ms",
                "solve_iterations",
                "solve_converged",
                "solve_projected_step_inf_norm",
                "solve_hit_iteration_cap",
            ],
            timing_rows,
        )
        np.savez_compressed(
            realization_dir / "model.npz",
            K=quantized_model.K,
            A=quantized_model.A,
            B=quantized_model.B,
            C=quantized_model.C,
            Z1=quantized_model.Z1,
            Z2=quantized_model.Z2,
            n_basis=quantized_model.n_basis,
        )
        np.savez_compressed(
            realization_dir / "closed_loop.npz",
            t=mpc_result.t,
            X=mpc_result.X,
            U=mpc_result.U,
            X_ref=mpc_result.X_ref,
            Z=mpc_result.Z,
            Z_ref=mpc_result.Z_ref,
            solve_times_ms=mpc_result.solve_times_ms,
            solve_iterations=mpc_result.solve_iterations,
            solve_converged=mpc_result.solve_converged,
            solve_projected_step_inf_norms=mpc_result.solve_projected_step_inf_norms,
            solve_hit_iteration_cap=mpc_result.solve_hit_iteration_cap,
        )

        realization_rows.append(
            {
                "realization_index": realization_index,
                "success": success,
                "prediction_rmse_x": prediction_metrics["x"],
                "prediction_rmse_dx": prediction_metrics["dx"],
                "prediction_rmse_theta": prediction_metrics["theta"],
                "prediction_rmse_wb": prediction_metrics["wb"],
                "position_rmse_m": tracking_metrics["position_rmse_m"],
                "velocity_rmse_mps": tracking_metrics["velocity_rmse_mps"],
                "theta_rmse_rad": tracking_metrics["theta_rmse_rad"],
                "wb_rmse_rad_s": tracking_metrics["wb_rmse_rad_s"],
                "final_position_error_m": tracking_metrics["final_position_error_m"],
                "max_position_error_m": tracking_metrics["max_position_error_m"],
                "A_relative_difference": a_diff,
                "B_relative_difference": b_diff,
                **solve_metrics,
            }
        )

    prediction_mean, prediction_std = _aggregate_metric_dicts(prediction_metric_dicts)
    tracking_mean, tracking_std = _aggregate_metric_dicts(tracking_metric_dicts)
    matrix_mean, matrix_std = _aggregate_metric_dicts(matrix_metric_dicts)
    solve_mean, solve_std = _aggregate_metric_dicts(solve_metric_dicts)

    actual_state_stack = np.stack(actual_state_trajectories, axis=0)
    control_stack = np.stack(control_trajectories, axis=0)
    solve_time_stack = np.stack(solve_time_trajectories, axis=0)
    solve_iteration_stack = np.stack(solve_iteration_trajectories, axis=0)
    solve_converged_stack = np.stack(solve_converged_trajectories, axis=0)
    solve_projected_step_inf_norm_stack = np.stack(solve_projected_step_inf_norm_trajectories, axis=0)
    solve_hit_iteration_cap_stack = np.stack(solve_hit_iteration_cap_trajectories, axis=0)
    reference_positions = tracked_reference_template[0:3, :]
    actual_positions_mean = np.mean(actual_state_stack[:, 0:3, :], axis=0)
    actual_positions_std = np.std(actual_state_stack[:, 0:3, :], axis=0)
    position_error_norm_stack = np.linalg.norm(actual_state_stack[:, 0:3, :] - reference_positions[np.newaxis, :, :], axis=1)
    control_mean = np.mean(control_stack, axis=0)
    control_std = np.std(control_stack, axis=0)
    solve_time_mean_per_step = np.mean(solve_time_stack, axis=0)
    solve_time_std_per_step = np.std(solve_time_stack, axis=0)
    solve_iteration_mean_per_step = np.mean(solve_iteration_stack, axis=0)
    solve_iteration_max_per_step = np.max(solve_iteration_stack, axis=0)
    solve_converged_fraction_per_step = np.mean(solve_converged_stack, axis=0)
    solve_projected_step_inf_norm_mean_per_step = np.mean(solve_projected_step_inf_norm_stack, axis=0)
    solve_projected_step_inf_norm_max_per_step = np.max(solve_projected_step_inf_norm_stack, axis=0)
    solve_hit_iteration_cap_fraction_per_step = np.mean(solve_hit_iteration_cap_stack, axis=0)

    metrics_payload = {
        "run_id": run_id,
        "success": successful_realizations == config.realizations,
        "successful_realizations": successful_realizations,
        "solver_backend": SOLVER_BACKEND,
        "word_length": config.word_length,
        "realizations": config.realizations,
        "prediction_rmse_mean": prediction_mean,
        "prediction_rmse_std": prediction_std,
        "tracking_metrics_mean": tracking_mean,
        "tracking_metrics_std": tracking_std,
        "matrix_metrics_mean": matrix_mean,
        "matrix_metrics_std": matrix_std,
        "solve_metrics_mean": solve_mean,
        "solve_metrics_std": solve_std,
        "unquantized_prediction_rmse": asdict(unquantized_eval.rmse),
        "unquantized_fit_duration_ms": unquantized_fit_duration_s * 1000.0,
        "n_basis": config.n_basis,
        "training_n_control": config.training_n_control,
        "training_source": config.training_source,
        "training_seed_count": config.training_seed_count,
        "n_training_snapshots": int(train_batch.X1.shape[1]),
        "pred_horizon": config.pred_horizon,
        "effective_sim_duration": _effective_sim_duration(config),
        "parameter_profile": config.parameter_profile,
        "A_shape": list(unquantized_model.A.shape),
        "B_shape": list(unquantized_model.B.shape),
        "C_shape": list(unquantized_model.C.shape),
    }

    config_payload = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": scenario_name,
        "word_length": config.word_length,
        "realizations": config.realizations,
        **asdict(config),
        "effective_sim_duration": _effective_sim_duration(config),
        "reference_duration": _reference_duration(config),
        "reference_steps": int(reference_states.shape[1]),
    }

    environment_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(repo_root),
        "parameter_profile": config.parameter_profile,
        "solver_backend": SOLVER_BACKEND,
        "parameter_values": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in params.items()
        },
    }

    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "metrics.json", metrics_payload)
    _write_json(run_dir / "environment.json", environment_payload)
    _write_csv(run_dir / "realization_metrics.csv", list(realization_rows[0].keys()), realization_rows)
    aggregate_trajectory_rows = [
        {
            "t_s": float(time_template[step_index]),
            "x_ref": float(reference_positions[0, step_index]),
            "y_ref": float(reference_positions[1, step_index]),
            "z_ref": float(reference_positions[2, step_index]),
            "x_mpc_mean": float(actual_positions_mean[0, step_index]),
            "y_mpc_mean": float(actual_positions_mean[1, step_index]),
            "z_mpc_mean": float(actual_positions_mean[2, step_index]),
            "x_mpc_std": float(actual_positions_std[0, step_index]),
            "y_mpc_std": float(actual_positions_std[1, step_index]),
            "z_mpc_std": float(actual_positions_std[2, step_index]),
            "position_error_norm_mean": float(np.mean(position_error_norm_stack[:, step_index])),
            "position_error_norm_std": float(np.std(position_error_norm_stack[:, step_index])),
        }
        for step_index in range(time_template.size)
    ]
    _write_csv(
        run_dir / "trajectory.csv",
        [
            "t_s",
            "x_ref",
            "y_ref",
            "z_ref",
            "x_mpc_mean",
            "y_mpc_mean",
            "z_mpc_mean",
            "x_mpc_std",
            "y_mpc_std",
            "z_mpc_std",
            "position_error_norm_mean",
            "position_error_norm_std",
        ],
        aggregate_trajectory_rows,
    )
    aggregate_control_rows = [
        {
            "t_s": float(time_template[step_index]),
            "Fb_mean": float(control_mean[step_index, 0]),
            "Mbx_mean": float(control_mean[step_index, 1]),
            "Mby_mean": float(control_mean[step_index, 2]),
            "Mbz_mean": float(control_mean[step_index, 3]),
            "Fb_std": float(control_std[step_index, 0]),
            "Mbx_std": float(control_std[step_index, 1]),
            "Mby_std": float(control_std[step_index, 2]),
            "Mbz_std": float(control_std[step_index, 3]),
        }
        for step_index in range(control_mean.shape[0])
    ]
    _write_csv(
        run_dir / "control.csv",
        ["t_s", "Fb_mean", "Mbx_mean", "Mby_mean", "Mbz_mean", "Fb_std", "Mbx_std", "Mby_std", "Mbz_std"],
        aggregate_control_rows,
    )
    aggregate_timing_rows = [
        {
            "step_index": int(step_index),
            "t_s": float(time_template[step_index]),
            "solve_time_ms_mean": float(solve_time_mean_per_step[step_index]),
            "solve_time_ms_std": float(solve_time_std_per_step[step_index]),
            "solve_iterations_mean": float(solve_iteration_mean_per_step[step_index]),
            "solve_iterations_max": float(solve_iteration_max_per_step[step_index]),
            "solve_converged_fraction": float(solve_converged_fraction_per_step[step_index]),
            "solve_projected_step_inf_norm_mean": float(solve_projected_step_inf_norm_mean_per_step[step_index]),
            "solve_projected_step_inf_norm_max": float(solve_projected_step_inf_norm_max_per_step[step_index]),
            "solve_hit_iteration_cap_fraction": float(solve_hit_iteration_cap_fraction_per_step[step_index]),
        }
        for step_index in range(solve_time_mean_per_step.size)
    ]
    _write_csv(
        run_dir / "timing.csv",
        [
            "step_index",
            "t_s",
            "solve_time_ms_mean",
            "solve_time_ms_std",
            "solve_iterations_mean",
            "solve_iterations_max",
            "solve_converged_fraction",
            "solve_projected_step_inf_norm_mean",
            "solve_projected_step_inf_norm_max",
            "solve_hit_iteration_cap_fraction",
        ],
        aggregate_timing_rows,
    )
    np.savez_compressed(
        run_dir / "unquantized_reference_model.npz",
        K=unquantized_model.K,
        A=unquantized_model.A,
        B=unquantized_model.B,
        C=unquantized_model.C,
        Z1=unquantized_model.Z1,
        Z2=unquantized_model.Z2,
        n_basis=unquantized_model.n_basis,
    )

    (run_dir / "notes.txt").write_text(
        "Offline quantized EDMD-MPC batch run.\n"
        "This path quantizes training snapshots and control inputs before EDMD fitting.\n"
        "Figures are intentionally deferred; use the saved CSV/JSON artifacts for plotting.\n",
        encoding="utf-8",
    )

    summary_root = results_root / "summary"
    run_index_row = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": scenario_name,
        "word_length": config.word_length,
        "realizations": config.realizations,
        "config_path": str(run_dir / "config.json"),
        "metrics_path": str(run_dir / "metrics.json"),
        "trajectory_path": str(run_dir / "trajectory.csv"),
        "control_path": str(run_dir / "control.csv"),
        "timing_path": str(run_dir / "timing.csv"),
        "environment_path": str(run_dir / "environment.json"),
        "figure_dir": str(figure_dir),
    }
    _append_csv_row(
        summary_root / "run_index.csv",
        [
            "run_id",
            "run_family",
            "controller_variant",
            "scenario",
            "word_length",
            "realizations",
            "config_path",
            "metrics_path",
            "trajectory_path",
            "control_path",
            "timing_path",
            "environment_path",
            "figure_dir",
        ],
        run_index_row,
    )

    metric_rows = []
    for row in _flatten_metric_groups("prediction_rmse_mean", prediction_mean):
        metric_rows.append({"run_id": run_id, **row})
    for row in _flatten_metric_groups("tracking_mean", tracking_mean):
        units = {
            "position_rmse_m": "m",
            "velocity_rmse_mps": "m/s",
            "theta_rmse_rad": "rad",
            "wb_rmse_rad_s": "rad/s",
            "final_position_error_m": "m",
            "max_position_error_m": "m",
        }[row["metric_name"].replace("tracking_mean_", "")]
        row["units"] = units
        metric_rows.append({"run_id": run_id, **row})
    for key, value in matrix_mean.items():
        metric_rows.append({"run_id": run_id, "metric_name": key, "metric_value": value, "units": "fraction"})
    for key, value in solve_mean.items():
        units = "ms" if "ms" in key else ("fraction" if "fraction" in key else ("norm" if "norm" in key else "iterations"))
        metric_rows.append({"run_id": run_id, "metric_name": key, "metric_value": value, "units": units})

    for row in metric_rows:
        _append_csv_row(
            summary_root / "metrics_long.csv",
            ["run_id", "metric_name", "metric_value", "units"],
            row,
        )

    _append_csv_row(
        summary_root / "metrics_wide.csv",
        [
            "run_id",
            "run_family",
            "controller_variant",
            "scenario",
            "word_length",
            "realizations",
            "success",
            "hover_rmse_m",
            "tracking_rmse_m",
            "max_error_m",
            "solve_time_ms_mean",
        ],
        {
            "run_id": run_id,
            "run_family": RUN_FAMILY,
            "controller_variant": CONTROLLER_VARIANT,
            "scenario": scenario_name,
            "word_length": config.word_length,
            "realizations": config.realizations,
            "success": metrics_payload["success"],
            "hover_rmse_m": tracking_mean["position_rmse_m"] if scenario_name == "hover_5s" else "",
            "tracking_rmse_m": "" if scenario_name == "hover_5s" else tracking_mean["position_rmse_m"],
            "max_error_m": tracking_mean["max_position_error_m"],
            "solve_time_ms_mean": solve_mean["solve_time_ms_mean"],
        },
    )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline quantized EDMD-MPC experiment.")
    parser.add_argument("--word-length", type=int, default=12, choices=[8, 12, 14, 16])
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--training-n-control", type=int, default=500)
    parser.add_argument("--training-dt", type=float, default=1e-3)
    parser.add_argument("--training-t-span", type=float, default=0.1)
    parser.add_argument("--training-source", type=str, default="reference_seeded_random", choices=["single_initial_state_random", "reference_seeded_random"])
    parser.add_argument("--training-seed-count", type=int, default=25)
    parser.add_argument("--eval-mode", type=str, default="val", choices=["train", "val", "mpc"])
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2141444)
    parser.add_argument("--parameter-profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--scenario-name", type=str, default="line_tracking", choices=available_scenarios())
    parser.add_argument("--pred-horizon", type=int, default=10)
    parser.add_argument("--sim-time-step", type=float, default=1e-3)
    parser.add_argument("--sim-duration", type=float, default=5.0)
    parser.add_argument("--control-lower-bound", type=float, default=-50.0)
    parser.add_argument("--control-upper-bound", type=float, default=-50.0 + 100.0)
    parser.add_argument("--qp-max-iter", type=int, default=100)
    parser.add_argument("--qp-tol", type=float, default=1e-8)
    parser.add_argument("--reference-mode", type=str, default="mpc", choices=["train", "val", "mpc"])
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OfflineQuantizedConfig(
        word_length=args.word_length,
        realizations=args.realizations,
        training_n_control=args.training_n_control,
        training_dt=args.training_dt,
        training_t_span=args.training_t_span,
        training_source=args.training_source,
        training_seed_count=args.training_seed_count,
        eval_mode=args.eval_mode,
        n_basis=args.n_basis,
        seed=args.seed,
        parameter_profile=args.parameter_profile,
        scenario_name=args.scenario_name,
        pred_horizon=args.pred_horizon,
        sim_time_step=args.sim_time_step,
        sim_duration=args.sim_duration,
        control_lower_bound=args.control_lower_bound,
        control_upper_bound=args.control_upper_bound,
        qp_max_iter=args.qp_max_iter,
        qp_tol=args.qp_tol,
        reference_mode=args.reference_mode,
    )
    run_dir = run_offline_quantized_experiment(config=config, output_root=args.output_root)
    print(f"offline_quantized_run_dir={run_dir}")


if __name__ == "__main__":
    main()
