"""Offline closed-loop learned EDMD-MPC experiment runner."""

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
from koopman_python.edmd.evaluate import compute_rmse, decode_full_state_trajectory
from koopman_python.edmd.fit import fit_edmd
from koopman_python.experiments.figures import generate_offline_learned_mpc_figures
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
    get_scenario_definition,
)
from koopman_python.mpc import MpcSimulationConfig, simulate_closed_loop
from koopman_python.training import get_random_trajectories, get_reference_seeded_random_trajectories


RUN_FAMILY = "learned"
CONTROLLER_VARIANT = "learned_edmd_mpc"
WORD_LENGTH = "unquantized"
SOLVER_BACKEND = "projected_gradient_box_qp"


@dataclass(frozen=True)
class OfflineLearnedMpcConfig:
    training_n_control: int = 100
    training_dt: float = 1e-3
    training_t_span: float = 0.1
    training_source: str = "single_initial_state_random"
    training_seed_count: int = 10
    n_basis: int = 3
    seed: int = 2141444
    parameter_profile: str = DEFAULT_PROFILE
    scenario_name: str = "hover_5s"
    pred_horizon: int = 10
    sim_time_step: float = 1e-3
    sim_duration: float = 5.0
    control_lower_bound: float = -50.0
    control_upper_bound: float = 50.0
    qp_max_iter: int = 2000
    qp_tol: float = 1e-8
    reference_mode: str = "mpc"


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_offline_learned_mpc")


def _effective_scenario_name(config: OfflineLearnedMpcConfig) -> str:
    return config.scenario_name or RANDOM_REFERENCE_SCENARIO


def _reference_duration(config: OfflineLearnedMpcConfig) -> float:
    return _effective_sim_duration(config) + config.pred_horizon * config.sim_time_step


def _effective_sim_duration(config: OfflineLearnedMpcConfig) -> float:
    scenario_name = _effective_scenario_name(config)
    if scenario_name == RANDOM_REFERENCE_SCENARIO:
        return config.sim_duration
    definition = get_scenario_definition(scenario_name)
    return min(config.sim_duration, definition.duration_s)


def _rms_norm(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    return float(np.sqrt(np.mean(np.sum(matrix**2, axis=0))))


def _tracking_metrics(actual_states: np.ndarray, reference_states: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    actual = decode_full_state_trajectory(actual_states)
    reference = decode_full_state_trajectory(reference_states)

    position_error = actual.x - reference.x
    velocity_error = actual.dx - reference.dx
    attitude_error = actual.theta - reference.theta
    wb_error = actual.wb - reference.wb

    position_error_norm = np.linalg.norm(position_error, axis=0)
    control_metrics = {
        "position_rmse_m": _rms_norm(position_error),
        "velocity_rmse_mps": _rms_norm(velocity_error),
        "theta_rmse_rad": _rms_norm(attitude_error),
        "wb_rmse_rad_s": _rms_norm(wb_error),
        "final_position_error_m": float(position_error_norm[-1]),
        "max_position_error_m": float(np.max(position_error_norm)),
    }
    component_payload = {
        "actual_x": actual.x,
        "actual_dx": actual.dx,
        "actual_theta": actual.theta,
        "actual_wb": actual.wb,
        "reference_x": reference.x,
        "reference_dx": reference.dx,
        "reference_theta": reference.theta,
        "reference_wb": reference.wb,
    }
    return control_metrics, component_payload


def _trajectory_rows(
    time_values: np.ndarray,
    actual_states: np.ndarray,
    reference_states: np.ndarray,
) -> list[dict[str, float]]:
    metrics, components = _tracking_metrics(actual_states, reference_states)
    del metrics

    rows = []
    for i, t_s in enumerate(time_values):
        rows.append(
            {
                "t_s": float(t_s),
                "x_ref": float(components["reference_x"][0, i]),
                "y_ref": float(components["reference_x"][1, i]),
                "z_ref": float(components["reference_x"][2, i]),
                "x_mpc": float(components["actual_x"][0, i]),
                "y_mpc": float(components["actual_x"][1, i]),
                "z_mpc": float(components["actual_x"][2, i]),
                "vx_ref": float(components["reference_dx"][0, i]),
                "vy_ref": float(components["reference_dx"][1, i]),
                "vz_ref": float(components["reference_dx"][2, i]),
                "vx_mpc": float(components["actual_dx"][0, i]),
                "vy_mpc": float(components["actual_dx"][1, i]),
                "vz_mpc": float(components["actual_dx"][2, i]),
                "theta_x_ref": float(components["reference_theta"][0, i]),
                "theta_y_ref": float(components["reference_theta"][1, i]),
                "theta_z_ref": float(components["reference_theta"][2, i]),
                "theta_x_mpc": float(components["actual_theta"][0, i]),
                "theta_y_mpc": float(components["actual_theta"][1, i]),
                "theta_z_mpc": float(components["actual_theta"][2, i]),
                "wb_x_ref": float(components["reference_wb"][0, i]),
                "wb_y_ref": float(components["reference_wb"][1, i]),
                "wb_z_ref": float(components["reference_wb"][2, i]),
                "wb_x_mpc": float(components["actual_wb"][0, i]),
                "wb_y_mpc": float(components["actual_wb"][1, i]),
                "wb_z_mpc": float(components["actual_wb"][2, i]),
            }
        )
    return rows


def run_offline_learned_mpc_experiment(
    config: OfflineLearnedMpcConfig,
    output_root: Path | None = None,
) -> Path:
    """Fit the EDMD model and run the first offline closed-loop learned MPC rollout."""

    v2_root = _v2_root()
    repo_root = _repo_root()
    results_root = output_root or (v2_root / "results")
    run_id = _timestamp_run_id()
    scenario_name = _effective_scenario_name(config)
    run_dir = results_root / "learned" / "unquantized" / scenario_name / run_id
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    params = get_params(config.parameter_profile)
    training_time_grid = _time_grid(config.training_dt, config.training_t_span)
    reference_time_grid = _time_grid(config.sim_time_step, _reference_duration(config))
    rng = np.random.default_rng(config.seed)

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
        reference_states = reference_batch.X[:, 1:]
    else:
        deterministic_reference = generate_reference_states(scenario_name, reference_time_grid)
        initial_state = deterministic_reference[:, 0]
        reference_states = deterministic_reference[:, 1:]

    fit_start = time.perf_counter()
    if config.training_source == "single_initial_state_random":
        train_batch = get_random_trajectories(
            initial_state=initial_state,
            n_control=config.training_n_control,
            t_traj=training_time_grid,
            mode="train",
            params=params,
            rng=rng,
        )
    elif config.training_source == "reference_seeded_random":
        train_batch = get_reference_seeded_random_trajectories(
            reference_states=np.column_stack((initial_state.reshape(18, 1), reference_states)),
            n_control_total=config.training_n_control,
            t_traj=training_time_grid,
            num_seed_states=config.training_seed_count,
            mode="train",
            params=params,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported training_source: {config.training_source}")
    model = fit_edmd(
        X1=train_batch.X1,
        X2=train_batch.X2,
        U1=train_batch.U1,
        n_basis=config.n_basis,
    )
    fit_duration_s = time.perf_counter() - fit_start

    training_observed_true = model.C @ model.Z2
    training_observed_pred = model.C @ (model.A @ model.Z1 + model.B @ train_batch.U1)
    training_rmse = compute_rmse(training_observed_pred, training_observed_true)

    lifted_reference = lift_trajectory(reference_states, config.n_basis)
    initial_lifted_state = lift_state(initial_state, config.n_basis)

    sim_start = time.perf_counter()
    mpc_result = simulate_closed_loop(
        model=model,
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
    tracking_metrics, _ = _tracking_metrics(actual_states, tracked_reference_states)

    actual_observed = model.C @ lift_trajectory(actual_states, model.n_basis)
    reference_observed = model.C @ lift_trajectory(tracked_reference_states, model.n_basis)
    normalized_tracking_rmse = compute_rmse(actual_observed, reference_observed)

    metrics_payload = {
        "run_id": run_id,
        "success": bool(np.all(np.isfinite(actual_states)) and np.all(np.isfinite(mpc_result.U))),
        "solver_backend": SOLVER_BACKEND,
        "training_rmse": asdict(training_rmse),
        "normalized_tracking_rmse": asdict(normalized_tracking_rmse),
        "tracking_metrics": tracking_metrics,
        "n_basis": config.n_basis,
        "training_n_control": config.training_n_control,
        "training_source": config.training_source,
        "training_seed_count": config.training_seed_count,
        "n_training_snapshots": int(train_batch.X1.shape[1]),
        "n_sim_steps": int(actual_states.shape[1]),
        "pred_horizon": config.pred_horizon,
        "effective_sim_duration": _effective_sim_duration(config),
        "fit_duration_ms": fit_duration_s * 1000.0,
        "simulation_duration_ms": sim_duration_s * 1000.0,
        "solve_time_ms_mean": float(np.mean(mpc_result.solve_times_ms)),
        "solve_time_ms_max": float(np.max(mpc_result.solve_times_ms)),
        "solve_iterations_mean": float(np.mean(mpc_result.solve_iterations)),
        "solve_iterations_max": int(np.max(mpc_result.solve_iterations)),
        "solve_converged_fraction": float(np.mean(mpc_result.solve_converged.astype(float))),
        "solve_projected_step_inf_norm_mean": float(np.mean(mpc_result.solve_projected_step_inf_norms)),
        "solve_projected_step_inf_norm_max": float(np.max(mpc_result.solve_projected_step_inf_norms)),
        "solve_hit_iteration_cap_fraction": float(np.mean(mpc_result.solve_hit_iteration_cap.astype(float))),
        "A_shape": list(model.A.shape),
        "B_shape": list(model.B.shape),
        "C_shape": list(model.C.shape),
        "parameter_profile": config.parameter_profile,
    }

    config_payload = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": scenario_name,
        "word_length": WORD_LENGTH,
        "realizations": 1,
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
    np.savez_compressed(
        run_dir / "model.npz",
        K=model.K,
        A=model.A,
        B=model.B,
        C=model.C,
        Z1=model.Z1,
        Z2=model.Z2,
        n_basis=model.n_basis,
    )
    np.savez_compressed(
        run_dir / "closed_loop.npz",
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

    trajectory_rows = _trajectory_rows(mpc_result.t, actual_states, tracked_reference_states)
    _write_csv(run_dir / "trajectory.csv", list(trajectory_rows[0].keys()), trajectory_rows)

    control_rows = _control_rows(mpc_result.t, mpc_result.U.T)
    _write_csv(run_dir / "control.csv", list(control_rows[0].keys()), control_rows)

    timing_rows = [
        {
            "step_index": int(i),
            "t_s": float(mpc_result.t[i]),
            "solve_time_ms": float(mpc_result.solve_times_ms[i]),
            "solve_iterations": int(mpc_result.solve_iterations[i]),
            "solve_converged": bool(mpc_result.solve_converged[i]),
            "solve_projected_step_inf_norm": float(mpc_result.solve_projected_step_inf_norms[i]),
            "solve_hit_iteration_cap": bool(mpc_result.solve_hit_iteration_cap[i]),
        }
        for i in range(mpc_result.solve_times_ms.size)
    ]
    _write_csv(
        run_dir / "timing.csv",
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

    generate_offline_learned_mpc_figures(
        figure_dir,
        time_values=mpc_result.t,
        actual_states=actual_states,
        reference_states=tracked_reference_states,
        control_matrix=mpc_result.U.T,
    )

    (run_dir / "notes.txt").write_text(
        "First offline closed-loop learned EDMD-MPC run.\n"
        f"The optimization backend is {SOLVER_BACKEND}, not MATLAB quadprog.\n"
        "SVG figures are generated automatically under figures/.\n",
        encoding="utf-8",
    )

    summary_root = results_root / "summary"
    run_index_row = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": scenario_name,
        "word_length": WORD_LENGTH,
        "realizations": 1,
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

    metric_rows = [
        {"run_id": run_id, "metric_name": "training_rmse_x", "metric_value": training_rmse.x, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "training_rmse_dx", "metric_value": training_rmse.dx, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "training_rmse_theta", "metric_value": training_rmse.theta, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "training_rmse_wb", "metric_value": training_rmse.wb, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "tracking_rmse_x", "metric_value": normalized_tracking_rmse.x, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "tracking_rmse_dx", "metric_value": normalized_tracking_rmse.dx, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "tracking_rmse_theta", "metric_value": normalized_tracking_rmse.theta, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "tracking_rmse_wb", "metric_value": normalized_tracking_rmse.wb, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "position_rmse_m", "metric_value": tracking_metrics["position_rmse_m"], "units": "m"},
        {"run_id": run_id, "metric_name": "velocity_rmse_mps", "metric_value": tracking_metrics["velocity_rmse_mps"], "units": "m/s"},
        {"run_id": run_id, "metric_name": "theta_rmse_rad", "metric_value": tracking_metrics["theta_rmse_rad"], "units": "rad"},
        {"run_id": run_id, "metric_name": "wb_rmse_rad_s", "metric_value": tracking_metrics["wb_rmse_rad_s"], "units": "rad/s"},
        {"run_id": run_id, "metric_name": "final_position_error_m", "metric_value": tracking_metrics["final_position_error_m"], "units": "m"},
        {"run_id": run_id, "metric_name": "max_position_error_m", "metric_value": tracking_metrics["max_position_error_m"], "units": "m"},
        {"run_id": run_id, "metric_name": "fit_duration_ms", "metric_value": fit_duration_s * 1000.0, "units": "ms"},
        {"run_id": run_id, "metric_name": "simulation_duration_ms", "metric_value": sim_duration_s * 1000.0, "units": "ms"},
        {"run_id": run_id, "metric_name": "solve_time_ms_mean", "metric_value": float(np.mean(mpc_result.solve_times_ms)), "units": "ms"},
        {"run_id": run_id, "metric_name": "solve_time_ms_max", "metric_value": float(np.max(mpc_result.solve_times_ms)), "units": "ms"},
        {"run_id": run_id, "metric_name": "solve_iterations_mean", "metric_value": float(np.mean(mpc_result.solve_iterations)), "units": "iterations"},
        {"run_id": run_id, "metric_name": "solve_iterations_max", "metric_value": int(np.max(mpc_result.solve_iterations)), "units": "iterations"},
        {"run_id": run_id, "metric_name": "solve_converged_fraction", "metric_value": float(np.mean(mpc_result.solve_converged.astype(float))), "units": "fraction"},
        {"run_id": run_id, "metric_name": "solve_projected_step_inf_norm_mean", "metric_value": float(np.mean(mpc_result.solve_projected_step_inf_norms)), "units": "norm"},
        {"run_id": run_id, "metric_name": "solve_projected_step_inf_norm_max", "metric_value": float(np.max(mpc_result.solve_projected_step_inf_norms)), "units": "norm"},
        {"run_id": run_id, "metric_name": "solve_hit_iteration_cap_fraction", "metric_value": float(np.mean(mpc_result.solve_hit_iteration_cap.astype(float))), "units": "fraction"},
    ]
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
            "word_length": WORD_LENGTH,
            "realizations": 1,
            "success": metrics_payload["success"],
            "hover_rmse_m": tracking_metrics["position_rmse_m"] if scenario_name == "hover_5s" else "",
            "tracking_rmse_m": "" if scenario_name == "hover_5s" else tracking_metrics["position_rmse_m"],
            "max_error_m": tracking_metrics["max_position_error_m"],
            "solve_time_ms_mean": float(np.mean(mpc_result.solve_times_ms)),
        },
    )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first offline closed-loop learned EDMD-MPC experiment.")
    parser.add_argument("--training-n-control", type=int, default=100)
    parser.add_argument("--training-dt", type=float, default=1e-3)
    parser.add_argument("--training-t-span", type=float, default=0.1)
    parser.add_argument("--training-source", type=str, default="single_initial_state_random", choices=["single_initial_state_random", "reference_seeded_random"])
    parser.add_argument("--training-seed-count", type=int, default=10)
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2141444)
    parser.add_argument("--parameter-profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--scenario-name", type=str, default="hover_5s", choices=available_scenarios())
    parser.add_argument("--pred-horizon", type=int, default=10)
    parser.add_argument("--sim-time-step", type=float, default=1e-3)
    parser.add_argument("--sim-duration", type=float, default=5.0)
    parser.add_argument("--control-lower-bound", type=float, default=-50.0)
    parser.add_argument("--control-upper-bound", type=float, default=50.0)
    parser.add_argument("--qp-max-iter", type=int, default=2000)
    parser.add_argument("--qp-tol", type=float, default=1e-8)
    parser.add_argument("--reference-mode", type=str, default="mpc", choices=["train", "val", "mpc"])
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OfflineLearnedMpcConfig(
        training_n_control=args.training_n_control,
        training_dt=args.training_dt,
        training_t_span=args.training_t_span,
        training_source=args.training_source,
        training_seed_count=args.training_seed_count,
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
    run_dir = run_offline_learned_mpc_experiment(config=config, output_root=args.output_root)
    print(f"offline_learned_mpc_run_dir={run_dir}")


if __name__ == "__main__":
    main()
