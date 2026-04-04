"""Offline unquantized EDMD experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from koopman_python.dynamics.params import DEFAULT_PROFILE, get_params
from koopman_python.edmd.evaluate import compute_rmse, evaluate_edmd_fixed_trajectory
from koopman_python.edmd.fit import fit_edmd
from koopman_python.training import get_random_trajectories


RUN_FAMILY = "learned"
CONTROLLER_VARIANT = "learned_edmd_mpc"
SCENARIO = "offline_validation"
WORD_LENGTH = "unquantized"


@dataclass(frozen=True)
class OfflineUnquantizedConfig:
    n_control: int = 100
    dt: float = 1e-3
    t_span: float = 0.1
    n_basis: int = 3
    seed: int = 2141444
    parameter_profile: str = DEFAULT_PROFILE
    eval_mode: str = "val"


def _v2_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_offline_unquantized")


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _default_initial_state() -> np.ndarray:
    state = np.zeros(18, dtype=float)
    state[6] = 1.0
    state[10] = 1.0
    state[14] = 1.0
    state[15] = 0.1
    return state


def _time_grid(dt: float, t_span: float) -> np.ndarray:
    steps = int(round(t_span / dt))
    return np.linspace(0.0, dt * steps, steps + 1)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def _trajectory_rows(time_grid: np.ndarray, evaluation_result) -> list[dict[str, float]]:
    rows = []
    for i, t_s in enumerate(time_grid):
        rows.append(
            {
                "t_s": float(t_s),
                "x_true": float(evaluation_result.true_components.x[0, i]),
                "y_true": float(evaluation_result.true_components.x[1, i]),
                "z_true": float(evaluation_result.true_components.x[2, i]),
                "x_pred": float(evaluation_result.pred_components.x[0, i]),
                "y_pred": float(evaluation_result.pred_components.x[1, i]),
                "z_pred": float(evaluation_result.pred_components.x[2, i]),
                "vx_true": float(evaluation_result.true_components.dx[0, i]),
                "vy_true": float(evaluation_result.true_components.dx[1, i]),
                "vz_true": float(evaluation_result.true_components.dx[2, i]),
                "vx_pred": float(evaluation_result.pred_components.dx[0, i]),
                "vy_pred": float(evaluation_result.pred_components.dx[1, i]),
                "vz_pred": float(evaluation_result.pred_components.dx[2, i]),
                "theta_x_true": float(evaluation_result.true_components.theta[0, i]),
                "theta_y_true": float(evaluation_result.true_components.theta[1, i]),
                "theta_z_true": float(evaluation_result.true_components.theta[2, i]),
                "theta_x_pred": float(evaluation_result.pred_components.theta[0, i]),
                "theta_y_pred": float(evaluation_result.pred_components.theta[1, i]),
                "theta_z_pred": float(evaluation_result.pred_components.theta[2, i]),
                "wb_x_true": float(evaluation_result.true_components.wb[0, i]),
                "wb_y_true": float(evaluation_result.true_components.wb[1, i]),
                "wb_z_true": float(evaluation_result.true_components.wb[2, i]),
                "wb_x_pred": float(evaluation_result.pred_components.wb[0, i]),
                "wb_y_pred": float(evaluation_result.pred_components.wb[1, i]),
                "wb_z_pred": float(evaluation_result.pred_components.wb[2, i]),
            }
        )
    return rows


def _control_rows(time_grid: np.ndarray, control_matrix: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for i, t_s in enumerate(time_grid):
        rows.append(
            {
                "t_s": float(t_s),
                "Fb": float(control_matrix[0, i]),
                "Mbx": float(control_matrix[1, i]),
                "Mby": float(control_matrix[2, i]),
                "Mbz": float(control_matrix[3, i]),
            }
        )
    return rows


def run_offline_unquantized_experiment(
    config: OfflineUnquantizedConfig,
    output_root: Path | None = None,
) -> Path:
    """Run the first end-to-end offline EDMD experiment and save artifacts."""

    v2_root = _v2_root()
    repo_root = _repo_root()
    results_root = output_root or (v2_root / "results")
    run_id = _timestamp_run_id()
    run_dir = results_root / "learned" / "unquantized" / SCENARIO / run_id
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    params = get_params(config.parameter_profile)
    initial_state = _default_initial_state()
    time_grid = _time_grid(config.dt, config.t_span)
    rng = np.random.default_rng(config.seed)

    fit_start = time.perf_counter()
    train_batch = get_random_trajectories(
        initial_state=initial_state,
        n_control=config.n_control,
        t_traj=time_grid,
        mode="train",
        params=params,
        rng=rng,
    )
    model = fit_edmd(
        X1=train_batch.X1,
        X2=train_batch.X2,
        U1=train_batch.U1,
        n_basis=config.n_basis,
    )
    fit_duration_s = time.perf_counter() - fit_start

    training_observed_true = model.C @ model.Z2
    training_observed_pred = model.A @ model.Z1 + model.B @ train_batch.U1
    training_observed_pred = model.C @ training_observed_pred
    training_rmse = compute_rmse(training_observed_pred, training_observed_true)

    eval_start = time.perf_counter()
    eval_batch = get_random_trajectories(
        initial_state=initial_state,
        n_control=1,
        t_traj=time_grid,
        mode=config.eval_mode,
        params=params,
        rng=rng,
    )
    evaluation = evaluate_edmd_fixed_trajectory(
        initial_state=initial_state,
        true_states=eval_batch.X[:, : time_grid.size],
        controls=eval_batch.U1[:, : time_grid.size - 1],
        model=model,
    )
    eval_duration_s = time.perf_counter() - eval_start

    metrics_payload = {
        "run_id": run_id,
        "success": True,
        "training_rmse": asdict(training_rmse),
        "validation_rmse": asdict(evaluation.rmse),
        "n_basis": config.n_basis,
        "n_control": config.n_control,
        "n_training_snapshots": int(train_batch.X1.shape[1]),
        "n_validation_steps": int(time_grid.size - 1),
        "fit_duration_ms": fit_duration_s * 1000.0,
        "evaluation_duration_ms": eval_duration_s * 1000.0,
        "A_shape": list(model.A.shape),
        "B_shape": list(model.B.shape),
        "C_shape": list(model.C.shape),
        "parameter_profile": config.parameter_profile,
    }

    config_payload = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": SCENARIO,
        "word_length": WORD_LENGTH,
        "realizations": 1,
        **asdict(config),
        "time_grid_length": int(time_grid.size),
    }

    environment_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(repo_root),
        "parameter_profile": config.parameter_profile,
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
    )

    trajectory_rows = _trajectory_rows(time_grid, evaluation)
    _write_csv(run_dir / "trajectory.csv", list(trajectory_rows[0].keys()), trajectory_rows)

    evaluation_controls = eval_batch.U[:, : time_grid.size]
    control_rows = _control_rows(time_grid, evaluation_controls)
    _write_csv(run_dir / "control.csv", list(control_rows[0].keys()), control_rows)

    timing_rows = [
        {"stage": "fit_edmd", "duration_ms": fit_duration_s * 1000.0},
        {"stage": "evaluate_edmd_fixed_trajectory", "duration_ms": eval_duration_s * 1000.0},
    ]
    _write_csv(run_dir / "timing.csv", ["stage", "duration_ms"], timing_rows)

    (run_dir / "notes.txt").write_text(
        "First offline unquantized EDMD validation run.\n"
        "No figures are generated yet; use trajectory.csv and metrics.json for plotting.\n",
        encoding="utf-8",
    )

    summary_root = results_root / "summary"
    run_index_row = {
        "run_id": run_id,
        "run_family": RUN_FAMILY,
        "controller_variant": CONTROLLER_VARIANT,
        "scenario": SCENARIO,
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
        {"run_id": run_id, "metric_name": "validation_rmse_x", "metric_value": evaluation.rmse.x, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "validation_rmse_dx", "metric_value": evaluation.rmse.dx, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "validation_rmse_theta", "metric_value": evaluation.rmse.theta, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "validation_rmse_wb", "metric_value": evaluation.rmse.wb, "units": "normalized_rmse"},
        {"run_id": run_id, "metric_name": "fit_duration_ms", "metric_value": fit_duration_s * 1000.0, "units": "ms"},
        {"run_id": run_id, "metric_name": "evaluation_duration_ms", "metric_value": eval_duration_s * 1000.0, "units": "ms"},
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
            "scenario": SCENARIO,
            "word_length": WORD_LENGTH,
            "realizations": 1,
            "success": True,
            "hover_rmse_m": "",
            "tracking_rmse_m": evaluation.rmse.x,
            "max_error_m": "",
            "solve_time_ms_mean": fit_duration_s * 1000.0,
        },
    )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline unquantized EDMD validation experiment.")
    parser.add_argument("--n-control", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--t-span", type=float, default=0.1)
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2141444)
    parser.add_argument("--parameter-profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--eval-mode", type=str, default="val", choices=["train", "val", "mpc"])
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OfflineUnquantizedConfig(
        n_control=args.n_control,
        dt=args.dt,
        t_span=args.t_span,
        n_basis=args.n_basis,
        seed=args.seed,
        parameter_profile=args.parameter_profile,
        eval_mode=args.eval_mode,
    )
    run_dir = run_offline_unquantized_experiment(config=config, output_root=args.output_root)
    print(f"offline_unquantized_run_dir={run_dir}")


if __name__ == "__main__":
    main()
