from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.config import HoverLocalOfflineConfig, hover_local_v1_profile, initial_state
from ..core.types import ExperimentOutput
from ..dynamics.params import get_params
from ..edmd.fit import get_edmd
from ..edmd.evaluate import eval_edmd_fixed_traj
from ..utils.io import create_run_directory, ensure_dir, save_npz, write_csv, write_json
from .training_data import get_hover_local_trajectories


def run_hover_local_experiment(config: HoverLocalOfflineConfig, tag: str | None = None) -> ExperimentOutput:
    rng = np.random.default_rng(config.random_seed)
    x0 = initial_state()
    t_traj = np.arange(0.0, config.train_traj_duration + config.dt, config.dt)
    output_dir = create_run_directory(config.output_root, config.profile_name, tag=tag)
    ensure_dir(output_dir)

    x_train, u_train, x1_train, x2_train, u1_train, _ = get_hover_local_trajectories(
        x0,
        config.training_n_control,
        t_traj,
        rng,
        collective_std_newton=config.collective_std_newton,
        collective_band_newton=config.collective_band_newton,
        body_moment_std_nm=config.body_moment_std(),
        body_moment_band_nm=config.body_moment_bounds(),
    )
    model = get_edmd(x1_train, x2_train, u1_train, config.n_basis)

    x_eval, u_eval, _, _, _, _ = get_hover_local_trajectories(
        x0,
        config.prediction_eval_n_control,
        t_traj,
        rng,
        collective_std_newton=config.collective_std_newton,
        collective_band_newton=config.collective_band_newton,
        body_moment_std_nm=config.body_moment_std(),
        body_moment_band_nm=config.body_moment_bounds(),
    )
    prediction_rmse, _, _ = eval_edmd_fixed_traj(
        x0,
        x_eval,
        u_eval,
        config.dt,
        model,
    )

    x_train_min = np.min(x_train, axis=1, keepdims=True)
    x_train_max = np.max(x_train, axis=1, keepdims=True)
    u_train_min = np.min(u1_train, axis=1, keepdims=True)
    u_train_max = np.max(u1_train, axis=1, keepdims=True)
    artifact_path = output_dir / "edmd_unquantized.npz"
    save_npz(
        artifact_path,
        A=model.A,
        B=model.B,
        C=model.C,
        Z1=model.Z1,
        Z2=model.Z2,
        n_basis=np.array([model.n_basis]),
        x_train_min=x_train_min,
        x_train_max=x_train_max,
        u_train_min=u_train_min,
        u_train_max=u_train_max,
    )

    params = get_params()
    hover_thrust_newton = float(params.mass * params.g)
    collective_min, collective_max = config.collective_bounds(hover_thrust_newton)
    metrics_csv = output_dir / "metrics_summary.csv"
    write_csv(
        metrics_csv,
        [
            {
                "profile_name": config.profile_name,
                "prediction_error_x": prediction_rmse.x,
                "prediction_error_dx": prediction_rmse.dx,
                "prediction_error_theta": prediction_rmse.theta,
                "prediction_error_wb": prediction_rmse.wb,
                "hover_thrust_newton": hover_thrust_newton,
                "train_collective_min_newton": collective_min,
                "train_collective_max_newton": collective_max,
                "train_body_moment_bound_x_nm": config.body_moment_bounds()[0],
                "train_body_moment_bound_y_nm": config.body_moment_bounds()[1],
                "train_body_moment_bound_z_nm": config.body_moment_bounds()[2],
            }
        ],
    )

    summary_json = output_dir / "summary.json"
    write_json(
        summary_json,
        {
            "config": config.to_dict(),
            "prediction_rmse": prediction_rmse.as_dict(),
            "training_dataset": {
                "trajectory_count": config.training_n_control,
                "trajectory_duration_s": config.train_traj_duration,
                "collective_bounds_newton": [collective_min, collective_max],
                "body_moment_bounds_nm": config.body_moment_band_nm,
                "state_min": x_train_min.reshape(-1).tolist(),
                "state_max": x_train_max.reshape(-1).tolist(),
                "control_min": u_train_min.reshape(-1).tolist(),
                "control_max": u_train_max.reshape(-1).tolist(),
            },
            "notes": [
                "This hover-local path is separate from the MATLAB parity offline pipeline.",
                "The artifact is trained on absolute collective thrust near hover with small body-moment perturbations.",
            ],
        },
    )

    return ExperimentOutput(
        root_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        plot_paths=[],
        artifact_paths=[artifact_path],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hover-local EDMD artifact for SITL debugging.")
    parser.add_argument("--results-root", type=Path, default=Path("results/offline"))
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    config = hover_local_v1_profile(output_root=args.results_root)
    output = run_hover_local_experiment(config, tag=args.tag)
    print(output.root_dir)


if __name__ == "__main__":
    main()
