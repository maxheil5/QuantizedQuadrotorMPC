from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.config import OfflineExperimentConfig, initial_state, matlab_v2_profile, paper_v2_profile
from ..core.types import EDMDModel, ExperimentOutput, WordLengthResult
from ..edmd.basis import lift_state
from ..edmd.evaluate import eval_edmd_fixed_traj
from ..edmd.fit import get_edmd
from ..mpc.simulate import sim_mpc
from ..plotting.matlab_style import (
    plot_control_series,
    plot_mpc_trajectory_ensemble,
    plot_quantization_boxplots,
    plot_state_series,
)
from ..quantization.dither import dither_signal
from ..quantization.partition import partition_range
from ..utils.io import create_run_directory, ensure_dir, save_npz, write_csv, write_json
from ..utils.metrics import position_tracking_rmse
from .training_data import get_random_trajectories


def _split_dithered_state_snapshots(dithered_state: np.ndarray, n_control: int, trajectory_length: int) -> tuple[np.ndarray, np.ndarray]:
    x1_columns: list[np.ndarray] = []
    x2_columns: list[np.ndarray] = []
    for idx in range(n_control):
        lower = idx * trajectory_length
        upper = (idx + 1) * trajectory_length - 1
        x1_columns.append(dithered_state[:, lower:upper])
        x2_columns.append(dithered_state[:, lower + 1 : upper + 1])
    return np.hstack(x1_columns), np.hstack(x2_columns)


def _reference_tracking_bundle(config: OfflineExperimentConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x0 = initial_state()
    t_traj = np.arange(0.0, config.reference_horizon_duration + config.dt, config.dt)
    x_ref, _, _, _, _, _ = get_random_trajectories(x0, 1, t_traj, "mpc", rng)
    physical_reference = x_ref[:, 1:]
    lifted_reference = np.column_stack(
        [lift_state(physical_reference[:, idx], config.n_basis) for idx in range(physical_reference.shape[1])]
    )
    return physical_reference, lifted_reference


def _tracking_rmse(config: OfflineExperimentConfig, model: EDMDModel, physical_reference: np.ndarray, lifted_reference: np.ndarray) -> tuple[float, object]:
    z0 = lift_state(initial_state(), config.n_basis)
    simulation = sim_mpc(model, z0, lifted_reference, physical_reference, config.mpc)
    tracking_error = position_tracking_rmse(simulation.X[:, 0:3].T, simulation.X_ref[:, 0:3].T)
    return tracking_error, simulation


def run_offline_experiment(config: OfflineExperimentConfig, tag: str | None = None) -> ExperimentOutput:
    rng = np.random.default_rng(config.random_seed)
    x0 = initial_state()
    t_traj = np.arange(0.0, config.train_traj_duration + config.dt, config.dt)
    trajectory_length = t_traj.size
    output_dir = create_run_directory(config.output_root, config.profile_name, tag=tag)
    ensure_dir(output_dir)

    x_train, u_train, x1_train, x2_train, u1_train, _ = get_random_trajectories(
        x0,
        config.training_n_control,
        t_traj,
        "train",
        rng,
    )
    model_unquantized = get_edmd(x1_train, x2_train, u1_train, config.n_basis)
    x_eval_unquantized, u_eval_unquantized, _, _, _, _ = get_random_trajectories(
        x0,
        config.training_n_control,
        t_traj,
        "train",
        rng,
    )
    rmse_unquantized, _, _ = eval_edmd_fixed_traj(
        x0,
        x_eval_unquantized,
        u_eval_unquantized,
        config.dt,
        model_unquantized,
    )
    prediction_error_unquantized = rmse_unquantized.x
    tracking_error_unquantized = 0.0

    physical_reference = None
    lifted_reference = None
    if config.tracking_enabled:
        physical_reference, lifted_reference = _reference_tracking_bundle(config, rng)
        tracking_error_unquantized, simulation_unquantized = _tracking_rmse(
            config,
            model_unquantized,
            physical_reference,
            lifted_reference,
        )
    else:
        simulation_unquantized = None

    results_by_word_length = [WordLengthResult(word_length=str(bits)) for bits in config.word_lengths]
    ensemble_by_word_length: dict[int, list[np.ndarray]] = {bits: [] for bits in config.paper_tracking_word_lengths}
    tracked_examples: dict[int, object] = {}

    for run_idx in range(config.run_count):
        for result in results_by_word_length:
            bits = int(result.word_length)

            x_min = np.min(x_train, axis=1, keepdims=True)
            x_max = np.max(x_train, axis=1, keepdims=True)
            epsilon_x, x_min_new, x_max_new, _, mid_points_x = partition_range(x_min, x_max, bits)
            x_dithered, _ = dither_signal(x_train, epsilon_x, x_min_new, x_max_new, mid_points_x, rng)
            x1_dithered, x2_dithered = _split_dithered_state_snapshots(
                x_dithered,
                config.training_n_control,
                trajectory_length,
            )

            u_min = np.min(u1_train, axis=1, keepdims=True)
            u_max = np.max(u1_train, axis=1, keepdims=True)
            epsilon_u, u_min_new, u_max_new, _, mid_points_u = partition_range(u_min, u_max, bits)
            u1_dithered, _ = dither_signal(u1_train, epsilon_u, u_min_new, u_max_new, mid_points_u, rng)

            model_dithered = get_edmd(x1_dithered, x2_dithered, u1_dithered, config.n_basis)
            result.matrix_a_difference.append(
                float(np.linalg.norm(model_unquantized.A - model_dithered.A, ord="fro") / np.linalg.norm(model_unquantized.A, ord="fro"))
            )
            result.matrix_b_difference.append(
                float(np.linalg.norm(model_unquantized.B - model_dithered.B, ord="fro") / np.linalg.norm(model_unquantized.B, ord="fro"))
            )

            x_eval, u_eval, _, _, _, _ = get_random_trajectories(x0, 1, t_traj, "train", rng)
            rmse_dithered, _, _ = eval_edmd_fixed_traj(x0, x_eval, u_eval, config.dt, model_dithered)
            result.prediction_error.append(float(rmse_dithered.x))

            if config.tracking_enabled and physical_reference is not None and lifted_reference is not None:
                tracking_error, simulation = _tracking_rmse(config, model_dithered, physical_reference, lifted_reference)
                result.tracking_error.append(float(tracking_error))
                if bits in ensemble_by_word_length:
                    ensemble_by_word_length[bits].append(simulation.X[:, 0:3])
                    tracked_examples.setdefault(bits, simulation)
            else:
                result.tracking_error.append(0.0)

            artifact_name = f"edmd_bits_{bits:02d}_run_{run_idx + 1:02d}.npz"
            save_npz(
                output_dir / artifact_name,
                A=model_dithered.A,
                B=model_dithered.B,
                C=model_dithered.C,
                Z1=model_dithered.Z1,
                Z2=model_dithered.Z2,
                n_basis=np.array([model_dithered.n_basis]),
                x_train_min=x_min,
                x_train_max=x_max,
                u_train_min=u_min,
                u_train_max=u_max,
            )

    save_npz(
        output_dir / "edmd_unquantized.npz",
        A=model_unquantized.A,
        B=model_unquantized.B,
        C=model_unquantized.C,
        Z1=model_unquantized.Z1,
        Z2=model_unquantized.Z2,
        n_basis=np.array([model_unquantized.n_basis]),
        x_train_min=np.min(x_train, axis=1, keepdims=True),
        x_train_max=np.max(x_train, axis=1, keepdims=True),
        u_train_min=np.min(u1_train, axis=1, keepdims=True),
        u_train_max=np.max(u1_train, axis=1, keepdims=True),
    )

    plot_paths = plot_quantization_boxplots(
        results_by_word_length,
        output_dir,
        include_unquantized=config.include_unquantized,
        prediction_unquantized=prediction_error_unquantized,
        tracking_unquantized=tracking_error_unquantized,
    )

    if config.tracking_enabled and physical_reference is not None:
        plot_paths.append(
            plot_mpc_trajectory_ensemble(
                ensemble_by_word_length,
                physical_reference[:3, : config.mpc.max_iter].T,
                output_dir / "tracking_ensemble.png",
            )
        )
        representative_bits = 12 if 12 in tracked_examples else config.word_lengths[-1]
        representative = tracked_examples[representative_bits]
        plot_paths.append(plot_state_series(representative, output_dir / f"state_plots_bits_{representative_bits}.png"))
        plot_paths.append(plot_control_series(representative, output_dir / f"control_plots_bits_{representative_bits}.png"))
        if simulation_unquantized is not None:
            plot_paths.append(plot_state_series(simulation_unquantized, output_dir / "state_plots_unquantized.png"))
            plot_paths.append(plot_control_series(simulation_unquantized, output_dir / "control_plots_unquantized.png"))

    rows = [result.summary() for result in results_by_word_length]
    metrics_csv = output_dir / "metrics_summary.csv"
    write_csv(metrics_csv, rows)
    summary_json = output_dir / "summary.json"
    write_json(
        summary_json,
        {
            "config": config.to_dict(),
            "prediction_error_unquantized": prediction_error_unquantized,
            "tracking_error_unquantized": tracking_error_unquantized,
            "word_lengths": rows,
            "notes": [
                "MATLAB behavior is preserved ahead of paper text where they differ.",
                "Dither amplitude follows MATLAB/Dither_Func.m, which is narrower than the Schuchman interval stated in the paper.",
            ],
        },
    )
    return ExperimentOutput(
        root_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        plot_paths=plot_paths,
        artifact_paths=sorted(output_dir.glob("*.npz")),
    )


def _profile_from_name(name: str, output_root: Path | None = None) -> OfflineExperimentConfig:
    if name == "matlab_v2":
        return matlab_v2_profile(output_root=output_root)
    if name == "paper_v2":
        return paper_v2_profile(output_root=output_root)
    raise ValueError(f"unsupported profile: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline MATLAB-parity experiment.")
    parser.add_argument("--profile", choices=["matlab_v2", "paper_v2"], default="matlab_v2")
    parser.add_argument("--results-root", type=Path, default=Path("results/offline"))
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    config = _profile_from_name(args.profile, output_root=args.results_root)
    output = run_offline_experiment(config, tag=args.tag)
    print(output.root_dir)


if __name__ == "__main__":
    main()
