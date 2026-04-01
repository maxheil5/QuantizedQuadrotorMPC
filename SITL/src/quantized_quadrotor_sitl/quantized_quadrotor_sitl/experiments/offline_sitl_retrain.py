from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.types import ExperimentOutput
from ..edmd.fit import get_edmd
from ..utils.io import create_run_directory, save_npz, write_csv, write_json
from .sitl_dataset import (
    build_sitl_edmd_snapshots,
    compute_sitl_run_diagnostics,
    evaluate_model_on_sitl_run_with_controls,
    excitation_warnings_from_diagnostics,
    load_sitl_run_dataset,
)


def _resolve_run_spec(runs_root: Path, raw_spec: str) -> Path:
    raw_path = Path(raw_spec)
    if raw_path.is_absolute():
        return raw_path if raw_path.name == "runtime_log.csv" else raw_path / "runtime_log.csv"
    candidate = runs_root / raw_path
    return candidate if candidate.name == "runtime_log.csv" else candidate / "runtime_log.csv"


def run_sitl_retrain(
    train_runs: list[Path],
    eval_runs: list[Path],
    results_root: Path,
    state_source: str,
    control_source: str,
    n_basis: int,
    tag: str | None = None,
) -> ExperimentOutput:
    output_dir = create_run_directory(results_root, "sitl_baseline_v1", tag=tag)

    train_datasets = [load_sitl_run_dataset(path, state_source=state_source, control_source=control_source) for path in train_runs]
    eval_datasets = [load_sitl_run_dataset(path, state_source=state_source, control_source=control_source) for path in eval_runs]
    x_history, u_history, x1_history, x2_history, u1_history = build_sitl_edmd_snapshots(train_datasets)
    u_train_min = np.min(u1_history, axis=1, keepdims=True)
    u_train_max = np.max(u1_history, axis=1, keepdims=True)
    u_train_mean = np.mean(u1_history, axis=1, keepdims=True)
    u_train_std = np.std(u1_history, axis=1, keepdims=True)
    u_trim = u_train_mean.copy()
    u_scale = np.maximum(u_train_std, 1.0e-6)
    u1_internal = (u1_history - u_trim) / u_scale
    model = get_edmd(x1_history, x2_history, u1_internal, n_basis)

    artifact_path = output_dir / "edmd_unquantized.npz"
    save_npz(
        artifact_path,
        A=model.A,
        B=model.B,
        C=model.C,
        Z1=model.Z1,
        Z2=model.Z2,
        n_basis=np.array([model.n_basis]),
        x_train_min=np.min(x_history, axis=1, keepdims=True),
        x_train_max=np.max(x_history, axis=1, keepdims=True),
        u_train_min=u_train_min,
        u_train_max=u_train_max,
        u_train_mean=u_train_mean,
        u_train_std=u_train_std,
        u_trim=u_trim,
    )

    metrics_rows: list[dict[str, object]] = []
    train_diagnostics = [compute_sitl_run_diagnostics(dataset) for dataset in train_datasets]
    eval_diagnostics = [compute_sitl_run_diagnostics(dataset) for dataset in (eval_datasets if eval_datasets else train_datasets)]
    for split_name, datasets in (("train", train_datasets), ("eval", eval_datasets if eval_datasets else train_datasets)):
        for dataset in datasets:
            scores = evaluate_model_on_sitl_run_with_controls(dataset, model, control_trim=u_trim, control_scale=u_scale)
            diagnostics = compute_sitl_run_diagnostics(dataset)
            metrics_rows.append(
                {
                    "split": split_name,
                    "run_name": dataset.run_name,
                    "samples": dataset.sample_count,
                    "pairs": dataset.pair_count,
                    "tick_dt_ms_mean": float(np.mean(dataset.tick_dt_ms)),
                    "tick_dt_ms_std": float(np.std(dataset.tick_dt_ms)),
                    "solver_ms_mean": float(np.mean(dataset.solver_ms)),
                    "rmse_x": float(scores.x),
                    "rmse_dx": float(scores.dx),
                    "rmse_theta": float(scores.theta),
                    "rmse_wb": float(scores.wb),
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if key
                        not in {
                            "run_name",
                            "samples",
                            "pairs",
                            "tick_dt_ms_mean",
                            "tick_dt_ms_std",
                            "solver_ms_mean",
                        }
                    },
                }
            )

    metrics_csv = output_dir / "metrics_summary.csv"
    write_csv(metrics_csv, metrics_rows)
    excitation_thresholds = {
        "collective_std_min_newton": 1.0,
        "body_moment_std_min_nm": [0.02, 0.02, 0.01],
        "state_z_range_min_m": 0.30,
        "state_xy_range_min_m": 0.10,
        "collective_below_1n_fraction_max": 0.25,
    }
    warnings = excitation_warnings_from_diagnostics(
        train_diagnostics,
        collective_std_threshold=float(excitation_thresholds["collective_std_min_newton"]),
        body_moment_std_threshold=np.asarray(excitation_thresholds["body_moment_std_min_nm"], dtype=float),
        z_range_threshold=float(excitation_thresholds["state_z_range_min_m"]),
        xy_range_threshold=float(excitation_thresholds["state_xy_range_min_m"]),
        collective_below_1n_fraction_threshold=float(excitation_thresholds["collective_below_1n_fraction_max"]),
    )
    summary_json = output_dir / "summary.json"
    write_json(
        summary_json,
        {
            "profile_name": "sitl_baseline_v1",
            "train_runs": [str(path) for path in train_runs],
            "eval_runs": [str(path) for path in eval_runs],
            "state_source": state_source,
            "control_source": control_source,
            "n_basis": n_basis,
            "u_train_mean": u_train_mean.reshape(-1).tolist(),
            "u_train_std": u_train_std.reshape(-1).tolist(),
            "u_trim": u_trim.reshape(-1).tolist(),
            "excitation_thresholds": excitation_thresholds,
            "warnings": warnings,
            "train_diagnostics": train_diagnostics,
            "eval_diagnostics": eval_diagnostics,
            "notes": [
                "This path fits EDMD from actual SITL runtime logs rather than the simplified SRB training simulator.",
                "The existing paper/offline parity paths are intentionally left untouched.",
                "Audit finding: the current random offline generator uses one constant control per trajectory, while the manuscript text describes random controls applied at each time step.",
            ],
        },
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    return ExperimentOutput(
        root_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        plot_paths=[],
        artifact_paths=[artifact_path],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an EDMD artifact from stable SITL runtime logs.")
    parser.add_argument("--train-run", action="append", required=True, help="SITL results folder name or runtime_log.csv path")
    parser.add_argument("--eval-run", action="append", default=[], help="Optional held-out SITL results folder name or runtime_log.csv path")
    parser.add_argument("--runs-root", type=Path, default=Path("results/sitl"))
    parser.add_argument("--results-root", type=Path, default=Path("results/offline"))
    parser.add_argument("--state-source", choices=["raw", "used"], default="raw")
    parser.add_argument("--control-source", choices=["raw", "used"], default="used")
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    train_runs = [_resolve_run_spec(args.runs_root, raw_spec) for raw_spec in args.train_run]
    eval_runs = [_resolve_run_spec(args.runs_root, raw_spec) for raw_spec in args.eval_run]
    output = run_sitl_retrain(
        train_runs=train_runs,
        eval_runs=eval_runs,
        results_root=args.results_root,
        state_source=args.state_source,
        control_source=args.control_source,
        n_basis=args.n_basis,
        tag=args.tag,
    )
    print(output.root_dir)


if __name__ == "__main__":
    main()
