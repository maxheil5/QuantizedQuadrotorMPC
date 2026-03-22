from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.types import ExperimentOutput
from ..edmd.fit import get_edmd
from ..utils.io import create_run_directory, save_npz, write_csv, write_json
from .sitl_dataset import build_sitl_edmd_snapshots, evaluate_model_on_sitl_run, load_sitl_run_dataset


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
    model = get_edmd(x1_history, x2_history, u1_history, n_basis)

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
        u_train_min=np.min(u1_history, axis=1, keepdims=True),
        u_train_max=np.max(u1_history, axis=1, keepdims=True),
    )

    metrics_rows: list[dict[str, object]] = []
    for split_name, datasets in (("train", train_datasets), ("eval", eval_datasets if eval_datasets else train_datasets)):
        for dataset in datasets:
            scores = evaluate_model_on_sitl_run(dataset, model)
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
                }
            )

    metrics_csv = output_dir / "metrics_summary.csv"
    write_csv(metrics_csv, metrics_rows)
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
            "notes": [
                "This path fits EDMD from actual SITL runtime logs rather than the simplified SRB training simulator.",
                "The existing paper/offline parity paths are intentionally left untouched.",
                "Audit finding: the current random offline generator uses one constant control per trajectory, while the manuscript text describes random controls applied at each time step.",
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
