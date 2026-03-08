from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.types import MPCSimulation, ParsedStateSeries, WordLengthResult
from ..utils.state import parse_state18_history


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _set_log_scale_if_positive(ax, series: list[list[float]]) -> None:
    values = [value for sublist in series for value in sublist]
    if values and all(value > 0.0 for value in values):
        ax.set_yscale("log")


def plot_quantization_boxplots(
    results: list[WordLengthResult],
    output_dir: Path,
    include_unquantized: bool,
    prediction_unquantized: float,
    tracking_unquantized: float,
) -> list[Path]:
    labels = [entry.word_length for entry in results]
    prediction_data = [entry.prediction_error for entry in results]
    tracking_data = [entry.tracking_error for entry in results]
    if include_unquantized:
        labels = [*labels, "Inf"]
        prediction_data = [*prediction_data, [prediction_unquantized]]
        tracking_data = [*tracking_data, [tracking_unquantized]]

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(prediction_data, labels=labels)
    ax.set_xlabel("Word Length (bits)")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Prediction Error vs. Word Length")
    _set_log_scale_if_positive(ax, prediction_data)
    ax.grid(True)
    paths.append(_save(fig, output_dir / "prediction_error_comparison_log.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(tracking_data, labels=labels)
    ax.set_xlabel("Word Length (bits)")
    ax.set_ylabel("Tracking Error")
    ax.set_title("MPC Tracking Error vs. Word Length")
    _set_log_scale_if_positive(ax, tracking_data)
    ax.grid(True)
    paths.append(_save(fig, output_dir / "tracking_error_comparison_log.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([entry.matrix_a_difference for entry in results], labels=[entry.word_length for entry in results])
    ax.set_xlabel("Word Length (bits)")
    ax.set_ylabel(r"$||A - \bar{A}|| / ||A||$")
    ax.set_title("A Matrix Difference vs. Word Length")
    _set_log_scale_if_positive(ax, [entry.matrix_a_difference for entry in results])
    ax.grid(True)
    paths.append(_save(fig, output_dir / "matrix_a_difference.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([entry.matrix_b_difference for entry in results], labels=[entry.word_length for entry in results])
    ax.set_xlabel("Word Length (bits)")
    ax.set_ylabel(r"$||B - \bar{B}|| / ||B||$")
    ax.set_title("B Matrix Difference vs. Word Length")
    _set_log_scale_if_positive(ax, [entry.matrix_b_difference for entry in results])
    ax.grid(True)
    paths.append(_save(fig, output_dir / "matrix_b_difference.png"))

    return paths


def plot_mpc_trajectory_ensemble(
    ensemble: dict[int, list[np.ndarray]],
    reference_trajectory: np.ndarray,
    output_path: Path,
) -> Path:
    labels = sorted(ensemble.keys())
    fig = plt.figure(figsize=(12, 10))
    for idx, bits in enumerate(labels[:4], start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        for trajectory in ensemble[bits]:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.35)
        ax.plot(
            reference_trajectory[:, 0],
            reference_trajectory[:, 1],
            reference_trajectory[:, 2],
            "k--",
            linewidth=2.5,
            label="reference",
        )
        ax.set_title(f"b = {bits} bits")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    return _save(fig, output_path)


def plot_state_series(simulation: MPCSimulation, output_path: Path) -> Path:
    x_series = parse_state18_history(simulation.X.T)
    x_ref_series = parse_state18_history(simulation.X_ref.T)
    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    plot_pairs = [
        ("x", 0), ("x", 1), ("x", 2),
        ("dx", 0), ("dx", 1), ("dx", 2),
        ("theta", 0), ("theta", 1), ("theta", 2),
        ("wb", 0), ("wb", 1), ("wb", 2),
    ]
    for axis, (field, component) in zip(axes.ravel(), plot_pairs, strict=False):
        axis.plot(simulation.t, getattr(x_ref_series, field)[component, :], label="reference")
        axis.plot(simulation.t, getattr(x_series, field)[component, :], "--", label="MPC")
        axis.set_ylabel(f"{field}[{component}]")
        axis.grid(True)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("t (s)")
    axes[-1, 1].set_xlabel("t (s)")
    return _save(fig, output_path)


def plot_control_series(simulation: MPCSimulation, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    labels = ["f_t", "M_x", "M_y", "M_z"]
    for idx, axis in enumerate(axes.ravel()):
        axis.plot(simulation.t, simulation.U[:, idx])
        axis.set_ylabel(labels[idx])
        axis.grid(True)
    axes[-1, 0].set_xlabel("t (s)")
    axes[-1, 1].set_xlabel("t (s)")
    return _save(fig, output_path)
