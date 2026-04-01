from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class EDMDModel:
    A: FloatArray
    B: FloatArray
    C: FloatArray
    Z1: FloatArray
    Z2: FloatArray
    n_basis: int
    bias: FloatArray | None = None
    affine_enabled: bool = False

    def affine_bias(self) -> FloatArray:
        if not self.affine_enabled or self.bias is None:
            return np.zeros(self.A.shape[0], dtype=float)
        return np.asarray(self.bias, dtype=float).reshape(self.A.shape[0])

    def predict_next_lifted(self, lifted_state: FloatArray, control: FloatArray) -> FloatArray:
        state = np.asarray(lifted_state, dtype=float).reshape(self.A.shape[1])
        command = np.asarray(control, dtype=float).reshape(self.B.shape[1])
        return self.A @ state + self.B @ command + self.affine_bias()


@dataclass(slots=True)
class RMSEBreakdown:
    x: float
    dx: float
    theta: float
    wb: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "dx": float(self.dx),
            "theta": float(self.theta),
            "wb": float(self.wb),
        }


@dataclass(slots=True)
class ParsedStateSeries:
    x: FloatArray
    dx: FloatArray
    theta: FloatArray
    wb: FloatArray


@dataclass(slots=True)
class MPCSimulation:
    t: FloatArray
    X: FloatArray
    U: FloatArray
    X_ref: FloatArray


@dataclass(slots=True)
class WordLengthResult:
    word_length: str
    matrix_a_difference: list[float] = field(default_factory=list)
    matrix_b_difference: list[float] = field(default_factory=list)
    prediction_error: list[float] = field(default_factory=list)
    tracking_error: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | str]:
        def _mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        def _std(values: list[float]) -> float:
            return float(np.std(values)) if values else float("nan")

        return {
            "word_length": self.word_length,
            "matrix_a_difference_mean": _mean(self.matrix_a_difference),
            "matrix_a_difference_std": _std(self.matrix_a_difference),
            "matrix_b_difference_mean": _mean(self.matrix_b_difference),
            "matrix_b_difference_std": _std(self.matrix_b_difference),
            "prediction_error_mean": _mean(self.prediction_error),
            "prediction_error_std": _std(self.prediction_error),
            "tracking_error_mean": _mean(self.tracking_error),
            "tracking_error_std": _std(self.tracking_error),
        }


@dataclass(slots=True)
class ExperimentOutput:
    root_dir: Path
    metrics_csv: Path
    summary_json: Path
    plot_paths: list[Path]
    artifact_paths: list[Path]
