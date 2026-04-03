from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..utils.io import write_json


def _load_optional_metadata(metadata_path: Path | None) -> dict[str, object]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _runtime_thresholds(control_rate_hz: float, reference_duration_s: float) -> dict[str, float]:
    if control_rate_hz >= 90.0:
        return {
            "sample_count_min": 850.0,
            "effective_control_rate_hz_min": 85.0,
            "tick_dt_ms_p90_max": 14.0,
            "solver_ms_p90_max": 10.0,
        }
    if control_rate_hz >= 45.0:
        return {
            "sample_count_min": 440.0,
            "effective_control_rate_hz_min": 45.0,
            "tick_dt_ms_p90_max": 25.0,
            "solver_ms_p90_max": 18.0,
        }
    target_samples = max(reference_duration_s * control_rate_hz, 1.0)
    target_period_ms = 1000.0 / max(control_rate_hz, 1.0)
    return {
        "sample_count_min": float(np.floor(0.85 * target_samples)),
        "effective_control_rate_hz_min": 0.85 * control_rate_hz,
        "tick_dt_ms_p90_max": 1.4 * target_period_ms,
        "solver_ms_p90_max": 0.9 * target_period_ms,
    }


def compute_runtime_health_summary(
    log_path: Path,
    *,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    resolved_log_path = Path(log_path)
    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else (resolved_log_path.parent / "run_metadata.json")
    metadata = _load_optional_metadata(resolved_metadata_path)

    with resolved_log_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{resolved_log_path} does not contain any runtime samples")

    sample_count = len(rows)
    experiment_time_s = np.asarray([float(row["experiment_time_s"]) for row in rows], dtype=float)
    tick_dt_ms = np.asarray([float(row["tick_dt_ms"]) for row in rows], dtype=float)
    solver_ms = np.asarray([float(row["solver_ms"]) for row in rows], dtype=float)
    final_time_s = float(experiment_time_s[-1])
    effective_control_rate_hz = float(max(sample_count - 1, 0) / final_time_s) if final_time_s > 0.0 else 0.0

    control_rate_hz = float(metadata.get("control_rate_hz", 0.0) or 0.0)
    pred_horizon = int(metadata.get("pred_horizon", 0) or 0)
    reference_duration_s = float(metadata.get("reference_duration_s", final_time_s) or final_time_s)
    thresholds = _runtime_thresholds(control_rate_hz, reference_duration_s)

    failure_reasons: list[str] = []
    if float(sample_count) < thresholds["sample_count_min"]:
        failure_reasons.append("sample_count")
    if effective_control_rate_hz < thresholds["effective_control_rate_hz_min"]:
        failure_reasons.append("effective_control_rate_hz")
    if _percentile(tick_dt_ms, 90.0) > thresholds["tick_dt_ms_p90_max"]:
        failure_reasons.append("tick_dt_ms_p90")
    if _percentile(solver_ms, 90.0) > thresholds["solver_ms_p90_max"]:
        failure_reasons.append("solver_ms_p90")

    return {
        "config_path": metadata.get("config_path"),
        "control_rate_hz": control_rate_hz,
        "pred_horizon": pred_horizon,
        "reference_duration_s": reference_duration_s,
        "sample_count": sample_count,
        "final_time_s": final_time_s,
        "effective_control_rate_hz": effective_control_rate_hz,
        "tick_dt_ms_mean": float(np.mean(tick_dt_ms)),
        "tick_dt_ms_p50": _percentile(tick_dt_ms, 50.0),
        "tick_dt_ms_p90": _percentile(tick_dt_ms, 90.0),
        "tick_dt_ms_max": float(np.max(tick_dt_ms)),
        "solver_ms_mean": float(np.mean(solver_ms)),
        "solver_ms_p50": _percentile(solver_ms, 50.0),
        "solver_ms_p90": _percentile(solver_ms, 90.0),
        "solver_ms_max": float(np.max(solver_ms)),
        "runtime_validity": "valid_runtime" if not failure_reasons else "timing_collapse",
        "runtime_failure_reason": None if not failure_reasons else ",".join(failure_reasons),
    }


def analyze_runtime_health(
    *,
    log_path: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    summary = compute_runtime_health_summary(log_path, metadata_path=metadata_path)
    write_json(Path(output_dir) / "runtime_health_summary.json", summary)
    return summary
