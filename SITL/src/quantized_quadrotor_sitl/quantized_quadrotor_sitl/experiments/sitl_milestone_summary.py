from __future__ import annotations

import csv
import json
from pathlib import Path

from .sitl_runtime_metrics import evaluate_hover_gates


MILESTONE_SUMMARY_FIELDS = [
    "run_name",
    "run_dir",
    "config_path",
    "controller_mode",
    "control_rate_hz",
    "pred_horizon",
    "reference_seed",
    "runtime_validity",
    "runtime_failure_reason",
    "hover_gate_profile",
    "hover_gate_passed",
    "controller_quality_evaluated",
    "final_altitude_error",
    "max_lateral_error_radius",
    "final_position_error",
    "position_rmse",
    "solver_mean_ms",
    "divergence_time_s",
    "u2_first_used_mismatch_time_s",
]


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def milestone_profile_for_metadata(metadata: dict[str, object]) -> str:
    controller_mode = str(metadata.get("controller_mode", ""))
    control_rate_hz = float(metadata.get("control_rate_hz", 0.0) or 0.0)
    pred_horizon = int(metadata.get("pred_horizon", 0) or 0)
    if controller_mode != "edmd_mpc":
        return "unsupported"
    if control_rate_hz >= 90.0 and pred_horizon == 8:
        return "standard_anchor_h8_promotion"
    return "light_anchor_confirmation"


def summarize_milestone_run(run_dir: Path) -> dict[str, object]:
    resolved_run_dir = Path(run_dir).resolve(strict=False)
    log_path = resolved_run_dir / "runtime_log.csv"
    metadata = _load_optional_json(resolved_run_dir / "run_metadata.json")
    drift_summary = _load_optional_json(resolved_run_dir / "drift_summary.json")
    control_summary = _load_optional_json(resolved_run_dir / "control_audit_summary.json")

    profile = milestone_profile_for_metadata(metadata)
    evaluation = evaluate_hover_gates(log_path, profile=profile) if profile != "unsupported" else {
        "passed": False,
        "runtime_validity": None,
        "runtime_failure_reason": "unsupported_profile",
        "controller_quality_evaluated": False,
        "metrics": {},
        "u2_first_used_mismatch_time_s": None,
    }
    metrics = evaluation.get("metrics", {})

    row = {
        "run_name": resolved_run_dir.name,
        "run_dir": str(resolved_run_dir),
        "config_path": metadata.get("config_path"),
        "controller_mode": metadata.get("controller_mode"),
        "control_rate_hz": metadata.get("control_rate_hz"),
        "pred_horizon": metadata.get("pred_horizon"),
        "reference_seed": metadata.get("reference_seed"),
        "runtime_validity": evaluation.get("runtime_validity"),
        "runtime_failure_reason": evaluation.get("runtime_failure_reason"),
        "hover_gate_profile": profile,
        "hover_gate_passed": evaluation.get("passed"),
        "controller_quality_evaluated": evaluation.get("controller_quality_evaluated"),
        "final_altitude_error": metrics.get("final_altitude_error"),
        "max_lateral_error_radius": metrics.get("max_lateral_error_radius"),
        "final_position_error": metrics.get("final_position_error"),
        "position_rmse": metrics.get("position_rmse"),
        "solver_mean_ms": metrics.get("solver_mean_ms"),
        "divergence_time_s": drift_summary.get("divergence_time_s"),
        "u2_first_used_mismatch_time_s": (
            evaluation.get("u2_first_used_mismatch_time_s")
            if evaluation.get("u2_first_used_mismatch_time_s") is not None
            else control_summary.get("u2_first_used_mismatch_time_s")
        ),
    }
    return row


def update_milestone_summary_csv(run_dir: Path) -> Path:
    resolved_run_dir = Path(run_dir).resolve(strict=False)
    summary_path = resolved_run_dir.parent / "milestone_summary.csv"
    row = summarize_milestone_run(resolved_run_dir)

    rows: list[dict[str, object]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            for existing_row in csv.DictReader(stream):
                if existing_row.get("run_name") == row["run_name"]:
                    continue
                rows.append({field: existing_row.get(field) for field in MILESTONE_SUMMARY_FIELDS})
    rows.append(row)
    rows.sort(key=lambda item: str(item["run_name"]))

    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MILESTONE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path
