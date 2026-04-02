from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.config import load_runtime_config
from .sitl_control_audit import analyze_runtime_control_audit
from .sitl_drift_analysis import analyze_runtime_drift


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


def run_postrun_edmd_analyses(config_path: Path, base_dir: Path) -> dict[str, object]:
    resolved_config_path = _resolve_path(config_path, base_dir)
    config = load_runtime_config(resolved_config_path)

    if config.controller_mode != "edmd_mpc":
        return {
            "skipped": True,
            "reason": "non-edmd controller mode",
        }

    artifact_path = _resolve_path(config.model_artifact, base_dir)
    results_dir = _resolve_path(config.results_dir, base_dir)
    run_dir = results_dir.resolve(strict=False) if results_dir.name == "latest" else results_dir
    log_path = run_dir / "runtime_log.csv"

    if not artifact_path.exists():
        return {
            "skipped": True,
            "reason": f"missing artifact: {artifact_path}",
        }

    if not log_path.exists():
        return {
            "skipped": True,
            "reason": f"missing runtime log: {log_path}",
        }

    drift_summary = analyze_runtime_drift(log_path=log_path, artifact_path=artifact_path, output_dir=run_dir)
    control_summary = analyze_runtime_control_audit(log_path=log_path, artifact_path=artifact_path, output_dir=run_dir)
    return {
        "skipped": False,
        "drift_summary_path": str(run_dir / "drift_summary.json"),
        "drift_trace_path": str(run_dir / "drift_trace.csv"),
        "selected_branch": drift_summary.get("selected_branch"),
        "dominant_error_group": drift_summary.get("dominant_error_group"),
        "control_audit_summary_path": str(run_dir / "control_audit_summary.json"),
        "control_audit_trace_path": str(run_dir / "control_audit_trace.csv"),
        "mapping_status": control_summary.get("mapping_status"),
        "dominant_mismatch_axis": control_summary.get("dominant_mismatch_axis"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate post-run drift and control audit sidecars for an EDMD SITL run.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    args = parser.parse_args()

    summary = run_postrun_edmd_analyses(config_path=args.config_path, base_dir=args.base_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
