from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.config import load_runtime_config
from .sitl_control_audit import analyze_runtime_control_audit
from .sitl_drift_analysis import analyze_runtime_drift
from .sitl_milestone_summary import milestone_summary_contains_run, update_milestone_summary_csv
from .sitl_runtime_health import analyze_runtime_health
from .sitl_u2_root_cause import analyze_runtime_u2_root_cause


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


def _load_optional_metadata(metadata_path: Path | None) -> dict[str, object]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _resolve_artifact_path(raw_path: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if base_dir is None:
        raise ValueError(f"relative artifact path requires a base directory: {raw_path}")
    return base_dir / path


def _resolve_run_dir_from_config(config_path: Path, base_dir: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    resolved_config_path = _resolve_path(config_path, base_dir)
    config = load_runtime_config(resolved_config_path)

    if config.controller_mode != "edmd_mpc":
        raise ValueError("non-edmd controller mode")

    results_dir = _resolve_path(config.results_dir, base_dir)
    run_dir = results_dir.resolve(strict=False) if results_dir.name == "latest" else results_dir
    metadata_path = run_dir / "run_metadata.json"
    metadata = _load_optional_metadata(metadata_path)
    metadata_run_dir = metadata.get("run_dir")
    if isinstance(metadata_run_dir, str) and metadata_run_dir:
        run_dir = Path(metadata_run_dir).resolve(strict=False)
        metadata_path = run_dir / "run_metadata.json"
        metadata = _load_optional_metadata(metadata_path)
    artifact_raw = metadata.get("model_artifact", config.model_artifact)
    artifact_path = _resolve_artifact_path(str(artifact_raw), base_dir=base_dir)
    return run_dir, run_dir / "runtime_log.csv", artifact_path, metadata


def _resolve_direct_run_paths(
    *,
    run_dir: Path | None,
    log_path: Path | None,
    artifact_path: Path | None,
    metadata_path: Path | None,
    base_dir: Path | None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    resolved_log_path = None if log_path is None else Path(log_path)
    resolved_run_dir = Path(run_dir) if run_dir is not None else None
    if resolved_run_dir is None:
        if resolved_log_path is None:
            raise ValueError("run_dir or log_path is required for direct post-run analysis")
        resolved_run_dir = resolved_log_path.parent
    else:
        resolved_run_dir = resolved_run_dir.resolve(strict=False)
    if resolved_log_path is None:
        resolved_log_path = resolved_run_dir / "runtime_log.csv"
    resolved_metadata_path = metadata_path if metadata_path is not None else (resolved_run_dir / "run_metadata.json")
    resolved_metadata_path = Path(resolved_metadata_path)
    metadata = _load_optional_metadata(resolved_metadata_path)
    resolved_artifact_path = artifact_path
    if resolved_artifact_path is None:
        artifact_raw = metadata.get("model_artifact")
        if artifact_raw is None:
            raise ValueError("artifact_path is required when run metadata does not provide model_artifact")
        resolved_artifact_path = _resolve_artifact_path(str(artifact_raw), base_dir=base_dir)
    else:
        resolved_artifact_path = Path(resolved_artifact_path)
        if not resolved_artifact_path.is_absolute():
            if base_dir is None:
                raise ValueError(f"relative artifact path requires a base directory: {resolved_artifact_path}")
            resolved_artifact_path = base_dir / resolved_artifact_path
    return resolved_run_dir, resolved_log_path, resolved_artifact_path, metadata


def run_postrun_edmd_analyses(
    *,
    config_path: Path | None = None,
    base_dir: Path | None = None,
    run_dir: Path | None = None,
    log_path: Path | None = None,
    artifact_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    metadata: dict[str, object]
    if config_path is not None:
        if base_dir is None:
            raise ValueError("base_dir is required when config_path is provided")
        try:
            run_dir, log_path, artifact_path, metadata = _resolve_run_dir_from_config(config_path, base_dir)
        except ValueError as exc:
            if str(exc) == "non-edmd controller mode":
                return {
                    "skipped": True,
                    "reason": "non-edmd controller mode",
                }
            raise
    else:
        run_dir, log_path, artifact_path, metadata = _resolve_direct_run_paths(
            run_dir=run_dir,
            log_path=log_path,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            base_dir=base_dir,
        )

    if not artifact_path.exists():
        return {
            "skipped": True,
            "reason": f"missing artifact: {artifact_path}",
            "run_dir": str(run_dir),
            "log_path": str(log_path),
        }

    if not log_path.exists():
        return {
            "skipped": True,
            "reason": f"missing runtime log: {log_path}",
            "run_dir": str(run_dir),
            "artifact_path": str(artifact_path),
        }

    runtime_health_summary = analyze_runtime_health(
        log_path=log_path,
        output_dir=run_dir,
        metadata_path=(run_dir / "run_metadata.json") if metadata_path is None else Path(metadata_path),
    )
    drift_summary = analyze_runtime_drift(log_path=log_path, artifact_path=artifact_path, output_dir=run_dir)
    control_summary = analyze_runtime_control_audit(log_path=log_path, artifact_path=artifact_path, output_dir=run_dir)
    u2_root_cause_summary = analyze_runtime_u2_root_cause(
        log_path=log_path,
        artifact_path=artifact_path,
        metadata_path=(run_dir / "run_metadata.json") if metadata_path is None else Path(metadata_path),
        output_dir=run_dir,
    )
    milestone_summary_path = update_milestone_summary_csv(run_dir)
    milestone_summary_snapshot_path = run_dir / "milestone_summary.csv"
    milestone_summary_contains_current_run = milestone_summary_contains_run(milestone_summary_path, run_dir.name)
    return {
        "skipped": False,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "artifact_path": str(artifact_path),
        "metadata_path": str((run_dir / "run_metadata.json") if metadata_path is None else metadata_path),
        "controller_mode": str(metadata.get("controller_mode", "edmd_mpc")),
        "cost_state_mode": drift_summary.get("cost_state_mode"),
        "runtime_health_summary_path": str(run_dir / "runtime_health_summary.json"),
        "runtime_validity": runtime_health_summary.get("runtime_validity"),
        "runtime_failure_reason": runtime_health_summary.get("runtime_failure_reason"),
        "milestone_summary_path": str(milestone_summary_path),
        "milestone_summary_snapshot_path": str(milestone_summary_snapshot_path),
        "milestone_summary_contains_run": milestone_summary_contains_current_run,
        "drift_summary_path": str(run_dir / "drift_summary.json"),
        "drift_trace_path": str(run_dir / "drift_trace.csv"),
        "selected_branch": drift_summary.get("selected_branch"),
        "dominant_error_group": drift_summary.get("dominant_error_group"),
        "control_audit_summary_path": str(run_dir / "control_audit_summary.json"),
        "control_audit_trace_path": str(run_dir / "control_audit_trace.csv"),
        "mapping_status": control_summary.get("mapping_status"),
        "dominant_mismatch_axis": control_summary.get("dominant_mismatch_axis"),
        "u2_root_cause_summary_path": str(run_dir / "u2_root_cause_summary.json"),
        "u2_root_cause_trace_path": str(run_dir / "u2_root_cause_trace.csv"),
        "u2_root_cause_classification": u2_root_cause_summary.get("u2_root_cause_classification"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate post-run drift and control audit sidecars for an EDMD SITL run.")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    args = parser.parse_args()

    summary = run_postrun_edmd_analyses(
        config_path=args.config_path,
        base_dir=args.base_dir,
        run_dir=args.run_dir,
        log_path=args.log_path,
        artifact_path=args.artifact_path,
        metadata_path=args.metadata_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
