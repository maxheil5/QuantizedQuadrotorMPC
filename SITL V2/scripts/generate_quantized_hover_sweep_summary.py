#!/usr/bin/env python3
"""Summarize a quantized hover-validation sweep across word lengths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "word_length",
        "realizations",
        "realization_index",
        "success",
        "final_z_m",
        "target_z_absolute_m",
        "tail_z_error_m",
        "xy_rmse_m",
        "z_rmse_m",
        "time_to_first_z_band_s",
        "solve_time_ms_mean",
        "solve_iterations_mean",
        "solve_converged_fraction",
        "solve_hit_iteration_cap_fraction",
    ]
    lines = [
        "# Quantized Hover Sweep Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["word_length"]),
                    str(row["realizations"]),
                    str(row["realization_index"]),
                    str(row["success"]),
                    f"{float(row['final_z_m']):.6f}",
                    f"{float(row['target_z_absolute_m']):.6f}",
                    f"{float(row['tail_z_error_m']):.6f}",
                    f"{float(row['xy_rmse_m']):.6f}",
                    f"{float(row['z_rmse_m']):.6f}",
                    "" if row["time_to_first_z_band_s"] == "" else f"{float(row['time_to_first_z_band_s']):.6f}",
                    f"{float(row['solve_time_ms_mean']):.6f}",
                    f"{float(row['solve_iterations_mean']):.3f}",
                    f"{float(row['solve_converged_fraction']):.3f}",
                    f"{float(row['solve_hit_iteration_cap_fraction']):.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir
    manifest_rows = _read_csv(sweep_dir / "manifest.csv")
    summary_rows: list[dict[str, Any]] = []

    for manifest_row in manifest_rows:
        run_dir = Path(manifest_row["validation_run_dir"])
        metrics = _load_json(run_dir / "metrics.json")
        command_metrics = metrics["command_metrics"]
        summary_rows.append(
            {
                "word_length": int(manifest_row["word_length"]),
                "realizations": int(manifest_row["realizations"]),
                "realization_index": int(manifest_row["realization_index"]),
                "success": bool(metrics["success"]),
                "final_z_m": float(metrics["final_position_m"]["z"]),
                "target_z_absolute_m": float(metrics["target_z_absolute_m"]),
                "tail_z_error_m": float(metrics["position_error_tail_mean_m"]["z"]),
                "xy_rmse_m": float(metrics["position_rmse_m"]["xy"]),
                "z_rmse_m": float(metrics["position_rmse_m"]["z"]),
                "time_to_first_z_band_s": metrics["time_to_first_z_band_s"] if metrics["time_to_first_z_band_s"] is not None else "",
                "solve_time_ms_mean": float(command_metrics["solve_time_ms"]["mean"]),
                "solve_iterations_mean": float(command_metrics["solve_iterations"]["mean"]),
                "solve_converged_fraction": float(command_metrics["solve_converged_fraction"]),
                "solve_hit_iteration_cap_fraction": float(command_metrics["solve_hit_iteration_cap_fraction"]),
                "validation_run_dir": str(run_dir),
                "figure_dir": str(run_dir / "figures"),
            }
        )

    summary_rows.sort(key=lambda row: row["word_length"])
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    _write_csv(sweep_dir / "summary.csv", fieldnames, summary_rows)
    _write_markdown(sweep_dir / "summary.md", summary_rows)
    (sweep_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"quantized_hover_sweep_summary={sweep_dir}")


if __name__ == "__main__":
    main()
