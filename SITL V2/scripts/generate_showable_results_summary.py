#!/usr/bin/env python3
"""Package the first thesis-ready offline result set into one summary bundle."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from koopman_python.experiments.figures import COLORS, SvgSeries, SvgXYSeries, write_line_plot, write_xy_plot


@dataclass(frozen=True)
class OfflineRunRecord:
    label: str
    controller_variant: str
    scenario: str
    run_dir: Path
    figure_dir: Path
    metrics: dict[str, Any]


def _v2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_showable_results")


def _latest_run_dir(pattern: str) -> Path:
    candidates = sorted((path for path in _v2_root().glob(pattern) if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run directories matched pattern: {pattern}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _series_with_fallback(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    for key in keys:
        if key in rows[0]:
            return _as_float_array(rows, key)
    raise KeyError(f"None of the requested keys were present: {keys}")


def _build_record(label: str, controller_variant: str, scenario: str, run_dir: Path) -> OfflineRunRecord:
    return OfflineRunRecord(
        label=label,
        controller_variant=controller_variant,
        scenario=scenario,
        run_dir=run_dir,
        figure_dir=run_dir / "figures",
        metrics=_load_json(run_dir / "metrics.json"),
    )


def _autodiscover_runs() -> list[OfflineRunRecord]:
    return [
        _build_record(
            label="unquantized_hover_5s",
            controller_variant="unquantized_learned_edmd_mpc",
            scenario="hover_5s",
            run_dir=_latest_run_dir("results/learned/unquantized/hover_5s/*"),
        ),
        _build_record(
            label="unquantized_line_tracking",
            controller_variant="unquantized_learned_edmd_mpc",
            scenario="line_tracking",
            run_dir=_latest_run_dir("results/learned/unquantized/line_tracking/*"),
        ),
        _build_record(
            label="quantized_wl12_n5_hover_5s",
            controller_variant="quantized_learned_edmd_mpc_wl12_n5",
            scenario="hover_5s",
            run_dir=_latest_run_dir("results/learned/quantized/wl_12/N_5/hover_5s/*"),
        ),
        _build_record(
            label="quantized_wl12_n5_line_tracking",
            controller_variant="quantized_learned_edmd_mpc_wl12_n5",
            scenario="line_tracking",
            run_dir=_latest_run_dir("results/learned/quantized/wl_12/N_5/line_tracking/*"),
        ),
    ]


def _summary_row(record: OfflineRunRecord) -> dict[str, Any]:
    metrics = record.metrics
    if record.controller_variant.startswith("unquantized"):
        tracking = metrics["tracking_metrics"]
        solve_time_ms_mean = metrics["solve_time_ms_mean"]
        solve_iterations_mean = metrics["solve_iterations_mean"]
        solve_converged_fraction = metrics["solve_converged_fraction"]
        success = metrics["success"]
    else:
        tracking = metrics["tracking_metrics_mean"]
        solve = metrics["solve_metrics_mean"]
        solve_time_ms_mean = solve["solve_time_ms_mean"]
        solve_iterations_mean = solve["solve_iterations_mean"]
        solve_converged_fraction = solve["solve_converged_fraction"]
        success = metrics["success"]

    return {
        "label": record.label,
        "scenario": record.scenario,
        "controller_variant": record.controller_variant,
        "run_id": metrics["run_id"],
        "success": success,
        "position_rmse_m": tracking["position_rmse_m"],
        "final_position_error_m": tracking["final_position_error_m"],
        "max_position_error_m": tracking["max_position_error_m"],
        "solve_time_ms_mean": solve_time_ms_mean,
        "solve_iterations_mean": solve_iterations_mean,
        "solve_converged_fraction": solve_converged_fraction,
        "run_dir": str(record.run_dir),
        "figure_dir": str(record.figure_dir),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "scenario",
        "controller_variant",
        "success",
        "position_rmse_m",
        "final_position_error_m",
        "max_position_error_m",
        "solve_time_ms_mean",
        "solve_iterations_mean",
        "solve_converged_fraction",
    ]
    lines = [
        "# Showable Results Summary",
        "",
        "Projected-gradient `solve_converged_fraction` is reported for completeness, but it should not be used as a standalone failure criterion.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["controller_variant"]),
                    str(row["success"]),
                    f"{float(row['position_rmse_m']):.6f}",
                    f"{float(row['final_position_error_m']):.6f}",
                    f"{float(row['max_position_error_m']):.6f}",
                    f"{float(row['solve_time_ms_mean']):.6f}",
                    f"{float(row['solve_iterations_mean']):.3f}",
                    f"{float(row['solve_converged_fraction']):.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_quantized_figures(record: OfflineRunRecord) -> None:
    if not record.controller_variant.startswith("quantized"):
        return

    trajectory_rows = _read_csv(record.run_dir / "trajectory.csv")
    control_rows = _read_csv(record.run_dir / "control.csv")
    if not trajectory_rows or not control_rows:
        return

    figure_dir = record.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    time_values = _as_float_array(trajectory_rows, "t_s")
    x_ref = _as_float_array(trajectory_rows, "x_ref")
    y_ref = _as_float_array(trajectory_rows, "y_ref")
    z_ref = _as_float_array(trajectory_rows, "z_ref")
    x_mpc = _series_with_fallback(trajectory_rows, "x_mpc_mean", "x_mpc")
    y_mpc = _series_with_fallback(trajectory_rows, "y_mpc_mean", "y_mpc")
    z_mpc = _series_with_fallback(trajectory_rows, "z_mpc_mean", "z_mpc")
    position_error_norm = (
        _as_float_array(trajectory_rows, "position_error_norm_mean")
        if "position_error_norm_mean" in trajectory_rows[0]
        else np.sqrt((x_mpc - x_ref) ** 2 + (y_mpc - y_ref) ** 2 + (z_mpc - z_ref) ** 2)
    )

    write_line_plot(
        figure_dir / "position_tracking.svg",
        title=f"{record.scenario} Position Tracking",
        x_label="Time (s)",
        y_label="Position (m)",
        x_values=time_values,
        series=[
            SvgSeries("x ref", x_ref, COLORS["ref"]),
            SvgSeries("x mpc mean", x_mpc, COLORS["mpc"]),
            SvgSeries("z ref", z_ref, "#2563eb"),
            SvgSeries("z mpc mean", z_mpc, "#f59e0b"),
        ],
    )
    write_line_plot(
        figure_dir / "position_error_norm.svg",
        title=f"{record.scenario} Position Error Norm",
        x_label="Time (s)",
        y_label="Error (m)",
        x_values=time_values,
        series=[SvgSeries("||p - p_ref||", position_error_norm, COLORS["error"])],
    )
    write_xy_plot(
        figure_dir / "xy_path.svg",
        title=f"{record.scenario} XY Path",
        x_label="x (m)",
        y_label="y (m)",
        series=[
            SvgXYSeries("reference", x_ref, y_ref, COLORS["ref"]),
            SvgXYSeries("mpc mean", x_mpc, y_mpc, COLORS["mpc"]),
        ],
    )

    control_time = _as_float_array(control_rows, "t_s")
    write_line_plot(
        figure_dir / "control_inputs.svg",
        title=f"{record.scenario} Control Inputs",
        x_label="Time (s)",
        y_label="Control",
        x_values=control_time,
        series=[
            SvgSeries("Fb mean", _series_with_fallback(control_rows, "Fb_mean", "Fb"), COLORS["u1"]),
            SvgSeries("Mbx mean", _series_with_fallback(control_rows, "Mbx_mean", "Mbx"), COLORS["u2"]),
            SvgSeries("Mby mean", _series_with_fallback(control_rows, "Mby_mean", "Mby"), COLORS["u3"]),
            SvgSeries("Mbz mean", _series_with_fallback(control_rows, "Mbz_mean", "Mbz"), COLORS["u4"]),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (_v2_root() / "results" / "summary" / "showable_results" / _timestamp())
    output_root.mkdir(parents=True, exist_ok=True)

    records = _autodiscover_runs()
    for record in records:
        _ensure_quantized_figures(record)

    rows = [_summary_row(record) for record in records]
    fieldnames = [
        "label",
        "scenario",
        "controller_variant",
        "run_id",
        "success",
        "position_rmse_m",
        "final_position_error_m",
        "max_position_error_m",
        "solve_time_ms_mean",
        "solve_iterations_mean",
        "solve_converged_fraction",
        "run_dir",
        "figure_dir",
    ]
    _write_csv(output_root / "comparison_table.csv", fieldnames, rows)
    _write_markdown_table(output_root / "comparison_table.md", rows)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "runs": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"showable_results_summary_dir={output_root}")


if __name__ == "__main__":
    main()
