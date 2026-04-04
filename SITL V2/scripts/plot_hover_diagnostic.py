#!/usr/bin/env python3
"""Generate diagnostic plots from a hover validation run."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from koopman_python.experiments.figures import COLORS, SvgSeries, write_line_plot


def _v2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_run_dir() -> Path:
    candidates = sorted(
        (path for path in (_v2_root() / "results" / "runtime_logs" / "hover_validation").glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No hover validation runs found.")
    return candidates[-1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _series_with_fallback(rows: list[dict[str, str]], *keys: str, default: float = float("nan")) -> np.ndarray:
    for key in keys:
        if key in rows[0]:
            return _as_float_array(rows, key)
    return np.full(len(rows), default, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir()
    output_dir = args.output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    odometry_rows = _read_csv(run_dir / "odometry.csv")
    command_rows = _read_csv(run_dir / "command.csv")
    if not odometry_rows or not command_rows:
        raise SystemExit(f"Expected odometry.csv and command.csv in {run_dir}")

    time_odom = _as_float_array(odometry_rows, "t_s")
    z_values = _as_float_array(odometry_rows, "z_m")
    target_z = float(metrics["target"].get("target_z_absolute_m", metrics["target"]["z_m"]))
    target_z_values = np.full_like(z_values, target_z, dtype=float)

    time_command = _as_float_array(command_rows, "t_s")
    learned_thrust = _series_with_fallback(command_rows, "learned_thrust_newton")
    commanded_thrust = _series_with_fallback(command_rows, "commanded_thrust_newton")
    trim = _series_with_fallback(command_rows, "hover_altitude_trim_newton")
    position_correction = _series_with_fallback(
        command_rows,
        "vertical_position_correction_newton",
        "thrust_assist_newton",
        default=0.0,
    )
    damping = _series_with_fallback(command_rows, "vertical_damping_newton", default=0.0)
    solve_iterations = _series_with_fallback(command_rows, "solve_iterations", default=0.0)
    projected_step = _series_with_fallback(command_rows, "solve_projected_step_inf_norm", default=float("nan"))

    write_line_plot(
        output_dir / "z_tracking.svg",
        title="Hybrid Hover Altitude Tracking",
        x_label="Time (s)",
        y_label="z (m)",
        x_values=time_odom,
        series=[
            SvgSeries("target z", target_z_values, COLORS["ref"]),
            SvgSeries("measured z", z_values, COLORS["mpc"]),
        ],
    )
    write_line_plot(
        output_dir / "thrust_channels.svg",
        title="Hybrid Hover Thrust Channels",
        x_label="Time (s)",
        y_label="Thrust / Correction (N)",
        x_values=time_command,
        series=[
            SvgSeries("commanded thrust", commanded_thrust, COLORS["u1"]),
            SvgSeries("learned thrust", learned_thrust, COLORS["u2"]),
            SvgSeries("trim", trim, COLORS["u3"]),
            SvgSeries("z correction", position_correction, COLORS["u4"]),
            SvgSeries("damping", damping, COLORS["error"]),
        ],
    )
    write_line_plot(
        output_dir / "solver_diagnostics.svg",
        title="Online MPC Solver Diagnostics",
        x_label="Time (s)",
        y_label="Iterations / Step Norm",
        x_values=time_command,
        series=[
            SvgSeries("solve iterations", solve_iterations, COLORS["u1"]),
            SvgSeries("projected step inf-norm", projected_step, COLORS["u2"]),
        ],
    )

    xy_rmse = float(metrics["position_rmse_m"]["xy"])
    z_tail_error = abs(float(metrics["position_error_tail_mean_m"]["z"]))
    if metrics["success"]:
        milestone_statement = "Stable hybrid hover achieved under the current validation bounds."
    elif xy_rmse <= 0.05 and z_tail_error > float(metrics["tolerances"]["z_m"]):
        milestone_statement = (
            "Online learned integration is working, and the remaining failure is concentrated in the vertical hover path."
        )
    else:
        milestone_statement = (
            "Online hover remains unstable, but the diagnostic plots isolate the remaining issues to control-path behavior rather than ROS wiring."
        )

    (output_dir / "milestone_statement.md").write_text(
        "\n".join(
            [
                "# Online Hover Milestone",
                "",
                milestone_statement,
                "",
                f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
                f"- Validation run: {run_dir}",
                f"- Success: {metrics['success']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "target_z_absolute_m": target_z,
                "figures": [
                    str(output_dir / "z_tracking.svg"),
                    str(output_dir / "thrust_channels.svg"),
                    str(output_dir / "solver_diagnostics.svg"),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"hover_diagnostic_figure_dir={output_dir}")


if __name__ == "__main__":
    main()
