"""Dependency-free SVG figures for offline learned-MPC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from koopman_python.edmd.evaluate import decode_full_state_trajectory


@dataclass(frozen=True)
class SvgSeries:
    label: str
    values: np.ndarray
    color: str


@dataclass(frozen=True)
class SvgXYSeries:
    label: str
    x_values: np.ndarray
    y_values: np.ndarray
    color: str


COLORS = {
    "ref": "#0f766e",
    "mpc": "#dc2626",
    "u1": "#1d4ed8",
    "u2": "#7c3aed",
    "u3": "#ea580c",
    "u4": "#059669",
    "error": "#111827",
    "grid": "#d1d5db",
    "axis": "#374151",
    "text": "#111827",
    "bg": "#ffffff",
}


def _with_margin(min_value: float, max_value: float) -> tuple[float, float]:
    if np.isclose(min_value, max_value):
        pad = 1.0 if np.isclose(min_value, 0.0) else 0.1 * abs(min_value)
        return min_value - pad, max_value + pad
    span = max_value - min_value
    pad = 0.08 * span
    return min_value - pad, max_value + pad


def _polyline_points(x_values: np.ndarray, y_values: np.ndarray, plot_box: tuple[float, float, float, float], data_bounds: tuple[float, float, float, float]) -> str:
    left, top, width, height = plot_box
    x_min, x_max, y_min, y_max = data_bounds
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)

    points = []
    for x_val, y_val in zip(x_values, y_values, strict=True):
        px = left + width * ((float(x_val) - x_min) / x_span)
        py = top + height * (1.0 - ((float(y_val) - y_min) / y_span))
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    series: list[SvgSeries],
) -> None:
    width = 960
    height = 540
    margin_left = 90
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_box = (margin_left, margin_top, width - margin_left - margin_right, height - margin_top - margin_bottom)

    x = np.asarray(x_values, dtype=float).reshape(-1)
    y_arrays = [np.asarray(item.values, dtype=float).reshape(-1) for item in series]
    y_min = min(float(np.min(values)) for values in y_arrays)
    y_max = max(float(np.max(values)) for values in y_arrays)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min, y_max = _with_margin(y_min, y_max)
    x_min, x_max = _with_margin(x_min, x_max)
    data_bounds = (x_min, x_max, y_min, y_max)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{COLORS['bg']}' />",
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='22' fill='{COLORS['text']}'>{escape(title)}</text>",
    ]

    for tick in np.linspace(y_min, y_max, 5):
        y_px = plot_box[1] + plot_box[3] * (1.0 - ((tick - y_min) / max(y_max - y_min, np.finfo(float).eps)))
        lines.append(f"<line x1='{plot_box[0]:.2f}' y1='{y_px:.2f}' x2='{plot_box[0] + plot_box[2]:.2f}' y2='{y_px:.2f}' stroke='{COLORS['grid']}' stroke-width='1' />")
        lines.append(f"<text x='{plot_box[0]-10:.2f}' y='{y_px+4:.2f}' text-anchor='end' font-size='12' fill='{COLORS['text']}'>{tick:.3f}</text>")

    for tick in np.linspace(x_min, x_max, 6):
        x_px = plot_box[0] + plot_box[2] * ((tick - x_min) / max(x_max - x_min, np.finfo(float).eps))
        lines.append(f"<line x1='{x_px:.2f}' y1='{plot_box[1]:.2f}' x2='{x_px:.2f}' y2='{plot_box[1] + plot_box[3]:.2f}' stroke='{COLORS['grid']}' stroke-width='1' />")
        lines.append(f"<text x='{x_px:.2f}' y='{plot_box[1] + plot_box[3] + 22:.2f}' text-anchor='middle' font-size='12' fill='{COLORS['text']}'>{tick:.2f}</text>")

    lines.append(f"<rect x='{plot_box[0]:.2f}' y='{plot_box[1]:.2f}' width='{plot_box[2]:.2f}' height='{plot_box[3]:.2f}' fill='none' stroke='{COLORS['axis']}' stroke-width='1.5' />")

    for item, values in zip(series, y_arrays, strict=True):
        points = _polyline_points(x, values, plot_box, data_bounds)
        lines.append(f"<polyline fill='none' stroke='{item.color}' stroke-width='2.4' points='{points}' />")

    lines.append(f"<text x='{width/2:.1f}' y='{height - 18:.2f}' text-anchor='middle' font-size='16' fill='{COLORS['text']}'>{escape(x_label)}</text>")
    lines.append(f"<text x='24' y='{height/2:.1f}' text-anchor='middle' font-size='16' fill='{COLORS['text']}' transform='rotate(-90 24 {height/2:.1f})'>{escape(y_label)}</text>")

    legend_x = plot_box[0] + 18
    legend_y = 18
    for i, item in enumerate(series):
        y_pos = legend_y + 20 * i
        lines.append(f"<line x1='{legend_x:.2f}' y1='{y_pos:.2f}' x2='{legend_x + 18:.2f}' y2='{y_pos:.2f}' stroke='{item.color}' stroke-width='3' />")
        lines.append(f"<text x='{legend_x + 26:.2f}' y='{y_pos + 4:.2f}' font-size='13' fill='{COLORS['text']}'>{escape(item.label)}</text>")

    lines.append("</svg>")
    _write_svg(path, lines)


def write_xy_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[SvgXYSeries],
) -> None:
    width = 700
    height = 700
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_box = (margin_left, margin_top, width - margin_left - margin_right, height - margin_top - margin_bottom)

    all_x = np.concatenate([np.asarray(item.x_values, dtype=float).reshape(-1) for item in series])
    all_y = np.concatenate([np.asarray(item.y_values, dtype=float).reshape(-1) for item in series])
    x_min, x_max = _with_margin(float(np.min(all_x)), float(np.max(all_x)))
    y_min, y_max = _with_margin(float(np.min(all_y)), float(np.max(all_y)))
    data_bounds = (x_min, x_max, y_min, y_max)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{COLORS['bg']}' />",
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='22' fill='{COLORS['text']}'>{escape(title)}</text>",
        f"<rect x='{plot_box[0]:.2f}' y='{plot_box[1]:.2f}' width='{plot_box[2]:.2f}' height='{plot_box[3]:.2f}' fill='none' stroke='{COLORS['axis']}' stroke-width='1.5' />",
    ]

    for item in series:
        points = _polyline_points(
            np.asarray(item.x_values, dtype=float).reshape(-1),
            np.asarray(item.y_values, dtype=float).reshape(-1),
            plot_box,
            data_bounds,
        )
        lines.append(f"<polyline fill='none' stroke='{item.color}' stroke-width='2.4' points='{points}' />")

    lines.append(f"<text x='{width/2:.1f}' y='{height - 18:.2f}' text-anchor='middle' font-size='16' fill='{COLORS['text']}'>{escape(x_label)}</text>")
    lines.append(f"<text x='24' y='{height/2:.1f}' text-anchor='middle' font-size='16' fill='{COLORS['text']}' transform='rotate(-90 24 {height/2:.1f})'>{escape(y_label)}</text>")
    lines.append("</svg>")
    _write_svg(path, lines)


def generate_offline_learned_mpc_figures(
    figure_dir: Path,
    *,
    time_values: np.ndarray,
    actual_states: np.ndarray,
    reference_states: np.ndarray,
    control_matrix: np.ndarray,
) -> None:
    """Write the first reusable offline learned-MPC figure set as SVG files."""

    actual = decode_full_state_trajectory(actual_states)
    reference = decode_full_state_trajectory(reference_states)
    control = np.asarray(control_matrix, dtype=float)
    t = np.asarray(time_values, dtype=float).reshape(-1)

    write_line_plot(
        figure_dir / "position_tracking.svg",
        title="Position Tracking",
        x_label="Time (s)",
        y_label="Position (m)",
        x_values=t,
        series=[
            SvgSeries("x ref", reference.x[0, :], COLORS["ref"]),
            SvgSeries("x mpc", actual.x[0, :], COLORS["mpc"]),
            SvgSeries("z ref", reference.x[2, :], "#2563eb"),
            SvgSeries("z mpc", actual.x[2, :], "#f59e0b"),
        ],
    )

    position_error_norm = np.linalg.norm(actual.x - reference.x, axis=0)
    write_line_plot(
        figure_dir / "position_error_norm.svg",
        title="Position Error Norm",
        x_label="Time (s)",
        y_label="Error (m)",
        x_values=t,
        series=[SvgSeries("||p - p_ref||", position_error_norm, COLORS["error"])],
    )

    write_xy_plot(
        figure_dir / "xy_path.svg",
        title="XY Path",
        x_label="x (m)",
        y_label="y (m)",
        series=[
            SvgXYSeries("reference", reference.x[0, :], reference.x[1, :], COLORS["ref"]),
            SvgXYSeries("mpc", actual.x[0, :], actual.x[1, :], COLORS["mpc"]),
        ],
    )

    write_line_plot(
        figure_dir / "control_inputs.svg",
        title="Control Inputs",
        x_label="Time (s)",
        y_label="Control",
        x_values=t,
        series=[
            SvgSeries("Fb", control[0, :], COLORS["u1"]),
            SvgSeries("Mbx", control[1, :], COLORS["u2"]),
            SvgSeries("Mby", control[2, :], COLORS["u3"]),
            SvgSeries("Mbz", control[3, :], COLORS["u4"]),
        ],
    )
