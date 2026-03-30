from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .data import DashboardRunData
from .style import ACTUAL_COLOR, ERROR_NORM_COLOR, MATLAB_COLOR_ORDER, REFERENCE_COLOR, apply_matlab_style, style_figure


apply_matlab_style()


def _component_labels(prefix: str) -> list[str]:
    return [f"{prefix}x", f"{prefix}y", f"{prefix}z"]


def _state_figure(
    time_s: np.ndarray,
    actual: np.ndarray,
    reference: np.ndarray,
    labels: list[str],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, axis in enumerate(axes):
        axis.plot(time_s, actual[idx, :], color=ACTUAL_COLOR, linewidth=1.8, label="Actual")
        axis.plot(time_s, reference[idx, :], color=REFERENCE_COLOR, linestyle="--", linewidth=1.8, label="Reference")
        axis.set_ylabel(labels[idx])
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    return style_figure(fig)


def _error_figure(
    time_s: np.ndarray,
    errors: np.ndarray,
    norm: np.ndarray,
    labels: list[str],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for idx in range(3):
        axes[idx].plot(time_s, errors[idx, :], color=MATLAB_COLOR_ORDER[idx], linewidth=1.5)
        axes[idx].set_ylabel(labels[idx])
    axes[3].plot(time_s, norm, color=ERROR_NORM_COLOR, linewidth=1.5)
    axes[3].set_ylabel("norm")
    axes[3].set_xlabel("Time (s)")
    axes[0].set_title(title)
    return style_figure(fig)


def plot_trajectory_3d(data: DashboardRunData) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data.state_history[0, :],
            y=data.state_history[1, :],
            z=data.state_history[2, :],
            mode="lines",
            name="Actual",
            line={"color": f"rgb({ACTUAL_COLOR[0] * 255:.0f}, {ACTUAL_COLOR[1] * 255:.0f}, {ACTUAL_COLOR[2] * 255:.0f})", "width": 6},
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=data.reference_history[0, :],
            y=data.reference_history[1, :],
            z=data.reference_history[2, :],
            mode="lines",
            name="Reference",
            line={
                "color": f"rgb({REFERENCE_COLOR[0] * 255:.0f}, {REFERENCE_COLOR[1] * 255:.0f}, {REFERENCE_COLOR[2] * 255:.0f})",
                "width": 6,
                "dash": "dash",
            },
        )
    )
    fig.update_layout(
        title="3D Trajectory",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.85)"},
        scene={
            "xaxis": {"title": "x (m)", "backgroundcolor": "white", "gridcolor": "#d9d9d9", "showbackground": True},
            "yaxis": {"title": "y (m)", "backgroundcolor": "white", "gridcolor": "#d9d9d9", "showbackground": True},
            "zaxis": {"title": "z (m)", "backgroundcolor": "white", "gridcolor": "#d9d9d9", "showbackground": True},
            "aspectmode": "data",
        },
    )
    return fig


def plot_xy_trajectory(data: DashboardRunData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data.state_history[0, :], data.state_history[1, :], color=ACTUAL_COLOR, linewidth=1.8, label="Actual")
    ax.plot(
        data.reference_history[0, :],
        data.reference_history[1, :],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Reference",
    )
    ax.set_title("XY Trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    return style_figure(fig)


def plot_position_states(data: DashboardRunData) -> plt.Figure:
    return _state_figure(
        data.time_s,
        data.state_history[0:3, :],
        data.reference_history[0:3, :],
        _component_labels(""),
        "Position States",
    )


def plot_velocity_states(data: DashboardRunData) -> plt.Figure:
    return _state_figure(
        data.time_s,
        data.state_history[3:6, :],
        data.reference_history[3:6, :],
        ["vx", "vy", "vz"],
        "Velocity States",
    )


def plot_attitude_states(data: DashboardRunData) -> plt.Figure:
    return _state_figure(
        data.time_s,
        data.parsed_state.theta,
        data.parsed_reference.theta,
        ["theta_x", "theta_y", "theta_z"],
        "Attitude States",
    )


def plot_angular_velocity_states(data: DashboardRunData) -> plt.Figure:
    return _state_figure(
        data.time_s,
        data.parsed_state.wb,
        data.parsed_reference.wb,
        ["wb_x", "wb_y", "wb_z"],
        "Angular Velocity States",
    )


def plot_position_errors(data: DashboardRunData) -> plt.Figure:
    return _error_figure(
        data.time_s,
        data.position_error,
        data.position_error_norm,
        ["ex", "ey", "ez"],
        "Position Errors",
    )


def plot_velocity_errors(data: DashboardRunData) -> plt.Figure:
    return _error_figure(
        data.time_s,
        data.velocity_error,
        data.velocity_error_norm,
        ["evx", "evy", "evz"],
        "Velocity Errors",
    )


def plot_attitude_errors(data: DashboardRunData) -> plt.Figure:
    attitude_norm = np.linalg.norm(data.attitude_error, axis=0)
    return _error_figure(
        data.time_s,
        data.attitude_error,
        attitude_norm,
        ["etheta_x", "etheta_y", "etheta_z"],
        "Attitude Errors",
    )


def plot_angular_velocity_errors(data: DashboardRunData) -> plt.Figure:
    angular_velocity_norm = np.linalg.norm(data.angular_velocity_error, axis=0)
    return _error_figure(
        data.time_s,
        data.angular_velocity_error,
        angular_velocity_norm,
        ["ewb_x", "ewb_y", "ewb_z"],
        "Angular Velocity Errors",
    )


def plot_control_series(data: DashboardRunData) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    labels = ["u0", "u1", "u2", "u3"]
    for idx, axis in enumerate(axes.ravel()):
        axis.plot(data.time_s, data.control_history[idx, :], color=MATLAB_COLOR_ORDER[idx], linewidth=1.5)
        axis.set_ylabel(labels[idx])
    axes[0, 0].set_title(f"Control Inputs ({data.control_source})")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    return style_figure(fig)


def plot_px4_diagnostics(data: DashboardRunData) -> plt.Figure | None:
    if not data.px4_available:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    diagnostics = [
        ("px4_collective_command_newton", "collective_cmd_N"),
        ("px4_collective_normalized", "collective_norm"),
        ("px4_thrust_body_z", "thrust_body_z"),
    ]
    for idx, (column, label) in enumerate(diagnostics):
        axes[idx].plot(data.time_s, data.frame[column].to_numpy(dtype=float), color=MATLAB_COLOR_ORDER[idx], linewidth=1.5)
        axes[idx].set_ylabel(label)
    axes[0].set_title("PX4 Thrust Diagnostics")
    axes[-1].set_xlabel("Time (s)")
    return style_figure(fig)


def plot_timing_series(data: DashboardRunData) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(data.time_s, data.frame["tick_dt_ms"].to_numpy(dtype=float), color=MATLAB_COLOR_ORDER[0], linewidth=1.5)
    axes[0].set_ylabel("tick_dt_ms")
    axes[0].set_title("Timing")
    axes[1].plot(data.time_s, data.frame["solver_ms"].to_numpy(dtype=float), color=MATLAB_COLOR_ORDER[1], linewidth=1.5)
    axes[1].set_ylabel("solver_ms")
    axes[1].set_xlabel("Time (s)")
    return style_figure(fig)
