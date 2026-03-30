from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st

# Allow direct execution via `streamlit run dashboard/app.py` or `python dashboard/app.py`
# without requiring the launcher to preconfigure PYTHONPATH.
SITL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SITL_ROOT / "src" / "quantized_quadrotor_sitl"
matplotlib_cache_dir = SITL_ROOT / ".cache" / "matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
for candidate in (SITL_ROOT, PACKAGE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashboard.data import (
    available_sources,
    default_results_root,
    default_run_name,
    discover_run_names,
    prepare_run_data,
)
from dashboard.plots import (
    plot_angular_velocity_errors,
    plot_angular_velocity_states,
    plot_attitude_errors,
    plot_attitude_states,
    plot_control_series,
    plot_position_errors,
    plot_position_states,
    plot_px4_diagnostics,
    plot_timing_series,
    plot_trajectory_3d,
    plot_velocity_errors,
    plot_velocity_states,
    plot_xy_trajectory,
)
from dashboard.style import apply_matlab_style


st.set_page_config(page_title="SITL Paper Dashboard", layout="wide")
apply_matlab_style()


@st.cache_data(show_spinner=False)
def cached_run_names(results_root: str) -> list[str]:
    return discover_run_names(Path(results_root))


@st.cache_data(show_spinner=False)
def cached_run_frame_sources(results_root: str, run_name: str) -> tuple[list[str], list[str]]:
    from dashboard.data import load_run_frame

    frame = load_run_frame(run_name, Path(results_root))
    return available_sources(frame, "state", 18), available_sources(frame, "control", 4)


@st.cache_data(show_spinner=False)
def cached_run_data(results_root: str, run_name: str, state_source: str, control_source: str):
    return prepare_run_data(run_name, state_source, control_source, Path(results_root))


def _default_index(options: list[str], preferred: str | None) -> int:
    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def _render_overview(data) -> None:
    st.subheader("Overview")
    columns = st.columns(4)
    columns[0].metric("Samples", f"{data.summary['sample_count']}")
    columns[1].metric("Final Time (s)", f"{data.summary['final_time_s']:.2f}")
    columns[2].metric("Position RMSE (m)", f"{data.summary['position_rmse']:.3f}")
    columns[3].metric("Max Altitude (m)", f"{data.summary['max_altitude']:.3f}")

    columns = st.columns(3)
    columns[0].metric("Final Position Error (m)", f"{data.summary['final_position_error']:.3f}")
    columns[1].metric("Max Lateral Deviation (m)", f"{data.summary['max_lateral_deviation']:.3f}")
    columns[2].metric("Run", data.run_name)

    if data.metadata:
        columns = st.columns(3)
        columns[0].metric("Controller", str(data.summary.get("controller_mode", "unknown")))
        columns[1].metric("Reference", str(data.summary.get("reference_mode", "unknown")))
        columns[2].metric("Quantization", str(data.summary.get("quantization_mode", "unknown")))
        with st.expander("Run Metadata"):
            st.json(data.metadata)
    else:
        st.caption("run_metadata.json not found for this run. Showing CSV-only analysis.")

    st.caption(f"Loaded from `{data.log_path}`")


def main() -> None:
    st.title("SITL Paper Dashboard")
    st.caption("Minimal MATLAB-style plots for paper-facing SITL analysis.")

    results_root = default_results_root()

    with st.sidebar:
        st.header("Selection")
        if st.button("Refresh Runs", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        run_names = cached_run_names(str(results_root))
        if not run_names:
            st.warning(f"No SITL runs found under {results_root}")
            st.stop()

        preferred_run = default_run_name(run_names)
        run_name = st.selectbox(
            "Results Run",
            options=run_names,
            index=_default_index(run_names, preferred_run),
        )

        state_sources, control_sources = cached_run_frame_sources(str(results_root), run_name)
        if not state_sources or not control_sources:
            st.error("The selected run does not contain the required state/control columns.")
            st.stop()

        state_source = st.selectbox(
            "State Source",
            options=state_sources,
            index=_default_index(state_sources, "raw"),
        )
        control_source = st.selectbox(
            "Control Source",
            options=control_sources,
            index=_default_index(control_sources, "used" if "used" in control_sources else control_sources[0]),
        )

    data = cached_run_data(str(results_root), run_name, state_source, control_source)

    tabs = st.tabs(["Overview", "Trajectory", "Errors", "States", "Controls & Timing"])

    with tabs[0]:
        _render_overview(data)

    with tabs[1]:
        st.pyplot(plot_trajectory_3d(data), clear_figure=True)
        st.pyplot(plot_xy_trajectory(data), clear_figure=True)

    with tabs[2]:
        st.pyplot(plot_position_errors(data), clear_figure=True)
        st.pyplot(plot_velocity_errors(data), clear_figure=True)
        st.pyplot(plot_attitude_errors(data), clear_figure=True)
        st.pyplot(plot_angular_velocity_errors(data), clear_figure=True)

    with tabs[3]:
        st.pyplot(plot_position_states(data), clear_figure=True)
        st.pyplot(plot_velocity_states(data), clear_figure=True)
        st.pyplot(plot_attitude_states(data), clear_figure=True)
        st.pyplot(plot_angular_velocity_states(data), clear_figure=True)

    with tabs[4]:
        st.pyplot(plot_control_series(data), clear_figure=True)
        px4_fig = plot_px4_diagnostics(data)
        if px4_fig is not None:
            st.pyplot(px4_fig, clear_figure=True)
        else:
            st.info("PX4 thrust diagnostic columns are not available for this run.")
        st.pyplot(plot_timing_series(data), clear_figure=True)


if __name__ == "__main__":
    main()
