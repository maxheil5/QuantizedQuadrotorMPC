from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from cycler import cycler


MATLAB_COLOR_ORDER: list[tuple[float, float, float]] = [
    (0.0, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]
ACTUAL_COLOR = MATLAB_COLOR_ORDER[0]
REFERENCE_COLOR = MATLAB_COLOR_ORDER[1]
ERROR_NORM_COLOR = (0.0, 0.0, 0.0)


def apply_matlab_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.prop_cycle": cycler(color=MATLAB_COLOR_ORDER),
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": False,
        }
    )


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)


def style_figure(fig: plt.Figure) -> plt.Figure:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            style_axes(ax)
    return fig
