"""Central matplotlib style for persona-opt visualizations.

Edit ONCE here to restyle every plot. Each plot script imports `apply_style()`
and the palette/helpers below, so per-plot style overrides stay local to that
plot's own file (see plot_*.py).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt

# --- global palette (tweak here) -------------------------------------------
PALETTE = {
    "difficulty": "#d1495b",  # red  = harder = better (the objective)
    "validity": "#2e7d32",  # green = legal/valid
    "timeout_safety": "#ef8a17",
    "semantic": "#1f77b4",
    "feasible": "#7b6cc4",
    "user": "#2e7d32",  # user-sim messages
    "assistant": "#1f77b4",  # agent messages
    "tool": "#9e9e9e",
    "accepted": "#2e7d32",
    "rejected": "#bdbdbd",
    "best": "#d1495b",
    "neutral": "#5b6770",
    "grid": "#dddddd",
}
RUN_COLORS = {"p0001": "#d1495b", "p0002": "#1f77b4", "p0003": "#7b6cc4"}

# diverging colormap for per-task difficulty heatmaps (0=easy .. 1=hard)
HEATMAP_CMAP = "magma"

DPI = 150
FIGSIZE = (9, 5.5)


def apply_style() -> None:
    """Apply the shared rcParams. Call at the top of every plot script."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#999999",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def new_fig(figsize=FIGSIZE):
    apply_style()
    return plt.subplots(figsize=figsize)


def save(fig, out_path) -> str:
    """Save and close a figure; return the path string."""
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)
