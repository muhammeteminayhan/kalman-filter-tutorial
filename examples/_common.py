"""Shared helpers for the examples: import path + figure directory + style."""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # render to files, no GUI needed
import matplotlib.pyplot as plt  # noqa: E402

# Make the top-level ``kalman`` package importable no matter the CWD.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "legend.framealpha": 0.9,
    }
)


def save(fig, name):
    """Save a figure into ``figures/`` and report the path."""
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved figures/{name}")
    return path
