"""Shared publication-quality matplotlib style for the ICLR figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def apply_style():
    plt.rcParams.update({
        # Times-like serif text to match the ICLR body font
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        # clean axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # subtle grid
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        # export
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
    })


def seq_colors(n, cmap="viridis", lo=0.08, hi=0.85):
    """n perceptually ordered colors from a colormap."""
    return [cm.get_cmap(cmap)(x) for x in np.linspace(lo, hi, n)]


# Okabe--Ito colorblind-safe palette for categorical series
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def ema(x, alpha=0.12):
    """Exponential moving average for smoothing noisy traces."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i - 1] + alpha * x[i]
    return y


def raw_plus_smooth(ax, t, x, color, label, alpha_raw=0.28,
                    lw_raw=0.7, lw_smooth=1.8, smooth_alpha=0.12):
    """Plot a noisy trace as a faint raw line plus a bold EMA overlay."""
    ax.plot(t, x, color=color, lw=lw_raw, alpha=alpha_raw, zorder=2)
    ax.plot(t, ema(x, smooth_alpha), color=color, lw=lw_smooth,
            label=label, zorder=3)
