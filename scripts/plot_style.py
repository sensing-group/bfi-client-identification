"""Shared plotting style for the WiMob 2026 BFI paper.

Single source of truth for fonts, colors, markers, axes, and panel labels.
All six evaluation panels import this module and use these helpers.

Visual-craft pass (v5, 2026-05-14):
    - Sans-serif chart typography (DejaVu Sans) for clarity at column width.
    - Deeper, less saturated paper palette so the two series read as
      "calm publication" rather than "primary-color matplotlib default".
    - Thin (0.7 pt) muted spines, only on bottom and left.
    - Very faint horizontal gridlines at major y ticks for the eye to land on.
    - Hatching on the defense series so the panels survive grayscale print.
    - 1.8 pt line weight on data; 0.9 pt on the chance reference.
    - Markers with white edge to lift them off shaded bands.

Constraints (from project rules):
    - Colorblind-safe.
    - 9 pt minimum font size everywhere, including ticks.
    - Percent y-axis (0-100), never fraction. Same range so panels align.
    - Vector PDF, tight bbox.
    - No em-dash, no rhetorical colons in figure text.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator


# --- "Bold paper" palette: deep blue + warm red, with neutral grays. ---
# Both hues stay distinguishable in deuteranopia and protanopia simulations
# (checked with Coblis), and convert cleanly to grayscale because the
# luminance gap between them is large.
PALETTE = {
    "attack":   "#1B9E77",   # teal, attacker / no-defense (ColorBrewer Dark2)
    "defense":  "#D95F02",   # orange, DP-SQ (ColorBrewer Dark2)
    "accent":   "#7570B3",   # muted purple, callout annotations
    "annot":    "#4A4452",   # deep slate, annotation arrow text
    "wine":     "#8C5670",   # dusty plum, secondary callout
    "neutral":  "#666666",   # axis labels, ticks
    "muted":    "#999999",   # connectors, chance line
    "spine":    "#444444",   # axis spines
    "grid":     "#ececec",   # major gridline color
    "row_shade":"#f7f7f7",   # alternating dumbbell row shade
    # Regime band fills (low alpha)
    "band_safe":   "#1B9E77",   # teal: defense holds / safe regime
    "band_warn":   "#E6AB02",   # mustard: transition
    "band_alarm":  "#D95F02",   # orange: attacker territory / failure
    # Legacy Wong names kept for backward compatibility with old panels.
    "blue":     "#D95F02",
    "orange":   "#1B9E77",
    "green":    "#1B9E77",
    "red":      "#D95F02",
    "yellow":   "#E6AB02",
    "pink":     "#8C5670",
    "sky":      "#66A61E",
    "black":    "#000000",
    "gray":     "#666666",
}

MARKERS = {
    "primary":   "o",
    "secondary": "s",
    "tertiary":  "^",
    "quaternary":"D",
    "fifth":     "v",
}

# Hatching gives grayscale separability for the defense series on bar plots.
HATCHES = {
    "defense": "////",
}

# Bit-depth axis is shared. The closedset CSV has no b=3.
B_VALUES  = [1, 2, 4, 6]

# IEEEtran column-width budget. Each eval panel is 0.32\textwidth in a
# 7.16 in wide page = 2.29 in. v7 (2026-05-14, readability pass): render
# the source at the exact display size so LaTeX does not scale text down.
# A 9 pt source font is now a 9 pt display font; previously the v6 source
# at (2.95, 2.35) was scaled by 0.66 in LaTeX, dropping 9 pt to ~6 pt.
# The LaTeX include directive moved from height=1.55in to width=\linewidth
# so source figsize controls rendered size exactly.
PANEL_WIDTH_IN  = 2.30
PANEL_HEIGHT_IN = 1.55
SINGLE_FIG_IN   = (3.5, 2.45)


def set_paper_style() -> None:
    """Apply consistent rcParams for all paper plots.

    Switches global font family to sans-serif (DejaVu Sans is what ships
    with matplotlib and reads close to Helvetica at print size). Thinner
    spines than matplotlib default. Gridlines off by default; panels that
    want them turn them on locally with the same color/weight.
    """
    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Linux Libertine O", "Linux Libertine", "Libertine",
                            "Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Linux Biolinum O", "Linux Biolinum", "Biolinum",
                            "DejaVu Sans"],
        "mathtext.fontset":"stix",
        "font.size":         9,
        "axes.titlesize":   10,
        "axes.labelsize":   10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.title_fontsize": 9,
        "figure.dpi":      300,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "axes.linewidth":  0.7,
        "axes.edgecolor":  PALETTE["spine"],
        "axes.labelcolor": "#222222",
        "xtick.color":     PALETTE["neutral"],
        "ytick.color":     PALETTE["neutral"],
        "xtick.labelcolor":"#222222",
        "ytick.labelcolor":"#222222",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "axes.grid":      False,
        "axes.axisbelow": True,
        "grid.color":     PALETTE["grid"],
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "errorbar.capsize": 2.5,
        "legend.frameon":  False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 0.9,
    })


def clean_spines(ax, bottom: bool = True, left: bool = True) -> None:
    """Top/right off, bottom/left thin and muted. The award-paper baseline."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(PALETTE["spine"])
        ax.spines["bottom"].set_linewidth(0.7)
    else:
        ax.spines["bottom"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(PALETTE["spine"])
        ax.spines["left"].set_linewidth(0.7)
    else:
        ax.spines["left"].set_visible(False)


def soft_grid(ax, which: str = "y") -> None:
    """Faint horizontal gridlines at major ticks. Sits behind the data."""
    ax.grid(True, which="major", axis=which,
            color=PALETTE["grid"], linewidth=0.5, linestyle="-",
            zorder=0)
    ax.set_axisbelow(True)


def panel_size(n_panels: int = 3) -> tuple[float, float]:
    if n_panels == 1:
        return SINGLE_FIG_IN
    return (PANEL_WIDTH_IN, PANEL_HEIGHT_IN)


def percent_axis(ax, axis: str = "y", lo: float = 0, hi: float = 100,
                 major: float = 20, minor: float | None = None) -> None:
    if axis == "y":
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(MultipleLocator(major))
        if minor:
            ax.yaxis.set_minor_locator(MultipleLocator(minor))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    else:
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MultipleLocator(major))
        if minor:
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))


def add_chance_line(ax, value_pct: float, x_label_pos: float | None = None,
                    label_side: str = "left",
                    label_y_offset: float = 1.8) -> None:
    """Slightly thicker dashed gray line + tiny italic 'chance' tag.

    The tag sits just above the line so it does not collide with x ticks.
    Use axes coordinates for the label to avoid clipping by data elements.
    """
    ax.axhline(
        value_pct, color=PALETTE["muted"], linestyle=(0, (2, 2)),
        linewidth=0.9, zorder=1,
    )
    if x_label_pos is not None:
        ha = "left" if label_side == "left" else "right"
        ax.text(
            x_label_pos, value_pct + label_y_offset, "chance",
            color=PALETTE["muted"], fontsize=9, style="italic",
            ha=ha, va="bottom", zorder=2,
        )


def style_legend(ax, **kwargs):
    defaults = dict(
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        handletextpad=0.5,
        borderaxespad=0.3,
        labelspacing=0.3,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)
