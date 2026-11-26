#!/usr/bin/env python3
"""F7: what the listener gains from listening longer and from enrolling more.

Two panels at column width sharing the y axis (accuracy across days,
other_day_any protocol, all six devices in the gallery, chance 1/6):

(a) Minutes of listening: accuracy against the median listening time of a
    query of T consecutive windows, T in {1, 2, 4, 8, 16, 32}, one line per
    enrolment size K in {1, 16, all}. Each device keeps the weight it has at
    T = 1: the Samsung TV's short sessions (recall 0) hold 100 of 1,600
    decisions at T = 1 and one at T = 8, so pooling over the decisions of
    each T would show a rise no device makes.
(b) Windows enrolled: accuracy against K in {1, 2, 4, 8, 16, all} for a
    single-window query (T = 1), with the block-clustered bootstrap CI.

Data (figures/):
  m_time_to_identify.csv / _summary.json   M2 run (scripts/m_time_to_identify.py):
      per-device recall per (K, T); rows device = ALL give minutes_median, the
      median span of the T query windows on the device's own timestamps.
  r_main.csv / r_eval_summary.json         R1 run (scripts/r_eval.py):
      condition other_day_any, feature angles, gap_s = the paper's session
      gap, T = 1: accuracy is the pooled recall, ci_lo/ci_hi its CI.

The two runs draw different random episodes (M2 seed 200, R1 seed 0), so the
K = 1, T = 1 point differs between the panels by under half a point
(82.25 against 82.63); both are drawn as the files hold them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_time_enroll"

M_CSV = FIG_DIR / "m_time_to_identify.csv"
M_JSON = FIG_DIR / "m_time_to_identify_summary.json"
R_CSV = FIG_DIR / "r_main.csv"
R_JSON = FIG_DIR / "r_eval_summary.json"

POOLED_LABEL = "ALL"
CONDITION = "other_day_any"
FEATURE = "angles"
SESSION_GAP_S = 300                    # the paper's session gap (r_common.SESSION_GAP_S)
T_GRID = (1, 2, 4, 8, 16, 32)
K_GRID_TIME = ("1", "16", "all")
K_GRID_ENROL = ("1", "2", "4", "8", "16", "all")
T_ENROL = 1
N_DEVICES = 6
CHANCE = 100.0 / N_DEVICES
N_QUERIES_T1 = 1600
CI_PCT = 95.0                          # r_eval.CI_LEVEL, block-clustered bootstrap
TOL_PP = 0.01                          # percent points
TOL_MIN = 0.001                        # minutes

# Values the figure prints (percent, minutes); the script refuses to draw if the files move.
EXPECT_MINUTES = (0.753, 1.548, 3.124, 6.272, 12.614, 24.985)
EXPECT_RECALL_TIME = {            # at the device weights of T = 1
    "1": (np.float64(82.25), np.float64(82.92), np.float64(82.79), np.float64(82.87), np.float64(83.46), np.float64(82.27)),
    "16": (np.float64(87.5), np.float64(88.29), np.float64(88.83), np.float64(88.41), np.float64(89.06), np.float64(87.46)),
    "all": (np.float64(88.0), np.float64(88.65), np.float64(89.01), np.float64(88.3), np.float64(89.2), np.float64(87.48)),
}
EXPECT_RECALL_ENROL = (82.63, 84.50, 85.75, 86.94, 87.31, 87.88)

K_STYLE = {   # colour, marker, line style: three series must survive greyscale
    "1": dict(color=ps.OKABE_ITO["blue"], marker="o", linestyle="-", zorder=3),
    "16": dict(color=ps.OKABE_ITO["vermilion"], marker="s", linestyle="--", zorder=4),
    "all": dict(color=ps.OKABE_ITO["black"], marker="^", linestyle=":", markerfacecolor="white", zorder=3),
}
K_LABEL = {"1": "$K$ = 1", "16": "$K$ = 16", "all": "$K$ = all"}
COL_ENROL = ps.OKABE_ITO["black"]       # panel (b) is its own series: not the blue circle of K = 1
MARK_ENROL = "D"
COL_CHANCE = ps.OKABE_ITO["grey"]
TEXT_CHANCE = "#5E5E5E"          # the chance label, 6:1 on white
FIG_H_IN = 1.95
Y_MAX = 100
Y_TICKS = (0, 25, 50, 75, 100)
Y_LABEL = "Accuracy across days (%)"
X_TICKS_MIN = (1, 2, 4, 8, 16)
X_LIM_MIN = (0.55, 34)
X_LIM_K = (-0.5, len(K_GRID_ENROL) - 0.5)
CHANCE_TEXT_DY = 1.5
MARKER_SIZE = 3.0
LEGEND_Y = 0.0
LEGEND_NCOL = 3                        # six entries in two rows fit the column width
LAYOUT = dict(left=0.135, right=0.985, bottom=0.40, top=0.875, wspace=0.12)


def _close(a: float, b: float, tol: float = TOL_PP) -> bool:
    return abs(a - b) <= tol


def load_time() -> pd.DataFrame:
    """The M2 run as a (K x T) table of accuracy in percent, every device at
    its weight of T = 1, one row per (K, T) with the pooled row's minutes."""
    meta = json.loads(M_JSON.read_text())
    assert meta["condition"] == CONDITION, meta["condition"]
    assert tuple(meta["k_grid"]) == K_GRID_TIME, meta["k_grid"]
    assert tuple(meta["t_grid"]) == T_GRID, meta["t_grid"]
    df = pd.read_csv(M_CSV, dtype={"K": str})
    pooled = df[df["device"] == POOLED_LABEL].copy()
    assert set(pooled["K"]) == set(K_GRID_TIME), set(pooled["K"])
    assert set(pooled["T"]) == set(T_GRID), set(pooled["T"])
    assert len(pooled) == len(K_GRID_TIME) * len(T_GRID), len(pooled)
    assert np.allclose(pooled["chance"], 1.0 / N_DEVICES)
    assert (pooled.loc[pooled["T"] == T_ENROL, "n_queries"] == N_QUERIES_T1).all()
    per_dev = df[df["device"] != POOLED_LABEL]
    weight = per_dev[per_dev["T"] == T_ENROL].set_index(["K", "device"])["n_queries"]
    weight = weight / weight.groupby("K").transform("sum")
    per_dev = per_dev.assign(w=[weight[(k, d)] for k, d in zip(per_dev["K"], per_dev["device"])])
    fixed = (per_dev["w"] * per_dev["recall"]).groupby([per_dev["K"], per_dev["T"]]).sum()
    pooled["recall_pct"] = [100 * fixed[(k, t)] for k, t in zip(pooled["K"], pooled["T"])]
    for k in K_GRID_TIME:
        d = pooled[pooled["K"] == k].set_index("T").loc[list(T_GRID)]
        for t, m, r, em, er in zip(T_GRID, d["minutes_median"], d["recall_pct"],
                                   EXPECT_MINUTES, EXPECT_RECALL_TIME[k]):
            assert _close(m, em, TOL_MIN), (k, t, m)
            assert _close(r, er), (k, t, r)
    minutes = pooled.groupby("T")["minutes_median"].nunique()
    assert (minutes == 1).all(), "listening time must not depend on K"
    return pooled


def load_enrol() -> pd.DataFrame:
    """R1 rows: other_day_any, angles, session gap, T = 1, all six in the gallery."""
    meta = json.loads(R_JSON.read_text())
    assert tuple(meta["k_grid"]) == K_GRID_ENROL, meta["k_grid"]
    assert T_ENROL in meta["t_grid"], meta["t_grid"]
    df = pd.read_csv(R_CSV, dtype={"K": str})
    sel = df[(df["condition"] == CONDITION) & (df["feature"] == FEATURE)
             & (df["gap_s"] == SESSION_GAP_S) & (df["T"] == T_ENROL)].copy()
    assert list(sel["K"]) == list(K_GRID_ENROL), list(sel["K"])
    assert (sel["n_decisions"] == N_QUERIES_T1).all()
    assert (sel["n_devices_queried"] == N_DEVICES).all()
    for h in sel["gallery_size_hist"]:
        assert json.loads(h) == {str(N_DEVICES): N_QUERIES_T1}, h
    assert np.allclose(sel["chance"], 1.0 / N_DEVICES)
    for c in ("accuracy", "ci_lo", "ci_hi"):
        sel[c + "_pct"] = sel[c] * 100
    for k, r, er in zip(K_GRID_ENROL, sel["accuracy_pct"], EXPECT_RECALL_ENROL):
        assert _close(r, er), (k, r)
    assert ((sel["ci_lo_pct"] <= sel["accuracy_pct"]) & (sel["accuracy_pct"] <= sel["ci_hi_pct"])).all()
    return sel


def chance_line(ax, label: str | None) -> None:
    ax.axhline(CHANCE, color=COL_CHANCE, linestyle="--", linewidth=0.8, zorder=1)
    if label:
        ax.text(ax.get_xlim()[1], CHANCE + CHANCE_TEXT_DY, label, color=TEXT_CHANCE,
                fontsize=ps.SMALL_PT, ha="right", va="bottom")


def panel_time(ax, pooled: pd.DataFrame) -> list:
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM_MIN)
    handles = []
    for k in K_GRID_TIME:
        d = pooled[pooled["K"] == k].sort_values("T")
        (h,) = ax.plot(d["minutes_median"], d["recall_pct"], markersize=MARKER_SIZE,
                       label=K_LABEL[k], **K_STYLE[k])
        handles.append(h)
    chance_line(ax, None)
    ax.set_xticks(list(X_TICKS_MIN))
    ax.set_xticklabels([str(t) for t in X_TICKS_MIN])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Listening time (min, median)")
    ax.set_title("(a) Minutes of listening", loc="left")
    return handles


def panel_enrol(ax, sel: pd.DataFrame) -> list:
    x = np.arange(len(K_GRID_ENROL))
    yerr = np.vstack([sel["accuracy_pct"] - sel["ci_lo_pct"], sel["ci_hi_pct"] - sel["accuracy_pct"]])
    h = ax.errorbar(x, sel["accuracy_pct"], yerr=yerr, color=COL_ENROL, marker=MARK_ENROL,
                    markersize=MARKER_SIZE, linestyle="-", capsize=1.5, linewidth=1.0,
                    label=f"$T$ = 1, {CI_PCT:.0f}% CI", zorder=3)
    chance_line(ax, f"chance 1/{N_DEVICES}")
    ax.set_xlim(*X_LIM_K)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(K_GRID_ENROL))
    ax.set_xlabel("$K$ enrolled (windows)")
    ax.set_title("(b) Windows enrolled", loc="left")
    return [h]


def main() -> None:
    ps.apply()
    pooled = load_time()
    sel = load_enrol()
    fig, axes = plt.subplots(1, 2, figsize=(ps.COLUMN_W_IN, FIG_H_IN), sharey=True)
    fig.subplots_adjust(**LAYOUT)
    for ax in axes:
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks(list(Y_TICKS))
        ax.grid(axis="x", visible=False)
    h_time = panel_time(axes[0], pooled)
    h_enrol = panel_enrol(axes[1], sel)
    for i, ax in enumerate(axes):
        ax.set_ylabel(Y_LABEL)
        ax.yaxis.label.set_visible(i == 0)
    h_chance = plt.Line2D([], [], color=COL_CHANCE, linestyle="--", linewidth=0.8, label="chance")
    # The legend fills column by column: this order puts the three K series on the
    # first row and the panel (b) entries on the second.
    handles = [h_time[0], h_enrol[0], h_time[1], h_chance, h_time[2]]
    fig.legend(handles=handles, loc="lower center", ncol=LEGEND_NCOL,
               bbox_to_anchor=(0.56, LEGEND_Y), handlelength=1.9, columnspacing=1.2,
               handletextpad=0.5, labelspacing=0.3)
    ps.save(fig, OUT_STEM)

    for k in K_GRID_TIME:
        d = pooled[pooled["K"] == k].sort_values("T")
        print(f"(a) K={k}: " + ", ".join(f"T={t} {m:.2f} min {r:.2f}%" for t, m, r in
                                        zip(d["T"], d["minutes_median"], d["recall_pct"])))
    print("(b) T=1: " + ", ".join(f"K={k} {a:.2f}% [{lo:.2f}, {hi:.2f}]" for k, a, lo, hi in
                                  zip(sel["K"], sel["accuracy_pct"], sel["ci_lo_pct"], sel["ci_hi_pct"])))
    print(f"chance {CHANCE:.1f}%; wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
