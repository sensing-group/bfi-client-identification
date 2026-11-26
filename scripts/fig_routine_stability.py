#!/usr/bin/env python3
"""F9: cross-day routine stability per device, column width.

One row per ground-truth device. Dot = mean cosine between the device's
hour-of-day vectors over all its day pairs; whisker = min..max over the
pairs; "n d" = days with >= 50 frames. The vertical reference is the
population median over every (device, day pair). A device seen on fewer
than three days (the paper's rule) is listed but not scored.

Data: figures/bfi_routine_daypairs.csv, columns mac, n_days, day_a, day_b,
cosine; one row per (device, day pair) for devices with >= 2 eligible days
in the 2026-05-08..2026-05-20 archive. Produced from the per-day archive with
the hour-of-day aggregation of scripts/cross_day_routine_stability.py. The
`mac` column holds a raw address (OUI prefix names the device) or the salted
`dev_` id of the released archive (named through the release's
device_labels.csv, BFI_LABELS, or the ids of release/README.md).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
from device_names import TABLE2_NAME

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
CSV = FIG_DIR / "bfi_routine_daypairs.csv"
OUT_STEM = "fig_routine_stability"

MIN_DAYS = 3                                    # the paper's rule for scoring a routine
# Ground truth, Table "devices" in the paper: the archive's label (device_labels.csv
# of the release, xres_common.DEVICE_GT_OUI) -> the name drawn on the figure.
DEVICES = TABLE2_NAME
# Raw archive: OUI prefix -> label.
OUI_LABELS = {
    "8c:26:aa": "Apple TV (living room)",
    "d0:d2:b0": "Apple TV (bedroom)",
    "b8:b4:09": "Samsung TV",
    "54:b7:bd": "LG TV",
    "64:4e:d7": "Laptop",
    "4c:b9:ea": "iRobot vacuum",
}
PREFIX_LEN = 8
# Released archive: salted id -> label (release/README.md, Section 1a); the
# device_labels.csv named by BFI_LABELS takes precedence when it is set.
RELEASED_LABELS = {
    "dev_a674c6": "Apple TV (living room)",
    "dev_f7a353": "Apple TV (bedroom)",
    "dev_668a61": "Samsung TV",
    "dev_ec5bcd": "LG TV",
    "dev_4e4b2c": "Laptop",
    "dev_4c5439": "iRobot vacuum",
}
RELEASED_ID_PREFIX = "dev_"
LABELS_CSV = os.environ.get("BFI_LABELS")

# Values the paper quotes (Section 6 of the paper); the script refuses to draw if the CSV disagrees.
EXPECT = {
    "Apple TV (bedroom)": dict(n_days=6, mean=0.70, max=0.999),
    "Apple TV (living room)": dict(n_days=11, mean=0.66, max=0.995),
    "LG TV": dict(n_days=5, mean=0.28),
    "iRobot vacuum": dict(n_days=3, mean=0.16),
    "Laptop": dict(n_days=4, mean=0.07),
}
EXPECT_N_PAIRS = 89
EXPECT_MEDIAN = 0.61
EXPECT_Q95 = 0.994
TOL = 0.005

COL_DOT = ps.OKABE_ITO["blue"]
COL_WHISKER = ps.OKABE_ITO["sky"]
COL_REF = ps.OKABE_ITO["vermilion"]
COL_UNSCORED = "#5E5E5E"
FIG_H_IN = 1.60
X_MAX = 1.0
X_TICKS = (0, 0.25, 0.5, 0.75, 1.0)
DAYS_X = 1.02                                   # axis fraction for the "n d" column
WHISKER_LW = 2.2
DOT_SIZE = 5
ROW_PAD = 0.6                                   # rows kept clear of the bottom spine


def released_labels() -> dict[str, str]:
    """dev_ id -> label: the labelled rows of BFI_LABELS when set, else the
    ids of release/README.md."""
    if not LABELS_CSV:
        return RELEASED_LABELS
    with open(LABELS_CSV, newline="") as f:
        return {r["dev_id"]: r["label"] for r in csv.DictReader(f) if r.get("label")}


def device_label(addr: str, released: dict[str, str]) -> str:
    """The archive label of a `mac` cell: salted id (release) or OUI prefix (raw)."""
    if addr.startswith(RELEASED_ID_PREFIX):
        return released.get(addr, addr)
    return OUI_LABELS.get(addr[:PREFIX_LEN], addr)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(CSV)
    released = released_labels()
    pairs["prefix"] = pairs["mac"].map(lambda a: device_label(a, released))
    unknown = set(pairs["prefix"]) - set(DEVICES)
    assert not unknown, unknown
    per = pairs.groupby("prefix").agg(n_days=("n_days", "first"), mean=("cosine", "mean"),
                                      lo=("cosine", "min"), hi=("cosine", "max"), n_pairs=("cosine", "size"))
    for p, e in EXPECT.items():
        r = per.loc[p]
        assert r["n_days"] == e["n_days"], (p, r["n_days"])
        assert abs(round(r["mean"], 2) - e["mean"]) < TOL, (p, r["mean"])
        if "max" in e:
            assert abs(round(r["hi"], 3) - e["max"]) < TOL, (p, r["hi"])
    scored = pairs[pairs["n_days"] >= MIN_DAYS]
    assert len(scored) == EXPECT_N_PAIRS, len(scored)
    assert abs(round(scored["cosine"].median(), 2) - EXPECT_MEDIAN) < TOL
    assert abs(round(scored["cosine"].quantile(0.95), 3) - EXPECT_Q95) < TOL
    return per, scored


def main() -> None:
    ps.apply()
    per, scored = load()
    median = float(scored["cosine"].median())
    order = sorted(DEVICES, key=lambda p: per.loc[p, "n_days"] < MIN_DAYS)   # Table 2 order, unscored last

    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.39, right=0.90, bottom=0.394, top=0.978)
    ax.grid(axis="y", visible=False)
    for y, p in enumerate(order):
        r = per.loc[p]
        if r["n_days"] >= MIN_DAYS:
            ax.hlines(y, r["lo"], r["hi"], color=COL_WHISKER, linewidth=WHISKER_LW, zorder=2)
            ax.plot(r["mean"], y, marker="o", color=COL_DOT, markersize=DOT_SIZE, linestyle="none", zorder=3)
            ax.text(DAYS_X, y, f"{int(r['n_days'])} d", transform=ax.get_yaxis_transform(),
                    fontsize=ps.SMALL_PT, va="center", ha="left")
        else:
            ax.text(0.0, y, f"seen on {int(r['n_days'])} days only, not scored", color=COL_UNSCORED,
                    fontsize=ps.SMALL_PT, va="center", ha="left", zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
            ax.text(DAYS_X, y, f"{int(r['n_days'])} d", transform=ax.get_yaxis_transform(),
                    fontsize=ps.SMALL_PT, va="center", ha="left", color=COL_UNSCORED)
    ax.axvline(median, color=COL_REF, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DEVICES[p] for p in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 1 + ROW_PAD, -ROW_PAD)
    ax.set_xlim(0, X_MAX)
    ax.set_xticks(list(X_TICKS))
    ax.set_xlabel("Day-pair cosine of hour-of-day vectors (unitless)")
    ax.set_ylabel("Device")

    handles = [
        plt.Line2D([], [], marker="o", color=COL_DOT, linestyle="none", markersize=DOT_SIZE, label="mean of day pairs"),
        plt.Line2D([], [], color=COL_WHISKER, linewidth=WHISKER_LW, label="min to max"),
        plt.Line2D([], [], color=COL_REF, linestyle="--", label=f"median, all pairs ({median:.2f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    ps.save(fig, OUT_STEM)

    print(per.round(3).to_string())
    print(f"{len(scored)} scored pairs, median {median:.3f}, q95 {scored['cosine'].quantile(0.95):.3f}")
    print(f"wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
