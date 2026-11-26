#!/usr/bin/env python3
"""F5: why one labeled session suffices, two panels at column width.

(a) Cohort-out K=1 1-NN top-1 accuracy from each single coordinate alone (one
    dot per coordinate, 174 on NTU-Fi, 118 on He) against all coordinates
    jointly (cosine 1-NN, the attacker's metric), with the chance line of
    each decision.
(b) Within-target and between-target Euclidean distance in the standardised
    space the attacker uses, with the DP-SQ (b=4, eps=0.5) displacement of one
    window drawn on the same axis; the ratio displacement / between-target
    median is printed in the panel.

Data: figures/bfi_identity_invariant.json, written by scripts/identity_invariant.py
(X2). Panel (b) carries medians only: the JSON holds summaries of the distance
distributions (median, mean, Cohen's d, AUC, pair counts), not the pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_identity_invariant"
DATA_JSON = FIG_DIR / "bfi_identity_invariant.json"

DATASETS = ("ntufi", "he")                       # left to right
LABELS = {"ntufi": "NTU-Fi proxy\n7 subjects", "he": "He et al.\n19 positions"}
COLOURS = {"ntufi": ps.OKABE_ITO["orange"], "he": ps.OKABE_ITO["blue"]}
COL_CHANCE = ps.OKABE_ITO["grey"]
COL_DPSQ = ps.OKABE_ITO["vermilion"]

# Headline values the paper quotes; the script refuses to draw if the JSON disagrees.
EXPECT_DIM = {"ntufi": 174, "he": 118}
EXPECT_CHANCE = {"ntufi": 1 / 7, "he": 1 / 19}
EXPECT_SINGLE_MAX_PCT = {"ntufi": 47.3, "he": 27.0}
EXPECT_JOINT_COS_PCT = {"ntufi": 75.2, "he": 39.0}
EXPECT_DPSQ_OVER_BETWEEN = {"ntufi": 0.26, "he": 0.72}
TOL_PP = 0.1
TOL_RATIO = 0.005

FIG_H_IN = 2.4
RNG_SEED = 0
STRIP_HALF_W = 0.16          # jitter half-width of the single-coordinate strip (x units)
DOT_SIZE = 4                 # scatter marker area (pt^2) for the single-coordinate dots
DOT_ALPHA = 0.55
MARK_JOINT = "D"
MARK_WITHIN = "o"
MARK_BETWEEN = "s"
MARK_DPSQ = "^"
X_POS = {"ntufi": 0.0, "he": 1.0}
X_LIM = (-0.45, 1.45)
Y_ACC_MAX = 100
Y_ACC_TICKS = (0, 25, 50, 75, 100)
Y_ACC_LABEL = "Top-1 accuracy (%)"
Y_DIST_LABEL = "Median distance (z units)"
CHANCE_HALF_W = 0.3          # half-length of the per-dataset chance segment (x units)
DUMBBELL_LW = 1.0
RATIO_TEXT_DX = 0.14
LEGEND_NCOL = 3


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def load() -> dict:
    raw = json.loads(DATA_JSON.read_text())
    out = {}
    for ds in DATASETS:
        r = raw["results"][ds]
        acc = np.asarray(raw["per_coord_acc"][ds]) * 100
        assert r["dim"] == EXPECT_DIM[ds] == len(acc), (ds, r["dim"], len(acc))
        assert np.isclose(r["chance"], EXPECT_CHANCE[ds]), (ds, r["chance"])
        assert _close(acc.max(), EXPECT_SINGLE_MAX_PCT[ds], TOL_PP), (ds, acc.max())
        assert _close(r["single_coord"]["max"] * 100, acc.max(), TOL_PP)
        joint = r["joint_cos"] * 100
        assert _close(joint, EXPECT_JOINT_COS_PCT[ds], TOL_PP), (ds, joint)
        dist, dp = r["distances"], r["dpsq"]
        ratio = dp["displacement_median"] / dist["between_median"]
        assert np.isclose(ratio, dp["displacement_over_between_median"])
        assert _close(ratio, EXPECT_DPSQ_OVER_BETWEEN[ds], TOL_RATIO), (ds, ratio)
        out[ds] = dict(acc=acc, chance=r["chance"] * 100, joint=joint,
                       within=dist["within_median"], between=dist["between_median"],
                       within_mean=dist["within_mean"], between_mean=dist["between_mean"],
                       d=dist["cohens_d"], auc=dist["auc_between_gt_within"],
                       n_within=dist["n_within"], n_between=dist["n_between"],
                       dpsq=dp["displacement_median"], eps=dp["eps"], ratio=ratio)
    return out


def acc_panel(ax, d: dict) -> None:
    rng = np.random.default_rng(RNG_SEED)
    for ds in DATASETS:
        x, c, r = X_POS[ds], COLOURS[ds], d[ds]
        jitter = rng.uniform(-STRIP_HALF_W, STRIP_HALF_W, size=len(r["acc"]))
        ax.scatter(x + jitter, r["acc"], s=DOT_SIZE, color=c, alpha=DOT_ALPHA, linewidths=0, zorder=3)
        ax.plot([x - CHANCE_HALF_W, x + CHANCE_HALF_W], [r["chance"]] * 2, color=COL_CHANCE,
                linestyle="--", linewidth=0.8, zorder=2)
        ax.plot([x], [r["joint"]], marker=MARK_JOINT, color=c, markeredgecolor="black",
                markeredgewidth=0.6, markersize=5, linestyle="none", zorder=4)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, Y_ACC_MAX)
    ax.set_yticks(list(Y_ACC_TICKS))
    ax.set_ylabel(Y_ACC_LABEL)
    ax.set_title("(a) One coordinate vs all", loc="left")


def dist_panel(ax, d: dict) -> None:
    for ds in DATASETS:
        x, c, r = X_POS[ds], COLOURS[ds], d[ds]
        ax.plot([x, x], [r["within"], r["between"]], color=c, linewidth=DUMBBELL_LW, zorder=2)
        ax.plot([x], [r["within"]], marker=MARK_WITHIN, color=c, markeredgecolor="black",
                markeredgewidth=0.6, markersize=5, linestyle="none", zorder=4)
        ax.plot([x], [r["between"]], marker=MARK_BETWEEN, color=c, markeredgecolor="black",
                markeredgewidth=0.6, markersize=5, linestyle="none", zorder=4)
        ax.plot([x], [r["dpsq"]], marker=MARK_DPSQ, color=COL_DPSQ, markeredgecolor="black",
                markeredgewidth=0.6, markersize=5, linestyle="none", zorder=5)
        # Ratio label on the outer side of each column, so the right-hand one stays inside the axes.
        right = x <= sum(X_LIM) / 2
        ax.text(x + (RATIO_TEXT_DX if right else -RATIO_TEXT_DX), r["dpsq"],
                f"{r['ratio']:.2f}" + r"$\times$", color=COL_DPSQ, fontsize=ps.SMALL_PT,
                ha="left" if right else "right", va="center")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, None)
    ax.set_ylabel(Y_DIST_LABEL)
    ax.set_title("(b) Distances, DP-SQ shift", loc="left")


def main() -> None:
    ps.apply()
    d = load()
    fig, axes = plt.subplots(1, 2, figsize=(ps.COLUMN_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.33, top=0.91, wspace=0.42)
    acc_panel(axes[0], d)
    dist_panel(axes[1], d)
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_xticks([X_POS[ds] for ds in DATASETS])
        ax.set_xticklabels([LABELS[ds] for ds in DATASETS])

    handles = [
        plt.Line2D([], [], color="black", marker="o", markersize=2.5, alpha=DOT_ALPHA, linestyle="none",
                   label="one coordinate"),
        plt.Line2D([], [], color="white", marker=MARK_JOINT, markeredgecolor="black", markersize=5,
                   linestyle="none", label="all coordinates"),
        plt.Line2D([], [], color=COL_CHANCE, linestyle="--", label="chance"),
        plt.Line2D([], [], color="white", marker=MARK_WITHIN, markeredgecolor="black", markersize=5,
                   linestyle="none", label="within-target"),
        plt.Line2D([], [], color="white", marker=MARK_BETWEEN, markeredgecolor="black", markersize=5,
                   linestyle="none", label="between-target"),
        plt.Line2D([], [], color=COL_DPSQ, marker=MARK_DPSQ, markeredgecolor="black", markersize=5, linestyle="none",
                   label=r"DP-SQ shift ($\times$ between)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=LEGEND_NCOL, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.6, columnspacing=1.0, handletextpad=0.5)
    ps.save(fig, OUT_STEM)

    for ds in DATASETS:
        r = d[ds]
        print(f"{ds}: {len(r['acc'])} coords, single min/median/max {r['acc'].min():.1f}/"
              f"{np.median(r['acc']):.1f}/{r['acc'].max():.1f}%, joint (cos) {r['joint']:.1f}%, "
              f"chance {r['chance']:.1f}%")
        print(f"{ds}: within median {r['within']:.2f} (mean {r['within_mean']:.2f}, n={r['n_within']}), "
              f"between median {r['between']:.2f} (mean {r['between_mean']:.2f}, n={r['n_between']}), "
              f"d={r['d']:.2f}, AUC={r['auc']:.3f}, DP-SQ shift {r['dpsq']:.2f} = {r['ratio']:.2f} x between")
    print(f"wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
