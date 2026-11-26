#!/usr/bin/env python3
"""Figure 9: public-dataset results, three panels at text width.

(a) He et al. real feedback, 19 held-out room positions, one window enrolled
    per position: the nearest-neighbor rule (1-NN) on raw quantized features,
    1-NN on features scaled on the other 20 positions, and the SupCon encoder.
(b) NTU-Fi amplitude-CSI proxy, 7 held-out subjects, the same three attackers
    (features scaled on the other 7 subjects).
(c) DP-SQ at the budgets the joint-defense runs hold, 1-NN on scaled features,
    both datasets, with the unprotected level of each as a dotted reference.

Data (all in figures/, ten seeds per cell, b = 4):
  s_calibration_size.csv             dataset, n_cal, attacker, K, acc_mean, acc_sd:
                                     n_cal = 0 is raw features, n_cal = all other
                                     targets is the scaled condition
  bfi_joint_defense_he_summary.json  cells "<defense>|<attacker>|T=<T>": mean, sd, n
  bfi_joint_defense_ntufi_summary.json
Error bars are one SD over the ten seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import paper_style as ps

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_public_data"

CAL_CSV = FIG_DIR / "s_calibration_size.csv"
JOINT_JSON = {"he": FIG_DIR / "bfi_joint_defense_he_summary.json",
              "ntufi": FIG_DIR / "bfi_joint_defense_ntufi_summary.json"}

N_CLASSES = {"he": 19, "ntufi": 7}
N_SEEDS = 10
SCALED_N_CAL = {"he": 20, "ntufi": 7}      # every other position / subject
ONE_WINDOW = "1"                           # K: one window enrolled per target
QUERY_T = 1                                # one query window per decision
NN, ENC = "nn1", "supcon"
NONE = "none"
DPSQ_PREFIX = "dpsq_e"                     # dpsqstd (He only) is a different defense

# Values the paper quotes (percent); the script refuses to draw if the files disagree.
EXPECT = {"he": dict(raw=48.9, scaled=39.0, enc=33.6, dpsq=(26.0, 30.7)),
          "ntufi": dict(raw=82.1, scaled=75.2, enc=65.6, dpsq=(74.3, 75.2))}
EXPECT_EPS = (0.01, 0.5, 2.0)
TOL_PP = 0.05

COL = {"he": ps.OKABE_ITO["blue"], "ntufi": ps.OKABE_ITO["orange"]}
COL_CHANCE = ps.OKABE_ITO["grey"]
TEXT_CHANCE = "#5E5E5E"          # the chance labels, 6:1 on white
MARK = {"raw": "o", "scaled": "D", "enc": "s"}
FILLED = {"raw": True, "scaled": True, "enc": False}
ATTACKERS = ("raw", "scaled", "enc")
ATTACKER_TICK = {"raw": "1-NN\nraw", "scaled": "1-NN\nscaled", "enc": "encoder\nscaled"}
FIG_H_IN = 2.15
Y_MAX = 100
Y_TICKS = (0, 25, 50, 75, 100)
Y_LABEL = "Top-1 accuracy (%)"
CAP = 1.5
CAT_XLIM = (-0.5, 2.6)
VALUE_DX = 0.09                       # value label right of its point (category units)
CHANCE_TEXT_DY = 1.5
EPS_XLIM = (0.004, 5.0)
EPS_TICKS = (0.01, 0.1, 1.0)
EPS_VALUE_DY = 2.5                    # value label clears the error bar by this much (pp)
GAP_X = 0.07                          # (c) line labels centred in the gap between budgets


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL_PP


def load() -> dict:
    cal = pd.read_csv(CAL_CSV, dtype={"K": str})
    out = {}
    for ds in ("he", "ntufi"):
        joint = json.loads(JOINT_JSON[ds].read_text())
        assert joint["n_classes"] == N_CLASSES[ds]
        assert np.isclose(joint["chance"], 1 / N_CLASSES[ds])
        cells = joint["cells"]

        def cell(defense: str, attacker: str) -> dict:
            c = cells[f"{defense}|{attacker}|T={QUERY_T}"]
            assert c["n"] == N_SEEDS, (ds, defense, attacker, c["n"])
            return dict(mean=c["mean"] * 100, sd=c["sd"] * 100)

        def cal_cell(n_cal: int) -> dict:
            r = cal[(cal["dataset"] == ds) & (cal["n_cal"] == n_cal) & (cal["attacker"] == NN)
                    & (cal["K"] == ONE_WINDOW)]
            assert len(r) == 1 and r["n_seeds"].iloc[0] == N_SEEDS, (ds, n_cal, len(r))
            assert np.isclose(r["chance"].iloc[0], 1 / N_CLASSES[ds])
            return dict(mean=r["acc_mean"].iloc[0] * 100, sd=r["acc_sd"].iloc[0] * 100)

        assert cal[cal["dataset"] == ds]["n_cal"].max() == SCALED_N_CAL[ds]
        d = dict(raw=cal_cell(0), scaled=cal_cell(SCALED_N_CAL[ds]), enc=cell(NONE, ENC))
        joint_none = cell(NONE, NN)
        # the scaled condition of the calibration sweep is the joint-defense no-defense run
        assert _close(joint_none["mean"], d["scaled"]["mean"]), (ds, joint_none, d["scaled"])
        dp_keys = sorted((k.split("|")[0] for k in cells
                          if k.startswith(DPSQ_PREFIX) and k.endswith(f"|{NN}|T={QUERY_T}")),
                         key=lambda k: float(k[len(DPSQ_PREFIX):]))
        dp = pd.DataFrame([dict(eps=float(k[len(DPSQ_PREFIX):]), **cell(k, NN)) for k in dp_keys])
        assert tuple(dp["eps"]) == EXPECT_EPS, (ds, dp_keys)
        exp = EXPECT[ds]
        for k in ATTACKERS:
            assert _close(round(d[k]["mean"], 1), exp[k]), (ds, k, d[k]["mean"])
        lo, hi = dp["mean"].min(), dp["mean"].max()
        assert _close(round(lo, 1), exp["dpsq"][0]) and _close(round(hi, 1), exp["dpsq"][1]), (ds, lo, hi)
        out[ds] = dict(att=d, dp=dp, chance=100 / N_CLASSES[ds])
    return out


def chance_line(ax, value: float, label: str, x: float, ha: str, inline: bool = False,
                data_x: bool = False) -> None:
    """Dashed chance line; its label above it, or on it in a white box (inline).
    x is in axes fraction, or in data units with data_x."""
    ax.axhline(value, color=COL_CHANCE, linestyle="--", linewidth=0.8, zorder=1)
    tr = ax.transData if data_x else blended_transform_factory(ax.transAxes, ax.transData)
    if inline:
        ax.text(x, value, label, color=TEXT_CHANCE, fontsize=ps.SMALL_PT, ha=ha, va="center",
                transform=tr, zorder=2, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    else:
        ax.text(x, value + CHANCE_TEXT_DY, label, color=TEXT_CHANCE, fontsize=ps.SMALL_PT,
                ha=ha, va="bottom", transform=tr)


def attacker_panel(ax, d: dict, ds: str, title: str) -> None:
    colour = COL[ds]
    for i, k in enumerate(ATTACKERS):
        v = d["att"][k]
        ax.errorbar([i], [v["mean"]], yerr=[v["sd"]], color=colour, marker=MARK[k],
                    markerfacecolor=colour if FILLED[k] else "white", linestyle="none",
                    capsize=CAP, zorder=4)
        ax.text(i + VALUE_DX, v["mean"], f"{v['mean']:.1f}", ha="left", va="center",
                fontsize=ps.SMALL_PT, color="black", zorder=5)
    ax.set_xlim(*CAT_XLIM)
    ax.set_xticks(range(len(ATTACKERS)))
    ax.set_xticklabels([ATTACKER_TICK[k] for k in ATTACKERS])
    ax.tick_params(axis="x", length=0)
    chance_line(ax, d["chance"], f"chance 1/{N_CLASSES[ds]}", 0.99, "right")
    ax.set_title(title, loc="left")


def eps_panel(ax, data: dict) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*EPS_XLIM)
    for ds, above in (("ntufi", True), ("he", False)):
        d, colour = data[ds], COL[ds]
        ref = d["att"]["scaled"]["mean"]
        ax.axhline(ref, color=colour, linestyle=":", linewidth=0.9, zorder=2)
        ax.text(GAP_X, ref + (CHANCE_TEXT_DY if above else -CHANCE_TEXT_DY),
                f"unprotected {ref:.1f}", color=colour, fontsize=ps.SMALL_PT, ha="center",
                va="bottom" if above else "top")
        dp = d["dp"]
        ax.errorbar(dp["eps"], dp["mean"], yerr=dp["sd"], color=colour, marker=MARK["scaled"],
                    linestyle="-", linewidth=1.0, capsize=CAP, zorder=3)
        for e, m, s in zip(dp["eps"], dp["mean"], dp["sd"]):
            ax.text(e, m - s - EPS_VALUE_DY, f"{m:.1f}", ha="center", va="top",
                    fontsize=ps.SMALL_PT, zorder=5)
    chance_line(ax, data["ntufi"]["chance"], "chance 1/7", GAP_X, "center", data_x=True)
    chance_line(ax, data["he"]["chance"], "chance 1/19", GAP_X, "center", inline=True, data_x=True)
    ax.set_xticks(list(EPS_TICKS))
    ax.set_xticklabels([f"{t:g}" for t in EPS_TICKS])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("DP-SQ privacy budget $\\varepsilon$ (log scale)")
    ax.set_title("(c) Under DP-SQ, 1-NN scaled", loc="left")


def main() -> None:
    ps.apply()
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_W_IN, FIG_H_IN), sharey=True,
                             gridspec_kw=dict(width_ratios=(1, 1, 1.25)))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.328, top=0.891, wspace=0.10)
    for ax in axes:
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks(list(Y_TICKS))
        ax.grid(axis="x", visible=False)
    attacker_panel(axes[0], d["he"], "he", "(a) He et al., real feedback, 19 positions")
    attacker_panel(axes[1], d["ntufi"], "ntufi", "(b) NTU-Fi proxy, 7 subjects")
    eps_panel(axes[2], d)
    for i, ax in enumerate(axes):           # shared axis: every panel carries the label, the first shows it
        ax.set_ylabel(Y_LABEL)
        ax.yaxis.label.set_visible(i == 0)

    handles = [
        plt.Line2D([], [], color=COL["he"], marker="o", linestyle="none", label="He et al., real feedback"),
        plt.Line2D([], [], color=COL["ntufi"], marker="o", linestyle="none", label="NTU-Fi proxy"),
        plt.Line2D([], [], color="black", marker=MARK["raw"], linestyle="none", label="1-NN, raw"),
        plt.Line2D([], [], color="black", marker=MARK["scaled"], linestyle="none",
                   label="1-NN, scaled on other targets"),
        plt.Line2D([], [], color="black", marker=MARK["enc"], markerfacecolor="white", linestyle="none",
                   label="learned encoder, scaled"),
        plt.Line2D([], [], color=COL_CHANCE, linestyle="--", label="chance"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.53, 0.0),
               handlelength=1.6, columnspacing=1.4, handletextpad=0.4)
    ps.save(fig, OUT_STEM)

    for ds in ("he", "ntufi"):
        a = d[ds]["att"]
        print(f"{ds}: " + ", ".join(f"{k} {a[k]['mean']:.1f} (SD {a[k]['sd']:.1f})" for k in ATTACKERS))
        print(f"{ds} DP-SQ:", [(e, round(m, 1), round(s, 1)) for e, m, s in
                               zip(d[ds]["dp"]["eps"], d[ds]["dp"]["mean"], d[ds]["dp"]["sd"])])
    print(f"wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
