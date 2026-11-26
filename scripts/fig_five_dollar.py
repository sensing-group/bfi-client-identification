#!/usr/bin/env python3
"""C5b, T1: what the 512-byte cut of the ESP32-C5 firmware costs, per device.

One row per device. Filled marker: recall across days from one whole labeled
session on the full reports (Pi 5 + AX210, untruncated). Hollow marker: the
same reports cut to 512 bytes as the ESP32-C5 firmware keeps them, with the
"common" mask (the 11 subcarrier bins every device still covers). Both are
1-NN on Givens angles, K = all windows of the enrolled session, T = 1 window,
condition other_day_any, chance 1/6. The tones each device keeps after the
cut are printed at the right margin.

Data (all in figures/):
  c5b_trunc_side_by_side.csv  table=per_device rows: condition, feature, K, T,
                              device, mask_policy, untruncated, truncated, chance
  c5b_trunc_per_device.csv    the truncated runs alone (cross-check of the
                              truncated column)
  c5b_trunc_summary.json      formats[]: device, tones_full, tones_kept, windows
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import paper_style as ps
from device_names import TABLE2_NAME_OF_SHORT, TABLE2_ORDER

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_five_dollar"
SIDE_BY_SIDE_CSV = FIG_DIR / "c5b_trunc_side_by_side.csv"
PER_DEVICE_CSV = FIG_DIR / "c5b_trunc_per_device.csv"
SUMMARY_JSON = FIG_DIR / "c5b_trunc_summary.json"

CONDITION = "other_day_any"
FEATURE = "angles"
K = "all"
T = 1
MASK = "common"
BODY_CAP_B = 512
N_GALLERY = 6

# Top-to-bottom row order, with the device names of Table 2.
DEVICES = tuple((k, TABLE2_NAME_OF_SHORT[k]) for k in TABLE2_ORDER)

# Values the report quotes (recall, fraction); the script refuses to draw if the CSV disagrees.
EXPECT = {"LGTV": (1.00, 1.00), "AppleTV-BR": (0.997, 0.983), "Laptop": (0.99, 0.83),
          "AppleTV-LR": (0.985, 0.731), "Vacuum": (0.585, 0.375), "SamsungTV": (0.00, 0.00)}
TOL = 0.0051

COL_MARK = ps.OKABE_ITO["blue"]
COL_LINK = ps.OKABE_ITO["grey"]
COL_CHANCE = ps.OKABE_ITO["grey"]
COL_TONES = "#5E5E5E"
MARK = "o"
MS_FULL = 3.6
MS_CUT = 5.2                        # the ring is wider than the dot, so a coincident pair shows both
LINK_LW = 0.8
FIG_H_IN = 2.3
X_MAX = 100
X_TICKS = (0, 25, 50, 75, 100)
X_LABEL = "Recall across days from one whole session (%)"
TONES_X_AXES = 1.045                # axes fraction: just right of the frame
Y_TICK_PAD_PT = 6                   # keeps a marker at 0% off the device name
TONES_HEADER = "tones kept"
CHANCE_LABEL = f"chance 1/{N_GALLERY}"
LEGEND_FULL = "full report (Pi 5)"
LEGEND_CUT = f"Pi report cut to {BODY_CAP_B} bytes"


def load() -> pd.DataFrame:
    sbs = pd.read_csv(SIDE_BY_SIDE_CSV, dtype={"K": str})
    rows = sbs[(sbs["table"] == "per_device") & (sbs["condition"] == CONDITION)
               & (sbs["feature"] == FEATURE) & (sbs["K"] == K) & (sbs["T"] == T)
               & (sbs["mask_policy"] == MASK)].set_index("device")
    assert len(rows) == N_GALLERY, rows.index.tolist()

    per = pd.read_csv(PER_DEVICE_CSV, dtype={"K": str})
    per = per[(per["condition"] == CONDITION) & (per["feature"] == FEATURE) & (per["K"] == K)
              & (per["T"] == T) & (per["mask_policy"] == MASK)].set_index("device")
    assert len(per) == N_GALLERY

    summary = json.loads(SUMMARY_JSON.read_text())
    assert summary["body_cap"] == BODY_CAP_B
    assert summary["masks"][MASK]["bins_kept"] == 11
    formats = pd.DataFrame(summary["formats"])
    # The format that carries the device's windows gives its tone count after the cut.
    main_fmt = formats.sort_values("windows", ascending=False).drop_duplicates("device").set_index("device")

    out = []
    for key, name in DEVICES:
        full, cut = rows.loc[key, "untruncated"], rows.loc[key, "truncated"]
        assert abs(cut - per.loc[key, "recall"]) < 1e-9, (key, cut, per.loc[key, "recall"])
        assert abs(rows.loc[key, "chance"] - 1 / N_GALLERY) < 1e-6
        assert abs(full - EXPECT[key][0]) <= TOL, (key, full, EXPECT[key][0])
        assert abs(cut - EXPECT[key][1]) <= TOL, (key, cut, EXPECT[key][1])
        out.append(dict(device=key, name=name, full=full * 100, cut=cut * 100,
                        tones_kept=int(main_fmt.loc[key, "tones_kept"]),
                        tones_full=int(main_fmt.loc[key, "tones_full"]),
                        n_decisions=int(rows.loc[key, "n_decisions"])))
    return pd.DataFrame(out)


def main() -> None:
    ps.apply()
    d = load()
    n = len(d)
    y = list(range(n - 1, -1, -1))              # first device on top

    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.31, right=0.84, bottom=0.30, top=0.90)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(d["name"])
    ax.set_xticks(list(X_TICKS))
    ax.set_xlabel(X_LABEL)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=Y_TICK_PAD_PT)

    chance = 100 / N_GALLERY
    ax.axvline(chance, color=COL_CHANCE, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(chance + 1.5, n - 0.5, CHANCE_LABEL, color=COL_TONES, fontsize=ps.SMALL_PT,
            ha="left", va="top")

    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.cut, r.full], [yi, yi], color=COL_LINK, linewidth=LINK_LW, zorder=2)
        ax.plot([r.full], [yi], marker=MARK, color=COL_MARK, markersize=MS_FULL,
                linestyle="none", clip_on=False, zorder=4)
        ax.plot([r.cut], [yi], marker=MARK, color=COL_MARK, markerfacecolor="none",
                markersize=MS_CUT, linestyle="none", clip_on=False, zorder=5)
        ax.text(TONES_X_AXES, yi, f"{r.tones_kept}/{r.tones_full}", color=COL_TONES,
                fontsize=ps.SMALL_PT, ha="left", va="center", clip_on=False,
                transform=ax.get_yaxis_transform())
    ax.text(TONES_X_AXES, n - 0.4, TONES_HEADER, color=COL_TONES, fontsize=ps.SMALL_PT,
            ha="left", va="bottom", clip_on=False, transform=ax.get_yaxis_transform())

    handles = [
        plt.Line2D([], [], color=COL_MARK, marker=MARK, markersize=MS_FULL, linestyle="none",
                   label=LEGEND_FULL),
        plt.Line2D([], [], color=COL_MARK, marker=MARK, markerfacecolor="none", markersize=MS_CUT,
                   linestyle="none", label=LEGEND_CUT),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.005),
               handlelength=1.4, columnspacing=1.6, handletextpad=0.5)
    ps.save(fig, OUT_STEM)

    for r in d.itertuples():
        print(f"{r.name:22s} full {r.full:6.1f}%  cut {r.cut:6.1f}%  tones {r.tones_kept}/{r.tones_full}"
              f"  decisions {r.n_decisions}")
    print(f"wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
