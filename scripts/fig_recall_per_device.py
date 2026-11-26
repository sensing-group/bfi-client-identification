"""Figure 6: recall across days per device from one labeled session (column width).

Cross-day, all-six gallery (condition other_day_any), nearest neighbour on the
angles, K enrolled windows from one session, one query window (T = 1). The
format-only baseline (bandwidth and antenna configuration announced in the
report) is drawn as a grey tick on the same row. Every plotted number is read
from figures/r_per_device.csv and asserted below.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402
from device_names import TABLE2_NAME_OF_SHORT, TABLE2_ORDER  # noqa: E402

paper_style.apply()
C = paper_style.OKABE_ITO

PER_DEVICE_CSV = paper_style.FIG_DIR / "r_per_device.csv"
SUMMARY_JSON = paper_style.FIG_DIR / "r_eval_summary.json"
STEM = "fig_recall_per_device"

GAP_S = "300"
CONDITION = "other_day_any"            # cross-day, every labeled device in the gallery
T_QUERY = "1"
K_ONE, K_MANY = "1", "16"
FEATURE_ANGLES, FEATURE_FORMAT = "angles", "format"

DEVICE_LABELS = TABLE2_NAME_OF_SHORT     # short key -> device name of Table 2

# Values the paper quotes; the script fails if the CSV disagrees (recall in %).
EXPECTED = {
    (FEATURE_ANGLES, K_ONE): {"LGTV": 100.0, "AppleTV-BR": 99.0, "AppleTV-LR": 92.4,
                              "Laptop": 88.5, "Vacuum": 42.0, "SamsungTV": 6.0},
    (FEATURE_ANGLES, K_MANY): {"LGTV": 100.0, "AppleTV-BR": 99.7, "AppleTV-LR": 96.4,
                               "Laptop": 99.0, "Vacuum": 60.0, "SamsungTV": 0.0},
    (FEATURE_FORMAT, K_ONE): {"LGTV": 100.0, "AppleTV-BR": 92.0, "AppleTV-LR": 100.0,
                              "Laptop": 60.5, "Vacuum": 32.0, "SamsungTV": 7.0},
}
EXPECTED_CHANCE_PCT = 16.7             # one of six enrolled devices per decision
TOLERANCE_PCT = 0.05

FIG_W_IN = paper_style.COLUMN_W_IN
FIG_H_IN = 2.3
MARKER_PT = 5.0
TICK_HALF = 0.32                       # half-height of the format-only tick, in row units
TEXT_GREY = "#5E5E5E"            # the format ticks and small text, 6:1 on white


def load_rows() -> list[dict]:
    with open(PER_DEVICE_CSV, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["gap_s"] == GAP_S and r["condition"] == CONDITION and r["T"] == T_QUERY]


def recall_pct(rows: list[dict], feature: str, k: str) -> dict[str, float]:
    out = {}
    for r in rows:
        if r["feature"] == feature and r["K"] == k:
            out[r["device"]] = 100.0 * float(r["recall"])
    assert set(out) == set(DEVICE_LABELS), (feature, k, sorted(out))
    return out


def chance_pct(rows: list[dict]) -> float:
    vals = {round(100.0 * float(r["chance"]), 1) for r in rows if r["feature"] == FEATURE_ANGLES}
    assert vals == {EXPECTED_CHANCE_PCT}, vals
    return EXPECTED_CHANCE_PCT


def check(values: dict[str, float], expected: dict[str, float], what: tuple) -> None:
    for dev, exp in expected.items():
        assert abs(values[dev] - exp) <= TOLERANCE_PCT, (what, dev, values[dev], exp)


def main() -> None:
    summary = json.load(open(SUMMARY_JSON))
    assert K_ONE in summary["k_grid"] and K_MANY in summary["k_grid"], summary["k_grid"]
    assert int(T_QUERY) in summary["t_grid"], summary["t_grid"]

    rows = load_rows()
    k1 = recall_pct(rows, FEATURE_ANGLES, K_ONE)
    k16 = recall_pct(rows, FEATURE_ANGLES, K_MANY)
    fmt = recall_pct(rows, FEATURE_FORMAT, K_ONE)
    chance = chance_pct(rows)
    check(k1, EXPECTED[(FEATURE_ANGLES, K_ONE)], (FEATURE_ANGLES, K_ONE))
    check(k16, EXPECTED[(FEATURE_ANGLES, K_MANY)], (FEATURE_ANGLES, K_MANY))
    check(fmt, EXPECTED[(FEATURE_FORMAT, K_ONE)], (FEATURE_FORMAT, K_ONE))

    # Devices in the row order of Table 2.
    order = list(TABLE2_ORDER)
    y = {d: len(order) - 1 - i for i, d in enumerate(order)}

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0.34, 0.37, 0.63, 0.57])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    ax.axvline(chance, color=C["grey"], linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.text(chance + 1.5, len(order) - 0.55, "chance", color=TEXT_GREY,
            fontsize=paper_style.SMALL_PT, ha="left", va="center", zorder=20)

    for d in order:
        ax.plot([k1[d], k16[d]], [y[d], y[d]], color=C["blue"], linewidth=0.8, zorder=4)
        ax.plot([fmt[d], fmt[d]], [y[d] - TICK_HALF, y[d] + TICK_HALF], color=TEXT_GREY,
                linewidth=1.2, solid_capstyle="butt", zorder=12)
    ax.scatter([k16[d] for d in order], [y[d] for d in order], s=MARKER_PT ** 2,
               facecolors="white", edgecolors=C["blue"], linewidths=0.9, zorder=10, clip_on=False)
    ax.scatter([k1[d] for d in order], [y[d] for d in order], s=MARKER_PT ** 2,
               facecolors=C["blue"], edgecolors=C["blue"], linewidths=0.9, zorder=11, clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks([y[d] for d in order])
    ax.set_yticklabels([DEVICE_LABELS[d] for d in order])
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel("Recall across days (%)")
    ax.set_ylabel("Device", rotation=0, ha="right", va="bottom")
    ax.yaxis.set_label_coords(-0.02, 1.0)

    handles = [
        plt.Line2D([], [], marker="o", color=C["blue"], markerfacecolor=C["blue"], linestyle="none",
                   markersize=MARKER_PT, label="angles, 1 labeled window"),
        plt.Line2D([], [], marker="o", color=C["blue"], markerfacecolor="white", linestyle="none",
                   markersize=MARKER_PT, label="angles, 16 labeled windows"),
        plt.Line2D([], [], marker="|", color=TEXT_GREY, linestyle="none", markersize=6,
                   markeredgewidth=1.2, label="format only"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              handletextpad=0.4, columnspacing=1.0)

    paper_style.save(fig, STEM)
    print("plotted (recall %, device: K=1 / K=16 / format-only):")
    for d in order:
        print(f"  {DEVICE_LABELS[d]:22s} {k1[d]:5.1f} / {k16[d]:5.1f} / {fmt[d]:5.1f}")
    print(f"  chance {chance:.1f}")


if __name__ == "__main__":
    main()
