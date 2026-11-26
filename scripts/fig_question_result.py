"""Figure 1, right panel: the 13-day strip chart.

One row per labelled device, one column per capture day. The one enrolled
session is a filled block on its day; every other day on which the device
sounds is coloured by the cross-day recall of that device from that session
(angles, K = 16 enrolled windows, T = 1, all-six gallery), binned as
identified (>= 90%), partly (50-90%) and not (< 50%). Days without any capture
are hatched; days on which the device did not sound stay white. Every number
is read from figures/r_recall_by_day.csv and the summary JSON and asserted.
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402

paper_style.apply()
C = paper_style.OKABE_ITO

BY_DAY_CSV = paper_style.FIG_DIR / "r_recall_by_day.csv"
SUMMARY_JSON = paper_style.FIG_DIR / "r_recall_by_day_summary.json"
STEM = "fig_question_result"

K_ENROL = "16"
FEATURE = "angles"
DEVICE_LABELS = {
    "AppleTV-LR": "Apple TV, living room",
    "LGTV": "LG TV",
    "Laptop": "Laptop",
    "Vacuum": "Robot vacuum",
    "AppleTV-BR": "Apple TV, bedroom",
    "SamsungTV": "Samsung TV",
}
# Rows in order of the enrolled day, so the enrolled blocks form a staircase.
ROW_ORDER = ["AppleTV-LR", "LGTV", "Laptop", "Vacuum", "AppleTV-BR", "SamsungTV"]

EXPECTED_DAYS = [f"2026-05-{d:02d}" for d in range(8, 21)]
EXPECTED_NO_DATA = ["2026-05-12", "2026-05-13"]
EXPECTED_CHANCE_PCT = 16.7
EXPECTED_ENROL_DAY = {"AppleTV-LR": "2026-05-08", "LGTV": "2026-05-09",
                      "Laptop": "2026-05-11", "Vacuum": "2026-05-11",
                      "AppleTV-BR": "2026-05-15", "SamsungTV": "2026-05-15"}
# Recall in % per (device, query day); the script fails if the CSV disagrees.
EXPECTED_RECALL = {
    "AppleTV-LR": {"09": 100.0, "10": 94.0, "11": 80.4, "14": 43.9, "15": 51.5, "16": 57.9,
                   "17": 60.2, "18": 49.1, "19": 38.6, "20": 18.0},
    "LGTV": {"10": 100.0, "11": 100.0, "14": 100.0, "15": 100.0},
    "Laptop": {"14": 100.0, "15": 98.9, "16": 99.5},
    "Vacuum": {"08": 100.0, "15": 88.3, "18": 45.6},
    "AppleTV-BR": {"16": 94.7, "17": 100.0, "18": 100.0, "19": 100.0, "20": 100.0},
    "SamsungTV": {"16": 0.0},
}
TOLERANCE_PCT = 0.05

BIN_IDENTIFIED_PCT = 90.0
BIN_PARTLY_PCT = 50.0
COLOR_ENROLLED = C["blue"]
COLOR_IDENTIFIED = C["green"]
COLOR_PARTLY = "#99D8C4"               # green at ~45% tint
COLOR_NOT = C["vermilion"]
COLOR_NO_DATA = "#EEEEEE"
COLOR_SILENT = "white"
CELL_EDGE = "white"
CELL_PAD = 0.06                        # gap between cells, in cell units
MIN_QUERY_WINDOWS = 10                 # days with fewer query windows are drawn hollow
NO_CAPTURE_TEXT = "no capture"

FIG_W_IN = 0.55 * paper_style.TEXT_W_IN
FIG_H_IN = 2.1


def load() -> tuple[list[dict], dict]:
    with open(BY_DAY_CSV, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["K"] == K_ENROL and r["feature"] == FEATURE]
    return rows, json.load(open(SUMMARY_JSON))


def recall_bin_color(pct: float) -> str:
    if pct >= BIN_IDENTIFIED_PCT:
        return COLOR_IDENTIFIED
    if pct >= BIN_PARTLY_PCT:
        return COLOR_PARTLY
    return COLOR_NOT


def main() -> None:
    rows, summary = load()
    days = summary["capture_days"]
    assert days == EXPECTED_DAYS, days
    assert summary["days_without_data"] == EXPECTED_NO_DATA, summary["days_without_data"]
    assert round(100.0 * summary["chance"], 1) == EXPECTED_CHANCE_PCT, summary["chance"]
    assert set(summary["gallery"]) == set(DEVICE_LABELS), summary["gallery"]
    enrol_day = {d: summary["enrolled"][d]["day"] for d in DEVICE_LABELS}
    assert enrol_day == EXPECTED_ENROL_DAY, enrol_day

    recall = {}                                   # (device, day) -> recall %
    n_query = {}
    for r in rows:
        recall[(r["device"], r["query_day"])] = 100.0 * float(r["recall"])
        n_query[(r["device"], r["query_day"])] = int(r["query_windows"])
    for dev, exp in EXPECTED_RECALL.items():
        got = {k[1][-2:]: v for k, v in recall.items() if k[0] == dev}
        assert set(got) == set(exp), (dev, sorted(got), sorted(exp))
        for dd, v in exp.items():
            assert abs(got[dd] - v) <= TOLERANCE_PCT, (dev, dd, got[dd], v)
    sounding = summary["sounding_days"]
    for dev in DEVICE_LABELS:                     # every sounding day is enrolled or queried
        assert set(sounding[dev]) == {enrol_day[dev]} | {k[1] for k in recall if k[0] == dev}, dev

    col = {d: i for i, d in enumerate(days)}
    row = {d: len(ROW_ORDER) - 1 - i for i, d in enumerate(ROW_ORDER)}

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0.26, 0.36, 0.72, 0.57])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    def cell(x: int, y: int, color: str, hollow: bool = False, **kw) -> None:
        ax.add_patch(Rectangle((x + CELL_PAD, y + CELL_PAD), 1 - 2 * CELL_PAD, 1 - 2 * CELL_PAD,
                               facecolor="white" if hollow else color,
                               edgecolor=color if hollow else CELL_EDGE,
                               linewidth=0.8 if hollow else 0.4, **kw))

    for d in EXPECTED_NO_DATA:
        ax.add_patch(Rectangle((col[d], 0), 1, len(ROW_ORDER), facecolor=COLOR_NO_DATA,
                               edgecolor="none", zorder=1))
    ax.text(col[EXPECTED_NO_DATA[0]] + len(EXPECTED_NO_DATA) / 2, len(ROW_ORDER) / 2,
            NO_CAPTURE_TEXT, rotation=90, ha="center", va="center", color=C["grey"],
            fontsize=paper_style.SMALL_PT, zorder=2)
    for dev in ROW_ORDER:
        for d in days:
            if d == enrol_day[dev]:
                cell(col[d], row[dev], COLOR_ENROLLED, zorder=3)
            elif (dev, d) in recall:
                cell(col[d], row[dev], recall_bin_color(recall[(dev, d)]),
                     hollow=n_query[(dev, d)] < MIN_QUERY_WINDOWS, zorder=3)

    ax.set_xlim(0, len(days))
    ax.set_ylim(0, len(ROW_ORDER))
    ax.set_xticks([col[d] + 0.5 for d in days])
    ax.set_xticklabels([str(int(d[-2:])) for d in days])
    ax.set_yticks([row[d] + 0.5 for d in ROW_ORDER])
    ax.set_yticklabels([DEVICE_LABELS[d] for d in ROW_ORDER])
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Capture day (May 2026)")
    ax.set_ylabel("Device", rotation=0, ha="right", va="bottom")
    ax.yaxis.set_label_coords(-0.01, 1.0)

    handles = [
        Patch(facecolor=COLOR_ENROLLED, label="enrolled session"),
        Patch(facecolor=COLOR_IDENTIFIED, label=f"recall ≥ {BIN_IDENTIFIED_PCT:.0f}%"),
        Patch(facecolor=COLOR_PARTLY, label=f"{BIN_PARTLY_PCT:.0f}–{BIN_IDENTIFIED_PCT:.0f}%"),
        Patch(facecolor=COLOR_NOT, label=f"< {BIN_PARTLY_PCT:.0f}%"),
        Patch(facecolor="white", edgecolor=C["grey"], linewidth=0.8,
              label=f"< {MIN_QUERY_WINDOWS} windows that day"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0, labelspacing=0.3)

    paper_style.save(fig, STEM)
    print("plotted (recall %, device: day=value):")
    for dev in ROW_ORDER:
        cells = ", ".join(f"{k[1][-2:]}={v:.1f} (n={n_query[k]})"
                          for k, v in sorted(recall.items()) if k[0] == dev)
        print(f"  {DEVICE_LABELS[dev]:22s} enrolled {enrol_day[dev][-2:]}; {cells}")
    print(f"  chance {EXPECTED_CHANCE_PCT}; no capture {', '.join(EXPECTED_NO_DATA)}")


if __name__ == "__main__":
    main()
