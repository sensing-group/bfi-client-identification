#!/usr/bin/env python3
"""Figure 12(b): what an observer still reads under each protection, on the home capture.

Accuracy across days, one window enrolled, T = 1, chance 17 %, whiskers the 95 %
bootstrap intervals of the source rows. Rotated angles come from r_rotation.py;
the format is unchanged by any protection below the frame and, since each format
has one body length (r_snr_field.py), reads the same from the length of an
encrypted frame. The SNR rows come from r_snr_field.py: a unitary does not change
the field, so the unrotated bytes are those of the rotated report. Hollow marks
are derived, not measured: the length row repeats the format, and a padded frame
leaves no field of the report, so its mark sits at chance. RSSI and report timing,
which no protection of the frame changes, are measured on their own by
r_round4_checks.py (part 1) and drawn filled.

Data (figures/): r_rotation_main.csv, r_snr_field.csv, r_snr_field_summary.json,
r_round4_checks.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_what_stays"
ROTATION_CSV = FIG_DIR / "r_rotation_main.csv"
SNR_CSV = FIG_DIR / "r_snr_field.csv"
SNR_JSON = FIG_DIR / "r_snr_field_summary.json"
ROUND4_CSV = FIG_DIR / "r_round4_checks.csv"
ROUND4_PART = "1"                  # the timing and RSSI baseline
CHANCE_PCT = 100 / 6

FIG_H_IN = 1.61
LAYOUT = dict(left=0.315, right=0.975, bottom=0.222, top=0.891)
X_LIM = (0, 100)
X_TICKS = (0, 20, 40, 60, 80, 100)
X_LABEL = "Observer accuracy across days (%)"
TITLE = "(b) What stays readable, home capture"
ROW_STEP = 1.0
GROUP_GAP = 0.45
COL_LEAK, COL_HIDDEN, COL_REF = ps.OKABE_ITO["vermilion"], ps.OKABE_ITO["green"], ps.OKABE_ITO["grey"]
TEXT_REF = "#5E5E5E"
REF_DASH = (0, (3, 3))
REF_LW = 0.6
WHISKER_LW = 0.7
DOT_MS, STAR_MS = 4.0, 8.0
CHANCE_LABEL_XY = (CHANCE_PCT + 1.5, -0.15)
EXPECTED = {"none": 83, "rotated": 21, "format": 79, "snr": 71, "both": 88, "rssi": 74, "timing": 41}
TOL_PCT = 1.0


def row(rows: list[dict], **where) -> tuple[float, float, float]:
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} rows match {where}")
    r = hits[0]
    return 100 * float(r["accuracy"]), 100 * float(r["ci_lo"]), 100 * float(r["ci_hi"])


def load() -> dict[str, tuple[float, float, float]]:
    with open(ROTATION_CSV, newline="") as f:
        rot = list(csv.DictReader(f))
    with open(SNR_CSV, newline="") as f:
        snr = list(csv.DictReader(f))
    base = dict(condition="other_day_any", K="1", T="1")
    level = dict(base, streams="1", power="none", observer="level")
    d = dict(none=row(rot, body="none", feature="angles", **base),
             rotated=row(rot, body="rot_group", feature="angles", **base),
             format=row(rot, body="rot_group", feature="format", **base),
             snr=row(snr, feature="angles", **level),
             both=row(snr, feature="angles+format", **level))
    with open(ROUND4_CSV, newline="") as f:
        air = [r for r in csv.DictReader(f) if r["part"] == ROUND4_PART and r["K"] == "1" and r["T"] == "1"
               and r["metric"] == "accuracy"]
    for cell in ("rssi", "timing"):
        hits = [r for r in air if r["cell"] == cell]
        assert len(hits) == 1, (cell, len(hits))
        d[cell] = tuple(100 * float(hits[0][k]) for k in ("value", "lo", "hi"))
    for k, v in EXPECTED.items():
        assert abs(d[k][0] - v) <= TOL_PCT, (k, d[k][0], v)
    lengths = json.loads(SNR_JSON.read_text())["format_body_lengths"].values()
    assert all(len(v) == 1 for v in lengths) and len({v[0] for v in lengths}) == len(lengths)
    return d


def main() -> None:
    d = load()
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN))
    fig.subplots_adjust(**LAYOUT)
    groups = [[("No defense: angles", d["none"], "o", "black", "black")],
              [("Rotation: angles", d["rotated"], "*", COL_HIDDEN, "black"),
               ("Rotation: format", d["format"], "s", COL_LEAK, COL_LEAK),
               ("Rotation: SNR field", d["snr"], "s", "white", COL_LEAK),
               ("Rotation: format, SNR", d["both"], "s", "white", COL_LEAK)],
              [("Protected frame: length", d["format"], "s", "white", COL_LEAK),
               ("Padded frame: report", (CHANCE_PCT,) * 3, "o", "white", TEXT_REF),
               ("Padded frame: RSSI", d["rssi"], "s", COL_LEAK, COL_LEAK),
               ("Padded frame: timing", d["timing"], "s", COL_LEAK, COL_LEAK)]]
    y, ticks, names = 0.0, [], []
    ax.axvline(CHANCE_PCT, color=COL_REF, ls=REF_DASH, lw=REF_LW, zorder=1)
    for group in groups:
        for name, (acc, lo, hi), marker, face, edge in group:
            ax.plot([lo, hi], [y, y], color=edge, lw=WHISKER_LW, zorder=2, solid_capstyle="butt")
            ax.plot([acc], [y], marker=marker, ms=STAR_MS if marker == "*" else DOT_MS, mfc=face, mec=edge,
                    mew=0.7, ls="none", zorder=3, clip_on=False)
            ticks.append(y)
            names.append(name)
            y += ROW_STEP
        y += GROUP_GAP
    ax.set_yticks(ticks, names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y - GROUP_GAP - ROW_STEP + 0.6, -0.6)
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel(X_LABEL)
    ax.set_title(TITLE, loc="left")
    ax.text(*CHANCE_LABEL_XY, "chance", color=TEXT_REF, fontsize=ps.SMALL_PT, ha="left", va="center")
    ps.save(fig, OUT_STEM)
    print(f"{ {k: round(v[0], 1) for k, v in d.items()} }; wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
