"""Section 6: the 13-day strip, whole enrolled session (K = all), one panel.

Rows are the six labelled devices, columns the 13 capture days. Each day on
which a device sounds carries a square tile coloured by the recall of that
device from its one enrolled session (angles, T = 1, 1-NN), binned at 90 and
50%, with the recall printed inside; the enrolled day is a blue tile marked
"enrolled". Days with fewer than MIN_QUERY_WINDOWS query windows print the
number in grey italics with no tile; 12-13 May, with no capture, are one grey
band. Every number is read from figures/r_recall_by_day_drift.csv and the
summary JSON and asserted against EXPECTED below.

"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402
from device_names import TABLE2_NAME_OF_SHORT, TABLE2_ORDER  # noqa: E402
from fig_question_result import (  # noqa: E402
    BIN_IDENTIFIED_PCT, BIN_PARTLY_PCT, COLOR_ENROLLED, COLOR_IDENTIFIED, COLOR_NO_DATA,
    COLOR_NOT, COLOR_PARTLY, DEVICE_LABELS, EXPECTED_DAYS, EXPECTED_ENROL_DAY,
    EXPECTED_NO_DATA, EXPECTED_RECALL, MIN_QUERY_WINDOWS, NO_CAPTURE_TEXT,
    TOLERANCE_PCT, recall_bin_color)

paper_style.apply()
C = paper_style.OKABE_ITO
ROW_ORDER = TABLE2_ORDER

DRIFT_CSV = paper_style.FIG_DIR / "r_recall_by_day_drift.csv"
DRIFT_JSON = paper_style.FIG_DIR / "r_recall_by_day_drift_summary.json"
STEM = "fig_recall_strip"
K_ENROL = "all"

FEATURE, T_QUERY, RULE = "angles", "1", "nn"
EXPECTED_RECALL_ALL = {
    "AppleTV-LR": {"09": 100.0, "10": 99.7, "11": 100.0, "14": 100.0, "15": 100.0, "16": 94.7,
                   "17": 99.9, "18": 100.0, "19": 99.9, "20": 100.0},
    "LGTV": {"10": 100.0, "11": 100.0, "14": 100.0, "15": 100.0},
    "Laptop": {"14": 100.0, "15": 99.2, "16": 100.0},
    "Vacuum": {"08": 100.0, "15": 93.3, "18": 13.2},
    "AppleTV-BR": {"16": 94.7, "17": 100.0, "18": 100.0, "19": 100.0, "20": 100.0},
    "SamsungTV": {"16": 0.0},
}
EXPECTED = {"16": EXPECTED_RECALL, "all": EXPECTED_RECALL_ALL}

FIG_W_IN = paper_style.TEXT_W_IN
FIG_H_IN = 1.9
AX_RECT = (0.152, 0.205, 0.843, 0.785)   # figure fraction; the bottom holds ticks and one label line
TILE_FRAC = 0.86                         # tile height as a fraction of the row pitch
TILE_W_FRAC = 0.86                       # tile width as a fraction of the column pitch; None = square
TILE_TEXT_PT = paper_style.SMALL_PT
NOTE_PT = paper_style.FONT_PT             # the legend line under the axis, 8 pt (italic reads small)
ENROLLED_TEXT = "enrolled"
TEXT_ON_DARK = "white"
TEXT_ON_LIGHT = "#1A1A1A"
DARK_TILES = {COLOR_ENROLLED, COLOR_IDENTIFIED, COLOR_NOT}
COLOR_FEW = "#5E5E5E"                     # days with too few query windows: grey italic number
X_LABEL = "Capture day (May 2026)"
NOTE_X = 1.0                              # right-aligned at the axes' right edge
NOTE_LEAD = "tile colour: recall"
NOTE_FEW = f"grey italic: < {MIN_QUERY_WINDOWS} windows; blank: not heard"
SWATCH_W_AX = 0.013                       # swatch width, axes fraction
SWATCH_H_AX = 0.085
NOTE_GAP_AX = 0.006                       # gap between a swatch and its text


def load(k: str) -> tuple[dict, dict, dict]:
    with open(DRIFT_CSV, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["feature"] == FEATURE and r["T"] == T_QUERY and r["rule"] == RULE
                and r["K"] == k and int(r["n_queries"]) > 0]
    recall, n_query = {}, {}
    for r in rows:
        recall[(r["device"], r["query_day"])] = 100.0 * float(r["recall"])
        n_query[(r["device"], r["query_day"])] = int(r["n_queries"])
    return recall, n_query, json.load(open(DRIFT_JSON))


def check(k: str, recall: dict, summary: dict) -> dict:
    assert summary["published_strip"] == {"feature": FEATURE, "K": 16, "T": 1, "rule": RULE}
    enrol_day = {d: summary["enrolled"][d]["day"] for d in DEVICE_LABELS}
    assert enrol_day == EXPECTED_ENROL_DAY, enrol_day
    for dev, exp in EXPECTED[k].items():
        got = {key[1][-2:]: v for key, v in recall.items() if key[0] == dev}
        assert set(got) == set(exp), (k, dev, sorted(got), sorted(exp))
        for dd, v in exp.items():
            assert abs(got[dd] - v) <= TOLERANCE_PCT, (k, dev, dd, got[dd], v)
    return enrol_day


def note(ax) -> None:
    """One line under the axis: three swatches for the three recall bins, then the grey-italic rule."""
    items = [(COLOR_IDENTIFIED, f"≥ {BIN_IDENTIFIED_PCT:.0f}"),
             (COLOR_PARTLY, f"{BIN_PARTLY_PCT:.0f}–{BIN_IDENTIFIED_PCT:.0f}"),
             (COLOR_NOT, f"< {BIN_PARTLY_PCT:.0f} %")]
    # Lay the pieces out right to left, measuring each, so the line ends exactly at NOTE_X.
    fig = ax.figure
    fig.canvas.draw()                      # places the x label, whose line the note shares
    renderer = fig.canvas.get_renderer()
    bb = ax.xaxis.label.get_window_extent(renderer)
    note_y = ax.transAxes.inverted().transform((0, bb.y0 + bb.height / 2))[1]
    x = NOTE_X
    kw = dict(transform=ax.transAxes, va="center", fontsize=NOTE_PT, clip_on=False)

    def width_ax(artist) -> float:
        bb = artist.get_window_extent(renderer)
        return bb.width / ax.get_window_extent(renderer).width

    t = ax.text(x, note_y, NOTE_FEW, ha="right", color=COLOR_FEW, fontstyle="italic", **kw)
    x -= width_ax(t) + 0.02
    t = ax.text(x, note_y, "·", ha="right", color=COLOR_FEW, **kw)
    x -= width_ax(t) + 0.012
    for colour, label in reversed(items):
        t = ax.text(x, note_y, label, ha="right", **kw)
        x -= width_ax(t) + NOTE_GAP_AX
        ax.add_patch(Rectangle((x - SWATCH_W_AX, note_y - SWATCH_H_AX / 2), SWATCH_W_AX, SWATCH_H_AX,
                               transform=ax.transAxes, facecolor=colour, edgecolor="none", clip_on=False))
        x -= SWATCH_W_AX + 0.014
    ax.text(x, note_y, NOTE_LEAD, ha="right", **kw)


def draw(k: str, stem: str) -> None:
    recall, n_query, summary = load(k)
    enrol_day = check(k, recall, summary)
    days = EXPECTED_DAYS
    col = {d: i for i, d in enumerate(days)}
    row = {d: len(ROW_ORDER) - 1 - i for i, d in enumerate(ROW_ORDER)}
    n_rows, n_cols = len(ROW_ORDER), len(days)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes(AX_RECT)
    ax.grid(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    # square tiles: the side is TILE_FRAC of the row pitch, converted to column units
    row_in = FIG_H_IN * AX_RECT[3] / n_rows
    col_in = FIG_W_IN * AX_RECT[2] / n_cols
    side_y = TILE_FRAC
    side_x = TILE_FRAC * row_in / col_in if TILE_W_FRAC is None else TILE_W_FRAC

    def tile(x: int, y: int, colour: str, text: str, text_colour: str) -> None:
        ax.add_patch(Rectangle((x + 0.5 - side_x / 2, y + 0.5 - side_y / 2), side_x, side_y,
                               facecolor=colour, edgecolor="none", zorder=3))
        ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", color=text_colour,
                fontsize=TILE_TEXT_PT, zorder=4)

    for d in EXPECTED_NO_DATA:
        ax.add_patch(Rectangle((col[d], 0), 1, n_rows, facecolor=COLOR_NO_DATA, edgecolor="none",
                               zorder=1))
    ax.text(col[EXPECTED_NO_DATA[0]] + len(EXPECTED_NO_DATA) / 2, n_rows / 2, NO_CAPTURE_TEXT,
            ha="center", va="center", color=COLOR_FEW, fontsize=paper_style.SMALL_PT, zorder=2)
    for dev in ROW_ORDER:
        y = row[dev]
        for d in days:
            if d == enrol_day[dev]:
                tile(col[d], y, COLOR_ENROLLED, ENROLLED_TEXT, TEXT_ON_DARK)
            elif (dev, d) in recall:
                value = f"{recall[(dev, d)]:.0f}"
                if n_query[(dev, d)] < MIN_QUERY_WINDOWS:
                    ax.text(col[d] + 0.5, y + 0.5, value, ha="center", va="center", color=COLOR_FEW,
                            fontstyle="italic", fontsize=TILE_TEXT_PT, zorder=4)
                else:
                    colour = recall_bin_color(recall[(dev, d)])
                    tile(col[d], y, colour, value,
                         TEXT_ON_DARK if colour in DARK_TILES else TEXT_ON_LIGHT)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([col[d] + 0.5 for d in days])
    ax.set_xticklabels([str(int(d[-2:])) for d in days])
    ax.set_yticks([row[d] + 0.5 for d in ROW_ORDER])
    ax.set_yticklabels([TABLE2_NAME_OF_SHORT[d] for d in ROW_ORDER])
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlabel(X_LABEL, loc="left")    # left, so the note can take the right of the same line
    note(ax)

    paper_style.save(fig, stem)
    print(f"K = {k}: tile {side_x * col_in * 72:.1f} x {side_y * row_in * 72:.1f} pt, "
          f"column pitch {col_in * 72:.1f} pt, row pitch {row_in * 72:.1f} pt")
    for dev in ROW_ORDER:
        cells = ", ".join(f"{key[1][-2:]}={v:.1f} (n={n_query[key]})"
                          for key, v in sorted(recall.items()) if key[0] == dev)
        print(f"  {TABLE2_NAME_OF_SHORT[dev]:22s} enrolled {enrol_day[dev][-2:]}; {cells}")


if __name__ == "__main__":
    draw(K_ENROL, STEM)
