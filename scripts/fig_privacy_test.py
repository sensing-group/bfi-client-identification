#!/usr/bin/env python3
"""F10: the definition on the angles of He et al. real BFI, one column.

Every defense as one point or curve: observer accuracy at T = 1, K = 1, the
higher of the plain 1-NN and the "inv" 1-NN that works on what the defense
leaves unchanged, against the beamforming loss, -10 log10 of the first-column
gain the access point keeps (1.0 = continuous feedback); for keyed defenses the
"AP holds the secret" value. Whiskers are one standard deviation over seeds.

A defense hides the angles when the observer stays at chance (within
PASS_ACC_MARGIN_PP of 1/19) and the AP keeps at least PASS_GAIN_MIN of the gain,
the shaded corner. The report's SNR field is not in this dataset and not plotted,
so the corner is not the whole definition (Section 7.4). The axis starts at
0.05 dB so that the corner is wide enough to read.
The rotation fixed per device lies above the axis and is marked by an arrow. The
hollow circle on the chance line is the strongest noise decoded with a prior, the
gain of the best fixed beam learned on other positions (prior_decoder.py).
Every point comes from the JSON files and is asserted against EXPECTED below.

Manifold DP and BeamDancer come from their sweeps (plain 1-NN only; the noise
leaves no invariant to read).

Data (figures/):
  bfi_joint_defense_he_summary.json    cells "<defense>|<attacker>|T=<T>" -> mean, sd, n
  bfi_he_beamforming_gain_summary.json cells "<defense>|<who holds Q>" -> gain_col1, ...
  bfi_manifold_sweep_he_summary.json   cells "<defense>" -> nn1_T=<T>, gain_col1 (mean, sd, n)
  bfi_beamdancer_sweep_he_summary.json cells "b<bits>_R<R>" -> nn1_T=<T>, gain_col1 (mean, sd, n)
  bfi_prior_decoder_he_summary.json    cells "fixed" -> gain (mean, sd, n)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import paper_style as ps

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_STEM = "fig_privacy_test"
ATTACK_JSON = FIG_DIR / "bfi_joint_defense_he_summary.json"
GAIN_JSON = FIG_DIR / "bfi_he_beamforming_gain_summary.json"
SWEEP_JSON = FIG_DIR / "bfi_manifold_sweep_he_summary.json"
BEAMDANCER_JSON = FIG_DIR / "bfi_beamdancer_sweep_he_summary.json"
PRIOR_JSON = FIG_DIR / "bfi_prior_decoder_he_summary.json"
SWEEP = "sweep"                     # gain-cell key of the rows read from SWEEP_JSON
BEAMDANCER = "beamdancer"           # gain-cell key of the rows read from BEAMDANCER_JSON

T_DECISION = 1                      # single-window decision, K=1
ATTACKERS = ("nn1", "nn1_inv")      # the observer accuracy is the higher of the two
GAIN_FIELD = "gain_col1"
PASS_GAIN_MIN = 0.95
PASS_ACC_MARGIN_PP = 1.0            # "at chance" = within this of 1/19
RANDOM_BEAM_GAIN = 0.25             # 1 / N_t at N_t = 4
N_SEEDS = 10
EXPECTED_CHANCE_PCT = 5.3

# (defense key, gain-cell key), one list per mark
NONE = [("none", "none|none")]
DPSQ = [(f"dpsq_e{e}", f"dpsq_e{e}|none") for e in ("0.01", "0.5", "2.0")]
MANIFOLD = [(f"manifold_e{e}", SWEEP) for e in ("0.5", "2.0", "8.0", "32.0", "128.0", "512.0")]
BEAMDANCER_ROWS = [(f"b42_R{r}", BEAMDANCER) for r in (10, 25, 50, 75, 100)]
ROTATIONS = [("unitary_global", "unitary_global|AP holds Q"), ("unitary_pkt", "unitary_pkt|AP holds Q")]
FIXED = [("unitary_fixed", "unitary_fixed|AP holds Q")]
PASSING = [("unitary_pkt_sc", "unitary_pkt_sc|AP holds Q")]
KEYED = [("orthq_window", "orthq_window|AP holds Q (float feedback)")]
# Plotted values (accuracy %, gain) per defense; the script refuses to draw if the JSON disagrees.
EXPECTED = {
    "none": (39.0, 0.96), "dpsq_e0.01": (26.0, 0.87), "dpsq_e0.5": (26.1, 0.89),
    "dpsq_e2.0": (30.7, 0.94), "manifold_e0.5": (5.3, 0.24), "manifold_e2.0": (5.4, 0.24),
    "manifold_e8.0": (5.5, 0.24), "manifold_e32.0": (10.3, 0.38), "manifold_e128.0": (22.4, 0.84),
    "manifold_e512.0": (34.2, 0.95), "b42_R10": (36.7, 0.94), "b42_R25": (29.6, 0.87),
    "b42_R50": (19.7, 0.59), "b42_R75": (12.9, 0.41), "b42_R100": (10.0, 0.33),
    "unitary_fixed": (85.7, 0.97), "unitary_global": (34.8, 0.97), "unitary_pkt": (31.5, 0.97),
    "unitary_pkt_sc": (5.4, 0.97), "orthq_window": (16.0, 0.96),
}
EXPECTED_FIXED_BEAM_GAIN = 0.85
EXPECTED_PASSING = {"unitary_pkt_sc"}
EXPECTED_INVARIANT = {"unitary_pkt", "unitary_pkt_sc", "orthq_window"}
INVARIANT = "nn1_inv"
ACC_FMT, GAIN_FMT = "{:.1f}", "{:.2f}"

FIG_H_IN = 2.10
LAYOUT = dict(left=0.125, right=0.975, bottom=0.188, top=0.907)
X_LIM, Y_LIM = (0.05, 10.0), (0, 50)
X_TICKS, X_TICK_LABELS = (0.1, 0.3, 1, 3, 10), ("0.1", "0.3", "1", "3", "10")
Y_TICKS = (0, 10, 20, 30, 40)
X_LABEL, Y_LABEL = "Beamforming loss (dB)", "Observer accuracy on angles (%)"
TITLE = "(a) The angles, real feedback, 19 positions"

COL_NOISE, COL_ROT, COL_PASS = ps.OKABE_ITO["vermilion"], ps.OKABE_ITO["blue"], ps.OKABE_ITO["green"]
COL_REF = ps.OKABE_ITO["grey"]
TEXT_NOISE, TEXT_PASS, TEXT_REF = "#A34700", "#00694C", "#5E5E5E"   # the hues darkened to 4.5:1 on white
TEXT_ROT = COL_ROT
PASS_ALPHA = 0.22
REF_DASH = (0, (3, 3))
REF_LW = 0.6
WHISKER_LW = 0.5
WHISKER_ALPHA = 0.7
NO_DEFENSE_MS, NOISE_MS, TRIANGLE_MS, ROTATION_MS, STAR_MS = 4.5, 3.2, 4.5, 4.0, 9.0
ARROW_Y = (44.5, 49.0)              # the off-scale mark, data units
ARROW_LW = 0.8
GLYPH_X_RATIO = 1.12                 # a label's marker glyph sits this factor left of its text

# text anchors in data units: (x, y, ha, va)
LABEL_SHARED = "rotation shared\nor per report"
LABEL_FIXED = "rotation fixed per device: 86%"
LABEL_PRIOR = "decoder with a prior"
LABEL_PASS = "rotation per group"
LABELS = {
    "no defense": (0.135, 42.6, "right", "center"),
    "DP-SQ": (0.275, 21.6, "left", "center"),
    "Manifold DP": (0.95, 8.6, "left", "bottom"),
    "BeamDancer": (1.3, 27.5, "left", "bottom"),
    LABEL_SHARED: (0.054, 22.6, "left", "center"),
    "keyed transform": (0.2, 16.0, "left", "center"),
    LABEL_PASS: (0.054, 7.4, "left", "bottom"),
    LABEL_PRIOR: (0.8, 2.3, "left", "center"),
    "hidden": (0.054, 0.5, "left", "bottom"),
    "chance": (0.26, 2.3, "left", "center"),
    "random beam": (5.65, 45.0, "right", "top"),
    LABEL_FIXED: (0.17, 47.2, "left", "center"),
}
GLYPH_DPSQ = dict(marker="^", mfc="white", mew=1.0, ms=TRIANGLE_MS)
GLYPH_MANIFOLD = dict(marker="o", ms=NOISE_MS)
GLYPH_BEAMDANCER = dict(marker="D", mfc="white", mew=0.9, ms=NOISE_MS)
CORNER_EDGE_LW = 0.8


def loss_db(gain: float) -> float:
    return -10 * math.log10(gain)


def load() -> dict:
    att = json.loads(ATTACK_JSON.read_text())
    gain = json.loads(GAIN_JSON.read_text())["cells"]
    sweeps = {SWEEP: json.loads(SWEEP_JSON.read_text()), BEAMDANCER: json.loads(BEAMDANCER_JSON.read_text())}
    for s in sweeps.values():
        assert s["chance"] == att["chance"], (s["chance"], att["chance"])
    chance = att["chance"] * 100
    assert round(chance, 1) == EXPECTED_CHANCE_PCT, chance

    def accuracy(defense: str) -> tuple[float, float, str]:
        best = None
        for a in ATTACKERS:
            cell = att["cells"].get(f"{defense}|{a}|T={T_DECISION}")
            if cell is None:
                continue
            assert cell["n"] == N_SEEDS, (defense, a, cell["n"])
            if best is None or cell["mean"] > best[0]:
                best = (cell["mean"], cell["sd"], a)
        assert best is not None, defense
        return best[0] * 100, best[1] * 100, best[2]

    def point(key: str, gkey: str) -> dict:
        if gkey in sweeps:
            c = sweeps[gkey]["cells"][key]
            o = c[f"nn1_T={T_DECISION}"]
            assert o["n"] == N_SEEDS and c[GAIN_FIELD]["n"] == N_SEEDS, key
            acc, sd, who, g = o["mean"] * 100, o["sd"] * 100, "nn1", c[GAIN_FIELD]["mean"]
        else:
            (acc, sd, who), g = accuracy(key), gain[gkey][GAIN_FIELD]
        exp_acc, exp_gain = EXPECTED[key]
        assert ACC_FMT.format(acc) == ACC_FMT.format(exp_acc), (key, acc, exp_acc)
        assert GAIN_FMT.format(g) == GAIN_FMT.format(exp_gain), (key, g, exp_gain)
        return dict(key=key, acc=acc, sd=sd, attacker=who, gain=g, loss=loss_db(g),
                    passes=acc <= chance + PASS_ACC_MARGIN_PP and g >= PASS_GAIN_MIN)

    marks = {name: [point(k, g) for k, g in rows] for name, rows in dict(
        none=NONE, dpsq=DPSQ, manifold=MANIFOLD, beamdancer=BEAMDANCER_ROWS, rotations=ROTATIONS,
        fixed=FIXED, passing=PASSING, keyed=KEYED).items()}
    points = [p for rows in marks.values() for p in rows]
    assert {p["key"] for p in points} == set(EXPECTED)
    assert {p["key"] for p in points if p["passes"]} == EXPECTED_PASSING
    assert {p["key"] for p in points if p["attacker"] == INVARIANT} == EXPECTED_INVARIANT
    assert all(p["acc"] < Y_LIM[1] for rows in marks.values() if rows is not marks["fixed"] for p in rows)
    fixed_beam = json.loads(PRIOR_JSON.read_text())["cells"]["fixed"]["gain"]
    assert fixed_beam["n"] == N_SEEDS
    assert GAIN_FMT.format(fixed_beam["mean"]) == GAIN_FMT.format(EXPECTED_FIXED_BEAM_GAIN), fixed_beam
    return dict(chance=chance, marks=marks, points=points, fixed_beam_gain=fixed_beam["mean"])


def plot(ax, rows, fmt, color, zorder, **kw) -> None:
    xs, ys = [p["loss"] for p in rows], [p["acc"] for p in rows]
    ax.errorbar(xs, ys, yerr=[p["sd"] for p in rows], fmt="none", ecolor=color, elinewidth=WHISKER_LW,
                alpha=WHISKER_ALPHA, zorder=zorder - 1)
    ax.plot(xs, ys, fmt, color=color, zorder=zorder, **kw)


def label(ax, text: str, color, size=ps.FONT_PT, glyph=None, **kw) -> None:
    """Direct label; with `glyph`, the mark it names is repeated left of the text."""
    x, y, ha, va = LABELS[text]
    t = ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, zorder=8, **kw)
    if glyph:
        box = t.get_window_extent(ax.figure.canvas.get_renderer())
        y_mid = ax.transData.inverted().transform((box.x0, (box.y0 + box.y1) / 2))[1]
        ax.plot([x / GLYPH_X_RATIO], [y_mid], linestyle="none", color=COL_NOISE, zorder=8, **glyph)


def main() -> None:
    ps.apply()
    plt.rcParams["axes.grid"] = False
    d = load()
    m = d["marks"]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN))
    fig.subplots_adjust(**LAYOUT)
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    # the corner where the angles are hidden and the two references
    corner = ((X_LIM[0], 0), loss_db(PASS_GAIN_MIN) - X_LIM[0], d["chance"] + PASS_ACC_MARGIN_PP)
    ax.add_patch(Rectangle(*corner, color=COL_PASS, alpha=PASS_ALPHA, lw=0, zorder=1))
    ax.add_patch(Rectangle(*corner, facecolor="none", edgecolor=COL_PASS, lw=CORNER_EDGE_LW, zorder=3))
    ax.axhline(d["chance"], color=COL_REF, ls=REF_DASH, lw=REF_LW, zorder=2)
    ax.axvline(loss_db(RANDOM_BEAM_GAIN), color=COL_REF, ls=REF_DASH, lw=REF_LW, zorder=2)
    label(ax, "chance", TEXT_REF, ps.SMALL_PT)
    label(ax, "random beam", TEXT_REF, ps.SMALL_PT)
    label(ax, "hidden", TEXT_PASS, ps.SMALL_PT)

    plot(ax, m["manifold"], "-o", COL_NOISE, 4, ms=NOISE_MS)
    plot(ax, m["beamdancer"], "--D", COL_NOISE, 4, mfc="white", mew=0.9, ms=NOISE_MS, lw=0.9)
    plot(ax, m["dpsq"], "^", COL_NOISE, 4, mfc="white", mew=1.0, ms=TRIANGLE_MS)
    plot(ax, m["none"], "o", "black", 5, ms=NO_DEFENSE_MS)
    plot(ax, m["rotations"], "s", COL_ROT, 5, ms=ROTATION_MS)
    plot(ax, m["keyed"], "s", COL_ROT, 5, mfc="white", mew=0.9, ms=ROTATION_MS)
    plot(ax, m["passing"], "*", COL_PASS, 6, ms=STAR_MS, mec="black", mew=0.4)
    ax.plot([loss_db(d["fixed_beam_gain"])], [m["manifold"][0]["acc"]], "o", color=COL_NOISE, mfc="white",
            mew=1.0, ms=NO_DEFENSE_MS, zorder=6)
    label(ax, LABEL_PRIOR, TEXT_NOISE, ps.SMALL_PT)
    x_fixed = m["fixed"][0]["loss"]
    ax.annotate("", (x_fixed, ARROW_Y[1]), (x_fixed, ARROW_Y[0]), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=COL_ROT, lw=ARROW_LW, shrinkA=0, shrinkB=0))

    label(ax, "no defense", "black")
    label(ax, "DP-SQ", TEXT_NOISE, glyph=GLYPH_DPSQ)
    label(ax, "Manifold DP", TEXT_NOISE, glyph=GLYPH_MANIFOLD)
    label(ax, "BeamDancer", TEXT_NOISE, glyph=GLYPH_BEAMDANCER)
    label(ax, LABEL_SHARED, TEXT_ROT, linespacing=0.95)
    label(ax, "keyed transform", TEXT_ROT)
    label(ax, LABEL_FIXED, TEXT_ROT, ps.SMALL_PT)
    label(ax, LABEL_PASS, TEXT_PASS, fontweight="bold")

    ax.set_xticks(list(X_TICKS))
    ax.set_xticklabels(list(X_TICK_LABELS))
    ax.set_yticks(list(Y_TICKS))
    ax.minorticks_off()
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE, loc="left")
    ps.save(fig, OUT_STEM)

    for p in d["points"]:
        print(f"{p['key']:18s} acc {ACC_FMT.format(p['acc']):>5s}% (sd {p['sd']:.1f}, {p['attacker']})"
              f"  gain {GAIN_FMT.format(p['gain'])}  loss {p['loss']:.2f} dB  {'PASS' if p['passes'] else 'fail'}")
    print(f"chance {d['chance']:.2f}%; wrote {FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
