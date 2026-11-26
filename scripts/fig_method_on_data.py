#!/usr/bin/env python3
"""F4b: the method on one real window, four panels at text width.

(a) Captured frame: one Compressed Beamforming Action frame of the window
    as a strip of fields (MAC header, category, action, HE MIMO control,
    SNR, report), with callouts that expand the MIMO control field (Nc, Nr,
    bandwidth, Ng, codebook, angle bit depths) and the report (tones x bits
    per tone = report bytes), all decoded from the frame.
(b) Decoded angles: the window's 32 frames x (4 angles x 32 tone bins),
    phi on a cyclic map over [0, 2 pi), psi on a sequential map over
    [0, pi/2].
(c) Window summary: the 320-d feature vector of r_common.window_feature,
    drawn as the per-bin mean angle of each of the four angles with a
    +-1 SD band (circular SD sqrt(-2 ln R) for phi, where 1 - R is the
    stored spread; plain SD for psi).
(d) Decision: Euclidean distance in the z-scored feature space from this
    window to the single enrolled window (K = 1) of each of the six devices,
    the r_eval distance; nearest wins; chance 1/6 per decision.

The episode is one draw of the r_eval "other_day_any" protocol (enrol one
window on some other day, recognise this window): the first episode of
r_eval.episodes with the Apple TV (living room) as the queried device at
EPISODE_SEED, one random window from each enrolled block and one random
window from the query block. The 32 raw bodies of the query window are
read from the archive day file (read-only), decoded with r_common
(validated decoder angle_decode.py) and asserted equal to the cached
feature vector; the distances are asserted against EXPECT_DIST, the chance
against r_main.csv.

Memory: the window cache (about 70 MB) and one 32-row archive query;
single process; peak RSS logged.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps      # noqa: E402
from device_names import TABLE2_NAME_OF_SHORT  # noqa: E402
import r_common as R          # noqa: E402
import r_eval as E            # noqa: E402
import xres_common as X       # noqa: E402

OUT_STEM = "fig_method_on_data"
OUT_JSON = ps.FIG_DIR / f"{OUT_STEM}.json"
R_MAIN_CSV = ps.FIG_DIR / "r_main.csv"

# Episode -------------------------------------------------------------------
QUERY_DEVICE = "AppleTV-LR"
CONDITION = "other_day_any"
EPISODE_SEED = 0
K_ENROLLED = 1
# Expected values (from the cache; the script refuses to draw if they move)
EXPECT_GEOM = ["HE", 2, 3, 4, 2, 80, 4, "SU", 0]
EXPECT_BODY_LEN = 572
EXPECT_QUERY_WINDOW = 292
EXPECT_DIST = {"AppleTV-LR": 19.59, "AppleTV-BR": 25.47, "LGTV": 26.61,
               "Laptop": 29.10, "SamsungTV": 39.62, "Vacuum": 134.12}
DIST_TOL = 0.01
EXPECT_CHANCE = 1.0 / 6.0
EXPECT_GALLERY_HIST = {"6": 1600}
TOL_CHANCE = 1e-9

# Frame layout (802.11ax HE Compressed Beamforming/CQI action body) ---------
HEX_BYTES_SHOWN = 2 + X.AD.HE_MIMO_CTRL_LEN  # category, action, MIMO control (JSON sidecar)
N_HEADER_BYTES = 2 + X.AD.HE_MIMO_CTRL_LEN  # category, action, MIMO control
# Panel (a) is drawn in centimetres: the frame as a strip of fields, with two
# callouts that expand the MIMO control field and the report.
A_W_CM = 3.84                             # panel width at text width
EDGE_PAD_CM = 0.06                        # keeps the right-hand borders inside the axes
A_BOTTOM = 0.04                           # figure fraction; (a) also uses the legend row
FIELDS = (  # (key, strip label, strip width in cm, fill)
    ("mac", "MAC", 0.62, "#EEEEEE"), ("cat", None, 0.66, "#EEEEEE"),
    ("act", None, 0.3, "#EEEEEE"), ("mimo", "MIMO", 0.8, "#D6EAF5"),
    ("snr", "SNR", 0.58, "#EEEEEE"), ("rep", "report", 0.79, "#FCEBD2"))
FIELD_EDGE = "#8C8C8C"
MIMO_EDGE, REP_EDGE = "#6BAED6", "#E6A35C"   # callout borders, darker than the field tints
STRIP_Y, STRIP_H = 3.72, 0.46             # cm
DIRECTION_Y = 4.36                        # "client -> AP" above the MAC header
CALLOUT_MIMO = (0.0, 2.05, 2.3, 1.38)     # x, y, w, h in cm
CALLOUT_REP = (0.44, 0.6, 3.31, 1.0)
REP_WEDGE_GAP = 0.15                      # cm between the MIMO callout and the report wedge
CALLOUT_PAD = 0.12                        # cm, text inset
CALLOUT_LINE_CM = 0.33                    # line pitch at 8 pt
FRAME_PT = 7.5                            # field labels; nothing in (a) below this
MFP_NOTE = "plaintext, outside MFP"

# Figure geometry ----------------------------------------------------------
FIG_H_IN = 2.2
AX_BOTTOM, AX_TOP = 0.33, 0.87
PANEL_X = {  # (left, right) in figure fraction
    "a": (0.012, 0.228), "b": (0.322, 0.458), "c": (0.606, 0.742),
    "d": (0.882, 0.975)}
CBAR_X = 0.466                            # one column: phi above psi
CBAR_W = 0.011
CBAR_GAP = 0.34                           # of the axes height, between the two bars
CBAR_TITLE_PAD_PT = 6                     # colorbar title clears its 2 pi tick label
CBAR_TITLE_PAD_PSI_PT = 6                 # psi title clears its pi/2 tick label
ARROW_GAPS = (("a", "b"), ("b", "c"), ("c", "d"))
ARROW_PAD = 0.006
MIN_ARROW_W = 0.012                       # figure fraction; shorter arrows are not drawn legibly
PAPER_FONT = "Linux Libertine O"          # the layout check holds only in the paper's font
LEGEND_Y = 0.16
ARROW_COLOR = ps.OKABE_ITO["grey"]
PHI_CMAP, PSI_CMAP = "twilight", "viridis"
PHI_MAX, PSI_MAX = 2 * np.pi, np.pi / 2
ANGLE_TEX = {"phi11": r"$\phi_{11}$", "phi21": r"$\phi_{21}$",
             "psi21": r"$\psi_{21}$", "psi31": r"$\psi_{31}$"}
ANGLE_COLOR = {"phi11": ps.OKABE_ITO["blue"], "phi21": ps.OKABE_ITO["vermilion"],
               "psi21": ps.OKABE_ITO["green"], "psi31": ps.OKABE_ITO["purple"]}
ANGLE_STYLE = {"phi11": "-", "phi21": "--", "psi21": ":", "psi31": "-."}  # greyscale print
BAND_ALPHA = 0.22
WRAP_BREAK_RAD = np.pi                    # break a phi line at a wrap jump
DIST_XMAX = 48.0                          # the vacuum bar is clipped and labelled
BREAK_X = (31.0, 33.5)                    # the clipped bar's break mark, data units
BREAK_DX, BREAK_DY, BREAK_LW = 0.9, 0.42, 1.6
DIST_XTICKS = (0, 20, 40)
BAR_TRUE = ps.OKABE_ITO["blue"]
BAR_OTHER = "#BBBBBB"
TEXT_ON_DARK, TEXT_ON_LIGHT = "white", "#1A1A1A"   # a label inside a bar follows the bar's shade
BAR_LABEL_DX = 0.8
TITLE_D_X = -0.72                         # (d) title starts above the device names
DEVICE_LABEL = TABLE2_NAME_OF_SHORT        # short key -> device name of Table 2


def utc(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def draw_episode(tb: E.Table, rng: np.random.Generator):
    """First r_eval episode with QUERY_DEVICE as the queried device, K=1."""
    m_q = tb.names.index(QUERY_DEVICE)
    for G, m, qb, _ in E.episodes(tb, CONDITION, rng, 1):
        if m != m_q:
            continue
        enrol = {g: int(rng.choice(tb.blocks[G[g]])) for g in G}
        qi = int(rng.choice(tb.blocks[qb]))
        return G, qb, enrol, qi
    raise RuntimeError("no episode for " + QUERY_DEVICE)


def load() -> dict:
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    tb = E.Table(w)
    G, qb, enrol, qi = draw_episode(tb, np.random.default_rng(EPISODE_SEED))
    assert qi == EXPECT_QUERY_WINDOW, qi
    assert len(G) == len(tb.names) == 6
    dist = {tb.names[g]: float(np.linalg.norm(tb.Z[enrol[g]] - tb.Z[qi]))
            for g in G}
    for k, v in EXPECT_DIST.items():
        assert abs(dist[k] - v) <= DIST_TOL, (k, dist[k])
    nearest = min(dist, key=dist.get)
    assert nearest == QUERY_DEVICE, nearest

    # chance per decision from the r_eval output
    import csv
    with open(R_MAIN_CSV) as f:
        rows = [r for r in csv.DictReader(f)
                if r["condition"] == CONDITION and r["feature"] == "angles"
                and r["K"] == str(K_ENROLLED) and r["T"] == "1"
                and r["gap_s"] == str(int(R.SESSION_GAP_S))]
    assert len(rows) == 1, rows
    assert abs(float(rows[0]["chance"]) - EXPECT_CHANCE) < TOL_CHANCE
    assert json.loads(rows[0]["gallery_size_hist"]) == EXPECT_GALLERY_HIST
    recall_row = rows[0]

    # raw bodies of the query window (read-only archive query)
    day = str(w["day"][qi])
    mac = devs[tb.names.index(QUERY_DEVICE)]
    c = X.connect_ro(X.ARCHIVE_DIR / f"bfi_{day}.sqlite")
    rows = c.execute(f"SELECT ts, body_hex FROM bfi WHERE {X.SRC_COL}=? AND ts>=? "
                     "AND ts<=? ORDER BY ts",
                     (mac, float(w["ts0"][qi]), float(w["ts1"][qi]))).fetchall()
    c.close()
    assert len(rows) == R.WINDOW_FRAMES, len(rows)
    bodies = [bytes.fromhex(r[1]) for r in rows]
    assert {len(b) for b in bodies} == {EXPECT_BODY_LEN}
    geom = X.AD.parse_geometry(bodies[0])
    assert [geom[k] for k in R.GEOM_FIELDS] == EXPECT_GEOM, geom
    assert json.loads(w["geom_table"][w["geom"][qi]]) == EXPECT_GEOM
    mimo = X.AD.parse_he_mimo_ctrl(bodies[0][2:2 + X.AD.HE_MIMO_CTRL_LEN])
    rad, _q, _rq = R.decode_first_column(bodies, geom)
    fb = R.bin_frames(rad)
    feat = R.window_feature(fb)
    assert np.allclose(feat, w["feat"][qi], atol=1e-5)
    n_sub = rad.shape[2]
    bits_per_sc = int(X.AD._layout(geom["nr"], geom["nc"], geom["bphi"],
                                   geom["bpsi"])[3])
    assert N_HEADER_BYTES + geom["nc"] + (n_sub * bits_per_sc + 7) // 8 \
        == EXPECT_BODY_LEN

    info = dict(
        # the released pseudonymized id; a raw address is never written out
        query=dict(device=QUERY_DEVICE, mac=mac if str(mac).startswith("dev_") else None,
                   window=qi, day=day,
                   session=int(w["session"][qi]), ts0=float(w["ts0"][qi]),
                   ts1=float(w["ts1"][qi]), utc0=utc(w["ts0"][qi]),
                   utc1=utc(w["ts1"][qi]), frames=len(bodies)),
        enrolled={tb.names[g]: dict(window=enrol[g], day=G[g][2],
                                    session=G[g][1],
                                    utc0=utc(w["ts0"][enrol[g]]))
                  for g in G},
        frame=dict(category=bodies[0][0], action=bodies[0][1], **mimo,
                   bphi=geom["bphi"], bpsi=geom["bpsi"], body_bytes=len(bodies[0]),
                   n_sub=n_sub, bits_per_tone=bits_per_sc,
                   hex_head=bodies[0][:HEX_BYTES_SHOWN].hex(" ")),
        distance=dist, nearest=nearest, chance=EXPECT_CHANCE,
        protocol=dict(condition=CONDITION, K=K_ENROLLED, T=1,
                      accuracy_all_devices=float(recall_row["accuracy"]),
                      n_decisions=int(recall_row["n_decisions"])),
        seed=EPISODE_SEED)
    del tb, w
    return dict(info=info, bodies=bodies, geom=geom, mimo=mimo, rad=rad,
                fb=fb, feat=feat, dist=dist)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_frame(ax, d: dict) -> None:
    """The frame as a strip of fields; callouts expand MIMO control and the report."""
    ax.set_axis_off()
    x0, x1, y0, y1 = -EDGE_PAD_CM / 2, A_W_CM, 0, A_W_CM * ax.bbox.height / ax.bbox.width
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    body = d["bodies"][0]
    g, m, fr = d["geom"], d["mimo"], d["info"]["frame"]
    n_rep = len(body) - N_HEADER_BYTES - g["nc"]
    label = {"cat": f"0x{body[0]:02X}", "act": f"{body[1]}"}
    pos, x = {}, 0.0
    for key, short, w_, fill in FIELDS:
        ax.add_patch(Rectangle((x, STRIP_Y), w_, STRIP_H, facecolor=fill,
                               edgecolor=FIELD_EDGE, linewidth=0.6))
        ax.text(x + w_ / 2, STRIP_Y + STRIP_H / 2, short or label[key], ha="center",
                va="center", fontsize=FRAME_PT)
        pos[key] = (x, x + w_)
        x += w_
    ax.text(pos["mac"][0], DIRECTION_Y, "client \u2192 AP", ha="left", va="center",
            fontsize=FRAME_PT, color=ps.OKABE_ITO["grey"])
    ax.text(pos["rep"][1], DIRECTION_Y, f"body {len(body)} B", ha="right", va="center",
            fontsize=FRAME_PT, color=ps.OKABE_ITO["grey"])

    def callout(field, box, edge, wedge, lines):
        bx, by, bw, bh = box
        fx0, fx1 = pos[field]
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=edge, linewidth=0.8))
        for xa, xb in zip((fx0, fx1), wedge):
            ax.plot([xa, xb], [STRIP_Y, by + bh], color=edge, linewidth=0.8)
        for i, s_ in enumerate(lines):
            ax.text(bx + CALLOUT_PAD, by + bh - CALLOUT_PAD - (i + 0.5) * CALLOUT_LINE_CM, s_,
                    ha="left", va="center", fontsize=ps.FONT_PT)

    callout("mimo", CALLOUT_MIMO, MIMO_EDGE,
            (CALLOUT_MIMO[0], CALLOUT_MIMO[0] + CALLOUT_MIMO[2]), [
        f"$N_c$ = {m['nc']}, $N_t$ = {m['nr']}",       # 802.11's Nr: rows of V, the AP's antennas (N_t of Section 2)
        f"{m['bw_mhz']} MHz, $N_g$ = {m['ng']}",
        f"codebook {m['codebook']}",
        f"$\\phi$ {g['bphi']} bit, $\\psi$ {g['bpsi']} bit"])
    callout("rep", CALLOUT_REP, REP_EDGE,
            (CALLOUT_MIMO[0] + CALLOUT_MIMO[2] + REP_WEDGE_GAP, CALLOUT_REP[0] + CALLOUT_REP[2]), [
        f"{fr['n_sub']} tones \u00d7 {fr['bits_per_tone']} bit = {n_rep} B",
        MFP_NOTE])
    ax.set_title("(a) Captured frame", loc="left")


def angle_matrix(fb: np.ndarray) -> np.ndarray:
    """(T, 6, B) per-frame bins -> (T, 4B): phi11, phi21 as circular bin
    means in [0, 2 pi), psi21, psi31 as bin means."""
    phi11 = np.mod(np.arctan2(fb[:, 1], fb[:, 0]), PHI_MAX)
    phi21 = np.mod(np.arctan2(fb[:, 3], fb[:, 2]), PHI_MAX)
    return np.concatenate([phi11, phi21, fb[:, 4], fb[:, 5]], axis=1)


def panel_angles(ax, fig, d: dict) -> None:
    M = angle_matrix(d["fb"])
    T, B = M.shape[0], R.N_SC_BINS
    ax.grid(False)
    im_phi = ax.imshow(M[:, :2 * B], cmap=PHI_CMAP, vmin=0, vmax=PHI_MAX,
                       extent=(0, 2 * B, T, 0), aspect="auto", interpolation="nearest")
    im_psi = ax.imshow(M[:, 2 * B:], cmap=PSI_CMAP, vmin=0, vmax=PSI_MAX,
                       extent=(2 * B, 4 * B, T, 0), aspect="auto", interpolation="nearest")
    for k in (1, 2, 3):
        ax.axvline(k * B, color="white", linewidth=0.8)
    ax.set_xlim(0, 4 * B)
    ax.set_ylim(T, 0)
    ax.set_xticks([B / 2 + k * B for k in range(4)])
    ax.set_xticklabels([ANGLE_TEX[a] for a in R.ANGLE_NAMES])
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.set_xlabel(f"tone bin ({B} per angle)")
    ax.set_ylabel("frame")
    ax.set_title("(b) Decoded angles", loc="left")
    pos = ax.get_position()
    h = pos.height * (1 - CBAR_GAP) / 2
    for y0, im, ticks, labels, name, title_pad in (
            (pos.y0 + h + CBAR_GAP * pos.height, im_phi, [0, np.pi, PHI_MAX],
             ["0", r"$\pi$", r"$2\pi$"], r"$\phi$ (rad)", CBAR_TITLE_PAD_PT),
            (pos.y0, im_psi, [0, PSI_MAX / 2, PSI_MAX],
             ["0", r"$\pi/4$", r"$\pi/2$"], r"$\psi$ (rad)", CBAR_TITLE_PAD_PSI_PT)):
        cax = fig.add_axes([CBAR_X, y0, CBAR_W, h])
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.ax.set_yticklabels(labels)
        cb.ax.tick_params(length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)
        cax.set_title(name, fontsize=ps.SMALL_PT, pad=title_pad, loc="left")


def summary_curves(feat: np.ndarray) -> dict:
    """320-d vector -> per angle (mean angle, sd) over bins."""
    B = R.N_SC_BINS
    out = {}
    k = 0
    for a in R.ANGLE_NAMES[:2]:
        c, s, spread = (feat[k:k + B], feat[k + B:k + 2 * B], feat[k + 2 * B:k + 3 * B])
        k += 3 * B
        mean = np.mod(np.arctan2(s, c), PHI_MAX)
        r = np.clip(1.0 - spread, 1e-12, 1.0)
        out[a] = (mean, np.sqrt(-2.0 * np.log(r)))
    for a in R.ANGLE_NAMES[2:]:
        out[a] = (feat[k:k + B].copy(), feat[k + B:k + 2 * B].copy())
        k += 2 * B
    assert k == feat.size
    return out


def panel_summary(ax, d: dict) -> list:
    curves = summary_curves(d["feat"])
    x = np.arange(R.N_SC_BINS)
    handles = []
    for a in R.ANGLE_NAMES:
        mean, sd = curves[a]
        y = mean.astype(float).copy()
        if a.startswith("phi"):            # break the line where the mean wraps
            jump = np.abs(np.diff(y)) > WRAP_BREAK_RAD
            y_plot = y.copy()
            seg_x, seg_y, seg_lo, seg_hi = [], [], [], []
            start = 0
            for j in np.flatnonzero(jump) + 1:
                seg_x.append(x[start:j]); seg_y.append(y_plot[start:j])
                seg_lo.append((y - sd)[start:j]); seg_hi.append((y + sd)[start:j])
                start = j
            seg_x.append(x[start:]); seg_y.append(y_plot[start:])
            seg_lo.append((y - sd)[start:]); seg_hi.append((y + sd)[start:])
        else:
            seg_x, seg_y = [x], [y]
            seg_lo, seg_hi = [y - sd], [y + sd]
        col = ANGLE_COLOR[a]
        for sx, sy, lo, hi in zip(seg_x, seg_y, seg_lo, seg_hi):
            ax.fill_between(sx, lo, hi, color=col, alpha=BAND_ALPHA, linewidth=0, zorder=1)
            ax.plot(sx, sy, color=col, linestyle=ANGLE_STYLE[a], linewidth=1.0, zorder=3)
        handles.append(plt.Line2D([], [], color=col, linestyle=ANGLE_STYLE[a],
                                  linewidth=1.0, label=ANGLE_TEX[a]))
    handles.append(Patch(facecolor="black", alpha=BAND_ALPHA, linewidth=0,
                         label=r"$\pm$1 SD (circular for $\phi$)"))
    ax.set_xlim(0, R.N_SC_BINS - 1)
    ax.set_ylim(0, PHI_MAX)
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, PHI_MAX])
    ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_xlabel("tone bin")
    ax.set_ylabel("mean angle (rad)")
    ax.set_title("(c) Window summary", loc="left")
    return handles


def panel_decision(ax, d: dict) -> None:
    dist = d["dist"]
    order = sorted(dist, key=dist.get)           # nearest first
    y = np.arange(len(order))
    vals = np.array([dist[k] for k in order])
    cols = [BAR_TRUE if k == QUERY_DEVICE else BAR_OTHER for k in order]
    ax.barh(y, np.minimum(vals, DIST_XMAX), color=cols, height=0.62, zorder=3)
    for yi, v, col in zip(y, vals, cols):
        if v <= DIST_XMAX:
            ax.text(v + BAR_LABEL_DX, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=ps.SMALL_PT)
        else:                                  # clipped bar: a break mark, the value inside it
            for xb in BREAK_X:
                ax.plot([xb - BREAK_DX, xb + BREAK_DX], [yi + BREAK_DY, yi - BREAK_DY],
                        color="white", linewidth=BREAK_LW, zorder=4, solid_capstyle="butt")
            ax.text(DIST_XMAX - BAR_LABEL_DX, yi, f"{v:.0f}", va="center", ha="right",
                    fontsize=ps.SMALL_PT,
                    color=TEXT_ON_DARK if col == BAR_TRUE else TEXT_ON_LIGHT)
    ax.set_yticks(y)
    ax.set_yticklabels([DEVICE_LABEL[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlim(0, DIST_XMAX)
    ax.set_xticks(list(DIST_XTICKS))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("distance (z units)")
    ax.set_title("(d) Decision, chance 1/6", loc="left", x=TITLE_D_X)


# ---------------------------------------------------------------------------

def paper_font_present() -> bool:
    try:
        font_manager.findfont(PAPER_FONT, fallback_to_default=False)
        return True
    except ValueError:
        return False

def main() -> None:
    X.set_vm_limit()
    ps.apply()
    d = load()
    fig = plt.figure(figsize=(ps.TEXT_W_IN, FIG_H_IN))
    axes = {k: fig.add_axes([x0, A_BOTTOM if k == "a" else AX_BOTTOM, x1 - x0,
                             AX_TOP - (A_BOTTOM if k == "a" else AX_BOTTOM)])
            for k, (x0, x1) in PANEL_X.items()}
    panel_frame(axes["a"], d)
    panel_angles(axes["b"], fig, d)
    handles = panel_summary(axes["c"], d)
    panel_decision(axes["d"], d)
    # Arrows span the white space between panels as drawn: from the rightmost
    # ink of the panel on the left (colorbars included) to the leftmost ink of
    # the next panel (tick labels and y-label included).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    ink = [ax.get_tightbbox(renderer).transformed(to_fig) for ax in fig.axes]
    ym = (AX_BOTTOM + AX_TOP) / 2
    spans = {}
    for a, b in ARROW_GAPS:
        left_of_b = [bb for bb in ink if (bb.x0 + bb.x1) / 2 < PANEL_X[b][0]]
        x0 = max(bb.x1 for bb in left_of_b) + ARROW_PAD
        # Leftmost ink of panel b at the arrow's height: its spine, the y-label
        # and the tick labels whose vertical extent crosses the arrow.
        texts = [axes[b].yaxis.label] + list(axes[b].get_yticklabels())
        crossing = [t.get_window_extent(renderer).transformed(to_fig) for t in texts
                    if t.get_visible() and t.get_text()]
        crossing = [bb.x0 for bb in crossing if bb.y0 - ARROW_PAD <= ym <= bb.y1 + ARROW_PAD]
        x1 = min([PANEL_X[b][0]] + crossing) - ARROW_PAD
        if x1 - x0 < MIN_ARROW_W:
            if paper_font_present():
                raise SystemExit(f"no room for the {a}->{b} arrow ({x1 - x0:.3f} of the width)")
            # Without the paper font the fallback's wider text leaves no gap;
            # draw the panels and omit the arrows rather than fail.
            print(f"[fig_method_on_data] {PAPER_FONT} not installed: no room for the "
                  f"{a}->{b} arrow, arrows omitted", file=sys.stderr)
            spans = {}
            break
        spans[(a, b)] = (x0, x1)
    # All arrows share the length of the shortest gap and start just right of
    # the panel they leave, so none reads as pointing at one row of the next.
    length = min((x1 - x0 for x0, x1 in spans.values()), default=0.0)
    for a, b in (ARROW_GAPS if spans else ()):
        x0 = spans[(a, b)][0]
        x1 = x0 + length
        fig.add_artist(FancyArrowPatch((x0, ym), (x1, ym), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7,
                                       color=ARROW_COLOR, linewidth=0.9))
    cx = (PANEL_X["c"][0] + PANEL_X["c"][1]) / 2
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(cx, LEGEND_Y),
               ncol=3, columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    ps.save(fig, OUT_STEM)
    X.write_json(OUT_JSON, d["info"])
    q = d["info"]["query"]
    print(f"query window {q['window']}: {q['device']} {q['mac']} day {q['day']} "
          f"session {q['session']} {q['utc0']} -> {q['utc1']} UTC, {q['frames']} frames")
    print("frame:", d["info"]["frame"])
    print("distances:", {k: round(v, 2) for k, v in sorted(d["dist"].items(), key=lambda x: x[1])})
    print("nearest:", d["info"]["nearest"], "chance", round(EXPECT_CHANCE, 4),
          "protocol accuracy (all devices, K=1, T=1)",
          d["info"]["protocol"]["accuracy_all_devices"])
    X.log_rss("figure")
    print(f"wrote {ps.FIG_DIR / OUT_STEM}.pdf/.png")


if __name__ == "__main__":
    main()
