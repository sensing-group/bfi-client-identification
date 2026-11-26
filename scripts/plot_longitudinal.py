"""Hour-of-day footprints of the six labeled devices (paper Section 6.5, Appendix B).

(a) Hour-of-day heatmap over the capture's days with data (device x local
    hour, log-scaled frame counts).
(b) Minute-resolution timeline of one day; devices first heard that day are
    annotated with the time of their first frame and their days with data,
    both computed from the archive.
(c) Burst rate (one over the median gap between consecutive frames shorter
    than BURST_GAP_S) against the standard deviation of RSSI, coloured by
    mean RSSI.
Lanes and points carry the device names of Table 2.

Source: the daily SQLite files of the Pi capture in BFI_ARCHIVE_DIR over
the paper's window (xres_common.day_paths), raw or released schema. Saves
figures/fig_longitudinal.pdf.
"""
from __future__ import annotations

import datetime
import pathlib
import sqlite3
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import paper_style as ps  # noqa: E402
import xres_common as X  # noqa: E402  (archive path, device table)
import r_common as R  # noqa: E402
from device_names import TABLE2_NAME  # noqa: E402  (device names of Table 2)

UTC_TO_LOCAL_OFFSET_HOURS = 2   # CEST in May
TIMELINE_DAY = "2026-05-15"
OUT_PDF = X.FIG_DIR / "fig_longitudinal.pdf"
# Lane order of the panels, by paper label (X.DEVICE_GT values).
ORDER_PREF = ("Apple TV (living room)", "Apple TV (bedroom)", "LG TV", "Laptop",
              "iRobot vacuum", "Samsung TV")
BURST_GAP_S = 60.0            # gaps longer than this end a burst (panel c)
FIG_H_IN = 3.40
GRID = dict(height_ratios=[1, 1.05], hspace=0.55, wspace=0.40, width_ratios=[1.4, 1],
            left=0.112, right=0.938, top=0.943, bottom=0.107)
CALLOUT_COLOR = ps.OKABE_ITO["blue"]
BURST_TICKS, BURST_TICK_LABELS = (1, 10), ("1", "10")
# (b) annotation boxes in data coordinates (local hour, lane y), by paper label: each
# sits on its own lane, in the blank stretch before that device is heard, and its
# arrow runs along the still empty lane to the first frame.
FIRST_SEEN_TEXT_POS = {"Apple TV (bedroom)": (6.6, 4.5), "Samsung TV": (6.4, 0.5)}
NEEDED_COLS = {"ts", X.SRC_COL, X.RSSI_COL}


def device_label(addr: str) -> str:
    """Tick label: the device name of Table 2 (same on the raw and the released archive)."""
    return TABLE2_NAME[R.label(addr)[0]]


def load_frames(with_rssi: bool = False) -> list[tuple]:
    """(ts, address[, rssi]) of every frame of the paper's days."""
    cols = f"ts, {X.SRC_COL}" + (f", {X.RSSI_COL}" if with_rssi else "")
    frames = []
    for db in X.day_paths():
        try:
            c = X.connect_ro(db)
            have = {r[1] for r in c.execute("PRAGMA table_info(bfi)")}
            if not NEEDED_COLS <= have:
                print(f"  skip {db.name}: columns {sorted(have)}", flush=True)
                continue
            frames.extend(c.execute(f"SELECT {cols} FROM bfi"))
            c.close()
        except sqlite3.DatabaseError as e:
            print(f"  skip {db}: {e}", flush=True)
    return frames


def panel_a_heatmap(ax, frames, top_macs):
    """Hour-of-day x device heatmap (log frame counts)."""
    matrix = np.zeros((len(top_macs), 24), dtype=int)
    for ts, mac in frames:
        if mac not in top_macs:
            continue
        h_utc = datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc).hour
        h_local = (h_utc + UTC_TO_LOCAL_OFFSET_HOURS) % 24
        i = top_macs.index(mac)
        matrix[i, h_local] += 1

    M = matrix.astype(float)
    M_log = np.log10(M + 1)
    im = ax.imshow(
        M_log, aspect="auto", cmap="YlOrRd",
        norm=mcolors.Normalize(vmin=0, vmax=M_log.max()),
        interpolation="nearest",
    )
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)])
    ax.set_xlabel("Local hour (CEST)")
    ax.set_yticks(range(len(top_macs)))
    ax.set_yticklabels([device_label(m) for m in top_macs])
    ax.set_title("(a) Frames per hour of day, all days with data", loc="left")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_{10}(\mathrm{frames}+1)$")
    cbar.outline.set_linewidth(0.4)


def panel_b_timeline(ax, frames, top_macs):
    """Minute-resolution timeline for 2026-05-15. Lanes per device."""
    bins = np.zeros((len(top_macs), 24 * 60), dtype=int)
    for ts, mac in frames:
        if mac not in top_macs:
            continue
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) \
             + datetime.timedelta(hours=UTC_TO_LOCAL_OFFSET_HOURS)
        if dt.strftime("%Y-%m-%d") != TIMELINE_DAY:
            continue
        m_idx = dt.hour * 60 + dt.minute
        i = top_macs.index(mac)
        bins[i, m_idx] += 1

    # Draw each lane as a horizontal band; darker = more frames in that minute
    cmap = plt.cm.YlOrRd
    M = bins.astype(float)
    M_log = np.log10(M + 1)
    for i, m in enumerate(top_macs):
        for minute in range(24 * 60):
            if M[i, minute] > 0:
                intensity = M_log[i, minute] / max(M_log.max(), 1)
                ax.axvspan(minute / 60, (minute + 1) / 60,
                           ymin=(len(top_macs) - 1 - i + 0.1) / len(top_macs),
                           ymax=(len(top_macs) - i - 0.1) / len(top_macs),
                           color=cmap(0.2 + 0.7 * intensity),
                           linewidth=0)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)])
    ax.set_ylim(0, len(top_macs))
    # Yticks map: device i (top to bottom in list) sits at lane y = N-1-i+0.5
    # so top lane (TV) -> y=5.5, bottom lane (Guest LP) -> y=0.5
    ax.set_yticks([len(top_macs) - i - 0.5 for i in range(len(top_macs))])
    ax.set_yticklabels([device_label(m) for m in top_macs])
    ax.set_xlabel(f"Local hour on {TIMELINE_DAY} (CEST)")
    ax.set_title("(b) One day at minute resolution", loc="left")
    # Devices whose first frame of the whole capture falls on this day:
    # annotate the time of that frame and the number of days with data.
    bbox_kw = dict(boxstyle="round,pad=0.18", fc="white",
                   ec=CALLOUT_COLOR, lw=0.4, alpha=1.0)
    for i, m in enumerate(top_macs):
        local = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                 + datetime.timedelta(hours=UTC_TO_LOCAL_OFFSET_HOURS)
                 for ts, mm in frames if mm == m]
        first = min(local)
        if first.strftime("%Y-%m-%d") != TIMELINE_DAY:
            continue
        n_days = len({d.strftime("%Y-%m-%d") for d in local})
        x = first.hour + first.minute / 60
        y = len(top_macs) - i - 0.5
        ax.annotate(f"first heard {first:%H:%M} ({n_days} days with data)",
                    xy=(x, y), xytext=FIRST_SEEN_TEXT_POS[R.label(m)[0]],
                    textcoords="data", fontsize=ps.SMALL_PT,
                    color=CALLOUT_COLOR, ha="center", va="center",
                    bbox=bbox_kw, zorder=10,
                    arrowprops=dict(arrowstyle="->", color=CALLOUT_COLOR, lw=0.8))


def panel_c_mobility(ax, frames_full, top_macs):
    """Mobility scatter. X = sounding rate (log Hz), Y = RSSI std (dB).
    Stationary devices land low on Y; mobile devices land high.
    """
    xs, ys, rssis, ns, labels = [], [], [], [], []
    for m in top_macs:
        fr = [f for f in frames_full if f[1] == m]
        rssi = np.array([f[2] for f in fr if f[2] is not None])
        ts = sorted(f[0] for f in fr)
        diffs = np.diff(ts) if len(ts) >= 2 else [1.0]
        active = np.array(diffs)[np.array(diffs) < BURST_GAP_S]
        rate = 1.0 / np.median(active) if len(active) else 0.1
        xs.append(rate)
        ys.append(rssi.std())
        rssis.append(rssi.mean())
        ns.append(len(fr))
        labels.append(R.label(m)[0])

    sc = ax.scatter(xs, ys, c=rssis, cmap="viridis_r",
                    s=[max(40, np.log10(n + 1) * 38) for n in ns],
                    edgecolors="white", linewidths=1.0, zorder=3,
                    alpha=0.85, vmin=-75, vmax=-35)
    # Per-MAC label offsets in points (positive y up). The 13-day capture
    # places three devices in the right-side high-rate cluster, two of
    # them at similar (rate, sigma); previous symmetric above/below
    # offsets collided ("4c:b9:eb8:b4:09" merged). Below each label
    # sits at a unique (dx, dy) with the right horizontal alignment so
    # the labels can fan out without overlap. Anchors come from leader
    # lines drawn explicitly so the reader can trace each label back to
    # its dot.
    # All labels float ABOVE their dots (sigma axis has no room below 0).
    # Co-located dots are split horizontally with leader lines so the
    # reader can trace each label back. The high-rate cluster on the
    # right uses a side label for 4c:b9:ea so it does not stack onto
    # 64:4e:d7.
    # Label positions in data coordinates (burst rate Hz, sigma dB): the two
    # Apple TVs sit on one point, so both labels go to its right, one above
    # the other; every label has a leader line to its dot.
    label_pos = {                          # (x, y, ha) by paper label
        "Apple TV (living room)": (1.3, 0.72, "left"),
        "Apple TV (bedroom)": (0.34, 1.35, "left"),
        "LG TV": (0.45, 3.85, "left"),
        "Laptop": (11.5, 4.05, "center"),
        "iRobot vacuum": (7.0, 2.2, "right"),
        "Samsung TV": (52.0, 1.3, "right"),
    }
    for x, y, lbl, m in zip(xs, ys, labels, top_macs):
        tx, ty, ha = label_pos[lbl]
        ax.annotate(
            device_label(m), (x, y), xytext=(tx, ty), textcoords="data",
            ha=ha, va="center", fontsize=ps.SMALL_PT, color="black",
            arrowprops=dict(arrowstyle="-", color=ps.OKABE_ITO["grey"],
                            lw=0.4, shrinkA=1, shrinkB=4),
        )

    ax.set_xscale("log")
    ax.set_xlim(0.3, 60)
    ax.set_ylim(0, 6)
    ax.set_xticks(list(BURST_TICKS))
    ax.set_xticklabels(list(BURST_TICK_LABELS))
    ax.set_xlabel("Burst rate (Hz, log)")
    ax.set_ylabel(r"RSSI std. dev. $\sigma$ (dB)")
    ax.set_title("(c) Burst rate against RSSI spread", loc="left")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean RSSI (dBm)")
    cbar.outline.set_linewidth(0.4)


def main(out_pdf: str) -> None:
    ps.apply()
    plt.rcParams["axes.grid"] = False
    frames_full = load_frames(with_rssi=True)      # (ts, addr, rssi)
    frames = [(f[0], f[1]) for f in frames_full]
    # the six labeled devices, in the paper's lane order
    seen = {m for m in Counter(f[1] for f in frames)
            if m and X.device_key(m) in X.DEVICE_GT}
    top_macs = sorted(seen, key=lambda m: ORDER_PREF.index(R.label(m)[0]))

    fig = plt.figure(figsize=(ps.TEXT_W_IN, FIG_H_IN))
    gs = fig.add_gridspec(2, 2, **GRID)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_a_heatmap(ax_a, frames, top_macs)
    panel_b_timeline(ax_b, frames, top_macs)
    panel_c_mobility(ax_c, frames_full, top_macs)

    fig.savefig(out_pdf, format="pdf")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    X.set_vm_limit()
    main(sys.argv[1] if len(sys.argv) > 1 else str(OUT_PDF))
