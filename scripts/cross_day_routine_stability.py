#!/usr/bin/env python3
"""Per-MAC behavioural-routine stability across days.

For each (MAC, day) computes:
  - frame count
  - sounding rate (Hz)
  - mean RSSI and its std
  - hour-of-day vector (24 bins, frame-count-normalised)

Then, for each MAC observed on >= MIN_DAYS_FOR_CORRELATION days, computes
the pairwise cosine similarity matrix of its day-by-day hour-of-day
vectors. A MAC whose routine repeats has near-1 off-diagonal entries.

Outputs (figures/):
  cross_day_routine_stability.json   per-MAC stability and the population
  bfi_routine_daypairs.csv           one row per (MAC, day pair) for MACs
                                     with >= MIN_DAYS_IN_CSV eligible days
                                     (mac, n_days, day_a, day_b, cosine);
                                     read by fig_routine_stability.py

Reads the daily files of BFI_ARCHIVE_DIR over the paper's window
(xres_common.day_paths), raw or released schema. This is the
label-free routine evidence of Section 6.5 and Appendix B. Memory bound:
the per-day per-MAC aggregation streams through each daily DB and keeps
only the 24-bin histogram per (MAC, day) in memory. RSS stays well under
100 MB.
"""
from __future__ import annotations

import datetime
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xres_common as X  # noqa: E402  (archive path and schema, memory)

UTC_TO_LOCAL_OFFSET_H = 2          # Europe/Zurich CET/CEST
MIN_FRAMES_PER_MAC_DAY = 50        # below this the rate / hour vector is noise
MIN_DAYS_FOR_CORRELATION = 3       # need 3 day-pairs to call a routine
MIN_DAYS_IN_CSV = 2                # the figure lists 2-day MACs as unscored
OUT_JSON = X.FIG_DIR / "cross_day_routine_stability.json"
OUT_CSV = X.FIG_DIR / "bfi_routine_daypairs.csv"


def aggregate_one_day(db_path: Path) -> dict[str, dict]:
    """Returns {mac: {n: int, rate_hz: float, rssi_mean: float,
                       rssi_std: float, hod_vec: list[24]}}."""
    conn = X.connect_ro(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bfi)").fetchall()}
    if not {"ts", X.SRC_COL, X.RSSI_COL} <= cols:
        conn.close()
        return {}
    per_mac: dict[str, dict] = defaultdict(
        lambda: {"ts": [], "rssi": [], "hod": np.zeros(24, dtype=np.int64)}
    )
    for ts, mac, rssi in conn.execute(
            f"SELECT ts, {X.SRC_COL}, {X.RSSI_COL} FROM bfi"):
        if mac is None:
            continue
        per_mac[mac]["ts"].append(ts)
        if rssi is not None:
            per_mac[mac]["rssi"].append(rssi)
        h = (datetime.datetime.fromtimestamp(
                ts, tz=datetime.timezone.utc).hour
             + UTC_TO_LOCAL_OFFSET_H) % 24
        per_mac[mac]["hod"][h] += 1
    conn.close()
    out: dict[str, dict] = {}
    for mac, d in per_mac.items():
        n = len(d["ts"])
        if n < MIN_FRAMES_PER_MAC_DAY:
            continue
        ts = np.sort(np.asarray(d["ts"], dtype=np.float64))
        diffs = np.diff(ts) if ts.size >= 2 else np.asarray([1.0])
        med_dt = float(np.median(diffs)) if diffs.size else 1.0
        rate = 1.0 / med_dt if med_dt > 0 else 0.0
        rssi = (np.asarray(d["rssi"], dtype=np.float32)
                if d["rssi"] else np.zeros(0, dtype=np.float32))
        hod = d["hod"].astype(np.float64)
        hod /= max(1.0, hod.sum())
        out[mac] = dict(
            n=int(n),
            rate_hz=rate,
            rssi_mean=float(rssi.mean()) if rssi.size else None,
            rssi_std=float(rssi.std()) if rssi.size else None,
            hod_vec=hod.tolist(),
        )
    return out


def write_daypairs_csv(days: dict[str, dict[str, dict]]) -> int:
    """One row per (MAC, day pair) over the MACs seen on at least
    MIN_DAYS_IN_CSV eligible days; cosine of the hour-of-day vectors."""
    mac_days: dict[str, list[str]] = defaultdict(list)
    for day in sorted(days):
        for mac in days[day]:
            mac_days[mac].append(day)
    rows = ["mac,n_days,day_a,day_b,cosine"]
    for mac, dl in sorted(mac_days.items()):
        if len(dl) < MIN_DAYS_IN_CSV:
            continue
        for a, b in combinations(dl, 2):
            va = np.asarray(days[a][mac]["hod_vec"])
            vb = np.asarray(days[b][mac]["hod_vec"])
            na = np.linalg.norm(va) or 1.0
            nb = np.linalg.norm(vb) or 1.0
            rows.append(f"{mac},{len(dl)},{a},{b},{float(va @ vb / (na * nb)):.6f}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text("\n".join(rows) + "\n")
    print(f"Saved {OUT_CSV} ({len(rows) - 1} pairs)", file=sys.stderr)
    return len(rows) - 1


def main() -> int:
    X.set_vm_limit()
    days: dict[str, dict[str, dict]] = {}
    for db_path in X.day_paths():
        day = db_path.stem.replace("bfi_", "")
        agg = aggregate_one_day(db_path)
        days[day] = agg
        print(f"  {day}: {len(agg)} MAC(s) above threshold",
              file=sys.stderr)
        X.log_rss(day)

    # MAC -> sorted list of days seen
    mac_days: dict[str, list[str]] = defaultdict(list)
    for day, agg in days.items():
        for mac in agg:
            mac_days[mac].append(day)
    for mac in mac_days:
        mac_days[mac].sort()

    print("\n=== Per-MAC stability (cosine of hour-of-day vector "
          f"across days, MACs with >= {MIN_DAYS_FOR_CORRELATION} days) ===",
          file=sys.stderr)
    per_mac_stability: dict[str, dict] = {}
    for mac, day_list in mac_days.items():
        if len(day_list) < MIN_DAYS_FOR_CORRELATION:
            continue
        vecs = np.array([days[d][mac]["hod_vec"] for d in day_list])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vecs / norms
        sim = unit @ unit.T
        n = sim.shape[0]
        if n < 2:
            continue
        off_diag = sim[np.triu_indices(n, k=1)]
        per_mac_stability[mac] = dict(
            mac_prefix=mac[:11],
            n_days=int(n),
            cosine_mean=float(off_diag.mean()),
            cosine_min=float(off_diag.min()),
            cosine_max=float(off_diag.max()),
            days=day_list,
            mean_daily_rate_hz=float(np.mean(
                [days[d][mac]["rate_hz"] for d in day_list])),
            rssi_mean=float(np.mean(
                [days[d][mac]["rssi_mean"] for d in day_list
                 if days[d][mac]["rssi_mean"] is not None])),
            total_frames=int(np.sum(
                [days[d][mac]["n"] for d in day_list])),
        )
        print(f"  {mac[:11]:>12s}  n_days={n:>2d}  "
              f"<cos>={off_diag.mean():.3f}  "
              f"[{off_diag.min():.3f}, {off_diag.max():.3f}]  "
              f"rate={per_mac_stability[mac]['mean_daily_rate_hz']:>5.2f}Hz",
              file=sys.stderr)

    # Cross-MAC: for each day-pair, what is the distribution of
    # off-diagonal cosines? Lets us claim "the routine repeats" with a
    # population number, not a per-MAC anecdote.
    pop_cosines: list[float] = []
    for mac, s in per_mac_stability.items():
        if s["n_days"] >= 2:
            day_list = s["days"]
            vecs = np.array([days[d][mac]["hod_vec"] for d in day_list])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = vecs / norms
            sim = unit @ unit.T
            pop_cosines.extend(sim[np.triu_indices(sim.shape[0], k=1)])
    pop_cosines_arr = np.asarray(pop_cosines)
    summary = dict(
        n_mac_day_pairs=int(pop_cosines_arr.size),
        cosine_median=float(np.median(pop_cosines_arr))
                       if pop_cosines_arr.size else None,
        cosine_mean=float(pop_cosines_arr.mean())
                     if pop_cosines_arr.size else None,
        cosine_q05=float(np.quantile(pop_cosines_arr, 0.05))
                    if pop_cosines_arr.size else None,
        cosine_q95=float(np.quantile(pop_cosines_arr, 0.95))
                    if pop_cosines_arr.size else None,
    )
    print(f"\nPopulation across {summary['n_mac_day_pairs']} "
          f"(MAC, day-pair) measurements:", file=sys.stderr)
    if summary["cosine_median"] is not None:
        print(f"  median cosine = {summary['cosine_median']:.3f}, "
              f"mean = {summary['cosine_mean']:.3f}, "
              f"5–95% range = [{summary['cosine_q05']:.3f}, "
              f"{summary['cosine_q95']:.3f}]", file=sys.stderr)

    Path(OUT_JSON).write_text(json.dumps(dict(
        per_mac=per_mac_stability,
        population=summary,
        config=dict(
            archive_dir=str(X.ARCHIVE_DIR), days=sorted(days),
            min_frames_per_mac_day=MIN_FRAMES_PER_MAC_DAY,
            min_days_for_correlation=MIN_DAYS_FOR_CORRELATION,
            utc_to_local_offset_h=UTC_TO_LOCAL_OFFSET_H,
        ),
    ), indent=2))
    print(f"\nSaved {OUT_JSON}", file=sys.stderr)
    write_daypairs_csv(days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
