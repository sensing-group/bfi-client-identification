#!/usr/bin/env python3
"""C5b-T4: one number for the risk, on the ESP32-C5 listener's own data (B).

The three labelled TVs' windows from both (B) months (July rotation,
September live file; 512-byte truncated reports) are pooled on the r_common
layout and scored with r_eval's protocol unchanged: one labelled (session,
day) block enrolled per TV with K windows, query = T consecutive windows of
another block, 1-NN over 320-d Givens features restricted to the bins fully
covered for all three TVs (common mask), z-scored label-free on all (B)
windows; chance per decision = 1/|gallery| (3 when all TVs have a block).
  same_day        enrol and query blocks on the same day, different sessions
  other_day_pair  enrol day != query day (every pair, incl. July->September)
  other_day_any   gallery enrolled from one block on some other day
Features: angles, angles_centroid, angles_rq, format, angles+format (r_eval).
Time to identify (m_time_to_identify protocol, other_day_any, angles): recall
against T with the listening time measured on the TVs' own frame stamps.

Outputs (figures/): c5b_risk_main.csv, c5b_risk_per_device.csv,
c5b_risk_daypair.csv, c5b_risk_time.csv, c5b_risk_summary.json.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5b_common as B  # noqa: E402
import m_time_to_identify as M  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_GRID = (1, 16, 0)
T_GRID = (1, 4, 16, 32)
N_TRIALS = E.N_TRIALS
SEED = 300
CONDITIONS = E.CONDITIONS
OUT = dict(main=X.FIG_DIR / f"{B.OUT_PREFIX}risk_main.csv",
           per_device=X.FIG_DIR / f"{B.OUT_PREFIX}risk_per_device.csv",
           daypair=X.FIG_DIR / f"{B.OUT_PREFIX}risk_daypair.csv",
           time=X.FIG_DIR / f"{B.OUT_PREFIX}risk_time.csv",
           summary=X.FIG_DIR / f"{B.OUT_PREFIX}risk_summary.json")


def main() -> int:
    X.set_vm_limit()
    t0 = time.time()
    vocab: dict = {}
    ws = B.load_b_sources(vocab)
    w = B.compact_devices(B.pool_sources(ws))      # drop MACs absent from (B)
    names = [R.short(str(m)) for m in w["macs"]]
    present = sorted(set(w["dev"].tolist()))
    mask = B.common_mask(w, np.ones(w["dev"].size, bool))
    wm = B.apply_mask(w, mask)
    tb = E.Table(wm)
    X.log_rss("table")
    extra = dict(gap_s=int(R.SESSION_GAP_S), dataset="B", bins_kept=int(mask.sum()))
    rng_b = np.random.default_rng(SEED + 1)
    main_rows, dev_rows, dp_rows = [], [], []
    for ci, cond in enumerate(CONDITIONS):
        t = time.time()
        recs = E.run_condition(tb, cond, K_GRID, T_GRID, N_TRIALS, SEED + ci)
        r1, r2 = E.aggregate(recs, tb.names, extra, rng_b)
        main_rows += r1
        dev_rows += r2
        dp_rows += E.daypair_rows(recs, tb.names, extra)
        print(f"[eval] {cond}: {len(recs)} records {time.time() - t:.0f}s "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
        del recs
        gc.collect()
    recs = M.run(tb, wm, N_TRIALS, SEED + 50)
    time_rows = M.rows_for(recs, tb.names, wm)
    del recs
    E.write_csv(OUT["main"], main_rows)
    E.write_csv(OUT["per_device"], dev_rows)
    E.write_csv(OUT["daypair"], dp_rows)
    E.write_csv(OUT["time"], time_rows)
    blocks = {}
    for (d, s, day), ii in tb.blocks.items():
        blocks.setdefault(names[d], []).append(dict(session=s, day=day, windows=int(ii.size)))
    X.write_json(OUT["summary"], dict(
        trials=N_TRIALS, k_grid=[E.klabel(k) for k in K_GRID], t_grid=list(T_GRID),
        time_k_grid=[E.klabel(k) for k in M.K_GRID], time_t_grid=list(M.T_GRID),
        devices=[names[d] for d in present],
        windows_per_device={names[d]: int((w["dev"] == d).sum()) for d in present},
        days=sorted(set(w["day"].tolist())),
        bins_kept=int(mask.sum()), bins_kept_idx=np.flatnonzero(mask).tolist(),
        format_by_device_month=B.format_counts(w, names), blocks=blocks,
        window_seconds_median={names[d]: float(np.median(
            (w["ts1"] - w["ts0"])[w["dev"] == d])) for d in present},
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
