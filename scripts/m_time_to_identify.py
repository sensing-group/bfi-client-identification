#!/usr/bin/env python3
"""M2: time to identify (reviewer D). Cross-day recall as a function of
listening time.

Protocol = r_eval other_day_any (gallery = every device enrolled from one
labelled block on some day other than the query day, K windows; query = T
consecutive windows from one block of the query day, 320-d z-scored Givens
vectors averaged, 1-NN over the K enrolled windows). T runs over T_GRID;
the listening time of a query is the span of its T consecutive windows
measured on the device's own frame timestamps (last frame of the last
window minus first frame of the first window), reported as the median over
queries in minutes, next to T times the device's median single-window span.

Output (figures/): m_time_to_identify.csv, one row per (device, K, T) plus
pooled rows (device = ALL), and m_time_to_identify_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_GRID = (1, 16, 0)                      # 0 = all windows of the session
T_GRID = (1, 2, 4, 8, 16, 32)
CONDITION = "other_day_any"
N_TRIALS = E.N_TRIALS
SEED = 200
POOLED_LABEL = "ALL"
S_PER_MIN = 60.0
RECORD_BYTES_EST = 300                   # dry-run memory estimate per record

OUT = dict(main=R.FIG_DIR / "m_time_to_identify.csv",
           summary=R.FIG_DIR / "m_time_to_identify_summary.json")


def run(tb: E.Table, w: dict, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    recs = []
    for G, m, qb, (de, dq) in E.episodes(tb, CONDITION, rng, n_trials):
        gdev = list(G)
        perms = {g: rng.permutation(tb.blocks[G[g]]) for g in gdev}
        qwin = tb.blocks[qb]
        for T in T_GRID:
            if qwin.size < T:
                continue
            s0 = int(rng.integers(qwin.size - T + 1))
            qi = qwin[s0:s0 + T]
            span_s = float(w["ts1"][qi[-1]] - w["ts0"][qi[0]])
            qz = tb.Z[qi].mean(0)
            dz = {g: np.sqrt(((tb.Z[perms[g]] - qz) ** 2).sum(1))
                  for g in gdev}
            for K in K_GRID:
                kk = {g: perms[g].size if K == 0 else min(K, perms[g].size)
                      for g in gdev}
                d = np.array([dz[g][:kk[g]].min() for g in gdev])
                pred = gdev[E._argmin_random(d, rng)]
                recs.append(dict(K=K, T=T, dev=m, block=qb, G=len(gdev),
                                 hit=pred == m, span_s=span_s,
                                 capped=K != 0 and perms[m].size < K))
    return recs


def rows_for(recs: list[dict], names: list[str], w: dict) -> list[dict]:
    med_span = {}
    for i, m in enumerate(w["macs"]):
        s = w["dev"] == i
        med_span[i] = float(np.median(w["ts1"][s] - w["ts0"][s]))
    grp = defaultdict(list)
    for r in recs:
        grp[(r["K"], r["T"], r["dev"])].append(r)
        grp[(r["K"], r["T"], None)].append(r)
    rows = []
    for (K, T, dev), rr in sorted(grp.items(),
                                  key=lambda x: (x[0][0] or 999, x[0][1],
                                                 -1 if x[0][2] is None
                                                 else x[0][2])):
        span = np.array([r["span_s"] for r in rr]) / S_PER_MIN
        hits = np.array([r["hit"] for r in rr])
        if dev is None:
            t_med = float(np.mean([T * med_span[i] for i in med_span])
                          / S_PER_MIN)
        else:
            t_med = T * med_span[dev] / S_PER_MIN
        rows.append(dict(
            device=POOLED_LABEL if dev is None else names[dev],
            K=E.klabel(K), T=T,
            minutes_median=float(np.median(span)),
            minutes_p10=float(np.percentile(span, 10)),
            minutes_p90=float(np.percentile(span, 90)),
            minutes_T_x_median_window=t_med,
            recall=float(hits.mean()),
            n_queries=int(hits.size),
            n_query_blocks=len({r["block"] for r in rr}),
            chance=float(np.mean([1.0 / r["G"] for r in rr])),
            enrol_capped_frac=float(np.mean([r["capped"] for r in rr]))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    tb = E.Table(w)
    if a.dry_run:
        rng = np.random.default_rng(SEED)
        n = sum(1 for _ in E.episodes(tb, CONDITION, rng, 1))
        est = n * a.trials * len(K_GRID) * len(T_GRID)
        print(json.dumps(dict(windows=int(len(w["dev"])),
                              episodes_per_trial=n, est_records=est,
                              est_record_mb=est * RECORD_BYTES_EST / 1e6,
                              table_mb=tb.Z.nbytes * 2 / 1e6), indent=2))
        return 0
    recs = run(tb, w, a.trials, SEED)
    X.log_rss("records")
    rows = rows_for(recs, tb.names, w)
    E.write_csv(OUT["main"], rows)
    X.write_json(OUT["summary"], dict(
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        trials=a.trials, condition=CONDITION,
        k_grid=[E.klabel(k) for k in K_GRID], t_grid=list(T_GRID),
        n_records=len(recs)))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
