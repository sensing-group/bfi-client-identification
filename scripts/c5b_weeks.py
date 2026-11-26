#!/usr/bin/env python3
"""C5b-T2: weeks, not days. The longest horizon the data allow: the three
labelled TVs heard by the ESP32-C5 mesh (B) in July (rotation file,
2026-07-15..24) and in September (live file, 2026-09-06..08), 44 to 55 days
apart, both with the firmware's 512-byte truncation.

Protocol = c5_cross_receiver.run_condition: enrol K windows from ONE
(session, day) block of each TV in the enrolment month, query T consecutive
windows of every block of the query month (1-NN, euclidean on features
z-scored label-free on the pooled enrolment+query windows), chance per
decision = 1/3. Every condition is run once per seed in SEEDS (one fresh
generator per seed, conditions in the order below) and the recall is the
mean over seeds; the per-seed rows go to c5b_weeks_seeds.csv. Features: angles (first-column Givens vector on the bins
fully covered for all three TVs in both months), angles_rq (re-quantised
to 4/2 bits) and the format one-hot baseline.

Conditions
  Jul_to_Sep   enrol July, query September      Sep_to_Jul   the reverse
  Jul_within   enrol one July block, query other July days
  Sep_within   enrol one September block, query other September days

Also reported: every TV's report format per month in (B) (the Samsung
changed format between the AX210 capture (A, May) and (B); whether it
changed within (B) is checked here window by window).

Outputs (figures/): c5b_weeks.csv (mean over seeds, with recall_sd),
c5b_weeks_seeds.csv, c5b_weeks_summary.json.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_cross_receiver as CX  # noqa: E402
import c5b_common as B  # noqa: E402
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = tuple(range(10))
JULY, SEPT = "rot20260724", "live"
CONDITIONS = (("Jul_to_Sep", JULY, SEPT), ("Sep_to_Jul", SEPT, JULY),
              ("Jul_within", JULY, JULY), ("Sep_within", SEPT, SEPT))
S_PER_DAY = 86400.0
OUT = dict(csv=X.FIG_DIR / f"{B.OUT_PREFIX}weeks.csv",
           seeds=X.FIG_DIR / f"{B.OUT_PREFIX}weeks_seeds.csv",
           summary=X.FIG_DIR / f"{B.OUT_PREFIX}weeks_summary.json")


def mean_over_seeds(seed_rows: list[dict]) -> list[dict]:
    """One row per (condition, feature, K, T, device): recall averaged over
    seeds (sample SD alongside), decisions summed over seeds."""
    key = lambda r: (r["condition"], r["feature"], r["K"], r["T"], r["device"])
    groups: dict = {}
    for r in seed_rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for rs in groups.values():
        rec = np.array([r["recall"] for r in rs], float)
        row = {k: v for k, v in rs[0].items() if k != "seed"}
        row.update(recall=float(rec.mean()),
                   recall_sd=float(rec.std(ddof=1)) if rec.size > 1 else 0.0,
                   n_decisions=int(sum(r["n_decisions"] for r in rs)), n_seeds=len(rs))
        out.append(row)
    return out


def main() -> int:
    X.set_vm_limit()
    t0 = time.time()
    vocab: dict = {}
    ws = B.load_b_sources(vocab)
    if JULY not in ws or SEPT not in ws:
        print("both (B) sources are needed", file=sys.stderr)
        return 1
    devs = [str(m) for m in ws[JULY]["macs"]]
    names = [R.short(m) for m in devs]
    sets = {n: CX.WinSet(n, ws[n], {}) for n in (JULY, SEPT)}
    gallery = sorted(set(sets[JULY].dev.tolist()) & set(sets[SEPT].dev.tolist()))
    # one mask for every condition: bins covered in every window of the
    # gallery devices in BOTH months
    mask = np.ones(C.N_SC_BINS, bool)
    for s in sets.values():
        mask &= s.subset(np.isin(s.dev, gallery)).bin_mask()
    n_vocab = len(vocab)
    seed_rows = []
    per_cond = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for cond, en, qn in CONDITIONS:
            E = sets[en].subset(np.isin(sets[en].dev, gallery))
            Q = E if en == qn else sets[qn].subset(np.isin(sets[qn].dev, gallery))
            hits = CX.run_condition(cond, E, Q, gallery, mask, n_vocab, rng)
            seed_rows += [dict(r, seed=seed) for r in CX.tally_rows(
                cond, B.SOURCE_MONTH[en], B.SOURCE_MONTH[qn],
                f"{B.SOURCE_MONTH[en]}->{B.SOURCE_MONTH[qn]}",
                hits, gallery, names, mask.sum())]
        print(f"[seed {seed}] peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    rows = mean_over_seeds(seed_rows)
    for cond, en, qn in CONDITIONS:
        E = sets[en].subset(np.isin(sets[en].dev, gallery))
        Q = E if en == qn else sets[qn].subset(np.isin(sets[qn].dev, gallery))
        gap_days = None
        if en != qn:
            early, late = (E, Q) if en == JULY else (Q, E)
            gap_days = dict(min=float((late.ts0.min() - early.ts0.max()) / S_PER_DAY),
                            max=float((late.ts0.max() - early.ts0.min()) / S_PER_DAY))
        per_cond[cond] = dict(
            enrol_month=B.SOURCE_MONTH[en], query_month=B.SOURCE_MONTH[qn],
            enrol_windows=int(E.dev.size), query_windows=int(Q.dev.size),
            enrol_blocks_per_device={names[g]: len([k for k in E.blocks() if k[0] == g])
                                     for g in gallery},
            query_blocks_per_device={names[g]: len([k for k in Q.blocks() if k[0] == g])
                                     for g in gallery},
            horizon_days=gap_days)
    pooled = B.pool_sources(ws)
    summary = dict(
        seeds=list(SEEDS), k_grid=[CX.klabel(k) for k in CX.K_GRID], t_grid=list(CX.T_GRID),
        n_trials=CX.N_TRIALS, max_queries_per_device=CX.MAX_QUERIES_PER_DEVICE,
        gallery=[names[g] for g in gallery], chance=1.0 / len(gallery),
        bins_kept=int(mask.sum()), bins_kept_idx=np.flatnonzero(mask).tolist(),
        dims=int(mask.sum()) * B.N_BLOCKS,
        days={B.SOURCE_MONTH[n]: sorted(set(ws[n]["day"].tolist())) for n in ws},
        format_by_device_month=B.format_counts(pooled, names),
        conditions=per_cond,
        zscore="label-free on the pooled enrolment + query windows",
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb())
    with open(OUT["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CX.CSV_COLS + ["recall_sd", "n_seeds"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT["seeds"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CX.CSV_COLS + ["seed"])
        w.writeheader()
        w.writerows(seed_rows)
    X.write_json(OUT["summary"], summary)
    print(f"[out] {OUT['csv']} ({len(rows)} rows) {summary['runtime_s']:.0f}s "
          f"peak RSS {summary['peak_rss_mb']:.0f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
