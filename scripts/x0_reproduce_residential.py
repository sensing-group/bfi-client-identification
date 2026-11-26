#!/usr/bin/env python3
"""Sanity check before X4/X5/X8: reproduce the paper's residential numbers
from the streamed per-day cache (xres_common.py).

(A) Full-window K=1 (pi_one_shot_full_window.py -> fig_q1_fullwindow):
    every day with the parsed schema, rows WHERE nc IS NOT NULL, frames per
    MAC concatenated in day order, MACs with >= 64 frames, rng seed 0, the
    five channel sets run in the original order so the RNG stream matches.
    Paper: body 73.4%, power+body 78.5%, all three 79.6% on 9 MACs.
(B) Cross-day matrix (bfi/demo_rebuttal_crossday.py ->
    figures/bfi_rebuttal_crossday.csv): rng seed 42, 10 trials x 30
    queries per ordered day pair, channels power+body+capability.
    Paper: same-day 98.4%, cross-day 97.4% over 62 pairs, worst 70.7%.

The original scripts load every frame of every day into Python dicts
(body_hex strings included); this re-implementation reads the compact
cache instead, uses the original window/cosine/RNG code path, and must
reproduce the published values exactly.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict

import numpy as np

import xres_common as X

FULL_SEED = 0
CROSS_SEED = 42
CROSS_TRIALS = 10
CROSS_QUERIES = 30
FULL_CHANNEL_SETS = [["power"], ["capability"], ["body"], ["power", "body"],
                     ["power", "capability", "body"]]
CROSS_CHANNELS = ["power", "body", "capability"]
OUT_JSON = X.FIG_DIR / "x0_reproduce_residential.json"
OUT_CROSS_CSV = X.FIG_DIR / "x0_reproduce_crossday.csv"
PAPER_CROSSDAY_CSV = X.FIG_DIR / "bfi_rebuttal_crossday.csv"


def win_mean(feats: dict, s: int) -> dict:
    return {k: np.mean(v[s:s + X.WINDOW_FRAMES], axis=0)
            for k, v in feats.items()}


def full_window() -> dict:
    by_mac: dict[str, list[dict]] = defaultdict(list)
    days_used = []
    for p in X.day_paths():
        d = X.load_day(p.stem)
        if not d.pop("__parsed__"):
            continue                    # original skips on missing columns
        d.pop("__geoms__")
        days_used.append(p.stem)
        for m, v in d.items():
            keep = v["cap"][:, 0] >= 0  # WHERE nc IS NOT NULL
            if keep.any():
                by_mac[m].append(X.paper_frame_features(
                    {k: a[keep] for k, a in v.items()}, fb_null_as_su=False))
        del d
    feats = {m: {k: np.concatenate([f[k] for f in lst])
                 for k in lst[0]} for m, lst in by_mac.items()}
    elig = sorted(m for m, f in feats.items()
                  if len(f["body"]) >= X.MIN_FRAMES_PER_MAC)
    n = {m: len(feats[m]["body"]) for m in elig}
    rng = np.random.default_rng(FULL_SEED)
    res = {}
    for chans in FULL_CHANNEL_SETS:
        correct = total = 0
        per_mac = {m: [0, 0] for m in elig}
        for _ in range(X.N_TRIALS):
            protos = {m: win_mean(feats[m], int(rng.integers(
                0, n[m] - X.WINDOW_FRAMES))) for m in elig}
            for m in elig:
                for _q in range(X.QUERIES_PER_MAC):
                    q = win_mean(feats[m], int(rng.integers(
                        0, n[m] - X.WINDOW_FRAMES)))
                    sc = {om: sum(X.cosine(q[c], protos[om][c])
                                  for c in chans) for om in elig}
                    hit = int(max(sc, key=sc.get) == m)
                    correct += hit
                    total += 1
                    per_mac[m][0] += hit
                    per_mac[m][1] += 1
        res["+".join(chans)] = dict(accuracy=correct / total, n=total,
                                    n_macs=len(elig), per_mac=per_mac)
        print(f"  full-window {'+'.join(chans):22s} "
              f"{correct / total:.4f} ({total})", file=sys.stderr)
    return dict(days=days_used, eligible={m: n[m] for m in elig},
                results=res)


def crossday() -> dict:
    per_day = {}
    all_days = sorted(p.stem for p in X.day_paths())
    for stem in all_days:
        d = X.load_day(stem)
        d.pop("__parsed__"); d.pop("__geoms__")
        per_day[stem] = {
            m: X.paper_frame_features(
                {k: a[~v["body_null"]] for k, a in v.items()},
                fb_null_as_su=True)
            for m, v in d.items() if (~v["body_null"]).any()}
        del d
    counts = defaultdict(dict)
    for day, bm in per_day.items():
        for m, f in bm.items():
            counts[m][day] = len(f["body"])
    elig = sorted(m for m, dc in counts.items()
                  if sum(c >= X.MIN_FRAMES_PER_MAC for c in dc.values()) >= 2)
    rng = np.random.default_rng(CROSS_SEED)
    W = X.WINDOW_FRAMES
    rows = []
    per_mac = {m: [0, 0, 0.0] for m in elig}   # hits, queries, sum chance
    for ds in all_days:
        for dq in all_days:
            sup = {m: per_day[ds][m] for m in elig
                   if m in per_day[ds] and len(per_day[ds][m]["body"]) >= W}
            qry = {m: per_day[dq][m] for m in elig
                   if m in per_day[dq] and len(per_day[dq][m]["body"]) >= W}
            shared = sorted(set(sup) & set(qry))
            if len(shared) < 2:
                continue
            correct = total = 0
            for _ in range(CROSS_TRIALS):
                protos = {m: win_mean(sup[m], int(rng.integers(
                    0, len(sup[m]["body"]) - W + 1))) for m in shared}
                for m in shared:
                    for _q in range(CROSS_QUERIES):
                        q = win_mean(qry[m], int(rng.integers(
                            0, len(qry[m]["body"]) - W + 1)))
                        sc = {om: sum(X.cosine(q[c], protos[om][c])
                                      for c in CROSS_CHANNELS)
                              for om in shared}
                        hit = int(max(sc, key=sc.get) == m)
                        correct += hit
                        total += 1
                        if ds != dq:
                            per_mac[m][0] += hit
                            per_mac[m][1] += 1
                            per_mac[m][2] += 1.0 / len(shared)
            rows.append(dict(day_support=ds, day_query=dq,
                             condition="same_day" if ds == dq else "cross_day",
                             n_macs=len(shared), acc=correct / total,
                             n_queries=total))
    same = [r["acc"] for r in rows if r["condition"] == "same_day"]
    cross = [r["acc"] for r in rows if r["condition"] == "cross_day"]
    paper = {(r["day_support"], r["day_query"]): float(r["acc"])
             for r in csv.DictReader(open(PAPER_CROSSDAY_CSV))}
    diffs = [abs(paper[(r["day_support"], r["day_query"])] - r["acc"])
             for r in rows if (r["day_support"], r["day_query"]) in paper]
    with open(OUT_CROSS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[out] {OUT_CROSS_CSV}", file=sys.stderr)
    return dict(eligible=elig, n_rows=len(rows), per_mac_cross_day=per_mac,
                day_counts={m: counts[m] for m in elig},
                same_day_mean=float(np.mean(same)), n_same=len(same),
                cross_day_mean=float(np.mean(cross)), n_cross=len(cross),
                cross_day_min=float(np.min(cross)),
                rows_matching_paper_csv=len(diffs),
                max_abs_diff_vs_paper_csv=float(max(diffs)) if diffs else None,
                paper_csv_rows=len(paper))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    if a.dry_run:
        print(X.dry_run_report())
        return 0
    t0 = time.time()
    X.build_all()
    out = dict(full_window=full_window())
    X.log_rss("after full-window")
    out["crossday"] = crossday()
    out["runtime_s"] = time.time() - t0
    out["peak_rss_mb"] = X.log_rss("end")
    X.write_json(OUT_JSON, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
