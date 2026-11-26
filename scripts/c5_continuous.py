#!/usr/bin/env python3
"""C5 continuous: the whole mesh recording, not only its two ends. The
ESP32-C5 boards logged the gateway's feedback from 2026-07-15 to 2026-09-08,
all but 18 hours on 31 July; the paper's (B) uses the first rotation file (July) and the
live file (September). This script registers the 24 rotation files in between
as further sources of c5_common, streams each once into the same caches, and
runs the c5b_weeks protocol (mean over the same ten seeds) with July enrolled and every later file queried,
so recognition is read against the days elapsed since enrolment.

Protocol = c5_cross_receiver.run_condition (as c5b_weeks Jul_to_Sep): enrol
K windows from ONE (session, day) block of each TV in July, query T
consecutive windows of every block of the query file, 1-NN, chance = 1/n
gallery. The gallery is the labelled TVs heard in July and in the query file;
the bin mask is the bins covered in every gallery window of the pair.

Stages
  --build   stream every rotation file not yet cached (coverage + labelled
            frame cache + windows); about 4 min per GB of gzip, one at a time
  --eval    July -> each later source, angles / angles_rq / format
  --pairs   every file enrolled, every other file queried (both directions)
  --dry-run list the sources and their sizes, read nothing

Outputs (BFI_X_FIG_DIR): c5_continuous.csv, c5_continuous_summary.json,
c5_continuous_pairs.csv, c5_continuous_pairs_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_cross_receiver as CX  # noqa: E402
import c5b_common as B  # noqa: E402
import c5b_weeks as W  # noqa: E402
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = tuple(range(10))         # as c5b_weeks: recall = mean over seeds
JULY = "rot20260724"
# The middle files: raw rotations (bfi_captures.jsonl.<stamp>.gz in
# BFI_C5_ROTATED_DIR) or the released copies (bfi_c5_<stamp>.jsonl.gz in
# BFI_C5_MIDDLE_DIR, which takes precedence when set).
MIDDLE_DIR = os.environ.get("BFI_C5_MIDDLE_DIR")
ROTATED_GLOB = "bfi_c5_*Z.jsonl.gz" if MIDDLE_DIR else "bfi_captures.jsonl.*.gz"
STAMP_RE = re.compile(r"(?:bfi_captures\.jsonl\.|bfi_c5_)(\d{8})T(\d{4})\d{2}Z\.(?:jsonl\.)?gz$")
S_PER_DAY = 86400.0
OUT = dict(csv=X.FIG_DIR / "c5_continuous.csv",
           summary=X.FIG_DIR / "c5_continuous_summary.json",
           pairs=X.FIG_DIR / "c5_continuous_pairs.csv",
           pairs_summary=X.FIG_DIR / "c5_continuous_pairs_summary.json")
# --pairs: every file enrolled, every other file queried; fewer seeds and one
# query window keep the 650 pairs to about three hours (3.2 h measured).
PAIR_SEEDS = (0, 1, 2)
PAIR_K_GRID = (1, 0)
PAIR_T_GRID = (1,)
# Summary of --pairs (Section 6.4): one window enrolled, the pairs in which all
# three televisions are heard, binned by the days between the two files.
PAIR_SUMMARY_WHERE = dict(feature="angles", K="1", n_gallery="3")
PAIR_GAP_EDGES_D = (0, 3, 7, 14, 21, 28, 35, 56)
PAIR_RHO_SPLIT_D = 21.0                 # the correlation is reported within and beyond three weeks
PAIR_COLS = ("enrol_source", "query_source", "gap_days", "feature", "K", "n_gallery",
             "chance", "bins_kept", "recall", "recall_sd", "n_seeds")
CSV_EXTRA = ("source", "first_day", "last_day", "days_after_enrol_min",
             "days_after_enrol_max")


def register_rotations() -> list[str]:
    """Add every rotation file other than the July one to C.SOURCES."""
    names = []
    for p in sorted((Path(MIDDLE_DIR) if MIDDLE_DIR else C.B_ROTATED_DIR).glob(ROTATED_GLOB)):
        m = STAMP_RE.search(p.name)
        if not m:
            continue
        if f"rot{m.group(1)}" == JULY:
            continue
        name = f"rot{m.group(1)}T{m.group(2)}"   # two files share 20260828
        C.SOURCES.setdefault(name, p)
        names.append(name)
    return names


def recording_totals(names: list[str]) -> dict:
    """Section 4 facts of the whole recording from the coverage caches: unique
    reports and source addresses over all files, boards per file."""
    files = [JULY] + names + ["live"]
    reports, addrs, boards = 0, set(), {}
    for n in files:
        c = C.load_coverage(n)
        reports += sum(v["unique_reports"] for v in c["sources"].values())
        addrs |= set(c["sources"])
        boards[n] = sum(1 for v in c["nodes"].values() if v["hours_active"] > 0)
    first, last = C.load_coverage(JULY)["ts_min_iso"], C.load_coverage("live")["ts_max_iso"]
    return dict(first=first, last=last, n_files=len(files), unique_reports=reports,
                source_addresses=len(addrs), boards_per_file=boards,
                boards_min=min(boards.values()), boards_max=max(boards.values()))


def build(names: list[str]) -> None:
    devs = C.labelled_macs()
    for n in names:
        if not C.coverage_path(n).exists():
            agg = C.SourcePass(n, C.SOURCES[n], set(devs)).run()
            C.log(f"[{n}] {agg['lines']} lines in {agg['seconds']:.0f}s, "
                  f"{agg['ts_min_iso']}..{agg['ts_max_iso']}, peak RSS "
                  f"{agg['peak_rss_mb']:.0f} MB")
            gc.collect()
        C.get_b_windows(n, devs)
        gc.collect()


def evaluate() -> int:
    vocab: dict = {}
    ws = B.load_b_sources(vocab)
    if JULY not in ws:
        print("the July source is needed", file=sys.stderr)
        return 1
    devs = [str(m) for m in ws[JULY]["macs"]]
    names = [R.short(m) for m in devs]
    july = CX.WinSet(JULY, ws[JULY], {})
    july_end = float(july.ts0.max())
    n_vocab = len(vocab)
    order = sorted((n for n in ws if n != JULY), key=lambda n: float(ws[n]["ts0"].min()))
    rows, per_source = [], {}
    for qn in order:
        Q = CX.WinSet(qn, ws[qn], {})
        gallery = sorted(set(july.dev.tolist()) & set(Q.dev.tolist()))
        if len(gallery) < 2:
            per_source[qn] = dict(skipped="fewer than two TVs in both")
            continue
        E = july.subset(np.isin(july.dev, gallery))
        Qg = Q.subset(np.isin(Q.dev, gallery))
        mask = E.bin_mask() & Qg.bin_mask()
        if not mask.any():
            per_source[qn] = dict(skipped="no bin covered in both")
            continue
        tt = time.time()
        days = sorted(set(ws[qn]["day"].tolist()))
        extra = dict(source=qn, first_day=days[0], last_day=days[-1],
                     days_after_enrol_min=(float(Qg.ts0.min()) - july_end) / S_PER_DAY,
                     days_after_enrol_max=(float(Qg.ts0.max()) - july_end) / S_PER_DAY)
        seed_rows = []
        for seed in SEEDS:
            hits = CX.run_condition(f"Jul_to_{qn}", E, Qg, gallery, mask, n_vocab,
                                    np.random.default_rng(seed))
            seed_rows += [dict(r, seed=seed, **extra) for r in CX.tally_rows(
                f"Jul_to_{qn}", "2026-07", days[0], f"2026-07->{days[0]}",
                hits, gallery, names, mask.sum())]
        rows += W.mean_over_seeds(seed_rows)
        per_source[qn] = dict(
            **extra, gallery=[names[g] for g in gallery], bins_kept=int(mask.sum()),
            query_windows=int(Qg.dev.size),
            query_blocks_per_device={names[g]: len([k for k in Qg.blocks() if k[0] == g])
                                     for g in gallery},
            seconds=time.time() - tt)
        print(f"[{qn}] {days[0]}..{days[-1]} gallery {len(gallery)} bins {mask.sum()} "
              f"{time.time() - tt:.0f}s peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
        gc.collect()
    with open(OUT["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CX.CSV_COLS) + list(CSV_EXTRA) + ["recall_sd", "n_seeds"])
        w.writeheader()
        w.writerows(rows)
    X.write_json(OUT["summary"], dict(
        seeds=list(SEEDS), enrol=JULY, july_last_window=july_end,
        k_grid=[CX.klabel(k) for k in CX.K_GRID], t_grid=list(CX.T_GRID),
        n_trials=CX.N_TRIALS, sources=per_source,
        recording=recording_totals(order and [n for n in order if n != "live"] or []),
        peak_rss_mb=X.peak_rss_mb()))
    print(f"[out] {OUT['csv']} ({len(rows)} rows)", file=sys.stderr)
    return 0


def evaluate_pairs() -> int:
    """Recognition for every (enrolment file, query file) pair: one labelled
    block of each TV in the enrolment file, macro recall over the TVs heard in
    both, mean over PAIR_SEEDS; gap = days between the two files, negative
    when the query file precedes the enrolment file."""
    CX.K_GRID, CX.T_GRID = PAIR_K_GRID, PAIR_T_GRID
    vocab: dict = {}
    ws = B.load_b_sources(vocab)
    order = sorted(ws, key=lambda n: float(ws[n]["ts0"].min()))
    sets = {n: CX.WinSet(n, ws[n], {}) for n in order}
    n_vocab = len(vocab)
    rows = []
    t0 = time.time()
    for en in order:
        for qn in order:
            if qn == en:
                continue
            E0, Q0 = sets[en], sets[qn]
            gallery = sorted(set(E0.dev.tolist()) & set(Q0.dev.tolist()))
            if len(gallery) < 2:
                continue
            E = E0.subset(np.isin(E0.dev, gallery))
            Q = Q0.subset(np.isin(Q0.dev, gallery))
            mask = E.bin_mask() & Q.bin_mask()
            if not mask.any():
                continue
            per = {}
            for seed in PAIR_SEEDS:
                hits = CX.run_condition("pair", E, Q, gallery, mask, n_vocab,
                                        np.random.default_rng(seed))
                for f in CX.FEATURES:
                    for K in PAIR_K_GRID:
                        rec = [hits[(f, K, 1, m)][0] / hits[(f, K, 1, m)][1]
                               for m in gallery if hits[(f, K, 1, m)][1]]
                        if rec:
                            per.setdefault((f, K), []).append(float(np.mean(rec)))
            ahead = float(Q.ts0.min()) - float(E.ts0.max())
            gap = (ahead if ahead >= 0 else
                   min(float(Q.ts0.max()) - float(E.ts0.min()), 0.0)) / S_PER_DAY
            for (f, K), v in per.items():
                rows.append(dict(enrol_source=en, query_source=qn, gap_days=gap, feature=f,
                                 K=CX.klabel(K), n_gallery=len(gallery),
                                 chance=1.0 / len(gallery), bins_kept=int(mask.sum()),
                                 recall=float(np.mean(v)),
                                 recall_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                 n_seeds=len(v)))
        print(f"[enrol {en}] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
              file=sys.stderr)
        gc.collect()
    with open(OUT["pairs"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PAIR_COLS))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {OUT['pairs']} ({len(rows)} rows)", file=sys.stderr)
    summarise_pairs()
    return 0


def summarise_pairs() -> None:
    """Recognition against the days between enrolment and query file, from
    c5_continuous_pairs.csv: mean recall per gap bin and Spearman's rho."""
    from scipy.stats import spearmanr
    with open(OUT["pairs"], newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if all(r[k] == v for k, v in PAIR_SUMMARY_WHERE.items())]
    gap = np.array([abs(float(r["gap_days"])) for r in rows])
    rec = np.array([float(r["recall"]) for r in rows])
    bins = []
    for lo, hi in zip(PAIR_GAP_EDGES_D[:-1], PAIR_GAP_EDGES_D[1:]):
        v = rec[(gap >= lo) & (gap < hi)]
        bins.append(dict(gap_lo_d=lo, gap_hi_d=hi, n_pairs=int(v.size),
                         recall_mean=float(v.mean()) if v.size else None,
                         recall_sd=float(v.std(ddof=1)) if v.size > 1 else None))
    rho = spearmanr(gap, rec)
    near = gap < PAIR_RHO_SPLIT_D
    rho_near, rho_far = spearmanr(gap[near], rec[near]), spearmanr(gap[~near], rec[~near])
    X.write_json(OUT["pairs_summary"], dict(
        where=PAIR_SUMMARY_WHERE, seeds=list(PAIR_SEEDS), n_pairs=int(rec.size),
        gap_max_d=float(gap.max()), bins=bins,
        spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
        rho_split_d=PAIR_RHO_SPLIT_D,
        spearman_rho_within=float(rho_near.statistic), spearman_p_within=float(rho_near.pvalue),
        spearman_rho_beyond=float(rho_far.statistic), spearman_p_beyond=float(rho_far.pvalue)))
    print(f"[out] {OUT['pairs_summary']} ({rec.size} pairs)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    names = register_rotations()
    if a.dry_run:
        for n in [JULY] + names + ["live"]:
            p = C.SOURCES[n]
            print(f"{n:14s} {p.stat().st_size / 1e6:8.0f} MB  cached={C.coverage_path(n).exists()}")
        return 0
    for n in names:
        B.SOURCE_MONTH.setdefault(n, n[3:7] + "-" + n[7:9])
    if a.build:
        build(names)
    if a.eval:
        rc = evaluate()
        if rc:
            return rc
    if a.pairs:
        return evaluate_pairs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
