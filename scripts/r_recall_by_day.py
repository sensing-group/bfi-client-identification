#!/usr/bin/env python3
"""Per-day recall from ONE enrolled session per device (Figure 1, right).

Same protocol as r_eval.py (other_day_any, feature "angles", T = 1), with the
enrollment fixed instead of drawn per episode: every labelled device is
enrolled once, from the earliest day on which it has a (session, day) block
of at least K_ENROL windows (the largest such block of that day). The gallery
is all six devices. For each device and each other day on which it sounds,
every window of that day is a query (windows of the enrolled session itself
are excluded, so a session that crosses midnight does not query itself);
K_ENROL windows are drawn at random from each enrolled block, N_TRIALS times,
and recall is the hit rate over windows and trials.

Outputs (figures/): r_recall_by_day.csv (one row per device and query day)
and r_recall_by_day_summary.json (enrolled blocks, capture days, days with
no data, chance).

--drift repeats the same per-day protocol under several decision variants
(K = 16 or all enrolled windows; T = 1 or the mean of 16 consecutive query
windows, non-overlapping within a block; 1-NN over the enrolled windows or
the distance to their centroid; and the format-only 1-NN with ties scored as
the expected hit) and writes r_recall_by_day_drift.csv and
r_recall_by_day_drift_summary.json. The published strip uses K = 16, T = 1,
1-NN. Distances use the Gram expansion, so no (queries x enrolled x 320)
tensor is ever built.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import r_common as R
import r_eval as E
import xres_common as X

K_ENROL = 16
N_TRIALS = 50
SEED = 0
FEATURE = "angles"
OUT_CSV = R.FIG_DIR / "r_recall_by_day.csv"
OUT_JSON = R.FIG_DIR / "r_recall_by_day_summary.json"
OUT_DRIFT_CSV = R.FIG_DIR / "r_recall_by_day_drift.csv"
OUT_DRIFT_JSON = R.FIG_DIR / "r_recall_by_day_drift_summary.json"

K_ALL = 0                                  # every window of the enrolled block
T_LONG = 16                                # consecutive query windows averaged
# (feature, K, T, rule); the first entry is the published strip.
DRIFT_VARIANTS = (
    ("angles", K_ENROL, 1, "nn"),
    ("angles", K_ALL, 1, "nn"),
    ("angles", K_ENROL, T_LONG, "nn"),
    ("angles", K_ALL, T_LONG, "nn"),
    ("angles", K_ALL, 1, "centroid"),
    ("angles", K_ALL, T_LONG, "centroid"),
    ("format", K_ALL, 1, "nn"),
)
DRIFT_FOCUS_DEVICE = "AppleTV-LR"


def sq_dist(Q: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Squared euclidean distances (n_q x n_g) via the Gram expansion."""
    d2 = (Q * Q).sum(1)[:, None] + (G * G).sum(1)[None, :] - 2.0 * (Q @ G.T)
    return np.maximum(d2, 0.0)


def query_sets(tb: E.Table, g: int, day: str, enrol_session: int,
               T: int) -> list[np.ndarray]:
    """Query vectors (index arrays of T consecutive windows) for device g on day."""
    out = []
    for b in tb.on_day.get((g, day), []):
        if b[1] == enrol_session:
            continue
        ii = tb.blocks[b]
        for s in range(0, ii.size - T + 1, T):
            out.append(ii[s:s + T])
    return out


def decide(Q: np.ndarray, gal: dict[int, np.ndarray], rule: str,
           true_dev: int) -> tuple[np.ndarray, Counter]:
    """Per-query hit probability and fractional prediction counts."""
    devs = list(gal)
    if rule == "nn":
        d = np.stack([np.sqrt(sq_dist(Q, gal[h]).min(1)) for h in devs], 1)
    elif rule == "centroid":
        d = np.stack([np.sqrt(((Q - gal[h].mean(0)) ** 2).sum(1)) for h in devs], 1)
    else:
        raise ValueError(rule)
    tie = d <= d.min(1, keepdims=True) + E.TIE_TOL      # ties share the hit
    share = tie / tie.sum(1, keepdims=True)
    hit = share[:, devs.index(true_dev)]
    counts = Counter({h: float(share[:, j].sum()) for j, h in enumerate(devs)})
    return hit, counts


def run_drift(tb: E.Table, enrol: dict, n_trials: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    gallery = list(tb.devs)
    rows = []
    for feat, K, T, rule in DRIFT_VARIANTS:
        Z = tb.Z if feat == "angles" else tb.F
        trials = n_trials if K else 1
        hits: dict[tuple[int, str], list[float]] = defaultdict(list)
        preds: dict[tuple[int, str], Counter] = defaultdict(Counter)
        nq: dict[tuple[int, str], int] = {}
        for _ in range(trials):
            gal = {h: Z[tb.blocks[enrol[h]] if not K else
                        rng.choice(tb.blocks[enrol[h]], K, replace=False)]
                   for h in gallery}
            for g in gallery:
                for day in tb.days:
                    if day == enrol[g][2]:
                        continue
                    qs = query_sets(tb, g, day, enrol[g][1], T)
                    nq[(g, day)] = len(qs)
                    if not qs:
                        continue
                    Q = np.stack([Z[q].mean(0) for q in qs])
                    hit, counts = decide(Q, gal, rule, g)
                    hits[(g, day)].append(float(hit.mean()))
                    preds[(g, day)].update(counts)
        X.log_rss(f"drift {feat} K={K or 'all'} T={T} {rule}")
        for (g, day), n in sorted(nq.items(), key=lambda x: (x[0][0], x[0][1])):
            wrong = {tb.names[h]: c for h, c in preds[(g, day)].items() if h != g}
            top = max(wrong, key=wrong.get) if wrong else ""
            total = sum(preds[(g, day)].values())
            rows.append(dict(
                gap_s=int(R.SESSION_GAP_S), feature=feat, K=E.klabel(K), T=T, rule=rule,
                device=tb.names[g], enrol_day=enrol[g][2],
                enrol_windows=int(tb.blocks[enrol[g]].size), query_day=day,
                query_windows=int(sum(tb.blocks[b].size for b in tb.on_day[(g, day)]
                                      if b[1] != enrol[g][1])),
                n_queries=n, n_trials=trials,
                recall=float(np.mean(hits[(g, day)])) if n else float("nan"),
                top_wrong=top,
                top_wrong_share=float(wrong[top] / total) if top else 0.0,
                predicted=json.dumps({tb.names[h]: round(c / trials, 2)
                                      for h, c in preds[(g, day)].items()})))
    return rows


def drift_summary(rows: list[dict]) -> dict:
    def key(r):
        return f"{r['feature']} K={r['K']} T={r['T']} {r['rule']}"
    focus = defaultdict(dict)
    minima = defaultdict(dict)
    for r in rows:
        if np.isnan(r["recall"]):
            continue
        if r["device"] == DRIFT_FOCUS_DEVICE:
            focus[key(r)][r["query_day"]] = dict(
                recall=round(r["recall"], 4), n_queries=r["n_queries"],
                top_wrong=r["top_wrong"], top_wrong_share=round(r["top_wrong_share"], 3))
        cur = minima[key(r)].get(r["device"])
        if cur is None or r["recall"] < cur["recall"]:
            minima[key(r)][r["device"]] = dict(recall=round(r["recall"], 4),
                                               day=r["query_day"], n_queries=r["n_queries"])
    return dict(focus_device=DRIFT_FOCUS_DEVICE, per_day=focus, per_device_minimum=minima)


def pick_enrol_block(tb: E.Table, dev: int) -> tuple:
    """Earliest day with a block of >= K_ENROL windows; largest block that day."""
    for day in tb.days:
        cands = [b for b in tb.on_day.get((dev, day), [])
                 if tb.blocks[b].size >= K_ENROL]
        if cands:
            return max(cands, key=lambda b: tb.blocks[b].size)
    raise RuntimeError(f"{tb.names[dev]}: no block with {K_ENROL} windows")


def day_range(days: list[str]) -> list[str]:
    d0 = dt.date.fromisoformat(days[0])
    d1 = dt.date.fromisoformat(days[-1])
    return [(d0 + dt.timedelta(n)).isoformat() for n in range((d1 - d0).days + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the enrolled blocks and query counts only")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--drift", action="store_true",
                    help="run the decision-variant sweep instead of the strip data")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    tb = E.Table(w)
    X.log_rss("table")

    enrol = {g: pick_enrol_block(tb, g) for g in tb.devs}
    gallery = list(tb.devs)
    queries: dict[tuple[int, str], np.ndarray] = {}
    for g in gallery:
        for day in tb.days:
            ii = np.concatenate([tb.blocks[b] for b in tb.on_day.get((g, day), [])
                                 if b[1] != enrol[g][1]] or [np.empty(0, int)])
            if day != enrol[g][2] and ii.size:
                queries[(g, day)] = ii
    if a.dry_run:
        print(json.dumps(dict(
            windows=int(len(w["dev"])),
            enrolled={tb.names[g]: dict(session=int(b[1]), day=b[2],
                                        windows=int(tb.blocks[b].size))
                      for g, b in enrol.items()},
            queries={f"{tb.names[g]}|{d}": int(v.size) for (g, d), v in queries.items()},
            est_distance_mb=max(v.size for v in queries.values()) * K_ENROL
            * len(gallery) * 8 / 1e6), indent=2))
        return 0

    if a.drift:
        rows = run_drift(tb, enrol, a.trials)
        E.write_csv(OUT_DRIFT_CSV, rows)
        X.write_json(OUT_DRIFT_JSON, dict(
            protocol="as r_recall_by_day.csv (one fixed enrolled block per device, "
                     "all-six gallery) under each (feature, K, T, rule) variant; "
                     "T=16 queries are non-overlapping runs of 16 consecutive windows "
                     "within a block; ties share the hit",
            published_strip=dict(zip(("feature", "K", "T", "rule"), DRIFT_VARIANTS[0])),
            variants=[dict(zip(("feature", "K", "T", "rule"), v)) for v in DRIFT_VARIANTS],
            trials=a.trials, seed=SEED, chance=1.0 / len(gallery),
            enrolled={tb.names[g]: dict(session=int(b[1]), day=b[2],
                                        windows=int(tb.blocks[b].size))
                      for g, b in enrol.items()},
            **drift_summary(rows),
            runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
        print(f"[done] drift {len(rows)} rows {time.time() - t0:.0f}s peak RSS "
              f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
        return 0

    rng = np.random.default_rng(SEED)
    hits: dict[tuple[int, str], list[float]] = defaultdict(list)
    preds: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for _ in range(a.trials):
        G = {g: tb.Z[rng.choice(tb.blocks[enrol[g]], K_ENROL, replace=False)]
             for g in gallery}
        for (g, day), qi in queries.items():
            Q = tb.Z[qi]                                   # (n, 320)
            d = np.stack([np.sqrt(((Q[:, None, :] - G[h][None, :, :]) ** 2)
                                  .sum(2)).min(1) for h in gallery], axis=1)
            p = np.argmin(d, axis=1)
            hits[(g, day)].append(float(np.mean(np.array(gallery)[p] == g)))
            preds[(g, day)].update(tb.names[gallery[k]] for k in p)
    X.log_rss("trials")

    rows = []
    for (g, day), h in sorted(hits.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(dict(
            gap_s=int(R.SESSION_GAP_S), feature=FEATURE, K=K_ENROL, T=1,
            device=tb.names[g], enrol_session=int(enrol[g][1]),
            enrol_day=enrol[g][2], enrol_windows=int(tb.blocks[enrol[g]].size),
            query_day=day, query_windows=int(queries[(g, day)].size),
            n_trials=a.trials, chance=1.0 / len(gallery),
            recall=float(np.mean(h)),
            predicted=json.dumps(dict(preds[(g, day)]))))
    E.write_csv(OUT_CSV, rows)
    days = day_range(tb.days)
    X.write_json(OUT_JSON, dict(
        protocol="other_day_any with one fixed enrolled block per device; "
                 "all windows of each other day are queries; T=1",
        feature=FEATURE, K=K_ENROL, trials=a.trials, seed=SEED,
        gallery=[tb.names[g] for g in gallery], chance=1.0 / len(gallery),
        capture_days=days, days_without_data=[d for d in days if d not in tb.days],
        sounding_days={tb.names[g]: sorted({b[2] for b in tb.blocks if b[0] == g})
                       for g in gallery},
        enrolled={tb.names[g]: dict(session=int(b[1]), day=b[2],
                                    windows=int(tb.blocks[b].size))
                  for g, b in enrol.items()},
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {len(rows)} rows {time.time() - t0:.0f}s peak RSS "
          f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
