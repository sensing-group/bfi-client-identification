#!/usr/bin/env python3
"""K=1 evaluators over the residential window table (xres_common.build_windows).

Protocols (non-overlapping 32-frame windows, so a query never shares a frame
with its support window, unlike the paper's random-start windows):

* pooled:   per trial one random support window per MAC from the whole
            archive, up to QUERIES_POOLED other windows per MAC as queries.
* crossday: per trial and per query day d, the gallery holds one random
            window per MAC drawn from days != d; queries are up to
            QUERIES_CROSS windows per MAC from day d. Only MACs with
            windows on >= 2 days are queried.

Prediction is argmax similarity (cosine) or argmin distance (euclidean);
ties are broken uniformly at random, which matters for the geometry-only
baseline.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

N_TRIALS = 50
QUERIES_POOLED = 50
QUERIES_CROSS = 30
TIE_TOL = 1e-12


def _scores(Q: np.ndarray, Gal: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-300)
        gn = Gal / np.maximum(np.linalg.norm(Gal, axis=1, keepdims=True),
                              1e-300)
        return qn @ gn.T
    d2 = ((Q[:, None, :] - Gal[None, :, :]) ** 2).sum(-1)
    return -d2


def _predict(S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    best = S.max(axis=1, keepdims=True)
    out = np.empty(S.shape[0], dtype=np.int64)
    for i in range(S.shape[0]):
        cand = np.flatnonzero(S[i] >= best[i, 0] - TIE_TOL)
        out[i] = cand[0] if cand.size == 1 else rng.choice(cand)
    return out


def pooled(F: np.ndarray, mac: np.ndarray, macs: list[str], metric: str,
           seed: int = 0, n_trials: int = N_TRIALS,
           n_queries: int = QUERIES_POOLED) -> dict:
    rng = np.random.default_rng(seed)
    idx = {m: np.flatnonzero(mac == m) for m in macs}
    tally = {m: [0, 0] for m in macs}
    conf = np.zeros((len(macs), len(macs)), dtype=np.int64)
    for _ in range(n_trials):
        sup = {m: int(rng.choice(idx[m])) for m in macs}
        Gal = F[[sup[m] for m in macs]]
        for mi, m in enumerate(macs):
            pool = idx[m][idx[m] != sup[m]]
            if pool.size == 0:
                continue
            q = rng.choice(pool, size=min(n_queries, pool.size),
                           replace=False)
            pred = _predict(_scores(F[q], Gal, metric), rng)
            tally[m][0] += int((pred == mi).sum())
            tally[m][1] += int(q.size)
            np.add.at(conf[mi], pred, 1)
    return _summarise(tally, macs, conf, chance={m: 1 / len(macs)
                                                 for m in macs})


def crossday(F: np.ndarray, mac: np.ndarray, day: np.ndarray,
             macs: list[str], metric: str, seed: int = 0,
             n_trials: int = N_TRIALS, n_queries: int = QUERIES_CROSS) -> dict:
    rng = np.random.default_rng(seed)
    days = sorted(set(day[np.isin(mac, macs)].tolist()))
    md = {(m, d): np.flatnonzero((mac == m) & (day == d))
          for m in macs for d in days}
    multi = [m for m in macs
             if sum(md[(m, d)].size > 0 for d in days) >= 2]
    tally = {m: [0, 0] for m in multi}
    chance = defaultdict(float)
    conf = np.zeros((len(macs), len(macs)), dtype=np.int64)
    per_day = defaultdict(lambda: [0, 0])
    for _ in range(n_trials):
        for d in days:
            gal_macs, gal_idx = [], []
            for m in macs:
                other = np.flatnonzero((mac == m) & (day != d))
                if other.size:
                    gal_macs.append(m)
                    gal_idx.append(int(rng.choice(other)))
            if len(gal_macs) < 2:
                continue
            Gal = F[gal_idx]
            for m in multi:
                pool = md[(m, d)]
                if pool.size == 0 or m not in gal_macs:
                    continue
                q = rng.choice(pool, size=min(n_queries, pool.size),
                               replace=False)
                pred = _predict(_scores(F[q], Gal, metric), rng)
                gi = gal_macs.index(m)
                hit = int((pred == gi).sum())
                tally[m][0] += hit
                tally[m][1] += int(q.size)
                chance[m] += q.size / len(gal_macs)
                per_day[d][0] += hit
                per_day[d][1] += int(q.size)
                for p in pred:
                    conf[macs.index(m), macs.index(gal_macs[p])] += 1
    out = _summarise(tally, multi, conf,
                     chance={m: chance[m] / max(1, tally[m][1])
                             for m in multi})
    out["confusion_labels"] = macs
    out["per_day"] = {d: v[0] / v[1] for d, v in per_day.items() if v[1]}
    return out


def _summarise(tally, macs, conf, chance) -> dict:
    per = {m: dict(recall=(t[0] / t[1]) if t[1] else None, hits=t[0],
                   queries=t[1], chance=chance.get(m))
           for m, t in tally.items()}
    hits = sum(t[0] for t in tally.values())
    n = sum(t[1] for t in tally.values())
    rec = [v["recall"] for v in per.values() if v["recall"] is not None]
    return dict(accuracy=hits / n if n else None, queries=n,
                macro_recall=float(np.mean(rec)) if rec else None,
                per_mac=per, confusion=conf.tolist())


def zscore_fit(F: np.ndarray):
    mu = np.nanmean(F, axis=0)
    sd = np.nanstd(F, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    return mu, sd


def zscore_apply(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((F - mu) / sd)
