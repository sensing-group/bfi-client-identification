#!/usr/bin/env python3
"""Section 6.3: the NTU-Fi proxy across walking conditions.

One window per held-out subject is enrolled from recording prefix "a"; every
window of prefixes "b" and "c" is a query, so enrollment and query never share
a condition. Same features, quantization and scaling as joint_defense_eval
(defense "none"): b = 4, scaler fitted on the calibration subjects' train
split, cosine 1-NN.

Usage: python3 scripts/ntu_cross_condition.py
Output: figures/ntu_cross_condition.json
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.preprocessing import StandardScaler

import joint_defense_common as C

N_SEEDS = 10
SEED_BASE = 20_000
SUPPORT_PREFIX = "a"
QUERY_PREFIXES = ("b", "c")
OUT = C.FIG_DIR / "ntu_cross_condition.json"


def main() -> None:
    d = C.load_ntufi_packets()
    x = C.ntufi_features(d["A"])
    y, split, prefix = d["y"], d["split"], d["prefix"]
    calib = y % 2 == 0
    held = ~calib
    q = C.quant_midpoint(x, float(x.min()), float(x.max()), C.B_HEADLINE)
    z = StandardScaler().fit(q[calib & (split == "train")]).transform(q[held])
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    y, p = y[held], prefix[held]
    query = np.isin(p, QUERY_PREFIXES)
    subjects = np.unique(y)
    acc = []
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(SEED_BASE + seed)
        sup = np.array([rng.choice(np.flatnonzero((y == s) & (p == SUPPORT_PREFIX)))
                        for s in subjects])
        pred = subjects[np.argmax(z[query] @ z[sup].T, axis=1)]
        acc.append(float(np.mean([np.mean(pred[y[query] == s] == s) for s in subjects])))
    out = dict(mean=float(np.mean(acc)), sd=float(np.std(acc, ddof=1)), seeds=N_SEEDS,
               chance=1.0 / subjects.size, per_seed=acc)
    OUT.write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
