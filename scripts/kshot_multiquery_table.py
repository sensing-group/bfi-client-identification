"""X6 / B-m1: accuracy vs K (labelled windows) and vs T (unlabelled query windows)
from the existing figures/bfi_kshot.csv and figures/bfi_multi_query.csv, plus
per-query cost and parameter counts for 1-NN, the SupCon MLP and a BFId-style head.

Both source CSVs were produced on NTU-Fi with the MEAN-ONLY 87-d feature
(load_ntufi pkt_reduce='mean'), not the 174-d mean+std the paper text describes;
cost is therefore reported for D=87, D=174 and He D=118.

Usage: python3 scripts/kshot_multiquery_table.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import pandas as pd

import joint_defense_common as C

KSHOT_CSV = C.FIG_DIR / "bfi_kshot.csv"
MQ_CSV = C.FIG_DIR / "bfi_multi_query.csv"
OUT_CSV = C.FIG_DIR / "bfi_kshot_multiquery_table.csv"
OUT_JSON = C.FIG_DIR / "bfi_attack_cost.json"
BFID_HIDDEN = 128               # demo_rebuttal_bfid.HIDDEN_DIM
DIMS = {"ntufi_csv_mean_only": 87, "ntufi_paper_mean_std": 174, "he_pair0": 118}
N_SUPPORT = {"cohort_out_ntufi": 7, "closed_set_ntufi": 14, "cohort_out_he": 19}
N_TIMING_QUERIES = 10_000
TIMING_REPEATS = 5


def tables():
    k = pd.read_csv(KSHOT_CSV)
    rows = []
    for (d, b, arch, K), g in k.groupby(["defense", "b", "encoder_arch", "K"]):
        per_seed = g.groupby("seed")["attacker_acc"].mean()
        rows.append(dict(table="K", defense=d, b=b, encoder=arch, K=K, T=1, mean=g.attacker_acc.mean(),
                         sd_all_runs=g.attacker_acc.std(ddof=1), sd_over_encoder_seeds=per_seed.std(ddof=1),
                         n_runs=len(g), chance=g.chance.iloc[0]))
    m = pd.read_csv(MQ_CSV)
    for (d, b, T), g in m.groupby(["defense", "b", "T"]):
        rows.append(dict(table="T", defense=d, b=b, encoder="baseline", K=1, T=T, mean=g.mean_acc.mean(),
                         sd_all_runs=float(np.sqrt((g.std_acc ** 2).mean() + g.mean_acc.var(ddof=1))),
                         sd_over_encoder_seeds=g.mean_acc.std(ddof=1), n_runs=int(g.n_seeds.sum()),
                         chance=g.chance.iloc[0]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def costs():
    import torch
    import torch.nn as nn
    from bfi.framework.encoder import build_encoder

    torch.set_num_threads(1)
    out = {}
    for dname, D in DIMS.items():
        mlp = build_encoder("mlp", D, hidden_dim=C.SUPCON_HIDDEN, embed_dim=C.SUPCON_EMBED).eval()
        mlp2 = build_encoder("mlpv2", D, hidden_dim=C.SUPCON_HIDDEN, embed_dim=C.SUPCON_EMBED).eval()
        n_cls = N_SUPPORT["closed_set_ntufi"] if dname.startswith("ntufi") else 39
        bfid = nn.Sequential(nn.Linear(D, BFID_HIDDEN), nn.ReLU(), nn.Linear(BFID_HIDDEN, n_cls)).eval()
        n_sup = N_SUPPORT["cohort_out_he"] if dname == "he_pair0" else N_SUPPORT["cohort_out_ntufi"]
        X = torch.randn(N_TIMING_QUERIES, D)
        S = np.random.default_rng(0).standard_normal((n_sup, D))
        Xn = X.numpy()

        def t_nn():
            (Xn / np.linalg.norm(Xn, axis=1, keepdims=True)) @ (S / np.linalg.norm(S, axis=1, keepdims=True)).T

        def t_mod(mod):
            def f():
                with torch.no_grad():
                    mod(X)
            return f

        def timeit(f):
            f()
            ts = []
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts) / N_TIMING_QUERIES * 1e6)

        params = lambda m: int(sum(p.numel() for p in m.parameters()))
        out[dname] = dict(
            D=D,
            nn1=dict(params=0, macs_per_query=D * n_sup, us_per_query=timeit(t_nn), supports=n_sup),
            supcon_mlp=dict(params=params(mlp),
                            macs_per_query=D * C.SUPCON_HIDDEN + C.SUPCON_HIDDEN * C.SUPCON_EMBED + C.SUPCON_EMBED * n_sup,
                            us_per_query=timeit(t_mod(mlp))),
            supcon_mlpv2=dict(params=params(mlp2), us_per_query=timeit(t_mod(mlp2))),
            bfid_head=dict(params=params(bfid), classes=n_cls,
                           macs_per_query=D * BFID_HIDDEN + BFID_HIDDEN * n_cls, us_per_query=timeit(t_mod(bfid)),
                           note="closed-set softmax head; needs labelled training data of every target"),
            multi_query_note="T-window consensus costs T encoder passes + one cosine against n_sup supports",
        )
    return out


def main():
    C.set_vm_limit()
    rows = tables()
    cost = costs()
    OUT_JSON.write_text(json.dumps(cost, indent=1))
    for r in rows:
        if r["b"] == C.B_HEADLINE:
            print(f"{r['table']} {r['defense']:>13s} {r['encoder']:>8s} K={r['K']} T={r['T']:>2d} "
                  f"{r['mean']:.4f} sd_runs={r['sd_all_runs']:.4f} sd_enc={r['sd_over_encoder_seeds']:.4f} n={r['n_runs']}")
    print(json.dumps(cost, indent=1))
    print(f"wrote {OUT_CSV} and {OUT_JSON}; peak RSS {C.peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
