#!/usr/bin/env python3
"""X8 (reviewer A: MAC grouping). Can windows of BFI frames be grouped by
device WITHOUT the source MAC?

Windows (non-overlapping 32 frames, as X4) of the 9 eligible MACs are
clustered on MAC-free features, and the clusters are scored against the MAC
labels (purity, adjusted Rand index, NMI, per-MAC dominant-cluster share).

Feature variants
  givens        decoded (phi_11, psi_21) 5x32 window statistics, z-scored
  givens_rssi   givens + window RSSI mean/std (z-scored, appended)
  rssi_only     window RSSI mean/std alone
  body_hist     the paper's 64-bin byte histogram (L2-normalised)
  geometry_only one-hot report geometry (format, Nc, Nr, codebook, SU/MU,
                bandwidth); a reference for what the FORMAT alone groups

Regimes
  natural   every window, k = number of MACs (class sizes 2 .. 12,979)
  balanced  up to BAL_PER_MAC windows per MAC, BAL_SEEDS resamples
Both on the 9 eligible MACs and on the 6 ground-truth-labelled devices.

Below-cut MACs (report only): every window of a MAC outside the eligible
set (including 'short' windows of 8..31 frames) is assigned to the nearest
k-means centroid fitted on the eligible windows; the table lists the
cluster's majority MAC, whether the window falls inside that cluster's
95th-percentile radius, the 1-NN eligible MAC, and whether the report
geometry matches that MAC's. No claim about which physical device a
random MAC belongs to is made.

Outputs: figures/x8_clustering.json, figures/x8_clustering_summary.csv,
         figures/x8_below_cut_assignment.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import xres_common as X
import xres_eval as E
from x4_mobility import eligible_macs

OUT_JSON = X.FIG_DIR / "x8_clustering.json"
OUT_SUM = X.FIG_DIR / "x8_clustering_summary.csv"
OUT_BELOW = X.FIG_DIR / "x8_below_cut_assignment.csv"
BAL_PER_MAC = 200
BAL_SEEDS = [0, 1, 2, 3, 4]
KMEANS_N_INIT = 10
RADIUS_Q = 0.95
SEED = 0


def features(w: dict, sel: np.ndarray) -> dict:
    mu, sd = E.zscore_fit(w["givens"][sel])
    G = E.zscore_apply(w["givens"].astype(np.float64), mu, sd)
    rmu, rsd = E.zscore_fit(w["rssi"][sel])
    R = E.zscore_apply(w["rssi"].astype(np.float64), rmu, rsd)
    H = w["hist"].astype(np.float64)
    H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    geo = np.eye(int(w["geom"].max()) + 1)[w["geom"]]
    return dict(givens=G, givens_rssi=np.hstack([G, R]), rssi_only=R,
                body_hist=H, geometry_only=geo)


def purity(y: np.ndarray, c: np.ndarray) -> float:
    tot = 0
    for k in np.unique(c):
        _, cnt = np.unique(y[c == k], return_counts=True)
        tot += cnt.max()
    return tot / len(y)


def score(F: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict:
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed)
    c = km.fit_predict(F)
    per = {}
    for m in np.unique(y):
        cl, cnt = np.unique(c[y == m], return_counts=True)
        dom = cl[np.argmax(cnt)]
        members, mc = np.unique(y[c == dom], return_counts=True)
        per[str(m)] = dict(
            n=int((y == m).sum()),
            dominant_share=float(cnt.max() / cnt.sum()),
            dominant_cluster_majority=str(members[np.argmax(mc)]),
            macs_in_dominant_cluster=int(len(members)))
    return dict(purity=purity(y, c), ari=adjusted_rand_score(y, c),
                nmi=normalized_mutual_info_score(y, c), per_mac=per), km, c


def run_set(Fs: dict, w: dict, macs: list[str], tag: str) -> dict:
    sel = np.isin(w["mac"], macs) & ~w["short"]
    y = w["mac"][sel]
    out = {}
    for name, F in Fs.items():
        nat, _, _ = score(F[sel], y, len(macs), SEED)
        bal = []
        for s in BAL_SEEDS:
            rng = np.random.default_rng(s)
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == m),
                           min(BAL_PER_MAC, int((y == m).sum())),
                           replace=False) for m in macs])
            r, _, _ = score(F[sel][idx], y[idx], len(macs), s)
            bal.append(r)
        out[name] = dict(
            natural=nat,
            balanced=dict(
                purity=float(np.mean([b["purity"] for b in bal])),
                purity_sd=float(np.std([b["purity"] for b in bal])),
                ari=float(np.mean([b["ari"] for b in bal])),
                ari_sd=float(np.std([b["ari"] for b in bal])),
                nmi=float(np.mean([b["nmi"] for b in bal])),
                per_mac_dominant_share={
                    m: float(np.mean([b["per_mac"][m]["dominant_share"]
                                      for b in bal])) for m in macs}))
        print(f"  [{tag}] {name:14s} natural P={nat['purity']:.3f} "
              f"ARI={nat['ari']:.3f} | balanced P="
              f"{out[name]['balanced']['purity']:.3f} ARI="
              f"{out[name]['balanced']['ari']:.3f}", file=sys.stderr)
    return out


def below_cut(Fs: dict, w: dict, macs: list[str]) -> list[dict]:
    sel = np.isin(w["mac"], macs) & ~w["short"]
    below = np.flatnonzero(~np.isin(w["mac"], macs))
    geom_table = list(w["geom_table"])
    mac_geoms = {m: set(w["geom"][sel & (w["mac"] == m)].tolist())
                 for m in macs}
    rows = []
    for name in ("givens", "givens_rssi"):
        F = Fs[name]
        _, km, c = score(F[sel], w["mac"][sel], len(macs), SEED)
        ysel = w["mac"][sel]
        Fe = F[sel]
        d_mem = np.linalg.norm(Fe - km.cluster_centers_[c], axis=1)
        radius = {k: float(np.quantile(d_mem[c == k], RADIUS_Q))
                  for k in np.unique(c)}
        major = {}
        for k in np.unique(c):
            mm, cnt = np.unique(ysel[c == k], return_counts=True)
            major[k] = (str(mm[np.argmax(cnt)]), float(cnt.max() / cnt.sum()))
        for i in below:
            dc = np.linalg.norm(km.cluster_centers_ - F[i], axis=1)
            k = int(np.argmin(dc))
            nn = int(np.argmin(np.linalg.norm(Fe - F[i], axis=1)))
            nn_mac = str(ysel[nn])
            rows.append(dict(
                feature=name, mac=str(w["mac"][i]),
                mac_class=X.mac_group(str(w["mac"][i]))[1],
                day=str(w["day"][i]), short_window=bool(w["short"][i]),
                geometry=geom_table[int(w["geom"][i])],
                cluster=k, cluster_majority_mac=major[k][0],
                cluster_majority_share=major[k][1],
                dist_to_centroid=float(dc[k]),
                cluster_radius_q95=radius[k],
                inside_q95=bool(dc[k] <= radius[k]),
                nn_eligible_mac=nn_mac,
                geometry_shared_with_nn=int(w["geom"][i]) in
                mac_geoms[nn_mac]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    if a.dry_run:
        print(X.dry_run_report())
        return 0
    t0 = time.time()
    w = X.build_windows()
    macs = eligible_macs(w)
    labelled = [m for m in macs if m[:8] in X.DEVICE_GT]
    sel = np.isin(w["mac"], macs) & ~w["short"]
    Fs = features(w, sel)
    res = dict(eligible=macs, labelled=labelled,
               eligible9=run_set(Fs, w, macs, "9 eligible"),
               labelled6=run_set(Fs, w, labelled, "6 labelled"))
    X.log_rss("clustering")
    rows = below_cut(Fs, w, macs)
    res["below_cut"] = rows
    with open(OUT_BELOW, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"[out] {OUT_BELOW}", file=sys.stderr)
    srows = []
    for setname in ("eligible9", "labelled6"):
        for feat, v in res[setname].items():
            srows.append(dict(
                set=setname, feature=feat,
                natural_purity=v["natural"]["purity"],
                natural_ari=v["natural"]["ari"],
                natural_nmi=v["natural"]["nmi"],
                balanced_purity=v["balanced"]["purity"],
                balanced_purity_sd=v["balanced"]["purity_sd"],
                balanced_ari=v["balanced"]["ari"],
                balanced_ari_sd=v["balanced"]["ari_sd"],
                balanced_nmi=v["balanced"]["nmi"],
                **{f"bal_dom_{m[:8]}": s for m, s in
                   v["balanced"]["per_mac_dominant_share"].items()}))
    keys = sorted({k for r in srows for k in r},
                  key=lambda k: list(srows[0]).index(k)
                  if k in srows[0] else 99)
    with open(OUT_SUM, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(srows)
    print(f"[out] {OUT_SUM}", file=sys.stderr)
    res["runtime_s"] = time.time() - t0
    res["peak_rss_mb"] = X.log_rss("end")
    X.write_json(OUT_JSON, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
