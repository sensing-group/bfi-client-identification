#!/usr/bin/env python3
"""X5 (reviewer: cross-router; PI scope: cross-SOURCE calibration only).

Calibrate on the He real-BFI dataset (a different AP, a lab deployment) and
run the residential K=1 enrolment + query with that calibration; compare to
calibrating on the residential data itself.

Layout problem and how it is handled
------------------------------------
He ships decoded, unquantised angles as (N, 234 subcarriers, 6 antenna
pairs, [phi, psi]): a per-antenna-pair 2x1 decomposition at VHT 80 MHz
Ng=1. The residential reports are quantised Givens sequences with
per-device layouts (HE Nc=2/Nr=3 at Ng=4 with 250 subcarriers, VHT Nc=2
Nr=3 at 234, VHT MU Nc=1 Nr=3, ...). No coordinate-level correspondence
exists. Both sides are therefore reduced to the same layout-agnostic
representation (xres_common): the LEADING angle pair only (residential
phi_11/psi_21; He antenna pair 01), averaged into 32 equal-width bins
across the band, summarised over 32-frame windows as 5x32 statistics
(phi mean cos, mean sin, circular spread; psi mean, psi std). Only that
160-dim vector crosses the source boundary. A matched-quantisation variant
dequantises He angles to b_phi=6, b_psi=4 (the residential codebook-1 SU
depth) before the statistics.

Calibration variants (all scored with the X4 pooled and cross-day K=1
protocols on the same 9 eligible residential MACs):
  none            raw 160-dim features, euclidean 1-NN
  res_zscore      z-score fitted on the residential windows (= X4 givens)
  he_cr1_zscore   z-score fitted on He Conference Room 1 windows
  he_cr1q_zscore  same, He angles quantised to (6, 4) bits first
  he_all_zscore   z-score fitted on all five He rooms pooled
  he_cr1_supcon   He-CR1 z-score + SupCon MLP encoder trained on the 39 He
                  CR1 position classes (bfi.framework.OneShotAttacker,
                  paper hyper-parameters), cosine 1-NN in embedding space
  res_z_he_supcon residential z-score into the same He-trained encoder

An in-domain SupCon encoder is not trained: the residential cohort has 6
labelled devices (9 eligible MACs), too few for a cohort-disjoint
calibration split, and training on the enrolled devices' labels would no
longer be K=1.

Outputs: figures/x5_cross_source.json, figures/x5_cross_source.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time

import numpy as np

import xres_common as X
import xres_eval as E
from x4_mobility import eligible_macs

OUT_JSON = X.FIG_DIR / "x5_cross_source.json"
OUT_CSV = X.FIG_DIR / "x5_cross_source.csv"
HE_PRIMARY = "Conference Room 1"
HE_ROOMS = ["Conference Room 1", "Conference Room 2", "Lab 1", "Lab 2",
            "Rooftop"]
HE_PAIR = 0                     # antenna pair 01
HE_QUANT_BITS = (6, 4)          # (b_phi, b_psi), residential cb1 SU depth
HE_N_CALIB = 20                 # He sanity: lowest 20 classes calibrate
SUPCON_SEEDS = [1, 2, 3]
SUPCON_KW = dict(embed_dim=128, hidden_dim=256, contrastive="supcon",
                 temperature=0.1)
SUPCON_FIT = dict(epochs=200, batch_size=64, lr=1e-3)
SEED = 0


def quantise(a: np.ndarray, bits: int, kind: str) -> np.ndarray:
    """Midpoint quantiser matching xres_common._angle_to_rad."""
    if kind == "phi":
        step = np.pi / 2 ** (bits - 1)
        q = np.floor(np.mod(a, 2 * np.pi) / step)
        return q * step + step / 2
    step = np.pi / 2 ** (bits + 1)
    q = np.clip(np.floor(a / step), 0, 2 ** bits - 1)
    return q * step + step / 2


def he_windows(room: str, quant: bool = False):
    import torch
    root = X.HE_ROOT / room
    t = torch.load(str(root / "BFI.pth"), map_location="cpu",
                   weights_only=True, mmap=True)
    pair = t[:, :, HE_PAIR, :].numpy().astype(np.float64)
    y = torch.load(str(root / "y.pth"), map_location="cpu",
                   weights_only=True).numpy().astype(np.int64)
    del t
    gc.collect()
    phi, psi = pair[..., 0], pair[..., 1]
    del pair
    if quant:
        phi = quantise(phi, HE_QUANT_BITS[0], "phi")
        psi = quantise(psi, HE_QUANT_BITS[1], "psi")
    c, s, p = X.bin_pair(phi, psi)
    del phi, psi
    feats, labels = [], []
    T = X.WINDOW_FRAMES
    starts = np.flatnonzero(np.r_[True, np.diff(y) != 0])
    ends = np.r_[starts[1:], len(y)]
    for a, b in zip(starts, ends):
        for w in range((b - a) // T):
            sl = slice(a + w * T, a + (w + 1) * T)
            feats.append(X.givens_window_features(c[sl], s[sl], p[sl]))
            labels.append(int(y[a]))
    return np.asarray(feats), np.asarray(labels)


def train_supcon(F: np.ndarray, y: np.ndarray, seed: int):
    # xres_common put the `bfi` package on sys.path (release/vendor/ first).
    from bfi.framework import OneShotAttacker
    att = OneShotAttacker(encoder="mlp", **SUPCON_KW)
    att.fit(F.astype(np.float32), y, seed=seed, **SUPCON_FIT)
    return att


def evaluate(F, w, sel, macs, metric) -> dict:
    return dict(
        pooled=E.pooled(F[sel], w["mac"][sel], macs, metric, seed=SEED),
        crossday=E.crossday(F[sel], w["mac"][sel], w["day"][sel], macs,
                            metric, seed=SEED))


def he_sanity(Fh, yh) -> dict:
    """Common representation on He itself: cohort-out K=1 on the 19
    held-out CR1 positions, z-score from the 20 calibration positions."""
    cls = np.unique(yh)
    cal = np.isin(yh, cls[:HE_N_CALIB])
    mu, sd = E.zscore_fit(Fh[cal])
    Z = E.zscore_apply(Fh, mu, sd)
    held = [str(c) for c in cls[HE_N_CALIB:]]
    lab = yh.astype(str)
    r = E.pooled(Z[~cal], lab[~cal], held, "euclidean", seed=SEED)
    return dict(accuracy=r["accuracy"], chance=1 / len(held),
                n_classes=len(held), queries=r["queries"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    if a.dry_run:
        print(X.dry_run_report())
        for r in HE_ROOMS:
            p = X.HE_ROOT / r / "BFI.pth"
            print(r, f"{p.stat().st_size / 1e6:.0f} MB")
        return 0
    t0 = time.time()
    w = X.build_windows()
    macs = eligible_macs(w)
    sel = np.isin(w["mac"], macs) & ~w["short"]
    G = w["givens"].astype(np.float64)

    he = {}
    for room in HE_ROOMS:
        he[room] = he_windows(room)
        print(f"  He {room}: {he[room][0].shape}", file=sys.stderr)
        X.log_rss(f"He {room}")
    Fq, yq = he_windows(HE_PRIMARY, quant=True)
    Fh, yh = he[HE_PRIMARY]

    res = dict(eligible=macs, he_windows={r: int(v[0].shape[0])
                                          for r, v in he.items()},
               he_sanity=he_sanity(Fh, yh), variants={}, domain_gap={})
    scalers = dict(
        res_zscore=E.zscore_fit(G[sel]),
        he_cr1_zscore=E.zscore_fit(Fh),
        he_cr1q_zscore=E.zscore_fit(Fq),
        he_all_zscore=E.zscore_fit(np.concatenate([v[0] for v in
                                                   he.values()])),
    )
    res["variants"]["none"] = evaluate(np.nan_to_num(G), w, sel, macs,
                                       "euclidean")
    for name, (mu, sd) in scalers.items():
        Z = E.zscore_apply(G, mu, sd)
        res["variants"][name] = evaluate(Z, w, sel, macs, "euclidean")
        res["domain_gap"][name] = dict(
            median_abs_mean_z=float(np.median(np.abs(Z[sel].mean(0)))),
            median_std_z=float(np.median(Z[sel].std(0))))
        print(f"  {name}: pooled "
              f"{res['variants'][name]['pooled']['accuracy']:.3f} cross-day "
              f"{res['variants'][name]['crossday']['accuracy']:.3f}",
              file=sys.stderr)
    X.log_rss("zscore variants")

    mu_h, sd_h = scalers["he_cr1_zscore"]
    mu_r, sd_r = scalers["res_zscore"]
    Zh_he = E.zscore_apply(Fh, mu_h, sd_h)
    Zr_he = E.zscore_apply(G, mu_h, sd_h)
    Zr_res = E.zscore_apply(G, mu_r, sd_r)
    sup = {"he_cr1_supcon": [], "res_z_he_supcon": []}
    for seed in SUPCON_SEEDS:
        att = train_supcon(Zh_he, yh, seed)
        for name, Z in (("he_cr1_supcon", Zr_he),
                        ("res_z_he_supcon", Zr_res)):
            emb = att.embed(Z.astype(np.float32)).astype(np.float64)
            sup[name].append(evaluate(emb, w, sel, macs, "cosine"))
        del att
        gc.collect()
        X.log_rss(f"supcon seed {seed}")
    for name, runs in sup.items():
        res["variants"][name] = dict(
            seeds=SUPCON_SEEDS,
            pooled=dict(accuracy=float(np.mean(
                [r["pooled"]["accuracy"] for r in runs])),
                accuracy_per_seed=[r["pooled"]["accuracy"] for r in runs],
                macro_recall=float(np.mean(
                    [r["pooled"]["macro_recall"] for r in runs])),
                per_mac={m: dict(recall=float(np.mean(
                    [r["pooled"]["per_mac"][m]["recall"] for r in runs])))
                    for m in macs}),
            crossday=dict(accuracy=float(np.mean(
                [r["crossday"]["accuracy"] for r in runs])),
                accuracy_per_seed=[r["crossday"]["accuracy"] for r in runs],
                macro_recall=float(np.mean(
                    [r["crossday"]["macro_recall"] for r in runs])),
                per_mac={m: dict(recall=float(np.mean(
                    [r["crossday"]["per_mac"][m]["recall"] for r in runs])))
                    for m in runs[0]["crossday"]["per_mac"]}))

    rows = []
    for name, v in res["variants"].items():
        row = dict(variant=name,
                   pooled_acc=v["pooled"]["accuracy"],
                   pooled_macro=v["pooled"]["macro_recall"],
                   crossday_acc=v["crossday"]["accuracy"],
                   crossday_macro=v["crossday"]["macro_recall"])
        for m in macs:
            row[f"pooled_{m[:8]}"] = v["pooled"]["per_mac"][m]["recall"]
            cm = v["crossday"]["per_mac"].get(m)
            row[f"crossday_{m[:8]}"] = cm["recall"] if cm else None
        rows.append(row)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"[out] {OUT_CSV}", file=sys.stderr)
    res["runtime_s"] = time.time() - t0
    res["peak_rss_mb"] = X.log_rss("end")
    X.write_json(OUT_JSON, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
