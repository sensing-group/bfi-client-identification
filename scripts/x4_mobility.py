#!/usr/bin/env python3
"""X4 (reviewer: moving devices). Per-device K=1 recall and cross-day recall,
stationary TVs versus mobile devices (laptop, vacuum), plus within-device
feature variability.

Three feature sets, all on the residential archive 2026-05-08..20:
  paper-protocol : the paper's exact pipeline (byte-histogram "body" plus
                   power/capability, cosine, random-start windows), per-MAC
                   tallies of the reproduced fig_q1_fullwindow and
                   bfi_rebuttal_crossday runs (x0_reproduce_residential).
  givens         : decoded (phi_11, psi_21) common representation, 5x32
                   window statistics, z-scored on all residential windows of
                   the eligible MACs (label-free), euclidean 1-NN.
  body-hist      : the paper's byte histogram on the same non-overlapping
                   windows, cosine 1-NN (bridges the two protocols).
  geometry-only  : one-hot of the window's report geometry (VHT/HE, Nc,
                   Nr, codebook, SU/MU, bandwidth), random tie-break. A
                   reference for how much the report FORMAT alone identifies.

Outputs: figures/x4_mobility.json, figures/x4_mobility_per_device.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

import xres_common as X
import xres_eval as E
import x0_reproduce_residential as X0

OUT_JSON = X.FIG_DIR / "x4_mobility.json"
OUT_CSV = X.FIG_DIR / "x4_mobility_per_device.csv"
VAR_MAX_WINDOWS = 1500       # per MAC, for pairwise distance statistics
SEED = 0


def eligible_macs(w: dict) -> list[str]:
    """Same eligibility as the paper: >= 64 frames in the archive. The
    window table has >= 2 windows exactly for those MACs (non-short)."""
    keep = ~w["short"]
    macs, cnt = np.unique(w["mac"][keep], return_counts=True)
    return sorted(m for m, c in zip(macs, cnt) if c >= 2)


def feature_sets(w: dict, sel: np.ndarray) -> dict:
    mu, sd = E.zscore_fit(w["givens"][sel])
    geo = np.eye(int(w["geom"].max()) + 1)[w["geom"]]
    return dict(
        givens=(E.zscore_apply(w["givens"], mu, sd), "euclidean"),
        body_hist=(w["hist"].astype(np.float64), "cosine"),
        geometry_only=(geo, "cosine"),
    )


def variability(F: np.ndarray, w: dict, macs: list[str], metric: str,
                rng: np.random.Generator) -> dict:
    out = {}
    samp = {}
    for m in macs:
        ii = np.flatnonzero((w["mac"] == m) & ~w["short"])
        if ii.size > VAR_MAX_WINDOWS:
            ii = np.sort(rng.choice(ii, VAR_MAX_WINDOWS, replace=False))
        samp[m] = ii
    allidx = np.concatenate(list(samp.values()))

    def dist(A, B):
        if metric == "cosine":
            a = A / np.linalg.norm(A, axis=1, keepdims=True)
            b = B / np.linalg.norm(B, axis=1, keepdims=True)
            return 1.0 - a @ b.T
        return np.sqrt(np.maximum(
            (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T,
            0))

    for m in macs:
        ii = samp[m]
        D = dist(F[ii], F[ii])
        dd = w["day"][ii]
        iu = np.triu_indices(ii.size, k=1)
        same = dd[iu[0]] == dd[iu[1]]
        vals = D[iu]
        other = allidx[w["mac"][allidx] != m]
        Db = dist(F[ii], F[other])
        out[m] = dict(
            n_windows=int(ii.size),
            within_same_day_median=float(np.median(vals[same]))
            if same.any() else None,
            within_cross_day_median=float(np.median(vals[~same]))
            if (~same).any() else None,
            between_median=float(np.median(Db)),
            nearest_other_mac_median=float(np.median(Db.min(axis=1))),
            nearest_same_mac_median=float(np.median(
                np.where(np.eye(ii.size, dtype=bool), np.inf, D).min(axis=1)))
            if ii.size > 1 else None,
        )
        wd = out[m]["within_cross_day_median"] or out[m][
            "within_same_day_median"]
        out[m]["within_over_between"] = (wd / out[m]["between_median"]
                                         if wd is not None else None)
    return out


GROUP_METRICS = ("paper_pooled_body", "paper_pooled_power+capability+body",
                 "paper_crossday", "givens_pooled", "givens_crossday",
                 "body_hist_pooled", "body_hist_crossday",
                 "geometry_only_pooled", "geometry_only_crossday",
                 "givens_within_crossday", "givens_within_over_between",
                 "body_hist_within_over_between")

MOBILITY_GROUPS = ("stationary", "mobile")


def group_macro(rows: list[dict]) -> dict:
    """Unweighted mean over devices in each mobility group."""
    out = {}
    for grp in MOBILITY_GROUPS:
        rs = [r for r in rows if r["group"] == grp]
        out[grp] = {"devices": [r["mac"][:8] for r in rs]}
        for k in GROUP_METRICS:
            v = [r[k] for r in rs if r[k] is not None]
            out[grp][k] = float(np.mean(v)) if v else None
    return out


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
    sel = np.isin(w["mac"], macs) & ~w["short"]
    print(f"eligible MACs: {macs}", file=sys.stderr)
    days_per_mac = {m: sorted(set(w["day"][sel & (w["mac"] == m)].tolist()))
                    for m in macs}
    nwin = {m: int((sel & (w["mac"] == m)).sum()) for m in macs}

    res = dict(eligible=macs, windows=nwin, days=days_per_mac,
               groups={m: X.mac_group(m) for m in macs}, runs={})
    # Paper protocol per-MAC (exact reproduction with tallies)
    fw = X0.full_window()
    cd = X0.crossday()
    res["paper_protocol"] = dict(
        full_window={k: v for k, v in fw["results"].items()},
        full_window_frames=fw["eligible"],
        crossday_per_mac=cd["per_mac_cross_day"],
        crossday_day_counts=cd["day_counts"],
        crossday_mean_pairs=cd["cross_day_mean"])
    X.log_rss("paper protocol")

    Fs = feature_sets(w, sel)
    rng = np.random.default_rng(SEED)
    for name, (F, metric) in Fs.items():
        Fm = F[sel]
        mm = w["mac"][sel]
        dm = w["day"][sel]
        res["runs"][name] = dict(
            metric=metric,
            pooled=E.pooled(Fm, mm, macs, metric, seed=SEED),
            crossday=E.crossday(Fm, mm, dm, macs, metric, seed=SEED))
        print(f"  {name}: pooled {res['runs'][name]['pooled']['accuracy']:.3f}"
              f" cross-day {res['runs'][name]['crossday']['accuracy']:.3f}",
              file=sys.stderr)
        if name != "geometry_only":
            res["runs"][name]["variability"] = variability(
                F, w, macs, metric, rng)
    # Geometry-matched pair: laptop vs vacuum share the VHT MU 1x3 format.
    pair = [m for m in macs if m[:8] in ("64:4e:d7", "4c:b9:ea")]
    if len(pair) == 2:
        psel = sel & np.isin(w["mac"], pair)
        res["geometry_matched_pair"] = {}
        for name in ("givens", "body_hist"):
            F, metric = Fs[name]
            res["geometry_matched_pair"][name] = dict(
                pooled=E.pooled(F[psel], w["mac"][psel], pair, metric,
                                seed=SEED),
                crossday=E.crossday(F[psel], w["mac"][psel], w["day"][psel],
                                    pair, metric, seed=SEED))
    X.log_rss("x4 runs")

    rows = []
    for m in macs:
        lab, grp = X.mac_group(m)
        r = dict(mac=m, device=lab, group=grp, n_windows=nwin[m],
                 n_days=len(days_per_mac[m]),
                 days=" ".join(d[5:] for d in days_per_mac[m]))
        for ch in ("body", "power+capability+body"):
            h, n = res["paper_protocol"]["full_window"][ch]["per_mac"].get(
                m, [0, 0])
            r[f"paper_pooled_{ch}"] = h / n if n else None
            r[f"paper_pooled_{ch}_n"] = n
        cdm = res["paper_protocol"]["crossday_per_mac"].get(m)
        r["paper_crossday"] = cdm[0] / cdm[1] if cdm and cdm[1] else None
        r["paper_crossday_n"] = cdm[1] if cdm else 0
        r["paper_crossday_chance"] = cdm[2] / cdm[1] if cdm and cdm[1] \
            else None
        for name in Fs:
            p = res["runs"][name]["pooled"]["per_mac"].get(m, {})
            c = res["runs"][name]["crossday"]["per_mac"].get(m, {})
            r[f"{name}_pooled"] = p.get("recall")
            r[f"{name}_pooled_n"] = p.get("queries")
            r[f"{name}_crossday"] = c.get("recall")
            r[f"{name}_crossday_n"] = c.get("queries")
            r[f"{name}_crossday_chance"] = c.get("chance")
        for name in ("givens", "body_hist"):
            v = res["runs"][name]["variability"][m]
            r[f"{name}_within_sameday"] = v["within_same_day_median"]
            r[f"{name}_within_crossday"] = v["within_cross_day_median"]
            r[f"{name}_between"] = v["between_median"]
            r[f"{name}_within_over_between"] = v["within_over_between"]
        rows.append(r)
    res["group_macro"] = group_macro(rows)
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
