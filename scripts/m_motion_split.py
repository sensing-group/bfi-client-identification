#!/usr/bin/env python3
"""M1: does occupant motion break the device fingerprint? (reviewers A, C)

Label-free activity split. For every window of the R-series table
(r_common: 32 frames, 320-d Givens vector, session gap 300 s) the window's
own angle-change rate is computed from the per-day frame cache: the mean
absolute frame-to-frame change of the four dequantised angles (phi_11,
phi_21 circular, psi_21, psi_31 linear), over the 32 subcarrier bins the
cache stores, in radians per frame. Windows are split per device at the
device's median rate: "quiet" below, "active" at or above. The within-window
spread already in the feature vector (phi 1-R, psi std) is kept as a second
activity measure and its rank correlation with the rate is reported.

Protocol = r_eval (one-session enrollment, 1-NN over z-scored angles,
T = 1 query window, cross-day conditions other_day_pair and other_day_any),
with the enrollment windows restricted to the enrol class and the query
window restricted to the query class. A gallery device whose enrolled block
has no window of the enrol class drops out of that decision's gallery (the
reported chance follows the gallery actually used); an episode whose query
device has no enrolled window of the enrol class, or whose query block has
no window of the query class, is skipped and counted.

Outputs (figures/): m_motion_split.csv, m_motion_split_per_device.csv,
m_activity_by_hour.csv, m_motion_split_summary.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_GRID = (1, 16)
T_QUERY = 1
N_TRIALS = E.N_TRIALS
SEED = 100
CONDITIONS = ("other_day_pair", "other_day_any")
CLASSES = ("quiet", "active")
LOCAL_TZ = ZoneInfo(os.environ.get("BFI_LOCAL_TZ", "Europe/Vienna"))
NIGHT_HOURS = range(0, 6)        # 00-06 local
EVENING_HOURS = range(18, 24)    # 18-23 local
# Column blocks of the 320-d vector (r_common layout) holding within-window
# spread: phi11 spread, phi21 spread, psi21 std, psi31 std.
SPREAD_SLICES = (slice(64, 96), slice(160, 192), slice(224, 256),
                 slice(288, 320))
# Per-frame cache channels: cos/sin phi11, cos/sin phi21, psi21, psi31.
CH_COS11, CH_SIN11, CH_COS21, CH_SIN21, CH_PSI21, CH_PSI31 = range(6)
# Dry-run memory estimate: only feat (half of a day cache) is read; the
# Table holds float64 copies of feat and feat_rq plus the float32 originals.
DAY_CACHE_LOAD_FRACTION = 0.5
TABLE_COPIES = 3
BASE_RSS_MB = 150

OUT = dict(main=R.FIG_DIR / "m_motion_split.csv",
           per_device=R.FIG_DIR / "m_motion_split_per_device.csv",
           hours=R.FIG_DIR / "m_activity_by_hour.csv",
           summary=R.FIG_DIR / "m_motion_split_summary.json")


# ---------------------------------------------------------------------------
# Per-window angle-change rate (streams the day caches, same window order
# as r_common.build_windows; verified against the table's ts0/ts1)
# ---------------------------------------------------------------------------
def window_rate(fb: np.ndarray) -> float:
    """(T, 6, B) per-frame bins -> mean |delta angle| per frame, radians."""
    a11 = np.arctan2(fb[:, CH_SIN11], fb[:, CH_COS11])
    a21 = np.arctan2(fb[:, CH_SIN21], fb[:, CH_COS21])
    d_phi = np.concatenate([np.diff(a11, axis=0), np.diff(a21, axis=0)])
    d_phi = np.abs((d_phi + np.pi) % (2 * np.pi) - np.pi)
    d_psi = np.abs(np.concatenate([np.diff(fb[:, CH_PSI21], axis=0),
                                   np.diff(fb[:, CH_PSI31], axis=0)]))
    return float(np.concatenate([d_phi.ravel(), d_psi.ravel()]).mean())


def window_rates(devs: list[str], w: dict, gap_s: float) -> np.ndarray:
    ts_all = R.all_timestamps(devs)
    sess_of = {m: (t, R.cut_sessions(t, gap_s)) for m, t in ts_all.items()}
    del ts_all
    rates, ts0, ts1 = [], [], []
    for p in X.day_paths():
        z = np.load(R.day_cache_path(p.stem), allow_pickle=False)
        if "ts" not in z.files or not z["ts"].size:
            continue
        ts, dev, feat = z["ts"], z["dev"], z["feat"]
        for i, m in enumerate(devs):
            sel = np.flatnonzero(dev == i)
            if sel.size == 0:
                continue
            t_all, sid_all = sess_of[m]
            sid = sid_all[np.searchsorted(t_all, ts[sel])]
            for s in np.unique(sid):
                ii = sel[sid == s]
                for k in range(ii.size // R.WINDOW_FRAMES):
                    jj = ii[k * R.WINDOW_FRAMES:(k + 1) * R.WINDOW_FRAMES]
                    rates.append(window_rate(feat[jj]))
                    ts0.append(float(ts[jj[0]]))
                    ts1.append(float(ts[jj[-1]]))
        del z, ts, dev, feat
        gc.collect()
        X.log_rss(p.stem)
    ts0, ts1 = np.asarray(ts0), np.asarray(ts1)
    if ts0.size != len(w["dev"]) or not (np.array_equal(ts0, w["ts0"])
                                         and np.array_equal(ts1, w["ts1"])):
        raise RuntimeError("window order differs from the cached table")
    return np.asarray(rates)


def spread_measure(w: dict) -> np.ndarray:
    return np.nanmean(np.concatenate([w["feat"][:, s] for s in SPREAD_SLICES],
                                     axis=1), axis=1)


def activity_split(w: dict, rate: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-device median split. Returns class array (0 quiet, 1 active)
    and the per-device thresholds."""
    cls = np.zeros(rate.size, np.int8)
    thr = {}
    for i, m in enumerate(w["macs"]):
        s = w["dev"] == i
        med = float(np.median(rate[s]))
        cls[s] = (rate[s] >= med).astype(np.int8)
        thr[R.short(str(m))] = dict(
            median_rate_rad_per_frame=med, n_windows=int(s.sum()),
            n_active=int(cls[s].sum()),
            rate_p10=float(np.percentile(rate[s], 10)),
            rate_p90=float(np.percentile(rate[s], 90)))
    return cls, thr


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Time of day of active windows
# ---------------------------------------------------------------------------
def local_hour(ts: np.ndarray) -> np.ndarray:
    return np.array([datetime.fromtimestamp(float(t), tz=LOCAL_TZ).hour
                     for t in ts], dtype=np.int8)


def hour_rows(w: dict, cls: np.ndarray, hour: np.ndarray) -> list[dict]:
    rows = []
    for i, m in enumerate(w["macs"]):
        s = w["dev"] == i
        for h in range(24):
            sh = s & (hour == h)
            n, na = int(sh.sum()), int(cls[sh].sum())
            rows.append(dict(device=R.short(str(m)), tz=str(LOCAL_TZ.key),
                             hour=h, n_windows=n, n_active=na,
                             frac_active=(na / n) if n else float("nan"),
                             share_of_device_active=(
                                 na / cls[s].sum()) if cls[s].sum() else 0.0,
                             share_of_device_windows=n / s.sum()))
    return rows


def hour_summary(w: dict, cls: np.ndarray, hour: np.ndarray) -> dict:
    out = {}
    for i, m in enumerate(w["macs"]):
        s = w["dev"] == i
        d = {}
        for name, hrs in (("night_00_06", NIGHT_HOURS),
                          ("evening_18_23", EVENING_HOURS)):
            sh = s & np.isin(hour, list(hrs))
            n, na = int(sh.sum()), int(cls[sh].sum())
            d[name] = dict(n_windows=n, n_active=na,
                           frac_active=(na / n) if n else None,
                           share_of_device_active=float(
                               na / cls[s].sum()) if cls[s].sum() else None,
                           share_of_device_windows=float(n / s.sum()))
        out[R.short(str(m))] = d
    return out


# ---------------------------------------------------------------------------
# Split evaluation
# ---------------------------------------------------------------------------
def run_split(tb: E.Table, cls: np.ndarray, cond: str, ce: int, cq: int,
              n_trials: int, seed: int) -> tuple[list, dict]:
    """Records (K, dev, block, G, hit, de, dq, gal, capped) for enrol class
    ce and query class cq at T = T_QUERY."""
    rng = np.random.default_rng(seed)
    recs, skipped = [], defaultdict(int)
    for G, m, qb, (de, dq) in E.episodes(tb, cond, rng, n_trials):
        pools = {g: tb.blocks[G[g]][cls[tb.blocks[G[g]]] == ce] for g in G}
        gdev = [g for g in G if pools[g].size]
        if m not in gdev:
            skipped["query_device_no_enrol_class"] += 1
            continue
        if len(gdev) < 2:
            skipped["gallery_below_two"] += 1
            continue
        qpool = tb.blocks[qb][cls[tb.blocks[qb]] == cq]
        if qpool.size < T_QUERY:
            skipped["query_block_no_query_class"] += 1
            continue
        perms = {g: rng.permutation(pools[g]) for g in gdev}
        qi = qpool[rng.integers(qpool.size - T_QUERY + 1):][:T_QUERY]
        qz = tb.Z[qi].mean(0)
        dz = {g: np.sqrt(((tb.Z[perms[g]] - qz) ** 2).sum(1)) for g in gdev}
        for K in K_GRID:
            d = np.array([dz[g][:min(K, perms[g].size)].min() for g in gdev])
            pred = gdev[E._argmin_random(d, rng)]
            recs.append(dict(K=K, dev=m, block=qb, G=len(gdev), hit=pred == m,
                             pred=pred, de=de, dq=dq,
                             gal=tuple(sorted(gdev)),
                             capped=perms[m].size < K))
    return recs, dict(skipped)


def aggregate(recs: list, names: list[str], extra: dict,
              rng: np.random.Generator):
    rows, dev_rows = [], []
    grp = defaultdict(list)
    for r in recs:
        grp[r["K"]].append(r)
    for K, rr in sorted(grp.items()):
        hits = np.array([r["hit"] for r in rr])
        cl = np.array([f"{r['dev']}|{r['block']}" for r in rr])
        lo, hi = E.boot_ci(hits, cl, rng)
        per = defaultdict(list)
        for r in rr:
            per[r["dev"]].append(r)
        rec = {g: float(np.mean([x["hit"] for x in v])) for g, v in per.items()}
        sizes = np.array([r["G"] for r in rr])
        rows.append(dict(
            **extra, K=K, T=T_QUERY, accuracy=float(hits.mean()),
            ci_lo=lo, ci_hi=hi, chance=float(np.mean(1.0 / sizes)),
            macro_recall=float(np.mean(list(rec.values()))),
            macro_chance=float(np.mean([np.mean([1.0 / x["G"] for x in v])
                                        for v in per.values()])),
            n_decisions=int(hits.size), n_query_blocks=int(len(set(cl))),
            n_devices_queried=len(per),
            gallery_size_hist=json.dumps(
                {int(k): int(v) for k, v in
                 zip(*np.unique(sizes, return_counts=True))}),
            enrol_capped_frac=float(np.mean([r["capped"] for r in rr]))))
        for g, v in sorted(per.items()):
            conf = defaultdict(int)
            for x in v:
                conf[names[x["pred"]]] += 1
            dev_rows.append(dict(
                **extra, K=K, T=T_QUERY, device=names[g], recall=rec[g],
                n_decisions=len(v),
                n_query_blocks=len({x["block"] for x in v}),
                chance=float(np.mean([1.0 / x["G"] for x in v])),
                predicted=json.dumps(dict(conf))))
    return rows, dev_rows


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    if a.dry_run:
        sizes = {p.stem: R.day_cache_path(p.stem).stat().st_size / 1e6
                 for p in X.day_paths()}
        n_win = int(len(w["dev"]))
        print(json.dumps(dict(
            windows=n_win, day_cache_mb=sizes,
            est_peak_mb=max(sizes.values()) * DAY_CACHE_LOAD_FRACTION
            + n_win * R.FEATURE_DIM * 8 * TABLE_COPIES / 1e6
            + BASE_RSS_MB), indent=2))
        return 0
    tb = E.Table(w)
    rate = window_rates(devs, w, R.SESSION_GAP_S)
    X.log_rss("rates")
    spread = spread_measure(w)
    cls, thr = activity_split(w, rate)
    for i, m in enumerate(w["macs"]):
        s = w["dev"] == i
        thr[R.short(str(m))]["spearman_rate_vs_spread"] = spearman(
            rate[s], spread[s])
    hour = local_hour(w["ts0"])
    E.write_csv(OUT["hours"], hour_rows(w, cls, hour))

    rng_b = np.random.default_rng(SEED + 1)
    rows, dev_rows, skipped = [], [], {}
    for ci, cond in enumerate(CONDITIONS):
        for ie, ce in enumerate(CLASSES):
            for iq, cq in enumerate(CLASSES):
                t = time.time()
                recs, sk = run_split(tb, cls, cond, ie, iq, a.trials,
                                     SEED + 10 * ci + 2 * ie + iq)
                extra = dict(gap_s=int(R.SESSION_GAP_S), condition=cond,
                             feature="angles", enrol=ce, query=cq)
                r1, r2 = aggregate(recs, tb.names, extra, rng_b)
                rows += r1
                dev_rows += r2
                skipped[f"{cond}|{ce}|{cq}"] = dict(
                    sk, n_episodes_used=len(recs) // len(K_GRID))
                print(f"[m1] {cond} enrol={ce} query={cq}: {len(recs)} "
                      f"records {time.time() - t:.0f}s", file=sys.stderr)
                del recs
        X.log_rss(cond)
    E.write_csv(OUT["main"], rows)
    E.write_csv(OUT["per_device"], dev_rows)
    X.write_json(OUT["summary"], dict(
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        trials=a.trials, k_grid=list(K_GRID), T=T_QUERY,
        rate_definition="mean |frame-to-frame change| of phi11, phi21 "
                        "(circular), psi21, psi31 over 32 subcarrier bins, "
                        "rad/frame; split at the device median",
        local_tz=str(LOCAL_TZ.key),
        thresholds=thr, episodes_skipped=skipped,
        activity_by_time_of_day=hour_summary(w, cls, hour)))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
