#!/usr/bin/env python3
"""R-eval checks: feature scaling and interval choice of the home protocol.

r_eval.py standardises the window vectors with the mean and standard
deviation of ALL windows of the six devices (r_eval.Table), query windows
included, and its interval resamples device|query-session blocks. This
script repeats r_eval's decisions, episode for episode, under three scalings

  stored    r_eval's own: z-score fitted on every window of the table
  enrolled  z-score fitted on the windows enrolled in that decision only
            (the K windows of each gallery device; no query window, no
            window of another session or day)
  raw       the window vector as built, no scaling

and gives accuracy, macro recall and per-device recall with bootstrap
intervals for both, under three resampling schemes

  block       device|query-session blocks, as r_eval.boot_ci
  day         capture days of the query (all devices of a day move together;
              a device absent from a draw is left out of that draw's macro
              recall, and the share of such draws is reported)
  device_day  the days of each device resampled within the device

Nothing of the protocol is re-implemented: r_eval.run_condition draws the
episodes with r_eval's grids and seeds, and a tap on the feature matrix
keeps the rows it selects (the query window and each gallery device's
shuffled enrollment block). The 'stored' decisions rebuilt from those rows
must equal run_condition's own records and figures/r_main.csv (asserted).

Sections
  home     unprotected capture, angles, CONDITIONS x K_REPORT, T = 1
  defense  DEFENSE_BODIES x DEFENSE_OBSERVERS, other_day_any, DEFENSE_K,
           T = 1: one-sided upper bound of macro recall, and of the paired
           difference to the pure-noise reference (same episodes, same
           resampled clusters). The rotated window tables come from
           r_rotation.get_windows, which runs the rotation when its cache
           is absent (--dry-run says whether it is); 'stored' rows must
           equal figures/r_rotation_main.csv (asserted).

Usage: python3 scripts/r_eval_checks.py [--dry-run] [--skip-defense]
Outputs (figures/): r_eval_checks.json, r_eval_checks.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import warnings
from collections import namedtuple
from pathlib import Path

BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_rotation as RO  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITIONS = ("other_day_any", "same_day")
K_REPORT = (1, 16, 0)                     # 0 = all windows of the session
T_REPORT = 1
FEATURE = "angles"                        # the 1-NN rule on the window vector
SCALINGS = ("stored", "enrolled", "raw")
SCHEMES = ("block", "day", "device_day")
SD_FLOOR = 1e-9                           # r_eval.Table's floor on the std
N_BOOT = E.N_BOOT
BOOT_SEED = 11
CI_LEVEL = E.CI_LEVEL
UPPER_LEVEL = 0.95                        # one-sided upper confidence bound
CHANCE_MARGIN = 0.01                      # "chance + 1 point"
REF_TOL = 1e-12                           # against the stored CSV values
DEFENSE_CONDITION = "other_day_any"
DEFENSE_BODIES = ("rot_group_cb42_n234", "rot_group", RO.NULL_BODY)
DEFENSE_OBSERVERS = ("angles", "invariant")
DEFENSE_K = (1, 0)
EST_TABLE_COPIES = 6                      # float64 copies of a window table held at once
REF_HOME = E.OUT["main"]
REF_HOME_DEV = E.OUT["per_device"]
REF_DEFENSE = RO.OUT["main"]
OUT_JSON = R.FIG_DIR / "r_eval_checks.json"
OUT_CSV = R.FIG_DIR / "r_eval_checks.csv"

Episode = namedtuple("Episode", "dev block day gal qi perms stored_hit")


# ---------------------------------------------------------------------------
# Episodes of r_eval.run_condition, with the rows it selected
# ---------------------------------------------------------------------------
class Tap:
    """Feature matrix that keeps every row selection made on it."""

    def __init__(self, a: np.ndarray):
        self.a = a
        self.log: list = []

    def __getitem__(self, idx):
        self.log.append(idx)
        return self.a[idx]


def traced_episodes(tb: E.Table, cond: str, trials: int) -> list[Episode]:
    """r_eval.run_condition on r_eval's full grid (so that its random stream
    is the stored one), returned as the T_REPORT episodes with the query
    rows, the shuffled enrollment rows per gallery device and the stored
    'angles' hit per K."""
    z = tb.Z
    tap = Tap(z)
    tb.Z = tap
    try:
        recs = E.run_condition(tb, cond, E.K_GRID, E.T_GRID, trials,
                               E.SEED + E.CONDITIONS.index(cond))
    finally:
        tb.Z = z
    n_feat, n_k = len(E.FEATURES), len(E.K_GRID)
    i_feat = E.FEATURES.index(FEATURE)
    out, pos = [], 0
    for c in range(0, len(recs), n_k * n_feat):
        r = recs[c]
        qi, perms = tap.log[pos], tap.log[pos + 1:pos + 1 + r.G]
        pos += 1 + r.G * (1 + n_k)        # query, blocks, centroid prefixes
        if len(qi) != r.T or tb.dev[qi[0]] != r.dev or any(
                tb.dev[p[0]] != g for p, g in zip(perms, r.gal)):
            raise RuntimeError("tap out of step with run_condition")
        if r.T != T_REPORT:
            continue
        hit = {}
        for ki, K in enumerate(E.K_GRID):
            x = recs[c + ki * n_feat + i_feat]
            if (x.feat, x.K, x.T, x.block) != (FEATURE, K, r.T, r.block):
                raise RuntimeError("record order differs from r_eval's")
            hit[K] = bool(x.hit)
        out.append(Episode(r.dev, r.block, r.dq, r.gal, qi, perms, hit))
    if pos != len(tap.log):
        raise RuntimeError("tap log not consumed")
    return out


# ---------------------------------------------------------------------------
# Decisions under a scaling
# ---------------------------------------------------------------------------
def fit_scale(a: np.ndarray):
    """r_eval.Table's z-score rule on the rows of a."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0)
    sd[~np.isfinite(sd) | (sd < SD_FLOOR)] = 1.0
    return mu, sd


def decide(ep: Episode, K: int, scaling: str, z: np.ndarray, raw: np.ndarray,
           rng: np.random.Generator) -> tuple[bool, bool]:
    """(hit, tie) of the 1-NN decision of one episode."""
    kk = [p.size if K == 0 else min(K, p.size) for p in ep.perms]
    rows = np.concatenate([p[:k] for p, k in zip(ep.perms, kk)])
    if scaling == "stored":
        a, q = z[rows], z[ep.qi].mean(0)
    elif scaling == "raw":
        a, q = np.nan_to_num(raw[rows]), np.nan_to_num(raw[ep.qi]).mean(0)
    elif scaling == "enrolled":
        mu, sd = fit_scale(raw[rows])
        a = np.nan_to_num((raw[rows] - mu) / sd)
        q = np.nan_to_num((raw[ep.qi] - mu) / sd).mean(0)
    else:
        raise ValueError(scaling)
    d = np.sqrt(((a - q) ** 2).sum(1))
    dmin = np.minimum.reduceat(d, np.cumsum([0] + kk[:-1]))
    tie = int((dmin <= dmin.min() + E.TIE_TOL).sum()) > 1
    return ep.gal[E._argmin_random(dmin, rng)] == ep.dev, tie


def hits_of(eps: list[Episode], K: int, scaling: str, z: np.ndarray,
            raw: np.ndarray, rng: np.random.Generator):
    res = [decide(ep, K, scaling, z, raw, rng) for ep in eps]
    hits = np.array([h for h, _ in res])
    if scaling == "stored" and not np.array_equal(
            hits, np.array([ep.stored_hit[K] for ep in eps])):
        raise RuntimeError(f"stored decisions not reproduced at K={K}")
    return hits, int(sum(t for _, t in res))


# ---------------------------------------------------------------------------
# Cluster bootstrap of accuracy and macro recall
# ---------------------------------------------------------------------------
def scheme_clusters(eps: list[Episode], scheme: str):
    """(cluster label per decision, stratum per decision or None)."""
    dev = np.array([ep.dev for ep in eps])
    if scheme == "block":
        return np.array([f"{ep.dev}|{ep.block}" for ep in eps]), None
    if scheme == "day":
        return np.array([ep.day for ep in eps]), None
    if scheme == "device_day":
        return np.array([f"{ep.dev}|{ep.day}" for ep in eps]), dev
    raise ValueError(scheme)


def boot_draws(hits: list[np.ndarray], dev: np.ndarray, clusters: np.ndarray,
               strata: np.ndarray | None, rng: np.random.Generator):
    """Clusters resampled with replacement (within each stratum if given),
    the same draw for every hit vector: accuracy and macro recall, each
    (N_BOOT, len(hits)), and the share of draws with a device absent."""
    u, ci = np.unique(clusters, return_inverse=True)
    devs, di = np.unique(dev, return_inverse=True)
    n = np.zeros((len(u), len(devs)))
    np.add.at(n, (ci, di), 1.0)
    if strata is None:
        pick = rng.integers(len(u), size=(N_BOOT, len(u)))
    else:
        cs = np.zeros(len(u), dtype=strata.dtype)
        cs[ci] = strata
        pick = np.empty((N_BOOT, len(u)), dtype=np.int64)
        col = 0
        for s in np.unique(cs):
            members = np.flatnonzero(cs == s)
            pick[:, col:col + members.size] = members[rng.integers(
                members.size, size=(N_BOOT, members.size))]
            col += members.size
    ns = n[pick].sum(1)                                   # (N_BOOT, n_dev)
    present = ns > 0
    acc, macro = [], []
    for h in hits:
        hm = np.zeros_like(n)
        np.add.at(hm, (ci, di), h.astype(float))
        hs = hm[pick].sum(1)
        acc.append(hs.sum(1) / ns.sum(1))
        rec = np.divide(hs, ns, out=np.zeros_like(hs), where=present)
        macro.append(rec.sum(1) / present.sum(1))
    return (np.stack(acc, 1), np.stack(macro, 1),
            float((~present.all(1)).mean()))


def interval(x: np.ndarray) -> dict:
    a = (1 - CI_LEVEL) / 2
    return dict(lo=float(np.quantile(x, a)), hi=float(np.quantile(x, 1 - a)),
                ub95=float(np.quantile(x, UPPER_LEVEL)))


def point(hits: np.ndarray, eps: list[Episode], names: list[str]) -> dict:
    dev = np.array([ep.dev for ep in eps])
    rec = {names[g]: float(hits[dev == g].mean()) for g in np.unique(dev)}
    sizes = np.array([len(ep.gal) for ep in eps])
    return dict(
        accuracy=float(hits.mean()),
        macro_recall=float(np.mean(list(rec.values()))),
        chance=float(np.mean(1.0 / sizes)),
        macro_chance=float(np.mean([np.mean(1.0 / sizes[dev == g])
                                    for g in np.unique(dev)])),
        n_decisions=int(hits.size),
        n_query_blocks=len({(ep.dev, ep.block) for ep in eps}),
        n_days=len({ep.day for ep in eps}),
        per_device_recall=rec,
        per_device_decisions={names[g]: int((dev == g).sum())
                              for g in np.unique(dev)},
        per_device_query_blocks={names[g]: len({ep.block for ep in eps
                                                if ep.dev == g})
                                 for g in np.unique(dev)},
        per_device_days={names[g]: len({ep.day for ep in eps if ep.dev == g})
                         for g in np.unique(dev)})


def score(eps: list[Episode], hits: np.ndarray, names: list[str],
          rng: np.random.Generator, ref: np.ndarray | None = None) -> dict:
    """Point values and, per scheme, intervals of accuracy and macro recall;
    with ref (hits of the reference body on the same episodes) also the
    paired difference hits - ref."""
    out = point(hits, eps, names)
    dev = np.array([ep.dev for ep in eps])
    out["intervals"] = {}
    for scheme in SCHEMES:
        cl, strata = scheme_clusters(eps, scheme)
        acc, mac, absent = boot_draws(
            [hits] if ref is None else [hits, ref], dev, cl, strata, rng)
        iv = dict(n_clusters=int(len(set(cl.tolist()))),
                  draws_with_device_absent=absent,
                  accuracy=interval(acc[:, 0]),
                  macro_recall=interval(mac[:, 0]))
        if ref is not None:
            iv["accuracy_minus_reference"] = interval(acc[:, 0] - acc[:, 1])
            iv["macro_recall_minus_reference"] = interval(mac[:, 0] - mac[:, 1])
        out["intervals"][scheme] = iv
    return out


# ---------------------------------------------------------------------------
# Stored values
# ---------------------------------------------------------------------------
def stored_home() -> tuple[dict, dict]:
    gap = str(int(R.SESSION_GAP_S))
    main = {(r["condition"], r["K"]): r for r in RO.read_csv_rows(REF_HOME)
            if r["gap_s"] == gap and r["feature"] == FEATURE
            and int(r["T"]) == T_REPORT}
    dev = {(r["condition"], r["K"], r["device"]): r
           for r in RO.read_csv_rows(REF_HOME_DEV)
           if r["gap_s"] == gap and r["feature"] == FEATURE
           and int(r["T"]) == T_REPORT}
    return main, dev


def stored_defense() -> dict:
    return {(r["body"], r["feature"], r["K"]): r
            for r in RO.read_csv_rows(REF_DEFENSE)
            if r["condition"] == DEFENSE_CONDITION and int(r["T"]) == T_REPORT}


def assert_stored(res: dict, ref: dict, what: str) -> dict:
    """The 'stored' scaling against a row of the stored CSV."""
    diff = {k: abs(res[k] - float(ref[k]))
            for k in ("accuracy", "macro_recall")}
    counts_ok = (res["n_decisions"] == int(ref["n_decisions"])
                 and res["n_query_blocks"] == int(ref["n_query_blocks"]))
    if max(diff.values()) > REF_TOL or not counts_ok:
        raise RuntimeError(f"{what}: stored value not reproduced {diff}")
    return dict(max_abs_diff=max(diff.values()), counts_equal=counts_ok,
                stored_ci=[float(ref["ci_lo"]), float(ref["ci_hi"])])


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def run_table(w: dict, cond: str, k_report: tuple, trials: int,
              rng_b: np.random.Generator, rng_t: np.random.Generator,
              ref_hits: dict | None = None):
    """{(K, scaling): result} of one window table and condition, and the
    hit vectors (the reference of a paired difference)."""
    tb = E.Table(w)
    eps = traced_episodes(tb, cond, trials)
    raw = w["feat"].astype(np.float64)
    res, hits_all = {}, {}
    for K in k_report:
        for scaling in SCALINGS:
            hits, ties = hits_of(eps, K, scaling, tb.Z, raw, rng_t)
            ref = None if ref_hits is None else ref_hits[(K, scaling)]
            r = score(eps, hits, tb.names, rng_b, ref)
            r["ties"] = ties
            res[(K, scaling)] = r
            hits_all[(K, scaling)] = hits
    key = [(ep.dev, ep.block, ep.day) for ep in eps]
    del tb, raw, eps
    gc.collect()
    return res, hits_all, key


def null_windows(w: dict) -> dict:
    """r_rotation.run_null's table: the same windows, every feature vector
    i.i.d. standard normal, drawn in run_null's order and seed."""
    rng = np.random.default_rng(RO.NULL_SEED)
    n = w["feat"].shape[0]
    return dict(
        w, feat=rng.standard_normal(w["feat"].shape).astype(np.float32),
        feat_rq=rng.standard_normal(w["feat_rq"].shape).astype(np.float32),
        inv=rng.standard_normal((n, RO.INV_DIM)).astype(np.float32))


def defense_windows(body: str, devs: list[str], w_ref: dict) -> dict:
    if body == RO.NULL_BODY:
        return null_windows(w_ref)
    w = RO.get_windows(devs, R.SESSION_GAP_S, body)
    if not RO.check_variant_alignment(w, w_ref):
        raise RuntimeError(f"{body}: windows not aligned with r_common's")
    return w


def flat_rows(section: str, body: str, observer: str, cond: str, K: int,
              scaling: str, r: dict) -> dict:
    row = dict(section=section, body=body, observer=observer, condition=cond,
               K=E.klabel(K), T=T_REPORT, scaling=scaling,
               accuracy=r["accuracy"], macro_recall=r["macro_recall"],
               chance=r["chance"], macro_chance=r["macro_chance"],
               n_decisions=r["n_decisions"],
               n_query_blocks=r["n_query_blocks"], n_days=r["n_days"],
               ties=r["ties"])
    for scheme, iv in r["intervals"].items():
        for m in ("accuracy", "macro_recall"):
            for b in ("lo", "hi", "ub95"):
                row[f"{m}_{scheme}_{b}"] = iv[m][b]
        row[f"device_absent_{scheme}"] = iv["draws_with_device_absent"]
        d = iv.get("macro_recall_minus_reference")
        for b in ("lo", "hi", "ub95"):
            row[f"macro_minus_ref_{scheme}_{b}"] = d[b] if d else ""
    row["per_device_recall"] = json.dumps(r["per_device_recall"])
    return row


def dry_run(devs: list[str]) -> dict:
    w = R.get_windows(devs, R.SESSION_GAP_S)
    n, dim = w["feat"].shape
    summ = json.loads(RO.OUT["summary"].read_text()) \
        if RO.OUT["summary"].exists() else {}
    cached = {b: RO.windows_path(b, R.SESSION_GAP_S).exists()
              for b in DEFENSE_BODIES if b != RO.NULL_BODY}
    return dict(
        windows=int(n), feature_dim=int(dim),
        est_table_mb=n * dim * 8 * EST_TABLE_COPIES / 1e6,
        rotated_window_cache_present=cached,
        rotation_rerun_needed=[b for b, ok in cached.items() if not ok],
        stored_rotation_run=dict(
            runtime_s_three_variants=summ.get("runtime_s"),
            runtime_s_controls=summ.get("controls_runtime_s"),
            peak_rss_mb=summ.get("controls_peak_rss_mb")),
        vm_limit_mb=X.VM_LIMIT_BYTES / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="window counts, memory estimate and whether the "
                         "rotated window tables are cached")
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--skip-defense", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        print(json.dumps(dry_run(devs), indent=2))
        return 0
    check = a.trials == E.N_TRIALS         # stored values are 50-trial runs
    rng_b = np.random.default_rng(BOOT_SEED)
    rng_t = np.random.default_rng(BOOT_SEED + 1)
    w_ref = R.get_windows(devs, R.SESSION_GAP_S)
    rows, out = [], dict(home={}, defense={})

    # Home capture, unprotected ------------------------------------------
    ref_main, ref_dev = stored_home()
    for cond in CONDITIONS:
        res, _, _ = run_table(w_ref, cond, K_REPORT, a.trials, rng_b, rng_t)
        for (K, scaling), r in res.items():
            if scaling == "stored" and check:
                kl = E.klabel(K)
                r["reproduces_r_main"] = assert_stored(
                    r, ref_main[(cond, kl)], f"home {cond} K={kl}")
                for d, v in r["per_device_recall"].items():
                    if abs(v - float(ref_dev[(cond, kl, d)]["recall"])) \
                            > REF_TOL:
                        raise RuntimeError(f"per-device recall {cond} {d}")
            out["home"][f"{cond}|K={E.klabel(K)}|{scaling}"] = r
            rows.append(flat_rows("home", "none", FEATURE, cond, K, scaling, r))
        X.log_rss(f"home {cond} {time.time() - t0:.0f}s")

    # Defense bodies -----------------------------------------------------
    if not a.skip_defense:
        ref_def = stored_defense()
        tables = {}
        for body in (RO.NULL_BODY, *[b for b in DEFENSE_BODIES
                                     if b != RO.NULL_BODY]):
            w = defense_windows(body, devs, w_ref)
            for obs in DEFENSE_OBSERVERS:
                wo = w if obs == "angles" else dict(w, feat=w["inv"])
                null = tables.get((RO.NULL_BODY, obs))
                res, hits, key = run_table(
                    wo, DEFENSE_CONDITION, DEFENSE_K, a.trials, rng_b, rng_t,
                    None if null is None else null[0])
                if null is not None and key != null[1]:
                    raise RuntimeError(f"{body} {obs}: episodes differ from "
                                       "the noise reference's")
                tables[(body, obs)] = (hits, key)
                for (K, scaling), r in res.items():
                    if scaling == "stored" and check:
                        kl = E.klabel(K)
                        r["reproduces_r_rotation_main"] = assert_stored(
                            r, ref_def[(body, obs, kl)], f"{body} {obs} K={kl}")
                    out["defense"][f"{body}|{obs}|K={E.klabel(K)}|{scaling}"] = r
                    rows.append(flat_rows("defense", body, obs,
                                          DEFENSE_CONDITION, K, scaling, r))
                X.log_rss(f"defense {body} {obs} {time.time() - t0:.0f}s")
            del w
            gc.collect()

    E.write_csv(OUT_CSV, rows)
    X.write_json(OUT_JSON, dict(
        note=__doc__.split("\n\n")[1].replace("\n", " "),
        trials=a.trials, n_boot=N_BOOT, boot_seed=BOOT_SEED,
        ci_level=CI_LEVEL, upper_level=UPPER_LEVEL,
        chance_margin=CHANCE_MARGIN, scalings=list(SCALINGS),
        schemes=list(SCHEMES), feature=FEATURE, T=T_REPORT,
        session_gap_s=R.SESSION_GAP_S, stored_values_asserted=check,
        defense_reference=RO.NULL_BODY, results=out,
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
