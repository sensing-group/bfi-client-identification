#!/usr/bin/env python3
"""R1-R3: one-labelled-session identification on the residential archive.

Enrollment = K windows drawn at random (without replacement) from ONE
labelled (session, day) block of each gallery device; K is capped at the
block's window count, and the fraction of capped enrollments is reported.
Query = T consecutive windows (random start) from ONE unlabelled block of
another session, feature vectors averaged. Decision = nearest gallery
device.

Conditions
  same_day        enrollment and query blocks on the same archive day,
                  different sessions; gallery = devices with a block that day.
  other_day_pair  enrollment day de != query day dq; gallery = devices with
                  a block on de; queried devices must also appear on dq.
  other_day_any   gallery = every device with a block on some day != dq
                  (one random such block each); the usual "enrol once,
                  recognise later" attack.
Chance for each decision = 1 / |gallery| (random guess over the gallery);
reported as the mean over decisions.

Features
  angles          z-scored 320-d Givens window vector (r_common), 1-NN over
                  the K enrolled windows (euclidean).
  angles_centroid same vector, distance to the mean of the K windows.
  angles_rq       format-blind control: the same 1-NN on the vector built
                  after re-quantising all angles to 4-bit phi / 2-bit psi.
  format          one-hot of the window's report format (format, Nc, Nr,
                  angle bits, bandwidth, Ng, SU/MU, codebook); 1-NN, ties
                  broken uniformly at random.
  angles+format   format gate then angles: candidates are gallery devices
                  whose enrolled block's majority format equals the query's
                  majority format (all devices if none); angle 1-NN among them.
The z-score is fitted label-free on all windows of the six devices.

R3 (format-matched pair): Laptop vs Vacuum, restricted to windows in their
shared report format (VHT, Nc=1, Nr=3, 9/7-bit MU), 2-way gallery.

Outputs (figures/): r_main.csv, r_per_device.csv, r_daypair_chance.csv,
r_pair.csv, r_pair_per_device.csv, r_sessions.csv, r_device_set.json,
r_eval_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, namedtuple

import numpy as np

import r_common as R
import xres_common as X

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_GRID = (1, 2, 4, 8, 16, 0)            # 0 = all windows of the session
T_GRID = (1, 4, 16, 32)
K_GRID_SENS = (1, 0)
T_GRID_SENS = (1, 32)
N_TRIALS = 50
N_BOOT = 1000
CI_LEVEL = 0.95
SEED = 0
TIE_TOL = 1e-9
FEATURES = ("angles", "angles_centroid", "angles_rq", "format",
            "angles+format")
CONDITIONS = ("same_day", "other_day_pair", "other_day_any")
PAIR_DEVICES = ("Laptop", "Vacuum")
PAIR_FORMAT = ["VHT", 1, 3, 9, 7, 80, 1, "MU", 1]
PAIR_CONDITIONS = ("same_day", "other_day_any", "any_other_session")

OUT = dict(main=R.FIG_DIR / "r_main.csv",
           per_device=R.FIG_DIR / "r_per_device.csv",
           pair=R.FIG_DIR / "r_pair.csv",
           daypair=R.FIG_DIR / "r_daypair_chance.csv",
           pair_dev=R.FIG_DIR / "r_pair_per_device.csv",
           sessions=R.FIG_DIR / "r_sessions.csv",
           device_set=R.FIG_DIR / "r_device_set.json",
           summary=R.FIG_DIR / "r_eval_summary.json")


Rec = namedtuple("Rec", "cond feat K T dev block G hit pred capped de dq gal")


def klabel(k: int) -> str:
    return "all" if k == 0 else str(k)


# ---------------------------------------------------------------------------
# Window table -> blocks
# ---------------------------------------------------------------------------
class Table:
    def __init__(self, w: dict, keep: np.ndarray | None = None):
        keep = np.ones(len(w["dev"]), bool) if keep is None else keep
        self.idx = np.flatnonzero(keep)
        mu = np.nanmean(w["feat"], axis=0)        # label-free, all windows
        sd = np.nanstd(w["feat"], axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
        self.Z = np.nan_to_num((w["feat"] - mu) / sd).astype(np.float64)
        mu = np.nanmean(w["feat_rq"], axis=0)
        sd = np.nanstd(w["feat_rq"], axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
        self.Zrq = np.nan_to_num((w["feat_rq"] - mu) / sd).astype(np.float64)
        n_geom = len(w["geom_table"])
        self.F = np.eye(n_geom)[w["geom"]]
        self.geom = w["geom"]
        self.dev = w["dev"]
        self.names = [R.short(m) for m in w["macs"]]
        self.blocks: dict[tuple, np.ndarray] = {}
        by = defaultdict(list)
        order = self.idx[np.argsort(w["ts0"][self.idx], kind="stable")]
        for i in order:
            by[(int(w["dev"][i]), int(w["session"][i]), str(w["day"][i]))
               ].append(i)
        self.blocks = {k: np.asarray(v) for k, v in by.items()}
        self.days = sorted({k[2] for k in self.blocks})
        self.on_day = defaultdict(list)          # (dev, day) -> [block keys]
        for k in self.blocks:
            self.on_day[(k[0], k[2])].append(k)
        self.devs = sorted({k[0] for k in self.blocks})

    def present(self, day: str) -> list[int]:
        return [g for g in self.devs if self.on_day.get((g, day))]

    def majority_geom(self, ii: np.ndarray) -> int:
        v, c = np.unique(self.geom[ii], return_counts=True)
        return int(v[np.argmax(c)])


# ---------------------------------------------------------------------------
# Episodes: (gallery {dev: enrol block}, query dev, query block)
# ---------------------------------------------------------------------------
def episodes(tb: Table, cond: str, rng: np.random.Generator,
             n_trials: int):
    for _ in range(n_trials):
        if cond == "same_day":
            for d in tb.days:
                G = {g: tb.on_day[(g, d)][rng.integers(
                    len(tb.on_day[(g, d)]))] for g in tb.present(d)}
                if len(G) < 2:
                    continue
                for m in G:
                    q = [b for b in tb.on_day[(m, d)] if b[1] != G[m][1]]
                    if q:
                        yield G, m, q[rng.integers(len(q))], (d, d)
        elif cond == "other_day_pair":
            for de in tb.days:
                for dq in tb.days:
                    if de == dq:
                        continue
                    G = {g: tb.on_day[(g, de)][rng.integers(
                        len(tb.on_day[(g, de)]))] for g in tb.present(de)}
                    if len(G) < 2:
                        continue
                    for m in G:
                        q = [b for b in tb.on_day.get((m, dq), [])
                             if b[1] != G[m][1]]
                        if q:
                            yield G, m, q[rng.integers(len(q))], (de, dq)
        elif cond in ("other_day_any", "any_other_session"):
            for dq in tb.days:
                for m in tb.present(dq):
                    qb = tb.on_day[(m, dq)][rng.integers(
                        len(tb.on_day[(m, dq)]))]
                    G = {}
                    for g in tb.devs:
                        if cond == "other_day_any":
                            c = [b for b in tb.blocks
                                 if b[0] == g and b[2] != dq and b[1] != qb[1]]
                        else:
                            c = [b for b in tb.blocks
                                 if b[0] == g and b[1] != qb[1]]
                        if c:
                            G[g] = c[rng.integers(len(c))]
                    if len(G) >= 2 and m in G:
                        yield G, m, qb, ("any", dq)
        else:
            raise ValueError(cond)


def _argmin_random(d: np.ndarray, rng: np.random.Generator) -> int:
    c = np.flatnonzero(d <= d.min() + TIE_TOL)
    return int(c[0]) if c.size == 1 else int(rng.choice(c))


def run_condition(tb: Table, cond: str, k_grid, t_grid, n_trials: int,
                  seed: int) -> list[dict]:
    """One record per (decision episode, K, T, feature)."""
    rng = np.random.default_rng(seed)
    recs = []
    for G, m, qb, (de, dq) in episodes(tb, cond, rng, n_trials):
        gdev = list(G)
        gal = tuple(sorted(gdev))
        perms = {g: rng.permutation(tb.blocks[G[g]]) for g in gdev}
        gmaj = {g: tb.majority_geom(tb.blocks[G[g]]) for g in gdev}
        qwin = tb.blocks[qb]
        for T in t_grid:
            if qwin.size < T:
                continue
            s0 = int(rng.integers(qwin.size - T + 1))
            qi = qwin[s0:s0 + T]
            qz = tb.Z[qi].mean(0)
            qr = tb.Zrq[qi].mean(0)
            qf = tb.F[qi].mean(0)
            qg = int(np.argmax(qf))
            dz = {g: np.sqrt(((tb.Z[perms[g]] - qz) ** 2).sum(1))
                  for g in gdev}
            dr = {g: np.sqrt(((tb.Zrq[perms[g]] - qr) ** 2).sum(1))
                  for g in gdev}
            df = {g: np.sqrt(((tb.F[perms[g]] - qf) ** 2).sum(1))
                  for g in gdev}
            for K in k_grid:
                kk = {g: perms[g].size if K == 0 else min(K, perms[g].size)
                      for g in gdev}
                capped = K != 0 and perms[m].size < K
                d_ang = np.array([dz[g][:kk[g]].min() for g in gdev])
                d_cen = np.array([np.sqrt(((tb.Z[perms[g][:kk[g]]].mean(0)
                                            - qz) ** 2).sum()) for g in gdev])
                d_fmt = np.array([df[g][:kk[g]].min() for g in gdev])
                d_rq = np.array([dr[g][:kk[g]].min() for g in gdev])
                cand = [j for j, g in enumerate(gdev) if gmaj[g] == qg] \
                    or list(range(len(gdev)))
                d_gate = np.full(len(gdev), np.inf)
                d_gate[cand] = d_ang[cand]
                for feat, dd in (("angles", d_ang),
                                 ("angles_centroid", d_cen),
                                 ("angles_rq", d_rq),
                                 ("format", d_fmt),
                                 ("angles+format", d_gate)):
                    pred = gdev[_argmin_random(dd, rng)]
                    recs.append(Rec(cond, feat, K, T, m, qb, len(gdev),
                                    pred == m, pred, capped, de, dq, gal))
    return recs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def boot_ci(hits: np.ndarray, clusters: np.ndarray,
            rng: np.random.Generator) -> tuple[float, float]:
    u, inv = np.unique(clusters, return_inverse=True)
    h = np.bincount(inv, weights=hits.astype(float))
    n = np.bincount(inv).astype(float)
    accs = []
    for _ in range(N_BOOT):
        s = rng.integers(len(u), size=len(u))
        accs.append(h[s].sum() / n[s].sum())
    a = (1 - CI_LEVEL) / 2
    return float(np.quantile(accs, a)), float(np.quantile(accs, 1 - a))


def aggregate(recs: list, names: list[str], extra: dict,
              rng: np.random.Generator):
    rows, dev_rows = [], []
    grp = defaultdict(list)
    for r in recs:
        grp[(r.cond, r.feat, r.K, r.T)].append(r)
    for (cond, feat, K, T), rr in sorted(grp.items(),
                                         key=lambda x: (x[0][0], x[0][1],
                                                        x[0][2] or 999,
                                                        x[0][3])):
        hits = np.array([r.hit for r in rr])
        cl = np.array([f"{r.dev}|{r.block}" for r in rr])
        lo, hi = boot_ci(hits, cl, rng)
        per = defaultdict(list)
        for r in rr:
            per[r.dev].append(r)
        rec = {g: float(np.mean([x.hit for x in v]))
               for g, v in per.items()}
        sizes = np.array([r.G for r in rr])
        rows.append(dict(
            **extra, condition=cond, feature=feat, K=klabel(K), T=T,
            accuracy=float(hits.mean()), ci_lo=lo, ci_hi=hi,
            chance=float(np.mean(1.0 / sizes)),
            macro_recall=float(np.mean(list(rec.values()))),
            macro_chance=float(np.mean([np.mean([1.0 / x.G for x in v])
                                        for v in per.values()])),
            n_decisions=int(hits.size),
            n_query_blocks=int(len(set(cl))),
            n_devices_queried=len(per),
            gallery_size_hist=json.dumps(
                {int(k): int(v) for k, v in
                 zip(*np.unique(sizes, return_counts=True))}),
            enrol_capped_frac=float(np.mean([r.capped for r in rr]))))
        for g, v in sorted(per.items()):
            conf = defaultdict(int)
            for x in v:
                conf[names[x.pred]] += 1
            dev_rows.append(dict(
                **extra, condition=cond, feature=feat, K=klabel(K), T=T,
                device=names[g], recall=rec[g], n_decisions=len(v),
                n_query_blocks=len({x.block for x in v}),
                chance=float(np.mean([1.0 / x.G for x in v])),
                predicted=json.dumps(dict(conf))))
    return rows, dev_rows


DAYPAIR_KT = ((1, 1), (0, 1), (0, 32))
DAYPAIR_FEATURES = ("angles", "angles_rq", "format", "angles+format")


def daypair_rows(recs: list, names: list[str], extra: dict) -> list[dict]:
    """Chance and recall for every (enrol day, query day, device, gallery)
    cell: the gallery and so the chance differ from cell to cell."""
    grp = defaultdict(list)
    for r in recs:
        if (r.K, r.T) in DAYPAIR_KT and r.feat in DAYPAIR_FEATURES:
            grp[(r.cond, r.feat, r.K, r.T, r.de, r.dq, r.dev, r.gal)
                ].append(r.hit)
    out = []
    for (cond, feat, K, T, de, dq, dev, gal), h in sorted(
            grp.items(), key=lambda x: tuple(map(str, x[0]))):
        out.append(dict(**extra, condition=cond, feature=feat, K=klabel(K),
                        T=T, enrol_day=de, query_day=dq, device=names[dev],
                        gallery="+".join(names[g] for g in gal),
                        gallery_size=len(gal), chance=1.0 / len(gal),
                        n_decisions=len(h), recall=float(np.mean(h))))
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"[out] {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Descriptive tables
# ---------------------------------------------------------------------------
def session_rows(w: dict, ts_all: dict, gap_s: float) -> list[dict]:
    rows = []
    for i, m in enumerate(w["macs"]):
        t = ts_all[str(m)]
        sid = R.cut_sessions(t, gap_s)
        for s in np.unique(sid):
            tt = t[sid == s]
            sel = (w["dev"] == i) & (w["session"] == s)
            rows.append(dict(
                gap_s=gap_s, device=R.short(str(m)), session=int(s),
                start=float(tt[0]), end=float(tt[-1]),
                duration_s=float(tt[-1] - tt[0]), frames=int(tt.size),
                windows=int(sel.sum()),
                days=";".join(sorted(set(w["day"][sel].tolist())))))
    return rows


def session_stats(rows: list[dict]) -> dict:
    out = {}
    for dev in sorted({r["device"] for r in rows}):
        rr = [r for r in rows if r["device"] == dev]
        usable = [r for r in rr if r["windows"] >= 1]
        dur = np.array([r["duration_s"] for r in usable])
        win = np.array([r["windows"] for r in usable])
        out[dev] = dict(
            sessions_total=len(rr), sessions_with_window=len(usable),
            duration_s_min=float(dur.min()) if dur.size else None,
            duration_s_median=float(np.median(dur)) if dur.size else None,
            duration_s_max=float(dur.max()) if dur.size else None,
            windows_median=float(np.median(win)) if win.size else None,
            windows_min=int(win.min()) if win.size else None,
            windows_max=int(win.max()) if win.size else None)
    return out


def device_set_report(w: dict, devs: list[str], gap_s: float) -> dict:
    per = {}
    tab = [json.loads(g) for g in w["geom_table"]]
    for i, m in enumerate(devs):
        s = w["dev"] == i
        dur = w["ts1"][s] - w["ts0"][s]
        g, c = np.unique(w["geom"][s], return_counts=True)
        per[R.short(m)] = dict(
            mac=m, label=R.label(m)[0],
            mobility=R.label(m)[1], windows=int(s.sum()),
            days=sorted(set(w["day"][s].tolist())),
            window_seconds_median=float(np.median(dur)),
            window_seconds_p10=float(np.percentile(dur, 10)),
            window_seconds_p90=float(np.percentile(dur, 90)),
            formats={json.dumps(tab[int(a)]): int(b) for a, b in zip(g, c)})
    return per


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report window counts and episode counts only")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    tb = Table(w)
    if a.dry_run:
        rng = np.random.default_rng(SEED)
        n = {c: sum(1 for _ in episodes(tb, c, rng, 1)) for c in CONDITIONS}
        est_recs = sum(n.values()) * a.trials * len(K_GRID) * len(T_GRID) \
            * len(FEATURES)
        print(json.dumps(dict(windows=int(len(w["dev"])),
                              episodes_per_trial=n,
                              est_records=est_recs,
                              est_record_mb=est_recs * 600 / 1e6), indent=2))
        return 0

    # Descriptives -------------------------------------------------------
    ts_all = R.all_timestamps(devs)
    gap_rep = R.gap_report(ts_all)
    sess_rows, sess_stats = [], {}
    for g in (R.SESSION_GAP_S, *R.SENSITIVITY_GAPS_S):
        wg = w if g == R.SESSION_GAP_S else R.get_windows(devs, g)
        rr = session_rows(wg, ts_all, g)
        sess_rows += rr
        sess_stats[str(int(g))] = session_stats(rr)
    write_csv(OUT["sessions"], sess_rows)
    excl = R.exclusions(devs)
    ver = {}
    for d in ("2026-05-10", "2026-05-15"):
        ver.update(R.verify_decoder(X.ARCHIVE_DIR / f"bfi_{d}.sqlite", devs))
    X.write_json(OUT["device_set"], dict(
        devices=device_set_report(w, devs, R.SESSION_GAP_S),
        exclusions=excl,
        angles=list(R.ANGLE_NAMES), n_sc_bins=R.N_SC_BINS,
        window_frames=R.WINDOW_FRAMES, feature_dim=R.FEATURE_DIM,
        session_gap_s=R.SESSION_GAP_S, gap_report=gap_rep,
        session_stats=sess_stats, decoder_check=ver))
    del ts_all
    X.log_rss("descriptives")

    rng_b = np.random.default_rng(SEED + 1)
    main_rows, dev_rows, dp_rows = [], [], []
    # R1/R2 at the chosen gap, full grid --------------------------------
    for ci, cond in enumerate(CONDITIONS):
        t = time.time()
        recs = run_condition(tb, cond, K_GRID, T_GRID, a.trials, SEED + ci)
        r1, r2 = aggregate(recs, tb.names, dict(gap_s=int(R.SESSION_GAP_S)),
                           rng_b)
        main_rows += r1
        dev_rows += r2
        dp_rows += daypair_rows(recs, tb.names,
                                dict(gap_s=int(R.SESSION_GAP_S)))
        print(f"[eval] {cond}: {len(recs)} records {time.time() - t:.0f}s",
              file=sys.stderr)
        del recs
        X.log_rss(cond)
    # Gap sensitivity, reduced grid -------------------------------------
    for g in R.SENSITIVITY_GAPS_S:
        tbg = Table(R.get_windows(devs, g))
        for ci, cond in enumerate(CONDITIONS):
            recs = run_condition(tbg, cond, K_GRID_SENS, T_GRID_SENS,
                                 a.trials, SEED + ci)
            r1, _ = aggregate(recs, tbg.names, dict(gap_s=int(g)), rng_b)
            main_rows += r1
            del recs
        del tbg
        X.log_rss(f"gap {g:.0f}")
    write_csv(OUT["main"], main_rows)
    write_csv(OUT["per_device"], dev_rows)
    write_csv(OUT["daypair"], dp_rows)

    # R3: format-matched pair --------------------------------------------
    gi = [json.loads(s) for s in w["geom_table"]].index(PAIR_FORMAT)
    pair_ids = [tb.names.index(n) for n in PAIR_DEVICES]
    keep = np.isin(w["dev"], pair_ids) & (w["geom"] == gi)
    tp = Table(w, keep)
    pair_rows, pair_dev = [], []
    for ci, cond in enumerate(PAIR_CONDITIONS):
        recs = run_condition(tp, cond, K_GRID, T_GRID, a.trials,
                             SEED + 10 + ci)
        r1, r2 = aggregate(recs, tp.names, dict(gap_s=int(R.SESSION_GAP_S)),
                           rng_b)
        pair_rows += r1
        pair_dev += r2
        del recs
    write_csv(OUT["pair"], pair_rows)
    write_csv(OUT["pair_dev"], pair_dev)
    pair_blocks = {tp.names[k[0]] + "|" + str(k[1]) + "|" + k[2]: int(v.size)
                   for k, v in tp.blocks.items()}

    X.write_json(OUT["summary"], dict(
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        trials=a.trials, k_grid=[klabel(k) for k in K_GRID],
        t_grid=list(T_GRID), pair_format=PAIR_FORMAT,
        pair_blocks_windows=pair_blocks))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
