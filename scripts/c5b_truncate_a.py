#!/usr/bin/env python3
"""C5b-T1: the 512-byte firmware truncation of the ESP32-C5 listener, emulated
on the full Pi dataset (A), so that every one of the six labelled devices is
scored as the 5-dollar listener would see it.

Emulation: every action body of (A) is cut to the first BFI_MAX_BODY bytes
BEFORE decoding (exactly what main.c does on the C5), the first-column
Givens angles are decoded from the kept prefix, tones are binned on the
FULL standard tone grid of the format (234 VHT 80/Ng1, 250 HE 80/Ng4) and a
bin is kept only when every tone it covers survived the cut (c5_common
bin_frames_full_grid, the (B) definition). Kept tones per format are
derived from the format's bits per tone, not assumed: 223/250 for the HE
4/2-bit Apple TV, 134/250 for the HE 6/4 LG, 134/234 for the VHT Nc=2 6/4
Apple TV (bedroom), 126/234 for the VHT Nc=1 9/7 MU laptop and vacuum,
84/234 for the Samsung's VHT Nc=2 9/7 MU reports.

Windows and sessions are rebuilt exactly as in r_common (same gap, same
blocks); the protocol is r_eval's (one labelled block enrolled, K windows,
T consecutive query windows, 1-NN, chance = 1/|gallery| per decision), run
under two bin policies:
  common  bins fully covered in EVERY window of the device set being
          compared (AND over devices, the c5_cross_receiver rule); for the
          six-device gallery that is the Samsung's 84 tones = 11 of 32 bins
  none    all 320 dims; a bin missing from a window is set to the pooled
          mean after z-scoring (nan -> 0), so the truncation pattern itself
          acts as a format cue, as it would for a real listener
The format-matched pair (Laptop vs Vacuum, VHT 1/3 9/7 MU) is re-run with
its own common mask (126 tones = 17 bins).

Outputs (figures/): c5b_trunc_main.csv, c5b_trunc_per_device.csv,
c5b_trunc_daypair.csv, c5b_trunc_pair.csv, c5b_trunc_pair_per_device.csv,
c5b_trunc_side_by_side.csv (truncated next to r_main / r_per_device /
r_pair), c5b_trunc_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_PREFIX = "c5b_"
CACHE_DIR = X.CACHE_DIR / "c5b"
T_CACHE_VERSION = 1
BODY_CAP = C.BFI_MAX_BODY                     # 512, the C5 firmware cap
K_GRID = (1, 16, 0)
T_GRID = (1, 4, 16, 32)
N_TRIALS = E.N_TRIALS                         # 50, as r_eval
SEED = E.SEED
MASK_POLICIES = ("common", "none")
CONDITIONS = E.CONDITIONS
PAIR_CONDITIONS = E.PAIR_CONDITIONS
N_BLOCKS = R.FEATURE_DIM // R.N_SC_BINS       # 10 feature blocks x 32 bins
SIDE_KEYS = ("condition", "feature", "K", "T")
OUT = dict(main=X.FIG_DIR / f"{OUT_PREFIX}trunc_main.csv",
           per_device=X.FIG_DIR / f"{OUT_PREFIX}trunc_per_device.csv",
           daypair=X.FIG_DIR / f"{OUT_PREFIX}trunc_daypair.csv",
           pair=X.FIG_DIR / f"{OUT_PREFIX}trunc_pair.csv",
           pair_dev=X.FIG_DIR / f"{OUT_PREFIX}trunc_pair_per_device.csv",
           side=X.FIG_DIR / f"{OUT_PREFIX}trunc_side_by_side.csv",
           summary=X.FIG_DIR / f"{OUT_PREFIX}trunc_summary.json")
REF = dict(main=X.FIG_DIR / "r_main.csv", per_device=X.FIG_DIR / "r_per_device.csv",
           pair=X.FIG_DIR / "r_pair.csv", pair_dev=X.FIG_DIR / "r_pair_per_device.csv")


# ---------------------------------------------------------------------------
# Truncated per-day cache
# ---------------------------------------------------------------------------
def day_cache_path(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.t{BODY_CAP}.v{T_CACHE_VERSION}.npz"


def build_day(p: Path, devs: list[str], force: bool = False) -> dict:
    """Same streaming as r_common.build_day; bodies cut to BODY_CAP bytes;
    bins on the full tone grid, partially covered bins NaN; nsub kept."""
    out = day_cache_path(p.stem)
    if out.exists() and not force:
        return dict(cached=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c = X.connect_ro(p)
    q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN (%s) "
         "ORDER BY ts" % ",".join("?" * len(devs)))
    cur = c.execute(q, devs)
    geoms: dict[tuple, int] = {}
    cols = defaultdict(list)
    n_skip, n_cut = 0, 0
    while True:
        rows = cur.fetchmany(R.FETCH_CHUNK)
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n)
        dev = np.empty(n, np.int8)
        gid = np.full(n, -1, np.int16)
        nsub = np.zeros(n, np.int16)
        feat = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
        feat_rq = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
        groups: dict[tuple, list[int]] = defaultdict(list)
        gdict: dict[tuple, dict] = {}
        bodies: list = [None] * n
        for i, (t, m, bh) in enumerate(rows):
            ts[i] = t
            dev[i] = devs.index(m)
            if bh is None:
                continue
            raw = bytes.fromhex(bh)
            if len(raw) > BODY_CAP:
                raw = raw[:BODY_CAP]
                n_cut += 1
            g, key = R._geom_from_body(raw)
            if g is None:
                continue
            gid[i] = geoms.setdefault(key, len(geoms))
            bodies[i] = raw
            gdict[key] = g
            groups[(key, len(raw))].append(i)
        for (key, _L), idx in groups.items():
            g = gdict[key]
            n_full = X.AD.expected_subcarriers(g)
            res = R.decode_first_column([bodies[i] for i in idx], g) \
                if n_full else None
            if res is None:
                continue
            ia = np.asarray(idx)
            nsub[ia] = res[0].shape[2]
            feat[ia] = C.bin_frames_full_grid(res[0], n_full)
            feat_rq[ia] = C.bin_frames_full_grid(res[2], n_full)
        ok = nsub > 0
        n_skip += int((~ok).sum())
        for k, v in dict(ts=ts[ok], dev=dev[ok], gid=gid[ok], nsub=nsub[ok],
                         feat=feat[ok], feat_rq=feat_rq[ok]).items():
            cols[k].append(v)
        del rows, bodies, groups, feat, feat_rq
    c.close()
    arr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    np.savez(out, **arr, n_undecodable=np.asarray(n_skip), n_cut=np.asarray(n_cut),
             geoms=np.asarray(json.dumps([list(k) for k in geoms])))
    rep = dict(frames=int(arr["ts"].size) if "ts" in arr else 0,
               undecodable=n_skip, cut=n_cut)
    del arr, cols
    gc.collect()
    return rep


def load_day(stem: str) -> dict:
    z = np.load(day_cache_path(stem), allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "geoms"}
    d["geoms"] = [tuple(g) for g in json.loads(str(z["geoms"]))]
    return d


# ---------------------------------------------------------------------------
# Windows (r_common.build_windows + nsub_min)
# ---------------------------------------------------------------------------
def windows_path(gap_s: float) -> Path:
    return CACHE_DIR / f"windows_gap{int(gap_s)}.t{BODY_CAP}.v{T_CACHE_VERSION}.npz"


def all_timestamps(devs: list[str]) -> dict[str, np.ndarray]:
    ts = defaultdict(list)
    for p in X.day_paths():
        d = np.load(day_cache_path(p.stem))
        if "ts" in d.files and d["ts"].size:
            for i, m in enumerate(devs):
                ts[m].append(d["ts"][d["dev"] == i])
    return {m: np.sort(np.concatenate(v)) for m, v in ts.items()}


def build_windows(devs: list[str], gap_s: float) -> dict:
    sess_of = {m: (t, R.cut_sessions(t, gap_s)) for m, t in
               all_timestamps(devs).items()}
    cols = defaultdict(list)
    geom_global: dict[tuple, int] = {}
    for p in X.day_paths():
        d = load_day(p.stem)
        if "ts" not in d or not d["ts"].size:
            continue
        day = p.stem.replace("bfi_", "")
        for i, m in enumerate(devs):
            sel = np.flatnonzero(d["dev"] == i)
            if sel.size == 0:
                continue
            t_all, sid_all = sess_of[m]
            sid = sid_all[np.searchsorted(t_all, d["ts"][sel])]
            for s in np.unique(sid):
                ii = sel[sid == s]
                for w in range(ii.size // R.WINDOW_FRAMES):
                    jj = ii[w * R.WINDOW_FRAMES:(w + 1) * R.WINDOW_FRAMES]
                    g, cnt = np.unique(d["gid"][jj], return_counts=True)
                    gk = d["geoms"][int(g[np.argmax(cnt)])]
                    cols["dev"].append(i)
                    cols["session"].append(int(s))
                    cols["day"].append(day)
                    cols["ts0"].append(float(d["ts"][jj[0]]))
                    cols["ts1"].append(float(d["ts"][jj[-1]]))
                    cols["geom"].append(geom_global.setdefault(gk, len(geom_global)))
                    cols["geom_purity"].append(float(cnt.max()) / jj.size)
                    cols["nsub_min"].append(int(d["nsub"][jj].min()))
                    cols["feat"].append(R.window_feature(d["feat"][jj]))
                    cols["feat_rq"].append(R.window_feature(d["feat_rq"][jj]))
        del d
        gc.collect()
    w = {k: np.asarray(v) for k, v in cols.items()}
    w["geom_table"] = np.asarray([json.dumps(list(k)) for k in geom_global])
    w["macs"] = np.asarray(devs)
    w["gap_s"] = np.asarray(gap_s)
    return w


def get_windows(devs: list[str], gap_s: float, force: bool = False) -> dict:
    p = windows_path(gap_s)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    w = build_windows(devs, gap_s)
    np.savez(p, **w)
    return w


# ---------------------------------------------------------------------------
# Bin masks
# ---------------------------------------------------------------------------
def common_mask(w: dict, keep: np.ndarray) -> np.ndarray:
    """Bins fully covered in every kept window (AND over windows)."""
    gt = [tuple(json.loads(s)) for s in w["geom_table"]]
    m = np.ones(R.N_SC_BINS, bool)
    for g, ns in set(zip(w["geom"][keep].tolist(), w["nsub_min"][keep].tolist())):
        n_full = X.AD.expected_subcarriers(dict(zip(C.GEOM_FIELDS, gt[g])))
        m &= C.covered_bins(n_full, ns)
    return m


def masked_table(w: dict, keep: np.ndarray, policy: str) -> tuple[dict, int]:
    """r_common-layout window dict restricted to the mask policy."""
    if policy == "common":
        mask = common_mask(w, keep)
    else:
        mask = np.ones(R.N_SC_BINS, bool)
    cols = np.tile(mask, N_BLOCKS)
    w2 = dict(w)
    w2["feat"] = w["feat"][:, cols]
    w2["feat_rq"] = w["feat_rq"][:, cols]
    return w2, int(mask.sum())


# ---------------------------------------------------------------------------
# Side by side with the untruncated r_eval numbers
# ---------------------------------------------------------------------------
def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def side_by_side(trunc_rows: list[dict], ref_rows: list[dict], keys, value: str,
                 table: str) -> list[dict]:
    ref = {tuple(r[k] for k in keys): r for r in ref_rows if r.get("gap_s", "300") == "300"}
    out = []
    for r in trunc_rows:
        kk = tuple(str(r[k]) for k in keys)
        rr = ref.get(kk)
        out.append(dict(table=table, **{k: r[k] for k in SIDE_KEYS},
                        device=r.get("device", ""),
                        mask_policy=r["mask_policy"], bins_kept=r["bins_kept"],
                        untruncated=float(rr[value]) if rr else None,
                        truncated=float(r[value]),
                        delta=(float(r[value]) - float(rr[value])) if rr else None,
                        chance=float(r["chance"]), n_decisions=r["n_decisions"]))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        rep = R.dry_run(devs)
        rep["_note"] = ("same streaming as r_common.build_day (fetchmany of "
                        f"{R.FETCH_CHUNK}); r_common measured 572 MB peak")
        print(json.dumps(rep, indent=2))
        return 0
    day_rep = {}
    for p in X.day_paths():
        t = time.time()
        day_rep[p.stem] = build_day(p, devs, force=a.force)
        print(f"[cache] {p.stem} {day_rep[p.stem]} {time.time() - t:.0f}s "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    w = get_windows(devs, R.SESSION_GAP_S, force=a.force)
    X.log_rss("windows")
    names = [R.short(m) for m in devs]
    gt = [json.loads(s) for s in w["geom_table"]]
    fmt_rows = []
    for d, g, ns in sorted(set(zip(w["dev"].tolist(), w["geom"].tolist(),
                                   w["nsub_min"].tolist()))):
        geom = dict(zip(C.GEOM_FIELDS, gt[g]))
        n_full = X.AD.expected_subcarriers(geom)
        fmt_rows.append(dict(device=names[d], format=gt[g], tones_full=n_full,
                             tones_kept=ns, bins_kept=int(C.covered_bins(n_full, ns).sum()),
                             windows=int(((w["dev"] == d) & (w["geom"] == g)).sum())))
    summary = dict(body_cap=BODY_CAP, trials=a.trials, k_grid=[E.klabel(k) for k in K_GRID],
                   t_grid=list(T_GRID), windows=int(w["dev"].size),
                   windows_per_device={names[i]: int((w["dev"] == i).sum())
                                       for i in range(len(devs))},
                   untruncated_windows_per_device=None, formats=fmt_rows,
                   day_cache=day_rep, masks={})
    zr = np.load(R.windows_path(R.SESSION_GAP_S), allow_pickle=False)
    summary["untruncated_windows_per_device"] = {
        names[i]: int((zr["dev"] == i).sum()) for i in range(len(devs))}
    del zr

    rng_b = np.random.default_rng(SEED + 1)
    main_rows, dev_rows, dp_rows, pair_rows, pair_dev = [], [], [], [], []
    all_keep = np.ones(w["dev"].size, bool)
    for policy in MASK_POLICIES:
        w2, nb = masked_table(w, all_keep, policy)
        summary["masks"][policy] = dict(bins_kept=nb, dims=nb * N_BLOCKS)
        tb = E.Table(w2)
        extra = dict(gap_s=int(R.SESSION_GAP_S), mask_policy=policy, bins_kept=nb)
        for ci, cond in enumerate(CONDITIONS):
            t = time.time()
            recs = E.run_condition(tb, cond, K_GRID, T_GRID, a.trials, SEED + ci)
            r1, r2 = E.aggregate(recs, tb.names, extra, rng_b)
            main_rows += r1
            dev_rows += r2
            dp_rows += E.daypair_rows(recs, tb.names, extra)
            print(f"[eval] {policy} {cond}: {len(recs)} records "
                  f"{time.time() - t:.0f}s", file=sys.stderr)
            del recs
            gc.collect()
        # format-matched pair, own mask
        gi = gt.index(E.PAIR_FORMAT)
        pair_ids = [names.index(n) for n in E.PAIR_DEVICES]
        keep = np.isin(w["dev"], pair_ids) & (w["geom"] == gi)
        wp, nbp = masked_table(w, keep, policy)
        summary["masks"][policy]["pair_bins_kept"] = nbp
        tp = E.Table(wp, keep)
        extra_p = dict(gap_s=int(R.SESSION_GAP_S), mask_policy=policy, bins_kept=nbp)
        for ci, cond in enumerate(PAIR_CONDITIONS):
            recs = E.run_condition(tp, cond, K_GRID, T_GRID, a.trials, SEED + 10 + ci)
            r1, r2 = E.aggregate(recs, tp.names, extra_p, rng_b)
            pair_rows += r1
            pair_dev += r2
            del recs
        del tb, tp, w2, wp
        gc.collect()
        X.log_rss(f"policy {policy}")
    E.write_csv(OUT["main"], main_rows)
    E.write_csv(OUT["per_device"], dev_rows)
    E.write_csv(OUT["daypair"], dp_rows)
    E.write_csv(OUT["pair"], pair_rows)
    E.write_csv(OUT["pair_dev"], pair_dev)
    side = []
    if REF["main"].exists():
        side += side_by_side(main_rows, read_csv(REF["main"]), SIDE_KEYS, "accuracy", "main")
    if REF["per_device"].exists():
        side += side_by_side(dev_rows, read_csv(REF["per_device"]),
                             SIDE_KEYS + ("device",), "recall", "per_device")
    if REF["pair"].exists():
        side += side_by_side(pair_rows, read_csv(REF["pair"]), SIDE_KEYS, "accuracy", "pair")
    if REF["pair_dev"].exists():
        side += side_by_side(pair_dev, read_csv(REF["pair_dev"]),
                             SIDE_KEYS + ("device",), "recall", "pair_per_device")
    E.write_csv(OUT["side"], side)
    summary["runtime_s"] = time.time() - t0
    summary["peak_rss_mb"] = X.peak_rss_mb()
    X.write_json(OUT["summary"], summary)
    print(f"[done] {summary['runtime_s']:.0f}s peak RSS {summary['peak_rss_mb']:.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
