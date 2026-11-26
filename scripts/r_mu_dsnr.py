#!/usr/bin/env python3
"""The per-tone SNR of multi-user feedback as an observer's only input, on the
laptop and the vacuum of the residential capture.

A multi-user (MU) compressed beamforming frame carries, after its angles, the
MU Exclusive Beamforming Report: one delta-SNR per reported tone and stream,
4 bits in two's complement, -8 to 7 dB, the tone's SNR against the stream's
average. Like the average-SNR byte it follows the channel's singular values,
which a unitary rotation of the sounding does not change, so the rotation of
Section 7.2 leaves it as it is; unlike that byte it is a profile over
frequency. The laptop and the vacuum report MU feedback in one shared format
(r_eval.PAIR_FORMAT, 122 delta-SNR tones on one stream), so the format cannot
decide between them and chance is 50%.

Windows, episodes and seeds are r_eval's pair protocol on the r_common cache.
Each window's feature is the mean and standard deviation over its frames of
the delta-SNR per tone bin (r_common's 32 bins over the 122 tones), or of the
stream-1 average-SNR byte for comparison. Nothing here reads the angles.

Usage: python3 scripts/r_mu_dsnr.py [--dry-run] [--trials N]
Outputs (figures/): r_mu_dsnr.csv, r_mu_dsnr_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

import r_common as R
import r_eval as E
import xres_common as X

K_GRID = (1, 16, 0)
T_GRID = (1, 32)
FEATURES = ("angles",)              # r_eval's name for the vector under test
OBSERVERS = ("dsnr", "avg_snr")
PAIR_SEED_OFFSET = 10                 # r_eval's pair seeds, as in r_noise_home
NIBBLE_SIGN = 8                     # 4-bit two's complement: n >= 8 is n - 16
NIBBLE_RANGE = 16
OUT_CSV = R.FIG_DIR / "r_mu_dsnr.csv"
OUT_JSON = R.FIG_DIR / "r_mu_dsnr_summary.json"


def mu_fields(raw: bytes) -> tuple[float, np.ndarray] | None:
    """(stream-1 average-SNR byte, delta-SNR per tone of stream 1 in dB) of one
    body in the pair's format, or None for any other body."""
    g = X.AD.parse_geometry(raw)
    if g is None or [g[k] for k in R.GEOM_FIELDS] != E.PAIR_FORMAT:
        return None
    n_exp = X.AD.expected_subcarriers(g)
    _o, _w, _k, bps = X.AD._layout(g["nr"], g["nc"], g["bphi"], g["bpsi"])
    s = 2 + g["mimo_len"]
    start = s + g["nc"] + math.ceil(n_exp * bps / 8)
    tail = np.frombuffer(raw[start:], np.uint8)
    if tail.size == 0:
        return None
    nib = np.empty(2 * tail.size, np.int8)
    nib[0::2] = tail & 0x0F                  # first tone in the low nibble
    nib[1::2] = tail >> 4
    nib = np.where(nib >= NIBBLE_SIGN, nib - NIBBLE_RANGE, nib)
    return float(np.frombuffer(raw[s:s + 1], np.int8)[0]), nib[::g["nc"]].astype(np.float32)


def frame_fields(devs: list[str]) -> tuple[dict, int]:
    """Per pair device: sorted timestamps, average-SNR bytes and (n, tones) delta-SNR."""
    ts, avg, ds = defaultdict(list), defaultdict(list), defaultdict(list)
    n_tones = set()
    for p in X.day_paths():
        c = X.connect_ro(p)
        q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
             f"({','.join('?' * len(devs))}) ORDER BY ts")
        for t, m, bh in c.execute(q, devs):
            r = None if bh is None else mu_fields(bytes.fromhex(bh))
            if r is None:
                continue
            i = devs.index(m)
            ts[i].append(t)
            avg[i].append(r[0])
            ds[i].append(r[1])
            n_tones.add(r[1].size)
        c.close()
        X.log_rss(p.stem)
    if len(n_tones) != 1:
        raise RuntimeError(f"delta-SNR tone counts {sorted(n_tones)}")
    return ({i: (np.asarray(ts[i]), np.asarray(avg[i]), np.stack(ds[i])) for i in ts},
            n_tones.pop())


def window_features(w: dict, fields: dict, observer: str, n_tones: int) -> np.ndarray:
    edges = X.bin_edges(n_tones, R.N_SC_BINS)
    m = np.zeros((n_tones, R.N_SC_BINS))
    m[np.arange(n_tones), edges] = 1.0 / np.bincount(edges, minlength=R.N_SC_BINS)[edges]
    width = 2 * R.N_SC_BINS if observer == "dsnr" else 2
    out = np.full((len(w["dev"]), width), np.nan, np.float32)
    for j, (i, a, b) in enumerate(zip(w["dev"], w["ts0"], w["ts1"])):
        if int(i) not in fields:
            continue
        t, avg, ds = fields[int(i)]
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, b, side="right")
        if hi == lo:
            continue
        x = ds[lo:hi] @ m if observer == "dsnr" else avg[lo:hi, None]
        out[j] = np.concatenate([x.mean(0), x.std(0)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print("about 80k frames of two devices, 122 delta-SNR values each, ~40 MB; windows from the r_common cache")
        return 0
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    names = [R.short(m) for m in w["macs"]]
    pair_macs = [devs[names.index(n)] for n in E.PAIR_DEVICES]
    fields, n_tones = frame_fields(pair_macs)
    fields = {devs.index(pair_macs[i]): v for i, v in fields.items()}
    gi = [json.loads(g) for g in w["geom_table"]].index(E.PAIR_FORMAT)
    keep = np.isin(w["dev"], [names.index(n) for n in E.PAIR_DEVICES]) & (w["geom"] == gi)
    rows, missing = [], {}
    for obs in OBSERVERS:
        f = window_features(w, fields, obs, n_tones)
        missing[obs] = int(np.isnan(f[keep]).all(1).sum())
        tp = E.Table({**w, "feat": f, "feat_rq": f}, keep)
        for ci, cond in enumerate(E.PAIR_CONDITIONS):
            recs = [r for r in E.run_condition(tp, cond, K_GRID, T_GRID, a.trials,
                                               E.SEED + PAIR_SEED_OFFSET + ci)
                    if r.feat in FEATURES]
            rows += E.aggregate(recs, tp.names, dict(observer=obs),
                                np.random.default_rng(E.SEED + 1))[0]
            del recs
        X.log_rss(obs)
    E.write_csv(OUT_CSV, rows)
    X.write_json(OUT_JSON, dict(
        trials=a.trials, runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        n_tones=n_tones, frames={names[i]: int(v[0].size) for i, v in fields.items()},
        windows=int(keep.sum()), windows_without_frames=missing,
        cells={f"{r['observer']}|{r['condition']}|{r['K']}|{r['T']}":
               dict(accuracy=r["accuracy"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                    macro_recall=r["macro_recall"], chance=r["chance"]) for r in rows}))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
