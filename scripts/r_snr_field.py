#!/usr/bin/env python3
"""The per-stream SNR field of the report as an observer's only input, on the
residential capture.

A compressed beamforming report carries, before its angles, one average-SNR
byte per stream. It follows the channel's singular values, which a unitary
rotation of the sounding does not change (the singular values of H Q are those
of H), so the rotation of Section 7.2 leaves this field as it is.

Windows and episodes are those of r_eval; each window's feature is the mean
and standard deviation of the stream-1 byte over its frames, or of every
stream's byte. Nothing here reads the angles.

The access point can move the field only as a whole, by lowering the power of
the sounding: a drop of d dB lowers every stream's SNR by d, and it knows d, so
it can undo it. Replayed here, d uniform on [0, A], drawn fresh per report or
per session (one drop per device and session). Two observers read the result:
"level" as above on stream 1, and "shape", what a common drop leaves unchanged,
the spread of stream 1 and the mean and spread of the gap between the streams.
"shape" runs on the two-stream windows only, so that the presence of a second
stream, which is format, cannot decide. r_eval's "angles+format" rows gate the
SNR vector by the format, the two together being what the rotation leaves.

Usage: python3 scripts/r_snr_field.py [--dry-run]
Outputs (figures/): r_snr_field.csv, r_snr_field_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np

import r_common as R
import r_eval as E
import xres_common as X

K_GRID = (1, 16, 0)
T_GRID = (1, 32)
POWER_DROPS_DB = (3.0, 6.0, 12.0)
POWER_MODES = ("report", "session")
OBSERVERS = ("level", "shape")
BYTES_PER_DB = 4                    # the field's step is 0.25 dB
BYTE_MIN, BYTE_MAX = -128, 127
N_DRAWS = 5                         # N_DRAWS x trials / N_DRAWS keeps r_eval's 50 draws
POWER_SEED = 8000
CONDITIONS = ("same_day", "other_day_any")
FEATURES = ("angles", "format", "angles+format")   # "angles" is r_eval's name for the vector under test
N_STREAMS = 2                       # every device reports one or two streams
OUT_CSV = R.FIG_DIR / "r_snr_field.csv"
OUT_JSON = R.FIG_DIR / "r_snr_field_summary.json"


def snr_bytes(raw: bytes) -> tuple[int, np.ndarray] | None:
    """(streams, signed SNR bytes) of one report body, or None if undecodable."""
    g = X.AD.parse_geometry(raw)
    if g is None:
        return None
    s = 2 + g["mimo_len"]
    return g["nc"], np.frombuffer(raw[s:s + g["nc"]], np.int8)


def frame_snr(devs: list[str]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, list[int]]]:
    """Per device: sorted timestamps and (n, N_STREAMS) SNR bytes, NaN for an absent stream;
    and per report format the body lengths seen, which encryption would not change."""
    ts, val, lengths = defaultdict(list), defaultdict(list), defaultdict(set)
    for p in X.day_paths():
        c = X.connect_ro(p)
        q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
             f"({','.join('?' * len(devs))}) ORDER BY ts")
        for t, m, bh in c.execute(q, devs):
            r = None if bh is None else snr_bytes(bytes.fromhex(bh))
            if r is None:
                continue
            v = np.full(N_STREAMS, np.nan)
            v[:r[0]] = r[1][:N_STREAMS]
            ts[devs.index(m)].append(t)
            val[devs.index(m)].append(v)
            raw = bytes.fromhex(bh)
            g = X.AD.parse_geometry(raw)
            lengths[json.dumps([g[k] for k in R.GEOM_FIELDS])].add(len(raw))
        c.close()
        X.log_rss(p.stem)
    return ({i: (np.asarray(ts[i]), np.asarray(val[i])) for i in ts},
            {k: sorted(v) for k, v in lengths.items()})


def lowered(snr: dict, mode: str, drop_db: float, rng) -> dict:
    """The field after a sounding power drop per report or per session."""
    out = {}
    for i, (t, v) in snr.items():
        if mode == "report":
            d = rng.uniform(0.0, drop_db, t.size)
        else:
            d = rng.uniform(0.0, drop_db, int(R.cut_sessions(t, R.SESSION_GAP_S)[-1]) + 1)
            d = d[R.cut_sessions(t, R.SESSION_GAP_S)]
        out[i] = (t, np.clip(np.round(v - BYTES_PER_DB * d[:, None]), BYTE_MIN, BYTE_MAX))
    return out


def shape_features(w: dict, snr: dict) -> np.ndarray:
    """Spread of stream 1 and mean and spread of the stream gap, per window."""
    out = np.zeros((len(w["dev"]), 3), np.float32)
    for j, (i, a, b) in enumerate(zip(w["dev"], w["ts0"], w["ts1"])):
        t, v = snr[int(i)]
        x = v[np.searchsorted(t, a):np.searchsorted(t, b, side="right")]
        gap = x[:, 0] - x[:, 1]
        out[j] = np.nan_to_num([x[:, 0].std(), np.nanmean(gap), np.nanstd(gap)])
    return out


def two_streams(w: dict) -> dict:
    """The windows of two-stream reports, with their columns only."""
    nc = np.array([json.loads(g)[1] for g in w["geom_table"]])[w["geom"]]
    keep = nc == N_STREAMS
    return {k: (v[keep] if v.ndim and v.shape[0] == keep.size else v) for k, v in w.items()}


def observer_rows(w: dict, f: np.ndarray, trials: int, extra: dict) -> list[dict]:
    tb = E.Table({**w, "feat": f, "feat_rq": f})
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        recs = [r for r in E.run_condition(tb, cond, K_GRID, T_GRID, trials, E.SEED + ci)
                if r.feat in FEATURES]
        rows += E.aggregate(recs, tb.names, extra, np.random.default_rng(E.SEED + 1))[0]
        del recs
    return rows


def window_features(w: dict, snr: dict, n_streams: int) -> np.ndarray:
    out = np.zeros((len(w["dev"]), 2 * n_streams), np.float32)
    for j, (i, a, b) in enumerate(zip(w["dev"], w["ts0"], w["ts1"])):
        t, v = snr[int(i)]
        x = v[np.searchsorted(t, a):np.searchsorted(t, b, side="right"), :n_streams]
        out[j] = np.nan_to_num(np.concatenate([np.nanmean(x, 0), np.nanstd(x, 0)]))
    return out


def summary_cells(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["feature"] != "format":
            groups["|".join(str(r[c]) for c in ("streams", "power", "drop_db", "observer", "feature",
                                                "condition", "K", "T"))].append(r)
    return {k: dict(accuracy=float(np.mean([r["accuracy"] for r in v])),
                    macro_recall=float(np.mean([r["macro_recall"] for r in v])),
                    chance=float(np.mean([r["chance"] for r in v])), draws=len(v))
            for k, v in groups.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print("one SNR byte vector per frame, ~30 MB; windows from the r_common cache")
        return 0
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    w = R.get_windows(devs, R.SESSION_GAP_S)
    snr, lengths = frame_snr(devs)
    rows = []
    for n in (1, N_STREAMS):
        rows += observer_rows(w, window_features(w, snr, n), a.trials,
                              dict(streams=n, power="none", drop_db=0, draw=0, observer="level"))
    w2 = two_streams(w)
    rows += observer_rows(w2, shape_features(w2, snr), a.trials,
                          dict(streams=N_STREAMS, power="none", drop_db=0, draw=0, observer="shape"))
    for mode in POWER_MODES:
        for drop in POWER_DROPS_DB:
            for draw in range(N_DRAWS):
                low = lowered(snr, mode, drop, np.random.default_rng([POWER_SEED, draw, int(drop)]))
                extra = dict(power=mode, drop_db=drop, draw=draw)
                n = max(1, a.trials // N_DRAWS)
                rows += observer_rows(w, window_features(w, low, 1), n, {**extra, "streams": 1, "observer": "level"})
                rows += observer_rows(w2, shape_features(w2, low), n,
                                      {**extra, "streams": N_STREAMS, "observer": "shape"})
            X.log_rss(f"{mode} {drop}")
    E.write_csv(OUT_CSV, rows)
    X.write_json(OUT_JSON, dict(trials=a.trials, runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
                                format_body_lengths=lengths, cells=summary_cells(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
