#!/usr/bin/env python3
"""R4: offline processing cost on this machine, single thread.

Stages timed (median over REPEATS runs, wall clock, perf_counter):
  read        sqlite fetch of (ts, src_mac, body_hex), per frame; an equal
              share of N_FRAMES from each of the six devices
  decode1     per-frame path: hex -> bytes, MIMO-control parse, first-column
              Givens decode (4 angles), dequantise, bin to 32 subcarrier
              groups; one frame at a time (what an online sniffer would do)
  decodeB     same, batched by (format, body length) as the cache builder does
  window      320-d window vector from 32 binned frames
  query       z-score + 1-NN of one averaged query against a gallery of six
              devices x K windows, for K in GALLERY_K

Output: figures/r_cost.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # single-thread numbers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

COST_DAY = os.environ.get("BFI_R_COST_DAY", "2026-05-15")
N_FRAMES = 5000
REPEATS = 5
N_WINDOWS = 2000
N_QUERIES = 500
GALLERY_K = (1, 16, 1000)
N_GALLERY_DEV = 6
SEED = 0
OUT = R.FIG_DIR / "r_cost.json"


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def timed(fn, repeats=REPEATS) -> float:
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    devs = R.device_macs()
    p = X.ARCHIVE_DIR / f"bfi_{COST_DAY}.sqlite"
    if a.dry_run:
        print(json.dumps(dict(day=str(p), frames=N_FRAMES,
                              est_mb=N_FRAMES * 3000 / 1e6), indent=2))
        return 0
    q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} = ? "
         "ORDER BY ts LIMIT %d" % (N_FRAMES // len(devs)))

    def read():
        """Equal share of frames per device, so every report format of the
        six devices present that day is timed."""
        c = X.connect_ro(p)
        rows = [r for m in devs for r in c.execute(q, (m,)).fetchall()]
        c.close()
        return rows

    rows = read()
    t_read = timed(read)
    hexes = [r[2] for r in rows if r[2] is not None]
    n = len(hexes)

    def decode_one():
        out = []
        for h in hexes:
            raw = bytes.fromhex(h)
            g = X.AD.parse_geometry(raw)
            res = R.decode_first_column([raw], g)
            if res is not None:
                out.append(R.bin_frames(res[0])[0])
        return out

    def decode_batch():
        groups = defaultdict(list)
        gd = {}
        for h in hexes:
            raw = bytes.fromhex(h)
            g = X.AD.parse_geometry(raw)
            k = tuple(g[f] for f in R.GEOM_FIELDS)
            gd[k] = g
            groups[(k, len(raw))].append(raw)
        out = []
        for (k, _L), bodies in groups.items():
            res = R.decode_first_column(bodies, gd[k])
            if res is not None:
                out.append(R.bin_frames(res[0]))
        return out

    frames = np.concatenate(decode_batch())
    t_dec1 = timed(decode_one)
    t_decB = timed(decode_batch)
    geoms = sorted({json.dumps(list(
        tuple(X.AD.parse_geometry(bytes.fromhex(h))[f]
              for f in R.GEOM_FIELDS))) for h in hexes})

    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, frames.shape[0] - R.WINDOW_FRAMES, N_WINDOWS)

    def windows():
        return [R.window_feature(frames[s:s + R.WINDOW_FRAMES])
                for s in starts]

    W = np.asarray(windows())
    t_win = timed(windows)
    mu, sd = W.mean(0), W.std(0) + 1e-9
    qs = W[rng.integers(0, N_WINDOWS, N_QUERIES)]
    t_q = {}
    for K in GALLERY_K:
        gal = rng.standard_normal((N_GALLERY_DEV * K, R.FEATURE_DIM))
        lab = np.repeat(np.arange(N_GALLERY_DEV), K)

        def query():
            for v in qs:
                z = (v - mu) / sd
                d = ((gal - z) ** 2).sum(1)
                _ = lab[int(np.argmin(d))]
        t_q[str(K)] = timed(query) / N_QUERIES * 1e6

    rep = dict(
        machine=dict(cpu=cpu_name(), python=platform.python_version(),
                     numpy=np.__version__,
                     threads=os.environ.get("OPENBLAS_NUM_THREADS")),
        day=COST_DAY, frames=n, formats=geoms, repeats=REPEATS,
        us_per_frame=dict(
            sqlite_read=t_read / len(rows) * 1e6,
            decode_bin_single=t_dec1 / n * 1e6,
            decode_bin_batched=t_decB / n * 1e6),
        us_per_window_feature=t_win / N_WINDOWS * 1e6,
        us_per_window_incl_decode_single=(t_dec1 / n * R.WINDOW_FRAMES
                                          + t_win / N_WINDOWS) * 1e6,
        us_per_query_1nn={f"K={k}_x{N_GALLERY_DEV}dev": v
                          for k, v in t_q.items()},
        frames_per_s_single_core=n / t_dec1,
        peak_rss_mb=X.peak_rss_mb())
    X.write_json(OUT, rep)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
