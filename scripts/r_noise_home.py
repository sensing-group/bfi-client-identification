#!/usr/bin/env python3
"""The noise defenses of Section 7.2 on the client devices of the home capture.

The defenses' trade-off with the gain is measured on real feedback from 19
positions (joint_defense_eval, manifold_sweep, beamdancer_sweep), where the
true beam is known. This replay asks the other half on real clients: does an
observer of the six labeled devices stay above chance under the budgets that
keep the beam there? The gain cannot be measured here, since the capture holds
no unquantized channel.

Each report's Givens angles are decoded at the device's own codebook. A client
perturbs an angle it holds unquantized, which the capture does not keep, so
the input is drawn uniformly inside the reported cell. Then:

  dpsq_eE       DP-SQ (Seif et al.) at budget E, to the nearest midpoints
  manifold_eE   Manifold DP: Gaussian noise before quantization, phi wrapped, psi
                clipped; sigma = 2 pi * sigma_over_range(E) on both, the calibration
                of manifold_sweep.py, so that budgets compare with the 19 positions
  bd_RP         BeamDancer: every angle index offset within P% of its range

The windows, sessions, episodes and seeds are r_eval's, through r_rotation's
per-day pass, so the "none" rows reproduce figures/r_main.csv.

Usage: python3 scripts/r_noise_home.py [--dry-run] [--trials N]
Outputs (figures/): r_noise_home.csv, r_noise_home_per_device.csv,
r_noise_home_pair.csv, r_noise_home_short_window.csv, r_noise_home_summary.json
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time

import numpy as np

import beamdancer_sweep as B
import joint_defense_common as C
import r_common as R
import r_eval as E
import r_rotation as T
import xres_common as X
from bfi.quantisation import _dp_sq_quantise_1d

DPSQ_EPS = (0.01, 0.5, 2.0)
MANIFOLD_EPS = (8.0, 128.0, 512.0)
BEAMDANCER_R = (10, 25, 100)
MANIFOLD_EPS_LOW = (0.5, 2.0)         # last, so that the other bodies keep their random streams
BODIES = (("none",) + tuple(f"dpsq_e{e}" for e in DPSQ_EPS)
          + tuple(f"manifold_e{e:g}" for e in MANIFOLD_EPS)
          + tuple(f"bd_R{r}" for r in BEAMDANCER_R)
          + tuple(f"manifold_e{e:g}" for e in MANIFOLD_EPS_LOW))
PAIR_SEED_OFFSET = 10                 # r_eval's pair seeds, so that "none" matches r_pair.csv
SHORT_WINDOW_FRAMES = 8               # does the window's averaging carry the result?
SHORT_WINDOW_BODIES = ("none", "manifold_e8", "bd_R100")
PHI_RANGE = (0.0, 2 * np.pi)
PSI_RANGE = (0.0, np.pi / 2)
OUT = dict(main=R.FIG_DIR / "r_noise_home.csv",
           per_device=R.FIG_DIR / "r_noise_home_per_device.csv",
           pair=R.FIG_DIR / "r_noise_home_pair.csv",
           short=R.FIG_DIR / "r_noise_home_short_window.csv",
           summary=R.FIG_DIR / "r_noise_home_summary.json")


def dither(mid: np.ndarray, rng_span: tuple, bits: int,
           rng: np.random.Generator) -> np.ndarray:
    """A point drawn uniformly in the cell of each mid-cell angle."""
    cell = (rng_span[1] - rng_span[0]) / (1 << bits)
    return mid + rng.uniform(-0.5, 0.5, mid.shape) * cell


def noise_angles(q: np.ndarray, geom: dict, body: str,
                 rng: np.random.Generator):
    """The report after a client-side defense: phi (n, n_sub, n_phi), psi
    (n, n_sub, n_psi), radians, for r_rotation's re-quantization."""
    phi, psi = T.dequantise(q, geom)
    if body == "none":
        return phi, psi
    bphi, bpsi = geom["bphi"], geom["bpsi"]
    if body.startswith("bd_R"):
        r_pct = int(body[len("bd_R"):])
        kinds = T.split_kinds(geom)
        out = q.copy()
        out[:, kinds == 0] = B.offset(q[:, kinds == 0], bphi, r_pct, rng)
        out[:, kinds == 1] = B.offset(q[:, kinds == 1], bpsi, r_pct, rng)
        return T.dequantise(out, geom)
    x_phi = dither(phi, PHI_RANGE, bphi, rng)
    x_psi = dither(psi, PSI_RANGE, bpsi, rng)
    if body.startswith("dpsq_e"):
        eps = float(body[len("dpsq_e"):])
        return (_dp_sq_quantise_1d(x_phi, *PHI_RANGE, bphi, eps, rng, circular=True),
                _dp_sq_quantise_1d(x_psi, *PSI_RANGE, bpsi, eps, rng, circular=False))
    sigma = C.manifold_sigma_over_range(float(body[len("manifold_e"):]))
    sigma *= PHI_RANGE[1] - PHI_RANGE[0]
    x_phi = x_phi + rng.normal(0.0, sigma, x_phi.shape)
    x_psi = x_psi + rng.normal(0.0, sigma, x_psi.shape)
    return np.mod(x_phi, PHI_RANGE[1]), np.clip(x_psi, *PSI_RANGE)


def evaluate(w: dict, body: str, trials: int, rng_b):
    tb = E.Table(w)
    rows, dev_rows = [], []
    for ci, cond in enumerate(E.CONDITIONS):
        recs = E.run_condition(tb, cond, E.K_GRID, E.T_GRID, trials, E.SEED + ci)
        r1, r2 = E.aggregate(recs, tb.names, dict(body=body), rng_b)
        rows += r1
        dev_rows += r2
        del recs
    del tb
    gc.collect()
    return rows, dev_rows


def pair_rows(w: dict, body: str, trials: int, rng_b) -> list[dict]:
    """The laptop and the vacuum on their shared format, r_eval's pair protocol:
    the format cannot decide here, so what stays above 50% is the angles."""
    tb = E.Table(w)
    gi = [json.loads(g) for g in w["geom_table"]].index(E.PAIR_FORMAT)
    keep = (np.isin(w["dev"], [tb.names.index(n) for n in E.PAIR_DEVICES])
            & (w["geom"] == gi))
    tp = E.Table(w, keep)
    rows = []
    for ci, cond in enumerate(E.PAIR_CONDITIONS):
        recs = E.run_condition(tp, cond, E.K_GRID, E.T_GRID, trials, E.SEED + PAIR_SEED_OFFSET + ci)
        rows += E.aggregate(recs, tp.names, dict(body=body), rng_b)[0]
        del recs
    del tb, tp
    gc.collect()
    return rows


def short_window_rows(devs: list[str], body: str, trials: int, rng_b) -> list[dict]:
    """The same frames in shorter windows, all six devices and the shared-format
    pair: noise that a 32-frame window averages out survives a shorter one."""
    frames, R.WINDOW_FRAMES = R.WINDOW_FRAMES, SHORT_WINDOW_FRAMES
    w = T.build_windows(devs, R.SESSION_GAP_S, body)
    R.WINDOW_FRAMES = frames
    rows = [dict(r, gallery="six") for r in evaluate(w, body, trials, rng_b)[0]]
    rows += [dict(r, gallery="pair") for r in pair_rows(w, body, trials, rng_b)]
    del w
    gc.collect()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        print(json.dumps(dict(devices=devs, rows=R.dry_run(devs)), indent=2))
        return 0
    for p in X.day_paths():
        R.build_day(p, devs)
    # r_rotation's per-day pass, with the defense in place of the rotation
    T.rotate_angles = noise_angles
    T.VARIANTS = tuple(dict.fromkeys(T.VARIANTS + BODIES))
    T.ROT_CACHE_DIR = R.CACHE_DIR.parent / "r_noise"
    rng_b = np.random.default_rng(E.SEED + 1)
    main_rows, dev_rows, pairs, short = [], [], [], []
    for body in BODIES:
        t = time.time()
        w = T.get_windows(devs, R.SESSION_GAP_S, body)
        m, d = evaluate(w, body, a.trials, rng_b)
        main_rows += m
        dev_rows += d
        pairs += pair_rows(w, body, a.trials, rng_b)
        del w
        if body in SHORT_WINDOW_BODIES:
            short += short_window_rows(devs, body, a.trials, rng_b)
            E.write_csv(OUT["short"], short)
        gc.collect()
        for day in T.variant_dir(body).glob("bfi_*"):
            day.unlink()
        for r in m:
            if r["feature"] == "angles" and r["T"] == 1 and r["K"] in ("1", "all"):
                print(f"[{body}] {r['condition']} K={r['K']} "
                      f"acc={r['accuracy']:.3f} macro={r['macro_recall']:.3f}",
                      file=sys.stderr)
        print(f"[{body}] {time.time() - t:.0f}s peak RSS "
              f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
        E.write_csv(OUT["main"], main_rows)
        E.write_csv(OUT["per_device"], dev_rows)
        E.write_csv(OUT["pair"], pairs)
    ref = T.reference_rows()
    diff = max(abs(r["accuracy"] - ref[(r["condition"], r["feature"], r["K"], r["T"])])
               for r in main_rows if r["body"] == "none"
               and (r["condition"], r["feature"], r["K"], r["T"]) in ref)
    X.write_json(OUT["summary"], dict(
        bodies=list(BODIES), trials=a.trials, none_vs_r_main_max_abs_diff=diff,
        cells={f"{r['body']}|{r['feature']}|{r['condition']}|{r['K']}|{r['T']}":
               dict(accuracy=r["accuracy"], macro_recall=r["macro_recall"],
                    chance=r["chance"]) for r in main_rows},
        pair_cells={f"{r['body']}|{r['feature']}|{r['condition']}|{r['K']}|{r['T']}":
                    dict(accuracy=r["accuracy"], macro_recall=r["macro_recall"],
                         chance=r["chance"]) for r in pairs},
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    shutil.rmtree(T.ROT_CACHE_DIR, ignore_errors=True)
    pair_ref = {(r["condition"], r["feature"], r["K"], int(r["T"])): float(r["accuracy"])
                for r in T.read_csv_rows(R.FIG_DIR / "r_pair.csv")}
    print("[repro] max |r_pair - rerun| = " + format(max(
        abs(r["accuracy"] - pair_ref[(r["condition"], r["feature"], r["K"], r["T"])])
        for r in pairs if r["body"] == "none"), ".2e"), file=sys.stderr)
    print(f"[repro] max |r_main - rerun| = {diff:.2e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
