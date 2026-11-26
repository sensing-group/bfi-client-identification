#!/usr/bin/env python3
"""Address rotation replayed on the residential capture.

802.11bi frame anonymization would change a multi-link client's per-link
address every epoch. No client ships it and none of the six devices is
multi-link, so this replay only asks what such a change costs the observer of
Section 5: frames are grouped by address before windowing, so an address that
changes every E seconds stops a window at the epoch edge. Every device gets
its own random epoch phase; frames left over at an edge, fewer than one
window, are lost. The decision itself reads no address and is that of r_eval.

Enrollment
  session  the labeled session is known from outside the radio across its
           epochs, as in Section 3; the query is any other session.
  epoch    the label covers one epoch only, one address: the observer enrolls
           the windows of that epoch and queries any other one.
  pair     as session, on r_eval's format-matched pair (laptop and vacuum in
           their shared format), where the format cannot decide.

Reported per (E, window frames, enrollment): accuracy and macro recall of the
angles and of the format alone, at T = 1 (averaging T windows would cross
addresses), and the share of frames that still fall in a window. Recall is
over the windows that exist; a device with no window at some E drops out,
which n_devices_queried shows. E = none with 32-frame windows runs r_eval's
own grids and seeds and must reproduce figures/r_main.csv.

Usage: python3 scripts/r_address_rotation.py [--trials 10] [--phases 5] [--dry-run]
Outputs (figures/): r_address_rotation.csv, r_address_rotation_per_device.csv,
r_address_rotation_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np

import r_common as R
import r_eval as E
import xres_common as X

EPOCHS_S = (None, 3600.0, 600.0, 60.0)
WINDOWS = (32, 16, 8)
CONDITIONS = {"session": ("same_day", "other_day_any"), "epoch": ("other_day_any",),
              "pair": ("other_day_any",)}
FEATURES = ("angles", "format")
K_GRID = (1, 16, 0)
T_GRID = (1,)
N_PHASES = 5                    # N_PHASES x N_TRIALS = r_eval's 50 draws
N_TRIALS = 10
SEED = 7000                     # stream of the epoch phases and of the draws
REPRO_TOL = 1e-9
OUT_CSV = R.FIG_DIR / "r_address_rotation.csv"
OUT_DEV = R.FIG_DIR / "r_address_rotation_per_device.csv"
OUT_JSON = R.FIG_DIR / "r_address_rotation_summary.json"


def per_epoch_blocks(w: dict) -> dict:
    """The window table with each (session, epoch) as its own session."""
    _, inv = np.unique(np.stack([w["dev"], w["session"], w["epoch"]], 1), axis=0, return_inverse=True)
    return {**w, "session": inv.reshape(-1)}


def enrollment_table(w: dict, mode: str) -> E.Table:
    if mode == "session":
        return E.Table(w)
    if mode == "epoch":
        return E.Table(per_epoch_blocks(w))
    names = [R.short(str(m)) for m in w["macs"]]
    gi = [json.loads(s) for s in w["geom_table"]].index(E.PAIR_FORMAT)
    return E.Table(w, np.isin(w["dev"], [names.index(n) for n in E.PAIR_DEVICES]) & (w["geom"] == gi))


def frames_per_device(devs: list[str]) -> np.ndarray:
    ts = R.all_timestamps(devs)
    return np.array([ts[m].size for m in devs], float)


def reproduction_check(devs: list[str], w: dict) -> float:
    """Largest gap to r_main.csv of r_eval's own run on this window table."""
    tb = E.Table(w)
    ref = {(r["condition"], r["feature"], r["K"], r["T"]): float(r["accuracy"]) for r in _r_main()}
    worst = 0.0
    for ci, cond in enumerate(E.CONDITIONS):
        rows, _ = E.aggregate(E.run_condition(tb, cond, E.K_GRID, E.T_GRID, E.N_TRIALS, E.SEED + ci),
                              tb.names, {}, np.random.default_rng(E.SEED + 1))
        worst = max([worst] + [abs(r["accuracy"] - ref[(r["condition"], r["feature"], r["K"], str(r["T"]))])
                               for r in rows])
    return worst


def _r_main() -> list[dict]:
    with open(E.OUT["main"], newline="") as f:
        return [r for r in csv.DictReader(f) if r["gap_s"] == str(int(R.SESSION_GAP_S))]


def run_cell(devs, frames, epoch_s, n_win, n_phases, trials):
    rows, dev_rows, cover = [], [], []
    for ph in range(1 if epoch_s is None else n_phases):
        rng = np.random.default_rng([SEED, ph, int(epoch_s or 0), n_win])
        phase = rng.uniform(0.0, epoch_s or 1.0, len(devs))
        w = R.build_windows(devs, R.SESSION_GAP_S, epoch_s, phase, n_win)
        cover.append(np.bincount(w["dev"], minlength=len(devs)) * n_win / frames)
        for mode, conds in CONDITIONS.items():
            tb = enrollment_table(w, mode)
            for ci, cond in enumerate(conds):
                n = trials * (n_phases if epoch_s is None else 1)
                recs = [r for r in E.run_condition(tb, cond, K_GRID, T_GRID, n, SEED + ph * 10 + ci)
                        if r.feat in FEATURES]
                extra = dict(epoch_s=epoch_s or "none", window_frames=n_win, enrollment=mode, phase=ph)
                a, b = E.aggregate(recs, tb.names, extra, np.random.default_rng([SEED, ph, ci]))
                rows += a
                dev_rows += b
                del recs
        del w
        X.log_rss(f"E={epoch_s} W={n_win} phase {ph}")
    return rows, dev_rows, np.mean(cover, axis=0)


def device_recall(dev_rows: list[dict]) -> dict:
    """Mean recall over phases per (E, window frames, enrollment, condition, feature, K, device)."""
    groups: dict[str, list[float]] = {}
    for r in dev_rows:
        k = "|".join(str(r[c]) for c in ("epoch_s", "window_frames", "enrollment", "condition", "feature", "K", "device"))
        groups.setdefault(k, []).append(r["recall"])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def summarise(rows, dev_rows, cover, names, repro, trials, phases, runtime) -> dict:
    cells = {}
    keys = sorted({(str(r["epoch_s"]), r["window_frames"], r["enrollment"], r["condition"], r["feature"], r["K"])
                   for r in rows})
    for k in keys:
        rr = [r for r in rows if (str(r["epoch_s"]), r["window_frames"], r["enrollment"], r["condition"],
                                  r["feature"], r["K"]) == k]
        cells["|".join(map(str, k))] = dict(
            accuracy=float(np.mean([r["accuracy"] for r in rr])),
            accuracy_sd=float(np.std([r["accuracy"] for r in rr], ddof=1)) if len(rr) > 1 else 0.0,
            macro_recall=float(np.mean([r["macro_recall"] for r in rr])),
            chance=float(np.mean([r["chance"] for r in rr])),
            devices_queried=min(r["n_devices_queried"] for r in rr), phases=len(rr))
    return dict(trials=trials, phases=phases, epochs_s=[str(e) for e in EPOCHS_S], windows=list(WINDOWS),
                repro_max_abs_diff=repro, cells=cells, device_recall=device_recall(dev_rows),
                frames_in_windows={f"{e}|{n}": dict(zip(names, map(float, c)))
                                   for (e, n), c in cover.items()},
                runtime_s=runtime, peak_rss_mb=X.peak_rss_mb())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--phases", type=int, default=N_PHASES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(f"{len(EPOCHS_S)} epochs x {len(WINDOWS)} windows, {a.phases} phases, {a.trials} trials; "
              f"one day cache at a time, as r_common (~600 MB)")
        return 0
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    repro = reproduction_check(devs, R.build_windows(devs, R.SESSION_GAP_S))
    print(f"[repro] max |r_main - rerun| = {repro:.2e}", file=sys.stderr)
    if repro > REPRO_TOL:
        raise SystemExit("window builder does not reproduce r_main.csv")
    frames = frames_per_device(devs)
    rows, dev_rows, cover = [], [], {}
    for n_win in WINDOWS:
        for epoch_s in EPOCHS_S:
            t = time.time()
            r, d, c = run_cell(devs, frames, epoch_s, n_win, a.phases, a.trials)
            rows += r
            dev_rows += d
            cover[(epoch_s or "none", n_win)] = c
            print(f"[cell] E={epoch_s} W={n_win} {time.time() - t:.0f}s", file=sys.stderr, flush=True)
            E.write_csv(OUT_CSV, rows)
            E.write_csv(OUT_DEV, dev_rows)
    names = [R.short(m) for m in devs]
    X.write_json(OUT_JSON, summarise(rows, dev_rows, cover, names, repro, a.trials, a.phases, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
