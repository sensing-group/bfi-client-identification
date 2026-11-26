"""S2: environment-out on real Givens-angle BFI (He et al. localization archive).

The He 'transfer and light weight' archive has five rooms (Conference Room 1,
Conference Room 2, Lab 1, Lab 2, Rooftop); each room's y.pth labels are its
transmitter positions (y_pos.pth maps 1:1 to them). Positions are not shared
across rooms, so enrollment and queries must both come from the held-out
room; what can cross rooms is the calibration (standardisation) statistics.

Per held-out room, 10 seeds, K=1 support per position (one labelled window),
queries = every other window of that room, cosine 1-NN, b=4 codebook features
exactly as joint_defense_eval.py (defense 'none'). Calibration variants:
    none            raw quantised features, no standardisation
    other_rooms     StandardScaler fit on all windows of the other four rooms
    in_room         StandardScaler fit on all windows of the test room (same
                    targets; unsupervised use of the room's own stream)
    in_room_cohort  the paper's cohort-out rule inside the room: a random half of
                    the positions calibrates, the other half are the targets
                    (own chance = 1/|targets|)

Usage:
    python3 scripts/s2_env_out.py
    python3 scripts/s2_env_out.py --dry-run
Writes figures/s_env_out.csv (test_room, n_positions, calib, acc_mean, acc_sd,
chance, n_seeds), figures/s_env_out_seeds.csv and figures/s_env_out_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import joint_defense_common as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HE_ARCHIVE_ROOT = Path(os.environ.get("BFI_HE_ARCHIVE_ROOT", C.HE_ROOT.parent))
ROOMS = ("Conference Room 1", "Conference Room 2", "Lab 1", "Lab 2", "Rooftop")
CALIBS = ("none", "other_rooms", "in_room", "in_room_cohort")
DEFAULT_SEEDS = 10
COHORT_CAL_FRAC = 0.5
OUT_CSV = C.FIG_DIR / "s_env_out.csv"
OUT_SEEDS_CSV = C.FIG_DIR / "s_env_out_seeds.csv"
OUT_JSON = C.FIG_DIR / "s_env_out_summary.json"
AGG_FIELDS = ("test_room", "n_positions", "calib", "acc_mean", "acc_sd", "chance", "n_seeds")


def load_room_features(room: str) -> tuple[np.ndarray, np.ndarray]:
    """b=4 codebook pair-0 features (N, 118) and position labels for one room."""
    d = C.load_he_angles(HE_ARCHIVE_ROOT / room)
    pq, sq = C.he_quantise(d["phi"], d["psi"])
    F = C.he_features(pq, sq)
    y = d["y"]
    del d, pq, sq
    gc.collect()
    return F, y


def nn1_k1(Z, y, supports):
    return C.multi_query_eval(Z, y, supports, metric="cos", T_grid=(1,), n_decisions=1, rng=None)[1]


def run(args):
    timer = C.Timer()
    C.set_vm_limit()
    C.check_rss("start")
    feats, labels = {}, {}
    for room in ROOMS:
        feats[room], labels[room] = load_room_features(room)
        print(f"{room}: {len(labels[room])} windows, {len(np.unique(labels[room]))} positions", flush=True)
        C.check_rss(f"loaded {room}")

    rows = []
    for room in ROOMS:
        F, y = feats[room], labels[room]
        positions = np.unique(y)
        others = np.concatenate([feats[r] for r in ROOMS if r != room])
        scalers = {
            "none": None,
            "other_rooms": StandardScaler().fit(others),
            "in_room": StandardScaler().fit(F),
        }
        del others
        gc.collect()
        for seed in range(1, args.seeds + 1):
            t0 = timer()
            supports = C.pick_supports(y, seed)
            msg = []
            for calib in ("none", "other_rooms", "in_room"):
                Z = F if scalers[calib] is None else scalers[calib].transform(F)
                acc = nn1_k1(Z, y, supports)
                rows.append(dict(test_room=room, n_positions=len(positions), calib=calib, seed=seed,
                                 acc=acc, chance=1.0 / len(positions)))
                msg.append(f"{calib}={acc:.3f}")
                del Z
            rng = np.random.default_rng([seed, 7])
            cal_pos = np.sort(rng.choice(positions, size=int(round(COHORT_CAL_FRAC * len(positions))),
                                         replace=False))
            cal_m = np.isin(y, cal_pos)
            S = StandardScaler().fit(F[cal_m])
            yt = y[~cal_m]
            Zt = S.transform(F[~cal_m])
            n_t = len(np.unique(yt))
            acc = nn1_k1(Zt, yt, C.pick_supports(yt, seed))
            rows.append(dict(test_room=room, n_positions=n_t, calib="in_room_cohort", seed=seed,
                             acc=acc, chance=1.0 / n_t))
            msg.append(f"in_room_cohort={acc:.3f} ({n_t} targets)")
            print(f"  {room:>18s} seed {seed:2d} " + " ".join(msg) + f"  ({timer() - t0:.1f}s)", flush=True)
            del Zt
            gc.collect()
        C.check_rss(f"after {room}")
    agg = _aggregate(rows)
    _write(rows, agg)
    summary = dict(rooms=list(ROOMS), seeds=args.seeds, calibs=list(CALIBS), cohort_cal_frac=COHORT_CAL_FRAC,
                   n_windows={r: int(len(labels[r])) for r in ROOMS},
                   n_positions={r: int(len(np.unique(labels[r]))) for r in ROOMS},
                   runtime_s=timer(), peak_rss_mb=C.peak_rss_mb(), aggregate=agg)
    OUT_JSON.write_text(json.dumps(summary, indent=1))
    for r in agg:
        print(f"  {r['test_room']:>18s} {r['calib']:>15s} n_pos={r['n_positions']:2d} "
              f"{r['acc_mean']:.3f} +- {r['acc_sd']:.3f} chance {r['chance']:.3f}")
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def _aggregate(rows):
    out = []
    for room in ROOMS:
        for calib in CALIBS:
            sel = [r for r in rows if r["test_room"] == room and r["calib"] == calib]
            v = [r["acc"] for r in sel]
            out.append(dict(test_room=room, n_positions=sel[0]["n_positions"], calib=calib,
                            acc_mean=float(np.mean(v)), acc_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                            chance=sel[0]["chance"], n_seeds=len(v)))
    return out


def _write(rows, agg):
    with open(OUT_SEEDS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(AGG_FIELDS))
        w.writeheader()
        w.writerows(agg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        largest = max(C.he_estimate_mb(HE_ARCHIVE_ROOT / r) for r in ROOMS)
        print(f"rooms: {ROOMS}; largest BFI.pth {largest:.0f} MB -> transient peak ~{largest * 2.5:.0f} MB; "
              f"retained features ~{sum(C.he_estimate_mb(HE_ARCHIVE_ROOT / r) for r in ROOMS) / 12:.0f} MB "
              f"(VM cap {C.VM_LIMIT_GB} GB, RSS cap {C.MAX_RSS_MB} MB)")
        return
    run(args)


if __name__ == "__main__":
    main()
