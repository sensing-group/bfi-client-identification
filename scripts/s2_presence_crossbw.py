"""S2b: He presence classifier trained at one bandwidth, tested at another.

The He presence files identify the bandwidth in their path and name
(<scenario>/<20M|40M|80M>/<scenario>_<MMDD>_<bw>_<0p|1p>_<i>_bfm_direct.npy).
Features follow bfi.framework.datasets.load_he_presence exactly as used for
figures/bfi_learned_he_presence.csv (bfi/demo_learned_attacker_he.py): one row
per file, mean over packets of the 6-bit cell index mapped to the mid-cell
angle, (n_sc, 2) angles flattened; then deterministic quantisation at b=4.

The three bandwidths report 29 / 57 / 122 grouped subcarriers, so a classifier
trained at one bandwidth has no native feature space at another. Two mappings
to a common 29-point grid are evaluated on every (train, test) pair:
    stretch   linear interpolation along the normalised subcarrier position
    center    the central 29 grouped subcarriers (same Hz spacing, Ng=2)
Same-bandwidth cells use the demo's 70/30 split (seed 43); cross-bandwidth
cells train on every file of the training bandwidth and test on every file of
the test bandwidth. Attackers 1nn / lr / mlp as the source CSV, 3 seeds.
The bandwidths were recorded on different days (Meeting: 11-30, 11-29,
11-26), so a cross-bandwidth cell is also a cross-day cell.

Usage:
    python3 scripts/s2_presence_crossbw.py [--scenario Meeting]
Writes figures/s_presence_crossbw.csv and figures/s_presence_crossbw_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import joint_defense_common as C
from bfi.framework import ClosedSetAttacker, deterministic
from bfi.framework.datasets import load_he_presence

# ---------------------------------------------------------------------------
# Constants (match bfi/demo_learned_attacker_he.py)
# ---------------------------------------------------------------------------

HE_PRESENCE_ROOT = Path(os.environ.get("BFI_HE_PRESENCE_ROOT", C.REPO_ROOT / "data" / "he_presence" / "dataset"))
DEFAULT_SCENARIO = "Meeting"
BANDWIDTHS = ("20M", "40M", "80M")
B_CAPTURE = 6
B_EVAL = 4
TRAIN_FRAC = 0.70
SEED_BASE = 42
SPLIT_SEED = SEED_BASE + 1
N_SEEDS = 3
ATTACKERS = ("1nn", "lr", "mlp")
MAPPINGS = ("stretch", "center")
N_ANGLES = 2
COMMON_SC = 29                   # grouped subcarriers at 20 MHz
ANGLE_LO, ANGLE_HI = -float(np.pi), float(np.pi)
CHANCE_BINARY = 0.5
OUT_CSV = C.FIG_DIR / "s_presence_crossbw.csv"
OUT_JSON = C.FIG_DIR / "s_presence_crossbw_summary.json"
FIELDS = ("scenario", "mapping", "train_bw", "test_bw", "attacker", "seed", "acc", "n_train", "n_test",
          "test_majority", "chance")


def to_common_grid(X: np.ndarray, mapping: str) -> np.ndarray:
    """(N, n_sc*2) mean angles -> (N, COMMON_SC*2)."""
    n_sc = X.shape[1] // N_ANGLES
    A = X.reshape(len(X), n_sc, N_ANGLES)
    if n_sc == COMMON_SC:
        return X
    if mapping == "center":
        s = (n_sc - COMMON_SC) // 2
        return A[:, s:s + COMMON_SC, :].reshape(len(X), -1)
    src = np.linspace(0.0, 1.0, n_sc)
    dst = np.linspace(0.0, 1.0, COMMON_SC)
    out = np.empty((len(X), COMMON_SC, N_ANGLES))
    for k in range(N_ANGLES):
        out[:, :, k] = np.stack([np.interp(dst, src, row) for row in A[:, :, k]])
    return out.reshape(len(X), -1)


def quantise(X):
    return deterministic(X, b=B_EVAL, lo=ANGLE_LO, hi=ANGLE_HI, circular=True)


def run(args):
    timer = C.Timer()
    C.set_vm_limit()
    data = {}
    for bw in BANDWIDTHS:
        sub = load_he_presence(HE_PRESENCE_ROOT, scenario=args.scenario, bandwidth=bw, b_capture=B_CAPTURE)
        data[bw] = (sub.X, sub.y)
        print(f"{args.scenario}/{bw}: n={len(sub.y)} (1p={int(sub.y.sum())}, 0p={int((sub.y == 0).sum())}), "
              f"features={sub.X.shape[1]}", flush=True)
    C.check_rss("loaded")
    rows = []
    for mapping in MAPPINGS:
        for tr in BANDWIDTHS:
            Xtr_all, ytr_all = data[tr]
            Xtr_all = to_common_grid(Xtr_all, mapping)
            for te in BANDWIDTHS:
                Xte_all, yte_all = data[te]
                Xte_all = to_common_grid(Xte_all, mapping)
                if tr == te:
                    n = len(ytr_all)
                    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
                    n_train = int(n * TRAIN_FRAC)
                    Xtr, ytr = Xtr_all[perm[:n_train]], ytr_all[perm[:n_train]]
                    Xte, yte = Xte_all[perm[n_train:]], yte_all[perm[n_train:]]
                else:
                    Xtr, ytr, Xte, yte = Xtr_all, ytr_all, Xte_all, yte_all
                Xtr_q, Xte_q = quantise(Xtr), quantise(Xte)
                maj = float(max(yte.mean(), 1 - yte.mean()))
                for atk in ATTACKERS:
                    for s in range(N_SEEDS):
                        acc = ClosedSetAttacker(method=atk, seed=SEED_BASE + s).fit(Xtr_q, ytr).accuracy(Xte_q, yte)
                        rows.append(dict(scenario=args.scenario, mapping=mapping, train_bw=tr, test_bw=te,
                                         attacker=atk, seed=s, acc=acc, n_train=len(ytr), n_test=len(yte),
                                         test_majority=maj, chance=CHANCE_BINARY))
                summary = " ".join(
                    f"{a}={np.mean([r['acc'] for r in rows[-N_SEEDS * len(ATTACKERS):] if r['attacker'] == a]):.3f}"
                    for a in ATTACKERS)
                print(f"  {mapping:>7s} train {tr} test {te}: {summary}  majority={maj:.3f}  "
                      f"(n_train={len(ytr)}, n_test={len(yte)})", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        k = f"{r['mapping']}|{r['train_bw']}->{r['test_bw']}|{r['attacker']}"
        agg.setdefault(k, []).append(r["acc"])
    OUT_JSON.write_text(json.dumps(dict(
        scenario=args.scenario, b_eval=B_EVAL, b_capture=B_CAPTURE, seeds=N_SEEDS, mappings=list(MAPPINGS),
        cells={k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
               for k, v in agg.items()},
        runtime_s=timer(), peak_rss_mb=C.peak_rss_mb()), indent=1))
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        n = sum(len(list((HE_PRESENCE_ROOT / args.scenario / bw).glob("*.npy"))) for bw in BANDWIDTHS)
        print(f"{n} files under {HE_PRESENCE_ROOT / args.scenario}; one file (~1200 x 122 x 2 objects) "
              f"is held at a time, working set well under 1 GB")
        return
    run(args)


if __name__ == "__main__":
    main()
