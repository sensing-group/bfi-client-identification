"""S1: how much calibration data does the cohort-out observer need?

The cohort-out attacker standardises features with statistics from a
calibration cohort of OTHER targets. This script varies the size of that
cohort (random draw per seed) and reports the K=1 attack accuracy on the
fixed held-out targets, no defense, b=4, same features and protocol as
joint_defense_eval.py (defense 'none').

    NTU-Fi HumanID   calibration subjects 0,1,2,3,5,7 drawn from the 7 even
                     subjects (train split), targets = the 7 odd subjects
    He CR1 (real BFI) calibration classes 0,1,2,5,10,20 drawn from the first 20
                     location classes, targets = the 19 remaining classes

n_cal = 0 means no standardisation (raw b=4 quantised features).
Attackers: nn1 (cosine 1-NN) at K=1 and, on NTU-Fi, K=all-of-one-session
(support = every held-out window of recording prefix 'a', queries = prefixes
b and c; He has no session field). SupCon encoder (paper settings) trained on
the drawn calibration subjects/classes, K=1; needs >= 2 classes.

Usage:
    python3 scripts/s1_calibration_size.py --dataset ntufi
    python3 scripts/s1_calibration_size.py --dataset he
    python3 scripts/s1_calibration_size.py --dataset he --dry-run
Writes figures/s_calibration_size.csv (aggregate; both datasets append/replace
their own rows), figures/s_calibration_size_seeds.csv (per seed) and
figures/s_calibration_size_<dataset>_summary.json (runtime, peak RSS).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

import joint_defense_common as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NTUFI_CAL_SIZES = (0, 1, 2, 3, 5, 7)
HE_CAL_SIZES = (0, 1, 2, 5, 10, 20)
DEFAULT_SEEDS = 10
SUPCON_MIN_CLASSES = 2          # SupCon needs at least two classes in the training set
HE_CAL_PER_CLASS = 300          # SupCon training subsample per calibration class (as X1)
HE_SUPCON_EPOCHS = 100          # as X1
NTUFI_SESSION_PREFIX = "a"      # K=all-of-one-session: support = all prefix-'a' windows of a target
K_ONE = "1"
K_SESSION = "all_of_session_a"

OUT_CSV = C.FIG_DIR / "s_calibration_size.csv"
OUT_SEEDS_CSV = C.FIG_DIR / "s_calibration_size_seeds.csv"
AGG_FIELDS = ("dataset", "n_cal", "attacker", "K", "acc_mean", "acc_sd", "chance", "n_seeds")
SEED_FIELDS = ("dataset", "n_cal", "attacker", "K", "seed", "acc", "chance", "cal_ids")


# ---------------------------------------------------------------------------
# Attack rules (identical decision rules to joint_defense_common.multi_query_eval)
# ---------------------------------------------------------------------------

def nn1_k1(Z: np.ndarray, y: np.ndarray, supports: dict) -> float:
    return C.multi_query_eval(Z, y, supports, metric="cos", T_grid=(1,), n_decisions=1, rng=None)[1]


def nn1_multi_support(Z: np.ndarray, y: np.ndarray, sup_idx: np.ndarray, q_idx: np.ndarray) -> float:
    """Cosine 1-NN over a support SET (several labelled windows per target)."""
    S = C._normalise(Z[sup_idx])
    Q = C._normalise(Z[q_idx])
    pred = y[sup_idx][np.argmax(Q @ S.T, axis=1)]
    return float((pred == y[q_idx]).mean())


class Identity:
    def transform(self, X):
        return X


def fit_scaler(Fq_cal: np.ndarray):
    return StandardScaler().fit(Fq_cal) if len(Fq_cal) else Identity()


# ---------------------------------------------------------------------------
# Substrates (features and cohorts exactly as joint_defense_eval.py, defense 'none')
# ---------------------------------------------------------------------------

class NTUFi:
    name = "ntufi"
    cal_sizes = NTUFI_CAL_SIZES
    supcon_epochs = C.SUPCON_EPOCHS

    def __init__(self):
        d = C.load_ntufi_packets()
        A, y, split, prefix = d["A"], d["y"], d["split"], d["prefix"]
        cal = (y % 2 == 0) & np.isin(split, C.NTUFI_CAL_SPLITS)
        held = y % 2 == 1
        Fall = C.ntufi_features(A)
        lo, hi = float(Fall.min()), float(Fall.max())
        Fq = C.quant_midpoint(Fall, lo, hi, C.B_HEADLINE)
        self.Fq_cal, self.y_cal = Fq[cal], y[cal]
        self.Fq_held, self.y_held = Fq[held], y[held]
        self.prefix_held = prefix[held]
        self.cal_ids = np.unique(self.y_cal)
        del d, A, Fall, Fq
        gc.collect()

    def session_split(self):
        sup = np.flatnonzero(self.prefix_held == NTUFI_SESSION_PREFIX)
        q = np.flatnonzero(self.prefix_held != NTUFI_SESSION_PREFIX)
        return sup, q

    def supcon_train_rows(self, chosen):
        m = np.isin(self.y_cal, chosen)
        return self.Fq_cal[m], self.y_cal[m]


class He:
    name = "he"
    cal_sizes = HE_CAL_SIZES
    supcon_epochs = HE_SUPCON_EPOCHS

    def __init__(self):
        d = C.load_he_angles()
        y = d["y"]
        classes = np.unique(y)
        cal_ids = classes[: C.HE_N_CALIB_CLASSES]
        cal = np.isin(y, cal_ids)
        pq, sq = C.he_quantise(d["phi"], d["psi"])
        Fq = C.he_features(pq, sq)
        del d, pq, sq
        gc.collect()
        self.Fq_cal, self.y_cal = Fq[cal], y[cal]
        self.Fq_held, self.y_held = Fq[~cal], y[~cal]
        self.cal_ids = cal_ids
        rng = np.random.default_rng(0)      # same subsample rule as X1
        self.sub_idx = np.concatenate([
            rng.choice(np.flatnonzero(self.y_cal == c),
                       size=min(HE_CAL_PER_CLASS, int((self.y_cal == c).sum())), replace=False)
            for c in cal_ids])
        del Fq

    def session_split(self):
        return None

    def supcon_train_rows(self, chosen):
        idx = self.sub_idx[np.isin(self.y_cal[self.sub_idx], chosen)]
        return self.Fq_cal[idx], self.y_cal[idx]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    timer = C.Timer()
    C.set_vm_limit()
    C.check_rss("start")
    ds = NTUFi() if args.dataset == "ntufi" else He()
    C.check_rss("data loaded")
    y_held = ds.y_held
    chance = 1.0 / len(np.unique(y_held))
    session = ds.session_split()
    print(f"{ds.name}: held {len(y_held)} windows, {len(np.unique(y_held))} targets, chance {chance:.4f}; "
          f"cal pool {len(ds.cal_ids)} ids, {len(ds.y_cal)} windows", flush=True)

    rows = []
    for n_cal in ds.cal_sizes:
        for seed in range(1, args.seeds + 1):
            t0 = timer()
            rng = np.random.default_rng([seed, n_cal])
            chosen = np.sort(rng.choice(ds.cal_ids, size=n_cal, replace=False)) if n_cal else np.array([], int)
            scaler = fit_scaler(ds.Fq_cal[np.isin(ds.y_cal, chosen)])
            Zh = scaler.transform(ds.Fq_held)
            supports = C.pick_supports(y_held, seed)
            cal_str = ",".join(str(int(c)) for c in chosen)
            base = dict(dataset=ds.name, n_cal=n_cal, seed=seed, chance=chance, cal_ids=cal_str)
            rows.append(dict(base, attacker="nn1", K=K_ONE, acc=nn1_k1(Zh, y_held, supports)))
            if session is not None:
                sup, q = session
                rows.append(dict(base, attacker="nn1", K=K_SESSION, acc=nn1_multi_support(Zh, y_held, sup, q)))
            msg = f"nn1={rows[-1]['acc']:.3f}" if session is None else \
                f"nn1={rows[-2]['acc']:.3f} nn1_sess={rows[-1]['acc']:.3f}"
            if not args.no_supcon and n_cal >= SUPCON_MIN_CLASSES:
                Xc, yc = ds.supcon_train_rows(chosen)
                enc = C.train_supcon(scaler.transform(Xc), yc, seed, epochs=ds.supcon_epochs)
                acc = nn1_k1(enc.embed(Zh), y_held, supports)
                rows.append(dict(base, attacker="supcon", K=K_ONE, acc=acc))
                msg += f" supcon={acc:.3f} (train rows {len(yc)})"
                del enc, Xc
            print(f"  n_cal={n_cal:2d} seed {seed:2d} [{cal_str}] {msg}  ({timer() - t0:.1f}s)", flush=True)
            del Zh
            gc.collect()
        C.check_rss(f"after n_cal={n_cal}")
    _write_seeds(rows)
    agg = _aggregate(rows)
    _write_agg(agg, ds.name)
    summary = dict(dataset=ds.name, seeds=args.seeds, chance=chance, cal_sizes=list(ds.cal_sizes),
                   supcon=not args.no_supcon, supcon_epochs=ds.supcon_epochs,
                   he_cal_per_class=HE_CAL_PER_CLASS if ds.name == "he" else None,
                   session_prefix=NTUFI_SESSION_PREFIX if session is not None else None,
                   n_session_support=int(len(session[0])) if session is not None else None,
                   n_session_queries=int(len(session[1])) if session is not None else None,
                   runtime_s=timer(), peak_rss_mb=C.peak_rss_mb(), aggregate=agg)
    (C.FIG_DIR / f"s_calibration_size_{ds.name}_summary.json").write_text(json.dumps(summary, indent=1))
    for r in agg:
        print(f"  {r['dataset']} n_cal={r['n_cal']:2d} {r['attacker']:>6s} K={r['K']:<16s} "
              f"{r['acc_mean']:.3f} +- {r['acc_sd']:.3f}")
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def _aggregate(rows):
    out = []
    keys = sorted({(r["dataset"], r["n_cal"], r["attacker"], r["K"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2], k[3]))
    for d, n, a, K in keys:
        v = [r["acc"] for r in rows if (r["dataset"], r["n_cal"], r["attacker"], r["K"]) == (d, n, a, K)]
        out.append(dict(dataset=d, n_cal=n, attacker=a, K=K, acc_mean=float(np.mean(v)),
                        acc_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        chance=rows[0]["chance"], n_seeds=len(v)))
    return out


def _read_csv(path, fields):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if set(r) >= set(fields)]


def _write_seeds(rows):
    keep = [r for r in _read_csv(OUT_SEEDS_CSV, SEED_FIELDS) if r["dataset"] != rows[0]["dataset"]]
    with open(OUT_SEEDS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SEED_FIELDS))
        w.writeheader()
        w.writerows(keep)
        w.writerows({k: r[k] for k in SEED_FIELDS} for r in rows)


def _write_agg(agg, dataset):
    keep = [r for r in _read_csv(OUT_CSV, AGG_FIELDS) if r["dataset"] != dataset]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(AGG_FIELDS))
        w.writeheader()
        w.writerows(keep)
        w.writerows(agg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ntufi", "he"], required=True)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--no-supcon", action="store_true", help="1-NN only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        est = C.ntufi_estimate_mb() * 3 if args.dataset == "ntufi" else C.he_estimate_mb() * 2.5
        n_enc = (len(NTUFI_CAL_SIZES if args.dataset == "ntufi" else HE_CAL_SIZES) - 2) * args.seeds
        print(f"estimated peak working set ~{est:.0f} MB (VM cap {C.VM_LIMIT_GB} GB, RSS cap {C.MAX_RSS_MB} MB); "
              f"{n_enc} SupCon encoders" + (" (skipped)" if args.no_supcon else ""))
        return
    run(args)


if __name__ == "__main__":
    main()
