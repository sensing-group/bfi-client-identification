"""X2: where does identity live? Single coordinates vs the joint pattern.

On NTU-Fi (b=4, 174-d mean+std amplitude) and He CR1 (codebook b_phi=4/b_psi=2,
118-d pair-0 angles), in the standardised space the attacker uses:
  (a) cohort-out K=1 1-NN accuracy from each single coordinate alone
      (Euclidean on the scalar) vs all coordinates jointly (cosine and Euclidean);
  (b) within-target vs between-target Euclidean distance distributions
      (Cohen's d, AUC = P(between > within));
  (c) DP-SQ (b=4, eps=0.5) displacement ||x_dp - x|| and the distance between
      two independent DP-SQ draws of one window, against the between-target
      distance; per-window orthogonal Q displacement for contrast.

Usage: python3 scripts/identity_invariant.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import mannwhitneyu

import joint_defense_common as C
from joint_defense_eval import He, NTUFi

N_SEEDS = 10
HE_PAIR_SUBSAMPLE = 2000        # He held reports used for the pairwise-distance analysis
DPSQ_EPS = 0.5
N_COORD_ABOVE = 2.0             # report coordinates above this multiple of chance
NOISE_SEED = 7


def single_coord_acc(X, y, seeds):
    """(D,) mean K=1 1-NN accuracy per coordinate (Euclidean on the scalar)."""
    D = X.shape[1]
    acc = np.zeros(D)
    for s in seeds:
        sup = C.pick_supports(y, s)
        labels = np.array(sorted(sup))
        S = X[[sup[c] for c in labels]]                      # (C, D)
        q = np.setdiff1d(np.arange(len(y)), list(sup.values()))
        pred = labels[np.argmin(np.abs(X[q][:, None, :] - S[None, :, :]), axis=1)]  # (Nq, D)
        acc += (pred == y[q][:, None]).mean(axis=0)
    return acc / len(seeds)


def joint_acc(X, y, seeds, metric):
    return float(np.mean([C.multi_query_eval(X, y, C.pick_supports(y, s), metric=metric, T_grid=(1,),
                                             n_decisions=1, rng=None)[1] for s in seeds]))


def pair_distances(X, y):
    iu = np.triu_indices(len(y), k=1)
    sq = (X ** 2).sum(1)
    d = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0, None))[iu]
    same = (y[:, None] == y[None, :])[iu]
    return d[same], d[~same]


def effect(within, between):
    pooled = np.sqrt((within.var(ddof=1) + between.var(ddof=1)) / 2)
    u = mannwhitneyu(between, within, alternative="greater").statistic
    return dict(within_median=float(np.median(within)), between_median=float(np.median(between)),
                within_mean=float(within.mean()), between_mean=float(between.mean()),
                cohens_d=float((between.mean() - within.mean()) / pooled),
                auc_between_gt_within=float(u / (len(within) * len(between))),
                n_within=int(len(within)), n_between=int(len(between)))


def analyse(X, y, noisy_a, noisy_b, rotated, seeds, label, pair_idx):
    chance = 1.0 / len(np.unique(y))
    sc = single_coord_acc(X, y, seeds)
    out = dict(dataset=label, dim=int(X.shape[1]), chance=chance,
               single_coord=dict(max=float(sc.max()), median=float(np.median(sc)), mean=float(sc.mean()),
                                 min=float(sc.min()), argmax=int(sc.argmax()),
                                 n_above_2x_chance=int((sc > N_COORD_ABOVE * chance).sum())),
               joint_cos=joint_acc(X, y, seeds, "cos"), joint_euc=joint_acc(X, y, seeds, "euc"))
    within, between = pair_distances(X[pair_idx], y[pair_idx])
    out["distances"] = effect(within, between)
    out["distances"]["n_windows_used"] = int(len(pair_idx))
    disp = np.linalg.norm(noisy_a - X, axis=1)
    two = np.linalg.norm(noisy_a - noisy_b, axis=1)
    rot = np.linalg.norm(rotated - X, axis=1)
    bmed, wmed = out["distances"]["between_median"], out["distances"]["within_median"]
    out["dpsq"] = dict(eps=DPSQ_EPS, displacement_median=float(np.median(disp)), displacement_mean=float(disp.mean()),
                       two_draw_distance_median=float(np.median(two)),
                       displacement_over_between_median=float(np.median(disp) / bmed),
                       displacement_over_within_median=float(np.median(disp) / wmed),
                       displacement_over_gap=float(np.median(disp) / (bmed - wmed)),
                       per_coord_rms=float(np.sqrt((disp ** 2).mean() / X.shape[1])))
    out["orthq_window"] = dict(displacement_median=float(np.median(rot)),
                               displacement_over_between_median=float(np.median(rot) / bmed))
    return out, sc


def run(n_seeds):
    timer = C.Timer()
    C.set_vm_limit()
    seeds = range(1, n_seeds + 1)
    results, per_coord = {}, {}

    nt = NTUFi()
    C.check_rss("NTU-Fi loaded")
    X, y = nt.observe("held", "none", None)[0], nt.y["held"]
    rng = np.random.default_rng(NOISE_SEED)
    na = C.dpsq_std(X, nt.std_lo, nt.std_hi, DPSQ_EPS, rng)
    nb = C.dpsq_std(X, nt.std_lo, nt.std_hi, DPSQ_EPS, rng)
    Fq = C.quant_midpoint(nt.F["held"], nt.lo, nt.hi, C.B_HEADLINE)
    rot = nt.S0.transform(C.orth_q_features(Fq, rng, mode="fresh"))
    results["ntufi"], per_coord["ntufi"] = analyse(X, y, na, nb, rot, seeds, "ntufi", np.arange(len(y)))
    del nt
    C.check_rss("NTU-Fi done")

    he = He(cal_per_class=1)
    C.check_rss("He loaded")
    X, y = he.observe("held", "none", None)[0], he.y["held"]
    r2 = np.random.default_rng(NOISE_SEED)
    na = he.observe("held", f"dpsq_e{DPSQ_EPS}", r2)[0]
    nb = he.observe("held", f"dpsq_e{DPSQ_EPS}", r2)[0]
    rot = he.observe("held", "orthq_window", r2)[0]
    sub = np.sort(np.random.default_rng(NOISE_SEED).choice(len(y), size=HE_PAIR_SUBSAMPLE, replace=False))
    # pairwise distances on a subsample (full 18k x 18k is 2.7 GB)
    res_he, sc_he = analyse(X, y, na, nb, rot, seeds, "he", sub)
    bmed = res_he["distances"]["between_median"]
    # paper-code DP-SQ (standardised space) for comparison
    nps = C.dpsq_std(X, he.std_lo, he.std_hi, DPSQ_EPS, r2)
    res_he["dpsq_std_paper_code"] = dict(displacement_median=float(np.median(np.linalg.norm(nps - X, axis=1))),
                                         displacement_over_between_median=float(
                                             np.median(np.linalg.norm(nps - X, axis=1)) / bmed))
    results["he"], per_coord["he"] = res_he, sc_he
    C.check_rss("He done")

    out = dict(results=results, per_coord_acc={k: v.tolist() for k, v in per_coord.items()},
               seeds=n_seeds, runtime_s=timer(), peak_rss_mb=C.peak_rss_mb())
    path = C.FIG_DIR / "bfi_identity_invariant.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(results, indent=1))
    print(f"wrote {path}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        n = HE_PAIR_SUBSAMPLE
        print(f"pairwise matrices: NTU-Fi 420^2, He {n}^2 x 8 B = {n * n * 8 / 1e6:.0f} MB; "
              f"He single-coord pass ~ 18385 x 19 x 118 x 8 B = {18385 * 19 * 118 * 8 / 1e6:.0f} MB")
        return
    run(a.seeds)


if __name__ == "__main__":
    main()
