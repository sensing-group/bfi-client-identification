"""X1c: per-stream beamforming cost of each defense on He CR1 real BFI, Nc = 2.

Companion of he_beamforming_gain.py, which scores the first stream only
(|v_hat_1^H v_1|^2) and the Nc = 3 subspace. Here the client reports two
streams: the truth is V = first two columns of the He right-singular matrix
(Nr = 4), whose 802.11 Givens angles are the first five phi and five psi of
the Nc = 3 report (the decomposition is column-sequential, checked in
``self_test``). Every defense of the first-stream script is applied in the
same way, with the same per-seed report draw (rng seeded [seed, 99], the
3000-report index is the first draw), and V_hat is what the AP rebuilds.

Metrics per report, averaged over the 59 stride-4 subcarriers:
    gain_s1 = |v_hat_1^H v_1|^2      first stream
    gain_s2 = |v_hat_2^H v_2|^2      second stream
    gain_sum = gain_s1 + gain_s2     (2 for perfect feedback)
    gain_sub = ||V_hat^H V||_F^2 / Nc  subspace overlap (1 for perfect feedback)
Haar-random V_hat: E[gain_s] = 1/Nr = 0.25, E[gain_sum] = 0.5, E[gain_sub] = 0.5.

The 95% clause of the paper (Section 7.1) is checked per stream and for the
sum: gain >= CLAUSE_FRACTION of the unquantized value (1 per stream, 2 for
the sum), and, as a second reading, relative to unprotected quantized feedback.

Usage: python3 scripts/he_beamforming_gain_streams.py [--reports 3000] [--seeds 10]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

import joint_defense_common as C

N_REPORTS = 3000
N_SEEDS = 10
NC = 2                                   # streams reported
N_ANGLES = 5                             # phi and psi each for Nr=4, Nc=2
DPSQ_EPS = (0.01, 0.5, 2.0)              # the published DP-SQ budgets
MANIFOLD_EPS = 0.5
CODEBOOK_FINE = (6, 4)                   # 802.11ac/ax SU (b_phi, b_psi), reference only
CLAUSE_FRACTION = 0.95
CLAUSE_DEFENSES = ("none|none", "unitary_pkt|AP holds Q",
                   "unitary_pkt_sc|AP holds Q", "dpsq_e0.01|none",
                   "dpsq_e0.5|none", "dpsq_e2.0|none")
UNITARY_NAMES = ("unitary_pkt", "unitary_pkt_sc", "unitary_fixed", "unitary_global")
ORTHQ_NAMES = (("orthq_window", None), ("orthq_fixed", "y"), ("orthq_global", "zeros"))
SELF_TEST_N = 200
OUT_CSV = C.FIG_DIR / "bfi_he_beamforming_gain_streams.csv"
OUT_JSON = C.FIG_DIR / "bfi_he_beamforming_gain_streams.json"


def to_v(phi, psi):
    return C.givens_to_v(phi, psi, nr=C.HE_NR, nc=NC)


def from_v(V):
    return C.v_to_givens(V, nr=C.HE_NR, nc=NC)


def decode(phi, psi):
    """What an AP does with whatever angle values arrive: wrap phi, clip psi, rebuild V."""
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    psi = np.clip(psi, C.PSI_LO, C.PSI_HI)
    return to_v(phi, psi)


def gains(V_hat, V):
    """Per-report (stream 1, stream 2, sum, subspace) averaged over subcarriers."""
    per = np.abs(np.einsum("nkrc,nkrc->nkc", np.conj(V_hat), V)) ** 2      # (n, k, NC)
    sub = (np.abs(np.conj(np.swapaxes(V_hat, -1, -2)) @ V) ** 2).sum(axis=(-1, -2)) / NC
    s = per.mean(axis=1)
    return s[:, 0], s[:, 1], s.sum(axis=1), sub.mean(axis=1)


def self_test(phi, psi):
    """First NC columns of the Nc=3 V equal the Nc=2 V of the first N_ANGLES
    angles, and the Nc=2 decomposition inverts givens_to_v."""
    p, s = phi[:SELF_TEST_N], psi[:SELF_TEST_N]
    V3 = C.givens_to_v(p, s)[..., :NC]
    V2 = to_v(p[..., :N_ANGLES], s[..., :N_ANGLES])
    p2, s2 = from_v(V2)
    dphi = np.abs(np.mod(p2 - p[..., :N_ANGLES] + np.pi, 2 * np.pi) - np.pi)
    return dict(columns_equal=bool(np.allclose(V3, V2)),
                max_abs_v_diff=float(np.abs(V3 - V2).max()),
                round_trip_max_phi_err=float(dphi.max()),
                round_trip_max_psi_err=float(np.abs(s2 - s[..., :N_ANGLES]).max()))


def run(n_reports, n_seeds):
    timer = C.Timer()
    C.set_vm_limit()
    d = C.load_he_angles()
    C.check_rss("He loaded")
    phi_all = d["phi"][..., :N_ANGLES].copy()
    psi_all = d["psi"][..., :N_ANGLES].copy()
    y_all = d["y"]
    check = self_test(d["phi"], d["psi"])
    del d
    layout = dict(nr=C.HE_NR, nc=NC, n_angles_each=N_ANGLES,
                  subcarriers_used=int(phi_all.shape[1]), sc_stride=C.HE_SC_STRIDE,
                  givens_phi_order="phi11 phi21 phi31 phi22 phi32",
                  givens_psi_order="psi21 psi31 psi41 psi32 psi42",
                  self_test=check)
    rows = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng([seed, 99])
        idx = np.sort(rng.choice(len(y_all), size=n_reports, replace=False))
        phi, psi, y = phi_all[idx], psi_all[idx], y_all[idx]
        V = to_v(phi, psi)                                           # truth (n, 59, 4, 2)
        res = {}

        def add(name, key, Vh):
            res[(name, key)] = gains(Vh, V)

        add("continuous", "none", V)
        pq, sq = C.he_quantise(phi, psi)
        add("none", "none", to_v(pq, sq))
        add("none_codebook_6_4", "none",
            to_v(C.quant_midpoint(phi, C.PHI_LO, C.PHI_HI, CODEBOOK_FINE[0]),
                 C.quant_midpoint(psi, C.PSI_LO, C.PSI_HI, CODEBOOK_FINE[1])))
        for e in DPSQ_EPS:
            a, b = C.he_quantise(phi, psi, eps=e, rng=rng)
            add(f"dpsq_e{e}", "none", to_v(a, b))
        # AP-side unitary precoding of the sounding: client reports Q^H V (phase-normalised, quantised)
        for name in UNITARY_NAMES:
            n, k = V.shape[:2]
            if name == "unitary_pkt":
                Q = C.haar_unitary(C.HE_NR, (n,), rng)[:, None]
            elif name == "unitary_pkt_sc":
                Q = C.haar_unitary(C.HE_NR, (n, k), rng)
            else:
                groups = y if name == "unitary_fixed" else np.zeros_like(y)
                per = {int(g): C.haar_unitary(C.HE_NR, (), rng) for g in np.unique(groups)}
                Q = np.stack([per[int(g)] for g in groups])[:, None]
            p2, s2 = from_v(np.conj(np.swapaxes(Q, -1, -2)) @ V)
            a, b = C.he_quantise(p2, s2)
            Vq = to_v(a, b)
            add(name, "AP without Q", Vq)
            add(name, "AP holds Q", np.broadcast_to(Q, Vq.shape[:2] + Q.shape[-2:]) @ Vq)
            del Q
        # Feature-vector orthogonal Q on all transmitted angles (59 x 10 per report)
        x = np.concatenate([pq, sq], axis=-1).reshape(n_reports, -1)
        for name, grp in ORTHQ_NAMES:
            groups = None if grp is None else (y if grp == "y" else np.zeros_like(y))
            mode = "fresh" if groups is None else "fixed"
            xo = C.orth_q_features(x, rng, mode=mode, groups=groups).reshape(
                pq.shape[0], pq.shape[1], -1)
            add(name, "AP without Q", decode(xo[..., :N_ANGLES], xo[..., N_ANGLES:]))
            add(name, "AP holds Q (float feedback)", to_v(pq, sq))
        # Manifold DP as implemented (Gaussian, sensitivity = full range, then b=4 grid over [-pi, pi])
        for_m = np.concatenate([phi, psi], axis=-1).reshape(n_reports, -1)
        mo = C.manifold_dp(for_m, C.PHI_LO, C.PHI_HI, MANIFOLD_EPS, rng).reshape(
            phi.shape[0], phi.shape[1], -1)
        add(f"manifold_e{MANIFOLD_EPS}", "none", decode(mo[..., :N_ANGLES], mo[..., N_ANGLES:]))
        for (name, key), (g1, g2, gsum, gsub) in res.items():
            rows.append(dict(seed=seed, defense=name, ap_secret=key,
                             gain_s1_mean=float(g1.mean()), gain_s1_p05=float(np.percentile(g1, 5)),
                             gain_s2_mean=float(g2.mean()), gain_s2_p05=float(np.percentile(g2, 5)),
                             gain_sum_mean=float(gsum.mean()), gain_sum_p05=float(np.percentile(gsum, 5)),
                             gain_sub_mean=float(gsub.mean()), gain_sub_p05=float(np.percentile(gsub, 5)),
                             n_reports=n_reports))
        C.check_rss(f"seed {seed}")
        del V, res
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for r in rows:
        summ.setdefault(f"{r['defense']}|{r['ap_secret']}", []).append(r)
    cells = {}
    for k, v in summ.items():
        cell = {}
        for m in ("s1", "s2", "sum", "sub"):
            means = [r[f"gain_{m}_mean"] for r in v]
            cell[f"gain_{m}"] = float(np.mean(means))
            cell[f"gain_{m}_sd_over_seeds"] = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
            cell[f"gain_{m}_p05"] = float(np.mean([r[f"gain_{m}_p05"] for r in v]))
        cells[k] = cell
    ref = cells["none|none"]
    clause = {}
    for k in CLAUSE_DEFENSES:
        c = cells[k]
        clause[k] = dict(
            s1_vs_unquantized=bool(c["gain_s1"] >= CLAUSE_FRACTION),
            s2_vs_unquantized=bool(c["gain_s2"] >= CLAUSE_FRACTION),
            sum_vs_unquantized=bool(c["gain_sum"] >= CLAUSE_FRACTION * NC),
            sub_vs_unquantized=bool(c["gain_sub"] >= CLAUSE_FRACTION),
            s1_over_quantized=float(c["gain_s1"] / ref["gain_s1"]),
            s2_over_quantized=float(c["gain_s2"] / ref["gain_s2"]),
            sum_over_quantized=float(c["gain_sum"] / ref["gain_sum"]),
            sub_over_quantized=float(c["gain_sub"] / ref["gain_sub"]))
    out = dict(layout=layout, cells=cells, clause_fraction=CLAUSE_FRACTION, clause=clause,
               n_reports=n_reports, seeds=n_seeds,
               random_reference=dict(gain_s1=1.0 / C.HE_NR, gain_s2=1.0 / C.HE_NR,
                                     gain_sum=NC / C.HE_NR, gain_sub=NC / C.HE_NR),
               runtime_s=timer(), peak_rss_mb=C.peak_rss_mb())
    OUT_JSON.write_text(json.dumps(out, indent=1))
    for k, v in cells.items():
        print(f"{k:45s} s1={v['gain_s1']:.4f} s2={v['gain_s2']:.4f} "
              f"sum={v['gain_sum']:.4f} sub={v['gain_sub']:.4f}")
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=int, default=N_REPORTS)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"He tensor ~{C.he_estimate_mb():.0f} MB on disk; working set ~{C.he_estimate_mb() * 1.5:.0f} MB")
        return
    run(a.reports, a.seeds)


if __name__ == "__main__":
    main()
