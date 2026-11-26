"""X1b: beamforming cost of each defense on He CR1 real BFI.

The He angles are continuous; we take V built from them as the true
right-singular matrix (Nr=4, Nc=3, 802.11 Givens order) and compare it with
the V-hat the AP reconstructs from the defended feedback.

Metrics, per report, averaged over the 59 stride-4 subcarriers:
    gain_col1  = |v_hat_1^H v_1|^2                  (first stream, SU beamforming vector)
    gain_sub   = ||V_hat^H V||_F^2 / Nc              (3-stream subspace capture)
Both are 1 for perfect feedback. For a Haar-random V_hat the expectations are
1/Nr = 0.25 (gain_col1) and Nc/Nr = 0.75 (gain_sub).

Each keyed defense is scored twice: AP without the secret, AP holding it.

Usage: python3 scripts/he_beamforming_gain.py [--reports 3000] [--seeds 10]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

import joint_defense_common as C

N_REPORTS = 3000
N_SEEDS = 10
DPSQ_EPS = (0.01, 0.5, 2.0)
MANIFOLD_EPS = 0.5
CODEBOOK_FINE = (6, 4)          # 802.11ac/ax SU (b_phi, b_psi), reference only


def gains(V_hat, V):
    col1 = np.abs(np.einsum("nkr,nkr->nk", np.conj(V_hat[..., 0]), V[..., 0])) ** 2
    sub = (np.abs(np.conj(np.swapaxes(V_hat, -1, -2)) @ V) ** 2).sum(axis=(-1, -2)) / V.shape[-1]
    return col1.mean(axis=1), sub.mean(axis=1)


def decode(phi, psi):
    """What an AP does with whatever angle values arrive: wrap phi, clip psi, rebuild V."""
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    psi = np.clip(psi, C.PSI_LO, C.PSI_HI)
    return C.givens_to_v(phi, psi)


def run(n_reports, n_seeds):
    timer = C.Timer()
    C.set_vm_limit()
    d = C.load_he_angles()
    C.check_rss("He loaded")
    phi_all, psi_all, y_all = d["phi"], d["psi"], d["y"]
    n_unique = int(np.unique(phi_all[:1000, :, 0]).size)
    layout = dict(raw_shape=list(d["raw_shape"]), nr=C.HE_NR, nc=C.HE_NC, n_pairs=C.HE_N_PAIRS,
                  subcarriers_used=int(phi_all.shape[1]), sc_stride=C.HE_SC_STRIDE,
                  phi_range=[float(phi_all.min()), float(phi_all.max())],
                  psi_range=[float(psi_all.min()), float(psi_all.max())],
                  unique_phi_values_first_1000x59=n_unique,
                  givens_phi_order="phi11 phi21 phi31 phi22 phi32 phi33",
                  givens_psi_order="psi21 psi31 psi41 psi32 psi42 psi43",
                  assumption="pair files 01,02,03,12,13,23 = Givens row pairs of G_{l,i}; pair p = (phi[p], psi[p])")
    rows = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng([seed, 99])
        idx = np.sort(rng.choice(len(y_all), size=n_reports, replace=False))
        phi, psi, y = phi_all[idx], psi_all[idx], y_all[idx]
        V = C.givens_to_v(phi, psi)                                  # truth (n, 59, 4, 3)
        res = {}

        def add(name, key, Vh):
            g1, gs = gains(Vh, V)
            res[(name, key)] = (g1, gs)

        add("continuous", "none", V)
        pq, sq = C.he_quantise(phi, psi)
        add("none", "none", C.givens_to_v(pq, sq))
        add("none_codebook_6_4", "none",
            C.givens_to_v(C.quant_midpoint(phi, C.PHI_LO, C.PHI_HI, CODEBOOK_FINE[0]),
                          C.quant_midpoint(psi, C.PSI_LO, C.PSI_HI, CODEBOOK_FINE[1])))
        for e in DPSQ_EPS:
            a, b = C.he_quantise(phi, psi, eps=e, rng=rng)
            add(f"dpsq_e{e}", "none", C.givens_to_v(a, b))
        # AP-side unitary precoding of the sounding: client reports V' = Q^H V (phase-normalised, quantised)
        for name in ("unitary_pkt", "unitary_pkt_sc", "unitary_fixed", "unitary_global"):
            n, k = V.shape[:2]
            if name == "unitary_pkt":
                Q = C.haar_unitary(C.HE_NR, (n,), rng)[:, None]
            elif name == "unitary_pkt_sc":
                Q = C.haar_unitary(C.HE_NR, (n, k), rng)
            else:
                groups = y if name == "unitary_fixed" else np.zeros_like(y)
                per = {int(g): C.haar_unitary(C.HE_NR, (), rng) for g in np.unique(groups)}
                Q = np.stack([per[int(g)] for g in groups])[:, None]
            p2, s2 = C.v_to_givens(np.conj(np.swapaxes(Q, -1, -2)) @ V)
            a, b = C.he_quantise(p2, s2)
            Vq = C.givens_to_v(a, b)
            add(name, "AP without Q", Vq)
            add(name, "AP holds Q", np.broadcast_to(Q, Vq.shape[:2] + Q.shape[-2:]) @ Vq)
            del Q
        # Feature-vector orthogonal Q on all transmitted angles (59 x 12 per report)
        x = np.concatenate([pq, sq], axis=-1).reshape(n_reports, -1)
        for name, groups in (("orthq_window", None), ("orthq_fixed", y), ("orthq_global", np.zeros_like(y))):
            mode = "fresh" if groups is None else "fixed"
            xo = C.orth_q_features(x, rng, mode=mode, groups=groups).reshape(pq.shape[0], pq.shape[1], -1)
            add(name, "AP without Q", decode(xo[..., :6], xo[..., 6:]))
            add(name, "AP holds Q (float feedback)", C.givens_to_v(pq, sq))
        # Manifold DP as implemented (Gaussian, sensitivity = full range, then b=4 grid over [-pi, pi])
        for_m = np.concatenate([phi, psi], axis=-1).reshape(n_reports, -1)
        mo = C.manifold_dp(for_m, C.PHI_LO, C.PHI_HI, MANIFOLD_EPS, rng).reshape(phi.shape[0], phi.shape[1], -1)
        add("manifold_e0.5", "none", decode(mo[..., :6], mo[..., 6:]))
        for (name, key), (g1, gs) in res.items():
            rows.append(dict(seed=seed, defense=name, ap_secret=key,
                             gain_col1_mean=float(g1.mean()), gain_col1_p05=float(np.percentile(g1, 5)),
                             gain_sub_mean=float(gs.mean()), gain_sub_p05=float(np.percentile(gs, 5)),
                             n_reports=n_reports))
        C.check_rss(f"seed {seed}")
        del V
    out_csv = C.FIG_DIR / "bfi_he_beamforming_gain.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for r in rows:
        summ.setdefault(f"{r['defense']}|{r['ap_secret']}", []).append(r)
    cells = {k: dict(gain_col1=float(np.mean([r["gain_col1_mean"] for r in v])),
                     gain_col1_sd_over_seeds=float(np.std([r["gain_col1_mean"] for r in v], ddof=1)),
                     gain_col1_p05=float(np.mean([r["gain_col1_p05"] for r in v])),
                     gain_sub=float(np.mean([r["gain_sub_mean"] for r in v])),
                     gain_sub_p05=float(np.mean([r["gain_sub_p05"] for r in v])))
             for k, v in summ.items()}
    out = dict(layout=layout, cells=cells, n_reports=n_reports, seeds=n_seeds,
               random_reference=dict(gain_col1=1.0 / C.HE_NR, gain_sub=C.HE_NC / C.HE_NR),
               runtime_s=timer(), peak_rss_mb=C.peak_rss_mb())
    (C.FIG_DIR / "bfi_he_beamforming_gain_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in cells.items():
        print(f"{k:45s} col1={v['gain_col1']:.4f} sub={v['gain_sub']:.4f} p05={v['gain_col1_p05']:.3f}")
    print(f"wrote {out_csv}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


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
