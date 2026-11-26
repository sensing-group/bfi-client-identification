"""What an access point keeps from noisy reports if it decodes them with a prior, on He CR1.

The gain of he_beamforming_gain and manifold_sweep decodes each report as it
arrives. A decoder may also know where beams usually point. With v the first
right-singular vector and Sigma_k = E[v v^H] on subcarrier k, a report independent
of v gives any decoder at most lambda_max(Sigma_k) (Section 7.3), the gain of the
best fixed beam. This script measures, on the 19 held-out positions:

  bound       mean over subcarriers of lambda_max(Sigma_k), Sigma_k from those positions
  fixed       the top eigenvector of Sigma_k learned on the 20 calibration positions
  prior       per noise level, the top eigenvector of Sigma_k + w u u^H, u the decoded
              report; w is the grid value with the best mean over seeds
  plain       the decoded report u alone

Sigma_k comes from clean calibration reports and w is picked on the test reports,
so `prior` favors the access point. Observer accuracy at each level is that of
manifold_sweep.py and beamdancer_sweep.py; a decoder does not change what is on air.

Usage: python3 scripts/prior_decoder.py [--seeds 10] [--dry-run]
Outputs (figures/): bfi_prior_decoder_he.csv, bfi_prior_decoder_he_summary.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np

import beamdancer_sweep as B
import he_beamforming_gain as G
import joint_defense_common as C

MANIFOLD_EPS = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
BEAMDANCER_R = (10, 25, 50, 75, 100)
PRIOR_WEIGHTS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_SEEDS = 10
SAMPLE_STREAM = 77               # per-seed stream of the held-out report draw
NOISE_STREAM = 78                # base of the per-level noise streams
SIGMA_CHUNK = 2000
OUT_CSV = C.FIG_DIR / "bfi_prior_decoder_he.csv"
OUT_JSON = C.FIG_DIR / "bfi_prior_decoder_he_summary.json"


def first_stream(phi, psi) -> np.ndarray:
    return C.givens_to_v(phi, psi)[..., 0]                     # (n, K, N_t)


def covariance(phi, psi) -> np.ndarray:
    acc = 0.0
    for s in range(0, phi.shape[0], SIGMA_CHUNK):
        v = first_stream(phi[s:s + SIGMA_CHUNK], psi[s:s + SIGMA_CHUNK])
        acc = acc + np.einsum("nkr,nkc->krc", v, np.conj(v))
    return acc / phi.shape[0]


def gain(v_hat, v) -> float:
    return float((np.abs(np.einsum("...r,...r->...", np.conj(v_hat), v)) ** 2).mean())


def prior_gains(sigma, u, v) -> dict[float, float]:
    out = {}
    for w in PRIOR_WEIGHTS:
        a = sigma[None] + w * np.einsum("nkr,nkc->nkrc", u, np.conj(u))
        out[w] = gain(np.linalg.eigh(a)[1][..., -1], v)
        del a
    return out


def levels(phi, psi, seed):
    """(cell name, decoded first stream) per noise level."""
    flat = np.concatenate([phi, psi], axis=-1).reshape(phi.shape[0], -1)
    for i, e in enumerate(MANIFOLD_EPS):
        mo = C.manifold_dp(flat, C.PHI_LO, C.PHI_HI, e, np.random.default_rng([seed, NOISE_STREAM, i]))
        mo = mo.reshape(phi.shape[0], phi.shape[1], -1)
        yield f"manifold_e{e}", G.decode(mo[..., :6], mo[..., 6:])[..., 0]
    bits = (C.HE_B_PHI, C.HE_B_PSI)
    for r_pct in BEAMDANCER_R:
        p, s = B.report(phi, psi, bits, r_pct, np.random.default_rng([seed, NOISE_STREAM, 100 + r_pct]))
        yield B.cell_name(bits, r_pct), G.decode(p, s)[..., 0]


def run(n_seeds: int) -> tuple[list[dict], dict]:
    d = C.load_he_angles()
    phi, psi, y = d["phi"], d["psi"], d["y"]
    cal = np.isin(y, np.unique(y)[: C.HE_N_CALIB_CLASSES])
    held = np.flatnonzero(~cal)
    sigma = covariance(phi[cal], psi[cal])
    fixed = np.linalg.eigh(sigma)[1][..., -1]                  # (K, N_t)
    lam_held = np.linalg.eigvalsh(covariance(phi[held], psi[held]))[:, -1]
    per_position = [float(np.linalg.eigvalsh(covariance(phi[y == c], psi[y == c]))[:, -1].mean())
                    for c in np.unique(y[held])]
    rows = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng([seed, SAMPLE_STREAM])
        idx = np.sort(rng.choice(held, size=G.N_REPORTS, replace=False))
        p, s = phi[idx], psi[idx]
        v = first_stream(p, s)
        rows.append(dict(seed=seed, cell="fixed", decoder="fixed", w="", value=gain(fixed[None], v)))
        for name, u in levels(p, s, seed):
            rows.append(dict(seed=seed, cell=name, decoder="plain", w="", value=gain(u, v)))
            rows += [dict(seed=seed, cell=name, decoder="prior", w=w, value=g)
                     for w, g in prior_gains(sigma, u, v).items()]
            del u
            gc.collect()
        print(f"  seed {seed} done", flush=True)
        C.check_rss(f"seed {seed}")
    meta = dict(bound_held=float(lam_held.mean()), bound_held_sc_min=float(lam_held.min()),
                bound_held_sc_max=float(lam_held.max()), bound_per_position_mean=float(np.mean(per_position)),
                bound_per_position_min=float(np.min(per_position)),
                n_calibration_positions=C.HE_N_CALIB_CLASSES, n_held_positions=len(per_position))
    return rows, meta


def summarise(rows: list[dict], meta: dict, n_seeds: int, runtime: float) -> dict:
    def stat(v):
        return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))

    cells = {"fixed": dict(gain=stat([r["value"] for r in rows if r["cell"] == "fixed"]))}
    for name in dict.fromkeys(r["cell"] for r in rows if r["decoder"] == "plain"):
        by_w = {w: [r["value"] for r in rows if r["cell"] == name and r["decoder"] == "prior" and r["w"] == w]
                for w in PRIOR_WEIGHTS}
        best = max(by_w, key=lambda w: np.mean(by_w[w]))
        cells[name] = dict(plain=stat([r["value"] for r in rows if r["cell"] == name and r["decoder"] == "plain"]),
                           prior=stat(by_w[best]), prior_w=best)
    return dict(dataset="he", seeds=n_seeds, gain_reports=G.N_REPORTS, weights=list(PRIOR_WEIGHTS), **meta,
                cells=cells, runtime_s=runtime, peak_rss_mb=C.peak_rss_mb())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"{len(MANIFOLD_EPS) + len(BEAMDANCER_R)} levels, {len(PRIOR_WEIGHTS)} weights; "
              f"working set ~{C.he_estimate_mb() * 2.5:.0f} MB")
        return
    timer = C.Timer()
    C.set_vm_limit()
    rows, meta = run(a.seeds)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = summarise(rows, meta, a.seeds, timer())
    OUT_JSON.write_text(json.dumps(s, indent=1))
    print({k: round(v, 3) for k, v in meta.items() if k.startswith("bound")})
    for name, c in s["cells"].items():
        print(name, {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in c.items()})
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
