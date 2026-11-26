"""Manifold DP across privacy budgets on He CR1 real BFI: the plain 1-NN
observer and the beamforming gain the access point keeps, per budget.

The joint-defense run (joint_defense_eval.py) and the gain run
(he_beamforming_gain.py) score Manifold DP at one budget (eps = 0.5), where it
reaches chance and loses the beam. This sweep asks whether any budget hides the
client (observer at chance, 5.3 %) while the access point keeps its gain.

Observer: joint_defense_eval.He, held-out positions, one support window per
class, cosine 1-NN on S0-standardised features, T query windows averaged; the
random streams of eps = 0.5 and of the undefended reference are those of
joint_defense_eval, so both reproduce its cells. Gain: he_beamforming_gain's
Manifold DP path (same report draws per seed), |v_hat_1^H v_1|^2 averaged over
subcarriers.

Usage: python3 scripts/manifold_sweep.py [--seeds 10] [--dry-run]
Outputs (figures/): bfi_manifold_sweep_he.csv, bfi_manifold_sweep_he_summary.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np

import he_beamforming_gain as G
import joint_defense_common as C
import joint_defense_eval as J

SWEEP_EPS = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0)
REFERENCE = "none"
N_SEEDS = 10
SWEEP_STREAM_OFFSET = 100        # random-stream index of the budgets not in joint_defense_eval
GAIN_STREAM = 99                 # he_beamforming_gain's per-seed stream
NN1_ATTACKER_ID = 0              # joint_defense_eval scores nn1 first
OUT_CSV = C.FIG_DIR / "bfi_manifold_sweep_he.csv"
OUT_JSON = C.FIG_DIR / "bfi_manifold_sweep_he_summary.json"


def _defense(eps: float) -> str:
    return f"manifold_e{eps}"


def _stream(defense: str, i: int) -> int:
    return J.HE_DEFENSES.index(defense) if defense in J.HE_DEFENSES else SWEEP_STREAM_OFFSET + i


def observer_rows(n_seeds: int) -> list[dict]:
    ds = J.He(J.HE_CAL_PER_CLASS)
    y = ds.y["held"]
    chance = 1.0 / len(np.unique(y))
    defenses = [REFERENCE] + [_defense(e) for e in SWEEP_EPS]
    rows = []
    for seed in range(1, n_seeds + 1):
        supports = C.pick_supports(y, seed)
        for i, d in enumerate(defenses):
            did = _stream(d, i)
            Xh, _, _ = ds.observe("held", d, np.random.default_rng([seed, did, 1]))
            accs = C.multi_query_eval(Xh, y, supports, metric="cos", T_grid=J.T_GRID,
                                      n_decisions=J.N_DECISIONS_PER_CLASS,
                                      rng=np.random.default_rng([seed, did, NN1_ATTACKER_ID, 3]))
            rows += [dict(kind="observer", defense=d, seed=seed, T=T, value=a, chance=chance)
                     for T, a in accs.items()]
            print(f"  seed {seed} {d:>16s} T=1 {accs[1]:.3f}", flush=True)
            del Xh
            gc.collect()
        C.check_rss(f"observer seed {seed}")
    del ds
    gc.collect()
    return rows


def gain_rows(n_seeds: int) -> list[dict]:
    d = C.load_he_angles()
    phi_all, psi_all, y_all = d["phi"], d["psi"], d["y"]
    rows = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng([seed, GAIN_STREAM])
        idx = np.sort(rng.choice(len(y_all), size=G.N_REPORTS, replace=False))
        phi, psi = phi_all[idx], psi_all[idx]
        V = C.givens_to_v(phi, psi)
        pq, sq = C.he_quantise(phi, psi)
        rows.append(dict(kind="gain", defense=REFERENCE, seed=seed, T="",
                         value=float(G.gains(C.givens_to_v(pq, sq), V)[0].mean()), chance=""))
        flat = np.concatenate([phi, psi], axis=-1).reshape(G.N_REPORTS, -1)
        for i, e in enumerate(SWEEP_EPS):
            mo = C.manifold_dp(flat, C.PHI_LO, C.PHI_HI, e, np.random.default_rng([seed, GAIN_STREAM, i]))
            mo = mo.reshape(phi.shape[0], phi.shape[1], -1)
            g1 = G.gains(G.decode(mo[..., :6], mo[..., 6:]), V)[0]
            rows.append(dict(kind="gain", defense=_defense(e), seed=seed, T="", value=float(g1.mean()),
                             chance=""))
        C.check_rss(f"gain seed {seed}")
        del V
    return rows


def summarise(rows: list[dict], n_seeds: int, runtime: float) -> dict:
    cells = {}
    for d in [REFERENCE] + [_defense(e) for e in SWEEP_EPS]:
        c = dict(sigma_over_range=None if d == REFERENCE else C.manifold_sigma_over_range(J._eps_of(d)))
        for T in J.T_GRID:
            v = [r["value"] for r in rows if r["kind"] == "observer" and r["defense"] == d and r["T"] == T]
            c[f"nn1_T={T}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
        v = [r["value"] for r in rows if r["kind"] == "gain" and r["defense"] == d]
        c["gain_col1"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
        cells[d] = c
    chance = next(r["chance"] for r in rows if r["kind"] == "observer")
    return dict(dataset="he", chance=chance, seeds=n_seeds, eps=list(SWEEP_EPS),
                delta=C.MANIFOLD_DELTA, T_grid=list(J.T_GRID), gain_reports=G.N_REPORTS,
                cells=cells, runtime_s=runtime, peak_rss_mb=C.peak_rss_mb())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"budgets {SWEEP_EPS}; working set ~{C.he_estimate_mb() * 2.5:.0f} MB")
        return
    timer = C.Timer()
    C.set_vm_limit()
    rows = observer_rows(a.seeds) + gain_rows(a.seeds)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = summarise(rows, a.seeds, timer())
    OUT_JSON.write_text(json.dumps(s, indent=1))
    for d, c in s["cells"].items():
        print(f"{d:>16s} nn1 T=1 {c['nn1_T=1']['mean']:.3f} T=32 {c['nn1_T=32']['mean']:.3f} "
              f"gain {c['gain_col1']['mean']:.3f}")
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
