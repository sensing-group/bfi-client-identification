"""BeamDancer (Cominelli et al., IEEE TWC 2024) on He CR1 real BFI: the plain 1-NN
observer and the beamforming gain the access point keeps, per randomization R.

BeamDancer crafts the client's report so that the access point's beam keeps
changing: Algorithm 1 adds to every quantized angle, on every subcarrier and in
every report, a discrete uniform offset whose support is R % of the angle's
range, reflected at the range edges, and the access point steers with the result.
The paper runs it on a 3x1 link (four angles); here it covers every angle of the
He 4x3 reports, on the He codebook the other defenses use (4-bit phi, 2-bit psi)
and on BeamDancer's own (6, 4).

Observer: as manifold_sweep.py, on the offset reports, enrolled and queried alike,
S0 fitted on the undefended calibration reports of that codebook. Gain: |v_hat_1^H v_1|^2
of he_beamforming_gain's report draws. R = 0 on the He codebook reproduces the
undefended cell of joint_defense_eval and he_beamforming_gain.

Usage: python3 scripts/beamdancer_sweep.py [--seeds 10] [--dry-run]
Outputs (figures/): bfi_beamdancer_sweep_he.csv, bfi_beamdancer_sweep_he_summary.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

import he_beamforming_gain as G
import joint_defense_common as C
import joint_defense_eval as J

R_GRID = (0, 10, 25, 50, 75, 100)
CODEBOOKS = ((C.HE_B_PHI, C.HE_B_PSI), (6, 4))   # (phi bits, psi bits)
N_SEEDS = 10
OBSERVER_STREAM = 500            # random-stream base of the offsets; R is added to it
QUERY_STREAM = 3                 # multi_query_eval's stream, as in manifold_sweep
GAIN_STREAM = 99                 # he_beamforming_gain's per-seed stream
OUT_CSV = C.FIG_DIR / "bfi_beamdancer_sweep_he.csv"
OUT_JSON = C.FIG_DIR / "bfi_beamdancer_sweep_he_summary.json"


def cell_name(bits: tuple[int, int], r_pct: int) -> str:
    return f"b{bits[0]}{bits[1]}_R{r_pct}"


def to_index(x, lo, hi, bits):
    n = 1 << bits
    return np.clip(np.floor((x - lo) / ((hi - lo) / n)).astype(np.int32), 0, n - 1)


def to_value(i, lo, hi, bits):
    return lo + (i + 0.5) * (hi - lo) / (1 << bits)


def offset(i, bits, r_pct, rng):
    """Algorithm 1 on 0-based indices: r in [-h, h - 1], h = R % of half the range, reflected."""
    n = 1 << bits
    half = int(round(r_pct / 100 * n / 2))
    if half == 0:
        return i
    a = i + rng.integers(-half, half, size=i.shape, dtype=np.int32)
    a = np.where(a > n - 1, 2 * n - 1 - a, a)
    a = np.where(a < 0, -a, a)
    return np.clip(a, 0, n - 1)


def report(phi, psi, bits, r_pct, rng):
    bp, bs = bits
    return (to_value(offset(to_index(phi, C.PHI_LO, C.PHI_HI, bp), bp, r_pct, rng), C.PHI_LO, C.PHI_HI, bp),
            to_value(offset(to_index(psi, C.PSI_LO, C.PSI_HI, bs), bs, r_pct, rng), C.PSI_LO, C.PSI_HI, bs))


def sweep(n_seeds: int) -> tuple[list[dict], float]:
    ds = J.He(J.HE_CAL_PER_CLASS)
    y = ds.y["held"]
    chance = 1.0 / len(np.unique(y))
    d = C.load_he_angles()
    rows = []
    for bits in CODEBOOKS:
        S0 = StandardScaler().fit(C.he_features(*report(ds.phi["cal"], ds.psi["cal"], bits, 0, None)))
        for r_pct in R_GRID:
            for seed in range(1, n_seeds + 1):
                stream = [seed, OBSERVER_STREAM + r_pct, bits[0]]
                p, s = report(ds.phi["held"], ds.psi["held"], bits, r_pct, np.random.default_rng(stream))
                accs = C.multi_query_eval(S0.transform(C.he_features(p, s)), y, C.pick_supports(y, seed),
                                          metric="cos", T_grid=J.T_GRID, n_decisions=J.N_DECISIONS_PER_CLASS,
                                          rng=np.random.default_rng(stream + [QUERY_STREAM]))
                del p, s
                rng = np.random.default_rng([seed, GAIN_STREAM])
                idx = np.sort(rng.choice(len(d["y"]), size=G.N_REPORTS, replace=False))
                phi, psi = d["phi"][idx], d["psi"][idx]
                g1 = G.gains(G.decode(*report(phi, psi, bits, r_pct, rng)), C.givens_to_v(phi, psi))[0]
                name = cell_name(bits, r_pct)
                rows += [dict(kind="observer", cell=name, seed=seed, T=T, value=a) for T, a in accs.items()]
                rows.append(dict(kind="gain", cell=name, seed=seed, T="", value=float(g1.mean())))
                gc.collect()
            print(f"  {cell_name(bits, r_pct)} done", flush=True)
            C.check_rss(cell_name(bits, r_pct))
    return rows, chance


def summarise(rows: list[dict], chance: float, n_seeds: int, runtime: float) -> dict:
    cells = {}
    for name in dict.fromkeys(r["cell"] for r in rows):
        c = {}
        for T in J.T_GRID:
            v = [r["value"] for r in rows if r["kind"] == "observer" and r["cell"] == name and r["T"] == T]
            c[f"nn1_T={T}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
        v = [r["value"] for r in rows if r["kind"] == "gain" and r["cell"] == name]
        c["gain_col1"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))
        cells[name] = c
    return dict(dataset="he", chance=chance, seeds=n_seeds, r_pct=list(R_GRID),
                codebooks=[list(b) for b in CODEBOOKS], T_grid=list(J.T_GRID), gain_reports=G.N_REPORTS,
                cells=cells, runtime_s=runtime, peak_rss_mb=C.peak_rss_mb())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"R {R_GRID}, codebooks {CODEBOOKS}; working set ~{C.he_estimate_mb() * 2.5:.0f} MB")
        return
    timer = C.Timer()
    C.set_vm_limit()
    rows, chance = sweep(a.seeds)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = summarise(rows, chance, a.seeds, timer())
    OUT_JSON.write_text(json.dumps(s, indent=1))
    for name, c in s["cells"].items():
        print(f"{name:>10s} nn1 T=1 {c['nn1_T=1']['mean']:.3f} T=32 {c['nn1_T=32']['mean']:.3f} "
              f"gain {c['gain_col1']['mean']:.3f}")
    print(f"wrote {OUT_CSV}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
