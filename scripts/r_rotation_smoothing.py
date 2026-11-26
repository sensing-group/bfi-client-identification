#!/usr/bin/env python3
"""R-smoothing: beamforming gain of the access point's sounding rotation when
the client smooths its channel estimate across tone groups.

The rotation (r_rotation_space.py, structure s4_unitary) sends the sounding
of tone group k through a Haar unitary Q_k; the client reports Q_k^H V_k and
the access point undoes Q_k. he_beamforming_gain.py and r_rotation_space.py
score that with a client that estimates every group on its own. A client
that averages its estimate over neighbouring groups averages across
different Q_k, and this script measures what that costs.

Model
  The He archive holds the continuous right-singular vectors V_k, not the
  channel H_k. We use the rank-one model in which the client's effective
  first-stream channel on group k is the unit vector g_k = Q_k^H v_k
  (v_k the first column of V_k; g_k = v_k without rotation).
    1. gauge   g_k is multiplied by a unit phase so that its last row is
               real and non-negative, the phase the standard's feedback
               removes (joint_defense_common.v_to_givens). Done BEFORE
               smoothing, so the arbitrary per-group phase of an SVD does
               not penalise smoothing of the unrotated channel.
    2. smooth  s_k = sum_j h_j g_{k+j} with a centred kernel h: w1 (none),
               tap3 [0.25, 0.5, 0.25], box<w> (boxcar, w in 2, 3, 4, 8; an
               even w covers offsets -w/2 .. w/2-1). Groups beyond the edge
               of the report are left out of the sum.
    3. report  r_k = s_k / |s_k|, continuous or decomposed into the
               first-column Givens angles and quantised at a codebook.
    4. steer   the access point uses v_hat_k = Q_k r_k;
               gain = |v_hat_k^H v_k|^2 (he_beamforming_gain.gains),
               averaged over groups and reports.

Limits of the model
  * A real client smooths H_k (or its per-antenna estimate) before the SVD,
    with a singular-value weight per group and receiver noise that smoothing
    reduces. The archive has neither, so the cost of smoothing is counted
    and its noise benefit is not; the unrotated rows are therefore an upper
    bound on what smoothing loses on a real unrotated channel only in the
    sense of this unit-vector model.
  * The gauge is the feedback's own (last row real). A client that smooths
    a channel estimate has the receive-side phase instead; the two agree
    only where the last-row gauge is itself smooth across groups.
  * First stream only; the other columns of V_k are not smoothed or scored.
  * Groups are the loader's stride-4 tones (59 per report), so a window of w
    groups spans 4 w tones of the 234 of the archive.

Cells = rotation x kernel x codebook.
  Rotation   none, report (one Q for all groups), group (fresh Q per group),
             band2 / band4 / band8 (one Q per band of M consecutive groups)
  Codebook   continuous; cb_4_2 (joint_defense_common.he_quantise, the
             paper's); cb_6_4 (he_beamforming_gain.CODEBOOK_FINE)
Reports are drawn exactly as he_beamforming_gain.py draws them (same seeds,
same indices); the Q of a (seed, rotation) pair is shared by all kernels.

The band rows are also written next to the observer-facing leak of the same
band size, read from figures/r_rotation_space.json (inv_cross accuracy on
He at T = 1 and 32); nothing of that file is recomputed here.

Memory: the He angle tensor (loader, as in he_beamforming_gain.py) plus one
chunk of CHUNK reports of 4 x 4 unitaries per group at a time.
Outputs (figures/): r_rotation_smoothing.json, r_rotation_smoothing.csv.

Usage: python3 scripts/r_rotation_smoothing.py [--reports 3000] [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_defense_common as C  # noqa: E402
import he_beamforming_gain as HG  # noqa: E402
import r_eval as E  # noqa: E402
import r_rotation_space as RS  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_REPORTS = HG.N_REPORTS
N_SEEDS = HG.N_SEEDS
DRAW_STREAM = 99                 # he_beamforming_gain's report-draw stream
ROT_STREAM = 7                   # Haar draws; one stream per (seed, rotation)
STRUCTURE = "s4_unitary"
# rotation -> freshness key of r_rotation_space (None = no rotation)
ROTATIONS = {"none": None, "report": "f1_report", "group": "f2_group",
             "band2": "f3_band2", "band4": "f3_band4", "band8": "f3_band8"}
BAND_ROTATIONS = ("band2", "band4", "band8")
BOX_WIDTHS = (2, 3, 4, 8)
KERNELS = {"w1": (1.0,), "tap3": (0.25, 0.5, 0.25),
           **{f"box{w}": (1.0 / w,) * w for w in BOX_WIDTHS}}
# codebook -> (b_phi, b_psi); None = the continuous reported vector
CODEBOOKS = {"continuous": None, "cb_4_2": (C.HE_B_PHI, C.HE_B_PSI),
             "cb_6_4": HG.CODEBOOK_FINE}
PAPER_CODEBOOK = "cb_4_2"
N_STREAMS = 1                    # first stream only
CHUNK = 1000                     # reports per rotation chunk
P_LOW = 5                        # percentile of the per-report gain
GAIN_MIN = RS.GAIN_MIN           # the paper's threshold
COMPLEX_BYTES = 16
QR_COPIES = 4                    # draw, Q, R and the phase-fixed product
SPACE_JSON = RS.OUT["json"]
SPACE_OBSERVER = "inv_cross"
SPACE_T = RS.T_GRID
GAIN_JSON = C.FIG_DIR / "bfi_he_beamforming_gain_summary.json"
# this script's cell -> the cell of he_beamforming_gain's summary it repeats
REFERENCE_CELLS = {("none", "w1", "continuous"): "continuous|none",
                   ("none", "w1", "cb_4_2"): "none|none",
                   ("none", "w1", "cb_6_4"): "none_codebook_6_4|none",
                   ("report", "w1", "cb_4_2"): "unitary_pkt|AP holds Q",
                   ("group", "w1", "cb_4_2"): "unitary_pkt_sc|AP holds Q"}
OUT = dict(json=C.FIG_DIR / "r_rotation_smoothing.json",
           csv=C.FIG_DIR / "r_rotation_smoothing.csv")


def cell_name(rot: str, kern: str, cb: str) -> str:
    return f"{rot}|{kern}|{cb}"


# ---------------------------------------------------------------------------
# Client model
# ---------------------------------------------------------------------------
def draw_rotation(rot: str, n: int, k: int, rng):
    """Q (n, k or 1, nr, nr) of the rotation, None for 'none'."""
    f = ROTATIONS[rot]
    if f is None:
        return None
    m = RS.band_m(f, k)
    nb = math.ceil(k / m)
    Q = RS.draw_q(STRUCTURE, C.HE_NR, (n, nb), rng)
    if nb > 1:
        Q = np.repeat(Q, m, axis=1)[:, :k]        # one Q per band of m groups
    return Q


def gauge(g: np.ndarray) -> np.ndarray:
    """g (n, k, nr) -> the same vectors with the last row real non-negative."""
    return g * np.exp(-1j * np.angle(g[..., -1:]))


def smooth(g: np.ndarray, kernel: tuple) -> np.ndarray:
    """Centred kernel along the group axis of g (n, k, nr), unit norm out."""
    k = g.shape[1]
    centre = len(kernel) // 2
    out = np.zeros_like(g)
    for j, wt in enumerate(kernel):
        off = j - centre
        lo, hi = max(0, -off), min(k, k - off)
        out[:, lo:hi] += wt * g[:, lo + off:hi + off]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def quantise(r: np.ndarray, cb: str) -> np.ndarray:
    """Reported unit vectors (n, k, nr) -> the column the access point
    rebuilds from the first-column angles at codebook cb, (n, k, nr, 1)."""
    R1 = r[..., None]
    if CODEBOOKS[cb] is None:
        return R1
    b_phi, b_psi = CODEBOOKS[cb]
    phi, psi = C.v_to_givens(R1, nc=N_STREAMS)
    return C.givens_to_v(C.quant_midpoint(phi, C.PHI_LO, C.PHI_HI, b_phi),
                         C.quant_midpoint(psi, C.PSI_LO, C.PSI_HI, b_psi),
                         nc=N_STREAMS)


def chunk_gains(V1: np.ndarray, rot: str, rng) -> dict:
    """V1 (n, k, nr, 1) true first columns -> per-report gain of every
    (kernel, codebook) under one draw of the rotation."""
    Q = draw_rotation(rot, V1.shape[0], V1.shape[1], rng)
    g = V1 if Q is None else np.conj(np.swapaxes(Q, -1, -2)) @ V1
    g = gauge(g[..., 0])
    out = {}
    for kern, h in KERNELS.items():
        r = smooth(g, h)
        for cb in CODEBOOKS:
            Vh = quantise(r, cb)
            if Q is not None:
                Vh = Q @ Vh
            out[(kern, cb)] = HG.gains(Vh, V1)[0]
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summarise(per_seed: dict, n_reports: int) -> dict:
    cells = {}
    for (rot, kern, cb), v in per_seed.items():
        mean = float(np.mean(v["mean"]))
        cells[cell_name(rot, kern, cb)] = dict(
            rotation=rot, kernel=kern, kernel_width=len(KERNELS[kern]),
            codebook=cb, gain=mean,
            gain_sd_over_seeds=float(np.std(v["mean"], ddof=1)) if len(v["mean"]) > 1 else 0.0,
            gain_p05=float(np.mean(v["p_low"])), n_seeds=len(v["mean"]),
            n_reports_per_seed=n_reports, meets_gain_min=bool(mean >= GAIN_MIN))
    return cells


def band_tradeoff(cells: dict) -> dict:
    """Gain under smoothing next to the observer's accuracy for each band
    size; the accuracies are read from r_rotation_space.json."""
    space = json.loads(SPACE_JSON.read_text())["cells"]
    out = {}
    for rot in BAND_ROTATIONS:
        key = RS.cell_name(STRUCTURE, ROTATIONS[rot])
        obs = space[key]["he"][SPACE_OBSERVER]
        out[rot] = dict(
            band_groups=RS.FRESHNESS[ROTATIONS[rot]], space_cell=key,
            observer=SPACE_OBSERVER,
            observer_accuracy={str(T): obs[str(T)]["accuracy"] for T in SPACE_T},
            observer_sd={str(T): obs[str(T)]["sd"] for T in SPACE_T},
            chance=obs[str(SPACE_T[0])]["chance"],
            gain={cb: {kern: cells[cell_name(rot, kern, cb)]["gain"] for kern in KERNELS}
                  for cb in CODEBOOKS},
            gain_sd_over_seeds={cb: {kern: cells[cell_name(rot, kern, cb)]["gain_sd_over_seeds"]
                                     for kern in KERNELS} for cb in CODEBOOKS})
    return out


def reference_check(cells: dict) -> dict:
    """This script's unsmoothed cells against he_beamforming_gain's summary."""
    if not GAIN_JSON.exists():
        return {}
    ref = json.loads(GAIN_JSON.read_text())
    out = {}
    for (rot, kern, cb), name in REFERENCE_CELLS.items():
        here = cells[cell_name(rot, kern, cb)]["gain"]
        there = ref["cells"][name]["gain_col1"]
        out[cell_name(rot, kern, cb)] = dict(reference_cell=name, gain=here,
                                             reference_gain=there, diff=here - there)
    out["reference_n_reports"] = ref["n_reports"]
    out["reference_seeds"] = ref["seeds"]
    return out


def estimate_mb(n_reports: int, k: int) -> dict:
    chunk = min(CHUNK, n_reports) * k * C.HE_NR ** 2 * COMPLEX_BYTES * QR_COPIES / 1e6
    return dict(he_tensor_mb=C.he_estimate_mb(), he_working_set_mb=C.he_estimate_mb() * 1.5,
                rotation_chunk_mb=chunk, cells=len(ROTATIONS) * len(KERNELS) * len(CODEBOOKS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(n_reports: int, n_seeds: int) -> int:
    timer = C.Timer()
    C.set_vm_limit()
    d = C.load_he_angles()
    C.check_rss("He loaded")
    phi_all, psi_all = d["phi"], d["psi"]
    k = int(phi_all.shape[1])
    per_seed = {(rot, kern, cb): dict(mean=[], p_low=[])
                for rot in ROTATIONS for kern in KERNELS for cb in CODEBOOKS}
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng([seed, DRAW_STREAM])
        idx = np.sort(rng.choice(len(phi_all), size=n_reports, replace=False))
        V1 = C.givens_to_v(phi_all[idx], psi_all[idx])[..., :N_STREAMS]   # truth (n, 59, 4, 1)
        for ri, rot in enumerate(ROTATIONS):
            rng_q = np.random.default_rng([seed, ROT_STREAM, ri])
            parts = {}
            for s in range(0, n_reports, CHUNK):
                for key, g1 in chunk_gains(V1[s:s + CHUNK], rot, rng_q).items():
                    parts.setdefault(key, []).append(g1)
            for (kern, cb), v in parts.items():
                g1 = np.concatenate(v)
                per_seed[(rot, kern, cb)]["mean"].append(float(g1.mean()))
                per_seed[(rot, kern, cb)]["p_low"].append(float(np.percentile(g1, P_LOW)))
        C.check_rss(f"seed {seed}")
        del V1
    cells = summarise(per_seed, n_reports)
    rows = [dict(cell=name, **c) for name, c in cells.items()]
    E.write_csv(OUT["csv"], rows)
    X.write_json(OUT["json"], dict(
        model="rank-one, first stream: g_k = Q_k^H v_k, last-row gauge, kernel "
              "over tone groups, normalise, quantise, access point applies Q_k",
        parameters=dict(
            n_reports=n_reports, seeds=n_seeds, groups_per_report=k,
            sc_stride=C.HE_SC_STRIDE, nr=C.HE_NR, structure=STRUCTURE,
            rotations={r: f for r, f in ROTATIONS.items()},
            kernels={kn: list(h) for kn, h in KERNELS.items()},
            codebooks={cb: None if b is None else list(b) for cb, b in CODEBOOKS.items()},
            paper_codebook=PAPER_CODEBOOK, gain_min=GAIN_MIN, chunk=CHUNK,
            draw_stream=DRAW_STREAM, rot_stream=ROT_STREAM, p_low=P_LOW),
        random_reference=dict(gain=1.0 / C.HE_NR),
        cells=cells, band_tradeoff=band_tradeoff(cells),
        reference_check=reference_check(cells),
        runtime_s=timer(), peak_rss_mb=C.peak_rss_mb()))
    for cb in CODEBOOKS:
        print(f"{cb}: mean gain (* = below {GAIN_MIN})")
        print(f"{'':8s}" + "".join(f"{kern:>9s}" for kern in KERNELS))
        for rot in ROTATIONS:
            g = [cells[cell_name(rot, kern, cb)]["gain"] for kern in KERNELS]
            print(f"{rot:8s}" + "".join(f"{x:8.4f}{' ' if x >= GAIN_MIN else '*'}" for x in g))
    print(f"runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", type=int, default=N_REPORTS)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        k = math.ceil(json.loads(GAIN_JSON.read_text())["layout"]["raw_shape"][1]
                      / C.HE_SC_STRIDE) if GAIN_JSON.exists() else RS.GROUPS
        print(json.dumps(estimate_mb(a.reports, k), indent=1))
        return 0
    return run(a.reports, a.seeds)


if __name__ == "__main__":
    sys.exit(main())
