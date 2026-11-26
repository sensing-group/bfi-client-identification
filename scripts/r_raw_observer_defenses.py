"""The raw 1-NN observer under every defense scored on He CR1 real BFI.

The defenses on the 19 held-out positions are scored against the scaled
observer (cosine 1-NN on features standardised with clean statistics of the
20 calibration positions). s1_calibration_size.py (n_cal = 0) shows that the
same rule on the raw quantised features is stronger without a defense. This
script scores that raw rule under each defense.

Nothing is re-implemented: the defended features come from
joint_defense_eval.He.observe (DP-SQ, Manifold DP, the unitary and the keyed
orthogonal transforms, with the random streams of joint_defense_eval and
manifold_sweep) and from beamdancer_sweep.report (codebook RAW_BD_BITS, the
streams of beamdancer_sweep). A tap on the scaler keeps the feature matrix the
scaler receives, so one defended realisation is scored twice:
    raw      cosine 1-NN on that matrix (the n_cal = 0 rule of s1_calibration_size)
    scaled   cosine 1-NN on the scaler's output (the stored observer; the rerun
             is written next to the stored value as a check)
Supports are joint_defense_common.pick_supports(y, seed), seeds 1..N, T query
windows averaged with the query stream of the stored run, for both observers.

Usage: python3 scripts/r_raw_observer_defenses.py [--seeds 10] [--dry-run]
Outputs (figures/): r_raw_observer_defenses.csv, r_raw_observer_defenses.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os

# One BLAS thread: the batched 118x118 QR of the per-window transform runs about
# seven times slower with eight threads (measured), with identical output.
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import beamdancer_sweep as B  # noqa: E402
import joint_defense_common as C  # noqa: E402
import joint_defense_eval as J  # noqa: E402
import manifold_sweep as M  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SEEDS = 10
T_REPORT = (1, 32)                       # T values written (J.T_GRID is scored to keep the streams)
RAW_BD_BITS = (C.HE_B_PHI, C.HE_B_PSI)   # BeamDancer on the He codebook, the b42 cells
RAW_BD_R = (10, 25, 50, 100)
RAW_MANIFOLD_EPS = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
RAW_DPSQ_EPS = (0.01, 0.5, 2.0)
JOINT_KEYED = ("unitary_pkt", "unitary_pkt_sc", "unitary_fixed", "unitary_global", "orthq_window")
LABELS = {
    "none": "none",
    "unitary_pkt": "unitary per report",
    "unitary_pkt_sc": "unitary per report and tone group",
    "unitary_fixed": "unitary fixed per device",
    "unitary_global": "one shared unitary",
    "orthq_window": "keyed orthogonal transform per window",
}
GAIN_KEY_SUFFIX = {                      # cell of bfi_he_beamforming_gain_summary.json
    "orthq_window": "AP holds Q (float feedback)",
}
GAIN_KEYED_DEFAULT = "AP holds Q"
GAIN_UNKEYED = "none"
NN1_ATTACKER_ID = M.NN1_ATTACKER_ID
HELD_STREAM = 1                          # joint_defense_eval's held-set stream
QUERY_STREAM = 3                         # joint_defense_eval's query stream
PCT = 100.0
SANITY_TOL = 1e-12                       # raw 'none' against s1_calibration_size, per seed

JOINT_JSON = C.FIG_DIR / "bfi_joint_defense_he_summary.json"
GAIN_JSON = C.FIG_DIR / "bfi_he_beamforming_gain_summary.json"
S1_SEEDS_CSV = C.FIG_DIR / "s_calibration_size_seeds.csv"
OUT_CSV = C.FIG_DIR / "r_raw_observer_defenses.csv"
OUT_JSON = C.FIG_DIR / "r_raw_observer_defenses.json"
CSV_FIELDS = ("defense", "label", "T", "raw_mean", "raw_sd", "scaled_rerun_mean", "scaled_rerun_sd",
              "scaled_stored_mean", "scaled_stored_sd", "scaled_stored_source", "gain_stored",
              "gain_source", "n_seeds", "chance")


class Tap:
    """A scaler that keeps the matrix it was asked to transform."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def transform(self, X):
        self.last = X
        return self.inner.transform(X)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def cells() -> list[dict]:
    """Every scored defense: its name in the source script and its random streams."""
    out = [dict(name="none", kind="joint", label=LABELS["none"])]
    out += [dict(name=f"dpsq_e{e}", kind="joint", label=f"DP-SQ eps={e:g}") for e in RAW_DPSQ_EPS]
    out += [dict(name=M._defense(e), kind="manifold", label=f"Manifold DP eps={e:g}")
            for e in RAW_MANIFOLD_EPS]
    out += [dict(name=B.cell_name(RAW_BD_BITS, r), kind="beamdancer", r_pct=r,
                 label=f"BeamDancer R={r}") for r in RAW_BD_R]
    out += [dict(name=d, kind="joint", label=LABELS[d]) for d in JOINT_KEYED]
    return out


def streams(cell: dict, seed: int) -> tuple[list[int], list[int]]:
    """(defense stream, query stream) of the stored run of this cell."""
    if cell["kind"] == "beamdancer":
        s = [seed, B.OBSERVER_STREAM + cell["r_pct"], RAW_BD_BITS[0]]
        return s, s + [B.QUERY_STREAM]
    if cell["kind"] == "manifold":
        order = [M.REFERENCE] + [M._defense(e) for e in M.SWEEP_EPS]
        did = M._stream(cell["name"], order.index(cell["name"]))
    else:
        did = J.HE_DEFENSES.index(cell["name"])
    return [seed, did, HELD_STREAM], [seed, did, NN1_ATTACKER_ID, QUERY_STREAM]


def observe(ds, tap: Tap, bd_scaler, cell: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    """(raw features, scaled features) of the held set under the cell's defense."""
    if cell["kind"] == "beamdancer":
        F = C.he_features(*B.report(ds.phi["held"], ds.psi["held"], RAW_BD_BITS, cell["r_pct"], rng))
        return F, bd_scaler.transform(F)
    Xs, _, _ = ds.observe("held", cell["name"], rng)
    return tap.last, Xs


def score(Z, y, supports, q_stream) -> dict:
    return C.multi_query_eval(Z, y, supports, metric="cos", T_grid=J.T_GRID,
                              n_decisions=J.N_DECISIONS_PER_CLASS, rng=np.random.default_rng(q_stream))


# ---------------------------------------------------------------------------
# Stored values
# ---------------------------------------------------------------------------

def stored(cell: dict, T: int) -> dict:
    """Scaled accuracy and beamforming gain already in figures/ for this cell."""
    name = cell["name"]
    if cell["kind"] == "joint":
        acc = json.loads(JOINT_JSON.read_text())["cells"][f"{name}|nn1|T={T}"]
        suffix = GAIN_UNKEYED if name not in JOINT_KEYED else GAIN_KEY_SUFFIX.get(name, GAIN_KEYED_DEFAULT)
        key = f"{name}|{suffix}"
        gain = json.loads(GAIN_JSON.read_text())["cells"][key]["gain_col1"]
        return dict(acc=acc, acc_src=JOINT_JSON.name, gain=gain, gain_src=f"{GAIN_JSON.name}: {key}")
    path = M.OUT_JSON if cell["kind"] == "manifold" else B.OUT_JSON
    c = json.loads(path.read_text())["cells"][name]
    return dict(acc=c[f"nn1_T={T}"], acc_src=path.name, gain=c["gain_col1"]["mean"],
                gain_src=f"{path.name}: {name}")


def s1_raw_reference() -> dict:
    """Per-seed raw accuracy of s1_calibration_size (he, n_cal = 0, nn1, K = 1)."""
    with open(S1_SEEDS_CSV, newline="") as f:
        return {int(r["seed"]): float(r["acc"]) for r in csv.DictReader(f)
                if r["dataset"] == "he" and r["n_cal"] == "0" and r["attacker"] == "nn1" and r["K"] == "1"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _stat(v) -> tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def run(n_seeds: int) -> None:
    timer = C.Timer()
    C.set_vm_limit()
    C.check_rss("start")
    ds = J.He(J.HE_CAL_PER_CLASS)
    tap = Tap(ds.S0)
    ds.S0 = tap
    bd_scaler = StandardScaler().fit(
        C.he_features(*B.report(ds.phi["cal"], ds.psi["cal"], RAW_BD_BITS, 0, None)))
    C.check_rss("data loaded")
    y = ds.y["held"]
    chance = 1.0 / len(np.unique(y))
    todo = cells()
    acc = {c["name"]: dict(raw={T: [] for T in J.T_GRID}, scaled={T: [] for T in J.T_GRID}) for c in todo}
    cell_s = {c["name"]: 0.0 for c in todo}
    for seed in range(1, n_seeds + 1):
        supports = C.pick_supports(y, seed)
        for c in todo:
            t0 = timer()
            d_stream, q_stream = streams(c, seed)
            F, Xs = observe(ds, tap, bd_scaler, c, np.random.default_rng(d_stream))
            for obs, Z in (("raw", F), ("scaled", Xs)):
                for T, a in score(Z, y, supports, q_stream).items():
                    acc[c["name"]][obs][T].append(a)
            cell_s[c["name"]] += timer() - t0
            del F, Xs
            tap.last = None
            gc.collect()
        C.check_rss(f"after seed {seed}")
        print(f"  seed {seed} done ({timer():.0f}s)", flush=True)

    rows, per_seed = [], {}
    for c in todo:
        per_seed[c["name"]] = {obs: {str(T): v for T, v in byT.items()} for obs, byT in acc[c["name"]].items()}
        for T in T_REPORT:
            st = stored(c, T)
            rm, rs = _stat(acc[c["name"]]["raw"][T])
            sm, ss = _stat(acc[c["name"]]["scaled"][T])
            rows.append(dict(defense=c["name"], label=c["label"], T=T, raw_mean=rm, raw_sd=rs,
                             scaled_rerun_mean=sm, scaled_rerun_sd=ss,
                             scaled_stored_mean=st["acc"]["mean"], scaled_stored_sd=st["acc"]["sd"],
                             scaled_stored_source=st["acc_src"], gain_stored=st["gain"],
                             gain_source=st["gain_src"], n_seeds=n_seeds, chance=chance))
    ref = s1_raw_reference()
    got = acc["none"]["raw"][1]
    d_raw = [abs(got[s - 1] - ref[s]) for s in range(1, n_seeds + 1) if s in ref]
    sanity = dict(
        raw_none_vs_s1_max_abs_diff_per_seed=max(d_raw) if d_raw else None,
        raw_none_matches_s1=bool(d_raw) and max(d_raw) <= SANITY_TOL,
        scaled_rerun_vs_stored_max_abs_diff=max(abs(r["scaled_rerun_mean"] - r["scaled_stored_mean"])
                                                for r in rows),
        scaled_rerun_vs_stored_valid=n_seeds == N_SEEDS)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(dict(
        dataset="he", room="Conference Room 1", chance=chance, seeds=n_seeds, T_report=list(T_REPORT),
        T_scored=list(J.T_GRID), n_decisions_per_class=J.N_DECISIONS_PER_CLASS,
        raw_rule="cosine 1-NN, one support window per position, raw defended features (no scaling)",
        scaled_rule="same rule on features standardised with clean calibration statistics",
        beamdancer_codebook=list(RAW_BD_BITS), rows=rows, per_seed=per_seed, sanity=sanity,
        run=dict(runtime_s=timer(), seconds_per_cell_all_seeds=cell_s, peak_rss_mb=C.peak_rss_mb())),
        indent=1))
    print(f"{'defense':>40s} {'raw T=1':>8s} {'raw T=32':>8s} {'scl T=1':>8s} {'stored':>8s} {'gain':>6s}")
    for c in todo:
        r1, r32 = (next(r for r in rows if r["defense"] == c["name"] and r["T"] == T) for T in T_REPORT)
        print(f"{c['label']:>40s} {PCT * r1['raw_mean']:8.1f} {PCT * r32['raw_mean']:8.1f} "
              f"{PCT * r1['scaled_rerun_mean']:8.1f} {PCT * r1['scaled_stored_mean']:8.1f} "
              f"{r1['gain_stored']:6.3f}")
    print(f"sanity {sanity}")
    print(f"wrote {OUT_CSV} and {OUT_JSON.name}; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"{len(cells())} defenses x {a.seeds} seeds, two observers each; estimated working set "
              f"~{C.he_estimate_mb() * 2.5:.0f} MB (VM cap {C.VM_LIMIT_GB} GB, RSS cap {C.MAX_RSS_MB} MB)")
        for c in cells():
            print(f"  {c['name']:>18s}  {c['label']}")
        return
    run(a.seeds)


if __name__ == "__main__":
    main()
