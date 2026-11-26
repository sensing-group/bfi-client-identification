#!/usr/bin/env python3
"""R-round4 checks: six analyses on existing data, no new capture.

1  TIMING AND RSSI BASELINE, home capture. What a padded, encrypted report
   frame still shows of its sender. r_eval's episodes (other_day_any, r_eval's
   grids and seed, traced by r_eval_checks on the pure-noise table), every
   feature set decided on those episodes (r_eval_checks.decide), K = 1 and
   all, T = 1. Window features, none of which reads the body, the format, the
   SNR byte or the address:
     timing        log10 of the median, mean, standard deviation, minimum and
                   maximum of the 31 intervals between the 32 reports of the
                   window
     rssi          mean and standard deviation of the 32 RSSI values
     timing+rssi   both
   next to i.i.d. standard normal features of the same width, to r_rotation's
   320-d noise reference and to the unprotected angles on the same episodes.
2  GAIN AT DEEPER CODEBOOKS, He CR1. The unprotected reports, DP-SQ, Manifold
   DP, BeamDancer and the rotation per report and per group at 4/2, 6/4 and
   9/7 bits (phi/psi): the scaled, raw and invariant cosine 1-NN observers
   (r_raw_observer_defenses' cells and streams, joint_defense_eval.He.observe)
   and the first-stream gain (he_beamforming_gain's report draws and stream
   order). The repo's functions are called with joint_defense_common's
   codebook constants set to the codebook under test (codebook()); at 4/2
   every cell must reproduce the stored value (checks in the JSON).
3  ALL FIVE ROOMS, He. The cells of part 2 at 4/2 bits in each room:
   joint_defense_eval.He on that room, its first round(n / 2) positions (in
   label order) as calibration and the rest held out, which is He's own
   20 / 19 split in Conference Room 1 and the target count of
   s2_env_out's in_room_cohort in every room. Chance, observers, gain.
4  TWO-USER ZERO-FORCING, He CR1. Two held-out positions are served together:
   the access point builds a zero-forcing precoder from the two fed-back
   first columns, as decoded after each defense (after un-rotation for the
   rotation per group), and transmits to both. Metric and normalisation are
   written in the JSON (part4_zero_forcing.metric).
5  EQUIVALENCE, home capture. The rotation per group on the common grid, on
   the devices' own grids and combined with one format and no SNR field
   (r_round3_checks part 2), against the pure-noise reference on the same
   episodes: the paired bootstrap of the difference (r_eval_checks.boot_draws,
   BOOT_REPEATS calls), its 90 % and 95 % intervals and the smallest margin
   for which two one-sided tests at alpha = TOST_ALPHA declare equivalence
   (the larger absolute end of the 1 - 2 alpha interval).
6  FIVE DEVICES. The home protocol without the Samsung TV in gallery and
   queries, under the three scalings of r_eval_checks: (a) r_eval's protocol
   rerun on the five-device window table (its own draws, scaling fitted on
   the five devices' windows), (b) the stored six-device episodes with the
   device removed from gallery and queries. Windows and decisions per device.

Memory: the home parts hold window tables only (tens of MB) and read two
columns per archive day; He one room at a time, reports in chunks.
Usage: python3 scripts/r_round4_checks.py [--dry-run] [--part 1 2 3 4 5 6]
Outputs (figures/): r_round4_checks.json, r_round4_checks.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import functools
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# One BLAS thread, as r_round3_checks.py and r_raw_observer_defenses.py.
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402
from scipy import stats as SS  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beamdancer_sweep as B  # noqa: E402
import he_beamforming_gain as HG  # noqa: E402
import joint_defense_common as C  # noqa: E402
import joint_defense_eval as J  # noqa: E402
import manifold_sweep as MS  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_eval_checks as K  # noqa: E402
import r_raw_observer_defenses as RD  # noqa: E402
import r_rotation as RR  # noqa: E402
import r_rotation_ndp as RN  # noqa: E402
import r_rotation_space as RS  # noqa: E402
import r_round3_checks as P3  # noqa: E402
import s2_env_out as S2  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARTS = ("1", "2", "3", "4", "5", "6")
ROUND4_SEED = 29
RSS_BUDGET_MB = 3000.0
LOCK_FILE = X.CACHE_DIR / "r_round4_checks.lock"
PCT = 100.0

# Home protocol (parts 1, 5, 6) ------------------------------------------
HOME_COND = K.DEFENSE_CONDITION           # other_day_any
HOME_K = K.DEFENSE_K                      # (1, 0), 0 = all enrolled windows
REPORT_SCHEME = P3.REPORT_SCHEME          # r_eval.boot_ci's clusters
STORED = P3.STORED_SCALING                # r_eval's z-score on all windows
MAIN_METRICS = ("accuracy", "macro_recall")

# Part 1 ----------------------------------------------------------------
INTERVAL_FLOOR_S = 1e-6                   # two reports with one timestamp
TIMING_STATS = ("median", "mean", "std", "min", "max")
RSSI_STATS = ("mean", "std")
BASELINE_SETS = ("timing", "rssi", "timing+rssi")
STREAM_NULL = 1

# He (parts 2, 3, 4) -----------------------------------------------------
HE_BITS = (C.HE_B_PHI, C.HE_B_PSI)        # the paper's codebook, 4/2
CODEBOOKS = tuple(B.CODEBOOKS) + ((9, 7),)
CODEBOOK_BIT_GAP = C.HE_B_PHI - C.HE_B_PSI  # phi bits - psi bits in 802.11
ROOMS = S2.ROOMS
ROOM_FIRST = ROOMS[0]                     # Conference Room 1, the paper's
HE_SEEDS = RD.N_SEEDS                     # 10
REF_NAME = "none"
ROT_REPORT, ROT_GROUP = "unitary_pkt", "unitary_pkt_sc"
BD_R = tuple(r for r in B.R_GRID if r > 0)
HE_OBSERVERS = ("scaled", "raw", "invariant")
INV_ATTACKER_ID = 2                       # joint_defense_eval scores nn1_inv third
FIGURE_OBSERVERS = ("scaled", "invariant")  # fig_privacy_test's ATTACKERS
T_ONE = 1
UPPER_LEVEL = K.UPPER_LEVEL               # one-sided bound over seeds
HE_LOAD_FACTOR = RN.EST_HE_LOAD_FACTOR

# Part 4 ----------------------------------------------------------------
ZF_SNR_DB = 20.0
ZF_USERS = 2
ZF_PAIRS = HG.N_REPORTS                   # pairs of reports per seed
ZF_CODEBOOKS = tuple(B.CODEBOOKS)         # 4/2 and 6/4
ZF_SINGULAR_TOL = 1e-9                    # det of the fed-back Gram matrix
ZF_LOW_PCT = 5.0
DB_PER_DECADE = 10.0
STREAM_ZF_PAIRS, STREAM_ZF_DEFENSE = 2, 3
ZF_IDEAL, ZF_MRT = "true_vectors_zf", "true_vectors_mrt"
ZF_SELECT_QUANTILE = 0.25                 # an access point that pairs users by correlation
ZF_SCOPES = ("all_pairs", "least_correlated_quarter")

# Part 5 ----------------------------------------------------------------
EQ_BODIES = (RN.COMMON_BODY, P3.ROTATION)  # common grid, devices' own grids
EQ_COMBINED = f"{RN.COMMON_BODY}+one_format+no_snr"
TOST_ALPHA = 0.05
CI_LEVELS = (0.90, 0.95)
BOOT_REPEATS = 10                         # calls of K.boot_draws (K.N_BOOT each)
QUANTILE_GRID = (0.005, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
                 0.975, 0.995)
HIST_EDGES = np.linspace(-0.15, 0.15, 61)
MARGINS = (0.01, 0.02, 0.04, 0.05)

# Part 6 ----------------------------------------------------------------
EXCLUDED_DEVICE = "SamsungTV"
MIN_GALLERY = 2                           # r_eval skips smaller galleries

OUT_JSON = R.FIG_DIR / "r_round4_checks.json"
OUT_CSV = R.FIG_DIR / "r_round4_checks.csv"
CSV_FIELDS = P3.CSV_FIELDS
row = P3.row


def log(tag: str) -> None:
    pk = X.log_rss(tag)
    if pk > RSS_BUDGET_MB:
        print(f"[rss] WARNING above the {RSS_BUDGET_MB:.0f} MB budget", file=sys.stderr)


def stat(v) -> dict:
    """Mean, sd and one-sided upper bound of the mean over seeds."""
    v = np.asarray(v, float)
    out = P3.stat(v)
    out["ub95"] = out["mean"] if v.size < 2 else float(
        out["mean"] + SS.t.ppf(UPPER_LEVEL, v.size - 1) * out["sd"] / math.sqrt(v.size))
    return out


@contextlib.contextmanager
def patched(module, **values):
    """Module constants set for the duration of a block."""
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def codebook(bits: tuple[int, int]):
    """joint_defense_common quantising at (phi bits, psi bits): he_quantise,
    DP-SQ and Manifold DP read these constants when called. Manifold DP's
    single grid over [-pi, pi] gives psi its bits only while the two differ
    by CODEBOOK_BIT_GAP, as in every 802.11 codebook."""
    if bits[0] - bits[1] != CODEBOOK_BIT_GAP:
        raise ValueError(bits)
    return patched(C, HE_B_PHI=bits[0], HE_B_PSI=bits[1], B_HEADLINE=bits[0])


def bits_name(bits: tuple[int, int]) -> str:
    return f"{bits[0]}/{bits[1]}"


# ---------------------------------------------------------------------------
# Home: r_eval's episodes, decided on any feature matrix
# ---------------------------------------------------------------------------
def home_episodes(devs: list[str], trials: int):
    """r_common's windows, r_rotation's noise table, r_eval's episodes traced
    on it (r_round3_checks part 2) and the noise table's z-scores."""
    w_ref = R.get_windows(devs, R.SESSION_GAP_S)
    w_null = K.null_windows(w_ref)
    tb = E.Table(w_null)
    eps = K.traced_episodes(tb, HOME_COND, trials)
    return w_ref, w_null, eps, tb.names, tb.Z


def table_z(w: dict, f: np.ndarray) -> np.ndarray:
    """r_eval.Table's own z-score of a feature matrix (stored values)."""
    return E.Table(dict(w, feat=f)).Z


def decide(eps: list, scaling: str, z: np.ndarray, raw: np.ndarray, rng_t) -> tuple[dict, dict]:
    hits, ties = {}, {}
    for k in HOME_K:
        res = [K.decide(ep, k, scaling, z, raw, rng_t) for ep in eps]
        hits[k] = np.array([h for h, _ in res])
        ties[k] = int(sum(t for _, t in res))
    return hits, ties


def stored_reference_checks(eps, names, z_null, w_ref, rng_t, trials: int) -> dict:
    """The noise table and the unprotected angles on the traced episodes
    against figures/r_rotation_main.csv (the first asserted by hits_of)."""
    stored = K.stored_defense()
    z_real = table_z(w_ref, w_ref["feat"])
    out, pts = {}, {}
    for k in HOME_K:
        kl = E.klabel(k)
        h_null, _ = K.hits_of(eps, k, STORED, z_null, z_null, rng_t)
        h_real = np.array([K.decide(ep, k, STORED, z_real, z_real, rng_t)[0] for ep in eps])
        pts[kl] = dict(null_noise=K.point(h_null, eps, names),
                       unprotected=K.point(h_real, eps, names))
        if trials == E.N_TRIALS:
            for body, name in ((RR.NULL_BODY, "null_noise"), ("none", "unprotected")):
                ref = stored.get((body, K.FEATURE, kl))
                if ref is not None:
                    out[f"{name}_K={kl}_vs_r_rotation_main"] = max(
                        abs(pts[kl][name][m] - float(ref[m])) for m in MAIN_METRICS)
    return dict(checks=out, points=pts)


def score_rows(part: str, analysis: str, cell: str, obs: str, k: int, r: dict) -> list[dict]:
    iv = r["intervals"][REPORT_SCHEME]
    base = dict(observer=obs, K=E.klabel(k), T=K.T_REPORT, chance=r["chance"],
                n=r["n_decisions"], note=f"{REPORT_SCHEME} bootstrap")
    out = []
    for m in MAIN_METRICS:
        out.append(row(part, analysis, cell, m, r[m], lo=iv[m]["lo"], hi=iv[m]["hi"],
                       ub95=iv[m]["ub95"], **base))
        if f"reference_{m}" in r:
            dm = iv[f"{m}_minus_reference"]
            out.append(row(part, analysis, cell, f"reference_{m}", r[f"reference_{m}"], **base))
            out.append(row(part, analysis, cell, f"{m}_minus_reference",
                           r[m] - r[f"reference_{m}"], lo=dm["lo"], hi=dm["hi"],
                           ub95=dm["ub95"], **base))
    return out


# ---------------------------------------------------------------------------
# 1  timing and RSSI of the frames
# ---------------------------------------------------------------------------
def frame_rssi(devs: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per device: timestamps (ascending) and RSSI of every report, two
    columns per archive day."""
    ts, rs = defaultdict(list), defaultdict(list)
    q = (f"SELECT ts, {X.SRC_COL}, {X.RSSI_COL} FROM bfi WHERE {X.SRC_COL} IN "
         f"({','.join('?' * len(devs))}) ORDER BY ts")
    for p in X.day_paths():
        c = X.connect_ro(p)
        for t, m, r in c.execute(q, devs):
            i = devs.index(m)
            ts[i].append(t)
            rs[i].append(np.nan if r is None else r)
        c.close()
    return {i: (np.asarray(ts[i]), np.asarray(rs[i], float)) for i in ts}


def window_frames(w: dict, devs: list[str]) -> np.ndarray:
    """Timestamps of the WINDOW_FRAMES reports of every window, (n, 32):
    the decodable reports of r_common's cache between the window's ends."""
    ts_dec = R.all_timestamps(devs)
    out = np.empty((len(w["dev"]), R.WINDOW_FRAMES))
    for j, (i, a, b) in enumerate(zip(w["dev"], w["ts0"], w["ts1"])):
        t = ts_dec[devs[int(i)]]
        s = int(np.searchsorted(t, a))
        x = t[s:s + R.WINDOW_FRAMES]
        if x.size != R.WINDOW_FRAMES or x[-1] != b:
            raise RuntimeError(f"window {j}: frames not aligned with r_common's cut")
        out[j] = x
    return out


def timing_features(t: np.ndarray) -> np.ndarray:
    d = np.maximum(np.diff(t, axis=1), INTERVAL_FLOOR_S)
    s = dict(median=np.median(d, 1), mean=d.mean(1),
             std=np.maximum(d.std(1), INTERVAL_FLOOR_S), min=d.min(1), max=d.max(1))
    return np.log10(np.stack([s[k] for k in TIMING_STATS], axis=1)).astype(np.float32)


def rssi_features(w: dict, t: np.ndarray, rssi: dict) -> tuple[np.ndarray, int]:
    """Window mean and standard deviation of the RSSI of the window's
    reports, and the number of reports without an RSSI value."""
    out = np.zeros((t.shape[0], len(RSSI_STATS)), np.float32)
    missing = 0
    for j, i in enumerate(w["dev"]):
        ts, rs = rssi[int(i)]
        at = np.searchsorted(ts, t[j])
        if not np.array_equal(ts[at], t[j]):
            raise RuntimeError(f"window {j}: reports not found in the archive")
        x = rs[at]
        missing += int(np.isnan(x).sum())
        if not np.isnan(x).all():
            out[j] = (np.nanmean(x), np.nanstd(x))
    return out, missing


def part1(devs: list[str], trials: int) -> tuple[dict, list[dict]]:
    w_ref, w_null, eps, names, z_null = home_episodes(devs, trials)
    log("1 episodes traced")
    rng_b = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED])
    rng_t = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED + 1])
    ref = stored_reference_checks(eps, names, z_null, w_ref, rng_t, trials)
    del w_null, z_null
    t = window_frames(w_ref, devs)
    f_time = timing_features(t)
    f_rssi, missing = rssi_features(w_ref, t, frame_rssi(devs))
    log("1 frames read")
    feats = dict(zip(BASELINE_SETS, (f_time, f_rssi, np.hstack([f_time, f_rssi]))))
    cells, rows = {}, []
    for si, (name, f) in enumerate(feats.items()):
        f_ref = np.random.default_rng([ROUND4_SEED, STREAM_NULL, si]) \
            .standard_normal(f.shape).astype(np.float32)
        z, z_ref = P3.zscore(f), P3.zscore(f_ref)
        hits, ties = decide(eps, STORED, z, z, rng_t)
        ref_hits, _ = decide(eps, STORED, z_ref, z_ref, rng_t)
        for k in HOME_K:
            r = K.score(eps, hits[k], names, rng_b, ref_hits[k])
            r0 = K.point(ref_hits[k], eps, names)
            r.update(ties=ties[k], feature_dim=int(f.shape[1]),
                     reference_accuracy=r0["accuracy"],
                     reference_macro_recall=r0["macro_recall"])
            cells[f"{name}|K={E.klabel(k)}"] = r
            rows += score_rows("1", "timing_rssi_baseline", name, "1nn", k, r)
    for kl, pts in ref["points"].items():
        for body, pt in pts.items():
            for m in MAIN_METRICS:
                rows.append(row("1", "timing_rssi_baseline", body, m, pt[m], observer="1nn",
                                K=kl, T=K.T_REPORT, chance=pt["chance"], n=pt["n_decisions"],
                                note="320-d window vector, same episodes"))
    cols = list(TIMING_STATS) + [f"rssi_{s}" for s in RSSI_STATS]
    both = feats[BASELINE_SETS[-1]]
    per_dev = {names[g]: dict(windows=int((w_ref["dev"] == g).sum()), **{
        (c if c.startswith("rssi") else f"log10_interval_{c}_s"):
        float(np.median(both[w_ref["dev"] == g, j])) for j, c in enumerate(cols)})
        for g in np.unique(w_ref["dev"])}
    return dict(
        condition=HOME_COND, T=K.T_REPORT, trials=trials, K=[E.klabel(k) for k in HOME_K],
        n_boot=K.N_BOOT, ci_level=K.CI_LEVEL, interval_scheme_reported=REPORT_SCHEME,
        windows=int(t.shape[0]), window_frames=R.WINDOW_FRAMES, episodes=len(eps),
        features=dict(
            timing="log10 of the " + ", ".join(TIMING_STATS) + " of the 31 intervals "
                   "between the window's 32 reports (seconds, floor "
                   f"{INTERVAL_FLOOR_S:g} s)",
            rssi="mean and standard deviation of the 32 RSSI values "
                 f"(column {X.RSSI_COL})",
            note="no angle, format, SNR byte, body length or address; the "
                 "window is r_common's (32 decodable reports of one session)"),
        observer="r_eval's 1-NN on the z-scored features (fitted on all "
                 "windows), same episodes for every feature set",
        reference="i.i.d. standard normal features of the same width, same "
                  "episodes (reference_* and *_minus_reference); null_noise = "
                  "r_rotation's 320-d noise table",
        rssi_values_missing=missing, per_device_window_median=per_dev,
        stored_points=ref["points"], checks=ref["checks"], cells=cells), rows


# ---------------------------------------------------------------------------
# He: cells, observers and gains of one room at one codebook
# ---------------------------------------------------------------------------
def room_root(room: str) -> Path:
    return S2.HE_ARCHIVE_ROOT / room


def room_positions(room: str) -> int:
    import torch
    y = torch.load(str(room_root(room) / "y.pth"), map_location="cpu", weights_only=True)
    return int(np.unique(y.numpy()).size)


def n_calibration(room: str) -> int:
    return int(round(S2.COHORT_CAL_FRAC * room_positions(room)))


def load_room(room: str) -> J.He:
    """joint_defense_eval.He on one room, its first n_calibration positions
    as calibration (He's own 20 of 39 in Conference Room 1)."""
    with patched(C, load_he_angles=functools.partial(C.load_he_angles, room_root(room)),
                 HE_N_CALIB_CLASSES=n_calibration(room)):
        return J.He(J.HE_CAL_PER_CLASS)


def he_cells(bits: tuple[int, int]) -> list[dict]:
    """The defenses scored, in r_raw_observer_defenses.cells' form."""
    out = [dict(name=REF_NAME, kind="joint", label="unprotected")]
    out += [dict(name=f"dpsq_e{e}", kind="joint", label=f"DP-SQ eps={e:g}") for e in J.DPSQ_EPS]
    out += [dict(name=MS._defense(e), kind="manifold", label=f"Manifold DP eps={e:g}")
            for e in MS.SWEEP_EPS]
    out += [dict(name=B.cell_name(bits, r), kind="beamdancer", r_pct=r,
                 label=f"BeamDancer R={r}") for r in BD_R]
    out += [dict(name=ROT_REPORT, kind="rotation", label="rotation per report"),
            dict(name=ROT_GROUP, kind="rotation", label="rotation per group")]
    return out


def he_streams(cell: dict, bits: tuple[int, int], seed: int) -> tuple[list[int], list[int]]:
    """(defense stream, query stream) of the stored run of this cell;
    beamdancer_sweep keys its streams by the codebook's phi bits."""
    if cell["kind"] == "beamdancer":
        s = [seed, B.OBSERVER_STREAM + cell["r_pct"], bits[0]]
        return s, s + [B.QUERY_STREAM]
    return RD.streams(cell, seed)


def he_scalers(ds: J.He, bits: tuple[int, int]):
    """He's three scalers refitted at the codebook in force: S0 behind a tap
    (r_raw_observer_defenses), S0_inv, and BeamDancer's own."""
    pq, sq = C.he_quantise(ds.phi["cal"], ds.psi["cal"])
    tap = RD.Tap(StandardScaler().fit(C.he_features(pq, sq)))
    ds.S0 = tap
    ds.S0_inv = StandardScaler().fit(ds._gram_invariant(pq, sq))
    bd = StandardScaler().fit(
        C.he_features(*B.report(ds.phi["cal"], ds.psi["cal"], bits, 0, None)))
    return tap, bd


def he_observe(ds: J.He, tap, bd_scaler, cell: dict, bits, rng) -> dict:
    """{observer: features of the held set under the cell's defense}."""
    if cell["kind"] == "beamdancer":
        F = C.he_features(*B.report(ds.phi["held"], ds.psi["held"], bits, cell["r_pct"], rng))
        return dict(raw=F, scaled=bd_scaler.transform(F))
    Xs, inv, _ = ds.observe("held", cell["name"], rng)
    out = dict(raw=tap.last, scaled=Xs)
    if inv is not None:
        out["invariant"] = inv
    return out


def he_observers(ds: J.He, bits: tuple[int, int], seeds: int, tag: str) -> dict:
    """{(cell, observer): {T: [accuracy per seed]}} at one codebook."""
    y = ds.y["held"]
    acc = defaultdict(lambda: defaultdict(list))
    with codebook(bits):
        tap, bd = he_scalers(ds, bits)
        for seed in range(1, seeds + 1):
            supports = C.pick_supports(y, seed)
            for cell in he_cells(bits):
                d_stream, q_stream = he_streams(cell, bits, seed)
                obs = he_observe(ds, tap, bd, cell, bits, np.random.default_rng(d_stream))
                for name, Z in obs.items():
                    qs = q_stream if name != "invariant" else \
                        d_stream[:-1] + [INV_ATTACKER_ID, RD.QUERY_STREAM]
                    for T, a in RD.score(Z, y, supports, qs).items():
                        acc[(cell["name"], name)][T].append(a)
                del obs
                tap.last = None
            gc.collect()
            log(f"{tag} {bits_name(bits)} seed {seed}")
    return acc


def fed_back_v(cell: dict, phi, psi, bits, rng, Q=None) -> np.ndarray:
    """V (n, k, nr, nc) the access point steers with after a defense, at the
    codebook in force; Q (n, k or 1, nr, nr) is the rotation it undoes."""
    if cell["kind"] == "rotation":
        V2 = np.conj(np.swapaxes(Q, -1, -2)) @ C.givens_to_v(phi, psi)
        return Q @ C.givens_to_v(*C.he_quantise(*C.v_to_givens(V2)))
    if cell["kind"] == "beamdancer":
        return HG.decode(*B.report(phi, psi, bits, cell["r_pct"], rng))
    eps = J._eps_of(cell["name"])
    if cell["kind"] == "manifold":
        flat = np.concatenate([phi, psi], axis=-1).reshape(phi.shape[0], -1)
        mo = C.manifold_dp(flat, C.PHI_LO, C.PHI_HI, eps, rng).reshape(phi.shape[:2] + (-1,))
        return HG.decode(mo[..., :C.HE_N_PAIRS], mo[..., C.HE_N_PAIRS:])
    if eps is None:
        return C.givens_to_v(*C.he_quantise(phi, psi))
    return C.givens_to_v(*C.he_quantise(phi, psi, eps=eps, rng=rng))


def draw_q(name: str, n: int, k: int, rng) -> np.ndarray:
    return C.haar_unitary(C.HE_NR, (n, k), rng) if name == ROT_GROUP \
        else C.haar_unitary(C.HE_NR, (n,), rng)[:, None]


def he_gains(phi_all, psi_all, bits: tuple[int, int], seeds: int) -> dict:
    """First-stream gain per cell and seed against the continuous V, on
    he_beamforming_gain's report draw and in its stream order (DP-SQ, then
    the rotation per report, then per group); Manifold DP and BeamDancer on
    the streams of their sweeps."""
    out = defaultdict(list)
    cells = he_cells(bits)
    with codebook(bits):
        for seed in range(1, seeds + 1):
            rng = np.random.default_rng([seed, MS.GAIN_STREAM])
            idx = np.sort(rng.choice(len(phi_all), size=HG.N_REPORTS, replace=False))
            phi, psi = phi_all[idx], psi_all[idx]
            V = C.givens_to_v(phi, psi)
            n, k = V.shape[:2]
            for cell in cells:
                name, kind = cell["name"], cell["kind"]
                if kind == "manifold":
                    r = np.random.default_rng(
                        [seed, MS.GAIN_STREAM, MS.SWEEP_EPS.index(J._eps_of(name))])
                elif kind == "beamdancer":       # beamdancer_sweep redraws the reports first
                    r = np.random.default_rng([seed, MS.GAIN_STREAM])
                    r.choice(len(phi_all), size=HG.N_REPORTS, replace=False)
                else:
                    r = rng
                Q = draw_q(name, n, k, rng) if kind == "rotation" else None
                Vh = fed_back_v(cell, phi, psi, bits, r, Q)
                out[name].append(float(HG.gains(Vh, V)[0].mean()))
                del Vh, Q
            del V
    return out


def he_summary(acc: dict, gains: dict, bits, chance: float) -> dict:
    """Per cell: observers at T in RD.T_REPORT, gain, and the two clauses of
    the definition (every observer's upper bound over seeds within
    RS.PASS_MARGIN of chance at one query window; gain >= RS.GAIN_MIN)."""
    out = {}
    for cell in he_cells(bits):
        name = cell["name"]
        c = dict(label=cell["label"], gain_col1=stat(gains[name]))
        c["gain_relative_to_unprotected"] = c["gain_col1"]["mean"] / float(np.mean(gains[REF_NAME]))
        for obs in HE_OBSERVERS:
            for T in RD.T_REPORT:
                v = acc.get((name, obs), {}).get(T)
                if v:
                    c[f"{obs}_T={T}"] = stat(v)
        seen = [o for o in HE_OBSERVERS if f"{o}_T={T_ONE}" in c]
        c["figure_observer_accuracy"] = max(
            c[f"{o}_T={T_ONE}"]["mean"] for o in seen if o in FIGURE_OBSERVERS)
        c["worst_observer_accuracy"] = max(c[f"{o}_T={T_ONE}"]["mean"] for o in seen)
        c["worst_observer_ub95"] = max(c[f"{o}_T={T_ONE}"]["ub95"] for o in seen)
        c["at_chance"] = bool(c["worst_observer_ub95"] <= chance + RS.PASS_MARGIN)
        c["keeps_gain"] = bool(c["gain_col1"]["mean"] >= RS.GAIN_MIN)
        c["passes"] = c["at_chance"] and c["keeps_gain"]
        out[name] = c
    return out


def he_rows(part: str, analysis: str, where: str, cells: dict, chance: float) -> list[dict]:
    rows = []
    for name, c in cells.items():
        for obs in HE_OBSERVERS:
            for T in RD.T_REPORT:
                st = c.get(f"{obs}_T={T}")
                if st:
                    rows.append(row(part, analysis, f"{where}|{name}", "accuracy", st["mean"],
                                    observer=obs, K=1, T=T, ub95=st["ub95"], chance=chance,
                                    n=st["n"], note=f"sd over seeds {st['sd']:.4f}"))
        g = c["gain_col1"]
        rows.append(row(part, analysis, f"{where}|{name}", "gain_col1", g["mean"], n=g["n"],
                        note=f"sd over seeds {g['sd']:.4f}"))
    return rows


def he_stored_checks(cells: dict, bits: tuple[int, int]) -> dict:
    """Largest absolute difference to the stored 10-seed values of the
    scripts this part reuses, per source."""
    def cell_of(path):
        return json.loads(path.read_text())["cells"] if path.exists() else {}

    diff = defaultdict(float)

    def cmp(key: str, mine, theirs) -> None:
        if mine is not None and theirs is not None:
            diff[key] = max(diff[key], abs(mine - theirs))

    gain, joint, mani, bd = (cell_of(p) for p in (RD.GAIN_JSON, RD.JOINT_JSON, MS.OUT_JSON, B.OUT_JSON))
    if bits == HE_BITS:
        if RD.OUT_JSON.exists():
            for r in json.loads(RD.OUT_JSON.read_text())["rows"]:
                c = cells.get(r["defense"], {})
                for obs, key in (("raw", "raw_mean"), ("scaled", "scaled_rerun_mean")):
                    cmp("observers_vs_r_raw_observer_defenses",
                        c.get(f"{obs}_T={r['T']}", {}).get("mean"), r[key])
        for cell in he_cells(bits):
            name, c = cell["name"], cells[cell["name"]]
            for T in RD.T_REPORT:
                cmp("invariant_vs_joint_defense_he", c.get(f"invariant_T={T}", {}).get("mean"),
                    joint.get(f"{name}|nn1_inv|T={T}", {}).get("mean"))
            if cell["kind"] == "manifold":
                cmp("gain_vs_manifold_sweep", c["gain_col1"]["mean"],
                    mani.get(name, {}).get("gain_col1", {}).get("mean"))
            elif cell["kind"] != "beamdancer":
                suffix = RD.GAIN_KEYED_DEFAULT if cell["kind"] == "rotation" else RD.GAIN_UNKEYED
                cmp("gain_vs_he_beamforming_gain", c["gain_col1"]["mean"],
                    gain.get(f"{name}|{suffix}", {}).get("gain_col1"))
    elif bits == tuple(HG.CODEBOOK_FINE):
        cmp("unprotected_gain_vs_he_beamforming_gain_6_4", cells[REF_NAME]["gain_col1"]["mean"],
            gain.get(f"{REF_NAME}_codebook_6_4|{RD.GAIN_UNKEYED}", {}).get("gain_col1"))
    for name, c in cells.items():
        if name in bd:
            cmp("beamdancer_gain_vs_sweep", c["gain_col1"]["mean"], bd[name]["gain_col1"]["mean"])
            for T in RD.T_REPORT:
                cmp("beamdancer_scaled_vs_sweep", c[f"scaled_T={T}"]["mean"],
                    bd[name][f"nn1_T={T}"]["mean"])
    return dict(diff)


def part2(seeds: int) -> tuple[dict, list[dict]]:
    d = C.load_he_angles(room_root(ROOM_FIRST))
    gains = {bits: he_gains(d["phi"], d["psi"], bits, seeds) for bits in CODEBOOKS}
    del d
    gc.collect()
    log("2 gains")
    ds = load_room(ROOM_FIRST)
    chance = 1.0 / len(np.unique(ds.y["held"]))
    books, rows, checks = {}, [], {}
    for bits in CODEBOOKS:
        acc = he_observers(ds, bits, seeds, "2")
        cells = he_summary(acc, gains[bits], bits, chance)
        books[bits_name(bits)] = cells
        rows += he_rows("2", "deeper_codebooks", bits_name(bits), cells, chance)
        if seeds == HE_SEEDS:
            checks[bits_name(bits)] = he_stored_checks(cells, bits)
        del acc
    del ds
    gc.collect()
    return dict(
        room=ROOM_FIRST, chance=chance, seeds=seeds, codebooks=[bits_name(b) for b in CODEBOOKS],
        T_report=list(RD.T_REPORT), gain_reports=HG.N_REPORTS,
        protocol="held-out positions, cosine 1-NN, K = 1; scaled = features "
                 "standardised on the calibration positions quantised at the same "
                 "codebook, raw = no scaling, invariant = |V_k^H V_0| "
                 "(joint_defense_eval's nn1_inv); gain = |v_hat_1^H v_1|^2 against the "
                 "unquantised V, the access point undoing Q for the rotations",
        codebook_note="joint_defense_common's functions with HE_B_PHI, HE_B_PSI and "
                      "B_HEADLINE set to the codebook; random streams are those of "
                      "the 4/2 runs, so the codebooks share their draws",
        clauses=dict(margin=RS.PASS_MARGIN, gain_min=RS.GAIN_MIN,
                     at_chance="largest one-sided 95 % bound over seeds of the three "
                               "observers at T = 1 <= chance + margin"),
        checks_max_abs_diff_to_stored=checks, cells=books), rows


# ---------------------------------------------------------------------------
# 3  the five rooms
# ---------------------------------------------------------------------------
def part3(seeds: int) -> tuple[dict, list[dict]]:
    rooms, rows = {}, []
    for room in ROOMS:
        t = time.time()
        d = C.load_he_angles(room_root(room))
        gains = he_gains(d["phi"], d["psi"], HE_BITS, seeds)
        n_reports = int(len(d["y"]))
        del d
        gc.collect()
        ds = load_room(room)
        n_held = len(np.unique(ds.y["held"]))
        chance = 1.0 / n_held
        acc = he_observers(ds, HE_BITS, seeds, f"3 {room}")
        cells = he_summary(acc, gains, HE_BITS, chance)
        rooms[room] = dict(
            positions=len(np.unique(ds.y["cal"])) + n_held,
            calibration_positions=len(np.unique(ds.y["cal"])), held_out_positions=n_held,
            reports=n_reports, held_out_reports=int(len(ds.y["held"])), chance=chance,
            runtime_s=time.time() - t, cells=cells,
            passing=[n for n, c in cells.items() if c["passes"]],
            checks_max_abs_diff_to_stored=(he_stored_checks(cells, HE_BITS)
                                           if room == ROOM_FIRST and seeds == HE_SEEDS else {}))
        rows += he_rows("3", "five_rooms", room, cells, chance)
        del ds, acc
        gc.collect()
        log(f"3 {room}")
    return dict(
        rooms_order=list(ROOMS), seeds=seeds, codebook=bits_name(HE_BITS),
        T_report=list(RD.T_REPORT), gain_reports=HG.N_REPORTS,
        split="per room: the first round(n_positions / 2) position labels calibrate "
              "(features standardised on them), the others are enrolled and queried; "
              "Conference Room 1 keeps joint_defense_eval's 20 / 19",
        clauses=dict(margin=RS.PASS_MARGIN, gain_min=RS.GAIN_MIN),
        rooms=rooms), rows


# ---------------------------------------------------------------------------
# 4  two-user zero-forcing
# ---------------------------------------------------------------------------
def zf_precoder(va: np.ndarray, vb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-forcing beams (n, k, nt, 2), unit norm, for two users whose
    fed-back channel directions are va, vb (n, k, nt); the pseudo-inverse
    serves the groups where the two fed-back vectors coincide, which the
    second value marks (n, k)."""
    H = np.stack([va, vb], axis=-1)
    G = np.conj(np.swapaxes(H, -1, -2)) @ H
    W = H @ np.linalg.pinv(G, hermitian=True)
    W = W / np.maximum(np.linalg.norm(W, axis=-2, keepdims=True), ZF_SINGULAR_TOL)
    return W, np.linalg.det(G).real < ZF_SINGULAR_TOL


def mrt_precoder(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    return np.stack([va, vb], axis=-1)


def zf_measure(W: np.ndarray, va: np.ndarray, vb: np.ndarray) -> dict:
    """SINR of each user and sum rate per (pair, group) for beams W on the
    true directions va, vb, power split equally."""
    H = np.stack([va, vb], axis=-1)
    P = np.abs(np.conj(np.swapaxes(H, -1, -2)) @ W) ** 2      # [user, beam]
    p = 10 ** (ZF_SNR_DB / DB_PER_DECADE) / ZF_USERS
    sig = p * np.diagonal(P, axis1=-2, axis2=-1)
    sinr = sig / (1.0 + p * P.sum(-1) - sig)
    db = DB_PER_DECADE * np.log10(sinr)
    return dict(sum_rate=float(np.log2(1.0 + sinr).sum(-1).mean()),
                sinr_db=float(db.mean()), sinr_db_low=float(np.percentile(db, ZF_LOW_PCT)))


def zf_pairs(y: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """ZF_PAIRS pairs of reports from two different positions."""
    ia = rng.choice(y.size, size=ZF_PAIRS, replace=False)
    ib = rng.integers(y.size, size=ZF_PAIRS)
    while (same := y[ia] == y[ib]).any():
        ib[same] = rng.integers(y.size, size=int(same.sum()))
    return ia, ib


def part4(seeds: int) -> tuple[dict, list[dict]]:
    ds = load_room(ROOM_FIRST)
    y = ds.y["held"]
    log("4 He loaded")
    res = defaultdict(lambda: defaultdict(list))
    for seed in range(1, seeds + 1):
        users = zf_pairs(y, np.random.default_rng([ROUND4_SEED, STREAM_ZF_PAIRS, seed]))
        phi = [ds.phi["held"][i] for i in users]
        psi = [ds.psi["held"][i] for i in users]
        V = [C.givens_to_v(p, s) for p, s in zip(phi, psi)]
        va, vb = V[0][..., 0], V[1][..., 0]
        n, k = va.shape[:2]
        corr = (np.abs(np.einsum("nkr,nkr->nk", np.conj(va), vb)) ** 2).mean(axis=1)
        scopes = dict(zip(ZF_SCOPES, (np.ones(n, bool),
                                      corr <= np.quantile(corr, ZF_SELECT_QUANTILE))))
        W_ideal = zf_precoder(va, vb)[0]
        ideal = {sc: zf_measure(W_ideal[m], va[m], vb[m]) for sc, m in scopes.items()}

        def keep(name: str, W: np.ndarray, **extra) -> None:
            """The beams' measures in every scope; extra: (pair, group) or pair arrays."""
            for sc, m in scopes.items():
                r = zf_measure(W[m], va[m], vb[m])
                r.update(sum_rate_ratio=r["sum_rate"] / ideal[sc]["sum_rate"],
                         sinr_loss_db=ideal[sc]["sinr_db"] - r["sinr_db"],
                         **{key: float(v[m].mean()) for key, v in extra.items()})
                for key, v in r.items():
                    res[(sc, name)][key].append(v)

        keep(ZF_IDEAL, W_ideal, true_vector_correlation=corr)
        keep(ZF_MRT, mrt_precoder(va, vb), true_vector_correlation=corr)
        for bits in ZF_CODEBOOKS:
            with codebook(bits):
                for ci, cell in enumerate(he_cells(bits)):
                    if cell["name"] == ROT_REPORT:
                        continue
                    stream = [ROUND4_SEED, STREAM_ZF_DEFENSE, seed, bits[0], ci]
                    # one sounding serves both users: one Q_k for the pair
                    Q = draw_q(cell["name"], n, k, np.random.default_rng(stream + [ZF_USERS])) \
                        if cell["kind"] == "rotation" else None
                    Vh = [fed_back_v(cell, phi[u], psi[u], bits,
                                     np.random.default_rng(stream + [u]), Q)
                          for u in range(ZF_USERS)]
                    W, singular = zf_precoder(Vh[0][..., 0], Vh[1][..., 0])
                    gain = np.mean([HG.gains(Vh[u], V[u])[0] for u in range(ZF_USERS)], axis=0)
                    keep(f"{bits_name(bits)}|{cell['name']}", W, gain_col1=gain,
                         fed_back_vectors_coincide=singular)
                    del Vh, W, Q
        del V, phi, psi, W_ideal
        gc.collect()
        log(f"4 seed {seed}")
    cells = {sc: {} for sc in ZF_SCOPES}
    rows = []
    for (sc, name), m in res.items():
        cells[sc][name] = {key: stat(v) for key, v in m.items()}
        for key, st in cells[sc][name].items():
            rows.append(row("4", "two_user_zero_forcing", name, key, st["mean"], observer=sc,
                            n=st["n"], note=f"sd over seeds {st['sd']:.4f}"))
    labels = {f"{bits_name(b)}|{c['name']}": c["label"] for b in ZF_CODEBOOKS for c in he_cells(b)}
    return dict(
        room=ROOM_FIRST, seeds=seeds, pairs_per_seed=ZF_PAIRS, snr_db=ZF_SNR_DB,
        codebooks=[bits_name(b) for b in ZF_CODEBOOKS], labels=labels,
        metric=dict(
            pairs="two reports of two different held-out positions per pair, "
                  f"{ZF_PAIRS} pairs per seed, every one of the 59 tone groups",
            channel="each user is one stream on its true first right-singular vector "
                    "v_u (the unquantised column of the dataset): it receives "
                    "y_u = v_u^H x + n_u with unit noise power; the dataset holds no "
                    "singular values, so both users have unit channel gain",
            precoder="W = F (F^H F)^+ with F = [v_hat_a, v_hat_b] the fed-back first "
                     "columns as the access point decodes them (after Q_k for the "
                     "rotation, one Q_k per group shared by the pair), columns scaled "
                     "to unit norm, power split equally",
            sinr="SINR_u = (P/2) |v_u^H w_u|^2 / (1 + (P/2) |v_u^H w_j|^2), "
                 f"P = {ZF_SNR_DB:g} dB over the noise",
            sum_rate="sum over the two users of log2(1 + SINR_u), bit/s/Hz, mean over "
                     "pairs and groups",
            sum_rate_ratio=f"sum_rate / sum_rate of {ZF_IDEAL} (same pairs, per seed)",
            sinr_db="mean over users, pairs and groups of 10 log10 SINR_u; sinr_db_low = "
                    f"its {ZF_LOW_PCT:g}th percentile",
            sinr_loss_db=f"sinr_db of {ZF_IDEAL} minus sinr_db of the cell (per seed)",
            scopes=dict(zip(ZF_SCOPES, (
                "every pair drawn",
                f"per seed, the {ZF_SELECT_QUANTILE:g} of the pairs with the lowest "
                "true_vector_correlation, as an access point that chooses whom to serve "
                "together would pair"))),
            true_vector_correlation="|v_a^H v_b|^2 of the two true vectors, mean over "
                                    "groups and pairs",
            references={ZF_IDEAL: "zero-forcing on the unquantised true vectors (no "
                                  "interference left)",
                        ZF_MRT: "no nulling: each user's beam is its true vector"},
            gain_col1="single-user gain |v_hat^H v|^2 of the same fed-back vectors, "
                      "mean over both users",
            fed_back_vectors_coincide="share of (pair, group) whose two fed-back "
                                      "vectors are parallel, served by the pseudo-inverse"),
        cells=cells), rows


# ---------------------------------------------------------------------------
# 5  equivalence to the noise reference
# ---------------------------------------------------------------------------
def difference_summary(x: np.ndarray) -> dict:
    """A bootstrap distribution of a difference: intervals at CI_LEVELS and
    the smallest symmetric margin two one-sided tests at TOST_ALPHA accept."""
    lo, hi = np.quantile(x, [TOST_ALPHA, 1.0 - TOST_ALPHA])
    out = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), draws=int(x.size),
               quantiles={f"{q:g}": float(np.quantile(x, q)) for q in QUANTILE_GRID},
               histogram=dict(edges=[float(e) for e in HIST_EDGES],
                              counts=np.histogram(x, HIST_EDGES)[0].tolist(),
                              outside=int(((x < HIST_EDGES[0]) | (x > HIST_EDGES[-1])).sum())),
               tost_alpha=TOST_ALPHA,
               smallest_equivalence_margin=float(max(abs(lo), abs(hi))),
               upper_bound_one_sided=float(hi),
               share_of_draws_above_zero=float((x > 0).mean()))
    for level in CI_LEVELS:
        a = (1.0 - level) / 2.0
        out[f"ci{level * PCT:.0f}"] = [float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))]
    out["equivalent_within"] = {f"{m:g}": bool(out["smallest_equivalence_margin"] <= m)
                                for m in MARGINS}
    return out


def boot_difference(eps: list, hits: np.ndarray, ref: np.ndarray, rng) -> dict:
    """Paired bootstrap of hits - ref per resampling scheme of r_eval_checks."""
    dev = np.array([ep.dev for ep in eps])
    out = {}
    for scheme in K.SCHEMES:
        cl, strata = K.scheme_clusters(eps, scheme)
        draws = [K.boot_draws([hits, ref], dev, cl, strata, rng) for _ in range(BOOT_REPEATS)]
        acc = np.concatenate([d[0] for d in draws])
        mac = np.concatenate([d[1] for d in draws])
        out[scheme] = dict(
            n_clusters=int(len(set(cl.tolist()))),
            accuracy_minus_reference=difference_summary(acc[:, 0] - acc[:, 1]),
            macro_recall_minus_reference=difference_summary(mac[:, 0] - mac[:, 1]))
    return out


def part5(devs: list[str], trials: int) -> tuple[dict, list[dict]]:
    w_ref, w_null, eps, names, z_null = home_episodes(devs, trials)
    log("5 episodes traced")
    rng_b = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED])
    rng_t = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED + 1])
    null = dict(angles=w_null["feat"], invariant=w_null["inv"])
    sets = {}                                # (body, observer) -> (z, raw, scalings)
    refs = {"angles": (z_null, null["angles"].astype(np.float64), K.SCALINGS),
            "invariant": (table_z(w_ref, null["invariant"]),
                          null["invariant"].astype(np.float64), K.SCALINGS)}
    for body in EQ_BODIES:
        w = K.defense_windows(body, devs, w_ref)
        for obs in K.DEFENSE_OBSERVERS:
            f = w["feat"] if obs == "angles" else w["inv"]
            sets[(body, obs)] = (table_z(w_ref, f), f.astype(np.float64), K.SCALINGS)
        if body == RN.COMMON_BODY:           # r_round3_checks part 2, SNR removed
            fmt = np.ones((w["feat"].shape[0], P3.FORMAT_DIM), np.float32)
            z = P3.zscore(np.hstack([w["feat"], fmt]))
            sets[(EQ_COMBINED, "combined")] = (z, z, (STORED,))
            z = P3.zscore(np.hstack([null["angles"], fmt]))
            refs["combined"] = (z, z, (STORED,))
        del w
        gc.collect()
    del w_null
    ref_hits = {(obs, s): decide(eps, s, z, raw, rng_t)[0]
                for obs, (z, raw, scalings) in refs.items() for s in scalings}
    stored = K.stored_defense()
    cells, rows, checks = {}, [], {}
    for (body, obs), (z, raw, scalings) in sets.items():
        for scaling in scalings:
            hits, ties = decide(eps, scaling, z, raw, rng_t)
            for k in HOME_K:
                kl = E.klabel(k)
                pt = K.point(hits[k], eps, names)
                p0 = K.point(ref_hits[(obs, scaling)][k], eps, names)
                c = dict(ties=ties[k], n_decisions=pt["n_decisions"], chance=pt["chance"],
                         bootstrap=boot_difference(eps, hits[k], ref_hits[(obs, scaling)][k], rng_b))
                for m in MAIN_METRICS:
                    c[m] = pt[m]
                    c[f"reference_{m}"] = p0[m]
                    c[f"{m}_minus_reference"] = pt[m] - p0[m]
                cells[f"{body}|{obs}|K={kl}|{scaling}"] = c
                s_ref = stored.get((body, obs, kl))
                if scaling == STORED and s_ref is not None and trials == E.N_TRIALS:
                    checks[f"{body}|{obs}|K={kl}_vs_r_rotation_main"] = max(
                        abs(pt[m] - float(s_ref[m])) for m in MAIN_METRICS)
                for scheme, bs in c["bootstrap"].items():
                    for m in MAIN_METRICS:
                        d = bs[f"{m}_minus_reference"]
                        base = dict(observer=f"{obs}|{scaling}", K=kl, T=K.T_REPORT,
                                    chance=pt["chance"], n=pt["n_decisions"])
                        for level in CI_LEVELS:
                            ci = d[f"ci{level * PCT:.0f}"]
                            rows.append(row("5", "equivalence", body, f"{m}_minus_reference",
                                            c[f"{m}_minus_reference"], lo=ci[0], hi=ci[1],
                                            note=f"{scheme} bootstrap, {level:g} interval", **base))
                        rows.append(row("5", "equivalence", body, f"{m}_tost_margin",
                                        d["smallest_equivalence_margin"],
                                        note=f"{scheme} bootstrap, alpha {TOST_ALPHA:g}", **base))
        log(f"5 {body} {obs}")
    return dict(
        condition=HOME_COND, T=K.T_REPORT, trials=trials, K=[E.klabel(k) for k in HOME_K],
        boot_draws=K.N_BOOT * BOOT_REPEATS, schemes=list(K.SCHEMES),
        scheme_reported=REPORT_SCHEME, tost_alpha=TOST_ALPHA,
        bodies={RN.COMMON_BODY: "rotation per group, 4/2 bits, 234 tone groups",
                P3.ROTATION: "rotation per group at each device's own codebook",
                EQ_COMBINED: "the first with a constant format column and no SNR "
                             "field (r_round3_checks part 2, snr_step=removed)"},
        reference="r_rotation's pure-noise table on the same episodes (for the "
                  "invariant observer its 64-d noise), same scaling",
        tost="equivalence within +-m holds at alpha when the 1 - 2 alpha percentile "
             "interval of the paired bootstrap difference lies inside (-m, m); "
             "smallest_equivalence_margin is the larger absolute end of that "
             "interval, upper_bound_one_sided its upper end",
        checks=checks, cells=cells), rows


# ---------------------------------------------------------------------------
# 6  the home protocol without the Samsung TV
# ---------------------------------------------------------------------------
def without_rows(w: dict, keep: np.ndarray) -> dict:
    """The window table restricted to rows (r_snr_field.two_streams' rule)."""
    return {k: (v[keep] if v.ndim and v.shape[0] == keep.size else v) for k, v in w.items()}


def without_device(eps: list, g: int) -> list:
    """Episodes with device g neither queried nor enrolled."""
    out = []
    for ep in eps:
        keep = [i for i, d in enumerate(ep.gal) if d != g]
        if ep.dev != g and len(keep) >= MIN_GALLERY:
            out.append(ep._replace(gal=tuple(ep.gal[i] for i in keep),
                                   perms=[ep.perms[i] for i in keep]))
    return out


def query_counts(w: dict, eps: list, names: list[str]) -> dict:
    """Per device: windows, and what the episodes queried of them."""
    out = {}
    for g in sorted({ep.dev for ep in eps}):
        mine = [ep for ep in eps if ep.dev == g]
        blocks = {ep.block for ep in mine}
        in_blocks = sum(int(((w["dev"] == g) & (w["session"] == b[1]) & (w["day"] == b[2])).sum())
                        for b in blocks)
        out[names[g]] = dict(
            windows=int((w["dev"] == g).sum()), decisions=len(mine),
            query_days=len({ep.day for ep in mine}), query_blocks=len(blocks),
            windows_in_query_blocks=in_blocks,
            distinct_windows_queried=len({int(ep.qi[0]) for ep in mine}))
    return out


def cross_day_windows(tb: E.Table, names: list[str]) -> dict:
    """Per device: the windows a cross-day query can come from, i.e. those of
    the (session, day) blocks for which the device has a block of another
    session on another day to enrol from (r_eval's other_day_any)."""
    out = {}
    for g in tb.devs:
        mine = [b for b in tb.blocks if b[0] == g]
        ok = [b for b in mine if any(c[2] != b[2] and c[1] != b[1] for c in mine)]
        out[names[g]] = dict(
            days=len({b[2] for b in mine}), blocks=len(mine),
            windows=int(sum(tb.blocks[b].size for b in mine)),
            cross_day_query_blocks=len(ok),
            cross_day_query_windows=int(sum(tb.blocks[b].size for b in ok)))
    return out


def flat(r: dict) -> dict:
    """A K.score result without the per-device maps of days and blocks."""
    return {k: v for k, v in r.items() if k not in ("per_device_query_blocks", "per_device_days")}


def part6(devs: list[str], trials: int) -> tuple[dict, list[dict]]:
    w = R.get_windows(devs, R.SESSION_GAP_S)
    names = [R.short(str(m)) for m in w["macs"]]
    g = names.index(EXCLUDED_DEVICE)
    w5 = without_rows(w, w["dev"] != g)
    rng_b = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED])
    rng_t = np.random.default_rng([ROUND4_SEED, K.BOOT_SEED + 1])
    raw = w["feat"].astype(np.float64)
    stored = json.loads(K.OUT_JSON.read_text())["results"]["home"] if K.OUT_JSON.exists() else {}
    out = dict(rerun_five_devices={}, same_episodes_five_devices={}, six_devices={})
    counts, rows, checks = {}, [], {}
    eligible = cross_day_windows(E.Table(w), names)
    for dev, v in eligible.items():
        for key, val in v.items():
            rows.append(row("6", "cross_day_query_windows", HOME_COND, key, val, observer=dev))
    for cond in K.CONDITIONS:
        # (a) the protocol on the five-device table
        res, _, _ = K.run_table(w5, cond, HOME_K, trials, rng_b, rng_t)
        for (k, scaling), r in res.items():
            out["rerun_five_devices"][f"{cond}|K={E.klabel(k)}|{scaling}"] = flat(r)
            rows += score_rows("6", "five_devices_rerun", cond, scaling, k, r)
        log(f"6 rerun {cond}")
        # (b) the six-device episodes, the device removed afterwards
        tb = E.Table(w)
        eps6 = K.traced_episodes(tb, cond, trials)
        eps5 = without_device(eps6, g)
        counts[cond] = dict(six_devices=query_counts(w, eps6, names),
                            five_devices_same_episodes=query_counts(w, eps5, names))
        for name, eps in (("six_devices", eps6), ("same_episodes_five_devices", eps5)):
            for k in HOME_K:
                for scaling in K.SCALINGS:
                    hits = np.array([K.decide(ep, k, scaling, tb.Z, raw, rng_t)[0] for ep in eps])
                    r = K.score(eps, hits, names, rng_b)
                    key = f"{cond}|K={E.klabel(k)}|{scaling}"
                    out[name][key] = flat(r)
                    rows += score_rows("6", name, cond, scaling, k, r)
                    if name == "six_devices" and key in stored and trials == E.N_TRIALS:
                        checks[key] = max(abs(r[m] - stored[key][m]) for m in MAIN_METRICS)
        del tb, eps6, eps5
        gc.collect()
        log(f"6 same episodes {cond}")
    for cond, c in counts.items():
        for name, per in c.items():
            for dev, v in per.items():
                for key, val in v.items():
                    rows.append(row("6", f"counts_{name}", cond, key, val, observer=dev))
    return dict(
        excluded=EXCLUDED_DEVICE, T=K.T_REPORT, trials=trials,
        K=[E.klabel(k) for k in HOME_K], conditions=list(K.CONDITIONS),
        scalings=dict(stored="z-score fitted on all windows (of the five devices in the "
                             "rerun, of the six in the same-episode variant)",
                      enrolled="z-score fitted on the windows enrolled in the decision",
                      raw="no scaling"),
        variants=dict(
            rerun_five_devices="r_eval's protocol, grids and seed on the window table "
                               "without the device: its own draws",
            same_episodes_five_devices="r_eval's stored episodes with the device's "
                                       "queries dropped and the device taken out of every "
                                       "gallery: the decisions of the paper's run",
            six_devices="the stored run, recomputed (checks: against r_eval_checks.json)"),
        n_boot=K.N_BOOT, ci_level=K.CI_LEVEL, interval_scheme_reported=REPORT_SCHEME,
        windows={n: int((w["dev"] == i).sum()) for i, n in enumerate(names)},
        cross_day_query_windows=eligible, counts=counts,
        checks_max_abs_diff_to_r_eval_checks=checks, **out), rows


# ---------------------------------------------------------------------------
def dry_run(devs: list[str]) -> dict:
    est = R.dry_run(devs)
    wp = R.windows_path(R.SESSION_GAP_S)
    rooms = {r: C.he_estimate_mb(room_root(r)) for r in ROOMS}
    return dict(
        parts=list(PARTS),
        archive_rows_per_day={k: v for k, v in est.items() if not k.startswith("_")},
        home=dict(r_common_windows_cached=wp.exists(),
                  rotated_windows_cached={b: RR.windows_path(b, R.SESSION_GAP_S).exists()
                                          for b in EQ_BODIES},
                  est_table_mb=(wp.stat().st_size * K.EST_TABLE_COPIES / 1e6
                                if wp.exists() else None),
                  part1="two columns (time, RSSI) per report of the labelled devices"),
        he=dict(room_file_mb=rooms,
                est_working_set_mb={r: v * HE_LOAD_FACTOR for r, v in rooms.items()},
                rooms_in_memory_at_once=1, seeds=HE_SEEDS,
                codebooks=[bits_name(b) for b in CODEBOOKS],
                cells=[c["name"] for c in he_cells(HE_BITS)], zf_pairs_per_seed=ZF_PAIRS),
        rss_budget_mb=RSS_BUDGET_MB, vm_limit_mb=X.VM_LIMIT_BYTES / 1e6)


def merge_and_write(results: dict, parts: list[str], timing: dict, peaks: dict, last: dict) -> None:
    """Results of this run merged into the files of earlier runs, under a
    lock so that parts run in separate processes do not overwrite each other."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        rows = [r for r in out.pop("csv_rows", []) if r["part"] not in parts]
        for key, (res, new_rows) in results.items():
            out[key] = res
            rows += new_rows
        rows.sort(key=lambda r: r["part"])
        out.update(
            runtime_s_per_part=dict(out.get("runtime_s_per_part", {}), **timing),
            peak_rss_mb_per_part=dict(out.get("peak_rss_mb_per_part", {}), **peaks),
            settings_per_part=dict(out.get("settings_per_part", {}), **{
                key: dict(trials=last["trials"], he_seeds=last["he_seeds"]) for key in results}),
            peak_rss_note="high-water mark of the process when the part ended; the "
                          "part's own peak when it ran alone (--part N)",
            last_run=last, csv_rows=rows)
        X.write_json(OUT_JSON, out)
        with open(OUT_CSV, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
            wr.writeheader()
            wr.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="memory estimate and cache state, nothing loaded")
    ap.add_argument("--part", "--parts", dest="parts", nargs="*", default=list(PARTS),
                    choices=PARTS)
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--he-seeds", type=int, default=HE_SEEDS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        print(json.dumps(dry_run(devs), indent=2))
        return 0
    todo = {
        "1": ("part1_timing_rssi", lambda: part1(devs, a.trials)),
        "2": ("part2_deeper_codebooks", lambda: part2(a.he_seeds)),
        "3": ("part3_five_rooms", lambda: part3(a.he_seeds)),
        "4": ("part4_zero_forcing", lambda: part4(a.he_seeds)),
        "5": ("part5_equivalence", lambda: part5(devs, a.trials)),
        "6": ("part6_five_devices", lambda: part6(devs, a.trials)),
    }
    results, timing, peaks = {}, {}, {}
    for p in PARTS:
        if p not in a.parts:
            continue
        key, fn = todo[p]
        t = time.time()
        results[key] = fn()
        timing[key] = time.time() - t
        peaks[key] = X.peak_rss_mb()
        gc.collect()
        print(f"[part {p}] {timing[key]:.0f}s peak RSS {peaks[key]:.0f} MB", file=sys.stderr)
    merge_and_write(results, a.parts, timing, peaks, dict(
        parts=a.parts, runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        trials=a.trials, he_seeds=a.he_seeds))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB; "
          f"wrote {OUT_JSON.name}, {OUT_CSV.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
