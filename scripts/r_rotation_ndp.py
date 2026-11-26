#!/usr/bin/env python3
"""R-rotation-NDP: an observer that also measures the rotated sounding.

The rotation of Section 7.2 sends the sounding of tone group k through a
fresh Haar unitary Q_k (N_t x N_t); the client reports the Givens angles of
Q_k^H V_k, quantised. r_rotation.py and r_rotation_space.py score observers
of the report alone. The same passive receiver also hears the rotated
sounding through its own channel G_k (m x N_t, m = its antennas): from the
training fields it estimates A_k = G_k Q_k (+ noise), and from the report it
reads B_k = quant(Q_k^H V_k). The product A_k B_k = G_k V_k (up to
quantisation, noise and the column phases the client removes) no longer
contains Q_k. If G_k is stable, it is a fixed projection of V_k.

Model (what is drawn, since the data hold no observer channel)
  G_k       drawn once on G_GRID frequency points and read at the points a
            report's tone groups fall on (group index scaled to the grid,
            so 234 VHT tones and 250 HE groups see the same channel).
              g1_iid    i.i.d. complex Gaussian per point
              g2_taps   L_TAPS random taps transformed to the grid (smooth)
            One draw serves every device, report and day (one static
            observer and access point). m = 1 is the first antenna of the
            m = M_MAX observer.
  stability the query side sees rho G + sqrt(1 - rho^2) G', G' independent.
              he     support windows through G, query windows through the mix
              home   every capture day d gets its own mix with a G'_d, as
                     the protocol queries other days; two days then
                     correlate with rho^2, not rho
  noise     A_k = G_k Q_k + N, N complex Gaussian at SNR_DB below the mean
            power of each observer antenna's row of G (inf = none). The
            same unit noise is scaled for every SNR, and the same G' for
            every rho, so cells differ by the parameter only.
  report    rotated, re-decomposed, quantised at the report's own codebook
            (r_rotation's emulation); the quantisation error does not cancel
            in A_k B_k. The common-grid body re-quantises at 4/2 bits and
            keeps the first 234 groups (r_rotation's A2), same Q draws.
  rotations none (A_k = G_k, B_k = the report), rot_report, rot_group.

Feature: |[A_k B_k]_{i,c}| per observer antenna i and reported column c, so
a phase per (report, group), which timing and carrier offsets put on the
sounding estimate, changes nothing.
  home   first column only (the devices report N_c = 1 or 2; the other
         invariant observers of r_rotation_space.py also read the first
         column), averaged into N_SC_BINS tone bins per frame, window mean
         and standard deviation over 32 frames (2 x m x 32 values); formats
         are not separated, so tone count and codebook stay readable.
  he     all three columns, 59 groups, no binning, standardised on the
         calibration classes through the same G (no rotation, no noise).

Protocol
  home   r_eval, other_day_any, K in {1, all}, T in {1, 32}, r_eval's seed,
         blocks capped at WINDOWS_PER_BLOCK windows as r_rotation_space.py
         (one run_condition call per K, so the K = all rows use
         r_rotation_space's draws).
  he     cohort-out cosine 1-NN, K = 1, T in {1, 32}, HE_SEEDS seeds; G, G',
         Q and the noise are drawn per seed.

Limits: G is drawn, not measured, and one draw serves the home capture;
the observer estimates A_k on every group of every sounding; G does not
change inside a day; the phase offset is assumed common to the observer's
antennas at most per (report, group), which the magnitudes do not use.

Checks stored in the JSON: (a) no rotation, no noise, rho = 1; (b) rho = 0;
(c) rot_group without quantisation against no rotation without
quantisation, feature by feature (Q cancels) and, on He, in accuracy; the
beamforming gain of the emulated rotation (he_beamforming_gain.gains).

Memory: one archive day at a time, rotation in chunks of ROT_CHUNK reports;
He in chunks of HE_CHUNK; features in float32.
Outputs (figures/): r_rotation_ndp.csv, r_rotation_ndp.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# One BLAS thread, as r_raw_observer_defenses.py: the batched small QR and
# matmul do not gain from more, and one heavy process stays one core.
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_defense_common as C  # noqa: E402
import he_beamforming_gain as HG  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_rotation as RR  # noqa: E402
import r_rotation_space as RS  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Rotation -> {body: output spec}, r_rotation.build_day's layout; the bodies
# of one rotation share its Q draws.
COMMON_BODY = "rot_group_cb42_n234"
HOME_ROTATIONS = {
    "none": {"none": RR.NATIVE_SPEC},
    "rot_report": {"rot_report": RR.NATIVE_SPEC},
    "rot_group": {"rot_group": RR.NATIVE_SPEC,
                  COMMON_BODY: RR.A2_OUTPUTS[COMMON_BODY]},
}
ROTATIONS = tuple(HOME_ROTATIONS)
ROTATION_OF = {b: r for r, o in HOME_ROTATIONS.items() for b in o}
CHECK_ROTATION = "rot_group"             # check (c) runs on its Q draws
NOQ = "_noq"                             # He body suffix: report not quantised
HE_CHECK_BODIES = {"none": "none" + NOQ, CHECK_ROTATION: CHECK_ROTATION + NOQ}
G_VARIANTS = ("g1_iid", "g2_taps")
L_TAPS = 4
G_GRID = RS.GROUPS                       # frequency points of the drawn G
M_GRID = (1, 2)
M_MAX = max(M_GRID)
SNR_DB_GRID = (math.inf, 20.0, 10.0)
RHO_GRID = (1.0, 0.9, 0.5, 0.0)
RHO_STABLE, RHO_INDEPENDENT = 1.0, 0.0
CONFIGS = [(g, s, r) for g in G_VARIANTS for s in SNR_DB_GRID for r in RHO_GRID]
NDP_SEED = 17
STREAM_G, STREAM_G_QUERY, STREAM_Q, STREAM_NOISE = 0, 1, 2, 3
NULL_SEED = RR.NULL_SEED                 # pure-noise control of this run
HOME_K_GRID = (1, 0)                     # 0 = all windows of the enrolled block
HE_SEEDS = 10
HE_EVAL_STREAM = 3                       # r_rotation_space's last seed word
CACHE_PREFIX = "ndp_"
DB_PER_DECADE = 10.0
FRAME_CHANNELS = 6                       # r_common.bin_frames' channels
WINDOW_STATS = 2                         # window mean and standard deviation
# Dry-run estimate only.
EST_NC_HOME = 2                          # widest home report
EST_HE_GROUPS = 59                       # He stride-4 groups
EST_HE_HELD_REPORTS = 18000
EST_HE_LOAD_FACTOR = 2.5                 # r_rotation_space's allowance
EST_PRODUCT_COPIES = 3                   # an array, its product, one temporary
# Reference results (read, not recomputed).
REF_MAIN = RR.OUT["main"]                # full window set, K in {1, all}
REF_SPACE = RS.OUT["json"]               # capped blocks, K = all; He, 3 seeds
REF_HE = C.FIG_DIR / "bfi_joint_defense_he_summary.json"     # He, 10 seeds
SPACE_CELL = {"none": ("none", "f1_report"),
              "rot_report": ("s4_unitary", "f1_report"),
              "rot_group": ("s4_unitary", "f2_group")}
HE_DEFENSE = {"none": "none", "rot_report": "unitary_pkt",
              "rot_group": "unitary_pkt_sc"}
REF_KEYS = ("ref_plain_angles", "ref_cross_invariant", "ref_plain_angles_space",
            "ref_cross_invariant_space", "ref_null_noise", "ref_null_noise_macro",
            "null_noise_this_run", "null_noise_this_run_macro")
OUT = dict(csv=R.FIG_DIR / "r_rotation_ndp.csv",
           json=R.FIG_DIR / "r_rotation_ndp.json")


def label(x: float) -> str:
    return f"{x:g}"


def config_key(g: str, snr: float, rho: float) -> str:
    return f"{CACHE_PREFIX}{g}_snr{label(snr)}_rho{label(rho)}"


def cell_name(body: str, m: int, g: str, snr: float, rho: float) -> str:
    return f"{body}|m{m}|{g}|snr{label(snr)}|rho{label(rho)}"


def antenna_cols(width: int, m: int) -> np.ndarray:
    """Columns of a flat feature built from M_MAX antennas, laid out
    (..., antenna, width), that belong to the first m antennas."""
    return np.arange(M_MAX * width).reshape(M_MAX, width)[:m].ravel()


# ---------------------------------------------------------------------------
# Observer channel and the product A_k B_k
# ---------------------------------------------------------------------------
def unit_complex(shape: tuple, rng) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def draw_g(variant: str, nt: int, rng) -> np.ndarray:
    """Observer channel on the grid, (G_GRID, M_MAX, nt), unit mean power."""
    if variant == "g1_iid":
        return unit_complex((G_GRID, M_MAX, nt), rng)
    if variant == "g2_taps":
        h = unit_complex((L_TAPS, M_MAX, nt), rng) / np.sqrt(L_TAPS)
        ph = np.exp(-2j * np.pi * np.outer(np.arange(G_GRID), np.arange(L_TAPS))
                    / G_GRID)
        return np.einsum("fl,lmt->fmt", ph, h)
    raise ValueError(variant)


def mix(G: np.ndarray, G2: np.ndarray, rho: float) -> np.ndarray:
    return rho * G + math.sqrt(1.0 - rho ** 2) * G2


def on_groups(G: np.ndarray, k: int) -> np.ndarray:
    """The grid channel at the k tone groups of a report, (k, M_MAX, nt)."""
    return G[X.bin_edges(k, G_GRID)]


def rotate_keep_q(V: np.ndarray, rotation: str, rng):
    """r_rotation.rotate_angles' draw, with Q returned: Q (n, k or 1, nt, nt)
    and Q^H V; (None, V) without rotation."""
    if rotation == "none":
        return None, V
    n, k, nt = V.shape[:3]
    Q = C.haar_unitary(nt, (n, k) if rotation == "rot_group" else (n, 1), rng)
    return Q, np.conj(np.swapaxes(Q, -1, -2)) @ V


def ndp_mag(Gk: np.ndarray, Q, B: np.ndarray, noise: np.ndarray,
            snr_db: float) -> np.ndarray:
    """|(G Q + N) B| -> (n, k, M_MAX, nc). Gk (k, M_MAX, nt), Q from
    rotate_keep_q, B (n, k, nt, nc), noise unit complex (n, k, M_MAX, nt)."""
    A = Gk if Q is None else Gk @ Q
    if math.isfinite(snr_db):
        power = (np.abs(Gk) ** 2).mean(axis=(0, 2))          # per antenna
        A = A + np.sqrt(power * 10 ** (-snr_db / DB_PER_DECADE))[:, None] * noise
    return np.abs(A @ B)


# ---------------------------------------------------------------------------
# Home: one pass over the archive, all bodies and channel settings
# ---------------------------------------------------------------------------
def day_channels(G: dict, nt: int, daynum: int) -> dict:
    """(variant, rho) -> the day's channel rho G + sqrt(1 - rho^2) G'_day."""
    out = {}
    for gi, g in enumerate(G_VARIANTS):
        G2 = draw_g(g, nt, np.random.default_rng(
            [NDP_SEED, STREAM_G_QUERY, gi, daynum]))
        for rho in RHO_GRID:
            out[(g, rho)] = mix(G[g], G2, rho)
    return out


def observe_home(q: np.ndarray, geom: dict, rotation: str, Gd: dict,
                 rng_q, rng_n):
    """One chunk of reports sharing a geometry through one rotation:
    {body: (rotated ints, output geometry, {config key: binned |A B| of the
    first column})} and check (c)'s largest error (0 off CHECK_ROTATION)."""
    nr, nc = geom["nr"], geom["nc"]
    V = RR.reconstruct_v(q, geom)
    Q, V2 = rotate_keep_q(V, rotation, rng_q)
    phi2, psi2 = RR.dequantise(q, geom) if Q is None \
        else C.v_to_givens(V2, nr=nr, nc=nc)
    noise = unit_complex(V.shape[:2] + (M_MAX, nr), rng_n)
    res = {}
    for body, spec in HOME_ROTATIONS[rotation].items():
        g_out = RR.output_geom(geom, spec)
        q2 = RR.quantise(phi2, psi2, g_out)
        if spec[2]:
            q2 = q2[:, :, :spec[2]]
        k = q2.shape[2]
        B = RR.reconstruct_v(q2, g_out)[..., :1]             # first column
        Qk = Q if Q is None or Q.shape[1] == 1 else Q[:, :k]
        res[body] = (q2, g_out, {
            config_key(g, snr, rho): RS.bin_tones(ndp_mag(
                on_groups(Gd[(g, rho)], k), Qk, B, noise[:, :k], snr)[..., 0])
            for g, snr, rho in CONFIGS})
        del B
    err = 0.0
    if rotation == CHECK_ROTATION:          # (c): the report not quantised
        Bc = C.givens_to_v(phi2, psi2, nr=nr, nc=nc)[..., :1]
        for g in G_VARIANTS:
            Gk = on_groups(Gd[(g, RHO_STABLE)], V.shape[1])
            err = max(err, float(np.abs(
                ndp_mag(Gk, Q, Bc, noise, math.inf)
                - ndp_mag(Gk, None, V[..., :1], noise, math.inf)).max()))
        del Bc
    del V, V2, Q, noise
    return res, err


def build_ndp_day(p: Path, devs: list[str], sess, cap: int, G: dict,
                  force: bool = False) -> tuple[int, float, bool]:
    """Per-day caches of every body (r_rotation layout plus one column per
    channel setting, r_rotation_space's block cap); returns the frames kept,
    check (c)'s largest error and whether the day was rebuilt."""
    keys = [config_key(*c) for c in CONFIGS]
    outs = {b: RR.day_file(CACHE_PREFIX + b, p.stem) for b in ROTATION_OF}
    if not force and all(o.exists() for o in outs.values()):
        zs = [np.load(o, allow_pickle=False) for o in outs.values()]
        if all(int(z["cap"]) == cap and set(keys) <= set(z.files) for z in zs):
            return (int(zs[0]["n_rotated"]),
                    max(float(z["check_c"]) for z in zs), False)
    daynum = int(p.stem.replace("bfi_", "").replace("-", ""))
    rng_q = {r: np.random.default_rng([NDP_SEED, STREAM_Q, i, daynum])
             for i, r in enumerate(ROTATIONS)}
    rng_n = {r: np.random.default_rng([NDP_SEED, STREAM_NOISE, i, daynum])
             for i, r in enumerate(ROTATIONS)}
    Gd = day_channels(G, R.NR_COMMON, daynum)
    conn = X.connect_ro(p)
    cur = conn.execute(
        f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
        f"({','.join('?' * len(devs))}) ORDER BY ts", devs)
    geoms: dict[tuple, int] = {}
    cols = {b: defaultdict(list) for b in outs}
    rank: dict[tuple, int] = defaultdict(int)
    n_keep, err = 0, 0.0
    while True:
        rows = cur.fetchmany(R.FETCH_CHUNK)
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n)
        dev = np.empty(n, np.int8)
        gid = np.full(n, -1, np.int16)
        groups: dict[tuple, list[int]] = defaultdict(list)
        gdict: dict[tuple, dict] = {}
        bodies: list = [None] * n
        for i, (t, m, bh) in enumerate(rows):
            ts[i] = t
            dev[i] = devs.index(m)
            if bh is None:
                continue
            raw = bytes.fromhex(bh)
            g, key = R._geom_from_body(raw)
            if g is None:
                continue
            gid[i] = geoms.setdefault(key, len(geoms))
            bodies[i] = raw
            gdict[key] = g
            groups[(key, len(raw))].append(i)
        decoded = {}
        ok = np.zeros(n, bool)
        for (key, _L), idx in groups.items():
            qa = RR.decode_all([bodies[i] for i in idx], gdict[key])
            if qa is None:
                continue
            decoded[(key, _L)] = (np.asarray(idx), qa)
            ok[idx] = True
        # r_rotation_space's cap: the first cap * WINDOW_FRAMES decodable
        # frames of each (device, session) on this day, in time order
        keep = np.zeros(n, bool)
        for i in np.flatnonzero(ok):
            t_all, sid_all = sess[int(dev[i])]
            sid = int(sid_all[np.searchsorted(t_all, ts[i])])
            rank[(int(dev[i]), sid)] += 1
            keep[i] = cap == 0 or rank[(int(dev[i]), sid)] <= cap * R.WINDOW_FRAMES
        n_keep += int(keep.sum())
        feat = {b: np.full((n, FRAME_CHANNELS, R.N_SC_BINS), np.nan, np.float32)
                for b in outs}
        feat_rq = {b: np.full((n, FRAME_CHANNELS, R.N_SC_BINS), np.nan, np.float32)
                   for b in outs}
        ndp = {b: {k: np.full((n, M_MAX * R.N_SC_BINS), np.nan, np.float32)
                   for k in keys} for b in outs}
        for rotation in ROTATIONS:
            for (key, _L), (idx, qa) in decoded.items():
                kk = keep[idx]
                ia, qk = idx[kk], qa[kk]
                for s in range(0, qk.shape[0], RR.ROT_CHUNK):
                    sl = slice(s, s + RR.ROT_CHUNK)
                    res, e = observe_home(qk[sl], gdict[key], rotation, Gd,
                                          rng_q[rotation], rng_n[rotation])
                    err = max(err, e)
                    for b, (q2, g_out, blocks) in res.items():
                        rad, rad_rq = RR.first_column_rads(q2, g_out)
                        feat[b][ia[sl]] = R.bin_frames(rad)
                        feat_rq[b][ia[sl]] = R.bin_frames(rad_rq)
                        for k, arr in blocks.items():
                            ndp[b][k][ia[sl]] = arr
                    del res
        for b in outs:
            for k, v in dict(ts=ts[keep], dev=dev[keep], gid=gid[keep],
                             feat=feat[b][keep], feat_rq=feat_rq[b][keep]).items():
                cols[b][k].append(v)
            for k in keys:
                cols[b][k].append(ndp[b][k][keep])
        del rows, bodies, groups, decoded, feat, feat_rq, ndp
    conn.close()
    for b, out in outs.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = {k: np.concatenate(v) if v else np.zeros(0)
               for k, v in cols[b].items()}
        np.savez(out, **arr, n_rotated=np.asarray(n_keep), cap=np.asarray(cap),
                 check_c=np.asarray(err),
                 geoms=np.asarray(json.dumps([list(k) for k in geoms])))
        del arr
    del cols
    gc.collect()
    return n_keep, err, True


def home_windows(devs: list[str], gap_s: float, body: str, force: bool) -> dict:
    p = RR.windows_path(CACHE_PREFIX + body, gap_s)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    w = RR.build_windows(devs, gap_s, CACHE_PREFIX + body,
                         inv_keys=tuple(config_key(*c) for c in CONFIGS))
    np.savez(p, **w)
    return w


def home_eval(w: dict, feat: np.ndarray, trials: int, rng_b) -> dict:
    """{K: {T: result}} of the 1-NN on one window vector; r_rotation_space's
    home_eval with one run_condition call per K."""
    tb = E.Table(dict(w, feat=feat))
    out = {}
    for K in HOME_K_GRID:
        recs = E.run_condition(tb, RS.HOME_COND, (K,), RS.T_GRID, trials,
                               E.SEED + RS.HOME_COND_INDEX)
        rows, _ = E.aggregate(recs, tb.names, {}, rng_b)
        out[E.klabel(K)] = {
            r["T"]: dict(accuracy=r["accuracy"], chance=r["chance"],
                         macro_recall=r["macro_recall"],
                         ci=[r["ci_lo"], r["ci_hi"]], n_decisions=r["n_decisions"])
            for r in rows if r["feature"] == "angles"}
        del recs
    del tb
    return out


def home_null(w: dict, trials: int, rng_b) -> dict:
    """r_rotation.run_null on this run's capped windows: {m: {K: {T: ..}}}
    for i.i.d. standard normal vectors of the observer's width."""
    rng = np.random.default_rng(NULL_SEED)
    width = WINDOW_STATS * R.N_SC_BINS
    Z = rng.standard_normal((w["dev"].size, M_MAX * width)).astype(np.float32)
    return {m: home_eval(w, Z[:, :m * width], trials, rng_b) for m in M_GRID}


def window_cols(m: int) -> np.ndarray:
    """Window vector = [mean, std], each (antenna, tone bin)."""
    one = antenna_cols(R.N_SC_BINS, m)
    return np.concatenate([one, one + M_MAX * R.N_SC_BINS])


def home_refs() -> dict:
    """Stored results of the other observers under each body."""
    main = {(r["body"], r["feature"], r["K"], int(r["T"])): r
            for r in RR.read_csv_rows(REF_MAIN) if r["condition"] == RS.HOME_COND}
    space = json.loads(REF_SPACE.read_text())["cells"]

    def get(body: str, K: str, T: int) -> dict:
        def m(b, feat, col="accuracy"):
            r = main.get((b, feat, K, T))
            return float(r[col]) if r else None

        def s(obs):
            c = space.get(RS.cell_name(*SPACE_CELL[body])) \
                if body in SPACE_CELL and K == E.klabel(0) else None
            r = (c or {}).get("home", {}).get(obs, {}).get(str(T))
            return r["accuracy"] if r else None
        return dict(ref_plain_angles=m(body, "angles"),
                    ref_cross_invariant=m(body, "invariant"),
                    ref_plain_angles_space=s("plain"),
                    ref_cross_invariant_space=s("inv_cross"),
                    ref_null_noise=m(RR.NULL_BODY, "angles"),
                    ref_null_noise_macro=m(RR.NULL_BODY, "angles", "macro_recall"))
    return get


def run_home(devs: list[str], a) -> tuple[dict, list[dict]]:
    G = {g: draw_g(g, R.NR_COMMON, np.random.default_rng([NDP_SEED, STREAM_G, gi]))
         for gi, g in enumerate(G_VARIANTS)}
    sess = RS.kept_frame_counter(devs, R.SESSION_GAP_S)
    frames, err, rebuilt = {}, 0.0, a.force
    for p in X.day_paths():
        t = time.time()
        frames[p.stem], e, new = build_ndp_day(p, devs, sess, a.cap, G, a.force)
        err, rebuilt = max(err, e), rebuilt or new
        print(f"[ndp] {p.stem}: {frames[p.stem]} frames kept "
              f"({time.time() - t:.0f}s) peak RSS {X.peak_rss_mb():.0f} MB",
              file=sys.stderr)
    del sess
    gc.collect()
    rng_b = np.random.default_rng(E.SEED + 1)
    ref = home_refs()
    cells, rows, plain, null = {}, [], {}, None
    for body, rotation in ROTATION_OF.items():
        t = time.time()
        w = home_windows(devs, R.SESSION_GAP_S, body, rebuilt)
        if null is None:                 # every body has the same windows
            null = home_null(w, a.trials, rng_b)
        plain[body] = home_eval(w, w["feat"], a.trials, rng_b)
        for g, snr, rho in CONFIGS:
            for m in M_GRID:
                res = home_eval(w, w[config_key(g, snr, rho)][:, window_cols(m)],
                                a.trials, rng_b)
                for K, byT in res.items():
                    for T, r in byT.items():
                        r.update(ref(body, K, T),
                                 null_noise_this_run=null[m][K][T]["accuracy"],
                                 null_noise_this_run_macro=null[m][K][T]["macro_recall"],
                                 plain_angles_this_run=plain[body][K][T]["accuracy"])
                        rows.append(dict(
                            dataset="home", body=body, rotation=rotation,
                            grid="common" if body == COMMON_BODY else "own",
                            m=m, g=g, snr_db=snr, rho=rho, K=K, T=T,
                            accuracy=r["accuracy"], sd=None,
                            macro_recall=r["macro_recall"], chance=r["chance"],
                            ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                            n=r["n_decisions"], **{k: r[k] for k in REF_KEYS}))
                cells[cell_name(body, m, g, snr, rho)] = res
        print(f"[home] {body}: {w['dev'].size} windows ({time.time() - t:.0f}s) "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
        del w
        gc.collect()
    return dict(condition=RS.HOME_COND, k_grid=[E.klabel(k) for k in HOME_K_GRID],
                t_grid=list(RS.T_GRID), trials=a.trials,
                windows_per_block_cap=a.cap, frames_kept_per_day=frames,
                enrol_query_correlation={label(r): r ** 2 for r in RHO_GRID},
                plain_angles_this_run=plain, null_noise_this_run=null,
                check_c_max_abs_err=err, cells=cells), rows


# ---------------------------------------------------------------------------
# He CR1
# ---------------------------------------------------------------------------
def he_scalers(he: RS.HeSpace, G: dict) -> dict:
    """Per G variant: standardisation of |G V| on the calibration classes
    (quantised report, no rotation, no noise), fitted in chunks."""
    from sklearn.preprocessing import StandardScaler
    S = {g: StandardScaler() for g in G}
    phi, psi = he.phi["cal"], he.psi["cal"]
    for s in range(0, phi.shape[0], RS.HE_CHUNK):
        B = C.givens_to_v(*C.he_quantise(phi[s:s + RS.HE_CHUNK],
                                         psi[s:s + RS.HE_CHUNK]))
        for g in G:
            S[g].partial_fit(ndp_mag(on_groups(G[g], B.shape[1]), None, B, None,
                                     math.inf).reshape(B.shape[0], -1))
        del B
    return S


def he_seed(he: RS.HeSpace, seed: int, acc: dict) -> dict:
    """One seed: every body and channel setting, accuracies appended to acc
    {(body, m, g, snr, rho): {T: [..]}}; returns the seed's checks."""
    phi, psi, y = he.phi["held"], he.psi["held"], he.y["held"]
    supports = C.pick_supports(y, seed)
    sup = np.array(sorted(supports.values()))
    G = {g: draw_g(g, C.HE_NR, np.random.default_rng([NDP_SEED, STREAM_G, gi, seed]))
         for gi, g in enumerate(G_VARIANTS)}
    Gq = {(g, rho): mix(G[g], draw_g(g, C.HE_NR, np.random.default_rng(
        [NDP_SEED, STREAM_G_QUERY, gi, seed])), rho)
        for gi, g in enumerate(G_VARIANTS) for rho in RHO_GRID}
    S = he_scalers(he, G)
    k = phi.shape[1]
    cols = {m: np.arange(k * M_MAX * C.HE_NC).reshape(k, M_MAX, C.HE_NC)[:, :m].ravel()
            for m in M_GRID}
    checks = dict(check_c_max_abs_err=0.0)

    def observe(g, snr, rho, Q, B, noise, rows):
        """Standardised |A B| of a chunk; its support rows through G."""
        mag = ndp_mag(on_groups(Gq[(g, rho)], k), Q, B, noise, snr)
        if rows.size:
            mag[rows] = ndp_mag(on_groups(G[g], k), None if Q is None else Q[rows],
                                B[rows], noise[rows], snr)
        return S[g].transform(mag.reshape(mag.shape[0], -1)).astype(np.float32)

    for ri, rotation in enumerate(ROTATIONS):
        rng_q = np.random.default_rng([NDP_SEED, STREAM_Q, ri, seed])
        rng_n = np.random.default_rng([NDP_SEED, STREAM_NOISE, ri, seed])
        feats = defaultdict(list)
        for s in range(0, phi.shape[0], RS.HE_CHUNK):
            sl = slice(s, s + RS.HE_CHUNK)
            V = C.givens_to_v(phi[sl], psi[sl])
            Q, V2 = rotate_keep_q(V, rotation, rng_q)
            p2, q2 = (phi[sl], psi[sl]) if Q is None else C.v_to_givens(V2)
            B = C.givens_to_v(*C.he_quantise(p2, q2))
            noise = unit_complex(V.shape[:2] + (M_MAX, C.HE_NR), rng_n)
            rows = sup[(sup >= s) & (sup < s + RS.HE_CHUNK)] - s
            for g, snr, rho in CONFIGS:
                feats[(rotation, g, snr, rho)].append(
                    observe(g, snr, rho, Q, B, noise, rows))
            if rotation in HE_CHECK_BODIES:          # (c): no quantisation
                Bc = C.givens_to_v(p2, q2)
                for g in G_VARIANTS:
                    feats[(HE_CHECK_BODIES[rotation], g, math.inf, RHO_STABLE)
                          ].append(observe(g, math.inf, RHO_STABLE, Q, Bc, noise, rows))
                    if Q is not None:
                        Gk = on_groups(G[g], k)
                        checks["check_c_max_abs_err"] = max(
                            checks["check_c_max_abs_err"], float(np.abs(
                                ndp_mag(Gk, Q, Bc, noise, math.inf)
                                - ndp_mag(Gk, None, V, noise, math.inf)).max()))
                del Bc
            if rotation == CHECK_ROTATION and s == 0:
                # the access point undoing Q on the quantised report
                g1, _ = HG.gains(Q @ B, V)
                checks["gain_col1"] = float(g1.mean())
            del V, V2, Q, B, noise
        for (body, g, snr, rho), parts in feats.items():
            Z = np.concatenate(parts)
            ci = CONFIGS.index((g, snr, rho))
            for m in M_GRID:
                r = C.multi_query_eval(
                    Z[:, cols[m]], y, supports, metric="cos", T_grid=RS.T_GRID,
                    n_decisions=RS.HE_N_DECISIONS,
                    rng=np.random.default_rng([seed, ci, m, HE_EVAL_STREAM]))
                for T, v in r.items():
                    acc[(body, m, g, snr, rho)][T].append(v)
            del Z
        del feats
        gc.collect()
    return checks


def he_refs() -> dict:
    cells = json.loads(REF_HE.read_text())["cells"]
    space = json.loads(REF_SPACE.read_text())["cells"]

    def get(body: str, T: int) -> dict:
        out = dict.fromkeys(REF_KEYS)
        if body not in HE_DEFENSE:
            return out
        c = space.get(RS.cell_name(*SPACE_CELL[body]), {}).get("he") or {}
        for key, attacker, obs in (("ref_plain_angles", "nn1", "plain"),
                                   ("ref_cross_invariant", "nn1_inv", "inv_cross")):
            r = cells.get(f"{HE_DEFENSE[body]}|{attacker}|T={T}")
            out[key] = r["mean"] if r else None
            r = c.get(obs, {}).get(str(T))
            out[key + "_space"] = r["accuracy"] if r else None
        return out
    return get


def run_he(seeds: int) -> tuple[dict, list[dict]]:
    he = RS.HeSpace()
    X.log_rss("He loaded")
    acc = defaultdict(lambda: defaultdict(list))
    checks = []
    for seed in range(1, seeds + 1):
        t = time.time()
        checks.append(he_seed(he, seed, acc))
        print(f"[he] seed {seed} ({time.time() - t:.0f}s) "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    chance = 1.0 / len(np.unique(he.y["held"]))
    ref = he_refs()
    cells, rows = {}, []
    for (body, m, g, snr, rho), byT in acc.items():
        res = {T: dict(accuracy=float(np.mean(v)), sd=float(np.std(v)),
                       chance=chance, n_seeds=len(v), **ref(body, T))
               for T, v in byT.items()}
        cells[cell_name(body, m, g, snr, rho)] = res
        for T, r in res.items():
            rows.append(dict(
                dataset="he", body=body, rotation=body.replace(NOQ, ""),
                grid="own", m=m, g=g, snr_db=snr, rho=rho, K="1", T=T,
                accuracy=r["accuracy"], sd=r["sd"], macro_recall=None,
                chance=chance, ci_lo=None, ci_hi=None, n=r["n_seeds"],
                **{k: r[k] for k in REF_KEYS}))
    noq = {f"m{m}|{g}|T={T}": [
        acc[(b, m, g, math.inf, RHO_STABLE)][T] for b in HE_CHECK_BODIES.values()]
        for m in M_GRID for g in G_VARIANTS for T in RS.T_GRID}
    space = json.loads(REF_SPACE.read_text())["cells"]
    return dict(seeds=seeds, t_grid=list(RS.T_GRID), chance=chance,
                n_decisions_per_class=RS.HE_N_DECISIONS,
                check_c_max_abs_err=max(c["check_c_max_abs_err"] for c in checks),
                check_c_accuracy_max_abs_diff=max(
                    float(np.abs(np.subtract(*v)).max()) for v in noq.values()),
                gain_col1_rot_group=float(np.mean([c["gain_col1"] for c in checks])),
                gain_col1_ref_space=space[RS.cell_name(*SPACE_CELL[CHECK_ROTATION])
                                          ]["gain"]["gain_col1"],
                cells=cells), rows


# ---------------------------------------------------------------------------
# Checks (a) and (b) from the written rows
# ---------------------------------------------------------------------------
def summarise_checks(rows: list[dict]) -> dict:
    """(a) no rotation, no noise, stable G; (b) independent G, per rotation
    body: accuracy range over m, G variant, SNR (and K on the home capture)."""
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for T in RS.T_GRID:
            sel = [r for r in rows if r["dataset"] == ds and r["T"] == T
                   and not r["body"].endswith(NOQ)]
            a = [r["accuracy"] for r in sel if r["body"] == "none"
                 and r["rho"] == RHO_STABLE and math.isinf(r["snr_db"])]
            out[f"a|{ds}|T={T}"] = dict(min=min(a), max=max(a),
                                        chance=sel[0]["chance"])
            for body in sorted({r["body"] for r in sel}):
                b = [r["accuracy"] for r in sel if r["body"] == body
                     and r["rho"] == RHO_INDEPENDENT]
                out[f"b|{ds}|{body}|T={T}"] = dict(
                    min=min(b), max=max(b), mean=float(np.mean(b)),
                    chance=sel[0]["chance"])
    return out


def dry_run(cap: int) -> dict:
    """Memory of the largest arrays, without loading anything."""
    c16 = np.dtype(np.complex128).itemsize
    f4 = np.dtype(np.float32).itemsize

    def chunk(n, k, nt, nc):
        # V, Q^H V and B; Q; noise, A and A B
        return n * k * c16 * (EST_PRODUCT_COPIES * nt * nc + nt * nt
                              + EST_PRODUCT_COPIES * M_MAX * nt)
    home_frames = R.FETCH_CHUNK * len(ROTATION_OF) * f4 * R.N_SC_BINS * (
        len(CONFIGS) * M_MAX + 2 * FRAME_CHANNELS)
    he_feats = (EST_HE_HELD_REPORTS * (len(CONFIGS) + len(G_VARIANTS))
                * EST_HE_GROUPS * M_MAX * C.HE_NC * f4)
    return dict(bodies=list(ROTATION_OF), configs=len(CONFIGS), m=list(M_GRID),
                cap=cap,
                home_rotation_chunk_mb=chunk(RR.ROT_CHUNK, G_GRID, R.NR_COMMON,
                                             EST_NC_HOME) / 1e6,
                home_fetch_chunk_features_mb=home_frames / 1e6,
                he_load_mb=C.he_estimate_mb() * EST_HE_LOAD_FACTOR,
                he_rotation_chunk_mb=chunk(RS.HE_CHUNK, EST_HE_GROUPS, C.HE_NR,
                                           C.HE_NC) / 1e6,
                he_features_one_rotation_mb=he_feats / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--cap", type=int, default=RS.WINDOWS_PER_BLOCK,
                    help="windows kept per (device, session, day) block; 0 = all")
    ap.add_argument("--he-seeds", type=int, default=HE_SEEDS)
    ap.add_argument("--no-he", action="store_true")
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild this script's per-day caches")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate the memory only")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    if a.dry_run:
        print(json.dumps(dry_run(a.cap), indent=1))
        return 0
    home = he = None
    rows: list[dict] = []
    if not a.no_home:
        devs = R.device_macs()
        for p in X.day_paths():
            R.build_day(p, devs)
        R.get_windows(devs, R.SESSION_GAP_S)
        home, r = run_home(devs, a)
        rows += r
        X.log_rss("home done")
    if not a.no_he:
        he, r = run_he(a.he_seeds)
        rows += r
        X.log_rss("He done")
    E.write_csv(OUT["csv"], rows)
    X.write_json(OUT["json"], dict(
        model=dict(g_variants=list(G_VARIANTS), l_taps=L_TAPS, g_grid=G_GRID,
                   m=list(M_GRID), snr_db=[label(s) for s in SNR_DB_GRID],
                   rho=list(RHO_GRID), seed=NDP_SEED, bodies=ROTATION_OF,
                   feature="|A_k B_k| per observer antenna and reported column",
                   common_grid=dict(body=COMMON_BODY,
                                    spec=list(RR.A2_OUTPUTS[COMMON_BODY]))),
        references=dict(home_main=REF_MAIN.name, space=REF_SPACE.name,
                        he=REF_HE.name,
                        note="ref_plain_angles and ref_cross_invariant: full "
                             "window set (home) and 10 seeds (He); *_space: "
                             "r_rotation_space, capped blocks, K = all (home) "
                             "and 3 seeds (He); ref_null_noise: body "
                             f"{RR.NULL_BODY}, full window set"),
        home=home, he=he, checks=summarise_checks(rows),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
