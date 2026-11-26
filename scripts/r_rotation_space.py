#!/usr/bin/env python3
"""R-space: the design space of the access point's sounding randomization,
scored by the paper's test, to find the cheapest cell that passes.

Cells = structure of Q_k (N_t x N_t) x freshness.
  Structures   s1_diag     random diagonal of unit phases
               s2_perm     random permutation times random diagonal phases
               s3_orth     real Haar orthogonal
               s4_unitary  complex Haar unitary (LeakyBeam's, Section 7.2)
               none        no rotation (reference)
  Freshness    f1_report   one Q per report, the same on all tone groups
               f2_group    one Q per report and per tone group
               f3_bandM    one Q per report and per band of M consecutive
                           groups, M in {2, 4, 8}
The client reports Q_k^H V_k re-decomposed into Givens angles and quantised
at the report's codebook (r_rotation.py emulation, generic in N_t, N_c).

Observers per cell
  plain        the paper's angle observer (r_common / joint_defense features)
  inv_<block>  what the structure leaves readable, computed from the reported
               angles: mag (|v_i| per group, s1), sortmag (sorted |v_i| per
               group, s2), tsym (|v^T v| per group, s3: an orthogonal Q
               preserves v^T v up to the column phase), cross (|v_k^H v_j|
               between groups that share a Q: f1 and f3); inv_combined when
               two blocks apply. s4 x f2 leaves no invariant.
  gain         first-stream beamforming gain on He CR1 with the access point
               undoing Q (he_beamforming_gain.gains), against the continuous V.

Data
  home   the six residential devices, r_eval protocol, other_day_any,
         K = all windows of the enrolled block, T in {1, 32}, 50 draws,
         r_eval's seed for that condition. To fit 21 cells in one pass every
         (device, session, day) block is capped at its first
         WINDOWS_PER_BLOCK windows (the 'none' cell carries the same cap, and
         r_rotation.py has the two unitary cells on the full set).
  he     He CR1 positions, joint_defense_eval's cohort-out 1-NN (K = 1,
         cosine, clean calibration statistics), T in {1, 32}, HE_SEEDS seeds.

Pass = every observer on both datasets within PASS_MARGIN of chance at both
T, and gain >= GAIN_MIN. Cost per sounding for GROUPS groups at N_T antennas:
random numbers drawn, complex values stored, whether a QR is needed.

Memory: one archive day at a time, decoded once, rotated per cell in chunks;
He in chunks of HE_CHUNK reports; invariant features in float32.
Outputs (figures/): r_rotation_space.csv, r_rotation_space.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_defense_common as C  # noqa: E402
import he_beamforming_gain as HG  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_rotation as RR  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRUCTURES = ("none", "s1_diag", "s2_perm", "s3_orth", "s4_unitary")
FRESHNESS = {"f1_report": 0, "f2_group": 1, "f3_band2": 2, "f3_band4": 4,
             "f3_band8": 8}                 # 0 = one Q for all groups
CELLS = [("none", "f1_report")] + [(s, f) for s in STRUCTURES[1:]
                                   for f in FRESHNESS]
BLOCKS = ("mag", "sortmag", "tsym", "cross")
# Width of each home block after tone binning (first column, Nr = 3).
HOME_BLOCK_WIDTH = {"mag": R.NR_COMMON * R.N_SC_BINS,
                    "sortmag": R.NR_COMMON * R.N_SC_BINS,
                    "tsym": R.N_SC_BINS, "cross": R.N_SC_BINS}
HOME_COND = "other_day_any"
HOME_COND_INDEX = E.CONDITIONS.index(HOME_COND)      # r_eval's seed offset
HOME_K_GRID = (0,)
T_GRID = (1, 32)
WINDOWS_PER_BLOCK = 32
HE_SEEDS = 3
HE_CHUNK = 3000
HE_N_DECISIONS = 200
GAIN_REPORTS = 3000
GAIN_SEED = 1
SPACE_SEED = 11
PASS_MARGIN = 0.01
GAIN_MIN = 0.95
N_T = 4
GROUPS = 250
OUT = dict(csv=R.FIG_DIR / "r_rotation_space.csv",
           json=R.FIG_DIR / "r_rotation_space.json")


def cell_name(s: str, f: str) -> str:
    return f"{s}|{f}"


def variant_of(s: str, f: str) -> str:
    return f"space_{s}_{f}"


def blocks_for(s: str, f: str) -> list[str]:
    if s == "none":
        return list(BLOCKS)
    b = {"s1_diag": ["mag"], "s2_perm": ["sortmag"], "s3_orth": ["tsym"],
         "s4_unitary": []}[s]
    if FRESHNESS[f] != 1:
        b.append("cross")
    return b


def band_m(f: str, k: int) -> int:
    m = FRESHNESS[f]
    return k if m == 0 else m


# ---------------------------------------------------------------------------
# Q structures and freshness
# ---------------------------------------------------------------------------
def draw_q(struct: str, n: int, size: tuple, rng) -> np.ndarray:
    if struct == "s1_diag":
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size + (n,)))
        return np.eye(n) * ph[..., None, :]
    if struct == "s2_perm":
        perm = np.argsort(rng.random(size + (n,)), axis=-1)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size + (n,)))
        return np.eye(n)[perm] * ph[..., None, :]
    if struct == "s3_orth":
        return C.haar_orthogonal(n, size, rng).astype(np.complex128)
    if struct == "s4_unitary":
        return C.haar_unitary(n, size, rng)
    raise ValueError(struct)


def rotate_v(V: np.ndarray, struct: str, f: str, rng) -> np.ndarray:
    """V (n, k, nr, nc) -> Q^H V with the cell's structure and freshness."""
    n, k, nr = V.shape[:3]
    m = band_m(f, k)
    nb = math.ceil(k / m)
    Q = draw_q(struct, nr, (n, nb), rng)          # (n, nb, nr, nr)
    if nb > 1:
        Q = np.repeat(Q, m, axis=1)[:, :k]        # one Q per band of m groups
    return np.conj(np.swapaxes(Q, -1, -2)) @ V    # nb == 1 broadcasts


def rotate_q(q: np.ndarray, geom: dict, struct: str, f: str, rng):
    """Home reports: ints -> rotated, re-decomposed, re-quantised ints."""
    if struct == "none":
        return q
    nr, nc = geom["nr"], geom["nc"]
    out = np.empty_like(q)
    for s in range(0, q.shape[0], RR.ROT_CHUNK):
        sl = slice(s, s + RR.ROT_CHUNK)
        V2 = rotate_v(RR.reconstruct_v(q[sl], geom), struct, f, rng)
        phi2, psi2 = C.v_to_givens(V2, nr=nr, nc=nc)
        out[sl] = RR.quantise(phi2, psi2, geom)
        del V2, phi2, psi2
    return out


# ---------------------------------------------------------------------------
# Invariant blocks
# ---------------------------------------------------------------------------
def cross_pairs(k: int, m: int):
    """Reference group of every group under bands of m, and the mask of the
    groups that are not their band's reference."""
    ref = (np.arange(k) // m) * m
    return ref, np.arange(k) != ref


def bin_tones(arr: np.ndarray) -> np.ndarray:
    """(n, k', c) -> (n, c * N_SC_BINS) float32: equal-width tone bins."""
    n, kk, c = arr.shape
    b = X.bin_edges(kk, R.N_SC_BINS)
    M = np.zeros((kk, R.N_SC_BINS))
    M[np.arange(kk), b] = 1.0
    M /= M.sum(axis=0, keepdims=True)
    return np.einsum("nkc,kb->ncb", arr, M).reshape(n, -1).astype(np.float32)


def home_blocks(q: np.ndarray, geom: dict, f: str, names) -> dict:
    """Invariant blocks of the first column, from the reported angles."""
    v = RR.reconstruct_v(q, geom)[..., 0]                # (n, k, nr)
    k = v.shape[1]
    out = {}
    a = np.abs(v)
    if "mag" in names:
        out["mag"] = bin_tones(a)
    if "sortmag" in names:
        out["sortmag"] = bin_tones(np.sort(a, axis=-1))
    if "tsym" in names:
        out["tsym"] = bin_tones(np.abs(np.einsum("nkr,nkr->nk", v, v))[..., None])
    if "cross" in names:
        ref, mask = cross_pairs(k, band_m(f, k))
        g = np.abs(np.einsum("nkr,nkr->nk", np.conj(v), v[:, ref]))
        out["cross"] = bin_tones(g[:, mask][..., None])
    return out


def he_blocks(V: np.ndarray, f: str, names) -> dict:
    """He: V (n, k, 4, 3) -> flat float32 blocks (no tone binning)."""
    n, k = V.shape[:2]
    out = {}
    a = np.abs(V)
    if "mag" in names:
        out["mag"] = a.reshape(n, -1).astype(np.float32)
    if "sortmag" in names:
        out["sortmag"] = np.sort(a, axis=-2).reshape(n, -1).astype(np.float32)
    if "tsym" in names:
        G = np.abs(np.swapaxes(V, -1, -2) @ V)                 # (n, k, 3, 3)
        iu = np.triu_indices(V.shape[-1])
        out["tsym"] = G[..., iu[0], iu[1]].reshape(n, -1).astype(np.float32)
    if "cross" in names:
        ref, mask = cross_pairs(k, band_m(f, k))
        g = np.abs(np.einsum("nkrc,nkrc->nkc", np.conj(V), V[:, ref]))
        out["cross"] = g[:, mask].reshape(n, -1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Home: one pass over the archive, all cells
# ---------------------------------------------------------------------------
def kept_frame_counter(devs: list[str], gap_s: float):
    """Device index -> (sorted timestamps, session id per timestamp)."""
    ts_all = R.all_timestamps(devs)
    return {i: (ts_all[m], R.cut_sessions(ts_all[m], gap_s))
            for i, m in enumerate(devs)}


def build_space_day(p: Path, devs: list[str], cells, sess, cap: int,
                    force: bool = False) -> int:
    """Per-day caches of every cell (r_rotation layout plus inv_<block>);
    returns the number of frames kept."""
    outs = {c: RR.variant_dir(variant_of(*c)) / f"{p.stem}.r{R.R_CACHE_VERSION}.npz"
            for c in cells}
    todo = [c for c in cells if force or not outs[c].exists()]
    if not todo:
        z = np.load(outs[cells[0]], allow_pickle=False)
        return int(z["ts"].size) if "ts" in z.files else 0
    daynum = int(p.stem.replace("bfi_", "").replace("-", ""))
    rngs = {c: np.random.default_rng([SPACE_SEED, CELLS.index(c), daynum])
            for c in todo}
    conn = X.connect_ro(p)
    cur = conn.execute(
        f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
        f"({','.join('?' * len(devs))}) ORDER BY ts", devs)
    geoms: dict[tuple, int] = {}
    cols = {c: defaultdict(list) for c in todo}
    rank: dict[tuple, int] = defaultdict(int)
    n_keep = 0
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
        # cap: the first cap * WINDOW_FRAMES decodable frames of each
        # (device, session) on this day, in time order (r_common's cut)
        keep = np.zeros(n, bool)
        for i in np.flatnonzero(ok):
            t_all, sid_all = sess[int(dev[i])]
            sid = int(sid_all[np.searchsorted(t_all, ts[i])])
            rank[(int(dev[i]), sid)] += 1
            keep[i] = cap == 0 or rank[(int(dev[i]), sid)] <= cap * R.WINDOW_FRAMES
        n_keep += int(keep.sum())
        for c in todo:
            s_, f_ = c
            names = blocks_for(s_, f_)
            feat = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
            feat_rq = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
            inv = {b: np.full((n, HOME_BLOCK_WIDTH[b]), np.nan, np.float32)
                   for b in names}
            for (key, _L), (idx, qa) in decoded.items():
                kk = keep[idx]
                if not kk.any():
                    continue
                ia, qk = idx[kk], qa[kk]
                g = gdict[key]
                q2 = rotate_q(qk, g, s_, f_, rngs[c])
                rad, rad_rq = RR.first_column_rads(q2, g)
                feat[ia] = R.bin_frames(rad)
                feat_rq[ia] = R.bin_frames(rad_rq)
                for b, arr in home_blocks(q2, g, f_, names).items():
                    inv[b][ia] = arr
                del q2, rad, rad_rq
            cols[c]["ts"].append(ts[keep])
            cols[c]["dev"].append(dev[keep])
            cols[c]["gid"].append(gid[keep])
            cols[c]["feat"].append(feat[keep])
            cols[c]["feat_rq"].append(feat_rq[keep])
            for b in names:
                cols[c][f"inv_{b}"].append(inv[b][keep])
            del feat, feat_rq, inv
        del rows, bodies, groups, decoded
    conn.close()
    for c in todo:
        outs[c].parent.mkdir(parents=True, exist_ok=True)
        arr = {k: np.concatenate(v) if v else np.zeros(0)
               for k, v in cols[c].items()}
        np.savez(outs[c], **arr, n_rotated=np.asarray(n_keep),
                 geoms=np.asarray(json.dumps([list(k) for k in geoms])))
        del arr
    del cols
    gc.collect()
    return n_keep


def home_windows(devs, gap_s, cell, force=False) -> dict:
    v = variant_of(*cell)
    p = RR.windows_path(v, gap_s)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    keys = tuple(f"inv_{b}" for b in blocks_for(*cell))
    w = RR.build_windows(devs, gap_s, v, inv_keys=keys)
    np.savez(p, **w)
    return w


def home_eval(w: dict, feat: np.ndarray, trials: int, rng_b) -> dict:
    tb = E.Table(dict(w, feat=feat))
    recs = E.run_condition(tb, HOME_COND, HOME_K_GRID, T_GRID, trials,
                           E.SEED + HOME_COND_INDEX)
    rows, _ = E.aggregate(recs, tb.names, {}, rng_b)
    del recs, tb
    return {r["T"]: dict(accuracy=r["accuracy"], chance=r["chance"],
                         macro_recall=r["macro_recall"], ci=[r["ci_lo"], r["ci_hi"]],
                         n_decisions=r["n_decisions"])
            for r in rows if r["feature"] == "angles"}


def home_observers(w: dict, cell, trials: int, rng_b) -> dict:
    names = blocks_for(*cell)
    res = {"plain": home_eval(w, w["feat"], trials, rng_b)}
    for b in names:
        res[f"inv_{b}"] = home_eval(w, w[f"inv_{b}"], trials, rng_b)
    if len(names) > 1:
        res["inv_combined"] = home_eval(
            w, np.concatenate([w[f"inv_{b}"] for b in names], axis=1),
            trials, rng_b)
    return res


# ---------------------------------------------------------------------------
# He CR1
# ---------------------------------------------------------------------------
class HeSpace:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        d = C.load_he_angles()
        y = d["y"]
        cal = np.isin(y, np.unique(y)[:C.HE_N_CALIB_CLASSES])
        self.phi = {"cal": d["phi"][cal], "held": d["phi"][~cal]}
        self.psi = {"cal": d["psi"][cal], "held": d["psi"][~cal]}
        self.y = {"cal": y[cal], "held": y[~cal]}
        self.phi_all, self.psi_all = d["phi"], d["psi"]
        del d
        gc.collect()
        pq, sq = C.he_quantise(self.phi["cal"], self.psi["cal"])
        self.S0 = StandardScaler().fit(C.he_features(pq, sq))
        self.S_inv = {}
        blocks = defaultdict(list)
        for s in range(0, pq.shape[0], HE_CHUNK):
            V = C.givens_to_v(pq[s:s + HE_CHUNK], sq[s:s + HE_CHUNK])
            for f in FRESHNESS:
                for b, arr in he_blocks(V, f, BLOCKS if f != "f2_group"
                                        else [x for x in BLOCKS if x != "cross"]
                                        ).items():
                    blocks[(f, b)].append(arr)
            del V
        for key, parts in blocks.items():
            self.S_inv[key] = StandardScaler().fit(np.concatenate(parts))
        del blocks
        gc.collect()

    def observe(self, cell, rng) -> tuple[np.ndarray, dict]:
        """Held set through the cell: standardised plain features and
        standardised invariant blocks (cosine 1-NN reads both)."""
        s_, f_ = cell
        names = blocks_for(s_, f_)
        phi, psi = self.phi["held"], self.psi["held"]
        Fs, inv = [], defaultdict(list)
        for s in range(0, phi.shape[0], HE_CHUNK):
            sl = slice(s, s + HE_CHUNK)
            if s_ == "none":
                p2, q2 = phi[sl], psi[sl]
            else:
                V2 = rotate_v(C.givens_to_v(phi[sl], psi[sl]), s_, f_, rng)
                p2, q2 = C.v_to_givens(V2)
                del V2
            pq, sq = C.he_quantise(p2, q2)
            Fs.append(self.S0.transform(C.he_features(pq, sq)).astype(np.float32))
            Vq = C.givens_to_v(pq, sq)
            for b, arr in he_blocks(Vq, f_, names).items():
                inv[b].append(self.S_inv[(f_, b)].transform(arr).astype(np.float32))
            del Vq
        return np.concatenate(Fs), {b: np.concatenate(v) for b, v in inv.items()}

    def gain(self, cell, rng) -> dict:
        """First-stream gain with the access point undoing Q, GAIN_REPORTS
        reports drawn as he_beamforming_gain does."""
        s_, f_ = cell
        idx = np.sort(rng.choice(len(self.phi_all), size=GAIN_REPORTS, replace=False))
        V = C.givens_to_v(self.phi_all[idx], self.psi_all[idx])
        n, k, nr = V.shape[:3]
        if s_ == "none":
            pq, sq = C.he_quantise(self.phi_all[idx], self.psi_all[idx])
            Vh = C.givens_to_v(pq, sq)
        else:
            m = band_m(f_, k)
            nb = math.ceil(k / m)
            Q = draw_q(s_, nr, (n, nb), rng)
            if nb > 1:
                Q = np.repeat(Q, m, axis=1)[:, :k]
            p2, q2 = C.v_to_givens(np.conj(np.swapaxes(Q, -1, -2)) @ V)
            pq, sq = C.he_quantise(p2, q2)
            Vh = Q @ C.givens_to_v(pq, sq)
        g1, gs = HG.gains(Vh, V)
        return dict(gain_col1=float(g1.mean()), gain_col1_p05=float(np.percentile(g1, 5)),
                    gain_sub=float(gs.mean()))


def he_observers(he: HeSpace, cell, seeds: int) -> dict:
    y = he.y["held"]
    acc = defaultdict(lambda: defaultdict(list))
    for seed in range(1, seeds + 1):
        supports = C.pick_supports(y, seed)
        ci = CELLS.index(cell)
        Xs, inv = he.observe(cell, np.random.default_rng([seed, ci, 1]))
        reps = {"plain": Xs}
        reps.update({f"inv_{b}": v for b, v in inv.items()})
        if len(inv) > 1:
            reps["inv_combined"] = np.concatenate(list(inv.values()), axis=1)
        for ai, (name, Z) in enumerate(reps.items()):
            r = C.multi_query_eval(Z, y, supports, metric="cos", T_grid=T_GRID,
                                   n_decisions=HE_N_DECISIONS,
                                   rng=np.random.default_rng([seed, ci, ai, 3]))
            for T, a in r.items():
                acc[name][T].append(a)
        del Xs, inv, reps
        gc.collect()
    chance = 1.0 / len(np.unique(y))
    return {name: {T: dict(accuracy=float(np.mean(v)), sd=float(np.std(v)),
                           chance=chance) for T, v in d.items()}
            for name, d in acc.items()}


# ---------------------------------------------------------------------------
# Cost and verdict
# ---------------------------------------------------------------------------
def cost(s: str, f: str) -> dict:
    nq = 0 if s == "none" else (1 if FRESHNESS[f] == 0 else
                                math.ceil(GROUPS / FRESHNESS[f]))
    per = {"none": (0, 0, 0, False),
           "s1_diag": (N_T, N_T, 0, False),
           "s2_perm": (2 * N_T, N_T, N_T, False),
           "s3_orth": (N_T ** 2, N_T ** 2 / 2, 0, True),
           "s4_unitary": (2 * N_T ** 2, N_T ** 2, 0, True)}[s]
    return dict(q_matrices=nq, random_numbers=nq * per[0],
                complex_values_stored=nq * per[1], indices_stored=nq * per[2],
                needs_qr=per[3])


def verdict(home: dict, he: dict | None, gain: dict | None) -> dict:
    """Pass per dataset and overall: every observer within PASS_MARGIN of
    chance at both T, and gain >= GAIN_MIN. The breaking observers are
    sorted by excess over chance; the first is what breaks the cell."""
    breaks = []
    for ds, res in (("home", home), ("he", he or {})):
        for obs, byT in res.items():
            for T, r in byT.items():
                ex = r["accuracy"] - r["chance"]
                if ex > PASS_MARGIN:
                    breaks.append(dict(dataset=ds, observer=obs, T=int(T),
                                       accuracy=round(r["accuracy"], 4),
                                       chance=round(r["chance"], 4),
                                       excess=round(ex, 4)))
    breaks.sort(key=lambda b: -b["excess"])
    gain_ok = gain is None or gain["gain_col1"] >= GAIN_MIN
    per = {ds: gain_ok and not any(b["dataset"] == ds for b in breaks)
           for ds in ("home", "he")}
    first = {ds: next((b for b in breaks if b["dataset"] == ds), None)
             for ds in ("home", "he")}
    return dict(passes=not breaks and gain_ok, passes_home=per["home"],
                passes_he=per["he"], gain_ok=gain_ok,
                breaking_observers=breaks,
                first_break=(breaks[0] if breaks else None),
                first_break_home=first["home"], first_break_he=first["he"])


def cheapest(cells: dict, key: str) -> str | None:
    ok = [k for k, c in cells.items() if c["verdict"][key]]
    return min(ok, key=lambda k: (cells[k]["cost"]["random_numbers"],
                                 cells[k]["cost"]["complex_values_stored"])) \
        if ok else None


def reverdict(path: Path) -> None:
    """Recompute the verdicts of a written JSON from its stored numbers."""
    d = json.loads(path.read_text())
    for c in d["cells"].values():
        c["verdict"] = verdict(c["home"], c["he"], c["gain"])
    d["passing"] = [k for k, c in d["cells"].items() if c["verdict"]["passes"]]
    d["passing_he"] = [k for k, c in d["cells"].items() if c["verdict"]["passes_he"]]
    d["passing_home"] = [k for k, c in d["cells"].items()
                         if c["verdict"]["passes_home"]]
    d["cheapest_passing"] = cheapest(d["cells"], "passes")
    d["cheapest_passing_he"] = cheapest(d["cells"], "passes_he")
    d["cheapest_passing_home"] = cheapest(d["cells"], "passes_home")
    X.write_json(path, d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--cap", type=int, default=WINDOWS_PER_BLOCK,
                    help="windows kept per (device, session, day) block; 0 = all")
    ap.add_argument("--he-seeds", type=int, default=HE_SEEDS)
    ap.add_argument("--no-he", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverdict", action="store_true",
                    help="recompute the verdicts of the written JSON only")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    if a.reverdict:
        reverdict(OUT["json"])
        return 0
    if a.dry_run:
        est = dict(cells=len(CELLS), per_chunk_mb=RR.ROT_CHUNK * 250 * 15 * 16 * 3 / 1e6,
                   he_mb=C.he_estimate_mb() * 2.5, cap=a.cap)
        print(json.dumps(est, indent=1))
        return 0
    devs = R.device_macs()
    for p in X.day_paths():
        R.build_day(p, devs)
    R.get_windows(devs, R.SESSION_GAP_S)
    sess = kept_frame_counter(devs, R.SESSION_GAP_S)
    frames = {}
    for p in X.day_paths():
        t = time.time()
        frames[p.stem] = build_space_day(p, devs, CELLS, sess, a.cap, a.force)
        print(f"[space] {p.stem}: {frames[p.stem]} frames kept "
              f"({time.time() - t:.0f}s) peak RSS {X.peak_rss_mb():.0f} MB",
              file=sys.stderr)
    del sess
    gc.collect()

    rng_b = np.random.default_rng(E.SEED + 1)
    he = None if a.no_he else HeSpace()
    X.log_rss("He loaded")
    cells_out, rows = {}, []
    for cell in CELLS:
        t = time.time()
        s_, f_ = cell
        w = home_windows(devs, R.SESSION_GAP_S, cell, a.force)
        home = home_observers(w, cell, a.trials, rng_b)
        n_win = int(w["dev"].size)
        del w
        he_res = gain = None
        if he is not None:
            he_res = he_observers(he, cell, a.he_seeds)
            gain = he.gain(cell, np.random.default_rng([GAIN_SEED, 99, CELLS.index(cell)]))
        v = verdict(home, he_res, gain)
        cells_out[cell_name(*cell)] = dict(
            structure=s_, freshness=f_, blocks=blocks_for(s_, f_), cost=cost(s_, f_),
            home=home, home_windows=n_win, he=he_res, gain=gain, verdict=v)
        for ds, res in (("home", home), ("he", he_res or {})):
            for obs, byT in res.items():
                for T, r in byT.items():
                    rows.append(dict(cell=cell_name(*cell), structure=s_, freshness=f_,
                                     dataset=ds, observer=obs, T=int(T),
                                     accuracy=r["accuracy"], chance=r["chance"],
                                     excess=r["accuracy"] - r["chance"],
                                     gain_col1=(gain or {}).get("gain_col1"),
                                     **{k: v_ for k, v_ in cost(s_, f_).items()
                                        if k in ("random_numbers", "complex_values_stored")}))
        print(f"[cell] {cell_name(*cell)}: pass={v['passes']} "
              f"first_break={v['first_break']} ({time.time() - t:.0f}s) "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
        gc.collect()
    passing = [k for k, c in cells_out.items() if c["verdict"]["passes"]]
    E.write_csv(OUT["csv"], rows)
    X.write_json(OUT["json"], dict(
        structures=list(STRUCTURES), freshness=FRESHNESS, blocks=list(BLOCKS),
        home=dict(condition=HOME_COND, k="all", t_grid=list(T_GRID), trials=a.trials,
                  windows_per_block_cap=a.cap, frames_kept_per_day=frames),
        he=None if a.no_he else dict(seeds=a.he_seeds, t_grid=list(T_GRID),
                                     n_decisions_per_class=HE_N_DECISIONS,
                                     gain_reports=GAIN_REPORTS),
        pass_rule=dict(margin=PASS_MARGIN, gain_min=GAIN_MIN),
        cost_basis=dict(n_t=N_T, groups=GROUPS),
        cells=cells_out, passing=passing,
        passing_he=[k for k, c in cells_out.items() if c["verdict"]["passes_he"]],
        passing_home=[k for k, c in cells_out.items()
                      if c["verdict"]["passes_home"]],
        cheapest_passing=cheapest(cells_out, "passes"),
        cheapest_passing_he=cheapest(cells_out, "passes_he"),
        cheapest_passing_home=cheapest(cells_out, "passes_home"),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB; "
          f"passing {passing}; cheapest {cheapest(cells_out, 'passes')}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
