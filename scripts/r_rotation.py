#!/usr/bin/env python3
"""R-rotation: the sounding rotation of Section 7.2 on the residential capture.

For every report of the six labelled devices the full Givens angle set is
decoded (phi_11, phi_21, psi_21, psi_31 and, for Nc = 2, phi_22, psi_32),
V_k (Nr x Nc) is reconstructed per reported tone group k, a Haar-random
Nr x Nr unitary Q is drawn and Q^H V_k is re-decomposed into Givens angles
and re-quantised at the report's own codebook (bphi, bpsi). Two variants:

  rot_report   one Q per report, the same on all tone groups
  rot_group    one fresh Q per report AND per tone group (the proposal)

From the rotated angles the paper's 320-d window vector is rebuilt exactly
as r_common does (first column, 32 tone bins, circular stats), plus the
format-blind copy at the coarsest codebook (feat_rq). A third per-frame
quantity is kept for the invariant-reading observer of joint_defense_eval
("nn1_inv"): |v_k^H v_0| between the first column at tone k and at the
reference tone, binned into 32 tone bins; per window its mean and std
(64-d). The per-report rotation leaves these magnitudes unchanged up to
re-quantisation; the per-group rotation does not.

Sessions and windows are the ones of r_common (same frames, same session
cut); the evaluation is r_eval.run_condition with r_eval's grids and seeds,
so the unprotected rows reproduce figures/r_main.csv.

Controls (--skip-main runs them alone and replaces their rows in the outputs)
  A2  after the per-group rotation every report is re-quantised at one
      common codebook (4/2 bits, 6/4 bits) before the features are built,
      and in a third variant also cut to the 234 tone groups every format
      shares; same Q draws as rot_group. If the observers fall to chance,
      what the rotation left in the body was the format.
  A3  galleries restricted to one codebook class (VHT MU 9/7: laptop,
      vacuum, Samsung TV; SU 6/4: Apple TV bedroom, LG TV), unprotected
      and per-group rotated; r_rotation_gallery.csv.
  A4  (--common-grid-only) the unprotected reports on the same common grid
      as A2's third variant (4/2 bits, 234 tone groups), so that the angles
      identify without the format's grid; r_rotation_common_grid*.csv.

Memory: one day of bodies at a time, rotation in chunks of ROT_CHUNK
reports; the per-day caches hold only the binned per-frame quantities.

Outputs (figures/): r_rotation_main.csv, r_rotation_per_device.csv,
r_rotation_summary.json. Window tables per variant are cached under
BFI_X_CACHE_DIR/r_rot/<variant>/ for r_format_equalized.py.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_defense_common as C  # noqa: E402  (givens_to_v, v_to_givens, haar)
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANTS = ("none", "rot_report", "rot_group")
VARIANT_NOTES = {
    "none": "unprotected, the report as captured",
    "rot_report": "one Haar unitary per report, shared by all tone groups",
    "rot_group": "one fresh Haar unitary per report and per tone group",
}
OBSERVERS = ("angles", "angles_centroid", "angles_rq", "format",
             "angles+format", "invariant")
# Output specs (bphi, bpsi, n_sub cap); None = the report's own value.
NATIVE_SPEC = (None, None, None)
# A2: after the per-group rotation every report is re-quantised at ONE
# common codebook before the features are built (4/2 and 6/4 bits), and,
# in the third variant, also cut to the tone count every format shares
# (234 tones of VHT 80 MHz Ng = 1 against 250 of HE Ng = 4), so that the
# body statistics carry no format at all.
N_SUB_COMMON = 234
A2_ROTATION = "rot_group"
A2_OUTPUTS = {"rot_group_cb42": (4, 2, None),
              "rot_group_cb64": (6, 4, None),
              "rot_group_cb42_n234": (4, 2, N_SUB_COMMON)}
A2_NOTES = {
    "rot_group_cb42": "per-group rotation, all reports re-quantised at "
                      "4-bit phi / 2-bit psi",
    "rot_group_cb64": "per-group rotation, all reports re-quantised at "
                      "6-bit phi / 4-bit psi",
    "rot_group_cb42_n234": "per-group rotation, 4/2 bits and the first "
                           f"{N_SUB_COMMON} tone groups of every report",
}
# A4: the unprotected reports on A2's common grid (rotation "none").
COMMON_GRID_ROTATION = "none"
COMMON_GRID_OUTPUTS = {"none_cb42_n234": (4, 2, N_SUB_COMMON)}
# A3: galleries restricted to devices of one codebook class.
SAME_CODEBOOK_GALLERIES = {
    "VHT_MU_9_7": ("Laptop", "Vacuum", "SamsungTV"),
    "SU_6_4": ("AppleTV-BR", "LGTV"),
}
A3_BODIES = ("none", "rot_group")
A3_SEED_OFFSET = 20
A3_FEATURES = ("angles", "angles_centroid", "angles_rq", "invariant")
ROT_CACHE_DIR = R.CACHE_DIR.parent / "r_rot"
ROT_SEED = 7                    # Haar draws; one stream per (variant, day)
ROT_CHUNK = 2000                # reports per reconstruction/rotation chunk
INV_REF_TONE = 0                # reference tone of |v_k^H v_ref|
INV_DIM = 2 * R.N_SC_BINS       # window mean and std of the 32 binned magnitudes
SELF_TEST_PER_GEOM = 200        # bodies per geometry in the identity round trip
HEADLINE_K = ("1", "all")
HEADLINE_T = (1, 32)
OUT = dict(main=R.FIG_DIR / "r_rotation_main.csv",
           per_device=R.FIG_DIR / "r_rotation_per_device.csv",
           gallery=R.FIG_DIR / "r_rotation_gallery.csv",
           summary=R.FIG_DIR / "r_rotation_summary.json",
           common_grid=R.FIG_DIR / "r_rotation_common_grid.csv",
           common_grid_per_device=R.FIG_DIR / "r_rotation_common_grid_per_device.csv",
           common_grid_summary=R.FIG_DIR / "r_rotation_common_grid_summary.json")
REF_MAIN = R.FIG_DIR / "r_main.csv"


# ---------------------------------------------------------------------------
# Givens: decode all angles, (de)quantise, rotate
# ---------------------------------------------------------------------------
def decode_all(bodies: list[bytes], geom: dict):
    """Quantised Givens angles of bodies sharing geometry and length:
    (n, n_ang, n_sub) int32 in 802.11 layout order, or None with the same
    skip rule as r_common.decode_first_column (so the frame set is equal)."""
    if (geom["nr"] != R.NR_COMMON or geom["bphi"] < R.PHI_BITS_MIN
            or geom["bpsi"] < R.PSI_BITS_MIN):
        return None
    offsets, widths, kinds, bits_per_sc = X.AD._layout(
        geom["nr"], geom["nc"], geom["bphi"], geom["bpsi"])
    start = 2 + geom["mimo_len"] + geom["nc"]
    L = len(bodies[0]) - start
    if L <= 0:
        return None
    mat = np.frombuffer(b"".join(b[start:] for b in bodies),
                        dtype=np.uint8).reshape(len(bodies), L)
    bits = np.unpackbits(mat, axis=1, bitorder="little")
    n_sub = bits.shape[1] // bits_per_sc
    n_exp = X.AD.expected_subcarriers(geom)
    if n_exp is not None:
        n_sub = min(n_sub, n_exp)
    if n_sub < R.N_SC_BINS:
        return None
    base = np.arange(n_sub, dtype=np.int64) * bits_per_sc
    q = np.zeros((len(bodies), len(widths), n_sub), dtype=np.int32)
    for j in range(len(widths)):
        for k in range(int(widths[j])):
            q[:, j] += bits[:, base + offsets[j] + k].astype(np.int32) << k
    return q


def split_kinds(geom: dict) -> np.ndarray:
    """0 = phi, 1 = psi per angle in layout order."""
    return X.AD._layout(geom["nr"], geom["nc"], geom["bphi"],
                        geom["bpsi"])[2]


def dequantise(q: np.ndarray, geom: dict):
    """(n, n_ang, n_sub) ints -> phi (n, n_sub, n_phi), psi (n, n_sub, n_psi)
    radians, 802.11 mid-cell rule, in the order givens_to_v expects."""
    kinds = split_kinds(geom)
    phi = X._angle_to_rad(q[:, kinds == 0], geom["bphi"], "phi")
    psi = X._angle_to_rad(q[:, kinds == 1], geom["bpsi"], "psi")
    return np.moveaxis(phi, 1, -1), np.moveaxis(psi, 1, -1)


def quantise(phi: np.ndarray, psi: np.ndarray, geom: dict) -> np.ndarray:
    """Inverse of dequantise: radians -> (n, n_ang, n_sub) ints at the
    report's codebook. phi cells wrap on [0, 2 pi); psi cells cover
    [0, pi/2) and are clipped."""
    bphi, bpsi = geom["bphi"], geom["bpsi"]
    kp = np.floor(np.mod(phi, 2 * np.pi) * 2 ** (bphi - 1) / np.pi)
    kp = np.mod(kp, 2 ** bphi).astype(np.int32)
    ks = np.floor(psi * 2 ** (bpsi + 1) / np.pi)
    ks = np.clip(ks, 0, 2 ** bpsi - 1).astype(np.int32)
    kinds = split_kinds(geom)
    q = np.empty((phi.shape[0], kinds.size, phi.shape[1]), dtype=np.int32)
    q[:, kinds == 0] = np.moveaxis(kp, -1, 1)
    q[:, kinds == 1] = np.moveaxis(ks, -1, 1)
    return q


def reconstruct_v(q: np.ndarray, geom: dict) -> np.ndarray:
    """(n, n_ang, n_sub) ints -> V (n, n_sub, nr, nc) complex."""
    phi, psi = dequantise(q, geom)
    return C.givens_to_v(phi, psi, nr=geom["nr"], nc=geom["nc"])


def rotate_angles(q: np.ndarray, geom: dict, variant: str,
                  rng: np.random.Generator):
    """Q^H V_k per report (and per tone group for rot_group), re-decomposed
    into continuous Givens angles: phi (n, n_sub, n_phi), psi (n, n_sub,
    n_psi). 'none' returns the mid-cell angles of q."""
    if variant == "none":
        return dequantise(q, geom)
    nr, nc = geom["nr"], geom["nc"]
    phis, psis = [], []
    for s in range(0, q.shape[0], ROT_CHUNK):
        sl = slice(s, s + ROT_CHUNK)
        V = reconstruct_v(q[sl], geom)                  # (n, k, nr, nc)
        n, k = V.shape[:2]
        if variant == "rot_report":
            Q = C.haar_unitary(nr, (n,), rng)[:, None]   # same on all groups
        elif variant == "rot_group":
            Q = C.haar_unitary(nr, (n, k), rng)
        else:
            raise ValueError(variant)
        V2 = np.conj(np.swapaxes(Q, -1, -2)) @ V
        phi2, psi2 = C.v_to_givens(V2, nr=nr, nc=nc)
        phis.append(phi2)
        psis.append(psi2)
        del V, V2, Q
    return np.concatenate(phis), np.concatenate(psis)


def rotate(q: np.ndarray, geom: dict, variant: str,
           rng: np.random.Generator) -> np.ndarray:
    """rotate_angles re-quantised at the report's own codebook."""
    if variant == "none":
        return q
    return quantise(*rotate_angles(q, geom, variant, rng), geom)


def output_geom(geom: dict, spec: tuple) -> dict:
    """Geometry with the codebook of an output spec (bphi, bpsi, n_sub cap);
    None keeps the report's own value."""
    bphi, bpsi, _cap = spec
    return dict(geom, bphi=bphi or geom["bphi"], bpsi=bpsi or geom["bpsi"])


def first_column_rads(q: np.ndarray, geom: dict):
    """The four first-column angles (layout order = R.ANGLE_NAMES) in
    radians at the report's codebook and at the coarsest codebook."""
    kinds = split_kinds(geom)[:R.N_FIRST_COL]
    rad = np.empty((q.shape[0], R.N_FIRST_COL, q.shape[2]))
    rad_rq = np.empty_like(rad)
    for j in range(R.N_FIRST_COL):
        kind = "phi" if kinds[j] == 0 else "psi"
        nb = geom["bphi"] if kind == "phi" else geom["bpsi"]
        nmin = R.PHI_BITS_MIN if kind == "phi" else R.PSI_BITS_MIN
        rad[:, j] = X._angle_to_rad(q[:, j], nb, kind)
        rad_rq[:, j] = X._angle_to_rad(q[:, j] >> (nb - nmin), nmin, kind)
    return rad, rad_rq


def invariant_bins(q: np.ndarray, geom: dict) -> np.ndarray:
    """|v_k^H v_ref| of the first column across tone groups (k != ref),
    averaged into N_SC_BINS equal-width bins -> (n, N_SC_BINS) float32."""
    V = reconstruct_v(q, geom)[..., 0]                 # (n, k, nr) first column
    g = np.abs(np.einsum("nkr,nr->nk", np.conj(V), V[:, INV_REF_TONE]))
    g = np.delete(g, INV_REF_TONE, axis=1)
    n_sub = g.shape[1]
    b = X.bin_edges(n_sub, R.N_SC_BINS)
    M = np.zeros((n_sub, R.N_SC_BINS))
    M[np.arange(n_sub), b] = 1.0
    M /= M.sum(axis=0, keepdims=True)
    return (g @ M).astype(np.float32)


def invariant_window(fb: np.ndarray) -> np.ndarray:
    """(T, N_SC_BINS) -> 64-d: per-bin mean and std over the window."""
    return np.concatenate([fb.mean(0), fb.std(0)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-day cache per variant (same layout as r_common plus 'inv')
# ---------------------------------------------------------------------------
def variant_dir(variant: str) -> Path:
    return ROT_CACHE_DIR / variant


def day_file(name: str, stem: str) -> Path:
    return variant_dir(name) / f"{stem}.r{R.R_CACHE_VERSION}.npz"


def build_day(p: Path, devs: list[str], variant: str,
              outputs: dict | None = None, force: bool = False) -> list[Path]:
    """One rotation pass (variant) over one day, written once per output
    spec {name: (bphi, bpsi, n_sub cap)}; the default output is the
    report's own codebook under the variant's name. Same Q draws whatever
    the outputs."""
    outputs = outputs or {variant: NATIVE_SPEC}
    outs = {k: day_file(k, p.stem) for k in outputs}
    if not force and all(o.exists() for o in outs.values()):
        return list(outs.values())
    for o in outs.values():
        o.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([ROT_SEED, VARIANTS.index(variant),
                                 int(p.stem.replace("bfi_", "")
                                     .replace("-", ""))])
    c = X.connect_ro(p)
    q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
         f"({','.join('?' * len(devs))}) ORDER BY ts")
    cur = c.execute(q, devs)
    geoms: dict[tuple, int] = {}
    cols = {k: defaultdict(list) for k in outputs}
    n_skip = n_rot = 0
    while True:
        rows = cur.fetchmany(R.FETCH_CHUNK)
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n)
        dev = np.empty(n, np.int8)
        gid = np.full(n, -1, np.int16)
        feat = {k: np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
                for k in outputs}
        feat_rq = {k: np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
                   for k in outputs}
        inv = {k: np.full((n, R.N_SC_BINS), np.nan, np.float32)
               for k in outputs}
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
        for (key, _L), idx in groups.items():
            g = gdict[key]
            qa = decode_all([bodies[i] for i in idx], g)
            if qa is None:
                continue
            phi2, psi2 = rotate_angles(qa, g, variant, rng)
            n_rot += qa.shape[0]
            ia = np.asarray(idx)
            for name, spec in outputs.items():
                g_out = output_geom(g, spec)
                qr = quantise(phi2, psi2, g_out)
                if spec[2]:
                    qr = qr[:, :, :spec[2]]
                rad, rad_rq = first_column_rads(qr, g_out)
                feat[name][ia] = R.bin_frames(rad)
                feat_rq[name][ia] = R.bin_frames(rad_rq)
                inv[name][ia] = invariant_bins(qr, g_out)
                del qr, rad, rad_rq
            del qa, phi2, psi2
        ok = ~np.isnan(feat[variant if variant in outputs
                            else next(iter(outputs))][:, 0, 0])
        n_skip += int((~ok).sum())
        for name in outputs:
            for k, v in dict(ts=ts[ok], dev=dev[ok], gid=gid[ok],
                             feat=feat[name][ok], feat_rq=feat_rq[name][ok],
                             inv=inv[name][ok]).items():
                cols[name][k].append(v)
        del rows, bodies, groups, feat, feat_rq, inv
    c.close()
    for name in outputs:
        arr = {k: np.concatenate(v) if v else np.zeros(0)
               for k, v in cols[name].items()}
        np.savez(outs[name], **arr, n_undecodable=np.asarray(n_skip),
                 n_rotated=np.asarray(n_rot),
                 geoms=np.asarray(json.dumps([list(k) for k in geoms])))
        del arr
    del cols
    gc.collect()
    return list(outs.values())


def load_day(variant: str, stem: str) -> dict:
    z = np.load(day_file(variant, stem), allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "geoms"}
    d["geoms"] = [tuple(g) for g in json.loads(str(z["geoms"]))]
    return d


# ---------------------------------------------------------------------------
# Windows (r_common's cut, plus the invariant window vector)
# ---------------------------------------------------------------------------
def build_windows(devs: list[str], gap_s: float, variant: str,
                  inv_keys: tuple = ("inv",)) -> dict:
    """r_common's window cut on a variant's per-day cache; every column in
    inv_keys is windowed with invariant_window (mean and std per bin)."""
    ts_all = R.all_timestamps(devs)          # from r_common's own cache
    sess_of = {m: (t, R.cut_sessions(t, gap_s)) for m, t in ts_all.items()}
    del ts_all
    cols = defaultdict(list)
    geom_global: dict[tuple, int] = {}
    for p in X.day_paths():
        d = load_day(variant, p.stem)
        if "ts" not in d or not d["ts"].size:
            continue
        day = p.stem.replace("bfi_", "")
        for i, m in enumerate(devs):
            sel = np.flatnonzero(d["dev"] == i)
            if sel.size == 0:
                continue
            t_all, sid_all = sess_of[m]
            sid = sid_all[np.searchsorted(t_all, d["ts"][sel])]
            for s in np.unique(sid):
                ii = sel[sid == s]
                for w in range(ii.size // R.WINDOW_FRAMES):
                    jj = ii[w * R.WINDOW_FRAMES:(w + 1) * R.WINDOW_FRAMES]
                    g, cnt = np.unique(d["gid"][jj], return_counts=True)
                    gk = d["geoms"][int(g[np.argmax(cnt)])]
                    cols["dev"].append(i)
                    cols["session"].append(int(s))
                    cols["day"].append(day)
                    cols["ts0"].append(float(d["ts"][jj[0]]))
                    cols["ts1"].append(float(d["ts"][jj[-1]]))
                    cols["geom"].append(geom_global.setdefault(
                        gk, len(geom_global)))
                    cols["geom_purity"].append(float(cnt.max()) / jj.size)
                    cols["feat"].append(R.window_feature(d["feat"][jj]))
                    cols["feat_rq"].append(R.window_feature(d["feat_rq"][jj]))
                    for key in inv_keys:
                        cols[key].append(invariant_window(d[key][jj]))
        del d
        gc.collect()
    w = {k: np.asarray(v) for k, v in cols.items()}
    w["geom_table"] = np.asarray([json.dumps(list(k)) for k in geom_global])
    w["macs"] = np.asarray(devs)
    w["gap_s"] = np.asarray(gap_s)
    return w


def windows_path(variant: str, gap_s: float) -> Path:
    return variant_dir(variant) / \
        f"windows_gap{int(gap_s)}.r{R.R_CACHE_VERSION}.npz"


def get_windows(devs: list[str], gap_s: float, variant: str,
                force: bool = False) -> dict:
    p = windows_path(variant, gap_s)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if variant in A2_OUTPUTS:
        rotation, outputs = A2_ROTATION, A2_OUTPUTS
    elif variant in COMMON_GRID_OUTPUTS:
        rotation, outputs = COMMON_GRID_ROTATION, COMMON_GRID_OUTPUTS
    else:
        rotation, outputs = variant, None
    for day in X.day_paths():
        t = time.time()
        build_day(day, devs, rotation, outputs=outputs, force=force)
        print(f"[{variant}] {day.stem} ({time.time() - t:.0f}s) "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    w = build_windows(devs, gap_s, variant)
    np.savez(p, **w)
    X.log_rss(f"{variant} windows built")
    return w


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def self_test(p: Path, devs: list[str]) -> dict:
    """(1) decode_all agrees with r_common.decode_first_column on the first
    four angles; (2) reconstruct -> decompose -> quantise with Q = I returns
    the captured integers; (3) with Q Haar per report the first-column
    |v_k^H v_ref| are preserved before re-quantisation."""
    c = X.connect_ro(p)
    q = (f"SELECT body_hex FROM bfi WHERE {X.SRC_COL} IN "
         f"({','.join('?' * len(devs))}) ORDER BY ts")
    per: dict[tuple, list] = defaultdict(list)
    for (bh,) in c.execute(q, devs):
        if bh is None:
            continue
        raw = bytes.fromhex(bh)
        g, key = R._geom_from_body(raw)
        if g is None or len(per[(key, len(raw))]) >= SELF_TEST_PER_GEOM:
            continue
        per[(key, len(raw))].append(raw)
    c.close()
    rep = {}
    rng = np.random.default_rng(ROT_SEED)
    for (key, _L), bodies in per.items():
        g = X.AD.parse_geometry(bodies[0])
        qa = decode_all(bodies, g)
        ref = R.decode_first_column(bodies, g)
        if qa is None or ref is None:
            rep[json.dumps(list(key))] = dict(checked=len(bodies),
                                              skipped=True)
            continue
        first_col_ok = bool(np.array_equal(qa[:, :R.N_FIRST_COL], ref[1]))
        phi, psi = C.v_to_givens(reconstruct_v(qa, g), nr=g["nr"], nc=g["nc"])
        roundtrip_bad = int((quantise(phi, psi, g) != qa).any(axis=(1, 2)).sum())
        V = reconstruct_v(qa, g)
        Q = C.haar_unitary(g["nr"], (V.shape[0],), rng)[:, None]
        V2 = np.conj(np.swapaxes(Q, -1, -2)) @ V
        g0 = np.abs(np.einsum("nkr,nr->nk", np.conj(V[..., 0]),
                              V[:, INV_REF_TONE, :, 0]))
        g1 = np.abs(np.einsum("nkr,nr->nk", np.conj(V2[..., 0]),
                              V2[:, INV_REF_TONE, :, 0]))
        rep[json.dumps(list(key))] = dict(
            checked=len(bodies), first_column_matches_r_common=first_col_ok,
            identity_roundtrip_mismatched=roundtrip_bad,
            gram_invariance_max_abs_err=float(np.abs(g0 - g1).max()),
            unitary_max_abs_err=float(np.abs(
                np.swapaxes(np.conj(Q), -1, -2) @ Q
                - np.eye(g["nr"])).max()))
    return rep


def check_none_equals_r_common(w_none: dict, w_ref: dict) -> dict:
    """The 'none' variant must give r_common's windows frame for frame."""
    same_rows = (w_none["ts0"].shape == w_ref["ts0"].shape
                 and np.array_equal(w_none["ts0"], w_ref["ts0"])
                 and np.array_equal(w_none["dev"], w_ref["dev"])
                 and np.array_equal(w_none["session"], w_ref["session"]))
    feat_err = float(np.nanmax(np.abs(w_none["feat"] - w_ref["feat"]))) \
        if same_rows else float("nan")
    rq_err = float(np.nanmax(np.abs(w_none["feat_rq"] - w_ref["feat_rq"]))) \
        if same_rows else float("nan")
    geom_same = same_rows and all(
        w_none["geom_table"][a] == w_ref["geom_table"][b]
        for a, b in zip(w_none["geom"], w_ref["geom"]))
    return dict(same_windows=bool(same_rows), feat_max_abs_err=feat_err,
                feat_rq_max_abs_err=rq_err, same_geom=bool(geom_same),
                n_windows=int(w_none["ts0"].size))


def check_variant_alignment(w: dict, w_ref: dict) -> bool:
    return (w["ts0"].shape == w_ref["ts0"].shape
            and np.array_equal(w["ts0"], w_ref["ts0"])
            and np.array_equal(w["geom"], w_ref["geom"]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(w: dict, extra: dict, trials: int, rng_b):
    """r_eval's protocol, grids and seeds on one window table."""
    tb = E.Table(w)
    rows, dev_rows = [], []
    for ci, cond in enumerate(E.CONDITIONS):
        t = time.time()
        recs = E.run_condition(tb, cond, E.K_GRID, E.T_GRID, trials,
                               E.SEED + ci)
        r1, r2 = E.aggregate(recs, tb.names, extra, rng_b)
        rows += r1
        dev_rows += r2
        print(f"[eval] {extra.get('body')} {extra.get('observer_input')} "
              f"{cond}: {len(recs)} records {time.time() - t:.0f}s "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
        del recs
        gc.collect()
    del tb
    return rows, dev_rows


def observer_rows(w: dict, body: str, trials: int, rng_b):
    """Plain observers on the angle vector, then the invariant observer
    (the 'angles' rows of a table whose feature slot holds the 64-d
    |v_k^H v_ref| window vector)."""
    main, dev = evaluate(w, dict(body=body, observer_input="angles"), trials,
                         rng_b)
    w_inv = dict(w, feat=w["inv"])
    m2, d2 = evaluate(w_inv, dict(body=body, observer_input="invariant"),
                      trials, rng_b)
    for r in m2 + d2:
        r["feature"] = "invariant" if r["feature"] == "angles" else None
    main += [r for r in m2 if r["feature"]]
    dev += [r for r in d2 if r["feature"]]
    for r in main + dev:
        r.pop("observer_input", None)
    return main, dev


def gallery_rows(w: dict, body: str, trials: int, rng_b) -> list[dict]:
    """A3: r_eval's protocol on tables restricted to the devices of one
    codebook class (chance = 1 / |gallery| of that class)."""
    names = [R.short(str(m)) for m in w["macs"]]
    rows = []
    for gname, members in SAME_CODEBOOK_GALLERIES.items():
        keep = np.isin(w["dev"], [names.index(m) for m in members])
        for inp, feat in (("angles", w["feat"]), ("invariant", w["inv"])):
            tb = E.Table(dict(w, feat=feat), keep)
            for ci, cond in enumerate(E.CONDITIONS):
                recs = E.run_condition(tb, cond, E.K_GRID, E.T_GRID, trials,
                                       E.SEED + A3_SEED_OFFSET + ci)
                r1, _ = E.aggregate(recs, tb.names,
                                    dict(body=body, gallery=gname), rng_b)
                for r in r1:
                    if inp == "invariant":
                        if r["feature"] != "angles":
                            continue
                        r["feature"] = "invariant"
                    if r["feature"] in A3_FEATURES:
                        rows.append(r)
                del recs
            del tb
            gc.collect()
        print(f"[a3] {body} {gname}: {len(rows)} rows "
              f"peak RSS {X.peak_rss_mb():.0f} MB", file=sys.stderr)
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def reference_rows() -> dict:
    """Unprotected rows of figures/r_main.csv at the paper's gap."""
    ref = {}
    if not REF_MAIN.exists():
        return ref
    with open(REF_MAIN, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["gap_s"]) == int(R.SESSION_GAP_S):
                ref[(r["condition"], r["feature"], r["K"], int(r["T"]))] = \
                    float(r["accuracy"])
    return ref


def headline(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        if r["K"] in HEADLINE_K and r["T"] in HEADLINE_T:
            out[f"{r['body']}|{r['feature']}|{r['condition']}|K={r['K']}"
                f"|T={r['T']}"] = dict(
                accuracy=round(r["accuracy"], 4),
                macro_recall=round(r["macro_recall"], 4),
                chance=round(r["chance"], 4),
                ci=[round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
                unprotected_r_main=r.get("unprotected_r_main"))
    return out


NULL_SEED = 13
NULL_BODY = "null_noise"


def run_null(trials: int) -> int:
    """The protocol on pure noise: the same windows, sessions, galleries
    and seeds, every feature vector i.i.d. standard normal. What it scores
    above chance is the decision rule's own bias (1-NN with K = all favours
    the device with the most enrolled windows), not a leak."""
    t0 = time.time()
    devs = R.device_macs()
    w = get_windows(devs, R.SESSION_GAP_S, "none")
    rng = np.random.default_rng(NULL_SEED)
    w_null = dict(w, feat=rng.standard_normal(w["feat"].shape).astype(np.float32),
                  feat_rq=rng.standard_normal(w["feat_rq"].shape).astype(np.float32),
                  inv=rng.standard_normal(w["inv"].shape).astype(np.float32))
    del w
    rng_b = np.random.default_rng(E.SEED + 1)
    m, d = observer_rows(w_null, NULL_BODY, trials, rng_b)
    ref = reference_rows()
    for r in m:
        r["unprotected_r_main"] = ref.get(
            (r["condition"], r["feature"], r["K"], r["T"]))
    main_rows = [r for r in read_csv_rows(OUT["main"])
                 if r["body"] != NULL_BODY] + m
    dev_rows = [r for r in read_csv_rows(OUT["per_device"])
                if r["body"] != NULL_BODY] + d
    E.write_csv(OUT["main"], main_rows)
    E.write_csv(OUT["per_device"], dev_rows)
    summary = json.loads(OUT["summary"].read_text())
    summary["null_noise"] = dict(
        note=run_null.__doc__.strip(), seed=NULL_SEED, headline=headline(m),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb())
    X.write_json(OUT["summary"], summary)
    print(f"[done] null {time.time() - t0:.0f}s peak RSS "
          f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
    return 0


def run_common_grid(trials: int) -> int:
    """A4: r_eval's protocol on the unprotected reports re-quantised at 4/2
    bits and cut to the 234 tone groups every format shares, with a fresh
    evaluation stream so that no other output changes."""
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    for p in X.day_paths():
        R.build_day(p, devs)
    w_ref = R.get_windows(devs, R.SESSION_GAP_S)
    rows, dev_rows, aligned = [], [], {}
    rng_b = np.random.default_rng(E.SEED + 1)
    for variant in COMMON_GRID_OUTPUTS:
        w = get_windows(devs, R.SESSION_GAP_S, variant)
        aligned[variant] = check_variant_alignment(w, w_ref)
        m, d = observer_rows(w, variant, trials, rng_b)
        rows += m
        dev_rows += d
        del w
        gc.collect()
    E.write_csv(OUT["common_grid"], rows)
    E.write_csv(OUT["common_grid_per_device"], dev_rows)
    X.write_json(OUT["common_grid_summary"], dict(
        note=run_common_grid.__doc__.strip(), outputs=COMMON_GRID_OUTPUTS,
        aligned=aligned, trials=trials, headline=headline(rows),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] common grid {time.time() - t0:.0f}s peak RSS "
          f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate the memory of the rotation pass only")
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-main", action="store_true",
                    help="keep the main variants' outputs; run and append "
                         "the A2 and A3 controls only")
    ap.add_argument("--null-only", action="store_true",
                    help="append the pure-noise null (every window vector "
                         "i.i.d. standard normal, same windows and protocol)")
    ap.add_argument("--common-grid-only", action="store_true",
                    help="A4: the unprotected reports on the common grid "
                         "(4/2 bits, 234 tone groups), separate outputs")
    a = ap.parse_args()
    if a.null_only:
        return run_null(a.trials)
    if a.common_grid_only:
        return run_common_grid(a.trials)
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        est = R.dry_run(devs)
        max_frames = max(v for k, v in est.items() if not k.startswith("_"))
        per_report = 250 * (3 * 2 + 3 * 3) * 16 * 3      # V, Q, temporaries
        est["_est_rotation_chunk_mb"] = ROT_CHUNK * per_report / 1e6
        est["_est_peak_day_mb"] = max_frames * (7 * 32 * 4 + 16 + 2500) / 1e6
        print(json.dumps(dict(devices=devs, rows=est), indent=2))
        return 0

    # Reference windows from r_common (builds its cache if absent) --------
    for p in X.day_paths():
        R.build_day(p, devs)
    w_ref = R.get_windows(devs, R.SESSION_GAP_S)
    X.log_rss("r_common windows")
    tests = dict(decoder=self_test(X.ARCHIVE_DIR / "bfi_2026-05-15.sqlite",
                                   devs))
    print(json.dumps(tests, indent=1), file=sys.stderr)

    ref = reference_rows()
    rng_b = np.random.default_rng(E.SEED + 1)
    main_rows, dev_rows, frames = [], [], {}

    def run_variant(variant: str):
        t = time.time()
        w = get_windows(devs, R.SESSION_GAP_S, variant, force=a.force)
        if variant == "none":
            tests["none_equals_r_common"] = check_none_equals_r_common(w, w_ref)
            print(json.dumps(tests["none_equals_r_common"]), file=sys.stderr)
        else:
            tests[f"{variant}_aligned"] = check_variant_alignment(w, w_ref)
        frames[variant] = {p.stem: int(load_day(variant, p.stem)
                                       .get("n_rotated", 0))
                           for p in X.day_paths()}
        m, d = observer_rows(w, variant, a.trials, rng_b)
        for r in m:
            r["unprotected_r_main"] = ref.get(
                (r["condition"], r["feature"], r["K"], r["T"]))
        print(f"[variant] {variant}: {time.time() - t:.0f}s", file=sys.stderr)
        del w
        gc.collect()
        X.log_rss(variant)
        return m, d

    if a.skip_main:
        # replace the A2 rows a previous run wrote (the main run writes them
        # too), so running the controls again leaves one row per condition
        main_rows = [r for r in read_csv_rows(OUT["main"]) if r["body"] not in A2_OUTPUTS]
        dev_rows = [r for r in read_csv_rows(OUT["per_device"]) if r["body"] not in A2_OUTPUTS]
        summary = json.loads(OUT["summary"].read_text())
        summary["checks"].update(tests)
    else:
        for variant in VARIANTS:
            m, d = run_variant(variant)
            main_rows += m
            dev_rows += d
        none_vs_ref = [abs(r["accuracy"] - r["unprotected_r_main"])
                       for r in main_rows if r["body"] == "none"
                       and r["unprotected_r_main"] is not None]
        summary = dict(
            variants=VARIANT_NOTES, observers=list(OBSERVERS),
            rotation_seed=ROT_SEED, trials=a.trials,
            k_grid=[E.klabel(k) for k in E.K_GRID], t_grid=list(E.T_GRID),
            session_gap_s=R.SESSION_GAP_S, reports_rotated_per_day=frames,
            checks=tests,
            none_vs_r_main_max_abs_diff=(max(none_vs_ref) if none_vs_ref
                                         else None),
            headline=headline(main_rows),
            runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb())

    # A2: one common codebook (and tone count) after the rotation ---------
    a2_rows = []
    for variant in A2_OUTPUTS:
        m, d = run_variant(variant)
        a2_rows += m
        main_rows += m
        dev_rows += d
    # A3: galleries of one codebook class -------------------------------
    a3_rows = []
    for body in A3_BODIES:
        w = get_windows(devs, R.SESSION_GAP_S, body)
        a3_rows += gallery_rows(w, body, a.trials, rng_b)
        del w
        gc.collect()

    E.write_csv(OUT["main"], main_rows)
    E.write_csv(OUT["per_device"], dev_rows)
    E.write_csv(OUT["gallery"], a3_rows)
    summary["a2_common_codebook"] = dict(
        variants=A2_NOTES, n_sub_common=N_SUB_COMMON,
        aligned={v: tests.get(f"{v}_aligned") for v in A2_OUTPUTS},
        headline=headline(a2_rows))
    summary["a3_same_codebook_gallery"] = dict(
        galleries=SAME_CODEBOOK_GALLERIES, features=list(A3_FEATURES),
        headline={f"{r['gallery']}|{r['body']}|{r['feature']}|{r['condition']}"
                  f"|K={r['K']}|T={r['T']}": dict(
                      accuracy=round(r["accuracy"], 4),
                      macro_recall=round(r["macro_recall"], 4),
                      chance=round(r["chance"], 4),
                      ci=[round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
                      n_decisions=r["n_decisions"])
                  for r in a3_rows if r["K"] in HEADLINE_K
                  and r["T"] in HEADLINE_T})
    summary["controls_runtime_s"] = time.time() - t0
    summary["controls_peak_rss_mb"] = X.peak_rss_mb()
    X.write_json(OUT["summary"], summary)
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
