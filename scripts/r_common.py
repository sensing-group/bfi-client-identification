#!/usr/bin/env python3
"""Residential evaluation for the current revision ("One Labeled Session
Is Enough"): device set, Givens-angle features, sessions and windows.

Device set (R1): the six ground-truth-labelled appliances of the residential
archive (Pi 5 sniffer, one stock router, 2026-05-08..20). Every other source
MAC is excluded; ``exclusions()`` lists them with the reason.

Features (R1): the FIRST COLUMN of the compressed beamforming matrix V,
i.e. the four Givens angles phi_11, phi_21, psi_21, psi_31. All six devices
report Nr = 3 (three AP antennas), so these four angles exist in every
report, whether Nc = 1 or Nc = 2 (the Nc = 2 angles phi_22, psi_32 are
dropped so that one representation covers all devices). Angles are decoded
with the externally validated decoder (esp32-csi-tomography
backend/src/core/bfi/angle_decode.py, bit-exact on the Wi-BFI traces),
dequantised to radians with the 802.11 rule, and averaged into N_SC_BINS
equal-width bins over the reported subcarrier index (234 tones for VHT
80 MHz Ng=1, 250 for HE 80 MHz Ng=4). Per window of WINDOW_FRAMES frames
and per bin: phi -> circular mean (cos, sin) and spread 1-R; psi -> mean and
standard deviation. 2 x 3 x 32 + 2 x 2 x 32 = 320 dimensions.
A second, format-blind copy of the same vector is built after re-quantising
every angle to the coarsest 802.11 codebook (4-bit phi, 2-bit psi), so that
devices reporting at different angle resolutions share one value grid.

Sessions (R2): a session is a maximal run of a device's frames on the global
timeline with no inter-frame gap above SESSION_GAP_S. Windows are
non-overlapping runs of WINDOW_FRAMES consecutive frames inside one
(session, archive day) block, so no window straddles a session gap or a
day file.

Memory: pass 1 reads only (ts, src_mac) per day to cut sessions; pass 2
streams one day at a time (sqlite fetchmany, six MACs only, index on
src_mac), writes a compact per-day cache, and keeps only window-level
features across days (about 30 MB).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xres_common as X  # noqa: E402  (paths, decoder, memory helpers)
from device_names import SHORT_OF_LABEL  # noqa: E402  (archive label -> short key)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = X.CACHE_DIR / "r"
FIG_DIR = X.FIG_DIR
R_CACHE_VERSION = 2
FETCH_CHUNK = 5000
N_SC_BINS = 32
WINDOW_FRAMES = X.WINDOW_FRAMES          # 32, the paper's window
NR_COMMON = 3                             # Nr reported by all six devices
N_FIRST_COL = 2 * (NR_COMMON - 1)         # phi_11, phi_21, psi_21, psi_31
ANGLE_NAMES = ("phi11", "phi21", "psi21", "psi31")
SESSION_GAP_S = float(os.environ.get("BFI_R_SESSION_GAP_S", "300"))
SENSITIVITY_GAPS_S = (120.0, 600.0)
GAP_CANDIDATES_S = (10, 30, 60, 120, 300, 600, 1800)
# Feature block layout: per phi angle (cos, sin, spread), per psi (mean, std)
PHI_STATS = ("cos", "sin", "spread")
PSI_STATS = ("mean", "std")
# Format-blind control: every angle is re-quantised to the coarsest 802.11
# codebook (SU codebook 0: 4-bit phi, 2-bit psi). The 802.11 cells nest, so
# this is exact: k_coarse = k >> (b - b_min), then mid-cell dequantisation.
PHI_BITS_MIN = 4
PSI_BITS_MIN = 2
GEOM_FIELDS = ("fmt", "nc", "nr", "bphi", "bpsi", "bw_mhz", "ng",
               "feedback_type", "codebook")

# Ground truth: X.DEVICE_GT, keyed by OUI prefix on the raw archive and by
# salted id on the released one (X.device_key); each key maps to exactly one
# address in the archive (checked at build).
DEVICE_GT = X.DEVICE_GT


def label(mac: str) -> tuple[str, str]:
    """(paper label, mobility) of a device address."""
    return DEVICE_GT[X.device_key(mac)]


def short(mac: str) -> str:
    return SHORT_OF_LABEL[label(mac)[0]]


# ---------------------------------------------------------------------------
# Device set
# ---------------------------------------------------------------------------
def labelled_macs(p: Path) -> list[str]:
    c = X.connect_ro(p)
    out = [r[0] for r in c.execute(f"SELECT DISTINCT {X.SRC_COL} FROM bfi")
           if r[0] and X.device_key(r[0]) in DEVICE_GT]
    c.close()
    return out


def device_macs() -> list[str]:
    seen: dict[str, set] = defaultdict(set)
    for p in X.day_paths():
        for m in labelled_macs(p):
            seen[X.device_key(m)].add(m)
    for key, ms in seen.items():
        if len(ms) != 1:
            raise RuntimeError(f"device key {key} maps to {sorted(ms)}")
    return sorted((next(iter(v)) for v in seen.values()), key=X.device_rank)


def exclusions(devs: list[str]) -> list[dict]:
    """Every other src_mac in the archive, with frame count, days and the
    reason it is not in the device set."""
    frames: dict[str, int] = defaultdict(int)
    days: dict[str, set] = defaultdict(set)
    ap_addrs: set = set()
    for p in X.day_paths():
        c = X.connect_ro(p)
        for m, n in c.execute(
                f"SELECT {X.SRC_COL}, COUNT(*) FROM bfi GROUP BY {X.SRC_COL}"):
            frames[m] += n
            days[m].add(p.stem.replace("bfi_", ""))
        ap_addrs |= {r[0] for r in c.execute(
            f"SELECT DISTINCT {X.AP_COL} FROM bfi")}
        c.close()
    out = []
    for m in sorted(frames):
        if m in devs:
            continue
        why = []
        if m and X.is_random(m):
            why.append("locally administered (random) MAC")
        if m in ap_addrs:
            why.append("also seen as an AP address")
        why.append("no ground-truth label")
        if frames[m] < 2 * WINDOW_FRAMES:
            why.append(f"below the paper's {2 * WINDOW_FRAMES}-frame cut")
        out.append(dict(mac=m, frames=frames[m], days=sorted(days[m]),
                        reason="; ".join(why)))
    return out


# ---------------------------------------------------------------------------
# Givens decoding of the first column of V
# ---------------------------------------------------------------------------
def decode_first_column(bodies: list[bytes], geom: dict):
    """Decode phi_11, phi_21, psi_21, psi_31 for bodies sharing geometry and
    length. Returns (angles (n, 4, n_sub) radians, quantised ints, angles
    re-quantised to PHI_BITS_MIN/PSI_BITS_MIN bits) or None."""
    if (geom["nr"] != NR_COMMON or geom["bphi"] < PHI_BITS_MIN
            or geom["bpsi"] < PSI_BITS_MIN):
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
    if n_sub < N_SC_BINS:
        return None
    base = np.arange(n_sub, dtype=np.int64) * bits_per_sc
    q = np.zeros((len(bodies), N_FIRST_COL, n_sub), dtype=np.int32)
    for j in range(N_FIRST_COL):            # layout order = ANGLE_NAMES
        for k in range(int(widths[j])):
            q[:, j] += bits[:, base + offsets[j] + k].astype(np.int32) << k
    rad = np.empty(q.shape)
    rad_rq = np.empty(q.shape)
    for j in range(N_FIRST_COL):
        kind = "phi" if kinds[j] == 0 else "psi"
        nb = geom["bphi"] if kind == "phi" else geom["bpsi"]
        nmin = PHI_BITS_MIN if kind == "phi" else PSI_BITS_MIN
        rad[:, j] = X._angle_to_rad(q[:, j], nb, kind)
        rad_rq[:, j] = X._angle_to_rad(q[:, j] >> (nb - nmin), nmin, kind)
    return rad, q, rad_rq


def bin_frames(rad: np.ndarray) -> np.ndarray:
    """(n, 4, n_sub) radians -> (n, 6, N_SC_BINS) float32 per-frame bin
    means: cos phi11, sin phi11, cos phi21, sin phi21, psi21, psi31."""
    n, _, n_sub = rad.shape
    b = X.bin_edges(n_sub, N_SC_BINS)
    M = np.zeros((n_sub, N_SC_BINS))
    M[np.arange(n_sub), b] = 1.0
    M /= M.sum(axis=0, keepdims=True)
    chans = [np.cos(rad[:, 0]), np.sin(rad[:, 0]),
             np.cos(rad[:, 1]), np.sin(rad[:, 1]), rad[:, 2], rad[:, 3]]
    return np.stack([c @ M for c in chans], axis=1).astype(np.float32)


def window_feature(fb: np.ndarray) -> np.ndarray:
    """(T, 6, B) per-frame bins -> 320-d window vector (see module doc)."""
    out = []
    for c, s in ((0, 1), (2, 3)):
        mc, ms = fb[:, c].mean(0), fb[:, s].mean(0)
        out += [mc, ms, 1.0 - np.sqrt(mc ** 2 + ms ** 2)]
    for p in (4, 5):
        out += [fb[:, p].mean(0), fb[:, p].std(0)]
    return np.concatenate(out).astype(np.float32)


FEATURE_DIM = (2 * len(PHI_STATS) + 2 * len(PSI_STATS)) * N_SC_BINS


def feature_names() -> list[str]:
    names = []
    for a in ANGLE_NAMES[:2]:
        names += [f"{a}_{s}_b{j}" for s in PHI_STATS for j in range(N_SC_BINS)]
    for a in ANGLE_NAMES[2:]:
        names += [f"{a}_{s}_b{j}" for s in PSI_STATS for j in range(N_SC_BINS)]
    return names


# ---------------------------------------------------------------------------
# Per-day frame cache (six devices only)
# ---------------------------------------------------------------------------
def day_cache_path(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.r{R_CACHE_VERSION}.npz"


def _geom_from_body(raw: bytes):
    g = X.AD.parse_geometry(raw)
    if g is None:
        return None, None
    return g, tuple(g[k] for k in GEOM_FIELDS)


def build_day(p: Path, devs: list[str], force: bool = False) -> Path:
    out = day_cache_path(p.stem)
    if out.exists() and not force:
        return out
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c = X.connect_ro(p)
    q = (f"SELECT ts, {X.SRC_COL}, body_hex FROM bfi WHERE {X.SRC_COL} IN "
         f"({','.join('?' * len(devs))}) ORDER BY ts")
    cur = c.execute(q, devs)
    geoms: dict[tuple, int] = {}
    cols = defaultdict(list)
    n_skip = 0
    while True:
        rows = cur.fetchmany(FETCH_CHUNK)
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n)
        dev = np.empty(n, np.int8)
        gid = np.full(n, -1, np.int16)
        feat = np.full((n, 6, N_SC_BINS), np.nan, np.float32)
        feat_rq = np.full((n, 6, N_SC_BINS), np.nan, np.float32)
        groups: dict[tuple, list[int]] = defaultdict(list)
        gdict: dict[tuple, dict] = {}
        bodies: list = [None] * n
        for i, (t, m, bh) in enumerate(rows):
            ts[i] = t
            dev[i] = devs.index(m)
            if bh is None:
                continue
            raw = bytes.fromhex(bh)
            g, key = _geom_from_body(raw)
            if g is None:
                continue
            gid[i] = geoms.setdefault(key, len(geoms))
            bodies[i] = raw
            gdict[key] = g
            groups[(key, len(raw))].append(i)
        for (key, _L), idx in groups.items():
            res = decode_first_column([bodies[i] for i in idx], gdict[key])
            if res is None:
                continue
            feat[np.asarray(idx)] = bin_frames(res[0])
            feat_rq[np.asarray(idx)] = bin_frames(res[2])
        ok = ~np.isnan(feat[:, 0, 0])
        n_skip += int((~ok).sum())
        for k, v in dict(ts=ts[ok], dev=dev[ok], gid=gid[ok],
                         feat=feat[ok], feat_rq=feat_rq[ok]).items():
            cols[k].append(v)
        del rows, bodies, groups, feat, feat_rq
    c.close()
    arr = {k: np.concatenate(v) if v else np.zeros(0)
           for k, v in cols.items()}
    np.savez(out, **arr, n_undecodable=np.asarray(n_skip),
             geoms=np.asarray(json.dumps([list(k) for k in geoms])))
    del arr, cols
    gc.collect()
    return out


def load_day_cache(stem: str) -> dict:
    z = np.load(day_cache_path(stem), allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "geoms"}
    d["geoms"] = [tuple(g) for g in json.loads(str(z["geoms"]))]
    return d


def verify_decoder(p: Path, devs: list[str], n_per_geom: int = 200) -> dict:
    """Bit-exact check of decode_first_column against AD.decode_angles on up
    to n_per_geom bodies per geometry of one day."""
    c = X.connect_ro(p)
    q = (f"SELECT body_hex FROM bfi WHERE {X.SRC_COL} IN "
         f"({','.join('?' * len(devs))}) ORDER BY ts")
    per: dict[tuple, list] = defaultdict(list)
    for (bh,) in c.execute(q, devs):
        if bh is None:
            continue
        raw = bytes.fromhex(bh)
        g, key = _geom_from_body(raw)
        if g is None or len(per[key]) >= n_per_geom:
            continue
        per[key].append(raw)
    c.close()
    rep = {}
    for key, bodies in per.items():
        g = X.AD.parse_geometry(bodies[0])
        n_ang = len(X.AD._layout(g["nr"], g["nc"], g["bphi"], g["bpsi"])[1])
        bad = 0
        for b in bodies:
            res = decode_first_column([b], g)
            ref = X.AD.decode_angles(b, g).reshape(-1, n_ang)[:, :N_FIRST_COL]
            if res is None or not np.array_equal(res[1][0].T, ref):
                bad += 1
        rep[json.dumps(list(key))] = dict(checked=len(bodies), mismatched=bad)
    return rep


# ---------------------------------------------------------------------------
# Sessions and windows
# ---------------------------------------------------------------------------
def all_timestamps(devs: list[str]) -> dict[str, np.ndarray]:
    ts = defaultdict(list)
    for p in X.day_paths():
        d = np.load(day_cache_path(p.stem))
        if "ts" in d.files and d["ts"].size:
            for i, m in enumerate(devs):
                ts[m].append(d["ts"][d["dev"] == i])
    return {m: np.sort(np.concatenate(v)) for m, v in ts.items()}


def cut_sessions(t: np.ndarray, gap_s: float) -> np.ndarray:
    """Session id per (sorted) timestamp."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)])


def gap_report(ts: dict[str, np.ndarray]) -> dict:
    rep = {}
    for m, t in ts.items():
        g = np.diff(t)
        rep[short(m)] = dict(
            frames=int(t.size),
            gap_median_s=float(np.median(g)),
            gap_p90_s=float(np.percentile(g, 90)),
            gap_p99_s=float(np.percentile(g, 99)),
            gaps_above={str(e): int((g > e).sum()) for e in GAP_CANDIDATES_S})
    return rep


def split_epochs(t: np.ndarray, epoch_s: float | None, phase_s: float) -> list[np.ndarray]:
    """Index arrays of the runs of t inside one address epoch; all of t if epoch_s is None."""
    if epoch_s is None:
        return [np.arange(t.size)]
    e = np.floor((t - phase_s) / epoch_s)
    return [np.flatnonzero(e == v) for v in np.unique(e)]


def build_windows(devs: list[str], gap_s: float, epoch_s: float | None = None,
                  phase_s: np.ndarray | None = None,
                  window_frames: int = WINDOW_FRAMES) -> dict:
    """Window table for one session gap threshold. Streams day caches.
    With epoch_s, no window straddles an epoch edge of its device, whose edges
    sit at phase_s[device] + n * epoch_s, as if its address changed there."""
    ts_all = all_timestamps(devs)
    sess_of = {m: (t, cut_sessions(t, gap_s)) for m, t in ts_all.items()}
    del ts_all
    cols = defaultdict(list)
    geom_global: dict[tuple, int] = {}
    for p in X.day_paths():
        d = load_day_cache(p.stem)
        if "ts" not in d or not d["ts"].size:
            continue
        day = p.stem.replace("bfi_", "")
        for i, m in enumerate(devs):
            sel = np.flatnonzero(d["dev"] == i)
            if sel.size == 0:
                continue
            tt = d["ts"][sel]
            t_all, sid_all = sess_of[m]
            sid = sid_all[np.searchsorted(t_all, tt)]
            for s in np.unique(sid):
                block = sel[sid == s]
                phase = 0.0 if phase_s is None else float(phase_s[i])
                for ep, part in enumerate(split_epochs(d["ts"][block], epoch_s, phase)):
                    ii = block[part]
                    for w in range(ii.size // window_frames):
                        jj = ii[w * window_frames:(w + 1) * window_frames]
                        g, cnt = np.unique(d["gid"][jj], return_counts=True)
                        gk = d["geoms"][int(g[np.argmax(cnt)])]
                        cols["dev"].append(i)
                        cols["session"].append(int(s))
                        cols["epoch"].append(ep)
                        cols["day"].append(day)
                        cols["ts0"].append(float(d["ts"][jj[0]]))
                        cols["ts1"].append(float(d["ts"][jj[-1]]))
                        cols["geom"].append(geom_global.setdefault(
                            gk, len(geom_global)))
                        cols["geom_purity"].append(float(cnt.max()) / jj.size)
                        cols["feat"].append(window_feature(d["feat"][jj]))
                        cols["feat_rq"].append(
                            window_feature(d["feat_rq"][jj]))
        del d
        gc.collect()
    w = {k: np.asarray(v) for k, v in cols.items()}
    w["geom_table"] = np.asarray([json.dumps(list(k)) for k in geom_global])
    w["macs"] = np.asarray(devs)
    w["gap_s"] = np.asarray(gap_s)
    return w


def windows_path(gap_s: float) -> Path:
    return CACHE_DIR / f"windows_gap{int(gap_s)}.r{R_CACHE_VERSION}.npz"


def get_windows(devs: list[str], gap_s: float, force: bool = False) -> dict:
    p = windows_path(gap_s)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    w = build_windows(devs, gap_s)
    np.savez(p, **w)
    X.log_rss(f"windows gap={gap_s:.0f}s built")
    return w


def dry_run(devs: list[str]) -> dict:
    rep, mx = {}, 0
    for p in X.day_paths():
        c = X.connect_ro(p)
        n = c.execute(f"SELECT COUNT(*) FROM bfi WHERE {X.SRC_COL} IN "
                      f"({','.join('?' * len(devs))})", devs).fetchone()[0]
        c.close()
        rep[p.stem] = n
        mx = max(mx, n)
    per_frame = 6 * N_SC_BINS * 4 + 16
    rep["_est_cache_total_mb"] = sum(
        v for k, v in rep.items() if not k.startswith("_")) * per_frame / 1e6
    rep["_est_peak_day_mb"] = mx * (per_frame + 2500) / 1e6
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the R-series cache.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    devs = device_macs()
    if a.dry_run:
        print(json.dumps(dict(devices=devs, rows=dry_run(devs)), indent=2))
        return 0
    t0 = time.time()
    for p in X.day_paths():
        t = time.time()
        build_day(p, devs, force=a.force)
        print(f"[cache] {p.stem} ({time.time() - t:.1f}s)", file=sys.stderr)
        X.log_rss(p.stem)
    for g in (SESSION_GAP_S, *SENSITIVITY_GAPS_S):
        get_windows(devs, g, force=a.force)
    print(f"[done] {time.time() - t0:.0f}s, peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
