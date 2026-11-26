#!/usr/bin/env python3
"""Person identification on REAL beamforming feedback (CSI-BFI-HAR) with the
paper's observer: no calibration, one labelled session enrolled, Euclidean
1-NN on z-scored Givens-angle window features.

Data: CSI-BFI-HAR (Haque et al., GPL-3.0), BFI pcapng traces of 802.11ac
compressed beamforming reports. Files X_D_YY_ZZ: X activity (A jogging,
H standing, J reading a book, L walking), D day, YY BFI device, ZZ subject.
HAR-1 = kitchen (LoS, day 1, M1+M2), HAR-3 = classroom (LoS, day 3, M1+M2;
M1's P3 files are empty, so M2 is the HAR-3 device).

Features: exactly r_common (first column of V: phi_11, phi_21, psi_21,
psi_31; 32 equal-width tone bins over the 234 reported subcarriers;
per 32-frame window circular mean + spread per phi, mean + std per psi;
320 dims). Decoder: angle_decode.py (validated bit-exact on Wi-BFI).

Observer (r_eval "angles"): z-score fitted label-free on all windows of the
condition, Euclidean 1-NN over the K enrolled windows (K = all means every
window of the enrolled file), ties uniformly at random, T = 1.
"format": 1-NN on a one-hot of the report format (ties random): chance when
every file shares one format. "angles_rq": the same 1-NN on the
format-blind copy (angles re-quantised to 4-bit phi / 2-bit psi).

Conditions
  same_room    enrol one activity file per subject in one (room, device)
               set; query every OTHER activity of the same set.
  cross_room   enrol one activity file per subject in set A; query ALL
               activities of set B (rows carry the query activity so the
               same-activity subset can be read off).

Memory: pcapng blocks are streamed from a buffered reader, one file at a
time; bodies are decoded in DECODE_CHUNK groups; per-file per-frame
features go to a compact npz cache; only window vectors cross files.
RLIMIT_AS is capped; peak RSS is logged.

Outputs (figures/): har_person_test.csv, har_person_test_summary.json.
Run with ``python3 -I`` (the pcapng files are untrusted downloads).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import r_common as R      # noqa: E402  (paper feature definition)
import xres_common as X   # noqa: E402  (decoder module AD, _angle_to_rad)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("BFI_HAR_DATA_DIR",
                               REPO_ROOT / "data" / "csi_bfi_har"))
CACHE_DIR = Path(os.environ.get("BFI_HAR_CACHE_DIR",
                                REPO_ROOT / "data" / "csi_bfi_har_cache"))
FIG_DIR = Path(os.environ.get("BFI_HAR_FIG_DIR", REPO_ROOT / "figures"))
OUT_CSV = FIG_DIR / "har_person_test.csv"
OUT_JSON = FIG_DIR / "har_person_test_summary.json"
CACHE_VERSION = 1

VM_LIMIT_BYTES = int(os.environ.get("BFI_HAR_VM_LIMIT",
                                    str(8 * 1024 ** 3)))
RSS_WARN_FRACTION = 0.5

# Dataset layout actually downloaded (README: P1-P6, files: P1-P3 only).
SUBJECTS = ("P1", "P2", "P3")
ACTIVITIES = ("A", "H", "J", "L")
ACTIVITY_NAMES = {"A": "jogging", "H": "standing", "J": "reading",
                  "L": "walking"}
# (set name, folder, day, device, room)
SETS = {
    "K1-M1": dict(folder="HAR-1/BFI/M1", day="1", dev="M1",
                  room="kitchen"),
    "K1-M2": dict(folder="HAR-1/BFI/M2", day="1", dev="M2",
                  room="kitchen"),
    "C3-M2": dict(folder="HAR-3/BFI/M2", day="3", dev="M2",
                  room="classroom"),
}
SAME_ROOM_SETS = ("K1-M1", "K1-M2", "C3-M2")
CROSS_ROOM_PAIRS = (("K1-M1", "C3-M2"), ("C3-M2", "K1-M1"),
                    ("K1-M2", "C3-M2"), ("C3-M2", "K1-M2"))

# Observer
WINDOW_FRAMES = R.WINDOW_FRAMES              # 32
K_GRID = (1, 16, 0)                          # 0 = all windows of the file
T_QUERY = 1
N_TRIALS = 50
SEED = 20261024
TIE_TOL = 1e-9
Z_EPS = 1e-6
FEATURES = ("angles", "angles_centroid", "angles_rq", "format")
SECONDS_PER_HOUR = 3600.0

# pcapng / 802.11 parsing
PCAPNG_SHB = 0x0A0D0D0A
PCAPNG_IDB = 0x00000001
PCAPNG_EPB = 0x00000006
PCAPNG_OPT_END = 0
PCAPNG_OPT_TSRESOL = 9
PCAPNG_DEFAULT_TSRESOL = 6             # microseconds
LINKTYPE_RADIOTAP = 127
RADIOTAP_LEN_OFFSET = 2
MAC_HDR_LEN = 24
FC_ORDER_BIT = 0x80                    # HT Control present (+4 bytes)
HT_CONTROL_LEN = 4
FC_ACTION = 0xD0
FC_ACTION_NOACK = 0xE0
ACTION_FRAMES = (FC_ACTION, FC_ACTION_NOACK)
VHT_ACTION_COMPRESSED_BF = 0
DECODE_CHUNK = 2000                    # bodies per vectorised decode
MIN_BLOCK_LEN = 12
FEATURE_DIM = R.FEATURE_DIM


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def set_vm_limit(limit: int = VM_LIMIT_BYTES) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new, hard))


def _status_mb(field: str) -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def peak_rss_mb() -> float:
    return _status_mb("VmHWM")


def log_rss(tag: str) -> float:
    pk = peak_rss_mb()
    warn = (" WARNING >50% RAM"
            if pk > RSS_WARN_FRACTION * X.mem_total_mb() else "")
    print(f"[rss] {tag}: peak {pk:.0f} MB{warn}", file=sys.stderr)
    return pk


# ---------------------------------------------------------------------------
# pcapng streaming
# ---------------------------------------------------------------------------
def iter_pcapng_frames(path: Path):
    """Yield (ts_seconds, link_type, packet_bytes) per Enhanced Packet
    Block, streaming block by block."""
    tsres = {}
    link = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            btype, blen = struct.unpack("<II", hdr)
            if blen < MIN_BLOCK_LEN:
                return
            body = f.read(blen - 8)
            if len(body) < blen - 8:
                return
            if btype == PCAPNG_IDB:
                lt, _res, _snap = struct.unpack_from("<HHI", body, 0)
                idx = len(link)
                link[idx] = lt
                tsres[idx] = PCAPNG_DEFAULT_TSRESOL
                o = 8
                while o + 4 <= len(body) - 4:
                    code, ln = struct.unpack_from("<HH", body, o)
                    if code == PCAPNG_OPT_END:
                        break
                    if code == PCAPNG_OPT_TSRESOL and ln >= 1:
                        v = body[o + 4]
                        tsres[idx] = v  # bit 7 set would mean base-2; not seen
                    o += 4 + ((ln + 3) & ~3)
            elif btype == PCAPNG_EPB:
                iface, tsh, tsl, caplen, _orig = struct.unpack_from(
                    "<IIIII", body, 0)
                raw = body[20:20 + caplen]
                res = tsres.get(iface, PCAPNG_DEFAULT_TSRESOL)
                if res & 0x80:
                    ts = ((tsh << 32) | tsl) / float(2 ** (res & 0x7F))
                else:
                    ts = ((tsh << 32) | tsl) / float(10 ** res)
                yield ts, link.get(iface), raw


def action_body(pkt: bytes, link_type: int):
    """Strip radiotap + MAC header; return (fc0, ta, ra, body) or None."""
    if link_type != LINKTYPE_RADIOTAP or len(pkt) < 4:
        return None
    rt_len = struct.unpack_from("<H", pkt, RADIOTAP_LEN_OFFSET)[0]
    fr = pkt[rt_len:]
    if len(fr) < MAC_HDR_LEN + 2:
        return None
    fc0, fc1 = fr[0], fr[1]
    if fc0 not in ACTION_FRAMES:
        return None
    hl = MAC_HDR_LEN + (HT_CONTROL_LEN if fc1 & FC_ORDER_BIT else 0)
    return fc0, fr[10:16].hex(":"), fr[4:10].hex(":"), fr[hl:]


# ---------------------------------------------------------------------------
# Per-file decode and cache
# ---------------------------------------------------------------------------
def file_tag(set_name: str, act: str, subj: str) -> str:
    s = SETS[set_name]
    return f"{act}_{s['day']}_{s['dev']}_{subj}"


def file_path(set_name: str, act: str, subj: str) -> Path:
    return DATA_DIR / SETS[set_name]["folder"] / (
        file_tag(set_name, act, subj) + ".pcapng")


def cache_path(tag: str) -> Path:
    return CACHE_DIR / f"{tag}.v{CACHE_VERSION}.npz"


def _flush(groups, bodies, gdict, feat, feat_rq, nsub_len, nsub_dec):
    for (key, L), idx in groups.items():
        g = gdict[key]
        res = R.decode_first_column([bodies[i] for i in idx], g)
        if res is None:
            continue
        ia = np.asarray(idx)
        feat[ia] = R.bin_frames(res[0])
        feat_rq[ia] = R.bin_frames(res[2])
        bits_per_sc = X.AD._layout(g["nr"], g["nc"], g["bphi"],
                                   g["bpsi"])[3]
        n_len = ((L - (2 + g["mimo_len"] + g["nc"])) * 8) // bits_per_sc
        nsub_len[n_len] += len(idx)
        nsub_dec[res[0].shape[2]] += len(idx)


def build_file(path: Path, tag: str, force: bool = False) -> dict:
    """Decode one pcapng into the per-frame feature cache; return meta."""
    out = cache_path(tag)
    if out.exists() and not force:
        z = np.load(out, allow_pickle=False)
        return json.loads(str(z["meta"]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pkt = n_action = n_bf = n_other_cat = 0
    fc_c: Counter = Counter()
    ta_c: Counter = Counter()
    ra_c: Counter = Counter()
    geom_c: Counter = Counter()
    blen_c: Counter = Counter()
    nsub_len: Counter = Counter()
    nsub_dec: Counter = Counter()
    geoms: dict[tuple, int] = {}
    gdict: dict[tuple, dict] = {}
    ts_l, gid_l, feat_l, feat_rq_l = [], [], [], []
    ts_c, gid_c, bodies, groups = [], [], [], defaultdict(list)
    first_ts = last_ts = None

    def close_chunk():
        n = len(bodies)
        if n == 0:
            return
        feat = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
        feat_rq = np.full((n, 6, R.N_SC_BINS), np.nan, np.float32)
        _flush(groups, bodies, gdict, feat, feat_rq, nsub_len, nsub_dec)
        ok = ~np.isnan(feat[:, 0, 0])
        ts_l.append(np.asarray(ts_c)[ok])
        gid_l.append(np.asarray(gid_c, np.int16)[ok])
        feat_l.append(feat[ok])
        feat_rq_l.append(feat_rq[ok])
        bodies.clear(); groups.clear(); ts_c.clear(); gid_c.clear()

    for ts, lt, pkt in iter_pcapng_frames(path):
        n_pkt += 1
        first_ts = ts if first_ts is None else first_ts
        last_ts = ts
        ab = action_body(pkt, lt)
        if ab is None:
            if lt == LINKTYPE_RADIOTAP and len(pkt) > 4:
                rt_len = struct.unpack_from("<H", pkt, RADIOTAP_LEN_OFFSET)[0]
                if len(pkt) > rt_len:
                    fc_c[f"0x{pkt[rt_len]:02x}"] += 1
            continue
        fc0, ta, ra, body = ab
        n_action += 1
        fc_c[f"0x{fc0:02x}"] += 1
        g = X.AD.parse_geometry(body)
        if g is None or (g["fmt"] == "VHT"
                         and body[1] != VHT_ACTION_COMPRESSED_BF):
            n_other_cat += 1
            continue
        n_bf += 1
        ta_c[ta] += 1
        ra_c[ra] += 1
        key = tuple(g[k] for k in R.GEOM_FIELDS)
        geom_c[json.dumps(list(key))] += 1
        blen_c[len(body)] += 1
        gid_c.append(geoms.setdefault(key, len(geoms)))
        gdict[key] = g
        ts_c.append(ts)
        bodies.append(body)
        groups[(key, len(body))].append(len(bodies) - 1)
        if len(bodies) >= DECODE_CHUNK:
            close_chunk()
    close_chunk()
    ts_a = np.concatenate(ts_l) if ts_l else np.zeros(0)
    order = np.argsort(ts_a, kind="stable")
    arr = dict(ts=ts_a[order],
               gid=(np.concatenate(gid_l) if gid_l else np.zeros(0, np.int16))[order],
               feat=(np.concatenate(feat_l) if feat_l
                     else np.zeros((0, 6, R.N_SC_BINS), np.float32))[order],
               feat_rq=(np.concatenate(feat_rq_l) if feat_rq_l
                        else np.zeros((0, 6, R.N_SC_BINS), np.float32))[order])
    gaps = np.diff(arr["ts"]) if arr["ts"].size > 1 else np.zeros(1)
    meta = dict(
        file=str(path.relative_to(DATA_DIR)), bytes=path.stat().st_size,
        packets=n_pkt, action_frames=n_action, bf_reports=n_bf,
        non_bf_action=n_other_cat, decoded=int(arr["ts"].size),
        frame_control=dict(fc_c), ta=dict(ta_c), ra=dict(ra_c),
        geometries=dict(geom_c), body_lengths={str(k): v for k, v in blen_c.items()},
        n_sub_from_length={str(k): v for k, v in nsub_len.items()},
        n_sub_decoded={str(k): v for k, v in nsub_dec.items()},
        ts_first=first_ts, ts_last=last_ts,
        duration_s=(float(arr["ts"][-1] - arr["ts"][0]) if arr["ts"].size > 1
                    else 0.0),
        gap_median_s=float(np.median(gaps)), gap_max_s=float(gaps.max()),
        windows=int(arr["ts"].size // WINDOW_FRAMES),
        geom_table=[list(k) for k in geoms],
        decode_s=round(time.time() - t0, 2))
    np.savez(out, **arr, meta=np.asarray(json.dumps(meta)))
    del arr, ts_l, gid_l, feat_l, feat_rq_l
    gc.collect()
    return meta


def load_windows(tag: str) -> dict:
    """Non-overlapping WINDOW_FRAMES windows of one file -> dict of arrays."""
    z = np.load(cache_path(tag), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = z["ts"].size // WINDOW_FRAMES
    feat, feat_rq, gid, ts = z["feat"], z["feat_rq"], z["gid"], z["ts"]
    F = np.empty((n, FEATURE_DIM), np.float32)
    Frq = np.empty((n, FEATURE_DIM), np.float32)
    geom = np.empty(n, np.int16)
    ts0 = np.empty(n)
    ts1 = np.empty(n)
    gt = [json.dumps(g) for g in meta["geom_table"]]
    gkeys = []
    for w in range(n):
        jj = slice(w * WINDOW_FRAMES, (w + 1) * WINDOW_FRAMES)
        F[w] = R.window_feature(feat[jj])
        Frq[w] = R.window_feature(feat_rq[jj])
        g, c = np.unique(gid[jj], return_counts=True)
        geom[w] = g[np.argmax(c)]
        ts0[w], ts1[w] = ts[jj][0], ts[jj][-1]
    gkeys = [gt[int(g)] for g in geom]
    return dict(F=F, Frq=Frq, geom_key=np.asarray(gkeys), ts0=ts0, ts1=ts1)


# ---------------------------------------------------------------------------
# Window table
# ---------------------------------------------------------------------------
class Table:
    def __init__(self):
        self.F, self.Frq, self.gk = [], [], []
        self.set_, self.act, self.subj, self.ts0, self.ts1 = [], [], [], [], []

    def add(self, set_name: str, act: str, subj: str, w: dict) -> None:
        n = w["F"].shape[0]
        self.F.append(w["F"]); self.Frq.append(w["Frq"])
        self.gk.append(w["geom_key"])
        self.set_ += [set_name] * n; self.act += [act] * n
        self.subj += [subj] * n
        self.ts0.append(w["ts0"]); self.ts1.append(w["ts1"])

    def finish(self) -> "Table":
        self.F = np.concatenate(self.F); self.Frq = np.concatenate(self.Frq)
        self.gk = np.concatenate(self.gk)
        self.set_ = np.asarray(self.set_); self.act = np.asarray(self.act)
        self.subj = np.asarray(self.subj)
        self.ts0 = np.concatenate(self.ts0); self.ts1 = np.concatenate(self.ts1)
        keys = sorted(set(self.gk.tolist()))
        self.Fmt = np.zeros((self.F.shape[0], len(keys)), np.float32)
        for j, k in enumerate(keys):
            self.Fmt[self.gk == k, j] = 1.0
        self.format_keys = keys
        return self


def zscore(Fs: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    mu = Fs[fit_mask].mean(0)
    sd = Fs[fit_mask].std(0)
    return ((Fs - mu) / np.maximum(sd, Z_EPS)).astype(np.float32)


def sqdist(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    """(nq, ns) squared Euclidean distances via the matmul identity."""
    q2 = (Q.astype(np.float64) ** 2).sum(1)[:, None]
    s2 = (S.astype(np.float64) ** 2).sum(1)[None, :]
    d = q2 + s2 - 2.0 * Q.astype(np.float64) @ S.astype(np.float64).T
    return np.maximum(d, 0.0)


def argmin_random(D: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmin with uniform random tie-breaking. D: (n, g)."""
    m = D.min(1, keepdims=True)
    ties = D <= m + TIE_TOL
    r = rng.random(D.shape) * ties
    return np.argmax(r, axis=1)


def klabel(k: int) -> str:
    return "all" if k == 0 else str(k)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
def run_episode(tb: Table, Z: dict, enrol_mask_by_subj: dict,
                query_mask: np.ndarray, rng: np.random.Generator,
                rows_acc: dict, cond: str, enrol_set: str, query_set: str,
                enrol_act: str) -> None:
    """One enrolment episode: for every trial draw a permutation of each
    subject's enrolled windows; classify every query window for every K and
    feature; accumulate hits per (feature, K, true subject, query activity)
    and the confusion counts."""
    gal = list(SUBJECTS)
    qi = np.flatnonzero(query_mask)
    q_subj = tb.subj[qi]
    q_act = tb.act[qi]
    y_true = np.array([gal.index(s) for s in q_subj])
    for trial in range(N_TRIALS):
        perms = {g: rng.permutation(np.flatnonzero(enrol_mask_by_subj[g]))
                 for g in gal}
        n_enr = {g: perms[g].size for g in gal}
        # full distance matrices once per trial, per feature
        D = {}
        for feat, key in (("angles", "Z"), ("angles_rq", "Zrq"),
                          ("format", "Fmt")):
            D[feat] = {g: sqdist(Z[key][qi], Z[key][perms[g]]) for g in gal}
        for K in K_GRID:
            kk = {g: n_enr[g] if K == 0 else min(K, n_enr[g]) for g in gal}
            for feat in FEATURES:
                if feat == "angles_centroid":
                    cen = np.stack([Z["Z"][perms[g][:kk[g]]].mean(0)
                                    for g in gal])
                    dd = sqdist(Z["Z"][qi], cen)
                else:
                    dd = np.stack([D[feat][g][:, :kk[g]].min(1) for g in gal],
                                  axis=1)
                pred = argmin_random(dd, rng)
                hit = pred == y_true
                for s_idx, s in enumerate(gal):
                    m = y_true == s_idx
                    for a in np.unique(q_act[m]):
                        ma = m & (q_act == a)
                        acc = rows_acc[(cond, enrol_set, query_set, enrol_act,
                                        feat, K, s, a)]
                        acc["hits"] += int(hit[ma].sum())
                        acc["n"] += int(ma.sum())
                        acc["trials"] += 1
                        acc["capped"] += int(K != 0 and n_enr[s] < K)
                        for p_idx in range(len(gal)):
                            acc["conf"][p_idx] += int((pred[ma] == p_idx).sum())
    # K = all is deterministic up to tie-breaking; it is repeated per trial
    # so that its accumulator shares the same denominator (N_TRIALS * n).


def new_acc():
    return dict(hits=0, n=0, trials=0, capped=0, conf=[0] * len(SUBJECTS))


def run_same_room(tb: Table, set_name: str, rng, rows_acc) -> None:
    in_set = tb.set_ == set_name
    Z = dict(Z=zscore(tb.F, in_set), Zrq=zscore(tb.Frq, in_set), Fmt=tb.Fmt)
    for ea in ACTIVITIES:
        enrol = {s: in_set & (tb.act == ea) & (tb.subj == s) for s in SUBJECTS}
        if any(m.sum() == 0 for m in enrol.values()):
            print(f"[skip] {set_name} enrol {ea}: a subject has no windows",
                  file=sys.stderr)
            continue
        query = in_set & (tb.act != ea)
        run_episode(tb, Z, enrol, query, rng, rows_acc, "same_room",
                    set_name, set_name, ea)


def run_cross_room(tb: Table, enrol_set: str, query_set: str, rng,
                   rows_acc) -> None:
    both = (tb.set_ == enrol_set) | (tb.set_ == query_set)
    Z = dict(Z=zscore(tb.F, both), Zrq=zscore(tb.Frq, both), Fmt=tb.Fmt)
    for ea in ACTIVITIES:
        enrol = {s: (tb.set_ == enrol_set) & (tb.act == ea) & (tb.subj == s)
                 for s in SUBJECTS}
        if any(m.sum() == 0 for m in enrol.values()):
            print(f"[skip] {enrol_set}->{query_set} enrol {ea}: a subject has "
                  f"no windows", file=sys.stderr)
            continue
        query = tb.set_ == query_set
        run_episode(tb, Z, enrol, query, rng, rows_acc, "cross_room",
                    enrol_set, query_set, ea)


def file_pair_distances(tb: Table, set_name: str) -> list[dict]:
    """Confound check: for every ordered pair of files in one set, the
    median over the query file's windows of the 1-NN distance (z-scored
    angles) to the other file's windows, with the recording-time gap and
    the same-subject / same-activity flags. If same-subject pairs and
    different-subject pairs do not overlap in time gap, identity and
    recording block cannot be separated on this dataset."""
    in_set = tb.set_ == set_name
    Z = zscore(tb.F, in_set)
    files = sorted({(a, s) for a, s in zip(tb.act[in_set], tb.subj[in_set])})
    out = []
    for a1, s1 in files:
        m1 = in_set & (tb.act == a1) & (tb.subj == s1)
        for a2, s2 in files:
            if (a1, s1) == (a2, s2):
                continue
            m2 = in_set & (tb.act == a2) & (tb.subj == s2)
            d = np.sqrt(sqdist(Z[m2], Z[m1]).min(1))
            out.append(dict(
                set=set_name, enrol=f"{a1}_{s1}", query=f"{a2}_{s2}",
                dt_hours=float(abs(tb.ts0[m2].min() - tb.ts0[m1].min()))
                / SECONDS_PER_HOUR,
                same_subject=bool(s1 == s2), same_activity=bool(a1 == a2),
                median_nn_dist=float(np.median(d))))
    return out


def pair_confound_table(pairs: list[dict]) -> dict:
    """Time-gap range and mean distance for same- vs different-subject
    pairs, per set."""
    rep = {}
    for same in (True, False):
        sel = [p for p in pairs if p["same_subject"] == same]
        if not sel:
            continue
        rep["same_subject" if same else "different_subject"] = dict(
            n_pairs=len(sel),
            dt_hours_min=min(p["dt_hours"] for p in sel),
            dt_hours_max=max(p["dt_hours"] for p in sel),
            median_nn_dist_mean=float(np.mean([p["median_nn_dist"]
                                               for p in sel])))
    return rep


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------
def rows_from_acc(rows_acc: dict) -> list[dict]:
    rows = []
    for (cond, es, qs, ea, feat, K, s, a), v in sorted(rows_acc.items(),
                                                        key=str):
        rows.append(dict(
            condition=cond, enrol_set=es, query_set=qs, enrol_activity=ea,
            query_activity=a, feature=feat, K=klabel(K), T=T_QUERY,
            subject=s, n_query_windows=v["n"] // max(v["trials"], 1),
            trials=v["trials"], recall=v["hits"] / v["n"] if v["n"] else float("nan"),
            chance=1.0 / len(SUBJECTS), capped_fraction=v["capped"] / max(v["trials"], 1),
            **{f"pred_{p}": c / v["n"] if v["n"] else float("nan")
               for p, c in zip(SUBJECTS, v["conf"])}))
    return rows


def pooled(rows: list[dict], cond: str, feat: str, K: str,
           sets: tuple | None = None, same_activity: bool | None = None,
           by_subject: bool = False) -> dict:
    """Pool hits over enrol activities (and query activities) with the
    window-weighted mean; returns recall overall or per subject."""
    acc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r["condition"] != cond or r["feature"] != feat or r["K"] != K:
            continue
        if sets is not None and (r["enrol_set"], r["query_set"]) != sets:
            continue
        if same_activity is True and r["query_activity"] != r["enrol_activity"]:
            continue
        if same_activity is False and r["query_activity"] == r["enrol_activity"]:
            continue
        key = r["subject"] if by_subject else "all"
        acc[key][0] += r["recall"] * r["n_query_windows"]
        acc[key][1] += r["n_query_windows"]
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}


def main() -> int:
    global N_TRIALS  # noqa: PLW0603  (CLI override of the module default)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list files, sizes and the memory estimate; no decode")
    ap.add_argument("--force", action="store_true", help="rebuild caches")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    N_TRIALS = a.trials
    set_vm_limit()
    t0 = time.time()
    files = [(sn, act, subj) for sn in SETS for act in ACTIVITIES
             for subj in SUBJECTS]
    sizes = {}
    for sn, act, subj in files:
        p = file_path(sn, act, subj)
        sizes[str(p.relative_to(DATA_DIR))] = p.stat().st_size if p.exists() else -1
    missing = [k for k, v in sizes.items() if v < 0]
    if missing:
        print("missing files:", missing, file=sys.stderr)
        return 2
    tot = sum(sizes.values())
    # body ~1 kB per frame; per frame cache 6*32*4*2 B; decode chunk bits
    est_peak = (DECODE_CHUNK * 1100 * 8 * 2 + max(sizes.values()) * 2) / 1e6
    print(f"[files] {len(files)} pcapng, {tot / 1e6:.1f} MB total; "
          f"largest {max(sizes.values()) / 1e6:.1f} MB; est. peak decode "
          f"{est_peak:.0f} MB above baseline", file=sys.stderr)
    if a.dry_run:
        print(json.dumps(sizes, indent=1))
        return 0

    metas = {}
    tb = Table()
    for sn, act, subj in files:
        tag = file_tag(sn, act, subj)
        metas[tag] = build_file(file_path(sn, act, subj), tag, force=a.force)
        tb.add(sn, act, subj, load_windows(tag))
        print(f"[file] {tag}: {metas[tag]['bf_reports']} reports, "
              f"{metas[tag]['windows']} windows, {metas[tag]['decode_s']}s",
              file=sys.stderr)
        log_rss(tag)
    tb.finish()
    gc.collect()
    log_rss("table built")

    rng = np.random.default_rng(SEED)
    rows_acc: dict = defaultdict(new_acc)
    for sn in SAME_ROOM_SETS:
        run_same_room(tb, sn, rng, rows_acc)
        log_rss(f"same_room {sn}")
    for es, qs in CROSS_ROOM_PAIRS:
        run_cross_room(tb, es, qs, rng, rows_acc)
        log_rss(f"cross_room {es}->{qs}")
    rows = rows_from_acc(rows_acc)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary tables
    summary = dict(
        dataset="CSI-BFI-HAR (foysalhaque/CSI-BFI-HAR-Dataset, GPL-3.0)",
        sets=SETS, subjects=SUBJECTS, activities=ACTIVITY_NAMES,
        window_frames=WINDOW_FRAMES, k_grid=[klabel(k) for k in K_GRID],
        T=T_QUERY, trials=N_TRIALS, seed=SEED, features=FEATURES,
        chance=1.0 / len(SUBJECTS),
        download_bytes=sizes, download_total_mb=round(tot / 1e6, 1),
        files=metas,
        formats_seen=sorted(set(k for m in metas.values()
                                for k in m["geometries"])),
        ta_macs=sorted(set(k for m in metas.values() for k in m["ta"])),
        ra_macs=sorted(set(k for m in metas.values() for k in m["ra"])),
        windows_per_file={t: m["windows"] for t, m in metas.items()},
        recording_order={
            sn: [t for t, m in sorted(metas.items(),
                                      key=lambda kv: kv[1]["ts_first"])
                 if f"_{SETS[sn]['day']}_{SETS[sn]['dev']}_" in t]
            for sn in SETS},
        tables={})
    T = summary["tables"]
    for sn in SAME_ROOM_SETS:
        T[f"same_room/{sn}"] = {
            f"{feat}/K={K}": dict(
                overall=pooled(rows, "same_room", feat, K, (sn, sn))["all"],
                per_subject=pooled(rows, "same_room", feat, K, (sn, sn),
                                   by_subject=True))
            for feat in FEATURES for K in map(klabel, K_GRID)}
        T[f"same_room/{sn}"]["per_enrol_activity/angles"] = {
            f"enrol={ea}/K={K}": float(np.average(
                [r["recall"] for r in rows if r["condition"] == "same_room"
                 and r["enrol_set"] == sn and r["enrol_activity"] == ea
                 and r["feature"] == "angles" and r["K"] == K],
                weights=[r["n_query_windows"] for r in rows
                         if r["condition"] == "same_room"
                         and r["enrol_set"] == sn and r["enrol_activity"] == ea
                         and r["feature"] == "angles" and r["K"] == K]))
            for ea in ACTIVITIES for K in map(klabel, K_GRID)}
    for es, qs in CROSS_ROOM_PAIRS:
        T[f"cross_room/{es}->{qs}"] = {
            f"{feat}/K={K}": dict(
                all_activities=pooled(rows, "cross_room", feat, K, (es, qs))["all"],
                same_activity=pooled(rows, "cross_room", feat, K, (es, qs),
                                     same_activity=True)["all"],
                other_activity=pooled(rows, "cross_room", feat, K, (es, qs),
                                      same_activity=False)["all"],
                per_subject=pooled(rows, "cross_room", feat, K, (es, qs),
                                   by_subject=True))
            for feat in FEATURES for K in map(klabel, K_GRID)}
    pairs = {sn: file_pair_distances(tb, sn) for sn in SAME_ROOM_SETS}
    summary["file_pair_distances"] = pairs
    summary["block_confound"] = {sn: pair_confound_table(v)
                                 for sn, v in pairs.items()}
    summary["runtime_s"] = round(time.time() - t0, 1)
    summary["peak_rss_mb"] = round(peak_rss_mb(), 1)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"[done] {summary['runtime_s']}s, peak RSS {summary['peak_rss_mb']}"
          f" MB -> {OUT_CSV}, {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
