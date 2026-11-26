#!/usr/bin/env python3
"""Shared helpers for the revision experiments X4, X5, X8 on the
residential BFI archive (Pi 5 sniffer, one stock router, 2026-05-08..20).

Two feature families are carried side by side so every new number can be
read against the paper's existing residential numbers:

* PAPER features (power / capability / body) are the exact per-frame
  features of ~/bfi-privacy/tools/run_pi_one_shot.py, the code that
  produced fig_q1_fullwindow (pi_one_shot_full_window.py) and
  bfi_rebuttal_crossday.csv (bfi/demo_rebuttal_crossday.py). NOTE: that
  pipeline's "body" channel is a 64-bin histogram of the raw Action-body
  BYTES (``body_proxy``); it never decodes Givens angles.
* GIVENS features decode the quantised phi/psi angles with the
  externally validated decoder (esp32-csi-tomography
  backend/src/core/bfi/angle_decode.py, bit-exact against the four
  Wi-BFI reference traces) and reduce them to a layout-agnostic common
  representation: the leading coordinate pair (phi_11, psi_21), converted
  to radians and averaged into N_SC_BINS equal-width bins across the band.
  The same representation is computed for the He dataset (antenna pair
  01), which is what makes the cross-source calibration in X5 possible.

Memory: the archive is streamed one day at a time with a sqlite cursor
(fetchmany), only the needed columns are selected, per-frame features are
written to a compact per-day .npz cache, and nothing but the cache is kept
across days. ``set_vm_limit`` caps the address space so a runaway turns
into a MemoryError instead of an OOM kill.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import resource
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (env-overridable, no absolute paths)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(os.environ.get("BFI_X_FIG_DIR", REPO_ROOT / "figures"))
ARCHIVE_DIR = Path(os.environ.get(
    "BFI_ARCHIVE_DIR", Path.home() / "bfi-archive" / "data"))
CACHE_DIR = Path(os.environ.get(
    "BFI_X_CACHE_DIR", Path.home() / ".cache" / "bfi-x-experiments"))
BFI_PRIVACY_DIR = Path(os.environ.get(
    "BFI_PRIVACY_DIR", Path.home() / "bfi-privacy"))
# Sibling-repository code: the copies under release/ first, the sibling
# checkouts named by the environment variables behind them.
import vendor_paths as VP  # noqa: E402
ANGLE_DECODE_PY = Path(os.environ.get(
    "BFI_ANGLE_DECODE_PY", VP.vendored_or(
        VP.VENDOR_DECODER,
        Path.home() / "esp32-csi-tomography" / "backend" / "src" / "core"
        / "bfi" / "angle_decode.py")))
HE_ROOT = Path(os.environ.get(
    "BFI_HE_ROOT", REPO_ROOT / "data" / "he_bfi" / "dataset"
    / "transfer and light weight"))

# Capture window used by the paper (05-08..05-20; 05-12/13 are empty files).
DAY_FIRST = os.environ.get("BFI_X_DAY_FIRST", "2026-05-08")
DAY_LAST = os.environ.get("BFI_X_DAY_LAST", "2026-05-20")

# Released (anonymised) archive: set BFI_LABELS to the device_labels.csv of
# the release and BFI_ARCHIVE_DIR to its folder. The release carries salted
# ids (src_id, ap_id) and RSSI in dBm (rssi; rssi_bin, in 5 dB bins, in files
# written before the paper's release); the six devices and
# the locally-administered flag then come from the CSV instead of the OUI
# table and the first address byte. Unset (default): the raw archive.
LABELS_CSV = os.environ.get("BFI_LABELS")
ANON = LABELS_CSV is not None
RSSI_EXACT_COL, RSSI_BINNED_COL = "rssi", "rssi_bin"
ARCHIVE_GLOB = "bfi_*.sqlite"


def _release_rssi_col() -> str:
    """The RSSI column of the released archive: exact dBm (`rssi`) in the
    release that accompanies the paper, 5 dB bins (`rssi_bin`) in the files
    written before it. Read from the first day file; exact when none exists."""
    for p in sorted(ARCHIVE_DIR.glob(ARCHIVE_GLOB)):
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in c.execute("PRAGMA table_info(bfi)")}
        finally:
            c.close()
        return RSSI_EXACT_COL if RSSI_EXACT_COL in cols else RSSI_BINNED_COL
    return RSSI_EXACT_COL


SRC_COL, AP_COL, RSSI_COL = (("src_id", "ap_id", _release_rssi_col()) if ANON
                             else ("src_mac", "ap_mac", RSSI_EXACT_COL))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FETCH_CHUNK = 5000              # sqlite rows per fetchmany
N_SC_BINS = 32                  # subcarrier bins of the common Givens rep
HIST_BINS = 64                  # run_pi_one_shot.DEFAULT_BODY_PROXY_DIM
HIST_SHIFT = 2                  # byte >> 2 == histogram bin for 64 bins
SNR_KEEP = 2                    # frame_features keeps the first 2 SNR bytes
CACHE_VERSION = 2
VM_LIMIT_BYTES = int(os.environ.get("BFI_X_VM_LIMIT", str(8 * 1024 ** 3)))
RSS_WARN_FRACTION = 0.5
FB_SU, FB_OTHER, FB_NULL = 1, 0, -1

# Ground truth from the paper (OUI prefix -> device, mobility group).
DEVICE_GT_OUI = {
    "8c:26:aa": ("Apple TV (living room)", "stationary"),
    "d0:d2:b0": ("Apple TV (bedroom)", "stationary"),
    "b8:b4:09": ("Samsung TV", "stationary"),
    "54:b7:bd": ("LG TV", "stationary"),
    "64:4e:d7": ("Laptop", "mobile"),
    "4c:b9:ea": ("iRobot vacuum", "mobile"),
}
# The order in which the raw archive sorts the six addresses (by OUI). The
# device index of every script follows it in both modes, so the seeded draws
# of the evaluations are the same on the raw and on the released archive.
DEVICE_ORDER = tuple(lab for _, (lab, _) in sorted(DEVICE_GT_OUI.items()))
# Label files written before the correction called the LG TV "relocated".
# It never moved, so both spellings load as the same stationary device.
LEGACY_LABELS = {"LG TV (relocated)": "LG TV"}
LEGACY_MOBILITY = {"relocated": "stationary"}


def _read_labels(path: str) -> tuple[dict, set]:
    """device_labels.csv of the release -> ({id: (label, mobility)} for the
    rows that carry a label, {ids with random_mac = 1})."""
    import csv
    gt, rnd = {}, set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("label"):
                gt[r["dev_id"]] = (LEGACY_LABELS.get(r["label"], r["label"]),
                                   LEGACY_MOBILITY.get(r["mobility"], r["mobility"]))
            if int(r["random_mac"]):
                rnd.add(r["dev_id"])
    missing = set(DEVICE_ORDER) - {v[0] for v in gt.values()}
    if missing:
        raise RuntimeError(f"{path}: no id labelled {sorted(missing)}")
    return gt, rnd


if ANON:
    DEVICE_GT, RANDOM_IDS = _read_labels(LABELS_CSV)
else:
    DEVICE_GT, RANDOM_IDS = DEVICE_GT_OUI, set()


def device_key(addr: str) -> str:
    """The key of DEVICE_GT for an address: OUI prefix (raw) or the id."""
    return addr if ANON else addr[:8]


def is_random(addr: str) -> bool:
    """Locally administered address: bit 1 of the first octet (raw), or the
    random_mac column of the release."""
    if ANON:
        return addr in RANDOM_IDS
    return bool(int(addr[:2], 16) & 0x02)


def device_rank(addr: str) -> int:
    return DEVICE_ORDER.index(DEVICE_GT[device_key(addr)][0])

# Paper protocol constants (imported to stay byte-identical).
VP.add_bfi_privacy(BFI_PRIVACY_DIR)
from run_pi_one_shot import (  # noqa: E402
    WINDOW_FRAMES, QUERIES_PER_MAC, N_TRIALS, decode_avg_snr_dB, cosine,
)
# Window length in frames; the paper uses 32, the sensitivity run 16 and 64.
WINDOW_FRAMES = int(os.environ.get("BFI_WINDOW_FRAMES", WINDOW_FRAMES))

MIN_FRAMES_PER_MAC = 2 * WINDOW_FRAMES   # the paper's 64-frame cut


def _load_angle_decode():
    spec = importlib.util.spec_from_file_location("angle_decode",
                                                  ANGLE_DECODE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AD = _load_angle_decode()


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def set_vm_limit(limit: int = VM_LIMIT_BYTES) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new, hard))


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_total_mb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def log_rss(tag: str) -> float:
    pk = peak_rss_mb()
    warn = " WARNING >50% RAM" if pk > RSS_WARN_FRACTION * mem_total_mb() \
        else ""
    print(f"[rss] {tag}: peak {pk:.0f} MB{warn}", file=sys.stderr)
    return pk


# ---------------------------------------------------------------------------
# Archive enumeration
# ---------------------------------------------------------------------------
def day_paths() -> list[Path]:
    out = []
    for p in sorted(ARCHIVE_DIR.glob("bfi_*.sqlite")):
        day = p.stem.replace("bfi_", "")
        if DAY_FIRST <= day <= DAY_LAST:
            out.append(p)
    return out


def connect_ro(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def dry_run_report() -> dict:
    """Row counts per day and the cache size they imply, without loading."""
    rep = {}
    total = 0
    for p in day_paths():
        c = connect_ro(p)
        n = c.execute("SELECT COUNT(*) FROM bfi").fetchone()[0]
        nm = c.execute(f"SELECT COUNT(DISTINCT {SRC_COL}) FROM bfi"
                       ).fetchone()[0]
        c.close()
        rep[p.stem] = dict(rows=n, macs=nm)
        total += n
    bytes_per_row = HIST_BINS * 2 + 3 * N_SC_BINS * 4 + 64
    rep["_total_rows"] = total
    rep["_est_cache_mb"] = total * bytes_per_row / 1e6
    rep["_est_peak_day_mb"] = max(
        (v["rows"] for k, v in rep.items() if not k.startswith("_")),
        default=0) * (bytes_per_row + 2500) / 1e6
    return rep


# ---------------------------------------------------------------------------
# Givens decoding (vectorised per (geometry, body length) group)
# ---------------------------------------------------------------------------
def _angle_to_rad(q: np.ndarray, nbits: int, kind: str) -> np.ndarray:
    """802.11 dequantisation: phi = k*pi/2^(b-1) + pi/2^b,
    psi = k*pi/2^(b+1) + pi/2^(b+2)."""
    q = q.astype(np.float64)
    if kind == "phi":
        return q * np.pi / 2 ** (nbits - 1) + np.pi / 2 ** nbits
    return q * np.pi / 2 ** (nbits + 1) + np.pi / 2 ** (nbits + 2)


def bin_edges(n_sub: int, n_bins: int = N_SC_BINS) -> np.ndarray:
    return (np.arange(n_sub) * n_bins) // n_sub


def bin_pair(phi: np.ndarray, psi: np.ndarray, n_bins: int = N_SC_BINS):
    """phi, psi: (n, n_sub) radians -> (cos, sin, psi) each (n, n_bins)."""
    n, n_sub = phi.shape
    b = bin_edges(n_sub, n_bins)
    cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
    out = []
    for v in (np.cos(phi), np.sin(phi), psi):
        acc = np.zeros((n, n_bins))
        for j in range(n_bins):
            acc[:, j] = v[:, b == j].mean(axis=1) if cnt[j] else np.nan
        out.append(acc.astype(np.float32))
    return out


def decode_group(bodies: list[bytes], geom: dict):
    """Decode the leading (phi_11, psi_21) pair for a list of bodies sharing
    geometry and length. Returns (cos, sin, psi, n_sub) or None."""
    offsets, widths, kinds, bits_per_sc = AD._layout(
        geom["nr"], geom["nc"], geom["bphi"], geom["bpsi"])
    if bits_per_sc == 0 or not (kinds == 1).any():
        return None                     # nr == 1: no angles to decode
    start = 2 + geom["mimo_len"] + geom["nc"]
    mat = np.frombuffer(b"".join(b[start:] for b in bodies), dtype=np.uint8)
    L = len(bodies[0]) - start
    if L <= 0:
        return None
    mat = mat.reshape(len(bodies), L)
    bits = np.unpackbits(mat, axis=1, bitorder="little")
    n_sub = bits.shape[1] // bits_per_sc
    n_exp = AD.expected_subcarriers(geom)
    if n_exp is not None:
        n_sub = min(n_sub, n_exp)
    if n_sub < N_SC_BINS:
        return None
    j_phi = int(np.flatnonzero(kinds == 0)[0])
    j_psi = int(np.flatnonzero(kinds == 1)[0])
    base = np.arange(n_sub, dtype=np.int64) * bits_per_sc
    vals = {}
    for j in (j_phi, j_psi):
        acc = np.zeros((len(bodies), n_sub), dtype=np.int32)
        for k in range(int(widths[j])):
            acc += bits[:, base + offsets[j] + k].astype(np.int32) << k
        vals[j] = acc
    phi = _angle_to_rad(vals[j_phi], geom["bphi"], "phi")
    psi = _angle_to_rad(vals[j_psi], geom["bpsi"], "psi")
    c, s, p = bin_pair(phi, psi)
    return c, s, p, n_sub


# ---------------------------------------------------------------------------
# Per-day cache builder
# ---------------------------------------------------------------------------
PAPER_COLS = ["nc", "nr", "bw_mhz", "codebook", "feedback_type",
              "avg_snr_hex"]


def cache_path(day_stem: str) -> Path:
    return CACHE_DIR / f"{day_stem}.v{CACHE_VERSION}.npz"


def build_day_cache(p: Path, force: bool = False) -> Path:
    out = cache_path(p.stem)
    if out.exists() and not force:
        return out
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c = connect_ro(p)
    cols = {r[1] for r in c.execute("PRAGMA table_info(bfi)")}
    has_parsed = set(PAPER_COLS).issubset(cols)
    sel = ["ts", SRC_COL, AP_COL, RSSI_COL, "body_hex"]
    if has_parsed:
        sel += PAPER_COLS
    cur = c.execute(f"SELECT {', '.join(sel)} FROM bfi ORDER BY ts")
    macs: dict[str, int] = {}
    aps: dict[str, int] = {}
    geoms: dict[tuple, int] = {}
    cols_out = defaultdict(list)
    while True:
        rows = cur.fetchmany(FETCH_CHUNK)
        if not rows:
            break
        n = len(rows)
        ts = np.empty(n); mac = np.empty(n, np.int16); ap = np.empty(n, np.int16)
        rssi = np.full(n, np.nan, np.float32)
        snr = np.zeros((n, SNR_KEEP), np.float32)
        cap = np.full((n, 4), -1, np.int16)
        fb = np.full(n, FB_NULL, np.int8)
        hist = np.zeros((n, HIST_BINS), np.uint16)
        blen = np.zeros(n, np.int32)
        body_null = np.zeros(n, bool)
        gid = np.full(n, -1, np.int16)
        nsub = np.zeros(n, np.int16)
        gc_ = np.full((n, N_SC_BINS), np.nan, np.float32)
        gs_ = np.full((n, N_SC_BINS), np.nan, np.float32)
        gp_ = np.full((n, N_SC_BINS), np.nan, np.float32)
        groups: dict[tuple, list[int]] = defaultdict(list)
        bodies: list[bytes | None] = [None] * n
        for i, r in enumerate(rows):
            ts[i] = r[0]
            mac[i] = macs.setdefault(r[1], len(macs))
            ap[i] = aps.setdefault(r[2], len(aps))
            if r[3] is not None:
                rssi[i] = r[3]
            if r[4] is None:
                body_null[i] = True
            else:
                raw = bytes.fromhex(r[4])
                bodies[i] = raw
                blen[i] = len(raw)
                if raw:
                    hist[i] = np.bincount(
                        np.frombuffer(raw, np.uint8) >> HIST_SHIFT,
                        minlength=HIST_BINS)
                g = AD.parse_geometry(raw)
                if g is not None:
                    key = (g["fmt"], g["nc"], g["nr"], g["bphi"], g["bpsi"],
                           g["bw_mhz"], g["ng"], g["feedback_type"],
                           g["codebook"])
                    gid[i] = geoms.setdefault(key, len(geoms))
                    groups[(key, len(raw))].append(i)
            if has_parsed:
                for k in range(4):
                    if r[5 + k] is not None:
                        cap[i, k] = r[5 + k]
                fb[i] = (FB_NULL if r[9] is None
                         else FB_SU if r[9] == "SU" else FB_OTHER)
                s = decode_avg_snr_dB(r[10])
                s = (s + [0.0] * SNR_KEEP)[:SNR_KEEP]
                snr[i] = s
        for (key, _L), idx in groups.items():
            g = dict(zip(["fmt", "nc", "nr", "bphi", "bpsi", "bw_mhz", "ng",
                          "feedback_type", "codebook"], key))
            g["mimo_len"] = (AD.HE_MIMO_CTRL_LEN if g["fmt"] == "HE"
                             else AD.VHT_MIMO_CTRL_LEN)
            res = decode_group([bodies[i] for i in idx], g)
            if res is None:
                continue
            ia = np.asarray(idx)
            gc_[ia], gs_[ia], gp_[ia], nsub[ia] = res
        for k, v in dict(ts=ts, mac=mac, ap=ap, rssi=rssi, snr=snr, cap=cap,
                         fb=fb, hist=hist, blen=blen, body_null=body_null,
                         gid=gid, nsub=nsub, gcos=gc_, gsin=gs_,
                         gpsi=gp_).items():
            cols_out[k].append(v)
        del rows, bodies, groups
    c.close()
    arrays = {k: np.concatenate(v) if v else np.zeros(0)
              for k, v in cols_out.items()}
    np.savez(out, has_parsed=np.asarray(has_parsed), **arrays,
             macs=np.asarray(list(macs.keys())),
             aps=np.asarray(list(aps.keys()) or [""]),
             geoms=np.asarray(json.dumps([list(k) for k in geoms])))
    del arrays, cols_out
    gc.collect()
    return out


def build_all(force: bool = False) -> list[Path]:
    paths = []
    for p in day_paths():
        t = time.time()
        paths.append(build_day_cache(p, force=force))
        print(f"[cache] {p.stem} -> {paths[-1].name} "
              f"({time.time() - t:.1f}s)", file=sys.stderr)
        log_rss(p.stem)
    return paths


# ---------------------------------------------------------------------------
# Loading the cache into per-(MAC, day) frame tables
# ---------------------------------------------------------------------------
FRAME_FIELDS = ("ts", "rssi", "snr", "cap", "fb", "hist", "blen",
                "body_null", "gid", "nsub", "gcos", "gsin", "gpsi")


def load_day(day_stem: str, fields=FRAME_FIELDS) -> dict:
    """Returns {mac: {field: array}} for one day, rows in ts order, plus the
    geometry table under key '__geoms__' and schema flag '__parsed__'."""
    z = np.load(cache_path(day_stem), allow_pickle=False)
    out = {"__geoms__": [], "__parsed__": bool(z["has_parsed"])}
    if "mac" not in z.files:            # empty capture day (05-12, 05-13)
        return out
    macs = z["macs"]
    mac = z["mac"]
    for mi, m in enumerate(macs):
        sel = mac == mi
        out[str(m)] = {f: z[f][sel] for f in fields}
    out["__geoms__"] = json.loads(str(z["geoms"]))
    out["__parsed__"] = bool(z["has_parsed"])
    return out


def mac_group(mac: str) -> tuple[str, str]:
    lab = DEVICE_GT.get(device_key(mac))
    if lab:
        return lab
    if is_random(mac):
        return ("random (locally administered)", "random")
    return ("unlabelled (global)", "unlabelled")


# ---------------------------------------------------------------------------
# Paper feature reconstruction (exactly run_pi_one_shot.frame_features)
# ---------------------------------------------------------------------------
def paper_frame_features(d: dict, fb_null_as_su: bool) -> dict:
    """power = [rssi or 0, snr0, snr1]; capability = [nc, nr, bw, cb, SU];
    body = byte histogram / len. fb_null_as_su reproduces the crossday
    loader, which maps a NULL feedback_type to 'SU'."""
    rssi = np.nan_to_num(d["rssi"].astype(np.float64), nan=0.0)
    power = np.column_stack([rssi, d["snr"].astype(np.float64)])
    cap = np.where(d["cap"] < 0, 0, d["cap"]).astype(np.float64)
    su = (d["fb"] == FB_SU) | ((d["fb"] == FB_NULL) & fb_null_as_su)
    capability = np.column_stack([cap, su.astype(np.float64)])
    body = d["hist"].astype(np.float64) / np.maximum(1, d["blen"])[:, None]
    return dict(power=power, capability=capability, body=body)


# ---------------------------------------------------------------------------
# Window features
# ---------------------------------------------------------------------------
def givens_window_features(gcos, gsin, gpsi) -> np.ndarray:
    """(T, B) per-frame bins -> 5B vector: phi mean cos, mean sin,
    circular spread 1-R, psi mean, psi std."""
    mc = np.nanmean(gcos, axis=0)
    ms = np.nanmean(gsin, axis=0)
    spread = 1.0 - np.sqrt(mc ** 2 + ms ** 2)
    return np.concatenate([mc, ms, spread, np.nanmean(gpsi, axis=0),
                           np.nanstd(gpsi, axis=0)])


def windows_for_mac_day(d: dict, T: int = WINDOW_FRAMES,
                        require_givens: bool = True):
    """Non-overlapping consecutive T-frame windows over one (MAC, day).
    Returns list of dicts with start index, start ts, givens feat, paper
    feats (window means), rssi mean/std, dominant geometry id."""
    ok = ~d["body_null"]
    if require_givens:
        ok &= ~np.isnan(d["gcos"][:, 0])
    idx = np.flatnonzero(ok)
    out = []
    for w in range(len(idx) // T):
        ii = idx[w * T:(w + 1) * T]
        rssi = d["rssi"][ii]
        gids, cnt = np.unique(d["gid"][ii], return_counts=True)
        out.append(dict(
            i0=int(ii[0]), ts0=float(d["ts"][ii[0]]),
            ts1=float(d["ts"][ii[-1]]),
            givens=givens_window_features(d["gcos"][ii], d["gsin"][ii],
                                          d["gpsi"][ii]),
            rssi=np.array([np.nanmean(rssi), np.nanstd(rssi)]),
            gid=int(gids[np.argmax(cnt)]),
            blen=float(np.median(d["blen"][ii])),
            hist=(d["hist"][ii].astype(np.float64)
                  / np.maximum(1, d["blen"][ii])[:, None]).mean(axis=0),
        ))
    return out


MIN_SHORT_FRAMES = 8        # a below-cut MAC/day needs this many frames
WINDOWS_FILE = "windows.v{v}.npz"


def build_windows(force: bool = False) -> dict:
    """One row per non-overlapping 32-frame window over every (MAC, day),
    plus one 'short' window per (MAC, day) that has fewer than 32 decodable
    frames but at least MIN_SHORT_FRAMES (used only by X8). Streams the
    per-day cache; result is a few tens of MB."""
    out_p = CACHE_DIR / WINDOWS_FILE.format(v=CACHE_VERSION)
    if out_p.exists() and not force:
        z = np.load(out_p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    build_all()
    cols = defaultdict(list)
    geoms_global: dict[str, int] = {}
    for p in day_paths():
        d = load_day(p.stem)
        G = d.pop("__geoms__"); d.pop("__parsed__")
        for m, v in d.items():
            ws = windows_for_mac_day(v)
            short = False
            if not ws:
                ok = (~v["body_null"]) & ~np.isnan(v["gcos"][:, 0])
                k = int(ok.sum())
                if k < MIN_SHORT_FRAMES:
                    continue
                ws = windows_for_mac_day(v, T=k)
                short = True
            for w in ws:
                gkey = json.dumps(G[w["gid"]]) if w["gid"] >= 0 else "none"
                cols["mac"].append(m)
                cols["day"].append(p.stem.replace("bfi_", ""))
                cols["short"].append(short)
                cols["ts0"].append(w["ts0"]); cols["ts1"].append(w["ts1"])
                cols["givens"].append(w["givens"].astype(np.float32))
                cols["rssi"].append(w["rssi"].astype(np.float32))
                cols["hist"].append(w["hist"].astype(np.float32))
                cols["blen"].append(w["blen"])
                cols["geom"].append(geoms_global.setdefault(
                    gkey, len(geoms_global)))
        del d
        gc.collect()
    arr = {k: np.asarray(v) for k, v in cols.items()}
    arr["geom_table"] = np.asarray(list(geoms_global.keys()))
    np.savez(out_p, **arr)
    log_rss("windows built")
    return arr


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[out] {path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser(description="Build the per-day cache.")
    ap_.add_argument("--dry-run", action="store_true")
    ap_.add_argument("--force", action="store_true")
    a = ap_.parse_args()
    set_vm_limit()
    if a.dry_run:
        print(json.dumps(dry_run_report(), indent=2))
        sys.exit(0)
    t0 = time.time()
    build_all(force=a.force)
    print(f"[done] {time.time() - t0:.0f}s, peak RSS {peak_rss_mb():.0f} MB",
          file=sys.stderr)
