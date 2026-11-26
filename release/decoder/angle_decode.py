"""Decode the quantised Givens angles (phi/psi) from an 802.11 VHT/HE
Compressed Beamforming Report body, and measure angle-space drift.

The MIMO-Control header parsers follow the VHT and HE MIMO Control field
layouts, cross-checked against Wireshark's wlan.he.mimo.* bitmasks
(epan/dissectors/packet-ieee80211.c). They began as
copies of the capture service
(tools/pi-capture/bfi_capture.py), whose HE parser used the
VHT offsets; that copy was corrected here, so the two are no longer identical
for HE. This module adds the angle bit-unpacking that service omits.

Why angles, not raw bytes: on the E1 capture, moving-vs-empty separation went
from 1.86x (byte-histogram variation) to 3.61x once the real phi/psi angles
were decoded, confirmed by a sounding-cadence-independent phi circular std at
3.19x. phi is a full 2*pi rotation (circular distance); psi is a [0, pi/2]
magnitude (linear). A raw byte diff respects neither the bit boundaries nor the
circular topology.

Vectorised (np.unpackbits) so it is cheap enough for the live receiver path.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# ---- MIMO Control header parsers --------------------------------------------
VHT_BW_MAP = {0: 20, 1: 40, 2: 80, 3: 160}
VHT_NG_MAP = {0: 1, 1: 2, 2: 4}
VHT_FB_MAP = {0: "SU", 1: "MU"}
HE_BW_MAP = {0: 20, 1: 40, 2: 80, 3: 160}
HE_NG_MAP = {0: 4, 1: 16}
HE_FB_MAP = {0: "SU", 1: "MU", 2: "CQI"}
CATEGORY_HE = 0x1E
CATEGORY_VHT = 0x15
HE_MIMO_CTRL_LEN = 5
VHT_MIMO_CTRL_LEN = 3

MIN_BFI_BODY_BYTES = 7  # Shortest body that can hold category + MIMO control fields

# Angle bit widths (bpsi, bphi) by (feedback_type, codebook). bphi = bpsi + 2.
ANGLE_BITS = {
    ("SU", 0): (2, 4), ("SU", 1): (4, 6),
    ("MU", 0): (5, 7), ("MU", 1): (7, 9),
}

# Number of subcarriers carried in the compressed beamforming report, keyed by
# (bandwidth MHz, Ng). The angle field is NOT the whole remainder of the body:
# MU feedback appends a MU Exclusive Beamforming Report (per-subcarrier delta
# SNR) after the angles, and a captured body may still carry the FCS. Deriving
# n_sub from the body length therefore OVER-counts. Measured on the Wi-BFI
# reference capture (VHT MU cb1, nr=3, nc=1, 80 MHz): length inference gives
# 249 subcarriers where the true count is 234 -- 15 phantom subcarriers decoded
# out of the delta-SNR field. The trailing values are appended, not interleaved,
# so capping recovers the exact angle set (verified bit-exact against
# Wi-BFI's published bfa_ac_mu_3x1_80.npy, 200 x 234 x 4 angles).
VHT_NSUB = {
    (20, 1): 52, (20, 2): 30, (20, 4): 16,
    (40, 1): 108, (40, 2): 58, (40, 4): 30,
    (80, 1): 234, (80, 2): 122, (80, 4): 62,
    (160, 1): 468, (160, 2): 244, (160, 4): 124,
}
# HE Ng=4 counts cross-checked against Wi-BFI's own subcarrier-index
# construction. Ng=16 is deliberately absent: an entry that is too SMALL would
# silently truncate real angles, so unverified combinations fall back to length
# inference (i.e. today's behaviour) rather than risk that.
HE_NSUB = {
    (20, 4): 64, (40, 4): 122, (80, 4): 250, (160, 4): 500,
}


def _bits(v: int, start: int, n: int) -> int:
    return (v >> start) & ((1 << n) - 1)


def parse_vht_mimo_ctrl(b: bytes) -> dict:
    if len(b) < VHT_MIMO_CTRL_LEN:
        return {}
    v = int.from_bytes(b[:VHT_MIMO_CTRL_LEN], "little")
    return dict(nc=_bits(v, 0, 3) + 1, nr=_bits(v, 3, 3) + 1,
                bw_mhz=VHT_BW_MAP.get(_bits(v, 6, 2)),
                ng=VHT_NG_MAP.get(_bits(v, 8, 2)), codebook=_bits(v, 10, 1),
                feedback_type=VHT_FB_MAP.get(_bits(v, 11, 1)),
                dialog_token=_bits(v, 18, 6))


def parse_he_mimo_ctrl(b: bytes) -> dict:
    """HE MIMO Control (5 octets). Unlike VHT, grouping is ONE bit (B8),
    codebook is B9, feedback type B10-B11, and the RU start/end indices
    (B16-B22, B23-B29) sit before the sounding dialog token (B30-B35)."""
    if len(b) < HE_MIMO_CTRL_LEN:
        return {}
    v = int.from_bytes(b[:HE_MIMO_CTRL_LEN], "little")
    return dict(nc=_bits(v, 0, 3) + 1, nr=_bits(v, 3, 3) + 1,
                bw_mhz=HE_BW_MAP.get(_bits(v, 6, 2)),
                ng=HE_NG_MAP.get(_bits(v, 8, 1)), codebook=_bits(v, 9, 1),
                feedback_type=HE_FB_MAP.get(_bits(v, 10, 2)),
                remaining_segments=_bits(v, 12, 3),
                first_segment=_bits(v, 15, 1),
                ru_start=_bits(v, 16, 7), ru_end=_bits(v, 23, 7),
                dialog_token=_bits(v, 30, 6))


# layout cache: (nr, nc, bphi, bpsi) -> (offsets, widths, kinds_pattern, bits_per_sc)
_LAYOUT_CACHE: Dict[Tuple[int, int, int, int], tuple] = {}


def _layout(nr: int, nc: int, bphi: int, bpsi: int):
    key = (nr, nc, bphi, bpsi)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    widths = []
    kinds = []  # 0 = phi, 1 = psi
    for i in range(1, min(nc, nr - 1) + 1):
        for _ in range(nr - i):
            widths.append(bphi); kinds.append(0)
        for _ in range(nr - i):
            widths.append(bpsi); kinds.append(1)
    widths = np.asarray(widths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    kinds = np.asarray(kinds, dtype=np.uint8)
    bits_per_sc = int(widths.sum())
    out = (offsets, widths, kinds, bits_per_sc)
    _LAYOUT_CACHE[key] = out
    return out


def parse_geometry(body: bytes) -> Optional[dict]:
    """Return dict(fmt, mimo_len, nc, nr, bphi, bpsi, feedback_type, codebook)
    or None if not a compressed beamforming body we can decode."""
    if len(body) < MIN_BFI_BODY_BYTES:
        return None
    cat = body[0]
    if cat == CATEGORY_VHT:
        p = parse_vht_mimo_ctrl(body[2:2 + VHT_MIMO_CTRL_LEN])
        mimo_len = VHT_MIMO_CTRL_LEN
        fmt = "VHT"
    elif cat == CATEGORY_HE:
        p = parse_he_mimo_ctrl(body[2:2 + HE_MIMO_CTRL_LEN])
        mimo_len = HE_MIMO_CTRL_LEN
        fmt = "HE"
    else:
        return None
    if not p or p.get("nc") is None or p.get("nr") is None:
        return None
    fb = p.get("feedback_type")
    cb = p.get("codebook", 0)
    bpsi, bphi = ANGLE_BITS.get((fb, cb), (2, 4))
    return dict(fmt=fmt, mimo_len=mimo_len, nc=p["nc"], nr=p["nr"],
                bphi=bphi, bpsi=bpsi, feedback_type=fb, codebook=cb,
                bw_mhz=p.get("bw_mhz"), ng=p.get("ng"))


def expected_subcarriers(geom: dict) -> Optional[int]:
    """Subcarriers the report should carry for this geometry, or None when the
    (bandwidth, Ng) pair is not one we have verified."""
    table = VHT_NSUB if geom.get("fmt") == "VHT" else HE_NSUB
    return table.get((geom.get("bw_mhz"), geom.get("ng")))


def geometry_key(g: dict) -> tuple:
    return (g["fmt"], g["nc"], g["nr"], g["bphi"], g["bpsi"])


def decode_angles(body: bytes, geom: dict) -> Optional[np.ndarray]:
    """Vectorised unpack of the angle field -> int array of quantised values,
    length n_sub * angles_per_subcarrier (phi/psi interleaved per layout).
    Returns None if the body is too short to hold even one subcarrier."""
    start = 2 + geom["mimo_len"] + geom["nc"]      # skip header + per-stream SNR
    ab = body[start:]
    if not ab:
        return None
    offsets, widths, _kinds, bits_per_sc = _layout(
        geom["nr"], geom["nc"], geom["bphi"], geom["bpsi"])
    bits = np.unpackbits(np.frombuffer(ab, dtype=np.uint8), bitorder="little")
    n_sub = len(bits) // bits_per_sc
    # Trailing MU-exclusive/FCS bytes inflate the length-derived count; cap it
    # at the standard-defined value. min() keeps short or segmented reports
    # (Remaining Feedback Segments) decoding at their actual length.
    n_expected = expected_subcarriers(geom)
    if n_expected is not None:
        n_sub = min(n_sub, n_expected)
    if n_sub < 1:
        return None
    n_ang = len(widths)
    vals = np.zeros((n_sub, n_ang), dtype=np.int32)
    base = np.arange(n_sub, dtype=np.int64) * bits_per_sc
    for j in range(n_ang):
        o = offsets[j]; w = int(widths[j])
        acc = np.zeros(n_sub, dtype=np.int32)
        for k in range(w):
            acc += bits[base + o + k].astype(np.int32) << k
        vals[:, j] = acc
    return vals.reshape(-1)


def kinds_for(geom: dict, n_sub: int) -> np.ndarray:
    _o, _w, kinds, _b = _layout(geom["nr"], geom["nc"], geom["bphi"], geom["bpsi"])
    return np.tile(kinds, n_sub)


def angle_drift(vals_a: np.ndarray, vals_b: np.ndarray, geom: dict) -> float:
    """Mean normalised angle-space distance in [0,1] between two soundings of
    identical geometry. phi circular over 2^bphi; psi linear over 2^bpsi - 1."""
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return 0.0
    a = vals_a[:n].astype(np.int64)
    b = vals_b[:n].astype(np.int64)
    _o, _w, kind_pat, _bps = _layout(
        geom["nr"], geom["nc"], geom["bphi"], geom["bpsi"])
    kinds = np.tile(kind_pat, n // len(kind_pat) + 1)[:n]
    d = np.abs(a - b).astype(np.float64)
    out = np.empty(n, dtype=np.float64)
    phi = kinds == 0
    rng_phi = float(1 << geom["bphi"])
    out[phi] = np.minimum(d[phi], rng_phi - d[phi]) / (rng_phi / 2.0)
    psi = kinds == 1
    out[psi] = d[psi] / float((1 << geom["bpsi"]) - 1)
    return float(out.mean())
