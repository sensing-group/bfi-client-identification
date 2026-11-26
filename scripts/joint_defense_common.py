"""Shared pieces for the joint-pattern defense experiments (X1, X1b, X2).

Data loaders (NTU-Fi packets, He CR1 Givens angles), 802.11 Givens
reconstruction/decomposition, the defenses, the attackers and the
multi-query decision rule. Nothing here writes to the paper; the
experiment scripts import this module and write CSV/JSON to figures/.

Paths come from environment variables with defaults relative to this file:
    BFI_PRIVACY_ROOT       framework repo (import only; the copy in
                           release/vendor/ is tried first)  default ../../bfi-privacy
    BFI_NTUFI_ROOT         NTU-Fi HumanID root             default <BFI_PRIVACY_ROOT>/data/ntu_fi/NTU-Fi-HumanID
    BFI_HE_ROOT            He five-room archive            default ../data/he_bfi/dataset/transfer and light weight
    BFI_HE_LOCATION_ROOT   He CR1 folder (BFI.pth, y.pth)  default <BFI_HE_ROOT>/Conference Room 1
    BFI_CACHE_DIR          npy cache for the NTU-Fi packet tensor (default: system temp dir)
    BFI_VM_LIMIT_GB        address-space cap per process  (default 6)
    BFI_MAX_RSS_MB         RSS abort threshold             (default 5000)
"""
from __future__ import annotations

import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
BFI_PRIVACY_ROOT = Path(os.environ.get("BFI_PRIVACY_ROOT", REPO_ROOT.parent / "bfi-privacy"))
NTUFI_ROOT = Path(os.environ.get(
    "BFI_NTUFI_ROOT", BFI_PRIVACY_ROOT / "data" / "ntu_fi" / "NTU-Fi-HumanID"))
HE_ARCHIVE = Path(os.environ.get(
    "BFI_HE_ROOT", REPO_ROOT / "data" / "he_bfi" / "dataset" / "transfer and light weight"))
HE_ROOT = Path(os.environ.get("BFI_HE_LOCATION_ROOT", HE_ARCHIVE / "Conference Room 1"))
# The vendored he_location loader reads BFI_HE_LOCATION_ROOT only; give it the same folder.
os.environ.setdefault("BFI_HE_LOCATION_ROOT", str(HE_ROOT))
CACHE_DIR = Path(os.environ.get("BFI_CACHE_DIR", Path(tempfile.gettempdir()) / "bfi_joint_cache"))

# The `bfi` package: the copy in release/vendor/ first, BFI_PRIVACY_ROOT behind it.
import vendor_paths as VP  # noqa: E402
VP.add_bfi_privacy(BFI_PRIVACY_ROOT)

# ---------------------------------------------------------------------------
# Memory safety
# ---------------------------------------------------------------------------

VM_LIMIT_GB = float(os.environ.get("BFI_VM_LIMIT_GB", "6"))
MAX_RSS_MB = float(os.environ.get("BFI_MAX_RSS_MB", "5000"))
BYTES_PER_GB = 1024 ** 3
KB_PER_MB = 1024.0
TOTAL_RAM_WARN_FRAC = 0.5


def set_vm_limit(limit_gb: float = VM_LIMIT_GB) -> None:
    """Cap the address space so a runaway allocation fails instead of OOM-killing the box."""
    lim = int(limit_gb * BYTES_PER_GB)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def _proc_status_mb(field: str) -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field + ":"):
                return float(line.split()[1]) / KB_PER_MB
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB_PER_MB


def rss_mb() -> float:
    return _proc_status_mb("VmRSS")


def peak_rss_mb() -> float:
    return _proc_status_mb("VmHWM")


def total_ram_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / KB_PER_MB
    except OSError:
        pass
    return float("nan")


def check_rss(tag: str) -> None:
    """Log RSS; warn above half the machine RAM; abort above MAX_RSS_MB."""
    cur, peak = rss_mb(), peak_rss_mb()
    print(f"  [mem] {tag}: rss={cur:.0f} MB peak={peak:.0f} MB", flush=True)
    if cur > TOTAL_RAM_WARN_FRAC * total_ram_mb():
        print(f"  [mem] WARNING rss above {TOTAL_RAM_WARN_FRAC:.0%} of RAM", flush=True)
    if cur > MAX_RSS_MB:
        raise MemoryError(f"RSS {cur:.0f} MB exceeds BFI_MAX_RSS_MB={MAX_RSS_MB:.0f}")


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

NTUFI_SC_STRIDE = 4
NTUFI_N_RECORDS = 840
NTUFI_RX = 3
NTUFI_SC_RAW = 114
NTUFI_PKTS = 500
NTUFI_CAL_SPLITS = ("train",)   # rebuttal_loso headline: scaler/encoder fit on calibration TRAIN split

HE_SC_STRIDE = 4
HE_N_PAIRS = 6                   # pairs 01,02,03,12,13,23 -> Nr=4, Nc>=3
HE_NR = 4
HE_NC = 3
HE_FEATURE_PAIR = 0              # paper features: pair "01" = (phi11, psi21)
HE_N_CALIB_CLASSES = 20
# Codebook equivalent to the paper's He "b=4" pipeline: a 16-cell grid over
# [-pi, pi] for both angles, i.e. 4-bit phi over [-pi, pi) and 2-bit psi over [0, pi/2].
HE_B_PHI = 4
HE_B_PSI = 2
PHI_LO, PHI_HI = -np.pi, np.pi
PSI_LO, PSI_HI = 0.0, np.pi / 2.0

B_HEADLINE = 4
MANIFOLD_DELTA = 1e-5
ORTHQ_CHUNK_BYTES = 100e6        # bytes of Q per batched QR (QR temporaries roughly triple this)


# ---------------------------------------------------------------------------
# NTU-Fi
# ---------------------------------------------------------------------------

def _ntufi_prefix(path: str) -> str:
    name = Path(path).name
    return name[0] if name and name[0] in "abc" else "?"


def ntufi_estimate_mb() -> float:
    return NTUFI_N_RECORDS * NTUFI_RX * (NTUFI_SC_RAW // NTUFI_SC_STRIDE + 1) * NTUFI_PKTS * 4 / 1e6


def load_ntufi_packets() -> dict:
    """(840, 3, 29, 500) float32 amplitude packets + labels/split/prefix. Cached as .npy."""
    from bfi.datasets.ntu_fi import NTUFiLoader

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ntufi_packets_stride{NTUFI_SC_STRIDE}.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    loader = NTUFiLoader(NTUFI_ROOT, task="HumanID")
    A, y, split, prefix = [], [], [], []
    for sp in ("train", "test"):
        for rec in loader.iter_records(split=sp):
            A.append(rec.csi[:, ::NTUFI_SC_STRIDE, :].astype(np.float32))
            y.append(int(rec.label))
            split.append(sp)
            prefix.append(_ntufi_prefix(rec.path))
    out = dict(A=np.stack(A), y=np.asarray(y, np.int64),
               split=np.asarray(split), prefix=np.asarray(prefix))
    np.savez(cache, **out)
    return out


def ntufi_features(A: np.ndarray) -> np.ndarray:
    """Per-coordinate mean and std over packets -> (N, 174), same order as demo_rebuttal_loso."""
    A64 = A.astype(np.float64)
    red = np.concatenate([A64.mean(axis=-1), A64.std(axis=-1)], axis=1)  # (N, 6, 29)
    return red.reshape(red.shape[0], -1)


def ntufi_norm_invariant(A: np.ndarray) -> np.ndarray:
    """Antenna-axis norm per (subcarrier, packet), then mean/std over packets -> (N, 58).

    Invariant to any per-packet orthogonal transform on the antenna axis that is
    shared across subcarriers.
    """
    n = np.sqrt((A.astype(np.float64) ** 2).sum(axis=1))  # (N, 29, P)
    return np.concatenate([n.mean(axis=-1), n.std(axis=-1)], axis=1)


# ---------------------------------------------------------------------------
# He CR1
# ---------------------------------------------------------------------------

def he_estimate_mb(root: Path = HE_ROOT) -> float:
    size = (root / "BFI.pth").stat().st_size
    return size / 1e6


def load_he_angles(root: Path = HE_ROOT) -> dict:
    """He continuous Givens angles at stride-4 subcarriers (CR1 by default; any
    room folder of the 'transfer and light weight' archive via ``root``).

    Returns phi, psi of shape (N, 59, 6) float64 in 802.11 order
    [11, 21, 31, 22, 32, 33] / [21, 31, 41, 32, 42, 43], labels y.
    """
    import torch

    bfi = torch.load(str(root / "BFI.pth"), map_location="cpu", weights_only=True)
    y = torch.load(str(root / "y.pth"), map_location="cpu", weights_only=True).numpy().astype(np.int64)
    raw_shape = tuple(bfi.shape)
    sub = bfi[:, ::HE_SC_STRIDE, :, :].numpy().astype(np.float64).copy()
    del bfi
    phi, psi = sub[..., 0].copy(), sub[..., 1].copy()
    del sub
    return dict(phi=phi, psi=psi, y=y, raw_shape=raw_shape)


def quant_midpoint(x, lo, hi, b):
    n = 1 << b
    cell = (hi - lo) / n
    idx = np.clip(np.floor((x - lo) / cell).astype(int), 0, n - 1)
    return lo + (idx + 0.5) * cell


def he_quantise(phi, psi, *, eps=None, rng=None):
    """Codebook quantisation (b_phi=4 circular, b_psi=2); DP-SQ when eps is given."""
    if eps is None:
        return quant_midpoint(phi, PHI_LO, PHI_HI, HE_B_PHI), quant_midpoint(psi, PSI_LO, PSI_HI, HE_B_PSI)
    from bfi.quantisation import _dp_sq_quantise_1d
    return (_dp_sq_quantise_1d(phi, PHI_LO, PHI_HI, HE_B_PHI, eps, rng, circular=True),
            _dp_sq_quantise_1d(psi, PSI_LO, PSI_HI, HE_B_PSI, eps, rng, circular=False))


def he_features(phi, psi, pair=HE_FEATURE_PAIR):
    """Paper He features: pair-0 (phi, psi) interleaved over 59 subcarriers -> (N, 118)."""
    x = np.stack([phi[:, :, pair], psi[:, :, pair]], axis=-1)
    return x.reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# 802.11 Givens (IEEE 802.11-2020 19.3.12.3.6), Nr=4, Nc=3
# ---------------------------------------------------------------------------

def _givens_index(nr=HE_NR, nc=HE_NC):
    """Angle order: for column i, phi_{i..nr-1, i}; psi_{i+1..nr, i} (1-indexed)."""
    phis, psis = [], []
    for i in range(min(nc, nr - 1)):
        for k in range(i, nr - 1):
            phis.append((k, i))
        for l in range(i + 1, nr):
            psis.append((l, i))
    return phis, psis


GIVENS_PHI_IDX, GIVENS_PSI_IDX = _givens_index()


def givens_to_v(phi: np.ndarray, psi: np.ndarray, nr=HE_NR, nc=HE_NC) -> np.ndarray:
    """phi, psi (..., 6) -> V (..., nr, nc) complex."""
    shape = phi.shape[:-1]
    M = np.broadcast_to(np.eye(nr, dtype=np.complex128), shape + (nr, nr)).copy()
    pi_ = 0
    si_ = 0
    for i in range(min(nc, nr - 1)):
        # M <- M @ D_i
        for k in range(i, nr - 1):
            M[..., :, k] *= np.exp(1j * phi[..., pi_])[..., None]
            pi_ += 1
        # M <- M @ G_{l,i}^T for l = i+1..nr-1
        for l in range(i + 1, nr):
            c = np.cos(psi[..., si_])[..., None]
            s = np.sin(psi[..., si_])[..., None]
            ci, cl = M[..., :, i].copy(), M[..., :, l].copy()
            # G^T has (i,i)=c, (l,i)=s, (i,l)=-s, (l,l)=c ; (M G^T)[:, i] = c*M_i + s*M_l
            M[..., :, i] = c * ci + s * cl
            M[..., :, l] = -s * ci + c * cl
            si_ += 1
    return M[..., :, :nc]


def v_to_givens(V: np.ndarray, nr=HE_NR, nc=HE_NC):
    """V (..., nr, nc) with orthonormal columns -> (phi, psi) (..., 6) in 802.11 order.

    The column phases are first normalised so the last row is real non-negative
    (the D-tilde the beamformee removes), then angles are peeled column by column.
    """
    W = V.astype(np.complex128).copy()
    last = W[..., nr - 1, :]
    W = W * np.exp(-1j * np.angle(last))[..., None, :]
    phis, psis = [], []
    for i in range(min(nc, nr - 1)):
        for k in range(i, nr - 1):
            a = np.angle(W[..., k, i])
            phis.append(a)
            W[..., k, :] *= np.exp(-1j * a)[..., None]
        for l in range(i + 1, nr):
            xi = W[..., i, i].real
            xl = W[..., l, i].real
            p = np.arctan2(xl, xi)
            psis.append(p)
            c, s = np.cos(p)[..., None], np.sin(p)[..., None]
            ri, rl = W[..., i, :].copy(), W[..., l, :].copy()
            W[..., i, :] = c * ri + s * rl
            W[..., l, :] = -s * ri + c * rl
    phi = np.stack(phis, axis=-1)
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    psi = np.clip(np.stack(psis, axis=-1), PSI_LO, PSI_HI)
    return phi, psi


def haar_unitary(n: int, size: tuple, rng: np.random.Generator) -> np.ndarray:
    z = (rng.standard_normal(size + (n, n)) + 1j * rng.standard_normal(size + (n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / np.abs(d))[..., None, :]


def haar_orthogonal(n: int, size: tuple, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(size + (n, n))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diagonal(r, axis1=-2, axis2=-1))
    d[d == 0] = 1.0
    return q * d[..., None, :]


# ---------------------------------------------------------------------------
# Defenses shared across substrates
# ---------------------------------------------------------------------------

def orth_q_features(X: np.ndarray, rng, *, mode: str, groups=None) -> np.ndarray:
    """Per-window (fresh) or per-group (fixed) orthogonal Q on the feature vector."""
    n, d = X.shape
    if mode == "fresh":
        out = np.empty_like(X)
        chunk = max(1, int(ORTHQ_CHUNK_BYTES // (d * d * 8)))
        for s in range(0, n, chunk):
            Q = haar_orthogonal(d, (min(chunk, n - s),), rng)
            out[s:s + chunk] = np.einsum("nij,nj->ni", Q, X[s:s + chunk])
            del Q
        return out
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = X[m] @ haar_orthogonal(d, (), rng).T
    return out


def manifold_dp(X: np.ndarray, lo: float, hi: float, eps: float, rng) -> np.ndarray:
    from bfi.manifold_dp import manifold_quantise_arr
    return manifold_quantise_arr(X, B_HEADLINE, lo, hi, eps=eps, delta=MANIFOLD_DELTA, rng=rng)


def manifold_sigma_over_range(eps: float) -> float:
    from bfi.manifold_dp import gaussian_sigma
    return gaussian_sigma(eps=eps, delta=MANIFOLD_DELTA, sensitivity=1.0)


def dpsq_std(Xs: np.ndarray, lo: float, hi: float, eps: float, rng) -> np.ndarray:
    from bfi.framework import dp_sq
    return dp_sq(Xs, b=B_HEADLINE, eps=eps, rng=rng, lo=lo, hi=hi, circular=False)


# ---------------------------------------------------------------------------
# Attack evaluation (K=1 support, multi-query T)
# ---------------------------------------------------------------------------

def pick_supports(y: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {int(c): int(rng.choice(np.flatnonzero(y == c))) for c in np.unique(y)}


def _normalise(Z):
    return Z / np.clip(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12, None)


def multi_query_eval(Z: np.ndarray, y: np.ndarray, supports: dict, *, metric: str,
                     T_grid, n_decisions: int, rng) -> dict:
    """{T: accuracy}. T=1 uses every non-support window; T>1 averages T distinct
    same-class non-support windows per decision (n_decisions per class)."""
    labels = np.array(sorted(supports))
    S = Z[[supports[c] for c in labels]]
    sup_set = set(supports.values())

    def decide(Q):
        if metric == "cos":
            return labels[np.argmax(_normalise(Q) @ _normalise(S).T, axis=1)]
        d = ((Q[:, None, :] - S[None, :, :]) ** 2).sum(-1)
        return labels[np.argmin(d, axis=1)]

    out = {}
    pools = {c: np.array([i for i in np.flatnonzero(y == c) if i not in sup_set]) for c in labels}
    for T in T_grid:
        if T == 1:
            q = np.concatenate([pools[c] for c in labels])
            out[T] = float((decide(Z[q]) == y[q]).mean())
            continue
        correct = total = 0
        for c in labels:
            pool = pools[c]
            picks = np.stack([rng.choice(pool, size=T, replace=len(pool) < T) for _ in range(n_decisions)])
            Q = Z[picks].mean(axis=1)
            correct += int((decide(Q) == c).sum())
            total += n_decisions
        out[T] = correct / total
    return out


# ---------------------------------------------------------------------------
# SupCon encoder (paper settings)
# ---------------------------------------------------------------------------

SUPCON_EPOCHS = 200
SUPCON_BATCH = 64
SUPCON_LR = 1e-3
SUPCON_WD = 1e-4
SUPCON_TEMP = 0.1
SUPCON_HIDDEN = 256
SUPCON_EMBED = 128
TORCH_THREADS = int(os.environ.get("BFI_TORCH_THREADS", "3"))


def train_supcon(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = SUPCON_EPOCHS):
    import torch
    from bfi.framework import OneShotAttacker

    torch.set_num_threads(TORCH_THREADS)
    att = OneShotAttacker(encoder="mlp", embed_dim=SUPCON_EMBED, hidden_dim=SUPCON_HIDDEN,
                          contrastive="supcon", temperature=SUPCON_TEMP)
    att.fit(X.astype(np.float32), y, epochs=epochs, batch_size=SUPCON_BATCH, lr=SUPCON_LR,
            weight_decay=SUPCON_WD, seed=seed, verbose=False)
    return att


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
