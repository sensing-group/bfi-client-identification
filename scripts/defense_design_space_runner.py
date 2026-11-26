"""Defense design-space sweep for the AP-side firmware patches.

Compares four soft-patch defenses on the same NTU-Fi HumanID pipeline:

    deterministic     midpoint quantisation, no defense (reference)
    dp_sq             Seif et al. DP stochastic quantisation, the
                       published software-only baseline
    unitary_obfusc    LeakyBeam-style per-window random orthogonal
                       obfuscation (information-theoretic surrogate
                       for transmitter-side phase randomisation; see
                       caveat below)
    grouping          max-N_g subcarrier averaging at the AP before
                       quantisation (frequency-domain smoothing)
    sound_rate        per-packet temporal subsampling at the AP,
                       which throttles the sounding rate observed by
                       both the eavesdropper and the AP

Each defense reports two numbers:

    privacy = 1-NN re-identification accuracy of a closed-set
              eavesdropper that trains on the defended train set and
              queries on the defended test set. 14 NTU-Fi subjects
              gives a chance line of 1/14 \\approx 7.14%.

    utility = NMSE of the AP-side reconstruction of the original
              feature vector against the clean amplitude reference.
              For sound_rate the AP-side cost is "channel staleness"
              rather than NMSE; the NMSE column for that defense
              measures the noise added by averaging fewer packets.

Caveat on unitary_obfusc. Protocol-level phase randomisation applies
a diagonal phase D in U(N_TX) to the AP's transmit antennas. On
amplitude CSI |HD| = |H| pointwise because |D| = 1, so the literal
mechanism is invisible on NTU-Fi. The framework's structural defense
applies a per-window orthogonal Q on the flattened amplitude feature
vector, which is the information-theoretic surrogate for the same
mechanism at the BFI-Givens level (Xiao et al. NDSS 2025). It is
strictly stronger than antenna-axis phase rand on the surrogate
substrate; the result must be read as "what an AP-side per-window
unitary on the BFI channel reaches", not as a literal NDP-level
measurement.

Usage:
    python3 bfi/demo_defense_design_space.py \\
        --b 4 --eps 0.5 --n-seeds 10 \\
        --ng-grid 2 4 8 16 \\
        --rate-grid 500 100 25 5 1 \\
        --out-csv /tmp/bfi_defense_design_space.csv
"""

from __future__ import annotations

import argparse
import csv
import os as _os
import sys
from pathlib import Path

import numpy as np

# The `bfi` package: the copy in release/vendor/ first, BFI_PRIVACY_ROOT behind it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vendor_paths as VP  # noqa: E402

BFI_PRIVACY_ROOT = Path(_os.environ.get(
    "BFI_PRIVACY_ROOT", VP.REPO_ROOT.parent / "bfi-privacy"))
VP.add_bfi_privacy(BFI_PRIVACY_ROOT)

from bfi.datasets import NTUFiLoader  # noqa: E402
from bfi.quantisation import _dp_sq_quantise_1d  # noqa: E402
from bfi.structural_defense import structural_obfuscate  # noqa: E402


# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

DEFAULT_SC_STRIDE = 4
DEFAULT_B = 4
DEFAULT_EPS = 0.5
DEFAULT_N_SEEDS = 10
DEFAULT_SEED_BASE = 42
DEFAULT_OUT_CSV = str(Path(__file__).resolve().parent.parent / "figures" / "bfi_defense_design_space.csv")

# 802.11ac/ax max grouping is 16 (80 MHz), 8 (40 MHz), 4 (20 MHz).
# We sweep up to 16 to stay inside the standard.
DEFAULT_NG_GRID = [2, 4, 8, 16]

# NTU-Fi records ship 500 packets each. The native pkt-mean uses all
# 500. Throttling to K packets per record models the AP reducing the
# sounding rate by 500/K; K=1 is the most aggressive cap.
DEFAULT_RATE_GRID = [500, 100, 25, 5, 1]
NATIVE_PACKETS_PER_RECORD = 500


# --------------------------------------------------------------------------
# Quantisers
# --------------------------------------------------------------------------


def midpoint_q(x, lo, hi, b):
    if b <= 0:
        return np.full_like(x, (lo + hi) / 2.0)
    n_bins = 1 << b
    cell = (hi - lo) / n_bins
    idx = np.clip(np.floor((x - lo) / cell).astype(int), 0, n_bins - 1)
    return lo + (idx + 0.5) * cell


def quantise_arr(amp, b, lo, hi, eps, rng):
    out = np.empty_like(amp, dtype=float)
    if rng is None:
        for i in range(amp.shape[0]):
            out[i] = midpoint_q(amp[i], lo, hi, b)
    else:
        for i in range(amp.shape[0]):
            out[i] = _dp_sq_quantise_1d(
                amp[i], lo=lo, hi=hi, b=b, eps=eps, rng=rng, circular=False,
            )
    return out


# --------------------------------------------------------------------------
# Defenses
# --------------------------------------------------------------------------


def subcarrier_group(amp, ng, rx_antennas):
    """AP requests N_g grouping. Average ng adjacent subcarriers,
    then expand back to original feature length by repetition so the
    quantiser sees the same dimensionality. Operates over the
    subcarrier axis inside the (rx, sc) tensor reshape.
    """
    if ng <= 1:
        return amp.copy()
    n_sc_eff = amp.shape[1] // rx_antennas
    out = amp.reshape(amp.shape[0], rx_antennas, n_sc_eff).copy()
    full = (n_sc_eff // ng) * ng
    head = out[:, :, :full].reshape(
        amp.shape[0], rx_antennas, -1, ng).mean(axis=-1)
    head = np.repeat(head, ng, axis=-1)
    if full < n_sc_eff:
        tail_mean = out[:, :, full:].mean(axis=-1, keepdims=True)
        tail = np.repeat(tail_mean, n_sc_eff - full, axis=-1)
        out_grouped = np.concatenate([head, tail], axis=-1)
    else:
        out_grouped = head
    return out_grouped.reshape(amp.shape[0], -1)


def packet_subsample(csi_raw, n_pkts, rng):
    """Sound-rate throttling at the packet level.

    csi_raw : ndarray of shape (N, RX, SC, P) with P = 500 native pkts.
    n_pkts  : how many packets per record to keep (uniform random K of P).
    Returns the mean-pooled amplitude feature over the kept packets,
    shape (N, RX, SC).
    """
    if n_pkts >= csi_raw.shape[-1]:
        return csi_raw.mean(axis=-1)
    p = csi_raw.shape[-1]
    out = np.empty(csi_raw.shape[:-1], dtype=csi_raw.dtype)
    for i in range(csi_raw.shape[0]):
        idx = rng.choice(p, size=n_pkts, replace=False)
        out[i] = csi_raw[i, ..., idx].mean(axis=0)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def load_raw(loader, split, sc_stride):
    """Return raw CSI tensor of shape (N, RX, SC_eff, P), labels (N,),
    and the number of RX antennas. SC_eff strides the original SC.
    """
    csi_list, labels = [], []
    rx_antennas = None
    for rec in loader.iter_records(split=split):
        x = rec.csi  # (RX, SC, P)
        if rx_antennas is None:
            rx_antennas = x.shape[0]
        x = x[:, ::sc_stride, :]
        csi_list.append(x)
        labels.append(rec.label)
    return (
        np.stack(csi_list, axis=0),
        np.asarray(labels, dtype=int),
        rx_antennas,
    )


def one_nn_accuracy(train_X, train_y, test_X, test_y):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf.fit(train_X, train_y)
    return float((clf.predict(test_X) == test_y).mean())


def nmse(recovered, original):
    num = ((recovered - original) ** 2).sum(axis=1)
    den = (original ** 2).sum(axis=1).clip(min=1e-12)
    return float((num / den).mean())


def features_from_csi(csi_3d):
    """Flatten (N, RX, SC) into (N, RX*SC) feature vector."""
    return csi_3d.reshape(csi_3d.shape[0], -1)


# --------------------------------------------------------------------------
# Defense runners
# --------------------------------------------------------------------------


def run_defense(name, raw_tr, ytr, raw_te, yte,
                 b, eps, seed, ng, n_pkts, rx_antennas):
    """Apply one (defense, parameter) point and return (acc, nmse).

    raw_tr, raw_te : raw CSI tensors of shape (N, RX, SC, P).
    """
    rng = np.random.default_rng(seed)

    if name == "sound_rate":
        Xtr_amp = features_from_csi(
            packet_subsample(raw_tr, n_pkts, rng))
        Xte_amp = features_from_csi(
            packet_subsample(raw_te, n_pkts, rng))
    else:
        Xtr_amp = features_from_csi(raw_tr.mean(axis=-1))
        Xte_amp = features_from_csi(raw_te.mean(axis=-1))

    # Native (full-rate) reference for utility comparisons.
    Xtr_native = features_from_csi(raw_tr.mean(axis=-1))
    Xte_native = features_from_csi(raw_te.mean(axis=-1))

    lo = float(min(Xtr_native.min(), Xte_native.min()))
    hi = float(max(Xtr_native.max(), Xte_native.max()))

    Xtr_clean_q = quantise_arr(Xtr_native, b, lo, hi, eps=0.0, rng=None)
    Xte_clean_q = quantise_arr(Xte_native, b, lo, hi, eps=0.0, rng=None)
    det_nmse = nmse(Xte_clean_q, Xte_native)

    if name == "deterministic":
        acc = one_nn_accuracy(Xtr_clean_q, ytr, Xte_clean_q, yte)
        return acc, det_nmse

    if name == "dp_sq":
        Xtr_q = quantise_arr(Xtr_native, b, lo, hi, eps=eps, rng=rng)
        Xte_q = quantise_arr(Xte_native, b, lo, hi, eps=eps, rng=rng)
        return one_nn_accuracy(Xtr_q, ytr, Xte_q, yte), nmse(Xte_q, Xte_native)

    if name == "unitary_obfusc":
        Xtr_eve = structural_obfuscate(
            Xtr_clean_q, rng, granularity="per-window")
        Xte_eve = structural_obfuscate(
            Xte_clean_q, rng, granularity="per-window")
        return one_nn_accuracy(Xtr_eve, ytr, Xte_eve, yte), det_nmse

    if name == "grouping":
        Xtr_g = subcarrier_group(Xtr_native, ng, rx_antennas)
        Xte_g = subcarrier_group(Xte_native, ng, rx_antennas)
        Xtr_q = quantise_arr(Xtr_g, b, lo, hi, eps=0.0, rng=None)
        Xte_q = quantise_arr(Xte_g, b, lo, hi, eps=0.0, rng=None)
        return one_nn_accuracy(Xtr_q, ytr, Xte_q, yte), nmse(Xte_q, Xte_native)

    if name == "sound_rate":
        # Quantise the throttled feature; NMSE is the gap to the
        # native-rate clean feature (i.e. cost of averaging fewer pkts).
        Xtr_q = quantise_arr(Xtr_amp, b, lo, hi, eps=0.0, rng=None)
        Xte_q = quantise_arr(Xte_amp, b, lo, hi, eps=0.0, rng=None)
        return one_nn_accuracy(Xtr_q, ytr, Xte_q, yte), nmse(Xte_q, Xte_native)

    raise ValueError(f"unknown defense {name}")


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def run(b, eps, n_seeds, seed_base, out_csv, sc_stride, ng_grid, rate_grid):
    print("Loading NTU-Fi HumanID")
    loader = NTUFiLoader("data/ntu_fi/NTU-Fi-HumanID")
    raw_tr, ytr, rx_tr = load_raw(loader, "train", sc_stride)
    raw_te, yte, rx_te = load_raw(loader, "test", sc_stride)
    rx_antennas = rx_tr or rx_te or 3
    print(f"  train {raw_tr.shape}  test {raw_te.shape}  rx={rx_antennas}")

    rows = []

    # Singletons
    for s in range(n_seeds):
        acc, nm = run_defense("deterministic", raw_tr, ytr, raw_te, yte,
                              b, eps, seed_base + s, ng=1,
                              n_pkts=NATIVE_PACKETS_PER_RECORD,
                              rx_antennas=rx_antennas)
        rows.append(dict(defense="deterministic", b=b, eps=None,
                          ng=1, n_pkts=NATIVE_PACKETS_PER_RECORD,
                          seed=s, attacker_acc=acc, nmse=nm))
        acc, nm = run_defense("dp_sq", raw_tr, ytr, raw_te, yte,
                              b, eps, seed_base + s, ng=1,
                              n_pkts=NATIVE_PACKETS_PER_RECORD,
                              rx_antennas=rx_antennas)
        rows.append(dict(defense="dp_sq", b=b, eps=eps,
                          ng=1, n_pkts=NATIVE_PACKETS_PER_RECORD,
                          seed=s, attacker_acc=acc, nmse=nm))
        acc, nm = run_defense("unitary_obfusc", raw_tr, ytr, raw_te, yte,
                              b, eps, seed_base + s, ng=1,
                              n_pkts=NATIVE_PACKETS_PER_RECORD,
                              rx_antennas=rx_antennas)
        rows.append(dict(defense="unitary_obfusc", b=b, eps=None,
                          ng=1, n_pkts=NATIVE_PACKETS_PER_RECORD,
                          seed=s, attacker_acc=acc, nmse=nm))

    for ng in ng_grid:
        for s in range(n_seeds):
            acc, nm = run_defense("grouping", raw_tr, ytr, raw_te, yte,
                                  b, eps, seed_base + s, ng=ng,
                                  n_pkts=NATIVE_PACKETS_PER_RECORD,
                                  rx_antennas=rx_antennas)
            rows.append(dict(defense="grouping", b=b, eps=None,
                              ng=ng, n_pkts=NATIVE_PACKETS_PER_RECORD,
                              seed=s, attacker_acc=acc, nmse=nm))
        accs = [r["attacker_acc"] for r in rows
                if r["defense"] == "grouping" and r["ng"] == ng]
        print(f"  grouping  Ng={ng:2d}  mean acc={np.mean(accs):.4f}")

    for npk in rate_grid:
        for s in range(n_seeds):
            acc, nm = run_defense("sound_rate", raw_tr, ytr, raw_te, yte,
                                  b, eps, seed_base + s, ng=1,
                                  n_pkts=npk, rx_antennas=rx_antennas)
            rows.append(dict(defense="sound_rate", b=b, eps=None,
                              ng=1, n_pkts=npk, seed=s,
                              attacker_acc=acc, nmse=nm))
        accs = [r["attacker_acc"] for r in rows
                if r["defense"] == "sound_rate" and r["n_pkts"] == npk]
        print(f"  sound_rate  n_pkts={npk:3d}  mean acc={np.mean(accs):.4f}")

    for name in ("deterministic", "dp_sq", "unitary_obfusc"):
        accs = [r["attacker_acc"] for r in rows if r["defense"] == name]
        nms = [r["nmse"] for r in rows if r["defense"] == name]
        print(f"  {name:>15s}  acc={np.mean(accs):.4f}  nmse={np.mean(nms):.4f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["defense", "b", "eps", "ng",
                                           "n_pkts", "seed",
                                           "attacker_acc", "nmse"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=DEFAULT_B)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--sc-stride", type=int, default=DEFAULT_SC_STRIDE)
    p.add_argument("--ng-grid", type=int, nargs="+", default=DEFAULT_NG_GRID)
    p.add_argument("--rate-grid", type=int, nargs="+",
                    default=DEFAULT_RATE_GRID)
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    args = p.parse_args()
    run(args.b, args.eps, args.n_seeds, args.seed_base,
        args.out_csv, args.sc_stride, args.ng_grid, args.rate_grid)


if __name__ == "__main__":
    main()
