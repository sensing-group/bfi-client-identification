#!/usr/bin/env python3
"""Test bench: score a defense of beamforming feedback against the paper's
privacy definition (Section 7.1) without editing the paper's scripts.

The definition. Feedback is private if
  (1) a training-free 1-nearest-neighbour observer that enrolls one window per
      target and may read anything the protection leaves stays at chance: the
      one-sided upper 95 % bound of its accuracy over seeds is at most
      chance + alpha (default 1 point), for every observer below, and
  (2) the access point keeps at least beta (default 0.95) of the beamforming
      gain of unquantized feedback, measured as |v1_hat^H v1|^2 on the first
      stream.

What the bench runs (He et al. positions dataset, Conference Room 1, the
split, seeds and random streams of joint_defense_eval.py):
  observers  cosine 1-NN on the 19 held-out positions, K = 1 enrolled window,
             T = 1 and 32 query windows averaged, on
               scaled     the reported pair-0 angles, standardised with the
                          statistics of unprotected calibration reports
               raw        the same angles, unscaled
               invariant  |v_k^H v_0| between subcarrier groups, which a
                          transform shared by the groups leaves readable
  gain       he_beamforming_gain.py's draw of 3000 reports per seed (all 39
             positions), V built from the unquantized angles as the truth
Nothing is re-derived: data, split and scalers are joint_defense_eval.He,
decisions are joint_defense_common.multi_query_eval, the gain is
he_beamforming_gain.gains, and each built-in defense calls the function that
produced the paper's numbers.

THE INTERFACE. A defense is a subclass of `Defense` with two methods.

    report(angles, targets, rng) -> (reported, secret)
        angles    (phi, psi), each float64 of shape (n, 59, 6): the unquantized
                  Givens angles of n reports, 59 subcarrier groups, in 802.11
                  order (joint_defense_common.load_he_angles). phi lies in
                  [-pi, pi), psi in [0, pi/2].
        targets   (n,) int: the client each report belongs to. The access
                  point knows whom it sounds, so a secret may depend on it.
        rng       numpy Generator; the only source of randomness to use.
        reported  (phi, psi) of the same shapes: the angle VALUES on the air,
                  after quantisation. Every observer reads these and only these.
        secret    whatever the access point keeps to undo the defense (None
                  for client-side noise). It is never shown to an observer.
      It is called twice per seed, once with every held-out report and once
      with the gain draw, each time with all reports of the call at once.
      Client-side noise acts on the angles before or during quantisation.
      An access-point-side transform acts on V = C.givens_to_v(*angles) and
      reports C.he_quantise(*C.v_to_givens(V_transformed)).

    recover(reported, secret) -> V_hat
        V_hat     complex (n, 59, 4, 3): what the access point steers with.
      The default rebuilds V from the reported angles (phi wrapped, psi
      clipped); override it to undo a transform with the secret.

The codebook of the run (--bits) is in force inside both calls:
C.he_quantise(phi, psi) quantises at it, `codebook_bits()` and `cell_widths()`
return it. Example, a complete client-side defense (examples/defense_example.py):

    import numpy as np
    import joint_defense_common as C
    from defense_bench import Defense, cell_widths

    class Dither(Defense):
        name = "dither"

        def __init__(self, cells: float = 0.5):
            self.cells = cells

        def report(self, angles, targets, rng):
            phi, psi = angles
            w_phi, w_psi = cell_widths()
            phi = phi + self.cells * w_phi * rng.uniform(-0.5, 0.5, phi.shape)
            psi = psi + self.cells * w_psi * rng.uniform(-0.5, 0.5, psi.shape)
            phi = np.mod(phi - C.PHI_LO, C.PHI_HI - C.PHI_LO) + C.PHI_LO
            return C.he_quantise(phi, np.clip(psi, C.PSI_LO, C.PSI_HI)), None

Usage:
    python3 scripts/defense_bench.py --list
    python3 scripts/defense_bench.py --selftest
    python3 scripts/defense_bench.py --defense rotation_group
    python3 scripts/defense_bench.py --defense dpsq --param eps=0.5 --bits 6/4
    python3 scripts/defense_bench.py --defense examples/defense_example.py:Dither --param cells=0.5
Output: a table on stdout and a JSON file (--out, default figures/bench/).
Data location and memory caps: the environment variables of
joint_defense_common.py (BFI_HE_ROOT, BFI_HE_LOCATION_ROOT, BFI_VM_LIMIT_GB,
BFI_MAX_RSS_MB). Working set about 1.3 GB; seeds are processed one at a time.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

# One BLAS thread, as r_raw_observer_defenses.py: batched small QRs run slower threaded.
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402
from scipy import stats as SS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beamdancer_sweep as B  # noqa: E402
import he_beamforming_gain as G  # noqa: E402
import joint_defense_common as C  # noqa: E402
import joint_defense_eval as J  # noqa: E402
import manifold_sweep as M  # noqa: E402
import r_raw_observer_defenses as RD  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.01                      # clause 1: margin over chance
DEFAULT_BETA = 0.95                       # clause 2: share of the unquantized gain
UPPER_LEVEL = 0.95                        # one-sided bound over seeds (r_eval_checks.UPPER_LEVEL)
CODEBOOKS = {"4/2": (C.HE_B_PHI, C.HE_B_PSI), "6/4": tuple(G.CODEBOOK_FINE), "9/7": (9, 7)}
DEFAULT_BITS = "4/2"
OBSERVERS = ("scaled", "raw", "invariant")
T_REPORT = RD.T_REPORT                    # (1, 32); J.T_GRID is scored to keep the query streams
INV_ATTACKER_ID = 2                       # joint_defense_eval scores nn1_inv third
OBSERVER_QUERY_ID = {"scaled": RD.NN1_ATTACKER_ID, "raw": RD.NN1_ATTACKER_ID,
                     "invariant": INV_ATTACKER_ID}
USER_STREAM_ID = 1000                     # random-stream id of a defense the paper did not score
DB_PER_DECADE = 10.0
PCT = 100.0
EST_LOAD_FACTOR = 2.5                     # working set over the size of BFI.pth, as the sibling scripts
OUT_DIR = C.FIG_DIR / "bench"

DEFAULT_DPSQ_EPS = 0.5
DEFAULT_MANIFOLD_EPS = 512.0
DEFAULT_BEAMDANCER_R = 100
ROTATIONS = {                             # bench name -> joint_defense_eval's name
    "rotation_report": "unitary_pkt",
    "rotation_group": "unitary_pkt_sc",
    "rotation_fixed": "unitary_fixed",
    "rotation_shared": "unitary_global",
}
BEAMDANCER_NOTE = ("BeamDancer (Cominelli et al.): uniform offset over R % of each angle's range "
                   "added to every quantised angle, reflected at the edges; no key")

# Synthetic dataset of --selftest: positions with a smooth channel over the subcarrier groups.
SYN_SEED = 7
SYN_POSITIONS = 20
SYN_CAL_POSITIONS = 10
SYN_WINDOWS = 200                         # reports per position
SYN_SUBCARRIERS = 59
SYN_TILT = 1.0                            # weight of the frequency-dependent part of the channel
SYN_NOISE = 0.15                          # per-report noise on the channel matrix
SELFTEST_SEEDS = 5
SELFTEST_ALPHA = 0.05                     # ten positions and five seeds resolve no finer margin
SELFTEST_GAIN_REPORTS = 1000
SELFTEST_NONE_FACTOR = 3.0                # 'none' must exceed this many times chance
SELFTEST_MIN_GAIN = 0.9

Angles = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Codebook in force
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _patched(module, **values) -> Iterator[None]:
    """Module attributes set for the duration of a block (r_round4_checks.patched)."""
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def codebook(bits: tuple[int, int]):
    """joint_defense_common quantising at (phi bits, psi bits): he_quantise, DP-SQ
    and Manifold DP read these constants when called (r_round4_checks.codebook)."""
    return _patched(C, HE_B_PHI=bits[0], HE_B_PSI=bits[1], B_HEADLINE=bits[0])


def codebook_bits() -> tuple[int, int]:
    """(phi bits, psi bits) of the codebook in force."""
    return C.HE_B_PHI, C.HE_B_PSI


def cell_widths() -> tuple[float, float]:
    """Width in radians of one quantisation cell of phi and of psi at the codebook in force."""
    b_phi, b_psi = codebook_bits()
    return (C.PHI_HI - C.PHI_LO) / (1 << b_phi), (C.PSI_HI - C.PSI_LO) / (1 << b_psi)


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Streams:
    """Seeds of the random streams of one seed: the defense's and each observer's queries."""
    report: tuple[int, ...]
    query: dict[str, tuple[int, ...]]


class Defense:
    """A defense of beamforming feedback; see the module docstring for the contract."""

    name: str = "defense"
    stream_id: int = USER_STREAM_ID

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        """(angle values on the air, secret the access point keeps) for n reports."""
        raise NotImplementedError

    def recover(self, reported: Angles, secret: Any) -> np.ndarray:
        """V_hat (n, K, 4, 3) the access point steers with; default: V of the reported angles."""
        return G.decode(*reported)

    def _streams(self, seed: int, bits: tuple[int, int]) -> Streams:
        """Bench bookkeeping, not part of the contract: joint_defense_eval's streams
        under `stream_id`, so that a defense the paper scored reproduces its cell."""
        return Streams(
            report=(seed, self.stream_id, RD.HELD_STREAM),
            query={o: (seed, self.stream_id, OBSERVER_QUERY_ID[o], RD.QUERY_STREAM) for o in OBSERVERS})


def _joint_stream_id(joint_name: str) -> int:
    """Stream id of a defense in joint_defense_eval's list, else the id of an unscored one."""
    return J.HE_DEFENSES.index(joint_name) if joint_name in J.HE_DEFENSES else USER_STREAM_ID


# ---------------------------------------------------------------------------
# Built-in defenses: thin wrappers over the functions behind the paper's numbers
# ---------------------------------------------------------------------------
class NoDefense(Defense):
    """Deterministic codebook quantisation only."""

    name = "none"
    stream_id = _joint_stream_id(name)

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        return C.he_quantise(*angles), None


class DpSq(Defense):
    """DP-SQ (Seif et al.): stochastic rounding of every angle with budget eps."""

    def __init__(self, eps: float = DEFAULT_DPSQ_EPS):
        self.eps = float(eps)
        self.name = f"dpsq_e{self.eps}"
        self.stream_id = _joint_stream_id(self.name)

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        return C.he_quantise(*angles, eps=self.eps, rng=rng), None


class ManifoldDp(Defense):
    """Manifold DP: Gaussian noise of budget eps on every angle, then the codebook grid.

    The noise of each Givens pair is drawn in turn, pair 0 first: that draw is the
    one joint_defense_eval and manifold_sweep make for the features the observer reads.
    """

    def __init__(self, eps: float = DEFAULT_MANIFOLD_EPS):
        self.eps = float(eps)
        self.name = M._defense(self.eps)
        swept = [M.REFERENCE] + [M._defense(e) for e in M.SWEEP_EPS]
        self.stream_id = M._stream(self.name, swept.index(self.name)) if self.name in swept \
            else USER_STREAM_ID

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        phi, psi = angles
        out_phi, out_psi = np.empty_like(phi), np.empty_like(psi)
        for pair in range(phi.shape[-1]):
            noisy = C.manifold_dp(C.he_features(phi, psi, pair), C.PHI_LO, C.PHI_HI, self.eps, rng)
            pair_values = noisy.reshape(phi.shape[0], phi.shape[1], 2)
            out_phi[..., pair], out_psi[..., pair] = pair_values[..., 0], pair_values[..., 1]
        return (out_phi, out_psi), None


class BeamDancer(Defense):
    """BeamDancer Algorithm 1 with randomization R (per cent of each angle's range)."""

    def __init__(self, R: int = DEFAULT_BEAMDANCER_R):
        self.r_pct = int(R)
        self.name = f"beamdancer_R{self.r_pct}"

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        return B.report(*angles, codebook_bits(), self.r_pct, rng), None

    def _streams(self, seed: int, bits: tuple[int, int]) -> Streams:
        """beamdancer_sweep's streams, keyed by R and the codebook's phi bits."""
        base = (seed, B.OBSERVER_STREAM + self.r_pct, bits[0])
        query = {o: base + (B.QUERY_STREAM,) for o in OBSERVERS}
        query["invariant"] = base + (INV_ATTACKER_ID, B.QUERY_STREAM)
        return Streams(report=base, query=query)


class _HaarRecorder:
    """joint_defense_common.haar_unitary that keeps what it drew, so that the
    access point's secret is the Q joint_defense_eval.He._unitary applied."""

    def __init__(self, draw: Callable[..., np.ndarray]):
        self.draw = draw
        self.drawn: list[np.ndarray] = []

    def __call__(self, n: int, size: tuple, rng: np.random.Generator) -> np.ndarray:
        q = self.draw(n, size, rng)
        self.drawn.append(q)
        return q

    def matrix(self, groups: np.ndarray) -> np.ndarray:
        """Q per report, (n, 1 or K, nr, nr): one draw per group id (fixed variants)
        or one batch per chunk of reports (fresh variants)."""
        if self.drawn[0].ndim == 2:
            per_group = np.stack(self.drawn)[np.searchsorted(np.unique(groups), groups)]
            return per_group[:, None]
        q = np.concatenate(self.drawn)
        return q[:, None] if q.ndim == 3 else q


class Rotation(Defense):
    """The access point precodes its sounding with a Haar unitary Q, the client
    reports Q^H V, and the access point steers with Q V_hat. `variant` is
    joint_defense_eval's name and sets who shares a Q and for how long."""

    def __init__(self, variant: str):
        self.variant = variant
        self.name = next(k for k, v in ROTATIONS.items() if v == variant)
        self.stream_id = _joint_stream_id(variant)

    def report(self, angles: Angles, targets: np.ndarray,
               rng: np.random.Generator) -> tuple[Angles, Any]:
        recorder = _HaarRecorder(C.haar_unitary)
        with _patched(C, haar_unitary=recorder):
            # He._unitary reads none of its instance's state.
            rotated = J.He._unitary(None, *angles, targets, self.variant, rng)
        return C.he_quantise(*rotated), recorder.matrix(J._groups(self.variant, targets))

    def recover(self, reported: Angles, secret: Any) -> np.ndarray:
        return secret @ G.decode(*reported)


def _rotation_factory(variant: str) -> Callable[[], Defense]:
    return lambda: Rotation(variant)


BUILTINS: dict[str, tuple[Callable[..., Defense], str]] = {
    "none": (NoDefense, J.DEFENSE_NOTES["none"]),
    "dpsq": (DpSq, f"{J.DEFENSE_NOTES['dpsq']}; --param eps= (default {DEFAULT_DPSQ_EPS})"),
    "manifold": (ManifoldDp,
                 f"{J.DEFENSE_NOTES['manifold']}; --param eps= (default {DEFAULT_MANIFOLD_EPS:g})"),
    "beamdancer": (BeamDancer, f"{BEAMDANCER_NOTE}; --param R= (default {DEFAULT_BEAMDANCER_R})"),
    **{name: (_rotation_factory(variant), J.DEFENSE_NOTES[variant])
       for name, variant in ROTATIONS.items()},
}


def _parse_value(text: str) -> int | float | str:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_params(items: list[str]) -> dict[str, int | float | str]:
    """['eps=0.5', 'R=100'] -> {'eps': 0.5, 'R': 100}."""
    bad = [i for i in items if "=" not in i]
    if bad:
        raise ValueError(f"--param expects k=v, got {bad}")
    return {k: _parse_value(v) for k, v in (i.split("=", 1) for i in items)}


def load_defense(spec: str, params: dict[str, Any]) -> Defense:
    """A built-in by name, or 'path/to/file.py:object' (a Defense subclass, built
    with the parameters, or a ready instance)."""
    if spec in BUILTINS:
        return BUILTINS[spec][0](**params)
    path, sep, attr = spec.rpartition(":")
    if not sep or not Path(path).is_file():
        raise ValueError(f"unknown defense {spec!r}: a built-in ({', '.join(BUILTINS)}) "
                         f"or path/to/file.py:object")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    obj = getattr(module, attr)
    defense = obj(**params) if isinstance(obj, type) else obj
    if not isinstance(defense, Defense):
        raise TypeError(f"{spec} is not a defense_bench.Defense")
    if params and not isinstance(obj, type):
        raise ValueError(f"{spec} is an instance; --param needs a class")
    return defense


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class BenchData:
    """The held-out split with its scalers (joint_defense_eval.He, fitted at the
    codebook in force) and every report of the room for the gain draw."""
    he: J.He
    phi: np.ndarray
    psi: np.ndarray
    y: np.ndarray
    source: str


def synthetic_angles() -> dict:
    """An in-memory stand-in for load_he_angles: SYN_POSITIONS positions, each a
    random 4x3 channel that tilts smoothly over the subcarrier groups, SYN_WINDOWS
    noisy reports per position."""
    rng = np.random.default_rng(SYN_SEED)
    shape = (C.HE_NR, C.HE_NC)

    def cplx(*size: int) -> np.ndarray:
        return rng.standard_normal(size + shape) + 1j * rng.standard_normal(size + shape)

    tilt = (SYN_TILT * np.arange(SYN_SUBCARRIERS) / SYN_SUBCARRIERS)[:, None, None]
    phi, psi, y = [], [], []
    for position in range(SYN_POSITIONS):
        channel = cplx() + tilt * cplx()                                    # (K, nr, nc)
        noisy = channel[None] + SYN_NOISE * cplx(SYN_WINDOWS, SYN_SUBCARRIERS)
        p, s = C.v_to_givens(np.linalg.qr(noisy)[0])
        phi.append(p)
        psi.append(s)
        y.append(np.full(SYN_WINDOWS, position, np.int64))
    phi, psi = np.concatenate(phi), np.concatenate(psi)
    return dict(phi=phi, psi=psi, y=np.concatenate(y), raw_shape=phi.shape + (2,))


def load_data(synthetic: bool) -> BenchData:
    """Build the split at the codebook in force (call inside `codebook`)."""
    d = synthetic_angles() if synthetic else C.load_he_angles()
    n_cal = SYN_CAL_POSITIONS if synthetic else C.HE_N_CALIB_CLASSES
    with _patched(C, load_he_angles=lambda: d, HE_N_CALIB_CLASSES=n_cal):
        he = J.He(J.HE_CAL_PER_CLASS)
    source = "synthetic (selftest)" if synthetic else str(C.HE_ROOT.name)
    return BenchData(he=he, phi=d["phi"], psi=d["psi"], y=d["y"], source=source)


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------
def _check_reported(reported: Angles, angles: Angles, name: str) -> None:
    for got, want, label in zip(reported, angles, ("phi", "psi")):
        if np.shape(got) != want.shape or not np.all(np.isfinite(got)):
            raise ValueError(f"{name}.report: reported {label} must be finite with shape "
                             f"{want.shape}, got {np.shape(got)}")


def observer_accuracy(defense: Defense, data: BenchData, seed: int,
                      bits: tuple[int, int]) -> dict[tuple[str, int], float]:
    """{(observer, T): accuracy} on the held-out positions for one seed."""
    he = data.he
    y = he.y["held"]
    angles = (he.phi["held"], he.psi["held"])
    streams = defense._streams(seed, bits)
    reported, _ = defense.report(angles, y, np.random.default_rng(streams.report))
    _check_reported(reported, angles, defense.name)
    raw = C.he_features(*reported)
    views = dict(scaled=he.S0.transform(raw), raw=raw,
                 invariant=he.S0_inv.transform(he._gram_invariant(*reported)))
    del reported
    supports = C.pick_supports(y, seed)
    acc = {}
    for observer in OBSERVERS:
        scored = RD.score(views[observer], y, supports, streams.query[observer])
        acc.update({(observer, T): scored[T] for T in T_REPORT})
    del views, raw
    gc.collect()
    return acc


def first_stream_gain(defense: Defense, data: BenchData, seed: int, n_reports: int) -> float:
    """Mean |v1_hat^H v1|^2 over he_beamforming_gain's report draw of one seed; the
    defense continues the generator that drew the reports, as beamdancer_sweep does."""
    rng = np.random.default_rng([seed, M.GAIN_STREAM])
    idx = np.sort(rng.choice(len(data.y), size=n_reports, replace=False))
    angles = (data.phi[idx], data.psi[idx])
    reported, secret = defense.report(angles, data.y[idx], rng)
    _check_reported(reported, angles, defense.name)
    v_hat = defense.recover(reported, secret)
    gain = float(G.gains(v_hat, C.givens_to_v(*angles))[0].mean())
    del v_hat, reported, secret
    gc.collect()
    return gain


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def stat(values: list[float]) -> dict[str, float]:
    """Mean, sd and one-sided upper bound of the mean over seeds (r_round4_checks.stat)."""
    v = np.asarray(values, float)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    ub = mean if v.size < 2 else mean + float(SS.t.ppf(UPPER_LEVEL, v.size - 1)) * sd / math.sqrt(v.size)
    return dict(mean=mean, sd=sd, ub95=ub, n=int(v.size))


def summarise(defense: Defense, data: BenchData, bits_name: str, acc: dict, gains: list[float],
              alpha: float, beta: float, runtime_s: float) -> dict:
    """The two clauses of the definition from the per-seed accuracies and gains."""
    chance = 1.0 / len(np.unique(data.he.y["held"]))
    observers = {f"{o}|T={T}": stat(acc[(o, T)]) for o in OBSERVERS for T in T_REPORT}
    worst = max(observers, key=lambda k: observers[k]["ub95"])
    gain = stat(gains)
    private = bool(observers[worst]["ub95"] <= chance + alpha)
    keeps_gain = bool(gain["mean"] >= beta)
    return dict(
        defense=defense.name, dataset=data.source, codebook=bits_name, seeds=gain["n"],
        chance=chance, alpha=alpha, beta=beta, upper_level=UPPER_LEVEL,
        n_held_reports=int(len(data.he.y["held"])), observers=observers,
        worst_observer=dict(name=worst, **observers[worst]),
        gain_first_stream=dict(mean=gain["mean"], sd=gain["sd"], n=gain["n"]),
        loss_db=-DB_PER_DECADE * math.log10(gain["mean"]),
        verdict=dict(private=private, keeps_gain=keeps_gain, passes=private and keeps_gain),
        runtime_s=runtime_s, peak_rss_mb=C.peak_rss_mb())


def run_bench(defense: Defense, *, bits_name: str = DEFAULT_BITS, seeds: int = J.DEFAULT_SEEDS,
              alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
              gain_reports: int = G.N_REPORTS, synthetic: bool = False) -> dict:
    """Score one defense at one codebook; returns the summary that is printed and written."""
    timer = C.Timer()
    bits = CODEBOOKS[bits_name]
    acc: dict[tuple[str, int], list[float]] = {(o, T): [] for o in OBSERVERS for T in T_REPORT}
    gains: list[float] = []
    with codebook(bits):
        data = load_data(synthetic)
        C.check_rss("data loaded")
        for seed in range(1, seeds + 1):
            for key, value in observer_accuracy(defense, data, seed, bits).items():
                acc[key].append(value)
            gains.append(first_stream_gain(defense, data, seed, gain_reports))
            C.check_rss(f"{defense.name} seed {seed}")
        return summarise(defense, data, bits_name, acc, gains, alpha, beta, timer())


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_table(s: dict) -> None:
    """The summary of one run as a table."""
    print(f"\ndefense {s['defense']}   data {s['dataset']}   codebook {s['codebook']} bits   "
          f"{s['seeds']} seeds   chance {PCT * s['chance']:.2f} %")
    print(f"  {'observer':<10s} {'T':>3s} {'acc %':>7s} {'sd':>6s} {'ub95 %':>7s}")
    for key, o in s["observers"].items():
        name, t = key.split("|T=")
        print(f"  {name:<10s} {t:>3s} {PCT * o['mean']:7.2f} {PCT * o['sd']:6.2f} {PCT * o['ub95']:7.2f}")
    w, g, v = s["worst_observer"], s["gain_first_stream"], s["verdict"]
    print(f"  clause 1  worst observer {w['name']}: upper bound {PCT * w['ub95']:.2f} % against "
          f"chance + alpha = {PCT * (s['chance'] + s['alpha']):.2f} %   {_verdict(v['private'])}")
    print(f"  clause 2  first-stream gain {g['mean']:.4f} (sd {g['sd']:.4f}), loss {s['loss_db']:.2f} dB, "
          f"against beta = {s['beta']:.2f}   {_verdict(v['keeps_gain'])}")
    print(f"  verdict   {_verdict(v['passes'])}   "
          f"({s['runtime_s']:.0f} s, peak RSS {s['peak_rss_mb']:.0f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------
def selftest() -> dict[str, dict]:
    """On the synthetic set: 'none' is identified well above chance, and the
    rotation per group is at chance with the gain kept. Raises AssertionError."""
    kwargs = dict(seeds=SELFTEST_SEEDS, alpha=SELFTEST_ALPHA, gain_reports=SELFTEST_GAIN_REPORTS,
                  synthetic=True)
    none = run_bench(load_defense("none", {}), **kwargs)
    rotation = run_bench(load_defense("rotation_group", {}), **kwargs)
    for s in (none, rotation):
        print_table(s)
    scaled = none["observers"][f"scaled|T={T_REPORT[0]}"]["mean"]
    if scaled < SELFTEST_NONE_FACTOR * none["chance"]:
        raise AssertionError(f"'none' at {scaled:.3f}, under {SELFTEST_NONE_FACTOR} x chance")
    if none["verdict"]["private"]:
        raise AssertionError("'none' judged private")
    if not rotation["verdict"]["private"]:
        raise AssertionError(f"rotation_group above chance: {rotation['worst_observer']}")
    if rotation["gain_first_stream"]["mean"] < SELFTEST_MIN_GAIN:
        raise AssertionError(f"rotation_group gain {rotation['gain_first_stream']['mean']:.3f}")
    return dict(none=none, rotation_group=rotation)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def default_out(summary: dict) -> Path:
    return OUT_DIR / f"{summary['defense']}_b{summary['codebook'].replace('/', '')}.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--defense", help="a built-in name (--list) or path/to/file.py:object")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="parameter of the defense, repeatable")
    p.add_argument("--bits", choices=list(CODEBOOKS), default=DEFAULT_BITS,
                   help="codebook, phi/psi bits")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="margin over chance (clause 1)")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="gain to keep (clause 2)")
    p.add_argument("--seeds", type=int, default=J.DEFAULT_SEEDS)
    p.add_argument("--gain-reports", type=int, default=G.N_REPORTS, help="reports per seed in the gain draw")
    p.add_argument("--out", type=Path, help=f"JSON path (default {OUT_DIR.relative_to(C.REPO_ROOT)}/)")
    p.add_argument("--list", action="store_true", help="print the built-in defenses")
    p.add_argument("--selftest", action="store_true", help="run on a synthetic dataset, no data files")
    p.add_argument("--dry-run", action="store_true", help="print the memory estimate and stop")
    a = p.parse_args(argv)
    if a.list:
        for name, (_, note) in BUILTINS.items():
            print(f"{name:<16s} {note}")
        return 0
    C.set_vm_limit()
    if a.selftest:
        selftest()
        print("selftest OK")
        return 0
    if not a.defense:
        p.error("--defense, --list or --selftest is required")
    defense = load_defense(a.defense, parse_params(a.param))
    if a.dry_run:
        print(f"{defense.name}: {a.seeds} seeds at {a.bits} bits; estimated working set "
              f"~{C.he_estimate_mb() * EST_LOAD_FACTOR:.0f} MB (VM cap {C.VM_LIMIT_GB} GB, "
              f"RSS cap {C.MAX_RSS_MB} MB)")
        return 0
    summary = run_bench(defense, bits_name=a.bits, seeds=a.seeds, alpha=a.alpha, beta=a.beta,
                        gain_reports=a.gain_reports)
    print_table(summary)
    out = a.out or default_out(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    # Run as the importable module, so a user's `from defense_bench import Defense` is this class.
    import defense_bench
    sys.exit(defense_bench.main())
