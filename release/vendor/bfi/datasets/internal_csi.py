"""Loader for the HSG ICV WiFi Mesh Ghost Signal Dataset (`dataset-release/`).

This is the internal CSI corpus collected on the 14-node ESP32-C3 testbed
(802.11n, 2.4 GHz, HT20). It powers the IoT-J `Less-is-More` paper's ghost-
signal evaluation; for the BFI privacy-utility project (Thread A) we use it
as the *real-data* methodological substrate before the 180 GB BeamSense
download. The labels here are **presence** and **person_count** — there are
no per-person identity labels in this corpus.

Disk layout:

    <root>/
      manifest.json
      ground_truth/{camera_positions.json, confirmed_empty_periods.json}
      sessions/<session_id>/
        manifest.json
        samples_0000.jsonl.gz
        samples_0001.jsonl.gz
        ...

Each `samples_NNNN.jsonl.gz` file is a JSONL stream where each line is one
analysis window:

    {
      "timestamp": float,
      "window_id":  int,
      "window_start": float, "window_end": float,
      "csi_data": {
        "links": [{tx, rx, rssi_mean, rssi_values, csi_frames, packet_count}],
        "total_packets": int,
        "duration_ms":    float
      },
      "ground_truth": {
        "presence":  bool,
        "person_count": int,
        "positions": list, "confidence": float,
        "camera_data": [{camera_id, timestamp, person_count, persons: [...]}]
      }
    }

`csi_frames` is a list of 104-int lists (52 OFDM subcarriers × I/Q). The
loader assembles one canonical complex CSI tensor per window of shape
(n_links_canonical, n_subcarriers) where n_subcarriers = 52 and the link
ordering is a deterministic sort of all (tx, rx) MAC pairs observed in
the session (or globally, see `link_universe`).

Quick test usage:

    from bfi.datasets import InternalCSILoader

    loader = InternalCSILoader('dataset-release')
    print(loader.summary())              # session counts
    for rec in loader.iter_records(limit=4):
        print(rec.csi_complex.shape, rec.presence, rec.person_count)
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

N_SUBCARRIERS = 52
FRAME_INTS = 2 * N_SUBCARRIERS  # 104


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def csi_frame_to_complex(frame: list[int]) -> np.ndarray:
    """One 104-int [I0, Q0, I1, Q1, ...] frame → complex array of length 52."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.size != FRAME_INTS:
        # Some sessions vary; pad/trim defensively.
        if arr.size < FRAME_INTS:
            arr = np.pad(arr, (0, FRAME_INTS - arr.size))
        else:
            arr = arr[:FRAME_INTS]
    return arr[0::2] + 1j * arr[1::2]


def average_link_csi(csi_frames: list[list[int]]) -> np.ndarray:
    """Average complex CSI across the frames of one link in one window.

    Returns a complex array of shape (52,). For empty input, returns
    zeros (caller can detect via `np.all(... == 0)` if it cares).
    """
    if not csi_frames:
        return np.zeros(N_SUBCARRIERS, dtype=np.complex128)
    stack = np.stack([csi_frame_to_complex(f) for f in csi_frames], axis=0)
    return stack.mean(axis=0)


# ---------------------------------------------------------------------------
# Record + loader
# ---------------------------------------------------------------------------


@dataclass
class InternalCSIRecord:
    """One analysis window from the internal CSI corpus.

    Attributes:
        csi_complex:   complex array of shape (n_links, 52). Missing links
                       have rows of zeros.
        link_macs:     list of (tx, rx) tuples in the same order as rows of
                       `csi_complex`.
        presence:      bool, ground-truth occupancy in this window.
        person_count:  int, number of detected persons.
        session_id:    e.g. "20260213_211853".
        window_id:     int, monotonically increasing within session.
        timestamp:     POSIX timestamp.
    """

    csi_complex: np.ndarray
    link_macs: list[tuple[str, str]]
    presence: bool
    person_count: int
    session_id: str
    window_id: int
    timestamp: float


class InternalCSILoader:
    """Walks the HSG ICV CSI dataset and yields per-window records.

    Args:
        root:               path to `dataset-release/`.
        sessions:           optional list of session IDs to include
                            (default = all in `sessions/`).
        link_universe:      'global' (default) computes the union of all
                            (tx, rx) pairs across requested sessions and uses
                            it as the canonical row ordering. 'session' uses
                            a per-session ordering (rows differ between
                            sessions; better for memory but can't stack across
                            sessions). 'fixed' uses the explicit list passed
                            in `link_universe_list`.
        link_universe_list: explicit ordered list of (tx, rx) tuples, used
                            only when `link_universe='fixed'`.
    """

    def __init__(
        self,
        root: str,
        sessions: Optional[list[str]] = None,
        link_universe: str = "global",
        link_universe_list: Optional[list[tuple[str, str]]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"internal CSI root not found: {self.root}")
        self.sessions_dir = self.root / "sessions"
        if not self.sessions_dir.is_dir():
            raise FileNotFoundError(f"missing sessions/ under {self.root}")

        all_sessions = sorted(d.name for d in self.sessions_dir.iterdir() if d.is_dir())
        if sessions:
            missing = set(sessions) - set(all_sessions)
            if missing:
                raise FileNotFoundError(f"sessions not present: {sorted(missing)}")
            self.session_ids = list(sessions)
        else:
            self.session_ids = all_sessions

        self.link_universe_mode = link_universe
        if link_universe == "fixed":
            if not link_universe_list:
                raise ValueError("link_universe='fixed' requires link_universe_list")
            self._fixed_universe = list(link_universe_list)
        else:
            self._fixed_universe = None
        self._global_universe: Optional[list[tuple[str, str]]] = None

    # ------------------------------------------------------------------
    # Walking helpers
    # ------------------------------------------------------------------

    def _session_chunks(self, session_id: str) -> list[Path]:
        sd = self.sessions_dir / session_id
        files = sorted(sd.glob("samples_*.jsonl.gz"))
        if not files:
            files = sorted(sd.glob("samples_*.jsonl"))
        return files

    def _iter_raw_windows(self, session_id: str) -> Iterator[dict]:
        """Yield decoded JSON windows from one session, skipping metadata records."""
        for fp in self._session_chunks(session_id):
            opener = gzip.open if fp.suffix == ".gz" else open
            with opener(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("_metadata") or obj.get("_chunk_header") or obj.get("_footer"):
                        continue
                    yield obj

    # ------------------------------------------------------------------
    # Link universe
    # ------------------------------------------------------------------

    def _build_global_link_universe(self) -> list[tuple[str, str]]:
        """One pass over the corpus to collect every (tx, rx) MAC pair."""
        seen: set[tuple[str, str]] = set()
        for sid in self.session_ids:
            for w in self._iter_raw_windows(sid):
                for link in w.get("csi_data", {}).get("links", []):
                    tx = link.get("tx", "")
                    rx = link.get("rx", "")
                    if tx and rx:
                        seen.add((tx, rx))
        return sorted(seen)

    def link_universe(self) -> list[tuple[str, str]]:
        """Canonical (tx, rx) ordering for the requested link_universe mode."""
        if self.link_universe_mode == "fixed":
            return list(self._fixed_universe)
        if self.link_universe_mode == "global":
            if self._global_universe is None:
                self._global_universe = self._build_global_link_universe()
            return self._global_universe
        # 'session' mode: caller must provide a per-session universe via
        # iter_records(session_id=...). For convenience we still expose the
        # global one as a fallback.
        if self._global_universe is None:
            self._global_universe = self._build_global_link_universe()
        return self._global_universe

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def iter_records(
        self,
        limit: Optional[int] = None,
        session_filter: Optional[list[str]] = None,
    ) -> Iterator[InternalCSIRecord]:
        """Yield `InternalCSIRecord` instances across all selected sessions.

        Args:
            limit:           stop after this many records.
            session_filter:  if given, restrict to these session IDs (must
                             be a subset of `self.session_ids`).
        """
        sessions = session_filter or self.session_ids
        if self.link_universe_mode == "session":
            universe = None  # will be computed per session
        else:
            universe = self.link_universe()

        emitted = 0
        for sid in sessions:
            if self.link_universe_mode == "session":
                # Build per-session ordering on first read.
                seen = set()
                for w in self._iter_raw_windows(sid):
                    for link in w.get("csi_data", {}).get("links", []):
                        tx = link.get("tx", "")
                        rx = link.get("rx", "")
                        if tx and rx:
                            seen.add((tx, rx))
                universe = sorted(seen)
            link_idx = {pair: i for i, pair in enumerate(universe)}
            n_links = len(universe)

            for w in self._iter_raw_windows(sid):
                csi = np.zeros((n_links, N_SUBCARRIERS), dtype=np.complex128)
                for link in w.get("csi_data", {}).get("links", []):
                    tx, rx = link.get("tx", ""), link.get("rx", "")
                    if (tx, rx) not in link_idx:
                        continue
                    csi[link_idx[(tx, rx)]] = average_link_csi(link.get("csi_frames", []))

                gt = w.get("ground_truth", {}) or {}
                yield InternalCSIRecord(
                    csi_complex=csi,
                    link_macs=list(universe),
                    presence=bool(gt.get("presence", False)),
                    person_count=int(gt.get("person_count", 0)),
                    session_id=sid,
                    window_id=int(w.get("window_id", -1)),
                    timestamp=float(w.get("timestamp", 0.0)),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Lightweight summary: per-session window counts and presence ratios."""
        out = {"n_sessions": len(self.session_ids), "sessions": {}}
        for sid in self.session_ids:
            n = 0
            empty = 0
            occ = 0
            for w in self._iter_raw_windows(sid):
                n += 1
                gt = w.get("ground_truth", {}) or {}
                if gt.get("presence"):
                    occ += 1
                else:
                    empty += 1
            out["sessions"][sid] = {
                "n_windows": n,
                "empty": empty,
                "occupied": occ,
            }
        out["n_links_in_universe"] = len(self.link_universe())
        return out

    # ------------------------------------------------------------------
    # Quantisation helper for complex CSI
    # ------------------------------------------------------------------

    @staticmethod
    def quantise_complex(
        csi: np.ndarray,
        b_amp: int,
        b_phase: int,
        amp_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Deterministic-midpoint quantisation of a complex CSI array.

        Amplitude is quantised over [0, max(|csi|)] (or the supplied
        `amp_scale`); phase over [-π, π).

        Args:
            csi:        complex array of any shape.
            b_amp:      amplitude bit-depth (≥ 0; 0 collapses to a single value).
            b_phase:    phase bit-depth.
            amp_scale:  optional fixed upper bound for amplitude; default = max(|csi|).

        Returns:
            complex array of the same shape with quantised amplitude × exp(j quantised_phase).
        """
        amp = np.abs(csi)
        phase = np.angle(csi)
        scale = float(amp_scale) if amp_scale is not None else float(amp.max() + 1e-12)
        # Midpoint quantise amplitude
        if b_amp <= 0:
            amp_q = np.full_like(amp, scale / 2.0)
        else:
            n_bins = 1 << b_amp
            cell = scale / n_bins
            idx = np.clip(np.floor(amp / cell).astype(int), 0, n_bins - 1)
            amp_q = (idx + 0.5) * cell
        if b_phase <= 0:
            phase_q = np.zeros_like(phase)
        else:
            n_bins = 1 << b_phase
            cell = (2 * np.pi) / n_bins
            idx = np.clip(np.floor((phase + np.pi) / cell).astype(int), 0, n_bins - 1)
            phase_q = -np.pi + (idx + 0.5) * cell
        return amp_q * np.exp(1j * phase_q)

    @staticmethod
    def quantise_complex_dpsq_phase(
        csi: np.ndarray,
        b_amp: int,
        b_phase: int,
        eps_phase: float,
        rng: np.random.Generator | None = None,
        amp_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Like `quantise_complex` but applies DP-SQ to the phase axis.

        Amplitude uses the same deterministic midpoint quantiser as
        `quantise_complex` so that the only privacy knob between this
        method and the deterministic baseline is the DP randomisation on
        the phase.

        Args:
            csi:        complex array of any shape.
            b_amp:      amplitude bit-depth (deterministic, midpoint).
            b_phase:    phase bit-depth.
            eps_phase:  privacy budget per phase release (ε > 0).
            rng:        optional numpy Generator.
            amp_scale:  optional fixed upper bound for amplitude.

        Returns:
            complex array with shape == `csi.shape`.
        """
        from ..quantisation import _dp_sq_quantise_1d

        if rng is None:
            rng = np.random.default_rng()

        amp = np.abs(csi)
        phase = np.angle(csi)
        scale = float(amp_scale) if amp_scale is not None else float(amp.max() + 1e-12)

        # Amplitude: deterministic midpoint as in the baseline
        if b_amp <= 0:
            amp_q = np.full_like(amp, scale / 2.0)
        else:
            n_bins = 1 << b_amp
            cell = scale / n_bins
            idx = np.clip(np.floor(amp / cell).astype(int), 0, n_bins - 1)
            amp_q = (idx + 0.5) * cell

        # Phase: DP-SQ stochastic quantiser, circular in [-π, π)
        phase_q = _dp_sq_quantise_1d(
            phase, lo=-np.pi, hi=np.pi,
            b=b_phase, eps=eps_phase, rng=rng, circular=True,
        )

        return amp_q * np.exp(1j * phase_q)
