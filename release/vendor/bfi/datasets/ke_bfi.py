"""Loader for Ke's "WiFi-based Human Presence Detection Dataset (BFI format)".

Source:    IEEE DataPort, DOI 10.21227/0120-rj61
URL:       https://ieee-dataport.org/documents/wifi-based-human-presence-detection-dataset-bfi-format
Size:      72 MB (`dataset.zip`)
Substrate: BFI extracted from real 802.11ac packets — phi and psi values per subcarrier.
Labels:    binary presence (`1p` = occupied, `0p` = empty), encoded in filenames.
Configs:   3 environments (empty / meeting / general) × 3 bandwidths (20/40/80 MHz).
Storage:   one `.npy` file per recording; each file holds multiple packets,
           each with phi/psi tensors over subcarriers.

The exact directory layout and per-`.npy` tensor shape are not fully
documented on the IEEE DataPort page — the description says only that
"each `.npy` file contains multiple packets, and each packet includes
the phi and psi values for subcarriers". To remain robust we accept any
of these layouts:

    1. {root}/{env}/{bw_label}/{label}_{tag}.npy
       where {label} is "0p" or "1p", {bw_label} is e.g. "20MHz",
       and the .npy holds shape (n_packets, n_subcarriers, 2)
       (last axis = [phi, psi]).
    2. As above, but the .npy holds two interleaved arrays of shape
       (n_packets, n_subcarriers); we accept (n_packets, 2*n_subcarriers)
       and split.
    3. As above, but the .npy is a structured ndarray with named fields
       'phi' and 'psi'.
    4. {root}/{env}_{bw_label}_{label}_{tag}.npy   (flat layout)

Files whose name does not match any pattern above are skipped with a warning.

Usage:

    from bfi.datasets.ke_bfi import KeBFILoader

    loader = KeBFILoader('data/bfi/ke_bfi_presence')
    print(loader.summary())
    for rec in loader.iter_records(limit=4):
        print(rec.environment, rec.bandwidth_mhz, rec.presence,
              rec.phi.shape, rec.psi.shape)

If the dataset is not on disk, see
`docs/4-research/in-progress/bfi-privacy-research/open_datasets_survey.md` §1.

A synthetic fixture is provided in `bfi/datasets/ke_bfi_mock.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# Filename token patterns. Recognised label tokens encode the binary
# presence: '1p' (one person) → True; '0p' (zero people) → False.
_LABEL_RE = re.compile(r"(?<![a-zA-Z0-9])(?P<label>[01])p(?![a-zA-Z0-9])")
_BW_RE = re.compile(r"(?P<bw>20|40|80|160)\s*[mM][hH]?[zZ]?")
_ENV_TOKENS = ("empty", "meeting", "general", "office", "lab", "kitchen")


# ---------------------------------------------------------------------------
# Record + loader
# ---------------------------------------------------------------------------


@dataclass
class KeBFIRecord:
    """One recording (one .npy file) from the Ke BFI corpus.

    Attributes:
        phi:               (n_packets, n_subcarriers) array of φ angles.
        psi:               (n_packets, n_subcarriers) array of ψ angles.
        presence:          True if the recording is labelled '1p'.
        environment:       e.g. 'empty' / 'meeting' / 'general'.
        bandwidth_mhz:     20 / 40 / 80 / 160 (per filename token).
        path:              absolute path to the source .npy file.
    """

    phi: np.ndarray
    psi: np.ndarray
    presence: bool
    environment: str
    bandwidth_mhz: int
    path: str


class KeBFILoader:
    """Walks a Ke BFI presence corpus and yields per-recording records.

    Args:
        root:           absolute path to the unzipped dataset root.
        environments:   optional whitelist of environments.
        bandwidths:     optional whitelist of bandwidths (MHz, ints).
    """

    def __init__(
        self,
        root: str | Path,
        environments: Optional[tuple[str, ...]] = None,
        bandwidths: Optional[tuple[int, ...]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Ke BFI root not found: {self.root}")
        self.environments_filter = environments
        self.bandwidths_filter = bandwidths

    # ------------------------------------------------------------------
    # Walk + tag inference
    # ------------------------------------------------------------------

    def _all_npy_paths(self) -> list[Path]:
        return sorted(self.root.rglob("*.npy"))

    @staticmethod
    def _infer_tags(rel_path: Path) -> tuple[Optional[bool], Optional[str], Optional[int]]:
        """From a relative path, infer (presence, environment, bandwidth_mhz).

        Tries the directory tokens first, then the filename. Returns
        (None, None, None) for any token that cannot be inferred.
        """
        joined = str(rel_path).lower()
        # presence
        m = _LABEL_RE.search(joined)
        presence: Optional[bool] = None
        if m:
            presence = (m.group("label") == "1")
        # bandwidth
        m = _BW_RE.search(joined)
        bw: Optional[int] = int(m.group("bw")) if m else None
        # environment
        env: Optional[str] = None
        for token in _ENV_TOKENS:
            if token in joined:
                env = token
                break
        return presence, env, bw

    @staticmethod
    def _extract_phi_psi(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pull phi and psi out of an arbitrary ndarray layout."""
        # Structured ndarray with named fields
        if arr.dtype.names and "phi" in arr.dtype.names and "psi" in arr.dtype.names:
            return np.asarray(arr["phi"], dtype=float), np.asarray(arr["psi"], dtype=float)
        if arr.ndim >= 3 and arr.shape[-1] >= 2:
            # Shape (n_packets, ..., n_subcarriers, 2) or (n_packets, n_subcarriers, n_angles)
            phi = arr[..., 0]
            psi = arr[..., 1]
            return phi.astype(float), psi.astype(float)
        if arr.ndim == 2 and arr.shape[1] % 2 == 0:
            # Interleaved (n_packets, 2 * n_subcarriers)
            half = arr.shape[1] // 2
            return arr[:, :half].astype(float), arr[:, half:].astype(float)
        raise ValueError(
            f"Cannot extract (phi, psi) from array with shape {arr.shape} and dtype "
            f"{arr.dtype}; please update KeBFILoader._extract_phi_psi."
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def list_recordings(self) -> list[tuple[Path, bool, str, int]]:
        """Return (path, presence, env, bw) tuples for every recording on disk
        that we can fully label. Files with missing tags are skipped (with a
        printed warning) so the caller doesn't have to worry about partial rows.
        """
        out = []
        for path in self._all_npy_paths():
            rel = path.relative_to(self.root)
            presence, env, bw = self._infer_tags(rel)
            if presence is None or env is None or bw is None:
                print(f"[KeBFILoader] skip (tags missing): {rel}")
                continue
            if self.environments_filter and env not in self.environments_filter:
                continue
            if self.bandwidths_filter and bw not in self.bandwidths_filter:
                continue
            out.append((path, presence, env, bw))
        return out

    def iter_records(self, limit: Optional[int] = None) -> Iterator[KeBFIRecord]:
        """Yield `KeBFIRecord` instances for every recording on disk."""
        emitted = 0
        for path, presence, env, bw in self.list_recordings():
            try:
                raw = np.load(path, allow_pickle=False)
                phi, psi = self._extract_phi_psi(raw)
            except (ValueError, OSError) as e:
                print(f"[KeBFILoader] skip (load failed): {path.name}: {e}")
                continue
            yield KeBFIRecord(
                phi=phi, psi=psi,
                presence=presence,
                environment=env,
                bandwidth_mhz=bw,
                path=str(path),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Quick stats: counts per (environment, bandwidth) and per presence."""
        from collections import Counter
        recs = self.list_recordings()
        envs = Counter(r[2] for r in recs)
        bws = Counter(r[3] for r in recs)
        presence = Counter("1p" if r[1] else "0p" for r in recs)
        return {
            "total_files": len(recs),
            "environments": dict(envs),
            "bandwidths": dict(bws),
            "presence": dict(presence),
        }
