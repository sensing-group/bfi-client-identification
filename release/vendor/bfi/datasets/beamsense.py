"""BeamSense dataset loader.

Reads the post-MATLAB-extraction `.mat` batches produced by
`BeamSense/BFI_Extraction/bfa_to_batches.m` (Haque, Zhang, Meneghello,
Restuccia, Computer Networks 2025).

Disk layout expected:

    <root>/
      Office/  Classroom/  Kitchen/
        processed_dataset/
          <station_id>/
            beamf_angles/
              A_batch/  B_batch/  ...  T_batch/
                <5char_person_id>batch_<n>.mat
                ...

Each `.mat` file contains a `bf_matrix` field of shape
    (window_size, dim, channels)
where `window_size` is typically 10 (BeamSense default), and
`(dim, channels)` are the per-packet BFA dimensions (subcarriers × Givens
angles, depending on the spatial-stream configuration).

The leading letter of the filename encodes the activity class (A → 0,
B → 1, …); the 4 characters following it encode the person identifier.
We expose both as labels so the privacy-utility analysis can compute
both I(activity; B_q) and I(identity; B_q).

If the dataset is not present on disk, see
`docs/4-research/in-progress/bfi-privacy-research/MASTER_PLAN.md` §2.2
for the download instructions (~180 GB; one-time setup).

Quick test usage with the synthetic fixture:

    from bfi.datasets.beamsense import BeamSenseLoader
    from bfi.datasets.beamsense_mock import make_mock_beamsense
    root = make_mock_beamsense('/tmp/beamsense_mock')
    loader = BeamSenseLoader(root)
    for record in loader.iter_records(limit=8):
        print(record.activity_label, record.person_id, record.bf_matrix.shape)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_beamsense_filename(filename: str) -> tuple[str, str, int]:
    """Decode a BeamSense .mat filename into (activity_letter, person_id, batch_idx).

    Expected pattern: `<activity><4char_person_suffix>batch_<n>.mat`.
    The activity letter is the first character of the 5-char `person_name`
    extracted by `bfa_to_batches.m` (which slices end-26:end-22 of the
    original pcap filename). The next four characters are the person id
    suffix.

    Args:
        filename: just the basename, no directory prefix.

    Returns:
        (activity_letter, person_id, batch_idx) where batch_idx is the
        integer parsed from `_<n>.mat`.

    Raises:
        ValueError: if the filename does not match the expected pattern.
    """
    base = os.path.basename(filename)
    if not base.endswith(".mat"):
        raise ValueError(f"not a .mat file: {filename}")
    # Strip extension
    stem = base[:-4]
    # The filename ends with `batch_<n>`; split on the last `batch_`.
    if "batch_" not in stem:
        raise ValueError(f"no 'batch_' marker in {filename}")
    head, _, tail = stem.rpartition("batch_")
    try:
        batch_idx = int(tail)
    except ValueError as e:
        raise ValueError(f"non-integer batch index in {filename}: {tail}") from e
    # `head` is the 5-char person name (activity + 4-char suffix).
    if len(head) < 1:
        raise ValueError(f"missing person prefix in {filename}")
    activity_letter = head[0].upper()
    person_id = head  # full 5-char prefix; uniquely identifies the person.
    return activity_letter, person_id, batch_idx


# ---------------------------------------------------------------------------
# Record + loader
# ---------------------------------------------------------------------------


@dataclass
class BeamSenseRecord:
    """One .mat batch from BeamSense, with parsed labels.

    Attributes:
        bf_matrix:        ndarray of shape (window_size, dim, channels), real-valued
                          BFA angles. Typically (10, dim, ch) with window_size=10.
        activity_label:   int in [0, n_activities); derived from filename leading char.
        activity_letter:  the original letter ('A', 'B', ...).
        person_id:        string, the 5-char person prefix.
        environment:      e.g. 'Classroom'.
        station:          e.g. '89'.
        path:             absolute path to the .mat file.
    """

    bf_matrix: np.ndarray
    activity_label: int
    activity_letter: str
    person_id: str
    environment: str
    station: str
    path: str


@dataclass
class BeamSenseSplit:
    """Train / val / test split (BeamSense default seed: 111).

    Each entry is a list of (path, activity_label, person_id) tuples.
    Use `BeamSenseLoader.materialise(split.train)` to load the .mat data
    for a given split.
    """

    train: list[tuple[str, int, str]]
    val:   list[tuple[str, int, str]]
    test:  list[tuple[str, int, str]]


class BeamSenseLoader:
    """Walks a BeamSense root and yields `BeamSenseRecord` instances.

    Args:
        root:               absolute path to the unzipped `BeamSense_Data` directory.
        environments:       which env subdirs to include; default = all that exist.
        stations:           which station IDs to include; default = all that exist.
        activity_to_label:  optional dict mapping activity letter → integer label.
                            Defaults to A=0, B=1, ... T=19, U=20, ...
    """

    DEFAULT_ENVS = ("Office", "Classroom", "Kitchen")

    def __init__(
        self,
        root: str | os.PathLike,
        environments: Optional[tuple[str, ...]] = None,
        stations: Optional[tuple[str, ...]] = None,
        activity_to_label: Optional[dict[str, int]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"BeamSense root not found: {self.root}")
        self.environments = tuple(environments) if environments else None
        self.stations = tuple(stations) if stations else None
        self.activity_to_label = activity_to_label or {
            chr(ord("A") + i): i for i in range(26)
        }

    # ------------------------------------------------------------------
    # Walk
    # ------------------------------------------------------------------

    def _iter_batch_dirs(self) -> Iterator[tuple[str, str, str, Path]]:
        """Yield (env, station, activity_letter, batch_dir_path) for each directory."""
        for env_dir in sorted(self.root.iterdir()):
            if not env_dir.is_dir():
                continue
            if self.environments and env_dir.name not in self.environments:
                continue
            proc_root = env_dir / "processed_dataset"
            if not proc_root.is_dir():
                continue
            for station_dir in sorted(proc_root.iterdir()):
                if not station_dir.is_dir():
                    continue
                station = station_dir.name
                if self.stations and station not in self.stations:
                    continue
                bf_root = station_dir / "beamf_angles"
                if not bf_root.is_dir():
                    continue
                for batch_dir in sorted(bf_root.iterdir()):
                    if not batch_dir.is_dir():
                        continue
                    name = batch_dir.name
                    if not (name.endswith("_batch") and len(name) >= 7):
                        continue
                    activity_letter = name[0].upper()
                    yield env_dir.name, station, activity_letter, batch_dir

    def list_paths(self) -> list[tuple[str, int, str, str, str]]:
        """Return (path, activity_label, person_id, env, station) tuples for every .mat
        file under the root, without loading any matrix data.

        Useful for splitting / sampling without reading 180 GB.
        """
        out = []
        for env, station, activity_letter, batch_dir in self._iter_batch_dirs():
            label = self.activity_to_label.get(activity_letter, -1)
            if label < 0:
                continue
            for mat_path in sorted(batch_dir.glob("*.mat")):
                try:
                    _, person_id, _ = parse_beamsense_filename(mat_path.name)
                except ValueError:
                    continue
                out.append((str(mat_path), label, person_id, env, station))
        return out

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _read_mat(self, path: str) -> np.ndarray:
        """Load `bf_matrix` from a single .mat file."""
        from scipy.io import loadmat

        d = loadmat(path)
        if "bf_matrix" not in d:
            raise KeyError(f"`bf_matrix` field missing in {path}; got {list(d.keys())}")
        return np.asarray(d["bf_matrix"])

    def iter_records(
        self,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> Iterator[BeamSenseRecord]:
        """Iterate `BeamSenseRecord` instances, loading bf_matrix from disk.

        Args:
            limit: stop after this many records; None = no limit.
            skip:  skip this many records before yielding (for sharding).
        """
        seen = 0
        emitted = 0
        for env, station, activity_letter, batch_dir in self._iter_batch_dirs():
            label = self.activity_to_label.get(activity_letter, -1)
            if label < 0:
                continue
            for mat_path in sorted(batch_dir.glob("*.mat")):
                if seen < skip:
                    seen += 1
                    continue
                seen += 1
                try:
                    _, person_id, _ = parse_beamsense_filename(mat_path.name)
                    bf = self._read_mat(str(mat_path))
                except (KeyError, ValueError) as e:
                    # Skip corrupt/unexpected files; warn once.
                    print(f"[BeamSenseLoader] skip {mat_path.name}: {e}")
                    continue
                yield BeamSenseRecord(
                    bf_matrix=bf,
                    activity_label=label,
                    activity_letter=activity_letter,
                    person_id=person_id,
                    environment=env,
                    station=station,
                    path=str(mat_path),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def materialise(
        self,
        entries: list[tuple[str, int, str]],
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        """Load .mat data for a list of (path, activity_label, person_id) tuples.

        Returns:
            bf_matrices:      list of arrays (shapes may differ across STAs).
            activity_labels:  int array of length len(entries).
            person_ids:       list of person id strings.
        """
        bf, labels, persons = [], [], []
        for path, label, person in entries:
            bf.append(self._read_mat(path))
            labels.append(label)
            persons.append(person)
        return bf, np.asarray(labels, dtype=int), persons

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def make_split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        seed: int = 111,
    ) -> BeamSenseSplit:
        """Build a stable train / val / test split using BeamSense's default seed.

        Replicates the rule used in `Learning_Models/create_csv_CNN.py`:
        per-file uniform draw; rand < train → train, rand < train+val → val,
        else → test.

        Args:
            train: fraction for train (default 0.7).
            val:   fraction for val   (default 0.15).
            seed:  RNG seed; default 111 matches the BeamSense paper's seed.
        """
        if train + val >= 1.0:
            raise ValueError("train + val must leave room for test (< 1.0)")
        rng = np.random.default_rng(seed)
        all_paths = self.list_paths()
        train_set, val_set, test_set = [], [], []
        for path, label, person, env, station in all_paths:
            r = float(rng.uniform(0.0, 1.0))
            triple = (path, label, person)
            if r < train:
                train_set.append(triple)
            elif r < train + val:
                val_set.append(triple)
            else:
                test_set.append(triple)
        return BeamSenseSplit(train=train_set, val=val_set, test=test_set)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Quick stats: file counts per (env, station, activity, person)."""
        from collections import Counter

        all_paths = self.list_paths()
        envs = Counter(p[3] for p in all_paths)
        stations = Counter(p[4] for p in all_paths)
        activities = Counter(p[1] for p in all_paths)
        persons = Counter(p[2] for p in all_paths)
        return {
            "total_files": len(all_paths),
            "environments": dict(envs),
            "stations": dict(stations),
            "activities": dict(activities),
            "persons": dict(persons),
            "n_persons": len(persons),
            "n_activities": len(activities),
        }
