"""Loader for NTU-Fi (HumanID + HAR) corpora from the SenseFi benchmark.

Sources:
    SenseFi  — Yang et al., Patterns 2023.
    HumanID  — Yang et al., "CAUTION", IEEE ICASSP 2022 (gait re-id).

Repository:  https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
Data:        Google Drive folder linked from the repo, plus a Kaggle mirror
             (https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid).

Disk layout expected (matches `dataset.py:CSI_Dataset` in SenseFi):

    <root>/
      train_amp/
        <class_name>/  *.mat   (each .mat has fields 'CSIamp' and 'CSIphase')
      test_amp/
        <class_name>/  *.mat

Each `.mat` holds a CSI tensor that, after the SenseFi convention
`x = x[:, ::4].reshape(3, 114, 500)`, is shaped (3 antennas × 114
subcarriers × 500 packets). The raw shape on disk is (342, 2000).

NTU-Fi-HumanID:  14 gait identities, 546 train + 294 test = 840 samples.
NTU-Fi-HAR:       6 activities,    936 train + 264 test = 1200 samples.

We expose this loader in *amplitude* mode by default; pass
`modal="CSIphase"` to read the phase tensor instead. (For Thread A's
DP-SQ-on-phase head-to-head, phase is the relevant axis.)

Quick test usage:

    from bfi.datasets.ntu_fi import NTUFiLoader

    loader = NTUFiLoader('data/ntu_fi/HumanID', task='HumanID')
    print(loader.summary())
    for rec in loader.iter_records(split='train', limit=4):
        print(rec.label, rec.csi.shape, rec.path)

If the dataset is not on disk, see
`docs/4-research/in-progress/bfi-privacy-research/open_datasets_survey.md` §1.

A synthetic fixture is provided in `bfi/datasets/ntu_fi_mock.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Record + loader
# ---------------------------------------------------------------------------


@dataclass
class NTUFiRecord:
    """One CSI sample from NTU-Fi (HumanID or HAR).

    Attributes:
        csi:       float array of shape (3, 114, 500) — antennas × subcarriers × packets.
        label:     integer class label (0..n_classes-1).
        class_name: original directory name (e.g. 'subject_01').
        split:     'train' or 'test'.
        path:      absolute path to the source .mat file.
        modal:     'CSIamp' or 'CSIphase' (which field was loaded).
    """

    csi: np.ndarray
    label: int
    class_name: str
    split: str
    path: str
    modal: str = "CSIamp"


class NTUFiLoader:
    """Walks a NTU-Fi root and yields (csi, label) records.

    Args:
        root:    absolute path to the dataset root (the directory that
                 contains `train_amp/` and `test_amp/`).
        task:    'HumanID' (14 gait identities) or 'HAR' (6 activities).
                 Used only for documentation / summary output; the loader
                 derives labels purely from directory names.
        modal:   'CSIamp' (default) or 'CSIphase'.
        downsample_to_500: if True, reproduce SenseFi's `x[:, ::4]` then
                 reshape to (3, 114, 500). If False, return the raw shape.
    """

    def __init__(
        self,
        root: str | Path,
        task: str = "HumanID",
        modal: str = "CSIamp",
        downsample_to_500: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"NTU-Fi root not found: {self.root}")
        if task not in {"HumanID", "HAR"}:
            raise ValueError(f"unknown task {task!r}; expected HumanID or HAR")
        if modal not in {"CSIamp", "CSIphase"}:
            raise ValueError(f"unknown modal {modal!r}")
        self.task = task
        self.modal = modal
        self.downsample_to_500 = downsample_to_500

        # Build label map by union over both splits' class directories
        # so train and test share the same integer labels.
        class_names: set[str] = set()
        for split in ("train_amp", "test_amp"):
            split_root = self.root / split
            if not split_root.is_dir():
                continue
            for d in split_root.iterdir():
                if d.is_dir():
                    class_names.add(d.name)
        self.class_names = sorted(class_names)
        self.label_of = {name: i for i, name in enumerate(self.class_names)}

    # ------------------------------------------------------------------
    # Walking
    # ------------------------------------------------------------------

    def _split_root(self, split: str) -> Path:
        if split == "train":
            return self.root / "train_amp"
        if split == "test":
            return self.root / "test_amp"
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'test'")

    def list_paths(self, split: str) -> list[tuple[Path, int, str]]:
        """Return (path, label, class_name) for each .mat in a split."""
        sr = self._split_root(split)
        if not sr.is_dir():
            return []
        out = []
        for class_dir in sorted(sr.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self.label_of.get(class_dir.name, -1)
            if label < 0:
                continue
            for mat in sorted(class_dir.glob("*.mat")):
                out.append((mat, label, class_dir.name))
        return out

    # ------------------------------------------------------------------
    # CSI reshape per SenseFi convention
    # ------------------------------------------------------------------

    def _reshape_to_canonical(self, x: np.ndarray) -> np.ndarray:
        """Apply the SenseFi reshape: stride-4 downsample then (3, 114, 500).

        If `self.downsample_to_500` is False, return the array unmodified.
        Defensively handles raw shapes that don't match the documented
        (342, 2000) — most common alternates: already (3, 114, 500), or
        (1, ...) batch-of-one.
        """
        if not self.downsample_to_500:
            return x
        x = np.squeeze(x)
        if x.ndim == 3 and x.shape == (3, 114, 500):
            return x
        if x.ndim == 2 and x.shape == (342, 2000):
            x = x[:, ::4]
            return x.reshape(3, 114, 500)
        if x.ndim == 2 and x.shape == (342, 500):
            return x.reshape(3, 114, 500)
        # Last-resort fallback: try to stride-4 the second axis if it's
        # a multiple of 500 and the first axis equals 342.
        if x.ndim == 2 and x.shape[0] == 342 and x.shape[1] % 500 == 0:
            stride = x.shape[1] // 500
            return x[:, ::stride].reshape(3, 114, 500)
        raise ValueError(
            f"unexpected NTU-Fi sample shape {x.shape}; pass "
            f"downsample_to_500=False or extend NTUFiLoader._reshape_to_canonical."
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _read_mat(self, path: Path) -> np.ndarray:
        from scipy.io import loadmat
        data = loadmat(str(path))
        if self.modal not in data:
            raise KeyError(
                f"{path.name}: missing field {self.modal!r}; available: "
                f"{[k for k in data.keys() if not k.startswith('__')]}"
            )
        return np.asarray(data[self.modal])

    def iter_records(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> Iterator[NTUFiRecord]:
        """Yield `NTUFiRecord` instances for one split."""
        emitted = 0
        for path, label, class_name in self.list_paths(split):
            try:
                raw = self._read_mat(path)
                csi = self._reshape_to_canonical(raw).astype(float)
            except (KeyError, ValueError, OSError) as e:
                print(f"[NTUFiLoader] skip (load failed): {path.name}: {e}")
                continue
            yield NTUFiRecord(
                csi=csi,
                label=label,
                class_name=class_name,
                split=split,
                path=str(path),
                modal=self.modal,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Counts per (split, class)."""
        out: dict = {
            "task": self.task,
            "modal": self.modal,
            "n_classes": len(self.class_names),
            "class_names": self.class_names,
            "splits": {},
        }
        for split in ("train", "test"):
            paths = self.list_paths(split)
            from collections import Counter
            per_class = Counter(p[2] for p in paths)
            out["splits"][split] = {
                "total": len(paths),
                "per_class": dict(per_class),
            }
        return out
