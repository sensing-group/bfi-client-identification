#!/usr/bin/env python3
"""C5b shared loader: the (B) mesh-capture windows of both sources (July
rotation, September live file) on one geometry vocabulary, with the bin mask
of c5_cross_receiver (bins fully covered in EVERY window of the devices
compared) and a converter to the r_common window layout so r_eval's Table,
episodes and aggregation run unchanged on (B).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

OUT_PREFIX = "c5b_"
N_BLOCKS = R.FEATURE_DIM // R.N_SC_BINS       # 10 feature blocks x 32 bins
SOURCE_MONTH = {"rot20260724": "2026-07", "live": "2026-09"}


def load_b_sources(vocab: dict) -> dict[str, dict]:
    """source -> window dict (r_common layout + nsub_min, month) with geom ids
    remapped onto the shared vocabulary (geom_table rebuilt from vocab)."""
    out = {}
    for name in C.SOURCES:
        p = C.b_windows_path(name)
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        w = {k: z[k] for k in z.files}
        if not w["dev"].size:
            continue
        gt = [tuple(json.loads(s)) for s in w["geom_table"]]
        for k in gt:
            vocab.setdefault(k, len(vocab))
        w["geom"] = np.array([vocab[gt[g]] for g in w["geom"].astype(int)])
        out[name] = w
    for w in out.values():
        w["geom_table"] = np.asarray([json.dumps(list(k)) for k, _ in
                                      sorted(vocab.items(), key=lambda kv: kv[1])])
    return out


def pool_sources(ws: dict[str, dict]) -> dict:
    """Concatenate sources; session ids offset so blocks stay distinct."""
    names = list(ws)
    base = dict(ws[names[0]])
    base["source"] = np.array([names[0]] * base["dev"].size)
    for n in names[1:]:
        extra = ws[n]
        off = base["session"].max() + 1
        for k in ("dev", "day", "month", "ts0", "ts1", "geom", "geom_purity",
                  "nsub_min", "nnodes_mean", "feat", "feat_rq"):
            base[k] = np.concatenate([base[k], extra[k]])
        base["session"] = np.concatenate([base["session"], extra["session"] + off])
        base["source"] = np.concatenate([base["source"], np.array([n] * extra["dev"].size)])
    return base


def common_mask(w: dict, keep: np.ndarray) -> np.ndarray:
    """Bins fully covered in every kept window (AND over windows)."""
    gt = [tuple(json.loads(s)) for s in w["geom_table"]]
    m = np.ones(R.N_SC_BINS, bool)
    for g, ns in set(zip(w["geom"][keep].tolist(), w["nsub_min"][keep].tolist())):
        n_full = X.AD.expected_subcarriers(dict(zip(C.GEOM_FIELDS, gt[g])))
        if n_full is None:
            return np.zeros(R.N_SC_BINS, bool)
        m &= C.covered_bins(n_full, ns)
    return m


def apply_mask(w: dict, mask: np.ndarray) -> dict:
    cols = np.tile(mask, N_BLOCKS)
    w2 = dict(w)
    w2["feat"] = w["feat"][:, cols]
    w2["feat_rq"] = w["feat_rq"][:, cols]
    return w2


def format_counts(w: dict, names: list[str]) -> dict:
    """device -> month -> {format json: windows}."""
    gt = list(w["geom_table"])
    out: dict = {}
    for d, mo, g in zip(w["dev"].tolist(), w["month"].tolist(), w["geom"].tolist()):
        out.setdefault(names[d], {}).setdefault(mo, {})
        out[names[d]][mo][gt[g]] = out[names[d]][mo].get(gt[g], 0) + 1
    return out


def compact_devices(w: dict) -> dict:
    """Keep only the MACs that have windows; device indices are remapped so
    per-device medians over w["macs"] see no empty device."""
    present = sorted(set(w["dev"].tolist()))
    remap = {d: i for i, d in enumerate(present)}
    w2 = dict(w)
    w2["dev"] = np.array([remap[d] for d in w["dev"].tolist()])
    w2["macs"] = np.asarray([w["macs"][d] for d in present])
    return w2
