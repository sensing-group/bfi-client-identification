#!/usr/bin/env python3
"""Run the multi-signal one-shot pipeline on real Pi-captured BFI.

Inputs the SQLite DB written by the Pi capture service and outputs:
  - per-MAC frame counts and feature distributions
  - one-shot K=1 closed-set accuracy across the MACs in the corpus,
    with the fusion encoder trained on power + body-proxy features
  - a check for which channels actually carry separation

The MACs in today's capture are not labelled subjects, so the result is
not a subject-ID number; it is a pipeline-on-real-BFI feasibility check
and a per-channel separability ablation. Real subject-ID needs the
labelling protocol in LABELLING_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DB = "/tmp/pi_today.sqlite"
WINDOW_FRAMES = 32
SUPPORT_PER_MAC = 1
QUERIES_PER_MAC = 50
N_TRIALS = 50
DEFAULT_BODY_PROXY_DIM = 64
SNR_BYTES_TO_DB = 0.25


def decode_avg_snr_dB(hex_str: str | None) -> list[float]:
    if not hex_str:
        return []
    raw = bytes.fromhex(hex_str)
    return [int.from_bytes([x], "big", signed=True) * SNR_BYTES_TO_DB for x in raw]


def body_proxy(body_hex: str, dim: int = DEFAULT_BODY_PROXY_DIM) -> np.ndarray:
    """Reduce variable-length body to a fixed-dim vector by histogramming
    bytes into `dim` bins. Cheap, no real Givens decoding, but enough to
    show whether body content is discriminative beyond power and capability.
    """
    raw = bytes.fromhex(body_hex)
    if not raw:
        return np.zeros(dim, dtype=np.float64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    edges = np.linspace(0, 256, dim + 1)
    h, _ = np.histogram(arr, bins=edges, density=False)
    return h.astype(np.float64) / max(1, len(arr))


def load_frames(db_path: Path) -> dict[str, list[dict]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = (
        "SELECT ts, src_mac, nc, nr, bw_mhz, codebook, feedback_type, "
        "       rssi, avg_snr_hex, body_hex, body_len, dialog_token "
        "FROM bfi WHERE nc IS NOT NULL ORDER BY ts"
    )
    by_mac: dict[str, list[dict]] = defaultdict(list)
    for row in conn.execute(sql):
        (ts, mac, nc, nr, bw, cb, fb, rssi, snr_hex, body_hex,
         body_len, tok) = row
        by_mac[mac].append(dict(
            ts=ts, nc=nc, nr=nr, bw=bw, cb=cb, fb=fb, rssi=rssi,
            snr_db=decode_avg_snr_dB(snr_hex), body_hex=body_hex,
            body_len=body_len, tok=tok,
        ))
    return by_mac


def frame_features(f: dict) -> dict[str, np.ndarray]:
    snr = np.asarray(f["snr_db"], dtype=np.float64)
    if snr.size < 2:
        snr = np.pad(snr, (0, 2 - snr.size))
    snr = snr[:2]
    power = np.concatenate([[float(f["rssi"] or 0.0)], snr])
    capability = np.asarray([
        float(f["nc"] or 0), float(f["nr"] or 0),
        float(f["bw"] or 0), float(f["cb"] or 0),
        1.0 if f["fb"] == "SU" else 0.0,
    ])
    body = body_proxy(f["body_hex"])
    return dict(power=power, capability=capability, body=body)


def aggregate_window(frames: list[dict], indices: list[int]) -> dict[str, np.ndarray]:
    feats = [frame_features(frames[i]) for i in indices]
    return {k: np.mean(np.stack([f[k] for f in feats]), axis=0)
            for k in feats[0].keys()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def one_shot_eval(by_mac: dict[str, list[dict]], channels: list[str],
                  rng: np.random.Generator) -> tuple[float, int]:
    """Closed-set K=1 with N_TRIALS trials. Each trial draws a 32-frame
    support window per MAC, then QUERIES_PER_MAC query windows per MAC.
    Each query is classified by max cosine similarity over channels
    (per-channel cosine, then summed across channels)."""
    macs = sorted(by_mac.keys())
    correct = 0
    total = 0
    for _ in range(N_TRIALS):
        prototypes = {}
        skip_trial = False
        for m in macs:
            n = len(by_mac[m])
            if n < WINDOW_FRAMES * 2:
                skip_trial = True
                break
            sup_start = int(rng.integers(0, n - WINDOW_FRAMES))
            sup = aggregate_window(
                by_mac[m], list(range(sup_start, sup_start + WINDOW_FRAMES))
            )
            prototypes[m] = sup
        if skip_trial:
            continue
        for m in macs:
            n = len(by_mac[m])
            for _q in range(QUERIES_PER_MAC):
                q_start = int(rng.integers(0, n - WINDOW_FRAMES))
                qry = aggregate_window(
                    by_mac[m],
                    list(range(q_start, q_start + WINDOW_FRAMES)),
                )
                scores = {
                    om: sum(
                        cosine(qry[c], prototypes[om][c])
                        for c in channels
                    )
                    for om in macs
                }
                pred = max(scores, key=scores.get)
                correct += int(pred == m)
                total += 1
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="/tmp/bfi_pi_one_shot.json")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"db not found: {db}", file=sys.stderr)
        return 1

    by_mac = load_frames(db)
    print(f"[load] {db}  total parsed frames: {sum(len(v) for v in by_mac.values())}")
    print(f"[load] MACs: {len(by_mac)}")
    for m, frames in sorted(by_mac.items(), key=lambda kv: -len(kv[1])):
        rssi = np.array([f["rssi"] for f in frames if f["rssi"] is not None])
        snr0 = np.array([f["snr_db"][0] for f in frames if len(f["snr_db"]) > 0])
        snr1 = np.array([f["snr_db"][1] for f in frames if len(f["snr_db"]) > 1])
        body = np.array([f["body_len"] for f in frames])
        nc = Counter(f["nc"] for f in frames)
        print(f"  {m}: n={len(frames):>5d}  rssi={rssi.mean():.1f}±{rssi.std():.1f}  "
              f"snr0={snr0.mean():.1f}  snr1={snr1.mean():.1f}  "
              f"body_len={body.mean():.0f}  nc={dict(nc)}")

    rng = np.random.default_rng(args.seed)
    channel_sets = [
        ("power-only",       ["power"]),
        ("body-only",        ["body"]),
        ("capability-only",  ["capability"]),
        ("power+body",       ["power", "body"]),
        ("fusion (P+B+C)",   ["power", "body", "capability"]),
    ]
    results = {}
    print()
    print(f"{'channel set':<22s} {'acc':>8s} {'n_decisions':>12s}")
    for name, chans in channel_sets:
        acc, n = one_shot_eval(by_mac, chans, rng)
        results[name] = {"channels": chans, "acc": acc, "n": n}
        print(f"{name:<22s} {acc:>8.4f} {n:>12d}")

    Path(args.out_json).write_text(json.dumps(dict(
        db=str(db), trials=N_TRIALS, queries_per_mac=QUERIES_PER_MAC,
        window_frames=WINDOW_FRAMES, results=results,
    ), indent=2))
    print(f"\n[done] -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
