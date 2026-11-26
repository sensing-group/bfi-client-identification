#!/usr/bin/env python3
"""Run the closed-set device-MAC K=1 attacker on the full 13-day Pi capture
window, not just the 8-day paper window.

Reads every bfi_<DATE>.sqlite under PI_ARCHIVE_GLOB, builds a single
by_mac dict, and invokes the one_shot_eval function from
bfi-privacy/tools/run_pi_one_shot.py. Outputs the per-channel and
fused accuracy as a JSON, plus a per-MAC frame budget summary.

Memory-safe: rows are streamed from each daily DB by sqlite cursor, no
intermediate copies. Peak RSS is dominated by the in-memory by_mac
dict (~400 MB on the current archive); we delete each per-day cursor
between days and call gc.collect() per day.

Outputs:
  <system temp dir>/bfi_pi_one_shot_full_window.json
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vendor_paths as VP  # noqa: E402

ARCHIVE_DIR = Path(os.environ.get(
    "BFI_ARCHIVE_DIR", Path.home() / "bfi-archive" / "data"))
PI_ARCHIVE_GLOB = str(ARCHIVE_DIR / "bfi_*.sqlite")
MIN_FRAMES_PER_MAC = 64       # one support + one query window minimum
N_TRIALS = 50
OUT_JSON = Path(tempfile.gettempdir()) / "bfi_pi_one_shot_full_window.json"
BFI_PRIVACY_DIR = Path(os.environ.get(
    "BFI_PRIVACY_DIR", Path.home() / "bfi-privacy"))

# Import the existing one_shot_eval / decode_avg_snr_dB helpers: the copy in
# release/vendor/ first, the bfi-privacy checkout (BFI_PRIVACY_DIR) behind it.
VP.add_bfi_privacy(BFI_PRIVACY_DIR)
from run_pi_one_shot import (  # noqa: E402
    decode_avg_snr_dB,
    one_shot_eval,
)


REQUIRED_COLS = {"ts", "src_mac", "nc", "nr", "bw_mhz", "codebook",
                 "feedback_type", "rssi", "avg_snr_hex", "body_hex",
                 "body_len", "dialog_token"}


def load_all_days(db_glob: str) -> dict[str, list[dict]]:
    import sqlite3
    by_mac: dict[str, list[dict]] = defaultdict(list)
    days_loaded = 0
    days_skipped = 0
    for db_path in sorted(glob.glob(db_glob)):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(bfi)").fetchall()}
        if not REQUIRED_COLS.issubset(cols):
            missing = REQUIRED_COLS - cols
            print(f"  {Path(db_path).name}: SKIP (missing {missing})",
                  file=sys.stderr)
            conn.close()
            days_skipped += 1
            continue
        days_loaded += 1
        sql = (
            "SELECT ts, src_mac, nc, nr, bw_mhz, codebook, feedback_type, "
            "       rssi, avg_snr_hex, body_hex, body_len, dialog_token "
            "FROM bfi WHERE nc IS NOT NULL ORDER BY ts"
        )
        n_added = 0
        for row in conn.execute(sql):
            (ts, mac, nc, nr, bw, cb, fb, rssi, snr_hex, body_hex,
             body_len, tok) = row
            by_mac[mac].append(dict(
                ts=ts, nc=nc, nr=nr, bw=bw, cb=cb, fb=fb, rssi=rssi,
                snr_db=decode_avg_snr_dB(snr_hex), body_hex=body_hex,
                body_len=body_len, tok=tok,
            ))
            n_added += 1
        conn.close()
        print(f"  {Path(db_path).name}: +{n_added:>7d} frames "
              f"({len(by_mac):>3d} cumulative MACs)", file=sys.stderr)
        gc.collect()
    print(f"Loaded {days_loaded} day-DBs (skipped {days_skipped} on schema), "
          f"{sum(len(v) for v in by_mac.values())} total frames, "
          f"{len(by_mac)} unique MACs", file=sys.stderr)
    return by_mac


def main() -> int:
    by_mac = load_all_days(PI_ARCHIVE_GLOB)

    eligible = {m: f for m, f in by_mac.items() if len(f) >= MIN_FRAMES_PER_MAC}
    print(f"\nEligible MACs (>= {MIN_FRAMES_PER_MAC} frames): {len(eligible)}",
          file=sys.stderr)
    for m in sorted(eligible, key=lambda x: -len(eligible[x])):
        print(f"  {m}: {len(eligible[m]):>7d} frames", file=sys.stderr)

    rng = np.random.default_rng(0)
    channels_to_try = [
        ["power"],
        ["capability"],
        ["body"],
        ["power", "body"],
        ["power", "capability", "body"],
    ]
    results = {}
    for chans in channels_to_try:
        acc, total = one_shot_eval(eligible, chans, rng)
        key = "+".join(chans)
        chance = 1.0 / len(eligible)
        results[key] = {
            "channels": chans,
            "accuracy": acc,
            "total_queries": total,
            "n_macs": len(eligible),
            "chance": chance,
            "lift_over_chance": acc / chance if chance > 0 else None,
        }
        print(f"\n  channels={key:30s}  acc={acc:.4f}  "
              f"chance={chance:.4f}  lift={acc/chance:.2f}x", file=sys.stderr)

    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
