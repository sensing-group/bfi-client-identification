#!/usr/bin/env python3
"""What padding every report to one length costs the network, on the
residential capture.

A protected frame hides the report's format only if every report in the
network has one length, that of the longest (Section 7.2). This script counts
what that padding adds for the six labeled devices: the bytes of the report
bodies as captured and padded to the longest body, per device and in total,
and the share of the airtime the feedback then takes.

The capture holds no transmit rate per frame, so the airtime is a bound: each
report is counted at one PHY rate of RATES_MBPS, a legacy preamble and the MAC
header, and the lowest rate, 6 Mb/s, bounds the share from above. The share is
taken over the minutes in which at least one labeled device reported.

Usage: python3 scripts/r_padding_cost.py [--dry-run]
Output (figures/): r_padding_cost.json
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

import r_common as R
import xres_common as X

RATES_MBPS = (6, 24)                # lowest OFDM rate (the bound) and the common management rate
PREAMBLE_S = 20e-6                  # legacy preamble and SIGNAL field
MAC_OVERHEAD_BYTES = 28             # management header (24) and frame check sequence (4)
SECONDS_PER_MINUTE = 60
BITS_PER_BYTE = 8
BYTES_PER_MB = 1e6
OUT_JSON = R.FIG_DIR / "r_padding_cost.json"


def airtime_s(body_len: int, rate_mbps: float) -> float:
    """Duration of one report frame of this body length at one PHY rate."""
    bits = (body_len + MAC_OVERHEAD_BYTES) * BITS_PER_BYTE
    return PREAMBLE_S + bits / (rate_mbps * 1e6)


def collect(devs: list[str]):
    """Body-length counts per device, the active minutes and the days with data."""
    lengths: dict[str, Counter] = defaultdict(Counter)
    minutes: set[int] = set()
    days = 0
    marks = ",".join("?" * len(devs))
    for p in X.day_paths():
        c = X.connect_ro(p)
        rows = c.execute(
            f"SELECT {X.SRC_COL}, body_len, CAST(ts / {SECONDS_PER_MINUTE} AS INT), COUNT(*) "
            f"FROM bfi WHERE {X.SRC_COL} IN ({marks}) GROUP BY 1, 2, 3", devs).fetchall()
        c.close()
        days += bool(rows)
        for src, body_len, minute, n in rows:
            lengths[R.short(src)][int(body_len)] += n
            minutes.add(minute)
    return lengths, minutes, days


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="list the day files and stop")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    if a.dry_run:
        print("\n".join(str(p) for p in X.day_paths()))
        return 0
    lengths, minutes, days = collect(R.device_macs())
    longest = max(max(c) for c in lengths.values())
    shortest = min(min(c) for c in lengths.values())
    per_device = {}
    for dev, c in lengths.items():
        n = sum(c.values())
        body = sum(length * k for length, k in c.items())
        per_device[dev] = dict(reports=n, body_bytes=body, padded_bytes=n * longest,
                               factor=n * longest / body, mean_body_len=body / n)
    reports = sum(d["reports"] for d in per_device.values())
    body = sum(d["body_bytes"] for d in per_device.values())
    padded = reports * longest
    active_s = len(minutes) * SECONDS_PER_MINUTE
    airtime = {}
    for rate in RATES_MBPS:
        as_captured = sum(airtime_s(length, rate) * k for c in lengths.values() for length, k in c.items())
        as_padded = reports * airtime_s(longest, rate)
        airtime[str(rate)] = dict(captured_s=as_captured, padded_s=as_padded,
                                  captured_share=as_captured / active_s,
                                  padded_share=as_padded / active_s)
    factors = [d["factor"] for d in per_device.values()]
    X.write_json(OUT_JSON, dict(
        days_with_data=days, active_minutes=len(minutes),
        longest_body_len=longest, shortest_body_len=shortest,
        reports=reports, body_mb=body / BYTES_PER_MB, padded_mb=padded / BYTES_PER_MB,
        factor=padded / body, factor_min=min(factors), factor_max=max(factors),
        extra_mb_per_day=(padded - body) / BYTES_PER_MB / days,
        rates_mbps=list(RATES_MBPS), airtime=airtime, per_device=per_device,
        assumptions=dict(preamble_s=PREAMBLE_S, mac_overhead_bytes=MAC_OVERHEAD_BYTES,
                         note="no transmit rate is captured; the lowest rate bounds the share from above"),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[out] {OUT_JSON}", file=sys.stderr)
    print(f"padding to {longest} bytes: x{padded / body:.2f} "
          f"({min(factors):.2f} to {max(factors):.2f} per device), share at {RATES_MBPS[0]} Mb/s "
          f"{airtime[str(RATES_MBPS[0])]['captured_share']:.4f} -> "
          f"{airtime[str(RATES_MBPS[0])]['padded_share']:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
