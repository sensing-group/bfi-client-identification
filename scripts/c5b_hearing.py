#!/usr/bin/env python3
"""C5b-T3: where a listener must be. Per labelled device and per ESP32-C5
board of the mesh capture (B): fraction of the device's beamforming reports
heard, median RSSI, and the number of boards needed to hear 95% of a
device's reports (greedy set cover).

Pass (``--stream``): each (B) source file is streamed once, line by line,
with the same report de-duplication as c5_common (a byte-identical body from
the same src_mac within DEDUP_WINDOW_S is one report heard by several
boards). Only action-0 beamforming reports of the six labelled MACs are
counted. Per source the pass keeps only counters: receptions and an RSSI
histogram per (device, board), and a histogram of the board SET that heard
each unique report (bitmask over boards). Peak RSS is logged.

Analysis (default, from the cached counters): per-board hearing table,
greedy set cover per device, and the board layout the tomography repo
carries (relative MDS positions and k-means zones in
scripts/watcher/zones_cache.json; the three PI-labelled placements in
scripts/watcher/zones.py). No room-to-board map exists in that repo (all
boards are in the living room; the bedroom Apple TV is absent from B), so
the table is per board, with position and zone where known.

Outputs (figures/): c5b_hearing.csv (device x board), c5b_hearing_setcover.csv,
c5b_hearing_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import vendor_paths as VP  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_PREFIX = "c5b_"
CACHE_DIR = X.CACHE_DIR / "c5b"
RAW_CACHE = CACHE_DIR / "hearing_raw.v1.json"
# Board layout: the copy in release/vendor/ first, the tomography checkout
# (ESP32_TOMO_ROOT) behind it.
ZONES_CACHE = Path(os.environ.get(
    "BFI_C5_ZONES_CACHE", VP.vendored_or(
        VP.VENDOR_ZONES_CACHE,
        C.TOMO_ROOT / "scripts" / "watcher" / "zones_cache.json")))
COVER_TARGETS = (0.50, 0.90, 0.95, 0.99)
RSSI_MIN, RSSI_MAX = -128, 0               # histogram support (dBm)
MIN_REPORTS_FOR_TABLE = 100                # devices with fewer are listed only
# PI-labelled placements (esp32-csi-tomography scripts/watcher/zones.py:122,
# docs/4-research/in-progress/LEARNED_GEOMETRY_MDS_2026_07_31.md:158).
PI_PLACEMENTS = {
    "dev_5e5d39": "entrance shelf (PI label, 2026-07-24)",
    "dev_1dc8f3": "right of the TV, near the entrance (PI label)",
    "dev_c27892": "kitchen back, near the fridge (PI label)",
}
OUT = dict(table=X.FIG_DIR / f"{OUT_PREFIX}hearing.csv",
           cover=X.FIG_DIR / f"{OUT_PREFIX}hearing_setcover.csv",
           summary=X.FIG_DIR / f"{OUT_PREFIX}hearing_summary.json")


# ---------------------------------------------------------------------------
# Streaming pass
# ---------------------------------------------------------------------------
class HearingPass:
    def __init__(self, name: str, path: Path, labelled: set[str]):
        self.name, self.path, self.labelled = name, path, labelled
        self.nodes: dict[str, int] = {}                 # board -> bit index
        self.rx = Counter()                             # (src, node) -> n
        self.rssi = defaultdict(Counter)                # (src, node) -> {dBm: n}
        self.sets = defaultdict(Counter)                # src -> {bitmask: n}
        self.n_unique = Counter()                       # src -> unique reports
        self.last: dict[str, list] = {}                 # src -> [body, ts, mask]
        self.n_lines = 0
        self.n_json_err = 0
        self.n_lines_labelled = 0

    def node_bit(self, node: str) -> int:
        if node not in self.nodes:
            self.nodes[node] = len(self.nodes)
        return 1 << self.nodes[node]

    def close_report(self, src: str) -> None:
        last = self.last.get(src)
        if last is not None:
            self.sets[src][last[2]] += 1
            self.n_unique[src] += 1

    def observe(self, rec: dict) -> None:
        src = rec["src_mac"].lower()
        if src not in self.labelled or rec.get("action") != C.BFI_ACTION_COMPRESSED:
            return
        self.n_lines_labelled += 1
        node = rec["node_mac"].lower()
        ts = float(rec["recv_ts"])
        bh = rec.get("body_hex") or ""
        bit = self.node_bit(node)
        self.rx[(src, node)] += 1
        r = int(rec.get("rssi", 0))
        self.rssi[(src, node)][max(RSSI_MIN, min(RSSI_MAX, r))] += 1
        last = self.last.get(src)
        if last is not None and last[0] == bh and ts - last[1] <= C.DEDUP_WINDOW_S:
            last[2] |= bit
            last[1] = ts
            return
        self.close_report(src)
        self.last[src] = [bh, ts, bit]

    def run(self) -> dict:
        t0 = time.time()
        with C.open_lines(self.path) as f:
            for line in f:
                self.n_lines += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.n_json_err += 1
                    continue
                self.observe(rec)
                if self.n_lines % C.PROGRESS_EVERY_LINES == 0:
                    C.log(f"[{self.name}] {self.n_lines / 1e6:.0f}M lines, "
                          f"{time.time() - t0:.0f}s, peak RSS "
                          f"{X.peak_rss_mb():.0f} MB")
        for src in list(self.last):
            self.close_report(src)
        return dict(
            source=self.name, path=str(self.path), lines=self.n_lines,
            json_errors=self.n_json_err, labelled_receptions=self.n_lines_labelled,
            seconds=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
            nodes=self.nodes,
            rx={f"{s}|{n}": c for (s, n), c in self.rx.items()},
            rssi={f"{s}|{n}": dict(h) for (s, n), h in self.rssi.items()},
            sets={s: {str(k): v for k, v in h.items()} for s, h in self.sets.items()},
            n_unique=dict(self.n_unique))


def stream_all(force: bool) -> dict:
    if RAW_CACHE.exists() and not force:
        return json.loads(RAW_CACHE.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labelled = set(C.labelled_macs())
    raw = {}
    for name, path in C.SOURCES.items():
        if not path.exists():
            C.log(f"[{name}] MISSING {path}")
            continue
        raw[name] = HearingPass(name, path, labelled).run()
        C.log(f"[{name}] done {raw[name]['lines']} lines "
              f"{raw[name]['seconds']:.0f}s peak RSS {raw[name]['peak_rss_mb']:.0f} MB")
    RAW_CACHE.write_text(json.dumps(raw))
    return raw


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def hist_median(h: dict) -> float:
    vals = np.array(sorted(int(k) for k in h))
    return hist_quantile(h, 0.5)


def greedy_cover(sets: dict[int, int], n_nodes: int, targets) -> list[dict]:
    """Greedy weighted set cover over report node-sets (bitmask -> count).
    Returns the chosen boards in order with the cumulative fraction heard."""
    masks = np.array([int(k) for k in sets], dtype=np.int64)
    cnt = np.array([sets[k] for k in sets], dtype=np.float64)
    total = cnt.sum()
    covered = np.zeros(masks.size, bool)
    chosen, steps = [], []
    for _ in range(n_nodes):
        best, gain = None, 0.0
        for b in range(n_nodes):
            if b in chosen:
                continue
            g = cnt[~covered & ((masks >> b) & 1 == 1)].sum()
            if g > gain:
                best, gain = b, g
        if best is None:
            break
        chosen.append(best)
        covered |= ((masks >> best) & 1 == 1)
        steps.append(dict(step=len(chosen), board_bit=best,
                          frac_heard=float(cnt[covered].sum() / total)))
    out = dict(steps=steps)
    for t in targets:
        k = next((s["step"] for s in steps if s["frac_heard"] >= t), None)
        out[f"boards_for_{int(t * 100)}pct"] = k
    return out


def load_layout() -> dict:
    if not ZONES_CACHE.exists():
        return {}
    z = json.loads(ZONES_CACHE.read_text())
    pos = {k.lower(): v for k, v in z.get("node_positions", {}).items()}
    zone = {k.lower(): v for k, v in z.get("zone_of_node", {}).items()}
    return dict(pos=pos, zone=zone, generated_at=z.get("generated_at"),
                door_zone=z.get("door_zone"), n_zones=z.get("n_zones"),
                door_node=(z.get("door_node_mac") or "").lower())


def analyse(raw: dict) -> tuple[list[dict], list[dict], dict]:
    layout = load_layout()
    rows, cover_rows = [], []
    summary = dict(sources={}, devices={}, layout_source=str(ZONES_CACHE),
                   layout_generated_at=layout.get("generated_at"),
                   layout_note=(
                       "Relative MDS positions (metres, arbitrary origin and "
                       "rotation) and k-means zone ids from the tomography "
                       "repo; NO room labels exist for the boards. All boards "
                       "are deployed in the living room (PI ground-truth scope "
                       "rule); three boards carry a PI placement note."),
                   pi_placements=PI_PLACEMENTS)
    for name, src in raw.items():
        nodes = src["nodes"]
        inv = {v: k for k, v in nodes.items()}
        summary["sources"][name] = dict(
            lines=src["lines"], labelled_receptions=src["labelled_receptions"],
            n_boards=len(nodes), seconds=src["seconds"],
            peak_rss_mb=src["peak_rss_mb"])
        for mac, n_uni in src["n_unique"].items():
            dev = R.short(mac)
            sets = {int(k): v for k, v in src["sets"][mac].items()}
            # per board: fraction of unique reports whose set includes it
            heard = Counter()
            for m, c in sets.items():
                for b, node in inv.items():
                    if (m >> b) & 1:
                        heard[node] += c
            nn = np.array([bin(m).count("1") for m in sets])
            cc = np.array([sets[m] for m in sets], float)
            dsum = dict(unique_reports=n_uni,
                        boards_per_report_mean=float((nn * cc).sum() / cc.sum()),
                        boards_per_report_median=weighted_median(nn, cc),
                        heard_by_one_board_only=float(cc[nn == 1].sum() / cc.sum()))
            if n_uni >= MIN_REPORTS_FOR_TABLE:
                cov = greedy_cover(sets, len(nodes), COVER_TARGETS)
                dsum.update({k: v for k, v in cov.items() if k != "steps"})
                for s in cov["steps"]:
                    cover_rows.append(dict(source=name, device=dev, step=s["step"],
                                           board=inv[s["board_bit"]],
                                           frac_heard=s["frac_heard"]))
            summary["devices"].setdefault(dev, {})[name] = dsum
            for node in sorted(nodes, key=lambda n: -heard[n]):
                key = f"{mac}|{node}"
                h = src["rssi"].get(key, {})
                pos = layout.get("pos", {}).get(node)
                rows.append(dict(
                    source=name, device=dev, board=node,
                    receptions=src["rx"].get(key, 0),
                    reports_heard=heard[node],
                    frac_heard=heard[node] / n_uni if n_uni else 0.0,
                    rssi_median_dbm=hist_median(h) if h else None,
                    rssi_p10_dbm=hist_quantile(h, 0.10) if h else None,
                    rssi_p90_dbm=hist_quantile(h, 0.90) if h else None,
                    x_m=pos[0] if pos else None, y_m=pos[1] if pos else None,
                    zone=layout.get("zone", {}).get(node),
                    placement=PI_PLACEMENTS.get(node, "")))
    return rows, cover_rows, summary


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(len(o) - 1, np.searchsorted(c, c[-1] / 2.0))])


def hist_quantile(h: dict, q: float) -> float:
    hh = {int(k): v for k, v in h.items()}          # JSON round-trip -> str keys
    vals = np.array(sorted(hh))
    cnt = np.array([hh[v] for v in vals], float)
    c = np.cumsum(cnt)
    return float(vals[min(len(vals) - 1, np.searchsorted(c, q * c[-1]))])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    C.log(f"[out] {path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="file sizes and line counts only")
    ap.add_argument("--force", action="store_true", help="re-stream")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    if a.dry_run:
        rep = {n: dict(path=str(p), bytes=p.stat().st_size if p.exists() else None,
                       est_peak_rss_mb="counters only, < 400 MB (c5_coverage pass "
                       "measured 313-323 MB with decoding)")
               for n, p in C.SOURCES.items()}
        print(json.dumps(rep, indent=2))
        return 0
    raw = stream_all(a.force)
    rows, cover_rows, summary = analyse(raw)
    write_csv(OUT["table"], rows)
    write_csv(OUT["cover"], cover_rows)
    summary["runtime_s"] = time.time() - t0
    summary["peak_rss_mb"] = X.peak_rss_mb()
    X.write_json(OUT["summary"], summary)
    C.log(f"[done] {summary['runtime_s']:.0f}s peak RSS {summary['peak_rss_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
