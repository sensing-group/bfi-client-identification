#!/usr/bin/env python3
"""C-series (current revision): the ESP32-C5 mesh as a second BFI
receiver. Shared streaming reader, coverage aggregation and per-device frame
cache for dataset (B), the mesh capture
``esp32-csi-tomography/csi_data/bfi_captures.jsonl`` (+ its rotations).

Dataset (B) record schema (one JSON object per line, written by
backend/src/receiver.py::_handle_binary_bfi):
  recv_ts     float   backend wall-clock receive time (epoch s)
  node_mac    str     receiving ESP32-C5 node (full MAC, upper case)
  node_ip     str
  src_mac     str     beamformee = the reporting device (full MAC, upper case)
  dst_mac     str     beamformer = the AP
  node_ts_us  int     node-local microsecond clock
  rssi        int     dBm at the receiving node
  channel     int
  bb_format   int     chip rx_ctrl.cur_bb_format (firmware enum, not 802.11)
  subtype     int     13 Action / 14 Action No-Ack
  category    int     21 VHT / 30 HE
  action      int     0 = Compressed Beamforming
  body_len    int     bytes of action body captured, CAPPED at BFI_MAX_BODY=512
                      by the firmware (main.c:1969); no FCS included
  length_ok   bool    transport sanity flag (header body_len == bytes present)
  body_hex    str     the action body, drop-in for the Givens decoder

The same report is overheard by several nodes and written once per node.
A unique report is the first reception; later receptions of a byte-identical
body from the same src_mac within DEDUP_WINDOW_S are duplicates (their count
is kept as n_nodes). Gaps between byte-identical consecutive bodies are
histogrammed so the window choice can be checked.

Memory: the file is streamed line by line (never json.load'ed whole), all
aggregates are dicts of counters, per-device frames are decoded in batches
of FLUSH_FRAMES and written to chunked .npz files, nothing accumulates
across the pass except the aggregates. set_vm_limit() caps the address
space. --dry-run counts lines and reads the time span from the line prefix
without decoding any JSON.

Features are those of r_common (first-column Givens angles phi11, phi21,
psi21, psi31; 32 equal-width tone bins; per-window circular mean/spread;
320 dims), with one difference forced by the 512-byte cap: tones are binned
on the FULL standard tone count of the format (234 VHT 80/Ng1, 250 HE
80/Ng4), exactly as in (A), and a bin is kept only if every tone it covers
is present in the truncated body; partially covered bins are NaN. So a kept
bin has the same definition in (A) and (B).
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xres_common as X  # noqa: E402
import r_common as R  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (env-overridable)
# ---------------------------------------------------------------------------
TOMO_ROOT = Path(os.environ.get(
    "ESP32_TOMO_ROOT", Path.home() / "esp32-csi-tomography"))
B_LIVE = Path(os.environ.get(
    "BFI_C5_LIVE", TOMO_ROOT / "csi_data" / "bfi_captures.jsonl"))
B_ROTATED_DIR = Path(os.environ.get(
    "BFI_C5_ROTATED_DIR", TOMO_ROOT / "csi_data" / "rotated"))
B_COVERAGE_SUMMARY = Path(os.environ.get(
    "BFI_C5_DECODE_SUMMARY", TOMO_ROOT / "csi_data" / "bfi_decoded"
    / "coverage" / "full_corpus_20260724.json"))
CACHE_DIR = X.CACHE_DIR / "c5"
FIG_DIR = X.FIG_DIR
OUT_PREFIX = "c5_"

# Sources of (B): name -> path. The live file holds 2026-09-06..08 only; the
# single rotation holds 2026-07-15..24. No August rotation exists.
SOURCES = {
    "live": B_LIVE,
    "rot20260724": Path(os.environ.get(
        "BFI_C5_ROTATED",
        B_ROTATED_DIR / "bfi_captures.jsonl.20260724T105426Z.gz")),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C5_CACHE_VERSION = 1
BFI_MAX_BODY = 512                 # firmware cap (esp-idf main.c BFI_MAX_BODY)
DEDUP_WINDOW_S = 0.25              # identical body, same src, within -> same report
DUP_GAP_EDGES_S = (0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf"))
FLUSH_FRAMES = 8192                # per-device frames decoded per batch
PROGRESS_EVERY_LINES = 1_000_000
HOUR_S = 3600.0
TS_PREFIX = '{"recv_ts": '         # dry-run reads the stamp from this prefix
N_SC_BINS = R.N_SC_BINS
WINDOW_FRAMES = R.WINDOW_FRAMES
SESSION_GAP_S = R.SESSION_GAP_S
GEOM_FIELDS = R.GEOM_FIELDS        # (fmt, nc, nr, bphi, bpsi, bw, ng, fb, cb)
BFI_ACTION_COMPRESSED = 0          # only action 0 is a beamforming report
NON_BFI_ACTION_NAMES = {
    (21, 1): "VHT Group ID Management", (21, 2): "VHT Operating Mode",
}

LOG = sys.stderr


def log(msg: str) -> None:
    print(msg, file=LOG, flush=True)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def open_lines(p: Path):
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def dry_run(p: Path) -> dict:
    """Line count, byte count and time span from the line prefix. No JSON."""
    n, nbad, tmin, tmax = 0, 0, math.inf, -math.inf
    t0 = time.time()
    with open_lines(p) as f:
        for line in f:
            n += 1
            if not line.startswith(TS_PREFIX):
                nbad += 1
                continue
            try:
                t = float(line[len(TS_PREFIX):line.index(",", len(TS_PREFIX))])
            except ValueError:
                nbad += 1
                continue
            if t < tmin:
                tmin = t
            if t > tmax:
                tmax = t
    return dict(path=str(p), bytes=p.stat().st_size, lines=n,
                lines_without_ts_prefix=nbad,
                ts_min=tmin, ts_max=tmax, ts_min_iso=iso(tmin),
                ts_max_iso=iso(tmax), span_h=(tmax - tmin) / HOUR_S,
                seconds=time.time() - t0, peak_rss_mb=X.peak_rss_mb())


def iso(t: float) -> str:
    if not math.isfinite(t):
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def month_of(t: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(t))


def day_of(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def geom_key(g: dict) -> tuple:
    return tuple(g[k] for k in GEOM_FIELDS)


def full_body_len(g: dict):
    """Bytes a complete angle field needs (header + per-stream SNR + all
    standard tones). None when the (bw, Ng) tone count is not verified.
    MU reports carry an extra MU-exclusive delta-SNR part after the angles,
    not counted here: 'complete' means the ANGLE field is complete."""
    n_exp = X.AD.expected_subcarriers(g)
    if n_exp is None:
        return None
    _o, _w, _k, bps = X.AD._layout(g["nr"], g["nc"], g["bphi"], g["bpsi"])
    return 2 + g["mimo_len"] + g["nc"] + math.ceil(n_exp * bps / 8)


def tones_in_body(g: dict, body_len: int):
    """Tones whose angles are present in a body of body_len bytes."""
    _o, _w, _k, bps = X.AD._layout(g["nr"], g["nc"], g["bphi"], g["bpsi"])
    if bps == 0:
        return 0
    n = (body_len - 2 - g["mimo_len"] - g["nc"]) * 8 // bps
    n_exp = X.AD.expected_subcarriers(g)
    return n if n_exp is None else min(n, n_exp)


def covered_bins(n_full: int, n_sub: int, n_bins: int = N_SC_BINS):
    """Boolean mask of bins whose every tone (on the n_full grid) is < n_sub."""
    edges = X.bin_edges(n_full, n_bins)
    last_tone = np.array([np.flatnonzero(edges == j).max() if (edges == j).any()
                          else -1 for j in range(n_bins)])
    return last_tone < n_sub


def bin_frames_full_grid(rad: np.ndarray, n_full: int) -> np.ndarray:
    """(n, 4, n_sub) radians -> (n, 6, N_SC_BINS); bins on the n_full tone
    grid; bins not fully covered by the n_sub present tones are NaN."""
    n, _, n_sub = rad.shape
    edges = X.bin_edges(n_full, N_SC_BINS)[:n_sub]
    M = np.zeros((n_sub, N_SC_BINS))
    M[np.arange(n_sub), edges] = 1.0
    cnt = M.sum(axis=0)
    M = M / np.where(cnt > 0, cnt, 1.0)
    chans = [np.cos(rad[:, 0]), np.sin(rad[:, 0]),
             np.cos(rad[:, 1]), np.sin(rad[:, 1]), rad[:, 2], rad[:, 3]]
    out = np.stack([c @ M for c in chans], axis=1).astype(np.float32)
    out[:, :, ~covered_bins(n_full, n_sub)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Pass 1: coverage aggregates + labelled-device frame cache
# ---------------------------------------------------------------------------
class SourcePass:
    def __init__(self, name: str, path: Path, labelled: set[str]):
        self.name, self.path, self.labelled = name, path, labelled
        self.out_dir = CACHE_DIR / f"{name}.v{C5_CACHE_VERSION}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.geoms: dict[tuple, int] = {}
        self.geom_dicts: dict[tuple, dict] = {}
        self.geom_cache: dict[str, tuple | None] = {}
        # format table: key = (geom_id or None, bb_format, subtype, category,
        # action, body_len) -> counters
        self.fmt_rx = Counter()
        self.fmt_uniq = Counter()
        self.fmt_src: dict[tuple, set] = defaultdict(set)
        self.fmt_nodes: dict[tuple, set] = defaultdict(set)
        self.fmt_len_ok = Counter()
        self.node_rx = Counter()
        self.node_rssi = defaultdict(float)
        self.node_hour = Counter()
        self.src_rx = Counter()
        self.src_uniq = Counter()
        self.src_hour_rx = Counter()
        self.src_hour_uniq = Counter()
        self.src_node = Counter()
        self.src_first: dict[str, float] = {}
        self.src_last: dict[str, float] = {}
        self.dup_gap_hist = Counter()      # identical-body consecutive gaps
        self.nnodes_hist = Counter()       # nodes hearing one unique report
        self.last: dict[str, list] = {}    # src -> [body_hex, ts, nnodes-ref]
        self.pending: dict[str, list] = defaultdict(list)
        self.chunk_idx = Counter()
        self.undecodable = Counter()       # (src, geom_id, body_len) -> n
        self.n_lines = 0
        self.n_json_err = 0
        self.ts_min, self.ts_max = math.inf, -math.inf

    # -- geometry ------------------------------------------------------------
    def geom_of(self, bh: str):
        pre = bh[:16]
        if pre in self.geom_cache:
            return self.geom_cache[pre]
        g = X.AD.parse_geometry(bytes.fromhex(pre)) if len(pre) >= 14 else None
        key = None
        if g is not None:
            key = geom_key(g)
            if key not in self.geoms:
                self.geoms[key] = len(self.geoms)
                self.geom_dicts[key] = g
            key = self.geoms[key]
        self.geom_cache[pre] = key
        return key

    # -- one record ----------------------------------------------------------
    def observe(self, rec: dict) -> None:
        ts = float(rec["recv_ts"])
        src = rec["src_mac"].lower()
        node = rec["node_mac"].lower()
        bh = rec.get("body_hex") or ""
        blen = int(rec.get("body_len", len(bh) // 2))
        hour = int(ts // HOUR_S)
        self.ts_min = min(self.ts_min, ts)
        self.ts_max = max(self.ts_max, ts)
        gid = (self.geom_of(bh) if rec.get("action") == BFI_ACTION_COMPRESSED
               else None)
        fkey = (gid, rec.get("bb_format"), rec.get("subtype"),
                rec.get("category"), rec.get("action"), blen)
        self.fmt_rx[fkey] += 1
        self.fmt_src[fkey].add(src)
        self.fmt_nodes[fkey].add(node)
        if rec.get("length_ok", True):
            self.fmt_len_ok[fkey] += 1
        self.node_rx[node] += 1
        self.node_rssi[node] += float(rec.get("rssi", 0.0))
        self.node_hour[(node, hour)] += 1
        self.src_rx[src] += 1
        self.src_hour_rx[(src, hour)] += 1
        self.src_node[(src, node)] += 1
        self.src_first.setdefault(src, ts)
        self.src_last[src] = ts
        # dedup
        last = self.last.get(src)
        if last is not None and last[0] == bh:
            gap = ts - last[1]
            for e in DUP_GAP_EDGES_S:
                if gap <= e:
                    self.dup_gap_hist[e] += 1
                    break
            if gap <= DEDUP_WINDOW_S:
                last[2][0] += 1
                last[1] = ts
                return
        ref = [1]
        if last is not None:
            self.nnodes_hist[last[2][0]] += 1
        self.last[src] = [bh, ts, ref]
        self.fmt_uniq[fkey] += 1
        self.src_uniq[src] += 1
        self.src_hour_uniq[(src, hour)] += 1
        if src in self.labelled and gid is not None:
            self.pending[src].append([ts, bh, ref, float(rec.get("rssi", 0.0)),
                                      gid])
            if len(self.pending[src]) > FLUSH_FRAMES:
                self.flush(src, keep_last=True)

    # -- frame cache ---------------------------------------------------------
    def flush(self, src: str, keep_last: bool) -> None:
        items = self.pending[src]
        if keep_last:
            items, self.pending[src] = items[:-1], items[-1:]
        else:
            self.pending[src] = []
        if not items:
            return
        n = len(items)
        ts = np.array([it[0] for it in items])
        nn = np.array([it[2][0] for it in items], np.int16)
        rssi = np.array([it[3] for it in items], np.float32)
        gid = np.array([it[4] for it in items], np.int16)
        nsub = np.zeros(n, np.int16)
        feat = np.full((n, 6, N_SC_BINS), np.nan, np.float32)
        feat_rq = np.full((n, 6, N_SC_BINS), np.nan, np.float32)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, it in enumerate(items):
            groups[(int(it[4]), len(it[1]) // 2)].append(i)
        inv = {v: k for k, v in self.geoms.items()}
        for (g_id, L), idx in groups.items():
            g = self.geom_dicts[inv[g_id]]
            n_full = X.AD.expected_subcarriers(g)
            bodies = [bytes.fromhex(items[i][1]) for i in idx]
            res = R.decode_first_column(bodies, g) if n_full else None
            if res is None:
                self.undecodable[(src, g_id, L)] += len(idx)
                continue
            ia = np.asarray(idx)
            nsub[ia] = res[0].shape[2]
            feat[ia] = bin_frames_full_grid(res[0], n_full)
            feat_rq[ia] = bin_frames_full_grid(res[2], n_full)
        ok = nsub > 0
        k = self.chunk_idx[src]
        self.chunk_idx[src] += 1
        np.savez(self.out_dir / f"frames_{src.replace(':', '')}_{k:04d}.npz",
                 ts=ts[ok], nnodes=nn[ok], rssi=rssi[ok], gid=gid[ok],
                 nsub=nsub[ok], feat=feat[ok], feat_rq=feat_rq[ok])
        del items, feat, feat_rq, groups
        gc.collect()

    # -- the pass ------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        with open_lines(self.path) as f:
            for line in f:
                self.n_lines += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.n_json_err += 1
                    continue
                self.observe(rec)
                if self.n_lines % PROGRESS_EVERY_LINES == 0:
                    log(f"[{self.name}] {self.n_lines / 1e6:.0f}M lines, "
                        f"{time.time() - t0:.0f}s, peak RSS "
                        f"{X.peak_rss_mb():.0f} MB")
        for src in list(self.pending):
            self.flush(src, keep_last=False)
        for src, last in self.last.items():
            self.nnodes_hist[last[2][0]] += 1
        agg = self.aggregate(time.time() - t0)
        (self.out_dir / "coverage.json").write_text(
            json.dumps(agg, indent=1, default=float))
        return agg

    def aggregate(self, seconds: float) -> dict:
        inv = {v: k for k, v in self.geoms.items()}
        fmt_rows = []
        for fkey in sorted(self.fmt_rx, key=lambda k: -self.fmt_rx[k]):
            gid, bbf, st, cat, act, blen = fkey
            row = dict(geom_id=gid, bb_format=bbf, subtype=st, category=cat,
                       action=act, body_len=blen, receptions=self.fmt_rx[fkey],
                       unique_reports=self.fmt_uniq[fkey],
                       n_src_macs=len(self.fmt_src[fkey]),
                       n_nodes=len(self.fmt_nodes[fkey]),
                       length_ok=self.fmt_len_ok[fkey],
                       labelled_srcs=sorted(s for s in self.fmt_src[fkey]
                                            if s in self.labelled))
            if gid is not None:
                g = self.geom_dicts[inv[gid]]
                row.update({k: g[k] for k in GEOM_FIELDS})
                fl = full_body_len(g)
                row["full_body_len"] = fl
                row["tones_present"] = tones_in_body(g, blen)
                row["tones_full"] = X.AD.expected_subcarriers(g)
                row["angle_field_complete"] = (None if fl is None
                                               else blen >= fl)
                row["at_firmware_cap"] = blen >= BFI_MAX_BODY
            else:
                row["non_bfi_name"] = NON_BFI_ACTION_NAMES.get((cat, act), "")
            fmt_rows.append(row)
        hours = sorted({h for _, h in self.node_hour})
        return dict(
            source=self.name, path=str(self.path), lines=self.n_lines,
            json_errors=self.n_json_err, seconds=seconds,
            peak_rss_mb=X.peak_rss_mb(),
            ts_min=self.ts_min, ts_max=self.ts_max,
            ts_min_iso=iso(self.ts_min), ts_max_iso=iso(self.ts_max),
            span_h=(self.ts_max - self.ts_min) / HOUR_S,
            n_hours_with_data=len(hours),
            dedup_window_s=DEDUP_WINDOW_S,
            dup_gap_hist={str(k): v for k, v in sorted(self.dup_gap_hist.items())},
            nnodes_hist={str(k): v for k, v in sorted(self.nnodes_hist.items())},
            geom_table=[dict(geom_id=i, **{k: g[k] for k in GEOM_FIELDS})
                        for key, i in sorted(self.geoms.items(),
                                             key=lambda kv: kv[1])
                        for g in [self.geom_dicts[key]]],
            formats=fmt_rows,
            nodes={n: dict(receptions=c, mean_rssi=self.node_rssi[n] / c,
                           hours_active=sum(1 for (nn, _h) in self.node_hour
                                            if nn == n),
                           receptions_per_active_hour=c / max(1, sum(
                               1 for (nn, _h) in self.node_hour if nn == n)))
                   for n, c in self.node_rx.items()},
            sources={s: dict(receptions=c, unique_reports=self.src_uniq[s],
                             first=iso(self.src_first[s]),
                             last=iso(self.src_last[s]),
                             hours_active=sum(1 for (ss, _h) in self.src_hour_uniq
                                              if ss == s),
                             unique_per_active_hour=self.src_uniq[s] / max(
                                 1, sum(1 for (ss, _h) in self.src_hour_uniq
                                        if ss == s)),
                             n_nodes_hearing=sum(1 for (ss, _n) in self.src_node
                                                 if ss == s),
                             labelled=s in self.labelled,
                             oui_labelled=X.device_key(s) in X.DEVICE_GT)
                     for s, c in self.src_rx.items()},
            src_node={f"{s}|{n}": c for (s, n), c in self.src_node.items()
                      if s in self.labelled},
            undecodable_first_column={f"{s}|geom{g}|len{L}": n for (s, g, L), n
                                      in self.undecodable.items()},
            frame_chunks={s: int(self.chunk_idx[s]) for s in self.labelled},
        )


def labelled_macs() -> list[str]:
    """The six labelled full MACs of (A), from the r cache (lower case)."""
    z = np.load(R.windows_path(SESSION_GAP_S), allow_pickle=False)
    return [str(m) for m in z["macs"]]


def coverage_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.v{C5_CACHE_VERSION}" / "coverage.json"


def load_coverage(name: str) -> dict:
    return json.loads(coverage_path(name).read_text())


def frame_chunks(name: str, mac: str) -> list[Path]:
    d = CACHE_DIR / f"{name}.v{C5_CACHE_VERSION}"
    return sorted(d.glob(f"frames_{mac.replace(':', '')}_*.npz"))


# ---------------------------------------------------------------------------
# Windows for (B): sessions by gap, blocks by (session, UTC day)
# ---------------------------------------------------------------------------
def build_b_windows(name: str, devs: list[str], gap_s: float = SESSION_GAP_S
                    ) -> dict:
    """Window table in the r_common layout plus nsub_min, nnodes_mean,
    month, source. Geometry ids are the source's own (see coverage.json)."""
    cols = defaultdict(list)
    for di, m in enumerate(devs):
        chunks = frame_chunks(name, m)
        if not chunks:
            continue
        ts_all = np.concatenate([np.load(c)["ts"] for c in chunks])
        order = np.argsort(ts_all, kind="stable")
        sid_sorted = R.cut_sessions(ts_all[order], gap_s)
        sid = np.empty_like(sid_sorted)
        sid[order] = sid_sorted
        del ts_all, order, sid_sorted
        # concatenate per-frame arrays (a few hundred MB at most per device)
        d = {k: np.concatenate([np.load(c)[k] for c in chunks])
             for k in ("ts", "nnodes", "gid", "nsub", "feat", "feat_rq")}
        o = np.argsort(d["ts"], kind="stable")
        d = {k: v[o] for k, v in d.items()}
        sid = sid[o]
        days = np.array([day_of(t) for t in d["ts"]])
        for s in np.unique(sid):
            for day in np.unique(days[sid == s]):
                ii = np.flatnonzero((sid == s) & (days == day))
                for w in range(ii.size // WINDOW_FRAMES):
                    jj = ii[w * WINDOW_FRAMES:(w + 1) * WINDOW_FRAMES]
                    g, cnt = np.unique(d["gid"][jj], return_counts=True)
                    cols["dev"].append(di)
                    cols["session"].append(int(s))
                    cols["day"].append(day)
                    cols["month"].append(day[:7])
                    cols["ts0"].append(float(d["ts"][jj[0]]))
                    cols["ts1"].append(float(d["ts"][jj[-1]]))
                    cols["geom"].append(int(g[np.argmax(cnt)]))
                    cols["geom_purity"].append(float(cnt.max()) / jj.size)
                    cols["nsub_min"].append(int(d["nsub"][jj].min()))
                    cols["nnodes_mean"].append(float(d["nnodes"][jj].mean()))
                    cols["feat"].append(R.window_feature(d["feat"][jj]))
                    cols["feat_rq"].append(R.window_feature(d["feat_rq"][jj]))
        del d, sid, days
        gc.collect()
        X.log_rss(f"[{name}] windows {R.short(m)}")
    w = {k: np.asarray(v) for k, v in cols.items()}
    cov = load_coverage(name)
    w["geom_table"] = np.asarray([json.dumps([g[k] for k in GEOM_FIELDS])
                                  for g in cov["geom_table"]])
    w["macs"] = np.asarray(devs)
    w["gap_s"] = np.asarray(gap_s)
    w["source"] = np.asarray(name)
    return w


def b_windows_path(name: str, gap_s: float = SESSION_GAP_S) -> Path:
    return CACHE_DIR / f"{name}.v{C5_CACHE_VERSION}" / \
        f"windows_gap{int(gap_s)}.npz"


def get_b_windows(name: str, devs: list[str], force: bool = False) -> dict:
    p = b_windows_path(name)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    w = build_b_windows(name, devs)
    np.savez(p, **w)
    return w


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Stream dataset (B) once: "
                                 "coverage aggregates + labelled frame cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="count lines and report the time span, no decoding")
    ap.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    X.set_vm_limit()
    names = list(SOURCES) if a.source == "all" else [a.source]
    if a.dry_run:
        rep = {n: dry_run(SOURCES[n]) for n in names if SOURCES[n].exists()}
        for n in names:
            if not SOURCES[n].exists():
                rep[n] = dict(path=str(SOURCES[n]), missing=True)
        print(json.dumps(rep, indent=2))
        return 0
    devs = labelled_macs()
    log(f"labelled MACs from the r cache: {devs}")
    t0 = time.time()
    for n in names:
        if coverage_path(n).exists() and not a.force:
            log(f"[{n}] cache present, skipping (use --force)")
            continue
        if not SOURCES[n].exists():
            log(f"[{n}] MISSING: {SOURCES[n]}")
            continue
        agg = SourcePass(n, SOURCES[n], set(devs)).run()
        log(f"[{n}] {agg['lines']} lines in {agg['seconds']:.0f}s, "
            f"{agg['ts_min_iso']}..{agg['ts_max_iso']}, peak RSS "
            f"{agg['peak_rss_mb']:.0f} MB")
        gc.collect()
    for n in names:
        if coverage_path(n).exists():
            get_b_windows(n, devs, force=a.force)
    log(f"[done] {time.time() - t0:.0f}s, peak RSS {X.peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
