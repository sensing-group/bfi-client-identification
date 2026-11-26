#!/usr/bin/env python3
"""C2: cross-receiver, cross-month identification between the paper's AX210
capture (A, 2026-05) and the ESP32-C5 mesh capture (B, per month).

Protocol (r_eval): enrolment = K windows drawn without replacement from ONE
(session, day) block of each gallery device (K in K_GRID, 0 = all windows
of the block); query = T consecutive windows of one block, features
averaged; decision = nearest gallery device (1-NN over the enrolled
windows, euclidean on z-scored features). Chance per decision = 1/|gallery|.

Conditions (each run per (B) month)
  A_to_B       enrol from A, query every window of B (gallery = devices with
               windows in both A and the B month)
  B_to_A       enrol from the B month, query every window of A
  B_within_B   enrol one block in the B month, query the device's windows on
               the OTHER days of that month
  A_within_A   same-receiver reference on A, restricted to the same gallery
               and the same bin mask (enrol one block, query other days)

Features (every row has all three)
  angles     320-d first-column Givens vector, restricted to the bins that
             the 512-byte firmware cap leaves fully covered for EVERY gallery
             device in the B month (mask reported), z-scored label-free on
             the pooled windows of the two sets being compared
  angles_rq  same after re-quantising every angle to 4-bit phi / 2-bit psi
  format     one-hot of the window's report format over a vocabulary shared
             by A and B (fmt, Nc, Nr, phi/psi bits, bandwidth, Ng, SU/MU,
             codebook); ties broken uniformly at random

Outputs (figures/): c5_cross_receiver.csv, c5_cross_receiver_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_common as C  # noqa: E402
import r_common as R  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_GRID = (1, 16, 0)                 # 0 = all windows of the enrolment block
T_GRID = (1, 16)
N_TRIALS = 20
MAX_QUERIES_PER_DEVICE = 1000       # per trial and T, random subsample
SEED = 0
TIE_TOL = 1e-9
FEATURES = ("angles", "angles_rq", "format")
CONDITIONS = ("A_to_B", "B_to_A", "B_within_B", "A_within_A")
N_BLOCKS = 10                       # 10 blocks x 32 bins = 320 dims
OUT = dict(csv=C.FIG_DIR / f"{C.OUT_PREFIX}cross_receiver.csv",
           summary=C.FIG_DIR / f"{C.OUT_PREFIX}cross_receiver_summary.json")
CSV_COLS = ["condition", "enrol", "query", "month", "feature", "K", "T",
            "device", "recall", "n_decisions", "n_gallery", "chance",
            "n_trials", "bins_kept", "dims"]


def klabel(k: int) -> str:
    return "all" if k == 0 else str(k)


# ---------------------------------------------------------------------------
# Window sets
# ---------------------------------------------------------------------------
class WinSet:
    """One dataset's windows with a shared geometry vocabulary."""

    def __init__(self, tag: str, w: dict, vocab: dict):
        self.tag = tag
        self.dev = w["dev"].astype(int)
        self.session = w["session"].astype(int)
        self.day = np.asarray(w["day"]).astype(str)
        self.ts0 = w["ts0"]
        self.feat = w["feat"].astype(np.float64)
        self.feat_rq = w["feat_rq"].astype(np.float64)
        gt = [tuple(json.loads(s)) for s in w["geom_table"]]
        for k in gt:
            vocab.setdefault(k, len(vocab))
        self.geom_key = [gt[g] for g in w["geom"].astype(int)]
        self.gid = np.array([vocab[k] for k in self.geom_key])
        if "nsub_min" in w:
            self.nsub = w["nsub_min"].astype(int)
        else:   # A: full tone count of each window's geometry
            self.nsub = np.array([X.AD.expected_subcarriers(
                dict(zip(C.GEOM_FIELDS, k))) or 0 for k in self.geom_key])
        self.month = np.asarray(w["month"]).astype(str) if "month" in w \
            else np.array([d[:7] for d in self.day])

    def subset(self, keep: np.ndarray) -> "WinSet":
        o = object.__new__(WinSet)
        o.tag = self.tag
        for k in ("dev", "session", "day", "ts0", "feat", "feat_rq", "gid",
                  "nsub", "month"):
            setattr(o, k, getattr(self, k)[keep])
        o.geom_key = [self.geom_key[i] for i in np.flatnonzero(keep)]
        return o

    def bin_mask(self) -> np.ndarray:
        """Bins fully covered in EVERY window (AND over windows)."""
        m = np.ones(C.N_SC_BINS, bool)
        for key, ns in set(zip(self.geom_key, self.nsub.tolist())):
            n_full = X.AD.expected_subcarriers(dict(zip(C.GEOM_FIELDS, key)))
            if n_full is None:
                return np.zeros(C.N_SC_BINS, bool)
            m &= C.covered_bins(n_full, ns)
        return m

    def blocks(self) -> dict:
        by = defaultdict(list)
        for i in np.argsort(self.ts0, kind="stable"):
            by[(int(self.dev[i]), int(self.session[i]), str(self.day[i]))
               ].append(i)
        return {k: np.asarray(v) for k, v in by.items()}

    def majority_geom(self, dev: int) -> tuple:
        ks = [self.geom_key[i] for i in np.flatnonzero(self.dev == dev)]
        vals, cnt = np.unique([json.dumps(k) for k in ks], return_counts=True)
        return tuple(json.loads(vals[np.argmax(cnt)]))


def zscore_pooled(*mats: np.ndarray) -> list[np.ndarray]:
    allm = np.concatenate(mats, axis=0)
    mu = np.nanmean(allm, axis=0)
    sd = np.nanstd(allm, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    return [np.nan_to_num((m - mu) / sd) for m in mats]


def feature_mats(E: WinSet, Q: WinSet, mask_bins: np.ndarray, n_vocab: int):
    """z-scored masked angle matrices, requantised copies and one-hot format
    for the enrolment set E and the query set Q."""
    cols = np.tile(mask_bins, N_BLOCKS)
    ze, zq = zscore_pooled(E.feat[:, cols], Q.feat[:, cols])
    re, rq = zscore_pooled(E.feat_rq[:, cols], Q.feat_rq[:, cols])
    fe, fq = np.eye(n_vocab)[E.gid], np.eye(n_vocab)[Q.gid]
    return dict(angles=(ze, zq), angles_rq=(re, rq), format=(fe, fq))


# ---------------------------------------------------------------------------
# 1-NN scoring
# ---------------------------------------------------------------------------
def min_dist(Qm: np.ndarray, Em: np.ndarray) -> np.ndarray:
    """Per query, the minimum euclidean distance to any enrolled row."""
    d2 = (Qm ** 2).sum(1)[:, None] + (Em ** 2).sum(1)[None] - 2 * Qm @ Em.T
    return np.sqrt(np.maximum(d2, 0.0)).min(axis=1)


def predict(dmat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """dmat (n_q, n_gallery) -> argmin with uniform random tie-break."""
    out = np.empty(dmat.shape[0], int)
    for i, row in enumerate(dmat):
        c = np.flatnonzero(row <= row.min() + TIE_TOL)
        out[i] = c[0] if c.size == 1 else rng.choice(c)
    return out


def query_rows(Q: WinSet, blocks: dict, dev: int, T: int, exclude_days: set,
               rng: np.random.Generator, Fm: dict) -> dict:
    """Averaged feature rows for T-window queries of dev in Q."""
    rows = {f: [] for f in Fm}
    for (d, _s, day), ii in blocks.items():
        if d != dev or day in exclude_days or ii.size < T:
            continue
        for w in range(ii.size // T):
            jj = ii[w * T:(w + 1) * T]
            for f in Fm:
                rows[f].append(Fm[f][jj].mean(0))
    n = len(rows["angles"])
    if n == 0:
        return {}
    keep = np.arange(n)
    if n > MAX_QUERIES_PER_DEVICE:
        keep = np.sort(rng.choice(n, MAX_QUERIES_PER_DEVICE, replace=False))
    return {f: np.asarray(v)[keep] for f, v in rows.items()}


def run_condition(cond: str, E: WinSet, Q: WinSet, gallery: list[int],
                  mask_bins: np.ndarray, n_vocab: int, rng) -> list[dict]:
    """Returns per-(feature, K, T) decision tallies per device."""
    F = feature_mats(E, Q, mask_bins, n_vocab)
    Fe = {f: F[f][0] for f in F}
    Fq = {f: F[f][1] for f in F}
    eb, qb = E.blocks(), Q.blocks()
    within = E is Q
    hits = defaultdict(lambda: [0, 0])      # (feat, K, T, dev) -> [hit, n]
    for _ in range(N_TRIALS):
        enrol = {}
        for g in gallery:
            cand = [k for k in eb if k[0] == g]
            enrol[g] = cand[rng.integers(len(cand))]
        perms = {g: rng.permutation(eb[enrol[g]]) for g in gallery}
        for m in gallery:
            excl = {enrol[m][2]} if within else set()
            for T in T_GRID:
                qr = query_rows(Q, qb, m, T, excl, rng, Fq)
                if not qr:
                    continue
                for K in K_GRID:
                    kk = {g: (perms[g].size if K == 0
                              else min(K, perms[g].size)) for g in gallery}
                    for f in FEATURES:
                        dmat = np.column_stack([
                            min_dist(qr[f], Fe[f][perms[g][:kk[g]]])
                            for g in gallery])
                        pred = np.asarray(gallery)[predict(dmat, rng)]
                        h = hits[(f, K, T, m)]
                        h[0] += int((pred == m).sum())
                        h[1] += int(pred.size)
    return hits


def tally_rows(cond, enrol, query, month, hits, gallery, names, bins_kept):
    rows = []
    for f in FEATURES:
        for K in K_GRID:
            for T in T_GRID:
                per = {m: hits[(f, K, T, m)] for m in gallery
                       if (f, K, T, m) in hits}
                if not per:
                    continue
                base = dict(condition=cond, enrol=enrol, query=query,
                            month=month, feature=f, K=klabel(K), T=T,
                            n_gallery=len(gallery), chance=1.0 / len(gallery),
                            n_trials=N_TRIALS, bins_kept=int(bins_kept),
                            dims=(int(bins_kept) * N_BLOCKS if f != "format"
                                  else -1))
                tot_h = sum(v[0] for v in per.values())
                tot_n = sum(v[1] for v in per.values())
                for m, (h, n) in per.items():
                    rows.append(dict(base, device=names[m], recall=h / n,
                                     n_decisions=n))
                rows.append(dict(base, device="ALL_micro",
                                 recall=tot_h / tot_n, n_decisions=tot_n))
                rows.append(dict(base, device="ALL_macro",
                                 recall=float(np.mean([v[0] / v[1] for v in
                                                       per.values()])),
                                 n_decisions=tot_n))
    return rows


# ---------------------------------------------------------------------------
def main() -> int:
    X.set_vm_limit()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    devs = C.labelled_macs()
    names = [R.short(m) for m in devs]
    vocab: dict = {}
    za = np.load(R.windows_path(C.SESSION_GAP_S), allow_pickle=False)
    A = WinSet("A", {k: za[k] for k in za.files}, vocab)
    B_all = []
    for name in C.SOURCES:
        if C.b_windows_path(name).exists():
            zb = np.load(C.b_windows_path(name), allow_pickle=False)
            if zb["dev"].size:
                B_all.append(WinSet(name, {k: zb[k] for k in zb.files}, vocab))
    n_vocab = len(vocab)
    rows, summary = [], dict(
        k_grid=[klabel(k) for k in K_GRID], t_grid=list(T_GRID),
        n_trials=N_TRIALS, max_queries_per_device=MAX_QUERIES_PER_DEVICE,
        format_vocab={str(i): list(k) for k, i in vocab.items()},
        a_windows=int(A.dev.size), months={}, per_device={},
        zscore="fitted label-free on the pooled windows of the two sets "
               "being compared (enrolment set + query set)",
        notes=[
            "A beamforming report is a digital payload: every receiver that "
            "decodes the frame obtains the same angle bits. The receiver "
            "changes only WHICH reports are heard and how much of each body "
            "is kept (512-byte firmware cap in B). Receiver and months "
            "therefore change together, but the angle content can only "
            "differ by time, not by receiver.",
            "Bins kept = bins fully covered in every B window of the gallery "
            "(AND over devices); the same mask is applied to A, including in "
            "the A_within_A reference.",
            "Devices absent from B (AppleTV-BR, Vacuum; Laptop has 114 "
            "Nr=4 reports in July only, first column undecodable at Nr=4) "
            "cannot be queried; chance is 1/|gallery| of devices present in "
            "both sets.",
            "A device whose majority format differs between A and B "
            "(per_device.format_differs_*) can never be matched by the "
            "format one-hot across sets; its angle features are compared in "
            "radians (angles) and after re-quantisation to 4/2 bits "
            "(angles_rq).",
        ])
    months = sorted({m for B in B_all for m in set(B.month.tolist())})
    for month in months:
        Bm_parts = [B.subset(B.month == month) for B in B_all]
        Bm_parts = [b for b in Bm_parts if b.dev.size]
        if not Bm_parts:
            continue
        # merge sources of the same month (sessions are per source already;
        # offset session ids so blocks stay distinct)
        Bm = Bm_parts[0]
        for extra in Bm_parts[1:]:
            off = Bm.session.max() + 1
            for k in ("dev", "ts0", "feat", "feat_rq", "gid", "nsub",
                      "month", "day"):
                setattr(Bm, k, np.concatenate([getattr(Bm, k),
                                               getattr(extra, k)]))
            Bm.session = np.concatenate([Bm.session, extra.session + off])
            Bm.geom_key += extra.geom_key
        present_B = sorted(set(Bm.dev.tolist()))
        gallery = [g for g in present_B if (A.dev == g).any()]
        Bg = Bm.subset(np.isin(Bm.dev, gallery))
        Ag = A.subset(np.isin(A.dev, gallery))
        mask = Bg.bin_mask()
        info = dict(
            b_windows=int(Bm.dev.size), b_days=sorted(set(Bm.day.tolist())),
            devices_in_B=[names[g] for g in present_B],
            gallery=[names[g] for g in gallery], chance=1.0 / max(1, len(gallery)),
            bins_kept=int(mask.sum()), bins_kept_idx=np.flatnonzero(mask).tolist(),
            b_windows_per_device={names[g]: int((Bm.dev == g).sum())
                                  for g in present_B},
            b_blocks_per_device={names[g]: len([k for k in Bm.blocks()
                                                if k[0] == g])
                                 for g in present_B},
            a_windows_per_device={names[g]: int((A.dev == g).sum())
                                  for g in gallery},
            a_blocks_per_device={names[g]: len([k for k in A.blocks()
                                                if k[0] == g])
                                 for g in gallery})
        for g in present_B:
            pd = summary["per_device"].setdefault(names[g], dict(
                format_A=list(A.majority_geom(g)) if (A.dev == g).any()
                else None))
            fb = Bm.majority_geom(g)
            pd[f"format_B_{month}"] = list(fb)
            pd[f"tones_present_B_{month}"] = int(Bm.nsub[Bm.dev == g].min())
            pd[f"format_differs_{month}"] = (pd["format_A"] is not None
                                             and list(fb) != pd["format_A"])
        if len(gallery) < 2 or mask.sum() == 0:
            info["skipped"] = "fewer than two devices in both sets or no bins"
            summary["months"][month] = info
            continue
        for cond, E, Q in (("A_to_B", Ag, Bg), ("B_to_A", Bg, Ag),
                           ("B_within_B", Bg, Bg), ("A_within_A", Ag, Ag)):
            tt = time.time()
            hits = run_condition(cond, E, Q, gallery, mask, n_vocab, rng)
            rows += tally_rows(cond, E.tag if cond[0] == "A" else f"B-{month}",
                               Q.tag if cond.endswith("A") else f"B-{month}",
                               month, hits, gallery, names, mask.sum())
            print(f"[{month}] {cond} {time.time() - tt:.0f}s, peak RSS "
                  f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
        summary["months"][month] = info
    summary["runtime_s"] = time.time() - t0
    summary["peak_rss_mb"] = X.peak_rss_mb()
    with open(OUT["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    X.write_json(OUT["summary"], summary)
    print(f"[out] {OUT['csv']} ({len(rows)} rows), {summary['runtime_s']:.0f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
