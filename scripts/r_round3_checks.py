#!/usr/bin/env python3
"""R-round3 checks: three analyses on existing data, no new capture.

1  SHARED SOUNDING. The rotation of Section 7.2 draws one unitary Q_k per
   sounding and tone group; a client reports Q_k^H V_k. Two clients a, b
   answering the SAME sounding give an observer (Q_k^H V_a)^H (Q_k^H V_b)
   = V_a^H V_b, in which Q_k has cancelled.
   1a  home capture: pairs of reports of two different labelled devices
       closer than NEAR_THRESHOLDS_S, per day and device pair, and the share
       of each device's reports with such a partner. A pair counts as one
       shared sounding when it also has the same access point, the same
       generation and the same sounding dialog token (MIMO Control field,
       release/decoder/angle_decode.py parse_vht_mimo_ctrl /
       parse_he_mimo_ctrl) inside the widest threshold.
       On the shared soundings found: |v_a^H v_b| of the two first columns
       per tone, 32 tone bins, as captured, after one shared Q_k per tone
       (re-quantised at the reports' codebook and at 4/2 bits) and after
       independent Q_k (control); 1-NN over the pair classes, one enrolled
       sounding per class from another day or another hour.
   1b  He CR1: every report of a held-out position is paired with a report
       of a partner position (a calibration position, the static second
       client) under the same Q_k per group, both reports quantised as the
       existing emulation does. Feature |v_a^H v_b| per group (59 values),
       cosine 1-NN, K = 1, T in {1, 32} (r_rotation_space's he protocol),
       next to the plain observer of the single report and to independent
       Q_k per client.
2  COMBINED DEFENSE, home capture: rotation per group on the common grid
   (body rot_group_cb42_n234), one report format for all devices (a
   constant format column, so the format carries nothing) and the stream-1
   SNR byte in steps of q dB or removed. Observer: r_eval's 1-NN on the
   z-scored concatenation [angle window vector, format column, SNR mean and
   standard deviation], and on the SNR pair alone. Episodes are r_eval's
   (other_day_any, r_eval's grids and seed, traced by r_eval_checks on the
   pure-noise table, whose stored decisions are reproduced); every feature
   set is decided on those same episodes (r_eval_checks.decide), so the
   difference to the noise reference is paired and no tie in a coarse SNR
   value can move the random stream.
3  DP-GSQ, OUR IMPLEMENTATION OF A FULL-SUPPORT VARIANT, not the authors'
   code: for an angle in cell c of M = 2^b cells, cell j is reported with
   probability proportional to exp(-eps d(c, j) / (2 Delta)), d the
   circular cell distance for phi and the linear one for psi, Delta its
   largest value (exponential mechanism, eps-local-DP per angle). He CR1,
   scored as the DP-SQ rows of r_raw_observer_defenses.py (raw and scaled
   cosine 1-NN) and of he_beamforming_gain.py (first-stream gain on the
   same 3000 reports per seed).

Memory: one archive day at a time; only the MIMO Control head of a body is
read except for the reports of a shared sounding; He in chunks.
Usage: python3 scripts/r_round3_checks.py [--dry-run] [--parts 1 2 3]
Outputs (figures/): r_round3_checks.json, r_round3_checks.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# One BLAS thread, as r_rotation_ndp.py and r_raw_observer_defenses.py.
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, BLAS_THREADS)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_defense_common as C  # noqa: E402
import joint_defense_eval as J  # noqa: E402
import he_beamforming_gain as HG  # noqa: E402
import manifold_sweep as MS  # noqa: E402
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_eval_checks as K  # noqa: E402
import r_format_equalized as FE  # noqa: E402
import r_raw_observer_defenses as RD  # noqa: E402
import r_rotation as RR  # noqa: E402
import r_rotation_ndp as RN  # noqa: E402
import r_rotation_space as RS  # noqa: E402
import r_snr_field as SN  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARTS = ("1", "2", "3")
ROUND3_SEED = 23
RSS_BUDGET_MB = 3000.0
ROTATION = "rot_group"                    # one Q per report and tone group

# Part 1a ---------------------------------------------------------------
NEAR_THRESHOLDS_S = (0.001, 0.005, 0.020)
SHARED_MAX_DT_S = max(NEAR_THRESHOLDS_S)
HEAD_BYTES = X.AD.MIN_BFI_BODY_BYTES      # category, action, MIMO Control
HEX_PER_BYTE = 2
TOKEN_VALUES = 64                         # 6-bit sounding dialog token
ROWID_CHUNK = 500
REAL_BLOCK_S = 3600.0                     # an hour of capture is one block
REAL_MIN_SOUNDINGS = 50                   # per pair class
REAL_TRIALS = E.N_TRIALS
REAL_VARIANTS = {                         # name -> (rotation, output spec)
    "as_captured": (None, RR.NATIVE_SPEC),
    "shared_q_native": ("shared", RR.NATIVE_SPEC),
    "shared_q_cb42": ("shared", RR.A2_OUTPUTS["rot_group_cb42"]),
    "independent_q_native": ("independent", RR.NATIVE_SPEC),
}
REAL_CONDITIONS = ("other_day", "other_hour")
STREAM_REAL_Q, STREAM_REAL_EVAL = 0, 1

# Part 1b ---------------------------------------------------------------
HE_SEEDS = RN.HE_SEEDS                    # 10
HE_T_GRID = RS.T_GRID                     # (1, 32)
HE_PARTNER_CLASS_INDICES = (0, 1, 2)      # calibration positions used as partner
PARTNER_MODES = ("random_report", "fixed_report")
HE_FIXED_PARTNER_REPORT = 0
HE_ROTATIONS = ("none", "shared_q", "independent_q")
STREAM_HE_Q, STREAM_HE_Q2, STREAM_HE_PARTNER, STREAM_HE_SCALER = 10, 11, 12, 13
SCALINGS = ("scaled", "raw")

# Part 2 ----------------------------------------------------------------
BODY = RN.COMMON_BODY                     # rot_group_cb42_n234
HOME_COND = K.DEFENSE_CONDITION           # other_day_any
HOME_K = K.DEFENSE_K                      # (1, 0), 0 = all enrolled windows
SNR_RECORDED_DB = 1.0 / SN.BYTES_PER_DB   # the field's own step, 0.25 dB
SNR_STEPS_DB = (SNR_RECORDED_DB, 1.0, 3.0, 6.0, 12.0)
SNR_REMOVED = "removed"
SNR_STREAMS = 1                           # stream-1 byte only
SNR_DIM = 2 * SNR_STREAMS                 # window mean and standard deviation
FORMAT_DIM = 1                            # one format for every device
STORED_SCALING = "stored"                 # r_eval's z-score on all windows
REPORT_SCHEME = "block"                   # r_eval.boot_ci's clusters
STREAM_NULL_EXTRA = 20

# Part 3 ----------------------------------------------------------------
GSQ_EPS = (0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
GSQ_STREAM_OFFSET = 300                   # defense streams not used elsewhere
GSQ_GAIN_FLOORS = (0.95, 0.90)
GSQ_LABEL = ("our implementation of a full-support variant (exponential "
             "mechanism over the 2^b cells of each angle), not the authors' code")
GSQ_SENSITIVITY_FACTOR = 2.0              # exp(-eps d / (2 Delta))
REF_NAME = "none"

OUT_JSON = R.FIG_DIR / "r_round3_checks.json"
OUT_CSV = R.FIG_DIR / "r_round3_checks.csv"
CSV_FIELDS = ("part", "analysis", "cell", "observer", "K", "T", "metric",
              "value", "lo", "hi", "ub95", "chance", "n", "note")


def row(part: str, analysis: str, cell: str, metric: str, value, **kw) -> dict:
    r = dict.fromkeys(CSV_FIELDS, "")
    r.update(part=part, analysis=analysis, cell=cell, metric=metric, value=value)
    r.update(kw)
    return r


def stat(v) -> dict:
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                n=int(v.size))


def log(tag: str) -> None:
    pk = X.log_rss(tag)
    if pk > RSS_BUDGET_MB:
        print(f"[rss] WARNING above the {RSS_BUDGET_MB:.0f} MB budget", file=sys.stderr)


def pair_feature(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """|v_a^H v_b| per report and tone group; va, vb (n, k, nr)."""
    return np.abs(np.einsum("nkr,nkr->nk", np.conj(va), vb))


# ---------------------------------------------------------------------------
# 1a  shared soundings in the home capture
# ---------------------------------------------------------------------------
def head_fields(head: bytes) -> tuple[int, int] | None:
    """(category, sounding dialog token) from the MIMO Control field."""
    if len(head) < HEAD_BYTES:
        return None
    if head[0] == X.AD.CATEGORY_VHT:
        p = X.AD.parse_vht_mimo_ctrl(head[2:2 + X.AD.VHT_MIMO_CTRL_LEN])
    elif head[0] == X.AD.CATEGORY_HE:
        p = X.AD.parse_he_mimo_ctrl(head[2:2 + X.AD.HE_MIMO_CTRL_LEN])
    else:
        return None
    return (head[0], p["dialog_token"]) if p else None


def scan_day(p: Path, devs: list[str]) -> dict:
    """Per report of the labelled devices: rowid, time, device, access point
    index, category and dialog token; only the body's head is read."""
    c = X.connect_ro(p)
    q = (f"SELECT rowid, ts, {X.SRC_COL}, {X.AP_COL}, substr(body_hex, 1, ?) "
         f"FROM bfi WHERE {X.SRC_COL} IN ({','.join('?' * len(devs))}) "
         "AND body_hex IS NOT NULL ORDER BY ts")
    cols = defaultdict(list)
    aps: dict[str, int] = {}
    for rid, t, m, ap, hh in c.execute(q, [HEX_PER_BYTE * HEAD_BYTES] + devs):
        f = head_fields(bytes.fromhex(hh))
        if f is None:
            continue
        cols["rowid"].append(rid)
        cols["ts"].append(t)
        cols["dev"].append(devs.index(m))
        cols["ap"].append(aps.setdefault(ap, len(aps)))
        cols["cat"].append(f[0])
        cols["token"].append(f[1])
    c.close()
    return {k: np.asarray(v) for k, v in cols.items()}


def near_pairs(ts: np.ndarray, dev: np.ndarray, max_dt: float):
    """Index pairs (i < j in time) of two different devices closer than max_dt."""
    ii, jj = [], []
    for lag in range(1, ts.size):
        close = ts[lag:] - ts[:-lag] < max_dt
        if not close.any():
            break
        i = np.flatnonzero(close & (dev[lag:] != dev[:-lag]))
        ii.append(i)
        jj.append(i + lag)
    if not ii:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def pair_name(names: list[str], a: int, b: int) -> str:
    return "+".join(sorted((names[a], names[b])))


def day_sharing(d: dict, names: list[str]) -> tuple[dict, dict]:
    """Counts of one day per threshold, and its shared soundings."""
    ts, dev = d["ts"], d["dev"]
    i, j = near_pairs(ts, dev, SHARED_MAX_DT_S)
    dt = ts[j] - ts[i]
    same = ((d["ap"][i] == d["ap"][j]) & (d["cat"][i] == d["cat"][j])
            & (d["token"][i] == d["token"][j]))
    frames = np.bincount(dev, minlength=len(names))
    out = {}
    for th in NEAR_THRESHOLDS_S:
        for kind, sel in (("near", dt < th), ("shared", (dt < th) & same)):
            per_pair = defaultdict(int)
            for a, b in zip(dev[i[sel]], dev[j[sel]]):
                per_pair[pair_name(names, a, b)] += 1
            with_partner = np.bincount(
                dev[np.unique(np.concatenate([i[sel], j[sel]]))], minlength=len(names))
            out[f"{kind}|{th * 1e3:g}ms"] = dict(
                pairs=int(sel.sum()), per_pair=dict(per_pair),
                reports_with_partner={n: int(v) for n, v in zip(names, with_partner)})
        same_gen = (dt < th) & (d["ap"][i] == d["ap"][j]) & (d["cat"][i] == d["cat"][j])
        out[f"near|{th * 1e3:g}ms"]["same_ap_and_generation"] = int(same_gen.sum())
    out["frames"] = {n: int(v) for n, v in zip(names, frames)}
    shared = dict(i=i[same], j=j[same], dt=dt[same])
    return out, shared


def fetch_bodies(p: Path, rowids: np.ndarray) -> dict[int, bytes]:
    c = X.connect_ro(p)
    out = {}
    ids = [int(r) for r in np.unique(rowids)]
    for s in range(0, len(ids), ROWID_CHUNK):
        part = ids[s:s + ROWID_CHUNK]
        for rid, bh in c.execute(
                f"SELECT rowid, body_hex FROM bfi WHERE rowid IN "
                f"({','.join('?' * len(part))})", part):
            out[rid] = bytes.fromhex(bh)
    c.close()
    return out


def rotated_first_columns(qa, ga, qb, gb, variant: str, rng):
    """First columns (n, k, nr) of the two reports of each shared sounding as
    an observer reads them under a variant of REAL_VARIANTS."""
    mode, spec = REAL_VARIANTS[variant]
    if mode is None:
        return RR.reconstruct_v(qa, ga)[..., 0], RR.reconstruct_v(qb, gb)[..., 0]
    Va, Vb = RR.reconstruct_v(qa, ga), RR.reconstruct_v(qb, gb)
    Q, Va2 = RN.rotate_keep_q(Va, ROTATION, rng)
    if mode == "shared":
        Vb2 = np.conj(np.swapaxes(Q, -1, -2)) @ Vb
    else:
        _, Vb2 = RN.rotate_keep_q(Vb, ROTATION, rng)
    out = []
    for V2, g in ((Va2, ga), (Vb2, gb)):
        g_out = RR.output_geom(g, spec)
        phi, psi = C.v_to_givens(V2, nr=g["nr"], nc=g["nc"])
        out.append(RR.reconstruct_v(RR.quantise(phi, psi, g_out), g_out)[..., 0])
    return out[0], out[1]


def real_pair_features(p: Path, d: dict, shared: dict, names: list[str],
                       daynum: int) -> tuple[dict, int]:
    """Pair class, time and the binned |v_a^H v_b| per variant of every
    shared sounding of one day whose two reports decode on one tone grid."""
    cols = defaultdict(list)
    if shared["i"].size == 0:
        return cols, 0
    bodies = fetch_bodies(p, np.concatenate([d["rowid"][shared["i"]],
                                             d["rowid"][shared["j"]]]))
    groups = defaultdict(list)
    gdict = {}
    for n, (i, j) in enumerate(zip(shared["i"], shared["j"])):
        if d["dev"][i] > d["dev"][j]:
            i, j = j, i                          # device order, not time order
        ra, rb = bodies[int(d["rowid"][i])], bodies[int(d["rowid"][j])]
        (ga, ka), (gb, kb) = R._geom_from_body(ra), R._geom_from_body(rb)
        if ga is None or gb is None:
            continue
        gdict[ka], gdict[kb] = ga, gb
        groups[(ka, len(ra), kb, len(rb))].append((i, j, ra, rb))
    skipped = 0
    for gi, gkey in enumerate(sorted(groups, key=str)):
        ka, _la, kb, _lb = gkey
        items = groups[gkey]
        ga, gb = gdict[ka], gdict[kb]
        qa = RR.decode_all([x[2] for x in items], ga)
        qb = RR.decode_all([x[3] for x in items], gb)
        if qa is None or qb is None or qa.shape[2] != qb.shape[2]:
            skipped += len(items)
            continue
        ii = np.array([x[0] for x in items])
        jj = np.array([x[1] for x in items])
        cols["cls"] += [pair_name(names, a, b) for a, b in zip(d["dev"][ii], d["dev"][jj])]
        cols["ts"].append(np.minimum(d["ts"][ii], d["ts"][jj]))
        for vi, variant in enumerate(REAL_VARIANTS):
            rng = np.random.default_rng([ROUND3_SEED, STREAM_REAL_Q, daynum, gi, vi])
            va, vb = rotated_first_columns(qa, ga, qb, gb, variant, rng)
            cols[variant].append(RS.bin_tones(pair_feature(va, vb)[..., None]))
    return cols, skipped


def real_pair_eval(feat: np.ndarray, cls: np.ndarray, group: np.ndarray,
                   rng) -> dict:
    """1-NN with one enrolled sounding per pair class. For every query group
    (a day or an hour) the gallery holds one random sounding per class from
    the other groups; every sounding of the group is a query. z-score fitted
    label-free on all soundings (r_eval.Table's rule)."""
    mu, sd = K.fit_scale(feat.astype(np.float64))
    z = np.nan_to_num((feat - mu) / sd)
    classes = sorted(set(cls.tolist()))
    hits = defaultdict(list)
    sizes = []
    for _ in range(REAL_TRIALS):
        for g in np.unique(group):
            gal = {}
            for c in classes:
                cand = np.flatnonzero((cls == c) & (group != g))
                if cand.size:
                    gal[c] = int(rng.choice(cand))
            if len(gal) < 2:
                continue
            gc_ = list(gal)
            S = z[[gal[c] for c in gc_]]
            for c in gc_:
                qi = np.flatnonzero((cls == c) & (group == g))
                if not qi.size:
                    continue
                dd = ((z[qi][:, None, :] - S[None]) ** 2).sum(-1)
                hits[c] += (np.array(gc_)[dd.argmin(1)] == c).tolist()
                sizes += [len(gc_)] * qi.size
    if not hits:
        return dict(n_decisions=0)
    allh = np.concatenate([np.asarray(v, float) for v in hits.values()])
    return dict(accuracy=float(allh.mean()),
                macro_recall=float(np.mean([np.mean(v) for v in hits.values()])),
                chance=float(np.mean(1.0 / np.asarray(sizes))),
                per_class_recall={c: float(np.mean(v)) for c, v in hits.items()},
                per_class_decisions={c: len(v) for c, v in hits.items()},
                n_decisions=int(allh.size), trials=REAL_TRIALS)


def part1a(devs: list[str]) -> tuple[dict, list[dict]]:
    names = [R.short(m) for m in devs]
    days, inst = {}, defaultdict(list)
    skipped = 0
    for p in X.day_paths():
        d = scan_day(p, devs)
        day = p.stem.replace("bfi_", "")
        if not d or not d["ts"].size:
            continue
        days[day], shared = day_sharing(d, names)
        cols, sk = real_pair_features(p, d, shared, names, int(day.replace("-", "")))
        skipped += sk
        if cols["cls"]:
            ts = np.concatenate(cols["ts"])
            inst["cls"] += cols["cls"]
            inst["day"] += [day] * ts.size
            inst["hour"] += [f"{day}|{int(t // REAL_BLOCK_S)}" for t in ts]
            for v in REAL_VARIANTS:
                inst[v].append(np.concatenate(cols[v]))
        del d, cols
        gc.collect()
        log(f"1a {day}")
    # totals over days
    totals = {}
    frames = defaultdict(int)
    for dd in days.values():
        for n, v in dd["frames"].items():
            frames[n] += v
    for key in next(iter(days.values())):
        if key == "frames":
            continue
        per_pair, partner = defaultdict(int), defaultdict(int)
        for dd in days.values():
            for k, v in dd[key]["per_pair"].items():
                per_pair[k] += v
            for k, v in dd[key]["reports_with_partner"].items():
                partner[k] += v
        totals[key] = dict(
            pairs=sum(dd[key]["pairs"] for dd in days.values()),
            per_pair=dict(per_pair),
            fraction_of_reports_with_partner={
                n: (partner[n] / frames[n] if frames[n] else None) for n in names})
        if key.startswith("near"):
            sg = sum(dd[key]["same_ap_and_generation"] for dd in days.values())
            totals[key].update(same_ap_and_generation=sg,
                               token_match_expected_by_chance=sg / TOKEN_VALUES)
    rows = []
    for key, t in totals.items():
        rows.append(row("1a", "shared_sounding_counts", key, "pairs", t["pairs"]))
        for pn, v in t["per_pair"].items():
            rows.append(row("1a", "shared_sounding_counts", key, "pairs", v, observer=pn))
        for n, v in t["fraction_of_reports_with_partner"].items():
            rows.append(row("1a", "shared_sounding_counts", key,
                            "fraction_of_reports_with_partner", v, observer=n,
                            n=frames[n]))
    # the pairwise feature on the real shared soundings
    real = dict(soundings_undecodable=skipped, variants=list(REAL_VARIANTS))
    if inst["cls"]:
        cls = np.asarray(inst["cls"])
        day, hour = np.asarray(inst["day"]), np.asarray(inst["hour"])
        cnt = {c: int((cls == c).sum()) for c in sorted(set(cls.tolist()))}
        keep = np.isin(cls, [c for c, n in cnt.items() if n >= REAL_MIN_SOUNDINGS])
        real.update(
            soundings_per_class=cnt,
            soundings_per_class_and_day={
                c: {dy: int(((cls == c) & (day == dy)).sum())
                    for dy in sorted(set(day[cls == c].tolist()))} for c in cnt},
            classes_scored=sorted(set(cls[keep].tolist())), results={})
        for vi, v in enumerate(REAL_VARIANTS):
            f = np.concatenate(inst[v])[keep]
            for ci, (cond, grp) in enumerate(zip(REAL_CONDITIONS, (day, hour))):
                r = real_pair_eval(f, cls[keep], grp[keep], np.random.default_rng(
                    [ROUND3_SEED, STREAM_REAL_EVAL, vi, ci]))
                real["results"][f"{v}|{cond}"] = r
                for m in ("accuracy", "macro_recall"):
                    if r.get("n_decisions"):
                        rows.append(row("1a", "real_pair_feature", v, m, r[m],
                                        observer="pair_1nn", K=1, T=1, chance=r["chance"],
                                        n=r["n_decisions"], note=cond))
    return dict(
        thresholds_s=list(NEAR_THRESHOLDS_S),
        shared_definition="two reports of different labelled devices, same access "
                          "point, same generation, same sounding dialog token, "
                          f"closer than {SHARED_MAX_DT_S * 1e3:g} ms",
        dialog_token_source="release/decoder/angle_decode.py: parse_vht_mimo_ctrl "
                            "(bits 18-23) and parse_he_mimo_ctrl (bits 30-35)",
        frames=dict(frames), totals=totals, per_day=days, real_pairs=real), rows


# ---------------------------------------------------------------------------
# 1b  He CR1: two clients under one Q_k
# ---------------------------------------------------------------------------
def partner_rows(mode: str, n: int, n_partner: int, rng) -> np.ndarray:
    if mode == "fixed_report":
        return np.full(n, HE_FIXED_PARTNER_REPORT)
    return rng.integers(n_partner, size=n)


def quantised_v(phi, psi) -> np.ndarray:
    return C.givens_to_v(*C.he_quantise(phi, psi))


def he_pair_scalers(ds: J.He, partners: dict) -> dict:
    """Per (partner, mode): standardisation of the pairwise feature on the
    other calibration positions (quantised reports, no rotation)."""
    from sklearn.preprocessing import StandardScaler
    S = {}
    for pi, (cls, (pphi, ppsi)) in enumerate(partners.items()):
        keep = np.flatnonzero(ds.y["cal"] != cls)
        for mi, mode in enumerate(PARTNER_MODES):
            rng = np.random.default_rng([ROUND3_SEED, STREAM_HE_SCALER, pi, mi])
            sc = StandardScaler()
            for s in range(0, keep.size, J.HE_CHUNK):
                rows_ = keep[s:s + J.HE_CHUNK]
                j = partner_rows(mode, rows_.size, pphi.shape[0], rng)
                sc.partial_fit(pair_feature(
                    quantised_v(ds.phi["cal"][rows_], ds.psi["cal"][rows_])[..., 0],
                    quantised_v(pphi[j], ppsi[j])[..., 0]))
            S[(cls, mode)] = sc
    return S


def he_pair_seed(ds: J.He, partners: dict, S: dict, seed: int, acc: dict) -> float:
    """One seed: plain and pairwise observers under every rotation; returns
    the largest |feature under a shared Q - feature without rotation| with
    unquantised reports (Q must cancel)."""
    phi, psi, y = ds.phi["held"], ds.psi["held"], ds.y["held"]
    supports = C.pick_supports(y, seed)
    rng_q = np.random.default_rng([ROUND3_SEED, STREAM_HE_Q, seed])
    rng_q2 = np.random.default_rng([ROUND3_SEED, STREAM_HE_Q2, seed])
    rng_p = {(cls, mode): np.random.default_rng(
        [ROUND3_SEED, STREAM_HE_PARTNER, seed, pi, mi])
        for pi, cls in enumerate(partners) for mi, mode in enumerate(PARTNER_MODES)}
    feats = defaultdict(list)
    err = 0.0
    for s in range(0, phi.shape[0], J.HE_CHUNK):
        sl = slice(s, s + J.HE_CHUNK)
        V = C.givens_to_v(phi[sl], psi[sl])
        Q, V2 = RN.rotate_keep_q(V, ROTATION, rng_q)
        QH = np.conj(np.swapaxes(Q, -1, -2))
        pq, sq = C.he_quantise(phi[sl], psi[sl])
        pr, sr = C.he_quantise(*C.v_to_givens(V2))
        feats[("none", "plain")].append(C.he_features(pq, sq))
        feats[("shared_q", "plain")].append(C.he_features(pr, sr))
        a_none = C.givens_to_v(pq, sq)[..., 0]
        a_rot = C.givens_to_v(pr, sr)[..., 0]
        for (cls, mode), rng in rng_p.items():
            pphi, ppsi = partners[cls]
            j = partner_rows(mode, V.shape[0], pphi.shape[0], rng)
            Vp = C.givens_to_v(pphi[j], ppsi[j])
            feats[("none", cls, mode)].append(
                pair_feature(a_none, quantised_v(pphi[j], ppsi[j])[..., 0]))
            Vp2 = QH @ Vp
            feats[("shared_q", cls, mode)].append(
                pair_feature(a_rot, quantised_v(*C.v_to_givens(Vp2))[..., 0]))
            err = max(err, float(np.abs(pair_feature(V2[..., 0], Vp2[..., 0])
                                        - pair_feature(V[..., 0], Vp[..., 0])).max()))
            if mode == PARTNER_MODES[0]:
                _, Vp3 = RN.rotate_keep_q(Vp, ROTATION, rng_q2)
                feats[("independent_q", cls, mode)].append(
                    pair_feature(a_rot, quantised_v(*C.v_to_givens(Vp3))[..., 0]))
                del Vp3
            del Vp, Vp2
        del V, V2, Q, QH
    for ki, key in enumerate(sorted(feats, key=str)):
        F = np.concatenate(feats[key])
        scaler = ds.S0 if key[1] == "plain" else S[(key[1], key[2])]
        for si, (scaling, Z) in enumerate((("scaled", scaler.transform(F)), ("raw", F))):
            r = C.multi_query_eval(
                Z, y, supports, metric="cos", T_grid=HE_T_GRID,
                n_decisions=RS.HE_N_DECISIONS,
                rng=np.random.default_rng([seed, ki, si, RN.HE_EVAL_STREAM]))
            for T, a in r.items():
                acc[key + (scaling,)][T].append(a)
        del F
    del feats
    gc.collect()
    return err


def part1b(ds: J.He, seeds: int) -> tuple[dict, list[dict]]:
    cal_classes = np.unique(ds.y["cal"])
    partners = {}
    for i in HE_PARTNER_CLASS_INDICES:
        m = ds.y["cal"] == cal_classes[i]
        partners[int(cal_classes[i])] = (ds.phi["cal"][m], ds.psi["cal"][m])
    S = he_pair_scalers(ds, partners)
    acc = defaultdict(lambda: defaultdict(list))
    err = 0.0
    for seed in range(1, seeds + 1):
        err = max(err, he_pair_seed(ds, partners, S, seed, acc))
        log(f"1b seed {seed}")
    chance = 1.0 / len(np.unique(ds.y["held"]))
    cells, rows = {}, []
    # per partner, and the mean over partners of the per-seed accuracies
    pooled = defaultdict(lambda: defaultdict(list))
    for key, byT in acc.items():
        rot, scaling = key[0], key[-1]
        obs = "plain" if key[1] == "plain" else f"pair|{key[2]}"
        name = f"{rot}|{obs}|{scaling}" + ("" if obs == "plain" else f"|partner{key[1]}")
        cells[name] = {f"T={T}": stat(v) for T, v in byT.items()}
        if obs != "plain":
            for T, v in byT.items():
                pooled[(rot, obs, scaling)][T].append(v)
    for (rot, obs, scaling), byT in pooled.items():
        cells[f"{rot}|{obs}|{scaling}|all_partners"] = {
            f"T={T}": dict(stat(np.mean(v, axis=0)), partners=len(v),
                           per_partner_mean=[float(np.mean(x)) for x in v])
            for T, v in byT.items()}
    for name, byT in cells.items():
        rot, obs = name.split("|")[0], "|".join(name.split("|")[1:])
        for T, st in byT.items():
            rows.append(row("1b", "he_shared_sounding", rot, "accuracy", st["mean"],
                            observer=obs, K=1, T=T.replace("T=", ""), chance=chance,
                            n=st["n"], note=f"sd over seeds {st['sd']:.4f}"))
    return dict(
        protocol="He CR1, 19 held-out positions, cosine 1-NN, K = 1, "
                 f"T in {list(HE_T_GRID)}, {seeds} seeds, {RS.HE_N_DECISIONS} "
                 "decisions per class at T > 1",
        feature="|v_a^H v_b| of the first columns per tone group (59 values), "
                "both reports quantised at 4-bit phi / 2-bit psi",
        partner="a calibration position; random_report = a fresh partner report "
                "per sounding, fixed_report = one partner report for all",
        partner_classes=[int(c) for c in partners], chance=chance, seeds=seeds,
        q_cancels_max_abs_err_unquantised=err, cells=cells), rows


# ---------------------------------------------------------------------------
# 2  combined defense on the home capture
# ---------------------------------------------------------------------------
def coarsen(snr: dict, step_db: float) -> dict:
    """The SNR bytes in cells of step_db, counted from the field's floor
    (cells nest, as a coarser field would); the cell index times the step."""
    step = SN.BYTES_PER_DB * step_db
    return {i: (t, np.floor((v - SN.BYTE_MIN) / step) * step + SN.BYTE_MIN)
            for i, (t, v) in snr.items()}


def zscore(f: np.ndarray) -> np.ndarray:
    f = f.astype(np.float64)
    mu, sd = K.fit_scale(f)
    return np.nan_to_num((f - mu) / sd)


def decide_all(eps: list, z: np.ndarray, rng_t) -> tuple[dict, dict]:
    hits, ties = {}, {}
    for k in HOME_K:
        res = [K.decide(ep, k, STORED_SCALING, z, z, rng_t) for ep in eps]
        hits[k] = np.array([h for h, _ in res])
        ties[k] = int(sum(t for _, t in res))
    return hits, ties


def verdict(r: dict) -> dict:
    d = r["intervals"][REPORT_SCHEME]["macro_recall_minus_reference"]
    return dict(macro_minus_reference=r["macro_recall"] - r["reference_macro_recall"],
                interval_contains_zero=bool(d["lo"] <= 0.0 <= d["hi"]),
                upper_bound_within_margin=bool(d["ub95"] <= K.CHANCE_MARGIN))


def part2(devs: list[str], trials: int) -> tuple[dict, list[dict]]:
    w_ref = R.get_windows(devs, R.SESSION_GAP_S)
    w_null = K.null_windows(w_ref)
    tb = E.Table(w_null)
    names = tb.names
    eps = K.traced_episodes(tb, HOME_COND, trials)
    log("2 episodes traced")
    rng_b = np.random.default_rng([ROUND3_SEED, K.BOOT_SEED])
    rng_t = np.random.default_rng([ROUND3_SEED, K.BOOT_SEED + 1])
    n = w_ref["feat"].shape[0]
    null_ang = w_null["feat"]
    # the stored noise reference, decision for decision (hits_of asserts it)
    stored = K.stored_defense()
    checks = {}
    for k in HOME_K:
        h, _ = K.hits_of(eps, k, STORED_SCALING, tb.Z, tb.Z, rng_t)
        pt = K.point(h, eps, names)
        ref = stored.get((RR.NULL_BODY, K.FEATURE, E.klabel(k)))
        if ref is not None and trials == E.N_TRIALS:
            checks[f"null_noise_K={E.klabel(k)}_vs_r_rotation_main"] = max(
                abs(pt[m] - float(ref[m])) for m in ("accuracy", "macro_recall"))
    del tb
    w_rot = K.defense_windows(BODY, devs, w_ref)
    ang = w_rot["feat"]
    del w_rot, w_null
    gc.collect()
    fmt = np.ones((n, FORMAT_DIM), np.float32)
    null_snr = np.random.default_rng([ROUND3_SEED, STREAM_NULL_EXTRA]) \
        .standard_normal((n, SNR_DIM)).astype(np.float32)
    snr, _ = SN.frame_snr(devs)
    log("2 SNR bytes read")
    sets = {}                                    # name -> (features, reference)
    for q in SNR_STEPS_DB:
        s = SN.window_features(w_ref, coarsen(snr, q), SNR_STREAMS)
        sets[(f"{q:g}", "combined")] = (np.hstack([ang, fmt, s]),
                                        np.hstack([null_ang, fmt, null_snr]))
        sets[(f"{q:g}", "snr_alone")] = (s, null_snr)
    s_rec = SN.window_features(w_ref, snr, SNR_STREAMS)
    checks["recorded_step_is_identity"] = bool(np.array_equal(
        s_rec, SN.window_features(w_ref, coarsen(snr, SNR_RECORDED_DB), SNR_STREAMS)))
    sets[(SNR_REMOVED, "combined")] = (np.hstack([ang, fmt]), np.hstack([null_ang, fmt]))
    del snr
    gc.collect()
    ref_cache, cells, rows = {}, {}, []
    for (q, obs), (f, f_ref) in sets.items():
        ref_key = (obs, f_ref.shape[1])
        if ref_key not in ref_cache:
            ref_cache[ref_key] = decide_all(eps, zscore(f_ref), rng_t)
        ref_hits, _ = ref_cache[ref_key]
        hits, ties = decide_all(eps, zscore(f), rng_t)
        for k in HOME_K:
            r = K.score(eps, hits[k], names, rng_b, ref_hits[k])
            r0 = K.point(ref_hits[k], eps, names)
            r.update(ties=ties[k], feature_dim=int(f.shape[1]),
                     reference_accuracy=r0["accuracy"],
                     reference_macro_recall=r0["macro_recall"])
            r["verdict"] = verdict(r)
            name = f"snr_step={q}|{obs}|K={E.klabel(k)}"
            cells[name] = r
            iv = r["intervals"][REPORT_SCHEME]
            base = dict(observer=obs, K=E.klabel(k), T=K.T_REPORT, chance=r["chance"],
                        n=r["n_decisions"], note=f"{REPORT_SCHEME} bootstrap")
            for m in ("accuracy", "macro_recall"):
                rows.append(row("2", "combined_defense", f"snr_step={q}", m, r[m],
                                lo=iv[m]["lo"], hi=iv[m]["hi"], ub95=iv[m]["ub95"], **base))
                rows.append(row("2", "combined_defense", f"snr_step={q}",
                                f"reference_{m}", r[f"reference_{m}"], **base))
                dm = iv[f"{m}_minus_reference"]
                rows.append(row("2", "combined_defense", f"snr_step={q}",
                                f"{m}_minus_reference", r[m] - r[f"reference_{m}"],
                                lo=dm["lo"], hi=dm["hi"], ub95=dm["ub95"], **base))
        log(f"2 {q} {obs}")
    for k in HOME_K:
        ref = stored.get((BODY, K.FEATURE, E.klabel(k)))
        c = cells[f"snr_step={SNR_REMOVED}|combined|K={E.klabel(k)}"]
        if ref is not None and trials == E.N_TRIALS:
            checks[f"removed_K={E.klabel(k)}_vs_r_rotation_main"] = max(
                abs(c[m] - float(ref[m])) for m in ("accuracy", "macro_recall"))
    fe = json.loads(FE.OUT["json"].read_text()) if FE.OUT["json"].exists() else {}
    return dict(
        body=BODY, condition=HOME_COND, T=K.T_REPORT, trials=trials,
        K=[E.klabel(k) for k in HOME_K], n_boot=K.N_BOOT, ci_level=K.CI_LEVEL,
        upper_level=K.UPPER_LEVEL, chance_margin=K.CHANCE_MARGIN,
        interval_scheme_reported=REPORT_SCHEME,
        features="z-scored concatenation [320 angle values, 1 constant format "
                 "column, mean and std of the stream-1 SNR byte]; snr_alone = "
                 "the last two",
        reference="pure noise of the same shape: r_rotation's null_noise table "
                  "for the angle block, the same constant format column, i.i.d. "
                  "standard normal SNR columns",
        snr_steps_db=[f"{q:g}" for q in SNR_STEPS_DB] + [SNR_REMOVED],
        snr_recorded_step_db=SNR_RECORDED_DB,
        one_format_is_an_assumption="every window carries the same format; the "
                                    "enforceable policies of r_format_equalized "
                                    "leave more than one class",
        format_classes_left_by_policy={p: v["n_classes_all"]
                                       for p, v in fe.get("classes", {}).items()},
        episodes=len(eps), checks=checks, cells=cells), rows


# ---------------------------------------------------------------------------
# 3  DP-GSQ (our full-support variant) on He CR1
# ---------------------------------------------------------------------------
def gsq_matrix(b: int, eps: float, circular: bool) -> np.ndarray:
    """P[c, j]: probability of reporting cell j for true cell c."""
    m = 1 << b
    d = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    if circular:
        d = np.minimum(d, m - d)
    p = np.exp(-eps * d / (GSQ_SENSITIVITY_FACTOR * d.max()))
    return p / p.sum(axis=1, keepdims=True)


def gsq_quantise_1d(x, lo: float, hi: float, b: int, eps: float, rng,
                    circular: bool) -> np.ndarray:
    m = 1 << b
    cell = (hi - lo) / m
    c = np.clip(np.floor((x - lo) / cell).astype(int), 0, m - 1)
    P = gsq_matrix(b, eps, circular)
    j = np.empty_like(c)
    for cc in range(m):
        sel = c == cc
        j[sel] = rng.choice(m, size=int(sel.sum()), p=P[cc])
    return lo + (j + 0.5) * cell


def gsq_quantise(phi, psi, eps: float, rng):
    """The He codebook of joint_defense_common.he_quantise, cells drawn by
    the full-support mechanism."""
    return (gsq_quantise_1d(phi, C.PHI_LO, C.PHI_HI, C.HE_B_PHI, eps, rng, True),
            gsq_quantise_1d(psi, C.PSI_LO, C.PSI_HI, C.HE_B_PSI, eps, rng, False))


def gsq_name(eps: float) -> str:
    return f"dpgsq_e{eps:g}"


def gsq_gains(seeds: int) -> dict:
    """he_beamforming_gain's measurement: the same HG.N_REPORTS reports per
    seed (manifold_sweep.gain_rows' draw), truth = the continuous V."""
    d = C.load_he_angles()
    phi_all, psi_all = d["phi"], d["psi"]
    n_all = len(d["y"])
    del d
    out = defaultdict(list)
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng([seed, MS.GAIN_STREAM])
        idx = np.sort(rng.choice(n_all, size=HG.N_REPORTS, replace=False))
        phi, psi = phi_all[idx], psi_all[idx]
        V = C.givens_to_v(phi, psi)
        out[REF_NAME].append(float(HG.gains(quantised_v(phi, psi), V)[0].mean()))
        for i, e in enumerate(GSQ_EPS):
            a, b = gsq_quantise(phi, psi, e, np.random.default_rng(
                [seed, MS.GAIN_STREAM, GSQ_STREAM_OFFSET + i]))
            out[gsq_name(e)].append(float(HG.gains(C.givens_to_v(a, b), V)[0].mean()))
        del V
    del phi_all, psi_all
    gc.collect()
    return out


def part3(ds: J.He, seeds: int, gains: dict) -> tuple[dict, list[dict]]:
    y = ds.y["held"]
    chance = 1.0 / len(np.unique(y))
    cells = [(REF_NAME, None, J.HE_DEFENSES.index(REF_NAME))] + [
        (gsq_name(e), e, GSQ_STREAM_OFFSET + i) for i, e in enumerate(GSQ_EPS)]
    acc = defaultdict(lambda: defaultdict(list))
    for seed in range(1, seeds + 1):
        supports = C.pick_supports(y, seed)
        for name, e, did in cells:
            rng = np.random.default_rng([seed, did, RD.HELD_STREAM])
            pq, sq = C.he_quantise(ds.phi["held"], ds.psi["held"]) if e is None \
                else gsq_quantise(ds.phi["held"], ds.psi["held"], e, rng)
            F = C.he_features(pq, sq)
            for obs, Z in (("raw", F), ("scaled", ds.S0.transform(F))):
                r = C.multi_query_eval(
                    Z, y, supports, metric="cos", T_grid=J.T_GRID,
                    n_decisions=J.N_DECISIONS_PER_CLASS,
                    rng=np.random.default_rng([seed, did, RD.NN1_ATTACKER_ID,
                                               RD.QUERY_STREAM]))
                for T, a in r.items():
                    acc[(name, obs)][T].append(a)
            del pq, sq, F
        gc.collect()
        log(f"3 seed {seed}")
    out, rows = {}, []
    for name, e, _did in cells:
        c = dict(eps=e, gain_col1=stat(gains[name]))
        for obs in ("raw", "scaled"):
            for T in RD.T_REPORT:
                c[f"{obs}_T={T}"] = stat(acc[(name, obs)][T])
        worst = max(c[f"{obs}_T={T}"]["mean"] for obs in ("raw", "scaled")
                    for T in RD.T_REPORT)
        c["worst_observer_accuracy"] = worst
        c["within_chance_margin"] = bool(worst <= chance + RS.PASS_MARGIN)
        for floor in GSQ_GAIN_FLOORS:
            c[f"passes_gain_{floor:g}"] = bool(
                c["within_chance_margin"] and c["gain_col1"]["mean"] >= floor)
        out[name] = c
        for obs in ("raw", "scaled"):
            for T in RD.T_REPORT:
                st = c[f"{obs}_T={T}"]
                rows.append(row("3", "dp_gsq_full_support_variant", name, "accuracy",
                                st["mean"], observer=obs, K=1, T=T, chance=chance,
                                n=st["n"], note=f"sd over seeds {st['sd']:.4f}"))
        rows.append(row("3", "dp_gsq_full_support_variant", name, "gain_col1",
                        c["gain_col1"]["mean"], n=c["gain_col1"]["n"],
                        note=f"sd over seeds {c['gain_col1']['sd']:.4f}"))
    # the deterministic row against the stored values
    checks = {}
    if RD.OUT_JSON.exists() and seeds == RD.N_SEEDS:
        st = {(r["defense"], r["T"]): r for r in json.loads(RD.OUT_JSON.read_text())["rows"]}
        checks["none_vs_r_raw_observer_defenses_max_abs_diff"] = max(
            abs(out[REF_NAME][f"{obs}_T={T}"]["mean"] - st[(REF_NAME, T)][key])
            for T in RD.T_REPORT
            for obs, key in (("raw", "raw_mean"), ("scaled", "scaled_rerun_mean")))
    if RD.GAIN_JSON.exists() and seeds == HG.N_SEEDS:
        g = json.loads(RD.GAIN_JSON.read_text())["cells"][f"{REF_NAME}|{RD.GAIN_UNKEYED}"]
        checks["none_gain_vs_he_beamforming_gain_abs_diff"] = abs(
            out[REF_NAME]["gain_col1"]["mean"] - g["gain_col1"])
    return dict(
        label=GSQ_LABEL,
        mechanism="P(report cell j | true cell c) proportional to "
                  "exp(-eps d(c, j) / (2 Delta)); phi: 16 cells, circular "
                  "distance, Delta = 8; psi: 4 cells, linear distance, Delta = 3; "
                  "eps-local-DP per angle (12 angles x 59 groups per report "
                  "compose)",
        protocol="He CR1, 19 held-out positions, cosine 1-NN, K = 1, raw and "
                 f"scaled features, {seeds} seeds; gain on {HG.N_REPORTS} reports "
                 "per seed against the continuous V",
        chance=chance, margin=RS.PASS_MARGIN, gain_floors=list(GSQ_GAIN_FLOORS),
        eps=list(GSQ_EPS), cells=out, checks=checks,
        passing={f"gain>={f:g}": [n for n, c in out.items() if c["eps"] is not None
                                  and c[f"passes_gain_{f:g}"]]
                 for f in GSQ_GAIN_FLOORS}), rows


# ---------------------------------------------------------------------------
def dry_run(devs: list[str]) -> dict:
    est = R.dry_run(devs)
    w = R.windows_path(R.SESSION_GAP_S)
    return dict(
        parts=list(PARTS),
        archive_rows_per_day={k: v for k, v in est.items() if not k.startswith("_")},
        part1a="MIMO Control head per report only (~60 bytes per row held)",
        part2=dict(r_common_windows_cached=w.exists(),
                   rotated_windows_cached=RR.windows_path(BODY, R.SESSION_GAP_S).exists(),
                   est_table_mb=(w.stat().st_size * K.EST_TABLE_COPIES / 1e6
                                 if w.exists() else None)),
        he=dict(file_mb=C.he_estimate_mb(),
                est_working_set_mb=C.he_estimate_mb() * RN.EST_HE_LOAD_FACTOR,
                seeds=HE_SEEDS, partners=len(HE_PARTNER_CLASS_INDICES),
                gsq_eps=list(GSQ_EPS)),
        rss_budget_mb=RSS_BUDGET_MB, vm_limit_mb=X.VM_LIMIT_BYTES / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parts", nargs="*", default=list(PARTS), choices=PARTS)
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--he-seeds", type=int, default=HE_SEEDS)
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    if a.dry_run:
        print(json.dumps(dry_run(devs), indent=2))
        return 0
    out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    rows = [r for r in out.pop("csv_rows", [])
            if r["part"][0] not in a.parts]
    timing = out.get("runtime_s_per_part", {})

    def keep(key: str, res: tuple, t: float) -> None:
        out[key] = res[0]
        rows.extend(res[1])
        timing[key] = time.time() - t

    if "1" in a.parts:
        t = time.time()
        keep("part1a_home_shared_soundings", part1a(devs), t)
    if "2" in a.parts:
        t = time.time()
        keep("part2_combined_defense", part2(devs, a.trials), t)
        gc.collect()
    if "1" in a.parts or "3" in a.parts:
        gains = gsq_gains(a.he_seeds) if "3" in a.parts else None
        ds = J.He(J.HE_CAL_PER_CLASS)
        log("He loaded")
        if "1" in a.parts:
            t = time.time()
            keep("part1b_he_shared_sounding", part1b(ds, a.he_seeds), t)
        if "3" in a.parts:
            t = time.time()
            keep("part3_dp_gsq", part3(ds, a.he_seeds, gains), t)
        del ds
    out.update(runtime_s_per_part=timing, last_run=dict(
        parts=a.parts, runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb(),
        trials=a.trials, he_seeds=a.he_seeds), csv_rows=rows)
    X.write_json(OUT_JSON, out)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        wr.writeheader()
        wr.writerows(rows)
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB; "
          f"wrote {OUT_JSON.name}, {OUT_CSV.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
