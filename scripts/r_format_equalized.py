#!/usr/bin/env python3
"""R-format: what the access point can equalize in the announced format,
as a policy it can enforce.

The format signature of a report is the tuple r_common keys windows by:
(generation, Nc, Nr, phi bits, psi bits, bandwidth, Ng, feedback type,
codebook), parsed by the decoder from the MIMO Control field of every body
(no column of the raw archive is needed).

Who sets which field (reading of the 802.11ax comment-resolution records
11-19/0099 and 11-19/1137 and of the VHT NDPA STA Info field; see
STANDARD_READING below, to be cited with care, not from the standard's
final text): for SU feedback the beamformee chooses Ng, codebook and Nc in
both generations (the HE NDPA sets those subfields to 0 in a non-TB
sounding that solicits SU feedback). The beamformer dictates them only when
it solicits MU feedback: in VHT the NDPA STA Info carries Feedback Type and
Nc Index, and MU feedback uses the MU codebook class with the beamformee's
codebook index and grouping; in HE, MU feedback (and trigger-based
sounding) lets the beamformer set Feedback Type, Ng, Codebook Size and Nc.
The generation (HE or VHT), Nr (the NDP's space-time streams, equal for all
clients of one access point) and, for an SU-only client, everything, stay
outside the beamformer's reach.

Policies
  P1           as captured
  P2_observed  the AP solicits MU feedback from every client that is seen
               returning MU feedback in the capture; the others are treated
               as SU-only and keep their format
  P2_all       the AP solicits MU feedback from all six (the case in which
               the devices never asked for MU feedback support it)
Under P2 a MU-capable client's every report is MU: VHT (MU, Nc from the AP,
client's codebook index and Ng, MU bit widths); HE (MU, Nc, Ng, codebook
size from the AP). MU support is not announced in a beamforming report, so
the capture shows it only for the devices that returned MU feedback; for
the others it is unknown, hence the two P2 cases.

Evaluation: r_eval's protocol, grids and seeds on the format feature alone
and on the gated angles+format observer, for the unprotected body and the
rotated body of r_rotation.py. Behind an equalized gate the honest body
readout is the format-blind grid (feat_rq, every device at 4-bit phi /
2-bit psi), reported next to the native-bit one.

Inputs: window tables of r_rotation.py (BFI_X_CACHE_DIR/r_rot/<variant>/,
built if absent), r_common per-day cache (per-frame classes).
Outputs (figures/): r_format_equalized.json, r_format_equalized.csv.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_common as R  # noqa: E402
import r_eval as E  # noqa: E402
import r_rotation as RR  # noqa: E402
import xres_common as X  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIELDS = R.GEOM_FIELDS      # fmt, nc, nr, bphi, bpsi, bw_mhz, ng, feedback_type, codebook
POLICIES = ("P1", "P2_observed", "P2_all")
POLICY_NOTES = {
    "P1": "as captured",
    "P2_observed": "AP solicits MU feedback from the clients seen returning "
                   "MU feedback in the capture; the others keep their format",
    "P2_all": "AP solicits MU feedback from all six devices",
}
# Fields the beamformer sets when it solicits MU feedback, per generation.
MU_AP_FIELDS = {"HE": ("feedback_type", "nc", "ng", "codebook"),
                "VHT": ("feedback_type", "nc")}
# Fields the beamformee keeps under MU feedback.
MU_CLIENT_FIELDS = {"HE": ("fmt", "nr", "bw_mhz"),
                    "VHT": ("fmt", "nr", "bw_mhz", "codebook", "ng")}
DERIVED_FIELDS = ("bphi", "bpsi")        # from (feedback_type, codebook)
MU_FEEDBACK_TYPE = "MU"
EQ_NC = 1
EQ_HE_NG = 4
EQ_HE_CODEBOOK = 0                       # HE MU codebook size 0 = 7-bit phi, 5-bit psi
MU_VALUES = {"feedback_type": MU_FEEDBACK_TYPE, "nc": EQ_NC,
             "ng": EQ_HE_NG, "codebook": EQ_HE_CODEBOOK}
STANDARD_READING = (
    "Field attribution follows the 802.11ax comment-resolution records "
    "11-19/0099 and 11-19/1137 (for SU feedback in a non-TB sounding the "
    "beamformer sets Feedback Type And Ng, Codebook Size and Nc to 0 and the "
    "beamformee chooses them; the beamformer dictates them for MU feedback "
    "and in TB sounding) and the VHT NDPA STA Info field (Feedback Type, "
    "Nc Index; codebook and grouping are the beamformee's, MU feedback uses "
    "the MU codebook class). Not checked against the standard's final text.")
BODIES = ("none", "rot_group")
HEADLINE_K = RR.HEADLINE_K
HEADLINE_T = RR.HEADLINE_T
OUT = dict(json=R.FIG_DIR / "r_format_equalized.json",
           csv=R.FIG_DIR / "r_format_equalized.csv")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def apply_policy(geom: tuple, mu_capable: bool, policy: str) -> tuple:
    if policy == "P1" or not mu_capable:
        return geom
    g = dict(zip(FIELDS, geom))
    for f in MU_AP_FIELDS[g["fmt"]]:
        g[f] = MU_VALUES[f]
    g["bpsi"], g["bphi"] = X.AD.ANGLE_BITS[(g["feedback_type"], g["codebook"])]
    return tuple(g[f] for f in FIELDS)


def mu_sets(cnt: dict, names: list[str]) -> dict:
    observed = sorted(d for d, c in cnt.items()
                      if any(dict(zip(FIELDS, g))["feedback_type"] == "MU"
                             for g in c))
    return {"P1": set(), "P2_observed": set(observed), "P2_all": set(names)}


def policy_table(w: dict, names: list[str], mu: set, policy: str) -> dict:
    """Window table with geometry ids remapped per (device, geometry)."""
    today = [tuple(json.loads(s)) for s in w["geom_table"]]
    classes: dict[tuple, int] = {}
    lut = np.empty((len(names), len(today)), dtype=np.int64)
    for di, name in enumerate(names):
        for gi, g in enumerate(today):
            lut[di, gi] = classes.setdefault(
                apply_policy(g, name in mu, policy), len(classes))
    out = dict(w)
    out["geom"] = lut[w["dev"], w["geom"]]
    out["geom_table"] = np.asarray([json.dumps(list(k)) for k in classes])
    return out


def frame_classes(devs: list[str]) -> dict:
    """Per device: frames per format class as captured, streamed from the
    per-day cache of the 'none' variant."""
    cnt: dict[str, Counter] = {R.short(m): Counter() for m in devs}
    for p in X.day_paths():
        d = RR.load_day("none", p.stem)
        if "ts" not in d or not d["ts"].size:
            continue
        for i, m in enumerate(devs):
            sel = d["dev"] == i
            for gi, n in zip(*np.unique(d["gid"][sel], return_counts=True)):
                cnt[R.short(m)][d["geoms"][int(gi)]] += int(n)
        del d
        gc.collect()
    return cnt


def mu_evidence(cnt: dict) -> dict:
    out = {}
    for dev, c in cnt.items():
        tot = sum(c.values())
        mu = sum(n for g, n in c.items()
                 if dict(zip(FIELDS, g))["feedback_type"] == "MU")
        out[dev] = dict(frames=tot, mu_frames=mu, mu_fraction=mu / tot,
                        mu_support=("observed" if mu else "unknown"))
    return out


def class_report(cnt: dict, mu: set, policy: str) -> dict:
    per, all_classes = {}, set()
    for dev, c in cnt.items():
        eq = Counter()
        for g, n in c.items():
            eq[apply_policy(g, dev in mu, policy)] += n
        all_classes |= set(eq)
        per[dev] = dict(
            classes={json.dumps(list(g)): n for g, n in eq.most_common()},
            n_classes=len(eq),
            dominant=json.dumps(list(eq.most_common(1)[0][0])))
    return dict(policy=policy, mu_capable=sorted(mu), per_device=per,
                n_classes_all=len(all_classes),
                classes_all=sorted(json.dumps(list(g)) for g in all_classes))


def separable_pairs(cnt: dict, mu: set, policy: str) -> list[dict]:
    """Device pairs whose class sets are disjoint under the policy, and the
    fields whose values never coincide across the pair."""
    sets = {dev: {apply_policy(g, dev in mu, policy) for g in c}
            for dev, c in cnt.items()}
    out = []
    for a, b in itertools.combinations(sorted(sets), 2):
        fields = [f for j, f in enumerate(FIELDS)
                  if not ({g[j] for g in sets[a]} & {g[j] for g in sets[b]})]
        out.append(dict(pair=[a, b],
                        separable_by_format=not (sets[a] & sets[b]),
                        separating_fields=fields))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rows_for(w: dict, body: str, policy: str, trials: int, rng_b) -> list:
    """Format-only, angles (ungated), angles+format (native bits) and
    angles_rq+format (format-blind grid behind the gate)."""
    main, _ = RR.evaluate(w, dict(body=body, policy=policy,
                                  observer_input="angles"), trials, rng_b)
    keep = [r for r in main if r["feature"] in ("format", "angles",
                                                 "angles+format")]
    m2, _ = RR.evaluate(dict(w, feat=w["feat_rq"]),
                        dict(body=body, policy=policy, observer_input="rq"),
                        trials, rng_b)
    for r in m2:
        if r["feature"] == "angles+format":
            r["feature"] = "angles_rq+format"
            keep.append(r)
    for r in keep:
        r.pop("observer_input", None)
    return keep


def headline(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        if r["K"] in HEADLINE_K and r["T"] in HEADLINE_T:
            out[f"{r['policy']}|{r['body']}|{r['feature']}|{r['condition']}"
                f"|K={r['K']}|T={r['T']}"] = dict(
                accuracy=round(r["accuracy"], 4),
                macro_recall=round(r["macro_recall"], 4),
                chance=round(r["chance"], 4),
                ci=[round(r["ci_lo"], 4), round(r["ci_hi"], 4)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--bodies", nargs="*", default=list(BODIES))
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--append", action="store_true",
                    help="add the rows of these bodies and policies to the "
                         "existing outputs instead of rewriting them")
    a = ap.parse_args()
    X.set_vm_limit()
    t0 = time.time()
    devs = R.device_macs()
    names = [R.short(m) for m in devs]
    for p in X.day_paths():
        R.build_day(p, devs)
    RR.get_windows(devs, R.SESSION_GAP_S, "none")
    cnt = frame_classes(devs)
    mu = mu_sets(cnt, names)
    classes = {pol: class_report(cnt, mu[pol], pol) for pol in POLICIES}
    pairs = {pol: separable_pairs(cnt, mu[pol], pol) for pol in POLICIES}
    for pol in POLICIES:
        print(f"[classes] {pol}: {classes[pol]['n_classes_all']} classes, "
              f"MU-capable {sorted(mu[pol])}, separable pairs "
              f"{sum(p['separable_by_format'] for p in pairs[pol])}/15",
              file=sys.stderr)
    X.log_rss("classes")

    ref = RR.reference_rows()
    rng_b = np.random.default_rng(E.SEED + 1)
    rows = []
    for body in a.bodies:
        w = RR.get_windows(devs, R.SESSION_GAP_S, body)
        for pol in a.policies:
            if pol == "P1" and body != "none":
                continue                 # P1 rows of rotated bodies: r_rotation
            rows += rows_for(policy_table(w, names, mu[pol], pol), body, pol,
                             a.trials, rng_b)
        del w
        gc.collect()
        X.log_rss(body)
    for r in rows:
        r["unprotected_r_main"] = ref.get(
            (r["condition"], r["feature"], r["K"], r["T"]))
    if a.append:
        old = json.loads(OUT["json"].read_text())
        new_keys = {(r["body"], r["policy"]) for r in rows}
        kept = [r for r in RR.read_csv_rows(OUT["csv"])
                if (r["body"], r["policy"]) not in new_keys]
        E.write_csv(OUT["csv"], kept + rows)
        old["headline"] = {k: v for k, v in old["headline"].items()
                           if tuple(k.split("|")[:2]) not in
                           {(p, b) for b, p in new_keys}}
        old["headline"].update(headline(rows))
        old.setdefault("appended_runs", []).append(dict(
            bodies=a.bodies, policies=a.policies,
            runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
        X.write_json(OUT["json"], old)
        print(f"[done] appended {time.time() - t0:.0f}s peak RSS "
              f"{X.peak_rss_mb():.0f} MB", file=sys.stderr)
        return 0
    E.write_csv(OUT["csv"], rows)
    p1_vs_ref = [abs(r["accuracy"] - r["unprotected_r_main"])
                 for r in rows if r["policy"] == "P1" and r["body"] == "none"
                 and r["unprotected_r_main"] is not None]
    X.write_json(OUT["json"], dict(
        standard_reading=STANDARD_READING,
        fields=list(FIELDS), mu_ap_fields=MU_AP_FIELDS,
        mu_client_fields=MU_CLIENT_FIELDS, derived_fields=list(DERIVED_FIELDS),
        mu_values=MU_VALUES, policies=POLICY_NOTES,
        mu_evidence=mu_evidence(cnt), classes=classes, pairs=pairs,
        pairs_still_separable={pol: [p["pair"] for p in pairs[pol]
                                     if p["separable_by_format"]]
                               for pol in POLICIES},
        trials=a.trials, k_grid=[E.klabel(k) for k in E.K_GRID],
        t_grid=list(E.T_GRID), session_gap_s=R.SESSION_GAP_S,
        p1_vs_r_main_max_abs_diff=max(p1_vs_ref) if p1_vs_ref else None,
        headline=headline(rows),
        runtime_s=time.time() - t0, peak_rss_mb=X.peak_rss_mb()))
    print(f"[done] {time.time() - t0:.0f}s peak RSS {X.peak_rss_mb():.0f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
