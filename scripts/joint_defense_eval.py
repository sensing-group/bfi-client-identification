"""X1: do defenses that disrupt the JOINT pattern stop the cohort-out attacker?

Defenses x attackers x multi-query T on NTU-Fi HumanID (b=4 amplitude proxy)
or He CR1 real Givens-angle BFI (codebook b_phi=4 / b_psi=2, the paper's He "b=4").

Attackers (all K=1, one labelled support window per target, cohort-out):
    nn1        1-NN, cosine, features standardised with CLEAN calibration stats
    supcon     SupCon MLP (256 hidden, 128 out) trained on clean calibration data
    supcon_da  same encoder trained on calibration data passed through the SAME
               defense (R independent realisations; fixed-Q variants redraw the
               per-subject Q in every realisation)
    nn1_inv    1-NN on features the attacker can compute that are INVARIANT to the
               defense's transform (only for orthogonal/unitary defenses, and for
               'none' as a reference): antenna-axis norms (NTU-Fi per-packet Q),
               cross-subcarrier |V_k^H V_0| (He per-packet Q), vector norm
               (per-window feature Q)
Multi-query: average T in {1, 8, 32} query representations, cosine vs one support.

Usage:
    python3 scripts/joint_defense_eval.py --dataset ntufi --seeds 10
    python3 scripts/joint_defense_eval.py --dataset he --seeds 10
    python3 scripts/joint_defense_eval.py --dataset he --dry-run
"""
from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

import joint_defense_common as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T_GRID = (1, 8, 32)
N_DECISIONS_PER_CLASS = 200
DEFAULT_SEEDS = 10
DA_REALISATIONS_NTUFI = 8
DA_REALISATIONS_HE = 1
HE_SUPCON_EPOCHS = 100          # He cal set is ~40x NTU-Fi; 100 epochs x 94 batches ~ 9.4k steps
HE_CAL_PER_CLASS = 300          # SupCon training subsample per calibration class (He)
HE_CHUNK = 3000                 # reports per chunk for V reconstruction
MANIFOLD_EPS = 0.5
DPSQ_EPS = (0.01, 0.5, 2.0)

NTUFI_DEFENSES = (
    ["none"] + [f"dpsq_e{e}" for e in DPSQ_EPS]
    + ["unitary_pkt", "unitary_fixed", "unitary_global", "orthq_window", "orthq_fixed", "orthq_global",
       "manifold_e0.5"]
)
HE_DEFENSES = (
    ["none"] + [f"dpsq_e{e}" for e in DPSQ_EPS] + ["dpsqstd_e0.5"]
    + ["unitary_pkt", "unitary_pkt_sc", "unitary_fixed", "unitary_global", "orthq_window", "orthq_fixed",
       "orthq_global", "manifold_e0.5"]
)

DEFENSE_NOTES = {
    "none": "deterministic b=4 quantisation only",
    "dpsq": "DP-SQ stochastic rounding, b=4, fresh per window/report, no key",
    "dpsqstd": "paper-code DP-SQ: applied in standardised feature space, linear, b=4 (He only)",
    "unitary_pkt": "fresh Haar Q per packet/report on the antenna axis, shared across subcarriers; AP-only secret",
    "unitary_pkt_sc": "fresh Haar Q per report AND per subcarrier (He only); AP-only secret",
    "unitary_fixed": "one Haar Q per target (device/session) for all its packets; AP-only secret",
    "orthq_window": "fresh orthogonal Q per window on the whole feature vector (existing 'structural'/'unitary_obfusc'); inversion needs Q at AP -> shared AP-client secret",
    "orthq_fixed": "one orthogonal Q per target on the feature vector; shared secret",
    "unitary_global": "one Haar Q for ALL targets of the deployment (per-AP fixed); AP-only secret",
    "orthq_global": "one orthogonal Q for ALL targets on the feature vector; shared secret",
    "manifold": "Gaussian noise sigma = range*sqrt(2 ln(1.25/delta))/eps, delta=1e-5, then b=4 quantisation; no key",
}


def _groups(defense: str, y: np.ndarray) -> np.ndarray:
    """Who shares a fixed Q: each target ('_fixed') or the whole deployment ('_global')."""
    return np.zeros_like(y) if defense.endswith("_global") else y


def _eps_of(defense: str) -> float | None:
    return float(defense.split("_e")[1]) if "_e" in defense else None


# ---------------------------------------------------------------------------
# NTU-Fi substrate
# ---------------------------------------------------------------------------

class NTUFi:
    name = "ntufi"
    chance_desc = "7 held-out subjects"

    def __init__(self):
        d = C.load_ntufi_packets()
        A, y, split = d["A"], d["y"], d["split"]
        cal = (y % 2 == 0) & np.isin(split, C.NTUFI_CAL_SPLITS)
        held = y % 2 == 1
        # Quantisation range: min/max over all windows (cal train+test and held), as the headline script.
        Fall = C.ntufi_features(A)
        self.lo, self.hi = float(Fall.min()), float(Fall.max())
        self.A = {"cal": A[cal], "held": A[held]}
        self.y = {"cal": y[cal], "held": y[held]}
        F = {"cal": Fall[cal], "held": Fall[held]}
        del d, A, Fall
        gc.collect()
        self.F = F
        self.S0 = StandardScaler().fit(C.quant_midpoint(F["cal"], self.lo, self.hi, C.B_HEADLINE))
        cal_s = self.S0.transform(C.quant_midpoint(F["cal"], self.lo, self.hi, C.B_HEADLINE))
        self.std_lo, self.std_hi = float(cal_s.min()), float(cal_s.max())
        self.cal_clean_s = cal_s
        self.S0_inv = StandardScaler().fit(C.ntufi_norm_invariant(self.A["cal"]))

    def observe(self, which: str, defense: str, rng):
        """Return (observed S0-standardised features, invariant features or None, inv metric)."""
        A, y = self.A[which], self.y[which]
        eps = _eps_of(defense)
        inv, inv_metric = None, None
        if defense.startswith("unitary"):
            if defense == "unitary_pkt":
                Q = C.haar_orthogonal(C.NTUFI_RX, (A.shape[0], A.shape[-1]), rng)  # (N, P, 3, 3)
            else:
                groups = _groups(defense, y)
                per = {int(g): C.haar_orthogonal(C.NTUFI_RX, (), rng) for g in np.unique(groups)}
                Q = np.stack([np.broadcast_to(per[int(g)], (A.shape[-1], 3, 3)) for g in groups])
            A2 = np.einsum("npij,njkp->nikp", Q, A.astype(np.float64))
            del Q
            F = C.ntufi_features(A2)
            # amplitude proxy has no codebook range: quantise over the defended range
            Fq = C.quant_midpoint(F, float(F.min()), float(F.max()), C.B_HEADLINE)
            inv, inv_metric = self.S0_inv.transform(C.ntufi_norm_invariant(A2)), "cos"
            del A2
        else:
            F = self.F[which]
            if defense.startswith("manifold"):
                Fq = C.manifold_dp(F, self.lo, self.hi, eps, rng)
            else:
                Fq = C.quant_midpoint(F, self.lo, self.hi, C.B_HEADLINE)
            if defense.startswith("orthq"):
                Fq = C.orth_q_features(Fq, rng, mode="fresh" if defense == "orthq_window" else "fixed",
                                        groups=_groups(defense, y))
                inv, inv_metric = np.linalg.norm(Fq, axis=1, keepdims=True), "euc"
            if defense == "none":
                inv, inv_metric = self.S0_inv.transform(C.ntufi_norm_invariant(A)), "cos"
        Xs = self.S0.transform(Fq)
        if defense.startswith("dpsq_"):
            Xs = C.dpsq_std(Xs, self.std_lo, self.std_hi, eps, rng)
        return Xs, inv, inv_metric

    def train_cal(self):
        return self.cal_clean_s, self.y["cal"]


# ---------------------------------------------------------------------------
# He CR1 substrate
# ---------------------------------------------------------------------------

class He:
    name = "he"
    chance_desc = "19 held-out location classes"

    def __init__(self, cal_per_class: int):
        d = C.load_he_angles()
        self.raw_shape = d["raw_shape"]
        y = d["y"]
        classes = np.unique(y)
        cal_classes = classes[: C.HE_N_CALIB_CLASSES]
        cal = np.isin(y, cal_classes)
        held = ~cal
        rng = np.random.default_rng(0)
        sub_idx = np.concatenate([
            rng.choice(np.flatnonzero(y == c), size=min(cal_per_class, int((y == c).sum())), replace=False)
            for c in cal_classes])
        self.phi = {"cal": d["phi"][cal], "held": d["phi"][held], "cal_sub": d["phi"][sub_idx]}
        self.psi = {"cal": d["psi"][cal], "held": d["psi"][held], "cal_sub": d["psi"][sub_idx]}
        self.y = {"cal": y[cal], "held": y[held], "cal_sub": y[sub_idx]}
        del d
        gc.collect()
        pq, sq = C.he_quantise(self.phi["cal"], self.psi["cal"])
        Fcal = C.he_features(pq, sq)
        self.S0 = StandardScaler().fit(Fcal)
        cal_s = self.S0.transform(Fcal)
        self.std_lo, self.std_hi = float(cal_s.min()), float(cal_s.max())
        self.S0_inv = StandardScaler().fit(self._gram_invariant(pq, sq))
        pqs, sqs = C.he_quantise(self.phi["cal_sub"], self.psi["cal_sub"])
        self.cal_sub_clean_s = self.S0.transform(C.he_features(pqs, sqs))

    @staticmethod
    def _gram_invariant(phi_q, psi_q):
        """|diag(V_k^H V_0)| for k=1..58, 3 columns -> (N, 174); invariant to a Q shared across subcarriers."""
        out = []
        for s in range(0, phi_q.shape[0], HE_CHUNK):
            V = C.givens_to_v(phi_q[s:s + HE_CHUNK], psi_q[s:s + HE_CHUNK])  # (n, 59, 4, 3)
            g = np.abs(np.einsum("nkrc,nrc->nkc", np.conj(V[:, 1:]), V[:, 0]))
            out.append(g.reshape(g.shape[0], -1))
            del V
        return np.concatenate(out)

    def _unitary(self, phi, psi, y, defense, rng):
        out_p, out_s = np.empty_like(phi), np.empty_like(psi)
        fixed = None
        groups = _groups(defense, y)
        if defense in ("unitary_fixed", "unitary_global"):
            fixed = {int(g): C.haar_unitary(C.HE_NR, (), rng) for g in np.unique(groups)}
        for s in range(0, phi.shape[0], HE_CHUNK):
            sl = slice(s, s + HE_CHUNK)
            V = C.givens_to_v(phi[sl], psi[sl])                 # (n, 59, 4, 3)
            n, k = V.shape[:2]
            if defense == "unitary_pkt":
                Q = C.haar_unitary(C.HE_NR, (n,), rng)[:, None]  # shared across subcarriers
            elif defense == "unitary_pkt_sc":
                Q = C.haar_unitary(C.HE_NR, (n, k), rng)
            else:
                Q = np.stack([fixed[int(g)] for g in groups[sl]])[:, None]
            V2 = np.conj(np.swapaxes(Q, -1, -2)) @ V             # V' = Q^H V
            out_p[sl], out_s[sl] = C.v_to_givens(V2)
            del V, V2, Q
        return out_p, out_s

    def observe(self, which: str, defense: str, rng):
        phi, psi, y = self.phi[which], self.psi[which], self.y[which]
        eps = _eps_of(defense)
        inv, inv_metric = None, None
        if defense.startswith("unitary"):
            phi, psi = self._unitary(phi, psi, y, defense, rng)
        if defense.startswith("dpsq_"):
            pq, sq = C.he_quantise(phi, psi, eps=eps, rng=rng)
        else:
            pq, sq = C.he_quantise(phi, psi)
        if defense.startswith("manifold"):
            F = C.manifold_dp(C.he_features(phi, psi), C.PHI_LO, C.PHI_HI, eps, rng)
        else:
            F = C.he_features(pq, sq)
        if defense.startswith("orthq"):
            F = C.orth_q_features(F, rng, mode="fresh" if defense == "orthq_window" else "fixed",
                                        groups=_groups(defense, y))
            inv, inv_metric = np.linalg.norm(F, axis=1, keepdims=True), "euc"
        if defense == "none" or defense.startswith("unitary"):
            inv, inv_metric = self.S0_inv.transform(self._gram_invariant(pq, sq)), "cos"
        Xs = self.S0.transform(F)
        if defense.startswith("dpsqstd"):
            Xs = C.dpsq_std(Xs, self.std_lo, self.std_hi, eps, rng)
        return Xs, inv, inv_metric

    def train_cal(self):
        return self.cal_sub_clean_s, self.y["cal_sub"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    timer = C.Timer()
    C.set_vm_limit()
    C.check_rss("start")
    if args.dataset == "ntufi":
        ds, defenses, R, cal_key = NTUFi(), NTUFI_DEFENSES, DA_REALISATIONS_NTUFI, "cal"
    else:
        ds, defenses, R, cal_key = He(args.he_cal_per_class), HE_DEFENSES, DA_REALISATIONS_HE, "cal_sub"
    if args.defenses:
        defenses = [d for d in defenses if d in args.defenses]
    C.check_rss("data loaded")
    y_held = ds.y["held"]
    n_cls = len(np.unique(y_held))
    chance = 1.0 / n_cls
    print(f"{ds.name}: held {len(y_held)} windows, {n_cls} classes, chance {chance:.4f}; "
          f"cal train rows {len(ds.train_cal()[1])}", flush=True)

    rows = []
    out_csv = C.FIG_DIR / f"bfi_joint_defense_{ds.name}.csv"
    for seed in range(1, args.seeds + 1):
        supports = C.pick_supports(y_held, seed)
        Xc, yc = ds.train_cal()
        enc = C.train_supcon(Xc, yc, seed, epochs=args.epochs)
        for did, defense in enumerate(defenses):
            t0 = timer()
            rng_h = np.random.default_rng([seed, did, 1])
            Xh, inv, inv_metric = ds.observe("held", defense, rng_h)
            reps = {"nn1": (Xh, "cos"), "supcon": (enc.embed(Xh), "cos")}
            if inv is not None:
                reps["nn1_inv"] = (inv, inv_metric)
            if defense == "none":
                reps["supcon_da"] = reps["supcon"]
            else:
                rng_c = np.random.default_rng([seed, did, 2])
                parts = [ds.observe(cal_key, defense, rng_c)[0] for _ in range(R)]
                Xda = np.concatenate(parts)
                yda = np.tile(ds.y[cal_key], R)
                Sd = StandardScaler().fit(Xda)
                enc_da = C.train_supcon(Sd.transform(Xda), yda, seed, epochs=args.epochs)
                reps["supcon_da"] = (enc_da.embed(Sd.transform(Xh)), "cos")
                del parts, Xda, enc_da
            for aid, (att, (Z, metric)) in enumerate(reps.items()):
                rng_q = np.random.default_rng([seed, did, aid, 3])
                accs = C.multi_query_eval(Z, y_held, supports, metric=metric, T_grid=T_GRID,
                                          n_decisions=N_DECISIONS_PER_CLASS, rng=rng_q)
                for T, acc in accs.items():
                    rows.append(dict(dataset=ds.name, defense=defense, attacker=att, seed=seed, T=T,
                                     acc=acc, chance=chance, n_classes=n_cls))
            msg = " ".join(f"{a}={r['acc']:.3f}" for a in reps for r in rows[-len(reps) * len(T_GRID):]
                           if r["attacker"] == a and r["T"] == 1)
            print(f"  seed {seed} {defense:>14s} T=1 {msg}  ({timer() - t0:.1f}s)", flush=True)
            del Xh, inv, reps
            gc.collect()
        C.check_rss(f"after seed {seed}")
        _write(rows, out_csv)
    summary = _summarise(rows, ds, defenses, args, timer())
    (C.FIG_DIR / f"bfi_joint_defense_{ds.name}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_csv} and summary json; runtime {timer():.0f}s peak RSS {C.peak_rss_mb():.0f} MB")


def _write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarise(rows, ds, defenses, args, runtime):
    cells = {}
    for d in defenses:
        for a in ("nn1", "supcon", "supcon_da", "nn1_inv"):
            for T in T_GRID:
                v = [r["acc"] for r in rows if r["defense"] == d and r["attacker"] == a and r["T"] == T]
                if v:
                    cells[f"{d}|{a}|T={T}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))
    return dict(dataset=ds.name, chance=rows[0]["chance"], n_classes=rows[0]["n_classes"], seeds=args.seeds,
                epochs=args.epochs, da_realisations=DA_REALISATIONS_NTUFI if ds.name == "ntufi" else DA_REALISATIONS_HE,
                T_grid=list(T_GRID), n_decisions_per_class=N_DECISIONS_PER_CLASS,
                defense_notes=DEFENSE_NOTES, manifold_sigma_over_range=C.manifold_sigma_over_range(MANIFOLD_EPS),
                cells=cells, runtime_s=runtime, peak_rss_mb=C.peak_rss_mb(),
                extra=dict(he_raw_shape=getattr(ds, "raw_shape", None),
                           he_cal_per_class=args.he_cal_per_class if ds.name == "he" else None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ntufi", "he"], required=True)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=None, help="default 200 (NTU-Fi) / 100 (He)")
    p.add_argument("--he-cal-per-class", type=int, default=HE_CAL_PER_CLASS)
    p.add_argument("--defenses", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = C.SUPCON_EPOCHS if args.dataset == "ntufi" else HE_SUPCON_EPOCHS
    if args.dry_run:
        est = C.ntufi_estimate_mb() * 3 if args.dataset == "ntufi" else C.he_estimate_mb() * 2.5
        print(f"estimated peak working set ~{est:.0f} MB (VM cap {C.VM_LIMIT_GB} GB, RSS cap {C.MAX_RSS_MB} MB)")
        return
    run(args)


if __name__ == "__main__":
    main()
