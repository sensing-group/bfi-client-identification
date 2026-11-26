#!/usr/bin/env python3
"""Compare the key numbers of the paper against the files in figures/.

Default: read release/expected_numbers.json and compare every entry with the
value currently in figures/ (summary JSON or CSV). Exit 1 on any mismatch
beyond --tol (absolute, on the 0..1 scale; 0.005 is half a point).

--write: regenerate release/expected_numbers.json from the current figures/.
Run it once after a full reproduction that the authors have checked against
the printed paper; never run it to make a failing check pass.

Every entry records the file it was read from and the row or cell it names,
so a mismatch points at the script to re-run (see release/README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RELEASE_DIR.parent
FIG_DIR = REPO_ROOT / "figures"
EXPECTED_PATH = RELEASE_DIR / "expected_numbers.json"
DEFAULT_TOL = 0.005
SESSION_GAP_S = "300"

DEVICES = ("AppleTV-LR", "AppleTV-BR", "LGTV", "Laptop", "Vacuum", "SamsungTV")
TIME_K_GRID = ("1", "16", "all")
TIME_T_GRID = ("1", "2", "4", "8", "16", "32")
# Table "Recall with full reports and with reports cut to 512 bytes":
# (condition, feature, K, T) of r_main.csv / c5b_trunc_main.csv, accuracy column.
CUT_TABLE_CELLS = (
    ("same_day", "angles", "1", "1"),
    ("same_day", "angles", "all", "1"),
    ("other_day_any", "angles", "1", "1"),
    ("other_day_any", "angles", "all", "1"),
    ("other_day_any", "angles+format", "all", "1"),
)
PAIR_CELL = ("other_day_any", "angles", "all", "1")
TRUNC_MASK = "common"
# Section 7: attacker accuracy at T = 1, K = 1 per defense (plain 1-NN), and
# the invariant-reading 1-NN where the paper quotes it.
HE_DEFENSES_NN1 = ("none", "dpsq_e0.01", "dpsq_e0.5", "dpsq_e2.0", "manifold_e0.5",
                   "unitary_fixed", "unitary_global", "unitary_pkt", "unitary_pkt_sc",
                   "orthq_window")
HE_DEFENSES_NN1_INV = ("none", "unitary_pkt", "unitary_pkt_sc", "orthq_window")
GAIN_CELLS = ("none|none", "dpsq_e0.01|none", "dpsq_e0.5|none", "dpsq_e2.0|none",
              "manifold_e0.5|none", "unitary_pkt|AP holds Q", "unitary_pkt_sc|AP holds Q",
              "unitary_fixed|AP holds Q", "unitary_global|AP holds Q",
              "orthq_window|AP holds Q (float feedback)")
WEEKS_CONDITIONS = ("Jul_to_Sep", "Sep_to_Jul", "Jul_within", "Sep_within")
WEEKS_K_GRID = ("1", "all")
WEEKS_DEVICES = ("ALL_macro", "LGTV", "AppleTV-LR", "SamsungTV")


# Section 7.4, Table 5: access-point-only protection on the home capture,
# one window enrolled, one query window. (body, observer) of r_rotation_main.csv
# and (body, policy, observer) of r_format_equalized.csv.
AP_CONDITIONS = ("other_day_any", "same_day")
# Section 7.4: centroid rule under the per-group rotation, whole session, native
# and common grid; macro recall of the plain and invariant observers.
ROT_CENTROID_CELLS = (("rot_group", "angles_centroid", "all", "1"), ("rot_group", "angles_centroid", "all", "32"),
                      ("rot_group_cb42_n234", "angles_centroid", "all", "1"),
                      ("rot_group_cb42_n234", "angles_centroid", "all", "32"),
                      ("null_noise", "angles_centroid", "all", "1"), ("null_noise", "angles_centroid", "all", "32"))
ROT_MACRO_CELLS = (("rot_group", "angles", "1", "1"), ("rot_group", "invariant", "1", "1"),
                   ("rot_group_cb42_n234", "angles_centroid", "all", "1"))
# Section 6.1: unprotected angles on the common grid (r_rotation.py --common-grid-only).
COMMON_GRID_K = ("1", "all")
AP_ROTATION_CELLS = (("none", "angles"), ("none", "format"),
                     ("rot_report", "angles"), ("rot_report", "invariant"),
                     ("rot_group", "angles"), ("rot_group", "invariant"),
                     ("rot_group_cb42_n234", "angles"), ("null_noise", "angles"))
AP_FORMAT_CELLS = (("none", "P2_all", "format"), ("rot_group", "P2_all", "angles+format"))
# Section 7.3, Figure 15: Manifold DP budgets beyond 0.5 (manifold_sweep.py).
MANIFOLD_SWEEP_CELLS = ("manifold_e32.0", "manifold_e128.0", "manifold_e512.0",
                        "manifold_e0.5", "manifold_e2.0", "manifold_e8.0")
# Section 7.3, Figure 15: BeamDancer per randomization (beamdancer_sweep.py).
BEAMDANCER_CELLS = ("b42_R0", "b42_R10", "b42_R50", "b42_R100", "b42_R25", "b42_R75")
# Section 8: address changes replayed on the home capture (r_address_rotation.py).
ADDRESS_EPOCHS = ("none", "3600.0", "600.0", "60.0")
ADDRESS_CELLS = ("session|other_day_any|angles|1", "session|other_day_any|format|1",
                 "pair|other_day_any|angles|all", "pair|other_day_any|format|all")
# Section 7.4 and Appendix D: the SNR field (r_snr_field.py), one window enrolled.
SNR_CELLS = ("1|none|0|level|angles|other_day_any", "1|none|0|level|angles|same_day",
             "1|none|0|level|angles+format|other_day_any", "1|none|0|level|angles+format|same_day",
             "1|report|12.0|level|angles|other_day_any", "1|session|12.0|level|angles|other_day_any",
             "1|session|12.0|level|angles+format|other_day_any", "2|none|0|shape|angles|other_day_any",
             "2|report|12.0|shape|angles|other_day_any", "2|session|12.0|shape|angles|other_day_any")
# Section 7.3: noise on the home clients (r_noise_home.py), body|feature|condition|K|T.
NOISE_BODIES = ("none", "dpsq_e0.01", "dpsq_e0.5", "dpsq_e2.0", "manifold_e0.5", "manifold_e2",
                "manifold_e8", "manifold_e128", "manifold_e512", "bd_R10", "bd_R25", "bd_R100")
NOISE_CELLS = ("angles|other_day_any|1|1", "angles|other_day_any|all|1")
NOISE_MACRO_CELL = "angles|other_day_any|all|1"
NOISE_PAIR_CELL = "angles|other_day_any|all|1"
# Appendix D: windows of 8 reports, whole session enrolled (r_noise_home_short_window.csv).
NOISE_SHORT_BODIES = ("none", "manifold_e8", "bd_R100")
# Section 6.1: macro recall beside the accuracy (r_main.csv), and device weights.
MACRO_CELLS = (("other_day_any", "1"), ("other_day_any", "all"), ("same_day", "1"))
# Section 6.1: 95% intervals of the headline; Appendix C: session-gap sensitivity (r_main.csv).
CI_K_GRID = ("1", "all")
SENSITIVITY_GAPS = ("120", "600")
# Section 7.4: multi-user delta-SNR and average-SNR byte on the shared-format pair (r_mu_dsnr.py).
MU_DSNR_CELLS = (("dsnr", "1"), ("dsnr", "all"), ("avg_snr", "1"), ("avg_snr", "all"))


# ---------------------------------------------------------------------------
# Results of the later revision rounds. Each *_CELLS tuple lists the cells of
# one result file that the paper prints: (where, column) pairs for a CSV, the
# row named by the filter and the column read from it, and (path, stat) pairs
# for a JSON file, the keys walked from the root and what is read at the end
# (JSON_STATS). collect() reads them through _csv_cells and _json_cells.
ACROSS_DAYS = "other_day_any"
DPSQ_BUDGETS = ("dpsq_e0.01", "dpsq_e0.5", "dpsq_e2.0")
ENROLLED = ("1", "all")                  # one window, the whole session
ACCURACY_AND_MACRO = ("accuracy", "macro_recall")


def _round_row(part, analysis, cell, observer="", k="", t="", metric="accuracy", note=None) -> dict:
    """Filter for one row of r_round3_checks.csv or r_round4_checks.csv."""
    where = dict(part=part, analysis=analysis, cell=cell, observer=observer, K=k, T=t, metric=metric)
    if note is not None:
        where["note"] = note
    return where


def _home_row(feature="angles", k="1", t="1", condition=ACROSS_DAYS, **more) -> dict:
    """Filter for one row of the home-capture tables (r_eval's columns)."""
    return dict(more, condition=condition, feature=feature, K=k, T=t)


def _ndp_row(dataset, body, rho="1.0", g="g1_iid", snr_db="20.0", m="1", k="1", t="1") -> dict:
    """Filter for one row of r_rotation_ndp.csv (one antenna, one enrolled and one query window)."""
    return dict(dataset=dataset, body=body, m=m, g=g, snr_db=snr_db, rho=rho, K=k, T=t)


# Sections 7.3 and 8 (r_round3_checks.py): soundings that two labeled devices
# answered and what their product tells with and without a shared rotation
# (home, an hour apart and across days; positions), the combined defense with
# the SNR field in steps of 12 dB or removed, and DP-SQ's full-support variant.
ROUND3_CSV = "r_round3_checks.csv"
ROUND3_PAIR_CELLS = (("as_captured", "accuracy", "other_hour"), ("shared_q_native", "accuracy", "other_hour"),
                     ("shared_q_native", "accuracy", "other_day"), ("shared_q_native", "macro_recall", "other_day"),
                     ("shared_q_cb42", "accuracy", "other_day"), ("shared_q_cb42", "macro_recall", "other_day"))
ROUND3_CELLS = (
    (_round_row("1a", "shared_sounding_counts", "shared|20ms", metric="pairs"), "value"),
    *((_round_row("1a", "real_pair_feature", variant, "pair_1nn", "1", "1", metric, note), "value")
      for variant, metric, note in ROUND3_PAIR_CELLS),
    (_round_row("1b", "he_shared_sounding", "shared_q", "pair|random_report|scaled|all_partners", "1", "1"), "value"),
    *((_round_row("2", "combined_defense", "snr_step=12", observer, k, "1", "macro_recall"), "value")
      for observer in ("combined", "snr_alone") for k in ENROLLED),
    *((_round_row("2", "combined_defense", "snr_step=removed", "combined", k, "1", metric), "value")
      for k in ENROLLED for metric in ("macro_recall", "reference_macro_recall")),
    (_round_row("3", "dp_gsq_full_support_variant", "dpgsq_e0.5", metric="gain_col1"), "value"),
    *((_round_row("3", "dp_gsq_full_support_variant", "dpgsq_e0.5", observer, "1", "1"), "value")
      for observer in ("scaled", "raw")),
)
# The 125 shared soundings outside the main day, and the format classes the
# multi-user policy leaves (seven to two).
ROUND3_JSON = "r_round3_checks.json"
ROUND3_SOUNDINGS_BY_DAY = ("part1a_home_shared_soundings", "real_pairs", "soundings_per_class_and_day")
ROUND3_JSON_CELLS = (
    ((*ROUND3_SOUNDINGS_BY_DAY, "Laptop+Vacuum", "2026-05-11"), "value"),
    ((*ROUND3_SOUNDINGS_BY_DAY, "Laptop+SamsungTV", "2026-05-16"), "value"),
    (("part2_combined_defense", "format_classes_left_by_policy", "P1"), "value"),
    (("part2_combined_defense", "format_classes_left_by_policy", "P2_all"), "value"),
)

# Sections 6.1, 7.1, 7.3 and 7.4 (r_round4_checks.py): RSSI and report timing
# alone, the 6/4-bit codebook, the dataset's four other rooms, two clients
# under zero-forcing, the equivalence margins against the noise reference and
# the capture without the Samsung TV.
ROUND4_CSV = "r_round4_checks.csv"
ROUND4_OTHER_ROOMS = ("Conference Room 2", "Lab 1", "Lab 2", "Rooftop")
ROUND4_TOST_NOTE = "block bootstrap, alpha 0.05"
ROUND4_INTERVAL_NOTE = "block bootstrap, 0.95 interval"
ROUND4_COMMON_GRID = "rot_group_cb42_n234"
ROUND4_CELLS = (
    *((_round_row("1", "timing_rssi_baseline", cell, "1nn", "1", "1", metric), "value")
      for cell in ("rssi", "timing") for metric in ACCURACY_AND_MACRO),
    *((_round_row("2", "deeper_codebooks", f"6/4|{defense}", "scaled", "1", "1"), "value")
      for defense in ("none", *DPSQ_BUDGETS)),
    *((_round_row("2", "deeper_codebooks", f"6/4|{defense}", metric="gain_col1"), "value")
      for defense in DPSQ_BUDGETS),
    *((_round_row("3", "five_rooms", f"{room}|{defense}", "scaled", "1", "1"), "value")
      for room in ROUND4_OTHER_ROOMS for defense in (*DPSQ_BUDGETS, "manifold_e0.5")),
    *((_round_row("3", "five_rooms", f"{room}|dpsq_e0.5", "scaled", "1", "1"), "chance")
      for room in ROUND4_OTHER_ROOMS),
    *((_round_row("3", "five_rooms", f"{room}|{defense}", metric="gain_col1"), "value")
      for room in ROUND4_OTHER_ROOMS for defense in ("manifold_e0.5", "unitary_pkt_sc")),
    *((_round_row("3", "five_rooms", f"{room}|unitary_pkt_sc", observer, "1", "1"), "value")
      for room in ROUND4_OTHER_ROOMS for observer in ("scaled", "raw", "invariant")),
    *((_round_row("4", "two_user_zero_forcing", f"4/2|{defense}", "all_pairs", metric="sum_rate_ratio"), "value")
      for defense in ("none", "unitary_pkt_sc", *DPSQ_BUDGETS)),
    *((_round_row("5", "equivalence", ROUND4_COMMON_GRID, observer, k, "1", "macro_recall_tost_margin",
                  ROUND4_TOST_NOTE), "value") for observer in ("angles|stored", "invariant|stored") for k in ENROLLED),
    *((_round_row("5", "equivalence", f"{ROUND4_COMMON_GRID}+one_format+no_snr", "combined|stored", k, "1",
                  "macro_recall_tost_margin", ROUND4_TOST_NOTE), "value") for k in ENROLLED),
    *((_round_row("5", "equivalence", ROUND4_COMMON_GRID, "angles|stored", "1", "1", "macro_recall_minus_reference",
                  ROUND4_INTERVAL_NOTE), column) for column in ("value", "lo", "hi")),
    *((_round_row("6", "five_devices_rerun", ACROSS_DAYS, "stored", k, "1", metric), "value")
      for k in ENROLLED for metric in ACCURACY_AND_MACRO),
)

# Section 7 and Appendix C, the test bench (defense_bench.py): the observers'
# accuracy, the first-stream gain and the verdict of the eight built-in defenses.
BENCH_FILES = ("none_b42", "dpsq_e0.5_b42", "manifold_e512.0_b42", "beamdancer_R100_b42", "rotation_report_b42",
               "rotation_group_b42", "rotation_fixed_b42", "rotation_shared_b42")
BENCH_PATH = "bench/{stem}.json"
BENCH_CELLS = (
    *((("observers", f"{observer}|T={t}", "mean"), "value")
      for observer in ("scaled", "raw", "invariant") for t in ("1", "32")),
    (("worst_observer", "ub95"), "value"),
    (("gain_first_stream", "mean"), "value"),
    *((("verdict", flag), "value") for flag in ("private", "keeps_gain", "passes")),
)

# Section 6.1 and Appendix C, "Scaling and intervals" (r_eval_checks.py).
EVAL_CHECKS_CSV = "r_eval_checks.csv"
EVAL_CHECKS_HOME = dict(section="home", body="none", observer="angles", condition=ACROSS_DAYS)
EVAL_CHECKS_CELLS = (
    *((dict(EVAL_CHECKS_HOME, K=k, T="1", scaling=scaling), column)
      for scaling in ("enrolled", "raw") for k in ENROLLED for column in ACCURACY_AND_MACRO),
    *((dict(EVAL_CHECKS_HOME, K="1", T="1", scaling="stored"), column)
      for column in ("macro_recall_block_lo", "macro_recall_block_hi", "macro_recall_day_lo", "macro_recall_day_hi")),
)

# Sections 6.2 and 7.3: the unscaled observer, unprotected and under DP-SQ (r_raw_observer_defenses.py).
RAW_OBSERVER_CSV = "r_raw_observer_defenses.csv"
RAW_OBSERVER_CELLS = (
    (dict(defense="none", T="1"), "raw_mean"),
    (dict(defense="none", T="1"), "scaled_rerun_mean"),
    *((dict(defense=defense, T="1"), "raw_mean") for defense in DPSQ_BUDGETS),
)

# Section 7.3, Table 4 and Appendix C: an observer of the sounding (r_rotation_ndp.py).
# Table rows by the weight of the enrolled channel, the macro recall quoted for
# the home, the plain observer on the positions, four random taps, and 10 dB.
NDP_CSV = "r_rotation_ndp.csv"
NDP_TABLE_ROWS = (("he", "rot_group"), ("home", "rot_group"), ("home", "rot_group_cb42_n234"))
NDP_WEIGHTS = ("1.0", "0.9", "0.5", "0.0")
NDP_DATASETS = ("home", "he")
NDP_CELLS = (
    *((_ndp_row(dataset, body, rho), "accuracy") for dataset, body in NDP_TABLE_ROWS for rho in NDP_WEIGHTS),
    (_ndp_row("home", "rot_group"), "macro_recall"),
    (_ndp_row("he", "none"), "ref_plain_angles"),
    *((_ndp_row(dataset, "rot_group", g="g2_taps"), "accuracy") for dataset in NDP_DATASETS),
    *((_ndp_row(dataset, "rot_group", snr_db="10.0"), "accuracy") for dataset in NDP_DATASETS),
)
# Table 4, last row: the plain observer on the windows of this run, no defense.
NDP_JSON = "r_rotation_ndp.json"
NDP_JSON_CELLS = ((("home", "cells", "none|m1|g1_iid|snr20|rho1", "1", "1", "plain_angles_this_run"), "value"),)

# Section 7.2 and Appendix C, "Client smoothing" (r_rotation_smoothing.py): gain
# at 4 and 2 bits without rotation, per group and with Q held over 2, 4, 8 groups.
SMOOTHING_CSV = "r_rotation_smoothing.csv"
SMOOTHING_CELLS = tuple((dict(cell=cell), "gain") for cell in (
    "none|tap3|cb_4_2", "group|tap3|cb_4_2", "group|box4|cb_4_2", "band2|tap3|cb_4_2", "band4|tap3|cb_4_2",
    "band8|tap3|cb_4_2"))

# Appendix C, Table 6 (r_rotation_space.py): each cheaper rotation on the
# positions with its best invariant observer (None: no invariant left), and the
# invariant observer from one query window when Q is held over 2, 4 or 8 groups.
SPACE_CSV = "r_rotation_space.csv"
SPACE_TABLE_ROWS = (("s1_diag|f2_group", "inv_mag"), ("s2_perm|f2_group", "inv_sortmag"),
                    ("s3_orth|f2_group", "inv_tsym"),
                    ("s4_unitary|f1_report", "inv_cross"), ("s4_unitary|f3_band8", "inv_cross"),
                    ("s4_unitary|f3_band2", "inv_cross"), ("s4_unitary|f2_group", None))
SPACE_BANDS = ("s4_unitary|f3_band2", "s4_unitary|f3_band4", "s4_unitary|f3_band8")
SPACE_CELLS = (
    *((dict(cell=cell, dataset="he", observer="plain", T="32"), column)
      for cell, _ in SPACE_TABLE_ROWS for column in ("accuracy", "gain_col1", "complex_values_stored")),
    *((dict(cell=cell, dataset="he", observer=observer, T="32"), "accuracy")
      for cell, observer in SPACE_TABLE_ROWS if observer is not None),
    *((dict(cell=cell, dataset="he", observer="inv_cross", T="1"), "accuracy") for cell in SPACE_BANDS),
)

# Section 7.4, Table 5, macro column and Figure 12(b) (r_rotation_main.csv,
# r_format_equalized.csv), and Section 5: the centroid rule without a defense.
ROTATION_CSV = "r_rotation_main.csv"
ROTATION_MACRO_ROWS = (("none", "angles"), ("none", "format"), ("rot_report", "angles"), ("rot_report", "invariant"),
                       ("rot_group_cb42_n234", "angles"), ("null_noise", "angles"))
ROTATION_EXTRA_CELLS = (
    *((_home_row(feature, body=body), "macro_recall") for body, feature in ROTATION_MACRO_ROWS),
    (_home_row("format", body="rot_group"), "accuracy"),
    (_home_row("angles_centroid", "all", body="none"), "accuracy"),
)
FORMAT_EQ_CSV = "r_format_equalized.csv"
FORMAT_EQ_MACRO_CELLS = tuple((_home_row(feature, body=body, policy=policy), "macro_recall")
                              for body, policy, feature in AP_FORMAT_CELLS)

# Section 7.4: macro recall of the SNR field alone and with the format, and the
# body lengths of the report formats (shortest, longest, how many).
SNR_JSON = "r_snr_field_summary.json"
SNR_JSON_CELLS = (
    (("cells", "1|none|0|level|angles|other_day_any|1|1", "macro_recall"), "value"),
    (("cells", "1|none|0|level|angles+format|other_day_any|1|1", "macro_recall"), "value"),
    (("format_body_lengths",), "min"),
    (("format_body_lengths",), "max"),
    (("format_body_lengths",), "len"),
)

# Appendix C, "Address changes": the label covers one epoch (r_address_rotation.py);
# the share of each device's frames left in windows when the address changes every minute.
ADDRESS_JSON = "r_address_rotation_summary.json"
ADDRESS_EPOCH_CELL = "{epoch}|32|epoch|other_day_any|angles|{k}"
ADDRESS_MINUTE = "60.0"
ADDRESS_JSON_CELLS = (
    *((("cells", ADDRESS_EPOCH_CELL.format(epoch=epoch, k=k), "accuracy"), "value")
      for epoch in ADDRESS_EPOCHS for k in ENROLLED),
    (("cells", ADDRESS_EPOCH_CELL.format(epoch=ADDRESS_MINUTE, k="1"), "macro_recall"), "value"),
    *((("frames_in_windows", f"{ADDRESS_MINUTE}|32", device), "value") for device in DEVICES),
)

# Section 6.1 (r_eval.py): 16 windows on the same day, chance over the same-day
# decisions, the decisions of 50 trials; recall per device with 1 and 16
# windows and from the format alone; the pair that shares a format.
MAIN_CSV = "r_main.csv"
MAIN_EXTRA_CELLS = (
    (_home_row(k="16", condition="same_day", gap_s=SESSION_GAP_S), "accuracy"),
    (_home_row(condition="same_day", gap_s=SESSION_GAP_S), "chance"),
    (_home_row(gap_s=SESSION_GAP_S), "n_decisions"),
)
PER_DEVICE_CSV = "r_per_device.csv"
FORMAT_DEVICES = ("AppleTV-LR", "LGTV", "Laptop", "Vacuum")
HEAVIEST_DEVICE = "AppleTV-LR"
PER_DEVICE_CELLS = (
    *((_home_row(k=k, gap_s=SESSION_GAP_S, device=device), "recall") for k in ("1", "16") for device in DEVICES),
    *((_home_row("format", gap_s=SESSION_GAP_S, device=device), "recall") for device in FORMAT_DEVICES),
    (_home_row(gap_s=SESSION_GAP_S, device=HEAVIEST_DEVICE), "n_decisions"),
)
PAIR_CSV = "r_pair.csv"
PAIR_EXTRA_CELLS = (
    (_home_row(gap_s=SESSION_GAP_S), "accuracy"),
    *((_home_row("format", k, gap_s=SESSION_GAP_S), "accuracy") for k in ENROLLED),
)

# Section 8, "Device or place": the devices that move, on the common grid, whole session.
COMMON_GRID_DEVICE_CSV = "r_rotation_common_grid_per_device.csv"
MOVING_DEVICES = ("Laptop", "Vacuum")
COMMON_GRID_DEVICE_CELLS = tuple((_home_row(k="all", body="none_cb42_n234", device=device), "recall")
                                 for device in MOVING_DEVICES)

# Appendix C, "Noise on the home capture": per device at budget 0.5, whole session.
NOISE_DEVICE_CSV = "r_noise_home_per_device.csv"
NOISE_DEVICE_CELLS = tuple((_home_row(k="all", body="manifold_e0.5", device=device), "recall")
                           for device in (*MOVING_DEVICES, HEAVIEST_DEVICE))

# Section 6.1, Figure 6 (r_recall_by_day.py): 16 windows of the first session
# on the next day and on the last; the whole session on the vacuum's two later
# days, and the lowest later day of the four devices that stay.
RECALL_BY_DAY_CSV = "r_recall_by_day.csv"
RECALL_BY_DAY_CELLS = tuple((dict(feature="angles", K="16", T="1", device=HEAVIEST_DEVICE, query_day=day), "recall")
                            for day in ("2026-05-09", "2026-05-20"))
DRIFT_CSV = "r_recall_by_day_drift.csv"
DRIFT_WHOLE_SESSION = dict(feature="angles", K="all", T="1", rule="nn")
DRIFT_CELLS = tuple((dict(DRIFT_WHOLE_SESSION, device="Vacuum", query_day=day), "recall")
                    for day in ("2026-05-15", "2026-05-18"))
DRIFT_MIN_CELLS = tuple((dict(DRIFT_WHOLE_SESSION, device=device), "recall")
                        for device in ("AppleTV-LR", "AppleTV-BR", "LGTV", "Laptop"))

# Section 4, Table 2 (r_eval.py --device-set): frames, sessions with a window,
# days and median window duration per device; the unlabeled addresses, the
# window vector's length, the report formats decoded and the laptop's 99th
# percentile gap.
DEVICE_SET_JSON = "r_device_set.json"
DEVICE_SET_CELLS = (
    *(cell for device in DEVICES for cell in (
        (("gap_report", device, "frames"), "value"),
        (("session_stats", SESSION_GAP_S, device, "sessions_with_window"), "value"),
        (("devices", device, "days"), "len"),
        (("devices", device, "window_seconds_median"), "value"))),
    (("feature_dim",), "value"),
    (("decoder_check",), "len"),
    (("gap_report", "Laptop", "gap_p99_s"), "value"),
)

# Section 7.3: gain kept on each stream under the rotation per group.
GAIN_STREAMS_JSON = "bfi_he_beamforming_gain_streams.json"
GAIN_STREAMS_CELLS = tuple((("cells", "unitary_pkt_sc|AP holds Q", field), "value")
                           for field in ("gain_s1", "gain_s2"))
# Section 7.2: what padding every report to the longest costs (r_padding_cost.py).
PADDING_JSON = "r_padding_cost.json"
PADDING_CELLS = (
    *(((field,), "value") for field in ("factor", "factor_min", "factor_max", "longest_body_len",
                                        "shortest_body_len", "days_with_data")),
)
# Section 5.3, Figure 5: the best single coordinate against all coordinates
# jointly, on the proxy and on real feedback (bfi_identity_invariant.py).
INVARIANT_JSON = "bfi_identity_invariant.json"
INVARIANT_CELLS = tuple((("results", dataset, *field), "value")
                        for dataset in ("ntufi", "he")
                        for field in (("dim",), ("single_coord", "max"), ("joint_cos",)))

CSV_TABLES = (
    ("r_round3_checks", ROUND3_CSV, ROUND3_CELLS),
    ("r_round4_checks", ROUND4_CSV, ROUND4_CELLS),
    ("r_eval_checks", EVAL_CHECKS_CSV, EVAL_CHECKS_CELLS),
    ("r_raw_observer_defenses", RAW_OBSERVER_CSV, RAW_OBSERVER_CELLS),
    ("r_rotation_ndp", NDP_CSV, NDP_CELLS),
    ("r_rotation_smoothing", SMOOTHING_CSV, SMOOTHING_CELLS),
    ("r_rotation_space", SPACE_CSV, SPACE_CELLS),
    ("r_rotation_main", ROTATION_CSV, ROTATION_EXTRA_CELLS),
    ("r_format_equalized_macro", FORMAT_EQ_CSV, FORMAT_EQ_MACRO_CELLS),
    ("r_main_cells", MAIN_CSV, MAIN_EXTRA_CELLS),
    ("r_per_device", PER_DEVICE_CSV, PER_DEVICE_CELLS),
    ("r_pair_cells", PAIR_CSV, PAIR_EXTRA_CELLS),
    ("r_rotation_common_grid_per_device", COMMON_GRID_DEVICE_CSV, COMMON_GRID_DEVICE_CELLS),
    ("r_noise_home_per_device", NOISE_DEVICE_CSV, NOISE_DEVICE_CELLS),
    ("r_recall_by_day", RECALL_BY_DAY_CSV, RECALL_BY_DAY_CELLS),
    ("r_recall_by_day_drift", DRIFT_CSV, DRIFT_CELLS),
)
CSV_MIN_TABLES = (("r_recall_by_day_drift", DRIFT_CSV, DRIFT_MIN_CELLS),)
JSON_TABLES = (
    ("r_round3_checks", ROUND3_JSON, ROUND3_JSON_CELLS),
    ("r_rotation_ndp", NDP_JSON, NDP_JSON_CELLS),
    ("r_snr_field_summary", SNR_JSON, SNR_JSON_CELLS),
    ("r_address_rotation_summary", ADDRESS_JSON, ADDRESS_JSON_CELLS),
    ("r_device_set", DEVICE_SET_JSON, DEVICE_SET_CELLS),
    ("he_beamforming_gain_streams", GAIN_STREAMS_JSON, GAIN_STREAMS_CELLS),
    ("identity_invariant", INVARIANT_JSON, INVARIANT_CELLS),
    ("r_padding_cost", PADDING_JSON, PADDING_CELLS),
    *((f"defense_bench.{stem}", BENCH_PATH.format(stem=stem), BENCH_CELLS) for stem in BENCH_FILES),
)


def _leaves(node):
    """Every scalar under a JSON node."""
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            yield from _leaves(item)
    else:
        yield node


JSON_STATS = {"value": float, "len": len,
              "min": lambda node: min(_leaves(node)), "max": lambda node: max(_leaves(node))}


def _rows(name: str) -> list[dict]:
    with open(FIG_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def _one(rows: list[dict], where: dict, col: str, name: str) -> float:
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(hits) != 1:
        raise KeyError(f"{name}: {len(hits)} rows match {where}")
    return float(hits[0][col])


def _json(name: str) -> dict:
    with open(FIG_DIR / name) as f:
        return json.load(f)



def _csv_cells(put, prefix: str, name: str, cells) -> None:
    """One number per (where, column): the column of the single row the filter names."""
    rows = _rows(name)
    for where, column in cells:
        key = ".".join([prefix, *(f"{k}={v}" for k, v in where.items()), column])
        put(key, _one(rows, where, column, name), f"figures/{name}", dict(where, column=column))


def _csv_min(put, prefix: str, name: str, cells) -> None:
    """The lowest value of a column over the rows a filter names (rows without a value skipped)."""
    rows = _rows(name)
    for where, column in cells:
        vals = [float(r[column]) for r in rows if all(r[k] == v for k, v in where.items())]
        vals = [v for v in vals if v == v]
        if not vals:
            raise KeyError(f"{name}: no row with a value matches {where}")
        key = ".".join([prefix, *(f"{k}={v}" for k, v in where.items()), column, "min"])
        put(key, min(vals), f"figures/{name}", dict(where, column=column, stat="min", n_rows=len(vals)))


def _json_cells(put, prefix: str, name: str, cells) -> None:
    """One number per (path, stat): the node at the path, reduced by JSON_STATS[stat]."""
    root = _json(name)
    for path, stat in cells:
        node = root
        for step in path:
            node = node[step]
        put(".".join([prefix, *path, stat]), JSON_STATS[stat](node), f"figures/{name}",
            dict(path=list(path), stat=stat))


def collect() -> dict[str, dict]:
    """Read every key number from figures/. Returns {key: {value, source, where}}."""
    out: dict[str, dict] = {}

    def put(key, value, source, where):
        out[key] = dict(value=round(float(value), 6), source=source, where=where)

    # Section 6.1: recall across days per device, whole session enrolled (r_eval).
    rows = _rows("r_per_device.csv")
    for dev in DEVICES:
        where = dict(gap_s=SESSION_GAP_S, condition="other_day_any", feature="angles",
                     K="all", T="1", device=dev)
        put(f"r_eval.other_day_any.angles.K=all.T=1.{dev}.recall",
            _one(rows, where, "recall", "r_per_device"), "figures/r_per_device.csv", where)

    # Section 6.1: pooled recall against listening time (m_time_to_identify).
    rows = _rows("m_time_to_identify.csv")
    for k in TIME_K_GRID:
        for t in TIME_T_GRID:
            where = dict(device="ALL", K=k, T=t)
            put(f"m_time_to_identify.ALL.K={k}.T={t}.recall",
                _one(rows, where, "recall", "m_time_to_identify"),
                "figures/m_time_to_identify.csv", where)

    # Sections 6.2 / 6.3: calibration-size sweep, no calibration and the largest cohort.
    for ds in ("he", "ntufi"):
        d = _json(f"s_calibration_size_{ds}_summary.json")
        agg = [a for a in d["aggregate"] if a["attacker"] == "nn1" and a["K"] == "1"]
        for n_cal in (0, max(a["n_cal"] for a in agg)):
            hit = [a for a in agg if a["n_cal"] == n_cal]
            if len(hit) != 1:
                raise KeyError(f"s_calibration_size {ds} n_cal={n_cal}: {len(hit)} rows")
            put(f"s_calibration_size.{ds}.nn1.K=1.n_cal={n_cal}.acc_mean", hit[0]["acc_mean"],
                f"figures/s_calibration_size_{ds}_summary.json",
                dict(attacker="nn1", K="1", n_cal=n_cal))

    # Section 7: observer accuracy per defense on He (K = 1, T = 1).
    d = _json("bfi_joint_defense_he_summary.json")["cells"]
    for defense in HE_DEFENSES_NN1:
        cell = f"{defense}|nn1|T=1"
        put(f"joint_defense_he.{cell}.mean", d[cell]["mean"],
            "figures/bfi_joint_defense_he_summary.json", dict(cell=cell))
    for defense in HE_DEFENSES_NN1_INV:
        cell = f"{defense}|nn1_inv|T=1"
        put(f"joint_defense_he.{cell}.mean", d[cell]["mean"],
            "figures/bfi_joint_defense_he_summary.json", dict(cell=cell))

    # Section 7: beamforming gain the access point keeps (first stream).
    d = _json("bfi_he_beamforming_gain_summary.json")["cells"]
    for cell in GAIN_CELLS:
        put(f"he_beamforming_gain.{cell}.gain_col1", d[cell]["gain_col1"],
            "figures/bfi_he_beamforming_gain_summary.json", dict(cell=cell))

    # Section 7.3, Figure 15: Manifold DP across budgets, plain 1-NN at T = 1 and gain.
    d = _json("bfi_manifold_sweep_he_summary.json")["cells"]
    for cell in MANIFOLD_SWEEP_CELLS:
        for field in ("nn1_T=1", "gain_col1"):
            put(f"manifold_sweep_he.{cell}.{field}.mean", d[cell][field]["mean"],
                "figures/bfi_manifold_sweep_he_summary.json", dict(cell=cell, field=field))

    # Section 7.3, Figure 15: BeamDancer, plain 1-NN at T = 1 and gain.
    d = _json("bfi_beamdancer_sweep_he_summary.json")["cells"]
    for cell in BEAMDANCER_CELLS:
        for field in ("nn1_T=1", "gain_col1"):
            put(f"beamdancer_sweep_he.{cell}.{field}.mean", d[cell][field]["mean"],
                "figures/bfi_beamdancer_sweep_he_summary.json", dict(cell=cell, field=field))

    # Section 7.3 and Appendix C: the bound and what a decoder with a prior keeps.
    d = _json("r_address_rotation_summary.json")["cells"]
    for epoch in ADDRESS_EPOCHS:
        for cell in ADDRESS_CELLS:
            key = f"{epoch}|32|{cell}"
            put(f"r_address_rotation.{key}.accuracy", d[key]["accuracy"],
                "figures/r_address_rotation_summary.json", dict(cell=key, field="accuracy"))

    d = _json("r_snr_field_summary.json")["cells"]
    for cell in SNR_CELLS:
        key = f"{cell}|1|1"
        put(f"r_snr_field.{key}.accuracy", d[key]["accuracy"], "figures/r_snr_field_summary.json",
            dict(cell=key, field="accuracy"))

    d = _json("r_noise_home_summary.json")["cells"]
    for body in NOISE_BODIES:
        for cell in NOISE_CELLS:
            key = f"{body}|{cell}"
            put(f"r_noise_home.{key}.accuracy", d[key]["accuracy"],
                "figures/r_noise_home_summary.json", dict(cell=key, field="accuracy"))
        key = f"{body}|{NOISE_MACRO_CELL}"
        put(f"r_noise_home.{key}.macro_recall", d[key]["macro_recall"],
            "figures/r_noise_home_summary.json", dict(cell=key, field="macro_recall"))
    d = _json("r_noise_home_summary.json")["pair_cells"]
    for body in NOISE_BODIES:
        key = f"{body}|{NOISE_PAIR_CELL}"
        put(f"r_noise_home.pair.{key}.accuracy", d[key]["accuracy"],
            "figures/r_noise_home_summary.json", dict(pair_cell=key, field="accuracy"))
    short = _rows("r_noise_home_short_window.csv")
    for body in NOISE_SHORT_BODIES:
        where = dict(body=body, gallery="six", condition="other_day_any", feature="angles", K="all", T="1")
        put(f"r_noise_home.short_window.{body}.accuracy", _one(short, where, "accuracy", "r_noise_home_short"),
            "figures/r_noise_home_short_window.csv", where)

    d = _json("ntu_cross_condition.json")
    put("ntu_cross_condition.mean", d["mean"], "figures/ntu_cross_condition.json", dict(field="mean"))

    main = _rows("r_main.csv")
    for cond, k in MACRO_CELLS:
        where = dict(gap_s=SESSION_GAP_S, condition=cond, feature="angles", K=k, T="1")
        put(f"r_main.{cond}.angles.K={k}.T=1.macro_recall",
            _one(main, where, "macro_recall", "r_main"), "figures/r_main.csv", where)

    d = _json("bfi_prior_decoder_he_summary.json")
    src = "figures/bfi_prior_decoder_he_summary.json"
    for field in ("bound_held", "bound_per_position_mean"):
        put(f"prior_decoder_he.{field}", d[field], src, dict(field=field))
    put("prior_decoder_he.fixed.gain.mean", d["cells"]["fixed"]["gain"]["mean"], src, dict(cell="fixed"))
    put("prior_decoder_he.manifold_e0.5.prior.mean", d["cells"]["manifold_e0.5"]["prior"]["mean"], src,
        dict(cell="manifold_e0.5"))

    # Section 6.3: the 512-byte cut, per device and the table cells.
    rows = _rows("c5b_trunc_per_device.csv")
    for dev in DEVICES:
        where = dict(gap_s=SESSION_GAP_S, mask_policy=TRUNC_MASK, condition="other_day_any",
                     feature="angles", K="all", T="1", device=dev)
        put(f"c5b_trunc.other_day_any.angles.K=all.T=1.{TRUNC_MASK}.{dev}.recall",
            _one(rows, where, "recall", "c5b_trunc_per_device"),
            "figures/c5b_trunc_per_device.csv", where)
    full = _rows("r_main.csv")
    cut = _rows("c5b_trunc_main.csv")
    for cond, feat, k, t in CUT_TABLE_CELLS:
        where = dict(gap_s=SESSION_GAP_S, condition=cond, feature=feat, K=k, T=t)
        put(f"r_main.{cond}.{feat}.K={k}.T={t}.accuracy",
            _one(full, where, "accuracy", "r_main"), "figures/r_main.csv", where)
        where_cut = dict(where, mask_policy=TRUNC_MASK)
        put(f"c5b_trunc_main.{cond}.{feat}.K={k}.T={t}.{TRUNC_MASK}.accuracy",
            _one(cut, where_cut, "accuracy", "c5b_trunc_main"),
            "figures/c5b_trunc_main.csv", where_cut)
    cond, feat, k, t = PAIR_CELL
    where = dict(gap_s=SESSION_GAP_S, condition=cond, feature=feat, K=k, T=t)
    put(f"r_pair.{cond}.{feat}.K={k}.T={t}.accuracy",
        _one(_rows("r_pair.csv"), where, "accuracy", "r_pair"), "figures/r_pair.csv", where)
    where_cut = dict(where, mask_policy=TRUNC_MASK)
    put(f"c5b_trunc_pair.{cond}.{feat}.K={k}.T={t}.{TRUNC_MASK}.accuracy",
        _one(_rows("c5b_trunc_pair.csv"), where_cut, "accuracy", "c5b_trunc_pair"),
        "figures/c5b_trunc_pair.csv", where_cut)

    # Section 6.3: seven weeks on the boards.
    rows = _rows("c5b_weeks.csv")
    for cond in WEEKS_CONDITIONS:
        for k in WEEKS_K_GRID:
            for dev in WEEKS_DEVICES:
                where = dict(condition=cond, feature="angles", K=k, T="1", device=dev)
                put(f"c5b_weeks.{cond}.angles.K={k}.T=1.{dev}.recall",
                    _one(rows, where, "recall", "c5b_weeks"), "figures/c5b_weeks.csv", where)

    # Section 6.3: the 54-day board recording, July enrolled, every later file
    # queried; one window, the files in which all three televisions are heard.
    where = dict(feature="angles", K="1", T="1", device="ALL_macro", n_gallery="3")
    vals = [float(r["recall"]) for r in _rows("c5_continuous.csv")
            if all(r[k] == v for k, v in where.items())]
    for stat, value in (("n_files", len(vals)), ("min", min(vals)), ("max", max(vals)),
                        ("mean", sum(vals) / len(vals))):
        put(f"c5_continuous.angles.K=1.T=1.ALL_macro.three_tv.{stat}", value,
            "figures/c5_continuous.csv", where)

    # Section 6.3: every file enrolled, every other file queried; recognition
    # against the days between the two files (c5_continuous.py --pairs).
    pairs = _json("c5_continuous_pairs_summary.json")
    src = "figures/c5_continuous_pairs_summary.json"
    for b in pairs["bins"]:
        where = dict(pairs["where"], gap_lo_d=b["gap_lo_d"], gap_hi_d=b["gap_hi_d"])
        key = f"c5_continuous_pairs.angles.K=1.three_tv.gap_{b['gap_lo_d']}_{b['gap_hi_d']}"
        put(f"{key}.recall_mean", b["recall_mean"], src, where)
        put(f"{key}.n_pairs", b["n_pairs"], src, where)
    for side in ("within", "beyond"):
        put(f"c5_continuous_pairs.angles.K=1.three_tv.spearman_rho_{side}",
            pairs[f"spearman_rho_{side}"], src, dict(field=f"spearman_rho_{side}"))
    put("c5_continuous_pairs.angles.K=1.three_tv.spearman_rho", pairs["spearman_rho"], src,
        pairs["where"])

    # Section 7.4, Table 5.
    rows = _rows("r_rotation_main.csv")
    for body, feat, k, tq in ROT_CENTROID_CELLS + ROT_MACRO_CELLS:
        where = dict(body=body, condition="other_day_any", feature=feat, K=k, T=tq)
        for col in ("accuracy", "macro_recall"):
            put(f"r_rotation.{body}.other_day_any.{feat}.K={k}.T={tq}.{col}",
                _one(rows, where, col, "r_rotation_main"), "figures/r_rotation_main.csv", where)
    cg = _rows("r_rotation_common_grid.csv")
    for k in COMMON_GRID_K:
        where = dict(body="none_cb42_n234", condition="other_day_any", feature="angles", K=k, T="1")
        for col in ("accuracy", "macro_recall"):
            put(f"r_rotation_common_grid.angles.K={k}.T=1.{col}",
                _one(cg, where, col, "r_rotation_common_grid"), "figures/r_rotation_common_grid.csv", where)
    for cond in AP_CONDITIONS:
        for body, feat in AP_ROTATION_CELLS:
            where = dict(body=body, condition=cond, feature=feat, K="1", T="1")
            put(f"r_rotation.{body}.{cond}.{feat}.K=1.T=1.accuracy",
                _one(rows, where, "accuracy", "r_rotation_main"),
                "figures/r_rotation_main.csv", where)
    for k in CI_K_GRID:
        where = dict(gap_s=SESSION_GAP_S, condition="other_day_any", feature="angles", K=k, T="1")
        for col in ("ci_lo", "ci_hi"):
            put(f"r_main.other_day_any.angles.K={k}.T=1.{col}", _one(main, where, col, "r_main"),
                "figures/r_main.csv", where)
        for gap in SENSITIVITY_GAPS:
            where = dict(gap_s=gap, condition="other_day_any", feature="angles", K=k, T="1")
            put(f"r_main.gap{gap}.other_day_any.angles.K={k}.T=1.accuracy", _one(main, where, "accuracy", "r_main"),
                "figures/r_main.csv", where)
    mu = _rows("r_mu_dsnr.csv")
    for obs, k in MU_DSNR_CELLS:
        where = dict(observer=obs, condition="other_day_any", feature="angles", K=k, T="1")
        put(f"r_mu_dsnr.{obs}.other_day_any.K={k}.T=1.accuracy", _one(mu, where, "accuracy", "r_mu_dsnr"),
            "figures/r_mu_dsnr.csv", where)
    rows = _rows("r_format_equalized.csv")
    for cond in AP_CONDITIONS:
        for body, policy, feat in AP_FORMAT_CELLS:
            where = dict(body=body, policy=policy, condition=cond, feature=feat, K="1", T="1")
            put(f"r_format_equalized.{body}.{policy}.{cond}.{feat}.K=1.T=1.accuracy",
                _one(rows, where, "accuracy", "r_format_equalized"),
                "figures/r_format_equalized.csv", where)

    # The later revision rounds: the tables of cells defined above.
    for prefix, name, cells in CSV_TABLES:
        _csv_cells(put, prefix, name, cells)
    for prefix, name, cells in CSV_MIN_TABLES:
        _csv_min(put, prefix, name, cells)
    for prefix, name, cells in JSON_TABLES:
        _json_cells(put, prefix, name, cells)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="regenerate expected_numbers.json from figures/")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"absolute tolerance on the 0..1 scale (default {DEFAULT_TOL})")
    a = ap.parse_args()
    current = collect()
    if a.write:
        EXPECTED_PATH.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(current)} numbers to {EXPECTED_PATH}")
        return 0
    expected = json.loads(EXPECTED_PATH.read_text())
    bad = 0
    for key, exp in sorted(expected.items()):
        cur = current.get(key)
        if cur is None:
            print(f"MISSING  {key}")
            bad += 1
            continue
        diff = abs(cur["value"] - exp["value"])
        flag = "ok      " if diff <= a.tol else "MISMATCH"
        if diff > a.tol:
            bad += 1
        print(f"{flag} {key}: expected {exp['value']:.4f}, got {cur['value']:.4f} "
              f"({exp['source']})")
    extra = sorted(set(current) - set(expected))
    for key in extra:
        print(f"NEW      {key} (not in expected_numbers.json)")
    print(f"\n{len(expected) - bad}/{len(expected)} numbers within {a.tol}; "
          f"{bad} mismatched or missing; {len(extra)} new")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
