#!/usr/bin/env bash
# Reproduce every number and figure of "Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback".
#
# Usage:
#   release/reproduce.sh            full run: caches, numbers, long runs, figures, check, pdf
#   release/reproduce.sh --quick    figures + check only (reads the CSV/JSON already in figures/; the two figures drawn from the capture itself are skipped without it)
#   release/reproduce.sh --no-long  everything except the long stage and the C5 stream (the runs over 5 minutes)
#   release/reproduce.sh --only figures,check   named stages only
#   release/reproduce.sh --list     print the stages and steps, run nothing
#
# Stages, in dependency order:
#   env      Python, packages, data paths
#   cache    per-day frame caches: r_common (Pi capture), c5_common (C5 mesh, LONG)
#   numbers  every script that writes a CSV/JSON a paper number is read from
#            (c5b_hearing streams both C5 files on its first run)
#   long     joint_defense_eval (He ~40 min, NTU-Fi ~13 min), he_beamforming_gain (~19 min),
#            manifold_sweep (~3 min), beamdancer_sweep, prior_decoder (~9 min),
#            r_rotation* / r_format_equalized / r_rotation_space / he_beamforming_gain_streams /
#            r_address_rotation (Appendix C, the address-change replay, ~3 h),
#            r_common + r_eval at 16 and 64 frames per window (~6 min each; r_main_w16/w64.csv)
#   figures  every figure script (reads figures/*.csv|json, writes figures/*.pdf|png)
#   check    release/check_numbers.py against release/expected_numbers.json
#   pdf      make docker (falls back to make local); skipped without paper.tex
#
# Every step is run in its own process; a failing step is logged and the run
# continues, so one missing input does not hide the rest. Exit status is the
# number of failed steps. Logs: release/logs/<step>.log.
#
# Memory: the scripts cap their own address space (RLIMIT_AS) and stream the
# captures; nothing here runs two scripts at once. BLAS threads are pinned to
# one so the numbers match the published runs and no thread pool wedges.
set -u

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$RELEASE_DIR")"
LOG_DIR="${REPRODUCE_LOG_DIR:-$RELEASE_DIR/logs}"
PY="${PYTHON:-python3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

ALL_STAGES="env cache numbers long figures check pdf"
QUICK_STAGES="figures check"
STAGES="$ALL_STAGES"
SKIP_LONG=0
LIST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) STAGES="$QUICK_STAGES" ;;
    --no-long) SKIP_LONG=1 ;;
    --only) shift; STAGES="${1//,/ }" ;;
    --list) LIST=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"
FAILED=()
N_RUN=0
T_START=$(date +%s)

# step <stage> <name> <command...>   (runs from the repository root)
step() {
  local stage="$1" name="$2"; shift 2
  if [ "$LIST" = 1 ]; then printf "  %-8s %-34s %s\n" "$stage" "$name" "$*"; return 0; fi
  case " $STAGES " in *" $stage "*) ;; *) return 0 ;; esac
  N_RUN=$((N_RUN + 1))
  local t0 rc
  t0=$(date +%s)
  printf "[%s] %-34s " "$stage" "$name"
  (cd "$REPO_ROOT" && "$@") > "$LOG_DIR/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    printf "ok   %5ds\n" $(( $(date +%s) - t0 ))
  else
    printf "FAIL %5ds  (exit %d, see %s)\n" $(( $(date +%s) - t0 )) "$rc" "$LOG_DIR/$name.log"
    FAILED+=("$name")
  fi
}

# window_run <frames>   r_common + r_eval at BFI_WINDOW_FRAMES=<frames> in a cache
# of their own (the window cache is not keyed by the window length), the
# r_main.csv they write copied to figures/r_main_w<frames>.csv
window_run() {
  local n="$1" cache
  cache="${BFI_X_CACHE_DIR:-$HOME/.cache/bfi-x-experiments}/w$n"
  mkdir -p "$cache/figures" || return 1
  BFI_WINDOW_FRAMES="$n" BFI_X_CACHE_DIR="$cache" BFI_X_FIG_DIR="$cache/figures" "$PY" scripts/r_common.py || return 1
  BFI_WINDOW_FRAMES="$n" BFI_X_CACHE_DIR="$cache" BFI_X_FIG_DIR="$cache/figures" "$PY" scripts/r_eval.py || return 1
  cp "$cache/figures/r_main.csv" "figures/r_main_w$n.csv"
}

# capture_step <stage> <name> <command...>   a figure drawn from the capture
# itself (the Pi archive and its window cache), not from figures/*.csv|json.
# Without the capture, as in a bare clone under --quick, it is skipped and the
# committed figure stays.
ARCHIVE_DIR_="${BFI_ARCHIVE_DIR:-$HOME/bfi-archive/data}"
WINDOW_CACHE_="${BFI_X_CACHE_DIR:-$HOME/.cache/bfi-x-experiments}/r"
capture_step() {
  if [ "$LIST" = 0 ] && { ! ls "$ARCHIVE_DIR_"/bfi_*.sqlite >/dev/null 2>&1 || ! ls "$WINDOW_CACHE_"/windows_*.npz >/dev/null 2>&1; }; then
    case " $STAGES " in *" $1 "*) printf "[%s] %-34s skipped (needs the capture and its cache: run the cache stage)\n" "$1" "$2" ;; esac
    return 0
  fi
  step "$@"
}

# long <stage> <name> <command...>   skipped under --no-long
long() {
  if [ "$SKIP_LONG" = 1 ] && [ "$LIST" = 0 ]; then
    case " $STAGES " in *" $1 "*) printf "[%s] %-34s skipped (--no-long)\n" "$1" "$2" ;; esac
    return 0
  fi
  step "$@"
}

[ "$LIST" = 1 ] && echo "stages: $ALL_STAGES   (--quick = $QUICK_STAGES)"

# ---------------------------------------------------------------- env
step env python-and-packages "$PY" - <<'EOF'
import importlib, sys
print(sys.version)
missing = []
for m in ("numpy", "scipy", "matplotlib", "pandas", "sklearn", "torch", "PIL"):
    try:
        mod = importlib.import_module(m); print(f"{m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        missing.append(m); print(f"{m:12s} MISSING ({e})")
sys.exit(1 if missing else 0)
EOF
step env data-paths "$PY" - <<'EOF'
import os, sys
from pathlib import Path
home = Path.home()
here = Path.cwd()
checks = {
  "BFI_ARCHIVE_DIR (Pi capture, one sqlite per day)":
      Path(os.environ.get("BFI_ARCHIVE_DIR", home / "bfi-archive" / "data")),
  "release/vendor (bfi package, run_pi_one_shot.py, zones_cache.json; else BFI_PRIVACY_DIR, ESP32_TOMO_ROOT)":
      here / "release" / "vendor" / "bfi" / "__init__.py",
  "BFI_ANGLE_DECODE_PY (decoder; default release/decoder/angle_decode.py)":
      Path(os.environ.get("BFI_ANGLE_DECODE_PY", here / "release" / "decoder" / "angle_decode.py")),
  "BFI_HE_ROOT (He et al. real-BFI archive, five rooms)":
      Path(os.environ.get("BFI_HE_ROOT", here / "data" / "he_bfi" / "dataset" / "transfer and light weight")),
  "BFI_NTUFI_ROOT (NTU-Fi HumanID)":
      Path(os.environ.get("BFI_NTUFI_ROOT",
           Path(os.environ.get("BFI_PRIVACY_ROOT", here.parent / "bfi-privacy")) / "data" / "ntu_fi" / "NTU-Fi-HumanID")),
  "BFI_C5_LIVE (ESP32-C5 capture, September 2026)":
      Path(os.environ.get("BFI_C5_LIVE",
           Path(os.environ.get("ESP32_TOMO_ROOT", home / "esp32-csi-tomography")) / "csi_data" / "bfi_captures.jsonl")),
  "BFI_C5_ROTATED (ESP32-C5 capture, July 2026 rotation)":
      Path(os.environ.get("BFI_C5_ROTATED",
           Path(os.environ.get("BFI_C5_ROTATED_DIR",
                Path(os.environ.get("ESP32_TOMO_ROOT", home / "esp32-csi-tomography")) / "csi_data" / "rotated"))
           / "bfi_captures.jsonl.20260724T105426Z.gz")),
  "BFI_C5_MIDDLE_DIR (ESP32-C5 recording, the 24 files between July and September)":
      Path(os.environ.get("BFI_C5_MIDDLE_DIR",
           Path(os.environ.get("BFI_C5_ROTATED_DIR",
                Path(os.environ.get("ESP32_TOMO_ROOT", home / "esp32-csi-tomography")) / "csi_data" / "rotated")))),
  "BFI_HE_LOCATION_ROOT (He et al. Conference Room 1: BFI.pth, y.pth)":
      Path(os.environ.get("BFI_HE_LOCATION_ROOT",
           Path(os.environ.get("BFI_HE_ROOT", here / "data" / "he_bfi" / "dataset" / "transfer and light weight"))
           / "Conference Room 1")),
}
missing = 0
for what, p in checks.items():
    ok = p.exists()
    missing += not ok
    print(f"{'ok     ' if ok else 'MISSING'} {what}\n        {p}")
print(f"\n{missing} path(s) missing; set the variable named in brackets to point at the data.")
sys.exit(1 if missing else 0)
EOF

# ---------------------------------------------------------------- cache
step cache r_common-cache            "$PY" scripts/r_common.py
long cache c5_common-stream          "$PY" scripts/c5_common.py --source all

# ---------------------------------------------------------------- numbers (Section 4, 5, 6.1)
step numbers r_eval                  "$PY" scripts/r_eval.py
step numbers r_recall_by_day         "$PY" scripts/r_recall_by_day.py
step numbers r_recall_by_day-drift   "$PY" scripts/r_recall_by_day.py --drift
step numbers r_cost                  "$PY" scripts/r_cost.py
step numbers m_motion_split          "$PY" scripts/m_motion_split.py
step numbers m_time_to_identify      "$PY" scripts/m_time_to_identify.py
step numbers cross_day_routine        "$PY" scripts/cross_day_routine_stability.py   # routine stability across days, not printed in the paper
# Section 5 and 6.2 / 6.3 (public data)
step numbers identity_invariant      "$PY" scripts/identity_invariant.py
step numbers s1_calibration_size-ntufi "$PY" scripts/s1_calibration_size.py --dataset ntufi
step numbers ntu_cross_condition "$PY" scripts/ntu_cross_condition.py
step numbers s1_calibration_size-he  "$PY" scripts/s1_calibration_size.py --dataset he
step numbers s2_env_out              "$PY" scripts/s2_env_out.py
# Section 4 and Section 6.3, "Low-Cost Receivers and Later Weeks" (ESP32-C5)
step numbers c5_coverage             "$PY" scripts/c5_coverage.py
step numbers c5_cross_receiver       "$PY" scripts/c5_cross_receiver.py
step numbers c5b_truncate_a          "$PY" scripts/c5b_truncate_a.py
step numbers c5b_weeks               "$PY" scripts/c5b_weeks.py
step numbers c5b_risk                "$PY" scripts/c5b_risk.py
step numbers c5b_hearing             "$PY" scripts/c5b_hearing.py   # streams both C5 files on its first run, then reads its counters

# ---------------------------------------------------------------- long (Section 7; Section 5 window sentence)
long long joint_defense_eval-he      "$PY" scripts/joint_defense_eval.py --dataset he
long long joint_defense_eval-ntufi   "$PY" scripts/joint_defense_eval.py --dataset ntufi
long long he_beamforming_gain        "$PY" scripts/he_beamforming_gain.py
long long manifold_sweep             "$PY" scripts/manifold_sweep.py             # Manifold DP across budgets, Figure 12(a) (~3 min)
long long beamdancer_sweep           "$PY" scripts/beamdancer_sweep.py           # BeamDancer per randomization, Figure 12(a)
long long prior_decoder              "$PY" scripts/prior_decoder.py              # the bound proposition and Figure 12(a) (~9 min)
long long r_eval-w16                 window_run 16
long long r_eval-w64                 window_run 64
long long r_rotation                 "$PY" scripts/r_rotation.py                 # Section 7.4, Table 5 (~37 min)
long long r_rotation-controls        "$PY" scripts/r_rotation.py --skip-main     # common grid/tones and galleries (~34 min)
long long r_rotation-null            "$PY" scripts/r_rotation.py --null-only     # pure-noise reference (~4 min)
long long r_rotation-common-grid     "$PY" scripts/r_rotation.py --common-grid-only  # Section 6.1, angles without the format's grid (~9 min)
long long r_format_equalized         "$PY" scripts/r_format_equalized.py         # MU-solicitation policy (~33 min)
long long r_format_equalized-common-grid "$PY" scripts/r_format_equalized.py --bodies rot_group_cb42_n234 --policies P2_all --append  # the same policy on the common grid (~5 min)
long long r_rotation_space           "$PY" scripts/r_rotation_space.py           # cheaper rotations (~24 min)
long long he_beamforming_gain_streams "$PY" scripts/he_beamforming_gain_streams.py # gain per stream (~16 min)
long long r_address_rotation         "$PY" scripts/r_address_rotation.py         # Appendix C, address changes replayed (~20 min)
long long r_noise_home               "$PY" scripts/r_noise_home.py               # Section 7.3, noise on the home clients (~105 min)
long long r_snr_field                "$PY" scripts/r_snr_field.py                # Section 7.4, the SNR field and sounding power (~4 min)
long long r_mu_dsnr                  "$PY" scripts/r_mu_dsnr.py                  # Section 7.4, the multi-user delta-SNR per tone (seconds)
long long r_padding_cost             "$PY" scripts/r_padding_cost.py             # Section 7.2, the cost of padding every report (seconds)
long long r_rotation_smoothing      "$PY" scripts/r_rotation_smoothing.py      # Section 7.2, the rotation under client smoothing (minutes)
long long r_raw_observer_defenses  "$PY" scripts/r_raw_observer_defenses.py  # Section 7.3, the unscaled observer under each defense (minutes)
long long r_rotation_ndp           "$PY" scripts/r_rotation_ndp.py           # Section 7.3, an observer of the sounding (~33 min)
long long r_eval_checks            "$PY" scripts/r_eval_checks.py            # Section 6.1, scaling and interval checks (~33 min)
long long r_round3_checks         "$PY" scripts/r_round3_checks.py         # Sections 7.3 and 8, shared soundings, combined defense, full-support DP-SQ (~13 min)
long long r_round4_checks         "$PY" scripts/r_round4_checks.py         # Sections 6.1, 7.1, 7.3, 7.4: timing and RSSI baseline, deeper codebooks, five rooms, zero-forcing, equivalence, five devices (~68 min)
for d in none dpsq manifold beamdancer rotation_report rotation_group rotation_fixed rotation_shared; do
  long long "defense_bench-$d" "$PY" scripts/defense_bench.py --defense "$d"   # Section 7, the test bench, one built-in defense (~1 min each)
done
long long c5_continuous              "$PY" scripts/c5_continuous.py --build --eval # Section 4 board totals, 24 middle files (~45 min)
long long c5_continuous-pairs        "$PY" scripts/c5_continuous.py --pairs        # Section 6.3, "Low-Cost Receivers and Later Weeks", every file enrolled (~3 h)

# ---------------------------------------------------------------- figures
capture_step figures fig_method_on_data      "$PY" scripts/fig_method_on_data.py
step figures fig_identity_invariant  "$PY" scripts/fig_identity_invariant.py
step figures fig_recall_strip        "$PY" scripts/fig_recall_strip.py
step figures fig_recall_per_device   "$PY" scripts/fig_recall_per_device.py
step figures fig_time_enroll         "$PY" scripts/fig_time_enroll.py
step figures fig_public_data         "$PY" scripts/fig_public_data.py
step figures fig_five_dollar         "$PY" scripts/fig_five_dollar.py         # per-device recall under the 512-byte cut, not printed in the paper
step figures fig_routine_stability   "$PY" scripts/fig_routine_stability.py   # routine stability per device, not printed in the paper
step figures fig_privacy_test        "$PY" scripts/fig_privacy_test.py
step figures fig_what_stays          "$PY" scripts/fig_what_stays.py
step figures fig_prototype_photos    "$PY" scripts/fig_prototype_photos.py
capture_step figures plot_longitudinal       "$PY" scripts/plot_longitudinal.py       # hour-of-day footprints, not printed in the paper

# ---------------------------------------------------------------- check
step check check_numbers             "$PY" release/check_numbers.py

# ---------------------------------------------------------------- pdf
if [ ! -f "$REPO_ROOT/paper.tex" ]; then
  :
elif command -v docker > /dev/null 2>&1; then
  step pdf make-docker make docker
else
  step pdf make-local make local
fi

[ "$LIST" = 1 ] && exit 0
echo
echo "$N_RUN step(s) run in $(( $(date +%s) - T_START )) s; ${#FAILED[@]} failed: ${FAILED[*]:-none}"
exit ${#FAILED[@]}
