#!/usr/bin/env bash
# The two public datasets the paper compares against (Sections 6.2, 6.3, 7).
# Neither is redistributed with this release and neither has an anonymous
# direct-download URL, so this script cannot fetch them for you. It tells
# you where to get each one, lays the files out where the scripts expect
# them, and verifies every file against release/public_data_sha256.txt.
#
# Usage:
#   release/fetch_public_data.sh            print the instructions and verify whatever is already in place
#   release/fetch_public_data.sh --verify   verify only (exit 1 on any mismatch or missing file)
#   PUBLIC_DATA_DIR=/path  release/fetch_public_data.sh    use another root than ./data
#
# Layout under $PUBLIC_DATA_DIR (default: <repo>/data, which .gitignore keeps out of git):
#   he_bfi/bfi_dataset.zip                                   the IEEE DataPort download, renamed
#   he_bfi/dataset/transfer and light weight/<room>/BFI.pth   unpacked from it (five rooms + security/)
#   ntu_fi/NTU-Fi-HumanID/{train_amp,test_amp}/<subject>/*.mat
#
# Then run the paper's scripts with:
#   export BFI_HE_ROOT="$PUBLIC_DATA_DIR/he_bfi/dataset/transfer and light weight"
#   export BFI_HE_LOCATION_ROOT="$BFI_HE_ROOT/Conference Room 1"
#   export BFI_NTUFI_ROOT="$PUBLIC_DATA_DIR/ntu_fi/NTU-Fi-HumanID"
# (BFI_HE_ROOT alone is not enough: joint_defense_common.py reads BFI_HE_LOCATION_ROOT.)
set -u

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$RELEASE_DIR")"
ROOT="${PUBLIC_DATA_DIR:-$REPO_ROOT/data}"
SUMS="$RELEASE_DIR/public_data_sha256.txt"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

if [ "$VERIFY_ONLY" = 0 ]; then
cat <<EOF
Public datasets for "Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback"
Target folder: $ROOT

A. He et al., Wi-Fi Beamforming Feedback (BFI) Localization Dataset
   Record:   https://ieee-dataport.org/documents/wi-fi-beamforming-feedback-bfi-localization-dataset
   DOI:      10.21227/47h2-3p42   (citation author Zhiyuan He; submitted by Shanyi Ke; created 2025-10-29)
   Licence:  none stated on the record page (checked 2026-10-25); use under IEEE DataPort's terms and cite the record.
   Access:   the download needs an IEEE DataPort account with a subscription (IEEE Society members
             log in through the site's SAML login); there is no anonymous URL, so no download here.
   Steps:    1. log in and download "BFI Dataset.zip" (1.8 GB, 1,932,219,735 bytes);
             2. mkdir -p "$ROOT/he_bfi" && mv "BFI Dataset.zip" "$ROOT/he_bfi/bfi_dataset.zip";
             3. cd "$ROOT/he_bfi" && unzip bfi_dataset.zip
                (gives dataset/transfer and light weight/{Conference Room 1,Conference Room 2,Lab 1,Lab 2,Rooftop}/{BFI,y,y_pos}.pth
                 and dataset/security/Antenna pair_*.pth; 3.8 GB unpacked, the scripts read only the five rooms).

B. NTU-Fi HumanID (14 subjects, amplitude CSI), the SenseFi processed release
   Paper:    Yang et al., "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing",
             Patterns 4(3), 2023, doi:10.1016/j.patter.2023.100703; the HumanID set is from Wang et al. (CAUTION, 2022).
   Code:     https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark (MIT); the README links the data.
   Data:     Google Drive folder https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt
   Licence:  none stated for the data (checked 2026-10-25); cite SenseFi and the original dataset paper.
   Access:   a shared Google Drive folder; it needs a browser or a Drive client (gdown works for the
             folder: pip install gdown && gdown --folder <url>), so it is not fetched here.
   Steps:    1. download NTU-Fi-HumanID (the README says to unzip on Linux to avoid decoding errors);
             2. mkdir -p "$ROOT/ntu_fi" && mv NTU-Fi-HumanID "$ROOT/ntu_fi/"
                (840 .mat files, 4,596,641,280 bytes: train_amp/ and test_amp/, 14 subject folders each).

Verifying what is in place under $ROOT ...
EOF
fi

if ! command -v sha256sum >/dev/null 2>&1; then echo "sha256sum not found" >&2; exit 2; fi
fail=0
for part in he_bfi NTU-Fi-HumanID; do
  case "$part" in
    he_bfi) base="$ROOT"; prefix="he_bfi/";;
    NTU-Fi-HumanID) base="$ROOT/ntu_fi"; prefix="NTU-Fi-HumanID/";;
  esac
  n_expected=$(awk -v p="$prefix" '!/^#/ && index($3,p)==1' "$SUMS" | wc -l)
  if [ ! -d "$base/${prefix%/}" ]; then
    printf "%-16s MISSING  (%d files expected under %s)\n" "$part" "$n_expected" "$base/${prefix%/}"
    fail=$((fail + 1)); continue
  fi
  # sha256sum -c wants "hash  path"; paths with spaces are fine in that format
  out=$(awk -v p="$prefix" '!/^#/ && index($3,p)==1 { $1=$1; h=$1; $1=""; $2=""; sub(/^  /,""); print h "  " $0 }' "$SUMS" \
        | (cd "$base" && sha256sum -c --quiet 2>&1))
  rc=$?
  if [ $rc -eq 0 ]; then
    printf "%-16s ok       (%d files)\n" "$part" "$n_expected"
  else
    printf "%-16s MISMATCH\n%s\n" "$part" "$out"
    fail=$((fail + 1))
  fi
done
exit $fail
