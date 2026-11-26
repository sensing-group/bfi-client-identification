# Artifact release: "Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback"

This folder documents what is released with the
paper and how every number and figure in it is produced. It was assembled on
2026-10-25 from the paper repository at commit `ba847c3f` and the data on the
authors' machine. What remains before deposit is the archive itself: the two
data folders go to Zenodo and their DOIs replace the footnote links in the
paper.

Contents of this folder:

| File | What it is |
|---|---|
| `README.md` | this document |
| `reproduce.sh` | runs the scripts in dependency order; `--quick` redraws the figures from the CSV/JSON in `figures/` and checks the numbers |
| `check_numbers.py` | reads the key numbers from `figures/` and compares them with `expected_numbers.json` |
| `expected_numbers.json` | the values behind the numbers printed in the paper, with the file and row each one comes from |
| `MANIFEST.txt` | sha256 and size of every released data file and of the vendored code |
| `requirements.txt` | the Python packages the scripts import, pinned to the versions of the published runs |
| `decoder/angle_decode.py` | the Givens-angle decoder, copied verbatim from `esp32-csi-tomography/backend/src/core/bfi/angle_decode.py` (commit `5709e87d`, 2026-10-07) |
| `vendor/` | the code the scripts import from the two sibling repositories (`bfi` package, `run_pi_one_shot.py`, `zones_cache.json`); see "Vendored code" in Section 4 |

The data files themselves are not in the repository (7.0 GB; packed for
upload as `pi-capture.tar.gz`, 0.39 GB, and `esp32-c5.tar`, 5.2 GB). They will be
deposited on Zenodo at publication: `https://doi.org/10.5281/zenodo.23650178`
(reserved, resolves once published).

## 1. The two captures

### 1a. Pi 5 + AX210, 13 days (Section 4, 6.1, Section 6.3 and Appendix C "Truncated reports")

**What was captured.** A Raspberry Pi 5 with an Intel AX210 card in monitor
mode on 5 GHz channel 40 (80 MHz), 11 m from the household's ISP-issued
gateway, logged every Compressed Beamforming Action frame it heard on the
channel from 8 to 20 May 2026: 832,827 frames from 49 source addresses.
Of these, 832,019 are frames that the six labeled devices of the household
sent to the gateway. The other 808 come from devices outside the study: 807
frames from 42 addresses to 20 neighbouring access points, and one frame
from an unlabeled address to the gateway. Their owners did not consent to
a release, and the paper does not study them, so they are withheld (see
"Released form").
The sniffer is `tools/pi-capture/bfi_capture.py` (Scapy; systemd unit and
monitor-mode setup in the same folder). From 11:55 on 11 May to 13:36 on
14 May the capture process was stopped while the hourly archive
sync kept running, so **12 and 13 May are present as empty files** (16,384
bytes each, the `bfi` schema and no rows). The two files are kept so that
the day index of the figures (13 columns) matches the archive.

**Released form.** The pseudonymized copy at `~/bfi-archive/anon_v2/` (13 SQLite
files, 1.77 GB, plus `DATASHEET.md`, `device_labels.csv`, `load_dataset.py`,
`SHA256SUMS`), written by `~/bfi-archive/anonymise_v2.py` from the raw
capture. It holds only the 832,019 frames of the six labeled devices
towards the household's gateway, so `src_id` takes six values and `ap_id`
one (`dev_231ac9`):

```
bfi_YYYY-MM-DD.sqlite, table bfi (indexes on ts and src_id)
  ts         REAL   POSIX timestamp at capture
  src_id     TEXT   opaque device id dev_XXXXXX (see salting)
  ap_id      TEXT   opaque access-point id
  category   INT    802.11 Action category: 21 VHT, 30 HE
  subtype    INT    802.11 frame subtype (13 Action, 14 Action No Ack)
  mimo_ctrl  TEXT   MIMO control field, hex
  body_hex   TEXT   the Action body, hex, verbatim (what the decoder reads)
  body_len   INT    bytes in the body
  rssi       INT    received signal strength at the receiver, dBm, as captured
```

Rows per day: 16,693 / 23,040 / 53,326 / 37,624 / 0 / 0 / 36,511 / 122,542 /
117,231 / 115,635 / 135,268 / 116,073 / 58,884 (8 to 20 May). The sum is
832,827, the figure in Section 4, which counts every frame the receiver
heard. Rows per day in the released files: 16,693 / 23,022 / 53,250 /
37,591 / 0 / 0 / 36,506 / 122,506 / 117,202 / 115,093 / 135,261 / 116,019 /
58,876, in total 832,019 (the withheld 808 are 0 / 18 / 76 / 33 / 0 / 0 /
5 / 36 / 29 / 542 / 7 / 54 / 8 per day). An earlier build of the release
held all 832,827 rows and the RSSI in 5 dB bins (column `rssi_bin`); it was
replaced on 2026-11-12 and `xres_common` still reads such files.

**Salting.** `src_id` and `ap_id` are `dev_` plus the first three bytes of
SHA-256(salt || MAC), with a 16-byte salt held by the authors and not
released. The same address maps to the same id on every day. The raw
capture, which the analysis scripts read, carries the real addresses and
the frames of the devices outside the study; it is not released. The RSSI
of the released rows is the raw value.

**The six labeled devices.** The household labeled six addresses. The paper
and the scripts name them by OUI prefix; the release names them by pseudonymized
id. The correspondence, fixed by the frame counts of Table 2:

| Paper name | OUI prefix (scripts) | Released id | Frames | Sessions | Days | Report format |
|---|---|---|---|---|---|---|
| Apple TV, living room | `8c:26:aa` | `dev_a674c6` | 415,518 | 17 | 11 | HE, Nc 2, Nr 3, 4/2 bits, SU |
| Apple TV, bedroom | `d0:d2:b0` | `dev_f7a353` | 279,355 | 8 | 6 | VHT, Nc 2, Nr 3, 6/4 bits, SU |
| LG TV | `54:b7:bd` | `dev_ec5bcd` | 53,352 | 7 | 5 | HE, Nc 2, Nr 3, 6/4 bits, SU |
| Laptop | `64:4e:d7` | `dev_4e4b2c` | 55,926 | 6 | 4 | VHT, Nc 1, Nr 3, 9/7 bits, MU |
| Robot vacuum | `4c:b9:ea` | `dev_4c5439` | 24,028 | 5 | 4 | VHT, Nc 1, Nr 3, 9/7 bits, MU |
| Samsung TV | `b8:b4:09` | `dev_668a61` | 3,840 | 29 | 2 | VHT, Nc 2, Nr 3, 9/7 bits, MU (6/4 SU on its second day) |

Sessions are runs with no gap over 300 s that hold at least one 32-frame
window (`figures/r_device_set.json`, `session_stats`). The other 43
addresses the receiver heard are unlabeled and not in the release; 40 of
them have the locally administered bit set, and two more sounded fewer than
ten times (`figures/r_device_set.json`, `exclusions`, computed on the raw
capture).

**`device_labels.csv`** has one row per released address, the six devices
of the table above: `dev_id, class,
frame_count, lifetime_h, random_mac, label, mobility`. `class` is an
activity-rate bucket (high-activity, medium-activity, low-activity,
intermittent, occasional visitor, phone/transient), not a product label.
`label` and `mobility` carry the paper's names and mobility groups (the
LG TV reads "LG TV", "stationary"; an earlier build called it "LG TV
(relocated)", although it never moved, and `xres_common` maps the old
spelling); `random_mac` is 0 for all six. The
scripts read them in place of the OUI table (Section 4, "The anonymised
capture and the scripts").

`DATASHEET.md` describes the released files (rewritten 2026-11-12: six
source addresses, one access point, the rows per day, the column `rssi`,
what is withheld and why).

### 1b. ESP32-C5 mesh captures (Section 4 "A low-cost receiver", Section 6.3 and Appendix C "Truncated reports")

One recording by ESP32-C5 boards in the same home, reporting to the same
gateway, from 2026-07-15 to 09-08 (54.3 days; one 18-hour gap, 07-31 11:27
to 08-01 05:31 UTC), written by the mesh backend as JSON lines (one object
per reception; the same report is heard by several boards and written once
per board). The backend rotated its log every one to three days; the July
and September data of Section 6.3, "Low-Cost Receivers and Later Weeks", are the first and
the last file:

| File | Period | Lines (receptions) | Unique reports | Boards | Source addresses | Labeled devices present |
|---|---|---|---|---|---|---|
| `csi_data/bfi_captures.jsonl` (3,758,037,737 B) | 2026-09-06 to 09-08, 45 h | 3,227,364 | 185,978 | 19 | 32 | Apple TV LR, LG TV, Samsung TV |
| `csi_data/rotated/bfi_captures.jsonl.20260724T105426Z.gz` (729,694,088 B) | 2026-07-15 to 07-24, 207 h | 14,853,853 | 1,775,058 | 10 | 61 | Apple TV LR, LG TV, Samsung TV, Laptop |

A unique report is the first reception; later receptions of a byte-identical
body from the same source within 0.25 s are duplicates (their count is kept
as `n_nodes`). Record schema, from `scripts/c5_common.py`:

```
recv_ts     float   backend wall-clock receive time (epoch s)
node_mac    str     receiving ESP32-C5 board (full MAC)
node_ip     str
src_mac     str     beamformee, the reporting device (full MAC)
dst_mac     str     beamformer, the access point
node_ts_us  int     board-local microsecond clock
rssi        int     dBm at the receiving board
channel     int
bb_format   int     chip rx_ctrl.cur_bb_format (firmware enum)
subtype     int     13 Action, 14 Action No-Ack
category    int     21 VHT, 30 HE
action      int     0 = Compressed Beamforming
body_len    int     bytes of the Action body kept, CAPPED AT 512 by the firmware
length_ok   bool    header body_len equals the bytes present
body_hex    str     the Action body, drop-in for the decoder
```

**The 512-byte cap.** The firmware keeps the first 512 bytes of each Action
body (`BFI_MAX_BODY` in the ESP-IDF receiver), so wide reports lose their
upper tones: 223 of 250 tones survive for the living-room Apple TV (HE 4/2
bits), 134 of 250 for the LG TV, 134 of 234 for the bedroom Apple TV, 126 of
234 for the laptop and the vacuum, 84 of 234 for the Samsung TV
(`figures/c5b_trunc_summary.json`, `formats`). The scripts bin tones on the
full standard grid of the format and keep a bin only when every tone it
covers is present, so a kept bin means the same thing in both captures.

**Released form.** The pseudonymized copies at `~/bfi-archive/anon_c5_v2/`, written
by `~/bfi-archive/anonymise_c5_v2.py` with the **same salt** as the Pi release.
The boards heard 115,265,058 receptions in the 26 files (the two of the
table above and the 24 rotations between them), from 291 source addresses.
Released are the 108,600,711 receptions of frames that a labeled device or
one of the boards sent to the gateway: 47,697,150 from the four labeled
devices present (living-room Apple TV 30,565,708, Samsung TV 12,009,260,
LG TV 5,067,497, laptop 54,685) and 60,903,561 from the 20 boards, which
are stations of the same network and send feedback themselves. Withheld
are 6,664,347 receptions: 113,689 of frames addressed to anything other
than the gateway (1,446 from the laptop, 21,469 from four boards, 90,774
from other addresses) and 6,550,658 of frames that unlabeled addresses
sent to the gateway (6,518,183 of them from one address). None of the 267
other source addresses remains, and `dst_mac` is the gateway on every line.

| File | From | Lines heard | Lines released | Bytes |
|---|---|---|---|---|
| `bfi_c5_2026-07.jsonl.gz` | the July rotation | 14,853,853 | 14,823,639 | 706,633,489 |
| `bfi_c5_2026-09.jsonl.gz` | `bfi_captures.jsonl` | 3,227,364 | 3,226,983 | 128,493,395 |
| `bfi_c5_<stamp>.jsonl.gz`, 24 files | the 24 rotations between them, `rotated/bfi_captures.jsonl.<stamp>.gz` | 97,183,841 | 90,550,089 | see `MANIFEST.txt` |

The counts of unique reports, boards and source addresses in the table of
raw files above are those of everything the boards heard; the scripts
recompute them on whichever files they are given.

`src_mac`, `dst_mac` and `node_mac` hold `dev_` ids (the three televisions
are `dev_a674c6`, `dev_ec5bcd`, `dev_668a61` as in the table above; the
gateway is `dev_231ac9`, the `ap_id` of the Pi release); the field names and
every other byte of every line are the originals (`rssi` included, in dBm
as the board reported it), the substitution being a
regular expression on the raw line. `DATASHEET.md` in that folder holds the schema,
counts, dates and the 512-byte cap; `export_stats.json` the per-id
reception counts of the released lines and, per file, the withheld lines by
reason as counts. Point the scripts at them with `BFI_C5_LIVE`,
`BFI_C5_ROTATED` and, for the 24 middle files, `BFI_C5_MIDDLE_DIR` (Section 4). `c5b_hearing.py` places the boards by the
`node_mac` of the board layout, `vendor/zones_cache.json`, which carries the
same pseudonymized ids as the released files (the C5 datasheet names the boards by
them), so its per-board table resolves on the release.

### 1c. Public datasets (Section 6.2, 7)

Not redistributed; obtain from the original authors and point the scripts at
them with the environment variables of Section 4 below.

- He et al., real Givens-angle BFI from five rooms (`data/he_bfi/dataset/transfer and light weight/<room>/BFI.pth, y.pth`), and the He presence set at three bandwidths (`data/he_presence`).
- NTU-Fi HumanID (amplitude CSI, 14 subjects), the proxy for people.

## 2. The decoder

`decoder/angle_decode.py` parses the VHT (3-octet) and HE (5-octet) MIMO
control fields and unpacks the quantized Givens angles of a Compressed
Beamforming report body, vectorised with `numpy.unpackbits`. The MIMO control
bit layouts follow the 802.11 field definitions cross-checked against
Wireshark's `wlan.he.mimo.*` bitmasks. The angle unpacking was validated
bit-exact against the Wi-BFI reference traces (the published
`bfa_ac_mu_3x1_80.npy`, VHT MU codebook 1, Nr 3, Nc 1, 80 MHz: 200 reports x
234 subcarriers x 4 angles); that check also fixed the subcarrier count,
which the body length over-estimates on MU reports because the MU Exclusive
Beamforming Report follows the angles. On the Pi capture,
`scripts/r_common.py::verify_decoder` re-checks the first-column decode
against this module on 200 bodies of each of the seven formats seen (0
mismatches, `figures/r_device_set.json`, `decoder_check`).

The analysis scripts load the decoder from this copy
(`xres_common.ANGLE_DECODE_PY`; `BFI_ANGLE_DECODE_PY` overrides it, and a
checkout without the copy falls back to
`esp32-csi-tomography/backend/src/core/bfi/angle_decode.py`). The copy is
byte-identical to that file at the commit named above. The file carries no license header, and the
`esp32-csi-tomography` repository has no LICENSE file; the sibling
`bfi-privacy` framework is MIT (Bruno Rodrigues, 2026). **To do before
deposit:** add the license header the authors choose.

## 3. The scripts

All scripts live in `scripts/` and write to `figures/`. Runtime and peak RSS
are what the script wrote into its own `*_summary.json` on the authors'
machine (Intel Xeon W-1290, 15 GB RAM, single process, one BLAS thread);
"n.r." means the script does not record them. Every figure script reads the
CSV/JSON of the number scripts and asserts the plotted values against the
values the paper prints, so a changed number fails the figure.

### Shared modules

| Module | Role |
|---|---|
| `xres_common.py` | paths (environment variables), the decoder loader, memory helpers (`set_vm_limit`, RSS logging), the six-device ground truth |
| `r_common.py` | the Pi-capture feature pipeline: six devices, first-column Givens angles (phi11, phi21, psi21, psi31) in 32 tone bins, 320-d window vector, sessions and windows, per-day cache |
| `joint_defense_common.py` | He and NTU-Fi loaders, 802.11 Givens reconstruction, the defenses, the attackers (1-NN, SupCon encoder, invariant-reading 1-NN) |
| `c5_common.py` | streaming reader of the ESP32-C5 JSON lines, de-duplication, coverage aggregates, per-device frame cache |
| `c5b_common.py` | the C5 windows of both months on one format vocabulary, bin masks, conversion to the `r_common` layout |
| `paper_style.py` | figure style: column and text width of the ACM format, 8 pt Linux Libertine (7.5 pt smallest), Okabe-Ito colours; `save()` writes PDF and PNG |
| `device_names.py` | the six devices' three names: archive label, short key of the result files, and the Table 2 name every figure prints |

### Section 4, capture facts

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `r_common.py` (run as a script) | Pi capture (`BFI_ARCHIVE_DIR`) | per-day caches and window tables in `BFI_X_CACHE_DIR/r/` | n.r.; `--dry-run` estimates the cache size |
| `r_eval.py` (also 6.1) | r cache | `r_device_set.json`: devices, exclusions, frames, sessions (72 with a window), window spans (45/44/45/10/63/3 s), gap percentiles (p99 under 62 s), decoder check | 281 s, 572 MB |
| `c5_common.py` | both C5 files | coverage aggregates and frame caches in `BFI_X_CACHE_DIR/c5/` | 793 s + 331 s, 323 MB |
| `c5_coverage.py` | c5 aggregates, r cache | `c5_coverage.csv`, `c5_coverage_nodes.csv`, `c5_coverage_sources.csv`, `c5_coverage_summary.json`: 1.8 M and 186 k reports, 61 and 32 addresses, 10 and 19 boards, tones kept per format | n.r. (seconds) |
| `c5b_hearing.py` | both C5 files (streamed once, counters cached in `BFI_X_CACHE_DIR`), `vendor/zones_cache.json` | `c5b_hearing.csv`, `c5b_hearing_setcover.csv`, `c5b_hearing_summary.json`: 82 to 98 % of a television's reports per board, median RSSI -48 to -86 dBm, one to three boards for 95 % | stream 21 s + 97 s, 33 MB; analysis < 1 s |

### Section 5, the method

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `identity_invariant.py` | He CR1, NTU-Fi | `bfi_identity_invariant.json`: per-coordinate accuracy (71 of 174, 48 of 118 above twice chance; best single 47 / 27 %, joint 75 / 39 %), DP-SQ shift 0.26 / 0.72 of the between-target distance | 29 s, 1,281 MB |
| `fig_identity_invariant.py` | `bfi_identity_invariant.json` | `fig_identity_invariant.pdf` (Figure 6, Section 5) | seconds |
| `fig_method_on_data.py` | r cache, one Pi archive day file (32 raw bodies), `r_main.csv` | `fig_method_on_data.pdf`, `fig_method_on_data.json` (Figure "The method on one real window") | seconds, about 70 MB cache |
| `r_cost.py` | one Pi archive day | `r_cost.json`: 48 us per frame batched, 10 us per query at K = 1 | n.r., 105 MB |

### Section 6.1, identification in one home

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `r_eval.py` | r cache | `r_main.csv`, `r_per_device.csv`, `r_pair.csv`, `r_pair_per_device.csv`, `r_daypair_chance.csv`, `r_sessions.csv`, `r_eval_summary.json`: same-day 93 / 96 %, per-device recall across days, laptop vs vacuum 81 / 95 % accuracy, format baseline, chance per decision | 281 s, 572 MB |
| `r_recall_by_day.py` | r cache | `r_recall_by_day.csv`, `r_recall_by_day_summary.json` (K = 16 per day) | 78 s, 410 MB |
| `r_recall_by_day.py --drift` | r cache | `r_recall_by_day_drift.csv`, `_summary.json` (K = all, centroid, T = 16, format variants; the day strips) | 21 s, 343 MB |
| `m_motion_split.py` | r cache | `m_motion_split.csv`, `_per_device.csv`, `m_activity_by_hour.csv`, `_summary.json`: quiet/active split, at most 9 points | 165 s, 435 MB |
| `m_time_to_identify.py` | r cache | `m_time_to_identify.csv`, `_summary.json`: recall against minutes of listening, 82 % at one window to 89 % at three minutes | 10 s, 341 MB |
| `fig_recall_strip.py` | `r_recall_by_day_drift.csv`, summaries | `fig_recall_strip.pdf` (K = all) | seconds |
| `fig_recall_per_device.py` | `r_per_device.csv` | `fig_recall_per_device.pdf` (Figure 7, Section 6.1) | seconds |
| `fig_time_enroll.py` | `m_time_to_identify.csv`, `r_main.csv`, summaries | `fig_time_enroll.pdf` | seconds |

### Sections 6.2 and 6.3, public data

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `s1_calibration_size.py --dataset he` | He CR1 | `s_calibration_size.csv`, `_seeds.csv`, `s_calibration_size_he_summary.json`: 49 % raw, 39 % scaled on 20 positions | 514 s, 1,103 MB |
| `s1_calibration_size.py --dataset ntufi` | NTU-Fi | same CSVs, `s_calibration_size_ntufi_summary.json`: 82 % raw, 83 % with one session enrolled | 195 s, 861 MB |
| `s2_env_out.py` | He, five rooms | `s_env_out.csv`, `_seeds.csv`, `s_env_out_summary.json`: scaling in other rooms costs 5 to 12 points, 11 to 17 times chance | 209 s, 1,286 MB |
| `joint_defense_eval.py --dataset ntufi` | NTU-Fi | `bfi_joint_defense_ntufi.csv`, `_summary.json`: 75 % at K = 1, the value the paper reports (also `s_calibration_size` with seven calibration subjects; the calibrated 1-NN of `bfi_rebuttal_1nn.csv` gives 77 %, mean over 25 draws), encoder 66 %, DP-SQ 74 to 75 % | 759 s, 940 MB |
| `ntu_cross_condition.py` | NTU-Fi | `ntu_cross_condition.json`: enrollment from one walking condition, queries from the others, 10 seeds (68 %) | seconds |
| `fig_public_data.py` | `s_calibration_size.csv`, `bfi_joint_defense_he_summary.json`, `bfi_joint_defense_ntufi_summary.json` | `fig_public_data.pdf` (Figure 9) | seconds |

The May 2026 CSVs (`bfi_rebuttal_he.csv`, `bfi_rebuttal_1nn.csv`,
`bfi_rebuttal_dpsq_sweep.csv`, `bfi_oneshot_he.csv`, `bfi_kshot.csv`) were produced by
the `bfi-privacy` framework (`bfi/demo_rebuttal_he.py`,
`bfi/demo_rebuttal_loso.py`, `bfi/demo_oneshot_he.py`,
`bfi/demo_kshot_attack.py`), not by a script in this repository; they are
shipped in `figures/`. The same holds for the closed-set baselines,
not printed in the paper (`bfi_learned_*.csv`, `bfi/demo_learned_attacker*.py`).

### Section 6.3, "Low-Cost Receivers and Later Weeks"

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `c5_cross_receiver.py` | r cache, c5 cache | `c5_cross_receiver.csv`, `_summary.json` (Pi to board and board to Pi enrollment; not quoted in the text) | 88 s, 674 MB |
| `c5b_truncate_a.py` | Pi archive (every body cut to 512 bytes before decoding), r cache | `c5b_trunc_main.csv`, `_per_device.csv`, `_daypair.csv`, `_pair.csv`, `_pair_per_device.csv`, `_side_by_side.csv`, `c5b_trunc_summary.json`: Section 6.3, "Low-Cost Receivers and Later Weeks" (accuracy over all decisions with full reports and with reports cut), per-device recall under the cut | 267 s, 627 MB |
| `c5b_weeks.py` | c5 cache | `c5b_weeks.csv` (mean over seeds 0 to 9, with `recall_sd`), `c5b_weeks_seeds.csv` (per seed), `_summary.json`: July to September 72 % at K = 1 (LG 82, Apple TV 65, Samsung 70), 77 % at K = all, SD about 4 points; reverse 78 %; within July 70 %, within September 93 % | 400 s, 530 MB |
| `c5_continuous.py --build --eval` | the 24 middle files (`BFI_C5_MIDDLE_DIR`) + c5 cache | `c5_continuous.csv` (July session queried on every later file, mean over seeds 0 to 9), `c5_continuous_summary.json` (`recording`: 11,515,723 unique reports over 26 files, 291 source ids, 9 to 20 boards; Section 4): on the 20 files with all three televisions 57 to 78 % at K = 1, mean 66 % (the July session alone; Section 6.3, "Low-Cost Receivers and Later Weeks", uses the pairs below) | 45 min, 1.1 GB |
| `c5_continuous.py --pairs` | the c5 cache of the step above | `c5_continuous_pairs.csv` (every file enrolled, every other queried, both directions, mean over seeds 0 to 2), `c5_continuous_pairs_summary.json` (one window enrolled, the pairs with all three televisions, by days between the files): 422 pairs with all three televisions, 86 % within three days, 66 to 72 % from three to six weeks, Spearman rho -0.41 (Section 6.3, "Low-Cost Receivers and Later Weeks") | 3.2 h, 1.4 GB |
| `c5b_risk.py` | c5 cache | `c5b_risk_*.csv`, `c5b_risk_summary.json` (pooled board-only protocol; not quoted in the text) | 86 s, 341 MB |

### Section 7.1 to 7.3, the privacy test

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `joint_defense_eval.py --dataset he` | He CR1 | `bfi_joint_defense_he.csv`, `bfi_joint_defense_he_summary.json`: observer accuracy per defense, attacker and T (none 39 %, DP-SQ 26 to 31 %, fixed unitary 86 %, shared 35 %, per-report 6 % plain / 31 % invariant, per-subcarrier 5 %, keyed orthogonal 16 % invariant) | 2,428 s, 1,277 MB |
| `he_beamforming_gain.py` | He CR1 | `bfi_he_beamforming_gain.csv`, `_summary.json`: first-stream gain per defense (96 % unprotected, 87 to 94 % DP-SQ, 24 % manifold, 97 % per-subcarrier unitary with the AP holding Q) | 1,118 s, 1,194 MB |
| `manifold_sweep.py` | He CR1 | `bfi_manifold_sweep_he.csv`, `_summary.json`: Manifold DP at budgets 0.5 to 2048, plain 1-NN (T = 1, 8, 32) and first-stream gain per budget; chance only at 24 % of the gain, 22 to 34 % where 84 to 95 % is kept (Section 7.3, Figure 12(a)); its eps = 0.5 and undefended observer cells equal those of `joint_defense_eval.py` | 150 s, 1,201 MB |
| `beamdancer_sweep.py` | He CR1 | `bfi_beamdancer_sweep_he.csv`, `_summary.json`: BeamDancer's offsets (Cominelli et al., TWC 2024, Algorithm 1) at R = 0 to 100 on the He and its own codebook, plain 1-NN (T = 1, 8, 32) and first-stream gain; 37 % at 94 % of the gain, 10 % at 33 % (Section 7.3, Figure 12(a)); R = 0 on the He codebook equals the undefended cells | 103 s, 1,397 MB |
| `prior_decoder.py` | He CR1 | `bfi_prior_decoder_he.csv`, `_summary.json`: the bound of the bound proposition (Appendix B) on the 19 held-out positions (89 %; 97 % per known position), the best fixed beam learned on the other 20 (85 %), and per noise level the gain with and without a prior (Section 7.3, Figure 12(a)) | 503 s, 1,103 MB |
| `fig_privacy_test.py` | the five summary JSONs | `fig_privacy_test.pdf` | seconds |
| `fig_what_stays.py` | `r_rotation_main.csv`, `r_snr_field.csv`, `r_snr_field_summary.json`, `r_round4_checks.csv` | `fig_what_stays.pdf`: Figure 12(b), what stays readable on the home capture, with RSSI and report timing alone | seconds |
| `he_beamforming_gain_streams.py` | He CR1 | `bfi_he_beamforming_gain_streams.{csv,json}`: gain per stream for N_c = 2 (rotation per group 96.7 % / 95.6 %, sum 96.1 %; DP-SQ 84 to 94 %) | 930 s, 1,124 MB |

### The test bench for a new defense

`scripts/defense_bench.py` scores a defense against the definition of
Section 7.1 without editing any other script. It runs on the 19 held-out
positions of the He dataset with the split, seeds and observers of
`joint_defense_eval.py` and the gain of `he_beamforming_gain.py`, and prints
the accuracy of every observer (plain, unscaled and invariant, one and 32
query windows), the first-stream gain and the verdict: private (the upper
95 % bound of every observer within `--alpha` of chance), keeps the gain (at
least `--beta`), and passes (both). `--beta` is in units of the gain from unquantized feedback; the paper sets it one point under what unprotected feedback keeps, 0.95 at the 4/2 codebook of the positions.

```
python3 scripts/defense_bench.py --list                      # the built-in defenses
python3 scripts/defense_bench.py --defense rotation_group    # the one that passes (87 s, 1.5 GB)
python3 scripts/defense_bench.py --defense dpsq --param eps=0.5 --bits 6/4
python3 scripts/defense_bench.py --defense examples/defense_example.py:Dither --param cells=2
python3 scripts/defense_bench.py --selftest                  # synthetic data, no dataset needed
python3 -m pytest tests/test_defense_bench.py                # 8 tests, about 20 s
```

A new defense is a subclass of `Defense` with `report(angles, targets, rng)`,
which returns the angle values on the air and the secret the access point
keeps, and optionally `recover(reported, secret)`, which returns the matrix
the access point steers with. `examples/defense_example.py` is a complete
one in 20 lines. Each run writes `figures/bench/<defense>_b<bits>.json`.
The built-ins call the functions behind the paper's numbers and reproduce
them (4/2 bits, 10 seeds, chance 5.3 %):

| `--defense` | Observer (plain, T = 1) | Worst observer | Gain | Passes |
|---|---|---|---|---|
| `none` | 39.0 % | 62 % | 0.96 | no |
| `dpsq` (eps 0.5) | 26.1 % | 51 % | 0.89 | no |
| `manifold` (eps 512) | 34.2 % | 62 % | 0.95 | no |
| `beamdancer` (R 100) | 10.0 % | 25 % | 0.33 | no |
| `rotation_report` | 6.4 % | 51 % (invariant) | 0.97 | no |
| `rotation_group` | 5.3 % | 5.4 % | 0.97 | yes |
| `rotation_fixed` | 85.7 % | 94 % | 0.97 | no |
| `rotation_shared` | 34.8 % | 52 % | 0.97 | no |

### Section 7.4, access-point-only protection (Table 5)

All read the released archive; the rotation is emulated on the recorded reports (rebuild V_k from the angles, rotate, re-decompose, re-quantize at the device's codebook) and the protocol of Section 6.1 is re-run with its seeds.

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `r_rotation.py` | window cache + archive | `r_rotation_main.csv`, `r_rotation_per_device.csv`, `r_rotation_summary.json`: unprotected, rotation per report, per report and group, with plain and invariant observers | 2,201 s, 930 MB |
| `r_rotation.py --skip-main` | same | the common-grid and common-tone controls, `r_rotation_gallery.csv` (same-codebook galleries) | 2,042 s, 1,207 MB |
| `r_rotation.py --null-only` | same | the pure-noise reference rows | 252 s, 521 MB |
| `r_rotation.py --common-grid-only` | same | `r_rotation_common_grid.csv`, `_per_device.csv`, `_summary.json`: the unprotected reports re-quantized at 4/2 bits on the 234 tone groups every format shares (Section 6.1: 82 and 89 % across days, macro 69 and 77 %) | 512 s, 885 MB |
| `r_format_equalized.py` | same | `r_format_equalized.{csv,json}`: format classes and format-only identification as captured (P1) and with multi-user feedback solicited (P2); the standard reading it assumes is recorded in the JSON | 1,981 s, 459 MB |
| `r_format_equalized.py --bodies rot_group_cb42_n234 --policies P2_all --append` | same + the common-grid cache of `r_rotation.py --common-grid-only` | adds the rows of the same policy on the common grid to `r_format_equalized.{csv,json}` | 301 s, 518 MB |
| `r_rotation_space.py` | same + He CR1 | `r_rotation_space.{csv,json}`: 21 rotation structures x freshness policies, plain and invariant observers, gain and cost per sounding | 1,444 s, 1,809 MB |
| `r_address_rotation.py` | window cache | `r_address_rotation.csv`, `_per_device.csv`, `_summary.json`: every window stopped at an address change every 1, 10 or 60 minutes, each device on its own random phase; first re-runs `r_eval` and stops unless it matches `r_main.csv` exactly | 20 min, 2.1 GB |
| `r_noise_home.py` | window cache + archive | `r_noise_home.csv`, `_per_device.csv`, `_pair.csv`, `_short_window.csv`, `_summary.json`: DP-SQ, Manifold DP and BeamDancer on the six labeled devices and the format-matched pair, each report's angles drawn inside their cell and re-quantized at the device's codebook, and windows of 8 reports for the no-noise, budget-8 and 100% BeamDancer bodies | 105 min, 2 GB |
| `r_mu_dsnr.py` | window cache + archive | `r_mu_dsnr.csv`, `_summary.json`: the delta-SNR per tone of the multi-user reports (MU Exclusive Beamforming Report) and the stream-1 average-SNR byte as the observer's only input, on the laptop and the vacuum in their shared format | 4 s, 230 MB |
| `r_padding_cost.py` | Pi archive | `r_padding_cost.json`: body bytes of the six labeled devices as captured and padded to the longest report, per device and in total, and the feedback's share of airtime at 6 and 24 Mb/s (a bound: no transmit rate is captured) | 1 s, 37 MB |
| `r_rotation_smoothing.py` | He CR1 positions | `r_rotation_smoothing.csv`, `.json`: first-stream gain when the client smooths its channel estimate across tone groups (rank-one model), for no rotation, a rotation per report, per group and per band of 2, 4 or 8 groups, beside the invariant observer's accuracy per band from `r_rotation_space.json` (Section 7.2, Appendix C) | 3 min, 1.0 GB |
| `r_raw_observer_defenses.py` | He CR1 positions | `r_raw_observer_defenses.csv`, `.json`: the unscaled nearest-neighbour observer under every defense of Section 7, beside the scaled one, 10 seeds (Section 7.3) | 7 min, 1.1 GB |
| `r_rotation_ndp.py` | He CR1 positions, window cache + archive | `r_rotation_ndp.csv`, `.json`: an observer that also measures the rotated sounding through a drawn channel of its own and multiplies it with the report, by antennas, channel model, estimation noise and stability of that channel between enrollment and query (Section 7.3, Appendix C) | 33 min, 1.8 GB |
| `r_eval_checks.py` | window cache + archive | `r_eval_checks.csv`, `.json`: the home protocol with the scaling fitted on enrolled windows only and with raw features, intervals for accuracy and macro recall by query session and by day, and upper bounds of the rotation against the noise reference (Section 6.1, Appendix C) | 33 min, 1.2 GB |
| `r_round3_checks.py` | He CR1 positions, window cache + archive | `r_round3_checks.csv`, `.json`: two clients answering one sounding (shared soundings in the home capture found by dialog token, and pairs of positions), the rotation combined with one format and a coarser or removed SNR field against the noise reference, and our implementation of a full-support variant of DP-SQ (Sections 7.3 and 8) | 13 min |
| `r_round4_checks.py` | He positions (five rooms), window cache + archive | `r_round4_checks.csv`, `.json`: an observer of report timing and RSSI alone, the defenses at 6/4 and 9/7 bits, the defenses in all five rooms, two-user zero-forcing sum rate under each defense, an equivalence test of the rotation against the noise reference, and the home results without the Samsung TV (Sections 6.1, 7.1, 7.3, 7.4) | 68 min |
| `r_snr_field.py` | window cache + archive | `r_snr_field.csv`, `_summary.json`: the per-stream SNR bytes of each report as the observer's only input, gated by the format, and after a sounding power drop per report or per session | 4 min, 300 MB |

### Figures drawn by hand

`figures/fig_teaser_v7c.tex` (Figure 1), `fig_study_design_v2.tex` (Figure 3), `fig_sounding.tex`,
`fig_rotation.tex` and `fig_prototype.tex` are TikZ sources compiled with the
paper. `fig_prototype.tex` places the photographs `photo_pi_hat.jpg`,
`photo_pi_stack.jpg`, `photo_router.jpg`, `photo_c5.jpg`, which
`scripts/fig_prototype_photos.py` crops and tone-matches from the originals in
`figures/` (`c5-open.jpg` for the C5 panel).

### Also in `scripts/`, not behind a printed number

`reproduce.sh` still runs four scripts whose results are not printed in the paper:

| Script | Inputs | Outputs | Runtime / RSS |
|---|---|---|---|
| `fig_five_dollar.py` | `c5b_trunc_side_by_side.csv`, `c5b_trunc_per_device.csv`, `c5b_trunc_summary.json` | `fig_five_dollar.pdf`: recall across days per device with full reports and with reports cut to 512 bytes | seconds |
| `fig_routine_stability.py` | `bfi_routine_daypairs.csv` (devices as pseudonymized ids, named through `BFI_LABELS` or the id table of Section 1a; raw addresses are accepted too) | `fig_routine_stability.pdf`: day-pair cosine 0.70 / 0.66 for the Apple TVs, 0.28 LG, 0.16 / 0.07 vacuum and laptop, median 0.61 | seconds |
| `cross_day_routine_stability.py` | Pi archive (`BFI_ARCHIVE_DIR`, the paper's days) | `bfi_routine_daypairs.csv` (mac, n_days, day_a, day_b, cosine; 90 pairs; the shipped file carries the pseudonymized ids, and the released archive regenerates its rows exactly, in id order), `cross_day_routine_stability.json` (per-device stability, population median 0.611) | 2 s, 48 MB |
| `plot_longitudinal.py` | Pi archive (`BFI_ARCHIVE_DIR`, the paper's days, raw or released schema) | `fig_longitudinal.pdf`: hour-of-day footprints of the six labeled devices, one day's timeline, and burst rate against RSSI spread | 9 s |

`plot_longitudinal.py` reads the 13 days of the paper's window (11 with data)
and labels lanes and points with the device names of Table 2 on both the raw
and the released archive. Panel (b) annotates the devices first heard on the
plotted day with the time of their first frame and their days with data, both
computed. Panel (c) plots the burst rate (one over the median gap shorter than
60 s between consecutive frames) against the RSSI standard deviation; the
released archive carries the RSSI as captured (column `rssi`), so the
spreads are those of the raw capture. On files of the earlier build, whose
RSSI was binned to 5 dB (`rssi_bin`), the spreads of the stationary devices
read about 1.5 to 2.5 dB higher.

`x0_reproduce_residential.py`, `x4_mobility.py`, `x5_cross_source.py`,
`x8_mac_free_clustering.py` (`xres_*`) are the transition
experiments between the paper's first and current versions; they key on OUI prefixes, run on the raw archive only, and their
outputs (`figures/x*.csv|json`, which carried raw addresses) are not in the
repository. `s2_presence_crossbw.py`, `har_person_test.py`
(CSI-BFI-HAR, confounded with the recording block, not used),
`kshot_multiquery_table.py`, `fig_question_*.py` and the `plot_*.py`,
`make_fig_*.py`, `graphical_abstract_*.py` scripts of the May 2026 version
are kept for the record (`fig_question_result.py` also supplies the expected
recalls to `fig_recall_strip*.py`). Scripts that wrote to another machine's
paths and produced nothing the paper or this release uses were removed
(`make_figures.py`, `make_fig_kshot.py`, `make_fig_bfi_exchange.py`,
`longitudinal_extended_analysis.py`, `figures/regen_1nn/`).

## 4. Environment

**Python** 3.12.3 with numpy 2.4.1, scipy 1.17.0, matplotlib 3.10.8,
pandas 3.0.1, scikit-learn 1.8.0, torch 2.5.1 (CPU; the SupCon encoder of
`joint_defense_common.py` and `s1_calibration_size.py`), Pillow 13.0.0
(`fig_prototype_photos.py`); `requirements.txt` pins them
(`python3 -m pip install -r release/requirements.txt`; the CPU build of torch
suffices). Keep torch at 2.5.1: under torch 2.14.1 every checked number
reproduces, but the encoder's accuracies move by up to 3.6 points on real
feedback and 11.5 on the proxy, and `fig_public_data.py` stops on its expected
values. Even at 2.5.1, the encoder's accuracies on real feedback averaged over
8 or 32 query windows differ by up to 1.2 points between machines, its
one-window accuracies, which the paper and `fig_public_data.py` use, by at most
0.3.

The figures use the Linux Libertine O font
(`paper_style.py`); matplotlib falls back to DejaVu Serif when it is not
installed, which changes the text metrics but no number; `fig_method_on_data.py`
then draws Figure 5 without the arrows between its panels, for which the wider
fallback leaves no room (with the font installed it stops instead, so a layout
change cannot go unnoticed).

**Paths.** Every script reads its inputs from environment variables with
defaults relative to the repository or the home directory:

| Variable | Default | Used by |
|---|---|---|
| `BFI_ARCHIVE_DIR` | `~/bfi-archive/data` | `xres_common`, `r_common`, `fig_method_on_data`, `c5b_truncate_a`, `r_cost`, `cross_day_routine_stability`, `plot_longitudinal` |
| `BFI_LABELS` | unset (raw archive) | `xres_common`: the `device_labels.csv` of the release; set it to run on the released files |
| `BFI_X_CACHE_DIR` | `~/.cache/bfi-x-experiments` | the r and c5 caches (4.0 GB after a full run) |
| `BFI_X_DAY_FIRST`, `BFI_X_DAY_LAST` | `2026-05-08`, `2026-05-20` | the paper's capture window |
| `BFI_ANGLE_DECODE_PY` | `release/decoder/angle_decode.py`; without it `~/esp32-csi-tomography/backend/src/core/bfi/angle_decode.py` | the decoder |
| `BFI_PRIVACY_DIR`, `BFI_PRIVACY_ROOT` | `~/bfi-privacy`, `../bfi-privacy` | fallback only: `release/vendor/` is searched first for `run_pi_one_shot.py` (`xres_common`, `pi_one_shot_full_window`: window and trial constants) and the `bfi` package (`joint_defense_common`, `defense_design_space_runner`, `s2_presence_crossbw`, `x5_cross_source`, `kshot_multiquery_table`) |
| `BFI_HE_ROOT` | `data/he_bfi/dataset/transfer and light weight` | He et al., the five rooms (`s2_env_out`, `joint_defense_common`) |
| `BFI_HE_LOCATION_ROOT` | `<BFI_HE_ROOT>/Conference Room 1` | He et al., Conference Room 1 (`BFI.pth`, `y.pth`): `joint_defense_common`, `identity_invariant`, `s1_calibration_size --dataset he`, `he_beamforming_gain`, the vendored `bfi.framework.datasets.he_location`; checked by `reproduce.sh` (`env` stage) |
| `BFI_NTUFI_ROOT` | `<BFI_PRIVACY_ROOT>/data/ntu_fi/NTU-Fi-HumanID` | NTU-Fi HumanID |
| `BFI_CACHE_DIR` | system temp dir | NTU-Fi packet tensor cache |
| `ESP32_TOMO_ROOT`, `BFI_C5_LIVE`, `BFI_C5_ROTATED_DIR`, `BFI_C5_ROTATED` | `~/esp32-csi-tomography`, `csi_data/bfi_captures.jsonl`, `csi_data/rotated`, `<rotated dir>/bfi_captures.jsonl.20260724T105426Z.gz` | the C5 captures (`BFI_C5_LIVE` and `BFI_C5_ROTATED` take the released `bfi_c5_2026-09.jsonl.gz` and `bfi_c5_2026-07.jsonl.gz`); `ESP32_TOMO_ROOT` is also the fallback for the board layout when `release/vendor/zones_cache.json` is absent (`BFI_C5_ZONES_CACHE` overrides both) |
| `BFI_C5_MIDDLE_DIR` | unset: the raw rotations `bfi_captures.jsonl.<stamp>.gz` in `BFI_C5_ROTATED_DIR` | `c5_continuous`: the folder of the released `bfi_c5_<stamp>.jsonl.gz` files (the 24 between July and September) |
| `BFI_WINDOW_FRAMES` | 32 | window length in frames; 16 and 64 give `figures/r_main_w16.csv` and `r_main_w64.csv` (Section 5, window sentence) |
| `BFI_X_VM_LIMIT`, `BFI_VM_LIMIT_GB`, `BFI_MAX_RSS_MB` | 8 GB, 6 GB, 5,000 MB | address-space caps |
| `PHOTO_ROOT` | `figures/` | photographs |

**Vendored code (`vendor/`).** The scripts import code from two sibling
repositories; `vendor/` holds verbatim copies (three exceptions below), and
`scripts/vendor_paths.py` puts them ahead of the sibling checkouts on
`sys.path`, so a clone of this repository runs without either sibling. From `bfi-privacy` (MIT, Bruno
Rodrigues 2026; `vendor/LICENSE.bfi-privacy`; commit `5678894`): the import
closure of the `bfi` package as the scripts use it (`bfi/`: `__init__`,
`givens`, `quantisation`, `structural_defense`, `manifold_dp`; `bfi/datasets/`
with its four loaders; `bfi/framework/` with `attacker`, `baselines`,
`cohort_split`, `defenses`, `encoder`, `inference`, `losses`, `quantize` and
`bfi/framework/datasets/`; the two package READMEs; 26 files), and
`tools/run_pi_one_shot.py` (the paper's window, query and trial constants and
the SNR decoder). Not copied: the package's demo scripts, tests, mock loaders,
`framework_v2`, `framework_v3`, `legacy`, `mi_estimators`, `synthetic`,
`learned_projection`. From `esp32-csi-tomography` (no LICENSE file; commit
`b0dc7dc5`): `scripts/watcher/zones_cache.json` (the board layout read by
`c5b_hearing.py`, last changed in commit `a928ea05`, 2026-07-24); the decoder
is the copy in `decoder/` (Section 2). The exceptions to "verbatim": the two
dataset loaders `bfi/framework/datasets/he_location.py` and `he_presence.py`
default to a repository-relative data path instead of the authors' home
directory (the environment variables are unchanged), and `zones_cache.json`
names the boards by the pseudonymized ids of the C5 release instead of their MACs.
`MANIFEST.txt` section C lists the checksums.

**The anonymised capture and the scripts.** By default the scripts read
the raw archive: columns `src_mac`, `ap_mac`, `rssi`, the six devices matched
by OUI prefix (`xres_common.DEVICE_GT_OUI`) and the locally administered bit
read from the first address byte. With `BFI_LABELS` set to the release's
`device_labels.csv` (and `BFI_ARCHIVE_DIR` to its folder) `xres_common`
switches to the released schema: columns `src_id`, `ap_id`, `rssi`
(`rssi_bin` when the files are those of the earlier build), the
six devices from the `label` column, the random flag from `random_mac`
(`X.SRC_COL`, `X.AP_COL`, `X.RSSI_COL`, `X.device_key`, `X.is_random`). The
device index follows the order the raw archive sorts the six addresses in
(`X.DEVICE_ORDER`), so the seeded draws are the same in both modes. The
released files lack the parsed columns of the raw capture (`nc`, `nr`,
`bw_mhz`, `codebook`, `feedback_type`, `avg_snr_hex`, `chan_freq`,
`mactime`, `dialog_token`, `seq_num`, `frag_num`); none of the paper's
scripts reads them (`xres_common.build_day_cache` keeps the `has_parsed`
flag false). Every script that queries the archive directly
(`r_common`, `r_cost`, `fig_method_on_data`, `c5b_truncate_a`,
`cross_day_routine_stability`, `plot_longitudinal`) takes the column names
from `xres_common` (`SRC_COL`, `AP_COL`, `RSSI_COL`). Checked 2026-11-12
on the released files:

```
BFI_ARCHIVE_DIR=~/bfi-archive/anon_v2 BFI_LABELS=~/bfi-archive/anon_v2/device_labels.csv \
BFI_X_CACHE_DIR=<fresh> BFI_X_FIG_DIR=<fresh> python3 scripts/r_common.py && python3 scripts/r_eval.py
```

writes `r_main.csv`, `r_per_device.csv`, `r_pair.csv`,
`r_pair_per_device.csv`, `r_daypair_chance.csv` and `r_sessions.csv`
byte-identical to the files in `figures/` (per-device recall across days
with the whole session enrolled: LG TV 100.0, Apple TV bedroom 99.7, laptop
99.0, Apple TV living room 98.5, vacuum 58.5, Samsung TV 0.0; same-day
96.7 / 98.4 and other-day 82.6 / 87.9 % at K = 1 / all, T = 1); the
`exclusions` list of `r_device_set.json` is empty there, because the 43
addresses it names on the raw capture are not in the release (the rest of
that file is identical); cache 120 s and
558 MB, evaluation 302 s and 573 MB. The full `reproduce.sh` run of 2026-11-13, in a container
(`python:3.12-slim`, 2 cores, 10 GB) on a fresh clone with only the released
files and the two public datasets (`BFI_PRIVACY_ROOT`, `BFI_PRIVACY_DIR`,
`ESP32_TOMO_ROOT` pointed at a non-existent directory), ran its 70 steps in
11.8 hours and regenerated every number the check reads (635 of 635 at the time, 641 now, within
tolerance); the tests pass in the same container. The tables it writes equal
the files in `figures/` apart from run times and row order, with one
exception: the cells of the SupCon encoder in `bfi_joint_defense_he.*`, which
move by up to 1.3 points between runs (Section 4, "Environment"), while the
1-NN cells and the defense-aware encoder repeat exactly. The tables that count
frames hold the counts of the release: the boards heard 11,515,723 distinct
reports from 291 source addresses, the totals the paper's capture section
states, and the release holds 11,019,290 of them from 24 addresses, which is
what `c5_continuous_summary.json` records.
The transition experiments `x4_mobility.py` and `x8_mac_free_clustering.py`
still key on OUI prefixes and run on the raw archive only.

**Identifiers in the repository.** Every address in the tracked files is a
pseudonymized id of the release (`dev_` ids in `figures/*.csv|json`,
`vendor/zones_cache.json`, `scripts/c5b_hearing.py`); the OUI
prefixes of the six labeled devices appear only in `xres_common.DEVICE_GT_OUI`,
which maps the raw archive to the device names of Table 2. The only full MAC addresses left are the transmitter and receiver
addresses of the public CSI-BFI-HAR traces in
`figures/har_person_test_summary.json` (that dataset's own capture, not the
household). `tools/pi-capture/` installs under `/home/pi`.

**The paper.** The manuscript is not part of the public code repository; the
`pdf` stage runs only in a checkout that has `paper.tex`. There, `make docker` compiles it in the `texlive/texlive:latest`
image (digest `sha256:cb60dfe4...` at build time, pulled 2026-01); `make local`
uses the host `pdflatex` and `bibtex`. Three `pdflatex` passes and one
`bibtex` pass; output `paper.pdf`.

## 5. Running it

```
bash release/reproduce.sh --list        # the stages and steps
bash release/reproduce.sh --quick       # redraw every figure from figures/*.csv|json, check the numbers (about 1 min)
                                        # fig_method_on_data and plot_longitudinal read the capture and are skipped without it
bash release/reproduce.sh --no-long     # everything except the long stage and the C5 stream (about 40 min)
bash release/reproduce.sh               # full run: caches, numbers, long runs, figures, check, PDF (about 8 h: 5.0 h measured on two cores, plus 3.2 h for the all-pairs board step)
```

Stages run in dependency order: `env` (packages and the data paths, every
variable of the table above that names data), `cache` (`r_common.py`,
`c5_common.py`), `numbers` (`c5b_hearing.py` streams the two C5 files on its
first run), `long` (`joint_defense_eval.py` He and NTU-Fi,
`he_beamforming_gain.py`, `manifold_sweep.py`, `beamdancer_sweep.py`, `prior_decoder.py`, `c5_continuous.py`, and `r_common.py` + `r_eval.py` at
`BFI_WINDOW_FRAMES` 16 and 64 in a cache of their own, whose `r_main.csv`
become `figures/r_main_w16.csv` and `r_main_w64.csv`), `figures`, `check`,
`pdf`. Each step runs in its own process, one at a time;
a failure is logged to `release/logs/<step>.log` and the run continues. The
exit status is the number of failed steps.

`check` compares the numbers with `expected_numbers.json` at a tolerance of
half a point: recall across days per device with the whole session enrolled
(Section 6.1), pooled recall against listening time (6.1), the
calibration-size endpoints on both public datasets (6.2, 6.3), observer
accuracy and beamforming gain per defense (7), the per-device recall and the
table cells under the 512-byte cut (Section 6.3, "Low-Cost Receivers and Later Weeks"), and
the seven-week numbers on the boards (Section 6.3, "Low-Cost Receivers and Later Weeks"). Every entry names its file and row, so a mismatch points at the
script to re-run. All scripts seed their random draws, so a full re-run
reproduces the values to the printed precision.
