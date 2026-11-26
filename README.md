# Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback

Companion code for the paper

> Karim Khamaisi and Bruno Rodrigues. *Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback.*
> Under submission, 2026.

WiFi clients return compressed beamforming feedback (BFI) to their access
point in plaintext, and anyone in range can read it. The paper defines when
that feedback is private (an observer that hears each labeled device once and
trains no model can only guess, while the access point keeps 95 % of its
beamforming gain) and tests the published defenses against the definition on
a 13-day capture of one home and on public data. Noise at the client hides a
device only where it ruins the steering. A rotation of the sounding, fresh
per subcarrier group, hides the angles at 97 % of the gain, yet the frame
still states its format and signal-to-noise ratio, which identify devices on
their own.

## Test a defense of your own

`scripts/defense_bench.py` scores any defense against the definition: write
one class with `report` (what goes on the air) and, if the access point holds
a secret, `recover` (what it steers with), and the bench returns the accuracy
of every observer, the gain kept and the verdict.

```sh
python3 scripts/defense_bench.py --list
python3 scripts/defense_bench.py --defense rotation_group
python3 scripts/defense_bench.py --defense examples/defense_example.py:Dither --param cells=2
python3 scripts/defense_bench.py --selftest      # synthetic data, no dataset needed
```

The built-in defenses reproduce the paper's Section 7; see "The test bench
for a new defense" in [`release/README.md`](release/README.md).

## Where things are

| What | Where |
|---|---|
| How every number and figure is produced, environment, data paths | [`release/README.md`](release/README.md) |
| One-command reproduction (`--quick`, `--no-long`, full) | [`release/reproduce.sh`](release/reproduce.sh) |
| The 641 key numbers checked against the result tables | [`release/check_numbers.py`](release/check_numbers.py), [`release/expected_numbers.json`](release/expected_numbers.json) |
| Checksums of the released data and vendored code | [`release/MANIFEST.txt`](release/MANIFEST.txt) |
| The two captures (Pi 5 + AX210, 13 days; ESP32-C5 mesh, 54 days in 26 files) | Zenodo, [doi:10.5281/zenodo.23650178](https://doi.org/10.5281/zenodo.23650178) (published with the paper; files, checksums and metadata in [`release/zenodo/`](release/zenodo/)) |
| Public datasets used for comparison (He et al. BFI localization; NTU-Fi HumanID) | not redistributed; [`release/fetch_public_data.sh`](release/fetch_public_data.sh) explains how to obtain them and verifies the files |
| The defense test bench, an example defense and its tests | `scripts/defense_bench.py`, `examples/`, `tests/` |
| Analysis and figure scripts | `scripts/` (shared modules `xres_common.py`, `r_common.py`, `c5_common.py`, `joint_defense_common.py`, `paper_style.py`) |
| The Givens-angle decoder | [`release/decoder/angle_decode.py`](release/decoder/angle_decode.py) |
| Vendored code from the sibling repositories (`bfi` package, MIT) | `release/vendor/` |
| The Pi capture stack (Scapy sniffer, systemd unit, monitor-mode setup) | `tools/pi-capture/` |
| Result tables the paper reads (CSV/JSON) and the figures (PDF/PNG) | `figures/` |

## Quick start

```sh
git clone https://github.com/sensing-group/bfi-client-identification.git
cd bfi-client-identification
bash release/reproduce.sh --quick          # redraw the figures from figures/*.csv|json and check the 641 numbers
```

For a full re-run, download the two captures from the Zenodo record, set
`BFI_ARCHIVE_DIR`, `BFI_LABELS`, `BFI_C5_LIVE`, `BFI_C5_ROTATED` and
`BFI_C5_MIDDLE_DIR` to the downloaded files, obtain the two public datasets as described in
`release/fetch_public_data.sh`, and run `release/reproduce.sh`. Section 4 of
`release/README.md` lists every environment variable; the full run takes
about 8 hours (5.0 hours measured in a clean python:3.12 container on two cores,
plus 3.2 hours for the all-pairs board step) and under 1.5 GB of RAM per step.

## Licence

Code: MIT ([`LICENSE`](LICENSE)); the vendored `bfi` package keeps its own
MIT notice (`release/vendor/LICENSE.bfi-privacy`). Data, including the
result tables under `figures/`: CC BY 4.0
([`release/LICENSE-DATA.md`](release/LICENSE-DATA.md)). The public datasets
keep their owners' terms.

## Ethics

The captures come from the authors' own home with the occupants' consent.
Only BFI Action bodies were stored; addresses are released pseudonymized
(keyed hash; the key is not released). See Appendix A of the paper.
