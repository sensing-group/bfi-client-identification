# Data licence: CC BY 4.0

The data released with *Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback* is
licensed under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

Copyright (c) 2026 Karim Khamaisi and Bruno Rodrigues.

## What this covers

1. The 13-day Pi 5 + AX210 capture, anonymised copy: `bfi_2026-05-08.sqlite`
   to `bfi_2026-05-20.sqlite`, `device_labels.csv`, `DATASHEET.md`,
   `load_dataset.py` (Zenodo record 10.5281/zenodo.23650178).
2. The two ESP32-C5 mesh captures, anonymised copy: `bfi_c5_2026-07.jsonl.gz`,
   `bfi_c5_2026-09.jsonl.gz`, `export_stats.json`, `DATASHEET.md` (same
   Zenodo record).
3. The result tables in `figures/*.csv` and `figures/*.json` of the code
   repository, which the paper's numbers are read from.

`load_dataset.py` is also available under the MIT licence of the code
repository (`LICENSE` at its root), whichever suits the reuse better.

## What it does not cover

- The public datasets the paper uses for comparison are not redistributed
  here and keep their owners' terms: the Wi-Fi BFI Localization Dataset of
  He et al. (IEEE DataPort, doi:10.21227/47h2-3p42, subscription access,
  no licence stated on the record page) and NTU-Fi HumanID of Yang et al.
  (SenseFi, Patterns 2023; Google Drive distribution, no licence stated).
  See `release/fetch_public_data.sh`.
- The photographs in `figures/photo_*.jpg` and `figures/fig_testbed*.jpg`
  are the authors' own and are released under the same CC BY 4.0 terms.
- The paper manuscript is not part of the data
  release.

## Attribution

Cite the paper and the Zenodo record:

> K. Khamaisi and B. Rodrigues. Hide the Client, Keep the Beam: What It Takes to Protect WiFi Beamforming Feedback.
> Under submission, 2026.
> Data: https://doi.org/10.5281/zenodo.23650178

## Privacy notice

The captures come from the authors' own household with the occupants'
informed consent. Device and access-point addresses are pseudonymized
(`dev_XXXXXX`, SHA-256 of a 16-byte secret salt and the
address); the salt is not released. RSSI in the Pi capture is binned to 5 dB.
The beamforming report bodies are released verbatim because they are the
object of study. Do not attempt to re-identify the household or its devices.
