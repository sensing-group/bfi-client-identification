# C5b: what the 5-dollar listener achieves

A = Pi/AX210, May, six devices, full bodies. B = ESP32-C5 mesh, July (10 boards), September (19 boards), bodies cut at 512 B. Recall = 1-NN on first-column Givens angles, one labelled session enrolled, chance = 1/|gallery|.

## T1. The 512-byte cut emulated on A (`c5b_trunc_*`)

Kept tones, from bits per tone: Apple TV LR 223/250, LG 134/250, Apple TV BR 134/234, laptop and vacuum 126/234, Samsung MU Nc=2 84/234. "common" = the 11 bins every device covers; "none" = all 320 dims, missing bins at the pooled mean (the cut pattern then leaks format). Windows unchanged (25948).

| angles | K | T | untruncated | common | none | chance |
|---|---|---|---|---|---|---|
| same day | 1 | 1 | 0.967 | 0.949 | 0.963 | 0.33 |
| same day | all | 32 | 0.993 | 0.981 | 0.981 | 0.34 |
| other day, any | 1 | 1 | 0.826 | 0.682 | 0.710 | 0.17 |
| other day, any | all | 32 | 0.936 | 0.788 | 0.800 | 0.17 |
| other day, any, angles+format | all | 32 | 0.950 | 0.938 | 0.960 | 0.17 |

Per device, other day any, K=all, T=1, untruncated -> common: LG 1.00 -> 1.00, Apple TV BR 0.997 -> 0.983, laptop 0.99 -> 0.83, Apple TV LR 0.985 -> 0.731, vacuum 0.585 -> 0.375, Samsung 0.00 -> 0.00. The cross-day loss of 10 to 15 points falls on the Apple TV LR and the two mobile devices; same day and the format-gated attack lose under 2 points.

Laptop vs vacuum (same format, 17 common bins): other day any, angles, K=1 0.811 -> 0.851, K=all 0.946 -> 0.957; same day 1.00 -> 0.98/1.00. The pair survives the cut.

## T2. Seven weeks (`c5b_weeks.*`)

Three TVs, 17 common bins, horizon 44 to 54 days, chance 0.33. Format scores 1.00 because the three TVs use three different formats. The Samsung changed format between A (VHT 9/7 MU) and B and not within B (VHT 2/3 6/4 SU in all 4161 windows).

| angles, T=1 | K | LG | Apple TV LR | Samsung | macro |
|---|---|---|---|---|---|
| July -> Sept | 1 | 0.90 | 0.54 | 0.85 | 0.77 |
| Sept -> July | all | 0.87 | 0.96 | 0.53 | 0.79 |
| July within, other days | all | 1.00 | 0.52 | 0.58 | 0.70 |
| Sept within, other days | all | 0.77 | 1.00 | 1.00 | 0.92 |

Angles alone hold 0.67 to 0.79 across seven weeks, but the Apple TV LR and the Samsung sit near chance within July, so the within-month floor is as weak as the cross-month number.

## T3. Where the listener must be (`c5b_hearing*`)

Every September board hears 82 to 95% of each TV's reports (Apple TV 0.91 to 0.95, LG 0.82 to 0.92, Samsung 0.91 to 0.98 on 18 boards; one board heard the Samsung once). Median RSSI -48 to -86 dBm; the Samsung decodes at -78 dBm median. Boards for 95% of a device's reports: Apple TV 2, LG 3, Samsung 1 (99%: 5, 9, 3); July 2, 2, 1.

No board-to-room map exists in the tomography repo: all boards are in the living room (PI scope rule), `scripts/watcher/zones_cache.json` gives relative MDS positions and k-means zones, three boards carry PI placement notes (entrance shelf, right of TV, kitchen). The bedroom question cannot be answered: the bedroom Apple TV is absent from both B months (off or unheard, indistinguishable). The farthest board (x = -4.5 m) hears 95/87/98%.

## T4. One number (`c5b_risk_*`)

B only, 3 TVs, 13 days, 17 bins, r_eval protocol, chance 0.33. Same day, K=1: LG 0.96, Apple TV 0.95, Samsung 0.88 (pooled 0.94; 0.98 at K=16). Other day any, K=1: LG 0.90, Apple TV 0.62, Samsung 0.56 (pooled 0.71; 0.76 at K=all). Format alone 1.00. Recall is flat from T=1 (0.75 min of listening) to T=32 (28 min): one window reaches the plateau, 0.73 to 0.76.

## Files, runtime, memory

`scripts/c5b_{truncate_a,weeks,risk,hearing,common}.py`; `figures/c5b_*` (CSV + summary JSON per task; `c5b_trunc_side_by_side.csv` joins truncated and untruncated); caches `~/.cache/bfi-x-experiments/c5b/`. T1 cache 3 min + eval 267 s, peak 627 MB (691 MB first run); T2 40 s, 530 MB; T3 118 s, 33 MB; T4 86 s, 341 MB. Single process, streamed, `set_vm_limit`, `--dry-run` on T1 and T3.

## Caveats

The Samsung has 100 windows in A and is at chance cross-day with or without truncation. September has 3 days and 2 to 4 blocks per TV. B galleries are three devices with three formats, so only the angle numbers inform. A and B differ in month as well as receiver; receiver effects cannot be isolated.
