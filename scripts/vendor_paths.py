"""Where the paper scripts find the code of the two sibling repositories.

`release/vendor/` holds copies of what the scripts import from
`bfi-privacy` (the `bfi` package and `tools/run_pi_one_shot.py`) and from
`esp32-csi-tomography` (`scripts/watcher/zones_cache.json`);
`release/decoder/` holds the Givens-angle decoder. The copies come first.
A checkout without them falls back to the sibling repositories named by
the environment variables (`BFI_PRIVACY_DIR`, `BFI_PRIVACY_ROOT`,
`ESP32_TOMO_ROOT`, `BFI_ANGLE_DECODE_PY`) with their original defaults,
which the importing modules keep.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "release" / "vendor"
DECODER_DIR = REPO_ROOT / "release" / "decoder"
VENDOR_DECODER = DECODER_DIR / "angle_decode.py"
VENDOR_ZONES_CACHE = VENDOR_DIR / "zones_cache.json"


def add_bfi_privacy(fallback_root: Path) -> None:
    """Put the `bfi` package and `run_pi_one_shot` on sys.path: the vendored
    copy first, the sibling repository (its root for `bfi`, its `tools/`
    for `run_pi_one_shot`) behind it."""
    for p in (fallback_root / "tools", fallback_root, VENDOR_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def vendored_or(vendored: Path, fallback: Path) -> Path:
    """The vendored file when the checkout has it, else the sibling path."""
    return vendored if vendored.exists() else fallback
