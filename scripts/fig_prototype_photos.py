#!/usr/bin/env python3
"""
Prototype photographs for the capture section, one image per panel; LaTeX
assembles them with subcaption so panel labels and callouts use the
paper's own font. No text is drawn into the pixels.

Per panel: load the original, rotate to level the subject where needed,
crop to the scene, blur what must stay private, stretch contrast lightly,
downsample and sharpen. Same procedure and constants as the Less is More
paper (its fig_deployments_photos.py), so the two
papers' photographs look alike.

Only the Pi 5 capture hardware and the router of the 13-day capture
appear here. The ESP32-C5 probe is a separate measurement.

Iteration log (figure-generator workflow):
  pass 1  framing: full-height stack crop without the burnt-in labels,
          router with the brand shown (PI decision); levelling the hat
          photo by rotation failed (perspective), so it keeps its framing
  pass 2  callouts: none in the pixels; LaTeX draws them (fig_prototype.tex)
  pass 3  tone: matched white balance and contrast across the three panels
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

PAPER_ROOT = Path(__file__).resolve().parent.parent
PHOTO_ROOT = Path(os.environ.get("PHOTO_ROOT", PAPER_ROOT / "figures"))
OUT_DIR = PAPER_ROOT / "figures"

PANEL_LONG_PX = 1400
JPEG_QUALITY = 90
AUTOCONTRAST_CUTOFF = 0.1
BLUR_RADIUS_PX = 40
SHARPEN = dict(radius=1.0, percent=35, threshold=3)


@dataclass
class Panel:
    src: Path
    out: str
    crop: tuple[float, float, float, float]      # x0, y0, x1, y1 of the (rotated) original, normalised
    rotate_deg: float = 0.0                      # counter-clockwise, applied before the crop
    blurs: list[tuple[float, float, float, float]] = field(default_factory=list)
    soften_glare: bool = False


PANELS = [
    # Pi 5 with the PCIe-to-M.2 hat and the Intel AX210 card. The shot has
    # perspective, so rotation cannot level it (tried -14..-32 deg: fill
    # triangles and a still-tilted board); kept as the photographer framed it.
    Panel(PHOTO_ROOT / "fig_testbed_hat-old.jpg", "photo_pi_hat.jpg",
          crop=(0.04, 0.10, 0.98, 0.95)),
    # The deployed two-Pi stack at full height; the right strip carried burnt-in labels.
    Panel(PHOTO_ROOT / "fig_testbed-orig-backup.jpg", "photo_pi_stack.jpg",
          crop=(0.02, 0.05, 0.74, 0.98), soften_glare=True),
    # The ESP32-C5 board of the two later captures, enclosure open (PI 2026-10-24: use the FuSo photo).
    Panel(PHOTO_ROOT / "c5-open.jpg", "photo_c5.jpg",
          crop=(0.30, 0.02, 0.98, 0.98)),
    # The ISP gateway of the capture (PI 2026-10-24: show the brand).
    Panel(PHOTO_ROOT / "fig_testbed_router-old.jpg", "photo_router.jpg",
          crop=(0.18, 0.08, 0.95, 0.92)),
]


# Status-LED glare (pass 3): saturated bright green pixels are desaturated and capped.
GLARE_HUE = (60, 120)                    # PIL HSV hue range, 0-255 scale
GLARE_MIN_VALUE = 120
GLARE_MIN_SAT = 90
GLARE_SAT_FACTOR = 0.35
GLARE_VALUE_CAP = 200
WHITE_TARGET = 236                      # grey level the brightest 2% of each panel is mapped to
WHITE_PERCENTILE = 98


def soften_led_glare(img: Image.Image) -> Image.Image:
    """Desaturate and cap the bloom around the status LEDs; the rest is untouched."""
    import numpy as np
    a = np.asarray(img.convert("HSV")).astype("float32")
    h, s_, v = a[..., 0], a[..., 1], a[..., 2]
    glare = (h > GLARE_HUE[0]) & (h < GLARE_HUE[1]) & (v > GLARE_MIN_VALUE) & (s_ > GLARE_MIN_SAT)
    s_[glare] *= GLARE_SAT_FACTOR
    v[glare] = np.minimum(v[glare], GLARE_VALUE_CAP)
    return Image.fromarray(a.astype("uint8"), "HSV").convert("RGB")


def match_white(img: Image.Image) -> Image.Image:
    """Scale each channel so its bright percentile lands on one neutral level (pass 3)."""
    import numpy as np
    a = np.asarray(img).astype("float32")
    hi = np.percentile(a.reshape(-1, 3), WHITE_PERCENTILE, axis=0)
    a = np.clip(a * (WHITE_TARGET / np.maximum(hi, 1.0)), 0, 255)
    return Image.fromarray(a.astype("uint8"))


def render(p: Panel) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(p.src)).convert("RGB")
    if p.rotate_deg:
        img = img.rotate(p.rotate_deg, resample=Image.BICUBIC, expand=False)
    W, H = img.size
    for x0, y0, x1, y1 in p.blurs:
        box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(BLUR_RADIUS_PX)), box[:2])
    cx0, cy0, cx1, cy1 = p.crop
    img = img.crop((int(cx0 * W), int(cy0 * H), int(cx1 * W), int(cy1 * H)))
    img = ImageOps.autocontrast(img, cutoff=AUTOCONTRAST_CUTOFF, preserve_tone=True)
    img = match_white(img)
    if p.soften_glare:
        img = soften_led_glare(img)
    scale = PANEL_LONG_PX / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(**SHARPEN))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in PANELS:
        img = render(p)
        img.save(OUT_DIR / p.out, quality=JPEG_QUALITY)
        print(f"{p.out}: {img.width}x{img.height} from {p.src.name}")


if __name__ == "__main__":
    main()
