"""Color perception — synthesize the scopes a colorist reads, because Resolve's API exposes NONE.

This is the moat. The grading *action* surface (CDL/LUT/groups/DRX) is narrow and well-trodden;
the hard, differentiating part of agentic color is **perception** — and Resolve has no scripting
API for its scopes. So film-grip computes them itself from frame pixels, reducing a frame to the
numbers an LLM/VLM can reason over and, crucially, **verify against** (the perceive→propose→apply→
verify loop that separates a real grading tool from the consumer "text→LUT" generators):

* **RGB parade** — per-channel percentiles (1/10/50/90/99): black point, shadows, mids, highlights,
  white point. The balance across R/G/B at each level is the white-balance read.
* **Luma waveform** — min/median/max + clipped(>=235)/crushed(<=16) fractions and flags.
* **Vectorscope** — dominant hue angle + saturation magnitude in the Cb/Cr plane, plus distance
  from the skin-tone line (the colorist's key reference).
* **White balance** — gray-point neutrality → a warm/cool/green/magenta cast read.
* **Exposure** — mid-grey placement → under/ok/over.

``analyze_rgb`` is pure numpy (frame in → numbers out), so it is unit-testable on synthetic frames
with known statistics — correctness is provable offline, no editor and no ffmpeg needed. Frame
extraction (``frame_rgb``) and the optional scope PNG (``render_scopes_png``) use ffmpeg, mirroring
the contact-sheet perception already in :mod:`filmgrip.perception.frames`.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

try:                                  # numpy powers the pixel statistics; optional install.
    import numpy as np
except ImportError:                   # pragma: no cover - exercised only on a numpy-less install
    np = None

from .transcribe import PerceptionUnavailable, ffmpeg_path

# Rec.709 luma + chroma coefficients.
_LUMA = (0.2126, 0.7152, 0.0722)
CLIP_THRESHOLD = 235      # 8-bit; >= this = highlights at risk of clipping
CRUSH_THRESHOLD = 16      # <= this = shadows at risk of crushing
# Vectorscope skin-tone line ("I-bar"): skin tones of every ethnicity cluster along ~115° in the
# Cb/Cr plane. Distance from it is the single most useful color-balance cue for footage with people.
SKIN_TONE_ANGLE_DEG = 115.0

_SCOPE_WIDTH = 160        # frames are downscaled for analysis; color stats are aspect-insensitive
_SCOPE_HEIGHT = 90


def _need_numpy():
    if np is None:
        raise PerceptionUnavailable(
            "color scopes need numpy — install it with: pip install 'film-grip[color]'")


def analyze_rgb(arr: Any) -> dict:
    """Reduce an ``(H, W, 3)`` uint8 RGB array to a colorist's scope reading (a JSON-able dict).

    Pure function — the heart of color perception, deliberately decoupled from ffmpeg so it can be
    unit-tested on synthetic frames. Raises if numpy isn't available or the array is the wrong shape.
    """
    _need_numpy()
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"analyze_rgb expects an (H, W, 3) array, got shape {a.shape}")
    h, w = int(a.shape[0]), int(a.shape[1])
    rgb = a[:, :, :3].astype(np.float64)
    r, g, b = rgb[:, :, 0].ravel(), rgb[:, :, 1].ravel(), rgb[:, :, 2].ravel()

    def pct(channel) -> dict:
        p = np.percentile(channel, [1, 10, 50, 90, 99])
        return {"p1": _r(p[0]), "p10": _r(p[1]), "p50": _r(p[2]), "p90": _r(p[3]), "p99": _r(p[4])}

    luma = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b
    n = luma.size
    crush_frac = float(np.count_nonzero(luma <= CRUSH_THRESHOLD)) / n
    clip_frac = float(np.count_nonzero(luma >= CLIP_THRESHOLD)) / n
    lmed = float(np.median(luma))

    # Vectorscope: Rec.709 chroma, normalized to 0..1 so angles/magnitude are encoding-stable.
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    cb = -0.1146 * rn - 0.3854 * gn + 0.5 * bn
    cr = 0.5 * rn - 0.4542 * gn - 0.0458 * bn
    mean_cb, mean_cr = float(np.mean(cb)), float(np.mean(cr))
    hue_deg = float(np.degrees(np.arctan2(mean_cr, mean_cb))) % 360.0
    sat_mag = float(np.mean(np.sqrt(cb ** 2 + cr ** 2)))
    skin_delta = _angle_diff(hue_deg, SKIN_TONE_ANGLE_DEG)

    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    hist = [int(c) for c in np.histogram(luma, bins=8, range=(0, 255))[0]]

    return {
        "size": [w, h],
        "parade": {"r": pct(r), "g": pct(g), "b": pct(b)},
        "luma": {
            "min": _r(float(np.min(luma))), "median": _r(lmed), "max": _r(float(np.max(luma))),
            "p10": _r(float(np.percentile(luma, 10))), "p90": _r(float(np.percentile(luma, 90))),
            "crush_frac": _r(crush_frac, 4), "clip_frac": _r(clip_frac, 4),
            "crushed": bool(crush_frac > 0.5), "clipped": bool(clip_frac > 0.1),
        },
        "vectorscope": {
            "hue_deg": _r(hue_deg), "saturation": _r(sat_mag, 4),
            "skin_tone_delta_deg": _r(skin_delta),
        },
        "white_balance": {
            "r_mean": _r(r_mean), "g_mean": _r(g_mean), "b_mean": _r(b_mean),
            "cast": _cast(r_mean, g_mean, b_mean),
        },
        "exposure": {"mid": _r(lmed / 255.0, 3), "verdict": _exposure_verdict(lmed)},
        "histogram": hist,
    }


def frame_rgb(media_path: str, at_s: float = 0.0, *, width: int = _SCOPE_WIDTH,
              height: int = _SCOPE_HEIGHT) -> Any:
    """Extract one frame at ``at_s`` (media seconds) as an ``(H, W, 3)`` uint8 numpy array, via
    ffmpeg raw rgb24 on stdout (no PNG decoder / imaging dependency)."""
    _need_numpy()
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable(
            "ffmpeg is required for scope extraction — install it (e.g. `brew install ffmpeg`).")
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, at_s):.3f}", "-i", media_path,
           "-frames:v", "1", "-vf", f"scale={width}:{height}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    expected = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < expected:
        raise PerceptionUnavailable(
            f"ffmpeg could not extract a frame at {at_s:.2f}s from '{media_path}': "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(height, width, 3)


def analyze_frame(media_path: str, at_s: float = 0.0) -> dict:
    """Extract a frame and return its scope reading, with the source/time stamped in."""
    report = analyze_rgb(frame_rgb(media_path, at_s))
    report["source"] = os.path.basename(media_path)
    report["at_s"] = round(float(at_s), 3)
    return report


def render_scopes_png(media_path: str, at_s: float, out_png: str) -> str:
    """Best-effort visual scope image (waveform + vectorscope + histogram stacked) via ffmpeg's own
    scope filters — a human-readable companion to the numbers. Raises on ffmpeg failure."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable("ffmpeg is required to render scopes.")
    fc = ("[0:v]scale=320:180,split=3[a][b][c];"
          "[a]waveform=intensity=0.2:mirror=1:components=7:display=overlay,scale=320:180[w];"
          "[b]vectorscope=mode=color3:graticule=green,scale=180:180,pad=320:180:70:0[v];"
          "[c]histogram=display_mode=stack,scale=320:180[h];"
          "[w][v]vstack[wv];[wv][h]vstack[out]")
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-ss", f"{max(0.0, at_s):.3f}", "-i", media_path,
         "-frames:v", "1", "-filter_complex", fc, "-map", "[out]", out_png],
        capture_output=True, timeout=120)
    if proc.returncode != 0 or not os.path.isfile(out_png):
        raise PerceptionUnavailable(
            f"ffmpeg could not render scopes: {proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return out_png


# --------------------------------------------------------------------------- helpers
def _r(x: float, ndigits: int = 2) -> float:
    return round(float(x), ndigits)


def _angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return _r(min(d, 360.0 - d))


def _cast(r: float, g: float, b: float, tol: float = 6.0) -> str:
    """Classify a color cast from channel means (neutral within ``tol`` 8-bit levels)."""
    if max(r, g, b) - min(r, g, b) <= tol:
        return "neutral"
    if r >= g and r >= b:
        return "warm"        # red-dominant
    if b >= r and b >= g:
        return "cool"        # blue-dominant
    if g >= r and g >= b:
        return "green"
    return "magenta"


def _exposure_verdict(median_luma: float) -> str:
    if median_luma < 60:
        return "under"
    if median_luma > 195:
        return "over"
    return "ok"


def report_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
