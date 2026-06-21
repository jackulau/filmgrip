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
# Vectorscope skin-tone line ("I-bar"): skin tones of every ethnicity cluster along the YIQ +I axis
# (they differ in saturation/brightness, not hue). Distance from it is the single most useful
# color-balance cue for footage with people. In this module's ``atan2(Cr, Cb)`` convention a 75% red
# bar plots at ~103° and the +I (flesh) axis at ~123° — real skin samples measure 124-129°. The old
# value 115° was wrong and biased ``skin_tone_delta_deg`` high by ~8°. See
# docs/research/color-science.md §1.4 (validated against measured skin patches on this machine).
SKIN_TONE_ANGLE_DEG = 123.0

_SCOPE_WIDTH = 160        # frames are downscaled for analysis; color stats are aspect-insensitive
_SCOPE_HEIGHT = 90

# Display transfer used to re-encode decoded log/linear footage back into the gamma domain a
# waveform/vectorscope reads. BT.709's reference display is BT.1886 ≈ pure 2.4 gamma (Lb=0 ⇒
# ``code = linear**(1/2.4)``). See color-science.md §2.1.
_DISPLAY_GAMMA = 2.4

# Color-space names film-grip reads display-referred (the current/legacy path — no decode needed).
# These mirror :data:`filmgrip.protocol.editplan.COLOR_SPACES` display-referred entries.
_DISPLAY_SPACES = frozenset({"", "rec709", "rec2020", "srgb"})


def _need_numpy():
    if np is None:
        raise PerceptionUnavailable(
            "color scopes need numpy — install it with: pip install 'film-grip[color]'")


# --------------------------------------------------------------------------- log/ACES decode
# Each decoder maps a *coded* value in 0..1 to scene-linear light (so 18% mid-gray → ≈0.18, black →
# ≈0). Curves and breakpoints are the official specs; anchors validated against color-science.md §2.2
# on this machine (black/18%-gray/90%-white land on the documented values). These read display-side
# instruments, so callers re-encode the linear result to the display gamma via
# :func:`_linear_to_display` before measuring luma/exposure.
def _decode_sony_slog3(v: Any) -> Any:
    """Sony S-Log3 coded → scene-linear (breakpoint at linear 0.01125)."""
    vbound = (0.01125 * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    return np.where(
        v >= vbound,
        (10.0 ** ((v * 1023.0 - 420.0) / 261.5)) * 0.19 - 0.01,
        (v * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0),
    )


def _decode_arri_logc3(v: Any) -> Any:
    """ARRI LogC3 (EI800) coded → scene-linear."""
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    vcut = e * cut + f
    return np.where(v > vcut, (10.0 ** ((v - d) / c) - b) / a, (v - f) / e)


def _decode_arri_logc4(v: Any) -> Any:
    """ARRI LogC4 coded → scene-linear (a/b/s/t/x from the LogC4 spec)."""
    a = (2.0 ** 18.0 - 16.0) / 117.45
    b = (1023.0 - 95.0) / 1023.0
    c = 95.0 / 1023.0
    s = (7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * c / b)) / (a * b)
    t = (2.0 ** (14.0 * (-c / b) + 6.0) - 64.0) / a
    p = (v - c) / b
    return np.where(p < 0.0, p * s + t, (2.0 ** (14.0 * p - 6.0) - 64.0) / a)


def _decode_panasonic_vlog(v: Any) -> Any:
    """Panasonic V-Log coded → scene-linear (breakpoint at linear 0.01)."""
    vcut = 5.6 * 0.01 + 0.125
    return np.where(v >= vcut, 10.0 ** ((v - 0.598206) / 0.241514) - 0.00873, (v - 0.125) / 5.6)


def _decode_red_log3g10(v: Any) -> Any:
    """RED Log3G10 coded → scene-linear (a/b/g + 0.01 offset from the RED white paper)."""
    a, b, g = 0.224282, 155.975327, 15.1927
    off = 0.01
    lin = (10.0 ** (v / a) - 1.0) / b
    return np.where(v < 0.0, v / g, lin) - off


def _decode_acescct(v: Any) -> Any:
    """ACEScct coded → scene-linear (AP1 log with a linear toe; toe break at 0.155251141552511)."""
    return np.where(
        v > 0.155251141552511,
        2.0 ** (v * 17.52 - 9.72),
        (v - 0.0729055341958355) / 10.5402377416545,
    )


def _decode_acescc(v: Any) -> Any:
    """ACEScc coded → scene-linear (AP1 log, no linear toe)."""
    lin = 2.0 ** (v * 17.52 - 9.72)
    return np.where(v < -0.301369863013699, (2.0 ** (v * 17.52 - 9.72) - 2.0 ** -16.0) * 2.0, lin)


# Coded-log decoders keyed by the editplan COLOR_SPACES vocabulary (filmgrip.protocol.editplan).
_LOG_DECODERS = {
    "sony_slog3": _decode_sony_slog3,
    "arri_logc3": _decode_arri_logc3,
    "arri_logc4": _decode_arri_logc4,
    "panasonic_vlog": _decode_panasonic_vlog,
    "red_log3g10": _decode_red_log3g10,
    "blackmagic_film_gen5": _decode_arri_logc3,  # BMD Film Gen5 is LogC3-like; reuse as approximation
    "canon_clog3": _decode_arri_logc3,           # C-Log3 anchors ≈ LogC3 (black 32, 18% gray 88)
    "acescct": _decode_acescct,
    "acescc": _decode_acescc,
}
# Spaces already in scene-linear — no log curve, just re-encode to the display gamma so the scope
# reads display-referred values (ACES/linear are "not directly viewable" without an output transform;
# a display-gamma encode is the honest minimal stand-in for an exposure read — color-science.md §2.3).
_LINEAR_SPACES = frozenset({"aces", "linear"})


def _linear_to_display(lin: Any) -> Any:
    """Scene-linear → display-coded 0..1 via the BT.1886/2.4 display gamma (clamped to 0..1)."""
    return np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0) ** (1.0 / _DISPLAY_GAMMA)


def _to_display_rgb(rgb: Any, color_space: str) -> Any:
    """Return an ``(H, W, 3)`` float64 array of *display-referred* 0..255 codes.

    For display-referred ``color_space`` (rec709/srgb/rec2020/"") the input is already correct and is
    returned unchanged (so existing callers stay byte-identical). For a known log encoding the coded
    values are decoded to scene-linear and re-encoded to the display gamma BEFORE any exposure/clip
    verdict, so the luma read reflects what the footage looks like normalized — not the flat,
    bunched log curve. Unknown spaces fall through to the display path (treated as already-709).
    """
    cs = (color_space or "").strip().lower()
    if cs in _DISPLAY_SPACES or (cs not in _LOG_DECODERS and cs not in _LINEAR_SPACES):
        return rgb
    coded = rgb / 255.0
    lin = _LOG_DECODERS[cs](coded) if cs in _LOG_DECODERS else coded
    return _linear_to_display(lin) * 255.0


def analyze_rgb(arr: Any, *, color_space: str = "rec709") -> dict:
    """Reduce an ``(H, W, 3)`` uint8 RGB array to a colorist's scope reading (a JSON-able dict).

    Pure function — the heart of color perception, deliberately decoupled from ffmpeg so it can be
    unit-tested on synthetic frames. Raises if numpy isn't available or the array is the wrong shape.

    ``color_space`` names the footage's encoding using film-grip's editplan ``COLOR_SPACES``
    vocabulary. The DEFAULT ``"rec709"`` (and every other display-referred space — ``srgb``,
    ``rec2020``, ``""``, or anything unknown) reads the pixels as-is, so every existing caller is
    byte-identical. For a known log/ACES encoding (``sony_slog3``, ``arri_logc3``/``logc4``,
    ``panasonic_vlog``, ``red_log3g10``, ``canon_clog3``, ``blackmagic_film_gen5``, ``acescct``,
    ``acescc``, ``aces``, ``linear``) the frame is decoded to scene-linear and re-encoded to the
    display gamma BEFORE the luma/exposure/clip verdict — grading log as 709 stacks two transfer
    functions and makes scopes lie (color-science.md §2.3), so we normalize first. The returned
    report records the requested ``color_space`` and a ``decoded`` flag.
    """
    _need_numpy()
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"analyze_rgb expects an (H, W, 3) array, got shape {a.shape}")
    h, w = int(a.shape[0]), int(a.shape[1])
    rgb_src = a[:, :, :3].astype(np.float64)
    rgb = _to_display_rgb(rgb_src, color_space)
    decoded = rgb is not rgb_src
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
        "color_space": (color_space or "rec709"),
        "decoded": bool(decoded),
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


def apply_cdl_array(arr: Any, cdl: Any) -> Any:
    """Apply an ASC CDL (anything with ``slope``/``offset``/``power``/``saturation``, e.g.
    :class:`filmgrip.color.CDL`) to an ``(H, W, 3)`` uint8 frame, returning the graded uint8 frame.

    Vectorized, and the SAME math as :meth:`filmgrip.color.CDL.apply` — this is the predictive
    engine for the verify loop: it lets film-grip compute what a grade WILL look like (its scopes)
    without asking the editor to render, which is what makes the perceive→propose→verify→iterate
    loop closeable offline (LumiVideo's "analytically guaranteed" CDL effect)."""
    _need_numpy()
    a = np.asarray(arr)[:, :, :3].astype(np.float64) / 255.0
    slope = np.asarray(cdl.slope, dtype=np.float64)
    offset = np.asarray(cdl.offset, dtype=np.float64)
    power = np.asarray(cdl.power, dtype=np.float64)
    out = np.clip(a * slope + offset, 0.0, None) ** power
    sat = float(cdl.saturation)
    if sat != 1.0:
        luma = out @ np.asarray(_LUMA, dtype=np.float64)
        out = luma[..., None] + sat * (out - luma[..., None])
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def predict_grade_reading(arr: Any, cdl: Any) -> dict:
    """Predict the scope reading of a frame AFTER a CDL is applied — perception of a proposed grade
    with no editor round-trip."""
    return analyze_rgb(apply_cdl_array(arr, cdl))


def detect_log_footage(arr: Any) -> dict:
    """ADVISORY: does this frame look like flat/log/ACES footage that should be normalized to a
    display space BEFORE it is graded or judged as Rec.709?

    This is **advisory only** and **never auto-transforms** anything — it cannot be proven from
    pixels alone (fog, overcast/flat light, low-key night, faded-film and teal-orange looks all land
    in the same feature region), and filename/metadata are far more reliable signals. It simply names
    which of five numpy-computable signals fired so the agent can decide whether to pass an explicit
    ``color_space=`` to :func:`analyze_rgb` (color-science.md §2.4). Honesty tier: **ADVISORY**.

    Five signals over the luma (0..1) and RGB spread of a downscaled-OK frame:

    * ``low_contrast``        — ``p99 − p1`` of luma ``< 0.55`` (graded 709 spans ~0.66+; log ~0.25-0.45)
    * ``lifted_blacks``       — ``0.09 < p1 < 0.20`` and the true min never reaches ≈0 (log floors ~24-32)
    * ``depressed_highlights``— ``p99 < 0.78`` (log diffuse white ≈123-150, never near 255)
    * ``low_saturation``      — ``mean((max−min)/255)`` over RGB ``< 0.10`` (flat curve + wide gamut desaturates)
    * ``midtone_bunching``    — ``std`` of luma ``< 0.18`` (mass concentrated in the mids)

    ``score`` = count of signals firing; ``likely_log`` is ``score >= 3`` (the most discriminating
    pair is low-contrast + lifted-blacks — graded 709 deliberately puts shadows near 0, log never
    does). Returns ``{likely_log, score, signals, advice}``.
    """
    _need_numpy()
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"detect_log_footage expects an (H, W, 3) array, got shape {a.shape}")
    rgb = a[:, :, :3].astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    luma = (_LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b) / 255.0
    p1, p99 = (float(x) for x in np.percentile(luma, [1, 99]))
    lmin = float(np.min(luma))
    sat = float(np.mean((rgb.max(axis=2) - rgb.min(axis=2)) / 255.0))
    std = float(np.std(luma))

    signals = {
        "low_contrast": bool((p99 - p1) < 0.55),
        "lifted_blacks": bool(0.09 < p1 < 0.20 and lmin > 0.04),
        "depressed_highlights": bool(p99 < 0.78),
        "low_saturation": bool(sat < 0.10),
        "midtone_bunching": bool(std < 0.18),
    }
    score = int(sum(signals.values()))
    likely = bool(score >= 3)
    fired = [k for k, v in signals.items() if v]
    advice = (
        "ADVISORY: " + str(score) + "/5 log signals fired (" + ", ".join(fired) + "); footage may be "
        "log/ACES. Confirm via filename/metadata and, if so, call analyze_rgb(color_space=...) to "
        "normalize before grading — this read never auto-transforms."
        if likely else
        "ADVISORY: " + str(score) + "/5 log signals fired; footage reads display-referred (Rec.709). "
        "No normalization indicated from pixels — confirm with filename/metadata if unsure."
    )
    return {"likely_log": likely, "score": score, "signals": signals, "advice": advice}


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
