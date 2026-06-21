"""Motion & scene perception — shot boundaries, motion magnitude, and shake, in pure numpy.

The companion to :mod:`filmgrip.perception.scopes`: where scopes reduces *color* to numbers a
model can reason over, this reduces *temporal structure* — where the cuts are, how much the frame
is moving, whether the camera is shaky — so the planner can propose honest cut/retime edits and
*verify* them.

The reference algorithm is PySceneDetect's ``ContentDetector`` (BSD-3) — a frame-to-frame
mean-absolute pixel delta in HSV, threshold-tested for a cut. We **reimplement its math in pure
numpy** rather than depend on it (or on opencv), exactly the way ``scopes.analyze_rgb`` keeps a
framework-free pure core: the deterministic part is provable on synthetic arrays, offline, with no
editor. opencv's Farneback dense optical flow is NOT available without the opencv wheel, so the
numpy-only path uses **mean-absolute frame difference** (luma + HSV) for motion magnitude and the
**variance of the inter-frame difference signal** for shake — coarser than dense flow but real,
offline, and honest.

Honesty tiers (see ``docs/research/motion-scene.md`` §6):

* **Shot detection & motion magnitude = reliable evidence** — deterministic, offline, scores
  reported. Hard cuts are detected reliably.
* **Dissolves / fades = advisory** — a gradual transition is a low, sustained bump, not a spike;
  a content/mean-diff detector *under*-detects it. We do not pretend otherwise.
* **In-NLE stabilization is NOT scriptable → advisory only** — Resolve/Premiere expose no
  scripting hook for their stabilizer (same wall as the color scopes). :func:`stability_score`
  returns *evidence*, never an edit; the most this layer does is advise "enable Stabilization in
  the Inspector."
* **ffmpeg ``vidstab`` bake is destructive / opt-in and NOT part of this reader** — a two-pass
  ``vidstab`` re-encodes, crops and zooms a *new* media file (generation loss). That bake is a
  separate, explicitly-destructive action, never run here and never silently.

Retimed clips are **refused** (mirroring :func:`filmgrip.perception.align.is_retimed` /
``frames``): a LinearTimeWarp/FreezeFrame breaks the linear media↔timeline frame mapping, so any
boundary frame this reader returned would be a lie. Missing numpy / ffmpeg degrade to
:class:`~filmgrip.perception.transcribe.PerceptionUnavailable` with the fix, never a fake result.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

try:                                  # numpy powers the pixel math; optional install (the color extra).
    import numpy as np
except ImportError:                   # pragma: no cover - exercised only on a numpy-less install
    np = None

from .transcribe import PerceptionUnavailable, ffmpeg_path

# Analysis is done on small frames — motion/scene scores are aspect-insensitive and this keeps the
# pure cores cheap, the same downscale trick ``scopes.frame_rgb`` uses.
_MOTION_WIDTH = 160
_MOTION_HEIGHT = 90

#: Default cut threshold for :func:`scene_boundaries`. Scores from :func:`motion_series` are
#: normalized mean-absolute frame deltas in 0..1; a hard cut (e.g. dark→bright) is ~1.0 while
#: continuous motion sits well below. 0.3 separates the two with margin (the ffmpeg ``scdet`` /
#: ``select=gt(scene,0.3..0.4)`` convention rescaled to 0..1; PySceneDetect's 27/255 ≈ 0.106 is the
#: floor — we sit above it to suppress pan false-positives on the coarser mean-diff metric).
DEFAULT_CUT_THRESHOLD = 0.3

#: Default debounce: PySceneDetect's ``min_scene_len`` is 15 frames. We derive the gap from fps in
#: :func:`scene_boundaries` (≈0.4 s) so two cuts can't register within a few frames of each other.
_DEFAULT_MIN_GAP_S = 0.4

_LUMA = (0.2126, 0.7152, 0.0722)      # Rec.709, same coefficients as scopes


def _need_numpy() -> None:
    if np is None:
        raise PerceptionUnavailable(
            "motion/scene analysis needs numpy — install it with: pip install 'film-grip[color]'")


# --------------------------------------------------------------------------- pure cores
def _as_stack(frames: Any) -> "np.ndarray":
    """Validate and coerce an ``(N, H, W, 3)`` uint8-ish RGB stack to float64. Pure."""
    _need_numpy()
    a = np.asarray(frames)
    if a.ndim != 4 or a.shape[0] < 1 or a.shape[-1] < 3:
        raise ValueError(
            f"expected a frame stack of shape (N, H, W, 3), got shape {a.shape}")
    return a[:, :, :, :3].astype(np.float64)


def _luma_stack(stack: "np.ndarray") -> "np.ndarray":
    """(N, H, W) Rec.709 luma from an (N, H, W, 3) float stack. Pure."""
    return stack @ np.asarray(_LUMA, dtype=np.float64)


def motion_series(frames: Any) -> "np.ndarray":
    """Per-adjacent-pair motion score for an ``(N, H, W, 3)`` RGB stack → ``(N-1,)`` array in 0..1.

    Each score is the **mean absolute pixel difference** between consecutive frames, averaged over
    the RGB channels and normalized by 255 so it is encoding-stable and lands in ``[0, 1]``. This
    is the numpy-only stand-in for dense optical-flow magnitude (Farneback needs opencv, which is
    not a dependency — see the module docstring): identical frames → 0, a hard cut → ~1, continuous
    motion somewhere in between. Pure function — no IO — so it is unit-testable on synthetic arrays
    exactly like :func:`scopes.analyze_rgb`.

    Returns an empty array for a single-frame stack (no pair to compare).
    """
    stack = _as_stack(frames)
    if stack.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    diffs = np.abs(stack[1:] - stack[:-1])              # (N-1, H, W, 3)
    return diffs.mean(axis=(1, 2, 3)) / 255.0           # (N-1,) in 0..1


def scene_boundaries(motion_or_diffs: Any, fps: float, *, thresh: float = DEFAULT_CUT_THRESHOLD,
                     min_gap_frames: Optional[int] = None) -> list[int]:
    """Frame indices where a cut occurs, given a per-pair motion/diff series. Pure — no IO.

    A boundary is registered at series index ``i`` (i.e. between input frame ``i`` and ``i+1``,
    reported as frame ``i+1`` — the first frame of the new shot) when ``series[i] >= thresh`` and at
    least ``min_gap_frames`` have passed since the last accepted boundary. This is the threshold +
    ``min_scene_len`` debounce of PySceneDetect's ``ContentDetector``, the gap defaulting to
    ~``0.4 s`` worth of frames (so a single pan/whip can't spray boundaries).

    Returned positions are offsets into the supplied series' frame space; for a stack starting at a
    timeline frame, the caller adds that base (``analyze_motion`` does this to return TIMELINE
    frames). ``fps`` only sets the default debounce window.
    """
    _need_numpy()
    series = np.asarray(motion_or_diffs, dtype=np.float64).ravel()
    if series.size == 0:
        return []
    if min_gap_frames is None:
        min_gap_frames = max(1, int(round(_DEFAULT_MIN_GAP_S * float(fps)))) if fps > 0 else 1
    min_gap_frames = max(1, int(min_gap_frames))
    boundaries: list[int] = []
    last = -min_gap_frames                              # allow a boundary at the very start
    for i in range(series.size):
        if series[i] >= thresh and (i + 1) - last >= min_gap_frames:
            boundaries.append(i + 1)                    # frame index of the new shot's first frame
            last = i + 1
    return boundaries


def stability_score(frames: Any) -> float:
    """Advisory 0..1 stability score for an ``(N, H, W, 3)`` RGB stack (1 = rock steady). Pure.

    Shake is **high-frequency, low-net-displacement** motion (the doc's definition): lots of
    frame-to-frame change that *cancels out* instead of accumulating into a smooth move. Without
    dense optical flow we read that directly from luma as a **path-length vs net-displacement
    ratio**:

    * ``path`` = sum of per-pair mean-abs luma deltas — the total distance the picture travelled.
    * ``net``  = mean-abs luma delta between the *first and last* frame — how far it actually got.

    A static shot has ``path ≈ net ≈ 0`` (steady). A steady constant pan accumulates: ``net`` is a
    large fraction of ``path`` (steady). Shake reverses every frame so motion cancels: ``path`` is
    large but ``net`` stays small (shaky). The score is ``net / path`` (the fraction of motion that
    "stuck"), so steady → ~1 and jitter → ~0, with ``path ≈ 0`` treated as perfectly steady.

    **Advisory only.** This is evidence the planner/agent can act on (e.g. advise enabling the NLE
    stabilizer); it is never itself an edit, and film-grip cannot toggle in-NLE stabilization.
    """
    stack = _as_stack(frames)
    if stack.shape[0] < 3:
        return 1.0                                      # too few frames to call it shaky
    luma = _luma_stack(stack)                           # (N, H, W)
    path = float(np.abs(luma[1:] - luma[:-1]).mean(axis=(1, 2)).sum())
    if path <= 1e-6:
        return 1.0                                      # no motion at all → perfectly steady
    net = float(np.abs(luma[-1] - luma[0]).mean())
    return round(float(np.clip(net / path, 0.0, 1.0)), 4)


# --------------------------------------------------------------------------- IO (ffmpeg)
def _extract_frames(media_path: str, *, fps: float, ss: float = 0.0,
                    duration: Optional[float] = None, width: int = _MOTION_WIDTH,
                    height: int = _MOTION_HEIGHT) -> "np.ndarray":
    """Decode low-res RGB frames at ``fps`` into an ``(N, H, W, 3)`` uint8 array via ffmpeg.

    Mirrors ``scopes.frame_rgb``'s exact arg-list subprocess pattern (``-v error``, ``scale`` to a
    small width, ``-f rawvideo -pix_fmt rgb24``, ``np.frombuffer``) but streams *many* frames: an
    ``fps`` filter resamples to a handful of frames per second, which is plenty to find cuts and
    score motion while keeping the decode cheap.
    """
    _need_numpy()
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable(
            "ffmpeg is required for motion/scene analysis — install it (e.g. `brew install ffmpeg`).")
    cmd = [ffmpeg, "-v", "error"]
    if ss > 0:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", media_path]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    frame_bytes = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < frame_bytes:
        raise PerceptionUnavailable(
            f"ffmpeg could not extract frames from '{media_path}': "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    n = len(proc.stdout) // frame_bytes
    return np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8).reshape(n, height, width, 3)


def analyze_motion(media_path: str, clip: Any = None, *, fps: float = 4) -> dict:
    """Read motion & scene structure from a media file (optionally scoped to one timeline clip).

    Extracts low-res frames at ``fps`` via ffmpeg, then runs the pure cores: :func:`motion_series`,
    :func:`scene_boundaries`, :func:`stability_score`. Returns a JSON-able dict::

        {"fps", "frames", "n_frames",
         "boundaries_frames": [...],          # TIMELINE frames when a clip is given, else sample idx
         "motion": {"min","mean","max","peak_frame"},
         "stability": float,                  # advisory 0..1 (1 = steady)
         "errors": [...]}

    With ``clip`` supplied the analysis is scoped to that clip's source window and boundaries are
    returned in **timeline frames** (sample index → timeline frame via the clip's start). A
    **retimed clip is refused** (returns an ``errors`` entry, no boundaries) because a time-warp
    breaks the media↔timeline frame mapping — same guard as ``align``/``frames``. Honest-degrade:
    :class:`PerceptionUnavailable` is raised when numpy or ffmpeg is absent.
    """
    _need_numpy()
    if fps <= 0:
        raise ValueError("fps must be positive")

    ss = 0.0
    duration: Optional[float] = None
    base_timeline_frame: Optional[int] = None
    timeline_rate: Optional[float] = None

    if clip is not None:
        # Refuse retimed clips honestly — a LinearTimeWarp/FreezeFrame makes every mapped frame a lie.
        from .align import is_retimed
        if is_retimed(clip):
            return {
                "fps": fps, "frames": "timeline", "n_frames": 0, "boundaries_frames": [],
                "motion": {}, "stability": None,
                "errors": [f"{getattr(clip, 'id', '?')}: retimed clip — frame mapping would be "
                           f"wrong, refused (remove the retime or address it by frames)"],
            }
        # Scope the decode to the clip's source window; report boundaries in timeline frames.
        src_rate = _source_rate(clip, fps)
        ss = float(clip.source_start) / src_rate if src_rate > 0 else 0.0
        timeline_rate = src_rate
        duration = float(clip.duration) / src_rate if src_rate > 0 else None
        base_timeline_frame = int(clip.start)

    frames = _extract_frames(media_path, fps=fps, ss=ss, duration=duration)
    series = motion_series(frames)
    sample_boundaries = scene_boundaries(series, fps)

    if base_timeline_frame is not None and timeline_rate and timeline_rate > 0:
        # sample index (at `fps`) → media-seconds-into-clip → timeline frame.
        scale = timeline_rate / float(fps)
        boundaries = [base_timeline_frame + int(round(b * scale)) for b in sample_boundaries]
    else:
        boundaries = list(sample_boundaries)

    if series.size:
        peak_idx = int(np.argmax(series))
        motion = {
            "min": round(float(series.min()), 4),
            "mean": round(float(series.mean()), 4),
            "max": round(float(series.max()), 4),
            "peak_frame": (base_timeline_frame + int(round(peak_idx * (timeline_rate / float(fps)))))
                          if (base_timeline_frame is not None and timeline_rate and timeline_rate > 0)
                          else peak_idx,
        }
    else:
        motion = {"min": 0.0, "mean": 0.0, "max": 0.0, "peak_frame": None}

    return {
        "fps": fps,
        "frames": "timeline" if base_timeline_frame is not None else "sample",
        "n_frames": int(frames.shape[0]),
        "boundaries_frames": boundaries,
        "motion": motion,
        "stability": stability_score(frames),
        "source": os.path.basename(media_path),
        "errors": [],
    }


# --------------------------------------------------------------------------- helpers
def _source_rate(clip: Any, fallback: float) -> float:
    """Source-media frame rate of a clip (falls back to ``fallback``). Mirrors align._source_rate."""
    sr = getattr(getattr(clip, "otio", None), "source_range", None)
    if sr is not None:
        try:
            rate = float(sr.start_time.rate)
            if rate > 0:
                return rate
        except Exception:
            pass
    return fallback
