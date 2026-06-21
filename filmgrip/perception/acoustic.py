"""Acoustic perception — quiet-span detection + J/L-cut candidate flagging.

This is the *energy + split-edit* sibling of :mod:`filmgrip.perception.align` (speech) and
:mod:`filmgrip.perception.scopes` (color). Two distinct jobs, two distinct honesty tiers:

* **quiet spans** — :func:`find_quiet` is pure numpy on the per-hop RMS envelope from
  :func:`filmgrip.perception.audio_io.rms_envelope`: contiguous runs whose energy sits below a
  dBFS threshold for at least ``min_dur_s``. No ASR, no model — just the waveform. It is the
  energy-domain complement to the transcript's silence gaps: it finds *room tone / breath /
  dead air* the words don't mark. Being numpy-in/numpy-out, its correctness is provable on a
  synthetic tone/silence/tone signal with **no ffmpeg and no media file**, exactly like
  ``rms_envelope`` itself. :func:`analyze_acoustic` is the thin IO wrapper that decodes a clip's
  media and projects the spans onto **timeline frames**.

* **J/L cuts** — :func:`detect_jl_cuts` is **advisory** (research §"J-cuts/L-cuts"): given the IR's
  video cut boundaries and per-clip *word* timings, it flags boundaries where a spoken word
  straddles the cut, i.e. cutting picture there would chop a word mid-syllable. The honest fix is a
  split-edit — hold the outgoing clip's *audio* past the picture cut (**L-cut**, sound lags) or pull
  the incoming clip's *audio* in before its picture (**J-cut**, sound leads). film-grip does not
  auto-apply these (a true split-edit needs audio on its own track and does not round-trip uniformly
  through every interchange format); it surfaces *candidates* the planner/human acts on.

Honesty rules (the film-grip signature), mirrored from :mod:`align`:

* a **retimed** clip (LinearTimeWarp/FreezeFrame) breaks the linear media↔timeline mapping, so
  :func:`analyze_acoustic` refuses it into ``errors`` rather than emitting wrong frames;
* a clip whose **media path is unknowable** (offline media / a reference without a file URL) goes
  to ``errors`` — never a guessed path;
* a missing **ffmpeg or numpy** raises :class:`~filmgrip.perception.transcribe.PerceptionUnavailable`
  with the fix (numpy is a lazy guard via ``audio_io._need_numpy``);
* J/L detection is a **heuristic** — it is reported in the ``"advisory"`` tier and never drives an op.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from ..core.ir import Clip, TimelineIR
from . import audio_io
from .align import is_retimed, _source_rate
from .audio_io import DEFAULT_HOP_S, rms_envelope
from .transcribe import PerceptionUnavailable

#: Default decode rate for acoustic analysis. 22.05k preserves the transients that mark a quiet
#: edge while staying far cheaper than 44.1k — the same rate the rest of the musicality layer uses.
DEFAULT_RATE = audio_io.DEFAULT_RATE

#: Floor (linear amplitude) substituted for a zero-energy hop before the dBFS conversion, so a
#: perfectly silent run maps to a finite, very-low dB value instead of ``-inf``. -200 dBFS is far
#: below any real noise floor, so it never changes which hops count as "quiet".
_SILENCE_FLOOR = 1e-10


def find_quiet(samples: Any, sr: int, *, thresh_db: float = -40.0,
               min_dur_s: float = 0.3, hop_s: float = DEFAULT_HOP_S) -> list[dict]:
    """Contiguous spans of ``samples`` quieter than ``thresh_db`` for at least ``min_dur_s``.

    Pure numpy (no IO): reduces ``samples`` to a per-hop RMS envelope via
    :func:`~filmgrip.perception.audio_io.rms_envelope`, converts each hop to **dBFS**
    (``20·log10(rms)``, referenced to full-scale 1.0 — so 0 dB is a full-scale signal and quiet
    audio is negative), and returns every maximal run of hops at or below ``thresh_db`` whose
    duration reaches ``min_dur_s``.

    ``thresh_db`` is **dBFS** (decibels relative to full scale): a tone normalized to ±1.0 is
    ~0 dBFS, ``-40`` dBFS is a quiet room, ``-60`` near-silence. Each returned span is a dict in
    **seconds**::

        {"start_s": float, "end_s": float, "duration_s": float}

    A hop covers ``hop_s`` seconds; span edges fall on hop boundaries (start at the first quiet
    hop, end at the hop *after* the last quiet hop). An empty/sub-hop signal yields ``[]``.
    """
    audio_io._need_numpy()
    import numpy as np

    env = rms_envelope(samples, sr, hop_s=hop_s)
    if env.size == 0:
        return []
    # Per-hop dBFS, with a finite floor so true-silent hops don't become -inf.
    db = 20.0 * np.log10(np.maximum(env.astype(np.float64), _SILENCE_FLOOR))
    quiet = db <= thresh_db                              # boolean mask, one per hop
    min_hops = max(1, int(math.ceil(min_dur_s / hop_s)))

    spans: list[dict] = []
    run_start: Optional[int] = None
    n = quiet.size
    for i in range(n + 1):                               # +1 sentinel closes a trailing run
        is_quiet = bool(quiet[i]) if i < n else False
        if is_quiet and run_start is None:
            run_start = i
        elif not is_quiet and run_start is not None:
            if i - run_start >= min_hops:
                spans.append({
                    "start_s": run_start * hop_s,
                    "end_s": i * hop_s,
                    "duration_s": (i - run_start) * hop_s,
                })
            run_start = None
    return spans


def analyze_acoustic(media_path: str, clip: Optional[Clip] = None, *,
                     rate: int = DEFAULT_RATE) -> dict:
    """Decode ``media_path``, find its quiet spans, and (when ``clip`` is given) map them to
    **timeline frames** for that clip.

    Pipeline: :func:`~filmgrip.perception.audio_io.decode_pcm` →
    :func:`~filmgrip.perception.audio_io.rms_envelope` → :func:`find_quiet`. Returns::

        {"media_path": str, "rate": int,
         "quiet_spans_s": [ {start_s, end_s, duration_s}, ... ],   # MEDIA seconds, always present
         "quiet_spans_frames": [ {start, end, duration}, ... ],    # TIMELINE frames, only with a clip
         "errors": [ str, ... ]}

    Honesty: a **retimed** clip is refused into ``errors`` (the time-warp breaks the linear
    media↔timeline projection, exactly as it does for word alignment) and ``quiet_spans_frames`` is
    omitted — never guessed. A missing **ffmpeg**/**numpy** raises
    :class:`~filmgrip.perception.transcribe.PerceptionUnavailable`; a decode failure (bad/offline
    media) is caught and surfaced in ``errors`` rather than crashing the reader.

    When ``clip`` is ``None`` this analyzes the whole file (media-second spans only).
    """
    audio_io._need_numpy()
    out: dict = {"media_path": media_path, "rate": int(rate),
                 "quiet_spans_s": [], "errors": []}

    if clip is not None and is_retimed(clip):
        out["errors"].append(
            f"{clip.id}: retimed clip — quiet-span timeline mapping would be wrong, skipped "
            f"(remove the retime or address it by frames)")
        return out

    try:
        samples, sr = audio_io.decode_pcm(media_path, rate=rate, mono=True)
    except PerceptionUnavailable:
        raise                                           # missing ffmpeg/numpy: dependency, not data
    except Exception as exc:                            # pragma: no cover - defensive
        out["errors"].append(f"decode failed for '{media_path}': {exc}")
        return out

    spans = find_quiet(samples, sr)
    out["quiet_spans_s"] = spans

    if clip is not None:
        out["quiet_spans_frames"] = _spans_to_frames(spans, clip)
    return out


def _spans_to_frames(spans: list[dict], clip: Clip) -> list[dict]:
    """Project MEDIA-second quiet spans into ``clip``'s TIMELINE frames.

    Mirrors :func:`filmgrip.perception.align.align_clip_words`: a span is kept only if it overlaps
    the clip's source window; its frames are clamped to the clip span so a boundary span never
    addresses a neighbour. With no retime, source seconds equal timeline seconds, so the offset
    from the clip's source-in maps linearly onto ``clip.start``.
    """
    rate = _source_rate(clip, 24.0)                     # timeline rate == source rate (no retime)
    clip_in_s = clip.source_start / rate
    clip_out_s = clip_in_s + clip.duration / rate
    frames: list[dict] = []
    for sp in spans:
        s, e = sp["start_s"], sp["end_s"]
        if e <= clip_in_s or s >= clip_out_s:           # entirely outside the clip's source window
            continue
        s = max(s, clip_in_s)
        e = min(e, clip_out_s)
        sf = clip.start + int(round((s - clip_in_s) * rate))
        ef = clip.start + int(round((e - clip_in_s) * rate))
        sf = max(clip.start, min(sf, clip.end - 1))
        ef = max(sf + 1, min(ef, clip.end))
        frames.append({"start": sf, "end": ef, "duration": ef - sf})
    return frames


# --------------------------------------------------------------------------- J/L cuts (advisory)
def _cut_boundaries(ir: TimelineIR) -> list[dict]:
    """Interior video cut points: where one clip's out meets the next clip's in on a video track.

    Returns ``[{"frame", "out_id", "in_id"}, ...]`` for each adjacent (A→B) pair of real clips on
    the same video track, ordered along the timeline. These are exactly the picture-edit points a
    split-edit would offset its audio from.
    """
    boundaries: list[dict] = []
    n_tracks = ir.track_count("video")
    for ti in range(1, n_tracks + 1):
        track = sorted(
            [c for c in ir.clips_on("video", ti) if c.kind == "clip"],
            key=lambda c: c.start)
        for a, b in zip(track, track[1:]):
            boundaries.append({"frame": b.start, "out_id": a.id, "in_id": b.id})
    return boundaries


def detect_jl_cuts(ir: TimelineIR, transcripts_by_clip: dict[str, Any]) -> list[dict]:
    """Flag J/L-cut **candidates**: video cut boundaries where a spoken word straddles the cut.

    ADVISORY (a heuristic, research §"J-cuts/L-cuts"): for each interior picture cut between an
    outgoing clip A and an incoming clip B at timeline frame ``f``, inspect the *word* timings
    (``transcripts_by_clip`` maps a clip id → an object/list exposing word ``start``/``end`` in
    **MEDIA seconds**, e.g. a :class:`~filmgrip.perception.transcribe.Transcript` or a list of
    :class:`~filmgrip.perception.transcribe.Word`). A word *straddles* the cut when, projected onto
    the timeline, it starts before ``f`` and ends after ``f`` — cutting picture at ``f`` would chop
    it. Classification:

    * **L-cut** — the straddling word belongs to the **outgoing** clip A (its sound *lags* into B's
      picture); the honest fix holds A's audio to the word's end.
    * **J-cut** — the straddling word belongs to the **incoming** clip B (its sound *leads* under
      A's picture); the honest fix pulls B's audio in to the word's start.

    Returns one candidate per flagged boundary::

        {"type": "J"|"L", "video_frame": int, "out_id": str, "in_id": str,
         "word": str, "word_start_frame": int, "word_end_frame": int,
         "suggested_audio_frame": int, "offset_frames": int,
         "basis": "word", "tier": "advisory"}

    ``suggested_audio_frame`` is the word boundary the audio edit should move to and
    ``offset_frames = suggested_audio_frame - video_frame`` (signed: negative ⇒ audio leads the
    picture, positive ⇒ audio lags it). A word that sits fully inside one shot is **not** flagged.
    Pure logic on timings — no IO. Never auto-applied; the planner/human turns a candidate into a
    split-edit (``move``/``trim`` of a split-off audio clip).
    """
    candidates: list[dict] = []
    for bnd in _cut_boundaries(ir):
        f = bnd["frame"]
        out_clip = ir.clip(bnd["out_id"])
        in_clip = ir.clip(bnd["in_id"])
        if out_clip is None or in_clip is None:
            continue

        # L-cut: a word from the OUTGOING clip ends after the picture cut → its sound lags into B.
        cand = _straddle_for(out_clip, transcripts_by_clip.get(out_clip.id), f, ir.rate,
                             kind="L", out_id=bnd["out_id"], in_id=bnd["in_id"])
        if cand is not None:
            candidates.append(cand)

        # J-cut: a word from the INCOMING clip starts before the picture cut → its sound leads B.
        cand = _straddle_for(in_clip, transcripts_by_clip.get(in_clip.id), f, ir.rate,
                             kind="J", out_id=bnd["out_id"], in_id=bnd["in_id"])
        if cand is not None:
            candidates.append(cand)
    return candidates


def _words_of(transcript: Any) -> list[Any]:
    """Accept either a Transcript (``.words``) or a bare list of Word-like objects."""
    if transcript is None:
        return []
    words = getattr(transcript, "words", transcript)
    return list(words) if words is not None else []


def _straddle_for(clip: Clip, transcript: Any, f: int, rate: float, *,
                  kind: str, out_id: str, in_id: str) -> Optional[dict]:
    """Return the J/L candidate if any word of ``clip`` straddles the boundary frame ``f``.

    ``kind`` is ``"L"`` (test the OUTGOING clip — word must end after ``f``) or ``"J"`` (test the
    INCOMING clip — word must start before ``f``). Word seconds are projected to timeline frames
    via the clip's source window (no retime ⇒ source seconds == timeline seconds), the same math
    as :func:`align.align_clip_words`.
    """
    words = _words_of(transcript)
    if not words:
        return None
    src_rate = _source_rate(clip, rate)
    clip_in_s = clip.source_start / src_rate
    for w in words:
        w_start = float(getattr(w, "start"))
        w_end = float(getattr(w, "end"))
        sf = clip.start + int(round((w_start - clip_in_s) * rate))
        ef = clip.start + int(round((w_end - clip_in_s) * rate))
        # A word straddles the picture cut when it opens strictly before f and closes strictly after.
        if sf < f < ef:
            suggested = ef if kind == "L" else sf       # L holds A's audio to word end; J pulls B's in
            return {
                "type": kind,
                "video_frame": f,
                "out_id": out_id,
                "in_id": in_id,
                "word": getattr(w, "text", ""),
                "word_start_frame": sf,
                "word_end_frame": ef,
                "suggested_audio_frame": suggested,
                "offset_frames": suggested - f,
                "basis": "word",
                "tier": "advisory",
            }
    return None
