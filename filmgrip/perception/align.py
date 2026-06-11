"""Align transcripts to the timeline: media-time words → TIMELINE frames per clip.

This is the piece that turns ASR into an *editing* superpower inside the NLE: once every word
carries the timeline frame range it occupies, "cut after 'right'" or "remove the um at the top"
resolves to exact ``split``/``trim`` frames the existing EditPlan ops already know how to apply.

Honesty rules:

* a clip whose media path can't be determined (missing reference, offline media) gets a
  per-clip error — never a guessed path;
* a retimed clip (LinearTimeWarp/FreezeFrame) breaks the linear media↔timeline mapping, so
  alignment refuses it with a warning instead of emitting wrong frames;
* word timestamps drift 50–100ms (ASR reality) — consumers padding cuts should use
  :data:`~filmgrip.perception.transcribe.ASR_DRIFT_PAD_S`.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.ir import Clip, TimelineIR
from .transcribe import Backend, Transcript, transcribe_media

PACK_GAP_S = 0.5  # phrase break threshold, same rule as transcribe.pack_transcript


@dataclass(frozen=True)
class AlignedWord:
    """One word placed on the timeline (frames are timeline-relative, planner-native)."""

    text: str
    start_frame: int
    end_frame: int
    speaker: Optional[str] = None


@dataclass
class ClipTranscript:
    """The aligned words for one clip, plus the packed lines the planner actually reads."""

    clip_id: str
    track: str
    src_ref: str
    media_path: str
    span: tuple[int, int]                 # (start, end) timeline frames of the clip
    words: list[AlignedWord] = field(default_factory=list)

    def packed_lines(self, rate: float, *, gap_s: float = PACK_GAP_S) -> list[str]:
        return pack_aligned(self.words, rate, gap_s=gap_s)


# --------------------------------------------------------------------------- media paths
def media_path_of(clip: Clip) -> Optional[str]:
    """Best-effort full path to the clip's source media; ``None`` when unknowable.

    Order: the OTIO media reference's ``target_url`` (fixtures/interchange keep full paths
    there), then the live editor handle (Resolve's media-pool ``File Path`` property).
    """
    mr = getattr(clip.otio, "media_reference", None)
    url = getattr(mr, "target_url", None)
    if url:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            path = urllib.parse.unquote(parsed.path)
            if parsed.netloc and os.name != "nt":  # file://host/.. — keep absolute form
                path = "/" + parsed.netloc + path if not path.startswith("/") else path
            return path
        if not parsed.scheme:                      # plain path, no URL scheme
            return urllib.parse.unquote(url)
    native = clip.native
    if native is not None:
        try:
            mpi = native.GetMediaPoolItem()
            path = mpi.GetClipProperty("File Path") if mpi else None
            if path:
                return str(path)
        except Exception:
            pass
    return None


def is_retimed(clip: Clip) -> bool:
    """True when the clip carries a time-warp effect (word alignment would lie)."""
    import opentimelineio as otio

    effects = getattr(clip.otio, "effects", None) or []
    return any(isinstance(e, (otio.schema.LinearTimeWarp, otio.schema.FreezeFrame))
               for e in effects)


def _source_rate(clip: Clip, fallback: float) -> float:
    sr = getattr(clip.otio, "source_range", None)
    if sr is not None:
        rate = float(sr.start_time.rate)
        if rate > 0:
            return rate
    return fallback


# --------------------------------------------------------------------------- alignment
def align_clip_words(clip: Clip, transcript: Transcript, rate: float) -> list[AlignedWord]:
    """Map the transcript words that play inside ``clip`` onto timeline frames.

    A word belongs to the clip when its midpoint falls inside the clip's source window; its
    frames are clamped to the clip span so boundary words never address a neighbor.
    """
    src_rate = _source_rate(clip, rate)
    clip_in_s = clip.source_start / src_rate
    clip_out_s = clip_in_s + clip.duration / rate   # no retime ⇒ source seconds == timeline seconds
    out: list[AlignedWord] = []
    for word in transcript.words:
        mid = (word.start + word.end) / 2.0
        if not (clip_in_s <= mid < clip_out_s):
            continue
        sf = clip.start + int(round((word.start - clip_in_s) * rate))
        ef = clip.start + int(round((word.end - clip_in_s) * rate))
        sf = max(clip.start, min(sf, clip.end - 1))
        ef = max(sf + 1, min(ef, clip.end))
        out.append(AlignedWord(text=word.text, start_frame=sf, end_frame=ef,
                               speaker=word.speaker))
    return out


def pack_aligned(words: list[AlignedWord], rate: float, *, gap_s: float = PACK_GAP_S) -> list[str]:
    """Pack aligned words into phrase lines keyed by TIMELINE FRAMES: ``[0264-0290] S0 text``."""
    gap_frames = max(1, int(round(gap_s * rate)))
    lines: list[str] = []
    phrase: list[AlignedWord] = []

    def flush() -> None:
        if not phrase:
            return
        spk = f" {phrase[0].speaker}" if phrase[0].speaker is not None else ""
        text = " ".join(w.text for w in phrase)
        lines.append(f"[{phrase[0].start_frame:04d}-{phrase[-1].end_frame:04d}]{spk} {text}")
        phrase.clear()

    for word in words:
        if phrase and (word.start_frame - phrase[-1].end_frame >= gap_frames
                       or word.speaker != phrase[0].speaker):
            flush()
        phrase.append(word)
    flush()
    return lines


Transcriber = Callable[..., Transcript]


def transcript_for_clips(
    ir: TimelineIR,
    ids: list[str],
    *,
    backend: Optional[Backend] = None,
    transcriber: Transcriber = transcribe_media,
) -> dict:
    """The ``get_transcript`` payload: per-clip packed phrases in timeline frames + errors.

    Transcription happens once per distinct media file (cached by the transcriber), then each
    clip projects its own source window. ``transcriber`` is injectable for tests.
    """
    clips: list[dict] = []
    errors: list[str] = []
    transcripts: dict[str, Transcript] = {}
    for cid in ids:
        clip = ir.clip(cid)
        if clip is None or clip.kind != "clip":
            errors.append(f"{cid}: not a clip on this timeline")
            continue
        if is_retimed(clip):
            errors.append(f"{cid}: retimed clip — word alignment would be wrong, skipped "
                          f"(remove the retime or address it by frames)")
            continue
        path = media_path_of(clip)
        if path is None:
            errors.append(f"{cid}: source media path unknown (offline media or a reference "
                          f"without a file URL) — cannot transcribe")
            continue
        try:
            if path not in transcripts:
                transcripts[path] = transcriber(path, backend=backend)
            words = align_clip_words(clip, transcripts[path], ir.rate)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
            continue
        ct = ClipTranscript(
            clip_id=cid, track=f"{clip.track_kind[0]}{clip.track_index}",
            src_ref=clip.src_ref, media_path=path, span=(clip.start, clip.end), words=words,
        )
        clips.append({
            "id": cid, "t": ct.track, "src": ct.src_ref, "span": [clip.start, clip.end],
            "phrases": ct.packed_lines(ir.rate),
        })
    from ..serialize.fgx import _rate_str  # single source of truth for the rate header

    return {"r": _rate_str(ir.rate), "frames": "timeline", "clips": clips, "errors": errors}


def aligned_srt(ir: TimelineIR, ids: list[str], *, backend: Optional[Backend] = None,
                transcriber: Transcriber = transcribe_media,
                max_chars: int = 42) -> tuple[str, list[str]]:
    """Timeline-time SRT for the given clips (importable as captions). Returns (srt, errors)."""
    from .transcribe import Word, to_srt

    payload_errors: list[str] = []
    words: list[Word] = []
    for cid in ids:
        clip = ir.clip(cid)
        if clip is None or clip.kind != "clip":
            payload_errors.append(f"{cid}: not a clip on this timeline")
            continue
        if is_retimed(clip):
            payload_errors.append(f"{cid}: retimed clip — skipped")
            continue
        path = media_path_of(clip)
        if path is None:
            payload_errors.append(f"{cid}: source media path unknown — skipped")
            continue
        try:
            transcript = transcriber(path, backend=backend)
        except Exception as exc:
            payload_errors.append(f"{cid}: {exc}")
            continue
        for aw in align_clip_words(clip, transcript, ir.rate):
            words.append(Word(text=aw.text, start=aw.start_frame / ir.rate,
                              end=aw.end_frame / ir.rate, speaker=aw.speaker))
    words.sort(key=lambda w: w.start)
    pseudo = Transcript(media_path="timeline", backend="aligned", words=words)
    return to_srt(pseudo, max_chars=max_chars), payload_errors
