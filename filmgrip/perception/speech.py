"""Deterministic speech analysis: silences + filler words → EditPlan-ready cut candidates.

No LLM anywhere in this module. Word timestamps in, exact ``cut_range`` ops out — the planner
(or the ``silence-cut`` pack, with no model call at all) just chooses which candidates to apply.

The two safety rules every candidate obeys (learned from video-use's production hard rules):

* **never cut inside a word** — silence cuts shrink inward by a drift pad before touching
  speech; filler cuts stop at the neighboring words' frames.
* **pad for ASR drift** — word timestamps are 50–100ms soft; pads default to 100ms
  (:data:`~filmgrip.perception.transcribe.ASR_DRIFT_PAD_S` bounds them).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..core.ir import TimelineIR
from .align import AlignedWord
from .transcribe import ASR_DRIFT_PAD_S

#: Words treated as disposable filler. Deliberately conservative — "like"/"so"/"well" carry
#: meaning too often to auto-cut; the planner can still target them via the transcript.
FILLER_WORDS: frozenset[str] = frozenset({
    "um", "uh", "erm", "uhm", "umm", "uhh", "hmm", "hm", "mm", "mmm", "er", "ah", "eh",
})

DEFAULT_MIN_SILENCE_S = 0.4   # video-use shades gaps ≥400ms; same default here
DEFAULT_PAD_S = 0.1           # inside ASR_DRIFT_PAD_S bounds


@dataclass(frozen=True)
class Silence:
    """A speech-free span inside one clip (timeline frames; ``kind``: head/interior/tail)."""

    clip_id: str
    start_frame: int
    end_frame: int
    seconds: float
    kind: str                  # "head" | "interior" | "tail"


@dataclass(frozen=True)
class Filler:
    """A disposable filler word occurrence (timeline frames)."""

    clip_id: str
    text: str
    start_frame: int
    end_frame: int


def _norm(text: str) -> str:
    return re.sub(r"[^a-z\-]", "", text.lower())


def find_silences(clip_id: str, span: tuple[int, int], words: list[AlignedWord], rate: float,
                  *, min_silence_s: float = DEFAULT_MIN_SILENCE_S) -> list[Silence]:
    """Speech-free spans ≥ ``min_silence_s`` inside the clip span, from aligned word frames."""
    min_frames = max(1, int(round(min_silence_s * rate)))
    start, end = span
    out: list[Silence] = []

    def add(a: int, b: int, kind: str) -> None:
        if b - a >= min_frames:
            out.append(Silence(clip_id, a, b, round((b - a) / rate, 3), kind))

    if not words:
        add(start, end, "head")  # an entirely speechless clip is one big head silence
        return out
    add(start, words[0].start_frame, "head")
    for prev, cur in zip(words, words[1:]):
        add(prev.end_frame, cur.start_frame, "interior")
    add(words[-1].end_frame, end, "tail")
    return out


def find_fillers(clip_id: str, words: list[AlignedWord], *,
                 extra: Optional[set[str]] = None) -> list[Filler]:
    """Occurrences of disposable filler words (normalized: case/punctuation stripped)."""
    vocab = FILLER_WORDS | {w.lower() for w in (extra or set())}
    return [Filler(clip_id, w.text, w.start_frame, w.end_frame)
            for w in words if _norm(w.text) in vocab]


# --------------------------------------------------------------------------- candidates
def silence_cut_ops(silences: list[Silence], rate: float, *,
                    pad_s: float = DEFAULT_PAD_S) -> list[dict]:
    """``cut_range`` op dicts that remove silences, shrunk inward so speech keeps a breath.

    Head silences keep the pad only on the speech side (the clip edge needs none); likewise
    tails. A silence that disappears under its pads yields no op.
    """
    pad = _pad_frames(pad_s, rate)
    ops: list[dict] = []
    for s in silences:
        a = s.start_frame + (0 if s.kind == "head" else pad)
        b = s.end_frame - (0 if s.kind == "tail" else pad)
        if b > a:
            ops.append({"op": "cut_range", "clip_id": s.clip_id,
                        "start_frame": a, "end_frame": b, "ripple": True})
    return _descending(ops)


def filler_cut_ops(fillers: list[Filler], words_by_clip: dict[str, list[AlignedWord]],
                   rate: float, *, pad_s: float = DEFAULT_PAD_S) -> list[dict]:
    """``cut_range`` op dicts that remove filler words, padded outward into surrounding
    silence but never into a neighboring word's frames."""
    pad = _pad_frames(pad_s, rate)
    ops: list[dict] = []
    for f in fillers:
        words = words_by_clip.get(f.clip_id, [])
        prev_end = max((w.end_frame for w in words if w.end_frame <= f.start_frame
                        and (w.start_frame, w.end_frame) != (f.start_frame, f.end_frame)),
                       default=None)
        next_start = min((w.start_frame for w in words if w.start_frame >= f.end_frame
                          and (w.start_frame, w.end_frame) != (f.start_frame, f.end_frame)),
                         default=None)
        a = f.start_frame - pad
        b = f.end_frame + pad
        if prev_end is not None:
            a = max(a, prev_end + 1)
        if next_start is not None:
            b = min(b, next_start - 1)
        if b > a:
            ops.append({"op": "cut_range", "clip_id": f.clip_id,
                        "start_frame": a, "end_frame": b, "ripple": True})
    return _descending(ops)


def _pad_frames(pad_s: float, rate: float) -> int:
    lo, hi = ASR_DRIFT_PAD_S
    return max(1, int(round(min(max(pad_s, lo), hi) * rate)))


def _descending(ops: list[dict]) -> list[dict]:
    """Last-to-first order — the validated contract for rippling cut_range ops."""
    return sorted(ops, key=lambda o: o["start_frame"], reverse=True)


# --------------------------------------------------------------------------- full analysis
def analyze_clips(ir: TimelineIR, ids: list[str], *, backend=None, transcriber=None,
                  min_silence_s: float = DEFAULT_MIN_SILENCE_S,
                  pad_s: float = DEFAULT_PAD_S,
                  include_fillers: bool = True) -> dict:
    """The ``analyze_speech`` payload: per-clip silences/fillers + ready-to-apply candidates.

    Same per-clip honesty as ``get_transcript``: offline media, missing ASR, retimed clips
    come back in ``errors``. ``candidates`` is one descending list the planner (or the
    silence-cut pack) can drop straight into an EditPlan.
    """
    from .align import align_clip_words, is_retimed, media_path_of
    from .transcribe import transcribe_media

    transcriber = transcriber or transcribe_media
    from ..serialize.fgx import _rate_str

    clips_out: list[dict] = []
    errors: list[str] = []
    silences_all: list[Silence] = []
    fillers_all: list[Filler] = []
    words_by_clip: dict[str, list[AlignedWord]] = {}
    transcripts: dict[str, object] = {}
    for cid in ids:
        clip = ir.clip(cid)
        if clip is None or clip.kind != "clip":
            errors.append(f"{cid}: not a clip on this timeline")
            continue
        if is_retimed(clip):
            errors.append(f"{cid}: retimed clip — speech analysis skipped")
            continue
        path = media_path_of(clip)
        if path is None:
            errors.append(f"{cid}: source media path unknown — cannot analyze")
            continue
        try:
            if path not in transcripts:
                transcripts[path] = transcriber(path, backend=backend)
            words = align_clip_words(clip, transcripts[path], ir.rate)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
            continue
        words_by_clip[cid] = words
        silences = find_silences(cid, (clip.start, clip.end), words, ir.rate,
                                 min_silence_s=min_silence_s)
        fillers = find_fillers(cid, words) if include_fillers else []
        silences_all.extend(silences)
        fillers_all.extend(fillers)
        clips_out.append({
            "id": cid,
            "silences": [[s.start_frame, s.end_frame, s.seconds, s.kind] for s in silences],
            "fillers": [[f.text, f.start_frame, f.end_frame] for f in fillers],
        })
    candidates = _descending(
        silence_cut_ops(silences_all, ir.rate, pad_s=pad_s)
        + filler_cut_ops(fillers_all, words_by_clip, ir.rate, pad_s=pad_s))
    return {"r": _rate_str(ir.rate), "frames": "timeline", "clips": clips_out,
            "candidates": candidates, "errors": errors}
