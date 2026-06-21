"""D7 — acoustic perception: quiet-span detection + J/L-cut candidate flagging.

``find_quiet`` is pure numpy on the RMS envelope, so its correctness is proved on a synthetic
tone/silence/tone signal (a known 1.0–2.0s quiet span) with no ffmpeg and no media file — exactly
like ``rms_envelope``. ``detect_jl_cuts`` is pure timing logic, proved on a hand-built two-clip IR
plus a synthetic word list (a word straddling the cut is flagged J vs L; a word fully inside one
shot is not). The retimed-clip honesty path is exercised through ``analyze_acoustic``'s guard with
no ffmpeg needed.
"""
from __future__ import annotations

import numpy as np
import opentimelineio as otio
import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.perception.acoustic import analyze_acoustic, detect_jl_cuts, find_quiet
from filmgrip.perception.transcribe import PerceptionUnavailable, Transcript, Word

SR = 22050
HOP_S = 0.01


# --------------------------------------------------------------------------- synthesis helpers
def _tone(dur_s: float, *, freq: float = 440.0, amp: float = 0.8) -> np.ndarray:
    t = np.arange(int(round(dur_s * SR)), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(dur_s: float) -> np.ndarray:
    return np.zeros(int(round(dur_s * SR)), dtype=np.float32)


# --------------------------------------------------------------------------- find_quiet (pure DSP)
def test_find_quiet_recovers_middle_silence():
    # tone(1s) | silence(1s) | tone(1s): the quiet span is media 1.0–2.0s.
    samples = np.concatenate([_tone(1.0), _silence(1.0), _tone(1.0)])
    spans = find_quiet(samples, SR, thresh_db=-40.0, min_dur_s=0.3, hop_s=HOP_S)

    assert len(spans) == 1
    sp = spans[0]
    assert sp["start_s"] == pytest.approx(1.0, abs=0.03)     # within ~3 hops of the true edge
    assert sp["end_s"] == pytest.approx(2.0, abs=0.03)
    assert sp["duration_s"] == pytest.approx(sp["end_s"] - sp["start_s"], abs=1e-9)
    assert sp["duration_s"] == pytest.approx(1.0, abs=0.06)


def test_find_quiet_all_loud_yields_no_spans():
    spans = find_quiet(_tone(2.0, amp=0.8), SR, thresh_db=-40.0, min_dur_s=0.3, hop_s=HOP_S)
    assert spans == []


def test_find_quiet_all_silent_yields_one_big_span():
    spans = find_quiet(_silence(2.0), SR, thresh_db=-40.0, min_dur_s=0.3, hop_s=HOP_S)
    assert len(spans) == 1
    assert spans[0]["start_s"] == pytest.approx(0.0, abs=1e-9)
    assert spans[0]["end_s"] == pytest.approx(2.0, abs=0.02)  # whole signal, minus sub-hop trim


def test_find_quiet_empty_signal_is_empty():
    assert find_quiet(np.zeros(0, dtype=np.float32), SR) == []


def test_find_quiet_thresh_db_boundary():
    # A constant -50 dBFS tone: amplitude 10**(-50/20). A sine's RMS is amp/sqrt(2), so its dBFS is
    # ~3 dB below the amplitude's dB. We assert the threshold *orders* correctly around the level.
    amp = 10 ** (-50.0 / 20.0)
    samples = _tone(1.0, amp=amp)
    # Threshold well above the signal level → the whole second is "quiet" (one span).
    loud_thresh = find_quiet(samples, SR, thresh_db=-30.0, min_dur_s=0.3, hop_s=HOP_S)
    assert len(loud_thresh) == 1
    # Threshold well below the signal level → nothing is quiet.
    strict = find_quiet(samples, SR, thresh_db=-70.0, min_dur_s=0.3, hop_s=HOP_S)
    assert strict == []


def test_find_quiet_min_dur_boundary():
    # A 0.20s gap between two tones. With min_dur 0.10s it is reported; with 0.40s it is too short.
    samples = np.concatenate([_tone(0.5), _silence(0.20), _tone(0.5)])
    kept = find_quiet(samples, SR, thresh_db=-40.0, min_dur_s=0.10, hop_s=HOP_S)
    assert len(kept) == 1
    assert kept[0]["duration_s"] == pytest.approx(0.20, abs=0.03)

    dropped = find_quiet(samples, SR, thresh_db=-40.0, min_dur_s=0.40, hop_s=HOP_S)
    assert dropped == []


# --------------------------------------------------------------------------- IR helpers
def _rt(frames: int, rate: float = 24.0) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, rate)


def _clip(name: str, media_uri: str, src_in_f: int, dur_f: int,
          *, retime: float | None = None) -> otio.schema.Clip:
    c = otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(target_url=media_uri),
        source_range=otio.opentime.TimeRange(_rt(src_in_f), _rt(dur_f)),
    )
    if retime is not None:
        c.effects.append(otio.schema.LinearTimeWarp(time_scalar=retime))
    return c


def _two_clip_ir() -> TimelineIR:
    """24fps: take1 = media 0–2s at frames 0–48; take2 = media 4–6s at frames 48–96.

    The picture cut is at timeline frame 48 — the same shape as the alignment test fixture.
    """
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("take1", "file:///m/interview.mov", 0, 48))
    track.append(_clip("take2", "file:///m/interview.mov", 96, 48))
    return TimelineIR(tl)


# --------------------------------------------------------------------------- detect_jl_cuts
def test_detect_jl_cuts_flags_l_cut_from_outgoing_word():
    ir = _two_clip_ir()
    out_id, in_id = (c.id for c in ir.real_clips())
    # take1 covers media 0–2s; a word at 1.9–2.1s straddles the clip's source-out (2.0s == frame 48).
    transcripts = {out_id: Transcript("interview.mov", "fake", [Word("over", 1.9, 2.1)])}

    cands = detect_jl_cuts(ir, transcripts)
    assert len(cands) == 1
    c = cands[0]
    assert c["type"] == "L"                              # outgoing clip's sound lags into B's picture
    assert c["video_frame"] == 48
    assert c["out_id"] == out_id and c["in_id"] == in_id
    assert c["word"] == "over"
    # 2.1s → frame 48 + round(0.1*24) = 50; audio should hold to the word end; offset is +2 (lags).
    assert c["suggested_audio_frame"] == 50
    assert c["offset_frames"] == 2
    assert c["basis"] == "word" and c["tier"] == "advisory"


def test_detect_jl_cuts_flags_j_cut_from_incoming_word():
    ir = _two_clip_ir()
    out_id, in_id = (c.id for c in ir.real_clips())
    # take2 covers media 4–6s placed at frames 48–96; a word at 3.9–4.1s straddles the source-in
    # (4.0s == frame 48): its sound LEADS under A's picture → J-cut.
    transcripts = {in_id: Transcript("interview.mov", "fake", [Word("hello", 3.9, 4.1)])}

    cands = detect_jl_cuts(ir, transcripts)
    assert len(cands) == 1
    c = cands[0]
    assert c["type"] == "J"
    assert c["video_frame"] == 48
    assert c["word"] == "hello"
    # 3.9s → frame 48 + round(-0.1*24) = 46; audio should lead in to the word start; offset -2.
    assert c["suggested_audio_frame"] == 46
    assert c["offset_frames"] == -2


def test_detect_jl_cuts_word_inside_shot_not_flagged():
    ir = _two_clip_ir()
    out_id, in_id = (c.id for c in ir.real_clips())
    # Words sitting comfortably inside each shot — neither straddles frame 48.
    transcripts = {
        out_id: Transcript("interview.mov", "fake", [Word("middle", 0.8, 1.2)]),
        in_id: Transcript("interview.mov", "fake", [Word("later", 5.0, 5.4)]),
    }
    assert detect_jl_cuts(ir, transcripts) == []


def test_detect_jl_cuts_accepts_bare_word_list():
    ir = _two_clip_ir()
    out_id, _ = (c.id for c in ir.real_clips())
    # The transcript value may be a bare list of Word objects, not a Transcript.
    cands = detect_jl_cuts(ir, {out_id: [Word("over", 1.9, 2.1)]})
    assert len(cands) == 1 and cands[0]["type"] == "L"


def test_detect_jl_cuts_empty_without_transcripts():
    assert detect_jl_cuts(_two_clip_ir(), {}) == []


# --------------------------------------------------------------------------- analyze_acoustic honesty
def test_analyze_acoustic_retimed_clip_errors_no_frames():
    # A retimed clip must NOT get a fabricated timeline mapping — and this path needs no ffmpeg,
    # because the retime guard returns before any decode.
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("warp", "file:///m/a.mov", 0, 48, retime=2.0))
    ir = TimelineIR(tl)
    clip = ir.real_clips()[0]

    res = analyze_acoustic("/m/a.mov", clip)
    assert res["errors"] and "retimed" in res["errors"][0]
    assert "quiet_spans_frames" not in res                # never guessed for a time-warped clip
    assert res["quiet_spans_s"] == []


def test_analyze_acoustic_honest_without_ffmpeg(monkeypatch, tmp_path):
    # No retime, real (empty) file on disk → the guard passes and decode_pcm must raise an
    # actionable PerceptionUnavailable (ffmpeg missing), not return a fake result.
    import filmgrip.perception.transcribe as tr

    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    media = tmp_path / "a.mov"
    media.write_bytes(b"\x00" * 16)
    with pytest.raises(PerceptionUnavailable) as exc:
        analyze_acoustic(str(media))
    assert "ffmpeg" in str(exc.value)
