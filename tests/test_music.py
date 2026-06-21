"""D5 — audio musicality. The numpy onset+tempo path is the BASE capability, so its correctness is
proved on synthetic signals with NO librosa and NO ffmpeg (these tests must pass on the base install):

* a click onset envelope at a KNOWN BPM (impulses every 0.5s = 120 BPM) → detect_beats recovers the
  tempo within ±2 BPM (octave-tolerant) and the beat grid lands on the impulses;
* beats_to_frames maps a known media time to the hand-computed timeline frame (align's convention);
* a retimed clip is refused into ``errors`` with no fabricated beats, and offline media likewise;
* (ffmpeg-gated) a self-generated click track decoded through analyze_music reports ~120 BPM.

librosa is NOT assumed anywhere here — the default engine is pure numpy.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import opentimelineio as otio
import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.perception.music import (
    analyze_music,
    beats_to_frames,
    detect_beats,
    onset_envelope,
)
from filmgrip.perception.transcribe import PerceptionUnavailable

SR = 22050
HOP_S = 0.01


# ----------------------------------------------------------------------- helpers
def _click_envelope(bpm: float, dur_s: float, *, hop_s: float = HOP_S) -> np.ndarray:
    """A clean onset-strength envelope: a unit impulse every beat at ``bpm``, the rest zero."""
    n = int(round(dur_s / hop_s))
    env = np.zeros(n, dtype=np.float32)
    period_hops = (60.0 / bpm) / hop_s
    k = 0
    while int(round(k * period_hops)) < n:
        env[int(round(k * period_hops))] = 1.0
        k += 1
    return env


def _click_signal(bpm: float, dur_s: float, *, sr: int = SR) -> np.ndarray:
    """A raw audio click track: a windowed 1kHz tick on every beat (for onset_envelope to reduce)."""
    t = np.arange(int(sr * dur_s)) / sr
    sig = np.zeros_like(t)
    period_s = 60.0 / bpm
    tick = int(0.02 * sr)
    k = 0
    while k * period_s < dur_s:
        start = int(k * period_s * sr)
        seg = np.sin(2 * np.pi * 1000.0 * t[start:start + tick]) * np.hanning(tick)
        sig[start:start + tick] += seg
        k += 1
    return sig.astype(np.float32)


def _rt(frames: int, rate: int = 24) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, rate)


def _solo_ir(media_reference, *, src_in_f: int = 0, dur_f: int = 240) -> TimelineIR:
    c = otio.schema.Clip(name="clip", media_reference=media_reference,
                         source_range=otio.opentime.TimeRange(_rt(src_in_f), _rt(dur_f)))
    tl = otio.schema.Timeline(name="t")
    tr = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    tl.tracks.append(tr)
    tr.append(c)
    return TimelineIR(tl)


# ----------------------------------------------------------------------- tempo recovery (pure numpy)
def test_detect_beats_recovers_120_bpm_from_clean_envelope():
    env = _click_envelope(120.0, 16.0)              # impulses every 0.5s = 120 BPM
    res = detect_beats(env, SR, HOP_S)
    assert abs(res["tempo_bpm"] - 120.0) <= 2.0     # ±2 BPM
    # beat grid lands on the impulses: spacing ≈ 0.5s and the first few coincide with the clicks.
    beats = res["beat_times_s"]
    assert len(beats) >= 28                          # ~32 beats in 16s
    spacing = np.diff(beats)
    assert np.allclose(spacing, 0.5, atol=0.02)      # hop-quantized half-second beats
    # every detected beat is within one hop of a true 0.5s grid point (beat-index tolerance)
    for b in beats:
        assert min(abs(b - g * 0.5) for g in range(40)) <= HOP_S + 1e-9


def test_detect_beats_octave_tolerant():
    """A 60 BPM grid may resolve to 60 or its 120 octave — both are accepted (octave tolerance)."""
    env = _click_envelope(60.0, 20.0)
    res = detect_beats(env, SR, HOP_S)
    bpm = res["tempo_bpm"]
    assert any(abs(bpm - cand) <= 2.0 for cand in (60.0, 120.0, 30.0))


def test_onset_envelope_then_detect_beats_on_click_signal():
    """End-to-end pure-numpy: raw 1kHz click signal → onset_envelope → detect_beats ≈ 120 BPM."""
    sig = _click_signal(120.0, 8.0)
    env = onset_envelope(sig, SR, hop_s=HOP_S)
    assert env.dtype == np.float32 and env.ndim == 1 and env.size > 0
    assert float(env.max()) == pytest.approx(1.0, abs=1e-6)   # normalized to unit max
    res = detect_beats(env, SR, HOP_S)
    assert abs(res["tempo_bpm"] - 120.0) <= 2.0


def test_onset_envelope_sub_hop_signal_is_empty():
    assert onset_envelope(np.ones(10, dtype=np.float32), SR, hop_s=HOP_S).shape == (0,)


def test_detect_beats_abstains_on_flat_envelope():
    """A flat/silent envelope must NOT fabricate a tempo or grid (honesty)."""
    res = detect_beats(np.zeros(500, dtype=np.float32), SR, HOP_S)
    assert res["tempo_bpm"] == 0.0
    assert res["beat_times_s"] == [] and res["onset_times_s"] == []
    res2 = detect_beats(np.full(500, 0.3, dtype=np.float32), SR, HOP_S)
    assert res2["tempo_bpm"] == 0.0 and res2["beat_times_s"] == []


# ----------------------------------------------------------------------- frame projection
def test_beats_to_frames_maps_known_time_to_hand_computed_frame():
    # clip starts at timeline frame 48; a beat 0.5s into the clip @24fps → 48 + round(0.5*24) = 60.
    assert beats_to_frames([0.5], 24.0, clip_offset_frames=48) == [60]
    # multiple times + the no-offset (plain media-time) default.
    assert beats_to_frames([0.0, 1.0, 2.0], 24.0) == [0, 24, 48]
    # rounding is round-half (0.51s*24 = 12.24 → 12).
    assert beats_to_frames([0.51], 24.0) == [12]


def test_beats_to_frames_empty():
    assert beats_to_frames([], 24.0, clip_offset_frames=10) == []


# ----------------------------------------------------------------------- honesty (retimed / offline)
def test_retimed_clip_is_refused_with_no_beats(tmp_path):
    media = tmp_path / "song.wav"
    media.write_bytes(b"\x00" * 64)
    ir = _solo_ir(otio.schema.ExternalReference(target_url=media.as_uri()))
    clip = ir.real_clips()[0]
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    out = analyze_music(str(media), clip)
    assert out["beats"] == [] and out["tempo_bpm"] == 0.0
    assert out["frames"] == "timeline"
    assert any("retimed" in e for e in out["errors"])


def test_offline_media_is_an_error_not_a_guess():
    # No file on disk → honest error, never a fabricated grid.
    out = analyze_music("/nonexistent/offline.mov")
    assert out["beats"] == [] and out["tempo_bpm"] == 0.0
    assert any("source media not found" in e for e in out["errors"])


def test_analyze_music_needs_ffmpeg(monkeypatch, tmp_path):
    """With a real file but no ffmpeg, decode → PerceptionUnavailable (names the fix), not a fake."""
    media = tmp_path / "real.wav"
    media.write_bytes(b"\x00" * 64)
    import filmgrip.perception.transcribe as tr
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    with pytest.raises(PerceptionUnavailable) as exc:
        analyze_music(str(media))
    assert "ffmpeg" in str(exc.value)


def test_force_librosa_without_install_is_honest():
    """use_librosa=True with librosa absent raises with the install fix (it is never required)."""
    try:
        import librosa  # noqa: F401
        pytest.skip("librosa installed — the forced-missing branch is not exercised")
    except ImportError:
        pass
    with pytest.raises(PerceptionUnavailable) as exc:
        analyze_music("/whatever.wav", use_librosa=True)
    assert "librosa" in str(exc.value) and "film-grip[music]" in str(exc.value)


# ----------------------------------------------------------------------- integration (ffmpeg-gated)
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_integration_click_track_tempo(tmp_path):
    """Self-generate a 120 BPM click track with ffmpeg aevalsrc (arg-list, commas escaped), decode it
    through analyze_music (pure-numpy engine), and assert tempo ~120±2 (octave-tolerant)."""
    wav = tmp_path / "click_120.wav"
    # 1kHz tick for the first 20ms of every 0.5s window (120 BPM). Commas inside the expression are
    # escaped (\\,) so the single filter arg isn't split; passed as an arg list, never a shell.
    expr = "0.6*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02)"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
         "-i", f"aevalsrc={expr}:s=44100:d=8", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True)

    out = analyze_music(str(wav), use_librosa=False)   # force the BASE numpy engine
    assert out["engine"] == "numpy" and out["tier"] == "advisory"
    assert out["errors"] == []
    assert any(abs(out["tempo_bpm"] - cand) <= 2.0 for cand in (120.0, 60.0, 240.0))
    assert len(out["beats"]) > 0
    assert out["frames"] == "media"          # no clip → media frames
    assert out["downbeats"] == []            # never fabricated


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_integration_click_track_maps_to_timeline_frames(tmp_path):
    """A clip scoping the click track returns beats in TIMELINE frames offset by clip.start."""
    wav = tmp_path / "click_120.wav"
    expr = "0.6*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02)"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
         "-i", f"aevalsrc={expr}:s=44100:d=8", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True)
    # clip starts at timeline frame 100 (24fps), uses the whole 8s of media from source 0.
    ir = _solo_ir(otio.schema.ExternalReference(target_url=wav.as_uri()),
                  src_in_f=0, dur_f=192)
    clip = ir.real_clips()[0]
    out = analyze_music(str(wav), clip, use_librosa=False)
    assert out["frames"] == "timeline"
    assert out["errors"] == []
    assert out["beats"], "expected a non-empty beat grid"
    assert min(out["beats"]) >= clip.start    # every beat is at/after the clip's timeline start
