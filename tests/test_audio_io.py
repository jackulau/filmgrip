"""D3 — audio primitive. rms_envelope is pure numpy, so its correctness is proved on a synthetic
signal (a silent run followed by a sine burst) with no ffmpeg and no media file. The ffmpeg decode
path is exercised when ffmpeg is present (lavfi synthesizes a 1s sine), else skipped; the
ffmpeg-absent branch is forced to prove decode_pcm raises an actionable error, not a fake result."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from filmgrip.perception.audio_io import decode_pcm, rms_envelope
from filmgrip.perception.transcribe import PerceptionUnavailable, ffmpeg_path


# ----------------------------------------------------------------------- rms_envelope (pure DSP)
def test_rms_envelope_silent_then_burst():
    sr, hop_s = 22050, 0.01
    hop = int(round(sr * hop_s))
    # 50 hops of silence, then 50 hops of a full-scale 440Hz sine burst.
    quiet = np.zeros(50 * hop, dtype=np.float32)
    t = np.arange(50 * hop, dtype=np.float32) / sr
    burst = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    env = rms_envelope(np.concatenate([quiet, burst]), sr, hop_s=hop_s)

    assert env.shape == (100,)                      # 100 full hops, sub-hop remainder dropped
    assert env.dtype == np.float32
    assert np.all(env[:50] == 0.0)                  # silent region reads exactly zero energy
    assert np.all(env[50:] > 0.5)                   # burst region is elevated (sine RMS ≈ 0.707)
    assert env[50:].mean() > 10 * env[:50].mean() + 1e-6   # burst >> silence


def test_rms_envelope_averages_stereo_channels():
    sr = 22050
    n = sr // 10                                    # 0.1s
    left = np.full(n, 0.4, dtype=np.float32)
    right = np.full(n, -0.4, dtype=np.float32)      # averages to 0.0 per sample
    stereo = np.stack([left, right], axis=1)        # shape (n, 2)
    env = rms_envelope(stereo, sr, hop_s=0.01)
    assert env.ndim == 1
    assert np.allclose(env, 0.0)                     # channel mean cancels → zero energy


def test_rms_envelope_sub_hop_signal_is_empty():
    env = rms_envelope(np.ones(5, dtype=np.float32), sr=22050, hop_s=0.01)  # 5 < one hop (220)
    assert env.shape == (0,)
    assert env.dtype == np.float32


# ----------------------------------------------------------------------- decode_pcm honesty (no ffmpeg)
def test_decode_pcm_honest_without_ffmpeg(monkeypatch):
    # ffmpeg_path() resolves via transcribe.shutil.which — force it to find nothing.
    import filmgrip.perception.transcribe as tr
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    with pytest.raises(PerceptionUnavailable) as exc:
        decode_pcm("clip.mov")
    assert "ffmpeg" in str(exc.value)               # names the missing dep + the fix
    assert "brew install ffmpeg" in str(exc.value)


def test_decode_pcm_raises_with_stderr_on_bad_media(tmp_path):
    if ffmpeg_path() is None:
        pytest.skip("ffmpeg not installed")
    bogus = tmp_path / "not-audio.wav"
    bogus.write_bytes(b"this is not media")
    with pytest.raises(PerceptionUnavailable) as exc:
        decode_pcm(str(bogus))
    assert str(bogus) in str(exc.value)             # surfaces the path + ffmpeg's own stderr


# ----------------------------------------------------------------------- ffmpeg decode (guarded)
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_pcm_synth_sine(tmp_path):
    # ffmpeg synthesizes a 1s 440Hz sine wav; decode it back to float PCM and check the contract.
    wav = tmp_path / "sine.wav"
    import subprocess
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", str(wav)],
        check=True, capture_output=True)

    samples, sr = decode_pcm(str(wav), rate=22050, mono=True)
    assert sr == 22050
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    # ~1s at 22050 — allow a little ffmpeg edge slack on the exact sample count.
    assert abs(samples.shape[0] - 22050) <= 256
    assert float(np.max(np.abs(samples))) <= 1.0 + 1e-6   # stays in [-1, 1]
    # ffmpeg's lavfi `sine` defaults to amplitude 0.1, so peak ≈ 0.1 and RMS ≈ 0.07.
    assert float(np.max(np.abs(samples))) > 0.05          # a real sine, not silence
    # the pure envelope rides on the decoded samples end-to-end
    env = rms_envelope(samples, sr)
    assert env.size > 0 and float(env.mean()) > 0.05


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_pcm_stereo_shape_and_trim(tmp_path):
    wav = tmp_path / "sine.wav"
    import subprocess
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", str(wav)],
        check=True, capture_output=True)

    # stereo → (n, 2); start_s + dur_s trims to roughly the requested half-second window.
    samples, sr = decode_pcm(str(wav), rate=22050, mono=False, start_s=0.25, dur_s=0.5)
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert abs(samples.shape[0] - int(0.5 * 22050)) <= 512
