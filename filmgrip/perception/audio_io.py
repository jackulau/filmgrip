"""Audio perception — float-PCM decode + an RMS energy envelope, the missing foundation for all
musicality and waveform-energy work.

Everything rhythm-aware in film-grip (beat-snapped cuts, music-driven montage, "cut on the drop",
silence energy beyond ASR gaps) needs the *samples*, not the timeline graph. This module is that
primitive: it shells ffmpeg to decode any media to raw 32-bit float PCM in ``[-1, 1]`` — the lingua
franca every DSP routine wants — and reduces it to a per-hop energy envelope.

It is the audio sibling of :mod:`filmgrip.perception.scopes` (color pixels → numbers) and is built
to the same honesty rules:

* **raw float, not ASR-WAV** — :func:`~filmgrip.perception.transcribe.extract_wav` makes the mono-16k
  WAV that speech engines want; :func:`decode_pcm` instead returns float32 samples at a musicality
  rate (22.05k by default) with optional stereo, so it loses nothing the downstream needs.
* **pure DSP is provable offline** — :func:`rms_envelope` is numpy in → numpy out, so its correctness
  is unit-testable on a synthetic signal with no ffmpeg and no media file.
* **degrade to an actionable error, never a fake result** — a missing ffmpeg or a failed decode
  raises :class:`~filmgrip.perception.transcribe.PerceptionUnavailable` with the fix (or the stderr),
  matching the rest of perception. numpy is a lazy guard (``pip install 'film-grip[color]'``).
* **arg-list, never a shell** — the ffmpeg call is an argument list (no ``shell=True``) and the media
  path is passed positionally after ``-i``, so a path beginning with ``-`` can never be parsed as a
  flag. This mirrors :func:`filmgrip.perception.scopes.frame_rgb`.
"""
from __future__ import annotations

import subprocess
from typing import Any, Optional

try:                                  # numpy powers the sample buffer + DSP; optional install.
    import numpy as np
except ImportError:                   # pragma: no cover - exercised only on a numpy-less install
    np = None

from .transcribe import PerceptionUnavailable, ffmpeg_path

#: Default decode rate — high enough to preserve musical transients (beats/onsets) while staying far
#: cheaper than 44.1k; this is the rate the musicality layer reasons over.
DEFAULT_RATE = 22050

#: Default RMS hop — 10ms windows give an envelope fine enough to find beats and silence edges.
DEFAULT_HOP_S = 0.01


def _need_numpy() -> None:
    if np is None:
        raise PerceptionUnavailable(
            "audio decode needs numpy — install it with: pip install 'film-grip[color]'")


def decode_pcm(media_path: str, *, rate: int = DEFAULT_RATE, mono: bool = True,
               start_s: float = 0.0, dur_s: Optional[float] = None) -> tuple[Any, int]:
    """Decode any media to raw 32-bit float PCM in ``[-1, 1]`` via ffmpeg, returning
    ``(samples, sample_rate)``.

    ``samples`` is a float32 numpy array: shape ``(n,)`` when ``mono``, else ``(n, 2)``. Decoding
    starts at ``start_s`` (seconds) and is limited to ``dur_s`` seconds when given. This is the raw
    foundation for the musicality / waveform-energy layer — deliberately NOT the mono-16k ASR WAV of
    :func:`~filmgrip.perception.transcribe.extract_wav`.

    Raises :class:`~filmgrip.perception.transcribe.PerceptionUnavailable` when ffmpeg is absent (with
    the install fix) or the decode fails (with ffmpeg's stderr). The call is an argument list (never
    ``shell=True``) and the media path is passed positionally after ``-i``, so a leading-``-`` path
    can never be misread as a flag.
    """
    _need_numpy()
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable(
            "ffmpeg is required to decode audio — install it (e.g. `brew install ffmpeg`) and re-run.")
    channels = 1 if mono else 2
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, start_s):.6f}"]
    if dur_s is not None:
        cmd += ["-t", f"{max(0.0, dur_s):.6f}"]
    # Path positionally after -i; output is headerless little-endian float32 on stdout.
    cmd += ["-i", media_path, "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(channels), "-ar", str(rate), "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise PerceptionUnavailable(
            f"ffmpeg could not decode audio from '{media_path}': "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:300]}")
    samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not mono:
        samples = samples.reshape(-1, 2)
    return samples, int(rate)


def rms_envelope(samples: Any, sr: int, hop_s: float = DEFAULT_HOP_S) -> Any:
    """Per-hop root-mean-square energy of ``samples`` — a 1-D float32 envelope, one value per hop.

    Pure function (no IO): the energy basis for silence detection and beat/onset strength. Windows
    are non-overlapping and ``hop_s`` seconds wide. Stereo input ``(n, 2)`` is averaged to mono
    first. A short trailing remainder (less than one hop) is dropped so every value covers a full
    window; an empty or sub-hop signal yields an empty envelope.
    """
    _need_numpy()
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim > 1:                       # stereo (or more) → mono by channel average
        a = a.mean(axis=1)
    hop = max(1, int(round(sr * hop_s)))
    n_hops = a.size // hop
    if n_hops == 0:
        return np.zeros(0, dtype=np.float32)
    frames = a[: n_hops * hop].reshape(n_hops, hop)
    return np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
