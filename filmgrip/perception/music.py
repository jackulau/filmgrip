"""Audio musicality — beat / tempo / onset perception, in pure numpy (librosa is an optional upgrade).

The rhythm sibling of :mod:`filmgrip.perception.scopes` (color → numbers) and
:mod:`filmgrip.perception.motion` (temporal structure → numbers): it reduces a clip's *audio* to a
beat grid + tempo a model can reason over and **verify against**, so the planner can propose honest
music-synced edits ("cut on the beat", "land each clip on the drop") and check them.

It is built on :func:`filmgrip.perception.audio_io.decode_pcm` (the project's float-PCM primitive) and
mirrors :mod:`filmgrip.perception.align`'s media-seconds→timeline-frames projection **and its honesty
rules** verbatim, so beats map to timeline frames exactly the way aligned words do.

Capability tiers (declared, never silently assumed — ``docs/research/audio-musicality.md`` §"Honesty"):

* **BASE = pure numpy, advisory.** :func:`onset_envelope` (spectral-flux onset strength via
  ``numpy.fft.rfft``) and :func:`detect_beats` (autocorrelation tempo + beat phase) need only the base
  install — no librosa, no extra. On steady/percussive material this is a trustworthy advisory grid;
  on rubato/classical it degrades like every onset method (treat as possibly undefined, not merely
  low-accuracy). This is the default and what the tests prove.
* **UPGRADE = librosa, better.** When ``librosa`` is importable (the optional ``[music]`` extra) and
  ``use_librosa`` is not ``False``, :func:`analyze_music` uses ``librosa.beat.beat_track`` (Ellis DP)
  for higher-accuracy beats/tempo. librosa is **never required** — its absence simply keeps the numpy
  path. ``downbeats`` are best-effort only (no detector exists in either tier); we do not fabricate them.

Honesty (same wall as :func:`align.is_retimed` / :func:`motion.analyze_motion`):

* a **retimed** clip (LinearTimeWarp/FreezeFrame) breaks the linear media↔timeline mapping, so it comes
  back as an ``errors`` entry with **no fabricated beats** — never a guessed grid;
* **offline / missing media** (no resolvable path, or a decode failure) is an ``errors`` entry, not a
  fake result;
* missing **ffmpeg** or **numpy** raises :class:`~filmgrip.perception.transcribe.PerceptionUnavailable`
  with the fix, matching the rest of perception. numpy is a lazy guard.
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:                                  # numpy powers the DSP + sample buffer; optional install.
    import numpy as np
except ImportError:                   # pragma: no cover - exercised only on a numpy-less install
    np = None

from .audio_io import DEFAULT_RATE, decode_pcm
from .transcribe import PerceptionUnavailable

#: Default onset-envelope hop — 10ms frames give a grid fine enough to resolve beats up to ~200 BPM.
DEFAULT_HOP_S = 0.01

#: Default plausible musical tempo window (BPM). Autocorrelation is octave-aware inside this range.
DEFAULT_TEMPO_RANGE = (60.0, 200.0)


def _need_numpy() -> None:
    if np is None:
        raise PerceptionUnavailable(
            "music analysis needs numpy — install it with: pip install 'film-grip[music]'")


# --------------------------------------------------------------------------- pure DSP cores
def onset_envelope(samples: Any, sr: int, *, hop_s: float = DEFAULT_HOP_S) -> Any:
    """Spectral-flux onset-strength envelope of ``samples`` — pure numpy, one value per hop.

    The standard feature-based onset detector (Bello 2005, librosa's default): frame the signal,
    take the magnitude spectrum of each frame with :func:`numpy.fft.rfft`, and sum the
    **half-wave-rectified** frame-to-frame magnitude difference over bins —
    ``SF(n) = Σ_k max(0, |X(n,k)| − |X(n-1,k)|)`` — so the function spikes on energy *increases*
    (onsets) and ignores decays/offsets. Stereo ``(n, 2)`` is averaged to mono first. Returns a 1-D
    float32 envelope normalized to a unit max (empty when the signal is shorter than one hop).

    Pure function (no IO), so peak-picking / tempo correctness is unit-testable on a synthetic signal
    with no ffmpeg and no librosa — exactly like :func:`filmgrip.perception.audio_io.rms_envelope`.
    """
    _need_numpy()
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim > 1:                       # stereo (or more) → mono by channel average
        a = a.mean(axis=1)
    hop = max(1, int(round(sr * hop_s)))
    # Window of ~4 hops gives spectral resolution without over-smoothing transients; FFT to next pow2.
    win = max(hop * 4, 256)
    n_fft = 1 << (int(win) - 1).bit_length()
    n_hops = (a.size - n_fft) // hop + 1
    if n_hops < 2:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(n_fft)
    # Build the (n_hops, n_fft) frame matrix via a strided view, window it, one batched rFFT.
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_hops)[:, None]
    frames = a[idx] * window[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    diff = mag[1:] - mag[:-1]
    flux = np.maximum(diff, 0.0).sum(axis=1)          # half-wave rectified spectral difference
    flux = np.concatenate([[0.0], flux])              # align length to n_hops (first frame has no prev)
    peak = float(flux.max())
    if peak > 0:
        flux = flux / peak
    return flux.astype(np.float32)


def _autocorrelate(x: "np.ndarray") -> "np.ndarray":
    """Unbiased one-sided autocorrelation of a mean-removed signal (pure numpy)."""
    x = x - x.mean()
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1:]      # lags 0..n-1
    # Unbias: divide each lag by its overlap count so long lags aren't artificially suppressed.
    counts = np.arange(n, 0, -1)
    return ac / counts


def _estimate_lag(env: "np.ndarray", hop_s: float,
                  tempo_range: tuple[float, float]) -> int:
    """Dominant autocorrelation lag (in hops) within ``tempo_range`` BPM, octave-resolved; 0 if none.

    Shared by :func:`estimate_tempo` and :func:`detect_beats` so the standalone tempo and the beat grid
    can never disagree about the period.
    """
    lo_bpm, hi_bpm = float(min(tempo_range)), float(max(tempo_range))
    # lag (hops) ↔ BPM:  bpm = 60 / (lag * hop_s)  ⇒  lag = 60 / (bpm * hop_s)
    min_lag = max(1, int(np.floor(60.0 / (hi_bpm * hop_s))))
    max_lag = min(env.size - 1, int(np.ceil(60.0 / (lo_bpm * hop_s))))
    if max_lag <= min_lag:
        return 0
    ac = _autocorrelate(env)
    lags = np.arange(min_lag, max_lag + 1)
    scores = ac[min_lag:max_lag + 1]
    if scores.size == 0 or float(scores.max()) <= 0.0:
        return 0
    best_lag = int(lags[int(np.argmax(scores))])
    return _resolve_octave(ac, best_lag, min_lag, max_lag)


def estimate_tempo(onset_env: Any, sr: int, hop_s: float = DEFAULT_HOP_S,
                   *, tempo_range: tuple[float, float] = DEFAULT_TEMPO_RANGE) -> float:
    """Dominant tempo (BPM) of an onset-strength envelope — pure numpy; ``0.0`` if flat/indeterminate.

    The tempo half of :func:`detect_beats` exposed on its own (the named entry downstream callers and
    the gate import): autocorrelate ``onset_env``, take the strongest periodicity inside ``tempo_range``,
    octave-correct it. ``sr`` is accepted for signature symmetry with the other readers; the envelope's
    time base is set by ``hop_s``.
    """
    _need_numpy()
    env = np.asarray(onset_env, dtype=np.float64).ravel()
    if env.size < 4 or float(env.max()) <= 0.0 or float(np.ptp(env)) <= 1e-9:
        return 0.0
    lag = _estimate_lag(env, hop_s, tempo_range)
    return round(60.0 / (lag * hop_s), 2) if lag > 0 else 0.0


def detect_beats(onset_env: Any, sr: int, hop_s: float,
                 *, tempo_range: tuple[float, float] = DEFAULT_TEMPO_RANGE) -> dict:
    """Estimate tempo + beat positions from an onset-strength envelope — pure numpy.

    Tempo is the dominant periodicity of ``onset_env``: autocorrelate the envelope, restrict the lag
    to ``tempo_range`` BPM, pick the peak, then resolve octave ambiguity (autocorrelation structurally
    favours sub-harmonics — half/third tempo) by preferring an in-range ½×/2× candidate that has
    comparable or stronger support. Beats are a constant-period grid at that tempo, phase-locked to the
    onset envelope by choosing the offset that maximizes summed onset strength on the grid.

    Returns ``{"tempo_bpm", "beat_times_s", "onset_times_s"}`` (times in **media seconds**). A flat or
    too-short envelope yields ``tempo_bpm = 0.0`` and empty lists rather than a fabricated grid.
    """
    _need_numpy()
    env = np.asarray(onset_env, dtype=np.float64).ravel()
    onset_times = _pick_onsets(env, hop_s)
    if env.size < 4 or float(env.max()) <= 0.0 or float(np.ptp(env)) <= 1e-9:
        return {"tempo_bpm": 0.0, "beat_times_s": [], "onset_times_s": onset_times}

    best_lag = _estimate_lag(env, hop_s, tempo_range)
    if best_lag <= 0:
        return {"tempo_bpm": 0.0, "beat_times_s": [], "onset_times_s": onset_times}
    tempo_bpm = 60.0 / (best_lag * hop_s)

    beat_idx = _beat_grid(env, best_lag)
    beat_times = [round(float(i * hop_s), 4) for i in beat_idx]
    return {"tempo_bpm": round(tempo_bpm, 2), "beat_times_s": beat_times,
            "onset_times_s": onset_times}


def _resolve_octave(ac: "np.ndarray", lag: int, min_lag: int, max_lag: int) -> int:
    """Octave-aware tempo correction: if ½× or 2× the candidate period sits in range with comparable
    autocorrelation support, prefer the one with the strongest peak. Counters the autocorrelation's
    structural bias toward sub-harmonics (the dominant real-world tempo error — half-tempo)."""
    candidates = {lag}
    for factor in (2, 3):
        if min_lag <= lag // factor <= max_lag:
            candidates.add(lag // factor)
        if min_lag <= lag * factor <= max_lag:
            candidates.add(lag * factor)
    base = float(ac[lag])
    best, best_score = lag, base
    for cand in candidates:
        score = float(ac[cand])
        # Accept a different octave only when its support is at least ~90% of the incumbent's, so a
        # weak harmonic never overrides a clear peak — but a near-tie resolves toward it.
        if score >= 0.9 * best_score and score > 0:
            if score > best_score or (abs(score - best_score) <= 0.1 * best_score and cand < best):
                best, best_score = cand, score
    return best


def _beat_grid(env: "np.ndarray", period: int) -> list[int]:
    """Place a constant-``period`` beat grid over the envelope, phased to the offset that maximizes
    total onset strength on the grid points (a coarse beat-phase estimate; pure numpy)."""
    if period < 1:
        return []
    n = env.size
    offsets = np.arange(period)
    # For each candidate phase, sum env at indices phase, phase+period, ...; pick the strongest phase.
    best_phase, best_energy = 0, -1.0
    for phase in offsets:
        grid = np.arange(phase, n, period)
        energy = float(env[grid].sum())
        if energy > best_energy:
            best_phase, best_energy = int(phase), energy
    return list(range(best_phase, n, period))


def _pick_onsets(env: "np.ndarray", hop_s: float) -> list[float]:
    """Local-maxima onset times (media seconds) above an adaptive threshold — pure numpy peak-pick."""
    if env.size < 3 or float(env.max()) <= 0.0:
        return []
    thresh = float(env.mean()) + 0.5 * float(env.std())
    prev, nxt = env[:-2], env[2:]
    mid = env[1:-1]
    is_peak = (mid > prev) & (mid >= nxt) & (mid >= thresh)
    idx = np.nonzero(is_peak)[0] + 1
    return [round(float(i * hop_s), 4) for i in idx]


# --------------------------------------------------------------------------- timeline projection
def beats_to_frames(times_s: Any, rate: float, *, clip_offset_frames: int = 0) -> list[int]:
    """Project event times (MEDIA seconds) onto integer TIMELINE frames.

    The single-time analog of :func:`filmgrip.perception.align.align_clip_words`: a time ``t`` maps to
    ``clip_offset_frames + round(t * rate)``. For a clip, pass the seconds *into the clip's source
    window* (``t − clip_in_s``) and ``clip_offset_frames = clip.start``; with the defaults it is a
    plain media-time→frame conversion. Pure arithmetic — no IO.
    """
    _need_numpy()
    arr = np.asarray(list(times_s), dtype=np.float64)
    if arr.size == 0:
        return []
    return [int(clip_offset_frames + int(round(t * rate))) for t in arr]


# --------------------------------------------------------------------------- the reader
def analyze_music(media_path: str, clip: Any = None, *, rate: int = DEFAULT_RATE,
                  use_librosa: Optional[bool] = None) -> dict:
    """Read beat / tempo / onset structure from a media file (optionally scoped to one timeline clip).

    Pipeline: :func:`~filmgrip.perception.audio_io.decode_pcm` → :func:`onset_envelope` →
    :func:`detect_beats`. When ``librosa`` is importable **and** ``use_librosa`` is not ``False`` the
    higher-accuracy ``librosa.beat.beat_track`` path is used instead of the numpy tempo/beat picker;
    librosa is **never required** — its absence (or ``use_librosa=False``) keeps the pure-numpy path.
    Set ``use_librosa=True`` to force it (raising :class:`PerceptionUnavailable` if it is missing).

    Returns a JSON-able dict::

        {"engine": "numpy"|"librosa", "tier": "advisory"|"better", "rate", "frames",
         "tempo_bpm": float,
         "beats": [...],            # TIMELINE frames when a clip is given, else MEDIA frames
         "downbeats": [...],        # best-effort only — empty unless detected; never fabricated
         "onsets": [...],
         "errors": [...]}

    With ``clip`` supplied the decode is scoped to the clip's source window and beats are returned in
    **timeline frames** (media seconds → timeline frame via the clip's start, reusing ``align``'s
    convention). A **retimed clip** or **offline/missing media** comes back as an ``errors`` entry with
    **no beats** — same honesty wall as ``align``/``motion``. :class:`PerceptionUnavailable` is raised
    when ffmpeg or numpy is absent.
    """
    _need_numpy()
    # Resolve the engine first: a forced-but-missing librosa must fail fast with the install fix,
    # before any clip/media-path branch can mask it with a different error.
    use_lib = _should_use_librosa(use_librosa)

    ss = 0.0
    duration: Optional[float] = None
    base_timeline_frame: Optional[int] = None
    timeline_rate: Optional[float] = None

    if clip is not None:
        # Refuse retimed clips honestly — a time-warp makes every mapped beat a lie (mirrors align).
        from .align import is_retimed
        if is_retimed(clip):
            return _empty_payload(
                rate, framed=True,
                error=f"{getattr(clip, 'id', '?')}: retimed clip — beat→frame mapping would be "
                      f"wrong, refused (remove the retime or address it by frames)")
        src_rate = _source_rate(clip, rate)
        ss = float(clip.source_start) / src_rate if src_rate > 0 else 0.0
        timeline_rate = src_rate
        duration = float(clip.duration) / src_rate if src_rate > 0 else None
        base_timeline_frame = int(clip.start)

    # Offline / unresolvable media is an honest error, not a fabricated grid (mirrors transcribe).
    if not media_path or not os.path.isfile(media_path):
        return _empty_payload(
            rate, framed=base_timeline_frame is not None,
            error=f"{getattr(clip, 'id', '?') if clip is not None else media_path}: source media not "
                  f"found ('{media_path}') — offline/proxy-only media cannot be analyzed")

    samples, sr = decode_pcm(media_path, rate=rate, mono=True, start_s=ss, dur_s=duration)

    if use_lib:
        engine, tier = "librosa", "better"
        tempo_bpm, beat_times, onset_times = _librosa_beats(samples, sr)
    else:
        engine, tier = "numpy", "advisory"
        env = onset_envelope(samples, sr, hop_s=DEFAULT_HOP_S)
        result = detect_beats(env, sr, DEFAULT_HOP_S)
        tempo_bpm = result["tempo_bpm"]
        beat_times = result["beat_times_s"]
        onset_times = result["onset_times_s"]

    # Project media-second events → frames. With a clip, t is seconds-into-clip and the offset is
    # clip.start; otherwise it is a plain media-time→frame mapping (offset 0).
    out_rate = timeline_rate if (timeline_rate and timeline_rate > 0) else float(rate)
    offset = base_timeline_frame if base_timeline_frame is not None else 0
    beats = beats_to_frames(beat_times, out_rate, clip_offset_frames=offset)
    onsets = beats_to_frames(onset_times, out_rate, clip_offset_frames=offset)

    return {
        "engine": engine,
        "tier": tier,
        "rate": _rate_str(out_rate),
        "frames": "timeline" if base_timeline_frame is not None else "media",
        "tempo_bpm": tempo_bpm,
        "beats": beats,
        "downbeats": [],                 # no detector in either tier — never fabricated (see docstring)
        "onsets": onsets,
        "source": os.path.basename(media_path),
        "errors": [],
    }


def beats_for_media(media_path: str, clip: Any = None, *, rate: int = DEFAULT_RATE,
                    use_librosa: Optional[bool] = None) -> dict:
    """IO reader entry — beats / tempo / onsets for a media file (optionally scoped to one clip).

    Thin alias of :func:`analyze_music`, named to mirror align's ``*_for_media`` convention — the name
    the ``beats`` CLI (D6), the deterministic beat-cut pack (D15), and the MCP ``payload_get_beats``
    import. Same honesty wall: retimed/offline clips come back as ``errors`` with no fabricated beats.
    """
    return analyze_music(media_path, clip, rate=rate, use_librosa=use_librosa)


# --------------------------------------------------------------------------- librosa upgrade
def _should_use_librosa(use_librosa: Optional[bool]) -> bool:
    """Resolve the engine: ``True`` forces librosa (raise if missing); ``False`` forces numpy;
    ``None`` uses librosa only when it is importable."""
    if use_librosa is False:
        return False
    try:
        import librosa  # noqa: F401
    except ImportError:
        if use_librosa is True:
            raise PerceptionUnavailable(
                "use_librosa=True but librosa is not installed — install the higher-accuracy engine "
                "with: pip install 'film-grip[music]' (the pure-numpy path is the default otherwise)")
        return False
    return True


def _librosa_beats(samples: "np.ndarray", sr: int) -> tuple[float, list[float], list[float]]:
    """Higher-accuracy tempo/beats/onsets via librosa (Ellis DP beat tracker). Optional path."""
    import librosa

    y = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    tempo_val = float(np.asarray(tempo).ravel()[0]) if np.size(tempo) else 0.0
    return (round(tempo_val, 2),
            [round(float(t), 4) for t in np.asarray(beat_times).ravel()],
            [round(float(t), 4) for t in np.asarray(onset_times).ravel()])


# --------------------------------------------------------------------------- helpers
def _empty_payload(rate: float, *, framed: bool, error: str) -> dict:
    return {
        "engine": "numpy", "tier": "advisory", "rate": _rate_str(float(rate)),
        "frames": "timeline" if framed else "media",
        "tempo_bpm": 0.0, "beats": [], "downbeats": [], "onsets": [], "errors": [error],
    }


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


def _rate_str(rate: float) -> str:
    """Integer rates render as "24"; non-integer rates keep their numeric form (mirrors fgx._rate_str)."""
    if abs(rate - round(rate)) < 1e-6:
        return str(int(round(rate)))
    return str(rate)
