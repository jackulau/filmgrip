# Audio Musicality + Music-Synced Editing — Research & film-grip Design

Research for adding **beat/tempo/onset/key perception** and **music-synced editing** to film-grip,
in the same shape as the existing perception layer (deterministic readers → typed `EditPlan` ops →
verify loop). Every quantitative claim is cited; URLs are collected under **Sources**.

> **Key codebase finding up front.** `filmgrip/perception/audio_io.py` already exists and was
> written *for this feature*: its docstring says it is "the missing foundation for all musicality and
> waveform-energy work … beat-snapped cuts, music-driven montage, 'cut on the drop'." It provides
> `decode_pcm(media) -> (float32 samples, sr)` and `rms_envelope(samples, sr, hop_s)`. The reader
> proposed here builds directly on these — no new ffmpeg plumbing required.

---

## Summary

- **Recommended dependency: `librosa` as a new optional extra** (call it `music`, mirroring the
  existing `transcribe` / `color` extras). It is the only credible candidate that is **permissively
  licensed (ISC), fully offline (pure-DSP, no model download), torch/tensorflow-free, and maintained
  for Python 3.13 + numpy 2.x** (v0.11.0, 2025-03). The accuracy leaders (madmom, BeatNet) are
  **disqualified**: madmom's pretrained models are **CC BY-NC-SA (non-commercial)** *and* its only
  PyPI release (2018) won't install on Python ≥3.10; aubio is **GPL**; essentia is **AGPL**; BeatNet
  drags in torch + madmom. See the comparison table and the explicit librosa-vs-madmom tradeoff.
- **Highest-value capabilities (in priority order):**
  1. **`analyze_music()` reader** — a deterministic reader that decodes a clip's media, runs
     librosa beat/tempo/onset/key, and (reusing `align.py`'s media↔timeline projection) returns
     **`tempo_bpm`, `beats[]`, `downbeats[]`, `onsets[]`, `key` mapped to TIMELINE frames per clip**.
     This is the moat piece — the "hear the music" analog of `scopes.analyze_rgb` / transcripts.
  2. **`snap_cuts` op + `beat-snap` pack** — snap existing cut points (or selected clip edges) to the
     nearest beat/downbeat within a tolerance window. Deterministic, zero-LLM, compiles to the
     existing `trim`/`move`/`cut_range` primitives. The single most-requested music-edit operation.
  3. **Honest retime warning surface** — `analyze_music` is also where we compute and surface the
     **pitch-shift / "chipmunk" warning** for `retime` ops on clips with linked audio (see below).
- **Verify strategy:** synthesize a **click track at a known BPM with ffmpeg** (`aevalsrc`, no new
  deps), assert detected `tempo_bpm` within tolerance (±2 BPM, with octave-aware acceptance of
  ½×/2×), assert each detected beat lands within **N ms** of the known grid, and assert the
  **media→timeline frame mapping** matches a hand-computed expected frame for a clip with a known
  `source_start`. Pure-numpy synthesis of a click envelope makes the peak-picking unit-testable with
  no ffmpeg and no librosa, exactly like `scopes`/`motion`/`audio_io` already do.

---

## Algorithms

### 1. Onset detection

Every feature-based onset detector is three stages (Bello et al. 2005): pre-process → reduce to a
1-D **detection / onset-strength function** that spikes on spectral change → **peak-pick** (normalize,
smooth, adaptive threshold, local-max with a minimum inter-onset gap). You never threshold the raw
waveform; you build an intermediate novelty signal.

- **Spectral flux** (the modern default; librosa's): per-frame STFT magnitude, frame-to-frame
  difference, **half-wave rectify** (keep only energy *increases* → responds to onsets, ignores
  offsets/decays), sum/mean over bins. `SF(n) = Σ_k H(|X(n,k)| − |X(n−1,k)|)`,
  `H(x)=(x+|x|)/2`. Simplest and fastest; weak on soft/bowed/low-pitched onsets and fooled by vibrato.
- **High-Frequency Content (HFC):** weight each bin by its index `k` before summing →
  sharp peaks on **percussive/broadband** attacks, poor on low/legato onsets (the `k`-weight
  suppresses the bass bins where those onsets live).
- **Complex-domain:** predict each bin assuming constant magnitude *and* linear phase advance, measure
  Euclidean distance prediction-vs-observation → fires on a magnitude jump (percussive) **or** a
  phase-trajectory break (a new pitched note, even at constant amplitude → catches soft/tonal onsets).
  Reduces exactly to spectral flux when phase is predictable.
- **SuperFlux** (Böck & Widmer 2013): spectral flux on a log-filtered spectrogram with a **±1-bin
  frequency maximum-filter** on the reference frame → suppresses vibrato false positives by up to
  **60%** on strings/opera with no extra misses.

**Accuracy reality (this is the load-bearing fact for honesty tiers).** Onset F-measure is **bimodal
by onset type**, and it has been for 20 years across the shift to deep learning. MIREX-2005 per-class
F-measure: bells/percussive **0.99**, solo drum **0.92**, plucked string 0.84, **sustained (bowed)
strings 0.58**, **singing voice 0.45**. Overall best systems cluster **~0.74–0.84** at the strict
±25 ms tolerance (≈0.88–0.97 at ±50 ms on easy/piano material); neural SOTA ≈ **0.85–0.90** on mixed
material. Percussive onset detection is essentially solved (~0.95+); **soft/bowed/vocal onset
detection is the open problem (0.45–0.66)**. (Bello 2005; Böck 2012/2013; MIREX onset results.)

### 2. Tempo / BPM estimation

Runs on the **onset-strength envelope**, not raw audio. Find the dominant periodicity, convert lag →
BPM (`BPM = 60 / lag_seconds`).

- **Autocorrelation of the onset envelope:** strong peak at the beat period (and its multiples).
  Structurally **emphasizes sub-harmonics** (half/third tempo). librosa disambiguates with a
  **log-normal prior centered at 120 BPM** (`start_bpm=120`, `std_bpm=1.0`).
- **Comb-filter / resonator bank (Scheirer 1998):** a bank of IIR resonators `y[n]=αy[n−T]+(1−α)x[n]`,
  one per candidate period; the resonator whose `T` matches the beat builds the most energy.
  Phase-preserving, so its state doubles as the beat-phase predictor.
- **Fourier tempogram:** STFT of the novelty function; **emphasizes tempo *harmonics*** (the dual of
  ACF). Combining ACF + Fourier cancels spurious octave candidates; the **cyclic tempogram** folds
  octave-equivalent tempi like chroma folds pitch.

**Accuracy — Acc1 vs Acc2, and why octave error dominates.** **Acc1** = within ±4% of the single
ground-truth tempo. **Acc2** = also counts ½×, 2×, ⅓×, 3× as correct — i.e. it **forgives
octave/metrical-level errors**. (Distinct from the MIREX P-score, which uses ±8% and two ground-truth
tempi.) Concrete (Schreiber & Müller 2018): Ballroom Acc1 92.0 / Acc2 98.4; pop (ACM Mirum) 79.5 /
97.4; EDM (GiantSteps) 73.0 / 89.3; mixed (GTZAN) 69.4 / 92.6; deliberately-hard (SMC) 33.6 / 50.2.
**Takeaway: expect Acc1 ≈ 70–80% / Acc2 ≈ 92–95% on mixed music**, with a persistent ~15–20-point
gap that is almost entirely **half-tempo confusion** — the single biggest error mode, present even on
easy genres. *This is why our `snap_cuts` op snaps to a beat phase, not an absolute BPM number, and
why the verify accepts ½×/2× tempo.*

### 3. Beat tracking — the algorithm librosa uses

`librosa.beat.beat_track` is **Ellis (2007) dynamic-programming beat tracking**: (1) onset strength,
(2) global tempo via autocorrelation × log-Gaussian prior, (3) DP that picks beats maximizing

```
C({t_i}) = Σ_i O(t_i)  +  α · Σ_i F(t_i − t_{i−1}, τ_p),   F(Δt,τ) = −(log(Δt/τ))²
```

— a **local-match** term (beats on onset peaks) plus a **transition cost** (squared-log penalty,
symmetric so doubling/halving cost the same) tuned by **`tightness` α (librosa default 100)**. The DP
+ backtrace is guaranteed-optimal in linear time. **Documented limitation:** it assumes a single
global tempo and "there is no notion of tracking a slowly-varying tempo … abrupt changes are not
accommodated" — it tolerates only ±~10% drift. For variable tempo, librosa offers a per-frame `bpm=`
array or `librosa.beat.plp` (Predominant Local Pulse).

**Beat accuracy** (F-measure, ±70 ms; the standard `mir_eval.beat` metric, plus continuity CMLt/AMLt
where a beat is correct only if **both** phase and period are within 17.5%; the **AMLt−CMLt gap
quantifies octave error**): near-ceiling on steady/percussive material, collapses on rubato. Böck
RNN+DBN: Ballroom **0.92–0.94**, GTZAN 0.856, Hainsworth 0.867, **SMC 0.52**. 2024 SOTA "Beat This!":
Ballroom **0.975–0.986**, GTZAN 0.891, **SMC 0.627**. The easy/hard spread is a stable **~0.40
absolute F-measure gap** (≈0.9+ vs ≈0.4–0.6) that no model has closed — the audio onset signal, not
the model, is the limiter for expressive/classical material.

### 4. Downbeat detection / meter

Strictly harder than beat tracking: "which beat is beat 1 of the bar" depends on **harmonic/chord
changes, phrase boundaries, and the metrical hierarchy** (long-range cues), not local energy. So
every system pairs a deep frame feature extractor with a **structured temporal model**, and downbeat
F runs **10–25 points below beat F on the same audio**.

- **RNN + DBN bar-pointer (madmom):** a BLSTM emits beat/downbeat/non-beat activations; a Dynamic
  Bayesian Network "bar pointer" (hidden bar-position Φ + tempo + time-signature) advances each frame
  and wraps at the bar, with **downbeats = frames where Φ=1**, decoded by Viterbi. The DBN adds ~15%
  F over thresholding the raw RNN.
- **Transformers (2024 SOTA, "Beat This!"):** time/frequency "partial" attention + a shift-tolerant
  loss, doing beat **and** downbeat **without a DBN** (just peak-picking). Surpasses prior SOTA in F1;
  trades a little continuity (CMLt/AMLt) for the simpler post-processing.

Numbers (downbeat F, ±70 ms): madmom 2016 — Ballroom 0.863, Hainsworth 0.684, **GTZAN 0.640**; the
GTZAN downbeat-F progression is clearly upward — `0.640 (2016) → 0.672 (2020) → 0.745–0.756 (2022)
→ 0.783 (Beat This! 2024)` — while beat F stays ~0.86–0.89.

> **Implication for film-grip:** librosa gives us solid **beat + tempo + onsets** but **no downbeat
> detector** (and the SOTA downbeat libs are all license/install-disqualified). So our reader treats
> **beats as a reliable advisory grid (easy genres) and `downbeats[]` as best-effort/derived** — see
> Honesty Tiers. We can still offer a *meter-assumed* downbeat (every Nth beat from a chosen phase),
> clearly labeled as an assumption rather than detection.

### 5. Musical key / scale detection

**Krumhansl-Schmuckler (K-S)**: build a 12-bin **chroma / pitch-class profile** (sum a chromagram
over time), correlate it against **24 templates** (one major + one minor template, each rotated
through all 12 transpositions), pick the max. Pearson correlation makes it loudness-invariant; **flat
(zero-variance) chroma is mathematically undefined** → the correct response for atonal/percussive
input is to **abstain**, not emit a confident wrong key.

- **Chroma** is computed via `librosa.feature.chroma_cqt` (preferred — the Constant-Q transform's
  log-spaced bins align with semitones, so the octave-fold is clean; STFT over-resolves highs and
  under-resolves lows). Tuning estimation and HPSS (drop drum transients) materially improve it.
- **Profiles:** Krumhansl-Kessler (probe-tone), Temperley, and **Albrecht-Shanahan (2013)** — the
  modern best, which uses **Euclidean distance (min-wins), not Pearson**, and is markedly better in
  minor keys (Chopin Op.28: A-S 83.3% vs K-K 45.8%). Kania et al. 2024: K-K 79.2 / Temperley 85.4 /
  **A-S 91.7** overall.
- **librosa has NO built-in key estimator** (GitHub issue #366, declined) — you roll your own ~20-line
  KS/A-S correlation on a mean chromagram. This is the standard community approach.
- **Accuracy & evaluation:** the **MIREX weighted score** credits correct=1.0, perfect-fifth=0.5,
  relative=0.3, parallel=0.2, else 0 (use `mir_eval.key` with `allow_descending_fifths=True` to match
  MIREX). Realistic in-genre ceilings: **~67–77% raw / ~70–84 weighted** (EDM/GiantSteps ~68/74,
  pop/Billboard ~77/84). Unreliable for **modulating** (one global label discards time), **atonal**
  (ill-defined), and **percussive** (broadband energy floods all 12 bins) music. The common error is
  the wrong *mode* (parallel) or the *relative* key — which is why the weighted metric exists.

> **Implication:** `key` is a **low-confidence advisory** field only. We return it with a confidence
> (the winning correlation / margin to runner-up) and abstain (`key=None`) on flat/low-confidence
> chroma. It is never used to drive an edit op — only to inform the human/agent.

---

## Library comparison

| Library | Capabilities (beat / tempo / onset / downbeat / key) | Deps / weight | License | Offline? | Accuracy | Speed | Maintenance (2025-26) |
|---|---|---|---|---|---|---|---|
| **librosa** ✅ | beat_track, tempo, onset_detect, PLP; chroma (key = ~20-line DIY). **No downbeat, no built-in key.** | numpy+scipy+**scikit-learn+numba(LLVM)**+soundfile. **No torch/TF.** | **ISC** (permissive) ✅ | **Yes, 100%** (DSP; no model weights) ✅ | Good baseline, not SOTA; small ~20-60 ms beat-late bias | Faster than realtime offline (numba JIT cold-start) | **Active.** v0.11.0 (2025-03); Py 3.8-3.13; **numpy 2.x** ✅ |
| aubio | onset, tempo/beat, pitch, notes. No downbeat/key. | C lib + numpy bindings; **no wheels → compiles**; optional ffmpeg | **GPL-3.0** ❌ | Yes (DSP) | decent onset/pitch; mediocre beat | fast (C) | **Dead.** last 0.4.9 (2019); **fails build on Py 3.12** (`imp` gone); no numpy 2 ❌ |
| madmom | **SOTA beat + downbeat** (RNN+DBN); onset, tempo, key | numpy+scipy+**Cython**+mido. No torch/TF. | code BSD-2 **but MODELS CC BY-NC-SA 4.0 (non-commercial)** ❌ | Yes (models bundled) | **best open-source** | slower (RNN+DBN) | **Broken.** only 0.16.1 (2018); breaks on Py ≥3.10 (`collections.MutableSequence`, `np.float`); git capped <3.10 ❌ |
| essentia | RhythmExtractor2013, BeatTrackerMultiFeature; onset; **KeyExtractor** | large C++; **sparse wheels** (cp314 only now; no 3.11-3.13; no Windows bindings) | **AGPL-3.0** ❌ | DSP offline; TF models download | strong rhythm + key | fast (C++) | alive but irregular; **no wheels for our Py targets** ❌ |
| BeatNet | **joint beat+downbeat+tempo+meter**, online+offline | pulls **torch + librosa + madmom + numba==0.54.1** | CC-BY-4.0 ⚠️ | weights bundled; **PF mode non-deterministic** (unseeded RNG) ⚠️ | strong, esp. online | ~3× realtime CPU | **frozen** (v1.1.3, 2023); **madmom dep ⇒ won't install on Py ≥3.11** ❌ |

Briefly: **Spotify pedalboard** is a VST/effects host (GPL), not a detector. **Spotify basic-pitch**
(Apache-2.0) is audio→MIDI note transcription — no tempo/beat/key, and pulls TensorFlow; wrong tool.
There is **no maintained tiny pure-numpy SOTA beat tracker**; the lightest credible permissive option
*is* librosa's Ellis DP tracker. (You *could* vendor a ~50-line numpy Ellis port on `audio_io`'s own
RMS/onset envelope to drop even librosa — viable as a future "zero-extra" fallback, noted below.)

### The librosa-vs-madmom tradeoff (explicit)

madmom wins on **accuracy** and is the only one of the two that does **downbeats** at all. It loses on
**everything film-grip actually requires**: its models are **non-commercial-licensed** (a legal
blocker for a distributed/commercial tool absent written permission from JKU) and its package **won't
install on the project's CI Python (3.11/3.13)** — its only PyPI release is from 2018. For a tool that
prizes deterministic, offline, permissive, lightweight, modern-Python behavior over research-grade
accuracy, **the tradeoff resolves decisively to librosa.** If best-in-class downbeats ever become a
hard product requirement, the path is a sandboxed pinned-Python-3.9 madmom env **plus** a JKU
commercial license — a deliberate future project, not a default dependency.

---

## Pitch / retime — what an honest tool must warn about

**The "chipmunk effect" is the core honesty issue.** The naive way to change a clip's duration is to
**resample** (play the same samples faster/slower), which **couples pitch to speed 1:1**:
`new_frequency = playback_rate × original_frequency`. Slower = deeper, faster = chipmunk.

Three distinct operations, and the identity that links them:

| Operation | Duration | Pitch |
|---|---|---|
| **Time-stretch** (TSM) | changes | **unchanged** |
| **Pitch-shift** | **unchanged** | changes |
| **Resample** (naive / chipmunk) | changes | changes (coupled) |

**Pitch-shift = time-stretch + resample.** Quality time-stretch is either **phase-vocoder**
(frequency-domain; artifacts called *phasiness* + transient smearing) or **WSOLA/OLA** (time-domain;
great on speech/monophonic, degrades on dense polyphony/transients). **Formant preservation** is a
*separate* concern from pitch preservation: naive pitch-shift drags the vocal-tract resonances
(formants) too, producing chipmunk/monster voices even when you "preserve pitch"; fixing it needs a
source-filter method (PSOLA, élastique, WORLD vocoder).

**Pro NLEs disagree on the default**, so users' expectations differ by tool — film-grip must state its
behavior rather than assume:

| NLE | Default on speed change | Setting |
|---|---|---|
| DaVinci Resolve | **OS-dependent** (macOS preserves; Win/Linux shift) | "Pitch Correction" |
| Premiere Pro | **shifts** (chipmunk) | "Maintain Audio Pitch" (off) |
| Final Cut Pro | **preserves** | "Preserve Pitch" (on) |

(Resolve also **mutes audio entirely during variable-speed ramps**; the Retime Process /
interpolation settings in all three are **video-only**.)

ffmpeg specifics film-grip already has the binary for: **`atempo`** changes tempo **without** pitch
(range `[0.5, 100.0]` in current builds — the famous `0.5–2.0` is outdated; chainable);
**`asetrate`** changes **speed AND pitch** (the resample/chipmunk method); **`rubberband`** is
high-quality independent time+pitch **with `formant=preserved`**, but **requires
`--enable-librubberband`** and is **often missing from stock builds** (detect at runtime).
`librosa.effects.time_stretch` / `pitch_shift` exist but librosa's vocoder self-describes as
"pedagogical … likely to produce many audible artifacts."

**What film-grip should do (honesty surface).** The existing `Retime` op maps to an OTIO
`LinearTimeWarp` (video). The honest gaps to close:

1. **`analyze_music` / a retime preflight emits a `warnings[]` entry** whenever a `retime` targets a
   clip that has **linked/under audio**: "speed_percent=200 will shift this clip's audio up ~1 octave
   (chipmunk) unless pitch is corrected; film-grip's OTIO retime affects video — the editor's own
   audio-pitch setting (Resolve Pitch Correction / Premiere Maintain Audio Pitch / FCP Preserve
   Pitch) governs the result, and defaults differ per NLE." This mirrors `align.is_retimed()` already
   *refusing* word-alignment on retimed clips — same honesty, applied to audio pitch.
2. **Surface the artifact cliff:** stretches beyond ~±10–20% degrade audibly regardless of algorithm;
   say so.
3. **Surface formant ≠ pitch:** preserving pitch still munchkinizes voices without formant correction.
4. This stays **advisory** (the pitch behavior is the editor's setting, not something Resolve's
   scripting API lets film-grip toggle) — declared, never silently assumed. It belongs in the same
   bucket as `AUDIO_PROPS_UNSUPPORTED` / `COLOR_ADVISORY`.

---

## J-cuts / L-cuts — detecting candidates

A **J-cut** = audio of the *next* shot starts **before** its picture (sound leads). An **L-cut** =
audio of the *current* shot continues **after** its picture cuts away (sound lags). Both are
split-edits where the **audio edit point is offset from the video edit point**.

film-grip already has the two inputs needed: **word-level timeline frames** (`align.AlignedWord`) and
a **per-hop RMS energy envelope** (`audio_io.rms_envelope`). Candidate detection at a video cut
between clip A (out) and clip B (in) at frame `f`:

- **Transcript-driven (preferred, deterministic):** if a spoken **word/phrase straddles** `f` — i.e.
  a word from B's transcript *starts before* `f`, or a word from A *ends after* `f` — cutting picture
  at `f` would chop a word. The fix is a split-edit: hold A's **audio** to the word boundary (L-cut)
  or bring B's **audio** in at the word start (J-cut). The exact offset is
  `word_boundary_frame − f`, already in timeline frames. This reuses `speech.py`'s "never cut inside a
  word" rule and the `ASR_DRIFT_PAD_S` pad.
- **Energy-driven (fallback when no speech):** find where B's audio energy **rises above a noise
  floor** shortly *before* `f` (→ J-cut candidate) or A's energy **stays above floor** shortly
  *after* `f` (→ L-cut candidate), from the RMS envelope. Less precise (no semantic boundary), so
  it's offered as a lower-confidence suggestion.

These are **suggestions surfaced by a reader**, not auto-applied: a true split-edit needs independent
audio/video edit points, which on the interchange path requires the audio on its own track. So the
honest output is *candidate {type: J|L, video_frame, suggested_audio_frame, offset_frames, basis:
"word"|"energy"}* the planner/human acts on (often by `move`/`trim` of a split-off audio clip), rather
than a one-click op that pretends every NLE round-trips split-edits.

---

## Proposed film-grip capabilities

### A. Reader — `filmgrip/perception/music.py` :: `analyze_music`

Same structure as `scopes.py` / `motion.py`: **pure analysis decoupled from IO**, librosa guarded
lazily behind the new extra, failures degrade to honest per-clip errors, and times are projected to
**timeline frames per clip** by reusing `align.py`'s media↔timeline math and its honesty rules
(offline media → error; **retimed clip → error/skip**, because a time-warp breaks the linear
media↔timeline mapping exactly as it does for word alignment).

```python
# filmgrip/perception/music.py
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_SNAP_UNIT = "beat"          # "beat" | "downbeat"
KEY_MIN_CONFIDENCE = 0.60           # below this, abstain (key=None)

@dataclass(frozen=True)
class MusicAnalysis:
    """Beat/tempo/onset/key for one media file, in MEDIA seconds (timeline projection added later)."""
    media_path: str
    tempo_bpm: float
    beat_times_s: list[float]
    downbeat_times_s: list[float]     # best-effort / meter-assumed; see honesty tiers
    onset_times_s: list[float]
    key: Optional[str]                # e.g. "A:min"; None when abstaining
    key_confidence: float
    beats_per_bar: int                # the meter ASSUMED for downbeats (e.g. 4); 0 if unknown
    duration_s: float

@dataclass(frozen=True)
class ClipMusic:
    """`analyze_music`'s per-clip payload — times mapped to TIMELINE FRAMES."""
    clip_id: str
    track: str
    tempo_bpm: float
    beats: list[int]                  # timeline frames
    downbeats: list[int]              # timeline frames (advisory)
    onsets: list[int]                 # timeline frames
    key: Optional[str]
    key_confidence: float

# --- pure DSP (unit-testable on synthetic signals; no ffmpeg, no librosa needed for the helpers) ---
def estimate_key(chroma_mean: Any) -> tuple[Optional[str], float]:
    """Krumhansl-Schmuckler / Albrecht-Shanahan correlation on a 12-bin mean chroma vector.
    Returns (key, confidence). Abstains (None) on flat/zero-variance chroma."""

def beats_to_frames(times_s: list[float], clip, rate: float) -> list[int]:
    """Project MEDIA-second event times into this clip's TIMELINE frames, keeping only events that
    fall inside the clip's source window (mirrors align.align_clip_words)."""

# --- librosa-backed analysis (the only librosa entry point) ---
def analyze_media_music(media_path: str, *, start_s: float = 0.0,
                        dur_s: Optional[float] = None) -> MusicAnalysis:
    """Decode via audio_io.decode_pcm, run librosa beat_track/onset_detect/tempo + chroma key.
    Raises PerceptionUnavailable when librosa (the [music] extra) is missing — with the install fix."""

# --- the reader the MCP tool / packs call (mirrors align.transcript_for_clips) ---
def analyze_music_clips(ir, ids: list[str], *,
                        analyzer=analyze_media_music) -> dict:
    """Per-clip beats/downbeats/onsets/tempo/key in TIMELINE frames + honest `errors`.
    Caches analysis per distinct media file (like transcript_for_clips); injectable `analyzer`
    for tests. Skips retimed clips and offline media into `errors`, never guesses."""
    return {"r": ..., "frames": "timeline", "clips": [...], "errors": [...], "warnings": [...]}
```

Notes that make it fit:
- **Build on `audio_io.decode_pcm`** — already the project's float-PCM primitive, already arg-list-safe.
- **Reuse `align.media_path_of` / `align.is_retimed` / `align._source_rate`** verbatim for the
  projection + honesty (don't re-implement). `beats_to_frames` is `align_clip_words` with a single
  time instead of a (start,end) word.
- **`tempo_bpm`** from `librosa.feature.tempo` (or the `tempo` returned by `beat_track`); pass
  `start_bpm`/`std_bpm` to expose the prior.
- **`downbeats`**: librosa has no downbeat detector. Offer two honest modes — (i) **derived**: assume
  a meter (`beats_per_bar`, default 4) and a downbeat phase (first strong-onset beat), labeling the
  result *assumed* not *detected*; (ii) leave empty if meter is unknown. Never present assumed
  downbeats as detected.
- **`key`** via `estimate_key` on a mean `chroma_cqt`; abstain below `KEY_MIN_CONFIDENCE`.

**Dependency wiring** (`pyproject.toml`, mirroring `transcribe`/`color`):
```toml
# Music perception (beat/tempo/onset/key) for rhythm-aware editing. librosa is ISC-licensed,
# pure-DSP (no torch/tensorflow), fully offline, and supports numpy 2.x / py3.13.
music = ["librosa>=0.11.0"]
```
Add `librosa>=0.11.0` to `all` and `dev`. **One caveat to flag in review:** `audio_io.py` currently
guards numpy behind the `color` extra's message ("pip install 'film-grip[color]'"). librosa already
brings numpy, so installing `[music]` satisfies it transitively — but the *error string* should be
generalized (or `music` made to depend on numpy explicitly) so a `[music]`-only user gets a coherent
message. Pin `>=0.11.0` to guarantee numpy-2 / py3.13 support (numba ≥0.61 is the real numpy-2 gate).

### B. EditPlan ops + packs

**New op `snap_cuts`** (fits the typed-op + validation model; compiles to existing primitives, so it
is *always applicable* — the pack honesty rule in `packs/builtins.py`):

```python
class SnapCuts(_Op):
    """Snap cut points to the nearest musical beat (or downbeat) within a tolerance window.

    Deterministic, music-synced editing. Resolves to the beat grid from analyze_music, then nudges
    each targeted cut to the closest beat/downbeat that is within `tolerance_frames`; cuts with no
    beat in range are left untouched (and reported), never force-moved. Compiles to trim/move/
    cut_range — no new adapter capability required."""
    op: Literal["snap_cuts"] = "snap_cuts"
    clip_ids: list[str] = Field(min_length=1)        # whose edges to snap (or which clips to cut on-grid)
    unit: Literal["beat", "downbeat"] = "beat"
    tolerance_frames: int = Field(default=6, ge=0, le=240)  # only snap if a grid point is this close
    edges: Literal["in", "out", "both"] = "both"
```

Register it in `AnyOp`; `all_op_names()` then auto-includes it in the capability table + honesty-gate
tests (the schema is the single source of truth, per `editplan.py`).

**New deterministic pack `beat-snap`** (mirrors `silence-cut`: zero-LLM, IR-aware, raises a real
`PackError` if no music analysis is possible — never silently no-ops):

```python
def _beat_snap(ir, ids, params) -> list:
    from ..perception.music import analyze_music_clips
    from . import PackError
    unit = params.get("unit", "beat")
    tol  = _frames(params.get("tolerance", "6"), ir.rate)
    analysis = analyze_music_clips(ir, [c.id for c in selected_clips(ir, ids)])
    if analysis["errors"]:
        raise PackError("beat-snap could not analyze every selected clip:\n  "
                        + "\n  ".join(analysis["errors"]))
    # turn the per-clip beat grid + existing edges into trim/move ops (descending, like _silence_cut)
    return _snap_ops(ir, ids, analysis, unit=unit, tol=tol)

register(Pack("beat-snap",
              "Snap the selected clips' cut points to the nearest musical beat/downbeat — "
              "deterministic, transcript-free, no LLM call.",
              compile=_beat_snap, params={"unit": "beat", "tolerance": "6"},
              requires=("the [music] extra (pip install 'film-grip[music]')",
                        "source media files on disk")))
```

**Prompt packs** (planner reads the beat grid, like `neutral-balance` reads scopes):
- `music-montage` — "Using the beat grid from analyze_music, cut the selected B-roll so each clip
  change lands on a **downbeat**; keep clips roughly {clip_beats} beats long." Emits
  `cut_range`/`trim`/`move`.
- `cut-on-the-drop` — "Find the strongest onset/energy rise (the drop) and place the hero clip's
  start there."

A second op, **`add_beat_markers`** (compiles to the existing `add_marker` — always applicable), is a
cheap, high-trust first deliverable: drop markers on every beat/downbeat so the human can edit to a
visible grid even when auto-snapping is undesired. (`marker-pass` already proves this pattern.)

### C. Verify strategy + fixtures

Mirror the existing two-layer honesty (`verify.py`, `scopes` synthetic-frame tests): a **pure-DSP unit
layer** (no ffmpeg, no librosa) plus an **integration layer** on a synthesized click track.

**Pure unit tests (always run — like `analyze_rgb`/`rms_envelope` tests):**
- Synthesize a **click envelope in numpy** at a known period (e.g. impulses every `sr*60/BPM`
  samples) and assert the project's peak-picker / `beats_to_frames` recover the impulse indices
  exactly. No external deps → runs in CI on the base install.
- `estimate_key`: feed a synthetic chroma vector that is a known rotated profile → assert the
  correct key; feed a **flat** vector → assert it **abstains** (`None`).
- `beats_to_frames`: given a clip with a known `source_start`, `start`, and `rate`, assert a
  media-second event maps to the **hand-computed** timeline frame, and that events outside the clip's
  source window are dropped (the `align_clip_words` contract).

**Integration test (skips cleanly when librosa/ffmpeg absent, like the `live`/transcribe tests):**
- **Synthesize a click track at a known BPM with ffmpeg — no new dependency.** A metronome via
  `aevalsrc` gated to a click each beat, e.g. for 120 BPM (0.5 s period):
  ```bash
  ffmpeg -y -f lavfi -i \
    "aevalsrc=0.6*sin(2*PI*1000*t)*lt(mod(t\,0.5)\,0.02):s=44100:d=8" \
    -c:a pcm_s16le tests/fixtures/click_120bpm.wav
  ```
  (Or `sine` + a tremolo gate; either yields sharp 1 kHz ticks at exactly 120/min.) Generate a couple
  of BPMs (90, 120, 128) as session-scoped fixtures.
- Assert `analyze_media_music(click).tempo_bpm` is within **±2 BPM**, accepting **½×/2×** (octave
  tolerance — the dominant real error, per the Acc2 discussion). Optionally assert the BPM is an
  *exact* match for the four-on-the-floor click where librosa is reliable, but keep octave acceptance
  for honesty.
- Assert each detected beat lands within **N ms** of the known grid (start with **N = 50 ms**;
  librosa's documented ~20–60 ms late-bias means tighter than ~40 ms will flake — encode the bias as a
  tolerance, don't fight it). Assert beat **count** ≈ duration·BPM/60 within ±1.
- **Frame-mapping assertion:** wrap the click in a one-clip fixture timeline with a known
  `source_start` and rate, run `analyze_music_clips`, and assert the first returned `beats` frame
  equals the hand-computed `clip.start + round((beat_s − clip_in_s) * rate)` — proving the
  media→timeline projection, the piece most likely to regress.
- **Honesty assertions** (the project's signature): a **retimed** clip and an **offline-media** clip
  come back in `errors`, never with guessed beats (assert parity with `transcript_for_clips`'s
  behavior); a **percussion-free / silent** clip yields a low-confidence/abstained `key` and an empty
  or clearly-low-confidence beat set rather than a fabricated grid.

---

## Honesty tiers

The signature film-grip move — declare capability honestly, never claim an edit the tool can't deliver:

- **Reliable (advisory beat grid, safe to drive deterministic ops):** **tempo + beat times on
  steady, percussive material** (EDM, pop/rock with drums, dance) — beat F ≈ 0.9+, tempo Acc1 ≈
  70–80% / Acc2 ≈ 92–95%. `snap_cuts` to **beats** with a tolerance window, `add_beat_markers`, and
  beat-driven montage packs are trustworthy here.
- **Best-effort / labeled-as-assumed:** **downbeats** (librosa can't detect them; we *derive* them
  from an assumed meter + phase) and **tempo on mixed/varying-tempo material** (single-global-tempo
  assumption; ±10% drift only — offer `plp`/per-frame tempo as the escape hatch). Surface these as
  suggestions with the assumption stated, and prefer snapping to *beats* over *downbeats* unless the
  user confirms the meter.
- **Unreliable / advisory-only (never drives an op):** **key** (low-confidence; abstain on
  flat/atonal/percussive; ~67–84% even in-genre), **beat/tempo on classical/rubato/ambient/a-cappella**
  (beat F ≈ 0.4–0.6 — *treat as possibly undefined, not merely low-accuracy*; confidence-gate or
  refuse rather than emit a wrong grid), and **energy-based J/L-cut** candidates (less precise than
  the transcript-based ones).
- **Declared limitations (the `AUDIO_PROPS_UNSUPPORTED` / `COLOR_ADVISORY` bucket):** film-grip's
  `retime` warps **video**; **audio pitch behavior on retime is the editor's own setting** (and its
  default differs per NLE) — film-grip warns about the chipmunk/formant effect but does not silently
  fix it. True **split-edits (J/L)** need audio on a separate track and don't round-trip uniformly
  through every interchange format — surfaced as a candidate + manual/explicit step, not a fake
  one-click op.

---

## Recommended dependency choice

**Add `librosa>=0.11.0` as a new optional extra `music`** (and to `all` + `dev`). It is the only
option that is **permissive (ISC), fully offline (no model download), torch/tensorflow-free, and
maintained for Python 3.13 + numpy 2.x**, and it slots cleanly next to the existing `transcribe`
(faster-whisper) and `color` (numpy) extras. Accept the one real cost — librosa pulls **numba/llvmlite
(LLVM)**, so `[music]` is heavier than `[color]` (numpy-only), but far lighter than anything pulling
torch/madmom and with **zero license or offline concerns**. Implement **key detection in-house**
(Krumhansl-Schmuckler / Albrecht-Shanahan on librosa chroma — librosa has no key estimator by
design). Do **not** add madmom/aubio/essentia/BeatNet: license (NC/GPL/AGPL) and/or modern-Python
install failures disqualify all four. **Future escape hatch:** a ~50-line pure-numpy Ellis DP beat
tracker on `audio_io`'s existing onset/RMS envelope would let beat/tempo work with **no extra at
all** — worth keeping in mind if the numba weight ever becomes a problem, with librosa remaining the
higher-quality default.

---

## Sources

**Onset detection**
- Bello et al. (2005), "A Tutorial on Onset Detection in Music Signals," IEEE TASLP — https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/reading/Bello_2005.pdf
- Böck & Widmer (2013), "Maximum Filter Vibrato Suppression for Onset Detection" (SuperFlux), DAFx-13 — https://www.dafx.de/paper-archive/2013/papers/09.dafx2013_submission_12.pdf
- Böck, Krebs, Schedl (2012), "Evaluating the Online Capabilities of Onset Detection Methods," ISMIR — https://archives.ismir.net/ismir2012/paper/000049.pdf
- Dixon (2006), "Onset Detection Revisited," DAFx-06 — https://www.dafx.de/paper-archive/2006/papers/p_133.pdf
- MIREX Audio Onset Detection results 2005 — https://music-ir.org/mirex/wiki/2005:Audio_Onset_Detection_Results
- Onset SOTA / per-class (singing-voice difficulty) — https://musicinformationretrieval.wordpress.com/2017/02/03/state-of-the-art-for-audio-onset-detection-week-3/
- librosa onset module — https://librosa.org/doc/0.11.0/_modules/librosa/onset.html
- mir_eval onset (±50 ms default) — https://raw.githubusercontent.com/mir-evaluation/mir_eval/main/mir_eval/onset.py

**Tempo / BPM**
- Scheirer (1998), "Tempo and beat analysis of acoustic musical signals," JASA — https://www.ee.columbia.edu/~dpwe/papers/Schei98-beats.pdf
- Klapuri, Eronen, Astola (2006), "Analysis of the meter of acoustic musical signals," IEEE TASLP — https://www.iro.umontreal.ca/~pift6080/H09/documents/papers/klapuri_meter.pdf
- Grosche, Müller, Kurth (2010), "Cyclic Tempogram," ICASSP — https://resources.mpi-inf.mpg.de/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf
- Schreiber & Müller (2018), "A Single-Step Approach to Musical Tempo Estimation Using a CNN," ISMIR — https://www.tagtraum.com/download/2018_schreiber_tempo_cnn.pdf
- Schreiber, Urbano & Müller (2020), "Music Tempo Estimation: Are We Done Yet?," TISMIR — https://transactions.ismir.net/articles/10.5334/tismir.43
- FMP notebooks — Autocorrelation Tempogram https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramAutocorrelation.html ; Fourier Tempogram https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramFourier.html
- librosa.feature.tempo — https://librosa.org/doc/main/generated/librosa.feature.tempo.html
- MIREX Audio Tempo Extraction (P-score, 8%) — https://www.music-ir.org/mirex/wiki/2006:Audio_Tempo_Extraction

**Beat tracking**
- Ellis (2007), "Beat Tracking by Dynamic Programming," JNMR — https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
- librosa.beat.beat_track — https://librosa.org/doc/main/generated/librosa.beat.beat_track.html ; source https://librosa.org/doc/main/_modules/librosa/beat.html
- mir_eval beat (metric constants) — https://raw.githubusercontent.com/mir-evaluation/mir_eval/main/mir_eval/beat.py
- Davies, Degara & Plumbley (2009), "Evaluation Methods for Musical Audio Beat Tracking" (CMLt/AMLt) — https://www.researchgate.net/publication/228724188_Evaluation_Methods_for_Musical_Audio_Beat_Tracking_Algorithms
- Holzapfel et al. (2012), "Selective Sampling for Beat Tracking Evaluation" (SMC) — https://repositorio.inesctec.pt/server/api/core/bitstreams/4c744b20-9085-4aa7-9739-d05e19033d84/content
- Davies & Böck (2019), TCN beat tracking, EUSIPCO — https://www.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570533824.pdf

**Downbeat / meter**
- Böck, Krebs & Widmer (2016), "Joint Beat and Downbeat Tracking with RNNs," ISMIR — https://archives.ismir.net/ismir2016/paper/000186.pdf
- Krebs, Böck & Widmer (2015), "An Efficient State-Space Model for Joint Tempo and Meter Tracking," ISMIR — https://www.cp.jku.at/research/papers/Krebs_etal_ISMIR_2015.pdf
- Foscarin, Schlüter & Widmer (2024), "Beat This! Accurate Beat Tracking Without DBN Postprocessing," ISMIR — https://arxiv.org/abs/2407.21658 ; code https://github.com/CPJKU/beat_this
- Zhao, Xia & Wang (2022), "Beat Transformer," ISMIR — https://archives.ismir.net/ismir2022/paper/000019.pdf
- madmom downbeat source — https://raw.githubusercontent.com/CPJKU/madmom/main/madmom/features/downbeats.py

**Key / chroma**
- Krumhansl-Schmuckler reference — http://rnhart.net/articles/key-finding/
- music21 key profiles (K-K, Temperley, A-S) — https://raw.githubusercontent.com/cuthbertLab/music21/master/music21/analysis/discrete.py
- Kania et al. (2024), profile comparison — https://journals.pan.pl/Content/132999/PDF/aoa.2024.148817.pdf
- C. White (2018), profile evaluation — https://mtosmt.org/issues/mto.18.24.2/mto.18.24.2.white.html
- Korzeniowski & Widmer (2017), key CNN + weighted score — https://arxiv.org/pdf/1706.02921
- librosa chroma_cqt — https://librosa.org/doc/main/generated/librosa.feature.chroma_cqt.html
- librosa has no key detector (issue #366) — https://github.com/librosa/librosa/issues/366
- mir_eval key (allow_descending_fifths) — https://raw.githubusercontent.com/mir-evaluation/mir_eval/main/mir_eval/key.py
- MIREX Audio Key Detection — https://music-ir.org/mirex/wiki/2025:Audio_Key_Detection
- Constant-Q transform (Brown 1991) — https://en.wikipedia.org/wiki/Constant-Q_transform

**Libraries**
- librosa — https://pypi.org/project/librosa/ ; LICENSE (ISC) https://github.com/librosa/librosa/blob/main/LICENSE.md ; changelog (v0.11.0 / numpy 2) https://librosa.org/doc/latest/changelog.html
- numba numpy-2 gating — https://numba.readthedocs.io/en/stable/release/0.61.0-notes.html
- aubio — https://pypi.org/pypi/aubio/json ; GPLv3 https://raw.githubusercontent.com/aubio/aubio/master/COPYING ; Py3.12 build failure https://github.com/aubio/aubio/issues/394
- madmom — LICENSE (BSD code + CC BY-NC-SA models) https://github.com/CPJKU/madmom/blob/main/LICENSE ; only 0.16.1/2018 https://pypi.org/pypi/madmom/json ; Py3.10 break https://github.com/CPJKU/madmom/issues/502 ; "<Python 3.10" https://github.com/CPJKU/beat_this/issues/9
- essentia — AGPL + wheels https://pypi.org/pypi/essentia/json ; licensing https://essentia.upf.edu/licensing_information.html ; RhythmExtractor2013 https://essentia.upf.edu/reference/std_RhythmExtractor2013.html
- BeatNet — deps (torch+librosa+madmom) https://pypi.org/pypi/BeatNet/1.1.3/json ; LICENSE (CC-BY-4.0) https://raw.githubusercontent.com/mjhydri/BeatNet/master/LICENSE ; paper https://archives.ismir.net/ismir2021/paper/000033.pdf
- basic-pitch (audio→MIDI, Apache-2.0) — https://pypi.org/project/basic-pitch/
- pedalboard (VST host, GPLv3) — https://github.com/spotify/pedalboard

**Retime / pitch / formants / NLEs / ffmpeg**
- Audio time-stretching & pitch scaling — https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
- Phase vocoder — https://en.wikipedia.org/wiki/Phase_vocoder ; PSOLA — https://en.wikipedia.org/wiki/PSOLA ; WSOLA — https://www.isca-archive.org/eurospeech_1993/roelands93_eurospeech.html
- Formant / source-filter — https://en.wikipedia.org/wiki/Formant ; https://en.wikipedia.org/wiki/Source%E2%80%93filter_model
- WORLD vocoder — https://github.com/mmorise/World
- DaVinci Resolve retime (Pitch Correction / ramp mute) — https://www.steakunderwater.com/VFXPedia/__man/Resolve18-6/DaVinciResolve18_Manual_files/part1263.htm
- Premiere "Maintain Audio Pitch" — https://helpx.adobe.com/premiere/desktop/edit-projects/change-clip-speed/change-clip-speed-using-the-speedduration-option.html
- Final Cut Pro "Preserve Pitch" — https://support.apple.com/guide/final-cut-pro/preserve-pitch-in-retimed-clips-verb6aca45d/mac
- ffmpeg filters (atempo / asetrate / rubberband) — https://ffmpeg.org/ffmpeg-filters.html ; atempo current range https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/atempo.html ; rubberband (needs --enable-librubberband) https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/rubberband.html
- librosa.effects.time_stretch / pitch_shift / phase_vocoder ("pedagogical") — https://librosa.org/doc/latest/generated/librosa.effects.time_stretch.html ; https://librosa.org/doc/latest/generated/librosa.effects.pitch_shift.html ; https://librosa.org/doc/latest/generated/librosa.phase_vocoder.html
