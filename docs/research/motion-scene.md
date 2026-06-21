# Motion & Scene Analysis for Video Editing — film-grip research

Deep research for film-grip's perception layer: adding **deterministic motion & scene perception**
(shot boundaries, motion magnitude, shake) so the agent can *see* footage structure and propose
honest cut and retime edits. Mirrors the existing readers in
`filmgrip/perception/{speech,transcribe,scopes,frames,verify,align}.py`.

---

## Summary

- **Recommended deps:** add a single optional extra `motion = ["opencv-python-headless>=4.8", "numpy>=1.24"]`.
  `opencv-python-headless` is BSD-3, offline, no GUI libs, and gives us Farneback dense optical
  flow + HSV/Canny primitives — enough to reimplement PySceneDetect's `ContentDetector` math in
  ~40 lines of pure numpy/cv2 without taking PySceneDetect itself as a runtime dependency. ffmpeg
  (already required) provides a **zero-new-dep fallback** path via `scdet`/`select='gt(scene,...)'`
  for shot detection and `vidstabdetect` for shake.
- **Top capabilities to implement:**
  1. `detect_shots(ir, clip_id) -> [{start,end,score,kind}]` in **timeline frames** — a deterministic
     content-aware reader (HSV-delta `ContentDetector` math), feeding a new `shot-split` pack that
     emits `split` ops at boundaries.
  2. `analyze_motion(ir, clip_id, window_s) -> per-window {motion, direction, peak/lull}` — Farneback
     magnitude reduced to a windowed score, feeding **motion-aware `retime` suggestions** (slow the
     peak, speed the lull) built on the existing `retime` op.
  3. `detect_shake(ir, clip_id) -> {shaky, score, advice}` — **advisory only** inside any NLE;
     optional opt-in ffmpeg `vidstab` bake as an explicitly destructive, separate path.
- **Verify approach:** synthesize fixtures with ffmpeg (no checked-in binaries) — concat two
  solid-color `lavfi` segments → assert the boundary lands within N frames; a static `color` clip
  vs a `testsrc`/panning clip → assert `motion(static) < motion(panning)`. Pure-numpy core
  functions (`shot_score`, `motion_score`) are unit-tested on synthetic arrays, exactly like
  `analyze_rgb`.
- **Honesty tiers:** detection (shots/motion) = **reliable** evidence; in-NLE stabilization =
  **not scriptable → advisory only**; ffmpeg-baked stabilization = **destructive, opt-in**, never
  silent. Retimed clips are refused by every reader (same guard as `align.is_retimed`).

---

## 1. Shot / scene boundary detection

### The families

| Approach | Metric | Strength | Weakness |
|---|---|---|---|
| **Threshold** (fixed) | absolute frame brightness vs a fixed level | catches fades to/from black | misses cuts between two bright scenes |
| **Content-aware** (delta) | frame-to-frame difference in HSV (+ edges) | catches hard cuts regardless of brightness | a fast pan/whip can exceed threshold → false positive |
| **Adaptive** (rolling) | content delta vs a *rolling average* of neighbours | suppresses motion-induced false positives | adds latency (a window), can miss two fast cuts in a row |

PySceneDetect (`scenedetect`, BSD-3) is the reference implementation and its math is simple enough
to reimplement directly on cv2 + numpy.

### ContentDetector math (the one we reimplement)

For each adjacent frame pair, convert BGR→HSV and split into `hue, sat, lum` 8-bit planes
([source](https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/detectors/content_detector.py)):

```
hue, sat, lum = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
```

Each component delta is a **mean absolute pixel distance** between this frame's plane and the
previous frame's plane:

```
_mean_pixel_distance(a, b) = sum(|a.astype(int32) - b.astype(int32)|) / (H*W)
```

`delta_edges` (optional, weight 0.0 by default) runs Canny on the luma plane and dilates it, then
takes the same mean-abs distance of the edge maps — an **edge-change ratio** that boosts sensitivity
to sharp structural changes:

```
edges = cv2.dilate(cv2.Canny(lum, low, high), kernel)
```

The **frame score** is a normalized weighted average:

```
frame_score = sum(component * weight) / sum(|weight|)
              over (delta_hue, delta_sat, delta_lum, delta_edges)
```

A cut is registered when `frame_score >= threshold`. **Default threshold = 27.0**, default
`min_scene_len = 15` frames, default weights `(hue=1, sat=1, lum=1, edges=0)`
([detectors docs](https://www.scenedetect.com/docs/latest/api/detectors.html),
[DeepWiki ContentDetector](https://deepwiki.com/Breakthrough/PySceneDetect/3.1-content-detector)).

`AdaptiveDetector` wraps this: it computes the same per-frame `content_val`, then over a
`window_width` (default 2) rolling window registers a cut where the **ratio** of this frame's score
to the windowed average exceeds `adaptive_threshold` (default **3.0**) *and* the raw score exceeds
`min_content_val` (default **15.0**). The ratio test is what kills pan/handheld false positives —
during a continuous pan every frame has a high score, so no single frame stands out as a *ratio*
spike ([detectors docs](https://www.scenedetect.com/docs/latest/api/detectors.html)).

`HistogramDetector` (Y-channel YUV histogram correlation, default threshold 0.05) and
`HashDetector` (perceptual-hash hamming distance, default 0.395) are alternatives; for film-grip's
needs the HSV content metric is the sweet spot of accuracy vs simplicity.

### ffmpeg path (zero new deps, the fallback)

ffmpeg exposes the same idea two ways
([scdet docs](https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Video/scdet.html),
[GDELT writeup](https://blog.gdeltproject.org/using-ffmpegs-scene-detection-to-generate-a-visual-shot-summary-of-television-news/)):

- **`select='gt(scene,T)'`** — `scene` is a 0..1 likelihood of a new scene; `0.3–0.4` is a typical
  hard-cut threshold. Combine with `showinfo`/`metadata=print` to get `pts_time` of each selected
  frame.
- **`scdet=s=1:t=T`** — sets frame metadata (`lavfi.scd.mafd`, `lavfi.scd.score`,
  `lavfi.scd.time`); `t` (threshold) is a percent-of-max-change, good range **[8.0, 14.0]**,
  default 10.

Parse-friendly invocation (we read stderr/metadata, no re-encode):

```
ffmpeg -hide_banner -i IN -vf "select='gt(scene,0.4)',metadata=print:file=-" -an -f null -
# emits: frame:N pts_time:S  lavfi.scene_score=...   → one line per boundary
```

This is the honest fallback when opencv isn't installed: slightly coarser, but real and offline.

### Accuracy & false positives (fades/dissolves)

- **Hard cuts:** both content and adaptive detectors are reliable; report `score` so the agent (and
  verify) can see confidence.
- **Pans / handheld / whip:** the **adaptive ratio** test is the defense; we default to it.
- **Fades / dissolves:** content detectors *under*-detect gradual transitions (no single big
  frame jump). A dissolve produces a low, sustained bump, not a spike. We **flag** this rather than
  pretend: report a `kind` of `"cut"` (sharp) vs `"gradual"` (a run of moderate scores), and
  document that dissolve boundaries are advisory. ThresholdDetector specifically targets fades to
  black and could be added later, but most editorial cut-detection wants hard cuts.

---

## 2. Optical flow / motion magnitude

### Farneback (the deterministic choice)

`cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)` returns a dense
`(H, W, 2)` flow field; `cv2.cartToPolar(flow[...,0], flow[...,1])` gives per-pixel `magnitude,
angle` ([OpenCV optical flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)).
Reduce to scalars per frame pair:

- **motion magnitude** = `float(np.mean(magnitude))` (mean pixel speed; downscale frames to ~160px
  wide first — same trick `scopes.frame_rgb` uses — so this is cheap, ~8 ms/frame
  ([NVIDIA optical flow blog](https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/))).
- **dominant direction** = circular mean of `angle` weighted by `magnitude` → a compass angle
  (pan-left/right/up/down/zoom is the magnitude-vs-divergence story; for v1 we report the weighted
  mean angle and call it "dominant motion direction").

### RAFT (deep) — explicitly out of scope

RAFT is more accurate on low-texture / motion-blur / hard camera moves, but it needs **PyTorch + a
GPU (≥12 GB) for real-time**, runs ~10 fps on a 1080Ti, and pulls a heavy model
([RAFT paper](https://arxiv.org/pdf/2003.12039),
[NVIDIA blog](https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/)).
That violates film-grip's "light, deterministic, offline, addable as an optional extra" rule.
**Decision: Farneback only.** If a future user wants RAFT, it slots behind the same
`analyze_motion` signature as a `backend=` choice (like `transcribe`'s backends), but it is not a
dependency.

### Using motion: peaks & lulls

Window the per-frame magnitudes into fixed windows (default `window_s = 0.5`, matching
`speech.PACK_GAP_S` granularity), take the mean per window, then:

- **action peak** = window(s) with the top-percentile motion score,
- **lull** = window(s) below a low percentile.

These windows are emitted in **timeline frames** so they line up with everything else and feed the
retime suggester (§4).

---

## 3. Camera shake detection & stabilization (+ honesty)

### Detecting shake (reliable, advisory output)

Shake = **high-frequency, low-net-displacement** motion: lots of frame-to-frame flow that doesn't
accumulate into a pan. Two deterministic reads, both offline:

- **opencv:** from the Farneback flow we already compute, shake score ≈ variance / high-frequency
  energy of the **global mean flow vector** across frames, normalized by net displacement (a true
  pan has high displacement, low jitter; shake has low displacement, high jitter).
- **ffmpeg:** run `vidstabdetect` (pass 1 only) and read the magnitude of per-frame transforms from
  the `.trf` it writes — large oscillating transforms = shaky
  ([vid.stab](https://github.com/georgmartius/vid.stab),
  [Paul Irish vidstab](https://www.paulirish.com/2021/video-stabilization-with-ffmpeg-and-vidstab/)).

Output is a `{shaky: bool, score: float, advice: str}` — **evidence + advice, never an edit**.

### Can we stabilize? The honesty line

- **Inside DaVinci Resolve / Premiere etc.: NOT scriptable.** Resolve's stabilization lives in the
  Inspector/Color page and is **not exposed by the scripting API** — film-grip cannot toggle it
  non-destructively. Same story as the color *scopes* (which is why `scopes.py` synthesizes them).
  So in-NLE stabilization is **advisory only**: the reader can say "clip X is shaky, enable
  Stabilization in the Inspector," and stop there.
- **ffmpeg `vidstab` two-pass: scriptable but DESTRUCTIVE.** `vidstabdetect` (pass 1) →
  `vidstabtransform` (pass 2) produces a *new, re-encoded, cropped/zoomed* file
  ([tk-sls guide](https://tk-sls.de/wp/4018),
  [vidstabtransform docs](http://underpop.online.fr/f/ffmpeg/help/vidstabtransform.htm.gz)). That is
  a generation-loss bake — the exact thing film-grip's non-destructive contract avoids. So it is an
  **opt-in, explicitly destructive** action (writes a new media file the user can relink), never run
  silently and never part of a normal plan.

```
# pass 1 (detect)
ffmpeg -i IN -vf vidstabdetect=shakiness=8:accuracy=15:result=tf.trf -f null -
# pass 2 (transform — DESTRUCTIVE, writes new media)
ffmpeg -i IN -vf "vidstabtransform=input=tf.trf:smoothing=10,unsharp=5:5:0.8:3:3:0.4" -c:a copy OUT
```

`deshake` is the single-pass alternative (lower quality) — same destructive caveat
([vidstab README](https://github.com/georgmartius/vid.stab/blob/master/README.md)).

**Honesty contract:** the shake reader returns advice; the bake is a distinct, named, opt-in op/CLI
that announces it creates new media. We never claim the NLE was stabilized when it wasn't (mirrors
the `add_transition` discipline in `packs/builtins.py`).

---

## 4. Motion-aware speed ramps

Cinematic speed ramping = progressively change speed within a clip, **easing** in/out so it isn't
jarring; the editorial instinct is **slow the action peak, speed through the lull**
([Beverly Boy speed ramps](https://beverlyboy.com/filmmaking/how-to-smooth-speed-ramps-in-premiere-pro/),
[Boris FX optical flow](https://borisfx.com/blog/how-to-use-optical-flow-in-premiere-pro/)).

### Mapping onto film-grip's existing `retime`

The existing `retime` op (`filmgrip/protocol/editplan.py:255`) is a single `speed_percent` per clip
→ `otio.schema.LinearTimeWarp(time_scalar=pct/100)` (or `FreezeFrame` at 0). It is a **constant**
speed warp per clip; OTIO has no multi-point speed curve that round-trips reliably across adapters,
and `retime` is carried by the rebuild path (`adapters/interchange.py:342`).

So the honest, tier-appropriate design is **a piecewise ramp = split + per-segment retime**, not a
fake keyframed curve:

1. `analyze_motion` finds the peak window and the lull window (§2).
2. `suggest_speed_ramp(ir, clip_id)` emits a **descending** op list:
   - `split` the clip at the window boundaries (peak start/end), then
   - `retime` the peak segment to e.g. `40%` (slow-mo) and the lull segment to e.g. `180%` (fast),
     leaving the rest at 100%.
3. This is a *suggestion* (candidate op dicts), exactly like `speech.silence_cut_ops` — the planner
   or a pack chooses to apply it; it runs through the same `validate`→apply pipeline.

True frame-interpolated slow-mo (optical-flow tween) is a **render-time** feature of the NLE and is
**not** something film-grip applies — so the suggestion notes "enable Optical Flow retime in the
NLE for smooth slow-mo," matching the advisory pattern. The "easing" curve itself is a GUI feature;
film-grip's contribution is *choosing where and how much*, deterministically, from motion.

---

## 5. Library comparison

| Library | Role | Dep weight | License | Offline | Speed | Verdict |
|---|---|---|---|---|---|---|
| **opencv-python-headless** | Farneback flow, HSV/Canny for shot math | ~40–60 MB wheel, no GUI libs | BSD-3 ([PyPI](https://pypi.org/project/scenedetect/)) | ✅ | Farneback ~8 ms/frame ([NVIDIA](https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/)) | **adopt** — single new dep |
| **numpy** | array math (already a `color` dep) | already present | BSD | ✅ | fast | **reuse** |
| **ffmpeg** | scdet/select shot fallback, vidstab shake, frame extraction | already required (system binary) | LGPL/GPL (system) | ✅ | fast | **reuse** as fallback |
| **PySceneDetect** | full scene-detect CLI/lib | pulls opencv + own code, BSD-3 ([PyPI](https://pypi.org/project/scenedetect/)) | BSD-3 | ✅ | good | **do not depend at runtime** — its ContentDetector math is ~40 lines we reimplement on cv2/numpy, keeping deps minimal and the core unit-testable on synthetic arrays. Cite it as the algorithm source. |
| **RAFT (torch)** | deep optical flow | PyTorch + GPU (≥12 GB for realtime) ([RAFT](https://arxiv.org/pdf/2003.12039)) | BSD (impl) | model download | ~10 fps GPU | **reject** — too heavy, violates light/offline rule |

Rationale for reimplementing rather than depending on PySceneDetect: the same pattern as
`scopes.analyze_rgb` — keep the **pure-numpy core decoupled from any framework** so correctness is
provable offline on synthetic frames, and keep the optional-extra footprint to one wheel.

---

## 6. Proposed film-grip capabilities

New module: **`filmgrip/perception/motion.py`** (sibling of `scopes.py`). Pure-numpy cores +
ffmpeg/opencv frame iteration, `PerceptionUnavailable` for honest failures, **all outputs in
timeline frames**, retimed clips refused via `align.is_retimed`.

### Readers (deterministic)

```python
# --- pure cores (unit-testable on synthetic arrays, like analyze_rgb) ---

def shot_score(prev_hsv, cur_hsv, *, weights=(1.0, 1.0, 1.0, 0.0),
               edges=False) -> float:
    """ContentDetector frame score: normalized weighted mean-abs HSV(+edge) delta.
    Pure numpy/cv2. >= threshold (default 27.0) ⇒ a cut between these frames."""

def motion_score(prev_gray, cur_gray) -> tuple[float, float]:
    """(mean_magnitude, dominant_angle_deg) from Farneback flow. Pure cv2/numpy."""

def shake_score(global_flow_vectors) -> float:
    """Jitter / net-displacement ratio over a run of per-frame global flow vectors."""

# --- timeline-aware readers (the public payloads) ---

def detect_shots(ir: TimelineIR, clip_id: str, *,
                 threshold: float = 27.0, min_scene_frames: int = 15,
                 adaptive: bool = True, backend: str = "auto",
                 ) -> dict:
    """Shot boundaries inside one clip, in TIMELINE frames.
    Returns {"r","frames":"timeline","clip_id",
             "shots":[{"start","end","score","kind"}],  # kind: "cut"|"gradual"
             "errors":[...]}.
    backend: "opencv" (HSV ContentDetector math) | "ffmpeg" (scdet/select fallback)
             | "auto" (opencv if installed else ffmpeg).
    Refuses retimed clips and offline media with per-clip errors (like align)."""

def analyze_motion(ir: TimelineIR, clip_id: str, *,
                   window_s: float = 0.5) -> dict:
    """Per-window motion in TIMELINE frames via Farneback.
    Returns {"r","frames":"timeline","clip_id",
             "windows":[{"start","end","motion","direction_deg"}],
             "peak":{...}, "lull":{...}, "errors":[...]}."""

def detect_shake(ir: TimelineIR, clip_id: str) -> dict:
    """{"clip_id","shaky":bool,"score":float,"advice":str,"errors":[...]} — advisory only."""
```

Frame iteration reuses the `scopes.frame_rgb` pattern (ffmpeg raw `rgb24`/`gray` on stdout, or a
single ffmpeg pipe decoded frame-by-frame) and `frames.media_time_at` to convert clip-source frames
→ timeline frames. A CLI verb `film-grip motion <clip_id>` / `film-grip shots <clip_id>` mirrors
`cli_scopes.py` / `cli_transcribe.py`.

### Ops / packs

No new *op* is strictly required — both capabilities emit **existing** validated ops:

- **`shot-split` pack** (deterministic, `packs/builtins.py`): `detect_shots` → one `split` op per
  internal boundary (descending order, like `_descending` in `speech.py`). `split` is in the
  rebuild set (`validate.py:143`) so it's always applicable → satisfies the
  `tests/test_packs_applicable.py` honesty contract. Honest failure (no opencv + no ffmpeg, offline
  media) raises `PackError` with the fix, like `silence-cut`.
- **`speed-ramp` suggester**: `suggest_speed_ramp(ir, clip_id)` → descending list of `split` +
  `retime` op dicts (slow the peak, speed the lull). Surfaced as a **prompt pack**
  (`motion-ramp`) so the planner decides magnitudes, *plus* a deterministic helper the planner/pack
  calls — both go through `validate`→apply. (Kept a suggester rather than a built-in deterministic
  pack because "how much to slow" is a creative choice; the deterministic part is *where*.)
- **Stabilization:** **no pack emits it.** Advisory via `detect_shake`; the ffmpeg bake is a
  separate, explicitly-destructive CLI (`film-grip stabilize <clip_id> --bake-new-media`) that
  writes a new file and announces it — never part of an EditPlan.

### Verify integration

`verify.py` already renders boundary contact sheets for *new cuts an edit introduced*
(`new_boundaries`). For shot-detection we add the dual check: after a `shot-split`/`speed-ramp`
plan, the **expected** new boundaries are exactly the detected shot frames — `verify_apply` already
diffs geometry and can render sheets at those frames, so the agent can *look* at each seam. For a
motion ramp, structural verify confirms the `retime` `time_scalar` landed on the right segment
(`geometry_rows` already captures `fx=LinearTimeWarp:scalar`).

### Honesty tiers (explicit)

| Capability | Tier | Behaviour |
|---|---|---|
| `detect_shots`, `analyze_motion`, `detect_shake` | **Reliable evidence** | deterministic, offline, scores reported; retimed/offline clips → honest per-clip errors |
| `shot-split`, `speed-ramp` ops | **Real edits** | only ever emit `split`/`retime` (validated, rebuildable) — never a fake op |
| Dissolve/fade boundaries | **Advisory** | reported as `kind:"gradual"`, documented as lower-confidence |
| In-NLE stabilization | **Not scriptable → advisory only** | reader advises "enable Stabilization in the Inspector"; film-grip cannot toggle it |
| ffmpeg `vidstab` bake | **Destructive, opt-in** | writes new media, announced; never silent, never in a plan |
| RAFT / optical-flow tween slow-mo | **NLE render feature** | film-grip suggests *where/how much*; the tween itself is the NLE's job |

---

## 7. Verify strategy + fixtures

All fixtures synthesized with ffmpeg `lavfi` at test time (no checked-in media), auto-skip when
ffmpeg/opencv absent — matching the existing live/optional-dep test discipline.

### Pure-core unit tests (no ffmpeg, no editor)

- `shot_score`: two synthetic HSV frames that are identical → score ≈ 0; a black frame vs a white
  frame → score well above 27.0. Asserts the weighted-mean-abs formula and threshold semantics.
- `motion_score`: a synthetic frame and a 1-pixel-shifted copy → magnitude > 0 with the expected
  dominant angle; identical frames → magnitude ≈ 0.
- `shake_score`: oscillating global-flow vector array → high; monotonic (pan) array → low.

### Shot-boundary fixture (hard cut)

```bash
# two 1s solid-color segments concatenated → one hard cut at t=1.0s (frame 25 @ 25fps)
ffmpeg -y -f lavfi -i color=c=red:s=320x240:d=1:r=25  red.mp4
ffmpeg -y -f lavfi -i color=c=blue:s=320x240:d=1:r=25 blue.mp4
printf "file 'red.mp4'\nfile 'blue.mp4'\n" > list.txt
ffmpeg -y -f concat -safe 0 -i list.txt -c copy two_shot.mp4
```

Assert: `detect_shots` finds exactly one boundary within **N frames** (N=2) of frame 25; its
`score` is high; `kind == "cut"`. Run for both `backend="opencv"` and `backend="ffmpeg"`.

### Motion-ordering fixtures (static vs panning)

```bash
# static: a single solid color (zero motion)
ffmpeg -y -f lavfi -i color=c=gray:s=320x240:d=2:r=25 static.mp4
# panning: scroll a high-frequency pattern horizontally (strong, directional motion)
ffmpeg -y -f lavfi -i "testsrc=s=640x240:d=2:r=25" \
       -vf "crop=320:240:t*100:0" panning.mp4   # crop window slides → horizontal pan
```

Assert: `analyze_motion(static).peak.motion < analyze_motion(panning).peak.motion`; the panning
clip's `direction_deg` is ~horizontal. (A `noise`-based clip can stand in for high-magnitude
non-directional motion to test the shake/jitter path.)

### Negative / honesty tests

- A **retimed** clip → every reader returns a per-clip error, never frames (assert the
  `is_retimed` guard fires, mirroring `tests/` patterns for transcripts).
- **No opencv + no ffmpeg** → `PerceptionUnavailable` with an install hint; **no opencv, yes
  ffmpeg** → `detect_shots` still works via the `ffmpeg` backend (proves the fallback).
- `shot-split` on offline media → `PackError` listing the real reason (no silent no-op).

---

## 8. Recommended deps

Add one optional extra to `pyproject.toml` (next to `color`):

```toml
# Motion & scene perception (shot detection, optical-flow motion, shake) for cut/retime
# suggestions. opencv-python-headless = BSD-3, offline, no GUI libs; numpy already used by color.
# ffmpeg (system binary) provides a zero-extra fallback for shot detection and shake.
motion = ["opencv-python-headless>=4.8", "numpy>=1.24"]
```

Fold into `all` and `dev` (so the suite exercises the opencv path), exactly as `color`/`transcribe`
are folded in. The ffmpeg fallback means the `shot-split` pack degrades gracefully without the extra.

---

## Sources

- PySceneDetect detectors (signatures, defaults, thresholds): https://www.scenedetect.com/docs/latest/api/detectors.html
- ContentDetector algorithm (DeepWiki): https://deepwiki.com/Breakthrough/PySceneDetect/3.1-content-detector
- ContentDetector source (frame-score formula, Canny edges): https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/detectors/content_detector.py
- PySceneDetect features overview: https://www.scenedetect.com/features/
- PySceneDetect PyPI (license BSD-3, opencv/headless deps): https://pypi.org/project/scenedetect/
- ffmpeg `scdet` filter (threshold range, metadata): https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Video/scdet.html
- ffmpeg scene detection (`select='gt(scene,...)'`) writeup: https://blog.gdeltproject.org/using-ffmpegs-scene-detection-to-generate-a-visual-shot-summary-of-television-news/
- ffmpeg scene detection notes (gist): https://gist.github.com/dudewheresmycode/054c8de34762091b43530af248b369e7
- OpenCV optical flow tutorial (Farneback, cartToPolar): https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- OpenCV optical flow performance (Farneback ~8 ms/frame): https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/
- RAFT paper (deep flow, GPU cost): https://arxiv.org/pdf/2003.12039
- Rethinking RAFT for efficiency: https://arxiv.org/html/2401.00833v1
- vid.stab library: https://github.com/georgmartius/vid.stab
- vid.stab README (deshake vs two-pass): https://github.com/georgmartius/vid.stab/blob/master/README.md
- ffmpeg `vidstabtransform` docs (smoothing, crop/zoom): http://underpop.online.fr/f/ffmpeg/help/vidstabtransform.htm.gz
- Video stabilization with ffmpeg + vidstab (two-pass, destructive): https://www.paulirish.com/2021/video-stabilization-with-ffmpeg-and-vidstab/
- Stabilize shaky video with ffmpeg + vidstab: https://tk-sls.de/wp/4018
- Speed ramping technique (ease in/out): https://beverlyboy.com/filmmaking/how-to-smooth-speed-ramps-in-premiere-pro/
- Optical flow for slow-mo in NLEs (render-time feature): https://borisfx.com/blog/how-to-use-optical-flow-in-premiere-pro/
