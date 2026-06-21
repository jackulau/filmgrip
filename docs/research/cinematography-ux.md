# Cinematography Heuristics + Easiest-UX Research

Research for **film-grip**: (A) deterministic, mostly *advisory* cinematography "perception" readers
(shot classification, composition, pacing) and (B) a survey of best-in-class CLI ergonomics with
concrete recommendations for film-grip's CLI, written as diffs against the real code in
`filmgrip/cli.py` and `filmgrip/cli_status.py`.

> **Honesty framing up front.** Everything in Part A is **ADVISORY** perception — it returns *notes*
> in timeline frames ("CU, slightly low headroom, horizon tilted 4°"), it does **not** apply edits.
> This maps onto film-grip's existing capability-tier discipline: like `scopes`/`transcribe`, these
> readers are pull-on-demand, lazily import their deps, and degrade to an actionable error
> (`PerceptionUnavailable`) instead of faking a result. Within Part A, individual numeric thresholds
> are themselves tiered — **SOURCED** (appears verbatim in authoritative references) vs
> **HEURISTIC/PATENT-ONLY** (practitioner convention or a single patent; calibrate before trusting).
> The whole layer is a *fourth* honesty tier below film-grip's existing ones: not "applied + verified"
> (edit ops), not "computed + verifiable" (scopes/transcripts), but **"computed + advisory"** —
> suggestions a human or the planner may act on, never auto-applied.

---

# PART A — Cinematography Heuristics

## A.1 Shot-type / framing classification

### A.1.1 The taxonomy

There is **no single industry-standard class count** (it ranges 3–9). The useful anchors:

| Taxonomy | Classes | Notes |
|---|---|---|
| **3-class** (CineScale coarse) | Close / Medium / Long | Most *robust* automatically (CNNs hit 94–97%). |
| **5-class** (MovieShots / MovieNet, ECCV 2020) | Long / Full / Medium / Close-Up / Extreme-CU | Cleanest modern standard; ~0.90 accuracy/F1. |
| **7-class** (film grammar) | EWS · WS/Full · Medium-Wide(American) · MS · MCU · CU · ECU | Best for **human-facing labels** incl. in-betweens. |

**Recommendation for film-grip:** emit the **7-class** human label (editors expect "MCU", "American")
but compute it from a continuous ratio so the boundary is one tunable constant, and *also* expose the
coarse 3-class bucket for robustness.

### A.1.2 How shot scale is computed (the codeable signal)

The defining quantity across the CV literature is the **ratio of the subject's HEIGHT to the frame
height** — *not* area or width:

- MovieShots (ECCV 2020): "shot *scale* is defined by the amount of subject figure included within
  the frame." [SOURCED]
- Savardi/Signoroni/Benini (ICIP 2018): the method "uses the ratio of the height of the actor's
  facial image to the height of the video frame." [SOURCED]
- **Why height:** "height ... is more robust than either the width or the area, as it is **invariant
  to face pan rotation**." [SOURCED] → **Use bbox HEIGHT as the primary signal.**

For full-body framing where the face is tiny/undetectable, switch to a **person bounding box** and use
`person_box_height / frame_height` (≈1.0 = full/wide; <0.5 = extreme wide).

### A.1.3 Per-shot frame-height bands

Body-cut lines (left column) are **SOURCED** from film references (Wikipedia, StudioBinder,
MediaCollege). The face-height ladder (right column) is a **HEURISTIC** synthesized from those cut
lines — defensible and codeable, but label it as film-grip's own constant, not an industry standard.

| Shot (7-class label) | Frame cuts at (SOURCED) | `face_h / frame_h` band (HEURISTIC) |
|---|---|---|
| ECU / Big CU | a single feature; chin & crown cropped | `> 0.80` |
| Close-Up (CU) | head, or head-and-shoulders | `0.40 – 0.80` |
| Medium Close-Up (MCU) | head & shoulders → mid-chest | `0.25 – 0.40` |
| Medium Shot (MS) | waist up | `0.15 – 0.25` |
| Medium-Wide / American / Cowboy | knees → mid-thigh up | `0.08 – 0.15` |
| Wide / Full (WS/FS) | head to toe | `0.05 – 0.08` (use person-box ≈1.0) |
| Extreme Wide (EWS/ELS) | figure tiny in environment | `< 0.05` (person-box `< 0.5`) |

**Hard SOURCED numeric anchors** (the only ones verbatim in authoritative *film* sources):
- Full/Wide = subject fills essentially the **entire frame height** head-to-toe.
- Extreme Long Shot = figure is **"less than half the height of the frame"**, down to a dot.
- Close-Up headshot convention = face fills **55–70%** of image height with **10–20% headroom**.
- **Eyes on the upper third** (eyeline ≈ 1/3 down) for CU/MCU.

**PATENT-ONLY area thresholds** (USPTO #8726161 — treat as patent-specific, not universal): Close-up
≥ **35%** of frame *area*; Medium **10–35%**; Long **< 10%**. Useful as a cross-check, not the primary.

**Learned-classifier note (out of scope for v1, documents the ceiling):** SOTA abandons explicit
ratios for CNNs — CineScale VGG-16 = 94% / DenseNet ≈ 97% on 3-class; MovieShots SGNet
(ResNet-50 + saliency "subject map") ≈ 0.90 on 5-class. film-grip should ship the **geometric
ratio** (zero training, deterministic, explainable) and only revisit a learned model if accuracy
demands it.

## A.2 Composition checks from a face/subject bbox

**Coordinate model.** `bbox = (x, y, w, h)`, top-left origin, all normalized `[0,1]`. `x`=left,
`y`=top (top of head/box), `cx = x + w/2`, `cy = y + h/2`.

> **Cross-cutting honesty note.** There is **no published, universally-agreed numeric tolerance** for
> rule-of-thirds, headroom, or lead room. The one hard anchor in nearly every source is **"eyes sit
> ~1/3 (33%) down from the top."** The ±5%/±10% bands and per-shot percentages below are
> **HEURISTIC** practitioner conventions. Horizon tilt is the exception — it has a genuine,
> CV-detectable threshold (auto-straighten tools validate "level" within ≈3°).

### A.2.1 Rule of thirds

Grid (exact): vertical lines `x = 1/3, 2/3`; horizontal `y = 1/3, 2/3`; four power points at their
intersections. **Portrait convention [SOURCED]:** subject body on a vertical line, **eyes on the
upper horizontal line (`y ≈ 1/3`)**.

**The key gotcha — eyes aren't in the bbox.** Derive `eye_y` from the box, with a constant that
depends on the detector type:

- face-only box: `eye_y ≈ y + 0.40*h`
- head box (incl. crown): `eye_y ≈ y + 0.10–0.15*h`
- full-body box: `eye_y ≈ y + 0.05*h`

This `k` is the single biggest error source — pick it per detector.

**Tolerance (HEURISTIC):** "on the line" within **±0.05**; "acceptable" within **±0.10**; beyond →
centered/off. (Empirical support for ±0.05: a Wikipedia example flags eyes "only 28% down, not 33%"
as a *noticeable* lack of headroom — a ~5-point error already reads.) Auto-crop research uses a
**continuous penalty** (saliency-coverage + RoT-distance), not a binary test — film-grip should return
a 0..1 score, not just pass/fail.

```python
POWER_POINTS = [(1/3,1/3),(2/3,1/3),(1/3,2/3),(2/3,2/3)]
def thirds_score(cx, eye_y, falloff=0.10):
    d = min(((cx-px)**2 + (eye_y-py)**2)**0.5 for px,py in POWER_POINTS)
    return max(0.0, 1.0 - d/falloff)   # 1.0 = on a power point, 0 at falloff
```

### A.2.2 Headroom

Definition: gap above the head. **Directly from the bbox:** `headroom = y` (if `y` is head-top; if
`y` is a *face* box, subtract ~0.3–0.5×face_h for the crown first).

**The principle [SOURCED]:** headroom is **dynamic** — set eyes at ≈33% and correct headroom falls
out per shot size ("the closer the subject, the less headroom needed"). So **drive the check off
`eye_y ≈ 0.333`**, then sanity-check raw `y`.

| Shot | Resulting headroom `y` | Tier |
|---|---|---|
| ECU | ~0 / negative (crown cropped) | SOURCED direction |
| CU | ~3–8% | HEURISTIC |
| Medium | ~5–10% | HEURISTIC |
| Wide/full | ~25–33%+ ("one-third or more") | "≥⅓" SOURCED |

Thresholds on `eye_y`: too little `< 0.28` (head crammed/clipped); well-framed `|eye_y − 1/3| ≤ 0.05`;
too much `> 0.40`, hard flag `> 0.45`. **Critical:** gate the too-little check on whether the crown is
actually in frame (i.e. `is_ecu`), or every legitimate tight CU false-positives.

### A.2.3 Lead room / nose room / looking room

Space **in front of** the subject's look/motion direction. Convention [SOURCED, proportional]: subject
on the third-line **opposite** the gaze, ≈ **1/3 behind, 2/3 in front** (Neil Oseman); short-siding
(space behind > in front) reads as tension/unease.

```python
def lead_room(x, w, g):                 # g = +1 looking RIGHT, -1 looking LEFT
    left, right = x, 1.0 - (x + w)
    front = right if g > 0 else left
    back  = left  if g > 0 else right
    if front < back:                       return "short_sided"          # flaw (or intentional)
    front_ratio = front / max(left + right, 1e-6)
    if front_ratio < 0.45:                 return "too_little_leadroom"
    if 0.55 <= front_ratio <= 0.75:        return "good_leadroom"        # ~2/3 convention
    return "acceptable"
```

`g` (gaze/motion sign) comes from face-landmark asymmetry (nose offset within the face box) or, for
motion, optical-flow direction between frames. **HEURISTIC** thresholds; scale the front target up for
fast motion.

### A.2.4 Horizon level / Dutch angle

**The one rule with a real, hard threshold.** Auto-straighten tools validate "level" within
**≈3° (50 mrad)** and correct beyond it (US Patent 10,652,472); on a clean horizon even ~1–2° reads
crooked [SOURCED]. Deliberate Dutch angles are large: subtle 5–15°, moderate 15–45° [SOURCED].

Recommended flags on tilt `|θ|`: `<1°` level; `1–5°` **unintentional crooked horizon** (flag /
suggest straighten `−θ`); `5–10°` ambiguous; `≥10–15°` likely **deliberate Dutch** (don't flag).

Detection = Canny → Hough → angle of the dominant near-horizontal line.

> **Convention gotcha that bites everyone:** `cv2.HoughLines` returns `(rho, theta)` where **theta is
> the NORMAL angle**, so a horizontal line has **theta ≈ 90° (π/2)**; tilt = `theta_deg − 90`.
> `cv2.HoughLinesP` returns endpoints → use `atan2(y2-y1, x2-x1)` where **horizontal = 0°** (preferred
> for horizons). Many blogs get this backwards.

```python
import cv2, numpy as np
def horizon_tilt_deg(img):
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=img.shape[1]//3, maxLineGap=20)
    if lines is None: return None
    tilts = []
    for x1,y1,x2,y2 in lines[:,0]:
        a = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 90) % 180 - 90   # 0 == horizontal
        if abs(a) < 30: tilts.append(a)                              # near-horizontal only
    return float(np.median(tilts)) if tilts else None               # robust dominant tilt
```

## A.3 Subject / face detection libraries

Goal: **light + offline + permissive license**. Summary:

| | Haar cascade | **res10 DNN SSD** | MediaPipe | YOLOv8n |
|---|---|---|---|---|
| pip footprint | opencv ~33–70 MB | **same wheel (DNN built in)** | ~10 MB **+ heavy deps** | <1 MB **+ PyTorch ~3–5.7 GB** |
| Model file | ~908 KB **(in wheel)** | **~10.2 MB** (download once) | ~224 KB `.tflite` (ship yourself) | ~6.3 MB `.pt` |
| License | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** | **AGPL-3.0** (or paid) |
| Offline | **100%, zero download** | offline after 1× manual DL | offline, **ship model**; API churn | **auto-downloads 1st run** |
| Accuracy | frontal only, many FPs | good frontal + some profile, ≥~70px | great near/frontal, weak tiny | **best overall** |
| CPU speed | **~5 ms** / 300² | ~20–25 ms / 300² | ~3–10 ms (near-frontal) | **~80 ms** / 640² |

**Recommendation: res10 DNN primary, Haar fallback. Avoid YOLO; MediaPipe only for dense landmarks.**

- **License is decisive.** OpenCV + MediaPipe = Apache-2.0 (safe to ship/sell). **Ultralytics YOLO =
  AGPL-3.0**, whose network-copyleft clause forces open-sourcing the *entire* tool even if exposed only
  as a service — disqualifying for film-grip's subscription/commercial seam unless an Enterprise
  license is bought. Third-party `yolov8-face` re-trainings **inherit AGPL**.
- **Lightest capable stack** = OpenCV-only: one wheel (`opencv-python-headless`) + a ~10 MB model, no
  PyTorch, no extra ML runtime — and `opencv-python-headless` is already the right dep for a CLI.
- **Offline:** Haar ships *inside* `cv2` (`cv2.data.haarcascades` — confirmed, nothing to fetch). res10
  needs a **one-time manual download** of two files (`deploy.prototxt` + the caffemodel from
  `opencv/opencv_3rdparty`), then runs forever network-free. **Bundle both** so the pipeline is
  deterministic. MediaPipe's Tasks `.task`/`.tflite` models are **not** in the wheel and the legacy
  Solutions API was deprecated (2023) — extra artifact + churn risk.

**Concrete pattern (mirrors film-grip's existing lazy-import + graceful-degradation style):**

```python
def _face_detector():
    import cv2  # lazy, like every perception reader
    proto, model = _bundled("deploy.prototxt"), _bundled("res10_300x300_ssd.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        net = cv2.dnn.readNetFromCaffe(proto, model)
        return ("dnn", net)
    casc = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # always present
    return ("haar", cv2.CascadeClassifier(casc))
```

If neither cv2 nor numpy is installed, raise `PerceptionUnavailable("composition needs opencv +
numpy — install it with: pip install 'film-grip[vision]'")` (exactly the message shape `scopes.py`
already uses for numpy).

## A.4 Pacing / rhythm metrics

film-grip already produces cut points (scene/silence detection in `perception/speech.py` and frames),
so pacing metrics are **pure arithmetic on a list — zero new deps**.

**Average Shot Length (ASL)** = total runtime / number of shots = the mean shot length (Barry Salt,
1974). Interpretation bands [SOURCED, Bordwell/Cutting]:

| ASL | Reads as |
|---|---|
| `< ~3 s` | very fast — action / montage / "intensified continuity" |
| `~3–6 s` | modern mainstream |
| `~6–11 s` | classical Hollywood / measured |
| `> ~15 s` | contemplative / slow cinema |
| minutes | long-take aesthetic (e.g. *Sátántangó* ASL ≈ 10–11 min) |

Historical anchors: classical Hollywood ASL ~8–11 s (1930–60); a roughly **linear decline to <4 s
after 2000** (Cutting, *Psychological Science* 2010; *i-Perception* 2011) — shorthand ~12 s → ~2.5 s.
Genre means (Follows): Action 4.0 s, Horror 15.7 s; director extremes Anderson 2.4 s vs Haneke 26.4 s.

**Rhythm needs dispersion, not just ASL.** Shot-length distributions are **strongly right-skewed**
(many short shots, a few long takes), so the mean is pulled right — across 1,520 Cinemetrics features
the **median is ~62% of the mean (ratio 0.620)**. Report **both** centers; the gap *is* the skew flag.
Redfern argues **median + MAD** beat mean + SD for skewed shot data (robust to outliers); Salt defends
the mean. Resolution: emit both and let the divergence speak.

**Acceleration is the most diagnostic rhythm cue.** Editors build tension by *progressively
shortening* shots toward a climax → a **negative slope of shot length vs. shot index** (Cinemetrics'
regression line). Positive slope = release; ≈0 = steady.

**Metric set film-grip should emit** (per scene / per selection):
`shot_count`, `total_duration`, `asl` (mean), `median`, `sd`, `cv = sd/mean`, `mad` (scaled ×1.4826),
`min`/`max`, `median_mean_ratio` (skew flag), and `trend_slope` of shot length vs. index
(acceleration flag, with sign).

```python
def pacing(cut_times, clip_start, clip_end):
    t = [clip_start, *sorted(cut_times), clip_end]
    s = [t[i+1]-t[i] for i in range(len(t)-1) if t[i+1] > t[i]]
    n = len(s); total = sum(s); mean = total/n
    med = sorted(s)[n//2]
    sd = (sum((x-mean)**2 for x in s)/n) ** 0.5
    mad = 1.4826 * sorted(abs(x-med) for x in s)[n//2]
    # least-squares slope of shot length vs index (negative => accelerating/tension)
    xs = list(range(n)); xm = sum(xs)/n
    slope = sum((xs[i]-xm)*(s[i]-mean) for i in range(n)) / sum((xi-xm)**2 for xi in xs)
    return {"shot_count": n, "total_duration": total, "asl": mean, "median": med,
            "sd": sd, "cv": sd/mean, "mad": mad, "median_mean_ratio": med/mean,
            "trend_slope": slope}
```

## A.5 Proposed reader, verification, and honesty tier

### A.5.1 The reader(s)

Two new modules in `filmgrip/perception/`, matching the existing pull-on-demand, lazy-import,
degrade-to-error contract:

- **`filmgrip/perception/composition.py`** —
  `analyze_composition(media, at=0.0) -> dict` and `classify_shot(bbox, frame_wh) -> dict`.
  Pure-function core: `analyze_frame(arr) -> {shot, shot_3class, face_h_ratio, thirds_score,
  headroom, lead_room, horizon_tilt_deg, notes:[...]}` taking a numpy frame in (decoupled from ffmpeg
  exactly like `scopes.analyze_rgb`), so it is unit-testable on synthetic frames. ffmpeg `frame_rgb`
  reuse for live/media extraction. Returns **ADVISORY notes**, never an EditPlan.
- **`filmgrip/perception/pacing.py`** — `analyze_pacing(cut_times, start, end) -> dict` (the function
  above); fed by the cut points `speech.py`/scene detection already compute. No new deps.

CLI surface (additive — see Part B for the exact `cli.py` diff): a `film-grip compose`
(shot/composition notes for clips or `--at` frames, `--png` to draw the thirds grid + bbox overlay,
mirroring `scopes --png`) and folding pacing into the existing analysis path (or a `--pacing` flag on
`transcribe`/`frames`). MCP-exposed for the planner as advisory context.

### A.5.2 Verification (the film-grip way: synthetic fixtures, no editor/ffmpeg)

The whole point of decoupling the pure functions is that correctness is **provable offline**, the same
way `scopes.analyze_rgb` is tested on synthetic frames:

- **Shot classification:** build an `H×W` array with a filled rectangle of known height at a known
  position → assert `classify_shot` returns the expected 7-class label and that `face_h_ratio` equals
  `rect_h / H` within tolerance. Sweep rectangle heights across every band boundary.
- **Headroom / thirds:** rectangle whose top edge is at a known `y` and centered on `x=1/3` → assert
  `headroom`, `eye_y`, and `thirds_score≈1.0`; move it to center → assert `thirds_score` drops and the
  headroom note flips to "too much".
- **Lead room:** rectangle on the left third with `g=+1` (looking right) → assert `"good_leadroom"`;
  flip `g=-1` → assert `"short_sided"`.
- **Horizon:** synthesize a frame with a single drawn line at a known angle (e.g. 4°) → assert
  `horizon_tilt_deg` recovers it within ~1° and the note is "crooked horizon"; draw at 0° → "level";
  at 30° → no flag (intentional).
- **Pacing:** a list of equal-spaced cuts → `cv≈0`, `trend_slope≈0`; a monotonically shortening list
  → `trend_slope < 0` ("accelerating"); a long-tail list → `median_mean_ratio < 1` (skew).
- **Degradation:** with numpy/cv2 absent, assert `PerceptionUnavailable` with the install hint — not a
  fake result (the project's honesty rule).

### A.5.3 Honesty tier — state it loudly

These readers are **Tier: COMPUTED + ADVISORY**. They:
- return **suggestions** ("consider more headroom", "horizon tilted 4°", "ASL 1.9 s — very fast"),
  expressed as notes in **timeline frames**, never as applied edits;
- never mutate the timeline and are never auto-bundled into FGX (opt-in tokens, like all perception);
- carry an explicit confidence/tier per note where it matters (e.g. horizon tilt = measured; headroom
  "ideal" band = heuristic), so the planner and the user know a thirds-score is a soft heuristic while
  a horizon angle is a measurement;
- degrade to `PerceptionUnavailable` rather than guessing when a dep/model/media is missing.

This slots cleanly under film-grip's `editors`/capability-matrix honesty story: the matrix already
declares what each adapter *can apply*; these readers declare what film-grip can *observe and
advise*, which is a strictly weaker (and clearly labeled) claim.

---

# PART B — Easiest-UX Survey & film-grip CLI Recommendations

## B.1 CLI ergonomics patterns (best-in-class)

Distilled from ffmpeg (anti-pattern), auto-editor, PySceneDetect, yt-dlp, ripgrep, git, gh, cargo,
plus the Elm/Rust error philosophy and **clig.dev** (the canonical modern doctrine).

1. **Progressive disclosure.** Terse `-h`, exhaustive `--help`/man one keystroke away (ripgrep). A
   noun→verb tree with *reused* verbs so learning transfers (gh: `gh <noun> list|view|create`). Group
   a large option set by intent and defer detail to prose (yt-dlp's ~17 labeled groups). **Hide
   deprecated flags** rather than letting `--help` rot (auto-editor prefixes them `[Deprecated]`).
   Anti-pattern: ffmpeg dumps its whole surface → users avoid it and build frontends.

2. **Great `--help`.** *Lead with runnable, scenario-diverse examples* (gh's `EXAMPLES` block);
   **show defaults inline with each flag** (auto-editor: `--margin … (default "0.2s")`) — the thing
   ffmpeg's man page makes you hunt for; document flags in terms of other flags (PySceneDetect:
   `-hq = --rate-factor=17 --preset=slow`); generate help from one source so it can't drift (cargo).

3. **Sensible defaults.** The bare invocation does the 90% job: `auto-editor video.mp4` removes silence
   with a *non-trivial* default (`margin=0.2s`) that makes the no-flag path actually usable; `rg foo`
   recurses, respects `.gitignore`, skips binaries, smart-cases. Express the default *in your own
   override language* so it's self-documenting (yt-dlp's default selector is literally `bv*+ba/b`).
   Never let a default fail silently (ffmpeg `-c copy` cutting on keyframes → black frames).

4. **Dry-run / preview.** Standard `-n/--dry-run` = "show what would happen without doing it"
   (clig.dev). Make **inspection ops default to simulate** (yt-dlp `-F` simulates unless
   `--no-simulate`). auto-editor's `--preview` = "show how the input will be cut and halt." Pair with
   an inspectable interchange so users round-trip the plan.

5. **doctor / self-diagnosis.** The gold standard is `flutter doctor`: per-component ✓/✗/! glyphs, the
   **literal fix command inline** on failure, tiered severity, **read-only**, one final verdict,
   **non-zero exit on problems**, `-v` to escalate, each check runnable in isolation
   (`brew doctor --list-checks`, `npm doctor <check>`). `gh auth status` is a doctor that *tests* the
   token, lists scopes, and ends with the exact re-auth command.

6. **Actionable errors (highest leverage).** The Elm→Rust lineage: "compilers should be assistants,
   not adversaries." Format = **WHAT failed + WHY + the EXACT command/flag to fix it**, friendly,
   non-blaming, with a doc link, and *only suggest when confident* (git refuses "did you mean" on
   ambiguity; rustc tags suggestions `MachineApplicable` vs `MaybeIncorrect` so a guess is never
   auto-applied). clig.dev's own example: *"Can't write to file.txt. You might need to make it
   writable by running `chmod +w file.txt`."* Keep stdout clean so the fix isn't buried (the npm
   `ERESOLVE` anti-pattern hides the fix in a wall).

7. **Progress output.** Friendly bar with ETA **and** a machine stream from the same engine, with
   paired human/raw fields (yt-dlp `_speed_str` vs `speed`); `--newline`/`--no-progress` for CI;
   NDJSON with a `reason` tag + completion terminator (cargo `--message-format json`); stdout = data,
   stderr = logs; self-discoverable `--json` fields (gh prints valid fields on a bad name).

8. **Shell completion.** A `completion <shell>` subcommand emitting a script to stdout, versioned with
   the binary (gh: `gh completion -s zsh > .../_gh`); cheap if the framework generates it
   (Cobra/clap; **Click + argcomplete** for Python).

9. **Quickstart in 30 s.** A single copy-paste command on a real input that visibly proves value — the
   bare invocation *is* the demo (`auto-editor video.mp4`; `cargo new x && cd x && cargo run` →
   `Hello, world!`).

10. **clig.dev principles** worth adopting wholesale: human-first; "conversation as the norm";
    composability (stdout primary, meaningful exit codes, `--json`/`--plain`); TTY-aware color
    (`NO_COLOR`); "if you change state, tell the user"; "suggest the next command"; prefer flags to
    positional args; order-independent flags; standard names (`-n/--dry-run`, `-o/--output`,
    `-q/--quiet`, `-v`, `--json`); **never read secrets from flags/env** (they leak into `ps`); config
    precedence flags > env > project > user > system.

## B.2 Competitor lessons (programmatic video tools)

- **auto-editor** — *the* zero-config model: `auto-editor input.mp4` removes silence; the
  `margin=0.2s` default is what makes the no-flag path non-robotic; `--export premiere|resolve|...`
  emits an **editable timeline** instead of a forced render (treats itself as one pipeline stage). →
  film-grip already does this with `--out` interchange + `--dry-run`; keep leaning in.
- **browser-use/video-use** ("edit videos with coding agents") — the architecture lesson film-grip is
  *already* built on: the LLM **never watches frames**; it reads a ~12 KB word-level transcript
  (always) + on-demand composite PNGs (only at decision points). Their math: "30,000 frames × 1,500
  tokens = 45M tokens of noise" vs "12 KB text + a handful of PNGs." Loop: **propose → approve →
  render → self-verify at each cut**, outputs to `edit/` (raw footage untouched), session memory in
  `project.md`. → validates film-grip's transcript-first + contact-sheet + `--verify` design.
- **OpenMontage** — agentic studio; the trust primitives are **approve-before-execute + a cost
  estimate before acting + a decision log + mandatory post-render quality gate**. → film-grip should
  surface the *billable-call* cost up front (its `--dry-run` help already warns a prompt still makes a
  billable planner call — make that louder).
- **moviepy** — readable object API is a moat, but **in-place mutation + `close()` not freeing memory +
  the v1→v2 `.set_*`→`.with_*` rename storm** broke every tutorial. → immutable copy-returning
  transforms + conservative public-API versioning from day one.
- **ffmpeg-python** — canonical error anti-pattern: `Error: ffmpeg error (see stderr output for
  detail)` while **hiding stderr by default**; the fix (`capture_stderr=True`, read `e.stderr`) is
  undiscoverable; `.compile()` (show the command it'll run) exists but is under-advertised. → if you
  wrap a subprocess, **you own its error UX**: attach stderr by default, surface the compiled plan.
- **Naive LLM editing** — a hands-on test (cutback.video) on a 30-min interview: **93% target miss,
  11/11 cuts mid-sentence, hallucinated timecodes**. Root cause = weak temporal grounding. → the
  honesty/verification difference, not a better prompt, is what makes an editor usable; **ground every
  edit in a time-coded representation and verify the render before claiming success** (film-grip's
  `--verify` is exactly this).

## B.3 Concrete film-grip CLI / UX recommendations

These are **diffs against the real code** in `filmgrip/cli.py` and `filmgrip/cli_status.py`.

### B.3.1 Error-message format — a standard + before/after rewrites

film-grip is already *above average* here: `cli_edit.FixtureError` produces friendly messages and the
exit-code contract is documented (`0 ok, 1 invalid/rejected, 2 editor unreachable, 3 unsupported`).
The gap is **consistency and the "exact fix command"**. Adopt one house format everywhere:

```
error: <WHAT failed, plainly>
  why: <the actual cause>
  fix: <the exact command or flag to run>
  see: <doc/--explain link>            # when one exists
```

**Before / after — fixture not found** (`cli_edit.load_fixture_ir`, current:
`raise FixtureError(f"fixture not found: {path}")`):

```
# BEFORE
FixtureError: fixture not found: cut.oti

# AFTER
error: can't read fixture 'cut.oti'
  why: no such file (did you mean 'cut.otio'?)
  fix: film-grip edit "..." --fixture cut.otio
  see: film-grip editors   (supported fixture types: .otio .fcpxml .mlt .kdenlive .wfp CapCut .json)
```

(The "did you mean" is the git pattern — only emit it when a close filename exists; never guess.)

**Before / after — editor unreachable** (today the user gets exit 2 from the edit path; the rich
diagnosis only lives in `status`). Make the *error* point at the doctor and the one blocking fix:

```
# BEFORE
(connection fails; exit code 2, terse message)

# AFTER
error: can't reach DaVinci Resolve
  why: the Resolve scripting module isn't importable on this machine
  fix: open Resolve (Studio) ▸ Preferences ▸ System ▸ General ▸ "External scripting using" = Local
  see: run `film-grip status` for the full preflight, or try --fixture cut.otio to work offline
```

Note this reuses the exact remediation string already in `cli_status.status_guidance` — **factor that
guidance into one helper** so the doctor and the live-edit error speak with one voice (DRY, and the
user never gets two different explanations of the same problem).

**Before / after — missing perception dep** (the new readers): keep the `scopes.py` shape but add the
`fix:` line consistently —

```
# AFTER
error: composition analysis needs OpenCV + numpy
  why: neither is importable in this environment
  fix: pip install 'film-grip[vision]'
```

### B.3.2 `film-grip quickstart` / onboarding

Add a `quickstart` subcommand (and lead the README with it) — a single copy-paste that proves value
**offline, with no editor, no API key, no billable call**, using a bundled fixture:

```
$ film-grip quickstart
1/3  Bundled demo timeline: cut.otio (3 clips)                         ✓
2/3  Dry-run edit (offline, no Claude call) — "trim 0.5s off clip 1":
       clip1  in 00:00:00:00 → 00:00:00:12   (was …:00:00)            [diff shown]
3/3  Next steps:
       • see what film-grip can do:   film-grip editors
       • check your editor is wired:  film-grip status
       • do it for real on Resolve:   film-grip edit "add a blue marker on the selected clip"
Tip: every command takes --fixture <file> to run with no editor.
```

This implements the clig.dev "30-second value" + "suggest the next command" patterns, and crucially the
demo path is `--plan`-backed (offline, deterministic) so it can never fail on a missing key or
hallucinate — honest by construction.

### B.3.3 Smart defaults (mostly already good; small wins)

- The `--editor resolve` default and the offline `--fixture`/`--plan` paths are already strong. Keep.
- **TTY-aware output + `--json`.** Perception/status output is plain text; add a `--json` flag (status,
  scopes, the new compose/pacing) for orchestrators, and disable any color/glyphs when stdout isn't a
  TTY or `NO_COLOR` is set (clig.dev). The `status` doctor is the highest-value `--json` target (CI can
  assert readiness).
- **Make the billable-call warning impossible to miss.** `edit --dry-run` help already notes "the
  planner still makes a billable call"; echo that to stderr at runtime too (OpenMontage's
  cost-before-acting), so `--dry-run` never surprises a user with a charge.
- **`completion` subcommand.** Add `film-grip completion <bash|zsh|fish>`; argparse needs `argcomplete`
  (or migrate to Click) — low effort, table-stakes ergonomics.

### B.3.4 `film-grip status` (the doctor) — what it should check

`cli_status.cmd_status` already does this well for the Resolve path (module importable, app running,
Studio, project/timeline open, ordered single-best next step, SFX summary, LLM auth, editor count) and
correctly **always exits 0**. Upgrades to make it a flutter-grade doctor:

- **Glyph column + severity tiers.** Render `✓ / ✗ / !` per row (it currently prints `yes/no/?`), so
  blockers vs warnings are scannable at a glance.
- **`--json` and a non-zero `--exit-code` mode.** Keep the default exit 0 (it's a diagnostic), but add
  `film-grip status --exit-code` that returns non-zero when a hard blocker is present, so CI/agents can
  gate on it (the `brew doctor`/`gh auth status` contract). `--json` emits the `report` dict + guidance.
- **Add the checks the doctor is currently missing** (it diagnoses the editor + SFX + auth, but not the
  perception toolchain the new readers and existing `scopes`/`frames`/`transcribe` depend on):
  - **ffmpeg present?** — resolve `ffmpeg`/`ffprobe` on PATH + version (the contact-sheet, scopes, and
    proposed composition readers all shell out to it; `perception/transcribe.ffmpeg_path` already
    exists — surface its result).
  - **numpy present?** — required by `scopes` and the proposed composition/pacing readers.
  - **OpenCV present? + face model bundled?** — for the proposed composition reader (`✓ res10 DNN`
    vs `! Haar fallback only` vs `✗ neither`).
  - **ASR backend?** — which of `faster-whisper | whisper-cpp | elevenlabs` is available (the
    `transcribe --asr` auto-detect), with the install fix for the preferred one if none.
  - **Resolve reachable?** — already covered; keep as the headline.
  - **SFX dir?** — already summarized (`base — N effects, M missing`); keep, and flag `!` when 0
    effects or any missing files.
  - Each missing item prints the **exact fix command** inline (`pip install 'film-grip[vision]'`,
    `brew install ffmpeg`, `pip install faster-whisper`) — the flutter/brew pattern.

A doctor that covers *both* the editor seam **and** the perception toolchain means a user can run one
command and know exactly why any subcommand (edit, scopes, frames, transcribe, compose) would or
wouldn't work — and get the one command to fix each gap.

### B.3.5 Ranked summary of UX improvements (by impact)

1. **Upgrade `film-grip status` into a full doctor** — add ffmpeg/numpy/OpenCV+model/ASR checks,
   `✓/✗/!` glyphs, `--json`, and an opt-in non-zero `--exit-code`, each with an inline fix command.
   Single highest-leverage change: one command tells the user why *any* subcommand works or doesn't.
2. **Standardize actionable errors (WHAT/WHY/FIX/SEE)** and **factor `status_guidance` into a shared
   helper** so the live-edit "editor unreachable" error and the doctor give one identical, fix-carrying
   message. Rewrite the terse `FixtureError`/exit-2 cases into the house format (with confidence-gated
   "did you mean").
3. **Add `film-grip quickstart`** — an offline, no-key, `--plan`-backed 30-second demo that ends with
   "suggest the next command", and lead the README with it. Honest by construction (no billable call,
   no hallucination).
4. **Add `--json` (machine output) across status/scopes/compose/pacing + TTY-aware color**, so
   film-grip composes into agents/CI — directly enabling the propose→approve→verify loop competitors
   prove out.
5. **Add `film-grip completion <shell>`** (bash/zsh/fish via argcomplete or a Click migration) — and
   make the **billable-call cost** explicit at runtime on `edit --dry-run`. Table-stakes ergonomics +
   the OpenMontage "cost before acting" trust primitive.

---

# Sources

### Part A — Shot scale / classification
- CineScale dataset (Data in Brief): https://www.sciencedirect.com/science/article/pii/S2352340921002869 · taxonomy: https://cinescale.github.io/shotscale/
- Savardi et al., "Shot Scale Analysis in Movies by CNNs" (ICIP 2018): https://ieeexplore.ieee.org/document/8451474/
- MovieShots / "Subject Centric Lens" (ECCV 2020): https://arxiv.org/pdf/2008.03548 · https://movienet.github.io/projects/eccv20shot.html
- MovieNet: https://ar5iv.labs.arxiv.org/html/2007.10937 · 7-class face-height/frame-height: https://www.researchgate.net/publication/224319737
- Lightweight weak-semantic framework (Sci Reports 2023): https://www.nature.com/articles/s41598-023-43281-w
- USPTO bbox-area % thresholds (patent #8726161, patent-specific): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8726161

### Part A — Film-grammar shot sizes
- StudioBinder shot-size guide: https://www.studiobinder.com/blog/types-of-camera-shots-sizes-in-film/ · Medium Long: https://www.studiobinder.com/blog/what-is-a-medium-long-shot-in-film/ · Extreme Long: https://www.studiobinder.com/blog/what-is-an-extreme-long-shot/
- Wikipedia: Shot (filmmaking): https://en.wikipedia.org/wiki/Shot_(filmmaking) · Long/Wide: https://en.wikipedia.org/wiki/Long_shot · Close-up: https://en.wikipedia.org/wiki/Close-up
- MediaCollege: https://www.mediacollege.com/video/shots/ · Wikibooks shot sizes: https://en.wikibooks.org/wiki/Movie_Making_Manual/Cinematography/Shot_Sizes
- Cadrage 7 shot sizes: https://www.cadrage.app/the-7-most-common-shot-sizes-in-filmmaking/ · headshot 55–70%: https://www.betterpic.io/blog/size-of-headshot

### Part A — Composition
- Wikipedia: Rule of thirds: https://en.wikipedia.org/wiki/Rule_of_thirds · Headroom: https://en.wikipedia.org/wiki/Headroom_(photographic_framing) · Lead room: https://en.wikipedia.org/wiki/Lead_room · Dutch angle: https://en.wikipedia.org/wiki/Dutch_angle
- StudioBinder: Rule of Thirds: https://www.studiobinder.com/camera-shots/composition/rule-of-thirds-in-film/ · Dutch angle: https://www.studiobinder.com/blog/dutch-angle-shot-camera-movement/
- Neil Oseman, lead/nose room: https://neiloseman.com/lead-room-nose-room-or-looking-space/
- Premiumbeat framing errors: https://www.premiumbeat.com/blog/common-framing-errors-in-cinematography/ · CineD headroom: https://www.cined.com/the-art-of-imbalance-exploring-the-impact-of-headroom-in-storytelling/
- Saliency/auto-crop (continuous RoT penalty): https://arxiv.org/pdf/1911.10492 · Twitter/X crop learnings: https://blog.x.com/engineering/en_us/topics/insights/2021/sharing-learnings-about-our-image-cropping-algorithm
- OpenCV Hough (theta = normal angle): https://docs.opencv.org/4.13.0/d6/d10/tutorial_py_houghlines.html · maritime horizon (arXiv 2110.13694): https://arxiv.org/pdf/2110.13694
- Horizon-correction ~3° (US 10,652,472): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10652472 · auto-framing dead-band (US 8,274,544): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8274544

### Part A — Detection libraries
- OpenCV license (Apache-2.0 since 4.5.0): https://opencv.org/license/ · opencv-python-headless: https://pypi.org/project/opencv-python-headless/
- Haar XML in-wheel (`cv2.data.haarcascades`): https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/ · res10 DL + size: https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
- LearnOpenCV (speed/accuracy): https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
- mediapipe (PyPI + deps): https://pypi.org/project/mediapipe/ · Solutions→Tasks deprecation: https://ai.google.dev/edge/mediapipe/solutions/guide · Face Detector (BlazeFace latency): https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
- Ultralytics license (AGPL-3.0 + Enterprise): https://www.ultralytics.com/license · on-device AGPL issue: https://github.com/ultralytics/ultralytics/issues/19390 · yolov8n CPU ~80ms (arXiv 2407.02988): https://arxiv.org/pdf/2407.02988

### Part A — Pacing
- Bordwell, "Intensified Continuity" (PDF): https://cinecdoque.wordpress.com/wp-content/uploads/2015/03/bordwell-intensified-continuity.pdf
- Cinemetrics — "Median or Mean" (Median:ASL = 0.620): https://cinemetrics.uchicago.edu/article/e8c730d6-7312-4af6-9454-f9c847789752 · "The Metrics in Cinemetrics" (Salt): https://cinemetrics.uchicago.edu/article/616e7ecc-7915-4768-b84d-7dec79aa77c2 · home: https://cinemetrics.uchicago.edu/
- Cutting, DeLong & Nothelfer (2010), Psychological Science: https://journals.sagepub.com/doi/10.1177/0956797610361679 · Cutting et al. (2011), "Quicker, Faster, Darker": https://journals.sagepub.com/doi/10.1068/i0441aap
- Redfern, "Statistical illiteracy in film studies" (robust median+MAD): https://nickredfern.wordpress.com/2012/02/02/statistical-illiteracy-in-film-studies/
- Stephen Follows, shots per movie / genre ASL: https://stephenfollows.com/p/many-shots-average-movie · Slow cinema (Sátántangó): https://en.wikipedia.org/wiki/Slow_cinema

### Part B — CLI ergonomics
- clig.dev — Command Line Interface Guidelines: https://clig.dev/ · NN/G error guidelines: https://www.nngroup.com/articles/error-message-guidelines/
- ffmpeg anti-patterns (HN): https://news.ycombinator.com/item?id=33771445 · https://news.ycombinator.com/item?id=25488027
- auto-editor: https://github.com/WyattBlue/auto-editor · cli source: https://raw.githubusercontent.com/WyattBlue/auto-editor/master/src/cli.nim · options: https://auto-editor.com/ref/options
- PySceneDetect CLI: https://www.scenedetect.com/docs/latest/cli.html · https://www.scenedetect.com/cli/
- yt-dlp man: https://www.mankier.com/1/yt-dlp · README: https://github.com/yt-dlp/yt-dlp/blob/master/README.md
- ripgrep guide + defaults: https://burntsushi.net/ripgrep/ · https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md
- git config (autocorrect/advice): https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration · detached HEAD: https://www.cloudbees.com/blog/git-detached-head
- gh manual: https://cli.github.com/manual/ · gh_pr_create: https://cli.github.com/manual/gh_pr_create · gh_auth_status: https://cli.github.com/manual/gh_auth_status · gh_completion: https://cli.github.com/manual/gh_completion · Primer CLI principles: https://primer.style/cli/getting-started/principles
- cargo book + JSON output: https://doc.rust-lang.org/cargo/reference/external-tools.html · new project: https://doc.rust-lang.org/cargo/guide/creating-a-new-project.html
- Elm: "Compilers as assistants": https://elm-lang.org/news/compilers-as-assistants · "Compiler errors for humans": https://elm-lang.org/news/compiler-errors-for-humans · "The syntax cliff": https://elm-lang.org/news/the-syntax-cliff
- Rust: "Shape of errors to come": https://blog.rust-lang.org/2016/08/10/Shape-of-errors-to-come/ · RFC 1644 (what/why labels): https://rust-lang.github.io/rfcs/1644-default-and-expanded-rustc-errors.html · diagnostics guide (Applicability; avoid "did you mean"): https://rustc-dev-guide.rust-lang.org/diagnostics.html · E0382 example: https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
- doctor commands: flutter https://docs.flutter.dev/get-started/install/help · brew https://docs.brew.sh/Manpage · npm https://docs.npmjs.com/cli/v11/commands/npm-doctor/
- shell completion frameworks: Cobra https://cobra.dev/docs/how-to-guides/shell-completion/

### Part B — Programmatic video competitors
- auto-editor: https://github.com/WyattBlue/auto-editor · "what is silent?": https://github.com/WyattBlue/auto-editor/discussions/398
- browser-use/video-use ("edit videos with coding agents"): https://github.com/browser-use/video-use · install: https://github.com/browser-use/video-use/blob/main/install.md
- OpenMontage: https://github.com/calesthio/OpenMontage · overview: https://www.scriptbyai.com/open-ai-video-production-agent/
- moviepy: https://github.com/Zulko/moviepy · v1→v2 migration: https://zulko.github.io/moviepy/getting_started/updating_to_v2.html · memory issues: https://github.com/Zulko/moviepy/issues/1892
- ffmpeg-python: https://github.com/kkroening/ffmpeg-python · hidden-stderr issue: https://github.com/kkroening/ffmpeg-python/issues/165 · stream mapping: https://github.com/kkroening/ffmpeg-python/issues/275
- LLM editing reliability (93% miss / mid-sentence cuts / hallucinated timecodes): https://cutback.video/blog/how-to-edit-videos-using-an-llm-chatgpt-vs-claude-vs-selects
- VideoAgent (HKUDS): https://github.com/HKUDS/VideoAgent · LAVE: https://www.dgp.toronto.edu/~bryanw/lave/ · VidHal (temporal hallucination): https://arxiv.org/pdf/2411.16771
