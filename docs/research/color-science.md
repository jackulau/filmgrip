# Color Science for Accurate Scopes + Honest Grading

Deep-research notes for **film-grip**'s synthesized scopes (`filmgrip/perception/scopes.py`) and
grading honesty. Every formula below was recomputed against `filmgrip`'s actual code and validated
with numpy + ffmpeg on this machine (numpy 2.4.6, ffmpeg with `smptehdbars`/`smptebars`/`testsrc`).
All non-obvious numbers carry a source URL.

---

## Summary

`scopes.py` is closer to correct than most "text→LUT" toys — it already uses the **right Rec.709
luma weights** (0.2126/0.7152/0.0722) and, despite the misleading variable comment, its vectorscope
chroma coefficients are the **correct Rec.709 Pb/Pr** matrix (not Rec.601). Validated:
a 75% red patch plots at **102.9°** in its `atan2(Cr, Cb)` convention, matching the textbook
"red ≈ 104°" vectorscope target. The math engine (`apply_cdl_array`) matches `CDL.apply`.

Three real correctness bugs / gaps were found (details + diffs in *Proposed changes*):

1. **Skin-tone line angle is wrong: `SKIN_TONE_ANGLE_DEG = 115.0` should be ≈ `123.0`.** Measured
   real skin samples land at 124–129° and the YIQ +I (orange) axis computes to ~121° in this
   convention; 115° systematically inflates `skin_tone_delta_deg` by ~8°.
2. **No legal/full range handling.** `frame_rgb` always emits *full-range* rgb24, but the clip/crush
   thresholds (16/235) and IRE semantics assume the colorist's 0–100 IRE scale. Legal-range (16–235)
   source decoded to full-range RGB will read blacks at 16 and never trip the crush flag correctly,
   and there is no IRE field at all. Need an explicit range model + IRE outputs.
3. **No log/ACES awareness.** `analyze_rgb` reads every clip as if display-referred Rec.709. Log/ACES
   footage reads as low-contrast/desaturated and any "verdict" (exposure/cast) is misleading. Need a
   `detect_log_footage` advisory.

Plus two missing capabilities the research strongly motivates: **`estimate_white_balance` → suggested
CDL offset** (gray-world / shades-of-gray) and **exposure/false-color** metrics. And a real
**scope-accuracy verify fixture**: ffmpeg SMPTE bars / solid patches / ramp → assert measured
angles/levels land at known values within tolerance.

---

## 1. Scope math (with formulas)

### 1.1 Luma waveform

Rec.709 luma from non-linear (gamma) R'G'B' — what a waveform monitor actually shows ("luma", the
weighted sum of the coded primaries), **not** linear-light luminance:

```
Y' = 0.2126·R' + 0.7152·G' + 0.0722·B'
```

These are the BT.709 coefficients ([Wikipedia — Rec. 709](https://en.wikipedia.org/wiki/Rec._709),
[Wikipedia — YCbCr](https://en.wikipedia.org/wiki/YCbCr)). `scopes.py` `_LUMA` is already correct.
(BT.601 SD uses 0.299/0.587/0.114; BT.2020 uses 0.2627/0.6780/0.0593 — only relevant if film-grip
ever needs SD or wide-gamut scopes.)

**Code values vs IRE vs full/legal range** — the single most confused area:

* **Full range ("data"/PC, 0–255 8-bit):** black = 0, white = 255. This is what `ffmpeg -pix_fmt
  rgb24` emits, and what `analyze_rgb` currently assumes.
* **Legal/video range ("studio swing", 8-bit):** black = **16**, white = **235** for luma;
  chroma 16–240 centered on 128 ([Wikipedia — Rec. 709](https://en.wikipedia.org/wiki/Rec._709):
  "reference black … code 16 and reference white … code 235";
  [thepostprocess.com — full vs video](https://www.thepostprocess.com/2019/09/24/how-to-deal-with-levels-full-vs-video/)).
* **IRE (0–100):** the colorist's waveform scale. 0 IRE = reference black, 100 IRE = reference white.

Conversions (8-bit):

```
# legal code -> IRE
IRE          = (code - 16) / (235 - 16) * 100            # = (code-16)/2.19
# full-range code -> IRE (if pixels are already full-range, map 0..255 -> 0..100)
IRE_full     = code / 255 * 100
# normalized 0..1 (legal)  ->  legal 8-bit code
code         = round(16 + v01 * 219)                      # luma
# full <-> legal scaling of an 8-bit value
legal_to_full = (code - 16) * 255/219                     # clamp 0..255
full_to_legal = round(16 + code * 219/255)
```

Validated on this machine: legal codes 16/128/235 → 0.00 / 51.14 / 100.00 IRE.

**Clip/crush.** Current `CLIP_THRESHOLD=235`, `CRUSH_THRESHOLD=16` are *legal-range* anchors applied
to *full-range* pixels — i.e. they currently flag "near the legal limits" inside full-range data,
which is neither the data limit (0/255) nor IRE-correct. The fix is to make thresholds a function of
the declared range (full: crush ≤ ~2/255, clip ≥ ~253/255; legal: crush ≤ 16, clip ≥ 235) and to
also emit IRE so the LLM reasons in the colorist's units.

### 1.2 RGB parade

Per-channel distribution of R', G', B'. `scopes.py` already reports the right summary — percentiles
(1/10/50/90/99) per channel — which is more robust and more LLM-legible than a raw column image.
The only fix is the same range concern: expose the percentiles in **both** code values and IRE, and
read black/white points off `p1`/`p99` per channel (the white-balance read is the spread of the three
channels at the same percentile). No coefficient change needed.

### 1.3 Vectorscope

A vectorscope plots the two **color-difference** components on a 2-D plane: horizontal ≈ B−Y (Cb/Pb/U),
vertical ≈ R−Y (Cr/Pr/V); hue = angle, saturation = radius
([chriskiehl — "what is red doing over there"](https://chriskiehl.com/article/what-is-red-doing-over-there);
[tek.com — bars vs targets](https://www.tek.com/en/support/faqs/why-are-my-color-bars-not-hitting-vectorscope-targets)).

**Rec.709 analog color-difference (normalized R'G'B' in 0..1, output ≈ −0.5..0.5)**
([Wikipedia — YCbCr](https://en.wikipedia.org/wiki/YCbCr)):

```
Pb (Cb) = -0.1146·R' - 0.3854·G' + 0.5·B'
Pr (Cr) =  0.5·R'   - 0.4542·G' - 0.0458·B'
```

**These are exactly the coefficients in `scopes.py`** (the inline comment calls them "Rec.709 chroma"
correctly; the docstring elsewhere worrying about 601-vs-709 does not apply here — they ARE 709).
BT.601 would be `Pb = -0.168736R' -0.331264G' +0.5B'`, `Pr = 0.5R' -0.418688G' -0.081312B'` — *not*
what the code uses, so **no bug here.** (U,V in classic analog YUV are scaled color differences:
`U = 0.492·(B'−Y')`, `V = 0.877·(R'−Y')`; Cb/Cr are the normalized form. For angle/relative-magnitude
purposes the scaling is irrelevant; only Pr-vs-Cr axis scaling would rotate angles, and the code's
unscaled Pb/Pr is the standard vectorscope mapping.)

**Angle convention.** `scopes.py` uses `hue = atan2(Cr, Cb)`, i.e. 0° points along +Cb (toward blue/
B−Y) and 90° along +Cr (toward red/R−Y). Measured graticule angles in THIS convention (computed from
75% bars RGB, validated on machine):

| Color (75%) | angle (this convention) | "o'clock" |
|-------------|-------------------------|-----------|
| **Red**     | **102.91°**             | ~11:00    |
| Yellow      | 174.77°                 | ~9:00     |
| Green       | 229.68°                 | ~7:30     |
| Cyan        | 282.91°                 | ~5:00     |
| Blue        | 354.77°                 | ~3:00     |
| Magenta     | 49.68°                  | ~1:00     |

Red at ~103° matches the textbook vectorscope "red ≈ 104°"
([chriskiehl](https://chriskiehl.com/article/what-is-red-doing-over-there): 104° = 123° − 19.3°,
i.e. +I axis minus the I-axis offset). Targets carry a tolerance of **±5° / ±5% saturation** on a
calibrated scope ([tek.com](https://www.tek.com/en/support/faqs/why-are-my-color-bars-not-hitting-vectorscope-targets)) —
a good tolerance for film-grip's verify assertions.

**Saturation magnitude** = `mean(sqrt(Cb² + Cr²))`. Fine as a *relative* metric (already used by the
verify loop). Note: averaging Cb/Cr over a multi-hue frame collapses opposite hues — confirmed on
machine, full-frame SMPTE bars report a meaningless mean hue of 247°. For dominant-hue reads, prefer
per-region or per-bar sampling (see fixtures). The mean-vector read is only meaningful for
single-cast frames (which is exactly the white-balance use case).

### 1.4 Skin-tone "I" line

The flesh-tone line is the YIQ **+I axis**, conventionally drawn at **~123°** ("11 o'clock") on a
vectorscope; skin tones of all ethnicities cluster along it (they differ in saturation/brightness,
not hue) ([Bram Stout — Vectorscopes](https://bramstout.nl/en/webbooks/vectorscopes/) notes phase
angles 116–126° with 123° used for the +I axis;
[larryjordan.com](https://larryjordan.com/articles/color-correction-make-people-look-normal/)).

In `scopes.py`'s `atan2(Cr,Cb)` convention I measured:
* YIQ +I (orange) direction → **~121°**
* real skin samples: (210,150,120)→**124.9°**, (228,180,150)→128.9°, (200,140,110)→124.9°

So the correct reference for this convention is **≈ 123°**, *not* the hard-coded `115.0`. The current
value biases `skin_tone_delta_deg` high by ~8° for correctly-balanced skin. **Fix: set
`SKIN_TONE_ANGLE_DEG = 123.0`** (and document the convention).

### 1.5 Histogram

Luma histogram is fine as-is (8 bins over 0–255). For exposure work a finer histogram (64–256 bins)
plus zone summary is more useful (see §3).

---

## 2. Color spaces / log / ACES + detection

### 2.1 Rec.709 vs sRGB gamma (why 2.2 ≠ 2.4)

sRGB and Rec.709 share **identical primaries + D65 white** — only the transfer function differs
([lightillusion.com](https://lightillusion.com/colourspace_manual.html)).

**sRGB EOTF / OETF** (piecewise, [Wikipedia — sRGB](https://en.wikipedia.org/wiki/SRGB)):

```
# encoded -> linear (EOTF)
lin = c/12.92                       if c <= 0.04045
lin = ((c+0.055)/1.055)**2.4        if c >  0.04045
# linear -> encoded (OETF)
c = 12.92*lin                       if lin <= 0.0031308
c = 1.055*lin**(1/2.4) - 0.055      if lin >  0.0031308
```

Literal exponent 2.4 but **effective gamma ≈ 2.2** because of the linear toe + 0.055 offset
([Wikipedia — Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction)).

**BT.709 OETF** (camera/encode only — [Wikipedia — Rec. 709](https://en.wikipedia.org/wiki/Rec._709)):

```
V = 4.5*L                  if 0 <= L < 0.018
V = 1.099*L**0.45 - 0.099  if L >= 0.018       # 0.45 ≈ 1/2.22
```

BT.709 **deliberately defines no decode EOTF**; the reference *display* uses **BT.1886 ≈ pure 2.4
gamma** (with Lb=0 it is exactly `L = V^2.4`)
([Wikipedia — BT.1886](https://en.wikipedia.org/wiki/ITU-R_BT.1886)). The round-trip is intentionally
non-unity: **system gamma ≈ 1.2** (0.5×2.4) to compensate for the dim viewing surround
([ITU-R BT.2390-10 §6.2](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2390-10-2021-PDF-E.pdf)).

**Why it matters for grading:** mid-gray (input 0.5) decodes to **0.2176** under 2.2 but **0.1895**
under 2.4 — a **~13% / 0.2-stop** midtone shift (recomputed). Judging contrast/saturation on the
wrong decode bakes a correction for a problem that only exists because of the gamma mismatch
([mixinglight.com — 2.2 vs 2.4](https://mixinglight.com/color-grading-tutorials/gamma-2-2-vs-gamma-2-4-davinci-resolve/)).
**Implication for film-grip:** `analyze_rgb` reads coded values, which is the right domain for a
*waveform/vectorscope* (they are display-side, gamma-domain instruments) — so we should NOT
linearize for the scope read, but we SHOULD record the assumed transfer (709/2.4 vs sRGB/2.2) in the
report so downstream reasoning is unambiguous.

### 2.2 Camera log curves — anchor table

Log is a **data container**, not a picture: a logarithmic OETF packs 12–14+ stops into limited bits,
trading midtone precision for highlight/shadow latitude
([ARRI — Log C](https://www.arri.com/en/learn-help/learn-help-camera-system/image-science/log-c)).
A detector keys off two anchors — where **black (0% reflectance)** and **18% mid-gray** land. Values
below are **normalized 0–1 full-range** (and naive 8-bit = ×255), recomputed from the official
curve specs:

| Curve | Black (x=0) 0–1 / 8-bit | 18% gray 0–1 / 8-bit | 90% white 0–1 | Source |
|-------|-------------------------|----------------------|---------------|--------|
| **Sony S-Log3** | **0.0929** / 24 | **0.4106** / 105 | 0.585 | [Sony tech PDF](https://pro.sony/s3/cms-static-content/uploadfile/06/1237494271406.pdf) |
| Sony S-Log2 | ~0.088 / 22 | ~0.32 / 82 | ~0.59 | [wolfcrow](https://www.wolfcrow.com/how-to-expose-s-log3/) |
| **Canon C-Log** | **0.1251** / 32 | **0.3434** / 88 | 0.600 | [Canon WP PDF](https://www.usa.canon.com/content/dam/canon-assets/white-papers/pro/white-paper-canon-log-gamma-curves.pdf) |
| **Canon C-Log2** | **0.0929** / 24 | **0.3983** / 101 | 0.562 | Canon WP (CV95 black) |
| **Canon C-Log3** | **0.1251** / 32 | **0.3434** / 88 | 0.564 | Canon WP (CV128 black) |
| **ARRI LogC3 EI800** | **0.0928** / 24 | **0.3910** / 100 | 0.559 | [ARRI LogC VFX PDF](https://www.arri.com/resource/blob/31918/66f56e6abb6e5b6553929edf9aa7483e/2017-03-alexa-logc-curve-in-vfx-data.pdf) |
| ARRI LogC4 | ~0.0929 / 24 | ~0.2784 / 71 | — | [LogC4 PDF](https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf) |
| **RED Log3G10** | **0.0916** / 23 (v1=0.0) | **0.3333** / 85 | 0.483 | [RED WP PDF](https://docs.red.com/955-0187/PDF/915-0187%20Rev-C%20%20%20RED%20OPS,%20White%20Paper%20on%20REDWideGamutRGB%20and%20Log3G10.pdf) |
| **Panasonic V-Log** | **0.1250** / 32 | **0.4233** / 108 | 0.588 | [Panasonic V-Log PDF](https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf) |

Representative formulas (full set in sources):

```
# Sony S-Log3 (linear L -> code V, 0..1), breakpoint L=0.01125
V = (420 + log10((L+0.01)/0.19)*261.5)/1023            if L >= 0.01125
V = (L*(171.2102946929-95)/0.01125 + 95)/1023          if L <  0.01125
# ARRI LogC3 EI800
t = c*log10(a*x+b)+d if x>cut else e*x+f
# cut=0.010591,a=5.555556,b=0.052272,c=0.247190,d=0.385537,e=5.367655,f=0.092809
# Panasonic V-Log, cut=0.01
out = 5.6*in+0.125             if in<0.01
out = 0.241514*log10(in+0.00873)+0.598206  if in>=0.01
```

**Keying insight:** every common log curve plants its black floor at 8-bit **~23–32** (never near 0)
and diffuse white at **~123–153** (never near 255), with mid-gray at **~85–108**. That non-zero black
floor + depressed white + bunched mids is the detectable fingerprint.

### 2.3 ACES, and why grading log/ACES as 709 is wrong

ACES encodings ([Wikipedia — ACES](https://en.wikipedia.org/wiki/Academy_Color_Encoding_System),
[docs.acescentral.com](https://docs.acescentral.com/)): **ACES2065-1** (AP0, linear, interchange);
**ACEScg** (AP1, linear, VFX working); **ACEScc/ACEScct** (AP1, log, grading — ACEScct adds a toe).
Pipeline: `Camera → IDT → ACES working → [LMT] → RRT+ODT (Output Transform) → display`.

Grading log/ACES **as if 709** stacks two transfer functions that were never meant to combine — the
math is wrong: shadows crush in the wrong place, skin casts, highlights blow, the image reads muddy
([filmit.io](https://filmit.io/blog/log-vs-rec709-lut-mistake/)). Crucially **scopes lie on raw log/
ACES**: a correctly-exposed S-Log3 clip reads reference white at ~61% and saturation artificially
low; only after the input transform do values expand to the full 0–100 IRE
([pixflow scope guide](https://pixflow.net/blog/read-video-scopes-fast/)). Even ACES *linear* must
pass through the Output Transform to be viewable — verbatim "ACES data (linear or log) is not directly
viewable" ([Wikipedia — ACES](https://en.wikipedia.org/wiki/Academy_Color_Encoding_System)).

**Correct order:** normalize first (IDT / Color Space Transform / technical log→709 LUT — film-grip
already has `apply_lut` for `.cube`/`.3dl`), *then* analyze 709 scopes and grade.

### 2.4 Detecting likely-log footage (numpy-computable, advisory)

No pixel-only test cleanly separates "needs de-log" from "intentional flat look" or "foggy morning",
so this MUST be an **advisory** that names which signals tripped — never a hard claim, never an
auto-transform. Compute on a downscaled frame over 0–1 floats; prefer percentiles over min/max.

Five signals (thresholds are reasoned heuristics — calibrate on a corpus):

| Signal | Metric | Fire when (8-bit / 0–1) | Rationale |
|--------|--------|--------------------------|-----------|
| Low contrast | `p99 − p1` of luma | `< 140` / `< 0.55` | graded 709 spans ~170–219 codes; log ~60–110 |
| Lifted blacks | `p1` of luma | `0.09 < p1 < 0.20` AND min never ≈0 | log floors at 24–32 |
| Depressed highlights | `p99` of luma | `< 199` / `< 0.78` | log diffuse white ≈123–150 |
| Low saturation | `mean((max−min)/255)` over RGB | `< 0.10` (strong `< 0.05`) | flat curve + wide gamut desaturates |
| Midtone bunching | `std(Y)` or frac in 40–180 | `std < 0.18`; or `>0.90` in band | mass concentrated mid |

Score = count of signals firing; **≥3 → advisory**. Most discriminating pair: **low contrast +
lifted blacks together** (graded 709 deliberately puts shadows near 0; log never does). Aggregate
the score across several sampled frames (median) so a single dark scene or title card doesn't mislead.

**False positives** (why advisory only): fog/haze, overcast/flat light, snow/aerial, low-key night,
intentional low-contrast/teal-orange/faded-film looks, underexposure, diffusion filtration — all land
in the same feature region ([studio-supplies.com](https://studio-supplies.com/blogs/guides/what-is-log-footage-guide)).

**Far more reliable than pixels: metadata + filename.** This is what NLEs do — Adobe Premiere's auto
log detection keys on camera-format header metadata and *loses detection when files are transcoded to
ProRes* even though pixels are unchanged
([Adobe helpx](https://helpx.adobe.com/premiere-pro/using/auto-detection-of-log-camera-formats-and-raw-media.html)).
Check filename tokens (`slog`,`s-log`,`slog2/3`,`vlog`,`v-log`,`logc`,`clog`,`c-log`,`nlog`,`flog`,
`redlog`,`dlog`,`arri`) and ffprobe `color_transfer`/`color_space` tags first; fall back to the pixel
advisory only when metadata is absent, and **never override explicit metadata with the pixel heuristic.**
You cannot reliably tell *which* log curve from pixels alone (anchors overlap; the problem is formally
ill-posed — [Rodrigues & Bernardino, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Rodrigues_Single-Image_Estimation_of_2015_CVPR_paper.pdf)).

**Adjacent ffmpeg primitives** worth reusing: `signalstats` emits per-frame `YLOW`(p10),`YHIGH`(p90),
`YAVG`,`SATAVG` ([FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)); `colordetect` (present in
this build) detects color-range/transfer properties. These can cross-check the numpy read.

---

## 3. Skin-tone + exposure

### 3.1 Skin-tone evaluation

* Compute dominant hue of skin pixels and report **angular distance to the I-line (123°)**. Already
  present as `skin_tone_delta_deg` — just fix the reference angle (115→123).
* Improvement: instead of the whole-frame mean (which mixes hues), isolate **skin-likely pixels**
  before measuring. A cheap, deterministic YCbCr skin gate (no model):
  `133 ≤ Cr ≤ 173 and 77 ≤ Cb ≤ 127` on legal 8-bit Cb/Cr (classic Cb/Cr skin box,
  [USPTO 7,426,296 — skin detection in YCbCr](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7426296)).
  Report `skin_pixel_frac` and only compute `skin_tone_delta_deg` over that mask when it is non-trivial
  (e.g. ≥1% of frame); otherwise mark it `null`/advisory. This makes the skin read meaningful on real
  shots and honest (no skin → no skin verdict).
* Tolerance for "on the line": colorists treat **within ~±10°** of 123° as acceptable skin balance
  (consistent with the ±5° scope-target tolerance plus natural skin spread).

### 3.2 Exposure metrics

* **Clipping %**: fraction at/above white limit (full: ≥253; legal: ≥235) and **crushing %** at/below
  black limit (full: ≤2; legal: ≤16) — per channel *and* on luma (per-channel catches a single blown
  channel that luma hides).
* **Luma zones** (Ansel-Adams-style): bucket luma into ~5–10 zones; report the modal zone + the
  shadow/mid/highlight mass split. More actionable than a flat 8-bin histogram for an "under/ok/over"
  call.
* **False-color mapping**: the colorist's exposure overlay — map luma/IRE ranges to flag colors so a
  VLM can *see* exposure. A common Resolve-like scheme:

  | IRE band | meaning | flag color |
  |----------|---------|------------|
  | 0–2.5 | crushed black | purple |
  | 2.5–10 | near-black | blue |
  | ~38–42 | **18% mid-gray** | green |
  | ~50–56 | **skin (Caucasian) sweet spot** | pink |
  | 90–100 | near-clip | yellow |
  | 100+ | clipped | red |

  (Bands vary by vendor; document film-grip's as its own scheme.) Implementation is a numpy LUT over
  IRE → RGB, rendered as a PNG companion alongside `render_scopes_png`. Deterministic and honest.
* Replace the magic exposure verdict (`median<60→under`, `>195→over` on full-range codes) with
  IRE-based, range-aware thresholds (e.g. under if median IRE < ~25, over if > ~75), and additionally
  surface "highlights clipped" / "shadows crushed" from the clip/crush fractions rather than from the
  median alone.

---

## 4. White-balance estimation → suggested CDL

Goal: from a frame, estimate the illuminant cast and emit a **CDL offset** (or per-channel slope) that
neutralizes it, as an *advisory suggestion* the agent can apply via the existing CDL path.

**Gray-world** — assume the scene averages to neutral gray
([mathworks — AWB comparison](https://www.mathworks.com/help/images/comparison-of-auto-white-balance-algorithms.html)):

```
mean_r, mean_g, mean_b = channel means
gray = (mean_r+mean_g+mean_b)/3          # or mean_g as the anchor
gain_r, gain_g, gain_b = gray/mean_r, gray/mean_g, gray/mean_b
```

**White-patch / Max-RGB** — assume the brightest patch is white; scale each channel by its max
(use a high percentile, e.g. p99, to dodge specular/noise):

```
gain_c = max_all / p99_c        for c in r,g,b
```

**Shades-of-Gray (Minkowski-p)** — generalizes both; p=1 → gray-world, p=∞ → white-patch.
**p≈6 is the recommended robust default** ([Finlayson & Trezzi, "Shades of Gray and Colour
Constancy"](https://www.atlantis-press.com/article/25839570.pdf)):

```
norm_c = ( mean(channel_c ** p) ) ** (1/p)     # p = 6
gain_c = mean(norm_r,norm_g,norm_b) / norm_c
```

Gray-world fails on scenes with a dominant real color (faces, sky, foliage)
([mathworks](https://www.mathworks.com/help/images/comparison-of-auto-white-balance-algorithms.html));
Shades-of-Gray (p=6) is more robust — make it the default, expose `method=`.

**Map gains → CDL.** WB gains are *multiplicative* → CDL **slope** is the natural target
(`slope = (gain_r, gain_g, gain_b)` normalized so green = 1.0, i.e. correct R and B relative to G).
If the agent prefers an additive lift instead (the brief says "offset correction"), convert the
implied neutral shift at mid-gray to an offset: `offset_c = (gain_c − 1)·gray01`. Recommend **slope**
(physically correct for illuminant gain) and document the offset alternative. Emit it as a
**suggested** `CDL` with `color_space` hinted, never auto-applied — consistent with film-grip's
honesty stance (it IS scriptable via CDL, so this is a tier-1 *applicable* suggestion, unlike
qualifiers/windows).

Sanity (gray-world fixture): a frame whose channel means are R>G>B (warm cast) must yield
`gain_r < 1 < gain_b` (pull red down, push blue up) → cooling the image. Use this as the WB unit test.

---

## 5. Validating scope accuracy (fixtures)

The load-bearing deliverable: generate a **known** signal with ffmpeg, run it through `analyze_rgb`,
assert measured scope values land at the known targets within tolerance. All commands verified on this
machine.

### 5.1 Solid color patches (best for vectorscope angle assertions)

Full-frame SMPTE bars average to a meaningless mean hue (confirmed: 247°), so assert per *patch*.
Build patches in-memory (no ffmpeg needed) — these are the **canonical expected angles** (measured):

```python
PATCH_TARGETS = {  # 75% bars, scopes.py atan2(Cr,Cb) convention, ±5° tolerance
    "red75":     (191,0,0,   102.91),
    "yellow75":  (191,191,0, 174.77),
    "green75":   (0,191,0,   229.68),
    "cyan75":    (0,191,191, 282.91),
    "blue75":    (0,0,191,   354.77),
    "magenta75": (191,0,191,  49.68),
}
# assert abs(angle_diff(analyze_rgb(patch).vectorscope.hue_deg, target)) <= 5
```

### 5.2 ffmpeg SMPTE bars / ramp (end-to-end through `frame_rgb`)

```bash
# SMPTE HD bars (RP 219) — sample individual bar regions, not the whole frame
ffmpeg -y -v error -f lavfi -i smptehdbars=size=1280x720 -frames:v 1 bars_hd.png
# Legacy SMPTE bars (EG 1-1990)
ffmpeg -y -v error -f lavfi -i smptebars=size=720x576 -frames:v 1 bars_sd.png
# Luma ramp 0..255 for parade/IRE linearity
ffmpeg -y -v error -f lavfi -i "gradients=s=256x64:c0=black:c1=white:x0=0:x1=256:y0=0:y1=0" \
       -frames:v 1 ramp.png            # or build with numpy.linspace (deterministic, no ffmpeg)
# Solid mid-gray (range/neutral check) — ffmpeg "gray" = full-range code 128
ffmpeg -y -v error -f lavfi -i "color=c=gray:s=64x64" -frames:v 1 gray.png
```

([smptehdbars/smptebars docs](https://ffmpeg-graph.site/filters/smptebars/);
[SMPTE color bars — Wikipedia](https://en.wikipedia.org/wiki/SMPTE_color_bars).)

### 5.3 Measured expected values (assert within tolerance)

Captured on this machine (numpy 2.4.6 + ffmpeg) — use as fixture expectations:

* **Gray (color=gray, full-range):** every parade percentile = 128; luma min=median=max=128; cast
  `neutral`; vectorscope saturation = 0.0; exposure `ok`.
* **0..255 luma ramp:** luma min/median/max = 0 / 127.5 / 255; parade p1≈2, p50≈127.5, p99≈253;
  histogram ≈ flat.
* **Per-patch vectorscope angles:** as in §5.1 (red 102.91°, etc.), saturation 0.376–0.446 at 75%.
* **White frame (255,255,255):** clip flag true, crush false, exposure `over` (existing test).
* **Black frame:** crush true, clip false, exposure `under` (existing test).

Tolerances: angles **±5°**, luma/parade levels **±3 codes** (ffmpeg scaling/rounding), saturation
relative. These mirror real scope target tolerances and make the suite robust to ffmpeg version drift.

### 5.4 White-balance fixture

```python
# warm-cast frame: R>G>B  -> estimator must cool it
warm = solid(180,120,90)
cdl  = estimate_white_balance(warm)        # gray-world or shades_of_gray
assert cdl.slope[0] < 1.0 < cdl.slope[2]   # pull R down, push B up
# neutral frame -> near-identity (slopes ~1.0 within tol)
```

---

## 6. Proposed film-grip changes

### 6.1 `scopes.py` fixes (diffs against reality)

1. **Skin-tone angle bug.** Change `SKIN_TONE_ANGLE_DEG = 115.0` → `123.0`. Add a comment documenting
   the `atan2(Cr,Cb)` convention and that red≈103°, I-line≈123°. *(Confirmed wrong by measurement;
   this is the one clear numeric bug.)*
2. **Fix the misleading docstring**, not the math: the chroma coefficients are correctly Rec.709
   Pb/Pr — keep them; update the comment so a future reader doesn't "fix" them to 601.
3. **Range model.** Add a `range: Literal["full","legal"] = "full"` (or auto-detect from ffprobe in
   `frame_rgb`/`analyze_frame`) and make clip/crush thresholds + a new `ire` block derive from it:
   * full: crush ≤2, clip ≥253; `IRE = code/255*100`.
   * legal: crush ≤16, clip ≥235; `IRE = (code-16)/219*100`.
   Emit `luma.ire_median`, `parade.<c>.ire_p1/p99`, and keep code values too. *(Today's hardcoded
   16/235-on-full-range is neither data-limit nor IRE-correct.)*
4. **Per-channel clip/crush** in the exposure block (catch a single blown channel).
5. **Record assumed transfer** (`"transfer": "bt709-2.4"` default, overridable) in the report so
   downstream gamma reasoning is explicit (§2.1).
6. **Skin masking** (§3.1): add `skin_pixel_frac` and compute `skin_tone_delta_deg` only over the
   YCbCr skin box when that fraction is non-trivial; else report it as advisory/null.

### 6.2 New readers (numpy-deterministic)

```python
def detect_log_footage(arr, *, frames=None) -> dict:
    """Advisory: does this look like flat/log footage that should be normalized before 709 grading?
    Returns {likely_log: bool, score: int(0..5), signals: {low_contrast, lifted_blacks,
    depressed_highlights, low_saturation, midtone_bunching}, advice: str}. NEVER auto-transforms.
    Honesty tier: ADVISORY (cannot be proven from pixels; metadata/filename more reliable)."""

def estimate_white_balance(arr, *, method="shades_of_gray", p=6) -> "CDL":
    """Estimate illuminant cast (gray_world | white_patch | shades_of_gray) and return a SUGGESTED
    CDL (slope-based) that neutralizes it, green anchored to 1.0. Honesty tier: APPLICABLE (CDL is
    scriptable) but emitted as a suggestion, not auto-applied."""

def exposure_report(arr, *, range="full") -> dict:
    """Clip/crush % (per-channel + luma), luma zones, IRE median, under/ok/over by IRE thresholds."""

def render_false_color_png(media_path, at_s, out_png) -> str:
    """IRE->flag-color overlay PNG (deterministic numpy LUT) — exposure as a VLM can see it."""
```

Also extend `analyze_rgb` to fold `detect_log_footage` signals + IRE into the main report (cheap), so
a single read warns the agent before it grades log as 709.

### 6.3 Verify + fixtures

Add `tests/test_scope_accuracy.py`:
* per-patch vectorscope angle assertions (§5.1, ±5°);
* ffmpeg bars/ramp/gray through `frame_rgb` → expected levels (§5.3, guarded by `ffmpeg_path()`);
* IRE mapping unit test (codes 16/128/235 → 0/51.14/100);
* `estimate_white_balance` gray-world fixture (§5.4): warm in → cooling CDL out; neutral → identity.

Wire a "scope accuracy vs reference" check so CI proves the synthesized scopes match known SMPTE
targets — this is the credibility anchor for the whole perception moat.

### 6.4 Honesty tiers (unchanged stance, made explicit in outputs)

* **Tier 1 — applicable/scriptable:** CDL (slope/offset/power/sat), `apply_lut` (.cube/.3dl). WB
  suggestion and log-normalization-via-LUT live here (a fix CAN be applied), but are surfaced as
  *suggestions* requiring confirmation.
* **Tier 2 — advisory only:** `detect_log_footage` (cannot be proven from pixels), skin/exposure
  *reads*, and anything implying primary wheels-by-value, curves, qualifiers, Power Windows, Magic
  Mask — GUI-only in every NLE, surfaced as advice, never faked as applied.

Every new reader must state its tier in its return/docstring so the agent (and user) never mistakes an
advisory read for an applied change.

---

## Sources

- ITU-R BT.709 / luma / legal range — https://en.wikipedia.org/wiki/Rec._709
- YCbCr matrices (601 & 709 Pb/Pr) — https://en.wikipedia.org/wiki/YCbCr
- Vectorscope angle derivation (red=104°, I=123°) — https://chriskiehl.com/article/what-is-red-doing-over-there
- Vectorscope / skin line (116–126°, 123° +I) — https://bramstout.nl/en/webbooks/vectorscopes/
- Scope target tolerance ±5° / 75 vs 100 bars — https://www.tek.com/en/support/faqs/why-are-my-color-bars-not-hitting-vectorscope-targets
- Skin correction on the line — https://larryjordan.com/articles/color-correction-make-people-look-normal/
- Full vs legal levels — https://www.thepostprocess.com/2019/09/24/how-to-deal-with-levels-full-vs-video/
- sRGB transfer function — https://en.wikipedia.org/wiki/SRGB
- Gamma 2.2 vs 2.4 (effective gamma) — https://en.wikipedia.org/wiki/Gamma_correction
- BT.1886 display EOTF — https://en.wikipedia.org/wiki/ITU-R_BT.1886
- System gamma 1.2 / OOTF — https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2390-10-2021-PDF-E.pdf
- 2.2 vs 2.4 in Resolve — https://mixinglight.com/color-grading-tutorials/gamma-2-2-vs-gamma-2-4-davinci-resolve/
- sRGB == Rec709 gamut — https://lightillusion.com/colourspace_manual.html
- ARRI Log C (flat/desat; encoding) — https://www.arri.com/en/learn-help/learn-help-camera-system/image-science/log-c
- ARRI LogC3 VFX data PDF — https://www.arri.com/resource/blob/31918/66f56e6abb6e5b6553929edf9aa7483e/2017-03-alexa-logc-curve-in-vfx-data.pdf
- ARRI LogC4 spec PDF — https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf
- Sony S-Log3 tech summary PDF — https://pro.sony/s3/cms-static-content/uploadfile/06/1237494271406.pdf
- Sony S-Log exposure — https://www.wolfcrow.com/how-to-expose-s-log3/
- Canon Log white paper PDF — https://www.usa.canon.com/content/dam/canon-assets/white-papers/pro/white-paper-canon-log-gamma-curves.pdf
- RED Log3G10 white paper PDF — https://docs.red.com/955-0187/PDF/915-0187%20Rev-C%20%20%20RED%20OPS,%20White%20Paper%20on%20REDWideGamutRGB%20and%20Log3G10.pdf
- Panasonic V-Log/V-Gamut PDF — https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/VARICAM_V-Log_V-Gamut.pdf
- ACES overview — https://en.wikipedia.org/wiki/Academy_Color_Encoding_System
- ACES docs (ACEScct toe, output transforms) — https://docs.acescentral.com/
- Grading log as 709 is wrong — https://filmit.io/blog/log-vs-rec709-lut-mistake/
- Scopes expand only after transform — https://pixflow.net/blog/read-video-scopes-fast/
- Log footage characteristics — https://studio-supplies.com/blogs/guides/what-is-log-footage-guide
- Premiere auto log detection (metadata, lost on transcode) — https://helpx.adobe.com/premiere-pro/using/auto-detection-of-log-camera-formats-and-raw-media.html
- Single-image CRF estimation ill-posed — https://openaccess.thecvf.com/content_cvpr_2015/papers/Rodrigues_Single-Image_Estimation_of_2015_CVPR_paper.pdf
- Gray-world / white-patch (AWB comparison) — https://www.mathworks.com/help/images/comparison-of-auto-white-balance-algorithms.html
- Shades of Gray (Minkowski p, robustness) — https://www.atlantis-press.com/article/25839570.pdf
- YCbCr skin detection box — https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7426296
- ffmpeg signalstats / sources — https://ffmpeg.org/ffmpeg-filters.html
- ffmpeg smptebars/smptehdbars — https://ffmpeg-graph.site/filters/smptebars/
- SMPTE color bars — https://en.wikipedia.org/wiki/SMPTE_color_bars
