# film-grip — agentic color grading

film-grip grades video the same way it cuts it: **one portable pivot + typed reversible primitives
+ perception + a verify loop + thin honest adapters + a capability gate**. Where the cut side
pivots on OpenTimelineIO, the color side pivots on **ASC CDL** — the vendor-neutral color decision
that travels everywhere.

The design follows the shape an agentic colorist actually needs (and that the LumiVideo research
system, [arXiv 2604.02409](https://arxiv.org/abs/2604.02409), validates):
**perceive → propose → apply → verify → iterate**, compiling to interpretable **CDL + LUT** rather
than generated pixels.

## The color model

### CDL is the pivot (the OTIO of color)

An [ASC CDL](https://en.wikipedia.org/wiki/ASC_CDL) is ten numbers — **S**lope, **O**ffset,
**P**ower per RGB channel (SOP) plus one **Sat**uration scalar:

```
per channel:  out = ( max(0, in*slope + offset) ) ** power      # clamp before the power
saturation:   luma  = 0.2126*R + 0.7152*G + 0.0722*B            # Rec.709
              out_c = luma + sat*(c - luma)
identity:     slope=[1,1,1] offset=[0,0,0] power=[1,1,1] sat=1
```

It carries the *decision*, not an editor's node graph, so the same grade lands live in DaVinci
Resolve **and** rides through interchange where node graphs, qualifiers and Power Windows cannot.
CDL declares no working color space, so film-grip records an honest `color_space` hint alongside it
rather than silently assuming one.

### Ergonomics → portable wire

Agents (and colorists) think in **lift / gamma / gain / contrast / temp / tint**.
`filmgrip.color.lgg_to_cdl(...)` compiles those deterministically to canonical CDL SOP+sat
(`gain→slope`, `lift→offset`, `gamma→power=1/gamma`, contrast about a pivot, temp/tint as channel
nudges), so the agent reasons naturally while the stored/applied primitive stays portable and
validated. It's an honest *approximation* of the wheels in CDL math — never confused with Resolve's
actual primary wheels.

### LUT is the look

`apply_lut` references a `.cube` (1D/3D) or `.3dl` for looks primaries can't express (film
emulation, camera→display transforms). film-grip parses + sanity-checks the file (size keyword, row
count, red-fastest order) so a malformed or hallucinated path is rejected before apply; a bare name
is allowed-but-warned (the editor resolves it against its LUT folder). LUTs are files — bare
references don't travel; **ship the `.cube` with the project or bake it**.

## The color ops

| Op | What it does | Resolve (live) | Interchange |
|---|---|---|---|
| `set_cdl` | ASC CDL primary grade (SOP + sat, per node) | `TimelineItem.SetCDL` | OTIO metadata + `.cc`/`.ccc`/`.cdl` + FCPXML |
| `apply_lut` | attach a `.cube`/`.3dl` LUT to a node | `SetLUT` (Graph v19+ / item) | OTIO metadata (ship the file) |
| `color_group` | assign clips to a shared color group | `AddColorGroup`/`AssignToColorGroup` | — (Resolve-live only) |
| `color_version` | add/load alternate grade versions | `AddVersion`/`LoadVersionByName` | — (Resolve-live only) |
| `apply_grade` | copy a hero grade / apply a `.drx` PowerGrade | `CopyGrades`/`ApplyGradeFromDRX` | — (Resolve-live only) |

Every op references **stable clip IDs** and is re-validated host-side, so a hallucinated id, frame,
LUT path or `.drx` cannot survive into an apply.

## Perception — the moat

Resolve exposes **no scripting API for its scopes**, so film-grip computes them itself from frame
pixels (`filmgrip.perception.scopes`, `film-grip scopes`):

- **RGB parade** — per-channel percentiles (1/10/50/90/99): black/shadow/mid/highlight/white points.
- **Luma waveform** — min/median/max + clipped(≥235)/crushed(≤16) fractions and flags.
- **Vectorscope** — dominant hue angle + saturation magnitude + distance from the skin-tone line.
- **White balance** — gray-point neutrality → a warm/cool/green/magenta cast read.
- **Exposure** — mid-grey placement → under/ok/over.

`analyze_rgb` is pure numpy (frame in → numbers out), so its correctness is provable on synthetic
frames — no editor, no ffmpeg. `render_scopes_png` produces a human-readable scope image via
ffmpeg's own scope filters.

```bash
film-grip scopes --media shot.mov --at 1.5            # JSON reading for one frame
film-grip scopes --fixture timeline.otio --select c01 # per-clip readings
film-grip scopes --media shot.mov --png scopes.png    # + a visual scope image
```

## The verify loop

`apply_cdl_array` applies a CDL to a frame with the *same math* as `CDL.apply`, so film-grip can
**predict** a grade's scopes without asking the editor to render (LumiVideo's "analytically
guaranteed" effect). `verify.grade_delta` diffs two readings; `verify.verify_grade` judges a
graded/predicted reading against a target (the hero shot, a look) within tolerance on luma, hue,
saturation and white balance. That closes the loop offline:

```
perceive (scopes) → propose (set_cdl) → predict (apply_cdl_array) → verify (vs target) → iterate
```

This verify step is what separates film-grip from consumer "text→LUT" generators, which emit a
`.cube` and stop.

## Grade packs (look presets)

Deterministic look packs compile (via `lgg_to_cdl`) to a portable `set_cdl` per selected clip —
zero LLM, always applicable:

- `teal-orange` · `filmstock-warm` · `bleach-bypass` · `day-for-night`

Two prompt packs drive the planner off the scopes: `neutral-balance` (neutralize cast + set
exposure) and `grade-match` (match selected clips to a reference). These are CDL **primary
approximations** of looks; a true split-tone look wants a 3D LUT.

```bash
film-grip pack apply teal-orange --fixture timeline.otio
```

## Interchange — a grade goes everywhere

`filmgrip.serialize.cdl` makes a grade portable across the whole adapter family:

- `.cc` / `.ccc` / `.cdl` — ASC CDL XML (`urn:ASC:CDL:v1.2`), bit-exact round-trip.
- EDL — `*ASC_SOP` / `*ASC_SAT` comment lines (Avid/Resolve conform).
- FCPXML — an `info-asc-cdl` element. **Honest caveat:** FCPXML carries CDL as *inert passthrough
  metadata* — Final Cut does not apply it on import; it rides for the next tool that does.

`grades_from_ir` collects a whole timeline's `set_cdl` grades into one `.ccc`/`.cdl`.

## Honesty — what is NOT scriptable

Resolve's color scripting API reaches **only** CDL + LUT + groups/versions + DRX/copy. Everything
richer is **GUI-only in every NLE** and is surfaced as an *advisory* step, never faked as an applied
edit (the color analog of film-grip's audio-levels honesty):

> primary wheels by value · log wheels · custom curves · HSL qualifiers / secondaries ·
> Power Windows / masks · Magic Mask · Auto Color / Color Match / Shot Match (AI) · Color Warper ·
> HDR palette

If asked for one of these, film-grip does the closest CDL/LUT grade and tells you the rest is a
manual step. Run `film-grip editors` for the live per-op capability matrix.
