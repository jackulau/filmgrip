# Perception — how the agent hears and sees your footage

Timeline metadata (FGX rows) tells the planner *where* clips are. Perception tells it *what's
in them* — without ever rendering, exporting, or modifying your media. Everything here is
**pull-on-demand**: it costs tokens only when an edit actually needs content awareness, and it
degrades to an actionable error (never a fabricated answer) when a dependency is missing.

The design follows what the strongest pipeline tools proved in production (browser-use's
video-use, the Fable launch-video workflow), adapted to film-grip's non-destructive, in-NLE
model: **word timestamps are the load-bearing data — the agent greps the transcript, it never
scrubs the video.**

## Layer 1 — transcripts (`film-grip transcribe`, `get_transcript`)

Word-level ASR with timestamps, cached per media file, packed into a token-frugal phrase
format:

```
## c12 v1 interview.mov [240-720]
[0264-0290] S0 Hey everyone welcome back
[0312-0395] S0 today we're talking about perception
```

Those are **timeline frames** — the same coordinates every EditPlan op uses, so "cut after
'perception'" resolves to an exact `split`/`cut_range` frame. The mapping goes word
media-seconds → source offset → timeline frame per clip, and **refuses retimed clips honestly**
(a time-warp breaks the linear mapping) instead of emitting wrong frames.

Backends, auto-detected in this order (override with `FILMGRIP_ASR_BACKEND` or `--asr`):

| Backend | Type | Needs | Extras |
|---|---|---|---|
| `faster-whisper` | local Python | `pip install 'film-grip[transcribe]'` | language detection |
| `whisper-cpp` | local CLI | `brew install whisper-cpp` + `FILMGRIP_WHISPER_CPP_MODEL` | fastest on Apple Silicon |
| `elevenlabs` | API | `ELEVENLABS_API_KEY` | speaker diarization + audio events |

Transcripts cache as JSON sidecars under `~/.filmgrip/transcripts/` keyed by file identity
(path + size + mtime + backend) — re-running on untouched footage is free; a re-export
re-transcribes. `--srt out.srt` writes captions (media-time for `--media`, timeline-time for a
timeline — importable into the NLE).

```bash
film-grip transcribe                          # live Resolve selection, timeline frames
film-grip transcribe --fixture cut.otio       # any fixture timeline
film-grip transcribe --media interview.mov    # one file, media seconds
film-grip transcribe --srt captions.srt       # caption export
```

**ASR honesty rule:** word timestamps drift 50–100ms. Every consumer in film-grip pads cuts
30–200ms and never cuts inside a word — the planner is told the same rule.

## Layer 2 — speech analysis (`analyze_speech`, the `silence-cut` pack)

Deterministic (zero LLM calls): from word timestamps, film-grip computes per-clip silences
(head / interior / tail, default ≥400ms), filler-word occurrences (um/uh/erm…, conservative
vocabulary), and **ready-to-validate `cut_range` candidates** — word-snapped, drift-padded,
ordered last-to-first as the validator requires.

```bash
film-grip pack apply silence-cut --fixture cut.otio --verify
```

Defaults: `min_silence: 0.4s`, `fillers: true` — override them by saving a user pack with your
own params in `~/.filmgrip/packs` (user packs shadow built-ins by name).

`silence-cut` is a deterministic pack: transcripts in, typed ops out, same validate→apply
pipeline as everything else. No backend (or offline media) → `PackError` with the fix, never a
partial silent edit.

## Layer 3 — looking at footage (`film-grip frames`, `view_frames`)

The agent can't scrub; it can read one composite PNG. `frames` renders a contact sheet from
the **source media** — a filmstrip row (evenly sampled across a clip, or exact timeline
frames) over a waveform strip — plus a machine-readable legend mapping each tile to its
timeline frame and media seconds:

```bash
film-grip frames --fixture cut.otio --count 8          # one sheet per clip
film-grip frames --at 100,432,672                      # exact timeline frames (live Resolve)
```

Requires `ffmpeg` and the source media on disk; a video-only file gets its sheet with a
"waveform omitted" note. Media-missing / retimed / gap frames come back as per-item errors.

## Layer 4 — the verification loop (`--verify`, `perception/verify.py`)

After an apply, film-grip can **prove** the editor ended up with what the plan meant:

1. **Simulate** the expected timeline by running the same OTIO mutator on a copy of the
   pre-edit graph (exact for every rebuild-path op; `import_audio`/`add_track` are reported as
   *not structurally verifiable* rather than assumed).
2. **Diff** the simulation against a fresh post-apply snapshot — positions, durations, source
   offsets, enabled flags, retime effects. Any divergence is a named mismatch
   (`v1 item 2: expected clip @100 dur 332, got dur 360`) and the CLI exits 1.
3. **Look** at every new cut boundary: contact sheets ±1.5s around each fresh edge, so the
   agent (or you) can see the seam it just created.

```bash
film-grip edit "remove the silences" --verify
film-grip pack apply silence-cut --fixture cut.otio --verify
```

This is the non-destructive equivalent of the render→re-watch self-eval loop pipeline tools
use: same evidence (geometry + boundary imagery), but no generation-loss bake and your project
file stays the source of truth. The in-Resolve panel runs the structural check on every apply.

## Rationale-bearing edits (v3 protocol)

Every op accepts optional `reason` and `quote` fields — *why* this edit exists and the spoken
words it anchors to. They surface in dry-run diffs (`cut interview [0046..0057) 11f (ripple)
"um" — remove the um`) and persist into OTIO clip metadata on apply, so an edit session reads
like a reviewed change, not an opaque mutation.

## MCP tools (what the planner can pull)

| Tool | Returns | Cost profile |
|---|---|---|
| `get_selection` / `get_context` / `query_clips` | FGX structure | tens–hundreds of tokens |
| `get_transcript(ids)` | packed phrases in timeline frames | ~1/10 of word-JSON |
| `analyze_speech(ids)` | silences + fillers + `cut_range` candidates | small, deterministic |
| `view_frames(ids \| frames)` | contact-sheet PNG path + legend | one image read |

Perception is never auto-bundled into the grab context — the planner pulls exactly what the
instruction needs.
