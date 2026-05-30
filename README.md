# film-grip

**react-grab, but for video editors.** Select clips/resources inside your editor, add them to
context with a natural-language prompt, and Claude produces a *typed, validated* edit that
film-grip applies through the right per-editor adapter.

```
select clips in editor ──► film-grip captures a compact context bundle ──► you type a prompt
        ──► Claude returns a typed EditPlan (ops over stable clip IDs) ──► film-grip validates
        ──► applies via the editor's adapter (live, or interchange round-trip)
```

## Why it works across many editors

There is no DOM in a desktop NLE, so there is no single hook like react-grab's React-fiber
patch. film-grip's equivalent is **one universal IR + one typed protocol + thin adapters**:

- **Core IR** — every timeline is normalized to an [OpenTimelineIO](https://opentimelineio.readthedocs.io/)
  graph. OTIO is the lossless pivot; everything else is a projection of it.
- **Typed `EditPlan`** — Claude never emits editor-specific code. It proposes bounded, reversible
  primitives (`trim`/`move`/`insert`/`delete`/`setProperty`/`addMarker`/`addTransition`/`split`/`ripple`)
  that reference **stable clip IDs**. Hallucinated IDs/frames can't survive validation.
- **FGX serializer** — a token-frugal projection (integer frames, abbreviated rows, selected
  subgraph + 1-hop neighbors, media by reference, multi-turn deltas). A selection that is
  multiple kilobytes of raw FCPXML becomes a few hundred tokens.
- **Adapters** — Resolve native (live flagship), an OTIO-interchange family (FCP / Premiere /
  Kdenlive / Shotcut / Avid), CapCut offline-JSON, and a read-only Filmora parser.

## Editor coverage (honest)

| Editor | Role | Mechanism | Write-back |
|---|---|---|---|
| DaVinci Resolve (Studio) | flagship — live | native Python scripting API | yes (live) |
| Final Cut Pro | interchange | FCPXML round-trip | via re-import |
| Premiere Pro | interchange now; UXP panel later | FCP7 XML / AAF (UXP panel is a future path) | via re-import |
| Kdenlive / Shotcut | interchange (near-free) | native MLT-XML | yes (offline) |
| Avid Media Composer | best-effort / conform | AAF | lossy |
| CapCut (International) | best-effort | offline `draft_content.json` | offline only, version-gated |
| Wondershare Filmora | **read-only** | `.wfp` parse | **no** — Filmora has no automation API |

film-grip never promises automation an editor cannot deliver. Resolve has no true multi-clip
timeline-selection API; Filmora has no write-back; interchange formats are lossy by a documented
amount. Each adapter declares its capabilities and the CLI surfaces them.

## Status

Early build. The core IR, protocol, serializer, validator and every non-live adapter are
unit-tested on fixtures **with zero editors installed**. The live Resolve path is demonstrable
with DaVinci Resolve open (Studio, external scripting enabled).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # core + tests
.venv/bin/pip install -e ".[all,dev]"    # + Claude Agent SDK + AAF
.venv/bin/pytest -q
```

## Usage

```bash
# Offline, provable without any editor:
film-grip edit --fixture tests/fixtures/cut.otio --plan tests/fixtures/plan.json --dry-run

# Live, with DaVinci Resolve open:
film-grip edit "add a blue marker on the selected clip"
```

## License

MIT
