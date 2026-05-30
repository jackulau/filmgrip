# The in-Resolve panel — how it works, and the honest trade-offs

The goal: a panel **inside DaVinci Resolve** where you see your selection, type a natural-language
instruction, and hit Apply — without leaving the editor. This note explains the mechanism film-grip
chose, the alternatives, and the constraints we don't hide.

## What ships

`film-grip panel install` copies `scripts/film_grip_resolve_panel.py` into Resolve's Edit-page
Scripts folder (`…/DaVinci Resolve/Fusion/Scripts/Edit/film-grip.py`) and bakes in the path to the
installed `filmgrip` package so the out-of-repo copy can import it. Launch it from
**Workspace ▸ Scripts ▸ film-grip**.

The panel is a floating window built with Resolve's Fusion **UIManager**:

```
┌ film-grip ─────────────────────────────┐
│ 3 clip(s) selected [reconstructed]      │
│ Describe the edit:                      │
│ ┌─────────────────────────────────────┐ │
│ │ add a whoosh as the title flies in… │ │
│ └─────────────────────────────────────┘ │
│ [ Apply ]  [ Dry-run ]                  │
│ ✓ marker Blue on intro …                │
└─────────────────────────────────────────┘
```

## How it stays testable

The script is a **thin translator**. All structure and behaviour live in the unit-tested
`filmgrip/ui/panel.py`:

- `build_panel_spec()` returns a declarative widget tree (`PanelSpec`) with stable ids. The script
  walks it to build native UIManager widgets; the tests walk the same tree to assert the panel's
  shape. One source of truth, no divergence.
- `PanelController` exposes the only behaviour the panel needs — `on_apply` / `on_dry_run` over a
  single injectable `run_edit(prompt, dry_run)` seam. Tests drive Apply/Dry-run/empty-prompt/error
  with a fake; `live_controller()` wires the seam to the real snapshot → plan → apply pipeline.

So everything except ~120 lines of UIManager glue is covered with **zero Resolve, zero SDK, zero
network**. The glue itself byte-compiles in CI (the Resolve globals are looked up lazily and guarded).

## Why a floating UIManager window — and not the alternatives

| Mechanism | In-app? | Extra deps | Testable | Verdict |
|---|---|---|---|---|
| **Scripts-menu UIManager window** (chosen) | yes (Workspace ▸ Scripts) | none | yes (spec + controller) | best fit |
| PyQt/Tk companion window | no (floats outside Resolve) | a GUI toolkit | partial | loses the "in-editor" feel |
| Electron + HTTP bridge | feels docked | Electron + a server in Resolve | heavy | overkill for this |
| Docked Edit-page panel | — | — | — | **not possible** (see below) |

## The honest constraints

- **No true docked panel.** Resolve's scripting model invokes a script, runs it, and exits; an
  Edit-page script has no Composition context (`GetCurrentComp()` is `None`), and `NewFloatFrame`
  needs one. A persistent docked panel would need an unsupported background thread or a separate
  process. The floating UIManager window launched from the Scripts menu is the honest in-app surface.
- **UIManager is undocumented + version-fragile.** It isn't in the public Scripting README; its API
  can shift between Resolve versions. Keeping the script thin (all logic in the tested core) means a
  UIManager break is a small, contained fix.
- **Live display needs Resolve open.** The panel's *behaviour* is fully tested headless, but actually
  rendering the window requires Resolve (Studio, external scripting enabled). The script reports
  clear guidance when UIManager isn't available instead of failing opaquely.
- **Selection is reconstructed.** Resolve exposes no multi-clip timeline selection, so the panel
  shows `[reconstructed]` next to the count — it's the current clip + media-pool selection, surfaced
  honestly, not a fake multi-select.

## Future paths

A Premiere **UXP** panel is the natural next host (declared `uxp-future` in the capability matrix).
An Electron companion with an HTTP bridge would give a richer UI if docking ever becomes a hard
requirement — at the cost of the dependencies above.
