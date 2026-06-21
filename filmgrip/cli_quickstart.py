"""``film-grip quickstart`` — a zero-config, offline, no-key 30-second onboarding demo.

This is the "30-second value" + "suggest the next command" recommendation from
``docs/research/cinematography-ux.md`` (Part B.3.2), made real and **honest by construction**:

* It runs entirely **offline** — no editor, no network, no API key, no billable Claude call.
* It exercises the *real* offline pipeline (``--plan`` replay), the same code path
  :mod:`filmgrip.cli_edit` uses: load a bundled timeline → show it → apply a recorded
  :class:`~filmgrip.protocol.editplan.EditPlan` → print what actually changed.
* It only ever demonstrates things that genuinely work with the core install, so the closing
  next-step suggestions can never point a new user at something that fails.

The demo runs on the committed fixtures (``tests/fixtures/cut.otio`` + ``plan.json``). Those live
outside the installed ``filmgrip`` package (the wheel ships ``filmgrip*`` only), so we *discover*
them next to the source checkout; if they can't be found (e.g. a bare wheel with no repo alongside),
we synthesize an equivalent tiny demo in a temp dir rather than crash — zero config, always exit 0.
"""
from __future__ import annotations

import os
import sys
import tempfile

# --------------------------------------------------------------------------- TTY-aware color
#: ANSI codes, only ever emitted when stdout is a real TTY and NO_COLOR is unset (clig.dev).
_CODES = {"bold": "\033[1m", "dim": "\033[2m", "green": "\033[32m",
          "cyan": "\033[36m", "yellow": "\033[33m", "reset": "\033[0m"}


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, *styles: str) -> str:
    if not _use_color():
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}" if prefix else text


def _emit(text: str = "") -> None:
    print(text)


# --------------------------------------------------------------------------- fixture discovery
def _candidate_fixture_dirs() -> list[str]:
    """Where the committed demo fixtures might live, most-specific first.

    Primary location is the repo's ``tests/fixtures`` discovered relative to this source file
    (``filmgrip/cli_quickstart.py`` → ``../tests/fixtures``); the cwd is also checked so the demo
    works when run from a checkout root regardless of where film-grip is installed from.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)  # parent of the filmgrip/ package
    dirs = [
        os.path.join(repo_root, "tests", "fixtures"),
        os.path.join(os.getcwd(), "tests", "fixtures"),
    ]
    seen, out = set(), []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            out.append(d)
    return out


def find_demo_fixtures() -> tuple[str, str] | None:
    """Return ``(cut.otio, plan.json)`` paths from the first dir that has BOTH, else ``None``."""
    for d in _candidate_fixture_dirs():
        cut = os.path.join(d, "cut.otio")
        plan = os.path.join(d, "plan.json")
        if os.path.isfile(cut) and os.path.isfile(plan):
            return cut, plan
    return None


def _synthesize_demo(dest_dir: str) -> tuple[str, str]:
    """Last-resort fallback: build an equivalent tiny timeline + recorded plan in ``dest_dir``.

    Only used when the committed fixtures aren't discoverable (a bare wheel with no repo). It keeps
    ``quickstart`` honest — it still runs the *real* offline apply path, just on a generated stand-in
    — instead of crashing or faking output. Mirrors the shape of ``tests/fixtures``.
    """
    import json

    import opentimelineio as otio

    rate = 24.0

    def rt(f: float):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="Demo Cut")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for nm, dur in (("intro", 48), ("dialogue_a", 72), ("broll_1", 60)):
        v.append(otio.schema.Clip(
            name=nm, media_reference=otio.schema.ExternalReference(target_url=f"/media/{nm}.mov"),
            source_range=otio.opentime.TimeRange(rt(0), rt(dur))))
    tl.tracks.append(v)
    cut = os.path.join(dest_dir, "cut.otio")
    otio.adapters.write_to_file(tl, cut)

    # Resolve the generated clip IDs so the recorded plan targets real clips (same as the committed
    # plan.json: tighten the open, flag the b-roll).
    from .core.ir import TimelineIR

    ir = TimelineIR.from_otio_file(cut)
    by_name = {c.name: c.id for c in ir.real_clips()}
    plan_obj = {
        "notes": "tighten the open and flag the b-roll",
        "ops": [
            {"op": "trim", "clip_id": by_name["intro"], "edge": "out", "delta": -12},
            {"op": "add_marker", "clip_id": by_name["dialogue_a"], "frame": 0,
             "color": "Blue", "name": "review"},
            {"op": "set_property", "clip_id": by_name["broll_1"], "key": "ZoomX", "value": 1.2},
        ],
    }
    plan = os.path.join(dest_dir, "plan.json")
    with open(plan, "w", encoding="utf-8") as fh:
        json.dump(plan_obj, fh, indent=2)
    return cut, plan


# --------------------------------------------------------------------------- timeline view
def _timeline_overview(ir) -> list[str]:
    """A short, human-skimmable view of the loaded timeline (no editor, no API)."""
    from .serialize.fgx import _rate_str, track_code

    reals = ir.real_clips()
    lines = [
        f"  sequence : {ir.timeline.name or 'seq'}   fps {_rate_str(ir.rate)}   "
        f"{len(reals)} clip(s) across {len(ir.timeline.tracks)} track(s)",
    ]
    for c in reals[:6]:
        lines.append(
            f"    [{c.id}] {c.name:<12} {track_code(c.track_kind, c.track_index)}  "
            f"{c.start}–{c.end}f  ({c.duration}f)  src {c.src_ref}")
    if len(reals) > 6:
        lines.append(f"    … +{len(reals) - 6} more clip(s)")
    return lines


def _next_steps() -> list[str]:
    """2–3 concrete next commands. Each one genuinely works with no editor / no key."""
    return [
        (f"  {_c('•', 'cyan')} see exactly what film-grip can apply per editor:\n"
         f"      {_c('film-grip editors', 'bold')}"),
        (f"  {_c('•', 'cyan')} check whether your editor + toolchain are wired up (the doctor):\n"
         f"      {_c('film-grip status', 'bold')}"),
        (f"  {_c('•', 'cyan')} run this offline demo on YOUR OWN timeline (no editor needed):\n"
         f"      {_c('film-grip edit --fixture <your.otio> --dry-run \"tighten the open\"', 'bold')}"),
    ]


# --------------------------------------------------------------------------- the command
def cmd_quickstart(args=None) -> int:
    """Run the offline onboarding demo. Always returns 0 — onboarding never blocks on environment."""
    # Lazy imports keep --version/--help cheap and mean a missing optional dep degrades, not crashes.
    from .cli_edit import (
        FixtureError,
        _adapter_for_fixture,
        _load_recorded_plan,
        load_fixture_ir,
    )
    from .protocol.validate import dry_run, validate

    _emit(_c("film-grip quickstart", "bold") + "  —  a 30-second, fully offline demo")
    _emit(_c("  (no editor, no network, no API key, no billable call — honest by construction)",
             "dim"))
    _emit()

    # 1/3 — locate the bundled demo timeline (or synthesize an equivalent stand-in offline).
    found = find_demo_fixtures()
    tmpdir = None
    if found is not None:
        cut_path, plan_path = found
        source_note = ""
    else:
        tmpdir = tempfile.mkdtemp(prefix="filmgrip-quickstart-")
        try:
            cut_path, plan_path = _synthesize_demo(tmpdir)
        except Exception as exc:  # never let onboarding crash — say so plainly and stop cleanly
            _emit(_c(f"  note: couldn't prepare a demo timeline ({exc}); skipping the apply step.",
                     "yellow"))
            _emit_next_steps_only()
            return 0
        source_note = "  (bundled fixtures not found alongside the install — using a generated stand-in)"

    _emit(_c("1/3", "bold") + "  Bundled demo timeline: "
          + _c(os.path.basename(cut_path), "cyan") + _c("  " + _ok(), "green"))
    if source_note:
        _emit(_c(source_note, "dim"))

    try:
        ir = load_fixture_ir(cut_path)
    except FixtureError as exc:
        # The demo timeline couldn't be read — degrade to next-steps rather than crash.
        _emit(_c(f"  note: couldn't read the demo timeline ({exc}).", "yellow"))
        _emit_next_steps_only()
        _cleanup(tmpdir)
        return 0
    for line in _timeline_overview(ir):
        _emit(line)
    _emit()

    # 2/3 — apply a RECORDED plan offline (the --plan path: no planner, no Claude, deterministic).
    _emit(_c("2/3", "bold") + "  Apply a recorded edit plan offline (replayed via --plan, "
          + _c("no Claude call", "cyan") + "):")
    try:
        plan = _load_recorded_plan(plan_path)
    except Exception as exc:
        _emit(_c(f"  note: couldn't load the demo plan ({exc}); showing next steps.", "yellow"))
        _emit_next_steps_only()
        _cleanup(tmpdir)
        return 0

    if plan.notes:
        _emit(f"  goal: {_c(plan.notes, 'cyan')}")

    # Show the human-readable diff (what WILL change) — the same validator/dry_run the edit path uses.
    valid = validate(plan, ir)
    for line in dry_run(plan, ir).splitlines():
        if line.startswith("#"):
            continue  # the notes line; already shown above as the goal
        _emit("  " + line)

    # Now actually perform the offline apply to a throwaway output so we PROVE the path end-to-end
    # (raw footage / source fixture untouched — we always write a derived path, never overwrite).
    applied_note = ""
    if valid.ok:
        adapter = _adapter_for_fixture(cut_path)
        out_dir = tmpdir or tempfile.mkdtemp(prefix="filmgrip-quickstart-")
        if tmpdir is None:
            tmpdir = out_dir
        out_path = os.path.join(out_dir, "demo.edited" + os.path.splitext(cut_path)[1])
        try:
            res = adapter.apply(plan, cut_path, out_path=out_path)
            if res.ok and os.path.isfile(out_path):
                applied_note = (f"  {_ok()} applied {len(plan.ops)} op(s) — wrote edited timeline to "
                                f"{_c(out_path, 'dim')}")
            else:
                # apply ran but reported issues — be honest about it, still exit 0 (demo, not a job).
                problems = "; ".join(res.errors or res.unsupported) or "no output produced"
                applied_note = _c(f"  note: offline apply reported: {problems}", "yellow")
        except Exception as exc:  # any adapter hiccup degrades to the dry-run we already showed
            applied_note = _c(f"  note: showed the plan only (offline apply skipped: {exc})",
                              "yellow")
    else:
        applied_note = _c("  note: demo plan didn't validate against the timeline "
                          "(showed the errors above).", "yellow")
    _emit(applied_note)
    _emit()

    # 3/3 — concrete, honest next steps.
    _emit(_c("3/3", "bold") + "  Next steps:")
    for step in _next_steps():
        _emit(step)
    _emit()
    _emit(_c("  Tip:", "bold") + " every command takes "
          + _c("--fixture <file>", "cyan") + " to run with no editor at all.")

    _cleanup(tmpdir)
    return 0


def _ok() -> str:
    return _c("✓", "green")


def _emit_next_steps_only() -> None:
    _emit()
    _emit(_c("Next steps:", "bold"))
    for step in _next_steps():
        _emit(step)


def _cleanup(tmpdir: str | None) -> None:
    if tmpdir and os.path.isdir(tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
