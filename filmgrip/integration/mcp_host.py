"""Claude integration — film-grip as an in-process MCP host with structured EditPlan output.

DECISION (from research): an in-process MCP server hosted inside film-grip's own Claude Agent SDK
process, NOT a plain ``claude -p`` shell-out. Claude *pulls* exactly the FGX context it needs
through read-only tools (``get_selection`` / ``get_context`` / ``query_clips``) instead of being
force-fed a project dump — the single biggest token win — and returns a schema-validated
:class:`~filmgrip.protocol.editplan.EditPlan` via ``output_format=json_schema``.

The orchestration here is behind a :class:`Transport` seam so it is fully unit-testable with no
SDK, no network, and no editor: :class:`ClaudeAgentTransport` is the real path; tests inject a
fake. Crucially we branch on the result *subtype* — handling ``error_max_structured_output_retries``
(the documented terminal failure when the model can't conform) as well as success — and capture
the ``session_id`` so the D16 repair loop can ``resume`` with validation errors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..core.ir import TimelineIR
from ..protocol import editplan as ep
from ..protocol.editplan import EditPlan
from ..protocol.validate import ValidationResult, validate
from ..serialize import fgx

SUBTYPE_SUCCESS = "success"
SUBTYPE_MAX_RETRIES = "error_max_structured_output_retries"


class SelectionLike(Protocol):
    """Duck type of adapters.base.Selection — avoids coupling integration to adapters."""
    ids: list[str]

    def as_header(self, ir: TimelineIR) -> dict: ...


# --------------------------------------------------------------------------- context + tools
@dataclass
class PlannerContext:
    """Everything the read-only tools serve to Claude for one editing turn."""
    ir: TimelineIR
    selection: SelectionLike
    source: Any = None       # live editor session / file path (for the apply step, not the LLM)
    adapter: Any = None


def payload_get_selection(ctx: PlannerContext) -> dict:
    """Tiny payload: the selected ids + rate header. A few dozen tokens, no clip detail."""
    return ctx.selection.as_header(ctx.ir)


def payload_get_context(ctx: PlannerContext, ids: Optional[list[str]] = None, *,
                        hops: int = 1, vertical: bool = False) -> dict:
    """The FGX subgraph for the given ids (default: the current selection)."""
    target = ids if ids else list(ctx.selection.ids)
    return fgx.bundle(ctx.ir, target, hops=hops, include_vertical=vertical)


def payload_query_clips(ctx: PlannerContext, *, name: Optional[str] = None,
                        track: Optional[str] = None) -> dict:
    """Search the whole timeline by name substring and/or track code; returns FGX rows."""
    rows = []
    for c in ctx.ir.real_clips():
        if name and name.lower() not in c.name.lower():
            continue
        if track and fgx.track_code(c.track_kind, c.track_index) != track:
            continue
        rows.append(fgx.clip_row(c))
    return {"cols": fgx.COLS, "clips": rows}


def payload_get_transcript(ctx: PlannerContext, ids: Optional[list[str]] = None, *,
                           asr: Optional[str] = None) -> dict:
    """Word-level transcript phrases for the given clips, keyed by TIMELINE frames.

    Pull-on-demand perception: the planner pays these tokens only when the instruction is
    about content ("cut the um", "tighten the answer"). Per-clip failures (offline media, no
    ASR backend, retimed clip) come back in ``errors`` — never fabricated text.
    """
    from ..perception.align import transcript_for_clips
    from ..perception.transcribe import PerceptionUnavailable, detect_backend

    target = ids if ids else list(ctx.selection.ids)
    try:
        backend = detect_backend(asr)
    except PerceptionUnavailable as exc:
        return {"clips": [], "errors": [str(exc)]}
    return transcript_for_clips(ctx.ir, target, backend=backend)


def payload_analyze_speech(ctx: PlannerContext, ids: Optional[list[str]] = None, *,
                           min_silence_ms: int = 400, asr: Optional[str] = None) -> dict:
    """Deterministic silence/filler analysis + ready-to-apply ``cut_range`` candidates."""
    from ..perception.speech import analyze_clips
    from ..perception.transcribe import PerceptionUnavailable, detect_backend

    target = ids if ids else list(ctx.selection.ids)
    try:
        backend = detect_backend(asr)
    except PerceptionUnavailable as exc:
        return {"clips": [], "candidates": [], "errors": [str(exc)]}
    return analyze_clips(ctx.ir, target, backend=backend,
                         min_silence_s=max(0.05, min_silence_ms / 1000.0))


def payload_view_frames(ctx: PlannerContext, ids: Optional[list[str]] = None,
                        frames: Optional[list[int]] = None, *, count: int = 8) -> dict:
    """Contact-sheet PNGs (filmstrip + waveform) for clips or exact timeline frames.

    Returns file paths + a tile legend; the caller reads the PNG multimodally. Source-media
    based — film-grip never renders/exports to look at the timeline.
    """
    from ..perception.frames import clip_sheet, timeline_sheet
    from ..perception.transcribe import PerceptionUnavailable

    sheets, errors = [], []
    if frames:
        results, errors = timeline_sheet(ctx.ir, [int(f) for f in frames])
        sheets = list(results)
    else:
        target = ids if ids else list(ctx.selection.ids)
        for cid in target:
            try:
                sheets.append(clip_sheet(ctx.ir, cid, count=count))
            except PerceptionUnavailable as exc:
                errors.append(str(exc))
    return {
        "sheets": [{"png": s.png_path, "legend": s.legend, "notes": s.notes} for s in sheets],
        "errors": errors,
    }


def payload_get_scopes(ctx: PlannerContext, ids: Optional[list[str]] = None, *,
                       samples: int = 5) -> dict:
    """Color scopes per clip (parade/waveform/vectorscope/white-balance/exposure) — the perception
    an agent reads to PROPOSE a grade and re-reads to VERIFY it. Pull-on-demand: paid only when the
    instruction is about color.

    Uses :func:`~filmgrip.perception.scopes.analyze_clip_scopes`: each clip is sampled over ``samples``
    frames spanning its source window and the scope stats are returned as a temporally-robust
    **median + spread** (so one flashy frame can't skew the read the planner grades against), together
    with a per-pixel **skin-over-skin-pixels** segmentation (``None`` when a clip shows no skin). Each
    clip entry keeps ``clip_id``; the legacy keys (``luma``/``vectorscope``/``white_balance``/
    ``exposure``/``parade``) move under ``aggregate``. Per-clip failures — retimed clip (refused),
    offline media (curated), no ffmpeg/numpy — come back in ``errors`` (never a fabricated reading)."""
    from ..perception.align import media_path_of
    from ..perception.scopes import analyze_clip_scopes
    from ..perception.transcribe import PerceptionUnavailable

    target = ids if ids else list(ctx.selection.ids)
    clips, errors = [], []
    for cid in target:
        c = ctx.ir.clip(cid)
        if c is None or c.kind != "clip":
            continue
        try:
            rep = analyze_clip_scopes(media_path_of(c), c, samples=samples)
        except PerceptionUnavailable as exc:           # missing dep — applies to every clip
            errors.append(f"{c.name}: {exc}")
            continue
        # Honest passthrough: per-clip errors (retimed/offline) surface; only real reads join clips.
        for e in rep.get("errors", []):
            errors.append(f"{c.name}: {e}")
        if rep.get("n_analyzed", 0) > 0:
            rep["clip_id"] = cid
            clips.append(rep)
    return {"clips": clips, "errors": errors}


def payload_get_beats(ctx: PlannerContext, ids: Optional[list[str]] = None) -> dict:
    """Musical beat grid + tempo per clip — the rhythm perception an agent reads to PROPOSE music-synced
    edits ("cut on the beat", "land each clip on the drop") and re-reads to VERIFY them. Pull-on-demand:
    paid only when the instruction is about rhythm/music.

    Uses :func:`~filmgrip.perception.music.beats_for_media`: each clip's audio is decoded over its source
    window and reduced to a tempo (BPM) + a beat grid in **timeline frames** (plus onsets), via the
    pure-numpy advisory engine (or the librosa upgrade when installed). Each clip entry keeps ``clip_id``.
    Per-clip failures — retimed clip (refused), offline/unresolvable media, decode failure — come back in
    ``errors`` (never a fabricated grid); missing ffmpeg/numpy surfaces as a per-clip error too. Mirrors
    :func:`payload_get_scopes`' ``{clips, errors}`` shape exactly: only real reads join ``clips``."""
    from ..perception.align import media_path_of
    from ..perception.music import beats_for_media
    from ..perception.transcribe import PerceptionUnavailable

    target = ids if ids else list(ctx.selection.ids)
    clips, errors = [], []
    for cid in target:
        c = ctx.ir.clip(cid)
        if c is None or c.kind != "clip":
            continue
        try:
            rep = beats_for_media(media_path_of(c), c)
        except PerceptionUnavailable as exc:           # missing dep — applies to every clip
            errors.append(f"{c.name}: {exc}")
            continue
        # Honest passthrough: per-clip errors (retimed/offline) surface; only real reads join clips.
        rep_errors = rep.get("errors", [])
        for e in rep_errors:
            errors.append(f"{c.name}: {e}")
        if not rep_errors:
            rep["clip_id"] = cid
            clips.append(rep)
    return {"clips": clips, "errors": errors}


def build_system_prompt(ctx: PlannerContext) -> str:
    """Compact planner prompt. Declares the FGX column legend ONCE so rows stay key-free."""
    cols = ",".join(fgx.COLS)
    # Derived from the schema itself (single source of truth) so a newly-added op is taught to the
    # planner automatically — the same no-drift guarantee the honesty gate enforces on the op table.
    ops = ", ".join(ep.all_op_names())
    props = ", ".join(sorted(ep.ALLOWED_PROPERTIES))
    no_audio = ", ".join(sorted(ep.AUDIO_PROPS_UNSUPPORTED))
    color_advisory = ", ".join(sorted(ep.COLOR_ADVISORY))
    return (
        "You are film-grip's video-editing planner. The user selected clips in their editor and "
        "gave a natural-language instruction. Produce an EditPlan: a list of typed ops over STABLE "
        "CLIP IDS.\n"
        f"FGX rows are positional arrays with columns [{cols}] "
        "(id, track e.g. 'v1', start frame, duration frames, source ref, source-in frame). "
        "Frames are integers relative to sequence start. 'GAP'/'XFADE' rows (id '-') are context, "
        "not targetable.\n"
        "Call get_selection first, then get_context(ids) to see the neighborhood; use query_clips "
        "to find clips by name. Only reference clip ids you have seen. Never invent ids or frames.\n"
        f"Ops: {ops}. set_property keys are limited to: {props}.\n"
        "Audio/SFX: use import_audio with sfx=<name from the SFX library you are given> (or "
        "src_ref=<path>) to place a sound effect on an audio track (e.g. 'a1'); add an audio track "
        f"first with add_track if none fits. You CANNOT set audio {no_audio} — those are not "
        "scriptable; never emit them, and if asked, place the audio and note the level is a manual "
        "step.\n"
        "Organizing: add_track / rename_track / create_bin / move_to_bin tidy tracks and media-pool "
        "bins.\n"
        "Color grading: set_cdl applies an ASC CDL primary grade (slope/offset/power per RGB + "
        "saturation; identity = slope[1,1,1] offset[0,0,0] power[1,1,1] sat 1) — think in "
        "lift/gamma/gain and convert (gain→slope, lift→offset, gamma→power=1/gamma). apply_lut "
        "attaches a .cube/.3dl LUT (a baked look). color_group puts clips in a shared color group; "
        "color_version adds/loads alternate grades; apply_grade copies a hero clip's grade onto "
        "targets (from_clip) or applies a saved PowerGrade .drx (drx_path) — the match-this-shot "
        "primitive. To SEE color before/after grading, call get_scopes(ids): RGB parade, luma "
        "waveform (clip/crush flags), vectorscope (hue/saturation + skin-tone delta), white-balance "
        "cast, and exposure — perceive→propose set_cdl→verify against the scopes. CDL is portable; "
        "declare color_space when grading log footage. These color tools are GUI-only in every NLE "
        f"and are NOT scriptable — never emit or claim them: {color_advisory}. If asked for one, do "
        "the closest CDL/LUT grade and note the rest as a manual step.\n"
        "Speed/visibility: retime sets a clip's speed_percent (200=2x faster, 50=half, negative="
        "reverse, 0=freeze-frame); set_enabled enables/disables a clip without deleting it. retime "
        "warps playback within the clip's existing span — add a ripple if you want the gap closed. "
        "speed_ramp (variable speed, ease in/out) is a real decision film-grip can VALIDATE but NOT "
        "apply on any editor — no NLE scripting API or interchange format carries a portable speed "
        "curve. For a CONSTANT speed change use retime; a variable ramp is a manual step (Resolve "
        "Retime Curve / Premiere Time Remapping) — note it, never claim it was applied.\n"
        "Content: when the instruction is about WHAT IS SAID (umms, filler, 'cut after she says "
        "X', tightening an answer), call get_transcript(ids) — phrases come back as "
        "[startFrame-endFrame] lines in timeline frames. For 'remove silences/fillers/tighten', "
        "call analyze_speech(ids) and reuse its ready-made cut_range candidates. cut_range "
        "removes [start_frame,end_frame) from inside ONE clip (ripple=true closes the hole); "
        "order rippling cut_ranges on a track LAST-TO-FIRST. ASR timestamps drift 50-100ms: "
        "place cuts in the silent gaps between phrases, never inside a word, and pad word edges "
        "by 1-5 frames. To LOOK at footage (framing, content, cut boundaries), call "
        "view_frames(ids or frames) and read the returned PNG. When the edit is about RHYTHM or "
        "MUSIC ('cut on the beat', 'land each clip on the drop', 'sync to the beat'), call "
        "get_beats(ids): tempo (BPM) + a beat grid in timeline frames (plus onsets) — perceive the "
        "grid, place cuts on it, then re-read to verify. If these tools return errors "
        "(no ASR backend, offline media, no ffmpeg), say so — do not guess content.\n"
        "Rationale: optionally set reason (why this edit) and quote (the spoken words it anchors "
        "to) on each op — they show up in the diff and are stored on the clip, making the edit "
        "auditable.\n"
        "Return ONLY an EditPlan matching the provided JSON schema. Keep ops minimal and reversible."
    )


# --------------------------------------------------------------------------- transport seam
@dataclass
class PlanResponse:
    """Normalized transport result, independent of the SDK."""
    subtype: str
    structured_output: Optional[dict] = None
    session_id: Optional[str] = None
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw: Any = None


@dataclass
class PlanResult:
    ok: bool
    terminal: bool = False                 # True = don't bother repairing (model gave up / hard error)
    plan: Optional[EditPlan] = None
    validation: Optional[ValidationResult] = None
    errors: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: Any = None


class Transport:
    """Run one planning turn. Implementations: ClaudeAgentTransport (real), fakes (tests)."""

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict,
            ctx: PlannerContext, session_id: Optional[str] = None,
            model: Optional[str] = None) -> PlanResponse:
        raise NotImplementedError


def plan_edit(ctx: PlannerContext, user_prompt: str, transport: Transport, *,
              session_id: Optional[str] = None, model: Optional[str] = None) -> PlanResult:
    """Ask Claude (via ``transport``) for an EditPlan; parse + validate it host-side.

    Branches on the transport subtype — success vs the terminal
    ``error_max_structured_output_retries`` vs any other error — and never returns a plan that
    hasn't been validated against the live IR.
    """
    schema = ep.schema()
    sys_prompt = build_system_prompt(ctx)
    try:
        resp = transport.run(system_prompt=sys_prompt, user_prompt=user_prompt, schema=schema,
                             ctx=ctx, session_id=session_id, model=model)
    except Exception as exc:  # a backend that errors hard (missing SDK, network) must not crash callers
        return PlanResult(ok=False, terminal=True, errors=[f"planner backend failed: {exc}"])

    if resp.subtype == SUBTYPE_MAX_RETRIES:
        return PlanResult(
            ok=False, terminal=True,
            errors=["model could not produce a schema-valid EditPlan after retries "
                    "(error_max_structured_output_retries)"],
            session_id=resp.session_id, cost_usd=resp.cost_usd, usage=resp.usage, raw=resp.raw)

    if resp.subtype != SUBTYPE_SUCCESS or resp.structured_output is None:
        return PlanResult(
            ok=False, terminal=True,
            errors=resp.errors or [f"no structured plan returned (subtype={resp.subtype})"],
            session_id=resp.session_id, cost_usd=resp.cost_usd, usage=resp.usage, raw=resp.raw)

    try:
        plan = EditPlan.parse(resp.structured_output)
    except Exception as exc:  # well-formed JSON but not a valid EditPlan
        return PlanResult(ok=False, terminal=False, errors=[f"plan parse failed: {exc}"],
                          session_id=resp.session_id, cost_usd=resp.cost_usd,
                          usage=resp.usage, raw=resp.structured_output)

    vres = validate(plan, ctx.ir)
    return PlanResult(ok=vres.ok, terminal=False, plan=plan, validation=vres,
                      errors=[str(e) for e in vres.errors], session_id=resp.session_id,
                      cost_usd=resp.cost_usd, usage=resp.usage, raw=resp.structured_output)


# --------------------------------------------------------------------------- real SDK wiring
def build_mcp_server(ctx: PlannerContext):
    """Build the in-process MCP server exposing the three read-only tools (SDK required)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool  # lazy: optional dep
    try:
        from claude_agent_sdk import ToolAnnotations  # type: ignore
        read_only = ToolAnnotations(readOnlyHint=True)
    except Exception:  # pragma: no cover
        read_only = None

    def _text(obj) -> dict:
        return {"content": [{"type": "text", "text": json.dumps(obj, separators=(",", ":"))}]}

    @tool("get_selection", "Get the user's selected clip ids + rate header (tiny).",
          {}, read_only)
    async def get_selection(_args):  # pragma: no cover - exercised only with the live SDK
        return _text(payload_get_selection(ctx))

    @tool("get_context", "Get the FGX subgraph for clip ids (selection by default).",
          {"ids": list, "hops": int, "vertical": bool}, read_only)
    async def get_context(args):  # pragma: no cover
        return _text(payload_get_context(ctx, args.get("ids"),
                                         hops=int(args.get("hops", 1)),
                                         vertical=bool(args.get("vertical", False))))

    @tool("query_clips", "Find clips by name substring and/or track code.",
          {"name": str, "track": str}, read_only)
    async def query_clips(args):  # pragma: no cover
        return _text(payload_query_clips(ctx, name=args.get("name"), track=args.get("track")))

    @tool("get_transcript", "Word-level transcript phrases for clips, in timeline frames "
          "(use when the edit is about spoken content).",
          {"ids": list, "asr": str}, read_only)
    async def get_transcript(args):  # pragma: no cover
        return _text(payload_get_transcript(ctx, args.get("ids"), asr=args.get("asr")))

    @tool("analyze_speech", "Silence/filler analysis with ready-to-apply cut_range candidates "
          "(use for 'remove silences', 'cut the umms', 'tighten this').",
          {"ids": list, "min_silence_ms": int, "asr": str}, read_only)
    async def analyze_speech(args):  # pragma: no cover
        return _text(payload_analyze_speech(
            ctx, args.get("ids"), min_silence_ms=int(args.get("min_silence_ms", 400)),
            asr=args.get("asr")))

    @tool("view_frames", "Render a contact-sheet PNG (filmstrip + waveform) for clips or exact "
          "timeline frames; returns the PNG path + tile legend — read the PNG to actually look.",
          {"ids": list, "frames": list, "count": int}, read_only)
    async def view_frames(args):  # pragma: no cover
        return _text(payload_view_frames(ctx, args.get("ids"), args.get("frames"),
                                         count=int(args.get("count", 8))))

    @tool("get_scopes", "Color scopes for clips (RGB parade, luma waveform with clip/crush flags, "
          "vectorscope hue/saturation + skin-tone delta, white-balance cast, exposure) — read these "
          "to propose a grade and to verify one landed.",
          {"ids": list}, read_only)
    async def get_scopes(args):  # pragma: no cover
        return _text(payload_get_scopes(ctx, args.get("ids")))

    @tool("get_beats", "Musical beat grid + tempo for clips (tempo BPM, beat frames in timeline time, "
          "onsets) — read these to propose music-synced cuts ('cut on the beat') and to verify one "
          "landed.",
          {"ids": list}, read_only)
    async def get_beats(args):  # pragma: no cover
        return _text(payload_get_beats(ctx, args.get("ids")))

    return create_sdk_mcp_server("filmgrip", "0.1.0",
                                 [get_selection, get_context, query_clips, get_transcript,
                                  analyze_speech, view_frames, get_scopes, get_beats])


_FG_TOOLS = ["mcp__filmgrip__get_selection", "mcp__filmgrip__get_context",
             "mcp__filmgrip__query_clips", "mcp__filmgrip__get_transcript",
             "mcp__filmgrip__analyze_speech", "mcp__filmgrip__view_frames",
             "mcp__filmgrip__get_scopes", "mcp__filmgrip__get_beats"]


class ClaudeAgentTransport(Transport):
    """The real path: drive the Claude Agent SDK with an in-process MCP server (network required).

    By default the planning turn runs under :func:`~filmgrip.integration.auth.subscription_billing`,
    so it bills to the user's Claude subscription (any ``ANTHROPIC_API_KEY`` in the environment is
    dropped for the call). Pass ``prefer_subscription=False`` (or set ``FILMGRIP_USE_SUBSCRIPTION=0``)
    to bill an API key instead.
    """

    def __init__(self, *, prefer_subscription: Optional[bool] = None):
        self._prefer_subscription = prefer_subscription

    def run(self, *, system_prompt, user_prompt, schema, ctx, session_id=None, model=None):  # pragma: no cover
        import asyncio

        from .auth import subscription_billing, use_subscription_default

        prefer = (use_subscription_default() if self._prefer_subscription is None
                  else self._prefer_subscription)
        with subscription_billing(prefer):
            return asyncio.run(self._run_async(
                system_prompt=system_prompt, user_prompt=user_prompt, schema=schema,
                ctx=ctx, session_id=session_id, model=model))

    async def _run_async(self, *, system_prompt, user_prompt, schema, ctx,
                         session_id=None, model=None) -> PlanResponse:  # pragma: no cover
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        server = build_mcp_server(ctx)
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"filmgrip": server},
            allowed_tools=_FG_TOOLS,
            output_format={"type": "json_schema", "schema": schema, "name": "EditPlan"},
            permission_mode="default",
            resume=session_id,
            model=model,
        )
        final: Optional[ResultMessage] = None
        async for msg in query(prompt=user_prompt, options=options):
            if isinstance(msg, ResultMessage):
                final = msg
        return self._to_plan_response(final)

    @staticmethod
    def _to_plan_response(final) -> PlanResponse:
        """Map the SDK's final ``ResultMessage`` (or ``None``) to a transport-neutral PlanResponse.

        Pure — no SDK import, no network — so the field/subtype mapping that "Claude is live" rests on
        is unit-testable (the live ``_run_async`` is the only un-coverable glue). ``subtype`` falls back
        to ``error``/``success`` from ``is_error`` when the SDK omits it; ``None`` totals coerce to 0/{}/
        [] so a caller never sees a stray ``None``.
        """
        if final is None:
            return PlanResponse(subtype="error", errors=["no ResultMessage from SDK"])
        return PlanResponse(
            subtype=final.subtype or ("error" if final.is_error else SUBTYPE_SUCCESS),
            structured_output=final.structured_output,
            session_id=final.session_id,
            cost_usd=float(final.total_cost_usd or 0.0),
            usage=dict(final.usage or {}),
            errors=list(final.errors or []),
            raw=final.result,
        )
