"""The typed EditPlan protocol — the only thing Claude is allowed to emit.

Claude never writes editor-specific code. It returns an :class:`EditPlan`: a list of bounded,
reversible primitive ops that reference **stable clip IDs**. This is what makes film-grip
accurate, cheap, and safe:

* **accurate/safe** — a hallucinated clip ID or out-of-bounds frame cannot survive the D6
  validator; the worst case is a rejected plan, never a corrupted timeline.
* **cheap** — each op is ~10–30 tokens of structured output, not a paragraph of code.
* **portable** — the same plan applies to Resolve (live) or to any interchange adapter, because
  ops are defined against the universal IR, not an editor's API.

The pydantic models double as the JSON Schema handed to Claude for structured output, so the
model is constrained to valid op shapes at generation time, and re-validated host-side before
anything touches a project.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 4  # v2 added audio + organize ops; v3 added cut_range + per-op rationale; v4 adds color grading (set_cdl …)

# Working color spaces a grade may declare. CDL itself carries NO working space — the same numbers
# look different in log vs display-referred footage — so film-grip lets a grade name its intended
# space honestly rather than silently assuming one. "" = unspecified (use the project's space).
COLOR_SPACES: frozenset[str] = frozenset({
    "", "rec709", "rec2020", "srgb", "aces", "acescct", "acescc", "davinci_wide_gamut",
    "arri_logc3", "arri_logc4", "sony_slog3", "red_log3g10", "blackmagic_film_gen5",
    "panasonic_vlog", "canon_clog3", "linear",
})

# Properties an adapter is allowed to set. Anything outside this set is rejected at parse time —
# Claude cannot invent a destructive or unsupported property key. Adapters map these to their
# native equivalents (e.g. Resolve TimelineItem.SetProperty('ZoomX', ...)).
ALLOWED_PROPERTIES: frozenset[str] = frozenset({
    # transform
    "ZoomX", "ZoomY", "Pan", "Tilt", "RotationAngle", "AnchorPointX", "AnchorPointY",
    "Pitch", "Yaw", "FlipX", "FlipY",
    # crop
    "CropLeft", "CropRight", "CropTop", "CropBottom", "CropSoftness",
    # composite / opacity / retime
    "Opacity", "compositeMode", "Speed",
    # film-grip meta (adapters translate these)
    "name", "enabled", "color",
})

# Resolve's marker/flag palette (also a sensible generic set). Validated case-insensitively.
MARKER_COLORS: frozenset[str] = frozenset({
    "Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia", "Rose",
    "Lavender", "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream",
})

TRANSITION_TYPES: frozenset[str] = frozenset({
    "cross_dissolve", "dip_to_color", "smooth_cut", "fade_in", "fade_out", "wipe",
})

# Audio attributes Resolve's scripting API does NOT expose on a timeline item (they live on the
# Fairlight page and aren't scriptable). film-grip refuses to pretend it can set these — they are
# surfaced as an honest limitation (in the planner prompt + adapter capabilities), never silently
# dropped. A user asking to "lower the volume" is told it's a manual step, not given a false success.
AUDIO_PROPS_UNSUPPORTED: frozenset[str] = frozenset({
    "volume", "gain", "level", "fade_in", "fade_out", "pan", "mute", "solo",
})


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields -> no smuggled instructions

    # Rationale fields (v3, optional on EVERY op): the edit explains itself. `reason` is why this
    # op exists; `quote` is the transcript words it anchors to ("cut the um at the top"). They ride
    # along into dry-run diffs and OTIO metadata, making a plan auditable text a reviewer can read —
    # and costing zero tokens when omitted.
    reason: str = Field(default="", max_length=200,
                        description="why this edit (optional; shown in diffs and stored on the clip)")
    quote: str = Field(default="", max_length=200,
                       description="the spoken words this op anchors to (optional)")


class Trim(_Op):
    """Adjust a clip's in- or out-point by ``delta`` frames (negative = shorten/move earlier)."""
    op: Literal["trim"] = "trim"
    clip_id: str
    edge: Literal["in", "out"]
    delta: int = Field(description="frames to move the edge; +out lengthens, +in shortens head")


class Move(_Op):
    """Move a clip to a new timeline start frame, optionally to another track."""
    op: Literal["move"] = "move"
    clip_id: str
    to_start: int = Field(ge=0, description="new timeline start frame")
    to_track: Optional[str] = Field(default=None, description="e.g. 'v2'; None keeps current track")


class Insert(_Op):
    """Insert a media reference as a new clip on a track at a timeline frame."""
    op: Literal["insert"] = "insert"
    src_ref: str = Field(description="media basename or media-pool id to place")
    track: str = Field(description="target track code, e.g. 'v1' or 'a1'")
    at_start: int = Field(ge=0)
    source_in: int = Field(default=0, ge=0)
    duration: int = Field(gt=0)
    media_type: Literal["auto", "video", "audio"] = Field(
        default="auto",
        description="'auto' infers from the track code (a* = audio); set explicitly to force")


class Delete(_Op):
    op: Literal["delete"] = "delete"
    clip_id: str
    ripple: bool = Field(default=False, description="close the gap left behind")


class SetProperty(_Op):
    op: Literal["set_property"] = "set_property"
    clip_id: str
    key: str
    value: Union[bool, int, float, str]

    @field_validator("key")
    @classmethod
    def _key_allowed(cls, v: str) -> str:
        if v not in ALLOWED_PROPERTIES:
            raise ValueError(
                f"property '{v}' is not in the allowlist ({sorted(ALLOWED_PROPERTIES)})"
            )
        return v


class AddMarker(_Op):
    """Add a marker. ``frame`` is an offset within the clip (0 = clip start)."""
    op: Literal["add_marker"] = "add_marker"
    clip_id: str
    frame: int = Field(default=0, ge=0)
    color: str = "Blue"
    name: str = ""
    note: str = ""
    duration: int = Field(default=1, gt=0)

    @field_validator("color")
    @classmethod
    def _color_known(cls, v: str) -> str:
        match = {c.lower(): c for c in MARKER_COLORS}.get(v.lower())
        if match is None:
            raise ValueError(f"unknown marker color '{v}'; one of {sorted(MARKER_COLORS)}")
        return match


class AddTransition(_Op):
    """Add a transition on a clip edge (e.g. a cross dissolve at the 'in' edge)."""
    op: Literal["add_transition"] = "add_transition"
    clip_id: str
    edge: Literal["in", "out"] = "out"
    type: str = "cross_dissolve"
    duration: int = Field(default=12, gt=0)

    @field_validator("type")
    @classmethod
    def _type_known(cls, v: str) -> str:
        if v not in TRANSITION_TYPES:
            raise ValueError(f"unknown transition '{v}'; one of {sorted(TRANSITION_TYPES)}")
        return v


class Split(_Op):
    """Split a clip into two at a timeline frame inside it."""
    op: Literal["split"] = "split"
    clip_id: str
    at_frame: int = Field(ge=0, description="absolute timeline frame, must fall inside the clip")


class Ripple(_Op):
    """Shift every clip at/after ``from_frame`` on a track by ``delta`` frames."""
    op: Literal["ripple"] = "ripple"
    from_frame: int = Field(ge=0)
    delta: int
    track: Optional[str] = Field(default=None, description="restrict to one track code, else all")


class ImportAudio(_Op):
    """Import a sound effect / audio file and place it on an audio track.

    Reference the effect by ``sfx`` (a name in the user's SFX library) OR ``src_ref`` (an explicit
    audio file path). The host resolves ``sfx`` → a real path via the SFX library, imports it into the
    project's media pool, and lands it on ``track`` at ``at_start``.

    Honest limitation: per-clip volume/gain/fades are NOT settable via Resolve scripting (Fairlight
    only). This op *places* audio; level/fade tweaks stay a manual step — see ``AUDIO_PROPS_UNSUPPORTED``.
    """
    op: Literal["import_audio"] = "import_audio"
    track: str = Field(description="target AUDIO track code, e.g. 'a1'")
    at_start: int = Field(default=0, ge=0, description="timeline frame to place the audio")
    sfx: str = Field(default="", description="name of an effect in the user's SFX library")
    src_ref: str = Field(default="", description="explicit audio file path (alternative to 'sfx')")
    source_in: int = Field(default=0, ge=0, description="frame offset into the audio source")
    duration: Optional[int] = Field(default=None, gt=0,
                                    description="frames to place; None = the file's full length")

    @field_validator("track")
    @classmethod
    def _is_audio_track(cls, v: str) -> str:
        if v[:1].lower() != "a":
            raise ValueError(f"import_audio target must be an audio track (e.g. 'a1'), got '{v}'")
        return v

    @model_validator(mode="after")
    def _has_a_source(self) -> "ImportAudio":
        if not (self.sfx or self.src_ref):
            raise ValueError("import_audio needs either 'sfx' (library name) or 'src_ref' (file path)")
        return self


class AddTrack(_Op):
    """Add a new track to the timeline (organizing primitive)."""
    op: Literal["add_track"] = "add_track"
    kind: Literal["video", "audio", "subtitle"] = "video"
    audio_type: str = Field(default="stereo",
                            description="for audio tracks: mono/stereo/5.1/7.1 etc (Resolve subTrackType)")


class RenameTrack(_Op):
    """Rename an existing track (organizing primitive)."""
    op: Literal["rename_track"] = "rename_track"
    track: str = Field(description="track code to rename, e.g. 'v2' or 'a1'")
    name: str = Field(min_length=1, max_length=120)


class CreateBin(_Op):
    """Create a media-pool bin/folder (organizing primitive)."""
    op: Literal["create_bin"] = "create_bin"
    name: str = Field(min_length=1, max_length=120)
    parent: str = Field(default="", description="parent bin name; empty = the media-pool root")


class MoveToBin(_Op):
    """Move a clip's source media into a media-pool bin (organizing primitive)."""
    op: Literal["move_to_bin"] = "move_to_bin"
    clip_id: str
    bin: str = Field(min_length=1, max_length=120, description="destination bin name")


class Retime(_Op):
    """Change a clip's playback speed — the real retime primitive, via an OTIO time-warp effect.

    ``speed_percent`` is a percentage of normal speed: 100 = normal, 200 = 2× faster, 50 = half
    speed, a negative value plays in reverse (e.g. -100 = reverse at normal speed), and 0 = a
    freeze-frame. It maps to an ``otio.schema.LinearTimeWarp`` (``FreezeFrame`` at 0) on the clip,
    which the OTIO-rebuild path carries — so it lands on every rebuild/interchange editor and in
    Resolve via the rebuild. The warp changes how the source plays *within* the clip's existing
    timeline span; it does not move neighbours (re-ripple separately if you want the gap closed).
    """
    op: Literal["retime"] = "retime"
    clip_id: str
    speed_percent: float = Field(
        default=100.0, ge=-10000.0, le=10000.0,
        description="100=normal, 200=2x faster, 50=half speed, negative=reverse, 0=freeze-frame")


class SetEnabled(_Op):
    """Enable or disable a clip without removing it (a disabled clip stays on the timeline but is
    skipped on playback/render). Lands live in Resolve (``SetClipEnabled``) and via the OTIO
    rebuild (the clip's ``enabled`` flag)."""
    op: Literal["set_enabled"] = "set_enabled"
    clip_id: str
    enabled: bool = Field(description="True = enable the clip, False = disable (mute) it")


class CutRange(_Op):
    """Remove the timeline frame range ``[start_frame, end_frame)`` from inside one clip — the
    silence/filler-removal primitive.

    Atomic split+split+delete that stays addressable: it references the ORIGINAL clip id and
    absolute timeline frames, so one plan can carve several ranges out of one clip (a chain of
    ``split`` ops can't — the middle piece's id doesn't exist until after apply). ``ripple=True``
    closes the hole (everything later on the track shifts left); ``False`` leaves a gap.

    Ordering contract (validated): when ``ripple=True``, cut_range ops on the SAME track must be
    ordered last-to-first (descending ``start_frame``) so earlier ranges aren't shifted by the
    time they apply.
    """
    op: Literal["cut_range"] = "cut_range"
    clip_id: str
    start_frame: int = Field(ge=0, description="absolute timeline frame the cut starts at")
    end_frame: int = Field(gt=0, description="absolute timeline frame the cut ends before")
    ripple: bool = Field(default=True, description="True = close the hole; False = leave a gap")

    @model_validator(mode="after")
    def _range_is_forward(self) -> "CutRange":
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"cut_range needs start_frame < end_frame (got {self.start_frame}..{self.end_frame})")
        return self


def _triple_in_range(name: str, lo: Optional[float], hi: Optional[float]):
    """Build a validator that enforces an SOP field is exactly 3 floats, each within [lo, hi]."""
    def _check(v: list[float]) -> list[float]:
        vals = list(v)
        if len(vals) != 3:
            raise ValueError(f"{name} must be exactly 3 values (r, g, b), got {len(vals)}")
        out = []
        for ch in vals:
            f = float(ch)
            if lo is not None and f < lo:
                raise ValueError(f"{name} value {f} below minimum {lo}")
            if hi is not None and f > hi:
                raise ValueError(f"{name} value {f} above maximum {hi}")
            out.append(f)
        return out
    return _check


class SetCDL(_Op):
    """Apply an **ASC CDL** primary grade to a clip — film-grip's portable color primitive.

    Ten numbers: ``slope``/``offset``/``power`` per RGB channel (SOP) + a ``saturation`` scalar.
    This carries the *decision*, not an editor's node graph, so the SAME grade lands live in
    DaVinci Resolve (``TimelineItem.SetCDL`` per node), rides through OTIO clip metadata, and
    exports to ``.cdl``/``.cc`` sidecars + FCPXML ``info-asc-cdl`` — portable across every adapter.

    Identity grade (no-op): ``slope=[1,1,1] offset=[0,0,0] power=[1,1,1] saturation=1``. Math is
    ``out = (max(0, in*slope + offset))**power`` then saturation about Rec.709 luma. ``power`` must
    be > 0; ``slope`` and ``saturation`` >= 0; ``offset`` is unbounded by the standard. CDL carries
    no working space, so declare ``color_space`` when the footage isn't already in the grade's
    intended space (e.g. grading log footage) — film-grip surfaces it rather than guessing.

    Colorist controls (lift/gamma/gain/contrast/temp/tint) compile to this via
    ``filmgrip.color.lgg_to_cdl`` — emit raw SOP here, or let a grade pack / the ``film-grip grade``
    CLI do the lowering. Richer color work (curves, qualifiers, Power Windows, Magic Mask) is
    GUI-only in every NLE and is surfaced as an advisory step, never silently applied.
    """
    op: Literal["set_cdl"] = "set_cdl"
    clip_id: str
    slope: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0],
                               description="RGB multiplicative gain (>=0); identity = [1,1,1]")
    offset: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                description="RGB additive lift; identity = [0,0,0]")
    power: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0],
                               description="RGB gamma exponent (>0); identity = [1,1,1]")
    saturation: float = Field(default=1.0, ge=0.0, le=4.0,
                              description="1 = unchanged, 0 = greyscale, >1 boosts")
    node_index: int = Field(default=1, ge=1, le=128,
                            description="1-based color node the grade lands on in Resolve")
    color_space: str = Field(default="", max_length=40,
                             description="intended working space (CDL declares none); '' = project default")

    _v_slope = field_validator("slope")(_triple_in_range("slope", 0.0, 100.0))
    _v_offset = field_validator("offset")(_triple_in_range("offset", -10.0, 10.0))
    _v_power = field_validator("power")(_triple_in_range("power", 1e-6, 100.0))

    @field_validator("color_space")
    @classmethod
    def _space_known(cls, v: str) -> str:
        key = v.strip().lower()
        if key not in COLOR_SPACES:
            raise ValueError(f"unknown color_space '{v}'; one of {sorted(c for c in COLOR_SPACES if c)}")
        return key


class ApplyLut(_Op):
    """Apply a **LUT** (look-up table) to a clip — the baked *look* primitive CDL primaries can't
    express (film emulation, camera→display transforms, creative looks).

    A LUT is referenced by file: an absolute/relative ``path`` to a ``.cube`` (1D or 3D) or ``.3dl``,
    or a bare name the editor resolves against its own LUT folder. film-grip validates an on-disk
    file's shape (size keyword, row count, numeric rows) so a malformed or hallucinated path is
    rejected before apply; a bare name is allowed with a warning (it can't be verified host-side).

    Lands live in DaVinci Resolve via ``SetLUT(nodeIndex, path)`` (per node) and rides through OTIO
    clip metadata for the interchange path. Honest limitation: a LUT is just a file — bare path
    references break across machines, FCPXML carries it by name only, and AAF can't carry a 3D LUT
    at all. Ship the ``.cube`` with the project or bake it; film-grip surfaces this, never hides it.
    """
    op: Literal["apply_lut"] = "apply_lut"
    clip_id: str
    path: str = Field(min_length=1, max_length=1024,
                      description="path to a .cube/.3dl LUT, or a bare name in the editor's LUT folder")
    node_index: int = Field(default=1, ge=1, le=128,
                            description="1-based color-node the LUT attaches to (Resolve)")


AnyOp = Annotated[
    Union[
        Trim, Move, Insert, Delete, SetProperty, AddMarker, AddTransition, Split, Ripple,
        ImportAudio, AddTrack, RenameTrack, CreateBin, MoveToBin, Retime, SetEnabled, CutRange,
        SetCDL, ApplyLut,
    ],
    Field(discriminator="op"),
]


def all_op_names() -> list[str]:
    """Every op type the schema accepts, derived from the ``AnyOp`` union.

    The single source of truth for "what ops exist", so a newly-added op automatically shows up in
    the capability table and the honesty-gate tests instead of needing a hand-maintained list kept in
    sync by discipline (exactly the drift the gate exists to prevent).
    """
    import typing

    union = typing.get_args(AnyOp)[0]  # AnyOp == Annotated[Union[...], Field(discriminator="op")]
    return [model.model_fields["op"].default for model in typing.get_args(union)]


class EditPlan(BaseModel):
    """An ordered list of typed ops plus an optional one-line rationale."""
    model_config = ConfigDict(extra="forbid")

    version: int = SCHEMA_VERSION
    notes: str = Field(default="", max_length=400, description="one-line rationale (optional)")
    ops: list[AnyOp] = Field(default_factory=list)

    @classmethod
    def parse(cls, data) -> "EditPlan":
        if isinstance(data, str):
            data = json.loads(data)
        return cls.model_validate(data)


def schema() -> dict:
    """The JSON Schema for an EditPlan — handed to Claude as the structured-output contract."""
    return EditPlan.model_json_schema()


def write_schema(path: str = "editplan.schema.json") -> str:
    """Write the EditPlan JSON Schema to ``path`` and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(schema(), fh, indent=2)
    return path


if __name__ == "__main__":  # pragma: no cover
    print("wrote", write_schema())
