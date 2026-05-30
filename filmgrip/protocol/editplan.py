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

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

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


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields -> no smuggled instructions


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
    track: str = Field(description="target track code, e.g. 'v1'")
    at_start: int = Field(ge=0)
    source_in: int = Field(default=0, ge=0)
    duration: int = Field(gt=0)


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


AnyOp = Annotated[
    Union[Trim, Move, Insert, Delete, SetProperty, AddMarker, AddTransition, Split, Ripple],
    Field(discriminator="op"),
]


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
