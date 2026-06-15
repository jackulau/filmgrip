"""ASC CDL — film-grip's portable color primitive (the OTIO of color).

A CDL is ten numbers: **S**lope, **O**ffset, **P**ower per RGB channel (SOP, nine numbers) plus
one **Sat**uration scalar. It carries the colorist's *decision*, not an editor's implementation,
so it is vendor-neutral and survives interchange where node graphs, qualifiers, and Power Windows
do not. This is exactly why it is the right pivot for a portable, agent-emittable primary grade —
the same role OTIO plays for the cut.

The math (ASC CDL v1.2, verified against the published controls):

* SOP, per channel, on a normalized ``0..1`` value::

      out = ( max(0, in * slope + offset) ) ** power

  The base is clamped non-negative *before* the power because a negative base to a fractional
  exponent is undefined. Identity: ``slope=1, offset=0, power=1``.

* Saturation, applied after SOP about Rec.709 luma::

      luma  = 0.2126*R + 0.7152*G + 0.0722*B
      out_c = luma + sat * (c - luma)        for c in {R, G, B}

  ``sat=1`` is identity, ``0`` is greyscale, ``>1`` boosts. (Rec.709 weights — some secondary
  sources wrongly cite Rec.601; CDL uses 709.)

CDL does NOT declare its own working color space: the same numbers look different applied in log
vs display-referred encodings. film-grip records an honest ``color_space`` hint alongside the
grade rather than pretending the ambiguity away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

# Rec.709 luma coefficients (saturation pivot).
_LUMA = (0.2126, 0.7152, 0.0722)

# A neutral mid-grey pivot for contrast (scene-linear-ish 18% grey lands near here in many
# display encodings; used only by the ergonomic compiler, never by the canonical math).
DEFAULT_PIVOT = 0.435

Triple = tuple[float, float, float]
Scalarish = Union[float, int, Sequence[float]]


def _as_triple(v: Scalarish, *, name: str) -> Triple:
    """Coerce a scalar or length-3 sequence into a ``(r, g, b)`` float triple."""
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, f, f)
    seq = list(v)
    if len(seq) != 3:
        raise ValueError(f"{name} must be a number or 3 values (r, g, b), got {seq!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class CDL:
    """An ASC CDL value: slope/offset/power per RGB channel + a saturation scalar.

    Immutable so a grade can be shared/cached safely (e.g. as the undo value of a write-only
    ``SetCDL``). Construct with :meth:`identity`, :func:`lgg_to_cdl`, or directly with SOP triples.
    """

    slope: Triple = (1.0, 1.0, 1.0)
    offset: Triple = (0.0, 0.0, 0.0)
    power: Triple = (1.0, 1.0, 1.0)
    saturation: float = 1.0
    # Honest working-space hint — CDL itself carries none. "" = unspecified (assume the project's).
    color_space: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", _as_triple(self.slope, name="slope"))
        object.__setattr__(self, "offset", _as_triple(self.offset, name="offset"))
        object.__setattr__(self, "power", _as_triple(self.power, name="power"))
        object.__setattr__(self, "saturation", float(self.saturation))
        # Validity: power must be > 0 (the power function is undefined / degenerate otherwise),
        # slope and saturation must be >= 0. Offset is unbounded by the standard.
        for i, p in enumerate(self.power):
            if p <= 0.0:
                raise ValueError(f"CDL power[{i}] must be > 0 (got {p})")
        for i, s in enumerate(self.slope):
            if s < 0.0:
                raise ValueError(f"CDL slope[{i}] must be >= 0 (got {s})")
        if self.saturation < 0.0:
            raise ValueError(f"CDL saturation must be >= 0 (got {self.saturation})")

    @classmethod
    def identity(cls) -> "CDL":
        """The do-nothing grade."""
        return cls()

    @property
    def is_identity(self) -> bool:
        return (self.slope == (1.0, 1.0, 1.0)
                and self.offset == (0.0, 0.0, 0.0)
                and self.power == (1.0, 1.0, 1.0)
                and self.saturation == 1.0)

    # ------------------------------------------------------------------ math
    def apply(self, rgb: Sequence[float]) -> Triple:
        """Apply this grade to one normalized ``0..1`` RGB sample. Pure reference math — used by
        the color-perception/verify layer to predict a grade's effect without an editor."""
        r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        out = []
        for c, s, o, p in zip((r, g, b), self.slope, self.offset, self.power):
            base = c * s + o
            base = base if base > 0.0 else 0.0
            out.append(base ** p)
        if self.saturation != 1.0:
            luma = sum(w * v for w, v in zip(_LUMA, out))
            out = [luma + self.saturation * (v - luma) for v in out]
        return (_clamp01(out[0]), _clamp01(out[1]), _clamp01(out[2]))

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        d = {
            "slope": list(self.slope),
            "offset": list(self.offset),
            "power": list(self.power),
            "saturation": self.saturation,
        }
        if self.color_space:
            d["color_space"] = self.color_space
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CDL":
        return cls(
            slope=tuple(d.get("slope", (1.0, 1.0, 1.0))),
            offset=tuple(d.get("offset", (0.0, 0.0, 0.0))),
            power=tuple(d.get("power", (1.0, 1.0, 1.0))),
            saturation=float(d.get("saturation", 1.0)),
            color_space=str(d.get("color_space", "")),
        )

    def to_resolve(self, node_index: int = 1) -> dict:
        """The dict DaVinci Resolve's ``TimelineItem.SetCDL`` expects (space-joined ``"R G B"``
        SOP strings, scalar saturation, 1-based node index)."""
        def s3(t: Triple) -> str:
            return f"{t[0]:.6g} {t[1]:.6g} {t[2]:.6g}"

        return {
            "NodeIndex": str(node_index),
            "Slope": s3(self.slope),
            "Offset": s3(self.offset),
            "Power": s3(self.power),
            "Saturation": f"{self.saturation:.6g}",
        }

    def __str__(self) -> str:
        def s3(t: Triple) -> str:
            return f"({t[0]:.4g}, {t[1]:.4g}, {t[2]:.4g})"

        cs = f" @{self.color_space}" if self.color_space else ""
        return (f"CDL slope={s3(self.slope)} offset={s3(self.offset)} "
                f"power={s3(self.power)} sat={self.saturation:.4g}{cs}")


def lgg_to_cdl(
    *,
    lift: Scalarish = 0.0,
    gamma: Scalarish = 1.0,
    gain: Scalarish = 1.0,
    contrast: float = 1.0,
    pivot: float = DEFAULT_PIVOT,
    saturation: float = 1.0,
    temp: float = 0.0,
    tint: float = 0.0,
    color_space: str = "",
) -> CDL:
    """Compile colorist-ergonomic controls into a canonical ASC :class:`CDL`.

    An agent (or a grade pack, or the CLI) reasons in the controls a colorist actually thinks in;
    film-grip deterministically lowers them to the portable SOP+sat primitive that is stored,
    validated, and applied. This is an *approximation* of the lift/gamma/gain control set in terms
    of CDL's SOP math — honest by construction (CDL cannot express log wheels exactly), and it is
    documented as such so nobody mistakes it for Resolve's primary wheels.

    Mapping (each of lift/gamma/gain may be a scalar or an ``(r, g, b)`` triple):

    * ``gain``     → slope        (multiplicative, lifts highlights)
    * ``lift``     → offset       (additive, lifts shadows)
    * ``gamma``    → ``power = 1/gamma``  (gamma>1 brightens mids, matching the colorist convention)
    * ``contrast`` about ``pivot`` → folded into slope/offset: ``out = (in-pivot)*c + pivot``
    * ``temp``     → warm/cool: ``temp>0`` boosts R slope, cuts B slope (range ~[-1, 1])
    * ``tint``     → green/magenta: ``tint>0`` boosts G slope (range ~[-1, 1])
    * ``saturation`` passes straight through.

    ``lgg_to_cdl()`` with all defaults returns the identity grade.
    """
    lift_t = _as_triple(lift, name="lift")
    gamma_t = _as_triple(gamma, name="gamma")
    gain_t = _as_triple(gain, name="gain")

    # Temperature/tint as gentle per-channel slope nudges. 0.2 keeps a full ±1 within a sane range.
    k = 0.2
    temp_mult = (1.0 + k * temp, 1.0, 1.0 - k * temp)         # warm = +R, -B
    tint_mult = (1.0, 1.0 + k * tint, 1.0)                    # green = +G

    slope = []
    offset = []
    power = []
    for i in range(3):
        g = gain_t[i] * temp_mult[i] * tint_mult[i]
        # contrast about pivot: out = in*contrast + pivot*(1-contrast), composed with gain.
        s = g * contrast
        o = lift_t[i] + pivot * (1.0 - contrast)
        slope.append(s)
        offset.append(o)
        gm = gamma_t[i]
        if gm <= 0.0:
            raise ValueError(f"gamma must be > 0 (got {gm})")
        power.append(1.0 / gm)

    return CDL(
        slope=tuple(slope),
        offset=tuple(offset),
        power=tuple(power),
        saturation=float(saturation),
        color_space=color_space,
    )
