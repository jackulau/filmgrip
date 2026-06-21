"""D10 — temporal multi-frame scope sampling + per-pixel skin segmentation.

Like the rest of perception, the deterministic core is proved on synthetic numpy with NO ffmpeg:
``frame_rgb`` is monkeypatched to return known frames, so ``analyze_clip_scopes`` exercises its
sampling → :func:`analyze_rgb` → aggregation pipeline offline. We assert:

* aggregation reports the correct median + spread over a known set of frames, and a single flashy
  frame does not move the median or flip the verdict (the whole point of going multi-frame);
* the per-pixel skin mask isolates a skin-coloured region from a mostly sky-blue frame (the skin
  delta is computed from the patch, not the blue whole-frame average);
* a frame with NO skin pixels reports skin absent (``None``) — never a fabricated reading;
* a retimed clip is refused into ``errors`` with no fabricated scopes, and offline media yields the
  curated "source media not found" error (the same honesty wall as music/motion);
* the 0x0-frame fix: ``analyze_rgb`` raises ``ValueError`` (not ``ZeroDivisionError``) on an empty
  frame.
"""
from __future__ import annotations

import numpy as np
import opentimelineio as otio
import pytest

from filmgrip.perception import scopes
from filmgrip.perception.scopes import analyze_clip_scopes, analyze_rgb, skin_segment


# ----------------------------------------------------------------------- frame builders
def _solid(r, g, b, h=16, w=16):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = r, g, b
    return a


def _grey(v, h=8, w=8):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :] = v
    return a


def _sky_with_skin_patch(h=20, w=20, patch=5):
    """A frame that is mostly sky-blue with a small skin-tone patch in the corner."""
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = 110, 160, 235          # sky blue everywhere
    a[0:patch, 0:patch, 0] = 210                                # skin patch (210,150,120)
    a[0:patch, 0:patch, 1] = 150
    a[0:patch, 0:patch, 2] = 120
    return a


# ----------------------------------------------------------------------- otio clip helpers
def _rt(frames: int, rate: int = 24) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, rate)


def _clip(media_url: str, *, src_in_f: int = 0, dur_f: int = 240, start_f: int = 0):
    """A flattened film-grip Clip wrapping an OTIO clip (mirrors test_music's construction)."""
    from filmgrip.core.ir import TimelineIR

    c = otio.schema.Clip(name="clip",
                         media_reference=otio.schema.ExternalReference(target_url=media_url),
                         source_range=otio.opentime.TimeRange(_rt(src_in_f), _rt(dur_f)))
    tl = otio.schema.Timeline(name="t")
    tr = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    tl.tracks.append(tr)
    tr.append(c)
    ir = TimelineIR(tl)
    return ir.real_clips()[0]


def _patch_frames(monkeypatch, frames):
    """Make ``frame_rgb`` return ``frames`` in order (cycling), recording the timestamps it was asked
    for — so the aggregation is over a KNOWN set and we can assert on the sampled at_s too."""
    seen: list[float] = []
    seq = list(frames)

    def fake_frame_rgb(media_path, at_s=0.0, **kw):
        seen.append(round(float(at_s), 4))
        return seq[(len(seen) - 1) % len(seq)]

    monkeypatch.setattr(scopes, "frame_rgb", fake_frame_rgb)
    return seen


# ----------------------------------------------------------------------- aggregation: median + spread
def test_aggregate_reports_median_and_spread_over_known_frames(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")                                   # real file so os.path.isfile passes
    # luma medians 50,100,128,150,250 — the 250 is a flashy outlier that must NOT move the median.
    greys = [50, 100, 128, 150, 250]
    seen = _patch_frames(monkeypatch, [_grey(v) for v in greys])

    out = analyze_clip_scopes(str(media), samples=5)

    assert out["n_analyzed"] == 5 and len(seen) == 5
    lm = out["aggregate"]["luma"]["median"]
    assert lm["median"] == 128.0                                # median of {50,100,128,150,250}
    assert lm["iqr"] == 50.0                                    # p75−p25 = 150−100
    assert lm["min"] == 50.0 and lm["max"] == 250.0
    assert lm["std"] > 0.0                                      # real spread reported
    # the lone 250 frame does not flip the aggregate verdict to "over".
    assert out["verdict"]["exposure"] == "ok"
    assert out["errors"] == []


def test_single_flashy_frame_does_not_dominate_median(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    # four steady mid frames + one blown-out white frame.
    _patch_frames(monkeypatch, [_grey(120), _grey(120), _grey(120), _grey(120), _solid(255, 255, 255)])
    out = analyze_clip_scopes(str(media), samples=5)
    assert out["aggregate"]["luma"]["median"]["median"] == 120.0     # robust to the flash
    assert out["verdict"]["clipped"] is False                        # 1/5 clipped ≠ majority


def test_samples_span_the_clip_source_window(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    seen = _patch_frames(monkeypatch, [_grey(128)])
    # clip: source-in 24f @24fps = 1.0s, duration 96f = 4.0s → samples inside [1.0, 5.0].
    clip = _clip(media.as_uri(), src_in_f=24, dur_f=96)
    out = analyze_clip_scopes(str(media), clip, samples=4)
    assert out["n_analyzed"] == 4
    assert all(1.0 <= t <= 5.0 for t in seen), seen
    assert seen == sorted(seen) and len(set(seen)) == 4          # distinct, increasing samples


# ----------------------------------------------------------------------- per-pixel skin segmentation
def test_skin_mask_isolates_patch_from_blue_background():
    a = _sky_with_skin_patch()
    rep = analyze_rgb(a, skin=True)
    # whole-frame hue is blue-dominated (~325-330°), far from the skin axis...
    assert scopes._angle_diff(rep["vectorscope"]["hue_deg"], scopes.SKIN_TONE_ANGLE_DEG) > 90.0
    # ...but the skin SEGMENT reads the patch: hue near the skin axis, tiny delta.
    skin = rep["skin"]
    assert skin is not None and skin["present"] is True
    assert skin["skin_tone_delta_deg"] < 8.0
    assert skin["pixel_frac"] == pytest.approx(25 / 400, abs=1e-3)   # 5x5 patch in 20x20
    # the public helper agrees.
    assert skin_segment(a)["hue_deg"] == skin["hue_deg"]


def test_frame_with_no_skin_reports_absent():
    # pure sky-blue: no pixel falls in the skin band → honest None, never a fabricated reading.
    assert skin_segment(_solid(110, 160, 235)) is None
    assert analyze_rgb(_solid(60, 200, 70), skin=True)["skin"] is None    # green, also absent


def test_skin_flag_default_keeps_output_byte_identical():
    a = _solid(210, 150, 120)
    assert "skin" not in analyze_rgb(a)                          # default: no skin key
    assert analyze_rgb(a) == analyze_rgb(a, skin=False)          # byte-identical
    assert "skin" in analyze_rgb(a, skin=True)                   # opt-in only


def test_clip_scopes_aggregates_skin_over_skin_frames_only(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    # two frames with skin, two without (pure sky) — skin aggregate uses only the skin frames.
    _patch_frames(monkeypatch, [_sky_with_skin_patch(), _solid(110, 160, 235),
                                _sky_with_skin_patch(), _solid(110, 160, 235)])
    out = analyze_clip_scopes(str(media), samples=4)
    skin = out["skin"]
    assert skin is not None and skin["frames_with_skin"] == 2 and skin["frames_total"] == 4
    assert skin["skin_tone_delta_deg"]["median"] < 8.0


def test_clip_scopes_skin_none_when_no_frame_has_skin(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    _patch_frames(monkeypatch, [_solid(110, 160, 235), _solid(60, 200, 70)])
    out = analyze_clip_scopes(str(media), samples=3)
    assert out["skin"] is None                                  # never fabricated


# ----------------------------------------------------------------------- honesty wall
def test_retimed_clip_is_refused_with_no_scopes(monkeypatch, tmp_path):
    media = tmp_path / "clip.mov"
    media.write_bytes(b"\x00")
    # frame_rgb must never be called for a refused clip.
    monkeypatch.setattr(scopes, "frame_rgb",
                        lambda *a, **k: pytest.fail("frame_rgb called on a retimed clip"))
    clip = _clip(media.as_uri())
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    out = analyze_clip_scopes(str(media), clip, samples=5)
    assert out["n_analyzed"] == 0 and out["aggregate"] == {} and out["skin"] is None
    assert any("retimed" in e for e in out["errors"])


def test_offline_media_is_curated_error_not_a_guess():
    out = analyze_clip_scopes("/nonexistent/offline.mov", samples=5)
    assert out["n_analyzed"] == 0 and out["aggregate"] == {}
    assert any("source media not found" in e and "offline" in e for e in out["errors"])


def test_clip_scopes_needs_numpy_or_ffmpeg(monkeypatch, tmp_path):
    """A real file but no ffmpeg → frame_rgb raises PerceptionUnavailable; with every sample failing
    the result is an honest empty payload carrying the reason (not a fabricated reading)."""
    from filmgrip.perception.transcribe import PerceptionUnavailable

    media = tmp_path / "real.mov"
    media.write_bytes(b"\x00")

    def boom(*a, **k):
        raise PerceptionUnavailable("ffmpeg is required for scope extraction")

    monkeypatch.setattr(scopes, "frame_rgb", boom)
    out = analyze_clip_scopes(str(media), samples=3)
    assert out["n_analyzed"] == 0
    assert any("ffmpeg" in e for e in out["errors"])


# ----------------------------------------------------------------------- 0x0-frame ValueError fix
def test_empty_frame_raises_valueerror_not_zerodivision():
    # (0,0,3) passes the ndim/shape[2] check but has zero pixels — must be a clear ValueError now.
    with pytest.raises(ValueError) as exc:
        analyze_rgb(np.zeros((0, 0, 3), dtype=np.uint8))
    assert "empty" in str(exc.value).lower()
    # a zero-height frame is the same class of bug.
    with pytest.raises(ValueError):
        analyze_rgb(np.zeros((0, 8, 3), dtype=np.uint8))
