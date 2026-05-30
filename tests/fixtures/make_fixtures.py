"""Generate checked-in test fixtures deterministically.

Run with the project venv:  ``.venv/bin/python tests/fixtures/make_fixtures.py``
Produces ``cut.otio`` (a 12-clip, multi-track, 24fps timeline with a gap and a dissolve).
Kept in-repo so fixtures are reproducible rather than opaque binaries.
"""
from __future__ import annotations

import os

import opentimelineio as otio

RATE = 24.0
HERE = os.path.dirname(os.path.abspath(__file__))


def rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, RATE)


def clip(name: str, src_start: int, dur: int, url: str | None = None) -> otio.schema.Clip:
    return otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(target_url=url or f"/media/{name}.mov"),
        source_range=otio.opentime.TimeRange(start_time=rt(src_start), duration=rt(dur)),
    )


def gap(dur: int) -> otio.schema.Gap:
    return otio.schema.Gap(source_range=otio.opentime.TimeRange(start_time=rt(0), duration=rt(dur)))


def dissolve(frames: int = 6) -> otio.schema.Transition:
    return otio.schema.Transition(
        name="dissolve",
        transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
        in_offset=rt(frames),
        out_offset=rt(frames),
    )


def build_cut() -> otio.schema.Timeline:
    tl = otio.schema.Timeline(name="Demo Cut")
    tl.global_start_time = rt(0)

    v1 = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v1.append(clip("intro", 0, 48))         # 0..48
    v1.append(clip("dialogue_a", 100, 72))  # 48..120
    v1.append(dissolve())                   # overlaps b/c, no track time
    v1.append(clip("dialogue_b", 20, 36))   # 120..156
    v1.append(gap(24))                       # 156..180
    v1.append(clip("broll_1", 0, 60))       # 180..240
    v1.append(clip("broll_2", 0, 60))       # 240..300
    v1.append(clip("outro", 0, 60))         # 300..360

    v2 = otio.schema.Track(name="V2", kind=otio.schema.TrackKind.Video)
    v2.append(gap(60))                       # 0..60
    v2.append(clip("lower_third", 0, 60))   # 60..120
    v2.append(gap(80))                       # 120..200
    v2.append(clip("logo_bug", 0, 60))      # 200..260

    a1 = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    a1.append(clip("vo_take", 0, 156, url="/media/vo.wav"))   # 0..156
    a1.append(clip("vo_take2", 0, 204, url="/media/vo2.wav"))  # 156..360

    a2 = otio.schema.Track(name="A2", kind=otio.schema.TrackKind.Audio)
    a2.append(clip("music_bed", 0, 100, url="/media/music.wav"))  # 0..100
    a2.append(gap(160))                                           # 100..260
    a2.append(clip("stinger", 0, 100, url="/media/stinger.wav"))  # 260..360

    tl.tracks.extend([v1, v2, a1, a2])
    return tl


def write_recorded_plan(otio_path: str) -> str:
    """Emit plan.json referencing real clip IDs from cut.otio, so the offline e2e never drifts."""
    import json

    from filmgrip.core.ir import TimelineIR  # local import; deps installed in the venv

    ir = TimelineIR.from_otio_file(otio_path)
    by_name = {c.name: c for c in ir.real_clips()}
    plan = {
        "notes": "tighten the open and flag the b-roll",
        "ops": [
            {"op": "trim", "clip_id": by_name["intro"].id, "edge": "out", "delta": -12},
            {"op": "add_marker", "clip_id": by_name["dialogue_a"].id, "frame": 0,
             "color": "Blue", "name": "review"},
            {"op": "set_property", "clip_id": by_name["broll_1"].id, "key": "ZoomX", "value": 1.2},
        ],
    }
    out = os.path.join(HERE, "plan.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    return out


def _clip_with_media(name: str, src_start: int, dur: int) -> otio.schema.Clip:
    return otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(
            target_url=f"/media/{name}.mov",
            available_range=otio.opentime.TimeRange(rt(0), rt(2000))),  # known media length
        source_range=otio.opentime.TimeRange(rt(src_start), rt(dur)),
    )


def write_sample_fcpxml() -> str:
    """A FCPXML with a transition, so interchange write-back emits a lossy-fidelity warning."""
    tl = otio.schema.Timeline(name="Sample FCPXML")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v.append(_clip_with_media("shotA", 100, 60))
    v.append(dissolve())
    v.append(_clip_with_media("shotB", 200, 80))
    v.append(_clip_with_media("shotC", 0, 40))
    tl.tracks.append(v)
    out = os.path.join(HERE, "sample.fcpxml")
    otio.adapters.write_to_file(tl, out, adapter_name="fcp_xml")
    return out


def write_sample_edl() -> str:
    """A simple single-track EDL (cuts only — the lowest common denominator)."""
    tl = otio.schema.Timeline(name="Sample EDL")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V", kind=otio.schema.TrackKind.Video)
    for nm, ss, d in [("a", 100, 48), ("b", 200, 72), ("c", 0, 36)]:
        v.append(_clip_with_media(nm, ss, d))
    tl.tracks.append(v)
    out = os.path.join(HERE, "sample.edl")
    otio.adapters.write_to_file(tl, out, adapter_name="cmx_3600")
    return out


_MLT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<mlt version="7.22.0" title="{title}">
  <profile frame_rate_num="24" frame_rate_den="1" width="1920" height="1080"/>
  <producer id="p0" in="0" out="47"><property name="mlt_service">avformat</property><property name="resource">/media/alpha.mp4</property></producer>
  <producer id="p1" in="0" out="71"><property name="mlt_service">avformat</property><property name="resource">/media/bravo.mp4</property></producer>
  <producer id="p2" in="0" out="35"><property name="mlt_service">avformat</property><property name="resource">/media/charlie.mp4</property></producer>
  <playlist id="playlist0">
    <entry producer="p0" in="0" out="47"/>
    <entry producer="p1" in="0" out="71"/>
    <entry producer="p2" in="0" out="35"/>
  </playlist>
  <tractor id="tractor0">
    <track producer="playlist0"/>
  </tractor>
</mlt>
"""


def write_sample_mlt() -> list[str]:
    """Shotcut (.mlt) and Kdenlive (.kdenlive) share the MLT XML schema; emit both."""
    paths = []
    for ext, title in ((".mlt", "fg-sample-shotcut"), (".kdenlive", "fg-sample-kdenlive")):
        out = os.path.join(HERE, f"sample{ext}")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(_MLT_TEMPLATE.format(title=title))
        paths.append(out)
    return paths


def write_capcut() -> list[str]:
    """An unencrypted CapCut International draft (fps 25 -> 40000us/frame, exact) + an encrypted blob."""
    import json

    def seg(sid, mid, start_us, dur_us):
        return {"id": sid, "material_id": mid,
                "target_timerange": {"start": start_us, "duration": dur_us},
                "source_timerange": {"start": 0, "duration": dur_us}}

    draft = {
        "name": "fg-capcut", "fps": 25, "duration": 6000000,
        "materials": {
            "videos": [
                {"id": "mat1", "path": "/media/clip_one.mp4", "material_name": "clip_one.mp4"},
                {"id": "mat2", "path": "/media/clip_two.mp4"},
                {"id": "mat3", "path": "/media/clip_three.mp4"},
            ],
            "audios": [{"id": "aud1", "path": "/media/song.mp3"}],
        },
        "tracks": [
            {"type": "video", "segments": [
                seg("s1", "mat1", 0, 2400000),         # clip_one 0..60
                seg("s2", "mat2", 2400000, 2400000),   # clip_two 60..120
                seg("s3", "mat3", 4800000, 1200000),   # clip_three 120..150
            ]},
            {"type": "audio", "segments": [seg("a1", "aud1", 0, 6000000)]},  # song 0..150
        ],
    }
    paths = []
    good = os.path.join(HERE, "capcut_draft.json")
    with open(good, "w", encoding="utf-8") as fh:
        json.dump(draft, fh, ensure_ascii=False, indent=1)
    paths.append(good)

    enc = os.path.join(HERE, "capcut_encrypted.json")
    with open(enc, "wb") as fh:
        fh.write(b"\x00\x07JIANYING-ENCRYPTED\xff\xfe" + bytes(range(16)))
    paths.append(enc)
    return paths


def write_filmora() -> str:
    """A newer ZIP-based .wfp containing a project.json with a tracks/clips graph (frame-based)."""
    import json
    import zipfile

    doc = {
        "name": "fg-filmora", "version": "13.0", "fps": 30,
        "tracks": [
            {"type": "video", "clips": [
                {"name": "opening", "start": 0, "duration": 90, "in": 0, "path": "opening.mp4"},
                {"name": "interview", "start": 90, "duration": 150, "in": 0, "path": "interview.mp4"},
            ]},
            {"type": "audio", "clips": [
                {"name": "bgm", "start": 0, "duration": 240, "in": 0, "path": "bgm.mp3"},
            ]},
        ],
    }
    out = os.path.join(HERE, "sample.wfp")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(doc, ensure_ascii=False))
        zf.writestr("resources/thumb.dat", b"\x00" * 8)  # non-JSON sibling, must be ignored
    return out


def main() -> None:
    tl = build_cut()
    out = os.path.join(HERE, "cut.otio")
    otio.adapters.write_to_file(tl, out)
    n_clips = sum(
        1 for tr in tl.tracks for ch in tr if isinstance(ch, otio.schema.Clip)
    )
    print(f"wrote {out} — {n_clips} clips across {len(tl.tracks)} tracks")
    plan_path = write_recorded_plan(out)
    print(f"wrote {plan_path}")
    for fn in (write_sample_fcpxml, write_sample_edl):
        try:
            print(f"wrote {fn()}")
        except Exception as exc:  # interchange adapters may be absent on a minimal install
            print(f"skipped {fn.__name__}: {exc}")
    for p in write_sample_mlt():
        print(f"wrote {p}")
    for p in write_capcut():
        print(f"wrote {p}")
    print(f"wrote {write_filmora()}")


if __name__ == "__main__":
    main()
