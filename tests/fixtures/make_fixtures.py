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


def main() -> None:
    tl = build_cut()
    out = os.path.join(HERE, "cut.otio")
    otio.adapters.write_to_file(tl, out)
    n_clips = sum(
        1 for tr in tl.tracks for ch in tr if isinstance(ch, otio.schema.Clip)
    )
    print(f"wrote {out} — {n_clips} clips across {len(tl.tracks)} tracks")


if __name__ == "__main__":
    main()
