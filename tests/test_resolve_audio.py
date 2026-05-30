"""D6 — Resolve audio apply: ImportMedia + AppendToTimeline(mediaType=2) + AddTrack('audio')."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.audio.library import SfxLibrary
from filmgrip.protocol.editplan import EditPlan
from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem

SFX_DIR = "tests/fixtures/sfx"


def _session(resolve):
    return rc.connect(resolve)


def _adapter():
    # Inject the fixture SFX library so 'whoosh' resolves to whoosh_01.wav.
    return ResolveAdapter(sfx_library=SfxLibrary.load(SFX_DIR))


def _build_with_audio():
    """V1 one clip, A1 one short clip (gap after 60) — room to drop an effect later in a1."""
    v1 = [FakeTimelineItem("intro", 0, 48)]
    a1 = [FakeTimelineItem("music", 0, 60, src_path="/media/music.wav")]
    tl = FakeTimeline("T")
    tl.add_track("video", 1, v1)
    tl.add_track("audio", 1, a1)
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), tl


def test_import_audio_imports_media_and_appends_to_audio_track():
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "whoosh", "duration": 18},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    # the resolved SFX file was imported into the media pool
    assert any("whoosh_01.wav" in str(p) for p in mp.imported_media)
    # it was appended as AUDIO (mediaType 2) on track index 1 at the record frame
    info = mp.append_log[-1][0][0]
    assert info["mediaType"] == 2 and info["trackIndex"] == 1 and info["recordFrame"] == 120
    assert info["endFrame"] - info["startFrame"] == 18


def test_import_audio_explicit_src_ref_skips_library():
    resolve, tl = _build_with_audio()
    adapter = ResolveAdapter()  # no library injected — must not be needed for src_ref
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "src_ref": "/snd/boom.wav"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert "/snd/boom.wav" in mp.imported_media


def test_import_audio_unknown_sfx_fails_cleanly():
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "nonexistent_zither"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("not found" in e for e in res.errors)


def test_add_track_audio_maps_to_native_addtrack():
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    before = tl.GetTrackCount("audio")
    plan = EditPlan.parse({"ops": [{"op": "add_track", "kind": "audio", "audio_type": "stereo"}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert ("AddTrack", "audio", "stereo") in tl.calls
    assert tl.GetTrackCount("audio") == before + 1


def test_add_track_then_import_audio_in_one_plan():
    resolve, tl = _build_with_audio()  # starts with a1 only
    adapter = _adapter()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    plan = EditPlan.parse({"ops": [
        {"op": "add_track", "kind": "audio", "audio_type": "stereo"},      # creates a2
        {"op": "import_audio", "track": "a2", "at_start": 0, "sfx": "riser", "duration": 40},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert mp.append_log[-1][0][0]["trackIndex"] == 2  # placed on the newly-created a2


def test_append_failure_aborts():
    resolve, tl = _build_with_audio()
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    mp.AppendToTimeline = lambda *a: []  # Resolve's falsy-on-failure
    adapter = _adapter()
    session = _session(resolve)
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "whoosh"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("rolled back" in e or "AppendToTimeline" in e for e in res.errors)


def test_import_audio_missing_file_rejected_before_apply():
    # ghost_missing -> not_here.wav is a loadable manifest entry whose file is absent; importing it
    # must fail with a clear "missing" message, not a cryptic ImportMedia failure.
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "ghost_missing"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("missing" in e for e in res.errors)


def test_import_audio_full_length_emits_warning():
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "whoosh"},  # no duration
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert any("FULL length" in w for w in res.warnings)


def test_import_audio_resolves_description_via_library_fallback():
    # 'slam impact' is not an exact name -> lib.get() misses, lib.resolve() matches door_slam by tag.
    resolve, tl = _build_with_audio()
    adapter = _adapter()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "slam impact", "duration": 20},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert any("door_slam.wav" in str(p) for p in mp.imported_media)


@pytest.mark.live
def test_live_import_audio_smoke():
    """Real Resolve: import a tiny SFX and place it on a1. Skips unless Resolve is open."""
    try:
        rc.load_module()
    except rc.ResolveUnavailable:
        pytest.skip("DaVinciResolveScript not importable")
    session = rc.connect()
    if session is None or session.current_timeline() is None:
        pytest.skip("Resolve not running or no timeline open")
    if int(session.current_timeline().GetTrackCount("audio") or 0) < 1:
        pytest.skip("timeline has no audio track")
    # Use a real fixture wav as the source (placeholder bytes are fine for the call path).
    import os
    src = os.path.abspath(os.path.join(SFX_DIR, "whoosh_01.wav"))
    adapter = ResolveAdapter()
    plan = EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 0, "src_ref": src, "duration": 12},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
