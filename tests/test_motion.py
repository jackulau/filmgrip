"""Deliverable D8 — motion & scene perception. The pure cores (``motion_series``,
``scene_boundaries``, ``stability_score``) are proved on synthetic frame stacks with known temporal
structure (no editor, no ffmpeg). The ffmpeg extraction path (``analyze_motion``) is exercised when
ffmpeg is present, else skipped — mirroring ``tests/test_scopes.py``'s optional-dep discipline."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from filmgrip.perception import motion
from filmgrip.perception.motion import motion_series, scene_boundaries, stability_score


# --------------------------------------------------------------------------- synthetic stacks
def _solid_stack(n, level, h=16, w=16):
    """N identical solid-gray frames at ``level`` (0..255)."""
    return np.full((n, h, w, 3), level, dtype=np.uint8)


def _hard_cut_stack(h=16, w=16):
    """5 dark frames then 5 bright frames → one hard cut at frame index 5."""
    dark = _solid_stack(5, 10, h, w)
    bright = _solid_stack(5, 240, h, w)
    return np.concatenate([dark, bright], axis=0)


def _moving_stack(n=10, h=24, w=40, step=3):
    """A bright block that pans steadily one direction (no wrap) → continuous motion, no cut."""
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        x = min(i * step, w - 5)
        frames[i, :, x:x + 4, :] = 200
    return frames


def _jittery_stack(n=12, h=16, w=16):
    """A block that oscillates back and forth (shake): high-frequency, low net displacement."""
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        x = 6 if (i % 2 == 0) else 9      # toggles every frame → jitter
        frames[i, :, x:x + 3, :] = 200
    return frames


# ----------------------------------------------------------------------- scene boundaries (cut)
def test_scene_boundaries_finds_hard_cut_within_one_frame():
    series = motion_series(_hard_cut_stack())
    cuts = scene_boundaries(series, fps=25, min_gap_frames=1)
    assert len(cuts) == 1
    # the cut is between frame 4 and 5 → reported as frame 5 (new shot's first frame), ±1.
    assert abs(cuts[0] - 5) <= 1


def test_no_boundary_in_static_stack():
    assert scene_boundaries(motion_series(_solid_stack(8, 128)), fps=25) == []


def test_min_gap_debounces_adjacent_spikes():
    # two cuts only 2 frames apart; a 5-frame gap should keep only the first.
    stack = np.concatenate([_solid_stack(3, 10), _solid_stack(2, 240), _solid_stack(3, 10)], axis=0)
    series = motion_series(stack)
    assert len(scene_boundaries(series, fps=25, thresh=0.3, min_gap_frames=5)) == 1


# ----------------------------------------------------------------------- motion magnitude
def test_static_motion_below_moving_motion():
    static = motion_series(_solid_stack(10, 128))
    moving = motion_series(_moving_stack())
    assert float(static.max() if static.size else 0.0) < float(moving.mean())
    assert float(static.mean() if static.size else 0.0) == 0.0   # identical frames → exactly 0


def test_motion_series_is_normalized_0_1():
    s = motion_series(_hard_cut_stack())
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s.max() > 0.5     # a dark→bright cut is a large normalized delta


def test_motion_series_single_frame_is_empty():
    assert motion_series(_solid_stack(1, 128)).size == 0


# ----------------------------------------------------------------------- stability / shake
def test_steady_more_stable_than_jittery():
    steady = stability_score(_solid_stack(12, 128))        # static → perfectly steady
    pan = stability_score(_moving_stack())                 # constant-speed move → still steady-ish
    jittery = stability_score(_jittery_stack())            # oscillation → shaky
    assert steady > jittery
    assert pan > jittery
    assert 0.0 <= jittery <= 1.0


def test_static_stack_is_maximally_stable():
    assert stability_score(_solid_stack(10, 80)) == 1.0


# ----------------------------------------------------------------------- shape / robustness
def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        motion_series(np.zeros((8, 8, 3), dtype=np.uint8))   # missing the N axis


# ----------------------------------------------------------------------- negative: no numpy
def test_missing_numpy_raises_perception_unavailable(monkeypatch):
    from filmgrip.perception.transcribe import PerceptionUnavailable
    monkeypatch.setattr(motion, "np", None)
    with pytest.raises(PerceptionUnavailable):
        motion._need_numpy()


# ----------------------------------------------------------------------- ffmpeg-gated end-to-end
def _ffmpeg():
    from filmgrip.perception.transcribe import ffmpeg_path
    return ffmpeg_path()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_analyze_motion_finds_boundary_near_cut(tmp_path):
    # Concatenate two 1s solid-color lavfi segments → a single hard cut at t=1.0s.
    red = tmp_path / "red.mp4"
    blue = tmp_path / "blue.mp4"
    two = tmp_path / "two_shot.mp4"
    ff = _ffmpeg()
    for path, color in ((red, "red"), (blue, "blue")):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"color=c={color}:s=160x90:d=1:r=25", str(path)],
                       check=True, capture_output=True)
    listing = tmp_path / "list.txt"
    listing.write_text(f"file '{red}'\nfile '{blue}'\n")
    subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(two)], check=True, capture_output=True)

    fps = 4
    report = motion.analyze_motion(str(two), fps=fps)
    assert report["n_frames"] >= 6
    assert report["boundaries_frames"], "expected at least one detected boundary"
    # the cut is at t=1.0s; at fps=4 that's around sample frame 4 (±2 frames of slack).
    assert any(abs(b - int(round(1.0 * fps))) <= 2 for b in report["boundaries_frames"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_analyze_motion_static_clip_has_no_boundary(tmp_path):
    ff = _ffmpeg()
    static = tmp_path / "static.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=gray:s=160x90:d=2:r=25", str(static)],
                   check=True, capture_output=True)
    report = motion.analyze_motion(str(static), fps=4)
    assert report["boundaries_frames"] == []
    assert report["stability"] == 1.0
