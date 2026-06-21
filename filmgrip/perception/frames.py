"""Visual perception: ffmpeg contact sheets — a filmstrip + waveform the model reads as ONE image.

An agent can't scrub video; it CAN read one composite PNG at a decision point. This module
renders that PNG from the *source media* (never a render/export — film-grip stays
non-destructive): evenly-sampled frames across a clip, or exact timeline frames (e.g. ±1.5s
around a cut for the D5 verify loop), with an audio waveform strip underneath.

Pure ffmpeg composition — no imaging dependency. Each sheet returns a machine-readable legend
(tile → timeline frame → media seconds) so the picture and the numbers travel together. All
failure modes (no ffmpeg, offline media, no audio stream, retimed clip) surface as honest
errors/notes, never an empty image pretending to be evidence.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from ..core.ir import Clip, TimelineIR
from .align import is_retimed, media_path_of
from .transcribe import PerceptionUnavailable, ffmpeg_path

TILE_HEIGHT = 180
WAVE_HEIGHT = 96


@dataclass
class SheetResult:
    """One rendered contact sheet + the legend that decodes its tiles."""

    png_path: str
    legend: list[dict]          # [{tile, timeline_frame, media_s, clip_id}]
    notes: list[str]            # honest annotations (e.g. "no audio stream — waveform omitted")


def frames_dir() -> str:
    base = os.environ.get("FILMGRIP_CACHE_DIR") or os.path.expanduser("~/.filmgrip")
    return os.path.join(base, "frames")


def _run(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _require_ffmpeg() -> str:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable(
            "ffmpeg is required for frame extraction — install it (e.g. `brew install ffmpeg`).")
    return ffmpeg


def extract_frame(media_path: str, at_s: float, out_png: str, *,
                  height: int = TILE_HEIGHT) -> None:
    """One frame at ``at_s`` (media seconds) scaled to ``height`` px tall."""
    ffmpeg = _require_ffmpeg()
    proc = _run([ffmpeg, "-y", "-v", "error", "-ss", f"{max(0.0, at_s):.3f}",
                 "-i", media_path, "-frames:v", "1", "-vf", f"scale=-2:{height}", out_png])
    if proc.returncode != 0 or not os.path.isfile(out_png):
        raise PerceptionUnavailable(
            f"ffmpeg could not extract a frame at {at_s:.2f}s from '{media_path}': "
            f"{proc.stderr.strip()[:200]}")


def _source_fps(media_path: str) -> Optional[float]:
    """Video stream frame rate (``r_frame_rate``) via ffprobe, or ``None`` if unprobeable.

    Used to turn requested media-second timestamps into exact decoded-frame indices so a single
    decode can emit one frame per timestamp (``select='eq(n,..)+..'``). Returns ``None`` for
    variable/unknown rates so callers can fall back to the per-tile seek path.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=r_frame_rate", "-of", "csv=p=0", media_path])
    raw = proc.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def _extract_frames_batch(media_path: str, times_s: list[float], out_dir: str, *,
                          height: int = TILE_HEIGHT) -> list[str]:
    """Emit one tile per timestamp from a SINGLE ffmpeg decode (vs one spawn+seek per tile).

    Maps each media-second timestamp to its decoded-frame index (needs a probeable constant rate)
    and selects exactly those frames in one pass via ``select='eq(n,F0)+eq(n,F1)+..'`` →
    ``image2`` numbered output. Returns the tile paths in request order.

    Honest failure: a non-zero ffmpeg exit raises :class:`PerceptionUnavailable` with stderr; and
    if the decode yields *fewer* frames than requested (duplicate/out-of-range indices, a short or
    truncated source) that is surfaced as an error too — never a silently short sheet. When the rate
    is unprobeable, falls back to the per-tile seek path (still correct, just N decodes).
    """
    ffmpeg = _require_ffmpeg()
    n = len(times_s)
    fps = _source_fps(media_path)
    if fps is None:                       # can't index frames reliably — honest, correct fallback.
        tiles = []
        for i, t in enumerate(times_s):
            tile = os.path.join(out_dir, f"f{i:03d}.png")
            extract_frame(media_path, t, tile, height=height)
            tiles.append(tile)
        return tiles

    # Sorted, de-duplicated frame indices keep ffmpeg's output order deterministic; we re-expand to
    # one tile per requested timestamp afterwards (so repeated timestamps each get a tile).
    indices = [max(0, int(round(max(0.0, t) * fps))) for t in times_s]
    uniq = sorted(set(indices))
    expr = "+".join(f"eq(n\\,{f})" for f in uniq)
    pattern = os.path.join(out_dir, "batch_%05d.png")
    proc = _run([ffmpeg, "-y", "-v", "error", "-i", media_path, "-vf",
                 f"select='{expr}',scale=-2:{height}", "-fps_mode", "passthrough", pattern])
    produced = sorted(
        f for f in os.listdir(out_dir) if f.startswith("batch_") and f.endswith(".png"))
    if proc.returncode != 0:
        raise PerceptionUnavailable(
            f"ffmpeg could not batch-extract {n} frame(s) from '{media_path}': "
            f"{proc.stderr.strip()[:200]}")
    if len(produced) != len(uniq):
        raise PerceptionUnavailable(
            f"ffmpeg returned {len(produced)} of {len(uniq)} requested frame(s) from "
            f"'{media_path}' — source may be shorter than the timeline or truncated "
            f"(no silent short sheet).")
    # Map each unique frame index → its emitted file, then fan back out to request order.
    by_index = {idx: os.path.join(out_dir, name) for idx, name in zip(uniq, produced)}
    tiles = []
    for i, idx in enumerate(indices):
        tile = os.path.join(out_dir, f"f{i:03d}.png")
        shutil.copyfile(by_index[idx], tile)
        tiles.append(tile)
    return tiles


def _png_width(path: str) -> Optional[int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = _run([ffprobe, "-v", "error", "-show_entries", "stream=width",
                 "-of", "csv=p=0", path])
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def _waveform(media_path: str, out_png: str, width: int) -> bool:
    """Waveform strip; False (not an exception) when the media has no usable audio."""
    ffmpeg = _require_ffmpeg()
    proc = _run([ffmpeg, "-y", "-v", "error", "-i", media_path, "-filter_complex",
                 f"aformat=channel_layouts=mono,showwavespic=s={width}x{WAVE_HEIGHT}:colors=white",
                 "-frames:v", "1", out_png])
    return proc.returncode == 0 and os.path.isfile(out_png)


def compose_sheet(media_path: str, times_s: list[float], out_png: str, *,
                  waveform: bool = True) -> list[str]:
    """Extract one tile per timestamp, hstack them, optionally vstack a waveform. Returns notes."""
    if not times_s:
        raise PerceptionUnavailable("no timestamps to render")
    ffmpeg = _require_ffmpeg()
    notes: list[str] = []
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="filmgrip-frames-") as tmp:
        tiles = _extract_frames_batch(media_path, times_s, tmp)
        strip = os.path.join(tmp, "strip.png")
        if len(tiles) == 1:
            shutil.copyfile(tiles[0], strip)
        else:
            inputs: list[str] = []
            for tile in tiles:
                inputs += ["-i", tile]
            chain = "".join(f"[{i}]" for i in range(len(tiles)))
            proc = _run([ffmpeg, "-y", "-v", "error", *inputs, "-filter_complex",
                         f"{chain}hstack=inputs={len(tiles)}", strip])
            if proc.returncode != 0:
                raise PerceptionUnavailable(
                    f"ffmpeg hstack failed: {proc.stderr.strip()[:200]}")
        if waveform:
            width = _png_width(strip)
            wave = os.path.join(tmp, "wave.png")
            if width and _waveform(media_path, wave, width):
                proc = _run([ffmpeg, "-y", "-v", "error", "-i", strip, "-i", wave,
                             "-filter_complex", "[0][1]vstack", out_png])
                if proc.returncode != 0:
                    raise PerceptionUnavailable(
                        f"ffmpeg vstack failed: {proc.stderr.strip()[:200]}")
                return notes
            notes.append("no usable audio stream — waveform omitted")
        shutil.copyfile(strip, out_png)
    return notes


# --------------------------------------------------------------------------- timeline-aware
def media_time_at(clip: Clip, timeline_frame: int, rate: float) -> float:
    """Media seconds playing at an absolute timeline frame inside ``clip`` (no retime)."""
    src_rate = rate
    sr = getattr(clip.otio, "source_range", None)
    if sr is not None and float(sr.start_time.rate) > 0:
        src_rate = float(sr.start_time.rate)
    return clip.source_start / src_rate + (timeline_frame - clip.start) / rate


def _clip_at(ir: TimelineIR, frame: int) -> Optional[Clip]:
    """The topmost video clip under ``frame`` (lowest track index wins, like a viewer)."""
    hits = [c for c in ir.real_clips()
            if c.track_kind == "video" and c.start <= frame < c.end]
    return min(hits, key=lambda c: c.track_index) if hits else None


def _sheet_path(key: str) -> str:
    return os.path.join(frames_dir(), f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.png")


def _source_sig(path: str) -> str:
    """Source-content signature (size + mtime_ns) for cache keys.

    Folding this into the sheet cache key means a changed source file invalidates its cached sheet —
    without it, editing/re-rendering the media silently served a stale picture as fresh evidence.
    """
    try:
        st = os.stat(path)
    except OSError:
        return "nosig"
    return f"{st.st_size}:{st.st_mtime_ns}"


def clip_sheet(ir: TimelineIR, clip_id: str, *, count: int = 8,
               waveform: bool = True) -> SheetResult:
    """Evenly-sampled contact sheet across one clip's source window."""
    clip = ir.clip(clip_id)
    if clip is None or clip.kind != "clip":
        raise PerceptionUnavailable(f"{clip_id}: not a clip on this timeline")
    if is_retimed(clip):
        raise PerceptionUnavailable(f"{clip_id}: retimed clip — frame mapping would be wrong")
    path = media_path_of(clip)
    if path is None or not os.path.isfile(path):
        raise PerceptionUnavailable(
            f"{clip_id}: source media not on disk ({path or 'no file URL'})")
    count = max(1, min(count, 24))
    step = clip.duration / count
    frames = [clip.start + int(round(step * (i + 0.5))) for i in range(count)]
    frames = [min(max(f, clip.start), clip.end - 1) for f in frames]
    times = [media_time_at(clip, f, ir.rate) for f in frames]
    out = _sheet_path(
        f"clip|{path}|{_source_sig(path)}|{clip_id}|{count}|{clip.start}|{clip.duration}")
    notes = compose_sheet(path, times, out, waveform=waveform)
    legend = [{"tile": i, "timeline_frame": f, "media_s": round(t, 3), "clip_id": clip_id}
              for i, (f, t) in enumerate(zip(frames, times))]
    return SheetResult(png_path=out, legend=legend, notes=notes)


def timeline_sheet(ir: TimelineIR, timeline_frames: list[int], *,
                   waveform: bool = True) -> tuple[list[SheetResult], list[str]]:
    """Contact sheets for exact timeline frames, grouped per source media file.

    Frames over gaps (nothing playing) or retimed/offline clips come back in ``errors``.
    """
    errors: list[str] = []
    groups: dict[str, list[tuple[int, float, str]]] = {}
    for frame in timeline_frames:
        clip = _clip_at(ir, frame)
        if clip is None:
            errors.append(f"frame {frame}: no video clip playing there")
            continue
        if is_retimed(clip):
            errors.append(f"frame {frame}: clip {clip.id} is retimed — skipped")
            continue
        path = media_path_of(clip)
        if path is None or not os.path.isfile(path):
            errors.append(f"frame {frame}: source media not on disk for {clip.id}")
            continue
        groups.setdefault(path, []).append(
            (frame, media_time_at(clip, frame, ir.rate), clip.id))
    sheets: list[SheetResult] = []
    for path, entries in groups.items():
        entries.sort()
        out = _sheet_path("tl|" + path + "|" + _source_sig(path) + "|"
                          + ",".join(str(e[0]) for e in entries))
        try:
            notes = compose_sheet(path, [t for _, t, _ in entries], out, waveform=waveform)
        except PerceptionUnavailable as exc:
            errors.append(str(exc))
            continue
        sheets.append(SheetResult(
            png_path=out,
            legend=[{"tile": i, "timeline_frame": f, "media_s": round(t, 3), "clip_id": cid}
                    for i, (f, t, cid) in enumerate(entries)],
            notes=notes,
        ))
    return sheets, errors
