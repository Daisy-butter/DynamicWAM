"""DOMINO-compatible evaluation artifacts without in-loop video encoding.

Integration: keep a ``BufferedHeadVideo`` on the physics owner. In the existing
``finish_capture`` callback, append ONLY the already downloaded head RGB and
its frozen capture timestamps. After the episode clock stops, drain and append
any last pending capture, then call ``finalize``. No scene, camera, GPU tensor,
renderer, or thread is accepted by this module.

The main MP4 intentionally follows native DOMINO: one captured head image per
encoded frame, 10 fps, H.264/yuv420p/CRF23 with one FFmpeg encoder thread.
This is NOT a continuous recording at physical-time 10 Hz. The timestamp
sidecar states that distinction and preserves each frame's simulator time.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction
from typing import Any, Callable, Iterable

import numpy as np


NATIVE_VIDEO_FPS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeHeadVideoWriter:
    """Original DOMINO codec and cadence, with bounded CPU thread use.

    This class must only be instantiated after the episode clock has stopped.
    CPU encoding is limited to one thread to reduce interference with other
    evaluation workers. Output publication and verification belong to finalize.
    """

    def __init__(self, path: Path):
        self.path = path
        self._proc: subprocess.Popen | None = None
        self._shape = None
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("native head encoder requires HWC uint8 RGB")
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError("video frame shape changed")
        if self._proc is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required to encode evaluation MP4")
            height, width = frame.shape[:2]
            self._shape = frame.shape
            self._proc = subprocess.Popen([
                ffmpeg, "-y", "-loglevel", "error", "-filter_threads", "1",
                "-filter_complex_threads", "1", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", str(NATIVE_VIDEO_FPS),
                "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                "-threads", "1", "-crf", "23", str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close()  # Raise the more useful FFmpeg exit status/stderr.
            raise
        self.frames += 1

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.stdin = None
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"ffmpeg exited with status {proc.returncode}: {detail}")


def probe_head_video(path: Path) -> dict[str, Any]:
    """Decode/count every frame using ffprobe outside the physical-time loop."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to verify evaluation MP4")
    result = subprocess.run([
        ffprobe, "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=False)
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(f"ffprobe could not fully decode MP4: {result.stderr[-4000:]}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("MP4 must contain a readable video stream")
    stream = streams[0]
    return {"codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
            "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "frame_count": int(stream["nb_read_frames"])}


def _validate_probe(probe: dict[str, Any], width: int, height: int, count: int) -> bool:
    return (probe.get("codec") == "h264" and probe.get("pixel_format") == "yuv420p"
            and probe.get("width") == width and probe.get("height") == height
            and probe.get("frame_count") == count
            and abs(float(probe.get("fps", 0)) - NATIVE_VIDEO_FPS) < 1e-6)


def validate_video_artifacts(report: dict[str, Any]) -> bool:
    """Validate a previously committed video's hashes and timestamp contract.

    Hashing occurs only outside physical-time execution. The stored ffprobe
    record is bound to the exact encoded bytes by SHA256, so resumptions need
    not decode the entire video again. Missing/partial/corrupt artifacts make
    the episode incomplete; callers must not silently include its metrics.
    """
    try:
        path, sidecar = Path(report["video"]), Path(report["video_timestamps"])
        count = report["video_frames"]
        if (path.suffix.lower() != ".mp4" or path.name.endswith(".part.mp4")
                or not path.is_file() or not sidecar.is_file()
                or type(count) is not int or count < 1
                or report["video_fps"] != NATIVE_VIDEO_FPS
                or report["video_continuous_physics"] is not False):
            return False
        if (path.stat().st_size != report["video_bytes"] or report["video_bytes"] <= 0
                or sidecar.stat().st_size != report["video_timestamps_bytes"]
                or report["video_timestamps_bytes"] <= 0
                or _sha256(path) != report["video_sha256"]
                or _sha256(sidecar) != report["video_timestamps_sha256"]):
            return False
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (metadata["schema"] != "domino_native_head_video_timestamps_v1"
                or metadata["video"] != path.name
                or metadata["frame_count"] != count or metadata["fps"] != NATIVE_VIDEO_FPS
                or metadata["video_sha256"] != report["video_sha256"]
                or metadata["video_bytes"] != report["video_bytes"]
                or metadata["continuous_physics_video"] is not False
                or metadata["sampling"] != "existing_head_captures"
                or metadata["playback_timing"] != "native_frame_index"
                or len(metadata["frames"]) != count
                or metadata["probe"] != report["video_probe"]
                or not _validate_probe(metadata["probe"], metadata["width"], metadata["height"], count)):
            return False
        last_physics = -math.inf
        last_wall = -math.inf
        for index, frame in enumerate(metadata["frames"]):
            physics = _finite_time(frame["physics_time_s"], "physics_time_s")
            if frame["frame_index"] != index or physics < last_physics:
                return False
            last_physics = physics
            if frame["capture_wall_perf_s"] is not None:
                wall = _finite_time(frame["capture_wall_perf_s"], "capture_wall_perf_s")
                if wall < last_wall:
                    return False
                last_wall = wall
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return False


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _finite_time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class BufferedHeadVideo:
    """Retain owned CPU RGB copies; encode only after physics has finished.

    ``max_bytes`` bounds host memory without silently dropping frames. A buffer
    overflow is an artifact failure, not a valid episode with a missing video.
    append performs an RGB copy but no rendering, disk I/O, or encoding.
    """

    def __init__(self, *, max_bytes: int = 2 * 1024**3) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.buffered_bytes = 0
        self._frames: list[np.ndarray] = []
        self._timestamps: list[dict[str, Any]] = []
        self._shape: tuple[int, ...] | None = None
        self._finalized = False

    @property
    def frames(self) -> int:
        return len(self._timestamps)

    def append(self, rgb: np.ndarray, *, physics_time_s: float,
               capture_wall_perf_s: float | None = None) -> None:
        if self._finalized:
            raise RuntimeError("video has already been finalized")
        # Do not convert arbitrary objects: that could trigger GPU readback.
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise TypeError("video requires an existing CPU uint8 NumPy RGB array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError(f"expected nonempty HWC RGB, got {rgb.shape}")
        if self._shape is not None and rgb.shape != self._shape:
            raise ValueError("head camera image shape changed during the episode")
        physics = _finite_time(physics_time_s, "physics_time_s")
        wall = (None if capture_wall_perf_s is None
                else _finite_time(capture_wall_perf_s, "capture_wall_perf_s"))
        if self._timestamps:
            previous = self._timestamps[-1]
            if physics < previous["physics_time_s"]:
                raise ValueError("capture physics timestamps must be nondecreasing")
            if (wall is not None and previous["capture_wall_perf_s"] is not None
                    and wall < previous["capture_wall_perf_s"]):
                raise ValueError("capture wall timestamps must be nondecreasing")
        if self.buffered_bytes + rgb.nbytes > self.max_bytes:
            raise MemoryError("head video buffer limit exceeded; no frames were dropped")
        frame = np.array(rgb, dtype=np.uint8, order="C", copy=True)
        frame.setflags(write=False)
        self._shape = frame.shape
        self._frames.append(frame)
        self._timestamps.append({"frame_index": len(self._timestamps),
                                 "physics_time_s": physics,
                                 "capture_wall_perf_s": wall})
        self.buffered_bytes += frame.nbytes

    def finalize(self, path: Path | str, *, episode_end_physics_s: float | None = None,
                 writer_factory: Callable[[Path], Any] | None = None,
                 probe_fn: Callable[[Path], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Encode after terminal/clock stop; fail if any requested artifact fails.

        ``writer_factory`` defaults to the compatible single-thread encoder;
        ``probe_fn`` defaults to real ffprobe. Both are injectable in tests.
        A successful return means MP4 decoding, codec/size/cadence/frame-count
        checks, hashing, and timestamp publication all succeeded; only then
        should the caller mark the episode
        complete. Retry/archive incomplete episode artifacts as one unit.
        """
        if self._finalized:
            raise RuntimeError("video has already been finalized")
        if not self._frames:
            raise ValueError("cannot publish a required MP4 without captured frames")
        end = (None if episode_end_physics_s is None
               else _finite_time(episode_end_physics_s, "episode_end_physics_s"))
        if end is not None and end < self._timestamps[-1]["physics_time_s"]:
            raise ValueError("episode end precedes the final captured frame")
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("native episode video path must have .mp4 suffix")
        sidecar = path.with_name(path.stem + "_timestamps.json")
        temporary = path.with_name(path.stem + ".part.mp4")
        sidecar_temporary = sidecar.with_suffix(".json.part")
        if path.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite completed video artifacts: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if writer_factory is None:
            writer_factory = NativeHeadVideoWriter
        if probe_fn is None:
            probe_fn = probe_head_video
        metadata = {
            "schema": "domino_native_head_video_timestamps_v1",
            "video": path.name,
            "fps": NATIVE_VIDEO_FPS,
            "frame_count": self.frames,
            "height": self._shape[0],
            "width": self._shape[1],
            "sampling": "existing_head_captures",
            "playback_timing": "native_frame_index",
            "continuous_physics_video": False,
            "timing_note": ("One existing head-camera capture per encoded frame at native 10 fps. "
                            "Playback duration does not represent physical elapsed time; "
                            "no frames are interpolated, repeated, or additionally rendered."),
            "encoded_duration_s": self.frames / NATIVE_VIDEO_FPS,
            "first_capture_physics_s": self._timestamps[0]["physics_time_s"],
            "last_capture_physics_s": self._timestamps[-1]["physics_time_s"],
            "episode_end_physics_s": end,
            "capture_wall_clock": "time.perf_counter; monotonic process clock, not Unix time",
            "frames": list(self._timestamps),
        }
        writer = None
        try:
            writer = writer_factory(temporary)
            for frame in self._frames:
                writer.write(frame)
            writer.close()
            writer = None
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("video writer returned without a nonempty MP4")
            probe = probe_fn(temporary)
            if not _validate_probe(probe, self._shape[1], self._shape[0], self.frames):
                raise RuntimeError(f"encoded MP4 does not match the native frame contract: {probe}")
            metadata.update({"probe": probe, "video_sha256": _sha256(temporary),
                             "video_bytes": temporary.stat().st_size})
            sidecar_temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
            sidecar_temporary.replace(sidecar)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass  # Preserve the initial encoder failure.
            for incomplete in (temporary, sidecar_temporary):
                if incomplete.exists():
                    incomplete.unlink()
        self._finalized = True
        self._frames.clear()
        self.buffered_bytes = 0
        return {"video": str(path), "video_timestamps": str(sidecar),
                "video_frames": self.frames, "video_fps": NATIVE_VIDEO_FPS,
                "video_continuous_physics": False, "video_probe": metadata["probe"],
                "video_sha256": metadata["video_sha256"], "video_bytes": metadata["video_bytes"],
                "video_timestamps_sha256": _sha256(sidecar),
                "video_timestamps_bytes": sidecar.stat().st_size}


def aggregate_native_metrics(episodes: Iterable[dict[str, Any]], *,
                             metrics_module: Any = None) -> Any:
    """Rehydrate saved dataclass metrics using DOMINO's own metric classes.

    The caller supplies only successfully validated, unique episode records,
    in episode-index order. No score, penalty, success, or seed is recomputed.
    """
    if metrics_module is None:
        import eval_metrics as metrics_module
    aggregated = metrics_module.AggregatedMetrics()
    for record in episodes:
        restored = dict(record)
        restored["penalty_events"] = [metrics_module.PenaltyEvent(**event)
                                       for event in restored.get("penalty_events", [])]
        _fields = getattr(metrics_module.EpisodeMetrics, '__dataclass_fields__', None)
        if _fields is not None:
            restored = {k: v for k, v in restored.items() if k in _fields}
        aggregated.add_episode(metrics_module.EpisodeMetrics(**restored))
    return aggregated


def write_domino_results(save_dir: Path | str, stamp: str, instruction_type: str,
                         aggregated: Any) -> None:
    """Write the same four task-level files as native DOMINO eval_policy.py.

    Mirrors scripts/eval_domino_rt.py:_write_domino_results without importing
    the evaluator. _result.txt contains a fraction; _metrics.json contains SR
    in percent and MS on its native 0--100 scale. An empty aggregate is an
    incomplete task, not a zero-success completed evaluation.
    """
    episodes = aggregated.episodes
    if not episodes:
        raise ValueError("native result files require at least one completed episode")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    successes = sum(1 for episode in episodes if episode.success)
    rate = np.array([successes]) / len(episodes)
    summary = aggregated.get_summary()
    for key, value in list(summary.items()):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        summary[key] = value
    artifacts = {
        "_result.txt": (f"Timestamp: {stamp}\n\nInstruction Type: {instruction_type}\n\n"
                        + "\n".join(map(str, rate))),
        "_metrics_report.txt": aggregated.to_detailed_report(),
        "_metrics.json": json.dumps(summary, indent=2, default=_plain),
        "_episodes_detail.json": json.dumps(aggregated.get_all_episodes(), indent=2, default=_plain),
    }
    for name, content in artifacts.items():
        destination = save_dir / name
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
