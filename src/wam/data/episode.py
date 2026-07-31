"""Episode format writer/reader (T-07, PRD Anhang A, FR-01/FR-08).

On-disk layout (flat, one directory per episode):

    <episode_dir>/
      manifest.json               versioned index: spec, cameras, tables, checksums, stats
      <camera>.mp4                one lossy video stream per camera (av/x264, cv2 fallback)
      <camera>_timestamps.parquet per-frame capture timestamps (mp4 pts are NOT sufficient
                                  for sync analysis)
      states.parquet              canonical RobotState rows
      actions.parquet             ActionChunk steps flattened via chunk_idx/step_idx

Contracts:
- Streaming writer: frames are encoded to disk as they arrive and are never all held in RAM.
  State/action rows are buffered as small plain rows and flushed on ``close()``.
- Exact roundtrip for states/actions (float32 bit-exact via parquet); video roundtrip is
  structural only (shape/dtype exact, pixel values within lossy-codec tolerance).
- ``manifest.json`` carries sha256 checksums of every data file; the reader verifies them on
  open (opt-out via ``verify_checksums=False``). The manifest itself is not checksummed here;
  immutable dataset snapshots are a dataset-level concern (T-11).
- ``max_sync_error_ns`` is the maximum spread between camera timestamps at the same frame
  index across cameras (0 with fewer than two cameras).
- Versioned: ``EPISODE_FORMAT_VERSION`` major must match on read, as must the schema major.
- Torch-free; numpy + pyarrow (+ av or opencv for video).
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from wam.interfaces.schema import (
    SCHEMA_VERSION,
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    NormalizationSpec,
    RobotState,
    ValidityMask,
)

EPISODE_FORMAT_VERSION = "0.1.0"

MANIFEST_FILENAME = "manifest.json"
STATES_TABLE = "states"
ACTIONS_TABLE = "actions"

_CAMERA_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

_F32_LIST = pa.list_(pa.float32())

_STATES_SCHEMA = pa.schema(
    [
        ("timestamp_ns", pa.int64()),
        ("q", _F32_LIST),
        ("dq", _F32_LIST),
        ("imu_orientation_wxyz", _F32_LIST),
        ("imu_angular_velocity", _F32_LIST),
        ("imu_linear_acceleration", _F32_LIST),
        ("gripper_state", _F32_LIST),
        ("valid_q", pa.bool_()),
        ("valid_dq", pa.bool_()),
        ("valid_imu", pa.bool_()),
        ("valid_gripper", pa.bool_()),
        ("schema_version", pa.string()),
    ]
)

_ACTIONS_SCHEMA = pa.schema(
    [
        ("chunk_idx", pa.int64()),
        ("step_idx", pa.int64()),
        ("timestamp_ns", pa.int64()),
        ("executed_prefix", pa.int64()),
        ("mode", pa.string()),
        ("dt_s", pa.float64()),
        ("targets", _F32_LIST),
        ("gripper_target", pa.float32()),
        ("schema_version", pa.string()),
    ]
)

_TIMESTAMPS_SCHEMA = pa.schema([("frame_idx", pa.int64()), ("timestamp_ns", pa.int64())])


class EpisodeFormatError(ValueError):
    """Episode directory violates the format contract (version, structure, consistency)."""


class EpisodeChecksumError(EpisodeFormatError):
    """A file's sha256 does not match the manifest (missing or tampered data)."""


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _f32_row(name: str, arr: Any) -> list[float]:
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError(f"{name}: expected 1-D np.ndarray, got {type(arr).__name__}")
    return np.asarray(arr, dtype=np.float32).tolist()


# -- video backends ------------------------------------------------------------------------

_AV_AVAILABLE: bool | None = None


def _av_available() -> bool:
    global _AV_AVAILABLE
    if _AV_AVAILABLE is None:
        try:
            import av  # noqa: F401

            _AV_AVAILABLE = True
        except ImportError:
            _AV_AVAILABLE = False
    return _AV_AVAILABLE


class _AvVideoWriter:
    """Streams RGB uint8 frames into an mp4 via PyAV (libx264, mpeg4 fallback)."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        import av

        self._av = av
        rate = Fraction(fps).limit_denominator(65535)
        self._container = av.open(str(path), mode="w")
        codec = "libx264"
        try:
            stream = self._container.add_stream(codec, rate=rate)
        except av.codec.codec.UnknownCodecError:  # codec not in this ffmpeg build
            codec = "mpeg4"
            stream = self._container.add_stream(codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "18", "preset": "veryfast"}
        else:
            stream.bit_rate = 4_000_000
        self._stream = stream
        self.codec = codec
        self._pts = 0

    def write(self, img: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        frame.pts = self._pts
        self._pts += 1
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None


class _CvVideoWriter:
    """OpenCV fallback writer (mp4v). Used only when PyAV is unavailable."""

    codec = "mp4v"

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {path}")

    def write(self, img: np.ndarray) -> None:
        self._writer.write(np.ascontiguousarray(img[..., ::-1]))  # RGB -> BGR

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None


def _new_video_writer(
    path: Path, fps: float, width: int, height: int
) -> _AvVideoWriter | _CvVideoWriter:
    if _av_available():
        return _AvVideoWriter(path, fps, width, height)
    return _CvVideoWriter(path, fps, width, height)


def _decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 to a uint8 [n, H, W, 3] RGB array."""
    frames: list[np.ndarray] = []
    if _av_available():
        import av

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
    else:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(np.ascontiguousarray(bgr[..., ::-1]))
        finally:
            cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8, copy=False)


# -- manifest -------------------------------------------------------------------------------


class CameraStreamInfo(BaseModel):
    """One recorded camera stream."""

    model_config = ConfigDict(frozen=True)

    file: str
    codec: str
    fps: float = Field(gt=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    num_frames: int = Field(ge=0)


class TableInfo(BaseModel):
    """One recorded parquet table."""

    model_config = ConfigDict(frozen=True)

    file: str
    num_rows: int = Field(ge=0)


class EpisodeManifest(BaseModel):
    """Versioned index of one episode directory (PRD Anhang A ``manifest.json``)."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    format_version: str = EPISODE_FORMAT_VERSION
    schema_version: str = SCHEMA_VERSION
    spec: CanonicalSpaceSpec
    cameras: dict[str, CameraStreamInfo] = Field(default_factory=dict)
    tables: dict[str, TableInfo] = Field(default_factory=dict)
    normalization: dict[str, dict[str, Any]] | None = None
    instruction: str = ""
    checksums: dict[str, str] = Field(default_factory=dict)
    t0_ns: int = 0
    t1_ns: int = 0
    max_sync_error_ns: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    def normalization_specs(self) -> dict[str, NormalizationSpec] | None:
        """Parse the stored normalization dicts back into ``NormalizationSpec`` objects."""
        if self.normalization is None:
            return None
        return {k: NormalizationSpec.from_dict(v) for k, v in self.normalization.items()}


# -- writer ---------------------------------------------------------------------------------


class EpisodeWriter:
    """Streaming episode recorder; use as a context manager.

    Frames are encoded to ``<camera>.mp4`` immediately (never all in RAM). ``close()`` writes
    the parquet tables and ``manifest.json`` (with sha256 checksums of every data file).
    On an exception inside the ``with`` block, resources are released but NO manifest is
    written — the partial directory is invisible to ``list_episodes``.

    ``normalization`` specs are STORED in the manifest as provenance only — nothing in the
    MVP pipeline applies them (identity normalization end-to-end; ``EpisodeDataset`` rejects
    non-identity target specs rather than silently ignoring them).
    """

    def __init__(
        self,
        dir: str | Path,
        episode_id: str,
        spec: CanonicalSpaceSpec,
        fps: float,
        instruction: str,
        *,
        normalization: dict[str, NormalizationSpec | dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._dir = Path(dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        if (self._dir / MANIFEST_FILENAME).exists():
            raise FileExistsError(f"episode already recorded at {self._dir}")
        self._episode_id = str(episode_id)
        self._spec = spec
        self._fps = float(fps)
        self._instruction = str(instruction)
        self._normalization: dict[str, dict[str, Any]] | None = None
        if normalization is not None:
            self._normalization = {
                key: (ns if isinstance(ns, NormalizationSpec) else NormalizationSpec.from_dict(ns))
                .to_dict()
                for key, ns in normalization.items()
            }
        self._extra = dict(extra) if extra else {}

        self._video_writers: dict[str, _AvVideoWriter | _CvVideoWriter] = {}
        self._camera_wh: dict[str, tuple[int, int]] = {}
        self._frame_ts: dict[str, list[int]] = {}
        self._state_rows: list[dict[str, Any]] = []
        self._action_rows: list[dict[str, Any]] = []
        self._num_chunks = 0
        self._t_min: int | None = None
        self._t_max: int | None = None
        self._closed = False
        self._manifest: EpisodeManifest | None = None

    # -- context manager ---------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._release()

    # -- recording ---------------------------------------------------------------------

    def add_frame(self, camera: str, img: np.ndarray, timestamp_ns: int) -> None:
        """Append one RGB frame (uint8 [H, W, 3], even H/W) to ``<camera>.mp4``."""
        self._check_open()
        if not _CAMERA_NAME_RE.fullmatch(camera):
            raise ValueError(f"invalid camera name {camera!r}")
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
            raise ValueError("img: expected uint8 np.ndarray of shape [H, W, 3]")
        h, w, c = img.shape
        if c != 3:
            raise ValueError(f"img: expected 3 channels, got {c}")
        if h % 2 or w % 2:
            raise ValueError(f"img: H and W must be even for yuv420p encoding, got {h}x{w}")
        writer = self._video_writers.get(camera)
        if writer is None:
            writer = _new_video_writer(self._dir / f"{camera}.mp4", self._fps, w, h)
            self._video_writers[camera] = writer
            self._camera_wh[camera] = (w, h)
            self._frame_ts[camera] = []
        elif self._camera_wh[camera] != (w, h):
            ew, eh = self._camera_wh[camera]
            raise ValueError(f"{camera}: frame size changed from {ew}x{eh} to {w}x{h}")
        ts = int(timestamp_ns)
        prev = self._frame_ts[camera]
        if prev and ts < prev[-1]:
            raise ValueError(f"{camera}: timestamps must be non-decreasing ({ts} < {prev[-1]})")
        writer.write(img)
        prev.append(ts)
        self._note_ts(ts)

    def add_state(self, state: RobotState) -> None:
        """Append one canonical robot state row (converted immediately, not referenced)."""
        self._check_open()
        row = {
            "timestamp_ns": int(state.timestamp_ns),
            "q": _f32_row("q", state.q),
            "dq": _f32_row("dq", state.dq),
            "imu_orientation_wxyz": _f32_row("imu.orientation_wxyz", state.imu.orientation_wxyz),
            "imu_angular_velocity": _f32_row("imu.angular_velocity", state.imu.angular_velocity),
            "imu_linear_acceleration": _f32_row(
                "imu.linear_acceleration", state.imu.linear_acceleration
            ),
            "gripper_state": _f32_row("gripper_state", state.gripper_state),
            "valid_q": bool(state.validity.q),
            "valid_dq": bool(state.validity.dq),
            "valid_imu": bool(state.validity.imu),
            "valid_gripper": bool(state.validity.gripper),
            "schema_version": str(state.schema_version),
        }
        self._state_rows.append(row)
        self._note_ts(row["timestamp_ns"])

    def add_action(self, commanded: ActionChunk, executed_prefix: int, timestamp_ns: int) -> None:
        """Append one commanded chunk, flattened to one parquet row per step."""
        self._check_open()
        if not isinstance(commanded.mode, ActionMode):
            raise TypeError(f"mode: expected ActionMode, got {type(commanded.mode).__name__}")
        if not isinstance(commanded.targets, np.ndarray) or commanded.targets.ndim != 2:
            raise ValueError("targets: expected np.ndarray of shape [T, D]")
        num_steps = commanded.num_steps
        if not 0 <= executed_prefix <= num_steps:
            raise ValueError(f"executed_prefix must be in [0, {num_steps}], got {executed_prefix}")
        targets = np.asarray(commanded.targets, dtype=np.float32)
        gripper = np.asarray(commanded.gripper_target, dtype=np.float32)
        if gripper.shape != (num_steps,):
            raise ValueError(
                f"gripper_target: expected shape ({num_steps},), got {gripper.shape}"
            )
        ts = int(timestamp_ns)
        chunk_idx = self._num_chunks
        self._num_chunks += 1
        for step in range(num_steps):
            self._action_rows.append(
                {
                    "chunk_idx": chunk_idx,
                    "step_idx": step,
                    "timestamp_ns": ts,
                    "executed_prefix": int(executed_prefix),
                    "mode": commanded.mode.value,
                    "dt_s": float(commanded.dt_s),
                    "targets": targets[step].tolist(),
                    "gripper_target": float(gripper[step]),
                    "schema_version": str(commanded.schema_version),
                }
            )
        self._note_ts(ts)

    # -- finalization ------------------------------------------------------------------

    def close(self) -> EpisodeManifest:
        """Flush videos, write parquet tables + manifest.json. Idempotent."""
        if self._closed:
            assert self._manifest is not None
            return self._manifest
        self._closed = True
        for writer in self._video_writers.values():
            writer.close()

        files: list[str] = [f"{cam}.mp4" for cam in self._video_writers]
        tables: dict[str, TableInfo] = {}
        for cam, ts_list in self._frame_ts.items():
            name = f"{cam}_timestamps"
            fname = f"{name}.parquet"
            rows = [{"frame_idx": i, "timestamp_ns": t} for i, t in enumerate(ts_list)]
            pq.write_table(
                pa.Table.from_pylist(rows, schema=_TIMESTAMPS_SCHEMA), self._dir / fname
            )
            tables[name] = TableInfo(file=fname, num_rows=len(rows))
            files.append(fname)

        pq.write_table(
            pa.Table.from_pylist(self._state_rows, schema=_STATES_SCHEMA),
            self._dir / "states.parquet",
        )
        tables[STATES_TABLE] = TableInfo(file="states.parquet", num_rows=len(self._state_rows))
        files.append("states.parquet")

        pq.write_table(
            pa.Table.from_pylist(self._action_rows, schema=_ACTIONS_SCHEMA),
            self._dir / "actions.parquet",
        )
        tables[ACTIONS_TABLE] = TableInfo(file="actions.parquet", num_rows=len(self._action_rows))
        files.append("actions.parquet")

        cameras = {
            cam: CameraStreamInfo(
                file=f"{cam}.mp4",
                codec=self._video_writers[cam].codec,
                fps=self._fps,
                width=self._camera_wh[cam][0],
                height=self._camera_wh[cam][1],
                num_frames=len(self._frame_ts[cam]),
            )
            for cam in self._video_writers
        }

        manifest = EpisodeManifest(
            episode_id=self._episode_id,
            spec=self._spec,
            cameras=cameras,
            tables=tables,
            normalization=self._normalization,
            instruction=self._instruction,
            checksums={fname: _sha256(self._dir / fname) for fname in files},
            t0_ns=self._t_min if self._t_min is not None else 0,
            t1_ns=self._t_max if self._t_max is not None else 0,
            max_sync_error_ns=_max_sync_error_ns(self._frame_ts),
            extra=self._extra,
        )
        payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        (self._dir / MANIFEST_FILENAME).write_text(payload + "\n")
        self._manifest = manifest
        return manifest

    # -- internals ---------------------------------------------------------------------

    def _release(self) -> None:
        """Close encoders without writing tables/manifest (exception path)."""
        if self._closed:
            return
        self._closed = True
        for writer in self._video_writers.values():
            writer.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("EpisodeWriter is closed")

    def _note_ts(self, ts: int) -> None:
        self._t_min = ts if self._t_min is None else min(self._t_min, ts)
        self._t_max = ts if self._t_max is None else max(self._t_max, ts)


def _max_sync_error_ns(frame_ts: dict[str, list[int]]) -> int:
    """Max spread between camera timestamps at the same frame index (0 if < 2 cameras)."""
    series = [ts for ts in frame_ts.values() if ts]
    if len(series) < 2:
        return 0
    worst = 0
    for values in zip(*series):
        worst = max(worst, max(values) - min(values))
    return int(worst)


# -- reader ---------------------------------------------------------------------------------


class EpisodeReader:
    """Reads one episode directory; verifies manifest version + checksums on open."""

    def __init__(self, dir: str | Path, *, verify_checksums: bool = True) -> None:
        self._dir = Path(dir)
        manifest_path = self._dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_FILENAME} in {self._dir}")
        manifest = EpisodeManifest.model_validate(json.loads(manifest_path.read_text()))
        if _major(manifest.format_version) != _major(EPISODE_FORMAT_VERSION):
            raise EpisodeFormatError(
                f"incompatible episode format_version {manifest.format_version!r}, "
                f"expected major {_major(EPISODE_FORMAT_VERSION)}.x.x"
            )
        if _major(manifest.schema_version) != _major(SCHEMA_VERSION):
            raise EpisodeFormatError(
                f"incompatible schema_version {manifest.schema_version!r}, "
                f"expected major {_major(SCHEMA_VERSION)}.x.x"
            )
        if verify_checksums:
            for fname, expected in manifest.checksums.items():
                path = self._dir / fname
                if not path.is_file():
                    raise EpisodeChecksumError(f"missing file {fname!r} in {self._dir}")
                actual = _sha256(path)
                if actual != expected:
                    raise EpisodeChecksumError(
                        f"checksum mismatch for {fname!r}: expected {expected}, got {actual}"
                    )
        self._manifest = manifest

    @property
    def manifest(self) -> EpisodeManifest:
        return self._manifest

    # -- tables ------------------------------------------------------------------------

    def read_states(self) -> list[RobotState]:
        """Reconstruct all recorded states; float32 arrays are bit-exact."""
        data = pq.read_table(self._table_path(STATES_TABLE)).to_pydict()
        states: list[RobotState] = []
        for i in range(len(data["timestamp_ns"])):
            states.append(
                RobotState(
                    timestamp_ns=int(data["timestamp_ns"][i]),
                    q=np.asarray(data["q"][i], dtype=np.float32),
                    dq=np.asarray(data["dq"][i], dtype=np.float32),
                    imu=IMUState(
                        orientation_wxyz=np.asarray(
                            data["imu_orientation_wxyz"][i], dtype=np.float32
                        ),
                        angular_velocity=np.asarray(
                            data["imu_angular_velocity"][i], dtype=np.float32
                        ),
                        linear_acceleration=np.asarray(
                            data["imu_linear_acceleration"][i], dtype=np.float32
                        ),
                    ),
                    gripper_state=np.asarray(data["gripper_state"][i], dtype=np.float32),
                    validity=ValidityMask(
                        q=bool(data["valid_q"][i]),
                        dq=bool(data["valid_dq"][i]),
                        imu=bool(data["valid_imu"][i]),
                        gripper=bool(data["valid_gripper"][i]),
                    ),
                    schema_version=str(data["schema_version"][i]),
                )
            )
        return states

    def read_actions(self) -> list[tuple[ActionChunk, int, int]]:
        """Reconstruct commanded chunks as ``(chunk, executed_prefix, timestamp_ns)``."""
        data = pq.read_table(self._table_path(ACTIONS_TABLE)).to_pydict()
        num_rows = len(data["chunk_idx"])
        groups: dict[int, list[int]] = {}
        for row in range(num_rows):
            groups.setdefault(int(data["chunk_idx"][row]), []).append(row)
        chunks: list[tuple[ActionChunk, int, int]] = []
        for chunk_idx in sorted(groups):
            rows = groups[chunk_idx]
            steps = [int(data["step_idx"][r]) for r in rows]
            if steps != list(range(len(rows))):
                raise EpisodeFormatError(
                    f"actions table: chunk {chunk_idx} has non-contiguous step_idx {steps}"
                )
            first = rows[0]
            chunk = ActionChunk(
                mode=ActionMode(data["mode"][first]),
                targets=np.asarray([data["targets"][r] for r in rows], dtype=np.float32),
                gripper_target=np.asarray(
                    [data["gripper_target"][r] for r in rows], dtype=np.float32
                ),
                dt_s=float(data["dt_s"][first]),
                schema_version=str(data["schema_version"][first]),
            )
            chunks.append((chunk, int(data["executed_prefix"][first]), int(data["timestamp_ns"][first])))
        return chunks

    # -- video -------------------------------------------------------------------------

    def read_frames(self, camera: str) -> np.ndarray:
        """Decode ``<camera>.mp4`` to uint8 [n, H, W, 3] (RGB). Lossy but structural-exact."""
        info = self._camera_info(camera)
        frames = _decode_video(self._dir / info.file)
        if frames.shape[0] != info.num_frames:
            raise EpisodeFormatError(
                f"{camera}: decoded {frames.shape[0]} frames, manifest says {info.num_frames}"
            )
        if info.num_frames and frames.shape[1:] != (info.height, info.width, 3):
            raise EpisodeFormatError(
                f"{camera}: decoded shape {frames.shape[1:]}, manifest says "
                f"({info.height}, {info.width}, 3)"
            )
        return frames

    def frame_timestamps(self, camera: str) -> np.ndarray:
        """Per-frame capture timestamps (int64 [n]) from the parquet sidecar."""
        self._camera_info(camera)
        data = pq.read_table(self._table_path(f"{camera}_timestamps")).to_pydict()
        return np.asarray(data["timestamp_ns"], dtype=np.int64)

    # -- internals ---------------------------------------------------------------------

    def _camera_info(self, camera: str) -> CameraStreamInfo:
        info = self._manifest.cameras.get(camera)
        if info is None:
            raise KeyError(f"unknown camera {camera!r}; have {sorted(self._manifest.cameras)}")
        return info

    def _table_path(self, name: str) -> Path:
        info = self._manifest.tables.get(name)
        if info is None:
            raise EpisodeFormatError(f"manifest has no table {name!r}")
        return self._dir / info.file


def frame_window_indices(frame_idx: int, num_frames: int, num_available: int) -> np.ndarray:
    """The ``num_frames`` frame indices a sample at ``frame_idx`` sees, oldest first.

    **The single definition of "which frames does the model get".** It exists as one function
    because it used to exist as two: the training dataset selected the real window ending at the
    chunk while ``predict()`` tiled a single frame, and nobody noticed until a world-action model
    had been scored on freeze-frames (T-29, ``docs/improvements.md`` I-7). Training and evaluation
    now call this, so the two cannot drift apart again without changing one line.

    The window ends **at** ``frame_idx`` (inclusive) and reaches backwards. Near the start of an
    episode it is clamped, i.e. the earliest frame repeats — a real observation stream has no
    frames before its first one either, and clamping is what a rolling buffer does at startup.
    The last element is always ``frame_idx``.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if num_available < 1:
        raise ValueError(f"num_available must be >= 1, got {num_available}")
    lo = frame_idx - num_frames + 1
    return np.clip(np.arange(lo, frame_idx + 1), 0, num_available - 1)


def list_episodes(root_dir: str | Path) -> list[Path]:
    """All episode directories (containing ``manifest.json``) under ``root_dir``, sorted."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob(MANIFEST_FILENAME))
