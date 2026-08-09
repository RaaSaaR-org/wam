#!/usr/bin/env python3
"""Lay LeRobot v2.1 episodes out as the clip tree the Cosmos3 SFT captioner consumes (PR-09 §3).

This script produces **no captions and no jsonl**. Both of those are NVIDIA's:
``cosmos_framework.scripts.caption_from_video`` writes ``captions/<uuid>/caption.json`` and
``cosmos_framework.scripts.captions_to_sft_jsonl`` turns the pair of directories into
``video_dataset_file.jsonl``. Our job is the part they do not do — pick the episodes, split them,
name them stably, and record where every clip came from.

Output layout, which is what ``captions_to_sft_jsonl`` expects (``docs/dataset_jsonl.md``)::

    <out>/
      train/videos/<uuid>.mp4
      val/videos/<uuid>.mp4
      manifest.json          <- ours, provenance. Not read by the framework.

``DATASET_PATH`` for the trainer is then ``<out>``, because the framework resolves
``<DATASET_PATH>/train/video_dataset_file.jsonl``.

Three things this does that a ``cp -r`` does not:

1. **Mirrors the loader's silent filters.** ``sft_dataset.py`` drops clips longer than 61.0 s and
   windows shorter than 61 frames, without saying so. Applying them here means the manifest's
   episode count is the count that trains, rather than a number that is quietly larger than what
   the run consumed. Both bounds are the framework's, not ours; ``--num-video-frames`` matches the
   recipe's window setting and raises the lower bound the same way the framework's converter does.

2. **Splits before anything is generated.** PR-09 §5 requires the eval prompts to come from
   episodes the SFT never saw. The split is seeded and written into the manifest, so "held out" is
   checkable afterwards rather than asserted.

3. **Records provenance per clip.** AC-04: every rollout traceable to checkpoint + dataset snapshot
   + config hash. The manifest carries the source repo id, the resolved revision, and a sha256 per
   copied mp4 — so a corpus can be re-derived, and a corpus that was silently rebuilt is detectable.

Source resolution is preserved. ``datasets/gr00t-apple-full`` is 120x160 and is NOT usable here:
the generator trains on pixels, and our converted copy threw them away.

Usage::

    python scripts/prepare_cosmos_corpus.py \
        --source ~/.cache/huggingface/.../GR00T-N1.7-AppleToPlate \
        --source ~/.cache/huggingface/.../G1_Dex3_ToastedBread_Dataset \
        --out /valhalla/.../data/cosmos-g1-embodiment \
        --val-episodes 30 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

# Both bounds belong to cosmos-framework's sft_dataset.py, mirrored here so the manifest count is
# the count that trains. Changing either without changing it there makes the manifest lie.
MAX_CLIP_SECONDS = 61.0
MIN_WINDOW_FRAMES = 61


@dataclass(frozen=True)
class Clip:
    uuid: str
    source_id: str
    episode_index: int
    src_path: str
    frames: int
    fps: float
    duration_s: float
    width: int
    height: int
    task: str
    #: Window inside ``src_path``, in seconds. ``None`` means "the whole file", which is always the
    #: case for v2.1 where one mp4 *is* one episode. v3.0 concatenates many episodes into one file,
    #: so there the window is the only thing separating one episode from its neighbours.
    from_s: float | None = None
    to_s: float | None = None
    #: Codec of the source, as declared by the dataset metadata. Recorded because it is the field
    #: that explains why a corpus did or did not need transcoding.
    src_codec: str = ""
    #: sha256 of the source bytes — provenance, survives the output being re-encoded.
    src_sha256: str = ""
    #: sha256 of the file that actually trains. Equal to ``src_sha256`` when symlinked or copied.
    sha256: str = ""


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def video_keys(info: dict) -> list[str]:
    """Video feature keys in a LeRobot v2.1 ``meta/info.json``, in declaration order."""
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def resolve_camera(info: dict, requested: str | None, source_id: str) -> str:
    keys = video_keys(info)
    if not keys:
        raise SystemExit(f"FATAL: {source_id} declares no video features — nothing to train on.")
    if requested:
        if requested not in keys:
            raise SystemExit(
                f"FATAL: camera {requested!r} not in {source_id}: {keys}"
            )
        return requested
    if len(keys) > 1:
        # Never guess. A generator trained on the wrong view is a plausible, finite, wrong run.
        raise SystemExit(
            f"FATAL: {source_id} has {len(keys)} cameras {keys} — pass --camera-key to choose."
        )
    return keys[0]


def video_shape(info: dict, key: str) -> tuple[int, int]:
    """(width, height) for a video feature.

    ``shape`` is NOT reliably [H, W, C]. Half our sources write [C, H, W], and reading index 1 as
    the width then yields (480, 3) — a plausible-looking pair that is silently wrong, which went
    into 1712 of 3462 manifest entries before anyone looked. The manifest is the provenance record
    AC-04 depends on, so guessing by position is not good enough.

    Order of preference: the probed stream info (every one of our 14 sources supplies it and it
    comes from the file itself), then the declared axis ``names``, and only then ``shape`` — and
    then only by locating the channel axis rather than assuming where it sits.
    """
    feature = info["features"][key]
    probed = feature.get("info") or {}
    w, h = int(probed.get("video.width") or 0), int(probed.get("video.height") or 0)
    if w and h:
        return (w, h)

    shape = [int(x) for x in (feature.get("shape") or [])]
    names = [str(n).lower() for n in (feature.get("names") or [])]
    if len(names) == len(shape) and "height" in names and "width" in names:
        return (shape[names.index("width")], shape[names.index("height")])

    if len(shape) == 3:
        # No names and no probe: find the channel axis (3 or 1) and take the other two in order.
        chan = next((i for i, v in enumerate(shape) if v in (1, 3)), None)
        if chan is not None:
            hw = [v for i, v in enumerate(shape) if i != chan]
            return (hw[1], hw[0])
    if len(shape) >= 2:
        return (shape[1], shape[0])
    return (0, 0)


#: The two LeRobot layouts this reads, and they are genuinely different formats rather than
#: versions of one:
#:
#: **v2.1** — one mp4 per episode at ``videos/chunk-NNN/<key>/episode_NNNNNN.mp4``, one JSON object
#: per episode in ``meta/episodes.jsonl``. A clip is a file.
#:
#: **v3.0** — episodes are CONCATENATED into a few large mp4s at
#: ``videos/<key>/chunk-NNN/file-NNN.mp4`` (301 episodes of G1_Dex3_BlockStacking live in 6 of them
#: for one camera), with the per-episode boundaries in ``meta/episodes/chunk-NNN/file-NNN.parquet``
#: as ``videos/<key>/from_timestamp`` / ``to_timestamp``. A clip is a *window*, and getting it out
#: is an extraction pass, not a path fix.
#:
#: Anything else is refused by name. Jobs 186353/186354 died on a FileNotFoundError pointing at
#: episodes.jsonl, which reads like a broken download rather than an unsupported format, and cost
#: real time pointed at the fetch step.
_SUPPORTED_CODEBASES = ("v2.1", "v3.0")
#: Kept as a name because tests and readers reach for "the one this was written for".
_SUPPORTED_CODEBASE = "v2.1"


def require_supported(info: dict, source_id: str) -> str:
    version = str(info.get("codebase_version") or "unknown")
    if version not in _SUPPORTED_CODEBASES:
        raise SystemExit(
            f"FATAL: {source_id} is LeRobot {version}; this script reads "
            f"{' and '.join(_SUPPORTED_CODEBASES)}.\n"
            "       The two known layouts differ in where episode boundaries live: v2.1 gives one\n"
            "       mp4 per episode, v3.0 concatenates them and records each episode's timestamp\n"
            "       window in meta/episodes/*/*.parquet. An unrecognised version is neither, and\n"
            "       guessing from the directory layout is how a corpus silently becomes one\n"
            "       episode repeated. The download is fine; the format is not one we can read."
        )
    return version


def episode_video_path(root: Path, key: str, episode_index: int, chunk_size: int) -> Path:
    """v2.1 only: the mp4 that *is* episode ``episode_index``."""
    chunk = episode_index // max(chunk_size, 1)
    return root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"


#: v3.0's template, from ``info.json["video_path"]``. Hard-coded only as the fallback for a file
#: that omits it; the dataset's own value wins.
DEFAULT_V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _episode_rows_v30(root: Path, key: str, source_id: str) -> list[dict]:
    """Read every ``meta/episodes/*/*.parquet``, keeping only the columns a clip needs.

    Globbed, not indexed: LeRobot splits the episode metadata across as many parquets as it needs
    and nothing says there is exactly one.

    Only seven columns are pulled out of roughly eighty. The rest are per-feature ``stats/*``
    aggregates — reading them would multiply the cost of this scan for data no clip uses.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            f"FATAL: {source_id} is LeRobot v3.0, whose episode boundaries are in parquet, and\n"
            "       pyarrow is not installed. pip install pyarrow"
        ) from None

    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise SystemExit(
            f"FATAL: {source_id} declares v3.0 but has no meta/episodes/*/*.parquet.\n"
            "       That is where the episode boundaries live; without them the concatenated\n"
            "       mp4s cannot be cut into episodes at all."
        )

    wanted = ["episode_index", "length", "tasks",
              f"videos/{key}/chunk_index", f"videos/{key}/file_index",
              f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"]
    rows: list[dict] = []
    for f in files:
        table = pq.read_table(f)
        missing = [c for c in wanted if c not in table.column_names]
        if missing:
            raise SystemExit(
                f"FATAL: {f} is missing {missing}.\n"
                f"       Columns for camera {key!r} are per-key, so this usually means the camera\n"
                f"       name is wrong for this source rather than that the file is corrupt."
            )
        rows.extend(table.select(wanted).to_pylist())
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def _scan_v30(root: Path, source_id: str, key: str, fps: float, width: int, height: int,
              info: dict, min_frames: int) -> tuple[list[Clip], dict[str, int]]:
    template = str(info.get("video_path") or DEFAULT_V30_VIDEO_PATH)
    kept: list[Clip] = []
    dropped = {"too_short": 0, "too_long": 0, "missing_video": 0}
    for ep in _episode_rows_v30(root, key, source_id):
        idx = int(ep["episode_index"])
        frames = int(ep["length"])
        duration = frames / fps
        if frames < min_frames:
            dropped["too_short"] += 1
            continue
        if duration > MAX_CLIP_SECONDS:
            dropped["too_long"] += 1
            continue
        # Resolve chunk/file PER CAMERA. Each video key rolls over to a new mp4 when that key's
        # own bytes cross the size limit, so at episode 50 of G1_Dex3_BlockStacking cam_left_high
        # is already on file-001 while the other three cameras are still on file-000. Resolving
        # once per episode and reusing it across keys would read the wrong file for most of them.
        src = root / template.format(video_key=key,
                                     chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                                     file_index=int(ep[f"videos/{key}/file_index"]))
        if not src.is_file():
            dropped["missing_video"] += 1
            continue
        tasks = list(ep.get("tasks") or [])
        # from/to are offsets into THAT mp4 — the accumulator resets to 0.0 on every file
        # rollover — so they go straight to ffmpeg. to_timestamp is exclusive and equals the next
        # episode's from_timestamp, which is why the cut is driven by frame count rather than by
        # -to: an inclusive end would pull in the first frame of the neighbouring episode.
        kept.append(
            Clip(
                uuid=f"{source_id}_episode_{idx:06d}_clip000",
                source_id=source_id,
                episode_index=idx,
                src_path=str(src),
                frames=frames,
                fps=fps,
                duration_s=round(duration, 3),
                width=width,
                height=height,
                task=tasks[0] if tasks else "",
                from_s=float(ep[f"videos/{key}/from_timestamp"]),
                to_s=float(ep[f"videos/{key}/to_timestamp"]),
                src_codec=str(info["features"][key].get("info", {}).get("video.codec", "")),
            )
        )
    return kept, dropped


def _scan_v21(root: Path, source_id: str, key: str, fps: float, width: int, height: int,
              info: dict, min_frames: int) -> tuple[list[Clip], dict[str, int]]:
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    chunk_size = int(info.get("chunks_size") or 1000)
    kept: list[Clip] = []
    dropped = {"too_short": 0, "too_long": 0, "missing_video": 0}
    for ep in episodes:
        idx = int(ep["episode_index"])
        frames = int(ep["length"])
        duration = frames / fps
        if frames < min_frames:
            dropped["too_short"] += 1
            continue
        if duration > MAX_CLIP_SECONDS:
            dropped["too_long"] += 1
            continue
        src = episode_video_path(root, key, idx, chunk_size)
        if not src.is_file():
            dropped["missing_video"] += 1
            continue
        tasks = ep.get("tasks") or []
        kept.append(
            Clip(
                uuid=f"{source_id}_episode_{idx:06d}_clip000",
                source_id=source_id,
                episode_index=idx,
                src_path=str(src),
                frames=frames,
                fps=fps,
                duration_s=round(duration, 3),
                width=width,
                height=height,
                task=tasks[0] if tasks else "",
                src_codec=str(info["features"][key].get("info", {}).get("video.codec", "")),
            )
        )
    return kept, dropped


def scan_source(
    root: Path, source_id: str, camera_key: str | None, min_frames: int
) -> tuple[list[Clip], dict[str, int]]:
    """Enumerate the episodes of one LeRobot root that survive the loader's filters."""
    info = json.loads((root / "meta" / "info.json").read_text())
    version = require_supported(info, source_id)
    key = resolve_camera(info, camera_key, source_id)
    fps = float(info.get("fps") or info["features"][key].get("info", {}).get("video.fps") or 0.0)
    if fps <= 0:
        raise SystemExit(f"FATAL: {source_id} declares no usable fps — cannot apply the 61 s bound.")
    width, height = video_shape(info, key)
    scan = _scan_v21 if version == "v2.1" else _scan_v30
    return scan(root, source_id, key, fps, width, height, info, min_frames)


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()


#: Encoder presets. H.264 High profile / yuv420p is the most widely decodable thing we can write,
#: which is the entire point: the corpus has to survive whatever OpenCV build the captioner's
#: virtualenv happens to pull. AV1 — LeRobot's own default — did not (see scripts/verify_clip_decode.py).
ENCODERS = {
    # Quality first. At 640x480 this is fast enough on any many-core box and beats NVENC per bit.
    "libx264": ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-profile:v", "high"],
    # Speed first, for a GPU box. cq 19 is visually near-lossless at this resolution; NVENC is
    # weaker than x264 at equal bitrate, so the quality knob is deliberately generous.
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                   "-profile:v", "high"],
}


class TranscodeError(RuntimeError):
    pass


def ffmpeg_cmd(clip: Clip, dest: Path, encoder: str, ffmpeg: str) -> list[str]:
    """Build the extract-and-re-encode command for one clip.

    ``-ss`` goes BEFORE ``-i`` on purpose. As an input option it seeks, so cutting episode 300 out
    of a concatenated v3.0 file costs a seek instead of decoding the 299 episodes in front of it —
    the difference between minutes and hours over a corpus. Frame accuracy is not sacrificed
    because we re-encode: ffmpeg decodes from the preceding keyframe and discards the lead-in, and
    LeRobot encodes AV1 with ``g=2``, so the preceding keyframe is never more than a frame away.

    The length is ``-frames:v``, NOT ``-to``. v3.0's ``to_timestamp`` is exclusive and identical to
    the next episode's ``from_timestamp``; handing that to ``-to`` appends the neighbouring
    episode's first frame to every clip in the corpus. Counting frames also makes the output length
    exactly the number the metadata promised, which is what the check after the encode compares
    against.

    ``-fps_mode cfr -r`` pins the output to the dataset's declared rate. The sources are already
    constant-rate, so this changes nothing about the pixels; it guarantees that "frame i" means the
    same thing in the clip as it does in the metadata.
    """
    if encoder not in ENCODERS:
        raise TranscodeError(f"unknown encoder {encoder!r}; known: {sorted(ENCODERS)}")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-y"]
    if clip.from_s is not None:
        cmd += ["-ss", f"{clip.from_s:.6f}"]
    cmd += ["-i", clip.src_path]
    if clip.frames > 0:
        cmd += ["-frames:v", str(clip.frames)]
    # -an/-sn/-dn: audio, subtitles and data streams are not training signal, and dropping them
    # keeps the output a single-stream file that every decoder agrees about.
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn"]
    cmd += ENCODERS[encoder]
    cmd += ["-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", f"{clip.fps:g}",
            "-movflags", "+faststart", str(dest)]
    return cmd


def probe_frames(path: Path, ffprobe: str) -> int:
    """Frame count the muxer wrote into the output container. 0 when it did not write one."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def materialize(clip: Clip, dest_dir: Path, mode: str, encoder: str,
                ffmpeg: str, ffprobe: str, frame_tolerance: int) -> Clip:
    """Put one clip where the captioner will find it, in the form it can read.

    Three modes, in increasing order of cost and of how much they guarantee:

    ``link``      the historical default. Free, and correct only when the source file is already
                  exactly one episode in a decodable codec.
    ``copy``      same bytes, independent tree. Needed when the source is a scratch cache.
    ``transcode`` re-encode to H.264. The only mode that can cut a v3.0 window out of a
                  concatenated file, and the only one that fixes an undecodable codec.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{clip.uuid}.mp4"
    src = Path(clip.src_path)
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    # Hash the source, not the link target's path: the value has to survive the tree being moved.
    src_digest = _sha256(src)

    if mode == "transcode":
        cmd = ffmpeg_cmd(clip, dest, encoder, ffmpeg)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not dest.is_file():
            raise TranscodeError(
                f"{clip.uuid}: ffmpeg exited {proc.returncode}\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  {proc.stderr.strip()[:800]}"
            )
        # A cut that silently lands short is the failure mode that matters here: the clip still
        # plays, the manifest still counts it, and the corpus quietly contains a truncated episode.
        # The frame count is the cheapest thing that catches it.
        got = probe_frames(dest, ffprobe)
        if got and abs(got - clip.frames) > frame_tolerance:
            raise TranscodeError(
                f"{clip.uuid}: expected ~{clip.frames} frames, got {got} "
                f"(tolerance {frame_tolerance}). Window {clip.from_s}–{clip.to_s} s of "
                f"{clip.src_path} did not produce the episode the metadata describes."
            )
        return Clip(**{**asdict(clip), "src_sha256": src_digest, "sha256": _sha256(dest)})

    if clip.from_s is not None or clip.to_s is not None:
        raise TranscodeError(
            f"{clip.uuid} is a window ({clip.from_s}–{clip.to_s} s) inside a shared file, so it "
            f"cannot be {mode}ed — linking or copying would hand the captioner every episode in "
            f"{Path(clip.src_path).name} under one episode's name. Use --transcode."
        )
    if mode == "link":
        os.symlink(os.path.realpath(src), dest)
    else:
        shutil.copy2(src, dest)
    return Clip(**{**asdict(clip), "src_sha256": src_digest, "sha256": src_digest})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="DIR",
                    help="LeRobot v2.1 root. Repeat to pool corpora. Directory name becomes the id.")
    ap.add_argument("--source-id", action="append", default=[], metavar="ID",
                    help="Override the id for the Nth --source (order matters).")
    # Repeatable and index-matched, because pooled corpora do not agree on camera names:
    # AppleToPlate's ego view and a Unitree set's head view are different keys, and one global
    # flag would either pick the wrong view for one of them or make pooling impossible.
    # A single value applies to every source.
    ap.add_argument("--camera-key", action="append", default=[], metavar="KEY",
                    help="Video feature key. Required when a source has more than one camera. "
                         "Repeat to give one per --source, or pass once to apply to all.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-episodes", type=int, default=30,
                    help="Held out from SFT and reserved for PR-09 §5's eval prompts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-video-frames", type=int, default=-1,
                    help="Recipe window length. -1 keeps the framework's 61-frame floor.")
    ap.add_argument("--mode", choices=("link", "copy", "transcode"), default="link",
                    help="link: symlink the source mp4 (free, v2.1 only, keeps the source codec). "
                         "copy: same bytes in an independent tree. "
                         "transcode: re-encode to H.264 — required for v3.0 windows, and the fix "
                         "for a source codec the captioner's decoder cannot read.")
    ap.add_argument("--copy", action="store_true", help=argparse.SUPPRESS)  # pre---mode spelling
    ap.add_argument("--encoder", choices=tuple(ENCODERS), default="libx264",
                    help="h264_nvenc on a machine with an NVIDIA GPU; libx264 everywhere else.")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    ap.add_argument("--ffprobe", default=os.environ.get("FFPROBE", "ffprobe"))
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel transcodes. Consumer NVIDIA cards cap concurrent NVENC "
                         "sessions, so raising this past ~4 helps only with libx264.")
    ap.add_argument("--frame-tolerance", type=int, default=2,
                    help="Frames a transcoded clip may differ from its metadata before it is "
                         "treated as a bad cut rather than a rounding artefact.")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report; write nothing.")
    args = ap.parse_args(argv)
    mode = "copy" if args.copy else args.mode

    min_frames = MIN_WINDOW_FRAMES if args.num_video_frames < 0 else max(MIN_WINDOW_FRAMES, args.num_video_frames)

    all_clips: list[Clip] = []
    per_source: dict[str, dict] = {}
    for i, src in enumerate(args.source):
        root = Path(src).expanduser().resolve()
        if not (root / "meta" / "info.json").is_file():
            raise SystemExit(f"FATAL: {root} is not a LeRobot v2.1 root (no meta/info.json).")
        sid = args.source_id[i] if i < len(args.source_id) else slugify(root.name)
        if len(args.camera_key) > 1 and len(args.camera_key) != len(args.source):
            raise SystemExit(
                f"FATAL: {len(args.camera_key)} --camera-key values for {len(args.source)} "
                "--source values. Pass one per source, or exactly one for all."
            )
        cam = (args.camera_key[i] if len(args.camera_key) == len(args.source)
               else (args.camera_key[0] if args.camera_key else None))
        clips, dropped = scan_source(root, sid, cam, min_frames)
        per_source[sid] = {"root": str(root), "camera_key": cam or "(sole)",
                           "kept": len(clips), "dropped": dropped}
        print(f"{sid}: kept {len(clips)}  dropped {dropped}", file=sys.stderr)
        all_clips.extend(clips)

    if not all_clips:
        raise SystemExit("FATAL: no episode survived the filters — nothing to train on.")
    if args.val_episodes >= len(all_clips):
        raise SystemExit(
            f"FATAL: --val-episodes {args.val_episodes} >= {len(all_clips)} surviving clips."
        )

    # Sort before shuffling: dict/glob order is not reproducible across filesystems, and a split
    # that depends on it is not the seeded split the manifest claims it is.
    all_clips.sort(key=lambda c: (c.source_id, c.episode_index))
    rng = random.Random(args.seed)
    order = list(range(len(all_clips)))
    rng.shuffle(order)
    val_idx = set(order[: args.val_episodes])
    train = [c for i, c in enumerate(all_clips) if i not in val_idx]
    val = [c for i, c in enumerate(all_clips) if i in val_idx]

    print(f"total {len(all_clips)}  train {len(train)}  val {len(val)}", file=sys.stderr)
    if args.dry_run:
        print("dry run — nothing written", file=sys.stderr)
        return 0

    if mode == "transcode":
        for tool in (args.ffmpeg, args.ffprobe):
            if shutil.which(tool) is None:
                raise SystemExit(f"FATAL: {tool} not on PATH — --mode transcode needs it.")

    def run_split(name: str, clips: list[Clip]) -> list[Clip]:
        dest_dir = args.out / name / "videos"
        work = lambda c: materialize(c, dest_dir, mode, args.encoder,  # noqa: E731
                                     args.ffmpeg, args.ffprobe, args.frame_tolerance)
        if args.jobs <= 1:
            out = []
            for i, c in enumerate(clips, 1):
                out.append(work(c))
                if mode == "transcode" and i % 25 == 0:
                    print(f"  {name}: {i}/{len(clips)}", file=sys.stderr)
            return out
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # map, not submit-and-gather: it preserves clip order, and the manifest's clip list is
            # the thing the seeded split is checked against later.
            return list(pool.map(work, clips))

    try:
        placed = {"train": run_split("train", train), "val": run_split("val", val)}
    except TranscodeError as exc:
        # No partial manifest. A manifest that describes clips which were never written is worse
        # than no manifest: everything downstream trusts it.
        raise SystemExit(f"FATAL: {exc}")

    manifest = {
        "schema": "wam.cosmos_corpus/1",
        "prereg": "docs/preregistration/PR-09-cosmos-super-finetune.md",
        "seed": args.seed,
        "filters": {
            "max_clip_seconds": MAX_CLIP_SECONDS,
            "min_window_frames": min_frames,
            "note": "cosmos-framework sft_dataset.py's own silent filters, applied here so the "
                    "counts below are the counts that train.",
        },
        "sources": per_source,
        # What the clips on disk actually are. A corpus re-encoded with different settings is a
        # different corpus, and "which encoder" is the first question asked when a run looks off.
        "materialization": {
            "mode": mode,
            "encoder": args.encoder if mode == "transcode" else None,
            "encoder_args": ENCODERS[args.encoder] if mode == "transcode" else None,
            "frame_tolerance": args.frame_tolerance if mode == "transcode" else None,
        },
        "counts": {k: len(v) for k, v in placed.items()},
        "clips": {k: [asdict(c) for c in v] for k, v in placed.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Hash the BYTES ON DISK, trailing newline included. This used to hash `body` while writing
    # `body + "\n"`, so MANIFEST_SHA256 recorded a digest that `sha256sum manifest.json` could
    # never reproduce — and `sha256sum manifest.json` is exactly what 92b_register_corpus.sbatch
    # runs to decide whether the shipped corpus is the one that was prepared. A provenance hash
    # nobody can recompute with the obvious command is not provenance, it is decoration.
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.out / "manifest.json").write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (args.out / "MANIFEST_SHA256").write_text(digest + "\n")
    print(f"wrote {args.out}/manifest.json  sha256={digest}", file=sys.stderr)
    print("next: verify the clips decode with the captioner's own cv2 BEFORE captioning —\n"
          f"      <captioner-venv>/bin/python scripts/verify_clip_decode.py {args.out}/train/videos\n"
          "      (skipping this is what produced '0/372 videos were successfully captioned')",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
