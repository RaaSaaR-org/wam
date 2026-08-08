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
import sys
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
    """(width, height) from the feature's declared shape, which LeRobot writes as [H, W, C]."""
    shape = info["features"][key].get("shape") or []
    if len(shape) < 2:
        return (0, 0)
    return (int(shape[1]), int(shape[0]))


#: Everything below assumes LeRobot v2.1: one mp4 per episode at
#: ``videos/chunk-NNN/<key>/episode_NNNNNN.mp4``, and one JSON object per episode in
#: ``meta/episodes.jsonl``. v3.0 breaks both. It CONCATENATES episodes into a few large files at
#: ``videos/<key>/chunk-NNN/file-NNN.mp4`` — 301 episodes of G1_Dex3_BlockStacking live in 19 of
#: them — and moves the per-episode boundaries into ``meta/episodes/chunk-NNN/file-NNN.parquet``
#: as ``videos/<key>/from_timestamp`` and ``.../to_timestamp``.
#:
#: Supporting it is not a path fix, it is an extraction pass: every clip has to be cut out of a
#: concatenated file by timestamp. Until that exists, say so in one line. The alternative is what
#: actually happened on jobs 186353/186354 — a FileNotFoundError traceback pointing at
#: episodes.jsonl, which reads like a broken download rather than an unsupported format, and sent
#: me looking at the fetch step for a problem that was never there.
_SUPPORTED_CODEBASE = "v2.1"


def require_v21(info: dict, source_id: str) -> None:
    version = str(info.get("codebase_version") or "unknown")
    if version != _SUPPORTED_CODEBASE:
        raise SystemExit(
            f"FATAL: {source_id} is LeRobot {version}; this script reads {_SUPPORTED_CODEBASE} only.\n"
            f"       {version} packs many episodes into one mp4 and keeps the boundaries in\n"
            f"       meta/episodes/*/*.parquet, so each clip must be cut out by timestamp — a\n"
            f"       conversion step this script does not have. The download is fine; the format\n"
            f"       is not the one prepare_cosmos_corpus.py was written for."
        )


def episode_video_path(root: Path, key: str, episode_index: int, chunk_size: int) -> Path:
    chunk = episode_index // max(chunk_size, 1)
    return root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"


def scan_source(
    root: Path, source_id: str, camera_key: str | None, min_frames: int
) -> tuple[list[Clip], dict[str, int]]:
    """Enumerate the episodes of one LeRobot root that survive the loader's filters."""
    info = json.loads((root / "meta" / "info.json").read_text())
    require_v21(info, source_id)
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    key = resolve_camera(info, camera_key, source_id)
    fps = float(info.get("fps") or info["features"][key].get("info", {}).get("video.fps") or 0.0)
    if fps <= 0:
        raise SystemExit(f"FATAL: {source_id} declares no usable fps — cannot apply the 61 s bound.")
    chunk_size = int(info.get("chunks_size") or 1000)
    width, height = video_shape(info, key)

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
            )
        )
    return kept, dropped


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()


def place(clip: Clip, dest_dir: Path, link: bool) -> Clip:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{clip.uuid}.mp4"
    src = Path(clip.src_path)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if link:
        os.symlink(os.path.realpath(src), dest)
    else:
        shutil.copy2(src, dest)
    # Hash the source, not the link target's path: the value has to survive the tree being moved.
    return Clip(**{**asdict(clip), "sha256": _sha256(src)})


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
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of symlink. Needed when the source tree is a scratch cache.")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report; write nothing.")
    args = ap.parse_args(argv)

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

    placed = {
        "train": [place(c, args.out / "train" / "videos", link=not args.copy) for c in train],
        "val": [place(c, args.out / "val" / "videos", link=not args.copy) for c in val],
    }

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
        "counts": {k: len(v) for k, v in placed.items()},
        "clips": {k: [asdict(c) for c in v] for k, v in placed.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True)
    (args.out / "manifest.json").write_text(body + "\n")
    digest = hashlib.sha256(body.encode()).hexdigest()
    (args.out / "MANIFEST_SHA256").write_text(digest + "\n")
    print(f"wrote {args.out}/manifest.json  sha256={digest}", file=sys.stderr)
    print(f"next: caption {args.out}/train/videos and {args.out}/val/videos "
          f"(cluster/discoverer/92_caption_corpus.sbatch)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
