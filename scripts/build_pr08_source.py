#!/usr/bin/env python3
"""Build ``pr08-apple-640x480`` — the SOURCE corpus PR-08 §3 restyles.

    .venv/bin/python scripts/build_pr08_source.py \\
        --snapshot ~/.cache/huggingface/hub/datasets--nvidia--GR00T-N1.7-AppleToPlate/snapshots/<sha> \\
        --out /valhalla/projects/.../data/pr08-apple-640x480

WHY THIS EXISTS. ``97_transfer25_restyle.sbatch`` reads ``${SOURCE}/manifest.json`` and its header
says, in as many words, *"Nothing writes it today — workstation/10_fetch_corpus.sh and
scripts/convert_lerobot_g1.py produce the 120x160 tree PR-08 §3 forbids."* That sentence has been
true since the job was written, which means the one measurement PR-08 §1 licenses unconditionally —
timing one episode on an H200, §8 item 3 — has had no input to run on. This is that input.

WHAT IT IS NOT. It is not a converter. The HF source is **already 640x480** (``meta/info.json``:
``observation.images.ego_view`` has shape ``[480, 640, 3]``), so nothing here rescales, re-encodes
or transcodes anything: doing so would put a lossy generation between the recorded pixels and the
restyle, and the whole premise of Transfer2.5 as an augmentation is that the pixels under the
carried-over actions are the real ones. This script reads the snapshot, CHECKS every claim it makes
about it, and writes a manifest plus a tree of links. If a check fails it refuses; it never repairs.

THE CHECK THAT MATTERS MOST is per-episode frame count against ``meta/episodes.jsonl``'s declared
``length``. The actions are carried over from the recording unchanged, so the manifest's ``frames``
is what pairs a label index with a pixel index. If the video holds a different number of frames than
the label column declares, every downstream pair after the divergence describes a different instant
than its label — silently, with no decode error, and with the restyle looking perfect. That is the
single defect this corpus can have that no later gate would catch, because G0a checks the actions
are unchanged (they are) and G0b checks geometry within a clip (it is fine), and neither compares
the two lengths.

WHAT THE MANIFEST DELIBERATELY OMITS. ``depth`` and ``segmentation`` are absent from every episode
entry, and that is a decision rather than an oversight. ``restyle_transfer25.py`` treats an absent
map as "let Transfer2.5 estimate it with its own models" and a *named but missing* map as a fatal
error, precisely so the estimator a run used is never ambiguous. PR-08 §4's estimated conditioning
is blocked on §8 items 4 and 5; until those land, the honest manifest is one that claims no maps at
all. Adding the keys later is an edit to this script and a regenerated manifest, which is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

# PR-08 §3. Not a parameter: a manifest that declares anything else is not this corpus, and 97
# refuses it on the other side anyway.
TARGET_W, TARGET_H = 640, 480

VIDEO_KEY = "observation.images.ego_view"


class BuildError(RuntimeError):
    """A refusal. Every one of these means the snapshot is not what the manifest would claim."""


# --------------------------------------------------------------------------------------------
# reading the snapshot
# --------------------------------------------------------------------------------------------


def read_info(snapshot: pathlib.Path) -> dict:
    p = snapshot / "meta" / "info.json"
    if not p.is_file():
        raise BuildError(f"{p} missing — this is not a LeRobot snapshot root.")
    return json.loads(p.read_text())


def read_episodes(snapshot: pathlib.Path) -> list[dict]:
    """``meta/episodes.jsonl``, one object per episode, carrying the authoritative ``length``.

    A partially-downloaded HF cache has ``info.json`` and no ``episodes.jsonl`` — the metadata-only
    state ``hf download`` leaves when it is given ``--include meta/info.json``. Naming that case is
    worth a line: without it the failure is a bare FileNotFoundError on a path most readers have
    never seen.
    """
    p = snapshot / "meta" / "episodes.jsonl"
    if not p.is_file():
        raise BuildError(
            f"{p} missing. The snapshot carries meta/info.json but not the per-episode lengths, "
            "which is what a metadata-only download looks like. Fetch the dataset in full "
            "(videos included) before building the source corpus."
        )
    out = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BuildError(f"{p}:{n} is not JSON: {exc}") from exc
    if not out:
        raise BuildError(f"{p} is empty.")
    return out


def declared_resolution(info: dict, video_key: str) -> tuple[int, int]:
    """(width, height) from the feature's ``shape``, which LeRobot writes as [H, W, C]."""
    feats = info.get("features") or {}
    if video_key not in feats:
        raise BuildError(
            f"info.json declares no feature {video_key!r}. Cameras present: "
            f"{sorted(k for k in feats if 'image' in k or 'video' in k) or '(none)'}"
        )
    shape = feats[video_key].get("shape")
    if not (isinstance(shape, list) and len(shape) == 3):
        raise BuildError(f"{video_key} has no usable shape: {shape!r}")
    h, w, _ = shape
    return int(w), int(h)


def video_relpath(info: dict, episode_index: int, video_key: str) -> str:
    template = info.get("video_path")
    if not template:
        raise BuildError("info.json has no video_path template.")
    chunks_size = int(info.get("chunks_size") or 1000)
    return template.format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )


# --------------------------------------------------------------------------------------------
# probing the pixels
# --------------------------------------------------------------------------------------------


def ffprobe_stream(video: pathlib.Path, ffprobe: str) -> dict:
    """Width, height, codec and frame count of the first video stream.

    ``nb_frames`` is authoritative when the container carries it and absent often enough that a
    fallback is not optional. ``-count_packets`` is the cheap fallback: for these single-stream mp4s
    one packet is one frame, and it does not decode. Counting DECODED frames would be exact for any
    container but costs a full decode of 402 AV1 videos, which is not a price worth paying for a
    number the container already knows.
    """
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=width,height,codec_name,nb_frames,nb_read_packets",
        "-of",
        "json",
        str(video),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise BuildError(
            f"{ffprobe} not found. Pass --ffprobe, or put ffmpeg's bin on PATH — on the cluster "
            "that is ${FFMPEG_PREFIX}/bin."
        ) from exc
    if proc.returncode != 0:
        raise BuildError(f"ffprobe failed on {video}:\n{proc.stderr.strip()}")
    streams = (json.loads(proc.stdout) or {}).get("streams") or []
    if not streams:
        raise BuildError(f"{video} has no video stream.")
    s = streams[0]
    frames = s.get("nb_frames") or s.get("nb_read_packets")
    if frames in (None, "", "N/A"):
        raise BuildError(f"{video}: ffprobe reports no frame count, so it cannot be verified.")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "codec": s.get("codec_name") or "unknown",
        "frames": int(frames),
    }


# --------------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------------


def build(
    snapshot: pathlib.Path,
    out: pathlib.Path,
    *,
    ffprobe: str = "ffprobe",
    video_key: str = VIDEO_KEY,
    copy: bool = False,
    revision: str | None = None,
    repo_id: str | None = None,
    limit: int | None = None,
) -> dict:
    info = read_info(snapshot)
    w, h = declared_resolution(info, video_key)
    if (w, h) != (TARGET_W, TARGET_H):
        raise BuildError(
            f"{video_key} declares {w}x{h}; PR-08 §3 restyles at {TARGET_W}x{TARGET_H}. This "
            "script does not rescale — a lossy generation between the recorded pixels and the "
            "restyle is exactly what §3 refuses. Point --snapshot at the full-resolution source."
        )

    episodes = read_episodes(snapshot)
    if limit is not None:
        episodes = episodes[:limit]

    videos_root = out / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    codecs: set[str] = set()
    for ep in episodes:
        if "episode_index" not in ep or "length" not in ep:
            raise BuildError(f"episodes.jsonl row lacks episode_index/length: {ep!r}")
        idx = int(ep["episode_index"])
        declared_len = int(ep["length"])
        rel = video_relpath(info, idx, video_key)
        src = snapshot / rel
        if not src.is_file():
            raise BuildError(
                f"episode {idx}: {src} missing. The snapshot has metadata but not the video — "
                "fetch the dataset in full before building the source corpus."
            )
        probe = ffprobe_stream(src, ffprobe)
        if (probe["width"], probe["height"]) != (TARGET_W, TARGET_H):
            raise BuildError(
                f"episode {idx}: {src} is {probe['width']}x{probe['height']}, not "
                f"{TARGET_W}x{TARGET_H}, while info.json declares {w}x{h}. The metadata and the "
                "pixels disagree; trusting either one silently would be a guess."
            )
        if probe["frames"] != declared_len:
            raise BuildError(
                f"episode {idx}: the video holds {probe['frames']} frames and episodes.jsonl "
                f"declares length {declared_len}. The actions are carried over by index, so a "
                "mismatch pairs every later label with a different instant than its pixels — "
                "silently, and with the restyle looking perfect. Refusing to write a manifest "
                "that would claim these agree."
            )
        codecs.add(probe["codec"])

        dest_rel = f"videos/episode_{idx:06d}.mp4"
        dest = out / dest_rel
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if copy:
            shutil.copy2(src, dest)
        else:
            # A symlink by default and a real copy on request. The corpus is ~0.9 GB of mp4 that
            # already exists in the HF cache, and rsync's -l (which cluster/discoverer/sync.sh
            # carries) ships links as links. --copy is for the case where the cache is not going
            # to travel with the tree.
            dest.symlink_to(os.path.relpath(src.resolve(), dest.parent))

        entries.append({"id": f"episode_{idx:06d}", "frames": declared_len, "video": dest_rel})

    if not entries:
        raise BuildError("no episodes — nothing to write.")

    manifest = {
        "resolution": [TARGET_W, TARGET_H],
        "fps": info.get("fps"),
        "video_key": video_key,
        # Provenance, per AC-04: a restyle has to be traceable to the pixels it restyled.
        "source": {
            "repo_id": repo_id,
            "revision": revision,
            "codebase_version": info.get("codebase_version"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "codecs": sorted(codecs),
            "materialized": "copy" if copy else "symlink",
        },
        "episodes": entries,
    }
    return manifest


def write_manifest(out: pathlib.Path, manifest: dict) -> pathlib.Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "manifest.json"
    text = json.dumps(manifest, indent=2) + "\n"
    p.write_text(text)
    # The same .sha256 discipline configs/transfer25/pr08_style_partition.json uses: 97 records
    # what it read, and a sidecar is how "the file the gate read" and "the file that was built"
    # are shown to be one file.
    (out / "manifest.json.sha256").write_text(
        hashlib.sha256(text.encode()).hexdigest() + "  manifest.json\n"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snapshot", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--ffprobe", default=os.environ.get("FFPROBE", "ffprobe"))
    ap.add_argument("--video-key", default=VIDEO_KEY)
    ap.add_argument("--copy", action="store_true", help="materialize videos instead of linking")
    ap.add_argument("--repo-id", default="nvidia/GR00T-N1.7-AppleToPlate")
    ap.add_argument("--revision", default=None, help="recorded in the manifest for AC-04")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes; for smoke runs only")
    args = ap.parse_args(argv)

    try:
        manifest = build(
            args.snapshot,
            args.out,
            ffprobe=args.ffprobe,
            video_key=args.video_key,
            copy=args.copy,
            revision=args.revision,
            repo_id=args.repo_id,
            limit=args.limit,
        )
    except BuildError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    p = write_manifest(args.out, manifest)
    n = len(manifest["episodes"])
    total = sum(e["frames"] for e in manifest["episodes"])
    print(f"wrote {p}")
    print(
        f"  {n} episodes, {total} frames, {TARGET_W}x{TARGET_H}, codecs {manifest['source']['codecs']}"
    )
    if args.limit is not None:
        print(
            f"  --limit {args.limit} was used: this is a SMOKE corpus, not the {n}-episode source."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
