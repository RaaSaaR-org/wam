#!/usr/bin/env python
"""Assemble restyled clips + their SOURCE action labels into a LeRobot v2.1 dataset root.

    .venv/bin/python scripts/assemble_restyled_lerobot.py \
        --source ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \
        --clips  ~/runs/t040-.../clips/train \
        --work-list ~/runs/t040-.../chunks/train-01of08/work.jsonl \
        --with-real \
        --label arm-B \
        --out ~/wam-t041/datasets/arm-B

WHAT THIS CLOSES. 97_transfer25_restyle.sbatch stops at a flat directory of mp4s stamped
NOT_TRAINING_DATA. docs/contracts/vla-training-consumer.md §1 says the deliverable is "a LeRobot
dataset root ... Not a new format, not a sidecar, not a manifest", because every entry point in the
consumer repo takes an ordinary LeRobot root. Nothing bridged those two facts. This does.

THE ONE INVARIANT THIS SCRIPT EXISTS TO ENFORCE. From the contract, §4:

    The action labels come from the SOURCE RECORDING. Never from the generator.

So no column here is ever computed, inferred, resampled or smoothed. Every restyled episode's
parquet is its source episode's parquet with exactly two columns rewritten -- `episode_index` and
`index`, both pure bookkeeping -- and every other column, including all nine `action.*` extras,
copied through untouched. If a future edit finds itself deriving an action value, the edit is wrong.

WHY FRAME COUNTS ARE A HARD GATE, NOT A WARNING. GR00T addresses video by frame index
(gr00t/utils/video_utils.py). A restyle that drops or duplicates one frame leaves a clip whose
pixels are offset from the actions by one step, for every step after the drop. Nothing downstream
notices: the shapes still line up, the loss still falls, and the policy learns a systematically
shifted control law. `ffprobe -count_frames` is slow and is the default anyway, because the fast
path reads a container field that AV1 muxers frequently omit or get wrong.

WHAT THIS SCRIPT DOES NOT DO. It does not write meta/stats.json or meta/relative_stats.json -- those
come from gr00t's own generator, and the exact command is printed at the end. It does not run the
G0a/G0b/G0c gates. It does not decide whether the clips it was handed are fit to train on; a clip
directory still marked NOT_TRAINING_DATA is assembled without complaint, because the gate that
clears that marker is a separate decision by a person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_KEY = "observation.images.ego_view"
CODEBASE_VERSION = "v2.1"

# Rewritten per episode. Everything else is carried through byte-identically.
BOOKKEEPING_COLUMNS = ("episode_index", "index")


# --------------------------------------------------------------------------------------------
# unit model
# --------------------------------------------------------------------------------------------


@dataclass
class Unit:
    """One output episode: a video, and the source episode whose labels it carries."""

    source_index: int          # episode_index in --source
    video: Path                # the mp4 that becomes this episode's video
    origin: str                # "real" or the clip's unit id
    style: str | None = None   # style id, when known from the work list
    repeat: int | None = None

    new_index: int = -1        # assigned during layout
    frames: int = -1           # verified against the source parquet


@dataclass
class Probe:
    frames: int
    width: int
    height: int
    codec: str
    pix_fmt: str
    fps: float


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.problems.append(msg)


# --------------------------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------------------------


def _ffprobe_json(path: Path, count_frames: bool) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,width,height,codec_name,pix_fmt,avg_frame_rate",
        "-of", "json",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd.append(str(path))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    streams = json.loads(out.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    return streams[0]


def probe(path: Path, mode: str) -> Probe:
    """mode: strict (-count_frames), container (nb_frames), none (frames = -1)."""
    stream = _ffprobe_json(path, count_frames=(mode == "strict"))

    if mode == "strict":
        raw = stream.get("nb_read_frames")
    elif mode == "container":
        raw = stream.get("nb_frames")
    else:
        raw = None
    try:
        frames = int(raw)
    except (TypeError, ValueError):
        if mode == "none":
            frames = -1
        else:
            raise RuntimeError(
                f"{path}: no frame count available in {mode!r} mode. AV1 muxers often omit "
                f"nb_frames; re-run with --verify-frames strict."
            )

    num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = int(num) / int(den) if int(den) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0

    return Probe(
        frames=frames,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        codec=str(stream.get("codec_name") or "?"),
        pix_fmt=str(stream.get("pix_fmt") or "?"),
        fps=fps,
    )


# --------------------------------------------------------------------------------------------
# source dataset
# --------------------------------------------------------------------------------------------


def read_source(root: Path) -> dict:
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2."):
        raise SystemExit(
            f"FATAL: --source is LeRobot {info.get('codebase_version')!r}. This script writes "
            f"v2.1 and copies the source's parquet schema; a v3.0 source has a different layout "
            f"and must be converted first."
        )
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return {
        "root": root,
        "info": info,
        "episodes": episodes,
        "by_index": {int(e["episode_index"]): e for e in episodes},
        "tasks_raw": (meta / "tasks.jsonl").read_text(),
        "modality_raw": (meta / "modality.json").read_text(),
        "episode_stats": meta / "episodes_stats.jsonl",
    }


def source_paths(src: dict, index: int) -> tuple[Path, Path]:
    info, root = src["info"], src["root"]
    chunk = index // int(info["chunks_size"])
    parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
    video = root / info["video_path"].format(
        episode_chunk=chunk, episode_index=index, video_key=VIDEO_KEY
    )
    return parquet, video


# --------------------------------------------------------------------------------------------
# clip -> source episode resolution
# --------------------------------------------------------------------------------------------

# 97_transfer25_restyle.sbatch:933 -- unit = f'{episode_id}__{style_id}__r{repeat:02d}'
UNIT_RE = re.compile(r"^(?P<episode>.+)__(?P<style>[^_].*?)__r(?P<repeat>\d+)$")
SIX_DIGITS = re.compile(r"(\d{6})")


def load_work_lists(paths: list[Path]) -> dict[str, dict]:
    """unit id -> row, from the sbatch's work.jsonl. The authoritative mapping."""
    units: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            unit = row.get("unit")
            if unit is None:
                raise SystemExit(f"FATAL: {path} has a row without a 'unit' field: {row}")
            if unit in units and units[unit] != row:
                raise SystemExit(
                    f"FATAL: unit {unit!r} appears twice in the work lists with different rows. "
                    f"Two chunks claim the same clip; the partition is not a partition."
                )
            units[unit] = row
    return units


def resolve_episode(episode_id: str, episode_map: dict[str, int] | None, n_source: int) -> int:
    """Map an opaque corpus episode id onto a --source episode_index.

    The map is authoritative. The six-digit heuristic exists only because the corpus manifest is
    produced by a different script, and it refuses rather than guesses when it is not certain.
    """
    if episode_map is not None:
        if episode_id not in episode_map:
            raise SystemExit(
                f"FATAL: episode id {episode_id!r} is absent from --episode-map. Resolving it by "
                f"guesswork would attach one episode's actions to another episode's pixels."
            )
        return int(episode_map[episode_id])

    found = SIX_DIGITS.findall(str(episode_id))
    if len(found) != 1:
        raise SystemExit(
            f"FATAL: cannot resolve episode id {episode_id!r} to a source index -- found "
            f"{len(found)} six-digit groups. Supply --episode-map."
        )
    index = int(found[0])
    if not 0 <= index < n_source:
        raise SystemExit(
            f"FATAL: episode id {episode_id!r} resolves to index {index}, outside the source's "
            f"0..{n_source - 1}."
        )
    return index


def collect_clip_units(
    clip_dirs: list[Path],
    work: dict[str, dict],
    episode_map: dict[str, int] | None,
    n_source: int,
) -> list[Unit]:
    units: list[Unit] = []
    seen: dict[str, Path] = {}

    for clip_dir in clip_dirs:
        mp4s = sorted(clip_dir.glob("*.mp4"))
        if not mp4s:
            raise SystemExit(f"FATAL: {clip_dir} contains no .mp4 files.")
        for mp4 in mp4s:
            unit_id = mp4.stem
            if unit_id in seen:
                raise SystemExit(
                    f"FATAL: unit {unit_id!r} appears in two clip directories:\n"
                    f"  {seen[unit_id]}\n  {mp4}"
                )
            seen[unit_id] = mp4

            row = work.get(unit_id)
            if row is not None:
                episode_id, style, repeat = row["episode"], row.get("style"), row.get("repeat")
            else:
                match = UNIT_RE.match(unit_id)
                if not match:
                    raise SystemExit(
                        f"FATAL: {mp4.name} matches neither the work list nor the unit-id pattern "
                        f"<episode>__<style>__rNN. Pass --work-list."
                    )
                episode_id = match["episode"]
                style = match["style"]
                repeat = int(match["repeat"])

            units.append(
                Unit(
                    source_index=resolve_episode(episode_id, episode_map, n_source),
                    video=mp4,
                    origin=unit_id,
                    style=style,
                    repeat=repeat,
                )
            )

    # Deterministic and independent of filesystem order: source episode, then style, then repeat.
    units.sort(key=lambda u: (u.source_index, str(u.style), u.repeat if u.repeat is not None else -1))
    return units


# --------------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------------


def rewrite_parquet(table: pa.Table, new_index: int, index_start: int) -> pa.Table:
    """Replace the two bookkeeping columns. Every other column is passed through by reference."""
    n = table.num_rows
    for name in BOOKKEEPING_COLUMNS:
        if name not in table.column_names:
            raise SystemExit(f"FATAL: source parquet has no {name!r} column.")

    values = {
        "episode_index": [new_index] * n,
        "index": list(range(index_start, index_start + n)),
    }

    for name, vals in values.items():
        pos = table.column_names.index(name)
        field_type = table.schema.field(pos).type
        # LeRobot writes these as plain int64, but info.json declares shape [1]; some exporters
        # honour that with a length-1 list. Match whatever the source actually used.
        if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
            array = pa.array([[v] for v in vals], type=field_type)
        elif pa.types.is_fixed_size_list(field_type):
            array = pa.array([[v] for v in vals], type=field_type)
        else:
            array = pa.array(vals, type=field_type)
        table = table.set_column(pos, table.schema.field(pos), array)
    return table


def place_video(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(src, dest)
        return
    if mode == "symlink":
        dest.symlink_to(src.resolve())
        return
    try:
        os.link(src, dest)
    except OSError:
        # Different filesystems are ordinary here: clips land on scratch, datasets on project
        # storage. Fall back rather than fail, but say so once.
        shutil.copy2(src, dest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="LeRobot v2.1 root holding the real episodes")
    ap.add_argument("--clips", type=Path, action="append", default=[], help="clip directory (repeatable)")
    ap.add_argument("--work-list", type=Path, action="append", default=[], help="work.jsonl (repeatable)")
    ap.add_argument("--episode-map", type=Path, help='JSON {"<episode id>": <source index>}')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True, help="arm label recorded in PROVENANCE.json, e.g. arm-B")

    real = ap.add_mutually_exclusive_group(required=True)
    real.add_argument("--with-real", dest="with_real", action="store_true",
                      help="include every source episode (arms A, B, C)")
    real.add_argument("--without-real", dest="with_real", action="store_false",
                      help="restyled clips only (arm D)")

    ap.add_argument("--verify-frames", choices=("strict", "container", "none"), default="strict",
                    help="strict decodes every clip (slow, authoritative). none is a footgun.")
    ap.add_argument("--link", choices=("hard", "symlink", "copy"), default="hard")
    ap.add_argument("--image-stats", choices=("omit", "inherit"), default="omit",
                    help="episodes_stats.jsonl pixel stats. inherit is FALSE for restyled clips.")
    ap.add_argument("--dry-run", action="store_true", help="plan and verify, write nothing")
    args = ap.parse_args()

    if not args.clips and not args.with_real:
        raise SystemExit("FATAL: --without-real and no --clips leaves nothing to assemble.")

    src = read_source(args.source)
    n_source = len(src["episodes"])
    print(f"source: {args.source}")
    print(f"  {n_source} episodes, {src['info']['total_frames']} frames, "
          f"LeRobot {src['info']['codebase_version']}")

    episode_map = json.loads(args.episode_map.read_text()) if args.episode_map else None
    work = load_work_lists(args.work_list) if args.work_list else {}
    if work:
        print(f"  work list: {len(work)} units")

    units: list[Unit] = []
    if args.with_real:
        for episode in src["episodes"]:
            index = int(episode["episode_index"])
            units.append(Unit(source_index=index, video=source_paths(src, index)[1], origin="real"))
    clip_units = collect_clip_units(args.clips, work, episode_map, n_source) if args.clips else []
    units.extend(clip_units)

    print(f"plan '{args.label}': {len(units)} episodes "
          f"({len(units) - len(clip_units)} real + {len(clip_units)} restyled)")
    if not units:
        raise SystemExit("FATAL: nothing to assemble.")

    # ---- verify before writing anything -----------------------------------------------------
    report = Report()
    probes: dict[str, Probe] = {}
    print(f"verifying frame counts (--verify-frames {args.verify_frames}) ...")
    if args.verify_frames == "none":
        print("  WARNING: frame verification disabled. A dropped frame will silently offset every "
              "action after it.")

    for position, unit in enumerate(units):
        source_episode = src["by_index"].get(unit.source_index)
        if source_episode is None:
            report.fail(f"{unit.origin}: source episode {unit.source_index} not in episodes.jsonl")
            continue
        unit.frames = int(source_episode["length"])

        if not unit.video.exists():
            report.fail(f"{unit.origin}: video missing at {unit.video}")
            continue

        if args.verify_frames != "none":
            try:
                info = probe(unit.video, args.verify_frames)
            except RuntimeError as exc:
                report.fail(str(exc))
                continue
            probes[unit.origin] = info
            if info.frames != unit.frames:
                report.fail(
                    f"{unit.origin}: clip has {info.frames} frames, source episode "
                    f"{unit.source_index} has {unit.frames}. Pixels and actions would be "
                    f"misaligned by {info.frames - unit.frames}."
                )
        if position % 250 == 0 and position:
            print(f"  {position}/{len(units)}")

    # Geometry and codec must be uniform: info.json declares ONE video format for the dataset.
    if probes:
        shapes = Counter((p.height, p.width) for p in probes.values())
        codecs = Counter(p.codec for p in probes.values())
        if len(shapes) > 1:
            report.fail(f"clips disagree on resolution: {dict(shapes)}")
        if len(codecs) > 1:
            report.fail(f"clips disagree on codec: {dict(codecs)} -- info.json declares exactly one")

    if report.problems:
        print(f"\nFAIL -- {len(report.problems)} problem(s):")
        for problem in report.problems[:40]:
            print(f"  {problem}")
        if len(report.problems) > 40:
            print(f"  ... and {len(report.problems) - 40} more")
        return 1
    print("  all clips match their source episode's frame count")

    if args.dry_run:
        print("\n--dry-run: verified, nothing written.")
        return 0

    # ---- write -------------------------------------------------------------------------------
    out = args.out
    (out / "meta").mkdir(parents=True, exist_ok=True)
    chunks_size = int(src["info"]["chunks_size"])

    episodes_out: list[dict] = []
    stats_out: list[dict] = []
    provenance_units: list[dict] = []
    running_index = 0
    total_frames = 0

    source_stats: dict[int, dict] = {}
    if args.image_stats == "inherit" and src["episode_stats"].exists():
        for line in src["episode_stats"].read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                source_stats[int(row["episode_index"])] = row

    print(f"writing {len(units)} episodes -> {out}")
    for new_index, unit in enumerate(units):
        unit.new_index = new_index
        chunk = new_index // chunks_size

        src_parquet, _ = source_paths(src, unit.source_index)
        table = pq.read_table(src_parquet)
        if table.num_rows != unit.frames:
            raise SystemExit(
                f"FATAL: {src_parquet} has {table.num_rows} rows, episodes.jsonl says "
                f"{unit.frames}. The source dataset is internally inconsistent."
            )
        table = rewrite_parquet(table, new_index, running_index)

        dest_parquet = out / src["info"]["data_path"].format(
            episode_chunk=chunk, episode_index=new_index)
        dest_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest_parquet)

        dest_video = out / src["info"]["video_path"].format(
            episode_chunk=chunk, episode_index=new_index, video_key=VIDEO_KEY)
        place_video(unit.video, dest_video, args.link)

        episodes_out.append({
            "episode_index": new_index,
            "tasks": src["by_index"][unit.source_index]["tasks"],
            "length": unit.frames,
        })
        if args.image_stats == "inherit" and unit.source_index in source_stats:
            row = dict(source_stats[unit.source_index])
            row["episode_index"] = new_index
            stats_out.append(row)

        provenance_units.append({
            "episode_index": new_index,
            "source_episode_index": unit.source_index,
            "origin": unit.origin,
            "style": unit.style,
            "repeat": unit.repeat,
            "frames": unit.frames,
            "video": str(unit.video),
        })

        running_index += unit.frames
        total_frames += unit.frames
        if new_index % 250 == 0 and new_index:
            print(f"  {new_index}/{len(units)}")

    # ---- meta --------------------------------------------------------------------------------
    info = json.loads(json.dumps(src["info"]))  # deep copy
    info["codebase_version"] = CODEBASE_VERSION
    info["total_episodes"] = len(units)
    info["total_frames"] = total_frames
    info["total_videos"] = len(units)
    info["total_chunks"] = (len(units) + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{len(units)}"}

    if probes:
        sample = next(iter(probes.values()))
        video_info = info["features"][VIDEO_KEY]["info"]
        before = (video_info["video.height"], video_info["video.width"], video_info["video.codec"])
        after = (sample.height, sample.width, sample.codec)
        if before != after:
            print(f"  video format changed by the restyle: {before} -> {after}; info.json updated")
            video_info["video.height"] = sample.height
            video_info["video.width"] = sample.width
            video_info["video.codec"] = sample.codec
            video_info["video.pix_fmt"] = sample.pix_fmt
            info["features"][VIDEO_KEY]["shape"] = [sample.height, sample.width, 3]

    meta = out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_out))
    (meta / "tasks.jsonl").write_text(src["tasks_raw"])
    (meta / "modality.json").write_text(src["modality_raw"])
    if stats_out:
        (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats_out))

    provenance = {
        "label": args.label,
        "source": str(args.source.resolve()),
        "source_info_sha256": sha256_file(args.source / "meta/info.json"),
        "source_modality_sha256": sha256_file(args.source / "meta/modality.json"),
        "with_real": args.with_real,
        "clip_dirs": [str(c.resolve()) for c in args.clips],
        "work_lists": [str(w.resolve()) for w in args.work_list],
        "verify_frames": args.verify_frames,
        "image_stats": args.image_stats,
        "episodes": len(units),
        "episodes_real": len(units) - len(clip_units),
        "episodes_restyled": len(clip_units),
        "frames": total_frames,
        # The label-integrity witness G0a needs: which source episode every output episode's
        # actions came from.
        "units": provenance_units,
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"\nWROTE {len(units)} episodes / {total_frames} frames -> {out}")
    if args.image_stats == "omit":
        print("  meta/episodes_stats.jsonl omitted (--image-stats omit): inheriting a real "
              "episode's pixel statistics for a restyled clip would be false. GR00T does not "
              "read this file; meta/stats.json is what it asserts on.")
    print("\nNEXT -- meta/stats.json is a hard assert in GR00T's loader and is NOT written here:")
    print("  ~/Isaac-GR00T/.venv/bin/python -m gr00t.data.stats \\")
    print(f"    --dataset-path {out} \\")
    print("    --embodiment-tag NEW_EMBODIMENT \\")
    print(f"    --modality-config-path {Path('configs/groot/new_embodiment_config_defaults.py').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
