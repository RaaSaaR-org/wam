#!/usr/bin/env python3
"""Build the G0b calibration clip set for T-041 (PR-09 §6), at the geometry the judge must see.

WHY THIS EXISTS. G0b asks one question of the rubric: on REAL footage, where the answer is known
without anyone adjudicating a generated frame, can it separate a Unitree Dex3 hand from a hand that
is definitively not one? 10 positives must score YES, 10 negatives must score NO, all 20 or the
verdict is VOID (``eval_t041_embodiment.py:42``). That check is only worth running if the two sides
differ in *embodiment* and in nothing else a vision model can latch onto.

Resolution is the one that nearly got through. ``configs/cosmos3/t041_eval_selection.toml`` pins
generation at ``resolution = "256"``, which is a key into ``VIDEO_RES_SIZE_INFO`` and resolves to
320x256 — the geometry the LoRA was fit in, because ``vision_sft_super.py`` pins the same key for
the SFT dataset. The calibration clips are real recordings: 640x480 from our own robot, 256x256
from BridgeData2. Handing the judge 640x480 positives and 256x256 negatives would let a rubric that
cannot see a hand at all still score 20/20 by reading the frame size, and G0b would certify
nothing. So both sides are put through the SAME transform onto the SAME 320x256 grid, and that
transform is the training dataloader's own: scale-to-cover, then centre-crop
(``sft_dataset.py:192-195``). The TOML's ``[generation] resolution`` note used to end by saying
this was not done yet and that G0b must not run until it is; this script is what retired that
sentence, and the ``[calibration]`` section now points here instead.

DURATION IS THE SECOND CONFOUND, and it is handled the same way. The judge also scores 60 generated
clips, each ``num_frames`` long at ``fps``. A calibration set of 30 s recordings would calibrate the
rubric in a regime the measurement never enters, so every clip here is cut to the same *duration*
from the centre of its source. Centre, not start: the opening seconds of a teleoperated episode are
the arm leaving home, and a rubric asked about an end effector that has not entered frame yet is
being asked a different question.

THAT DURATION IS NEVER WRITTEN DOWN HERE. It is ``num_frames / fps`` read out of the TOML at run
time, because the two have already desynchronised once: the set was first built at 189 frames =
6.3 s, ``num_frames`` was then amended to 397 = 13.23 s, and nothing in the calibration set would
have objected. A judge calibrated on 6.3 s of footage and then applied to 13.23 s of footage has
been calibrated on a different measurement. Hardcoding the seconds here would make that silent
again, so the coupling is the code path, and MANIFEST.json records the ``num_frames`` it came from.

WHAT IT REFUSES TO DO. It will not substitute another dataset if the pre-registered negative repo
is unreachable — the repo is named in the TOML and swapping it is choosing negatives after seeing
which ones are convenient. It will not write a clip that ffprobe does not confirm as 320x256, at
the requested frame count, at the duration that frame count implies. It will not draw positives
from a source whose LeRobot root is not a ``G1_Dex3_*`` dataset, so the GR00T AppleToPlate footage
in the same corpus cannot leak in as a "G1" positive. It will not pick a positive whose bytes also
appear in the val split, which is where the eval's 30 prompts come from. And it will not use a
selected positive whose file is missing from disk or whose bytes no longer hash to what the corpus
manifest says they do — the corpus is edited in place by ``dedupe_cosmos_corpus.py``, so "it was
there last time" is not evidence.

SOURCES TOO SHORT FOR THE WINDOW ARE SKIPPED, NOT TRUNCATED, on both sides. A clip cut short would
be the one item of a different length in a set whose entire purpose is that length is not a cue.
The skip is part of the deterministic rule — walk the same sorted list, pass over what cannot fill
the window, take the next — and every skip is recorded in MANIFEST.json with its measured duration,
because a selection rule that quietly drops candidates is a selection rule nobody can recompute.

    FRAMEWORK=~/wam-t041/third_party/cosmos/packages/cosmos3 \
        ~/wam-t041/third_party/cosmos/packages/cosmos3/.venv/bin/python \
        scripts/build_t041_calibration.py \
        --corpus ~/wam-t041/cosmos-g1-embodiment --out ~/wam-t041/t041-calibration \
        --hf-cache ~/wam-t041/hf-cache

Needs ``huggingface_hub`` and ``ffmpeg``/``ffprobe`` on PATH; no GPU, no cluster. The system python3
has no ``huggingface_hub``, hence the framework venv above. Idempotent: a second run re-probes and
re-hashes what is already there and rewrites MANIFEST.json unchanged — but an output that does not
match the *current* window is re-encoded rather than kept, so changing ``num_frames`` and re-running
is enough.
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
import tomllib
from datetime import date, datetime, timezone

# The one entry of cosmos_framework's VIDEO_RES_SIZE_INFO this script depends on. It is duplicated
# here so the script runs on a workstation with no framework checkout, and cross-checked against
# the real table whenever one is reachable (--framework) — a copy that can drift silently would be
# worse than no copy at all.
PINNED_RES_ENTRY = {("256", "4,3"): (320, 256)}

# Matches ffmpeg_decode_video (local_datasets/helper.py:138-139). Training resampled every frame
# the adapter ever saw with these flags; the calibration clips have no reason to use different ones.
SWS_FLAGS = "bicubic+accurate_rnd"


def window_frames(clip_seconds: float, fps: float) -> int:
    """How many frames ``clip_seconds`` is at ``fps`` — the same arithmetic everywhere.

    The window is pinned in SECONDS, not frames, so a 5 fps source keeps its own temporal sampling
    instead of having frames invented for it. Every place that needs a frame count derives it here,
    so the count a source is screened against and the count ffmpeg is asked for cannot diverge.
    """
    return round(clip_seconds * fps)


# --------------------------------------------------------------------------------------------
# geometry — the dataloader's, not an approximation of it
# --------------------------------------------------------------------------------------------

def resolve_target(cfg: dict, framework: pathlib.Path | None) -> tuple[tuple[int, int], str]:
    """(width, height) the generated clips will have, from the pre-registered sampler settings."""
    gen = cfg["generation"]
    key = (str(gen["resolution"]), str(gen["aspect_ratio"]))
    pinned = PINNED_RES_ENTRY.get(key)
    if pinned is None:
        raise SystemExit(
            f"FATAL: this script pins only {sorted(PINNED_RES_ENTRY)} of VIDEO_RES_SIZE_INFO and "
            f"the config asks for {key}. Add the entry deliberately; do not let it be guessed."
        )
    table_src = "pinned copy in build_t041_calibration.py"
    if framework is not None:
        live = _read_res_table(framework)
        got = tuple(live[key[0]][key[1]])
        if got != pinned:
            raise SystemExit(
                f"FATAL: {framework} says VIDEO_RES_SIZE_INFO{list(key)} = {got}, this script has "
                f"{pinned}. The clips would be built for a geometry the model does not generate."
            )
        table_src = str(framework)
    return pinned, table_src


def _read_res_table(framework: pathlib.Path) -> dict:
    """Lift VIDEO_RES_SIZE_INFO out of the framework source with ast, rather than importing it.

    Importing ``cosmos_framework.data.generator.utils`` drags in torch and the whole package. This
    script has to run on a workstation whose only job is ffmpeg, so it reads the literal instead.
    """
    import ast

    path = framework / "cosmos_framework" / "data" / "generator" / "utils.py"
    if not path.is_file():
        raise SystemExit(f"FATAL: --framework given but {path} does not exist.")
    tree = ast.parse(path.read_text())
    for node in tree.body:
        targets = getattr(node, "targets", []) or ([node.target] if hasattr(node, "target") else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "VIDEO_RES_SIZE_INFO":
                return ast.literal_eval(node.value)
    raise SystemExit(f"FATAL: no VIDEO_RES_SIZE_INFO assignment in {path}.")


def cover_crop_filter(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> str:
    """The training dataloader's spatial transform, as an ffmpeg filter string.

    ``sft_dataset.py:192-195`` computes ``max`` of the two scale ratios (cover, never letterbox),
    rounds the scaled size, and centre-crops. ffmpeg's ``crop`` defaults to the same centre offset,
    but with integer division rather than ``round``; where they disagree this refuses instead of
    quietly shifting the frame by a pixel relative to what the adapter was trained on.
    """
    ratio = max(tgt_w / src_w, tgt_h / src_h)
    rw, rh = round(src_w * ratio), round(src_h * ratio)
    if rw < tgt_w or rh < tgt_h:
        raise SystemExit(f"FATAL: cover scale {src_w}x{src_h} -> {rw}x{rh} does not cover "
                         f"{tgt_w}x{tgt_h}; rounding went the wrong way.")
    torch_x, torch_y = round((rw - tgt_w) / 2), round((rh - tgt_h) / 2)
    ff_x, ff_y = (rw - tgt_w) // 2, (rh - tgt_h) // 2
    if (torch_x, torch_y) != (ff_x, ff_y):
        raise SystemExit(
            f"FATAL: centre crop of {rw}x{rh} -> {tgt_w}x{tgt_h} is ({torch_x},{torch_y}) in the "
            f"dataloader and ({ff_x},{ff_y}) in ffmpeg. Pass the offsets explicitly before using "
            "this source geometry."
        )
    return f"scale={rw}:{rh}:flags={SWS_FLAGS},crop={tgt_w}:{tgt_h}"


# --------------------------------------------------------------------------------------------
# ffmpeg / ffprobe
# --------------------------------------------------------------------------------------------

def _tool(name: str) -> str:
    p = shutil.which(name)
    if p is None:
        raise SystemExit(f"FATAL: {name} not on PATH.")
    return p


def probe(path: pathlib.Path) -> dict:
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    obj = json.loads(out)
    if not obj.get("streams"):
        raise SystemExit(f"FATAL: {path} has no video stream.")
    st = obj["streams"][0]
    num, _, den = st["r_frame_rate"].partition("/")
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": int(num) / int(den or 1),
        "frames": int(st["nb_frames"]) if st.get("nb_frames") else None,
        "codec": st.get("codec_name"),
        "duration_s": round(float(obj["format"]["duration"]), 4),
    }


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transcode(src: pathlib.Path, dst: pathlib.Path, vf: str, start_s: float, frames: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    cmd = [_tool("ffmpeg"), "-y", "-v", "error", "-nostdin",
           "-ss", f"{start_s:.6f}", "-i", str(src),
           "-frames:v", str(frames), "-vf", vf, "-sws_flags", SWS_FLAGS,
           # CRF 18 is visually lossless at this size; the judge must not be shown compression
           # artefacts that correlate with which side of the calibration set a clip is on.
           "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-fps_mode", "passthrough", "-an", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


# --------------------------------------------------------------------------------------------
# positives — real G1 + Dex3, drawn from the prepared corpus
# --------------------------------------------------------------------------------------------

def g1_dex3_sources(manifest: dict) -> list[str]:
    """Source ids whose LeRobot root is a ``G1_Dex3_*`` dataset.

    By root directory, not by source id spelling. The corpus also holds GR00T AppleToPlate, which
    is a humanoid but not a G1 with Dex3 hands, and a positive set that quietly included it would
    be calibrating the rubric against the wrong ground truth.
    """
    out = []
    for sid, meta in manifest["sources"].items():
        if pathlib.PurePath(meta["root"]).name.startswith("G1_Dex3_"):
            out.append(sid)
    return sorted(out)


def collapse_duplicate_sources(clips: list[dict], sources: list[str]) -> tuple[list[str], list[dict]]:
    """Drop a source whose clip bytes are wholly contained in another source's.

    This corpus really does contain one: every clip of ``g1-dex3-graspsquare-dataset`` is
    byte-identical (same sha256) to the same-numbered clip of ``g1-dex3-blockstacking-dataset``.
    Two positives drawn one from each would look like two tasks and be one video, which is exactly
    the kind of fake diversity a calibration set must not have. Keep the lexicographically first.

    ``clips`` must be BOTH splits. Comparing only train misses this pair: the random split sent two
    blockstacking clips and two different graspsquare clips to val, so each source's train half
    holds two shas the other's does not and neither is a subset of the other. The duplication is a
    property of the recordings, not of the split, so it has to be measured on all of them.
    """
    by_src = {s: {c["sha256"] for c in clips if c["source_id"] == s} for s in sources}
    dropped, kept = [], []
    for s in sources:
        dup_of = next((o for o in kept if by_src[s] and by_src[s] <= by_src[o]), None)
        if dup_of is None:
            kept.append(s)
        else:
            dropped.append({"source_id": s, "duplicate_of": dup_of, "n_clips": len(by_src[s])})
    return kept, dropped


def select_positives(manifest: dict, n: int,
                     clip_seconds: float) -> tuple[list[dict], list[dict]]:
    """Deterministic: one clip per G1_Dex3 source, sources alphabetical, first uuid within each.

    No sampling and no seed, for the same reason the eval's own prompt rule has none — a knob here
    is a knob that could be turned after seeing how the judge scored. Three exclusions apply, all
    stated as rules rather than applied clip by clip:

    - anything whose bytes appear in ``val``. The eval's 30 prompts are the whole val split, so
      this is content-level disjointness from the measurement — the same question
      ``make_t041_eval_prompts.py`` asks from the other side, and needed for the same reason: a
      uuid is a filename, and this corpus shipped one source as a byte-copy of another;
    - anything whose bytes were already selected;
    - anything shorter than the window, screened from the manifest's own frame counts before any
      decoding. 443 of 3133 train clips are under 397 frames, so this is a live hazard and not a
      formality — and the alternative, letting ffmpeg emit whatever it can reach, is the failure
      this whole function exists to make impossible.

    A source with no clip that passes is SKIPPED and reported, not fatal. The corpus really does
    contain one now: ``dedupe_cosmos_corpus.py`` found every ``g1-dex3-graspsquare-dataset`` train
    clip to be a byte-duplicate and removed all of them, leaving the source with two val clips and
    nothing selectable. Refusing outright would mean a re-run of the dedupe tool breaks the
    calibration build; taking the next source along keeps the "one clip per source" diversity the
    rule is for. What is NOT allowed is dropping below ``n``, because two clips from one source
    would be two views of one task pretending to be two pieces of evidence.
    """
    train, val = manifest["clips"]["train"], manifest["clips"]["val"]
    val_sha = {c["sha256"] for c in val}
    sources = g1_dex3_sources(manifest)
    sources, dropped = collapse_duplicate_sources(train + val, sources)
    skipped = [{"source_id": d["source_id"], "reason": "byte-duplicate source",
                "duplicate_of": d["duplicate_of"], "n_clips": d["n_clips"]} for d in dropped]
    for d in dropped:
        print(f"note: source {d['source_id']} dropped — its {d['n_clips']} clips are byte-"
              f"identical to {d['duplicate_of']}", file=sys.stderr)

    chosen: list[dict] = []
    seen: set[str] = set()
    for sid in sources:
        if len(chosen) == n:
            break
        pool = sorted((c for c in train if c["source_id"] == sid), key=lambda c: c["uuid"])
        eligible = [c for c in pool
                    if c["sha256"] not in val_sha and c["sha256"] not in seen
                    and c["frames"] >= window_frames(clip_seconds, c["fps"])]
        if not eligible:
            long_enough = sum(1 for c in pool
                              if c["frames"] >= window_frames(clip_seconds, c["fps"]))
            reason = ("no train clips in the manifest" if not pool else
                      f"no train clip both unused and >= {clip_seconds}s "
                      f"({long_enough}/{len(pool)} are long enough)")
            skipped.append({"source_id": sid, "reason": reason, "n_train_clips": len(pool)})
            print(f"note: source {sid} skipped — {reason}", file=sys.stderr)
            continue
        seen.add(eligible[0]["sha256"])
        chosen.append(eligible[0])

    if len(chosen) < n:
        raise SystemExit(
            f"FATAL: only {len(chosen)} of {len(sources)} G1_Dex3 sources yielded a clip that fills "
            f"the {clip_seconds}s window, need {n}. Do not take two from one source without saying "
            "so — task diversity is the only reason to prefer this rule."
        )
    return chosen, skipped


# --------------------------------------------------------------------------------------------
# negatives — the pre-registered BridgeData2 subset, and nothing else
# --------------------------------------------------------------------------------------------

def fetch_negatives(repo: str, n: int, cache_dir: pathlib.Path, revision: str | None,
                    clip_seconds: float
                    ) -> tuple[str, list[tuple[str, pathlib.Path]], list[dict]]:
    """Sorted-then-first-n over the repo's train videos, passing over anything too short.

    BridgeData2 episodes are 13-19 s, so at a 13.23 s window the rule is no longer safely inside
    the source material the way 6.3 s was — some candidates cannot fill it. The skip keeps the rule
    recomputable by anyone: same sorted list, same predicate, no judgement. It cannot be applied
    from a listing, though, because file length is not in the repo metadata, so each candidate is
    downloaded and probed in list order and the walk stops at ``n`` accepted. That is why this
    fetches one file at a time instead of one snapshot: a batch large enough to survive the skips
    would be a batch chosen by guessing how many skips there will be.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("FATAL: huggingface_hub is not importable. Install it, or run this with "
                         "the Cosmos venv's python; do not hand-copy the negatives.")
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = HfApi().repo_info(repo, repo_type="dataset", revision=revision)
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"FATAL: cannot read dataset {repo}: {exc}\n"
            "The negative source is pre-registered in configs/cosmos3/t041_eval_selection.toml "
            "(negative_repo). STOP and resolve access. Substituting another manipulator dataset "
            "here would mean choosing the negatives after discovering which ones are available."
        )
    if revision is not None and info.sha != revision:
        raise SystemExit(f"FATAL: asked for revision {revision}, hub resolved {info.sha}.")

    # One clip per file in this repo — no shards, so a per-file download fetches exactly what the
    # walk reaches and nothing of the other ~1300.
    names = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.startswith("sft_dataset_bridge/train/videos/")
                   and s.rfilename.endswith(".mp4"))
    if len(names) < n:
        raise SystemExit(f"FATAL: {repo} exposes {len(names)} train videos, need {n}.")

    taken: list[tuple[str, pathlib.Path]] = []
    skipped: list[dict] = []
    for rel in names:
        if len(taken) == n:
            break
        local = pathlib.Path(hf_hub_download(repo, rel, repo_type="dataset", revision=info.sha,
                                             cache_dir=str(cache_dir)))
        got = probe(local)
        if got["frames"] is not None and got["frames"] < window_frames(clip_seconds, got["fps"]):
            skipped.append({"hf_file": rel, "reason": "shorter than the window",
                            "source_duration_s": got["duration_s"], "source_fps": got["fps"],
                            "source_frames": got["frames"],
                            "frames_needed": window_frames(clip_seconds, got["fps"])})
            print(f"note: negative {rel} skipped — {got['frames']} frames at {got['fps']} fps, "
                  f"needs {window_frames(clip_seconds, got['fps'])}", file=sys.stderr)
            continue
        taken.append((rel, local))
    if len(taken) < n:
        raise SystemExit(
            f"FATAL: walked all {len(names)} train videos in {repo} and only {len(taken)} fill the "
            f"{clip_seconds}s window, need {n}."
        )
    return info.sha, taken, skipped


# --------------------------------------------------------------------------------------------

def build_one(src: pathlib.Path, dst: pathlib.Path, target: tuple[int, int],
              clip_seconds: float, force: bool) -> dict:
    """Cut, rescale and verify one clip; return its MANIFEST record."""
    info = probe(src)
    vf = cover_crop_filter(info["width"], info["height"], *target)
    # At the corpus's 30 fps this is exactly the generated clips' num_frames; at BridgeData2's 5 fps
    # it is the same number of seconds carried by fewer frames, which is the intended behaviour.
    frames = window_frames(clip_seconds, info["fps"])
    if info["frames"] is not None and info["frames"] < frames:
        raise SystemExit(
            f"FATAL: {src} has {info['frames']} frames at {info['fps']} fps and the window needs "
            f"{frames}. Callers are supposed to screen this out before getting here; a clip cut "
            "short would be the one item of a different length in the set."
        )
    start = (info["duration_s"] - clip_seconds) / 2

    # Re-encode not only on --force but whenever what is on disk is not what the CURRENT window
    # asks for. num_frames moved from 189 to 397 while a built set was sitting in this directory,
    # and "the file exists" would have kept every 6.3 s clip of it.
    stale = None
    if dst.is_file() and not force:
        have = probe(dst)
        if (have["width"], have["height"]) != target or have["frames"] != frames:
            stale = (f"{have['width']}x{have['height']} {have['frames']}f -> "
                     f"{target[0]}x{target[1]} {frames}f")
            print(f"  ! re-encoding {dst.name}: {stale}", file=sys.stderr)
    if force or stale or not dst.is_file():
        transcode(src, dst, vf, start, frames)

    out = probe(dst)
    expected_duration = frames / info["fps"]
    if (out["width"], out["height"]) != target:
        dst.unlink(missing_ok=True)
        raise SystemExit(f"FATAL: {dst} came out {out['width']}x{out['height']}, wanted "
                         f"{target[0]}x{target[1]}. Removed it rather than leave it for the judge.")
    # ffmpeg does not fail when the input runs out before -frames:v is satisfied; it writes what it
    # reached and exits 0. Every screening rule above assumes that never happens, so it is checked
    # rather than assumed, on both sides of the set and on every single clip.
    if out["frames"] != frames:
        dst.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: {dst.name} came out {out['frames']} frames, asked for {frames} from a source "
            f"of {info['frames']} at {info['fps']} fps starting {start:.6f}s. Removed it. A short "
            "clip in the calibration set is a length cue the judge can score instead of a hand."
        )
    if abs(out["duration_s"] - expected_duration) > 0.5 / info["fps"]:
        dst.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: {dst.name} is {out['duration_s']}s, expected {expected_duration:.4f}s "
            f"({frames} frames at {info['fps']} fps). Removed it."
        )
    return {
        "output": dst.name,
        "source_width": info["width"],
        "source_height": info["height"],
        "source_fps": info["fps"],
        "source_duration_s": info["duration_s"],
        "window_start_s": round(start, 4),
        "frames_requested": frames,
        "ffmpeg_vf": vf,
        "sws_flags": SWS_FLAGS,
        "output_width": out["width"],
        "output_height": out["height"],
        "output_fps": out["fps"],
        "output_frames": out["frames"],
        "output_duration_s": out["duration_s"],
        "sha256": sha256_file(dst),
    }


def prune_stale(directory: pathlib.Path, keep: list[dict]) -> None:
    """Delete mp4s the current rules did not produce.

    ``build_sheet`` takes ``sorted(glob('*.mp4'))[:n]`` (eval_t041_embodiment.py:114-120), so a
    leftover from an earlier rule does not sit harmlessly beside the real set — it can displace a
    clip out of the first n and into nothing, silently changing what G0b measured.
    """
    wanted = {r["output"] for r in keep}
    for path in sorted(directory.glob("*.mp4")):
        if path.name not in wanted:
            path.unlink()
            print(f"  x removed stale {path.name}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path(os.environ["DATASET_PATH"])
                    if os.environ.get("DATASET_PATH") else None,
                    help="Prepared corpus root (holds manifest.json). Defaults to $DATASET_PATH.")
    ap.add_argument("--config", type=pathlib.Path,
                    default=repo_root / "configs" / "cosmos3" / "t041_eval_selection.toml")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="Calibration root; positive/ and negative/ are created under it.")
    ap.add_argument("--framework", type=pathlib.Path,
                    default=pathlib.Path(os.environ["FRAMEWORK"]) if os.environ.get("FRAMEWORK")
                    else None,
                    help="cosmos3 package root, to cross-check VIDEO_RES_SIZE_INFO.")
    ap.add_argument("--hf-cache", type=pathlib.Path, default=None,
                    help="huggingface_hub cache_dir for the negatives (default: <out>/../hf-cache).")
    ap.add_argument("--hf-revision", default=None,
                    help="Override [calibration] negative_revision; refuses if the hub disagrees. "
                         "For investigation only — the pinned revision is the registered one.")
    ap.add_argument("--force", action="store_true", help="Re-encode clips that already exist.")
    args = ap.parse_args(argv)

    if args.corpus is None:
        raise SystemExit("FATAL: --corpus not given and $DATASET_PATH is unset.")
    cfg = tomllib.loads(args.config.read_text())
    cal, gen, sel = cfg["calibration"], cfg["generation"], cfg["selection"]
    manifest = json.loads((args.corpus / "manifest.json").read_text())

    # Same refusal as make_t041_eval_prompts.py:83-88, for the same reason: the positives are
    # chosen relative to a val split, and a corpus split with a different seed has a different one.
    if manifest["seed"] != sel["corpus_seed"]:
        raise SystemExit(
            f"FATAL: corpus split seed {manifest['seed']} != pre-registered {sel['corpus_seed']}. "
            "The val split this excludes positives against is not the one the eval draws from.")

    # The corpus manifest is edited in place — dedupe_cosmos_corpus.py rewrites it — and this build
    # is a function of its contents. Recording the hash is not enough on its own; where the corpus
    # ships the hash it expects, disagreeing with it means one of the two is from a different run.
    corpus_manifest_sha = hashlib.sha256((args.corpus / "manifest.json").read_bytes()).hexdigest()
    sidecar = args.corpus / "MANIFEST_SHA256"
    if sidecar.is_file() and sidecar.read_text().split()[0] != corpus_manifest_sha:
        raise SystemExit(
            f"FATAL: {sidecar} says {sidecar.read_text().split()[0]} but manifest.json hashes to "
            f"{corpus_manifest_sha}. The corpus was edited without its sidecar being updated; "
            "resolve which one is current before drawing positives from it.")

    target, table_src = resolve_target(cfg, args.framework)
    # DERIVED, NEVER TYPED. See the module docstring: the calibration window is the generated clips'
    # window or it is calibrating on footage the measurement does not produce. Changing num_frames
    # in the TOML changes this, and the re-encode check in build_one makes the change take effect.
    clip_seconds = round(gen["num_frames"] / gen["fps"], 6)
    print(f"target {target[0]}x{target[1]} ({table_src}); {clip_seconds}s per clip "
          f"({gen['num_frames']} frames at {gen['fps']} fps)", file=sys.stderr)

    # The [calibration] keys this script implements, checked against what it actually does. A
    # config key nothing reads is a comment that looks like a commitment: it can say "positives come
    # from train" long after the code stopped doing that, and nothing anywhere would disagree.
    expect = {
        "positive_split": "train",
        "positive_rule": "one clip per G1_Dex3_* source, sources alphabetical, first uuid "
                         "within each",
        "window_seconds_from": "generation.num_frames / generation.fps",
        "normalise_to": f"{target[0]}x{target[1]}",
    }
    for key, want in expect.items():
        if cal.get(key) != want:
            raise SystemExit(
                f"FATAL: [calibration] {key} = {cal.get(key)!r}, this script implements {want!r}. "
                "Change the code to match the pre-registration or amend the pre-registration "
                "deliberately; do not let the two describe different calibration sets.")

    pos_dir, neg_dir = args.out / "positive", args.out / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # -- positives ---------------------------------------------------------------------------
    positives = []
    chosen, pos_skipped = select_positives(manifest, cal["n_positive"], clip_seconds)
    train_uuids = {c["uuid"] for c in manifest["clips"]["train"]}
    for i, clip in enumerate(chosen):
        # Three separate questions, because the corpus is mutable and each has failed somewhere:
        # is this uuid still in the manifest's train list, is the file still on disk, and are its
        # bytes still the bytes the manifest attributes to that uuid. The last one is what makes
        # "positive is not a val clip under another name" mean anything — that test was done on
        # manifest shas, so a file whose content has drifted from its manifest entry was never
        # actually checked.
        if clip["uuid"] not in train_uuids:
            raise SystemExit(f"FATAL: selected positive {clip['uuid']} is not in the corpus "
                             "manifest's train split.")
        if clip["source_id"] == cal.get("positive_excludes_source"):
            raise SystemExit(
                f"FATAL: {clip['uuid']} comes from {clip['source_id']}, which [calibration] "
                "positive_excludes_source names as not a G1 with Dex3 hands. The root-directory "
                "filter that should have caught this did not.")
        src = args.corpus / "train" / "videos" / f"{clip['uuid']}.mp4"
        if not src.is_file():
            raise SystemExit(f"FATAL: {src} is in the manifest but not on disk.")
        on_disk = sha256_file(src)
        if on_disk != clip["sha256"]:
            raise SystemExit(
                f"FATAL: {src} hashes to {on_disk}, the manifest says {clip['sha256']}. Every "
                "eligibility test above was run against the manifest's hash, so this file was "
                "never the one that passed them.")
        dst = pos_dir / f"pos_{i:02d}_{clip['uuid']}.mp4"
        rec = build_one(src, dst, target, clip_seconds, args.force)
        rec |= {"source_path": str(src), "corpus_uuid": clip["uuid"],
                "corpus_source_id": clip["source_id"], "corpus_split": "train",
                "corpus_clip_sha256": clip["sha256"], "expected_answer": "YES"}
        positives.append(rec)
        print(f"  + {dst.name}", file=sys.stderr)
    prune_stale(pos_dir, positives)

    # -- negatives ---------------------------------------------------------------------------
    cache = args.hf_cache or (args.out.parent / "hf-cache")
    # The pin lives in the config, where it is reviewable, not in whatever the last person typed on
    # the command line. --hf-revision only exists to override it while investigating, and it says
    # so; a run that used the override is a run whose negatives are not the pre-registered ones.
    pinned_rev = args.hf_revision or cal.get("negative_revision")
    if args.hf_revision and cal.get("negative_revision") \
            and args.hf_revision != cal["negative_revision"]:
        print(f"note: --hf-revision {args.hf_revision} overrides the pre-registered "
              f"{cal['negative_revision']}", file=sys.stderr)
    revision, files, neg_skipped = fetch_negatives(cal["negative_repo"], cal["n_negative"], cache,
                                                   pinned_rev, clip_seconds)
    negatives = []
    for i, (rel, src) in enumerate(files):
        dst = neg_dir / f"neg_{i:02d}_bridge_{pathlib.Path(rel).stem}.mp4"
        rec = build_one(src, dst, target, clip_seconds, args.force)
        rec |= {"hf_repo": cal["negative_repo"], "hf_revision": revision, "hf_file": rel,
                "source_sha256": sha256_file(src), "expected_answer": "NO"}
        negatives.append(rec)
        print(f"  - {dst.name}", file=sys.stderr)
    prune_stale(neg_dir, negatives)

    # -- manifest ------------------------------------------------------------------------------
    ffmpeg_v = subprocess.run([_tool("ffmpeg"), "-version"], capture_output=True,
                              text=True, check=True).stdout.splitlines()[0]
    doc = {
        "built_date": date.today().isoformat(),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_by": "scripts/build_t041_calibration.py",
        "purpose": "PR-09 §6 G0b calibration set for T-041",
        "ffmpeg_version": ffmpeg_v,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "corpus": str(args.corpus),
        "corpus_manifest_sha256": corpus_manifest_sha,
        "corpus_train_clips": len(manifest["clips"]["train"]),
        "corpus_val_clips": len(manifest["clips"]["val"]),
        "target_width": target[0],
        "target_height": target[1],
        "video_res_size_info_source": table_src,
        "clip_seconds": clip_seconds,
        "clip_seconds_from": (f"[generation] num_frames {gen['num_frames']} / fps {gen['fps']} in "
                              f"{args.config.name} — never hardcoded here"),
        "generation_num_frames": gen["num_frames"],
        "generation_fps": gen["fps"],
        "transform": ("scale-to-cover then centre-crop to the target, sws_flags "
                      f"{SWS_FLAGS} — sft_dataset.py:192-195, applied identically to both sides"),
        "positive_rule": ("one clip per G1_Dex3_* source, sources alphabetical, first uuid within "
                          "each, from the TRAIN split; byte-duplicate sources collapsed; clips "
                          "whose bytes appear in val excluded; clips too short for clip_seconds "
                          "excluded, and a source with none left is skipped for the next source"),
        "negative_rule": ("sorted sft_dataset_bridge/train/videos/*.mp4, walked in order, taking "
                          "the first n that are at least clip_seconds long"),
        "positive_skipped": pos_skipped,
        "negative_skipped": neg_skipped,
        "clips": {"positive": positives, "negative": negatives},
    }
    out = args.out / "MANIFEST.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}: {len(positives)} positive + {len(negatives)} negative, all "
          f"{target[0]}x{target[1]}, all {clip_seconds}s; skipped {len(pos_skipped)} positive "
          f"source(s) and {len(neg_skipped)} negative candidate(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
