#!/usr/bin/env python3
"""The masker-independent auxiliary of `T40_RULE_V15` §6: how far each empty-mask frame sits from
its own episode's static background.

    PYTHONPATH=src:scripts .venv/bin/python scripts/measure_empty_mask_motion.py \
        --out runs/pr08-empty-mask-look/MOTION.json

THE INSTRUMENT
--------------
Per episode, the **pixel-wise median over its own frames** is a model of the static scene: the cloth,
the table edge, the wall. The arm moves through that scene and the median is unmoved by it, so a
frame containing the arm deviates from the median far more than a frame that does not. Nothing here
uses a detector, a segmenter, a threshold of the masker's, or the masker's output in any form --
which is what V15 §6 requires of it. POOLED.json is read only to say WHICH frames to report on,
never what they contain.

**Two confounds, both recorded rather than corrected.** The apple is carried during the episode, so
it is not fully static and contributes deviation of its own; it covers roughly 1.5 % of the frame
against an arm's much larger footprint, so the two are separable by magnitude but not by this
instrument alone. And ``ego_view`` is head-mounted: if the head moves, the whole scene deviates at
once, which shows up as a near-total ``frac_dev`` and is visible as such.

WHAT THIS IS NOT
----------------
**Not a verdict, and not admissible as one.** V15 §6 permits this to speak for the corpus ONLY if
it reaches balanced accuracy >= 0.90 against held-out human labels, it may never override a human
label, and no outcome of V15 §5 may be evaluated on it. This script writes numbers; whether they
mean anything is decided elsewhere, after the human sample exists.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_THRESHOLD = 25          # grey levels; a pixel deviating by more than this is "changed"
BACKGROUND_STRIDE = 5       # every 5th frame builds the median; the scene is static


def episode_features(frames: np.ndarray, empty_indices: list[int]) -> list[dict]:
    grey = frames.astype(np.float32).mean(axis=3)                       # [F, H, W]
    background = np.median(grey[::BACKGROUND_STRIDE], axis=0)           # [H, W]
    out = []
    for index in empty_indices:
        deviation = np.abs(grey[index] - background)
        out.append({
            "frame_index": index,
            "frac_dev": float((deviation > DEV_THRESHOLD).mean()),
            "mad_dev": float(deviation.mean()),
            "p99_dev": float(np.percentile(deviation, 99)),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", type=pathlib.Path,
                        default=REPO_ROOT / "runs/pr08-robot-mask-area/POOLED.json")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "runs/pr08-empty-mask-look/MOTION.json")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import robot_composite as rc  # noqa: PLC0415

    pooled = json.loads(args.pooled.read_text())
    if not pooled.get("measurement_qualified"):
        raise SystemExit(f"{args.pooled}: measurement_qualified is not true")

    per_episode = []
    total = len(pooled["per_episode"])
    for done, episode in enumerate(sorted(pooled["per_episode"], key=lambda e: e["episode"]), 1):
        name = episode["episode"]
        empty = [i for i, v in enumerate(episode["area_fractions"]) if v == 0.0]
        if not empty:
            continue
        frames = rc.decode_clip(args.corpus / "videos" / f"{name}.mp4")
        if len(frames) != len(episode["area_fractions"]):
            raise SystemExit(
                f"{name}: video has {len(frames)} frames, POOLED.json records "
                f"{len(episode['area_fractions'])}. Reported rather than indexed around."
            )
        per_episode.append({"episode": name, "frames": episode_features(frames, empty)})
        print(f"[{done}/{total}] {name}: {len(empty)} empty frames", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rule": "T40_RULE_V15 section 6",
        "what_this_is": "masker-independent deviation-from-static-background per empty-mask frame",
        "admissibility": (
            "NOT a verdict. V15 section 6 admits this corpus-wide only at balanced accuracy >= 0.90 "
            "against held-out human labels; it never overrides a human label, and no V15 section 5 "
            "outcome may be evaluated on it."
        ),
        "produced_by": "scripts/measure_empty_mask_motion.py",
        "dev_threshold_grey_levels": DEV_THRESHOLD,
        "background": f"pixel-wise median over every {BACKGROUND_STRIDE}th frame of the same episode",
        "confounds_not_corrected": [
            "the apple is carried during the episode and is not static (~1.5% of frame)",
            "ego_view is head-mounted; head motion deviates the whole scene at once",
        ],
        "pooled_git_commit": pooled.get("git_commit"),
        "corpus": str(args.corpus),
        "n_episodes": len(per_episode),
        "n_frames": sum(len(e["frames"]) for e in per_episode),
        "per_episode": per_episode,
    }) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
