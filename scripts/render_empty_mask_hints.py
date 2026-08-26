#!/usr/bin/env python3
"""The "what is different here" toggle of `T40_RULE_V16` §4 — an aid to FINDING, not to deciding.

    PYTHONPATH=src:scripts .venv/bin/python scripts/render_empty_mask_hints.py

WHAT IT DRAWS
-------------
A thin outline around each connected region where the frame differs from its own episode's
pixel-wise median background. Nothing is classified: the outline says "this changed", not "this is
the robot", and V16 §4 admits it on exactly those terms. A reader hunting a dark sliver against
dark cloth should not have to find it by eye first and judge it second.

THE APPLE IS EXCLUDED, AND THAT IS A MEASUREMENT
------------------------------------------------
The apple is carried during every episode, so it appears in the difference twice -- where it sat in
the median and where it sits now -- and it dominates. Over ten episodes and 1 403 empty-mask frames
the median difference area falls from 0.0205 to 0.0076 when the apple's swept region is removed, so
this is not a guess about what the outlines would otherwise be full of. The region is found with
``T40_RULE_V9``'s own warm-saturated colour test, reused rather than re-invented, unioned over the
episode and dilated.

**This changes the rendering and nothing else.** No threshold in V16 §6 is touched, no verdict is
recorded, and the outlines are drawn identically whatever the frame turns out to contain.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_THRESHOLD = 25      # grey levels, the same cut MOTION.json uses
MIN_REGION_PX = 120     # below this an outline is noise the reader would have to dismiss
OUTLINE_RGB = (63, 194, 214)


def apple_sweep(frames: np.ndarray, stride: int = 5, dilate: int = 9) -> np.ndarray:
    """Every pixel the apple ever occupied, by `T40_RULE_V9`'s warm-saturated test."""
    import cv2

    ever = np.zeros(frames.shape[1:3], dtype=bool)
    for index in range(0, len(frames), stride):
        rgb = frames[index]
        r = rgb[..., 0].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        mx = rgb.max(axis=2).astype(np.float32)
        mn = rgb.min(axis=2).astype(np.float32)
        saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
        ever |= (r > 90) & ((r - b) > 50) & (saturation > 0.35)
    return cv2.dilate(ever.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--look", type=pathlib.Path, default=REPO_ROOT / "runs/pr08-empty-mask-look")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless"))
    parser.add_argument("--jpeg-quality", type=int, default=86)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import cv2  # noqa: PLC0415
    import robot_composite as rc  # noqa: PLC0415

    sample = json.loads((args.look / "SAMPLE.json").read_text())
    hints = args.look / "hints"
    hints.mkdir(parents=True, exist_ok=True)

    by_episode: dict[str, list[dict]] = {}
    for record in sample["tiles"]:
        by_episode.setdefault(record["episode"], []).append(record)

    for done, (episode, records) in enumerate(sorted(by_episode.items()), start=1):
        frames = rc.decode_clip(args.corpus / "videos" / f"{episode}.mp4")
        grey = frames.astype(np.float32).mean(axis=3)
        background = np.median(grey[::5], axis=0)
        sweep = apple_sweep(frames)

        for record in records:
            index = record["frame_index"]
            changed = (np.abs(grey[index] - background) > DEV_THRESHOLD) & ~sweep
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((3, 3), np.uint8))
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
            keep = np.zeros_like(changed)
            for label in range(1, n_labels):
                if stats[label, cv2.CC_STAT_AREA] >= MIN_REGION_PX:
                    keep[labels == label] = 1

            canvas = np.clip(frames[index].astype(np.float32) * 1.9, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, OUTLINE_RGB, 2)
            cv2.imwrite(str(hints / f"hint-{record['tile']:03d}.jpg"),
                        canvas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        print(f"  [{done}/{len(by_episode)}] {episode}", flush=True)

    print(f"\n{len(sample['tiles'])} hints -> {hints}")


if __name__ == "__main__":
    main()
