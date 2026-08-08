#!/usr/bin/env python3
"""Decode every clip with the *consumer's* decoder and fail loudly if any of them yields no frames.

WHY THIS EXISTS. Job 186357 captioned 372 clips and produced zero captions. Nothing crashed: the
corpus was valid, ffprobe was happy, the manifest was correct, the captioner ran to completion at
3 it/s. vLLM's OpenCV backend opened each mp4, read the container header correctly — 377 frames,
30 fps, 640x480 — and then failed every single ``cap.grab()``. The model was handed
``array([], shape=(0, 480, 640, 3))`` and the run burned a GPU hour to write 372 empty files.

The lesson is narrow and worth encoding: **a corpus is only readable by the decoder that will
actually read it.** ffprobe proves a file is well-formed. It does not prove that the specific
OpenCV build inside the captioner's virtualenv can pull pixels out of it. Those are different
questions and today they had different answers.

So this script deliberately does NOT use ffmpeg. It uses ``cv2``, and it is meant to be run with
the interpreter of the environment that will consume the clips::

    /path/to/captioner/.venv/bin/python scripts/verify_clip_decode.py <corpus>/train/videos

Run it against a different Python and you have verified a different decoder, which is worth
approximately nothing. The script prints which cv2 and which backend it used so that mistake is
visible in the log rather than silent.

Exit status is 0 only when every clip decoded. That makes it usable as a gate between preparation
and captioning, which is the only place it can still save anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def probe(path: Path, frames_to_read: int) -> dict:
    """Open one clip and actually pull pixels out of it.

    Returns a record rather than raising: one unreadable clip in a corpus of thousands is a fact to
    report with its neighbours, not a reason to abandon the scan.
    """
    import cv2
    import numpy as np

    rec: dict = {"path": str(path), "ok": False, "decoded": 0, "declared": 0, "error": ""}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            rec["error"] = "VideoCapture did not open the file"
            return rec
        rec["declared"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rec["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rec["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rec["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)

        # Sequential reads from the start. This mirrors what vLLM's OpenCVVideoBackend does
        # (grab() in a loop, retrieve() on the wanted indices) rather than a random-access seek,
        # because seeking can succeed on files whose sequential decode fails and vice versa.
        decoded = 0
        for _ in range(frames_to_read):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # A decoder can hand back a correctly shaped buffer it never wrote into. Touching the
            # data is what distinguishes "returned a frame" from "returned a frame with pixels".
            if frame.size == 0 or not np.isfinite(frame.astype("float32").mean()):
                break
            decoded += 1
        rec["decoded"] = decoded
        rec["ok"] = decoded > 0
        if decoded == 0:
            rec["error"] = (
                f"opened and declared {rec['declared']} frames but decoded none — "
                "container is readable, the codec is not"
            )
        return rec
    except Exception as exc:  # noqa: BLE001 - the point is to survive and report
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, nargs="+",
                    help="Directories of .mp4 clips, or individual .mp4 files.")
    ap.add_argument("--frames", type=int, default=2,
                    help="Frames to actually decode per clip. 2 is enough to separate "
                         "'container parses' from 'codec decodes'; raise it to catch corruption "
                         "that only appears later in a file.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Check at most N clips (0 = all). A sample proves a codec works; only a "
                         "full pass proves a corpus does.")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--report", type=Path, help="Write the full per-clip result as JSON here.")
    args = ap.parse_args(argv)

    try:
        import cv2
    except ImportError:
        print("FATAL: cv2 is not importable by this interpreter.\n"
              f"       {sys.executable}\n"
              "       Run this with the python of the environment that will read the clips — "
              "verifying with any other decoder proves nothing.", file=sys.stderr)
        return 2

    clips: list[Path] = []
    for target in args.videos:
        if target.is_dir():
            clips.extend(sorted(target.rglob("*.mp4")))
        elif target.is_file():
            clips.append(target)
        else:
            print(f"FATAL: {target} is neither a directory nor a file", file=sys.stderr)
            return 2
    if not clips:
        print("FATAL: no .mp4 found — nothing was verified, which is not the same as passing.",
              file=sys.stderr)
        return 2
    if args.limit:
        clips = clips[: args.limit]

    print(f"cv2 {cv2.__version__} from {Path(cv2.__file__).parent}", file=sys.stderr)
    print(f"interpreter {sys.executable}", file=sys.stderr)
    print(f"checking {len(clips)} clips, {args.frames} frame(s) each", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda p: probe(p, args.frames), clips))

    bad = [r for r in results if not r["ok"]]
    if args.report:
        args.report.write_text(json.dumps(
            {"cv2": cv2.__version__, "interpreter": sys.executable,
             "checked": len(results), "failed": len(bad), "results": results},
            indent=2) + "\n")

    if not bad:
        sample = results[0]
        print(f"OK: {len(results)}/{len(results)} clips decoded "
              f"({sample.get('width')}x{sample.get('height')} @ {sample.get('fps')} fps)",
              file=sys.stderr)
        return 0

    print(f"\nFAIL: {len(bad)}/{len(results)} clips decoded no frames", file=sys.stderr)
    for r in bad[:10]:
        print(f"  {Path(r['path']).name}: {r['error']}", file=sys.stderr)
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more", file=sys.stderr)
    print("\nThis is the failure that produced '0/N videos were successfully captioned'.\n"
          "Do not caption this corpus. Re-run preparation with --transcode so the clips are\n"
          "written in a codec this decoder can read.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
