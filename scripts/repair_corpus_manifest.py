#!/usr/bin/env python3
"""Repair the width/height provenance in a prepared Cosmos corpus manifest, in place.

WHY THIS EXISTS. ``prepare_cosmos_corpus.video_shape`` used to read a LeRobot feature ``shape`` as
[H, W, C] and return ``(shape[1], shape[0])``. Half our sources declare [C, H, W] = [3, 480, 640],
so those entries recorded ``width=480, height=3`` — a pair that looks like a resolution, sorts like
a resolution, and is wrong. 1712 of 3462 entries in the shipped manifest carry it.

THE VIDEOS ARE FINE. ``verify_clip_decode.py`` opened all 3462 with the captioner's own cv2 and
measured 640x480 @ 30.0 fps for every one of them, 0 failures. Only the provenance record lies —
which still matters, because AC-04 makes the manifest the thing a rollout is traced back to, and
because a future reader has no way to tell a wrong ``height=3`` from a real one.

WHAT THIS DOES NOT DO. It does not re-derive anything from the LeRobot metadata. The decode report
is a direct measurement of the bytes that will train, taken by the same library that will read
them; re-running the buggy path's replacement against the source metadata would be a second
opinion where a first-hand measurement already exists. If a clip has no decode record, or its
record failed, this refuses rather than guessing.

    python3 scripts/repair_corpus_manifest.py ~/wam-t041/cosmos-g1-embodiment --check
    python3 scripts/repair_corpus_manifest.py ~/wam-t041/cosmos-g1-embodiment

Idempotent: a second run reports 0 changes and leaves the digest alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def _measured(root: pathlib.Path, split: str) -> dict[str, dict]:
    """uuid -> decode record, for one split. Keyed by filename stem, which IS the uuid."""
    report = root / split / "decode_report.json"
    if not report.is_file():
        raise SystemExit(f"FATAL: {report} missing — run scripts/verify_clip_decode.py first.")
    data = json.loads(report.read_text())
    if data.get("failed"):
        raise SystemExit(
            f"FATAL: {report} records {data['failed']} decode failures. Fix the corpus, not the "
            "manifest — a manifest that describes clips which do not decode is worse than a wrong "
            "width."
        )
    return {pathlib.Path(r["path"]).stem: r for r in data["results"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=pathlib.Path, help="the prepared corpus directory")
    ap.add_argument("--check", action="store_true",
                    help="report what would change and exit non-zero if anything would; write nothing")
    args = ap.parse_args()

    root: pathlib.Path = args.corpus
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"FATAL: {manifest_path} missing.")
    manifest = json.loads(manifest_path.read_text())

    changed, checked = 0, 0
    for split, clips in manifest["clips"].items():
        measured = _measured(root, split)
        for clip in clips:
            uuid = clip["uuid"]
            rec = measured.get(uuid)
            if rec is None:
                raise SystemExit(
                    f"FATAL: {split}/{uuid} is in the manifest but has no decode record. The "
                    "manifest and the corpus on disk disagree about what exists; that is not a "
                    "provenance bug and this script will not paper over it."
                )
            checked += 1
            w, h, fps = int(rec["width"]), int(rec["height"]), float(rec["fps"])
            # fps is verified, never patched. width/height had a known bad code path feeding them;
            # fps did not, so a disagreement here means something this script does not understand.
            if abs(float(clip["fps"]) - fps) > 1e-6:
                raise SystemExit(
                    f"FATAL: {split}/{uuid} manifest fps={clip['fps']} but the decode measured "
                    f"{fps}. Only width/height are known-bad; refusing to touch this corpus."
                )
            if (clip["width"], clip["height"]) != (w, h):
                print(f"  {split}/{uuid}: "
                      f"{clip['width']}x{clip['height']} -> {w}x{h}")
                clip["width"], clip["height"] = w, h
                changed += 1

    print(f"checked {checked} clips, {changed} corrected", file=sys.stderr)
    if args.check:
        return 1 if changed else 0
    if not changed:
        print("nothing to do; manifest already agrees with the measured decodes", file=sys.stderr)

    # Rewrite unconditionally even at 0 changes: the digest convention itself was wrong (the writer
    # hashed the body while writing body + "\n"), so a corpus prepared before that fix has a
    # MANIFEST_SHA256 that `sha256sum manifest.json` cannot reproduce. Re-stamping costs nothing
    # and makes the obvious command the authoritative one.
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (root / "MANIFEST_SHA256").write_text(digest + "\n")
    print(f"wrote {manifest_path}\nMANIFEST_SHA256={digest}", file=sys.stderr)
    print("verify with:  sha256sum manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
