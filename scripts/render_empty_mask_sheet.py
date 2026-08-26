#!/usr/bin/env python3
"""Tiles for the empty-mask (a)/(b) split — the instrument `T40_RULE_V15` §3 specifies.

    PYTHONPATH=src:scripts .venv/bin/python scripts/render_empty_mask_sheet.py \
        --pooled runs/pr08-robot-mask-area/POOLED.json \
        --corpus /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \
        --out    runs/pr08-empty-mask-look

WHAT THIS IS FOR
----------------
``T40_RULE_V12`` §2 says the (a)/(b) split must be established **by an instrument that does not
involve the robot masker**, and ``T40_RULE_V15`` §3 fixes that instrument as a person shown the raw
source frame and asked one question:

    Is any part of the robot -- arm, hand, or gripper -- visible anywhere in this frame?

**NO MASK IS RENDERED, AND THAT OMISSION IS THE POINT.** A person shown the masker's answer before
giving their own is a correlated observer, exactly as a model reading the masker's masks is; this
project has already recorded that failure once. Nothing about the pipeline's verdict may reach the
tile: no overlay, no crop chosen by the mask, no area fraction in a caption.

THE OTHER THREE THINGS THE TILES MUST NOT LEAK, AND WHY EACH ONE MATTERS
-----------------------------------------------------------------------
**The stratum.** V15 §2's strata are positional -- start of episode, end of episode, interior -- and
a reader who knows a frame is the first of its episode has been handed the answer, because "the arm
has not entered yet" is what that stratum means. Tiles are numbered in a shuffled presentation
order and carry no stratum.

**The episode id and the frame index.** Both reconstruct the stratum. They live in ``SAMPLE.json``,
which the reader never opens.

**Neighbouring frames.** A frame's neighbours show the arm entering or leaving, which answers the
question for it. One frame per tile, and the shuffle keeps two frames of one episode apart.

WHY THE LOSSLESS COPY
---------------------
``--corpus`` defaults to the h264-lossless transcode rather than the AV1 original because the pip
``opencv-python`` wheel cannot decode AV1 and returns zero frames silently. The transcode carries
``TRANSCODE_PROOF.json`` -- every clip, frame count and max abs channel delta 0 in both yuv420p and
rgb24 -- so the pixels a reader judges are the pixels the masker was shown. The proof's presence is
checked here rather than assumed, because "lossless" in a directory name is not a measurement.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It renders evidence. It records no verdict, computes no split, evaluates none of V15 §5's outcomes,
signs no rule and discharges no blocker. Producing evidence and judging it are two acts, and only
the second is a judgement -- ``render_area_tail_sheet.py`` and ``audit_apple_masks.py`` say this in
the same words, because it is the same weakness: a script that could do both would discharge a
decision by being executed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: V15 §3. Changing either of these makes a run something other than the registered protocol, so
#: they are constants here and are written into the artifact for a reader to check.
SAMPLE_SEED = 40015
ALLOCATION = {
    "S1_lead": 60,
    "S2_trail": 60,
    "S3_int_1_2": 40,
    "S4_int_3_25": 40,
    "S5_int_26plus": 40,
}
RULE = "T40_RULE_V15"


def stratify(area_fractions: list[float]) -> list[tuple[int, str]]:
    """``(frame_index, stratum)`` for every empty-mask frame of one episode.

    The definitions are V15 §2's, verbatim: the leading and trailing contiguous zero-runs, and then
    interior runs binned by their own length. A whole-empty episode would be all leading run; the
    corpus contains none and the branch is kept so this function is total rather than lucky.
    """
    n = len(area_fractions)
    lead = 0
    while lead < n and area_fractions[lead] == 0.0:
        lead += 1
    if lead == n:
        return [(i, "S1_lead") for i in range(n)]
    trail = 0
    while area_fractions[n - 1 - trail] == 0.0:
        trail += 1

    out: list[tuple[int, str]] = [(i, "S1_lead") for i in range(lead)]
    out += [(i, "S2_trail") for i in range(n - trail, n)]

    run: list[int] = []
    for i in range(lead, n - trail):
        if area_fractions[i] == 0.0:
            run.append(i)
            continue
        if run:
            out += _bin_run(run)
            run = []
    if run:
        out += _bin_run(run)
    return out


def _bin_run(run: list[int]) -> list[tuple[int, str]]:
    length = len(run)
    name = "S3_int_1_2" if length <= 2 else ("S4_int_3_25" if length <= 25 else "S5_int_26plus")
    return [(i, name) for i in run]


def draw(pooled: dict, seed: int, allocation: dict[str, int]) -> list[dict]:
    """The stratified sample, then shuffled into presentation order. One RNG, seeded once."""
    population: dict[str, list[dict]] = {k: [] for k in allocation}
    for episode in pooled["per_episode"]:
        for frame_index, stratum in stratify(episode["area_fractions"]):
            population[stratum].append(
                {"episode": episode["episode"], "frame_index": frame_index, "stratum": stratum}
            )

    rng = random.Random(seed)
    drawn: list[dict] = []
    for stratum in sorted(allocation):
        pool = population[stratum]
        want = allocation[stratum]
        if len(pool) < want:
            raise SystemExit(
                f"{stratum} holds {len(pool)} frames but the allocation asks for {want}. "
                "The protocol's allocation is not satisfiable against this pooled artifact; "
                "that is a finding about the artifact, not a reason to shrink the ask silently."
            )
        drawn += rng.sample(pool, want)

    rng.shuffle(drawn)
    for tile_number, record in enumerate(drawn):
        record["tile"] = tile_number
    return drawn


def population_sizes(pooled: dict) -> dict[str, int]:
    sizes = {k: 0 for k in ALLOCATION}
    for episode in pooled["per_episode"]:
        for _, stratum in stratify(episode["area_fractions"]):
            sizes[stratum] += 1
    return sizes


def render(drawn: list[dict], corpus: pathlib.Path, out: pathlib.Path, quality: int) -> None:
    """Decode each needed episode ONCE, forward, and write the tiles it carries.

    Forward decode and never seek: the corpus convention everywhere in this repo, because a seek
    lands on the nearest keyframe and the frame indices here are the masker's own.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import cv2  # noqa: PLC0415
    import robot_composite as rc  # noqa: PLC0415

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    by_episode: dict[str, list[dict]] = {}
    for record in drawn:
        by_episode.setdefault(record["episode"], []).append(record)

    for done, (episode, records) in enumerate(sorted(by_episode.items()), start=1):
        wanted = {r["frame_index"]: r for r in records}
        video = corpus / "videos" / f"{episode}.mp4"
        highest = max(wanted)
        for index, rgb in enumerate(rc._decode_frames(video)):
            record = wanted.pop(index, None)
            if record is not None:
                path = frames_dir / f"tile-{record['tile']:03d}.jpg"
                cv2.imwrite(str(path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
                record["file"] = path.name
            if index >= highest:
                break
        if wanted:
            raise SystemExit(
                f"{episode}: frames {sorted(wanted)} were never decoded, but POOLED.json records an "
                f"area fraction for each. The pooled artifact and this video disagree about the "
                f"episode's length; that is a provenance failure and is not rendered around."
            )
        print(f"  [{done}/{len(by_episode)}] {episode}: {len(records)} tile(s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", type=pathlib.Path,
                        default=REPO_ROOT / "runs/pr08-robot-mask-area/POOLED.json")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path("/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "runs/pr08-empty-mask-look")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    args = parser.parse_args()

    proof = args.corpus / "TRANSCODE_PROOF.json"
    if args.corpus.name.endswith("h264-lossless") and not proof.is_file():
        raise SystemExit(
            f"{args.corpus} is named lossless and carries no {proof.name}. A directory name is not "
            "a measurement; render from the original or restore the proof."
        )

    pooled = json.loads(args.pooled.read_text())
    if not pooled.get("measurement_qualified"):
        raise SystemExit(
            f"{args.pooled} records measurement_qualified={pooled.get('measurement_qualified')!r}. "
            "The frame indices in a disqualified measurement are not a population."
        )

    sizes = population_sizes(pooled)
    drawn = draw(pooled, args.seed, ALLOCATION)
    render(drawn, args.corpus, args.out, args.jpeg_quality)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SAMPLE.json").write_text(json.dumps({
        "rule": RULE,
        "what_this_is": (
            "The key to runs/pr08-empty-mask-look/frames. THE READER MUST NOT OPEN THIS FILE while "
            "judging: it names each tile's episode, frame index and stratum, and the stratum alone "
            "answers the question the tiles ask. It exists so the verdicts can be weighted back to "
            "population shares afterwards."
        ),
        "produced_by": "scripts/render_empty_mask_sheet.py",
        "pooled": str(args.pooled.relative_to(REPO_ROOT)),
        "pooled_git_commit": pooled.get("git_commit"),
        "pooled_source_manifest_sha256": pooled.get("source_manifest_sha256"),
        "corpus": str(args.corpus),
        "corpus_note": "h264-lossless transcode, TRANSCODE_PROOF.json: max abs channel delta 0",
        "sample_seed": args.seed,
        "allocation": ALLOCATION,
        "population_sizes": sizes,
        "population_total": sum(sizes.values()),
        "question": (
            "Is any part of the robot - arm, hand, or gripper - visible anywhere in this frame?"
        ),
        "verdicts_accepted": ["yes", "no", "cannot_tell"],
        "verdict_meaning": {
            "yes": "case (b) - the robot IS present and the masker returned an empty mask",
            "no": "case (a) - the robot is genuinely absent, the composite is a correct no-op",
            "cannot_tell": "neither; V15 §4 excludes these and caps them at 25% per stratum",
        },
        "tiles": drawn,
    }, indent=1) + "\n")

    print(f"\n{len(drawn)} tiles -> {args.out / 'frames'}")
    print(f"key -> {args.out / 'SAMPLE.json'}")
    print("\nNo verdict is recorded by this script and no outcome of V15 §5 is evaluated here.")


if __name__ == "__main__":
    main()
