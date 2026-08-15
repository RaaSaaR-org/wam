#!/usr/bin/env python3
"""Build the apple pick-and-place variation prompt set for T-041 (exploratory, NOT PR-09).

WHAT THIS IS NOT. This is not part of the PR-09 experiment and cannot feed it. PR-09's prompt set
is fixed by ``make_t041_eval_prompts.py`` (sorted-by-uuid, first 30, val split) and its sha256 is
recorded in the verdict; this file writes a *different* set, with a different name, into a
different run directory. Nothing here is scored, gated, or compared under ``T041_RULE_V1``. It
exists to look at what the adapter does on one task, which is a demo, not a measurement.

WHY APPLE IS A FAIR CHOICE. ``g1-dex3-pickapple-dataset`` contributed 198 clips to the fine-tune
corpus (**197 train + 1 val**, corpus manifest ``sources``), so this is in-distribution for the
adapter rather than a generalisation probe. Do not read these clips as evidence about held-out
behaviour: **exactly one** of the fifteen prompts is a caption the adapter never trained on, four
are literally training captions labelled ``real-train`` for that reason, and the ten remaining are
edits of, or reseeds of, a *training* caption -- so they test steerability, not generalisation.

FOUR FAMILIES, each answering a different question:

    real-heldout  the pickapple clips in the val split -- prompts the adapter never trained on
    real-train    pickapple captions the adapter DID train on -- the "can it reproduce" floor
    variation     one authored edit each to a real caption -- does the prompt still steer it
    seed          one caption, several seeds -- how much of the output is the prompt at all

The ``variation`` family edits a real caption rather than writing one from scratch. The caption
schema the model consumes has eighteen top-level keys (subjects, cinematography, lighting,
segments, temporal_caption, ...) and a hand-written one would differ from training prompts in
style as well as content -- so any difference in the video would confound "the model followed my
edit" with "the model got an unfamiliar kind of prompt". Editing means the only difference is the
thing being varied.

    python scripts/make_apple_variation_prompts.py --corpus $DATASET_PATH -o prompts.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

SOURCE_ID = "g1-dex3-pickapple-dataset"
N_REAL_TRAIN = 4
SEEDS = [1, 2, 3, 4]

# Applied to every string value in the caption, top to bottom, longest-match-first within an entry.
# Ordering matters: "pink plate" must be rewritten before a bare "plate" rule would catch it.
VARIATIONS: list[dict] = [
    {
        "id": "green-apple",
        "note": "colour swap: does an adjective the caption never carried still land",
        "replace": [("an apple", "a green apple"), ("the apple", "the green apple"),
                    ("Apple", "Green apple")],
        "overrides": {"context": "A robotic arm performing a task of placing a green apple on a "
                                 "plate."},
    },
    {
        "id": "white-bowl",
        "note": "target swap: plate -> bowl, a different receptacle geometry",
        "replace": [("pink plate", "white ceramic bowl"), ("a plate", "a white ceramic bowl"),
                    ("the plate", "the white ceramic bowl")],
        "overrides": {"context": "A robotic arm performing a task of placing an apple into a "
                                 "white ceramic bowl."},
    },
    {
        "id": "two-apples",
        "note": "object count: two candidates, one of which must be chosen",
        "replace": [("an apple", "two apples, one red and one green"),
                    ("the apple", "the red apple")],
        "overrides": {"context": "A robotic arm selecting the red apple from two apples on the "
                                 "table and placing it on the plate."},
    },
    {
        "id": "banana-distractor",
        "note": "distractor: an object that must be left alone",
        # Only the introducing phrase is rewritten. "the apple" occurs 14 times in this caption,
        # so hanging the distractor clause off that rule instead would repeat "leaving the banana
        # untouched" fourteen times -- text no captioner would produce, which makes the prompt
        # itself the variable rather than the scene it describes.
        "replace": [("an apple", "an apple and a banana")],
        "overrides": {"context": "A robotic arm placing the apple on the plate while a banana "
                                 "remains untouched on the table.",
                      "background_setting": "An indoor setting with a white table holding an "
                                            "apple, a banana and a pink plate, a gray carpeted "
                                            "floor, and a dark-colored couch in the background."},
    },
    {
        "id": "wooden-table",
        "note": "scene swap: the background the whole corpus shares",
        "replace": [("white table", "dark wooden table"), ("a white table", "a dark wooden table")],
        "overrides": {},
    },
    {
        "id": "right-arm",
        "note": "laterality: the corpus is overwhelmingly left-arm; swap which arm acts",
        "replace": [("arm on the left", "arm on the right"), ("left arm", "right arm"),
                    ("Left arm", "Right arm"), ("on the left", "on the right")],
        "overrides": {"context": "The right robotic arm placing an apple on a plate."},
    },
]


def load_caption(corpus: Path, split: str, uuid: str) -> dict:
    p = corpus / split / "captions" / uuid / "caption.json"
    if not p.is_file():
        raise SystemExit(f"FATAL: {p} missing -- 93_caption_corpus.sbatch has not captioned {split}.")
    return json.loads(p.read_text())


def serialise(obj: dict) -> str:
    """Byte-identical serialisation to make_t041_eval_prompts.py:caption_prompt.

    Whitespace changes the token count and the recipe's max_caption_tokens is a real bound, so a
    demo prompt that serialises differently from a training prompt is not the same kind of object.
    """
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def apply_variation(caption: dict, spec: dict) -> dict:
    out = copy.deepcopy(caption)

    def walk(node):
        if isinstance(node, str):
            for old, new in spec["replace"]:
                node = node.replace(old, new)
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        return node

    out = walk(out)
    out.update(copy.deepcopy(spec["overrides"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="prepare_cosmos_corpus.py root")
    ap.add_argument("-o", "--out", required=True, type=Path)
    args = ap.parse_args(argv)

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    by_split = {s: sorted((c["uuid"] for c in manifest["clips"][s] if c["source_id"] == SOURCE_ID))
                for s in ("train", "val")}
    if not by_split["train"]:
        raise SystemExit(
            f"FATAL: no {SOURCE_ID} clips in the corpus train split. This corpus is not the one "
            "the adapter was fine-tuned on, and apple prompts would be out of distribution."
        )
    print(f"{SOURCE_ID}: {len(by_split['train'])} train, {len(by_split['val'])} val",
          file=sys.stderr)

    rows: list[dict] = []

    for uuid in by_split["val"]:
        rows.append({"uuid": f"heldout__{uuid}", "family": "real-heldout", "base_uuid": uuid,
                     "seed": 0, "note": "val split -- the adapter never trained on this caption",
                     "prompt": serialise(load_caption(args.corpus, "val", uuid))})

    train_pick = by_split["train"][:N_REAL_TRAIN]
    for uuid in train_pick:
        rows.append({"uuid": f"train__{uuid}", "family": "real-train", "base_uuid": uuid,
                     "seed": 0, "note": "TRAINING caption -- reproduction floor, not evidence",
                     "prompt": serialise(load_caption(args.corpus, "train", uuid))})

    # The template for the authored edits and the seed sweep. First train uuid, so which caption
    # got edited is a function of the corpus and not of anything anyone chose after looking.
    template_uuid = by_split["train"][0]
    template = load_caption(args.corpus, "train", template_uuid)

    for spec in VARIATIONS:
        edited = apply_variation(template, spec)
        if serialise(edited) == serialise(template):
            raise SystemExit(
                f"FATAL: variation {spec['id']!r} changed nothing -- none of its replace rules "
                f"matched {template_uuid}. A prompt identical to the template would silently "
                "become a duplicate of the seed-0 clip."
            )
        rows.append({"uuid": f"var__{spec['id']}", "family": "variation",
                     "base_uuid": template_uuid, "seed": 0, "note": spec["note"],
                     "prompt": serialise(edited)})

    for seed in SEEDS:
        rows.append({"uuid": f"seed__{seed:02d}", "family": "seed", "base_uuid": template_uuid,
                     "seed": seed, "note": f"template caption at seed {seed}",
                     "prompt": serialise(template)})

    # The payload writer derives each output directory from "name" == uuid, and the framework's own
    # duplicate-name guard cannot fire (see 95_eval_t041_embodiment.sbatch). A collision here would
    # mean one clip silently overwriting another under a name that looks right.
    names = [r["uuid"] for r in rows]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"FATAL: duplicate prompt names {dupes}")

    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (args.out.parent / (args.out.name + ".sha256")).write_text(digest + "\n")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    print(f"wrote {args.out}  ({len(rows)} prompts)  sha256={digest}", file=sys.stderr)
    print(f"families: {counts}   template: {template_uuid}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
