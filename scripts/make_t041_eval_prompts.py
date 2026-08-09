#!/usr/bin/env python3
"""Materialise T-041's 30 held-out eval prompts from the captioned val split (PR-09 §5).

Deterministic by construction: sort the val clips by uuid, take the first N. The prompt *text* is
the structured-JSON caption the captioner produced, which is also the format the model consumes at
inference — so the eval prompt and the training prompt are the same kind of object, and a
difference in results is not a difference in prompt style.

The selection rule, N, and every generation setting live in
``configs/cosmos3/t041_eval_selection.toml``, which is committed before generation. This script
only executes it. Two refusals keep that honest:

- the corpus seed in the manifest must equal the one in the config, or the holdout being drawn
  from is not the holdout the training run excluded;
- every selected uuid must be in the manifest's ``val`` list, never ``train``.

Output ``t041_eval_prompts.jsonl``, one object per line::

    {"uuid": ..., "source_id": ..., "prompt": "<serialised caption_json>", "vision_path": ...}

Its sha256 goes into the verdict. A prompt set that does not match was not the registered one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def select(manifest: dict, rule: str, n: int) -> list[dict]:
    if rule != "sorted-by-uuid-first-n":
        raise SystemExit(
            f"FATAL: unknown selection rule {rule!r}. The rule is pre-registered; adding one "
            "here after the fact is the thing the config exists to prevent."
        )
    val = sorted(manifest["clips"]["val"], key=lambda c: c["uuid"])
    if len(val) < n:
        raise SystemExit(
            f"FATAL: the val split has {len(val)} clips, the rule needs {n}. Re-run "
            "prepare_cosmos_corpus.py with a larger --val-episodes; do not shrink n."
        )
    return val[:n]


def caption_prompt(captions_dir: Path, uuid: str) -> str:
    """The structured caption, serialised exactly as the model consumes it at inference."""
    p = captions_dir / uuid / "caption.json"
    if not p.is_file():
        raise SystemExit(
            f"FATAL: {p} missing. Run 93_caption_corpus.sbatch before this — the eval prompt is "
            "the captioner's structured JSON, not the LeRobot task string."
        )
    obj = json.loads(p.read_text())
    # Same serialisation the cookbook's own payload builder uses. Whitespace differences change
    # the token count, and the recipe's max_caption_tokens is a real bound.
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path,
                    help="Corpus root from prepare_cosmos_corpus.py (holds manifest.json).")
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "cosmos3"
                    / "t041_eval_selection.toml")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sel = cfg["selection"]
    manifest = json.loads((args.corpus / "manifest.json").read_text())

    if manifest["seed"] != sel["corpus_seed"]:
        raise SystemExit(
            f"FATAL: the corpus was split with seed {manifest['seed']}, the config pre-registered "
            f"{sel['corpus_seed']}. These prompts would come from a different holdout than the "
            "one the training run excluded — which is a training-set eval wearing a holdout's name."
        )

    chosen = select(manifest, sel["rule"], sel["n"])
    train_uuids = {c["uuid"] for c in manifest["clips"]["train"]}
    leaked = [c["uuid"] for c in chosen if c["uuid"] in train_uuids]
    if leaked:
        raise SystemExit(f"FATAL: {len(leaked)} selected clips are in the TRAIN split: {leaked[:3]}")

    # ...and the same question asked of the bytes, because a uuid is a filename. The prepared
    # corpus shipped one source as a byte-copy of another, which put four val clips' pixels in
    # train under a different name; the uuid test above passed on all four. Refusing here rather
    # than dropping them is deliberate — the rule is "first n sorted by uuid" and n is
    # pre-registered, so a prompt set silently one short is a different experiment.
    unhashed = [c["uuid"] for c in manifest["clips"]["train"] + chosen if not c.get("sha256")]
    if unhashed:
        raise SystemExit(
            f"FATAL: {len(unhashed)} clips carry no sha256 in the manifest ({unhashed[:3]}), so "
            "the selection cannot be checked against train's *content*. Re-prepare the corpus "
            "with prepare_cosmos_corpus.py; skipping this is how the duplicate source got through."
        )
    train_by_sha: dict[str, str] = {}
    for c in manifest["clips"]["train"]:
        train_by_sha.setdefault(c["sha256"], c["uuid"])
    duplicated = [f"{c['uuid']} == {train_by_sha[c['sha256']]}"
                  for c in chosen if c["sha256"] in train_by_sha]
    if duplicated:
        raise SystemExit(
            f"FATAL: {len(duplicated)} selected clips are byte-identical to a TRAIN clip under a "
            f"different uuid: {duplicated[:3]}. Run scripts/dedupe_cosmos_corpus.py first."
        )

    captions_dir = args.corpus / "val" / "captions"
    lines = []
    for c in chosen:
        lines.append(json.dumps({
            "uuid": c["uuid"],
            "source_id": c["source_id"],
            "prompt": caption_prompt(captions_dir, c["uuid"]),
            "vision_path": f"val/videos/{c['uuid']}.mp4",
        }, sort_keys=True))

    body = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (args.out.parent / (args.out.name + ".sha256")).write_text(digest + "\n")
    print(f"wrote {args.out}  ({len(lines)} prompts)  sha256={digest}", file=sys.stderr)
    print(f"sources: {sorted({c['source_id'] for c in chosen})}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
