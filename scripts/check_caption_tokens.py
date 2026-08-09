#!/usr/bin/env python3
"""Count caption tokens exactly, and fail if any would be truncated by the SFT recipe.

WHY AN EXACT COUNT. caption_from_video.py warns at ~N tokens using a rough heuristic, and on this
corpus that heuristic peaked at 1973 against the recipe's max_caption_tokens=2048 — 96% of the cap,
with a third of the corpus still to caption. A heuristic that close to a limit is not evidence the
limit holds.

WHAT TRUNCATION COSTS. vision_sft_super.toml sets max_caption_tokens=2048. A caption over that is
cut, not rejected (sft_dataset.py:174-176): the clip still trains, on a prompt whose last sentences
are missing. It does log one warning per occurrence, but nothing counts them and nothing fails, so
in a 500-iteration run it is a line in a log nobody greps. This script makes it a gate instead.

WHAT IT MEASURES, and why not the obvious thing. The prompt is NOT any top-level jsonl field. It is
``caption_json_to_prompt(t2w_windows[i].caption_json)``, then wrapped in the Qwen3 chat template by
``tokenize_caption``. Tokenizing the serialised JSON instead over-counts by ~10%; skipping the chat
template under-counts. Both give a plausible number. This calls the framework's own two functions
so the count is the count the loader will compute.

TOKENIZER. The authoritative tokenizer is the model's own VLM tokenizer, which lives with the
weights on the cluster. Run locally it defaults to the 8B captioner's — same Qwen3 family and
vocab, close enough to decide whether there is a problem, not close enough to be the record.

    <framework>/.venv/bin/python scripts/check_caption_tokens.py ~/wam-t041/cosmos-g1-embodiment
    ... --tokenizer "${BASE_CHECKPOINT_PATH}"        # on the cluster, the real one

Exits non-zero if anything is at or over the cap, and names the clips.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

TOKENIZER_ID = "Qwen/Qwen3-VL-8B-Instruct-FP8"  # local stand-in; the cluster run uses the model's own
# vision_sft_super.toml. Mirrored here so a mismatch is a diff, not a surprise at iteration 300.
DEFAULT_MAX_CAPTION_TOKENS = 2048


def _prompts_from_jsonl(path: pathlib.Path) -> list[tuple[str, str]]:
    """(name, prompt) per training sample, built the way sft_dataset builds it.

    Do NOT reimplement this. The caption is not a top-level field: it lives in
    ``t2w_windows[*].caption_json`` as a dict, and what reaches the tokenizer is
    ``caption_json_to_prompt(that_dict)`` (sft_dataset._select_caption, line 92) — not the
    serialised JSON. Measuring ``json.dumps`` of it instead reports a number that is confidently
    wrong in the safe direction, which is the worst kind.
    """
    from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY, caption_json_to_prompt

    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        uuid = rec.get("uuid") or rec.get("vision_path") or "?"
        for i, win in enumerate(rec.get("t2w_windows") or []):
            raw = win.get(CAPTION_JSON_KEY)
            if raw is None:
                # _select_caption would fall back to the dense prose here, with its own
                # rstrip(".") + "." normalisation. Mirror that rather than skipping the sample.
                dense = win.get("caption")
                if dense is None:
                    continue
                out.append((f"{uuid}[{i}]", dense.strip().rstrip(".") + "."))
                continue
            prompt = caption_json_to_prompt(raw) if isinstance(raw, dict) else str(raw).strip()
            out.append((f"{uuid}[{i}]", prompt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_CAPTION_TOKENS)
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    args = ap.parse_args()

    # Both imported late: transformers is slow, and the framework is only importable from its own
    # venv. Run this with <framework>/.venv/bin/python.
    from transformers import AutoTokenizer

    from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # sft_dataset does this to its tokenizer before use (line 157); the extra vision/special tokens
    # shift ids and can shift length, so measuring without it is measuring a different tokenizer.
    tok, _ = add_special_tokens(tok)
    print(f"tokenizer: {args.tokenizer}   use_system_prompt=False (vision_sft_super.py:276)",
          file=sys.stderr)

    failures, grand = [], []
    for split in ("train", "val"):
        jsonl = args.corpus / split / "video_dataset_file.jsonl"
        if not jsonl.is_file():
            print(f"  {split}: no jsonl — captioning has not finished", file=sys.stderr)
            continue
        items, src = _prompts_from_jsonl(jsonl), str(jsonl)

        counts = []
        for name, text in items:
            # The real thing: chat template, generation prompt, no system prompt — exactly the call
            # sft_dataset._tokenize_caption makes before comparing against max_caption_tokens.
            n = len(tokenize_caption(text, tok, is_video=True, use_system_prompt=False))
            counts.append(n)
            grand.append(n)
            if n >= args.max_tokens:
                failures.append((split, name, n))
        if not counts:
            continue
        counts.sort()
        q = lambda p: counts[min(len(counts) - 1, int(len(counts) * p))]  # noqa: E731
        print(f"  {split}: n={len(counts)} min={counts[0]} p50={q(.5)} p95={q(.95)} "
              f"p99={q(.99)} max={counts[-1]}   [{src}]")

    if not grand:
        raise SystemExit("FATAL: no captions found.")
    print(f"\nmax_caption_tokens = {args.max_tokens}; headroom = "
          f"{args.max_tokens - max(grand)} tokens on the longest of {len(grand)} captions")

    if failures:
        print(f"\nFATAL: {len(failures)} caption(s) at or over the cap. These would be TRUNCATED "
              f"silently during SFT:")
        for split, name, n in sorted(failures, key=lambda f: -f[2])[:20]:
            print(f"  {n:>6}  {split}/{name}")
        return 1
    print("OK: every caption fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
