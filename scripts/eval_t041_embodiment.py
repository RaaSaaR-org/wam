#!/usr/bin/env python3
"""T-041 / PR-09 §6 — `T041_RULE_V1`. Does the LoRA fix the embodiment defect?

The measurement is paired and binary: for each of 30 held-out prompts, does the generated clip
render a **three-fingered Dex3 hand on a Unitree G1 arm**, or a generic manipulator? Both arms
generate from the same prompts with the same seed and sampler, so the pair differs in the adapter
and nothing else.

Three subcommands, deliberately separate files between them:

    --build-sheet   shuffle the 60 clips together with 20 real calibration items, write a BLINDED
                    scoring_sheet.jsonl and a separate key.json
    --judge         run the VLM over the sheet, write scores.jsonl
    --verdict       join sheet + key + scores, apply G0a/G0b/G0c, compute the exact McNemar test,
                    write verdict.json

Why three and not one: the scorer must not see arm labels, and "the code did not look" is a weaker
guarantee than "the labels were in a different file the scoring step never opened". It also means
a **human** can produce scores.jsonl by watching 80 clips, and the identical rule applies — which
is the fallback PR-09 §6 G0b requires if the VLM cannot pass calibration.

No threshold is coined anywhere in here. The statistic is the exact McNemar test (a one-sided
binomial on the discordant pairs) at alpha = 0.05, borrowed rather than chosen. G0a's 15/30 is the
"is there a defect at all" floor and is a majority, not a tuned number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tomllib
from pathlib import Path

ALPHA = 0.05
G0A_MIN_BASE_FAILURES = 15   # a majority of 30: below this there is no defect to fix
G0B_REQUIRED = 20            # all 20 calibration items, both directions
N_MCNEMAR_ITERS = 500        # the run must reach this iteration; PR-09 §6 G0c

RUBRIC = (
    "You are shown a short video of a robot manipulating an object.\n"
    "Answer ONE question about the robot's arm and hand, and nothing else.\n"
    "\n"
    "Does the robot have a THREE-FINGERED humanoid hand (a Unitree Dex3-style hand with three "
    "articulated fingers including an opposable thumb) attached to a humanoid arm?\n"
    "\n"
    "Answer NO if the end effector is a two-jaw parallel gripper, a suction cup, a five-fingered "
    "hand, an industrial claw, or if no end effector is visible.\n"
    "Answer NO if you are unsure.\n"
    "\n"
    'Reply with exactly one word: YES or NO.'
)


# --------------------------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------------------------

def mcnemar_exact_one_sided(b: int, c: int) -> float:
    """One-sided exact McNemar p-value for "the second arm fixes what the first got wrong".

    ``b`` = base wrong, lora right (the direction we predicted).
    ``c`` = base right, lora wrong.
    Under H0 each discordant pair is a fair coin, so b ~ Binomial(b + c, 1/2) and
    p = P(X >= b). Concordant pairs carry no information and do not enter — that is the whole
    point of pairing, and dividing by 60 instead would be a different, weaker test.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def verdict_from(b: int, c: int, p: float) -> tuple[str, str]:
    """PR-09 §6's table, applied. Fixed before the numbers existed."""
    if p < ALPHA:
        return "P", ("significant at alpha=0.05 in the pre-registered direction — ~100 GPU-h of "
                     "LoRA on G1 footage fixes the embodiment defect")
    if b <= 2:
        return "N", ("not significant and the adapter fixed <=2 of the base's failures — it does "
                     "not. The defect is T-040's to patch by compositing")
    return "I", ("not significant but the adapter fixed >=3 — underpowered, not refuted. Record "
                 "and stop; there is no second run under T041_RULE_V1")


# --------------------------------------------------------------------------------------------
# build-sheet
# --------------------------------------------------------------------------------------------

def build_sheet(args) -> int:
    cfg = tomllib.loads(args.config.read_text())
    cal = cfg["calibration"]
    prompts = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]

    items: list[dict] = []
    for p in prompts:
        for arm in ("base", "lora"):
            clip = args.clips / arm / f"{p['uuid']}.mp4"
            if not clip.is_file():
                raise SystemExit(f"FATAL: missing generated clip {clip} — generation is incomplete.")
            items.append({"kind": "paired", "arm": arm, "uuid": p["uuid"], "path": str(clip)})

    # Calibration: real footage on both sides, so the correct answer is known without anyone
    # adjudicating a generated frame.
    for kind, sub, want, n in (("cal_pos", "positive", True, cal["n_positive"]),
                               ("cal_neg", "negative", False, cal["n_negative"])):
        found = sorted((args.calibration / sub).glob("*.mp4"))
        if len(found) < n:
            raise SystemExit(
                f"FATAL: calibration/{sub} has {len(found)} clips, G0b needs {n}. Without the "
                "calibration set the rubric is unvalidated and no verdict may be issued."
            )
        for clip in found[:n]:
            items.append({"kind": kind, "arm": None, "uuid": clip.stem,
                          "path": str(clip), "expected": want})

    # Stable ids that carry no signal, then shuffle. The seed is recorded so the shuffle is
    # reproducible; it is not secret, because secrecy is not what blinding needs — separation is.
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item_id"] = f"item_{i:04d}"

    # THE PATH IS PART OF THE LABEL. clips/base/foo.mp4 names the arm in the filename, so a sheet
    # that pointed at the original files would be blinded in the schema and unblinded in the
    # string — and the human-scoring fallback reads exactly that string. Materialise a neutral
    # tree instead, so the only name the scorer ever sees is the item id.
    args.out.mkdir(parents=True, exist_ok=True)
    blinded = args.out / "items"
    blinded.mkdir(exist_ok=True)
    for it in items:
        link = blinded / f"{it['item_id']}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.realpath(it["path"]), link)
        it["blinded_path"] = str(link)

    sheet = args.out / "scoring_sheet.jsonl"
    key = args.out / "key.json"
    # id + neutral path. Nothing else — no arm, no uuid, no expected answer, no source filename.
    sheet.write_text("".join(
        json.dumps({"item_id": it["item_id"], "path": it["blinded_path"]}, sort_keys=True) + "\n"
        for it in items))
    key.write_text(json.dumps({"shuffle_seed": args.shuffle_seed, "items": items},
                              indent=2, sort_keys=True) + "\n")
    print(f"wrote {sheet} ({len(items)} items) and {key}", file=sys.stderr)
    print("the sheet carries neither arm labels nor source filenames; --verdict is the only step "
          "that opens the key", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------------------------

def parse_answer(text: str) -> bool | None:
    """YES/NO out of a model reply. Anything else is None and counts as unscored, not as NO.

    Word boundaries, not substrings: "I can**no**t tell" contains "NO" and is an abstention, and
    a substring match would silently record it as the defect being present. Both tokens present
    ("YES or NO") is also an abstention — a reply that restates the question answered nothing.
    Defaulting an unparseable reply to NO would bias every arm toward "generic manipulator",
    which is the very thing being measured.
    """
    t = text.strip().upper()
    has_yes = re.search(r"\bYES\b", t) is not None
    has_no = re.search(r"\bNO\b", t) is not None
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def judge(args) -> int:
    import urllib.request

    cfg = tomllib.loads(args.config.read_text())
    model = cfg["judge"]["model"]
    sheet = [json.loads(l) for l in (args.out / "scoring_sheet.jsonl").read_text().splitlines()
             if l.strip()]

    out = args.out / "scores.jsonl"
    done = set()
    if out.exists() and args.resume:
        done = {json.loads(l)["item_id"] for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming: {len(done)} already scored", file=sys.stderr)

    with out.open("a" if args.resume else "w") as fh:
        for item in sheet:
            if item["item_id"] in done:
                continue
            payload = {
                "model": model,
                "temperature": 0.0,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "file://" + item["path"]}},
                    {"type": "text", "text": RUBRIC},
                ]}],
            }
            req = urllib.request.Request(
                args.server.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"]
            fh.write(json.dumps({"item_id": item["item_id"], "raw": text,
                                 "answer": parse_answer(text)}, sort_keys=True) + "\n")
            fh.flush()
    print(f"wrote {out}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------------------------

def compute_verdict(key: dict, scores: dict[str, bool | None], run_meta: dict,
                    prompts_sha256: str) -> dict:
    items = key["items"]
    by_id = {it["item_id"]: it for it in items}

    unscored = [i for i in by_id if scores.get(i) is None]

    # --- G0b: the rubric must be able to see the thing, on real footage, both directions.
    cal = [it for it in items if it["kind"] in ("cal_pos", "cal_neg")]
    cal_correct = sum(1 for it in cal if scores.get(it["item_id"]) is it["expected"])
    g0b = cal_correct == G0B_REQUIRED == len(cal)

    # --- pair up
    paired: dict[str, dict[str, bool | None]] = {}
    for it in items:
        if it["kind"] == "paired":
            paired.setdefault(it["uuid"], {})[it["arm"]] = scores.get(it["item_id"])
    complete = {u: v for u, v in paired.items()
                if v.get("base") is not None and v.get("lora") is not None}

    base_failures = sum(1 for v in complete.values() if v["base"] is False)
    g0a = base_failures >= G0A_MIN_BASE_FAILURES

    # --- G0c: the run has to have been a run.
    g0c = (run_meta.get("iteration_reached") == N_MCNEMAR_ITERS
           and bool(run_meta.get("resume_diffs_logged", True))
           and bool(run_meta.get("export_nonempty")))

    b = sum(1 for v in complete.values() if v["base"] is False and v["lora"] is True)
    c = sum(1 for v in complete.values() if v["base"] is True and v["lora"] is False)
    p = mcnemar_exact_one_sided(b, c)

    result = {
        "rule": "T041_RULE_V1",
        "prereg": "docs/preregistration/PR-09-cosmos-super-finetune.md",
        "prompts_sha256": prompts_sha256,
        "n_pairs_complete": len(complete),
        "n_pairs_total": len(paired),
        "unscored_items": len(unscored),
        "calibration_correct": cal_correct,
        "calibration_total": len(cal),
        "base_failures": base_failures,
        "discordant_base_wrong_lora_right": b,
        "discordant_base_right_lora_wrong": c,
        "mcnemar_p_one_sided": p,
        "alpha": ALPHA,
        "gates": {"G0a_defect_present": g0a, "G0b_rubric_calibrated": g0b, "G0c_run_complete": g0c},
        "run_metadata": run_meta,
    }

    failed = [k for k, v in result["gates"].items() if not v]
    if failed:
        result["verdict"] = "VOID"
        result["reading"] = (
            f"VOID on {', '.join(failed)} — a defect report against the rig, not a statement "
            "about Cosmos. PR-09 §6 forbids reading a VOID as a weaker pass."
        )
        return result
    if len(complete) != len(paired) or len(paired) == 0:
        result["verdict"] = "VOID"
        result["reading"] = (
            f"VOID — {len(paired) - len(complete)} of {len(paired)} pairs are incomplete. A "
            "partial pairing is a different experiment from the registered one."
        )
        return result

    v, reading = verdict_from(b, c, p)
    result["verdict"] = v
    result["reading"] = reading
    return result


def check_prompts_are_held_out(prompts_path: Path, corpus: Path) -> int:
    """Re-derive disjointness here, against the corpus manifest, not against a sidecar.

    ``make_t041_eval_prompts.py`` already refuses to emit a prompt drawn from ``train`` — but it
    also writes the ``.sha256`` sidecar, so hash-matching only proves the file has not changed
    *since*, not that it ever came from that script. A hand-written pair passes that check. This
    one cannot be satisfied by anything except prompts that really are in the manifest's val list.
    """
    manifest = json.loads((corpus / "manifest.json").read_text())
    val = {c["uuid"] for c in manifest["clips"]["val"]}
    train = {c["uuid"] for c in manifest["clips"]["train"]}
    uuids = [json.loads(l)["uuid"]
             for l in prompts_path.read_text().splitlines() if l.strip()]
    leaked = sorted(u for u in uuids if u in train)
    if leaked:
        raise SystemExit(
            f"FATAL: {len(leaked)} eval prompts are in the TRAINING split ({leaked[:3]}). "
            "Every number this eval could produce would be a training score."
        )
    unknown = sorted(u for u in uuids if u not in val)
    if unknown:
        raise SystemExit(
            f"FATAL: {len(unknown)} eval prompts are in neither split of "
            f"{corpus}/manifest.json ({unknown[:3]}). The prompt set does not belong to the "
            "corpus this run trained on."
        )
    return len(uuids)


def verdict(args) -> int:
    key = json.loads((args.out / "key.json").read_text())
    scores = {}
    for line in (args.out / "scores.jsonl").read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            scores[o["item_id"]] = o.get("answer")
    run_meta = json.loads(args.run_metadata.read_text()) if args.run_metadata else {}
    sha = (args.prompts.parent / (args.prompts.name + ".sha256"))
    prompts_sha = sha.read_text().strip() if sha.is_file() else ""
    # Recompute rather than trust the sidecar: the sidecar is written by the same hand.
    actual = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if prompts_sha and actual != prompts_sha:
        raise SystemExit(
            f"FATAL: {args.prompts} hashes {actual} but its sidecar says {prompts_sha}. The "
            "prompt set changed after it was registered."
        )
    n_checked = check_prompts_are_held_out(args.prompts, args.corpus)
    print(f"disjointness re-derived against {args.corpus}/manifest.json: "
          f"{n_checked} prompts, all in val", file=sys.stderr)

    result = compute_verdict(key, scores, run_meta, actual)
    result["disjointness_checked_against"] = str(args.corpus / "manifest.json")
    out = args.out / "verdict.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n=== T-041 VERDICT: {result['verdict']} ===", file=sys.stderr)
    print(result["reading"], file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "cosmos3"
                    / "t041_eval_selection.toml")
    ap.add_argument("--out", type=Path, required=True, help="Working directory for this eval.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-sheet")
    b.add_argument("--prompts", type=Path, required=True)
    b.add_argument("--clips", type=Path, required=True, help="Holds base/ and lora/ subdirs.")
    b.add_argument("--calibration", type=Path, required=True,
                   help="Holds positive/ and negative/ subdirs of REAL clips.")
    b.add_argument("--shuffle-seed", type=int, default=0)
    b.set_defaults(func=build_sheet)

    j = sub.add_parser("judge")
    j.add_argument("--server", default="http://localhost:8000/v1")
    j.add_argument("--timeout", type=float, default=300.0)
    j.add_argument("--resume", action="store_true")
    j.set_defaults(func=judge)

    v = sub.add_parser("verdict")
    v.add_argument("--prompts", type=Path, required=True)
    v.add_argument("--corpus", type=Path, required=True,
                   help="Corpus root, so disjointness is re-derived here and not taken on trust.")
    v.add_argument("--run-metadata", type=Path, default=None)
    v.set_defaults(func=verdict)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
