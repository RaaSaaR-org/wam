#!/usr/bin/env python3
"""Evaluate `T40_RULE_V15` §5 — once, on a complete sample, exactly as registered.

    PYTHONPATH=src:scripts .venv/bin/python scripts/score_empty_mask_split.py \
        --look runs/pr08-empty-mask-look \
        --verdicts runs/pr08-empty-mask-look/VERDICTS.json \
        --out runs/pr08-empty-mask-look/SPLIT.json

WHY THIS SCRIPT MAKES NO CHOICES
--------------------------------
Every threshold, weight and exclusion below is quoted from a document committed **before the tiles
were rendered** (`docs/preregistration/PR-08-V15-g0c-empty-mask-ab-split-protocol.md`, commit
52b8714). This script is a calculator for that document and deliberately has no options that could
change an outcome: no ``--threshold``, no ``--exclude``, no ``--stratum``. If a number here is
wrong, the fix is a new rule version, not a flag.

WHAT IT REFUSES
---------------
**An incomplete sample.** §5's quantity is defined over the registered 240 and a partial sample
evaluated early is the peek the whole protocol exists to prevent.

**A stratum over §4's undecidability cap.** More than 25 % ``cannot_tell`` in any stratum and §5 is
NOT evaluated at all: the finding is then that this rendering does not answer the question, and the
repair is a version that changes the rendering rather than one that reinterprets the tiles.

**A verdict outside the registered vocabulary**, and a tile number that is not in the sample.

DISCLOSURE, BECAUSE IT BELONGS NEXT TO THE CODE AND NOT ONLY IN A RESULT DOCUMENT
--------------------------------------------------------------------------------
The session that wrote this file had already seen a PARTIAL tally — the first 101 of the 240
verdicts, as counts by verdict, arriving through an intermediate save. It had not seen any tile's
stratum-resolved outcome, and every threshold this script applies was fixed in V15 §5 before any
verdict existed. The disclosure is made because "the analysis was written blind" would otherwise be
an overstatement, and a reader is entitled to discount this code accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

RULE = "T40_RULE_V16"
ACCEPTED = ("arm", "edge", "none", "unclear")
POSITIVE = "arm"          # §6's p_A numerator: a DEFINITE arm, not an edge fragment
UNDECIDED = "unclear"
UNDECIDABLE_CAP = 0.25          # §4
Q1_A_UPPER = 0.05               # §5 Q1 outcome A
Q1_B_LOWER = 0.33               # §5 Q1 outcome B
Q2_A_SURVIVE = 302              # §5 Q2 outcome A2 (three quarters of 402)
Q2_B_SURVIVE = 101              # §5 Q2 outcome B2 (one quarter of 402)
Q1_A_Q99 = 0.01                 # §6 outcome A's second condition, on the corpus-wide bound
AUX_BALANCED_ACCURACY = 0.90    # V15 §6, carried forward
AUX_SPLIT_SEED = 40016


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Point estimate and Wilson 95 % bounds. Wilson because a stratum may come back 0/60."""
    if trials == 0:
        return float("nan"), 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def evaluate(sample: dict, verdicts: dict[str, str], q99: float | None = None) -> dict:
    tiles = {record["tile"]: record for record in sample["tiles"]}

    unknown = sorted(set(verdicts) - {str(t) for t in tiles})
    if unknown:
        raise SystemExit(f"verdicts name tiles that are not in the sample: {unknown[:8]}")
    bad = sorted({v for v in verdicts.values()} - set(ACCEPTED))
    if bad:
        raise SystemExit(f"verdicts outside the registered vocabulary {ACCEPTED}: {bad}")
    missing = sorted(t for t in tiles if str(t) not in verdicts)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(tiles)} tiles carry no verdict (first: {missing[:8]}). "
            "V15 §5 is defined over the complete registered sample; a partial sample evaluated "
            "early is the peek this protocol exists to prevent."
        )

    per_stratum: dict[str, dict] = {}
    for stratum in sample["allocation"]:
        rows = [t for t in tiles.values() if t["stratum"] == stratum]
        counts = {v: 0 for v in ACCEPTED}
        for row in rows:
            counts[verdicts[str(row["tile"])]] += 1
        # §6: `unclear` leaves numerator AND denominator. `edge` is denominator, not numerator —
        # it is a positive finding, not a hedge, and reading it as a partial `arm` would be the
        # reinterpretation V15 §4 forbids.
        judged = sum(counts[v] for v in ACCEPTED if v != UNDECIDED)
        undecidable = counts[UNDECIDED] / len(rows) if rows else 0.0
        point, low, high = wilson(counts[POSITIVE], judged)
        per_stratum[stratum] = {
            "n_tiles": len(rows), "counts": counts, "n_judged": judged,
            "undecidable_fraction": undecidable,
            "over_cap": undecidable > UNDECIDABLE_CAP,
            "p_A": point, "p_A_low": low, "p_A_high": high,
            "population": sample["population_sizes"][stratum],
            "weight": sample["population_sizes"][stratum] / sample["population_total"],
        }

    over = sorted(s for s, r in per_stratum.items() if r["over_cap"])
    if over:
        return {
            "rule": RULE, "evaluated": False,
            "refusal": "V15 §4, carried forward by V16 §4: a stratum exceeded the 25% undecidable cap",
            "strata_over_cap": over,
            "what_this_means": (
                "This rendering does not answer the question V15 §3 asks, for those strata. The "
                "finding is about the instrument. V15 §4 forbids reinterpreting the tiles and "
                "requires a version that changes the rendering."
            ),
            "per_stratum": per_stratum,
        }

    p_A = sum(r["weight"] * r["p_A"] for r in per_stratum.values())
    # Half-widths combined in quadrature: the strata are drawn independently, so their errors are
    # independent, and this is the weighted sum's normal-approximation interval built on Wilson
    # half-widths rather than on p(1-p)/n, which collapses to zero at a 0/60 stratum.
    lo_half = math.sqrt(sum((r["weight"] * (r["p_A"] - r["p_A_low"])) ** 2 for r in per_stratum.values()))
    hi_half = math.sqrt(sum((r["weight"] * (r["p_A_high"] - r["p_A"])) ** 2 for r in per_stratum.values()))
    p_A_low, p_A_high = max(0.0, p_A - lo_half), min(1.0, p_A + hi_half)

    # §6 outcome A is a CONJUNCTION and the second half is read from the corpus, not the sample.
    # It is known to have failed (q99 = 0.0718 against 0.01, PR-08-RESULT-2026-08-27), so A is
    # unreachable. That is not repaired here: the conjunction was registered before the
    # distribution was seen, and loosening it now is the move handoff.md §3 forbids.
    if q99 is not None and p_A_high <= Q1_A_UPPER and q99 <= Q1_A_Q99:
        q1, q1_text = "A", "definite arms are rare AND the worst case for the rest is under 1% of a frame; a BOUNDED rule is licensed to be drafted, not adopted"
    elif p_A_low >= Q1_B_LOWER:
        q1, q1_text = "B", "the masker is failing on frames with a plain arm; V12 §3.3 — leave G0c alone and revisit T40_RULE_V1 §3's route"
    else:
        q1, q1_text = "M", "neither; a further rule version, which must say what it does about whichever condition failed"

    return {
        "rule": RULE, "evaluated": True,
        "p_A": p_A, "p_A_low": p_A_low, "p_A_high": p_A_high,
        "q99_frac_dev": q99,
        "q99_condition_met": (q99 is not None and q99 <= Q1_A_Q99),
        "interval": "Wilson 95% per stratum, half-widths combined in quadrature",
        "q1_outcome": q1, "q1_means": q1_text,
        "q1_thresholds": {"p_A_upper": Q1_A_UPPER, "p_A_lower": Q1_B_LOWER, "q99_max": Q1_A_Q99},
        "per_stratum": per_stratum,
    }


def survival(sample: dict, per_stratum: dict, pooled: dict) -> dict:
    """§5 Q2's `n_survive`, model-dependent and labelled as such."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from render_empty_mask_sheet import stratify  # noqa: PLC0415

    expected = 0.0
    certain_zero = 0
    for episode in pooled["per_episode"]:
        counts: dict[str, int] = {}
        for _, stratum in stratify(episode["area_fractions"]):
            counts[stratum] = counts.get(stratum, 0) + 1
        probability = 1.0
        for stratum, n in counts.items():
            probability *= (1.0 - per_stratum[stratum]["p_A"]) ** n
        expected += probability
        if probability > 0.5:
            certain_zero += 1
    return {
        "n_survive_expected": expected,
        "n_episodes": len(pooled["per_episode"]),
        "n_episodes_more_likely_than_not_clean": certain_zero,
        "assumption": (
            "(b) occurrences are independent within an episode given its stratum composition. "
            "V15 §5 labels this model-dependent; it does not enter Q1."
        ),
    }


def calibrate(sample: dict, verdicts: dict[str, str], motion: dict) -> dict:
    """§6. Fit a one-feature cut on two thirds of the labelled tiles, test on the held-out third."""
    lookup = {(e["episode"], f["frame_index"]): f
              for e in motion["per_episode"] for f in e["frames"]}
    rows = []
    for record in sample["tiles"]:
        verdict = verdicts[str(record["tile"])]
        if verdict == UNDECIDED:
            continue
        feature = lookup.get((record["episode"], record["frame_index"]))
        if feature is None:
            continue
        rows.append((feature["frac_dev"], verdict == POSITIVE))

    rng = random.Random(AUX_SPLIT_SEED)
    rng.shuffle(rows)
    cut = (len(rows) * 2) // 3
    train, test = rows[:cut], rows[cut:]

    def balanced(threshold: float, data: list[tuple[float, bool]]) -> float:
        pos = [r for r in data if r[1]]
        neg = [r for r in data if not r[1]]
        if not pos or not neg:
            return float("nan")
        tpr = sum(1 for v, _ in pos if v > threshold) / len(pos)
        tnr = sum(1 for v, _ in neg if v <= threshold) / len(neg)
        return (tpr + tnr) / 2

    candidates = sorted({v for v, _ in train})
    best, best_score = None, -1.0
    for value in candidates:
        score = balanced(value, train)
        if score == score and score > best_score:
            best, best_score = value, score

    held_out = balanced(best, test) if best is not None else float("nan")
    admissible = bool(held_out == held_out and held_out >= AUX_BALANCED_ACCURACY)
    return {
        "feature": "frac_dev",
        "n_labelled_rows": len(rows), "n_train": len(train), "n_test": len(test),
        "threshold_fitted_on_train": best,
        "balanced_accuracy_train": best_score,
        "balanced_accuracy_held_out": held_out,
        "required": AUX_BALANCED_ACCURACY,
        "admissible_corpus_wide": admissible,
        "if_not_admissible": (
            "V15 §6: reported as having failed. The finding rests on the stratified sample alone, "
            "with its interval. The auxiliary never overrides a human label and no §5 outcome may "
            "be evaluated on it."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--look", type=pathlib.Path, default=REPO_ROOT / "runs/pr08-empty-mask-look")
    parser.add_argument("--verdicts", type=pathlib.Path, default=None)
    parser.add_argument("--pooled", type=pathlib.Path,
                        default=REPO_ROOT / "runs/pr08-robot-mask-area/POOLED.json")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    sample = json.loads((args.look / "SAMPLE.json").read_text())
    raw = json.loads((args.verdicts or (args.look / "VERDICTS.json")).read_text())
    verdicts = raw.get("verdicts", raw)
    verdicts = {str(k): v for k, v in verdicts.items()}

    motion_path = args.look / "MOTION.json"
    q99 = None
    if motion_path.is_file():
        import numpy as np  # noqa: PLC0415
        motion = json.loads(motion_path.read_text())
        q99 = float(np.percentile([f["frac_dev"] for e in motion["per_episode"]
                                   for f in e["frames"]], 99))

    result = evaluate(sample, verdicts, q99)
    result["n_verdicts"] = len(verdicts)

    if result["evaluated"]:
        pooled = json.loads(args.pooled.read_text())
        result["q2"] = survival(sample, result["per_stratum"], pooled)
        n = result["q2"]["n_survive_expected"]
        if result["q1_outcome"] != "A":
            result["q2_outcome"] = "not evaluated — V15 §5 evaluates Q2 only when Q1 returns A"
        elif n >= Q2_A_SURVIVE:
            result["q2_outcome"] = "A2 — building the §3.2 witness is worth its cost"
        elif n >= Q2_B_SURVIVE:
            result["q2_outcome"] = "M2 — a corpus survives but the loss is an owner decision"
        else:
            result["q2_outcome"] = "B2 — not enough corpus to justify the witness; §3.3 by the practical route"

        result["auxiliary"] = (
            calibrate(sample, verdicts, json.loads(motion_path.read_text()))
            if motion_path.is_file() else
            {"status": "MOTION.json not present; §6 not evaluated"}
        )

    result["licenses"] = (
        "Nothing. T40_RULE_V12 stays unsigned, GATE_QUALIFIED stays False, the blocker tuple is "
        "unchanged, T40_RULE_V1 §1 binds, and no clip is licensed by any outcome here."
    )

    out = args.out or (args.look / "SPLIT.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_stratum"}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
