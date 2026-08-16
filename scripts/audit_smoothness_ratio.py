#!/usr/bin/env python
"""Does ``smoothness_ratio`` measure smoothness? Decompose its sum and find out.

WHY THIS EXISTS. PR-10 and PR-10-RESULT-T-44 both read ``smoothness_ratio ~ 8`` as "the commanded
stream carries high-frequency content the executed trajectory does not". PR-11 then low-passed the
commanded column at 1 Hz — a fifteenth of Nyquist on a 30 Hz signal, which discards essentially all
high-frequency content — and ``smoothness_ratio`` moved by 5 %. Those two statements cannot both be
right, and the second one is a measurement.

THIS SCRIPT SCORES NO ARM AND HAS NO VERDICT. It is not an experiment and it carries no
pre-registration, because it asks a question about our own code rather than about the corpus: given
the prediction files already on disk, which terms of the jerk sum carry it? Nothing here can change
a recorded verdict — L4 smoothness has never gated one; PR-07, PR-10 and PR-11 all turned on L1/L2.

WHAT IT CHECKS. ``bench_metrics`` accumulates, per chunk and per arm::

    d2   = x[2:] - 2*x[1:-1] + x[:-2]        # second difference along the WITHIN-CHUNK time axis
    jerk = sum(d2**2) / count

over a chunk ``x`` in JOINT_DELTA. A second difference of a delta is a third difference of a
position, so the name is defensible — as long as every element of ``x`` is the same quantity.

For the TARGET it is: ``relabel_chunks`` builds ``targets[t] = q[s+t+1] - q[s+t]`` for every t,
one uniform first difference of the executed positions.

For the PREDICTION of ``oracle_action`` it is not. ``commanded_to_chunk`` anchors step 0 on the
observed state and chains the rest through the commands::

    targets[0]   = q_cmd[0] - q_anchor       # the standing TRACKING ERROR, command minus state
    targets[t>0] = q_cmd[t] - q_cmd[t-1]     # a per-step command increment

Element 0 is a different physical quantity from elements 1..15 whenever tracking is imperfect, and
``d2[0] = x[2] - 2*x[1] + x[0]`` is the only term of the sum that contains it. Because the sum is
over SQUARES, a single contaminated term needs only to be a few times larger than the others to
carry the whole statistic. That is the hypothesis, and the numbers below either show it or do not.

This is not a defect in ``commanded_to_chunk``: ``action[t] - q[t]`` is the correct commanded
displacement over step t, the docstring argues it, and the mutation tests pin it. Under PERFECT
tracking (``action[i] == q[i+1]``) element 0 is a plain first difference like the others and no
discontinuity exists at all. The contamination is proportional to how badly the arm tracks, which
is precisely the quantity T-39 set out to measure.

    .venv/bin/python scripts/audit_smoothness_ratio.py \\
        runs/t39-baseline-seed0/pr10-anchor-sweep/variant-A/k+0 \\
        runs/t39-baseline-seed0/pr10-anchor-sweep/variant-A/k-2

Each argument is a directory holding ``predictions.jsonl`` and ``bench.json``. No re-evaluation, no
dataset, no GPU: the chunks were written to disk when the cell was scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wam.evaluation.offline import load_predictions_jsonl  # noqa: E402


def second_differences(x: np.ndarray) -> np.ndarray:
    """``benchmark.py``'s ``d2``, spelled identically so this audits it rather than paraphrases."""
    return x[2:] - 2.0 * x[1:-1] + x[:-2]


def decompose(predictions, key: str) -> dict:
    """Per-index jerk contributions for one arm, plus the total ``benchmark.py`` would compute."""
    per_index: dict[int, float] = {}
    count = 0
    n_terms = 0
    for pred in predictions:
        chunk = pred.predicted if key == "pred" else pred.target
        x = np.asarray(chunk.targets, dtype=np.float64)
        if x.shape[0] < 3:
            continue
        d2 = second_differences(x)
        # (d2**2).sum() split by the WITHIN-CHUNK index of each second difference. Summing the
        # values below reproduces benchmark.py's accumulator exactly; that is the point.
        for i, row in enumerate(d2):
            per_index[i] = per_index.get(i, 0.0) + float((row**2).sum())
        count += d2.size
        n_terms = max(n_terms, d2.shape[0])
    total = sum(per_index.values())
    return {
        "total_sq": total,
        "count": count,
        "jerk": total / count if count else 0.0,
        "per_index": [per_index.get(i, 0.0) for i in range(n_terms)],
        "n_terms": n_terms,
    }


def ratio(pred: float, target: float) -> float:
    """``benchmark._ratio``'s contract for the non-degenerate case, which is the only one here."""
    if target == 0.0:
        return float("inf") if pred > 0.0 else 1.0
    return pred / target


def audit(run: Path) -> dict:
    predictions = load_predictions_jsonl(run / "predictions.jsonl")
    if not predictions:
        raise SystemExit(f"{run}: no predictions")
    p = decompose(predictions, "pred")
    t = decompose(predictions, "target")
    measured = ratio(p["jerk"], t["jerk"])

    published = None
    bench = run / "bench.json"
    if bench.is_file():
        published = float(json.loads(bench.read_text())["smoothness_ratio"])

    # The same ratio with the one contaminated term removed from BOTH arms. Removing it from both
    # is what keeps this a like-for-like comparison: the target has no discontinuity at index 0, so
    # dropping its index 0 as well costs the target a legitimate term and biases the result AGAINST
    # the hypothesis. If the ratio collapses anyway, it collapses for the stated reason.
    def without_first(d: dict) -> float:
        kept = sum(d["per_index"][1:])
        n = d["count"] - d["count"] // d["n_terms"]
        return kept / n if n else 0.0

    return {
        "run": str(run),
        "n_chunks": len(predictions),
        "smoothness_ratio_published": published,
        "smoothness_ratio_recomputed": measured,
        "pred": p,
        "target": t,
        "smoothness_ratio_without_index0": ratio(without_first(p), without_first(t)),
        "index0_share_pred": p["per_index"][0] / p["total_sq"] if p["total_sq"] else 0.0,
        "index0_share_target": t["per_index"][0] / t["total_sq"] if t["total_sq"] else 0.0,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['run']}  ({result['n_chunks']} chunks)")
    pub = result["smoothness_ratio_published"]
    got = result["smoothness_ratio_recomputed"]
    print(f"    smoothness_ratio  published {pub!r}  recomputed {got:.6g}")
    if pub is not None:
        drift = abs(pub - got)
        flag = "OK" if drift < 1e-6 else "MISMATCH — this audit is not measuring bench.json"
        print(f"                      drift {drift:.3g}  [{flag}]")

    for key in ("pred", "target"):
        d = result[key]
        shares = [v / d["total_sq"] if d["total_sq"] else 0.0 for v in d["per_index"]]
        print(f"    {key:6s} jerk {d['jerk']:.6g} over {d['n_terms']} second differences")
        print("           share by within-chunk index:")
        print("           " + "  ".join(f"{i}:{s:6.1%}" for i, s in enumerate(shares)))

    print(f"    index 0 carries {result['index0_share_pred']:.1%} of the PREDICTED jerk sum")
    print(f"    index 0 carries {result['index0_share_target']:.1%} of the TARGET jerk sum")
    print(f"    smoothness_ratio with index 0 dropped from both arms: "
          f"{result['smoothness_ratio_without_index0']:.6g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+", help="directories holding predictions.jsonl")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    results = []
    for run in args.runs:
        result = audit(run)
        report(result)
        results.append(result)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
