#!/usr/bin/env python3
"""PR-10 — sweep the ``oracle_action`` anchoring convention and apply ``PR10_RULE_V1``.

WHAT THIS ANSWERS. T-39 reported ``VOID (labels)``: the corpus's own commanded ``action`` column
fails our L1 gate by 359 pp while ``oracle_state``, the identity of our own label pipeline, scores
a bit-exact ``mse 0.0``. Three numbers describe the failure — tiny absolute error, ``horizon_ratio
0.0044``, ``smoothness_ratio 8.52`` — and they are consistent with two different projects:

  a SHIFT   the command leads the executed state by some number of control steps. An adapter
            defect, fixable in ``commanded_to_chunk``, and T-39 becomes re-runnable.
  CONTENT   the commanded stream carries signal the executed trajectory does not. Not fixable by
            re-anchoring; the label space needs a redesign.

This script decides between them by re-scoring the same arm at every offset in a fixed grid and
reading which one, if any, clears the gates T-39 used. Nothing is recalibrated: ``skill_vs_repeat_pct``,
``ci_skill_vs_repeat_pct`` and ``MATERIAL_FLOOR_PP = 10.0`` are T39_RULE_V1's, borrowed unchanged.

THE RULE IS PRE-REGISTERED. ``docs/preregistration/PR-10-anchor-delay-sweep.md``, written before any
offset other than ``k = 0`` was evaluated. The verdict block below is that document's §3 as code
and must not be edited to fit a grid. In particular ``k_best = 0`` is verdict **J**, decided in
advance, because "the convention we already use is the best one" is evidence about content.

TWO VARIANTS, and the second is a control:

  A  ``action[i+k]``, anchor ``state[i]``   — is the command that produced step i actually i+k?
  B  ``action[i+k]``, anchor ``state[i+k]`` — is OUR conversion's time base offset?

B is predicted flat, because ``raw_anchor_indices`` already refuses an inexact timestamp match.
It is run anyway: a control predicted to say nothing is how you learn that something you believed
was proven is not.

THE MARGIN IS NOT OPTIONAL. Every cell is scored with ``--chunk-margin = max|k|`` so that all
offsets share one identical chunk set. Without it a shifted window falls off the end of an episode
at a different chunk and the sweep reports a change of sample set as a delay. The cost is that this
grid's own ``k = 0`` cell does **not** equal the archived −359.41 pp, and must never be quoted as
if it did — the grid's baseline is its own centre cell.

COST. Nine offsets, two variants, CPU, roughly three minutes total. No GPU, no cluster, no billing.

    .venv/bin/python scripts/sweep_t39_anchor.py \\
        --run-dir runs/t39-baseline-seed0 \\
        --dataset datasets/gr00t-apple-full \\
        --raw-dataset ~/wam-t041/raw/GR00T-N1.7-AppleToPlate \\
        --holdout configs/splits/t18_holdout_episodes.txt \\
        --train-episodes configs/splits/i8_train_362.txt \\
        --out runs/t39-baseline-seed0/pr10-anchor-sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = _REPO_ROOT / "scripts" / "eval_t39_baseline.py"

RULE_VERSION = "PR10_RULE_V1"

OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
"""PR-10 §2, fixed. At dt = 33.33 ms this brackets +/-133 ms, wider than any plausible
position-control lag on a 30 Hz corpus. NOT extended after seeing the grid: an optimum on an edge
is reported as unbounded on that side and a wider grid is a new pre-registration."""

MARGIN = max(abs(k) for k in OFFSETS)

MATERIAL_FLOOR_PP = 10.0
"""Borrowed from I8_RULE_V3 exactly as T39_RULE_V1 borrowed it. Not tuned here."""

ARCHIVED_T39 = {
    "mse": 4.197913627356202e-05,
    "skill_vs_repeat_pct": -359.4077743907937,
    "ci_skill_vs_repeat_pct": -102.5407753511892,
    "num_predictions": 1040,
}
"""T-39's published ``oracle_action`` cell, PR-07-RESULT.md. The plumbing check of PR-10 §6
reproduces this with NO sweep flags at all; a mismatch VOIDs the sweep, because a grid built on a
scorer that no longer reproduces the number it is explaining is explaining something else."""


def run_cell(
    *,
    common: list[str],
    out: Path,
    offset: int,
    margin: int,
    co_shift: bool,
) -> dict[str, Any]:
    """One evaluation, through the real command line, and its ``bench.json``."""
    cmd = [
        sys.executable,
        str(EVAL),
        *common,
        "--arm",
        "oracle_action",
        "--out",
        str(out),
        "--action-offset",
        str(offset),
        "--chunk-margin",
        str(margin),
    ]
    if co_shift:
        cmd.append("--co-shift-anchor")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"cell offset={offset:+d} co_shift={co_shift} failed: exit {proc.returncode}")
    return json.loads((out / "bench.json").read_text())


def plumbing_check(common: list[str], out: Path) -> tuple[bool, dict[str, Any], list[str]]:
    """PR-10 §6: with no sweep flags the scorer must still be T-39's, to every published digit."""
    bench = run_cell(common=common, out=out, offset=0, margin=0, co_shift=False)
    complaints = []
    for key, expected in ARCHIVED_T39.items():
        got = bench[key]
        if isinstance(expected, int):
            ok = int(got) == expected
        else:
            ok = abs(float(got) - expected) <= 1e-9 * max(1.0, abs(expected))
        if not ok:
            complaints.append(f"{key}: archived {expected!r}, reproduced {got!r}")
    return (not complaints), bench, complaints


def verdict(grid_a: dict[int, dict[str, Any]]) -> tuple[str, int, float, str]:
    """``PR10_RULE_V1`` §3, in evaluation order. First match wins."""
    k_best = max(grid_a, key=lambda k: grid_a[k]["skill_vs_repeat_pct"])
    gain = grid_a[k_best]["skill_vs_repeat_pct"] - grid_a[0]["skill_vs_repeat_pct"]
    l1 = grid_a[k_best]["skill_vs_repeat_pct"] > 0
    l2 = grid_a[k_best]["ci_skill_vs_repeat_pct"] > 0

    if k_best == 0:
        return "J", k_best, gain, "the convention already in use is the best in the grid"
    if l1 and l2:
        return "D", k_best, gain, f"offset {k_best:+d} clears both L1 and L2"
    if l1:
        return "P", k_best, gain, f"offset {k_best:+d} clears L1 but not L2"
    if gain >= MATERIAL_FLOOR_PP:
        return (
            "P",
            k_best,
            gain,
            f"offset {k_best:+d} gains {gain:.2f} pp (>= {MATERIAL_FLOOR_PP}) without clearing L1",
        )
    return "J", k_best, gain, f"the best offset gains only {gain:.2f} pp and clears nothing"


VERDICT_TEXT = {
    "D": (
        "D — DELAY. The mismatch is an anchoring/latency defect. Licenses a defect report against\n"
        "commanded_to_chunk AND a corrected re-run of T-39's G0b at k_best. It does NOT lift T-39's\n"
        "VOID by itself, and it does NOT license training, generation, or any statement about GR00T."
    ),
    "P": (
        "P — PARTIAL. A real shift, but not the whole story: re-anchoring alone will not produce a\n"
        "label space that clears the bar. The next question is what the residual content is, not\n"
        "which model to try."
    ),
    "J": (
        "J — CONTENT. The commanded stream carries signal the executed trajectory does not, and no\n"
        "offset repairs it. Licenses a defect report against the LABEL SPACE. Retires 'the adapter\n"
        "is mis-anchored' as an available explanation for the fourteen negatives. Licenses no model\n"
        "work of any kind."
    ),
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--raw-dataset", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--train-episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", default="ego")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    common = [
        "--run-dir", str(args.run_dir),
        "--dataset", str(args.dataset),
        "--raw-dataset", str(args.raw_dataset),
        "--holdout", str(args.holdout),
        "--train-episodes", str(args.train_episodes),
        "--camera", args.camera,
        "--device", "cpu",
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"=== {RULE_VERSION} — PR-10 anchor-delay sweep")
    print(f"=== grid {OFFSETS}, margin {MARGIN}, variants A and B, {2 * len(OFFSETS) + 1} evaluations\n")

    print("--- plumbing check: no sweep flags, must reproduce T-39's published cell")
    ok, base, complaints = plumbing_check(common, args.out / "plumbing-k0-unmargined")
    if not ok:
        for line in complaints:
            print(f"    MISMATCH {line}")
        print("\n=== VERDICT: VOID (plumbing) — this scorer does not reproduce the arm it explains.")
        (args.out / "sweep.json").write_text(
            json.dumps({"rule_version": RULE_VERSION, "verdict": "VOID (plumbing)",
                        "complaints": complaints}, indent=2) + "\n"
        )
        return 2
    print(f"    reproduced: mse {base['mse']:.6g}  L1 {base['skill_vs_repeat_pct']:+.2f}  "
          f"L2 {base['ci_skill_vs_repeat_pct']:+.2f}  on {base['num_predictions']} chunks\n")

    grids: dict[str, dict[int, dict[str, Any]]] = {"A": {}, "B": {}}
    for variant, co_shift in (("A", False), ("B", True)):
        print(f"--- variant {variant}{' (control)' if co_shift else ''}")
        for k in OFFSETS:
            bench = run_cell(
                common=common,
                out=args.out / f"variant-{variant}" / f"k{k:+d}",
                offset=k,
                margin=MARGIN,
                co_shift=co_shift,
            )
            grids[variant][k] = bench
            print(
                f"    k {k:+d}  L1 {bench['skill_vs_repeat_pct']:+9.2f}  "
                f"L2 {bench['ci_skill_vs_repeat_pct']:+9.2f}  "
                f"mse {bench['mse']:.4g}  horizon {bench['horizon_ratio']:.4g}  "
                f"smooth {bench['smoothness_ratio']:.4g}  n {bench['num_predictions']}"
            )
        print()

    sizes = {g["num_predictions"] for g in grids["A"].values()} | {
        g["num_predictions"] for g in grids["B"].values()
    }
    if len(sizes) != 1:
        raise SystemExit(
            f"the grid's cells were scored on different chunk counts {sizes} — the margin did not "
            "hold the sample set fixed and the comparison is between sample sets, not offsets"
        )

    code, k_best, gain, why = verdict(grids["A"])
    best = grids["A"][k_best]

    print("=" * 78)
    print(f"=== VERDICT ({RULE_VERSION}): {code}")
    print(f"=== k_best {k_best:+d} ({k_best * 33.333:+.1f} ms), gain {gain:+.2f} pp over k=0 — {why}")
    print("=" * 78)
    print(VERDICT_TEXT[code])
    print()
    print("Secondary readings — PR-10 §5, recorded, never gates:")
    print(f"  smoothness_ratio(k_best) {best['smoothness_ratio']:.4g}  (gate would be <= 2)")
    print(f"  horizon_ratio(k_best)    {best['horizon_ratio']:.4g}  (gate would be <= 4)")
    b_best = max(grids["B"], key=lambda k: grids["B"][k]["skill_vs_repeat_pct"])
    print(f"  variant B best at k {b_best:+d} — predicted 0; anything else questions the clock match")
    print("  gripper_accuracy is WITHHELD on every cell; no cell of this grid can see a grasp.")

    (args.out / "sweep.json").write_text(
        json.dumps(
            {
                "rule_version": RULE_VERSION,
                "preregistration": "docs/preregistration/PR-10-anchor-delay-sweep.md",
                "offsets": list(OFFSETS),
                "margin": MARGIN,
                "material_floor_pp": MATERIAL_FLOOR_PP,
                "dt_ms": 1000.0 / 30.0,
                "plumbing_check": {"passed": True, "archived": ARCHIVED_T39},
                "verdict": code,
                "k_best": k_best,
                "gain_pp": gain,
                "reason": why,
                "chunks_per_cell": sizes.pop(),
                "variant_b_best_k": b_best,
                "grid": {
                    variant: {
                        str(k): {
                            key: bench[key]
                            for key in (
                                "mse", "ci_mse", "skill_vs_zero_pct", "skill_vs_repeat_pct",
                                "ci_skill_vs_repeat_pct", "horizon_ratio", "smoothness_ratio",
                                "level_name", "score", "num_predictions",
                            )
                        }
                        for k, bench in grid.items()
                    }
                    for variant, grid in grids.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
