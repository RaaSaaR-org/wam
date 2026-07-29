#!/usr/bin/env python3
"""Score archived predictions on the WAM-Bench ladder (T-27).

Reads a run's ``predictions.jsonl`` — the same artifact E1 and the T-18 ablation consume — and
writes ``bench.json`` + ``bench.md`` next to it. No model, no GPU, no robot: everything on the
ladder is recomputable from stored predictions, which is what makes past runs re-scorable when
the metric set changes.

    .venv/bin/python scripts/run_bench.py runs/d1-full-gen-seed0
    .venv/bin/python scripts/run_bench.py runs/a runs/b --compare   # side by side

``--compare`` requires the runs to share a holdout split, exactly like ``compare_runs`` in the
ablation harness — otherwise the two ladders are scored on different data and the columns mean
nothing next to each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import (
    BenchReport,
    bench_metrics,
    compare_bench,
    load_predictions_jsonl,
)


def _score(run_dir: Path, quantile: float, write: bool) -> tuple[BenchReport, frozenset[str]]:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"{predictions_path} not found — run E1 for this run first")
    predictions = load_predictions_jsonl(predictions_path)
    report = bench_metrics(predictions, run_name=run_dir.name, critical_quantile=quantile)
    if write:
        (run_dir / "bench.json").write_text(report.to_json() + "\n", encoding="utf-8")
        (run_dir / "bench.md").write_text(report.render_markdown(), encoding="utf-8")
    return report, frozenset(p.episode_id for p in predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+", help="run directory/-ies to score")
    parser.add_argument(
        "--critical-quantile",
        type=float,
        default=None,
        help="motion-energy quantile above which a chunk counts as task-critical "
        "(default: the pre-registered CRITICAL_QUANTILE)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="with exactly two runs, also print a side-by-side table (same holdout required)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only, write no artifacts")
    args = parser.parse_args()

    from wam.evaluation import CRITICAL_QUANTILE

    quantile = CRITICAL_QUANTILE if args.critical_quantile is None else args.critical_quantile
    if args.compare and len(args.run_dir) != 2:
        raise SystemExit("--compare needs exactly two run directories")

    scored = [_score(d, quantile, not args.no_write) for d in args.run_dir]
    for report, _ in scored:
        print(report.render_markdown())

    if args.compare:
        (baseline, ids_a), (candidate, ids_b) = scored
        if ids_a != ids_b:
            only_a, only_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
            raise SystemExit(
                f"holdout mismatch — not comparable: {len(only_a)} episode(s) only in "
                f"{baseline.run_name} ({only_a[:3]}), {len(only_b)} only in "
                f"{candidate.run_name} ({only_b[:3]})"
            )
        print(compare_bench(baseline, candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
