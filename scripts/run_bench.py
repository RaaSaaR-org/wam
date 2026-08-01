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

Bench specs (``--spec``). L4 became a two-sided band on 2026-08-01 and that is a change to a
scoring RULE, so it is versioned rather than applied retroactively: ``bench.json`` keeps meaning
spec 0.1.0 forever and every other spec writes ``bench-<version>.json``. The default scores ALL
registered specs and prints them together, because a reader who sees one headline without the
other has been handed the flattering half of a rule change.

REPORT PAYLOADS ARE ALSO VERSIONED, AND OLDER ONES ARE NOT OVERWRITTEN. ``spec_version`` says
which RULE scored a run; ``report_version`` says which FIELDS the file carries, and the two move
independently. A re-score writes today's payload, so pointing this script at an archived
``bench.json`` from an older ``report_version`` would silently replace a recorded artifact with
a differently-shaped one — the T-27 runs, for instance, recorded a concrete ``gripper_accuracy``
that today's code correctly WITHHOLDS as inadmissible, and rewriting the file in place would
erase the number a published verdict was read from. Every such overwrite therefore refuses
unless ``--overwrite-archived`` is passed; ``--no-write`` prints without touching anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation import (
    BENCH_SPEC_DEFAULT,
    BENCH_SPECS,
    BENCH_VERSION,
    BenchReport,
    bench_metrics,
    compare_bench,
    load_predictions_jsonl,
)


def _archived_report_version(path: Path) -> str | None:
    """``report_version`` of an existing bench json, or None if there is nothing readable.

    An unparseable or absent file is not an archive worth protecting, so it never blocks a
    write; only a file that says it is a DIFFERENT payload version does.
    """
    if not path.is_file():
        return None
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = recorded.get("report_version") if isinstance(recorded, dict) else None
    return str(version) if version else None


def _stem(version: str) -> str:
    """The registered spec owns ``bench.json``; anything else gets its own filename, so a
    re-score under a newer RULE can never overwrite the artifact a recorded verdict was read
    from."""
    return "bench" if version == BENCH_SPEC_DEFAULT else f"bench-{version}"


def _refuse_archived_overwrites(run_dir: Path, specs: list[str]) -> None:
    """Refuse before writing ANYTHING when a target holds an older report PAYLOAD.

    Checked up front rather than inside the write loop: a refusal that fires halfway through
    the spec list would leave a run directory half re-scored, which is a worse artifact than
    either the old one or the new one.
    """
    stale = {
        stem: archived
        for stem in (_stem(v) for v in specs)
        if (archived := _archived_report_version(run_dir / f"{stem}.json")) is not None
        and archived != BENCH_VERSION
    }
    if not stale:
        return
    listed = ", ".join(f"{stem}.json (report_version {v})" for stem, v in sorted(stale.items()))
    raise SystemExit(
        f"{run_dir}: {listed} predates this code's report_version {BENCH_VERSION}. Rewriting "
        "in place would replace an archived artifact with a differently-shaped one, and "
        "fields a recorded verdict was read from can disappear that way (the T-27 runs' "
        "gripper_accuracy is now correctly WITHHELD, so a re-score erases the number). Use "
        "--no-write to print only, copy the file aside first, or pass --overwrite-archived "
        "to replace it deliberately."
    )


def _score(
    run_dir: Path, quantile: float, write: bool, specs: list[str], overwrite_archived: bool
) -> tuple[dict[str, BenchReport], frozenset[str]]:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"{predictions_path} not found — run E1 for this run first")
    if write and not overwrite_archived:
        _refuse_archived_overwrites(run_dir, specs)
    predictions = load_predictions_jsonl(predictions_path)
    reports: dict[str, BenchReport] = {}
    for version in specs:
        report = bench_metrics(
            predictions,
            run_name=run_dir.name,
            critical_quantile=quantile,
            spec_version=version,
        )
        reports[version] = report
        if write:
            stem = _stem(version)
            (run_dir / f"{stem}.json").write_text(report.to_json() + "\n", encoding="utf-8")
            (run_dir / f"{stem}.md").write_text(report.render_markdown(), encoding="utf-8")
    return reports, frozenset(p.episode_id for p in predictions)


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
        "--spec",
        choices=[*sorted(BENCH_SPECS), "all"],
        default="all",
        help="bench spec to score under (default: all registered specs, printed together)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="with exactly two runs, also print a side-by-side table (same holdout required)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only, write no artifacts")
    parser.add_argument(
        "--overwrite-archived",
        action="store_true",
        help=(
            "replace an existing bench json written at a DIFFERENT report_version. Off by "
            "default: the default invocation must never destroy an archived report in place"
        ),
    )
    args = parser.parse_args()

    from wam.evaluation import CRITICAL_QUANTILE

    quantile = CRITICAL_QUANTILE if args.critical_quantile is None else args.critical_quantile
    if args.compare and len(args.run_dir) != 2:
        raise SystemExit("--compare needs exactly two run directories")
    specs = sorted(BENCH_SPECS) if args.spec == "all" else [args.spec]

    write = not args.no_write
    scored = [_score(d, quantile, write, specs, args.overwrite_archived) for d in args.run_dir]
    for reports, _ in scored:
        for version in specs:
            print(reports[version].render_markdown())
        if len(specs) > 1:
            print("Headline under each bench spec (both are reported, neither replaces the other):")
            for version in specs:
                r = reports[version]
                print(f"  spec {version}: level {r.level_name} · score {r.score:.1f}/100")
            print()

    if args.compare:
        (reports_a, ids_a), (reports_b, ids_b) = scored
        if ids_a != ids_b:
            only_a, only_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
            name_a = reports_a[specs[0]].run_name
            name_b = reports_b[specs[0]].run_name
            raise SystemExit(
                f"holdout mismatch — not comparable: {len(only_a)} episode(s) only in "
                f"{name_a} ({only_a[:3]}), {len(only_b)} only in {name_b} ({only_b[:3]})"
            )
        for version in specs:
            print(compare_bench(reports_a[version], reports_b[version]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
