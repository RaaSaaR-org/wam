#!/usr/bin/env python3
"""MVP acceptance run AC-01..AC-07 (T-23) over recorded rollout logs.

Usage:
  .venv/bin/python scripts/run_acceptance.py --rollout-logs "runs/rollouts/*.jsonl"
      [--e1-report REPORT.json ...] [--ablation-json ABLATION.json]
      [--out-dir runs/acceptance] [--known-task pick_place_known]
      [--generalization-task pick_place_generalization] [--backbones flux3,wan_i2v]

Reads every jsonl log matching the glob (``kind == 'rollout_summary'`` records per the
shared rollout-log contract; the first ``kind == 'run_metadata'`` line feeds AC-04),
evaluates the seven acceptance criteria and writes ``acceptance_report.json`` +
``acceptance_report.md`` into ``--out-dir``. The markdown dashboard is printed to stdout.

Exit code: 0 when no criterion FAILED (pending criteria are allowed while data is still
being collected), 1 on any failed criterion or usage/data error.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from wam.evaluation.acceptance import (
    DEFAULT_GENERALIZATION_TASK,
    DEFAULT_KNOWN_TASK,
    evaluate_acceptance,
    load_rollout_summaries,
    load_run_metadata,
)
from wam.evaluation.offline import E1Report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rollout-logs",
        required=True,
        metavar="GLOB",
        help="glob for rollout jsonl logs, e.g. 'runs/rollouts/*.jsonl' (quote it)",
    )
    parser.add_argument(
        "--e1-report",
        action="append",
        type=Path,
        default=None,
        metavar="REPORT.json",
        help="E1Report JSON to attach as offline evidence (repeatable)",
    )
    parser.add_argument(
        "--ablation-json",
        type=Path,
        default=None,
        metavar="ABLATION.json",
        help="AblationReport JSON for AC-07 (omit while D2 data is pending)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/acceptance"),
        help="output directory for acceptance_report.{json,md} (default: runs/acceptance)",
    )
    parser.add_argument("--known-task", default=DEFAULT_KNOWN_TASK, help="AC-01 task label")
    parser.add_argument(
        "--generalization-task", default=DEFAULT_GENERALIZATION_TASK, help="AC-02 task label"
    )
    parser.add_argument(
        "--backbones",
        default=None,
        metavar="a,b,...",
        help="comma-separated backbone registry names for the AC-05 swap check "
        "(default: all available)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    log_files = sorted(glob.glob(args.rollout_logs, recursive=True))
    if not log_files:
        print(f"error: no rollout logs match {args.rollout_logs!r}", file=sys.stderr)
        return 1

    try:
        summaries = load_rollout_summaries(log_files)
        metadata_line = load_run_metadata(log_files)
        e1_reports = {
            path.stem: E1Report.from_json(path.read_text(encoding="utf-8"))
            for path in (args.e1_report or [])
        }
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    backbone_names = None
    if args.backbones:
        backbone_names = tuple(name.strip() for name in args.backbones.split(",") if name.strip())

    report = evaluate_acceptance(
        summaries,
        metadata_line=metadata_line,
        e1_reports=e1_reports or None,
        ablation_json=args.ablation_json,
        known_task=args.known_task,
        generalization_task=args.generalization_task,
        backbone_names=backbone_names,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "acceptance_report.json"
    md_path = args.out_dir / "acceptance_report.md"
    markdown = report.render_markdown()
    json_path.write_text(report.to_json() + "\n", encoding="utf-8")
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print()
    print(f"[report] {json_path}")
    print(f"[report] {md_path}")
    print(f"[logs] {len(log_files)} file(s), {len(summaries)} rollout summaries")

    return 0 if not report.failed_criteria() else 1


if __name__ == "__main__":
    sys.exit(main())
