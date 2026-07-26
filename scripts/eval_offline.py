#!/usr/bin/env python3
"""E1 offline evaluation dashboard + AC-07 ablation comparison (T-14 / T-18).

Usage:
  .venv/bin/python scripts/eval_offline.py PREDICTIONS.jsonl [--joint-names j0,j1,...]
                                           [--out report.json]
  .venv/bin/python scripts/eval_offline.py --compare BASELINE.json CANDIDATE.json
                                           [--threshold-pct 5.0]

Mode 1 prints the E1 metrics dashboard (markdown) for a JSONL file of serialized chunk
predictions; ``--out`` additionally writes the E1Report as JSON (feed those into --compare).
Mode 2 loads two E1Report JSON files — the ACTION-ONLY baseline first, the WORLD-ACTION
candidate second — and prints the ablation report (markdown).

JSONL prediction format (one object per line, float32 semantics, normalized targets):
  {"episode_id": "ep001", "t_ns": 0,
   "predicted": {"mode": "joint_delta", "dt_s": 0.05,
                 "targets": [[...], ...], "gripper_target": [...]},
   "target":    {"mode": "joint_delta", "dt_s": 0.05,
                 "targets": [[...], ...], "gripper_target": [...]}}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wam.evaluation import E1Report, compare_runs, e1_metrics, load_predictions_jsonl
from wam.evaluation.ablation import DEFAULT_THRESHOLD_PCT
from wam.interfaces import CanonicalSpaceSpec


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "predictions",
        type=Path,
        nargs="?",
        default=None,
        help="JSONL file of serialized chunk predictions (see module docstring)",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        default=None,
        metavar="j0,j1,...",
        help="comma-separated canonical joint names to label per-joint metrics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="REPORT.json",
        help="also write the E1Report as JSON (input for --compare)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        default=None,
        metavar=("BASELINE.json", "CANDIDATE.json"),
        help="compare two E1Report JSON files: action-only baseline, world-action candidate",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help="verdict threshold on relative MSE improvement (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if (args.predictions is None) == (args.compare is None):
        parser.error("provide either PREDICTIONS.jsonl or --compare, not both/neither")
    return args


def _run_e1(args: argparse.Namespace) -> int:
    spec = None
    if args.joint_names:
        names = tuple(n.strip() for n in args.joint_names.split(",") if n.strip())
        spec = CanonicalSpaceSpec(joint_names=names)
    predictions = load_predictions_jsonl(args.predictions)
    report = e1_metrics(predictions, spec)
    print(report.render_markdown())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    baseline_path, candidate_path = args.compare
    baseline_name = baseline_path.stem
    candidate_name = candidate_path.stem
    if baseline_name == candidate_name:
        baseline_name, candidate_name = f"baseline:{baseline_name}", f"candidate:{candidate_name}"
    reports = {
        baseline_name: E1Report.from_json(baseline_path.read_text(encoding="utf-8")),
        candidate_name: E1Report.from_json(candidate_path.read_text(encoding="utf-8")),
    }
    ablation = compare_runs(
        reports,
        baseline=baseline_name,
        candidate=candidate_name,
        threshold_pct=args.threshold_pct,
    )
    print(ablation.render_markdown())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.compare is not None:
            return _run_compare(args)
        return _run_e1(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
