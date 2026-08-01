#!/usr/bin/env python3
"""Gate: does this dataset's gripper channel contain a grasp? (T-31)

Exit code 0 = PASS, 1 = FAIL. That is the point of this script — it is something a runbook or a
CI step can call before anything is trained or any grasping claim is written down, not a report a
human has to remember to read. No GPU, no video decode: parquet columns only.

    .venv/bin/python scripts/audit_gripper.py datasets/gr00t-apple-full
    .venv/bin/python scripts/audit_gripper.py --lerobot data/raw/gr00t_apple

Run both when a converted set fails. A FAIL on the converted set alone is equally consistent with
a dataset that has no grasp and a converter that destroyed one; only the ``--lerobot`` audit of
the source snapshot tells the two apart — which is exactly how the GR00T mapping bug was found
(``docs/benchmark.md``, "What the bench refuses to report").

The exit code is decided by the pre-registered CLAUSES alone. A PASS can still carry findings —
the report's "## Findings" section holds ``NOTE (not gated)`` lines, today for clipping — so read
the report even when the gate is green: a notice says the numbers above it are an upper bound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.evaluation.gripper import audit_lerobot_dataset, audit_wam_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset root (WAM layout, or LeRobot snapshot)")
    parser.add_argument(
        "--lerobot",
        action="store_true",
        help="read a raw LeRobot v2.1 snapshot instead of a converted WAM dataset",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="audit only the first N episodes"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip per-episode checksum verification (WAM mode only; verification is the "
        "default because an audit of silently corrupted parquet proves nothing)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)

    if args.lerobot:
        report = audit_lerobot_dataset(args.root, max_episodes=args.max_episodes)
    else:
        report = audit_wam_dataset(
            args.root,
            verify_checksums=not args.no_verify,
            max_episodes=args.max_episodes,
        )

    print(report.to_json() if args.json else report.render_markdown())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.to_json() + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
