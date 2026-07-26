#!/usr/bin/env python3
"""Render a T-09 episode report (FR-08) from a recorded episode directory.

Usage: .venv/bin/python scripts/episode_report.py <episode_dir> [--json] [--no-verify]
Prints a markdown report (default) or the full report as JSON (``--json``).
``--no-verify`` skips manifest checksum verification (faster, for local inspection only).
Exit code 0 on success; errors (missing/tampered episode) raise normally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wam.data import EpisodeReader
from wam.data.replay import episode_report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path, help="episode directory (with manifest.json)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--no-verify", action="store_true", help="skip manifest checksum verification"
    )
    parser.add_argument(
        "--sync-tolerance-ns",
        type=int,
        default=None,
        help="override the sync-error flag tolerance (default: half nominal frame period)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reader = EpisodeReader(args.episode_dir, verify_checksums=not args.no_verify)
    report = episode_report(reader, sync_tolerance_ns=args.sync_tolerance_ns)
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(report.render_markdown(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
