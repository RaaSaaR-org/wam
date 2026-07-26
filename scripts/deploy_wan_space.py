#!/usr/bin/env python3
"""Deploy the Wan backbone smoke test as a ZeroGPU Space (free GPU on a PRO account).

The alternative to `launch_wan_smoke_job.py`: HF Jobs bills against a pre-paid credit
balance, ZeroGPU is included in PRO (40 min/day). Same checks either way — this uploads
`scripts/hf_job_wan_smoke.py` verbatim as the Space's `smoke.py`, so there is one
implementation of the checks and no vendored copy to drift.

    python scripts/deploy_wan_space.py --dry-run
    python scripts/deploy_wan_space.py                  # create/update, then open the URL

Requires `pip install 'huggingface_hub>=0.34'` and `hf auth login`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO_ROOT / "deploy" / "wan-smoke-space"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "hf_job_wan_smoke.py"
DEFAULT_SPACE = "wam-wan-smoke"
# 'zero-a10g' is the API's (legacy) identifier for ZeroGPU; the actual hardware is an
# RTX Pro 6000 Blackwell — 48 GB for `large`, the @spaces.GPU default.
ZERO_GPU = "zero-a10g"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--space", default=None, help=f"repo id; default <user>/{DEFAULT_SPACE}")
    p.add_argument("--model", default=None, help="override MODEL_ID variable on the Space")
    p.add_argument("--duration", type=int, default=None, help="GPU_DURATION seconds variable")
    p.add_argument("--hardware", default=ZERO_GPU, help="Space hardware (default ZeroGPU)")
    p.add_argument("--public", action="store_true", help="create it public (default private)")
    p.add_argument("--dry-run", action="store_true", help="show what would be uploaded")
    return p.parse_args(argv)


def payload() -> list[tuple[Path, str]]:
    """(local file, path in the Space). The smoke script is renamed, never edited."""
    files = [(path, path.name) for path in sorted(SPACE_DIR.iterdir()) if path.is_file()]
    return [*files, (SMOKE_SCRIPT, "smoke.py")]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = payload()
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        print(f"missing source files: {missing}", file=sys.stderr)
        return 2

    print(f"hardware={args.hardware}  visibility={'public' if args.public else 'private'}")
    for src, dst in files:
        print(f"  {src.relative_to(REPO_ROOT)} -> {dst}  ({src.stat().st_size / 1024:.1f} KB)")

    if args.dry_run:
        print(f"\nwould deploy to: {args.space or f'<user>/{DEFAULT_SPACE}'}")
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print("pip install 'huggingface_hub>=0.34' first", file=sys.stderr)
        return 2

    api = HfApi()
    space_id = args.space or f"{api.whoami()['name']}/{DEFAULT_SPACE}"
    variables = {}
    if args.model:
        variables["MODEL_ID"] = args.model
    if args.duration:
        variables["GPU_DURATION"] = str(args.duration)

    api.create_repo(
        space_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=args.hardware,
        private=not args.public,
        exist_ok=True,
    )
    for key, value in variables.items():
        api.add_space_variable(space_id, key, value)

    api.create_commit(
        space_id,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files
        ],
        commit_message="deploy WAM Wan backbone smoke test",
    )
    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\nspace: {url}")
    print(f"build logs: {url}?logs=build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
