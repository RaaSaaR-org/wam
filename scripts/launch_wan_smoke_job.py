#!/usr/bin/env python3
"""Launch the Wan backbone smoke test on Hugging Face Jobs (OD-05 GPU, pay-per-second).

Mounts the Wan model repo read-only (no download inside the job) plus this repo's ``src/``,
so the job exercises the real ``wam`` package, then runs ``hf_job_wan_smoke.py``.

    python scripts/launch_wan_smoke_job.py --dry-run          # print the CLI equivalent
    python scripts/launch_wan_smoke_job.py --flavor l40sx1    # actually spend money

Requires: ``pip install 'huggingface_hub>=0.34'`` and ``hf auth login`` (PRO + credit balance).
The JSON report is printed to the job log; pass ``--bucket <user/bucket>`` to also persist it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "hf_job_wan_smoke.py"
DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# $/h at the time of writing — see docs/hf_jobs.md.
FLAVOR_COST = {"a10g-small": 1.00, "l4x1": 0.80, "l40sx1": 1.80, "rtx-pro-6000": 2.75, "h200": 5.00}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Wan repo mounted read-only at /model")
    p.add_argument(
        "--flavor", default="l40sx1", help=f"one of {sorted(FLAVOR_COST)} (or any HF flavor)"
    )
    p.add_argument("--timeout", default="45m")
    p.add_argument("--namespace", default=None, help="run under an org instead of your user")
    p.add_argument("--episode", default=None, help="local episode dir to mount at /episode")
    p.add_argument("--bucket", default=None, help="storage bucket 'user/name' for the report")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--blocks", default="", help="comma-separated readout blocks (default auto)")
    p.add_argument("--offload-text", action="store_true", help="needed on 24 GB flavors")
    p.add_argument("--dry-run", action="store_true", help="print the hf CLI command, run nothing")
    return p.parse_args(argv)


def script_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--source", "/model",
        "--device", "cuda",
        "--dtype", args.dtype,
        "--frames", str(args.frames),
        "--height", str(args.height),
        "--width", str(args.width),
        "--out", "/outputs/wan_smoke_report.json" if args.bucket else "/tmp/wan_smoke_report.json",
    ]  # fmt: skip
    if args.blocks:
        out += ["--blocks", args.blocks]
    if args.offload_text:
        out += ["--offload-text"]
    if args.episode:
        out += ["--episode", "/episode"]
    return out


def mounts(args: argparse.Namespace) -> list[str]:
    """``-v`` sources: Hub repos use the hf:// scheme, local dirs sync to jobs-artifacts."""
    specs = [f"hf://{args.model}:/model", f"{REPO_ROOT / 'src'}:/wam-src"]
    if args.episode:
        specs.append(f"{args.episode}:/episode")
    if args.bucket:
        specs.append(f"hf://buckets/{args.bucket}:/outputs")
    return specs


def cli_equivalent(args: argparse.Namespace) -> str:
    tokens = ["hf", "jobs", "uv", "run", "--flavor", args.flavor, "--timeout", args.timeout]
    tokens += ["--name", "wan-smoke", "--secrets", "HF_TOKEN"]
    for spec in mounts(args):
        tokens += ["-v", spec]
    tokens += [str(SMOKE_SCRIPT.relative_to(REPO_ROOT)), "--", *script_args(args)]
    return shlex.join(tokens)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cost = FLAVOR_COST.get(args.flavor)
    print(f"flavor={args.flavor}" + (f"  (~${cost:.2f}/h)" if cost else "  (unknown price)"))
    print(f"model={args.model}\n")
    print(cli_equivalent(args), "\n")
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import Volume, get_token, run_uv_job, sync_job_volume
    except ImportError:
        print("pip install 'huggingface_hub>=0.34' first", file=sys.stderr)
        return 2

    volumes = [
        Volume(type="model", source=args.model, mount_path="/model"),
        sync_job_volume(str(REPO_ROOT / "src"), "/wam-src"),
    ]
    if args.episode:
        volumes.append(sync_job_volume(args.episode, "/episode"))
    if args.bucket:
        volumes.append(
            Volume(type="bucket", source=args.bucket, mount_path="/outputs", read_only=False)
        )

    job = run_uv_job(
        str(SMOKE_SCRIPT),
        script_args=script_args(args),
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        volumes=volumes,
        secrets={"HF_TOKEN": get_token()},
        name="wan-smoke",
    )
    print(f"job {job.id} -> {job.url}")
    print(f"logs: hf jobs logs {job.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
