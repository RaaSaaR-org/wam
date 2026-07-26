#!/usr/bin/env python3
"""Run the T-03 / M0 exit-criterion loop: dummy policy + safety + watchdog + JSONL logging.

Usage: .venv/bin/python scripts/run_mock_loop.py --iterations 20
Loads configs/robot/mock.yaml + configs/safety/default.yaml, builds everything via the
robot registry, runs the closed loop on the mock robot (no hardware) and prints a
one-line summary. Exit code 0 on success. ``--stall-at N`` simulates a late policy at
iteration N to demonstrate the watchdog HOLD/STOP path.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from wam.interfaces import CanonicalSpaceSpec, JsonlRunLogger, RunMetadata, load_config
from wam.robot import get_robot
from wam.runtime.mock_loop import DummyPolicy, run_mock_loop
from wam.safety import SafetyConfig, SafetyLayer, Watchdog

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config", type=Path, default=_REPO_ROOT / "configs" / "robot" / "mock.yaml"
    )
    parser.add_argument(
        "--safety-config", type=Path, default=_REPO_ROOT / "configs" / "safety" / "default.yaml"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--log-dir", type=Path, default=_REPO_ROOT / "runs")
    parser.add_argument(
        "--stall-at",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="simulate a policy stall at iteration N (repeatable; triggers the watchdog)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    robot_cfg = load_config(args.robot_config)
    safety_cfg = SafetyConfig.model_validate(load_config(args.safety_config))

    robot_section = robot_cfg["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    limits: dict[str, Any] = robot_section.get("limits", {})
    dt_s = float(robot_section.get("control", {}).get("dt_s", 0.05))

    robot_kwargs: dict[str, Any] = {"spec": spec}
    for key in ("q_min", "q_max", "dq_max"):
        if key in limits:
            robot_kwargs[key] = np.asarray(limits[key], dtype=np.float64)
    robot = get_robot(robot_section["name"], **robot_kwargs)

    policy = DummyPolicy(spec, steps=args.chunk_steps, dt_s=dt_s)
    safety = SafetyLayer(safety_cfg, spec=spec)
    watchdog = Watchdog.from_config(safety_cfg)

    stall_at = tuple(args.stall_at or ())
    loop_cfg = {
        "iterations": args.iterations,
        "prefix_steps": args.prefix_steps,
        "chunk_steps": args.chunk_steps,
        "dt_s": dt_s,
        "stall_at": sorted(stall_at),
    }
    run_id = f"mock-loop-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    metadata = RunMetadata.create(
        run_id, {"robot": robot_cfg, "safety": safety_cfg, "loop": loop_cfg}
    )
    log_path = args.log_dir / f"{run_id}.jsonl"

    with JsonlRunLogger(log_path, metadata) as logger:
        logger.log_metadata()
        result = run_mock_loop(
            robot,
            policy,
            safety,
            logger,
            args.iterations,
            args.prefix_steps,
            watchdog=watchdog,
            stall_at=stall_at,
        )

    print(
        f"OK run_id={run_id} iterations={result.iterations} "
        f"executed={result.executed_iterations} watchdog_timeouts={result.watchdog_timeouts} "
        f"interventions={result.interventions_total} log={log_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
