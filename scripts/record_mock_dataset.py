#!/usr/bin/env python3
"""Record a synthetic mock dataset (D0 systems-test style) and run the T-11 validation gates.

Usage: .venv/bin/python scripts/record_mock_dataset.py --out datasets/mock-d0 --episodes 3
Records N episodes with per-seed DummyPolicy variations (amplitude/period/start pose) on the
mock robot — frames + states + commanded/executed actions, all through the safety layer —
then runs ``validate_dataset`` over the output directory, writes the JSON report next to the
episodes and prints a summary. Exit code 0 iff ALL dataset validation gates pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wam.data import MockCaptureSession, ValidationThresholds, validate_dataset
from wam.interfaces import CanonicalSpaceSpec, load_config
from wam.robot import get_robot
from wam.runtime.mock_loop import DummyPolicy
from wam.safety import SafetyConfig, SafetyLayer

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Rotating short imperative instructions (German is primary, PRD §6 Sprachumfang).
_INSTRUCTIONS = (
    "Greife die rote Tasse.",
    "Stelle die Tasse auf die Markierung.",
    "Greife den blauen Würfel.",
    "Lege den Würfel in die Zielzone.",
)

# Simulated wrist-camera clock skew: exercises the sync accounting, stays within tolerance.
_WRIST_SKEW_NS = 2_000_000  # 2 ms


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "mock-d0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25, help="capture steps per episode")
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=20.0)
    parser.add_argument(
        "--robot-config", type=Path, default=_REPO_ROOT / "configs" / "robot" / "mock.yaml"
    )
    parser.add_argument(
        "--safety-config", type=Path, default=_REPO_ROOT / "configs" / "safety" / "default.yaml"
    )
    parser.add_argument(
        "--report-file", type=Path, default=None, help="default: <out>/validation_report.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    robot_cfg = load_config(args.robot_config)
    safety_cfg = SafetyConfig.model_validate(load_config(args.safety_config))
    robot_section = robot_cfg["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    limits: dict[str, Any] = robot_section.get("limits", {})
    dt_s = float(robot_section.get("control", {}).get("dt_s", 0.05))
    fps = 1.0 / (dt_s * args.prefix_steps)  # one frame per capture step
    tolerance_ns = round(args.sync_tolerance_ms * 1e6)
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(args.episodes):
        episode_seed = args.seed + i
        rng = np.random.default_rng(episode_seed)
        # D0 variations per seed: stay well inside the default safety limits.
        amplitude_rad = float(0.05 + 0.15 * rng.random())
        period_s = float(1.6 + 1.4 * rng.random())
        gripper_period_s = float(4.0 + 2.0 * rng.random())
        initial_q = rng.uniform(-0.3, 0.3, spec.num_joints)

        robot_kwargs: dict[str, Any] = {"spec": spec, "seed": episode_seed, "initial_q": initial_q}
        for key in ("q_min", "q_max", "dq_max"):
            if key in limits:
                robot_kwargs[key] = np.asarray(limits[key], dtype=np.float64)
        robot = get_robot(robot_section["name"], **robot_kwargs)
        policy = DummyPolicy(
            spec,
            steps=args.chunk_steps,
            dt_s=dt_s,
            amplitude_rad=amplitude_rad,
            period_s=period_s,
            gripper_period_s=gripper_period_s,
        )
        safety = SafetyLayer(safety_cfg, spec=spec)
        cameras = tuple(robot.render_frames(1))
        session = MockCaptureSession(
            robot,
            policy,
            safety,
            fps=fps,
            sync_tolerance_ns=tolerance_ns,
            camera_offsets_ns={cam: _WRIST_SKEW_NS for cam in cameras if cam == "wrist"},
            instruction=_INSTRUCTIONS[i % len(_INSTRUCTIONS)],
        )
        episode_id = f"mock-{i:04d}"
        result = session.record_episode(
            args.out / episode_id,
            episode_id,
            iterations=args.iterations,
            prefix_steps=args.prefix_steps,
            extra={
                "d_phase": "D0",
                "seed": episode_seed,
                "policy": {
                    "name": "DummyPolicy",
                    "amplitude_rad": amplitude_rad,
                    "period_s": period_s,
                    "gripper_period_s": gripper_period_s,
                    "steps": args.chunk_steps,
                    "dt_s": dt_s,
                },
            },
        )
        print(
            f"recorded {episode_id}: iterations={result.iterations} "
            f"max_sync_error_ns={result.max_sync_error_ns} "
            f"interventions={result.interventions_total}"
        )

    thresholds = ValidationThresholds(
        sync_tolerance_ns=tolerance_ns, min_episodes=args.episodes
    )
    report = validate_dataset(args.out, thresholds)
    report_path = args.report_file or args.out / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_json() + "\n")

    for episode_report in report.episodes:
        status = "PASS" if episode_report.passed else "FAIL"
        failed = (
            "" if episode_report.passed else f" failed={','.join(episode_report.failed_gates())}"
        )
        print(f"{status} {episode_report.episode_id} duration_s={episode_report.duration_s:.2f}{failed}")
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"{status} dataset gate {gate.name}" + (f" ({gate.detail})" if gate.detail else ""))
    print(
        f"{'OK' if report.passed else 'FAILED'} episodes={len(report.episodes)} "
        f"report={report_path}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
