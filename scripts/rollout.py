#!/usr/bin/env python3
"""Closed-loop rollout CLI — the E2/E3 entry point (M4 integration, T-22/T-23).

Builds robot + SafetyLayer + Watchdog + ClosedLoopExecutor, runs the T-22 E2 static
release gate, then executes N rollouts and writes rollout logs per the SHARED ROLLOUT
LOG CONTRACT (one ``kind="control_cycle"`` line per cycle, one ``kind="rollout_summary"``
line per rollout) plus one ``kind="run_metadata"`` line with checkpoint_ref +
dataset_snapshot_ref (AC-04) — the input to ``scripts/run_acceptance.py``.

Usage examples::

  .venv/bin/python scripts/rollout.py --rollouts 100 --task sim:reach
  .venv/bin/python scripts/rollout.py --rollouts 10 --fault-injection
  .venv/bin/python scripts/rollout.py --policy remote --server-uri ws://127.0.0.1:8765
  .venv/bin/python scripts/rollout.py --robot g1 --policy dummy --rollouts 1
  .venv/bin/python scripts/rollout.py --robot mujoco_g1 --policy dummy --rollouts 1
  .venv/bin/python scripts/rollout.py --robot mujoco_g1 --policy joint \\
      --checkpoint runs/t18-real-ablation-seed0/checkpoint.safetensors \\
      --policy-camera head --image-hw 120 160

POLICY KINDS: ``dummy`` (sinusoid), ``checkpoint`` (action-only baseline, T-13), ``joint``
(the world-action model, T-16) and ``remote`` (either, over WebSocket). ``joint`` and
``checkpoint`` load different artifacts and are not interchangeable.

WHAT A ``--policy joint`` SIM RUN MEASURES, AND WHAT IT DOES NOT: on ``mujoco_g1`` it
measures latency against the deadline, the safety/watchdog paths under real model timing,
and that predicted chunks survive the filter at all. It does NOT measure task competence —
MuJoCo renderings are not RealSense images and no video backbone in this repo has seen one
(docs/sim.md, "What this does not prove", item 1). Label such runs accordingly.

THE ``sim:reach`` SUCCESS PROXY (MockRobot only, ``--task sim:reach``):
  ``success_fn(state) = mean_j |q_j - REACH_TARGET_RAD_j| < --reach-tolerance-rad``.
  ``REACH_TARGET_RAD`` is the fixed pose the D1-overfit checkpoint
  (``runs/d1-overfit-seed0``) drives MockRobot to after ~1.0-1.4 s (4-7 replan cycles)
  of coherent closed-loop sinusoid tracking — measured once, frozen here. Start poses are
  jittered per rollout (``--start-jitter-rad``), so the initial error (~0.18 rad mean) is
  far above the tolerance (0.05 rad): success REQUIRES several genuine replan cycles and
  cannot trigger on the start pose or a held robot. This proxies only the "reach the
  pre-grasp pose" phase of pick-and-place; real AC-01/AC-02 success rates need hardware
  rollouts + D2 data (grasp/contact outcomes are not simulated by MockRobot), which is why
  the acceptance harness reports the ``sim:``-prefixed task as pending_hardware for AC-01.

SAFETY LIMITS: per-joint safety arrays (q/dq/ddq) come from the ROBOT'S DECLARED envelope
(robot config / adapter — FR-06: robot specifics live in the HAL), while scalar EE /
gripper-rate / watchdog parameters come from ``--safety-config``. The demo margins in
``configs/safety/default.yaml`` (e.g. ddq 4 rad/s^2) were tuned for the smooth
DummyPolicy; the checkpoint policy replans with velocity discontinuities up to
~6 rad/s^2, inside MockRobot's declared 8 rad/s^2 envelope. Every limit the robot
declares is still enforced deterministically (FR-07) and E2 gate thresholds are unchanged.

FAULT INJECTION (``--fault-injection``, AC-06): wraps the policy so every k-th predict
returns an all-NaN chunk (safety layer must nan_reject -> HOLD) and every m-th predict
stalls past the executor deadline (chunk discarded -> hold, watchdog not fed). Rollouts
are labeled ``task="sim:fault_injection"`` — EVERY supported robot runs simulated today
("mock" = MockRobot, "g1" = G1Adapter on FakeG1Transport, "mujoco_g1" = the same G1Adapter
on MuJoCo contact physics), and the acceptance harness detects sim evidence solely via the
``sim:`` task prefix — so AC-03 excludes them and AC-06 reports pending_hardware instead of
claiming real-robot safe-stop evidence.

Exit codes: 0 = rollouts completed; 1 = usage/config error; 2 = E2 static gate failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from wam.evaluation import e2_static_checks
from wam.evaluation.acceptance import FAULT_INJECTION_TASK, SIM_TASK_PREFIX
from wam.interfaces import (
    ActionChunk,
    CanonicalSpaceSpec,
    JsonlRunLogger,
    Policy,
    RobotAdapter,
    RobotState,
    RunMetadata,
    load_config,
)
from wam.robot import MockRobot, get_robot
from wam.runtime import (
    DEFAULT_INSTRUCTION,
    ClosedLoopExecutor,
    DummyPolicy,
    ExecutorConfig,
    RolloutResult,
)
from wam.safety import SafetyConfig, SafetyLayer, Watchdog

_REPO_ROOT = Path(__file__).resolve().parent.parent

SIM_REACH_TASK = "sim:reach"

# Fixed sim:reach target pose [rad] for the 6-joint mock space (see module docstring):
# the pose the D1-overfit checkpoint reaches after ~1.0-1.4 s of closed-loop tracking.
REACH_TARGET_RAD: tuple[float, ...] = (0.04, 0.09, 0.17, 0.20, 0.25, 0.31)
DEFAULT_REACH_TOLERANCE_RAD = 0.05

# Conservative accel placeholder for robots that do not declare ddq_max (matches the
# MVP caps in configs/robot/g1.yaml — OD-08 pending vendor verification).
FALLBACK_DDQ_MAX = 4.0


class FaultInjectionPolicy:
    """Deterministic induced-failure wrapper (AC-06). Implements ``Policy``.

    Every ``nan_every``-th predict returns an all-NaN chunk (the safety layer must
    reject it: ``nan_reject`` -> HOLD chunk) and every ``stall_every``-th predict sleeps
    ``stall_s`` seconds so the executor's deadline gate trips (late chunk discarded,
    ``robot.hold()``, watchdog deliberately not fed). All other predictions pass through
    unchanged — faults are injected on top of the real policy, not instead of it.
    """

    def __init__(
        self,
        inner: Policy,
        *,
        nan_every: int = 3,
        stall_every: int = 5,
        stall_s: float = 0.0,
    ) -> None:
        if nan_every < 0 or stall_every < 0:
            raise ValueError("nan_every/stall_every must be >= 0 (0 disables)")
        if stall_s < 0:
            raise ValueError(f"stall_s must be >= 0, got {stall_s}")
        self._inner = inner
        self._nan_every = int(nan_every)
        self._stall_every = int(stall_every)
        self._stall_s = float(stall_s)
        self._calls = 0

    def predict(self, observation: Any) -> ActionChunk:
        self._calls += 1
        if self._stall_every and self._calls % self._stall_every == 0 and self._stall_s > 0:
            time.sleep(self._stall_s)  # induced stall -> executor deadline miss
        chunk = self._inner.predict(observation)
        if self._nan_every and self._calls % self._nan_every == 0:
            targets = np.full_like(np.asarray(chunk.targets, dtype=np.float32), np.nan)
            return ActionChunk(
                mode=chunk.mode,
                targets=targets,
                gripper_target=np.asarray(chunk.gripper_target, dtype=np.float32).copy(),
                dt_s=chunk.dt_s,
            )
        return chunk


def build_safety_config(
    base: SafetyConfig,
    spec: CanonicalSpaceSpec,
    robot_limits: Mapping[str, Sequence[float]],
) -> SafetyConfig:
    """Safety config for one robot: per-joint arrays from the robot's declared envelope.

    ``q_min/q_max/dq_max/ddq_max`` are taken from ``robot_limits`` when declared there;
    otherwise the base yaml arrays are kept if their length matches the canonical space
    (``ddq_max`` falls back to :data:`FALLBACK_DDQ_MAX` per joint). Scalar parameters
    (workspace AABB, EE caps, gripper rate, watchdog timeout/policy) come from ``base``.
    """
    n = spec.num_joints
    data = base.model_dump()
    for key in ("q_min", "q_max", "dq_max", "ddq_max"):
        declared = robot_limits.get(key)
        if declared is not None:
            values = tuple(float(x) for x in np.asarray(declared, dtype=np.float64))
        elif len(data[key]) == n:
            values = tuple(float(x) for x in data[key])
        elif key == "ddq_max":
            values = (FALLBACK_DDQ_MAX,) * n
        else:
            raise ValueError(
                f"safety config: no '{key}' for {n} joints — robot declares none and the "
                f"base config has {len(data[key])} entries"
            )
        if len(values) != n:
            raise ValueError(f"robot limits '{key}': expected {n} entries, got {len(values)}")
        data[key] = values
    return SafetyConfig.model_validate(data)


def make_reach_success_fn(
    target_rad: Sequence[float], tolerance_rad: float
) -> Callable[[RobotState], bool]:
    """sim:reach success predicate: mean absolute joint error below tolerance."""
    target = np.asarray(target_rad, dtype=np.float64)

    def success(state: RobotState) -> bool:
        q = np.asarray(state.q, dtype=np.float64)
        if q.shape != target.shape or not np.isfinite(q).all():
            return False
        return float(np.mean(np.abs(q - target))) < tolerance_rad

    return success


# ------------------------------------------------------------------------- construction


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", choices=("mock", "g1", "mujoco_g1"), default="mock")
    parser.add_argument(
        "--policy", choices=("checkpoint", "joint", "dummy", "remote"), default="checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "runs" / "d1-overfit-seed0" / "checkpoint.safetensors",
    )
    parser.add_argument(
        "--policy-camera",
        default=None,
        help=(
            "which Observation.images key the policy reads, overriding the trained "
            "config.camera (--policy checkpoint|joint; e.g. 'head' for the MuJoCo scene)"
        ),
    )
    parser.add_argument(
        "--policy-device", default="cpu", help="torch device for --policy checkpoint|joint"
    )
    parser.add_argument("--server-uri", default=None, help="ws://host:port for --policy remote")
    parser.add_argument("--remote-timeout-s", type=float, default=1.0)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--task", default=SIM_REACH_TASK)
    parser.add_argument(
        "--fault-injection",
        action="store_true",
        help=(
            "wrap the policy with induced faults; forces task "
            f"'{SIM_TASK_PREFIX}{FAULT_INJECTION_TASK}' (all supported robots are simulated)"
        ),
    )
    parser.add_argument("--fault-nan-every", type=int, default=3)
    parser.add_argument("--fault-stall-every", type=int, default=5)
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--policy-deadline-ms", type=float, default=500.0)
    parser.add_argument("--min-policy-rate-hz", type=float, default=2.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="robot yaml (default: configs/robot/<robot>.yaml)",
    )
    parser.add_argument(
        "--image-hw",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="override the sim render size (--robot mujoco_g1); must match the policy's backbone",
    )
    parser.add_argument(
        "--safety-config", type=Path, default=_REPO_ROOT / "configs" / "safety" / "default.yaml"
    )
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "runs" / "rollouts")
    parser.add_argument("--run-id", default=None, help="default: <robot>-<task>-<utc timestamp>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-jitter-rad", type=float, default=0.03)
    parser.add_argument("--reach-tolerance-rad", type=float, default=DEFAULT_REACH_TOLERANCE_RAD)
    parser.add_argument("--e2-probes", type=int, default=16)
    parser.add_argument("--skip-e2", action="store_true", help="skip the E2 static release gate")
    return parser.parse_args(argv)


def _build_mock(args: argparse.Namespace):
    """(spec, dt_s, robot_limits, robot_factory) for the mock robot config."""
    robot_section = load_config(args.robot_config)["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    limits: dict[str, Any] = dict(robot_section.get("limits", {}))
    dt_s = float(robot_section.get("control", {}).get("dt_s", 0.05))
    jitter_rng = np.random.default_rng(args.seed)

    kwargs: dict[str, Any] = {"spec": spec}
    for key in ("q_min", "q_max", "dq_max"):
        if key in limits:
            kwargs[key] = np.asarray(limits[key], dtype=np.float64)

    def factory(index: int, *, jitter: bool) -> RobotAdapter:
        initial_q = None
        if jitter and args.start_jitter_rad > 0:
            initial_q = jitter_rng.uniform(
                -args.start_jitter_rad, args.start_jitter_rad, spec.num_joints
            )
        return MockRobot(seed=args.seed + index, initial_q=initial_q, **kwargs)

    return spec, dt_s, limits, factory


def _build_g1(args: argparse.Namespace):
    """(spec, dt_s, robot_limits, robot_factory) for the G1 adapter on FakeG1Transport.

    Canonical space + limits come from ``--robot-config`` (the versioned
    ``configs/robot/g1.yaml``); the declared canonical space must match the adapter's
    hard-wired ``G1_SPEC`` and ``G1Config`` is built from the yaml limits — never from
    hardcoded defaults (G1Config's placeholder limits are for SDK-free construction only).
    """
    from wam.robot.g1 import G1_SPEC, G1Adapter, G1Config
    from wam.robot.g1_transport import FakeG1Transport

    robot_section = load_config(args.robot_config)["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    if spec != G1_SPEC:
        raise SystemExit(
            f"--robot-config {args.robot_config}: canonical_space does not match the G1 "
            f"adapter's G1_SPEC ({G1_SPEC.num_joints} joints, "
            f"gripper_dims={G1_SPEC.gripper_dims})"
        )
    limits_cfg: dict[str, Any] = dict(robot_section.get("limits", {}))
    config_kwargs: dict[str, Any] = {
        key: tuple(float(x) for x in limits_cfg[key])
        for key in ("q_min", "q_max", "dq_max")
        if key in limits_cfg
    }
    dt_s = robot_section.get("control", {}).get("dt_s")
    if dt_s is not None:
        config_kwargs["control_dt_s"] = float(dt_s)
    config = G1Config(**config_kwargs)
    limits: dict[str, Any] = {
        "q_min": config.q_min,
        "q_max": config.q_max,
        "dq_max": config.dq_max,
        "ddq_max": tuple(
            float(x)
            for x in limits_cfg.get("ddq_max", (FALLBACK_DDQ_MAX,) * G1_SPEC.num_joints)
        ),
    }

    def factory(index: int, *, jitter: bool) -> RobotAdapter:
        del jitter  # start-pose jitter is a MockRobot-only knob
        adapter = G1Adapter(config=config, transport=FakeG1Transport(seed=args.seed + index))
        adapter.connect()
        return adapter

    return G1_SPEC, config.control_dt_s, limits, factory


def _build_mujoco_g1(args: argparse.Namespace):
    """(spec, dt_s, robot_limits, robot_factory) for MujocoG1Robot — the E2 sim robot.

    The SAME canonical space, the SAME ``G1Adapter`` and the SAME safety chain as
    ``--robot g1``; only the transport differs (MuJoCo contact physics and rendered
    cameras instead of ``FakeG1Transport``). Limits AND gains come from ``--robot-config``
    (``configs/robot/mujoco_g1.yaml``) — the ``gains`` there are SIM gains measured on
    ``configs/sim/g1_scene.xml``, never the hardware placeholders of ``g1.yaml`` (OD-08).
    Needs the optional ``mujoco`` dependency (``uv pip install wam[sim]``) plus the fetched
    vendor model (``scripts/fetch_g1_model.py``).
    """
    from wam.robot.g1 import G1_SPEC, G1Config

    robot_section = load_config(args.robot_config)["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    if spec != G1_SPEC:
        raise SystemExit(
            f"--robot-config {args.robot_config}: canonical_space does not match the G1 "
            f"adapter's G1_SPEC ({G1_SPEC.num_joints} joints, "
            f"gripper_dims={G1_SPEC.gripper_dims})"
        )
    limits_cfg: dict[str, Any] = dict(robot_section.get("limits", {}))
    gains_cfg: dict[str, Any] = dict(robot_section.get("gains", {}))
    config_kwargs: dict[str, Any] = {
        key: tuple(float(x) for x in limits_cfg[key])
        for key in ("q_min", "q_max", "dq_max")
        if key in limits_cfg
    }
    config_kwargs.update(
        {key: tuple(float(x) for x in gains_cfg[key]) for key in ("kp", "kd") if key in gains_cfg}
    )
    dt_s = robot_section.get("control", {}).get("dt_s")
    if dt_s is not None:
        config_kwargs["control_dt_s"] = float(dt_s)
    config = G1Config(**config_kwargs)
    limits: dict[str, Any] = {
        "q_min": config.q_min,
        "q_max": config.q_max,
        "dq_max": config.dq_max,
        "ddq_max": tuple(
            float(x)
            for x in limits_cfg.get("ddq_max", (FALLBACK_DDQ_MAX,) * G1_SPEC.num_joints)
        ),
    }

    sim_cfg: dict[str, Any] = dict(robot_section.get("sim", {}))
    robot_kwargs: dict[str, Any] = {"config": config}
    if "scene" in sim_cfg:
        robot_kwargs["scene_path"] = _REPO_ROOT / str(sim_cfg["scene"])
    if "keyframe" in sim_cfg:
        robot_kwargs["keyframe"] = str(sim_cfg["keyframe"])
    if "cameras" in sim_cfg:
        robot_kwargs["cameras"] = tuple(str(c) for c in sim_cfg["cameras"])
    if "image_hw" in sim_cfg:
        robot_kwargs["image_hw"] = tuple(int(x) for x in sim_cfg["image_hw"])
    # CLI override: a trained policy dictates its own input resolution (the backbone's
    # patchifier and positional table are built for exactly one H x W), while the yaml's
    # 256x256 is what every measured number in docs/sim.md was taken at. Overriding here
    # keeps that config — and its measurements — untouched.
    if args.image_hw is not None:
        robot_kwargs["image_hw"] = tuple(args.image_hw)

    def factory(index: int, *, jitter: bool) -> RobotAdapter:
        del index, jitter  # the scene keyframe is fixed; start-pose jitter is a mock-only knob
        return get_robot("mujoco_g1", **robot_kwargs)

    return G1_SPEC, config.control_dt_s, limits, factory


def _build_policy(args: argparse.Namespace, spec: CanonicalSpaceSpec, dt_s: float) -> Policy:
    if args.policy == "checkpoint":
        from wam.runtime.policies import CheckpointPolicy  # torch import stays lazy

        return CheckpointPolicy(
            args.checkpoint, device=args.policy_device, camera=args.policy_camera
        )
    if args.policy == "joint":
        # load_joint_policy, not JointCheckpointPolicy directly: a Wan-backed T-16 checkpoint
        # carries no base weights, so it needs the frozen backbone built and strict=False.
        from wam.runtime.policies import load_joint_policy

        return load_joint_policy(
            args.checkpoint, device=args.policy_device, camera=args.policy_camera
        )
    if args.policy == "remote":
        if not args.server_uri:
            raise SystemExit("--policy remote requires --server-uri ws://host:port")
        from wam.runtime.client import RemotePolicy

        return RemotePolicy(args.server_uri, timeout_s=args.remote_timeout_s)
    # Calm sinusoid parameters: E2 probes use phase-shifted synthetic states, where the
    # DummyPolicy defaults would trip gripper-rate/accel clamps (see tests' _CALM note).
    return DummyPolicy(spec, dt_s=dt_s, amplitude_rad=0.05, period_s=8.0, gripper_period_s=60.0)


def _provenance(args: argparse.Namespace, policy: Policy) -> tuple[str | None, str | None, dict]:
    """(checkpoint_ref, dataset_snapshot_ref, extra provenance) for the run_metadata line."""
    if args.policy in ("checkpoint", "joint"):
        md = policy.metadata  # type: ignore[attr-defined]
        # Which camera key was actually read, and whether that was an override. Like every
        # other entry here this reaches the log only via config_hash, so two runs that differ
        # only in camera are distinguishable artifacts; the value itself is echoed on stdout.
        extra = {
            "checkpoint_run_id": md.run_id,
            "checkpoint_config_hash": md.config_hash,
            "policy_camera": policy.camera,  # type: ignore[attr-defined]
            "policy_camera_overridden": args.policy_camera is not None,
        }
        return (str(Path(args.checkpoint).resolve()), md.dataset_snapshot_ref, extra)
    if args.policy == "remote":
        try:
            md = dict(policy.info().get("metadata") or {})  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — provenance query must not kill the rollout run
            md = {}
        return (
            md.get("checkpoint_ref"),
            md.get("dataset_snapshot_ref"),
            {"server_uri": args.server_uri, "remote_run_id": md.get("run_id")},
        )
    return None, None, {"policy": "dummy"}


# ------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.rollouts < 0:
        raise SystemExit(f"--rollouts must be >= 0, got {args.rollouts}")

    # Both supported robots are simulated today (MockRobot / G1Adapter on FakeG1Transport),
    # so fault-injection evidence is labeled 'sim:' — the acceptance harness must report
    # AC-06 as pending_hardware, never as real-robot safe-stop evidence (AC-06 contract).
    task = (SIM_TASK_PREFIX + FAULT_INJECTION_TASK) if args.fault_injection else args.task
    if args.robot_config is None:
        args.robot_config = _REPO_ROOT / "configs" / "robot" / f"{args.robot}.yaml"
    if args.robot == "mujoco_g1":
        spec, dt_s, robot_limits, robot_factory = _build_mujoco_g1(args)
    elif args.robot == "g1":
        spec, dt_s, robot_limits, robot_factory = _build_g1(args)
    else:
        spec, dt_s, robot_limits, robot_factory = _build_mock(args)

    safety_cfg = build_safety_config(
        SafetyConfig.from_yaml(args.safety_config), spec, robot_limits
    )
    base_policy = _build_policy(args, spec, dt_s)

    exec_cfg = ExecutorConfig(
        prefix_steps=args.prefix_steps,
        max_cycles=args.max_cycles,
        policy_deadline_ms=args.policy_deadline_ms,
        min_policy_rate_hz=args.min_policy_rate_hz,
        instruction=args.instruction,
        task=task,
        stop_on_estop=True,
    )

    # sim:reach success proxy — 6-joint mock space only (see module docstring).
    success_fn: Callable[[RobotState], bool] | None = None
    if (
        not args.fault_injection
        and task == SIM_REACH_TASK
        and args.robot == "mock"
        and spec.num_joints == len(REACH_TARGET_RAD)
    ):
        success_fn = make_reach_success_fn(REACH_TARGET_RAD, args.reach_tolerance_rad)

    try:
        # ---- T-22: E2 static release gate (on the UNWRAPPED policy) ----------------------
        if not args.skip_e2:
            e2 = e2_static_checks(
                base_policy,
                robot_factory(0, jitter=False),
                SafetyLayer(safety_cfg, spec=spec),
                spec,
                n_probes=args.e2_probes,
                seed=args.seed,
                max_mean_latency_ms=args.policy_deadline_ms,
                instruction=args.instruction,
            )
            status = "PASS" if e2.passed else f"FAIL ({', '.join(e2.failed_gates())})"
            print(f"[e2] static checks: {status} ({e2.n} probes)")
            for warning in e2.warnings:
                print(f"[e2] warning: {warning}")
            if not e2.passed:
                for gate in e2.gates:
                    if not gate.passed:
                        print(f"[e2] failed gate {gate.name}: {gate.detail}")
                return 2

        # ---- provenance + logger (AC-04) --------------------------------------------------
        checkpoint_ref, dataset_ref, provenance_extra = _provenance(args, base_policy)
        run_id = args.run_id or (
            f"{args.robot}-{task.replace(':', '-')}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        config_record: dict[str, Any] = {
            "robot": args.robot,
            "policy": args.policy,
            "task": task,
            "rollouts": args.rollouts,
            "seed": args.seed,
            "executor": exec_cfg.model_dump(),
            "safety": safety_cfg.model_dump(),
            "start_jitter_rad": args.start_jitter_rad,
            "provenance": provenance_extra,
        }
        if success_fn is not None:
            config_record["sim_reach"] = {
                "target_rad": list(REACH_TARGET_RAD),
                "tolerance_rad": args.reach_tolerance_rad,
            }
        if args.fault_injection:
            config_record["fault_injection"] = {
                "nan_every": args.fault_nan_every,
                "stall_every": args.fault_stall_every,
            }
        metadata = RunMetadata.create(
            run_id,
            config_record,
            checkpoint_ref=checkpoint_ref,
            dataset_snapshot_ref=dataset_ref,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.out_dir / f"{run_id}.jsonl"

        # ---- E3-style rollouts -------------------------------------------------------------
        stall_s = 1.5 * args.policy_deadline_ms / 1e3
        results: list[RolloutResult] = []
        with JsonlRunLogger(log_path, metadata) as logger:
            logger.log_metadata()
            for i in range(args.rollouts):
                policy: Policy = base_policy
                if args.fault_injection:
                    policy = FaultInjectionPolicy(  # fresh wrapper: deterministic per rollout
                        base_policy,
                        nan_every=args.fault_nan_every,
                        stall_every=args.fault_stall_every,
                        stall_s=stall_s,
                    )
                executor = ClosedLoopExecutor(
                    robot=robot_factory(i, jitter=True),
                    policy=policy,
                    safety=SafetyLayer(safety_cfg, spec=spec),
                    watchdog=Watchdog.from_config(safety_cfg),
                    logger=logger,
                    config=exec_cfg,
                )
                results.append(executor.run_rollout(f"{run_id}-{i:04d}", success_fn=success_fn))

        # ---- summary -----------------------------------------------------------------------
        kinds: dict[str, int] = {}
        for r in results:
            for kind, count in r.intervention_kinds.items():
                kinds[kind] = kinds.get(kind, 0) + count
        n_success = sum(1 for r in results if r.success)
        n_estop = sum(1 for r in results if r.estopped)
        wd = sum(r.watchdog_timeouts for r in results)
        misses = sum(r.deadline_misses for r in results)
        rates = [r.policy_rate_hz for r in results]
        if results:
            # The camera is echoed because it is the one policy input whose being wrong is
            # invisible everywhere else: the chunk stays finite, in-bounds and on time. It
            # reaches the log only through config_hash, which nothing reads by eye.
            camera = getattr(base_policy, "camera", None)
            print(
                f"[rollout] task={task} robot={args.robot} policy={args.policy} "
                + (f"camera={camera} " if camera else "")
                + f"n={len(results)} success={n_success} estops={n_estop} "
                f"watchdog_timeouts={wd} deadline_misses={misses} "
                f"interventions={kinds} min_rate_hz={min(rates):.1f}"
            )
        else:
            print("[rollout] no rollouts requested")
        print(f"[log] {log_path}")
        return 0
    finally:
        close = getattr(base_policy, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
