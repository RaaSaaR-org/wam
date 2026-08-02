#!/usr/bin/env python3
"""Watch the MuJoCo G1 sim run the REAL closed loop, live (T-25).

Same chain as ``scripts/rollout.py --robot mujoco_g1``: ``ClosedLoopExecutor`` ->
``SafetyLayer`` + ``Watchdog`` -> ``G1Adapter`` -> ``MujocoG1Transport``. This script only
adds an interactive viewer window; it never bypasses the safety layer and never drives
``data.ctrl`` itself. Pacing is wall-clock by default, so what you see runs at the rate the
robot would.

macOS: launch through ``scripts/view_sim.sh``, which wraps ``mjpython`` (the interactive
viewer needs the main thread's native event loop) and exports the ``libpython`` directory
that ``mjpython``'s dlopen cannot find under a uv-managed CPython::

    scripts/view_sim.sh --amplitude-rad 0.2

Calling ``.venv/bin/mjpython scripts/view_sim.py`` directly works only if
``DYLD_FALLBACK_LIBRARY_PATH`` already contains that directory; see the wrapper.

The executor runs on a worker thread while the main thread syncs the viewer, so every
``MjData`` read here holds ``MujocoG1Transport.lock`` (the same lock that makes a
cross-thread ``estop()`` safe).

Needs the optional ``mujoco`` dependency (``uv pip install mujoco``) and the fetched vendor
model (``scripts/fetch_g1_model.py``).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from rollout import build_safety_config

from wam.interfaces import CanonicalSpaceSpec, JsonlRunLogger, RunMetadata, load_config
from wam.robot import get_robot
from wam.robot.g1 import G1_SPEC, G1Config
from wam.runtime import ClosedLoopExecutor, DummyPolicy, ExecutorConfig
from wam.safety import SafetyConfig, SafetyLayer, Watchdog

FALLBACK_DDQ_MAX = 4.0
VIEWER_FPS = 60.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--robot-config", type=Path, default=_REPO_ROOT / "configs/robot/mujoco_g1.yaml")
    p.add_argument("--safety-config", type=Path, default=_REPO_ROOT / "configs/safety/default.yaml")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "runs" / "view")
    p.add_argument("--max-cycles", type=int, default=400, help="control cycles before the run ends")
    p.add_argument("--prefix-steps", type=int, default=5, help="chunk steps executed per cycle (FR-05)")
    p.add_argument("--amplitude-rad", type=float, default=0.05, help="DummyPolicy sinusoid amplitude")
    p.add_argument("--period-s", type=float, default=8.0, help="DummyPolicy sinusoid period")
    p.add_argument(
        "--fast",
        action="store_true",
        help="pace on sim time instead of the wall clock (runs as fast as the machine allows)",
    )
    return p.parse_args(argv)


def _build_robot(robot_config: Path, *, realtime: bool) -> tuple[Any, G1Config, dict[str, Any]]:
    """Robot + config + safety limits from the versioned sim config (same fields as rollout.py)."""
    section = load_config(robot_config)["robot"]
    spec = CanonicalSpaceSpec(**section["canonical_space"])
    if spec != G1_SPEC:
        raise SystemExit(f"{robot_config}: canonical_space does not match the G1 adapter's G1_SPEC")

    limits_cfg = dict(section.get("limits", {}))
    gains_cfg = dict(section.get("gains", {}))
    kwargs: dict[str, Any] = {
        key: tuple(float(x) for x in limits_cfg[key])
        for key in ("q_min", "q_max", "dq_max")
        if key in limits_cfg
    }
    kwargs.update({k: tuple(float(x) for x in gains_cfg[k]) for k in ("kp", "kd") if k in gains_cfg})
    control_cfg = section.get("control", {})
    dt_s = control_cfg.get("dt_s")
    if dt_s is not None:
        kwargs["control_dt_s"] = float(dt_s)
    # Bounded feed-forward (T-25c). Must be read here as well as in rollout.py: this script's
    # whole claim is that it drives the SAME chain, and dropping the window would silently run
    # the viewer on the pre-T-25c control law while a rollout from the same config uses the new
    # one — 0.44 vs 0.96 of a commanded travel, and 3.3x the accel_limit interventions.
    window = control_cfg.get("q_track_window")
    if window is not None:
        kwargs["q_track_window"] = tuple(float(x) for x in window)
    config = G1Config(**kwargs)

    sim_cfg = dict(section.get("sim", {}))
    robot_kwargs: dict[str, Any] = {"config": config}
    if "scene" in sim_cfg:
        robot_kwargs["scene_path"] = _REPO_ROOT / str(sim_cfg["scene"])
    if "keyframe" in sim_cfg:
        robot_kwargs["keyframe"] = str(sim_cfg["keyframe"])
    if "cameras" in sim_cfg:
        robot_kwargs["cameras"] = tuple(str(c) for c in sim_cfg["cameras"])
    if realtime:
        # Wall-clock pacing: G1Adapter.execute() spaces commands control_dt_s apart, which is
        # what makes its dq_max*dt clip a real velocity limit. Sim-time pacing steps physics
        # instead of blocking, so the window would fast-forward.
        robot_kwargs["clock"] = time.monotonic
        robot_kwargs["sleep"] = time.sleep

    robot = get_robot("mujoco_g1", **robot_kwargs)
    limits = {
        "q_min": config.q_min,
        "q_max": config.q_max,
        "dq_max": config.dq_max,
        "ddq_max": tuple(
            float(x) for x in limits_cfg.get("ddq_max", (FALLBACK_DDQ_MAX,) * G1_SPEC.num_joints)
        ),
    }
    return robot, config, limits


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        import mujoco.viewer
    except ImportError:
        raise SystemExit(
            "MuJoCo simulation support requires the 'mujoco' package — "
            "install it with 'uv pip install mujoco' from the repo root"
        ) from None

    robot, config, limits = _build_robot(args.robot_config, realtime=not args.fast)
    safety_cfg = build_safety_config(SafetyConfig.from_yaml(args.safety_config), G1_SPEC, limits)
    policy = DummyPolicy(
        G1_SPEC,
        dt_s=config.control_dt_s,
        amplitude_rad=args.amplitude_rad,
        period_s=args.period_s,
        gripper_period_s=60.0,
    )
    # Same rollout-log contract as scripts/rollout.py, so a viewed run stays traceable
    # (AC-04): run_metadata + one control_cycle per cycle + one rollout_summary.
    run_id = f"view-{time.strftime('%Y%m%d-%H%M%S')}"
    metadata = RunMetadata.create(
        run_id,
        {
            "robot": "mujoco_g1",
            "robot_config": str(args.robot_config),
            "safety_config": str(args.safety_config),
            "policy": "dummy",
            "amplitude_rad": args.amplitude_rad,
            "period_s": args.period_s,
            "prefix_steps": args.prefix_steps,
            "pacing": "sim_time" if args.fast else "realtime",
        },
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{run_id}.jsonl"

    transport = robot.transport
    result: dict[str, Any] = {}
    done = threading.Event()

    print(
        f"[view] {'sim-time' if args.fast else 'realtime'} pacing, "
        f"{args.max_cycles} cycles, prefix={args.prefix_steps} — close the window to stop"
    )
    with JsonlRunLogger(log_path, metadata) as logger:
        logger.log_metadata()
        executor = ClosedLoopExecutor(
            robot=robot,
            policy=policy,
            safety=SafetyLayer(safety_cfg, spec=G1_SPEC),
            watchdog=Watchdog.from_config(safety_cfg),
            logger=logger,
            config=ExecutorConfig(
                prefix_steps=args.prefix_steps,
                max_cycles=args.max_cycles,
                task="sim:view",
                stop_on_estop=True,
            ),
        )

        def run() -> None:
            try:
                result["rollout"] = executor.run_rollout(f"{run_id}-0000")
            except Exception as exc:  # noqa: BLE001 — re-raised once the window is closed
                result["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=run, name="closed-loop", daemon=True)
        with mujoco.viewer.launch_passive(transport.model, transport.data) as viewer:
            worker.start()
            while viewer.is_running() and not done.is_set():
                with transport.lock:  # MjData is being stepped by the worker thread
                    viewer.sync()
                time.sleep(1.0 / VIEWER_FPS)
        done.wait(timeout=5.0)

    if (exc := result.get("error")) is not None:
        raise exc
    rollout = result.get("rollout")
    if rollout is None:
        print(f"[view] window closed before the rollout finished — log: {log_path}")
        return 0
    print(
        f"[view] cycles={rollout.executed_cycles}/{rollout.cycles} "
        f"rate={rollout.policy_rate_hz:.1f} Hz "
        f"estop={rollout.estopped} watchdog_timeouts={rollout.watchdog_timeouts} "
        f"deadline_misses={rollout.deadline_misses} interventions={rollout.intervention_kinds}"
    )
    print(f"[log] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
