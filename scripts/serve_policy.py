#!/usr/bin/env python3
"""Serve a WAM policy over WebSocket (T-20).

Usage:
  .venv/bin/python scripts/serve_policy.py --dummy                       # DummyPolicy
  .venv/bin/python scripts/serve_policy.py --checkpoint runs/d1-overfit-seed0/checkpoint.safetensors
  .venv/bin/python scripts/serve_policy.py --joint \
      --checkpoint runs/t18-real-ablation-seed0/checkpoint.safetensors --device cuda

Runs the inference side only — safety layer, watchdog and robot control stay on the
client (see wam.runtime.server). Prints ``serving ws://host:port`` once bound and runs
until SIGINT (Ctrl-C), then shuts down gracefully with exit code 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wam.interfaces import CanonicalSpaceSpec, load_config
from wam.interfaces.protocols import Policy
from wam.runtime.offload import OFFLOAD_TEXT_HELP, advise_alloc_conf, offload_text_encoder
from wam.runtime.server import PolicyServer

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", type=Path, default=None, help="serve a trained checkpoint (.safetensors)"
    )
    source.add_argument(
        "--dummy",
        action="store_true",
        help="serve the deterministic DummyPolicy (spec from --robot-config)",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "robot" / "mock.yaml",
        help="robot config providing the canonical space for --dummy",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="--checkpoint is a JointWorldActionModel (world-action, T-16), not action-only",
    )
    parser.add_argument("--device", default="cpu", help="torch device for --checkpoint")
    parser.add_argument(
        "--policy-camera",
        default=None,
        help="Observation.images key to read, overriding the trained config.camera (--joint only)",
    )
    parser.add_argument(
        "--backbone-source",
        default=None,
        help="local dir holding the frozen base weights (--joint); needed when the checkpoint "
        "was trained on another machine, whose recorded path does not exist here",
    )
    parser.add_argument(
        "--offload-text",
        action="store_true",
        help=OFFLOAD_TEXT_HELP + " (--joint only). A served policy encodes the instruction "
        "once per distinct prompt and then reads the memoized result, so the tower is dead "
        "weight on the GPU for the rest of the session.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="0 = OS-assigned port")
    args = parser.parse_args(argv)
    if args.joint and args.checkpoint is None:
        parser.error("--joint requires --checkpoint")
    if args.policy_camera is not None and not args.joint:
        parser.error("--policy-camera applies to --joint checkpoints only")
    if args.offload_text and not args.joint:
        # Refused, not ignored: only the Wan backbone behind --joint has a separate umT5 tower
        # to move, so accepting this anywhere else would promise memory relief it cannot give.
        parser.error("--offload-text applies to --joint checkpoints only")
    return args


def _build_dummy_policy(robot_config: Path) -> Policy:
    from wam.runtime.mock_loop import DummyPolicy

    robot_cfg = load_config(robot_config)
    robot_section = robot_cfg["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    dt_s = float(robot_section.get("control", {}).get("dt_s", 0.05))
    return DummyPolicy(spec, dt_s=dt_s)


def _build_checkpoint_policy(path: Path) -> Policy:
    try:  # preferred: the closed-loop runtime's checkpoint policy (T-19)
        from wam.runtime import CheckpointPolicy  # type: ignore[attr-defined]

        return CheckpointPolicy(path)
    except (ImportError, AttributeError):
        # fallback: load the action-only model directly (it implements Policy)
        from wam.training import load_action_only_checkpoint

        model, metadata = load_action_only_checkpoint(path)
        print(f"loaded checkpoint run_id={metadata.run_id} config_hash={metadata.config_hash}")
        return model


def _build_joint_policy(
    path: Path,
    device: str,
    camera: str | None,
    backbone_source: str | None = None,
    offload_text: bool = False,
) -> Policy:
    """Serve a world-action checkpoint (T-16). No fallback: unlike the action-only model,
    ``JointWorldActionModel`` needs ``JointCheckpointPolicy`` to reach ``predict``.

    Via ``load_joint_policy`` rather than that class directly, because a Wan-backed checkpoint
    holds no base weights — they have to be built and loaded alongside it, from
    ``backbone_source`` when this is not the machine that trained them."""
    from wam.runtime.policies import load_joint_policy

    advise_alloc_conf(device)
    policy = load_joint_policy(
        path,
        device=device,
        camera=camera,
        backbone_source=backbone_source,
        # The load-time half: without it the umT5 tower is resident for the whole load, and the
        # offload below cannot run until that has already happened.
        cpu_pinned=("text_encoder",) if offload_text else (),
    )
    if offload_text:
        # After the loader, which is where the .to(device) that WanFlowBackbone._apply forwards
        # to the held towers happens; an offload before it would be silently undone. With the pin
        # above this is a no-op move, kept for the loud non-Wan refusal.
        offload_text_encoder(policy, log=print)
    md = policy.metadata
    print(
        f"loaded world-action checkpoint run_id={md.run_id} config_hash={md.config_hash} "
        f"camera={policy.camera!r} device={device}"
    )
    return policy


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.checkpoint is not None and args.joint:
        policy: Policy = _build_joint_policy(
            args.checkpoint,
            args.device,
            args.policy_camera,
            args.backbone_source,
            args.offload_text,
        )
    elif args.checkpoint is not None:
        policy = _build_checkpoint_policy(args.checkpoint)
    else:
        policy = _build_dummy_policy(args.robot_config)

    server = PolicyServer(policy, host=args.host, port=args.port)
    thread, actual_port = server.run_in_thread()
    print(f"serving ws://{args.host}:{actual_port}", flush=True)
    print(f"policy={type(policy).__name__} (Ctrl-C to stop)", flush=True)
    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.stop()
        thread.join(timeout=10.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
