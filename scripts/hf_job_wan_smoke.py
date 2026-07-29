# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "diffusers>=0.35.0",
#   "transformers>=4.51.0",
#   "accelerate",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "ftfy",
#   "numpy",
#   "pydantic>=2",
#   "pyyaml",
#   "pyarrow",
#   "opencv-python-headless",
# ]
# ///
"""Wan backbone smoke test — runs on an HF Jobs GPU (T-15, OD-04/OD-05).

Answers one question: does the real Wan DiT produce usable action-readout features through
the WAM interfaces, with the shapes ``WanI2VAdapter`` claims? Nothing is trained here.

Checks: load -> geometry -> condition_video/text/state -> features -> determinism ->
ActionHead.decode -> peak VRAM / wall time. Writes a JSON report and exits non-zero on the
first failed assertion.

Run it on HF Jobs (see docs/hf_jobs.md); locally with a GPU it works too:

    uv run scripts/hf_job_wan_smoke.py --source /model --device cuda
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# The job mounts this repo's `src/` — make `wam` importable before anything else.
for candidate in ("/wam-src", str(Path(__file__).resolve().parents[1] / "src")):
    if Path(candidate).is_dir() and candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np
import torch

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_INSTRUCTION = "pick up the red cube and place it in the bin"

# Last report payload, so an in-process caller (the ZeroGPU Space in
# deploy/wan-smoke-space/) can read the result without parsing stdout. CLI runs ignore it.
LAST_REPORT: dict[str, Any] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_argument_group("model source")
    src.add_argument("--source", default=None, help="local snapshot dir (e.g. mounted /model)")
    src.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="hub repo id (needs --download)")
    src.add_argument("--download", action="store_true", help="allow fetching from the Hub")
    src.add_argument("--dtype", default="bfloat16")
    src.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    src.add_argument("--blocks", default="", help="comma-separated readout blocks (default auto)")
    src.add_argument("--timestep", type=int, default=0)
    src.add_argument(
        "--device-map",
        default=None,
        help="accelerate device_map (e.g. 'cuda'): stream shards to the GPU instead of "
        "materializing the model in host RAM — required where RAM < checkpoint size",
    )
    src.add_argument(
        "--offload-text",
        action="store_true",
        help="move the umT5 tower to CPU after encoding (peak-VRAM relief on 24 GB cards)",
    )

    inp = p.add_argument_group("input")
    inp.add_argument("--episode", default=None, help="episode dir; synthetic frames if omitted")
    inp.add_argument("--camera", default="front")
    inp.add_argument("--frames", type=int, default=5)
    inp.add_argument("--height", type=int, default=256)
    inp.add_argument("--width", type=int, default=448)
    inp.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    out = p.add_argument_group("output")
    out.add_argument("--out", default="wan_smoke_report.json")
    out.add_argument("--action-steps", type=int, default=16)
    out.add_argument("--target-dim", type=int, default=7, help="canonical action target dim")
    out.add_argument(
        "--ablate",
        action="store_true",
        help="after the checks, probe EVERY DiT block for motion/instruction/state "
        "sensitivity (4 extra forwards, one load) and rank readout candidates",
    )
    return p.parse_args(argv)


class Report:
    """Collects check results; any failure makes the job exit non-zero."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return bool(ok)

    def expect(self, name: str, actual: Any, expected: Any) -> bool:
        return self.check(name, actual == expected, f"got {actual}, expected {expected}")

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["ok"]]


def synthetic_frames(num_frames: int, height: int, width: int) -> np.ndarray:
    """Deterministic moving-square clip [F, H, W, 3] uint8 — a stand-in for a wrist camera."""
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    box = max(8, min(height, width) // 6)
    for i in range(num_frames):
        frames[i, :, :, 2] = 40  # dim blue background
        y = int((height - box) * (0.2 + 0.6 * i / max(num_frames - 1, 1)))
        x = int((width - box) * (0.7 - 0.5 * i / max(num_frames - 1, 1)))
        frames[i, y : y + box, x : x + box] = (220, 40, 40)  # the "cube"
    return frames


def load_frames(args: argparse.Namespace, report: Report) -> np.ndarray:
    """Episode frames if a dataset is mounted, else synthetic ones; resized to H x W."""
    if args.episode:
        from wam.data.episode import EpisodeReader

        reader = EpisodeReader(args.episode)
        frames = reader.read_frames(args.camera)[: args.frames]
        report.info["frame_source"] = f"episode:{args.episode}:{args.camera}"
    else:
        frames = synthetic_frames(args.frames, args.height, args.width)
        report.info["frame_source"] = "synthetic"
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    if tensor.shape[-2:] != (args.height, args.width):
        tensor = torch.nn.functional.interpolate(
            tensor, size=(args.height, args.width), mode="bilinear", align_corners=False
        )
    resized = tensor.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).numpy()
    return resized[None]  # [1, F, H, W, 3]


def synthetic_state(fill_q: float = 0.0, fill_dq: float = 0.0) -> Any:
    """A canonical RobotState with constant joint values — the smoke/ablation probe input."""
    from wam.interfaces.schema import IMUState, RobotState, ValidityMask

    num_joints = 7
    return RobotState(
        timestamp_ns=0,
        q=np.full(num_joints, fill_q, dtype=np.float32),
        dq=np.full(num_joints, fill_dq, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
        validity=ValidityMask(),
    )


def state_embedding(args: argparse.Namespace, report: Report) -> torch.Tensor:
    """Real StateMLP embedding from the episode's first state, else from a synthetic state."""
    from wam.encoders.state_mlp import StateMLP, StateMLPConfig

    if args.episode:
        from wam.data.episode import EpisodeReader

        state = EpisodeReader(args.episode).read_states()[0]
    else:
        state = synthetic_state()
    torch.manual_seed(0)
    # Both dims come from the state, never from StateMLPConfig's defaults: a real G1 episode has
    # gripper_dims=2 (one per hand), and the default 1 made StateMLP.encode reject the very episode
    # this check exists to accept.
    encoder = StateMLP(
        StateMLPConfig(
            embedding_dim=32,
            num_joints=len(state.q),
            gripper_dims=len(state.gripper_state),
        )
    )
    report.info["state_dim"] = len(state.q)
    report.info["gripper_dims"] = len(state.gripper_state)
    return encoder.encode(state)


ABLATE_INSTRUCTION = "push the blue block slowly to the left edge of the table"


def relative_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / max(||a||, ||b||) — scale-free across blocks with different norms."""
    return float((a - b).norm()) / max(float(a.norm()), float(b.norm()), 1e-6)


def ablation_ranking(
    base: dict[int, torch.Tensor], probes: dict[str, dict[int, torch.Tensor]]
) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    """Per-block probe distances + a combined score (each probe min-max normalized, averaged)."""
    blocks = sorted(base)
    per_block = {
        b: {name: relative_distance(base[b], variant[b]) for name, variant in probes.items()}
        for b in blocks
    }
    scores = dict.fromkeys(blocks, 0.0)
    for name in probes:
        column = {b: per_block[b][name] for b in blocks}
        lo, hi = min(column.values()), max(column.values())
        span = (hi - lo) or 1.0
        for b in blocks:
            scores[b] += (column[b] - lo) / span / len(probes)
    return per_block, scores


def run_ablation(adapter: Any, args: argparse.Namespace, report: Report) -> None:
    """Label-free readout ablation (docs/hf_jobs.md "next steps" #1).

    Which residual-stream depth reacts most to the three inputs the action head must read —
    scene motion, the instruction, the robot state? Per block: relative distance between a
    base forward and one with exactly that input changed. One load, four forwards. This is a
    proxy ranking; the final verdict needs linear probes on real D2 action labels.
    """
    from wam.encoders.state_mlp import StateMLP, StateMLPConfig

    started = time.perf_counter()
    geometry = adapter.describe()
    blocks = tuple(range(geometry["num_layers"]))

    frames = synthetic_frames(args.frames, args.height, args.width)[None]
    video_base = adapter.condition_video(frames)
    video_motion = adapter.condition_video(frames[:, ::-1])  # same frames, reversed trajectory
    text_base = adapter.condition_text(args.instruction)
    text_alt = adapter.condition_text(ABLATE_INSTRUCTION)
    torch.manual_seed(0)
    probe_state = synthetic_state()
    encoder = StateMLP(
        StateMLPConfig(
            embedding_dim=32,
            num_joints=len(probe_state.q),
            gripper_dims=len(probe_state.gripper_state),
        )
    )
    state_base = adapter.condition_state(encoder.encode(probe_state))
    state_alt = adapter.condition_state(encoder.encode(synthetic_state(fill_q=0.5, fill_dq=0.3)))

    def forward(video_ctx: Any, text_ctx: Any, state_ctx: Any) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            captured = adapter.features_by_block(video_ctx, text_ctx, state_ctx, blocks=blocks)
        return {b: t.float().cpu() for b, t in captured.items()}

    base = forward(video_base, text_base, state_base)
    probes = {
        "motion": forward(video_motion, text_base, state_base),
        "instruction": forward(video_base, text_alt, state_base),
        "state": forward(video_base, text_base, state_alt),
    }
    per_block, scores = ablation_ranking(base, probes)

    report.check("ablation.captured_all_blocks", len(base) == len(blocks), f"{len(base)} blocks")
    values = [row[name] for row in per_block.values() for name in probes]
    report.check("ablation.finite", bool(np.isfinite(values).all()), f"{len(values)} distances")
    for name in probes:  # a probe nobody reacts to means that conditioning path is dead
        peak = max(per_block[b][name] for b in blocks)
        report.check(f"ablation.{name}_moves_features", peak > 1e-6, f"peak distance {peak:.4f}")

    ranked = sorted(blocks, key=lambda b: scores[b], reverse=True)
    suggested = sorted(ranked[:2])
    default = tuple(geometry["feature_blocks"])
    print("\nblock  motion  instruct  state   score")
    for b in blocks:
        row = per_block[b]
        marker = "  <- suggested" if b in suggested else ("  <- default" if b in default else "")
        print(
            f"{b:5d}  {row['motion']:.4f}  {row['instruction']:.4f}  "
            f"{row['state']:.4f}  {scores[b]:.3f}{marker}",
            flush=True,
        )
    report.info["ablation"] = {
        "metric": "relative L2 distance to the base forward, per residual-stream block",
        "probes": {
            "motion": "frame order reversed",
            "instruction": f"{args.instruction!r} -> {ABLATE_INSTRUCTION!r}",
            "state": "StateMLP embedding of q=0.5/dq=0.3 instead of zeros",
        },
        "per_block": {str(b): per_block[b] for b in blocks},
        "scores": {str(b): round(scores[b], 4) for b in blocks},
        "suggested_blocks": list(suggested),
        "default_blocks": list(default),
        "default_scores": {str(b): round(scores[b], 4) for b in default},
        "wall_s": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report()
    report.info.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "args": vars(args),
        }
    )

    import wam
    from wam.backbones.registry import get_backbone
    from wam.decoders.action_head import ActionHead, ActionHeadConfig

    report.info["wam_version"] = getattr(wam, "__version__", "unknown")

    blocks = tuple(int(b) for b in args.blocks.split(",") if b.strip()) or None
    adapter = get_backbone(
        "wan_i2v",
        checkpoint_path=args.source,
        model_id=None if args.source else args.model_id,
        feature_blocks=blocks,
        device=args.device,
        dtype=args.dtype,
        timestep=args.timestep,
        allow_download=args.download,
        device_map=args.device_map,
    )

    t0 = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - t0
    report.check("load", adapter.is_loaded, f"{load_s:.1f}s")
    geometry = adapter.describe()
    report.info["geometry"] = geometry
    print(json.dumps(geometry, indent=2), flush=True)
    report.check(
        "readout_blocks_in_range",
        all(b < geometry["num_layers"] for b in geometry["feature_blocks"]),
        f"{geometry['feature_blocks']} of {geometry['num_layers']} blocks",
    )

    video = load_frames(args, report)
    t0 = time.perf_counter()
    video_ctx = adapter.condition_video(video)
    latents = video_ctx["latents"]
    encode_s = time.perf_counter() - t0
    stride_t, stride_s = geometry["vae_temporal_stride"], geometry["vae_spatial_stride"]
    expected_latents = (
        1,
        geometry["latent_channels"],
        1 + (args.frames - 1) // stride_t,
        args.height // stride_s,
        args.width // stride_s,
    )
    report.expect("condition_video.shape", tuple(latents.shape), expected_latents)
    report.check("condition_video.finite", bool(torch.isfinite(latents).all()), f"{encode_s:.2f}s")
    report.check(
        "condition_video.image_embeds",
        (video_ctx["image_embeds"] is not None) == bool(geometry["image_dim"]),
        "CLIP tower" if geometry["image_dim"] else "no CLIP tower (TI2V-style)",
    )

    text_ctx = adapter.condition_text(args.instruction)
    report.expect(
        "condition_text.shape",
        tuple(text_ctx.shape),
        (1, adapter.max_text_tokens, geometry["text_dim"]),
    )
    if args.offload_text:
        adapter.offload("text_encoder")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    state_ctx = adapter.condition_state(state_embedding(args, report))
    report.expect("condition_state.shape", tuple(state_ctx.shape), (1, 1, geometry["text_dim"]))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        features = adapter.features(video_ctx, text_ctx, state_ctx)
    forward_s = time.perf_counter() - t0
    expected_tokens = adapter.expected_token_count(args.frames, args.height, args.width)
    report.expect(
        "features.shape",
        tuple(features.shape),
        (1, expected_tokens, geometry["feature_dim"]),
    )
    report.check("features.finite", bool(torch.isfinite(features).all()), f"{forward_s:.2f}s")
    report.check(
        "features.nonconstant",
        float(features.float().std()) > 1e-4,
        f"std={float(features.float().std()):.4f}",
    )

    with torch.no_grad():
        again = adapter.features(video_ctx, text_ctx, state_ctx)
    report.check("features.deterministic", bool(torch.equal(features, again)), "two forwards")

    head = ActionHead(
        ActionHeadConfig(
            feature_dim=geometry["feature_dim"],
            num_steps=args.action_steps,
            target_dim=args.target_dim,
        )
    ).to(features.device)
    chunk = head.decode(features.float())
    report.expect("action_head.steps", len(chunk.targets), args.action_steps)
    report.check(
        "action_head.finite",
        bool(np.isfinite(np.asarray(chunk.targets, dtype=np.float32)).all()),
        f"mode={chunk.mode}, dt={chunk.dt_s}s",
    )

    timings = {"load_s": load_s, "vae_encode_s": encode_s, "dit_forward_s": forward_s}
    if torch.cuda.is_available():
        timings["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        timings["reserved_vram_gb"] = torch.cuda.max_memory_reserved() / 1e9
    report.info["timings"] = timings
    print(json.dumps(timings, indent=2), flush=True)

    if args.ablate and not report.failed:
        run_ablation(adapter, args, report)

    payload = {"ok": not report.failed, "checks": report.checks, "info": report.info}
    LAST_REPORT.clear()
    LAST_REPORT.update(payload)
    serialized = json.dumps(payload, indent=2, default=str)
    print(f"\n===== REPORT =====\n{serialized}", flush=True)  # the log is the durable artifact
    try:
        Path(args.out).write_text(serialized)
        print(f"report -> {args.out}", flush=True)
    except OSError as err:  # a missing/read-only mount must not sink a passing run
        print(f"could not write {args.out}: {err}", flush=True)
    if report.failed:
        print(f"FAILED: {', '.join(report.failed)}", flush=True)
        return 1
    print(f"ALL {len(report.checks)} CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
