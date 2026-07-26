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


def state_embedding(args: argparse.Namespace, report: Report) -> torch.Tensor:
    """Real StateMLP embedding from the episode's first state, else from a synthetic state."""
    from wam.encoders.state_mlp import StateMLP, StateMLPConfig
    from wam.interfaces.schema import IMUState, RobotState, ValidityMask

    if args.episode:
        from wam.data.episode import EpisodeReader

        state = EpisodeReader(args.episode).read_states()[0]
    else:
        num_joints = 7
        state = RobotState(
            timestamp_ns=0,
            q=np.zeros(num_joints, dtype=np.float32),
            dq=np.zeros(num_joints, dtype=np.float32),
            imu=IMUState(
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                linear_acceleration=np.zeros(3, dtype=np.float32),
            ),
            gripper_state=np.zeros(1, dtype=np.float32),
            validity=ValidityMask(),
        )
    torch.manual_seed(0)
    encoder = StateMLP(StateMLPConfig(embedding_dim=32, num_joints=len(state.q)))
    report.info["state_dim"] = len(state.q)
    return encoder.encode(state)


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
