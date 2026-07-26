# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "diffusers>=0.39.0",
#   "transformers>=4.51.0",
#   "accelerate",
#   "safetensors",
#   "numpy",
#   "pydantic>=2",
#   "pyyaml",
#   "pyarrow",
#   "opencv-python-headless",
#   "imageio",
#   "imageio-ffmpeg",
#   "huggingface_hub>=0.34",
# ]
# ///
"""Cosmos3-Nano frozen-feature probe on real robot data (T-24, backbone bake-off vs. Wan).

Same experiment as ``hf_job_wan_probe.py`` — identical windows, labels, split and ridge
machinery (imported from that script, one implementation) — with the feature extractor
swapped for NVIDIA Cosmos3-Nano's generator tower (diffusers ``Cosmos3OmniTransformer``).
Real GR00T-G1 ego windows are VAE-encoded (the Cosmos3 VAE *is* the Wan2.2 VAE) and packed
as clean conditioning frames after the tokenized instruction; one joint MoT forward per
window, hooks on all 36 layers collect the generation-pathway (vision-token) residual
stream, token-pooled per block, ridge-regressed onto the BC action chunks.

Why: the Wan probe's central finding was that no frozen Wan features beat a state-only
ridge. Cosmos3 is pretrained on robot video *with actions* — if any frozen video prior
linearly encodes next-chunk actions, it should be this one. Beating state-only here makes
Cosmos3 the primary backbone candidate; failing keeps the burden on LoRA fine-tuning.

Block pairs under test mirror the Wan roles, scaled from 30 to 36 layers: ``measured`` =
(2, 12), the Wan label-validated early-block pick transferred by depth fraction;
``heuristic`` = (18, 26), the mid+late depth heuristic.

Usage (HF Jobs / local GPU; the ZeroGPU Space wraps the same functions):

    uv run scripts/hf_job_cosmos3_probe.py --source /model --data-dir data/raw/gr00t_apple
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Side-by-side layouts: scripts/ locally and on HF Jobs, flat on the Space.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for candidate in ("/wam-src", str(_HERE.parent / "src")):
    if Path(candidate).is_dir() and candidate not in sys.path:
        sys.path.insert(0, candidate)

try:  # deployed side-by-side as probe.py on the Space
    import probe as wanprobe
except ImportError:
    import hf_job_wan_probe as wanprobe

smoke = wanprobe.smoke
DEFAULT_MODEL_ID = "nvidia/Cosmos3-Nano"
# The recommended Cosmos quality negative prompt (NVIDIA docs); generation only. The Wan
# negative is Wan-specific, so each backbone samples with its own recommended settings —
# the *positive* prompts are what the side-by-side keeps identical.
COSMOS_NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated "
    "images, poorly lit areas, underexposed and overexposed scenes, poor color balance, "
    "washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, "
    "color banding, unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)
LAST_REPORT: dict[str, Any] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_argument_group("model source")
    src.add_argument("--source", default=None, help="local snapshot dir (e.g. mounted /model)")
    src.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="hub repo id (needs --download)")
    src.add_argument("--download", action="store_true", help="allow fetching from the Hub")
    src.add_argument("--device", default=None, help="default: cuda if available else cpu")
    src.add_argument("--device-map", default=None, help="accelerate device_map (e.g. 'cuda')")

    data = p.add_argument_group("real data (LeRobot GR00T-G1 snapshot)")
    data.add_argument("--data-dir", required=True, help="snapshot root with data/ + videos/")
    data.add_argument("--episodes", type=int, default=12, help="use the first N episodes")
    data.add_argument("--start", type=int, default=0, help="first source episode index")
    data.add_argument("--windows-per-episode", type=int, default=8)
    data.add_argument("--frames", type=int, default=5, help="context frames per window")
    data.add_argument("--height", type=int, default=192, help="probe frame height (mult. of 32)")
    data.add_argument("--width", type=int, default=256, help="probe frame width (mult. of 32)")
    data.add_argument("--chunk-steps", type=int, default=16, help="action label chunk length")
    data.add_argument("--fps", type=float, default=30.0, help="source video fps (mRoPE timing)")
    data.add_argument("--instruction", default=None, help="default: task string from meta/")

    pr = p.add_argument_group("probe")
    pr.add_argument("--alphas", default="1,10,100,1000,10000", help="ridge alpha grid")
    pr.add_argument("--measured-blocks", default="2,12", help="Wan early-block pick, depth-scaled")
    pr.add_argument("--heuristic-blocks", default="18,26", help="mid+late depth heuristic")

    gen = p.add_argument_group("generation (--generate)")
    gen.add_argument("--generate", action="store_true", help="sample a future video instead")
    gen.add_argument("--gen-prompt", default=None, help="default: the probe instruction")
    gen.add_argument("--gen-episode", type=int, default=0, help="episode for the start frame")
    gen.add_argument("--gen-frame", type=int, default=0, help="frame index for the start frame")
    gen.add_argument(
        "--gen-cond-frames",
        type=int,
        default=1,
        help="real frames ending at --gen-frame used as conditioning; 1 = image mode, "
        "4k+1 (e.g. 9) = Video2World mode (Wan2.2-TI2V has no such mode)",
    )
    gen.add_argument("--gen-num-frames", type=int, default=49, help="(F-1) %% 4 == 0")
    gen.add_argument("--gen-steps", type=int, default=35, help="denoising steps")
    gen.add_argument("--gen-height", type=int, default=480)
    gen.add_argument("--gen-width", type=int, default=640)
    gen.add_argument("--gen-guidance", type=float, default=6.0, help="Cosmos3 default CFG")
    gen.add_argument("--gen-seed", type=int, default=0)
    gen.add_argument("--gen-out", default="cosmos3_future.mp4")

    p.add_argument("--out", default="cosmos3_probe_report.json")
    return p.parse_args(argv)


# ---- GPU: pooled per-block generation-pathway features -----------------------------------


def load_pipeline(args: argparse.Namespace, device: str) -> Any:
    """Transformer + tokenizer + Wan VAE + scheduler; no safety checker, no sound/vision towers."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
    from transformers import AutoTokenizer

    source = args.source
    if source is None:
        if not args.download:
            raise SystemExit("no --source and --download not set")
        source = args.model_id
    transformer = Cosmos3OmniTransformer.from_pretrained(
        source, subfolder="transformer", torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    if args.device_map is None:
        transformer = transformer.to(device)
    transformer.eval()
    # The Cosmos3 VAE is the Wan2.2 VAE (see vae/config.json _name_or_path); keep it fp32
    # like the Wan adapter — WanVAE was trained without autocast.
    vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(source, subfolder="text_tokenizer")
    scheduler = UniPCMultistepScheduler.from_pretrained(source, subfolder="scheduler")
    return Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        enable_safety_checker=False,
    )


def _pack_clean_window(pipe: Any, latents: Any, input_ids: list[int], fps: float, device: str):
    """Pipeline-identical packing with EVERY latent frame a clean conditioning frame.

    This is the conditioning-token path of i2v generation (no noise, no timestep embeds),
    applied to the whole context window — the closest Cosmos3 analog of the Wan probe's
    single clean-latent DiT forward.
    """
    import torch

    text_seg = pipe._prepare_text_segment(input_ids, device)
    latent_t = int(latents.shape[2])
    vis_seg = pipe._prepare_vision_segment(
        input_vision_tokens=latents,
        has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=fps,
        curr=text_seg["und_len"],
        device=device,
        condition_frame_indexes=list(range(latent_t)),
    )
    if int(vis_seg["num_noisy_vision_tokens"]) != 0:
        raise RuntimeError("expected an all-clean window, got noisy vision tokens")
    return {
        "input_ids": text_seg["input_ids"],
        "text_indexes": text_seg["text_indexes"],
        "position_ids": torch.cat([text_seg["text_mrope_ids"], vis_seg["vision_mrope_ids"]], dim=1),
        "und_len": text_seg["und_len"],
        "sequence_length": text_seg["und_len"] + vis_seg["num_vision_tokens"],
        "vision_token_shapes": vis_seg["vision_token_shapes"],
        "vision_sequence_indexes": vis_seg["vision_sequence_indexes"],
        "vision_mse_loss_indexes": vis_seg["vision_mse_loss_indexes"],
        "vision_noisy_frame_indexes": vis_seg["vision_noisy_frame_indexes"],
    }


def _forward_features(pipe: Any, packed: dict[str, Any], latents: Any) -> Any:
    """One joint forward; per-layer generation-pathway output token-pooled -> [blocks, D]."""
    import torch

    captured: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(
            lambda module, inputs, output, idx=i: captured.__setitem__(idx, output[1].detach())
        )
        for i, layer in enumerate(pipe.transformer.layers)
    ]
    try:
        with torch.no_grad():
            pipe.transformer(
                input_ids=packed["input_ids"],
                text_indexes=packed["text_indexes"],
                position_ids=packed["position_ids"],
                und_len=packed["und_len"],
                sequence_length=packed["sequence_length"],
                vision_tokens=[latents.to(dtype=pipe.transformer.dtype)],
                vision_token_shapes=packed["vision_token_shapes"],
                vision_sequence_indexes=packed["vision_sequence_indexes"],
                vision_mse_loss_indexes=packed["vision_mse_loss_indexes"],
                vision_timesteps=torch.zeros((0,), device=latents.device),
                vision_noisy_frame_indexes=packed["vision_noisy_frame_indexes"],
            )
    finally:
        for handle in handles:
            handle.remove()
    # The gen half holds exactly the vision tokens here (no sound/action segments).
    return torch.stack(
        [captured[i].float().mean(dim=0) for i in range(len(pipe.transformer.layers))]
    )


def extract_features(
    args: argparse.Namespace,
    windows: list[dict[str, Any]],
    instruction: str,
    report: smoke.Report,
) -> np.ndarray:
    """One frozen MoT forward per window -> pooled per-block features ``[N, blocks, D]``."""
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    pipe = load_pipeline(args, device)
    report.check("probe.load", True, f"{time.perf_counter() - t0:.1f}s")
    config = pipe.transformer.config
    num_blocks = int(config.num_hidden_layers)
    report.info["geometry"] = {
        "num_layers": num_blocks,
        "feature_dim": int(config.hidden_size),
        "latent_channel": int(config.latent_channel),
        "latent_patch_size": int(config.latent_patch_size),
        "vae_spatial": int(pipe.vae.config.scale_factor_spatial),
        "vae_temporal": int(pipe.vae.config.scale_factor_temporal),
    }

    cond_ids, _ = pipe.tokenize_prompt(
        instruction,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    report.info["text_tokens"] = len(cond_ids)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pooled = np.zeros((len(windows), num_blocks, int(config.hidden_size)), dtype=np.float32)
    t0 = time.perf_counter()
    for i, window in enumerate(windows):
        frames = np.ascontiguousarray(window["frames"])  # [F, H, W, 3] uint8
        x = torch.from_numpy(frames).to(device).float().permute(3, 0, 1, 2)[None] / 127.5 - 1.0
        latents = pipe._encode_video(x)
        packed = _pack_clean_window(pipe, latents, cond_ids, args.fps, device)
        feats = _forward_features(pipe, packed, latents)
        pooled[i] = feats.cpu().numpy()
        if i == 0:
            again = _forward_features(pipe, packed, latents)
            report.check(
                "probe.deterministic",
                bool(torch.equal(feats, again)),
                "two forwards bit-identical",
            )
            report.info["tokens"] = {
                "text": int(packed["und_len"]),
                "vision": int(packed["sequence_length"]) - int(packed["und_len"]),
            }
    wall = time.perf_counter() - t0
    timings = {"forwards_s": round(wall, 2), "per_window_s": round(wall / max(len(windows), 1), 3)}
    if torch.cuda.is_available():
        timings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report.info["timings"] = timings

    report.check("probe.features_finite", bool(np.isfinite(pooled).all()), f"{pooled.shape}")
    spread = float(pooled[:, 0].std(axis=0).mean())
    report.check(
        "probe.features_vary_across_windows", spread > 1e-6, f"block-0 spread {spread:.4f}"
    )
    return pooled


# ---- generation (--generate): what does the Cosmos3 prior imagine? -----------------------


def load_gen_clip(args: argparse.Namespace) -> np.ndarray:
    """RGB clip of --gen-cond-frames real frames ending at --gen-frame ([N, H, W, 3]).

    The Video2World conditioning context: the model sees the real motion leading up to
    the reference frame instead of that frame alone. Decoded via imageio (GR00T ships
    AV1, which cv2's bundled FFmpeg may lack).
    """
    import imageio.v3 as iio

    first = args.gen_frame - args.gen_cond_frames + 1
    if first < 0:
        raise ValueError(
            f"--gen-frame {args.gen_frame} has no {args.gen_cond_frames}-frame history"
        )
    video = (
        Path(args.data_dir)
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"episode_{args.gen_episode:06d}.mp4"
    )
    frames: list[np.ndarray] = []
    for i, rgb in enumerate(iio.imiter(str(video), plugin="FFMPEG")):
        if i > args.gen_frame:
            break
        if i >= first:
            frames.append(np.asarray(rgb))
    if len(frames) != args.gen_cond_frames:
        raise ValueError(f"read {len(frames)} frames from {video}, wanted {args.gen_cond_frames}")
    return np.stack(frames)


def generate_future(
    args: argparse.Namespace, image_rgb: np.ndarray, report: smoke.Report
) -> dict[str, Any]:
    """Sample a future clip from the start frame + prompt; writes mp4 (+ start-frame png).

    Mirror of the Wan probe's ``generate_future`` for the side-by-side: identical start
    frame and positive prompt, each backbone with its own recommended negative prompt and
    guidance. Not a training path — a qualitative probe and presentation material.
    """
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    if (args.gen_num_frames - 1) % 4:
        raise ValueError(f"--gen-num-frames must satisfy (F-1) % 4 == 0, got {args.gen_num_frames}")
    if (args.gen_cond_frames - 1) % 4:
        raise ValueError(f"--gen-cond-frames must satisfy (N-1) % 4 == 0, got {args.gen_cond_frames}")
    if args.gen_height % 32 or args.gen_width % 32:
        raise ValueError("--gen-height/--gen-width must be multiples of 32")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompt = args.gen_prompt or wanprobe.DEFAULT_INSTRUCTION

    t0 = time.perf_counter()
    pipe = load_pipeline(args, device)
    # The fp32 VAE decode of the full clip lands on top of the ~32 GB bf16 transformer;
    # tiling keeps the peak safely inside a 48 GB card.
    pipe.vae.enable_tiling()
    load_s = time.perf_counter() - t0
    report.check("generate.load", True, f"{load_s:.1f}s")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    size = (args.gen_width, args.gen_height)
    if image_rgb.ndim == 4:  # Video2World: real clip ending at --gen-frame conditions the run
        clip = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in image_rgb]
        cond = {
            "video": clip,
            "condition_frame_indexes_vision": tuple(range((len(clip) - 1) // 4 + 1)),
        }
        start = clip[-1]
    else:
        start = Image.fromarray(image_rgb).resize(size, Image.LANCZOS)
        cond = {"image": start}
    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=COSMOS_NEGATIVE,
        **cond,
        num_frames=args.gen_num_frames,
        height=args.gen_height,
        width=args.gen_width,
        fps=24,
        num_inference_steps=args.gen_steps,
        guidance_scale=args.gen_guidance,
        generator=torch.Generator(device).manual_seed(args.gen_seed),
        enable_safety_check=False,
    )
    sample_s = time.perf_counter() - t0

    out_path = Path(args.gen_out)
    export_to_video(result.video, str(out_path), fps=24)
    start.save(out_path.with_suffix(".start.png"))
    ok = out_path.is_file() and out_path.stat().st_size > 0
    report.check("generate.wrote_mp4", ok, f"{out_path} ({sample_s:.0f}s sampling)")
    info = {
        "prompt": prompt,
        "negative_prompt": "cosmos recommended quality prompt",
        "episode": args.gen_episode,
        "start_frame": args.gen_frame,
        "conditioning": "video" if image_rgb.ndim == 4 else "image",
        "cond_frames": args.gen_cond_frames,
        "num_frames": args.gen_num_frames,
        "steps": args.gen_steps,
        "guidance_scale": args.gen_guidance,
        "size": [args.gen_height, args.gen_width],
        "seed": args.gen_seed,
        "fps": 24,
        "load_s": round(load_s, 1),
        "sample_s": round(sample_s, 1),
        "mp4": str(out_path),
        "start_png": str(out_path.with_suffix(".start.png")),
    }
    if torch.cuda.is_available():
        info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report.info["generate"] = info
    return info


# ---- entry -------------------------------------------------------------------------------


def finalize(report: smoke.Report, out_path: str) -> int:
    payload = {"ok": not report.failed, "checks": report.checks, "info": report.info}
    LAST_REPORT.clear()
    LAST_REPORT.update(payload)
    serialized = json.dumps(payload, indent=2, default=str)
    print(f"\n===== REPORT =====\n{serialized}", flush=True)
    try:
        Path(out_path).write_text(serialized)
        print(f"report -> {out_path}", flush=True)
    except OSError as err:  # a read-only mount must not sink a passing run
        print(f"could not write {out_path}: {err}", flush=True)
    if report.failed:
        print(f"FAILED: {', '.join(report.failed)}", flush=True)
        return 1
    print(f"ALL {len(report.checks)} CHECKS PASSED", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = smoke.Report()
    import torch

    import wam

    report.info.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "wam_version": getattr(wam, "__version__", "unknown"),
            "model": args.source or args.model_id,
            "args": vars(args),
        }
    )
    if args.generate:
        generate_future(args, wanprobe.load_gen_frame(args), report)
    else:
        windows, instruction, data_info = wanprobe.build_windows(args)
        report.info["data"] = data_info
        report.check("probe.windows_built", len(windows) >= 24, f"{len(windows)} windows")
        pooled = extract_features(args, windows, instruction, report)
        wanprobe.analyze_probes(pooled, windows, args, report)
    return finalize(report, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
