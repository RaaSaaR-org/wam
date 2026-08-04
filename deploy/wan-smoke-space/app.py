"""ZeroGPU Space: WAM Wan-backbone checks on a free PRO GPU (T-15/T-16 prep, OD-04/OD-05).

Four tabs, one deployed implementation each — no vendored logic that could drift:

- **smoke test** wraps `smoke.py` (deployed verbatim from `scripts/hf_job_wan_smoke.py`).
- **readout probes** wraps `probe.py` (`scripts/hf_job_wan_probe.py`): real GR00T-G1
  episodes -> frozen DiT features per block -> ridge probes against real action labels.
- **generate future** wraps the same `probe.py`: sample what the backbone imagines from a
  real episode's start frame + instruction (presentation material, not a training path).
- **dream (WAM sampler)** wraps `dream_cli.py` (`scripts/dream.py`) + `wam.evaluation.dream`:
  integrate WAM's OWN flow, conditioned on instruction *and robot state*, and measure motion
  against the VAE round-trip of the same clips. The distinction from the tab above is not
  cosmetic — a diffusers pipeline has no state port, so every clip generated there is missing
  the proprioception token the DiT was trained with (T-35).

Two constraints shape the design:

- The Wan repo is ~34 GB (the transformer ships fp32) against a documented 16 GB Space RAM
  default, so the adapter loads with `--device-map cuda` and accelerate streams shards
  straight to the GPU. Measured, a ZeroGPU host is far bigger (104 GB cgroup limit, 192
  cores), so this is insurance rather than a hard requirement; it also cuts load to ~7 s.
- ZeroGPU only exposes a real GPU *inside* `@spaces.GPU`, and releases it on return. Model
  and dataset downloads, probe-window building and the ridge analysis all run outside the
  decorator, so they cost no GPU quota.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DATA_REPO = os.environ.get("DATA_REPO", "nvidia/GR00T-N1.7-AppleToPlate")
# The dream tab needs a TRAINED checkpoint, not just an exported LoRA: the state projection and
# the heads live in the checkpoint, and the state token is the whole point of that tab. Private
# repos need --set-hf-token on deploy_wan_space.py (a Space carries no token of its own).
CHECKPOINT_REPO = os.environ.get("CHECKPOINT_REPO", "")
GPU_DURATION = int(os.environ.get("GPU_DURATION", "240"))
GEN_GPU_DURATION = int(os.environ.get("GEN_GPU_DURATION", "420"))
DEFAULT_INSTRUCTION = "pick up the red cube and place it in the bin"
DEFAULT_GEN_PROMPT = (
    "the humanoid robot reaches for the apple, picks it up and places it on the plate"
)
# Nothing here needs the demo videos/images that ship with the model card.
IGNORE = ["assets/*", "examples/*", "*.mp4", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"]


def host_info() -> dict[str, Any]:
    """Host RAM/disk/accelerator — undocumented for ZeroGPU, so we measure and report it."""
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "accelerator_env": os.environ.get("ACCELERATOR"),
        "memory_env": os.environ.get("MEMORY"),
        "space_id": os.environ.get("SPACE_ID"),
    }
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1e6, 1)
                break
    with contextlib.suppress(OSError):
        usage = shutil.disk_usage("/")
        info["disk_total_gb"] = round(usage.total / 1e9, 1)
        info["disk_free_gb"] = round(usage.free / 1e9, 1)
    return info


def download_episodes(episodes: list[int]) -> str:
    """Fetch just the needed GR00T episodes (parquet + ego mp4 + meta) — CPU, no quota."""
    patterns = ["meta/*"]
    for i in episodes:
        patterns.append(f"data/chunk-000/episode_{i:06d}.parquet")
        patterns.append(f"videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4")
    return snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=patterns)


@spaces.GPU(duration=GPU_DURATION)
def run_on_gpu(source: str, argv: list[str]) -> tuple[str, dict[str, Any], str]:
    """Load Wan onto the real GPU and run every smoke check. Returns (log, report, gpu)."""
    import smoke
    import torch

    free, total = torch.cuda.mem_get_info()
    gpu = f"{torch.cuda.get_device_name()} — {total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free"

    buffer = io.StringIO()
    started = time.perf_counter()
    try:
        common = ["--source", source, "--device", "cuda", "--device-map", "cuda"]
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = smoke.main([*common, *argv])
        report = dict(smoke.LAST_REPORT)
        report["exit_code"] = code
    except Exception:  # noqa: BLE001 - a crashed run must still surface its log in the UI
        buffer.write("\n" + traceback.format_exc())
        report = {"ok": False, "error": "exception before the report was built"}
    report["gpu"] = gpu
    report["gpu_wall_s"] = round(time.perf_counter() - started, 1)
    return buffer.getvalue(), report, gpu


@spaces.GPU(duration=GPU_DURATION)
def probe_on_gpu(
    args: Any, windows: list[dict[str, Any]], instruction: str
) -> tuple[str, Any, Any]:
    """One frozen DiT forward per window -> pooled per-block features [N, blocks, D]."""
    import probe

    buffer = io.StringIO()
    started = time.perf_counter()
    report = probe.smoke.Report()
    pooled = None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            pooled = probe.extract_features(args, windows, instruction, report)
    except Exception:  # noqa: BLE001
        buffer.write("\n" + traceback.format_exc())
    buffer.write(f"\ngpu wall: {time.perf_counter() - started:.1f}s")
    return buffer.getvalue(), pooled, report


@spaces.GPU(duration=GEN_GPU_DURATION)
def generate_on_gpu(args: Any, image: Any) -> tuple[str, Any]:
    """Sample a future clip with WanImageToVideoPipeline; returns (log, report)."""
    import probe

    buffer = io.StringIO()
    started = time.perf_counter()
    report = probe.smoke.Report()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            probe.generate_future(args, image, report)
    except Exception:  # noqa: BLE001
        buffer.write("\n" + traceback.format_exc())
        report.check("generate.crashed", False, "see log")
    buffer.write(f"\ngpu wall: {time.perf_counter() - started:.1f}s")
    return buffer.getvalue(), report


def run(frames: int, height: int, width: int, blocks: str, instruction: str, ablate: bool):
    """Smoke tab: download on CPU (free), then hand off to the GPU. Yields progress."""
    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    yield "\n".join(log), None

    log.append("\ndownloading weights (no GPU quota consumed, ~34 GB on first run)…")
    yield "\n".join(log), None
    t0 = time.perf_counter()
    try:
        source = snapshot_download(MODEL_ID, ignore_patterns=IGNORE)
    except Exception:  # noqa: BLE001
        log.append(traceback.format_exc())
        yield "\n".join(log), {"ok": False, "error": "download failed"}
        return
    log.append(f"downloaded to {source} in {time.perf_counter() - t0:.0f}s")
    log.append(f"\nrequesting GPU (duration cap {GPU_DURATION}s)…")
    yield "\n".join(log), None

    argv = [
        "--frames", str(int(frames)),
        "--height", str(int(height)),
        "--width", str(int(width)),
        "--instruction", instruction,
    ]  # fmt: skip
    if blocks.strip():
        argv += ["--blocks", blocks.strip()]
    if ablate:
        argv += ["--ablate"]

    try:
        output, report, gpu = run_on_gpu(source, argv)
    except Exception:  # noqa: BLE001 - quota exhaustion / duration overrun land here
        log.append(traceback.format_exc())
        yield "\n".join(log), {"ok": False, "error": "GPU call failed"}
        return

    log.append(f"gpu: {gpu}")
    log.append(output)
    verdict = "ALL CHECKS PASSED" if report.get("ok") else "FAILED"
    log.append(f"\n=== {verdict} ===")
    yield "\n".join(log), report


def run_probe(episodes: int, windows_per_ep: int, frames: int, height: int, width: int,
              chunk_steps: int, readout: str = ""):  # fmt: skip
    """Probe tab: data + windows on CPU, features on GPU, ridge analysis on CPU."""
    import probe

    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    log.append(f"data: {DATA_REPO}")
    yield "\n".join(log), None

    try:
        log.append("\ndownloading weights + episodes (no GPU quota consumed)…")
        yield "\n".join(log), None
        t0 = time.perf_counter()
        source = snapshot_download(MODEL_ID, ignore_patterns=IGNORE)
        data_dir = download_episodes(list(range(int(episodes))))
        log.append(f"downloaded in {time.perf_counter() - t0:.0f}s")

        probe_argv = [
            "--data-dir", data_dir,
            "--source", source,
            "--device", "cuda",
            "--device-map", "cuda",
            "--episodes", str(int(episodes)),
            "--windows-per-episode", str(int(windows_per_ep)),
            "--frames", str(int(frames)),
            "--height", str(int(height)),
            "--width", str(int(width)),
            "--chunk-steps", str(int(chunk_steps)),
            "--readout", (readout or "").strip() or probe.DEFAULT_READOUTS,
            "--out", "/tmp/wan_probe_report.json",
        ]  # fmt: skip
        args = probe.parse_args(probe_argv)
        t0 = time.perf_counter()
        windows, instruction, data_info = probe.build_windows(args)
        log.append(f"built {len(windows)} windows in {time.perf_counter() - t0:.0f}s (CPU)")
        log.append(f"instruction: {instruction!r}")
        log.append(f"\nrequesting GPU (duration cap {GPU_DURATION}s)…")
        yield "\n".join(log), None

        gpu_log, pooled, report = probe_on_gpu(args, windows, instruction)
        log.append(gpu_log)
        if pooled is None:
            yield "\n".join(log), {"ok": False, "error": "feature extraction failed"}
            return
        report.info["data"] = data_info
        report.check("probe.windows_built", len(windows) >= 24, f"{len(windows)} windows")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            probe.analyze_probes(pooled, windows, args, report)
            code = probe.finalize(report, args.out)
        log.append(buffer.getvalue())
        payload = dict(probe.LAST_REPORT)
        payload["exit_code"] = code
        yield "\n".join(log), payload
    except Exception:  # noqa: BLE001 - quota exhaustion / bad inputs land here
        log.append(traceback.format_exc())
        yield "\n".join(log), {"ok": False, "error": "probe run failed"}


def run_generate(prompt: str, episode: int, frame: int, num_frames: int, steps: int,
                 height: int, width: int, seed: int, lora: str, lora_scale: float):  # fmt: skip
    """Generate tab: start frame on CPU, diffusion sampling on GPU, mp4 back to the UI."""
    import probe

    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    yield "\n".join(log), None, None

    try:
        log.append("\ndownloading weights + start episode (no GPU quota consumed)…")
        yield "\n".join(log), None, None
        source = snapshot_download(MODEL_ID, ignore_patterns=IGNORE)
        data_dir = download_episodes([int(episode)])

        # The LoRA scale is in the filename: a sweep writes one file per strength, so the
        # clips stay side by side instead of overwriting each other.
        tag = f"_lora{float(lora_scale):g}" if lora.strip() else ""
        out = f"/tmp/wan_future_ep{int(episode)}_f{int(frame)}_seed{int(seed)}{tag}.mp4"
        argv = [
            "--data-dir", data_dir,
            "--source", source,
            "--device", "cuda",
            "--generate",
            "--gen-episode", str(int(episode)),
            "--gen-frame", str(int(frame)),
            "--gen-num-frames", str(int(num_frames)),
            "--gen-steps", str(int(steps)),
            "--gen-height", str(int(height)),
            "--gen-width", str(int(width)),
            "--gen-seed", str(int(seed)),
            "--gen-out", out,
        ]  # fmt: skip
        if prompt.strip():
            argv += ["--gen-prompt", prompt.strip()]
        if lora.strip():
            argv += ["--gen-lora", lora.strip(), "--gen-lora-scale", str(float(lora_scale))]
            log.append(f"LoRA: {lora.strip()} at scale {float(lora_scale):g}")
        else:
            log.append("LoRA: none (base model)")
        args = probe.parse_args(argv)
        image = probe.load_gen_frame(args)
        log.append(f"start frame: episode {episode} frame {frame}, {image.shape}")
        log.append(f"\nrequesting GPU (duration cap {GEN_GPU_DURATION}s)…")
        yield "\n".join(log), None, None

        gpu_log, report = generate_on_gpu(args, image)
        log.append(gpu_log)
        payload = {"ok": not report.failed, "checks": report.checks, "info": report.info}
        video = out if Path(out).is_file() else None
        verdict = "DONE" if payload["ok"] and video else "FAILED"
        log.append(f"\n=== {verdict} ===")
        yield "\n".join(log), video, payload
    except Exception:  # noqa: BLE001
        log.append(traceback.format_exc())
        yield "\n".join(log), None, {"ok": False, "error": "generation failed"}


@spaces.GPU(duration=GEN_GPU_DURATION)
def dream_on_gpu(
    checkpoint: str, source: str, windows: list[dict[str, Any]], instruction: str, opts: Any
) -> tuple[str, Any, Any]:
    """Load the WAM checkpoint, sample every arm, measure motion. Returns (log, report, sheets)."""
    import dream_cli
    import torch

    from wam.evaluation.dream import build_report
    from wam.runtime.policies import load_joint_policy

    buffer = io.StringIO()
    started = time.perf_counter()
    report = None
    sheets: dict[str, str] = {}
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            # snapshot_download returns a repo ROOT; load_checkpoint_raw opens a FILE (the config
            # and provenance ride in the safetensors metadata). One resolver, shared with the CLI.
            checkpoint = dream_cli.resolve_checkpoint(checkpoint)
            policy = load_joint_policy(checkpoint, device="cuda", backbone_source=source)
            model = policy.model
            model.eval()
            batch = dream_cli.batch_from_windows(windows, instruction)
            batch = {
                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }
            arms = dream_cli.run_arms(opts, batch, model)
            arms["gt"] = dream_cli.strip_anchor(
                batch["frames"], opts.anchor, dream_cli.vae_temporal_stride(model)
            )
            pairs = {}
            if "lora" in arms and "base" in arms:
                pairs["lora_vs_base"] = ("lora", "base")
            if "base_seed1" in arms:
                pairs["base_seed_null"] = ("base", "base_seed1")
            info = {
                "checkpoint": checkpoint,
                "run_id": getattr(policy.metadata, "run_id", None),
                "config_hash": getattr(policy.metadata, "config_hash", None),
                "clips": int(batch["frames"].shape[0]),
                "steps": opts.steps,
                "anchor_latent_frames": opts.anchor,
                "state_conditioned": True,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
            report = build_report(arms, reference_arm="recon", pairs=pairs, info=info)
            for name, frames in arms.items():
                path = Path(f"/tmp/dream_{name}.png")
                if dream_cli.write_contact_sheet(frames, path):
                    sheets[name] = str(path)
    except Exception:  # noqa: BLE001 - a crashed run must still surface its log in the UI
        buffer.write("\n" + traceback.format_exc())
    buffer.write(f"\ngpu wall: {time.perf_counter() - started:.1f}s")
    return buffer.getvalue(), report, sheets


def run_dream(checkpoint: str, episodes: int, windows_per_ep: int, steps: int, seed: int,
              anchor: int, height: int, width: int, frames: int, instruction: str):  # fmt: skip
    """Dream tab: episodes + checkpoint on CPU (free), sampling and metrics on GPU."""
    import probe

    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    yield "\n".join(log), None, None

    try:
        repo = checkpoint.strip() or CHECKPOINT_REPO
        if not repo:
            raise ValueError(
                "no checkpoint: set the CHECKPOINT_REPO variable on the Space or paste a repo id "
                "/ local path. This tab samples a TRAINED WAM model — unlike 'generate future', "
                "which runs the exported LoRA inside a stock diffusers pipeline and therefore "
                "cannot supply the proprioception token the DiT was trained with."
            )
        log.append("\ndownloading weights + episodes + checkpoint (no GPU quota consumed)…")
        yield "\n".join(log), None, None
        source = snapshot_download(MODEL_ID, ignore_patterns=IGNORE)
        data_dir = download_episodes(list(range(int(episodes))))
        ckpt = repo if Path(repo).exists() else snapshot_download(repo)

        argv = [
            "--data-dir", data_dir,
            "--episodes", str(int(episodes)),
            "--windows-per-episode", str(int(windows_per_ep)),
            "--frames", str(int(frames)),
            "--height", str(int(height)),
            "--width", str(int(width)),
        ]  # fmt: skip
        if instruction.strip():
            argv += ["--instruction", instruction.strip()]
        probe_args = probe.parse_args(argv)
        built, resolved_instruction, data_info = probe.build_windows(probe_args)
        log.append(f"windows: {len(built)} from {data_info['episodes']}")
        log.append(f"instruction: {resolved_instruction}")

        import argparse as _argparse

        opts = _argparse.Namespace(
            steps=int(steps), seed=int(seed), anchor=int(anchor), no_base_arm=False
        )
        log.append(f"\nrequesting GPU (duration cap {GEN_GPU_DURATION}s)…")
        yield "\n".join(log), None, None

        gpu_log, report, sheets = dream_on_gpu(ckpt, source, built, resolved_instruction, opts)
        log.append(gpu_log)
        if report is None:
            yield "\n".join(log), None, {"ok": False, "error": "dream failed — see log"}
            return
        payload = json.loads(report.model_dump_json())
        for name, metrics in report.arms.items():
            log.append(
                f"  {name:<12} motion {metrics.motion:8.3f}  "
                f"ratio {report.motion_ratio.get(name, float('nan')):6.3f}  "
                f"static {metrics.static_fraction:5.2f}"
            )
        for label, value in report.pair_distance.items():
            log.append(f"  {label:<12} {value:.4f}")
        for name, verdict in report.verdicts.items():
            log.append(f"  VERDICT {name}: {verdict}")
        gallery = [(path, name) for name, path in sorted(sheets.items())]
        yield "\n".join(log), gallery, payload
    except Exception:  # noqa: BLE001
        log.append(traceback.format_exc())
        yield "\n".join(log), None, {"ok": False, "error": "dream failed"}


with gr.Blocks(title="WAM · Wan backbone on ZeroGPU") as demo:
    gr.Markdown(
        f"""# WAM — Wan backbone on ZeroGPU

Model: `{MODEL_ID}` · real data: `{DATA_REPO}` · code:
[RaaSaaR-org/wam](https://github.com/RaaSaaR-org/wam) (`smoke.py` / `probe.py` deployed
verbatim from `scripts/`). First run downloads ~34 GB before any GPU is requested.
"""
    )
    with gr.Tab("smoke test"):
        gr.Markdown(
            "Does the real Wan DiT produce usable action-readout features through the WAM "
            "interfaces, with the shapes `WanI2VAdapter` claims? Nothing is trained here."
        )
        with gr.Row():
            frames = gr.Number(value=5, label="frames", precision=0)
            height = gr.Number(value=256, label="height", precision=0)
            width = gr.Number(value=448, label="width", precision=0)
        blocks = gr.Textbox(value="", label="readout blocks (blank = auto, mid/late depth)")
        instruction = gr.Textbox(value=DEFAULT_INSTRUCTION, label="instruction")
        ablate = gr.Checkbox(
            value=False,
            label="ablate readout blocks (probe every DiT block for motion/instruction/state "
            "sensitivity — 4 extra forwards, same load)",
        )
        smoke_btn = gr.Button("Run smoke test", variant="primary")
        smoke_log = gr.Textbox(label="log", lines=28, max_lines=28, show_copy_button=True)
        smoke_report = gr.JSON(label="report")
        smoke_btn.click(
            run, [frames, height, width, blocks, instruction, ablate], [smoke_log, smoke_report]
        )

    with gr.Tab("readout probes (real data)"):
        gr.Markdown(
            "Label-validated readout check: real GR00T-G1 windows through the frozen DiT, "
            "every block's features ridge-regressed onto the BC action labels, ranked by "
            "held-out-episode R². A state-only ridge is the floor to beat.\n\n"
            "**Readouts (I-1).** The recorded verdicts were all measured through `mean` — a "
            "mean-pool over the token grid, which deletes *where* things are. `grid<R>x<C>` "
            "keeps that geometry; `rand<N>` pools the same tokens into N equally sized random "
            "groups, so it has the identical feature width and isolates geometry from "
            "dimensionality. **grid > rand ⇒ position carries signal; grid ≈ rand ⇒ the "
            "mean-pool verdict stands.**"
        )
        with gr.Row():
            p_episodes = gr.Number(value=12, label="episodes", precision=0)
            p_windows = gr.Number(value=8, label="windows / episode", precision=0)
            p_frames = gr.Number(value=5, label="context frames", precision=0)
        with gr.Row():
            p_height = gr.Number(value=192, label="height", precision=0)
            p_width = gr.Number(value=256, label="width", precision=0)
            p_chunk = gr.Number(value=16, label="chunk steps (label)", precision=0)
        # Blank resolves to probe.DEFAULT_READOUTS inside run_probe: `probe` is imported
        # lazily (a broken import must cost one tab, not the whole Space), so the default
        # cannot be read here without vendoring a second copy of it.
        p_readout = gr.Textbox(
            value="",
            placeholder="mean,grid2x2,rand4",
            label="readouts — blank = default (first is primary; 192x256 → a 6x8 token grid)",
        )
        probe_btn = gr.Button("Run readout probes", variant="primary")
        probe_log = gr.Textbox(label="log", lines=28, max_lines=28, show_copy_button=True)
        probe_report = gr.JSON(label="report")
        probe_btn.click(
            run_probe,
            [p_episodes, p_windows, p_frames, p_height, p_width, p_chunk, p_readout],
            [probe_log, probe_report],
        )

    with gr.Tab("generate future"):
        gr.Markdown(
            "Sample what the backbone *imagines*: diffusion video from a real episode's "
            "start frame + an instruction. Not a training path — a qualitative probe and "
            "presentation material. One clip per GPU call."
        )
        g_prompt = gr.Textbox(value=DEFAULT_GEN_PROMPT, label="prompt")
        with gr.Row():
            g_episode = gr.Number(value=0, label="episode", precision=0)
            g_frame = gr.Number(value=0, label="start frame", precision=0)
            g_seed = gr.Number(value=0, label="seed", precision=0)
        with gr.Row():
            g_frames = gr.Number(value=49, label="frames ((F-1)%4==0)", precision=0)
            g_steps = gr.Number(value=35, label="steps", precision=0)
            g_height = gr.Number(value=480, label="height", precision=0)
            g_width = gr.Number(value=640, label="width", precision=0)
        gr.Markdown(
            "**Apply a WAM fine-tune.** Leave blank for the base prior. With a LoRA, the same "
            "seed + start frame + prompt at scale 0 / 0.5 / 1 isolates exactly what training "
            "changed — the thing to watch is the *arm*, since the base model has never seen a "
            "G1 and invents a generic manipulator. Caveats: the adapter was trained at 9 frames "
            "of 128x160, so generating larger and longer is out of its distribution; and the "
            "DiT was trained with proprioception tokens on the text context, which a diffusers "
            "pipeline cannot supply."
        )
        with gr.Row():
            g_lora = gr.Textbox(
                value="",
                label="LoRA repo or path (blank = base model)",
                placeholder="<user>/wam-t16-lora-seed0",
            )
            g_lora_scale = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=1.0,
                step=0.05,
                label="LoRA scale (1.0 = as trained)",
            )
        gen_btn = gr.Button("Generate future video", variant="primary")
        gen_video = gr.Video(label="generated future")
        gen_log = gr.Textbox(label="log", lines=16, max_lines=28, show_copy_button=True)
        gen_report = gr.JSON(label="report")
        gen_btn.click(
            run_generate,
            [
                g_prompt,
                g_episode,
                g_frame,
                g_frames,
                g_steps,
                g_height,
                g_width,
                g_seed,
                g_lora,
                g_lora_scale,
            ],
            [gen_log, gen_video, gen_report],
        )

    with gr.Tab("dream (WAM sampler)"):
        gr.Markdown(
            "Sample the video branch **through WAM's own flow**, conditioned on the instruction "
            "*and the robot state* — the input the tab above structurally cannot supply, because "
            "a diffusers pipeline has no state port and the DiT was trained with proprioception "
            "on the text context.\n\n"
            "Five arms in one GPU call: `recon` (the VAE round-trip of the same clips — every "
            "ratio divides by it), `lora` (the dream), `base` (adapter disabled, same weights in "
            "memory), `base_seed1` (the null `d(lora, base)` is compared against) and `gt`. "
            "**Motion is only readable as a ratio**: the base prior scores 29.5 at 9x128x160 and "
            "2.93 at 49x480x640, a 10x spread that is about geometry, not imagination.\n\n"
            "A dream is a diagnostic. It is not evidence the policy works, and it is not "
            "training data — a generator fitted to 402 success-only episodes cannot invent the "
            "failures `PR-04` says the next corpus needs."
        )
        d_checkpoint = gr.Textbox(
            value="",
            label="WAM checkpoint repo or path (blank = the CHECKPOINT_REPO variable)",
            placeholder="<user>/wam-t16-lora-seed0-ckpt",
        )
        with gr.Row():
            d_episodes = gr.Number(value=8, label="episodes", precision=0)
            d_windows = gr.Number(value=2, label="windows / episode", precision=0)
            d_steps = gr.Number(value=32, label="Euler steps", precision=0)
            d_seed = gr.Number(value=0, label="seed", precision=0)
        with gr.Row():
            d_frames = gr.Number(value=9, label="frames (trained: 9)", precision=0)
            d_height = gr.Number(value=128, label="height (trained: 128)", precision=0)
            d_width = gr.Number(value=160, label="width (trained: 160)", precision=0)
            d_anchor = gr.Number(value=0, label="anchor latent frames (0 = faithful)", precision=0)
        d_instruction = gr.Textbox(value="", label="instruction (blank = the episode's own)")
        dream_btn = gr.Button("Dream", variant="primary")
        dream_gallery = gr.Gallery(label="contact sheets (one clip per row)", columns=1)
        dream_log = gr.Textbox(label="log", lines=20, max_lines=28, show_copy_button=True)
        dream_report = gr.JSON(label="report")
        dream_btn.click(
            run_dream,
            [
                d_checkpoint,
                d_episodes,
                d_windows,
                d_steps,
                d_seed,
                d_anchor,
                d_height,
                d_width,
                d_frames,
                d_instruction,
            ],
            [dream_log, dream_gallery, dream_report],
        )

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    demo.queue().launch()
