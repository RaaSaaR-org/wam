"""ZeroGPU Space: WAM Wan-backbone checks on a free PRO GPU (T-15/T-16 prep, OD-04/OD-05).

Three tabs, one deployed implementation each — no vendored logic that could drift:

- **smoke test** wraps `smoke.py` (deployed verbatim from `scripts/hf_job_wan_smoke.py`).
- **readout probes** wraps `probe.py` (`scripts/hf_job_wan_probe.py`): real GR00T-G1
  episodes -> frozen DiT features per block -> ridge probes against real action labels.
- **generate future** wraps the same `probe.py`: sample what the backbone imagines from a
  real episode's start frame + instruction (presentation material, not a training path).

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
                 height: int, width: int, seed: int):  # fmt: skip
    """Generate tab: start frame on CPU, diffusion sampling on GPU, mp4 back to the UI."""
    import probe

    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    yield "\n".join(log), None, None

    try:
        log.append("\ndownloading weights + start episode (no GPU quota consumed)…")
        yield "\n".join(log), None, None
        source = snapshot_download(MODEL_ID, ignore_patterns=IGNORE)
        data_dir = download_episodes([int(episode)])

        out = f"/tmp/wan_future_ep{int(episode)}_f{int(frame)}_seed{int(seed)}.mp4"
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
        gen_btn = gr.Button("Generate future video", variant="primary")
        gen_video = gr.Video(label="generated future")
        gen_log = gr.Textbox(label="log", lines=16, max_lines=28, show_copy_button=True)
        gen_report = gr.JSON(label="report")
        gen_btn.click(
            run_generate,
            [g_prompt, g_episode, g_frame, g_frames, g_steps, g_height, g_width, g_seed],
            [gen_log, gen_video, gen_report],
        )

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    demo.queue().launch()
