"""ZeroGPU Space: Cosmos3-Nano frozen-feature probe on real GR00T-G1 data (T-24).

One tab, one deployed implementation — `cosmos_probe.py` is `scripts/hf_job_cosmos3_probe.py`
deployed verbatim (which itself reuses `probe.py` = the Wan probe's windows/labels/ridge
machinery, so the two backbones are scored by the same code on the same data).

Same shape as the Wan Space: downloads (model + episodes) and the ridge analysis run on
CPU outside `@spaces.GPU`; only VAE encodes + the 36-layer MoT forwards spend GPU quota.
The transformer (~32 GB bf16) loads with device_map=cuda so accelerate streams shards
straight to the GPU.
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

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Cosmos3-Nano")
DATA_REPO = os.environ.get("DATA_REPO", "nvidia/GR00T-N1.7-AppleToPlate")
GPU_DURATION = int(os.environ.get("GPU_DURATION", "360"))
# The probe needs transformer/vae/text_tokenizer/scheduler only — not the reasoner's
# vision encoder, the sound tokenizer, or the model-card demo assets.
IGNORE = [
    "assets/*",
    "images/*",
    "vision_encoder/*",
    "sound_tokenizer/*",
    "model.safetensors.index.json",
    "*.mp4",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
]


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
def probe_on_gpu(
    args: Any, windows: list[dict[str, Any]], instruction: str
) -> tuple[str, Any, Any]:
    """One frozen MoT forward per window -> pooled per-block features [N, blocks, D]."""
    import cosmos_probe

    buffer = io.StringIO()
    started = time.perf_counter()
    report = cosmos_probe.smoke.Report()
    pooled = None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            pooled = cosmos_probe.extract_features(args, windows, instruction, report)
    except Exception:  # noqa: BLE001 - a crashed run must still surface its log in the UI
        buffer.write("\n" + traceback.format_exc())
    buffer.write(f"\ngpu wall: {time.perf_counter() - started:.1f}s")
    return buffer.getvalue(), pooled, report


def run_probe(episodes: int, windows_per_ep: int, frames: int, height: int, width: int,
              chunk_steps: int):  # fmt: skip
    """Probe tab: data + windows on CPU, features on GPU, ridge analysis on CPU."""
    import cosmos_probe

    log: list[str] = [f"host: {json.dumps(host_info())}", f"model: {MODEL_ID}"]
    log.append(f"data: {DATA_REPO}")
    yield "\n".join(log), None

    try:
        log.append("\ndownloading weights + episodes (no GPU quota consumed, ~32 GB first run)…")
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
            "--out", "/tmp/cosmos3_probe_report.json",
        ]  # fmt: skip
        args = cosmos_probe.parse_args(probe_argv)
        t0 = time.perf_counter()
        windows, instruction, data_info = cosmos_probe.wanprobe.build_windows(args)
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
            cosmos_probe.wanprobe.analyze_probes(pooled, windows, args, report)
            code = cosmos_probe.finalize(report, args.out)
        log.append(buffer.getvalue())
        payload = dict(cosmos_probe.LAST_REPORT)
        payload["exit_code"] = code
        yield "\n".join(log), payload
    except Exception:  # noqa: BLE001 - quota exhaustion / bad inputs land here
        log.append(traceback.format_exc())
        yield "\n".join(log), {"ok": False, "error": "probe run failed"}


with gr.Blocks(title="WAM · Cosmos3 readout probes on ZeroGPU") as demo:
    gr.Markdown(
        f"""# WAM — Cosmos3-Nano readout probes on ZeroGPU (T-24)

Model: `{MODEL_ID}` · real data: `{DATA_REPO}` · code:
[RaaSaaR-org/wam](https://github.com/RaaSaaR-org/wam) (`cosmos_probe.py` / `probe.py`
deployed verbatim from `scripts/`). Same windows, labels and ridge machinery as the Wan
probe — the question is whether Cosmos3's robotics-pretrained generator features beat the
state-only floor that Wan's could not. First run downloads ~32 GB before any GPU is
requested.
"""
    )
    with gr.Row():
        p_episodes = gr.Number(value=12, label="episodes", precision=0)
        p_windows = gr.Number(value=8, label="windows / episode", precision=0)
        p_frames = gr.Number(value=5, label="context frames", precision=0)
    with gr.Row():
        p_height = gr.Number(value=192, label="height", precision=0)
        p_width = gr.Number(value=256, label="width", precision=0)
        p_chunk = gr.Number(value=16, label="chunk steps (label)", precision=0)
    probe_btn = gr.Button("Run readout probes", variant="primary")
    probe_log = gr.Textbox(label="log", lines=28, max_lines=28, show_copy_button=True)
    probe_report = gr.JSON(label="report")
    probe_btn.click(
        run_probe,
        [p_episodes, p_windows, p_frames, p_height, p_width, p_chunk],
        [probe_log, probe_report],
        api_name="run_probe",
    )

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    demo.queue().launch()
