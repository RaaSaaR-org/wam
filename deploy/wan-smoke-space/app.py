"""ZeroGPU Space: run the WAM Wan-backbone smoke test on a free PRO GPU (T-15, OD-04/OD-05).

Same checks as an HF Jobs run — this file only supplies the ZeroGPU-shaped harness around
`smoke.py`, which is deployed verbatim from `scripts/hf_job_wan_smoke.py` so there is exactly
one implementation of the checks.

Two constraints shape the design:

- A Space has ~16 GB host RAM but the Wan repo is ~34 GB (fp32 transformer). The model is
  therefore loaded with `--device-map cuda`, which streams shards straight to the GPU instead
  of materializing them in RAM first.
- ZeroGPU only exposes a real GPU *inside* `@spaces.GPU`, and releases it on return. Loading
  must happen in the same call as the forward pass; the weights cannot survive between calls.
  Model download runs outside the decorator, so it costs no GPU quota.
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
GPU_DURATION = int(os.environ.get("GPU_DURATION", "240"))
DEFAULT_INSTRUCTION = "pick up the red cube and place it in the bin"
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


def run(frames: int, height: int, width: int, blocks: str, instruction: str):
    """Download on CPU (free), then hand off to the GPU. Yields progress into the UI."""
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


with gr.Blocks(title="WAM · Wan backbone smoke test") as demo:
    gr.Markdown(
        f"""# WAM — Wan backbone smoke test

Does the real Wan DiT produce usable action-readout features through the WAM interfaces,
with the shapes `WanI2VAdapter` claims? Nothing is trained here.

Model: `{MODEL_ID}` · checks: `scripts/hf_job_wan_smoke.py` from
[RaaSaaR-org/wam](https://github.com/RaaSaaR-org/wam).
First run downloads ~34 GB before the GPU is requested — expect several minutes.
"""
    )
    with gr.Row():
        frames = gr.Number(value=5, label="frames", precision=0)
        height = gr.Number(value=256, label="height", precision=0)
        width = gr.Number(value=448, label="width", precision=0)
    blocks = gr.Textbox(value="", label="readout blocks (blank = auto, mid/late depth)")
    instruction = gr.Textbox(value=DEFAULT_INSTRUCTION, label="instruction")
    button = gr.Button("Run smoke test", variant="primary")
    log = gr.Textbox(label="log", lines=28, max_lines=28, show_copy_button=True)
    report = gr.JSON(label="report")
    button.click(run, [frames, height, width, blocks, instruction], [log, report])

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    demo.queue().launch()
