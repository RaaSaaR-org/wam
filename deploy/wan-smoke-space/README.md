---
title: WAM Wan Backbone Smoke Test
emoji: 🤖
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
short_description: Verify the WAM Wan backbone adapter on a real GPU
models:
  - Wan-AI/Wan2.2-TI2V-5B-Diffusers
---

# WAM — Wan backbone smoke test (ZeroGPU)

Verification harness for [`RaaSaaR-org/wam`](https://github.com/RaaSaaR-org/wam) task T-15:
does the real Wan DiT produce usable action-readout features through the WAM interfaces,
with the shapes `WanI2VAdapter` claims? Nothing is trained here.

`smoke.py` is deployed verbatim from `scripts/hf_job_wan_smoke.py` in that repo, so this Space
and an HF Jobs run execute exactly the same checks:

load → derived geometry → `condition_video` → `condition_text` → `condition_state` →
`features()` token count / finiteness / non-constancy → determinism across two forwards →
`ActionHead.decode` → peak VRAM and wall time.

## Why it is built this way

- **A ~34 GB checkpoint.** The model loads with `--device-map cuda`, so accelerate streams
  shards straight to the GPU rather than materializing them in host RAM — required against the
  documented 16 GB Space default, and ~7 s to load either way. (Measured, a ZeroGPU host is
  much larger: 104 GB, 192 cores. The `host` line in the log records what you actually got.)
- **ZeroGPU only exposes a GPU inside `@spaces.GPU`** and reclaims it on return, so loading
  and the forward pass share one call. The download runs outside the decorator and costs no
  GPU quota.

## Configuration

Set as Space variables: `MODEL_ID` (default `Wan-AI/Wan2.2-TI2V-5B-Diffusers`; the adapter
also handles the 40-block `Wan2.1-I2V-14B-480P-Diffusers` layout) and `GPU_DURATION`
(seconds, default 240).

Frame height and width must be multiples of `vae_spatial_stride * patch_size` — 32 px for the
5B, 16 px for the 14B.
