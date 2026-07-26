---
title: WAM Cosmos3 Readout Probes
emoji: 🌌
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
short_description: Cosmos3-Nano frozen features vs. state-only on real G1 data
models:
  - nvidia/Cosmos3-Nano
datasets:
  - nvidia/GR00T-N1.7-AppleToPlate
---

# WAM — Cosmos3-Nano readout probes on ZeroGPU (T-24)

Backbone bake-off harness for [`RaaSaaR-org/wam`](https://github.com/RaaSaaR-org/wam).
`cosmos_probe.py` (= `scripts/hf_job_cosmos3_probe.py`) and `probe.py` (= the Wan probe)
are deployed verbatim, so both backbones are scored by the same windows, labels, split and
ridge code on the same real data.

The experiment: real GR00T-G1 ego windows are VAE-encoded (the Cosmos3 VAE **is** the
Wan2.2 VAE) and packed as clean conditioning frames after the tokenized instruction; one
joint MoT forward per window; hooks on all 36 layers collect the generation-pathway
residual stream at the vision tokens, token-pooled per block, then ridge-regressed onto
the BC-relabeled 16-step action chunks with an episode-level split. A state-only ridge
(raw q/dq/gripper) is the floor to beat.

Why: on Wan2.2-TI2V-5B **no frozen video features beat state-only** (probe 2026-07-26).
Cosmos3 is pretrained on robot video *with actions* — if any frozen prior linearly encodes
next-chunk actions, it is this one. Beating state-only makes Cosmos3 the primary WAM
backbone candidate; failing keeps the burden on LoRA fine-tuning (T-16).

## Why it is built this way

- **A ~32 GB bf16 transformer.** Loads with `device_map=cuda` so accelerate streams shards
  straight to the GPU. Only `transformer/`, `vae/`, `text_tokenizer/` and `scheduler/` are
  downloaded — not the reasoner's vision encoder or the sound tokenizer.
- **ZeroGPU only exposes a GPU inside `@spaces.GPU`** and reclaims it on return: downloads,
  window building and the ridge analysis run outside the decorator and cost no quota.

## Configuration

Space variables: `MODEL_ID` (default `nvidia/Cosmos3-Nano`) and `GPU_DURATION` (seconds,
default 360). Frame height/width must be multiples of 32 (VAE 16× spatial · patch size 2).
