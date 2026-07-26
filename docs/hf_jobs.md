# HF Jobs runbook — GPU for the Wan backbone (OD-04/OD-05)

Pay-per-second GPU on Hugging Face infrastructure: our own Docker/UV script, the Wan weights
mounted straight from the Hub, no cluster to manage. This is how M3 work (T-15/T-16) gets a GPU
before there is a training box.

## Why not Inference Providers

Inference Providers (fal, Replicate, Novita, WaveSpeed serve Wan) return a finished **MP4**.
WAM needs the DiT's *internal* residual stream (`features()` -> `[B, S, inner_dim]`) and,
later, gradients for LoRA. Neither is reachable through a hosted generation API, so the
serverless route can only ever answer "does an I2V model imagine plausible manipulation
futures" — useful as a go/no-go probe for the video branch, useless as a backbone.

## One-time setup

```bash
pip install 'huggingface_hub>=0.34'   # or: uv tool install huggingface_hub
hf auth login                          # PRO account + positive credit balance required
```

## Run the smoke test

```bash
python scripts/launch_wan_smoke_job.py --dry-run     # prints the exact hf CLI command
python scripts/launch_wan_smoke_job.py --flavor l40sx1
hf jobs logs <job_id>
```

Equivalent CLI (what `--dry-run` prints):

```bash
hf jobs uv run --flavor l40sx1 --timeout 45m --name wan-smoke --secrets HF_TOKEN \
  -v hf://Wan-AI/Wan2.2-TI2V-5B-Diffusers:/model \
  -v ./src:/wam-src \
  scripts/hf_job_wan_smoke.py -- --source /model --device cuda --frames 5 --height 256 --width 448
```

`-v hf://<repo>:/model` mounts the weights read-only (no download step inside the job);
`-v ./src:/wam-src` syncs this repo's package so the job runs the **real** adapter code, and
the script puts `/wam-src` on `sys.path` before importing `wam`.

What it checks (`scripts/hf_job_wan_smoke.py`): load → derived geometry → `condition_video`
latent shape → `condition_text` → `condition_state` → `features()` token count and finiteness →
determinism across two forwards → `ActionHead.decode` → peak VRAM + wall time. The JSON report
goes to the log; add `--bucket <user/bucket>` to persist it.

## Hardware and cost

| flavor | GPU | $/h | verdict for WAM |
|---|---|---|---|
| `a10g-small` | A10G 24 GB | 1.00 | TI2V-5B smoke test **with `--offload-text`** |
| `l4x1` | L4 24 GB | 0.80 | same, slower |
| `l40sx1` | L40S 48 GB | 1.80 | **default** — 5B fits with the umT5 tower resident |
| `rtx-pro-6000` | 96 GB | 2.75 | best value for 14B LoRA (T-16) |
| `h200` | H200 141 GB | 5.00 | 14B comfortable, multi-GPU variants exist |

Rough footprint of `Wan2.2-TI2V-5B-Diffusers` in bf16: DiT ~10 GB + umT5 encoder ~11 GB +
fp32 VAE ~1 GB. That is why 24 GB flavors need `--offload-text` (the instruction is encoded
once, then the tower moves to CPU). Default timeout is 30 min — the launcher sets 45 min.

A first smoke run is a couple of dollars. Set a budget expectation before scaling to training
runs; every job is billed per second of wall clock, including dependency install (~2 min).

## Model choice (OD-04)

Both Wan generations are **Apache 2.0** — commercially clean.

- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` — 30 blocks, inner dim 3072, 48 latent channels, VAE
  16x16x4. Start here: same I2V capability, half the cost, and what we are validating is the
  adapter's hook logic, not video quality. Auto readout blocks: 15 and 22.
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` — 40 blocks, inner dim 5120, 16 latent channels,
  VAE 8x8x4, plus a CLIP image tower and the 36-channel `[latents, mask, condition]` DiT
  input. The adapter handles both layouts; readout blocks 20/30 (the DreamZero default).

Switch with `--model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --flavor rtx-pro-6000`.

## Frame geometry

Height and width must be multiples of `vae_spatial_stride * patch_size` — 32 px for the 5B
(16x2), 16 px for the 14B (8x2). Token count is
`S = F' * (H/s/p) * (W/s/p)` with `F' = 1 + (F-1)//temporal_stride`; the adapter exposes it as
`expected_token_count()` and the smoke test asserts against it.

## Next steps after a green smoke run

1. Record which blocks give the most action-predictive features (ablate 2-3 pairs — the same
   job with `--blocks`).
2. T-16 on the same infrastructure: LoRA on the DiT + the state projection, checkpoints to a
   storage bucket volume (`-v hf://buckets/<user>/<bucket>:/outputs`), `--timeout 4h`.
3. Compare against the `tiny` backbone on D1 (the T-18 ablation harness) — that is the real
   "does the video branch help" verdict, and it needs real teleop data (D2), not synthetic.
