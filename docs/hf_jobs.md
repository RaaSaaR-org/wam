# Rented GPU for the Wan backbone (OD-04/OD-05)

Two ways to get a GPU on Hugging Face infrastructure before there is a training box. Both run
the same checks — `scripts/hf_job_wan_smoke.py` is the single implementation.

| | HF Jobs | ZeroGPU Space |
|---|---|---|
| cost | $0.80–5.00/h, per second | **free** with PRO (40 min/day) |
| needs | pre-paid credit balance | PRO subscription |
| hardware | any flavor up to 8x H200 | RTX Pro 6000, 48 GB (`large`) |
| session | up to days | one `@spaces.GPU` call |
| good for | T-16 LoRA training, long runs | T-15 smoke test |
| launch | `scripts/launch_wan_smoke_job.py` | `scripts/deploy_wan_space.py` |

**PRO alone does not fund Jobs.** The $2/month included with PRO is Inference Providers
credit; Jobs bills per minute against a separate pre-paid balance and returns
`402 Payment Required` when it is empty. ZeroGPU is the one that is genuinely included.

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

## ZeroGPU Space (free path)

```bash
python scripts/deploy_wan_space.py --dry-run
python scripts/deploy_wan_space.py            # creates a private <user>/wam-wan-smoke
```

Source in `deploy/wan-smoke-space/`; `scripts/hf_job_wan_smoke.py` is uploaded verbatim as the
Space's `smoke.py`, and `wam` is installed from the public GitHub repo — so **push before you
deploy**, or the Space runs an older adapter. Two constraints shape the app:

- The checkpoint is ~34 GB (the transformer ships fp32) against a documented 16 GB Space RAM
  default, so the model loads with `--device-map cuda` and accelerate streams shards straight
  to the GPU. A measured ZeroGPU host is much larger than that default — 104 GB and 192 cores,
  1.9 TB free disk, weights downloaded in 26 s — so treat this as insurance, not a hard
  requirement. Load time was 7.4 s.
- ZeroGPU only exposes a real GPU inside `@spaces.GPU` and reclaims it on return, so load and
  forward pass share one call (`GPU_DURATION`, default 240 s). The weight *download* happens
  outside the decorator and costs no quota.

Quota is 40 min/day on PRO, 2× if a call requests `size="xlarge"` (96 GB). Beyond the daily
quota it bills pre-paid credits at $1 per 10 min — the same balance Jobs needs.

## Verified run (2026-07-26)

`Wan2.2-TI2V-5B-Diffusers` on a ZeroGPU RTX PRO 6000 Blackwell (MIG 2g.48gb, 51 GB):
**13/13 checks passed**.

| | |
|---|---|
| derived geometry | 30 blocks, inner dim 3072, 48 latent ch, text dim 4096, no CLIP tower |
| readout blocks | 15 / 22 (auto, mid+late depth) |
| latents (5x256x448) | `[1, 48, 2, 16, 28]` |
| features | `[1, 224, 3072]`, finite, std 0.71, bit-identical across two forwards |
| load / VAE encode / DiT forward | 7.1 s / 0.37 s / 0.08 s |
| peak VRAM | 24.3 GB (26.3 GB reserved) |

The 12 ms/token-block forward is what matters for FR-05: the DiT pass is nowhere near the
2 Hz policy-rate budget, so the closed loop is limited by chunk length, not the backbone.

Two things this caught that stub tests could not — see the git history for both: the Wan 2.2
VAE compresses 16x, not the 8x derived from `temperal_downsample`; and a Space rebuild silently
reused a pip-cached `wam` wheel, so the first "fixed" deploy still ran the old adapter.

## Readout ablation (2026-07-26)

`--ablate` (the Space's "ablate readout blocks" checkbox) probes **every** DiT block in the
same load: four forwards — base, frame order reversed, different instruction, different robot
state — and per block the relative L2 distance to the base features. One ZeroGPU call covers
the 13 smoke checks plus the ablation (18/18 passed, 9.3 s GPU wall, ablation itself 0.7 s).
Full table: `runs/wan_ablation/2026-07-26-zerogpu-5b.json`.

Result on `Wan2.2-TI2V-5B` (30 blocks):

| blocks | motion | instruction | state | combined score |
|---|---|---|---|---|
| 0–19 | 0.28–0.39 | 0.02–0.05 | 0.010–0.020 | 0.02–0.26 |
| **20–24** | 0.26–0.37 | **0.07–0.11** | **0.025–0.030** | 0.34–0.58 |
| 25–28 | 0.32–0.43 | 0.03–0.04 | 0.022–0.025 | 0.20–0.38 |
| **29** | **0.56** | 0.09 | **0.057** | **0.93** |

Three readings: every conditioning path is alive (each probe moves features somewhere);
instruction/state integration *jumps* at blocks 20–24 and again in the final block; the depth
heuristic default (15, 22) sits below both. Measured pick was **(20, 29)** — but this is a
label-free proxy, and the label-validated probes below overturned it.

## Label-validated readout probes (2026-07-26)

`scripts/hf_job_wan_probe.py` (Space tab "readout probes", endpoint `/run_probe`) closes the
loop the ablation left open: 96 windows (12 GR00T-AppleToPlate episodes × 8 windows, 5 frames
at 192×256), one frozen DiT forward each, per-block token-pooled features, ridge regression
onto the next 16-step canonical action chunk. Honest evaluation: episode-level split
(train 0–6 / val 7–8 / test 9–11), alpha chosen on val only, joints and gripper scored
separately (their label scales differ by ~50×). 8/8 checks, 73 s wall on ZeroGPU.
Full table: `runs/wan_probe/2026-07-26-zerogpu-5b.json`.

| features | joints test R² | gripper test R² |
|---|---|---|
| **suggested (2, 10)** — top-2 by val R² | **0.365** | **0.698** |
| heuristic (15, 22) | 0.341 | 0.629 |
| ablation pick (20, 29) | 0.335 | 0.522 |
| state-only ridge (raw q/dq/gripper, 32-dim) | 0.456 | 0.881 |

Two findings. First, the label-free sensitivity ranking does **not** predict action
readability: with real labels, *early* blocks (2, 10) win, and (20, 29) is the worst of the
three pairs — `configs/model/wan22_ti2v_5b.yaml` now records (2, 10). Second, and more
important: **no frozen video features beat the trivial state-only baseline.** A 32-dim raw
state vector out-predicts 6144-dim frozen Wan features on both label groups. The frozen
prior alone does not linearly encode next-chunk actions — the action value has to come from
fine-tuning (LoRA, T-16), which can also re-rank the blocks on adapted features.

## Next steps after a green smoke run

1. ~~Record which blocks give the most action-predictive features~~ — done twice: label-free
   ablation said (20, 29), label-validated probes overturned it → (2, 10) in
   `configs/model/wan22_ti2v_5b.yaml`, provisional until LoRA.
2. T-16 on the same infrastructure: LoRA on the DiT + the state projection, checkpoints to a
   storage bucket volume (`-v hf://buckets/<user>/<bucket>:/outputs`), `--timeout 4h`.
3. Compare against the `tiny` backbone on D1 (the T-18 ablation harness) — that is the real
   "does the video branch help" verdict, and it needs real teleop data (D2), not synthetic.
