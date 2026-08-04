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

**Caveat, and why T-26 existed:** every number above was measured *through a mean-pool* over
the token grid. That shows the spatial signal does not survive averaging — not that it is
absent from the backbone. The next section closes that gap: it is absent either way.

## Spatial readouts (T-26 / I-1, 2026-07-29)

**Result: no geometry gain, verdict unchanged.** 10/10 checks on ZeroGPU,
`runs/wan_probe/2026-07-29-zerogpu-5b-readouts.json`. Joints test R², block pair chosen on val:

| readout | width | val | **test** |
|---|---:|---:|---:|
| `mean` | 3 072 | 0.404 | **0.310** |
| `grid2x2` | 12 288 | 0.424 | **0.370** |
| `rand4` (control) | 12 288 | 0.417 | **0.376** |
| `state_only` | 52 | 0.547 | **0.456** |

Gripper: 0.881 state-only against 0.704 for the best readout. On the val-selected pair
`grid2x2` scores 0.338 against the control's 0.3657 — the grid is *behind* random grouping of
the same tokens, so the rise from `mean` to `grid2x2` is width and nothing else. Both bits came
back false:

```
any_geometry_gain_over_control: false
any_spatial_beats_state_only:   false
```

Actual cost: **7.6 s GPU**, 0.079 s/window, 24.6 GB peak VRAM. Geometry check passed in-run
(`S=96, grid (2, 6, 8) implies 96`). Consequence for the roadmap: T-16 keeps its premise, and
re-running T-24 (Cosmos3) with a spatial readout is off the table — its precondition was Wan's
readout moving the number.

### How it works



`--readout` scores several token→vector readouts on the *same* forward passes, with windows,
labels, episode split and ridge code untouched, so the output stays directly comparable to
both tables above:

| readout | what it does | width |
|---|---|---|
| `mean` | the historical mean-pool, byte-for-byte — still reported as `info.probe` | 3 072 |
| `grid<R>x<C>` | average-pool the token grid into R×C cells, kept separate | R·C · 3 072 |
| `rand<N>` | the same tokens in N equally sized **random** groups | N · 3 072 |

The random control is the point: a coarse grid has more dimensions than a mean-pool, and more
dimensions alone can raise a ridge R². `rand<N>` has the identical width and group sizes with
geometry removed, so **grid > rand** means position carries action signal, while **grid ≈ rand**
means we only bought dimensions and the 2026-07-26 verdict stands.

The grid is derived, never guessed: `WanI2VAdapter.token_grid(5, 192, 256)` → **(F'=2, H'=6,
W'=8)**, 96 tokens (VAE spatial 16 / temporal 4, patch (1, 2, 2)), and the probe asserts the
real activation's token count against it before any reshape — `probe.token_count_matches_grid`
fails loudly rather than reshaping garbage. Time is averaged out first for the spatial
readouts; what is under test is space.

```bash
uv run scripts/hf_job_wan_probe.py --source /model --data-dir data/raw/gr00t_apple \
    --readout mean,grid2x2,rand4          # default; grid2x2 = cells of 3x4 tokens
```

On the Space, leave the readout box blank for the default. Read
`info.probe.readout_comparison`: `any_spatial_beats_state_only` and
`any_geometry_gain_over_control` are the two bits this run exists to produce.

**Cost.** The GPU side does not change at all — still one forward per window, all readouts
computed from the same activation (7.6 s measured on ZeroGPU). What grows is the ridge, and it
runs *outside* `@spaces.GPU`, so it burns no quota. Measured on this Mac at the real 96-window /
12-episode layout: ~3 s for `mean`, ~12 s per `grid2x2`-width readout, ~44 s at `grid3x4`
width. The default trio lands around 30 s; going wider than 3×4 cells is the first thing that
actually costs something.

## Backbone bake-off: Cosmos3-Nano probe (T-24, 2026-07-26)

`nvidia/Cosmos3-Nano` (16B MoT: 8B AR reasoner + 8B diffusion generator, robotics-pretrained
*with* action data, OpenMDW-1.1) is diffusers-native since 0.37 — and its VAE **is** the
Wan2.2 VAE (`AutoencoderKLWan`, same 48-ch / 16×spatial / 4×temporal geometry). That made an
apples-to-apples probe cheap: `scripts/hf_job_cosmos3_probe.py` imports the Wan probe's
window building, labels, episode split and ridge code unchanged and only swaps the feature
extractor — windows packed as all-clean conditioning frames (zero noisy tokens, no timestep
embeds) behind the pipeline-tokenized instruction, one forward through the generator tower,
hooks pool the gen-pathway residual stream of all 36 MoT layers.

Deploy: `scripts/deploy_cosmos3_space.py` (private ZeroGPU Space, `diffusers==0.39.0` exact
pin — the probe drives private packing helpers). Run 2026-07-26: **9/9 checks**, 96 windows,
features `(96, 36, 4096)`, load 8.1 s, 0.106 s/window forward, peak VRAM 36.2 GB, ~33 s GPU
wall. Full table: `runs/cosmos3_probe/2026-07-26-zerogpu-nano.json`.

| features | joints test R² | gripper test R² |
|---|---|---|
| heuristic (18, 26) — best Cosmos3 pair | 0.359 | 0.708 |
| depth-scaled Wan pick (2, 12) | 0.327 | 0.706 |
| suggested (11, 24) — top-2 by val R² | 0.324 | 0.613 |
| best single block (15 joints / 17 gripper) | 0.399 | 0.822 |
| state-only ridge (same as Wan probe) | **0.456** | **0.881** |
| *Wan2.2-TI2V-5B best pair (2, 10), for reference* | *0.365* | *0.698* |

Verdict: **frozen Cosmos3 features do not beat the state-only ridge either → stay on Wan for
the T-16 LoRA.** The robotics pretraining is visible but small: gripper readability is
clearly above Wan's (0.822 vs. 0.698 single-block) and approaches the state-only ceiling,
while joints land in the same ~0.33–0.40 band as Wan. Note the top-2-by-val pair (11, 24)
*overfits the val episodes* (val 0.465 → test 0.324) — with 2 val episodes, pair selection
by val R² is noisy; the depth heuristic generalized better here. Neither backbone's frozen
prior linearly encodes next-chunk actions, so the earlier conclusion stands unchanged: the
action value must come from fine-tuning, and Wan stays primary (Apache 2.0, 5B vs. 16B —
cheaper to LoRA and to serve). Cosmos3 remains the fallback candidate if the Wan LoRA
underdelivers; the probe harness reruns against any diffusers-native backbone.

**Qualitative side-by-side (2026-07-26):** the Cosmos3 script also has a `--generate` mode
(mirror of the Wan probe's, "generate future" tab on the same Space) that sampled the two
`wan_futures/` prompts from the identical start frame (49 frames 640×480, 35 steps, ~46 s
sampling, peak 35.6 GB with tiled VAE decode) → `runs/presentation/cosmos3_futures/`,
mirrored to `huhn511/wam-presentation`. Both clips follow their instruction (apple lands on
the plate / plate pushed left, apple stays) with plausible physics — but Cosmos3 invents a
generic white manipulator instead of the G1, the same embodiment gap Wan shows. Consistent
with the probe verdict: the priors bring physics and instruction-following; the embodiment
must come from LoRA fine-tuning (T-16). Presentation cut: `runs/presentation/
wam_04_futures_sidebyside{,_en}.mp4` (21 s, both futures side by side, rebuild via
`runs/presentation/build/make_futures_video.py`). A follow-up conditioned on a
hand-visible start frame (episode 0, frame 150 — Dex3 hand at the apple;
`faithful_hand_visible.mp4` in both futures dirs) confirms the mechanism: both priors
keep the visible black hand through the grasp but invent the never-seen arm behind it
(Wan a green-white tube, Cosmos3 a bulky black-silver mechanism + rod artifact) —
visible embodiment is preserved, unseen embodiment is hallucinated, LoRA still needed.
Presentation cut: `runs/presentation/wam_05_futures_handvisible{,_en}.mp4` (18 s, rebuild
via `runs/presentation/build/make_handvisible_video.py`). A second follow-up swaps the
single conditioning frame for **video context**: `Cosmos3OmniPipeline` supports
Video2World natively (clean leading latent frames), exposed as `--gen-cond-frames` /
"cond frames" on the Space's generate tab; Wan2.2-TI2V-5B has no such mode (image +
optional last-image only) — a genuine capability edge for the fallback. First tries
conditioned on 9 frames (142–150, nearly static: mean abs pixel diff 2.5/255) and 33
frames (118–150, 1.1 s) were too short — the motion window matters; the kept run uses
the 97 real frames 54–150 (3.2 s: the hand entering + the whole approach), 169 output
frames = 97 real + 72 predicted (same prompt/seed, 175 s sampling, peak 36.9 GB — the
Space's 420 s GPU cap leaves room). Result: the black hand stays correct through the
grasp, no rod artifact, the apple lands on the plate — but once the never-seen arm
must enter the frame, Cosmos3 still invents a white manipulator with cables
(`cosmos3_futures/faithful_video_conditioned.mp4`). Pattern: more real context buys a
longer faithful prediction; invention is confined to what the context has not shown.
Presentation cut: `runs/presentation/wam_06_futures_videoconditioned{,_en}.mp4` (22 s,
rebuild via `runs/presentation/build/make_videocond_video.py`).

The third follow-up runs the whole task as **closed-loop chunks** (the FR-05 pattern
against fixed footage): 5 Video2World runs, each conditioned on the 97 real frames
ending at an anchor (150/210/270/330/390), each predicting 48 frames (2 s, the PRD
chunk length; 145-frame canvas, ~165 s sampling per chunk, one ZeroGPU queue timeout
retried). Conditioning verified at VAE-roundtrip level (1.3–2.0/255) on every chunk.
Stitched (`cosmos3_futures/chunks/stitched_fulltask.mp4`, predicted 24 fps segments
resampled to 30 fps) this covers the full 15-s task time-aligned with the real episode
(`stitched_vs_real.mp4` is the side-by-side). Findings: (1) it corrects an earlier
claim — the G1 arm *does* enter the ego view from ~frame 200 on, and every context
window that contains it yields the true embodiment (silver arm, Dex3 hand, grasp,
place, retreat all plausible); only chunk 1, whose context predates the arm, still
invents a white manipulator. (2) Re-anchor seams show drift of 14–39/255 — e.g. chunk 1
slides the apple instead of lifting it — and the next anchor pulls the prediction back:
predict, re-observe, re-anchor is exactly what the runtime loop will do. (3) Generated
re-creations of recorded demos are diagnosis/presentation material, not training data —
the LoRA (T-16) on real demos stays the way to close the fresh-start embodiment gap.
Presentation cut: `runs/presentation/wam_07_fulltask_chunked{,_en}.mp4` (30 s, rebuild
via `runs/presentation/build/make_chunked_video.py`).

## Watching a fine-tune's video prior (T-16 LoRA on the generate tab, 2026-07-30)

Every generation above ran the **base** prior. Once T-16 produced a checkpoint, the same tab can
run the fine-tuned one — the "generate future" tab grew a **LoRA repo/path** box and a **scale**
slider, and `--gen-lora` / `--gen-lora-scale` do the same on the CLI.

This costs no sampler. `WanFlowBackbone.save_adapter` was always meant to be portable ("it
loads into a stock diffusers Wan pipeline, so a checkpoint can be watched as video without
WAM"), but it needs the 19 GB base resident just to walk a module tree.
`scripts/export_lora.py` does the same conversion as a pure key rename, on a laptop, from the
330 MB checkpoint alone:

```bash
.venv/bin/python scripts/export_lora.py \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors \
    --out runs/t16-lora-seed0/lora-diffusers
# 600 tensors, r=32 alpha=64, 322.5 MB fp32
```

Three transforms: `backbone.lora.blocks__0__attn1__to_q__lora_A__wam__weight` →
`blocks.0.attn1.to_q.lora_A.weight` (undo `WanFlowBackbone`'s `nn.ParameterDict` mangling, drop
the adapter name); the peft `LoraConfig` into the safetensors `lora_adapter_metadata`; and
`state_proj` to its own file.

**The metadata is load-bearing, and this is the trap.** `load_lora_adapter` reconstructs the
config from the weights when no metadata is present, and infers `lora_alpha = r`. We train at
`alpha=64, r=32` — scaling 2.0 — so an adapter without it loads at **half** the strength it was
trained with and reads as a weak fine-tune. Measured both ways:

| | rebuilt `r` | rebuilt `lora_alpha` | scaling |
|---|---|---|---|
| with `lora_adapter_metadata` | 32 | 64 | **2.0** ✓ |
| without | 32 | 32 | 1.0 ✗ |

`save_lora_adapter` writes that key too, so the resume path was never affected — only a
hand-rolled export would have been.

Loading goes through `pipe.transformer.load_lora_adapter(..., prefix=None)`, not the
pipeline-level `load_lora_weights`: the exported keys are model-relative, which is exactly what
`WanI2VAdapter.load_lora` reads back. Same file, same loader, so the Space shows what a resume
would restore. `peft` had to join `requirements.txt`, and `--set-hf-token` on
`deploy_wan_space.py` puts an `HF_TOKEN` secret on the Space so it can read a private adapter
repo (a Space carries no token of its own).

**What this can and cannot answer.** The sharp question is embodiment: the base prior has never
seen a G1 and invents a generic manipulator for the arm it cannot see (above). Same start frame,
same prompt, same seed, scale 0 vs 1 attributes every difference to training. What it cannot
answer is competence — three reasons, all worth stating before anyone reads a nice clip as
progress:

- the adapter trained at **9 frames of 128×160**; generating 49 frames at 480×640 is
  extrapolation, and the trained geometry is recorded in `wam_provenance.json` for that reason;
- the video loss was deliberately downweighted to **0.5** so the latent flow residual would not
  dominate the action branch (R-07) — fidelity was never the objective;
- the DiT trained with **proprioception tokens concatenated onto the text context**
  (`wan_i2v.py:605`), which a diffusers pipeline cannot supply, so this runs without them.

And the action branch is not in the file at all. On WAM-Bench the full checkpoint reaches L0 and
loses to repeat-last-action: 48.4/100 and −32.4 % as published (tiled, i.e. one frame repeated
nine times), 50.6/100 and −21.8 % once T-29 re-scored it with a real frame window. Either way a
plausible-looking video is not evidence the policy works. This is a diagnostic, not a demo; `runs/presentation/` remains the
presentation material.

### Measured (2026-07-30): the fine-tune is geometry-bound, and it learned to stand still

Five ZeroGPU calls, episode 0 / frame 150 (Dex3 hand at the apple, arm not yet in view — where
the base prior has to invent it), same prompt and seed 0 throughout. Artifacts in
`runs/presentation/t16_lora_futures/`. `motion` is the mean absolute frame-to-frame difference
in 0–255.

| geometry | LoRA scale | motion | what it looks like |
|---|---|---|---|
| 49 × 480×640 | 0.0 (base) | 2.93 | coherent grasp + place, **green-white tube** for the arm |
| 49 × 480×640 | 0.35 | 2.96 | coherent grasp + place, **white/silver segmented arm** |
| 49 × 480×640 | 1.0 (as trained) | 1.57 | collapses after ~frame 5: plate becomes a blob, contrast washes out |
| 9 × 128×160 | 0.0 (base) | 29.48 | psychedelic colour noise — **the base cannot generate here at all** |
| 9 × 128×160 | 1.0 (as trained) | 0.73 | clean, stable, correct scene — and almost nothing moves |

Three findings, and the third is the one that matters:

1. **The embodiment gap does close, at reduced strength.** Scale 0.35 keeps the base's structure
   (motion 2.96 vs 2.93, std 53 vs 55) and replaces the hallucinated green tube with a
   white/silver segmented arm — much closer to the G1's actual silver arm. Everything else was
   held fixed, so that difference is attributable to training alone. Effective scaling at
   "1.0" is 2.0 (alpha/r), so 0.35 is ~0.7× the trained delta.
2. **Full strength is out of distribution, not broken.** At 4× the trained resolution and 5× the
   length the adapter destroys the prior; at its own 9 × 128×160 it is the *base* that produces
   garbage while the adapter renders a clean, stable scene. Neither model is bad — each is bad
   away from where it trained. Any future presentation clip should use a low scale.
3. **It learned to predict almost no change.** In-distribution motion is **0.73** where the base
   manages 29.5 and a real 0.3 s of demonstration moves considerably more. That is the same
   pathology WAM-Bench found on the action side — `smoothness_ratio` 0.293, i.e. 3.4× smoother
   than the demonstrations, which is why the model loses to repeat-last-action. The video branch
   and the action branch failed the *same* way: both converged on "nothing much happens next".

   > **Correction (2026-08-05, T-35).** "a real 0.3 s of demonstration moves considerably more" is
   > the load-bearing clause of this finding and it had **no number behind it**. Measured since
   > (`scripts/dream.py --gt-only`, 200 clips over 40 episodes): real 9-frame clips move **0.958**
   > at 120×160 and **0.922** on the 128×160 grid a decoded clip lives on, with 68.9 % of frame
   > pairs moving under one grey level. This corpus barely moves — 0.73 is roughly **0.8×** the
   > recorded motion, not ~0, and the 29.5 it was contrasted with is this same table's
   > "psychedelic colour noise", an upper bound on noise rather than on plausible motion.
   > That does not make the finding false: these clips came from `generate_future`, i.e. a stock
   > diffusers pipeline with **no state port**, so they were generated without the proprioception
   > token the DiT trained with, and no like-for-like arm existed. It makes the finding
   > **unsupported as stated**. `docs/preregistration/PR-05-dream-motion.md` fixes the threshold
   > (≥ 0.5 of the VAE round-trip) before the run that can settle it.

The per-block adapter magnitudes agree. `lora_B` starts at exactly zero in peft and 0 % of it is
still zero, so training moved everything — but blocks 23–27 moved most (mean |B| 1.2e-2) and
block 10 least (7.0e-3). Blocks 11–29 receive gradient *only* from the video loss (the readouts
are at 2 and 10), so the downweighted video objective still reshaped the adapter more than the
action objective did. That is a lead for the next run: the action losses are not driving this
fine-tune.

## Next steps after a green smoke run

1. ~~Record which blocks give the most action-predictive features~~ — done twice: label-free
   ablation said (20, 29), label-validated probes overturned it → (2, 10) in
   `configs/model/wan22_ti2v_5b.yaml`, provisional until LoRA.
2. T-16 on the same infrastructure: LoRA on the DiT + the state projection, checkpoints to a
   storage bucket volume (`-v hf://buckets/<user>/<bucket>:/outputs`), `--timeout 4h`.
3. Compare against the `tiny` backbone on D1 (the T-18 ablation harness) — that is the real
   "does the video branch help" verdict, and it needs real teleop data (D2), not synthetic.
