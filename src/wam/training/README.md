# `wam.training` — losses, datasets, trainers, monitoring

**TL;DR** — The torch domain. Two trainers: an **action-only baseline** (M2, the thing the world
model must beat) and the **joint video+action** flow-matching trainer (M3/T-16). Plus the loss
library, an episode-backed dataset, and divergence monitoring.

## Files

| File | Contains |
|------|----------|
| `losses.py` | All loss terms (PRD §10.4) — plain tensor functions, no module state |
| `datasets.py` | `EpisodeDataset` + `collate_episode_batch` |
| `action_only.py` | `ActionOnlyModel` / `ActionOnlyTrainer` — the M2 overfit gate (T-13) |
| `joint.py` | `JointWorldActionModel` / `JointTrainer` — co-denoising (T-16) |
| `monitor.py` | `TrainingMonitor`, `TrainingDiverged` (T-17, R-07) |
| `_utils.py` | batching, instruction encoding, safetensors checkpoints (private) |

## `losses.py`

Torch in, scalar tensor out; deterministic and differentiable. Reductions are means over
contributing elements, so magnitudes stay comparable across batch sizes and chunk lengths.

| Function | Purpose |
|---|---|
| `make_flow_targets(x0, x1, t)` | rectified-flow: `x_t = (1-t)·x0 + t·x1`, target `v = x1 - x0` |
| `video_flow_loss` | velocity MSE in video latent space |
| `action_flow_matching_loss` | velocity MSE on action latents (optional step mask) |
| `action_regression_loss` | direct L1/L2 on chunk targets — the action-only objective |
| `alignment_loss` | **cosine**, not InfoNCE (see below) |
| `smoothness_loss` | second-difference (jerk) penalty |
| `limit_penalty` | soft squared hinge outside ±limit |

Two choices worth knowing. **Alignment is cosine** because a contrastive objective needs large
batches and a temperature to mean anything, and D1 overfit batches are tiny; `1 - cos(v, a)` is
stable at any batch size. **Smoothness is a second difference**, so constant and linear ramps
cost exactly zero — intentional motion is not penalized, only jerk. `limit_penalty` is a
training regularizer only; hard limits stay in the safety layer.

## `datasets.py`

`EpisodeDataset` windows episode directories into supervised samples: **one sample per recorded
commanded chunk**. The observation is the last frame and last state at or before the chunk's
command timestamp; the frame window is the `num_frames` frames ending there, left-padded by
repeating the first frame.

`chunk_steps` fixes T — longer chunks are truncated, **shorter ones skipped** (deterministic,
documented). `None` keeps native lengths, which then requires equal T per batch.

Two guards fail **loudly** rather than training on wrong data:

1. An episode whose manifest declares a **non-identity `NormalizationSpec`** for action targets
   is rejected — nothing in the pipeline would apply it, so the declared stats would be silently
   ignored.
2. Per-step `|targets|` must stay **< 1.0** — the shipped decoders are tanh-bounded, so larger
   deltas are unreachable and would silently floor the training loss.

`wam.data` is imported lazily inside methods; construction stores paths only, decoding happens
on first `len()`/index.

## `action_only.py` — the baseline

`StateMLP` + `TinyVideoBackbone` features + `ActionHead`, regressing chunks directly from
(frames, instruction, state). **No video prediction** — this is what the world-action model has
to beat (AC-07). It implements the `Policy` protocol, so it drops straight into the runtime;
`predict` tiles the configured camera's single frame to the backbone's context length.

`ActionOnlyTrainer` is a plain seeded AdamW loop with gradient clipping, deterministic on CPU.
`overfit(data, steps, target_loss)` is the D1 go/no-go gate (P6: overfit first, scale later).

## `joint.py` — co-denoising

Three branches sharing one backbone forward:

- **video** — rectified flow on the backbone's video latents (for tiny: pixels, identity VAE)
  via `backbone.forward_flow` → velocity prediction + shared features.
- **action** — demonstrated chunks embedded by `ActionChunkEncoder`, noised with the **same
  schedule and the same `t`** as the video latents (that is what "co-denoised" means); an
  `ActionVelocityHead` predicts the action-latent velocity from `[z_t | pooled features | t]`.
- **decoder** — `ActionHead` regresses the clean chunk from the shared features (inverse
  dynamics), so the smoothness and limit regularizers act in action space.

**The collapse guard.** The flow input and target are built from a **detached** copy of the
action latent. If gradients flowed from the flow target back into the encoder, AdamW could
collapse the encoder to a constant `c` — then `v_target = c - noise` is an exact function of
`(z_t, t)`, `action_flow → 0`, and the latent encodes *nothing about the action*. The encoder is
instead anchored by a small reconstruction decoder (`action_recon`) that regresses
(targets, gripper) back from the clean latent, forcing the latent to stay action-informative.

**Frozen parts.** Text/VAE equivalents are frozen at construction (PRD §10.3 step 4). For tiny
that is the text embedding table + text positional table; the tiny "VAE" is the identity and has
no parameters. The registry is `JointWorldActionModel.frozen_parts`, and
`frozen_parameter_names()` exposes it for audits. The optimizer only ever sees trainable params.

`JointTrainer` optimizes the weighted sum: video / action_flow / action_recon / action_reg /
gripper / alignment / smoothness / limit. Weights are the R-07 tuning surface.

## `monitor.py`

Passive: trainers call `record_step` once per optimizer step. It raises `TrainingDiverged` when

- **any** loss component is NaN/Inf → immediately, or
- total loss > `divergence_factor × EMA(total)` after `warmup_steps` (bias-corrected EMA).

Per-module gradient norms exist for **R-07** specifically: comparing `grad/backbone` against
`grad/action_head` is how branch imbalance — the video loss drowning out action learning —
becomes visible. `param_update_ratio` tracks `‖Δθ‖ / ‖θ‖`.

`to_jsonl(path, metadata)` exports through `JsonlRunLogger`, so every line carries `run_id` +
`config_hash` (AC-04).

## Checkpoints

safetensors, with the **config and `RunMetadata` embedded in the file metadata**. Loading
rebuilds the model bit-exactly from the checkpoint alone — no separate config file to lose. A
file missing the embedded keys is rejected as "not a WAM checkpoint".
