# Improvements — state encoder, action head, backbone readout

**Status 2026-07-28.** M0–M4 code-complete (617 tests green). The blockers are real teleop data
and running T-16 on Discoverer+, *not* code. This document collects architectural improvements
that are worth doing **after** D1 overfits on real data — plus one cheap experiment that should
run before T-16, because it may invalidate a conclusion we already recorded.

Each item carries an evidence label:

- **measured** — we ran it, numbers are in `runs/`
- **literature** — published result on someone else's system
- **hypothesis** — reasoned from our code, not yet tested

---

## The through-line: we mean-pool everything

Three separate places collapse the backbone's token grid to a single vector before anything
learns from it:

| Where | Line | What | Status |
|---|---|---|---|
| Joint training | `src/wam/training/joint.py:337` | `pooled = features.mean(dim=1)` → action branch | open (I-2) |
| Inference | `src/wam/decoders/action_head.py:119` | leading dims mean-pooled in `decode()` | open (I-2) |
| Wan probe | `scripts/hf_job_wan_probe.py:328` | `tokens.float().mean(dim=1)` | ✅ **tested** (I-1): pooling is not the limit here |
| Cosmos3 probe | `scripts/hf_job_cosmos3_probe.py:285` | its own mean-pooled extractor | open — but I-1 removed the reason to chase it |

Mean-pooling destroys *where* things are. For pick-and-place, "the cube is at token (7,12)" is
the single most useful item in the feature map, and the average deletes it. Precision is also
the dimension where the entire field collapses on RoboDojo — no policy exceeds ~12 %.

This was the suspected common cause behind most items below. I-1 tested it on the one row we
could test cheaply — the frozen Wan probe — and found **no gain from keeping the geometry**
(details below). That result is specific: frozen features, linear readout. It does not clear the
training-path rows, where the pooling sits in front of a *learned* head that could exploit
positions a ridge cannot. But it does mean the mean-pool is no longer an assumed culprit
everywhere; each row now has to earn its own evidence.

---

## I-1 · Re-run the frozen-feature probe with a spatial readout

**✅ Done — ran 2026-07-29 on ZeroGPU, 10/10 checks. Outcome: the confound was real but the
verdict survives it.** `runs/wan_probe/2026-07-29-zerogpu-5b-readouts.json`. Result first, the
reasoning that led here below it.

**What we concluded** (T-15, T-24, *measured*): neither Wan2.2-TI2V-5B nor Cosmos3-Nano frozen
features beat the state-only ridge. Wan best block pair joints R² — Cosmos3 0.359 / gripper
0.708 vs. state-only 0.456 / 0.881. Recorded as: "no frozen features beat state-only yet — LoRA
(T-16) carries the burden."

**The confound** (*hypothesis*): the probe measured the features **through a mean-pool**
(`hf_job_wan_probe.py:269`). We did not show that the spatial information is absent from the
backbone. We showed it does not survive averaging. Those are different claims, and we currently
have the weaker one written down as the stronger one.

**The result** (2026-07-29, *measured*): joints, held-out-episode R², block pair chosen on val —

| Readout | Width | val | **test** |
|---|---:|---:|---:|
| `mean` | 3 072 | 0.404 | **0.310** |
| `grid2x2` | 12 288 | 0.424 | **0.370** |
| `rand4` (control) | 12 288 | 0.417 | **0.376** |
| `state_only` | 52 | 0.547 | **0.456** |

Gripper says the same thing more loudly: `state_only` 0.881 against 0.704 for the best readout.

`grid2x2` scores **below** its own random control (0.338 vs. 0.3657 on the val-selected pair).
The step up from `mean` to `grid2x2` looks like progress and is entirely explained by the 4×
width — shuffling the same tokens into meaningless groups buys the same thing. Both report bits
came back false:

```
any_geometry_gain_over_control: false
any_spatial_beats_state_only:   false
```

**So the mean-pool was not the mistake.** The confound was real — we genuinely had not measured
what we claimed — but testing it changed nothing except our confidence. T-15/T-24's verdict is
now checked against the obvious alternative explanation and holds: 52 dimensions of raw robot
state beat 12 288 dimensions of frozen video feature, and not because of how they were pooled.

Cost: 7.6 s GPU (0.079 s/window, 24.6 GB peak). The three readouts share one set of forward
passes, so the extra two were free. Token geometry verified in-run: `S=96, grid (2, 6, 8)`.

**What it does not license.** This is about *frozen* features under a *linear* readout, on one
task with 96 windows and a 12-episode split. It says nothing about whether a fine-tuned backbone
carries action signal (T-16), nor whether a non-linear head could find what a ridge cannot. It
also does not retire I-2: the training path still mean-pools, and a cross-attention head is
motivated by literature, not by this measurement.

---

**How it was built.** `scripts/hf_job_wan_probe.py` grew a `--readout`
flag; everything else (windows, labels, episode split, ridge code) is untouched, so the output
is directly comparable to `runs/wan_probe/` and `runs/cosmos3_probe/`.

Three readouts, all fitted on the *same* forward passes:

| Readout | What it does | Width (D = 3072) |
|---|---|---|
| `mean` | the historical mean-pool, byte-for-byte | 3 072 |
| `grid<R>x<C>` | average-pool the token grid into R×C cells, keep them separate | R·C · 3 072 |
| `rand<N>` | pool the same tokens into N equally sized **random** groups | N · 3 072 |

`rand<N>` is what makes the comparison worth anything. A coarse grid has more dimensions than
a mean-pool, and more dimensions alone can raise a ridge R². The random control has the
*identical* width and group sizes with geometry removed, so:

- **grid > rand** → position carries action signal, and the mean-pool was hiding it
- **grid ≈ rand** → we only bought dimensions; the recorded verdict stands and gets stronger

The Wan geometry is known: 5 context frames at 192×256 through the TI2V-5B VAE (spatial 16,
temporal 4) and patch (1, 2, 2) give a **(F'=2, H'=6, W'=8)** token grid — 96 tokens, which is
why `grid2x2` (cells of 3×4 tokens) and its `rand4` control are the default pair. `token_grid()`
on the adapter is the single source of truth and the probe asserts S against it before any
reshape, because a silent mismatch would make every spatial number meaningless.

Run it on the free ZeroGPU Space (`scripts/deploy_wan_space.py`, "readout probes" tab, the
readout box is blank = default), or:

```bash
uv run scripts/hf_job_wan_probe.py --source /model --data-dir data/raw/gr00t_apple \
    --readout mean,grid2x2,rand4
```

Read `info.probe.readout_comparison` in the report: `any_spatial_beats_state_only` and
`any_geometry_gain_over_control` are the two bits this experiment exists to produce. The
primary readout (`mean`) still occupies `info.probe` unchanged, so the recorded numbers stay
the anchor rather than being overwritten.

**Why it mattered:** the decision rule was written before the run, and both branches were worth
having. Had the spatial readout beaten state-only, T-16's premise would have changed and OD-04
would have deserved a second look. It lost, so LoRA still carries the burden — and now for a
reason we tested rather than one we assumed.

**Still open:** the Cosmos3 probe (`scripts/hf_job_cosmos3_probe.py`) has its own extractor and
still hands `analyze_probes` a mean-pooled array — which the legacy path accepts, so it keeps
working and keeps reporting a single `mean` readout. Re-running T-24 spatially needs the MoT
token grid worked out the same way. **Not worth doing:** the condition for it was Wan's spatial
readout moving the number, and it did not.

---

## I-2 · Replace mean-pool with cross-attention in the action head

**Evidence: literature + hypothesis.** Best value/effort ratio of the architectural changes.

Today `ActionHead` consumes one pooled vector. The standard alternative is action queries that
**attend into** the feature tokens, so the head can look at the region it needs per step.
AHA-WAM does exactly this ("observation-guided context routing"); it is also why its action
expert can be a DiT rather than an MLP.

**Scope:** changes only the head. `ActionDecoder` protocol, `ActionChunk`, the safety layer and
every robot adapter stay untouched — `decode()` would take `[*, F]` and *not* pool it.

**Ordering note:** do this before I-3. Cross-attention also improves the flow branch, since
`ActionVelocityHead` consumes the same `pooled` (`joint.py:338`).

**Caveat:** at D1 scale this may not show up at all, or may hurt — more capacity on a tiny
dataset. Gate it through `scripts/run_ablation.py` on the same 362/40 split as T-18, not on
intuition.

---

## I-3 · Make the flow branch the deployed path

**Evidence: literature.**

We already train two action paths:

- `velocity_head` — rectified flow on action latents, co-denoised with video, shared `t`
  (`joint.py:338`)
- `action_head` — direct regression from pooled features (`joint.py:339`)

The **deployed** path is the regression head: `ActionOnlyModel.predict` →
`action_head.decode(features[0])` (`src/wam/training/action_only.py:152`).

Single-shot MSE regression is mean-seeking. Where two valid actions exist (grasp the cube from
the left or the right), the regressor outputs the average — reaching between them, which is
invalid. This is the standard argument for iterative action heads and why π0, π0.5 and
GR00T-N1.x all ship a flow-matching action expert.

**What it needs:** a sampler on the existing velocity head, plus a latency budget check. The
executor floor is ≥2 Hz (`ExecutorConfig.min_policy_rate_hz`, PRD §11.1) and the deadline is
500 ms — n denoising steps must fit. AHA-WAM reaches ~57 Hz with a comparable structure, so the
budget is not obviously a problem, but it is the thing that decides feasibility.

**Not urgent while D1 is a single demonstrated way to do one task** — multimodality is a
D2/scale problem. Listed here so the reason the regression head is fine *today* is written down.

---

## I-4 · State history window

**Evidence: literature (RoboDojo Memory dimension).**

`StateMLP.encode()` sees one state snapshot (`src/wam/encoders/state_mlp.py:145`). `dq` gives
first-order motion; nothing carries "what was visible three seconds ago."

RoboDojo's Memory dimension is where every policy sits in the low single digits, and the paper's
finding is specifically that *temporal prediction alone is insufficient for sparse-evidence
recall* — i.e. our video branch will not fix this for free either.

**Only worth doing if memory-conditioned tasks become a target.** The MVP pick-and-place is
memoryless. Deliberately parked.

---

## I-5 · State as latent frames rather than global conditioning

**Evidence: literature.** Bigger surgery — post-MVP.

Today: `StateMLP` → `backbone.condition_state(state_emb)` → one global conditioning vector.

Cosmos Policy instead injects proprioception, future states and value estimates as **additional
latent frames inside the video denoising sequence**, preserving the pretrained model's frame
format. Proprioception then flows through the full attention stack and can interact with the
spatial tokens, instead of biasing everything uniformly.

Attractive because it needs no new architecture — it reuses the backbone's own interface. But it
touches `FlowBackbone` and both adapters, so it is an M6-shaped change, not an M5 one.

---

## I-6 · Add FLUX.2 as an image-editing backbone candidate

**Evidence: literature.** Touches OD-04 / OD-06 / M5.

Our backbone assumption is a **video** model — OD-04 settled on Wan2.2-TI2V-5B (Apache 2.0),
verified on real weights, and OD-06 defers FLUX 3 to M5.

**ImageWAM** (arXiv 2606.19531, June 2026) argues world-action models do not need video
generation at all: an **image editing** model is better aligned with manipulation, because
"transform the current observation toward a desired visual state" *is* the manipulation problem.
It extracts intermediate KV caches from the editing denoiser and conditions an action head on
them — no future-frame decoding. With FLUX.2 4B it reports 93.38 % on RoboTwin 2.0, comparable
to Fast-WAM (91.83 %) and LingBot-VA (92.20 %), without extra policy pretraining.

**Read those numbers carefully.** Fast-WAM scores 91.83 % on RoboTwin 2.0 and **2.03 %** on
RoboDojo. RoboTwin-class benchmarks are exactly what RoboDojo was built to expose, and ImageWAM
is not in RoboDojo. Treat this as "worth a probe," not "better."

**Why it is attractive anyway:** FLUX.2 is open and available **today**, unlike FLUX 3 Dev
(API/private early access only, license unpublished as of 2026-07-26). It gives M5 something to
work with that does not depend on a release we do not control. FLUX.2 4B→9B scaling on
LIBERO-Plus was 83.1 % → 85.21 % — modest, so start at 4B.

**Licensing, checked 2026-07-29 — this is the part that bears on OD-06.** BFL ships the FLUX.2
collection under a *split* licence:

| Repo | Licence | Usable |
|---|---|---|
| `black-forest-labs/FLUX.2-dev` (32B) | FLUX Non-Commercial | no |
| `black-forest-labs/FLUX.2-klein-4B` / `-9B` (+ `-base`, FP8/NVFP4 quants) | **Apache 2.0** | yes |

If FLUX 3 follows the same pattern — and FLUX.1 did too — then **"FLUX 3 Dev", the PRD's
preferred backbone, will be non-commercial**, and only a `klein` variant would be usable under
the same criterion that decided OD-04 in Wan's favour. That is worth knowing *before* the
release rather than after: it turns I-6 from "a probe worth running at M5" into the hedge for
OD-06. `klein-4B` also runs in ~13 GB at 4 distilled steps, i.e. on a single 4090 — so it is
the cheap rig for debugging the T-16 training loop against real 4B weights, separately from
whether it ever becomes a backbone.

**What it is not:** a video model. FLUX.2 is image-to-image with no temporal attention and no
temporal VAE compression, so it cannot satisfy `FlowBackbone` in the sense Wan does. ImageWAM's
claim is precisely that a world-action model does not need that — but adopting it means
adopting their reformulation (autoregressive next-frame editing, KV-cache readout), not
dropping FLUX.2 into the existing seam.

**Cheapest form:** a third `hf_job_*_probe.py` reusing the same windows/labels/split as T-15 and
T-24. Same decision rule. If I-1 lands first, run it with the spatial readout.

---

## What not to change

Recorded so these do not get "improved" by accident:

- **tanh + identity normalization** (`action_head.py:95`). Targets are physical canonical units
  with no denormalization anywhere; `EpisodeDataset` enforces per-step `|targets| < 1`. This is
  a deliberate MVP simplification with a documented failure mode — it will need revisiting when
  action ranges grow, not before.
- **Validity mask + learned missing embeddings** (`state_mlp.py:142`). Genuinely ahead of most
  published policies, which either concat proprioception raw (breaks on sensor dropout) or drop
  it. `torch.where` keeps it differentiable-safe and stops NaN/Inf in masked groups from
  reaching the backward pass. Keep.
- **`hidden_dims=(64,)` on the action head.** Small on purpose. For "overfit one small task" a
  tiny MLP on pooled features is the *right* choice — fastest to debug, and if it cannot overfit,
  the fault is data or wiring, not capacity. Revisit only with I-2.
- **Deterministic safety layer downstream.** Non-negotiable (FR-07). It is also why soft bounds
  in the head are acceptable.
- **Backbone-agnostic protocol** (FR-09, `FlowBackbone`). It is what makes I-6 a probe instead of
  a rewrite. Most published work hardcodes its backbone; we should not follow.

---

## Suggested order

| # | Item | When | Cost |
|---|---|---|---|
| ~~1~~ | ~~I-1 spatial-readout probe~~ — **✅ ran 2026-07-29, verdict unchanged** | done | 7.6 s GPU, free |
| 1 | I-2 cross-attention head | after D1 overfits on real data | days |
| 2 | I-6 FLUX.2 probe | M5, alongside the FLUX 3 decision | hours |
| 3 | I-3 flow branch deployed | D2 / scale | days + latency work |
| 4 | I-4 state history | only if memory tasks become a target | days |
| 5 | I-5 state as latent frames | M6 | weeks |

I-1 jumped the queue because it tested a claim we had already written into `TASKS.md` as settled.
It came back negative, which is the cheapest possible outcome: nothing downstream has to change,
and the claim is now one we have earned rather than one we inherited from a pooling choice.

---

## External context

Backbones used by the five world-action models in RoboDojo, for the bake-off record:

| Model | RoboDojo rank | Backbone |
|---|---|---|
| X-WAM | 5 | Wan2.2-TI2V-5B + its causal VAE; depth as a light added branch |
| GigaWorld-Policy | 8 | undisclosed pretrained video backbone / GigaWorld-0.5 |
| AHA-WAM | 12 | Wan2.2-5B (video branch only); action DiT 1.02B trained from scratch |
| Fast-WAM | 14 | Wan 5.5B |
| LDA-1B | 19 | latent dynamics, no pixel diffuser |

Wan 2.2 (28.07.2025, Apache 2.0) is not the strongest video model — it is the **newest open**
one. Wan 2.5 / 2.6 / 2.7 are all closed, API-only. That licensing wall, not model quality,
explains the field's backbone monoculture, and it independently confirms OD-04.

Evidence that a stronger backbone helps is thinner than the marketing suggests: ImageWAM's
4B→9B gain was ~2 points and non-uniform, and "Turning Video Models into Generalist Robot
Policies" has no backbone ablation at all — only pretrained-vs-random init (0.40 → 0.13
validation MSE), whose authors explicitly warn it is not a substitute for closed-loop success.

**Read that alongside T-18** (*measured*): our own world-action-vs-action-only ablation came out
at −89.5 % holdout MSE — the video branch *hurt* at tiny scale. Both point the same way: what
matters is **whether** you pretrain, far more than **on which** backbone.

Sources: [RoboDojo](https://arxiv.org/abs/2607.04434) ·
[ImageWAM](https://arxiv.org/html/2606.19531v1) ·
[AHA-WAM](https://arxiv.org/html/2606.09811) ·
[Cosmos Policy](https://arxiv.org/html/2601.16163v1) ·
[NVIDIA WAM overview](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
