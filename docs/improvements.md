# Improvements — state encoder, action head, backbone readout

**Status 2026-07-30.** M0–M4 code-complete (681 tests green). T-16 has run: the Wan2.2-TI2V-5B
LoRA scores WAM-Bench **L0, 48.4/100** with `skill_vs_repeat_pct` **−32.4 %**, so the pretrained
prior does not clear the bar either (`docs/benchmark.md`, `TASKS.md` T-16). Every route to
"video helps" has now returned a negative on the same 402 success-only episodes of one task.

That makes this document's job different from a week ago. It is no longer a list of things to do
after D1 — **it is the list of reasons the negative might not mean what it looks like**, ordered
by how cheaply each can be ruled out. Two of them (I-7, I-3) are deviations between the trained
path and the deployed path, found by reading the code *after* the result came in, and one of them
(I-7) sits directly under the verdict.

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

## The second through-line: the deployed path is not the trained path

Found while reading the code after the T-16 verdict. In **two** places, `predict()` does
something training never did — and both deviations point at the same measured failure signature.

| # | Trained on | Deployed / evaluated on | Item |
|---|---|---|---|
| 1 | a real 9-frame window, `frames[indices]` (`training/datasets.py:156`) | **one** frame tiled 9×, `image.expand(num_frames, …)` (`training/joint.py:388`) | **I-7** |
| 2 | both `velocity_head` (flow) and `action_head` (regression) (`joint.py:338-339`) | `action_head` only — the flow branch is never sampled | **I-3** |

The T-16 result is `skill_vs_zero` **+25.9 %**, `skill_vs_repeat` **−32.4 %**, `smoothness_ratio`
**0.29**. Read as a symptom rather than a score, that is: *moves in roughly the right direction,
far too smoothly, and loses to motion continuity.* Deviation 1 removes motion from the visual
input, which is what a model would need to beat motion continuity. Deviation 2 is a mean-seeking
regressor, which is what produces a trajectory 3.4× smoother than a demonstration. Neither is
proof of anything, and this ordering is post-hoc — the symptom was known before the causes were
looked for, so both need their own falsifiable run and both get one below.

**Neither deviation affects T-15 / T-24 / T-26.** The frozen-feature probes build their own real
multi-frame windows and use a ridge, not `predict()`. "Frozen features carry no action signal past
a state-only ridge" stands independently. What is under review is only the verdict on *trained*
world-action models — T-18 and T-16, which share the `predict()` path.

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

## I-7 · Give the policy a frame history at inference — the T-16 eval measured a still

**Evidence: the mismatch is measured (it is in the code, both sides quoted below); its effect on
the verdict is hypothesis.** Cheapest item in this document and the only one that can retract a
recorded conclusion. **Run it before acting on the T-16 result.**

> **Status 2026-08-01: built, tested, not yet run.** The offline path is implemented (T-29,
> 700 tests green) and waits only for a GPU:
>
> ```bash
> sbatch cluster/discoverer/61_eval_t29_frame_history.sbatch    # both modes + the verdict
> ```
>
> Local equivalent and the full runbook: `docs/local_gpu.md` §3b. What shipped:
> `Observation.image_history` (optional, `INTERFACES_VERSION` 0.3.0),
> `wam.data.episode.frame_window_indices` as the **one** definition of the window that both
> `EpisodeDataset` and `build_eval_pairs` now call, `resolve_frame_context` shared by both
> `predict()` implementations, and `eval_t16.py --frame-history` — **off by default**, so the
> archived runs stay reproducible and the A/B is an explicit experiment. Verified bit-identical
> to the pre-T-29 path on real chunks from `d1-full-gen-seed0`.

`EpisodeDataset` hands the model the `num_frames` frames *ending at* the chunk timestamp:

```python
frame_idx = max(int(np.searchsorted(frame_ts, ts, side="right")) - 1, 0)
lo = frame_idx - self.num_frames + 1
indices = np.clip(np.arange(lo, frame_idx + 1), 0, frames.shape[0] - 1)   # datasets.py:154-156
```

`JointWorldActionModel.predict` hands it the same frame nine times:

```python
frames = image.unsqueeze(0).expand(self.config.backbone.num_frames, -1, -1, -1)   # joint.py:388
```

So a backbone trained to read a moving clip is evaluated on a freeze-frame repeated nine times.
Both world-action numbers on record (T-18 tiny, T-16 LoRA) were produced this way — `eval_t16.py`
and `run_ablation.py` both go through `build_eval_pairs` → `evaluate_policy` → `policy.predict()`.

**Why this is not a rounding error.** The baseline the model must beat is repeat-last-action,
which is nothing but motion continuity. A static clip carries no velocity at all, so the only
motion evidence left is `dq` from the state vector — the very channel the action-only baseline
already has. Whatever the video branch might contribute about *where things are going*, the
evaluation deleted before asking. This is not a new discovery so much as an unjoined dot: the
`predict()` docstring already calls the tiling "a real limitation … a rolling frame buffer is the
follow-up" (`joint.py:376-379`). Nobody connected it to the number.

**Scope.** `Observation.images` is documented as one `HxWxC` array per camera
(`interfaces/protocols.py:27-30`), so this is a versioned-interface question, not a one-liner.
Two shapes, and they are not exclusive:

- *Offline (does the verdict change?)* — **done.** The frames were already in hand:
  `build_eval_pairs` reads the whole episode, so `num_frames=` now attaches the window
  `EpisodeDataset` selected, through the same `frame_window_indices`. `Observation` grew an
  optional `image_history` so `evaluate_policy` stays policy-agnostic; the invariant that its last
  frame **is** `images[key]` is what makes the field safe to be optional, and it is checked rather
  than trusted. Robot-facing code untouched.
- *Closed loop (does the robot get one?)* — **not done, deliberately.** Needs a rolling buffer,
  because `ClosedLoopExecutor` supplies one render per cycle. That makes the policy stateful, which
  drags in a reset-between-episodes rule and a defined answer for the first N cycles (pad by
  repeating — today's behaviour, but only at startup rather than forever). It changes the deployed
  path, so it gets its own review, and only if the offline run says it matters.

**Cost:** one eval pass over the 40 holdout episodes — minutes, ~0.2 GPU-h, **no retraining**. The
checkpoint is unchanged; only what we show it changes.

**Decision rule, fixed before the run.** Re-score `runs/t16-lora-seed0` with the true frame window,
identical split, identical bench code:

- `skill_vs_repeat_pct` **moves materially toward or past 0** → the T-16 verdict was measured out
  of distribution and must be re-stated; T-18's too, and `docs/benchmark.md` needs a correction
  rather than an addendum. AC-07 goes back to open.
- **essentially unchanged** → the negative gets substantially stronger: the model had the motion
  available and still lost to inertia, which is a claim about the model rather than the harness.
  Then the honest bottleneck really is data, and I-8 is next.

Either way the result is worth more than the run costs, which is why it is item 1.

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

**Evidence: literature — plus, since T-16, a symptom of our own.** Promoted from "not urgent".

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

**What changed on 2026-07-30.** The argument above was purely a priori: mean-seeking regression
*should* average over valid alternatives. T-16 then produced the matching symptom — a
`smoothness_ratio` of **0.29**, i.e. predictions 3.4× smoother than the demonstrations they are
scored against, together with a positive `skill_vs_zero` and a negative `skill_vs_repeat`. A
model outputting the blandly-averaged trajectory of the task is what that combination looks like.
The earlier note said multimodality is a D2/scale problem; the evidence now says the *smoothing*
shows up at D1 scale, whether or not the cause is multimodality.

**Cheap version first.** The velocity head is already trained in every T-16 checkpoint and simply
never sampled. Adding a sampler and re-scoring `runs/t16-lora-seed0` needs **no retraining** — the
same "re-score what we already have" move as I-7, and it can share that eval pass. Do it after
I-7, since a static-clip conditioning signal would handicap both heads equally and confound the
comparison.

**Then the latency question**, which decides whether it can ship: the executor floor is ≥2 Hz
(`ExecutorConfig.min_policy_rate_hz`, PRD §11.1) with a 500 ms deadline, so n denoising steps must
fit alongside the backbone pass. AHA-WAM reaches ~57 Hz with a comparable structure, so this is a
budget to measure, not an obvious wall.

**Measured 2026-08-01 on the real checkpoint, before writing any sampler.** On
`runs/t16-lora-seed0/checkpoints/step-020000` against the 1 040 archived holdout chunks:

| path | chunk MSE | RMS \|targets\| |
|---|---|---|
| deployed regression head | 1.21027e-05 | 0.00226 |
| `action_encoder` → `action_recon` round-trip | **8.10372e-07** | 0.00412 |
| the demonstrations | — | 0.00404 |
| repeat-last-action (the L1 bar) | 9.14e-06 | — |

The action **latent** carries the chunk to within 8.1e-07 — 15× better than the deployed readout
and 11× better than the bar T-16 failed. Whatever T-16 measured, it was not an encoder that cannot
represent these actions. The single-shot readout discards it and under-shoots magnitude by 44 %.
That is what makes I-3 a *readout* experiment rather than a retrain, and it also sets the two
pre-registered anchors in `63_eval_t30_flow_head.sbatch`: `CEILING_MSE` (the round-trip, what a
perfect sampler would reach) and `FLOOR_MSE` **1.68201e-05** — the score of a chunk with the right
content and no position, because step index is recoverable from the latent at ~100 % accuracy and
`ActionVelocityHead` takes no step index. Landing near the floor is the *expected* outcome of an
order-blind sampler, not a bug, which is why the rule reads it as a mechanism only when the number
actually lands there.

**"Consistent with mean-seeking", not "the signature of it."** An earlier version of this entry,
and of `docs/benchmark.md`, claimed the smoothness/magnitude pair diagnosed a mean-seeking head.
It does not. Ruled out: a bounded-output artifact (max `|target|` 0.0192 against a `tanh`;
`limit_penalty` bites only outside ±0.95). **Not** ruled out, each sufficient on its own:

- **the one-sided jerk regulariser.** `configs/training/joint_wan_gr00t.yaml` sets
  `weights.smoothness = 0.01`, and `JointTrainer.compute_losses` applies `smoothness_loss` to
  `decoded_targets` — the *regression* head's output — and to nothing else. The flow branch is
  never charged for jerk, so a smoothness gap between the readouts is the objective's own
  consequence. This confound applies to the **positive** branches of T-30 as much as the negative
  ones, which is a correction to the first draft of that rule: it held the negatives to an
  alternative reading and the positives to none.
- **plain L2 shrinkage** toward zero under a small-target distribution, with weight decay on top.

Separating any of these from mean-seeking needs an intervention — retrain at
`weights.smoothness = 0`, or a genuinely multi-modal task — not a readout swap. T-30 answers the
narrower question a readout swap *can* answer: does the branch trained as a distribution carry
anything the regression head does not?

**The sampler's own confound, named in the rule rather than discovered afterwards.** Training
co-noises video and action at a **shared** `t` (`co_denoise`). `sample_action_chunk` computes
**one** backbone pass at `t=1` on the clean observation and reuses those features at every `t_k`,
so near `t_k=0` the velocity head is asked about (near-pure-noise latent, clean-video features,
t≈0) — a combination it never saw. That is I-7's structure again, one level down. The faithful
sampler (n backbone passes, observation noised to each `t_k`) is deliberately **not** run: it costs
n backbone passes and destroys the observation the readout exists to read. So **no branch of T-30
refutes the flow branch as trained** — it is bounded by a warm-start arm (`--flow-t0`, integrating
only where the head's `t` and the features' `t=1` roughly agree) rather than removed.

**And the metric penalises the thing under test.** `skill_vs_repeat_pct` is MSE-derived, and for
any calibrated conditional `E‖a − draw‖² = E‖a − mean‖² + E‖draw − mean‖²`: a single *unbiased*
draw scores worse than the conditional mean by exactly the conditional variance. Scoring one draw
against a mean-seeking regressor would reward the defect under test. The rule therefore keys on a
mean-of-8 arm — a **measurement**, never a deployment candidate — and gates deployment separately
on the single-draw arm, since averaging draws re-introduces the mean-seeking the flow branch
exists to avoid.

**Ready to run:** `sbatch cluster/discoverer/63_eval_t30_flow_head.sbatch` (T-30). Ten arms, one
GPU, ~6 h; the rule is `T30_RULE_V2`, in git before any arm ran, with V1's three defects recorded
in the file rather than silently edited out.

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

## I-8 · Measure the data-scaling curve before buying more data with months

**Evidence: hypothesis. Uses assets we already own.**

Every negative we have — T-18 (tiny trunk hurts), T-15/T-24/T-26 (frozen features carry nothing),
T-16 (LoRA loses to inertia) — was produced on the *same* 402 success-only GR00T episodes of one
task. "Not enough data" is the standing explanation for all three, and it has never been tested.
It is also the most expensive conclusion in the project: it implies a G1 EDU4, a teleop rig and
months of recording (D1/D2, `docs/ROADMAP.md`).

Test it with what is already staged. Retrain T-16 at **40 / 120 / 362** training episodes,
identical everything else, and score each on the same 40-episode holdout. `run_bench.py` already
compares runs and refuses mismatched holdouts, so the curve costs no new measurement code.

**Cost:** 3 runs; the full one took a fraction of the 5 000 GPU-h allocation, and the two smaller
ones are cheaper still. Nothing here needs a robot.

**Decision rule, fixed before the run:**

- `skill_vs_repeat_pct` **improves monotonically** with episode count → data is the binding
  constraint, the extrapolation says roughly how much is needed, and the case for D1/D2 collection
  is evidence rather than hope.
- **flat or non-monotonic** → more of *this* data will not fix it. Then the question is the kind of
  data (task diversity, failure cases, a live gripper — I-9) or the architecture (I-2), and months
  of recording the same task would have been the wrong move.

Weakness to state up front: three points on one task cannot distinguish "needs more episodes" from
"needs more *tasks*". A useful variant, at the same cost, is holding episode count fixed and
varying task diversity across public LeRobot sets.

**Staged 2026-08-01, rule `I8_RULE_V3` in git before the first rung is submitted**
(`cluster/discoverer/55_train_i8_rung.sbatch` trains, `62_eval_i8_curve.sbatch` scores and prints
the verdict). Splits are nested and committed — `configs/splits/i8_train_{040,120,362}.txt`, with
40 ⊂ 120 ⊂ 362, zero overlap with the 40-episode holdout, and a seeded shuffle rather than a
sorted prefix, so a rung is a random sample of the corpus and not its alphabetical head. Rung 362
is `runs/t16-lora-seed0`, already scored; its `bench.json` is consumed rather than recomputed.

Three things the rule had to be taught, all of them before seeing a number:

- **The confound is symmetric.** Equal STEPS means rung 40 runs 147.5 epochs against rung 362's
  16.9, and that gap manufactures a flat curve exactly as readily as a steep one. The first draft
  held verdict A (*build the dataset*) provisional for it and let verdict C (*not data-limited*)
  through free — so the confound could produce the cheap conclusion at no cost. **All three**
  verdicts are now gated on the same equal-EPOCH control (`EPOCH_INFLATED`, read from the run log's
  final `action_reg`). A rule that holds only the expensive conclusion to a standard is a
  preference, not a decision rule.
- **N\* is an extrapolation, not a measurement.** It is OLS through three points, 2.8 doublings
  past the largest measured N, with one residual degree of freedom. Verdict A now requires the
  whole bootstrapped interval — propagated from the *measured* seed spread, via the
  `i8-rung040-seed1` replicate — to sit inside what a collection campaign could deliver, not just
  the point estimate.
- **"C" was two different worlds sharing one sentence.** C fires on `not (MONOTONE and SPAN ≥
  MATERIAL)`, so three noisy rungs landing out of order while spanning materially printed "more of
  this data did not move the headline" when the headline had moved by more than a seed does. Split
  out as **C-NOISY**, which claims no readable curve, routes to a seed replicate rather than to
  I-9, and licenses no recording campaign either way.

Sequencing: run this **after** T-30. A readout swap changes the headline metric for every rung at
once, and fitting a scaling curve through numbers that a pending decode change may move is fitting
a curve to a moving target.

---

## I-9 · Score on a dataset whose gripper actually opens

**Evidence: measured (T-27).**

The demonstrated gripper channel in `datasets/gr00t-apple-full` has peak-to-peak range **0.120**,
sitting on the 0.5 binarization threshold — it never opens or closes. `gripper_accuracy` 0.89 is
thresholding noise, and WAM-Bench emits it as a warning rather than a metric.

The consequence is larger than one bad number: **the benchmark is structurally blind to grasping**,
which is the entire point of pick-and-place and the thing AC-01/02 are about. Every verdict we have
is a verdict about arm trajectories only.

**The prescription above was wrong, and the correction is the whole point of this entry
(2026-08-01).** "Select a public dataset with real transitions" assumed the grasp was missing from
the data. It is not. Measured on `data/raw/gr00t_apple`, parquet only, no decode:

- the **left** hand's joints span up to **0.826 rad** peak-to-peak, and the commanded
  `action[29:36]` spans the full range;
- the **right** hand is frozen — 0.0007 rad across all 402 episodes;
- `hand_synergy`'s `clip((mean + 1) / 2, 0, 1)` assumes source joints in `[-1, 1]`, a range the
  Dex3 hand never uses, and squashes 0.826 → **0.157**;
- `relabel_chunks` then averages **both** hands into `gripper_target`, so the dead right hand
  halves it again → **0.0785**, which is the 0.120 T-27 measured, centred on the 0.5 threshold.

Rescale the same recordings and **30/30 episodes show exactly two debounced open/close
transitions** — one grasp, one release, which is what "move the apple to the plate" should look
like. The grasp was always on disk. We destroyed it in the converter and then read the flat
channel as a property of the task.

**Fix, as it actually shipped:** `--gripper-mapping active-hand` fits one **dataset-level** affine
over the raw synergy of the hand that moves and takes `gripper_target` from that hand alone.
Dataset-level rather than per-episode on purpose — a per-episode min-max makes the same physical
aperture a different number in every episode, which is unlearnable. Because the fit depends on
which episodes are in the conversion (30 eps → offset −0.39980/span 0.41004; 402 eps →
−0.43865/0.46675), two conversions are comparable only if they share an affine: `--gripper-affine
OFFSET SPAN` pins it, and a pinned affine that would clip **any** sample is refused, because
clipping is silent in the output and moves every admissibility clause in the *passing* direction.
The legacy mapping is now held to the same bar (`legacy_clipped_frac`); until 2026-08-01 the one
mapping that *assumes* a scale was the only one allowed to be silently wrong about it.

`scripts/audit_gripper.py` is the gate: four clauses (dynamic range, debounced transitions per
episode, fraction of episodes with a transition, fraction of episodes with a full
close-**and**-reopen **cycle**). The cycle clause exists because a monotone ramp from 0 to 1 clears
a transition count and is not a grasp. Saturation is reported against
`expected_saturated_frac` — a fitted affine rails exactly the two extremal samples it was fitted
on, so a rate far above that is "clipped, not measured".

**What this does not retract:** T-16 and T-18 are arm-trajectory verdicts and stay exactly as
recorded — `skill_vs_repeat_pct` never touched the gripper channel. What it retracts is the
*explanation*, and the size of the fix: this was hours of converter work on data already on disk,
not a dataset acquisition.

Related and cheap, in `docs/benchmark.md` rather than here: **L4's gate is one-sided.**
`smoothness_ratio ≤ 2` scored T-16's 0.29 a full 20/20 while that value means the prediction is
3.4× *smoother* than a demonstration — a defect, not a virtue. A two-sided band would catch it.
Recorded rather than quietly patched, because changing a pre-registered threshold after seeing the
number it scored is exactly the move the pre-registration exists to prevent; it changes no level
here (L4 is only reachable through L1 and L2, and this run stops at L0).

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

Re-ordered 2026-07-30 around one principle: **anything that can change the meaning of a number we
have already recorded comes before anything that produces a new number.** Items 1–2 re-score an
existing checkpoint and need no retraining at all.

| # | Item | When | Cost |
|---|---|---|---|
| ~~—~~ | ~~I-1 spatial-readout probe~~ — **✅ ran 2026-07-29, verdict unchanged** | done | 7.6 s GPU, free |
| ~~—~~ | ~~I-9 converter (the gripper we destroyed)~~ — **✅ built 2026-08-01, needs no GPU** | done | hours, no GPU |
| **1** | **I-7 frame history at inference** (T-29) — may retract the T-16/T-18 verdict | **submitted, waiting on the GPU** | ~0.2 GPU-h, no retrain |
| **2** | **I-3 flow branch deployed** (T-30) — sampler on the existing velocity head | right after T-29 reports | ~6 GPU-h, no retrain |
| 3 | I-9 re-score the ladder on the rescaled gripper (T-31) | after the converter's audit passes | ~0.2 GPU-h, no retrain |
| 4 | I-8 data-scaling curve (T-32) | after T-30 — a readout swap moves every rung at once | 3 runs, existing allocation |
| 5 | I-2 cross-attention head | after 1–4 say whether the readout was the problem | days + retrain |
| 6 | I-6 FLUX.2 probe | M5, alongside the FLUX 3 decision | hours |
| 7 | I-4 state history | only if memory tasks become a target | days |
| 8 | I-5 state as latent frames | M6 | weeks |

**Why I-9 moved but did not jump the queue.** Its *build* is done and cost no GPU, because the
finding turned out to be a converter bug rather than a missing dataset. Its *re-score* still sits
at 3, behind the two items that can change the meaning of a number already recorded. Nothing about
a rescaled gripper channel changes what `skill_vs_repeat_pct` measured on arm trajectories.

**Why I-8 slipped from 3 to 4.** T-30 changes the headline metric for every rung simultaneously.
Fitting a scaling curve through numbers a pending decode change may move is fitting a curve to a
moving target — and N\*, the number that would buy a robot and months of recording, is the most
extrapolated quantity in the file.

I-1 jumped the queue because it tested a claim we had already written into `TASKS.md` as settled.
It came back negative, which is the cheapest possible outcome: nothing downstream has to change,
and the claim is now one we have earned rather than one we inherited from a pooling choice.

I-7 is now in the same position, one level more serious: I-1 questioned how we *probed* the
backbone, I-7 questions what we *showed* it. If the answer is "nothing changes", the T-16 negative
becomes one of the better-supported results in the project. If the answer is "it moves", then two
recorded verdicts were measured out of distribution and the cost of finding that out was one eval
pass. The asymmetry is why it is item 1 despite being the smallest item in the file.

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
