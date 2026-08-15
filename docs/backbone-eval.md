# Backbone evaluation — the screen, and what would actually change our mind

**Scope.** Whether to move off Wan2.2-TI2V-5B (OD-04) for a candidate raised in 2026-08:
NVIDIA Cosmos (3 / Predict2.5) and Tencent HunyuanVideo. Written against the
`BackboneAdapter` / `FlowBackbone` interface (FR-09/AC-05), so "evaluate a backbone" means a
concrete, costed experiment rather than a capability table.

**Read §1 first.** In this project a backbone swap is the intervention with the worst measured
track record, and a candidate screen that does not say so up front is backbone tourism.

---

## 1. What is already decided, and by measurement

| Recorded | Where |
|---|---|
| Wan2.2-TI2V-5B, Apache 2.0, is the backbone of record | OD-04 |
| Frozen **Wan** features lose to a state-only ridge | T-15 |
| Frozen **Cosmos3-Nano** features lose to it too — joints 0.359 / gripper 0.708 vs **0.456 / 0.881** | T-24, `runs/cosmos3_probe/2026-07-26-zerogpu-nano.json` |
| The mean-pool was not the reason — spatial readouts land *below* their own random control | T-26 |
| The T-16 Wan LoRA is negative: WAM-Bench L0, loses to repeat-last-action | T-16, T-29 |
| The dream is **39 % further from the truth than a frozen frame** | T-36, `PR-06-RESULT.md` |
| Head-to-head on identical windows: the Wan/Cosmos **ranking reverses** between 12 and 48 episodes, and both lose to the floor at every size | §7, T-38, `runs/backbone_eval/compare_backbones.json` |

Cosmos3-Nano is therefore **not a new idea here** — it was probed on 2026-07-26 and lost, and
`docs/hf_jobs.md` already records it as the fallback candidate rather than the primary. Two
consequences that a fresh comparison table will not tell you:

- **Cosmos3's VAE *is* the Wan2.2 VAE** (`AutoencoderKLWan`, 48-ch, 16× spatial, 4× temporal).
  A Cosmos-vs-Wan result cannot be attributed to the latent space, in either direction. That
  control is free and already established.
- The robotics pretraining *is* visible, and only in one channel: best single block gripper
  **0.822** (Cosmos3, block 17) vs Wan's **0.734** (block 6) — both read out of `per_block` in
  `runs/{cosmos3_probe/2026-07-26-zerogpu-nano,wan_probe/2026-07-26-zerogpu-5b}.json`. *Corrected
  2026-08-06:* this line read "Wan's 0.698", which is Wan's `suggested_2_10` **pair** value, not a
  single block — comparing a single block against a pair inflated the gap from 8.8 pp to 12.4 pp.
  Joints stay level, 0.410 (Wan, block 16) vs 0.399 (Cosmos3, block 15). Whatever the robot
  pretraining bought, it did not buy linearly readable joint actions.

So the burden on any new candidate is not "is it a good video model". It is: **does it change
something we have already measured to be the limitation?**

## 2. The screen — four criteria, none needing a GPU

- **S1 · Licence.** Usable commercially, in the EU. This is the criterion that decided OD-04.
- **S2 · Fits the box.** Resident weights + activations under **34.36 decimal GB** (an RTX
  5090's 32 GiB; units per `docs/local_gpu.md` — decimal GB throughout, `max_memory_allocated`).
- **S3 · Reachable through the existing harness.** diffusers-native, module imports torch-free,
  VAE + text tower freezable — i.e. `hf_job_*_probe.py` reruns against it with the feature
  extractor swapped and nothing else.
- **S4 · Offers something measured to be missing.** Per §1: a **native action port trained with
  actions**, not a state token bolted onto a text-context slot. Wan has none — `condition_state`
  projects the StateEncoder embedding into text-context space (`wan_i2v.py`), which is our
  invention, trained from scratch on 402 episodes.

S4 is the discriminating one. A candidate that fails S4 is Wan with different weights, and we
have measured that class twice.

## 3. Candidates against the screen

### HunyuanVideo (13B) and HunyuanVideo-1.5 (8.3B) — **FAIL S1, before any GPU**

Both ship under the **Tencent Hunyuan Community License**, whose Territory clause reads
verbatim:

> "Territory" shall mean the worldwide territory, **excluding the territory of the European
> Union, United Kingdom and South Korea**.

and whose grant is "for the Territory only". The agreement's own header states it **DOES NOT
APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA**. There is also a
no-improving-other-models clause ("You may not use ... any Output or results ... to improve any
other AI model"), which is squarely what a WAM adapter does with backbone features.

This is a licence fact, not legal advice — but it is the same criterion class that decided
OD-04 for Wan and that OD-06 flags against FLUX 3. Recorded as a fail so nobody re-derives it.

It would also fail S4: HunyuanVideo-1.5 is a T2V/I2V DiT with no action port. On S2 it is the
*best* of the field (8.3B, ~13.6 GB peak at 720p × 121 frames, comfortably inside the 5090) —
worth stating precisely, because it is the opposite shape of failure to Cosmos's.

### Cosmos3-Nano (16B MoT) — **FAIL S2 as-run**, PASS S4

Our own artifact, not an estimate: the T-24 probe peaked at **36.2 GB** and `--generate` at
35.6–36.9 GB, all on a 96 GB ZeroGPU RTX PRO 6000. That is **1.8–2.5 GB over the whole 5090**,
and no FP8/NVFP4 checkpoint ships — NVIDIA publishes BF16 and recommends a 96 GB card. So the
box we are preparing cannot run the configuration we already ran, and Cosmos3 stays a
ZeroGPU/H200 experiment until it is offloaded or quantized.

It passes S4: Cosmos 3 takes JSON action arrays in and emits action states out. **T-24 never
used that port** — it packed clean conditioning frames behind a tokenized instruction and
hooked the generation-pathway residual stream. The action port is untested.

> **Followed up 2026-08-15, against the model card and cookbooks rather than this paragraph.**
> "Emits action states out" is not a throwaway: the family ships **three** action modes — forward
> dynamics (actions → frames), **inverse dynamics (frames → the trajectory that produced them)**
> and policy (observation + prompt → actions). Inverse dynamics is a video-to-action labeller, and
> §4 below is written as though the port were input-only, which is wrong for Cosmos 3 (it remains
> correct for Predict2, a different model). Two bounds keep this from being a free lunch: the
> supported action vocabulary lists no humanoid, no G1 and no 28-dim Dex3, and NVIDIA's route to
> adding one is post-training *on action-labelled data* — so it amortises labels we already have
> rather than creating any. All twelve action notebooks in
> `cookbooks/cosmos3/generator/action/` are **Nano**; the only `finetune/` recipe is
> Nano-Policy-DROID; Super ships `action_gen=True` with no recipe and a 121 GB export. Written up
> as **T-042**, with `docs/action-labels.md` §3b as the standing explanation.

### Cosmos-Predict2.5 (2B / 14B) — **the only candidate that passes all four**

NVIDIA Open Model License; 2B is ~4 GB of BF16 weights, trivially inside the 5090; diffusers
path is the same family the T-24 harness already drives. And it is action-conditioned by
construction: a single conditioning image plus a sequence of robot actions in, a chunk of future
frames out, with the actions entering through an **action-embedder MLP added to the timestep
embeddings** of the DiT blocks.

That is a different injection site from ours (we add a token to the text context) and, unlike
ours, it was **pretrained** with it. That is the one property in the whole 2026-08 candidate
sweep that our measured record does not already cover.

### FLUX 3 — unchanged, still OD-06

Nothing here moves it. The FLUX.2 signal recorded under OD-06 stands: a `dev` tier under a
non-commercial licence would fail S1, and FLUX.2 being image-to-image means adopting ImageWAM's
reformulation rather than a drop-in `FlowBackbone` (I-6).

## 4. The experiment, and the gates

One probe, and it isolates the *only* thing that is new — the pretrained action port — rather
than re-running a bake-off we have twice concluded.

**Design.** The T-24/T-15 experiment unchanged — same GR00T windows, labels, episode split and
ridge code — run **twice**:

- **arm A**: the action port fed the true past-action chunk.
- **arm B**: the identical forward with the port fed zeros.

Same weights, same process, same windows. The A−B delta is the port, and nothing else. This is
the `set_lora_enabled` pattern from T-35, which is why that delta is trustworthy where a
second-model comparison would not be.

### 4a. The gate this design needed, measured before anything was run

**The first draft of G2 was wrong, and the CPU could show it.** It read
`R²_joints > 0.456 and R²_gripper > 0.881` — the state-only ridge. But arm A is *fed* the past
actions, and T-34 measured lag-1 autocorrelation of 0.927 on this corpus. So the bar had to be
what a ridge gets from the probe's **own inputs with no video model in the loop**, not what it
gets from proprioception. `scripts/probe_action_baselines.py` measures exactly that, importing
the windows, split and ridge from `hf_job_wan_probe` unchanged.

Joints / gripper test R², identical harness, three corpus sizes
(`runs/backbone_eval/action_baselines*.json`):

| features | dim | 12 ep | 24 ep | 48 ep |
|---|---|---|---|---|
| `state_only` — the archived floor | 32 | **0.4563** / 0.8812 | 0.4879 / 0.8036 | 0.5129 / 0.8808 |
| `past_joint` — previous chunk, raw | 256 | −0.0950 / 0.6440 | 0.4371 / 0.6273 | 0.4561 / 0.7206 |
| `past_joint_proj` — same, at matched width, 3 seeds | 112 | 0.453 / 0.452 / 0.384 | 0.495 / 0.465 / 0.479 | **0.539 / 0.522 / 0.546** |
| `past_ee` — Bridge-shaped, what the port eats | 112 | 0.4576 / 0.2930 | 0.4446 / 0.4323 | 0.3954 / 0.4183 |
| `past_ee_plus_state` | 144 | **0.5407** / 0.8424 | 0.5115 / 0.5008 | 0.5025 / 0.8728 |
| `past_joint_proj + state`, 3 seeds | 144 | 0.529 / 0.512 / 0.431 | 0.516 / 0.519 / 0.523 | **0.540 / 0.539 / 0.541** / **0.911** |
| `past_ee_shuffled` — control | 112 | 0.0008 | −0.0055 | −0.0027 |

Four things follow, and three of them are corrections:

1. **`state_only` reproduces the archived 0.456 / 0.881 to four digits at 12 episodes.** That is
   the check that these windows and this split are T-24's.
2. **The floor is not a constant.** It climbs 0.4563 → 0.4879 → 0.5129 with corpus size, so
   "0.456" is a *12-episode* number. Any gate quoting it as an absolute is comparing across
   sample sizes. The comparator has to be recomputed on the probe's own windows.
3. **`past_ee` does not beat the floor.** At 12 episodes it reads 0.4576 against 0.4563 and
   looks like it does; at 48 it is 0.3954 against 0.5129 and clearly does not. It degrades as
   the corpus grows, which is the signature of a small-n result, and the 12-episode reading is
   the one that would have been quoted. **The end-effector representation is the *worse* of the
   two**, and `past_ee_pos` (0.4631 → 0.3474) is worse still.
4. **What is robust is `past_joint_proj + state`: 0.540 / 0.539 / 0.541 joints and ~0.91
   gripper at 48 episodes, three seeds, spread 0.002.** Past actions plus proprioception beat
   proprioception alone, by about +0.03. `past_joint`'s raw −0.0950 was width against 56
   training rows, not information — matching the width reverses it.

The shuffled control sits at 0.000 ± 0.006 at every size, so none of this is the split leaking.

**Corrected gates.** The comparator is computed, not quoted:

- **G1 · the port carries something.** `R²_joints(A) > R²_joints(B)`. If equal, the pretrained
  action conditioning is not linearly readable off the residual stream, every candidate fails
  S4, and the backbone question closes.
- **G2 · it is worth switching.** Arm A must beat **`past_joint_proj + state` on the same
  windows, same split, same run** — a *single* feature set, selected on validation, both
  channels read off that same row: on 12 episodes joints **0.5118** and gripper **0.9484**
  (`past_joint_proj_s1_plus_state`), on 48 joints **0.5399** and gripper **0.9110**
  (`s0`). Not the archived 0.456 / 0.881, and not the row-wise maxima either — 0.5407 is
  `past_ee_plus_state`'s joints and 0.9601 is a gripper from a feature set without state, so
  quoting the pair as one bar builds a comparator no run ever achieved.

For scale: the best frozen-backbone number recorded in this project is **0.4267** (T-38,
Cosmos3 blocks 16+18, 48 episodes, val-selected). At 12 episodes the best is Wan's **0.4096**
(single block 16 — but selected on *test*, so optimistic); the honest val-selected 12-episode
best is 0.3652. G2 asks for 0.51 on those same 12 episodes.

**Verdicts:**

| | outcome |
|---|---|
| G1 fail | Close the backbone question. Record it and stop spending on backbones. |
| G1 pass, G2 fail | The port carries signal but does not displace proprioception-plus-past-actions. Stay on Wan; the number goes in the OD-04 row as the second failed challenge. |
| both pass | Cosmos becomes the primary candidate for a T-16-style LoRA, and OD-04 is reopened with evidence. |

**What this does not test:** a fine-tuned backbone of any kind. T-15, T-24 and this probe all
measure *frozen* features under a *linear* readout. T-16 is the fine-tuned arm and it is
negative on Wan. A G1 pass is a reason to spend GPU hours on a Cosmos LoRA; it is not itself
evidence that the LoRA would land.

### 4b. What the GPU arm actually costs — more than §3 assumed

Verified while building the action representation, and it changes the estimate:

- The checkpoint with a **published, pretrained** action port is
  `nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned` — **Predict2, not 2.5** — post-trained
  on Bridge (IRASim splits). Its port takes 7-D `[x, y, z, roll, pitch, yaw, gripper]`
  end-effector *displacements*, 12 actions → 12 frames, 640×480 at 4 fps, and NVIDIA quotes
  **32.54 GB** for it. Predict2.5 lists a `robot/action-cond` variant too.
- It is **not documented as diffusers-native**, and the T-24/T-15 harness is diffusers-based.
  So this is not "swap the feature extractor" — it needs the `cosmos-predict2` inference stack
  alongside, which is a different order of work from what §3 implied.
- Feeding it our robot needs FK, because our corpus is joint-space and every published port is
  EE-space. That is `wam.robot.kinematics` (13 tests, 7 mutations) — and §4a's headline is that
  the EE representation those ports demand is the *weaker* one on our data.
- Bridge is a WidowX at 4 fps; ours is a G1 at 30 fps with a mean per-frame EE displacement of
  **1.6 mm** (episode 0, path length 0.94 m over 590 frames). Arm A is out of distribution for
  that port on scale alone, which does not invalidate the question — "does this pretrained port
  help *our* data" is the question — but it does lower the prior.

## 5. Cosmos-Transfer2.5 — a separate question, deliberately not gated here

Transfer2.5 consumes depth + segmentation + Canny and emits photorealistic video. The Isaac
backend (`src/wam/robot/isaac_transport.py`) can emit exactly those. That is a real pipeline and
the most concrete thing in the 2026-08 material.

It is **not** a backbone question — it is synthetic training data, and it collides head-on with
a standing decision: **"Sim frames are NOT training data"** (`docs/sim.md`, T-25). T-36 also
already priced generated video as supervision and the answer was worse than a frozen frame.
Overturning that needs its own pre-registration with `screen_corpus.py` (T-34) run on the
generated corpus, not a paragraph in a backbone doc. **That is now T-040**
(`.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md`) — backlog, blocked on T-39,
and its deliverable is the pre-registration, not the corpus.

## 6. Adding a backbone, mechanically

For whoever runs §4 — the interface work is small and the registry is the only place that
learns the name:

1. `src/wam/backbones/<name>.py` — implement `condition_video` / `condition_text` /
   `condition_state` / `features` / `name` / `feature_dim`. Torch-free at module scope; import
   torch inside `load()`; nothing downloads unless `allow_download`.
2. A factory in `backbones/registry.py::_FACTORIES` (lazy import — listing backbones must not
   pull torch).
3. For the training path, a `kind` member in the `BackboneConfig` union (`wam.training.joint`)
   and a branch in `build_backbone`. Never fall back to `tiny` on ImportError — that trains junk
   silently, which is why `wan_i2v` raises instead.
4. `FlowBackbone` extras if it is to be trained rather than probed: `encode_video`,
   `decode_video`, `num_video_tokens`, `frozen_part_names`, `forward_flow`.

Swapping it in must not change the data schema or the robot API (FR-09/AC-05); the swap tests
enforce that.

## 7. The head-to-head, run as one experiment (T-38)

§1 records two verdicts — Wan loses, Cosmos3-Nano loses — reached in two runs a month apart, each
compared against **0.456 / 0.881**, a constant lifted from the first of them. §4a then measured
that the constant moves. Three defects follow, and `scripts/compare_backbones.py` is the driver
that removes them. It drives the two Spaces that are already deployed
(`huhn511/wam-wan-smoke`, `huhn511/wam-cosmos3-probe`) through `gradio_client`, so this costs
ZeroGPU minutes and no new deployment.

**1 · Two runs are not a comparison.** T-15 and T-24 ran the *same window code*, which is not the
same as having run the *same windows*. The driver asks both Spaces for identical parameters and
then verifies from the returned reports that the episode list, window count, context frames,
resize, chunk length, instruction, the train/val/test **episode split** and the dataset snapshot
revision all agree. Flags are intent; reports are evidence. A disagreement exits 2 and prints no
table — a comparison across two window sets is not a weaker result, it is not a result.

That check earned itself on the first real run: the Wan Space records `window_select` in
`info.data` and the deployed Cosmos copy predates the field. Same windows, different vintages of
`probe.py`. The resolution is `--assume-default window_select`, which writes the assumption into
the artifact and *refuses* if the recorded value is anything but `linspace` — `motion` selects a
different subpopulation, so it is a different experiment, not an omission.

**2 · Every archived verdict is a 12-episode verdict**, i.e. 56 training windows. The driver
sweeps 12 / 24 / 48 and recomputes the floor and the best input-only comparator at each size with
`probe_action_baselines.py`, on the same windows. Nothing is quoted.

**3 · Wan is 3072-dim per block and Cosmos3 is 4096.** The two-block candidate each report picks
is therefore **6144** dims for Wan and **8192** for Cosmos, and §4a measured that a ridge on 56
rows is very sensitive to that: the same information scored −0.0950 at 256 dims and ~0.45 at 112.
A raw Wan-vs-Cosmos delta is part prior and part tensor width.

### 7a. What was measured (2026-08-05, `runs/backbone_eval/compare_backbones.json`)

Joints / gripper test R², val-selected block pair, identical windows per row:

| episodes | Wan2.2-TI2V-5B | Cosmos3-Nano | state-only floor | best input-only |
|---|---|---|---|---|
| 12 (96 win, 56 train) | **0.3652** / 0.6976 | 0.3240 / 0.6126 | 0.4563 / 0.8812 | 0.5118 / 0.9484 |
| 24 (192 win) | **0.3011** / 0.6017 | 0.2837 / 0.6998 | 0.4879 / 0.8036 | 0.5193 / 0.8677 |
| 48 (384 win) | 0.3867 / 0.5420 | **0.4267** / 0.8215 | 0.5129 / 0.8808 | 0.5399 / 0.9110 |

The comparator column is `past_joint_proj_s1_plus_state` at 12 and 24 episodes and
`past_joint_proj_s0_plus_state` at 48 — the driver takes whichever input-only row wins on *these*
windows rather than fixing a feature set, which is why the column changes identity down the table.

**It wins on validation, not on test, and that is a correction to an earlier version of this
table.** The backbone columns are each report's *val-selected* block pair; a comparator picked by
test R² across ~12 rows is not the same protocol, and at these sample sizes the difference is
visible: the test-argmax rows are 0.5407 / 0.5230 / 0.5463 (`past_ee_plus_state`,
`past_joint_proj_s2_plus_state`, `past_joint_proj_s2`), up to **0.029** above the val-selected row
that replaced them, and at 48 episodes the test-argmax is precisely the luckiest of three
interchangeable projection seeds (0.5385 / 0.5223 / 0.5463). Both numbers are in the artifact and
both are printed, the second one labelled optimistic. The headline is unaffected — the floor is not
selected at all — but the bar itself was inflated.

Three readings, and the second is the one that matters:

1. **The 12-episode row reproduces T-15 and T-24 to four digits** — Wan 0.3652, Cosmos 0.3240,
   floor 0.4563 / 0.8812. Nothing drifted in the month between, so the rows below are comparable
   with the archive.
2. **The ranking reverses.** Wan is ahead by 0.041 joints at 12 episodes and behind by 0.040 at
   48. Whichever backbone you prefer, there is a corpus size that agrees with you. **A
   single-size head-to-head between these two backbones does not measure anything about the
   backbones** — and a single-size head-to-head is exactly what the archived record consists of.
3. **What does not reverse: both lose to proprioception at all three sizes.** The better of the
   two sits **0.091 / 0.187 / 0.086** below the floor at 12 / 24 / 48 — no trend, no closing.
   Cosmos's gripper at 48 (0.8215) is the only cell that comes near its floor, and it still does
   not clear it. Against the *best input-only* comparator, which is the bar §4a argues for, every
   cell is worse still.

Free control, already established in §1 and worth restating because it removes an explanation
before anyone reaches for it: **Cosmos3-Nano's VAE *is* the Wan2.2 VAE** (`AutoencoderKLWan`,
48-channel). Whatever separates the two rows, it is not the latent space.

### 7b. The width control

The Spaces return a report, not features — their Gradio outputs are `[log, JSON]`, and changing
that means redeploying. So the arm that actually projects both backbones' pooled features to one
width runs only when they are supplied (`--features wan=…npy cosmos=…npy`), over three seeds with
a row-shuffled control. What the driver measures without them is the width effect itself, by
carrying one fixed known-informative feature set — proprioception, 32 dims, the floor — through a
random projection to 6144, 8192 and 112 dims and scoring it with the same ridge, split and labels.

It does that **two ways**, and an earlier version of this section had only the first one and read
a width result out of it that it cannot produce. Joints test R², mean of three seeds:

| carried to | 12 ep | 24 ep | 48 ep |
|---|---|---|---|
| 32 — unprojected, reproduces the floor exactly | 0.4563 | 0.4879 | 0.5129 |
| 32 — *projected, same width* | 0.5429 | 0.6045 | 0.5813 |
| **6144** — Wan's candidate width | 0.5542 | 0.6057 | 0.5749 |
| **8192** — Cosmos's candidate width | 0.5584 | 0.6011 | 0.5783 |
| 112 — T-37's matched width | 0.5586 | 0.6014 | 0.5766 |
| *8192 − 6144* | *+0.0042* | *−0.0046* | *+0.0034* |
| shuffled, 3 permutations | −0.0001 | −0.0023 | −0.0006 |

**This family cannot answer a width question, and its flatness is not evidence that width does not
matter.** A random projection is a change of basis: for any target width at or above the source
rank the projected tensor spans the same row space, so the ridge is offered the same information at
112 dims as at 8192. Measured on the 12-episode windows over widths 32 / 48 / 64 / 112 / 256 / 1024
/ 3072 / 6144 / 8192 / 32768, joints stays inside 0.5429–0.5616 with no trend, and 112 vs 8192 — a
73× change — moves it **0.0002**, well inside the 112 row's own seed spread of 0.0116. The previous
reading of the *8192 − 6144* line ("±0.005, sign unstable, so the head-to-head is not
width-confounded") was an instrument returning zero, not a measurement of the two backbones.

The `32 — projected, same width` row is the other correction. **Almost the entire "+0.06 to +0.12
from being carried up" happens at zero width change**: 32 → 32 through a square projection is
already +0.087 / +0.117 / +0.068, and everything from there out to 6144 adds +0.011 / +0.001 /
−0.006. It is the rotation plus `probe_r2`'s per-column standardisation moving the ridge's
spherical prior onto a different basis. It was never a width effect.

What a wide backbone tensor actually imposes is directions the ridge must regularise away, and the
projection adds exactly none of them. So the driver also carries the same 32 informative dims
*padded* with uninformative columns to the same widths:

| 32 informative dims padded to | 12 ep | 24 ep | 48 ep |
|---|---|---|---|
| 112 | 0.3939 | 0.3931 | 0.4399 |
| **6144** — Wan's candidate width | 0.0223 | 0.0372 | 0.0595 |
| **8192** — Cosmos's candidate width | 0.0095 | 0.0351 | 0.0670 |
| *8192 − 6144* | *−0.0128* | *−0.0021* | *+0.0075* |
| *worst seed spread in the column* | *0.0461* | *0.0419* | *0.0644* |

Two readings, and the second is the one that changes what §7a may claim:

- **Width at this sample size is expensive, not free.** The floor's own 32 dims, worth 0.46–0.51
  alone, are worth **0.01–0.07** once they sit inside 6144 or 8192 dims. Whatever a frozen backbone
  carries, it carries it in a tensor shaped like the second row of this table.
- **The 4:3 ratio between the two backbones' widths is still not resolvable** — but now that is a
  measurement rather than an artifact of the instrument. The padded rows put 6144 and 8192 within
  0.013 of each other, inside their own seed spread (up to 0.064). So: the width difference
  *between these two backbones* is below what any control available here can see, and the 0.04 that
  separates them in §7a is the same size as the noise of the only arm that reads width at all. The
  earlier claim that the head-to-head is "not width-confounded at the magnitude that matters" is
  **withdrawn**: it is unresolved, not excluded, and resolving it needs `--features`.

And the shuffled row is why the control runs three permutations rather than one: at 12 episodes
the first permutation of the state vector scored **+0.1344** on gripper while its two siblings sat
at −0.0038 and −0.0273. A single permutation is a draw from a null with a fat tail, not a null —
and a 0.13 "control" is exactly the number that gets read as a leak and sends someone hunting for
a bug in the split that is not there.

**Not controlled:** `carried_state` says what happens to *one 32-dim signal* at these widths on
this split. It does not say what *these backbones' features* do at matched width, which needs
`--features` and therefore needs the pooled tensors off the Space. Nothing here should be read as
the width-matched backbone comparison; the artifact labels the mode for exactly that reason.

### 7c. What this does not settle

Everything in §4's "what this does not test" still applies — frozen features, linear readout, no
fine-tuning. Three things specific to this table:

- **Whether the 0.04 between the two backbones is width is open.** §7b withdrew the claim that it
  is not: the arm that can read width cannot separate 6144 from 8192 at this sample size, and its
  noise is the size of the delta. That is a limit of the control, not evidence either way.

- The headline number is each report's **val-selected** block pair. The `measured_*` and
  `heuristic_*` pairs are fixed block indices that mean different depths in the two Spaces, and
  the best single block is chosen on test. All four are in the artifact, and the ranking is not
  stable across them either: `heuristic_*` puts Cosmos ahead at 12 and 24, `measured_*` at 24, and
  the val-selected and best-block readings put Wan ahead at both. Only at 48 do all four agree.
  §1's archived Cosmos number, 0.359, *is* `heuristic_18_26` — the one arrangement at 12 episodes
  where Cosmos leads. A further reason not to read a 0.04 delta as a ranking.
- **What would change the picture** is not another frozen bake-off. §4's Cosmos-Predict2 action
  port is still the only untested thing on this page, and §7a is evidence for running that arm at
  more than one corpus size rather than evidence about which backbone to prefer.

To re-run: `.venv/bin/python scripts/compare_backbones.py --assume-default window_select`. It cost
about three minutes of ZeroGPU forwards across the six calls (peak 24.61 GB for Wan, 36.2 GB for
Cosmos) and no redeployment. Every report it fetched is kept under
`runs/backbone_eval/reports/`, so the tables can be reassembled with `--from-reports` and no GPU.

---

**Sources for the licence and geometry claims in §3** (verified 2026-08-05, everything else on
this page is a repo artifact):
[Tencent Hunyuan licence text](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE) ·
[HunyuanVideo 1.5 technical report](https://arxiv.org/html/2511.18870v1) ·
[NVIDIA Cosmos 3 launch](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) ·
[Cosmos Predict/Transfer 2.5](https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5)
