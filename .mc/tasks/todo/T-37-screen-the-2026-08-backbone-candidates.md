---
id: T-37
aliases:
- T-37
title: "Screen the 2026-08 backbone candidates"
slug: screen-the-2026-08-backbone-candidates
status: todo
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- backbone
- eval
- sim
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-05
updated: 2026-08-05
status_note: "The screen itself is written (`docs/backbone-eval.md`); T-38 ran the Wan-vs-Cosmos comparison that came out of it. What is left is the probe of the one axis our record does not already cover."
---

# Screen the 2026-08 backbone candidates

## Description

Screen the 2026-08 backbone candidates, and probe the one thing our record does not already cover
(`docs/backbone-eval.md`). *Prompted by an external survey recommending NVIDIA Cosmos 3 and listing
HunyuanVideo as a second open-weight option. Most of it is already answered in this repo, and the
doc says so with the receipts: **Cosmos3-Nano was probed on 2026-07-26 and lost** (T-24 — joints
0.359 / gripper 0.708 against the state-only ridge's 0.456 / 0.881), T-26 then removed the mean-pool
explanation, and Cosmos3's VAE **is** the Wan2.2 VAE, so neither direction of that comparison can be
attributed to the latent space.* **Two candidates die on a screen that needs no GPU.**
**HunyuanVideo (13B and 1.5/8.3B) fails on licence in the EU** — the Tencent Hunyuan Community
License defines Territory as "the worldwide territory, **excluding** the territory of the European
Union, United Kingdom and South Korea" and grants "for the Territory only", plus a
no-improving-other-AI-models clause that is squarely what a WAM adapter does with backbone features.
Same criterion class that decided OD-04 for Wan and that OD-06 flags against FLUX 3. It is also the
*best* candidate on memory (~13.6 GB peak at 720p × 121 frames) — the opposite shape of failure to
Cosmos's, which is why both are recorded rather than one. **Cosmos3-Nano fails on memory for the
5090**, from our own artifact rather than an estimate: the T-24 probe peaked at **36.2 GB** and
`--generate` at 35.6–36.9 GB on a 96 GB RTX PRO 6000, i.e. **1.8–2.5 GB over the whole 34.36 decimal
GB card**, and no FP8/NVFP4 checkpoint ships. **What survives is Cosmos-Predict2.5-2B**, and for
exactly one reason: it is the only candidate offering a **pretrained action port** — actions enter
through an action-embedder MLP added to the DiT's timestep embeddings, where ours is a state token
bolted into a text-context slot and trained from scratch on 402 episodes. T-24 never used Cosmos's
action port at all. **The probe isolates that and nothing else:** the unchanged T-15/T-24 experiment
— same windows, labels, split, ridge code — run twice against the same weights in the same process,
port fed the true past actions vs. port fed zeros (the `set_lora_enabled` pattern from T-35, which
is why the delta is trustworthy where a second model build would not be). **Two gates, fixed before
the run:** G1 `R²_joints(A) > R²_joints(B)` — if equal, the pretrained conditioning is not linearly
readable and **the backbone question closes**, because every remaining candidate fails the same
screen criterion; G2 `R²_joints(A) > 0.456` **and** `R²_gripper(A) > 0.881`, the exact bar Wan and
Cosmos3 both failed on the same split. Both pass → OD-04 reopens with evidence. **Bounds stated:**
like T-15/T-24 this measures *frozen* features under a *linear* readout, so a G1 pass buys a reason
to spend GPU hours on a Cosmos LoRA, not evidence that one would land — T-16 is the fine-tuned arm
and it is negative. Cosmos-Transfer2.5 off the Isaac backend (depth/segmentation/Canny → photoreal)
is deliberately **out of scope**: it is synthetic training data, it collides with "sim frames are
NOT training data" (T-25, `docs/sim.md`), and T-36 already priced generated video as supervision at
worse-than-a-frozen-frame. Overturning that is its own pre-registration with `screen_corpus.py` on
the generated corpus — *✅ **screen + gate calibration done 2026-08-05 on CPU; the GPU arm is not
run.** Building the action representation turned up two things that change the experiment.* **(1)
The gate I wrote was too low, and the CPU could prove it.** G2 read `joints > 0.456 and gripper >
0.881` — the state-only ridge. But arm A is *fed* the past actions and T-34 measured lag-1
autocorrelation **0.927**, so the bar has to be what a ridge gets from the probe's own inputs with
no video model at all. `scripts/probe_action_baselines.py` measures that, importing
windows/split/ridge from `hf_job_wan_probe` unchanged. **`state_only` reproduces the archived 0.456
/ 0.881 to four digits** at 12 episodes — the check that these are T-24's windows — but **the floor
is not a constant**: 0.4563 → 0.4879 → 0.5129 at 12 / 24 / 48 episodes, so quoting 0.456 as an
absolute compares across sample sizes. **`past_ee` — the Bridge-shaped tensor the port actually eats
— does not beat the floor.** It reads 0.4576 vs 0.4563 at 12 episodes and 0.3954 vs 0.5129 at 48; it
*degrades* as the corpus grows, which is the signature of a small-n result, and the 12-episode
reading is the one that would have been quoted. **The robust finding is `past_joint_proj + state` at
0.540 / 0.539 / 0.541 joints and ~0.911 gripper at 48 episodes, three seeds, spread 0.002** — past
actions plus proprioception beat proprioception alone by ~+0.03. `past_joint`'s raw **−0.0950** was
256 dims against 56 training rows, not information: matching the width to 112 reverses it. The
shuffled control sits at 0.000 ± 0.006 at every size, so none of it is the split leaking.
**Corrected G2: beat `past_joint_proj + state` computed on the probe's own windows** — 0.5407 /
0.9601 at 12 episodes, 0.546 / 0.911 at 48. For scale, the best frozen-backbone number ever recorded
in this project is **0.399** (T-24, best single Cosmos3 block, 12 episodes), so G2 now asks for 0.54
where nothing has cleared 0.40. **(2) The GPU arm costs more than the screen assumed.** The
checkpoint with a published pretrained action port is
`nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned` — **Predict2, not 2.5** — post-trained on
Bridge, taking 7-D `[x,y,z,roll,pitch,yaw,gripper]` EE *displacements*, 12 actions → 12 frames at
640×480/4 fps, 32.54 GB quoted. It is **not documented as diffusers-native** and our probe harness
is diffusers-based, so this is not "swap the feature extractor". Bridge is a WidowX at 4 fps against
our G1 at 30 fps with a mean per-frame EE displacement of **1.6 mm** (episode 0, 0.94 m of path over
590 frames), so arm A is out of distribution on scale alone. **Shipped:**
`src/wam/robot/kinematics.py` (FK, canonical 15 joints → EE pose; no physics, no stepping, shares
`configs/sim/g1_scene.xml` with the sim backend so an FK number and a rendered frame cannot disagree
about the arm), `scripts/probe_action_baselines.py`,
`runs/backbone_eval/action_baselines{,_ep24,_ep48}.json`, **13 tests, 7 mutations killed** —
including the one that survived the first pass: a **transposed `xmat`** yields the inverse rotation,
a perfectly finite and plausible triple of angles that every shape, range and continuity assertion
accepts, so the convention is now pinned by reconstructing MuJoCo's own matrix from the reported
euler angles

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
