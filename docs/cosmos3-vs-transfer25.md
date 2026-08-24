# Cosmos 3 against Cosmos-Transfer2.5 — what is verified, and what it would cost us

**Investigated 2026-08-25 against primary sources only: the two GitHub repositories, the HuggingFace
model cards, and the HuggingFace API. Read-only: no file under `src/`, `scripts/`, `configs/` or
`cluster/` was modified, and no job was submitted.**

This is a **findings document with options**, not a recommendation. It registers no rule, signs
nothing, discharges no blocker, and changes no gate. `T40_RULE_V1` §1 binds in full: nothing here
licenses generating a corpus or training a weight.

`PR-08`'s header names the generator — *"Generator: **Cosmos-Transfer2.5, frozen** — fine-tuning a
generator is [[T-041]] and is a different document."* That string is the premise this document puts
a question against. It does not answer it; §8.4 says why answering it is a rule change and whose
call it is.

> **One constraint on how this document is written, and it is not stylistic.** `PR-07` §6 forbids
> **any statement about GR00T** on this project's VOID verdict. GR00T is therefore named here in
> exactly one role — the **consumer** that `T40_RULE_V7` fixes `PR-08` §8 item 2 on
> (`docs/contracts/vla-training-consumer.md` §0). Nothing in this document says, implies or estimates
> how it performs. Anything a reader thinks they can infer about that from this file, they cannot.

---

## 0. Headline

| question | answer |
|---|---|
| is Cosmos-Transfer2.5 dead? | **no — FROZEN.** Soft-deprecated in its README, not archived, not withdrawn, no HF or docs banner |
| does Cosmos 3 have a transfer/restyle equivalent? | **yes, and the conditioning is a superset** (adds WSM) |
| is it a drop-in replacement for `97_transfer25_restyle.sbatch`? | **no.** Four reasons, §6.2 |
| does any released Cosmos 3 checkpoint support Unitree G1? | **no.** The string "Unitree" appears **zero** times across the README, the four model cards and all 1 148 paths of `cosmos-framework` |
| can a new embodiment be post-trained in? | **yes — demonstrably, and it is code, not config.** Five things must be added, §4 |
| minimum dataset size for that? | **NOT FOUND.** Not stated anywhere. Calibration only: DROID reference is 76 K trajectories; ours is 402 episodes |
| licence direction | **upgrade.** OpenMDW-1.1, ungated, no AUP, no field-of-use limits, no copyleft — against Transfer2.5's gated NVIDIA Open Model License |
| does anything here touch the consumer? | **no.** `T40_RULE_V7` fixes it on `nvidia/GR00T-N1.7-AppleToPlate` and this document does not go near that |
| generation licensed | **no** |
| training licensed | **no** |

---

## 1. Deprecation status — frozen, not broken

**Cosmos 3.** Released **2026-05-31**. `NVIDIA/cosmos-framework` was created **2026-05-19**; it and
its companion Cosmos 3 repository were both **last pushed 2026-08-24** — i.e. yesterday. This is a
live tree.
<https://github.com/NVIDIA/cosmos-framework>

**Cosmos-Transfer2.5.** Last release **v1.5.4, 2026-05-13**. Last commit **2026-06-07**, subject
*"Add Cosmos 3 README redirect"*. Its README then says, verbatim:

> *"This repository is no longer under active development and will receive only limited maintenance
> updates. Future model releases, features, documentation, and community support will be focused on
> Cosmos 3. […] We encourage all users to migrate to Cosmos 3"*

<https://github.com/nvidia-cosmos/cosmos-transfer2.5>

**The nuance is the finding, and it cuts our way.** Three things that would make this urgent are
**not** true:

- the repository is **not archived** on GitHub — issues and clones still work;
- the HuggingFace model card for `nvidia/Cosmos-Transfer2.5-2B` carries **no deprecation notice**;
- `docs.nvidia.com/cosmos/latest/transfer2.5/` has **no banner**.

So Transfer2.5 is **frozen**, not broken. Weights already staged at a pinned revision
(`cluster/discoverer/99_stage_transfer25_weights.sbatch`, `TRANSFER_MODEL_ID` /
`TRANSFER_MODEL_REVISION`) keep working exactly as they did. What has stopped is *future* work on it:
no new features, no new checkpoints, and bug fixes only on a "limited maintenance" basis. **Nothing
about `PR-08`'s current path has been invalidated by this. It has been given an end date nobody has
named.**

---

## 2. What is actually released as a policy

| checkpoint | base | released | action space |
|---|---|---|---|
| `nvidia/Cosmos3-Nano-Policy-DROID` | 16 B | 2026-05-31 | DROID-only, **8 D** (7 joints + gripper) |
| `nvidia/Cosmos3-Edge-Policy-DROID` | 4 B | 2026-07-20 | DROID-only, **8 D** (7 joints + gripper) |

Action chunks `[16, 8]` or `[32, 8]`, at 5 Hz or 15 Hz.

<https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID> ·
<https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>

### 2.1 Trap 1 — the Nano-Policy card's embodiment list is base-model boilerplate

**This is the finding most likely to cost someone a week, and it must not be softened.**

The `Cosmos3-Nano-Policy-DROID` card carries the **base model's** wide embodiment list — camera 9 D,
AV 9 D, egocentric 57 D, Franka 10 D / dual-Franka 20 D, Agibot 29 D, UR 10 D, Google robot 10 D,
WidowX 10 D, UMI 9 D — as inherited boilerplate. **Read at face value it says the policy supports ten
embodiments. It does not.**

Two independent sources say DROID-only: the **Edge**-Policy card, and the actual training recipe
`docs/action_policy_droid_posttrain.md` in `cosmos-framework`. **The recipe is authoritative.**

**This repository has already measured the same trap from the other side, and it agrees.** E-02
(`subprojects/edge-wam/tasks/E-02-*.md`, 2026-08-16) recovered every embodiment's trained width from
the checkpoint tensors by per-output-channel norms against the untrained-row noise floor, via
`scripts/probe_cosmos3_domain_rows.py`:

- base `nvidia/Cosmos3-Edge` — **10 of 32 rows trained**, `agibotworld` at width 29 among them;
- released `nvidia/Cosmos3-Edge-Policy-DROID` — **exactly ONE row trained**, `droid`, width 8.
  Every other row, `agibotworld` included, is **back at random init**.

So the card's list is not merely stale prose: on the policy checkpoint the rows it names are
**measurably untrained**. E-02's consequence — *"a row that is not trained is not an embodiment"* —
is the correct reading, and `Cosmos3EdgeConfig` deliberately registers no G1 row for that reason.

### 2.2 Trap 2 — `G1_omnipicker_calibrated.urdf` is **not** a Unitree G1

**Record this prominently. A grep for "G1" in `cosmos-framework` produces a false positive that looks
exactly like the thing we want.**

`cosmos_framework/data/generator/action/robot_assets/G1_omnipicker_calibrated.urdf` is **AgiBot
Genie-1**, the 29 D humanoid. Three independent confirmations, all inside the file or one call away:

1. its XML declares `<robot name="genie">`;
2. its header records that it was autogenerated from
   `genie_robot_description/xacro/genie.robot.xacro`;
3. it is loaded by `agibot_fk.py`, through `get_agibot_world_urdf_path()`.

There is no Unitree asset in that tree. **"Unitree" appears zero times** across the Cosmos 3 README,
the four relevant model cards, and the entire 1 148-path `cosmos-framework` file listing.

This is a strictly stronger version of a bound `docs/action-labels.md` §3b already carries and that
E-02 already sharpened (*"no G1, no Dex3"* — AgiBot **is** a supported humanoid, at 29 D, but its
29 D is end-effector pose obtained by FK, not a joint vector, so it is not the near-neighbour of our
28 D that the name suggests).

---

## 3. Post-training onto a new embodiment — supported, and it is **code, not config**

### 3.1 There is no "bring your own embodiment" guide — NOT FOUND

`docs/custom_dataset.md` is about the generic VLM dataflow and **says nothing about action spaces**.
No document in the repository walks through adding an embodiment.

### 3.2 But it is demonstrably not DROID-only

`docs/action_policy_libero_posttrain.md` is a **complete second recipe for a different embodiment**,
with its own dataset class, its own normalizer stats, its own experiment config, its own launch
script and its own policy server. The path exists; it is just not written up as a general procedure.

### 3.3 What has to be added — five things, all code

1. **Register the embodiment in two hard-coded dicts** in
   `cosmos_framework/data/generator/action/utils/domain_utils.py`: `EMBODIMENT_TO_DOMAIN_ID` and
   `EMBODIMENT_TO_RAW_ACTION_DIM`. `get_domain_id()` raises `KeyError` on anything unregistered.

   **That registry already contains embodiments that are not in the shipped conditioning list** —
   BEHAVIOR-1K R1Pro 23 D, RoboTwin ALOHA 14 D, ManipArena 20 D. That is evidence NVIDIA adds
   embodiments this way **routinely**, rather than by an architecture change. It matches E-02's
   independent finding that the name→id map is ordinary Python with 14 unassigned ids, while the
   id→weights are trained parameters.

2. **Declare the action layout via the DSL** in `utils/action_spec.py`, e.g.

   ```python
   build_action_spec(Joint(n=14, label="arm"), Joint(n=14, label="end"), Joint(n=2, label="gripper"))
   ```

3. **Subclass `BaseActionLeRobotDataset`.**

4. **Ship normalizer stats.** `_load_norm_stats` raises `FileNotFoundError` **by design** — there is
   no silent fallback. **NOT IN THE OSS REPO:** the generator `compute_action_stats.py` is referenced
   in a docstring but lives at an internal NVIDIA path and does **not** ship. So we write our own
   q01/q99 extraction. This is small, but it is work nobody would budget for from reading the docs.

5. **Register a Hydra experiment + TOML + launch script + policy server.**

### 3.4 Our corpus format is already correct

LeRobot v3.0 is **first-class**: the framework imports upstream
`lerobot.datasets.lerobot_dataset.LeRobotDataset` directly. No converter, no shim.

*(Note for whoever picks this up: `docs/contracts/vla-training-consumer.md` §0 records that corpus A
— AppleToPlate — is LeRobot **v2.1**/43-dim, and corpus B — `unitreerobotics/G1_Dex3_*` — is
v3.0/28-dim. "Our corpus format is already correct" is a statement about the v3.0 corpus. It is not
a claim that corpus A needs nothing.)*

### 3.5 A concrete blocker for our state column

**`use_state` is NOT a separate proprioception stream, and reading it as one would design the wrong
pipeline.**

In `droid_lerobot_dataset.py` the state is **concatenated as an extra leading timestep of the same
`[T, D]` action tensor**. It must therefore have the **same dimensionality as the action**. There is
a guard: `use_state` is valid **only** with `action_space=joint_pos`.

**Consequence for us, stated precisely:** our **43-dim** state (seven groups incl. legs and waist,
`vla-training-consumer.md` §0) cannot be fed alongside a **28-dim** action. It must first be
projected into the action space. That projection is a design decision — which of the 43 channels
survive, and what the discarded ones cost — and it does not exist anywhere in this repository today.

*(This refines E-02's §4, which records the DROID policy as *"un-normalized, with `use_state`
proprioception"*. That phrasing is correct but can be read as a second input stream; the mechanism
above is what it actually is.)*

### 3.6 Minimum dataset size — **NOT FOUND**

Not stated in the README, the model cards, either post-training recipe, or the framework docs.

**For calibration only, and not as a threshold:** the DROID reference corpus is **76 K trajectories /
~350 hours**. Ours is **402 episodes**. Recorded as **untested, not ruled out** — the recipes do not
say the small number fails, and nothing here licenses either the hope or the despair.

### 3.7 Hardware, from the published recipes

| recipe | scale |
|---|---|
| DROID reference | **256 ranks** — 64 nodes × 4 GB200 — 10 000 iters, global batch 8192 |
| DROID, downscaled | the TOML documents **one 8-GPU node** via `replicate_degree=1`, `grad_accum_iter=32` |
| LIBERO | **16 GPUs**, 2 000 iters |

E-02 already carried the relevant warning forward and it stands: **the 256-GPU reference is for
Cosmos3-Nano (16 B), not the 4 B Edge target. Do not size our allocation against it.**

---

## 4. Third-party precedent — indicative, and **unverified**

`JeffrinSam/Cosmos3-Nano-G1-BrainCo-PolicySFT` on HuggingFace post-trained Cosmos3-Nano onto a
**Unitree G1** (tags: `unitree-g1`): 26 D joint-angle trajectory at 15 Hz, on an apple-picking
dataset. `domain_name="g1_brainco"`, `raw_action_dim=26`, q01/q99 stats. Training: **7 × A100 80 GB**
FSDP, 10 000 iters, ~22 s/iter → **~61 hours**, lr 5e-6, training only four adapter modules
(`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`) with the Qwen3-VL visual encoder and the text
backbone **frozen**.

<https://huggingface.co/JeffrinSam/Cosmos3-Nano-G1-BrainCo-PolicySFT>

**This is a community model card, it is not verified, and it contains at least one outright error:**
it calls Nano **"8B"** when NVIDIA states **16 B**. A card that is wrong about the size of the model
it fine-tuned is not a specification.

**What it is worth:** it is evidence that the §3.3 route is walkable by somebody outside NVIDIA, and
an order-of-magnitude cost (tens of GPU-days on A100-class hardware, adapter-only). **What it is not
worth:** any number in it may not be quoted as a measurement, and its recipe may not be adopted
without re-derivation from the framework source.

---

## 5. Transfer / restyle in Cosmos 3

### 5.1 The equivalent exists, and the conditioning is a **superset**

`TransferHintKey` in `cosmos_framework/inference/args.py:225`:

| hint | in Transfer2.5 | in Cosmos 3 |
|---|---|---|
| EDGE | ✅ | ✅ |
| BLUR | ✅ | ✅ |
| DEPTH | ✅ | ✅ |
| SEG | ✅ | ✅ |
| **WSM** (world-surface-map) | — | ✅ **new** |

Multi-control with a per-hint `weight` is supported (Cosmos Framework only). On conditioning
vocabulary alone, Cosmos 3 is strictly ahead.

### 5.2 Four things stop it being drop-in

1. **`Cosmos3-Edge currently doesn't support video-to-video transfer`** — README footnote 2. So a
   restyle must run on **Nano (16 B)** or **Super (64 B)**, i.e. **not** the 4 B model the edge-wam
   half would deploy. Generator and deployed policy are different models under Cosmos 3, exactly as
   they are today.

2. **There are no separate Transfer weights.** Transfer is a **mode** of the base checkpoints —
   `model_mode: "video2video"` — not a distinct model. Verified by enumerating **every** `nvidia/*`
   Cosmos3 repository on the HuggingFace API: there is no `Cosmos3-*-Transfer`.

3. **Only `edge` and `blur` are derived on the fly.** For **DEPTH / SEG / WSM**, `transfer.py`
   raises *"Missing pre-computed control input"*, and **no depth estimator and no segmenter ship with
   Cosmos 3**. Transfer2.5 has had on-the-fly depth and segmentation since October 2025.
   **On this axis Cosmos 3 is a REGRESSION in convenience.** See §7.3 — this is the single most
   decision-relevant consequence in this document.

4. **NOT FOUND: any Cosmos 3 equivalent of Transfer2.5's `mask_path`** spatiotemporal control
   masking. Stated as *not found*, not as *absent* — the framework is large and this was a targeted
   search, not an exhaustive one.

---

## 6. Licence — this direction is an upgrade

| | `nvidia/Cosmos-Transfer2.5-2B` | Cosmos3 checkpoints (incl. **both** policy models) |
|---|---|---|
| licence | NVIDIA Open Model License | **OpenMDW-1.1** |
| gating | **`gated: auto`** (`99_stage_transfer25_weights.sbatch:13` records this) | **ungated** — fetched from the HF API with no token |
| acceptable-use policy | yes | **none** |
| field-of-use limits | yes | **none** |
| guardrail-circumvention termination | **yes** | **none** |

OpenMDW-1.1, as carried in the model cards' `license` field: commercial use of the weights **and of
fine-tunes** is permitted; there is an **express patent grant**; **no copyleft** — verbatim, *"No
further 'copyleft' or 'share-alike' requirements are imposed by OpenMDW-1.1"*. The **only** ongoing
obligation is to retain the licence text and origin notices when distributing.

**Two things recorded as costs rather than glossed:**

- **Termination is defensive but broad.** It covers **copyright** suits as well as patent suits, and
  triggers on *"file, maintain, or voluntarily participate in"*. That is wider than a bare patent
  retaliation clause.
- **AS IS, with no training-data indemnity.**

**Open question, flagged for counsel and not answerable from the licence text:** whether shipping
**fine-tuned weights inside a robot binary** counts as *"distribute any portion"*, and therefore
whether the notice-retention obligation attaches to a shipped robot. This bears directly on
edge-wam's half and on nothing in data-factory's. It is not a blocker for any measurement; it is a
blocker for a product decision, and it belongs to a lawyer, not to a session.

The gating difference is not cosmetic here. `docs/handoff.md` §4 records that
`nvidia/Cosmos-Guardrail1` being a **gated** repo killed job 187249, and that accepting a licence is
the account holder's act and not an agent's. An ungated family removes that class of failure.

---

## 7. What this means for the data factory

### 7.1 The data factory is consumer-agnostic **by construction**, which is why this question is answerable at all

`PR-08` §2's load-bearing argument is one sentence: **the labels do not come from the generator.**

> *"Here the actions are the recorded teleop trajectory, carried over unchanged, and the generated
> pixels are only an input perturbation."*

The product of this pipeline is therefore **a LeRobot corpus carrying real labels** — and **who
trains on it is a separate decision from which generator restyled it.** That separation is not a
convenience; it is the whole reason a generator swap can even be discussed without reopening the
consumer.

`T40_RULE_V7` fixes `PR-08` §8 item 2 on **`nvidia/GR00T-N1.7-AppleToPlate`** as the consumer's
corpus (`docs/contracts/vla-training-consumer.md` §0), and item 2's fourth clause — *"the action
labels come from the source recording, never from the generator"* — is unchanged, verbatim, and is
the load-bearing one. **Nothing in this document touches either.** A generator change is a change to
the producer's tooling; the contract's subject, its 43-dim corpus A description, and its consumer
are all untouched by every option in §8.

### 7.2 What of the existing gate work is portable across generators, and what is not

**Portable — these are statements about OUR estimators and OUR corpus, not about the generator:**

| asset | why it survives a generator swap |
|---|---|
| `GEOM_TOL = 0.4786 px` (402/402 ep, 171 625 frames) | a property of the **source** clips' own per-step centroid displacement |
| `EST_DRIFT_P95 = 0.2361 px` (MuJoCo route) | a property of **our** estimator against ground truth |
| the `GATE_QUALIFICATION_BLOCKERS` in `scripts/estimators/apple_sam2.py` | all three are about our masker, none about the generator |
| G0a / G0b / G0c | G0a is label integrity, G0b is geometry invariance, G0c composites the real robot back. All generator-agnostic by design |
| the robot-mask area work (`106_measure_robot_mask_area.sbatch`, `robot_composite.py`) | a distribution over **source-derived** masks |
| the blind adjudication instrument (`scripts/build_identity_prompt_sheet.py`, `runs/t040-identity-prompt/`, 40 blank verdicts) | an instrument for judging clips, indifferent to what made them |
| the committed style partition (`T40_STYLES_V1`, `configs/transfer25/pr08_style_partition.json`) | a pre-commitment about **which styles** are train vs eval — text, not weights |

**Not portable:**

- **`cluster/discoverer/97_transfer25_restyle.sbatch`.** It is built around Transfer2.5's API from
  the ground up: `CONTROL=depth:0.5,seg:0.5`, `RESTYLE_DRIVER=scripts/restyle_transfer25.py`,
  `TRANSFER_ENV`/`TRANSFER_MODEL_ID` written by `98`/`99`, and a whole comment block citing
  `docs/transfer25-api.md` file-and-line against `nvidia-cosmos/cosmos-transfer2.5 @ main`. Under
  Cosmos 3 the driver, the env, the model id and the control-spec grammar all change. The **ledger,
  the staging, the chunking and the composite step** are the reusable parts; the inference call is
  not.
- **`PR-08` §8 item 3's TIMING number.** Any figure measured on **Transfer2.5-2B on an H200** is a
  statement about a 2 B model; Cosmos3-Nano is **16 B** and Super is **64 B**, so the number would
  simply be wrong under a swap.

  > **Correction to a premise this document was written against, checked against the repo.**
  > **§8 item 3's TIMING number does not exist yet.** It is OPEN. The only measurement of a
  > Transfer2.5 frame cost anywhere in this project is **1.16 s/frame** (96 frames in ~111 s of H200
  > time, job 189926, the V8 hallucination probe), and the T-040 notes state in capitals that
  > **it is not §8 item 3's measurement and may not be a budget line** — one diagnostic clip, likely
  > optimistic against episodes averaging ~427 frames. So what a generator swap invalidates here is
  > a **sizing diagnostic and the method that would produce the real number**, not a committed
  > budget line. Nothing is lost that was ever of record.

### 7.3 Our estimator stack becomes **more** load-bearing under Cosmos 3, not less

**This is the most decision-relevant consequence in this document, and it runs opposite to the
intuition that a newer generator needs less from us.**

`97_transfer25_restyle.sbatch:371` already states the current arrangement explicitly: *"our manifest
carries no depth or segmentation maps, so each control block Transfer2.5 has to **ESTIMATE** is GPU
time the measurement must include."* Under Transfer2.5, the generator estimates depth and
segmentation itself, and we pay for it in GPU seconds.

**Under Cosmos 3 that option is gone.** `transfer.py` **refuses** with *"Missing pre-computed control
input"* for DEPTH, SEG and WSM, and **no estimator ships**. The producer would have to supply those
maps itself — which means `scripts/estimators/apple_sam2.py` stops being a *measurement* instrument
for `EST_DRIFT_P95` and additionally becomes a **production** instrument feeding the generator.

Two consequences, both worth stating plainly:

1. **The gate-qualification work on `apple_sam2.py` is an asset either way**, and under Cosmos 3 it
   is on the critical path twice over. `GATE_QUALIFIED = False` with three named blockers — nobody
   has looked at a mask; the operating point is unmeasured on this corpus; per-frame re-detection is
   not upstream's propagation — is not a Transfer2.5-specific liability. It gets **more** expensive
   to leave open, not less.
2. **`PR-08` §4's whole argument for "the generator's own segmenter, down to the checkpoint id"
   changes character.** Today `apple_sam2.py` copies Transfer2.5's operating point verbatim
   (`threshold=0.15`, `text_threshold=0.25`, one `(0.10, 0.10)` retry, highest-score box) precisely
   so that the drift we budget is the drift the generator commits. Under Cosmos 3 **there is no
   generator segmenter to match** — ours *is* the generator's. That makes §4's strong reading
   trivially satisfied and simultaneously removes the external anchor that justified those exact
   constants. A new pre-registration would have to say what anchors them instead.

### 7.4 Changing the generator is a **rule change**, not a config change

`PR-08`'s header names *"Generator: **Cosmos-Transfer2.5, frozen**"*. Swapping it changes that
document's premise. `docs/handoff.md` §3 is unambiguous: **rules are versioned, never edited in
place** — *"a gate rewritten after seeing its output is not a gate"*.

So a generator swap requires a **new V-document under `T40_RULE_V*` or a new pre-registration**, and
it may **not** be a quiet substitution inside an sbatch. Two further constraints a drafter must
carry:

- `PR-08`'s header also says **fine-tuning a generator is [[T-041]] and is a different document.**
  Anything that post-trains a Cosmos 3 checkpoint is on that side of the line, not this one.
- `PR-07` §7's freeze names, by name, *"any Cosmos3-Super generation, any Cosmos3-Edge work"*,
  frozen **until T-39 reports**. T-39 reported (2026-08-16), so the freeze's stated condition has
  lapsed on its letter — **but options (b) and (c) in §8 are exactly the two things it named**, and
  a session is not the right party to declare its own path unfrozen. Confirm the status with the
  owner before either is drafted.

---

## 8. Three options, costed. **This document does not pick one.**

### (a) Stay on frozen Transfer2.5 for augmentation

| | |
|---|---|
| what it costs | nothing new. `97`/`98`/`99`/`99b` work as written; weights are staged at a pinned revision |
| what it buys | `PR-08` proceeds on its registered premise, no new rule document, no re-derivation |
| what falls away | nothing today |
| what it accepts | a generator with an unnamed end date, a `gated: auto` licence, no future fixes, and a slowly widening gap to the tooling everyone else migrates to |
| open blockers unchanged | §8 items 3 and 4; `GATE_QUALIFIED = False`; G0b's budget is still `null` in `configs/transfer25/pr08_geom_tol.json` |

### (b) Migrate augmentation to Cosmos3-Nano (16 B) or Super (64 B)

| | |
|---|---|
| what it costs | a new V-document or pre-registration (§7.4); a new inference driver replacing `restyle_transfer25.py`; a new `docs/cosmos3-api.md` doing what `docs/transfer25-api.md` did; **the producer must now generate depth/seg/WSM itself** (§7.3); a fresh TIMING measurement on a 4×–16× larger model |
| what it buys | a maintained tree, a superset of conditioning hints incl. WSM, an ungated OpenMDW-1.1 licence, and per-hint weights |
| what falls away | `97`'s inference call and its control-spec grammar; the "match the generator's segmenter" anchor in `PR-08` §4 (§7.3 item 2) |
| what survives | every row in §7.2's portable table, and `97`'s ledger / staging / chunking / composite scaffolding |
| the number nobody has | GPU-h. A 16 B or 64 B generator against a 2 B one, over a committed partition already sized at **~1 380 GPU-h ≈ 27 %** of the 5 000-h allocation at 1.16 s/frame. **That multiplier is unmeasured and this document does not estimate it.** |

### (c) Skip the restyle entirely and post-train `Cosmos3-Edge-Policy` on the 402 episodes

| | |
|---|---|
| what it costs | the five items in §3.3, all code; the 43→28 state projection in §3.5; normalizer stats we write ourselves (§3.3 item 4); a pre-registration (E-05 exists for exactly this and is unstarted) |
| what it buys | no generator in the loop at all — no G0b, no G0c, no `EST_DRIFT_P95` in a gate, no restyle GPU-h |
| what falls away | most of `PR-08`. The style partition, the arms A/B/C/D design, the identity-prompt judge, and the geometry gates all exist to adjudicate **generated pixels**, and there would be none |
| **the honest objection** | **it does not remove the data problem. 402 episodes is still 402 episodes.** `PR-08` §9 says this of itself and it is just as true here — this option changes what is trained, not how much data exists. And §3.6 records the minimum dataset size as **NOT FOUND**, against a 76 K-trajectory reference corpus |
| **and it is a different project shape** | it is a **policy**, which is **edge-wam's** half, not data-factory's (`subprojects/README.md`). It lands on **E-04 / E-05 / E-06**, not on T-040. Treating it as an option "for T-040" would move work across the 2026-08-15 split without saying so. E-02 also records that any humanoid warm start must come from base `nvidia/Cosmos3-Edge`, **not** from the policy variant, because the policy's `agibotworld` row is back at random init |

**The choice between (a), (b) and (c) is the project owner's.** All three are live; none is
dominated; and (c) is not a variant of the other two but a different question asked of a different
sub-project.

---

## 9. What this document cannot answer

- **Whether the restyle helps at all.** That is `PR-08`'s headline and it has never been measured.
  A better generator does not make an unmeasured intervention work.
- **Anything about the consumer.** `PR-07` §6 forbids it, and this document obeys.
- **The GPU-h multiplier for a 16 B or 64 B generator.** Unmeasured. Not estimated here.
- **Whether 402 episodes is enough for any post-training route.** NOT FOUND, §3.6.
- **Whether shipping fine-tuned weights in a robot binary is "distribution"** under OpenMDW-1.1. §6,
  for counsel.
- **Whether `PR-07` §7's freeze is discharged.** Its condition has lapsed on the letter; whether that
  discharges it is the owner's reading, not a session's (§7.4).

---

## 10. Provenance

| | |
|---|---|
| kind | decision memo. **Registers no rule, signs nothing, discharges no blocker** |
| date | 2026-08-25 |
| sources | GitHub (`NVIDIA/cosmos-framework`, `nvidia-cosmos/cosmos-transfer2.5`), HuggingFace model cards, the HuggingFace API. Primary sources only |
| verified in-repo against | `PR-08` §§1–3, §8; `docs/contracts/vla-training-consumer.md` §0, §6; `subprojects/edge-wam/tasks/E-02-*.md`; `scripts/estimators/apple_sam2.py`; `cluster/discoverer/97_transfer25_restyle.sbatch`; `.mc/tasks/todo/T-040-*.md` |
| corrects | one premise this memo was written against — `PR-08` §8 item 3's TIMING number **does not exist**; only the 1.16 s/frame sizing diagnostic does (§7.2) |
| refines | E-02 §4's `use_state` phrasing — it is a concatenated leading timestep of the action tensor, not a separate stream (§3.5) |
| does **not** change | `PR-08`'s header, `T40_RULE_V1`, `T40_RULE_V7`, the consumer contract, any gate, any threshold, any config |
| files modified | **none** outside `docs/` and one dated note in `.mc/tasks/todo/T-040-*.md` |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
