# PR-08 V7 — §8 item 2's consumer contract, amended onto the corpus §3 actually restyles

**Rule `T40_RULE_V7`. Registered 2026-08-22, before any clip is generated, before any weight is
trained, and before any job is submitted. Nothing has been generated; no clip exists.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md) (`T40_RULE_V1`),
[`PR-08-V2-arm-c-frame-matching.md`](PR-08-V2-arm-c-frame-matching.md) (`T40_RULE_V2`),
[`PR-08-V3-seed-schedule.md`](PR-08-V3-seed-schedule.md) (`T40_RULE_V3`),
[`PR-08-V4-t39-gate-premise.md`](PR-08-V4-t39-gate-premise.md) (`T40_RULE_V4`),
[`PR-08-V5-ground-truth-route.md`](PR-08-V5-ground-truth-route.md) (`T40_RULE_V5`) and
[`PR-08-V6-mask-validity.md`](PR-08-V6-mask-validity.md) (`T40_RULE_V6`). **None of the six has
been edited and none may be.** The discipline is `docs/handoff.md` §3 — *"Rules are versioned,
never edited in place. A gate rewritten after seeing its output is not a gate."* V7 is that
versioning, not a revision.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

**Nothing in this document licenses generation, training, or any statement of a result.**
`T40_RULE_V1` §1's prohibition is untouched and binds in full. §8 items 3 and 4 remain **open**;
V7 closes item 2 and nothing else, and an item closed is not a gate lifted.

---

## 0. What V7 does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent. **V7 moves no gate, no threshold, no verdict, no arm,
no clip count, no style, no seed, no ceiling and no estimator setting.** It changes three
descriptive fields in one sentence of `T40_RULE_V1` §8 item 2, and makes the artifact that sentence
points at true.

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined here. V7 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` — definition | unchanged and still **derived, not coined**: the median per-step object-centroid displacement in the **source** clips, measured on the real corpus, computed and committed before generation. V7 supplies **no value** for it |
| `GEOM_TOL` — the step | unchanged: `GEOM_STEP_FRAMES = 1`, one source frame at `fps = 30`, overlapping offsets, as `T40_RULE_V3` §4.3 registers |
| `EST_DRIFT_P95` | unchanged — still the p95 of the object-centroid displacement in pixels between the estimated and the true segmentation, still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still **not a pass**. V7 supplies no value for it |
| **V5's ground-truth route** | unchanged — `T40_RULE_V5` decides *which simulator* renders §4's ground truth. V7 touches no renderer and no capture |
| **V6's mask-validity filter** | unchanged — `mask_validity_min_iou` / `mask_validity_reference` and the frames each side of §6's subtraction is measured on are exactly as `T40_RULE_V6` registers them |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still **VOID** |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start. V7 moves neither term, and changes neither the subtraction nor the ≤ 0 rule |
| **G0c** embodiment | unchanged — the real robot's pixels are unconditionally composited back over every generated frame; robot-mask IoU is recorded as a **diagnostic, never as a gate** |
| **The ladder** | unchanged — **L1** `skill_vs_repeat_pct > 0`, **L2** `ci_skill_vs_repeat_pct > 0` (`ci_` = the task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (V1 §6) | unchanged in every cell, including that **P** requires *both* B − A ≥ floor *and* B − C ≥ floor, that **F** is the generator-attributable case, that **N** is B − A ≤ 0, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **The headline** | unchanged — `skill_vs_repeat_pct` on `EVAL_STYLES`, arm B against arm A, with arm C deciding attributability |
| **Arms A / B / C / D** | unchanged. B is the intervention, C is the generator-fingerprint control, D is diagnostic and never the headline |
| **Arm C's size** (`T40_RULE_V2` §1–§2) | unchanged — R2, frame-matched: 1 identity style × 10 repeats × 402 episodes = 4 020 clips against arm B's 10 × 1 × 402 = 4 020. **Arm B is still not subsampled** |
| **Clip totals** | unchanged — train 4 020, identity 4 020, eval 2 010, whole partition 10 050 over 25 style-instances |
| **The seed schedule** (`T40_RULE_V3` §1) | unchanged — train `[7001..7010]`, identity identical, eval `[7011..7015]` disjoint, assignment by style-instance index |
| **The two-quantity GPU-h ceiling reading** (`T40_RULE_V3` §3) | unchanged. V7 supplies no ceiling value and exempts nothing from one |
| **The committed style partition** | `configs/transfer25/styles.toml` (rule `T40_STYLES_V1`) and its rendering `configs/transfer25/pr08_style_partition.json`. V7 changes **no style, no id, no slug and no prompt string**, and therefore changes no partition hash. Authority remains the sidecars plus `scripts/check_style_partition.py`, never a value quoted in prose. *(Observation, not a registration: the verifier passes at content hash `9334fd01…` as of 2026-08-22. **V7 does not register that value**, exactly as V4 did not.)* |
| **The detection operating point** | unchanged — `box_threshold = 0.15`, `text_threshold = 0.25`, one retry at `(0.10, 0.10)`, highest-scoring box, prompt `"apple."` |
| **`GATE_QUALIFIED`** | still `False`. V7 flips nothing |
| **`GATE_QUALIFICATION_BLOCKERS` / `GATE_QUALIFICATION_DISCHARGED`** | **not edited by V7.** All entries stand verbatim; nothing was moved between them |
| **V4's determination on §8 item 7** | unchanged and undisturbed — item 7 is **CLOSED on `VERDICT N`** issued 2026-08-17 under `T39_RULE_V2` (job 188408, `PR-07-V2-RESULT.md`), a T-39 **VOID** still closes PR-08 rather than opening it, and `PR08_OVERRIDE_T39_VOID` remains **ungranted and not exercised**. V7 neither re-opens, re-signs nor leans on that determination |
| **§8 items 3 and 4** | unchanged and still **OPEN**. V7 produces no throughput number, no GPU-h ceiling, no `GEOM_TOL` and no `EST_DRIFT_P95` |
| **§1's prohibition** | unchanged and still binding in full — nothing is generated, no weight is trained on generated frames, and no number from PR-08 is quoted as a result, until **every** §8 item is closed |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a **P** is a claim about generalising to held-out *generated* appearance, not to real appearance, and it licenses **exactly one thing**: recording a small real shifted eval set and re-running arms A and B against it. It never licenses adding restyled data to any training corpus |

**And, said loudly because it is the failure mode this project keeps having to name:**

- **V7 does not license training on generated frames, or on anything else.** Whether training may
  start at all, and against which label space, is the project owner's separate call
  (`CLAUDE.md`); V7 does not touch it and must not be cited in it.
- **V7 says nothing whatever about `docs/benchmark.md`'s L4 gate**, nor about GR00T's capability.
  `PR-07` §6's prohibition on any statement about GR00T is untouched.
- **Closing §8 item 2 opens nothing.** Two of seven preconditions are still open, and V1 §1 is
  conjunctive.

V7 changes exactly one thing: **which corpus §8 item 2's descriptive fields describe.**

---

## 1. The finding: item 2 did not describe the corpus PR-08 restyles

### 1.1 The sentence

`T40_RULE_V1` §8 item 2, verbatim
(`docs/preregistration/PR-08-photoreal-augmentation.md:222-224`):

> 2. **The consumer contract with `emai/vla-training`**, written down: LeRobot v3.0, 28-dim
>    arms+hands, right hand index-before-middle, and the action labels come from the *source*
>    recording, never from the generator.

Four clauses. Three describe a dataset. The fourth describes where labels come from.

### 1.2 What was already on record

- **`T40_RULE_V2` §4** found that the three descriptive fields belong to
  `unitreerobotics/G1_Dex3_*` — the corpus of **T-043**, route 1 — and not to
  `nvidia/GR00T-N1.7-AppleToPlate`, which `T40_RULE_V1` §3 chose to restyle. V2 §4.4 recorded item
  2 **OPEN** and refused to resolve it, on the ground that choosing which corpus the deliverable is
  is *"a decision about the experiment, not a clerical correction, and it is not an agent's to
  make."* V2 also had to carry the v2.1 claim rather than measure it: *"no `meta/info.json` is
  present on this workstation"* (V2 §4.2).
- **`T40_RULE_V3` §5.3** corrected that last statement. The snapshot's own `meta/info.json` **is**
  present, and it states `codebase_version` `v2.1`, 402 episodes, 171 625 frames, and
  `observation.state` / `action` of shape `[43]`. V3 kept item 2 **OPEN** for the same reason V2
  did.
- **`T40_RULE_V4` §6** carried item 2 forward as **OPEN**, repeating that the choice was the
  owner's.

### 1.3 This is not imprecision. It is a different dataset.

Worth stating plainly, because "the contract was a bit vague" and "the contract described some
other corpus" call for different repairs and only the second one is true here.

The two corpora differ in **version**, in **width**, in **group structure**, in **camera count**
and in **scale**. `unitreerobotics/G1_Dex3_*` is LeRobot v3.0, `float32[28]` flat, arm-first, no
waist and no legs, up to four cameras, 3 152 episodes. `nvidia/GR00T-N1.7-AppleToPlate` is LeRobot
v2.1, `float32[43]` in seven named groups including both legs and the waist, exactly one camera,
402 episodes. A pipeline built to the first specification and pointed at the second does not
degrade; it mis-slices. Nothing in item 2's three fields is a rounding of the truth about
AppleToPlate.

The **fourth** clause is the exception, and §4 below is about it.

---

## 2. The measurement — every replacement field cited to what it was read from

**Primary source, and it is the corpus's own metadata rather than a description of it:**

```
~/.cache/huggingface/hub/datasets--nvidia--GR00T-N1.7-AppleToPlate/
    snapshots/d89c126a713c6632432a607c12661546ff4d6ea9/meta/info.json
        sha256 0e3dd494bace81fcb172f77b5cfe089ed7f7d6babb4b008cb9ada425e9efaf36   (174 lines)
    .../meta/modality.json
        sha256 84e5eb4f708a214923b387e2258336e137b6669bdeab1fdf01142aa967f0d62a   (116 lines)
```

Read 2026-08-22 with `.venv/bin/python`. **These two files were also diffed, on 2026-08-22, against
the checked-in copies the consumer holds at
`/home/humanoid/develop/vla-training/eval/real_reference/{info.json,modality.json}` and are
BYTE-IDENTICAL to them.** That matters: the producer's authority for the on-disk layout and the
consumer's reference copy of it are not merely consistent, they are the same bytes, so the two
sides cannot be reading different specifications of the corpus.

### 2.1 Field by field

| # | V1 §8 item 2 said | measured, for `nvidia/GR00T-N1.7-AppleToPlate` | read from |
|---|---|---|---|
| 1 | LeRobot **v3.0** | **v2.1** | `meta/info.json:2` — `"codebase_version": "v2.1"` |
| 2 | **28-dim** arms+hands | **43-dim**, in **seven** groups, `observation.state` and `action` alike | `meta/info.json:17-25` (`observation.state`, `float32`, `shape [43]`) and `:26-34` (`action`, `float32`, `shape [43]`); group boundaries at `meta/modality.json:2-31` (state) and `:32-60` (action) |
| 3 | right hand **index-before-middle** | **true of the right hand's 7 elements as NVIDIA's own export names them — and only half the story, because the LEFT hand is middle-before-index.** Not measurable from the data on this box: see §5.1 | `/home/humanoid/models/GR00T-N1.7-ApplePnP-V1/exported_leapp.yaml:85-87` (state `right_hand` element names; tensor block `:80-87`) and `:77-79` (state `left_hand`; block `:72-79`) |
| 4 | labels from the source recording, never the generator | **correct, load-bearing, kept verbatim** | §4 |

Two further facts item 2 never mentioned and the replacement must, because a restyle that drops
either produces a corpus the consumer cannot train on:

| | measured | read from |
|---|---|---|
| camera | **exactly one**, `observation.images.ego_view`, `[480, 640, 3]`, AV1 / yuv420p / 30 fps | `meta/info.json:116-136`; short key mapping at `meta/modality.json:107-111`; `preprocess_video: [ego_view]` is the entire model input list, `exported_leapp.yaml:349` |
| the nine extra `action.*` columns | `effort_{left,right}_leg[6]`, `effort_waist[3]`, `effort_{left,right}_arm[7]`, `effort_{left,right}_hand[7]`, `navigate_command[3]`, `base_height_command[1]` — separate columns, not part of the flat `[43]` | `meta/info.json:35-115`; declared at `meta/modality.json:61-105` |

### 2.2 The 43-dim layout, exactly

From the corpus's own `meta/modality.json` — **identical for `state` and for `action`**, which is
itself a fact worth writing down, since the model's *decoder* block is a permutation of its state
block and the two must not be confused (§5.2):

```
[ 0: 6]  left_leg     6      modality.json:3-6    (state)   :33-36  (action)
[ 6:12]  right_leg    6                 :7-10               :37-40
[12:15]  waist        3                 :11-14              :41-44
[15:22]  left_arm     7                 :15-18              :45-48
[22:29]  right_arm    7                 :19-22              :49-52
[29:36]  left_hand    7                 :23-26              :53-56
[36:43]  right_hand   7                 :27-30              :57-60
                     ---
                      43
```

**Both hands are 7 joints: three thumb joints and two 2-joint fingers.** Element names, read from
`exported_leapp.yaml` on 2026-08-22 — this is NVIDIA's ONNX export of `GR00T-N1.7-ApplePnP-V1`,
their own fine-tune *of this corpus*, so it is corpus-A evidence and not a transplant from a
neighbouring dataset:

| block | element order | line |
|---|---|---|
| **state** `left_hand` | `thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1` | `:77-79` |
| **state** `right_hand` | `thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1` | `:85-87` |
| **decoder** `left_hand` | `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2` | `:259-261` |
| **decoder** `right_hand` | `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2` | `:267-269` |

So the **dataset/state** order is asymmetric across hands, and the **model's decoder** order is a
permutation of it that is index-first for both. `LEFT_HAND_PERM = [4,5,6,2,3,0,1]`,
`RIGHT_HAND_PERM = [4,5,6,0,1,2,3]` map decoder output back to dataset order
(`/home/humanoid/develop/vla-training/eval/onnx_leapp_server.py:104-111`). **A producer writes the
dataset order and never the decoder's permutation.**

### 2.3 Independent re-derivation of the block boundaries

Done here rather than cited, on `data/chunk-000/episode_000000.parquet` (590 rows) in the same
snapshot, 2026-08-22, `.venv/bin/python` + `pyarrow`:

| check | result | reading |
|---|---|---|
| `rms(action[15:22] − state[15:22])` | **0.0287 rad** | the command tracks the measured joint |
| `rms(action[15:22] − state[22:29])` | **0.6292 rad** | 22× worse against the *other* arm |
| `rms(action[22:29] − state[22:29])` | **0.0182 rad** | same, for the right arm |

The 0.0287 figure reproduces, on one episode, the **0.0288 rad against a 0.7634 cross-check**
that `eu-hub/make_dataset_state31.py:33-45` recorded over all 171 625 frames. The arm block
boundaries are therefore confirmed by two independent routes. **This is n = 1 episode**, as
`T40_RULE_V2` §4.3 already warned; it is corroboration of the corpus's own `modality.json`, not a
replacement for it.

---

## 3. The amendment

**`T40_RULE_V7` amends `T40_RULE_V1` §8 item 2's three descriptive fields — and only those — to
describe `nvidia/GR00T-N1.7-AppleToPlate`, the corpus `T40_RULE_V1` §3 chose to restyle.**

Registered replacement text:

> 2. **The consumer contract with `emai/vla-training`**, written down: **LeRobot v2.1**; state and
>    action both `float32[43]` in the **same seven groups** —
>    `left_leg[0:6]`, `right_leg[6:12]`, `waist[12:15]`, `left_arm[15:22]`, `right_arm[22:29]`,
>    `left_hand[29:36]`, `right_hand[36:43]` — plus the **nine separate `action.*` columns** and
>    **exactly one camera**, `observation.images.ego_view` at `[480, 640, 3]`; each hand seven
>    joints as **three thumb joints then two 2-joint fingers**, asymmetric across hands in the
>    dataset/state order — **left `thumb×3, middle_0, middle_1, index_0, index_1`; right
>    `thumb×3, index_0, index_1, middle_0, middle_1`** — attested by NVIDIA's export of their own
>    fine-tune of this corpus and **NOT independently verified against the corpus's parquet** (§5.1);
>    **and the action labels come from the *source* recording, never from the generator.**

**What moved:** `v3.0` → `v2.1`; `28-dim arms+hands` → `43-dim in seven groups`, with the groups
named and their slices given; `right hand index-before-middle` → the full per-joint order of
**both** hands, marked with the verification status it actually has. The camera and the nine
`action.*` columns are added because a contract that omits them is not specific enough to build
against.

**What did not move:** the fourth clause, verbatim. **And nothing else in `T40_RULE_V1`.**

The three fields are copied verbatim into
`.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md:46-48` and `:189-190`. That is
**task text, not a registered rule**, and may be corrected in place; doing so is not an edit to
V1 and is not done by this document.

`docs/contracts/vla-training-consumer.md` — the artifact item 2 points at — is amended in the same
change, with its §0 rewritten to record the resolution rather than the disagreement, and its old
§0 preserved verbatim in a superseded-text block. **§8 item 2 is CLOSED.**

---

## 4. The fourth clause is kept verbatim, and it is the load-bearing one

> **"and the action labels come from the *source* recording, never from the generator."**

Unchanged, not paraphrased, not "clarified". It is the only clause in item 2 that is an argument
rather than a description, and it is **the entire argument that lets generated frames into a
training corpus at all**. `T40_RULE_V1` §2:

> *"The labels do not come from the generator. In PR-06 the generated pixels were the thing being
> scored against truth. Here the actions are the recorded teleop trajectory, carried over
> unchanged, and the generated pixels are only an input perturbation. A generator error in PR-06
> was a wrong answer; a generator error here is a corrupted training input whose label is still
> correct — **unless it moves geometry**, which is what §6's G0b exists to catch."*
> — `PR-08-photoreal-augmentation.md:44-48`

Three consequences, restated because they are what the clause actually buys:

1. **Nothing inferred, dreamed, or produced by inverse dynamics may enter a corpus delivered under
   this contract.** This is the standing decision `docs/handoff.md` §3 records and
   `docs/action-labels.md` indexes; PR-08's route keeps labels *because it restyles a real episode*,
   not because generation acquired them.
2. **Frame count and frame order are preserved.** Restyled frame *i* of episode *e* pairs with
   source label *i* of episode *e*. G0a is the instrument that catches a violation, and a G0a
   deviation is **VOID** — not a finding about the corpus but proof the pipeline corrupted or
   reordered labels.
3. **The targets are absolute joint positions in radians**, not corrections
   (`exported_leapp.yaml:240-274`, `kind: target/joint/position`), so a geometry shift between
   pixels and label is attenuated nowhere downstream. It lands on the commanded pose. That is why
   G0b exists and why V5 and V6 spent two documents on the estimator behind it.

The clause is also the one item 2 got right **about both corpora at once**, which is precisely why
it survives a change of corpus untouched: it is a statement about the *pipeline*, not about a
dataset's shape. Everything that was wrong in item 2 was a shape.

---

## 5. What remains unverified, and what would settle it

### 5.1 The right hand's index/middle order, **for this corpus's parquet: UNVERIFIED**

`T40_RULE_V1` §8 item 2 asserted *"right hand index-before-middle"* of `G1_Dex3_*`. It is **not**
carried across on the strength of both being Unitree Dex3 hands. What it rests on instead:

- **What supports it.** `exported_leapp.yaml:80-87` names the right hand's seven state elements
  `thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1`. This is corpus-A-specific
  evidence — NVIDIA's export of NVIDIA's fine-tune of *this* corpus — and is materially stronger
  than an argument from family resemblance.
- **What it is not.** It is an **assertion in an export manifest**, not a measurement against the
  data. The corpus's own `meta/modality.json` gives group boundaries and **no element names at
  all** — there is nothing in the corpus's own metadata to check the export against.
- **Why it could not be measured here.** In `episode_000000` — **the only one of 402 episodes on
  this workstation** — the right hand never moves. Measured 2026-08-22: all seven
  `observation.state[36:43]` columns have per-column std ≤ 3 × 10⁻⁴ (four of them exactly 0.0), and
  `action[36:43]` is identically zero to float precision. A column that does not vary cannot be
  identified. This reproduces `T40_RULE_V2` §4.2's finding that the episode is a one-armed,
  left-side demonstration.
- **Why the usual trick does not work either.** One might hope to identify hand columns by matching
  `action` against `observation.state`, as §2.3 does for the arms. It fails on the hands: measured
  2026-08-22, `rms(action[29:36] − state[29:36]) = 0.498` on the left hand, against 0.0287 for the
  left arm, and the two blocks do not even share a range (state left-hand values run to −0.88 rad
  while the action column is a near-binary code in [−1, +0.7]). This is consistent with
  `configs/groot/new_embodiment_config_defaults.py`'s recorded reason for making the hand channels
  `ABSOLUTE` — *"the G1 hand is driven by near-binary open/close codes rather than a continuous
  trajectory"*. **The hand action block and the hand state block are not the same quantity, so a
  hand-column permutation is invisible to a state↔action comparison.**

**What would settle it.** Correlate against the parquet on episodes in which the right hand
actually moves — the method that settled corpus B's block order, and the method
`docs/contracts/vla-training-consumer.md` §3.2 already prescribes (*"correlate against the parquet,
do not read the card"*). Concretely: fetch enough of the 402 episodes to find right-hand motion,
then discriminate `index_*` from `middle_*` by cross-referencing the two 2-joint pairs against
either (a) a Dex3-1 URDF's joint limits, or (b) the `action.effort_right_hand[7]` column, whose
per-element names come from the same ordering and whose torque signature differs between fingers
under a grasp. **Neither is available on this box today**, and neither is requested by this
document.

Marked **`[?]` UNVERIFIED**, not `[OK]`, in both artifacts. The reason to be strict about a clause
that is probably true: a silently transposed finger pair produces a policy that closes the wrong
digit, and it produces it without raising anything.

### 5.2 Also unverified, carried forward rather than quietly dropped

- **The physical left/right LABELS.** `T40_RULE_V2` §4.3 proved the *side pairing* — that `[15:22]`
  and `[29:36]` are the same side as each other and `[22:29]`/`[36:43]` the other — from the motion
  signature. Which of those is physically the robot's left rests entirely on NVIDIA's column names.
  §2.3's re-derivation here confirms the pairing again and tests the labels no better.
- **n = 1 episode.** Every parquet-derived statement in §2.3 and §5.1 is from `episode_000000`.
  The corpus-wide claims rest on `meta/info.json` and `meta/modality.json`, which are corpus-wide
  by construction.
- **`_data/apple_pnp/CONTRACT.md` still does not exist on this machine.** It is cited throughout
  `vla-training` as the authority on the Dex3 asymmetry. If it surfaces and disagrees about the
  hand element order, it wins and both artifacts must be revised.

### 5.3 The consumer's side — what this repository can and cannot settle

Asked directly: *what does `emai/vla-training` actually consume today?* What could be established
from this box on 2026-08-22:

**Settled.**

- **Nothing in this repository's committed configuration records a consumer contract.**
  `configs/transfer25/styles.toml`'s `[consumer]` table (`:534-588`) is about the *style
  partition's* rendering — `rendering`, `rendering_keys`, `rendering_style_keys`,
  `rendering_volume_key`, `rendering_seed_schedule_key` — i.e. which file
  `97_transfer25_restyle.sbatch` reads and which keys it indexes. `pr08_style_partition.json`
  carries **no `consumer` key at all** (verified: `json.load(...).get("consumer") is None`). The
  name of that table is the only thing about it that suggests otherwise.
- **What the producer builds is a v2.1, 43-dim LeRobot root.**
  `scripts/assemble_restyled_lerobot.py` pins `CODEBASE_VERSION = "v2.1"` (`:58`), refuses a v3.0
  source (`:173`), copies the source's `meta/modality.json` through verbatim (`:183`, `:568`),
  rewrites only `episode_index` and `index`, carries all nine `action.*` columns untouched, and
  prints `configs/groot/new_embodiment_config_defaults.py` as the `--modality-config-path` to use
  (`:601`). That file declares the seven-group 43-dim state and the twelve-key action block.
- **The consumer's own reference copy of the corpus meta is byte-identical to the corpus's**
  (§2). So the AppleToPlate route on the consumer side is reading the same specification.

**Not settled, and this is the one thing the amendment cannot close from this side.**

- **The consumer repository contains *both* routes, and the repo cannot say which one it would
  point at a delivered restyled corpus.** Read 2026-08-22 at
  `/home/humanoid/develop/vla-training` (HEAD `5be48ff`, dated 2026-08-22):
  `scripts/40_groot_prepare_dataset.sh` runs the official **v3.0 → v2** conversion and then drops
  the **28-dim** `groot/modality_g1_dex3.json` into `meta/`, and `scripts/41_groot_finetune.sh:25`
  passes the 28-dim `groot/g1_dex3_modality_config.py`. That is corpus B's route, and its fields
  are **exactly the three V1 §8 item 2 named** — which is very likely where they came from. The
  cluster route `eu-hub/train_apple_nvidia.sbatch:99-104` instead passes
  `$ROOT/dataset/new_embodiment_config_defaults.py`, the 43-dim AppleToPlate config. Both are live
  in the same checkout.
- **A read-only checkout is not a commitment.** Nothing here obliges the consumer to keep either
  route, and the contract's own header already says the producer-machine paths are *"read-only
  evidence, not deliverables."*
- **Even the identity of the repository is by convention.** The checkout at
  `/home/humanoid/develop/vla-training` has origin
  `https://github.com/RaaSaaR-org/vla-training.git`. This repo calls the consumer
  `emai/vla-training` throughout; nothing on this box maps the one name to the other.

**What would settle it:** a statement from the consumer side — an issue, a commit, or a written
reply — naming which modality config a delivered restyled AppleToPlate root will be trained under,
and confirming that the 28-dim `40_*`/`41_*` path is not it. Until that exists, the amendment
describes **what the producer delivers**, measured on the corpus and on the assembler, and does not
claim to describe what the consumer will do with it. That asymmetry is deliberate and is recorded
in the contract's §0 as well.

---

## 6. Why an item-2 that named the wrong dataset survived from 2026-07 to 2026-08-22

Recorded because the *reason* is more useful than the correction, and because the same shape will
recur.

**Nothing in the pipeline reads §8 item 2. So nothing failed.**

Searched 2026-08-22 across this repository: every reference to the three fields is prose — the
rule itself, `docs/contracts/vla-training-consumer.md` §0/§7, `T40_RULE_V2` §4,
`T40_RULE_V3` §5.3, `T40_RULE_V4` §6, and the two copies in T-040's task text. **No script, sbatch,
config or test consumes them.** The generator reads `configs/transfer25/pr08_style_partition.json`;
the assembler reads the source root's own `meta/`; the trainer reads a modality config. A wrong
sentence in a preregistration is inert until somebody builds from it.

That is exactly why it was dangerous rather than harmless. The failure mode was not a red test; it
was a future session reading item 2, believing it, and writing a 28-dim converter for a 43-dim
corpus — which is the same transposition `T40_RULE_V2` §4 and `docs/contracts/vla-training-consumer.md`
§3.1 both had to unwind for the *other* corpus, at the cost of five documents. The correction here
is cheap because it landed before that build existed.

Two practices this argues for, neither of them registered as a rule by this document:

- **A contract nobody parses is checked by nobody.** The one guard added alongside this amendment
  is `tests/test_g1_dex3_28.py::test_contract_hand_order_citation_still_points_at_the_hand_table`,
  which pins the line-number citation that `src/wam/robot/g1_dex3_28.py` makes into the contract.
  It is small on purpose; the point is that the count of automated readers of that document went
  from zero to one.
- **Fields copied between documents should carry their source.** Item 2's fields appear in four
  places and cited nothing in any of them. Every field in the replacement text carries the file it
  was read from.

---

## 7. What V7 licenses

**Nothing.**

- It closes `T40_RULE_V1` §8 **item 2**, and no other item. **Items 3 and 4 remain open.**
- `T40_RULE_V1` §1's prohibition binds in full: **no clip is generated, no weight is trained on
  generated frames, and no number from PR-08 is quoted as a result.**
- It licenses **no training run**, on generated data or real. That remains the project owner's
  separate call (`CLAUDE.md`).
- It says **nothing about GR00T's capability** — `PR-07` §6's prohibition stands — and nothing
  about `docs/benchmark.md`'s L4 gate.
- It flips no `GATE_QUALIFIED`, submits no job, and edits no blocker tuple.

**A correction to a description is not permission to act on it.**

---

## 8. Determination and provenance

### 8.1 Determination

> **`T40_RULE_V7` determines that `T40_RULE_V1` §8 item 2's consumer contract is fixed on
> `nvidia/GR00T-N1.7-AppleToPlate` — the corpus `T40_RULE_V1` §3 chose to restyle — and that its
> three descriptive fields are replaced by §3's registered text, each field cited to the metadata
> it was measured from. The fourth clause is unchanged, verbatim. `T40_RULE_V1` §8 item 2 is
> CLOSED.**
>
> **This determination opens nothing.** §8 items 3 and 4 are open; §1's prohibition binds in full;
> no training run of any kind is licensed.

**Decision.** The choice between the two candidate corpora was made by the project owner and
relayed to the session that drafted this document on 2026-08-22, in the terms *"amend PR-08 §8 item
2's consumer contract to describe the corpus actually being restyled, rather than converting the
corpus or waiting on the consumer."* That is the decision `T40_RULE_V2` §4.4, `T40_RULE_V3` §5.3
and `T40_RULE_V4` §6 each declined to make on the owner's behalf.

```
Project owner: huhn.dev@gmail.com               Date: 2026-08-22

Determination:   [x] decided as above — item 2 is fixed on nvidia/GR00T-N1.7-AppleToPlate
                 [ ] decided otherwise — see notes

Notes:

  The deciding act was the owner's; the keystrokes were the session's, as in
  T40_RULE_V4 §7. The owner directed the amendment and its scope; the session
  performed the verification in §2 and §5 and is answerable for it. Nothing
  about the scope in §7 was varied: no generation, no training run, no
  statement about GR00T, no determination about docs/benchmark.md's L4 gate.
```

### 8.2 Provenance

| | |
|---|---|
| rule | `T40_RULE_V7` |
| registered | 2026-08-22 |
| supersedes | nothing. It **supplements** `T40_RULE_V1`, which stands and is unedited, as do V2–V6 |
| changes | `T40_RULE_V1` §8 item 2's three descriptive fields (§3), and nothing else |
| closes | `T40_RULE_V1` §8 **item 2** |
| leaves open | `T40_RULE_V1` §8 **items 3 and 4** |
| decided by | the project owner, 2026-08-22 (§8.1) |
| artifacts amended | `docs/contracts/vla-training-consumer.md` — §0 rewritten with the superseded text preserved verbatim as §0.1; §1 gains the corpus-B-route caution; §2 gains the corpus's own metadata, the re-derivation and the `[?]` hand-order block; §3.2, §5 and §8 updated; §7 records D1–D4 as resolved and gains §7.1 on what the consumer side could not settle |
| tests | `tests/test_g1_dex3_28.py` — citation updated to the amended document, plus one new guard pinning it |
| partition | `configs/transfer25/styles.toml` / `pr08_style_partition.json`, `T40_STYLES_V1` — **unchanged**, no style, id, slug or prompt touched, therefore no hash moved. V7 registers no hash value |
| measurements | §2.1–§2.3 and §5.1, taken 2026-08-22 with `.venv/bin/python` against the HF snapshot's `meta/info.json` and `meta/modality.json` (hashes in §2), `data/chunk-000/episode_000000.parquet` (590 rows), and `/home/humanoid/models/GR00T-N1.7-ApplePnP-V1/exported_leapp.yaml`. Parquet claims are **n = 1 episode** |
| unverified | the right hand's index/middle order against this corpus's own data (§5.1); the physical left/right labels (§5.2); which modality config the consumer would use for a delivered restyled root (§5.3) |
| generation licensed | **no** |
| training licensed | **no** |
