# PR-08 V2 — arm C frame-matching, and the corpus named in §8 item 2

**Rule `T40_RULE_V2`. Registered 2026-08-15, before any clip is generated, before any weight is
trained, and before any job is submitted.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), which is registered as
`T40_RULE_V1` and **has not been edited and must not be**. The repo's discipline is
`docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."* V2 is that versioning, not a revision.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

**Nothing in this document licenses generation.** `T40_RULE_V1` §1 forbids generating a corpus,
training any weight on generated frames, and quoting any number from PR-08 as a result, *until
every §8 item is closed and T-39 has reported*. Both conditions still hold. §3 below leaves one
§8 item explicitly **open**.

---

## 0. What V2 does not change

Stated first, because a V2 that quietly moves a threshold is the failure the versioning discipline
exists to prevent.

**Every gate, threshold and verdict in `T40_RULE_V1` stands unchanged.** Specifically and
exhaustively:

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined here. V2 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` | still **derived** — median per-step object-centroid displacement in the source clips, computed and committed before generation |
| `EST_DRIFT_P95` | still **measured** per §4, still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still not a pass |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — generator held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start |
| **G0c** embodiment | unchanged — real robot pixels unconditionally composited back; robot-mask IoU recorded as diagnostic, never as a gate |
| **Ladder** | unchanged — L1 `skill_vs_repeat_pct > 0`, L2 `ci_skill_vs_repeat_pct > 0` (`ci_` = task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (§6) | unchanged in every cell, including that P requires *both* B − A ≥ floor *and* B − C ≥ floor, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **§1's prohibition** | unchanged and still binding — nothing is generated until every §8 item is closed **and** T-39 has reported |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a P is a claim about held-out *generated* appearance, and licenses exactly one thing: recording a small real shifted eval set and re-running A and B against it |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json`, rule `T40_STYLES_V1`, committed 2026-08-15, `source_content_sha256 = 4da3875d0c76e9b23821c1ca9fe20f965f9fc0867edcd085fb792b587a680da8`. V2 changes **no style, no id, no slug and no prompt string**, and therefore does not change that hash |

V2 changes exactly one thing: **how many clips arm C contains**, and the arithmetic that follows
from it. It also opens one item that V1 recorded as closable. Nothing else.

---

## 1. The ambiguity in §5, and the reading adopted

### 1.1 What §5 says

`T40_RULE_V1` §5, arm C row:

> **C `real+identity`** | A + restyles whose style prompt *is the source's own appearance* |
> **generator-fingerprint control.** Same generator, same pipeline, **same frame count**, no added
> diversity. Separates "visual diversity helped" from "passing frames through a diffusion model
> helped"

**"Same frame count" has no stated referent.** It says same *as what*, implicitly, and the two
available answers give arm C sizes that differ by 10×.

### 1.2 The two readings

The committed partition (`configs/transfer25/pr08_style_partition.json`, verified 2026-08-15) is
**10 `train` styles, 5 `eval` styles, 1 `identity` style**, over the **402** real AppleToPlate
episodes (~171 625 frames, ~427 mean frames per episode, range 249–749 — `T40_RULE_V1` §8 item 3).

| reading | what "same frame count" is matched to | arm C size |
|---|---|---|
| **R1 — source coverage** | the same *source* frames pass through the generator in C as in B: every one of the 402 episodes is restyled once under `identity-source`, so the underlying real footage is fully and identically covered | 1 × 402 = **402 clips** ≈ 171 625 frames |
| **R2 — added training volume** | the *generated frames added to arm A* are equal in count between B and C | 10 × 402 = **4 020 clips** ≈ 1 716 250 frames |

R1 is what the committed partition shipped: one identity style against ten train styles. Under R1,
arm C is **402 clips against arm B's 4 020** — a 10× volume difference.

### 1.3 The reading adopted: **R2**

**`T40_RULE_V2` adopts R2. Arm C is frame-matched to arm B on added training volume.**

The reason is that R1 destroys the only thing arm C exists to do. `T40_RULE_V1` §6's headline reads
**B − C** to decide whether a gain from B is *attributable to diversity* rather than to *passing
frames through a diffusion model*. Under R1, arm B has ten times the added data of arm C, so a
`B − C ≥ MATERIAL_FLOOR_PP` is equally consistent with:

- visual diversity helped (the intended reading, which licenses verdict **P**), and
- **more data helped**, which R1 leaves entirely unmeasured.

That is the exact confound arm C was added to remove — V1 §5 says so in its own words: *"without C
a gain from B is uninterpretable."* Under R1, with C at a tenth of B's volume, a gain from B is
still uninterpretable, and verdict **P** would be unsound. R1 also collides with `T40_RULE_V1` §9's
first bullet — *"It is not T-32. Ten restyles of 402 demos is still 402 trajectories. It cannot
speak to data volume"* — by making a data-volume difference load-bearing inside the headline
comparison of a document that disclaims data-volume findings.

R2 makes B − C a comparison in which volume is held fixed and only the style diversity differs,
which is the comparison §5 describes in prose.

---

## 2. The resolution, and how arm C is built

**Decided 2026-08-15. This was a human decision, not an agent's.** It is recorded here because it
resolves an ambiguity in a registered rule, and an ambiguity resolved after the fact by whoever is
running the job is not a pre-registration.

**Resolution:** the single `identity-source` style is generated **10 times per episode, with 10
different generator seeds**.

```
arm C  =  1 identity style  ×  10 seeds  ×  402 episodes  =  4 020 clips
arm B  =  10 TRAIN_STYLES   ×   1 seed   ×  402 episodes  =  4 020 clips
```

**Arm B is NOT subsampled.** The frame-match is achieved by *adding to C*, never by *removing from
B*. Subsampling B to 402 was available and was rejected: B is the intervention and the headline is
B − A, so shrinking B by 10× would weaken the thing being measured and would change the B − A
comparison in order to fix the B − C comparison. Two arms are matched to each other by raising the
control, not by lowering the treatment.

**The style partition file is unchanged.** The identity style's `id`, four axis slugs and prompt
string are exactly as committed under `T40_STYLES_V1`; the ten seeds are a generation parameter,
not a new style. The partition hash in §0 therefore still holds and is still the hash that
`T40_RULE_V1` §6 requires to be recorded with the verdict. **The ten seed values must themselves be
recorded with the verdict**, alongside the partition hash, for the same reason the hash is: a run
whose seeds are not written down cannot be shown to have been the run that was pre-registered.

### 2.1 What the ten seeds buy, and what they do not — stated so it is not read as more than it is

The ten identity clips per episode are **not ten copies of one clip**. They differ by generator
sampling noise: same prompt, same conditioning, different seed. So arm C's added variation is
**generator stochasticity, and nothing else** — it contains no lighting change, no material change,
no background change.

That is the correct content for a fingerprint control, and it is also the honest limit of the
comparison: B − C now separates *domain diversity* from *(diffusion pass + generator sampling
noise)*, not from *diffusion pass alone*. If seed-only variation turns out to act as an
augmentation in its own right, that effect sits on C's side of the subtraction and makes B − C
**conservative** — it can shrink a real diversity gain toward **I** or **F**, but it cannot
manufacture a **P**. The direction of that bias is stated here in advance so it cannot be
discovered afterwards and read the convenient way.

---

## 3. The consequence that must be carried through: volume, and where the ceiling is derived

This is not a corollary to be re-derived later. It is part of the rule.

### 3.1 Volume

| | under R1 (as shipped) | **under `T40_RULE_V2`** |
|---|---|---|
| arm B — `TRAIN_STYLES` | 10 × 402 = 4 020 clips | 10 × 402 = **4 020 clips** *(unchanged)* |
| arm C — identity | 1 × 402 = 402 clips | 10 × 402 = **4 020 clips** |
| **generated training clips (B + C)** | 4 422 | **~8 040 — roughly double** |
| `EVAL_STYLES` (the headline eval domain) | 5 × 402 = 2 010 clips | 5 × 402 = **2 010 clips** *(unchanged)* |
| **whole partition (train + eval + identity)** | 6 432 clips | **10 050 clips ≈ 4.29 M frames** |

The decided figure is the one that must be planned against: **total generated volume roughly
doubles, to ~8 040 clips.** That is the B + C training total, up from 4 422. The `EVAL_STYLES` set
is a further 2 010 clips on top of it, which is exactly why §3.2 is written the way it is.

At `T40_RULE_V1` §8 item 3's multiplier (~171 625 frames per style-instance over the 402 episodes),
the whole partition is **20 training style-instances + 5 eval style-instances = 25 × ~171 625 ≈
4.29 M generated frames**, against V1's implied 15 × ~171 625 ≈ 2.57 M.

### 3.2 The GPU-h ceiling is derived over the WHOLE partition, not per style-set invocation

**`T40_RULE_V1` §8 item 3 requires a measured throughput number and a GPU-h ceiling derived from
it, "enforced in the sbatch as `MAX_RESTARTS` enforces T-39's". Under `T40_RULE_V2` that ceiling is
derived over the whole partition — train + eval + identity, 10 050 clips — and not per style-set
invocation.**

This has a concrete target. `cluster/discoverer/97_transfer25_restyle.sbatch` takes `STYLE_SET` as
one of `train | eval | identity` and `CEILING_GPU_H` as a required per-invocation variable with no
default (`:115-116`). That shape is correct and is not being changed — a required value with no
default is the right design, and the *reason* given in the script (*"a default would be a budget
line nobody measured"*) is the reason this section exists. What V2 fixes is the **quantity passed
in**: a `CEILING_GPU_H` sized for one style-set is not a budget, because the identity set is no
longer a tenth of the train set and the three sets are no longer comparable in size. Under V1's
shipped partition an operator could plausibly have derived one per-set number and reused it across
all three invocations; under V2 that number would under-count the identity set by 10×.

Therefore, before generation:

1. Take the timed measurement `T40_RULE_V1` §8 item 3 requires — one episode on an H200 at
   640×480. **No budget line exists until that measurement does.** That is unchanged.
2. Derive the ceiling from it over **10 050 clips**, i.e. over all 25 style-instances, and record
   the derivation.
3. Split that whole-partition ceiling across the `STYLE_SET` invocations and chunks, so the sum of
   the `CEILING_GPU_H` values actually passed to `97_transfer25_restyle.sbatch` is bounded by the
   whole-partition figure. The per-invocation enforcement stays; what is enforced is a share of a
   whole-partition budget rather than an independent per-set allowance.

The cluster constraints in §8 item 3 are unchanged and still bind: 4 h `MaxWall`, `MaxJobsPU=4`,
billing `GPUs×1.0 + MemGB×0.25 + Threads×0.036` per minute so `--mem` is not free, chunked and
resumable in the `submit_chain.sh` shape, and the login node off limits for anything that computes.
The doubling makes the chunking more load-bearing, not less.

---

## 4. Second open item — PR-08 §8 item 2 names the wrong corpus, and is **NOT closed**

Recorded in the same document because it is a second defect in the same registered rule, found the
same day, and because leaving it in a working note rather than in a versioned rule would let it be
read as closed.

### 4.1 The finding

`T40_RULE_V1` §8 item 2 requires, before a single clip is generated:

> **The consumer contract with `emai/vla-training`**, written down: LeRobot v3.0, 28-dim
> arms+hands, right hand index-before-middle, and the action labels come from the *source*
> recording, never from the generator.

Those three descriptive fields — **LeRobot v3.0**, **28-dim arms+hands**, **right hand
index-before-middle** — are a real and internally coherent contract. **They describe
`unitreerobotics/G1_Dex3_*`.** They do not describe `nvidia/GR00T-N1.7-AppleToPlate`, which is the
corpus `T40_RULE_V1` §3 chose to restyle and the only corpus this pre-registration operates on.

AppleToPlate is **LeRobot v2.1** and **43-dim in seven groups**, not 28-dim in two. Two of the
three fields are wrong for the corpus the document is about. (The third, *right hand
index-before-middle*, is correct for AppleToPlate's *state* block and under-specified — the *left*
hand is middle-before-index. See `docs/contracts/vla-training-consumer.md` §7 D4.)

The fourth clause of §8 item 2 — *"the action labels come from the source recording, never from the
generator"* — is **correct, is the load-bearing clause, and is untouched by this finding.**

### 4.2 The 43-dim layout, measured 2026-08-15

Verified independently for this document on the one locally available episode,
`/home/humanoid/models/apple_pnp_golden/dataset/data/chunk-000/episode_000000.parquet`, **590
rows**, `observation.state` and `action` both `[590, 43]`.

```
left_leg[0:6]  right_leg[6:12]  waist[12:15]  left_arm[15:22]
right_arm[22:29]  left_hand[29:36]  right_hand[36:43]
```

Confirmed three independent ways:

**(1) NVIDIA's own group widths sum to exactly 43, in that order.** Measured from the parquet's
`action.effort_*` columns in declaration order:

```
action.effort_left_leg    width=6
action.effort_right_leg   width=6
action.effort_waist       width=3
action.effort_left_arm    width=7
action.effort_right_arm   width=7
action.effort_left_hand   width=7
action.effort_right_hand  width=7
sum = 43
```

**(2) The repo's own converter already hard-codes it.** `scripts/convert_lerobot_g1.py:99-106`:

```python
SOURCE_STATE_DIM = 43
WAIST_YAW = 12
LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(22, 29)
LEFT_HAND = slice(29, 36)
RIGHT_HAND = slice(36, 43)
```

**(3) The measured motion signature is coherent with the layout.** Per-group maximum per-joint
range over the episode, from `observation.state`:

| group | max per-joint range | reading |
|---|---|---|
| `left_leg[0:6]` | 0.009255 rad | static |
| `right_leg[6:12]` | 0.003120 rad | static |
| `waist[12:15]` | 0.011468 rad | static |
| `left_arm[15:22]` | **1.226669 rad** | **moves** |
| `right_arm[22:29]` | 0.019271 rad | static |
| `left_hand[29:36]` | **0.826484 rad** | **moves** |
| `right_hand[36:43]` | 0.000740 rad | static |

The corresponding `action` block for `right_hand[36:43]` is **exactly 0.0** — `np.unique` over all
590 × 7 values returns the single value `[0.]`. A gantry-mounted static G1 performing a one-armed
pick-and-place is exactly this signature: legs and waist pinned, one arm and the hand on the *same
side* moving together, the other arm and hand untouched.

Two corrections to the figures as they were handed over, made here rather than repeated
uncorrected:

- **`waist` is 0.011468 rad, not "< 0.011 rad."** It is marginally above that bound. It is still
  two orders of magnitude below the moving groups and the qualitative reading is unaffected, but
  the stated bound is wrong and is not repeated.
- **"right hand EXACTLY 0.000" is true of the `action` column, not of `observation.state`.** In
  state the right hand's largest per-joint range is 7.4 × 10⁻⁴ rad — static to measurement noise,
  but not identically zero.

### 4.3 Two caveats on that evidence, both load-bearing

- **The arm ↔ hand SIDE PAIRING is proven by the coherence above. The "left" / "right" LABELS are
  not.** What the motion signature demonstrates is that `[15:22]` and `[29:36]` are the same side as
  each other, and that `[22:29]` and `[36:43]` are the other side. Which of those two sides is
  physically the robot's left rests entirely on **NVIDIA's column ordering** — the names in
  `action.effort_*` and in the converter — and no measurement here tests it. This is the same class
  of error T-041 found on `USC-PSI-Lab/Humanoid-Everyday-G1` and T-043 records as *"a source that is
  wrong about the block order earns no trust about the finger order"*; correlate, do not read the
  card.
- **n = 1 episode.** One episode of 402 is on this workstation. A layout that holds in
  `episode_000000` is strong evidence and is not a corpus-wide proof, and a single episode cannot
  distinguish a layout constant from a coincidence of one demonstration in which only the left side
  happened to move.

Separately: the **v2.1** version claim is **not** verifiable from the one local file — no
`meta/info.json` is present on this workstation, only `data/chunk-000/`. It is carried from
`docs/contracts/vla-training-consumer.md` §7 D1, which names `eval/real_reference/info.json:2` on
the producer machine, and from `scripts/convert_lerobot_g1.py:2` (*"Convert LeRobot v2.1 Unitree G1
episodes (GR00T-N1.7 layout)"*). It is cited, not re-measured here.

### 4.4 Status: OPEN. §1's prohibition still binds.

**`T40_RULE_V2` does not resolve §8 item 2 and does not close it.** Deliberately: choosing which
corpus the consumer contract is *supposed* to name — and therefore what the deliverable actually is
— is a decision about the experiment, not a clerical correction, and it is not an agent's to make.
The two candidate resolutions are visibly different experiments (a 43-dim v2.1 one-camera
AppleToPlate deliverable versus a 28-dim v3.0 `G1_Dex3_*` one), and `docs/contracts/vla-training-consumer.md`
§7 records four disagreements (D1–D4) plus two more against the data-factory README (D5–D6) that a
resolution has to dispose of.

Therefore, and explicitly:

- **§8 item 2 is NOT closed.**
- **`T40_RULE_V1` §1's prohibition still binds in full** — no corpus is generated, no weight is
  trained on generated frames, and no number from PR-08 is quoted as a result — **until a human
  resolves §8 item 2 and T-39 has reported.**
- The resolution, when made, is a further version alongside these two. It is not an edit to either.

The related text in `.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md:46-48` and
`:189-190` copies the same three fields. That is **task text, not a registered rule**, and may be
corrected in place — which is a separate matter from this item's status.

---

## 5. Provenance

| | |
|---|---|
| rule | `T40_RULE_V2` |
| registered | 2026-08-15 |
| supersedes | nothing. It **supplements** `T40_RULE_V1`, which stands and is unedited |
| changes | arm C size only (§1, §2) and the ceiling derivation that follows from it (§3) |
| opens | `T40_RULE_V1` §8 item 2 (§4) |
| decided by | a human, 2026-08-15 (§2) |
| partition | `configs/transfer25/pr08_style_partition.json`, `T40_STYLES_V1`, sha256 `4da3875d…a680da8` — **unchanged** |
| measurements | §4.2, verified 2026-08-15 against `episode_000000.parquet` (590 rows) and `scripts/convert_lerobot_g1.py:99-106`; n = 1 episode |
| generation licensed | **no** |
