# PR-08 V3 — the seed schedule, the superseded partition hash, the ceiling reading, and the G0b step

**Rule `T40_RULE_V3`. Registered 2026-08-15, before any clip is generated, before any weight is
trained, and before any job is submitted.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md) (`T40_RULE_V1`) and
[`PR-08-V2-arm-c-frame-matching.md`](PR-08-V2-arm-c-frame-matching.md) (`T40_RULE_V2`). **Neither
has been edited and neither may be.** The discipline is `docs/handoff.md` §3 — *"Rules are
versioned, never edited in place. A gate rewritten after seeing its output is not a gate."* V3 is
that versioning. It is written **because** V2 cannot be corrected in place, and one of the four
things below is a correction to V2.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

**Nothing in this document licenses generation.** `T40_RULE_V1` §1 forbids generating a corpus,
training any weight on generated frames, and quoting any number from PR-08 as a result, *until
every §8 item is closed and T-39 has reported*. Both conditions still hold, and §5 below records
which §8 items are open and why a particular T-39 outcome would close PR-08 rather than open it.

**Nothing has been generated. No clip exists.** Every number in this document is a property of the
corpus, the committed configuration or the submit path — none is a result.

---

## 0. What V3 does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent.

**Every gate, threshold and verdict in `T40_RULE_V1` stands unchanged, and so does every resolution
in `T40_RULE_V2` except the one provenance field §2 below corrects.**

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`. V3 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` | still **derived** — the median per-step object-centroid displacement in the source clips, measured and committed before generation. §4 registers what *"per-step"* means; it does not change how the number is obtained, and it does not supply a value |
| `EST_DRIFT_P95` | still **measured** per §4 of V1, still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still not a pass |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start |
| **G0c** embodiment | unchanged — real robot pixels unconditionally composited back; robot-mask IoU recorded as a diagnostic, never as a gate |
| **Ladder** | unchanged — L1 `skill_vs_repeat_pct > 0`, L2 `ci_skill_vs_repeat_pct > 0` (`ci_` = task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (V1 §6) | unchanged in every cell, including that P requires *both* B − A ≥ floor *and* B − C ≥ floor, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **Arms A / B / C / D** | unchanged. B is the intervention, C is the generator-fingerprint control, D is diagnostic and never the headline |
| **Arm C's size** (`T40_RULE_V2` §1, §2) | unchanged — R2, frame-matched: 1 identity style × 10 repeats × 402 episodes = 4 020 clips against arm B's 10 × 1 × 402 = 4 020. **Arm B is still not subsampled** |
| **Clip totals** | unchanged — train 4 020, identity 4 020, eval 2 010, whole partition 10 050 over 25 style-instances. The seed schedule of §1 costs **zero extra clips** |
| **The style pool** | unchanged — 10 `TRAIN_STYLES`, 5 `EVAL_STYLES`, 1 identity. §1 adds no style, drops none, re-slugs none, and moves none between the sets. Which appearances are held out is exactly what it was |
| **§8 item 2's status** | still **OPEN** (`T40_RULE_V2` §4). V3 adds evidence in §5.2 and resolves nothing |
| **§1's prohibition** | unchanged and still binding — nothing is generated until every §8 item is closed **and** T-39 has reported |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a P is a claim about held-out *generated* appearance and licenses exactly one thing: recording a small real shifted eval set and re-running A and B against it |

V3 changes exactly four things, all of them registrations or corrections of record:

1. **the seed schedule becomes part of the pre-registration** (§1);
2. **`T40_RULE_V2`'s recorded partition hash is superseded** (§2);
3. **the GPU-h ceiling reading is fixed** so the experiment is launchable by someone obeying the
   rule (§3);
4. **`"per-step"` in V1 §6's `GEOM_TOL` is defined** against the corpus's measured rate (§4).

---

## 1. The seed schedule is a REGISTERED matter, not an implementation detail

### 1.1 Why this is in a pre-registration at all

The initial-noise seed looks like a submit-time knob. It is not, because arm C's entire job is to
hold everything except prompt diversity fixed between itself and arm B, and **the seed is one of
the things that has to be held fixed.** A schedule that varies the seed differently in the two arms
puts a second difference inside `B − C`, and a schedule that shares seeds between one arm's
*training* clips and the *eval* clips the headline is scored on puts a generator artefact inside
the headline. Both were true of the schedule as first committed today. Neither is visible in any
clip count, in any style, or in any gate — which is exactly why it belongs in the registered rule
and not in a comment.

### 1.2 What the previous schedule actually was, and the two defects

The partition previously carried the seeds as a bare list, `volume.repeat_seeds`, under the
committed rule *"repeat r uses `repeat_seeds[r]`; a style with `repeats = 1` uses
`repeat_seeds[0]`"*. **Every `TRAIN` style and every `EVAL` style has `repeats = 1`** — that is
deliberate and unchanged (arm B's diversity must come from having ten appearances, not from
sampling one of them ten times; and repeating an eval style would weight one appearance five times
over in the headline mean without adding a held-out appearance). Read literally — and the sbatch's
work-list expansion did read it literally — the rule therefore resolved to:

| set | clips | seeds actually used |
|---|---|---|
| **train** (arm B) | 4 020 | **7001 only** |
| **eval** (the headline's scoring set) | 2 010 | **7001 only** |
| **identity** (arm C) | 4 020 | 7001–7010 |

**Defect 1 — `B − C` was a difference of two effects.** Arm B varied its **prompt** at a fixed
seed; arm C varied its **seed** at a fixed prompt. `B − C` was then "ten prompts at one seed" minus
"one prompt at ten seeds". `T40_RULE_V1` §5 asks arm C to differ from arm B in diversity **and in
nothing else**; under that schedule it differed in two things, and the subtraction isolated neither.

**Defect 2, which is the one that breaks the experiment — arm B's training clips shared their
initial-noise seed with the eval clips the headline is scored on, and arm C's mostly did not.** All
4 020 arm-B training clips and all 2 010 eval clips sat on 7001; nine tenths of arm C's did not
touch 7001 at all. If the generator leaves **any** transferable seed-specific fingerprint — and arm
C exists precisely because we do not get to assume it does not — then arm B's training distribution
is matched to the *scoring* distribution more closely than arm C's, for a reason that has nothing
to do with appearance diversity. That **inflates `B − C` on the headline**, i.e. it **can
manufacture a P**: the exact class of confound arm C was added to remove, re-entering through the
seed schedule instead of through the clip counts.

### 1.3 `T40_RULE_V2` §2.1 asserted a one-directional bias, and that assertion was INCOMPLETE

Recorded plainly rather than left to be noticed. `T40_RULE_V2` §2.1 states of arm C's ten seeds:

> If seed-only variation turns out to act as an augmentation in its own right, that effect sits on
> C's side of the subtraction and makes B − C **conservative** — it can shrink a real diversity
> gain toward **I** or **F**, but it **cannot manufacture a P**. The direction of that bias is
> stated here in advance so it cannot be discovered afterwards and read the convenient way.

That paragraph is right about the mechanism it considered — seed variation acting as augmentation
*within arm C's training set* — and **it is incomplete, and its conclusion is therefore wrong as
written.** It reasons about the training arms only and never asks which seeds the **eval** clips
were generated under. Under the schedule that was live when V2 was written, the eval clips shared
their seed with the whole of arm B, and the resulting train↔eval fingerprint match is a second,
opposite-signed channel that V2 did not consider. **`B − C` was therefore not conservative, and the
"cannot manufacture a P" claim did not hold.** V3 records this as a defect in a registered
statement, not as a clarification of it.

The claim becomes true again only under the schedule registered below, where the eval block is
disjoint from both training blocks and no arm is matched to the scoring distribution by seed.

### 1.4 The registered schedule

**`T40_RULE_V3` registers the seed schedule as committed in `configs/transfer25/styles.toml`
`[seed_schedule]`, verified against `scripts/check_style_partition.py::check_seed_schedule` on
2026-08-15.** It is inside the partition content hash of §2, and it is carried into the
generator-facing rendering already resolved per style, so the consumer derives no index rule of its
own.

**The assignment rule** (`[seed_schedule].assignment` / `.rule`, pinned to the code as
`SEED_ASSIGNMENT` / `SEED_RULE` in `scripts/check_style_partition.py:71-75`):

```
assignment = "style-instance-index"
rule       = "seed = blocks[<set>][style_index * repeats + repeat_index];
              style_index and repeat_index are 0-based, styles in committed order"
one_seed_per_style_instance = true
```

**The blocks** (`[seed_schedule.blocks]`):

```
train    = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
identity = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
eval     = [7011, 7012, 7013, 7014, 7015]
```

**The arithmetic, worked through rather than asserted:**

| set | styles × repeats = instances/episode | seed index | clips | seeds spanned |
|---|---|---|---|---|
| train (arm B) | 10 × 1 = 10 | `i·1 + 0 = i`, i ∈ 0..9 | 10 × 402 = 4 020 | 7001–7010 |
| identity (arm C) | 1 × 10 = 10 | `0·10 + r = r`, r ∈ 0..9 | 10 × 402 = 4 020 | 7001–7010 |
| eval (scoring) | 5 × 1 = 5 | `j·1 + 0 = j`, j ∈ 0..4 | 5 × 402 = 2 010 | 7011–7015 |

**The two properties this registers, and they are the whole point:**

- **Arms B and C span the identical seed set, seed by seed.** Not merely the same set: the same
  *count* per seed — 4 020 / 10 = **402 clips per seed in each arm, one per episode**. That falls
  straight out of `train_instances == identity_instances`, i.e. out of the frame match
  `T40_RULE_V2` already registered. **Prompt diversity is therefore the only thing left differing
  between arm B and arm C**, which is what `B − C` is supposed to measure and what V1 §5 asks for
  in prose.
- **The eval block is disjoint from both training blocks.** No eval clip shares an initial latent
  with any training clip in **either** arm, so no arm is matched to the scoring distribution by the
  generator's seed-specific behaviour. The channel that could manufacture a P is closed.

**The repair cost zero extra clips** — 4 020 / 4 020 / 2 010, exactly as `T40_RULE_V2` §3.1 fixed
them — **and it is only available before the first clip exists.** That is why it is registered now
rather than reviewed later.

**Enforcement, so this is not prose.** `check_seed_schedule` fails the partition if the train and
identity blocks stop spanning the same seed set, fails it if the eval block intersects either, fails
a block that is longer or shorter than its set's style-instance count, fails duplicate seeds within
a block, and fails an `assignment`/`rule` string that does not match the one implemented. The
sbatch re-checks the two schedule invariants before it writes a work list, because it is the last
gate before a clip exists, and the seed travels in the work unit, is hashed into `WORK_SHA` (so a
resume under a different schedule is refused) and is recorded in `chunk_metadata.json`.

**Recorded with the verdict**, extending `T40_RULE_V1` §6's list and `T40_RULE_V2` §2's requirement
that the ten seed values be written down: **the seed values AND the assignment**, i.e. the three
blocks above and the rule string. `T40_RULE_V2` §2 required "the ten seed values"; a list of ten
values is satisfied by a schedule that hands arm B one of them and hands the eval set the same one,
which is what happened. **The schedule is the commitment; the values are arbitrary.**

---

## 2. `T40_RULE_V2`'s recorded partition hash is SUPERSEDED

### 2.1 What V2 records, and why both limbs of it are false in the tree

`T40_RULE_V2` §0 (last row) and §5 (provenance table) both record:

> `configs/transfer25/pr08_style_partition.json`, rule `T40_STYLES_V1`, committed 2026-08-15,
> `source_content_sha256 = 4da3875d0c76e9b23821c1ca9fe20f965f9fc0867edcd085fb792b587a680da8`. V2
> changes **no style, no id, no slug and no prompt string**, and therefore does not change that
> hash

Both limbs are false against the committed tree:

- **The justification is false.** One axis **slug** and three **prompt strings** changed on
  2026-08-15 (§2.3). V2's own framing — *"V2 changes no …"* — is a claim about V2's authorship, and
  the partition file agrees that V2 did not make those edits (*"a prompt string change, which V2 §0
  says V2 does not make, and it did not: the lint did"*). That distinction does not rescue the
  sentence, because the sentence is offered as the **reason** the hash is unchanged, and the reason
  is about the file, not about who typed in it. As a statement about the partition, it is wrong.
- **The value is false.** The content hash is no longer `4da3875d…`. It moved for three independent
  content reasons: arm C's sizing (`[volume]`, `[identity_style].repeats`), the seed schedule
  (`[seed_schedule]` replacing `volume.repeat_seeds`), and the three rewordings.

### 2.2 The current hashes, determined for this document

Both recomputed on 2026-08-15 by running the verifier and reading the sidecars it maintains, not by
quoting anything:

```
$ .venv/bin/python scripts/check_style_partition.py
configs/transfer25/styles.toml  rule=T40_STYLES_V1  committed=2026-08-15
  TRAIN_STYLES 10   EVAL_STYLES 5   identity 1 (control, in neither)
  arm B/D 4020 clips (x1)   arm C 4020 clips (x10)   eval 5/episode
  ...
  hash        OK  reproducible across re-read and key reordering; sidecar matches
              8d8565ffcd12ad17318f10979abb44f639084e21857866cbc3b2f32f1332628b
  derived     OK  consumer JSON rendering matches the committed source
  todos       1 OPEN of 1 — these block GENERATION, not this check
PASS
```

| quantity | value | what it is |
|---|---|---|
| **partition content hash** — this is `T40_RULE_V1` §6's *"style partition hash"* | `8d8565ffcd12ad17318f10979abb44f639084e21857866cbc3b2f32f1332628b` | sha256 over the canonical JSON of `configs/transfer25/styles.toml` **minus its `[hash]` and `[consumer]` tables** (`json.dumps(..., ensure_ascii=True, sort_keys=True, separators=(",",":"))`). Sidecar: `configs/transfer25/styles.toml.sha256`, which matches |
| **rendering byte hash** | `20c250d44f1bf117fecb2204f176ef1d4fb0b89c5627edf471d1a44bbcb5367a` | plain `sha256sum` of the derived consumer file `configs/transfer25/pr08_style_partition.json`. Sidecar: `configs/transfer25/pr08_style_partition.json.sha256`, which matches. This is what the sbatch compares as `STYLES_FILE_SHA` before it trusts the rendering |
| **superseded** | `4da3875d0c76e9b23821c1ca9fe20f965f9fc0867edcd085fb792b587a680da8` | the value recorded in `T40_RULE_V2` §0 and §5. **It identifies no partition that exists.** |

The two are **different kinds of hash and are not interchangeable**: one is over content (immune to
comment rewording, key order and whitespace), the other is over bytes. The value `T40_RULE_V1` §6
requires to be recorded with the verdict is the **content** hash.

### 2.3 Every string that changed today, verbatim before and after

Three, all recorded at the style itself in `configs/transfer25/styles.toml` under `WAS:` /
`REWORDED` comments, all caused by the geometry lint gaining vocabulary (a false positive there
costs one edit to a file nobody has generated from; a false negative costs a corpus):

**(1) `train-04-slate-lowkey` — an axis slug AND a prompt string** (`styles.toml:559-578`). The
lint's `FORBIDDEN_TERMS` now holds the plain stem `"low"` and derives `lower`/`lowest` from it;
previously only the comparative was listed, which is what let *"A low table, waist-high."* through.

```
slug    WAS: lighting = "low-key-desk-lamp"
        NOW: lighting = "moody-dim-desk-lamp"

prompt  WAS: "... lit by a single dim desk lamp, low key with deep shadows. ..."
        NOW: "A brown russet apple with rough dry skin on a grey slate slab, against a dark green
              painted wall, lit by a single dim desk lamp, moody with deep shadows. The white plate
              keeps its own appearance. Scene geometry, camera framing and the robot are unchanged."
```

The `id` is unchanged (`train-04-slate-lowkey` — `"lowkey"` is one token the lint cannot read as
`"low"`). The light described is unchanged: one dim desk lamp, deep shadows, most of the frame dark.

**(2) `train-08-cork-softbox` — a prompt string** (`styles.toml:607-622`). The stem `"mirror"` was
added to catch *"The scene is mirrored horizontally."*, which names no object, fires no position
word, and swaps left for right in the pixels while the carried-over trajectory still reaches right.

```
prompt  WAS: "A dark crimson apple with a mirror gloss on a cork covered table top, ..."
        NOW: "A dark crimson apple with a polished gloss on a cork covered table top, against a
              pale blue wall, lit by neutral softbox lighting, gentle and even. The white plate
              keeps its own appearance. Scene geometry, camera framing and the robot are unchanged."
```

The apple slug `dark-crimson-glossy` is unchanged — it never said *mirror*.

**(3) `eval-01-terracotta-rim` — a prompt string** (`styles.toml:675-693`). The lint gained a
spatial-relation group after a reviewer showed *"The apple sits to the left of the plate."* passed
clean; `"left"` then matched here, where it described the **lighting** direction and was innocent.
The style was fixed rather than the term exempted, because *"exempt `left` when it follows `light
from the`"* is exactly the shape a real violation takes.

```
prompt  WAS: "... lit by a strong rim light from the left with dark falloff. ..."
        NOW: "A smooth lime green apple on a terracotta tiled surface, against a lavender painted
              wall, lit by a strong rim light from one side with dark falloff. The white plate
              keeps its own appearance. Scene geometry, camera framing and the robot are unchanged."
```

The lighting slug `strong-side-rim` is unchanged.

**Recorded for completeness, and it did NOT move the digest:** the `[hash].rule` string was
corrected from `sha256(canonical_json(document minus [hash]))` to
`sha256(canonical_json(document minus [hash] and [consumer]))`. The code's `HASH_EXCLUDES` has
always been `("hash", "consumer")`, so the old rule string could never reproduce its own digest.
`[hash]` is itself outside the hash, so this correction changes no digest — but a rule string that
cannot reproduce its own value is worse than no rule string, and it is noted here so a future
reader does not mistake the fix for a content change.

**The structural changes that DID move the digest**, alongside the three rewordings: `[volume]`
(arm C's `repeats`, the clip counts, the two totals, the ceiling-scope keys) and `[seed_schedule]`
replacing `volume.repeat_seeds`.

### 2.4 The registration

**`T40_RULE_V3` registers `8d8565ffcd12ad17318f10979abb44f639084e21857866cbc3b2f32f1332628b` as the
partition content hash `T40_RULE_V1` §6 requires to be recorded with the verdict, and records
`T40_RULE_V2`'s `4da3875d…` as SUPERSEDED.**

- **`4da3875d…` must not be quoted with any verdict, in any report, or in any run record.** It
  names a partition that no longer exists and that no clip will ever be generated under.
- **The authority for the partition hash is the sidecar `configs/transfer25/styles.toml.sha256`,
  recomputed by `scripts/check_style_partition.py` on every run — never a value quoted in prose,
  including the value quoted in this section.** If they ever disagree, the sidecar and the checker
  are right and this document is stale, exactly as V2's §0 is stale now.
- The **rendering** byte hash `20c250d4…` is recorded beside it because the sbatch verifies the
  rendering by bytes before the checker re-derives the content hash from the source. Two files, two
  kinds of digest, both recorded, neither substitutable for the other.
- `T40_STYLES_V1` remains the partition's rule name. The partition file argues its own
  amended-in-place-while-still-V1 status (no clip exists, so no finished run's recorded digest is
  orphaned) and states that the next amendment is very likely a `T40_STYLES_V2` because *"the
  moment a clip exists this comment stops applying."* V3 records that reasoning and does not
  disturb it; the partition's versioning is the partition's to keep.

---

## 3. The GPU-h ceiling: the reading, and what is superseded

### 3.1 The contradiction

`T40_RULE_V2` §3.2 requires, in three numbered steps: (1) take the timed measurement; (2) derive
the ceiling **over the whole partition**, all 25 style-instances / 10 050 clips; and (3)

> Split that whole-partition ceiling across the `STYLE_SET` invocations and chunks, so the sum of
> the `CEILING_GPU_H` values actually passed to `97_transfer25_restyle.sbatch` is bounded by the
> whole-partition figure.

For a period on 2026-08-15 the sbatch implemented a **single-number** reading — `CEILING_GPU_H` as
"one number for the WHOLE partition", gated against the **whole-partition** projection. The two are
irreconcilable, and the way they fail is not subtle: an operator who follows V2 step 3 literally
passes each submission its share, and **each share is smaller than the whole-partition projection
it is compared against, so all three submissions fail the gate.** A reviewer demonstrated it with
the arithmetic — at a 1 068.19 GPU-h whole-partition projection the projection-proportional shares
are **427.28 / 213.64 / 427.28** (train 10/25, eval 5/25, identity 10/25) and every one of them is
below 1 068.19. **The experiment was unlaunchable by anyone obeying the registered rule.**

*(1 068.19 is the sbatch header's worked illustration, not a measurement. No timed episode exists,
so no budget line exists — `T40_RULE_V1` §8 item 3.)*

### 3.2 The sbatch's actual current behaviour, read rather than remembered

Verified against `cluster/discoverer/97_transfer25_restyle.sbatch` on 2026-08-15. **The file no
longer requires one number for the whole partition.** It requires **two**, both with **no default**:

| variable | scope | gate |
|---|---|---|
| `PARTITION_CEILING_GPU_H` | V2 §3 step 2's figure — the whole partition, all 25 style-instances / 10 050 clips. The **same** number in all three `STYLE_SET` submissions of a `RUN_ID`, pinned run-wide in `${OUT}/run.env`; a later submission that contradicts it is refused | the **whole-partition projection** must fit it (`:1397`) |
| `CEILING_GPU_H` | V2 §3 step 3's **share** for *this* invocation. Pinned per style set in `${OUT}/ceiling_shares/<set>.share`; a share that changes between submissions of the same set is refused | **this style set's** projection must fit it (`:1413`), `MAX_PASSES` is derived from it (`:1359`), and it may not exceed `PARTITION_CEILING_GPU_H` (`:322`) |

and it enforces the sentence V2 step 3 actually is: **the sum of the pinned shares must fit
`PARTITION_CEILING_GPU_H`** (`:1406`), read from disk across the three submissions rather than from
the submit line, so "one budget spent three times" is caught by the sum instead of by forbidding
shares. One persisted ledger under `${OUT}/gpu_ledger/` reserves every pass's worst case before it
runs, bounded both by the set's share and by the run total. `TIMING=1` asks for neither ceiling and
records `ceiling_gpu_hours_supplied_at_measurement_time: null`, because the measurement is what
derives the budget and cannot be gated on it.

### 3.3 The reading registered

**`T40_RULE_V3` registers the two-quantity reading. `T40_RULE_V2` §3.2 steps 1–3 stand exactly as
written — the ceiling is derived over the whole partition and then split across the `STYLE_SET`
invocations, and the shares sum within the whole-partition figure. Both quantities are
pre-registered terms of PR-08 and both are recorded with the verdict.**

Explicitly:

- **`CEILING_GPU_H` means a share, never the whole partition.** V2 §3 step 3 is the operative
  sentence and it is not weakened.
- **`PARTITION_CEILING_GPU_H` is V2 §3 step 2's whole-partition figure**, named separately so that
  the property step 3's split would otherwise lose — that the *whole experiment* fits a budget
  derived from a measurement — is still gated. V2 states step 2 and step 3 as separate steps; V3
  registers that they are separate *quantities*, which is what makes both checkable at once.
- **Superseded: the single-number reading**, i.e. `CEILING_GPU_H` as the whole-partition ceiling
  compared against the whole-partition projection. It is recorded here as superseded because it was
  briefly the file's behaviour, it satisfies V2 step 2 while making V2 step 3 unfollowable, and it
  fails all three submissions of anyone who obeys the rule. It must not be reintroduced.
- **Neither has a default, and neither may acquire one.** `T40_RULE_V1` §8 item 3: *"No budget line
  exists until that measurement does."* A default is a budget nobody measured.
- **The split is derived and recorded, not chosen at the gate.** At the committed partition a
  projection-proportional split is train 10/25, eval 5/25, identity 10/25 of the whole-partition
  figure. Any other split is permitted only with its derivation recorded alongside — what may never
  happen is raising `PARTITION_CEILING_GPU_H` so a recipe fits. If the whole partition does not fit
  the measured budget, *that is the finding*: record the shortfall, cut the style count or the
  episode count, re-derive.
- **Recorded with the verdict:** the timed measurement, `PARTITION_CEILING_GPU_H`, the three
  `CEILING_GPU_H` shares, the derivation of the split, and the ledger's final totals — alongside
  `T40_RULE_V1` §6's existing list.

**One operational hazard, recorded for a human and deliberately NOT fixed here** (V3 changes no
code): the sbatch prints `proportional_shares_from_projection` with `round(..., 2)`, and the sum
check is a strict `>`. Rounded half-up shares can exceed the ceiling by a cent of a GPU-hour —
427.28 + 213.64 + 427.28 = **1 068.20**, against a `PARTITION_CEILING_GPU_H` of 1 068.19 — and the
sum gate would then fail an otherwise correct split. **The shares must be derived so that their sum
is ≤ the whole-partition ceiling, i.e. rounded down.** This is arithmetic hygiene in deriving the
split, not a change to any gate, and no threshold moves either way.

---

## 4. `"per-step"` in `GEOM_TOL`, registered against the corpus's measured rate

### 4.1 Why an unregistered default is a coined threshold

`T40_RULE_V1` §6 opens *"No threshold is coined"*, and defines G0b's tolerance as

> `GEOM_TOL = median per-step object-centroid displacement in the source clips`

**and never defines "step".** `GEOM_TOL` scales **~linearly** with whatever a step is taken to be:
gate at one frame and the tolerance is roughly a tenth of what it is at ten. A silent default
therefore does not pick a *convention*, it picks **the G0b tolerance**, by a factor of the step,
and nothing anywhere would record that a threshold had been chosen. That is the same defect
`CEILING_GPU_H`'s no-default discipline exists to prevent, wearing different clothes.
`scripts/measure_geom_tol.py` already flags this as the first of "three places PR-08 §6 is silent",
and the sbatch has removed `GEOM_STEP_FRAMES`'s former silent default of 1 and now **refuses** on
the generation path until the step is supplied — correctly, because it is a registered quantity and
not an operator preference.

### 4.2 The measurement — working shown

**Source A — the corpus's own metadata.** `meta/info.json` **is** available on this workstation,
inside the HF snapshot the local episode symlinks into:
`~/.cache/huggingface/hub/datasets--nvidia--GR00T-N1.7-AppleToPlate/snapshots/d89c126a713c6632432a607c12661546ff4d6ea9/meta/info.json`

```json
"codebase_version": "v2.1",  "robot_type": "unitree_g1",
"total_episodes": 402,  "total_frames": 171625,  "total_videos": 402,  "fps": 30,
"observation.state": {"shape": [43]},  "action": {"shape": [43]},
"observation.images.ego_view": {"shape": [480, 640, 3],
    "info": {"video.height": 480, "video.width": 640, "video.fps": 30, "video.codec": "av1"}}
```

`171625 / 402 = 426.93` mean frames per episode, which is `T40_RULE_V1` §8 item 3's "~427" measured
rather than repeated. The ego_view stream is **30 fps at 640×480**, which is also the resolution
`T40_RULE_V1` §3 fixes for the restyle.

**Source B — the parquet's own timestamps.**
`~/models/apple_pnp_golden/dataset/data/chunk-000/episode_000000.parquet`, 590 rows,
read with `.venv/bin/python` + `pyarrow`:

```
rows                       590
timestamp[0:4]             0.0, 0.03333334, 0.06666667, 0.1
diff(timestamp)  median    0.033333302 s     min 0.033332825   max 0.033334732
                 -> fps    30.000029  (float32 storage noise only; 11 distinct values, all ~1/30)
duration                   19.633333 s over 589 intervals -> 30.000000 Hz
diff(frame_index) unique   [1]        (exactly one row per frame, no gaps, no repeats)
```

**Source C — is the 30 Hz row rate the *action* rate, or a slower control tick upsampled into it?**
This is the question that decides whether the step is 1 frame or *n*. If the recorded action were a
decimated control tick held constant across frames, consecutive `action` rows would repeat in a
staircase. They do not:

```
consecutive-identical action rows          0 of 589
steps with max|Δaction| <= 1e-9            0.0000
steps with max|Δaction| <= 1e-4            0.0000
gaps between changing steps, unique        [1]      (the action changes at EVERY frame)
```

**Every one of the 589 transitions moves the 43-dim action vector**, at every scale tested down to
1e-9. There is no zero-order hold and therefore no evidence of a slower control tick upsampled to
the frame rate.

**Conclusion.** AppleToPlate carries **exactly one action row per `ego_view` frame, at 30 Hz**.
`frame_index` increments by one, `timestamp` increments by 1/30 s, and the action advances at every
one of them. One video frame *is* one action step; there is no decimation factor to discover.

### 4.3 The registration

**`T40_RULE_V3` registers, for `T40_RULE_V1` §6's `GEOM_TOL`:**

| | registered value |
|---|---|
| **one step** | **one source frame** of the `observation.images.ego_view` stream |
| `GEOM_STEP_FRAMES` | **1** |
| step duration | **1/30 s = 0.0333 s**, at the corpus's `fps = 30` |
| step construction | **overlapping offsets `i → i+1`** (not disjoint windows) — the two are different measurements that would both report `step_frames = 1` |
| units and grid | **pixels at 640×480**, the source resolution `T40_RULE_V1` §3 fixes and the same grid `EST_DRIFT_P95` must be measured on, because §6 subtracts one from the other and that subtraction is arithmetic only on one grid |
| what the artifact must carry | `step_frames = 1`, `step_seconds`, `fps`, and a non-empty `step_definition` in prose. The sbatch refuses an artifact whose `step_frames` differs from `GEOM_STEP_FRAMES`, and refuses one carrying no `step_definition` at all |

**Registered against the corpus, not chosen.** The rationale in `T40_RULE_V1` §6 is that the
tolerance must be *"what one action step actually moves the scene"* — a drift larger than that makes
the carried-over label describe a different scene than the pixels. On this corpus one action step is
one frame, measured, so the registered step is one frame. It is the rate of the **label stream that
is carried over** — the thing G0b is about — and not a convention imported from elsewhere.

**Two honest limits, stated so they cannot be discovered later:**

- **The staircase test is n = 1 episode.** One of 402 is on this workstation. It rules out an
  upsampled slower control tick *in `episode_000000`*; the corpus-wide claim rests on `info.json`'s
  single `fps: 30` and on the LeRobot v2.1 one-row-per-frame layout, which are corpus-wide but are
  metadata rather than a per-episode measurement. A cheap confirmation exists and is worth taking
  when the 640×480 re-derivation of `T40_RULE_V1` §3 is built: re-run the staircase test across all
  402 and record the fraction of held steps.
- **A faster *upstream* teleop controller, decimated to 30 Hz before storage, is invisible from
  here and would not change the registration.** G0b's budget is against the label the restyled
  frame is paired with, and there is exactly one such label per frame. A hypothetical 200 Hz
  upstream tick nobody stored is not a step this experiment can be gated on.

**This is a registration, not a re-coining.** The registered value happens to equal the silent
default that was removed from the sbatch. That coincidence is not the argument — the argument is
§4.2 — and the registration is what turns the number from an accident into a choice with a
derivation attached. **`GEOM_TOL` itself is still derived, still measured on the corpus, still
committed before generation, and V3 supplies no value for it.** No `configs/transfer25/pr08_geom_tol.json`
exists yet (§5.2); §8 item 4 is open.

---

## 5. Generation is still forbidden, and one T-39 outcome forbids it permanently

### 5.1 Nothing here licenses generation

**`T40_RULE_V1` §1's prohibition binds in full and is unchanged: no corpus is generated, no weight
is trained on generated frames, and no number from PR-08 is quoted as a result, until EVERY §8 item
is closed AND T-39 has reported.** §1 licenses writing the pipeline, committing the style
partition, measuring the estimator error budget, and timing one episode on an H200 — and that is
all it licenses. Writing this document is on the licensed side; so was committing the seed
schedule.

### 5.2 The §8 items, checked rather than remembered (2026-08-15)

| item | status |
|---|---|
| 1 — the recipe (`--tune-visual`, Recipe B, lr 5e-5) | fixed in V1; not disturbed here |
| **2 — the consumer contract** | **OPEN.** See §5.3 |
| 3 — a measured throughput number and a GPU-h ceiling derived from it | **OPEN.** No timed episode exists, so no `THROUGHPUT.json` and no budget line. §3 registers the *shape* of the ceiling, not a value |
| 4 — `GEOM_TOL` and `EST_DRIFT_P95` measured and committed | **OPEN.** `configs/transfer25/pr08_geom_tol.json` does not exist (`configs/transfer25/` holds only `styles.toml`, `pr08_style_partition.json` and their two sidecars). §4 registers the step the measurement must be taken at; it supplies no number |
| 5 — depth and segmentation annotators in `isaac_binding.py` | the annotators now exist in `src/wam/robot/isaac_binding.py` as an opt-in alongside `rgb`, with tests. The **code** half appears to have landed since V1; the **measurement** it unblocks (`EST_DRIFT_P95`, item 4) has not been taken. Recorded as an observation, not adjudicated here |
| **6 — the partition committed** | **NOT CLOSED.** See below |
| 7 — T-39 has reported | **OPEN.** See §5.3 |

**§8 item 6 is NOT closed.** `T40_RULE_V1` §5 requires the partition *"as a committed file, **in
git**, before the first clip"*, and:

```
$ git ls-files configs/transfer25/
(no output — zero tracked files)
$ git status --short configs/transfer25/ scripts/check_style_partition.py \
      cluster/discoverer/97_transfer25_restyle.sbatch
?? cluster/discoverer/97_transfer25_restyle.sbatch
?? configs/transfer25/
?? scripts/check_style_partition.py
```

The partition, its rendering, both sidecars, the verifier and the sbatch are **all untracked**. A
file that is not in git cannot be the pre-commitment §5 asks for: the whole property is that the
eval domain could not have been chosen after seeing which restyles came out well, and only the
history proves the order. `styles.toml` itself asserts it "Closes PR-08 §8 item 6" — **it does not,
until it is tracked.** This is a git operation for a human; V3 makes no commit and pushes nothing.

### 5.3 Two conditions that are not a formality

**§8 item 2 remains OPEN because it names the wrong corpus.** `T40_RULE_V1` §8 item 2 requires the
consumer contract written down as *"LeRobot v3.0, 28-dim arms+hands, right hand
index-before-middle"*. `T40_RULE_V2` §4 found that this describes `unitreerobotics/G1_Dex3_*`, not
the `nvidia/GR00T-N1.7-AppleToPlate` corpus §3 chose to restyle. **V3 adds one piece of evidence and
resolves nothing:** V2 §4.3 recorded that the v2.1 version claim was *"not verifiable from the one
local file — no `meta/info.json` is present on this workstation"*. **That is no longer true.** The
snapshot's own `meta/info.json` is present (§4.2, path given there) and states
`"codebase_version": "v2.1"`, `"total_episodes": 402`, `"total_frames": 171625`, and
`observation.state`/`action` of shape `[43]` — so **v2.1 and 43-dim are now measured on the corpus's
own metadata, not carried from a contract document.** Two of §8 item 2's three descriptive fields
are wrong for the corpus PR-08 operates on, now on direct evidence. Its **fourth clause** — *"the
action labels come from the source recording, never from the generator"* — is correct, is the
load-bearing clause, and is untouched.

Choosing which corpus the contract is *supposed* to name is a decision about what the deliverable
is, and it is **not an agent's to make**. §8 item 2 stays **OPEN**; the resolution, when a human
makes it, is a further version alongside these three, never an edit to any of them.

**A T-39 verdict of VOID must STOP PR-08, not unlock it.** This is the one outcome where "T-39 has
reported" and "PR-08 may proceed" come apart, so it is registered explicitly:

- **PR-07 §4** defines the outcome: if the ground-truth `action` column itself cannot clear L1 under
  our scorer, then *"**no policy trained on this dataset can clear our bar**, and the finding is
  about our label pipeline, not about GR00T."*
- **PR-07 §6**'s VOID row licenses exactly one thing — *"a defect report against the adapter
  (`oracle_state`) or the label pipeline (`oracle_action`)"* — and forbids *"any statement about
  GR00T"*.
- **`T40_RULE_V1` §1**'s stated reason for gating on T-39 is that until T-39 reports whether **any**
  method clears the bar on this corpus, *"the data is wrong"* and *"the method is wrong"* are not
  separable, and **generating data is a bet on the first**.

A VOID does not settle that bet — **it loses it, on the record, in the direction that says the
corpus and its labels are the problem.** PR-08 restyles 10 050 clips whose action labels are copied
**unchanged** from the very pipeline a VOID indicts. Spending the allocation on it would be
spending it on the one hypothesis T-39 had just ruled out. So: **`T40_RULE_V3` registers that a
T-39 VOID closes PR-08 rather than opening it. P, N, M and I satisfy §1's "T-39 has reported"; VOID
does not.**

The sbatch now enforces this, verified by reading it (`97_transfer25_restyle.sbatch:353-409`): the
verdict token pattern still admits `VOID` **so that a VOID attestation reaches its own diagnosis
instead of falling out as "no verdict token"**, and the branch immediately below **refuses** it.
Until 2026-08-15 the file did the opposite — `VOID` was accepted and the error message's worked
example was literally `VOID (labels) 2026-08-14`, so the gate satisfied §1's letter while inverting
its reason and taught the operator to type the one verdict that must stop the job.

There is an override, `PR08_OVERRIDE_T39_VOID`, and it is deliberately not a boolean but one exact
sentence that has to be typed out, recorded verbatim in `chunk_metadata.json` beside the verdict.
The sbatch's own message says it *"belongs in a V3 before it belongs on a submit line."* **This V3
does not grant it, does not exercise it, and registers no circumstance under which it may be used.**
Using it would be generating against PR-07 §4's finding, and that is a human decision requiring its
own registered document.

---

## 6. Provenance

| | |
|---|---|
| rule | `T40_RULE_V3` |
| registered | 2026-08-15 |
| supplements | `T40_RULE_V1` and `T40_RULE_V2`, both of which stand and are **unedited** |
| supersedes | exactly one thing: `T40_RULE_V2` §0 / §5's `source_content_sha256 = 4da3875d…` and its stated justification (§2). Also records as superseded the single-number `CEILING_GPU_H` reading that was briefly the sbatch's behaviour (§3) |
| registers | the seed schedule and its assignment rule (§1); the current partition hash (§2); the two-quantity ceiling reading (§3); `GEOM_STEP_FRAMES = 1` and what a step is (§4) |
| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style** |
| corrects in a registered document | `T40_RULE_V2` §2.1's "cannot manufacture a **P**" (incomplete — §1.3); `T40_RULE_V2` §0/§5's partition hash and its justification (§2); `T40_RULE_V2` §4.3's "no `meta/info.json` is present on this workstation" (§5.3) |
| leaves open | `T40_RULE_V1` §8 items 2, 3, 4, 6 and 7 (§5.2). Item 6 is open for a reason V2 did not record: `configs/transfer25/` has **zero tracked files** |
| partition | `configs/transfer25/styles.toml`, rule `T40_STYLES_V1`, content sha256 **`8d8565ffcd12ad17318f10979abb44f639084e21857866cbc3b2f32f1332628b`**; rendering `configs/transfer25/pr08_style_partition.json`, byte sha256 **`20c250d44f1bf117fecb2204f176ef1d4fb0b89c5627edf471d1a44bbcb5367a`**. Authority is the sidecars + `scripts/check_style_partition.py`, never a value quoted in prose |
| seeds | train `[7001..7010]`, identity `[7001..7010]` (identical by rule), eval `[7011..7015]` (disjoint from both). Assignment `style-instance-index` |
| measurements | §2.2 and §4.2, taken 2026-08-15 with `.venv/bin/python` against `configs/transfer25/`, the HF snapshot's `meta/info.json`, and `episode_000000.parquet` (590 rows). The staircase test is **n = 1 episode** |
| decided by | nothing here is a design decision. §1–§4 register what is committed or what the corpus measures; every open question is left open and named |
| generation licensed | **no** |
| a T-39 VOID | **stops PR-08.** It does not satisfy §1 |
