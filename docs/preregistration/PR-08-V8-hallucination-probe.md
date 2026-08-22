# PR-08 V8 — can Cosmos-Transfer2.5 hallucinate a manipulator into a robot-free frame?

**Rule `T40_RULE_V8`. Registered 2026-08-23, before any frame is generated, before any weight is
trained, and before any job is submitted. Nothing has been generated; no clip and no probe frame
exists.**

> ## THIS DOCUMENT IS IN FORCE AS OF 2026-08-23.
> The determination in §8 was **signed by the project owner on 2026-08-23** and transcribed by the
> session, on the terms §8 sets out. Before that it licensed nothing and the probe refused to run by
> rule name. **No agent may sign this document, and a session that fills the owner line in without
> the owner having decided has forged it.**
>
> **What it determines, when signed, is narrow: that at most 726 source frames, selected *because
> the robot is absent from them*, may be passed through the frozen generator once, for the sole
> purpose of finding out whether the generator puts a manipulator into a frame that does not
> contain one.** It opens nothing else. `T40_RULE_V1` §1's prohibition on **generating a corpus**
> binds in full and is untouched by this document.

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md) (`T40_RULE_V1`),
[`PR-08-V2-arm-c-frame-matching.md`](PR-08-V2-arm-c-frame-matching.md) (`T40_RULE_V2`),
[`PR-08-V3-seed-schedule.md`](PR-08-V3-seed-schedule.md) (`T40_RULE_V3`),
[`PR-08-V4-t39-gate-premise.md`](PR-08-V4-t39-gate-premise.md) (`T40_RULE_V4`),
[`PR-08-V5-ground-truth-route.md`](PR-08-V5-ground-truth-route.md) (`T40_RULE_V5`),
[`PR-08-V6-mask-validity.md`](PR-08-V6-mask-validity.md) (`T40_RULE_V6`) and
[`PR-08-V7-consumer-contract.md`](PR-08-V7-consumer-contract.md) (`T40_RULE_V7`). **None of the
seven has been edited and none may be.** The discipline is `docs/handoff.md` §3 — *"Rules are
versioned, never edited in place. A gate rewritten after seeing its output is not a gate."* V8 is
that versioning, not a revision.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

---

## 0. What V8 does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent. **V8 moves no gate, no threshold, no verdict, no arm,
no clip count, no style, no seed, no ceiling, no estimator setting and no consumer field.** It adds
one measurement and forbids everything that measurement could be mistaken for.

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined here. V8 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` | still **derived** — the median per-step object-centroid displacement in the source clips, measured and committed before generation, at `T40_RULE_V3` §4's step (`GEOM_STEP_FRAMES = 1`). Still `null` in `configs/transfer25/pr08_geom_tol.json` as this is written. V8 supplies no value and changes no method |
| `EST_DRIFT_P95` | still **measured** per V1 §4 as amended by `T40_RULE_V5`, still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still **not a pass**. Still unmeasured by any route |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start. **V8 measures no geometry and proposes no tolerance** |
| **G0c** embodiment | **unchanged, including its refusal.** The real robot's pixels are still unconditionally composited back over every generated frame; `robot_composite.check_mask` still refuses a clip on an empty robot mask with no threshold; robot-mask IoU is still a diagnostic and **never** a gate. **V8 changes not one line of `scripts/robot_composite.py` and proposes no change to it.** It measures the premise underneath that refusal's error text and stops there |
| **The G0c robot-mask area bound** | unchanged and still **uncommitted**. `configs/transfer25/pr08_robot_mask_area.json` still carries `max_frame_fraction: null`; `load_area_bound` still refuses; V8 **derives no bound, suggests no bound, and writes no number into that file** |
| **The ladder** | unchanged — **L1** `skill_vs_repeat_pct > 0`, **L2** `ci_skill_vs_repeat_pct > 0` (`ci_` = the task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (V1 §6) | unchanged in every cell, including that **P** requires *both* B − A ≥ floor *and* B − C ≥ floor, that **F** is the generator-attributable case, that **N** is B − A ≤ 0, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P. **No outcome of this probe is any of those letters** |
| **Arms A / B / C / D** | unchanged. B is the intervention, C is the generator-fingerprint control, D is diagnostic and never the headline |
| **Arm C's size** (`T40_RULE_V2`) | unchanged — R2, frame-matched: 4 020 identity clips against arm B's 4 020, and **arm B is still not subsampled** |
| **Clip totals** | unchanged — train 4 020, identity 4 020, eval 2 010, whole partition 10 050 over 25 style-instances ≈ 4.29 M frames |
| **The seed schedule** (`T40_RULE_V3` §1) | unchanged — train `[7001..7010]`, identity identical, eval `[7011..7015]` disjoint, assignment `style-instance-index`. **V8 coins no seed**: the probe uses the committed seed of the committed style it runs, and nothing else |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json` / `configs/transfer25/styles.toml`, rule `T40_STYLES_V1`. V8 changes **no style, no id, no slug, no prompt string, no repeat count and no seed**, and therefore changes no partition hash. The probe **reads** the rendering and verifies it exactly as `97` does |
| **The two-quantity GPU-h ceiling reading** (`T40_RULE_V3` §3) | unchanged. V8 supplies no ceiling value and exempts nothing from one. The probe is not a chunk of the partition and consumes none of its budget |
| **`T40_RULE_V4`'s determination** | unchanged — §8 item 7 is closed on `VERDICT N`; a T-39 **VOID** still closes PR-08 rather than opening it; `PR08_OVERRIDE_T39_VOID` is still not granted, still not to be used, and its value is written nowhere here either |
| **`T40_RULE_V5`'s ground-truth route** | unchanged — the `EST_DRIFT_P95` capture, its registered fields and its stated direction-of-bound are untouched |
| **`T40_RULE_V6`'s mask-validity filter** | unchanged. V8 adds no filter, moves no filter threshold, and does not apply V6's filter to anything |
| **`T40_RULE_V7`'s consumer contract** | unchanged. V8 writes nothing a consumer reads, so it has no contract |
| **§1's prohibition on generating a corpus** | **unchanged and still binding in full.** Nothing is generated as a corpus, no weight is trained on generated frames, and no number from PR-08 is quoted as a result, until **every** §8 item is closed **and** T-39 has reported. **§8 items 3 and 4 are open.** §3 below licenses a diagnostic that is not a corpus and says exactly why that is not a loophole |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a **P** is a claim about generalising to held-out *generated* appearance, and it licenses exactly one thing: recording a small real shifted eval set and re-running arms A and B against it |
| **§8 item 3's throughput measurement** | **unchanged, open, and NOT satisfiable by this probe.** §4 forbids the substitution explicitly. The timed episode on an H200 remains the timing run's job alone |
| **`docs/benchmark.md`'s L4 gate** | untouched. `CLAUDE.md` records it as a separate open decision for the owner, with bench specs 0.1.0 and 0.2.0 disagreeing about the repaired cell. V8 says nothing about it |
| **`CLAUDE.md`'s training gate** | untouched and unliftable here. Whether training may start, and against which label space, is the project owner's call. `C`, `W` and `N` are none of them permission, and neither is any outcome of this probe |

---

## 1. Why a diagnostic needs a registration at all, and the circularity that is V8's actual reason to exist

### 1.1 The licence in `T40_RULE_V1` §1 is an enumeration, not an exclusion

§1's licensing sentence is exactly four items:

> **Licenses:** writing the pipeline, committing the style partition, measuring the estimator error
> budget (§4), and timing one episode on an H200 (§8 item 3).

A diagnostic restyle of a few hundred frames is **not a corpus** — §1's forbidding sentence names
*"generating a corpus, training any weight on generated frames, and quoting any number from this
document as a result"*, and none of those three describes it. But it is **not on the licensed list
either.** A document that licenses by enumeration does not silently permit what it failed to
imagine, and reading it that way is how a narrow diagnostic becomes a wide generation path one
convenience at a time. **So it is registered before it runs, in advance, in a versioned file, with
its forbidden readings written down beside its licensed one.**

### 1.2 The circularity, which is the honest reason this document exists

This is not a convenience. Four facts close a loop that **nothing in the licensed set can break**:

1. **`robot_composite.check_mask` refuses a clip on an empty robot mask, with no threshold.** Its
   words: *"There is no threshold in this check and no number to loosen: zero is zero."* A clip is
   refused on its **first** empty frame.
2. **The robot is genuinely absent from 36.2 % of this corpus's frames** (§2). With a median of
   **152 robot-absent frames per episode**, every clip contains one, so **G0c as written cannot
   produce a single composited clip** — `runs/pr08-robot-mask-empty/DIAGNOSIS.json`,
   `practical_consequence.headline`, in those words.
3. **The licensed timing run is blocked behind the same G0c prerequisites.** `97_transfer25_restyle.sbatch`
   composites on the `TIMING=1` path too, deliberately — *"THIS MAKES THE TIMING PATH DEPEND ON
   G0c's OWN PREREQUISITES"* (`97:1178`) — and job **189644** stopped on `load_area_bound`'s refusal
   because no G0c robot-mask area bound has ever been committed.
4. **The bound cannot be chosen without knowing what an empty robot mask means on a robot-free
   frame**, because the corpus's observed maximum *is* the defect the bound exists to catch
   (10 of the 11 whole-tablecloth masks contain no robot), and because the threshold-free option
   the T-040 record names — *"run the committed masker on the generated frame too, and accept when
   both are empty"* — is only sound if the generator cannot put a robot where the source had none.

So: **bound → timing → this answer → bound.** The timing run cannot be the instrument that answers
it, because the timing run is the thing waiting on the answer. The estimator error budget (§4) is a
segmentation measurement about the apple and says nothing about it. Committing the style partition
is done. Writing the pipeline is done. **There is no licensed activity that produces this fact, and
PR-08 cannot proceed without it.** That is what V8 is for, and it says so plainly rather than
presenting itself as a small extra.

### 1.3 One thing the loop does *not* justify

**It does not justify widening the licence to "whatever unblocks PR-08".** The reasoning above
identifies exactly one missing fact and licenses exactly the measurement of that one fact, at a
hard cap, with the artifact forbidden from reaching any consumer (§3, §4). If a *second* missing
fact is found, it gets a `V9`; it does not get read into this one.

---

## 2. The evidence that motivates it — every number verified against the artifact before being cited

Read from `runs/pr08-robot-mask-empty/DIAGNOSIS.json` (schema
`wam.robot_mask_empty_diagnosis_summary/1`), produced by `scripts/diagnose_robot_mask_empty.py`, and
cross-read against `.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md`'s 2026-08-22
entries. Verified 2026-08-23 while writing this document.

**The robot is genuinely absent, and the pattern is not a detector misfiring.** A **non-learned**
reference predicate — dark relative to the frame's own modal luminance, near-neutral, and different
from the episode's per-pixel temporal median, **with no GroundingDINO anywhere in it** — over 101
episodes / 21 639 frames:

| | |
|---|---|
| robot **present** | 56.5 % |
| robot **absent** | **36.2 %** |
| in the classification band | 7.3 % |
| sensitivity of the absent share over the predicate's own constants | 31.2 – 40.9 % |
| per-episode absent fraction | median 0.387, p10 0.208, p90 0.511, max 0.587 |
| episodes with at least one absent frame | **100 of 101** |

Absent fraction **by corpus decile**, read from `q2_clustering.absent_by_decile_corpus`:

```
d0 65.6 %   d1 23.1 %   d2 1.3 %   d3 0.2 %   d4 0.2 %
d5  2.1 %   d6 20.5 %   d7 64.3 %  d8 90.3 %  d9 94.8 %
```

That is **the approach and the retreat**, U-shaped and blockwise: 98 of 101 episodes end inside an
absent run, 76 of 101 begin inside one.

**Detection is not failing.** Contingency on the pilot's three episodes: **`present_empty = 0`,
`present_nonempty = 917`** — zero detector failures on 917 robot-present frames. The corpus sample
showed 19 apparent failures; `q4_agreement.present_empty_inspected` records that **all 19 were
rendered and looked at and none contains a robot**. Every empty mask is
`no_boxes_above_threshold`; SAM 2 never segmented a box to nothing.

**And the finding nobody went looking for, which shapes this probe's design.** On robot-**absent**
frames the committed masker returns a **non-empty** mask **14 % of the time** on the pilot's
episodes and **41 % corpus-wide** (`corpus.absent`: 98 non-empty of 240), and the mask is **the
apple** (~6–7 k px), the plate (~40–48 k px) or the whole tablecloth (> 0.9 of frame). Their
detector scores (0.150–0.414) overlap true robot detections (0.165–0.623), so **no score threshold
separates them**.

**One more fact, established for this document rather than quoted.** Verified 2026-08-23 against
`configs/transfer25/pr08_style_partition.json`: **all 16 committed prompts — 10 train, 5 eval,
1 identity — end with the clause `"Scene geometry, camera framing and the robot are unchanged."`**
On a robot-free frame, every committed prompt therefore refers to *"the robot"* as though one were
present. That is not a defect in the partition — the clause is there to hold the robot still on the
56.5 % of clip-time where it is in shot — but it is a plausible *mechanism* for the exact failure
this probe asks about, and it is the reason the probe **must** use the committed prompts unmodified
(§3.4). A probe run under a prompt that does not mention a robot would measure a generator this
project is not going to run.

---

## 3. What V8 licenses — one thing, written as narrowly as it can be written

> **`T40_RULE_V8` licenses passing a hard-capped number of source frames, selected *because a
> non-learned reference predicate says the robot is absent from them*, through the frozen
> Cosmos-Transfer2.5 generator once, and running the committed robot masker on the source frame and
> on the generated frame, for the sole purpose of determining whether the generator introduces a
> manipulator that the source frame does not contain.**

Everything below narrows that sentence further. All of it is registered, all of it is enforced in
`cluster/discoverer/107_hallucination_probe.sbatch` and `scripts/probe_hallucination.py`, and none
of it is settable from a submit line.

### 3.1 The cap

| | value | why this and not more |
|---|---|---|
| frames per probe clip | ≤ **121**, default 96 | one contiguous robot-absent run; the pilot's measured leading runs are ~95–175 frames, so 121 fits inside a real one without stitching |
| probe clips | ≤ **6**, default 4 | (≤ 3 episodes) × (≤ 3 committed train styles), so no single episode and no single prompt carries the answer |
| **total generated frames** | ≤ **726** | the product, and the number that matters |

**726 frames in context, so the cap can be judged rather than accepted:** it is **1.70 ×** the mean
episode (427 frames) that §8 item 3's *already licensed* timing run restyles; **0.42 %** of one
style-instance (171 625 frames); and **0.017 %** of the whole registered partition (≈ 4.29 M
frames). The cap is a constant in the script, refused above its value with a non-zero exit, and
**cannot be raised from the submit line**. Raising it is a `V9`.

### 3.2 The frames, and the predicate that picks them

Frames are selected **because they are robot-free**, by `scripts/diagnose_robot_mask_empty.py`'s
**existing** reference predicate (`robot_dark_mask`, via `frame_fields` / `apply_setting`) at its
**primary** committed setting (`dark_offset 45`, `sat_max 0.25`, `change_min 25`), classified by
that module's own `classify` band at `absent_below = 800` / `present_above = 3000` — the band
`DIAGNOSIS.json`'s `instrument.band_px` records. **No new predicate is written and no constant is
re-chosen.** The band is a module constant in the probe, not a flag: a band typed at submit time is
a per-run decision about which frames count as robot-free, recorded nowhere anybody would look.

Every selected frame must classify **`absent`** — not merely "not present". The probe records, per
frame, the reference area in pixels, its largest connected component, its classification, the
episode, the frame index and the run it came from, so *which frames and why* is in the artifact
rather than in this prose.

The predicate is **not ground truth** and V8 does not treat it as such. Its own docstring records
that it **understates** presence (it cannot see the white wrist or bare-metal segments) and
**over-calls** presence at transitions (it scores the arm's moving shadow). Both directions are
recorded in §7.

### 3.3 The pairing, which is the measurement

For every selected frame the committed masker — `robot_composite.build_masker()`, the committed
`ROBOT_TEXT_PROMPT`, the pinned checkpoints, the adapter's own thresholds, **unmodified** — is run
on the **source** frame and on the **generated** frame, and the pair is recorded:

| source mask | generated mask | reading |
|---|---|---|
| empty | empty | **nothing was invented on this frame** |
| empty | non-empty | **candidate invention** — the generated frame grounds something the source never did |
| non-empty | either | **excluded from the headline.** The masker already grounds the apple/plate/cloth on 41 % of robot-absent frames (§2); on such a frame "the generated frame grounds something" proves nothing about invention |

The excluded frames are counted and reported, never silently dropped — their share is itself a
check that the probe's frame population behaves like the corpus the diagnosis measured.

`check_mask` is **not called**. The probe imports the masker and not the gate: G0c's refusal is the
thing being measured around, and running it would refuse every unit before a number existed.

### 3.4 The prompts, the seeds and the generator

The probe uses **committed `TRAIN_STYLES` prompts, verbatim**, in committed id order, with each
style's **committed seed**, against the verified partition rendering (sidecar sha256 + `scripts/check_style_partition.py`,
exactly as `97` verifies it). It coins no prompt, no seed and no style, and it has no flag that
selects `eval` or `identity`. The generator is Cosmos-Transfer2.5 **frozen**, at the staged pinned
revision, with the same `--control` decision `97` refuses to default — which is required and
recorded, never inferred.

Rationale for `train` rather than `eval` or `identity`: `TRAIN_STYLES` is what arm **B**, the
intervention, actually runs; `EVAL_STYLES` is the held-out domain and has no business being touched
by anything before the experiment; the identity style is arm C's and is a different question.

### 3.5 The artifact

Output goes to a single **quarantined** run directory whose name says what it is, carrying a
`NOT_A_CORPUS` marker, per-frame JSON, and **overlay contact sheets a person can look at** — the
same rig `scripts/audit_apple_masks.py` and the diagnosis sheets use, with the source and the
generated frame side by side and both masks outlined. **This artifact exists to be looked at.**
Its `human_review.looked_at` field is written **`false`** and stays false until a person has looked,
for the reason the mask audit records: a model checking masks produced by a pipeline a model wired
up is a correlated observer.

---

## 4. What V8 forbids, explicitly and in this document

Each of these is a live way the probe could be misused, and each is enforced in the job as well as
stated here.

1. **No frame this probe produces may enter a training set, a LeRobot dataset, an evaluation set,
   or any artifact a downstream consumer reads.** Not arm A, B, C or D; not `EVAL_STYLES`; not a
   held-out set; not a demo. The probe writes no `manifest.json`, no `work.jsonl`, no
   `sample_outputs.json`, no `vision.mp4`, no parquet and no `meta/`, and it writes nothing
   `scripts/assemble_restyled_lerobot.py` or `97_transfer25_restyle.sbatch` can consume. The job
   audits its own output tree afterwards and **fails non-zero** if any such file exists.
2. **No number from this probe may be quoted as a PR-08 result.** It is not `P`, not `F`, not `N`,
   not `I`, not `L1`, not `L2`, not a `skill_vs_repeat_pct`, not a `GEOM_TOL`, not an
   `EST_DRIFT_P95`, and not a robot-mask area bound. `T40_RULE_V1` §1's third prohibition —
   *"quoting any number from this document as a result"* — is untouched, and this document adds
   itself to it.
3. **It may not be used to satisfy `T40_RULE_V1` §8 item 3's throughput measurement.** Item 3
   requires *one timed episode on an H200 at 640×480* and a GPU-h ceiling derived from it. This
   probe restyles truncated sub-clips of at most 121 frames under a probe-specific code path; a
   seconds-per-frame number taken from it would be a measurement of a different run shape sold as
   the licensed one. **Item 3 stays open, and the timing run remains the only thing that closes
   it.** The probe's own elapsed time is recorded as an operational fact about the probe and is
   labelled, in the artifact, as not admissible for item 3.
4. **It does not license the G0c area bound, or any edit to `check_mask`.** It measures the premise
   under that refusal's error text. Turning the answer into a rule is a further version and a
   further owner decision.
5. **It does not license a second run under a wider cap.** A repeat under the same cap, for a
   reason recorded before it is run, is inside this rule. A repeat with more frames, more styles,
   more episodes or a different frame-selection predicate is not.
6. **It licenses no training, on real frames or generated ones.** `CLAUDE.md`'s gate stands and no
   session may edit that file to lift it.

---

## 5. What each outcome means, written down before the output exists

**This is the part that stops the probe becoming a gate rewritten after seeing its own output.**
Both outcomes are findings; **neither licenses generation.**

### 5.1 Outcome **H** — the generator invents a manipulator

*Definition, fixed here:* the probe reports **one or more candidate-invention frames** (source mask
empty, generated mask non-empty) and a person, looking at the sheets, confirms that the generated
mask is on a manipulator rather than on the apple, the plate or the cloth.

**What it establishes:** that G0c's premise is wrong in the direction that matters. The
threshold-free option the T-040 record proposed — accept a clip when the source mask and the
generated mask are both empty — is **not** safe on this corpus, because "both empty" would then be
one draw of a stochastic generator rather than a property of the pipeline. It also establishes that
the committed prompts' `"...and the robot are unchanged"` clause is at least compatible with the
generator supplying a robot to keep unchanged.

**What it licenses:** **nothing.** Not a bound, not a prompt edit, not a corpus, not a `V9` written
by an agent. It is reported to the owner as a finding against G0c's premise, and what to do about it
is a decision that gets its own document.

### 5.2 Outcome **N** — the generator invents nothing

*Definition, fixed here:* the probe reports **zero** candidate-invention frames across every unit,
with at least `MIN_PAIRED_PROBE_FRAMES` (16) paired frames per unit and at least two distinct
episodes and two distinct committed styles represented.

**What it establishes:** that on this corpus, at this cap, under these committed prompts, the
frozen generator did not put a manipulator into a robot-free frame. That is **evidence about the
premise, not a guarantee about 4.29 M frames**, and §7 bounds it: 726 frames is 0.017 % of the
partition, and absence over 726 frames does not bound a rate below roughly 1 in 240 at any
conventional confidence.

**What it licenses:** **nothing.** In particular it does **not** license the "both empty ⇒ accept"
rule, does not license a bound, and does not license generation. It removes one objection to a
decision that remains the owner's.

### 5.3 Outcome **U** — the probe cannot answer

*Definition, fixed here:* fewer than `MIN_PAIRED_PROBE_FRAMES` paired frames survive in a unit; or
the source-mask-empty rate on the selected frames departs grossly from the corpus rate the
diagnosis measured (59–86 % empty on robot-absent frames); or the generator refuses, pads, or
returns a clip whose frame count does not match its input.

**What it establishes:** that the instrument, not the generator, is what was measured. It is
reported as `U`, and **`U` is not a quiet `N`.** A run that produces `U` may be re-run once under
the same cap with the defect fixed and the fix recorded; it may not be re-read as evidence of
absence.

**No outcome — H, N or U — closes a `T40_RULE_V1` §8 item, moves a threshold, or licenses a clip.**

---

## 6. Why the answer cannot be got more cheaply, recorded so the cost is defensible

Three cheaper routes were considered and each fails for a stated reason:

- **Read it off the T-041 clips.** 60 paired Cosmos3-Super clips exist. They are a different
  generator, a different task and a different corpus, and `docs/handoff.md` §3 records that nobody
  on this project has looked at their frames deliberately. They cannot speak about Transfer2.5.
- **Reason from the prompts.** The clause `"the robot are unchanged"` makes hallucination
  *plausible*; it does not make it *measured*, and PR-08's whole discipline is that a plausible
  mechanism is not a finding. This is the same error as `T40_RULE_V1` §3's corrected premise, where
  *"can emit"* was hardened into *"already emits"*.
- **Wait for the timing run.** It is the licensed measurement and it would produce robot-free
  generated frames — but it refuses to start without the G0c bound, which is §1.2's loop. And even
  if it ran, its robot-free frames sit in a clip that is 56.5 % robot-present, so
  a manipulator appearing in them could be **temporal propagation from adjacent conditioning
  frames** rather than invention from nothing. **The probe's contiguous all-absent sub-clip is what
  separates those two, and that separation is the reason the probe is a different measurement from
  the timing run rather than a cheaper copy of it.**

---

## 7. Threats to validity — every one of them stated before the run

1. **The reference predicate is not ground truth, in both directions.** It *understates* robot
   presence (it scores none of the white wrist or bare-metal segments, so a wrist-only frame is
   called absent) and *over-calls* it at transitions (it scores the arm's moving shadow). An
   understated frame admitted into the probe would put a real robot into a "robot-free" clip, and a
   generated mask on it would be a **false candidate invention** — which biases the probe **toward
   outcome H**, i.e. toward the alarming answer, which is the safe direction for a premise check and
   is why the classification *band* (not a threshold) is used and why the sheets exist.
2. **The masker's own false positives are 41 % on robot-absent frames** (§2), and they are the
   apple, the plate and the cloth. On the *generated* side the apple has been restyled, so the false
   positive rate on generated frames is **not known to equal** the source rate the diagnosis
   measured. **The automated candidate count is therefore an UPPER BOUND on invention, never the
   finding.** The finding is a person reading the sheets. This is registered here so that a nonzero
   candidate count cannot later be reported as "the generator hallucinates a robot" without that
   reading having happened.
3. **The probe clip is a truncated, re-encoded sub-clip.** The selected contiguous run is written
   out with `robot_composite.encode_clip` (libx264, `-crf 10`), so the generator sees a re-encode of
   the H.264-lossless source rather than the source container. The re-encode is immaterial to
   whether a manipulator appears and is material to nothing else this probe measures, but it is a
   difference from the generation path and it is recorded rather than discovered.
4. **A short clip is out of distribution for a generator that will be run on whole episodes.** A
   121-frame conditioning window is not a 427-frame one, and the generator's behaviour may differ.
   The probe deliberately buys this cost to remove temporal propagation as a confound (§6); it does
   **not** claim its answer transfers unchanged to whole-episode generation, and §5's readings are
   written to be about the premise, not about the corpus.
5. **726 frames bound nothing about a rate.** An outcome `N` over 726 frames is consistent with an
   invention rate up to roughly 1 in 240 frames at conventional confidence — which, over 4.29 M
   frames, would still be thousands of affected frames. §5.2 says so; nothing may quote `N` as
   "the generator does not hallucinate".
6. **The `--control` choice moves the answer.** How much geometry survives is decided by the
   control block and weight, which `97` and the driver both refuse to default. A different control
   spec is a different measurement, and the probe records the one it ran. It is not swept.
7. **One generator revision.** The answer is about the staged pinned revision and no other. It is
   recorded with the artifact for the same reason `T40_RULE_V1` §6 requires the generator checkpoint
   id **and** revision with every verdict.

---

## 8. Determination — **SIGNED 2026-08-23**

**Proposed determination.**

> **`T40_RULE_V8` determines that a single diagnostic run of
> `cluster/discoverer/107_hallucination_probe.sbatch` is licensed, on at most 726 source frames
> selected because a non-learned reference predicate classifies them robot-absent, drawn from at
> most 3 episodes of `data/pr08-apple-640x480-h264-lossless` under at most 3 committed
> `TRAIN_STYLES` prompts at their committed seeds, through the frozen Cosmos-Transfer2.5 generator
> at its staged pinned revision, with the committed robot masker run on both the source frame and
> the generated frame and the pair recorded; and that the sole purpose and sole admissible reading
> of that run is whether the generator introduces a manipulator into a frame that does not contain
> one.**
>
> **This determination opens nothing else.** `T40_RULE_V1` §1's prohibition binds in full: **§8
> items 3 and 4 are open**, no corpus may be generated, no weight may be trained on generated
> frames, and no number from PR-08 — including any number from this probe — may be quoted as a
> result. It does not satisfy §8 item 3's throughput measurement, does not license a G0c area
> bound, does not license any edit to `robot_composite.check_mask`, and licenses no training run on
> any label space. Outcomes `H`, `N` and `U` are defined in §5 **before** the run, and **none of the
> three licenses generation.**

**Signature.** This document takes effect only when the line below is completed by the project
owner. **No agent may sign it, and no agent may act as though it were signed.**
`107_hallucination_probe.sbatch` reads this block and refuses, by rule name, while the owner line is
blank or no box is ticked.

**On transcription.** The project owner authorised this measurement in a Claude Code session on
**2026-08-23**, in the words *"Can Cosmos-Transfer2.5 hallucinate a manipulator into a robot-free
frame?"*, and the signature line below will be **transcribed** by that session, exactly as was done
for `T40_RULE_V4` §7. **A transcribed signature is the owner's decision and not an agent's:** the
deciding act is the owner's, the keystrokes are not. The account identity is the one this repository
holds for the owner; if the owner wants their own name on the line, replacing it is a one-word edit
and does not reopen the determination. **A session that fills this line in without the owner having
decided has forged it**, and the probe would then be running on an agent's authority — which is the
one thing this whole file exists to make impossible.

```
Project owner: huhn.dev@gmail.com                 Date: 2026-08-23

Determination:   [x] signed as proposed
                 [ ] signed with the amendments noted below
                 [ ] declined — T40_RULE_V8 is not in force, no frame may be generated under it,
                     and the circularity in §1.2 stands unbroken

Amendments / notes:

  (blank until signed)
```

---

## 9. Provenance

| | |
|---|---|
| rule | `T40_RULE_V8` |
| registered | 2026-08-23, **before any frame is generated**, before any weight is trained, before any job is submitted |
| status | **SIGNED 2026-08-23. IN FORCE**, on the narrow terms in §8 and subject to §4's six prohibitions. |
| supplements | `T40_RULE_V1` … `T40_RULE_V7` — all seven stand and all seven are **unedited** |
| supersedes | nothing |
| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style, no seed, no ceiling, no estimator setting, no consumer field, and no line of `scripts/robot_composite.py`** |
| licenses | one diagnostic run, ≤ 726 generated frames, §3 |
| forbids | §4, in six numbered items |
| leaves open | `T40_RULE_V1` §8 items **3 and 4**, the G0c area bound, and every decision `CLAUDE.md` reserves to the owner |
| generation of a corpus licensed | **no** |
| training licensed | **no** |
| §8 item 3 satisfiable by this | **no** — §4 item 3 |
| the job | `cluster/discoverer/107_hallucination_probe.sbatch` (refuses while §8 is unsigned) |
| the driver | `scripts/probe_hallucination.py` |
| the artifact | one quarantined run directory, `NOT_A_CORPUS`-marked, with overlay sheets; `human_review.looked_at` starts **false** |
| sources verified while drafting | `PR-08-photoreal-augmentation.md` §1 and §6 in full; `PR-08-V2-…` §0; `PR-08-V4-…` §0, §5, §7; `PR-08-V5/V6/V7` §0; `runs/pr08-robot-mask-empty/DIAGNOSIS.json` (every number in §2 read out of it); `scripts/diagnose_robot_mask_empty.py`; `scripts/robot_composite.py` (`check_mask`, `Sam2RobotMasker`, `build_masker`); `scripts/restyle_transfer25.py`; `scripts/assemble_restyled_lerobot.py`; `cluster/discoverer/97_transfer25_restyle.sbatch` and `106_measure_robot_mask_area.sbatch`; `configs/transfer25/pr08_style_partition.json`; `docs/transfer25-api.md` §1, §2, §8, §9; `docs/handoff.md` §3; `.mc/tasks/todo/T-040-…` 2026-08-22 entries |
| measurements taken here | **none.** V8 computes nothing and submits nothing. Every number in §2 is quoted from a committed artifact and was re-read from it while writing |
| decided by | the **project owner**, in session 2026-08-23; §8 signed and transcribed, with the transcription recorded there |
