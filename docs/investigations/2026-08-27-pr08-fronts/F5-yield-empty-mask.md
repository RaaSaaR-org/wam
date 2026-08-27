# FRONT 5 — the empty-mask half of G0c, and what replaces it

**Prepared 2026-08-27 by a Claude Code session on the user's workstation. Nothing in the repository
was modified. No cluster command was run. No rule was signed, edited, or discharged.**

Every claim below carries a label:

- **[M]** measured by me now, on this workstation, with the command shown;
- **[A]** recorded in a committed artifact or a committed document, quoted with its path;
- **[I]** my inference from [M]/[A];
- **[NOT MEASURED]** with the thing it would require.

---

## 0. The one-paragraph answer

V16 outcome **A** is dead on a measurement. Outcome **B** requires a number (`p_A`) that **does not
exist on disk**, so the evidence available today selects outcome **M** — a further version. That
version should not do what V12 §3.2 proposed. **It should not change `check_mask` at all.** The
whole failure is that `check_mask` is asked a question about frames that never should have been
sent to the generator: it runs *after* ~8 minutes of H200 time per clip has been spent [M, §5], on
masks computed **entirely from the source video** [M, §5], for a corpus whose per-frame source mask
areas have **already been measured at stride 1 over all 171 625 frames and committed** [A]. The
refusal set is therefore knowable for **zero GPU-hours today**, and I computed it below. The
successor semantics I propose changes the *work unit* (trim, and substitute source frames on the
frames that would refuse) and leaves G0c's "zero is zero" refusal byte-identical — where it becomes
an assertion that can never fire rather than a gate that always does. Yield: **402 of 402 episodes**
[M], at **307.0 GPU-h** instead of 442.4 for stage 1 [M], against today's **17 of 402** [M].

---

## 1. V16's outcomes B and M, quoted exactly

`docs/preregistration/PR-08-V16-empty-mask-graded-instrument.md` §6, rows **B** and **M** of the
decision table, verbatim, including emphasis [A]:

> | outcome | condition | what it licenses |
> |---|---|---|
> | **B** | `p_A` lower CI ≥ 0.33 | The masker is failing on frames with a plain arm in them. **V12 §3.3**: leave G0c alone and revisit `T40_RULE_V1` §3's compositing route. |
> | **M** | otherwise | Neither is licensed. A further version, which must say what it does about whichever of the two conditions failed. |

The quantities those conditions are stated over are fixed in the same section [A]:

> Let **`p_A`** be the population-weighted fraction of empty-mask frames graded **A**, Wilson 95 %
> per stratum combined in quadrature — V15 §5's estimator, over the `A` grade instead of `yes`. `D`
> is excluded from numerator and denominator; `B` and `C` are denominator, not numerator.

and the grade **A** the fraction is over is [A]:

> **A** — a **definite arm or hand**: unmistakable robot structure, more than an edge fragment

### 1.1 What each licenses, and what each requires as evidence

| | **outcome B** | **outcome M** |
|---|---|---|
| **condition** | `p_A` **lower** Wilson CI bound ≥ 0.33 | anything that is neither A's conjunction nor B |
| **evidence it requires** | The full V16 human instrument, run to completion: the 240 V15 tiles (`sample_seed = 40015`, `S1 60 / S2 60 / S3 40 / S4 40 / S5 40`), **re-judged from scratch** under the A/B/C/D vocabulary, with `D` under 25 % per stratum. V16 §7: "The 101 verdicts of the V15 run are **not** reinterpreted, not mapped onto A/B/C/D, and not used in any estimate." | **No new evidence at all.** M is the residual clause. It is selected by the *failure* of the other two conditions, and one of them has already failed on a measurement. |
| **what it licenses** | Nothing to be generated and nothing to be adopted. It licenses one **reading**: that the masker is failing on frames with a plain arm, and therefore that **V12 §3.3** — "change nothing, and accept the finding" — is the answer, with `T40_RULE_V1` §3's *compositing route* revisited rather than G0c's gate adjusted. | The writing of **a further version**, which "must say what it does about whichever of the two conditions failed." Nothing else. |
| **what it forbids** | Any relaxation of G0c. Under B the empty masks are the defect, not a nuisance. | Treating M as permission. V16 §8: "Adopts nothing, signs nothing, discharges nothing." |

**The asymmetry a reader should hold on to.** B is a *substantive* finding that closes the route.
M is *procedural*: it says the instrument did not settle it and hands the question to a successor
document with one obligation attached — name the failed condition and say what you do about it.
**This deliverable's §6 is a draft of exactly that document.**

---

## 2. Which outcome the evidence on disk selects

**Outcome M. [M] + [A]**

Three facts, in the order that decides it.

**2.1 Outcome A is unreachable, and this is a measurement, not a reading.** V16 §6 requires for A
both `p_A` upper CI ≤ 0.05 **and** `q99 ≤ 0.01`, where `q99` is the 99th percentile of `frac_dev`
over every empty-mask frame of the population. I recomputed it from the artifact [M]:

```
$ .venv/bin/python -c "... runs/pr08-empty-mask-look/MOTION.json ..."
n 57835 episodes 366
median 0.02490 p90 0.05155 p95 0.05710 p99 0.07180 max 0.12230
frac > 0.01: 0.8724301893317196  (50457 frames)
```

`q99 = 0.07180` against a threshold of `0.01` — over by a factor of 7.18. This reproduces
`PR-08-RESULT-2026-08-27-the-area-bound-cannot-separate-a-shadow-from-a-finger.md` §3 to five
decimal places [A/M]. That result's §6 states the consequence and refuses to repair it [A]:

> V16 §6 outcome **A** requires `p_A` upper CI ≤ 0.05 **and** `q99` ≤ 0.01. The second condition has
> now failed on a measurement, so **outcome A is unreachable.** The remaining outcomes are **B** …
> and **M** …

**2.2 Outcome B is not selected, because `p_A` does not exist.** `runs/pr08-empty-mask-look/` holds
`VERDICTS-partial-101.json`, 101 of 240 tiles, and its own `status` field reads [A]:

> `"status": "PARTIAL AND NOT EVALUATED under V15 §5 — see V16"`

Those verdicts are in V15's `yes`/`no`/`cannot_tell` vocabulary, not V16's A/B/C/D, and V16 §7
forbids mapping them across. **I have therefore computed no `p_A` from them and none exists
elsewhere in the tree.** (I did read the file. It contains 2 `yes`, 77 `no`, 22 `cannot_tell`. I
record that I looked so that a reader knows what I could have been tempted by, and I state that it
is inadmissible under V16 §7 and is used nowhere below. It is not `p_A` and cannot be converted
into one.)

**2.3 Therefore: `otherwise`.** A's conjunction is measurably false; B's condition is unevaluated
and cannot be true on evidence that does not exist. The residual clause fires. **The artifact that
selects it is `runs/pr08-empty-mask-look/MOTION.json` via
`PR-08-RESULT-2026-08-27-the-area-bound-cannot-separate-a-shadow-from-a-finger.md` §3 and §6.**

**One qualification, and it is the honest one [A].** M here is selected *by the absence of the
other half of the instrument*, and that same result document says the other half is still worth
running:

> **The human half is still worth running.** `p_A` — how often an empty mask hides a *definite* arm
> — is the number every route needs, it decides between **B** and **M**, and the reviewer's report
> says that class is the one they can judge.

So the selection is M **as of today**, and running V16's human half could move it to B. §6's draft
is written to survive that: it does not depend on `p_A` being small, and §3.3 explains why.

---

## 3. The successor semantics

### 3.0 What the gate is actually defending, stated as a failure set

Constraint (a) — "no generated robot pixel may enter the corpus uncomposited" — decomposes on an
empty-mask frame into **two** paths, and V12 already names both [A]:

- **P1, the masker missed a present robot** (V12 §1.4 case (b)). Source robot pixels existed, were
  not found, so the generated frame's robot survives. The composite would have fixed it and did not.
- **P2, the robot is genuinely absent and the generator invents one** (V12 §1.4, last paragraph,
  verbatim [A]): *"A frame with no source robot is unprotected against a generator that **invents**
  a robot. The composite cannot fix that in either direction — it has no pixels to write."*

**Every proposal that "passes" an empty-mask frame closes P1 at best and leaves P2 wide open.** That
includes V12 §3.2's frustum witness, which by construction answers only "was the arm in frame".
And P2 is not hypothetical on this generator: `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/PROBE.json`
returned **`"verdict": "H"`**, `candidate_frames: 56` of `paired_frames: 370` on frames selected as
robot-free [A]. Its own `human_review.looked_at` is `false` and V8 fixed in advance that the count
is an **upper bound**; the T-040 note reads most of it as the masker grounding the apple [A]. So
**P2 is unrefuted, not established** — which is precisely the state in which a gate must still
defend against it.

Constraint (b) — "must not refuse a clip because the robot is legitimately out of frame" — is the
9.19 %/90.81 % structure: 90.81 % of empty frames are contiguous runs at an episode's start or end
[M, reproduced in §4.1].

### 3.1 The candidate the brief names: a per-frame area bound on the CHANGED region (source vs generated)

**Rejected, on three independent grounds, any one of which is sufficient.**

**(i) It is not measurable on this workstation, today. [NOT MEASURED — requires a generation run.]**
`find /home/humanoid/develop/wam/runs -name "vision*.mp4"` returns nothing [M]. The only generated
frames this project has ever produced are job 189926's 384, and
`runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/{raw,probe_clips}/` are **both empty** on
this machine [M] — only the JSON and the contact sheets were brought back. So the distribution a
threshold would sit above cannot be read, and `T40_RULE_V1` §1 forbids producing it.

**(ii) It runs after generation, so it cannot touch this front's problem.** The changed region is
`|generated − source|`. It needs the generated clip. Front 5 is that the GPU cost is already paid
by then (§5). A bound on this statistic changes which clips are *kept*; it changes nothing about
which clips are *paid for*. Even in the best case it is a yield rule, not an economics rule.

**(iii) The statistic is degenerate for this experiment, and the committed sheets show it.** I
opened `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/sheets/episode_000000__train-01-oak-tungsten__probe__candidate_invention.png`
[M — I looked at the image]. Left half of each tile is the source: grey cloth wall, white plate, a
red-orange apple. Right half is the generated frame under `train-01-oak-tungsten`: warm oak table,
a **green** apple, different wall, different light. **There is no pixel of the frame that the
restyle leaves alone — that is what a restyle is.** So `frac_changed ≈ 1.0` on every frame of a
*working* arm B, whether or not a robot was invented. A threshold on it either fires on everything
or on nothing. (This is an observation of one committed sheet, not a number: I am deliberately not
computing a fraction from that tree, because `NOT_A_CORPUS` and V8 §4 restrict what numbers from it
may be quoted as.)

**(iv) And even repaired, it inherits the exact failure of 2026-08-27.** Suppose one narrowed the
statistic to "structurally changed" region rather than "changed". You are then back to separating a
zero-harm change (restyled cloth, restyled shadow) from a some-harm change (an invented finger).
That is the same conflation the area bound already died of [A]:

> **Area cannot separate a zero-harm shadow from a some-harm finger, and that is precisely the pair
> the human reader could not separate either.**

**Threshold, if one insisted anyway: NOT MEASURED, and I will not coin one.** V13 §3.3 is explicit
that a number placed on a continuum nobody has seen is "a coined threshold wearing a measurement's
clothes". There is no distribution here at all, so it would not even be that.

### 3.2 The alternatives, against both constraints

| # | rule | constraint (a): no uncomposited generated robot | constraint (b): don't refuse a legitimately-absent robot | costs GPU before it decides? |
|---|---|---|---|---|
| R0 | **today**: clip-fatal on any empty mask | **Held, absolutely** — by refusing everything | **Violated**: 366/402 refused, 17/402 survive both halves [M] | **Yes**, all of it |
| R1 | V12 §3.2: frustum witness, pass (a)/refuse (b) | **P1 only.** P2 open, and P2 is where verdict `H` lives | Held, if the witness is right | Yes (runs in the composite) | 
| R2 | Bounded area on the empty region (V16 §3) | **Not held** — a bound on a conflated statistic is not a guarantee | Held | Yes |
| R3 | Both-masks-empty test on the generated frame (T-040's option 1 [A]) | **P1 and P2, but only as well as the masker sees.** Grounding-DINO's best score on this scene is 0.12–0.149 against a 0.15 threshold [A] — the detector that misses is the same one being asked to catch | **Violated in practice**: the masker grounds the apple on **41 %** of robot-absent source frames corpus-wide [A], so this refuses on fruit | Yes |
| R4 | Refuse **frames**, not clips: split into pass-segments | **Held by the original construction** — every emitted frame has a non-empty in-bound source mask | Held | **No** — decidable from source | 
| R5 | Trim boundary runs + **identity fallback** on the residue, with an accounted budget | **Held, and strictly more strongly than R0** — see below | Held | **No** — decidable from source |

**Why the identity fallback is not a relaxation, which is the whole argument.** On a frame where the
source robot mask is empty, write the **source frame** into the output, verbatim. Then:

- no generated pixel of any kind enters the corpus on that frame, so *a fortiori* no generated robot
  pixel. **Constraint (a) is satisfied by construction, with no threshold and no number** — the same
  form of guarantee `T40_RULE_V1` §6 claims for the composite itself [A]: *"The defect cannot enter,
  and no threshold is needed."*
- **It is correct under BOTH branches of the undecidable disjunction.** If the robot was present and
  the masker missed it (case b / P1), the source frame contains the real arm — correct. If the robot
  was genuinely absent (case a), the source frame contains no arm and no hallucination — correct.
  **It therefore does not need the (a)/(b) split that V12 §2 made a precondition, that V15's
  instrument failed to produce, and that V16's bound could not replace.** That is the property that
  makes this a successor rather than a rationalisation: it is the one proposal whose correctness
  does not depend on the number nobody has been able to measure since 2026-08-24.
- it closes **P2** on exactly the frames where the composite provably cannot.

**Its cost is a different failure, and it is visible rather than silent:** the clip becomes part
restyled and part not-restyled. Un-restyled frames are literally arm A pixels inside an arm B clip,
which contaminates the very comparison `T40_RULE_V11` stage 2 turns on [A]: *"B does not beat C →
the gain is the generator's fingerprint."* **That is an experiment-validity harm, not a safety
harm**, and it is the thing the budget in §3.3 bounds. The distinction matters for `handoff.md` §3:
constraint (a) is discharged by construction and needs no number, so the only coined number left
guards interpretability, and it is coined in the open with its distribution stated first.

**And R4 is genuinely competitive** — it has zero identity frames and needs no budget at all. It
loses on yield and on temporal coverage (§4.3), and it multiplies one episode into several clips
with separate action slices. I report both.

### 3.3 The proposal, in one paragraph

> **Trim each source episode to `[first non-empty-mask frame … last non-empty-mask frame]`. Generate
> that span. On any frame inside the span whose source mask is empty *or* over the committed
> `max_frame_fraction`, write the source frame instead of the generated one. Refuse the episode only
> if the resulting identity fraction exceeds a budget fixed in advance. `check_mask` is not
> modified, not relaxed, and not given a threshold; it is still run on every frame the generator
> wrote, where it must never fire — and if it ever does, the plan and the run disagree and the clip
> is refused exactly as today.**

**Two properties worth naming.** First, **the gate is not edited after seeing its output** — which
is `handoff.md` §3's hazard and V12 §2's self-accusation. The gate's code and its semantics are
untouched; what changes is which frames are offered to it, and that change is justified by the
structure of the corpus (90.81 % boundary runs [M]), not by the refusal rate. The test V12 §2 named
for its own legitimacy applies here and passes [I]: **this proposal would be worth making if the
refusal rate were 3 %**, because trimming approach and retreat frames out of a restyle unit is
correct on its own terms — those frames contain no manipulator to restyle, they are the least
informative frames in the episode, and paying an H200 to restyle a picture of an empty tablecloth
is waste whatever the gate does with it afterwards.

Second, it is **decidable before generation**, from an artifact that already exists (§5).

### 3.4 The residual harm I am admitting, in full

1. **Un-restyled frames dilute arm B toward arm A.** Bounded and accounted (§4.2): median 2.01 % of
   a trimmed clip, p95 20.05 %, max 50.39 % [M]. **This is the reason the budget exists and the
   reason it is the owner's number, not a session's.**
2. **A false-positive mask still passes.** The masker returns a non-empty mask on 41 % of
   robot-absent source frames corpus-wide, and those masks are "the APPLE (~6-7 k px), the PLATE
   (~40-48 k px) or the [cloth]" [A, `runs/pr08-robot-mask-empty/DIAGNOSIS.json`
   `secondary_finding_false_positives`]. Such a frame is **not** identity-protected, so a robot
   invented elsewhere in it enters uncomposited. **This residual exists under R0 too** — those
   frames pass `check_mask` today. What is new is that the successor makes them *reachable*, because
   today their whole clip is refused for a different reason. **I state this as the single worst
   thing about the proposal.** It is P2 on non-empty-mask frames, it is V8's business, and V8's one
   run returned `H`.
3. **Cross-hardware instability of the plan.** `runs/pr08-bulk-stability/TAIL_SAMPLE.json` records
   that of 48 bulk-band frames re-rendered on the RTX 5090, **19 disagreed with the cluster beyond
   0.01 and 16 of those recomputed to exactly `0.0`** [M, I counted them from the artifact]. So
   "empty mask" is partly a property of the machine. A plan computed from the cluster's POOLED.json
   and executed on the cluster is self-consistent; **a plan computed on one machine and executed on
   another is not**, and the successor must bind the plan to the segmenter identity and the hardware
   the way `load_area_bound` already binds the bound. `PR-08-RESULT-2026-08-26-the-area-fraction-is-stable-except-in-the-band-nobody-uses.md`
   §5 makes exactly this point about the area bound [A].
4. **Trimming changes the corpus's temporal distribution.** Approach and retreat go. Arms A and C
   must be trimmed to the identical index range or the arms are not comparable — `T40_RULE_V2`
   requires C to match B's frame count [A]. **This is a corpus-wide decision with a scientific
   cost, and it is the second thing that belongs to the owner.**
5. **The action column must be sliced identically.** `scripts/assemble_restyled_lerobot.py` today
   copies the source parquet whole with only `episode_index` and `index` rewritten, and treats a
   frame-count mismatch as "a hard gate, not a warning" [A, module docstring lines 22-30]. Trimming
   requires slicing that parquet by the same `[i, j]`. Until that change lands, the assembler will
   refuse every trimmed clip — **correctly**.

### 3.5 Code changes owed — proposed as diffs, NOT APPLIED

Nothing under `/home/humanoid/develop/wam` was modified. These are unified diffs written here for
review. **They are unrun and untested**: the masker needs pinned weights and a GPU, and the corpus
lives on `/home/humanoid/wam-t041`, which this session did not touch.

#### (1) `scripts/robot_composite.py` — identity frames in `composite_clip`

`check_mask` is unchanged. The loop gains one branch.

```diff
--- a/scripts/robot_composite.py
+++ b/scripts/robot_composite.py
@@ def composite_clip(
     source_video: pathlib.Path,
     generated_video: pathlib.Path,
     context: CompositeContext,
     expected_frames: int | None = None,
+    identity_frames: frozenset[int] = frozenset(),
 ) -> dict:
@@
     fractions: list[float] = []
     ious: list[float] = []
     iou_frames: list[int] = []
     out = np.empty_like(gen)
     for index in range(src.shape[0]):
+        if index in identity_frames:
+            # T40_RULE_V20 §3: this frame's SOURCE robot mask is empty or over the bound, so the
+            # composite has nothing to write with. The source frame is written verbatim instead of
+            # the generated one. No generated pixel enters on this frame, so no generated ROBOT
+            # pixel can — which is stronger than what the composite guarantees elsewhere, and it is
+            # correct whether the robot was absent (V12 §1.4 case a) or present and missed (case b).
+            # check_mask is deliberately NOT called: there is no mask to check and nothing this
+            # frame could carry into the corpus.
+            out[index] = src[index]
+            fractions.append(float("nan"))
+            continue
         mask = masks[index]
         fractions.append(
             check_mask(mask, frame_index=index, bound=context.bound, source=str(generated_video))
         )
         out[index] = composite_frame(src[index], gen[index], mask)
```

and, in the returned record, next to `frames_composited`:

```diff
+        "identity_frames": sorted(int(i) for i in identity_frames),
+        "frames_identity": len(identity_frames),
+        "identity_fraction": len(identity_frames) / float(src.shape[0]),
+        "identity_rule": (
+            "T40_RULE_V20 §3 — the source frame was written verbatim on these indices because the "
+            "SOURCE robot mask was empty or over the committed bound. G0c's check_mask is "
+            "UNCHANGED and was not consulted on them."
+        ),
```

**Note the deliberate asymmetry:** `check_mask` is skipped on identity frames rather than made to
pass. Making it pass would require giving it a threshold, which is the one thing V12 §3.4 and V13
§3.4 both forbid. Skipping it is honest, because on those frames there is nothing for it to gate.

#### (2) new `scripts/plan_g0c_units.py` — the pre-flight, and the thing that saves the money

Reads the committed distribution and the committed bound; writes a **tracked** plan artifact. It
opens no video, imports no segmenter, and touches no GPU — same shape as
`robot_composite measure --merge`, which already runs on the free CPU QoS.

```python
#!/usr/bin/env python3
"""T40_RULE_V20 §3 — turn the committed SOURCE mask distribution into a work plan, before any GPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/plan_g0c_units.py \
        --pooled runs/pr08-robot-mask-area/POOLED.json \
        --bound  configs/transfer25/pr08_robot_mask_area.json \
        --budget 0.20 \
        --out    configs/transfer25/pr08_g0c_unit_plan.json

WHY THIS EXISTS. check_mask's inputs are a SOURCE mask and a committed bound; it never looks at a
generated pixel. restyle_transfer25.run_unit nevertheless calls it only AFTER the backend has run,
so today every refusal is discovered having already paid ~1.16 s of H200 time per frame. The
distribution this reads was measured at stride 1 over all 171 625 source frames with
measurement_qualified: true, so the refusal set is knowable for zero GPU-hours. This writes it down.

WHAT IT IS NOT. It sets no bound, coins nothing, and licenses nothing. --budget is a REQUIRED flag
with no default on purpose: the number is a decision about how much of arm B may be arm A, it is
the project owner's, and a default here would be a session making it silently.
"""
```

with the body computing, per episode:

```python
af    = episode["area_fractions"]
lead  = next((i for i, v in enumerate(af)          if v != 0.0), None)
trail = next((i for i in range(len(af) - 1, -1, -1) if af[i] != 0.0), None)
span  = af[lead : trail + 1]
ident = [i for i, v in enumerate(span) if v == 0.0 or v > bound.max_frame_fraction]
frac  = len(ident) / len(span)
verdict = "generate" if frac <= budget else "refuse_over_budget"
```

and stamping into the artifact `pooled_git_commit`, `source_manifest_sha256`,
`segmenter_identity(...)` and `max_frame_fraction` verbatim from their sources, so a plan measured
under one segmenter cannot be executed under another — the discipline `MaskCache.key` and
`load_area_bound` already enforce [A].

#### (3) `scripts/restyle_transfer25.py` — carry the plan, and refuse without it

`WorkUnit` gains `frame_start: int`, `frame_end: int`, `identity_frames: tuple[int, ...]`, read from
`work.jsonl`; `build_sample` hands the backend the trimmed span; `run_unit` forwards
`identity_frames=frozenset(unit.identity_frames)` into `composite.composite(...)`. **`main` must
refuse outright when the work list carries no plan**, in the same place and the same style as the
`build_context` refusal at `scripts/restyle_transfer25.py:711` — a driver that silently falls back
to whole-episode units would restore the 442 GPU-h failure with no message.

#### (4) `scripts/assemble_restyled_lerobot.py` — slice the parquet

The hard frame-count gate stays a hard gate; its right-hand side becomes `frame_end - frame_start +
1`, and the parquet is sliced on the same range before the two bookkeeping columns are rewritten.
**Not sketched here** — GR00T addresses video by frame index (module docstring lines 26-30 [A]), so
an off-by-one is a shifted control law, and that hunk deserves its own tests rather than a sketch in
a report.

---

## 4. Yield, measured from the committed artifacts

All figures below are **[M]**, computed by me now from
`runs/pr08-robot-mask-area/POOLED.json` (`git_commit 8b710d0119b65fd3c4eff0e968e3d92edc84d2ae`,
`source_manifest_sha256 a988dd60db6b…`, `measurement_qualified: true`, 402 episodes / 171 625 frames
at stride 1) and `configs/transfer25/pr08_robot_mask_area.json`
(`max_frame_fraction = 0.64091145833333329`).

### 4.1 The baseline reproduces, exactly

```
episodes with >= 1 empty-mask frame  366  (91.04 %)      <- refused by the empty half
episodes with >= 1 over-bound frame  175  (43.53 %)      <- refused by the area half
episodes refused by EITHER           385  (95.77 %)
episodes surviving BOTH halves        17  ( 4.23 %)
frames 171 625; empty 57 835 (33.6985 %); over-bound 1 385 (0.8070 %)
```

`366` / `175` / `385` match the brief and the committed documents [A]. **The number that is not on
record anywhere I could find is `17`** — every document quotes one half or the other. With both
halves of `check_mask` armed, as `configs/transfer25/pr08_robot_mask_area.json` armed them on
2026-08-26, **the corpus yield today is 17 episodes, not 36.**

The V15 strata reproduce to the frame [M]:

```
lead runs   309, 15 888 frames   (27.47 %)
trail runs  344, 36 634 frames   (63.34 %)
interior:   len<=2   543 runs,   764 frames
            len 3-25 309 runs, 2 226 frames
            len>=26   49 runs, 2 323 frames
boundary share 90.8135 %
```

identical to `T40_RULE_V15` §2's table [A]. Episodes: **36** with no empty frame, **84** whose empty
frames are *only* boundary runs, **282** with at least one interior run.

### 4.2 The proposal, R5 (trim + identity fallback + budget)

Per episode: trim to `[first non-zero … last non-zero]`; identity-substitute every frame in that
span that is `0.0` or `> 0.64091145833333329`.

```
trimmed clip length      min 156   median 276   max 749
kept frames              119 103   (69.40 % of the corpus)
identity frames in them    6 698   (5.62 % of kept)   = 5 313 interior zeros + 1 385 over-bound
per-episode identity fraction  median 0.0201  p75 0.0580  p90 0.1442  p95 0.2005  max 0.5039
episodes with ZERO identity frames after trimming: 66
```

**Every one of the 402 episodes retains a contiguous span of at least 156 frames.** No episode is
lost to trimming. What loses episodes is the budget, and only the budget:

| identity budget | episodes | stage-1 clips (×8) | clip frames | stage-1 GPU-h @1.16 s/frame | identity frames in clips |
|---:|---:|---:|---:|---:|---:|
| ≤ 0.05 | 286 (71.1 %) | 2 288 | 82 410 | **212.4** | 1 099 (1.33 %) |
| ≤ 0.10 | 333 (82.8 %) | 2 664 | 96 157 | **247.9** | 2 075 (2.16 %) |
| ≤ 0.15 | 366 (91.0 %) | 2 928 | 106 821 | **275.4** | 3 419 (3.20 %) |
| ≤ 0.20 | 381 (94.8 %) | 3 048 | 111 760 | **288.1** | 4 325 (3.87 %) |
| ≤ 0.30 | 393 (97.8 %) | 3 144 | 115 494 | **297.7** | 5 232 (4.53 %) |
| no budget | **402 (100 %)** | 3 216 | 119 103 | **307.0** | 6 698 (5.62 %) |
| **today (R0)** | **17 (4.2 %)** | 136 succeed of 3 216 | 171 625 | **442.4** | — |

**The `442.4` in the last row is my own arithmetic and it lands on the committed `~442 GPU-h`
exactly** — `171 625 frames × 8 instances × 1.16 s ÷ 3600 = 442.4` [M] against T-040's *"3 216 clips,
**~442 GPU-h (8.8 %)**"* [A]. That agreement is what tells me I have stage 1's shape right: 8
style-instances × all 402 episodes × every frame.

So the proposal at no budget: **402 episodes instead of 17, for 307.0 GPU-h instead of 442.4.**
Yield per GPU-hour improves by a factor of **34** [I, from the two rows].

**The 1.16 s/frame figure is not a budget line and I am not making it one.** T-040 says so in as
many words [A]: *"THE 1.16 s/frame FIGURE IS NOT §8 ITEM 3's MEASUREMENT AND MAY NOT BE A BUDGET
LINE."* Every GPU-h column above is therefore a **ratio dressed as an absolute**: what it actually
establishes is `307.0 / 442.4 = 0.694`, i.e. the proposal generates **30.6 % fewer frames**, and
that ratio is independent of the per-frame cost.

### 4.3 The alternative, R4 (split into pure pass-segments, no identity frames)

Segments where `0 < area_fraction ≤ bound`, contiguous:

| minimum segment length | segments | frames | share of corpus | episodes contributing |
|---:|---:|---:|---:|---:|
| ≥ 48 | 496 | 103 453 | 60.3 % | 402 / 402 |
| ≥ 96 | 431 | 98 661 | 57.5 % | 392 / 402 |
| ≥ 121 | 384 | 93 624 | 54.6 % | 367 / 402 |
| ≥ 150 | 361 | 90 517 | 52.7 % | 353 / 402 |
| ≥ 200 | 294 | 78 466 | 45.7 % | 294 / 402 |

**R4 is viable and it needs no budget and no coined number at all** — its only free parameter is the
minimum clip length, which is a generator fact rather than a judgement. At ≥ 96 frames it keeps
392 of 402 episodes and 98 661 frames. Its costs against R5: it produces **431 clips from 402
episodes** (so the action column must be sliced 431 ways, and one episode contributes several
correlated clips to a training set), and it discards the 6 698 frames R5 would have kept as source.

**The minimum-length column is where I have the least evidence.** The only datum is T-040's
*"96 frames in ~111 s … `Average time per chunk: 55.47`, two chunks"* [A] → **48 frames per chunk
[I]**, and Cosmos-Transfer2.5's actual minimum clip length is **NOT MEASURED — requires reading
upstream's inference config, which is not vendored in this tree** (`grep -n chunk
docs/transfer25-api.md` returns one line, about losing a chunk, not about its size [M]).

---

## 5. Does `restyle_transfer25.py` really pay the full GPU cost before `check_mask` runs?

**Yes. Unambiguously, and the order is explicit in the code. [M — read at the paths below]**

`scripts/restyle_transfer25.py:511-526`, `run_unit`, in order:

```python
        extra = (
            _null_backend(sample, out_dir)
            if backend == "null"
            else _transfer25_backend(sample, out_dir, setup)   # <-- the whole generation
        )
        video = out_dir / "vision.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise DriverError(...)
        extra["g0c"] = composite.composite(                     # <-- G0c starts here
            source_video=pathlib.Path(sample["video_path"]),
            generated_video=video,
            expected_frames=unit.frames,
        )
```

Its own docstring says so: *"G0c SITS BETWEEN THE BACKEND AND THE STATUS"* [A].

**And the check that fires is computed entirely from the source.** In `composite_clip`
(`scripts/robot_composite.py:1578-1643`) the masks come from `source_masks(source_video, src,
context)` — `src`, never `gen`. `check_mask`'s signature confirms it [M, run just now]:

```
$ PYTHONPATH=src:scripts .venv/bin/python -c "... inspect.signature(rc.check_mask) ..."
signature: (mask: 'np.ndarray', *, frame_index: 'int', bound: 'AreaBound', source: 'str') -> 'float'
EMPTY     -> SRC: the robot mask is EMPTY on frame 7.
OVERBOUND -> SRC: the robot mask covers 0.8333 of frame 7, above the committed bound 0.6409…
```

**There is no generated pixel anywhere in the decision.** The only thing in `composite_clip` that
needs the generated clip before the loop is the `gen.shape != src.shape` guard.

### 5.1 So the whole economic problem changes shape, and the artifact to do it with already exists

Three consequences [I, from the [M] facts above]:

1. **A source-side pre-flight is not a new instrument.** `robot_composite measure` **is** that
   instrument — it "measure[s] the robot-mask area distribution over the SOURCE corpus (sets no
   bound)" [A, `build_parser`] — and it has already been run to completion over the whole corpus at
   stride 1. `runs/pr08-robot-mask-area/POOLED.json` **is** the pre-flight result. §4 is me reading
   the refusal set out of it for zero GPU-hours.
2. **The cost of computing it is ~2 orders of magnitude below the cost of discovering it the
   current way.** Job 106's recipe is 16 shards at `--time=01:30:00` [A,
   `cluster/discoverer/106_measure_robot_mask_area.sbatch:16-19`] → **≤ 24 GPU-h as a requested-
   walltime ceiling** (not a measurement of used time) against stage 1's 442.4.
3. **Today the same refusal is re-discovered 8 times per episode at full price.** Stage 1 is 8
   style-instances over the same 402 source episodes; nothing in `main`'s loop
   (`scripts/restyle_transfer25.py:742-765`) memoises "this episode's source refuses" [M — I read
   the loop; there is no such branch]. The `MaskCache` makes the *second* instance's mask
   computation free, but only after that instance's generation has already run. So of the 442.4
   GPU-h, on today's numbers, **423.7 buy clips that are then quarantined** [M: `442.4 × (1 −
   17/402)`].

**One caveat that belongs to the pre-flight and not to the current code.** POOLED.json lives under
`runs/`, which is **gitignored** [M, `.gitignore`], so it is not the pre-commitment a rule can point
at. The plan artifact in §3.5(2) must be written under `configs/` for exactly the reason
`AREA_BOUND_ARTIFACT` is [A, `scripts/robot_composite.py:374-377`].

---

## 6. DRAFT successor rule — `T40_RULE_V20`

**This is a draft in a scratch directory. It is not in `docs/preregistration/`, it has not been
committed, and it is UNSIGNED. It exists to be read and refused or filled in by a person.**

---

### PR-08 V20 — G0c stops being asked about frames that were never worth generating

**Rule `T40_RULE_V20`. DRAFTED 2026-08-27. UNSIGNED — see §6. Nothing here is in force until the
project owner signs it, and no number produced under it may be quoted before that.**

Sits alongside `PR-08-photoreal-augmentation.md` (`T40_RULE_V1`), which is **not edited**.
`docs/handoff.md` §3.

#### §1. Why `T40_RULE_V12` is superseded

V12 proposed to change how `robot_composite.check_mask` **adjudicates** an empty robot mask, and
made its own case against itself: it was drafted after seeing the gate refuse most of the corpus,
and V12 §2 names the measurement that would tell a repair from a rationalisation. **That
measurement has now been attempted twice and has not been produced.**

- `T40_RULE_V15` put the (a)/(b) question to a person and **stopped itself** on its own 25 %
  undecidability cap: `S3_int_1_2` returned 70.6 % `cannot_tell` and `S4_int_3_25` 33.3 %, so §5 was
  never evaluated and no split was computed.
- `T40_RULE_V16` reframed the question from a classification to an area bound and fixed
  `q99 ≤ 0.01` before looking. The measurement returned **`q99 = 0.0718`**, over by a factor of
  seven, and the 2026-08-27 result states why the route closes on itself: *"Area cannot separate a
  zero-harm shadow from a some-harm finger, and that is precisely the pair the human reader could
  not separate either."*
- **V16 §6's outcome A is therefore unreachable, and outcome B is unevaluated because `p_A` does not
  exist.** The residual clause fires: **outcome M**, whose whole content is that a further version
  is required and *"must say what it does about whichever of the two conditions failed."*

**This is that version, and here is what it does about the failed condition.** It does not repair
`q99`, does not loosen V16 §6's conjunction, and does not separate its two clauses. **It withdraws
the question.** V12 asked how G0c should adjudicate an empty mask. V20's answer is that G0c should
never be shown one — because a frame whose source contains no robot is a frame the compositing route
cannot protect *in either direction* (V12 §1.4's own last paragraph), and paying an H200 to restyle
it and then refusing the clip over it is the worst available combination of the two.

**V20 changes no gate.** `check_mask` is not modified. `GATE_QUALIFIED`,
`GATE_QUALIFICATION_BLOCKERS`, `ROBOT_MASK_OBJECT_MAX_IOU`, `MASK_VALIDITY_REFERENCE_*`, `GEOM_TOL`,
`EST_DRIFT_P95`, `max_frame_fraction` and every seed and budget are untouched. `T40_RULE_V12` is
superseded as a **proposal**, not overruled as a finding: its §1.4 distinction stands and its §3.2
route remains available to a later version if the camera geometry it needs is ever recovered.

#### §2. The measured position, as of 2026-08-27

All from `runs/pr08-robot-mask-area/POOLED.json` (`git_commit 8b710d01…`,
`source_manifest_sha256 a988dd60…`, `measurement_qualified: true`, 402 episodes, 171 625 frames,
stride 1) and `configs/transfer25/pr08_robot_mask_area.json`
(`max_frame_fraction = 0.64091145833333329`, signed under `T40_RULE_V13` §5, 2026-08-26).

| | |
|---|---:|
| episodes refused by the empty half | **366 / 402 (91.04 %)** |
| episodes refused by the area half | **175 / 402 (43.53 %)** |
| episodes refused by either | **385 / 402 (95.77 %)** |
| **episodes surviving both halves** | **17 / 402 (4.23 %)** |
| empty frames | 57 835 / 171 625 (33.70 %) |
| of which contiguous runs at an episode boundary | **52 522 (90.81 %)** |
| interior empty frames | 5 313 (3.09 % of the corpus) |
| over-bound frames | 1 385 (0.81 %) |

**And four facts about what an empty mask is, none of them settled:**

1. **Zero confirmed detector failures on robot-present frames.** `runs/pr08-robot-mask-empty/DIAGNOSIS.json`:
   *"0 of 917 (pilot) and 0 of 240 (corpus, after inspecting all 19 apparent disagreements)
   confidently-robot-present frames returned an empty mask."*
2. **But the interval is wide.** The blind adjudication's only unbiased arm decided **13 of 40**
   tiles, 27 undecidable; `b_rate_ci95_wilson` upper bound **0.228**, and
   **0.675** counting every undecidable as a failure (`runs/pr08-blind-adjudication/BLIND_SCORE.json`).
   The honest reading is *unrefuted*, not *established*.
3. **The masker false-positives heavily in the other direction.** On robot-**absent** frames it
   returns a non-empty mask 14 % of the time on the pilot and **41 % corpus-wide**, grounding the
   apple, the plate, or the cloth (`DIAGNOSIS.json` `secondary_finding_false_positives`).
4. **The generator's invention rate is unmeasured and its one probe returned `H`.**
   `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/PROBE.json`: `verdict H`,
   `candidate_frames 56` of `paired_frames 370`, `human_review.looked_at false`, and V8 fixed in
   advance that the count is an upper bound.

**Fact 4 is why this version does not adopt V12 §3.2.** A frustum witness answers *"was the arm in
frame"*. It does not answer *"did the generator draw one"*, and on an empty-mask frame the composite
has no pixels to write either way.

#### §3. What V20 proposes

**§3.1 The unit is trimmed at the source, before generation.** Each source episode's work unit
becomes the closed index range `[first frame with a non-empty source robot mask … last such
frame]`. Frames outside that range are not restyled, not composited, and not delivered.

**§3.2 Inside the range, a failing frame is written from the source.** For any frame in the range
whose source robot mask is empty **or** covers more than the committed `max_frame_fraction`, the
**source frame is written into the output verbatim**. No generated pixel enters the corpus on that
frame.

**§3.3 `check_mask` is unchanged and still runs on every generated frame.** It is not given a
threshold, a tolerance, a budget or an exemption. Under a correct plan it can never fire; **if it
fires, the plan and the run disagree about the corpus or about the segmenter, and the clip is
refused exactly as it is today.** It stops being the gate that decides the corpus and becomes the
assertion that the plan was honest.

**§3.4 The plan is an artifact, computed with no GPU, committed under `configs/`.**
`scripts/plan_g0c_units.py` reads a `measurement_qualified: true` distribution and the committed
bound and writes per-episode `[start, end]`, the identity indices, and the identity fraction,
stamped with `pooled_git_commit`, `source_manifest_sha256` and `segmenter_identity(...)`. A run
whose segmenter identity differs from the plan's **refuses**, in the manner `load_area_bound` and
`MaskCache.key` already establish.

**§3.5 Why this is not a gate rewritten after seeing its output.** V12 §2 states the test in its own
words — the justification must hold *"even if the refusal rate were 3 % instead of 99.2 %"*. It
does. Trimming a restyle unit to the frames that contain a manipulator is correct because those are
the frames a manipulator-preserving restyle is *about*; the approach and retreat frames contain a
tablecloth. **The refusal rate is not an argument in §3 and is not needed by it.** What the refusal
rate establishes is only that the current arrangement is expensive, which is §4's business.

**§3.6 What it costs, stated against itself.** A trimmed clip with identity frames is **part arm A**.
Un-restyled frames dilute exactly the comparison `T40_RULE_V11` stage 2 turns on. That harm is
bounded by §4's budget, and **the budget is the one number this rule needs a person for.**

#### §4. Outcomes, FIXED IN ADVANCE

Let **`ι(e)`** be episode `e`'s identity fraction under §3 — the number of frames in its trimmed
range that are empty or over-bound, divided by the range's length. Let **`β`** be the identity
budget, a single number in `[0, 1]`.

**`β` is coined, not measured, and this document says so rather than dressing it as derived.**
Unlike `T40_RULE_V13`'s `max_frame_fraction`, there is no gap in a distribution to read it off:
`ι` is a smooth quantity (median 0.0201, p90 0.1442, max 0.5039) with no bimodality, and V13 §3.3
is explicit that a number placed on a continuum *"is a coined threshold wearing a measurement's
clothes"*. **The difference from V13, and the reason a coined number is admissible here, is what it
guards: `β` guards the interpretability of arm B, not the integrity of the corpus.** Constraint (a)
is discharged by §3.2's construction and needs no number at all. A `β` set wrongly produces a
weaker experiment, not a contaminated one.

**The distribution of `ι` is stated before `β` is chosen, and is in §2's table and here:**
`median 0.0201 · p75 0.0580 · p90 0.1442 · p95 0.2005 · max 0.5039`, 66 episodes at exactly zero.
**A signer choosing `β` is therefore choosing with the distribution in front of them, which is the
opposite of V13's ordering, and this document flags it as such rather than hiding it.** The
justification: `β` is not protecting against a defect the corpus might contain, so there is nothing
for foreknowledge of the distribution to corrupt. A reader who rejects that argument should refuse
this rule and require `β` to be fixed from a downstream requirement instead — for instance, from
the maximum arm-A contamination under which `T40_RULE_V11` stage 2's B-vs-A comparison retains its
declared power, which **has not been computed and would be the better route if anyone will pay for
it.**

| outcome | condition | what it licenses |
|---|---|---|
| **P** | The plan artifact validates: `measurement_qualified: true`, segmenter identity matches the run's, and every episode's trimmed range is non-empty | The plan may be **built and reviewed**. Not generated from. |
| **Y** | Under the chosen `β`, at least **302** of 402 episodes (¾) carry `ι ≤ β` | The trimmed partition is worth its machinery. Whether generation starts is still `T40_RULE_V1` §1's and the owner's. |
| **M2** | 101 ≤ surviving episodes < 302 | A corpus survives but a quarter to three quarters is lost to the budget. Whether that is worth it is an owner decision, not a session's. |
| **N** | fewer than 101 episodes survive | The budget and the corpus are incompatible. Either `β` is wrong or V12 §3.3 was right after all, and this rule does not choose between those. |

`302` and `101` are `T40_RULE_V15` §5's Q2 thresholds, reused **unchanged and deliberately**, so
that this version cannot be accused of moving a bar it has already seen.

**Measured, at the time of drafting, against `β = 0.20`: 381 of 402 (94.8 %) → outcome Y.** At
`β = 0.05`: 286 → still Y. At every `β ≥ 0.02`: Y. **This rule therefore returns Y for any budget a
person is plausibly going to pick, and that is a weakness a signer should weigh: a rule whose
outcome is insensitive to its own free parameter is not testing much.** It is stated here rather
than discovered later.

#### §5. What would refute this rule

Any one of these, and each is checkable:

1. **`check_mask` fires during a run executed under a validated plan.** That is the plan claiming
   the corpus is something it is not. It refutes §3.3's central claim and the run must stop, not
   continue with the frame skipped. **The most likely cause is already on record**: cross-hardware
   mask instability — 19 of 48 bulk-band frames disagreed between the cluster and the RTX 5090
   beyond 0.01, and 16 of those recomputed to exactly `0.0`
   (`runs/pr08-bulk-stability/TAIL_SAMPLE.json`). A plan computed on one machine and executed on
   another is refuted by construction, and §3.4's segmenter stamp does **not** capture the GPU.
2. **V16's human half returns outcome B** — `p_A` lower CI ≥ 0.33, a definite arm in a third of
   empty masks. That does not refute §3.2 (writing the source frame is correct when the arm is
   present too) but it **does** refute §2's reading and would make trimming a loss of manipulator
   frames rather than of tablecloth. §3.1's trim would then be discarding real data and V12 §3.3's
   reading — the compositing route does not work here — becomes the honest one.
3. **A person looks at the trimmed boundaries and finds the arm in them.** The trim's whole premise
   is that the boundary runs are approach and retreat. A tile sheet of the frames immediately
   inside and outside each trim point is cheap, is masker-independent if rendered without overlays,
   and has not been made.
4. **V8's hallucination probe, reviewed by a person, returns a non-trivial invention rate on
   frames whose source mask is NON-empty.** Those frames are not identity-protected under §3.2, so
   an invented manipulator enters uncomposited there. This rule does not defend that case and does
   not claim to; a measured rate above zero on it would mean G0c's compositing route needs a second
   instrument, and §3 would not be it.
5. **The trimmed corpus fails `assemble_restyled_lerobot.py`'s frame-count gate** in a way that
   cannot be repaired by slicing the parquet on the same range. GR00T addresses video by frame
   index; a mis-sliced action column is a shifted control law and is silent.

#### §6. Determination

**Decided by: nobody. UNSIGNED, and a session may not sign it.**

A Claude Code session drafted this. It may not sign it, may not treat it as a licence, and may not
act as though `β` had been chosen. Two things are the project owner's and are named so they cannot
be taken silently: **`β`**, and **whether arms A and C are trimmed to the same index ranges** —
`T40_RULE_V2` requires C to match B's frame count, so trimming B without trimming C breaks the
comparison, and trimming all three changes what the corpus is.

```
determination:  ____________________
beta:           ____________________
arms A and C trimmed to the same ranges:  ____________________
decided by:     nobody yet
date:           ____________________
```

Nothing in this document licenses generation, training, or any statement of a result.
`T40_RULE_V1` §1's prohibition is untouched and binds in full. `GATE_QUALIFIED` stays `False` and
`GATE_QUALIFICATION_BLOCKERS` is not shortened by it.

#### §7. Provenance

| | |
|---|---|
| rule | `T40_RULE_V20` |
| status | **UNSIGNED DRAFT, in a scratch directory. Not in `docs/preregistration/`. Not in force** |
| drafted | 2026-08-27, after V16 §6 outcome A was measured unreachable |
| supersedes | `T40_RULE_V12` as a **proposal**. V12's §1.4 finding stands; its §3.2 route stays available |
| would change | `scripts/plan_g0c_units.py` (new), `composite_clip`'s frame loop, `restyle_transfer25`'s work unit, `assemble_restyled_lerobot`'s parquet slice |
| would NOT change | `check_mask`, `GATE_QUALIFIED`, `GATE_QUALIFICATION_BLOCKERS`, `max_frame_fraction`, `ROBOT_MASK_OBJECT_MAX_IOU`, `ROBOT_TEXT_PROMPT`, any seed, any budget, any signed rule document, `T40_RULE_V1` §1 |
| coined numbers | **one: `β`.** §4 states what it guards, why it is admissible where V13's was not, and what would be a better route |
| evidence | `runs/pr08-robot-mask-area/POOLED.json`; `configs/transfer25/pr08_robot_mask_area.json`; `runs/pr08-empty-mask-look/MOTION.json`; `runs/pr08-robot-mask-empty/DIAGNOSIS.json`; `runs/pr08-blind-adjudication/BLIND_SCORE.json`; `runs/pr08-bulk-stability/TAIL_SAMPLE.json`; `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/PROBE.json` |
| jobs submitted | **none** |
| generation licensed | **no** |
| training licensed | **no** |

---

## 7. Ledger — what I measured, what I read, what I inferred

**[M] measured by me now, on this workstation:**

- 366 / 175 / 385 / **17** of 402, and 57 835 / 1 385 / 171 625 frames, from POOLED.json.
- The V15 strata, to the frame: 15 888 / 36 634 / 764 / 2 226 / 2 323, boundary share 90.8135 %.
- `frac_dev` quantiles over 57 835 empty frames: median 0.02490, p99 **0.07180**, max 0.12230;
  87.243 % above 0.01. Reproduces the 2026-08-27 result exactly.
- R5 yields: 119 103 kept frames, 6 698 identity, `ι` median 0.0201 / p95 0.2005 / max 0.5039, 66
  episodes at zero, every episode retaining ≥ 156 frames; the budget table in §4.2.
- R4 yields: the segment table in §4.3.
- `442.4` GPU-h reconstructed from 171 625 × 8 × 1.16 s, matching T-040's committed `~442`.
- `check_mask`'s signature and both refusal messages, by executing it against synthetic masks.
- `run_unit`'s ordering, `composite_clip`'s source-only mask path, the absence of any per-episode
  refusal memo in `main`'s loop — all read directly.
- 16 of 48 bulk frames recomputing to exactly `0.0` across hardware, counted from `TAIL_SAMPLE.json`.
- `runs/` is gitignored; POOLED.json is untracked.
- The `candidate_invention` contact sheet, looked at: the restyle changes every pixel of the frame.

**[A] quoted from committed artifacts/documents:** V16 §6's outcome rows and grade definitions; V16
§7's non-carry-forward; the 2026-08-27 result §3/§5/§6; V15 §2/§4/§5; V12 §1.4/§2/§3.2/§3.3/§3.4;
V13 §3.3/§3.4/§4/§5; `DIAGNOSIS.json`'s q4 and false-positive headlines; `BLIND_SCORE.json`'s
unbiased arm; `PROBE.json`'s verdict `H`; T-040's stage-1 partition and the 1.16 s/frame caveat;
`106_measure_robot_mask_area.sbatch`'s shard recipe.

**[I] inference, labelled as such:** that the identity fallback is correct under both branches of the
(a)/(b) disjunction; that 48 frames/chunk follows from "96 frames, two chunks"; that 423.7 of 442.4
GPU-h currently buy quarantined clips; that a trimmed-unit rule passes V12 §2's "would it hold at
3 %" test.

**[NOT MEASURED]:**

- `p_A`, V16's deciding quantity — **requires the human half of V16's instrument** (240 tiles
  re-judged under A/B/C/D). Nothing on disk substitutes for it.
- The changed-region statistic of §3.1 — **requires a generation run**, which `T40_RULE_V1` §1
  forbids, and whose only prior output is not on this machine.
- Cosmos-Transfer2.5's minimum clip length — **requires upstream's inference config**, not vendored.
- Whether the trimmed boundaries actually contain no arm — **requires a person looking at a tile
  sheet that does not exist** (§5 item 3 of the draft rule).
- The arm-A contamination `T40_RULE_V11` stage 2 can tolerate — **requires a power calculation
  against `docs/comparison-power-analysis.md`, which I did not do.**
- Any statement about GR00T. PR-07 §6 forbids it and nothing here touches it.

---

# Adversarial re-read

**Prepared 2026-08-27 by a second Claude Code session, tasked to REFUTE the above. Nothing in
`/home/humanoid/develop/wam` was modified. No cluster command was run. No rule was signed.**

Same labels: **[M]** measured now, **[A]** committed artifact quoted with its path, **[I]** inference.

**Verdict: the deliverable does not survive.** Its *arithmetic* is clean — I reproduced §4.1, §4.2
and §4.3 to the last frame, and the `442.4`/`307.0`/`34×` figures all check out [M, §A.0 below].
What fails is the *proposal*. Four of its load-bearing claims are refuted against the repo, two
"read from the artifact" counts do not match the artifact, and the successor rule would break a
committed harvest contract that its own §7 "would NOT change" table does not mention.

---

## A.0 What reproduces (stated first, so the refutations are not read as a general attack)

Re-run independently from `runs/pr08-robot-mask-area/POOLED.json` and
`runs/pr08-empty-mask-look/MOTION.json` [M]:

| claim | deliverable | my re-run | |
|---|---|---|---|
| empty / area / either / survive, of 402 | 366 / 175 / 385 / **17** | 366 / 175 / 385 / **17** | ✅ |
| frames, empty, over-bound | 171 625 / 57 835 / 1 385 | identical | ✅ |
| V15 strata (lead/trail/S3/S4/S5) | 15 888 / 36 634 / 764 / 2 226 / 2 323 | identical, and matches `PR-08-V15…md:44-48` [A] | ✅ |
| `frac_dev` q99 | 0.07180 (87.243 % > 0.01) | 0.0717958 (87.2430 %) | ✅ |
| R5: kept / identity / min span / ι quantiles / 66 zeros | 119 103 / 6 698 / 156 / .0201·.0580·.1442·.2005·.5039 | identical | ✅ |
| R5 budget table (all six rows) | 286/333/366/381/393/402, GPU-h 212.4…307.0 | identical | ✅ |
| R4 segment table (all five rows) | 496/431/384/361/294 | identical | ✅ |
| `171625 × 8 × 1.16 / 3600` | 442.4 | 442.411 | ✅ |
| V16 §6 rows **B** and **M** | quoted verbatim | matches `PR-08-V16…md:123-124` character for character | ✅ |
| `runs/` gitignored, POOLED.json untracked | yes | `.gitignore:19:runs/`; `git ls-files --error-unmatch` fails | ✅ |
| `composite_clip` masks come from `src` only | yes | `robot_composite.py:1620` `source_masks(source_video, src, context)` | ✅ |
| backend runs before G0c | yes | `restyle_transfer25.py:515` then `:523` | ✅ |
| no per-episode refusal memo in `main`'s loop | yes | `:742-765`, no such branch | ✅ |
| DIAGNOSIS q4 headline, 41 % false-positive rate | as quoted | verbatim; corpus absent 98/240 = 40.8 % | ✅ |
| `PROBE.json` verdict `H`, 56 of 370 | as quoted | `/totals/candidate_frames 56`, `/totals/paired_frames 370` | ✅ |
| `BLIND_SCORE` unbiased arm 13/40, 27 undecidable, CI upper 0.228, 0.675 | as quoted | 13 / 0 / 27, `[0.0, 0.228102]`, `0.675` | ✅ |
| generated frames absent from this machine | yes | `raw/` and `probe_clips/` are empty; `find runs -name "vision*.mp4"` → nothing | ✅ |
| **outcome M is what the evidence selects today** | yes | **I agree.** A is measurably unreachable; `p_A` does not exist; the residual clause fires | ✅ |

**§1, §2 and §4 stand. §3, §5's economics and §6's draft rule do not.** The refutations below are
ordered by weight.

---

## A.1 REFUTED — the identity fallback on the OVER-BOUND half *is* the failure the area half exists to prevent

This is the one that closes the proposal as written.

§3.2's rule and draft §3.2 both substitute the source frame on a frame that is empty **or** over the
committed `max_frame_fraction`. The deliverable's table row R5 claims for constraint (a): *"Held, and
strictly more strongly than R0"*, and §3.2 argues *"no generated pixel of any kind enters the corpus
on that frame … Constraint (a) is satisfied by construction, with no threshold and no number."*

**For the 1 385 over-bound frames that claim inverts.** The repo states three separate times, in the
code the deliverable itself executed and quoted from, that writing the source over the whole frame
*is* the defect the area half is there to catch [A]:

`scripts/robot_composite.py:1405-1408`, `check_mask`'s own over-bound message — the deliverable
printed this message in §5 and quoted only its first line:

> A mask this large has grounded on something that is not the robot — the table, or the whole scene.
> **Compositing it copies the SOURCE back over everything, the restyle becomes a no-op, and arms B
> and C silently become arm A while still costing their GPU hours.**

`scripts/robot_composite.py:1225-1236`, `load_area_bound`'s refusal of a bound of exactly `1.0`:

> a committed bound of exactly 1.0 makes the over-large refusal **UNREACHABLE — half of this check
> switched off**, in a committed file, passing every other validation … The empty-mask half would
> still fire, so the failure is silent: over-large masks composite the source back over the whole
> frame, the restyle becomes a no-op, and arms B and C become arm A at full GPU cost.

`scripts/robot_composite.py:1163-1170`, the segmenter cross-check, same sentence again.

**So V20 §3.2 achieves, per frame and by design, the exact end state the repo classifies as "half of
this check switched off".** It reaches it through the work unit instead of through the config file,
which is a difference in route and not in outcome. Three consequences:

1. **The R5 row is wrong.** Constraint (a) is not held "strictly more strongly than R0" on those
   frames; it is held on the *generated-robot* reading and abandoned on the *restyle-is-a-no-op*
   reading, which is the reading the bound was signed for. `configs/transfer25/pr08_robot_mask_area.json`'s
   own `bound_rationale` says what the number is for [A]: *"WHAT THIS NUMBER DOES NOT DO: it ARMS
   G0c's area half."* V20 disarms it on every frame where it would fire.
2. **The deliverable's §3.4(1) does not cover this.** It classes the harm as arm-A dilution bounded
   by β. The repo does not: it calls it *"the failure … silent"* and treats reaching it as switching
   a check off. A signer told "β guards interpretability, not corpus integrity" (draft §4) is being
   told something the committed rationale for that very bound contradicts.
3. **It does not even save the GPU.** Over-bound frames are non-empty, therefore inside the trimmed
   span, therefore generated. Same for the 5 313 interior zeros. **6 698 frames × 8 style-instances
   = 53 584 generated frames are produced and thrown away.** The §3.2 table's R5 column *"costs GPU
   before it decides? **No** — decidable from source"* is false for the identity half of R5; only
   the trim half is free. [M, from the deliverable's own span arithmetic.]

**A repair exists and the deliverable does not take it:** split the two halves. Trim on empty runs
(defensible), and let the over-bound frames stay clip-fatal — they are 1 385 frames in 175 episodes
and they are, on the committed reading, evidence the masker grounded on the scene, which is a
different fact from "the robot is out of shot" and has a different correct response.

---

## A.2 REFUTED — the proposal silently breaks a committed harvest contract, and §7 does not list it

`cluster/discoverer/97_transfer25_restyle.sbatch:2065-2072` [A]:

```python
    want = int(row["frames"])
    if whole_frames(g0c, "frames_composited") != want or whole_frames(g0c, "frames_total") != want:
        refuse(unit,
               f"it composited {g0c.get('frames_composited')!r} of {g0c.get('frames_total')!r} "
               f"frames and this work-list row declares {want}.",
               "Every frame or none: a clip composited on part of its frames carries the generated "
               "manipulator on the rest, and nothing downstream reads a corpus frame by frame.")
```

and the prose contract at `:849-853` [A]: *"only after all of these held: status success; backend
transfer25 …; g0c.composited true; **frames_composited == frames_total == the frame count that unit's
work-list row declares**; and an area bound cross-checked …"*

**"Every frame or none" is precisely what V20 §3.2 abolishes.** §3.5(1)'s diff leaves
`"frames_composited": int(src.shape[0])` and `"composited": True` untouched and adds parallel
`frames_identity` / `identity_fraction` keys that the harvest does not read. Both branches fail:

- **If the source video is left whole** (as §3.5(1)'s diff has it — `composite_clip` decodes
  `source_video` and loops `range(src.shape[0])`), then `frames_composited` is the *full* episode
  count while the trimmed work-list row declares the *span* count, and the harvest refuses **every
  clip**. On top of that `composite_clip:1610`'s `gen.shape != src.shape` guard fires first, because
  the backend was handed the trimmed span.
- **If the source is trimmed too** (which §3.5(3) implies but never states and never sketches), the
  counts line up and the harvest files the clip — with a `<unit>.g0c.json` beside it asserting that
  every one of its frames was composited, when up to **50.39 %** of them were never composited at
  all. That is the evidence-travels-with-the-clip discipline made to state a falsehood.

§7's *"would NOT change"* row and §5's five refutation conditions both omit
`cluster/discoverer/97_transfer25_restyle.sbatch`. **A successor that changes what a clip is may not
leave the artifact that certifies clips out of its blast radius.**

---

## A.3 REFUTED — the sketch contains the exact off-by-one class it declines to sketch elsewhere

§3.5(2)'s plan body computes identity indices **relative to the span**:

```python
span  = af[lead : trail + 1]
ident = [i for i, v in enumerate(span) if v == 0.0 or v > bound.max_frame_fraction]
```

§3.5(1)'s diff consumes them **relative to the source array**:

```python
    for index in range(src.shape[0]):
+        if index in identity_frames:
```

**These are two coordinate systems.** They coincide only if `src` is itself the trimmed span, which
§3.5(1) does not do and §3.5(3) does not say. Otherwise every identity index is short by `lead`, and
the effect is that the source frame is written onto the wrong frame — 309 episodes carry a lead run
[M, reproduced §4.1], so the offset is non-zero for most of the corpus. There is no exception, no
crash and no gate: `check_mask` would then fire on the *actual* empty frame and refuse the clip,
which at least makes it loud — but on the 1 385 over-bound frames it would silently substitute a
source frame at a wrong index, i.e. **put the robot from one instant into the scene of another**,
the failure `composite_clip:1612-1616` names in as many words [A].

§3.5(4) declines to sketch the parquet slice because *"an off-by-one is a shifted control law, and
that hunk deserves its own tests rather than a sketch in a report."* The same standard applied to
§3.5(1)+(2) would have caught this.

---

## A.4 REFUTED — the diff writes `NaN` into an artifact, which this exact module refuses by name

§3.5(1) appends `float("nan")` to `fractions` on identity frames. `fractions` is consumed at
`scripts/robot_composite.py:1691-1695` [A]:

```python
        "mask_area_fraction": {
            "min": float(np.min(fractions)),
            "mean": float(np.mean(fractions)),
            "max": float(np.max(fractions)),
        },
```

`np.min`/`mean`/`max` propagate NaN [M — verified: `np.min([0.1, nan, 0.3]) → nan`], so all three
become NaN for **every clip with at least one identity frame — 336 of 402 episodes** [M, 402 − 66].
`json.dumps` then emits a bare `NaN` token [M — verified: `{"min": NaN}`], which is not JSON under
RFC 8259 and which any non-Python reader rejects.

The same file forbids this in as many words, in the docstring of the function 30 lines above
`check_mask` [A, `scripts/robot_composite.py:1360-1364`]:

> Two empty masks are 1.0, **not NaN**. The degenerate case is decided rather than left to 0/0
> because this number is written into an artifact: **a NaN there would be read as "not measured" by
> anyone skimming**, when what actually …

The deliverable's §3.5(1) note calls skipping `check_mask` on identity frames *"honest"*. Writing
NaN into the one field that records what the masks were is the opposite: it erases the mask-area
record for five sixths of the corpus.

---

## A.5 REFUTED — a count claimed as read from an artifact does not match the artifact

§2.2 [and `blocking_facts` #4]: *"(I did read the file. It contains **2 `yes`, 77 `no`, 22
`cannot_tell`**.)"*

`runs/pr08-empty-mask-look/VERDICTS-partial-101.json` actually contains [M]:

```
{'yes': 2, 'no': 75, 'cannot_tell': 24}   (101 total)
```

**75 / 24, not 77 / 22.** Two tiles were moved from `cannot_tell` into `no`. The totals agree, so
this is a miscount and not a different file. It is not load-bearing for outcome M — the deliverable
is right that V16 §7 makes these inadmissible, and I verified the §7 sentence exists verbatim
(`PR-08-V16…md`: *"The 101 verdicts of the V15 run are not reinterpreted, not mapped onto A/B/C/D,
and not used in any estimate"*). But it is the one place the deliverable says "I opened this file and
here is what is in it", and what is in it is different. Under this project's own standard that is
the claim to distrust the rest by.

---

## A.6 REFUTED, and it cuts against the proposal — the cross-hardware number is understated

`defects_found` #5 and §3.4(3): *"16 of 48 bulk-band frames the cluster measured at ~0.13 recompute
to EXACTLY 0.0"*.

`runs/pr08-bulk-stability/TAIL_SAMPLE.json` [M — I counted every frame, not only the mismatch block]:

```
frames recomputed to exactly 0.0:  19 of 48   (39.6 %)
   16 of them are in /mismatch/frames (recorded 0.107–0.134)
    3 are NOT (episode_000021:66, _000025:74, _000026:65 — recorded 0.003–0.007, so |delta| < 0.01)
```

The deliverable's §3.4(3) sentence *"19 disagreed … and 16 of those recomputed to exactly 0.0"* is
correct; the summary's *"16 of 48"* is not. **The true figure is 19 of 48, and it is worse for the
proposal than the one quoted.** The committed result says the same thing in prose [A,
`PR-08-RESULT-2026-08-26-the-area-fraction-is-stable-except-in-the-band-nobody-uses.md` §5]:

> The bulk moves DOWN if it moves at all — recorded median 0.109 against a re-rendered 0.0032, **with
> many masks vanishing entirely.**

**Consequence for draft §3.3.** V20 §3.3's central claim is that under a correct plan `check_mask`
*"can never fire"*. On the one cross-hardware sample that exists, **~40 % of bulk frames become
empty masks on a second GPU** — every one of them a frame the plan marked "generate". The draft's §5
item 1 concedes the mechanism, but §0's headline (*"Yield: **402 of 402 episodes** [M], at **307.0
GPU-h**"*) and the summary's `headline` state the yield unconditionally. **It is a plan-time figure,
not a run-time yield**, and the only measurement bearing on the difference says the plan and the run
disagree on two fifths of a bulk sample. `segmenter_identity` does not capture the GPU [M — the
fields are the detector/segmenter pins, prompt and thresholds, `robot_composite.py:394-423`], so
§3.4's stamp cannot detect it either.

---

## A.7 REFUTED — "stride 1" is not in the artifact it is attributed to

§4's header, draft §2's provenance block and `blocking_facts` #1 all attribute to
`runs/pr08-robot-mask-area/POOLED.json`: *"402 episodes / 171 625 frames, **stride 1**,
measurement_qualified true"*.

POOLED.json carries exactly six top-level keys [M]:

```
['git_commit', 'source_manifest_sha256', 'prompt', 'estimator', 'measurement_qualified', 'per_episode']
```

**No `stride`. No `n_episodes`. No `n_frames`. No `schema`. No merge-condition block.** The episode
and frame counts are derivable from `per_episode` (and I derived them, and they are right). **`stride
1` is not derivable from this file at all.** It appears in
`runs/pr08-robot-mask-area/pr08_robot_mask_area.MEASURED.json:438` and inside the `bound_rationale`
**prose** string. So a provenance stamp labelled **[M]** against POOLED.json was in fact read out of a
different file's prose — the precise failure mode this project's method exists to prevent.

**And it is load-bearing for §3.5(2).** The plan script is specified to read
`--pooled runs/pr08-robot-mask-area/POOLED.json` and to validate `measurement_qualified: true`. But
POOLED.json is *not* the `AREA_SCHEMA` ("wam.robot_mask_area/1") artifact: it has no `schema`, no
`measurement_disqualified_reasons`, no `MERGE_CONDITIONS` verdicts. `load_area_bound` never reads it
— it reads `configs/transfer25/pr08_robot_mask_area.json` and enforces `AREA_BOUND_FIELDS_REQUIRED`
(`max_frame_fraction`, `bound_rationale`, `measured`, `measurement_qualified`, `estimator`, `prompt`,
`source_manifest_sha256`) [A, `robot_composite.py:384-392`]. **The plan's admissibility check is
strictly weaker than the one the repo already performs, on a file that is not the qualified
artifact.** If the pre-flight is to be the thing that authorises 307 GPU-h, it must read
`pr08_robot_mask_area.MEASURED.json` (or the committed bound file), not POOLED.json.

---

## A.8 REFUTED — the "misleading error text" defect quotes past the clause that answers it

`defects_found` #7: *"check_mask's empty-mask message asserts 'the GENERATED manipulator went
straight into the corpus'. On a frame whose source genuinely has no robot that is not what happened
… and the message sends an operator to fix the prompt or the thresholds when there is nothing to
find."*

The message's **next sentence** [A, `scripts/robot_composite.py:1396-1398`]:

> **If the robot is genuinely absent from this frame the SOURCE corpus is not what PR-08 §3
> describes;** if it is present, the prompt or the detector thresholds do not find it, and that has
> to be fixed before generation rather than skipped per frame.

The message states the robot-absent branch **first**, by name, and routes it to the corpus spec — not
to the prompt or the thresholds. The deliverable's characterisation is produced by stopping the quote
one sentence early. This is the only "defect" in the list that is a defect in the reading rather than
in the code.

---

## A.9 REFUTED — V12 §2's own test is passed by the half of the proposal that is not the change

§3.3 and draft §3.5 claim the proposal clears V12 §2's legitimacy test — *"This justification would
hold even if the refusal rate were 3 % instead of 99.2 %"* [A]. The argument offered is about the
**trim**: approach-and-retreat frames contain a tablecloth, so trimming them is correct on its own
terms. **I accept that for §3.1.** It does not transfer to §3.2.

- At a 3 % refusal rate there would be no boundary runs worth trimming and **nothing to propose**.
  The trim has content only because 33.70 % of frames are empty and 90.81 % of those are boundary
  runs — facts that *are* the refusal rate.
- §3.2's identity fallback applies to the **5 313 interior empty frames and 1 385 over-bound
  frames** [M]. Those are not approach and retreat. They are exactly the frames V12, V15 and V16 are
  about. The only reason offered for substituting source pixels there is that the clip would
  otherwise be refused — **which is V12 §2's named illegitimate justification, verbatim**: *"G0c
  refuses too much, refusing too much is expensive, so the gate should refuse less … It would
  evaporate if the refusal rate were 3 %."*
- The deliverable's own abstract leads with the economics: *"Yield: 402 of 402 episodes, at 307.0
  GPU-h instead of 442.4, against today's 17 of 402"*, and its `defects_found` opens with **"THE
  ECONOMIC DEFECT"**. Draft §3.5 asserts *"The refusal rate is not an argument in §3 and is not
  needed by it."* §0 and the summary are that argument.

**And V12 §2's closing instruction is still unmet by the deliverable's own §2.2** [A]: *"§3 should
not be adopted before that measurement exists."* The measurement is the (a)/(b) split; §2.2 records
that it does not exist. V20's §3 reaches the same corpus-level outcome V12 §3 would have — 385
refused episodes become 402 accepted ones — through the work unit rather than through `check_mask`.
**Whether routing around a precondition satisfies it is exactly the question `handoff.md` §3 exists
to settle, and the honest answer is no.** The deliverable is not licensed to make this proposal
today; it is licensed to make §3.1's trim, and to report that §3.2 waits on `p_A`.

---

## A.10 REFUTED — two of six "runnable now" entries do not run

- **Entry 3** (`check_mask`'s signature and refusal messages) is a **SyntaxError** as written [M]:

  ```
  SyntaxError: unexpected character after line continuation character
  ```

  The literal `\n` inside the double-quoted `-c` string reaches Python as a backslash. The
  *underlying* facts are true — I re-derived them by other means: `check_mask`'s signature is
  `(mask, *, frame_index: int, bound: AreaBound, source: str) -> float` [M, `robot_composite.py:1378`]
  and both refusal branches are at `:1388-1410` — but the command as published does not produce them.
- **Entry 5** is `.venv/bin/python <a directory path>`, with a parenthetical saying the tables are
  reproduced by snippets quoted elsewhere. It is not a command.
- Entries 1, 2 and 4 run and reproduce exactly [M]. Entry 6 is correctly marked write-only.

No entry touches the cluster, costs money or mutates the repo. That part of the brief finds nothing.

---

## A.11 OVERSTATED — "the yield on record is optimistic by 2×"

The finding *"the joint count 17 is written down nowhere"* is **true** — I searched `docs/`,
`runs/` and the task files and could not find it either [M]. But the framing is not supported. The
committed `bound_rationale` in `configs/transfer25/pr08_robot_mask_area.json` states both halves and
states that they compose by OR, in the same paragraph [A]:

> WHAT IT REFUSES: 1385 frames of 171625 … spread over **175 of 402 episodes, which is 43.5
> percent**. Both units are given because section 3.2 asks for both and they read very differently:
> **check_mask refuses a whole clip on one over-bound frame, so under that OR this is not a 0.8
> percent cost, it is a 43.5 percent cost.** … The empty-mask half of G0c refuses **366 of 402
> episodes, or 91.0 percent**.

No committed document claims 36 is the joint yield; the 2026-08-26 result's title says *"the
**empty-mask half** refuses 91.0 %"*. The defensible finding is "nobody has multiplied the two out
and written the product down, and the product is 17". Calling the record *optimistic by 2×* charges
the record with a claim it declines to make.

---

## A.12 Minor, recorded for completeness

- **`CompositeContext.composite` is omitted from the blast radius.** It is keyword-only and takes
  `(source_video, generated_video, expected_frames)` [A, `robot_composite.py:1461-1473`]. §3.5(3)'s
  *"forwards `identity_frames=…` into `composite.composite(...)`"* is a `TypeError` until that method
  changes too; §7's "would change" row does not list it.
- **The `composite_clip` diff context is not the file.** The real signature is keyword-only (`*,` on
  line 1579) and the real loop body contains the IoU block at `:1641-1643`, which the `continue`
  silently skips for identity frames — so the generator diagnostic is sampled on a different frame
  set than the record's `stride` field declares.
- **`restyle_transfer25.py:711` is the `build_context` *call*, not the refusal.** The `DriverError`
  translation is at `:719-723`. §3.5(3) cites 711 as "the `build_context` refusal".
- **Chunk quantisation is unstated.** At the 48-frame chunk the deliverable itself infers [I], the
  ratio is `128 256 / 180 768 = 0.7095`, not `119 103 / 171 625 = 0.6940` [M]. A 2.2 % relative
  shift — immaterial to the conclusion, but it means the exact agreement between `442.4` and T-040's
  `~442` is agreement between two identical linear models, not a validation of the shape.
- **`frames_emptied_by_the_filter: 9`** is recorded in `TAIL_SAMPLE.json`'s own render block [A] and
  is a competing partial explanation for the zero-recomputes that §3.4(3) attributes wholly to the
  GPU. It does not explain all 19, so the hardware reading survives — but the artifact names another
  cause and the deliverable does not mention it.

---

## A.13 What survives, and what I would keep

**Survives, unqualified:**

- **Outcome M is what today's evidence selects.** A is unreachable on a measurement I reproduced;
  `p_A` does not exist; the residual clause fires. §1 and §2 are correct and correctly sourced.
- **`check_mask` decides on source pixels alone, and the generator runs first.** Verified at
  `restyle_transfer25.py:515` / `:523` and `robot_composite.py:1620`. **The ordering observation is
  the real finding in this deliverable** and it is independent of everything refuted above.
- **The refusal set is knowable for zero GPU-hours**, and no episode-level refusal memo exists in
  `main`'s loop, so today each refusal is re-discovered eight times at full price.
- **17 of 402 is the joint yield and has never been written down.** Worth committing on its own.
- **The changed-region candidate is correctly rejected**, on grounds I could not break.

**Would keep, as a much smaller successor than V20:**

1. Write the joint refusal count (17/402) into a result document. Zero GPU-h, zero contract risk.
2. Add the source-side pre-flight **as a refusal memo only** — read the committed
   `pr08_robot_mask_area.MEASURED.json` (not POOLED.json), and skip generating episodes whose source
   already refuses. This changes no gate, no record schema, no harvest contract and no corpus, and
   it captures most of the economics the deliverable is after.
3. Take §3.1's **trim** to the owner as its own question, separated from §3.2.
4. **Drop §3.2's identity fallback** until `p_A` exists, and in any case never apply it to the
   over-bound half.

**Bottom line: `survives = false`.** The measurements are sound and the ordering finding is real; the
successor rule is not one a session may propose in this form, and it would break `check_mask`'s area
half, the sample-outputs record schema and the 97-harvest contract to do it.
