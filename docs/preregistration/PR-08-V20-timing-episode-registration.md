# PR-08 V20 — the throughput measurement needs an episode G0c does not refuse, and choosing one is a selection

**Rule `T40_RULE_V20`. Registered 2026-08-27, BEFORE the timing job is submitted and before any
`THROUGHPUT.json` exists at the `RUN_ID` this rule names. §4's disclosure and §5's outcomes are
fixed here and are not moved afterwards.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place. A gate rewritten after seeing its output is not a gate."*

**Decided by the project owner on 2026-08-27**, as outcome **REGISTER-EPISODE** of decision D-E in
`docs/PR-08-DECISION-SHEET-2026-08-27.md` §5, against the two alternatives WAIVE and
RESOLVE-V12-FIRST.

---

## 1. The problem this exists to solve, and why the other two exits were refused

`97 TIMING=1` at HEAD generates a full 590-frame clip for the unit at the head of its work list and
**then** refuses in `robot_composite.check_mask` on frame 0. Measured from
`runs/pr08-robot-mask-area/POOLED.json`: `episode_000000` carries **254 empty-mask frames of 590**,
and frame 0 is one of them. So `§8` item 3's measurement cannot be taken at all without saying
something about G0c.

**WAIVE was refused**, and the sbatch pre-registered the refusal before this question was asked:

> a throughput number that excluded the gate would under-derive the GPU-h ceiling the generation
> path is then held to

> timing the generator without the composite would be a wall clock around a pipeline we do not run

**RESOLVE-V12-FIRST was refused** on cost: item 3 is otherwise fully independent of items 1–5 of the
sprint's critical path, and making it wait on the corpus-wide empty-mask semantics couples two
things that need not be coupled. That semantics question remains open and is **not** answered here.

**Since 2026-08-27 the refusal is cheap**: a source-mask preflight runs `check_mask` over the source
masks *before* the backend, so the unit is refused in seconds rather than after ~0.3–0.5 GPU-h. That
changed the price of this problem. It did not change the problem.

---

## 2. The population, and how it was computed

The measurement must use an episode that **both** halves of `check_mask` accept: no empty robot mask
on any frame, and no frame above the committed area bound
`max_frame_fraction = 0.64091145833333329` (`configs/transfer25/pr08_robot_mask_area.json`).

Computed from `runs/pr08-robot-mask-area/POOLED.json` (402 episodes, 171 625 frames,
`measurement_qualified: true`, sha256
`631103a8a97010c4804ac039aecc7fd8425c226c750294335fad5938c35233db`): 366 episodes carry an empty
mask, 175 exceed the area bound, 385 do either — **17 survive both.** That product appears in no
other document in this repository.

| episode | frames | | episode | frames | | episode | frames |
|---|---:|---|---|---:|---|---|---:|
| `episode_000093` | 448 | | `episode_000118` | 460 | | `episode_000243` | 417 |
| `episode_000098` | 439 | | `episode_000120` | 464 | | `episode_000244` | 469 |
| `episode_000114` | 451 | | `episode_000121` | 448 | | `episode_000245` | 497 |
| `episode_000115` | 386 | | `episode_000136` | 357 | | `episode_000371` | 422 |
| `episode_000116` | 424 | | `episode_000137` | 417 | | `episode_000373` | 366 |
| `episode_000117` | 426 | | `episode_000375` | 418 | | | |

Corpus median episode length **421.5** frames, mean 426.9.

---

## 3. The selection rule, and the episode it yields

**The rule is a criterion, not a name**, so that it can be checked rather than trusted: **of the 17,
take the episode whose frame count is closest to the corpus median of 421.5 frames; break ties by
lowest episode id.**

That yields **`episode_000371`, 422 frames, |Δ| = 0.5.** Runners-up: `episode_000116` (424, Δ 2.5),
`episode_000375` (418, Δ 3.5), `episode_000117` (426, Δ 4.5).

> **A correction to `docs/PR-08-DECISION-SHEET-2026-08-27.md` §5 D-E**, which recommended
> `episode_000093` as *"closest of the 17 to the corpus median of 421.5"*. It is not:
> `episode_000093` has **448** frames, |Δ| = 26.5, eighth-closest. The sheet is wrong on that line
> and this rule does not follow it.

**Reaching it.** `eps = sorted(man["episodes"], key=lambda e: str(e["id"]))` and
`mine = eps[idx - 1::total]`, so `CHUNK_TOTAL=402` with a 1-based `CHUNK_INDEX` selects exactly one
episode. **The index is not asserted here**, because it depends on the manifest rather than on this
document. The operator resolves it and checks it before submitting:

```bash
# on the cluster, from ${PROJ}/wam, as part of the submission — prints the 1-based CHUNK_INDEX
python -c "
import json,sys
m=json.load(open(sys.argv[1]))
eps=[str(e['id']) for e in sorted(m['episodes'], key=lambda e: str(e['id']))]
print('CHUNK_TOTAL=%d CHUNK_INDEX=%d' % (len(eps), eps.index('episode_000371')+1))" <manifest.json>
```

If the manifest does not contain `episode_000371`, **the submission does not proceed** and this rule
is re-evaluated against the manifest that exists. It is not silently retargeted at a neighbour.

---

## 4. The disclosure, fixed here because it is the whole cost of this exit

**The selection is post-hoc with respect to G0c, and the resulting `seconds_per_frame` is biased by
it.** An episode chosen *because* it survives G0c is an episode in which the robot is visible and
in-bound on every frame. That is not a median episode of this corpus — 95.8 % of episodes are not
like it — and the number this run produces is a wall clock around a 4.2 % population.

**The direction of the bias is NOT KNOWN and is not guessed here.** Mask work is per-frame and does
not obviously scale with whether a mask is empty; the composite's cost may differ; nothing has
measured it. Any document that later quotes this run's `seconds_per_frame` must carry the sentence
*"measured on an episode selected for surviving G0c"* — a ceiling derived from it and presented
without that clause is a misquotation of this rule.

**The area half of the 17-list is machine-conditional.** `pr08_robot_mask_area.json`'s
`bound_rationale` records that **37 of 44** near-bound frames moved by more than 0.01 when
re-rendered on an RTX 5090, and `runs/pr08-bulk-stability/TAIL_SAMPLE.json` records that **19 of 48**
bulk-band frames recompute to exactly 0.0 — i.e. to an *empty mask* — on a second GPU.
`segmenter_identity()` does not capture the GPU. **So the 17 is a property of the machine that
computed it**, and `episode_000371` is not guaranteed to survive `check_mask` on an H200. If it does
not, that is outcome **R** below and it is a result, not an accident.

**And the ceiling is derived from the corpus, not from this episode.** `CEILING_GPU_H` and
`PARTITION_CEILING_GPU_H` multiply a per-frame rate by the manifest's own `corpus_frames`, never by
this episode's 422. Selecting the episode selects the *rate*; it must not be allowed to select the
*extent*.

---

## 5. Outcomes, fixed before the job is submitted

**Outcome M — MEASURED.** The unit completes, `THROUGHPUT.json` is written with
`schema: wam.transfer25_throughput/1` and `units_succeeded >= 1`. Then §8 item 3's rate exists, and
it is carried forward **with §4's disclosure attached in the artifact's own text**. The derived
`PARTITION_CEILING_GPU_H` is a ceiling under that disclosure and under no other reading.

**Outcome R — REFUSED ON THE CLUSTER.** The source-mask preflight refuses `episode_000371` on the
H200 although this workstation's artifact says it survives. Then **the machine-conditionality in §4
is confirmed as load-bearing rather than a caveat**, item 3 stays open, and what is owed next is a
re-measurement of the robot-mask area pass on the machine that will generate — not a second guess at
a different episode from the same stale list. **Explicitly: outcome R does not license walking down
the 17 until one passes.** That is fixed here, blind, because it is the thing a frustrated operator
would do.

**Outcome F — FAILED FOR ANY OTHER REASON** (walltime, cold checkpoints, licence, a crash). Then
nothing about G0c or this selection was learned, the rule stands unchanged, and the run is repeated.
An `F` may not be reported as an `R`.

**In all three cases the run generates at most one clip, which is deleted**, and no frame it produces
enters any corpus. `T40_RULE_V1` §1 is not lifted by this rule and forbids everything else.

---

## 6. What this rule does NOT do

* **It does not resolve the corpus-wide empty-mask semantics.** `T40_RULE_V12` stays unsigned, V16's
  outcome stands at **M**, and the 17-of-402 yield is untouched. This rule is about one measurement,
  not about what a corpus run may refuse.
* **It does not weaken `check_mask`.** Neither half is waived, on the timing path or anywhere else.
  The preflight moved the discovery earlier; it did not move the decision.
* **It does not close §8 item 3.** Only outcome M does, and only with the disclosure attached.
* **It licenses no clip, no corpus and no training.**

## 7. What would refute this rule

A demonstration that the selection's bias runs in a *measurable* direction — for example, a second
timed run on an episode with many empty-mask frames, taken with G0c waived and quarantined, showing
a per-frame rate that differs materially from `episode_000371`'s. That would turn §4's "direction not
known" into a number, and the ceiling would have to be corrected by it. Nothing forbids that
experiment; it simply is not what §8 item 3 asks for, and it is not run first.
