# PR-08 — V17 ran, the drift rate is bounded, and the arm that produced a contiguous failure is ours

**Reading of a measurement. Registers no rule. Discharges nothing by itself.**
`T40_RULE_V17` (registered 2026-08-27, before the first capture was rendered), `T40_RULE_V18`,
`T40_RULE_V19`. Artifacts: `runs/pr08-est-drift/v17/` and
`runs/pr08-operating-point/EPISODE_094_CENSUS.json`.

## 1. The headline, and the sentence that has to travel with it

**Outcome N — NOT OBSERVED, RATE BOUNDED.** Across 3 840 simulated frames on eight distinct
trajectories and 16 846 frames of the real corpus over 40 whole episodes, the propagation arm
produced **zero** runs of ten or more frames, and the positive control fires. Under V17 §4 that
discharges the propagation blocker's second reason.

**And a bound is not an absence.** Forty clean episodes of 402 bounds the per-episode rate at
`3/40 = 7.5 %` by the rule of three — **a clean sweep is still consistent with about 30 of the 402
episodes containing an event.** This sample can detect divergence; it cannot certify its absence.
If the decision ever needs certification, the answer is a census of 402, not a larger sample.

**The most interesting number in this document is not the zero.** It is that the only contiguous
sub-0.5 run either arm produced anywhere — **thirteen frames, A3, f294–f306, IoU exactly 0.0** — was
produced by the **per-frame arm**, the one this project actually uses, on frames where the
propagated arm sailed through. §5.

## 2. The controls, in the order V17 §4 reads them

| control | what it is | median inter-frame motion | `n_runs` | `longest_run` | fires? |
|---|---|---|---|---|---|
| **C1** | the committed lattice, object teleports | 65.28 px | 10 | **5** | **no** |
| **C2-t20** | trajectory, `turns=20` | 27.37 px | 0 | 0 | no |
| **C2-t40** | trajectory, `turns=40` | 54.02 px | 0 | 0 | no |
| **C2-t80** | trajectory, `turns=80` | 110.64 px | 0 | 0 | no |
| **C3** | propagation held on the cube distractor, over A1 | 1.32 px | 1 | **480** | **yes** |

Fire condition, fixed blind in V17 §5 and unchanged since: `n_runs ≥ 1` **and** `longest_run ≥ 10`.

**C1 failed and the failure is structural.** Its runs begin at frames 5, 10, 15, 22, 27, 32, 37, 47,
52, 57 — spaced exactly five apart, which is the lattice's own period — with `n_unscored_frames = 0`,
so nothing was broken by a missing score. The object *returns under the stuck mask* every fifth
frame. A lattice control cannot produce a run longer than the lattice repeats; `longest_run = 5` is
a measurement of the schedule. `T40_RULE_V19` §1 records this, and V19 §2 states the "keep trying
controls until one fires" objection against itself before giving its own rule. **Both pooled
artifacts are kept**: `POOLED.json` (control C1, outcome **V**) and `POOLED-V19.json` (control C3,
outcome **N**). A reader who sees only the second cannot tell V19 happened.

**The C2 ladder is the uncomfortable result and it is not buried.** V17 §5 asked for a dose-response
curve — *at what inter-frame motion does propagation begin to lose the object?* The answer measured
is **it does not, anywhere on the ladder.** At 110.64 px of median object motion per frame — 84×
A1's — SAM 2's video predictor tracked the apple with **zero** sub-0.5 frames. So there is no
measured threshold to report beside the corpus's own motion, and V17 §5's own conditional applies in
its strongest form:

> a control that fires only above some motion the real corpus never reaches would be a control that
> proves less than it looks … If it is outside, that is a limitation of this design and is recorded
> as one.

**It is outside.** The only thing in this study that made propagation fail is a **held wrong seed**,
which is the grossest possible version of limb (b). **Nothing here demonstrates that `low_iou_runs`
can detect a subtle or partial drift**, and the zero in §1 must be read with that attached.

## 3. Arm A — eight trajectories, 3 840 frames

| | `turns` | `yaw` | `arm` | median px | per-frame p95 | runs / longest | propagation p95 | runs / longest |
|---|---|---|---|---|---|---|---|---|
| A1 | 1 | 1 | 2 | 1.317 | 0.2908 | 1 / 1 | 0.4701 | 0 / 0 |
| A2 | 1 | 4 | 3 | 1.291 | 0.3061 | 1 / 1 | 0.4690 | 0 / 0 |
| A3 | 2 | 3 | 5 | 2.644 | 0.2654 | **1 / 13** | 0.3828 | 0 / 0 |
| A4 | 2 | 5 | 2 | 2.673 | 0.2260 | 0 / 0 | 0.4549 | 0 / 0 |
| A5 | 3 | 1 | 7 | 3.985 | 0.2949 | 0 / 0 | 0.4440 | 0 / 0 |
| A6 | 3 | 2 | 3 | 4.078 | 0.3601 | 0 / 0 | 0.4504 | 0 / 0 |
| A7 | 4 | 2 | 11 | 5.295 | 0.3503 | 0 / 0 | 0.4363 | 0 / 0 |
| A8 | 5 | 3 | 13 | 6.650 | 0.3371 | 0 / 0 | 0.4383 | 0 / 0 |

**No capture was excluded**: V17 §2's 25.0 px coherence bound was never approached, and A7's 5.295
px is 4.02× A1's 1.317, which is `turns=4` scaling as the schedule's arithmetic says it must.

**A1 reproduces `capture-mujoco-trajectory-f480` to the last digit** — per-frame p95
`0.290771`, propagation `0.470062`, delta `0.17929105291300962`, against the published
`0.29077062684224225` / `0.47006167975525187` / `+0.1793`. The capture was re-rendered from the
newly-exposed parameters, so this is a determinism check on the whole chain and it passed.

**Pooled — one percentile over the union of 3 840 per-frame displacements, never a mean of eight
p95s:**

| | p50 | p90 | **p95** | p99 | p100 | measured |
|---|---|---|---|---|---|---|
| per-frame | 0.0755 | 0.2064 | **0.31208** | 1.0431 | **67.633** | 3 827 / 3 840 |
| propagation | 0.2247 | 0.3798 | **0.44861** | 0.5631 | **19.399** | 3 840 / 3 840 |

**The crossing the single capture showed is confirmed at eight times the sample, and it is much
wider than it looked.** At p95 — the statistic §6 subtracts — propagation is the worse arm by
0.1365 px. At p99 the per-frame arm is 1.85× worse, and at p100 it is **3.5× worse: 67.6 px against
19.4**. Both readings are here because quoting either alone makes the sign of the difference look
settled when it depends on which percentile the gate uses.

**What that does to G0b**, against `GEOM_TOL = 0.47857992441961017 px`:

| arm | pooled `EST_DRIFT_P95` | margin | % of `GEOM_TOL` |
|---|---|---|---|
| per-frame | 0.3120786214 px | 0.1665013030 px | 34.79 % |
| propagation | 0.4486097454 px | **0.0299701790 px** | **6.26 %** |

The propagation margin was **1.78 %** on the single capture. Eight trajectories move it to 6.26 %.
**That is still not room**, and V17 §4 is explicit that the pooled number decides nothing here:
which p95 G0b subtracts is a separate open question this document does not answer. `GEOM_TOL`
itself remains uncommittable for an unrelated reason — the merged 16-shard artifact predates
`mask_validity_reference_max_frame_fraction` and `contract_disagreements` refuses it.

## 4. Arm B — the corpus the blocker actually names

40 whole episodes drawn stratified-systematic under `sample_seed = 40017`, fixed in V17 §3 before
anything was decoded. **16 846 frames.**

* **0 divergence runs.** `longest_divergence_run = 0`.
* **0 frames**, of 16 846, where the two arms' masks scored below IoU 0.5. Not "few" — none.
* Per-episode median cross-arm IoU spans **0.9788 to 0.9937**; the lowest 5th percentile in any
  episode is **0.9173**.
* `n_frames_both_masks_empty = 0`: every frame had at least one non-empty mask, so no run was
  broken by an unscored frame anywhere.

On the real pixels, at the real occlusions, over 40 episodes, **the two topologies produce very
nearly the same mask on every single frame.**

## 5. The premise Arm B's inference rested on is false, and this measurement is what falsified it

V17 §3 argued, in the open so it could be refused:

> The per-frame arm re-detects independently every frame, so its errors are **independent across
> frames** … Independent errors produce runs of length 1 overwhelmingly; a run of length 20 by
> independent single-frame failures at the measured rate is arithmetically negligible.

**A3 f294–f306 is a run of thirteen consecutive frames on which the per-frame arm scored IoU
exactly 0.0**, while the propagation arm scored no sub-0.5 frame anywhere in that capture. The
per-frame arm's errors are **not** independent across frames, because *scene difficulty* is
correlated across frames — a sustained occlusion is hard for thirteen frames running, and an
estimator with no memory fails for all thirteen while one with memory tracks through.

**So the attribution rule V17 §3 registered — that a long cross-arm divergence run belongs to the
propagation arm — is not sound as stated.** It would only have mattered had Arm B found a run, and
Arm B found none, so no conclusion in this document rests on it. **It is recorded because the next
person to reach for that inference needs to know it was tested and failed**, and because a design
whose reasoning is refuted by its own control data should say so in the same document as its result.

This also inverts the blocker's framing at the run level. Limb (b) said propagation's characteristic
failure is *"drifting off the object and staying off for a run of frames, invisible to a per-frame
estimator that recovers on the next frame."* Measured here, the contiguous failure is the per-frame
estimator's and propagation is the arm that recovers — because it never lost the object.

## 6. V18 — residue (i), and what the census actually settles

`episode_000094`, every one of its 509 frames, both decode trees, adapter unmodified at HEAD.

| | H.264-lossless | AV1 original |
|---|---|---|
| frames | 509 | 509 |
| frames with a mask | 478 | 478 |
| `n_frames_with_centroid` | 478 | 478 |
| refused as the wrong object | **31**, span `[109, 149]`, not contiguous | **31**, same frames |
| no detection / empty mask | 0 / 0 | 0 / 0 |
| median non-refused mask area | 5 650 px | 5 650 px |
| largest non-refused mask | 7 383 px = **1.31×** median | same |

**V18 outcome C — CONTAINED.** Outcome U does not fire: there are refusals and they sit inside the
documented `~f101–f155` low-score run. Outcome E does not fire: **zero** non-refused masks exceed
3× the episode's median non-refused area; the largest is 1.31×, and the plate masks the audit found
are ~4.3×. The threshold was coined blind in V18 §3 from that measured gap and did not move.

**The two decodes are bit-identical, which is stronger than the caveat they were meant to
measure.** 0 of 509 frames differ in mask area, and 0 differ in centroid. The operating-point result
attributed the `f129: 0.213 vs 0.232` score difference to *"different corpora … the decodes differ"*.
**They do not differ** — `TRANSCODE_PROOF.json` records max abs channel delta 0, and the estimator
sees identical pixels. That difference is between *machines*, not between codecs, and the earlier
attribution is corrected here.

**The determinism control, run the way the precedent runs it.** Two separate census runs on this
workstation, same command: **0 of 509 frames differ in mask area on either tree.** The masker does
not wander between runs here.

**An unexplained 5, named rather than smoothed.** `runs/pr08-geom-tol/shards/shard-7.json` records
`episode_000094` as `n_frames: 509, n_frames_with_centroid: 473`. This census measures **478**. The
gap is five frames and it is **not** explained by the largest-connected-component floor: every one
of the 31 frames without a centroid here is a refusal, and none is a fragmented mask. It also runs
against `apple_sam2`'s own claim that *"V10's only possible effect on a frame is a refusal"* — the
shards ran a pre-V10 adapter and should therefore refuse **fewer**, not five more. What is left is a
cross-machine difference of the class
`PR-08-RESULT-2026-08-26-the-area-fraction-is-stable-except-in-the-band-nobody-uses.md` already
established for the robot prompt: deterministic within a machine, different between machines.
**That is a hypothesis, not a measurement**, and closing it needs the census re-run on the cluster.

**Four things V18 §3 requires in the determination, and a fifth:**

1. 31 refusals, frames `[109, 149]`, identical on both decodes.
2. **This is one episode of 402.** The census establishes containment *here* and bounds nothing
   corpus-wide, because the corpus pass recorded no per-frame flag to census against.
3. **Nobody has looked.** The area test is a proxy for a wrong-object mask, not an observation of
   one, and a wrong-object mask of apple-plausible area would pass every test in V18.
4. The failure blocker 2 predicted **did occur**, by a route it did not predict. Accepting a
   0.054 % rate is a decision about a rate, not a finding that nothing happened.
5. **The 5-frame disagreement above is open**, and it sits in the evidence for one of
   `GATE_QUALIFIED`'s two preconditions.

## 7. What this discharges, and what it does not

**Discharged:** the propagation blocker's **second** reason — *"480 frames of ONE trajectory is not
a corpus"* — under V17 §4 outcome N, as a bound and not as an absence, carrying §1's rule-of-three
reading, §2's C2 limitation, §4's corpus numbers and §5's falsified premise. Its first reason was
closed by `T40_RULE_V14`. The tuple entry therefore retires in a separate, reviewable commit that
moves the wording verbatim into `GATE_QUALIFICATION_DISCHARGED`.

**Not discharged, and not touched:**

* **`GATE_QUALIFIED` is not flipped by this document or by the commit that retires the entry.** Its
  own text forbids a tuple-shortening commit from flipping it, and §6 item 5 is an open question
  inside the other precondition's evidence.
* **Blocker 1 — "nobody has looked at a mask" — is untouched.** Everything here is a counter.
* **G0b does not pass.** 6.26 % is a margin, and `GEOM_TOL` cannot be committed at all until the
  corpus is re-measured at HEAD.
* **No clip is licensed.** `T40_RULE_V1` §1 binds. §8 items 3 and 4 are open.

## 8. Provenance

| | |
|---|---|
| kind | reading of a measurement. Registers no rule. |
| rules | `T40_RULE_V17` (blind, pre-capture), `T40_RULE_V18` (blind, pre-census), `T40_RULE_V19` (post-C1, no threshold moved) |
| Arm A | `runs/pr08-est-drift/v17/EST_DRIFT-A{1..8}.json`, 3 840 frames, 8 trajectories |
| Arm B | `runs/pr08-est-drift/v17/ARM_DIVERGENCE.json`, 40 episodes, 16 846 frames, seed 40017 |
| controls | `EST_DRIFT-C1-lattice.json`, `EST_DRIFT-C2-t{20,40,80}.json`, `EST_DRIFT-C3-wrongseed.json` |
| pooled | `POOLED.json` (control C1 → **V**), `POOLED-V19.json` (control C3 → **N**) |
| V18 | `runs/pr08-operating-point/EPISODE_094_CENSUS.json`, 509 frames × 2 decodes |
| device | one RTX 5090, one process, every measurement in this document |
| simulator caveat | Arm A is MuJoCo: untextured 14-group convex proxy mesh, static prop, rasteriser, cube distractor in scene. `T40_RULE_V14` licenses the substitution for `EST_DRIFT_P95` and the arm comparison and for nothing else. |
| generation licensed | **none** |
| training licensed | **none** |
