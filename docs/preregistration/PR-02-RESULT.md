# PR-02 — result

**Measured 2026-08-02.** The pre-registration is `PR-02-hiw500-screen.md`, committed before any
predictive number existed on HIW-500. Per repo convention that document is **annotated, never
edited**, so the result lives here.

CPU only, no allocation spent, **no video decoded and none downloaded**. The whole screen is three
parquet files — `data/raw/hiw500/data/chunk-000/file-{000,001,002}.parquet`, 240.7 MB, 418 episodes,
739 120 rows, 6.84 h. The scoring code is **not in the repo**; four independent lenses each wrote
their own, all under the session scratchpad, and nothing in the repo was modified. Every lens gated
on the archived GR00T numbers before quoting an HIW number, and all four reproduced
`1.632760e-05 / 9.137664e-06 / 6.330899e-06 / 5.431371e-06` to the digit.

---

## VERDICT — MIXED

The three PASS clauses hold on the letter, by wide margins, on three independent implementations.
**The mandatory confound control fires, and PR-02 says that downgrades PASS to MIXED.** It is
downgraded.

| clause | requirement | measured (arm joints, manipulation subtasks only) | holds? |
|---|---|---|---|
| M1 — momentum share | ≤ 0.45 | **−1.28 … −2.13** across three implementations (GR00T +0.660) | ✅ |
| M2 — blind-unreachable energy | ≥ 0.45 | **0.761 … 0.800**, lowest CI floor 0.737 (GR00T 0.333) | ✅ |
| M3 — grasp liveness | > 0.5 /ep | **3.837 … 7.670** /ep, repo's own debounce; 4.24 on the strictest definition of the weakest channel (our converted corpus: 0.000) | ✅ |
| FAIL | M1 ≥ 0.60 **and** M2 ≤ 0.40 | neither holds | — |
| **mandatory control** | M2 excess over GR00T must **not** be present at step 0 | **it is: +0.17 … +0.22 at step 0** (HIW 0.291–0.346 vs GR00T 0.1236) | ❌ **fires** |

**Verdict: MIXED.** Not FAIL — HIW-500 is unambiguously not the same trap. Not PASS — the excess
that the screen exists to find is present at the 33 ms horizon where visual anticipation is worth
least, and PR-02 pre-committed that reading.

### The primary table

Three implementations, three different split seeds, three separately written chunkers, ridges and
RFF ceilings. Manipulation subtasks only, 14 arm joints (`observation.state[15:29]`), 16-step
non-overlapping chunks, targets = per-step joint deltas in rad.

| | holdout eps | chunks | zero-delta | const-velocity | ridge λ=1e-2 | blind ceiling | M1 | M2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lens A (primary) | 84 | 7 626 | 9.271691e-05 | 1.272265e-04 | 7.585845e-05 | 7.340017e-05 | −1.787 | 0.7917 |
| lens B (independent reimpl.) | 84 | 6 857 | 7.813362e-05 | 1.021759e-04 | 6.210722e-05 | 5.942197e-05 | −1.285 | 0.7605 |
| lens B, corrected ceiling | 84 | 6 857 | — | — | — | 5.982e-05 | −1.313 | **0.7656** |
| lens D (arm control) | 40 | 3 187 | 8.678208e-05 | 1.237560e-04 | — | 6.938204e-05 | −2.125 | 0.7995 |
| **GR00T (archive)** | 40 | 1 040 | 1.632760e-05 | 9.137664e-06 | 6.330899e-06 | 5.431371e-06 | **+0.660** | **0.333** |

The one number that needs no denominator: **const-velocity / zero-delta = 1.31–1.37 on HIW,
0.560 on GR00T.** A zero-parameter momentum rule is 31–37 % *worse* than holding still on HIW and
44 % better than holding still on ours. That sign flip is the finding, and it reproduced three
times from raw parquet.

### The mandatory control, in full

| | step 0 | steps 8–15 | whole chunk |
|---|---:|---:|---:|
| GR00T M2 | 0.1236 | 0.4130 | 0.333 |
| HIW M2 (lens A, manip) | 0.3233 | 0.9154 | 0.7917 |
| HIW M2 (lens C, manip) | 0.291 | — | 0.7656 |
| **excess** | **+0.17 … +0.20** | **+0.50** | **+0.43 … +0.46** |
| share of total excess already at step 0 | **39–46 %** | | |

PR-02: *"A PASS requires the excess to be concentrated in the late steps."* It is 2.2–2.5× larger
late than at step 0 — but it is not absent at step 0, and the clause is written on presence, not on
concentration ratio. Scored as written: **fires**.

**The control's named mechanism is refuted, and I am saying so rather than using it to argue the
clause away.** PR-02 attributes a step-0 excess to jerk / sensor noise. Four independent tests say
it is not that:

- implied white-noise share of per-step delta variance: **−0.014 (HIW), +0.016 (GR00T)** — neither
  corpus has a measurable noise floor; 11 stale frame-pairs in 738 702; smallest nonzero delta
  7.63e-06 rad, so no quantisation floor;
- **block aggregation.** Ridge is linear in the targets, so summing the 16 per-step deltas into
  *B*-step displacements and summing the same fitted predictions is exactly the fit at the
  aggregated horizon. At B = 1, 2, 4, 8, 16 the M2 gap is **+0.436, +0.441, +0.444, +0.440, +0.449
  — constant.** Integrating away every 30 Hz wiggle does not close it. At a 0.53 s horizon blind
  still leaves 65.5 % of HIW's displacement energy against GR00T's 20.6 %;
- **frequency banding.** Only 5.6 % of HIW's target energy is above 7.5 Hz. In the < 2.8 Hz band
  that carries 79.7 % of it, M2 is still 0.774 vs GR00T's 0.342;
- lens A's 81-dim control: handing the blind model the entire body (q29 + dq29 + wbc23) moves M2 by
  0.010. The 32-dim handicap is not the cause either.

So the symptom fires and the mechanism does not. PR-02 wrote the rule on the symptom. Same precedent
as PR-01's clause 4: score the clause as written, then say plainly that the number underneath it is
not what the clause assumed. **The verdict is MIXED and the reason is not jerk.** The reason is in
the next section.

### The 0.10 disagreement clause also fires

PR-02: *"If the two disagree by more than 0.10 on either quantity, the disagreement is the headline
and no verdict is issued from the pooled number."*

| | manipulation-only | all frames | Δ |
|---|---:|---:|---:|
| M1 (lens A) | −1.7865 | −1.9588 | **0.172** ❌ |
| M1 (lens B) | −1.2849 | −1.4736 | **0.189** ❌ |
| M2 (lens A) | 0.7917 | 0.7943 | 0.003 ✅ |
| M2 (lens B) | 0.7605 | 0.7737 | 0.013 ✅ |

It fires on M1 in both implementations. Two riders, stated and not used to wave it off: M2 does not
disagree at all, and **the direction is the opposite of the confound the clause was written to
catch** — locomotion frames make M1 *more* negative, not less, so parked arms during "move to bed"
are not inflating the primary. Measured directly: mean arm delta² during `move to` segments is
7.63e-05 rad² against 9.78e-05 in manipulation frames, a ratio of 1.28. The arm is not parked. The
consequence stands regardless: **no verdict comes from the all-frames pooled number.** The
manipulation-only primary carries the verdict, and it lands on MIXED anyway.

---

## What the M2 gap actually is

This is the part that decides whether HIW helps, and it is not in any lens's headline.

**Roughly half the gap is velocity autocorrelation, not task content.** HIW's per-step arm-delta
lag-1 autocorrelation is **0.827**; GR00T's is **0.927**. A stationary AR(1) toy carrying nothing but
each corpus's measured lag-1 correlation — zero task content, zero vision, zero clutter — already
implies M2 floors of **0.865 (HIW) vs 0.651 (GR00T)**. That accounts for **49 % of the measured gap
at B = 1 and 60 % at B = 16**.

What survives as "blind structure GR00T has and HIW does not" is how far each corpus beats its own
floor: **GR00T by 0.321, HIW by 0.099 — a residual structural gap of +0.221, not +0.436.** Real, and
half the size the headline implies.

**A further ~quarter of HIW's M2 is unreachable by any predictor, vision included.** DCT-II along the
16-step axis: HIW's chunk-mean (the net 0.53 s displacement — the most task-like, most
visually-determined quantity there is) carries **48.6 %** of scored target energy against GR00T's
**80.2 %**. Modes 0–2 (< 2.8 Hz, the fastest deliberate visually-guided motion) carry 79.4 % vs
91.8 %. A perfect < 2.8 Hz trajectory oracle still leaves **20.6 %** of HIW's scored energy on the
table against GR00T's 8.2 %. Of M2 = 0.766, at most ≈ 0.57 is claimable by any smooth visual
predictor.

**And the M2 ≥ 0.45 threshold does not discriminate on this platform.** Same pipeline, same split,
same chunking, only the target channel changes:

| target channel | M2 | M1 |
|---|---:|---:|
| base / pivot velocity **command** | 0.990 | −130.4 |
| measured EE pose (`wbc[7:19]`) | 0.970 | −23.1 |
| waist joints | 0.958 | −21.3 |
| commanded EE pose (`action[7:19]`) | 0.948 | −11.1 |
| leg joints | 0.946 | −18.6 |
| **arm joints — the pre-registered target** | **0.799** | **−2.05** |
| GR00T arm joints, same code | 0.396 | +0.725 |

Every channel of this robot clears the pre-registered bar, including a base-velocity *command* that
no camera informs and balance-controller leg joints where "blind fails" carries no information about
task content. **The pre-registered target sits at the bottom of the pack, not above it.** M2 on HIW
is measuring a platform-wide 30 Hz signal-smoothness property at least as much as visual dependence.
That is the single strongest argument against reading MIXED as a near-PASS.

The robustness that *does* survive, and it is not small:

- **M1 is negative on all 10 tasks** (−0.18 to −4.21) and across 6 split seeds (−1.41 to −2.05).
- **M2 is stable**: per-task 0.673–0.853, median 0.798, every 95 % CI floor ≥ 0.605. The task
  closest to our corpus (`setting the table` — tabletop, near-static base) gives 0.796; the most
  blind-friendly task (`kitchen organization`) still gives 0.712.
- **Length matching strengthens it.** Truncating every HIW episode to 427 frames — matching our
  chunks-per-episode and corpus size — moves M2 *up* (0.774 → 0.798, manip 0.795 → 0.822), not down.
  Reproduced independently by two lenses to the same chunk counts.
- **Data scarcity does not manufacture it.** GR00T's M2 at 17/33/66/132/264/362 train episodes:
  0.433/0.398/0.385/0.364/0.336/0.330. Even at 17 episodes GR00T stays below the 0.45 bar.
- **Per-episode distribution.** 0 of 84 HIW holdout episodes fall below the 0.45 bar (min 0.4905);
  37 of 40 GR00T episodes do.
- **Not outliers, not a dead joint, not idleness.** Trimming the top 5 % of chunks by energy makes
  const-velocity *worse* (1.372 → 1.451). All 14 arm joints have cv/zero > 1 (1.174–1.613) and
  per-joint M2 0.727–0.902. Re-scoring the frozen ceiling on the top decile by ‖dq_arm‖ gives
  M2 = 0.738, still 2.2× GR00T's.

---

## Corrections — applied, not buried

Every lens was adversarially verified. Two headline claims were **refuted outright**. Corrected
numbers are used above; the claimed ones are not.

| claimed | by | corrected to | why |
|---|---|---|---|
| M1 = −1.79 read as a 2.45-unit gap against GR00T's +0.660 | lens A | momentum gap is **−0.31…−0.37 vs +0.44** in units of zero-delta energy; M1 at GR00T's denominator = **−0.46…−0.56** | M1 shares its denominator `(zero − ceiling)` with M2. HIW's denominator is 0.21× zero, GR00T's 0.67× — a **3.2× compression**. Most of M1's distance from +0.660 is the M2 gap counted a second time. **The clause still holds under either normalisation.** |
| "no smoothing width rescues momentum" (best 1.274× zero-delta) | lens A | **0.846×** (M1 **+0.340**) under a 433 ms centred boxcar applied to `q` *before* chunking; 1.005× (M1 −0.012) at 300 ms | Their k-sweep smoothed only the **predictor**; the const-velocity penalty lives in the **target**. The sign flip is filter-dependent. **The M1 ≤ 0.45 clause survives anyway** — at the filter width that maximally rescues momentum on HIW, M1 = +0.340, still below the bar, while GR00T moves the other way to +0.886. |
| native action space M2 = 0.9864, M1 = −58.6 — *"the action space deepens the verdict"* | lens D | **REFUTED. Do not use.** Normalised: 0.9227 / −10.89. Increment over the arm control: **+0.033**, inside the control's own CI [0.740, 0.830] | 23 unnormalised channels of mixed units (m/s, rad, m, 0–10 counts); **97.3 % of the "target energy" is 4 trigger dims**, 95.6 % of it in 0.057 % of entries. `wbc[7:19]` is **zero-order-held at 24.87 Hz** onto a 30 Hz grid (17.15 % exact frame repeats); repairing it removes 45 % of the effect. λ hit the grid edge (1e5) on 3 of 5 blocks, so the "ceiling" is the maximally-shrunk predictor. And **a clairvoyant predictor delayed one frame scores R² = −0.675** on this target — a metric no modality can score on. |
| grasp flip "blind-unreachable": 28.10 % on HIW vs 70.91 % on GR00T | lens E | **REFUTED.** Dose-response in both directions: GR00T hard-binarised → **27.62 %**; HIW low-passed 0.30 s → **60.80 %**. A zero-parameter anti-repeat-last rule scores **95.12 %** on HIW | The contrast measures teleop-input **discreteness**, not scene richness, and reproduces inside either dataset with scene content fixed. The post-flip-only slice is oracle-conditioned; on the honest all-16-step slice the claimed 47.96 pts of headroom collapse to ~4.9. |
| "GR00T can never do better than ~15 pts MDE" | lens E | **10.9** at a matched 150-episode holdout of the same 402 episodes; **5.9** at matched channel encoding vs HIW's 3.8 | Compared HIW@150 against GR00T@40. Re-splitting PR-01-GRIPPER's own cache: MDE 18.0/16.1/12.3/10.9/8.6 at holdouts of 40/60/100/150/209. The real HIW power gain is ~1.55×, from episode count and two live hands. |
| `observation.state.wbc` dims bit-identical to `action`: 4 | lens E | **11** — dims 0–6 (whole base/torso command block) and 19–22, equal-fraction 1.000000, max\|diff\| exactly 0 over 739 120/739 120 rows | Only dims 7–18 (both EE poses) are measurements. The task brief's *"MEASURED counterpart"* is false for 11 of 23 dims. |
| n = 6 857 (chunks) as the sample size | lens B | **effective episode count 18.8 of 84** by chunk weight, 25.3 by energy weight | HIW holdout episodes contribute 6–889 chunks each; top 5 = 46.7 % of chunks. GR00T's holdout is 21–32/episode, effective 39.5 of 40. The CI survives (micro 0.7949 vs macro-median 0.7954); the stated *n* does not. |
| the M2 gap = 0.436 of task content | implied by lenses A/B | **0.221** | AR(1) velocity-autocorrelation floor accounts for 49–60 % of it. |
| 91 distinct subtask labels | lens A | **97** (3 083 segments, 418 episodes; 14 of them `^move to `, either way) | Counting error only; nothing downstream moves. |

### Deviations from the pre-registration itself

| PR-02 said | actually | consequence |
|---|---|---|
| slice = "the first ~340 episodes" | **418** episodes, 739 120 rows, 6.84 h | Scored slice is not literally the pre-registered one. Verdict unaffected. |
| 11 tasks | **10** with episodes — `building children table` (task_index 10) has **zero** | Per-task table is 10 rows. |
| *"`language_events` carries timestamped subtask labels"* | **Empty.** Arrow type `list<element: null>`, 0 non-empty of 739 120 rows in all three files | The pre-registered locomotion control is **unimplementable as written**. Three lenses silently substituted `language_persistent` (7.38 records/ep, 97 labels, the only column carrying the segmentation) — correct call, undisclosed at the time, disclosed here. The fourth lens could not apply the control at all and its pooled M1/M2 (−2.614/0.822) are therefore PR-02's *secondary*, not its primary, and carry no verdict. |
| — | 213 of 418 episodes are **truncated mid-plan** relative to their own subtask schedule (540 of 3 083 segment starts fall past the end of their episode) | A converter has to cope with it. Does not move M2. |

---

## Where the lenses disagree, and which I believe

**M1/M2 on the arm-joint target: no real disagreement.** Three separate implementations, three split
seeds, land at M2 ∈ [0.761, 0.800] and M1 ∈ [−1.28, −2.13]. I take **lens B as corrected by its
verifier (M2 = 0.7656, M1 = −1.313)** as the headline, for two reasons: its manipulation mask is the
stricter one (a chunk survives only if **all 18 frames it touches** are in a non-`move to` segment,
against lens A's label-at-chunk-start), and its ceiling was re-searched with γ extended *down* to
0.0005 and width up to 8192 — the exact direction that could manufacture a PASS — and M2 went **up**,
not down. Lens A's 0.7917 is the same quantity on a looser mask and a different seed. I do not
average them; I quote the range and use the conservative end.

**Native action space (lens D headline): I do not believe it and it is excluded.** Refuted on four
independent grounds above, the decisive one being that a clairvoyant predictor delayed 33 ms scores
R² = −0.675 on that target. What I *do* keep from lens D is its **arm-joint control on the same
split** (M2 = 0.7995, M1 = −2.125) — a fourth independent replication of the primary — and its
finding that **~90 % of the GR00T→HIW move is the dataset, ~6 % the action space** after the ZOH
repair.

**Grasp flip (lens E headline): I do not believe it and it is excluded.** The 28.10 % vs 70.91 %
contrast is a channel-encoding artefact with a clean two-directional dose-response, and a
zero-parameter rule beats the "ceiling" by 67 points on the same slice. **What survives from lens E
is M3 and only M3**, and M3 is confirmed independently by lens C using the repo's own
`src/wam/evaluation/gripper.py` debounce: 3.837 / 5.971 / 5.249 / 7.670 transitions per episode on
the four `wbc` channels. That clause holds.

**The jerk mechanism: I believe the refutation, and score the clause anyway.** Lenses B and C refute
noise as the cause with four tests. PR-02's clause is written on the *symptom* (excess present at
step 0), not the mechanism. I score it as written and state the mechanism separately — that is
exactly what PR-01 did with clause 4, and inventing a mechanism escape hatch after seeing the number
is the failure mode the pre-registration exists to prevent.

---

## Would this dataset help us?

**Partly, at a cost that is not worth paying at full scale, and the one clause it wins outright we
can win at home for free.**

### What it would actually cost

| | |
|---|---|
| HF repo total | **2.15 TB** |
| `data/` (state parquet, all 23 743 episodes, no video) | **13.2 GB** |
| what this screen used | 240.7 MB, 418 episodes, 1.76 % of the corpus |
| **action space** | native `action` is **23-dim base velocity + torso pose + EE Cartesian + trigger/squeeze**. Ours is `JOINT_DELTA` on 15 canonical joints. **Only the 14 arm joints of `observation.state` map directly.** The native space is unusable as an MSE target without per-channel normalisation (97.3 % of its energy is 4 trigger dims), 11 of its 23 dims are the teleop command echoed into the state, and its EE block is ZOH'd at 24.87 Hz. |
| **mobile base** | 7 pivot dims; **35 % of chunks carry a `move to` label**; even inside manipulation subtasks the **legs carry 1.14× the arms' per-step delta energy**. Our MuJoCo scene has a **welded base**. E2 sim eval does not transfer without a new scene, and part of HIW's blind-unreachable *arm* energy is plausibly base-induced. |
| **no finger joints** | `observation.state`'s 29 joints are 12 leg + 3 waist + 14 arm. **There is no hand DoF anywhere in the dataset.** The grasp channel is the teleoperator's trigger command, bit-identical to `action[19:23]`. 88.5 % of its flips complete in ≤ 1 frame; a physical hand takes ~0.3 s. M3 measures anticipating an operator's finger, not a hand closing. |
| episode geometry | lengths 106–18 130 frames (**100× spread**), median 1 148; 213 of 418 episodes truncated mid-plan; effective independent units ≈ 22 % of nominal episode count |

### What it would actually buy

| | gain | discount |
|---|---|---|
| **M3 alive** | 3.8–7.7 transitions/ep vs our converted **0.000** | **PR-01-FOLLOWUP already showed `--gripper-mapping active-hand` restores 2.015/ep from our own raw source.** Zero download, one CPU pass. HIW is not needed for this. |
| **M2 0.766 vs 0.330** | reproduced 3×, survives length matching, 10/10 tasks, 6 split seeds, outlier trimming, idleness control, an 81-dim blind model | ~49–60 % is velocity autocorrelation; ≥ 20.6 % of the energy is above 2.8 Hz where no smooth visual predictor can reach; and **every other channel of the robot, including the base-velocity command, scores 0.95–0.99 on the same metric** |
| **M1 sign flip** | const-velocity 1.31–1.37× *worse* than holding still, vs 0.56× on ours. The momentum trap that made our headline metric a momentum metric is genuinely absent | the flip itself is filter-dependent (gone under a 300 ms boxcar), though the ≤ 0.45 clause is not |
| **statistical power** | MDE 3.8 pts at a 150-episode holdout | GR00T at the *same* holdout size gives 10.9, and 5.9 at matched channel encoding. Gain ≈ **1.55×**, from episode count and two live hands — not from HIW's richness |

### Recommendation

1. **Do not convert 2 TB.** MIXED licenses nothing at scale, and the strongest counter-evidence is
   internal to HIW: the metric that would justify the spend returns 0.99 on a base-velocity command.
2. **Do the free thing first.** Re-convert our own corpus with `--gripper-mapping active-hand`
   (`PR-01-GRIPPER.md` recommendation 1, still unexecuted). M3 is the one clause HIW clears
   unambiguously and it is the one we can clear at home, with no download and no provenance cost.
3. **If HIW is touched further, bound it by task, not by terabytes.** The only defensible next step
   is a video pilot on **`setting the table`** — tabletop, near-static base, the closest analogue to
   our corpus, M2 = 0.796 [0.757, 0.827] on 9 holdout episodes. That is the cheapest way to answer
   the three things this screen structurally cannot see. It is **not** licensed by a PASS; it is
   licensed by MIXED leaving those questions open and cheap.
4. **Never adopt the 23-dim native space as an MSE target.** Independent of HIW, this is the
   first-order finding for our converter: unnormalised, it optimises the gripper trigger and ignores
   both arms.
5. **If a converter is ever written, `observation.state[15:29]` is the only column that maps.**
   Everything else needs a decision we have not made.

---

## What this does not establish

- **That vision helps.** A dataset where blind fails is a **precondition** for testing AC-07, not
  evidence about it. The same measurement returns 0.990 on a base-velocity command and 0.946 on
  balance-controller leg joints, where a camera is irrelevant.
- **That WAM works, or that the T-16 verdict was wrong.** Nothing here retires any archived number.
  Our GR00T runs stay exactly as scored, on the dataset they were scored on.
- **That the residual energy is reachable by anything.** ≥ 20.6 % of HIW's scored target energy is
  above 2.8 Hz. A perfect smooth visual oracle cannot claim it. That risk is *higher* here than on
  GR00T, not lower.
- **That the M3 channel is a grasp.** No hand DoF exists in the data. It is a teleop trigger, and
  its flips are single-frame steps.

## What this screen structurally could not see

PR-02 named three; all three remain open and the screen produced *negative* evidence on two of them.

1. **Whether the frames are usable by Wan.** No video decoded. Resolution, blur, exposure, camera
   mounting, occlusion, frame/state timestamp sync — all unmeasured. Timestamp sync is a release
   gate in our own PRD.
2. **Whether the mobile base breaks the fixed-base MVP.** PR-02 guessed yes. The screen supports the
   guess without settling it: 35 % of chunks are locomotion, and legs out-move arms even during
   manipulation subtasks.
3. **Whether the action space maps to `JOINT_DELTA` without losing the task.** The screen found
   evidence for *not as an MSE target*, and nothing about whether the arm joints alone preserve the
   task.

Four more, found by the screen and not anticipated by it:

4. **Whether the remaining 98.24 % of the corpus behaves like these 418 episodes.**
5. **Whether the operator's trigger corresponds to a physical grasp** — unanswerable from this
   dataset at all, since it contains no hand measurement to cross-check against.
6. **Anything closed-loop**: task success, safety-layer interaction, sim2real, generalisation.
   Every number here is offline chunk MSE, the metric PR-01 already demoted to a diagnostic.
7. **Whether the 11th task exists in the released data.**

---

## How much this was attacked

Four measurement lenses, each independently adversarially verified — 8 passes, all reproducing the
GR00T archive gate to the digit before quoting an HIW number.

- **2 of 4 headline claims refuted outright** (native action space; grasp flip), both with
  constructed dose-response controls rather than argument.
- **2 survived** (primary arm joints; independent reimplementation), both reproducing **bit-for-bit**
  under independent re-run, and lens B additionally rebuilt from raw parquet by a verifier sharing no
  code — agreeing to 7–8 significant figures on zero-delta and const-velocity.
- **Leakage audits clean on both survivors.** Episode-disjoint folds asserted at runtime; ceiling
  hyperparameters searched only on an inner episode-disjoint split of TRAIN; standardisers fitted on
  TRAIN rows only. The one identity — `dq[start]·dt_s` **is** the previous chunk's last target,
  100.0000 % exact — is present on **both** corpora (23 of 104 dims on HIW, 15 of 32 on GR00T) and is
  the causal backward-difference convention `PR-01-FOLLOWUP.md` §1 already adjudicated. It cannot
  bias the comparison.
- **Nine further confound attacks failed** and are reported here as support, not omitted: outlier
  trimming, per-joint decomposition, encoder-noise estimation, in-sample-oracle generalisation gap
  (0.011 of M2), episode-equal weighting, displacement-matched horizon sweep, DCT frequency banding,
  whole-body common-mode oracle (R² = 0.019), subtask-vocabulary audit.
- **Two knobs that could have manufactured a PASS were opened and did not move it**: the blind model
  was given the full 81-dim body state (M2 −0.010), and the ceiling grid was extended in the
  direction that inflates M2 (M2 went *up* 0.005).

Nothing in the repo was modified, no video was decoded or downloaded, no git operation was
performed, and no allocation was spent.

## Independent check before this file was committed

The four lenses above are subagents, and the two preceding agent reports in this investigation each
contained an error or an overreach. So the load-bearing numbers were re-measured once more, by hand,
from raw parquet, importing nothing from any lens and nothing from the repo's bench scripts:

| quantity | re-measured | as reported | |
|---|---:|---:|---|
| slice geometry | 418 episodes, 739 120 rows | 418, 739 120 | ✅ |
| const-velocity / zero-delta, HIW arm, manipulation-only | **1.360** | 1.31–1.37 | ✅ |
| const-velocity / zero-delta, GR00T arm | 0.565 | 0.560 (archive) | ✅ |
| M2, HIW arm, whole chunk | **0.7914** | 0.7917 (lens A) | ✅ |
| M2, HIW arm, steps 8–15 | 0.9086 | 0.9154 | ✅ |
| step-0 excess over GR00T | **+0.157** | +0.17 … +0.22 | ✅ same sign, control fires |
| M2, base-velocity **command** | **0.996** | 0.990 | ✅ |
| M2, commanded EE pose | 0.942 | 0.948 | ✅ |
| arm joints' rank among channels | **bottom of the pack** | bottom of the pack | ✅ |
| `wbc` dims bit-identical to `action` | **11 of 23** — dims 0–6, 19–22 | 11, dims 0–6, 19–22 | ✅ |
| `language_events` | Arrow type `list<element: null>` | empty | ✅ |

The step-0 excess came out smaller than reported (+0.157 vs +0.17…+0.22) in the direction that
*weakens* the control, and the reason is visible in the same run: this re-measurement's blind ceiling
is the weaker one (GR00T whole-chunk M2 0.427 against the archive's 0.333), which inflates GR00T's
step-0 baseline and shrinks the gap. A stronger ceiling widens it. The clause fires on either.

> **One error in this re-measurement, mine, left visible.** The first pass returned
> `1 − ss_res/ss_tot` under the name M2 — but that expression *is* R², so every number was the
> complement of what it was labelled. It made the arm channel look like the *best*-predicted on the
> robot (0.208) rather than the worst, which would have inverted the central argument of the
> recommendation. Inverted correctly, the table above reproduces the lenses. The tell was that the
> same code was implausibly weak on GR00T and implausibly strong on HIW at once — an asymmetry no
> real effect produces, and the reason a single-corpus check would not have caught it.
