# PR-10 — Is the label/command mismatch a *timing convention*, or is it the space?

Pre-registered **2026-08-16**, after PR-07-RESULT and **before any sweep is run**. Task **T-44**.
Rule **`T44_RULE_V1`**, fixed in §5 of this file and nowhere else. Zero GPU-hours: this is an
offline re-score of artifacts that already exist.

## 1. What PR-07 left on the table, exactly

T-39 returned **`VOID (labels)`**. Two numbers, both on the 40-episode holdout under both bench
specs:

| arm | `skill_vs_repeat_pct` (**L1**) | what it proves |
|---|---|---|
| `oracle_state` — future **executed** states | **+100.00**, `mse 0.0` | our adapter, joint order, delta anchoring and gripper synergy are **correct** |
| `oracle_action` — the corpus's own **commanded** column | **−359.41** | the corpus's own ground truth fails our bar |

PR-07-RESULT then measured the shape of the failure and, to its credit, refused to explain it:

> **Stated as measurement, not as mechanism:** the three numbers above are measured. The reading
> that the command leads the executed state by roughly one control step, with the remainder being
> high-frequency command jitter that the arm's own dynamics filter out, is an *interpretation*
> consistent with them and is not established here. Distinguishing the two requires a delay sweep
> over the anchoring convention, which is follow-up work and is not this document.

**This is that document.** The three measured numbers it has to account for are `mse 4.20e-05`
(absolute agreement is excellent), `horizon_ratio 0.0044` (last-step error is 1/227th of first-step
error, so essentially all disagreement sits in the chunk's **first** step) and
`smoothness_ratio 8.52` (the command carries 8.5× the jerk of the executed trajectory).

A pure constant lag predicts exactly that first-step concentration. So does nothing else obvious —
which is why the interpretation is tempting, and why it needs a gate written before the numbers
exist rather than after.

## 2. This does **not** re-open a settled test, and the distinction is the whole design

`scripts/eval_t39_baseline.py:208` defines the adapter, and its docstring pins the convention:

> the command issued at `s+t` is what produces the state at `s+t+1`. So the commanded displacement
> over step `s+t` is `action[s+t] - q[s+t]`, and a dataset with perfect tracking
> (`action[i] == q[i+1]`) makes the two identical.

`tests/test_t39_baseline.py` kills the three plausible mis-anchorings — `action[t+1] - q[t]`,
`action[t] - q[t+1]`, and the first-difference `action[t+1] - action[t]` — with mutants, because
each produces finite, plausible, wrong numbers no shape assertion catches.

**Those tests answer "what does our convention mean". This experiment asks a different question:
"does this corpus satisfy the convention's premise".** The premise is the parenthesis above —
`action[i] == q[i+1]`, perfect tracking within one step. If this recording's controller actually
tracks with a lag of `d` steps, then the convention is still correctly *implemented* and the data
still violates what it assumes. Reading a non-zero `d*` as "the mutation tests were wrong" would be
a misreading, and it is stated here so that it cannot be made after the fact.

Concretely: `d = +1` in the sweep below evaluates the same index expression as one of the killed
mutants. **It is not a candidate convention here.** It is a probe of the corpus's tracking latency,
and §6 fixes in advance what may and may not be concluded from it.

## 3. The manipulation — one variable, and nothing else moves

For each integer delay `d`, the commanded chunk is built exactly as `commanded_to_chunk` builds it
today, with the **source index shifted and nothing else**:

```
targets[t]        = canonical_q(commanded[s + t + d]) - q_start(t)
gripper_target[t] = gripper channel of commanded[s + t + d]
```

`q_start(0) = anchor_state` and `q_start(t>0)` = the position the previous command asked for. The
chaining is untouched; the chunk length is untouched; the anchor is untouched. `d = 0` **must**
reproduce PR-07-RESULT's `−359.41` to the recorded precision, and §5 makes that a gate rather than
a hope.

**Sweep range: `d ∈ {−4 … +4}`**, nine values, ±133 ms at the corpus's 30 fps. Fixed here. It is
centred on zero and symmetric because a lead and a lag are equally admissible answers and a
one-sided range would smuggle in the conclusion.

**The chunk set is the intersection over the whole sweep, not per-`d`.** A shifted index runs off
the end of the episode for `d > 0` and off the front for `d < 0`, so the eligible chunks differ by
`d` — and nine scores over nine different chunk sets are not comparable. The **first and last chunk
of every episode are dropped for every arm and every `d`**, uniformly, which covers `|d| ≤ 4` at
chunk length 16 with room to spare. The retained chunk count will therefore be **below** PR-07's
1 040 and is recorded rather than predicted. Consequently **no number in this document is directly
comparable to PR-07-RESULT's table**, and `d = 0` here is the only bridge between the two.

## 4. The trap this design exists to avoid

Nine values and "take the best" is a garden of forking paths: with nine draws, the maximum of a
noisy statistic is biased upward, and a `d*` chosen that way would be a description of this
holdout's noise dressed as a property of the controller.

So `d*` is **fitted and confirmed on disjoint halves**. The 40 holdout episodes are split into
**A** (even index in the committed `configs/splits/t18_holdout_episodes.txt` order) and **B** (odd),
20 each — a deterministic function of a file already in git, not a fresh seeded draw, so the split
is reviewable and cannot be re-rolled. `d*` is chosen on **A alone**. Every verdict-bearing number
is then read on **B alone**, at that one `d*`, with no further search.

Both halves' full sweeps are recorded regardless, because a `d*` that is obvious on A and absent on
B is itself the finding.

## 5. Gates — `T44_RULE_V1`

The bar is WAM-Bench's existing ladder, unchanged, and the one borrowed constant is
`MATERIAL_FLOOR_PP = 10.0` from `I8_RULE_V3` — taken rather than coined, for the same reason PR-07
took it, so that the choice of floor cannot be the finding.

- **L1** `skill_vs_repeat_pct > 0` on the retained chunks
- **L2** `ci_skill_vs_repeat_pct > 0` on the task-critical subset

**G0 · INVALID (runs first, can stop everything).** Two identity checks, both on the retained chunk
set:

1. `oracle_state` at `d = 0` must still reach `skill_vs_repeat_pct ≥ 90 %`. Below that the harness
   changed under us and no verdict is issued — the fix is a code fix, not a threshold change.
2. `oracle_action` at `d = 0` must land within **±0.5 pp** of PR-07-RESULT's `−359.41` **when
   scored on the full 1 040 chunks**, which is run once as a bridge before the trimmed set is
   adopted. A drift wider than that means this is not the same measurement and nothing may be
   compared to PR-07.

**Which conclusion is expensive here, and therefore which one is held to the margin.** **T is the
expensive verdict**, and this is the reverse of PR-07. A confirmed timing defect licenses
re-labelling the entire corpus and re-scoring fourteen recorded experiments against a moved ruler;
`docs/benchmark.md` would have to be re-read end to end. So **T** carries the material margin and
the held-out confirmation, and **J** — "it is not a delay" — is the cheap default that changes
nothing and starts no work.

**The verdicts**, read on half **B**, at the `d*` fitted on half **A**:

| | condition | reading |
|---|---|---|
| **T** | `d* ≠ 0`, **B** reaches **L1** at `d*`, **and** B's gain over its own `d = 0` is `≥ MATERIAL_FLOOR_PP` | the mismatch is a timing convention. Our labels are anchored `d*` steps off this corpus's controller |
| **J** | **no** `d` in the range reaches L1 on **A** | it is not a delay. The command and executed spaces differ in a way a shift cannot fix, and the next question is the 8.5× jerk, not the anchor |
| **E** | `d*` lands on a **range endpoint** (`|d*| = 4`) | the optimum may lie outside the window. **Nothing is concluded.** The range extends once, to `±8`, pre-registered here, and the verdict is re-read under the same rule. There is no second extension |
| **I** | anything else — in particular L1 cleared on A but not on B, or a gain under the floor | indeterminate. Recorded, nothing licensed, no relabelling |

**`d* = 0` cannot produce T.** If the best delay is the one we already use, the sweep has found
nothing and the verdict is **J** or **I** by the table above, never T.

**Recorded regardless of verdict:** the full nine-value curve on A and on B under both bench specs;
`horizon_ratio` and `smoothness_ratio` at `d = 0` and at `d*`; the retained chunk count; the
`d = 0` bridge score on the full 1 040; `dataset_snapshot_ref`; and the wall time.

## 6. Reading the outcome — decided before the numbers exist

- **T** licenses a defect report against the label pipeline naming a specific `d*`, and licenses
  *proposing* a re-labelling. It **does not** license re-labelling the corpus inside this task, and
  it does not retro-validate any of the fourteen negatives — those were scored against the old
  anchor and stay scored against it until something re-runs them.
- **J** is the more consequential outcome for the project even though it starts less work: it says
  the commanded and executed spaces are not a shifted copy of one another, and that
  `smoothness_ratio 8.52` is the object of study. PR-04's collection spec — *what kind* of data —
  becomes the live question rather than any anchoring fix.
- **No outcome licenses a statement about GR00T, or about any policy.** PR-07 §6 forbids it because
  the policy arm never ran, and nothing here runs one. This experiment scores two oracles against
  each other; there is no model in it.
- **No outcome unblocks training.** The standing question of whether a training run may start after
  a `VOID` gate is the project owner's, and a follow-up experiment does not answer it.

## 7. Cost

Zero GPU-hours. CPU only, on artifacts already on disk: the 402-episode converted corpus, the
committed holdout split, and the source parquet. Expected minutes. It needs no cluster, no
allocation and no download, which is most of why it is the right next experiment rather than
another model.

## 8. What must exist before this runs

1. `scripts/sweep_label_anchoring.py` — the sweep driver. It must reuse `commanded_to_chunk`,
   `build_eval_pairs` and the bench specs **by import from `scripts/eval_t39_baseline.py`**, not by
   re-implementation: a sweep run through a second copy of the adapter measures the copy.
2. Tests, in the shape `tests/test_t39_baseline.py` set: at minimum a mutant that shifts the anchor
   instead of the command index, and one that trims the chunk set per-`d` instead of by the
   intersection. Both produce plausible wrong curves that no assertion on shape or range catches.
3. The `d = 0` bridge (§5 G0.2) run and recorded **before** the trimmed set is adopted.

## 9. What this cannot answer

- **Whether a policy could learn the shifted labels.** It scores oracles. A corpus whose commanded
  column becomes predictable under `d*` is a necessary condition for that, not a sufficient one.
- **Whether the lag is constant across episodes, joints or speeds.** One scalar `d` per sweep is
  assumed by construction. A per-joint or velocity-dependent lag would show up here as a partial,
  unsatisfying improvement — verdict **I** — and would need its own design.
- **The gripper.** `gripper_accuracy` was withheld by the scorer on both PR-07 arms because the
  relabelled channel is degenerate (peak-to-peak `0.1196`, `0.00` debounced transitions/episode,
  85.3 % majority class). It stays withheld here. That the raw commanded channel carries 2.04
  transitions/episode is the same mismatch seen on one channel, and it is **evidence for running
  this experiment, not a result of it**. `scripts/audit_gripper.py` runs before any grasping claim.
- **Anything about `docs/benchmark.md`'s validity.** PR-07-RESULT already bounded those numbers as
  statements about predicting what the robot *did*. This experiment does not widen or narrow that.
