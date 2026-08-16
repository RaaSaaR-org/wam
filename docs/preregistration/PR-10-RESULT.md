# PR-10 result — **P (partial)**: the delay is real, and it is 8.7 % of the problem

Ran 2026-08-16 on this workstation, CPU only. Pre-registration:
`PR-10-anchor-delay-sweep.md`, rule `PR10_RULE_V1` in `scripts/sweep_t39_anchor.py`.
Follows T-39 / PR-07, which reported `VOID (labels)`. Artifacts:
`runs/t39-baseline-seed0/pr10-anchor-sweep/sweep.json` and one full bench report per cell.
**Zero GPU-hours, zero billing, nothing submitted.** 19 evaluations, about three minutes.

**Verdict: P — partial**, on branch 4 of `PR10_RULE_V1` §3: `k_best = −2` gains **+29.75 pp**
(≥ `MATERIAL_FLOOR_PP = 10.0`) without clearing L1.

> ## This experiment was run twice, independently, on the same afternoon
>
> A peer session pre-registered and ran the same sweep as **T-44** — different driver
> (`scripts/sweep_label_anchoring.py`), different rule (`T44_RULE_V1`), different chunk set,
> different verdict vocabulary — and **neither session knew about the other** until both had
> pushed. Their result is `PR-10-RESULT-T-44.md`. **Both claimed the pre-registration number
> PR-10, and that collision needs a human decision**: rules here are versioned and never edited in
> place, so neither document can be renumbered by whoever notices second.
>
> **The duplicated effort bought an unplanned independent replication, and it holds on every
> comparable quantity:**
>
> | | this run | T-44 | agree? |
> |---|---|---|---|
> | best `k` on **L1** | **−2** | **−2**, on both held-out halves | **yes** |
> | best `k` on **L2** | **−1** | **−1**, on both halves | **yes** |
> | gain at best `k` | **+29.75 pp** | **+28.81 / +30.35 pp** | **yes**, within 1.5 pp |
> | any cell clears L1 | no | no | **yes** |
> | `smoothness_ratio`, `k = 0` → best | 8.28 → 7.70 | 6.21 → 5.67 | **yes**, ~7–9 % of the excess |
> | `horizon_ratio` under the shift | rises | rises | **yes** |
> | bridge to PR-07's −359.41 | bit-identical | −359.4078, drift +0.002 pp | **yes** |
> | verdict | **P (partial)** | **J** | same finding, two vocabularies |
>
> **The verdict letters differ only in the rules' wording, not in the finding.** T-44's own result
> document says so directly: its J branch fired on the correct condition but was worded as "the
> commanded and executed spaces are not a shifted copy of one another", which the data contradicts —
> and it records that "the accurate statement is the peer's P branch". Both documents conclude: a
> real, material, replicated shift of about two control steps, worth roughly 29 pp, that does not
> come close to clearing the bar.
>
> **Why the headline percentages differ (8.7 % here, 11 % there) — it is the denominator, not the
> measurement.** The absolute gain replicates to within 1.5 pp. T-44 splits the holdout into two
> disjoint halves and scores 474/486 chunks against each half's own `d = 0` (L1 −253.70 on half A);
> this run pools all 40 episodes into 992 chunks with common support across the grid (L1 −342.24).
> Same numerator, different baseline deficit. Neither fraction is wrong and neither should be quoted
> without its chunk set.
>
> **Each design caught something the other could not.** T-44's held-out half turns `k = −2` from a
> grid maximum into a *replicated* one, and it scores both bench specs. This run's co-shifted
> control (variant B, flat at 17 pp against 346) is what rules out the peak being a property of our
> own conversion's time base rather than the robot's — T-44 records that it has no such control and
> does not claim it. **Read together they are stronger than either alone**, which is the only good
> thing to say about having spent two sessions on one experiment.

## Variant A — the command does lead the state, by about two control steps

| `k` | ms | `skill_vs_repeat_pct` (**L1**) | `ci_…` (**L2**) | `mse` | `horizon_ratio` | `smoothness_ratio` |
|---:|---:|---:|---:|---|---:|---:|
| −4 | −133 | −370.88 | −185.83 | 4.495e-05 | 0.00609 | 8.68 |
| −3 | −100 | −332.27 | −126.48 | 4.127e-05 | 0.00573 | 7.98 |
| **−2** | **−67** | **−312.49** | −87.79 | 3.938e-05 | 0.00551 | **7.70** |
| −1 | −33 | −317.57 | **−78.85** | 3.986e-05 | 0.00526 | 7.81 |
| 0 | 0 | −342.24 | −93.42 | 4.222e-05 | 0.00453 | 8.28 |
| +1 | +33 | −385.65 | −131.71 | 4.636e-05 | 0.00383 | 9.17 |
| +2 | +67 | −458.94 | −207.92 | 5.336e-05 | 0.00420 | 10.46 |
| +3 | +100 | −549.35 | −305.67 | 6.199e-05 | 0.00402 | 12.11 |
| +4 | +133 | −659.03 | −429.51 | 7.246e-05 | 0.00389 | 14.16 |

992 chunks per cell, 40 holdout episodes, identical chunk set throughout (PR-10 §2's margin).

**This is a real, single-peaked, interior optimum**, and that is worth saying plainly because it is
the one thing in this grid that could not have been an artifact. The curve falls monotonically from
both edges to a minimum at `k = −2` and rises monotonically away from it, across a span of **346.5
pp**. The optimum is inside the grid, so it is bounded on both sides and PR-10 §2's "unbounded on an
edge" clause is not invoked. Read as physics: **relative to the convention `commanded_to_chunk`
already uses, the command leads the executed state by roughly two further control steps — about
67 ms** on a 30 Hz corpus. That is an entirely ordinary transport lag for a position-controlled
humanoid arm.

**One to two steps, not exactly two.** `k = −1` sits 5.08 pp behind `k = −2` on L1, and on **L2 — the
task-critical subset — the optimum is `k = −1`, not `k = −2`** (−78.85 against −87.79). The two gates
disagree about the last step. The honest statement is *between one and two steps*; anyone re-running
G0b at a single corrected offset should know the choice is not sharp, and should register which one
before running it.

## Why it is nonetheless **P** and not **D**

The deficit at `k = 0` is 342.24 pp. The best offset in the grid recovers **29.75 pp of it — 8.7 %.**

**Every one of the nine cells is below L0.** Not one clears L1, none clears L2, and the best-aligned
command is still **4.1× worse than repeating the last action** (and 2.3× worse than holding still:
`skill_vs_zero_pct −132.26` at `k = −2`). Re-anchoring is a genuine fix for a genuine defect and it
does not come close to producing a label space that clears our bar.

**Both secondary readings say the same thing.** PR-10 §5 registered in advance that a shift should
flatten `horizon_ratio` toward 1.0 by moving error out of the chunk's first step. **It does not, in
any amount that matters.** `horizon_ratio` moves from 0.00453 at `k = 0` to 0.00551 at `k = −2` —
*toward* 1.0, by one part in a thousand of the distance it would have to travel, and it stays two
orders of magnitude below its own gate of 4 at every delay in the window. The first-step
concentration T-39 measured survives every anchoring this grid admits. And `smoothness_ratio` at the
optimum is **7.70**, down from 8.28 and still **3.9× its gate of 2** — a shift re-indexes a signal,
it does not smooth one. Optimal alignment leaves the command nearly as jerky as it was.

> **Correction, 2026-08-16.** This paragraph first read that `horizon_ratio` is "further from 1.0 at
> `k = −2` (0.00551) than at `k = 0` (0.00453)" and that the registered prediction therefore failed
> by moving the wrong way. That inverts its own numbers — 0.00551 is nearer 1.0, not further — and
> the error is repeated in commit `bacf513`'s message, which cannot be amended after pushing. Caught
> by the peer session's independent replication (see below), not by me. The direction was wrong; the
> conclusion the paragraph draws is not, and is restated above in terms of magnitude, which is what
> should have carried it in the first place.

So: a shift exists, it is measurable, it is worth fixing — and underneath it the commanded stream
still carries content the executed trajectory does not. That is branch 4, and PR-10 §4 wrote down
what it licenses before the grid existed:

> **P (partial).** Licenses the same defect report, and additionally establishes that a shift is
> *not the whole story* — re-anchoring alone will not produce a label space that clears the bar.
> The project's next question becomes what the residual content is, not which model to try.

## Variant B — the control did its job, which was to say nothing

Predicted flat, and flat: **17.19 pp** of total span against variant A's **346.55 pp**, a factor of
20. Its nominal best is `k = +1` at −341.70 against `k = 0`'s −342.24 — a **0.54 pp** difference on a
grid where the real signal is 29.75 pp, which is noise and is reported as noise rather than as a
finding. `raw_anchor_indices`'s exact-timestamp refusal is doing what its docstring claims: **our
conversion's time base is not offset against the raw parquet.** The lag is the robot's, not ours.

This is the cell that makes variant A's peak interpretable. A co-shifted control that had *also*
peaked at −2 would have meant the sweep was finding a property of episode structure rather than of
the command-state relationship, and the whole grid would have been unreadable.

## Plumbing check (PR-10 §6)

With no sweep flags at all, on the full 1040 chunks:

```
mse 4.19791e-05    vs zero −157.11%    vs repeat −359.41%    ci −102.54%
```

Bit-identical to the archived cluster result on all four figures. **As pre-registered in §2, the
sweep's own `k = 0` cell is −342.24, not −359.41** — it is scored on 992 chunks rather than 1040
because the margin reserves common support. The grid's baseline is its own centre cell; the two
numbers are not interchangeable and −342.24 must never be quoted as T-39's result.

## What this does not license

- **Nothing about GR00T.** The policy arm did not run in T-39 and did not run here. PR-07 §6's
  prohibition stands untouched by every number above.
- **No training run, no generation.** The gate in `CLAUDE.md` and `subprojects/README.md` is the
  project owner's to release, and P is not a release.
- **No lifting of T-39's VOID.** That verdict was recorded against `T39_RULE_V1` on a specific arm.
  A corrected G0b is a new run and a new result document.
- **No attribution of the lag.** `k_best ≠ 0` is equally consistent with the corpus's controller
  lagging, with the corpus timestamping the far end of a control interval, and with our conversion
  dropping a step somewhere `raw_anchor_indices` cannot see. The sweep measures the offset; PR-10 §9
  said in advance that it does not attribute it.
- **Nothing about the other twelve `G1_Dex3_*` corpora** (T-043), and **nothing about grasping** —
  `gripper_accuracy` was withheld by the scorer on all 19 cells.

## What it does establish

**"The adapter is mis-anchored" is retired as a sufficient explanation.** It was the cheapest
available account of the fourteen recorded negatives, and it would have been re-proposed
indefinitely. It is now measured: worth 8.7 % of one arm's deficit, and the residual is not a
shift. The live question for this project is what the commanded stream contains that the executed
trajectory does not — not which backbone to try next, and not more data.
