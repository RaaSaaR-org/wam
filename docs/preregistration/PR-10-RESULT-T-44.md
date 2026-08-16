# PR-10 (T-44) result — **J**: there is a real timing offset, and it is not the defect

Ran 2026-08-16 on the workstation, CPU only, **zero GPU-hours**. Pre-registration:
`PR-10-label-anchoring-delay-sweep.md`, rule `T44_RULE_V1`, committed in `d18666e` before the
driver existed; driver `scripts/sweep_label_anchoring.py` and its ten tests in `ffe7ec4`, before
any curve existed. Task **T-44**. Artifact `runs/t44-anchoring-sweep/sweep.json`. Dataset
`datasets/gr00t-apple-full`, source `GR00T-N1.7-AppleToPlate`, holdout
`configs/splits/t18_holdout_episodes.txt`, gripper mapping `legacy`, chunk length 16.

> ## Read this first: two sessions ran this experiment, independently, on the same afternoon
>
> A peer session pre-registered `PR-10-anchor-delay-sweep.md` (rule `PR10_RULE_V1`, driver
> `scripts/sweep_t39_anchor.py`) and ran it concurrently with this one. **Neither knew about the
> other**; both took PR-07-RESULT's closing paragraph as the obvious next experiment, and both
> claimed the number **PR-10**. Their result is in `PR-10-RESULT.md`; this file is deliberately
> named apart rather than overwriting it, and **the duplicate PR number needs a human decision**
> — rules here are versioned and never edited in place, so neither can simply be renumbered by
> whoever notices second.
>
> **The waste is real and so is the compensation: this is an unplanned independent replication,
> and it holds.** Two drivers, two chunk sets, two rules, two verdict vocabularies, same answer.

## G0 — both gates passed, and the second one is the load-bearing one

| gate | requirement | measured |
|---|---|---|
| **G0.1** `oracle_state` at `d = 0`, trimmed set | `skill_vs_repeat_pct ≥ 90 %` | **+100.00 %**, 960 chunks, L4 *moves-like-a-demo* |
| **G0.2** bridge: `oracle_action` at `d = 0`, **full 1 040 chunks** | within ±0.5 pp of PR-07's `−359.41` | **−359.4078 %**, drift **+0.002 pp** |

The bridge reproduces PR-07-RESULT to four decimal places, and its two diagnostics land on PR-07's
as well: `smoothness_ratio 8.5175` against the reported 8.52, `horizon_ratio 0.004413` against
0.0044. **This is the same measurement**, run through a different driver on a re-converted corpus,
so everything below is comparable to T-39 through that one point.

## The curve — nine delays, two disjoint halves, both bench specs

`skill_vs_repeat_pct` (**L1**) and `ci_skill_vs_repeat_pct` (**L2**), spec 0.1.0, on the trimmed
set (474 chunks in A, 486 in B, of 960):

| `d` | ms | A · L1 | B · L1 | A · L2 | B · L2 | A · smooth |
|---:|---:|---:|---:|---:|---:|---:|
| −4 | −133 | −282.68 | −439.54 | −131.24 | −235.10 | 6.64 |
| −3 | −100 | −244.20 | −400.32 | −75.25 | −173.45 | 5.95 |
| **−2** | **−67** | **−224.89** | **−379.68** | −39.14 | −132.68 | 5.67 |
| −1 | −33 | −229.83 | −384.69 | **−30.60** | **−123.56** | 5.77 |
| 0 | 0 | −253.70 | −410.03 | −44.27 | −139.04 | 6.21 |
| +1 | +33 | −295.55 | −454.96 | −79.70 | −179.97 | 7.04 |
| +2 | +67 | −367.69 | −529.50 | −152.72 | −259.18 | 8.25 |
| +3 | +100 | −456.03 | −622.23 | −244.91 | −362.24 | 9.73 |
| +4 | +133 | −564.26 | −733.64 | −363.78 | −490.41 | 11.63 |

**Spec 0.2.0 gives L1 and L2 identical to nine decimals at every `d`.** The two specs differ only
in the two-sided L4 smoothness band, which nothing here reaches. Both are in `sweep.json` because
PR-10 §5 said both, not because they were expected to differ.

**The sweep is well formed.** Smooth and unimodal in `d` on both halves, the optimum interior (so
verdict **E** does not apply and the window was wide enough), and **A and B pick the same `d*`
independently** — `−2` on L1, `−1` on L2, on both halves. The A/B design existed to catch a `d*`
that is an artifact of one half's noise. It did not catch one; it confirmed the shape.

## What is actually there: a real offset that closes 11 % of the gap

| | A | B |
|---|---:|---:|
| L1 at `d = 0` | −253.70 | −410.03 |
| L1 at `d* = −2` | −224.89 | −379.68 |
| **gain** | **+28.81 pp** | **+30.35 pp** |
| L2 gain at `d = −1` | +13.66 pp | +15.48 pp |

**The gain is material by the borrowed `MATERIAL_FLOOR_PP = 10.0` and it replicates out-of-sample.**
Paired with an L1 crossing, that is exactly what verdict **T** required. So the constant-lag reading
PR-07-RESULT declined to establish is **partially confirmed and decisively insufficient**: shifting
the command two steps earlier is worth ~29 pp against a ~254 pp deficit — about **11 %** of the gap
— and leaves the arm 3.2× worse than repeating the last action.

**Stated as measurement, not as mechanism**, the same discipline PR-07-RESULT used. `d* = −2` means
the best fit differences the executed state against a command issued two steps *earlier* than the
convention assumes. That this is actuator lag — ~67 ms at 30 fps, on top of the one step the
convention already grants — is an interpretation consistent with the curve and is **not** established
here. Two observations sit against a clean scalar lag: L1 and L2 disagree about `d*` (−2 vs −1), and
no single integer shift should be optimal for two different chunk populations if the lag were one
number.

## The jerk barely moves, which is what makes J the reading

| | `d = 0` | `d* = −2` | L4 gate |
|---|---:|---:|---:|
| `smoothness_ratio` A | 6.21 | 5.67 | ≤ 2.0 |
| `smoothness_ratio` B | 10.18 | 9.55 | ≤ 2.0 |
| `horizon_ratio` A | 0.0061 | 0.0077 | ≤ 4 |

Re-anchoring removes about **9 %** of the excess jerk and leaves the command 5.7× (A) and 9.6× (B)
jerkier than the trajectory the robot executed. `horizon_ratio` rises slightly and stays three
orders of magnitude below its gate, so the disagreement remains concentrated in the chunk's first
step at every delay in the window.

**A timing offset cannot produce that.** A pure shift re-indexes a signal; it does not smooth one.
The command carries high-frequency content the executed trajectory does not, and that content
survives every anchoring the window admits. That is the substance of **J**.

## The replication, cell by cell

The peer's sweep used a different chunk set (992, common support across the grid, all 40 episodes
pooled) and no A/B split, so absolute levels are not comparable — only the shape and the deltas
are, which is the point.

| | this run (T-44) | peer (`PR-10-RESULT.md`) | agree? |
|---|---|---|---|
| chunks per cell | 474 (A) / 486 (B) | 992 | different by design |
| L1 at `d = 0` | −253.70 / −410.03 | −342.24 | n/a — different sets |
| **best `d` on L1** | **−2**, on both halves | **−2** | **yes** |
| **gain at best `d`** | **+28.81 / +30.35 pp** | **+29.75 pp** | **yes** |
| **best `d` on L2** | **−1**, on both halves | **−1** | **yes** |
| any cell clears L1 | no | no | **yes** |
| `smoothness_ratio` at `d = 0` → best | 6.21 → 5.67 | 8.28 → 7.70 | **yes**, ~7–9 % of the excess |
| `horizon_ratio` under the shift | rises | rises | **yes** |
| bridge to PR-07's −359.41 | −359.4078 | bit-identical | **yes** |
| verdict | **J** | **P (partial)** | same finding, two vocabularies |

Two things the peer's design has that this one does not, and they are both good: a **co-shifted
control variant** (shift command *and* anchor together; predicted flat, came out flat at 17.19 pp
span against 346.55 pp, which is what makes the real peak interpretable rather than a property of
episode structure), and a **`ms` column**. Two things this one has that the peer's does not: the
**held-out half**, which is what turns `d* = −2` from a grid maximum into a replicated one, and
**both bench specs**.

## Where my own pre-registration was written badly, and I am not fixing it

`T44_RULE_V1`'s **condition** for J fired correctly — no delay in the window clears L1 on half A.
Its **prose** does not survive the data. I wrote J as *"the commanded and executed spaces are not a
shifted copy of one another, so the anchor is not the defect"*, and the sweep found a real,
material, out-of-sample-replicated shift. The accurate statement is the peer's P branch: a shift
exists, is worth fixing, and is not the whole story.

The rule is not amended and the verdict letter stands as `J`, because a gate rewritten after seeing
its output is not a gate. This paragraph is the correction, in the result document where it belongs.
**A rule can pick the right branch for the wrong stated reason, and saying so is cheaper than
pretending the wording was fine.**

## One thing in the peer's document to check before it is committed

`PR-10-RESULT.md` reads *"`horizon_ratio` is further from 1.0 at `k = −2` (0.00551) than at
`k = 0` (0.00453)"*. Both runs agree the value **rises** under the shift, but 0.00551 is *nearer*
1.0 than 0.00453, not further — the sentence inverts its own numbers. The conclusion it supports is
unaffected: `horizon_ratio` stays two to three orders of magnitude below its gate at every delay, so
the shift does not fix the first-step concentration either way. Flagged rather than edited: that
file is a peer's uncommitted work and its author should make the call.

## An asymmetry worth recording, which changes no verdict

Half B is uniformly ~155 pp worse than half A at every delay, and 64 % jerkier at `d = 0` (10.18 vs
6.21). The halves are an even/odd split of the committed holdout file, so this is a property of that
ordering, not of anything chosen here. It affects nothing — `T44_RULE_V1` compares B against **B's
own** `d = 0`, never against A — and is recorded because a reader comparing the columns directly
would mis-read it as instability in the method.

## What this licenses

**J is the cheap verdict by construction and it starts no relabelling.** PR-10 §5 put the material
margin and the held-out confirmation on **T** precisely because T would have licensed re-labelling
the corpus and re-reading `docs/benchmark.md` end to end. None of that is licensed.

What it does say is where the next question is: the commanded and executed spaces differ in
*content*, not in *alignment*, so `smoothness_ratio` is the object of study and **PR-04's collection
spec — what *kind* of data — becomes the live question** rather than any anchoring fix.

Per PR-10 §6, unchanged by the outcome:

- **No statement about GR00T or any policy.** No model was trained, loaded or consulted; this
  scored two oracles against each other.
- **No outcome unblocks training** after T-39's `VOID`. That remains the project owner's call.
- **None of the fourteen negatives is retro-validated.** They were scored against the `d = 0` anchor
  and stay scored against it.
- **No attribution of the offset.** `d* ≠ 0` is equally consistent with the corpus's controller
  lagging, with its timestamping convention, and with our conversion. The peer's flat co-shifted
  control is evidence against the last of those; this run has no such control and does not claim it.

## What this cannot answer

Everything PR-10 §9 already excludes, and one thing the curve makes concrete: **a per-joint or
velocity-dependent lag is not ruled out.** §9 predicted it would appear as *"a partial, unsatisfying
improvement"*, and that is what appeared — 29 pp of one, with L1 and L2 disagreeing about where.
Separating "one scalar lag plus unrelated jitter" from "a lag that varies across joints or speeds"
needs a design this sweep does not have, and inventing one after seeing this curve is what the
pre-registration exists to prevent. Named as follow-up, not answered.

The gripper stays withheld by the scorer on every arm, for the reason PR-07-RESULT records: the
relabelled channel is degenerate. `scripts/audit_gripper.py` runs before any grasping claim.
