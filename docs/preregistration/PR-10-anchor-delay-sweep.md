# PR-10 — Is the label-space mismatch a *shift*, or is it *content*?

Pre-registered 2026-08-16, **before any offset other than `k = 0` has been evaluated**. The one
number already in hand is `k = 0`, and it is not a new measurement: it is T-39's archived
`oracle_action` result, re-run locally to prove the plumbing (§6). Every other cell of the grid
below is unmeasured at the time of writing.

Follows: **T-39 / PR-07**, which reported `VOID (labels)` — `docs/preregistration/PR-07-RESULT.md`.
Venue: **this workstation, CPU only.** No cluster, no GPU-hours, no submission, no download.
Executable rule: `scripts/sweep_t39_anchor.py`, rule `PR10_RULE_V1`.

---

## 1. The question T-39 left open, stated as the only thing it can be

T-39 measured three things and interpreted none of them:

| | value | gate | what it says |
|---|---|---|---|
| `mse` (absolute) | `4.20e-05` | — | the command sits almost exactly on the state |
| `horizon_ratio` | **0.0044** | ≤ 4 | last-step error is **1/227th** of first-step error |
| `smoothness_ratio` | **8.52** | ≤ 2 | the command is **8.5× jerkier** than the demonstration |

PR-07-RESULT §"The mechanism" wrote down, explicitly as interpretation rather than finding:

> the reading that the command leads the executed state by roughly one control step, with the
> remainder being high-frequency command jitter that the arm's own dynamics filter out, is an
> *interpretation* consistent with them and is not established here. Distinguishing the two
> requires a delay sweep over the anchoring convention, which is follow-up work and is not this
> document.

This is that document. It exists because the two readings lead to **different projects**:

- **A shift** is an adapter defect. It is fixable in `commanded_to_chunk`, it makes T-39 re-runnable
  against a corrected label space, and every number in `docs/benchmark.md` gets re-read rather than
  re-bounded.
- **Content** is not fixable by re-anchoring. It means the commanded stream carries signal the
  executed trajectory does not, and the label space needs a redesign — which becomes the live
  question ahead of any new model, any new backbone, and any more data.

Nothing else currently on this project's board distinguishes them, and nothing else is this cheap.

## 2. The sweep

`oracle_action` builds each chunk as (`scripts/eval_t39_baseline.py:543`):

```
chunk[t_ns] = commanded_to_chunk(action[i : i+16], state[i])      i = raw row at t_ns
```

with `targets[t] = q(action[i+t]) − q_from[t]`, `q_from[0] = q(state[i])`, `q_from[t>0] = q(action[i+t−1])`.

**Variant A — command-only shift.** `action[i+k : i+k+16]`, anchor stays `state[i]`.

> Tests: *the command that produced executed step `i` is `action[i+k]`.* This is controller
> latency, or an off-by-one in which end of a control interval the corpus timestamps. The anchor
> does **not** move, because the label being predicted is the displacement out of the state the
> robot was in at `t_ns`; moving the anchor as well would score a different chunk of the episode
> against our chunk, which is a different experiment wearing this one's name.

**Variant B — co-shifted (control).** `action[i+k : i+k+16]`, anchor `state[i+k]`.

> Tests: *our converted chunk at `t_ns` corresponds to raw row `i+k`.* A time-base offset in our
> own conversion rather than in the robot. This is expected to be **flat and bad** away from
> `k = 0`, because `raw_anchor_indices` already refuses an inexact timestamp match and so the two
> clocks are proven to agree. It is registered anyway: a control that is predicted to say nothing
> is how you find out that the thing you thought was proven is not.

**Grid.** `k ∈ {−4, −3, −2, −1, 0, +1, +2, +3, +4}`, both variants — 18 evaluations. At `dt =
33.33 ms` this brackets **±133 ms**, wider than any plausible position-control lag on a 30 Hz
corpus. The grid is fixed here and is not extended after seeing results; if the optimum lands on an
edge, that is reported as *unbounded on that side* and a widened grid is a new pre-registration.

**Every offset scores the identical chunk set.** A shifted window falls off the end of an episode
at a different chunk, so the naive sweep compares different sample sets and calls the difference a
delay. The sweep therefore restricts every cell to the chunks valid for **all** `k` in the grid —
dropping `max(k)` chunks at the start and `max(k)` at the end of each episode. **Consequence,
recorded in advance: the sweep's own `k = 0` cell will not equal `−359.41 pp` exactly**, because it
is scored on fewer chunks. The sweep's internal baseline is its own `k = 0` cell; the archived
−359.41 is the plumbing check of §6 and nothing else.

## 3. Gates — `PR10_RULE_V1`, fixed here and in the script

Same metrics, same scorer, same thresholds as T-39. Nothing is recalibrated for this experiment.

```
L1(k) := skill_vs_repeat_pct(k)     > 0
L2(k) := ci_skill_vs_repeat_pct(k)  > 0
MATERIAL_FLOOR_PP = 10.0            (borrowed from I8_RULE_V3, as T39_RULE_V1 did)
k_best := argmax_k skill_vs_repeat_pct(k)   over variant A
gain   := skill_vs_repeat_pct(k_best) − skill_vs_repeat_pct(0)
```

Evaluated in this order, first match wins:

| | condition | verdict |
|---|---|---|
| 1 | `k = 0` plumbing check (§6) fails | **VOID (plumbing)** — the sweep is not measuring T-39's arm |
| 2 | `L1(k_best)` **and** `L2(k_best)`, `k_best ≠ 0` | **D — delay** |
| 3 | `L1(k_best)`, not `L2(k_best)`, `k_best ≠ 0` | **P — partial** |
| 4 | not `L1(k_best)`, `gain ≥ MATERIAL_FLOOR_PP` | **P — partial** |
| 5 | not `L1(k_best)`, `gain < MATERIAL_FLOOR_PP` | **J — content** |
| 6 | `k_best = 0` | **J — content** |

## 4. What each verdict licenses — decided before the numbers exist

**D (delay).** Licenses a defect report against `commanded_to_chunk`'s anchoring **and** a corrected
re-run of T-39's G0b at `k_best`. It does **not** by itself lift T-39's VOID: the verdict was
recorded against a rule, and re-running an arm produces a new result document, not an edit to an old
one. It does **not** license training, generation, or any statement about GR00T — PR-07 §6 still
holds until a corrected G0b actually clears L1.

**P (partial).** Licenses the same defect report, and additionally establishes that a shift is *not
the whole story* — re-anchoring alone will not produce a label space that clears the bar. The
project's next question becomes what the residual content is, not which model to try.

**J (content).** Licenses a defect report against the **label space**, not the anchoring. Explicitly
retires "the adapter is mis-anchored" as an available explanation for the fourteen negatives, which
is worth more than it sounds: it is currently the cheapest available excuse and it would otherwise
be re-proposed indefinitely. It licenses **no** model work of any kind.

**No verdict here licenses a training run.** The gate in `CLAUDE.md` and `subprojects/README.md` is
the project owner's to release and this document does not touch it.

## 5. Secondary readings — recorded, never gates

- `smoothness_ratio(k_best)`. If the best-aligned command is *still* > 2, then even optimal
  alignment leaves the command jerkier than the executed trajectory, which is a content difference
  sitting on top of whatever shift was found. Read as corroboration; it moves no verdict.
- `horizon_ratio(k)` across the grid. A shift should move error out of the first step and flatten
  this toward 1.0 near `k_best`.
- Variant B's shape. Predicted flat; a peak away from `k = 0` would mean the timestamp match in
  `raw_anchor_indices` is agreeing for a reason other than the one its docstring claims.
- The gripper channel is degenerate on every cell (`gripper_accuracy` withheld, PR-07-RESULT §"What
  this cannot answer"). No cell of this grid can see a grasp and none may be read as if it could.

## 6. The plumbing check, and its honest status

`k = 0`, variant A, full chunk set, run locally on CPU before this document was written:

```
E1 action mse 4.19791e-05   vs zero −157.11%   vs repeat −359.41%   ci −102.54%   1040 chunks, 8.5 s
```

Bit-identical to the archived cluster result on all four figures. **This is a re-run of a known
number, not a finding**, and it is disclosed here rather than presented as a result: it establishes
only that the local dataset, the local raw parquets and the frozen scorer reproduce T-39's arm, so
that a difference elsewhere in the grid is attributable to `k`.

One discrepancy is recorded rather than smoothed: the local `runs/t39-baseline-seed0/run_metadata.json`
carries `dataset_snapshot_ref sha256:6b8fe849…` and `git_commit 91cddf4…`, while the cluster run
carried `sha256:598f193f…` and `git_commit unknown`. The witness is therefore the **local** one; the
split it proves (362 train / 40 holdout, disjoint, hash matched) is the same split, and the scored
numbers are identical, which is what the check is for.

## 7. Cost

18 CPU evaluations × ~9 s ≈ **3 minutes**, on this workstation. **Zero GPU-hours, zero billing, no
cluster contact, no download, nothing submitted.** There is no budget ceiling to register because
there is no budget. If this document is wrong about something, the cost of finding out is three
minutes, which is the argument for running it before anything expensive.

## 8. What must exist before it runs

1. `--action-offset` on `scripts/eval_t39_baseline.py`, **default 0 and behaviour-preserving**, so
   that the archived T-39 command line produces the archived T-39 numbers unchanged.
2. `--chunk-margin`, the common-support restriction of §2, likewise default 0.
3. A mutation test: with the offset threaded but *ignored*, the sweep must fail. An offset knob that
   is silently a no-op produces a flat grid and a confident **J**, which is the failure mode this
   experiment is most likely to have and least likely to notice. That is T-37's transposed-`xmat`
   lesson and PR-07 §4's mis-anchoring tests, applied to the fix rather than to the original.
4. `scripts/sweep_t39_anchor.py` implementing `PR10_RULE_V1` §3 as code, printing the verdict.

## 9. What this cannot answer

- **Anything about GR00T.** The policy arm did not run in T-39 and does not run here. PR-07 §6's
  prohibition is unaffected by every outcome above.
- **Whether the corpus is usable.** A found delay says the label space is *repairable*, not that a
  repaired label space clears the bar. That is a re-run of G0b, and it is not this document.
- **Which side is wrong.** `k_best ≠ 0` is consistent with the corpus's controller lagging, with our
  conversion dropping a step, and with the corpus timestamping the far end of a control interval.
  This sweep measures the offset; it does not attribute it.
- **Anything about the other twelve `G1_Dex3_*` corpora** (T-043). One corpus, one conversion.
- **Anything about grasping.** See §5.
