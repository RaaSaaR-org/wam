# WAM-Bench — the offline ladder (T-27)

We have 402 real G1 episodes, no robot, and no sim for this task. Everything measurable today
therefore has to come out of held-out demonstrations. WAM-Bench is that layer: a fixed ladder of
rungs scored from an archived `predictions.jsonl`, no GPU and no hardware involved.

```
.venv/bin/python scripts/run_bench.py runs/d1-full-gen-seed0
.venv/bin/python scripts/run_bench.py runs/a runs/b --compare
```

Writes `bench.json` + `bench.md` into each run directory. Because it consumes only stored
predictions, **every past run can be re-scored when the metric set changes** — a new rung does not
cost a retrain.

---

## Where it sits in the E-ladder

`E0 unit → E1 offline replay → E2 sim → E3 real robot → E4 generalization`

E3 and E4 need hardware we do not have. E2 needs a sim of this task, which we also do not have.
So the whole real-data question lives inside E1, and E1's headline was a single number: action
MSE on the holdout. WAM-Bench subdivides that one rung into five, all computable on the same data.

---

## Why raw MSE was not enough

Two published results drive the design.

**Task-critical restriction beats whole-trajectory averaging.** Restricting the same action error
to task-critical intervals lifts the rank correlation with real rollout success from Spearman
≈ −0.61 to ≈ −0.87 ([CI-MSE, arXiv:2606.29898](https://arxiv.org/html/2606.29898v1)). Most
timesteps in a demonstration are quiet, so an average over all of them largely measures how well
a policy holds still.

**Fidelity is not controllability.** A world model can produce plausible video while ignoring the
commanded action ([MiraBench, arXiv:2605.29360](https://arxiv.org/pdf/2605.29360)). The two must
be scored on separate axes, or "looks right" gets mistaken for "responds right".

And the field itself is nowhere near saturation, which sets expectations: on
[RoboDojo](https://robodojo-benchmark.com/) — 42 sim + 18 real manipulation tasks — the best of 30
evaluated policies reaches **8.8% average success against 76.0% for human experts**; on the real
leaderboard π0.5 gets 12.8%. A benchmark whose top score is ~12% of human is measuring the right
thing. One where everyone passes is not.

---

## The ladder

Five rungs, each a yes/no gate plus continuous points (20 each, 100 total). The reported **level**
is the highest *contiguous* rung passed — clearing L3 while failing L1 does not earn L3, because
the rungs are premises for each other. All thresholds are module constants in
`src/wam/evaluation/benchmark.py`, fixed before any run is scored.

| rung | question | metric | gate |
|---|---|---|---|
| **L0** beats-doing-nothing | Better than holding still? | `skill_vs_zero_pct` | > 0% |
| **L1** beats-inertia | Better than repeating the last action? | `skill_vs_repeat_pct` | > 0% |
| **L2** acts-when-it-counts | Still better where the task actually happens? | `ci_skill_vs_repeat_pct` | > 0% |
| **L3** holds-the-horizon | Does the chunk hold together to its last step? | `horizon_ratio` | ≤ 4.0 |
| **L4** moves-like-a-demo | Moves like a demonstration — neither jerkier nor blander? | `smoothness_ratio` | 0.5 ≤ r ≤ 2.0 (spec 0.2.0) · ≤ 2.0 (spec 0.1.0) |

**The two trivial baselines.** *Zero-delta* predicts no motion. *Repeat-last-action* replays the
action executed one control period before the chunk starts — strictly causal, and derived from the
chunk cadence rather than by taking the previous chunk's last step, which would read the future
whenever chunks overlap (the regime FR-05's receding horizon actually runs in).

**Critical intervals.** CI-MSE annotates them with a VLM. We have one task and no annotations, so
the proxy is the demonstrator's own commanded motion energy: the top 20% of chunks by RMS action
magnitude are the moments the task is being performed. Documented as a proxy, not as the paper's
method.

**Diagnostics** (not gated): `timing_gain_pct` — error removed by allowing a ±1-step shift; a large
gain means the shape is right and the phase is wrong, which is a latency bug rather than a capacity
one. Plus `gripper_dynamic_range`, `gripper_majority_pct`, `gripper_transitions_per_episode`, and
any `warnings`. `gripper_accuracy` is emitted only when the channel is admissible — see
[What the bench refuses to report](#what-the-bench-refuses-to-report).

---

## Bench spec versions

A gate threshold is fixed before a run is scored. When one has to change anyway, the change is
**versioned** rather than applied backwards: every archived `predictions.jsonl` keeps the headline
it was recorded with, and the new rule is declared for future runs. Both are always reported.

| spec | declared | L3 | L4 | scored artifact |
|---|---|---|---|---|
| **0.1.0** | 2026-07-29 | `horizon_ratio` ≤ 4.0 | `smoothness_ratio` ≤ 2.0 | `bench.json` / `bench.md` |
| **0.2.0** | 2026-08-01 | `horizon_ratio` ≤ 4.0 | `0.5 ≤ smoothness_ratio ≤ 2.0` | `bench-0.2.0.json` / `.md` |

```
.venv/bin/python scripts/run_bench.py runs/t16-lora-seed0/eval-latest   # both, side by side
.venv/bin/python scripts/run_bench.py runs/... --spec 0.1.0             # one only
```

`bench.json` means spec 0.1.0 **forever**. Re-pointing it would retroactively restate three
recorded results under a rule nobody applied to them; 0.2.0 becomes the headline by being printed
next to 0.1.0, not by overwriting the file 0.1.0 already owns.

**Why L4 changed.** The one-sided gate scored T-16's `smoothness_ratio` 0.29 a full 20/20 — but
0.29 means the prediction is 3.4× *smoother* than a demonstration, which is a defect, not a
virtue. A jerk ratio is multiplicative, so 0.2.0 scores `|log r|`: 2× jerkier and 2× smoother are
the same size of deviation and cost the same. **No new number is introduced.** The floor is
derived, `MIN_SMOOTHNESS_RATIO = 1 / MAX_SMOOTHNESS_RATIO`, and never written as a literal, so it
cannot be tuned against the value that motivated it. The anchor stays `r == 1.0` — the value spec
0.1.0 already passed to `_points` — so 0.2.0 completes 0.1.0's intent rather than replacing it.

### The adoption rule, and the one that was withdrawn

**Withdrawn (2026-08-01), it was vacuous.** The rule first written down here was *"spec 0.2.0
ships only if re-scoring all three archived runs moves no run's level"*, and the table below was
presented as the test it passed. It was not a test. `level` is the highest **contiguous** rung
passed, and all three archived runs fail L1 — `skill_vs_repeat_pct` −20.9 % / −129.0 % / −32.4 %,
tabulated in this same file before the rule was authored. L4 is therefore unreachable for all
three under any L4 rule whatsoever, so the rule could not have failed. Recording it as a passed
pre-registration inside a change that is *about* pre-registration discipline is the failure mode
this repo keeps a docs section for; it is retracted rather than quietly deleted.
(`tests/test_benchmark.py::test_a_run_that_fails_l1_keeps_its_level_whatever_l4_says` pins the
mechanism so the same rule cannot be re-registered.)

**The rule that replaces it**, and one regression test that is *not* a rule. Rule 1 is falsifiable;
item 2 below is an identity, and is listed here — labelled — only because the withdrawn rule was
withdrawn for being one, and silently keeping a second would repeat the mistake:

1. **No run's score may increase.** An L4 change moves the score directly, and nothing about
   adding a floor forces the move downward. Two of the three archived runs score **0/20** on L4
   under the one-sided gate (`t18` at r = 5.10, `d1-full-gen` at r = 2.35), so any 0.2.0 that
   re-anchored the points function while adding the floor — a linear penalty on `|r − 1|`, or a
   wider ceiling to make room for the new side — would have raised them. The shipped rule
   satisfies the constraint for **every** `smoothness_ratio` and not only for the three archived
   ones: `test_no_ratio_scores_more_under_spec_0_2_0_than_under_0_1_0` sweeps a 6 000-point linear
   grid plus a 2 000-point log grid over `[1e-6, 1e3]` and the three archived ratios. (It checked
   11 hand-picked ratios while the prose said "every"; that gap is closed.)
2. **The two specs' L4 verdicts differ exactly on `r < MIN_SMOOTHNESS_RATIO`** — *not a rule; a
   regression test.* The intended change is "too bland is also a failure", and this states it. But
   0.1.0's gate is `r ≤ 2.0` and 0.2.0's is `0.5 ≤ r ≤ 2.0` with the same ceiling, so the
   difference set is `{r < 0.5}` by construction: no measurement could have made this false, and a
   condition that cannot fail is not evidence. Its value is against *implementation* drift — if a
   later edit moves the ceiling or reshapes the band, the identity breaks and
   `test_the_two_specs_disagree_exactly_below_the_derived_floor` fails. Keep it as a test; do not
   cite it as support for the change.

**What neither rule tests: the floor's VALUE.** Rule 1 cannot see it — `_smoothness_rung`'s 0.2.0
branch scores `-|ln r|` against `-ln(max_smoothness_ratio)` and never reads
`min_smoothness_ratio` — and item 2 holds for any floor by construction. Checked rather than
assumed: both hold for floors of 0.05, 0.2, 0.5, 0.8, 0.95, 0.999, 1.0, 1.9 and 1.999, and a
synthetic run at `r = 0.64` scores an identical 87.12 under a 0.5 floor and a 0.9 floor — only
`passed`/`level` move. So `MIN_SMOOTHNESS_RATIO = 1/MAX_SMOOTHNESS_RATIO` rests on the symmetry
argument alone, not on either rule. That is a defensible place to put it (the reciprocal is the
one value that needs no tuning, which is exactly what makes it un-tunable against the run that
motivated it) — but it is an argument, and it should not be read as something the tests
constrain.

**What re-scoring the three archived runs actually shows** — a description of the change, not a
test it passed:

| run | `smoothness_ratio` | score 0.1.0 | score 0.2.0 | level 0.1.0 | level 0.2.0 | L1 |
|---|---|---|---|---|---|---|
| `d1-full-gen-seed0` | 2.3469 | 28.6 | 28.6 | L0 | L0 | fails (−20.9%) |
| `t18-real-ablation-seed0` | 5.0980 | 19.9 | 19.9 | below L0 | below L0 | fails (−129.0%) |
| `t16-lora-seed0` | 0.2932 | **48.4** | **28.4** | L0 | L0 | fails (−32.4%) |

Only T-16's score changes, by exactly the 20 points L4 should never have awarded it. The level
column is constant for the reason given above and carries no information about the change.

---

## What the bench refuses to report

`gripper_accuracy` is **withheld** — emitted as `null`, rendered as `n/a — withheld (…)` — whenever
the demonstrated channel's peak-to-peak range is below `GRIPPER_MIN_DYNAMIC_RANGE` (0.25). The
majority-class baseline goes in its place, because that is what the "accuracy" was measuring.

A warning was not enough. T-27 already emitted one, and the number kept travelling: the ablation
harness (`compare_runs`) scored `gripper_accuracy` with no admissibility check at all, and that
path put `runs/t18-real-ablation-seed0`'s **0.8534** into TASKS.md T-18 as part of the AC-07
verdict — a number that is *exactly* the holdout's majority-class rate of 85.34%, i.e. zero
information. The ablation now omits the metric and says so in the rendered table.

Withholding is deliberately **not** spec-versioned. The gripper earns no rung points, so
suppressing it cannot move a score or a level; re-scoring all three archived runs under spec 0.1.0
after the change reproduces every scored field, every rung's points, `level` and `score`
**bit-identically**. That re-score is the test, not the claim.

The standing rule: **no offline result on a dataset that fails `scripts/audit_gripper.py` may be
described as evidence about grasping — only about reaching.**

```
.venv/bin/python scripts/audit_gripper.py datasets/gr00t-apple-full      # exit 0 = PASS, 1 = FAIL
.venv/bin/python scripts/audit_gripper.py --lerobot data/raw/gr00t_apple
```

The audit is a dataset-level gate, all clauses pre-registered in `src/wam/evaluation/gripper.py`.
It reads parquet only and **never decodes video**, which is what keeps it cheap enough to run over
402 episodes (3.8 s measured) before anything is trained.

**Audit spec 0.2.0 (2026-08-01)** — the clause set, and why it grew:

| clause | constant | refuses |
|---|---|---|
| peak-to-peak ≥ 0.25 | `GRIPPER_MIN_DYNAMIC_RANGE` | a channel with no open/close event to score |
| ≥ 1.0 debounced transitions/episode | `GRIPPER_MIN_TRANSITIONS_PER_EPISODE` | a constant channel (margin 0.10 around the 0.5 binarization threshold) |
| ≥ 50 % of episodes with **≥ 2** debounced transitions | `GRIPPER_MIN_EPISODES_WITH_GRASP_CYCLE` | a channel that never closes *and* reopens |
| ≥ 80 % of episodes with ≥ 1 | `GRIPPER_MIN_EPISODES_WITH_TRANSITION` | a reaching dataset with occasional grasps |

The third clause is new and it closes a hole in 0.1.0 that mattered: **the mean clause admitted a
channel containing no grasp anywhere.** 50 episodes of a pure monotone ramp 0 → 1 — never closing
and reopening in any episode — score peak-to-peak 1.00, exactly 1.00 debounced transitions per
episode and 1.00 episodes-with-a-transition, i.e. three clauses cleared and zero failures. The
relaxation from "2 transitions, because the task closes on the object and opens to release it" to
"a MEAN of 1.0, to tolerate partial episodes" swallowed the event being tested: `(1, 1, 1, …)`
averages to the same 1.0 as `(2, 0, 2, 0, …)`. The new clause is not a new tolerance — it is the
mean clause's own derivation ("up to half the episodes may be partial") stated per-episode, where
a mean cannot launder it, and half the episodes carrying a full grasp still passes.

**Saturation is reported, not gated.** Every mapping in the repo ends in `clip(·, 0, 1)`, and
clipping moves *all* the clauses in the passing direction — a clipped channel has more range, more
crossings and more debounced transitions than the same channel unclipped. The audit now names it:
whenever more samples sit exactly on a rail than a dataset-level min-max fit over the audited set
could put there (one sample per rail per episode, `2 · num_episodes / num_steps` — derived from
the mapping, not from a measurement), a `NOTE (not gated)` lands in the report's `reasons` on PASS
and FAIL alike, and the reported `scored_channel` now prefers a clean channel over a clipped one
so a PASS is never explained by inflated numbers. It is deliberately **not** a clause: a two-state
gripper *command* channel is
saturated at both rails by construction, and `data/raw/gr00t_apple` contains exactly that —
`action.left_hand.max_joint[0]` is on a rail for 97.6 % of its samples and is the cleanest grasp
signal in the snapshot (2.04 debounced transitions/episode, a complete cycle in 99.8 % of
episodes). A clause tight enough to catch that would refuse it. The exact gate lives
where the mapping is applied and the unclipped values still exist: `convert_lerobot_g1.py` refuses
a pinned `--gripper-affine` that clips a single sample.

---

## First results (2026-07-29)

Both existing real-data runs, identical 40-episode holdout:

| metric | `d1-full-gen-seed0` (action-only) | `t18-real-ablation-seed0` (world-action) |
|---|---|---|
| **level** | **L0** beats-doing-nothing | **none — below L0** |
| **score** | **28.6 / 100** | **19.9 / 100** |
| mse | 1.10439e-05 | 2.09285e-05 |
| ci_mse | 2.30187e-05 | 5.26316e-05 |
| skill_vs_zero_pct | +32.4% | −28.2% |
| skill_vs_repeat_pct | **−20.9%** | **−129.0%** |
| ci_skill_vs_repeat_pct | −7.0% | −144.6% |
| horizon_ratio | 1.66 | 1.02 |
| smoothness_ratio | 2.35 | 5.10 |
| gripper_accuracy | ~~0.87~~ withheld † | ~~0.85~~ withheld † |

† Both were withdrawn on 2026-08-01: the demonstrated channel's majority class on this holdout is
**85.34 %**, and `t18`'s 0.8534 is that rate to four decimals. See
[What the bench refuses to report](#what-the-bench-refuses-to-report).

Baselines on this holdout, for reference: zero-delta 1.63276e-05, repeat-last-action 9.13766e-06.

Three things the previous dashboard could not say:

1. **The action-only baseline loses to repeat-last-action.** T-14 recorded "E1 mse 1.10e-5 vs
   zero-delta 1.63e-5 (−32%)" as the first generalization result. True — but the repeat baseline
   scores 9.14e-6, **17% better than the trained model**. The −32% is the demonstration's own
   inertia, not learned behaviour. This does not retract T-14's number; it adds the reference that
   makes it readable.
2. **The world-action candidate is worse than predicting no motion at all** (−28.2% skill).
   "Hurts" was already the T-18 verdict at −89.5% relative to the baseline; against an absolute
   floor it is starker, and `smoothness_ratio` 5.10 says the multi-task tax shows up as jerk.
3. **`gripper_accuracy` is not a grasp proxy on this dataset.** The demonstrated gripper signal
   has peak-to-peak range 0.120 and sits on the 0.5 binarization threshold — it never opens or
   closes. The reported 0.87/0.85 is thresholding noise. The bench now emits this as a warning
   instead of letting the number stand.

   > **Correction, 2026-08-01 (T-31).** The measurement above is right; the sentence it was
   > written up as — *"the demonstrated gripper never opens or closes, this dataset cannot support
   > a grasping claim"* — is wrong, and wrong in the cheapest possible direction. It states a fact
   > about **our converter**, not about the dataset. `scripts/audit_gripper.py --lerobot
   > data/raw/gr00t_apple` **PASSES** the source snapshot: the left Dex3 hand sweeps a clean
   > close-then-open (per-joint p2p up to 0.83 rad; `action[29:36]` sweeps the full −1.0 … 0.7).
   > The scored channel `state.left_hand.max_joint[4]` carries **2.01 debounced transitions per
   > episode**, with a complete close-and-release in **98.8 %** of the 402 episodes. The right
   > hand is frozen (per-joint p2p ≤ 0.0007).
   >
   > `scripts/convert_lerobot_g1.py` then destroyed it in two documented halvings: the synergy
   > mapping `clip((mean(hand_7dof)+1)/2, 0, 1)` assumes a [−1, 1] joint range the hand never uses
   > and halves the sweep, and `relabel_chunks` averaged that against the **dead** right hand and
   > halved it again — landing on 0.137 global / 0.080 per-episode p2p, centred on the 0.5
   > threshold. The audit of `datasets/gr00t-apple-full` FAILS all four clauses: 0.00 debounced
   > transitions in 0 of 402 episodes, majority class 88.2 %.
   >
   > This is a mapping bug on data we already own, not a reason to go shopping for a dataset.
   > `--gripper-mapping active-hand` fits one dataset-level affine over the hand that actually
   > moves and takes `gripper_target` from that hand alone. Over **all 402 episodes** (fit:
   > offset −0.438654, span 0.466748) it produces per-episode p2p **0.6885**, **2.01 debounced
   > transitions per episode**, a complete close-and-release in **99.0 %** of episodes (exactly
   > two in 97.3 %, none in 4 episodes), majority class 77.7 % — and the re-audit **PASSES all
   > four clauses**, with no saturation notice. (Dataset-level, never per-episode: a per-episode
   > min-max would make the same physical aperture mean a different number in every episode.) The
   > affine is recorded in the manifest's `normalization` provenance slot, next to
   > `mapping.gripper_affine_source` and `mapping.gripper_clip`, so the mapping travels with the
   > data.
   >
   > These numbers are measured through the converter's own mapping and chunking on the parquet
   > columns, so they need no video decode; a real conversion additionally truncates each episode
   > to its decoded frame count.
   >
   > **Correction to the correction, same day.** This paragraph first validated the fix with
   > 30-episode numbers (per-episode p2p 0.76, "exactly 2 debounced transitions in every single
   > episode"). Those are real but they do **not** survive the full set, and the stronger of the
   > two sentences is the one that breaks: at 402 episodes four episodes carry no complete grasp,
   > so "every single episode" becomes 99.0 %. The cause is that **dataset-level is not
   > dataset-independent** — the affine is fitted over the episodes of one invocation, and 30 /
   > 120 / 402 episodes give offset −0.39980/−0.41043/−0.43865 and span 0.41004/0.43853/0.46675.
   > A raw synergy of −0.40 is 0.000 under the 30-episode fit and 0.083 under the 402-episode one.
   > Two conversions are comparable only if they share an affine, which is what
   > `--gripper-affine OFFSET SPAN` is for; a pinned affine that would clip any sample of the new
   > set is refused, because the 30-episode fit clips **1.34 %** of the 402-episode set and
   > clipped samples are indistinguishable from measurements once written.
   >
   > The fixed conversion goes to a **new** root (`datasets/gr00t-apple-grip/`).
   > `datasets/gr00t-apple-full/` is immutable — `runs/t16-lora-seed0`'s `dataset_snapshot_ref`
   > hashes its manifest bytes and `scripts/eval_t16.py` refuses to score on a mismatch — so
   > `--gripper-mapping legacy` stays the converter default and reproduces it exactly.
   >
   > And the first grasping number does **not** come from re-scoring `runs/t16-lora-seed0` against
   > the fixed channel: that checkpoint was trained against the degenerate one, so such a score
   > measures a model on an objective it never had. It is a lower bound at best. The first
   > admissible grasping number needs a retrain.

Point 1 is the load-bearing one: **the bar T-16 has to clear is `skill_vs_repeat_pct > 0`**, not
"beats the action-only baseline". Beating a model that itself loses to a one-line heuristic is not
evidence that video helps.

## The T-16 result (2026-07-30) — the bar was not cleared

`runs/t16-lora-seed0`, 20 000 steps of Wan2.2-TI2V-5B LoRA on the 362 training episodes, scored on
the same 40-episode holdout by `scripts/eval_t16.py` (split proven against the trainer's
`dataset_snapshot_ref`, not asserted):

| metric | action-only | world-action (tiny) | **T-16 LoRA (Wan 5B)** |
|---|---|---|---|
| **level** | L0 | below L0 | **L0** beats-doing-nothing |
| **score** | 28.6 / 100 | 19.9 / 100 | **48.4 / 100** |
| mse | 1.10439e-05 | 2.09285e-05 | 1.21027e-05 |
| ci_mse | 2.30187e-05 | 5.26316e-05 | 3.24412e-05 |
| skill_vs_zero_pct | +32.4% | −28.2% | +25.9% |
| **skill_vs_repeat_pct** | −20.9% | −129.0% | **−32.4%** |
| ci_skill_vs_repeat_pct | −7.0% | −144.6% | −50.7% |
| horizon_ratio | 1.66 | 1.02 | 1.30 |
| smoothness_ratio | 2.35 | 5.10 | 0.29 |
| gripper_accuracy | ~~0.87~~ withheld † | ~~0.85~~ withheld † | ~~0.89~~ withheld † |
| **score, bench spec 0.2.0** | 28.6 | 19.9 | **28.4** |

† Majority-class baseline on this holdout is 85.34 %; see
[What the bench refuses to report](#what-the-bench-refuses-to-report). No archived score rises
under spec 0.2.0 (the adoption rule); the levels are unchanged for a reason that is not evidence
about the change — see [Bench spec versions](#bench-spec-versions).

**Read the level, not the score.** 48.4 is the highest number any WAM run has produced and it is the
least informative column in the table: L1 and L2 both fail, so the 48.4 is L0's points plus the two
diagnostic rungs, which measure *shape* and not *skill*. On the one pre-registered bar — beat causal
repeat-last-action — the fine-tune is **worse than the action-only baseline** it was supposed to
improve on, on the full holdout and by more than double on the task-critical chunks.

`smoothness_ratio` **0.29** is the diagnosis and it is worth two points of care. Under spec 0.1.0
it scored 20/20, because that gate was "no jerkier than the demos" (≤ 2) — but 0.29 means the
prediction is 3.4× *smoother* than a real demonstration, which is not a demonstration-like
trajectory at all. Combined with a positive `skill_vs_zero` and a negative `skill_vs_repeat`, it is
**consistent with** a model that has learned the average pose trajectory of the task and not the
task: it moves in roughly the right direction, blandly, and a one-line heuristic that just keeps
doing what the arm was already doing beats it.

**"Consistent with", not "the signature of".** An earlier version of this paragraph claimed the
latter, and it was the only claim in this file that no measurement here supports. What these
numbers do rule out is a bounded-output artifact: max `|target|` is 0.0192 against a `tanh`, and
`limit_penalty` bites only outside ±0.95, so neither bound is active. Two alternatives survive and
each produces small, smooth chunks on its own — **the one-sided jerk regulariser**
(`configs/training/joint_wan_gr00t.yaml` sets `weights.smoothness = 0.01` and
`JointTrainer.compute_losses` applies `smoothness_loss` to `decoded_targets`, the *regression*
head's output, and to nothing else), and **plain L2 shrinkage** toward zero under a small-target
distribution. Separating any of the three needs an intervention — retrain at
`weights.smoothness = 0` — not a re-read of this table. `WAM-Bench` cannot make that call; see
`docs/improvements.md` I-3 and `cluster/discoverer/63_eval_t30_flow_head.sbatch`, which measures
what a readout swap *can* settle and states in its own pre-registration what it cannot.

Under **bench spec 0.2.0** the two-sided band scores it **0/20** and T-16's score is **28.4**, not
48.4. The level is unchanged — L0 — but that is not evidence of anything: L4 is only reachable
through L1 and L2, both of which fail, so no L4 rule could have moved it (see
[Bench spec versions](#bench-spec-versions) for the withdrawn adoption rule that mistook this for
a test). The ladder got the verdict right under both rules; what changes is that the headline
number no longer flatters a defect.

What this does and does not settle: it settles that on this dataset a pretrained video prior,
fine-tuned end-to-end with the action branch, adds nothing an inertia heuristic does not already
give — the last open route after T-15/T-24/T-26 ruled out frozen features. It settles nothing about
world-action modelling in general. 402 success-only episodes of one task cannot separate "the
approach does not work" from "there is not enough of the right data here to tell". The next
informative experiment is more and better data, not another backbone — and per the T-31 correction
above, the first instalment of "better data" is a re-conversion of the episodes we already have
with a gripper channel that survives the mapping, not a new download.

---

## What is deliberately not in here yet

- **Action-following / controllability** (MiraBench L2, [WorldArena](https://arxiv.org/pdf/2605.00080)):
  condition the video branch on true vs. perturbed actions and measure whether the prediction
  diverges. If it does not, the video is decorative. Needs model access, not just predictions —
  a separate harness. This is the most WAM-specific rung and the natural L5.
- **Optimism bias** (MiraBench L3): requires failure demonstrations. Our data is success-only, so
  it is not computable, not merely unimplemented.
- **Video fidelity** (PSNR/SSIM/LPIPS under action replay): needs stored predicted frames.
- **Closed-loop autoregressive divergence**: roll the policy on its own predictions and measure
  time-to-divergence. Needs the model in the loop.
- **Real success rate** — that is E3, and it needs the robot. Nothing offline substitutes for it;
  the ladder is a filter that stops bad candidates from reaching hardware, not a replacement.

---

## External landscape (July 2026)

| benchmark | what it scores | relevance here |
|---|---|---|
| [RoboDojo](https://arxiv.org/abs/2607.04434) | 42 sim + 18 real manipulation tasks; generalization, memory, precision, long-horizon, open-vocab | The laddered/leaderboard shape this borrows; sets the "best policy ≈ 8.8%" expectation |
| [CI-MSE](https://arxiv.org/html/2606.29898v1) | offline validation that correlates with rollout success | Directly implemented as L2 |
| [MiraBench](https://arxiv.org/pdf/2605.29360) | action-conditioned world-model reliability: physics, action-following, optimism bias | Source of the fidelity≠controllability split; future L5 |
| [WorldModelBench](https://papers.neurips.cc/paper_files/paper/2025/file/4ec03ed08a3fcb59e1c815b5598beff1-Paper-Datasets_and_Benchmarks_Track.pdf) | instruction following, commonsense, physics violations across 7 domains | Video-side scoring if/when we store predicted frames |
| [VLABench](https://github.com/OpenMOSS/VLABench) | large-scale VLA / embodied-agent leaderboard | Comparison point once a policy is worth submitting |
| [RoboTrustBench](https://arxiv.org/pdf/2606.01600) | trustworthiness of video world models for manipulation | Safety-adjacent, maps onto AC-03/AC-06 rather than here |
| ThorArena | humanoids under direct physical human contact | Hardware-gated; relevant at E3 |

---

## Sources

- [RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/abs/2607.04434)
- [Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies](https://arxiv.org/html/2606.29898v1)
- [MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models](https://arxiv.org/pdf/2605.29360)
- [WorldModelBench: Judging Video Generation Models As World Models](https://papers.neurips.cc/paper_files/paper/2025/file/4ec03ed08a3fcb59e1c815b5598beff1-Paper-Datasets_and_Benchmarks_Track.pdf)
- [VLABench](https://github.com/OpenMOSS/VLABench)
- [RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation](https://arxiv.org/pdf/2606.01600)
- [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/pdf/2605.00080)
