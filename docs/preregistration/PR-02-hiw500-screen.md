# PR-02 — is HIW-500 a dataset on which vision can be shown to matter?

**Pre-registered 2026-08-02, before any predictive number was computed.** Structure of the data was
inspected first (columns, dtypes, task list, episode lengths); no predictor was fitted and no
accuracy, MSE or R² was computed on HIW-500 before this file was committed.

## Why this screen exists

Every negative result in this project (T-15, T-18, T-24, T-26, T-16, T-29, T-30, PR-01) was measured
on the *same* 402 success-only episodes of one task. PR-01 established why that is a problem: on
`datasets/gr00t-apple-full`, **66.0 % of the achievable range of our headline metric is reachable
with no vision at all**, and a rule with zero fitted parameters beats the 82.5M-parameter fine-tune.
We cannot distinguish "the approach does not work" from "this dataset cannot show it working".

`BitRobot/HIW-500-LeRobot` (CC-BY-4.0, `unitree_g1`, 23 743 episodes, 500+ h, 12 real homes,
11 tasks, 30 fps) is the candidate replacement. The card claims exactly the property ours lacks:
*"layouts, object states, lighting, clutter, and operator styles vary from episode to episode."*

**This screen does not ask whether the dataset is good. It asks whether a blind baseline fails on
it.** That is the only property that makes a dataset able to answer AC-07, and it is measurable from
the parquet alone — **no video is decoded and none is downloaded** (`data/` is 13.2 GB of the repo's
2.15 TB; the screen uses three files, ~241 MB).

## What is measured

Slice: `data/chunk-000/file-{000,001,002}.parquet`, the first ~340 episodes — chosen to be
comparable in size to our 402, and fixed here before scoring. Chunks are built exactly as
`convert_lerobot_g1.py` builds ours: **16 steps, non-overlapping, targets = per-step joint deltas in
rad**, from `observation.state`'s 14 arm joints (indices 15–28), at the same 30 fps. State for the
predictors is `q`, backward-difference `dq`, and the grasp channel — the same 32-dim-equivalent
construction PR-01 used.

Predictor suite is PR-01's, unchanged, so the two datasets are compared on one axis only:

| predictor | fitted params | GR00T value (archived) |
|---|---:|---:|
| zero-delta (hold still) | 0 | 1.632760e-05 |
| const-velocity `dq·dt_s` | **0** | 9.137664e-06 |
| ridge, all state, λ=1e-2 | 7 920 | 6.330899e-06 |
| blind nonlinear ceiling (RFF) | 2 048 | 5.431371e-06 |

Hyperparameters for the ceiling are selected on an **inner, episode-disjoint split of the training
episodes** and never on the scored split — as in `scripts/bench_ridge_baseline.py`. All CIs are
episode-level bootstrap, 5 000 resamples.

### The primary quantities

- **M1 — momentum share.** `(zero_delta − const_velocity) / (zero_delta − ceiling)`. The fraction of
  everything a blind model could achieve that a **zero-parameter** rule already achieves.
  **GR00T: 0.660.**
- **M2 — blind-unreachable energy.** `1 − R²(ceiling)`, i.e. the share of target energy the best
  no-vision predictor leaves on the table. **GR00T: 0.333.**
- **M3 — grasp channel liveness.** Debounced transitions per episode on the trigger/squeeze channel
  of `observation.state.wbc`. GR00T source: 2.015/ep; our *converted* corpus: 0.000/ep (T-31).
- **M4 — grasp-timing predictability.** Cross-validated R² of the grasp instant from the episode's
  starting proprioceptive state. **GR00T: 0.077** — already near-unpredictable, so this is a
  guard against regression, not a target.

### The locomotion control, fixed in advance

HIW episodes contain locomotion; ours do not. A parked arm during "move to bed" is trivially
predictable and would **deflate M2 and inflate M1** for reasons that have nothing to do with vision.
`language_events` carries timestamped subtask labels, so:

**Primary M1/M2 are computed on manipulation subtasks only** — subtask strings not matching
`^move to `. The all-frames numbers are reported alongside as secondary. If the two disagree by more
than 0.10 on either quantity, the disagreement is the headline and no verdict is issued from the
pooled number.

## Decision rule — fixed before scoring

| verdict | condition | consequence |
|---|---|---|
| **PASS** | M1 ≤ 0.45 **and** M2 ≥ 0.45 **and** M3 > 0.5 | HIW-500 can discriminate. It becomes the testbed for AC-07; a converter and a video-decoding slice are justified. The "we must buy a G1 and record for months" conclusion weakens. |
| **FAIL** | M1 ≥ 0.60 **and** M2 ≤ 0.40 | Same trap, new dataset. Blind wins here too — which would be a **finding about the approach**, not about the data, and a much stronger negative than anything we hold today. Do not convert. |
| **MIXED** | anything else | No global claim. Report which of the two moved and by how much; the next step comes from the pattern, as in PR-01 VERDICT C. |

**Direction of the thumb.** M1 ≤ 0.45 and M2 ≥ 0.45 are set *against* the outcome I expect to be
convenient: momentum dominance at 30 Hz is physics (PR-01-FOLLOWUP §1), and HIW is also 30 Hz, so
a large M1 here is the null and PASS has to be earned.

### The confound that could fake a PASS, and its control

**M2 cannot distinguish signal from noise.** Teleoperated whole-body demonstrations in cluttered
homes are jerkier than a lab pick-and-place, and jerk is unpredictable from *anything* — including
video. A high M2 driven by sensor noise would look identical to a high M2 driven by real visual
dependence, and would send us to convert 2 TB for nothing.

Control, mandatory before any PASS is issued: the **1-step-ahead noise floor** — fit the same blind
ceiling to predict step 0 only, where visual anticipation is worth least and momentum most. If M2's
excess over GR00T is present at step 0 too, it is **jerk, not task**, and PASS is downgraded to
MIXED with that stated. A PASS requires the excess to be concentrated in the *late* steps of the
chunk, which is where PR-01 showed real task content lives (49.9 % of target energy in steps 8–15).

## What a PASS would and would not license

It would license converting a slice of HIW-500 and re-running the ladder on it. It would **not**
license any claim that WAM works, that video helps, or that the T-16 verdict was wrong: a dataset
where blind fails is a *precondition* for testing AC-07, not evidence about it. Nor does it retire
any archived number — our GR00T runs stay exactly as scored, on the dataset they were scored on.

Three things this screen structurally cannot see, all of which need video or hardware: whether the
frames are usable by Wan, whether the mobile base breaks our fixed-base MVP assumption (it very
likely does — MuJoCo's scene has a welded base), and whether the action space can be mapped to
`JOINT_DELTA` without losing the task. The 23-dim native space is base velocity + torso pose + EE
Cartesian poses + trigger/squeeze, and only the arm joints are directly comparable to ours.
