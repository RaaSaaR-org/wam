# PR-04 — what to record, how much, and the gate that decides whether to keep recording

**Pre-registered 2026-08-02, before any demonstration is recorded.** Thresholds are fixed here
and in `scripts/screen_corpus.py` as module constants. Per repo convention this document is
**annotated, never edited**; a threshold that has to change is versioned, not overwritten.

Five negatives (T-18, T-15/T-24/T-26, T-16, T-29, T-30) and one pre-registered STOP (PR-03) all
landed on the same conclusion: **the bottleneck is demonstrations.** That conclusion is also the
most expensive one in the project — a G1 EDU4, a teleop rig and months of recording. This
document exists so the money is spent against measured requirements rather than against "more
data", and so we find out whether the protocol works after **30 episodes instead of 400**.

## 1. What went wrong with the corpus we have, stated as requirements

Every line here is a measurement on `datasets/gr00t-apple-*`, not an impression.

| # | What is wrong now | Measured | Requirement for D1/D2 |
|---|---|---|---|
| R1 | A blind predictor gets most of the metric | M1 **0.660** (bar ≤ 0.45), M2 **0.333** (bar ≥ 0.45) | M1 ≤ 0.45 **and** M2 ≥ 0.45 |
| R2 | Reach target is mostly readable from proprioception | R²(grasp pose \| t=0 state) **+0.6136**, residual 0.4117 rad | R² ≤ 0.20 — randomize placement *after* the arm is home |
| R3 | Grasp timing is stereotyped | grasp at **0.545 ± 0.064** of the episode; a clock-only model scores **56.04 %** post-flip | clock-only post-flip ≤ 40 % |
| R4 | One hand is dead | right hand **0.0007 rad** across all 402 episodes and 171 625 samples; right arm parked (ratios 18–54) | both hands actuated, both arms moving within an episode |
| R5 | No failures at all | 0 of 402; optimism bias (MiraBench L3) is **not computable**, not merely unimplemented | ≥ 15 % failed or recovered episodes, labelled |
| R6 | Motion is over-smooth, so momentum wins | lag-1 autocorrelation **0.927** (HIW 0.827); **80.2 %** of chunk energy in the chunk mean; only **8.2 %** above 2.8 Hz | ≥ 15 % of energy above 2.8 Hz |
| R7 | One low-resolution view | one `ego` camera stored at **160×120** (source 480×640) | ≥ 2 views incl. one wrist, stored ≥ 224×224 |
| R8 | No IMU | `validity.imu=False` on all 402 episodes | record it |

**R1 is the one that matters most, and R2/R3/R6 are its causes.** M2 = 0.333 means two thirds of
what our headline metric can reward is reachable with no camera at all. On that corpus no
world-action result can separate "the video branch works" from "momentum works" — which is
precisely why five negatives could not be interpreted, and why PR-02 refused to call HIW-500 a
replacement without the same screen.

## 2. Volume — and why grasp events, not episodes, are the currency

PR-03 measured the power ladder on our own data: 150 holdout episodes → **2 682 post-flip steps
→ CI half-width 4.10 → MDE ~8.2 points**, from **2.01 grasp transitions per episode**. That is
**17.9 post-flip steps per episode**, and `h ∝ 1/√n`.

So an episode containing **one** pick-and-place buys 17.9 steps, and an episode containing
**four** buys roughly four times that at a fraction of the cost — no scene reset, no re-teach,
one continuous recording.

| target | post-flip steps needed | at 2 grasps/ep | at 4 | at 8 |
|---|---:|---:|---:|---:|
| h ≤ 3.5 (PR-03's own bar, MDE ~7) | 3 680 | 207 ep | 103 ep | 52 ep |
| h ≤ 2.0 (MDE ~4) | 11 271 | 634 ep | 317 ep | 158 ep |

The 2-grasp column is a check on the arithmetic, not a new result: 207 reproduces PR-03's own
"roughly 205 holdout episodes" for the same target, from the same measurement.

**The honest caveat, and it is a real one.** The bootstrap resamples **episodes**, not steps
(one grasp contributes a run of correlated steps). Extra grasps inside one episode reduce that
episode's own measurement noise but do not add an independent unit, so the columns above are
optimistic to an unknown degree — in the limit, `h` is floored by between-episode heterogeneity
over the episode count alone. Treat "grasps per episode" as a strong lever with diminishing
returns, not as a linear substitute for episodes. **The pilot measures the real exchange rate**
(§4, the exchange-rate diagnostic), and this table is superseded by that measurement
rather than defended.

**Recording plan.**

| set | episodes | grasps/ep | robot-time | purpose |
|---|---:|---:|---:|---|
| **D1-pilot** | 30 | ≥ 4 | ~30 min | run the gate in §3. Nothing else is recorded until it passes. |
| **D1** | 120 | ≥ 4 | ~2 h | overfit + first honest AC-07 arm |
| **D2** | 300 | ≥ 4 | ~5 h | 150 holdout / 150 train at ~4× PR-03's post-flip steps |

Robot-time is the data duration; operator time runs 3–4× that with resets and setup, landing D2
in `docs/ROADMAP.md`'s 10–30 h band. For calibration: the entire 402-episode GR00T corpus is
**95.1 minutes** of robot data (mean 14.20 s/episode).

## 3. The gate — run after D1-pilot, before anything else is recorded

```bash
.venv/bin/python scripts/screen_corpus.py --dataset datasets/d1-pilot --out runs/pr04/pilot.json
.venv/bin/python scripts/audit_gripper.py datasets/d1-pilot
```

`scripts/screen_corpus.py` is PR-02's screen as **committed, validated code**. It reproduces the
archived GR00T values (`--expect gr00t`, run 2026-08-02 on the committed 40-episode holdout):

| | measured | archived | Δ |
|---|---:|---:|---:|
| zero-delta MSE | 1.632760e-05 | 1.632760e-05 | **exact** |
| const-velocity MSE | 9.137664e-06 | 9.137664e-06 | **exact** |
| M1 | 0.6557 | 0.660 | −0.0043 |
| M2 | 0.3284 | 0.333 | −0.0046 |
| M3 | 2.0149 | 2.01 | +0.0049 |

The two zero-parameter rules reproduce to every digit; M1/M2 sit ~0.005 low because this ceiling
is slightly stronger (5.361517e-06 vs 5.431371e-06). **This is the like-for-like check PR-03's
gate 1 could not do**, because the code it needed to compare against had never been committed.
Note what that also means: the archived M1/M2/M3 were themselves produced by uncommitted code,
so agreement to ±0.005 is evidence these two implementations agree, not proof either is
canonical.

**Ceiling strength moves G1 and G2 in opposite directions, and that is what makes the pair
sound.** A stronger ceiling shrinks M2 = `ceiling/zero` (harder to pass G2) but also *grows* the
denominator of M1 = `(zero−constvel)/(zero−ceiling)`, shrinking M1 and making G1 *easier*. So a
better ceiling search alone can flip G1: on this very corpus, a hypothetical perfect blind
ceiling would give M1 = **0.4404**, which **passes** G1 on the dataset this document uses as its
canonical R1 failure. It would simultaneously give M2 = 0.000, failing G2 outright. **Neither
gate is trustworthy alone; requiring both is what cannot be gamed by fitting a better ceiling**,
because strengthening it trades one gate against the other. Any report quoting M1 without M2 is
misreading this screen.

| gate | clause | bar |
|---|---|---|
| **G1** | `m1_momentum_share` | ≤ 0.45 |
| **G2** | `m2_blind_unreachable` | ≥ 0.45 |
| **G3** | `m3_transitions_per_episode` | ≥ 2.0 |
| **G4** | `ceiling_dominates` | must be `true` |

**G3 is a floor, not the target.** The plan asks for ≥ 4 grasps per episode; the gate asks only
that the channel be alive, because its job is to catch a broken recording or conversion — T-31's
converter scored **0.00** on data that physically contained 2.015 transitions per episode, and
the current corpus sits at 2.01. How many grasps the protocol *actually* delivered is measured,
not gated, by §4.

G4 is not a formality either: PR-03 shipped a "ceiling" a zero-parameter rule beat, and every
M1/M2 read off such a ceiling is void (M1's denominator can go negative). It is now reported on
every run.

### Verdicts

- **A — PASS (all four).** The protocol produces a corpus a blind baseline fails on. Record D1,
  then D2. This is the first corpus in the project on which an AC-07 verdict would mean
  something.
- **B — FAIL on G1 or G2, PASS on G3/G4.** The grasp channel is alive but the arm trajectories
  are still momentum. **Do not scale.** Change the protocol along R2/R3/R6 — randomize placement
  further, break the timing stereotypy, teleop more decisively — and re-pilot 30 fresh episodes.
  Each re-pilot costs ~30 minutes and is the cheapest experiment available.
- **C — FAIL on G3.** The recording or conversion has killed the gripper channel, exactly as our
  own converter did in T-31. Fix the pipeline; this is a bug, not a data property. Check
  `m3_transitions_by_hand` first: the screen scores whichever channel is live, so a zero here
  means *no* channel moved, not that the wrong one was read.
- **E — FAIL on G4** (`ceiling_dominates: false`). **Says nothing about the corpus.** The
  ceiling fit collapsed, so M1 and M2 are void — M1's denominator can be negative and the
  reported numbers are arithmetic on a broken quantity, not measurements. Refit the ceiling
  (widen `GAMMA_SCALES`/`LAMBDAS`, check for too few training episodes) and re-run. This branch
  exists because the first version of this document routed a G4 failure to verdict C, which
  would have sent someone to debug a recording pipeline whose gripper channel had just passed
  its own gate.
- **D — FAIL on G1/G2 after three protocol revisions.** Then the finding is about **the task
  family, not the dataset**: single-arm tabletop pick-and-place at 30 Hz may not admit a metric
  a blind extrapolator fails. That is a genuine, publishable negative and a much stronger claim
  than anything currently held — and it redirects to task selection (contact-rich, multi-stage,
  or branch-point tasks), not to more episodes.

**Ties go to B**, the cheap branch. A verdict that differs between the pilot and a re-pilot on
the same protocol is no verdict.

## 4. What the pilot also measures, without any extra recording

- **exchange rate:** post-flip steps per episode at the achieved grasps/episode — the real
  exchange rate for §2's table, which is currently an assumption.
- **R2:** cross-validated R²(grasp pose | t=0 state), episode-disjoint folds. The method is
  `PR-01-TASK-VARIATION.md`'s, which reported +0.6136 on the current corpus.
- **R3:** clock-only post-flip accuracy via `scripts/bench_grasp_anticipation.py`.
- **R6:** DCT energy above 2.8 Hz along the 16-step axis, as in `PR-02-RESULT.md`.

These are diagnostics, not gates — they say *which* protocol change to make when the verdict is
B, rather than leaving it to taste.

## 5. What this document does not claim

- **Not that more data fixes the model.** T-32/I-8 — the data-scaling curve — has never run, so
  "not enough data" remains the untested explanation it has always been. This spec makes the
  *next* corpus interpretable; it does not retroactively license the claim that the current one
  was merely too small. `docs/improvements.md` is explicit that three rungs on one task cannot
  separate "needs more episodes" from "needs more tasks".
- **Not a robot claim.** Every gate here is offline. E3 is the only thing that speaks to
  hardware, and no offline number substitutes for it.
- **Not that the GR00T corpus was wasted.** It produced five interpretable negatives, the
  gripper-conversion bug (T-31), and every threshold in this document.
- **Not a fixed budget.** The 30/120/300 split is sized from PR-03's measured power ladder, and
  §2's caveat says plainly which part of it is extrapolated.
