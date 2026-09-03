# WAM — MVP Tasks

Derived from PRD roadmap (M0–M4 = MVP; M5/M6 post-MVP). Order is strict: don't start a milestone
before the previous exit criterion is met.

**This file is the milestone index. Each task is one file under `.mc/tasks/`** — MissionControl
(`mc`) format, one entry per file, its full record inside. The lines below link to them; the prose
that used to live here moved into those files unchanged. Conventions and the `mc` commands worth
knowing: `.mc/README.md`.

```bash
mc task next     # -> the next actionable critical task (T-39 re-reported N 2026-08-17)
mc task board    # backlog / todo / done
mc show T-16     # the whole T-16 record
```

> **This index covers `T-NN` only.** Since 2026-08-15 there are two sub-projects with their own task
> namespaces and their own indexes, which `mc` does not see:
> **[`subprojects/edge-wam/`](subprojects/edge-wam/TASKS.md)** (`E-NN` — image in, action out, on the
> robot) and **[`subprojects/data-factory/`](subprojects/data-factory/TASKS.md)** (`D-NN` — more and
> better training data from real episodes). Overview:
> [`subprojects/README.md`](subprojects/README.md). `T-040`/`T-041`/`T-042` stay here and are cited
> from there by ID. **No sub-project starts a training run before T-39 reports.** It reported **VOID (labels)** on
> 2026-08-16 — the corpus's own action column scores below L0 under our scorer
> ([`PR-07-RESULT.md`](docs/preregistration/PR-07-RESULT.md)) — so the gate's own premise came back
> negative rather than clear. Releasing it is a decision, not a formality.
> **RE-REPORTED 2026-08-17 under `T39_RULE_V2`: `VERDICT N`**
> ([`PR-07-V2-RESULT.md`](docs/preregistration/PR-07-V2-RESULT.md), job 188408). With step 0 anchored
> on the previous command, G0 **passes** — `oracle_action` **+68.10 %** L1 / **+75.40 %** L2, L4 — so
> the VOID premise above is superseded by measurement, not by argument. The policy arm then ran for
> the first time and scored **−239.69 %** on the holdout and **−186.73 %** on episodes it *trained
> on*: it never fit. **N is a verdict, so `T40_RULE_V3` §5.3's "T-39 has reported" is satisfied and
> PR-08 §8 item 7 is closed.** (Items 2, 3 and 4 have since closed too — item 2 on 2026-08-22
> under `T40_RULE_V7`, items 3 and 4 on 2026-09-01. **§8 stands at 7/7 as of 2026-09-01**, and the
> first chunk was released by the owner on 2026-09-02 —
> [`PR-08-DET-2026-09-02`](docs/preregistration/PR-08-DET-2026-09-02-the-first-real-chunk-released.md).
> The 2026-08-27 sprint page is superseded and says so twice in its own header.)
> N licenses
> "the corpus is the finding, the next move is the *kind* of data", and forbids reading it as
> refuting any WAM design. **It still does not license training: that remains the owner's call.**

> **Build status (2026-08-15):** M0–M4 code-complete and tested — **1 638 passed, 32 skipped,
> 6 failed, 10 errors, 19 min 49 s** (`.venv/bin/python -m pytest -q`). **Every one of the 16
> non-passes is a missing gitignored local artifact, not a code regression**, and they are listed
> here so the next clone does not re-derive them: 10 errors + 2 failures
> (`tests/test_kinematics.py`, `test_versioning.py::…test_view_sim_honours_every_robot_config_field…`)
> want the vendor MuJoCo model `assets/mujoco/unitree_g1/g1_with_hands.xml` —
> `.venv/bin/python scripts/fetch_g1_model.py` fetches it; 4 failures (`tests/test_runtime.py`,
> the rollout-CLI contract tests) want `runs/d1-overfit-seed0/checkpoint.safetensors`. Both
> `assets/` and `runs/` are gitignored, so **a fresh checkout cannot be green** — these tests fail
> rather than skip, which is a wart worth fixing, not evidence of breakage. Two further traps cost
> real time on 2026-08-15 and are recorded rather than re-learned: **`python` is anaconda's, not
> the project's — always `.venv/bin/python`**, and **piping pytest to `tail` masks its exit code**
> (use `${PIPESTATUS[0]}`). Earlier figures in this file and in `README.md` — 583, 604,
> 617, 861, 1 091, 1 618 — are the counts at the dates they were written and are left where they
> sit as part of those entries' record; **this line is the current one.** The closed
> loop now also runs on **MuJoCo contact physics + rendered pixels** (T-25, `docs/sim.md`) and the
> DDS wire layer is exercised in an arm64 container (T-25a, `docker/dds/README.md`) — but neither
> is covered by the test suite, and neither replaces real teleop data (D1/D2).
> Everything marked *hw* still needs real hardware, real teleop data (D1/D2), or an open decision.
> Ordered path to real usage: `docs/ROADMAP.md`.

> **Correction, 2026-08-16 (T-48).** The line above says **every** one of the 16 non-passes is a
> missing gitignored artifact. For 14 of them that holds. For the four `tests/test_runtime.py`
> rollout-CLI tests it is **half right, and the half it gets wrong is the interesting half.**
> `scripts/overfit_d1.py --dataset datasets/mock-d1 --run-id d1-overfit-seed0 --seed 0 --steps 400`
> regenerates the artifact in ~8 min on CPU and reproduces T-13's recorded result
> (`.mc/tasks/done/T-13-*.md:29`, "loss → 0.09 % of initial"; measured again at **0.10 %**). That
> takes the four failures to **two**. The remaining two do not want a file — they assert `rc == 0`
> from `scripts/rollout.py`, i.e. they require the checkpoint to **clear the E2 release gate**, and
> it does not: `safety_intervention_rate 1.000 > 0.1`, `intervention_kinds {'accel_limit': 60}`.
>
> That is a real, pre-existing gap between the D1 recipe and the E2 gate, not something the absent
> file was hiding. The data is clean — recorded ground-truth chunks peak at **1.47 rad/s²** against
> `ddq_max: 4.0` (`configs/safety/default.yaml`) and 0/25 trip the filter — while the model emits
> max |accel| **9.85** on its own training episodes and **18.1** on E2's static-frame probe
> (`src/wam/evaluation/e2_checks.py:166-172`). `overfit_d1.py`'s `build_training_config` sets
> `ActionLossWeights(action=1.0, gripper=0.5, smoothness=0.0, limit=0.0)` — smoothness is logged and
> weighted **zero** — and the action MSE plateau of ~1e-4 is an RMSE of 0.01/element, which at
> `dt = 0.05` is exactly 4.0 rad/s² of accel noise. **Training harder does not fix it:** a 2×
> better-converged model (`--lr 5e-4 --steps 4000`, train loss 4.8e-5) still fails with E1 holdout
> unchanged, so the residual is generalization, not optimization; 4000 steps at the recipe's own
> `lr 3e-3` diverges at step 1848. `docs/sim.md:73`'s "E2 static checks PASS on 16 probes" does not
> contradict this — that is `--policy dummy` on `mujoco_g1`, not this checkpoint.
>
> **The closed loop itself does work.** With the gate waived, `scripts/rollout.py --task sim:reach
> --contract-from-dataset datasets/mock-d1 --skip-e2 --rollouts 10` is **10/10, 0 estops, 0 watchdog
> timeouts, 0 deadline misses**, hitting a `REACH_TARGET_RAD` that was frozen from the *original*
> checkpoint's measured behaviour (`scripts/rollout.py:169`) — which is also what proves the
> regenerated artifact is the genuine one and not a file-existence stub.
>
> **Correction to the correction, 2026-08-25.** The T-48 entry above names the wrong limit, and the
> direction it is wrong in **strengthens** its own conclusion. It says the ground-truth chunks peak
> at 1.47 rad/s² "against `ddq_max: 4.0` (`configs/safety/default.yaml`)". That array never reaches
> this rollout: `build_safety_config` (`scripts/rollout.py:232-262`) takes `q_min/q_max/dq_max/
> ddq_max` from **the robot's declared envelope whenever the robot declares one**, and
> `configs/robot/mock.yaml:27` declares **`ddq_max: 8.0`**. So the limit in force is **8.0**, twice
> as permissive as recorded — and the model overshoots it anyway, by **2.3×**. Re-measured
> 2026-08-25 on both E2 probes: max demanded |ddq| **18.29** and **17.76** rad/s², with `accel_limit`
> firing on 4-5 of the 8 steps of each probe. Velocity is clean (max |v| 0.85 against `dq_max` 2.0),
> and `velocity_limit`, `joint_limit`, `nan_reject`, `schema_reject`, `state_reject` and the gripper
> gates never fire. **`accel_limit` is the only rule involved**, at `src/wam/safety/layer.py:226-233`.
> At `dt = 0.05` the limit means adjacent per-step joint deltas may differ by at most 0.02 rad.
>
> Two further things that were not in the T-48 record:
>
> - **The failure predates the tests.** Both were re-run in a throwaway worktree at `c59000e`, the
>   commit that introduced them (2026-08-01), and fail **identically**. There is no code regression.
>   Whether the *original* checkpoint passed is **not determinable** — `runs/` is gitignored
>   (`.gitignore:19`), so that artifact was never in git. T-48's `REACH_TARGET_RAD` argument is
>   suggestive, not conclusive.
> - **`safety_intervention_rate` is one gate name over two different metrics**, both against the
>   same 0.1 threshold: the static gate scores *fraction of probes touched at all*, bounded ≤ 1.0
>   (`src/wam/evaluation/e2_checks.py:313`), while the sim gate scores *events per cycle*, unbounded
>   (`:487`). A sim rollout measured 13 events over 12 cycles = **1.08**. That is a spec wart
>   independent of this bug, and `e2_sim_rollout_checks` is currently called from nowhere but
>   `tests/test_acceptance.py`, so nothing in-tree gates a rollout on it.
>
> **The closed loop still works** — `--skip-e2 --rollouts 2 --max-cycles 6`: 2/2 success, 0 estops,
> 0 watchdog timeouts, 0 deadline misses, min rate 4.3 Hz. The Safety Layer clamping jerk out of a
> noisy chunk and the task completing anyway is FR-07 behaving as designed.
>
> **What was changed in response, and what deliberately was not.** The two `tests/test_runtime.py`
> failures were tests *named for the `PolicyContract` record* that asserted `rc == 0` and so made
> the E2 release gate a silent precondition of a contract test. They now pass `--skip-e2` and assert
> what they are about; the four `--policy checkpoint` tests also gained the
> `skipif(not D1_CHECKPOINT.exists())` guard `tests/test_executor.py:561` already had, so a fresh
> clone skips instead of erroring. `tests/test_runtime.py` is **36 passed**. **No gate value moved:**
> `max_intervention_rate` is still 0.1, `ddq_max` is still the robot's 8.0, and the recipe gap is
> still open. Fixing the gap means turning on the zero `smoothness` weight at
> `scripts/overfit_d1.py:239` and/or retraining the head at `dt = 0.1` (which would also clear the
> sub-0.5 s FR-05 chunk warning) — **that is a training run and a shared gitignored artifact, so it
> is the owner's call, not a session's.** Relaxing `max_intervention_rate` or `ddq_max` to make a
> jerky model pass is the exact failure FR-07 exists to prevent and is not on the table.

## M0 · Architecture & Safety Baseline (2–4 weeks)

- [x] **[T-01](.mc/tasks/done/T-01-canonical-robot-state-action-schema-as-versioned-code.md)** Canonical robot state/action schema as versioned code
- [x] **[T-02](.mc/tasks/done/T-02-core-interfaces-in-wam-interfaces-typed-versioned-protocols.md)** Core interfaces in `wam.interfaces` — typed, versioned protocols
- [x] **[T-03](.mc/tasks/done/T-03-mock-robot-adapter-dummy-policy-end-to-end-loop-without-hard.md)** Mock robot adapter + dummy policy — end-to-end loop without hardware
- [x] **[T-04](.mc/tasks/done/T-04-safety-layer-v0-limits-nan-inf-rejection-timeout-logged-inte.md)** Safety layer v0 — limits, NaN/Inf rejection, timeout, logged interventions
- [x] **[T-05](.mc/tasks/done/T-05-config-experiment-versioning-in-every-log-record.md)** Config + experiment versioning in every log record
- [x] **[T-06](.mc/tasks/done/T-06-resolve-open-decisions-od-01-02-03-07.md)** Resolve open decisions OD-01/02/03/07

**Exit:** interfaces + canonical schema approved, dummy policy loop with E-stop + logging works.

## M1 · Data Factory (3–6 weeks)

- [x] **[T-07](.mc/tasks/done/T-07-episode-format-writer-reader-mp4-parquet-checksummed-manifes.md)** Episode format writer/reader — mp4 + parquet + checksummed manifest
- [x] **[T-08](.mc/tasks/done/T-08-synchronized-capture-with-a-timestamp-tolerance-check.md)** Synchronized capture with a timestamp tolerance check
- [x] **[T-09](.mc/tasks/done/T-09-replay-visualization-episode-report-from-stored-data.md)** Replay + visualization — episode report from stored data
- [x] **[T-10](.mc/tasks/done/T-10-teleop-workflow-camera-kinematics-calibration-versioned.md)** Teleop workflow + camera/kinematics calibration, versioned
- [x] **[T-11](.mc/tasks/done/T-11-automatic-dataset-validation-gates-record-d0-d1.md)** Automatic dataset validation gates; record D0 + D1

**Exit:** reproducible synchronized recording + replay — met with mock episodes; real teleop episodes pending hardware.

## M2 · Action-Only Baseline (3–5 weeks)

- [x] **[T-12](.mc/tasks/done/T-12-state-encoder-chunked-action-decoder.md)** State encoder + chunked action decoder
- [x] **[T-13](.mc/tasks/done/T-13-action-only-policy-overfit-d1.md)** Action-only policy — overfit D1
- [x] **[T-14](.mc/tasks/done/T-14-offline-eval-e1-on-holdout-episodes.md)** Offline eval E1 on holdout episodes

**Exit:** overfit proof — pipeline and action space are learnable. ✅ (synthetic D1 + real G1 data; generalizes to unseen episodes)

## M3 · World-Action Prototype (6–10 weeks)

- [x] **[T-15](.mc/tasks/done/T-15-backbone-adapters-behind-one-interface-tiny-wan-i2v-flux3-st.md)** Backbone adapters behind one interface — tiny, wan_i2v, flux3 stub
- [x] **[T-16](.mc/tasks/done/T-16-action-encoder-joint-video-action-flow-matching-training.md)** Action encoder + joint video/action flow-matching training
- [x] **[T-16a](.mc/tasks/done/T-16a-make-t-16-runnable-on-the-real-backbone.md)** Make T-16 runnable on the real backbone
- [x] **[T-17](.mc/tasks/done/T-17-loss-monitoring-gradient-checks-and-divergence-detection.md)** Loss monitoring, gradient checks and divergence detection
- [x] **[T-24](.mc/tasks/done/T-24-cosmos3-nano-frozen-feature-probe-the-backbone-bake-off-vs-w.md)** Cosmos3-Nano frozen-feature probe — the backbone bake-off vs. Wan
- [x] **[T-18](.mc/tasks/done/T-18-ablation-harness-world-action-vs-action-only-ac-07.md)** Ablation harness — world-action vs. action-only (AC-07)

- [x] **[T-26](.mc/tasks/done/T-26-spatial-readout-probe-was-the-mean-pool-the-limitation.md)** Spatial-readout probe — was the mean-pool the limitation?

- [x] **[T-27](.mc/tasks/done/T-27-wam-bench-a-laddered-offline-benchmark-on-real-held-out-epis.md)** WAM-Bench — a laddered offline benchmark on real held-out episodes
- [x] **[T-28](.mc/tasks/done/T-28-scripts-eval-t16-py-score-a-fine-tune-on-a-provable-holdout.md)** `scripts/eval_t16.py` — score a fine-tune on a provable holdout

- [x] **[T-29](.mc/tasks/done/T-29-frame-history-at-inference-clear-the-confound-under-t-16-t-1.md)** Frame history at inference — clear the confound under T-16/T-18

- [x] **[T-30](.mc/tasks/done/T-30-read-the-chunk-out-of-the-flow-branch-not-the-regression-hea.md)** Read the chunk out of the flow branch, not the regression head

- [x] **[T-31](.mc/tasks/done/T-31-the-gripper-was-never-flat-we-flattened-it.md)** The gripper was never flat — we flattened it

- [ ] **[T-32](.mc/tasks/todo/T-32-data-scaling-curve-test-the-standing-not-enough-data-explana.md)** Data-scaling curve — test the standing "not enough data" explanation — *backlog, blocked on T-39 (~109 GPU-h)*
- [x] **[T-33](.mc/tasks/done/T-33-grasp-anticipation-on-the-restored-gripper-channel.md)** Grasp anticipation on the restored gripper channel

- [x] **[T-34](.mc/tasks/done/T-34-collection-spec-the-screen-that-gates-it.md)** Collection spec + the screen that gates it

- [x] **[T-35](.mc/tasks/done/T-35-dream-the-video-branch-through-wam-s-own-flow-and-measure-wh.md)** Dream the video branch through WAM's own flow, and measure whether it moves
- [x] **[T-36](.mc/tasks/done/T-36-re-run-the-dream-where-the-robot-moves-against-a-baseline-it.md)** Re-run the dream where the robot moves, against a baseline it can lose to

- [ ] **[T-37](.mc/tasks/todo/T-37-screen-the-2026-08-backbone-candidates.md)** Screen the 2026-08 backbone candidates — *todo; the screen is written, the probe is not run*

- [x] **[T-38](.mc/tasks/done/T-38-wan-vs-cosmos-as-one-experiment-at-three-corpus-sizes.md)** Wan vs. Cosmos as one experiment, at three corpus sizes

- [x] **[T-39](.mc/tasks/done/T-39-the-positive-control-this-project-has-never-had.md)** The positive control this project has never had — **re-reported 2026-08-17 under `T39_RULE_V2`: `VERDICT N`.** G0 passes on the repaired anchoring — `oracle_state` +100.00 %, `oracle_action` **+68.10 %** L1 / +75.40 % L2 (L4) — and the policy scores **−239.69 %** on the holdout and **−186.73 %** on episodes it trained on. The corpus/bar is the finding; more of *this* data is not the story. [`PR-07-V2-RESULT.md`](docs/preregistration/PR-07-V2-RESULT.md) · superseded: [`PR-07-RESULT.md`](docs/preregistration/PR-07-RESULT.md) (VOID, 2026-08-16, `−359.41 %` — the defective instrument)

- [x] **[T-044](.mc/tasks/done/T-044-is-the-label-command-mismatch-a-timing-convention.md)**
  Is the label/command mismatch a timing convention, or is it the space? — **reported 2026-08-16:
  `J`.** Pre-registered as
  [`PR-10-label-anchoring-delay-sweep`](docs/preregistration/PR-10-label-anchoring-delay-sweep.md),
  rule `T44_RULE_V1`, before the driver existed; result in
  [`PR-10-RESULT-T-44.md`](docs/preregistration/PR-10-RESULT-T-44.md). Zero GPU-hours. Both G0 gates
  passed — `oracle_state` +100.00 %, and the `d=0` bridge reproduced PR-07's −359.41 to −359.4078
  (drift +0.002 pp). **No delay in `d ∈ {−4 … +4}` clears L1**, so the anchor is not the defect.
  **But a real offset is there:** `d* = −2` on *both* holdout halves independently, worth **+28.81 /
  +30.35 pp** — material by the borrowed 10 pp floor and replicated out-of-sample — and it closes
  only ~11 % of a ~254 pp gap. `smoothness_ratio` moves 6.21 → 5.67 against a gate of 2.0: a shift
  re-indexes a signal, it cannot smooth one. **The live question is now `smoothness_ratio` and
  PR-04's collection spec — what *kind* of data — not another anchoring fix and not another model.**
  ⚠️ **Duplicate PR number, needs a human decision:** a peer session independently pre-registered
  and ran the same experiment the same afternoon as
  [`PR-10-anchor-delay-sweep`](docs/preregistration/PR-10-anchor-delay-sweep.md) (rule
  `PR10_RULE_V1`, verdict `P (partial)`, result in `PR-10-RESULT.md`). Wasteful, and also an
  unplanned **independent replication that holds**: different drivers, different chunk sets,
  different rules — same `d* = −2`, same ~29–30 pp gain, same `d = −1` optimum on L2, same failure
  to clear L1. The cross-comparison is a table in `PR-10-RESULT-T-44.md`.

- [x] **[T-045](.mc/tasks/done/T-045-is-the-residual-the-arm-s-low-pass-filter-or-is-it-content.md)**
  Is the residual the arm's low-pass filter, or is it content? — **reported 2026-08-16: `R`.**
  Pre-registered as [`PR-11`](docs/preregistration/PR-11-command-lowpass-sweep.md), rule
  `T45_RULE_V1`, before anything was filtered; result in
  [`PR-11-RESULT.md`](docs/preregistration/PR-11-RESULT.md). Zero GPU-hours, 29 evaluations.
  All three gates passed, including the new **G0.3** — the filter provably reaches the array — which
  matters because an inert filter yields a flat grid that reads as exactly this verdict.
  **No cutoff clears L1 at either anchor.** Best cell is 2 Hz, worth **+4.63 pp against a 224.89 pp
  deficit — about 2 %**, less than half the borrowed 10 pp floor.
  **The diagnostic underneath is bigger than the verdict:** a **1 Hz cutoff on a 30 Hz signal moves
  `smoothness_ratio` only 5.675 → 5.381**, which should be impossible if the excess jerk were
  high-frequency, while `horizon_ratio` sits at 0.006–0.008 at *every* cutoff. Read (as
  interpretation, not measurement) as a **level offset at the chunk's anchor** — meaning
  `smoothness_ratio` may have been misread in every document citing it, this project's L4
  *moves-like-a-demo* gate included. **Both cheap repairs to the label space are now priced:
  anchor ≈ 11 %, filter ≈ 2 %.** The next thing is not a third repair — it is PR-04's collection
  spec. Two follow-ups named and deliberately **not** run after the verdict was known: the per-step
  error profile, and an audit of what `smoothness_ratio` actually measures.
  ✅ PR-11 was **claimed across live sessions before it was written** — the fix for the PR-10
  duplication above, which remains the user's to resolve.
  **Both follow-ups have since been answered, and the interpretation above is now measurement with
  its direction inverted — see T-046.**

- [x] **[T-046](.mc/tasks/done/T-046-step-0-is-the-whole-defect-does-homogenising-it-repair-the-.md)**
  Step 0 is the whole defect. Does making it homogeneous repair the labels? — **reported
  2026-08-16: `C`.** Pre-registered as
  [`PR-12`](docs/preregistration/PR-12-step-zero-anchor-heterogeneity.md), rule `T46_RULE_V1`,
  **claimed across live sessions before it was written**.
  **Read off the code, not hypothesised about the data**, which is what separates it from PR-10 and
  PR-11: `convert_lerobot_g1.py:365` builds all sixteen target steps as `q[s+t+1] − q[s+t]`, while
  `eval_t39_baseline.py:254-255` builds **step 0 alone** as `q_cmd[0] − q_state[s]` — the only
  element in either chunk that subtracts a quantity of one kind from a quantity of another. A
  steady-state tracking offset cancels in every other difference and survives at full magnitude
  there.
  **The diagnosis is measured, and it is a peer session's prior work** (`docs/smoothness-ratio-audit.md`,
  `scripts/audit_smoothness_ratio.py`): index 0 carries **96.7 % of the predicted jerk sum** against
  the target's flat **6.6 % (= 1/14)**, and `smoothness_ratio` with index 0 dropped from **both**
  arms is **0.2747**, against a published 7.7011. So `benchmark.py:538` is arithmetically **right**
  — the metric is correct and the vector it is fed has one corrupted element, which is a worse
  failure mode than a wrong formula because it survives every unit test the metric has — and **the
  published direction is inverted**: over steps 1–15 the command is ~3.6× *smoother* than the
  executed trajectory, not 8× jerkier, which crosses spec 0.2.0's two-sided L4 band and fails on the
  **bland** side.
  **PR-12 therefore registers exactly one prediction**, because a prediction whose answer already
  exists is not a pre-registration: **P2 — `targets[0] = q_cmd[0] − q_cmd[−1]` clears L1.** That
  anchoring is *identical* to the current one under perfect tracking, so it is not a change of
  premise but the same premise made robust to the premise failing; it is available at inference,
  since a policy knows its own last command; and its cost is that the chunk loses its only tie to
  the measured state.
  **P2 HOLDS, AND THE WHOLE FINDING IS ONE LINE** — per-step MSE, half A, `d = −2`:
  unmodified `[4.79e-04, 3.34e-06, 3.26e-06, …]`, V-chain `[3.51e-06, 3.34e-06, 3.26e-06, …]`.
  **Step 0 carried 143× the error of its neighbours**, G0.3 measured rows 1–15 at exactly
  `0.000e+00`, and that one element was **90.10 %** of the summed per-step MSE.
  **Held-out half B: −379.68 → +69.15** (gain **448.82 pp**), and `skill_vs_zero` **+82.91** rules
  out shrinkage — the repeat and zero baselines are *numerically identical* between cells, because
  the targets never changed. Result in
  [`PR-12-RESULT.md`](docs/preregistration/PR-12-RESULT.md), artifact `runs/t46-step-zero/probe.json`.
  **Corroboration nobody aimed at:** PR-10 registered that a correct anchoring would drive
  `horizon_ratio` toward 1.0 and recorded that the *delay* shift failed it; V-chain at `d = 0`
  gives **1.0021**.
  **It retires one of our own results:** under the unmodified anchoring `d = −2` beat `d = 0`, read
  by both PR-10 runs as a real ~67 ms controller lead. Under V-chain the preference **flips**. So
  **PR-10's ~11 % and PR-11's ~2 % were fractions of a deficit that was ~90 % a single
  subtraction.**
  Registered in advance and duly arrived: `smoothness_ratio` **0.276** is top-rung **L4** under spec
  0.1.0 and *below* spec 0.2.0's two-sided floor — the command is ~3.6× **smoother** than the
  executed trajectory over steps 1–15.
  Licenses a defect report against `commanded_to_chunk`'s step-0 anchoring, carrying its stated
  cost. Licenses **nothing** about GR00T or any policy, does **not** discharge T-39's `VOID`, and
  touches neither `benchmark.py` nor `docs/benchmark.md`. **Next is a code question, not another
  label experiment:** `commanded_to_chunk` is the *evaluation* adapter, and the same step-0 question
  has to be asked of the **training** path and the **runtime executor**.

- [x] **[T-047](.mc/tasks/done/T-047-does-t-39-s-void-survive-its-instrument-being-repaired.md)**
  Does T-39's `VOID` survive its instrument being repaired? — **reported 2026-08-16: `W`.**
  Pre-registered as [`PR-13`](docs/preregistration/PR-13-t39-g0-rederivation.md), rule
  `T47_RULE_V1`, with §4a's magnitude prediction committed **before the driver produced a cell**;
  result in [`PR-13-RESULT.md`](docs/preregistration/PR-13-RESULT.md). Zero GPU-hours, ~2 min.
  **All four gates passed on the first run, and the bridge reproduced `PR-07-RESULT.md`'s
  `−359.41` to a drift of `+0.002 pp`** on the full 1 040 chunks — so one number in the artifact is
  the archive's, produced by this driver.
  On the anchorable set (1 000 = 1 040 − 40 episodes, both compared cells identical):
  unmodified **−344.54**, **repaired +68.10 L1 / +75.40 L2 / +82.04 vs-zero**, `horizon_ratio`
  **0.9717**, level **L4 moves-like-a-demo**.
  **`T39_RULE_V1`'s G0 blocking clause does not fire on a repaired instrument, so the premise of
  `VOID (labels)` is withdrawn by measurement.**
  **The registered magnitude held:** §4a predicted +55 to +75 and *slightly below* PR-12's half-A,
  because the anchorable set re-adds each episode's low-motion **last** chunk; measured **+68.10**.
  **Newly measured:** dropping each episode's **first** chunk moves the unmodified arm −359.41 →
  −344.54 — **14.87 pp from 40 of 1 040 chunks**, four percent of the set carrying a
  disproportionate share, exactly as an anchor defect at episode start predicts.
  **`W` is not a verdict on T-39 and cannot be** — `P`/`N`/`M`/`I` all require the policy arm, which
  never ran. It licensed correcting the withdrawn claim in the root `CLAUDE.md`,
  `subprojects/README.md` and `subprojects/edge-wam/CLAUDE.md`; **all three were corrected with the
  gate text preserved verbatim, because correcting a claim is not lifting a gate.** The training
  decision remains the project owner's. **Still unanswered:** whether a policy can *learn* these
  labels — this scores oracles, and testing it is a training run.

- [x] **[T-048](.mc/tasks/done/T-048-is-the-jerk-regulariser-the-cause-of-the-bland-chunk.md)**
  Is the jerk regulariser the cause of the bland chunk? — **reported 2026-08-19: `S` (shrinkage).**
  Pre-registered as
  [`PR-14`](docs/preregistration/PR-14-smoothness-ablation.md), rule `T48_RULE_V1`, registered
  2026-08-16 before the run produced a single step; result in
  [`PR-14-RESULT.md`](docs/preregistration/PR-14-RESULT.md), commit `19f4eee`. **Zero Discoverer+
  GPU-hours** — local RTX 5090, ~6 h.
  `shape_moved` is a **conjunction, and it split**: `chunk_rms` **0.00226 → 0.00293** cleared its
  1.25× threshold while `smoothness_ratio` **0.3198 → 0.3548** fell far short of its 2× one
  (0.6396). **The chunks got bigger without getting jerkier**, so **the jerk regulariser is not a
  material cause of the bland output** and L2 shrinkage / mean-seeking survives as the explanation.
  Had the rule been written on `chunk_rms` alone it would have returned a positive for a model whose
  smoothness is still **0.71 of spec 0.2.0's floor** — the conjunction is what earned the verdict.
  **`S` licenses dropping the regulariser hypothesis, not adopting the mean-seeking one.**
  `l1_cleared` **FALSE** at **−3.4288**, so no `L` verdict fires; `l1_material` is TRUE at
  **+18.371 pp** against A's archived −21.80 — the closest any run has come to repeat-last-action
  and still on the wrong side of it — but PR-14 §3 registers the 5090-vs-H200 confound as live in
  exactly that positive direction, so it is a number and not an effect.
  **The run survived its own §6 validity clause**: an external `SIGTERM` at step 7385 was **resumed,
  not restarted**, and the log proves one continuous chain — the resume re-entered at step 7385
  *and* sampler position epoch 6 batch 1112 — so 20 000/20 000 is scoreable rather than `I`.
  Re-scored from stored predictions under both bench specs 2026-08-19: **56.0 / 36.0**, level **L0**
  under both, now the fifth row in [`docs/benchmark.md`](docs/benchmark.md).
  **The task ID is contested and the file records it rather than hiding it:** `T-48` was claimed on
  2026-08-16 both by `PR-14` and by the runtime-failures correction at line 58 of this file (commit
  `b8da4e3`), and neither minted a task. `T-048` is minted here for the pre-registration on the
  ground that a versioned rule cannot be amended to point elsewhere. **That is a tie-break, not a
  ruling** — the second such collision after PR-10, which remains the owner's to settle.

**Exit:** ablation machinery ready; first real AC-07 verdict recorded — at tiny scale the video branch hurts, so "video helps" now rests on the pretrained prior (T-16 LoRA). T-26 tested the one confound that could have moved that premise and did not move it, so the burden on T-16 is confirmed rather than assumed. T-27 then re-scored both real runs against trivial baselines and raised the bar T-16 must clear: beating the action-only baseline is not enough, because that baseline itself loses to repeat-last-action. T-28 then closed the gap that would have made a finished T-16 unreadable: the trainer does no eval, so `eval_t16.py` is the step that turns a checkpoint into a scorable `predictions.jsonl` — and it refuses to score a holdout it cannot prove was excluded. Neither GPU nor code is the blocker any more — Discoverer+ is scripted (`cluster/discoverer/`) and T-16a is done.

**T-16 has now run (2026-07-30), the confound found afterwards has been cleared and priced
(T-29, 2026-08-01), and the answer is still negative.** Scored in the mode it was trained in, the
5B LoRA reaches **L0 / 50.6** with `skill_vs_repeat_pct` **−21.80 %** — the published −32.4 % / 48.4
was the freeze-frame measurement, and every figure recorded before 2026-08-01 is a *tiled* figure
that should be labelled as such wherever it sits next to a history-mode one. The video branch has
still produced no positive evidence at any scale tried: tiny shared trunk (T-18, hurts), frozen
pretrained features (T-15/T-24/T-26, no signal past a state-only ridge), and now a fine-tuned
pretrained prior. All three on the *same* 402 success-only episodes of one task, with a gripper
channel our own converter had flattened (T-31).

**What T-29 settled, and what it did not.** `JointWorldActionModel.predict` tiled ONE camera frame
to the backbone's 9-frame context (`joint.py:388`), while training fed the real 9-frame window
ending at the chunk (`datasets.py:156`) — so every trained world-action number on record was scored
on a freeze-frame, and the baseline they lose to — repeat-last-action — is pure motion continuity.
Re-scoring the existing T-16 checkpoint in both modes (job 184648, both arms on one H200, one flag
apart) put a price on that: **+10.65 pp of `skill_vs_repeat_pct`, roughly a third of the gap, and
nowhere near the gate.** The tiled arm reproduces the archived `eval-latest/bench.json` to every
digit, so the A/B is clean and the published figure is confirmed to have been the freeze-frame one.
The verdict survived the correction; the number it was published as did not. The frozen-feature
probes were never affected (they build real windows and use a ridge, not `predict()`), so
T-15/T-24/T-26 stand either way. Result and decision rule: `docs/improvements.md` (I-7).

~~**Only `t16-lora-seed0` was re-measured.**~~ **Corrected 2026-08-01 — the other two were
re-measured the same day**, on a laptop CPU for zero allocation, and **neither moved**: −20.86 →
−20.88 % and −129.04 → −129.00 %. The ladder is single-mode and the table is a comparison again.
The confound is **backbone-specific**, which is the finding: a frozen frame costs the Wan
fine-tune 10.65 pp and the `tiny` runs ~0.03 pp, because that backbone does not use the frame axis
at all. So AC-07 reads, in distribution: the clean same-backbone pair is **−129.00 % (world-action)
against −20.88 % (action-only)** — the world branch costs 108 pp on `tiny` — and T-16 at −21.80 %
is **not** a clean ablation against it, differing in backbone *and* branch at once. The prior
recovers the 108 pp and buys nothing beyond it: **no measurable world-action advantage**, the same
answer as before, now about models rather than about frame windows (`docs/improvements.md` I-7).
The "measured out of distribution" hedge is spent on the frame
axis and **spent on the readout axis too** — T-30 has since reported, and the flow readout is 11.1×
worse than the regression one, so the number T-16 was scored on is the best readout we have.

**T-29 was never the only thing standing between us and that verdict (2026-08-01).** Three more
confounds were found and staged, all of them re-scoring or re-converting assets we already own, and
none of them needing a robot:

- **T-30** — the deployed readout is not the trained one either. The action *latent* reconstructs
  the holdout chunks at 8.10372e-07 against the deployed head's 1.21027e-05 and the 9.14e-06 bar
  T-16 failed. Whatever T-16 measured, it was not an encoder that cannot represent these actions.
  Its gate — run it after T-29 reports — is cleared; **ran 2026-08-01 as job 184670 and came back
  decisively negative.** Every flow arm sits below L0; the pre-registered mean-of-8 arm is 11.1×
  worse than the regression head and **7.4× above the order-blind floor**, so the sampler does not
  merely mis-time the chunk, it does not carry it. The latent still reconstructs at 8.10372e-07 —
  both facts hold at once, which makes this a retraining question, not a readout one.
- **T-31** — the gripper was never flat; the converter flattened it. 0.826 rad of real grasp,
  squashed to 0.157 by a mapping that assumes a joint range the hand never uses, then halved again
  by averaging in a hand that never moves. Fixed, audited, and it needed no new data at all.
- **T-32** — "not enough data" has explained every negative in this project and has never been
  tested. Three rungs, committed nested splits, and a rule that now holds the *cheap* conclusion
  to the same standard as the expensive one.

A fourth was found in our own tooling rather than in a model, and it is the one worth stating
plainly: **`eval_t16.py`'s split proof could not fail on any checkpoint that recorded its own
training set.** `train_episode_ids` and `dataset_snapshot_ref` are two fields of one
self-description, so hashing the episodes the checkpoint names and comparing the result to the
hash the checkpoint reports compared the checkpoint against itself. Demonstrated, not argued: a
checkpoint trained on all eight `mock-d1` episodes that declared it had trained on one printed
"split proven (disjoint): 1 train / 2 holdout, hash matches" and handed back the two episodes it
had trained on. Fixed by requiring an external witness — the reviewed, committed split file — on
that path, compared as a multiset because the hash is over the recorded sequence. Every eval job
now passes it, and a test reads the sbatch files, because the first version of this fix broke all
four cluster jobs while the suite stayed green.

None of those four changes a recorded number — T-29 already did that, and it is so far the only
item that has. All four change what the next number will mean.

## M4 · Real-Robot MVP (6–10 weeks)

- [x] **[T-19](.mc/tasks/done/T-19-closed-loop-runtime-receding-horizon-replanning-watchdog.md)** Closed-loop runtime — receding horizon, replanning, watchdog
- [x] **[T-20](.mc/tasks/done/T-20-inference-server-websocket-versioned-wire-protocol.md)** Inference server — WebSocket, versioned wire protocol
- [x] **[T-21](.mc/tasks/done/T-21-g1-robot-adapter-behind-a-swappable-transport.md)** G1 robot adapter behind a swappable transport
- [x] **[T-22](.mc/tasks/done/T-22-e2-kinematic-and-sim-checks.md)** E2 kinematic and sim checks
- [x] **[T-23](.mc/tasks/done/T-23-acceptance-harness-100-rollout-sim-run.md)** Acceptance harness + 100-rollout sim run
- [x] **[T-25](.mc/tasks/done/T-25-real-mujoco-simulation-of-the-g1-behind-the-g1transport-seam.md)** Real MuJoCo simulation of the G1 behind the `G1Transport` seam
- [x] **[T-25a](.mc/tasks/done/T-25a-docker-dds-conformance-track-on-a-real-cyclonedds-bus.md)** Docker DDS conformance track on a real CycloneDDS bus
- [x] **[T-25b](.mc/tasks/done/T-25b-adversarial-review-repairs-on-the-mujoco-track.md)** Adversarial-review repairs on the MuJoCo track
- [x] **[T-25c](.mc/tasks/done/T-25c-bounded-feed-forward-in-g1adapter-execute.md)** Bounded feed-forward in `G1Adapter.execute()`
- [x] **[T-25d](.mc/tasks/done/T-25d-lint-coverage-of-docker-checked-and-not-a-gap.md)** Lint coverage of `docker/` — checked, and not a gap

**Exit:** MVP acceptance criteria evaluated — 3 PASS / 0 FAIL / 4 pending hardware+data. The
pre-hardware surface is now much larger: physics + pixels via MuJoCo (T-25) and the DDS wire
layer via the arm64 container (T-25a). What is left genuinely needs the robot.

## Post-MVP

- M5: FLUX 3 Dev integration (backbone swap, license check, benchmark vs. fallback — AC-05)
- M6: generalization, video-only data, cross-embodiment, multiple future hypotheses (FR-11/12)
- [ ] **[T-040](.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md)** Cosmos-Transfer2.5
  photoreal augmentation *(**GENERATING, one chunk** — `docs/preregistration/PR-08-photoreal-augmentation.md`
  written 2026-08-06, rule `T40_RULE_V1`. Real-teleop path chosen; estimator calibrated against
  Isaac rather than the unlicensed Humanoid Everyday. **PR-08 §8 closed 7/7 on 2026-09-01**: the
  four that were open all landed — H200 throughput `1.6896 s/frame` (item 3), the measured
  `GEOM_TOL = 0.47857992441961017 px` against `EST_DRIFT_P95 = 0.36010037281174667 px` leaving a
  **`+0.1185 px`** G0b budget (item 4), the `emai/vla-training` consumer contract (item 2,
  `T40_RULE_V7`), and T-39's `VERDICT N` re-report (item 7). `PARTITION_CEILING_GPU_H = 2013.75`
  signed 2026-09-01; **one chunk released 2026-09-02**, `STAGE=1 STYLE_SET=train CHUNK_INDEX=1
  CHUNK_TOTAL=4` — `PR-08-DET-2026-09-02`. **Not released:** the other three chunks, stage 2, the
  eval set, the identity set. **Training remains unlicensed and is the owner's call**; every clip
  carries `NOT_TRAINING_DATA`. Open on the generate path: G0a/G0b have not run on the produced
  clips, and the clips' visual quality is under investigation —
  [`docs/investigations/`](docs/investigations/))*
- [ ] **[T-041](.mc/tasks/todo/T-041-cosmos-generator-finetune-on-g1.md)** Fine-tune a Cosmos
  generator on G1 data *(**RAN — verdict VOID on G0b, 2026-08-15.** Training completed to iteration
  500 with resume diffs and a non-empty export (G0c satisfied); the eval generated all 60 paired
  clips; the **VLM judge did not reach G0b's 20/20 on the calibration set**, so no verdict issues
  and PR-09 §6 forbids reading a VOID as a weaker pass. **The failure path was pre-registered** —
  §6: "If G0b fails, that is not a fallback, it is the required path" — so a **human scoring the
  same 80 blinded clips needs no amendment**, while repairing the judge and re-running it does.
  ~59 of §7's 122 GPU-h spent. The export is a **merged full model, 121 GB / 27 shards, not an
  adapter**, so it runs nowhere but Discoverer+. An exploratory apple pick-and-place set (job
  187623, 15 prompts × 2 arms, ~1.7 GPU-h) lives in `runs/t041-apple-variations/` and is **not
  PR-09 evidence** — `blinded: false, scored: false`, chosen after seeing the eval. Full record in
  the task file. Historical detail from the build-out follows.
  Pre-registered and built 2026-08-07.
  `docs/preregistration/PR-09-cosmos-super-finetune.md`, rule `T041_RULE_V1`;
  `scripts/prepare_cosmos_corpus.py` + 20 tests; `cluster/discoverer/90..95`. Recipe verified from
  `NVIDIA/cosmos@f76cd870` — LoRA rank 16 on the generation tower, revision pinned. **Goal fixed:
  embodiment fidelity**, on AppleToPlate (CC-BY-4.0) + the 13 `G1_Dex3_*` sets (Apache-2.0).
  `Humanoid-Everyday-G1` was added 2026-08-07 and **dropped again 2026-08-08 — the Hub 429s it
  without a token, so OD-09 is no longer engaged for T-041**; PR-09 §2 carries both dated notes.
  The captioner is NVIDIA's shipped `caption_from_video`, not ours to build.
  **Unblocked by OD-10** — PR-07 §7's Cosmos3-Super clause lifted, T-32 and Cosmos3-Edge still
  frozen, PR-07 itself unedited. Jobs still refuse to run unless `T041_FREEZE_LIFTED` names a
  reason, which lands verbatim in the artifact. **PR-09 §8 is closed except item 6**, which is a
  measurement the mandatory probe takes: 8-GPU VRAM on dgx1. **Status:** env built and weights
  staged (jobs 186281/186282, COMPLETED); **corpus 14/14 repos downloaded, 69 GB** across the
  self-resuming chain 186348–186350. **Two format blockers found 2026-08-08, both now fixed in
  code.** (1) *LeRobot v3.0* — the 13 `G1_Dex3_*` sets are v3.0, which concatenates episodes into a
  handful of mp4s and moves the boundaries into `meta/episodes/*/*.parquet` (jobs 186353/186354
  FAILED in 4 s). `prepare_cosmos_corpus.py` now reads both layouts, cutting each clip out by
  timestamp; three v3.0 traps are covered by tests, the sharpest being that **cameras roll over to
  new files independently** (at episode 50 of BlockStacking `cam_left_high` is in `file-001` while
  the other three cameras are still in `file-000`), so the file must be resolved per (episode,
  camera). (2) *AV1* — **every one of the 14 sources is AV1**, LeRobot's default. vLLM decodes video
  through OpenCV only, and that build opened each file, read the container header correctly, then
  failed every `cap.grab()`: job 186357 sent 372 requests, got **0 captions**, wrote 372 empty
  files and **exited 0**. The corpus is now transcoded to H.264 and `scripts/verify_clip_decode.py`
  re-checks it **with the captioner's own interpreter** before any caption is generated — ffprobe
  called the AV1 corpus valid throughout, so "is the file well-formed" and "can the decoder that
  will read it get pixels out" are different questions. **Preparation moved off the cluster
  2026-08-08** (`workstation/`, `configs/cosmos3/corpus_g1_embodiment.tsv`): all four failures so
  far were IO, format or scheduling, each costing hours of queue to learn something a workstation
  answers in seconds, and transcoding video is not what a GPU allocation is for. Jobs 92/93 are
  marked superseded, not deleted — their 69 GB download is a usable cache. **Corpus fetched to the
  workstation 2026-08-08: 14/14 sources, 26 GB** — one camera per source rather than the cluster's
  2–4, which is where the 69→26 GB comes from. **The resolution mismatch is closed (2026-08-08):**
  §5 generated 720p 16:9 against a corpus that is 640×480 4:3 throughout, so the adapter would have
  been evaluated in a geometry and scale it never trained on. Both arms shared the settings, so a
  false **P** was never possible — the risk was an **ambiguous N**, and G0b's real calibration
  clips are 640×480 besides. `t041_eval_selection.toml` now generates `480` / `4,3` = exactly
  640×480; moving the corpus instead was rejected (pillarboxing teaches black bars, cropping to
  640×360 discards the torso/arms/hands the `cam_left_high` choice exists to capture).
  `T041_RULE_V1` is unchanged — the verdict rule never mentioned resolution — and the amendment is
  dated in PR-09 §5, taken before any clip was generated. **Corrected 2026-08-09 by a second dated
  amendment in PR-09 §5:** `resolution` is not an output height but a key into
  `VIDEO_RES_SIZE_INFO`, whose 4:3 buckets are 320×256 / 736×544 / 1104×832 — `480`/`4,3` is
  736×544 and **there is no 640×480 bucket at all**, so the value registered above could never have
  done what it was registered to do. The settings now match TRAINING instead: `256`/`4,3` = 320×256,
  the geometry `vision_sft_super.py:272` pins and the TOML cannot override, and the one
  `max_sequence_length = 45056` is sized for. `fps` 24 → 30 in the same amendment, on its own
  evidence. Two items were left open in writing; **the first is now closed by a third dated
  amendment, 2026-08-09: `num_frames` 189 → 397.** The training durations were measured rather than
  quoted — `num_video_frames = -1` puts the loader in native-chunk mode and every window is written
  as the whole clip at interval 1, so the manifest's counts *are* the sequence lengths: 3432 train
  clips at 30 fps, min 249 / median 693.5 / max 1819, and **nothing at or below 189 in either
  split**. The "cap of 400" is also not a cap — `MAX_NUM_FRAMES["256"]` is only compared inside a
  `log.warning`, so 397 is chosen to stay inside a range NVIDIA states, not because anything would
  reject more. Of the legal `4N+1` values only 397 is in the distribution's interior (12.97 % of
  train clips are ≤ 397, against 4.22 % at 349, 0.20 % at 297 and **one clip** at 249), and 13.2 s is
  the closest reachable to the duration the structured-JSON prompt itself states (val median 25.3 s).
  The cost estimate is a back-of-envelope and says so: the 8-GPU benchmark column does not apply
  because `parallelism_preset = "throughput"` forces `cp = cfgp = 1` and the sbatch sends one payload
  per `torchrun`, giving ~40 s/clip at 189 against ~85 s/clip at 397 — a marginal **~45 min ≈ 6
  GPU-h** over 60 clips, while the term that actually threatens the 4 h wall is the **60 cold
  torchrun launches** (~90–180 min, unmeasured, and unmoved by frame count). Accepted because the job
  is restart-safe by construction (generation skips clips already written, the judge takes
  `--resume`, the job is `--requeue`); the mitigations are named in PR-09 §5 and **none is applied**.
  **Still open:** G0b's calibration clips are real 640×480 footage that must be downscaled to 320×256
  before the judge sees them — not done, and it blocks G0b. **Also recorded and not resolved:** §7
  budgets job `95` at 8 GPU-h = one hour on 8 GPUs, and every branch of the estimate puts the eval at
  25–35 GPU-h at 189 frames as much as at 397. **The corpus was deduplicated 2026-08-09 and a
  fourth dated amendment records it, this one against PR-09 §2 — train 3432 → 3133 clips, 14
  training sources → 13, val untouched at 30.** `g1-dex3-graspsquare-dataset` is a byte-for-byte
  copy of `g1-dex3-blockstacking-dataset`: the same six source mp4s by sha256, the same episode
  boundaries in 79 of 80 metadata columns over 301 episodes, differing only in the `tasks` string
  — which reads `"camera packaging"`, a third dataset's label. 299 duplicate pairs; 3462 clips,
  3163 unique sha256. **The weighting was not the problem. Four of the thirty pre-registered eval
  prompts (13 %) were byte-identical to TRAIN clips**, so the LoRA would have been scored on
  footage it had memorised — and only the LoRA arm, so the bias ran *toward* the registered
  hypothesis. Both holdout checks compared **uuids**, which really were disjoint.
  `scripts/dedupe_cosmos_corpus.py` deleted from train only (4 contaminating, then 295 redundant,
  keeping the lexicographically smallest uuid of each pair) precisely so `n = 30` and G0a's
  `>= 15/30` survive as registered instead of being renegotiated after the fact. No unique content
  was lost — corpus-wide unique sha256 is 3163 before and after — and GraspSquare now contributes
  zero train clips, its 2 val clips left in place because removing them is a re-split by another
  name. `MANIFEST_SHA256` is now
  `2af81b9997f0de42e3fee01600bf34c67b7cdcb86b8ac5ab1094e21dcf77c63e` (re-measured; the pre-dedupe
  `6bec507e2816…` is quoted rather than re-measured, that manifest being gone). The gate is hardened both
  ends: `check_prompts_are_held_out` and `make_t041_eval_prompts.py` now also refuse a prompt whose
  clip sha256 appears anywhere in train, from the sha256 the manifest already records. It passes on
  the current corpus and catches all four pairs on a reconstruction of the old one.
  **Workstation env built 2026-08-08**
  (step 00 green, idempotent, both repos at their pinned SHAs with clean trees, torch 2.10.0+cu128
  on an RTX 5090 sm_120). Three prerequisites were undeclared and each failed in a way that named
  the wrong culprit: no `git-lfs`, so the framework's *checkout* half-failed while `rev-parse` still
  reported the right SHA — `clone_at` now verifies a clean porcelain instead of trusting the SHA,
  in the cluster script too, **where the git-lfs install sat after the clones it was needed for**;
  no C compiler, so `uv sync --all-extras` died ten minutes in building evdev, a keyboard-teleop
  transitive that `--all-extras` gives no way to decline; and transformer_engine's no-toolkit path,
  which only triggers on a machine without a system CUDA toolkit — Discoverer+ has a module, a
  driver-only workstation does not — and looks for cudart under the CUDA 13 wheel name. The
  `curl | sh` uv bootstrap was removed rather than fixed: it was the one unpinned component in a
  pipeline whose premise is that everything is named by SHA. **Still open before any run:**
  whether 500 iterations is even one epoch cannot be read
  off the config — NVIDIA's `PackingDataLoader` batches by a 45 056-token budget with no sample
  cap, so the probe has to measure clips-per-iteration; how the prepared corpus reaches Discoverer+
  depends on the workstation's upstream and is undecided. Job 95 additionally needs 20 calibration
  clips nobody has picked yet)*
- [x] **[T-042](.mc/tasks/done/T-042-cosmos-inverse-dynamics-action-labels-for-the-g1.md)** Cosmos
  inverse dynamics — action labels for the G1 *(**closed 2026-08-15 by its own step 0**, written
  the same day. It was the follow-up `docs/backbone-eval.md` §3 never got: Cosmos 3's action port
  is **bidirectional** — forward dynamics, **inverse dynamics** (frames → the trajectory that
  produced them) and policy — so it is a real video-to-action labeller, and §4's input-only framing
  is correct for Predict2 but wrong for Cosmos 3. **Step 0 was free and decided the task: the count
  of real G1 footage we hold that is unlabelled and unlabellable is 0 clips.** The 3 554 episodes
  looked unlabelled only because `92_fetch_g1_corpus.sbatch` and `workstation/10_fetch_corpus.sh`
  fetched `meta/` + `videos/` and skipped `data/`; upstream all 14 repos publish the action
  parquets — 415 files, 647 MB, verified through the HF tree API without downloading. Recovering
  labels a fetch flag would download is not amortisation, and inferred labels are strictly worse
  than recorded ones. **No GPU hour spent, `PR-10` never written** — it was gated behind a *go*
  that never came. Step 0 also returned an answer it was not asked for: **3 152 of those episodes
  are the 13 `unitreerobotics/G1_Dex3_*` sets, every one `action float32[28]`** — the exact
  vocabulary this task meant to teach from scratch, already recorded and Apache-2.0. That is
  conversion work on recorded labels — route 1 — and is tracked as **T-043**. Re-open the day
  teleop produces video faster than it produces labels. Receipts: `docs/action-labels.md` §3b)*
- [ ] **[T-043](.mc/tasks/todo/T-043-convert-the-3152-g1-dex3-episodes-recorded-28-dim-labels.md)**
  Convert the 3 152 G1+Dex3 episodes — recorded 28-dim labels, route 1 *(**todo**, opened
  2026-08-15 out of T-042's step 0. **Measured from local metadata the same day, before writing
  code, and the measurement corrected a claim five documents were carrying.** The corpus: 13
  `unitreerobotics/G1_Dex3_*` sets, **3 152 episodes · 2 587 515 frames · 23.96 h @ 30 fps** —
  7.8× AppleToPlate's episode count and thirteen further tasks — Apache-2.0, LeRobot **v3.0**,
  `observation.state`/`action` both flat `float32[28]`, and **`cam_left_high` natively
  `[480, 640, 3]`**, which is exactly the GR00T N1.7 `ego_view` contract fixed the same day, so
  unlike `datasets/gr00t-apple-full/` (120×160) it feeds the consumer without re-derivation. It
  splits into two variants that may not be poolable: 6 sets `robot_type: Unitree_G1` / 4 cameras /
  hand limit **120°**, and 7 sets `Unitree_G1_Dex3` / 2 cameras / hand limit **100°** — different
  hand hardware, not different formatting. **THE BLOCK ORDER IS ARM-FIRST: `[0:14]` arm,
  `[14:28]` hand.** Every doc that mentioned this corpus said hand-first, carrying T-041's finding
  across from `USC-PSI-Lab/Humanoid-Everyday-G1` — a *different* corpus, LeRobot v2.1, separate
  `arm_joints`/`leg_joints`/`hand_joints` fields. Both facts are true of their own corpus, and the
  transplant would have caused the exact silent arm/hand transposition the warning exists to
  prevent. Two independent lines: `meta/stats.json` across all 13 sets gives **zero** one-sided
  dims in `[0:14]` against **4–10** in `[14:28]` railing at a clean 100.0°/120.0° mechanical limit
  (a finger, not a shoulder), and `vla-training/groot/modality_g1_dex3.json` independently declares
  `arms {0,14}` / `hands {14,28}`. **Left/right and intra-hand order stay unverified** — three
  mutually inconsistent orderings are on record, and a source wrong about the block order earns no
  trust about the finger order. Three structural gaps for the converter: the corpus has **no waist
  column** where canonical `g1.yaml` has one (a schema decision, not a coding one);
  `convert_lerobot_g1.py` is a **v2.1** reader while these are v3.0 (reuse
  `prepare_cosmos_corpus.py`, which already handles the independent per-camera file rollover); and
  the T-31 gripper apparatus must be **re-derived** for a different hand across two variants, never
  inherited. **Blocked on one thing, and it is not code: the action parquets were never
  downloaded** — the fetch scripts took `meta/` + `videos/` and skipped `data/`. 415 files, 647 MB,
  Apache-2.0, and an ask before it happens. **Not a substitute for T-39** — a bigger corpus under a
  recipe never shown to work is the same experiment at greater cost)*

## Open decisions (PRD §16) — resolved 2026-07-26

| ID | Decision | Status |
|----|----------|--------|
| OD-01 | Platform + gripper/hand | ✅ **Unitree G1 EDU4 + Dex3-1 three-finger hands.** MVP maps the canonical scalar gripper channel `[left, right]` to a grasp synergy in the G1 adapter; per-finger control is post-MVP |
| OD-02 | Action space | ✅ joint-delta primary, EE-delta supported in schema/safety |
| OD-03 | Cameras | ✅ standard G1 EDU4 sensor set (head RealSense D435i, RGB for WAM; depth/LiDAR unused in MVP) |
| OD-04 | Open fallback backbone + license | ✅ Wan2.2-TI2V-5B (Apache 2.0), verified on real weights (`docs/hf_jobs.md`). **Challenged and held (2026-08-05, T-37, `docs/backbone-eval.md`):** HunyuanVideo 13B/1.5 fails the licence criterion this row exists to apply — the Tencent Hunyuan Community License excludes the EU from its Territory — and Cosmos3-Nano's own T-24 probe peaked at 36.2 GB, over the 5090's 34.36. Only Cosmos-Predict2.5-2B survives the screen, on its pretrained action port; the gate that would reopen this row was `R²_joints > 0.456` **and** `R²_gripper > 0.881` — **corrected before the run and still unrun**: arm A is *fed* the past actions against a lag-1 autocorrelation of 0.927, so the bar is what a ridge reaches from the probe's own inputs with no video model at all, measured as `past_joint_proj + state` at **0.546 / 0.911** (48 episodes, three seeds, spread 0.002). The best frozen-backbone number ever recorded here is 0.4267 (T-38), so the corrected gate asks for 0.55 where nothing has cleared 0.43 |
| OD-05 | Training hardware + budget | ✅ free tier now (Mac MPS + ZeroGPU); for T-16 real-data training an account on EuroHPC **Discoverer+** (NVIDIA H200) — access verified 2026-07-27, **5 000 GPU-hours**, 4 h max walltime, runbook `docs/discoverer.md`; own RTX 5090 as fallback |
| OD-06 | FLUX 3 access, weights, fine-tuning rights | ⏳ deferred to M5 (post-MVP), nothing blocks on it. **Signal (2026-07-29):** BFL licences the FLUX.2 collection split — `FLUX.2-dev` (32B) is FLUX Non-Commercial, `FLUX.2-klein-4B/9B` are Apache 2.0. If FLUX 3 repeats that pattern, "FLUX 3 Dev" — the PRD's preferred backbone — fails the same commercial-licence criterion that decided OD-04 for Wan, and only a `klein` variant would qualify. Also note FLUX.2 is image-to-image, not video: adopting it means adopting ImageWAM's reformulation, not a drop-in `FlowBackbone` (I-6 in `docs/improvements.md`) |
| OD-07 | Teleoperation system | ✅ VR teleop (Unitree `xr_teleoperate` path; headset model — Vision Pro vs. Quest 3 — still to pick at purchase time) |
| OD-09 | **Dataset licensing** | ✅ **decided 2026-08-07 by the user: train on `USC-PSI-Lab/Humanoid-Everyday-G1` although it carries no licence.** This row is new — T-041 flagged that no open-decision covered *dataset* licensing (OD-04 is a **model** licence row and says nothing about data), so the risk had nowhere to be recorded. **What was accepted, stated plainly:** the corpus declares no licence in any field, tag, card or file, so default copyright applies and no permission is granted by the rightsholder — "unlicensed" is more restrictive than Apache-2.0, not less (evidence table in `.mc/tasks/todo/T-041-*.md` §Licence, five primary sources, checked 2026-08-06). **The basis for proceeding:** the EU TDM exception (DSM Directive Art. 4, §44b UrhG) permits text-and-data-mining on lawfully accessible works for any purpose including commercial, absent a machine-readable reservation — and there is none. **Its limits, which the decision does not cover:** Art. 4 covers the *mining*, not redistributing the corpus, and not distributing a model that can reproduce its frames. EmAI ships commercial products, so a model trained on this and then sold is a different question from training on it. **Not a legal opinion — no lawyer has reviewed this.** The licence request to the authors is **deferred, not cancelled** (`docs/outbound/humanoid-everyday-licence-request.md`, drafted, unsent). **Revisit before** anything trained on this corpus is distributed, sold or served to a customer. **Update 2026-08-08 — the decision stands but is currently unused.** T-041 dropped the corpus for an unrelated operational reason: the Hub rate-limited it (HTTP 429) on three consecutive jobs, the last transferring nothing, and fetching it needs an authenticated token the user chose not to place on a shared EuroHPC filesystem. **This is not a retraction** — the reasoning above was never the cause of the removal and applies unchanged if the corpus is wanted again. As of now nothing in this repo trains on unlicensed data, so no artifact carries the asterisk |
| OD-10 | **Lift PR-07 §7's Cosmos3-Super freeze** | ✅ **decided 2026-08-07 by the user: proceed with T-041 without waiting for T-39.** **Scope of the lift, deliberately narrow:** only the *Cosmos3-Super generation* clause. T-32 and Cosmos3-Edge stay frozen exactly as PR-07 §7 wrote them, and **PR-07 is not edited** — rules in this repo are versioned, never amended in place, so the freeze stands as written and this row is the decision taken against it. **What the freeze was protecting:** PR-07 §1's point that fourteen recorded negatives all compare a WAM variant against a *trivial* baseline, so none separates "our approach is wrong" from "nothing clears this bar on this corpus". T-39 is the first method with a published claim of success on the same data, and §7 froze generation on the grounds that generating anything before that may be answering the wrong question. **What was weighed:** (a) T-041 measures something T-39 does not bear on — whether the generator *draws a G1 rather than a generic manipulator*, a defect already recorded in `runs/backbone_eval/video/embodiment_grid.png`; (b) T-39 is not a short wait, it is **unsubmittable** with PR-07 §8 items 4–6 still open; (c) the price is small and pre-registered — PR-09 §7 caps the whole experiment at **122 GPU-h, 2.4 %** of the 5 000-hour allocation, with a mandatory probe that refuses to start the run if it will not fit. **What is accepted:** if T-39 later returns **N**, a **P** here is still a true statement about the generator but loses most of its downstream value — there would be no working policy to consume the frames. That is the risk, and it is bounded by (c). **This does not consume T-39's budget or change its priority: T-39 stays CRITICAL.** **How it is enforced:** every job in `cluster/discoverer/90..95` refuses to run unless `T041_FREEZE_LIFTED` names a reason, and that string is written verbatim into `run_metadata.json`, so every artifact this produces carries the decision that allowed it. **Revisit when** T-39 reports — a verdict of N is the trigger to re-read any T-041 result in its light, not to retract it |
| OD-08 | Vendor controller safety coverage | ⏳ verify during G1 bring-up (which functions Unitree's controller covers vs. WAM safety layer). Narrowed by T-25a: the wire-level damping e-stop is implemented and CRC-verified in the container, but the vendor RPC services (`MotionSwitcherClient`, `LocoClient().Damp()`) and the real limits/gains still need the robot. Sim gains (kp=500 + per-joint critical damping) live in `configs/robot/mujoco_g1.yaml` and were deliberately **not** copied into `g1.yaml` |
