# PR-14 — is the jerk regulariser why the WAM's chunks are too smooth to beat inertia?

**Task T-48. Registered 2026-08-16, before the run produced a single step.** Rule `T48_RULE_V1`.
Zero GPU-hours on Discoverer+; this runs on the local RTX 5090. Nothing is submitted.

## 1. Why this experiment exists

`runs/t16-lora-seed0` — 20 000 steps of Wan2.2-TI2V-5B LoRA on the 362 training episodes — is the
best-scoring WAM run on record and **it fails the one pre-registered bar.** In the frame mode it was
trained in (`--frame-history`, the "real window"): `skill_vs_repeat_pct` **−21.80 %**, level **L0**.
It does not beat causal repeat-last-action.

The diagnosis on record is a **shape**, not a score (`src/wam/training/joint.py:405-411`, measured
over the 1 040 archived holdout chunks):

- regression-head chunk RMS **0.00226** against the demonstrations' **0.00404** — a **44 %
  magnitude shortfall**;
- `smoothness_ratio` **0.29** — the prediction is **3.4× smoother than a real demonstration**.

`docs/benchmark.md` rules out a bounded-output artifact (max `|target|` 0.0192 against a `tanh`;
`limit_penalty` bites only outside ±0.95, so neither bound is active) and leaves **two** live
explanations, each of which produces small smooth chunks on its own:

1. **the one-sided jerk regulariser.** `weights.smoothness = 0.01` is applied by
   `JointTrainer.compute_losses` to `out["decoded_targets"]` — the regression head's output — and
   to **nothing else** (`src/wam/training/joint.py:766`, weighted at `:777`). The flow branch is
   never charged for jerk.
2. **plain L2 shrinkage.** `action_reg` is an MSE against targets whose own RMS is 0.004, under
   AdamW `weight_decay`. A head that under-shoots magnitude uniformly is indistinguishable, on RMS
   and jerk alone, from one that averages modes.

Both the docstring and `docs/benchmark.md` name the same separator, and both say explicitly that it
is not available by re-reading the table:

> "Separating any of the three needs an intervention — retrain at `weights.smoothness = 0` — not a
> re-read of this table."

**This is that retrain.** It is registered here before it runs because a gate rewritten after seeing
its output is not a gate.

## 2. What is being claimed

> **Removing the one-sided jerk regulariser materially changes the predicted chunk's shape** — it
> becomes jerkier and its magnitude moves toward the demonstrations'. Whether that is enough to
> clear **L1** (`skill_vs_repeat_pct > 0`) is a separate question and is registered separately.

Two questions, deliberately not conflated. The first is a mechanism question and this experiment can
settle it. The second is the project's actual bar and this experiment may well answer it *no*.

## 3. The arms, and the confound I am accepting on purpose

| arm | `weights.smoothness` | steps | where |
|---|---:|---:|---|
| **B — treatment** | **0.0** | 20 000 | run now, locally |
| **A — control** | 0.01 | 20 000 | **archived** `runs/t16-lora-seed0`, cluster |
| **A′ — local control** | 0.01 | 20 000 | **conditional; see below** |

Everything else is held identical by sharing the artifact rather than by copying a value:
`configs/training/joint_wan_gr00t_5090_nosmooth.yaml` is `joint_wan_gr00t_5090.yaml` with **one
body line changed** (`diff` of the non-comment bodies is that single line), the same
`configs/model/wan22_ti2v_5b.yaml`, the same `configs/splits/i8_train_362.txt` (362 episodes), the
same `configs/splits/t18_holdout_episodes.txt` (40 held out, zero overlap), seed 0, camera `ego`,
effective batch 8.

**B against archived A is confounded** and this document says so before the number exists. B runs
on an RTX 5090 with torch 2.13.0+cu130 at `batch_size 2 × grad_accum 4`; A ran on an H200 at
`batch_size 8 × grad_accum 1`. The loss is normalised by `1/len(batches)`
(`scripts/train_t16_lora.py:774,:779`) so the optimizer update is arithmetically the same, and
`config_hash` differs regardless because `batch_size` is part of `JointTrainingConfig` — AC-04
traceability is intact, "same experiment ⇒ same hash" is not.

**The design is asymmetric on purpose, and that is what makes one run enough:**

- If **B fails to move the shape**, the regulariser is not the cause and the confound is *moot* — a
  hardware difference cannot manufacture a null. Verdict `S`, one run, done.
- If **B moves the shape**, the confound is live and **A′ becomes mandatory before any causal
  claim.** Verdict `R-PENDING` until A′ runs; `R` only after.

So a negative costs ~6 h and a positive costs ~12 h, and neither is read as more than it is.

## 4. The rule — `T48_RULE_V1`, fixed here, in git, before the run

Scored by `scripts/eval_t16.py` on the 40-episode `t18` holdout, **in `--frame-history` mode**
(the mode the model is trained in; the archived tiled numbers are a train/inference mismatch and are
not comparators here). The split is *proven* against the trainer's `dataset_snapshot_ref`, not
asserted — the evaluator refuses to score on a mismatch.

```
MATERIAL_FLOOR_PP = 10.0     # borrowed from I8_RULE_V3 for the sixth time, not coined here,
                             # so the choice of floor cannot become the finding.

A_SMOOTHNESS_RATIO = 0.3198  # archived, t16-lora-seed0, --frame-history
A_SKILL_VS_REPEAT  = -21.80  # archived, t16-lora-seed0, --frame-history
DEMO_RMS           = 0.00404 # demonstrations, joint.py:409
A_CHUNK_RMS        = 0.00226 # regression head, joint.py:409

shape_moved = (B.smoothness_ratio >= 2 * A_SMOOTHNESS_RATIO) and (B.chunk_rms >= 1.25 * A_CHUNK_RMS)
l1_cleared  = (B.skill_vs_repeat_pct > 0)
l1_material = (B.skill_vs_repeat_pct - A_SKILL_VS_REPEAT) >= MATERIAL_FLOOR_PP
```

`shape_moved`'s two thresholds are **derived, not chosen**: `2×` is the ratio between spec 0.2.0's
two-sided floor (`MIN_SMOOTHNESS_RATIO = 0.5`, `src/wam/evaluation/benchmark.py:71`) and A's
measured 0.3198 is 0.64 of it — doubling A lands the arm inside the spec's own admissible band
rather than at a number I picked. `1.25×` closes half the measured 44 % magnitude shortfall
(0.00226 → 0.00283 against 0.00404). Neither is tuned and neither moves after this commit.

### Verdicts, in precedence order `I → S → R-PENDING/R`, with `L` reported orthogonally

| verdict | condition | what it licenses |
|---|---|---|
| **`I` invalid** | a run died, OOM'd, drifted config, or the split check failed | nothing. Re-run or report the failure. |
| **`S` shrinkage** | `not shape_moved` | the jerk regulariser is **not** a material cause of the bland chunk. L2 shrinkage / mean-seeking survives as the explanation. Licenses dropping the regulariser hypothesis, **not** adopting the other one. |
| **`R-PENDING`** | `shape_moved`, A′ not yet run | **no causal claim.** Licenses running A′ and nothing else. |
| **`R` regulariser** | `shape_moved` in B **and** reproduced against local A′ | the regulariser is a material cause. Licenses a follow-up on the loss inventory. |
| **`L` (orthogonal)** | `l1_cleared` | **the WAM beats causal repeat-last-action for the first time.** Reported whatever R/S says. |
| **`L-MATERIAL`** | `l1_cleared and l1_material` | as `L`, and the move is above the borrowed floor. |

**`L` is not implied by `R` and `R` is not implied by `L`.** A jerkier model that still loses to
inertia is `R` + not-`L`, and it is a real result — it would mean the regulariser shaped the output
without being what stands between this model and the bar.

## 5. What this cannot answer, written down before it runs

- **Nothing about GR00T.** No vendored model is loaded, trained or consulted. PR-07 §6 stands.
- **Nothing about the architecture class.** "The world branch costs 108 pp" bounds *our old
  implementation on our corpus* and is not evidence about world-action models generally. This run
  does not change that and must not be cited as though it did.
- **Nothing about the label space.** This trains on the corpus exactly as it sits on disk —
  `relabel_chunks`, state-to-state at every step. PR-12/PR-13's V-chain repair is an **evaluation
  adapter** fix (`scripts/eval_t39_baseline.py`) with zero hits in `src/wam/` and zero in the
  training path; it is not in this run's path in either arm, and this run is not evidence for or
  against relabelling.
- **Nothing about hardware transfer.** `docs/sim.md:536-539` stands: a recorded `(state, action)`
  pair is not a commanded/achieved pair, and magnitudes are not expected to transfer to a robot
  whose `q_track_window` is 0.
- **Whether 20 000 steps is enough.** Both arms share the budget, so the contrast is internally
  valid; neither arm is evidence about convergence.

## 6. What a `VOID` looks like here

If the split check fails, if `config_hash` does not match this config, or if the run cannot complete
20 000 steps within the wall clock and resumes do not reconstruct one continuous chain, the result
is `I` and **no shape claim is made at all**. A partially-trained B compared against a fully-trained
A is not a weaker version of this experiment; it is a different one, and it is not licensed here.

## 7. Cost

~6 h wall clock for B on the local 5090 (band 4.7–15.7 h; the 5.8 h point estimate carries an
explicitly-labelled 2.0× assumption over a measured 0.42 s/step on an H200). **Zero Discoverer+
GPU-hours; the ~4 875 h budget is untouched.** A′, if `shape_moved` fires, roughly doubles it. The
box is shared with peer sessions, which is a scheduling cost this document is recording rather than
hiding.
