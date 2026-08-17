# PR-14 RESULT — the jerk regulariser is not the cause of the bland chunk

**Rule:** `T48_RULE_V1`, `docs/preregistration/PR-14-smoothness-ablation.md`, registered before the
run and unedited since. Task T-48.
**Verdict: `S` (shrinkage).** Not `L`, not `L-MATERIAL`.
**Run:** `pr14-nosmooth-seed0`, `config_hash=69aba5309f911a450ebae8aa3395e9eb8bbdf255413a1478361725ce0e62e93b`,
git `6162d95`, local RTX 5090. Zero Discoverer+ GPU-hours, as costed in §7.

---

## 1. The run is valid under §6

§6 voids the experiment if the split check fails, if `config_hash` drifts, or if the run cannot
complete 20 000 steps and the resumes do not reconstruct one continuous chain. None of those
happened, and the chain is worth spelling out because it is the one that nearly did:

| event | log line | step | sampler |
|---|---|---|---|
| fresh start | `train.out:1` 2026-08-17T00:53:04 | 0 | epoch 0 batch 0 |
| checkpoint | `train.out` 2026-08-17T02:22:39 | 7385 | epoch 6 batch 1112 |
| **SIGTERM** | `train.out:42` — `stopped at step 7385/20000 (SIGTERM (stop requested))` | 7385 | — |
| resume | `train.out:66` 2026-08-17T23:07:46 | **7385** | **epoch 6 batch 1112** |
| complete | `train.out:1334` 2026-08-18T01:31:26 — `COMPLETE 20000/20000` | 20000 | epoch 16 batch 4192 |

The resume re-entered at the step *and the sampler position* the checkpoint recorded, so the two
legs are one continuous pass over the data rather than a replay of the first epochs. That is what
§6 asks for. The interruption was an external `SIGTERM`, not an OOM and not a divergence.

`config_hash` is identical in both legs and in `runs/pr14-nosmooth-seed0/DONE`. The evaluator
proved the split rather than assuming it: `split proven (disjoint): 362 train / 40 holdout, hash
matches`, against `dataset_snapshot_ref` `sha256:6b8fe849…`.

Scored by `scripts/eval_t16.py` on the 40-episode `t18` holdout in `--frame-history` mode, 1 040
chunks, 50.6 s (48.7 ms/chunk). Artifacts in `runs/pr14-nosmooth-seed0/eval-t29-history/`.

## 2. The rule, evaluated

`B.chunk_rms` is computed as the registered docstring defines it (`src/wam/training/joint.py:409`):
RMS over all 1 040 × 16 × 15 predicted chunk elements in `predictions.jsonl`. The same computation
over the `target` column returns **0.004041**, reproducing the registered `DEMO_RMS = 0.00404` — so
the instrument reading B is the instrument that produced the constants it is compared against.

| term | threshold | B measured | |
|---|---|---|---|
| `B.smoothness_ratio ≥ 2 × A_SMOOTHNESS_RATIO` | ≥ 0.6396 | **0.35478** | **FALSE** |
| `B.chunk_rms ≥ 1.25 × A_CHUNK_RMS` | ≥ 0.002825 | **0.002931** | TRUE |
| **`shape_moved`** | both | | **FALSE** |
| `l1_cleared` = `skill_vs_repeat_pct > 0` | > 0 | **−3.4288** | **FALSE** |
| `l1_material` = `B − A_SKILL_VS_REPEAT ≥ 10.0` | ≥ 10.0 pp | **+18.371 pp** | TRUE |

`shape_moved` is FALSE ⇒ **verdict `S`**. `l1_cleared` is FALSE ⇒ **not `L`**, and `L-MATERIAL`
requires `l1_cleared and l1_material`, so it does not fire either — `l1_material` alone licenses
nothing under the registered table.

## 3. What `S` licenses, and what it does not

Per §4: the jerk regulariser is **not** a material cause of the bland chunk. L2 shrinkage /
mean-seeking survives as the explanation. This licenses **dropping the regulariser hypothesis**. It
does **not** license adopting the mean-seeking one — that would need its own test.

§3's asymmetry is what makes one run enough here. B ran on a 5090 at `batch 2 × accum 4` where A
ran on an H200 at `batch 8 × accum 1`, and that confound is live only for a *positive*: a hardware
difference cannot manufacture a null. `shape_moved` is FALSE, so **A′ is not required** and is not
being run.

The split inside `shape_moved` is the informative part. Removing the regulariser moved the
**magnitude** — `chunk_rms` 0.00226 → 0.00293, closing 40 % of the 44 % shortfall against the
demonstrations, and clearing its own threshold. It did **not** move the **shape**:
`smoothness_ratio` 0.3198 → 0.3548, an 11 % change against the 100 % the rule required. The chunks
got bigger without getting jerkier. A regulariser that were the material cause of the bland output
would have moved both, and the conjunction in `shape_moved` is what separates those two readings —
had the rule been written on `chunk_rms` alone it would have returned a positive for a model whose
smoothness is still 0.71 of spec 0.2.0's admissible floor.

## 4. The L1 movement, reported and not claimed

`skill_vs_repeat_pct` went from A's archived **−21.80** to B's **−3.43**, a **+18.37 pp** move that
is above the borrowed `MATERIAL_FLOOR_PP` of 10.0. **The bar is still failed by 3.43 pp**, so this
is the closest the WAM has come to beating causal repeat-last-action and it is still on the wrong
side of it. Under the registered table this is not a verdict; it is a number.

It must not be read as an effect of removing the regulariser. That is a positive direction, which
is exactly where §3 says the hardware/batch confound is live, and no A′ has been run to separate
them. Everything the rule licenses on the L1 axis is: `l1_cleared` is FALSE.

The full ladder for the record: **L0 passed** (`skill_vs_zero_pct` +42.12), L1 failed (−3.43), L2
failed (`ci_skill_vs_repeat_pct` −3.98), **L3 passed** (`horizon_ratio` 1.127 ≤ 4), **L4 passed**
(`smoothness_ratio` 0.355 ≤ 2 under spec 0.1.0). Score 56.0/100, level **L0**.

The evaluator withheld `gripper_accuracy`: dynamic range 0.120 < 0.25 against an 85.3 % majority
class and 0.00 debounced transitions per episode. **No grasping claim of any kind is available from
this run**, and the withholding is the correct behaviour, not a gap in it.

## 5. Two facts about the environment, recorded rather than hidden

- **L4 is passed under spec 0.1.0 and would fail spec 0.2.0.** `smoothness_ratio` 0.3548 is below
  0.2.0's two-sided floor `MIN_SMOOTHNESS_RATIO` (`src/wam/evaluation/benchmark.py:72`, derived as
  `1.0 / MAX_SMOOTHNESS_RATIO` = 0.5 and deliberately never a literal — the pre-registration cites
  this as `:71`, which is `MAX_SMOOTHNESS_RATIO`; the value it uses, 0.5, is right). The
  bench wrote `spec_version: 0.1.0`, so the reported pass is correct as issued. This is the same
  open disagreement `docs/benchmark.md` already carries after PR-13, and it is the project owner's
  decision, not this document's. Under 0.2.0 the honest statement is that B moves *toward* the
  admissible band (0.3198 → 0.3548, still 0.71 of the floor) without entering it.
- **`runs/t16-lora-seed0/` is empty on this box.** A's constants were pinned as literals in the
  pre-registration precisely so the comparison does not depend on a directory surviving, so the
  verdict is unaffected. But A cannot be re-scored here without re-materialising that run, and
  anyone who needs A′ later should know that before planning it.

## 6. What this still cannot answer

§5 of the pre-registration stands unchanged and is not restated in full. In particular: **nothing
about GR00T** (PR-07 §6 — no vendored model was loaded, trained or consulted in either arm),
nothing about the architecture class, nothing about the label space (both arms train on the corpus
exactly as it sits on disk; PR-12/PR-13's V-chain repair is an evaluation-adapter fix with zero
hits in the training path), and nothing about convergence at 20 000 steps.

**This is not a training gate being discharged.** The gate in `CLAUDE.md` is the project owner's
call and this document does not touch it.
