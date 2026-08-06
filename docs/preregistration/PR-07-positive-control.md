# PR-07 — Does *anything* clear this bar on this corpus?

Pre-registered 2026-08-06, **before any GR00T weights are downloaded, any environment is built on
Discoverer+, and any job is submitted**. Nothing in this document was written with a measured
number in view, because there is not one.

Task: **T-39**. Venue: Discoverer+ (`ehpc-aif-2026pg01-905`, H200).
Executable rule: `cluster/discoverer/71_eval_t39_control.sbatch`.
Trainer: `cluster/discoverer/70_train_t39_baseline.sbatch`.
Code that must exist before submission: §8 — it does not yet.

---

## 1. The defect: fourteen negatives and no positive control

Every recorded result in this project is a negative, and not one of them can be read.

| recorded | verdict |
|---|---|
| T-15 / T-24 / T-26 | frozen Wan **and** Cosmos3 features lose to a state-only ridge |
| T-18 | adding the video branch to `tiny` costs 108 pp (AC-07) |
| T-16 / T-29 | the Wan LoRA reaches WAM-Bench **L0**, `skill_vs_repeat_pct` **−21.80 %** |
| T-30 | every flow arm sits below L0 |
| T-36 / PR-06 | the dream is 39 % further from the truth than a frozen frame |
| T-38 | Wan-vs-Cosmos reverses with corpus size; both lose to the floor at all three sizes |
| PR-03 | the power gate fails; the refit was not submitted |

Each of those compares a WAM variant against a trivial baseline. **None of them compares anything
against a method that is known to work.** The distinction they cannot draw is the only one that
matters: *our approach is wrong* versus *nothing clears this bar on this corpus, under this
scorer*. Fourteen experiments have been spent on the first reading without the second ever being
tested.

**The corpus makes the omission worse than generic.** `docs/ROADMAP.md` records the provenance:
every real number in this repo comes from `nvidia/GR00T-N1.7-AppleToPlate` (CC-BY-4.0), converted
by `scripts/convert_lerobot_g1.py`. That dataset is NVIDIA's **own tutorial corpus for
post-training GR00T N1.7 on a G1 apple-to-plate task** — it was published to demonstrate that a
specific recipe works on exactly this data. We adopted the data and never ran the recipe:

```
$ grep -rn 'gr00t_n1\|isaac-gr00t\|pi0\|pi05\|smolvla\|openvla' src/ scripts/ configs/ cluster/
$                                             # no matches, 2026-08-06
```

So the standing explanation for all fourteen — *"402 success-only episodes of one task is not
enough"* — has never been separated from *"we have not run a method that works on 402 episodes."*

**Secondary evidence, recorded as motivation and deliberately NOT load-bearing.** Community
guidance for pretrained manipulation policies puts useful success rates at the order of ~50
demonstrations, and Unitree's own published G1 task datasets sit at 201–351 episodes. If that is
right, 402 is several times more than the standard method needs, and T-32's premise is inverted.
This comes from a 2026-08-05 literature sweep, is not independently reproduced here, and **no gate
below depends on it.** It is why this experiment is worth running first, not part of how it is
scored.

## 2. What T-32 becomes if this is not run first

`TASKS.md` T-32 retrains WAM at 40 / 120 / 362 episodes for ~109 GPU-h to fit a scaling curve. If
the method is simply broken on this corpus, that measures the scaling of brokenness, and every
branch of `I8_RULE_V3` — A, B, C, C-NOISY, D — is a statement about a curve whose premise was never
established. **T-39 costs an order of magnitude less and is a precondition for reading T-32 at
all.** T-32 is not submitted until T-39 reports. That ordering is part of this pre-registration.

## 3. What is held identical — and what cannot be

The experiment is only worth running if a failure is attributable. Three things are pinned by
construction, and the four that differ are stated here rather than discovered afterwards.

**Held identical, by sharing the artifact, not by copying a value:**

| | held by |
|---|---|
| training episodes | `configs/splits/i8_train_362.txt` — the committed file T-32's rung 362 uses |
| holdout episodes | `configs/splits/t18_holdout_episodes.txt` — 40 episodes, 1 040 chunks |
| the chunks scored | `wam.evaluation.build_eval_pairs`, the same call `eval_t16.py` makes |
| the scorer | `wam.evaluation.bench_metrics` / `e1_metrics`, both spec 0.1.0 **and** 0.2.0 |
| the split proof | `eval_t16.verify_split`, disjointness path, external witness required |
| the label space | `scripts/convert_lerobot_g1.py`, the functions that produced our own labels |

**Necessarily different, and therefore not confounds to be removed but the intervention itself:**
the architecture, the action parameterisation, the pretraining corpus, the observation window and
the training budget. T-39 does not ask "is GR00T better than WAM". It asks **"does the bar move for
anyone"**, and for that question the intervention is allowed to be the whole method.

**One thing deliberately NOT held identical, stated so nobody re-derives it as a flaw:** the
trainer is NVIDIA's own, vendored unmodified under `third_party/`. A positive control run through
our reimplementation of someone else's recipe is not a positive control — a failure would again be
ambiguous between the recipe and our copy of it. Our code appears in exactly two places: the
witness writer (§5) and the eval adapter (§4).

## 4. The adapter is the weakest part of this design, so it gets its own gates

GR00T predicts in the dataset's native LeRobot action space. WAM-Bench scores canonical 15-joint
deltas plus a grasp-synergy gripper scalar. Something must map between them, and that something is
new code written by us — the one place where a failure could be ours and look like NVIDIA's.

The mapping calls `convert_lerobot_g1.py`'s own functions (`canonical_q`, `gripper_state`, and the
same BC relabeling of executed states into bounded joint deltas). One implementation, so a
difference here cannot be a difference in the converter. That is necessary and not sufficient, so
**two oracle arms run first and can veto the whole experiment**:

| arm | what goes in | what it proves |
|---|---|---|
| `oracle_state` | the holdout's future **executed states**, through the adapter | the plumbing. This is the identity of our own label pipeline and must score ~perfectly |
| `oracle_action` | the holdout's native **`action` column**, through the adapter | the **ceiling for any policy trained on that column** |

`oracle_action` is the arm worth the most. Our labels are relabeled from executed *state*; GR00T is
trained to predict the *commanded* `action`. If those two differ enough that the ground-truth
action column itself cannot clear L1 under our scorer, then **no policy trained on this dataset can
clear our bar, and the finding is about our label pipeline, not about GR00T.** That outcome is a
VOID for T-39 and a first-class result for the project — it would retroactively bound every number
in `docs/benchmark.md`. It is pre-registered here precisely because it is the outcome I would
otherwise be tempted to treat as a bug and quietly patch.

Horizon and rate: our chunks are 16 steps at 30 Hz. If the checkpoint's native action horizon
differs, the adapter truncates or pads and **the artifact records which**; a padded arm is reported
as padded and never as a clean comparison.

## 5. Gates — `T39_RULE_V1`, fixed here and in `71_eval_t39_control.sbatch`

The headline is **`skill_vs_repeat_pct` on the 40-episode holdout, 1 040 chunks**, driven through
`evaluate_policy` — the same metric, chunks and code path as every number in `docs/benchmark.md`.

No new thresholds are invented. The bar is WAM-Bench's existing ladder, which already defines:

- **L1** `skill_vs_repeat_pct > 0` — better than repeating the last action, over all chunks
- **L2** `ci_skill_vs_repeat_pct > 0` — still better on the **task-critical** chunks
  (`ci_` is *critical*, not *confidence interval*)

One constant is borrowed rather than coined: `MATERIAL_FLOOR_PP = 10.0`, from `I8_RULE_V3` in
`62_eval_i8_curve.sbatch`, where it is already the repo's floor for "material on this headline". It
is borrowed so that the negative verdict is held to a standard rather than being the free default —
`62`'s own header states that a rule which only gates the expensive conclusion is a preference and
not a decision rule, and here the *negative* is the expensive conclusion, because it licenses
"stop trying methods on this corpus."

**G0 · VOID (runs first, can stop everything).**
`oracle_state` must reach `skill_vs_repeat_pct >= 90 %`. Below that the adapter is broken, no
verdict is issued, and the fix is a code fix — not a threshold change.
`oracle_action` must reach **L1**. Below that, see §4: T-39 is VOID and the finding is recorded
against the label pipeline.

**The verdicts**, on `groot-holdout`:

| | condition | reading |
|---|---|---|
| **P** | reaches **L1** | the bar is clearable on this corpus |
| **N** | `skill_vs_repeat_pct <= −10.0` **and** `groot-train40` also fails L1 | no method clears it, and the policy cannot even fit the data |
| **M** | fails L1 on the holdout, **clears** L1 on `groot-train40` | it fits and does not generalise across our split |
| **I** | anything else — in particular `−10.0 < skill_vs_repeat_pct <= 0` | indeterminate; run seed 1 before recording anything |

`groot-train40` is 40 episodes drawn from GR00T's **own training split**, scored identically. It
reuses the committed `configs/splits/i8_train_040.txt` — a seeded random subset of the 362, nested
inside them and disjoint from the holdout by construction (`tests/test_splits.py`) — rather than
minting a fresh sample at run time, because a sample drawn after the fact is not reviewable and
could be chosen to flatter either verdict. It is a diagnostic and a train-set upper bound. It is
**never** the headline and no verdict may quote it as one.

**Recorded regardless of verdict:** the level reached, the score under both bench specs, all five
rungs, `horizon_ratio`, `smoothness_ratio`, the checkpoint id **and revision**, the exact step
count, and the wall time.

## 6. Reading the outcome — decided before the numbers exist

| verdict | what it licenses | what it forbids |
|---|---|---|
| **P** | the fourteen negatives become statements about **WAM**, not about the corpus. Adopt the working policy as the NeoDEM capability; WAM's contribution narrows to evaluation, safety and integration around it. T-32 is **descoped** — the data-volume question is answered by a method that works at 362 | claiming WAM would clear it with more compute. It had 36 GPU-h and did not |
| **N** | the corpus/bar is the finding. T-32 is answered for free: more of *this* data is not the story. The next move is the **kind** of data — `PR-04`'s collection spec — not another method | claiming any specific WAM design is refuted. N says the instrument saturates, not which arm is wrong |
| **M** | a generalisation statement, and proof the bar is reachable in principle. The live question becomes the 362/40 split itself | reading it as either P or N |
| **I** | one seed replicate, nothing else | recording a verdict |
| **VOID** | a defect report against the adapter (`oracle_state`) or the label pipeline (`oracle_action`) | any statement about GR00T |

## 7. Cost, and the one conditional second attempt

Ceiling **12 GPU-h** of a 5 000-hour allocation: at most 3 × 4 h walltime (train, possibly one
requeue, eval). NVIDIA quotes 2–4 h for this post-training on a single 40 GB GPU; H200 should beat
that, and `MAX_RESTARTS` in `70_train_t39_baseline.sbatch` enforces the ceiling regardless.

**Exactly one alternative candidate is pre-registered, and only on outcome N:** `lerobot/pi05_base`
on the identical split and scorer. It is fixed here so that "try another model until one passes" is
not available after the fact. If it runs, it is reported as **attempt 2 of 2**, and a P reached on
attempt 2 is recorded as "P on the second of two pre-registered candidates" — never as P. There is
no attempt 3 under this pre-registration.

**Frozen until T-39 reports:** T-32 (§2), any Cosmos3-Super generation, any Cosmos3-Edge work.

## 8. What must exist before anything is submitted

This document is the gate; these are the implementation. Items 1–3 landed **2026-08-06**, after
the gate and before any cluster contact — the ordering §1 argues for. Items 4–6 are still open and
**T-39 remains unsubmittable.**

1. ✅ `scripts/train_t39_baseline.py` — thin driver over the vendored NVIDIA trainer whose **only**
   jobs are to restrict the episode set to `i8_train_362.txt` and to write the witness
   (`train_episode_ids` + `dataset_snapshot_ref`, the shape `eval_t16.verify_split` consumes).
   The restriction is materialised as a symlinked subset **view** of the LeRobot source with
   filtered `meta/episodes.jsonl` and recomputed `info.json` totals, rather than requested through
   a trainer flag that may not exist: a trainer that trusts `info.json` over the directory listing
   would otherwise iterate 402 episodes over a directory holding 362. The trainer is invoked as a
   **subprocess**, as shipped, in its own venv — which also keeps its torch/flash-attn pins out of
   any process that touches WAM numbers.
2. ✅ `scripts/eval_t39_baseline.py` — the `Policy`-protocol adapter (§4), the four arms, and
   `save_predictions_jsonl` in the archived format so every arm is re-scorable on CPU forever.
   Both bench specs (0.1.0 and 0.2.0) are written per arm. The policy arm and `oracle_action`
   cross into canonical units through **one** function, `commanded_to_chunk`; that shared call is
   what makes the oracle a ceiling for the policy rather than a nearby separate measurement, and
   it is pinned by a test.
3. ✅ `tests/test_t39_baseline.py`, 31 tests. **Twelve mutants introduced and killed**, including
   the three plausible mis-anchorings (`action[t+1] - q[t]`, `action[t] - q[t+1]`, and the
   commanded column's own first difference) — T-37's transposed-`xmat` lesson, which is that a
   wrong convention returns finite, plausible, correctly-shaped numbers that no assertion about
   shape, range or finiteness ever catches. Two mutants that survived the first version of the
   suite are recorded here rather than quietly fixed: a gripper channel read one step late, and
   the wrong hand selected — both survived because the `oracle_action` test compared only
   `targets`. The joint delta spans `t → t+1` while the gripper is sampled *at* `t+1`, so the two
   channels are anchored differently by construction and one convention cannot cover both.
4. A separate cluster venv for the vendored trainer (its torch/flash-attn pins are not ours).
   `70_*.sbatch` exits FATAL if it is absent rather than importing into the WAM env.
5. `MODEL_ID` — **required**, no default. The exact checkpoint id and revision have not been
   verified from a primary source and must be recorded in the artifact, not assumed here.
6. `TRAINER_ENTRYPOINT` and `POLICY_ENTRYPOINT` — **required, no defaults**, added 2026-08-06 for
   exactly the reason item 5 exists. Writing the drivers made it explicit that we do not know the
   vendored trainer's entrypoint path or its inference API from a primary source, and a plausible
   guess would either fail late or run something adjacent and record it as NVIDIA's recipe. The
   inference contract the eval requires is small and stated in `load_commanded_policy`; a shim
   inside the t39 venv may adapt to it, but may **not** convert into canonical units — that
   happens once, in our code, shared with the oracle.

**One correction to `70_train_t39_baseline.sbatch`, made while implementing item 1 and recorded
rather than silently patched:** it now also passes `--wam-dataset`. The trainer eats the LeRobot
*source*, but `dataset_snapshot_ref` has to be taken over the *converted* episodes, because that is
what `eval_t16.verify_split` recomputes it against. The first version passed only the raw root, so
the witness it described could not have been verified at all. No threshold moved — the rule is
`71_eval_t39_control.sbatch`'s, it is untouched, and nothing has been submitted.

## 9. What this cannot answer

- **Not a robot result.** Offline chunk MSE on 40 held-out episodes. A P verdict is not a success
  rate and must never be quoted as one.
- **Not a WAM-vs-GR00T comparison.** §3: five things differ at once, on purpose. Anyone reading a
  ranking out of this is reading a confound.
- **Not evidence about world models.** GR00T N1.7 is an action policy. A P verdict says the corpus
  and the scorer admit a passing score; it says nothing about whether predicting video helps
  (AC-07), which is a different question with its own answer already recorded.
- **One task, 402 success-only episodes, one holdout, one seed** unless verdict I forces a second.
  The gripper channel still has peak-to-peak 0.120 and never opens, so nothing here can see a
  grasp — `docs/benchmark.md`'s standing rule applies to every number T-39 produces.
