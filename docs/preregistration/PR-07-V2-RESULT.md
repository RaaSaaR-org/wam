# PR-07 V2 — RESULT: T-39 reports `VERDICT N` under the repaired anchoring

**Run 2026-08-17, job `188408`, after `PR-07-V2-repaired-anchoring.md` (`T39_RULE_V2`) was
registered and committed.** This file records an outcome. It amends no rule, and it is written
alongside `PR-07-RESULT.md` rather than editing it — `docs/handoff.md` §3.

**`PR-07-RESULT.md` is not edited and remains the record of the `T39_RULE_V1` run.** Its
`−359.41` stands as what the unrepaired instrument produced; see §5.

---

## 1. The verdict

```
VERDICT N · NOTHING CLEARS THIS BAR, AND THE POLICY CANNOT EVEN FIT THE DATA.
```

| arm | chunks | `skill_vs_repeat` (L1) | `ci_skill` (L2) | mse | horizon | smooth | level |
|---|---|---|---|---|---|---|---|
| `oracle_state` | 1040 | **+100.00 %** | +100.00 % | 0 | 1.000 | 1.000 | L4 |
| `oracle_action` | 1000 | **+68.10 %** | +75.40 % | 3.022e-06 | 0.972 | 0.280 | L4 |
| **`policy` (holdout)** | 1000 | **−239.69 %** | −84.36 % | 3.218e-05 | 0.197 | 21.408 | **none — below L0** |
| `train40` (diagnostic) | 1045 | **−186.73 %** | −70.06 % | 3.201e-05 | 0.252 | 23.545 | none — below L0 |

Against `T39_RULE_V2` §1's thresholds, every one of which is quoted unchanged from V1:

- `oracle_state` **+100.00** ≥ the **90.00** floor → G0a passes.
- `oracle_action` **+68.10** clears **L1** → **G0b passes, so the VOID condition is not met.**
- `policy` fails L1 by far more than `MATERIAL_FLOOR_PP` = 10.0 pp, **and** `train40` also fails L1.
  V1 §5's definition of **N** is exactly that conjunction.

**This is a verdict, not a VOID.** `T40_RULE_V3` §5.3 registers that `P`, `N`, `M` and `I` satisfy
PR-08 §1's "T-39 has reported" while VOID does not. **§8 item 7 is therefore CLOSED** — see §6 for
the items that are not.

## 2. What N licenses, and what it forbids

Quoted from PR-07 §6, decided before any number existed:

> **licenses** — the corpus/bar is the finding. T-32 is answered for free: more of *this* data is
> not the story. The next move is the **kind** of data — `PR-04`'s collection spec — not another
> method.
>
> **forbids** — claiming any specific WAM design is refuted. N says the instrument saturates, not
> which arm is wrong.

The `train40` number is what carries this. A policy that scores **−186.73 %** on forty episodes it
**trained on** is not losing to generalisation; it never fit. Whatever is wrong is upstream of the
train/holdout split, which is why N points at the corpus and the scorer rather than at a method.

**PR-07 §9's standing bounds are unchanged by this verdict:** offline chunk MSE, one task, 402
success-only episodes, one holdout, one seed.

**No grasp claim may be read into any number above.** Both oracles and both policy arms emitted the
same warning: the gripper channel has peak-to-peak **0.120** (< 0.25) with **0.00** debounced
open/close transitions per episode, so `gripper_accuracy` was **WITHHELD** rather than computed
against an ~85 % majority class. `docs/benchmark.md`'s standing rule applies to every cell here.

## 3. The one thing V2 §7 left open is now measured

§7 registered the walltime as unverified and recorded that **no ms/chunk figure existed anywhere in
this repo**. It does now:

| | value |
|---|---|
| policy, holdout | **259.13 s / 1000 chunks = 0.2591 s/chunk** |
| policy, train40 | 245.9 s / 1045 chunks = 0.2353 s/chunk |
| oracle arms | ~18.7 s each |
| **job elapsed** | **00:14:43** against a 04:00:00 wall |

§7 worried about ≥ 3.46 s/chunk. The measured rate is **~13× faster than that threshold**, and the
whole job cost about a quarter-hour. **PR-07 §7's 12 GPU-h ceiling is not close to touched** — this
run and 188407 together spent under 16 minutes of GPU time.

Recorded as a measurement for whoever schedules the next eval; it moves no threshold.

## 4. Provenance (AC-04)

| | |
|---|---|
| job | `188408`, partition `common`, 1 × H200, `--qos=ehpc-aif-2026pg01-905` |
| repo commit | `d10db60d5fcae91b91b71427779bac3649326ee1-dirty` |
| the `-dirty` | a peer session's in-flight Isaac work (6 files under `docs/isaac.md`, `scripts/preflight_isaac.py`, `src/wam/robot/`, `tests/`). **None is read by this eval**; recorded so the stamp is not mistaken for uncommitted changes to the scored path |
| `anchor_kind` | **`command`** — recorded in every arm's `timing.json`, so no reader has to infer which set a number belongs to |
| `policy_entrypoint` | `t39_policy_shim:build_policy` |
| `model_dir` | `runs/t39-baseline-seed0/checkpoints/checkpoint-10000` (job `187804`, completed 2026-08-16, 01:22:14) |
| embodiment | `new_embodiment`, read out of the checkpoint's `experiment_cfg/conf.yaml`, not passed in |
| split | 362 train / 40 holdout, proven disjoint, hash matched on all four arms |

**Two prior states are preserved, because the eval overwrites arm directories in place:**

- `runs/t39-baseline-seed0/_archive-187813-oracles/` — the `T39_RULE_V1` oracles
  (`oracle_action −359.4077743907937`, `oracle_state +100.0`), the evidence `PR-07-RESULT.md` cites.
- `runs/t39-baseline-seed0/_archive-188407-v2-oracles-{state,action}/` — the V2 oracles from the
  first submission, kept so the reproduction in §5 could be checked rather than asserted.

## 5. The G0 expectation registered in V2 §5 was met

§5 registered, before the run, that `oracle_action` was **expected** to come back at **+68.10**, and
that a materially different figure would be *"a **defect report against this run**, not a new
finding about the labels."*

It came back at **+68.1014939946058**, in **two independent submissions** (188407 and 188408), on
1000 chunks each. PR-13's re-derivation, an independent driver, and the cluster eval agree.

This is the part that retires the `−359.41` premise by measurement rather than by argument. The
same corpus column that the defective instrument scored at −359.41 scores **+68.10 / +75.40** and
reaches **L4** once step 0 is anchored on the previous command instead of the state.

**It does not make the policy result better.** G0b passing is precisely what makes the policy
number interpretable: the ceiling is real, and the policy is 307 pp below it.

## 6. What this does NOT do

- **It does not open PR-08.** `T40_RULE_V1` §1 requires **every** §8 item closed **and** T-39
  reported. Item 7 is now closed; **items 2, 3 and 4 are open**, and item 5's measurement has not
  been taken. Item 6 (the partition in git) **closed on 2026-08-17** when
  `configs/transfer25/`, `scripts/check_style_partition.py` and
  `cluster/discoverer/97_transfer25_restyle.sbatch` were committed and pushed. Item 2 is a decision
  about what the deliverable is and, per V3 §5.3, **is not an agent's to make.**
- **It does not license training.** CLAUDE.md's gate reads: whether training may start, and against
  which label space, **is the project owner's call.** N is a verdict, not permission; a verdict that
  says the corpus is the finding is a particularly poor argument for training on it.
- **It does not refute any WAM design.** PR-07 §6's N row forbids that in as many words.
- **It does not, by itself, start attempt 2.** PR-07 §7 pre-registers exactly one alternative
  candidate — `lerobot/pi05_base`, identical split and scorer — **and only on outcome N**, which is
  now the outcome. That licence is real but it is a licence to spend allocation, so the submission
  remains the owner's call. If it runs it is reported as **attempt 2 of 2**, and a P reached there
  is recorded as *"P on the second of two pre-registered candidates"*, never as P. **There is no
  attempt 3 under this pre-registration.**
- **It changes no cell in `docs/benchmark.md`.** A `T39_RULE_V2` number is comparable only against
  other V2 numbers (V2 §3); none of the fourteen negatives was scored on this 1000-chunk set.
