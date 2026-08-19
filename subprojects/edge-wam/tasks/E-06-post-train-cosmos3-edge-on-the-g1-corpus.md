---
id: E-06
subproject: edge-wam
title: "Post-train Cosmos3-Edge on the G1 corpus"
slug: post-train-cosmos3-edge-on-the-g1-corpus
status: backlog
priority: 3
owner: ''
tags:
- edge
- cosmos3
- training
- cluster
depends_on:
- E-01
- E-02
- E-04
- E-05
- T-39
created: 2026-08-15
updated: 2026-08-15
status_note: "Backlog, and correctly so: four E-tasks and T-39 stand in front of it. Listed now only so the shape of the sub-project is visible; do not promote it."
---

# Post-train Cosmos3-Edge on the G1 corpus

## Description

The actual experiment: take the staged 4B `Cosmos3-Edge` checkpoint, post-train it on the G1
apple-to-plate corpus under `PR-15` (renumbered from `PR-10` on 2026-08-19 — see E-05), and evaluate
it under `E05_RULE_V1`.

**This task is deliberately last, and one of its dependencies is not in this sub-project.**

`T-39` is the positive control this project never had. Its `oracle_action` arm asked whether the
corpus's own action column clears L1 under our scorer. **It reported on 2026-08-16, and it does
not: `VOID (labels)`, −359.41 pp on L1, 4.59× worse than repeating the last action**
(`docs/preregistration/PR-07-RESULT.md`). The companion arm `oracle_state` scored a bit-exact
`mse 0.0` and +100 % on every rung, so this is the label space and not our adapter.

**That answers this paragraph's conditional in the worst direction, and it lands squarely on E-06.**
No policy trained on these labels can clear the bar — including this one — and the correct response
is to fix the label space, not to buy a better backbone. Running E-06 now would spend GPU-hours
reproducing a label-space defect and invite reading it as a model result; fourteen recorded
experiments already share that ambiguity, and this one still does not have to.

The follow-up PR-07-RESULT names is a **delay sweep over the anchoring convention** — our labels are
relabeled from executed state over `t → t+1` while the corpus's target is the commanded value at
`t`, with `horizon_ratio 0.0044` putting essentially all error in the chunk's first step and
`smoothness_ratio 8.52` making the command 8.5× jerkier than the demonstration. That is the
experiment this task now waits behind, not another backbone.

**Superseded 2026-08-16 — that sweep ran, and the anchoring convention turned out to be a defect
rather than a convention.** `commanded_to_chunk` built the chunk's step 0 as `command − STATE`
while every other step is `command − command`; a standing tracking error cancels in every
homogeneous difference and survived at full magnitude in that one term. **`smoothness_ratio 8.52`
never meant "the command is 8.5× jerkier"** — 96.8 % of the predicted-jerk sum sat in the index-0
term alone, and dropping index 0 from *both* arms gives **0.28**, i.e. over steps 1–15 the command
is ~3.6× *smoother* than the demonstration (`docs/smoothness-ratio-audit.md`). Repaired, the
corpus's own action column scores **+68.10 L1 / +75.40 L2, level L4** on T-39's own holdout with
`horizon_ratio` ~0.97 (`docs/preregistration/PR-12-RESULT.md`, `PR-13-RESULT.md`). **So this task no
longer waits behind the anchoring question** — it is answered — but behind the training decision
that follows from it, which is the project owner's call.

**Superseded again 2026-08-17 — T-39's policy arm ran, and the verdict is `N`.** Under
`T39_RULE_V2` (job `188408`, `docs/preregistration/PR-07-V2-RESULT.md`) `oracle_action` passed G0 at
**+68.10 L1**, so the `VOID` premise above is gone. The policy arm then ran for the first time and
**GR00T N1.7 — NVIDIA's own recipe, on NVIDIA's own tutorial corpus — scored −239.69 % on the
holdout and −186.73 % on the forty episodes it *trained on*.** A method that cannot beat
repeat-last-action on its own training data is not losing to generalisation.

**Read what N does to this task, because it cuts both ways.** PR-07 §6's N row **forbids** reading
the result as refuting any specific WAM or Edge design — N says the instrument saturates, not which
arm is wrong — so nothing here is evidence against Cosmos3-Edge. But the same row says the next move
is **the *kind* of data (PR-04's collection spec), not another method**, and **E-06 is another method
on the same corpus.** That is the sentence this task now has to be scheduled against: post-training a
4B Edge checkpoint on a corpus where a known-working 3B policy could not fit its own training split
would spend GPU-hours on the one ambiguity this sub-project exists to avoid. It does not make E-06
wrong; it makes E-06 **the second-best use of the next GPU-hour**, and the decision is the project
owner's.

**One hypothesis about that −186.73 % was checked and eliminated, 2026-08-19.** A policy failing on
its own training data admits an unglamorous explanation — that the finetune damaged the vision
encoder — which would make the number a statement about the recipe rather than about the corpus.
**It did not: T-39 froze the vision tower.** `runs/t39-baseline-seed0/checkpoints/config.json`
records `tune_visual: false`, `tune_llm: false`, `tune_projector: true`, `tune_diffusion_model:
true` (identical in `experiment_cfg/final_model_config.json`), which is NVIDIA's shipped default
verbatim — `third_party/isaac-gr00t/gr00t/configs/finetune_config.py:49,52,55,58`. So T-39 ran the
**default** recipe with the encoder untouched, and encoder damage is ruled out by construction.

**What is still open, and is recorded here rather than argued into the verdict:** T-39's scorer is
chunk MSE against relabelled first-difference targets, while the GR00T objective is flow-matching in
its own normalised action space. Those are not the same quantity, and **nothing in PR-07 tests that
they are** — so "the corpus and the scorer saturate" and "the eval measures something the training
objective never optimised" are both still live readings of the same number. `N` is a pre-registered
verdict and is not amended by this note; it already forbids reading itself as refuting any specific
design. Distinguishing the two is a separate, cheap, CPU-side question and would need its own
pre-registration.

## Acceptance

1. All of E-01, E-02, E-04, E-05 closed, and **T-39 reported** — it has, **`VERDICT N`** (2026-08-17,
   re-reported under `T39_RULE_V2`; the earlier `VOID (labels)` premise was withdrawn by
   measurement). So this clause is satisfied in letter, and the substance has changed rather than
   improved: **N is not a licence to start.** Before E-06 is scheduled, the owner has to decide
   against PR-07 §6's N row whether the next GPU-hour goes to another method on this corpus or to a
   different *kind* of data.
2. Run traceable to checkpoint + dataset snapshot + config hash (AC-04).
3. Verdict read strictly under `E05_RULE_V1`, including VOID — **a VOID is not a weak pass.**
4. Result written into §Notes here *and* into a `PR-15-RESULT.md`, whichever way it goes.
5. Negative or VOID results are recorded with the same care as a positive one. This project's value
   is in its receipts.

## Notes / Report

*(empty — fill in when the task runs)*
