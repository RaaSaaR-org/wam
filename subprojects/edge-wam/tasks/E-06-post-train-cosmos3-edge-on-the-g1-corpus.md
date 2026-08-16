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
apple-to-plate corpus under `PR-10`, and evaluate it under `E05_RULE_V1`.

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

## Acceptance

1. All of E-01, E-02, E-04, E-05 closed, and **T-39 reported** — it has, `VOID (labels)`, so this
   clause is satisfied in letter and failed in substance. **A VOID upstream is not a licence to
   start:** treat this as requiring an owner decision on the label space before E-06 is scheduled.
2. Run traceable to checkpoint + dataset snapshot + config hash (AC-04).
3. Verdict read strictly under `E05_RULE_V1`, including VOID — **a VOID is not a weak pass.**
4. Result written into §Notes here *and* into a `PR-10-RESULT.md`, whichever way it goes.
5. Negative or VOID results are recorded with the same care as a positive one. This project's value
   is in its receipts.

## Notes / Report

*(empty — fill in when the task runs)*
