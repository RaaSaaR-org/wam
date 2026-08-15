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

`T-39` is the positive control this project has never had. Its `oracle_action` arm asks whether the
corpus's own action column clears L1 under our scorer. **If it does not, no policy trained on this
corpus can** — including this one — and the correct response is to fix the label space, not to buy
a better backbone. Starting E-06 before T-39 reports would risk spending GPU-hours to reproduce a
label-space defect and reading it as a model result. Fourteen recorded experiments in this project
already share that ambiguity; this one does not have to.

## Acceptance

1. All of E-01, E-02, E-04, E-05 closed, and **T-39 reported**.
2. Run traceable to checkpoint + dataset snapshot + config hash (AC-04).
3. Verdict read strictly under `E05_RULE_V1`, including VOID — **a VOID is not a weak pass.**
4. Result written into §Notes here *and* into a `PR-10-RESULT.md`, whichever way it goes.
5. Negative or VOID results are recorded with the same care as a positive one. This project's value
   is in its receipts.

## Notes / Report

*(empty — fill in when the task runs)*
