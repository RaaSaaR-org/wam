---
id: E-05
subproject: edge-wam
title: "Pre-register the edge policy experiment before any weights move"
slug: pre-register-the-edge-policy-experiment
status: todo
priority: 1
owner: ''
tags:
- edge
- prereg
- gate
depends_on:
- E-01
- E-02
blocks:
- E-06
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started, and deliberately depends on E-01 and E-02: a pre-registration written before we know whether the policy needs language, or what an embodiment costs, would be a guess dressed as a gate."
---

# Pre-register the edge policy experiment

## Description

The rule this project runs on: **rules are versioned, never edited in place; a gate rewritten after
seeing its output is not a gate.** PR-05's G2 is recorded VOID rather than patched. PR-09's G0b
returned VOID and the 60 clips stay unread. This sub-project gets the same treatment from the start,
before it has any results to be tempted by.

The pre-registration — `docs/preregistration/PR-10-edge-policy.md`, following PR-07's nine-section
shape — must fix, in git, before any weights move:

- **The claim.** Something falsifiable, e.g. *a post-trained Cosmos3-Edge policy clears WAM-Bench L1
  (`skill_vs_repeat_pct > 0`) on the G1 apple-to-plate corpus at a control rate that fits FR-05's
  0.5–2.0 s chunk budget.*
- **The arms**, including the oracle arms that can veto the experiment, in the spirit of T-39's
  `oracle_state` / `oracle_action`.
- **The rule**, as executable code committed before the run — `E05_RULE_V1`, with the margin
  *borrowed* (`MATERIAL_FLOOR_PP = 10.0` from `I8_RULE_V3`) rather than coined, so the choice of
  floor cannot become the finding.
- **The GPU-hour ceiling**, against 4 875 h remaining.
- **What a VOID looks like** and what it licenses, written before it can happen.
- **Whether the video head is decoded at eval.** If image→action is an interface choice, then a run
  that decodes video is a different experiment and needs its own arm.

## Acceptance

1. `docs/preregistration/PR-10-edge-policy.md` written and **committed before** E-06 starts.
2. `E05_RULE_V1` in git as executable code, with tests, before any run.
3. Every number in it traceable to a source: the corpus, the bench ladder, the borrowed floor.
4. Reviewed against PR-07 §8's list of what a pre-registration must not leave open (no undefined
   `MODEL_ID`, no unstated entrypoint).

## Notes / Report

*(empty — fill in when the task runs)*
