---
id: T-18
aliases:
- T-18
title: "Ablation harness — world-action vs. action-only (AC-07)"
slug: ablation-harness-world-action-vs-action-only-ac-07
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- backbone
- eval
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-26
updated: 2026-07-26
---

# Ablation harness — world-action vs. action-only (AC-07)

## Description

Ablation harness: world-action vs. action-only (AC-07) — *first real-data verdict (2026-07-26,
`scripts/run_ablation.py`, 402 GR00T episodes, identical 362/40 split + config as the action-only
baseline): **hurts** at tiny scale — holdout MSE 2.09e-05 vs. 1.10e-05 (−89.5%), gripper acc 0.853
vs. 0.871. The tiny backbone's video loss plateaus (~0.72), so the shared trunk pays a multi-task
tax. Consistent with the T-15 probe: the AC-07 advantage must come from the pretrained prior (Wan
LoRA). Run: `runs/t18-real-ablation-seed0`*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
