---
id: T-27
aliases:
- T-27
title: "WAM-Bench — a laddered offline benchmark on real held-out episodes"
slug: wam-bench-a-laddered-offline-benchmark-on-real-held-out-epis
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
- sim
- prereg
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# WAM-Bench — a laddered offline benchmark on real held-out episodes

## Description

WAM-Bench: laddered offline benchmark on real held-out episodes (`docs/benchmark.md`,
`src/wam/evaluation/benchmark.py`, `scripts/run_bench.py`, 17 tests) — *E1 reported one number,
action MSE, and one number cannot say whether a policy learned anything. Five gated rungs, scored
from an archived `predictions.jsonl` alone (torch-free, no GPU, no robot), so **every past run is
re-scorable when the metric set changes**: **L0** beats zero-delta · **L1** beats repeat-last-action
· **L2** still beats it on task-critical chunks (CI-MSE, arXiv:2606.29898 — restricting to critical
intervals lifts rank correlation with real rollout success from ρ≈−0.61 to ≈−0.87; our proxy for the
paper's VLM annotation is the top 20% of chunks by demonstrated motion energy) · **L3**
horizon_ratio ≤ 4 · **L4** smoothness_ratio ≤ 2. Level = highest **contiguous** rung passed; score
0–100 with pre-registered anchors, all thresholds module constants fixed before scoring. The repeat
baseline is strictly causal: it indexes the predecessor chunk at `stride−1`, never `[−1]`, which
would read the future under the overlapping chunks FR-05's receding horizon actually produces.
**First results (2026-07-29), identical 40-episode holdout:** action-only `d1-full-gen-seed0` **L0,
28.6/100**; world-action `t18-real-ablation-seed0` **below L0, 19.9/100**. Three things E1 could not
say: (a) **the action-only baseline loses to repeat-last-action by 17%** (9.14e-6 vs 1.10e-5) —
T-14's "−32% vs zero-delta" is the demonstration's own inertia, so **T-16's real bar is
`skill_vs_repeat_pct > 0`**, not "beats the action-only baseline"; (b) the world-action candidate is
worse than predicting no motion at all (−28.2%), with smoothness_ratio 5.10 showing the multi-task
tax as jerk; (c) **`gripper_accuracy` is not a grasp proxy on this dataset** — the demonstrated
gripper has peak-to-peak range 0.120 sitting on the 0.5 binarization threshold, so 0.87/0.85 was
thresholding noise, now emitted as a warning. Deliberately absent and documented as such:
action-following/controllability (MiraBench L2 — needs model access, the natural L5), optimism bias
(needs failure demos we do not have), video fidelity (needs stored frames), closed-loop divergence,
and real success rate (that is E3 and it needs the robot)*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
