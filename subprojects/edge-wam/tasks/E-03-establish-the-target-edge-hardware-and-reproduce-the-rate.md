---
id: E-03
subproject: edge-wam
title: "Establish what edge hardware we actually have, and whether the published rate reproduces"
slug: establish-the-target-edge-hardware-and-reproduce-the-rate
status: todo
priority: 2
owner: ''
tags:
- edge
- hardware
- benchmark
depends_on: []
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. Step 0 is a question for the user, not a measurement: we do not know that a Jetson exists in this project."
---

# Establish the target edge hardware, and whether the published rate reproduces

## Description

The whole sub-project is justified by a number we have not measured, on hardware we may not own.

**The published figures disagree with each other.** NVIDIA's blog and the launch coverage say
**32 actions per inference at 15 Hz on Jetson Thor** at 640×360 [doc]. The `Cosmos3-Edge` model card
instead shows **action chunk size 16** and **20 fps** in its example configs [✓]. Both cannot
describe the same setup, and neither has been reproduced here.

**Step 0 is not a benchmark.** It is: *do we have a Jetson?* Nothing in this repo mentions one —
`cluster/discoverer/` is H200s and the workstation is a single RTX 5090 with 93 GB host RAM. If the
answer is no, the deployment target is a decision, not a fact, and this task's real output is that
decision plus its cost.

Note the 5090 is a *poor* proxy: 32 GB and desktop power envelope tell you little about a Jetson's
memory bandwidth or thermal budget. A 5090 measurement bounds the model's compute, not its
deployability.

## Acceptance

1. **The hardware question answered by the user**, recorded here: which Jetson (Thor / AGX Orin /
   none), or which alternative target.
2. If a device exists: 4B policy loaded, and actions-per-inference, control rate, resolution and
   power measured, against the two published claims, with the discrepancy resolved or recorded.
3. If no device exists: the options costed — buy a Thor, target the 5090, or run the policy off-board
   over the network (and what that does to the closed loop's 0.5–2.0 s chunk budget and to the
   safety layer's watchdog).
4. Whichever branch: a stated **latency budget** for the closed loop, since FR-05's re-observe /
   re-plan cycle is what this model has to fit inside.

## Notes / Report

*(empty — fill in when the task runs)*
