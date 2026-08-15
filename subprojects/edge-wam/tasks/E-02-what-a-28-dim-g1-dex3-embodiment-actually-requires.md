---
id: E-02
subproject: edge-wam
title: "What adding a 28-dim G1/Dex3 embodiment actually requires"
slug: what-a-28-dim-g1-dex3-embodiment-actually-requires
status: todo
priority: 1
owner: ''
tags:
- edge
- cosmos3
- schema
- probe
- blocking
depends_on: []
blocks:
- E-05
- E-06
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. Reading task, no GPU. The answer sets the size of the whole sub-project: a config entry is days, a new action head plus post-training is months."
---

# What adding a 28-dim G1/Dex3 embodiment actually requires

## Description

`Cosmos3-Edge` ships ten embodiments with fixed action dimensionality [✓ model card]:

| embodiment | dims | | embodiment | dims |
|---|---|---|---|---|
| general camera motion | 9D | | **Agibot** | **29D** |
| autonomous vehicle | 9D | | UR | 10D |
| egocentric motion | 57D | | Google robot | 10D |
| single Franka + Robotiq | 10D | | WidowX 250 | 10D |
| dual Franka + Robotiq | 20D | | UMI | 10D |

**No Unitree G1. No 28-dim Dex3.** That bound is real and unchanged.

**But as of 2026-08-15 we have the data to add one.** T-042's step 0 (`docs/action-labels.md` §3b)
found **3 152 real G1 episodes across the 13 `unitreerobotics/G1_Dex3_*` sets, every one declaring
`action float32[28]`** — exactly this vocabulary, Apache-2.0, 647 MB of action parquets in repos we
have already pulled video from. NVIDIA's stated route for a new embodiment is post-training on
action-labelled data; that is no longer the blocker it was this morning. Conversion is tracked as
**T-043**.

**The block order is `[0:14]` arm, `[14:28]` hand — arm-first, measured 2026-08-15 across all 13
sets** (`meta/stats.json`: zero one-sided dims in `[0:14]`, 4–10 in `[14:28]` railing at a clean
100°/120° mechanical limit; independently confirmed by `vla-training/groot/modality_g1_dex3.json`).
This line previously said hand-first, carried over from `Humanoid-Everyday-G1` — a different corpus
in a different LeRobot version. Getting *that* backwards is the "finite, plausible and wrong" risk,
and it was pointed the wrong way here. **Left/right and intra-hand order remain unverified**, with
three mutually inconsistent orderings on record. Detail: **T-043 §1**.

**But `docs/action-labels.md` §3b overstates it.** It says "no humanoid/G1/28-dim Dex3". *AgiBot is
a humanoid, supported at 29D.* So the honest statement is "no G1, no Dex3" — and the difference is
the whole task: a supported **29D humanoid** is a far closer neighbour to a **28D G1** than "no
humanoid at all" suggests. Whether that neighbourhood is worth anything is exactly what this task
establishes. (The doc correction itself is being made separately; this task is about the
consequence.)

## Acceptance

1. From the Cosmos repo **code**: how an embodiment is defined — a config entry, a registered name,
   a learned per-embodiment embedding, or a separate action head. State which, with file:line.
2. Whether a new embodiment can be added without touching model weights, and if not, what minimum
   post-training it needs (data volume, steps, GPU-hours) per the published recipes.
3. Whether the 29D AgiBot entry is usable as an initialization for a 28D G1 — i.e. is the action
   space a fixed-width slot with semantic channels, or an opaque per-embodiment learned space.
   **This is the question that decides the sub-project's cost.**
4. A GPU-hour estimate for the resulting plan, checked against the 4 875 h remaining and against
   the per-GPU billing rules in `cluster/discoverer/README.md`.

## Notes / Report

*(empty — fill in when the task runs)*
