---
id: E-02
subproject: edge-wam
title: "What adding a 28-dim G1/Dex3 embodiment actually requires"
slug: what-a-28-dim-g1-dex3-embodiment-actually-requires
status: done
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
updated: 2026-08-16
status_note: "Answered 2026-08-16: one new row in a 32-row trained table; 28<=64 so no architecture change. Measured from the weights — the released POLICY checkpoint has only ONE trained row (droid), AgiBot included in the untrained ones; warm start must come from the base."
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

**2026-08-16 — verdict: `new_action_head_plus_post_training`, but read the shape of it: the
architecture already fits us with room to spare; the weights do not exist and must be earned.**
A 28-dim G1/Dex3 embodiment is **one new row in an existing 32-row weight table**, trained by
post-training. Independently re-verified by an agent told to refute it: **not refuted, high
confidence**; the weights measurement reproduced by two independent routes.

**1. The width question is settled and favourable — this kills `architecture_change`.**
The action head is *not* per-embodiment-width. There is ONE global `action_dim = 64` for the
whole model, identical in base and policy `transformer/config.json`; the training config names
the same quantity `max_action_dim: 64`. Narrower domains are zero-padded on the channel axis
and predictions sliced back down (`pipeline_cosmos3_omni.py:861-869`, `:1769-1770`), with a
hard error above 64 (`:772-776`). **28 ≤ 64, so no tensor is resized and no module is added.**
The 57-D `hand_pose` embodiment already rides the same head.

**2. The mechanism is a learned table row — this kills `config_entry`.**
Embodiment = *string* → *domain id* → *row index* into two `nn.Embedding` tables inside
`DomainAwareLinear` (`transformer_cosmos3.py:154-177`, instantiated `:381-382`). The name→id
map is ordinary Python (`cosmos_framework/.../domain_utils.py`), free to extend, with 14
unassigned ids. The id→weights are **trained parameters**. Claiming a free row is a dict edit;
making it emit G1 actions is not.

**3. The measurement — and it kills the AgiBot warm-start hope where we assumed it lived.**
Per-output-channel norms of each domain row against the untrained-row noise floor recover every
embodiment's trained width. Reproduce in ~30 s, no checkpoint download (header + one 8 MB
tensor over HTTP Range):

```
.venv/bin/python scripts/probe_cosmos3_domain_rows.py nvidia/Cosmos3-Edge
.venv/bin/python scripts/probe_cosmos3_domain_rows.py nvidia/Cosmos3-Edge-Policy-DROID
```

Run on both 2026-08-16. The recovered widths reproduce NVIDIA's **published table exactly** —
av 9, camera_pose 9, hand_pose 57, umi/bridge/droid/robomind-ur/fractal 10, franka-dual 20,
agibotworld 29 — which is what makes the next line credible rather than merely asserted:

- **base `Cosmos3-Edge`: 10 of 32 rows trained**, `agibotworld` at width 29 among them.
- **released `Cosmos3-Edge-Policy-DROID`: exactly ONE row trained — d8 (droid), width 8.**
  Every other row, **`agibotworld` included, is back at random init.**

So the "start from the supported 29-D humanoid" route is **not available from the policy
checkpoint**. It exists only in the base. Any G1 post-training that wants a humanoid warm start
must start from `nvidia/Cosmos3-Edge`, not from the policy variant — and that is a decision
E-05 has to pre-register, because the two have different action heads.

**4. Two facts that make G1 a better fit than expected.** The DROID policy's action space is
**absolute joint position** (`action_space=joint_pos`, 8-D incl. gripper, un-normalized, with
`use_state` proprioception) — joint-space policies are a first-class supported mode, which is
exactly what our canonical space is. Conversely **AgiBot's 29-D is end-effector pose**
(9-D ego + 2×(9-D EE pose + 1-D gripper)) obtained by forward kinematics, *not* a joint vector
— so its 29-D is not the near-neighbour of our 28-D that the README implied. Its arm state
slice is nonetheless `slice(0, 14)`, 7 joints per arm, structurally identical to G1_Dex3's arm
block.

**Correction to the cost estimate carried into E-05/E-06:** the 256-GPU reference run cited in
NVIDIA's post-training recipe is for **Cosmos3-Nano (16B)**, not the 4B Edge target. Do not
size our allocation against it.

**Consequence, implemented:** `src/wam/robot/g1_dex3_28.py` is the 28-dim ↔ canonical mapping,
and `Cosmos3EdgeConfig` deliberately registers **no** G1 embodiment — per the above, a row that
is not trained is not an embodiment, and naming one would be the fabrication this project is
organised against.

**Open [?]:** diffusers pins `_EMBODIMENT_TO_RAW_ACTION_DIM['droid_lerobot'] = 10` while the
released checkpoint trained only 8 channels — 10 ≠ 8. `Cosmos3EdgeConfig.raw_action_dim` has no
default and raises until a forward pass or cosmos-framework source settles it.
