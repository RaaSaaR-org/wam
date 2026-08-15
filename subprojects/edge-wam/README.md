# Edge WAM — image in, action out, on the robot

**Start here if this is your session.** One paragraph of what this is, then the state, then what to
do next.

## What this is

A policy small enough to run **on the robot**: camera image in, action out, closed loop, no
datacenter in the path. The model class is NVIDIA's **World Action Model** — a policy built on a
*video world model* backbone rather than a vision-language one. The concrete candidate is
**`Cosmos3-Edge`, 4B, OpenMDW-1.1**, whose `Cosmos3-Edge-Policy-DROID` variant is reported at
**32 actions per inference and 15 Hz on Jetson Thor** at 640×360.

This sub-project does **not** build a model. It post-trains an existing one.

## The one thing to understand before designing anything here

"A WAM without the video" is a contradiction in NVIDIA's terms — the video world model **is** what
makes it a WAM rather than a VLA, and at inference the predicted video and the action come out of
the same forward pass.

So the design is:

> **image → action is an *interface* choice, not an architecture choice.** We simply do not decode
> the video head. The world modelling stays, on-device, at 15 Hz. Removing it would leave a VLA,
> which is the thing this project exists to be an alternative to.

Full primary-source pass, with what is verified and what is not:
[`research/2026-08-15-cosmos3-edge-and-dreamzero.md`](research/2026-08-15-cosmos3-edge-and-dreamzero.md).

## State — 2026-08-15

Nothing has run. The sub-project is five open questions and a gate.

| | |
|---|---|
| model chosen | `Cosmos3-Edge` 4B — **not yet staged, not yet downloaded** |
| target hardware | Jetson AGX Orin / Thor **[?] — we may not own one; E-03 establishes this** |
| training hardware | Discoverer+ (`ehpc-aif-2026pg01-905`), 4 875 of 5 000 GPU-h left |
| blocked by | **T-39** for any training run |
| corpus for a G1 embodiment | **3 152 episodes at `action float32[28]`** exist and are reachable (see below) |

**Changed 2026-08-15, and it matters more than anything else here:** T-042's count found **3 152
real G1 episodes declaring `action float32[28]`** — the exact 28-dim G1 + Dex3 vocabulary Cosmos
does not ship — Apache-2.0, in repos we already hold the video for. Adding a G1 embodiment by
NVIDIA's own route (post-training on action-labelled data) now has data behind it. Conversion is
tracked as **T-043**. **The block order is arm-first — `[0:14]` arm, `[14:28]` hand**, measured
2026-08-15 across all 13 sets; the hand-first figure this line used to carry belongs to a different
corpus (`Humanoid-Everyday-G1`) and pointed the trap the wrong way. See **T-043 §1**. Left/right and
intra-hand order are still unverified, with three conflicting orderings on record — that is where
the plausible-wrong-numbers risk actually lives now.

## Tasks

See [`TASKS.md`](TASKS.md). The two that gate the rest are **E-01** (can the policy run without
language?) and **E-02** (what does a 28-dim G1/Dex3 embodiment actually require?). Both are cheap,
neither needs a GPU, and the answers decide whether "no VLA" is a config flag or a re-training job.

## Rules that bind here

- **Everything in the root `CLAUDE.md` still applies** — canonical schema, the deterministic safety
  layer, the swappable-backbone interface (FR-09), traceability of every rollout to
  checkpoint + dataset snapshot + config hash (AC-04).
- **The learned model never outputs motor currents.** A 15 Hz on-device policy makes this *more*
  important, not less: the safety layer sits between this model and the controller, always.
- **Pre-register before training** (E-05). The project's rules are versioned, never edited in
  place; a gate rewritten after seeing its output is not a gate.
- **Nothing gets submitted, downloaded at scale, or paid for without asking first.**
- Our own measurement that the world branch costs 108 pp bounds *our old implementation on our
  corpus*. It is not evidence about the architecture class, and must not be cited as such here.
