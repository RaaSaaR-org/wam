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
> the video head. The world modelling stays on-device. Removing it would leave a VLA, which is the
> thing this project exists to be an alternative to.

**Amended 2026-08-16, by E-01 and E-03 — two clauses of that paragraph did not survive contact
with the code and the model cards.**

- **"no VLA" holds as an interface, not as an architecture.** The caller supplies no language, but
  the text tower and tokenizer stay resident: `text_tokenizer` is a required pipeline component,
  `input_ids`/`und_len` are required positionals of `forward`, and a probe showed that changing
  *only* the text tokens moves the predicted action. **Text reaches the action head**, not just the
  video branch. The cheap route is a **constant** instruction, not an empty one. E-01.
- **"at 15 Hz" was never ours to claim.** 15 Hz is published for exactly one part — Jetson AGX Thor
  T5000 128 GB — and the next SKU down misses it. `Jetson AGX Orin` does not appear on the *policy*
  variant's tested-hardware list at all. E-03.

Full primary-source pass, with what is verified and what is not:
[`research/2026-08-15-cosmos3-edge-and-dreamzero.md`](research/2026-08-15-cosmos3-edge-and-dreamzero.md).

## State — 2026-08-16

**E-01 and E-02 have reported; E-03 is answered except for the one part only a human can answer.**
The code path exists: `wam.backbones.cosmos3_edge` (registry-constructible, no weights, no torch at
module scope) and `wam.robot.g1_dex3_28` (the 28-dim ↔ canonical mapping) are in the shared tree
with tests. No training has run and none may — T-39 still gates that.

| | |
|---|---|
| model chosen | `Cosmos3-Edge` 4B — **not yet staged, not yet downloaded**; 9.14 GB resident, measured from the safetensors headers |
| target hardware | **[?] still open, and it is a purchasing question, not a research one.** `Jetson AGX Orin` is *absent* from the policy variant's tested-hardware list; the local RTX 5090 fits inference **and** LoRA |
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
