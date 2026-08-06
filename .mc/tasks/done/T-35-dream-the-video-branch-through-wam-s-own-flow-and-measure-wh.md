---
id: T-35
aliases:
- T-35
title: "Dream the video branch through WAM's own flow, and measure whether it moves"
slug: dream-the-video-branch-through-wam-s-own-flow-and-measure-wh
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
- prereg
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-05
updated: 2026-08-05
---

# Dream the video branch through WAM's own flow, and measure whether it moves

## Description

Dream the video branch through WAM's own flow, and measure whether it moves
(`docs/preregistration/PR-05-dream-motion.md`, pre-registered 2026-08-05 before any GPU sampling;
`src/wam/evaluation/dream.py`, `scripts/dream.py`, 40 tests, ZeroGPU **dream** tab). *Every clip the
video branch has ever produced came out of a stock `WanImageToVideoPipeline` (`generate_future`),
which has **no state port** — so the proprioception token the DiT was trained with
(`wan_i2v.py:605`) was absent from all of them. Same class of train/inference mismatch as T-29,
which was worth 10.65 pp when closed. `sample_video` is the third route: integrate `forward_flow` in
WAM's convention with the text **and state** context `co_denoise` builds. Unlike T-30's action
sampler it re-evaluates the backbone at every t_k, on the latent noised to exactly that t_k — the
pairing training used — so it carries no conditioning confound; that fidelity costs n backbone
passes, which is why the action branch cannot afford it at 2 Hz.* **The reference nobody had
computed, measured 2026-08-05 on CPU with no Wan weights:** the archived conclusion "it learned to
predict almost no change — 0.73 where the base manages 29.5 and a real 0.3 s of demonstration moves
considerably more" (`docs/hf_jobs.md`) rests entirely on that last clause, and it had no number.
Real 0.27 s clips move **0.958** (0–255, 200 clips over 40 episodes), **0.922** on the 128×160 grid
the decoded arms live on, with 68.9 % of frame pairs moving under one grey level — cross-checked
against a raw un-windowed episode at 0.86 mean / 0.425 median. **This corpus barely moves**, so 0.73
is ~0.8× the recorded motion, not ~0. That does not overturn the archived finding — those clips came
from a different route — it means the comparison was never available, and the pre-registered floor
(`MOTION_FLOOR_RATIO = 0.5` of the **VAE round-trip**, not of raw frames, since a decoded clip
inherits resize/stride/codec) exists so it cannot be chosen afterwards. **Five arms, one GPU call,
all at the trained 9 × 128×160:** `gt`, `recon` (the denominator), `lora`, `base` (adapter disabled
in place via `set_lora_enabled` — same weights, same process, so the delta is the adapter and not a
second model build), and `base_seed1`, the **null** without which a nonzero `d(lora, base)` says
only that sampling is stochastic. Two gates → verdicts A–D. **Shipped as code, not scratch output:**
the sampler's direction, grid and update rule are pinned by an exact test rather than argued —
rectified flow's velocity is constant along its path, so Euler with the true field must land on the
clean latent at *any* step count, and a reversed field fails the same assertion. Free sampling is
the faithful arm; anchoring is labelled an intervention and its copied pixel frames are stripped
before scoring. **What it explicitly does not claim:** that dreams are VLA training data — a
generator fitted to 402 success-only episodes cannot produce the 0 failure episodes or the
randomized placement PR-04 measures as missing, and the honest form of that question is
`screen_corpus.py` on a generated corpus. **✅ ran 2026-08-05 on ZeroGPU, one GPU call, peak VRAM
32.5 GB (`docs/preregistration/PR-05-RESULT.md`, `runs/dream/t35-zerogpu-seed0/`).** **G1 MOVES:
`lora` motion 0.567 against `recon` 0.543 — ratio 1.046.** The dream moves as much as the recorded
data, and the contact sheet is a clean, stable apple-and-plate scene next to a `base` arm that is
the archived "psychedelic colour noise" (22.456, std 88.3, static fraction 0.000). **That retires
"it learned to predict almost no change"** as a reading of the 2026-07-30 table: the unmeasured
clause was "a real 0.3 s of demonstration moves considerably more", and it does not — the corpus is
nearly static over 0.27 s (96 % of `recon`'s frame pairs move under one grey level), so the archived
reading mistook *the data being static* for *the model having learned to be static*. **G2 is VOID
and no A–D verdict is recorded.** It fires INDISTINGUISHABLE (84.48 < 101.11) which §4 maps to
verdict C, "the clips move because the base prior moves" — false on these numbers, because mean
absolute pixel distance cannot discriminate when one arm is noise: two noise draws differ from each
other more than a clean image differs from either. **That is a defect in my own pre-registration**,
and an avoidable one, since `docs/hf_jobs.md` had already recorded that the base produces garbage at
this geometry; recorded rather than quietly patched, because a gate rewritten after seeing its
output is not a gate. A corrected G2 compares distributional statistics against `recon` instead of
pixel distance against noise, and is a new pre-registration. Bounds stated: 16 clips (the per-clip
distribution is skewed — `gt` reads 0.453 here against 0.958 over 200 clips, which is why only the
ratio is safe), one seed, one geometry, and none of it touches whether dreams are training data.

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
