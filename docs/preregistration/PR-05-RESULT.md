# PR-05 result — the dream moves; one of my two gates was void

Ran 2026-08-05 on the ZeroGPU **dream** tab, one GPU call, peak VRAM 32.5 GB. Checkpoint
`t16-lora-seed0` step-020000 (`config_hash 45ee9e60…`), 16 clips from 8 episodes, 9 × 128×160 —
the trained geometry — 32 Euler steps, seed 0, free sampling (`anchor_latent_frames=0`).
Artifacts: `runs/dream/t35-zerogpu-seed0/`. Pre-registration: `PR-05-dream-motion.md`.

| arm | motion | ratio vs `recon` | static fraction | pixel std |
|---|---|---|---|---|
| `gt` (recorded) | 0.453 | 0.835 | 0.914 | 51.34 |
| **`recon`** (VAE round-trip, the denominator) | **0.543** | 1.000 | 0.961 | 51.40 |
| **`lora`** (WAM flow, text **+ state**) | **0.567** | **1.046** | 0.883 | 51.43 |
| `base` (adapter disabled) | 22.456 | 41.4 | 0.000 | 88.29 |
| `base_seed1` (the null) | 21.536 | 39.7 | 0.000 | 89.92 |

`d(lora, base)` = 84.48 · `d(base, base_seed1)` = 101.11

## G1 — does it move? **MOVES.** Ratio 1.046, floor 0.5.

The dream moves **as much as the recorded data does**, at the geometry it was trained at, with
the proprioception token supplied. The contact sheets say the same thing the numbers do:
`lora.png` is a clean, stable, correctly-coloured apple-and-plate scene, indistinguishable at a
glance from `recon.png`.

**This retires "it learned to predict almost no change" as a reading of the 2026-07-30 table.**
That finding contrasted 0.73 against a base arm scoring 29.5 and an unmeasured claim that real
demonstrations "move considerably more". They do not. `recon` — the same clips, same pipeline,
flow removed — is **0.543**, and 96 % of its frame pairs move less than one grey level. The
corpus is nearly static over 0.27 s, and the archived reading mistook *the data being static* for
*the model having learned to be static*.

What did not change: the model is still not a good policy. It reaches L0 and loses to
repeat-last-action (−21.80 %). A dream that matches the data's motion is a statement about the
video branch, not about control.

## G2 — did the fine-tune change the prior? **VOID. Not a verdict.**

The gate as pre-registered fires `INDISTINGUISHABLE` (84.48 < 101.11), which by §4's table would
read as verdict **C**, "the clips move because the base prior moves". **That reading is false on
these numbers and I am not recording it.**

The `base` arm is *noise* — motion 22.5, pixel std 88.3 against the real 51.4, static fraction
**0.000**, and `base.png` is the same "psychedelic colour noise" the archived table reported at
this geometry. Mean absolute pixel distance cannot discriminate when one arm is noise: two
independent noise draws differ from each other (101) by more than a clean image differs from
either (84). The gate measures the base's own variance, not the fine-tune's effect.

**This is a defect in the pre-registration, not a finding about the model** — and an avoidable
one: `docs/hf_jobs.md` had already recorded that the base produces garbage at 9 × 128×160, so
§4 should never have used a noise arm as its reference. Recorded here rather than quietly fixed,
because a gate rewritten after seeing its output is not a gate.

By every quantity that is not that gate, the fine-tune obviously changed the prior: 22.456 → 0.567
motion, std 88.3 → 51.4, static fraction 0.000 → 0.883. A corrected G2 would have to compare
*distributional* statistics against the reference arm rather than pixel distance against noise —
e.g. `|std(arm) − std(recon)|`, or `d(lora, recon)` against `d(base, recon)`. Fixing it is a new
pre-registration, and it should not be run on this checkpoint's already-seen numbers.

## One observation, explicitly not a result

Three of the four clips on the contact sheets put the apple in the same place the recording does
(row 3: apple beside the plate in both; rows 2 and 4: apple on the plate in both; row 1 differs).
If that holds it would mean the state token is carrying scene configuration into the sample,
which is the one thing this route can do and the diffusers route cannot. Four clips is nothing,
the comparison was made by eye, and no gate covered it. It is a reason to design a test, not a
claim.

## Caveats that bound all of the above

- **16 clips.** The per-clip motion distribution is heavily skewed (locally, over 200 clips:
  median 0.606, 90th percentile 2.244, max 5.216), and this sample's `gt` reads 0.453 where the
  200-clip measurement reads 0.958. The *ratio* is safe because `recon` comes from the same
  clips; any absolute number here is not.
- **One seed, one checkpoint, one geometry.** PR-05 §4 says a verdict that flips between seeds is
  no verdict, and the second seed has not been run for the `lora` arm.
- **Still not training data.** Nothing here touches PR-04's finding that the next corpus needs
  failures, randomized placement and an unfrozen right hand — none of which a generator fitted to
  402 success-only episodes can invent. The honest form of that question is `screen_corpus.py` on
  a generated corpus, and it remains unrun.
