# PR-06 result — it moves, it knows where things are, and it predicts worse than doing nothing

Ran 2026-08-05 on the ZeroGPU **dream** tab. Checkpoint `t16-lora-seed0` step-020000
(`config_hash 45ee9e60…`), 16 motion-selected clips from 8 episodes, 9 × 128×160 — the trained
geometry — 32 Euler steps, seed 0, `anchor_latent_frames=1`. GPU wall **56.3 s**, peak VRAM
32.5 GB. Artifacts: `runs/dream/t36-zerogpu-motion-seed0/`. Pre-registration:
`PR-06-dream-prediction.md`.

**Run twice on two independently assigned ZeroGPU workers; the two reports are byte-identical**
apart from `peak_vram_gb`. Nothing below rests on a single sampling draw.

## Free arms — G1

| arm | motion | ratio vs `recon` | static fraction |
|---|---|---|---|
| `gt` (recorded, 120×160) | 4.092 | 0.989 | 0.01 |
| **`recon`** (VAE round-trip, the denominator) | **4.138** | 1.000 | 0.01 |
| **`lora`** (free dream, text **+ state**) | **2.434** | **0.588** | 0.03 |
| `base` (adapter disabled) | 22.683 | 5.481 | 0.00 |

### G1 — does it move? **MOVES.** Ratio 0.588, floor 0.5.

It clears the pre-registered floor, and it is a **much weaker number than PR-05's 1.046 on the
same gate, same checkpoint, same geometry**. The only thing that changed is the windows. On the
moments the robot is actually acting the free dream generates ~59 % of the recorded motion; on
the two dead moments PR-05 sampled it generated 105 % of almost nothing.

**This supersedes PR-05's G1.** That result is not withdrawn — it is re-read as what it measured:
whether the dream stands still where the data stands still. It does. That was never the question.

Two side observations, neither gated. `recon` reproduces `gt`'s motion to 1 % (4.138 vs 4.092)
where on PR-05's static windows the round-trip read 1.20× the recording — the VAE is much closer
to lossless on real motion than on quantization noise, which makes the denominator more
trustworthy here than there. And `base` scores 22.683 against PR-05's 22.456: the adapter-disabled
prior produces the same colour noise at this geometry regardless of which windows it is given,
exactly as `docs/hf_jobs.md` recorded.

## Anchored arms — G2

Latent frame 0 pinned to the observation, the copied pixel frame stripped, 8 frames scored.

| arm | motion | ratio | distance to `recon_a` |
|---|---|---|---|
| **`recon_a`** (what actually happened) | 4.130 | 0.998 | — |
| **`freeze_a`** (frame 0 held) | 0.000 | 0.000 | **12.020** |
| **`lora_a`** (prediction, seed 0) | 3.435 | 0.830 | **16.656** |
| `lora_a_seed1` (prediction, seed 1) | 3.708 | 0.896 | 17.322 |

### G2 — does it beat freezing? **NO_BETTER_THAN_FREEZING.** And not narrowly.

The gate asked for `d(lora_a, recon_a) < 0.9 · 12.020 = 10.818`. The prediction scores **16.656**
— it is not 10 % closer to the truth than doing nothing, it is **39 % further away**. Seed 1 is
worse still (17.322), so both seeds land on the same side and the result is **stable**, not
`UNSTABLE`. Standing still would have been a better prediction than the world model's.

This is §6 row 2, the outcome the pre-registration named as most likely: **it generates motion of
roughly the right magnitude (ratio 0.830) without predicting the right motion.**

### What the clips show, and why it sharpens the verdict rather than softening it

`lora_a.png` beside `recon_a.png` is worth more than the two numbers. **The anchor works.** In all
four rendered clips the prediction has the scene in the right place — arm entering from the left
with the apple in its fingers, plate where the plate is, correct colours, correct framing, correct
apple-on-plate vs apple-beside-plate configuration. It is unmistakably a continuation of *that*
clip and not a generic one.

What it gets wrong is the motion. The arm wobbles instead of travelling, the plate smears and
deforms (clip 3 is the clearest), the apple jitters in place. Frame-to-frame it moves 4.50 where
the truth moves 4.14 — the right amount of change, spent in the wrong directions. Error that
freezing never incurs.

So the failure is specific and it is not "the model ignored the conditioning": scene layout is
carried, dynamics are not.

## What could still be wrong with this, and what it would take to know

**Anchoring by replacement is a mode the model was never trained in.** Training noised the whole
window and denoised the whole window; pinning a frame at each Euler step is a standard
inpainting-style intervention and `sample_video`'s docstring labels it one. A negative on G2 could
in principle be the technique rather than the branch.

Two things argue it is not the whole story — neither is a gate:

- the pin demonstrably reaches the free frames, or the scene layout would not match clip for clip;
- anchoring moves the sample toward the truth's motion level (free `lora` 2.434 → anchored
  `lora_a` 3.435 against a truth of 4.130), so the anchor is shaping generation, not being ignored.

The clean separation is a third point that was not computed: `d(lora[:, 1:], recon_a)`, the *free*
sample scored on the same target. If the free sample were no further from the truth than the
anchored one, the anchoring would be doing nothing and G2 would be void the way PR-05's was. That
is one cheap CPU comparison on saved arms and it is a new pre-registration, not a patch to this one.

## Caveats that bound all of the above

- **Not a policy result.** The checkpoint reaches L0 and loses to repeat-last-action by 21.80 %.
  `dream.py` never touches the action head; nothing here moves that number in either direction.
- **Not training data**, and this run makes that case *worse*, not better: a generator whose 0.3 s
  predictions are further from the truth than a still frame is not a source of supervision. PR-04's
  requirement — failures, randomized placement, an unfrozen right hand — is untouched.
- **Not a corpus statistic.** Every number here is on motion-selected windows. `gt motion 4.034`
  (CPU reference) and `4.092` (this run) describe the moments the robot acts, not the corpus.
- **16 clips, one checkpoint, one geometry.** Ratios are safe because numerator and denominator
  come from the same clips; no absolute number here generalizes.
- **`base` is reported for continuity only.** No gate reads it, and PR-05 already established what
  it is. It is not evidence for anything in this document.
