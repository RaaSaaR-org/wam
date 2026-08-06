---
id: T-30
aliases:
- T-30
title: "Read the chunk out of the flow branch, not the regression head"
slug: read-the-chunk-out-of-the-flow-branch-not-the-regression-hea
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- backbone
- eval
- cluster
- prereg
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-01
updated: 2026-08-02
---

# Read the chunk out of the flow branch, not the regression head

## Description

Read the chunk out of the flow branch instead of the regression head (I-3, `docs/improvements.md`).
**✅ ran 2026-08-01 (`cluster/discoverer/63_eval_t30_flow_head.sbatch`, job 184670, 10 arms, one GPU)
— decisively negative, and the pre-registered mechanism does not explain it either.** All nine flow
arms land **below L0** against the regression readout's L0/50.57. On the pre-registered mean-of-8
*measurement* arm — the one the rule keys on, chosen so the sampler is not charged its own
conditional variance — mse is **1.23988e-04 against the regression head's 1.11298e-05, 11.1× worse**
(`skill_vs_repeat_pct` −1256.9 vs −21.8, score 20.0 vs 50.57). The single-draw deployment arm is
28.1× worse (3.13045e-04, score 0.0), and the variance-matched warm start does not rescue it
(1.45874e-04, still 13.1× worse) — so branch (b), the sampler's own `t=1` conditioning mismatch,
bounds the effect without accounting for it. Step count is not the lever either: 1/4/16/32/64 steps
all sit in the same band. **The mechanism the pre-flight anchors were built to test is refuted, not
confirmed.** `FLOOR_MSE` = 1.68201e-05 is the score of a chunk with the right content and no
position, and the rule said "landing near the floor is the *expected* outcome of an order-blind
sampler". The arms land at **7.4× the floor**, far above it — so the flow readout is not merely
unable to place a chunk in time; it does not carry the chunk's content through the sampler at all.
Against `CEILING_MSE` = 8.10372e-07 (the `action_encoder` → `action_recon` round-trip that motivated
the whole experiment) the best arm is **153× worse**. The latent does carry the chunk — the
round-trip proves it — and neither readout we have gets it out: the regression head throws it away,
and the flow sampler as trained cannot reconstruct it. **Consequence:** the readout axis is closed
as a cheap win. The velocity-head repairs (D2 step index, D3 `t` embedding) were already contingent
on the flow branch having something transportable; this says the branch does not, *as trained*, so
they stay unimplemented and become a **retraining** question rather than a re-scoring one. Nothing
here moves the T-16 verdict, which stands on the regression readout that remains the best one we
have We train two action paths and deploy the cheaper one: `velocity_head` is rectified flow on
action latents, co-denoised with video at a shared `t`, and `action_head` regresses the whole chunk
in one shot from pooled features. Only the second is ever sampled. **Measured first, on the real
checkpoint against the 1 040 archived holdout chunks:** the deployed head scores 1.21027e-05 while
the `action_encoder` → `action_recon` round-trip scores **8.10372e-07** — 15× better than the
readout and 11× better than the 9.14e-06 repeat-last-action bar T-16 failed. The latent carries the
chunk; the single-shot readout throws it away and under-shoots magnitude by 44 % (RMS 0.00226
against the demonstrations' 0.00404). So this is a **readout** experiment, needs no retraining, and
re-scores a checkpoint we already have. What shipped: `sample_action_chunk(pooled, *, steps, seed,
mean_of, t0, init_latent)` on `JointWorldActionModel`,
`--flow-sampler`/`--flow-steps`/`--flow-mean-k`/`--flow-t0`/`--flow-seed` on `eval_t16.py` (all
**off** by default, so archived runs stay reproducible), `readout_tag()` as the single definition of
the artifact suffix plus an `--out` guard that refuses before the multi-GB base load when a
directory already holds a different readout's artifacts, and `timing.json` so the sampler's latency
cost against the 500 ms deadline / ≥2 Hz floor is measured rather than assumed. **Three things the
rule had to be taught before it was allowed to run, all recorded in the file as `T30_RULE_V2` with
V1's defects kept rather than edited out:** (a) the headline metric structurally penalises a sampler
— `E‖a−draw‖² = E‖a−mean‖² + E‖draw−mean‖²`, so an unbiased single draw loses to the conditional
mean by exactly the conditional variance, and scoring it against a mean-seeking regressor would
reward the defect under test; the rule keys on a mean-of-8 arm (a *measurement*, never a policy) and
gates deployment separately on the single-draw arm. (b) The **sampler's own** conditioning mismatch:
training co-noises video and action at a shared `t`, but `sample_action_chunk` does one backbone
pass at `t=1` and reuses it at every `t_k`, so near `t_k=0` the head sees a combination it never
trained on — I-7's structure one level down. A warm-start arm (`--flow-t0`, variance-matched) bounds
it, and every negative branch says "as sampled this way"; the faithful sampler is not run and no
branch refutes the branch as trained. (c) The **smoothness confound cuts both ways**:
`weights.smoothness = 0.01` is applied to `decoded_targets` only, so the regression head carries a
jerk penalty the flow branch does not — an unhandicapped readout beating a handicapped one is the
objective's own consequence, and the *positive* branches now say so too. V1 held the negatives to an
alternative reading and the positives to none. **Pre-flight, so the verdict is read against anchors
and not vibes:** `CEILING_MSE` 8.10372e-07 (the round-trip) and `FLOOR_MSE` **1.68201e-05** — the
score of a chunk with the right content and no position, since step index is recoverable from the
latent at ~100 % accuracy while `ActionVelocityHead` takes no step index. Landing near the floor is
the *expected* outcome of an order-blind sampler, so the "cannot place a chunk in time" mechanism is
gated on actually landing there and is otherwise recorded as a hypothesis

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
