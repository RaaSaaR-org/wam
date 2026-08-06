---
id: T-26
aliases:
- T-26
title: "Spatial-readout probe — was the mean-pool the limitation?"
slug: spatial-readout-probe-was-the-mean-pool-the-limitation
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
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Spatial-readout probe — was the mean-pool the limitation?

## Description

Spatial-readout probe (I-1, `docs/improvements.md`) — *✅ ran 2026-07-29 (ZeroGPU, 10/10 checks,
`runs/wan_probe/2026-07-29-zerogpu-5b-readouts.json`): **the mean-pool was not the limitation, and
the T-15/T-24 verdict survives the test.** Joints test R² — `mean` 0.310, `grid2x2` 0.370, `rand4`
control 0.376, `state_only` **0.456**; gripper 0.704 vs. state-only 0.881. `grid2x2` lands **below**
its own random control (0.338 vs. 0.366 on the val-selected pair), so the apparent gain over `mean`
is width, not geometry. Both bits false: `any_geometry_gain_over_control`,
`any_spatial_beats_state_only`. 7.6 s GPU, the three readouts share one set of forwards.
**Consequences:** T-16 keeps its premise unchanged and stays next; OD-04 needs no second look;
re-running T-24 spatially is now explicitly **not** worth it, since its precondition was Wan's
readout moving the number. Scope of the claim: frozen features under a linear readout, 96 windows,
one task — it does not speak to a fine-tuned backbone (T-16) or a non-linear head (I-2)* — T-15 and
T-24 both concluded "no frozen features beat the state-only ridge", but both measured the backbone
*through a mean-pool*, which deletes where things are. That shows the spatial signal does not
survive averaging, not that it is absent — and the weaker claim is what T-15/T-24 above record as
the stronger one. `scripts/hf_job_wan_probe.py --readout` now scores several token→vector readouts
on the *same* forward passes, same windows/labels/episode split/ridge code: `mean` (byte-for-byte
the historical pooling, so `runs/wan_probe/` stays reproducible and stays in `info.probe`),
`grid<R>x<C>` (average-pool the token grid into cells, keep them separate), and `rand<N>` (the same
tokens in N equally sized *random* groups — identical width, geometry removed). Wan's grid is
derived, not guessed: `WanI2VAdapter.token_grid()` gives (F'=2, H'=6, W'=8) for 5 frames at 192×256,
and the probe asserts the real token count against it before any reshape. **Decision rule, fixed
before the run:** grid > `rand` control *and* > state-only → position carries action signal, T-16's
premise changes and OD-04 deserves a second look; grid ≈ rand → the recorded verdict stands and gets
stronger. The second branch is what happened. Free on ZeroGPU (`scripts/deploy_wan_space.py`,
readout box blank = default `mean,grid2x2,rand4`). 12 tests in `tests/test_wan_probe.py` +
`token_grid` in `tests/test_wan_i2v.py`, incl. the miniature of the experiment itself: a signal
living in one spatial cell survives `grid1x2` and is annihilated by `mean`. Cosmos3 (T-24) still
hands over a mean-pooled array via the legacy path; that stays as it is

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
