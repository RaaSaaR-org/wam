---
id: T-33
aliases:
- T-33
title: "Grasp anticipation on the restored gripper channel"
slug: grasp-anticipation-on-the-restored-gripper-channel
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
- cluster
- prereg
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-02
updated: 2026-08-02
---

# Grasp anticipation on the restored gripper channel

## Description

Grasp anticipation on the restored gripper channel — the one question this corpus has left
(`docs/preregistration/PR-03-grasp-anticipation.md`, pre-registered 2026-08-02 before any predictor
was fitted on `datasets/gr00t-apple-grip`). PR-01 retired arm-trajectory MSE as a ranking metric
because 66.0 % of its achievable range is reachable blind; `PR-01-GRIPPER.md` then found the single
place in this data a blind predictor demonstrably cannot reach — the ~4 steps around a grasp flip,
where every blind family sits at a coin toss and **44 % of the accuracy range is not
blind-reachable**. Everything else about the channel is momentum (a zero-parameter repeat-last rule
scores 97.82 % on the full holdout), so **full-holdout gripper accuracy is pre-registered as
forbidden** and only `postflip_accuracy` on transition chunks is scored. **Shipped for it:**
`scripts/widen_holdout.py` + `configs/splits/pr03_holdout_150.txt` — 150 episodes, a strict
**superset** of the 40-episode T-18 holdout (seeded permutation, not a sorted prefix), leaving 252
to refit on. Nesting is the point twice over: it makes PR-03's archive gate a like-for-like
comparison (restrict to the original 40, reproduce `PR-01-GRIPPER`'s table), and it means widening
can only move episodes **out** of training, never a scored episode back in. **`runs/t16-lora-seed0`
cannot be scored here** — it trained on 110 of these 150 — and `eval_t16.py`'s split proof will
refuse it, which is why PR-01-GRIPPER §2 says convert once, re-split, **refit** once. **Two gates
before any GPU-hour is requested,** both CPU: (1) the blind control suite must reproduce the
archived 40-episode table (±0.5 pts for the two zero-parameter rules, ±2.0 for the three fitted
ones) or the converted channel is not the channel PR-01-GRIPPER restored analytically; (2) the
**power gate** — `n_postflip >= 2000` and bootstrap `ci_halfwidth <= 3.5` points, thresholds fixed
in the document, since MDE runs at ~2x the half-width and 40 episodes gave 660 steps / ±7.56 pts /
~15 pts MDE. **If gate 2 fails the refit is not submitted and no allocation is spent** — the
recorded consequence is PR-01-GRIPPER §4, collect demonstrations. Verdicts A/B/C, ties go to B, and
a verdict that differs across the three debounce definitions is no verdict. Splits pinned by
`tests/test_splits.py`, including the overlap that makes T-32's rungs (40/110/110 episodes)
**unusable against this holdout** without regeneration. **✅ both gates ran 2026-08-02, CPU only, and
the answer is a pre-registered STOP (`docs/preregistration/PR-03-RESULT.md`,
`scripts/bench_grasp_anticipation.py`, `runs/pr03/`).** Widening worked — post-flip steps **675 → 2
682** (4.0×) and the episode-bootstrap half-width **6.84 → 4.10** — and `n_postflip` clears its bar
(2 682 ≥ 2 000) over 302 transition chunks in 148 of 150 episodes. But **`ci_halfwidth` 4.10 misses
the pre-registered ≤ 3.5 by 0.60**, so the MDE is ~8.2 points and PR-03's clause fires: **the refit
is not submitted, no GPU-hours requested** (~20–40 GPU-h not spent), and the recorded consequence is
`PR-01-GRIPPER.md` §4 — collect demonstrations. 3.5 was fixed before measurement and the repo
convention versions thresholds rather than editing them, so this is a fail and not a near-pass. The
blind suite on the 150 reproduces the *shape* the whole hypothesis rests on — a **V, not a decay**:
pre-flip is easy for everything (76–84 %), repeat-last and the 32-dim ridge collapse **below
chance** post-flip (24.76 % / 32.92 %), const-velocity 66.59 %, ceiling **75.35 %**, and the worst
place for every blind predictor is the four steps at the flip (ceiling 59.01 %). Headroom is real
but **smaller than archived** (24.65 pts post-flip, 40.99 at k…k+3, vs ~29/47 implied by
PR-01-GRIPPER) because this ceiling is stronger — the conservative direction. **Gate 1 is a MISS,
and ambiguous by construction**: PR-01-GRIPPER's numbers came from a scratch script that was never
committed, so it could only be re-implemented from prose. Transition chunks reproduce **exactly**
(78/78, 39/40 episodes) — the numbers a botched conversion breaks first — while the two
zero-parameter rules miss a ±0.5 tolerance by 0.8–1.8 pts over 15 extra post-flip steps (675 vs
660), which is an indexing convention rather than a value, and `time-only` misses by +14.10 because
PR-03 fixed the metric but never that control's functional form (my defect; it is now in committed
code rather than in prose). So it does **not** license a claim that the conversion damaged the
channel. **Two things caught by building it:** the first ceiling scored 59.11 % — *below* a
zero-parameter rule — because the RBF bandwidth grid was carried from a 32-dim problem to a 256-dim
input (`exp(−0.02·512)` underflows) and because it was fitted by least squares on the continuous
channel while scored on accuracy at 0.5; fixed by scaling γ as 1/D and fitting the binarized label
(59.11 → 73.19 %), and `ceiling_dominates` is now reported on every run so it cannot pass silently.
And under the `self-contained` definition the ceiling is still beaten by a clock-only model (50.93 %
vs 54.48 %), so PR-03's binding robustness clause could not have been evaluated soundly even had
gate 2 passed. Reaching h ≤ 3.5 needs ~205 holdout episodes leaving ~197 to train on — a **new
pre-registration**, not an edit to PR-03, and it spends training data to buy resolution against an
untested assumption that the fine-tune is not starved below 250

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
