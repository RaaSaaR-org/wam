---
id: T-29
aliases:
- T-29
title: "Frame history at inference — clear the confound under T-16/T-18"
slug: frame-history-at-inference-clear-the-confound-under-t-16-t-1
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
- cluster
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-01
updated: 2026-08-01
---

# Frame history at inference — clear the confound under T-16/T-18

## Description

Frame history at inference — clear the confound under the T-16/T-18 verdicts (I-7,
`docs/improvements.md`). Training feeds the real `num_frames` window ending at the chunk
(`datasets.py:156`); `predict()` fed one frame tiled `num_frames` times (`joint.py:388`), so a
backbone trained on a moving clip was scored on a freeze-frame — with no visual motion left to beat
repeat-last-action with. **✅ ran 2026-08-01 (`cluster/discoverer/61_eval_t29_frame_history.sbatch`,
job 184648, 861 tests green): the mismatch was worth +10.65 pp and the bar is still not cleared.**
Both arms on one H200 from one checkpoint (`runs/t16-lora-seed0/checkpoints/step-020000`), the same
proven 40-episode holdout, the same 1 040 chunks, differing in exactly one flag
(`runs/t16-lora-seed0/eval-t29-{tiled,history}/`): `skill_vs_repeat_pct` −32.45 % → **−21.80 %**
(+10.65 pp), `ci_skill_vs_repeat_pct` −50.74 % → −23.11 % (+27.64 pp), `skill_vs_zero_pct` +25.88 %
→ +31.83 %, mse 1.21027e-05 → 1.11298e-05 (−8.0 %), level L0 unchanged, score 48.4 → 50.6 (spec
0.1.0) / 28.4 → 30.6 (spec 0.2.0). The tiled arm reproduces the archived `eval-latest/bench.json` to
every digit, so the published −32.4 % is **confirmed** to be the freeze-frame measurement and the
A/B is clean. The confound was real and worth about a third of the 32.45 pp gap; it did not come
close to closing it. **Only `t16-lora-seed0` was re-measured** — `t18-real-ablation-seed0` and
`d1-full-gen-seed0` are still tiled-only, so the ladder is mixed-mode, AC-07 is undetermined pending
their re-score (~0.4 GPU-h, no retraining), and every pre-2026-08-01 number is a "tiled" number
wherever it sits next to a history-mode one. **That re-score ran the same day for zero allocation
and neither run moved (−20.88 % / −129.00 %), so the ladder is single-mode again** — and the ~0.4
GPU-h estimate was itself wrong by the whole GPU: the archived checkpoints are ~0.9 MB and
`scripts/rescore_archived.py` scores 1 040 chunks on a laptop CPU in ~7 s. Local equivalent in
`docs/local_gpu.md` §3b. What shipped: `Observation.image_history` (optional — `evaluate_policy`
stays policy-agnostic; `INTERFACES_VERSION` 0.3.0), `wam.data.episode.frame_window_indices` as the
**single** definition of the window that `EpisodeDataset` and `build_eval_pairs` both call,
`resolve_frame_context` shared by both `predict()` implementations so the action-only and
world-action models cannot be fed different clips (AC-07), and `--frame-history` on `eval_t16.py` —
**off by default**, so archived runs stay reproducible and the A/B is explicit. Verified
bit-identical to the pre-T-29 path on real chunks from `d1-full-gen-seed0`. The regression guard
needed frames that move: the synthetic D1 recorder writes a constant image (max diff between
consecutive frames: 0), so a misaligned window was byte-identical to a correct one — a test that
could not fail is part of how this survived to a published number. **Still open:** the closed-loop
half (a rolling buffer in the policy — stateful, needs a reset rule and startup padding, changes the
deployed path), deliberately deferred until the offline run says it matters. **The offline run has
now answered that condition, and the deferral stands.** The frame window is worth 10.65 pp offline —
a real effect, and an argument for eventually building the rolling buffer — but it cleared no gate,
so the closed-loop half is still not what the next GPU-hour buys, on exactly the reasoning it always
had. **Decision rule fixed before the run:** `skill_vs_repeat_pct` moves materially toward 0 → both
trained world-action verdicts were measured out of distribution, `docs/benchmark.md` gets a
correction and AC-07 reopens; essentially unchanged → the negative gets much stronger, and the
bottleneck really is data (then I-8, the data-scaling curve, before committing months to D1/D2).
**Resolution 2026-08-01 — the rule above is left exactly as written, because it was wrong in a way
worth keeping.** It has two branches; the outcome had three. `skill_vs_repeat_pct` moved 10.65 pp
toward 0 and stayed 21.80 pp short of it — "materially toward 0" *and* "still fails L1" at the same
time, which this prose admits no verdict for. Of the six written copies, only the executable one
(`cluster/discoverer/61_eval_t29_frame_history.sbatch:143-154`) anticipated the middle case,
splitting on a numeric `b − a > 5.0` and routing it to "the negative stands but its size does not —
re-state both runs with the real window, then go to I-8". Picking that copy after seeing the data is
what pre-registration exists to stop, so it is recorded here as a judgement call rather than as the
rule having spoken; both readings agree L1 fails and send the next GPU-hour to the same place. AC-07
is back to open regardless, for a reason no copy anticipated — the mixed-mode ladder above, not
either branch. Full account: `docs/improvements.md` I-7

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
