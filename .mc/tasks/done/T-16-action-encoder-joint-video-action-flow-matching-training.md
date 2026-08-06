---
id: T-16
aliases:
- T-16
title: "Action encoder + joint video/action flow-matching training"
slug: action-encoder-joint-video-action-flow-matching-training
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
created: 2026-07-26
updated: 2026-08-01
---

# Action encoder + joint video/action flow-matching training

## Description

Action encoder + joint video/action flow-matching training; frozen parts registry, selective blocks
(FR-03, §10.3) — *✅ **run on real data 2026-07-30, and the answer is negative**: 20 000 steps of
Wan2.2-TI2V-5B LoRA (r=32, α=64, blocks [2, 10]) on the 362 real GR00T G1 episodes, one H200 on
Discoverer+ (`cluster/discoverer/50_train_t16.sbatch`, `configs/training/joint_wan_gr00t.yaml`, ~11
epochs), checkpoint `runs/t16-lora-seed0/checkpoints/step-020000` (config `45ee9e60`, dataset
`sha256:598f193f`, git `78fc56d`). Scored by `scripts/eval_t16.py` on the **proven** 40-episode
holdout — same split, same 1 040 chunks as both earlier runs (`runs/t16-lora-seed0/eval-latest/`):
**WAM-Bench L0, 48.4/100. It does not clear the bar.** skill_vs_zero **+25.9 %**, but
**skill_vs_repeat −32.4 %** (ci −50.7 %) — *worse* against causal repeat-last-action than the
action-only baseline's −20.9 %, and raw mse 1.21e-5 against that baseline's 1.10e-5. **Correction
2026-08-01 (T-29 / I-7): every number in this entry is a `tiled` number** — one camera frame
repeated 9× at inference, while training fed the real 9-frame window. Re-scored in the mode it was
trained in (`--frame-history`, same checkpoint, same proven holdout, same 1 040 chunks,
`runs/t16-lora-seed0/eval-t29-history/`): **skill_vs_repeat −21.80 %** (ci −23.11 %), skill_vs_zero
+31.83 %, mse 1.11298e-05, horizon_ratio 1.32, smoothness_ratio 0.32 — still **L0**, score **50.6**.
**L1 still fails, by 21.80 pp: the verdict below stands, the published figure does not.** What does
*not* survive is the comparison — `d1-full-gen-seed0` (−20.9 %) and `t18-real-ablation-seed0`
(−129.0 %) are still tiled-only, so "worse than the action-only baseline" rests on −32.4 % vs −20.9
%, and in distribution it is −21.80 % vs an unknown. **AC-07 is therefore back to OPEN**,
undetermined pending their re-score (~0.4 GPU-h, no retraining), which also qualifies "the
pretrained prior does not rescue AC-07" below. **Closed 2026-08-01, same day, zero allocation**
(`scripts/rescore_archived.py`, laptop CPU, ~7 s per run — both archived checkpoints are ~0.9 MB):
neither baseline moved (−20.86 → −20.88 %, −129.04 → −129.00 %), the ladder is single-mode, and the
frame-mode confound turns out to be **backbone-specific** — 10.65 pp for Wan, ~0.03 pp for `tiny`,
which is the first positive evidence here that the pretrained prior carries temporal information the
action head can reach. In one mode AC-07 reads: the clean same-backbone ablation is −129.00 %
against −20.88 %, so the world branch costs **108 pp** on `tiny`; T-16 against that baseline is a
0.92 pp gap between runs differing in backbone *and* branch, i.e. not a clean ablation. The prior
recovers the 108 pp and buys nothing past it — **no measurable world-action advantage**, unchanged,
but now an answer about models rather than about frame windows (`docs/improvements.md` I-7). Full
corrected table, mode labels and mixed-mode warning: `docs/benchmark.md` "The T-16 result". The
*tiled* score of 48.4 — the one comparable to what follows, since these are all tiled numbers —
beats both earlier runs (28.6 / 19.9) only on the diagnostics: horizon_ratio 1.30 and
smoothness_ratio **0.29**, i.e. the fine-tune predicts motion 3.4× smoother than demonstrated — a
damped, averaged trajectory, which is what a model does when it has learned the pose distribution
and not the task. **The pretrained prior does not rescue AC-07 either.** T-15/T-24/T-26 showed
frozen features carry no action signal past a state-only ridge; fine-tuning the backbone on the task
itself does not change the verdict on this dataset. What the number does **not** license: a claim
about world-action modelling in general. One task, 402 success-only episodes, and a gripper channel
with peak-to-peak 0.120 that never opens or closes (T-27's warning), so nothing measured here can
see a grasp. The bottleneck moved: it is **data (D1/D2 real teleop)**, not backbone choice, not
compute, not code. `scripts/export_lora.py` + the Space's LoRA box make the trained prior watchable
as video without WAM in the loop (`docs/hf_jobs.md`)*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
