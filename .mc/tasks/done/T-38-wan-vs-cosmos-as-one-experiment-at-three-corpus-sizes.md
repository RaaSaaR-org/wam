---
id: T-38
aliases:
- T-38
title: "Wan vs. Cosmos as one experiment, at three corpus sizes"
slug: wan-vs-cosmos-as-one-experiment-at-three-corpus-sizes
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
created: 2026-08-05
updated: 2026-08-05
---

# Wan vs. Cosmos as one experiment, at three corpus sizes

## Description

Run the Wan-vs-Cosmos comparison as **one** experiment instead of two, at three corpus sizes
(`docs/backbone-eval.md` §7, `scripts/compare_backbones.py`,
`runs/backbone_eval/compare_backbones.json`, `src/wam/evaluation/video_fidelity.py`). *§1 recorded
two verdicts — Wan loses, Cosmos3-Nano loses — reached a month apart and each compared against
**0.456 / 0.881**, a constant lifted from the first of them; T-37 §4a then measured that the
constant moves.* **Three defects, all removed by the driver rather than argued away.** (1) *Two runs
are not a comparison* — T-15 and T-24 ran the same window *code*, which is not the same as the same
*windows*. The driver asks both deployed ZeroGPU Spaces (`huhn511/wam-wan-smoke`,
`huhn511/wam-cosmos3-probe`) for identical parameters through `gradio_client` and then **verifies
from the returned reports** that the episode list, window count, context frames, resize, chunk
length, instruction, the train/val/test episode split and the dataset revision all agree; a
disagreement exits 2 and prints no table, because a comparison across two window sets is not a
weaker result, it is not a result. That check earned itself immediately: the Wan Space records
`window_select` and the deployed Cosmos copy predates the field, resolved by `--assume-default
window_select`, which writes the assumption into the artifact and **refuses** if the recorded value
is anything but `linspace`. (2) *Every archived verdict is a 12-episode verdict* (56 training
windows) — the driver sweeps 12 / 24 / 48 and recomputes the floor and the best input-only
comparator at each size on the same windows. Nothing is quoted. (3) *Wan is 3072-dim per block and
Cosmos3 is 4096*, so a raw delta is part prior and part tensor width — carried through a fixed
known-informative feature set at 6144 / 8192 / 112 dims with the same ridge, split and labels. **✅
ran 2026-08-05, ZeroGPU minutes, no new deployment.** Joints/gripper test R², val-selected block
pair: **12 eps** Wan **0.3652**/0.6976, Cosmos 0.3240/0.6126; **24 eps** Wan **0.3011**/0.6017,
Cosmos 0.2837/0.6998; **48 eps** Wan 0.3867/0.5420, Cosmos **0.4267**/0.8215 — against a floor of
0.4563 / 0.4879 / 0.5129 and a best-input-only comparator of 0.5118 / 0.5193 / 0.5399. **Three
readings.** The 12-episode row **reproduces T-15 and T-24 to four digits**, so nothing drifted in
the month between and the rows below are comparable with the archive. **The ranking reverses** — Wan
ahead by 0.041 joints at 12 episodes, behind by 0.040 at 48 — so *a single-size head-to-head between
these two backbones does not measure anything about the backbones*, and a single-size head-to-head
is exactly what the archived record consisted of. **What does not reverse: both lose to
proprioception at all three sizes**, the better of the two sitting 0.091 / 0.187 / 0.086 below the
floor with no trend and no closing; Cosmos's gripper at 48 (0.8215) is the only cell that comes near
its floor and still does not clear it. Free control, restated because it removes an explanation
before anyone reaches for it: **Cosmos3-Nano's VAE *is* the Wan2.2 VAE** (`AutoencoderKLWan`,
48-ch), so whatever separates the rows, it is not the latent space. **A correction to an earlier
version of that table, kept visible:** the backbone columns are each report's *val-selected* block
pair, and an earlier comparator column was picked by *test* R² across ~12 rows — up to **0.029**
higher, and at 48 episodes precisely the luckiest of three interchangeable projection seeds. Both
are in the artifact, the second labelled optimistic. The headline is unaffected (the floor is not
selected at all) but **the bar itself had been inflated**

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
