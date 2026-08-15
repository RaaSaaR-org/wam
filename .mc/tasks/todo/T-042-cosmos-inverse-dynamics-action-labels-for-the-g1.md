---
id: T-042
aliases:
- T-042
- T-42
title: "Cosmos inverse dynamics — action labels for the G1, if we have footage to label"
slug: cosmos-inverse-dynamics-action-labels-for-the-g1
status: todo
priority: 3
owner: ''
projects: []
customers: []
tags:
- post-mvp
- data
- backbone
- prereg
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-15
updated: 2026-08-15
status_note: "Written 2026-08-15 after checking the Cosmos 3 card and cookbooks against a fair
  objection — NVIDIA calls this a world *action* model, so why did T-041 train video only?
  `docs/backbone-eval.md` §3 recorded the action port on 2026-08-05 and nothing followed it up.
  This is that follow-up. **Step 0 is free and decides everything: count the unlabelled real G1
  footage.** With none, an inverse-dynamics labeller labels nothing and this task closes as a
  paragraph. No GPU until step 0 reports and a pre-registration exists."
---

# Cosmos inverse dynamics — action labels for the G1, if we have footage to label

## Description

Post-train Cosmos3-**Nano** for **inverse dynamics** on our action-labelled G1 corpus, then use it
to attach canonical actions to real G1 video we have never labelled. The deliverable of the first
pass is a count and a pre-registration, not a checkpoint.

**Why this exists as a task now.** `docs/backbone-eval.md` §3 wrote, on 2026-08-05, that "Cosmos 3
takes JSON action arrays in and emits action states out" and that T-24 never used that port — then
§4 went on to design a probe treating the port as input-only, and the thread was dropped. Checked
against primary sources 2026-08-15, the port is genuinely bidirectional: the family ships **forward
dynamics** (actions → frames), **inverse dynamics** (frames → the trajectory that produced them)
and **policy** (observation + prompt → actions). Inverse dynamics is a video-to-action labeller.
Standing explanation and the receipts: `docs/action-labels.md` §3b.

**What it is not, stated first because the failure mode is enthusiasm.**

- **Not a way to label generated video.** Labels inferred from generated frames describe pixels the
  generator invented. PR-06 measured the cost of treating generated video as supervision: the
  anchored dream scored 16.656 from the truth where holding the conditioning frame scored 12.020 —
  39 % worse than standing still. Pointing a labeller at `runs/t041-apple-variations/` is
  categorically forbidden, not merely unpromising.
- **Not new behaviour.** An inverse-dynamics model extends the labelling it learned over footage it
  has not seen. It cannot exceed that labelling, and it adds no trajectories. Only teleop does.
- **Not T-32.** Labelling more frames of the same task is not a data-scaling curve and must not be
  reported as one.

## Step 0 — the free measurement that decides whether this is a task

**How many hours of real, unlabelled G1 footage do we actually hold?** Nobody has counted. Every
corpus in this repo is action-labelled by construction (`convert_lerobot_g1.py` reads the parquet
alongside the mp4), which is precisely why an inverse-dynamics labeller has had nothing to do.

Candidate pools, to be counted rather than assumed:

- `USC-PSI-Lab/humanoid-everyday` — 8 949 episodes, 3 436 171 frames, same G1 + Dex3, same
  RealSense D435. **Its licence is unresolved** (T-040 keeps it off the critical path for exactly
  this reason), and per T-041 it ships states and actions, so much of it may need no labeller.
- The `G1_Dex3_*` sets fetched by `92_fetch_g1_corpus.sbatch` — **meta + videos only** was the
  fetch. Whether the sources carry action parquet we chose not to download is a metadata query,
  not a download.
- Anything recorded once real teleop starts (M1/D2). This is the pool that makes the task
  worthwhile long-term, and it does not exist yet.

**If the count is ~zero, close this task.** That outcome is a paragraph in `docs/action-labels.md`
and costs nothing. Recording it is the point — it stops the idea being re-proposed every time
someone reads NVIDIA's front page.

## What it would take, if step 0 says go

- **Nano, not Super.** All twelve action notebooks under `cookbooks/cosmos3/generator/action/` are
  Nano; the only `finetune/` recipe is **Nano-Policy-DROID**. Super ships `action_gen=True` with no
  notebook, no post-training recipe, and a 121 GB export. Nano is 16B and peaked **36.2 GB** in
  T-24 inference — over the 5090's 34.36 decimal GB, so this is a Discoverer+ or ZeroGPU job.
- **A 28-dim G1 + Dex3 action vocabulary has to be added.** The card's supported list is camera 9D,
  AV 9D, egocentric 57D, Franka 10/20D, Agibot 29D, UR/Google/WidowX 10D, UMI 9D — no humanoid.
  NVIDIA's route is post-training *on action-labelled data*, so the labelled corpus is the input,
  not the output.
- **Block order is a known trap.** T-041 measured `action[0:14]` ↔ hand and `action[14:28]` ↔ arm
  by correlating against the recorded state — the README's example builds arm-first and is wrong.
  Getting this backwards produces a model that trains, converges and is silently useless.
- **Canonical mapping stays in one place.** Whatever Cosmos emits is robot-specific; conversion to
  `ActionMode.JOINT_DELTA` belongs in the same converter path as route 1, never duplicated
  (FR-06).

## Acceptance criteria

- [ ] **Step 0 reported**: a written count of unlabelled real G1 footage by pool, with the
      licence status of each, and an explicit close-or-continue call. Free, no GPU.
- [ ] `docs/preregistration/PR-10-*.md` exists before any GPU hour — hypothesis, arms, gate,
      verdict table, and what each verdict forbids. Same shape as PR-06/PR-08/PR-09.
- [ ] **The gate is a beat-the-baseline gate, and the baseline is not trivial.** This project has
      now watched three separate methods lose to repeat-last-action and a state-only ridge
      (T-16/T-18/T-27/T-30). Predicted actions must be scored on **held-out labelled** episodes
      against `past_joint_proj + state` (0.540 / 0.539 / 0.541 joints, 0.911 gripper at 48
      episodes, `runs/backbone_eval/action_baselines_ep48.json`) — the comparator recomputed on
      this task's own windows, per `backbone-eval.md` §4a, not quoted across sample sizes.
- [ ] **A held-out-episode check that the labeller is not memorising.** Labels predicted for
      episodes inside the post-training set are not evidence; the split is seeded and recorded in
      the manifest, as `prepare_cosmos_corpus.py` does.
- [ ] The pre-registration states, by name, that generated video is out of scope as a labelling
      target, and why (PR-06's 39 %).
- [ ] Downstream use is gated separately: labelled footage entering a training corpus runs
      `screen_corpus.py` (T-34) first, as PR-08 G0a requires of restyled data.

## Notes

**Ordering.** Nothing here is blocked in the hard sense, but T-39 still governs the reading: until
the positive control reports, "the data is wrong" and "the method is wrong" are not separable, and
building a machine that produces *more* data of the same kind is a bet on the first. Step 0 is free
and can run regardless; the GPU half should not start before T-39.

**The honest prior.** Every attempt in this repo to extract action signal from video features has
lost to proprioception plus past actions — T-15, T-24, T-26, T-38, and the flow readout in T-30.
Inverse dynamics is a different mechanism (a task the vendor pretrained for, not a linear readout
we bolted on), which is why it is worth a gate rather than a dismissal. It is not a reason to
expect a win.

%% mc-links: [[T-041]] [[T-040]] [[T-37]] [[T-39]] [[T-34]] %%
