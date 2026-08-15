---
id: T-042
aliases:
- T-042
- T-42
title: "Cosmos inverse dynamics — action labels for the G1, if we have footage to label"
slug: cosmos-inverse-dynamics-action-labels-for-the-g1
status: done
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
status_note: "**Closed 2026-08-15 by step 0, exactly as written: the count is zero.** Real G1
  footage we hold with video, no actions and no way to get them: **0 clips**. The 3 554 episodes
  looked unlabelled only because two fetch scripts passed `--include 'meta/*' --include 'videos/**'`
  — upstream all 14 repos publish the action parquets (415 files, 647 MB), verified through the HF
  tree API without downloading. A labeller built to recover labels that `--include 'data/**'` would
  download is not amortisation, and its output would be strictly worse than the recording. No GPU
  hour was spent and no pre-registration was needed. Receipts and the wider finding:
  `docs/action-labels.md` §3b. **Re-open the day teleop produces video faster than it produces
  labels** — not before."
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
  **Scope note added 2026-08-15:** that measurement is on `USC-PSI-Lab/Humanoid-Everyday-G1`
  (LeRobot v2.1) and does **not** transfer to the `unitreerobotics/G1_Dex3_*` sets, which are
  v3.0 and **arm-first**. See T-043 §1 — the trap is real, and it is per-corpus.
  Getting this backwards produces a model that trains, converges and is silently useless.
- **Canonical mapping stays in one place.** Whatever Cosmos emits is robot-specific; conversion to
  `ActionMode.JOINT_DELTA` belongs in the same converter path as route 1, never duplicated
  (FR-06).

## Acceptance criteria

- [x] **Step 0 reported**: a written count of unlabelled real G1 footage by pool, with the
      licence status of each, and an explicit close-or-continue call. Free, no GPU.
      **Done 2026-08-15 — the call is *close*.** `docs/action-labels.md` §3b.

**The remaining criteria were conditional on step 0 saying *go*, and it said *stop*. They are
unmet because they were never entered, not because they failed** — the distinction matters if this
task is ever re-opened, at which point they still stand as written.

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

## Report — 2026-08-15, closed

**Step 0 ran, cost nothing, and killed the task.** That was the designed outcome of a cheap
measurement, so it is recorded as a result rather than an abandonment.

**The count.** Real G1 footage we hold with video, no actions, and no way to get them: **0 clips**.
The naive on-disk reading suggested otherwise — 3 554 real G1 episodes / ~25.5 h across 14 sources,
3 163 clips in `cosmos-g1-embodiment`, not one `actions.parquet` beside them. The cause was ours:
`cluster/discoverer/92_fetch_g1_corpus.sbatch` passes `--include 'meta/*' --include 'videos/**'`
and `workstation/10_fetch_corpus.sh` narrows to one camera. Upstream, all 14 repos publish the
action parquets — **415 files, 647 MB**, same Apache-2.0 / CC-BY-4.0 repos we already pulled 69 GB
of video from. Verified through the HF tree API, nothing downloaded.

So the premise fails on its own terms: recovering labels that `--include 'data/**'` would download
is not amortisation, and inferred labels are strictly worse than recorded ones. The two
outside-chance pools close identically — `USC-PSI-Lab/humanoid-everyday` (8 949 eps) and
`Humanoid-Everyday-G1` (4 064 eps) are fully action-labelled upstream, and the earlier "licence
unresolved" worry resolves into an account-holder question, not a labelling one.

**What step 0 found that it was not asked for**, kept here so it is not lost with the closure:
**3 152 of those episodes are the 13 `unitreerobotics/G1_Dex3_*` sets, every one declaring
`action float32[28]`** — the exact 28-dim G1 + Dex3 vocabulary this task proposed to teach Cosmos
from scratch. Labelled, Apache-2.0, 647 MB away. That bears directly on PR-07 §1's standing
"402 success-only episodes of one task is not enough", and it is **conversion work on recorded
labels — route 1, not route 3b**. Not a free win: 28-dim Dex3 ≠ 43-dim AppleToPlate,
`convert_lerobot_g1.py` targets canonical 15 joints + 2 grippers and reads v2.1 where these sets
are v3.0, and the corpus carries no waist column. **The block order for these sets is arm-first —
`[0:14]` arm, `[14:28]` hand** (measured 2026-08-15; the hand-first figure belongs to
Humanoid-Everyday-G1, a different corpus). Tracked as **T-043**.

**No GPU hour was spent, and PR-10 was never written** — the pre-registration was gated behind a
*go* from step 0 that never came.

**The re-open condition, stated so it is a trigger and not a mood:** the only pool that would make
this task real is teleop recorded after M1/D2, and it does not exist yet. **Re-open the day teleop
produces video faster than it produces labels.**

%% mc-links: [[T-041]] [[T-040]] [[T-37]] [[T-39]] [[T-34]] [[T-043]] %%
