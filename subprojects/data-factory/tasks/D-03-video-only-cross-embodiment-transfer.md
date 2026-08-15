---
id: D-03
subproject: data-factory
title: "Video-only cross-embodiment transfer — the route we had no task for"
slug: video-only-cross-embodiment-transfer
status: todo
priority: 2
owner: ''
tags:
- data
- dreamzero
- research
- new-route
depends_on:
- D-01
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. New on 2026-08-15 from the DreamZero paper. Reading and scoping only — this task must not turn into a training run without a pre-registration."
---

# Video-only cross-embodiment transfer

## Description

DreamZero (NVIDIA, arXiv 2602.15922, Apache-2.0 code) reports two results that this project has no
task for and no position on:

- **Cross-embodiment transfer from video-only demonstrations** — from other robots *or humans* —
  yielding **>42 % relative improvement on unseen tasks from 10–20 minutes** of data.
- **Few-shot embodiment adaptation** to a new embodiment with **30 minutes of play data**, while
  retaining zero-shot generalization.

**Why this is not just T-042 again.** T-042 is inverse dynamics: infer action labels from real
unlabelled footage, then train on the inferred labels — and it is bounded precisely because those
labels are inferred. The DreamZero route infers nothing. Video enters as video, into the world-model
objective, and never becomes an action label. If it holds, it is a way to use footage that T-042's
bounds would otherwise disqualify.

**Why it is filed under the data factory.** It is about what data can be used and how, not about the
on-robot policy. But it points at `edge-wam/` — the beneficiary would be E-06's post-training.

**Two cautions, stated up front so they are not discovered late.** First, DreamZero's numbers come
from a 14B backbone with large-scale pretraining and 2× H100/GB200 inference — the regime is not
ours, and a result that depends on scale may not survive the trip down to 4B. Second, our own clean
ablation says our world branch cost **108 pp**. That measurement bounds our old implementation on
our corpus, not the architecture class — but a route whose entire mechanism is the world-model
objective must engage with it rather than route around it.

## Acceptance

1. The paper read properly — not the abstract — with the ablations that isolate the video-only
   contribution extracted, or their absence recorded.
2. A statement of what the route would need *here*: which footage, how much, and against which
   backbone.

   **Read T-042's closure first — it cuts this task down.** Step 0 counted the unlabelled real G1
   footage on 2026-08-15 and the answer was **zero**: the clips only looked unlabelled because two
   fetch scripts never pulled `data/**`. So the obvious input for a video-only route does not exist,
   because the video we hold *has labels*. That leaves two honest cases: footage from *other*
   embodiments or humans (DreamZero's actual claim — nothing of ours), and teleop recorded after
   M1/D2, which does not exist yet. **If neither, this task's correct outcome is "not yet", and it
   should say so rather than manufacture a use.**
3. An explicit position on the 108 pp tension, written down.
4. A go/no-go recommendation with a GPU-hour estimate. **If go, it needs a pre-registration first**
   — no run starts from this task.

## Notes / Report

*(empty — fill in when the task runs)*
