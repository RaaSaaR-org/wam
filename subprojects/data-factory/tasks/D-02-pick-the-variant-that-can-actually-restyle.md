---
id: D-02
subproject: data-factory
title: "Pick the variant that can actually restyle — Edge cannot"
slug: pick-the-variant-that-can-actually-restyle
status: todo
priority: 2
owner: ''
tags:
- data
- cosmos3
- probe
depends_on:
- D-01
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. Reading task. Small, but it settles which model the augmentation work targets, and it is the cleanest evidence that the two sub-projects are genuinely different jobs."
---

# Pick the variant that can actually restyle

## Description

**`Cosmos3-Edge` does not support video-to-video transfer** [✓ repo]. Restyling a real episode —
the one route D-01 permits — is a video-to-video job. So the augmentation work can only run on
**Super** (64B) or **Nano** (16B), and can never migrate to the edge model.

That is worth stating explicitly for two reasons. It is the cleanest single argument that the split
into two sub-projects is real rather than cosmetic: the two halves cannot share a checkpoint even in
principle. And it removes a tempting shortcut — "just use the small model for everything" — before
someone spends a week discovering it.

What remains open is Super vs Nano, and it is not obvious. T-041 already established that the
Super-derived export is a **merged full model, 121 GB across 27 shards**, which runs nowhere but
Discoverer+ and cost ~59 of PR-09's 122 GPU-h ceiling. Nano at 16B is cheaper per clip and is where
the twelve action cookbooks live. If Nano's restyling is good enough, the corpus gets bigger per
GPU-hour — and GPU-hours are the binding constraint, with 4 875 left and no extensions.

## Acceptance

1. Video-to-video support confirmed per variant from the repo, with file:line — not from the blog.
2. Super vs Nano decided on a stated basis: quality evidence if any exists, cost per clip in
   GPU-hours, and what T-041's measurements already tell us.
3. Decision recorded with its reason, so it is not re-litigated per session.
4. If the honest answer is "we cannot tell without generating clips" — say so, and make it a
   pre-registered comparison rather than a preference.

## Notes / Report

*(empty — fill in when the task runs)*
