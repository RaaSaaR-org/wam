---
id: E-01
subproject: edge-wam
title: "Can the Edge policy run without language, or is 'no VLA' a re-training job?"
slug: can-the-edge-policy-run-without-language
status: todo
priority: 1
owner: ''
tags:
- edge
- cosmos3
- probe
- blocking
depends_on: []
blocks:
- E-05
- E-06
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. No GPU needed to answer the first half (read the config schema and the inference entrypoint); a smoke run answers the second. This is the premise of the whole sub-project, so it goes first."
---

# Can the Edge policy run without language?

## Description

The sub-project's premise is **image in, action out** — no language, no VLA. The released model does
not obviously support that.

`Cosmos3-Edge-Policy-DROID` is documented as generating action trajectories *"given language
instructions and visual observations"* [✓ model card]. Separately, the base `Cosmos3-Edge` is
documented as able to run **image-only** for image-to-video, without text [✓ model card]. Those two
facts are about different heads, and the question is whether the *policy* path inherits the second.

Three outcomes, and they cost wildly different amounts:

| outcome | what "no VLA" costs |
|---|---|
| the policy accepts an empty/constant instruction and behaves | a config flag — free |
| it accepts one but degrades badly | a post-training run with the text conditioning fixed or dropped |
| the text embedding is structurally required | an architecture change, and the premise needs revisiting |

**Do not guess this from the model card prose.** The house rule (`docs/discoverer.md` §1, the
`MODEL_ID` precedent in PR-07 §8) is that a load-bearing string comes from a primary source, not a
recollection. Here the primary source is the config schema and the inference entrypoint in the
Cosmos repo.

## Acceptance

1. The policy's input contract stated from **code**, not prose: which fields are required, what the
   text field's type and default are, and whether a null/empty value is accepted.
2. If an empty instruction is accepted — a smoke run on any available GPU producing an action chunk
   from a single image with no text, with the output shape recorded.
3. A one-paragraph verdict written into §Notes below naming which of the three outcomes holds, with
   the file and line the answer came from.
4. If outcome 2 or 3: **E-05's pre-registration must state it before any training**, because it
   changes what the experiment is.

## Not in scope

Any post-training run. This task reads and smoke-tests only.

## Notes / Report

*(empty — fill in when the task runs; record the verdict, the file:line evidence, and the date)*
