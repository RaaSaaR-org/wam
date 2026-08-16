---
id: E-01
subproject: edge-wam
title: "Can the Edge policy run without language, or is 'no VLA' a re-training job?"
slug: can-the-edge-policy-run-without-language
status: review
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
updated: 2026-08-16
status_note: "Verdict delivered 2026-08-16: outcome 2, accepted_but_degrades — text reaches the action head; 'constant instruction' works, 'empty' is off-distribution. AC-2 (GPU smoke run) still open."
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

**2026-08-16 — verdict: outcome 2, `accepted_but_degrades`.** Answered from code, in
`diffusers 0.39.0` as installed (`.venv/.../pipelines/cosmos/pipeline_cosmos3_omni.py`,
`.../models/transformers/transformer_cosmos3.py`), not from the model card. Independently
re-verified by a second agent told to refute it: **not refuted, high confidence** — every code
citation reproduced, corrections were line-number offsets only.

**Outcome 3 is ruled out.** `prompt=""` is accepted: `check_inputs` only type-checks it
(`pipeline_cosmos3_omni.py:963-966`; `None` raises, `""` passes). In action mode the string
becomes the `description` field of the structured action-JSON caption and an empty description
is emitted verbatim — confirmed by executing `_build_action_json_prompt('')`. A probe on the
real `Cosmos3OmniTransformer` (CPU, tiny random init, `action_gen=True`) ran with `und_len=0`
— zero text tokens — and returned a well-formed action prediction. Nothing crashes.

**Outcome 1 is ruled out too, on two counts.**
1. *Structurally, text is never absent from the plumbing.* `text_tokenizer` is a required
   pipeline component (`:366`, not in `_optional_components`); `input_ids`/`text_indexes`/
   `und_len` are required positionals of `forward` with no `None` branch
   (`transformer_cosmos3.py:554-559`); the chat template plus two special tokens guarantee
   `und_len >= 3` on every call even with an empty prompt (`:1128-1146`). Every generation
   token — vision **and action** — cross-attends to the text keys/values in every layer
   (`transformer_cosmos3.py:82-94, 683-692`). A probe holding vision, action latents, timestep
   and seed fixed and changing **only** the text token ids moved the predicted action by
   `max|Δ| = 0.787`: **text reaches the action head, not just the video branch.**
2. *Behaviourally, NVIDIA's own numbers price it.* On RoboLab the Edge policy scores
   **15.4 / 22.9 / 28.8 %** overall success for vague / default / specific instructions — a
   ~47 % relative collapse from merely making the sentence less specific. All 120 RoboLab
   tasks are language-conditioned; there is no no-language condition, so the empty case is
   **unmeasured by the vendor**.

**The refinement that matters for this sub-project: "empty" and "constant" are different
bets.** With `prompt=""` the conditional stream is a metadata-only caption and the
unconditional stream is the null string, so CFG amplifies only metadata and no task semantics
survive — the policy falls back to its unconditional prior. A single **constant, task-correct**
instruction is in-distribution and should land near the "Default" column at ~zero cost; a
**truly empty** one is off-distribution and should be expected at or below the 15.4 % "Vague"
number. Edge-WAM's MVP is one pick-and-place task, so the constant-instruction route is
available and cheap — but it is a language-conditioned policy with a frozen sentence, which is
an honest description the README should adopt.

**Consequence for the premise.** "Image in, action out" survives as an *interface* — the caller
supplies no language — but "no VLA" does not survive as an *architecture* claim: the text tower
and tokenizer stay resident on the robot. `Cosmos3EdgeAdapter.condition_text` therefore models
text as **constant, not absent** (`src/wam/backbones/cosmos3_edge.py`).

**Acceptance: 1 ✅ (input contract from code), 3 ✅ (verdict + file:line), 4 ✅ (outcome 2 → E-05
must pre-register it). AC-2 ❌ — no GPU smoke run; weights were deliberately not downloaded.**
That is the one thing outstanding, and it is what would separate outcome 1 from outcome 2
empirically: one DROID episode through the real checkpoint at `prompt=""` vs the true
instruction, comparing emitted action chunks. Now feasible locally (see E-03).

**Open [?] carried forward:** the served path NVIDIA supports is cosmos-framework / vLLM-Omni,
not the diffusers pipeline everything above was read from; whether a zero-length `und` segment
survives a fused/flash attention backend is untested (the probe used the default backend); and
whether the policy was post-trained with the same 10 % text-dropout as pre-training is not
stated — if it was not, the unconditional prior may be weaker than assumed.
