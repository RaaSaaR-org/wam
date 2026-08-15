---
id: D-01
subproject: data-factory
title: "Charter — what the data factory may and may not produce"
slug: charter-what-the-data-factory-may-produce
status: todo
priority: 1
owner: ''
tags:
- data
- charter
- prereg
depends_on: []
blocks:
- D-02
- D-03
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. Writing task, no compute. Exists because 'generate training data' is one careless sentence away from a question this project already closed."
---

# Charter — what the data factory may and may not produce

## Description

This sub-project's one-line description — *"use Cosmos to make more training data"* — is, read
loosely, exactly the thing `docs/handoff.md` §3 forbids:

> **Generated video is not training data, and nothing infers actions from it.**

The sub-project is nevertheless legitimate, because one route keeps real labels. The charter's job
is to write that boundary down precisely enough that a future session cannot drift across it by
paraphrase.

| | route | what happens to the action labels | status |
|---|---|---|---|
| ✅ | **restyle a real episode** — Transfer2.5, T-040 | trajectory unchanged, real labels survive | allowed |
| ❌ | synthesize a new episode, then infer its actions | invented | forbidden |
| ⚠️ | inverse dynamics on real *unlabelled* footage — T-042 | inferred from real pixels | open, bounded |
| ⚠️ | video-only cross-embodiment transfer — D-03 | **no action labels used at all** | new, see D-03 |

The fourth row is new as of 2026-08-15 and does not fit the existing taxonomy: DreamZero reports
>42 % relative improvement on unseen tasks from 10–20 minutes of *video-only* human or robot
demonstrations. Nothing is inferred and nothing is labelled — the video enters as video. The charter
has to say whether that counts as "generated video used as training data" (it is not generated at
all) and under what conditions it is permitted.

## Acceptance

1. `docs/data-factory-charter.md` written, with the table above as executable-in-review prose: for
   any proposal, it must be decidable in one reading which row it is.
2. The evidence for each row cited, not asserted — PR-06's 39 % for dreams, `docs/sim.md` / T-25 for
   sim frames, `docs/action-labels.md` §3b for the inverse-dynamics bounds.
3. A statement of what would *reopen* the forbidden row, so it is falsifiable rather than dogma.
4. Linked from `docs/handoff.md` §3 so the closed decision and its one legitimate route are one
   click apart.

## Notes / Report

*(empty — fill in when the task runs)*
