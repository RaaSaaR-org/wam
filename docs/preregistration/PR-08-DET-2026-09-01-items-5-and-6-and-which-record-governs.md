# PR-08 DET — §8 items 5 and 6 determined, and which record governs a §8 item

**SIGNED 2026-09-01 by the project owner.** This is a determination, not a draft. It answers the
four questions `PR-08-DET-DRAFT-2026-08-28-items-5-and-6-are-not-a-silence-they-are-a-disagreement.md`
§6 recorded as owed to the owner and to nobody else.

**It closes §8 items 5 and 6. It closes no other item, lifts no gate, and licenses no clip.**
`T40_RULE_V1` §1 is untouched: §8 is conjunctive and items 3 and 4 remain open at signing.

**The draft is not revised and not superseded.** `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place"* — so this sits beside the draft rather than editing it. Everything below cites the
draft's measurements; none of them is restated as if newly found, and none is contradicted.

---

## 1. The prior question, answered first because it governs the other three

### D-1. Which record governs a §8 item when the task ledger and the registered documents disagree?

```
determination:  THE REGISTERED DOCUMENTS GOVERN. A task-ledger status_note does not
                close a §8 item, and never did.

                .mc/tasks/ is the operational index over milestones, which is the role
                CLAUDE.md gives it. Nothing anywhere confers on a status_note the power
                to discharge a gate item. Where the ledger and a registered document
                disagree about whether an item is closed, the document is the record and
                the ledger entry is an observation about it.
```

**This is the strict direction, not the convenient one, and that is why it is chosen.** On the
ledger's own reading only items 3 and 4 were ever open; on this one, items 5 and 6 were open too and
had to be determined in writing — which is what §§2–3 below then do. The rule costs this session
work and buys the project an auditable boundary: **a gate closes when a document says so, over a
signature.**

It applies to all seven items, prospectively and retrospectively. It does not impugn the two ledger
entries (`T-39`'s and `T-040`'s): both recorded true facts about the code and the commits. They were
simply never the instrument that closes an item, and future sessions must not read them as one.

## 2. Item 6 — the style partition

### D-2. Does a §5 pre-commitment survive in-place amendment of the parts it does not protect?

```
determination:  YES, ON THIS RECORD, AND §8 ITEM 6 IS CLOSED.

                The property §5 exists to protect is provably untouched: TRAIN_STYLES
                and EVAL_STYLES are identical id-for-id and prompt-for-prompt between the
                pre-commitment 11af1d0 (2026-08-16) and HEAD -- 10 and 5 ids, same four
                axis slugs, seed_schedule and volume identical. The 98d402a amendment
                (2026-08-22) changed identity_style's prompt by two words and closed one
                blocking_todo. identity_style is in neither set by construction, and
                check_disjoint asserts it.
```

**The ground.** §5's requirement is one property, stated in its own words: the pool is split into
disjoint `TRAIN_STYLES` and `EVAL_STYLES`, committed before the first clip, *"so the evaluation
domain cannot be chosen after seeing which restyles came out well."* Zero clips have ever been
generated, and the eval domain has not moved since it was committed. The thing the rule exists to
prevent did not happen and can no longer happen.

**The counter-reading is recorded, because it is available on the text and was not refuted.** The
draft's §2.1 states it: a pre-commitment is the file, not the subset somebody later argues was
load-bearing, and `styles.toml`'s own in-file amendment justification covers an *earlier* event
(`4da3875d…` → `8d8565ff…`, 2026-08-15, before tracking) and does not reach `8d8565ff…` →
`9334fd01…` at `98d402a`. That is correct and stands. This determination accepts the narrower
reading — the pre-commitment binds the property, not the byte — and records that it is a choice.

**Two things this determination refuses to do.** It does not amend `styles.toml`,
`pr08_style_partition.json`, or their sidecars, and in particular it does not touch `styles.toml:2`'s
self-assertion *"Closes PR-08 §8 item 6"* — editing a pre-committed file to make a determination
easier is the very defect §2.1 identifies. And it does not retroactively bless in-place amendment as
a practice: **the next amendment to a pre-committed file is a new rule version, not a commit.**

**The one-day date discrepancy is noted and is not load-bearing** (draft §2.2).
`pr08_style_partition.json` carries `committed = '2026-08-15'`; git dates the commit
2026-08-16 16:05:49 +0200. The order that item 6 actually turns on is *partition before first clip*,
and that order is established by there having been no clip at all, not by the field. The field is
wrong by a day and should be left as it is rather than corrected, for the reason in the paragraph
above.

## 3. Item 5 — the ground-truth route

### D-3. Does `T40_RULE_V14`'s licence extend to §8 item 5?

```
determination:  NOT BY ITSELF -- V14's scoping sentence is honoured, not overridden.
                THIS DETERMINATION EXTENDS IT, on its own signature, to §8 item 5 and
                to nothing else.
```

`T40_RULE_V14:35`, verbatim: *"The substitution is licensed for that measurement and for no other."*
That sentence is correct and remains in force as written. A determination that claimed V14 already
covered item 5 would be contradicting the rule it cited. So the extension is made **here**, visibly,
and carries the same discipline it inherits:

> **The substitution is licensed for §8 item 5 and for no other item.** It does not reach items 1, 2,
> 3, 4, 6 or 7, it does not make MuJoCo the ground-truth route for any measurement not already
> licensed by V14, and it does not close the Isaac route, which `T40_RULE_V5` keeps available
> unchanged in every flag and refusal.

### D-4. Is item 5 closed by the MuJoCo route, or routed around by it?

```
determination:  CLOSED. V5's narrowing of §4 to its gated half is accepted as V1's
                meaning, and §8 item 5 is CLOSED as of 2026-09-01.
```

**The ground, and the asymmetry that decides it.** Item 5's own clause is *"Blocks §4 entirely"*, and
§4 step 3 names *"absolute depth error, **and** object-centroid displacement in pixels"*. Depth is
the one quantity the two routes genuinely disagree about: MuJoCo returns distance to the image plane,
`distance_to_camera` is euclidean ray length, 1.41× apart at 45°, and `mujoco_binding.py:58-66`
refuses to convert one into the other. That refusal is right and is not disturbed here.

But **the gated number is defined on segmentation alone.** §4 step 4 takes the 95th percentile of
object-*centroid displacement*; `run_g0_gates.py` reads `est_drift_p95_px` and never a depth field,
which reproduces. So the strict reading holds item 5 open on account of a quantity the gate does not
read — and holds it open against a route that **cannot be walked**: no Isaac capture has ever been
taken (`grep -rl '"ground_truth_route": *"isaac"' runs/` returns nothing), and
`tests/test_isaac_binding.py:1-6` says the binding *"CANNOT be executed here"*, its tests running
against a stub whose payload shapes the file itself calls *"two guesses."*

**That is the trade, stated plainly: this determination accepts a narrowing argued after V1 was
written, in exchange for not blocking a conjunctive gate indefinitely on a measurement that no
available instrument can produce and that the gate would not read if it existed.** The contrary
reading — that *entirely* means entirely and item 5 is routed around rather than closed — is
available on the text, was not refuted, and is recorded here as the alternative that was passed over.

**What this does not do:** it does not re-open V5, does not adopt V5's §2 argument beyond the single
use made of it above, and does not claim depth is unimportant — only that it is not what §8 item 5
gates through §4 step 4.

## 4. §8 after this determination

| item | status | by |
|---|---|---|
| 1 | closed | prior record |
| 2 | closed | prior record |
| 3 | **open at signing** | ceiling derived, spend authorised separately — see `PR-08-DET-2026-09-01-the-spend-authorised.md` |
| 4 | **OPEN** | needs the `pr08-geom-tol-v2` merge, then the carry under `T40_RULE_V21`, then a positive `gate_margin_px` |
| 5 | **closed 2026-09-01** | D-3 + D-4 above |
| 6 | **closed 2026-09-01** | D-2 above |
| 7 | closed | prior record |

**Item 4 is the one that can still refuse.** `gate_margin_px = geom_tol_px − est_drift_p95_px` and
`est_drift_p95_px` is fixed blind at `0.36010037281174667` by `T40_RULE_V21`, signed before the merge.
If the re-measured `GEOM_TOL` does not clear it, PR-08 §6 governs — *the estimator is not good enough
and generation does not start* — and no determination on this page changes that.

## 5. Provenance

| | |
|---|---|
| determination | `PR-08-DET-2026-09-01` — D-1 through D-4 |
| status | **SIGNED 2026-09-01.** In force |
| decided by | the project owner, 2026-09-01, on the instruction **"klingt gut, lass das so machen!"**, given in direct answer to *"was würdest du empfehlen?"* after the three available readings and their consequences had been put in front of them in writing. The owner chose among named alternatives; the recommendation was the session's. Prepared by a Claude Code session, which `T40_RULE_V13` §5 permits and no more. |
| opening instruction | **"was, lass nun alles lösen, damit wir den datensatz (oder zumindest mal 1 video + action)."** — recorded verbatim because it is what the owner was trying to achieve |
| answers | `PR-08-DET-DRAFT-2026-08-28-…` §6, decisions 1–4 |
| amends | nothing. It fills four holes that document registered as holes. |
| extends | `T40_RULE_V14`, to §8 item 5 **and to no other item** (D-3) |
| closes | §8 items 5 and 6 |
| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style, no seed, no ceiling, and no committed artifact** |
| generation licensed | **no** |
| training licensed | **no** |
