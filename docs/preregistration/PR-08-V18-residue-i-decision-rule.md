# PR-08 V18 — what would make residue (i) acceptable, decided before the census runs

**Rule `T40_RULE_V18`. Registered 2026-08-27. §3's outcomes are fixed BEFORE
`scripts/census_operating_point_episode.py` is run on anything, and the commit that lands this
document contains no census, no artifact and no verdict.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place. A gate rewritten after seeing its output is not a gate."*

## 1. The question, and why it is not the propagation blocker's

`GATE_QUALIFIED`'s own comment names **two** preconditions and only one of them is a blocker:

> It stays `False` until the remaining entry is closed **AND** somebody decides, on the record, what
> to do with the residue the two 2026-08-26 entries carry forward — in particular blocker 2's
> residue (i), a failure the discharged blocker predicted, occurring by a route it did not predict,
> on 92 frames.

`T40_RULE_V17` addresses the first. **V18 addresses the second and nothing else.** The two are
independent: V17 outcome N would not flip the flag on its own, and neither would V18.

**What residue (i) actually is.** Blocker 2 predicted that the `(0.10, 0.10)` retry would buy weak
detections and hide a degraded mask inside an inflated coverage number. Measured over the whole
corpus, **the retry fired zero times.** But the failure it predicted happened anyway, by the
*primary* 0.15 threshold: on 92 of 171 625 frames the operating point changed the outcome, 52 of
them in `episode_000094`, where `runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json` reports
*"a confident, well-formed mask of THE PLATE, at scores 0.155–0.259, ~31 000 px, plate overlap
0.985–0.992, zero IoU with the colour heuristic"* in the f109–f144 region.

**Why that matters to a gate rather than being a curiosity.** A mask of the wrong object does not
raise. It yields a centroid, a displacement and a percentile that all look like measurements. The
one thing standing between it and `GEOM_TOL` is the validity filter, which refuses a non-empty mask
containing essentially none of the warm saturated reference and returns all-False, whereupon both
harnesses drop and count the frame. **So the whole of residue (i) reduces to one measurable
question: does the filter actually catch them, or does some version of that failure survive into
the measured population?**

## 2. What is measured, and what the existing evidence cannot say

`scripts/census_operating_point_episode.py` runs the unmodified adapter over **every frame of
`episode_000094`**, on **both decode trees**, and reads the adapter's own counters around each call
so the three ways an all-False mask arrives — no detection, an empty mask from a real box, a mask
refused as the wrong object — are told apart. It writes per-frame mask area, warm-reference
fraction and validity IoU.

That closes two of the three limits the operating-point result placed on itself:

* **Limit 2, closed.** *"The exact 36 are not recorded."* The shard artifacts carry per-frame scores
  but no per-frame centroid-present flag. The census records refusals by frame index.
* **Limit 1, measured rather than caveated.** *"The phenomenon replicates; the numbers are not the
  same numbers."* Both decodes are run over the same episode, so the AV1/H.264 disagreement becomes
  a count.
* **Limit 3, NOT closed and not closable here.** *"A correlated observer."* Nothing in this census is
  a look at a picture. It measures what the filter did, not whether the filter was right, and §3's
  outcomes are written so that none of them claims otherwise.

## 3. The outcomes, fixed before the census runs

Read in order.

**Outcome U — UNDECIDABLE.** The census finds **zero** refusals on `episode_000094`, or finds them
outside that episode's documented low-score run of ~f101–f155. Then this census and the 16-shard
corpus pass are not describing the same phenomenon — the shard pass attributed all 36 of the
corpus's refusals to this episode — and neither decides anything. Residue (i) stays open, the flag
stays `False`, and the disagreement is the finding.

**Outcome E — IT ESCAPES.** Any frame of the episode carries a mask that is **not** refused and
whose area exceeds **three times the episode's own median non-refused mask area**. Then a
wrong-object mask survived the filter into the measured population, residue (i) is **not**
acceptable, `GATE_QUALIFIED` does not flip, and what is owed next is a bound on how often that
happens corpus-wide rather than another argument.

**Why area, and why 3×.** The filter's own criterion is validity IoU ≥ 0.10, so *by construction*
every surviving mask clears it and testing that again would be vacuous. The escape route the filter
cannot see is a mask that covers the fruit **and** something else — apple plus plate scores well
above 0.10 and is still not a mask of the apple. Area is the observable that separates them:
`runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json` records correct apple masks at 8 321–8 525 px
against an episode median warm area of 7 256 px, i.e. **1.1–1.2×**, while the plate masks it found
are **~31 000 px, about 4.3×**. **3× sits in that gap.** It is coined here, blind, and it is not
moved afterwards; both edges are named because a threshold whose rationale cannot name both has not
found a gap but chosen a number.

**Outcome C — CONTAINED.** Neither of the above. Every non-refused mask in the episode is of
apple-plausible area, and the refusals fall inside the documented low-score run. Then residue (i) is
**decided acceptable**, and the determination that records it must carry all four of the following
in its own text:

1. the refusal count and the frame indices, on both decodes, and their disagreement;
2. that this is **one episode of 402** — the census establishes containment *here* and bounds
   nothing corpus-wide, because the corpus pass recorded no per-frame flag to census against;
3. that limit 3 stands: **nobody has looked**, the area test is a proxy for a wrong-object mask and
   not an observation of one, and a wrong-object mask of apple-plausible area would pass every test
   in this document;
4. that the failure blocker 2 predicted **did occur**, by a route it did not predict, and that
   accepting a 0.054 % rate is a decision about a rate rather than a finding that nothing happened.

## 4. What V18 does NOT do

* **It does not flip `GATE_QUALIFIED`.** Outcome C is one of that flag's two preconditions; V17
  outcome N is the other, and neither implies the other.
* **It does not discharge blocker 1.** *"Nobody has looked at a mask"* is untouched: §3's outcomes
  are counters, and §3 outcome C item 3 says so in the determination itself.
* **It does not re-open a discharged blocker.** Blocker 2 is discharged and stays discharged; its
  residue is the thing being decided, which is what the residue list is for.
* **It licenses no clip and no training.** `T40_RULE_V1` §1 binds.

## 5. Determination

| | |
|---|---|
| rule | `T40_RULE_V18` |
| status | **REGISTERED 2026-08-27, before the census was run on any episode.** |
| supplements | `T40_RULE_V1`; the residue list at `scripts/estimators/apple_sam2.py:846` |
| amends | nothing |
| coins | the 3× area factor of §3, and the outcome table |
| generation licensed | **none** |
| training licensed | **none** |
| decided by | the project owner, 2026-08-27, by the instruction **"ja, löse alle blocker die du kannst. entscheide eigenständig. use subagents and workflows"**, given **before** the census existed and before this table was written. Recorded verbatim. |
| prepared by | a Claude Code session. §§1–4 were written before the census was run. |
| reversibility | the delegation was given blind. On the pattern V10 records for a blind delegation, **this determination is reversible on the owner's reading of it**, and §3 outcome C item 3 — that nobody has looked — is the item to read first. |
