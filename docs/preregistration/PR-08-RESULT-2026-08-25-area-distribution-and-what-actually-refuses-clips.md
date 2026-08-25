# PR-08 §6 G0c — the corpus area distribution, and the discovery that the area bound is not what refuses the clips

**Measured 2026-08-25 over the whole of AppleToPlate: 171 625 frames, 402 of 402 episodes, no
`--limit`, `--stride 1`, `measurement_qualified: true`. Job 190413 (16 shards) + merge job 190460.**

**This document decides no bound.** `T40_RULE_V13` §3 keeps `max_frame_fraction` a human decision
with a written rationale, and `configs/transfer25/pr08_robot_mask_area.json` still holds `null`.
What this does is supply every item of V13 §3.2's required rationale that a measurement can supply,
and report a second finding that was not the question and matters more than the answer.

---

## 0. The headline, because it inverts the priority

A single frame that fails G0c raises `CompositeError`, and `restyle_transfer25.py:720` fails **the
whole unit** — the clip is quarantined as `vision.uncomposited.mp4` after its GPU cost is paid. So
the number that governs yield is not the per-frame rate; it is **how many episodes contain at least
one failing frame.**

| rule | episodes of 402 with ≥1 failing frame | |
|---|---:|---:|
| a frame above the candidate area bound (0.6409) | **175** | 43.5 % |
| **at least one EMPTY robot mask** | **366** | **91.0 %** |
| **either** | **385** | **95.8 %** |

**The area bound is not the binding constraint. The empty-mask rule is.** Even with a perfect
bound, 91 % of source episodes are refused before the area check is reached — and every one of
their 25 restyles with them.

95.8 % sits beside the pilot's observed 128-of-129 (99.2 %) G0c refusal in
`runs/pr08-g0c-refusal/G0C_REFUSAL.json`. **That is a mechanism for a rate this project has had
since 2026-08-24 and could not account for**, now measured corpus-wide instead of on 129 clips.
Recorded as consistent-with: the pilot ran on a different subset and the two numbers are not the
same measurement.

---

## 1. What was measured, and against what

| | |
|---|---|
| corpus | `pr08-apple-640x480-h264-lossless`, 402 episodes, 171 625 frames |
| `git_commit` | `8b710d0119b6…`, identical on all 16 shards |
| `source_manifest_sha256` | `a988dd60db6ba8ab…`, identical on all 16 shards |
| prompt | `"robot arm. robotic hand. robotic gripper."` |
| agreement | all 16 shards agree on commit, prompt, manifest, estimator, `limit: null`, `stride: 1` |
| coverage | 402 episode ids, **402 unique** — no episode pooled twice, none missing |

The analysis was run by `scripts/analyze_area_gap.py`, **committed at `060485b` while 12 of 16
shards had landed and no merged artifact existed**, for the reason `T40_RULE_V13` §1 gives about
itself: an analysis written after the numbers are visible cannot be shown not to have been fitted
to them. It was run here **unmodified**. The merged artifact carries only the pooled five numbers,
so the per-frame fractions V13 §3.1 step 1 requires were concatenated from the 16 shards; the
concatenation reproduces the merged artifact's `frames` and `episodes` exactly (asserted, not
eyeballed).

## 2. The distribution

Over all 171 625 frames — the merge's own block, zeros included:

| min | median | p95 | p99 | max |
|---|---|---|---|---|
| 0.0 | 0.06226 | 0.23932 | 0.30888 | 0.97958 |

**57 835 frames (33.70 %) carry no robot mask at all** and are recorded as exactly `0.0`. Those are
neither "the robot is small" nor a bound's business — they are the subject of `T40_RULE_V12`, and
an area fraction of 0.0 cannot exceed any bound in either direction. The separation analysis
therefore runs over the 113 790 non-empty frames, with the empty count reported beside it:

| min | median | p95 | p99 | max |
|---|---|---|---|---|
| 0.000101 | 0.10361 | 0.25261 | **0.81051** | 0.97958 |

The jump from p95 0.253 to p99 0.811 is the whole finding, and the histogram says why.

### 2.1 It is bimodal, with a genuinely empty valley

| band | frames | share of non-empty |
|---|---:|---:|
| **bulk** ≤ 0.36 | **112 361** | 98.74 % |
| **valley** 0.36 – 0.76 | **50** | 0.04 % |
| **second mode** ≥ 0.76 | **1 379** | 1.21 % |

The second mode is not a tail thinning out; it is a **mode**, peaked at 0.80–0.84 (716 frames in
that bin alone). Between the two lies a band containing 50 frames out of 113 790.

### 2.2 The widest empty band, with both edges named

`T40_RULE_V13` §3.1 step 3: *"the bound goes strictly inside it, and `bound_rationale` records
**both edges** — the largest bulk fraction below and the smallest tail fraction above."*

| | |
|---|---|
| largest value below the band | **0.601546** |
| smallest value above the band | **0.680277** |
| width | **0.078731** |
| frames above 0.601546 | **1 385** (0.81 % of all frames, 1.22 % of non-empty) |

Both edges are individual frames, and that is a real limit on the claim: this is the widest gap
between two adjacent order statistics above the median, not a boundary between two clouds with
nothing near it. Below 0.60 the valley is thin rather than empty — 44 frames are scattered over
0.36–0.60.

## 3. Where the tail comes from — V13 §3.1 step 2, and it does not come back clean

> *"The per-episode block names which episodes the tail comes from, which is the check that the tail
> is a **failure mode** and not simply the grasp frames of every episode."*

At a cut of 0.601546:

| | |
|---|---|
| episodes contributing | **175 of 402** (43.5 %) |
| share from the largest contributor | **5.8 %** (`episode_000338`, 80 frames) |
| median contributing episode | **0.8 %** of its own frames |
| runs of consecutive frames | **673**, totalling 1 385 frames |
| run length | median **1**, mean 2.1, max 19 — **60 % are single isolated frames** |
| runs ≥ 5 frames | 55, covering 432 frames |

**This is the answer V13 warns about, not the one it hopes for.** The tail is not three broken
episodes; it is spread thinly across 43 % of the corpus, and most of it is a *single frame* whose
mask covers more than 60 % of the image with normal frames either side.

**The counter-argument, stated because the decision turns on it.** A mask covering 0.80 of a
640×480 frame is 245 000 px. That cannot be a robot arm, and it cannot be a grasp-pose artifact of
one — it is the masker grounding on the table or the scene, which is precisely the failure G0c's
area half exists to catch. So the tail *is* a failure mode by its magnitude even though it is not
concentrated by episode. **Whether that argument is sufficient is the judgement V13 reserves for a
person, and this document does not make it.**

## 4. V13 §3.2's checklist, filled as far as a measurement can fill it

| required content | status |
|---|---|
| the two edges of the gap, as numbers | **0.601546 / 0.680277** (§2.2) |
| frames and EPISODES above the bound, absolute and fractional | **1 385 frames (0.81 %), 175 episodes (43.5 %)** (§3) |
| whether those frames were **looked at**, and what they were | **NO. Nobody has looked at a single one.** |
| the commit and `source_manifest_sha256` measured over | `8b710d0119b6…` / `a988dd60db6ba8ab…` (§1) |
| that the bound has never been validated against a known-bad mask | **still true**, and unchanged by this document |

Two of five are open, and one of those two is the load-bearing one. **A bound committed today would
rest on a distribution nobody has inspected a single frame of**, which is the same objection
`GATE_QUALIFICATION_BLOCKERS`' first entry makes about the apple masks.

## 5. What this session recommends, without doing it

**If** a bound is placed under V13 §3.1, the defensible value is **strictly inside the measured
band and derived from both its edges** — the midpoint **0.640912** — never a round number (§3.4
forbids 0.5 for being 0.5) and never the observed maximum (§3.4 forbids it for being unfireable).

**But the recommendation attached to that is not to prioritise it.** §0 is the reason: committing a
bound moves `max_frame_fraction` from `null` to a number and unblocks `TIMING=1` procedurally, and
it changes the yield from ~4 % to ~4 %, because 91 % of episodes die on the empty-mask rule first.
**The bound is a procedural unblock, not an economic one.**

## 6. What actually stands between this project and a corpus

The composite's own refusal message names it:

> *"If the robot is genuinely absent from this frame the SOURCE corpus is not what PR-08 §3
> describes; if it is present, the prompt or the detector thresholds do not find it, and that has to
> be fixed before generation rather than skipped per frame."*

A median episode has **no robot mask on 36.4 % of its frames** (range 0.0 % – 61.5 %). Three
independent measurements now say the same thing about why:

- every one of the 237 empty-mask frames in the blind draw was `no_boxes_above_threshold` — zero
  from SAM2, zero from the V9 object filter (`runs/pr08-blind-adjudication/`);
- the detector's best score on those frames is **noise**: median 0.1277 against a 0.15 threshold,
  and statistically indistinguishable from frames with no edge object at all (Mann-Whitney
  **p = 0.42**, `PR-08-RESULT-2026-08-25-detector-noise-floor.md`);
- 41 of 52 frame-edge marks in the blind draw carry an object that moves over ±8 frames, on frames
  where the masker returned nothing — and the project owner's reading on 2026-08-25 was that these
  *"could well be Dex3 fingertips, hard to make out, only the tips"*.

Together those are consistent with: **on about a third of frames only the fingertips are in view,
`"robot arm. robotic hand. robotic gripper."` does not fire on fingertips, and G0c then refuses the
clip.** That is a hypothesis about a cause, not a measurement of one, and it is exactly the
question `T40_RULE_V12` was drafted to settle. **V12 is unsigned, its §3.2 precondition is
unavailable (`PR-08-RESULT-2026-08-25-v12-preconditions.md`), and the blind (a)/(b) adjudication
came back inconclusive.**

`PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §4 argued V12 was "the yield question, worth
resolving before spending a budget" on a 129-clip pilot. **This measurement upgrades that from an
argument to a corpus-wide number: 91.0 % of 402 episodes.**

---

## 7. Provenance

| | |
|---|---|
| kind | measurement report. **Registers no rule, decides no bound** |
| date | 2026-08-25 |
| jobs | 190413 (16 shards, GPU), 190460 (merge, free CPU QoS) |
| analysed by | `scripts/analyze_area_gap.py`, committed at `060485b` **before the distribution existed**, run unmodified |
| artifacts | `runs/pr08-robot-mask-area/pr08_robot_mask_area.MEASURED.json`, `runs/pr08-robot-mask-area/AREA_GAP_ANALYSIS.json` |
| `max_frame_fraction` | **still `null`**, in the tracked config and in the measured artifact |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
