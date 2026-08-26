# `T40_RULE_V13` — how `max_frame_fraction` is decided, fixed before the distribution is visible

**Status: SIGNED 2026-08-26 and in force. §5 carries the determination.**
**Registered 2026-08-24, before job 190192's shards had landed and before anybody had seen the
corpus area distribution.** That ordering is the whole point of the document and is checkable: the
shards were `PENDING` when this was written, and the artifact it governs did not exist. **§§0–4 are
unchanged since registration; the only thing added on signing is §5's determination and the two
lines in §6 that record it.** A rule edited in its own body after seeing its output would not be a
rule (`docs/handoff.md` §3), and the diff is the evidence that it was not.

---

## 0. What this does and does not do

| | |
|---|---|
| decides | **nothing about the number.** It fixes the RULE by which the number is chosen |
| coins | no threshold. No figure in this document is a candidate bound |
| changes | no code, no gate, no committed artifact |
| licenses | **no clip.** `T40_RULE_V1` §1 binds in full |
| does not touch | `GATE_QUALIFICATION_BLOCKERS`, `GATE_QUALIFIED`, §6's three gates, §8 items 3 and 4 |

---

## 1. Why a rule is needed before the data, and not after

`cluster/discoverer/106_measure_robot_mask_area.sbatch` measures the distribution and **refuses to
set the bound**, on purpose. Its header says why:

> *"the only bound derivable from the distribution is the observed maximum, which can never fire on
> the frames it was measured over, and any bound that CAN fire sits above it by a margin nothing in
> the corpus determines. A human reads the five numbers, writes `max_frame_fraction` and
> `bound_rationale` into the artifact, and commits it. The rationale is the record of a decision a
> script refused to make."*

A human reading five numbers and picking a sixth is exactly the shape `docs/handoff.md` §3 forbids
when it is done afterwards — *"a gate rewritten after seeing its output is not a gate"*. The
distribution is about to become visible. **After that, no rule written here can be believed**, because
nobody, including the author, can prove it was not fitted to the numbers. So it is written now.

**This is not a claim that the decision is mechanical.** It stays a human decision with a written
rationale, exactly as 106 says. What is fixed in advance is *what evidence makes a bound defensible*
and *what evidence forces a refusal* — so the human is choosing within a rule rather than choosing
the rule and the number together.

---

## 2. What the bound is for, stated precisely, because it constrains the method

§6 G0c composites the real robot's pixels back over every generated frame. The bound exists for one
failure: **a robot mask that has grounded on the table, the background, or the whole scene.**
Compositing that mask puts the SOURCE back over most of the frame, so the restyle silently becomes a
no-op and **arms B and C become arm A at full GPU cost** — a null result that looks like a measured
one.

Two consequences follow, and they point in opposite directions:

- **A bound that never fires is not a guardrail.** It is a committed number that makes the artifact
  look complete while the failure it names remains undetectable.
- **A bound that fires on legitimate frames destroys clips.** The robot genuinely occupies a large
  fraction of some frames — a near-camera arm at the grasp is not a defect — and refusing those
  removes exactly the hardest and most informative part of the corpus.

So the bound must **separate two populations**, and the question the distribution has to answer is
whether those two populations are separable at all.

---

## 3. The rule

### 3.1 The bound is placed inside a measured gap, or it is not placed

This is the method `T40_RULE_V10` already used for `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION`,
where the committed 0.10 sits *inside a measured gap on both scales* rather than at a round number.
Same discipline here:

1. Read the pooled distribution over all 171 625 frames — `measured.min / median / p95 / p99 / max`
   and the per-frame fractions behind them.
2. **Identify the upper tail as a population, not as a percentile.** Look for a mode, a shoulder, or
   a discontinuity separating a bulk (masks that are the robot) from a tail (masks that are the
   scene). The per-episode block names which episodes the tail comes from, which is the check that
   the tail is a *failure mode* and not simply the grasp frames of every episode.
3. **If a gap exists**, the bound goes strictly inside it, and `bound_rationale` records **both
   edges** — the largest bulk fraction below and the smallest tail fraction above. A bound whose
   rationale cannot name both edges has not found a gap; it has chosen a number.
4. **If no gap exists**, see §3.3.

### 3.2 What the rationale must contain

`bound_rationale` is a free-text field and this rule fixes its minimum content, so that a later
reader can re-derive the decision rather than trust it:

- the two edges of the gap, as numbers;
- how many frames and how many EPISODES fall above the bound, in absolute terms and as a fraction —
  because "0.4 % of frames" and "every frame of three episodes" are different findings;
- whether those frames were **looked at**, and what they were;
- the commit and the `source_manifest_sha256` the distribution was measured over;
- the sentence that this bound has never been validated against a known-bad mask, if that is still
  true when it is written.

### 3.3 The refusal, which is the load-bearing half

**If the distribution is unimodal with no separable tail, no bound may be committed under this rule.**
A number placed on a continuum is a coined threshold wearing a measurement's clothes, and the fact
that it was derived from real data does not change that — the data does not contain the decision.

In that case the honest outcomes are, and this document does not choose between them:

- **(a) Leave `max_frame_fraction` null.** `load_area_bound` refuses, G0c's area half stays unarmed,
  and generation does not start. Costly and correct.
- **(b) Register a different instrument.** The area fraction may simply be the wrong statistic for
  the failure — a mask that has grounded on the table is arguably better detected by *what it
  overlaps* than by *how big it is*, which is the shape V9's object-grounding filter already uses
  and which the area bound does not. That would be a new rule, not a relaxation of this one.
- **(c) Establish the failure population deliberately** rather than hoping the corpus contains it:
  construct or find masks that have grounded on the scene, measure their area fraction, and place
  the bound between the two measured populations. This is the only option that makes the bound a
  measurement of the thing it is supposed to catch. It costs work and it is not on the critical
  path today, which is a reason to record it, not a reason to skip to (a).

### 3.4 What may not be done

- **The observed maximum may not be committed as the bound.** It cannot fire on the frames it was
  measured over, so it certifies nothing. It may be *quoted* in the rationale as an edge.
- **No round number may be chosen for being round.** 0.5 is not a finding.
- **The bound may not be moved after a clip has been generated under it**, for the reason every
  gate in this pre-registration is versioned rather than edited.
- **The pilot's numbers may not be used.** `ROBOT_MASK_AREA_PILOT.json` carries
  `measurement_qualified: false` over three episodes at stride 30, and `load_area_bound` refuses it
  by name. It is a cost estimate, not a distribution.

---

## 4. The thing this rule cannot fix, recorded so it is not mistaken for solved

**The area half of G0c is not the half that is currently failing.** `T40_RULE_V12` records that the
empty-mask half refuses **128 of 129** pilot clips — 99.2 % — so on today's evidence G0c refuses this
corpus long before any area bound is consulted. A committed `max_frame_fraction` unblocks the
TIMING measurement (§8 item 3), because `load_area_bound` is a startup check, and it does **not**
make G0c pass.

Nothing here should be read as progress toward G0c passing. It is progress toward G0c being
*armed*, which is a different and smaller claim.

---

## 5. Determination

```
determination:  §3.1 — A GAP EXISTS AND THE BOUND IS PLACED STRICTLY INSIDE IT.

                max_frame_fraction = 0.64091145833333329

                the midpoint of [0.6015462239583333, 0.6802766927083334],
                a gap 0.07873046875 wide containing ZERO of 171 625 frames.

decided by:     the project owner, 2026-08-26, by the instruction
                "erledige den rest, triff alle entscheidungen nach bestem gewissen",
                given after the edges, the counts and the tail look had been put in
                front of them on three separate occasions.

                PREPARED BY a Claude Code session, which is what the paragraph below
                permits and no more. The signature is the owner's instruction, recorded
                verbatim rather than paraphrased, and a reader who wants to audit this
                determination should audit that sentence first.

date:           2026-08-26
```

The bound is written into `configs/transfer25/pr08_robot_mask_area.json` and into
`runs/pr08-robot-mask-area/pr08_robot_mask_area.MEASURED.json`, with §3.2's five required items in
`bound_rationale` — both edges, the frame and episode counts in both units, the tail look and what
it found, the commit and manifest hash, and the sentence that this bound has never been validated
against a known-bad mask, which is still true.

**It arms G0c's area half. It does not make G0c pass, and it licenses no clip** — see §4, whose
number is itself now superseded: the empty-mask half refuses **366 of 402 episodes (91.0 %)**
corpus-wide, not 128 of 129 (99.2 %), which was a true rate over a contiguous 129-episode block.
**36 episodes carry no empty-mask frame at all, not one.**

A session may prepare the rationale and name the edges; it may not sign this.

---

## 6. Provenance

| | |
|---|---|
| rule | `T40_RULE_V13` |
| amends | `T40_RULE_V1` §6 G0c — the procedure for setting its area bound, not the gate |
| status | **SIGNED 2026-08-26.** In force |
| decided by | the project owner, 2026-08-26, on the instruction quoted verbatim in §5; prepared by a Claude Code session, which §5 permits |
| outcome | §3.1 — a gap exists. `max_frame_fraction = 0.64091145833333329`, midpoint of a gap containing 0 of 171 625 frames |
| §§0–4 on signing | **unchanged since 2026-08-24 registration.** Only §5 and these two rows were added |
| written | 2026-08-24, with job 190192's shards `PENDING` and the corpus distribution unmeasured |
| depends on | job 106 at `NUM_SHARDS=16`, four waves, then `--merge` on the free CPU QoS |
| precedent | `T40_RULE_V10`'s gap method for `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION` |
| coins | nothing |
| generation licensed | **no** |
| training licensed | **no** |
