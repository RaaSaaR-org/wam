# PR-08 V12 — an empty robot mask is two different findings, and G0c cannot tell them apart

**Rule `T40_RULE_V12`. Drafted 2026-08-24. UNSIGNED — see §5. Nothing here is in force until the
project owner signs it, and no number produced under it may be quoted before that.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), which is registered as
`T40_RULE_V1` and **has not been edited and must not be**. The repo's discipline is
`docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."* V12 is that versioning, not a revision.

**It is also the document that discipline is aimed at.** Read §2 before §3. This draft proposes
changing the semantics of a gate *after* measuring that the gate refuses 99.2 % of the corpus, which
is the precise shape of the failure `handoff.md` §3 names. Whether that is a repair or a
rationalisation is not a question this document can settle about itself, which is why §5 is blank
and stays blank until a person fills it in.

## 0. What this does not change

- **Not the composite.** The real robot's pixels are still unconditionally composited over every
  generated frame wherever a mask exists. V12 proposes nothing about *how* compositing works.
- **Not `max_frame_fraction`, and not the area bound.** That is the other half of `check_mask` and
  it is untouched. It is also, on this corpus, unreachable — see §1.3.
- **Not `T40_RULE_V9`'s object-grounding filter**, adopted 2026-08-24. V12 assumes it and does not
  adjust `ROBOT_MASK_OBJECT_MAX_IOU`.
- **Not the partition, any budget, any seed, `GEOM_TOL`, `EST_DRIFT_P95`, `GATE_QUALIFIED`, or
  either blocker tuple.**
- **Not `T40_RULE_V1` §1.** Its prohibition binds in full. §8 items 3 and 4 are open, and adopting
  V12 would not close either.

## 1. The finding

### 1.1 What G0c does today

`T40_RULE_V1` §6 registers G0c as the gate that is *"solved by construction rather than by a
threshold"*:

> the real robot's pixels are unconditionally composited back over every generated frame, using the
> robot segmentation mask. The defect cannot enter, and no threshold is needed.

`scripts/robot_composite.py`'s `check_mask` enforces that by refusing a clip when the robot mask is
empty on a frame. There is deliberately no threshold in that check — an empty mask means there were
no source robot pixels to composite, so the frame went unprotected, so the clip is refused.

### 1.2 What it does on this corpus

Job **189707** (the PILOT branch of `106_measure_robot_mask_area.sbatch`, 1 603 dense frames at
stride 1 over 3 episodes) measured `empty_mask.fraction = 0.35246`. The robot is **genuinely out of
shot in roughly 36 % of source frames** of AppleToPlate.

Against 129 measured clips, that rule refuses **128 — 99.2 %**. The single clip that survives,
`episode_000243`, carries a non-empty robot mask on all 417 of its frames. **87 of the 128 refusals
(68 %) are on frames `T40_RULE_V9`'s object-grounding filter did not empty**, i.e. the emptiness is
not an artefact of V9 and would not be repaired by tuning it.

As written, then, G0c would refuse essentially the entire corpus at generation time. A gate that
rejects everything is not protecting a corpus; it is declining to produce one.

### 1.3 The area bound sits behind a door that never opens

`max_frame_fraction` — the other half of `check_mask` — is the bound this project has not yet
coined, and `configs/transfer25/pr08_robot_mask_area.json` does not exist. It is worth recording
that on **this** corpus that bound is close to moot: the empty-mask refusal fires first on 99.2 % of
clips, so the area check is rarely reached. The pilot's own distribution has `max = 0.3618` over its
sample, well under any plausible bound.

That is not an argument for skipping the area measurement — the distribution is a prerequisite for
`TIMING=1` and therefore for §8 item 3 — but it is the reason the empty-mask half, not the area
half, is what actually gates this corpus.

### 1.4 The distinction the current rule cannot make

An empty robot mask on a source frame means **one of two things**, and they have opposite
consequences:

**(a) The robot is genuinely absent from the frame.** There are no robot pixels in the source, so
there is nothing to composite, and the composite being a no-op is *correct*. Refusing here discards
a good clip.

**(b) The masker failed on a frame where the robot IS present.** Source robot pixels existed and
were not found, so the generated frame's robot pixels survive uncomposited — which is exactly the
defect G0c exists to exclude. Accepting here admits the failure the gate was built for.

**The current rule cannot tell (a) from (b), so it refuses both.** That is a defensible design when
(a) is rare. On a corpus where (a) is 36 % of frames it is not a conservative gate, it is an
uninformative one: the refusal carries no information about whether anything is wrong.

**One thing (a) does not make safe, and this document must not pretend otherwise.** A frame with no
source robot is unprotected against a generator that *invents* a robot. The composite cannot fix
that in either direction — it has no pixels to write. That residual risk is real, it is not
addressed by anything proposed in §3, and the instrument for it is V8's hallucination probe, not
G0c.

## 2. The hazard, stated against this document

`docs/handoff.md` §3: *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."*

**This document was written after seeing that G0c refuses 99.2 % of the corpus.** That is not a
detail to be disclosed in a footnote; it is the central fact about this draft's provenance, and it
is the reason the determination block is left for a person rather than filled in by the session that
drafted it.

There are two justifications available for the change proposed in §3, and **they are not the same
justification even though they point the same way**:

> **The legitimate one.** The (a)/(b) distinction in §1.4 is real, it is a property of the world and
> not of our convenience, and it is *independently measurable* — the source episode's own state
> record knows where the arm is, without reference to the masker. On that reading G0c has been
> asking a question whose answer it cannot interpret, and the repair is to ask the question it
> actually meant. This justification would hold **even if the refusal rate were 3 % instead of
> 99.2 %.** That is the test.

> **The illegitimate one.** G0c refuses too much, refusing too much is expensive, so the gate should
> refuse less. This justification is *entirely* a function of the number it saw. It would evaporate
> if the refusal rate were 3 %, which is precisely what marks it as a rationalisation rather than a
> finding.

**A reader who cannot tell which of these is doing the work in §3 should refuse this document.** The
author's own claim is that §1.4 was derivable from the corpus before the refusal rate was known —
the 36 % out-of-shot figure is a property of AppleToPlate that job 189707 measured directly, and it
implies the refusal rate rather than being implied by it. But that claim is made by the party with
an interest in it, and it is exactly the claim a signer should be sceptical of.

**What would settle it, and has not been done:** the (a)/(b) split is a testable proposition. Take a
sample of empty-mask frames, establish by an instrument that does not involve the robot masker
whether the arm is in frame, and report the split. If (b) turns out to dominate, §3 is wrong and the
correct reading is that the masker is broken. If (a) dominates as §1.2 suggests, the repair is
principled. **§3 should not be adopted before that measurement exists.**

## 3. The proposed rule, and the alternatives it was chosen over

Stated as a proposal. Nothing here is established.

### 3.1 Considered: refuse the frame, and bound the per-clip empty fraction against the source

Refuse the *frame* rather than the clip, and refuse the clip only when its empty-mask fraction
substantially exceeds the fraction its own source clip exhibits — a restyled clip whose mask goes
empty far more often than its source did indicates masker failure rather than genuine absence.

**Objection, and it is fatal in the general case.** G0c's mask is computed on the **source** frame,
not the restyled one; the composite writes source pixels onto generated frames. So for the pass that
matters there is no "source of the source" to compare against, and the comparison exists only for a
restyled re-segmentation that G0c does not perform. The proposal also introduces a new coined
number — how much excess is "substantial" — into a gate whose whole design is to need no threshold.

### 3.2 Proposed: an independent witness that the arm is out of frame

Do not infer absence from the masker's silence. **Ask an instrument that does not depend on the
masker.** The source episode carries its own 43-dimensional state column, recorded from the robot,
which knows where the arm was at every timestep. With the camera's intrinsics and extrinsics, that
determines whether any part of the arm lay within the camera frustum for a given frame.

Then an empty mask is adjudicated rather than assumed:

- **Arm out of frustum, mask empty** → case (a). The composite is a legitimate no-op. The frame
  passes, and the count is recorded in the clip's `g0c.json` proof rather than being silently
  tolerated.
- **Arm in frustum, mask empty** → case (b). The masker failed on a frame it should have found.
  **Refuse, exactly as today.**

**Why this is the narrow option.** It moves no threshold, coins no number, and adds no tolerance.
The gate refuses precisely the frames it always meant to refuse and stops refusing the ones it never
did. It converts a silence into a measurement, which is the same move `T40_RULE_V9` made for the
robot-vs-apple confusion and `T40_RULE_V10` made for the mask-validity reference.

**The objection, and it is a real cost, not a formality.** It requires camera intrinsics and
extrinsics for AppleToPlate that are *not committed anywhere in this repository as this is written*,
and a forward-kinematics chain from the 43-dim state to link poses in the camera frame. If those
cannot be established from the corpus, this option is not available and §3.3 is what remains. **That
is the first thing a signer should check, because it decides whether this rule is implementable at
all.** An approximate frustum test would reintroduce exactly the coined tolerance this option's
merit rests on avoiding.

### 3.3 Considered: change nothing, and accept the finding

Leave G0c as it is, accept that it refuses this corpus, and read that as evidence that **the
compositing route does not work here** — in which case `T40_RULE_V1` §3's route has to be revisited
rather than its gate adjusted.

**This is not a straw man and it should not be dismissed.** It is the only option that carries no
risk of being a rationalisation, and if §2's measurement comes back showing (b) dominates, it is the
correct answer. Its cost is that it forecloses PR-08's method on this corpus on the strength of a
gate whose question §1.4 argues was mis-posed — which would be its own kind of error, in the
opposite direction.

### 3.4 Rejected: a tolerance on the empty-frame count

Permit up to *N* % empty frames per clip. **Rejected.** It is a coined number chosen after seeing
the distribution it would be fitted to, it makes (a) and (b) interchangeable up to a quota, and it
converts a gate into a budget. This is the option §2's illegitimate justification would produce, and
naming it here is how this document keeps itself honest.

### 3.5 Recommendation

**§3.2, conditional on the §2 measurement and on the camera geometry existing.** In that order: run
the (a)/(b) split first, confirm the extrinsics are recoverable second, adopt third. If the geometry
is unavailable, the choice is between §3.3 and a further version — not between §3.2 and a
convenience.

## 4. What it costs if this is wrong

**If V12 is adopted and (b) actually dominates** — the masker is failing on frames where the robot
is present, and the frustum test is a poor witness — then uncomposited generated robot pixels enter
the corpus while G0c reports a pass. This is the worse direction by a wide margin: it is the exact
defect G0c exists to exclude, it would be invisible in the gate's own record, and every downstream
arm would train on it. §2's measurement is the thing standing between this draft and that outcome,
which is why it is a precondition and not a follow-up.

**If V12 is refused and (a) actually dominates**, PR-08 loses its corpus to a gate that was asking a
question it could not interpret — 402 episodes of real demonstrations discarded because the robot
leaves frame during approach and retreat, which is ordinary for a pick-and-place recording. The cost
is the whole photoreal-augmentation line of work, and the loss would be silent in the sense that the
gate would look like it was working.

**Neither cost is symmetric with the other, and they are not both borne by the same party.** The
first is paid by whatever trains on the corpus, later, and diffusely. The second is paid now and
visibly. A gate designer should be biased toward the second, and this document's author records that
its own recommendation runs the other way — which is one more reason §5 belongs to someone else.

## 5. Determination

**Decided by: nobody yet. UNSIGNED.**

A session may draft this document; **it may not sign it**, and it may not treat a draft as a licence.
This one is left blank deliberately and for a stated reason: it proposes changing a gate's semantics
after seeing that gate's output, and the session that drafted it is not a disinterested party to
that question. §2 names the measurement that would make the change legitimate. **That measurement
has not been taken, and this document should not be signed before it has been.**

```
determination:  ____________________
decided by:     nobody yet
date:           ____________________
```

Nothing in this document licenses generation, training, or any statement of a result.
`T40_RULE_V1` §1's prohibition is untouched and still binds in full.

## 6. Provenance

| | |
|---|---|
| rule | `T40_RULE_V12` |
| status | **UNSIGNED DRAFT.** Not in force |
| drafted | 2026-08-24, after the 99.2 % refusal rate was known — see §2 |
| supersedes | nothing. It would **supplement** `T40_RULE_V1`, which stands and is unedited |
| would change | how `robot_composite.check_mask` adjudicates an empty robot mask (§3.2). In the G0c compositor only |
| decided by | **nobody yet.** Signing is the project owner's, and no session may sign it or act as though it were signed |
| evidence | job **189707** (`ROBOT_MASK_AREA_PILOT.json`, `empty_mask.fraction = 0.35246` over 1 603 dense frames / 3 episodes); the 128-of-129 clip refusal count and the 87-of-128 post-V9 split recorded in `.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md`; `episode_000243` non-empty on all 417 frames |
| coined numbers | **none.** §3.2 introduces no threshold; §3.4 is the option that would, and is rejected |
| depends on | `T40_RULE_V9` (adopted 2026-08-24); camera intrinsics/extrinsics for AppleToPlate, **which are not committed in this repository as this is written** (§3.2) |
| not touched | the composite, `max_frame_fraction`, the area bound, `ROBOT_MASK_OBJECT_MAX_IOU`, the partition, every budget, every seed, `GEOM_TOL`, `EST_DRIFT_P95`, `GATE_QUALIFIED`, either blocker tuple, and every signed rule document |
| jobs submitted | **none** |
| generation licensed | **no** |
| training licensed | **no** |
| precondition on adoption | the (a)/(b) split measurement in §2, and confirmation that the camera geometry §3.2 needs is recoverable |
