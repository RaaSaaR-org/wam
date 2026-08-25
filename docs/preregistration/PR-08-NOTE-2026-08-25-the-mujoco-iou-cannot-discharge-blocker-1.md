# PR-08 — why the MuJoCo ground-truth IoU does **not** discharge "nobody has looked at a mask"

**Written 2026-08-25. A reading of documents already in the repository. Measures nothing, registers
no rule, discharges nothing.** It exists to close off an argument that is available, attractive,
and wrong, so that the next session does not spend the afternoon re-deriving it.

---

## 1. The argument

`scripts/estimators/apple_sam2.py`'s first gate-qualification blocker ends:

> *"Discharged by: a human looking at a sample of overlaid masks spanning the corpus (occluded
> frames, apple-out-of-frame frames, and the grasp), **and/or** a mask-vs-ground-truth IoU
> distribution from the Isaac capture recorded beside the centroid displacement. Neither exists."*

The disjunction is real: **either** limb discharges it. And as of 2026-08-25 the second limb looks
satisfied. `runs/pr08-est-drift/EST_DRIFT-mujoco-trajectory-f480.json` carries
`mask_vs_ground_truth_iou` over n = 480 frames — median **0.9881**, p5 0.9524, p1 0.9255, one
zero-IoU frame — measured against exact simulator segmentation, recorded beside the centroid
displacement, in the artifact the blocker names.

The only mismatch is the word **Isaac**. And `T40_RULE_V5` is precisely the rule that replaced
Isaac with "a simulator with exact segmentation", which is what produced this capture. So the
blocker's "Isaac capture" reads as stale wording that V5 already moved, and the number is
admissible.

**That argument fails, and it fails on V5's own text.**

## 2. V5 refuses it by name

`PR-08-V5-ground-truth-route.md` §0, the table of what V5 does *not* change:

> *"**The estimator's gate qualification** — unchanged. `scripts/estimators/apple_sam2.py`'s
> `GATE_QUALIFIED` is still `False` and its `GATE_QUALIFICATION_BLOCKERS` tuple is **untouched by
> this document** — in particular **"nobody has looked at a mask"** and "per-frame segmentation is
> not upstream's propagation" are still open, and `estimator_not_gate_qualified` will still be
> stamped on any capture measured today, by either route."*

and again, as a bullet:

> *"**V5 does not gate-qualify the estimator.** It moves no entry of
> `GATE_QUALIFICATION_BLOCKERS`."*

V5 names *this specific blocker*, states it is still open, and says any capture taken under V5 —
**"by either route"** — still carries `estimator_not_gate_qualified`. There is no gap to read into.
V5 changed which simulator renders §4's ground truth. It did not change what discharges a blocker,
and it says so twice.

## 3. Why a V14 cannot rescue it either

The obvious next move is to register a new rule extending V5's substitution into the blocker text.
**That move is foreclosed by the order in which things happened.**

The number already exists and has been read: median 0.9881 is in a committed artifact and in this
document. `docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten
after seeing its output is not a gate"* — applies exactly. A rule written today whose sole effect
is to make a favourable number already in hand admissible is not a versioning; it is the result
being made admissible, which is the failure V5's own preamble spends three paragraphs establishing
it did not commit.

**This is not a claim that MuJoCo ground truth is worthless.** It is `EST_DRIFT_P95`'s sanctioned
route under V5, it is the first ground-truth IoU in this project, and §4 uses it. The claim is
narrower and only about ordering: it cannot retroactively become the evidence for a blocker whose
discharge condition was written against a different capture, once its value is known.

## 4. What follows

**Blocker 1 terminates in the human look.** Its first limb is a person; its second limb is closed
to us for the reason above. And because blocker 2's discharge condition opens *"Discharged by the
same evidence as blocker 1, plus …"* — with the "plus" supplied on 2026-08-25 by
`PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md` — **blocker 2 terminates in the same
person.** One look discharges two blockers; nothing else discharges either.

The sheets exist: `runs/pr08-mask-audit/sheets/` (`spanning-00`, `occluded-00`, `grasp-00`,
`border-00`, `flagged-00`, `min_visibility-00`). Both MASK_AUDIT artifacts still carry
`human_review.looked_at: false`. **No session can flip that field**, and a model reading the sheets
is the correlated observer this project has refused three times already.

Blocker 3 (per-frame vs propagation) is unaffected by any of this and is a measurement, not a
judgement.

---

## 5. Provenance

| | |
|---|---|
| kind | reading of existing documents. **Registers no rule, measures nothing** |
| date | 2026-08-25 |
| sources | `apple_sam2.py` `GATE_QUALIFICATION_BLOCKERS` entries 1 and 2; `PR-08-V5-ground-truth-route.md` §0 and its bullet list; `docs/handoff.md` §3 |
| argument | **raised by this session and refuted by V5's own text** |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
