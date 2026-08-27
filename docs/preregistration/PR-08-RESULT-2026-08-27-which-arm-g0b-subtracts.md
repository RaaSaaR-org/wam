# PR-08 — which arm G0b subtracts, and why the answer is the one that changes nothing

**Determination on decision D-C of `docs/PR-08-DECISION-SHEET-2026-08-27.md` §5. 2026-08-27.**

**How this was decided, stated first because it is unusual here.** The session put the question to
the project owner with the evidence for and against and with **no recommendation**, on the ground
that a session must not answer it. The owner **delegated it back to the session** — *"entscheide
selbst"* — on 2026-08-27. The reasoning below is therefore the session's, made under that
delegation, and it is written to be refusable: §5 names what would overturn it. A reader who holds
that this decision needs a named human signatory regardless should treat §4 as the thing to check,
because §4 is the reason the delegation is survivable at all.

**Answer: `per_frame`.** `EST_DRIFT_P95` is the per-frame arm's p95. The committed contract field
`segmenter.propagation` stays `"per_frame"`, unchanged.

---

## 1. What was actually being asked

`configs/transfer25/pr08_geom_tol.json` records `segmenter.propagation: "per_frame"` and
`scripts/measure_est_drift.py` writes the per-frame p95 into `est_drift_p95_px` by definition, so
today's plumbing already produces `per_frame`. The question was whether that is a **decision** or an
**accident of a field name** — which is exactly the defect this session found and fixed: the carry
read only that one field, `--arm both` changed nothing about which number it picked up, and both
arms record the *same* `SEGMENTER_CONTRACT`, so `contract_disagreements()` could not see the
difference. The first successful carry would have handed G0b a number nobody chose.

| arm | pooled p95 | margin against `GEOM_TOL` 0.47857992441961017 | % |
|---|---:|---:|---:|
| `per_frame` | 0.3120786214328541 px | 0.16650130298675608 px | 34.7907 |
| `propagation` | 0.4486097454155794 px | 0.02997017900403076 px | 6.2623 |

Re-derived from `runs/pr08-est-drift/v17/POOLED-V19.json` this session. The gap is 0.13653 px =
**28.5 % of the budget.**

---

## 2. The ground, which is PR-08 §4 step 2 and not the margin

PR-08 §4 step 2 requires `GEOM_TOL` and `EST_DRIFT_P95` to be measured **with the same segmenter**,
and §6 subtracts them. `configs/transfer25/pr08_geom_tol.json` states in its own text why:

> the subtraction is arithmetic only if both sides ran the same detector, the same segmenter, the
> same prompt, the same thresholds and the same box rule on the same pixel grid.

`GEOM_TOL` is measured by `measure_geom_tol.py` through the **per-frame** adapter. That is not a
preference; it is what the pre-registered contract records and what the sixteen shards ran.

**So subtracting the propagation arm's p95 from a per-frame-measured `GEOM_TOL` is a subtraction
across two instruments.** It would still look like arithmetic — which is the specific failure the
same-segmenter clause exists to prevent, stated in the contract document before either number was
measured. `per_frame` is the only arm for which §4 step 2 is satisfied by the measurement that
actually exists.

---

## 3. Why the two rival tie-breaks are both unavailable

**"Take the friendlier margin" is forbidden.** The sprint document says so directly — *"which arm is
authoritative is settled by §4 step 2, not by picking the friendlier number."* `per_frame` happens
to be the friendlier arm by 28.5 % of the budget, and **that is a consequence of this decision, not
a reason for it.** If §4 step 2 had pointed the other way, the answer would be 6.26 % and the honest
response would be that 6.26 % is not room.

**"Take the worse one to be safe" is not available either**, and this is the part that is easy to
get wrong. `scripts/estimators/apple_sam2.py` records the bias as **two-sided**:

> The bias is **TWO-SIDED**, which is why this cannot be waved through as conservative … with (a)
> and (b) together this number is neither a lower nor an upper bound on the generator's mask error.

and the two distributions **cross between p95 and p99** (per-frame p99 1.0431 / p100 67.633 against
propagation 0.5631 / 19.399). So "propagation is the worse arm" is a **p95-only** statement, and
above p95 the ordering reverses. There is no conservative choice to retreat to; picking propagation
buys a smaller margin without buying a bound.

---

## 4. Why this is a determination and not a gate written after seeing its output

`docs/handoff.md` §3: *"Rules are versioned, never edited in place. A gate rewritten after seeing its
output is not a gate."* Both arms' numbers are already known, so **registering a new rule here would
be exactly that** — and this document deliberately is not one.

**What makes it survivable is that it moves nothing.** It declines to change a contract field that
was fixed in advance, in the direction the field already pointed, and it produces no new artifact,
no re-measurement and no rule version. The failure mode the prohibition guards against is a
threshold *moved* to fit an outcome; the act here is a threshold *left where it was*. A reader who
disagrees can check that claim mechanically: `git diff` for this determination touches no config, no
`SEGMENTER_CONTRACT`, and no measurement.

**And the decision is recorded in the artifact rather than defaulted.** As of this session
`measure_geom_tol.py --carry-est-drift` **refuses** a two-arm artifact until `--est-drift-arm` names
one, records which arm was written and how it was chosen, and requires a non-headline arm to clear
the coverage floor that document registered for itself. So the operator must type `per_frame`, and
the committed document will say that somebody did.

---

## 5. What would refute this, and the thing that would actually settle it

**The open architectural question is not §4 step 2's prose.** It is
`docs/PR-08-DECISION-SHEET-2026-08-27.md` §7 item 7: **no producer for pre-computed depth/seg
conditioning maps exists in this repository.** `scripts/build_pr08_source.py` omits them by a
declared-temporary design — *"until those land, the honest manifest is one that claims no maps at
all"*.

**If those maps land and are produced by a propagating masker, the referent changes and this
determination must be revisited**, because then the instrument the geometry budget characterises
really would be a propagating one and §4 step 2 would point the other way. That is the condition to
watch, and it is a design decision nobody has made yet rather than a reading of a paragraph.

Also refuting:

* a `GEOM_TOL` re-measured at HEAD under a propagating adapter — then the same-segmenter argument
  points at `propagation` and this document is simply wrong;
* any evidence that the gate's operating percentile is above p95, where the two distributions have
  already crossed;
* a demonstration that `restyle_transfer25.py:342-343`'s intended end state — *"the run uses the
  estimator the geometry budget characterises"* — is met by the propagation arm and not by this one.

**What this does not do.** It licenses no clip, lifts no rule, flips no flag and closes no §8 item.
Item 4 closes only when `GEOM_TOL` is re-measured at HEAD, `EST_DRIFT` is measured and carried, and
the margin comes out positive. `T40_RULE_V1` §1 binds.
