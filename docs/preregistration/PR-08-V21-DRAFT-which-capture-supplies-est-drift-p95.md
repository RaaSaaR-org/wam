# PR-08 V21 — which capture supplies `EST_DRIFT_P95`, fixed before the number is visible

**Rule `T40_RULE_V21`. DRAFT, UNSIGNED. §§0–4 prepared by a session; §5 is empty and is the project
owner's.** `T40_RULE_V13` §5, in its own words and applied here: *"A session may prepare the
rationale and name the edges; it may not sign this."* V13 itself was written in exactly this shape —
§§0–4 on 2026-08-24, §5 and the provenance rows added on signing two days later.

**This document must be signed BEFORE the `pr08-geom-tol-v2` merge lands, or it stops being a
pre-registration.** Waves 0–3 were submitted 2026-08-28 as job 191143. The carry step that needs
this answer runs after the merge, and only after it — so the window is open now and closes then.

---

## 0. What this does and does not do

**Does:** fix, blind, which `EST_DRIFT` capture supplies the `EST_DRIFT_P95` that G0b subtracts
from `GEOM_TOL`.

**Does not:** measure anything, move `GEOM_TOL`, change G0b, change what `gate_qualified` means,
license a clip, or lift `T40_RULE_V1` §1. It does not answer whether item 4 closes — that depends on
a margin nobody has yet, and §4 says so.

## 1. Why this must be fixed before the merge and not after

`gate_margin_px = geom_tol_px − est_drift_p95_px` (`scripts/measure_geom_tol.py:2180`), and **item 4
closes iff that margin is positive.** Thirteen candidate captures sit on disk, spanning
0.185 to 0.672 px. Choosing among them *after* the re-measured `GEOM_TOL` is visible is choosing the
margin's sign, and there is a spread of **0.134 px across Arm A alone** — 28 % of the discarded
`0.4786 px` — to choose it with.

`T40_RULE_V17` §4 left the question open in its own words, and `scripts/measure_geom_tol.py:3458-3463`
is where that abstention is implemented rather than merely stated:

> **No carry path is built for it, deliberately** — whether G0b's budget is the pooled number or a
> single capture's is an unanswered owner decision. […] Writing the plumbing would answer it by
> making one of the two the reachable one, which is the same class of mistake as picking an arm by a
> field name.

So the repository has been holding this open on purpose, and the cost of that discipline is that
somebody must now close it in writing.

## 2. What the number is for, stated precisely, because it constrains the choice

`EST_DRIFT_P95` is a **budget for estimator error**, not a measurement of the generator. PR-08 §4
step 4: the 95th percentile of object-centroid displacement between the estimated and the true
segmentation *"enters G0b's tolerance as a budget rather than being assumed to be zero."* It is
subtracted, so:

> **A larger `EST_DRIFT_P95` makes the gate STRICTER. A smaller one makes it more permissive.**

That asymmetry is the whole of the argument in §3. Under-stating this number does not fail safe.

Two facts the choice has to survive, both measured:

* **Every candidate carries `gate_qualified: false` today**, with reasons including
  `geom_tol_is_not_gate_qualified`. They must be re-measured after the merge (~4 min locally, 0
  allocation GPU-h) whichever one this rule names. Naming a capture does not skip that.
* **Every candidate is a MuJoCo capture.** `grep -rl '"ground_truth_route": *"isaac"' runs/` returns
  nothing; no Isaac capture has ever been taken. `T40_RULE_V14`, signed 2026-08-27, licenses that
  substitution *"for that measurement and for no other"*, which is this measurement.

## 3. The rule

**`EST_DRIFT_P95` is supplied by the gate-qualified Arm-A capture with the LARGEST `est_drift_p95_px`.**

Carried with `scripts/measure_geom_tol.py --carry-est-drift <that artifact> --est-drift-arm per_frame`;
the arm flag is not optional and is not defaulted.

Three clauses, each doing work:

**(a) A criterion, not a name.** The same property `T40_RULE_V20` §3 states for a different
selection — *"The rule is a criterion, not a name, so that it can be checked rather than trusted."*
On today's artifacts the criterion resolves to **`A6`, `est_drift_p95_px = 0.36010037281174667`**
(`runs/pr08-est-drift/v17/EST_DRIFT-A6.json`), against `A4`'s `0.22601791717922304` at the other end.
That resolution is written here **to be checked against, never to be used in place of the
criterion**: if re-measurement after the merge moves it, the criterion governs and the rule is
re-evaluated rather than silently retargeted.

**(b) LARGEST, because the largest is the conservative one by construction.** The margin is
`tol − drift`, so the largest budget yields the smallest margin and the strictest gate. It is the
only choice among the thirteen that cannot be accused of having been made to produce a positive
margin — and it is the one an operator who wanted item 4 to close would not pick.

**(b′) And it answers a defect the source document named rather than inventing a preference.**
`PR-08-RESULT-2026-08-27-geom-tol-is-measured-and-uncommittable.md:211` records that
***"SINGLE" is under-specified*** — at least four distinct `headline_valid` single-capture per-frame
values exist, so "use a single capture" is not yet an instruction. A criterion is what turns it into
one. The same page also measured where the pooled number sits: **pooled per-frame `0.31208` is
looser than `A6`'s `0.36010`**, so the rule below is stricter than the pooled alternative rather
than a way around it.

**(c) ARM A, and this restriction is load-bearing rather than decorative.** The thirteen captures are
eight Arm-A measurements and five controls (`C1-lattice`, `C2-t20`, `C2-t40`, `C2-t80`,
`C3-wrongseed`). **Two controls exceed every Arm-A value** — `C2-t40` at `0.6716102795890873` and
`C2-t20` at `0.36934032408774303`. Without (c), clause (b) would select a control, which measures
something the gate is not about. The controls stay controls.

### 3.1 What is refused, and what happens instead

| | |
|---|---|
| the criterion resolves to a different capture after re-measurement | the carry does not proceed; this rule is re-evaluated against the artifacts that exist. **It is NOT retargeted at the runner-up.** |
| no Arm-A capture is `gate_qualified` after the merge | the carry does not proceed. That is a statement about the merge, not a licence to relax (c). |
| a POOLED artifact is offered instead | refused four checks deep by `measure_geom_tol.py:3452`, which accepts only `wam.est_drift/1`. §4 records what that costs. |

## 4. The thing this rule cannot fix, recorded so it is not mistaken for solved

**It does not make POOLED reachable, and it does not argue that POOLED is wrong.** Option (a) of the
decision — pooling the arms — would need a new schema acceptance, new carry plumbing, a rule version
and a signature, and `PR-08-RESULT-2026-08-27-geom-tol-is-measured-and-uncommittable.md` §7 measured
that **POOLED is not uniformly the conservative choice across the eight A-captures**. This rule takes
the single-capture branch and says so; it does not claim the other branch was refuted.

**It does not produce a positive margin.** Whether a re-measured `GEOM_TOL` reproduces `0.4786 px` is
UNKNOWN and is the entire reason job 191143 is running. If the margin comes out `≤ 0`, PR-08 §6 is
explicit about what that means — *"the estimator is not good enough and generation does not start"* —
and this rule will have done its job by making that outcome unarguable rather than negotiable.

**It does not touch the depth half.** MuJoCo returns distance to the image plane and
`distance_to_camera` is euclidean ray length, 1.41× apart at 45°
(`src/wam/robot/mujoco_binding.py:58-66`). `T40_RULE_V5` argues §4 gates on segmentation alone; that
argument is not re-opened here and not relied upon beyond what V5 already registered.

**It does not close §8 item 5**, whose relationship to the MuJoCo route is a separate unadjudicated
question — see `PR-08-DET-DRAFT-2026-08-28-items-5-and-6-are-not-a-silence-they-are-a-disagreement.md`.

## 5. Determination

**Read this before signing, because the source document says it plainly and this rule does not
soften it:** *"Whoever signs D-D is choosing a number, not choosing safety."*
(`PR-08-RESULT-2026-08-27-geom-tol-is-measured-and-uncommittable.md:209-210`.) §3 argues that the
largest Arm-A value is the conservative choice *among the available candidates*, and that is a
narrower claim than safety.

*Empty. A session may prepare the rationale and name the edges; it may not sign this.*

*To sign: state the outcome, the date, and the ground. If the outcome is anything other than the
rule in §3, that alternative is the rule and §3 is superseded by this section rather than edited.*

## 6. Provenance

| | |
|---|---|
| rule | `T40_RULE_V21` — **DRAFT, UNSIGNED** |
| answers | decision **D-D** of `docs/PR-08-DECISION-SHEET-2026-08-27.md`, left open by `T40_RULE_V17` §4 |
| amends | nothing. It fills a hole `T40_RULE_V17` registered as a hole. |
| changes | **no gate, no threshold, no verdict, no clip count, no style, no seed, no ceiling, and no committed artifact** |
| must be signed before | the `pr08-geom-tol-v2` merge lands (job 191143 submitted 2026-08-28) |
| evidence | `runs/pr08-est-drift/v17/EST_DRIFT-{A1..A8,C1-lattice,C2-t20,C2-t40,C2-t80,C3-wrongseed}.json` |
| resolves today to | `A6`, `0.36010037281174667` px — **a value to check the criterion against, not a name to carry** |
| implemented by | nothing new. `scripts/measure_geom_tol.py --carry-est-drift … --est-drift-arm per_frame` already exists and already refuses the pooled schema. |
