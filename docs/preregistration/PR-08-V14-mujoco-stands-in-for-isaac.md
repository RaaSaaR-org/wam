# PR-08 V14 — a MuJoCo capture may stand in for the Isaac one, and what that does not buy

**Rule `T40_RULE_V14`. SIGNED 2026-08-27 by the project owner. See §4 for the signature and §3
for the two things this does not do.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place. A gate rewritten after seeing its output is not a gate."* V14 is that versioning.

## 1. The question this answers

`GATE_QUALIFICATION_BLOCKERS[0]` in `scripts/estimators/apple_sam2.py` — the propagation blocker —
names its own discharge condition:

> Discharged by: measuring the same **Isaac** capture BOTH ways — this adapter per frame, and the
> video predictor propagating from frame 0 — and recording the two p95s.

Both arms have been measured, over the same 480-frame capture, with the JPEG-transcode confound
excluded bitwise (`runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json`). **The capture
is MuJoCo.** The blocker's own text then records why that is not yet a discharge, in two
independent parts:

> FIRST, THE CAPTURE IS MUJOCO AND THIS BLOCKER SAYS ISAAC. Whether a MuJoCo capture may stand in
> for the Isaac one is a **rule question for the project owner** and is not settled by measuring
> more MuJoCo. SECOND, 480 frames of ONE trajectory is not a corpus.

V14 answers the first part only.

## 2. The rule

**A MuJoCo capture may stand in for the Isaac capture named in the propagation blocker, for the
purpose of measuring `EST_DRIFT_P95` and the per-frame/propagation arm difference.**

The substitution is licensed for that measurement and for no other. In particular it does not
extend to `GEOM_TOL`, which is measured on the real corpus and never on a capture, and it does not
license quoting a MuJoCo-derived number as an Isaac-derived one: every artifact must continue to
record which simulator produced it, and `EST_DRIFT-ARMS-mujoco-trajectory-f480.json` already does.

**What makes this a rule and not a convenience.** The quantity being measured is the *estimator's*
error against a known ground-truth mask — how far this adapter's centroid sits from the true
centroid when the true centroid is known. That is a property of the estimator and of the rendered
pixels it is shown, not of the physics engine that produced the trajectory. Isaac was named in the
blocker because Isaac was the capture route being built at the time; nothing in the blocker's
argument turns on Isaac specifically. A signer who disagrees with that reasoning should refuse this
document, and the reasoning is stated here rather than assumed so that refusal is possible.

## 3. What this does NOT do

**3.1 It does not discharge the propagation blocker.** The blocker names two independent reasons,
*either of which alone would suffice*, and this closes one. **480 frames of one trajectory is still
not a corpus.** The blocker stays in `GATE_QUALIFICATION_BLOCKERS`, `GATE_QUALIFIED` stays `False`,
and the wording is not edited on the strength of this document alone.

**3.2 It does not make G0b pass — and the owner was shown the opposite before signing.** With
`GEOM_TOL = 0.47857992441961017 px` measured over the full corpus (402 episodes, 171 625 frames,
`runs/pr08-geom-tol/pr08_geom_tol.json`), the margin `GEOM_TOL - EST_DRIFT_P95` reads:

| arm | `EST_DRIFT_P95` | margin | as % of `GEOM_TOL` |
|---|---|---|---|
| per-frame (this adapter) | 0.29077062684224225 px | 0.18780929757736792 px | 39.2 % |
| propagation (the generator's) | 0.47006167975525187 px | 0.00851824466435830 px | **1.8 %** |

**The generator propagates.** So the arm that matters leaves a margin of about eight thousandths of
a pixel. This table was put in front of the owner in the session message immediately preceding the
signature below, so the signature is given in knowledge of it.

Two caveats that keep the table from being read as a gate result. `EST_DRIFT_P95` is measured on
MuJoCo and `GEOM_TOL` on the real corpus, so the subtraction is indicative and not the gate's own
arithmetic; and neither number has been carried into `configs/transfer25/pr08_geom_tol.json`, whose
`measurement_fields` are all `null`, so `run_g0_gates.gate_budget()` still refuses before
subtracting anything. What the table establishes is a **direction**: the honest reading of G0b is
"passes by almost nothing, or does not pass", not "has room".

**3.3 It licenses no clip.** `T40_RULE_V1` §1 binds in full. §8 items 3 and 4 are open.

## 4. Determination

**SIGNED.** The project owner was asked, in German, on 2026-08-27:

> a) Darf eine MuJoCo-Aufnahme für die benannte Isaac-Aufnahme einstehen? Wenn ja, ist Blocker 3
> mit den vorhandenen Zahlen entladbar — und die Konsequenz ist, dass G0b mit 1.8 % Marge dasteht
> und du entscheiden musst, ob das ein Bestehen ist.

and answered, verbatim:

> **a) ja**

Recorded verbatim rather than paraphrased; this is the part of this document a reader should check
against the session transcript rather than take on trust.

**A correction belongs in this block, because it changes what the signature bought.** The question
as put to the owner asserted that a yes would make the blocker *"mit den vorhandenen Zahlen
entladbar"*. **That assertion was wrong**, and it was wrong against the blocker's own committed
text, which names two independent sufficient reasons and which the same session had written the
previous day. The owner therefore answered a question whose stated consequence overstated the
effect. §3.1 states the actual effect. The signature is recorded as valid for the rule in §2 — the
rule itself was stated correctly — and as **not** constituting agreement that the blocker is
discharged, which is a claim nobody has made in front of the owner accurately.

Prepared by a Claude Code session. §§1-3 were written before the signature was requested in the
form recorded above.
