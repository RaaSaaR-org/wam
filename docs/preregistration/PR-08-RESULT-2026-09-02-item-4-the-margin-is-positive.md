# PR-08 RESULT — §8 item 4: the margin is `+0.1184795516078635` px, and §8 is closed in full

**A RESULT, not a determination. Nothing here is signed and nothing here needed a signature** — see
§4, which is the load-bearing part of this document rather than a formality.

| | |
|---|---|
| `GEOM_TOL` | **0.47857992441961017 px** — `pr08-geom-tol-v2`, 16 shards merged, 402/402 episodes |
| `EST_DRIFT_P95` | **0.36010037281174667 px** — capture A6, per-frame arm |
| `gate_margin_px` | **+0.1184795516078635 px** |
| §8 item 4 | **CLOSED** |
| §8 overall | **closed in full**, 7 of 7 |

---

## 1. The criterion resolved, against the re-measured set and not the old one

`T40_RULE_V21` §3, signed 2026-09-01 **before** the merge: *`EST_DRIFT_P95` is supplied by the
gate-qualified Arm-A capture with the LARGEST `est_drift_p95_px`.* §3.1 requires that criterion to
be resolved against the **re-measured** captures.

All eight Arm-A captures were re-measured on 2026-09-02 under the repaired adapter:

| capture | `est_drift_p95_px` | `gate_qualified` | reproduces v17 |
|---|---:|:---:|:---:|
| A1 | 0.29077062684224225 | true | exactly |
| A2 | 0.30607750168872560 | true | exactly |
| A3 | 0.26542268503711120 | true | exactly |
| A4 | 0.22601791717922304 | true | exactly |
| A5 | 0.29489486077636870 | true | exactly |
| **A6** | **0.36010037281174667** | **true** | **exactly** |
| A7 | 0.35028571678743880 | true | exactly |
| A8 | 0.33711288869158440 | true | exactly |

**The criterion resolves to A6**, which is the resolution `T40_RULE_V21` §5 recorded to be checked
against. §3.1's other branch — *re-evaluate the rule, do not retarget it at the runner-up* — is
therefore not taken. A7 at 0.35029 is the nearest competitor and loses by 0.00981 px.

**Every value reproduces its v17 measurement to the last digit.** That is the substantive finding
of the re-measurement and it is worth stating plainly: the earlier disqualification was never about
the numbers. It was three metadata conditions — the adapter's flag, and a committed GEOM_TOL
document that was neither gate-qualified nor recording the field at all. The physics did not move.

## 2. The carry

```
.venv/bin/python scripts/measure_geom_tol.py \
  --carry-est-drift runs/pr08-est-drift/v21/EST_DRIFT-A6.json \
  --est-drift-arm per_frame
```

`--est-drift-arm per_frame` here **states a decision there was none to make**: A6 measured one arm.
It is not the open owner decision the flag's help describes, which arises only for a two-arm
artifact. The artifact records `selected_by` accordingly.

The carry's own preconditions all passed before anything was written: same segmenter name, same
segmenter contract field for field, same pixel grid, target measured and `gate_qualified`.

## 3. The margin

```
0.47857992441961017  -  0.36010037281174667  =  0.1184795516078635 px      (24.8 % of GEOM_TOL)
```

**Positive.** PR-08 §6's other outcome — *the estimator is not good enough and generation does not
start* — is not the one that occurred. Had it been, no document on this page would have changed it.

## 4. Why this is a RESULT and why signing it would have been wrong

`T40_RULE_V13` §5 — *"A session may prepare the rationale and name the edges; it may not sign
this"* — governs determinations. **Item 4 is not a determination.** Its criterion was fixed in
writing and signed by the owner on 2026-09-01, and everything since has been arithmetic under it:
resolve the criterion, carry, subtract.

**And a signature here would have destroyed the thing V21 was built to protect.** V21 exists
because choosing the `EST_DRIFT_P95` capture *after* `GEOM_TOL` is visible is choosing the margin's
sign. The owner signed blind — `configs/transfer25/pr08_geom_tol.json` held `geom_tol_px = null` at
the moment of signing, five minutes before the merge wrote to it. Asking for a second signature
*now*, with `+0.1185` on the screen, would be asking the owner to ratify an outcome they had
already been careful not to be able to see. **The blind signature is the stronger instrument and
this document does not weaken it by adding a sighted one.**

This is the same reasoning `PR-08-DET-2026-09-01` D-1 made in the strict direction: a gate closes
when a document says so, over a signature. The document that closes item 4 is `T40_RULE_V21`, and
it is signed. This page records that its criterion resolved and what the arithmetic produced.

## 5. §8 after this result

| item | status | by |
|---|---|---|
| 1 | closed | prior record |
| 2 | closed | prior record |
| 3 | closed 2026-09-01 | `PR-08-DET-2026-09-01-the-spend-authorised.md` |
| 4 | **closed 2026-09-02** | this result, under `T40_RULE_V21` |
| 5 | closed 2026-09-01 | `PR-08-DET-2026-09-01` D-3 + D-4 |
| 6 | closed 2026-09-01 | `PR-08-DET-2026-09-01` D-2 |
| 7 | closed | prior record |

**§8 is conjunctive and it is now closed in full.** `T40_RULE_V1` §1 forbade generation *"until
every item in §8 is closed and T-39 has reported"*. T-39 reported 2026-08-16. Both halves are
satisfied.

**What that licenses is exactly one clip and nothing more.** The spend determination of 2026-09-01
released *"one clip — one generated episode, video and action column"* against a ceiling of
`PARTITION_CEILING_GPU_H = 2013.75`, and states in its own words that the remainder is **a cap, not
a release**. A second clip is a separate decision.

## 6. What this result does not say

* **Nothing about G0a/G0b having passed.** They cannot have: `run_g0_gates.py --explain` requires a
  restyled corpus and restyled centroids, and no clip exists. They run *after* the first clip, and
  the margin recorded here is the budget they will be run against — not a substitute for running
  them.
* **Nothing about training.** `T40_RULE_V1`, CLAUDE.md and `PR-08-DET-2026-09-01` all leave that
  where it was: the project owner's call.
* **Nothing about GR00T.** PR-07 §6 still forbids it.
* **Nothing about the 17.** `T40_RULE_V20` §5 stands: *"outcome R does not license walking down the
  17 until one passes"*, and the clip is submitted as a chunk, not chosen as an episode.
