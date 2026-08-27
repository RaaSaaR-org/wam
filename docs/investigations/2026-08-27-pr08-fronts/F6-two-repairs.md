# FRONT 6 — the two smaller open repairs

**Written 2026-08-27 on the user's workstation. Read-only against `/home/humanoid/develop/wam` at
`19826cc`. No file under the repo was modified; every proposed change below is a diff, not an edit.
No cluster contact of any kind.**

Every claim is tagged:

- **[M]** measured *now*, in this session, on this workstation — the command is given.
- **[A]** recorded in a committed artifact or in tracked source — path and line given.
- **[I]** my inference from (M) and (A) — labelled as such, never mixed in.

---

## 0. Headline for each repair, before the detail

| | root cause | what it actually blocks | is it a session's call? |
|---|---|---|---|
| **REPAIR A** — the `plate.` pass | **the validity filter, not the detector.** `object_color_reference` is the warm-**fruit** predicate applied unconditionally to any label; a *correct* plate mask contains no warm fruit pixels and scores IoU **0.0000** against it | §6 G0b's **plate half** — i.e. G0b's *gate qualification*, which runs **after** generation. **NOT §8 item 3 or 4** | **No.** Any fix that lets a plate pass reach a committed number touches `SEGMENTER_CONTRACT` and `configs/transfer25/pr08_geom_tol.json`. New rule version + owner signature |
| **REPAIR B** — the warm reference on non-warm styles | the predicate encodes an appearance claim about the **source** corpus (`"the only saturated warm thing in any of these frames is the fruit"`) and is applied to restyled pixels, where it does not go quiet — **it moves to the table** | **nothing on the §8 path.** Measured: zero misfires on the source corpus and zero on the MuJoCo `EST_DRIFT` capture. It bites on the *restyled* side of G0b and on two generated-pixel diagnostics | The reference itself: **no** (both candidate fixes are rejected on measured grounds). The two **unguarded call sites**: **yes**, and they are the only part worth doing now |

**Neither repair blocks the sprint (§8 items 3 and 4).** Section 4 says so with the evidence, and
says it in the direction that is uncomfortable rather than the convenient one.

---

# 1. REPAIR A — the `plate.` pass

## 1.1 The premise has moved since the task was written, and the new state matters

The task states the pass *"refuses 100 % of source frames."* That was true **before PR-08 V10
landed (2026-08-24)**. It is not the current behaviour.

**[M] Reproduced now**, with the module imported directly and the heavy deps stubbed (no weights,
no network):

```
$ WAM_PR08_OBJECT_PROMPT="plate." .venv/bin/python -c '<import apple_sam2; call segment(zeros)>'
OBJECT_TEXT_PROMPT = 'plate.'
reference defined?  False
RAISED: MaskValidityReferenceUndefined
REFUSED: the mask-validity filter has no reference for 'plate.'.
counters: SEGMENT_CALLS=0  MASK_REFUSED_FRAMES=0
```

Exact repro (runnable, ~2 s, no GPU):

```bash
cd /home/humanoid/develop/wam && WAM_PR08_OBJECT_PROMPT="plate." .venv/bin/python - <<'PY'
import sys, types, importlib.util, numpy as np
for n in ("transformers","sam2","torch"):
    if importlib.util.find_spec(n) is None: sys.modules.setdefault(n, types.ModuleType(n))
s = importlib.util.spec_from_file_location("apple_sam2","scripts/estimators/apple_sam2.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("defined?", m.mask_validity_reference_is_defined())
try: m.segment(np.zeros((480,640,3),dtype=np.uint8))
except Exception as e: print("RAISED:", type(e).__name__)
print("SEGMENT_CALLS", m.SEGMENT_CALLS, "MASK_REFUSED_FRAMES", m.MASK_REFUSED_FRAMES)
PY
```

So today the pass **refuses the run at the first call, having loaded no weight and moved no
counter** — `scripts/estimators/apple_sam2.py:2010` calls `_require_mask_validity_reference()`
before `SEGMENT_CALLS += 1`. That is V10 §2 item 1 working exactly as registered.

**The consequence for §6 is identical**: the plate half still cannot be measured. What changed is
the *failure mode* — it used to present as `coverage: 0.0`, a fact about the corpus; it now presents
as a named refusal. **V10 §7 says this in as many words**
(`docs/preregistration/PR-08-V10-mask-validity-reference-scope.md:495`):

> *"**The plate half of §6 is still not measurable.** V10 makes the failure legible; it does not
> supply a reference for `plate.` and does not decide whether that half should be measured
> unfiltered."*

## 1.2 Root cause, proven from a committed artifact — it is the FILTER

The question asked was: detector, filter, or prompt/threshold. **It is the filter, and the
detector is provably innocent.** Two independent artifacts say so from opposite directions.

### (a) The apple pass catching the plate — V6's own audit

**[M]** computed now over the committed audits:

```bash
cd /home/humanoid/develop/wam && .venv/bin/python - <<'PY'
import json
for path in ("runs/pr08-mask-audit/MASK_AUDIT.json","runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json"):
    f=json.load(open(path))["frames"]
    low=[r for r in f if r["warm_apple_iou"]<0.10]
    ok=[r["warm_apple_iou"] for r in f if r["warm_apple_iou"]>=0.10]
    print(path, len(f), "below 0.10:", len(low), "min kept:", min(ok),
          "max warm_apple_px:", max(r["warm_apple_px"] for r in f))
PY
```

Result **[M]**, from `runs/pr08-mask-audit/MASK_AUDIT.json` (job 189637, 382 frames, 24 episodes):

| episode | frame | `warm_apple_iou` | `mask_area_px` | `warm_apple_px` | `plate_overlap_fraction` | `detection_score` |
|---|---|---|---|---|---|---|
| episode_000094 | 108 | **0.0** | 31 129 | 657 | 0.9762 | 0.23365 |
| episode_000094 | 133 | **0.0** | 30 913 | 54 | 0.9800 | 0.24554 |
| episode_000094 | 149 | **0.0** | 30 913 | 198 | 0.9797 | 0.30907 |
| … 12 rows in total, every one at `warm_apple_iou == 0.0` | | | | | | |
| **min IoU of every frame that was KEPT** | | **0.7492** | | | | |

The local-CPU audit (`runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json`, 169 frames) gives the same
shape: 9 rows at `0.0`, min kept `0.8415`.

**Reading [I]:** a mask that is 98 % coincident with the plate scores exactly `0.0000` against the
warm-fruit reference. That is not a threshold being slightly wrong — it is a predicate that has
**no signal at all** on the plate. `object_color_reference`'s own docstring
(`scripts/estimators/apple_sam2.py:1587`) states the corpus fact that makes it so:

> *"the cloth and the plate are **neutral to within two counts** and the robot is black or bare
> metal"*

### (b) The plate pass itself — V10 §1.1's matched control

**[A]** `docs/preregistration/PR-08-V10-mask-validity-reference-scope.md:78-96`, 20 source frames of
`episode_000000`, adapter **unmodified**:

| | `plate.` | `apple.`, the same twenty frames |
|---|---|---|
| `n_segment_calls` | 20 | 20 |
| `n_frames_without_detection` | **0** | 0 |
| `n_frames_with_empty_mask` | **0** | 0 |
| `n_frames_mask_refused` | **20** | **0** |
| `n_frames_mask_refused_no_reference` | 0 | 0 |
| `n_frames_retry_fired` | **0** | 0 |
| non-empty mask returned on | **0 of 20** | 20 of 20 |
| validity IoU | **0.0000 on every frame** | 0.9686 – 0.9744 |
| **winning detection score** | **0.7524 – 0.7773** | — |
| mask area | 0 px | 8 519 – 8 525 px |

**This settles all three candidate causes at once:**

- **Not the detector.** `n_frames_without_detection = 0`, scores **0.7524–0.7773** — an order of
  magnitude above the 0.167–0.309 the audit records for the *accidental* plate detections under the
  apple prompt. GroundingDINO finds the plate confidently on every frame.
- **Not the prompt or the thresholds.** `n_frames_retry_fired = 0`, so nothing ran at the
  `(0.10, 0.10)` fallback; the detector cleared `BOX_THRESHOLD = 0.15` /
  `TEXT_THRESHOLD = 0.25` unaided. SAM 2 drew a mask on every frame
  (`n_frames_with_empty_mask = 0`).
- **It is the validity filter.** The only stage that fired is `MASK_REFUSED_FRAMES`, and the score
  it fired on is `0.0000` — the same number the audit records from the other side.

`docs/preregistration/PR-08-V9-robot-mask-object-grounding.md:352-370` first reported this (§5.4
item 1, *"a defect in `T40_RULE_V6`'s blast radius, found while writing this and not fixed here"*),
and `.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md:808-815` records it as
**finding (2), "NEW AND NOT PREVIOUSLY NAMED … OPEN, not fixed."**

## 1.3 The second blocker nobody has named — and it is in the contract itself

**This is a defect I found while root-causing, and it is the one that makes REPAIR A bigger than it
looks.** Even with a perfect plate reference, a plate pass still cannot produce a gate-qualified
number, because **`object_text_prompt` is a `SEGMENTER_CONTRACT` field pinned to `"apple."`**.

**[A]** `configs/transfer25/pr08_geom_tol.json`, `segmenter` block:

```json
"object_text_prompt": "apple.",
```

**[A]** `scripts/estimators/apple_sam2.py:596` — `SEGMENTER_CONTRACT["object_text_prompt"] =
OBJECT_TEXT_PROMPT`, which is read live from `$WAM_PR08_OBJECT_PROMPT` at import.

**[A]** `scripts/run_g0_gates.py:1880` — the `--dump-centroids` path writes
`segmenter["contract"] = contract`, where `contract` is the adapter's own live `SEGMENTER_CONTRACT`
(`adapter_segmenter_contract`, line 1770).

**[A]** `scripts/run_g0_gates.py:1512-1519` — `instrument_disagreements` then runs
`contract_disagreements(side.segmenter["contract"], tolerance_contract)` **field for field**, and
`contract_disagreements` (line 1339-1347) compares every key both sides state.

**[I] Consequence:** a `plate.` centroid dump carries `contract.object_text_prompt: 'plate.'`; the
tolerance's contract carries `'apple.'`; the comparison yields
`contract.object_text_prompt: 'plate.' vs 'apple.'` for **both** sides, and those rows land in
`disqualified` (line 2148) — so **G0b runs, produces numbers, and is not gate-qualified.**
Independently, `scripts/measure_geom_tol.py:2035` `merge_committed_contract` **refuses the whole run
(exit 2, nothing written)** on the same field if a plate pass is ever pointed at the committed
document.

`scripts/run_g0_gates.py:1826-1829` already half-sees this — *"a run that measured only the object
cannot be gate-qualified for §6's 'object AND plate', and the artifact says so"* — but it names the
**missing label** as the problem, not the **pinned prompt**. The pinned prompt means that the
existence of a plate pass is, by itself, a contract disagreement.

**So REPAIR A has two locks, not one**, and the second one is inside the pre-commitment:

1. `_require_mask_validity_reference()` refuses the run (V10). *Repairable in the module.*
2. `object_text_prompt` is a single-valued contract field while §6 gates two labels. *Not
   repairable in the module — the committed document has to change shape.*

## 1.4 The minimal fix, and it touches the contract

### What was rejected first, and why (so it is not re-proposed)

- **A `plate.` colour predicate** — rejected by V10 §3.1 and by the module's own comment at
  `apple_sam2.py:479-486`: *"ADDING A LABEL HERE IS NOT A CODE CHANGE, IT IS A PRE-REGISTRATION …
  several numbers coined by us, in the gate path, with no measured gap to read them off."* A
  neutral-white plate on a cloth that is *"neutral to within two counts"* has no gap to read a
  number off. **This is exactly the failure PR-08 §4 step 2 forbids.**
- **Silently skipping the filter for an unregistered label** — rejected by V10 §3.1's third option:
  it fails open, and the artifact would still claim `mask_validity_min_iou` ran.

### What is left, and it is the one the module itself names

`apple_sam2.py:483-486`: *"Whether §6's plate half is measured **unfiltered** instead is the project
owner's call, and **it needs the committed contract to say so**."*

So the minimal fix is: **make "measured unfiltered" a declarable, contract-recorded, inert-by-default
state** — never a silent skip. Shipped **empty**, so the diff changes no behaviour on any existing
path and a signed rule version is what arms it.

### Diff 1 of 2 — `scripts/estimators/apple_sam2.py` (does NOT touch the committed JSON)

```diff
--- a/scripts/estimators/apple_sam2.py
+++ b/scripts/estimators/apple_sam2.py
@@ -486,6 +486,42 @@
 MASK_VALIDITY_REFERENCE_LABELS: frozenset[str] = frozenset({"apple."})
 
+#: The labels §6 is measured on with the validity filter DECLARED OFF, because no reference for
+#: them exists and none can be coined. **EMPTY AS SHIPPED, AND EMPTY IS THE POINT.** This constant
+#: changes nothing until a signed rule version puts a label in it; until then every path behaves
+#: exactly as it did under V10, value for value.
+#:
+#: WHY THIS EXISTS RATHER THAN A PLATE PREDICATE. `plate.` cannot be measured today (V10 §1.1:
+#: 20 of 20 source frames refused at validity IoU 0.0000 while the detector scored 0.7524-0.7773),
+#: and a colour discriminator for a neutral-white object on a cloth this module records as "neutral
+#: to within two counts" is several coined numbers in the gate path — what PR-08 §4 step 2 and V6
+#: §4 forbid. The remaining honest option is the one the comment above already names: measure that
+#: half UNFILTERED, and say so where it can be cross-checked.
+#:
+#: WHY IT IS NOT A SILENT SKIP, WHICH IS THE FAILURE MODE THIS SHAPE EXISTS TO PREVENT. V10 §3.1
+#: rejected "skip the filter for an unregistered label" because it fails open: the committed
+#: contract states mask_validity_min_iou, measure_est_drift.cross_check_geom_tol compares it field
+#: for field, and an artifact from a run whose filter never ran would still claim it did. So the
+#: declaration is (a) a constant, not an env var, (b) IN SEGMENTER_CONTRACT, so a committed document
+#: that does not carry it disqualifies the run rather than pooling with it, (c) counted per frame in
+#: MEASURED_UNFILTERED_FRAMES, and (d) in ESTIMATOR_VERSION, so G0b refuses to compare a side
+#: measured under it against a side measured without it.
+#:
+#: NO ENVIRONMENT OVERRIDE, for MASK_VALIDITY_MIN_IOU's reason exactly.
+MASK_VALIDITY_UNFILTERED_LABELS: frozenset[str] = frozenset()
+
+
+def mask_validity_is_declared_unfiltered() -> bool:
+    """Is THIS process's label one §6 measures with the filter declared off?
+
+    Cheap and importable, so a harness can find out before decoding a corpus. :func:`segment` asks
+    the same question. False for every label as shipped.
+    """
+    return OBJECT_TEXT_PROMPT in MASK_VALIDITY_UNFILTERED_LABELS
+
@@ -1092,6 +1092,15 @@
 MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES = 0
 
+#: Frames on which a mask was returned WITHOUT the validity filter having run, because this
+#: process's label is in :data:`MASK_VALIDITY_UNFILTERED_LABELS`. Counted rather than inferred: a
+#: coverage number produced with the filter off and one produced with it on are two different
+#: measurements, and only this counter tells them apart in an artifact. Zero on every existing path.
+MEASURED_UNFILTERED_FRAMES = 0
+
@@ -1655,10 +1655,25 @@
 def _require_mask_validity_reference() -> None:
     """Refuse the run when the mask-validity filter has no reference for this label. (PR-08 V10.)"""
     if mask_validity_reference_is_defined():
         return
+    if mask_validity_is_declared_unfiltered():
+        # A DECLARATION, NOT A FALLBACK. This branch is reachable only for a label a signed rule
+        # version put in MASK_VALIDITY_UNFILTERED_LABELS, the declaration is in SEGMENTER_CONTRACT
+        # and in ESTIMATOR_VERSION, and every frame it lets through is counted in
+        # MEASURED_UNFILTERED_FRAMES. It is empty as shipped, so this returns for no label today.
+        return
     raise MaskValidityReferenceUndefined(
@@ -2004,6 +2004,7 @@
     global MASK_REFUSED_FRAMES, MASK_REFUSED_NO_REFERENCE_FRAMES
     global MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES
+    global MEASURED_UNFILTERED_FRAMES
 
@@ -2043,6 +2044,15 @@
     # the exact threshold is a value read off a gap rather than a coined number.
+    if mask_validity_is_declared_unfiltered():
+        # The filter is DECLARED off for this label, so nothing is scored and nothing is appended to
+        # MASK_VALIDITY_IOU / MASK_VALIDITY_REFERENCE_FRACTION -- a zero in those lists would read as
+        # "the check ran and the mask failed". The frame is counted instead, so the artifact says
+        # how many frames were measured this way rather than leaving it to be inferred from a label.
+        MEASURED_UNFILTERED_FRAMES += 1
+        return mask
+
     reference = object_color_reference(frame)
     overlap = mask_validity_iou(mask, reference)
```

Plus the two recording hunks (`stats()` and `ESTIMATOR_VERSION`):

```diff
@@ -563,6 +563,11 @@
     f"mask_val_ref_max_frac={MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION}"
+    # Here for mask_val_min_iou's reason exactly: run_g0_gates.instrument_disagreements compares
+    # this string BETWEEN THE TWO SIDES of G0b, and a side measured with the filter declared off
+    # must not be comparable to one measured with it on. Empty renders as "()" and never moves.
+    f";mask_val_unfiltered={','.join(sorted(MASK_VALIDITY_UNFILTERED_LABELS)) or '()'}"
 )
@@ -1179,6 +1179,8 @@
         "mask_validity_reference_labels": sorted(MASK_VALIDITY_REFERENCE_LABELS),
+        "mask_validity_unfiltered_labels": sorted(MASK_VALIDITY_UNFILTERED_LABELS),
+        "mask_validity_is_declared_unfiltered_for_this_prompt": mask_validity_is_declared_unfiltered(),
@@ -1205,6 +1207,7 @@
         "n_mask_validity_iou": len(MASK_VALIDITY_IOU),
+        "n_frames_measured_unfiltered": MEASURED_UNFILTERED_FRAMES,
```

### Diff 2 of 2 — THE CONTRACT. **This one is not a session's to make.**

```diff
--- a/scripts/estimators/apple_sam2.py
+++ b/scripts/estimators/apple_sam2.py
@@ -613,6 +613,11 @@
     "mask_validity_reference_max_frame_fraction": MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION,
+    # A statement about WHICH FRAMES were measured, on the same grounds as the three above it, and
+    # it must be in the contract for the reason V10 §3.1 gives for rejecting a silent skip: an
+    # artifact from a run whose filter did not run would otherwise still claim it did.
+    "mask_validity_unfiltered_labels": sorted(MASK_VALIDITY_UNFILTERED_LABELS),

--- a/configs/transfer25/pr08_geom_tol.json
+++ b/configs/transfer25/pr08_geom_tol.json
@@   "segmenter": {
     "mask_validity_reference_max_frame_fraction": 0.1,
+    "mask_validity_unfiltered_labels": [],
```

### Does the fix touch the committed contract? **YES — and here is the exact cost.**

- `scripts/measure_geom_tol.py:1921-1923`: *"A field present on one side and absent on the other
  counts as a disagreement."* So **the moment `SEGMENTER_CONTRACT` grows the key, every artifact
  measured under the old contract stops cross-checking** unless
  `configs/transfer25/pr08_geom_tol.json` is edited in the same commit.
- The artifact at risk is real and is the sprint's own number. **[M]**
  `runs/pr08-geom-tol/pr08_geom_tol.json`: `GEOM_TOL_px = 0.47857992441961017`,
  `n_segment_calls = 171625`, `n_frames_mask_refused = 36`, `gate_qualified = false`. That is job
  189935's merged 16-shard measurement.
- `ESTIMATOR_VERSION` also moves, and `run_g0_gates.instrument_disagreements` (line 1479) refuses to
  compare two sides whose versions differ. V10 §6 already paid this cost once and wrote down the
  rule: *"the honest sequence is to land after that job completes, or to re-measure — not to leave
  the string alone so the two look the same."*

**Therefore: this fix needs a new rule version (a `PR-08-V20`-shaped document) and the project
owner's signature. It is NOT a session's call.** Three separate reasons, each sufficient:

1. `docs/handoff.md` §3 — rules are versioned, never edited in place; `PR-08-V6` and `PR-08-V10`
   may not be edited, and "measure the plate half unfiltered" is a *decision about which frames a
   gate number is measured on*, which is precisely what a rule registers.
2. `apple_sam2.py:483-486` says the module *cannot say it alone*, and names the committed contract
   as the thing that has to say it.
3. V10 §7: *"does not decide whether that half should be measured unfiltered. **That is the project
   owner's call and it needs the committed contract to say so.**"*

**And even with the owner's signature, Diff 1 + Diff 2 are still not sufficient**, because of
§1.3: `object_text_prompt` is pinned to `"apple."`. A signed version arming the plate half must
*also* decide what the contract's prompt field means when §6 gates two labels — either a per-label
map (`{"object": "apple.", "plate": "plate."}`) or an explicit exemption in
`run_g0_gates.contract_disagreements`. **That is a change to the contract's shape, not its values,
and it is a second signature question.** I am not proposing a diff for it: choosing between those
two shapes without the owner is exactly the kind of decision the pre-registration method exists to
prevent.

## 1.5 Does REPAIR A block the sprint (§8 items 3/4)? — **NO**

| | | evidence |
|---|---|---|
| §8 item 3 — throughput on an H200 + GPU-h ceiling | **unrelated** | requires a cluster run; the plate prompt appears nowhere in it |
| §8 item 4 — `GEOM_TOL` + `EST_DRIFT_P95` measured and committed | **not blocked** | **[A]** `scripts/measure_geom_tol.py:164-167`: *"G0b's prose gates 'Object *and* plate centroids'; the tolerance is derived from the OBJECT centroid alone. … **That does not block computing `GEOM_TOL`**; it blocks applying one number to both."* **[A]** `scripts/run_g0_gates.py:294-300` says the same from the gate side |

**[A]** `docs/preregistration/PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §1 puts the plate
half where it belongs: §8 items gate *generating a corpus*; G0a/G0b/G0c gate *training on the
result*. The plate half lives in the second row and therefore **downstream of the sprint**.

**[M]** And item 4's two numbers already exist without a plate pass: `GEOM_TOL_px = 0.4786` over
171 625 frames (`runs/pr08-geom-tol/pr08_geom_tol.json`), `est_drift_p95_px = 0.2361`
(`runs/pr08-est-drift/EST_DRIFT-mujoco-s60-f720.json`). Both carry `gate_qualified: false`, which is
what actually keeps item 4 open — and *that* is a residue/signature question, not a plate question.

**REPAIR A IS DOWNSTREAM. LABEL IT NOT BLOCKING.**

## 1.6 A pytest that would have caught the defect

Two tests. The first catches the original defect; the second catches the fail-open regression the
fix could introduce.

```diff
--- a/tests/test_apple_sam2_estimator.py
+++ b/tests/test_apple_sam2_estimator.py
@@
+def test_a_correct_mask_of_a_non_fruit_label_is_not_scored_against_the_fruit_predicate(monkeypatch):
+    """THE DEFECT, STATED AS A TEST. PR-08 V9 §5.4 / V10 §1.1.
+
+    Before V10 this passed a CORRECT plate mask through the warm-FRUIT predicate, scored 0.0000 and
+    returned all-False on every frame -- read out of the harness as `coverage: 0.0`, i.e. as a fact
+    about the corpus. The measured shape (V10 §1.1, 20 source frames of episode_000000) is exactly
+    this one: detector scores 0.7524-0.7773, zero empty masks, zero retries, twenty refusals.
+
+    The assertion is NOT "the plate is measured". It is the weaker, checkable thing: this module
+    must never silently report an unmeasurable label as an empty corpus. Either it has a reference,
+    or the label is declared unfiltered, or it REFUSES BY NAME -- and the refusal must arrive before
+    a single counter moves, because a partially-counted run writes an artifact.
+    """
+    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
+    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
+    monkeypatch.setenv("WAM_PR08_OBJECT_PROMPT", "plate.")
+    # A detection the detector is CONFIDENT about, at V10 §1.1's own measured score. The point of
+    # the test is that a confident, correct detection still produced coverage 0.0.
+    _install(monkeypatch, detections=[[(0.7649, [10.0, 10.0, 30.0, 30.0])]] * 8)
+    module = _fresh_import(monkeypatch)
+
+    assert module.OBJECT_TEXT_PROMPT == "plate."
+    assert module.mask_validity_reference_is_defined() is False
+    with pytest.raises(module.MaskValidityReferenceUndefined) as exc:
+        module.segment(_frame())
+    assert "plate." in str(exc.value)
+    # BEFORE ANY COUNTER MOVES. A run that counted eight calls and refused eight masks is the
+    # coverage-0.0 artifact this refusal exists to prevent.
+    assert module.SEGMENT_CALLS == 0, "the refusal must precede the first counted frame"
+    assert module.MASK_REFUSED_FRAMES == 0
+    assert module.stats()["n_segment_calls"] == 0
+
+
+def test_the_filter_can_only_be_turned_off_by_declaration_and_the_artifact_says_so(loaded, monkeypatch):
+    """THE FAIL-OPEN REGRESSION THE REPAIR COULD INTRODUCE. V10 §3.1's third rejected option.
+
+    A run whose validity filter did not run must never write an artifact that claims it did. Three
+    things have to be true at once, and all three are checked here because any one of them alone is
+    satisfiable by a silent skip.
+    """
+    module, _ = loaded
+    # (1) INERT AS SHIPPED. The declaration set is empty, so no label is unfiltered today and every
+    #     existing measurement is value-for-value what it was.
+    assert module.MASK_VALIDITY_UNFILTERED_LABELS == frozenset()
+    assert module.mask_validity_is_declared_unfiltered() is False
+    # (2) IT IS IN THE CONTRACT, so measure_geom_tol.contract_disagreements -- which counts an
+    #     absent field as a disagreement -- can see it. A knob outside the contract is a knob that
+    #     changes the measured population invisibly.
+    assert "mask_validity_unfiltered_labels" in module.SEGMENTER_CONTRACT
+    # (3) IT IS IN THE INSTRUMENT'S IDENTITY, so run_g0_gates refuses to compare a side measured
+    #     with the filter off against a side measured with it on.
+    assert "mask_val_unfiltered=" in module.ESTIMATOR_VERSION
+    # (4) NO ENVIRONMENT OVERRIDE, for MASK_VALIDITY_MIN_IOU's reason exactly.
+    for var in ("WAM_PR08_MASK_VALIDITY_UNFILTERED_LABELS", "WAM_PR08_UNFILTERED_LABELS"):
+        monkeypatch.setenv(var, "plate.,apple.")
+    assert _fresh_import(monkeypatch).MASK_VALIDITY_UNFILTERED_LABELS == frozenset()
+
+
+def test_an_unfiltered_label_is_counted_and_never_scored(monkeypatch):
+    """When a signed version DOES arm a label, the frames it lets through must be countable.
+
+    A 0.0 appended to MASK_VALIDITY_IOU would read as "the check ran and the mask failed", which is
+    the opposite of what happened. The frame goes into its own counter and into nothing else.
+    """
+    monkeypatch.setenv("WAM_PR08_ALLOW_DOWNLOAD", "1")
+    monkeypatch.setenv("WAM_PR08_DEVICE", "cpu")
+    monkeypatch.setenv("WAM_PR08_OBJECT_PROMPT", "plate.")
+    _install(monkeypatch, detections=[[(0.7649, [10.0, 10.0, 30.0, 30.0])]] * 4)
+    module = _fresh_import(monkeypatch)
+    monkeypatch.setattr(module, "MASK_VALIDITY_UNFILTERED_LABELS", frozenset({"plate."}))
+
+    mask = module.segment(_frame(apple=None))   # no warm fruit anywhere: IoU would have been 0.0
+    assert mask.any(), "a declared-unfiltered label must not be refused by the fruit predicate"
+    stats = module.stats()
+    assert stats["n_frames_measured_unfiltered"] == 1
+    assert stats["n_frames_mask_refused"] == 0
+    assert stats["n_mask_validity_iou"] == 0, "nothing was scored, so nothing may be recorded as scored"
+    assert stats["mask_validity_is_declared_unfiltered_for_this_prompt"] is True
```

---

# 2. REPAIR B — the warm-colour reference on non-warm styles

`MASK_VALIDITY_REFERENCE = "warm_saturated_rgb(r>90, r-b>50, saturation>0.35)"`
(`scripts/estimators/apple_sam2.py:431`), implemented at `apple_sam2.py:1596-1600` and restated
identically at `scripts/audit_apple_masks.py:350-365`.

## 2.1 Which styles it misfires on — and how much of that is measured

**Only two styles have ever been measured. The rest is inference from prompt strings, and I mark it
as such.**

### [M] Measured now, by me, on the committed contact sheets

The only restyled pixels on this workstation are job 189926's sheets under
`runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/sheets/` (11 PNGs, 2 styles, 2 episodes).
Panel geometry is deterministic from `probe_hallucination._sheet_tile` (640×480 pair → 1280×480 →
half → 640×240) plus `audit_apple_masks.captioned` (4-line band, `4*2 + 16*4 = 72 px`) and
`contact_sheet(cols=2, gap=6, header=28)` → tile 640×312, sheet width `2*646+6 = 1298` ✓ (matches
the files). Left half of each tile is SOURCE, right half is GENERATED.

I ran the module's **own** `object_color_reference` / `reference_frame_fraction` over both halves:

```bash
cd /home/humanoid/develop/wam && .venv/bin/python - <<'PY'
import sys, types, importlib.util, glob, numpy as np
from PIL import Image
for n in ("transformers","sam2","torch"):
    if importlib.util.find_spec(n) is None: sys.modules.setdefault(n, types.ModuleType(n))
s = importlib.util.spec_from_file_location("apple_sam2","scripts/estimators/apple_sam2.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
GAP,HEADER,TW,TH,PW,PH = 6,28,646,318,320,240
for p in sorted(glob.glob("runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/sheets/*.png")):
    im=np.asarray(Image.open(p).convert("RGB")); src=[];gen=[]
    for r in range((im.shape[0]-HEADER-GAP)//TH):
        for c in range(2):
            x0,y0 = GAP+c*TW, HEADER+GAP+r*TH
            a=im[y0:y0+PH, x0:x0+PW]; b=im[y0:y0+PH, x0+PW:x0+2*PW]
            if a.shape[:2]==(PH,PW)==b.shape[:2]:
                src.append(100*m.reference_frame_fraction(m.object_color_reference(a)))
                gen.append(100*m.reference_frame_fraction(m.object_color_reference(b)))
    over=sum(1 for v in gen if v>100*m.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION)
    print(f"{p.split('/')[-1][:60]:60s} n={len(src):2d} src {min(src):6.2f}-{max(src):6.2f}%  "
          f"gen {min(gen):6.2f}-{max(gen):6.2f}%  over-bound {over}/{len(gen)}")
PY
```

**[M] Result** (`%` = fraction of the 320×240 panel; the bound is 10 %):

| sheet | n | source ref % | **generated ref %** | over the 0.10 bound |
|---|---|---|---|---|
| `ep000000__train-01-oak-tungsten__candidate_invention` | 12 | 4.48 – 4.51 | **40.51 – 56.40** | **12/12** |
| `ep000000__train-01-oak-tungsten__both_empty` | 12 | 2.29 – 2.33 | **37.63 – 55.66** | **12/12** |
| `ep000000__train-01-oak-tungsten__excluded_source_mask_nonempty` | 4 | 0.00 – 2.32 | 0.00 – **38.80** | **3/4** |
| `ep000001__train-01-oak-tungsten__candidate_invention` | 12 | 4.46 – 4.48 | **39.05 – 40.58** | **12/12** |
| `ep000001__train-01-oak-tungsten__both_empty` | 12 | 2.29 – 2.31 | **37.46 – 39.06** | **12/12** |
| `ep000001__train-01-oak-tungsten__excluded_source_mask_nonempty` | 4 | 2.29 – 2.30 | **37.18 – 37.82** | **4/4** |
| `ep000000__train-02-linen-overcast__candidate_invention` | 12 | 4.47 – 4.51 | 4.27 – 4.39 | 0/12 |
| `ep000000__train-02-linen-overcast__both_empty` | 12 | 2.30 – 2.33 | 2.09 – 2.22 | 0/12 |
| `ep000000__train-02-linen-overcast__excluded_source_mask_nonempty` | 4 | 0.00 – 2.32 | 0.00 – 2.21 | 0/4 |
| `ep000001__train-02-linen-overcast__both_empty` | 12 | 2.29 – 2.30 | 2.19 – 2.22 | 0/12 |
| `ep000001__train-02-linen-overcast__excluded_source_mask_nonempty` | 4 | 2.29 – 2.30 | 2.19 – 2.21 | 0/4 |

**This reproduces V10 §4.1 value for value.** V10's table records
`train-01-oak-tungsten … 40.5 – 56.4 %`, `train-02-linen-overcast … 4.27 – 4.39 %`, and source
panels at `4.51 %`. My independent re-derivation gives `40.51 – 56.40`, `4.27 – 4.39`, `4.51`. **[A]
+ [M] agree exactly.**

**It also extends V10 by one episode.** V10 measured `episode_000000` only. `episode_000001` misfires
too (37.18 – 40.58 %, **28 of 28 generated panels over the bound**), so the misfire is a property of
the **style**, not of one episode. That is 56 generated panels over the bound across two episodes,
versus V10's 12.

**[A] V9 §5.1** independently records `34 632 px of warm oak table` on the same sheet, which sits
inside my measured 31 112 – 43 319 px range.

### [I] Inference — which of the other 13 styles are exposed

`configs/transfer25/styles.toml` commits 1 identity + 10 train + 5 eval styles. **No restyled
frame exists on this workstation for any of the other 13**, so what follows is my reading of the
committed prompt strings and nothing more.

| axis | styles | inference |
|---|---|---|
| **Apple is NOT warm-saturated → reference goes quiet on the fruit** (V6's empty-reference path, still ambiguous per V10 §5.3) | `train-01` bright green Granny Smith · `train-02` pale yellow Golden Delicious · `train-05` mottled pink-and-green Pink Lady · `train-06` pale green waxy · `train-09` speckled yellow-green · `eval-01` lime green · `eval-02` very dark purple-black · `eval-04` pale ivory white | **8 of 16.** V9 §5.1 and T-040's 2026-08-23 finding (3) name five of these ("green Granny, pale-green waxy, Golden Delicious, russet, Pink Lady") **[A]**; the eval three are mine **[I]** |
| **Background/light IS warm-saturated → reference latches onto the SCENE** (V10's bound, the measured failure) | `train-01` pale oak + warm tungsten **[M, confirmed]** · `train-05` red-and-white checked cloth + varnished wood + warm window daylight · `train-07` dark walnut + red brick + amber evening · `eval-01` terracotta tile + strong rim light · `eval-03` woven bamboo + yellow wallpaper · `eval-05` hessian burlap + warm lantern glow | **1 measured, 5 inferred.** `train-05` and `eval-01` are in **both** rows, which is the worst combination: a non-warm fruit *and* a warm scene |
| **Reference plausibly still works** | `identity-source` · `train-02` **[M, confirmed]** · `train-03` deep red matte on white melamine · `train-04` russet on grey slate · `train-08` dark crimson on cork/pale blue · `train-10` bronze-red on denim/olive | 1 measured, 5 inferred |

**[A] V10 §5.1 names this gap first and asks for exactly the missing measurement:**

> *"One `segment()` pass over a few hundred frames of **each committed style's** restyle … Until it
> is run, the claim this document supports is *"on the twelve panels of one style, on one
> episode"*, and it is not a corpus rate."*

**That run requires the cluster.** The clips are at
`/valhalla/…/runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/`, every one suffixed
`.mp4.quarantined`. **NOT MEASURED — requires a cluster run.**

## 2.2 GENERATION time or MEASUREMENT time? — **BOTH, and the split is the whole answer**

The question decides whether this is a live gate risk. It is **three** paths, not two, and they have
different answers. Quoting the call sites:

### Path 1 — the G0c composite. **SOURCE pixels only. NOT exposed. By placement, and it is pinned.**

`scripts/robot_composite.py:1567`:

```python
masks = np.stack([np.asarray(context.masker.mask(frame), dtype=bool) for frame in frames])
```

`frames` here is `src`, decoded from `source_video`. **[A] V9 §5.1** states it and names the guard:

> *"**G0c's composite is unexposed, and by placement rather than by luck.** The mask that decides
> which pixels are composited is made from the SOURCE frame — `composite_clip` masks `src`, never
> `gen` — and that is where the predicate's claim is true. This is pinned by
> `test_the_composite_takes_its_mask_from_the_source_frame_and_only_from_there`, which fails if a
> future edit ever masks a generated frame and composites that."*

**[M]** The source corpus is where the predicate holds: max warm fraction **2.904 %** over the 382
committed audit frames and **2.802 %** over the 169 local-CPU frames (computed in §1.2 above),
against a 10 % bound. **So the pixels the generator is allowed to touch are decided by a reference
that is working. This half is not a live gate risk.**

### Path 2 — two GENERATED-pixel call sites. **Exposed. Neither is a gate.**

**2a.** `scripts/robot_composite.py:1642`, inside `composite_clip`'s loop:

```python
if index % context.iou_stride == 0:
    ious.append(mask_iou(mask, np.asarray(context.masker.mask(gen[index]), dtype=bool)))
```

That reaches `Sam2RobotMasker.object_grounding_iou` (`robot_composite.py:839`) →
`module.object_color_reference(frame)` on a **restyled** frame. PR-08 §6 governs it:

> *"Robot-mask IoU between source and generated is still recorded, **as a diagnostic on the
> generator, never as a gate**."*

The file already knows: `robot_composite.py:1619-1626` differences the filter counters **before**
this loop, with the comment *"the IoU diagnostic below runs the masker over the GENERATED frames
too, and the object filter's behaviour there is a different question … Pooling the two would make a
block that answers neither."* **So the per-clip record is clean; the lifetime counters are not, and
the number in `ious` is produced by a filter that may have mis-fired.**

**2b.** `scripts/probe_hallucination.py:988`:

```python
gen_mask = np.asarray(masker.mask(generated[i]), dtype=bool)
```

This is `T40_RULE_V8`'s instrument. **[A] V9 §5.1** already says what it costs:

> *"**V9 therefore changes V8's instrument for any FUTURE probe run.** … on the five non-warm styles
> it will not. **A probe run under V9 must therefore report the filter's counters per style**, or
> its count is a mixture of two instruments."*

**[I]** Direction on `train-01-oak-tungsten`: the reference is 37–56 % of the panel while a robot
mask is ~10 % and an apple mask ~0.6 %, so the symmetric IoU of any candidate against it is bounded
well below `ROBOT_MASK_OBJECT_MAX_IOU = 0.70`. The filter therefore **under-fires**: an
apple-grounded candidate is *kept* as a robot, inflating V8's `candidate_invention` count. That is
the conservative direction (a false alarm, not a silent pass) — which matches T-040's own reading
that *"most of `H` is very likely the masker defect"*. **It is still a mixture of two instruments
and must not be read as a rate.**

### Path 3 — the one V9 does not cover: `apple_sam2.segment()` on RESTYLED frames. **This IS a gate path.**

G0b's restyled side segments restyled clips with the same adapter:
`scripts/run_g0_gates.py:2555` (`--restyled-clips`) → `measure_geom_tol.episode_centroids_from_video`
→ `measure_geom_tol.py:945` `raw = module.segment(...)` → `apple_sam2.py:2044`
`reference = object_color_reference(frame)`.

**So the warm reference runs on restyled pixels inside the G0b gate.** **[I] + [M]:** on a
warm-scene style it will exceed the 0.10 bound on essentially every frame (my measurement: 56 of 56
generated panels of `train-01-oak-tungsten`), `reference_is_object_scale` returns `False`
(`apple_sam2.py:2056`), and **every frame is refused** into
`MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`.

**Is that a live gate *risk*? No — V10 already converted it into a loud refusal.** **[A]** V10 §5.2:

> *"**V10 can only refuse, so it can only remove frames** … On anything else, the effect is a loud
> refusal rather than a quiet number. A style the reference does not fit refuses every frame,
> coverage collapses, and both harnesses and `run_g0_gates` refuse at the coverage floor — and now
> the counter beside it says which of the two reasons it was."*

`MIN_COVERAGE_DEFAULT = 0.90`. **[I]** A `train-01-oak-tungsten` restyled side would land at
coverage ≈ 0 and refuse. **That is a yield problem, not a wrong number** — the same shape as G0c's
99.2 % refusal, which
`PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §3 already classifies as *"a pipeline that
starts, costs everything, and delivers almost nothing."*

### The explicit answer

> **The reference runs at BOTH moments, and the distinction is between which *pixels* it is handed,
> not which *phase* it is in.** The one place it decides pixel content — G0c's composite — is handed
> SOURCE frames only and is unexposed. Every path handed RESTYLED frames either (a) produces a
> diagnostic §6 forbids gating on, or (b) produces a refusal, never a wrong measurement. **There is
> no path on which a mis-firing reference produces a plausible-looking gate number.** It produces a
> refused clip or an unqualified diagnostic. **So this is a measurement-and-yield concern, not a
> live gate risk.**

## 2.3 A defect I found while checking the call sites

**`scripts/estimators/apple_sam2_video.py:464-467` applies V10's defect-1 fix and NOT its defect-2
fix:**

```python
elif apple_sam2.mask_validity_reference_is_defined():
    reference = apple_sam2.object_color_reference(frames[i])
    if apple_sam2.mask_validity_iou(mask, reference) < apple_sam2.MASK_VALIDITY_MIN_IOU:
        WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
```

There is no `reference_is_object_scale` check. **[I]** `WOULD_HAVE_BEEN_REFUSED_FRAMES` is meant to
say *"how many frames the per-frame arm would have refused"* — the propagation-arm comparison that
discharged `apple_sam2`'s seventh blocker on 2026-08-27. On any capture with a warm background,
`segment()` refuses via the object-scale branch (`apple_sam2.py:2056`) while this counter only sees
the IoU branch, so **the two arms' refusal counts are not the same quantity**. On the MuJoCo capture
this arm actually ran on the exposure is zero (§2.4 below), so the impact today is nil — but the
counter's *meaning* is wrong, and the next capture with a warm background is where it costs a
reading.

## 2.4 Does REPAIR B block the sprint (§8 items 3/4)? — **NO, and it is measured, not argued**

**[M]** `EST_DRIFT_P95` (§8 item 4's second half) was measured on the MuJoCo capture with **zero**
reference misfires of any kind:

```bash
cd /home/humanoid/develop/wam && .venv/bin/python -c '
import json; d=json.load(open("runs/pr08-est-drift/EST_DRIFT-mujoco-s60-f720.json"))
'   # -> n_frames_mask_refused 0 / no_reference 0 / reference_not_object_scale 0 / p95 0.23609
```

| artifact | `mask_refused` | `no_reference` | `reference_not_object_scale` | `est_drift_p95_px` |
|---|---|---|---|---|
| `EST_DRIFT-mujoco-s60-f720.json` | **0** | 0 | 0 | 0.23609 |
| `EST_DRIFT-mujoco-s20-f240.json` | **0** | 0 | 0 | 0.20993 |
| `EST_DRIFT-mujoco-trajectory-f480.json` | **0** | 0 | 0 | 0.29077 |

**[M]** `GEOM_TOL` (§8 item 4's first half) was measured on the **source** corpus, where the
predicate holds: `runs/pr08-geom-tol/pr08_geom_tol.json` → `n_segment_calls 171625`,
`n_frames_mask_refused 36`, `n_frames_mask_refused_no_reference 0`, `GEOM_TOL_px 0.47857992`.
36 / 171 625 = **0.021 %**, and **[A]** `PR-08-RESULT-2026-08-26-the-empty-mask-half-refuses-91-
percent-not-99-and-that-is-36-clips-not-one.md` accounts for those 36.

§8 item 3 (throughput on an H200) never touches the reference at all.

**REPAIR B IS DOWNSTREAM. LABEL IT NOT BLOCKING.** What it *does* block is reading a G0b or a V8
number **on a warm-scene style** — which is a post-generation question.

## 2.5 The minimal fix

**The reference itself cannot be repaired by a session, and that is a measured conclusion rather
than a shrug.** Both candidate fixes are already rejected on evidence:

- **Style-aware / per-style predicates** — V10 §3.1: 25 coined numbers with no measured gap, in the
  gate path. `styles.toml` commits 16 style-instances; a predicate per style is 16×3 numbers nobody
  can read off anything.
- **The paired SOURCE frame's own mask as the reference** (V9 §5.4's own proposal) — V10 §3.2
  rejects it with three independent objections, the load-bearing one being arithmetic:
  `--g0b-percentile` **defaults to 100** (`run_g0_gates.py:2580`, and stated at `:110`; V10 §3.2
  cites `:2468`, which is line drift since 2026-08-24 — **[M]** verified at `19826cc`), so the gate statistic is the
  **maximum** per-frame displacement; a filter that refuses a generated mask when it disagrees with
  its paired source mask *"refuses exactly the frames whose displacement is largest"*, while the run
  stays inside a 10 % coverage floor. **The proposal converts G0b failures into dropped frames, in
  the gate's own most sensitive statistic.** That is fail-open arriving through the repair.

**What IS repairable now is the two unguarded generated-pixel call sites.** V10 §5.4 exported
`reference_is_object_scale()` and `reference_frame_fraction()` *specifically* so this could be done
in one place, and deliberately did not wire it:

> *"That file belongs to V9 and to whoever signs it, and this workstream did not touch it. What V10
> does is export `reference_is_object_scale()` … so that if the exposure is ever closed there is
> **exactly one definition of 'this reference is applicable here'** in the repository."*

### Direction of the fix, chosen deliberately

**The filter's drop decision is NOT changed.** Changing it would move the bias in a direction nobody
has measured: dropping more candidates shrinks the robot mask (→ generated manipulator survives,
G0c's defect), keeping more admits the apple (→ silent pass, V9's defect). **A filter that cannot
decide must not decide harder in either direction — it must say it could not decide.** So the fix is
purely one of *recording*, and it makes the misfire visible where it is currently invisible.

```diff
--- a/scripts/robot_composite.py
+++ b/scripts/robot_composite.py
@@ -591,6 +591,7 @@
     _COUNTERS = (
         "frames_masked",
         "detections_segmented",
         "detections_dropped_as_object",
         "frames_with_a_dropped_detection",
         "frames_emptied_by_the_filter",
         "frames_with_no_object_reference",
+        "frames_with_scene_scale_reference",
     )
@@ -616,6 +617,13 @@
             "max_iou": float(ROBOT_MASK_OBJECT_MAX_IOU),
             "reference": self._object_reference_name(),
+            # PR-08 V10 §5.4 -- the exposure V10 named here and did not close. A non-zero count
+            # means this masker was asked to arbitrate against a predicate covering more of the
+            # frame than an object can, so the IoUs beside it are not a reading of anything.
+            # MEASURED on job 189926's train-01-oak-tungsten panels: the reference is 37.2-56.4 %
+            # of the panel across 56 generated panels of two episodes, against a bound of 10 %.
+            "reference_applicable": self.filter_counters["frames_with_scene_scale_reference"] == 0,
             **{name: int(self.filter_counters[name]) for name in self._COUNTERS},
         }
@@ -827,7 +827,7 @@
         module = self._estimator()
-        for name in ("object_color_reference", "mask_validity_iou"):
+        for name in ("object_color_reference", "mask_validity_iou", "reference_is_object_scale"):
             if getattr(module, name, None) is None:
@@ -839,6 +839,17 @@
         reference = module.object_color_reference(frame)
         if not reference.any():
             self.filter_counters["frames_with_no_object_reference"] += 1
+        # PR-08 V10 §5.4, wired at last. `reference_is_object_scale` is reached rather than
+        # restated for `object_color_reference`'s reason exactly: one definition of "this reference
+        # is applicable here", in the module V9 already reaches into.
+        #
+        # THE DECISION IS NOT CHANGED, ONLY RECORDED, AND THAT IS THE POINT. Dropping harder shrinks
+        # the robot mask and lets generated manipulator through (G0c's defect); dropping less admits
+        # the apple (V9's defect). Nobody has measured which is worse here, so this refuses to
+        # choose and refuses to be silent instead: the count reaches filter_record(), which
+        # composite_clip differences into the per-clip record, and `reference_applicable` goes false.
+        if not module.reference_is_object_scale(reference):
+            self.filter_counters["frames_with_scene_scale_reference"] += 1
         return np.asarray(
             [float(module.mask_validity_iou(m, reference)) for m in masks], dtype=np.float64
         )
```

```diff
--- a/scripts/estimators/apple_sam2_video.py
+++ b/scripts/estimators/apple_sam2_video.py
@@ -463,8 +463,17 @@
         elif apple_sam2.mask_validity_reference_is_defined():
             reference = apple_sam2.object_color_reference(frames[i])
-            if apple_sam2.mask_validity_iou(mask, reference) < apple_sam2.MASK_VALIDITY_MIN_IOU:
+            # BOTH of segment()'s refusal branches, in segment()'s own order, because this counter
+            # claims to say what the PER-FRAME ARM would have done and the two arms' p95s are only
+            # comparable if that claim is true. PR-08 V10 added the object-scale branch to
+            # segment() (apple_sam2.py:2056) and this module was only mirroring the IoU one, so on
+            # any capture with a warm background it under-counts the per-frame arm's refusals.
+            if not apple_sam2.reference_is_object_scale(reference):
+                WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES += 1
+                WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
+            elif apple_sam2.mask_validity_iou(mask, reference) < apple_sam2.MASK_VALIDITY_MIN_IOU:
                 WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
```

### Does REPAIR B's fix touch the contract? **NO.**

- `SEGMENTER_CONTRACT` is unchanged — no field added, removed or altered.
- `ESTIMATOR_VERSION` is unchanged — no token moves, so `run_g0_gates.instrument_disagreements`
  keeps comparing existing artifacts.
- `configs/transfer25/pr08_geom_tol.json` is unchanged.
- `MASK_VALIDITY_MIN_IOU`, `MASK_VALIDITY_REFERENCE`, `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION`,
  `ROBOT_MASK_OBJECT_MAX_IOU`, `GATE_QUALIFIED`, `GATE_QUALIFICATION_BLOCKERS` are unchanged.
- **No mask changes, no candidate's keep/drop decision changes, no number changes.** Only two
  counters appear.

**One caveat I will not hide:** `robot_composite.SEGMENTER_IDENTITY_FIELDS` keys the mask cache and
the committed area bound (`configs/transfer25/pr08_robot_mask_area.json`). This diff adds nothing to
`provenance()`, so **[I]** no cached mask and no area bound is invalidated. That should be asserted
by a test rather than believed — the third test below does it.

**Whether this repair is enough is a rule question, not a code question.** V9 §5.1's demand — *"a
probe run under V9 must therefore report the filter's counters per style"* — is satisfied by this
diff only for a probe that is re-run. Job 189926's `H` verdict was produced by the **pre-V9** masker
and V9 licenses no re-reading of it. **This diff makes the next run legible; it repairs no
existing number.**

## 2.6 A pytest that would have caught the defect

```diff
--- a/tests/test_robot_composite_object_filter.py
+++ b/tests/test_robot_composite_object_filter.py
@@
+def _warm_scene_frame(h=240, w=320) -> np.ndarray:
+    """A restyled frame's shape: a WARM TABLE filling most of the picture and a COLD green apple.
+
+    This is `train-01-oak-tungsten` as the predicate sees it, at the measured proportions rather
+    than at invented ones. MEASURED on job 189926's contact sheets, this session (2026-08-27), over
+    56 generated panels of two episodes: `object_color_reference` returns 37.18-56.40 % of the panel
+    on train-01-oak-tungsten, against 2.09-4.39 % on train-02-linen-overcast and a maximum of
+    2.904 % over the 382 committed source frames of runs/pr08-mask-audit/MASK_AUDIT.json. The bound
+    is apple_sam2.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10, and 0.40 is inside the measured
+    misfire population, not near its edge.
+    """
+    frame = np.zeros((h, w, 3), dtype=np.uint8)
+    frame[:, :, 2] = 200                      # cold everywhere: b high, r low
+    frame[int(h * 0.6):, :] = (190, 120, 60)  # warm oak: r>90, r-b=130>50, sat=(190-60)/190=0.68
+    frame[20:40, 20:40] = (40, 200, 60)       # a bright green apple: r=40, fails r>90
+    return frame
+
+
+def test_the_filter_records_that_it_could_not_decide_when_the_reference_is_the_scene(masker, monkeypatch):
+    """THE DEFECT, STATED AS A TEST. PR-08 V9 §5.4 / V10 §5.4, open since 2026-08-23.
+
+    `object_color_reference` justifies itself with "the only saturated warm thing in any of these
+    frames is the fruit". That is a claim about AppleToPlate's REAL pixels, and PR-08's committed
+    prompts change the table and the fruit on purpose. On a warm-table restyle the predicate does
+    not go quiet -- it moves to the table -- and this masker then scores every robot candidate
+    against a TABLE. V10 §5.4 exported reference_is_object_scale() precisely so this call site could
+    see it, and deliberately did not wire it. This test fails until it is wired.
+
+    IT ASSERTS RECORDING, NOT A CHANGED DECISION, and that is deliberate. Dropping harder shrinks
+    the robot mask and lets generated manipulator through (the defect G0c exists to exclude);
+    dropping less admits the apple (the defect V9 exists to exclude). Nobody has measured which is
+    worse on restyled pixels, so the only defensible action is to say the filter could not decide.
+    """
+    frame = _warm_scene_frame()
+    reference = masker._module.object_color_reference(frame)
+    fraction = masker._module.reference_frame_fraction(reference)
+    assert fraction > masker._module.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION, (
+        f"the fixture must reproduce the measured misfire; got {fraction:.4f}"
+    )
+    assert masker._module.reference_is_object_scale(reference) is False
+
+    masks = np.zeros((1,) + frame.shape[:2], dtype=bool)
+    masks[0, 150:200, 100:160] = True                       # a plausible robot candidate
+    keep = masker.object_grounding_keep(frame, masks)
+
+    record = masker.filter_record()
+    assert record["frames_with_scene_scale_reference"] == 1
+    assert record["reference_applicable"] is False, (
+        "an IoU scored against a predicate covering 40 % of the frame is not a reading of anything, "
+        "and the record has to say so"
+    )
+    # THE DECISION IS UNCHANGED. This test is not a licence to alter which candidates survive.
+    assert keep.tolist() == [True]
+
+
+def test_the_filter_says_nothing_when_the_reference_IS_object_scale(masker):
+    """The counterpart, so the check cannot be satisfied by always reporting inapplicable.
+
+    A source frame -- max 2.904 % over runs/pr08-mask-audit/MASK_AUDIT.json's 382 frames, and
+    2.09-4.39 % on train-02-linen-overcast where the filter demonstrably works -- must leave the new
+    counter at zero and `reference_applicable` true.
+    """
+    frame = _warm_scene_frame()
+    frame[int(frame.shape[0] * 0.6):, :] = (0, 0, 200)   # take the warm table away
+    frame[20:40, 20:40] = (220, 30, 30)                  # leave a warm apple: ~2.1 % of the frame
+    fraction = masker._module.reference_frame_fraction(
+        masker._module.object_color_reference(frame))
+    assert fraction <= masker._module.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION
+
+    masks = np.zeros((1,) + frame.shape[:2], dtype=bool)
+    masks[0, 150:200, 100:160] = True
+    masker.object_grounding_keep(frame, masks)
+    record = masker.filter_record()
+    assert record["frames_with_scene_scale_reference"] == 0
+    assert record["reference_applicable"] is True
+
+
+def test_the_new_counter_does_not_touch_the_segmenter_identity(masker):
+    """A cached mask and the committed area bound may not be invalidated by a recording change.
+
+    `SEGMENTER_IDENTITY_FIELDS` keys MaskCache.key() and load_area_bound()'s cross-check, so a field
+    that leaked into provenance() would silently invalidate every cached mask and
+    configs/transfer25/pr08_robot_mask_area.json. Asserted rather than believed.
+    """
+    provenance = masker.provenance()
+    assert "frames_with_scene_scale_reference" not in provenance
+    assert "reference_applicable" not in provenance
+    for field in rc.SEGMENTER_IDENTITY_FIELDS:
+        assert field in provenance, "the identity fields themselves must not have moved"
```

And for the propagation arm (§2.3):

```diff
--- a/tests/test_apple_sam2_video_propagation.py
+++ b/tests/test_apple_sam2_video_propagation.py
@@
+def test_the_would_have_been_refused_counter_mirrors_BOTH_of_segment_s_refusal_branches():
+    """The two arms' p95s are comparable only if this counter says what the per-frame arm would do.
+
+    `segment()` refuses on TWO branches (scripts/estimators/apple_sam2.py:2056 and :2061) since
+    PR-08 V10, and this module mirrored only the second. On a capture with a warm background the
+    per-frame arm refuses via the object-scale branch and this counter does not see it, so the
+    propagation comparison is between two different refusal populations.
+
+    Read off the SOURCE rather than executed, because reaching the counter needs SAM 2's video
+    predictor and a decoded capture, and a test that skips when those are absent is not a test.
+    """
+    source = pathlib.Path("scripts/estimators/apple_sam2_video.py").read_text(encoding="utf-8")
+    assert "reference_is_object_scale" in source, (
+        "WOULD_HAVE_BEEN_REFUSED_FRAMES must mirror segment()'s object-scale branch, not only its "
+        "IoU branch -- otherwise it under-counts the per-frame arm on any warm-background capture"
+    )
```

---

# 3. Defects found (each one would waste a cluster job or invalidate an artifact)

1. **`object_text_prompt` is a single-valued contract field while §6 gates two labels**
   (`configs/transfer25/pr08_geom_tol.json`; `apple_sam2.py:596`; `run_g0_gates.py:1512-1519`,
   `1880`; `measure_geom_tol.py:1921-1930`). **[I], traced through source.** Even a perfectly
   repaired plate filter yields a G0b run that is measured and then disqualified on
   `contract.object_text_prompt: 'plate.' vs 'apple.'`, and a plate pass pointed at the committed
   document is refused outright by `merge_committed_contract` (exit 2). **Cost if unnoticed: the
   entire GPU spend of a plate G0b pass, discarded at the cross-check.** This is not named in V6,
   V9, V10, or T-040.
2. **`apple_sam2_video.py:464-467` mirrors only one of `segment()`'s two refusal branches**
   (§2.3). **[A], read off source.** Makes the propagation-vs-per-frame comparison — the evidence
   that discharged the seventh gate-qualification blocker on 2026-08-27 — a comparison of two
   different refusal populations on any warm-background capture. Zero impact on the MuJoCo capture
   that has actually run (**[M]**: 0 refusals of any kind in all three `EST_DRIFT-mujoco-*.json`).
3. **The V10 misfire is a property of the STYLE, not of one episode** (§2.1). **[M].** V10 §4.1
   measured 12 generated panels of `episode_000000`; I measured **56 generated panels across two
   episodes**, all over the bound (37.18 – 56.40 %). V10 §5.1's stated claim — *"on the twelve
   panels of one style, on one episode"* — can now honestly be widened to two episodes. It is still
   **not a corpus rate and still one style of sixteen.**
4. **`robot_composite.object_grounding_iou` runs the reference on generated pixels with no
   applicability check** (`robot_composite.py:839-843`), three days after V10 exported
   `reference_is_object_scale()` for exactly that call site. **[A] + [M].** Not a gate (§6 forbids
   gating on it) but it silently produces a diagnostic number with no indication that the instrument
   was inapplicable.

---

# 4. What is runnable here, what needs the cluster, what needs a signature

## 4.1 Runnable on this workstation today, no cluster, no money

```bash
# (a) reproduce REPAIR A's current refusal - ~2 s, no weights, no network
cd /home/humanoid/develop/wam && WAM_PR08_OBJECT_PROMPT="plate." .venv/bin/python -c "$(cat <<'PY'
import sys, types, importlib.util, numpy as np
for n in ("transformers","sam2","torch"):
    if importlib.util.find_spec(n) is None: sys.modules.setdefault(n, types.ModuleType(n))
s = importlib.util.spec_from_file_location("apple_sam2","scripts/estimators/apple_sam2.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("defined?", m.mask_validity_reference_is_defined())
try: m.segment(np.zeros((480,640,3),dtype=np.uint8))
except Exception as e: print("RAISED:", type(e).__name__)
print("SEGMENT_CALLS", m.SEGMENT_CALLS, "MASK_REFUSED_FRAMES", m.MASK_REFUSED_FRAMES)
PY
)"

# (b) REPAIR A's root cause, from the two committed audits - ~1 s
cd /home/humanoid/develop/wam && .venv/bin/python -c '
import json
for p in ("runs/pr08-mask-audit/MASK_AUDIT.json","runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json"):
    f=json.load(open(p))["frames"]
    lo=[r for r in f if r["warm_apple_iou"]<0.10]
    print(p, len(f), "refused:", len(lo),
          "plate_overlap:", min(r["plate_overlap_fraction"] for r in lo),
          "min kept iou:", min(r["warm_apple_iou"] for r in f if r["warm_apple_iou"]>=0.10),
          "max warm frac:", round(max(r["warm_apple_px"] for r in f)/307200,5))'

# (c) REPAIR B, the whole misfire table, from the committed contact sheets - ~20 s
#     (the full script is in section 2.1 of this document)

# (d) the existing suites, unchanged, as the baseline any diff above must not break
cd /home/humanoid/develop/wam && .venv/bin/pytest -q \
  tests/test_apple_sam2_estimator.py \
  tests/test_robot_composite_object_filter.py \
  tests/test_apple_sam2_video_propagation.py
```

## 4.2 Needs a cluster run — **NOT MEASURED, and I did not try to get it**

The one measurement both V6 §5.3, V9 §5.4 and V10 §5.1 have now asked for three times:

> One `segment()` pass over a few hundred frames of **each of the 16 committed style-instances'**
> restyle, recording per style: `n_frames_mask_refused`, `n_frames_mask_refused_no_reference`,
> `n_frames_mask_refused_reference_not_object_scale`, and the distributions of
> `MASK_VALIDITY_IOU` and `MASK_VALIDITY_REFERENCE_FRACTION`.

**Why it cannot run here:** the only restyled clips are at
`/valhalla/…/runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/`, suffixed
`.mp4.quarantined`, and they cover **2 styles of 16**. The other 14 styles have **never been
generated** — generating them is itself gated by §8, so this measurement cannot precede the very
thing it would inform. **[I]** That is a genuine ordering problem in PR-08 and it is worth naming to
the owner rather than working around.

**I have deliberately not written an sbatch, estimated GPU-h, or drafted a submission.** Doing so
would be proposing to spend the allocation on a measurement whose scope depends on a §8 decision
nobody has taken.

## 4.3 Needs the project owner's signature

1. **REPAIR A's arming.** A new rule version (`PR-08-V20`-shaped) that decides whether §6's plate
   half is measured **unfiltered**, and edits `configs/transfer25/pr08_geom_tol.json` to say so.
   Diff 2 in §1.4 is the shape; the decision is not a session's.
2. **REPAIR A's second lock.** The same version must decide what `object_text_prompt` means in a
   contract that gates two labels — a per-label map, or an exemption in
   `run_g0_gates.contract_disagreements`. **This is a change to the contract's SHAPE and it
   invalidates the field-for-field cross-check as currently written.**
3. **`PR-08-V10` is `ADOPTED 2026-08-24`** per its own provenance block, but its §8 records the
   adoption as *"delegated to a Claude Code session … **Reversible: the owner delegated without
   having seen the recommendation**."* Everything in §2 rests on V10 being in force. Worth
   confirming before any of it is quoted.
4. **REPAIR B's diffs touch no contract and no rule**, so they are ordinary code changes — but
   whether V9's exposure is considered *closed* by recording it (rather than by changing the
   decision) is V9's signer's call, and V9 is **UNSIGNED**.

---

# 5. What this document does not license

Nothing here licenses generation, training, or quoting any PR-08 number as a result. `PR-08 §1`'s
prohibition is untouched. `GATE_QUALIFIED` is still `False` and neither repair moves it — REPAIR A
is downstream of it and REPAIR B changes no gate number. No file under
`/home/humanoid/develop/wam` was modified. No job was submitted and no cluster was contacted.

---

# Adversarial re-read

**Written 2026-08-27 by a second session, read-only against `/home/humanoid/develop/wam` at
`19826cc` (working tree clean: `git status --porcelain` empty, `git diff --stat` empty). No file
under the repo was modified, no cluster was contacted, no job was submitted.**

Tags below are this session's own: **[M]** measured now, **[A]** read off a committed artifact or
tracked source, **[I]** inference.

## Verdict

**The document does not survive.** Four load-bearing claims fail, two of them tagged `[M]`:

| # | claim | status |
|---|---|---|
| **R1** | §2.4's EST_DRIFT table — "`n_frames_mask_refused` **0**" on all three MuJoCo artifacts `[M]` | **REFUTED.** The run-scoped values are **25, 12, 1**. The zeros are `counters_at_start_of_run`. |
| **R2** | §1.5 — item 4 is held open by "a residue/signature question, not a plate question" `[M]` | **REFUTED.** `GEOM_TOL = 0.4786` is *already* out of contract at HEAD and needs a **cluster re-measurement**, not a signature. |
| **R3** | §2.5's `apple_sam2_video.py` diff is a safe recording-only change | **REFUTED.** It raises `UnboundLocalError` on the first frame it is meant to count, and its own test cannot see that. |
| **R5** | "**56 of 56** generated panels over the bound" `[M]` | **REFUTED.** 55 of 56, contradicted by the document's own §2.1 table; the 56th is a blank padding tile. |

What survives is listed at the end, and a good deal does.

---

## R1 — §2.4's EST_DRIFT table reads the wrong block, and it is the table the whole "REPAIR B is not blocking" conclusion rests on

§2.4 prints, tagged **[M]**:

> | `EST_DRIFT-mujoco-s60-f720.json` | **0** | 0 | 0 | 0.23609 |
> | `EST_DRIFT-mujoco-s20-f240.json` | **0** | 0 | 0 | 0.20993 |
> | `EST_DRIFT-mujoco-trajectory-f480.json` | **0** | 0 | 0 | 0.29077 |

and §2.3 leans on it: *"On the MuJoCo capture this arm actually ran on the exposure is zero (§2.4
below), so the impact today is nil"*, restated in the summary as *"measured: 0 refusals of any
kind"*.

**[M] Measured now**, walking every key of the three artifacts:

```bash
.venv/bin/python -c '
import json
def walk(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,p+"/"+k)
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,p+f"[{i}]")
    else: yield p,o
for f in ("s60-f720","s20-f240","trajectory-f480"):
    d=json.load(open(f"runs/pr08-est-drift/EST_DRIFT-mujoco-{f}.json"))
    print(f, [(p,v) for p,v in walk(d) if p.endswith("n_frames_mask_refused")])'
```

| artifact | `counters_at_start_of_run` | **`this_run`** | `counters_at_end_of_run` | `n_segment_calls` (this_run) |
|---|---|---|---|---|
| `EST_DRIFT-mujoco-s60-f720.json` | 0 | **25** | 25 | 720 |
| `EST_DRIFT-mujoco-s20-f240.json` | 0 | **12** | 12 | 240 |
| `EST_DRIFT-mujoco-trajectory-f480.json` | 0 | **1** | 1 | 480 |

`counters_at_start_of_run` is zero **by construction** — `scripts/measure_geom_tol.py:1378`
snapshots `ADAPTER_RUN_COUNTERS` at the start of a fresh interpreter. Quoting it as the run's
refusal count is not a measurement; it is reading the wrong half of a block whose two halves exist
precisely so the two claims cannot be confused (`measure_geom_tol.py:1482-1487`, and the tuple's own
comment at `:1236-1239`: *"'the filter fired on 12 frames of this run' and 'it has fired 12 times
since this interpreter started' are different claims"*).

So **3.47 %** of the s60-f720 capture and **5.00 %** of the s20-f240 capture were refused by the
validity filter — on the capture the document calls clean.

Two further corrections that fall out of the same walk:

- **The `reference_not_object_scale` zeros are not run-scoped for two of the three.** `[M]`
  `s60-f720` and `s20-f240` carry `n_frames_mask_refused_reference_not_object_scale` **only** at
  `/estimator_stats/adapter/`, i.e. in the non-differenced half
  (`measure_geom_tol.py:1485` — the `adapter` block is *everything not in* `ADAPTER_RUN_COUNTERS`).
  They ran a pre-`e518a84` adapter, before that counter joined the tuple. Only
  `trajectory-f480` states it per run. The document presents all three as per-run zeros.
- **The two arms already disagree on the only capture that ran.** `[M]`
  `runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json` →
  `/arm_comparison/propagator/stats/n_frames_the_colour_filter_would_have_refused = 0`, while the
  per-frame arm on the same capture recorded `this_run/n_frames_mask_refused = 1`. The document's
  defect 2 says the two arms' populations differ only *"on any warm-background capture"* and that
  today's impact is *"nil"*. Measured, they differ **1 vs 0 on this capture**. `[I]` The single
  refusal was an IoU-branch refusal (`reference_not_object_scale = 0`), so the missing branch is
  probably not its cause — but "nil" was asserted from a number that is not zero, and the honest
  statement is that nobody has checked which frame that was.

**Does the §2.4 conclusion survive the correction?** Partly. `no_reference = 0` and (for
`trajectory-f480`) `reference_not_object_scale = 0` are correct, so the *specific* claim "the warm
reference did not misfire on the MuJoCo capture" still stands on one of three artifacts. The claim
actually printed — "zero refusals of any kind", "the exposure is zero" — is false, and it is the
sentence the document uses to close the question.

## R2 — §1.5 gets §8 item 4 backwards: `GEOM_TOL = 0.4786` needs a cluster RE-MEASUREMENT, not a signature

§1.5 says, tagged **[M]**:

> *"item 4's two numbers already exist without a plate pass … Both carry `gate_qualified: false`,
> which is what actually keeps item 4 open — and **that is a residue/signature question**, not a
> plate question."*

**[M] Measured now**, by running the repository's own comparator over the landed artifact and the
committed document:

```bash
cd /home/humanoid/develop/wam/scripts && ../.venv/bin/python -c '
import json, importlib.util, sys
sys.path.insert(0,".")
s=importlib.util.spec_from_file_location("measure_geom_tol","measure_geom_tol.py")
m=importlib.util.module_from_spec(s); sys.modules["measure_geom_tol"]=m; s.loader.exec_module(m)
committed=json.load(open("../configs/transfer25/pr08_geom_tol.json"))["segmenter"]
landed=json.load(open("../runs/pr08-geom-tol/pr08_geom_tol.json"))["estimator_stats"]["adapter"]["segmenter_contract"]
print(m.contract_disagreements(landed, committed))'
```

```
[{'field': 'mask_validity_reference_max_frame_fraction', 'geom_tol': 0.1, 'this_run': None}]
```

The landed artifact's recorded contract has **15** fields; the committed document has **16**. And
`[M]` its instrument identity is stale too:
`runs/pr08-geom-tol/pr08_geom_tol.json` → `/estimator_stats/adapter/estimator_version` **ends**
`…;mask_val_min_iou=0.1` — no `mask_val_ref_max_frac` token, which the live module emits
(`[M]` reproduced above: the module's `ESTIMATOR_VERSION` ends `…;mask_val_ref_max_frac=0.1`).

`[A]` `scripts/measure_geom_tol.py:1921-1923` — *"A field present on one side and absent on the
other counts as a disagreement. Absence is not agreement anywhere else in this cross-check and it is
not here."* `[A]` and the commit that did it says so in its own message
(`git log -1 e518a84`): *"**GEOM_TOL = 0.4786 px was measured by an instrument that is no longer
HEAD, which is a second and independent reason it may not be committed, on top of
`gate_qualified: false`.**"*

**So §1.5 is wrong in the direction that costs GPU-hours.** Flipping `GATE_QUALIFIED` does not
qualify `0.4786`; a second, independent condition requires the 16-shard corpus pass to be **run
again** on the cluster under the current contract. §8 item 4's first half is blocked by a
measurement, and the document tells the owner it is blocked by a signature.

**And the same fact inverts §1.4's cost accounting.** §1.4 warns:

> *"the moment `SEGMENTER_CONTRACT` grows the key, every artifact measured under the old contract
> stops cross-checking … **The artifact at risk is real and is the sprint's own number.**"*

`[M]` That cost was already paid on 2026-08-24. The artifact is not *at risk* from a future
`PR-08-V20`; it is **already outside** the committed contract by exactly one field. A `V20` that
adds `mask_validity_unfiltered_labels` costs that artifact nothing further, because there is nothing
left to lose. The owner is being shown a price that has already been paid, attached to a decision it
does not actually gate.

## R3 — the `apple_sam2_video.py` diff in §2.5 does not run

The proposed hunk:

```python
+            if not apple_sam2.reference_is_object_scale(reference):
+                WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES += 1
+                WOULD_HAVE_BEEN_REFUSED_FRAMES += 1
```

`[A]` `WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES` does not exist:

- not a module global — the counters are declared at `scripts/estimators/apple_sam2_video.py:200-210`
  and the only refusal counter there is `WOULD_HAVE_BEEN_REFUSED_FRAMES` (`:208`);
- not zeroed by `reset_counters()` (`:213-220`);
- not reported by `stats()` (`:222-232`, which returns
  `"n_frames_the_colour_filter_would_have_refused": WOULD_HAVE_BEEN_REFUSED_FRAMES` and nothing
  else);
- **and the diff does not extend the `global` statement at `:405-406`**
  (`global WOULD_HAVE_BEEN_REFUSED_FRAMES, LAST_SEED_BOX`).

`[I]` Python therefore binds the new name as a **local** of `propagate`, and the augmented
assignment raises `UnboundLocalError: cannot access local variable
'WOULD_HAVE_BEEN_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES' where it is not associated with a
value` — **on the first frame whose reference is not object-scale**, i.e. exactly the
warm-background capture the diff exists to measure. The propagation arm would not under-count; it
would die mid-clip.

**And §2.6's test cannot catch it.** The proposed
`test_the_would_have_been_refused_counter_mirrors_BOTH_of_segment_s_refusal_branches` reads the file
as text and asserts `"reference_is_object_scale" in source`. That passes on a module that crashes.
A test that greps for a string is not a test of a counter — which is the document's own standard,
stated in §2.6's docstring (*"a test that skips when those are absent is not a test"*).

## R4 — REPAIR A's Diff 1 repeats, verbatim, the defect commit `e518a84` fixed

Diff 1 adds `n_frames_measured_unfiltered` to `stats()` and stops there.

`[A]` `scripts/measure_geom_tol.py:1228-1252` — `ADAPTER_RUN_COUNTERS` is the tuple that gets
differenced into `this_run` / `counters_at_start_of_run` / `counters_at_end_of_run`;
`:1485` puts **everything not in that tuple** into the `adapter` block, i.e. as a process
lifetime total. The tuple's own comment on the last field added to it:

> *"T40_RULE_V10's sub-case, added 2026-08-24 … It is a refusal INSIDE `n_frames_mask_refused`, so no
> coverage number above changes value — only the attribution does, **from a lifetime total of the
> process to this run's**."*

`[I]` A run measured with the filter declared off would therefore report its unfiltered-frame count
as a lifetime total sitting beside a run-scoped coverage number — the precise failure the repository
paid a commit to remove three days ago. Diff 1's own test
(`assert stats["n_frames_measured_unfiltered"] == 1`) reads `stats()`, so it passes while the
artifact is wrong. If a `V20` is ever written, the counter must join `ADAPTER_RUN_COUNTERS`, and
that is a **fifth** file the diff does not touch.

## R5 — "56 of 56" is 55 of 56, and the document's own table says so

The summary states *"56/56 over the bound (37.18-56.40 %)"*; §2.1 states *"**28 of 28** generated
panels over the bound"* for `episode_000001` and *"56 generated panels over the bound across two
episodes"*; defect 3 states *"56 generated panels across two episodes, **all** over the bound"*.

`[M]` I re-ran §2.1's script verbatim and reproduced every cell **exactly** — including the row that
contradicts the total:

```
episode_000000__train-01-oak-tungsten__probe__excluded_source_mask_nonempty.png
   n= 4 src   0.00-  2.32%  gen   0.00- 38.80%  over 3/4
```

12 + 12 + **3** + 12 + 12 + 4 = **55** of 56. The document's own §2.1 table prints `3/4` in that row
and then totals it as if it were `4/4`.

`[M]` Worse for the denominator: the missing panel measures **0.00 % on both halves**, and the same
0.00 % appears in `episode_000000__train-02-linen-overcast__probe__excluded_source_mask_nonempty.png`
at the same tile position. `[I]` That is a **blank padding tile** in a sheet with three real pairs,
not a restyled panel. So "56 generated panels" counts one tile that contains no pixels of anything;
the honest figure is **55 of 55 real generated panels**, and the method that produced "56" cannot
tell a padded tile from a measurement.

The direction of the finding is unaffected. The number quoted three times, in a document whose whole
method is that a number is worth nothing without the artifact, is wrong.

## R6 — arithmetic in a proposed test fixture

§2.6's `test_the_filter_says_nothing_when_the_reference_IS_object_scale` docstring:

> *"leave a warm apple: **~2.1 %** of the frame"*

`[M]` The frame is `240 × 320 = 76 800` px; `frame[20:40, 20:40]` is `20 × 20 = 400` px.
`400 / 76 800 = **0.52 %**`, four times smaller. The assertion (`fraction <= 0.10`) still passes, so
the test is not broken — but the fixture is advertised as being *"at the measured proportions rather
than at invented ones"* and the measured source proportions are 2.29–4.51 %. It is off the measured
population it claims to sit in.

## R7 — §1.2(a)'s two "independent" audits are the same episode twice

§1.2(a) presents the 382-frame audit (24 episodes) and the 169-frame local-CPU audit (12 episodes)
as two artifacts giving *"the same shape"*.

`[M]` Every zero-IoU row in both comes from **one** episode:

```bash
cd /home/humanoid/develop/wam && .venv/bin/python -c '
import json
for p in ("runs/pr08-mask-audit/MASK_AUDIT.json","runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json"):
    f=json.load(open(p))["frames"]
    low=[r for r in f if r["warm_apple_iou"]<0.10]
    print(p, len(low), sorted({r["episode"] for r in low}))'
# runs/pr08-mask-audit/MASK_AUDIT.json          12 ['episode_000094']
# runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json 9 ['episode_000094']
```

`[I]` The two audits are not independent corroboration of a corpus-wide plate confusion; they are
`episode_000094` sampled twice. §1.2(a)'s table does list only `episode_000094` rows, so nothing is
hidden — but "12 of 382 frames across 24 episodes" is how the summary reads it, and the correct
reading is "every instance is one episode". This **does not** touch §1.2(b), which is the load-
bearing half: V10 §1.1's matched control is a separate 20-frame pass on `episode_000000`, and I
verified it reproduces as quoted (`docs/preregistration/PR-08-V10-mask-validity-reference-scope.md:85-100`).

## R8 — §1.4 heads Diff 1 as inert and its own hunk makes that false

§1.4: *"**Diff 1 of 2** — `scripts/estimators/apple_sam2.py` (does NOT touch the committed JSON)"*,
and the constant's comment: *"**this constant changes nothing until a signed rule version puts a
label in it**; until then every path behaves exactly as it did under V10, **value for value**."*

But Diff 1 carries the hunk `f";mask_val_unfiltered={…} or '()'"` into `ESTIMATOR_VERSION`, and its
own test asserts `"mask_val_unfiltered=" in module.ESTIMATOR_VERSION`. `[M]` `ESTIMATOR_VERSION` is
recorded in every landed artifact — `/estimator_stats/adapter/estimator_version` in
`runs/pr08-geom-tol/pr08_geom_tol.json` and in all three `EST_DRIFT-mujoco-*.json` — and `[A]`
`run_g0_gates.instrument_disagreements` compares it between G0b's two sides. Moving the string is a
behaviour change on an existing path, by the module's own argument at
`apple_sam2.py:557-565`. §1.4 concedes this four paragraphs later (*"`ESTIMATOR_VERSION` also
moves"*) while attributing it to Diff 2. Both statements cannot stand; the header and the constant's
comment are the wrong ones.

## R9 — mis-citations (small, none reverses a meaning)

`[M]` at `19826cc`:

| cited | actually |
|---|---|
| `apple_sam2.py:1587` for *"the cloth and the plate are neutral to within two counts"* | `:1587` is *"the warm, saturated pixels in a frame that has some is not a mask of the fruit. On this corpus"*; the quoted phrase is at `:1588-1589` |
| `apple_sam2_video.py:464-467` for the 4-line `elif` block | the block is `:465-468`; `:464` is `EMPTY_PROPAGATED_FRAMES += 1` |
| `robot_composite.py:1642` for the 2-line `if index % context.iou_stride == 0:` block | `:1641` is the `if`; `:1642` is the `ious.append(...)` |
| `apple_sam2.py:2044` and `:2056` | correct |
| `run_g0_gates.py:294-300`, `:1339-1347`, `:1512-1519`, `:1770`, `:1826-1829`, `:1880`, `:2148`, `:2555`, `:2580`, `:110` | all correct |
| `measure_geom_tol.py:164-167`, `:945`, `:1921-1923`, `:2035` | all correct |
| V10 §3.2 cites `run_g0_gates.py:2468` for the `--g0b-percentile` default | confirmed drift; `:2580` at `19826cc`, default `100.0` at `:2582`, prose at `:110`. The document's correction is right |

## R10 — the ordering hazard the front asked about is never named

The document declines to draft a cluster submission for the per-style pass, which is right. It never
names the canonical PR-08 ordering trap for it.

`[A]` `GATE_QUALIFIED = False` (`apple_sam2.py:938`) is **baked into every artifact at measurement
time**: `stats()["gate_qualified"]` (`:1154`) lands as `/estimator_stats/adapter/gate_qualified`,
`/mask_method/gate_qualified` and `/gate_qualified`, and `[M]`
`runs/pr08-geom-tol/pr08_geom_tol.json` carries **17** `gate_disqualified_reasons` rows — one for
the merge and one per shard — all of them *"mask method … is not gate-qualified"*.

`[M]` And the flip is close: `GATE_QUALIFICATION_BLOCKERS` is now `()` (empty) while
`GATE_QUALIFIED` is still `False` — `[A]` `apple_sam2.py:643` says as much
(*"`GATE_QUALIFIED` IS STILL `False`"*), the seventh blocker having been discharged on 2026-08-27.

`[I]` Combined with **R2**, the honest ordering for §8 item 4 is **sign → flip `GATE_QUALIFIED` →
re-measure `GEOM_TOL` on the cluster**, and any pass submitted before the flip re-buys the same
disqualification it is trying to clear. The document's §4.2 does not mention it, and its §1.5 says
the opposite.

---

## What survives, and it is not nothing

Checked independently and **confirmed**:

- **[M] The `plate.` repro.** Reproduced value for value: `mask_validity_reference_is_defined()`
  `False`, `MaskValidityReferenceUndefined` raised, `SEGMENT_CALLS = 0`,
  `MASK_REFUSED_FRAMES = 0`. `apple_sam2.py:2010` does call `_require_mask_validity_reference()`
  before `SEGMENT_CALLS += 1` at `:2013`. §1.1 is correct including its premise-has-moved framing.
- **[M] REPAIR A's root cause is the FILTER.** 12 rows at exactly `0.0000` with
  `plate_overlap_fraction` 0.9748–0.9807 and `mask_area_px` 30 892–31 151; min IoU of every kept
  frame **0.7492**; local-CPU 9 rows, min kept **0.8415**; detection scores of the 12 are
  0.167157–0.309071. All reproduced. V10 §1.1's matched control reproduces as quoted. The
  three-way elimination (not the detector, not the prompt/thresholds, it is the filter) stands.
- **[M] REPAIR B's contact-sheet measurement.** Reproduced **cell for cell**, and it matches V10
  §4.1 (`:136`, `:312-316`) exactly: 40.51–56.40 vs 40.5–56.4; 4.27–4.39 vs 4.27–4.39; 4.51 vs 4.51.
  The extension to `episode_000001` is real and new.
- **[M] Max source warm fraction 2.904 % / 2.802 %** against the 0.10 bound. Reproduced.
- **[A] Defect 1 (`object_text_prompt` is single-valued while §6 gates two labels).** Traced and
  confirmed: `run_g0_gates.contract_disagreements` (`:1339-1347`) compares the **intersection** and
  `object_text_prompt` is present on both sides, so a `plate.` centroid dump does disagree;
  `merge_committed_contract` exists at `measure_geom_tol.py:2035`. Genuinely unnamed in V6, V9, V10,
  T-040. This is the document's best finding.
- **[A]+[M] REPAIR B's `robot_composite.py` diff touches no contract.** Verified rather than
  believed: `filter_counters` is `dict.fromkeys(self._COUNTERS, 0)` (`:602`), so adding a key is
  initialised; `provenance()` (`:670-757`) builds `object_grounding_filter` from
  `ROBOT_MASK_OBJECT_MAX_IOU` and the reference **name** only, never from the counters; and
  `load_area_bound` (`:1147`) compares only `segmenter_identity`'s fields. `[M]`
  `configs/transfer25/pr08_robot_mask_area.json` **does** now exist (27 Aug), so the in-source
  comment at `:416` claiming it is absent is stale — the document's caveat is right and the source
  is out of date.
- **[A] Defect 2 exists.** `apple_sam2_video.py:465-468` really does mirror only the IoU branch.
  (Its *stated impact* fails — see R1 — and the *proposed fix* fails — see R3.)
- **[M] The baseline suite.** `.venv/bin/pytest -q tests/test_apple_sam2_estimator.py
  tests/test_robot_composite_object_filter.py tests/test_apple_sam2_video_propagation.py` →
  **150 passed in 1.54 s**. The helpers the proposed tests use (`_install`, `_fresh_import`,
  `loaded`, `_frame(apple=None)`, the `masker` fixture which sets `instance._module`) all exist with
  the signatures assumed.
- **Method compliance.** No "runnable now" command touches the cluster, spends money, or mutates the
  repo — all four are read-only and I ran three of them. No `PR-08-V*` document is edited in place;
  `GATE_QUALIFIED` and `GATE_QUALIFICATION_BLOCKERS` are untouched; the contract change is correctly
  routed to a new signed version rather than made. §1.4's refusal to choose the
  `object_text_prompt` shape without the owner, and §2.5's refusal to change the filter's drop
  decision, are both the right call for a session to make.

## What the corrections change about the conclusions

- **"Neither repair blocks the sprint."** Still true *of the repairs* — but the stated reason for §8
  item 4 being open is wrong (**R2**), and the evidence offered for REPAIR B's innocence is the
  wrong block of the artifact (**R1**). The conclusion needs re-deriving from the right numbers
  before it is handed to the owner.
- **"There is no path on which a mis-firing reference produces a plausible-looking gate number."**
  Unaffected by R1–R5; the placement argument (`robot_composite.py:1567` masks `src`) and V10's
  refuse-only property both hold.
- **The proposed diffs.** REPAIR B's `robot_composite.py` half is sound. REPAIR B's
  `apple_sam2_video.py` half **must not be applied as written** (R3). REPAIR A's Diff 1 is
  incomplete in a way that reproduces a fixed defect (R4) and is not inert as advertised (R8).
