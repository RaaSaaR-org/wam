# PR-08 §4 — PROPOSED discharge of gate-qualification blockers 1 and 2

> ## SUPERSEDED 2026-08-26 — the discharge was applied, in AMENDED form
>
> This proposal was accepted and the edit made, but **not as the diff below prints it.** Three
> things had changed by the time it was applied, and a reader comparing this document against the
> file will find all three:
>
> 1. **The 13 `no_mask` verdicts are gone.** The proposal's residue (ii) — *"the reviewer's account
>    and the counters DISAGREE about those 13 frames and the disagreement is unresolved"* — was
>    resolved the same day. A blind re-read disagreed on exactly those 13 and on nothing else in 430
>    tiles, the instrument's own counters had already excluded `no_mask` arithmetically, and the
>    reviewer re-set two mis-defaulted sheets. **The applied entry records the correction as part of
>    the evidence rather than carrying it forward as residue**, and the tally is
>    `apple 363 / partial 7 / wrong_object 12 / no_mask 0`, not `350 / 7 / 12 / 13`.
> 2. **The per-stratum numbers moved with it** — `min_visibility` is 87 apple / 3 wrong_object /
>    2 partial, and the non-apple verdicts are 19 rather than 32, of which 18 are `episode_000094`.
> 3. **§4.4's objection — the strongest one, that the tiles were made by an adapter this file has
>    since replaced — was checked and does not hold.** Ten of the twelve `estimator_version` tokens
>    are character-identical across the audit, the full pass and today; every differing token is a
>    mask-**validity** field; and `segment()` draws the mask before the filter runs, so a validity
>    field can replace a drawn mask with all-False and can never alter one. The applied entry keeps
>    the residue that remains from this — *which frames are measured* is a different set — and drops
>    the part that was refuted.
>
> **This document is kept, unedited below this box, because the reasoning is the audit trail.**
> §§4.1–4.6 and §5 in particular are not restated anywhere else and are what made the decision
> reviewable. Read §5 especially: it argues that `GATE_QUALIFIED` may not flip in the same commit
> that shortens the tuple, and the applied edit follows it.
>
> Applied in the commit whose subject is *"blockers 1 and 2 close on their own named evidence, and
> carry their residue with them"*. Result documents:
> `PR-08-RESULT-2026-08-26-the-retry-blocker-2-named-never-fired.md`,
> `PR-08-RESULT-2026-08-26-the-recorded-verdicts-checked-against-a-blind-re-read.md`.

**Written 2026-08-26. THIS DOCUMENT APPLIES NOTHING.** It contains a diff of
`scripts/estimators/apple_sam2.py` as *text*, for the project owner to read, reject or apply by
hand. No source file was edited to produce it, no gate was flipped, no artifact was rewritten, and
nothing here is a claim that any blocker **is** discharged. The blockers are a person's to close;
this is the reviewable edit laid out so that closing them is one decision rather than an
afternoon's reconstruction.

---

## 0. What this is and is not

| | |
|---|---|
| proposes | moving `GATE_QUALIFICATION_BLOCKERS` entries **1** (*"NOBODY HAS LOOKED AT A MASK"*) and **2** (*"BOX_THRESHOLD / TEXT_THRESHOLD / the retry are unmeasured on AppleToPlate"*) into `GATE_QUALIFICATION_DISCHARGED`, verbatim, with the evidence, the reviewer, the date, the tally and the residue |
| does not propose | any change to blocker **3** (per-frame vs. propagation), which the diff leaves byte-identical and alone in the tuple |
| does not propose | any change to `GATE_QUALIFIED`. It stays `False`. See §5 |
| applies | **nothing.** No file in `scripts/`, `src/`, `configs/` or `runs/` is modified by this document |
| licenses | **no clip, no training run.** `T40_RULE_V1` §1 binds in full; §8 item 3 is open independently of anything here |
| evidence | `runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json` (reviewer `human`, 382/382 tiles), `docs/preregistration/PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md` (full pass, 171 625 frames) |

---

## 1. The rule this follows, and why the diff looks the way it does

`runs/pr08-mask-audit/MASK_AUDIT.json`, field `not_a_discharge`:

> *"THIS ARTIFACT DOES NOT DISCHARGE ANY BLOCKER. It is the evidence blockers 1 and 2 ask for,
> produced so that a person can look at it. `GATE_QUALIFIED` is read from the adapter and copied
> here; nothing in this script writes it. **Discharging a blocker is a reviewable edit to
> `GATE_QUALIFICATION_BLOCKERS` made by a person who has looked, and it moves the retired wording
> into `GATE_QUALIFICATION_DISCHARGED` with the evidence rather than deleting it.**"*

Three constraints follow and the diff in §3 obeys all three.

1. **A person who has looked.** Not this session. The diff is text; applying it is the act.
2. **Move, not delete.** Both retired entries appear in `GATE_QUALIFICATION_DISCHARGED` **verbatim
   and unparaphrased**, delimited by `>>>` / `<<<` markers so a reader can see where the old
   wording ends and the new evidence begins. This was checked mechanically, not by eye: the exact
   string values of `GATE_QUALIFICATION_BLOCKERS[0]` and `[1]` (1 845 and 1 252 characters) are
   substrings of the two new discharged entries.
3. **The evidence beside it.** Each new entry carries the artifact path, the reviewer, the date,
   the full tally — and the residue the evidence does *not* close, because a discharge that hides
   its own residue is a deletion with better manners.

The file's own docstring says the same thing in its own words, and is the reason the format above
is not an invention of this document:

> *"Conditions that HAVE been discharged move to `GATE_QUALIFICATION_DISCHARGED` with the evidence,
> rather than disappearing — a blocker that vanishes between two commits looks identical whether it
> was satisfied or deleted, and only one of those is allowed to shorten this list."*

---

## 2. The evidence, in numbers

### 2.1 Blocker 1's first limb: the human look

A person recorded as reviewer `human` (`established_by`) opened all **382** overlaid apple-mask
tiles of `runs/pr08-mask-audit/MASK_AUDIT.json` (job 189637, 24 of 402 episodes, 480×640) through a
review page and recorded **one verdict per tile**.

- Artifact: `runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json`, recorded `2026-08-25T23:44:20Z`.
- Route: `runs/pr08-review-page/review.saved-1787701138-dff8.html` →
  `runs/pr08-mask-audit/REVIEW_PAGE_INGEST.json` → `scripts/record_mask_audit_verdicts.py`.
- Coverage: **382/382 tiles, 34/34 stratum sheets**, `frames_reviewed_fraction: 1.0`.
- `blocker_1_named_strata.not_covered: []` — the sample spans all three strata blocker 1 names.

**Tally: apple 350, partial 7, wrong_object 12, no_mask 13.** By stratum:

| stratum | tiles | apple | partial | wrong_object | no_mask |
|---|---|---|---|---|---|
| `grasp` | 180 | **180** | 0 | 0 | 0 |
| `spanning` | 92 | **92** | 0 | 0 | 0 |
| `border` | 4 | **4** | 0 | 0 | 0 |
| `min_visibility` | 92 | 74 | 2 | 3 | **13** |
| `census` | 8 | 0 | 1 | **7** | 0 |
| `occluded` | 6 | 0 | 4 | **2** | 0 |

**A note on the recording granularity, because it is the first thing a sceptic should attack and it
survives.** `MASK_AUDIT_VERDICTS.json` records 34 sheet defaults plus 9 explicit exceptions, which
would be 43 acts of judgement stretched over 382 tiles. But `REVIEW_PAGE_INGEST.json` carries
`per_tile` with **382 individual verdicts** — the page held one verdict per tile and the per-sheet
form is a *derivation* handed to the recorder, recomputable from `per_tile` and tallying exactly
(350 / 7 / 12 / 13 on both sides). The compression is lossless. This objection is dead; the ones in
§4 are not.

### 2.2 Blocker 1's second limb is closed to us, and does not need to be open

Blocker 1 is a disjunction: the human look **and/or** a mask-vs-ground-truth IoU distribution.
`docs/preregistration/PR-08-NOTE-2026-08-25-the-mujoco-iou-cannot-discharge-blocker-1.md` establishes
that the MuJoCo IoU (median 0.9881, n = 480) cannot be retrofitted into the second limb, on
`PR-08-V5-ground-truth-route.md` §0's own text and on `docs/handoff.md` §3's ordering rule. That
finding is unaffected here: this proposal rests on the **first** limb alone, which is what the
note's §4 says the blocker terminates in.

### 2.3 Blocker 2's second conjunct: the full pass

`docs/preregistration/PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md`, over
`runs/pr08-geom-tol/shards/shard-{0..15}.json` — 16/16 shards, 402/402 episodes, **171 625 frames**,
one identical `mask_method.version` across all sixteen:

| | |
|---|---|
| `n_frames_retry_fired` | **0** |
| `n_frames_retry_recovered` | **0** |
| detection score p1 / median / p99 | 0.5489 / **0.8497** / 0.9117 |
| frames in `[0.15, 0.35)` | **92** of 171 625 (0.054 %) |
| frames below 0.15 | **0** |
| `n_frames_mask_refused` | 36, **all in `episode_000094`** |

The **(0.10, 0.10) retry never fired on any frame of the corpus.** The mechanism blocker 2 predicts
did not occur — a stronger statement than "we measured it and it was small".

---

## 3. The proposed diff

Against `71edc63`, `scripts/estimators/apple_sam2.py` (1 896 lines, unmodified in the working tree
at the time of writing). Validated with `patch -p1 --dry-run` on a **copy** in a scratch directory:
applies cleanly, parses, and yields `len(GATE_QUALIFICATION_BLOCKERS) == 1`,
`len(GATE_QUALIFICATION_DISCHARGED) == 6`, `GATE_QUALIFIED is False`. The four pre-existing
discharged entries are unchanged and still first; the remaining blocker is blocker 3, byte-identical.


```diff
--- a/scripts/estimators/apple_sam2.py
+++ b/scripts/estimators/apple_sam2.py
@@ -388,7 +388,8 @@
 # are ~30 900 px against a median apple of 6 185, they sit at 0.97-0.98 plate overlap, they score
 # 0.167-0.309 where the correct masks score a median 0.829 — and they produce a centroid, a
 # displacement and a p95 that all look exactly like measurements. That is the failure mode
-# GATE_QUALIFICATION_BLOCKERS's first entry names in as many words. In episode_000094 the segmenter
+# GATE_QUALIFICATION_BLOCKERS's first entry named in as many words until 2026-08-26, when
+# that entry moved into GATE_QUALIFICATION_DISCHARGED. In episode_000094 the segmenter
 # OSCILLATES between the two objects (f00149 plate -> f00150 apple, 471 px -> f00151 apple, 670 px
 # -> f00152 plate), so the corruption hits both tails at once: near-zero displacements while it is
 # locked on the stationary plate, and a recorded 245.9 px step at every switch.
@@ -622,42 +623,13 @@
 #: Every condition that has to be true before this pair may set ``EST_DRIFT_P95`` or ``GEOM_TOL``.
 #: Written out rather than summarised because :data:`GATE_QUALIFIED` is a claim, and a claim whose
 #: grounds are not written down gets flipped by whoever is in a hurry.
+#:
+#: ONE entry, since 2026-08-26. The two that stood above it — the human look, and the operating
+#: point on this corpus — are in :data:`GATE_QUALIFICATION_DISCHARGED` VERBATIM, with the
+#: evidence, the reviewer, the date and the tally beside them, and with the residue each one
+#: carries forward stated in the same breath. A shorter tuple is not a smaller question: read
+#: those two entries before reading this one as "almost clear".
 GATE_QUALIFICATION_BLOCKERS: tuple[str, ...] = (
-    "NOBODY HAS LOOKED AT A MASK. The 2026-08-21 wording of this blocker ('never executed, no "
-    "checkpoint staged') is withdrawn as stale: job 189583 staged all three checkpoints at the "
-    "pinned revisions and verified them, and job 189588 drove this adapter end to end over the "
-    "AppleToPlate corpus in the GEOM_TOL pilot — 720 frames, two passes, 480x640, coverage 1.0 on "
-    "both. CITATION CAVEAT, because a blocker tuple is the load-bearing record of what is and is "
-    "not established: 189583 is recorded in .mc/tasks/todo/T-040-*.md, but 189588 IS NOT RECORDED "
-    "ANYWHERE TRACKED IN THIS REPOSITORY — its artifact was not readable from the session that "
-    "wrote this line, and the job id is an untracked claim until GEOM_TOL_PILOT.json lands. It is "
-    "also evidence about a configuration THIS FILE HAS SINCE REPLACED: that pilot necessarily ran "
-    "at the old operating point (box_threshold 0.35, no retry branch), so it is weaker evidence "
-    "for the current adapter than its numbers suggest. So the module runs and produces output. "
-    "What that does NOT establish is that the output "
-    "is right: coverage 1.0 says a box was returned on every frame, not that it was the APPLE's "
-    "box, and this adapter's whole failure mode is a plausible mask on the wrong object (the "
-    "plate, the hand, the whole tabletop) which produces a centroid, a displacement and a p95 that "
-    "all look like measurements. Lowering BOX_THRESHOLD to upstream's 0.15 with a 0.10 retry — "
-    "correct, and required by §4 step 2 — makes coverage an even weaker witness than it was at "
-    "0.35, because more frames now get a box and none of them get checked. Discharged by: a human "
-    "looking at a sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame "
-    "frames, and the grasp), and/or a mask-vs-ground-truth IoU distribution from the Isaac capture "
-    "recorded beside the centroid displacement. Neither exists.",
-    "BOX_THRESHOLD / TEXT_THRESHOLD / the retry are unmeasured on AppleToPlate, and after 2026-08-22 "
-    "that is a narrower objection than it was. They are no longer 'upstream demo defaults we "
-    "happened to copy' (0.35/0.25): they are Cosmos-Transfer2.5's own operating point, read off its "
-    "sam2_model.py, which is precisely what §4 step 2 asks for. The choice-defect half of this "
-    "blocker is therefore DISCHARGED and inverted — measuring these on our corpus and moving them "
-    "to whatever reads best would MAKE this a different segmenter from the generator's, and the "
-    "budget would then be a budget for an error nobody commits. What survives is not a choice, it "
-    "is an unknown: nothing has measured what this operating point does on THIS corpus, and the "
-    "retry at (0.10, 0.10) buys detections by accepting weak ones, which on an occluded frame can "
-    "replace an honest all-False mask with a confident box on the wrong object. That inflates "
-    "coverage while degrading the mask, i.e. it hides itself in the one number the harness gates "
-    "on. Discharged by the same evidence as blocker 1, plus the recorded detection-score "
-    "distribution and retry counts (n_frames_retry_fired / n_frames_retry_recovered) from a full "
-    "pass, so the retry's contribution is visible rather than assumed.",
     "PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION, and it is the one difference left. "
     "Everything else in §4 step 2's 'the same segmenter' now matches Cosmos-Transfer2.5's "
     "sam2_model.py exactly — both checkpoints at pinned revisions, the 'apple.' phrase, "
@@ -720,16 +692,104 @@
     "nothing written) instead of being measured. measure_geom_tol has no such flag at all — its "
     "sam2 method takes the prompt from this module — so there is no longer a second place where the "
     "object is chosen. This module still cannot see the flag; it no longer has to.",
+    "2026-08-26 — blocker 1, 'NOBODY HAS LOOKED AT A MASK'. THE RETIRED WORDING FOLLOWS "
+    "VERBATIM AND UNPARAPHRASED BETWEEN THE MARKERS >>> and <<<, because a blocker that is "
+    "summarised on its way out is a blocker nobody can re-read. >>> "
+    "NOBODY HAS LOOKED AT A MASK. The 2026-08-21 wording of this blocker ('never executed, no "
+    "checkpoint staged') is withdrawn as stale: job 189583 staged all three checkpoints at the "
+    "pinned revisions and verified them, and job 189588 drove this adapter end to end over the "
+    "AppleToPlate corpus in the GEOM_TOL pilot — 720 frames, two passes, 480x640, coverage 1.0 on "
+    "both. CITATION CAVEAT, because a blocker tuple is the load-bearing record of what is and is "
+    "not established: 189583 is recorded in .mc/tasks/todo/T-040-*.md, but 189588 IS NOT RECORDED "
+    "ANYWHERE TRACKED IN THIS REPOSITORY — its artifact was not readable from the session that "
+    "wrote this line, and the job id is an untracked claim until GEOM_TOL_PILOT.json lands. It is "
+    "also evidence about a configuration THIS FILE HAS SINCE REPLACED: that pilot necessarily ran "
+    "at the old operating point (box_threshold 0.35, no retry branch), so it is weaker evidence "
+    "for the current adapter than its numbers suggest. So the module runs and produces output. "
+    "What that does NOT establish is that the output "
+    "is right: coverage 1.0 says a box was returned on every frame, not that it was the APPLE's "
+    "box, and this adapter's whole failure mode is a plausible mask on the wrong object (the "
+    "plate, the hand, the whole tabletop) which produces a centroid, a displacement and a p95 that "
+    "all look like measurements. Lowering BOX_THRESHOLD to upstream's 0.15 with a 0.10 retry — "
+    "correct, and required by §4 step 2 — makes coverage an even weaker witness than it was at "
+    "0.35, because more frames now get a box and none of them get checked. Discharged by: a human "
+    "looking at a sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame "
+    "frames, and the grasp), and/or a mask-vs-ground-truth IoU distribution from the Isaac capture "
+    "recorded beside the centroid displacement. Neither exists."
+    " <<< DISCHARGED BY ITS FIRST LIMB, THE HUMAN LOOK. A person recorded as reviewer "
+    "'human' opened all 382 overlaid apple-mask tiles of runs/pr08-mask-audit/MASK_AUDIT.json "
+    "(job 189637, 24 of 402 episodes, 480x640) through a review page and recorded ONE VERDICT "
+    "PER TILE. Evidence: runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json (recorded "
+    "2026-08-25T23:44:20Z, established_by 'human', coverage 382/382 tiles and 34/34 stratum "
+    "sheets), derived from runs/pr08-review-page/review.saved-1787701138-dff8.html through "
+    "runs/pr08-mask-audit/REVIEW_PAGE_INGEST.json, which carries the 382 per-tile verdicts the "
+    "per-sheet recording was compressed from, so the compression is recomputable. TALLY: apple "
+    "350, partial 7, wrong_object 12, no_mask 13. BY STRATUM: grasp 180/180 apple, spanning "
+    "92/92 apple, border 4/4 apple — 276 of 276 on the grasp and the ordinary frames; "
+    "min_visibility 74 apple / 3 wrong_object / 2 partial / 13 no_mask; census 7 wrong_object / "
+    "1 partial; occluded 4 partial / 2 wrong_object. The sample spans all three strata this "
+    "blocker names by name. WHAT THIS ENTRY CARRIES FORWARD RATHER THAN CLOSES, because a "
+    "discharge that hides its own residue is a deletion with better manners: (i) 18 of the 32 "
+    "non-apple verdicts are ONE episode, episode_000094 (18 of its 24 tiles) — the same episode "
+    "the probe-scan census names as the only one of 362 with an eligible occlusion frame, and "
+    "the same episode that owns all 36 mask refusals of the full GEOM_TOL pass, so this reads "
+    "as one corpus event the masker fails on rather than a masker that fails everywhere; (ii) "
+    "the 13 no_mask verdicts sit on frames where this module's own counters record a mask of "
+    "4355-6275 px (sample median apple 6185 px) at detection scores 0.567-0.826, with "
+    "n_frames_with_empty_mask = 0 and n_frames_without_detection = 0 over the same 382 calls — "
+    "the reviewer's account and the counters DISAGREE about those 13 frames and the "
+    "disagreement is unresolved; (iii) the tiles were produced at an ESTIMATOR_VERSION with no "
+    "mask_val_min_iou / mask_val_ref_max_frac token, i.e. by this adapter BEFORE the validity "
+    "filter it now runs — the mask drawn is bit for bit the same mask, but which frames are "
+    "measured is not the same set. Recorded under runs/pr08-mask-audit/MASK_AUDIT.json's own "
+    "not_a_discharge rule: the wording is moved with its evidence, not deleted.",
+    "2026-08-26 — blocker 2, 'BOX_THRESHOLD / TEXT_THRESHOLD / the retry are unmeasured on "
+    "AppleToPlate'. THE RETIRED WORDING FOLLOWS VERBATIM AND UNPARAPHRASED BETWEEN THE MARKERS "
+    ">>> and <<<. >>> "
+    "BOX_THRESHOLD / TEXT_THRESHOLD / the retry are unmeasured on AppleToPlate, and after 2026-08-22 "
+    "that is a narrower objection than it was. They are no longer 'upstream demo defaults we "
+    "happened to copy' (0.35/0.25): they are Cosmos-Transfer2.5's own operating point, read off its "
+    "sam2_model.py, which is precisely what §4 step 2 asks for. The choice-defect half of this "
+    "blocker is therefore DISCHARGED and inverted — measuring these on our corpus and moving them "
+    "to whatever reads best would MAKE this a different segmenter from the generator's, and the "
+    "budget would then be a budget for an error nobody commits. What survives is not a choice, it "
+    "is an unknown: nothing has measured what this operating point does on THIS corpus, and the "
+    "retry at (0.10, 0.10) buys detections by accepting weak ones, which on an occluded frame can "
+    "replace an honest all-False mask with a confident box on the wrong object. That inflates "
+    "coverage while degrading the mask, i.e. it hides itself in the one number the harness gates "
+    "on. Discharged by the same evidence as blocker 1, plus the recorded detection-score "
+    "distribution and retry counts (n_frames_retry_fired / n_frames_retry_recovered) from a full "
+    "pass, so the retry's contribution is visible rather than assumed."
+    " <<< DISCHARGED BY THE SAME EVIDENCE AS BLOCKER 1 (the entry above: "
+    "runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json, reviewer 'human', 382/382 tiles), PLUS the "
+    "full-pass conjunct this blocker names, read off the 16 GEOM_TOL shards "
+    "(runs/pr08-geom-tol/shards/shard-{0..15}.json — 16/16 present, 402/402 episodes, 171 625 "
+    "frames, one identical mask_method.version across all sixteen) and reported in "
+    "docs/preregistration/PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md: "
+    "n_frames_retry_fired = 0 and n_frames_retry_recovered = 0 — the (0.10, 0.10) retry NEVER "
+    "FIRED on any frame of the corpus, so the mechanism this blocker predicts did not occur and "
+    "cannot have contributed to any number in this project; detection-score distribution p1 "
+    "0.5489 / median 0.8497 / p99 0.9117; 92 frames of 171 625 (0.054 %) in [0.15, 0.35), which "
+    "is the whole of what the move from 0.35 to 0.15 bought, and 0 frames below 0.15. WHAT THIS "
+    "ENTRY CARRIES FORWARD: a NEIGHBOURING mechanism did operate — 57 % of those 92 weak "
+    "detections are in episode_000094 and some are confident, well-formed masks of THE PLATE, "
+    "arriving by the PRIMARY threshold rather than by the retry, which is this blocker's "
+    "predicted failure by an unpredicted route; the full pass ran on the AV1 tree while the "
+    "human look ran on the H.264-lossless tree, so the two conjuncts are two decodes and their "
+    "per-frame scores must not be quoted interchangeably; and git_commit is null on all 16 "
+    "shards, so the code that produced the full-pass half cannot be pinned to a revision.",
 )
 
-#: Opt-IN, and this module still does not opt in. THREE conditions above are open, and the two that
-#: matter most are cheap to state: nobody has looked at a mask this adapter produced (the FIRST
-#: blocker), and it is not yet the same segmenter the generator runs — it re-detects per frame where
-#: Transfer2.5 propagates (the LAST). Counted rather than indexed on purpose: the tuple shrank on
-#: 2026-08-22 when the committed-contract blocker was discharged, and a comment that said "blocker
-#: 4" went on pointing at whatever had moved into that slot. :data:`GATE_QUALIFICATION_DISCHARGED`
-#: now carries four conditions closed by measurement rather than by deletion, and one of the three
-#: that remain is inverted — which is progress and is not permission. ``measure_est_drift`` reads this flag with a default of
+#: Opt-IN, and this module still does not opt in. ONE condition above is open, and it is the one
+#: this file has named from the start as the last difference from upstream: this adapter
+#: re-detects per frame where Transfer2.5 propagates. Counted rather than indexed on purpose: the
+#: tuple shrank on 2026-08-22 and again on 2026-08-26, and a comment that said "blocker 4" went
+#: on pointing at whatever had moved into that slot. :data:`GATE_QUALIFICATION_DISCHARGED` now
+#: carries SIX conditions closed by evidence rather than by deletion — which is progress and is
+#: not permission. A shorter blocker tuple does not flip this flag and no edit that shortens the
+#: tuple may flip it in the same commit: it stays False until the remaining entry is closed AND
+#: somebody decides, on the record, what to do with the residue the two 2026-08-26 entries carry
+#: forward. ``measure_est_drift`` reads this flag with a default of
 #: False and stamps ``estimator_not_gate_qualified``; the artifact is still written, and exits 3.
 GATE_QUALIFIED = False
 
@@ -834,8 +894,9 @@
 #
 # Since 2026-08-22 both harnesses do record them (``estimator_stats`` in
 # ``configs/transfer25/pr08_geom_tol.json`` and in ``pr08_est_drift.json``), which is where the
-# full-pass half of ``GATE_QUALIFICATION_BLOCKERS``'s second entry lands. Recording it is not
-# discharging it: the blocker asks for the numbers AND for somebody to read them.
+# full-pass half of what was ``GATE_QUALIFICATION_BLOCKERS``'s second entry until 2026-08-26
+# lands; that entry is now in ``GATE_QUALIFICATION_DISCHARGED``. Recording it was never
+# discharging it: the blocker asked for the numbers AND for somebody to read them.
 
 #: ``"metric"``, ``"relative"``, or None before the depth model has been loaded. Read off the loaded
 #: config; absent from the config is treated as ``"relative"``, because that is what transformers'
@@ -927,7 +988,8 @@
 #: were binned identically, which is the same argument that makes the shards emit raw
 #: displacements. And a distribution recorded as a digest cannot answer a question nobody asked
 #: yet, which for this list is the whole point: it is the evidence
-#: ``GATE_QUALIFICATION_BLOCKERS``'s second entry asks for, and the question it is meant to answer
+#: ``GATE_QUALIFICATION_BLOCKERS``'s second entry asked for until 2026-08-26 (it is now in
+#: ``GATE_QUALIFICATION_DISCHARGED``), and the question it is meant to answer
 #: — how much of ``coverage`` was bought at the retry's lower threshold — is READ OFF THE VALUES.
 #: A score below :data:`BOX_THRESHOLD` can only have come from the ``(0.10, 0.10)`` retry, because
 #: the first pass discards everything under it; so ``[s for s in DETECTION_SCORES if s <
```

Five hunks. Two carry the discharge itself (the tuple, and the two new
`GATE_QUALIFICATION_DISCHARGED` entries plus the count above `GATE_QUALIFIED`). The other three are
**index pointers inside the same file** that the discharge would silently invalidate — the comment
above `MASK_VALIDITY_MIN_IOU` ("blocker 1's first entry names in as many words") and the two that
point at "the second entry". The module's own docstring warns about exactly this failure
("a comment that said *blocker 4* went on pointing at whatever had moved into that slot"), so a
diff that shortens the tuple and leaves them is not a smaller diff, it is a wrong one.

---

## 4. What the evidence does and does not support

This section is written against the proposal in §3, not for it.

### 4.1 Is 32 non-apple verdicts out of 382 compatible with discharging blocker 1?

**On the blocker's letter: yes, and the letter is not a trick here.** Blocker 1's discharge
condition names an **act** and a **sample**, not a pass rate:

> *"Discharged by: a human looking at a sample of overlaid masks spanning the corpus (occluded
> frames, apple-out-of-frame frames, and the grasp) …"*

There is no threshold in that sentence, and there could not honestly be one written after the
fact — a rate bar invented today, with the tally already known, is the failure
`docs/handoff.md` §3 and the MuJoCo note §3 both name. The act happened, the sample spans all three
named strata, the coverage is 382/382.

**On the blocker's purpose: it depends entirely on where the 32 sit, and here they sit in one
place.** The blocker's purpose is stated a sentence earlier — *"what that does NOT establish is
that the output is right"*. A discharge that recorded "a person looked, and on 8 % of what they saw
the mask was not the apple" while implying the masker is fine would satisfy the letter and invert
the purpose. The split that decides it:

| | tiles | non-apple |
|---|---|---|
| `grasp` + `spanning` + `border` — the ordinary frames and the grasp | 276 | **0** |
| `census` + `occluded` + `min_visibility` — the deliberately over-weighted hard frames | 106 | **32** (30.2 %) |

**Zero of 276 on the strata drawn without regard to difficulty**, and `sampling.bias` says the
`spanning` stratum is exactly the control for this: *"a mask defect that appears on ORDINARY frames
would show up in the `spanning` stratum, which is drawn without regard to difficulty."* It did not.

**But 8 % is not a corpus rate and must never be quoted as one, in either direction.**
`MASK_AUDIT.json`'s `sampling.bias` is explicit that this sample over-weights the hard frames on
purpose, so 32/382 overstates the corpus failure rate and 0/276 understates the risk on frames the
sample never drew. The honest reading is the narrow one: **on frames of ordinary difficulty this
masker segments the apple, and on the corpus's occlusion event it does not.** That is what a
discharge of blocker 1 may claim and it is all it may claim.

### 4.2 Is a failure concentrated in one episode a masker failure or a corpus failure?

**It is a masker failure, on a corpus event that occurs in one episode.** Both halves matter and
the second is not an excuse.

- **18 of the 32 non-apple verdicts are `episode_000094`** — 18 of that episode's 24 tiles. Its
  `census` stratum is 0/8 apple and its `occluded` stratum is 0/6.
- `episode_000094` is not an arbitrary episode. `MASK_AUDIT.json`'s census block records that the
  probe-scan measured 154 447 frames across 362 episodes and found **`episodes_with_any_eligible_frame:
  ["episode_000094"]`** — it is the only episode in the scanned corpus with a qualifying occlusion.
- The full pass agrees from a second direction: **57 % of the corpus's 92 weak detections are in
  that episode**, and **all 36 mask refusals of the entire 171 625-frame pass are its frames**.

So the sample did not "get unlucky in one episode": the sampler *forced in* the one episode the
corpus's own census named, and the masker failed there. Calling that a corpus failure would be
backwards — the corpus contains a hard event and the estimator mishandles it. What is fair to say
is that the failure is **not distributed**: outside `episode_000094`, 344 of 358 tiles are `apple`,
and 13 of the remaining 14 are the disputed `no_mask` verdicts of §4.3 — so once those 13 are set
aside, the other 23 episodes carry exactly **one** substantive non-apple verdict
(`episode_000110` f87, `partial`) across 345 tiles.

**The unwelcome consequence for the discharge:** the evidence that blocker 1's *feared* failure mode
is real — *"a plausible mask on the wrong object … which produces a centroid, a displacement and a
p95 that all look like measurements"* — is now a human's verdict rather than a model's inference.
Discharging blocker 1 on evidence that **confirms** the blocker's fear is defensible only because
the blocker asked whether anyone had *looked*, and the answer is now yes with the finding written
down. It would not be defensible if the discharged entry did not carry the finding. §3's entry
carries it in three numbered clauses.

### 4.3 What do the 13 `no_mask` verdicts mean for blocker 2?

**They are the weakest part of this evidence and they are not resolved.**

`runs/pr08-mask-audit/RECORD_VERDICTS.sh` defines the vocabulary the reviewer worked in:

> *`no_mask` = keine Maske sichtbar, obwohl ein Apfel im Bild ist* — "no mask visible, although an
> apple is in the picture."

That verdict, on those 13 tiles, contradicts the adapter's own counters:

| | |
|---|---|
| where | all 13 in `min_visibility`; **12 on the single sheet `min_visibility-03`** (episodes 146, 165, 183, 201), 1 on `min_visibility-04` (`episode_000201` f270) |
| mask area the adapter recorded | **4 355 – 6 275 px** (the sample's median apple is 6 185 px, its minimum 471 px) |
| detection score | 0.567 – 0.826 (sample median 0.829) |
| triage flags | **none** on any of the 13 |
| adapter counters over the same 382 calls | `n_frames_without_detection: 0`, `n_frames_with_empty_mask: 0`, `n_frames_retry_fired: 0` |

A ~6 000 px mask is not invisible, and the adapter says it returned one on every frame of this
sample. So on 13 tiles **the reviewer's account of what is on screen and the instrument's own
counters disagree**, and `MASK_AUDIT.json` has a name for that condition in its own flag vocabulary:
`recorder_inconsistent` — *"the post-processing recorder and the adapter's own counters disagree
about what happened on this frame. Trust the counters and read this as a defect in the audit, not in
the adapter."* Three readings survive and this document cannot choose between them:

1. **The reviewer meant "no mask *on the apple*."** Then these are `wrong_object` by another name
   and the non-apple count is unchanged in kind.
2. **The reviewer meant "the apple is not here."** `min_visibility` is by construction the episode's
   least-visible frames, and a correct all-False mask on a frame with no apple is the adapter
   behaving *properly*. But the counters say the adapter did not return all-False, so under this
   reading the tile and the counter still describe different frames.
3. **The tiles on `min_visibility-03` did not render their overlay.** 12 of 13 on one sheet is
   sheet-shaped, which is what a rendering defect looks like and is not what 13 independent
   observations look like.

**What this costs blocker 2 specifically.** Blocker 2's surviving objection is about the operating
point and the retry, and its named hazard is the retry *"replac[ing] an honest all-False mask with a
confident box on the wrong object"*. The full pass answers that decisively — the retry fired **zero**
times over 171 625 frames, so the hazard did not occur and cannot have occurred. The 13 `no_mask`
verdicts do **not** re-open it: they are not retry frames (nothing in this sample is), and a
disagreement about whether a mask was visible is not evidence that a weak detection was accepted.

**What it does cost is confidence in the instrument that recorded the look**, which is blocker 2's
*first* conjunct — the same evidence as blocker 1. Thirteen tiles out of 382 where the human record
and the machine record describe different frames is 3.4 % of the evidence being uninterpretable, and
the honest options are (a) apply the diff with clause (ii) of the discharged entry carrying the
disagreement forward, as §3 does, or (b) resolve those 13 tiles first — open `min_visibility-03`,
look at four of them, and find out which of the three readings is true. **(b) is cheap and this
document recommends it.** It is a sheet, four tiles and ten minutes, and it is the difference
between a discharge that carries a known residue and one that carries an unexamined one.

### 4.4 The strongest objection: the tiles were made by an adapter this file has since replaced

This is the objection that most nearly sinks the proposal, and it is the file's own argument turned
against it.

`MASK_AUDIT.json` records the tiles' `estimator_version` ending at `…;box_sel=highest_score;prop=per_frame`.
Today's `ESTIMATOR_VERSION` ends `…;prop=per_frame;mask_val_min_iou=0.1;mask_val_ref_max_frac=0.1`.
The mask-validity filter was added **after** the audit — in fact *because* of it; the comment above
`MASK_VALIDITY_MIN_IOU` cites job 189637's 382 frames as its motivation. And the comment beside the
version string says what that difference means:

> *"In the version string because it decides which frames a recorded number was measured on, and an
> artifact whose version cannot answer 'was the mask-validity filter on?' cannot be compared against
> one measured before it existed."*

`run_g0_gates.instrument_disagreements` compares that string between the two sides of G0b and
**refuses** a comparison across it. By the module's own rule, the adapter that drew these 382 tiles
and the adapter in the file today are **two instruments**.

**This is exactly the caveat blocker 1 levels at job 189588** — *"evidence about a configuration
THIS FILE HAS SINCE REPLACED"* — and it now applies to the evidence being offered to retire it.

**The counter-argument, which is real but does not fully answer it.** The validity filter changes no
threshold, no prompt, no retry and no box rule; the file states that *"the mask drawn is bit for bit
the mask that was drawn before"* and that what changes is *which frames we are willing to measure
on*. So every per-tile verdict remains a true statement about what this detector-plus-segmenter
draws. What is no longer true is that the 382 frames are the population the current adapter would
measure: the filter would refuse some of them — on the full pass it refused 36, all in
`episode_000094`, which is precisely where 18 of the 32 non-apple verdicts live.

**Which cuts in the reassuring direction, and that is itself a reason for suspicion.** The frames
the human judged worst are the frames the current adapter would decline to measure. A person could
apply this diff believing the residue is smaller than the tally suggests — and they would probably
be right — but nobody has measured the overlap between "the 36 refused frames" and "the 18 non-apple
verdicts", and §2.2 of the operating-point result says the exact 36 are **not** recorded anywhere.
That is an unquantified favourable assumption sitting under a discharge, which is the shape of
defect this repository exists to catch.

### 4.5 Blocker 2's two conjuncts are measured on two different decodes

Blocker 2's condition is conjunctive: the same evidence as blocker 1, **plus** the full-pass numbers.
The first conjunct ran on the H.264-lossless tree
(`/valhalla/…/pr08-apple-640x480-h264-lossless`); the second ran on the AV1 tree
(`data/pr08-apple-640x480`). The operating-point result states this plainly and gives the example:
frame 129 of `episode_000094` scores 0.213 in the audit and 0.232 in the shards. *The phenomenon
replicates; the numbers are not the same numbers.* A conjunction assembled from two decodes is
weaker than one assembled from a single pass, and nothing here repairs that. Additionally,
`git_commit` is `null` on all 16 shards, so the code behind the full-pass half cannot be pinned to a
revision — which is an AC-04 problem sitting inside a discharge.

### 4.6 Summary of what is and is not supported

| claim | supported? |
|---|---|
| A person has looked at overlaid masks from this adapter, spanning occluded frames, minimum-visibility frames, border frames and the grasp | **Yes.** 382/382 tiles, reviewer `human` |
| On frames of ordinary difficulty, the mask is the apple | **Yes**, 276/276 on `grasp` + `spanning` + `border` |
| The masker is correct on the corpus's occlusion event | **No.** It is wrong on 14 of 14 `census`/`occluded` tiles of `episode_000094` |
| The retry buys weak detections on this corpus | **Refuted.** 0 firings in 171 625 frames |
| The operating point at 0.15 buys nothing harmful | **No.** 92 frames, 57 % of them in one episode, some of them confident plate masks by the *primary* threshold |
| 8.4 % is this corpus's non-apple rate | **No.** The sample is deliberately biased towards hard frames |
| The evidence describes the adapter as it stands today | **No.** Two `ESTIMATOR_VERSION` tokens apart; see §4.4 |
| Blocker 3 is affected | **No.** Untouched by every artifact cited here |

---

## 5. Whether `GATE_QUALIFIED` may flip

**No. It may not, and the diff in §3 does not touch it.**

### 5.1 Blocker 3 is open, and the tuple is conjunctive by its own header

`GATE_QUALIFICATION_BLOCKERS`' docstring: *"Every condition that has to be true before this pair may
set `EST_DRIFT_P95` or `GEOM_TOL`."* One entry remaining is one condition unmet. A one-element tuple
is not an empty one.

### 5.2 Blocker 3's evidence exists and still does not close it

`runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json` records both arms over one capture
— per-frame p95 **0.2908 px**, propagation p95 **0.4701 px**. That is the shape of measurement
blocker 3 asks for, and it closes nothing, for three reasons already on the record:

1. Blocker 3 says *"the same **Isaac** capture"*; this is MuJoCo.
2. `measure_est_drift`'s own `_ARM_COMPARISON_DISCHARGES` string says the answer in capitals:
   *"NOTHING … producing the evidence a blocker asks for is not the same act as accepting it."*
3. `PR-08-NOTE-2026-08-25-the-mujoco-iou-cannot-discharge-blocker-1.md` §3 forecloses a rule
   registered *after* the favourable number is known, on `docs/handoff.md` §3.

The number also happens to point at limb (a), the *safe* side of blocker 3's two-sided bias, and
says nothing about limb (b), the drift-and-stay-off failure that is invisible to a per-frame
estimator. Blocker 3 is explicit that limb (b) is what makes the budget neither an upper nor a lower
bound.

### 5.3 Even if blocker 3 fell, the flip is a separate act

Flipping `GATE_QUALIFIED` is what lets `configs/transfer25/pr08_geom_tol.json` stop holding nulls,
which is what closes PR-08 §8 **item 4**, which is one of the seven conjuncts of `T40_RULE_V1` §1.
`run_g0_gates` refuses in three places on `gate_qualified: false` (`_ca_gate_qualified`, the
`--geom-config` check, and the G0b per-side segmenter check). A flag with that reach must not
ride in the same commit as a tuple that just got shorter: the shortening is reviewable precisely
because it is separable from the flip. §8 item **3** (throughput and the GPU-h ceiling) is open
regardless, and `PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §2 records that §1 is
conjunctive — so **nothing in this document brings a first clip nearer than item 3 allows.**

---

## 6. What goes stale if this diff is applied, and is not in it

Applying §3 leaves six places elsewhere in the tree pointing at slots that have moved — one of which
is a test that fails, one of which is a test that passes and should not, and one of which must be
left stale on purpose. Naming them is part of the proposal; fixing the fixable ones is the
applier's, in the same commit.

| where | what breaks |
|---|---|
| `tests/test_apple_sam2_video_propagation.py::test_this_module_discharges_nothing` | **FAILS.** It asserts `len(apple_sam2.GATE_QUALIFICATION_BLOCKERS) == 3`. This is the one hard breakage, and it is a test whose whole purpose is to say "nothing is discharged" — editing it is an act no session should perform unattended |
| `tests/test_audit_apple_masks.py::test_the_adapter_itself_still_declares_the_blockers_this_audit_addresses` | **PASSES — and should not comfort anyone.** It greps for `"NOBODY HAS LOOKED AT A MASK"` and `"n_frames_retry_fired / n_frames_retry_recovered"` in the source, and move-don't-delete keeps both. Its docstring says *"If blocker 1's wording moves, this artifact's `addresses` line is stale and must move too"* — the wording has moved, the `addresses` line **is** stale, and the test does not notice |
| `scripts/measure_est_drift.py`, `_ARM_COMPARISON_DISCHARGES` | Asserts in prose that *"all three blockers are still in the tuple"* — becomes **false**, and it is written into every EST_DRIFT artifact |
| `scripts/measure_est_drift.py`, arm-comparison `blocker` field | Points at *"GATE_QUALIFICATION_BLOCKERS, third entry"*. After the diff, blocker 3 is the **first** entry |
| `scripts/audit_apple_masks.py` (`GATE_QUALIFICATION_BLOCKERS[0]`), `scripts/build_review_page.py` (`[0]` and `[1]`, in the docstring and in the page body) | Index pointers into a tuple that just changed shape |
| `runs/pr08-mask-audit/MASK_AUDIT.json`, field `addresses` | Names `[0]`, `[1]` and `[2]`. **This one must NOT be rewritten** — it is a committed artifact and a record of its own moment. It goes stale correctly |

The four in-file pointers (`apple_sam2.py`'s own comments at the mask-validity block, the
counter-recording block, the `DETECTION_SCORES` block, and the count above `GATE_QUALIFIED`) **are**
in the diff, for the reason the module gives: an index in a comment goes stale silently.

---

## 7. This document applies nothing

To be unambiguous, because the subject matter is a gate:

- **No source file was edited.** `scripts/estimators/apple_sam2.py` is untouched by this session;
  the diff above exists only as text inside this markdown file.
- **No git write was performed.** No `git add`, no `git commit`, no branch, no tag.
- **No blocker is discharged.** `GATE_QUALIFICATION_BLOCKERS` still has three entries and
  `GATE_QUALIFICATION_DISCHARGED` still has four.
- **`GATE_QUALIFIED` is `False`** and is not proposed to change.
- **No clip and no training run is licensed.** `T40_RULE_V1` §1 binds; §8 items 3 and 4 are open.
- The act this document exists to enable is a **person** reading §4, deciding whether the residue in
  §4.3 and §4.4 is acceptable, and applying §3 with their own hand — or declining to.

---

## 8. Provenance

| | |
|---|---|
| kind | **proposal.** A diff as text plus the argument against it. Registers no rule, measures nothing, applies nothing |
| date | 2026-08-26 |
| base revision | `71edc63`, `scripts/estimators/apple_sam2.py` (1 896 lines) |
| blockers the diff would move | `GATE_QUALIFICATION_BLOCKERS` **[0]** and **[1]** → `GATE_QUALIFICATION_DISCHARGED` |
| blockers the diff leaves | **[2]**, per-frame vs. propagation, byte-identical |
| evidence, blocker 1 | `runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json` (reviewer `human`, 382/382 tiles, recorded 2026-08-25T23:44:20Z), via `runs/pr08-mask-audit/REVIEW_PAGE_INGEST.json` from `runs/pr08-review-page/review.saved-1787701138-dff8.html`; tiles from `runs/pr08-mask-audit/MASK_AUDIT.json` (job 189637) |
| evidence, blocker 2 | the above **plus** `docs/preregistration/PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md` over `runs/pr08-geom-tol/shards/shard-{0..15}.json` |
| rule followed | `runs/pr08-mask-audit/MASK_AUDIT.json` field `not_a_discharge` |
| verbatim preservation | checked mechanically: `GATE_QUALIFICATION_BLOCKERS[0]` (1 845 chars) and `[1]` (1 252 chars) are exact substrings of the two proposed discharged entries |
| patch validated | `patch -p1 --dry-run` against a scratch copy: applies cleanly; result parses; 1 blocker, 6 discharged, `GATE_QUALIFIED is False` |
| known breakage if applied | `tests/test_apple_sam2_video_propagation.py::test_this_module_discharges_nothing` (`len(...) == 3`) |
| strongest objection to applying it | §4.4 — the 382 tiles were drawn by an `ESTIMATOR_VERSION` two mask-validity tokens older than the one in the file today, which is the same caveat blocker 1 levels at job 189588 |
| new jobs run | **none** |
| code changed | **none** |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
