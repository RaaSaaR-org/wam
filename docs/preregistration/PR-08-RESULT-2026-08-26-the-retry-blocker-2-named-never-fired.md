# PR-08 — blocker 2 asked for the retry's contribution from a full pass. It is zero, on every frame

**Written 2026-08-26 from measurements that already existed. No pixel was re-measured, no gate,
config or blocker was touched, and no bound is written or implied. `GATE_QUALIFIED` is still
`False`.**

`scripts/estimators/apple_sam2.py`'s **second** gate-qualification blocker names its own discharge
condition, and one half of that condition — "the recorded detection-score distribution and retry
counts (`n_frames_retry_fired` / `n_frames_retry_recovered`) **from a full pass**, so the retry's
contribution is visible rather than assumed" — was believed to be unsatisfiable, because the
harnesses were thought not to record it. **They do record it, they have recorded it since
2026-08-22, and the full pass of 2026-08-24 carries the numbers.** This document reads them out.

---

## 1. What blocker 2 actually asks, after its first half was already discharged

The blocker has two halves and only one is still open. On 2026-08-22 the **choice-defect** half was
discharged and *inverted*: `BOX_THRESHOLD` / `TEXT_THRESHOLD` / the retry are not "upstream demo
defaults we happened to copy" but Cosmos-Transfer2.5's own operating point, read off its
`sam2_model.py`, so measuring them on our corpus and moving them to whatever reads best would make
this a *different* segmenter from the generator's.

What survives is **not a choice, it is an unknown**, and the blocker states the hazard precisely:

> the retry at (0.10, 0.10) buys detections by accepting weak ones, which on an occluded frame can
> replace an honest all-False mask with a confident box on the wrong object. That inflates coverage
> while degrading the mask, i.e. it hides itself in the one number the harness gates on.

That is a claim about **events**, and events can be counted.

## 2. The full pass

`runs/pr08-geom-tol/pr08_geom_tol.json` — `measure_geom_tol.py --merge`, measured 2026-08-24, 16
shards pooled, corpus `pr08-apple-640x480`, **402 episodes, 171 625 frames**,
`partial_measurement: false`.

Its `estimator_stats.this_run` block:

| counter | value |
|---|---:|
| `n_segment_calls` | **171 625** |
| `n_frames_without_detection` | **0** |
| `n_frames_with_empty_mask` | **0** |
| **`n_frames_retry_fired`** | **0** |
| **`n_frames_retry_recovered`** | **0** |
| `n_frames_mask_refused` | 36 |
| `n_frames_mask_refused_no_reference` | 0 |

**The retry did not fire once in 171 625 frames.** The hazard blocker 2 names is not small on this
corpus; it has zero instances.

It follows, and is worth stating separately because coverage is the number the harness gates on:
`n_frames_without_detection: 0` means the first pass found a box on **every** frame at
`box_threshold = 0.15`. Coverage here was not bought at the lower threshold, because the lower
threshold was never reached.

## 3. The detection-score distribution

`estimator_stats.detection_scores`, n = 171 625 — the winning box's score on every frame, in call
order.

| | |
|---|---:|
| min | 0.16641 |
| p1 | 0.54899 |
| p5 | 0.73319 |
| p25 | 0.82043 |
| **p50** | **0.84972** |
| p75 | 0.87416 |
| p95 | 0.90020 |
| p99 | 0.91169 |
| max | 0.93838 |
| mean / std | 0.83709 / 0.06362 |

The artifact records one more field beside it, and it is a **second, independent witness** to §2:

```
"box_threshold": 0.15,
"n_below_box_threshold": 0
```

A detection scoring below the first-pass threshold could only have come from the retry. There are
none. The retry counters and the score distribution are computed by different code paths from
different state, and they agree.

The same field carries its own caveat, in the artifact's words: *"Nothing here says those masks are
on the right object — that is the adapter's first gate-qualification blocker and it is not answered
by a number."* §6 below keeps that separation.

## 4. Why these numbers are readable rather than merely present

`scripts/measure_geom_tol.py` states three properties this block had to have (the comment above
`ADAPTER_RUN_COUNTERS`, ~lines 1195–1245, which quotes blocker 2 by name). Each is checkable and
each was checked here.

**It describes THIS run.** The adapter's counters are cumulative over the lifetime of the import and
nothing resets them, so a second measurement driven from one interpreter would silently record the
other's frames. They are snapshotted before the pass and **differenced** afterwards, and the adapter
is never written to from the harness. The artifact's `counters_went_backwards` is `null`.

**The scores survive sharding exactly.** A distribution does not decompose the way a sum does, so
each shard records the **raw** per-episode scores and the merge concatenates them in the corpus's own
enumeration order *before* binning — which makes the merged distribution identical float for float to
what an un-sharded run would have written, not approximately equal to it.

**An adapter without `stats()` is recorded as absent-with-a-reason, never as zeros.** "The retry
fired 0 times" and "nobody asked" are different claims, and this artifact says the first:
`recorded: true`, `absent_because: null`.

I verified the sharding claim directly. Summing `estimator_stats.this_run` over all 16 shard files
under `runs/pr08-geom-tol/shards/`:

```
n_segment_calls                     shards=171625  merged=171625  OK
n_frames_without_detection          shards=     0  merged=     0  OK
n_frames_with_empty_mask            shards=     0  merged=     0  OK
n_frames_retry_fired                shards=     0  merged=     0  OK
n_frames_retry_recovered            shards=     0  merged=     0  OK
n_frames_mask_refused               shards=    36  merged=    36  OK
n_frames_mask_refused_no_reference  shards=     0  merged=     0  OK
n_mask_validity_iou                 shards=171625  merged=171625  OK
n_detection_scores                  shards=171625  merged=171625  OK
```

Nine of nine agree exactly. No shard is missing and none is double-counted.

## 5. A stale note in `MASK_AUDIT.json`, left in place deliberately

`runs/pr08-mask-audit/MASK_AUDIT.json` carries a `full_pass_gap` field asserting that neither
`measure_geom_tol.py` nor `measure_est_drift.py` reads `estimators.apple_sam2.stats()`, so "a full
GEOM_TOL run produces none of them and the full-pass half of blocker 2 has nowhere to land."

**That was true when it was written and is not true now.** `measure_geom_tol.py` was wired for
exactly this on 2026-08-22 — `ADAPTER_RUN_COUNTERS` lists `n_frames_retry_fired` and
`n_frames_retry_recovered` by name — and §2 above reads the result out of a real pass.

**`MASK_AUDIT.json` was NOT edited.** It is a measurement artifact and its text is the record of what
was known when it was produced; a session that quietly repairs the prose of an artifact makes every
other artifact's prose unciteable. The correction lives here instead, which is where a reader who
follows the `full_pass_gap` field will be sent.

## 6. Blocker 2's other half, and where it stands

Blocker 2 is discharged by *"the same evidence as blocker 1, **plus**"* these numbers. Blocker 1's
evidence — a person looking at overlaid masks spanning the corpus — was recorded on 2026-08-25 and
**corrected on 2026-08-26**: `runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json` now holds 382 verdicts
over 34/34 sheets, all three required strata covered, tally `apple 363 / partial 7 / wrong_object 12`
and **no `no_mask`**. The thirteen `no_mask` verdicts that stood there until today were one mis-set
sheet default, contradicted by the instrument's own per-frame counters; the reviewer re-set them.
`runs/pr08-mask-audit/SECOND_OPINION.json`, recomputed against the corrected record, reports **0
disagreements across 430 tiles** — which corroborates nothing, for the reason that document states in
its own §1, and is recorded here only because the contradiction it used to carry is gone.

**So both halves of blocker 2's named condition now have evidence. That is not the same as the
blocker being discharged**, and this document does not discharge it. Discharging is a reviewable edit
to `GATE_QUALIFICATION_BLOCKERS` made by a person, moving the retired wording into
`GATE_QUALIFICATION_DISCHARGED` with the evidence rather than deleting it. Producing evidence and
accepting it are two different acts.

## 7. The estimator version the full pass ran at

Blocker 1's proposed discharge raises an objection worth answering here too, because it applies to
this pass as much as to the tiles: is this evidence about a configuration the file has since
replaced? Splitting the three `estimator_version` strings on `;`:

| | audit tiles | **full pass** | today |
|---|---|---|---|
| the 10 detection tokens — both checkpoints and revisions, `prompt='apple.'`, `box_thr=0.15`, `text_thr=0.25`, `retry_box_thr=0.1`, `retry_text_thr=0.1`, `box_sel=highest_score`, `prop=per_frame` | identical | identical | identical |
| `mask_val_min_iou` | absent | 0.1 | 0.1 |
| `mask_val_ref_max_frac` | absent | absent | 0.1 |

**Ten of twelve tokens are character-identical across all three.** Every differing token is a
mask-**validity** field, and the code says what those can do: `segment()` computes the mask from the
box upstream's rule selected at upstream's thresholds, and only then asks whether the frame is one we
are willing to measure on. On refusal it returns `np.zeros(...)`
(`scripts/estimators/apple_sam2.py:1853` and `:1862`). Nothing is re-detected, re-prompted or
re-drawn. **A validity field can replace a drawn mask with nothing; it can never alter one, and it
can never accept a frame the module previously refused.**

For this pass the direction is therefore stated exactly: today's adapter carries one refusal
condition the full pass did not have (`mask_val_ref_max_frac`, added 2026-08-24 under V10), so today
it refuses *more* frames, never fewer — and a refusal cannot manufacture a retry that did not fire.
The `n_frames_retry_fired: 0` of §2 is not weakened by the version difference in either direction.

## 8. What this does NOT establish

- **Not that the retry is safe.** A branch that never executed is **untested, not proven**. Nothing
  here licenses the retry on another corpus, at another operating point, on restyled frames, or on
  the generated clips. It says the retry contributed nothing *to this pass on this corpus*.
- **Not that any mask is on the apple.** A detection-score distribution says how confident the
  detector was, never *what it was confident about*. That is blocker 1's question in full, and this
  document does not touch it — the score histogram would look exactly like this if every box were on
  the plate.
- **Not a discharge of blocker 2, or of anything.** §6.
- **Not an account of the 36 refused frames.** `n_frames_mask_refused: 36` (0.021 % of the pass) is
  reported here and not analysed; whether those refusals were correct is a separate question that
  this document does not open.
- **Not a bound.** No `max_frame_fraction` is written, proposed or implied. `T40_RULE_V13` stays an
  unsigned draft.
- **Not a statement about blocker 3.** The per-frame/propagation difference is untouched here.

---

## 9. Provenance

| | |
|---|---|
| kind | reading of existing measurements. **Registers no rule, measures no new pixels** |
| date | 2026-08-26 |
| primary artifact | `runs/pr08-geom-tol/pr08_geom_tol.json` (`--merge`, measured 2026-08-24, `headline_valid: true`, `partial_measurement: false`) |
| pass size | 402 episodes, **171 625 frames**, 16 shards |
| cross-check | per-shard `this_run` counters summed against the merged block — 9 of 9 exact |
| corroborating artifacts | `runs/pr08-geom-tol/shards/shard-{0..15}.json`, `runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json`, `runs/pr08-mask-audit/SECOND_OPINION.json` |
| stale note left unedited | `runs/pr08-mask-audit/MASK_AUDIT.json` → `full_pass_gap` (§5) |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched — the merged artifact itself records `gate_qualified: false` |
| generation licensed | **no** |
| training licensed | **no** |
