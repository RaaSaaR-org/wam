# PR-08 — the recorded mask verdicts, checked against a blind re-read

**Run 2026-08-26, at the project owner's request: *"kannst du selbst auch nochmal die bilder checken
und eine zweite analyse machen. ich bin mir nicht sicher ob 'human' das richtig gemacht hat"*.
Registers no rule, discharges no blocker, writes no bound, and edits no recorded verdict.**

---

## 1. What was checked, and what a check by a model can be

On 2026-08-25 a person recorded a verdict on all 382 apple-mask tiles and all 48 area-tail tiles
through the review page (`runs/pr08-mask-audit/MASK_AUDIT_VERDICTS.json`,
`runs/pr08-area-tail-look/TAIL_VERDICTS.json`, reviewer `human`). Blockers 1 and 2 are supposed to
discharge on the first of those, and `T40_RULE_V13` §3.2 on the second.

**The second reader here is a model, and that is the correlated observer `MASK_AUDIT.json` warns
about.** So this document is written asymmetrically and the asymmetry is the point:

| | |
|---|---|
| the two **agree** | **nothing is established.** Two correlated observers can reproduce one misreading |
| the two **disagree** | **one of them is wrong on a named tile**, which a person can settle cheaply |

No accuracy figure is computed anywhere, and `scripts/compare_second_opinion.py` carries a test that
forbids one, because an accuracy would silently promote one of two correlated observers to ground
truth. **The product of this exercise is 13 named tiles, not 403 agreements.**

## 2. Method

Eight readers, each given a disjoint set of contact sheets and no access to the recorded verdicts.
Each was instructed by name not to open `MASK_AUDIT_VERDICTS.json`, `REVIEW_PAGE_INGEST.json`,
`TAIL_VERDICTS.json` or anything under `runs/pr08-review-page/`. **That instruction is recorded, not
verified** — the same limitation `--reviewer` has.

Each judged every tile on its sheets in the section's own vocabulary and wrote a per-tile verdict
with a confidence and a one-line description of what it saw. Sheet assignments are disjoint and
`load_opinions` refuses if two groups claim the same tile. Several readers cropped and upscaled
individual tiles 3–12× before judging; the area-tail reader inverted the overlay's known linear
blend per pixel to recover the binary mask rather than reading the tint.

## 3. Result

| | |
|---|---:|
| tiles compared | **430** |
| identical | 403 |
| severity only — same side of "is the mask on the apple" | 14 |
| **disagreements across that line** | **13** |

Every one of the 13:

```
min_visibility-03    no_mask -> apple    x12
min_visibility-04    no_mask -> apple     x1   (episode_000201:270)
```

**There is no other disagreement anywhere in 430 tiles.**

## 4. The 13, and why they are not a tie

A disagreement between two correlated observers cannot be resolved by either of them. These can be,
because a third record exists that predates both readers and was written by the instrument itself:

`runs/pr08-mask-audit/MASK_AUDIT.json` records, for those same frames, a mask of **4 355–6 275 px**
with `warm_apple_iou` **0.92–0.97**, and `n_frames_with_empty_mask: 0` over the whole audit. A
`no_mask` verdict says a visible apple carries no mask at all. **The instrument's own counters say
there is a mask, and say how big it is.** That is not a second opinion; it is the measurement the
tiles were drawn from.

Reading the two sheets confirms it visually: every tile carries a green mask on the apple, a
detector box, a score of 0.55–0.87, and the colour-heuristic outline in agreement.

**So the 13 `no_mask` verdicts are wrong, and the shape of the error names its cause**: twelve
identical values filling exactly one sheet with no exception is a sheet default set to the wrong
value, not twelve judgements.

**The direction matters and cuts the safe way.** The error makes the masker look WORSE than the
evidence supports. It cannot have produced a discharge that the evidence does not carry.

## 5. The area tail

| | recorded | blind re-read |
|---|---:|---:|
| `arm` — a legitimate near-camera arm | **0** | **0** |
| `table` | 5 | 2 |
| `mixed` | 43 | 46 |

The two readers split `table` from `mixed` differently on five tiles, which is a within-class
difference: both labels say the mask contains scene. **On the question `T40_RULE_V13` §2 actually
asks — is any frame in this tail a legitimate near-camera arm that a bound must not discard — the
two readers agree, and the answer is no on all 48.**

## 6. What this does NOT establish

- **Not that the other 403 verdicts are right.** §1. Agreement between correlated observers is not
  corroboration, and this document must not be cited as though it were.
- **Not a discharge of any `GATE_QUALIFICATION_BLOCKERS` entry.** That is a person's edit. See
  `PR-08-PROPOSED-2026-08-26-discharge-of-blockers-1-and-2.md`, which raises a separate and
  unrelated objection: the audit's tiles were rendered under an estimator version older than the
  object-grounding filter.
- **Not a corrected record.** `MASK_AUDIT_VERDICTS.json` still carries the 13 `no_mask` verdicts.
  Overwriting a person's verdicts from a model's reading is exactly the substitution this project
  forbids; the fix is the reviewer re-setting those two sheets.
- **Not a bound, and not a rule.** V13 stays an unsigned draft.

---

## 7. Provenance

| | |
|---|---|
| kind | check on existing evidence. **Registers no rule, measures no new pixels** |
| date | 2026-08-26 |
| requested by | the project owner, in so many words |
| second reader | **a model — the correlated observer, by construction** |
| tool | `scripts/compare_second_opinion.py`, `tests/test_compare_second_opinion.py` |
| artifacts | `runs/pr08-mask-audit/SECOND_OPINION.json`, `runs/pr08-mask-audit/second-opinion-raw/` |
| checked | `MASK_AUDIT_VERDICTS.json` (382), `TAIL_VERDICTS.json` (48) |
| third record that settles the 13 | `MASK_AUDIT.json`'s own per-frame mask counters |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
