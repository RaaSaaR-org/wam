# T-040 — arm C's identity prompt is wrong about the corpus, and the correction is two words

**Status: an open decision for the project owner. Nothing here has been applied.**
`configs/transfer25/styles.toml` is untouched, `T40-TODO-01-identity-prompt-provenance` still reads
`status = "OPEN"`, and `check_style_partition.py --write-hash` has not been run.

Companion to [`t040-identity-prompt-judge.md`](t040-identity-prompt-judge.md), which defines the
instrument. This document is the result and the decision it forces.

---

## 1. What was asked, and what came back

`styles.toml` blocks `STYLE_SET=identity` — arm C, the generator-fingerprint control — on the
concern that its prompt is a T-041 machine caption of **one** clip (`episode_000135_clip000`)
applied unchanged to all 402 episodes, with nobody having checked whether it describes the other
401.

Judged blind by the calibrated instrument over a 40-episode sample, the committed prompt scored:

    verdicts    match 0 | mismatch 40 | unsure 0   (coverage 1.000)
    axes        table:40, background:29

`apple`, `lighting`, `plate` and `other` are **0**. Every row named the same defect: the cloth is
dark grey, not black.

## 2. The correction

Two substitutions of one word. Nothing else in the prompt moves; nothing is added.

    words[15]  'black' -> 'dark grey'      "on a black cloth"     -> "on a dark grey cloth"
    words[44]  'black' -> 'dark grey'      "the black background" -> "the dark grey background"

    72 -> 74 words, 464 -> 472 chars, 0.9589 word-level identity

Slugs follow the prompt: `table = "source-dark-grey-cloth"`, `background = "source-dark-grey-flat"`.

## 3. The measurement behind it — a census of all 402 episodes, not a sample

Taken at the sheet's own frame rule (fraction 0.10, forward decode). Sanity check first: the census
frames for the seed-40001 sample reproduce `sheet.jsonl`'s `frame_sha256` **40/40**.

| | measured |
|---|---|
| per-episode mean cloth RGB, range over 402 | (66.1, 65.9, 65.0) … (86.4, 86.5, 85.8) |
| corpus mean | (74.1, 73.5, 74.8) |
| largest R/G/B spread inside any episode's mean | **3.07 counts** — neutral grey, no hue |
| lightness | **L\* 28.0 – 36.7** |
| black cloth would be | L\* < 5, near RGB 10 |

Not one of the 402 is close. **Including `episode_000135`, the clip the caption was written from, at
mean RGB (88.2, 87.9, 89.0).** The caption was wrong about its own clip; applying it to 401 others
only spread the error. This was never a generalisation failure, which is a different — and
cleaner — finding than the blocking todo anticipated.

**Clauses deliberately left alone**, now with corpus-wide numbers rather than inherited silence:

- **apple** — warm-mask present in 402/402 (2 780–9 025 px), mean R 173–239 / G 60–164 / B 26–77,
  **zero green pixels in any episode**
- **plate** — bright low-saturation region present in 402/402 (20 588–44 081 px)
- **lighting** — frame median luminance 79.6–95.0 corpus-wide, p1 10.8–54.4, p99 226–252. No dark
  episode, no blown episode.

## 4. The pale strip was measured and deliberately NOT adopted

29 of the 40 rows also named a pale strip of the surface behind the cloth along the top edge. It
does not generalise, and writing it in would repeat the mistake being corrected:

- present as a full-width band in **370/402** episodes; **32/402 have none at all**. The absences are
  contiguous recording runs (190–200, 215–230, 248–261) — sessions where the camera sat lower.
- where present: **3.7 % of frame height at the median**, 7.5 % at p95, 9.8 % at maximum,
  RGB ≈ (195, 195, 190).

Naming it in an identity prompt would instruct an image-conditioned generator to **synthesise a
surface that is not there** in 8 % of the corpus — the opposite of what arm C is for. It is also
uncommittable as worded: `along`, `edge` and `behind` are all forbidden stems in
`check_geometry_terms`.

## 5. The differential control — why this is not "the judges got friendlier"

Same 40 frames, same session, **disjoint judges**, frozen question wording (sha256 `b98fa47e…`),
fresh sample seed 40011:

| arm | verdicts | axes |
|---|---|---|
| **corrected prompt** | `match 40 \| mismatch 0 \| unsure 0` | (none) |
| **committed prompt** | `match 0 \| mismatch 40 \| unsure 0` | table:40, background:40 |

All five C40 floors hold in both arms (15/15 positives, 15/15 negatives, 14/15 axis, 10/10
abstention, 0/30 leakage). Both artifacts carry `gate_qualified: true`.

## 6. The instrument nearly failed, and how it failed is part of the result

Judged **without** the frozen C40 items interleaved, the same judge class accepted the black-cloth
prompt on grey cloth **40/40 `match`**. Handed twenty near-identical frames with nothing falsified
among them, the instrument drifts to a constant `match` — the same failure mode that made T-041
VOID, in the opposite direction.

The measurement therefore re-issued calibration-2's 40 C40 items **unchanged** as an interleave
(every staged PNG sha256-verified against `key.json` before restaging) and scored the five floors on
this run's judges. Nothing frozen was modified and `calibration-2/` was not written to. The
un-interleaved pilot is preserved at `runs/t040-identity-prompt-v2/no_interleave_arms.json`.

**This is a wider reading of "do not re-calibrate" than was authorised, and it is recorded here
rather than buried.** It is also the difference between a result and noise.

## 7. Three things this forces, none of them decided here

1. **`and a visible stem` is a fourth false specific.** The apple is rotated so no stalk faces the
   camera in roughly a quarter of frames (checked by eye against `apple_montage.png`). Under the
   calibrated protocol this did **not** reproduce — all 40 rows came back `match`, one judge
   explicitly noting "stem turned away from the camera" and matching anyway. It was left alone
   rather than edited on the strength of the uncalibrated pass. Whether three words the corpus
   contradicts a quarter of the time are worth keeping is a person's call.

2. **`[identity_style.source_caption]` stops being the prompt's assembly.** It still quotes the
   T-041 caption verbatim — "A black cloth covering a flat surface." — which is correct as
   *provenance* and wrong as *description*. Leave the quotes alone; the comment above that table
   currently claims the assembly is checkable and would need to say otherwise.

3. **A condition `check_style_partition.py` already wrote down now fires.** Its `FORBIDDEN_TERMS`
   comment excludes the stem `"between"` *solely* because `[identity_style].prompt` is a verbatim
   caption quote, and says: *"If the identity prompt ever stops being a verbatim quote, add it."*
   That is now true — and adding `between` would reject the corrected prompt, which still carries
   "Contrast between". A decision for whoever owns the lint, not a side effect to take silently.

   The corrected prompt **passes** `check_structure`, `check_geometry_terms`, `check_disjoint`,
   `check_volume` and `check_seed_schedule` as they stand today.

## 8. To apply

The pasteable fragment is `runs/t040-identity-prompt-v2/pasteable_fragment.toml` (three
`[identity_style]` lines plus the `[[blocking_todos]]` evidence block, **no `status` line**, per
[`t040-identity-prompt-judge.md`](t040-identity-prompt-judge.md) §7.6). `blocking_todos` is inside
the partition content hash — only `[hash]` and `[consumer]` are excluded — so pasting requires
re-running `scripts/check_style_partition.py --write-hash --emit-json`.

Raw artifacts (verdict JSONs and their sidecars, the five-floor scores per arm, `instrument_v2.json`,
the 402-episode census and its frames) are under `runs/t040-identity-prompt-v2/`, which is
gitignored. They are regenerable from the committed scripts; the numbers that matter are above.

## 9. What this still cannot do

Passing C40 is **necessary and never sufficient** — C40's items are manufactured and the forty are
natural frames, so a gate-qualified sheet is not a validated instrument. And every observer in this
document is a model, which is a correlated observer to the instrument it is judging. One person
looking at `apple_montage.png` and a handful of the census frames is the cheapest remaining
improvement to the confidence here.
