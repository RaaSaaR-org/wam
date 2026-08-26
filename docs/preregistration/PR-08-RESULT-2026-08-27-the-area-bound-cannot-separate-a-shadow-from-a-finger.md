# PR-08 result, 2026-08-27 — the bound is valid, it is not tight, and the thing it cannot separate is the thing it was built to replace

**Rules in force: `T40_RULE_V15` (superseded §§3-6), `T40_RULE_V16` §§5-6. Nothing here is a
discharge, a signature, or a licence.**

## 1. V15's instrument stopped itself

V15 §4 fixed a 25 % `cannot_tell` cap per stratum before any tile was rendered. On the 101 tiles
judged before the run was halted:

| stratum | judged | `cannot_tell` | rate | cap |
|---|---:|---:|---:|---|
| `S1_lead` | 24 | 3 | 12.5 % | |
| `S2_trail` | 23 | 1 | 4.3 % | |
| `S3_int_1_2` | 17 | 12 | **70.6 %** | **over** |
| `S4_int_3_25` | 15 | 5 | **33.3 %** | **over** |
| `S5_int_26plus` | 22 | 3 | 13.6 % | |

**V15 §5 was not evaluated and no split was computed.** The per-stratum yes/no counts were
deliberately not computed either, so that V16 could be written without them.

The distribution of the failure is the finding: the question is answerable across the 90.8 % of the
population where the arm is out of the picture, and unanswerable in the short interior dropouts that
are the one stratum whose answer would distinguish a masker failure from an absent robot.

## 2. One rendering repair was tried and failed

**Local normalised cross-correlation between the frame and its episode's pixel-wise median
background.** The theory is sound — a shadow scales the cloth's weave and preserves the correlation,
an object replaces the weave and destroys it. On this corpus it does not work: the cloth carries
too little texture in the dark regions, so both variances are near zero exactly where the
discrimination is needed and the statistic is noise there. Recorded so it is not rebuilt.

## 3. The bound, measured

`T40_RULE_V16` §3 replaced the undecidable classification with a bound: whatever the dark shape at
the frame edge is, its **area** bounds the uncomposited robot area, because a shadow contributes
zero robot pixels and a fingertip contributes its own. §6 fixed the threshold `q99 ≤ 0.01` **before
the distribution was looked at**.

`frac_dev` over all **57 835** empty-mask frames of **366** episodes
(`runs/pr08-empty-mask-look/MOTION.json`, no detector, no segmenter):

| | value |
|---|---:|
| median | 0.02490 |
| p90 | 0.05155 |
| p95 | 0.05710 |
| **p99** | **0.07180** |
| max | 0.12230 |

**`q99 = 0.0718` against a threshold of `0.01`. The condition fails by a factor of seven.**
87.24 % of empty-mask frames (50 457) exceed 1 % of the frame.

## 4. What drives it — measured, not asserted

Over ten episodes and 1 403 empty-mask frames, with the apple's swept region removed by
`T40_RULE_V9`'s own warm-saturated colour test:

| region counted | median | p99 | max |
|---|---:|---:|---:|
| everything | 0.0205 | 0.0852 | 0.0902 |
| apple excluded | 0.0076 | 0.0408 | 0.0500 |
| apple and bright background excluded | 0.0060 | 0.0360 | 0.0489 |

**The apple is the dominant contributor to the typical frame** — the median falls by 63 % when its
swept region is removed, which V16 §5 anticipated as a confound and which is now a measurement.

**But the bound still fails without it.** p99 stays at 0.041, four times the threshold. Whatever a
tighter bound excluded next, it would still be counting the thing that is actually large.

## 5. Why the reframe fails on its own terms, and this is the finding

The residue after the apple is **shadow**. Shadows on this corpus are large — several percent of a
frame — and **a shadow contributes exactly zero robot pixels no matter how large it is.**

So the area bound is dominated by the one class whose harm is identically zero, while the class it
was built to bound — a fingertip's few hundred pixels — is far below the noise the shadow makes.
**Area cannot separate a zero-harm shadow from a some-harm finger, and that is precisely the pair
the human reader could not separate either.** V16 §3 proposed to convert an undecidable
classification into a decidable bound; the conversion does not survive contact with this corpus,
because the undecidable pair is also the pair the bound conflates.

Tightening the bound further would mean excluding shadows — which requires deciding what is a
shadow, which is the original undecidable question. The route closes on itself.

## 6. What this does to `T40_RULE_V16` §6, stated without repairing it

V16 §6 outcome **A** requires `p_A` upper CI ≤ 0.05 **and** `q99` ≤ 0.01. The second condition has
now failed on a measurement, so **outcome A is unreachable.** The remaining outcomes are **B** (the
masker is failing on frames with a plain arm, and V12 §3.3 applies) and **M** (a further version).

**That rule is not rewritten here, and the conjunction is not loosened.** It was registered before
the distribution was seen, it produced a constraining result, and a version that separated the two
conditions after one of them failed would be the exact move `docs/handoff.md` §3 forbids. A later
version may separate them; it must say that it is doing so, and why, and it may not call the result
a discharge.

**The human half is still worth running.** `p_A` — how often an empty mask hides a *definite* arm —
is the number every route needs, it decides between **B** and **M**, and the reviewer's report says
that class is the one they can judge.

## 7. What this does not do

Adopts nothing, signs nothing, discharges nothing. `T40_RULE_V12` remains unsigned and its §3.2
objection about missing camera geometry is untouched, `GATE_QUALIFIED` stays `False`,
`GATE_QUALIFICATION_BLOCKERS` is unchanged, `T40_RULE_V1` §1 binds, and no clip is licensed.
