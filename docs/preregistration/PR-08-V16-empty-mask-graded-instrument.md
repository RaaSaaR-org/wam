# PR-08 V16 — V15's instrument failed where it mattered, and the question it asked was the wrong one

**Rule `T40_RULE_V16`. Registered 2026-08-27, superseding `T40_RULE_V15` §§3-6 under V15 §4.
V15 §§1-2 (the population and the strata) are UNCHANGED and carried forward verbatim.**
Registered before any tile is re-rendered and before the auxiliary's distribution is looked at.

## 1. Why V15 was superseded, in its own terms

V15 §4 fixed in advance what happens if the tiles come back undecidable, and it happened:

| stratum | judged | `cannot_tell` | rate | V15 §4 cap 25 % |
|---|---:|---:|---:|---|
| `S1_lead` | 24 | 3 | 12.5 % | |
| `S2_trail` | 23 | 1 | 4.3 % | |
| `S3_int_1_2` | 17 | 12 | **70.6 %** | **over** |
| `S4_int_3_25` | 15 | 5 | **33.3 %** | **over** |
| `S5_int_26plus` | 22 | 3 | 13.6 % | |
| total | 101 | 24 | 23.8 % | |

Two strata over the cap, so **V15 §5 was not evaluated and no split was computed.** The protocol
stopped itself, which is the only thing a pre-registered undecidability cap is for.

**The failure is not uniform, and its shape is the finding.** Where the arm is out of the picture —
the boundary runs that carry 90.8 % of the population — the question is answerable. Where the
decision actually turns, the short interior dropouts that are the classic masker-failure signature,
**seven tiles in ten could not be judged at all.**

## 2. What the reviewer reported, which is evidence and is quoted rather than summarised

> "sieht wieder so aus, als ob nur 'fingerspitzen' zu sehen sind - und auch, nur einen schatten -
> das war das unklar. nur einige hatten wirklich den richten 'arm' zu sehen - nicht nur teile der
> hände. also es is noch 'zu früh' - vom video verlauf her um hier wirklich validate aussagen zu
> machen. Sonst nur schatten oder fingerspitzen, was schwer zu unterscheiden ist"

Three claims, and each one bears on a different part of the protocol:

1. **A definite arm is distinguishable and is rare.** The reader could tell that class apart.
2. **A fingertip and a shadow are not distinguishable from one frame.** They are the residue.
3. **The empty-mask population is dominated by the arm entering or leaving the frame** — which is
   what "zu früh vom Videoverlauf her" describes, and it matches the structure independently:
   90.8 % of empty frames are contiguous runs at an episode's start or end.

A rendering fix was attempted against claim 2 and is recorded as having **failed**: a local
normalised cross-correlation between the frame and its episode's static background, on the theory
that a shadow scales the cloth's weave and preserves the correlation while an object destroys it.
On this corpus the dark cloth carries too little texture where it matters, so the statistic is
noisy exactly in the region the discrimination is needed. It is not used.

## 3. The reframe, which is the substance of this version

**V12 §1.4's binary is not answerable at the margin, and the gate does not need it answered.**

G0c exists so that generated robot pixels cannot enter the corpus uncomposited. What that requires
is not a classification of every empty mask into (a) and (b); it requires knowing **how much robot
could have gone uncomposited**. Those are different questions, and the second one is decidable
where the first is not: whatever the dark shape at the frame edge is, finger or shadow, its **area
bounds the harm**. A shadow contributes zero robot pixels and a fingertip contributes its own area,
so the area of the changed region is an **upper bound on the uncomposited robot area** without the
classification ever being made.

That is why this version grades the question instead of sharpening the rendering. The reader is
asked only for distinctions they demonstrably can make, and the undecidable residue is handed to a
measurement that does not need to decide it.

**This is a change to the QUESTION, and V15 §4 licenses a change to the rendering.** The wider
change is claimed under the same clause and the claim is made explicitly so it can be refused: §4
forbids "a version that reinterprets the tiles", and none of the 101 verdicts is reinterpreted or
carried forward — see §7. Every tile is judged again, from scratch, under the vocabulary below. A
reader who holds that §4 permits only a rendering change should refuse V16 and require a new
sample; the cost of that is the reviewer's time, and it is stated here rather than glossed.

## 4. The instrument

**Population, strata, sample, seed and allocation are V15's, unchanged.** The same 240 tiles, the
same `sample_seed = 40015`, the same `S1 60 / S2 60 / S3 40 / S4 40 / S5 40`. Nothing about the
draw is re-rolled, because re-drawing after seeing a stratum fail is how a sample gets fitted.

**Still no mask, no overlay, no area fraction, no stratum, no episode id, no frame index.** V15 §3's
prohibition is carried forward in full and for the same reason: a reader shown the pipeline's answer
is not a check on the pipeline.

**The question, graded, in the reviewer's own distinctions:**

> **A** — a **definite arm or hand**: unmistakable robot structure, more than an edge fragment
> **B** — only a **small dark thing at the edge**: a fingertip, a sliver, or a shadow — the class
>   that cannot be resolved from one frame, and is not asked to be
> **C** — **nothing**: a clean scene
> **D** — genuinely undecidable

**B is not a hedge and must not be read as one.** It is a positive finding with a defined
consequence in §5. `D` remains available and V15 §4's 25 % cap applies to it unchanged; if `D` is
still over cap after `B` has absorbed the finger/shadow ambiguity, the instrument has failed a
second time and the finding is that this question cannot be answered from single frames at all.

**Two rendering changes, both aids to finding rather than to deciding.** A brightness boost and a
zoom were already present; a toggle is added that marks where the frame differs from its own
episode's static background, so a reader does not have to hunt a dark sliver against dark cloth. It
points; it does not classify, and it is derived from raw pixel medians with no model of any kind.

## 5. The mechanical bound, and how it is used

For every one of the 57 835 empty-mask frames, `runs/pr08-empty-mask-look/MOTION.json` already
records `frac_dev` — the fraction of the frame deviating from that episode's pixel-wise median
background by more than 25 grey levels, computed with no detector and no segmenter. **Taken as an
upper bound on uncomposited robot area, it is conservative in the safe direction**: it counts the
apple's displacement, head motion and every shadow as though all of it were robot.

**Its distribution has not been looked at as this is written**, which is why the threshold in §6 can
be fixed here.

## 6. The decision rule, fixed before the re-judged data and before the bound's distribution

Let **`p_A`** be the population-weighted fraction of empty-mask frames graded **A**, Wilson 95 % per
stratum combined in quadrature — V15 §5's estimator, over the `A` grade instead of `yes`. `D` is
excluded from numerator and denominator; `B` and `C` are denominator, not numerator.

Let **`q99`** be the 99th percentile of `frac_dev` over the empty-mask frames of the whole
population.

| outcome | condition | what it licenses |
|---|---|---|
| **A** | `p_A` upper CI ≤ 0.05 **and** `q99` ≤ 0.01 | Definite arms are rare in empty masks, and the worst case for the rest is under 1 % of a frame. G0c's clip-fatal empty-mask refusal is disproportionate to the harm it prevents, and a **bounded** rule is licensed to be drafted — not adopted, and not by this document. |
| **B** | `p_A` lower CI ≥ 0.33 | The masker is failing on frames with a plain arm in them. **V12 §3.3**: leave G0c alone and revisit `T40_RULE_V1` §3's compositing route. |
| **M** | otherwise | Neither is licensed. A further version, which must say what it does about whichever of the two conditions failed. |

**`0.01` is coined and fixed before the distribution is seen, and this document says so rather than
dressing it as derived.** The reasoning it rests on, which a signer may reject: a gate that refuses
a whole clip of several hundred frames because one frame carries a sub-1 % region that *might* be a
fingertip is disproportionate on its face, whatever that region turns out to be. `0.05` and `0.33`
are carried over from V15 §5 unchanged, so that this version cannot be accused of moving the bar it
already failed to reach.

**Q2 (`n_survive`) is carried forward from V15 §5 unchanged** and stays model-dependent, reported,
and outside Q1.

## 7. What is not carried forward

**The 101 verdicts of the V15 run are not reinterpreted, not mapped onto A/B/C/D, and not used in
any estimate.** They stand as exactly one thing: the §1 diagnostic that retired V15's instrument.
`runs/pr08-empty-mask-look/VERDICTS-partial-101.json` records them under that status.

## 8. What this does not do

Adopts nothing, signs nothing, discharges nothing. `T40_RULE_V12` remains an unsigned draft, its
§3.2 objection about missing camera geometry is untouched, `GATE_QUALIFIED` stays `False`,
`GATE_QUALIFICATION_BLOCKERS` is unchanged, `T40_RULE_V1` §1 binds in full, and no clip is licensed.

Prepared by a Claude Code session. The session had seen the V15 partial run's verdict counts and the
per-stratum `cannot_tell` rates in §1 before writing this; it had not seen the per-stratum yes/no
split, and did not compute it. Disclosed because "written blind" would otherwise overstate it.
