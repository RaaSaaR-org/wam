# PR-08 V15 — the protocol for V12's missing measurement, registered before any verdict exists

**Rule `T40_RULE_V15`. Registered 2026-08-27. §5's outcomes are fixed BEFORE the measurement is
run, and the commit that lands this document contains no verdict data.** Its purpose is to make
the answer to `T40_RULE_V12` §2 unfittable.

Sits alongside [`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md)
(`T40_RULE_V1`), which is **not edited**. `docs/handoff.md` §3.

## 1. What is being measured, and why this document exists

`T40_RULE_V12` proposes changing G0c's empty-mask semantics, and §2 of that draft states its own
disqualifying hazard: it was written after seeing that the gate refuses most of the corpus.
V12 §2 names the one thing that would settle whether the proposal is a repair or a
rationalisation, and records that it has not been done:

> Take a sample of empty-mask frames, establish **by an instrument that does not involve the robot
> masker** whether the arm is in frame, and report the split. If (b) turns out to dominate, §3 is
> wrong and the correct reading is that the masker is broken.

V12 §3.5 fixes the order: **run the (a)/(b) split first**, confirm the camera geometry second,
adopt third. This document is step one. It adopts nothing.

The two cases, from V12 §1.4:

- **(a)** the robot is genuinely absent from the frame — no source robot pixels exist, the
  composite is a correct no-op, and refusing discards a good clip.
- **(b)** the masker failed on a frame where the robot IS present — source pixels existed and were
  not composited, which is the defect G0c exists to catch.

## 2. The population, and what is already known about its shape

**Population:** all 57 835 empty-mask frames (`area_fraction == 0.0`) across 402 episodes and
171 625 frames, enumerated from `runs/pr08-robot-mask-area/POOLED.json`
(`git_commit 8b710d0119b65fd3c4eff0e968e3d92edc84d2ae`, `measurement_qualified: true`).

**A partial peek is disclosed here rather than hidden.** The session registering this document has
already computed the *structural* distribution of that population — where the empty frames sit
inside their episodes — and used it to design §3's strata. It has **not** seen any (a)/(b) verdict,
because none exists. The structural figures:

| stratum | definition | frames | share |
|---|---|---:|---:|
| `S1_lead` | contiguous zero-run at the episode's **start** | 15 888 | 27.47 % |
| `S2_trail` | contiguous zero-run at the episode's **end** | 36 634 | 63.34 % |
| `S3_int_1_2` | interior zero-run of length 1–2 | 764 | 1.32 % |
| `S4_int_3_25` | interior zero-run of length 3–25 | 2 226 | 3.85 % |
| `S5_int_26plus` | interior zero-run of length ≥ 26 | 2 323 | 4.02 % |

No episode is empty on every frame. 90.81 % of empty frames are boundary runs.

**Why this peek does not contaminate the result.** These numbers are derived from the masker's own
output and are therefore evidence about the masker, not about the world — they are exactly the
silence V12 §3.2 refuses to read as absence. They are a legitimate basis for *stratification*,
which only affects precision, and an illegitimate basis for *inference*, which §5 forbids by fixing
the outcomes before any verdict is seen. A reader who believes the strata were drawn to produce a
wanted answer should note that §5's thresholds are stated on the population-weighted quantity, so
re-weighting the strata cannot move them.

## 3. The instrument

**A person, shown the raw source frame, asked whether any robot is visible. No mask is displayed.**

This is the masker-independent instrument V12 §2 requires, and displaying no mask is what makes it
independent: the correlated-observer failure this project has already recorded — a model reading
masks produced by a model-built pipeline is not a check on that pipeline — applies with equal force
to a person shown the pipeline's answer before giving their own.

**Rendering.** Full frame at native 640×480, decoded forward from
`/home/humanoid/wam-t041/pr08-apple-640x480/videos/<episode>.mp4` at the frame index, never seeked.
No overlay, no crop, no annotation, no stratum label, no episode identifier visible to the reader.
Tiles are presented in an order fixed by the sampling seed, which shuffles strata together.

**The question, exactly as it will be asked:**

> Is any part of the robot — arm, hand, or gripper — visible anywhere in this frame?
> **yes** / **no** / **cannot tell**

**`yes` is case (b). `no` is case (a).** `cannot tell` is neither and is handled by §4.

**Sample:** 240 frames, stratified, drawn uniformly at random within each stratum under
`sample_seed = 40015`, allocated `S1 60 / S2 60 / S3 40 / S4 40 / S5 40`. The three interior strata
are over-sampled relative to their 9.19 % mass on purpose: they are where a masker failure is most
likely and where the population estimate is least precise. §5's quantity re-weights to true
population shares, so the over-sampling buys precision without buying a direction.

## 4. Handling of `cannot tell`, fixed in advance

The last human look under this project's review page returned 27 of 40 unbiased tiles as
undecidable, and a protocol that does not plan for that in advance will improvise around it
afterwards. Therefore:

- `cannot tell` tiles are **excluded from the numerator and the denominator** of §5's quantity, and
  their count and per-stratum distribution are reported alongside it.
- **If more than 25 % of tiles in any stratum come back `cannot tell`, that stratum's estimate is
  declared unusable** and §5 is not evaluated. The finding is then that this instrument does not
  answer the question at this rendering, and V15 is superseded by a version that changes the
  rendering — not by a version that reinterprets the tiles.
- No tile may be re-judged after §5 has been evaluated.

## 5. The decision rule, fixed before the data

Let **`p_b`** be the population-weighted fraction of empty-mask frames that are case (b):
`p_b = Σ_s (N_s / 57835) · (b_s / (b_s + a_s))` over the five strata, with a Wilson 95 % interval
per stratum propagated to the weighted total.

Let **`n_survive`** be the number of the 402 episodes estimated to contain **zero** case-(b)
frames — the count of episodes that would survive a rule passing (a) and refusing (b). It is
reported as model-dependent, under the stated assumption that (b) occurrences are independent
within an episode given its stratum composition, and it does not enter Q1.

**Q1 — V12 §2's question. Does (a) dominate?**

| outcome | condition | what it licenses |
|---|---|---|
| **A** | `p_b` upper CI bound ≤ 0.05 | (a) dominates. G0c's empty-mask refusal is asking a question it cannot interpret, and the V12 §3.2 route is licensed **to be built and validated** — not adopted. |
| **M** | otherwise, and `p_b` lower CI bound < 0.33 | Mixed. Neither §3.2 nor §3.3 is licensed on this evidence. A further rule version is required, and it must address how a witness validated to what standard may carry a (b) rate of this size. |
| **B** | `p_b` lower CI bound ≥ 0.33 | (b) is material: the masker is failing on this corpus. **V12 §3.3 is the answer** — G0c is left alone and `T40_RULE_V1` §3's compositing route is revisited rather than its gate adjusted. |

**Q2 — the practical question. Does a corpus survive?** Evaluated only if Q1 returns **A**.

| outcome | condition | what it licenses |
|---|---|---|
| **A2** | `n_survive` ≥ 302 (¾ of 402) | Building the §3.2 witness is worth its cost. |
| **M2** | 101 ≤ `n_survive` < 302 | A corpus survives but a quarter to three quarters is lost; whether that is worth the machinery is an owner decision, not a session one. |
| **B2** | `n_survive` < 101 (¼ of 402) | Not enough corpus to justify the witness. §3.3 by the practical route. |

**These five thresholds are coined, not measured, and this document says so rather than dressing
them as derived.** `T40_RULE_V13` §3.4 forbade a coined bound for the area gate because there a
measured distribution existed and a gap could be read off it. Here no such distribution exists:
`p_b` is the very quantity being measured for the first time, so there is nothing to read a
threshold off, and a pre-registered coined threshold is the correct instrument precisely because
coining one afterwards would not be. That asymmetry is the whole content of `handoff.md` §3.

**The §3.2 route is not adopted under any outcome above.** Outcome **A** licenses building a
witness; V12 remains unsigned, and its §3.2 objection — that camera intrinsics and extrinsics for
AppleToPlate are committed nowhere in this repository — is untouched by anything here.

## 6. The auxiliary instrument, and the condition on using it

240 tiles cannot adjudicate 57 835 frames. A **masker-independent** auxiliary is therefore computed
over the whole population — per-frame motion energy against the episode's own neighbouring frames,
raw pixel differencing, no model of any kind — and calibrated against the human labels.

**It may be used to report a corpus-wide split only if**, on a held-out third of the human-labelled
tiles not used to fit it, it reaches **balanced accuracy ≥ 0.90**. Below that it is reported as
having failed and the finding rests on the stratified sample alone, with its interval. The
threshold is coined and fixed here for the same reason as §5's.

The auxiliary never overrides a human label, and no §5 outcome may be evaluated on auxiliary
labels.

## 7. What this document does not do

Adopts nothing, signs nothing, discharges nothing. `T40_RULE_V12` stays an unsigned draft,
`GATE_QUALIFIED` stays `False`, `GATE_QUALIFICATION_BLOCKERS` is not shortened by it,
`T40_RULE_V1` §1 binds in full, and no clip is licensed.

Prepared by a Claude Code session under the owner's instruction of 2026-08-27, verbatim:
**"b) entscheide du was für uns am besten ist"**. That instruction delegates the choice of protocol.
It is **not** a signature on `T40_RULE_V12`, which §5 leaves exactly where it was.
