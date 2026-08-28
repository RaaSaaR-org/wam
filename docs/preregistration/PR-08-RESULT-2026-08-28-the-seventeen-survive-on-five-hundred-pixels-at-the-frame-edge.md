# PR-08 RESULT — the seventeen survive on five hundred pixels at the frame edge, and the timing episode re-renders above the bound

**Evidence, under `T40_RULE_V13` §3.2 — *"whether those frames were looked at, and what they
were."* This document writes no bound, moves no bound, proposes none, and discharges no blocker.
It records a look nobody had taken.**

Artifact: `runs/pr08-area-low-tail-look/TAIL_SAMPLE.json`, `SCHEMA pr08-area-tail-look/1`, rendered
at `9d6dd89` on an RTX 5090 (`torch 2.13.0+cu130`, CUDA 13.0), with the committed masker at its
committed operating point and the committed `ROBOT_TEXT_PROMPT`.

---

## 1. Why this look and not another

`robot_composite.check_mask` refuses a clip on **one** frame with an empty robot mask, and refuses
one on **one** frame above the committed area bound `0.6409114583333333`. Over the corpus
(`runs/pr08-robot-mask-area/POOLED.json`, 402 episodes, 171 625 frames, `measurement_qualified:
true`), **17 episodes survive both halves** — 7 309 frames, **4.26 %** of the corpus. That set is
the entire population a corpus run would produce today.

The upper tail of the area distribution has now been looked at twice: 48 tiles with a human pass,
and a blind re-read. **The lower tail had never been looked at at all** — and the lower tail is
where survivorship is decided, because `check_mask`'s empty-mask half refuses on `covered == 0`.
An episode is in the seventeen because its *smallest* mask was not empty.

## 2. What the lower tail of the survivors looks like, numerically

Recomputed here from `POOLED.json` and the committed bound:

| | |
|---|---:|
| survivor frames below area fraction 0.005 | **1 039 of 7 309 (14.2 %)** |
| survivor frames below 0.01 | 2 170 of 7 309 (29.7 %) |
| survivors with at least one frame below 0.005 | **8 of 17** |
| survivors with none | 9 of 17 |

The split is clean, and it is the first thing to notice. The nine clean survivors bottom out at
0.0057–0.0177 (1 762–5 440 px of 307 200). The eight others bottom out far lower:

| episode | frames | `< 0.005` | min fraction | min px |
|---|---:|---:|---:|---:|
| `episode_000093` | 448 | 231 | 0.002529 | **777** |
| `episode_000121` | 448 | 203 | 0.001576 | **484** |
| `episode_000120` | 464 | 199 | 0.002520 | 774 |
| `episode_000098` | 439 | 175 | 0.002705 | 831 |
| `episode_000373` | 366 | 100 | 0.003555 | 1 092 |
| **`episode_000371`** | **422** | **76** | **0.002005** | **616** |
| `episode_000375` | 418 | 48 | 0.003320 | 1 020 |
| `episode_000243` | 417 | 7 | 0.004867 | 1 495 |

`episode_000371` is the episode `T40_RULE_V20` §3 registered for the `§8` item 3 throughput
measurement, and the episode Slurm job **190981** is timing.

## 3. What they are, having looked

48 frames were rendered from those eight episodes with the mask overlaid — the band
`0.0001 ≤ f ≤ 0.005`, 1 039 candidates, an even stride, `--max-frames 48` for comparability with
the two upper-tail looks. Recorded fractions across the sample span **493 to 1 528 px**.

**In the frames that reproduce, the masked region is a thin sliver at the left or right border of
the picture, and the visible scene is an apple, a plate and a grey tablecloth.** No robot arm, hand
or gripper is in frame. Whether that sliver is the very edge of a robot part just outside the
picture, or an artifact, cannot be settled at this resolution and **is not claimed here.** What is
claimed is narrower and sufficient:

> **Membership in the seventeen is decided by five hundred to fifteen hundred pixels at the frame
> border, in frames whose visible content is apple, plate and tablecloth.**

This bears on `runs/pr08-robot-mask-empty/DIAGNOSIS.json` from the other side. That artifact asked
whether the masker returns *empty* on frames where a robot is *present*, and answered no — 0 of 917
and 0 of 240. This look asks the converse, on the population that survives, and the converse is
where the yield comes from.

## 4. The finding that changes something on the cluster

**11 of the 48 frames (23 %) re-render beyond the artifact's 0.01 tolerance on this machine.** Two
of them re-render **above the committed area bound**, and both are in `episode_000371`:

| episode | frame | recorded | recomputed here | vs bound 0.64091 |
|---|---:|---:|---:|---|
| `episode_000371` | 348 | 0.003057 | **0.939349** | **above** |
| `episode_000371` | 409 | 0.002920 | **0.870446** | **above** |

At 0.94 the mask is the tablecloth. Elsewhere in the sample it is the plate (0.074–0.102). These
are the false positives `DIAGNOSIS.secondary_finding_false_positives` records, met here at the
bottom of the distribution instead of the top.

**Consequence, stated and not decided.** `T40_RULE_V20` §4 pre-registered that the seventeen are a
property of the machine that computed them, and §5 fixed **outcome R** — the source-mask preflight
refusing `episode_000371` on the H200 — as a *result* rather than an accident. This look moves R
from a caveat to a **local measurement**: on an RTX 5090, `check_mask`'s area half refuses
`episode_000371` on at least two of its 422 frames, and one refused frame refuses the clip.

V20 §5 also fixed what R does **not** license: *"outcome R does not license walking down the 17
until one passes."* That still binds, and this document does not touch it. Whether job 190981
returns M, R or F is still the cluster's answer and not this workstation's.

## 5. What this does not establish

* **Not that the seventeen are wrong.** Nine of them carry no sub-0.005 frame at all and are
  untouched by anything above.
* **Not that a robot is absent.** A sliver at the picture's border is not resolvable into "arm
  just outside the frame" or "artifact" at this scale, and no reviewer confirmation is recorded
  anywhere in the artifact — there is deliberately no field for one.
* **Not a bound.** Neither edge of the band `0.0001 ≤ f ≤ 0.005` is a bound or a candidate bound.
  `T40_RULE_V13` §3.4's freeze is untouched.
* **Not a corpus-wide rate.** 48 frames of 1 039 candidates in 8 of 402 episodes, on one machine.
* **Not a licence.** `T40_RULE_V1` §1 is unchanged and forbids generation in full.

## 6. What is now owed, and to whom

1. **The owner** — whether a corpus of seventeen episodes selected this way can carry PR-08 §5's
   registered headline. `configs/transfer25/styles.toml` sizes all three arms over 402 episodes;
   at seventeen the arms become 170 / 170 / 85 clips, and the survivor ids cluster in contiguous
   blocks (93, 98, 114–121, 136–137, 243–245, 371–375) rather than spreading.
2. **The owner** — whether the robot-mask area pass is re-measured on the machine that will
   generate, before the seventeen are relied on. V20 §5 already owes this under outcome R; this
   look says the same debt exists under outcome M.
3. **Nobody yet** — a reviewer pass over the sheets. This document records what was rendered and
   what it measured; it records no verdict, and the tool has no field in which one could be put.

## 7. Reproducing it

```bash
.venv/bin/python scripts/render_area_tail_sheet.py \
    --episodes-surviving-composite \
    --threshold 0.0001 --max-fraction 0.005 --max-frames 48 \
    --out runs/pr08-area-low-tail-look
```

`--episodes-surviving-composite` derives the population from `POOLED.json` and
`configs/transfer25/pr08_robot_mask_area.json` rather than naming seventeen ids, for the reason
`T40_RULE_V20` §3 gives about a different selection: *"The rule is a criterion, not a name, so that
it can be checked rather than trusted."* If either input moves, the population moves with it and
the artifact records what it resolved to.
