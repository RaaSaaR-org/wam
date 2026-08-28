# PR-08 RESULT — the throughput is measured, and it is 46 % worse than the number everything was planned against

**`T40_RULE_V20` §5 outcome M.** Slurm job **191102**, 2026-08-28, `RUN_ID
t040-transfer25-restyle-timing-2026-08-28`. The outcomes M / R / F were fixed in the rule **before**
this was submitted and are not moved here.

**Every quotation of the number below must carry the clause V20 §5 attaches to it:** *measured on an
episode selected for surviving G0c.* It is in §2, and it is not decoration — §4 is what it is
because of it.

**This does not close §8 item 3.** Item 3 requires *"a measured throughput number … **and** a GPU-h
ceiling derived from it, enforced in the sbatch"*. The first half now exists. The second does not,
and §5 says who owes it. `T40_RULE_V1` §1 is not lifted and forbids generation in full.

---

## 1. The measurement

| | |
|---|---|
| **seconds per frame** | **1.6896** |
| measured on | 1 × H200, 640×480, one episode |
| episode | `episode_000371`, 422 frames, style `train-01-oak-tungsten` |
| wall clock | 713.0 s |
| units timed / succeeded | 1 / **1** |
| generator | `nvidia/Cosmos-Transfer2.5-2B@ce8440327c632d8313c3bde69db13b627ba5cae1` |
| control | `depth:0.5,seg:0.5` — PR-08's committed set, not a choice made at submit time |
| clips left on disk | **0** — one clip was generated and deleted, as V20 §5 requires |

`THROUGHPUT.json` carries `schema: wam.transfer25_throughput/1` and `units_succeeded: 1`, which is
what `throughput_qualification` (`97_transfer25_restyle.sbatch:1332-1376`) requires before any
budget may rest on it. Job 189142's artifact fails that check and still does: it is a wall clock
around a crash.

## 2. Which episode, and how it was chosen

`episode_000371` is the episode `T40_RULE_V20` §3 registered, as a **criterion and not a name**: of
the episodes both halves of `check_mask` accept, the one whose frame count is closest to the corpus
median of 421.5, ties by lowest id. It was re-derived at submit time rather than typed, and it was
re-derived again inside the job.

**The evidence that admitted it was the sixteen shard artifacts, not a pooled file** — the fix that
made this submission possible at all, after job 190981 died in six seconds on a `POOLED.json` that
no committed script writes:

```
evidence:        …/runs/pr08-robot-mask-area/shards
evidence_shards: 16, each by path and sha256
evidence_sha256: a4ec9a44ceadd3a6c548d824f449c8936511e09458a1b9bb93563b959f1bcb03
episode_000371:  422 frames, 0 empty, max area fraction 0.2271 vs bound 0.6409
```

The screen predicted G0c would not refuse; the driver's own `preflight_source_masks` recomputed the
masks on the H200 and agreed. **That is worth recording in its own right.** The 2026-08-28 low-tail
look found two frames of this same episode re-rendering at 0.939 and 0.870 — above the bound — on an
RTX 5090. The H200 reproduced its own 2026-08-25 numbers. So the disagreement is between the two
machines, and this run is one more measurement saying the seventeen are a property of the machine
that computed them, exactly as V20 §4 pre-registered.

## 3. The derivation, which is arithmetic and is not a ceiling

171 625 frames per variant (`partition_facts.json` `corpus_frames`, summed over this manifest — not
an estimate and not a round number):

| | GPU-h |
|---|---:|
| one variant | **80.55** |
| stage 1, 8 style-instances | **644.4** |
| **whole partition, 25 style-instances, 10 050 clips** | **2 013.7** |

`gpu_hours_per_variant_is_lower_bound_above_1_gpu: true` — the artifact says so itself.

## 4. What this changes, and it is the finding

**Everything sized before today used 1.16 s/frame.** That figure is where the committed *"~442
GPU-h"* for stage 1 comes from, and `F5-yield-empty-mask.md:470` shows the arithmetic reaching it:
`171 625 × 8 × 1.16 ÷ 3600 = 442.4`.

**The measured rate is 1.6896 s/frame — 45.7 % higher.**

| | at 1.16 s/frame (assumed) | at 1.6896 s/frame (measured) |
|---|---:|---:|
| stage 1 | 442.4 GPU-h | **644.4 GPU-h** |
| whole partition | 1 382.5 GPU-h | **2 013.7 GPU-h** |

**And the yield multiplies it.** `robot_composite.check_mask` refuses a clip on one empty robot mask
and on one frame above the committed area bound; **17 of 402 episodes survive both halves — 4.23 %**
(`PR-08-RESULT-2026-08-28-the-seventeen-survive-…`). Nothing in the generation loop memoises "this
episode's source refuses", so the refusal is re-discovered per style-instance at full price. Against
the whole partition:

> **85.2 GPU-h buy clips that are kept. 1 928.6 GPU-h buy clips that are then quarantined.**

That ratio was already known in shape — `F5` §5 put it at 423.7 of 442.4 for stage 1 — but it rested
on an assumed rate. **It now rests on a measurement, and the absolute number is 4.5× larger than the
one that shape was argued against.**

## 5. What is owed, and to whom

1. **The project owner — `PARTITION_CEILING_GPU_H`.** `T40_RULE_V2` §3 step 2 asks for a budget line
   derived from this measurement over the whole partition, with the derivation recorded; step 3
   splits it across the three `STYLE_SET` submissions and the sbatch checks that the shares sum
   within it (`97:434-443`, required with no default). §3 above is the arithmetic. **It is not the
   ceiling: a session may compute a number, and a budget line is a person's signature.** Item 3's
   second half is closed by that signature and by nothing else.
2. **The project owner — whether 2 013.7 GPU-h is spendable at all**, and whether 95.8 % of it
   buying quarantined clips is acceptable. This document deliberately makes no allocation claim: the
   `sreport` figure visible from the login node is **CPU** hours, and the GPU-h allocation is not
   readable from there.
3. **The project owner — the corpus-shape question, now with a price.** `PR-08-RESULT-2026-08-28-…`
   §6 already asked whether a corpus of seventeen episodes can carry §5's registered headline. The
   same question with today's number attached is whether the generation work unit should stay
   "every episode × every style-instance" when 95.8 % of that unit is known in advance to be
   discarded. **Changing the work unit is a rule change and is not proposed here.**

## 6. What this does not establish

* **Not the ceiling**, per §5.1. The artifact's own `ceiling_gpu_hours_supplied_at_measurement_time`
  is `null`, with the note that null is the expected value.
* **Not a licence.** V20 §5: *"In all three the run generates at most one clip, which is deleted.
  `T40_RULE_V1` §1 is not lifted and forbids everything else."*
* **Not a rate for any other control set, resolution or backbone.** One episode, one style, one
  control set, one checkpoint revision, one machine.
* **Not a statement that 1.16 s/frame was wrong when it was written.** It was a working figure that
  no measurement had yet touched. This is the measurement item 3 exists to require, and requiring it
  is why the gap is visible instead of being discovered at 2 000 GPU-h.
* **Not a claim about the other six §8 items.** On the reading that gives the task ledger no weight,
  four are unclosed: 3, 4, 5 and 6 — see
  `PR-08-DET-DRAFT-2026-08-28-items-5-and-6-are-not-a-silence-…`.

## 7. Reproducing it

```bash
./cluster/discoverer/submit_timing_episode.sh          # re-derives V20 §3, refuses if it moved
./cluster/discoverer/watch_timing_episode.sh           # one read-only snapshot
```

The artifact is `runs/t040-transfer25-restyle-timing-2026-08-28/THROUGHPUT.json`. The evidence that
admitted the episode is `runs/pr08-robot-mask-area/shards/`, sixteen artifacts, each named and
hashed inside the artifact's own `timed_unit_admissibility` block — so the selection is checkable
rather than trusted, which is the property `T40_RULE_V20` §3 exists to keep.
