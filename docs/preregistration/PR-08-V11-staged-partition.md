# PR-08 V11 — generating four train styles first, and deferring the eval set

**Rule `T40_RULE_V11`. Registered 2026-08-23, before a single clip of the partition has been
generated.** `GEOM_TOL` and `EST_DRIFT_P95` are both still `null` in
`configs/transfer25/pr08_geom_tol.json` as this is written, no G0 gate has ever returned a verdict,
and the only Cosmos-Transfer2.5 frames that exist anywhere in this project are the 384 quarantined
diagnostic frames of `T40_RULE_V8`'s hallucination probe (job 189926).

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. The discipline is `docs/handoff.md` §3 — *"Rules are
versioned, never edited in place. A gate rewritten after seeing its output is not a gate."* V11 is
that versioning, not a revision. `T40_RULE_V2` … `T40_RULE_V10` stand unchanged; V11 depends on V2
(arm C frame matching) and V3 (the seed schedule) and **changes nothing inside either**.

Task: [[T-040]]. Generator: **Cosmos-Transfer2.5, frozen**.

**Nothing in this document licenses generation.** `T40_RULE_V1` §1's prohibition is untouched and
binds in full, every §6 gate is undischarged, and §8 items 3 and 4 are open. V11 decides **what
would be generated first if generation were licensed** — it does not license it.

---

## 0. What V11 does not change

- **The committed style partition is untouched.** `configs/transfer25/styles.toml` keeps all ten
  `train_styles`, all five `eval_styles` and `repeats = 10` on the identity style. No style is
  deleted, reworded or moved between sets, and the partition hash does not change. §5's guarantee —
  *"the evaluation domain cannot be chosen after seeing which restyles came out well"* — depends on
  that file being fixed before the first clip, and it stays fixed.
- **Arm C is not weakened.** `T40_RULE_V2` requires arm C to match arm B's frame count. V11 keeps
  them matched at every stage. It never generates a train style without its identity counterpart.
- **No gate, threshold, cap or budget rule.** `MATERIAL_FLOOR_PP`, the L1/L2 ladder, `GEOM_TOL -
  EST_DRIFT_P95`, `max_frame_fraction`, `MAX_RESTARTS` and §8 item 3's requirement that a ceiling be
  *derived from a measurement* are all untouched.
- **The seed schedule.** V3 gives train and identity the same ten seeds, indexed by STYLE index and
  not by repeat index. Generating a prefix of the styles uses a prefix of those seeds. No seed is
  coined, re-drawn or re-assigned.

## 1. The finding: the committed partition is 27 % of the allocation, spent before any gate has
   returned a verdict

The whole partition is 25 style-instances over 402 episodes — **10 050 clips, 4 290 625 frames**,
and `97_transfer25_restyle.sbatch` already prices it as one number that the three `STYLE_SET`
invocations must split.

The only measurement of what a Cosmos-Transfer2.5 frame costs comes from job 189926, the V8 probe:
**96 frames in ~111 s of H200 time, 1.16 s/frame** (two chunks, `Average time per chunk: 55.47`).
That is a diagnostic-sized sample at 480×640 with `depth:0.5,seg:0.5`, and §8 item 3 still requires
a proper `TIMING=1` run before any ceiling is derived — **this figure does not satisfy item 3 and
nothing here may be used as the budget line.** As an order of magnitude, however:

```
4 290 625 frames x 1.16 s/frame  =  4 977 125 s  =  ~1 380 GPU-h
allocation                                          5 000 GPU-h
                                                    -> ~27 %
```

Roughly a quarter of everything this project has, committed to a generator whose geometric fidelity
has **never once been measured**: G0a, G0b and G0c have not returned a verdict between them, the
robot masker was grounding the apple until `T40_RULE_V9` this afternoon, and the mask-validity
reference is under two open defects recorded in `T40_RULE_V10`.

Billing is not the constraint — at `--mem=32G` the partition is ~4 % of the billing allocation. The
constraint is GPU-hours, and they are the resource that strands the whole project when they run out
(`docs/discoverer.md` §9).

## 2. What V11 changes, precisely

**Generation proceeds in stages. Stage 1 is the first four `train_styles` in committed order, with
their four matched identity repeats. The eval set is DEFERRED.**

| | style-instances | clips | ~GPU-h at 1.16 s/frame |
|---|---|---|---|
| **Stage 1 — arm B** | `train_styles[0:4]` | 4 × 402 = 1 608 | ~221 |
| **Stage 1 — arm C** | identity, `repeats` 1–4 | 4 × 402 = 1 608 | ~221 |
| **Stage 1 total** | **8 of 25** | **3 216** | **~442** |
| Deferred — arm B remainder | `train_styles[4:10]` | 2 412 | ~331 |
| Deferred — arm C remainder | identity, repeats 5–10 | 2 412 | ~331 |
| Deferred — eval | all 5 `eval_styles` | 2 010 | ~276 |

Stage 1 is **8.8 % of the allocation instead of 27 %**.

### 2.1 Which four, fixed here and not chosen later

`train-01-oak-tungsten`, `train-02-linen-overcast`, `train-03-melamine-fluorescent`,
`train-04-slate-lowkey` — **the first four in the committed file's order, taken as a prefix.**

A prefix and not a selection, deliberately. Any rule of the form "the four most visually distinct"
or "the four that restyle best" is a choice made with knowledge this document must not have, and it
is exactly the failure §5 guards against on the eval side. The prefix is verifiable by anyone
reading `styles.toml` and requires no judgement.

What the prefix happens to span, recorded as an observation and **not** as the reason for choosing
it: four distinct apple varieties (green Granny Smith, Golden Delicious, deep red matte, russet
brown), four table materials (oak, blue linen, white melamine, grey slate), four backgrounds and
four lighting regimes — warm tungsten side, cool overcast diffuse, bright fluorescent overhead, and
a moody dim desk lamp.

**Three of those four apples are non-warm**, which puts stage 1 squarely inside the defect
`T40_RULE_V10` addresses: `apple_sam2.object_color_reference` is a warm-and-saturated predicate, and
on `train-01-oak-tungsten` it fires on ~34 632 px of oak table and ~1 000 px of green apple. That is
not a reason to reorder the prefix — reordering to make a measurement easier is choosing the
experiment to fit the instrument. It is a reason V10 must land before stage 1's output is measured,
and it is recorded here so the dependency is visible rather than discovered.

### 2.2 Why the eval set is deferred and not cut

`eval_styles` exists to measure generalisation to an unseen domain. It is consumed **at evaluation
time, once**, and no arm trains on it. Generating it before stage 1 has shown any signal spends
~276 GPU-h on the held-out domain of an experiment that may not have an effect to generalise.

Deferral is sequencing, not a scope reduction: the five styles stay committed in `styles.toml`,
unmodified, and are generated in full before any arm is evaluated. **The partition may not be
re-cut after stage 1's results are seen** — that is the §5 prohibition, and V11 does not touch it.

### 2.3 The ceiling arithmetic

§8 item 3 and `T40_RULE_V2` §3 require a GPU-h ceiling derived from a measured throughput, split
across the `STYLE_SET` invocations so the shares sum within it. `97_transfer25_restyle.sbatch`
already implements this via `PARTITION_CEILING_GPU_H` and per-invocation `CEILING_GPU_H`, with
committed shares of **10/25, 5/25 and 10/25**.

Under V11 the shares of a stage are the stage's own composition. **Stage 1 is 4/8 train and 4/8
identity, with eval's share not drawn.** `PARTITION_CEILING_GPU_H` remains what §8 item 3 defines —
the ceiling over the **whole** 25-instance partition, derived from a `TIMING=1` run — because a
ceiling that shrinks with the stage would let a sequence of small stages exceed the number nobody
measured. **Stage 1 draws at most 8/25 of it.**

This requires a code change to `97_transfer25_restyle.sbatch`: the expansion currently prices the
whole partition and the share check hard-codes 10/25, 5/25, 10/25. That change is implementation of
this rule and must not alter the derivation, the refusal on a missing ceiling, or the requirement
that the number come from a measurement. **It has not been made as this is written.**

## 3. What decides whether stage 2 happens

Stage 1 is not a pilot to be waved through. It ends in the same four arms §5 registers — A, B, C, D
— trained under §8 item 1's recipe on 402 real episodes plus 1 608 restyled and 1 608 identity
clips, and scored on the committed ladder.

The stage-2 decision, fixed in advance:

- **Arm B beats arm A *and* arm C on L1** → generate the remaining six train styles and their six
  identity repeats, then the eval set. The effect is real and larger k should sharpen it.
- **Arm B beats arm C but not arm A**, or the reverse → the intervention is doing something other
  than what §5 says it does. Stop and read, do not scale.
- **Arm B does not beat arm C** → the gain, if any, is the generator's fingerprint and not visual
  diversity, which is precisely what arm C exists to detect. **Scaling k does not fix that**, and
  spending the remaining ~938 GPU-h would buy a larger version of an uninterpretable result.

**The threat to validity of stopping early, recorded because it is real.** A null at k = 4 is
weaker evidence than a null at k = 10: domain randomisation may need more variety than four domains
before it does anything, and stage 1 could produce a false negative that ends a line of work that
would have succeeded. This is accepted deliberately, on two grounds. First, the sbatch is chunked
and resumable by design, so scaling k = 4 → 10 costs the difference and not a restart — the failure
is recoverable in GPU-hours, unlike spending them. Second, four domains spanning four apple
varieties, four materials and four lighting regimes is a genuine randomisation and not a token one;
a null across that span is informative even if it is not conclusive.

**k = 2 was considered and rejected** for the opposite reason: two domains is not domain
randomisation, and a null there would have cost ~221 GPU-h to learn nothing.

## 4. What V11 does not discharge

- **It does not license generation.** V1 §1 binds. §6's gates are all undischarged, §8 item 3 has
  no throughput measurement and §8 item 4 has no committed `GEOM_TOL` or `EST_DRIFT_P95`.
- **It does not set or relax any ceiling.** `PARTITION_CEILING_GPU_H` still has no default and still
  must come from a `TIMING=1` run.
- **It does not touch `configs/transfer25/styles.toml`,** whose hash is part of the provenance of
  every clip.
- **The ~1.16 s/frame figure is not §8 item 3's measurement** and may not be used as a budget line.
  It is one diagnostic clip's observed rate, quoted to size a decision, and it is likely optimistic:
  the probe ran 96-frame clips where the partition's episodes average ~427 frames.

## 5. Determination

**Decided by: nobody yet. UNSIGNED.**

Whether generation proceeds in stages, and whether stage 1 is these four styles, is the project
owner's call. A session may draft this document; it may not sign it, and it may not treat a draft as
a licence. No clip may be generated under V11 until the determination below is filled in by a
person, and not then unless V1 §1, §6 and §8 independently permit it.

```
determination:  ____________________
decided by:     nobody yet
date:           ____________________
```

## 6. Provenance

- Partition, arms and the §5 prohibition: `docs/preregistration/PR-08-photoreal-augmentation.md`
  §5, §8 items 1–7.
- Arm C frame matching: `T40_RULE_V2`.
- Seed schedule: `T40_RULE_V3` and `[seed_schedule]` in `configs/transfer25/styles.toml`.
- The mask-validity reference defects that stage 1's non-warm apples run into: `T40_RULE_V10`.
- The robot-mask object-grounding filter: `T40_RULE_V9` (also unsigned).
- The 1.16 s/frame observation: job 189926, `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/`,
  Slurm log `t040-halluc-probe.189926.out`, `Average time per chunk: 55.4715211505536` over two
  96-frame chunks.
- Partition size and clip counts: `configs/transfer25/styles.toml`
  (10 `train_styles` + 5 `eval_styles` + identity `repeats = 10` = 25 instances × 402 episodes).
- Allocation and billing: `docs/discoverer.md` §9, `cluster/discoverer/README.md`.
