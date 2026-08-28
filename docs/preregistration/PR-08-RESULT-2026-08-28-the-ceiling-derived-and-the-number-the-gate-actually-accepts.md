# PR-08 RESULT — the GPU-h ceiling derived, and the number the gate actually accepts is not the one the arithmetic prints

**`T40_RULE_V2` §3 step 2, discharged: *"Derive the ceiling from it over 10 050 clips, i.e. over all
25 style-instances, and record the derivation."* This document is that derivation and that record.**

**It is not an authorisation to spend, and it does not lift `T40_RULE_V1` §1.** A ceiling caps what
a run may *reserve*; whether the run happens is a separate decision and it is the project owner's.
§5 says what that decision now looks like, and it is a smaller question than it was this morning.

**Every number below descends from a measurement `T40_RULE_V20` §5 requires be quoted with its
provenance: *measured on an episode selected for surviving G0c*, on 1 × H200 at 640×480, and the
direction of that selection's bias is NOT KNOWN.**

---

## 1. Who may write this

`T40_RULE_V2` §3 asks for three things before generation: take the measurement, *"derive the ceiling
… and record the derivation"*, and split it across the `STYLE_SET` invocations so the shares sum
within the whole. **A word-boundary search of §3 for `sign`, `signature` and `owner` returns
nothing**, and neither `PR-08-photoreal-augmentation.md:225-231` nor the sbatch's own
required-variable message at `97:434` names a signatory either.

So the derivation is arithmetic over committed artifacts, and a session may perform it and write it
down. What needs a person is authorising the spend — which is §5, and which this document does not
take. An earlier document in this session (`PR-08-RESULT-2026-08-28-the-throughput-is-measured-…`
§5.1) said *"a budget line is a person's signature"* about the derivation itself. **That was too
strong: it is the spend that is a signature, not the arithmetic.**

## 2. The inputs, read and not restated

| | | |
|---|---|---|
| `seconds_per_frame` | **1.6896** | `runs/t040-transfer25-restyle-timing-2026-08-28/THROUGHPUT.json`, job 191102, `units_succeeded: 1` |
| `whole_partition_frames` | **4 290 625** | `…/chunks/s1-train-372of402/partition_facts.json` |
| `corpus_frames` | 171 625 | same |
| `style_instances_per_set` | train 10, eval 5, identity 10 | same |
| whole-partition clips | 10 050 over 25 style-instances | same |

`T40_RULE_V20:122-125` fixes which frame count multiplies the rate: the manifest's `corpus_frames`,
**never the timed episode's 422**. That is honoured below.

## 3. The derivation

```
projected, whole partition   = 1.6896 × 4 290 625 / 3600 = 2013.733333… GPU-h
projected, train      (×10)  = 1.6896 ×   1 716 250 / 3600 =  805.493333… GPU-h
projected, identity   (×10)  =                              =  805.493333… GPU-h
projected, eval        (×5)  = 1.6896 ×     858 125 / 3600 =  402.746667… GPU-h
```

These are exactly the quantities the sbatch recomputes for itself at `97:2229-2230` — the gate does
not read a projection, it derives one, so a derivation recorded here that disagreed with it would be
caught at submit time rather than trusted.

**Two factors in those lines that the arithmetic above holds constant, named so the number is not
quoted outside the conditions it was derived under.** Both projections carry `* nproc`, so a
submission on more than one GPU projects proportionally higher and the ceiling that passes is a
different number — 1 is the file's own `--gres` and is what job 191102 measured on. And
`projected_style_set` multiplies by `staged_instances[style_set]`, not by the full count, so at
stage 1 (4 instances of 10) a set's projection is smaller than §3's; 805.50 covers both the staged
and the full case, which is why it is quoted as the share rather than a stage-1 figure.

### 3.1 And this is why the printed figure is not the number

**`PARTITION_CEILING_GPU_H = 2013.7` is REFUSED**, and would have been refused with a queue slot in
hand. `97:2272` compares `projected_partition > partition_ceiling` on a strict `>`, and
`2013.7333… > 2013.7`. The rounded figure quoted in this session's earlier result document is below
its own projection.

Four checks must all pass (`97:2272`, `:2281`, `:2288`, `:2294`). Recomputed here:

| ceiling | shares train / eval / identity | verdict |
|---|---|---|
| 2013.7 | 805.50 / 402.75 / 805.50 | **fails** — projection 2013.7333 exceeds it, and the shares sum above it |
| 2013.74 | 805.49 / 402.74 / 805.49 | **fails** — every share is below its own projection |
| **2013.75** | **805.50 / 402.75 / 805.50** | **passes all four** |

**The smallest two-decimal set that passes is `PARTITION_CEILING_GPU_H = 2013.75`, with
`CEILING_GPU_H` of 805.50 for `train`, 402.75 for `eval` and 805.50 for `identity`.** The shares sum
to exactly 2013.75 and `2013.75 > 2013.75` is false, so the sum check passes on equality. Rounding
*down* anywhere fails: the shares must round up, the ceiling must round up to hold their sum, and
there is no slack between the two.

2013.75 is also `80.55 × 25` — the artifact's own `gpu_hours_per_variant` times the partition — so
the number has a second derivation that does not go through this document's arithmetic.

## 4. The enforcement §8 item 3 asks for already exists, and exceeds the shape it names

Item 3 wants the ceiling *"enforced in the sbatch as `MAX_RESTARTS` enforces T-39's"*. That
comparison is a hard-coded literal — `70_train_t39_baseline.sbatch:207`, `MAX_RESTARTS=2 # 3 passes
x 4 h = the 12 GPU-h ceiling PR-07 §7 fixes`. What `97` already has is strictly stronger:

1. **Both variables required with no default**, `TIMING` exempt (`97:424-446`).
2. **Share ≤ whole** checked before anything else (`97:440-446`).
3. **The four-way gate** above (`97:2272`–`:2294`), which derives its own projection.
4. **Run-scoped pins**: the first submission under a `RUN_ID` stamps the whole-partition figure into
   `${OUT}/run.env` and a later one that contradicts it is refused rather than silently doubling the
   budget (`97:2149-2160`), with the same for each set's share.
5. **A persisted GPU-h ledger** that reserves each pass's *worst case* `NPROC × WALL_H` before the
   pass runs (`97:2419-2430`) — *"Slurm bills the whole allocation for the whole wall whether or not
   the pass uses it"* — and `MAX_PASSES` derived from the share at `97:2234` rather than typed.

**So nothing is owed here in code.** The owner is authorising a number, not commissioning an
enforcement.

## 5. What the number authorises, against what it will cost

A ceiling is not a bill. Layer 5 draws `NPROC × WALL_H` per pass reserved, so what is actually spent
is 4 GPU-h per 4 h pass on one GPU, and the ceiling only bounds what may be reserved in total.

And the projection is **not** the expected spend, because 385 of 402 episodes never reach the
generator: `restyle_transfer25.py:775-779` runs `preflight_source_masks` first and
`_transfer25_backend` only in the `else`. Measured, from the sixteen area-array tasks, a refused
episode costs a segmentation pass at **0.2396 s/frame** against **1.6896** for a generated one.

| | GPU-h |
|---|---:|
| **ceiling authorised** | **2013.75** |
| projection the gate checks against it | 2013.73 |
| generating the 17 survivors, × 25 instances | 85.2 |
| preflight over the other 385 — range, see the correction below | 10.9 – 273.5 |
| **realistically spent** | **≈ 96 – 359** |

**So the ceiling authorises roughly six to twenty-one times what the run will draw.** The full
argument for the range, and the correction of the 1 928.6 GPU-h figure it replaces, is
`PR-08-RESULT-2026-08-28-the-throughput-is-measured-…` §8.

**The alternative — a yield-aware ceiling in the ~100–360 range — is not proposed here**, and not
because it would be worse. It would require changing the projection formula at `97:2225`, which is
what `T40_RULE_V2` §3 fixes, so it is a new rule version; and an honest yield-aware formula needs
**two** rates, not one rate scaled by a yield fraction. That is a larger change than the tightening
buys against a run already bounded by `MAX_PASSES` and by the ledger.

## 6. What this does not do

* **It does not authorise a generation run.** `T40_RULE_V1` §1 is unchanged; §8 items 4, 5 and 6 are
  unclosed on the reading that gives the task ledger no weight.
* **It does not write a default into the sbatch.** `97:156-158` gives the reason there is none —
  *"No budget line exists until that measurement does"* — and that reason is now discharged by job
  191102. Turning a required-with-no-default variable into a default is a change to what an operator
  can omit, and that is not this document's to make.
* **It does not commit the number to `configs/`.** No tracked artifact holds a ceiling today
  (`git ls-files configs/transfer25` returns nine paths, none of them one), and inventing a home for
  it is a design decision rather than a derivation.
* **It does not claim the allocation can afford it.** The `sreport` figure readable from the login
  node is **CPU** hours; the GPU-h grant is not readable from there.

## 7. Reproducing it

```bash
.venv/bin/python - <<'PY'
spf, corpus = 1.6896, 171625
proj = {k: spf * corpus * n / 3600 for k, n in (("train", 10), ("eval", 5), ("identity", 10))}
print(f"partition {spf * corpus * 25 / 3600:.6f}", {k: round(v, 6) for k, v in proj.items()})
PY
```

The four checks the number must survive are a self-contained heredoc at `97:2191-2298`; it can be
extracted and run against the real `THROUGHPUT.json` and `partition_facts.json` with candidate
numbers, on this workstation, with no cluster and no GPU.
