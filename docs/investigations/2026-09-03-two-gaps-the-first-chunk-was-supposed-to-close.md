# Two gaps the first chunk was supposed to close — one measured, one widened

`PR-08-DET-2026-09-02` §4 said of the released chunk: *„It does not close gap 2. The disk-space
precondition is unchanged and its stated fix is to **measure one clip's on-disk size during this
chunk**, record it, then derive a floor. This run is the occasion for that measurement; taking it is
not the same as having closed the gap."*

**The measurement is taken and is §1. It is not the closure** — §1.3 says what a 10-clip sample
cannot bound. §2 records a second gap that the same run *opened*, on AC-04 traceability, which
nobody had budgeted for.

**Nothing here changes a rule, a ceiling or a gate.**

---

## 1. Gap 2 — clip size on disk, measured

### 1.1 What was measured

Ten clips of the released chunk `s1-train-01of04`, 4 266 frames total, 640×480 H.264, as written by
`robot_composite.encode_clip` after the G0c composite. Three episodes (`000116`, `000120`, `000136`)
across the four stage-1 train styles.

| | |
|---|---|
| total | **89.5 MB** for 10 clips |
| per clip | min **5.94 MB**, median **8.16 MB**, mean **8.95 MB**, max **14.70 MB** |
| **per frame** | min **12 802 B**, **mean 21 144 B**, max 34 667 B |

Per frame is the figure to carry forward, because episode lengths vary (357 / 424 / 464 frames among
these three) and a per-clip mean silently encodes this chunk's length distribution.

### 1.2 The floor this derives

The committed partition is **25 style-instances × 171 625 frames per variant**.

| reading | frames | at 21 144 B/frame |
|---|---|---|
| every episode survives G0c | 4 290 625 | **≈ 91 GB** |
| at the published G0c yield, 17/402 = 4.23 % | ≈ 181 400 | **≈ 3.8 GB** |

**The second row is the operative one and the first is the guard rail.** G0c refuses 385 of 402
episodes before the generator is called, so the corpus that actually reaches disk is the small
number — but the bound a disk-space precondition needs is the one that does not depend on a yield
nobody has re-measured at full scale. **Provision against ≈ 91 GB, expect ≈ 4 GB.**

Chunk 1's own yield was **4 surviving episodes of 101 = 3.96 %**, i.e. slightly below the published
4.23 % and consistent with it.

### 1.3 Why this is not the closure

* **Sample size.** Ten clips, three episodes, four styles. The corpus is 402 episodes and 25
  style-instances.
* **The per-style spread is large and systematic**, not noise:

  | style | mean B/frame | n |
  |---|---|---|
  | `train-02-linen-overcast` | **31 672** | 3 |
  | `train-04-slate-lowkey` | 18 428 | 2 |
  | `train-01-oak-tungsten` | 17 544 | 3 |
  | `train-03-melamine-fluorescent` | **13 468** | 2 |

  A factor of **2.35** between the extremes, and it tracks the prompt: busy cloth texture and diffuse
  light encode large, a flat white melamine surface under even light encodes small. **21 styles of
  the partition have never been generated**, and nothing here bounds where they fall.
* **It is a measurement of the current configuration.** Every lever in
  [`2026-09-03-the-first-clips-look-wrong-and-the-conditioning-is-why.md`](2026-09-03-the-first-clips-look-wrong-and-the-conditioning-is-why.md)
  changes the picture, and a sharper picture encodes larger. A move to bucket 720 in particular would
  invalidate this number along with the throughput.

**So gap 2 has a measured floor and stays open.** What closes it is the same measurement over a
style set that spans the partition, taken under whatever configuration is finally in force.

## 2. AC-04 — the run records a generator revision it never consulted

`CLAUDE.md`: *„Every rollout must be traceable to checkpoint + dataset snapshot + config hash
(AC-04)."* On the generate path that is currently **not** satisfied for the checkpoint half, and the
reason is a correct upstream behaviour meeting a record that has nowhere to put it.

### 2.1 What the durable record says

`97_transfer25_restyle.sbatch:3064` writes into `chunk_metadata.json`:

```python
"generator_model_id": model, "generator_model_revision": rev,
```

where `rev` is `TRANSFER_MODEL_REVISION` — the pinned `ce844032…` that `99_stage_transfer25_weights.sbatch`
staged. **That revision was not consulted by the run.**

### 2.2 What actually loaded

`restyle_transfer25.py:460` records `checkpoint_path_honoured: len(hint_keys) == 1`, and with two
control keys it is **`False`** — which is correct and documented, not a defect: on the multi-control
branch upstream ignores `--checkpoint-path` and resolves every control block from its own registry
in `checkpoints_transfer2.py`, at a revision that is **not** `TRANSFER_MODEL_REVISION`. The driver
therefore reads the checkpoints back off the object rather than restating what we passed in, and
`_raw/<unit>/sample_outputs.json` carries them:

```
edge_720p_…/checkpoints/iter_000032000
vis_720p_…/checkpoints/iter_000036000
depth_720p_…/checkpoints/iter_000044000
seg_720p_…/checkpoints/iter_000043000
```

**Four checkpoints, of which two carry weight 0** — upstream loads every modality on the
multi-control branch on purpose.

### 2.3 The gap, precisely

| where | what it holds | durable? |
|---|---|---|
| `chunk_metadata.json` | `generator_model_revision` = a pin that was **not used** | yes |
| `_raw/<unit>/sample_outputs.json` | `checkpoints_loaded` — the four that **were** used | yes, but per-unit and under `_raw` |
| `<unit>.g0c.json` | the clip's durable sidecar, built by `evidence()` at `:2701` | **has no `checkpoints_loaded` key** |

So the clip that ships next to a corpus carries a sidecar that does not name the weights that made
it, while the file that does name them sits one directory up in a tree named `_raw`.

**What keeps this from being a loss today:** `_raw` is **not** cleaned on the generation path, so the
information exists for every unit of this chunk and is recoverable. **What makes it a gap:** nothing
guarantees that, the `g0c.json` sidecar is the artefact designed to travel with the clip, and a
reader of `chunk_metadata.json` alone would take away a revision that is wrong for the purpose they
are asking about.

### 2.4 Deliberately not fixed here

Adding `checkpoints_loaded` to `evidence()` is a small change to the harvest. It was **not made**,
for the reason `dc4b0be` §4 already established for the sbatch: job 192149 reads the driver and the
harvest at runtime, `chunk_metadata.json` records no driver hash, and a change landing mid-chunk
would be invisible in the record and would split one chunk across two versions of its own
instrument. **The repair belongs in its own change, with its own tests, when nothing is pending on
it** — the same queue as the `THROUGHPUT.json` recipe fix.

## 3. Provenance

| | |
|---|---|
| written | 2026-09-03, while `s1-train-01of04` was still generating |
| measured over | 10 clips / 4 266 frames of the released chunk; `_raw/…/sample_outputs.json` of one successful unit |
| closes | **nothing.** Gap 2 has a floor and stays open; the AC-04 gap is newly recorded and open |
| changes | no rule, no ceiling, no gate, no code |
