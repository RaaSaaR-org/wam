# Two gaps on the generate path, recorded and deliberately NOT fixed today

**Findings, not changes.** Both were surfaced by the 2026-08-28 audit of what stands between the
measured throughput and a generation run. Both are real. **Neither is fixed here, and the reason for
not fixing them is the point of this note** — one because the moment is wrong, one because fixing it
would mean inventing a constant this project does not invent.

Neither blocks a first clip. Neither is a §8 item. `T40_RULE_V1` §1 is untouched.

---

## 1. The geometry constants are not bound to the corpus they will gate

### What is true

`robot_composite.load_area_bound` refuses an area bound whose `source_manifest_sha256` is not this
run's, and says exactly why (`scripts/robot_composite.py:1240-1242`):

> `expect_source_manifest` is the SOURCE manifest this run will restyle. Its sha256 must be the one
> the distribution was measured over: **a bound is a statement about a corpus**, and holding corpus
> A's distribution over corpus B is the same drift by another route.

**The geometry block does not do the equivalent.** `97_transfer25_restyle.sbatch:2010-2013` checks
`GEOM_CONSTANTS` against its own committed `.sha256` sidecar — which proves *the file has not
changed*, not *the file is about this corpus*. Grepping that whole block for `manifest` or `sha256`
returns only those sidecar lines.

And it could not do more today, because there is nothing to compare against:

* `grep -n "source_manifest" scripts/measure_geom_tol.py` → **zero hits.** The measurement never
  records the manifest it measured over.
* `configs/transfer25/pr08_geom_tol.json` carries eleven keys and **not one names a corpus**:
  `spec_version`, `what_this_is`, `contract_fields`, `measurement_fields`, `segmenter`,
  `geom_tol_px`, `geom_tol_source`, `est_drift_p95_px`, `est_drift_source`,
  `est_drift_estimator_name`, `gate_margin_px`.

The corpus identity **does** exist one level down and is dropped on the way up: a shard artifact
carries `corpus`, `corpus_episode_keys` and `n_episodes` (`runs/pr08-geom-tol/shards/shard-0.json`).
The merge does not carry them forward.

So `GEOM_TOL` is a tolerance measured over *some* corpus, held over *this* one, with nothing in the
committed artifact able to say whether they are the same. The area bound is protected against
exactly that and the tolerance is not.

### Why it is not fixed today

**Job 191143 is in flight.** Waves 0–3 of `pr08-geom-tol-v2` were submitted 2026-08-28 and are
producing shard artifacts under the current shape right now. Adding a field to that shape mid-run is
how a merge comes to refuse its own shards.

It is also not the one-line change it looks like. The fix has three parts, in order: the measurement
records the manifest sha256; the merge carries it into the committed artifact (which is a
`spec_version` question, since `contract_fields` is part of that document's own contract); and only
then can the sbatch compare. That is a change to a committed artifact's contract, and it belongs
after the waves land, not across them.

**Do it when `pr08-geom-tol-v2` has merged and before any generation submission.**

## 2. Nothing on either path asserts there is disk to write to

### What is true

`grep -n "df -\|quota\|No space\|ENOSPC"` over `97_transfer25_restyle.sbatch` returns **ten hits,
every one of them prose.** There is no free-space, quota or inode assertion on the timing path or on
the generate path.

The generate path writes three growing trees under `${PROJ}/runs/${RUN_ID}` (`97:603`): per-chunk
raw output at `${CHUNK_DIR}/_raw` (`:614`), filed clips at `${OUT}/clips/${STYLE_SET}` (`:613`), and
a mask cache — which grows for **the 385 episodes that never produce a clip**, because the source
masks are computed for every unit the preflight screens and cached whether or not the unit passes.

A run that fills the filesystem mid-chunk does not fail cleanly. It fails inside the driver, per
unit, and the harvest then reads a truncated or absent `vision.mp4` as a missing unit and requeues —
which is a paid retry of a failure that will recur.

### Why it is not fixed today

**Because a threshold would have to be invented, and this project does not invent constants.** A
useful check needs a per-clip size, and there is no measured one: `T40_RULE_V20` §5 requires the
timing run to delete its single clip, and it did — `runs/…/clips/` holds zero files. Estimating from
the source corpus would be estimating across a re-encode at settings nobody has measured.

An absolute floor picked out of the air ("refuse below 50 GB") is the same class of defect as a
detection threshold typed into a submit script: a per-run decision about what the generator may do,
recorded nowhere anybody would look. `106_measure_robot_mask_area.sbatch:345-352` refuses that
pattern by name.

**The honest sequence is: measure one clip's on-disk size during the first real chunk, record it,
then derive a floor from it.** Until then the gap is documented rather than papered over.

## 3. What this note is not

* **Not a claim that either gap has ever fired.** No generation run has happened.
* **Not a §8 item, and not a blocker.** Both sit outside the seven conjuncts.
* **Not a licence to skip them later.** §1 has a stated moment (after the merge, before any
  generation submission) and §2 has a stated precondition (one measured clip size).
* **Not the whole audit.** The other findings landed as commits: the harvest's refused/missing
  split, the ceiling derivation, and the D-D draft.
