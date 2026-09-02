# Gap 1 closed: the committed GEOM_TOL now names the corpus it is a tolerance for

**A change, not a finding.** `2026-08-28-two-generate-path-gaps-recorded-not-fixed.md` §1 recorded
that the geometry constants were not bound to the corpus they would gate, and named the moment to
fix it: *"Do it when `pr08-geom-tol-v2` has merged and before any generation submission."* The merge
landed 2026-09-02 (`f4dd57e`) and no generation has been submitted. This is that fix.

**Gap 2 — the disk-space precondition — is NOT closed here** and its stated precondition is
unchanged: one measured clip size. See §4.

---

## 1. What was wrong

`robot_composite.load_area_bound` refuses a mask-area bound whose `source_manifest_sha256` is not
this run's, in its own words: *"a bound is a statement about a corpus, and holding corpus A's
distribution over corpus B is the same drift by another route."*

The geometry block beside it checked only that `configs/transfer25/pr08_geom_tol.json` matched its
own committed `.sha256` sidecar. **That proves the file has not been edited since it was committed.
It does not prove the file is about the tree being restyled.** Grepping the whole block for
`manifest` returned nothing.

And the document could not have answered: it carried `corpus` as a bare filesystem path and no
digest of anything. A path is not an identity — a tree can be replaced under it.

## 2. What was done

Three parts, in the order the 2026-08-28 note required.

**`scripts/measure_geom_tol.py` gains `--bind-source-manifest`** (spec 1.2.0, two new measurement
slots: `source_manifest_sha256` and `source_manifest_binding`). It measures nothing. It records the
manifest's sha256 into the committed document — **after** requiring that the manifest and the
measurement describe the same corpus:

| checked | refuses on |
|---|---|
| episode ids | any id in one and not the other, naming them |
| per-episode frame counts | any disagreement, naming them |
| pixel grid | `resolution` ≠ the measured `frame_width`/`frame_height` |
| fps | a mismatch beyond 1e-9 |

It also refuses a document with no measured `geom_tol_px` and one whose `gate_qualified` is false
**or absent** — binding a corpus to a disqualified tolerance produces a file that reads as a
finished, corpus-bound gate and is neither. Every refusal writes nothing and no sidecar.

**`cluster/discoverer/97_transfer25_restyle.sbatch` now compares.** The geometry preflight takes
`${SOURCE}/manifest.json` as a third argument, recomputes the digest, and refuses on a mismatch —
and refuses an artifact that names no corpus at all, on the same rule the block already applies to
`gate_qualified`: an unstated claim is not a claim.

**`tests/test_measure_est_drift.py`** — one assertion updated, from the other side. It asserted
`geom_tol_is_not_gate_qualified`, which was true only while the corpus was unmeasured. That reason
is correctly absent after the merge, so the test now asserts that the committed document **is**
qualified and measured, rather than dropping the line: a disappearing reason must not be able to
make a test easier to pass.

## 3. Why the digest and not the episode ids

The ids match, and that is not the check. The sbatch already states why on the area-bound path:
*"Episode ids are reused across trees (the AV1 and the H.264 corpora share them), so this cannot be
waved through on the ids matching."* The manifest's bytes are the identity; the id and frame-count
comparison is what makes recording that digest a checked fact instead of an assertion moved into a
hex string.

### The binding is POST HOC, and the artifact says so

`bound_after_measurement: true` is written into the document and is not a disclaimer to read past.
The measurement ran over a clip-dir tree and never opened `manifest.json`, so this binding was made
afterwards, on 2026-09-02, over a merge from 2026-08-29.

**What licenses it, and it is checkable rather than argued:**

* the manifest's mtime on the cluster is **2026-08-20 11:55:17 UTC**, unchanged since;
* the earliest `pr08-geom-tol-v2` shard is **2026-08-29 00:44:31 UTC**, nine days later;
* so the manifest the tree carried when the measurement ran is the manifest on disk now;
* the local copy at `runs/pr08-source-manifest/pr08-apple-640x480.manifest.json` hashes to
  `28e4791a…`, byte-identical to the cluster's;
* and the four equality checks above passed over all **402** episodes — same ids, same per-episode
  frame counts, 640×480 at 30 fps, summing to the 171625 frames the merge measured.

**The residual weakness, stated rather than omitted.** A re-measurement would have produced this
binding as a *measurement*, not as a later assertion over one. It was not done because the corpus
pass costs four waves and days of backfill queueing, and because the checks above are what the
re-measurement would have proven. A reader who does not accept the mtime evidence should read this
field as bound-by-inspection and re-measure.

## 4. What this does not do

* **It closes no §8 item.** §8 item 4 is the margin, and this is not it. Neither gap was ever a §8
  item — the 2026-08-28 note says so in its own §3.
* **It licenses no clip and lifts no gate.** `T40_RULE_V1` §1 is untouched.
* **It does not close gap 2.** Nothing on either path asserts there is disk to write to, and the
  precondition for fixing that is unchanged: measure one clip's on-disk size during the first real
  chunk, record it, then derive a floor. An absolute floor picked out of the air stays refused.
* **It changes no measured value.** `geom_tol_px`, `est_drift_p95_px` and `gate_margin_px` are
  byte-identical before and after; the segmenter contract is byte-identical to the one committed
  2026-08-22. `spec_version` moved 1.1.0 → 1.2.0 because the document gained two slots, and that
  bump is the only edit to a contract-section field.
