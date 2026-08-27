# The six fronts, 2026-08-27 — working notes, not findings

Six investigations run in parallel on 2026-08-27, one per open front of the PR-08 §8 gate, each
followed by an adversarial reader whose only instruction was to refute it. Every file here carries
its re-read appended under **`## Adversarial re-read`**.

**These files are evidence, not conclusions, and several of their headline claims did not survive.**
Read them for the measurements, the paths and the line numbers; read
[`../../PR-08-DECISION-SHEET-2026-08-27.md`](../../PR-08-DECISION-SHEET-2026-08-27.md) for what is
actually true. **Where a front and the sheet disagree, the sheet is right** — it was written after
the re-reads and reports the knock-down rather than the finding. They are kept because the method
here is that a claim is worth nothing without the artifact it was measured from, and discarding the
working notes would leave the sheet's numbers unre-derivable.

Claims from these files that the re-read **refuted**, listed so nobody carries them forward:

| file | claim | why it fell |
|---|---|---|
| `F3-item4-est-drift.md` | `EST_DRIFT_P95` is already measured to the registered standard and needs no further decision | `T40_RULE_V17` §4:200-202 leaves pooled-vs-single explicitly open, and neither candidate has a carry path |
| `F3-item4-est-drift.md` | the project already resolved which arm G0b subtracts, in favour of propagation | refuted on four citations; both supporting quotes were truncated at the clause that undercuts them, and the committed contract reads `propagation: "per_frame"` |
| `F5-yield-empty-mask.md` | the yield on record is optimistic by 2× | no committed document ever claimed 36 was the joint yield; the defensible finding is that nobody multiplied the two halves out (it is 17/402) |
| `F5-yield-empty-mask.md` | the draft `T40_RULE_V20` (trim + identity fallback) | disarms `check_mask`'s area half through the work unit, breaks the harvest contract's "every frame or none", and mis-indexes the identity spans |
| `F5-yield-empty-mask.md` | the V15 tile counts | 2 yes / 75 no / 24 `cannot_tell`, not 2/77/22 |
| `F4-residue-i-473-vs-478.md` | the five frames are named | two rival 4-subsets tie with the accepted hypothesis at 0.165650 px; the artifact names three of them and one each from two candidate pairs |
| `F4-residue-i-473-vs-478.md` | the V18 outcome is *measured* to be invariant under the cluster's 36 | an inference: `shard-7.json` records no per-frame mask area, so it applied this workstation's areas to a cluster-derived index set |
| `F6-two-repairs.md` | the MuJoCo captures show zero refusals of any kind | read from `counters_at_start_of_run`, which is zero by construction; the real counts are 25/720, 12/240, 1/480 |
| `F6-two-repairs.md` | item 4 is held open by a residue/signature question | wrong in the direction that costs GPU-hours: the committed contract disagrees with every landed shard by one field, so the corpus pass must be run again regardless |
| `F2-item4-geom-tol.md` | six of sixteen shards ran a stale adapter | twelve did, and **zero** of the sixteen carry `mask_val_ref_max_frac` — the error ran against its own interest |
| `F1-item3-throughput.md` | the 1.16 s/frame figure is recorded in job 189926 | it appears in no line of that log; it is a division of four chunk timings that exclude the depth pass, the SAM2 pass and both encodes |

Two proposed patches in these files are **known broken** and must not be applied as written: Front 6's
`apple_sam2_video.py` refusal-mirror fix raises `UnboundLocalError` on the first frame it is meant to
count, and Front 3's `--carry-est-drift` fix does not compile and is backwards.
