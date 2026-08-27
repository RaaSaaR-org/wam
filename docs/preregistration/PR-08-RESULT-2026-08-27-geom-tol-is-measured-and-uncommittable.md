# PR-08 §8 item 4 — GEOM_TOL is measured, every margin is positive, and the artifact cannot be committed

**Type.** RESULT / determination of state. Not a pre-registration, not a rule, and **not a decision**.
It records what was found and names the fork. The fork is the project owner's to take.

**Date.** 2026-08-27. **HEAD at the time of the investigation:** `ea315e5`, working tree clean.

**Produced by.** A five-front local investigation, each front adversarially re-read by an
independent agent, then synthesised. No GPU-hours were spent on the cluster; nothing was submitted.
The corrections the re-readers forced are carried in §7 so they are not lost.

---

## 1. The one-sentence finding

**GEOM_TOL has already been measured over the whole corpus, all four candidate margins are
positive, and the artifact holding that number cannot be committed for three independent reasons —
one of which records *which frames were measured*. Repairing that record now, after its output is
visible, is exactly what this project's versioning rule forbids.**

So the path forward is a fork, not a task:

| | | cost |
|---|---|---|
| **1a — salvage** | write a new pre-registration that fixes, blind, whether the landed number is admissible; then run the blind census it registers | **UNKNOWN**, and must be bounded *in* that document |
| **1b — re-run** | `sync.sh`, then submit the 16-shard array with a **fresh `RUN_ID`** | **13.64 GPU-h** |

If the owner declines to write 1a's document, or writes it and its blind criterion comes back
above threshold, **1b is the answer.** That is the honest default and this document does not
argue against it.

---

## 2. The number exists

`runs/pr08-geom-tol/pr08_geom_tol.json`:

```
GEOM_TOL_px   0.47857992441961017
n_episodes    402
n_frames      171625
gate_qualified  false
```

Two agents independently re-pooled the sixteen shards' raw `per_episode[*].displacements_px`. One
reproduced the headline to the last digit by hand; the other ran the real
`measure_geom_tol.py --merge` and reproduced **every percentile, `std_px`, `min`, `max` and all 129
histogram counts bit-identically** — only `mean_px` differs by one ulp, a numpy summation order
difference. **The raw per-frame values are not gone.** Whatever else is true, "the measurement was
lost" is not.

### The four margins

The headline arm is `per_frame` (decided 2026-08-27, `ea315e5`, on PR-08 §4 step 2's
same-segmenter ground).

| EST_DRIFT route | p95 px | margin px | carryable today? |
|---|---|---|---|
| `per_frame`, single capture `capture-mujoco-trajectory-f480` | 0.29077062684224225 | **0.18780929757736792** | **yes** |
| `propagation`, same capture | 0.47006167975525187 | 0.008518244664358299 | yes, 1.78 % of budget |
| `per_frame`, pooled | 0.3120786214328541 | 0.16650130298675608 | **no** — see §5 |
| `propagation`, pooled | 0.4486097454155794 | 0.02997017900403076 | **no** |

**All four are positive.** Every one of them is computed against a GEOM_TOL that cannot be
committed, and **whether a re-measured GEOM_TOL reproduces `0.4786` is UNKNOWN.**

---

## 3. Why the artifact cannot be committed — three reasons, not two

Measured, not inferred. `contract_disagreements(committed_segmenter, shard-0_segmenter)` returns
exactly one disagreement; the same function against HEAD's live `SEGMENTER_CONTRACT` returns `[]`.

**(a) The gate flag was `False` when the shards were measured.** All sixteen carry
`mask_method.gate_qualified: false`. The flag flipped to `True` today (`13f0416`,
`scripts/estimators/apple_sam2.py:967`) on the owner's recorded decision.

**(b) The shards' segmenter block lacks `mask_validity_reference_max_frame_fraction`, and that
field is not metadata.** It is a behavioural frame-refusal predicate:
`apple_sam2.py:1650` `reference_is_object_scale`, `:1674` the comparison, `:2085` the refusal, with
its own counter `n_frames_mask_refused_reference_not_object_scale`. Commit `e518a84` says so in its
own message: *"It refuses frames, so like `mask_validity_min_iou` and `mask_validity_reference`
beside it, it is a statement about WHICH FRAMES were measured."*

The timing is the whole problem. The **enforcement** landed in `6a32143` (2026-08-23 16:07:20 UTC).
The **declaration** landed in `e518a84`, **24.77 h later** and 37 minutes after the array merged.
Every shard's `estimator_version` string ends `...;prop=per_frame;mask_val_min_iou=0.1` with **no**
`mask_val_ref_max_frac` token, so the adapter that produced them provably predates the refusal
branch.

**(c) They were therefore produced by a superseded instrument.** This one has no repair at all: it
is a statement about provenance, not about a field.

### What is UNKNOWN, and it is the crux

**Nobody knows whether the V10 rule would have refused zero of those 171,625 frames or thousands.**
No artifact in `runs/` or `configs/` records a colour-reference frame fraction over this corpus. A
grep for the V10 counter across the tree returns 21 hits — all MuJoCo EST_DRIFT captures or the
operating-point census, **none over the apple corpus**.

The only datapoint that exists is one episode: a re-derivation over the 509 frames of
`episode_000094` gives a maximum reference frame fraction of **0.024079**, against a bound of
**0.10** — a factor of four clear. **One episode of 402 bounds nothing**, and it is recorded here
as a reason the question is worth asking, not as an answer to it.

---

## 4. Why the repair is not a session's to make

Re-deriving the number is free. *Admitting* it is not.

Admitting it means writing (a) and (b) into a completed measurement record **after** the number it
yields — `0.4786` — and all four positive margins are visible to everyone reading this page. The
standing rule (`docs/handoff.md` §3) is:

> Rules are versioned, never edited in place. **A gate rewritten after seeing its output is not a
> gate.**

So salvage is not forbidden — it is **conditioned**. It needs a pre-registration written *before*
the deciding measurement runs, and that document has to fix five things blind:

1. **The decision rule, as a threshold on an unmeasured quantity.** For example: *"If the maximum
   `reference_frame_fraction` over all 171,625 frames of the 402-episode corpus is below
   `mask_validity_reference_max_frame_fraction = 0.10`, the V10 refusal branch is provably inert on
   this corpus, the landed shards' frame population is identical to what HEAD would measure, and
   `0.47857992441961017` is admissible. Otherwise the array re-runs."* **Both outcomes written
   before either is seen.**
2. **The instrument for that blind census, pinned to a commit and costed.** The quantity is a
   function of decoded pixels alone, so it plausibly needs decode without SAM2 or GroundingDINO —
   but **the cost of a decode-only replay over 402 episodes is UNKNOWN and must be bounded in the
   document, not assumed to be below 13.64 GPU-h.** A salvage that costs more than the re-run is
   not a salvage.
3. **A rule for (a)** — whether a landed shard may inherit the adapter's *later* `GATE_QUALIFIED`
   value. Registered before it is applied. The sbatch's own text already states the ground:
   `false` there recorded *"the adapter's standing flag and not this shard"*.
4. **A schema change for (b), not a value edit.** A *declared-but-inert* annotation with its own
   field, so the record never claims a filter ran that did not run. This touches
   `contract_disagreements()` semantics or the shard schema — gate code, so a version bump and the
   owner's signature.
5. **What (c) means** — whether "measured by a superseded instrument, shown inert by (1)" is
   acceptable provenance for a committed gate number, stated as a rule and not decided about this
   number.

---

## 5. What still blocks the chain after GEOM_TOL lands

The post-GEOM_TOL chain was rehearsed end to end against a synthetic qualified gate document, on
the local RTX 5090, twice by independent agents with bit-identical results. Three things survive.

**B1 — the pooled EST_DRIFT number cannot be written by any command line.**
`--carry-est-drift runs/pr08-est-drift/v17/POOLED-V19.json` exits 2, four refusals deep (no
`gate_qualified`, no top-level `est_drift_p95_px`, no `estimators.name`, no `resolution_hw`).
`measure_geom_tol.py:3454-3462` says the absence is deliberate: *"No carry path is built for it,
deliberately … Writing the plumbing would answer it by making one of the two the reachable one."*
**So D-D is not a preference, it is a precondition** — if the answer is POOLED, the carry path has
to be built first, and that is a rule change with its own version bump.

**B2 — `--est-drift-arm per_frame` is mandatory and is not in any runbook.** A two-arm artifact
carried without it exits 2 writing nothing (`measure_geom_tol.py:3816`). The refusal is
*unreachable today* — the `gate_qualified` check at `:3681` fires first — so it goes live at
exactly the moment it can cost a job. **It belongs in the runbook line, not in a comment.**

**B3 — ordering.** The carry's `--out` defaults to the tracked
`configs/transfer25/pr08_geom_tol.json`. **The carry must run strictly after the merge.**
Corollary worth stating on its own: **the carry exiting 0 is not evidence that GEOM_TOL is
gate-qualified.**

---

## 6. Two traps closed on the way

Both were found by the investigation and repaired the same day; both are fail-closed, i.e. they can
only cause *more* work to be done, never less.

**T1 — a default submission would have done nothing at all.** `RUN_ID` defaults to `pr08-geom-tol`
(`103_measure_geom_tol.sbatch:407`), so `SHARD_OUT` resolves onto the stale shards. The resume
check at `:886` runs **before** the contract-and-gate preflight at `:979`, and
`shard_artifact_landed` (`:855-876`) classified a not-gate-qualified shard as **reusable** — with
the justification *"that is the adapter's standing flag and not this shard"*. That justification
was correct while the flag was `False`: re-measuring would have produced another unqualified shard.
**With the flag now `True` it is inverted** — re-measuring today *would* produce a qualified shard.
Run against the real `shard-1.json`, the unrepaired function reported `LANDED=TRUE -> task would
exit 0 and skip`. Every array task would have printed *"already landed. Skipping."*, the preflight
would never have executed, and the merge would have pooled a permanently-disqualified median.

**T2 — the carry's write ordering.** The refusal on a *disqualified* artifact is clean and was
verified by hand (`exit 2`, *"Nothing was written."*, target md5 unchanged). The narrower path — a
*qualified* artifact carried onto a document whose `geom_tol_px` is still null — was reported to
write three fields and a `.sha256` sidecar before exiting 3.

**Precondition for any submission, and it is not a footnote: a FRESH `RUN_ID`.** Never `FORCE=1`
into `pr08-geom-tol`; that is correctly refused pre-GPU on the stale-shard trap. Never set
`GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT`.

---

## 7. Corrections the re-readers forced

Recorded so they are not carried forward by anyone reading the fronts rather than this page.

- The enforcement-to-declaration gap is **24.77 h**, not 25.8 h. Direction unaffected.
- The V6 IoU check refused **36 of 171,625** frames corpus-wide, not zero — the "zero" was shard 0
  only. All 36 localise to **shard 7** via `estimator_stats.per_shard`, which is a 16-element list
  on the merged artifact, not `null`.
- The pooled refusal is raised at `measure_geom_tol.py:3679`, not `:1900`.
- **POOLED is not uniformly the conservative choice.** Across the eight A-captures the per-frame
  p95 ranges 0.22601792 (A4) to 0.36010037 (A6). Pooled per-frame 0.31208 is *stricter* than A1 and
  *looser* than A6; pooled propagation 0.44861 is *looser* than A1's 0.47006. **Whoever signs D-D
  is choosing a number, not choosing safety.**
- **"SINGLE" is under-specified.** At least four distinct `headline_valid` single-capture per-frame
  numbers sit on disk. Answering D-D = SINGLE requires naming *which capture*.
- One claim was **refuted outright**: that the 13.64 GPU-h "would re-produce numbers that are
  already stored". The arithmetic is right; the conclusion answers a question that was never the
  blocker.

---

## 8. On whether the flag flip rests on a real signature

One re-reader raised an objection that deserves to be written down rather than dismissed: **git
alone cannot distinguish an owner decision from a session-written document asserting one.** Every
commit in this repository is authored `emai-zema-bot[bot]` with `Co-Authored-By: Claude`, so
`573c80d`'s claim that the owner decided outcome C is, *from the repository's evidence alone*,
indistinguishable from a session writing that sentence. `13f0416`'s own commit body concedes the
point: *"A session willing to write a determination document could pass these guards."*

**On the facts of this case the objection does not land.** The decision was put to the project owner
as an explicit question on 2026-08-27 and answered *"Ja — C annehmen, Flag umlegen"*. `573c80d` and
`13f0416` record a real answer.

**As a general weakness it stands, and no session should paper over it.** The repository has no
mechanism by which a future reader can tell the two cases apart. That is a gap in the method, and
naming it here is the only thing this document can do about it.

---

## 9. What would refute this determination

- A `reference_frame_fraction` census over the corpus showing frames **above** 0.10 — then the
  landed shards measured a different frame population and §1's fork collapses to 1b.
- A decode-only census costing **more** than 13.64 GPU-h — then salvage is not a salvage.
- A re-measured GEOM_TOL that does **not** reproduce `0.47857992441961017` — then §2's margins were
  about a number that no longer exists, and every one of them must be recomputed.
- Evidence that `0.4786` is itself wrong. Nothing here checks the measurement; it checks whether the
  record of it may be committed.
