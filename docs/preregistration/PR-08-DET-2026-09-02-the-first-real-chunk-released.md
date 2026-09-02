# PR-08 DET — the first real chunk released: generation beyond the single clip, bounded

**SIGNED 2026-09-02 by the project owner.** This is the "separate go" that
`PR-08-DET-2026-09-01-the-spend-authorised.md` §1(c) reserved: *"The remainder of the ceiling is
authorised as a CAP, not as a release. Generating beyond the first clip is a separate go, and it is
the owner's."*

**It releases one chunk against the already-signed ceiling. It signs no new number, it lifts no
gate, and it authorises nothing about training.**

---

## 1. The determination

```
determination:  (a) RELEASED: generation beyond the single clip of DET-2026-09-01 §1(b),
                    BOUNDED AT ONE CHUNK of the committed partition --

                        STAGE=1  STYLE_SET=train  CHUNK_INDEX=1  CHUNK_TOTAL=4

                    which is one quarter of stage 1's train set: ~100 of the 402
                    episodes by the sbatch's own id-sorted stride, and whatever
                    survives G0c among them across the four stage-1 train styles.

                (b) NOT RELEASED: the other three chunks, stage 2, the eval set, and
                    the identity set. DET-2026-09-01 §1(c) continues to govern them.
                    The ceiling stays a cap over the whole partition.

                (c) The ceiling is NOT re-signed and NOT changed.
                    PARTITION_CEILING_GPU_H = 2013.75 and the train share 805.50 are
                    the figures signed 2026-09-01 and they are passed unchanged.

decided by:     the project owner, 2026-09-02, by the instruction

                    "nehem das, was am sinnvollsten ist. erzeuge den datensatz
                     (nicht komplett)"

                given in direct answer to the session's question, put verbatim as

                    "Also: 4 Clips (kleinster ehrlicher Chunk, wahrscheinlich leer)
                     oder ~16 (Rezept-Chunk, liefert sicher etwas)? Ich submitte
                     nichts, bis du das sagst."

                and confirmed the same day, after the shape below had been named in
                writing, by

                    "ja, du kannst das machen. melde dich, wenn der datensatz fertig
                     ist"

                THE SHAPE IS THE SESSION'S, THE RELEASE IS THE OWNER'S. The owner
                delegated the form ("was am sinnvollsten ist") and set the bound
                ("nicht komplett"). §2 is the ground for the form; it is a session's
                reasoning and is recorded as such. T40_RULE_V13 §5 permits a session
                to prepare the rationale and name the edges; it may not sign this, and
                the signature above is the owner's instruction recorded verbatim.

date:           2026-09-02
```

## 2. Why this chunk, and why the shape is not a selection

**`CHUNK_INDEX=1 CHUNK_TOTAL=4` is the sbatch's own generation recipe**, `97_transfer25_restyle.sbatch:31-34`,
written there before any of the 17 survivors were known. It is not a number this session coined for
this run, and `CHUNK_INDEX=1` is the first slice rather than a chosen one.

**The chunking is by id, not by yield.** `eps = sorted(man["episodes"], key=lambda e: str(e["id"]))`
then `mine = eps[idx-1::total]` — a deterministic stride over the sorted episode ids. Which of the
17 G0c survivors land in chunk 1 is a fact about the sort order, not about anything anybody picked.

**AND IT WAS NOT COMPUTED FIRST.** This session did not evaluate where the survivors fall before
naming the chunk, and deliberately so: choosing the index after seeing the yield map is the same
class of act `T40_RULE_V20` §5 forbids when it says *"outcome R does not license walking down the 17
until one passes."* The cost of that discipline is real and is stated rather than hidden — **chunk 1
may contain few survivors, or none.** An empty chunk is an honest outcome of an unselected slice and
will be reported as one.

**Neither was `episode_000371` targeted.** Its selection is licensed for the timing measurement and
`T40_RULE_V14:35` says *"the substitution is licensed for that measurement and for no other."*

**Why `STAGE=1` and `STYLE_SET=train`.** `STAGE=1` is the stage V11 defines as first and whose
outcome V11 §3 makes the precondition for stage 2. `STYLE_SET=identity` is vetoed by
`T40-TODO-01-identity-prompt-provenance`. `STAGE=1` with `STYLE_SET=eval` is refused by the sbatch
by design, because deferring the eval set is the point. `train` is what is left, and it is what
stage 1 is.

**Why one chunk satisfies "nicht komplett" in both directions.** At the published G0c yield of
17/402 = 4.23 %, ~100 episodes carry ~4 survivors; across stage 1's four train styles that is ~17
clips and ~3.4 GPU-h, far inside the 805.50 train share signed 2026-09-01. It is large enough to be
a dataset rather than a demonstration, and it is one quarter of one stage of a partition of 25
style-instances — which is the definition of not complete.

## 3. What is passed, and where each figure comes from

| variable | value | its authority |
|---|---|---|
| `PARTITION_CEILING_GPU_H` | `2013.75` | signed `PR-08-DET-2026-09-01` §1(a) |
| `CEILING_GPU_H` | `805.50` | the train share of that same signature |
| `GEOM_STEP_FRAMES` | `1` | registered by `T40_RULE_V3` §4.3; matches the artifact's own `step_frames` |
| `CONTROL` | `depth:0.5,seg:0.5` | *"PR-08's committed set, not a choice made at submit time"* — and the set the throughput was measured under |
| `STAGE` / `STYLE_SET` | `1` / `train` | `T40_RULE_V11`; see §2 |
| `CHUNK_INDEX` / `CHUNK_TOTAL` | `1` / `4` | the sbatch header recipe, `97:34` |
| `PR08_T39_REPORTED` | the `N` of 2026-08-17 | `T-39` re-reported under `T39_RULE_V2`, job 188408, `docs/preregistration/PR-07-V2-RESULT.md` |
| `PR08_T39_ARTIFACT` | `runs/t39-baseline-seed0/eval-t39-oracle-action/bench.json` | the committed T-39 result; cluster copy byte-identical to local |
| `RUN_ID` | `t040-transfer25-restyle-2026-09-02` | explicit and dated, as `97:11-18` requires |
| `--time` | an honest short walltime, overriding the 4 h `#SBATCH` | `97:20-28` — *"Ask for what the run needs"* |

**On the T-39 verdict, because it is the item most likely to be misread.** The root `CLAUDE.md`
still describes T-39 as having reported `VOID (labels)` and never mentions the 2026-08-17 re-report.
That file is stale on this point; the task record and `PR-07-V2-RESULT.md` carry the `N`, and
`T40_RULE_V3` §5.3 registers that **N satisfies PR-08 §1 while VOID does not**, which is what closes
§8 item 7. **No session may edit `CLAUDE.md` to lift a gate and none was edited here** — the
staleness is flagged to the owner as a correction request and nothing more.

## 4. What this does not do

* **It does not lift `T40_RULE_V1` §1.** §8 is conjunctive and stands at 7/7 on its own evidence;
  this release is what §8 being closed makes exercisable, not a substitute for it.
* **It does not authorise training.** `CLAUDE.md` reserves that to the owner and this page does not
  touch it. `PR-07` §6's prohibition on statements about GR00T is unaffected.
* **It does not release the rest of the partition.** §1(b).
* **It does not close gap 2.** The disk-space precondition is unchanged and its stated fix is to
  **measure one clip's on-disk size during this chunk**, record it, then derive a floor. This run is
  the occasion for that measurement; taking it is not the same as having closed the gap.
* **It changes no rule and no committed constant.** `T40_RULE_V2` §3's formula, `97:2225`, the
  required-with-no-default ceiling variables and `configs/transfer25/pr08_geom_tol.json` are all
  untouched.

## 5. The cluster sync, and why it was safe to take

The cluster copy stood at `213815d`, four commits behind. A re-sync rewrites
`${PROJ}/wam/GIT_COMMIT`, and two jobs belonging to a peer session — `191922` and `191923`,
`pipe_g02_lr5e5_eingefroren` — were PENDING at the time. That is an AC-04 traceability question
about somebody else's run and it was checked rather than assumed:

* both jobs run `${PROJ}/pipeline/g02_lr5e5_eingefroren/train.sbatch`, with
  `WorkDir=${PROJ}/pipeline/g02_lr5e5_eingefroren`;
* a `grep` of that script for `wam` and for `GIT_COMMIT` returns **nothing** — it neither reads the
  synced tree nor stamps its provenance from it;
* and the four commits touch only `97_transfer25_restyle.sbatch`, `configs/transfer25/pr08_geom_tol.json`,
  `scripts/measure_geom_tol.py`, tests and docs. **Zero files under `src/wam/`.**

So the sync cannot reach those jobs by either route. Recorded because the check is the thing that
made it safe, and a later reader should be able to see that it was done rather than skipped.

## 6. Provenance

| | |
|---|---|
| determination | `PR-08-DET-2026-09-02` — the first real chunk |
| status | **SIGNED 2026-09-02.** In force |
| decided by | the project owner, 2026-09-02, on the instructions quoted verbatim in §1; shape prepared by a Claude Code session, which `T40_RULE_V13` §5 permits |
| exercises | `PR-08-DET-2026-09-01` §1(c) — the separate go it reserved |
| releases | **one chunk**: `STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=4` |
| ceiling | unchanged and not re-signed — `2013.75` whole, `805.50` train share |
| amends | nothing |
| generation licensed | **the released chunk, and nothing beyond it** |
| training licensed | **no** |
