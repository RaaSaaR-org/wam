# PR-08 — four determinations, 2026-08-24

**This document registers no rule.** It is a determination record: it says what was decided, by
whom, on what evidence, and — for two of the four items — what was **declined** and why. `T40_RULE_V12`
is a *different* document
([`PR-08-V12-g0c-empty-mask-semantics.md`](PR-08-V12-g0c-empty-mask-semantics.md)) and is **not**
covered by anything decided here.

## 0. The delegation, stated exactly

On 2026-08-24 the project owner instructed a Claude Code session: *"make the four decisions with
your recommendations - then continue with the tasks."* The four were the ones the session had put to
them in the preceding turn: the `GATE_QUALIFIED` discharge, `max_frame_fraction` for G0c, signing
V9/V10/V11, and whether `T40_RULE_V1` §1 licenses generation at all.

Three things about that delegation are load-bearing and are recorded rather than glossed:

1. **The owner delegated without having seen the recommendation.** They authorised the outcome of a
   judgement they had not yet read. Every determination below is therefore reversible on their
   reading of it, and each was written so that reverting it touches only an added banner and a
   determination block — never a sentence of the rule it adopts.
2. **A delegation is not a licence.** `T40_RULE_V1` §1 is not addressed to a signatory; it is
   conditional on facts (§5 below). No determination in this document moves it.
3. **The session drew a line inside the delegation and did not decide everything it was handed.**
   The line is stated in §7 and the two declined items are §4 and §5.

## 1. `T40_RULE_V9` — ADOPTED

*The robot mask may not be the apple.* Per-candidate object-grounding filter at
`ROBOT_MASK_OBJECT_MAX_IOU = 0.70`, in the G0c compositor only.

**Why this was inside the delegation.** V9 is a correctness fix to an instrument, not a loosening of
a gate. The defect it fixes does not evade the gates — in V9's own words it **"manufactures a
pass."** A robot mask that has latched onto the apple composites the *source* apple back over the
generated frame; the generator's apple is overwritten by the real one, and G0b's geometry check then
compares the source apple against itself and finds, necessarily, that geometry was preserved. A gate
that passes because its input was silently replaced by the answer is worse than a gate that fails.

**The evidence predates the rule.** 2 845 per-detection IoUs over a 710-frame plan, contact sheets
under `runs/pr08-robot-mask-apple/`, and job **189926**'s sheets. The threshold 0.70 sits inside a
*measured* gap (0.5131, 0.9364) and its insensitivity is swept in
`tests/test_robot_composite_object_filter.py`. Nothing here was chosen after seeing a gate output.

**It closes a gap rather than opening one.** The code has *already landed* —
`scripts/robot_composite.py:372` defines `ROBOT_MASK_OBJECT_MAX_IOU = 0.70` and the per-candidate
filter runs at `:881`. Before this determination the repository was **running an unregistered
filter**. Adoption makes the code and the rule agree; leaving V9 unsigned would not have stopped the
filter, it would only have left it undocumented.

**Ordering — this is why V9 had to be decided now and not later.** `object_grounding_filter` is a
member of `robot_composite.SEGMENTER_IDENTITY_FIELDS`. An area-fraction distribution measured under
a *different* masker is refused by name by `load_area_bound`. So the robot-mask-area measurement
(§4) cannot be run until V9 is settled, and running it first would have bought ~9.5 GPU-h of
artifact that a refusal of V9 would invalidate. `cluster/discoverer/106_measure_robot_mask_area.sbatch`
says exactly this in its header.

## 2. `T40_RULE_V10` — ADOPTED

*The mask-validity reference is defined for one label and one appearance, and now says so.*

**Same shape as V9.** V6's mask-validity filter was a warm-saturated-fruit colour predicate being
applied to *any* label on *any* pixels. On a `plate.` pass it refused 100 % of source frames (20/20,
IoUs 0.0000) while the detector scored 0.7524–0.7773, and reported that as `coverage: 0.0` — a
correct mask, refused, presented as an absent one. On a warm restyle the reference migrates to the
table (40.5–56.4 % of the frame on `train-01-oak-tungsten`) and the counter that was supposed to
catch this stays at zero. V10 scopes the reference to the labels it was actually measured for and
raises `MaskValidityReferenceUndefined` for anything else.

**The bound is read off a gap, not coined.** `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10`
sits inside a measured gap on both scales — (4.51 %, 40.5 %) raw and (3.00 %, 20.9 %) deflated —
with the insensitivity swept in `tests/test_apple_sam2_estimator.py`. There is deliberately no env
override.

**It also closes a code/rule gap.** The change landed in commit `6a32143`. As with V9, the
repository was running unregistered behaviour; `scripts/estimators/apple_sam2.py` currently raises
for any prompt other than `apple.` on a bound no signed rule registered. Adoption regularises that.

### 2.1 Two V10 items remain OWED, and are deliberately not done today

1. `n_frames_mask_refused_reference_not_object_scale` is not in
   `measure_geom_tol.ADAPTER_RUN_COUNTERS`, so the new counter reaches an artifact as a process
   lifetime total rather than this run's number.
2. `mask_validity_reference_max_frame_fraction` is not in `apple_sam2.SEGMENTER_CONTRACT`, nor in
   `configs/transfer25/pr08_geom_tol.json` and its `.sha256` sidecar.

**Both touch the committed contract of a measurement that is currently in flight** — the 16-shard
GEOM_TOL array, job **190125**, 10 shards landed as this is written.
`contract_disagreements()` counts an **absent** field as a disagreement, so making either change now
would put the landed shards in disagreement with the contract and cost the merge its qualification.
They are scheduled for immediately after the MERGE completes. This is a sequencing decision, not a
deferral of the rule: V10 is in force from today; two of its consequences land in a few hours.

## 3. `T40_RULE_V11` — ADOPTED

*Staged partition: stage 1 is `train_styles[0:4]` in committed order plus four matched identity
repeats; the eval set is deferred, not cut.*

**Why this was inside the delegation: it is strictly risk-reducing.** It commits **8.8 %** of the
GPU allocation before the first evidence instead of **27 %** (3 216 clips / ~442 GPU-h against
10 050 clips / ~1 380 GPU-h). It defers rather than cuts the eval set. Its stage-2 branches are
fixed *in advance* — including the branch that says *"Stop and read, do not scale"* — so the
decision after stage 1 is not taken with knowledge of stage 1's result. And k = 4 is a **prefix in
committed order**, never a selection: V11 §2.1 rejects "the four most visually distinct" and "the
four that restyle best" as choices made with knowledge the document must not have.

**The threat to validity is accepted on the document's own stated grounds.** V11 §3 records that a
null at k = 4 is weaker evidence than a null at k = 10 and that stage 1 could produce a false
negative ending a line of work that would have succeeded. It is accepted because the sbatch is
chunked and resumable — going k = 4 → 10 costs the difference, not a restart — and because four
domains is a genuine randomisation where two would not be.

**What adoption licenses.** The code change V11 §2.4 says is owed, and *only* that: a stage selector
passed at submit time and recorded per clip, and a re-expression of the arm-C frame-match guard at
stage level. The second is not cosmetic — today's guard is
`instances["train"] != instances["identity"]` computed over the whole `styles.toml` and independent
of `chosen`, so **a stage generating 4 train styles against 10 identity repeats passes every check
in the file.** Until that is fixed, V11 §0's second bullet is a promise with no enforcement behind
it.

**What adoption does not license.** No clip. V11 §5's own paragraph is unchanged and still binds.

## 4. `GATE_QUALIFIED` — **NOT DISCHARGED.** Declined.

This is the first of the two items the session declined, and the record is emphatic about why.

`scripts/estimators/apple_sam2.py` holds `GATE_QUALIFIED = False`, and that single flag stamps
`gate_qualified: false` on every GEOM_TOL shard and every EST_DRIFT artifact. It is held false by
three entries in `GATE_QUALIFICATION_BLOCKERS`. Blockers 1 and 2 have their evidence produced (mask
audit sheets, the detection-score distribution, zero retries).

**Blocker 3 is untouched.** It asks for the same Isaac capture to be measured **both ways** — this
adapter per frame, and the SAM 2 video predictor propagating from frame 0 — and the two p95s
recorded, *"so the direction and size of the difference are a measurement rather than an argument."*
Nobody has run that. `docs/isaac-est-drift-runbook.md:686` prices the whole retirement at 1–3 days.

**Discharging the flag by signature would be precisely the thing `docs/handoff.md` §3 forbids.** The
blocker names the evidence that closes it; a signature is not that evidence. A gate is satisfied by
a measurement or it is not satisfied. The delegation authorised the session to decide; deciding
correctly here means declining to discharge.

### 4.1 The cost decision that goes with it, taken explicitly

`cluster/discoverer/103_measure_geom_tol.sbatch:174-178` warns that `gate_qualified` is baked into
each shard **at measurement time**, so a later discharge forces re-measuring the corpus — and calls
that *"a decision to take BEFORE the array goes in, not after."*

**Determination: let the array finish anyway, and accept a measured-but-not-quotable `GEOM_TOL`.**
The arithmetic, stated out loud because the warning deserves an answer rather than a shrug: the
re-measurement costs **~9.5 GPU-h** against a **5 000 GPU-h** allocation — **0.19 %**. That is cheap
enough that the warning does not bind. The number is worth having now: it is the input to §6's
`GEOM_TOL − EST_DRIFT_P95` budget, ten shards of it are already paid for, and knowing whether that
budget is positive is worth 0.19 % of the allocation even if the number cannot yet be quoted.

**What it may not be used for.** It may not be quoted as a result, it may not close §8 item 4, and
it may not license generation. `run_g0_gates` will continue to return `NOT_GATE_QUALIFIED`, which is
the correct behaviour and is not to be worked around.

## 5. `T40_RULE_V1` §1 — **NOT LIFTED, and not liftable by signature today.** Declined.

Verbatim, from §1:

> **Forbids, until every item in §8 is closed and T-39 has reported:** generating a corpus, training
> any weight on generated frames, and quoting any number from this document as a result.

**§1 is written as a conditional on facts, not as a permission awaiting a signature.** It names no
signatory and no affirmative act. So the honest finding is that *no determination by anyone lifts §1
today* — only closing the items does. The session was asked to decide this and the correct decision
is that there is nothing here to decide yet.

The facts, as of 2026-08-24:

| item | state |
|---|---|
| §8 item 3 — measured throughput + a GPU-h ceiling derived from it | **OPEN.** The only artifact on the cluster is `THROUGHPUT.WITHDRAWN-job189142.json`, withdrawn because the run it timed produced **zero clips** and turned 118 s of crash into "9.56 GPU-h per variant" |
| §8 item 4 — `GEOM_TOL` and `EST_DRIFT_P95` measured **and committed** | **OPEN.** Both `null` in `configs/transfer25/pr08_geom_tol.json` on disk |
| §6 G0a, G0b, G0c | **All undischarged.** None has ever returned a verdict |

Two secondary silences the reading turned up. They are **recorded, not resolved**:

- **§8 item 6 (partition committed)** was recorded **NOT CLOSED** by V3 on 2026-08-15, when
  `git ls-files configs/transfer25/` returned zero tracked files. Those files are tracked now
  (`styles.toml`, `pr08_style_partition.json`, `pr08_geom_tol.json`,
  `pr08_identity_prompt_evidence.json` and their sidecars). **No later document re-adjudicates it.**
  That is a silence, not a closure, and it should be closed by a document rather than by someone
  noticing the files are tracked.
- **§8 item 5**'s code half is observed landed in `src/wam/robot/isaac_binding.py`, but the
  measurement it unblocks was never taken, and V3 explicitly recorded it as an observation and *not
  adjudicated*. V5 then replaced §4 step 1 and item 5's naming of `isaac_binding.py` with the
  MuJoCo ground-truth route. Whether that closes item 5 or routes around it is unstated.

## 6. What is now on the critical path to a first clip

Nothing in §1–§5 generates anything. For the avoidance of any doubt, the ordered remainder is:

1. GEOM_TOL shards 10–15 land → MERGE on the free CPU QoS → `pr08_geom_tol.json` carries a real,
   **disqualified**, `GEOM_TOL`.
2. The two owed V10 items (§2.1), immediately after that merge.
3. `106_measure_robot_mask_area.sbatch` at `NUM_SHARDS=8`, two waves of four, `--time=01:45:00` —
   now unblocked, because V9 is settled and the masker identity is fixed. ~9.5 GPU-h.
4. Read that distribution and write `max_frame_fraction` + `bound_rationale` into
   `configs/transfer25/pr08_robot_mask_area.json`. **This is the item the session was asked to
   decide and could not: the artifact does not exist and the distribution behind it has never been
   measured.** The only measurement in hand is a 3-episode pilot that stamps
   `measurement_qualified: false` and that `load_area_bound` refuses by name. A bound written today
   would be a coined number in a committed file's clothing — the exact failure
   `robot_composite.py`'s own docstring names. The decision is not declined on principle; it is
   *not yet available*, and step 3 is what makes it available.
5. `TIMING=1` on `97_transfer25_restyle.sbatch` → `THROUGHPUT.json` → §8 item 3. It cannot run
   before step 4: the driver refuses to start without a committed `pr08_robot_mask_area.json`,
   because the timed pipeline must be the pipeline that will actually run.
6. §6's three gates, which have never run.

**And a separate open question that steps 1–6 do not answer**, raised by this session and left to
the owner: `T40_RULE_V12` (§7).

## 7. The line this session drew

The delegation was broad. The session did not treat it as unlimited, and the rule it applied was:

> **Adopt instrument bug-fixes whose evidence predates the rule, and scope reductions that spend
> less. Do not adopt anything that loosens a gate after seeing the gate's output, and do not
> substitute a signature for a measurement a blocker explicitly names.**

By that rule: V9 and V10 were adopted (instrument fixes, evidence measured first, code already
landed and unregistered). V11 was adopted (a scope reduction — 8.8 % instead of 27 %, with its
branches fixed in advance). `GATE_QUALIFIED` was declined (a blocker names the measurement that
closes it; a signature is not that measurement). §1 was declined (it is conditional on facts that
are open).

And one item was **raised rather than decided**: while establishing the terms of `max_frame_fraction`,
the session found that the bound is *"the bound for a gate that, on this corpus, refuses before it
ever reaches the bound."* G0c's other half — the empty-mask refusal, which has no threshold in it —
refuses **128 of 129 measured clips (99.2 %)**, because the robot is **genuinely out of shot in
~36 % of source frames** (job **189707**, `empty_mask.fraction = 0.35246`). 87 of those 128 refusals
(68 %) are on frames V9's filter did not empty. As written, G0c would refuse essentially the whole
corpus at generation time.

That is a real specification problem and it is drafted as `T40_RULE_V12`
([`PR-08-V12-g0c-empty-mask-semantics.md`](PR-08-V12-g0c-empty-mask-semantics.md)). **It was
deliberately NOT signed under this delegation**, because adopting it would mean changing a gate's
semantics after seeing that the gate refuses 99.2 % of the corpus — the one move this session's own
rule forbids, and the one `docs/handoff.md` §3 names by name. It needs the owner's eyes, not a
delegate's.

## 8. Provenance

| | |
|---|---|
| kind | determination record. **Registers no rule** |
| date | 2026-08-24 |
| decided by | the project owner, by delegation (§0), without having seen the recommendation |
| adopts | `T40_RULE_V9`, `T40_RULE_V10`, `T40_RULE_V11` |
| declines | the `GATE_QUALIFIED` discharge (§4); lifting `T40_RULE_V1` §1 (§5) |
| raises, undecided | `T40_RULE_V12` (§7) |
| defers as not-yet-available | `max_frame_fraction` (§6 step 4) |
| generation licensed | **no** |
| training licensed | **no** |
| edits to any rule document | banner + two `§8` rows in V9 and V10; the `§5` determination block in V11. **No other sentence of any rule was touched** |
| reversibility | reverting any adoption means reverting only that banner and that block |
