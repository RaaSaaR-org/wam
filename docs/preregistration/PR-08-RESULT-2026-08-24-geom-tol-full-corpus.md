# PR-08 RESULT — `GEOM_TOL` measured over the whole corpus, and what it may not be used for

**Date:** 2026-08-24
**Measures:** `T40_RULE_V1` §4 / §6 G0b — the geometry tolerance and the estimator error budget
**Registers no rule. Discharges no gate. Licenses no clip.**

---

## 0. The one-paragraph version

The 16-shard `GEOM_TOL` array finished. Pooled over **402 of 402 episodes and 171 625 frames**,
coverage **1.000**, `GEOM_TOL = 0.4786 px`. `EST_DRIFT_P95` was measured the day before on the
MuJoCo ground-truth route at `0.2361 px`, so the G0b budget `GEOM_TOL − EST_DRIFT_P95` would be
**+0.2425 px** — positive, which is the thing §6 G0b requires and which nobody could previously
state. **Both numbers carry `gate_qualified: false`, for the same single reason, and neither may
be quoted as a result or written into `configs/transfer25/pr08_geom_tol.json`.** §8 item 4 is
**still OPEN**. What the run did buy is the second half of the evidence
`GATE_QUALIFICATION_BLOCKERS[1]` asks for, and one unplanned corroboration described in §4.

---

## 1. What was measured

| | |
|---|---|
| `GEOM_TOL` | **0.4786 px** (`0.47857992441961017`) |
| basis | ONE median over the pooled per-step displacements — shard medians are never averaged |
| corpus | 402 / 402 episodes, 171 625 frames, 171 181 of 171 223 steps |
| coverage | **0.99975** (`min_coverage` cleared) |
| steps dropped | 42, all `object_not_visible`, never folded in as zero displacement |
| shards | 16, jobs 189935 / 189971 / 189984 / 190125, 2026-08-23 → 2026-08-24 |
| merge | job 190191, free CPU QoS (`2cpu-single-host`), 8 s, **exit 3** |
| artifact | `runs/pr08-geom-tol/pr08_geom_tol.json` — **under `runs/`, deliberately not under `configs/`** |

Distribution of per-step object-centroid displacement, for whoever has to judge the tolerance
later: p50 **0.479**, p75 1.896, p90 4.126, p95 5.420, p99 7.856, max 72.92 px, mean 1.356,
std 2.114.

`EST_DRIFT_P95`, measured 2026-08-23 on the MuJoCo route (`T40_RULE_V5`), two sample sizes:

| capture | frames | scene states | `EST_DRIFT_P95` |
|---|---|---|---|
| `EST_DRIFT-mujoco-s20-f240.json` | 240 | 20 | 0.2099 px |
| `EST_DRIFT-mujoco-s60-f720.json` | 720 | 60 | **0.2361 px** |

The larger sample gives the larger p95, which is the direction that tightens G0b, so the
720-frame number is the one quoted above. Its `error_direction` is recorded as *"conservative
(argued): less photoreal frames → larger p95 → smaller `GEOM_TOL − EST_DRIFT_P95` → stricter
G0b"* — **argued, not measured**, and `is_lower_bound` is `false` without `true` being claimed
for the other side.

---

## 2. Why it may not be used, and the exact mechanism

Every one of the 16 shards, and the merge, carries a single disqualification reason:

> `mask method 'grounding-dino+sam2+depth-anything-v2' is not gate-qualified`

**There is no name allowlist and nothing about the string is wrong.** The chain is:
`measure_geom_tol.sam2_method()` reads `declared_gate = bool(getattr(module, "GATE_QUALIFIED", False))`,
and `scripts/estimators/apple_sam2.py` holds `GATE_QUALIFIED = False`. Every other conjunct of
`gate_qualified=declared_gate and bool(checkpoints) and contract is not None` is already true —
the checkpoints are staged at pinned revisions and the committed contract matched the shards
field for field. **One flag, held false by three entries in `GATE_QUALIFICATION_BLOCKERS`, stamps
every GEOM_TOL shard and every EST_DRIFT artifact.**

`EST_DRIFT`'s own disqualification lists three reasons — `estimator_not_gate_qualified`,
`geom_tol_does_not_record_gate_qualified`, `geom_tol_is_not_gate_qualified` — the last two only
because it read the committed `pr08_geom_tol.json`, which is still the pre-commitment with six
`null` measurement fields. Those two would clear on their own once a qualified GEOM_TOL is
committed; the first is the same flag.

**The committed pre-commitment was restored.** The merge writes over
`configs/transfer25/pr08_geom_tol.json` by design — §4 requires the contract and the measurement
to live in one document — so the disqualified copy was reverted on the cluster immediately after
the run and the tracked file still carries `null` in all six measurement fields. Nothing in
`configs/` has been changed by this measurement.

---

## 3. What the run bought against the blockers — and what it did not

**`GATE_QUALIFICATION_BLOCKERS[1]`** asks for its evidence *"from a full pass, so the retry's
contribution is visible rather than assumed."* That pass has now happened. Over all 171 625
frames:

| counter | value |
|---|---|
| `n_segment_calls` | 171 625 |
| `n_frames_without_detection` | **0** |
| `n_frames_with_empty_mask` | **0** |
| `n_frames_retry_fired` | **0** |
| `n_frames_retry_recovered` | 0 |
| `n_frames_mask_refused` | **36** |
| `n_frames_mask_refused_no_reference` | 0 |
| `n_below_box_threshold` | **0** |

Detection scores over the same frames: min 0.166, p1 0.549, p5 0.733, p50 0.850, mean 0.837,
max 0.938.

**The hazard blocker [1] names is empty on this corpus, not merely small.** Its surviving
objection was that the `(0.10, 0.10)` retry *"buys detections by accepting weak ones, which on an
occluded frame can replace an honest all-False mask with a confident box on the wrong object …
That inflates coverage while degrading the mask."* The retry **fired zero times in 171 625
frames**, and `n_below_box_threshold = 0` says the same thing from the other side: not one
detection in the corpus came in below the first-pass threshold. The mechanism could not have
operated, because it never ran. A 169-frame local audit had already reported `n_frames_retry_fired
= 0` on 2026-08-22; this is the same finding at roughly a thousand times the sample, over the
whole corpus rather than a sample of it.

**This does not discharge blocker [1].** Its discharge is *"the same evidence as blocker 1, **plus**
the recorded detection-score distribution and retry counts from a full pass."* The full-pass half
is now on disk. The blocker-[0] half — a human looking at overlaid masks — is not, and blocker
[1] cannot close ahead of it.

**Blocker [0] is untouched by this run** and is arguably strengthened by it. `n_frames_without_detection
= 0` and `n_frames_with_empty_mask = 0` mean the detector returned a box on **every single frame of
the corpus**, which is precisely the witness blocker [0] says is worthless: *"coverage 1.0 says a
box was returned on every frame, not that it was the APPLE's box."*

**Blocker [2] is untouched and was not addressed.** It asks for one capture measured **both ways** —
this adapter per frame, and `SAM2VideoPredictor` propagating from frame 0 — with both p95s
recorded. That has not been run. One thing worth recording for whoever attempts it: **the MuJoCo
captures that exist are structurally unsuitable for it.** They span 20 and 60 *discrete scene
configurations* with 12 frames each, and the apple is a static prop; propagation from frame 0
across a jump cut between configurations measures nothing about propagation. Blocker [2] needs a
temporally coherent capture with ground truth, and no such capture exists in this repository.
Whether the `T40_RULE_V5` simulator-agnostic route may substitute for the word "Isaac" in blocker
[2]'s text is a rule question, not a session's to answer.

---

## 4. An unplanned corroboration: the 36 refusals are all one episode, and it is *the* episode

This was not looked for. It is recorded because it is checkable and because it bears on blocker [0].

The `T40_RULE_V6` mask-validity filter refused **36 frames** across the whole corpus — 0.021 %.
All 36 landed in **shard 7**. Independently, of all 402 episodes, exactly **one** has
`n_frames != n_frames_with_centroid`:

```
episode_000094   n_frames 509   n_frames_with_centroid 473   (gap: 36)   n_steps_dropped 42
                 shard_of("episode_000094", 16) = 7
```

`episode_000094` is **the episode the 2026-08-22 human mask audit independently flagged** — 9 of
its sampled frames were *"a confident, well-formed mask of THE PLATE"*, plate overlap 0.985–0.992,
IoU 0.00 against the apple, described there as a run of *"~35 consecutive frames."*

So two instruments — a colour-reference IoU filter running on the cluster over the full corpus,
and a human reading overlay sheets from a 169-frame sample on a workstation — converge on the same
episode, and the filter's count (36) sits right on the audit's estimate of the run length (~35).

**What this is evidence for:** the V6 filter caught the one wrong-object failure anybody has ever
actually seen in this corpus, and it raised no refusal in any of the other 401 episodes.

**What this is not evidence for, and the limit is the important half.** The audit sampled 169 of
171 625 frames — **0.1 %**. That the filter caught the defect in the one episode a human happened
to look at says nothing about episodes nobody looked at. The two instruments are also not fully
independent: both are downstream of the same segmenter, and the audit's own record notes that
*"every observer here is a model checking masks produced by a pipeline a model wired up, which is
a correlated observer."* A convergence between a filter and a correlated observer is weaker than
it reads. **It does not discharge blocker [0], and it is not offered as doing so.**

---

## 5. Status after this run

| | |
|---|---|
| §8 item 3 (throughput) | **OPEN** — no `THROUGHPUT.json` exists anywhere in the repo |
| §8 item 4 (`GEOM_TOL` + `EST_DRIFT_P95` committed) | **OPEN** — both measured, both disqualified, `configs/transfer25/pr08_geom_tol.json` still all-`null`, `configs/transfer25/pr08_est_drift.json` does not exist |
| §6 G0a / G0b / G0c | **never run.** No gate has returned a verdict on any corpus |
| `GATE_QUALIFIED` | **`False`.** Blocker [1]'s full-pass half now has its evidence; blockers [0] and [2] are open |
| `T40_RULE_V1` §1 | **binds in full.** Not lifted, and declined for lifting on 2026-08-24 |
| generation licensed | **no** |
| training licensed | **no** |

**The re-measurement cost, restated because it is the thing that makes the decline cheap.**
`gate_qualified` is baked into each shard at measurement time, so discharging the flag later forces
re-measuring the corpus. That is ~9.5–14.6 GPU-h against a 5 000 GPU-h allocation — **0.19–0.29 %**.
The warning in `103_measure_geom_tol.sbatch` that this is *"a decision to take BEFORE the array
goes in"* was taken, deliberately, in `PR-08-DET-2026-08-24-four-determinations.md` §4.1.

---

## 6. Provenance

| | |
|---|---|
| kind | result record. **Registers no rule** |
| measures | `T40_RULE_V1` §4, §6 G0b |
| GEOM_TOL jobs | 189935 (shards 0–3), 189971 (4–7), 189984 (8–9), 190125 (10–15), merge 190191 |
| EST_DRIFT | workstation, MuJoCo route, 2026-08-23 |
| artifacts | `runs/pr08-geom-tol/pr08_geom_tol.json`, `runs/pr08-geom-tol/shards/shard-{0..15}.json`, `runs/pr08-est-drift/EST_DRIFT-mujoco-s{20,60}-f{240,720}.json` |
| written into `configs/` | **nothing** |
| discharges | **nothing** |
| generation licensed | **no** |
| training licensed | **no** |
