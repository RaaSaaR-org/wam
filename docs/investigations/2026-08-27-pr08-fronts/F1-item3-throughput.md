# FRONT 1 — PR-08 §8 item 3: the timing run, read at HEAD

**Repo:** `/home/humanoid/develop/wam` — HEAD `19826ccae06f36cd6153125042eda3e7d2a7560b`, working tree clean at read time (`git diff --stat` empty; a peer session had committed the `apple_sam2.py` edits the session-start snapshot showed as modified).
**Date of this reading:** 2026-08-27. No ssh, no sbatch, no cluster command was run. Nothing under `/home/humanoid/develop/wam` was modified.

**Evidence labels used throughout:**
`[M]` measured by me now on this workstation · `[A]` recorded in a committed artifact in this repo · `[I]` my inference from `[M]`/`[A]` · `[NM]` not measured, and what it would take.

---

## 0. Headline

**The `TIMING=1` run at HEAD is technically blocked, not merely awaiting an owner's "go".** It will generate the full 590-frame clip for `episode_000000__train-01-oak-tungsten__r00` on an H200 and then **refuse in `robot_composite.check_mask` on frame 0** — the robot mask on that frame is empty — write no `THROUGHPUT.json`, and exit 1. The episode it picks is not a coin flip: it is fixed by `head -1` of a deterministically sorted work list, and it is one of the **366 of 402** episodes that carry at least one empty robot mask.

`docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md:74` currently says of item 3: *"**Blocked on:** an owner decision to submit. Nothing technical."* `[A]` That sentence is wrong at HEAD.

A second, independent trap sits underneath it: **job 189142's disqualified `THROUGHPUT.json` (0.2 s/frame from a run that generated nothing) is still the file at the default `RUN_ID`'s path**, and the timing branch's first act is to `cat` it and `exit 0`.

---

## 1. Q1 — does the `TIMING=1` path run without `CEILING_GPU_H` / `PARTITION_CEILING_GPU_H`?

**YES.** `[A]` `cluster/discoverer/97_transfer25_restyle.sbatch`

The flag is resolved before any requirement is decided (lines 361–362):

```bash
TIMING_MODE=0
if [[ "${TIMING:-0}" != "0" ]]; then TIMING_MODE=1; fi
```

and the ceilings are then branched on it (lines 414–437, verbatim):

```bash
if (( TIMING_MODE )); then
  echo "=== TIMING=1: no GPU-h ceiling is asked for. PR-08 §8 item 3 — no budget line exists until"
  echo "    this measurement does, so the measurement cannot be gated on the budget it produces."
  if [[ -n "${CEILING_GPU_H:-}" || -n "${PARTITION_CEILING_GPU_H:-}" ]]; then
    echo "    NOTE: a ceiling was supplied anyway. It is IGNORED and recorded as supplied-but-"
    echo "          unused, so the derivation cannot later be read as having been checked."
  fi
  CEILING_GPU_H=""
  PARTITION_CEILING_GPU_H=""
else
  : "${PARTITION_CEILING_GPU_H:?T40_RULE_V2 §3 step 2: ...}"
  : "${CEILING_GPU_H:?T40_RULE_V2 §3 step 3: ...}"
  ...
fi
```

The T-39 attestation is exempt on the same branch (lines 440–447), setting
`PR08_T39_REPORTED="N/A — TIMING=1, exempt under PR-08 §1 (licensed: timing one episode on an H200)"`.

**Proved on the cluster, not only by reading:** job 189644's log (`runs/_slurm_logs/t040-restyle.189644.out:2` and `:4`) `[A]` prints exactly those two exemption banners and reaches the timing branch with neither ceiling and no attestation supplied. `TIMING=1` really does run without them.

The empty string is carried into the record: the writer converts `""` to JSON `null` (lines 1409–1411), and the artifact from job 189142 shows `"ceiling_gpu_hours_supplied_at_measurement_time": null` `[A]`.

---

## 2. Q2 — the composite dependency. **THIS IS THE ANSWER THAT MATTERS.**

### 2.1 `GATE_QUALIFIED` — **no dependency**

`[M]` `grep -rn 'GATE_QUALIFIED' scripts/robot_composite.py scripts/restyle_transfer25.py cluster/discoverer/97_transfer25_restyle.sbatch` → **zero hits in all three**. The masker imports `estimators.apple_sam2` for `_detector()`, `_predictor()`, `MASK_VALIDITY_REFERENCE`, `object_color_reference` and the pin strings only (`scripts/robot_composite.py:633-641`, `:641-670`). `apple_sam2.GATE_QUALIFIED` (`scripts/estimators/apple_sam2.py:938`, `False`) is read by `measure_est_drift`, `census_operating_point_episode` and `run_g0_gates` — **not** by anything on the timing path. `[M]`

**So the timing run is NOT blocked by the `GATE_QUALIFIED` owner signature.** (It is blocked by something else — §2.3.)

### 2.2 `pr08_geom_tol.json` — **no dependency**

`[A]` The geometry gate is at `97_transfer25_restyle.sbatch:1482-1566`, which is **after** the timing branch's `exit 0` at line 1453. The file says so at lines 739–742: *"GEOM_CONSTANTS is deliberately NOT in this list. PR-08 §1 licenses 'timing one episode on an H200' independently of the estimator work…"* and the existence loop at lines 743–749 does not include it. Confirmed empirically: job 189644 got past the work-list build into the timing branch while `configs/transfer25/pr08_geom_tol.json` still carried nulls `[A]`.

### 2.3 `robot_composite.check_mask` — **YES, and it kills the run**

The chain, each step quoted:

1. `97_transfer25_restyle.sbatch:1366-1375` — the file states the dependency itself:
   > *"THIS MAKES THE TIMING PATH DEPEND ON G0c's OWN PREREQUISITES, and that is deliberate. The driver refuses to start without staged GroundingDINO + SAM 2 checkpoints … and a committed `configs/transfer25/pr08_robot_mask_area.json` … Timing the generator without the composite would be … a wall clock around a pipeline we do not run."*
2. `scripts/restyle_transfer25.py:711` — `main()` builds `robot_composite.build_context(...)` **unconditionally**, before the first unit; `--require-success` is passed only on this path (`sbatch:1379`).
3. `scripts/restyle_transfer25.py:763 → 523` — `run_unit()` runs the backend first, asserts `vision.mp4` exists, and only then calls `composite.composite(...)`.
4. `scripts/robot_composite.py:1620` `masks, from_cache = source_masks(...)` → `:1635`
   `check_mask(mask, frame_index=index, bound=context.bound, source=str(generated_video))` **inside the per-frame loop, starting at index 0.**
5. `scripts/robot_composite.py:1386-1396` — the refusal:
   ```python
   if covered == 0:
       raise CompositeError(
           f"{source}: the robot mask is EMPTY on frame {frame_index}.\n"
           "       An empty robot mask means the composite is the identity on that frame and the "
           "GENERATED manipulator went straight into the corpus — the one failure PR-08 §6 G0c "
           "exists to make impossible. There is no threshold in this check and no number to "
           "loosen: zero is zero.\n" ...)
   ```
6. `restyle_transfer25.py:530-547` catches it per unit → `status: "error"`, output renamed to `vision.uncomposited.mp4`; `:768-773` `--require-success` → `return 1`; `sbatch:1387-1392` → **`THROUGHPUT.json` is deliberately not written**, exit 1.

### 2.4 Which episode does it pick?

`[A]` **`episode_000000__train-01-oak-tungsten__r00`, 590 frames, seed 7001** — not inferred, printed by the last real timing job:

`runs/_slurm_logs/t040-restyle.189644.out:42`
```
=== timing one episode at 640x480 on 1 GPU: {"unit": "episode_000000__train-01-oak-tungsten__r00", "episode": "episode_000000", "frames": 590, "style": "train-01-oak-tungsten", "repeat": 0, "seed": 7001}
```

That is stable at HEAD `[I]`: `TIMING_UNIT=$(head -1 "${WORK_LIST}")` (`sbatch:1337`); the work list is `eps = sorted(man["episodes"], key=lambda e: str(e["id"]))`, `mine = eps[idx-1::total]` (`sbatch:1163`, `:1221`) — with `CHUNK_INDEX=1 CHUNK_TOTAL=1` that is every episode in id order, first row = smallest id × first style of `chosen` × `r00`. Under `STAGE=1`, `chosen = train[:4]` (`sbatch:1131`), whose first element is `train-01-oak-tungsten` `[M]` (`configs/transfer25/pr08_style_partition.json`, train ids in committed order). Episode ids are exactly `episode_000000 … episode_000401` `[M]` (from `runs/pr08-robot-mask-area/POOLED.json`, 402 ids, contiguous), so the smallest is `episode_000000`. The 2026-08-24 `STAGE` selector therefore does **not** change the timed unit; 189644's unit is still today's unit.

### 2.5 Does that episode have an empty-mask frame? **Yes, on frame 0.**

Two independent committed measurements agree, both made with the committed masker:

| artifact | what it says about `episode_000000` |
|---|---|
| `runs/pr08-g0c-refusal/G0C_REFUSAL.json` (2026-08-23, RTX 5090, `Sam2RobotMasker` unmodified) `[A]` | `{"episode": "episode_000000", "frames_in_manifest": 590, "frames_scanned": 1, "first_empty_frame": 0, "refuses": true, "first_empty_emptied_by_v9_filter": false, "seconds": 0.41}` |
| `runs/pr08-robot-mask-area/POOLED.json` (402 episodes / 171 625 frames, stride 1, `measurement_qualified: true`, `source_manifest_sha256 a988dd60…`) `[M]` over the artifact | `episode_000000`: 590 frames, **254 empty**, `area_fractions[0:5] == [0.0, 0.0, 0.0, 0.0, 0.0]` |

Corpus-wide, computed by me from `POOLED.json` `[M]`:

* **366 / 402 episodes (91.0 %) carry ≥1 empty robot-mask frame** → refused by `check_mask`'s empty half.
* 175 / 402 carry ≥1 frame above the committed bound `0.64091145833333329` → refused by the area half.
* **385 / 402 (95.8 %) are refused by one half or the other. 17 episodes survive both.**
* The 91.0 % figure matches the number written into the committed bound's own rationale (`configs/transfer25/pr08_robot_mask_area.json`, `bound_rationale`): *"The empty-mask half of G0c refuses 366 of 402 episodes, or 91.0 percent, on this corpus, computed from runs/pr08-robot-mask-area/POOLED.json"* `[A]` — so this is not a new claim, it is a known one that nobody has connected to the timing submission.

I reproduced the refusal itself locally, with the committed bound and the committed code `[M]`:

```
$ .venv/bin/python  # sys.path.insert(0,'scripts'); import robot_composite as rc
bound: 0.6409114583333333  cross_checked: False  artifact: configs/transfer25/pr08_robot_mask_area.json
REFUSED:
 timing_raw/episode_000000__train-01-oak-tungsten__r00/vision.mp4: the robot mask is EMPTY on frame 0.
       An empty robot mask means the composite is the identity on that frame and the GENERATED
       manipulator went straight into the corpus — the one failure PR-08 §6 G0c exists to make
       impossible. There is no threshold in this check and no number to loosen: zero is zero.
```

I also verified `[M]` that the committed bound **passes** its segmenter cross-check against the masker HEAD builds (`load_area_bound(expect_segmenter=build_masker().provenance())` returns `0.6409114583333333`) — so `build_context` will *not* refuse the way job 189644 did on 2026-08-22 (`runs/_slurm_logs/t040-restyle.189644.out:43`, `FATAL: no usable robot-mask area bound`). The bound landing on 2026-08-26 moved the failure point **later and more expensively**: from "refuses before the model loads" to "refuses after generating a 590-frame clip".

### 2.6 What the wasted job actually costs

`[I]` from the call order in `run_unit` (backend → `vision.mp4` assert → composite): the generator runs **first**. Using the only measured generation figure (1.16 s/frame, job 189926 `[A]`) and the measured source-mask rate (0.673 s/frame over 2 653 frames on an RTX 5090, computed by me from `G0C_REFUSAL.json` `[A][M]`), a 590-frame episode is roughly `590 × 1.16 ≈ 11.4 min` of generation + model load + ~1 mask frame before the refusal. **≈ 15–25 min of H200 time, one of the 4 running-job slots, and no artifact.** `[NM]` on an H200 specifically — the 1.16 s/frame is a 96-frame diagnostic clip and the 0.673 s/frame is workstation-GPU.

### 2.7 The 17 G0c-viable episodes (if the timed episode is ever allowed to be chosen)

`[M]` from `POOLED.json`, "no empty frame **and** no frame above the committed bound":

| episode | manifest index | frames | max area fraction |
|---|---|---|---|
| episode_000093 | 93 | 448 | 0.2277 |
| episode_000098 | 98 | 439 | 0.2302 |
| episode_000114 | 114 | 451 | 0.2515 |
| episode_000115 | 115 | 386 | 0.2499 |
| episode_000116 | 116 | 424 | 0.2400 |
| episode_000117 | 117 | 426 | 0.2556 |
| episode_000118 | 118 | 460 | 0.2540 |
| episode_000120 | 120 | 464 | 0.2265 |
| episode_000121 | 121 | 448 | 0.2061 |
| episode_000136 | 136 | 357 | 0.2317 |
| episode_000137 | 137 | 417 | 0.2427 |
| episode_000243 | 243 | 417 | 0.2152 |
| episode_000244 | 244 | 469 | 0.2240 |
| episode_000245 | 245 | 497 | 0.2588 |
| episode_000371 | 371 | 422 | 0.2271 |
| episode_000373 | 373 | 366 | 0.3983 |
| episode_000375 | 375 | 418 | 0.5142 |

`episode_000243` is the one clip the 2026-08-23 scan of 129 clips found non-refusing `[A]` — the two artifacts agree.

There **is** a mechanism that reaches one of these without touching a line of code: the chunk stride. `mine = eps[idx-1::total]`, so `CHUNK_TOTAL=402 CHUNK_INDEX=94` yields exactly `episode_000093` `[I]` (episode ids are contiguous and sorted position equals `episode_index` `[M]`). **I am not recommending it as a submission.** Choosing the timed episode *because it is one of the 17 that survive G0c* is a selection made after seeing the data — the same class of move PR-08 §5's committed partition exists to forbid — and it changes what the budget line is a measurement of (a robot-always-visible episode is not a median episode). It is an **owner decision that needs a registered rule version**, not an operator flag.

---

## 3. Q3 — `THROUGHPUT.json`: contents and consumers

**Writer:** `97_transfer25_restyle.sbatch:1397-1449` (inline python). **Fields** (from the writer, and confirmed against the one real instance ever produced — job 189142's, in `runs/_slurm_logs/t040-restyle.189142.out` `[A]`, reproduced verbatim):

```json
{
  "measured_on": "1 x H200, 640x480, one episode",
  "episode": "episode_000000",
  "style": "train-01-oak-tungsten",
  "frames": 590,
  "wall_seconds": 118.0,
  "seconds_per_frame": 0.2,
  "gpu_seconds_per_frame": 0.2,
  "frames_per_variant": 172000,
  "gpu_hours_per_variant": 9.56,
  "gpu_hours_per_variant_is_lower_bound_above_1_gpu": true,
  "generator": "nvidia/Cosmos-Transfer2.5-2B@ce8440327c632d8313c3bde69db13b627ba5cae1",
  "control": "depth:0.5,seg:0.5",
  "ceiling_gpu_hours_supplied_at_measurement_time": null,
  "ceiling_gpu_hours_supplied_at_measurement_time_note": "null is the expected value. PR-08 §8 item 3: no budget line exists until this measurement does, so TIMING=1 asks for no ceiling and ignores one if supplied."
}
```

`seconds_per_frame = wall / frames`, where `wall = SECONDS` measured around the **whole driver invocation** (`sbatch:1346` `S=${SECONDS}` … `:1394`), and `gpu_hours_per_variant = spf × FRAMES_PER_VARIANT / 3600` with `FRAMES_PER_VARIANT = 172_000` hard-coded (`sbatch:1424`).

**Which script consumes it to derive the ceilings? None — a human does.** `[M]` `grep -rn 'CEILING_GPU_H' scripts/` returns only a comment in `restyle_transfer25.py:579`. There is no derivation script. The three machine consumers are all inside `97` itself:

1. `:1458-1464` — existence gate on the GENERATION path (*"PR-08 §8 item 3 makes the measurement a gate, not a warm-up"*).
2. `:1686-1790` — the ceiling gate. It reads **exactly one field**: `spf = float(tp["seconds_per_frame"])`. Everything else (`whole_partition_frames`, `corpus_frames`, `stage_instances_per_set`) comes from `partition_facts.json`. It computes `projected_chunk`, `projected_gpu_hours_this_style_set`, `projected_gpu_hours_whole_partition` and `max_passes_per_chunk = floor(share / (chunks × wall_h × nproc))`, and fails on any of: whole-partition projection > `PARTITION_CEILING_GPU_H`; pinned shares' sum > it; this set's projection > `CEILING_GPU_H`; `max_passes < 1`.
3. `:2305`, `:2415` — the whole file is embedded verbatim in each chunk's `chunk_metadata.json` as `"measured_throughput"`.

So the pipeline is: **`TIMING=1` → `THROUGHPUT.json` → a human reads `seconds_per_frame`, multiplies by `whole_partition_frames` (4 290 625, printed by the expansion `[A]`, job 189609 log), records the derivation, and passes `PARTITION_CEILING_GPU_H` + a `CEILING_GPU_H` share back on the generation submit line.** The sbatch prints those instructions at `:1443-1449`.

---

## 4. Q4 — are `STAGE` / `STYLE_SET` required with no default on the timing path?

**Yes — `STYLE_SET`, `CHUNK_INDEX`, `CHUNK_TOTAL`, `CONTROL` and `STAGE`, all five, unconditionally.** `[A]` `97_transfer25_restyle.sbatch:368-372` sits **above** the `if (( TIMING_MODE ))` branches and is not guarded by anything:

```bash
: "${STYLE_SET:?which half of the committed partition to generate: train | eval | identity. ...}"
: "${CHUNK_INDEX:?1-based chunk number. ...}"
: "${CHUNK_TOTAL:?how many chunks this run is split into. ...}"
: "${CONTROL:?which Transfer2.5 control blocks condition the restyle and at what weight ... Required on BOTH paths and never defaulted ... It also decides the TIMING number ...}"
: "${STAGE:?which stage of the committed partition to generate: 1 | 2 | all. T40_RULE_V11 ... There is deliberately NO default ...}"
```

Related refusals that also apply to `TIMING=1`:
* `STAGE=1` + `STYLE_SET=eval` is refused (`:389-395`).
* `NPROC != 1` is refused **on the timing path only** (`:1332-1336`): *"TIMING=1 measures ONE episode on ONE H200"*.
* OPEN `blocking_todos` veto the style set **including under `TIMING=1`** (`:1020-1076`). Today this is moot: `[M]` the only todo, `T40-TODO-01-identity-prompt-provenance`, is `CLOSED`, and it named `STYLE_SET=identity` anyway. `train` is unblocked.
* `GEOM_STEP_FRAMES` is *not* required on the timing path (`:641-644`).

---

## 5. Q5 — the exact submission command line (DO NOT SUBMIT)

### 5.1 What the `#SBATCH` header already sets — so the submit line must not re-invent it

`[A]` `97_transfer25_restyle.sbatch:228-255`: `--job-name=wam-t040-restyle`, `--partition=common`, `--account=ehpc-aif-2026pg01-905`, **`--qos=ehpc-aif-2026pg01-905`**, `--time=04:00:00`, `--nodes=1`, `--ntasks-per-node=1`, `--cpus-per-task=26`, `--gres=gpu:1`, `--mem=192G`, `--requeue`, `--signal=B:USR1@300`, `--open-mode=append`, `-o …/logs/t040-restyle.%j.out`.

That is already inside every hard limit: 1 GPU, 26 threads (= the 26-per-GPU ceiling), 192 GB (< 257 GB), `--qos` present, 1 job (≤ MaxJobsPU 4, ≤ MaxSubmitJobsPU 8). The runtime check at `:556-588` re-asserts the ratios against what Slurm actually allocated.

**The only overrides that belong on the line are `--time` (the header's 4 h is sized for a generation chunk and is a backfill trap) and, optionally, `--mem`.** The 4 h → 1.5 h override is documented as *part of the recipe* at `:11-19`, and the memory lever is measured: job 189584 was cut to `--mem=96G --time=01:00:00` and *"backfilled 22 hours early"*, billing weight 49 → 25 `[A]`; job 189644 ran at `--mem=98304` and the allocation check passed (`…189644.out:9`) `[A]`.

### 5.2 The command

Run on `login-plus` by a human, from `${PROJ}/wam/cluster/discoverer`, **after** the preflight in §5.3:

```bash
TIMING=1 \
STAGE=1 \
STYLE_SET=train \
CHUNK_INDEX=1 \
CHUNK_TOTAL=1 \
CONTROL=depth:0.5,seg:0.5 \
RUN_ID=t040-transfer25-restyle-timing-2026-08-27 \
sbatch --qos=ehpc-aif-2026pg01-905 \
       --time=01:30:00 \
       --gres=gpu:1 \
       --cpus-per-task=26 \
       --mem=98304 \
       97_transfer25_restyle.sbatch
```

* `--qos` is repeated on the line deliberately even though the header carries it — README rule 4, and the cost of the `normal` association (1 min / 0 GPUs) is a lost submission.
* **Do not** pass `--gres=gpu:N>1`: the timing branch refuses `NPROC != 1`, and `NPROC` defaults to `SLURM_GPUS_ON_NODE`.
* `--mem=98304` (96 GB) is the proven-working, cheaper-to-schedule value; `--mem=192G` also passes every check if a wider margin is wanted.
* **`RUN_ID` is not cosmetic here** — see defect D2. With the default `RUN_ID`, this job prints job 189142's disqualified file and exits 0 without measuring anything.
  Consequence to plan for: `THROUGHPUT.json` is written to `${PROJ}/runs/${RUN_ID}/THROUGHPUT.json`, so the later GENERATION submissions must either use this same `RUN_ID` or have the file copied into theirs. That coupling is real and undocumented in the sbatch header.

**Estimated cost:** 1.5 GPU-h reserved (1 GPU × 1.5 h wall); ~0.4–0.6 GPU-h actually consumed if it ran to completion `[I]`; **≈0.3–0.4 GPU-h consumed and nothing produced** on the current code, because it dies in `check_mask` after generating the clip.

### 5.3 Preflight that must be true before this line is worth typing

1. **The G0c blocker (D1) must be resolved** — otherwise the job is a guaranteed loss. Owner decision.
2. `${PROJ}/runs/<RUN_ID>/THROUGHPUT.json` must not already exist. `[NM]` — requires a cluster `ls`.
3. `${PROJ}/data/pr08-apple-640x480-h264-lossless/manifest.json` must still hash to `a988dd60db6ba8abec719dd9ec73ee74ca849465f1fa36666c8564f853bf91be`, or `load_area_bound`'s `expect_source_manifest` check refuses (`robot_composite.py:1281-1300`). `[NM]` — the corpus is not on this workstation (`data/` does not exist locally).
4. `${PROJ}/wam` must be re-synced from HEAD (`cluster/discoverer/sync.sh`, run from the Mac) so the cluster carries **`configs/transfer25/pr08_robot_mask_area.json` with the 2026-08-26 decided bound** — the cluster copy lags HEAD by design (`sync.sh` is rsync, not a clone, and excludes `/runs`, `/data`, `.git`). Without it the run reproduces job 189644's refusal.
5. The four multi-branch Transfer2.5 checkpoints must be warm in the HF cache (`99b_stage_transfer25_multibranch.sbatch`), or ~22 GB downloads **inside the measured window** (T-040 notes on job 189142's third defect `[A]`).
6. `HF_TOKEN` present and the `nvidia/Cosmos-Predict2.5-2B` licence accepted (job 189402 died on exactly this) `[A]`.

---

## 6. Q6 — defects that would make the run produce an unusable artifact

### D1 — **FATAL: the timing run dies in `check_mask` on frame 0, after paying for the clip**

Full evidence in §2. Severity: the job produces no `THROUGHPUT.json`, consumes an H200 slot, and the failure is *correct behaviour of a gate*, so no retry helps.

**This is not a code bug to patch quietly.** `check_mask`'s "zero is zero" is deliberate and pre-registered; `configs/transfer25/pr08_robot_mask_area.json`'s own rationale says the empty half refuses 91 % of the corpus and that **`T40_RULE_V12` (the empty-mask semantics) remains an unsigned draft** `[A]`. The three exits are all the owner's:

* **(a)** sign a rule that changes what an empty robot mask means for a **timing** run (e.g. G0c enforced for generated clips, waived for the measurement — with the argument that a timed episode is deleted and never enters a corpus, which the sbatch itself rejects for a different reason at `:1026-1040`);
* **(b)** register which episode the timing run measures (one of the 17 in §2.7), accepting that it is a selection and saying so in the rule;
* **(c)** resolve `T40_RULE_V12` for the corpus as a whole first, which is the open decision item 3 is really sitting behind.

The only change I would make to code *without* a rule is a **cheap refusal before the GPU is spent**, which changes no semantics and saves ~15 min of H200 per attempt:

```diff
--- a/cluster/discoverer/97_transfer25_restyle.sbatch
+++ b/cluster/discoverer/97_transfer25_restyle.sbatch
@@ -1345,6 +1345,26 @@
   echo "=== timing one episode at 640x480 on 1 GPU: ${TIMING_UNIT}"
+  # PR-08 §6 G0c refuses a clip on its FIRST empty robot mask, and the composite runs AFTER the
+  # generator has written the clip — so an episode that refuses costs a full generation and yields
+  # no measurement. The refusal is knowable before the GPU is touched: mask the source frames and
+  # run check_mask over them. It is the same masker, the same bound and the same predicate the
+  # composite will use; nothing here decides anything the composite would not have decided.
+  echo "=== G0c pre-flight on the SOURCE clip (no generation yet)"
+  python - "${RESTYLE_DRIVER}" "${SOURCE}/manifest.json" "${TIMING_UNIT}" "${CHUNK_DIR}/robot_masks" <<'PY' || {
+import json, pathlib, sys
+sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parent))
+import robot_composite as rc
+man_p, unit_s, cache = sys.argv[2], sys.argv[3], sys.argv[4]
+u = json.loads(unit_s)
+man = json.loads(pathlib.Path(man_p).read_text())
+ep = next(e for e in man["episodes"] if str(e["id"]) == u["episode"])
+video = pathlib.Path(man_p).parent / ep["video"]
+ctx = rc.build_context(source_manifest=pathlib.Path(man_p), cache_dir=pathlib.Path(cache))
+frames = rc.decode_clip(video)
+masks, _ = rc.source_masks(video, frames, ctx)
+for i in range(masks.shape[0]):
+    rc.check_mask(masks[i], frame_index=i, bound=ctx.bound, source=str(video))
+print(f"G0c pre-flight OK: {masks.shape[0]} frames, no empty and no over-bound mask")
+PY
+    echo "FATAL: the episode this run would time is refused by PR-08 §6 G0c on its SOURCE frames."
+    echo "       Refused BEFORE the generator ran, so no GPU time was spent on a clip that could"
+    echo "       never have been composited. THROUGHPUT.json is not written."
+    exit 1; }
   S=${SECONDS}
```

Note the mask pass is cached (`MaskCache`, keyed on the segmenter provenance), so the pre-flight's masks are reused by the composite and are **not** double-counted in the timed window that starts at `S=${SECONDS}` on the next line — but see D3: they are also not *counted* there, which changes the number. Either way this diff needs a decision, so it is written here and **not applied**.

### D2 — **the disqualified `THROUGHPUT.json` is still at the default `RUN_ID`'s path, and the timing branch short-circuits on it**

`[A]` `runs/_slurm_logs/t040-restyle.189142.out`, final line:
```
next: read /valhalla/projects/ehpc-aif-2026pg01-905/runs/t040-transfer25-restyle/THROUGHPUT.json, derive the ceiling, then submit the chunks without TIMING=1.
```
`RUN_ID` defaults to `t040-transfer25-restyle` (`sbatch:592`), and the timing branch's first act is (`:1319-1323`):
```bash
  if [[ -f "${THROUGHPUT}" ]]; then
    echo "=== ${THROUGHPUT} already exists — the measurement is done, nothing to time:"
    cat "${THROUGHPUT}"
    exit 0
  fi
```
The header's own recipe (`:8-9`) passes **no `RUN_ID`**. So the documented submit line, typed today, **exits 0 having re-blessed the 0.2 s/frame figure from a run that generated nothing** — and the generation path's gate would then price the partition at `0.2 × 4 290 625 / 3600 = 238 GPU-h` `[M]` against the ~1 380 GPU-h the only other measured figure implies (1.16 s/frame `[A]`), i.e. **under-priced by 5.8×**, with `max_passes_per_chunk` derived from the same fiction.

`[NM]` whether that file still exists on the cluster — requires a cluster `ls`, which I may not run. Job 189609 and 189644 used different `RUN_ID`s (`t040-transfer25-restyle` with a stale `chunk.env`, and `t040-transfer25-restyle-h264`) `[A]`, and nothing in the repo records the file being removed.

**Remedies, in order of preference:** (i) submit under a fresh `RUN_ID` as in §5.2; (ii) `cp` the file to `THROUGHPUT.DISQUALIFIED-189142.json`, verify, then `rm` the original — copy-verify-delete, never `mv`, per the cluster's own rule; (iii) a code change making the short-circuit refuse a file whose `wall_seconds` cannot be reconciled with its own unit — not recommended, the artifact is not the problem, the leftover is.

### D3 — `seconds_per_frame` fuses one-time cost, per-episode cost and per-clip cost, and the artifact cannot separate them

The clock starts at `sbatch:1346` (`S=${SECONDS}`) and stops at `:1394`, so `wall_seconds` includes: python startup, the framework import, the GroundingDINO + SAM 2 preflight, the Transfer2.5 checkpoint load (and any cold download), the **source-mask pass over all 590 frames**, generation, the composite, the re-encode, and the IoU diagnostic.

The sbatch argues this is safe (`:1360-1365`): *"The source mask is cached per episode and reused across the 25 restyles of it, so the marginal cost of a SECOND variant of the same episode is lower than this first one — the ceiling derived from here is therefore conservative, which is the correct direction for a budget."* **That direction claim is backwards for a budget line.** The number derived from this file is the number that *authorises* spend (`PARTITION_CEILING_GPU_H`). Inflating it does not protect the allocation, it enlarges the authorisation, and the gate cannot notice because it re-uses the same `spf` on both sides.

Quantified `[M]` from committed artifacts: the source-mask pass measures **0.673 s/frame** (2 653 frames, 1 785.3 s, RTX 5090, `G0C_REFUSAL.json`) against a generation figure of **1.16 s/frame** (job 189926). A per-episode cost paid once but multiplied by 25 style-instances adds `171 625 × 0.673 / 3600 ≈ 32 GPU-h` to the true bill and `4 290 625 × 0.673 / 3600 ≈ 802 GPU-h` to the projection — **~16 % of the whole 5 000 GPU-h allocation, authorised by arithmetic the artifact makes invisible.** `[NM]` on H200 hardware; the ratio, not the absolute value, is the point.

Minimal fix — record the phases so the human deriving the ceiling can do the division. Driver side (`scripts/restyle_transfer25.py`), unapplied:

```diff
@@ class Outcome / run_unit
-        extra["g0c"] = composite.composite(
+        t_mask = time.perf_counter()
+        extra["g0c"] = composite.composite(
             source_video=pathlib.Path(sample["video_path"]),
             generated_video=video,
             expected_frames=unit.frames,
         )
+        extra["g0c"]["composite_wall_seconds"] = round(time.perf_counter() - t_mask, 3)
+        extra["backend_wall_seconds"] = round(t_mask - t_start, 3)
```

and sbatch side, reading them back into the report:

```diff
@@ -1420,6 +1420,15 @@
 report = {
     "measured_on": "1 x H200, 640x480, one episode",
+    # The three phases the projection must NOT treat alike: startup is paid once per JOB, the
+    # source-mask pass once per EPISODE (cached across that episode's 25 style-instances), and
+    # generation once per CLIP. One fused s/frame multiplied by 4 290 625 frames prices all three
+    # per clip and over-derives the ceiling it is the input to.
+    "phase_seconds": {"driver_startup_and_model_load": startup_s,
+                      "backend_generation": backend_s,
+                      "g0c_source_mask_and_composite": composite_s},
+    "seconds_per_frame_generation_only": round(backend_s / frames, 4),
+    "seconds_per_frame_note": ("seconds_per_frame below is the FUSED wall clock. Derive the "
+                               "ceiling from the phases, not from it."),
```

### D4 — `FRAMES_PER_VARIANT = 172_000` is a hard-coded round number, 0.22 % above the measured corpus

`sbatch:1424`. The corpus is **171 625 frames** `[M]` (summed from `POOLED.json`; the same number appears in `pr08_robot_mask_area.json` and in T-040's corpus verification note `[A]`). `partition_facts.json` already carries `corpus_frames`, computed from the manifest in the same job, so the constant is both wrong and redundant. Direction: over-estimate, ~0.02 GPU-h at 0.2 s/frame — negligible in size, but it is a coined number inside the one artifact PR-08 §8 item 3 forbids inventing numbers in. Fix: read `corpus_frames` from `partition_facts.json` and record `frames_per_variant_source: "manifest"`.

### D5 — the timed episode is the 4th-longest in the corpus, and nothing says so

`[M]` frames: min 249, median 421.5, mean 426.9, max 749; `episode_000000`'s 590 frames rank **4 of 402**. One-time costs amortise better over a long episode, so the fused `seconds_per_frame` is *lower* than it would be on a median episode — the opposite bias to D3, and neither is recorded. `THROUGHPUT.json` carries `frames` but no corpus context, so a reader cannot tell. One extra field (`episode_frames_percentile`) makes it self-describing.

### D6 — non-defect, worth recording: the stale `chunk.env` that killed job 189609 can no longer collide

Job 189609 died on the stamp check because `${OUT}/chunks/train-01of01/chunk.env` was pinned to the AV1 `SOURCE` `[A]`. At HEAD, `CHUNK_TAG` includes the stage (`sbatch:601`, `s%s-%s-%02dof%02d`), so today's tag is `s1-train-01of01` — a different directory. The 2026-08-24 stage selector incidentally defused that trap. `[I]`

---

## 7. What is true about item 3, in one paragraph

`THROUGHPUT.json` does not exist and cannot be produced by the documented submit line at HEAD. It is **not** blocked by `GATE_QUALIFIED`, **not** by `pr08_geom_tol.json`, and **not** by the ceilings it derives — all three exemptions are real and verified. It is blocked by PR-08 §6 G0c, which the timing path deliberately runs in full, on a corpus where 366 of 402 episodes carry an empty robot mask, and on the one episode the timing path is hard-wired to pick — whose very first frame is empty. Until `T40_RULE_V12` (empty-mask semantics) is signed, or a rule registers which episode the measurement uses, submitting the timing run spends an H200 slot to reproduce a refusal that two committed artifacts already predict.

---

## Adversarial re-read

Second reader, 2026-08-27, repo `/home/humanoid/develop/wam` at HEAD `19826ccae06f36cd6153125042eda3e7d2a7560b`
(confirmed by `git rev-parse HEAD`). No ssh, no sbatch, no cluster command. Nothing under the repo
was modified; everything below was run read-only with `/home/humanoid/develop/wam/.venv/bin/python`
and `/home/humanoid/develop/wam/.venv/bin/pytest`. Same evidence labels as above.

**Verdict: the headline stands, two load-bearing supports do not.** The G0c blocker (D1's *fact*) is
confirmed independently. R1 and R2 below are load-bearing failures, so this deliverable does not
survive as written.

### What I re-checked and confirms (so a later reader does not re-do it)

Every one of these I computed or ran myself, not read off the text above.

* `runs/pr08-robot-mask-area/POOLED.json` `[M]`: 402 episodes, **171 625** frames,
  `measurement_qualified: true`, `source_manifest_sha256 a988dd60db6ba8abec719dd9ec73ee74ca849465f1fa36666c8564f853bf91be`.
  At bound `B = 0.64091145833333329`: **366** episodes carry ≥1 zero fraction, **175** carry ≥1
  fraction > B, **385** are refused by one half or the other, **17** survive both. `episode_000000`:
  590 frames, `area_fractions[0:5] == [0.0,0.0,0.0,0.0,0.0]`, **254** empty. Frame counts
  min 249 / median 421.5 / mean 426.9 / max 749; 590 ranks **4 of 402**. Every figure in §2.5, §2.7
  and D5 reproduces **exactly**, and the 17-episode table matches episode id, index, frames and max
  fraction to four decimals.
* `runs/pr08-g0c-refusal/G0C_REFUSAL.json` `[M]`: the `episode_000000` record is verbatim as quoted.
  Summed over its 129 records: 2 653 frames, 1 785.32 s → **0.6729 s/frame**. I additionally
  regressed `seconds` on `frames_scanned` across the 129 records: per-clip overhead **−0.16 s**,
  marginal **0.681 s/frame** — so 0.673 is a genuine per-frame masking rate on that workstation GPU
  and not an artefact of 129 clip set-ups. D3's mask figure survives *as an RTX 5090 number*.
* The refusal chain, re-read line by line `[M]`: `restyle_transfer25.py:711` builds the context
  before the loop; `:763` calls `run_unit`; inside `run_unit` the backend runs first, `vision.mp4`
  is asserted, and only then `:523 composite.composite(...)`; `robot_composite.py:1620`
  `source_masks`, `:1635` `check_mask(...)` inside `for index in range(src.shape[0])`, `:1388`
  `if covered == 0: raise CompositeError`. **Generation is paid before the refusal. Confirmed.**
* `TIMING=1` really does exempt both ceilings (`sbatch:414-437`) and the T-39 attestation
  (`:440-447`); `runs/_slurm_logs/t040-restyle.189644.out:2,4` prints both banners `[A]`.
* The timed unit is fixed. `sbatch:1163` `eps = sorted(man["episodes"], key=lambda e: str(e["id"]))`,
  `:1221` `mine = eps[idx-1::total]`, episode-major loop, `:1337` `head -1`. `train[0]` in
  `configs/transfer25/pr08_style_partition.json` is `train-01-oak-tungsten` `[M]`.
  `189644.out:42` prints the unit verbatim as quoted `[A]`.
* **No `GATE_QUALIFIED` dependency — and I checked the harder version of the claim too.** Zero grep
  hits in the three files, as stated; *and* the masker's `provenance()` carries no `gate_qualified`
  key (I ran `rc.build_masker().provenance()` — keys are adapter, adapter_version,
  adapter_version_note, box_rule, box_threshold, name, object_grounding_filter, prompt,
  text_threshold, upstream_retry_not_run, version) `[M]`. So the canonical ordering trap — *measuring
  before `GATE_QUALIFIED` flips bakes `gate_qualified: false` into the artifact* — **does not apply to
  item 3**: nothing on this path stamps it, and `THROUGHPUT.json`'s field list (`sbatch:1419-1435`)
  does not contain it. `GATE_QUALIFIED` is still `False` at `scripts/estimators/apple_sam2.py:938`,
  with `GATE_QUALIFICATION_BLOCKERS` now the empty tuple (`:635`) `[M]`.
* **No committed-contract breakage.** I hunted specifically for the two named in the brief:
  `SEGMENTER_CONTRACT` and `pr08_geom_tol.json`'s `contract_fields` occur only in
  `scripts/measure_geom_tol.py`, `scripts/measure_est_drift.py`, `scripts/estimators/apple_sam2.py`
  and one task file `[M]`. **No proposal in this deliverable touches either.** That hunt comes up
  empty and should be recorded as empty.
* **No gate is rewritten and nothing unsigned is signed.** §6 D1 correctly refuses to choose among
  (a)/(b)/(c), correctly names episode-selection-from-the-17 as a post-hoc selection needing a
  registered rule version, and correctly reports `T40_RULE_V12` as an unsigned draft — which
  `configs/transfer25/pr08_robot_mask_area.json`'s `bound_rationale` states in as many words, along
  with the 366/402 = 91.0 % figure `[A]`, both verified.
* **All five "runnable now" commands are local, read-only, and do not mutate the repo. I ran all
  five** `[M]`: `check_style_partition.py` → `PASS`, `todos OK 1 recorded, 0 open`;
  `load_area_bound(expect_segmenter=build_masker().provenance())` → `0.6409114583333333` (and
  `build_masker()` is lazy — no model download, no GPU); the `check_mask` call → `CompositeError`;
  the 17-episode list → reproduces; `pytest tests/test_restyle_transfer25.py
  tests/test_transfer25_staging.py` → **119 passed in 3.60s**. Nothing touches the cluster.
* `sbatch:592` `RUN_ID=${RUN_ID:-t040-transfer25-restyle}`; `:1319-1323` cats and `exit 0`; the
  header recipe at `:8-9` passes no `RUN_ID`; `189142.out` ends with `next: read
  /valhalla/.../runs/t040-transfer25-restyle/THROUGHPUT.json` and its printed artifact is verbatim
  as quoted, produced by a run whose own log says `=== done: 0 success, 1 error` `[A]`. **D2's
  premise is confirmed.**
* D4's fix is ordering-valid: `partition_facts.json` is written at `sbatch:1252` and
  `corpus_frames` computed at `:1166`, both **before** the timing branch at `:1314` `[M]`.
* A **third** artifact nobody cited independently confirms the emptiness:
  `runs/_slurm_logs/t040-halluc-probe.189926.out:35` — `episode_000000: source mask empty on 93/96
  frames` `[A]`.

### R1 — `[A]` on a number that is not in the artifact. **LOAD-BEARING, REFUTED.**

§2.6, D2 and D3 all rest on *"the only measured generation figure (1.16 s/frame, job 189926 `[A]`)"*.

`grep -n "1\.16" runs/_slurm_logs/t040-halluc-probe.189926.out` returns **no line stating any
per-frame rate** `[M]`. The number is the investigator's own arithmetic over four lines that say
`Average time per chunk:` — 57.75, 55.45, 55.46, 55.47 s (`:33498`, `:66932`, `:100368`, `:133804`)
`[A]` — at 2 chunks per 96-frame clip, i.e. 1.155–1.203 s/frame. That is `[I]`, not `[A]`, and the
front asked precisely for this: a claim labelled measured/recorded that was computed.

Worse, it is **the wrong scope**. The same log shows what those chunk timings exclude, all inside
the same unit and all real GPU seconds `[A]`: the depth pass (`11:33:19 → 11:33:26`, ~7 s), the
on-the-fly SAM2 seg pass (`11:33:26 → 11:33:45`, ~19 s), the control-video and output encodes, and
per-unit setup. Measured by me from the log's own timestamps `[M]`, unit boundaries
`Generating 1 samples:` → next such line: 11:23:21 → 11:26:48 (207 s), 11:26:48 → 11:30:02 (194 s),
11:30:02 → 11:33:19 (197 s), 11:33:19 → 11:35:39 (140 s, last), i.e. **1.46–2.16 s/frame per unit**,
and the job as a whole `=== probe_hallucination.py exited 1 after 988s` over 384 generated frames =
**2.57 s/frame** including model load. The `THROUGHPUT.json` clock (`sbatch:1346 S=${SECONDS}` →
`:1394`) is a *whole-invocation* wall, so the pipeline rates are the comparable ones, not 1.16.

Consequences that must be withdrawn as stated:

* **D2's "under-priced by 5.8×" is not supported.** `0.2 × 4 290 625 / 3600 = 238.4 GPU-h` is right
  `[M]`, but the denominator is wrong: against this artifact's own per-unit pipeline rate (~2.0
  s/frame) the whole partition is ~2 384 GPU-h and the factor is ~**10×**, not 5.8×. The *direction*
  of D2 survives and is in fact understated; the *number* does not.
* **D3's "0.673 against 1.16" ratio, which the deliverable calls "the point", compares an RTX 5090
  masking rate to an H200 diffusion-only rate** — two GPUs and two scopes. It cannot carry the
  "~16 % of the 5 000 GPU-h allocation" weight the summary puts on it.
* §2.6's `590 × 1.16 ≈ 11.4 min` and "≈0.3–0.4 GPU-h burned for nothing" both **understate** the loss
  D1 predicts.

### R2 — the D1 pre-flight diff changes the pre-registered measurand. **LOAD-BEARING, REFUTED.**

§6 D1 offers the diff as *"a cheap refusal before the GPU is spent, **which changes no semantics**"*.
It changes the one thing PR-08 §8 item 3 exists to fix.

`MaskCache.key` (`scripts/robot_composite.py:1005-1015`) is keyed on **`_file_sha256(source_video)`
plus `segmenter_identity(provenance)` — not on the path** `[M]`. The diff writes its cache to
`${CHUNK_DIR}/robot_masks`; the driver's default is `args.mask_cache or (args.out.parent /
"robot_masks")` (`scripts/restyle_transfer25.py:718`) and the timing invocation passes
`--out "${TROOT}"` with `TROOT=${CHUNK_DIR}/timing_raw` and **no `--mask-cache`** (`sbatch:1379-1387`)
`[M]`. `${CHUNK_DIR}/timing_raw`.parent is `${CHUNK_DIR}`. **The two cache directories are the same
directory and the two keys are the same key.** So the composite takes the hit branch
(`robot_composite.py:1565-1566`, `return hit, True`) and the entire 590-frame source-mask pass falls
**outside** the window that opens on the very next line at `sbatch:1346`.

At the deliverable's own 0.673 s/frame that is ~397 s removed from a wall the deliverable itself puts
at ~700–1 200 s — a **30–40 % cut in `seconds_per_frame`, in the under-deriving direction**. Which is
what the file pre-registers against, in the same paragraph the deliverable quotes approvingly in
§2.3: `sbatch:1358-1365` *"a throughput number that excluded the gate would under-derive the GPU-h
ceiling the generation path is then held to"*, and `:1373-1379` *"Timing the generator without the
composite would be … a wall clock around a pipeline we do not run, feeding a GPU-h ceiling the
pipeline we DO run is then held to."*

The deliverable's own footnote concedes it — *"they are also not counted there, which changes the
number"* — which **contradicts the D1 sentence that introduces the diff**, and the summary softens it
further to "changes no gate semantics". Under `docs/handoff.md` §3 this is not an implementation
detail a session may propose as semantically neutral: it silently redefines what PR-08 §8 item 3's
number is a measurement of. It belongs in the same bucket as D1's (a)/(b)/(c) — owner, with a
registered rule version — and must be labelled that way, not offered as the one change "I would make
*without* a rule".

### R3 — "saves ~15 min of H200 per attempt" does not net out the pre-flight's own cost. `[M]`

`source_masks` masks the **whole clip** before any `check_mask` runs
(`robot_composite.py:1567 masks = np.stack([... for frame in frames])`, then the loop at `:1633`).
So the proposed pre-flight pays 590 mask-frames — ~6.6 min at the deliverable's own rate — plus a
second GroundingDINO + SAM 2 load in a separate process (`build_context(preflight=True)`,
`:1533-1534`), to discover an emptiness that `G0C_REFUSAL.json` establishes on this very episode in
**0.41 s** by masking **one** frame (`"frames_scanned": 1, "seconds": 0.41`) `[A]`. The instrument
that produced that artifact is the cheap design; the diff as written is ~1 000× more expensive than
needed for `episode_000000` and its net saving is a fraction of the "~15 min" claimed.

### R4 — both D3 diffs would `NameError`. `[M]`

Driver-side: it calls `time.perf_counter()` and reads `t_start`. `scripts/restyle_transfer25.py`
imports `argparse, json, os, pathlib, sys, traceback, dataclasses, numpy, robot_composite`
(lines 85-101); `grep -c "import time"` → **0**; `t_start` appears nowhere in the file.
Sbatch-side: the report block reads `startup_s`, `backend_s`, `composite_s`, none of which any part
of the proposal defines, and nothing in it parses them back out of `sample_outputs.json`. Presented
under "**Minimal fix**", these are sketches, and the deliverable should say so.

### R5 — D3's summary number over-states its own body. `[M]`

The body correctly gives both halves: 171 625 × 0.673/3600 ≈ **32** GPU-h paid once,
4 290 625 × 0.673/3600 ≈ **802** GPU-h if priced per clip. The delta from counting it 25× instead of
1× is therefore **770**, not 802. The structured summary states 802 as the delta, and states
"Measured:" over a figure the body itself labels `[NM]` on H200 hardware.

### R6 — the diff would write a false record into a slurm log. `[M]`

`python - … <<'PY' || { echo "FATAL: the episode this run would time is refused by PR-08 §6 G0c on
its SOURCE frames." … exit 1; }` fires on **any** non-zero exit of that block: a missing or
unreadable bound, a `source_manifest_sha256` mismatch (`robot_composite.py:1292-1302`), an unstaged
SAM 2 checkpoint, a `StopIteration` on `next(e for e in man["episodes"] …)`. In a project that cites
slurm logs as artifacts — this deliverable cites five — a branch that prints "refused by G0c" over an
unrelated failure manufactures a false record of a gate firing. (The shell form itself is valid: I
tested the heredoc-plus-`||`-group construction under `bash -n` and at runtime `[M]`, and
`ep["video"]` is the right manifest key — `restyle_transfer25.py:326` uses `episode["video"]` `[M]`.)

### R7 — the named failure line is one of at least two candidates. `[M]` / `[NM]`

`composite_clip` checks `src.shape[0] != int(expected_frames)` at `robot_composite.py:1602` and
`gen.shape != src.shape` at `:1610`, **both before** `source_masks` at `:1620`. If Cosmos-Transfer2.5
returns any frame count other than 590 for a 590-frame source, the run dies on the shape refusal and
never reaches `check_mask`. **NOT MEASURED — requires a cluster run**: the only generation evidence in
this repo is a 96-frame probe (job 189926), and no 590-frame clip has been generated. The *outcome*
D1 asserts is unchanged either way — a full generation is paid and no `THROUGHPUT.json` is written —
but the headline states one specific line as the killer where the artifacts support only "one of the
composite's refusals".

### R8 — "reproduced the exact refusal" reproduces something else. `[M]`

§2.5 and the summary say the refusal was reproduced "against the committed bound and the committed
code". What was run is `check_mask(np.zeros((480,640), bool), frame_index=0, …)` — a hand-made zero
array. I re-ran it and it refuses identically, but it demonstrates only that `check_mask` raises on
zeros; **the masker was never run on `episode_000000`**. The substantive fact (frame 0 is empty) rests
entirely on `POOLED.json` and `G0C_REFUSAL.json`, and now also on `189926.out:35` — all three of which
do carry it, so the conclusion survives. The sentence claims a reproduction that was not performed.

### R9 — an unstated hardware caveat on the 17-episode table. `[A]`

`configs/transfer25/pr08_robot_mask_area.json`'s own `bound_rationale` records that per-frame
classification near the bound is **hardware-dependent** in one band (37 of 44 frames moved by >0.01
when re-rendered on an RTX 5090; one frame at the gap's lower edge landed *inside* the gap), and
concludes: *"check_mask refuses a WHOLE CLIP when a single frame exceeds the bound, so the set of
refused clips is a property of the machine that runs the composite as well as of the corpus."* §2.7
presents the 17 as "the G0c-viable episodes" without that caveat. The **area** half of the 17-way
filter is machine-conditional; the **empty** half — which is what kills `episode_000000` and 366 of
402 — is not, so the headline is unaffected and only the 17-list inherits the caveat.

### R10 — citation drift (all lines exist; none reverses in context). `[M]`

* `**Blocked on:** an owner decision to submit. Nothing technical.` is at
  `docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md:73`, cited as `:74`.
* `chosen = chosen[:k1] if stage == "1" else chosen[k1:]` is at
  `cluster/discoverer/97_transfer25_restyle.sbatch:1133`, cited as `:1131`.
* The `GEOM_CONSTANTS` existence check is at `:1483`, cited as `:1482` (the conclusion — after the
  timing branch's `exit 0` at `:1453` — is unaffected).
* §2.4 says 189644's printed unit "is still today's unit". The unit *identity* is; the printed JSON
  is not — the work-list row now carries a `"stage"` field (`sbatch:1237-1240`, added 2026-08-24),
  absent from the 2026-08-22 log line quoted. Cosmetic.

### What a corrected version of this deliverable owes

1. Re-label 1.16 s/frame as `[I]`, state its scope (diffusion chunks only, excludes depth/seg/encode),
   and re-derive D2 and §2.6 from the artifact's own per-unit rate (1.46–2.16 s/frame) or from the
   whole-job rate (2.57 s/frame), saying which and why.
2. Move the D1 pre-flight diff out of "changes no semantics" and into the owner's column, with the
   cache-key argument (`robot_composite.py:1005-1015` + `restyle_transfer25.py:718`) written out, and
   either place the pre-flight *inside* the timed window or state plainly that it removes ~30–40 % of
   the measurand.
3. Redesign the pre-flight to mask frame-by-frame and stop at the first refusal, as
   `G0C_REFUSAL.json`'s own instrument does, and separate its failure branch from a G0c verdict.
4. Fix or explicitly mark the D3 diffs as non-running sketches.
5. Correct 802 → 770 as the delta, and drop "Measured" from the H200 extrapolation.
6. State that the shape checks at `robot_composite.py:1602/1610` precede `check_mask`, so the named
   line is `[NM]` until a 590-frame clip has been generated.

**Unchanged by all of the above:** `THROUGHPUT.json` cannot be produced by the documented submit line
at HEAD; the block is PR-08 §6 G0c on `episode_000000`'s frame 0; it is not `GATE_QUALIFIED`, not
`pr08_geom_tol.json`, and not the ceilings; and the sprint document's *"Nothing technical"* is wrong.
