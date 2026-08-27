# RUNBOOK — the GEOM_TOL re-run, 13.64 GPU-h

**Decision it executes.** The owner took the fork in
[`preregistration/PR-08-RESULT-2026-08-27-geom-tol-is-measured-and-uncommittable.md`](preregistration/PR-08-RESULT-2026-08-27-geom-tol-is-measured-and-uncommittable.md)
on the **re-measure** side. The landed `0.47857992441961017 px` is not salvaged; the array runs
again at HEAD and produces a number that may be committed.

**Status.** Prepared, **not submitted.** Every command below is run by a human operator from the
workstation checkout. No session may submit any of them.

**What this run now buys, and it changed today.** The script's own header still says in several
places that `apple_sam2.GATE_QUALIFIED` is `False`, that every shard exits 3, and that the number
must not be committed. **That is stale text.** At HEAD the flag is `True`
(`scripts/estimators/apple_sam2.py:967`, commit `13f0416`), and all three conjuncts of
`gate_qualified = declared_gate and bool(checkpoints) and contract is not None` pass — verified
locally by running the preflight's own comparison:

```
SPEC estimators.apple_sam2   GATE_QUALIFIED True
committed contract found at 'segmenter' (16 fields)
contract_disagreements(live, committed) -> []      # agrees field for field
_adapter_checkpoints(adapter) -> 3 pinned repos
```

So the shards will exit **0**, the merge takes its committable branch, and **this array buys a
committable number** — provided step 0 has run.

---

## Step 0 — `sync.sh`. MANDATORY, AND IT IS FIRST.

```bash
./cluster/discoverer/sync.sh
```

Cost: minutes of rsync, **0 GPU-h**.

**This is not hygiene, it repairs damage.** Job 190191's merge wrote its merged, disqualified
artifact **over the cluster's own pre-commitment file**. From the job's log:

```
=== MERGE: 16 shard(s) from .../runs/pr08-geom-tol/shards
      -> /valhalla/.../wam/configs/transfer25/pr08_geom_tol.json
wrote  /valhalla/.../wam/configs/transfer25/pr08_geom_tol.json
```

That file is exactly what the shard preflight and the merge precheck read as *the* pre-commitment.
Left as it is, the preflight compares HEAD's adapter against a measured artifact and **refuses
every shard**. The refusal is cheap — it fires before the decode — but it is sixteen wasted
launches.

**`git -C ${WAM} checkout --` cannot repair it.** `sync.sh:66` rsyncs with `--exclude '.git'`, so
the cluster copy has no git history to restore from. The rsync push *is* the repair.

Step 0 also carries the `GATE_QUALIFIED` flip and the post-`6a32143` adapter across, and stamps
`${PROJ}/wam/GIT_COMMIT`, which is what makes the run traceable at all.

---

## Step 1 — four waves of four

```bash
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 \
  sbatch --qos=ehpc-aif-2026pg01-905 --array=0-3%4 --time=01:45:00 \
  cluster/discoverer/103_measure_geom_tol.sbatch
```

…then `4-7`, `8-11`, `12-15`, one wave at a time. Cost per wave ≈ 2.9–3.9 GPU-h, ≈ 56–66 min
elapsed. **Total 13.64 GPU-h.**

**Why waves and not one array.** The project QoS carries two different limits. `MaxJobsPU=4` caps
*running* jobs and `%4` is the throttle that respects it. `MaxSubmitJobsPU=8` caps *submitted*
(pending + running) jobs, **every array task counts as one**, and `%4` does nothing for it —
`--array=0-15%4` is sixteen submissions and is rejected at `sbatch` time. Getting these the wrong
way round already cost job 189658.

**Between waves, check two things, not one.**

```bash
squeue -u "$USER" -r -h -o '%i' | wc -l          # must be <= 4 before the next wave
sacct -j <ARRAY_JOB_ID> -X --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls /valhalla/projects/ehpc-aif-2026pg01-905/runs/pr08-geom-tol-v2/shards/
```

**A wave can leave the queue by dying.** An empty queue is necessary, not sufficient — four more
`shard-N.json` must exist after each wave. `squeue | wc -l` also over-counts, because it sees every
job of yours including a peer session's.

### Three things that will cost you the wave

**`RUN_ID` goes on every line, including the merge.** Nothing in the script infers it. The default
is `pr08-geom-tol`, which is where the sixteen permanently uncommittable shards live.

**Do not copy-paste the commands the script prints.** Until they were repaired they omitted
`RUN_ID` entirely, so pasting the printed merge line pooled the *stale* partition; they also
rendered `$(basename "$0")` as the literal string `slurm_script`, which `sbatch` cannot open.
Use this page.

**Do not pass `GEOM_SECONDS_PER_FRAME`.** The shipped defaults `0.2478 s/frame` and
`GEOM_LOAD_SECONDS=410` are a least-squares fit over the previous array's own sixteen
(frames, wall-clock) pairs — `0.2478 × 171 625 + 16 × 410 = 49 089 s` against a measured
`49 091 s`, two seconds apart. `docs/investigations/2026-08-27-pr08-fronts/F2-item4-geom-tol.md`
still instructed `GEOM_SECONDS_PER_FRAME=0.29`; that override was correct against the *old* default
`0.18` and now double-charges the load, pushing shard 5 to 5646 s against 5398 s of wall so that
**the self-check refuses the wave.** The override that armed the check now disarms it. That file
carries a correction as of today.

### Why `01:45:00` and not `01:30:00`

Both fit. `01:30:00` leaves the heaviest shard **499 s** of slack on a self-check that now runs
*after* the preflight has imported torch and the adapter. `01:45:00` widens that to ~1400 s, stays
far under the `04:00:00` MaxWall, and **costs nothing** — GPU-hours are billed on runtime, not on
the request. The only argument for the tighter number is queue position under backfill.

Do **not** reach for `GEOM_ALLOW_TIGHT_WALL=1`, and do **not** set
`GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT`.

### Do not run `PILOT=1` first

No pilot artifact is needed: with none present the walltime self-check falls back to the shipped
constants and prints `[no pilot artifact]`. A pilot written *today* would carry a segmenter block
matching the committed contract, so the self-check would **trust** it and replace a sixteen-shard
fit with a three-episode extrapolation. Cost of that mistake: ≈ 0.3 GPU-h spent making the cost
model worse.

---

## Step 2 — merge

```bash
RUN_ID=pr08-geom-tol-v2 MERGE=1 NUM_SHARDS=16 \
  sbatch --qos=2cpu-single-host --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00 \
  cluster/discoverer/103_measure_geom_tol.sbatch
```

Cost: ≈ 20 s of two CPUs, **0 GPU-h**. Precondition: all sixteen `shard-*.json` under the **v2**
directory.

`RUN_ID` is load-bearing here above all: omit it and you merge the sixteen stale shards into a
document that looks finished.

**All four overrides are required together.** The free QoS caps at `cpu=2` and rejects a `--gres`,
while the file asks for 26 threads, 32 GB and one GPU. If the override is rejected, the fallback is
to submit `MERGE` with **no** overrides — it then runs under the project QoS and spends twenty
seconds of an H200 doing arithmetic, which is wasteful and correct. It is **not** correct to drop
`--qos` to get around it: that lands on `normal`, which is one minute and zero GPUs.

The merge writes `${WAM}/configs/transfer25/pr08_geom_tol.json` plus its `.sha256`, then copies
both into `${OUT}` so `--pull` can see them. `gate_qualified` is written as a **top-level JSON
boolean**; `spec_version`, `what_this_is` and the whole `segmenter` block are carried forward
verbatim from the pre-commitment.

---

## Step 3 — pull, diff, commit

```bash
./cluster/discoverer/sync.sh --pull pr08-geom-tol-v2
```

Then, on the workstation, **diff the pulled document against
`configs/transfer25/pr08_geom_tol.json` before committing anything.** `spec_version`,
`what_this_is` and the `segmenter` block MUST be unchanged. That diff is the check that Section 1
of the gate document survived Section 2 being written; the merge asserts it, and this is where a
human confirms it.

Commit the artifact and its `.sha256` together. Then re-run `102_stage_sam2_weights.sbatch`.

**After a successful merge the cluster's tracked config path is a *measured* artifact, not the
pristine pre-commitment.** Any shard re-run afterwards compares against that file, so re-run
`sync.sh` first.

---

## Step 4 — what happens next, and it is not a submission

```bash
# EST_DRIFT, local, ~4 min on the RTX 5090, 0 GPU-h of allocation.
# MUST follow the merge: measure_est_drift reads the GEOM_TOL document at MEASURE time.
.venv/bin/python scripts/measure_est_drift.py measure ... --out <artifact>

# The carry. --est-drift-arm IS MANDATORY for a two-arm artifact.
.venv/bin/python scripts/measure_geom_tol.py \
    --carry-est-drift <artifact> \
    --est-drift-arm per_frame
```

**→ PR-08 §8 item 4 closes iff the margin comes out positive.** Against the previous
(now-discarded) GEOM_TOL it was `0.18780929757736792 px`; whether the re-measured number reproduces
`0.4786` is **UNKNOWN** and is the whole reason this array is running.

Item 3 (throughput) is independent and still open.

---

## Unknowns, stated as unknowns

The cluster filesystem was **not** inspected — no `ssh`, `sbatch`, `squeue` or `sacct` was run. The
provider forbids AI coding agents on `login-plus`, and a violation can terminate this allocation.
Everything above is derived from the repository and from the pulled artifacts and Slurm logs under
`runs/`, a snapshot of ~2026-08-24.

- **Whether `${PROJ}/runs/pr08-geom-tol-v2` is already taken.** `ls ${PROJ}/runs` before submitting;
  a colliding name reintroduces exactly the stale-artifact problem the fresh `RUN_ID` exists for.
- **Whether the cluster's `configs/transfer25/pr08_geom_tol.json` is still 190191's overwrite.** A
  peer session may already have re-synced. Run `sync.sh` regardless.
- **`MaxSubmitJobsPU=8`.** `docs/discoverer.md` records only `Jobs/user 4` for the project QoS. The
  value 8 is asserted in the sbatch header, not verified against `sacctmgr` here — and whether a
  pending `2cpu-single-host` merge consumes a project-QoS submit slot is also unverified.
- **Whether `--gres=none` is accepted by this Slurm build.** The fallback is documented in step 2.
- **The true per-frame cost of the HEAD adapter.** `0.2478` is fit over sixteen shards of which
  twelve ran a pre-`6a32143` adapter. Nothing has measured the post-`6a32143` rate over a full
  shard. The 1.27× margin over the worst observed marginal rate absorbs it, but **watch wave 1's
  actual wall clocks before submitting wave 2.**
- **Whether `${SOURCE}/manifest.json` still carries `episodes[].frames`.** If it does not, the
  self-check prints `NO SELF-CHECK` and **does not stop the run** — the one path on which a
  wall-clock death is still possible. The shard artifacts prove it was intact on 2026-08-24.
