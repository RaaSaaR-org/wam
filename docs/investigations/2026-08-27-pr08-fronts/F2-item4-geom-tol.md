# FRONT 2 — PR-08 §8 item 4, first half: GEOM_TOL re-measured at HEAD

Repo: `/home/humanoid/develop/wam` — HEAD `19826ccae06f36cd6153125042eda3e7d2a7560b`, working tree clean
(`git status --short` empty at the time of this run). Everything below is either MEASURED locally
today, quoted from a committed artifact, or labelled as inference.

**Headline: the refusal claim is TRUE and was reproduced locally (exit 2, nothing written), but it
is the SECOND reason, not the first — and the cost estimate in the task is ~50 % low. The measured
cost of the array that produced 0.47857992441961017 px was 13.636 GPU-h, not ~9.**

---

## 0. The three-line answer

| Question | Answer | Evidence class |
|---|---|---|
| Would a re-merge of the 16 existing shards refuse? | **YES — exit 2, nothing written.** Reproduced locally. | (a) MEASURED |
| Is that why 0.4786 px can never be committed? | Only partly. `gate_qualified: false` is baked into all 16 shards and no merge can lift it. | (a) MEASURED + (b) artifact |
| Cheaper path than the full re-run? | **No.** No contract-forward flag, no force flag, no migration path. | (a) MEASURED (argparse enumerated) |
| Cost of the re-run at HEAD | **13.64 GPU-h measured from the previous array's own Slurm logs**, +~20 s merge. Not 9.115. | (a) MEASURED |

---

## 1. Is the refusal claim TRUE? — YES, and here is the proof from the code and from a live run

### 1.1 `contract_disagreements()` — absence IS a disagreement

`/home/humanoid/develop/wam/scripts/measure_geom_tol.py:1913-1932`:

```python
def contract_disagreements(ours: Mapping, theirs: Mapping) -> list[dict[str, Any]]:
    """...
    A field present on one side and absent on the other counts as a disagreement. Absence is not
    agreement anywhere else in this cross-check and it is not here: a contract that has grown a
    field the committed one never had is, precisely, a segmenter the committed one did not describe.
    """
    out: list[dict[str, Any]] = []
    for key in sorted(set(ours) | set(theirs)):
        mine, yours = _canonical(ours.get(key)), _canonical(theirs.get(key))
        if mine != yours:
            out.append({"field": key, "geom_tol": yours, "this_run": mine})
    return out
```

The union `set(ours) | set(theirs)` plus `.get(key)` (→ `None` on the absent side) is the whole
mechanism. There is no `if key in both` guard anywhere.

### 1.2 `merge_committed_contract()` — a disagreement raises, and every caller turns that into exit 2

`scripts/measure_geom_tol.py:2035-2200`. The two load-bearing pieces:

`:2109-2129` —
```python
    disagreements = contract_disagreements(ours, theirs)
    if disagreements:
        lines = [ ... ]
        raise MethodUnavailable("".join(lines))
```

Caller on the merge path, `merge_main()`, `scripts/measure_geom_tol.py:3369-3376`:
```python
    try:
        refuse_default_out_without_contract(args.out)
        carried = merge_committed_contract(args.out, record)
    except MethodUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FATAL
```
`EXIT_FATAL = 2` (`scripts/measure_geom_tol.py:396`). `write_artifact()` is called at `:3384`,
*after* that block — so the raise happens strictly before any byte is written.

### 1.3 The two sides, measured

MEASURED today (`/home/humanoid/develop/wam/.venv/bin/python`, read-only):

* committed contract, `configs/transfer25/pr08_geom_tol.json` → `segmenter` block **contains**
  `"mask_validity_reference_max_frame_fraction": 0.1`
* every shard, `runs/pr08-geom-tol/shards/shard-*.json` → `mask_method.params.segmenter`
  **does not contain** it.

```
shard-vs-committed disagreeing fields: ['mask_validity_reference_max_frame_fraction']
shard-vs-module   disagreeing fields: ['mask_validity_reference_max_frame_fraction']
module-vs-committed disagreeing fields: []          <-- HEAD adapter and HEAD config AGREE
```

The shards also carry the pre-`e518a84` `ESTIMATOR_VERSION` (`runs/pr08-geom-tol/shards/shard-0.json`,
`mask_method.version`), ending `...;prop=per_frame;mask_val_min_iou=0.1` with **no**
`mask_val_ref_max_frac`. HEAD's `apple_sam2.ESTIMATOR_VERSION` ends
`...;mask_val_min_iou=0.1;mask_val_ref_max_frac=0.1`.

### 1.4 DEMONSTRATED — the real merge, run locally, against a scratch copy of the committed contract

```
cp configs/transfer25/pr08_geom_tol.json $SCRATCH/remerge_out.json
.venv/bin/python scripts/measure_geom_tol.py \
    --merge runs/pr08-geom-tol/shards/shard-*.json \
    --out $SCRATCH/remerge_out.json
```

**EXIT=2**, stderr verbatim:

```
FATAL: this run's segmenter disagrees with the contract committed at .../remerge_out.json (segmenter):
         mask_validity_reference_max_frame_fraction: committed 0.1, this run None
       That contract was committed BEFORE the measurement precisely so this comparison could be
       made. ...
       Nothing is written. Either run the segmenter the contract describes, or change the contract as
       a reviewed commit of its own, before the measurement and never after it.
```

`sha256` of the scratch `--out` **after** the run is byte-identical to
`configs/transfer25/pr08_geom_tol.json` (`06416b548242d34f...`) — nothing was written, as claimed.

### 1.5 The 37-minute sequencing, confirmed to the minute

* merge job 190191 stamped its artifact `measured_utc: 2026-08-24T16:16:47+00:00`
  (`runs/pr08-geom-tol/pr08_geom_tol.json`), i.e. **18:16:47 +02:00**.
* `git show -s --format=%cI e518a84` → **2026-08-24T18:53:16+02:00**.
* Delta = **36 min 29 s**. The claim's "37 minutes AFTER" is correct.

`e518a84`'s own commit message says so in as many words, and states the sequencing reason V10 gave
for not landing it earlier: *"moving mask_validity_reference_max_frame_fraction into
SEGMENTER_CONTRACT while the array was in flight would have disqualified every landed shard
mid-run. The array merged, the sequencing reason is spent."*

**VERDICT: the refusal claim is CONFIRMED, by code reading and by a local run of the real code path.**

### 1.6 …but the refusal is the SECOND reason, not the first

The claim's conclusion ("therefore … CAN NEVER BE COMMITTED") is right, but it would be right even
if `e518a84` had never happened. See §2. `e518a84`'s message says the same thing:
*"GEOM_TOL = 0.4786 px was measured by an instrument that is no longer HEAD, which is a second and
independent reason it may not be committed, **on top of gate_qualified: false**."*

---

## 2. `gate_qualified` IS baked into every shard at measurement time — the write sites

Three lines, all in the measuring path of `scripts/measure_geom_tol.py`, all executed before the
artifact is written:

1. **The adapter's flag is read once, at method resolution.** `scripts/measure_geom_tol.py:1044`:
   ```python
       gate_qualified=declared_gate and bool(checkpoints) and contract is not None,
   ```
   with `declared_gate` from `:996`:
   ```python
       declared_gate = bool(getattr(module, "GATE_QUALIFIED", False))
   ```

2. **It is stamped into the artifact's `mask_method` block.** `scripts/measure_geom_tol.py:3990`:
   ```python
           "gate_qualified": method.gate_qualified,
   ```

3. **And into the artifact's top-level verdict.** `scripts/measure_geom_tol.py:3849` and `:4017`:
   ```python
       gate_ok = bool(headline_valid and method.gate_qualified and not partial_reasons)
   ...
           "gate_qualified": gate_ok,
   ```

The merge does **not** re-derive it from the adapter — it reads it back out of the shard artifacts.
`scripts/measure_geom_tol.py:2982` and `:2996`:
```python
    method_gate_ok = bool((template.get("mask_method") or {}).get("gate_qualified"))
...
    gate_ok = bool(headline_valid and method_gate_ok and not partial and not reasons)
```
(`template = dict(by_index[0])`, i.e. shard 0's own JSON — `:2975`.)

**At HEAD the flag is still False.** `scripts/estimators/apple_sam2.py:938`:
```python
GATE_QUALIFIED = False
```
even though `GATE_QUALIFICATION_BLOCKERS` is now the empty tuple (`apple_sam2.py:635`). MEASURED by
import: `GATE_QUALIFIED = False`, `BLOCKERS = ()`. The module says why, `apple_sam2.py:637-648`:
*"**EMPTY IS NOT PERMISSION, AND THE TUPLE BEING EMPTY IS NOT WHY.** … `GATE_QUALIFIED` IS STILL
`False` … the flag has TWO preconditions and an empty blocker tuple satisfies one of them. The other
is a recorded decision on the residue the 2026-08-26 entries carry forward."*

**CONSEQUENCE (measured, not inferred): a full 16-shard re-measure submitted today, at HEAD, with a
perfectly synced cluster copy, produces sixteen shards each stamped `gate_qualified: false` and a
merged artifact stamped `gate_qualified: false`, exit 3 — another artifact that MUST NOT be
committed, at the full 13.6 GPU-h.** The sbatch file already says this in its header
(`cluster/discoverer/103_measure_geom_tol.sbatch:170-181`) and prints it at
`:485-487`: *"discharging it means re-measuring the corpus, because gate_qualified is baked into
each shard at measurement time."* The previous run's own log proves the outcome:
`runs/_slurm_logs/geom-tol.190191_4294967294.out` → `=== measure_geom_tol.py --merge exited 3`
with seventeen `is not gate-qualified` reasons.

**So the re-run is only worth paying for AFTER the owner flips `GATE_QUALIFIED`. Ordering matters
more than the contract question.**

---

## 3. Is there a cheaper path than the full 16-shard re-run?

**No. Plainly no.** Three independent checks:

### 3.1 There is no contract-forward flag, and no force flag

Complete enumeration of `measure_geom_tol.py`'s CLI (MEASURED —
`grep -o '"--[a-z-]*"' scripts/measure_geom_tol.py | sort -u`):

```
--camera-key --carry-est-drift --corpus --decoder --dump-displacements --hist-bin-px
--limit --masks --max-frames --merge --method --min-area-px --min-coverage
--num-shards --out --shard --step-frames
```

No `--force`, no `--allow-contract-drift`, no `--migrate-contract`, no `--contract`. The refusal
text itself names the only sanctioned escape and it is not an escape:
*"change the contract as a reviewed commit of its own, **before the measurement and never after
it**"* (`:2121-2128`).

### 3.2 A "contract migration" would work mechanically and is forbidden

The merge compares the **shard JSON's** `mask_method.params.segmenter` against the **document at
`--out`** — the module is not consulted at merge time (`:2094-2109`). So reverting
`configs/transfer25/pr08_geom_tol.json`'s `segmenter` block to the pre-`e518a84` shape would let the
re-merge through. That is precisely "a gate rewritten after seeing its output", forbidden by
`docs/handoff.md` §3 and by the task's own prohibitions, **and it would not help anyway** because of
§3.3.

### 3.3 Even a successful re-merge yields `gate_qualified: false`

`method_gate_ok` comes from shard 0's stamped JSON (§2). It is `false` in all sixteen files
(MEASURED: `shard-0.json` → `gate_qualified: false`, `mask_method.gate_qualified: false`). The merge
would write the same 0.47857992441961017 px, disqualified, and exit 3. Nothing in the merge path can
raise it.

### 3.4 And the instrument is not HEAD

`e518a84`'s message records that six of the sixteen shards ran hours after `6a32143` landed and
still carried the pre-`6a32143` adapter, because the cluster copy had not been re-synced. That is an
AC-04 traceability defect on top of everything else, and it is also why all sixteen shards record
`git_commit: null` (`runs/pr08-geom-tol/shards/shard-0.json` → `git_commit: None`;
`runs/pr08-geom-tol/pr08_geom_tol.json` → `git_commit: null`). `e518a84` fixed the *reader*
(`measure_geom_tol._git_commit()`, `:2268-2308`, now falling back to the `GIT_COMMIT` file
`cluster/discoverer/sync.sh:78` writes) — but only a **new run** can benefit.

**ANSWER: the corpus must be re-measured. There is no re-merge, no migration and no flag that
produces a committable number from what is on disk.**

---

## 4. Exact re-run command sequence at HEAD — DO NOT SUBMIT

> **PRECONDITION, and it is not optional: `scripts/estimators/apple_sam2.py:938` must read
> `GATE_QUALIFIED = True` in the *pushed* commit before the array goes in.** Otherwise this is
> 13.6 GPU-h for another disqualified artifact. That flip is the project owner's call
> (`apple_sam2.py:637-648`, and CLAUDE.md's standing rule). **This is the single blocking fact.**

Second precondition: the cluster's `${PROJ}/wam/configs/transfer25/pr08_geom_tol.json` is currently
the **merged, disqualified artifact** that job 190191 wrote over the pre-commitment
(`runs/_slurm_logs/geom-tol.190191_4294967294.out`: `wrote /valhalla/.../wam/configs/transfer25/pr08_geom_tol.json`),
and its top-level `segmenter` block **lacks** `mask_validity_reference_max_frame_fraction`
(MEASURED on the pulled copy `runs/pr08-geom-tol/pr08_geom_tol.json`). A merge against that file
refuses in the *opposite* direction (`committed None, this run 0.1`). A repo sync repairs it;
`git -C ${WAM} checkout --` does **not** (see Defect D-2).

```bash
# 0. FROM THE WORKSTATION, by a human operator. Pushes HEAD and stamps ${PROJ}/wam/GIT_COMMIT.
#    This is what makes the re-run traceable at all and what repairs the overwritten contract.
./cluster/discoverer/sync.sh

# 1. Fresh RUN_ID so the 16 stale shard artifacts cannot be reused or overwritten in place.
#    (Do NOT reuse pr08-geom-tol: see Defect D-1. `mv` inside /valhalla is a hard prohibition.)
#    Four waves of four; MaxSubmitJobsPU=8 counts every array task, %4 does not help the submit limit.
#    GEOM_SECONDS_PER_FRAME=0.29 is the MEASURED rate (§5, Defect D-3); the shipped default 0.18
#    disarms the walltime self-check.
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 GEOM_SECONDS_PER_FRAME=0.29 \
  sbatch --array=0-3%4  --time=01:50:00 cluster/discoverer/103_measure_geom_tol.sbatch
#   ... wait until `squeue -u $USER -r -h -o '%i' | wc -l` is <= 4, then:
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 GEOM_SECONDS_PER_FRAME=0.29 \
  sbatch --array=4-7%4  --time=01:50:00 cluster/discoverer/103_measure_geom_tol.sbatch
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 GEOM_SECONDS_PER_FRAME=0.29 \
  sbatch --array=8-11%4 --time=01:50:00 cluster/discoverer/103_measure_geom_tol.sbatch
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 GEOM_SECONDS_PER_FRAME=0.29 \
  sbatch --array=12-15%4 --time=01:50:00 cluster/discoverer/103_measure_geom_tol.sbatch

# 2. MERGE. No GPU is used; it belongs on the free QoS, whose limits must be on the command line.
RUN_ID=pr08-geom-tol-v2 MERGE=1 NUM_SHARDS=16 \
  sbatch --qos=2cpu-single-host --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00 \
  cluster/discoverer/103_measure_geom_tol.sbatch

# 3. Pull, then commit configs/transfer25/pr08_geom_tol.json AND its .sha256 from the workstation.
./cluster/discoverer/sync.sh --pull pr08-geom-tol-v2
```

### Cost, MEASURED from the previous array's own Slurm logs

`runs/_slurm_logs/geom-tol.189935_{0..3}.out`, `189971_{4..7}`, `189984_{8,9}`, `190125_{10..15}`,
line `=== shard N exited 3 after Ts`:

| shard | frames | wall s | | shard | frames | wall s |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 10008 | 3192 | | 8 | 8994 | 2742 |
| 1 | 5159 | 1630 | | 9 | 11157 | 3358 |
| 2 | 11860 | 3626 | | 10 | 12686 | 3378 |
| 3 | 9230 | 2825 | | 11 | 11440 | 3065 |
| 4 | 11669 | 3160 | | 12 | 12002 | 3270 |
| 5 | 14162 | 3741 | | 13 | 8419 | 2308 |
| 6 | 13372 | 3933 | | 14 | 10848 | 2911 |
| 7 | 10836 | 3369 | | 15 | 9783 | 2583 |

* **Σ wall = 49 091 s = 13.636 GPU-h** (16 × H200, `--gres=gpu:1`).
* Least-squares fit over all 16: **p = 0.2478 s/frame, L = 410 s fixed load** (max residual 302 s,
  RMS 196 s). Per-shard `t/frames` ranges **0.2640 – 0.3189**, mean **0.2879**.
* Heaviest shard actually took **3741 s = 62.4 min** (shard 5, 14 162 frames). Against
  `--time=01:50:00` (6600 s) that is 1.76×; against `--time=01:30:00` (5400 s) it is 1.44×.
  Waves 2–4 were in fact submitted at 01:50:00 (`walltime remaining 6598-6600s` in the logs) with
  `GEOM_SECONDS_PER_FRAME=0.31`; only wave 1 used 01:30:00 / 0.18.
* **Merge: no GPU, ~20 s of CPU.** (Documented in the sbatch header, `:38-39`.)

**Estimated re-run cost at HEAD: 13.6 GPU-h ± the fit residual, i.e. ~13.5–14 GPU-h. The task's
"~9 GPU-h" comes from the sbatch header's N=16 planning table
(`103_measure_geom_tol.sbatch:126`, "9.115"), which was built on p = 0.18 and is 1.5× optimistic
against the measured rate.**

---

## 5. Defects that would waste the ~13.6 GPU-h

### D-1 (SEVERE, would burn the entire array) — a `FORCE=1` re-measure into the existing shard directory refuses AFTER the GPU work, writing nothing

`merge_committed_contract()` is called on the **measuring** path too, with `--out` = the shard path
(`scripts/measure_geom_tol.py:4134`), and it runs *after* the corpus is decoded — the docstring's
"before a byte is written" means before the artifact write, not before the measurement.
`committed_segmenter_contract()` falls back to `mask_method.params.segmenter` (`:1892-1897`), so an
**existing `shard-N.json` from the old array acts as a committed contract**.

DEMONSTRATED locally against a copy of `runs/pr08-geom-tol/shards/shard-7.json`, with a record
carrying HEAD's `apple_sam2.SEGMENTER_CONTRACT`:

```
REFUSED (-> EXIT_FATAL = 2 )
FATAL: this run's segmenter disagrees with the contract committed at .../shard-7-stale.json
       (mask_method.params.segmenter):
         mask_validity_reference_max_frame_fraction: committed None, this run 0.1
...
file unchanged: True
```

Failure scenario, concretely: operator syncs HEAD, submits with `FORCE=1` into
`RUN_ID=pr08-geom-tol`. Each task decodes its ~11 000 frames (~50 min of H200), then exits 2 with
nothing written. Sixteen tasks × ~50 min ≈ **13.6 GPU-h destroyed**, and the message blames the
operator's segmenter. Without `FORCE=1` it is not better, only cheaper: `shard_artifact_landed()`
(`103_measure_geom_tol.sbatch:762-806`) declares the stale shards reusable, every task prints
"already landed. Skipping." and exits 0, and the merge then refuses — so the array silently
"succeeds" and produces nothing.

**Mitigation without a source change: a fresh `RUN_ID` (used in §4).** Proposed source fix, NOT
applied:

```diff
--- a/cluster/discoverer/103_measure_geom_tol.sbatch
+++ b/cluster/discoverer/103_measure_geom_tol.sbatch
@@ -813,6 +813,15 @@
   if [[ -f "${SHARD_OUT}" && "${FORCE:-0}" == "0" ]]; then
     if shard_artifact_landed "${SHARD_OUT}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${GEOM_STEP_FRAMES}"
     then
       echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} already landed. Skipping."
       echo "===   FORCE=1 re-measures it."
       exit 0
     fi
     echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} exists and is NOT reusable (above)."
     echo "===   Re-measuring. The old file is overwritten, so no stale shard can reach the merge."
   fi
+  # AND IT IS REMOVED FIRST, NOT MERELY OVERWRITTEN. measure_geom_tol.merge_committed_contract()
+  # treats ANY document at --out that carries a segmenter block as a committed contract, and a
+  # shard artifact carries one at mask_method.params.segmenter. A stale shard from an array that
+  # ran an older SEGMENTER_CONTRACT therefore refuses the new measurement with exit 2 AFTER the
+  # whole partition has been decoded -- ~50 min of H200 per task, sixteen of them, nothing written.
+  # Deleting it costs nothing: a shard that is being re-measured is by definition not wanted.
+  if [[ -f "${SHARD_OUT}" ]]; then
+    echo "===   removing the stale ${SHARD_OUT} before measuring (it would be read as a contract)"
+    rm -f "${SHARD_OUT}" "${SHARD_OUT}.sha256"
+  fi
```

### D-2 (MODERATE, breaks the documented recovery) — the printed recovery command cannot run on the cluster

`103_measure_geom_tol.sbatch:457` and `:489` print:
```
       Update the checkout: git -C ${WAM} checkout -- configs/transfer25/pr08_geom_tol.json
```
`cluster/discoverer/sync.sh:67` rsyncs with `--exclude '.git'`, and
`measure_geom_tol._git_commit()`'s docstring (`:2271-2278`) says so explicitly: *"There is no
`.git` beside it, so `git rev-parse` fails there."* The command fails with `not a git repository`
at exactly the moment the pre-commitment has been overwritten by a disqualified merge. The correct
recovery is a re-sync from the workstation. Proposed fix, NOT applied:

```diff
--- a/cluster/discoverer/103_measure_geom_tol.sbatch
+++ b/cluster/discoverer/103_measure_geom_tol.sbatch
-    echo "       Update the checkout: git -C ${WAM} checkout -- configs/transfer25/pr08_geom_tol.json"
+    echo "       ${WAM} is an rsync target with NO .git (sync.sh --exclude '.git'), so there is no"
+    echo "       checkout to do here. Restore it FROM THE WORKSTATION:  ./cluster/discoverer/sync.sh"
```
(same substitution at `:489`).

### D-3 (MODERATE, disarms the check that exists to prevent the 189658 loss) — `GEOM_SECONDS_PER_FRAME` defaults to 0.18 against a measured 0.29

`103_measure_geom_tol.sbatch:861`: `GEOM_SECONDS_PER_FRAME=${GEOM_SECONDS_PER_FRAME:-0.18}`.
MEASURED per-shard rate over the whole previous array: **0.2640 – 0.3189 s/frame, mean 0.2879**;
least-squares p = 0.2478 with L = 410 s (the shipped `GEOM_LOAD_SECONDS=120` is also ~3.4× low).
The operator already had to pass `GEOM_SECONDS_PER_FRAME=0.31` by hand for waves 2–4 — visible in
`runs/_slurm_logs/geom-tol.189971_5.out`: `cost model: p=0.3100 s/frame`. At the shipped default the
self-check estimates the heaviest shard at 44.5 min when the truth is 62.4 min, i.e. it would pass a
`--time=01:00:00` request that would die at the wall. That is exactly the class of failure the
self-check was written for after job 189658. Proposed fix, NOT applied:

```diff
-  GEOM_SECONDS_PER_FRAME=${GEOM_SECONDS_PER_FRAME:-0.18}
-  GEOM_LOAD_SECONDS=${GEOM_LOAD_SECONDS:-120}
+  # MEASURED 2026-08-23/24 by the 16-shard array itself (jobs 189935/189971/189984/190125): the
+  # per-shard rate ran 0.2640-0.3189 s/frame (mean 0.2879) and a least-squares fit over all
+  # sixteen gives p = 0.2478 with a fixed load of 410 s. 0.18 was a planning constant over job
+  # 189658's >= 0.166 floor and is 1.6x optimistic; 120 s was 3.4x low. A self-check built on
+  # those numbers passes a request that dies at the wall, which is the failure it exists to stop.
+  GEOM_SECONDS_PER_FRAME=${GEOM_SECONDS_PER_FRAME:-0.29}
+  GEOM_LOAD_SECONDS=${GEOM_LOAD_SECONDS:-410}
```
The N=16 planning table at `:124-135` and the "9.115" GPU-h total should be restated against the
measured 13.636 GPU-h in the same change.

### D-4 (MODERATE, the failure that already happened once) — nothing compares the adapter against the committed contract BEFORE the GPU work

On the shard path, `merge_committed_contract()` compares against `SHARD_OUT`, never against
`${WAM}/configs/transfer25/pr08_geom_tol.json`. So a stale cluster copy of
`scripts/estimators/apple_sam2.py` produces sixteen shards that agree perfectly *with each other*
and only fail at the merge — which is precisely what `e518a84` reports happened
(*"including the six that ran today, hours after that commit landed, because the cluster copy had
not been re-synced"*). The MERGE branch already has a preflight (`:434-449`) but it only checks the
contract is **present**, not that it **agrees**. Proposed fix, NOT applied — insert in the SHARD
branch before the walltime self-check (~`:840`):

```diff
+  # THE CONTRACT PREFLIGHT, BEFORE THE GPU AND NOT AT THE MERGE. measure_geom_tol.py compares the
+  # adapter's SEGMENTER_CONTRACT against the committed one only when it writes the tracked path,
+  # i.e. at the merge -- an array's worth of GPU-hours later. A cluster copy that lags the pushed
+  # commit therefore produces sixteen internally consistent shards that the merge then refuses.
+  # That is not hypothetical: six shards of the 2026-08-23/24 array ran a pre-6a32143 adapter
+  # hours after it landed, for exactly this reason (commit e518a84).
+  if ! CONTRACT_FILE="${WAM}/configs/transfer25/pr08_geom_tol.json" python - <<'PY'
+import json, os, sys
+sys.path.insert(0, os.path.join(os.environ["WAM"], "scripts"))
+from estimators.apple_sam2 import SEGMENTER_CONTRACT, GATE_QUALIFIED
+doc = json.loads(open(os.environ["CONTRACT_FILE"]).read())
+params = (doc.get("mask_method") or {}).get("params") or {}
+theirs = doc.get("segmenter") or params.get("segmenter")
+ours = json.loads(json.dumps(dict(SEGMENTER_CONTRACT), default=str))
+if not isinstance(theirs, dict):
+    print("    no committed segmenter block"); raise SystemExit(1)
+diff = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
+if diff:
+    print("    adapter vs committed contract disagree on: " + ", ".join(diff))
+    for k in diff:
+        print("      %s: committed %r, this adapter %r" % (k, theirs.get(k), ours.get(k)))
+    raise SystemExit(1)
+print("    adapter agrees with the committed contract field for field; "
+      "GATE_QUALIFIED=%s" % GATE_QUALIFIED)
+PY
+  then
+    echo "FATAL: this shard would measure with a segmenter the committed contract does not describe."
+    echo "       measure_geom_tol.py --merge would refuse the whole partition (exit 2, nothing"
+    echo "       written) AFTER every shard had been paid for. Re-sync the tree from the"
+    echo "       workstation (./cluster/discoverer/sync.sh) and re-submit. Nothing was measured."
+    exit 1
+  fi
```

### D-5 (LOW, cosmetic but misleading) — the refusal labels the two sides "committed" / "this run" by position, not by role

On the shard path the *stale artifact* is printed as "committed" and HEAD as "this run"
(`mask_validity_reference_max_frame_fraction: committed None, this run 0.1`), which reads as though
HEAD's adapter is the deviation. Source: the dict keys `"geom_tol"` / `"this_run"` in
`contract_disagreements()` (`scripts/measure_geom_tol.py:1929`) are rendered at `:2116-2118`. Not
worth a fix on its own; worth knowing when reading the D-1 refusal.

### Checked and NOT defective

* `--merge "${SHARD_DIR}"` passing a directory: supported and safe — `collect_shard_records()`
  (`:2405-2450`) expands directories and **skips** non-`wam.geom_tol_shard/1` files with a stderr
  line, while an explicitly named path is never skipped. The previous merge log shows it picking up
  exactly 16 and ignoring the pilot artifacts in the same directory.
* The exit-3 re-classification (`:478-491`, `:979-1001` of the sbatch): correct, and it is what
  transports the disqualified artifact for `sync.sh --pull`. It keeps the exit code at 3.
* The bash `if ...; then RC=0; else RC=$?; fi` idiom under `set -e` (`:465-469`, `:1002-1009`):
  correct, and commented as such.
* `SLURM_ARRAY_TASK_ID` / `NUM_SHARDS` / `GEOM_STEP_FRAMES` validation (`:682-736`): thorough,
  including the string-before-arithmetic check and the no-apostrophes-in-`${var:?}` rule.
* The merge's completeness refusals (`:3054` reasons list, `merge_shard_records`): a missing shard,
  a duplicate, a `num_shards` disagreement, a corpus that is not covered exactly once — all refuse.

---

## 6. What I could not measure here

* **NOT MEASURED — requires a cluster run:** the actual per-frame rate of the HEAD adapter. The
  0.2879 s/frame mean is from the *pre-`e518a84`* adapter. `mask_validity_reference_max_frame_fraction`
  is a cheap per-frame check, so the rate should be within noise of it, but that is (c) inference.
* **NOT MEASURED — requires a cluster run:** whether the cluster copy is currently synced to
  `19826cc`. `${PROJ}/wam/GIT_COMMIT` can only be read there, and reading it is a cluster action.
* **NOT MEASURED — requires the project owner:** whether `GATE_QUALIFIED` may flip. This is the one
  blocking fact for the whole front.

---

## Adversarial re-read

Independent re-check by a second session, 2026-08-27, repo `/home/humanoid/develop/wam` at HEAD
`19826ccae06f36cd6153125042eda3e7d2a7560b`, working tree clean (`git status --short` empty). No file
under the repo was modified; everything below is either (a) MEASURED by me now, (b) quoted from a
committed artifact, or (c) labelled inference.

**Bottom line: the four headline claims survive independent re-measurement, exactly.** I reproduced
the exit-2 refusal, the byte-identical `--out`, the `gate_qualified` bake-in, the CLI enumeration,
the 36 min 29 s sequencing, and the 49 091 s / 13.636 GPU-h total to the second. Six supporting
claims need correction; one of them is a factual error against an artifact, and one is an
operational instruction that would be rejected at `sbatch` time if followed verbatim. None of the
six overturns the front's conclusion — the largest one makes D-4 stronger, not weaker.

### What I re-measured and CONFIRMED

| Claim | My independent result |
|---|---|
| `--merge` of the 16 shards against a scratch copy of the committed contract exits 2 | **CONFIRMED.** `EXIT=2`, stderr `mask_validity_reference_max_frame_fraction: committed 0.1, this run None` |
| `--out` unchanged, sha256 `06416b548242d34f…` | **CONFIRMED.** `06416b548242d34f3067a41ca5bb05e3295fb48a402c310b80fe49637625d2c8` before and after, identical to `configs/transfer25/pr08_geom_tol.json` |
| `contract_disagreements()` at `:1913`, union + `.get` | **CONFIRMED** verbatim at `scripts/measure_geom_tol.py:1913-1932` |
| `EXIT_FATAL = 2` at `:396`; raise at `:2109`; `merge_main` catch at `:3369-3374`; `write_artifact` at `:3384` | **CONFIRMED** — all four line numbers exact |
| gate write sites `:996`, `:1044`, `:3849`, `:3990`, `:4017`; merge re-read `:2982`, `:2996` | **CONFIRMED** — `grep -n` returns exactly those seven lines |
| `GATE_QUALIFIED = False` at `apple_sam2.py:938`, `GATE_QUALIFICATION_BLOCKERS = ()` at `:635`, the "EMPTY IS NOT PERMISSION" block at `:637-648` | **CONFIRMED** by `sed` and by import (`GATE_QUALIFIED False BLOCKERS ()`) |
| All 16 shards `gate_qualified: false`, `mask_method.gate_qualified: false`, `git_commit: null` | **CONFIRMED** — 16/16, read from the JSON |
| Merge `measured_utc 2026-08-24T16:16:47+00:00`; `e518a84` `%cI` = `2026-08-24T18:53:16+02:00`; Δ = 36 min 29 s | **CONFIRMED.** I also checked the merge's stamp is its OWN and not shard 0's template: shard 15 is `15:34:38Z`, the merge `16:16:47Z`. The delta is real. |
| CLI has no `--force`, no `--contract`, no migration flag | **CONFIRMED** by `argparse` itself (17 `add_argument` calls, `--help` enumerated), not by the deliverable's `grep`, which could have missed a flag with a digit or a capital. It did not. |
| Σ wall = 49 091 s = 13.636 GPU-h; fit `p = 0.2478`, `L = 410.2`, max residual 302 s, RMS 196 s; rates 0.2640–0.3189, mean 0.2879 | **CONFIRMED to four decimals** by my own `numpy.linalg.lstsq` over the 16 log lines. The frame/wall table in §4 matches the artifacts and the logs row for row. |
| Heaviest shard 3741 s = 62.4 min; at `p=0.18, L=120` it estimates 2669 s = 44.5 min | **CONFIRMED** (`14162*0.18+120 = 2669.16`) |
| sbatch's 9.115 GPU-h is `p=0.18` arithmetic | **CONFIRMED and reconstructed**: `171625*0.18/3600 = 8.581` h `+ 16*120/3600 = 0.533` h `= 9.115` h. Ratio `13.636/9.115 = 1.496`. |
| module-vs-committed contract diffs = `[]`; shard-vs-committed and shard-vs-module = `['mask_validity_reference_max_frame_fraction']` | **CONFIRMED** by import + JSON compare |
| D-1's mechanism: a stale shard at `--out` is read as a committed contract and refuses AFTER the measurement | **CONFIRMED.** I re-ran it myself against a fresh copy of `shard-7.json` with HEAD's `SEGMENTER_CONTRACT`: `REFUSED -> exit 2 … committed None, this run 0.1`, `file unchanged: True`. `shard-0.json` has **no** top-level `segmenter`, so `committed_segmenter_contract()` (`:1874-1899`) falls through to `mask_method.params.segmenter` exactly as claimed. |
| D-1's no-FORCE variant: `shard_artifact_landed` (`:762-806`) declares stale shards reusable | **CONFIRMED by reading it.** It checks `schema`, `shard.index`, `shard.num_shards`, `step_frames`, the reason strings, `headline_valid`, `partial_measurement`, `limit`/`max_frames`, `n_steps_measured` — **and nothing about the segmenter contract.** A stale shard passes. |
| D-2: `git -C ${WAM} checkout --` printed at sbatch `:457` and `:489` | **CONFIRMED**, both line numbers exact (`grep -n "checkout -- configs/transfer25"`) |
| D-4: the MERGE preflight at `:434-449` checks presence only | **CONFIRMED** — its heredoc only `print`s `method_name`, `box_threshold`, `pixel_grid_hw`; it never compares |
| The "Checked and NOT defective" list (`collect_shard_records` `:2405`, the exit-3 reclassification, the `set -e` idiom, the completeness reasons at `:3050+`) | **CONFIRMED** — spot-checked all four; the citations land where claimed |
| §4's MERGE command line | **CONFIRMED** — it is character-for-character the one the sbatch header prescribes at `:23-24` (`--qos=2cpu-single-host --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00`) |

### R-1 — REFUTED (factual, prose-sourced): "six of the sixteen shards ran hours after `6a32143` landed". **Twelve did.**

§3.4 and D-4 both assert *"e518a84 records six shards running a pre-`6a32143` adapter hours after it
landed."* That is not what `e518a84` says and it is not what the artifacts say.

* (b) `e518a84`'s message, verbatim: *"they ran a PRE-6a32143 adapter — **including the six that ran
  today**, hours after that commit landed."* "Today" is 2026-08-24. The six are the 08-24 shards.
* (a) MEASURED, from the shard artifacts' own `measured_utc` against `git show -s --format=%cI
  6a32143` = `2026-08-23T18:07:20+02:00` = `16:07:20Z`:

  | ran BEFORE 6a32143 | ran AFTER 6a32143 | of those, on 08-24 |
  |---|---|---|
  | shards 0,1,2,3 (13:32–14:06 Z, 08-23) | shards 4–15 — **twelve** | shards 10–15 — six |

  and (a) MEASURED: **zero** of the sixteen carry `mask_val_ref_max_frac` in
  `mask_method.version`, i.e. all sixteen ran the pre-`6a32143` adapter.

The deliverable took the commit's "the six that ran today" and re-scoped it to "hours after the
commit landed", which is a different set and is twice as large. **The number is wrong; the direction
is against the deliverable's own interest** — twelve shards, not six, were measured with an adapter
the pushed tree had already replaced, so D-4's severity is understated, not overstated. D-4's
conclusion is unaffected. This is a claim carried from prose into a "measured"-labelled defect list
without being checked against the artifacts that could check it.

### R-2 — CITATION ERROR: `sync.sh:67` does not contain `--exclude '.git'`. Line **66** does.

Stated twice (deliverable D-2, and `blocking_facts` in the summary) as *"sync.sh:67 `--exclude
'.git'`"*. (a) MEASURED: `grep -n "exclude '.git'" cluster/discoverer/sync.sh` → `66:`. Line 67 is
`--exclude '/assets' --exclude '__pycache__' …`. The *substance* of D-2 is confirmed — `.git` is
excluded, `sync.sh:71-72` says so in prose (*"The cluster copy has no .git (excluded above)"*), and
`GIT_COMMIT` is written at `:78` as claimed — but the quoted line is not at the quoted address.

### R-3 — CITATION ERROR: §4 attributes "9.115" to `103_measure_geom_tol.sbatch:126`. It is at `:133`.

Line 126 is the table's dashed separator. The N=16 row carrying `9.115` is line 133. (The returned
summary gets this right and cites `:133`; the deliverable body does not, and the body is the artifact
a reader chases.) D-3's own citation of the table as `:124-135` is correct.

### R-4 — DEFECTIVE PROPOSAL: D-3's replacement constants are neither of the two models its own comment cites.

The proposed diff sets `GEOM_SECONDS_PER_FRAME=0.29` **and** `GEOM_LOAD_SECONDS=410`, and justifies
the pair with *"the per-shard rate ran 0.2640-0.3189 s/frame (mean 0.2879) and a least-squares fit
over all sixteen gives p = 0.2478 with a fixed load of 410 s."* Those are two mutually exclusive
models and the diff takes the slope from one and the intercept from the other:

* the **fit** is `(p, L) = (0.2478, 410)` — I reproduced it exactly;
* the **per-shard mean rate** `0.2879` is `wall / frames`, i.e. it has already absorbed the fixed
  load, and its matching intercept is ≈ 0.

Using `0.29` with `L = 410` double-counts the load. (a) MEASURED consequence on the heaviest shard:
`14162*0.29 + 410 = 4517 s` against an actual `3741 s` — **21 % over**, and the self-check multiplies
by a further 1.25, so the effective refusal threshold sits at 1.51× the truth. This is *safe*
(the check refuses too eagerly, never too late) and the §4 command still passes at `--time=01:50:00`
(`4517*1.25 = 5646 s ≤ 6600 s`), so nothing in the plan breaks. But the diff's comment claims a
measured provenance the pair does not have. The honest pair is `(0.2478, 410)` — the fit — or
`(0.2879, 0)`. Correct the comment or the constants; do not land both numbers under one sentence.

### R-5 — OPERATIONAL DEFECT: the wave sequence, followed as written, is the job-189658 failure.

The sbatch header at `:28-35` is explicit: *"MaxSubmitJobsPU=8 caps SUBMITTED (pending + running)
jobs, AN ARRAY TASK COUNTS AS ONE EACH, and `%4` DOES NOTHING FOR IT … Getting these two the wrong
way round is what cost job 189658."*

* In the deliverable's §4 the wait note (`… wait until squeue … is <= 4, then:`) appears **once**,
  after wave 1. Waves 2, 3 and 4 are then listed back to back with nothing between them — read
  literally, twelve array tasks at once.
* In the **returned summary's `needs_cluster` list the wait note is absent entirely**, and the four
  `sbatch` lines are consecutive — sixteen submissions, which the header says is rejected outright.

Cost is zero GPU-h (Slurm rejects at submit under DenyOnLimit), but this is precisely the mistake the
front is warning about, reproduced in its own runbook. The wait must be stated between every pair of
waves, or the four lines must be replaced by one wave plus "repeat three times, waiting each time".

### R-6 — INCOMPLETE PROPOSAL: D-1's `rm -f` deletes the shard path's only adapter-drift detector, and D-1 and D-4 are coupled without saying so.

D-1's diff removes `${SHARD_OUT}` before measuring so that `merge_committed_contract()` finds nothing
to compare against. That is correct as far as it goes, but the refusal it silences is the *only*
place on the shard path where a segmenter mismatch is currently detected at all — D-4 says so in the
same document (*"nothing compares the adapter against the committed contract BEFORE the GPU work …
only fail at the merge"*). **If D-1 lands without D-4, the shard path loses in-situ drift detection
entirely** and the mismatch surfaces only at the merge, an array later. The deliverable lists them as
four independent diffs in `needs_owner_signature` and never states the dependency. D-1 must not land
alone.

Also not considered: the cleaner fix is one line in `scripts/measure_geom_tol.py`, not in the sbatch
— `committed_segmenter_contract()` (`:1874-1899`) could decline to treat a document whose `schema` is
`wam.geom_tol_shard/1` as a pre-commitment, which is exactly what `merge_committed_contract()`'s own
docstring already intends (`:2050-2052`: *"a document already at `out` that carries no segmenter
block is scratch and is overwritten as before — this is only about a file that made the
pre-commitment"*). A shard artifact under `runs/` never made a pre-commitment. That is a repair to
the reader; the `rm` is a workaround at the caller.

### R-7 — CHECKED, NOT A DEFECT: is 13.636 GPU-h an under-count of *billed* GPU-h?

The 49 091 s are the sum of `ELAPSED` around the `python` call (`sbatch:970-978`), which excludes
everything Slurm charged before it. I checked whether that gap is material. (a) MEASURED, from the
self-check's own line in each log: `walltime remaining 6600s` of a `01:50:00` request (shards 5, 15),
`6599s` (shard 8), and `5398s` of a `01:30:00` request (shard 0) — i.e. **≤ 2 s of pre-`python`
overhead per task, ~32 s over the array.** The 13.636 GPU-h figure holds to within a minute. The
deliverable's number survives this attack.

### R-8 — CHECKED, NOT A DEFECT (and the deliverable never states it): flipping `GATE_QUALIFIED` does not disturb the committed contract.

The whole plan turns on an owner flipping `apple_sam2.py:938`. If that flag were part of
`SEGMENTER_CONTRACT` or of `ESTIMATOR_VERSION`, the flip would itself put the adapter out of
agreement with `configs/transfer25/pr08_geom_tol.json` and the re-measure would refuse at the merge
for a *third* reason. (a) MEASURED by import: `SEGMENTER_CONTRACT` has 16 keys
(`box_selection, box_threshold, depth, detector, mask_validity_min_iou, mask_validity_reference,
mask_validity_reference_max_frame_fraction, method_name, object_text_prompt, pixel_grid_hw,
propagation, retry_box_threshold, retry_text_threshold, segmenter, text_threshold,
upstream_propagation`) — **none containing "gate"** — and `"gate" not in ESTIMATOR_VERSION`. The flip
is contract-neutral. The deliverable happens to be right; it should say so, because a reader
checking the precondition will ask.

### R-9 — CHECKED, CONFIRMS D-3's premise (which the deliverable asserted but did not demonstrate)

D-3 is load-bearing only if the shipped `0.18` fallback is actually what the self-check uses. It is,
and the reason is an artifact the deliverable never opened. (a) MEASURED:
`runs/pr08-geom-tol/GEOM_TOL_PILOT.json` carries `seconds_per_frame = 0.08333…` and
`load_seconds = 116.0` but **no `segmenter` block at all** (neither top-level nor
`mask_method.params.segmenter`). The preflight (`sbatch:906-926`) therefore takes the branch
`basis = "the pilot records no segmenter block, so its operating point is unknown"` and falls back
to `P_FALLBACK`. Confirmed in the logs verbatim: every one of the 16 prints
`[the pilot records no segmenter block, so its operating point is unknown]`. D-3 stands. (Note in
passing: had the pilot carried a matching block, the self-check would have run at `p = 0.0833`,
**3.5× worse than 0.18**.)

### R-10 — NEW, missed by the front: the sbatch's operator-facing text about the blockers is stale at HEAD

`cluster/discoverer/103_measure_geom_tol.sbatch:171-172` states *"its GATE_QUALIFICATION_BLOCKERS
still lead with 'NOBODY HAS LOOKED AT A MASK'"* and `:486` prints the same sentence at run time to
the operator holding a disqualified artifact. (a) MEASURED: `GATE_QUALIFICATION_BLOCKERS` is `()` at
HEAD (`apple_sam2.py:635`). The tuple is empty; the human look was discharged on 2026-08-26. The
sbatch now tells the operator to go and do a thing that is already done, and points at a tuple that
no longer names it. This does not change the plan — `GATE_QUALIFIED` is still `False` and the
re-measure is still disqualified — but the runbook the operator reads *while deciding whether to
spend 13.6 GPU-h* is out of date about why. It belongs in the same commit as D-2 and D-3.

### Things I hunted for and did NOT find

* **No proposal that rewrites a gate after seeing its output.** §3.2 identifies the contract-rollback
  path, names it as forbidden under `docs/handoff.md` §3 (*"Rules are versioned, never edited in
  place. A gate rewritten after seeing its output is not a gate."*, `docs/handoff.md:164-167`), and
  declines it. Correct call.
* **No proposal that signs an unsigned rule.** The `GATE_QUALIFIED` flip is routed to
  `needs_owner_signature` and stated as not this session's to make, consistent with
  `apple_sam2.py:637-648` and with CLAUDE.md.
* **No proposed change that breaks a committed contract.** `contract_fields` in
  `configs/transfer25/pr08_geom_tol.json` is `['spec_version', 'what_this_is', 'contract_fields',
  'measurement_fields', 'segmenter']` and none of the four diffs touches that file or
  `SEGMENTER_CONTRACT`. D-1's `rm` targets `${SHARD_OUT}` under `runs/`, never the tracked config.
* **No `runnable_now` command that touches the cluster, spends money, or mutates the repo.** All four
  are local and read-only; I ran three of them and the fourth (`pytest
  tests/test_measure_geom_tol.py -q`), which the front did not run: **169 passed in 2.22 s**.
* **No ordering error on the canonical axis.** The front puts the `GATE_QUALIFIED` flip strictly
  before the array, in the headline, in §2, in §4's precondition block and in `needs_cluster[0]`.
  That is the right order and it is stated four times.

### Verdict

**The front SURVIVES.** Its four headline claims — the refusal is real and reproducible, the
`gate_qualified: false` bake-in is the prior and independent reason, there is no cheaper path, and
the re-run costs 13.64 GPU-h rather than 9.115 — all reproduce under independent measurement, and
the ~30 line-number citations I sampled land where they say they do with two exceptions (R-2, R-3).

Corrections required before this is quoted: **R-1** (six → twelve shards, an error against an
artifact), **R-5** (the wave sequence as written is rejected at submit), **R-4** (D-3's constants
are a blend presented as a measurement), **R-6** (D-1 must not land without D-4), plus the two
citation fixes and the R-10 addition.
