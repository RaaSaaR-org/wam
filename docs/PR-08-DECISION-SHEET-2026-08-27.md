# DECISION SHEET — everything still between this repo and a first Cosmos-Transfer2.5 clip

**Written 2026-08-27.** Repo `/home/humanoid/develop/wam` at HEAD `19826ccae06f36cd6153125042eda3e7d2a7560b`,
working tree clean at read time. **This sheet replaces §§2–4 and §7 of
`docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md`** and corrects that document where the
investigation refuted it. Six investigators covered one open front each; an adversarial reader
re-checked every one. Where a finding was knocked down, **this sheet reports the knock-down, not the
finding.**

**This sheet licenses nothing.** It does not lift `T40_RULE_V1` §1, does not flip
`apple_sam2.GATE_QUALIFIED`, signs no rule and authorises no submission. Nothing under
`/home/humanoid/develop/wam` was modified in producing it. No cluster was contacted.

Label key on every claim: **[M]** measured by me now on this workstation, command shown; **[A]**
recorded in a committed artifact or tracked source, path + line; **[I]** inference, labelled;
**[NOT MEASURED]** with what it would take.

---

## 1. What changed today

Nine claims in the sprint document are corrected or withdrawn. Four of them change what gets
submitted; three change a cost by more than 40 %.

### 1.1 Item 3 is technically blocked, not "blocked on an owner decision to submit"

`docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md:73` [A]: *"**Blocked on:** an owner decision to
submit. Nothing technical."* **That sentence is wrong at HEAD.**

The `97 TIMING=1` path generates the full clip for the episode it is hard-wired to pick and **then**
refuses in `robot_composite.check_mask` on frame 0, writing no `THROUGHPUT.json`:

* The timed unit is fixed, not a coin flip — `runs/_slurm_logs/t040-restyle.189644.out:42` [A] prints
  `{"unit": "episode_000000__train-01-oak-tungsten__r00", "frames": 590, "seed": 7001}`, and `head -1`
  of a deterministically sorted work list (`97_transfer25_restyle.sbatch:1337`, `:1163`, `:1221`) plus
  `STAGE=1`'s `train[:4]` prefix keeps it that unit today.
* That episode's frame 0 mask is empty. **[M]** from `runs/pr08-robot-mask-area/POOLED.json`:
  `episode_000000` has 590 frames, `area_fractions[0:5] == [0.0,0.0,0.0,0.0,0.0]`, **254 empty of
  590**. Two further committed artifacts agree: `runs/pr08-g0c-refusal/G0C_REFUSAL.json`
  (`first_empty_frame: 0, refuses: true`) [A] and `runs/_slurm_logs/t040-halluc-probe.189926.out:35`
  [M] — *"episode_000000: source mask empty on 93/96 frames"*.
* The generator runs first: `restyle_transfer25.py:515` backend → `:523` `composite.composite`;
  `robot_composite.py:1620` takes masks from `source_masks(source_video, **src**, context)`, `:1635`
  runs `check_mask` from index 0, `:1388` raises on `covered == 0` [M].

**Not blocked by:** `GATE_QUALIFIED` (zero grep hits in the three files on the path; `build_masker().provenance()`
carries no `gate_qualified` key; `THROUGHPUT.json`'s field list does not contain it) **[M]**;
`pr08_geom_tol.json` (the geometry gate sits at `sbatch:1483`, after the timing branch's `exit 0` at
`:1453`) **[A]**; or the ceilings it derives (`sbatch:414-437` exempts both under `TIMING=1`) **[A]**.

### 1.2 The 1.16 s/frame figure is an inference, and the wrong scope — so the mis-pricing is worse than reported

Front 1 built its cost arithmetic on *"1.16 s/frame, job 189926 `[A]`"*. **[M]**
`grep -c "1\.16" runs/_slurm_logs/t040-halluc-probe.189926.out` finds no line stating any per-frame
rate; the number is a division of four `Average time per chunk:` lines (57.75 / 55.45 / 55.46 / 55.47 s
at `:33498`, `:66932`, `:100368`, `:133804`) by 96 frames over 2 chunks. Those chunk timings **exclude**
the depth pass, the SAM2 seg pass and both encodes, which the `THROUGHPUT.json` clock
(`sbatch:1346 S=${SECONDS}` → `:1394`) includes. The artifact's own whole-job rate is
**988 s / 384 frames = 2.57 s/frame** [M]; per-unit boundaries give 1.46–2.16 s/frame [M, ADV].

Consequence, corrected: with job 189142's disqualified `0.2 s/frame` at the default `RUN_ID`, the
partition prices at `0.2 × 4 290 625 / 3600 = 238.4` GPU-h [M] against ~2 380 GPU-h at the artifact's
own pipeline rate — **under-priced by roughly 10×, not 5.8×.** Every GPU-h figure in this sheet that
uses 1.16 s/frame is therefore a **ratio**, and is marked as one.

### 1.3 The GEOM_TOL re-run costs 13.64 GPU-h, not ≈9

Sprint `:121` and `:213` [A] both say *"≈ **9 GPU-h**"*. **[M]** Summing the 16 committed shard logs'
own `=== shard N exited 3 after Ts` lines (`runs/_slurm_logs/geom-tol.1899*_*.out`,
`geom-tol.190125_*.out`): **Σ = 49 091 s = 13.636 GPU-h**. Least-squares over the 16: p = 0.2478
s/frame, L = 410 s; per-shard rate 0.2640–0.3189, mean 0.2879. The 9.115 in the sbatch's planning
table (`103_measure_geom_tol.sbatch:133`) reconstructs exactly as `171625×0.18/3600 + 16×120/3600`
[ADV, M] — a planning constant, **1.50× optimistic**.

Sprint `:215`'s *"about 18 GPU-hours for both item 4 measurements together"* is therefore wrong twice:
the GEOM_TOL half is 13.6, and the EST_DRIFT half is **0 cluster GPU-h** (§1.4).

### 1.4 EST_DRIFT does not need the cluster — but "already measured" does not survive

Sprint `:127` labels §3.3 *"**cluster**, after 3.1"*. **The compute half of that label describes a job
that does not need to exist.** The whole V17 Arm A grid ran on this workstation: eight MuJoCo captures
(CPU, headless, `MUJOCO_GL=egl`) plus thirteen `--arm both` measurements, ~38–40 min on the local
RTX 5090 by the artifacts' own `measured_utc` stamps; one 480-frame both-arm measure is ~3 min [F3, M].

**What does NOT survive** [ADV]: Front 3's stronger headline that `EST_DRIFT_P95` is *"already measured
to the standard the protocol registered and needs no cluster job ever."* `T40_RULE_V17` §4 lines
200-202 [A]: *"whether the pooled number or the single-capture number is the one G0b subtracts is a
separate question this document does not answer."* Neither candidate has plumbing (see defect 8). V17
measured a drift **rate** to its registered standard; the **budget** §6 subtracts is a different
quantity and is unfixed.

### 1.5 Which arm G0b subtracts is OPEN — it was not "already resolved"

Sprint §3.4 [A] states the tension correctly and declines to resolve it: *"which arm is authoritative
is settled by §4 step 2, not by picking the friendlier number."* Front 3 supplied the withheld
resolution (propagation). **REFUTED** [ADV], on four repo citations:

* `apple_sam2.py:701-710` [A] — *"The bias is **TWO-SIDED**, which is why this cannot be waved through
  as conservative … with (a) and (b) together this number is neither a lower nor an upper bound on the
  generator's mask error."*
* `apple_sam2.py:723-728` [A] — *"Both readings are recorded because reporting only the headline would
  make this file's own argument look settled when the sign depends on which percentile the gate uses."*
  The two distributions **cross between p95 and p99** (per-frame p99 1.0431 / p100 67.633 vs
  propagation 0.5631 / 19.399, `apple_sam2.py:690-692`).
* `apple_sam2.py:519-524` [A] — *"the argument about which way that biases the budget — **it is not one
  way**."*
* **The committed contract itself. [M]** `configs/transfer25/pr08_geom_tol.json` →
  `segmenter.propagation == "per_frame"`. I read it this session.

Front 3's two supporting quotations were truncated at the clause that undercuts them:
`measure_geom_tol.py:478-482` continues *"Used ONLY to make the failure message concrete … what Cosmos
names upstream is not evidence about what this adapter loaded"*, and `build_pr08_source.py:33-35`
continues *"until those land, the honest manifest is one that claims no maps at all"* — i.e. the
map-omission is declared **temporary and pending §8 item 4**, not the settled architecture.

The margins are unchanged and I re-derived both against `GEOM_TOL = 0.47857992441961017` [M]:

| arm | pooled p95 (px) | margin (px) | % of GEOM_TOL |
|---|---|---|---|
| per-frame (what the code writes) | 0.3120786214328541 | 0.16650130298675608 | 34.7907 |
| propagation (the generator's topology) | 0.4486097454155794 | **0.02997017900403076** | **6.2623** |

**The gap between them is 0.13653 px = 28.5 % of the budget, and it is an unmade owner decision.**

### 1.6 The joint G0c yield is 17 of 402, not 36 — and it is written down nowhere

Sprint §4 [A] quotes 366/402 empty, 175/402 area, 385/402 either. All three reproduce [M]. **With both
halves of `check_mask` armed — as `configs/transfer25/pr08_robot_mask_area.json` armed them on
2026-08-26 — exactly 17 of 402 episodes survive (4.23 %)** [M, from POOLED.json:
`empty 366 area 175 either 385 survive 17`, 402 episodes / 171 625 frames, `measurement_qualified: true`].
Searched `docs/`, `runs/` and the task files: the product is on record nowhere [F5, ADV both]. It is
the number a stage-1 launch decision turns on.

Front 5 framed this as *"the yield on record is optimistic by 2×"* — **overstated** [ADV]. The committed
`bound_rationale` states both halves and that they compose by OR in one paragraph; no committed document
claims 36 is the joint yield. The defensible finding is that nobody multiplied them out.

### 1.7 V16's outcome is M today, and Front 5's proposed successor does not survive

Sprint §4 [A]: *"`T40_RULE_V16`'s outcome **A** is unreachable … Remaining: **B** or **M**."* Correct,
and the evidence now selects **M**:

* **A is dead on a measurement.** `q99` of `frac_dev` over all 57 835 empty-mask frames = **0.07180**
  against the registered threshold 0.01; 87.24 % of frames exceed 0.01 (`runs/pr08-empty-mask-look/MOTION.json`)
  [F5, reproduced by ADV to 0.0717958].
* **B's quantity does not exist.** `runs/pr08-empty-mask-look/VERDICTS-partial-101.json` carries
  `status: "PARTIAL AND NOT EVALUATED under V15 §5 — see V16"` and 101 tiles in V15's vocabulary.
  **[M] I opened it: 2 `yes`, 75 `no`, 24 `cannot_tell`** — Front 5 reported 2/77/22, which is a
  miscount of the one file it claimed to have read. V16 §7 forbids mapping any of it onto A/B/C/D, so
  no `p_A` exists and none was computed.
* **Residual clause fires → M.**

**Front 5's draft `T40_RULE_V20` (trim + identity fallback) is REFUTED** [ADV] and must not be carried
forward as written: (i) writing the source frame on an **over-bound** frame is, per
`robot_composite.py:1405-1408` and `:1225-1236` [M, I read both], exactly *"the restyle becomes a no-op,
and arms B and C silently become arm A while still costing their GPU hours"* — it disarms the area half
through the work unit instead of the config; (ii) it breaks the committed harvest contract at
`97_transfer25_restyle.sbatch:2065-2072` [M] — *"Every frame or none"*; (iii) the sketch's identity
indices are span-relative while the consumer's loop index is source-relative; (iv) it writes bare `NaN`
into an artifact. **What survives from Front 5 is the ordering finding and the 17/402 count.**

### 1.8 The `plate.` premise has moved, and REPAIR A has a second lock nobody has named

Sprint §4's last paragraph [A]: *"the `plate.` pass refuses 100 % of source frames."* Since PR-08 V10
landed (2026-08-24) it refuses the **run** at the first call — `MaskValidityReferenceUndefined`,
`SEGMENT_CALLS=0`, `MASK_REFUSED_FRAMES=0`, no weight loaded [F6, M]. §6's plate half is still
unmeasurable; only the failure mode changed.

Root cause is the **filter, not the detector**: a *correct* plate mask scores warm-fruit IoU **0.0000**
with `plate_overlap_fraction` 0.9748–0.9807 while the detector's winning score is 0.7524–0.7773
(`runs/pr08-mask-audit/MASK_AUDIT.json`; V10 §1.1's matched control) [F6, confirmed by ADV].

**The second lock, unnamed in V6, V9, V10 and T-040:** `object_text_prompt` is a **single-valued**
`SEGMENTER_CONTRACT` field pinned to `"apple."` [M, read from `configs/transfer25/pr08_geom_tol.json`]
while §6 gates two labels. Even a perfectly repaired filter yields a G0b run disqualified on
`contract.object_text_prompt: 'plate.' vs 'apple.'`, and a plate pass pointed at the committed document
is refused outright (exit 2). Fixing it changes the contract's **shape**, not its values.

### 1.9 Corrections inside the fronts themselves, so they are not carried forward

* **Front 6's "zero refusals of any kind" on the MuJoCo captures is false.** **[M] I walked all three
  artifacts:** `/estimator_stats/this_run/n_frames_mask_refused` = **25** (s60-f720, of 720 calls),
  **12** (s20-f240, of 240), **1** (trajectory-f480, of 480). The zeros quoted are
  `counters_at_start_of_run`, zero by construction. Only `no_reference` is genuinely 0 on all three.
* **Front 6's §1.5 says item 4 is held open by "a residue/signature question".** Wrong in the direction
  that costs GPU-hours. **[M] I ran the repo's own comparator:**
  `contract_disagreements(landed, committed)` → `[{'field': 'mask_validity_reference_max_frame_fraction',
  'geom_tol': 0.1, 'this_run': None}]`; landed contract 15 fields, committed 16. Flipping
  `GATE_QUALIFIED` does not qualify 0.4786 — the corpus pass must be **run again**.
* **Front 2's "six of sixteen shards ran a stale adapter" is twelve** [ADV], and **zero** of the sixteen
  carry `mask_val_ref_max_frac` — the error runs against its own interest, so its defect D-4 is
  *understated*, not overstated.
* **Front 4's "the five frames are named" is withdrawn** [ADV]. Exhaustive search over 68 280 feasible
  4-subsets finds `{101,108,124,153}` and `{101,108,124,154}` **tied** with the accepted hypothesis on
  max|D| at 0.165650 px. The artifact names `f101, f108, f124`, **one of {f152, f153, f154}**, and one of
  {f113, f116, f125}. Front 4's V18-invariance recomputation is an **inference**, not a measurement:
  `shard-7.json` records no per-frame mask area, so it applied the workstation's areas to a
  cluster-derived index set.
* **Front 1's D1 pre-flight diff is not semantically neutral** [ADV]. `MaskCache.key`
  (`robot_composite.py:1005-1015`) is keyed on the source sha256 + segmenter identity, not the path, and
  the pre-flight's cache directory resolves to the same directory the driver defaults to — so the
  590-frame mask pass would fall **outside** the timed window, removing 30–40 % of the measurand in the
  under-deriving direction. That is an owner decision, not an implementation detail.
* **Front 4's two most-cited artifacts are untracked.** `.gitignore:19` ignores `runs/`;
  `runs/pr08-geom-tol/shards/shard-7.json` and `runs/pr08-operating-point/EPISODE_094_CENSUS.json` are
  **not committed** [ADV], and the census was already overwritten in place once the same night
  (`.json.v1` at 02:49:10Z vs `.json` at 02:56:24Z, and `.v1` lacks the `centroid` field entirely).

---

## 2. The critical path, as it actually is

Read the three LOUD lines first; they change what gets submitted and in what order.

> **LOUD 1 — the sprint's step 1 (cluster census of `episode_000094`) is NOT on the critical path.**
> `T40_RULE_V18` §2 registers **the census** as the instrument and §3's outcomes are written against it;
> the census has run, three times locally, agreeing on 0/509 rows. A cluster run cannot reproduce the
> shard's instrument anyway: its own Precondition 1 requires syncing to HEAD (post-V10) while the shard
> ran a pre-V10 adapter (`07965aa`), so it would vary **machine and code at once** [F4 + ADV].
>
> **LOUD 2 — `EST_DRIFT_P95` is a LOCAL run, not a cluster job.** Sprint `:127`'s "cluster" label costs a
> submission that does not need to exist. ~3 min on the local RTX 5090 per 480-frame capture; the whole
> V17 grid already ran here in ~38 min.
>
> **LOUD 3 — the arm decision must land BEFORE the GEOM_TOL array, not after.**
> `segmenter.propagation` is a **pre-registered contract field** currently reading `"per_frame"` [M]. If
> the owner decides G0b subtracts the propagation arm, that field changes, and changing it after the
> array means paying 13.6 GPU-h twice. Nobody has stated this ordering anywhere.

**Step 1 — OWNER SIGNATURE: residue (i), then flip `GATE_QUALIFIED` in its own commit.**
Cost: a signature and one commit. Produces: the precondition every gate-qualified artifact needs.
Unblocks: steps 3, 4, 5 — i.e. all of §8 item 4.
`GATE_QUALIFICATION_BLOCKERS` is `()` at `apple_sam2.py:635` and `GATE_QUALIFIED = False` at `:938` [M];
the module's own text at `:637-648` says emptiness satisfies one of two preconditions. The other is a
recorded decision on blocker 2's residue (i). **The evidence for it is complete except for one number
that cannot move the outcome** — see decision D-A.

**Step 2 — OWNER SIGNATURE: which arm G0b subtracts (and pooled vs single).**
Cost: a signature; a new V-document if it moves `SEGMENTER_CONTRACT["propagation"]`.
Produces: the definition of the number step 4 measures. Unblocks: steps 3–5 *at the right contract*.
**Must precede step 3** (LOUD 3). Worth 0.13653 px = 28.5 % of the budget.

**Step 3 — CLUSTER: `GEOM_TOL` re-measured at HEAD, 16 shards + merge.**
Cost: **13.64 GPU-h** [M] (not 9.115), + ~20 s CPU for the merge. Produces:
`configs/transfer25/pr08_geom_tol.json` with a `gate_qualified: true` number. Unblocks: step 4's
cross-check and the whole of item 4.
Two independent reasons the existing 0.4786 px can never be committed: `gate_qualified: false` baked into
all 16 shards at measurement time (`measure_geom_tol.py:996, 1044, 3849, 3990, 4017`; merge re-reads it
from shard 0 at `:2982, :2996`), **and** the one-field contract disagreement I measured in §1.9. Neither
a re-merge, a migration nor a flag exists — the CLI has no `--force`, no `--contract`, no migration flag
[F2, ADV via argparse].

**Step 4 — LOCAL, NOT CLUSTER: measure `EST_DRIFT` → `configs/transfer25/pr08_est_drift.json`.**
Cost: ~3 min of the local RTX 5090 per capture; **0 GPU-h of allocation**. Produces the drift half.
**Must follow step 3**: `measure_est_drift.cross_check_geom_tol` (`:1578-1589`) disqualifies on both the
*absence* and the *falsity* of `gate_qualified` in the committed GEOM_TOL document, and **[M]** that
document has no `gate_qualified` key at all today.

**Step 5 — LOCAL: `measure_geom_tol.py --carry-est-drift`.** Seconds. Produces `est_drift_p95_px`,
`est_drift_estimator_name`, `est_drift_source`, `gate_margin_px`. **→ §8 item 4 closes iff the margin
is > 0.** Blocked by defect 2 if step 2 answers "propagation": the carry reads only the top-level
`est_drift_p95_px`, which is the per-frame arm by construction.

**Step 6 — CLUSTER: `97 TIMING=1` → `THROUGHPUT.json` → derive the ceilings. → §8 item 3 closes.**
Independent of steps 1–5 (the `GATE_QUALIFIED` exemption is verified three ways, §1.1). **Blocked on an
owner decision (D-E), not on a signature already listed above** — at HEAD it is a guaranteed loss.

**Step 7 — OWNER: the empty-mask semantics (a V16 successor under outcome M).**
Formally optional; economically it is the difference between 17 of 402 episodes and most of them.
Cost: a rule version, and the human half of V16 if the owner wants `p_A`.

**Step 8 — OWNER: lift `T40_RULE_V1` §1 on the record, naming a signatory.** Only after 5 and 6.

---

## 3. Runnable on this workstation today, for free

All read-only against `runs/` and `configs/`, no GPU unless stated, no cluster, no repo mutation. Every
one of these was executed in this investigation.

```bash
cd /home/humanoid/develop/wam

# (1) THE NUMBER NOBODY HAS WRITTEN DOWN: the joint G0c yield, 17 of 402.  ~1 s, no GPU.
.venv/bin/python -c "
import json;p=json.load(open('runs/pr08-robot-mask-area/POOLED.json'));B=0.64091145833333329;pe=p['per_episode']
f=lambda t:sum(1 for e in pe if any(t(v) for v in e['area_fractions']))
print('episodes',len(pe),'frames',sum(e['n_frames'] for e in pe))
print('empty',f(lambda v:v==0.0),'area',f(lambda v:v>B),'either',f(lambda v:v==0.0 or v>B),
      'survive',len(pe)-f(lambda v:v==0.0 or v>B))"
# -> episodes 402 frames 171625 / empty 366 area 175 either 385 survive 17

# (2) V16 outcome A is dead on a measurement.  ~2 s, no GPU.
.venv/bin/python -c "
import json,numpy as np;d=json.load(open('runs/pr08-empty-mask-look/MOTION.json'))
v=np.array([f['frac_dev'] for e in d['per_episode'] for f in e['frames']])
print(v.size,np.median(v),*np.percentile(v,[90,95,99]),v.max(),(v>0.01).mean())"
# -> 57835 ... p99 0.07180, 87.24 % over the 0.01 threshold

# (3) THE GEOM_TOL REFUSAL, run against the real code path.  ~10 s, no GPU. Writes to scratch only.
S=$(mktemp -d)   # scratch only: the merge must NEVER be pointed at the tracked config
cp configs/transfer25/pr08_geom_tol.json "$S/remerge_out.json"
.venv/bin/python scripts/measure_geom_tol.py --merge runs/pr08-geom-tol/shards/shard-*.json --out "$S/remerge_out.json"
# -> EXIT 2, "mask_validity_reference_max_frame_fraction: committed 0.1, this run None", nothing written

# (4) The one-field contract disagreement, from the repo's own comparator.  ~2 s.
.venv/bin/python -c "
import json,sys,importlib.util;sys.path.insert(0,'scripts')
s=importlib.util.spec_from_file_location('m','scripts/measure_geom_tol.py')
m=importlib.util.module_from_spec(s);sys.modules['m']=m;s.loader.exec_module(m)
c=json.load(open('configs/transfer25/pr08_geom_tol.json'))['segmenter']
l=json.load(open('runs/pr08-geom-tol/pr08_geom_tol.json'))['estimator_stats']['adapter']['segmenter_contract']
print(m.contract_disagreements(l,c), len(l), len(c))"
# -> [{'field': 'mask_validity_reference_max_frame_fraction', ...}] 15 16

# (5) The 13.64 GPU-h, from the previous array's own logs.  instant.
for f in runs/_slurm_logs/geom-tol.1899*_*.out runs/_slurm_logs/geom-tol.190125_*.out; do
  grep -o 'exited [0-9]* after [0-9]*s' $f | head -1; done | awk -F'[ s]' '{s+=$4} END {print NR, s, s/3600}'
# -> 16 49091 13.6364

# (6) The EST_DRIFT refusal counters, read from the RIGHT block.  ~1 s.
.venv/bin/python -c "
import json
for f in ('s60-f720','s20-f240','trajectory-f480'):
    d=json.load(open('runs/pr08-est-drift/EST_DRIFT-mujoco-%s.json'%f))
    t=d['estimator_stats']['this_run']
    print(f,'refused',t['n_frames_mask_refused'],'of',t['n_segment_calls'])"
# -> 25/720, 12/240, 1/480 -- NOT the zeros in counters_at_start_of_run

# (7) The two margins, both arms, re-derived.  ~1 s.
.venv/bin/python -c "
import json;G=0.47857992441961017;p=json.load(open('runs/pr08-est-drift/v17/POOLED-V19.json'))
for a,b in p['arms'].items():
    v=b['pooled_est_drift_p95_px'];print(a,v,G-v,(G-v)/G*100)"

# (8) The timing run's refusal, reproduced against the committed bound.  ~5 s, no GPU.
.venv/bin/python -c "
import json,sys,numpy as np;sys.path.insert(0,'scripts');import robot_composite as rc
b=rc.load_area_bound();rc.check_mask(np.zeros((480,640),bool),frame_index=0,bound=b,source='timing_raw/.../vision.mp4')"
# -> CompositeError: the robot mask is EMPTY on frame 0

# (9) The plate. refusal at HEAD (premise-has-moved).  ~2 s, no weights, no network.
WAM_PR08_OBJECT_PROMPT="plate." .venv/bin/python - <<'PY'
import sys, types, importlib.util, numpy as np
for n in ("transformers","sam2","torch"):
    if importlib.util.find_spec(n) is None: sys.modules.setdefault(n, types.ModuleType(n))
s = importlib.util.spec_from_file_location("apple_sam2","scripts/estimators/apple_sam2.py")
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("defined?", m.mask_validity_reference_is_defined())
try: m.segment(np.zeros((480,640,3),dtype=np.uint8))
except Exception as e: print("RAISED:", type(e).__name__)
print("SEGMENT_CALLS", m.SEGMENT_CALLS, "MASK_REFUSED_FRAMES", m.MASK_REFUSED_FRAMES)
PY

# (10) The regression baselines any proposed diff must not break.  ~8 s total.
.venv/bin/pytest -q tests/test_restyle_transfer25.py tests/test_transfer25_staging.py   # 119 passed
.venv/bin/pytest -q tests/test_measure_geom_tol.py                                       # 169 passed
.venv/bin/pytest -q tests/test_apple_sam2_estimator.py tests/test_robot_composite_object_filter.py \
                    tests/test_apple_sam2_video_propagation.py                           # 150 passed
```

**Also free and local, after step 1 and step 3 land (GPU, ~3 min):** the EST_DRIFT measurement itself
and the carry. Do **not** run them before then — every artifact they produce carries
`gate_qualified: false` and the carry refuses on it.

---

## 4. Cluster submissions

Three, and only three. **None of them should be typed before the decision named beside it is answered.**

### 4.1 `GEOM_TOL` re-measure — 13.64 GPU-h

**Gated on:** decision D-B (flip `GATE_QUALIFIED`) **and** D-C (which arm, if it moves the contract).

```bash
# 0. From the workstation, BY A HUMAN. Pushes HEAD, stamps ${PROJ}/wam/GIT_COMMIT, and repairs the
#    cluster's configs/transfer25/pr08_geom_tol.json, which job 190191 overwrote with the merged
#    disqualified artifact. `git -C ${WAM} checkout --` CANNOT do this: sync.sh:66 excludes '.git'.
./cluster/discoverer/sync.sh

# 1. FRESH RUN_ID -- never into runs/pr08-geom-tol (defect 3). Four waves of four.
#    MaxSubmitJobsPU=8 counts EVERY ARRAY TASK and %4 does nothing for it.
#    WAIT between every pair of waves:  squeue -u "$USER" -r -h -o '%i' | wc -l   must be <= 4.
RUN_ID=pr08-geom-tol-v2 SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 GEOM_SECONDS_PER_FRAME=0.29 \
  sbatch --array=0-3%4 --time=01:50:00 cluster/discoverer/103_measure_geom_tol.sbatch
#    ... wait ... then --array=4-7%4 ... wait ... 8-11%4 ... wait ... 12-15%4, same env.

# 2. MERGE -- no GPU, ~20 s, free QoS, limits on the command line.
RUN_ID=pr08-geom-tol-v2 MERGE=1 NUM_SHARDS=16 \
  sbatch --qos=2cpu-single-host --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00 \
  cluster/discoverer/103_measure_geom_tol.sbatch

# 3. ./cluster/discoverer/sync.sh --pull pr08-geom-tol-v2   then commit the config AND its .sha256
```

**What would waste it:** (a) submitting before `GATE_QUALIFIED` flips — all 16 shards bake
`gate_qualified: false` at measurement time and the merge exits 3; (b) reusing `RUN_ID=pr08-geom-tol` —
with `FORCE=1` every task decodes ~11 000 frames and *then* exits 2 on the stale shard read as a
contract; without `FORCE=1` every task prints "already landed. Skipping." and the array silently
measures nothing; (c) not re-syncing — the cluster copy lags HEAD and twelve of the last sixteen shards
ran a stale adapter for exactly that reason; (d) submitting all four waves back to back — rejected at
submit under `DenyOnLimit` (0 GPU-h, but it is the job-189658 mistake); (e) shipping
`GEOM_SECONDS_PER_FRAME=0.18` — the self-check then estimates the heaviest shard at 44.5 min against an
actual 62.4 min and would pass a `--time=01:00:00` request that dies at the wall. *(Note: `0.29` with
`GEOM_LOAD_SECONDS=410` blends two models and over-estimates by 21 % — safe, and it still passes at
`--time=01:50:00`, but the honest pair is `(0.2478, 410)` or `(0.2879, 0)`.)*

### 4.2 `97 TIMING=1` — 1.5 GPU-h reserved, ~0.4–0.6 consumed if it completes

**Gated on:** decision D-E. **At HEAD it is a guaranteed loss** (~0.3–0.5 GPU-h and a job slot for no
artifact).

```bash
# on login-plus, by a human, from ${PROJ}/wam/cluster/discoverer, AFTER D-E is answered
TIMING=1 STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=1 CONTROL=depth:0.5,seg:0.5 \
RUN_ID=t040-transfer25-restyle-timing-2026-08-27 \
  sbatch --qos=ehpc-aif-2026pg01-905 --time=01:30:00 --gres=gpu:1 --cpus-per-task=26 --mem=98304 \
  97_transfer25_restyle.sbatch
```

**What would waste it:** (a) submitting with the empty-mask semantics unresolved — it dies in
`check_mask` after the clip is generated; (b) **omitting `RUN_ID`** — the header's own recipe
(`sbatch:8-9`) passes none, `RUN_ID` defaults to `t040-transfer25-restyle` (`:592`), and the timing
branch's first act (`:1319-1323`) is to `cat` job 189142's disqualified 0.2 s/frame file and `exit 0`;
(c) `--gres=gpu:N>1` — the timing branch refuses `NPROC != 1`; (d) cold Transfer2.5 checkpoints — ~22 GB
downloads *inside* the measured window; (e) a missing `HF_TOKEN` / unaccepted licence — job 189402 died
on exactly that.
**Preflights that need a cluster `ls` and that I could not run [NOT MEASURED]:** whether
`${PROJ}/runs/t040-transfer25-restyle/THROUGHPUT.json` still exists, and whether the source manifest
still hashes to `a988dd60db6ba8abec719dd9ec73ee74ca849465f1fa36666c8564f853bf91be`.

### 4.3 `episode_000094` census on the cluster — ~0.2–0.4 GPU-h [I, extrapolated from local wall clock]

**Recommended NO.** See decision D-G. It cannot move the V18 outcome, and as specified it varies machine
and code together, so it cannot even answer the question it is for. If it is ever run it must run
`07965aa`, not HEAD, and `--qos=<project qos>` is mandatory.

**Not a submission, and worth saying:** the per-style `segment()` pass that V6 §5.3, V9 §5.4 and V10 §5.1
have each asked for **cannot be scoped today** — restyled clips exist for 2 of 16 style-instances, and
generating the other 14 is itself gated by §8. That is a genuine ordering problem in PR-08 and it is
named here rather than worked around. **[NOT MEASURED — requires a corpus that §8 forbids producing.]**

---

## 5. Decisions only the owner can sign

Nine questions. Each is answerable with one word. **Nothing here is decided, and this sheet signs none
of them.**

### D-A — Accept `T40_RULE_V18` §3 outcome **C** (contained) for blocker 2's residue (i)?  **YES / NO**

**For.** The registered instrument is the census (V18 §2), and it has run three times locally with
0/509 rows differing. Outcome **U** cannot fire: its two triggers are zero refusals or refusals outside
`~f101–f155`, and the census found 31 refusals spanning `[109, 149]`. Outcome **E** cannot fire on the
census: median non-refused area 5 650.5 px, largest non-refused 7 383 px at f82 = **1.307×** against a
3× bar. The decode hypothesis for the 473-vs-478 gap is **refuted**: `episode_000094` decodes
bit-identically through cv2 5.0.0, imageio/FFMPEG and pyav 18.0.0 over both corpus trees — 509/509
frames identical on all four pairings.
**Against.** The gap is unexplained in its details. Front 4's claim that the outcome is *measured* to be
invariant under the cluster's 36 is an **inference** — `shard-7.json` records no per-frame mask area, so
the recomputation used the workstation's areas on a cluster-derived index set. The frames are not fully
named: `f101, f108, f124`, one of `{f152, f153, f154}`, one of `{f113, f116, f125}`. And nobody has
looked at them.
**Recommendation: YES**, on the narrower ground that V18 registered the census as the instrument and its
outcome table is written against the census, not against the shard — **not** on the invariance argument
Front 4 offered. The stronger surviving argument is displacement-based: the shard's 466 displacements
match the workstation's to max 0.166 px, which pins the cluster's centroids on 471 of the 473 kept
frames, and a ~31 000 px plate mask cannot share a centroid with a ~1 000 px apple mask.
**Cost of NO:** every downstream step stalls. `GATE_QUALIFIED` stays `False`, item 4 cannot close, and
the only route left is a cluster census that (D-G) cannot isolate the variable it is aimed at.

### D-B — Flip `apple_sam2.GATE_QUALIFIED` `False → True`, in its own commit?  **YES / NO**

**For.** `GATE_QUALIFICATION_BLOCKERS` is `()` at `apple_sam2.py:635` [M]; the human look was discharged
2026-08-26; the propagation blocker was discharged 2026-08-27 by recording both p95s, which is what its
registered discharge condition asked for. The flip is **contract-neutral**: `SEGMENTER_CONTRACT` has 16
keys, none containing "gate", and "gate" is not in `ESTIMATOR_VERSION` [ADV, by import] — so it creates
no new refusal.
**Against.** `apple_sam2.py:637-648` [M] states the flag has two preconditions and that emptiness
satisfies one. The second is D-A. The module and CLAUDE.md both say no session may make this edit, and
no session did.
**Recommendation: YES, if and only if D-A is YES, and as a commit that does nothing else.** The module's
own rule forbids a tuple-shortening commit from carrying it.
**Cost of NO:** 13.64 GPU-h buys another artifact stamped `gate_qualified: false`, `EST_DRIFT` cannot be
gate-qualified locally, and item 4 stays open indefinitely. It is the single blocking fact on the whole
item-4 chain.

### D-C — Which arm does G0b subtract: **PER-FRAME / PROPAGATION**?

**For per-frame.** It is what the committed pre-registered contract says today
(`segmenter.propagation == "per_frame"` [M]), what the code writes into `est_drift_p95_px` "by
definition" (`measure_est_drift.py:2333-2337, :2514`), and what the carry reads
(`measure_geom_tol.py:3467`). Margin **0.16650 px, 34.79 %**.
**For propagation.** The generator's own segmenter propagates, and `restyle_transfer25.py:342-343` says
the intended end state is *"the run uses the estimator the geometry budget characterises."* Margin
**0.02997 px, 6.26 %**.
**Against deciding it on the p95 alone:** `apple_sam2.py:723-728` [A] records that the two distributions
**cross between p95 and p99**, so "propagation is the worse arm" is a p95-only statement, and
`:701-710` records the bias as **two-sided** — neither a lower nor an upper bound.
**Recommendation: no recommendation on the value — this is the one question a session must not answer,
and Front 3's attempt to answer it was refuted.** What can be recommended is the **ordering**: answer it
**before** the GEOM_TOL array, because "propagation" changes a pre-registered contract field and
therefore needs a new V-document *and* a re-measure.
**Cost of NO / of deferring:** the first successful carry writes the per-frame number by default —
0.16650 px instead of 0.02997 px, a **5.56× wider** per-clip tolerance handed to the generator by a field
name, with nothing downstream able to see it (both arms record the same `SEGMENTER_CONTRACT`). Deferring
past the array costs 13.6 GPU-h a second time.

### D-D — Is G0b's budget the **POOLED** p95 or the **SINGLE**-capture p95?

**For.** `T40_RULE_V17` §4 lines 200-202 [A] explicitly leaves it open. The pooled number rests on eight
captures / 3 840 frames; a single capture is what the plumbing can actually carry.
**Against either, today.** Neither has a path. The pooled artifact carries
`schema: "wam.est_drift_pooled/1"` while `est_drift_measurement` requires `"wam.est_drift/1"`, plus no
`gate_qualified`, no `estimators.name`, no `resolution_hw` — four refusals deep. And a **12-frame**
capture reaches `headline_valid: true` today, because V5 §5's floor is advisory
(`independent_sample_block`: *"no disqualification reason depends on it"*).
**Recommendation: POOLED**, with the plumbing written to carry it, because the whole point of V17 was
that one 480-frame capture is not enough — but note that this needs code that does not exist, and that
the floor question is a separate V-document (a session may not convert an advisory block into
`gate_disqualified_reasons` after the artifacts exist).
**Cost of NO / of not deciding:** the operator reaches the last step of a four-step chain and discovers
the number V17 spent eight captures producing cannot reach the gate document by any committed tool.

### D-E — What does an empty robot mask mean for the **TIMING** run: **WAIVE / REGISTER-EPISODE / RESOLVE-V12-FIRST**?

**Context.** The timing run generates a 590-frame clip and then refuses on frame 0. The three registerable
exits are (a) waive G0c for the measurement only, on the argument that a timed clip is deleted and never
enters a corpus; (b) register **which** episode the measurement uses, from the 17 that survive both
halves; (c) resolve the corpus-wide empty-mask semantics first (D-F) and let the timing run follow.
**Against (a):** `sbatch:1358-1379` pre-registers the opposite — *"a throughput number that excluded the
gate would under-derive the GPU-h ceiling the generation path is then held to"* and *"timing the generator
without the composite would be a wall clock around a pipeline we do not run."*
**Against (b):** choosing the episode **because** it survives G0c is a selection made after seeing the
data, and a robot-always-visible episode is not a median episode. It is registerable, but only as a rule
that says so. Note also that the 17-list's **area** half is machine-conditional: 37 of 44 near-bound
frames moved by >0.01 when re-rendered on an RTX 5090 (`pr08_robot_mask_area.json` `bound_rationale`).
**Against (c):** it makes item 3 — which is otherwise fully independent — wait on item 5 of the sprint.
**Recommendation: (b) REGISTER-EPISODE**, written as a rule version that states plainly that the episode
was selected from the 17 and that the resulting `seconds_per_frame` is biased by the selection. It is the
only exit that produces a real end-to-end wall clock without switching a gate off. `episode_000093`
(448 frames, closest of the 17 to the corpus median of 421.5) is reachable with no code change via
`CHUNK_TOTAL=402 CHUNK_INDEX=94`.
**Cost of NO / of doing nothing:** every timing submission burns an H200 slot to reproduce a refusal two
committed artifacts already predict, and item 3 never closes.

### D-F — Run V16's human half (240 tiles, A/B/C/D) to produce `p_A`?  **YES / NO**

**For.** `p_A` is the number that decides between V16 outcome **B** and outcome **M**, and the
2026-08-27 result says the reviewer reports that class is the one they can judge. Today's evidence
selects **M** only because `p_A` does not exist.
**Against.** It is a person's task, not a job — 0 GPU-h, but reviewer time — and the previous attempt
stopped at 101 of 240 tiles. The recurring pattern is that these terminate in "a person must look" and
the review page is the tool built for exactly that.
**Recommendation: YES**, and before any successor to V12 is drafted. Front 5's attempt to write a
successor **without** `p_A` produced a rule that disarms the area half of `check_mask` and breaks the
harvest contract — which is evidence that the shortcut does not exist.
**Cost of NO:** the corpus-wide empty-mask question stays at outcome M indefinitely; any successor
written without `p_A` is a rationalisation of a 4.23 % yield rather than a response to a measured fact;
and stage 1 either launches at 17/402 or does not launch.

### D-G — Spend GPU-hours on a cluster census of `episode_000094`?  **YES / NO**

**For.** It would name the fifth frame and turn "36" into 36 indices. Sprint §3.1 currently names it as
*"what closes"* residue (i).
**Against.** It cannot move the V18 outcome (D-A). Its own Precondition 1 requires syncing to HEAD while
the artifact it would reproduce came from `07965aa`, so it varies machine and code at once. A cluster
census returning 31 would be a legitimate result meaning the 36 was a property of that job. And the
underlying phenomenon — cross-machine non-reproducibility of a marginal detector at scores ~0.15–0.20 —
is not closed by one more sample of one machine.
**Recommendation: NO.** If the answer is YES anyway, run `07965aa`, not HEAD.
**Cost of NO:** the fifth frame stays unnamed. It is unrecoverable from any artifact in this repository
and it changes nothing that is gated.

### D-H — Arm REPAIR A: is §6's plate half measured **UNFILTERED**?  **YES / NO / DEFER**

**For.** `apple_sam2.py:483-486` [A] names this as the owner's call and says *"it needs the committed
contract to say so."* Until it is answered, §6's plate half cannot be measured at all.
**Against.** It is two changes, not one: a declarable unfiltered-label state **and** a decision about
`object_text_prompt`, which is single-valued in the contract while §6 gates two labels (§1.8). The second
changes the contract's **shape** and invalidates the field-for-field cross-check as written. Both
candidate repairs to the *reference itself* are already rejected on measured grounds (a per-label colour
predicate is coined numbers in the gate path with no measured gap; a paired-source-mask reference refuses
exactly the frames with the largest displacement).
**Recommendation: DEFER.** The plate half gates **training on the result** (G0b's qualification), not
**generating a corpus** — `PR-08-NOTE-2026-08-25` §1 and `measure_geom_tol.py:164-167` both say so:
*"That does not block computing `GEOM_TOL`; it blocks applying one number to both."* It is downstream of
this sprint and should not be opened inside it.
**Cost of NO / DEFER:** G0b stays a single-label gate, and the eventual plate pass will need a rule
version anyway. **Cost of a careless YES:** the entire GPU spend of a plate G0b pass, discarded at the
cross-check on `contract.object_text_prompt: 'plate.' vs 'apple.'`.

### D-I — Lift `T40_RULE_V1` §1 and authorise generation?  **YES / NO** — *not answerable yet*

Listed for completeness and because it is the terminal decision. §8 items 3 and 4 are both open; §1's
prohibition is untouched by everything above. **A YES today would be a lift without the two measurements
it is conditional on** — the same ground on which the 2026-08-24 lifting proposal was declined.
**Recommendation: not yet.** Ask again after steps 5 and 6 of §2 produce a positive margin and a
`THROUGHPUT.json`.

---

## 6. Defects found, ranked by what they cost

| # | defect | cost if unnoticed |
|---|---|---|
| **1** | **`est_drift_p95_px` is hard-wired to the per-frame arm** (`measure_est_drift.py:2333-2337`, `:2514`) and the carry reads only that field (`measure_geom_tol.py:3467`). `--arm both` changes nothing about it. Both arms record the *same* `SEGMENTER_CONTRACT`, so `contract_disagreements` cannot see the difference. | **Corrupts a gate, not a job.** The first successful carry hands G0b **0.16650 px** instead of **0.02997 px** — a 5.56× wider per-clip tolerance, 28.5 % of `GEOM_TOL`, by a field name. Not mis-gating today only because all three slots are null. *Front 3's proposed fix does not compile and is backwards; re-site any check in `carry_est_drift_main`.* |
| **2** | **The `TIMING=1` run refuses in `check_mask` after full generation.** `restyle_transfer25.py:523` runs the composite after the backend; `robot_composite.py:1635` → `:1388`. | ~0.3–0.5 GPU-h + an H200 slot per attempt, no `THROUGHPUT.json`, and item 3 does not close. The sprint calls it "nothing technical". |
| **3** | **A `FORCE=1` re-measure into `runs/pr08-geom-tol/shards/` refuses AFTER the decode.** `merge_committed_contract` runs on the measuring path with `--out` = the shard path (`:4134`), and `committed_segmenter_contract` (`:1892-1897`) reads a stale shard's `mask_method.params.segmenter` as a committed contract. Demonstrated twice, independently. | **13.6 GPU-h destroyed**, nothing written, and the message blames the operator's segmenter. Without `FORCE=1` it is worse-but-cheaper: `shard_artifact_landed` (`sbatch:762-806`) checks nine things and **nothing about the segmenter**, so all 16 tasks exit 0 having measured nothing. Mitigation with no source change: a fresh `RUN_ID`. |
| **4** | **The default `RUN_ID` still holds job 189142's disqualified `THROUGHPUT.json`** (0.2 s/frame from a run whose own log says *"0 success, 1 error"*), and the timing branch `cat`s it and `exit 0`s (`sbatch:1319-1323`, `:592`). The header's own recipe passes no `RUN_ID`. | A silently wrong budget line: the partition prices at **238.4 GPU-h** against ~2 380 at the artifact's own rate — **~10× under**, with `max_passes_per_chunk` derived from the same fiction. |
| **5** | **`GEOM_SECONDS_PER_FRAME` defaults to 0.18** (`103_...sbatch:861`) against a measured 0.2640–0.3189, and `GEOM_LOAD_SECONDS=120` against a fitted 410. | The walltime self-check estimates the heaviest shard at 44.5 min against an actual 62.4 min — it passes a `--time=01:00:00` request that dies at the wall, which is the exact failure it was written for after job 189658. |
| **6** | **Nothing compares the adapter's `SEGMENTER_CONTRACT` against the committed one BEFORE the GPU work.** On the shard path the comparison is against `SHARD_OUT`; the real cross-check happens at the merge, an array later. | Already happened: **twelve** of the last sixteen shards ran a pre-`6a32143` adapter (and all sixteen lack `mask_val_ref_max_frac`) because the cluster copy lagged. A whole array of internally-consistent, unusable shards. *Note: defect 3's `rm -f` fix silences the only in-situ detector there is, so 3 must not land without 6.* |
| **7** | **`object_text_prompt` is single-valued while §6 gates two labels.** Unnamed in V6, V9, V10 and T-040. | The entire GPU spend of a plate G0b pass, measured and then disqualified at the cross-check; and a plate pass pointed at the committed document exits 2 with nothing written. |
| **8** | **The pooled `EST_DRIFT` artifact has no carry path at all** — `schema: "wam.est_drift_pooled/1"` vs the required `"wam.est_drift/1"`, plus three more refusals behind it. | The number eight captures produced cannot reach the gate document by any committed tool. Discovered at the last step of a four-step chain. |
| **9** | **`run_v17_arms.sh` skips any artifact that already exists**, and `V17=runs/pr08-est-drift/v17` is a bare assignment with no override flag. | After the flip, a re-run silently skips all thirteen artifacts and leaves the `gate_qualified: false` files in place; the carry then refuses with reasons that no longer describe reality. Remediation is "move the old artifacts aside" — **not** "point it at a fresh dir", which is impossible. |
| **10** | **V5 §5's sample floor is registered as a floor and enforced as a comment.** Demonstrated: a **12-frame** capture yields `headline_valid: true` with only the three flag-related reasons. | Once the two flags flip, `--carry-est-drift` would accept a twelve-frame capture as G0b's committed budget. *(Fixing it is a V-document, not a diff — it changes what `gate_qualified` means after the artifacts exist.)* |
| **11** | **No per-episode refusal memo in `restyle_transfer25.main`'s loop** (`:742-765`). Stage 1 is 8 style-instances over the same 402 episodes. | Each episode's refusal is re-discovered **8 times**, each time after paying full generation. `MaskCache` makes the second instance's *mask* free — only after its generation has already run. |
| **12** | **`FRAMES_PER_VARIANT = 172_000` is coined** (`97_...sbatch:1417`) while the corpus is **171 625** [M], and `partition_facts.json` already carries `corpus_frames` from the manifest (written at `:1252`, computed at `:1166`, both before the timing branch). | Small in size (~0.02 GPU-h at 0.2 s/frame), but it is an invented number inside the one artifact §8 item 3 forbids inventing numbers in. The fix is ordering-valid today. |
| **13** | **The printed recovery command cannot run on the machine that prints it.** `103_...sbatch:457`, `:489` print `git -C ${WAM} checkout -- configs/...`; `sync.sh:66` rsyncs with `--exclude '.git'`. | The operator is sent to run a command that fails with *"not a git repository"* at exactly the moment the pre-commitment has been overwritten. The real recovery is `./cluster/discoverer/sync.sh`. |
| **14** | **Stale operator-facing text in the runbook.** `103_...sbatch:171-172` and `:486` tell the operator `GATE_QUALIFICATION_BLOCKERS` *"still lead with NOBODY HAS LOOKED AT A MASK"*; the tuple is `()` [M] and the look was discharged 2026-08-26. | The page an operator reads *while deciding whether to spend 13.6 GPU-h* is wrong about why the artifact would be disqualified. Same commit as 5 and 13. |
| **15** | **`apple_sam2_video.py:465-468` mirrors only one of `segment()`'s two refusal branches.** | The propagation-vs-per-frame comparison — the evidence that discharged the seventh blocker — becomes a comparison of two different refusal populations on any warm-background capture. **The two arms already disagree 1 vs 0 on the only capture that ran.** *Front 6's proposed fix raises `UnboundLocalError` on the first frame it is meant to count; do not apply it as written.* |
| **16** | **`robot_composite.py:839-843` runs the warm-colour reference over GENERATED pixels with no applicability check**, three days after V10 exported `reference_is_object_scale()` for that call site. Measured 37.18–56.40 % frame coverage on `train-01-oak-tungsten` against a 0.10 bound. | Not a gate (§6 forbids gating on it), but it emits a diagnostic number with no record that the instrument was inapplicable. |
| **17** | **The two artifacts the residue-(i) argument rests on are untracked and overwritable.** `runs/` is gitignored (`.gitignore:19`); `shard-7.json` and `EPISODE_094_CENSUS.json` are not committed, and the census was already overwritten in place once (`.v1`, which lacks the `centroid` field entirely). | A determination citing them cites files that can change without a commit. Force-add the census before any determination quotes it. |
| **18** | **The census artifact still embeds a stale quotation embargo** — `estimator_stats.mask_validity_reference_scope` says V10 is *"UNSIGNED … nothing measured under it may be quoted"* while V10 §8 line 506 reads **ADOPTED 2026-08-24. In force**. Its top-level `estimator_stats` is also a two-pass cumulative total (`n_frames_mask_refused: 62` for a 509-frame episode with 31 per pass). | A reader checking inside the artifact whether it may be quoted gets the wrong answer; a reader quoting "62" against the shard's 36 manufactures a spurious second discrepancy. *(The artifact does carry a `counters_are_cumulative` note adjacent to the 62 — this is a nit, not a trap.)* |

---

## 7. Open and unresolved

Stated plainly, because none of the six fronts settled any of these.

1. **`p_A` does not exist**, and it is the number that decides V16's outcome B vs M. V16's human half has
   never been run to completion (101 of 240 tiles, in the wrong vocabulary, explicitly not evaluated).
2. **Which arm G0b subtracts** (D-C) and **pooled vs single-capture** (D-D). Together they are worth
   0.13653 px on a 0.47858 px tolerance, and neither has plumbing.
3. **`P2` — the generator inventing a robot on a robot-free frame — is unrefuted, not excluded.**
   `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/PROBE.json` returned `verdict: "H"`,
   `candidate_frames: 56` of `paired_frames: 370` on frames selected as robot-free, with
   `human_review.looked_at: false` and the count fixed in advance as an **upper bound**. V12 §3.2's
   frustum witness answers *"was the arm in frame"*, never *"did the generator draw one"*.
4. **The refusal set is partly a property of the machine.** `runs/pr08-bulk-stability/TAIL_SAMPLE.json`:
   of 48 bulk-band frames re-rendered on the RTX 5090, **19 recompute to exactly 0.0** — i.e. ~40 % of
   that sample becomes an *empty mask* on a second GPU. `segmenter_identity()` does not capture the GPU.
   Any pre-flight plan computed on one machine and executed on another inherits this.
5. **The fourth and fifth frames of the 473-vs-478 gap are unrecoverable** from this repository. Rival
   hypotheses tie at 0.165650 px on max|D| and are separated on the mean by ~1e-4 px.
6. **Whether `episode_000094` leaves the corpus** — recorded as an open dataset decision by the blocker-1
   discharge, untouched here.
7. **Whether the pipeline supplies its own conditioning maps.** No producer for pre-computed depth/seg
   maps exists in the repo; `build_pr08_source.py` omits them by declared-temporary design. This decision,
   not a reading of §4 step 2's prose, is what would finally fix which arm is authoritative de facto.
8. **PR-08 §4's *"a G0b margin that only clears under a lower bound is not a pass"* is enforced by no code
   path.** `measure_est_drift` computes `is_lower_bound` / `error_direction` per route; `run_g0_gates.py`
   contains **zero** occurrences of either.
9. **`T40_RULE_V12` §3.2's objection stands untouched:** AppleToPlate's camera intrinsics and extrinsics
   are committed nowhere in this repository.
10. **The `102` re-run and its six `unverified` rows** (sprint §7 step 3) were not investigated by any
    front. **NOT MEASURED.**
11. **Corpus-wide decode bit-identity is NOT MEASURED.** It was measured on **one** episode of 402
    (`episode_000094`, 509/509 frames, four decoder pairings). The provenance caveat it was used to
    retire governs a 171 625-frame pass.
12. **Cluster-side facts nobody may fetch from here [NOT MEASURED — requires a cluster run]:** whether
    `${PROJ}/runs/t040-transfer25-restyle/THROUGHPUT.json` still exists; whether the cluster copy is
    synced to `19826cc`; whether the source manifest still hashes to `a988dd60…`; and the HEAD adapter's
    actual per-frame rate on an H200.
13. **The allocation figure is stale.** Sprint §6's *"~4 879 of 5 000 GPU-h remaining"* is dated
    2026-08-15 and must be re-read before any of §4's submissions is priced against it.

---

## 8. What this sheet's numbers rest on, pinned

Defect 17 above: `runs/` is gitignored (`.gitignore:19`), so the artifacts the load-bearing numbers
come from can change without a commit — and `EPISODE_094_CENSUS.json` was already overwritten in
place once on the night it was written. Two of the four are therefore **force-added to git in the
same commit as this sheet** (`git add -f`), because decision **D-A** quotes them directly:

| artifact | sha256 | tracked |
|---|---|---|
| `runs/pr08-geom-tol/shards/shard-7.json` | `faddc46469e20b25b82eccb4c753c6d075e6c534f56c4bb0ef1a3fb51c727944` | **yes, from this commit** |
| `runs/pr08-operating-point/EPISODE_094_CENSUS.json` | `fb5a04d64ba5eeaea955da169f2d56fae9ae4afef7cc9f9d93e3442b42a5dfff` | **yes**, with its `.sha256` sidecar and the overwritten `.json.v1` |
| `runs/pr08-robot-mask-area/POOLED.json` (2.5 MB) — the 17/402 | `631103a8a97010c4804ac039aecc7fd8425c226c750294335fad5938c35233db` | no — pinned by hash only |
| `runs/pr08-empty-mask-look/MOTION.json` (6.4 MB) — V16 outcome A's death | `11c6188fe55cb71f3297cad0c231d64c19b2d3a378e1dfc013f5fa643b1992a7` | no — pinned by hash only |

The last two are pinned by hash rather than tracked because 8.9 MB into a 33 MB repository is a cost
every future clone pays, and §3's commands re-derive both numbers from them in about a second. **If a
determination is ever written that quotes either of them, force-add it first** — a hash in a document
proves a file has not changed, but it does not stop it from being deleted.
