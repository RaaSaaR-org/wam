# FRONT 4 — the five-frame gap: 473 (shard-7) vs 478 (local census)

**Scope.** Does closing `n_frames_with_centroid: 473` against the local census's `478` require a
cluster run? Answer below. Nothing in this document is a signature, a verdict, or a change to
`GATE_QUALIFIED`. No file under `/home/humanoid/develop/wam` was modified (`git status --porcelain`
empty at start and at end of this work).

**Label key**, used on every claim: **(M)** measured by me now, on this workstation, with the
command shown. **(A)** recorded in a committed artifact, quoted. **(I)** my inference from (M)/(A).

---

## 0. Headline

**The cluster run is NOT required to select a V18 outcome, and the decode hypothesis is refuted.**

1. **(M)** There is **no definitional mismatch.** Both sides count the same quantity — the frames of
   the same 509-frame decode on which `measure_geom_tol.centroid_of_mask(largest_component=True,
   min_area=40)` returns a centroid, applied to the mask `apple_sam2.segment()` returns **after**
   its validity filter. Same function, same arguments, same frame population. The 5-frame gap is
   real.
2. **(M)** The **"cross-machine decode difference" hypothesis is refuted on every route measurable
   here.** Four decode paths — cv2 5.0.0, imageio/FFMPEG, and **pyav (the cluster's own decoder
   family)** — over **both** corpus trees produce **bit-identical** `509 × (480, 640, 3)` uint8 RGB
   arrays for `episode_000094`. 0 differing frames, 0 differing pixels, on all four pairings.
3. **(M/I)** What differs is **the model arithmetic, not the pixels.** On the 16 frames of
   `episode_000094` that both the workstation CPU audit and the cluster shard record, the winning
   box's detection score differs by **mean 0.0188, max 0.0347** — in a region where the top score is
   ~0.2 and the competing box is the plate. Over the 466 steps both instruments measured, the
   centroid displacement differs by **mean 0.0073 px, p95 0.0194 px**.
4. **(M/I)** **The five frames are named**, arithmetically, out of the shard's own displacement
   sequence: **`f101, f108, f124, f152`, plus exactly one of `{f113, f116, f125}`.** The three-way
   residue is an information limit of the shard artifact, not of this analysis.
5. **(M)** **The gap cannot change the V18 outcome.** All seven candidate frames lie inside the
   registered `~f101–f155` window, and all seven carry *below-median* mask area, so under the
   cluster's reconstructed 473-frame kept set outcome E still reads **1.306×** against a 3× bound.
   **V18 §3 selects outcome C — CONTAINED — under the workstation's 31 and under the cluster's 36
   alike.**

---

## 1. Reconcile 473 vs 478 arithmetically first

### 1.1 Are the two numbers counting the same thing? Yes.

**Shard side (A).** `runs/pr08-geom-tol/shards/shard-7.json`, `per_episode[6]`:

```json
{"episode": "episode_000094", "episode_index": 94,
 "clip": ".../data/pr08-apple-640x480/videos/episode_000094.mp4",
 "n_frames": 509, "n_frames_with_centroid": 473,
 "n_steps": 508, "n_steps_measured": 466, "n_steps_dropped": 42,
 "median_px": 1.1177845079762903, "p95_px": 6.22475763523638}
```

`n_frames_with_centroid` is written at `scripts/measure_geom_tol.py:3763`
(`"n_frames_with_centroid": int(sum(c is not None for c in cents))`), where `cents` comes from
`episode_centroids_from_video` (`scripts/measure_geom_tol.py:1786`), whose per-frame body is
(`:1810–1812`):

```python
cents.append(
    centroid_of_mask(method.mask_fn(frame, method), largest_component=True,
                     min_area=min_area)
)
```

`method.mask_fn` is the `sam2_method` wrapper around `estimators.apple_sam2.segment()`, i.e. the
mask **after** the validity filter. `min_area` is `args.min_area_px`, and the shard records
`mask_method.min_area_px: 40` and `mask_method.centroid_rule: "largest connected component by
area"`.

**Census side (A).** `scripts/census_operating_point_episode.py:69` `MIN_AREA_PX = 40`; `:86–89`

```python
def _centroid(mask: np.ndarray) -> list[float] | None:
    got = _centroid_of_mask(mask, largest_component=True, min_area=MIN_AREA_PX)
```

with `_centroid_of_mask` **imported from `measure_geom_tol`** (`:63`), applied at `:126` to
`mask = np.asarray(est.segment(rgb), dtype=bool)` (`:99`) — again the post-validity mask. Written
out at `:139` as `"n_frames_with_centroid"`.

**So: same function, same two arguments, same post-filter mask, same field name.** There is no
"one counts before the filter and one after". **(M)**

### 1.2 Does the shard skip frames? No.

**(A)** `shard-7.json`: `step_frames: 1`, `step_definition: "one step = 1 source frame(s),
overlapping offsets i -> i+1"`, `limit: 0`, `max_frames: 0`, `partial_measurement: false`,
`coverage_scope` measured over decoded steps. No stride, no sampling, no head/tail drop.

### 1.3 Is one over 478 decoded frames and the other over 483? No — 509 both.

| | frames decoded | frames with centroid |
|---|---|---|
| `shard-7.json` `per_episode[episode_000094]` **(A)** | **509** | **473** |
| `EPISODE_094_CENSUS.json` pass 1 (H.264-lossless) **(A)** | **509** | **478** |
| `EPISODE_094_CENSUS.json` pass 2 (AV1) **(A)** | **509** | **478** |
| my re-run, both trees **(M)** | **509** | **478** |

The denominators are identical. **A definitional or population mismatch is excluded. The 5-frame
gap is a difference in what the instrument decided about five specific frames.**

### 1.4 The shard's 36 are all refusals, and all in this episode — airtight, not coincidence

**(M)**, computed over `shard-7.json`:

```
sum over the shard's 25 episodes of (n_frames - n_frames_with_centroid) = 36
the only episode with a non-zero deficit                                = episode_000094 (36)
estimator_stats.this_run.n_frames_mask_refused                          = 36
estimator_stats.this_run.n_frames_without_detection                     = 0
estimator_stats.this_run.n_frames_with_empty_mask                       = 0
```

**(I)** A refusal returns all-False, so every refusal is centroid-less: refusals ⊆ deficit. The two
sets have the same cardinality (36 = 36), so they are **equal**, and no frame of this shard lost its
centroid to the largest-connected-component floor. The census reports the mirror fact directly:
`n_no_centroid_that_are_not_refusals: 0` **(A)**.

### 1.5 The adapter-version hypothesis is dead, with a number

The shard's `mask_method.version` **(A)** ends `...;mask_val_min_iou=0.1` — **no
`mask_val_ref_max_frac` token**. The census's **(A)** ends `...;mask_val_min_iou=0.1;
mask_val_ref_max_frac=0.1`. **(I)** So the shard ran the adapter at `07965aa` (validity filter
present, V10 reference-scope guard absent); V10 landed in `6a32143`, committed
`2026-08-23 18:07:20 +0200`, i.e. **4 h 52 min before** the shard's own
`measured_utc: 2026-08-23T20:59:13+00:00` — the cluster copy lagged HEAD.

**(M)** I diffed the two revisions structurally (AST, function bodies, docstrings stripped):

```
IDENTICAL BODIES: _as_uint8_rgb, _best_box, _cache_probe, _check_packages, _check_pins,
  _depth_pipeline, _detector, _device, _hub_cache_dirs, _importable, _local_files_only,
  _offline_hub, _predictor, _require_cached, _sam2_api_message, available, estimate_depth,
  mask_validity_iou, missing_package_message, missing_weights_message, normalize_prompt,
  object_color_reference, post_process, reset_models
CHANGED: segment, stats
```

`stats()` only adds report fields. The whole behavioural delta in `segment()` is:

```python
+_require_mask_validity_reference()
+MASK_VALIDITY_REFERENCE_FRACTION.append(reference_frame_fraction(reference))
+if not reference_is_object_scale(reference):
+        MASK_REFUSED_FRAMES += 1
+        MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES += 1
+        return np.zeros((h, w), dtype=bool)
```

and every threshold constant is character-identical between the two revisions (`MASK_VALIDITY_MIN_IOU
= 0.10`, `MASK_VALIDITY_REFERENCE`, `BOX_THRESHOLD`, `TEXT_THRESHOLD`, both retry thresholds).

**(M)** The guard is provably inert on this episode: `n_frames_mask_refused_reference_not_object_scale
= 0`, and the **maximum** `warm_reference_frame_fraction` over all 509 frames is **0.02408**
(frame 23) against the bound of **0.10** — a factor of 4.15 of headroom. **(I)** So V10 could not
have fired here even under a moderately different instrument, and in any case it can only *add*
refusals. The module's own reasoning at `scripts/estimators/apple_sam2.py:934` ("the shards ran a
pre-V10 adapter and should refuse fewer rather than five more") is **confirmed, and now carries a
margin instead of a direction.**

---

## 2. Is the corpus here? Yes — both trees. I ran the census on both.

**(M)**

```
/home/humanoid/wam-t041/pr08-apple-640x480/videos/episode_000094.mp4                 2 441 664 B  (AV1)
/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless/videos/episode_000094.mp4  17 730 400 B  (H.264 lossless)
```

Command run (output written to scratch, **not** into the repo):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humanoid/develop/wam/.venv/bin/python \
  /home/humanoid/develop/wam/scripts/census_operating_point_episode.py \
  --episode episode_000094 \
  --corpus /home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless \
  --corpus /home/humanoid/wam-t041/pr08-apple-640x480 \
  --out docs/investigations/2026-08-27-pr08-fronts/EPISODE_094_CENSUS_F4_REPRO.json
```

Output:

```
pr08-apple-640x480-h264-lossless: 509 frames, 31 refused [109, 149] contiguous=False, 0 no-detection, 0 empty
pr08-apple-640x480:               509 frames, 31 refused [109, 149] contiguous=False, 0 no-detection, 0 empty
decode agreement: 31 in both, 0 / 0 only, jaccard 1.0
```

**(M)** Compared row-by-row against the committed `runs/pr08-operating-point/EPISODE_094_CENSUS.json`:
**0 of 509 per-frame rows differ, on either tree**, and the estimator version string is equal. Adding
the `.v1` artifact (`measured_utc 2026-08-27T02:49:10Z`) and the committed one (`02:56:24Z`), this
is now a **three-run determinism control** on one workstation with zero drift.

Refused set, identical on all three runs and both trees **(A/M)**:

```
109 110 111 112 114 115 117 126 127 128 129 130 131 132 133 134 135 136 137 138
139 140 141 142 143 144 145 146 147 148 149            (31 frames, 4 maximal runs)
```

### 2.1 The decode hypothesis, killed directly

**(M)** I decoded `episode_000094` from **both** trees through **three** decoder paths and compared
the arrays element-wise:

| pairing | frames | identical |
|---|---|---|
| `robot_composite._decode_frames` (AV1 → imageio/FFMPEG) vs pyav 18.0.0 (AV1) | 509 | **509 / 509** |
| `_decode_frames` (H.264-ll → cv2 5.0.0) vs pyav 18.0.0 (H.264-ll) | 509 | **509 / 509** |
| AV1 vs H.264-lossless, via `_decode_frames` | 509 | **509 / 509** |
| AV1 vs H.264-lossless, via pyav | 509 | **509 / 509** |

Zero differing frames, therefore zero differing pixels, in every pairing. **pyav is the decoder the
shard itself used** (`shard-7.json` `decoder: {"name": "pyav", "version": "16.0.1", "note": "av.open
+ libswscale to bgr24 … via libdav1d"}` **(A)**), and I drove it with the same
`frame.to_ndarray(format="bgr24")[:, :, ::-1]` conversion `measure_geom_tol._pyav_open`
(`:1685–1705`) uses.

**(I)** The pixels the estimator sees are decoder-invariant and codec-invariant on this clip. The
only decode variable I could not reach is **pyav 16.0.1 vs 18.0.0** (see §6, NOT MEASURED) — and AV1
decoding is normative, so a version bump changing output would itself be a decoder bug.

### 2.2 What *does* differ between the machines

**(A)** `runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json` ran this adapter **on this
workstation, on CPU**, over the H.264-lossless tree, and records 16 frames of `episode_000094` with
per-frame `detection_score`, `mask_area_px`, `warm_apple_iou`. The census ran on this workstation on
the GPU (RTX 5090). **(M)** They agree **exactly** on every mask:

```
frame  audit_area  census_area   audit_iou  census_iou
106    1259        1259          0.8984     0.8984
107    1038        1038          0.8569     0.8569
108     750         750          0.8562     0.8562
109   31154  →  REFUSED (0)      0.0        —          (plate mask, plate_overlap 0.9871)
151     673         673          0.8455     0.8455
152    1040        1040          0.8415     0.8415
153    1370        1370          0.8752     0.8752
154    1579        1579          0.9260     0.9260
```

**(M)** The same 16 frames against the **cluster's** recorded score
(`shard-7.json per_episode[…].detection_scores[i]`, length 509, one per frame,
`n_frames_without_detection = 0` so index == frame):

```
frame  workstation(CPU audit)   cluster(shard)    delta
106    0.366479                 0.360502          -0.005977
107    0.263643                 0.256904          -0.006739
108    0.214168                 0.233650          +0.019482
109    0.205249                 0.231227          +0.025978
129    0.213019                 0.231813          +0.018794
130    0.155249                 0.167157          +0.011908
133    0.231655                 0.245540          +0.013885
134    0.259338                 0.294064          +0.034726
136    0.233837                 0.260455          +0.026618
137    0.231203                 0.256925          +0.025722
143    0.238963                 0.270948          +0.031985
144    0.220040                 0.246194          +0.026154
151    0.223530                 0.216168          -0.007362
152    0.236690                 0.257397          +0.020707
153    0.269737                 0.262371          -0.007366
154    0.331209                 0.314073          -0.017136
                                       n=16  mean|Δ| 0.018784  max 0.034726
```

**(I)** Same bytes in, different number out. The disagreement is in **model evaluation**, not decode.
The plausible carriers are the GPU and the stack: this workstation is `torch 2.13.0+cu130`, CUDA 13.0,
**RTX 5090**, `av 18.0.0`, `cv2 5.0.0` **(M)**; the shard's own artifact records `pyav 16.0.1` and
`cv2 4.11.0` **(A)** and does not record the device, the torch build or a git commit
(`git_commit: null` on all 16 shards **(A)**).

**This is the correction the residue list needs.** `scripts/estimators/apple_sam2.py:893` carries, as
blocker 2 residue (iii): *"The two conjuncts are measured on TWO DIFFERENT DECODES — the full pass on
the AV1 tree, the human look on the H.264-lossless tree — so their per-frame scores must not be quoted
interchangeably."* **The stated reason is false (M): the two trees are bit-identical.** The
*conclusion* survives, for a different reason: the two conjuncts were measured on **two different
machines**, which disagree by up to 0.035 on the very scores in question. (I did not edit the file.
Correcting it is a new version document, per `docs/handoff.md` §3.)

---

## 3. Does shard-7 record enough to name the five? Yes — indirectly, and I named four of them

The shard records **no** per-frame centroid flag — that is exactly limit 2 of
`PR-08-RESULT-2026-08-25-operating-point-on-this-corpus.md` **(A)**. But it records something
equivalent: `per_episode[…].displacements_px`, **466 values, measured steps only, in order** (M:
`len(displacements_px) == n_steps_measured` on every episode of the shard). Which steps are *absent*
from that list is a per-frame flag in disguise.

**Method (M).** Take the census's 478 centroids. For a hypothesised missing-frame set `M`, the step
`i → i+1` is measurable iff neither endpoint is in `M`; predict `‖c[i+1] − c[i]‖`. Align the shard's
466 values against the census's 473 predicted values as a subsequence (DP, minimising Σ|Δ|). The
alignment skips exactly **7** predicted steps:

```
skipped steps: 100, 101, 107, 123, 124, 151, 152      (cost 3.4247 over 466 matches)
```

Steps `{100,101}` ⇒ frame **101** missing; step `{107}` alone (step 108 already dropped by the
census's own `109`) ⇒ frame **108**; `{123,124}` ⇒ frame **124**; `{151,152}` ⇒ frame **152**.

**Verification, element-wise and positional (M).** With `M = census31 ∪ {101, 108, 124, 152}`:

```
steps produced      466   ==   shard n_steps_measured 466        ✓
mean |Δ|            0.007349 px      p95 0.019365 px      max 0.165650 px
reconstructed median 1.113665 px  vs shard median_px 1.117785 px  (0.37 %)
reconstructed p95    6.215901 px  vs shard p95_px    6.224758 px  (0.14 %)
```

against an episode median displacement of ~1.12 px — the residual is two orders of magnitude below
the signal. **Decoys (M)**, each perturbing one frame of the hypothesis, all fit worse and fail
*locally at the perturbed step*, which is the signature of a correct identification:

```
{101,108,124,152}  mean 0.007349  max 0.165650 (step 385)   ← best on both statistics
{102,108,124,152}  mean 0.007954  max 0.281719 (step 100)
{101,108,123,152}  mean 0.009364  max 0.940631 (step 124)
{101,108,124,151}  mean 0.008800  max 0.689000 (step 152)
{100,108,124,152}  mean 0.009546  max 1.023629 (step 101)
{101,107,124,152}  length mismatch — cannot produce 466 steps
{101,108,125,152}  length mismatch — cannot produce 466 steps
```

**The independent arithmetic check that closes it (M).** For a missing set with `r` maximal interior
runs, dropped steps `= |M| + r`.

```
census 31 frames, 4 runs  →  35 dropped                     (shard's own field says 42)
census31 ∪ {101,108,124,152} = 35 frames in 7 runs  →  42   ==  shard n_steps_dropped 42   ✓
```

But the shard says **36** frames, not 35. **(M)** Solving for every frame set consistent with the
recovered dropped-step set: 471 of 509 frames are *forced present* (some step touching them survives
in the shard's list); the 38 free frames admit exactly one more missing frame, and it must be one of
the three that sit **between two already-missing neighbours** and are therefore invisible to any
step-level evidence:

> **`f113`** (between 112 and 114), **`f116`** (between 115 and 117), **`f125`** (between 124 and 126).

### The answer to question 3

**The cluster's 36 refused frames are the census's 31, plus `f101`, `f108`, `f124`, `f152`, plus
exactly one of `{f113, f116, f125}`.** Four are named with certainty; the fifth is a genuine
information limit of `shard-7.json`, not of this analysis — no artifact in this repository can
distinguish those three.

Context for all seven **(A)**, from the census's own per-frame rows — every one is a frame the
workstation kept with a *confident* apple mask, and the cluster refused:

```
frame  census area  census validity IoU   cluster detection score
101    2677         0.94980               0.235876
108     750         0.85620               0.233650
113     646         0.80996               0.197974
116    1074         0.89890               0.228094
124     654         0.87832               0.224536
125     599         0.88379               0.217656
152    1040         0.84150               0.257397
```

**(I)** Nothing marginal about the *validity* decision (IoU 0.81–0.95, threshold 0.10 — a 8–9×
margin). What is marginal is the *detection*: across `f101–f160` the winning box scores 0.166–0.422
**(A)** where the episode's normal score is ~0.85, and the runner-up in that window is the plate
mask the audit documents at ~31 000 px with plate overlap 0.985–0.992 **(A)**. A ±0.02 shift in the
detector's head — exactly the cross-machine magnitude measured in §2.2 — is enough to swap the
top-1 box from the fruit to the plate, whereupon the validity filter refuses it. **That is the
mechanism, and it is a mechanism residue (i) already describes; the five frames are five more
instances of it, not a new failure.**

---

## 4. Is a cluster census still required? — Not for the decision. Optionally, for the name.

**No, for the thing that matters.** The gap cannot change the V18 outcome:

* **(M)** All seven candidate frames (`101, 108, 113, 116, 124, 125, 152`) lie inside the
  registered `~f101–f155` window, so outcome **U** cannot be reached by any resolution of the
  ambiguity.
* **(M)** All seven carry mask area *below* the episode median (599–2 677 px against 5 650 px), so
  removing them **raises** the median and **raises** the 3× bar, while the maximum non-refused area
  is unchanged at 7 383 px. Recomputing outcome E on the cluster's reconstructed 473-frame kept set,
  for each of the three ambiguity branches:

  ```
  extra {…,113}: n kept 473  median 5652  max 7383  ratio 1.306  frames over 3x: []
  extra {…,116}: n kept 473  median 5652  max 7383  ratio 1.306  frames over 3x: []
  extra {…,125}: n kept 473  median 5652  max 7383  ratio 1.306  frames over 3x: []
  ```

**(I) V18 §3 selects outcome C under both instruments.** The 473/478 disagreement is a fact about
the *instrument's reproducibility across machines*; it is not a fact that moves V18's outcome table,
and it is not a route by which a wrong-object mask enters the measured population — it is the
opposite, the cluster refused *more*.

**Yes, if the project wants the fifth frame named and the cluster's own count reproduced.** That is
a documentation question, not a gate question, and it is the owner's to spend GPU-hours on. If it is
run, the submission must satisfy two preconditions **(I)**, or it measures something else:

1. `/valhalla/.../wam` must be synced to the revision under test, and the artifact must be read with
   the knowledge that **HEAD's adapter is post-V10 while the shard's was pre-V10** — inert here
   (§1.5), but the two artifacts then carry different `ESTIMATOR_VERSION` strings and
   `run_g0_gates.instrument_disagreements` compares that string.
2. It reproduces the *cluster's* count only if it lands on the same GPU class and stack as the
   2026-08-23 shard job; `git_commit: null` and no recorded device on all 16 shards **(A)** means
   that is not pinned and cannot be guaranteed. A cluster census that returns 31 would therefore
   also be a legitimate result, and would say the 36 was a property of that job rather than of the
   cluster.

**Exact submission (NOT run by me — I do not touch the cluster):**

```bash
# on the login node, by a human, after `rsync` of the repo to /valhalla/.../wam
sbatch --qos=<the project's qos> --time=00:25:00 --gres=gpu:1 --wrap="\
  python scripts/census_operating_point_episode.py \
    --episode episode_000094 \
    --corpus /valhalla/projects/ehpc-aif-2026pg01-905/data/pr08-apple-640x480 \
    --corpus /valhalla/projects/ehpc-aif-2026pg01-905/data/pr08-apple-640x480-h264-lossless \
    --out runs/pr08-operating-point/EPISODE_094_CENSUS_CLUSTER.json"
```

**Cost (I, extrapolated from the local wall clock — NOT MEASURED on the cluster):** 1 018 segment
calls plus model load. The two committed local artifacts are 7 min 14 s apart and each is a full
two-tree pass, so ≈7 min on an RTX 5090; a 25-minute walltime request is ample backfill-friendly
padding. **≈0.2–0.4 GPU-h.** The `--qos` flag is mandatory (omitting it traps the job at 1 min /
0 GPUs).

**What such a run would and would not buy.** It would name the fifth frame and turn "36" from a
number into 36 indices. It would **not** close the underlying phenomenon, because the phenomenon is
cross-machine non-reproducibility of a marginal detector, and one more cluster run is one more
sample of one machine.

---

## 5. Mapping onto V18 §3 — which fixed outcome the evidence selects

I read `docs/preregistration/PR-08-V18-residue-i-decision-rule.md` §3 in full. Its three outcomes,
against the evidence, with **no outcome invented and nothing signed**:

**Outcome U — UNDECIDABLE. Does not fire.** Its trigger is *"The census finds **zero** refusals on
`episode_000094`, or finds them outside that episode's documented low-score run of ~f101–f155."*
**(M)** The census finds 31 refusals, span `[109, 149]`, wholly inside `f101–f155`; and the five
frames this front recovers (`101, 108, 124, 152`, + one of `113/116/125`) are also inside it.
Neither disjunct is satisfied. *Note for the reader, offered as fact rather than as an outcome:*
U's prose contains the observation *"the shard pass attributed all 36 of the corpus's refusals to
this episode"*, and the census found 31, not 36. **That 31-vs-36 disagreement is not one of U's two
stated triggers, so it does not select U** — but it is material and §4 above is what is now known
about it.

**Outcome E — IT ESCAPES. Does not fire.** Its trigger is *"Any frame of the episode carries a mask
that is **not** refused and whose area exceeds three times the episode's own median non-refused mask
area."* **(M)** median non-refused area 5 650.5 px; 3× = 16 951.5 px; largest non-refused mask
7 383 px at frame 82 = **1.307×**; frames over the bound: **none**, on both decode trees, on all
three local runs, and on all three branches of the cluster reconstruction (1.306×).

**Outcome C — CONTAINED. This is the outcome the evidence selects.** "Neither of the above." Every
non-refused mask is of apple-plausible area and the refusals fall inside the documented low-score
run. **(I)** And this front adds a robustness statement V18 could not have asked for: **C is
selected identically under the workstation's 31-refusal reading and under the cluster's 36-refusal
reading**, so the five-frame gap does not put the outcome in play.

**What C obliges the determination to carry — and what this front changes about items 1 and 2.**
V18 §3 outcome C requires four items in the determination's own text. They are already carried in
`docs/preregistration/PR-08-RESULT-2026-08-27-the-rate-is-bounded-and-the-per-frame-arm-is-the-one-that-broke.md`
§6 **(A)**. Two of them now have more behind them, and a new version document (never an edit) is the
place for it **(I)**:

* **item 1** — *"the refusal count and the frame indices, on both decodes, and their disagreement"*:
  the two decodes have **no** disagreement, and now for a measured reason that covers the cluster's
  own decoder as well (§2.1). The disagreement that exists is cross-machine, and its frame indices
  are §3's list.
* **item 2** — *"this is one episode of 402 … the corpus pass recorded no per-frame flag to census
  against"*: strictly true of the recorded *flags*; §3 shows the corpus pass's **displacement
  sequence** is a per-frame flag in disguise and can be inverted for any episode where a census
  exists. That does not widen containment to the corpus — 401 episodes still have no census — but it
  means the method exists.
* **items 3 and 4** stand untouched. **Nobody has looked**; the area test is a proxy; and the failure
  blocker 2 predicted did occur by an unpredicted route. This front is arithmetic and adds no look.

**I am signing nothing.** Whether outcome C is accepted, and whether the fifth frame is worth a
cluster job, is the project owner's call.

---

## 6. NOT MEASURED — the honest edges

* **NOT MEASURED — requires a cluster run.** The cluster's own per-frame refusal indices, its mask
  areas, and whether it reproduces 36 today. Everything in §3 is a reconstruction of the 2026-08-23
  job from its recorded displacement sequence, not a re-execution.
* **NOT MEASURED — the fifth frame.** `f113` vs `f116` vs `f125` cannot be separated by any artifact
  in this repository; all three sit between two already-missing neighbours and contribute no step.
* **NOT MEASURED — pyav 16.0.1.** I measured `av 18.0.0`; the shard recorded 16.0.1. Both route AV1
  through libdav1d and both convert with libswscale, and AV1 decoding is normative, so I expect no
  difference — **that is an expectation, not a measurement.**
* **NOT MEASURED — the cluster's device and stack.** `shard-7.json` records no device, no torch
  build and `git_commit: null`, so "which GPU produced the 36" is not recoverable.
* **NOT MEASURED — whether the workstation or the cluster is *right* about those five frames.**
  Nobody has looked at them. This is limit 3, it is untouched, and V18 §3 outcome C item 3 says so.
* **(I) Attribution caveat on §2.2.** The audit ran on **CPU** and the shard on a cluster **GPU**, so
  the measured score delta is "workstation-CPU vs cluster-GPU". It is not decomposed into
  device-vs-machine. What it *does* exclude, by §2.1, is decode.

---

## 7. Defects found

**D1 — a committed record states a refuted reason.** `scripts/estimators/apple_sam2.py:893`, blocker
2 residue (iii): *"measured on TWO DIFFERENT DECODES … so their per-frame scores must not be quoted
interchangeably."* **(M)** The two decodes are bit-identical on this episode across three decoder
families. The prohibition is right; its reason is wrong, and the true reason (different machines,
mean |Δscore| 0.0188) is a stronger one. A reader who acted on the stated reason — "re-run both
conjuncts on one tree and the caveat lifts" — would spend a job and lift nothing.
**Fix: a new version document, per `docs/handoff.md` §3. NOT applied; I did not edit the file.**

**D2 — `EPISODE_094_CENSUS.json`'s `estimator_stats` is a two-pass total with no note saying so.**
`scripts/census_operating_point_episode.py:221` writes `est.stats()` after **both** passes, so the
artifact reads `n_segment_calls: 1018`, `n_mask_validity_iou: 1018`, **`n_frames_mask_refused: 62`**
**(A)** — 2 × 509 frames of *one* 509-frame episode. `shard-7.json` guards precisely this with a
`process_local_counters_note` and a before/after snapshot **(A)**; the census has neither. A reader
quoting "62 refusals in episode_000094" against the shard's 36 would manufacture a second, larger,
entirely spurious discrepancy. The per-pass blocks are correct — only the top-level total misleads.
**Proposed fix (NOT applied):** annotate the field, or snapshot deltas per pass as `measure_geom_tol`
does.

**D3 — the census does not record the one quantity that would have made §3 unnecessary.** The shard
records `detection_score` per frame per episode; the census does not, so a cluster-vs-workstation
comparison has to be reconstructed from displacements instead of subtracted. Proposed diff, **NOT
applied** — it goes into a scratch deliverable, not into the tree:

```diff
--- a/scripts/census_operating_point_episode.py
+++ b/scripts/census_operating_point_episode.py
@@ -94,12 +94,22 @@ def census(episode: str, corpus: pathlib.Path, est: Any) -> dict[str, Any]:
     rows: list[dict[str, Any]] = []
     before = {k: est.stats().get(k, 0) for k in EVENT_COUNTERS}
+    n_scores_before = len(est.DETECTION_SCORES)
     for index, rgb in enumerate(_decode_frames(video)):
         mask = np.asarray(est.segment(rgb), dtype=bool)
         after = {k: est.stats().get(k, 0) for k in EVENT_COUNTERS}
         fired = [k for k in EVENT_COUNTERS if after[k] > before[k]]
         before = after
+        # The winning box's score for THIS frame, or None when no box was found. It is the ONE
+        # per-frame quantity the 16-shard GEOM_TOL artifacts already carry
+        # (per_episode[].detection_scores, one entry per frame when n_frames_without_detection
+        # is 0), so recording it here makes a cluster-vs-workstation comparison a subtraction
+        # instead of a reconstruction out of the displacement sequence. Read as a DELTA on the
+        # module's append-only list, for EVENT_COUNTERS' reason: the totals are cumulative over
+        # the import and two passes over one episode share them.
+        n_scores_after = len(est.DETECTION_SCORES)
+        detection_score = (
+            float(est.DETECTION_SCORES[-1]) if n_scores_after > n_scores_before else None
+        )
+        n_scores_before = n_scores_after
         reference = est.object_color_reference(rgb)
         rows.append(
             {
                 "frame": index,
+                "detection_score": detection_score,
                 "mask_area_px": int(mask.sum()),
                 "has_mask": bool(mask.any()),
```

**Not a defect, recorded so it is not re-found:** the census's `MIN_AREA_PX = 40` genuinely matches
the shard's `mask_method.min_area_px: 40` **(A)**; the census's `_centroid` genuinely calls
`measure_geom_tol.centroid_of_mask` **(A)**; and `_decode_frames` genuinely yields RGB, matching
`sam2_method`'s `"color_order_in": "RGB — cv2 decodes BGR and the adapter is handed frame[:, :, ::-1]"`
**(A)**. All three were candidate explanations for a 5-frame gap and all three are excluded.

---

## 8. Artifacts this front produced

| path | what |
|---|---|
| `docs/investigations/2026-08-27-pr08-fronts/F4-residue-i-473-vs-478.md` | this document |
| `docs/investigations/2026-08-27-pr08-fronts/EPISODE_094_CENSUS_F4_REPRO.json` | third independent local census, both trees, 0/509 rows differing from the committed one |

`git status --porcelain` is empty. No repository file was created, modified or deleted.

---

## Adversarial re-read

**Reviewer:** a second session, tasked to refute rather than to agree. Everything below was re-run
from the artifacts on this workstation, at `HEAD = 19826cca` (`git status --porcelain` empty at
start and at end of this re-read; no file under `/home/humanoid/develop/wam` was created, modified
or deleted). Same label key: **(M)** measured now, **(A)** read from an artifact, **(I)** inference.

**Verdict: the headline does not survive as written.** Two load-bearing claims fail — the
identification of the fifth-from-last frame (§3, "four are named with certainty") and the
"MEASURED" status of the V18-invariance claim (§4/§5). The *decision* — that the registered V18 §3
outcome is C and that no cluster run is required to reach it — survives, but on a narrower ground
than the document argues, and three of the four "defects found" are overstated or wrong.

### What I reproduced exactly, and hereby second

All of the following I re-ran independently and got the document's numbers to the digit. **(M)**

| claim | document | my re-run |
|---|---|---|
| decode bit-identity, 4 pairings, `episode_000094` | 509/509 each | 509/509 each, `max abs diff = 0` cross-tree; `av 18.0.0`, `cv2 5.0.0`, RTX 5090, `torch 2.13.0+cu130` all confirmed |
| shard deficit is one episode and equals its refusals | 36 / 36 / ep 94 | `sum(n_frames − n_frames_with_centroid) = 36`, only `episode_000094`; `this_run.n_frames_mask_refused = 36`, `without_detection = 0`, `with_empty_mask = 0` |
| census | 509 / 478 / 31 refused `[109,149]`, jaccard 1.0, `n_no_centroid_that_are_not_refusals = 0` | identical, both trees |
| outcome E on the census | median 5650.5, max 7383 @ f82, **1.3066×** | identical |
| cross-machine detection-score delta, n=16 | mean 0.018784, max 0.034726 | identical to 6 dp |
| audit(CPU) vs census(GPU) agree on every mask | yes | yes — areas equal, IoU equal to 1e-4; `MODEL_OBSERVATIONS.json.how_they_were_produced` does say `WAM_PR08_DEVICE=cpu` |
| run arithmetic `dropped = |M| + r` | 31+4→35, 35+7→42, 36+6→42 | identical |
| element-wise fit of `{101,108,124,152}` | n 466, mean 0.007349, max 0.165650, median 1.113665, p95 6.215901 | identical (`np.percentile(...,95)` linear reproduces 6.215901 exactly) |
| AST diff vs `07965aa` | `CHANGED: segment, stats` | identical, and `07965aa` is the *only* committed adapter revision that is post-validity-filter and pre-V10 (`git log -- scripts/estimators/apple_sam2.py`: `07965aa` 2026-08-22T20:56 → `6a32143` 2026-08-23T18:07+0200 = 16:07 UTC, 4 h 52 min before `measured_utc 2026-08-23T20:59:13Z` — the timezone arithmetic in §1.5 is correct) |
| my own re-run of the census vs the committed one | 0/509 rows differ | 0/509 rows differ on **both** passes, `ESTIMATOR_VERSION` equal |

Sections 1.1–1.4 and 2.1–2.2 are sound and I could not break them.

### R1 — REFUTED, load-bearing. "Four are named with certainty" is false: `f152` is not separable from `f153` or `f154`

§3 tests **five** hand-picked decoys, all of which move a frame *earlier*, into a neighbourhood
where the local error blows up. It never tests moving `f152` *later*. I ran the search the document
should have run. **(M)**

* a full single-substitution scan (each of the four replaced by every frame 1–507): **1396**
  feasible hypotheses;
* an exhaustive enumeration of **all** 4-subsets of the non-refused frames in `f95–f174`:
  **68 280** feasible hypotheses.

Ranked by the document's own two statistics:

```
hypothesis              mean |Δ|    max |Δ|     argmax
(101, 108, 124, 152)    0.007349    0.165650    same step
(101, 108, 124, 153)    0.007464    0.165650    same step     <-- TIED on max
(101, 108, 124, 154)    0.007523    0.165650    same step     <-- TIED on max
(102, 108, 124, 152)    0.007954    0.281719
(103, 108, 124, 152)    0.007970    0.281719
```

`{101,108,124,153}` and `{101,108,124,154}` are **exactly tied** with the accepted hypothesis on
`max |Δ|` — 0.165650 px, attained at the *same* far-away step, i.e. neither is penalised anywhere —
and are separated from it on the mean by **0.000115 px and 0.000174 px**. Locally, where §3 says a
wrong hypothesis "fails at the perturbed step": **(M)**

```
                       hyp {…,152}   hyp {…,153}   hyp {…,154}
worst local residual      0.0193        0.0728        0.0728
                                        (2.3x BELOW the accepted hypothesis's own 0.165650 floor)
```

The accepted hypothesis carries a 0.1657 px residual of its own somewhere in the sequence. A rival
whose worst residual is 0.0728 px is **inside that noise floor**, not outside it. The discriminating
signal the document claims ("two orders of magnitude below the signal") exists for the *fit*, not
for the *choice between fits*.

**Consequence.** The artifact names `f101`, `f108`, `f124`, **one of `{f152, f153, f154}`**, and one
of `{f113, f116, f125}` — at least **nine** admissible frame sets, not three. (`f102`/`f103` are
also only weakly excluded: max 0.2817 against an accepted floor of 0.1657, a factor of 1.7, not a
separation.) The headline sentence "the five frames are named" and §3's "Four are named with
certainty" are **withdrawn by measurement**. §6's "NOT MEASURED — the fifth frame" understates the
residue by a factor of three.

**What this does NOT change (M):** the candidate set for the *sixth* frame is `{113,116,125}` under
all three fourth-frame branches (checked); `f153` and `f154` carry census areas 1370 px and 1579 px,
also below the 5650.5 median. So R1 does not move the V18 outcome. It refutes an identification
claim, not the decision.

### R2 — REFUTED, load-bearing mislabel. "MEASURED NOW — V18 §3 outcome is invariant to the gap" is an inference, and the areas in it are the wrong machine's

`shard-7.json` records **no per-frame mask area**. Verified per-episode keys **(A/M)**:

```
episode, episode_index, clip, n_frames, n_frames_with_centroid, n_steps,
n_steps_measured, n_steps_dropped, median_px, p95_px, displacements_px, detection_scores
```

So the line *"Recomputed on the cluster's reconstructed 473-frame kept set: median 5652, max 7383,
ratio 1.306"* computes the cluster's **index set** against the **workstation's areas**. The cluster's
own non-refused mask areas are unrecorded and unrecoverable from this repository. That matters here
more than it usually would, because the front's own §3 mechanism is that a ±0.02 score shift swaps
the winning box from the fruit to the plate — a ~31 000 px mask, **≈5.5×** the median, which is
outcome **E**. Whether that happened on a *non-refused* cluster frame is exactly what the shard
cannot say: a wrong-object mask that is *not* refused yields a centroid and a displacement and
therefore contributes **nothing** to the 36-frame deficit the whole reconstruction is built on.

The label on that claim in the summary is **"MEASURED NOW"**. It is **(I)**.

**The argument that would have worked, and is not made.** The shard's 466 displacements match the
workstation's to `max 0.166 px`, which pins the cluster's centroids to the workstation's on the
**471 forced-present frames** (I recomputed the forced-present set: 471, leaving 38 free frames, of
which 36 are the missing ones — so exactly **2** non-refused frames are unconstrained). A plate mask
moves the centroid by hundreds of pixels, so on those 471 frames a plate escape is excluded *by the
displacements*, not by the areas. That argument covers 471 of 473 kept frames and is far stronger
than the one in §4. It should replace it.

**Does the decision survive?** Yes, on a narrower ground the document does not lean on: V18 §2
defines the instrument (*"It writes per-frame mask area"* — the census) and §3's outcomes are
written against it, so outcome E was never a question about the cluster. The **decision** survives;
the **stated reason** does not.

### R3 — D2 is wrong on its central point

D2: *"`EPISODE_094_CENSUS.json`'s top-level `estimator_stats` is a TWO-PASS cumulative total **with
no note saying so**."* The artifact's `estimator_stats` contains, in the same dict, three keys after
the `62` **(A)**:

> `"counters_are_cumulative": "every n_* count above, and n_detection_scores, is a total since this
> module was imported. Nothing resets them, so two measurements driven from one interpreter share
> them. A caller wanting THIS RUN's numbers snapshots them before the pass and differences
> afterwards, as measure_geom_tol.EstimatorStatsProbe does; a caller that copies them verbatim is
> recording a lifetime total and should say so in its artifact."*

The hypothetical reader who "quotes 62 against the shard's 36 and manufactures a spurious
discrepancy" has to walk past a warning printed adjacent to the number. What is genuinely missing is
narrower — the artifact does not say *"two passes over one episode"* — and the severity claim in D2
does not survive. **D2 downgraded from a defect to a nit.**

### R4 — D1 is overstated three ways

1. **The retired wording's factual claim is true.** *"The two conjuncts are measured on TWO
   DIFFERENT DECODES — the full pass on the AV1 tree, the human look on the H.264-lossless tree"* is
   a statement of **provenance**, and it is correct: they were. Bit-identity refutes an *implication*
   the sentence does not make. "States a REFUTED reason" is not what was measured.
2. **Scope.** Bit-identity is measured on **one episode of 402**. The conjunct the caveat governs is
   the 171 625-frame full pass. Corpus-wide bit-identity is **NOT MEASURED** and is not implied by
   one clip.
3. **Not new.** The committed census already carries `decode_agreement.meaning` **(A)**: *"The two
   decodes are the same recorded pixels through two codecs"*, with `jaccard 1.0`. D1's novelty is
   pixel-level rather than refusal-level.

The *actionable* half of D1 — that the stronger reason is cross-machine, mean |Δscore| 0.0188 —
stands, and the prohibition stands. The characterisation does not.

### R5 — the two line citations are wrong at HEAD

**(M)** at `HEAD` (19826cca) and at `16f0a07`, the commit that landed this text:

* `scripts/estimators/apple_sam2.py:893` — **blank line**. The quoted residue (iii) sentence is at
  **886–889** (`grep -n "TWO DIFFERENT DECODES"` → 887).
* `scripts/estimators/apple_sam2.py:934` — the `docs/preregistration/PR-08-RESULT-2026-08-27-…md`
  URL. The quoted *"the shards ran a pre-V10 adapter and should refuse fewer rather than five more"*
  is at **931**.

Both appear in the summary's `defects_found` and in §1.5/§7. D1 proposes a **new version document**;
a version document that cites a blank line is a version document that ages badly. (V18 §5's own
`846` cite is likewise stale — that is not this front's doing, but it is the same failure and worth
naming in whatever document lands.)

### R6 — the `(A)` label is wrong for the two artifacts the whole front rests on

`(A)` is defined in the document as *"recorded in a **committed** artifact, quoted."* **(M)**
`.gitignore:19` ignores `runs/`, and 50 files under `runs/` are force-added. `git ls-files
--error-unmatch`:

```
runs/pr08-geom-tol/shards/shard-7.json                 tracked = NO
runs/pr08-geom-tol/pr08_geom_tol.json                  tracked = NO
runs/pr08-operating-point/EPISODE_094_CENSUS.json      tracked = NO
runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json tracked = YES
```

Of the three artifacts this front quotes as `(A)`, **only `MODEL_OBSERVATIONS.json` is committed**.
The census and the shard are untracked working-tree files with no commit and no history. That is not
a hypothetical: `EPISODE_094_CENSUS.json.v1` proves the census was already **overwritten in place**
the same night — `02:49:10Z` → `02:56:24Z` — and the older file has no `.sha256` sidecar. On a
project whose stated method is *"a claim is worth nothing without the artifact it was measured
from"*, `(A)` on an unversioned file is the (a)/(b) confusion the label key exists to prevent.
Nothing follows about the *numbers*; something follows about how they should be described, and about
whether the artifact V18's outcome rests on ought to be force-added before it is cited in a
determination.

### R7 — the "three-run determinism control" is a two-run control on the quantity in dispute

**(M)** `EPISODE_094_CENSUS.json.v1` has **8** per-frame keys; the current artifact has **9**. The
missing one is **`centroid`** — the field that carries the 478. So `.v1` cannot corroborate 478 at
all. It agrees on the 8 keys it does have (I checked: 0/509 rows differ on the common keys,
`n_refused` 31, `n_frames` 509), which is worth something — but the determinism control **on the
number this front is about** is *two* runs, not three. §2's sentence should say so.

### R8 — new defect the front did not find (D4): the census artifact carries a stale quotation embargo

**(A)** `runs/pr08-operating-point/EPISODE_094_CENSUS.json` → `estimator_stats.mask_validity_reference_scope`
opens:

> *"PR-08 V10 (T40_RULE_V10, docs/preregistration/PR-08-V10-mask-validity-reference-scope.md,
> **UNSIGNED as this is recorded — nothing measured under it may be quoted until the project owner
> signs it**)."*

That string is emitted by the adapter's `stats()` and embedded verbatim in **every** artifact
measured under the post-V10 adapter — including the census this front, `PR-08-RESULT-2026-08-27-…`,
and `apple_sam2.py`'s own `GATE_QUALIFIED` comment all quote. **(A)** V10 §8 line 506 reads
`| status | **ADOPTED 2026-08-24.** In force |`, and V10's header blockquote (lines 6–12) resolves
the `UNSIGNED` banner at line 3 explicitly (*"The header sentence above records this document's
state as drafted; §8 records its state now"*). So the embargo is stale — but it is *inside the
artifact*, where a reader checking whether the census may be quoted will find it and get the wrong
answer. Fixing it is an adapter edit and therefore a version document, not an edit here.

**Related process point.** The summary's `blocking_facts` asserts this V10-in-force finding as a
blocking fact. **It appears nowhere in the deliverable** (`grep -n "UNSIGNED\|in force\|506"` → no
hits). I verified it is *true*; it is nonetheless a summary-only claim, and "is this rule in force"
is exactly the kind of claim that belongs in the document a person reads.

### R9 — the one cluster proposal is confounded by its own Precondition 1

§4's Precondition 1 requires `/valhalla/.../wam` to be synced to *"the revision under test"* — i.e.
HEAD, a **post-V10** adapter — while the artifact it is meant to reproduce was produced by a
**pre-V10** one. The job would therefore vary **machine and code at once**, and could not answer
"does the cluster still produce 36" *for the instrument that produced 36*. To isolate the machine
the cluster would have to run `07965aa`, which Precondition 1 as written forbids.

**Mitigation, in fairness (M):** the V10 delta really is inert, and for a better reason than §1.5
gives. `reference_frame_fraction` is computed from `object_color_reference(rgb)` — **a function of
the decoded pixels alone**, not of the model — and the pixels are bit-identical across trees and
machines. Max over all 509 frames is 0.024079 against the 0.10 bound. So the guard cannot fire on
the cluster either, by construction rather than by hope. The confound is nominal — but
`ESTIMATOR_VERSION` would still differ, which is the very string §4 notes
`run_g0_gates.instrument_disagreements` compares.

### Checks that came back NEGATIVE — nothing to report

* **No committed contract is broken by D3.** `grep -rn "operating_point_census"` has exactly one hit
  in the tree: the `SCHEMA` constant in the script itself. Nothing reads it. D3 does not touch
  `configs/transfer25/pr08_geom_tol.json`'s `contract_fields`, `SEGMENTER_CONTRACT`,
  `MASK_VALIDITY_MIN_IOU`, any threshold, or any landed GEOM_TOL artifact. **(M)**
  *One flaw D3 does carry:* `SCHEMA = "wam.operating_point_census/1"` is already written unchanged
  into two artifacts with **different row shapes** (`.v1` 8 keys, current 9) — the version string
  already fails to distinguish them, and D3 would add a tenth key under the same string without
  bumping it. The diff should bump the schema.
  `est.DETECTION_SCORES` exists (`apple_sam2.py:1146`) and is appended exactly once per frame that
  finds a box (`:1956`, inside `_best_box`, with the alignment comment at `:1955`), so the delta
  logic in the diff is sound.
* **No `runnable_now` command touches the cluster, costs money, or mutates the repo.** All five are
  read-only against `runs/`, `configs/` and local corpora; the census re-run writes only to scratch;
  command 5 writes `/tmp/old_adapter.py`, outside the repo (though also outside the mandated scratch
  directory). I executed four of the five. **(M)**
* **No ordering error.** `gate_qualified: False` is baked into the census artifact (`"gate_qualified":
  bool(getattr(est, "GATE_QUALIFIED", False))`, `census_operating_point_episode.py:~200`), but the
  census is precisely the pre-flip instrument V18 registered, and V18 §4 says outcome C does not flip
  the flag. Nothing here measures something that would have to be re-measured after a flip. **(M)**
* **No gate is rewritten after seeing its output.** The document signs nothing, edits no
  preregistration, proposes new version documents rather than edits, and leaves `GATE_QUALIFIED`
  and `GATE_QUALIFICATION_BLOCKERS` untouched. §5's narrow reading of outcome U — that the 31-vs-36
  disagreement is not one of U's two stated triggers — is textually correct against V18 §3 as
  registered, and the document flags the tension rather than burying it. I have no objection to
  raise here. **(M/I)**
* **The AST method under-reports, harmlessly.** `[k for k in set(a)&set(b) if a[k]!=b[k]]` compares
  the **intersection** only, so functions present in one revision and not the other are invisible.
  HEAD adds four: `_require_mask_validity_reference`, `mask_validity_reference_is_defined`,
  `reference_frame_fraction`, `reference_is_object_scale` **(M)**. §1.5 describes all of their
  effects correctly in prose, so nothing is wrong — but the command in `runnable_now` prints an
  incomplete answer, and `ONLY IN NEW` should be printed alongside `CHANGED`.

### What I would tell the owner

The front's **decision** — outcome C, and no cluster run needed — I could not break, and its core
measurements reproduced to the digit. But three of the things it says it *knows*, it does not know:
the fourth frame is `f152` **or** `f153` **or** `f154`; the V18-invariance recomputation is an
inference resting on the wrong machine's areas (and a better argument, from displacements, is
available); and two of its three cited artifacts are not committed. D2 is a nit rather than a
defect, and D1's characterisation is wrong even though its prohibition is right. Read §3's
identification and §4's invariance as hypotheses, not as measurements.
