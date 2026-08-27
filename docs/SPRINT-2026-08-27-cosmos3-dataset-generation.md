# Sprint — what is left before the photoreal-augmentation corpus can be generated

Written 2026-08-27. Branch `edge-wam-e01-e05` at `d96dc42`, which now contains `origin/main`.

**This document licenses nothing.** It is an index over gates that live in other files, and where
it quotes one, the quoted file is the authority. In particular it does not lift `T40_RULE_V1` §1,
does not flip `apple_sam2.GATE_QUALIFIED`, and does not authorise a cluster submission. Any
disagreement between this page and a `PR-*` document, an sbatch header or a module comment is
resolved in favour of the other file, and this page is the one that gets corrected.

---

## 0. The sentence that defines the goal

`docs/preregistration/PR-08-photoreal-augmentation.md` §1:

> **Forbids, until every item in §8 is closed and T-39 has reported:** generating a corpus,
> training any weight on generated frames, and quoting any number from this document as a result.

So "the sprint is done" has an exact meaning: **§8 items 1–7 are closed and somebody lifts §1 on
the record.** Nothing else counts, and no amount of adjacent work substitutes.

`PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §2 states the same thing from the other side —
*"the set that gates a first clip is exactly §8 items 1–7"* — and its §1 separates three moments
that get conflated in conversation:

| moment | gated by | runs |
|---|---|---|
| generating **at all** | `T40_RULE_V1` §1 + PR-08 §8 items 1–7 | before any GPU is spent |
| an **individual clip** during generation | G0c's composite (`robot_composite.check_mask`) | per unit, on the cluster |
| **training** on the result | G0a / G0b / G0c as VOID gates | after generation, on CPU |

This sprint is the first row only.

---

## 1. §8, item by item

| # | item | state |
|---|---|---|
| 1 | `--tune-visual` Recipe B, lr 5e-5 | **closed** — fixed in PR-08 itself |
| 2 | consumer contract with `emai/vla-training` | **closed** — `T40_RULE_V7`, signed 2026-08-22, `docs/contracts/vla-training-consumer.md` |
| 3 | **measured throughput + GPU-h ceiling** | **OPEN** — §2 below |
| 4 | **`GEOM_TOL` and `EST_DRIFT_P95` measured *and committed*** | **OPEN** — §3 below |
| 5 | depth + segmentation annotators in `isaac_binding.py` | **closed** 2026-08-22 |
| 6 | `TRAIN_STYLES` / `EVAL_STYLES` partition committed | **closed** — `configs/transfer25/styles.toml` + sidecar |
| 7 | T-39 has reported | **closed** — `T40_RULE_V4` §7, signed 2026-08-22, on VERDICT `N` |

Two items are open. They are independent of each other and can run in either order; item 4 is much
the longer chain.

---

## 2. Item 3 — throughput (one cluster job, nothing else)

**What is owed:** one `97 TIMING=1` run against the transcoded corpus, producing `THROUGHPUT.json`,
from which `CEILING_GPU_H` and `PARTITION_CEILING_GPU_H` are derived.

**Why no number already counts.** Two candidates exist and both are disqualified in writing:

- job 189142 reported 0.2 s/frame → 9.56 GPU-h and **generated nothing**. T-040 records the
  retraction: *"That figure is the input PR-08 §8 item 3 derives the whole-partition ceiling from,
  so the ceiling would have been derived from a crash."*
- the V8 hallucination probe's 1.16 s/frame: *"**THE 1.16 s/frame FIGURE IS NOT §8 ITEM 3's
  MEASUREMENT AND MAY NOT BE A BUDGET LINE.** It sized a decision; it is one diagnostic clip and
  likely optimistic … A `TIMING=1` run still owes the ceiling."*

**The enforcement half is already built.** `cluster/discoverer/97_transfer25_restyle.sbatch`:
*"`CEILING_GPU_H` and `PARTITION_CEILING_GPU_H` are therefore REQUIRED WITH NO DEFAULT … and the
full mode refuses to run until `THROUGHPUT.json` exists."* The timing path deliberately does **not**
ask for either ceiling, so the measurement that derives them is runnable.

**Blocked on:** an owner decision to submit. Nothing technical.

---

## 3. Item 4 — the two geometry numbers, in strict order

`configs/transfer25/pr08_geom_tol.json` currently has `geom_tol_px: null`,
`est_drift_p95_px: null`, `gate_margin_px: null`. `configs/transfer25/pr08_est_drift.json` does not
exist. The four steps below must happen in this order; doing 3.2 before 3.1 wastes the GPU-hours.

### 3.1 Resolve the five frames, then flip `GATE_QUALIFIED` — **owner**

`scripts/estimators/apple_sam2.py`. `GATE_QUALIFICATION_BLOCKERS` is empty for the first time, and
the module pre-empts the misreading three separate times:

> **EMPTY IS NOT PERMISSION, AND THE TUPLE BEING EMPTY IS NOT WHY.**

> **A SHORTER BLOCKER TUPLE DOES NOT FLIP THIS FLAG, AND NO EDIT THAT SHORTENS THE TUPLE MAY FLIP
> IT IN THE SAME COMMIT.**

The flag has **two** preconditions and only one of them is the propagation blocker. The second is a
recorded decision on blocker 2's residue (i). `T40_RULE_V18` registered that decision rule blind and
the census returned **outcome C** — contained. What is still open is one number inside that
evidence:

> `runs/pr08-geom-tol/shards/shard-7.json` records the same episode as `n_frames_with_centroid:
> 473`; the census measures 478. … **That is a hypothesis, not a measurement.**

**What closes it is named and is a submission:** the same census
(`scripts/census_operating_point_episode.py`, `--episode episode_000094`, both decode trees) run on
the cluster rather than on this workstation. If it reproduces 478, the gap is cross-machine and the
hypothesis becomes a measurement; if it reproduces 473, the local run is what needs explaining.

Then the flip is one commit, on its own, and it is the owner's signature — not a session's.

### 3.2 Re-measure `GEOM_TOL` at HEAD — **cluster**, after 3.1

The existing full-corpus number (0.47857992441961017 px, 402 episodes, 171 625 frames, merge job
190191) **can never be committed**, and this was discovered after the fact:

> Commit `e518a84` added `mask_validity_reference_max_frame_fraction` to `SEGMENTER_CONTRACT` and
> to the committed config **37 minutes after that merge was written**, and `contract_disagreements`
> counts an absent field as a disagreement by design … So `merge_committed_contract` would refuse
> the re-merge outright, exit 2, nothing written. **The corpus must be re-measured at HEAD** — and,
> because `gate_qualified` is baked into every shard at measurement time, **only after
> `GATE_QUALIFIED` flips.** A re-measurement before the flip produces another unusable artifact at
> the same cost.

That is `cluster/discoverer/103_measure_geom_tol.sbatch`, 16 shards + merge, ≈ 9 GPU-h.

Note for whoever re-reads `T40_RULE_V17`: its §0 "unchanged" table still lists `GEOM_TOL =
0.478579…` as if it were usable. V17 predates the `e518a84` finding by hours. The number is fine as
a *magnitude* for sizing; it is not committable, and V17 has not been amended.

### 3.3 Measure `EST_DRIFT_P95` and carry it — **cluster**, after 3.1

`T40_RULE_V14` (signed by the owner 2026-08-27) licenses a MuJoCo capture to stand in *"for
`EST_DRIFT_P95` and the arm comparison and for nothing else"*, which retires the Isaac blocker.

The carry is not a text edit. `configs/transfer25/pr08_geom_tol.json` says so itself:

> **CARRY IT WITH `measure_geom_tol.py --carry-est-drift configs/transfer25/pr08_est_drift.json`,
> NOT WITH A TEXT EDITOR.**

`est_drift_estimator_name` is the join key; a number carried across without it *"leaves that
assertion permanently unanswerable — which costs every G0b run its gate qualification."*

### 3.4 The margin has to come out positive — arithmetic, no decision

PR-08 §4/§6: the generator is held to `GEOM_TOL − EST_DRIFT_P95`, and *"if that is ≤ 0, the
estimator is not good enough and generation does not start."*

Where V17 leaves that, with the current (uncommittable) `GEOM_TOL` as the magnitude:

| `EST_DRIFT_P95` from | value (px) | margin (px) | margin (%) |
|---|---|---|---|
| single MuJoCo capture, s60-f720 | 0.23609 | 0.24249 | 50.67 |
| **V17 Arm A pooled, per-frame** | 0.31208 | 0.16650 | 34.79 |
| **V17 Arm A pooled, propagation** | 0.44861 | **0.02997** | **6.26** |

The propagation row is the one to watch: 6.26 % of the budget is not room, and which arm is
authoritative is settled by §4 step 2, not by picking the friendlier number.

---

## 4. Not a gate, but it decides whether the run is worth its budget

`PR-08-NOTE-2026-08-25-what-actually-blocks-what.md` §4:

> **Formally, `T40_RULE_V12` is not on the critical path to a first clip.** Items 3 and 4 are. V12
> could stay unsigned forever and §1 would be satisfiable without it. **Practically, resolving V12
> before generating is the difference between a run that costs its budget and yields a corpus, and
> one that costs its budget and yields 80 clips.**

The measured position:

- `robot_composite.check_mask` refuses a clip on any empty robot mask, and `restyle_transfer25.py`
  fails the **whole unit** after its GPU cost is already paid.
- **366 of 402 episodes (91.0 %)** contain at least one empty-mask frame. Area-bound alone:
  175/402 (43.5 %). Either condition: **385/402 (95.8 %)**.
- The robot is *genuinely absent* on those frames — measured detector failures on robot-present
  frames: **0**. So `check_mask` is refusing **correct** masks.
- `T40_RULE_V12` is **unsigned** and forbids a session from signing it. `T40_RULE_V15` stopped
  itself. `T40_RULE_V16`'s outcome **A is unreachable** — *"Area cannot separate a zero-harm shadow
  from a some-harm finger."* Remaining: **B** or **M**.
- V13's area bound *is* signed, but *"it arms G0c's area half. It does not make G0c pass, and it
  licenses no clip."*

**Reading:** if stage 1 is launched with the empty-mask semantics unchanged, expect to spend
~442 GPU-h and keep a single-digit percentage of it. This is the one item on this page that is
worth resolving *before* the two formal gates rather than after.

Two smaller repairs are also open and unfixed: the `plate.` pass refuses 100 % of source frames, so
§6's plate half cannot be measured at all; and V6's warm-colour reference misfires on non-warm
styles.

---

## 5. Already decided — do not reopen

- **Scope.** `T40_RULE_V11`, adopted 2026-08-24: stage 1 = `train_styles[0:4]` + 4 matched identity
  repeats = 8 of 25 style-instances, 3 216 clips. `STAGE` is required with no default;
  `STAGE=1 STYLE_SET=eval` is refused. V11 §3 fixes stage-2 escalation *in advance*.
- **Output target.** Everything this produces feeds **GR00T N1.7**, `ego_view` float32
  `[1, 480, 640, 3]` — one view at 640×480, read off the ONNX export.
- **Actions are never generated.** Restyle a real episode so the real labels survive. `docs/sim.md`
  / T-25's "sim frames are not training data" stands and PR-08 §3 deliberately does not overturn it.
- **The corpus on the cluster** is `${PROJ}/data/pr08-apple-640x480`, 402 episodes / 171 625 frames,
  plus the H.264-lossless transcode (402/402 proven bit-exact).
- `T40_RULE_V1` §1 was proposed for lifting on 2026-08-24 and **declined**: *"it is conditional on
  facts, names no signatory, and §8 items 3 and 4 are open."*

---

## 6. Cost, as the repo states it

| item | figure |
|---|---|
| whole committed partition, 25 instances | 10 050 clips / 4 290 625 frames ≈ **1 380 GPU-h** (~27 % of the allocation) |
| **V11 stage 1** | 3 216 clips ≈ **442 GPU-h** (~8.8 %) |
| `GEOM_TOL` full pass at HEAD | ≈ **9 GPU-h**, 16 shards in four waves |
| robot-mask area full pass | ≈ 9.5 GPU-h (already spent) |
| both §8 item 4 measurements together | *"about 18 GPU-hours before a single clip is generated"* |
| allocation remaining | ~4 879 of 5 000 GPU-h — **stale, 2026-08-15, re-read before budgeting** |

Cluster constraints that shape every submission: 4 h `MaxWall`, `MaxJobsPU=4`,
`MaxSubmitJobsPU=8` (counts every array task), billing `GPUs×1.0 + MemGB×0.25 + Threads×0.036` per
minute, and the login node is off limits for anything that computes.

---

## 7. The order, and who owns each step

1. **Owner** — authorise the cluster census of `episode_000094` (§3.1). Cheap, minutes.
2. **Owner** — record the residue-(i) decision and flip `GATE_QUALIFIED` in its own commit (§3.1).
3. **Cluster** — `103` re-run at HEAD + merge; `EST_DRIFT` measured and carried with
   `--carry-est-drift`; `102` re-run so its six `unverified` rows resolve. **→ item 4 closes iff the
   margin is > 0** (§3.2–3.4).
4. **Cluster** — `97 TIMING=1` → `THROUGHPUT.json` → derive `PARTITION_CEILING_GPU_H`. **→ item 3
   closes** (§2). Independent of 1–3; can run in parallel.
5. **Owner** — resolve the empty-mask semantics: sign a successor to V12 under V16 outcome B or M
   (§4). Formally optional, economically not.
6. **Owner** — lift `T40_RULE_V1` §1 on the record, naming a signatory.

Only after 6 does a clip get generated.
