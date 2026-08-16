# Consumer contract — `emai/vla-training`

**Written 2026-08-15.** Closes the open acceptance criterion in
`.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md` and `PR-08` §8 item 2:
*"The consumer contract with `emai/vla-training` is written down."*

**Producer:** this repo (WAM / `subprojects/data-factory`). **Consumer:** `emai/vla-training`,
which fine-tunes **NVIDIA Isaac GR00T N1.7** (`nvidia/GR00T-N1.7-3B`) and evaluates in MuJoCo and
Isaac. This document is what the producer must hand over and what the consumer may assume.

> **Every measured claim below is dated and carries a named artifact.** A resolution or a
> field-order claim without one has an expiry date — re-verify against the named file before
> depending on it, and re-verify all of §2 if the base model or the export changes.
> The producer machine paths (`/home/humanoid/develop/vla-training`,
> `/home/humanoid/models/GR00T-N1.7-ApplePnP-V1`) are read-only evidence, not deliverables.

---

## 0. This contract covers **two** corpora, and PR-08 §8 item 2 names the wrong one

This is the first section because getting it wrong is the failure this document exists to prevent.

`PR-08` §8 item 2 fixes the contract as *"LeRobot v3.0, 28-dim arms+hands, right hand
index-before-middle"*. Those three fields are a real, coherent contract — **but they describe the
`unitreerobotics/G1_Dex3_*` corpus (corpus B below), not `nvidia/GR00T-N1.7-AppleToPlate`, which is
the corpus PR-08 §3 chose to restyle.** For AppleToPlate the on-disk version is **v2.1**, not v3.0,
and the width is **43-dim in seven groups**, not 28-dim in two.

| | **Corpus A — AppleToPlate** | **Corpus B — `unitreerobotics/G1_Dex3_*`** |
|---|---|---|
| what it is | `nvidia/GR00T-N1.7-AppleToPlate`, 402 ep / 171 625 frames | 13 sets, 3 152 ep / 2 587 515 frames |
| who targets it | **T-040 / PR-08 restyle**, the whole benchmark, `GR00T-N1.7-ApplePnP-V1` | **T-043** conversion, route 1 |
| LeRobot version | **v2.1** | **v3.0** |
| state / action width | **43**, seven groups incl. legs + waist | **28**, two blocks, no waist, no legs |
| camera key | one, `observation.images.ego_view` | `cam_left_high` (+ up to 3 more) |
| block order | fully specified, §2.4 | **arm-first**, §3, with open sub-questions |

**PR-08 §8 item 2's three fields are therefore correct for corpus B and wrong for corpus A on two
of three.** `T40_RULE_V1` is a registered rule and is **not edited** (`docs/handoff.md` §3); §7
below states what a superseding `T40_RULE_V2` would have to say.

---

## 1. Scope, and the one thing the consumer does **not** have a special path for

There is no "restyled corpus" ingest path anywhere in `vla-training`. Every entry point takes an
ordinary LeRobot dataset root:

- `scripts/40_groot_prepare_dataset.sh` (v3.0 → v2 conversion, then drops `modality.json` into
  `meta/`) — `scripts/40_groot_prepare_dataset.sh:3,12-16`, read 2026-08-15
- `scripts/41_groot_finetune.sh` → `gr00t/experiment/launch_finetune.py` with
  `--dataset-path` / `--modality-config-path` — `scripts/41_groot_finetune.sh:21-25`
- on cluster: `eu-hub/train_apple_{nvidia,frozen,visual}.sbatch`, same flags —
  `eu-hub/train_apple_nvidia.sbatch:99-104`

**Consequence:** the deliverable is a LeRobot dataset root that is byte-compatible with
`_data/apple_pnp/dataset`. Not a new format, not a sidecar, not a manifest.

---

## 2. Corpus A — `nvidia/GR00T-N1.7-AppleToPlate` (the PR-08 target)

Evidence base, all read 2026-08-15:
`/home/humanoid/develop/vla-training/eval/real_reference/info.json` and `.../modality.json`
(checked-in copies of the corpus `meta/`), and
`/home/humanoid/models/GR00T-N1.7-ApplePnP-V1/exported_leapp.yaml` — the ONNX export of the model
actually fine-tuned on this corpus. The export is the authority for *what the model eats*; the
`meta/` files are the authority for *what goes on disk*.

### 2.1 On-disk layout and LeRobot version

| field | value | evidence (read 2026-08-15) |
|---|---|---|
| `codebase_version` | **`v2.1`** | `eval/real_reference/info.json:2` |
| `robot_type` | `unitree_g1` | `info.json:3` |
| scale | 402 episodes / 171 625 frames / 1 chunk / 1 task | `info.json:4-10` |
| fps | **30** | `info.json:9` |
| `data_path` | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` | `info.json:14` |
| `video_path` | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` | `info.json:15` |
| embodiment tag | `new_embodiment` | `eu-hub/train_apple_nvidia.sbatch:103` |

**Required `meta/` files:** `info.json`, `modality.json`, `stats.json`, `relative_stats.json`,
`episodes.jsonl`, `tasks.jsonl`. Not optional — the cluster copy of this dataset failed for lack of
`stats.json` + `relative_stats.json` and they had to be shipped separately
(`eu-hub/RUNBOOK.md:66-72`, read 2026-08-15).

**Required at the dataset root, outside `meta/`:** a ModalityConfig **`.py`** file, stored as
`new_embodiment_config_defaults.py` and passed as `--modality-config-path` —
`eu-hub/RUNBOOK.md:39` ("incl. the required `new_embodiment_config_defaults.py`") and `:72`,
`eu-hub/train_apple_nvidia.sbatch:104`.

### 2.2 Image: key, resolution, dtype, view count

| field | value | evidence (read 2026-08-15) |
|---|---|---|
| dataset key | `observation.images.ego_view` | `info.json:116` |
| modality short key | `ego_view` ← `original_key: observation.images.ego_view` | `modality.json:107-111` |
| on-disk shape | `[480, 640, 3]` = (h, w, c) — **640×480 RGB** | `info.json:118-127` |
| codec / pix_fmt / fps / audio | `av1` / `yuv420p` / 30 / none | `info.json:129-136` |
| **number of views** | **exactly one** | `exported_leapp.yaml:349` — `preprocess_video: [ego_view]` is the entire input list |
| model-boundary tensor | `ego_view`, `float32`, `[1, 480, 640, 3]`, `kind: state/camera/image` | `exported_leapp.yaml:4-8` |
| **pixel range** | float32 **0..255**, *not* 0..1 | `eval/onnx_leapp_server.py:29-33` and `:176` — measured: 0..1 collapses `pixel_values` std to **0.004** vs **0.242** |
| client-side resize | **none**; native 640×480, `groot_image_size: null` | `../vla-training/docs/vla-benchmark.md:71-72` |

`preprocess_video` then emits `pixel_values [352, 1536]` + `image_grid_thw [1,3]` internally
(`exported_leapp.yaml:10-13, 22-25`) — the VLM does its own patching. **The contract at the model
boundary is 480×640×3 and nothing smaller.** A second camera has nowhere to go.

### 2.3 Language

- key `annotation.human.task_description` ← `original_key: task_index` — `modality.json:112-116`
- task string is exactly `move the apple to the plate`, **no trailing period** —
  `../vla-training/docs/vla-benchmark.md:66-68` (which also warns that the `g1_dex3` path uses a variant *with* a
  period; do not mix them)
- in the ONNX export the prompt is **baked in**: `input_ids` is an *output* of `preprocess_video`,
  so a `/predict` request's `task` field is ignored — `eval/onnx_leapp_server.py:33-37`

### 2.4 State and action: widths and **BLOCK ORDER**

**`observation.state` float32 `[43]`** (`info.json:17-25`) and **`action` float32 `[43]`**
(`info.json:26-34`). Both use the **same seven-group layout**, in this order
(`eval/real_reference/modality.json:2-31` for state, `:32-60` for action):

```
[ 0: 6]  left_leg     6
[ 6:12]  right_leg    6
[12:15]  waist        3
[15:22]  left_arm     7
[22:29]  right_arm    7
[29:36]  left_hand    7
[36:43]  right_hand   7            total 43
```

Three independent confirmations of that order, all read 2026-08-15:

1. `exported_leapp.yaml:347-348` — `preprocess_state: [left_leg, right_leg, waist, left_arm,
   right_arm, left_hand, right_hand]`, and each group's width at `exported_leapp.yaml:38-87`.
2. `eval/onnx_leapp_server.py:94-102` — `STATE_GROUPS` hard-codes the identical seven slices.
3. Numerically re-derived over all 171 625 frames: `state[15:22]` ↔ `action[0:7]` RMS
   **0.0288 rad** against a **0.7634** cross-check — `eu-hub/make_dataset_state31.py:33-45`.

**All seven groups are required.** The model will not run on arms+hands alone: `preprocess_state`
takes seven tensors, including `left_leg` and `right_leg` (`exported_leapp.yaml:38-52, 347-348`).
This is the reason `datasets/gr00t-apple-full/` is not merely low-res (120×160) but structurally
insufficient as a source.

**Joint element names — the hands are asymmetric AND the action block is permuted.**
Read from `exported_leapp.yaml` 2026-08-15:

| block | element order | line |
|---|---|---|
| **state** `left_hand` | `thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1` | `:72-79` |
| **state** `right_hand` | `thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1` | `:80-87` |
| **action** `left_hand` | `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2` | `:254-260` |
| **action** `right_hand` | `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2` | `:262-268` |

So: **the right hand is index-before-middle and the left hand is middle-before-index** in the
*state* block (this is the half PR-08 §8 item 2 gets right, and it names only the right hand);
and the model's *action* block is a permutation of its state block, index-first for **both** hands.
The permutation constants that map decoder output back to dataset/state order are
`LEFT_HAND_PERM = [4,5,6,2,3,0,1]`, `RIGHT_HAND_PERM = [4,5,6,0,1,2,3]` —
`eval/onnx_leapp_server.py:104-111`, rationale at `:39-54`.

**For writing data, the dataset/state order is the authoritative one** —
`scripts/33_make_dataset_action31.py:85-89`, whose comment is explicit that Dex3-1 is asymmetric
and must not be "symmetrically repaired". A producer never emits the export's action permutation
to disk.

### 2.5 The nine extra `action.*` columns

Beyond the flat 43-dim `action`, the corpus carries nine separate columns
(`info.json:35-115`, declared in `modality.json:61-105`, both read 2026-08-15):

`action.effort_left_leg[6]`, `action.effort_right_leg[6]`, `action.effort_waist[3]`,
`action.effort_left_arm[7]`, `action.effort_right_arm[7]`, `action.effort_left_hand[7]`,
`action.effort_right_hand[7]`, `action.navigate_command[3]`, `action.base_height_command[1]`.

They are not decoration. `decode_action` has **twelve** outputs, not the five the data-factory
README lists — the five `target/joint/position` groups **plus** `navigate_command [1,16,3]`,
`base_height_command [1,16,1]` and five `target/joint/effort` groups
(`exported_leapp.yaml:352-356`, detail at `:276-318`).

**A restyle must carry all nine through verbatim.** A related trap, for anyone tempted to strip
them: LeRobot's `feature_utils.py:161-162` classifies *every* `action`-prefixed key as an output
feature, so all nine would become policy outputs — which is why
`scripts/33_make_dataset_action31.py:34-42` strips them for the LeRobot-family trainers, and why
GR00T is immune (its `modality.json` selects groups explicitly). **Second-hand citation:** this is
reported *by* `scripts/33_make_dataset_action31.py:34-40` (read 2026-08-15);
`feature_utils.py` itself lives inside the Isaac-GR00T / LeRobot tree and is **not present on this
machine**, so the line numbers are unverified here.

### 2.6 Action horizon

**16.** `decode_action` emits `[1, 16, k]` for every group (`exported_leapp.yaml:239-318`), and
`ACTION_HORIZON = 16` independently in `groot/g1_dex3_1cam_modality_config.py:21`,
`groot/g1_dex3_modality_config.py:23`, `groot/g1_dex3_2cam_modality_config.py:17` and
`eval/onnx_leapp_server.py:113`. The benchmark executes 8 of the 16 (`EXEC_HORIZON = 8`,
`../vla-training/docs/vla-benchmark.md:69-70`).

### 2.7 The `modality.json` key-shape trap

`modality_keys` in the ModalityConfig `.py` must be **short** (`"arms"`, `"cam_right_high"`), never
dotted (`"state.arms"`). GR00T indexes `modality_meta[modality][key]` against `meta/modality.json`,
which uses short keys; dotted keys raise `KeyError` in `get_dataset_statistics` and **mis-map video
keys by position** — `groot/g1_dex3_1cam_modality_config.py:7-10`, repeated verbatim in
`groot/g1_dex3_2cam_modality_config.py:8-11`, both read 2026-08-15.

⚠ **Known inconsistency inside `vla-training`, reported not resolved:**
`groot/g1_dex3_modality_config.py:25-42` still uses the dotted form the other two files say is
broken, and it is the one wired into `scripts/41_groot_finetune.sh:25`. Producer side is
unaffected; flagged so nobody copies that file as a template.

---

## 3. Corpus B — `unitreerobotics/G1_Dex3_*` (28-dim, the fields PR-08 §8 item 2 actually names)

Relevant here only because PR-08 §8 item 2's three fields belong to it. Conversion is **T-043**, not
T-040. Numbers measured 2026-08-15 from local `meta/` at `~/wam-t041/raw/` (13 sets, no `data/`),
recorded in `.mc/tasks/todo/T-043-…md` §1–2.

| field | value |
|---|---|
| LeRobot version | **v3.0** (episodes concatenated, boundaries in `meta/episodes/*/*.parquet`) |
| state / action | `float32[28]` both, flat — **no waist column, no leg columns** |
| camera | `cam_left_high` at `[480, 640, 3]`, present in all 13 sets, 30 fps, AV1 |
| scale | 3 152 ep · 2 587 515 frames · 23.96 h @ 30 fps |
| variants | 6 sets `robot_type: Unitree_G1` (4 cameras, hand limit 120°) · 7 sets `Unitree_G1_Dex3` (2 cameras, hand limit 100°) |

### 3.1 Block order: **ARM-FIRST**, measured 2026-08-15

```
[ 0:14]  arm    (left + right, 7 + 7)
[14:28]  hand   (left + right, 7 + 7)
```

Two independent lines of evidence, both recorded in `T-043` §1:

1. **Mechanical joint limits from `meta/stats.json` of all 13 sets.** A finger's range is one-sided
   (opens from a hard zero) and ends at a round mechanical limit; an arm joint is bidirectional and
   bounded by nothing round. One-sided dims: **0 out of 14 in `[0:14]`, 4–10 out of 14 in
   `[14:28]`, unanimous across all 13 independently recorded sets**, with the far end railing at a
   clean **100.0°/100.1°** (7 sets) or **120.0°** (5 sets).
2. **An explicit modality spec from a pipeline that produced a working model** —
   `/home/humanoid/develop/vla-training/groot/modality_g1_dex3.json:2-9` (read 2026-08-15):
   `"state": {"arms": {0,14}, "hands": {14,28}}`, same for `"action"`. Written before T-043
   existed, by the pipeline whose output (`GR00T-N1.7-ApplePnP-V1`) trains and runs.

**Five documents previously said hand-first, and that was a transplant.** The hand-first finding is
genuine but belongs to a **different corpus**: `USC-PSI-Lab/Humanoid-Everyday-G1`, LeRobot **v2.1**,
whose state ships as separate `arm_joints` (14) / `leg_joints` (15) / `hand_joints` (14) fields.
T-041 measured it there by correlating against recorded state
(`.mc/tasks/todo/T-041-…md:268-270`: `action[0:14]` ↔ hand 0.61–0.67 vs arm 0.28–0.40;
`action[14:28]` ↔ arm 0.74–0.92 vs hand 0.30–0.43). Carrying it onto the `unitreerobotics` sets
would have produced exactly the silent arm/hand transposition it was written to warn against. Both
facts are true of their own corpus. Corrections landed 2026-08-15 in `docs/action-labels.md:170-183`,
`subprojects/data-factory/README.md:122-127`, `subprojects/edge-wam/README.md:47-49`,
`subprojects/edge-wam/tasks/E-02-…md:47-53`, `.mc/tasks/done/T-042-…md:166-170`.

### 3.2 OPEN RISK — left/right and intra-hand order are **UNVERIFIED**

**Not resolved here, deliberately.** Within each 14-dim block, neither the left/right order nor the
intra-hand joint order has been measured against the parquet, and **three mutually inconsistent
intra-hand orderings are on record** (`T-043` §1, 2026-08-15):

| source | ordering |
|---|---|
| the corpus card | thumb-first, **symmetric** |
| Arena | index-first |
| NVIDIA / this contract's corpus A | **asymmetric** — left `thumb,thumb,thumb,middle,middle,index,index`, right `thumb,thumb,thumb,index,index,middle,middle`; permutations `[4,5,6,2,3,0,1]` / `[4,5,6,0,1,2,3]` |

**Do not pick one.** T-041's lesson applies verbatim: a source that was wrong about the block order
earns no trust about the finger order. The resolution is the same method that settled the block
order — correlate against the parquet, do not read the card — and the action parquets
(647 MB, Apache-2.0) have not been fetched. This is a live question with a known-wrong default.

PR-08 §8 item 2's *"right hand index-before-middle"* is **confirmed for corpus A's state block**
(`exported_leapp.yaml:80-87`) and is **an assumption, not a measurement, for corpus B**. The
`vla-training` side asserts it for the 28-dim layout at
`groot/g1_dex3_1cam_modality_config.py:12-14` — an assertion from the same family of sources that
got the block order wrong, so it carries no weight until measured.

---

## 4. THE LOAD-BEARING CLAUSE — where the action labels come from

> **The action labels come from the SOURCE RECORDING. Never from the generator.**
>
> A restyle changes **pixels only**. The trajectory — `observation.state`, `action`, all nine
> `action.*` columns, `timestamp`, `frame_index`, `episode_index`, `task_index` — is carried over
> from the source episode **unchanged, byte-for-byte where the dtype permits**. The generator is
> never consulted about, and never permitted to influence, a single label.

This is not a style preference. It is the entire argument that lets generated frames into a
training corpus at all, and PR-08 §2 rests on it:

> *"The labels do not come from the generator. In PR-06 the generated pixels were the thing being
> scored against truth. Here the actions are the recorded teleop trajectory, carried over
> unchanged, and the generated pixels are only an input perturbation. A generator error in PR-06
> was a wrong answer; a generator error here is a corrupted training input whose label is still
> correct — **unless it moves geometry**, which is what §6's G0b exists to catch."*
> — `docs/preregistration/PR-08-photoreal-augmentation.md:44-48`

Three consequences the consumer is entitled to hold the producer to:

1. **No inferred, dreamed, or inverse-dynamics label may enter a corpus delivered under this
   contract.** WAM's own action decoder cannot supply labels without circularity — it is the
   negative result, not a labeller (`.mc/tasks/todo/T-040-…md:50-53`).
2. **Frame count and frame order are preserved.** Restyled frame *i* of episode *e* pairs with
   source label *i* of episode *e*. Any reordering, drop, or interpolation breaks the pairing
   silently — which is precisely what G0a is instrumented to catch (§5).
3. **Absolute targets, not corrections.** `decode_action` emits `kind: target/joint/position`
   (`exported_leapp.yaml:240-274`) — absolute joint positions in radians. A geometry shift between
   pixels and label is therefore not attenuated anywhere downstream; it lands directly on the
   commanded pose.

---

## 5. What the producer guarantees, and what the consumer may therefore assume

The three gates are `T40_RULE_V1`, `PR-08` §6, and all three are VOID gates that run on CPU
**before any training** (`docs/preregistration/PR-08-photoreal-augmentation.md:159-186`).

### G0a · Label identity

**Guarantee.** `screen_corpus.py` (T-34) runs on the generated corpus and must **reproduce the
source's M1/M2/M3 within `EXPECT_TOL`** (0.02, 0.02, 0.05 — the script's own archived tolerances),
recorded as `screen_corpus --expect` against the source's values. M1/M2/M3 are computed from
proprioception, the clock and the gripper channel; a restyle changes no action, so identity holds
**by construction**. A deviation is therefore not a finding about the corpus — it is proof that the
pipeline **corrupted or reordered the action labels**, and it is VOID.
— `PR-08:161-169`

**Consumer may assume.** The action column of a delivered restyled corpus is the source's action
column. If it is not, the corpus was never delivered, because G0a voids the run.

### G0b · Geometry invariance

**Guarantee.** Object and plate centroids in the restyled clip agree with the source within
`GEOM_TOL`, defined as *the median per-step object-centroid displacement in the source clips*,
computed and committed **before** generation. The generator is held to `GEOM_TOL − EST_DRIFT_P95`;
if that is ≤ 0, the estimator is not good enough and generation does not start.
— `PR-08:170-177`

**Consumer may assume.** The carried-over label describes the scene that is actually on screen —
the restyled pixels never drift further from the source than one action step moves the scene.

**Stated weakness, carried here rather than left in PR-08.** `EST_DRIFT_P95` is calibrated against
**Isaac renders**, not real footage, so it is a **lower bound** on the real estimator error, and a
G0b margin that only clears under a lower bound **is not a pass** (`PR-08:118-125`,
`PR-08:170-177`). PR-08 calls this "the single biggest soft spot in the design". The consumer
should read a G0b pass accordingly.

### G0c · Embodiment / robot pixels composited back

**Guarantee.** The real robot's pixels are **unconditionally composited back** over every generated
frame using the robot segmentation mask. Not thresholded, not conditional — `video_fidelity`
provably cannot see the generic-manipulator defect
(`runs/backbone_eval/video/embodiment_grid.png`), and any IoU threshold on the robot mask would be
a coined number. Robot-mask IoU between source and generated is still recorded, as a **diagnostic
on the generator, never as a gate**.
— `PR-08:178-183`

**Consumer may assume.** The robot in every delivered frame is the G1 + Dex3 that recorded the
episode, not a generator's idea of a manipulator. The defect cannot enter, so no threshold has to
be trusted.

### What the consumer may **not** assume

- **Not that anything has been delivered.** Under `PR-08` §1 nothing is generated until T-39
  reports; §8 lists seven preconditions of which items 2 (this document), 3, 4 and 5 were open as
  of 2026-08-15. This contract describes the shape of a delivery, not a delivery.
- **Not that a restyled corpus can be reported inside the benchmark table.** See §6, open item 4.

---

## 6. Open items — what the consumer needs and the producer cannot currently supply

Written as gaps, not papered over.

1. **The source resolution problem is solved only on paper.** The consumer needs 640×480
   (`info.json:118-127`). `datasets/gr00t-apple-full/` is **120×160** — a 4× shortfall in each
   dimension — so the restyle must re-derive from the HF source at full resolution
   (`subprojects/data-factory/README.md:82-85`, `PR-08:101-103`). That re-derivation pipeline is not
   written.
2. **Depth and segmentation are not wired.** Transfer2.5 consumes depth + segmentation + Canny;
   AppleToPlate ships one RGB camera, so only Canny is computable, and `isaac_binding.py` makes
   exactly one `AnnotatorRegistry.get_annotator` call (`"rgb"`). `distance_to_camera` and
   `semantic_segmentation` are unwired, which blocks §4 of PR-08 entirely and therefore blocks
   `EST_DRIFT_P95`, and therefore blocks G0b. — `PR-08:104-112`, `PR-08` §8 item 5,
   `.mc/tasks/todo/T-040-…md` Notes (correction of 2026-08-06).
3. **`GEOM_TOL` and `EST_DRIFT_P95` are unmeasured and uncommitted.** — `PR-08` §8 item 4.
4. **A restyled corpus violates the benchmark's own invariant.** `../vla-training/docs/vla-benchmark.md:61-62`,
   rule 1 of "what must stay identical between all candidates": *exactly the 402 episodes /
   171 625 frames (`info.json` byte-identical); no re-split, **no additional augmenting for
   individual models***. A real+restyled fine-tune therefore **cannot be reported inside that
   comparison table** without an explicit, written protocol amendment. Neither PR-08 nor the
   data-factory README mentions this. It is a procedural gap, not a format gap, and it must be
   closed before any B-arm number is quoted next to an A-arm number.
5. **`_data/apple_pnp/CONTRACT.md` does not exist on this machine.** It is cited throughout
   `vla-training` as the authoritative policy contract — e.g.
   `scripts/33_make_dataset_action31.py:85-86` cites `CONTRACT.md:19-24` for the Dex3 asymmetry —
   and a filesystem-wide search on 2026-08-15 found nothing. Every claim in this document is
   sourced from files that do exist. If that file surfaces and disagrees, it wins on corpus A's
   hand ordering and this document must be revised.
6. **Corpus B's intra-hand and left/right order** — §3.2. Blocked on a 647 MB fetch that has not
   been requested (repo rule: nothing downloaded at scale without asking first).
7. **The dotted-key ModalityConfig in `vla-training`** — §2.7. Consumer-side defect, reported, not
   ours to fix.

---

## 7. Disagreements with `PR-08` §8 item 2 / `T40_RULE_V1`

`docs/preregistration/PR-08-photoreal-augmentation.md` is a **registered rule and has not been
edited** (project constraint; `docs/handoff.md` §3 — rules are versioned, never edited in place).
Recorded here for a superseding **`T40_RULE_V2`**:

| # | PR-08 §8 item 2 says | evidence says, for the corpus PR-08 §3 chose | severity |
|---|---|---|---|
| **D1** | LeRobot **v3.0** | **v2.1** — `eval/real_reference/info.json:2`; GR00T prep is literally "v3.0 → v2" (`scripts/40_groot_prepare_dataset.sh:3,12-13`); `docs/vla-modelle.md:23,105` gives GR00T's format as "LeRobot v2 + `modality.json`"; v3.0 is SmolVLA / π0.5 / LingBot's requirement (`lingbot/README.md:89-96`) | **major** |
| **D2** | **28-dim** arms+hands | **43-dim**, seven groups incl. `left_leg`/`right_leg`/`waist` — `info.json:17-34`, `modality.json:2-60`, `exported_leapp.yaml:347-348`. 28-dim arms+hands is the `unitree_lerobot` G1_Dex3 converter layout (`groot/modality_g1_dex3.json:2-9`) — a **different corpus** | **major** |
| **D3** | (implied) multi-camera 28-dim path | AppleToPlate has **exactly one** camera, `ego_view` (`info.json:116`, `exported_leapp.yaml:349`). A four-camera loader on this one-camera corpus is a hard `KeyError` (`../vla-training/docs/vla-benchmark.md:1381`) | **major** |
| **D4** | right hand index-before-middle | **correct — and under-specified.** True of corpus A's *state* `right_hand` (`exported_leapp.yaml:80-87`); the *left* hand is middle-before-index; and the model's *action* block is index-first for **both** hands (`:254-268`). For corpus B it is an unverified assumption (§3.2) | **minor / correct** |

Two further disagreements are with the **data-factory README**, not with PR-08:

- **D5** — `subprojects/data-factory/README.md:71-74` lists the action output as five
  `target/joint/position` groups. `decode_action` has **twelve** outputs
  (`exported_leapp.yaml:352-356`), and the corpus carries nine `action.*` columns
  (`info.json:35-115`) a restyle must carry through. The README's list is a subset.
- **D6** — the same README states no **state** contract at all. It fixes video and action but never
  says the consumer requires a **43-dim `observation.state` including both legs**. For a restyle
  this is satisfied by construction, but it is the reason `datasets/gr00t-apple-full/` is
  structurally insufficient rather than merely low-res, so it must be written down.

**None of these are edits to a registered rule.** The same three fields are copied into
`.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md:46-48` and `:189-190` — those are
task text, not registered rule, and can be corrected in place.

---

## 8. Provenance and expiry

| claim | source artifact | read / measured |
|---|---|---|
| model input contract, hand element orders, 12 decoder outputs, horizon 16 | `/home/humanoid/models/GR00T-N1.7-ApplePnP-V1/exported_leapp.yaml` (365 lines) | 2026-08-15 |
| on-disk layout, widths, `action.*` columns, video info | `/home/humanoid/develop/vla-training/eval/real_reference/{info.json,modality.json}` | 2026-08-15 |
| 0..255 pixel range, state slices, hand permutations, baked prompt | `.../eval/onnx_leapp_server.py` | 2026-08-15 |
| meta-file requirements, ModalityConfig placement | `.../eu-hub/{RUNBOOK.md,pack_bundle.sh,train_apple_nvidia.sbatch}` | 2026-08-15 |
| state↔action RMS 0.0288 vs 0.7634 over 171 625 frames | `.../eu-hub/make_dataset_state31.py:33-45` | as recorded in that file |
| task string, no client resize, `EXEC_HORIZON`, benchmark invariant | `../vla-training/docs/vla-benchmark.md` | 2026-08-15 |
| corpus B arm-first block order, scale, variants | `~/wam-t041/raw/*/meta/stats.json` (13 sets), `.../groot/modality_g1_dex3.json` | 2026-08-15 |
| corpus B intra-hand order | **not measured — three conflicting sources** | — |
| G0a / G0b / G0c wording | `docs/preregistration/PR-08-photoreal-augmentation.md:159-186` (`T40_RULE_V1`) | 2026-08-15 |

**Re-verify §2 in full if the base model, the export, or the corpus changes.** The export was read
once, on one date, from one artifact.
