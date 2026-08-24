# Consumer contract — `emai/vla-training`

**Written 2026-08-15. Amended 2026-08-22 under `T40_RULE_V7`
(`docs/preregistration/PR-08-V7-consumer-contract.md`).** Closes the open acceptance criterion in
`.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md` and `PR-08` §8 item 2:
*"The consumer contract with `emai/vla-training` is written down."*

> **What the 2026-08-22 amendment did.** The project owner fixed `PR-08` §8 item 2 on
> **`nvidia/GR00T-N1.7-AppleToPlate`** — corpus A below, the corpus `PR-08` §3 chose to restyle —
> and replaced item 2's three *descriptive* fields (`LeRobot v3.0`, `28-dim arms+hands`, `right
> hand index-before-middle`) with the measured description of that corpus. Item 2's **fourth**
> clause — *"the action labels come from the source recording, never from the generator"* — is
> **unchanged, verbatim**, and is the load-bearing one (§4). Sections changed: §0 (rewritten;
> the superseded text is preserved verbatim inside it), §2 (evidence base and the hand-order
> verification status), §3.2, §5, §7, §8. **Nothing in the amendment licenses generation or
> training** — `T40_RULE_V1` §1 binds in full and `PR-08` §8 items 3 and 4 are still open.

**Producer:** this repo (WAM / `subprojects/data-factory`). **Consumer:** `emai/vla-training`,
which fine-tunes **NVIDIA Isaac GR00T N1.7** (`nvidia/GR00T-N1.7-3B`) and evaluates in MuJoCo and
Isaac. This document is what the producer must hand over and what the consumer may assume.

> **Every measured claim below is dated and carries a named artifact.** A resolution or a
> field-order claim without one has an expiry date — re-verify against the named file before
> depending on it, and re-verify all of §2 if the base model or the export changes.
> The producer machine paths (`/home/humanoid/develop/vla-training`,
> `/home/humanoid/models/GR00T-N1.7-ApplePnP-V1`) are read-only evidence, not deliverables.

---

## 0. This contract is fixed on **corpus A**, and it still covers two corpora

This is the first section because getting it wrong is the failure this document exists to prevent.

**RESOLVED 2026-08-22 by `T40_RULE_V7`.** `PR-08` §8 item 2 is fixed on
**`nvidia/GR00T-N1.7-AppleToPlate`** — corpus A — which is the corpus `PR-08` §3 chose to restyle
and the corpus everything under §2 describes. Item 2's three descriptive fields, which described
corpus B, have been replaced with corpus A's measured description; its fourth clause is unchanged.
`PR-08` §8 item 2 is **CLOSED**.

Corpus B is still documented here (§3) because **T-043** converts it and because the fields item 2
used to carry belong to it — a reader who finds those fields quoted somewhere else needs to know
which dataset they are true of.

| | **Corpus A — AppleToPlate** | **Corpus B — `unitreerobotics/G1_Dex3_*`** |
|---|---|---|
| what it is | `nvidia/GR00T-N1.7-AppleToPlate`, 402 ep / 171 625 frames | 13 sets, 3 152 ep / 2 587 515 frames |
| who targets it | **T-040 / PR-08 restyle**, the whole benchmark, `GR00T-N1.7-ApplePnP-V1` | **T-043** conversion, route 1 |
| **is this contract's subject** | **YES** — `PR-08` §8 item 2, as amended by `T40_RULE_V7` | no — documented for T-043 and for provenance |
| LeRobot version | **v2.1** | **v3.0** |
| state / action width | **43**, seven groups incl. legs + waist | **28**, two blocks, no waist, no legs |
| camera key | one, `observation.images.ego_view` | `cam_left_high` (+ up to 3 more) |
| block order | fully specified, §2.4 | **arm-first**, §3, with open sub-questions |

**The corpus's own metadata is the authority for corpus A, and it is on this machine.** Read
2026-08-22 from
`~/.cache/huggingface/hub/datasets--nvidia--GR00T-N1.7-AppleToPlate/snapshots/d89c126a713c6632432a607c12661546ff4d6ea9/meta/`:

| field | value | evidence |
|---|---|---|
| `codebase_version` | **`v2.1`** | `meta/info.json:2` |
| scale | 402 episodes / 171 625 frames | `meta/info.json:4-5` |
| `observation.state` / `action` | `float32`, **`[43]`** both | `meta/info.json:17-25`, `:26-34` |
| seven groups, same for state and action | §2.4 | `meta/modality.json:2-31`, `:32-60` |

sha256 `0e3dd494…af36` (`info.json`, 174 lines) and `84e5eb4f…0d62a` (`modality.json`, 116 lines).
Both were diffed on 2026-08-22 against the consumer's checked-in copies at
`/home/humanoid/develop/vla-training/eval/real_reference/` and are **byte-identical** to them, so
producer and consumer are not reading two different specifications of this corpus.

### 0.1 Superseded text — what §0 said from 2026-08-15 to 2026-08-22

Preserved verbatim rather than deleted, because the fact that a contract described the wrong
dataset for a month is part of the record. `T40_RULE_V7` §6 records why nothing failed in the
meantime: **nothing in the pipeline reads §8 item 2.**

> ## 0. This contract covers **two** corpora, and PR-08 §8 item 2 names the wrong one
>
> This is the first section because getting it wrong is the failure this document exists to prevent.
>
> `PR-08` §8 item 2 fixes the contract as *"LeRobot v3.0, 28-dim arms+hands, right hand
> index-before-middle"*. Those three fields are a real, coherent contract — **but they describe the
> `unitreerobotics/G1_Dex3_*` corpus (corpus B below), not `nvidia/GR00T-N1.7-AppleToPlate`, which is
> the corpus PR-08 §3 chose to restyle.** For AppleToPlate the on-disk version is **v2.1**, not v3.0,
> and the width is **43-dim in seven groups**, not 28-dim in two.
>
> *(the two-corpus table stood here; it is retained above, with one row added)*
>
> **PR-08 §8 item 2's three fields are therefore correct for corpus B and wrong for corpus A on two
> of three.** `T40_RULE_V1` is a registered rule and is **not edited** (`docs/handoff.md` §3); §7
> below states what a superseding `T40_RULE_V2` would have to say.

`T40_RULE_V1` remains a registered rule and remains **unedited** (`docs/handoff.md` §3). The
amendment is `T40_RULE_V7`, alongside it — not an edit to it. What §7 below anticipated a
superseding rule would have to say is now what one says.

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

⚠ **The first two entry points above are corpus B's route, and must not be pointed at a corpus A
root.** Added 2026-08-22. `scripts/40_groot_prepare_dataset.sh` converts **v3.0 → v2** and then
overwrites `meta/modality.json` with the **28-dim** `groot/modality_g1_dex3.json`; corpus A is
already v2.1 and its `meta/modality.json` is the 43-dim seven-group one, so running that script
against it would destroy the layout §2.4 depends on. `scripts/41_groot_finetune.sh:25` likewise
passes the 28-dim `groot/g1_dex3_modality_config.py`. The corpus A route is the third bullet —
`eu-hub/train_apple_{nvidia,frozen,visual}.sbatch`, which passes
`new_embodiment_config_defaults.py`. Which of the two the consumer would actually use for a
delivered restyled root is **not settled from this side**: see §7.1.

---

## 2. Corpus A — `nvidia/GR00T-N1.7-AppleToPlate` (the PR-08 target)

**This is the corpus the contract is fixed on** (§0, `T40_RULE_V7`).

Evidence base, all read 2026-08-15:
`/home/humanoid/develop/vla-training/eval/real_reference/info.json` and `.../modality.json`
(checked-in copies of the corpus `meta/`), and
`/home/humanoid/models/GR00T-N1.7-ApplePnP-V1/exported_leapp.yaml` — the ONNX export of the model
actually fine-tuned on this corpus. The export is the authority for *what the model eats*; the
`meta/` files are the authority for *what goes on disk*.

**Added 2026-08-22.** The corpus's **own** `meta/info.json` and `meta/modality.json` are present on
this workstation (path and hashes in §0) and were diffed against the two checked-in copies above:
**byte-identical**. Every `info.json:` / `modality.json:` line citation in this section therefore
resolves the same way against either file. Where the two disagree with the export, the `meta/`
files win for on-disk layout and the export wins for the model boundary, exactly as stated above.

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

**Independently re-derived 2026-08-22**, on `data/chunk-000/episode_000000.parquet` (590 rows) in
the corpus snapshot, with `.venv/bin/python` + `pyarrow`:
`rms(action[15:22] − state[15:22]) = 0.0287 rad` against `rms(action[15:22] − state[22:29]) =
0.6292 rad` — 22× worse against the *other* arm — and `rms(action[22:29] − state[22:29]) =
0.0182 rad`. That reproduces point 3's 0.0288 on a single episode by a second route.
**n = 1 episode**, as `T40_RULE_V2` §4.3 warns; it corroborates `meta/modality.json`, it does not
replace it.

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

#### ⚠ `[?]` The per-joint hand order is ATTESTED, not MEASURED — status as of 2026-08-22

The **grouping** is measured (`meta/modality.json`: seven joints per hand, at `[29:36]` and
`[36:43]`). The **element names inside each hand** are not. They come from `exported_leapp.yaml`
alone — NVIDIA's export of NVIDIA's fine-tune of *this* corpus, which is materially stronger than
an argument from family resemblance to another Unitree hand, but is still **an assertion in an
export manifest, never checked against the parquet**. The corpus's own `meta/modality.json` carries
**no element names at all**, so there is nothing in the corpus's own metadata to check it against.

Two reasons it could not be settled here, both measured 2026-08-22 on the one local episode:

1. **The right hand never moves in `episode_000000`** — the only one of 402 episodes on this
   workstation. All seven `observation.state[36:43]` columns have per-column std ≤ 3 × 10⁻⁴ (four
   are exactly 0.0) and `action[36:43]` is identically zero to float precision. A column that does
   not vary cannot be identified. This reproduces `T40_RULE_V2` §4.2: the episode is a one-armed,
   left-side demonstration.
2. **A state↔action comparison cannot see a hand permutation**, the way it can for the arms above.
   `rms(action[29:36] − state[29:36]) = 0.498` on the *moving* left hand, against 0.0287 for the
   left arm, and the two blocks do not share a range — state left-hand values run to −0.88 rad
   while the action column is a near-binary code in [−1, +0.7]. That matches
   `configs/groot/new_embodiment_config_defaults.py`'s recorded reason for making the hand channels
   `ABSOLUTE` (*"near-binary open/close codes rather than a continuous trajectory"*). **The hand
   action block and the hand state block are not the same quantity.**

**What would settle it.** The same method that settled corpus B's block order: correlate against
the parquet, on episodes where the right hand actually moves, then discriminate the two 2-joint
pairs against either a Dex3-1 URDF's joint limits or the `action.effort_right_hand[7]` column,
whose per-element names come from the same ordering and whose torque signature differs between
fingers under a grasp. Neither is on this box today, and neither is requested here.

Consumers of this section: treat the grouping as `[OK]` and the per-joint names as `[?]`. This is
the marking `src/wam/robot/g1_dex3_28.py` already applies to `HandJointOrder.NVIDIA_ASYMMETRIC`.
A silently transposed finger pair produces a policy that closes the wrong digit, without raising
anything — which is why a probably-correct ordering is still marked unverified.

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

PR-08 §8 item 2's *"right hand index-before-middle"* — the wording used until the 2026-08-22
amendment — is **attested for corpus A's state block** by `exported_leapp.yaml:85-87` and **not
verified against corpus A's parquet either** (§2.4's `[?]` block), and is **an assumption, not a
measurement, for corpus B**. The
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

- **Not that anything has been delivered.** Under `PR-08` §1 nothing is generated until **every**
  §8 item is closed. Status as of **2026-08-22**, taking only what a registered rule states: item 2
  (this document) **CLOSED** by `T40_RULE_V7`; item 7 (T-39 has reported) **CLOSED** by
  `T40_RULE_V4` on `VERDICT N`; **items 3 and 4 remain OPEN** (`T40_RULE_V4` §0, `T40_RULE_V7` §0).
  Items 1, 5 and 6 are not adjudicated by this document — `T40_RULE_V3` §5.2 is the last record
  that examined them. §1 is conjunctive, so the prohibition binds in full. This contract describes
  the shape of a delivery, not a delivery.
- **Not that a restyled corpus can be reported inside the benchmark table.** See §6, open item 4.
- **Not that closing item 2 licenses anything.** `T40_RULE_V7` §7: *"a correction to a description
  is not permission to act on it."* No training run, on generated data or real, is licensed by this
  document or by the amendment.

---

## 6. Open items — what the consumer needs and the producer cannot currently supply

Written as gaps, not papered over.

1. **The source resolution problem is solved only on paper.** The consumer needs 640×480
   (`info.json:118-127`). `datasets/gr00t-apple-full/` is **120×160** — a 4× shortfall in each
   dimension — so the restyle must re-derive from the HF source at full resolution
   (`subprojects/data-factory/README.md:82-85`, `PR-08:101-103`). That re-derivation pipeline is not
   written.
2. **Depth and segmentation for the *corpus* are still not available; the *annotators* are no
   longer the reason.** Transfer2.5 consumes depth + segmentation + Canny; AppleToPlate ships one
   RGB camera, so on the real corpus only Canny is computable. That half of this item stands.

   > **Corrected 2026-08-25.** The clause this item used to carry — *"`isaac_binding.py` makes
   > exactly one `AnnotatorRegistry.get_annotator` call (`"rgb"`) [and] `distance_to_camera` and
   > `semantic_segmentation` are unwired, which blocks §4 of PR-08 entirely and therefore blocks
   > `EST_DRIFT_P95`, and therefore blocks G0b"* — **is no longer true of this branch, and it was
   > superseded twice over.** (i) Both annotators are wired at
   > `src/wam/robot/isaac_binding.py:188-191` and `:952-962`, opt-in via the `ground_truth=` ctor
   > arg, with tests in `tests/test_isaac_binding.py`; `PR-08` §8 item 5 is **closed**. (ii) It
   > would not have mattered either way, because `T40_RULE_V5`
   > (`docs/preregistration/PR-08-V5-ground-truth-route.md`, signed 2026-08-22) re-routed §4's
   > calibration off Isaac and onto MuJoCo, and `EST_DRIFT_P95` was in fact measured on that route
   > at **0.2361 px** (`runs/pr08-est-drift/EST_DRIFT-mujoco-s60-f720.json`, 2026-08-23).
   >
   > **The stale sentence is preserved above rather than deleted**, because it is still an accurate
   > statement about `main` — the annotator work landed on `edge-wam-e01-e05` only — and because
   > the dated note it cited (`.mc/tasks/todo/T-040-…md` Notes, 2026-08-06) is a historical
   > record that is correct as of its own date and must not be rewritten.
   >
   > [**Note, 2026-08-25 — this item gets *harder*, not easier, if the generator changes.**
   > Transfer2.5 estimates depth and segmentation on the fly (`97_transfer25_restyle.sbatch:371`).
   > **Cosmos 3 ships no depth estimator and no segmenter**, and its `transfer.py` raises *"Missing
   > pre-computed control input"* for depth/seg/WSM — so under Cosmos 3 the **producer** would have
   > to supply both maps itself. Findings and sources: `docs/cosmos3-vs-transfer25.md` §5.2, §7.3.
   > No generator change is proposed or made here.]
   >
   > **What actually blocks G0b today is item 3 below, not this item.** `EST_DRIFT_P95` and
   > `GEOM_TOL` are both *measured* but both carry `gate_qualified: false`, so **nothing was
   > written into `configs/`**: `configs/transfer25/pr08_geom_tol.json` still has `geom_tol_px`,
   > `est_drift_p95_px` and `gate_margin_px` all `null`, and G0b cannot form its error budget from
   > nulls. Do not read this correction as item 3 having moved.
3. **`GEOM_TOL` and `EST_DRIFT_P95` are measured but NOT committed, and the distinction is the
   whole of the item.** — `PR-08` §8 item 4.

   > **Corrected 2026-08-25.** This item previously read *"unmeasured and uncommitted"*. The first
   > word is now wrong and the second is still right, which is the state worth naming precisely:
   >
   > - `GEOM_TOL` = **0.4786 px** over 402/402 episodes, 171 625 frames
   >   (`docs/preregistration/PR-08-RESULT-2026-08-24-geom-tol-full-corpus.md`).
   > - `EST_DRIFT_P95` = **0.2361 px** on the MuJoCo route
   >   (`runs/pr08-est-drift/EST_DRIFT-mujoco-s60-f720.json`), margin **+0.2425 px**.
   >
   > **Both carry `gate_qualified: false`**, because the mask method
   > (`grounding-dino+sam2+depth-anything-v2`) is not gate-qualified — `scripts/estimators/apple_sam2.py`
   > sets `GATE_QUALIFIED = False` with named blockers, and that one flag stamps every artifact
   > produced on either route. **So nothing was written into `configs/`.**
   > `configs/transfer25/pr08_geom_tol.json` carries `geom_tol_px`, `est_drift_p95_px`,
   > `gate_margin_px` and `est_drift_source` as `null` to this day, and **G0b cannot form an error
   > budget out of nulls** — which is why item 4 is open and why G0b has never returned a verdict.
   >
   > **Measuring harder does not close this.** The blockers ask for evidence about the masks
   > themselves — a comparison against ground truth, and a per-frame-vs-propagation capture that
   > does not exist — not for another pass over the corpus. A signature is not that evidence.
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

## 7. Disagreements with `PR-08` §8 item 2 / `T40_RULE_V1` — **D1–D4 RESOLVED 2026-08-22**

`docs/preregistration/PR-08-photoreal-augmentation.md` is a **registered rule and has not been
edited** (project constraint; `docs/handoff.md` §3 — rules are versioned, never edited in place).
It still has not been edited.

**Resolution.** `T40_RULE_V7` (`docs/preregistration/PR-08-V7-consumer-contract.md`, registered
2026-08-22) is the superseding rule this section was written for. It disposes of **D1, D2, D3 and
D4** by replacing §8 item 2's three descriptive fields with corpus A's measured description — each
field cited to the metadata it was read from — and by keeping item 2's fourth clause verbatim.
**D5 and D6 are against the data-factory README, not against PR-08, and are unaffected.**

| | disposal |
|---|---|
| **D1** v3.0 → **v2.1** | replaced. `meta/info.json:2`, on the corpus's own metadata |
| **D2** 28-dim → **43-dim, seven groups** | replaced, with all seven slices named. `meta/info.json:17-34`, `meta/modality.json:2-60` |
| **D3** multi-camera → **exactly one, `ego_view`** | replaced; the camera is now stated in item 2 rather than implied |
| **D4** right hand index-before-middle | replaced by the **full per-joint order of both hands**, marked `[?]` — attested by `exported_leapp.yaml`, **not** verified against the parquet (§2.4). The old wording named one hand and omitted the asymmetry |

The table below is retained as the record of what was found and when. It is history, not an open
list.

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
task text, not registered rule, and can be corrected in place. **Still outstanding as of
2026-08-22:** those two copies were not corrected by the `T40_RULE_V7` change and still carry the
corpus-B fields.

### 7.1 What this repository could NOT settle about the consumer side

Recorded as the one thing the amendment cannot close from the producer's side. Checked 2026-08-22.

**Settled here.** What the producer *builds* is a v2.1, 43-dim root:
`scripts/assemble_restyled_lerobot.py` pins `CODEBASE_VERSION = "v2.1"` (`:58`), refuses a v3.0
source (`:173`), copies the source's `meta/modality.json` through verbatim (`:183`, `:568`),
rewrites only `episode_index` and `index`, carries all nine `action.*` columns untouched, and
prints `configs/groot/new_embodiment_config_defaults.py` — the seven-group 43-dim config — as the
`--modality-config-path` to use (`:601`). Nothing in this repo's *committed configuration* records
a consumer contract: `configs/transfer25/styles.toml`'s `[consumer]` table (`:534-588`) is about
which file `97_transfer25_restyle.sbatch` reads and which keys it indexes, and
`configs/transfer25/pr08_style_partition.json` carries **no `consumer` key at all** (verified:
`json.load(...).get("consumer") is None`). The table's name is the only thing that suggests
otherwise.

**Not settled.** **The consumer checkout contains BOTH routes.** At
`/home/humanoid/develop/vla-training`, HEAD `5be48ff` dated 2026-08-22:
`scripts/40_groot_prepare_dataset.sh` runs the official **v3.0 → v2** conversion and then drops the
**28-dim** `groot/modality_g1_dex3.json` into `meta/`, and `scripts/41_groot_finetune.sh:25` passes
the 28-dim `groot/g1_dex3_modality_config.py` — corpus B's route, carrying **exactly the three
fields PR-08 §8 item 2 used to name**, which is very likely where they came from. The cluster route
`eu-hub/train_apple_nvidia.sbatch:99-104` instead passes
`$ROOT/dataset/new_embodiment_config_defaults.py`, the 43-dim AppleToPlate config. Both are live in
the same tree, and a read-only checkout is not a commitment. Note also that this checkout's origin
is `https://github.com/RaaSaaR-org/vla-training.git`; nothing on this machine maps that to the name
`emai/vla-training` used throughout this repo.

**What would settle it:** a statement from the consumer side — an issue, a commit, or a written
reply — naming which modality config a delivered restyled AppleToPlate root will be trained under,
and confirming that the 28-dim `40_*`/`41_*` path is not it. Until that exists, this contract
describes **what the producer delivers**, measured on the corpus and on the assembler, and does not
claim to describe what the consumer will do with it.

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

**Added by the 2026-08-22 amendment (`T40_RULE_V7`):**

| claim | source artifact | read / measured |
|---|---|---|
| `v2.1`, 402 ep, 171 625 frames, `[43]` state and action, the seven groups | the corpus's **own** `meta/info.json` (sha256 `0e3dd494…af36`) and `meta/modality.json` (sha256 `84e5eb4f…0d62a`) in the HF snapshot `d89c126a…6ea9` | 2026-08-22, `.venv/bin/python` |
| those two files are **byte-identical** to `vla-training/eval/real_reference/{info.json,modality.json}` | `diff` over `json.tool` output of both pairs | 2026-08-22 |
| arm block boundaries re-derived: 0.0287 rad vs 0.6292 rad cross-arm; right arm 0.0182 rad | `data/chunk-000/episode_000000.parquet` (590 rows), same snapshot | 2026-08-22, **n = 1 episode** |
| right hand static in `episode_000000`: state std ≤ 3e-4 on all 7 columns, action identically 0 | same parquet | 2026-08-22, **n = 1 episode** |
| hand state↔action mismatch (0.498 vs 0.0287; disjoint ranges) — why a hand permutation is invisible to that comparison | same parquet, plus `configs/groot/new_embodiment_config_defaults.py` for the reason | 2026-08-22 |
| per-joint hand element names | `exported_leapp.yaml:77-79` (left state), `:85-87` (right state), `:259-261` / `:267-269` (decoder) — **attested, `[?]` not `[OK]`** | 2026-08-22 |
| producer builds v2.1 / 43-dim; no `consumer` key in the partition rendering | `scripts/assemble_restyled_lerobot.py:58,173,183,568,601`; `configs/transfer25/pr08_style_partition.json` | 2026-08-22 |
| consumer checkout carries both the 28-dim and the 43-dim route | `/home/humanoid/develop/vla-training` @ `5be48ff` — `scripts/40_groot_prepare_dataset.sh`, `scripts/41_groot_finetune.sh:25`, `eu-hub/train_apple_nvidia.sbatch:99-104` | 2026-08-22 |
| which route the consumer would use for a delivered restyled root | **not settled — §7.1** | — |

**Re-verify §2 in full if the base model, the export, or the corpus changes.** The export was read
once, on one date, from one artifact. Every parquet-derived claim above is **one episode of 402**.
