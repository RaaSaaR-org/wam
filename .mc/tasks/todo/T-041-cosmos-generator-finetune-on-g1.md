---
id: T-041
aliases:
- T-041
- T-41
title: Fine-tune Cosmos3-Super on G1 data — unfrozen by OD-10, pre-registered as PR-09
slug: cosmos-generator-finetune-on-g1
status: todo
priority: 3
owner: ''
projects: []
customers: []
tags:
- post-mvp
- data
- backbone
- cluster
- prereg
sprint: ''
depends_on: []
# T-39 and T-040 were hard dependencies until OD-10 (2026-08-07) decided not to wait for either.
# They are still RELATED and the relationship is argued in prose below and in PR-09 §1/§9 — T-39
# remains CRITICAL, and its verdict is what a result here must eventually be read against. They
# are no longer BLOCKING, and leaving them here would keep this task out of `mc task next` for a
# reason the user has already decided against.
due_date: ''
created: 2026-08-06
updated: 2026-08-09
status_note: "CORPUS DEDUPLICATED, 2026-08-09 — a fourth dated amendment, and the first against
  PR-09 §2 rather than §5: train 3432 → 3133 clips, 14 training sources → 13, val untouched at 30.
  Still nothing generated, nothing trained, T041_RULE_V1 still never run; this precedes every
  measurement. `g1-dex3-graspsquare-dataset` is a byte-for-byte copy of
  `g1-dex3-blockstacking-dataset` — the same six cam_left_high mp4s by sha256, the same episode
  boundaries in 79 of 80 metadata columns over 301 episodes, differing in exactly one column: the
  LeRobot task string, which reads 'camera packaging'. A third dataset's label, on a second
  dataset's name, over the first dataset's pixels. 299 duplicate pairs; 3462 clips, 3163 unique
  sha256. THE WEIGHTING WAS NOT THE PROBLEM. Four of the thirty pre-registered eval prompts — 13% of
  the eval set — were byte-identical to TRAIN clips (blockstacking 000077/000126 == graspsquare
  000077/000126, graspsquare 000224/000239 == blockstacking 000224/000239), so the LoRA would have
  been scored on footage it had memorised. Only the LoRA arm memorised it, so the bias ran TOWARD
  the registered hypothesis — the direction least likely to be questioned when the number comes back
  and the most expensive to have believed. It passed because check_prompts_are_held_out
  (eval_t041_embodiment.py:303-329 as it stood) and make_t041_eval_prompts.py:91-94 both compare
  UUIDS, and the uuids really were disjoint. scripts/dedupe_cosmos_corpus.py deletes from TRAIN
  ONLY — rule 1 the 4 contaminating clips, rule 2 the 295 remaining duplicates keeping the
  lexicographically smallest uuid — so val is byte-identical to what was registered and n=30 plus
  G0a's >=15/30 stand as registered instead of being renegotiated after the defect was known.
  No unique content was lost: corpus-wide unique sha256 is 3163 before AND after. GraspSquare now
  contributes ZERO train clips (all 297 were duplicates); the 14th source was never a distinct
  source. Its 2 val clips stay — removing them is a re-split by another name, and their captions
  were generated from the pixels, so the prompts describe what is on screen. MANIFEST_SHA256 is now
  2af81b9997f0de42e3fee01600bf34c67b7cdcb86b8ac5ab1094e21dcf77c63e, re-measured with sha256sum; the
  pre-dedupe stamp 6bec507e2816… is quoted, not re-measured, because that manifest is gone.
  THE GATE IS
  HARDENED AT BOTH ENDS: check_prompts_are_held_out (eval_t041_embodiment.py:303-368) and
  make_t041_eval_prompts.py:96-117 now also refuse a prompt whose clip sha256 appears anywhere in
  train, read from the sha256 the manifest already records so nothing is hashed at eval time. The
  uuid check is KEPT, not replaced — it catches a prompt set built from the wrong split, which a
  sha comparison would not notice. Confirmed it cannot change the current outcome: it passes against
  the real manifest (30 prompts, all in val, none byte-identical to train) and names all four pairs
  against a reconstruction of the corpus as it was. Four new tests in
  tests/test_eval_t041_embodiment.py:327-378.
  ===
  NUM_FRAMES REGISTERED, 2026-08-09 — `num_frames` 189 → 397, closing the FIRST of the
  two items the geometry/fps amendment left open. Still nothing generated, nothing trained,
  T041_RULE_V1 still never run. Measured, not asserted: vision_sft_super.py:271 sets
  num_video_frames=-1 (native-chunk mode, sft_dataset.py:215-219) and captions_to_sft_jsonl.py:172-174
  writes every window as start 0 / end total-1 / temporal_interval 1, so the manifest's frame counts
  ARE the training sequence lengths — 3432 train clips at 30.0 fps, min 249 / p05 356 / p25 464 /
  median 693.5 / p75 911 / max 1819 (8.3 s to 60.6 s), and NOTHING at or below 189 in either split.
  Correction to the open note: the 256 bucket's 400 is NOT a cap. MAX_NUM_FRAMES[\"256\"]=400
  (args.py:146) is only compared in a log.warning (args.py:529-532); the sole rewrite is the 4N+1
  round-up at args.py:536-538. 397 is chosen to stay inside a range NVIDIA states, not because
  anything would reject 401. Of the legal 4N+1 values only 397 is in the distribution's interior —
  12.97% of train clips are <=397, against 4.22% at 349, 0.20% at 297 and 0.03% (ONE clip) at 249 —
  and 13.2 s is the closest this API comes to the duration the structured-JSON prompt itself states
  (val median 25.3 s). COST, back-of-envelope and labelled so in PR-09 §5: the 8-GPU benchmark column
  does not apply, because parallelism_preset=\"throughput\" forces cp=cfgp=1 (args.py:1364-1378) with
  dp_shard=world and 95:137-140 hands torchrun ONE payload per launch, so ~40 s/clip at 189 and
  ~85 s/clip at 397 — a marginal ~45 min (~6 GPU-h) over 60 clips. What actually threatens the 4 h
  wall is the 60 COLD torchrun launches, each loading a 64.6 GB DCP checkpoint, ~90-180 min and
  independent of num_frames; no measurement of that startup exists yet. Accepted because the job is
  restart-safe by construction (95:139 skips written clips, judge --resume, --requeue); mitigations
  are named in PR-09 §5 and NONE is applied — batching the payloads into one torchrun
  (scripts/inference.py:22-27 takes -i as a glob list) is the highest-leverage and is deliberately
  not made under cover of a frame-count decision. Also recorded and NOT resolved: §7 budgets job 95
  at 8 GPU-h = one hour on 8 GPUs, and every branch of the estimate puts the eval at 25-35 GPU-h at
  189 as much as at 397.
  ===
  GEOMETRY AND FPS CORRECTED, 2026-08-09 — still nothing generated, still nothing
  trained. The 2026-08-08 entry below is WRONG ON THE MECHANISM and is left standing, superseded by
  the dated amendment in PR-09 §5: `resolution` is not an output height, it is a key into
  `VIDEO_RES_SIZE_INFO` (cosmos_framework/data/generator/utils.py:42-74), whose only 4:3 buckets are
  256→320×256, 480→736×544 and 720→1104×832. `480`/`4,3` is 736×544, NOT the 'exactly 640×480' that
  entry and PR-09 §5 both claimed, and there is no 640×480 bucket at all — the corpus geometry is
  unreachable through this API, so the registered value could never have done what it was registered
  to do. The settings now match TRAINING instead: vision_sft_super.py:272 pins `resolution=\"256\"`
  and the TOML cannot override it (DataloaderTrainConfig forbids extras, sft_config.py:624-665), so
  320×256 is the only geometry the adapter ever sees; `max_sequence_length=45056` is sized for that
  bucket, and at 736×544 the median 693-frame clip needs ~68k tokens and would be dropped SILENTLY
  by PackingDataLoader. `fps = 24` → 30 in the same amendment, on its own evidence rather than under
  cover of the first: conditioning_fps=-1 passes each clip's own fps, the corpus is 30.0 throughout,
  and 24 would have asked mRoPE for a 1.0 temporal stride against the 0.8 training was fit on.
  T041_RULE_V1 untouched. TWO ITEMS OPEN AND WRITTEN DOWN AS OPEN: `num_frames = 189` is shorter
  than the shortest training clip (249; median 693) and no replacement is registered — the largest
  legal 4N+1 under the 256 bucket's cap of 400 is 397 [CLOSED the same day, see the entry above:
  397 registered, and the '400 cap' is a warning rather than a cap]; and G0b's calibration clips are
  real 640×480 footage that no bucket reproduces, so they must be downscaled to 320×256 before the
  judge sees them, which is NOT DONE and blocks G0b — STILL OPEN.
  WORKSTATION ENV BUILT, 2026-08-08 — step 00 runs green and idempotent; both repos at
  their pinned SHAs with clean working trees; torch 2.10.0+cu128 sees the RTX 5090 as sm_120; both
  captioner entry points import. Three prerequisites were undeclared, and each failed while naming
  the wrong culprit. (1) No git-lfs: cosmos-framework LFS-tracks assets/** and every media
  extension, so the clone succeeded and the CHECKOUT died half-way, leaving a repo where
  `git rev-parse HEAD` printed the pinned SHA over a gutted tree — the '=== framework @ <sha>' line
  was a lie. `clone_at` now requires a clean porcelain, in cluster/discoverer/90 as well, WHERE THE
  git-lfs INSTALL SAT AFTER THE CLONES IT WAS NEEDED FOR and would have reproduced this at Slurm
  queue cost. (2) No C compiler: `uv sync --all-extras` builds evdev from source (lerobot → pynput →
  evdev, a keyboard-teleop transitive nothing here uses, which --all-extras gives no way to decline)
  and failed ten minutes into a multi-GB resolve. (3) transformer_engine probes for a system CUDA
  toolkit via nvrtc and curand and, finding neither, re-loads cudart from the pip wheels under the
  CUDA 13 directory name `nvidia/cuda_cudart` while the cu12 wheel installs `nvidia/cuda_runtime`.
  This can only fire where there is no toolkit — Discoverer+ loads a CUDA 12.8 module, a driver-only
  workstation has no /usr/local/cuda — so the cluster recipe was never wrong, it just never reached
  that branch. Step 00 aliases the directory. Separately the `curl -LsSf astral.sh | sh` uv
  bootstrap was deleted rather than repaired: an unpinned installer that silently changes what
  `uv sync` resolves is the one unversioned component in a pipeline whose whole premise is that
  every part is named by SHA. uv is a prerequisite now. Also added imageio to the dev extra, which
  clears the two long-standing test_cosmos3_probe failures — the T-041 suites are 86/86 green.
  ===
  RESOLUTION MISMATCH CLOSED + CORPUS FETCHED, 2026-08-08 [SUPERSEDED 2026-08-09 — the
  `480`/`4,3` = 'exactly 640×480' below is false and the fps item is no longer open; see the top
  entry and PR-09 §5's 2026-08-09 amendment] — still no training
  submitted. PR-09 §5 generated 720p 16:9 against a corpus that is 640×480 4:3 throughout; the
  settings came from the cookbook's payload example and were never checked against our own data.
  Both arms shared them, so a false P was never possible — the risk was an ambiguous N, where 'the
  LoRA does not fix the embodiment defect' and 'it does, but not at 16:9 720p' return the same
  verdict. `t041_eval_selection.toml` now generates `480`/`4,3` = exactly 640×480, matching both
  the corpus and G0b's real calibration clips; `T041_RULE_V1` is unchanged and the amendment is
  dated in PR-09 §5, taken before any clip existed. Moving the corpus instead was rejected:
  pillarboxing teaches the adapter to draw bars, cropping to 640×360 discards the torso/arms/hands
  that `cam_left_high` was chosen for. `fps = 24` vs a 30 fps corpus is the same class of mismatch
  and is left OPEN on purpose. Workstation now has ffmpeg 8.1.2 (libdav1d + libx264/nvenc) and the
  full corpus: 14/14 sources, 26 GB, one camera each.
  PREPARATION MOVED OFF THE CLUSTER, 2026-08-08 — still no training submitted. Two
  format blockers were found and fixed in code: the 13 G1_Dex3_* sets are LeRobot v3.0 (episodes
  concatenated, boundaries in meta/episodes/*/*.parquet, cameras rolling over to new files
  independently), and ALL 14 sources are AV1, which vLLM's OpenCV cannot decode — job 186357
  captioned 372 clips, produced 0 captions and exited 0. `prepare_cosmos_corpus.py` now reads both
  layouts and transcodes to H.264; `scripts/verify_clip_decode.py` gates captioning on the
  captioner's OWN interpreter decoding every clip. Fetch/prepare/caption now run on a workstation
  (`workstation/`), the corpus is defined once in
  `configs/cosmos3/corpus_g1_embodiment.tsv`, and jobs 92/93 are marked superseded. Every T-041
  failure so far has been IO, format or scheduling — none in training.
  UNBLOCKED AND READY TO SUBMIT, 2026-08-07 — nothing has been submitted. The pipeline
  is `docs/preregistration/PR-09-cosmos-super-finetune.md` (rule `T041_RULE_V1`) plus
  `cluster/discoverer/90..95`, `scripts/prepare_cosmos_corpus.py`, `make_t041_eval_prompts.py`,
  `eval_t041_embodiment.py` and `configs/cosmos3/t041_eval_selection.toml` (51 tests).
  THE FREEZE IS LIFTED: OD-10 (2026-08-07, by the user) lifts PR-07 §7's *Cosmos3-Super generation*
  clause only — T-32 and Cosmos3-Edge stay frozen, and PR-07 is NOT edited, because rules here are
  versioned rather than amended. The jobs still refuse to run unless `T041_FREEZE_LIFTED` names a
  reason, and that string lands verbatim in `run_metadata.json`, so every artifact carries the
  decision that allowed it. PR-09 §8 item 6 is the only open item and it is a measurement, not a
  document: 8-GPU VRAM on dgx1, taken by the mandatory probe.
  TWO EARLIER BLOCKERS ARE GONE. (1) Captions: NOT our pipeline to build. cosmos-framework ships
  `caption_from_video` (Qwen3-VL-8B-FP8 via vLLM, two-phase) and `captions_to_sft_jsonl`; that
  'largest unpriced item' was answered by reading the framework, not by building anything.
  (2) Licence: PR-09 §2 takes the embodiment-fidelity goal, whose corpus is AppleToPlate
  (CC-BY-4.0) + the 13 unitreerobotics/G1_Dex3_* sets (Apache-2.0). Humanoid Everyday is the
  VARIETY corpus and that goal was dropped for the reasons in PR-09 §2 — so OD-09 is not engaged
  by this task at all any more."
---

# Fine-tune Cosmos3-Super on G1 data — unfrozen by OD-10, pre-registered as PR-09

## Description

**Idea.** Instead of restyling with a frozen Transfer2.5 ([[T-040]]), post-train a Cosmos generator
(Super, or a smaller Predict2.5 variant) on our own G1 + Dex3 footage, then generate with it.

**Why it is the better fix, and why that is not enough.** The embodiment defect — both priors
render a generic manipulator where the G1's arm should be
(`runs/backbone_eval/video/embodiment_grid.png`) — is the defect most likely to poison a VLA, and
T-040 can only patch it downstream by compositing the real robot back over the generated frame. A
generator that has actually seen a G1 would not produce it in the first place. That is a real
argument. What it buys is paid for by turning "augment with a frozen tool" into "train a video
model", which is the activity fourteen recorded negatives in this repo are about.

## Why it *was* frozen — kept, because OD-10 is a decision taken against these reasons

> **Lifted 2026-08-07 by OD-10**, narrowly: PR-07 §7's *Cosmos3-Super generation* clause only.
> T-32 and Cosmos3-Edge remain frozen. **PR-07 is not edited** — the rule stands as written and
> the decision is recorded beside it. None of the three reasons below was refuted; the third in
> particular is still true, and PR-09 §9 forbids reading a P here as evidence about T-040's
> question. What changed is the judgement that they are worth 122 GPU-h (2.4 % of the allocation)
> to act against, given that T-39 is not a short wait — it is unsubmittable with PR-07 §8 items
> 4–6 open.

- **PR-07 §7, by name:** *"Frozen until T-39 reports: T-32 (§2), any Cosmos3-Super generation, any
  Cosmos3-Edge work."* Not an inference from the spirit of the freeze — the words.
- **Nothing here has ever measured Super.** That freeze line is its only appearance in the
  repository. Cosmos3-**Nano** (16B MoT: 8B AR reasoner + 8B diffusion generator) peaked at
  **36.2 GB** in *inference* on ZeroGPU, over the 5090's 34.36 (T-24, OD-04). Super is larger, and
  the gap between running a model and fine-tuning it is not small.
- **Ordering.** T-040 has to establish that restyled data helps *at all* with a frozen generator
  before the expensive version of the same idea is worth pricing. If T-040 is null, this is moot.

## What G1 data it would need

**Video, not actions.** A generator fine-tune consumes (clip, caption) pairs plus the control
signals if it is Transfer-family. Joint states play no part in it; they matter only downstream,
when a restyled clip becomes VLA training data.

**What exists.** `nvidia/GR00T-N1.7-AppleToPlate`: 402 real teleop demos at 30 Hz, mean 427 frames
(range 249–749) ⇒ ~172 000 frames ≈ 95 minutes, one head RealSense D435 RGB at 640×480. Our
converted copy is 120×160 (`datasets/gr00t-apple-full/*/manifest.json`) and is not usable for
this — the source resolution is required.

**Volume is probably not the binding constraint. Diversity is, and ours is 1.** NVIDIA fixed the
scene by protocol — black tablecloth, white wall, ~75 cm table, red apple, white plate, consistent
pelvis and head pose (`real-record.html`). A generator fine-tuned on that learns to render *one
room*, which is circular against the stated goal of producing visual variety. The two goals
separate cleanly and only one is reachable with data we have:

| goal | corpus that would serve it |
|---|---|
| **Fix the embodiment defect** — stop it painting a generic manipulator where the G1 is | the 402 demos plausibly suffice as a subject fine-tune, and better pooled with the **UnifoLM collection: 13 public G1 + Dex3 sets, exact embodiment, Apache-2.0** (e.g. `unitreerobotics/G1_Dex3_ToastedBread_Dataset`, 418 episodes — `vla-training/scripts/00_fetch_dataset.sh`). Different tasks and scenes, same robot, free |
| **Teach visual variety** | **Humanoid Everyday** — see below. Do *not* read this row as "nothing we hold, it needs PR-04 collection"; that was the standing assumption until the corpus below was measured on 2026-08-05 |

### Humanoid Everyday — the corpus that changes this task's outlook

`USC-GVL/humanoid-everyday` (arXiv:2510.08807, Physical Superintelligence Lab / USC-GVL,
`github.com/physical-superintelligence-lab/Humanoid-Everyday`), checked 2026-08-06:

**Validated against the data itself on 2026-08-06** — `meta/info.json`, `meta/episodes.jsonl` and
five downloaded episodes spanning five task categories, not against the README's prose. Where the
two disagree, the measurement is recorded and the README is marked wrong.

**The repositories were renamed: `USC-GVL/*` → `USC-PSI-Lab/*`.** The old names redirect; the
datasets-server rejects them.

### Measured

| | |
|---|---|
| `USC-PSI-Lab/Humanoid-Everyday-G1` | `robot_type: "g1"` — **already filtered to G1**, no H1 to exclude |
| scale | **4 064 episodes · 1 779 287 frames · 16.5 h @ 30 fps** = **10.4×** AppleToPlate's 1.6 h |
| episode length | min 116 · median 409 · mean 437 · max 2402 frames |
| **size** | **3.21 GB video + 0.63 GB parquet = 3.84 GB** |
| video | h264, **640×480**, yuv420p, **30/1 fps**; frame count matches parquet rows exactly (515 = 515 on ep 0) |
| state | `arm_joints` **14** · `leg_joints` **15** · `hand_joints` **14**; `action` **28** — identical across all five sampled episodes |
| tasks | 247 in 7 categories: Basic 64, Articulated 48, **Locomanip 46**, HRI 41, Tool_use 21, deformable 19, Precision 8 |
| format | LeRobot **v2.1** (`codebase_version`), one parquet + one mp4 per episode |
| licence | **none.** The G1 repo declares no licence in any field, tag or file — see "Licence" below, re-checked against primary sources 2026-08-06. The `cardData.license: apache-2.0` that does exist on record belongs to the *mixed* repo and must not be carried across to this one |

**3.84 GB, not 935 GB.** That figure belongs to the *mixed* repo
(`USC-PSI-Lab/humanoid-everyday`: `robot_type: "mixed"`, 8 949 episodes, 3 436 171 frames), which
is inflated by float32 depth (480×640), LiDAR, IMU, odometry and tactile — not by the RGB. **The
G1 RGB corpus fits on a laptop and needs no cluster transfer to start.** Take the mixed repo only
if depth is actually wanted (it is, for [[T-040]] — see there).

### Two README claims that are false

- **"The lite sets are states and actions only."** They are not. `Humanoid-Everyday-G1` ships
  **4 064 mp4s** under `videos/chunk-*/egocentric/`. This is the repo to use, not the 935 GB one.
- **Action block order is `[hand(14), arm(14)]` — hand first.** The README's example builds
  `arm_actions + hand_actions`, arm first. Measured by correlating each action block against the
  recorded state, over all five episodes: `action[0:14]` ↔ hand **0.61–0.67** against arm
  0.28–0.40; `action[14:28]` ↔ arm **0.74–0.92** against hand 0.30–0.43. **Our Dex3 IL format is
  arm-first, so a swap is required**, and anyone following the README while loading the LeRobot
  version gets arm and hand silently transposed. Since the block order in the README is wrong, the
  *intra-hand* order it documents (thumb-first, symmetric) is **not** to be trusted either without
  the same kind of check.
  **THIS FINDING IS ABOUT THIS CORPUS ONLY — added 2026-08-15 after it was carried across and got
  five documents wrong.** It holds for `USC-PSI-Lab/Humanoid-Everyday-G1` (LeRobot v2.1, separate
  `arm_joints`/`leg_joints`/`hand_joints`). The `unitreerobotics/G1_Dex3_*` sets are LeRobot v3.0
  with a flat 28-dim state and are **arm-first** — `[0:14]` arm, `[14:28]` hand (T-043 §1). Block
  order is a per-corpus measurement, never an inherited constant.

### From the README (not independently verified)

- **The camera is ours**: one egocentric **Intel RealSense D435**, and the README additionally
  publishes per-platform **colour and depth intrinsics plus depth-to-colour extrinsics for the
  G1** — which NVIDIA does *not* for AppleToPlate, where
  `vla-training/docs/apple-pnp-ursachen.md` §2.5 records FOV, intrinsics and head→camera
  extrinsics as **unbelegt**. On that axis this corpus is better characterised than our own.
- **Hand ordering**: documented as thumb-first and *symmetric* across hands — `[ThumbRotation,
  ThumbLower, ThumbUpper, MiddleLower, MiddleUpper, IndexLower, IndexUpper]`. Arena's YAML is
  index-first; NVIDIA's real ApplePnP pipeline is asymmetric, left `[4,5,6,2,3,0,1]` and right
  `[4,5,6,0,1,2,3]`. A third mapping. **Irrelevant to a generator fine-tune** (which consumes
  video), blocking if this corpus ever feeds a policy — and see above: the README was wrong about
  the *block* order, so verify this one against data before relying on it.
- **The robot walks.** `leg_joints` is 15-dim for G1 (12 leg + waist yaw/roll/pitch) and
  **Locomanip is 46 of 247 tasks**. AppleToPlate is a gantry-mounted static G1, torso almost
  touching a 75 cm table, and teleoperated by a different rig (Apple Vision Pro here). For a
  *generator* that scene-geometry difference is the feature; for a policy it is distribution
  shift, and this task must not blur the two.
- Depth and LiDAR live only in the **mixed** repo, not in the G1 one — relevant to [[T-040]] more
  than to this task; see there.

### Licence — unresolved, checked against primary sources 2026-08-06

**No `LICENSE` file exists in any primary location, so nothing arbitrates the conflict — and the
repository we would actually use states no licence at all, in any form.** Verbatim, each statement
attached to the URL it came from:

| source, checked 2026-08-06 | what it says |
|---|---|
| `https://huggingface.co/api/datasets/USC-PSI-Lab/Humanoid-Everyday-G1` — **the repo we would use** (sha `6b6b599d`, lastModified 2025-11-27) | **Nothing.** No top-level `license`, and **no `cardData` key at all** — the response's top-level keys are exactly `_id, author, createdAt, disabled, downloads, gated, id, lastModified, likes, private, sha, siblings, tags, usedStorage`, identical under `?full=true`. The key is absent, not empty. No `license:*` entry among its `tags`. Of 8 133 files the only non-`data/`, non-`videos/` entries are `.gitattributes` and `meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}` — **no LICENSE and no README.md**. `…/raw/main/README.md` → **HTTP 404, "Entry not found"**. `…/commits/main?limit=100` returns **22 commits**: 21 titled *"Add files using upload-large-folder tool"* and one `initial commit` (`455a45cc`, author `zhenyuzhao`, 2025-11-27 04:37 UTC, empty message). No commit ever added or removed a card, so a card never existed rather than having been taken down. The `…/croissant` view carries no licence field either |
| `https://huggingface.co/api/datasets/USC-PSI-Lab/humanoid-everyday` — the **mixed** repo, *not* the one we would use (sha `71f6210e`, 2026-06-02) | `cardData.license: "apache-2.0"`, tag `license:apache-2.0`. Its `…/raw/main/README.md` carries `license: apache-2.0` on line 2 of the YAML frontmatter — the **only** occurrence of the string "licen" in the whole 60-line card. The prose has no License section. 26 869 files, no LICENSE file |
| `https://huggingface.co/api/datasets/USC-GVL/humanoid-everyday` | **HTTP 307** — *"Temporary Redirect. Redirecting to /api/datasets/USC-PSI-Lab/humanoid-everyday"*. There is no separate USC-GVL repo that could carry a third statement; the author listing for `USC-GVL` is empty. The sibling `USC-PSI-Lab/Humanoid-Everyday-H1` is in the same state as the G1 repo: same 14 top-level keys, **no `cardData` key**, no README, no LICENSE |
| `https://github.com/physical-superintelligence-lab/Humanoid-Everyday` (default branch `master`, head `7c427fdb`, 2026-06-21) | The README's last three lines, verbatim: `# License` / *"This dataset is released under the MIT License"* — no full stop, no licence text, end of file. **The repo contains no LICENSE file**: `GET repos/…/license` → **HTTP 404 Not Found**, the repo object has `"license": null`, and a recursive tree of `master` matches nothing on `licen|copying|terms|notice`. A README sentence is not a licence |
| `https://arxiv.org/abs/2510.08807` (v3, 04 Jul 2026) | The **paper** is `License: CC BY 4.0` — arXiv's licence on the paper, not on the data. The full text contains **no** dataset licence, data-use or terms statement (every apparent "MIT" hit is a substring of *limited* / *imitation*) |
| `https://humanoideveryday.github.io` — **the real project page**, fetched 2026-08-06 | **HTTP 200**, `<title>Humanoid Everyday</title>`, 20 446 bytes. It is the URL the paper itself gives: it is the only `humanoideveryday` link in the arXiv HTML of v3. Its **raw HTML contains 0 case-insensitive matches** for `licen`, `terms`, `copyright` or `data use` — not in the visible text (3 493 chars extracted), not in markup or hrefs; its single sub-page `cloud_evaluation_coming_soon.html` (HTTP 200) matches 0 as well. The page carries abstract, overview figure, task distribution, a "Coming Soon" cloud-evaluation button, BibTeX and author links, and **says nothing at all about licensing, terms or permitted data use**. So the silence is not an artefact of looking in the wrong place — there is no terms page because the project page states no terms. (The link in the GitHub README, line 314, is a typo — `https://humaoideveryday.com`, which does not resolve; the correctly spelled `humanoideveryday.com` is an unrelated **domain-for-sale parking page**, Spaceship.com, and belongs to nobody in this project) |

**The conflict does not resolve, and it is worse than "two names disagree".** MIT exists only as a
sentence in the README of a *code* repo; `apache-2.0` exists only as a metadata field on a
*different* dataset repo; and `Humanoid-Everyday-G1` — the 3.84 GB corpus this task and [[T-040]]
want — states nothing, in no field, no tag, no card and no file. That makes it **unlicensed data,
not permissively licensed data**, and the two permissive names on record cannot be borrowed across
repositories to cover it.

One near-miss, recorded so it is not rediscovered: the only `LICENSE` file anywhere in this
project's GitHub footprint sits in the anonymous review mirror
`github.com/anonymouse5202077/Humanoid-Everyday` (SPDX `NOASSERTION`), and its text is Apache-2.0
under `Copyright [2024] [HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")]` — the inherited
licence of the Unitree teleoperation *code* the pipeline was built on. It is not the official
repository and it does not licence the recordings.

**Who would have to be asked, and for what exactly.** The authors are USC (Zhenyu Zhao, Hongyi
Jing — equal contribution; Jiageng Mao, Yue Wang — equal advising) with Toyota Research Institute
(Sergey Zakharov, Vitor Guizilini); no email address is published in the arXiv HTML. Two public
channels exist: an issue on `github.com/physical-superintelligence-lab/Humanoid-Everyday` (issues
are enabled, 0 open) and a discussion on the `USC-PSI-Lab/Humanoid-Everyday-G1` HF page. The ask is
three specific things, not "what is the licence": **(1)** put a `LICENSE` file — or a `license:`
field in the card — on `Humanoid-Everyday-G1` *itself*; **(2)** state which of MIT (GitHub README)
and Apache-2.0 (mixed-repo card) governs the **data**, given that they contradict; **(3)** confirm
whether the Unitree Apache-2.0 code lineage above bears on the recordings. Until (1) exists, no
weight is trained on this corpus. That was an **unresolved dataset-licence blocker owned by this
task and by [[T-040]]** — it stands on its own evidence (the table above) and is not the
consequence of any decision recorded elsewhere in this repo.

> **Superseded 2026-08-07 by OD-09 — by decision, not by evidence.** Everything above still holds
> as a factual finding; none of it was refuted. The user weighed it and chose to train on the
> corpus anyway, on the basis of the EU TDM exception (Art. 4 DSM / §44b UrhG: mining lawfully
> accessible works, no machine-readable reservation present). Read the section as *the evidence the
> decision was taken against*, not as a live blocker. OD-09 records what the decision does **not**
> cover — redistribution, and selling or serving a model trained on this — and names that as its
> review trigger. The sentence "until (1) exists, no weight is trained on this corpus" no longer
> reflects the project's position.

**Gap, now filed as OD-09 (2026-08-07).** As originally written: no row in the open-decisions table covers *dataset* licensing —
OD-04 is "Open fallback backbone + license" (`TASKS.md:201`, `docs/ROADMAP.md:17`), a **model**
licence decision, ✅ closed on Wan2.2-TI2V-5B and challenged-and-held 2026-08-05 by T-37, saying
nothing about data; the rest of `TASKS.md:198-205` is platform (OD-01), action space (OD-02),
cameras (OD-03), hardware budget (OD-05), FLUX 3 access (OD-06), teleop (OD-07) and vendor
controller safety (OD-08), of which only OD-06 and OD-08 are still open (`README.md:57`) — so this
blocker currently has nowhere to be recorded, and whether to open such a row is the user's call,
not this task's.

AppleToPlate's own terms are likewise unverified here, and the UnifoLM sets are stated Apache-2.0
in `vla-training` and should be confirmed from their cards.

### How it was checked

```bash
# metadata only, no bulk download
curl -sL https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-G1/resolve/main/meta/info.json
# five episodes across five task categories (~4 MB)
python -c "from huggingface_hub import hf_hub_download; ..."   # ep 0, 500, 1500, 2500, 3900
# block order established by correlating action[0:14] / action[14:28] against the recorded state
```

## The recipe exists, and it is a LoRA — measured 2026-08-06

Checked against NVIDIA's own sources, not a blog post: `nvidia/Cosmos3-Super` on HF and
`nvidia/cosmos` `cookbooks/cosmos3/generator/audiovisual/finetune/`.

| | |
|---|---|
| model | `nvidia/Cosmos3-Super`, **64B** MoT (≈32B AR reasoner + 32B diffusion generator), 132.7 GB bf16, **not gated**, released 2026-05-31 |
| licence | **OpenMDW-1.1** (`license_link: openmdw.ai/license/1-1/`); card states *"ready for commercial and non-commercial use"*. The recipe files themselves carry `SPDX-License-Identifier: OpenMDW-1.1` |
| official recipe | `launch_sft_vision_super.sh` + `toml/sft_config/vision_sft_super.toml` — **LoRA-only** SFT, T2V/I2V/V2V, README says *"Tested on 8×H100 (80 GB)"* |
| what trains | `lora_rank 16`, `lora_alpha 32`, targets `q,k,v,o_proj_moe_gen` (**generation tower only**), `optimizer.keys_to_select = ["lora_"]`. The 64B optimizer-state problem does not arise |
| parallelism | FSDP, `context_parallel_shard_degree = 2`, DP=4, full activation checkpointing, `max_iter = 500`, `grad_accum_iter = 2`, `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| Nano/Edge | full fine-tunes (`vision_sft_nano`, `vision_sft_edge`); Edge is 2B dense and fits 4 GPUs |

This answers four of the six questions below outright: the model id, the licence, the VRAM
question (8×H200 141 GB is strictly more than the 8×H100 80 GB it is tested on) and the recipe.
**It does not lift the freeze** — PR-07 §7 is about whether generating anything is answering the
right question, not about whether it is technically possible.

### Two things the recipe demands that we do not have

- ~~**Structured-JSON captions.**~~ **Resolved 2026-08-07 — the captioner is NVIDIA's and we do not
  build it.** The claim below was right about the requirement and wrong about the cost. The loader
  does consume `caption_json`, and neither corpus ships one. But `cosmos-framework` ships the
  pipeline that produces it: `cosmos_framework.scripts.caption_from_video` drives a
  `Qwen/Qwen3-VL-8B-Instruct-FP8` vLLM server through two phases (structured-JSON scene analysis,
  then a dense narrative rewrite that is embedded back as `temporal_caption`), and
  `captions_to_sft_jsonl` assembles `video_dataset_file.jsonl` — mirroring the loader's own silent
  filters so the count matches what trains. That is one GPU for a few hours
  (`93_caption_corpus.sbatch`), not a project. **Recorded as a correction rather than edited away:
  "this is its own pipeline, cost it before anything else" was an inference from the requirement,
  and the requirement was real. What was missing was reading `docs/dataset_jsonl.md` §"Video
  Captioning" to the end.** Original text: *"Captioning ~18 h of video is its own pipeline —
  plausibly Cosmos-Reason2 — and an uncaptioned corpus cannot enter this recipe at all. Cost this
  before anything else."*
- **Action conditioning is not available for our embodiment.** Super's card lists supported action
  inputs — camera 9D, AV 9D, egocentric 57D, Franka 10/20D, Agibot 29D, UR/Google/WidowX 10D, UMI
  9D. **No humanoid, no G1, no 28-dim Dex3.** The action-conditioned SFT cookbooks are
  `..._nano.sh` only. So this task is a **video** fine-tune, as its own §"What G1 data it would
  need" already says — which is consistent, but it means Super cannot be turned into an
  action-conditioned G1 world model by this route.

### Generation cost, now that a vendor number exists

The card gives **~55 s for a 189-frame video at 50 steps on 8×H200** (vLLM-Omni, the recommended
serving config; 3 min on 2×H200). One AppleToPlate-scale variant is ~172 000 frames ⇒ ~910 clips ⇒
**~14 h wall on a full 8-GPU node ≈ 111 GPU-h per variant**, before any re-rolls. Against the
5 000 GPU-h allocation that is ~2 % per variant — affordable, and far more expensive than
Transfer2.5-2B ([[T-040]]) for the same frame count. It also needs the whole node for *inference*,
not just training, which the 4 h `MaxWall` and `MaxJobsPU=4` have to absorb.

## What must be answered before it is unfrozen

- [x] Which model, from a **primary source** — `nvidia/Cosmos3-Super`, 64B. **Revision pinned
      2026-08-07: `e0262be9d8f7586bc24c069a2aed2b665bdff266`** (HF API, 134.6 GB, not gated),
      hard-coded in `91_stage_cosmos_weights.sbatch`.
- [x] Whether the licence permits fine-tuning and downstream use — OpenMDW-1.1, above. Still read
      the licence text before training; "permissive" is not a reading.
- [x] A GPU-h estimate for **training** — **not estimated, measured, and made a gate.**
      `94_train_t041_cosmos_super.sbatch` runs `PROBE=1` first: two torchruns at 5 and 25
      iterations into throwaway output roots, so the per-iteration cost separates from the DCP
      load by subtraction instead of by parsing NVIDIA's log format. If 500 iterations do not fit
      the pre-registered 96 GPU-h / 3-pass ceiling, **the run is not started and the shortfall is
      the finding** (PR-09 §7). The ceiling is not raised to fit the recipe.
- [x] VRAM feasibility per H200 (141 GB) — the recipe is tested on 8×H100 80 GB, so the headroom
      is real. Confirm on the actual node before claiming it: the probe is that confirmation.
- [x] **The captioning pipeline** — see the correction above. It is NVIDIA's, it is shipped, and it
      is `93_caption_corpus.sbatch`.
- [x] Which of the two goals it is pursuing — **embodiment fidelity**, fixed in PR-09 §2 so the
      corpus cannot be reselected after a disappointing result. Visual variety is dropped, with
      this task's own argument as the first reason: a rank-16 LoRA on ~18 h learns *those* rooms,
      not arbitrary rooms.
- [x] Its own pre-registration — `docs/preregistration/PR-09-cosmos-super-finetune.md`,
      rule `T041_RULE_V1`. It does not inherit T-040's and does not lift PR-07 §7.
- [x] **PR-09 §8 items 3 and 4** — the eval *selection* (`configs/cosmos3/t041_eval_selection.toml`
      + `make_t041_eval_prompts.py`) and the §6 scorer (`eval_t041_embodiment.py`, 31 tests,
      `95_eval_t041_embodiment.sbatch`), both in git before generation. Item 3 was amended: the
      prompt *text* is a caption job `93` produces, so what is pre-committed is the deterministic
      selection rule, not the strings.
- [ ] **PR-09 §8 item 6** — 8-GPU VRAM on dgx1. The only open item, and it is a measurement rather
      than a document: the mandatory probe takes it.

## What was built 2026-08-07, and what it does not do

| | |
|---|---|
| `docs/preregistration/PR-09-cosmos-super-finetune.md` | goal, arms, `T041_RULE_V1`, the 122 GPU-h ceiling, what a P cannot be read as |
| `scripts/prepare_cosmos_corpus.py` + 20 tests | LeRobot → the clip tree the captioner eats, at **source** resolution; mirrors the loader's silent >61 s / <61 frame filters; seeded split; per-clip sha256 manifest (AC-04) |
| `90_build_cosmos_env.sbatch` | cosmos + cosmos-framework at pinned commits, `--group=cu128-train` (this cluster's CUDA is 12.8), its **own** venv — never `virt_envs/wam` |
| `91_stage_cosmos_weights.sbatch` | the pinned revision + Wan2.2 VAE + DCP conversion |
| `92_fetch_g1_corpus.sbatch` | AppleToPlate + the 13 `G1_Dex3_*` sets, meta+videos only, camera keys printed and never guessed |
| `93_caption_corpus.sbatch` | NVIDIA's captioner; fails if the jsonl count disagrees with the manifest |
| `94_train_t041_cosmos_super.sbatch` | the probe gate, the chained 4 h passes, and the resume patch below |

**The one trap worth naming, because it produces a plausible wrong number rather than a crash.**
The shipped TOML sets `checkpoint.keys_to_skip_loading = ["net_ema.", "lora_"]`. Correct on a cold
start — the base checkpoint has no adapter and it must init fresh. On a **resume** from
`iter_<N>/`, that same line skips the adapter we just trained and re-initialises it, while
`optim/`, `scheduler/` and the iteration counter in `trainer/` *are* restored. The run continues
from iteration 300 with a fresh adapter and a stale optimiser, logs plausible losses, and writes a
checkpoint that is not what its metadata says. With a 4 h walltime and `PreemptMode=REQUEUE`,
resuming is not the exception. `94_*` generates the resume TOML with that one key changed and
**diffs it into the job log**; a pass without that diff is not a valid resume (PR-09 §6 G0c). The
guard is scoped to that line only — `[optimizer].keys_to_select = ["lora_"]` must *stay*, or the
run silently becomes a 32B full fine-tune.

## What actually ran (2026-08-12 … 2026-08-15)

The freeze was lifted by OD-10 and the chain ran. **The training half succeeded; the verdict did
not issue.**

| | |
|---|---|
| training | completed to iteration 500, resume diffs printed, export non-empty — **G0c satisfied** |
| export | `runs/t041-super-lora/export/` — a **merged full model**, 121 GB / 27 shards. Not an adapter |
| eval `95` | ran; 60 clips, both arms, blinded sheet built |
| **verdict** | **VOID on G0b** — the VLM judge did not reach 20/20 on the calibration set |
| spend | ~59 of §7's 122 GPU-h, incl. the apple run below |

**G0b failing is the pre-registered path, not an accident.** PR-09 anticipated it in writing:
*"If G0b fails, that is not a fallback, it is the required path."* `scoring_sheet.jsonl` + `items/`
are a human-rescoreable artifact, and `--verdict` applies the identical rule to a person's
`scores.jsonl`. So the open decision is narrower than it looks: **a human scoring the same 80
blinded clips needs no amendment**; repairing the VLM judge and re-running it does, because that
would be a rule change made after seeing the rule fail. PR-09 §6's VOID row stands until one or
the other happens — and §6 forbids reading a VOID as a weaker pass.

> **Citation correction, 2026-08-15.** The quoted sentence is **§9**'s final bullet, not §6's, and
> the three-step `build-sheet`/`judge`/`verdict` split that makes it possible is registered in **§8
> item 4**. §6 says only what G0b *is* and that any G0 failure is VOID. This file and
> `docs/handoff.md` both attributed it to §6. The claim is unaffected; the pointer was wrong.

### Why the judge failed — diagnosed 2026-08-15, from recorded output only

Forensics over `scores.jsonl`, `key.json`, the vLLM server log and the corpus captions. **No frame
was viewed**, and the diagnosis did not need one.

**The judge answered the literal string `"NO"` to all 80 items.** Not one `YES`, not one
unparseable reply, not one abstention (`unscored_items: 0`). It was a constant classifier.

That single fact reframes the whole run:

| recorded | what it actually means |
|---|---|
| `calibration_correct: 10/20` | the ten negatives are "right" only because a constant NO scores 10/10 on a NO-labelled set. The instrument has zero discriminative output |
| `base_failures: 30` → `G0a_defect_present: true` | **vacuous.** 30/30 because everything scored NO. G0a did not pass, it failed to be tested |
| `b = c = 0`, `mcnemar_p_one_sided: 1.0` | both arms received the same constant. The test compared nothing |

**G0b is the only gate in this run that measured anything**, and what it measured is the scorer.
That is exactly the job PR-09 §6 gives it — *"a rubric that cannot separate a real Dex3 from a
recorded failure cannot adjudicate a generated one"* — so the pre-registration worked as designed.

**It is a defect of judgement, not of plumbing**, and the mechanical explanations were excluded
positively rather than assumed away: every raw reply is one clean token that `parse_answer` reads
correctly; the server log has exactly 80 × `200 OK` and no exceptions; per-request prompt tokens
(~1 130–1 540 against ~155 for the rubric alone) show a ~32-frame video actually reached the model,
with `MM cache hit rate: 0.0%` proving 80 *distinct* media items; all 80 symlinks dereference to
343 KB–8 MB of real H.264; and `build_sheet`'s shuffle was reproduced byte-for-byte from
`random.Random(0)`, so item→label pairing is sound. No off-by-one, no dangling path, no swallowed
API error.

**The mechanism is the rubric meeting this model.** The rubric (`eval_t041_embodiment.py:45-57`)
says *"Answer NO if the end effector is a two-jaw parallel gripper, a suction cup, a **five-fingered
hand**, an industrial claw, or if no end effector is visible. Answer NO if you are unsure."* The
judge is `Qwen/Qwen3-VL-8B-Instruct-FP8` — **the same model job `93` used as the corpus captioner**.
Its own free-form captions of the same real Dex3 footage, at 640×480 rather than the judge's
320×256 and with no yes/no framing, are the evidence: 0 of 30 captions say "three-finger", "Dex3",
"Unitree", "G1" or "thumb"; on the only two occasions it counted fingers at all it counted **five**
(*"Each hand has five fingers with black finger tips"*). So the model reads a Dex3 as a five-fingered
hand, which the rubric instructs it to answer NO to, and where it is unsure the rubric instructs NO
again. **Constant NO is the predicted output of this instrument on this footage.** A rubric tweak at
320×256 is therefore unlikely to be sufficient on its own.

**This was never validated.** The model had only ever been used as a captioner; nothing in the repo
had tested it as a discriminator, and no pre-flight or test catches a constant-label judge before
3.5 GPU-h are spent. That gap is being closed as detection only — it must not, and does not, change
this run's recorded VOID.

### Two divergences from PR-09 as written, found during the same forensics

Neither caused the failure. Both are recorded because whichever path is chosen re-uses these same
20 calibration items and inherits them.

1. **The negatives are not the ones §6 registered.** §6 says *"ten `embodiment_grid.png`
   negatives"*. What was built is **ten real BridgeData2 WidowX clips**
   (`nvidia/BridgeData2-Subset-Synthetic-Captions` @ `e841d88d`), per
   `configs/cosmos3/t041_eval_selection.toml` `[calibration]`. The words "Bridge" and "WidowX"
   appear nowhere in PR-09. The substitution is argued in the config and in `95_*.sbatch:20-25`
   (real footage on both sides, so nobody adjudicates a generated frame) and it makes the negative
   side *easier* than registered — but **it was never recorded as an amendment**, so the gate that
   failed is not literally the gate that was registered. Rebuilding the negatives to match the text
   is **not** a free correction: it would change the calibration set after seeing it fail.
2. **The two sides are not normalised in time.** Positives are 397 frames at 30 fps, negatives 66
   frames at 5 fps — both 13.2 s, both 320×256. Geometry was normalised; frame rate and count were
   not, leaving a latent cue in the instrument's own test set.

### What each path costs — no path is GPU-bound

≈63 of §7's 122 GPU-h remain, and every clip already exists (`eval/items/`, 80/80 symlinks intact).
The binding costs are pre-registration and a person's attention, not compute.

| path | GPU-h | pre-registration cost |
|---|---|---|
| **A** a human scores the same 80 blinded items, re-run `--verdict` | **0** | **none** — registered at §9 and §8 item 4 |
| **B** repair the judge and re-run only the judge step | ~0.2 (1 GPU) | **`T041_RULE_V2` alongside V1**, V1's VOID left standing |
| **C** record the VOID as the finding and stop | 0 | none — §6's VOID row already licenses it |
| **D** regenerate or retrain | — | **not available.** §6: *"there is no second run under this rule"*; §7: *"There is no attempt 3."* |

Three things path A's "a person's time" understates, and they are the reason this is the user's
call and not an obvious default:

- **The human must themselves clear 20/20** on the same calibration items — `compute_verdict` is
  identical for a person's `scores.jsonl`. A human who misses one positive re-issues VOID.
- **It is ~17.6 min of video** (80 × 13.2 s), realistically 1.5–3 h of careful scoring.
- **It permanently ends the "nobody has looked" discipline.** That is inherent to the registered
  path, not an objection to it, but it is not recoverable and should not be started casually.

On path B, the earlier framing here — "repairing the judge needs an amendment" — is right in
substance but understates the form. `docs/handoff.md` §3's standing rule is *"rules are versioned,
never edited in place"*, with `T30_RULE_V2` and PR-05's G2 as precedent. The registered remedy is a
**`T041_RULE_V2` recorded beside V1 with V1's VOID left visible**, not an edit to V1 and not a
re-run under a patched V1. And a rubric chosen because it passes G0b has been selected for passing
G0b — PR-09 §2's named failure mode, which no amendment erases.

### The export is not portable, and that was not anticipated

Checked 2026-08-14 against the question "can we run this anywhere but Discoverer+". The answer is
no, and the reason is the export shape rather than speed:

- It is a **merged full model at 121 GB**, so there is no ~45 MB adapter to move. (The DCP
  `iter_000000500/optim` at 177 MB implies ~22 M trainable params ≈ 45 MB bf16 — an *estimate* from
  optimiser-state size, not verified against the checkpoint keys. `scripts/export_lora.py` exists;
  whether it can recover a standalone adapter from this tree is untested.)
- Against the workstation's 32 GB 5090: bf16 is 4× over, FP8 ~2× over, INT4 ~31 GB leaves nothing
  for activations, the VAE or the vision tower — and 4-bit on a diffusion transformer destroys the
  fine spatial detail (fingers) this experiment exists to measure. 93 GB host RAM is under the
  model, so CPU offload does not close it either.
- ZeroGPU is 48 GB and cannot cold-start a 121 GB pull inside the GPU window. HF Jobs *would* fit
  (`rtx-pro-6000x2` 192 GB at \$5.50/h, ~\$15/run) and is **not recommended**: ~4 879 GPU-h remain
  on Discoverer+ at zero marginal cost, and the checkpoint already sits on cluster-local storage.

### The apple variation run — a demo, and deliberately not evidence

`96_generate_apple_variations.sbatch` + `scripts/make_apple_variation_prompts.py`, job **187623**,
**00:12:41** wall on 8×H200 (~1.7 GPU-h). 15 prompts × 2 arms = 30 clips, zero failures.

Sampler settings are byte-identical to `95`'s so the clips are directly comparable; the one
deliberate difference is a **per-prompt seed** instead of a global one, or the seed family would
collapse to four identical clips. Four families: `real-heldout` (1 — pickapple has 197 train /
**1** val), `real-train` (4, the reproduction floor, labelled so they cannot be read as held-out),
`variation` (6 one-factor authored edits to a real caption), `seed` (4 seeds on one caption).

**Isolated from PR-09 structurally**, not by intention: it writes to `runs/t041-apple-variations/`,
never opens `runs/t041-super-lora/eval/`, runs no judge and no gate, and its `index.json` records
`blinded: false, scored: false`. These 30 clips are the same *kind* of object as the eval's 60, and
what makes the eval's set evidence is that its prompts, settings and rule were fixed before
generation. These were chosen afterwards by someone who had seen the eval. Mixing them in would
cost the other set its only claim to being evidence.

## Notes

**The action question, answered against the card rather than against this file (2026-08-15).**
§"What G1 data it would need" opens "Video, not actions", and §"Two things the recipe demands"
records that Super's action-input list contains no humanoid. Both stand. But "no action port" —
used loosely in review — is wrong: Super ships `action_gen=True`, and the family ships inverse
dynamics, which *is* a video-to-action labeller. What Super lacks is our **vocabulary** and any
published post-training recipe at its scale, not the machinery. The follow-up is **T-042**; the
standing explanation is `docs/action-labels.md` §3b. Nothing here changes PR-09 §9's bound: this
run was a video fine-tune and cannot be read as anything else.

**Cluster access is down as of 2026-08-15** — the key is offered and rejected server-side
(`docs/discoverer.md` §1). Nothing in the open decision above can be executed until it returns.

%% mc-links: [[T-39]] [[T-040]] [[T-37]] %%
