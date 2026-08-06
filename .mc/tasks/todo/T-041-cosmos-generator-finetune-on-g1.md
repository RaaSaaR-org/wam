---
id: T-041
aliases:
- T-041
- T-41
title: Fine-tune a Cosmos generator on G1 data — frozen by PR-07, scoped here
slug: cosmos-generator-finetune-on-g1
status: backlog
priority: 4
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
depends_on:
- "[[T-39]]"
- "[[T-040]]"
due_date: ''
created: 2026-08-06
updated: 2026-08-06
status_note: "Frozen, not merely blocked. PR-07 §7 names Cosmos3-Super generation in its freeze
  list; this task exists so the idea is written down with its costs rather than picked up
  informally. It is scoped, not scheduled — nothing here is actionable while T-39 is unreported."
---

# Fine-tune a Cosmos generator on G1 data — frozen by PR-07, scoped here

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

## Why it is frozen

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
weight is trained on this corpus. That is an **unresolved dataset-licence blocker owned by this
task and by [[T-040]]** — it stands on its own evidence (the table above) and is not the
consequence of any decision recorded elsewhere in this repo.

**Gap, flagged not filed:** no row in the open-decisions table covers *dataset* licensing —
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

- **Structured-JSON captions.** The loader consumes `caption_json`, dense multi-sentence prose
  covering subject, scene, lighting, camera and motion (see `docs/dataset_jsonl.md` — the
  BridgeData2 examples run to ~100 words, measured max ~1790 tokens). Humanoid Everyday ships a
  **task string** (247 of them) and AppleToPlate ships **one** (`"move the apple to the plate"`).
  Captioning ~18 h of video is its own pipeline — plausibly Cosmos-Reason2 — and an uncaptioned
  corpus cannot enter this recipe at all. Cost this before anything else.
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

- [x] Which model, from a **primary source** — `nvidia/Cosmos3-Super`, 64B, above. Pin the
      revision when the task is actually taken up.
- [x] Whether the licence permits fine-tuning and downstream use — OpenMDW-1.1, above. Still read
      the licence text before training; "permissive" is not a reading.
- [ ] A GPU-h estimate for **training** against the 5 000 h allocation, from a measured step time.
      500 iters is the recipe default; the step time on our clip length is unmeasured. The
      *generation* side is estimated above from a vendor number, which is not the same as measured.
- [x] VRAM feasibility per H200 (141 GB) — the recipe is tested on 8×H100 80 GB, so the headroom
      is real. Confirm on the actual node before claiming it.
- [ ] **The captioning pipeline**, which is now the largest unpriced item. See above.
- [ ] Which of the two goals above it is pursuing — **embodiment fidelity** or **visual variety**.
      They need different corpora, and pooling HE + AppleToPlate does not merge them: a rank-16
      LoRA on 18 h learns *HE's* rooms plus *one* NVIDIA room, not arbitrary rooms.
- [ ] Its own pre-registration. This task does not inherit T-040's.

## Notes

Scoped deliberately as a written-down idea with its costs attached, so it is not picked up
informally as "while we are at it". Frozen means frozen.

%% mc-links: [[T-39]] [[T-040]] [[T-37]] %%
