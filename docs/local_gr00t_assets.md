# GR00T assets already on this box — inventory

**Measured 2026-08-16** on the workstation (`RTX 5090, 32 607 MiB`, 24 cores, 93 GB host RAM), by
walking the filesystem. Every path, size and commit below was read off this disk on that date.

**Why this page exists.** `PR-07 §8` items 4–6 have blocked T-39 since 2026-08-06, and all three
were written as *"needs SSH or a source document"*:

> 4. A separate cluster venv for the vendored trainer.
> 5. `MODEL_ID` — the exact checkpoint id, **not verified from a primary source**.
> 6. `TRAINER_ENTRYPOINT` / `POLICY_ENTRYPOINT` — *"we do not know the vendored trainer's entrypoint
>    path or its inference API from a primary source, and a plausible guess would run something
>    adjacent and record it as NVIDIA's recipe."*

**The source document is on this machine and has been since before the gate was written.** Nobody
looked. What follows is what is here, what is not, and precisely which of items 4–6 each one closes.

---

## 1. Source trees — two checkouts, neither at the pinned commit

| path | HEAD | date | `1a1837f` an ancestor? |
|---|---|---|---|
| `/home/humanoid/Isaac-GR00T` | `4b1dca9` *Readme Updates and Total Task Correction* | 2026-04-21 | **NO** |
| `/home/humanoid/IsaacLab-Arena/submodules/Isaac-GR00T` | `e29d8fc` *Update README.md (#531)* | — | **NO** |

Both are `https://github.com/NVIDIA/Isaac-GR00T.git`. `1a1837f` (*GR00T N1.7 General Release*,
2026-07-06) is the commit this project pinned — it names our own patch,
`third_party/patches/isaac-gr00t-pyav-fallback-1a1837f.patch`.

> **The pinned commit is fetched but checked out nowhere.** `/home/humanoid/Isaac-GR00T` is in
> detached HEAD at `4b1dca9`, which is *older* than the pin and not an ancestor of it. Anything run
> against either tree as it stands today is **not** running the recipe this project pinned. This is
> a one-line `git checkout`, but it is currently wrong, and it is exactly the class of error item 6
> exists to prevent.

### The entrypoint, item 6 — read from the pinned tree

```
$ git ls-tree -r --name-only 1a1837f | grep -i 'finetune\|train'
examples/finetune.sh
gr00t/configs/finetune_config.py
gr00t/configs/training/training_config.py
gr00t/experiment/launch_finetune.py      <-- TRAINER_ENTRYPOINT
gr00t/experiment/launch_train.py
gr00t/experiment/trainer.py
```

`launch_finetune.py` and `launch_train.py` both exist in the working tree too. **Item 6's trainer
half is answerable from a primary source on this disk.** `POLICY_ENTRYPOINT` (the inference API the
eval adapter needs) has *not* been read yet — `scripts/eval/` and `scripts/deployment/` at `1a1837f`
are where to look, and that is unfinished work, not a resolved item.

---

## 2. Python environments

| venv | `gr00t` | torch | transformers | `flash_attn` |
|---|---|---|---|---|
| `/home/humanoid/venvs/arena` | **0.1.0** (editable → `IsaacLab-Arena/submodules/Isaac-GR00T`) | **2.11.0+cu130** | 4.57.6 | **absent** |
| `/home/humanoid/venvs/onnx` | — | — | — | — |
| `/home/humanoid/develop/wam/.venv` (WAM) | — | 2.13.0+cu130 | — | absent |

```
torch.cuda.get_arch_list() -> ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
sm_120 supported: True        device: NVIDIA GeForce RTX 5090
```

**`venvs/arena` is a working GR00T environment that already sees the 5090.** It substantially
answers **item 4** — the requirement was a venv *separate from the WAM venv* so the vendored
trainer's torch/flash-attn pins stay out of anything that touches WAM numbers, and this is one.

Two caveats before it is used as the t39 venv:

- It is an **editable install pointing at the Arena submodule at `e29d8fc`**, not the pin. Using it
  as-is silently selects the wrong recipe — see §1.
- **`flash_attn` is not installed anywhere on this box**, and the recipe configs set
  `use_flash_attention: true` (`groot_recipeA_ckpt5000/experiment_cfg/conf.yaml:14`). Falling back
  to SDPA is a recipe change; on `sm_120` Blackwell, building flash-attn is its own problem. Neither
  option is free and the choice must be recorded, not defaulted into.

---

## 3. Checkpoints

### Trainable PyTorch `Gr00tN1d7` checkpoints — three, all post-trained

| path | size | steps | note |
|---|---|---|---|
| `models/finetunes/groot_recipeA_ckpt5000` | 6 917 749 360 B (6.5 G) | 5 000 | lr 1e-4 |
| `models/finetunes/groot_recipeB_ckpt10001` | 6 917 818 878 B (6.5 G) | 10 000 | lr 5e-5 |
| `models/isaaclab_arena/static_apple_tutorial/gn1x_tuned_static_apple` | 12 584 186 298 B (12 G) | — | 3 shards |

**None of these was trained on this box.** Both `finetunes/` configs point at
`/valhalla/projects/ehpc-aif-2026pg01-905/apple_pnp_h200/dataset` with `num_gpus: 8`,
`global_batch_size: 128` — Discoverer+ H200 artifacts, downloaded here. There is **no precedent of
local GR00T training on this machine.**

recipeA's `trainer_state.json`: loss 1.2102 → 0.0254 over 5 000 steps, `epoch 1.0`.

Recipe shape, from `groot_recipeA_ckpt5000/experiment_cfg/conf.yaml`:

```yaml
model_name: nvidia/Cosmos-Reason2-2B    # the VLM backbone
tune_llm: false                          # backbone frozen
tune_visual: false                       # vision tower frozen
tune_projector: true
tune_diffusion_model: true
max_state_dim: 132   max_action_dim: 132   action_horizon: 40
```

Only the projector, diffusion head and vlln train — which is why 32 GB is plausible at all.

### Not trainable

- `models/GR00T-N1.7-ApplePnP-V1` — **12 G, ONNX export only** (`backbone.onnx.data` 5.7 G,
  `action_head.onnx.data` 6.1 G). Inference/inspection evidence, per
  `docs/contracts/vla-training-consumer.md:15`. Cannot be fine-tuned.

### The backbone, cached

- `~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B` — **4.6 G**, full `model.safetensors`.
  This is the VLM half of GR00T N1.7.

---

## 4. What is NOT here — and it is the thing T-39 needs

**`nvidia/GR00T-N1.7-3B`, the base checkpoint, is absent.** Searched by name and by
`config.json` declaring `Gr00tN1d7`; the only three hits are the post-trained checkpoints in §3.

This matters because T-39 is a **positive control**: it runs NVIDIA's recipe from NVIDIA's released
base on our split. Starting from `groot_recipeA/B` instead would start from a model already trained
on this corpus by a different recipe — which is not a positive control, it is a warm start, and a
result from it could not answer the question T-39 exists to ask.

`cluster/discoverer/73_dryrun_t39_subset.sbatch:74` already names the id:
`MODEL_ID=${MODEL_ID:-nvidia/GR00T-N1.7-3B}` — but **a default in our own sbatch is not the primary
source item 5 demands.** The id must be confirmed against the vendor tree or model card, and then
the weights fetched. Expect ~6.5 GB by analogy with the §3 checkpoints (not measured).

**Nothing has been downloaded.** That needs an explicit go-ahead.

---

## 5. Data

| what | path | size |
|---|---|---|
| `nvidia/GR00T-N1.7-AppleToPlate` (raw LeRobot) | `~/wam-t041/raw/GR00T-N1.7-AppleToPlate` | 966 M |
| same, HF cache | `~/.cache/huggingface/hub/datasets--nvidia--GR00T-N1.7-AppleToPlate` | — |
| 13 × `unitreerobotics/G1_Dex3_*` | `~/.cache/huggingface/hub/datasets--unitreerobotics--G1_Dex3_*` | — |
| working set | `~/wam-t041` | 54 G |

`meta/info.json`: **402 episodes, 171 625 frames, 30 fps, `robot_type: unitree_g1`** — matching
PR-07's corpus exactly. All 13 G1_Dex3 sets behind T-043 are cached locally too.

---

## 6. Where item 4–6 actually stand

| item | before | after this inventory |
|---|---|---|
| 4 — separate trainer venv | open | **substantially met** by `venvs/arena`, once repointed at the pin and the flash-attn question is answered |
| 5 — `MODEL_ID` from a primary source | open | **still open** — the id is guessable from our own sbatch, the weights are absent, and neither is a primary source |
| 6 — `TRAINER_ENTRYPOINT` | open | **trainer half closed**: `gr00t/experiment/launch_finetune.py` @ `1a1837f`. `POLICY_ENTRYPOINT` still open |

**T-39 remains unsubmittable, and this page does not change that.** It changes *why*: the blocker
was never access to the cluster, it was that nobody had read the source sitting in `/home/humanoid`.
Two of three items are now within reach without leaving this box.

### Before anything runs locally, three unanswered questions

1. **32 GB vs the quote.** `cluster/discoverer/70_train_t39_baseline.sbatch:45` records NVIDIA's
   quote as *"2–4 h on one 40 GB GPU"*. This card is 32 607 MiB — **8 GB short**. The frozen
   backbone makes it plausible; nothing has measured it. **[?]**
2. **Recipe fidelity.** `global_batch_size: 128` came from 8 GPUs. Reproducing it on one card needs
   gradient accumulation, not a smaller batch — PR-07's whole design is that the trainer is
   NVIDIA's, unmodified, because *"a positive control run through our reimplementation of someone
   else's recipe is not a positive control."* Batch size and attention kernel are part of the
   recipe. **[?]**
3. **Venue.** PR-07 pre-registers sbatch files and a 12 GPU-h ceiling. The venue is not a threshold
   and `T39_RULE_V1` does not move if this runs locally — but the deviation is recorded here rather
   than assumed, per §8's own precedent of recording the `--wam-dataset` correction instead of
   silently patching it.

---

## 7. Unrelated, found while measuring — headless rendering

`MUJOCO_GL=egl` is **required** on this box. Without it MuJoCo defaults to GLFW, which needs an X
display this machine's shells do not have:

```
GLFWError: (65550) X11: The DISPLAY environment variable is missing
mujoco.FatalError: an OpenGL platform library has not been loaded into this process
```

That is 1 of the 5 full-suite failures on 2026-08-16; `osmesa` errors differently. With
`MUJOCO_GL=egl` the render test passes in 2.55 s and the suite is **4 failed, 1955 passed, 30
skipped in 93.6 s**. The remaining 4 are `tests/test_runtime.py` rollout-CLI tests needing
`runs/d1-overfit-seed0/checkpoint.safetensors`, a training artifact not present on this box —
unrelated to rendering and to GR00T.
