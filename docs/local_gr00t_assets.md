# GR00T assets already on this box — inventory

**Measured 2026-08-16** on the workstation (`RTX 5090, 32 607 MiB`, 24 cores, 93 GB host RAM), by
walking the filesystem. Every path, size and commit below was read off this disk on that date.

> **Second pass, same day — PR-07 §8 items 4, 5 and 6 are now CLOSED, all three without leaving
> this box.** The first pass below found the source document; this one used it. What changed:
> `third_party/isaac-gr00t` vendored at the pinned `1a1837f` and patched, `~/venvs/t39` built from
> upstream's own `uv.lock` and smoke-tested green on the 5090, the `nvidia/GR00T-N1.7-3B` base
> checkpoint downloaded at the exact HF revision the cluster staged, and both entrypoints read off
> the vendor tree. §6 carries the item-by-item state. **The one thing still unmeasured is whether
> 32 GB holds the recipe** — see §6, and note that the probe that would answer it was blocked for
> permission, not skipped.

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

## 1. Source trees — three now, and one of them is the pin

| path | HEAD | date | is the pin? |
|---|---|---|---|
| **`develop/wam/third_party/isaac-gr00t`** | **`1a1837f` *GR00T N1.7 General Release*** | **2026-07-06** | **YES — vendored 2026-08-16** |
| `/home/humanoid/Isaac-GR00T` | `4b1dca9` *Readme Updates and Total Task Correction* | 2026-04-21 | no — *older* than the pin, not an ancestor |
| `/home/humanoid/IsaacLab-Arena/submodules/Isaac-GR00T` | `e29d8fc` *Update README.md (#531)* | — | no |

All three are `https://github.com/NVIDIA/Isaac-GR00T.git`. The first pass found only the latter two
and recorded that *"the pinned commit is fetched but checked out nowhere"* — true when written, and
now fixed rather than merely noted. The vendored tree is gitignored (`third_party/*`, with
`!third_party/patches/`): it is a build artifact, reproducible with

```bash
GR00T_COMMIT=1a1837f20538b7d7e21f977a11a5aee14f99803c bash scripts/build_t39_env_local.sh
```

the local counterpart of `cluster/discoverer/72_build_t39_env.sbatch`, whose three deltas from it
are stated in its header. `third_party/isaac-gr00t/PROVENANCE.json` pins what was actually built:
commit, patch + `patch_sha256`, `uv.lock` and `pyproject.toml` hashes, and the `pip freeze` hash of
the venv (139 packages).

**The registered patch applies cleanly at this base and is inert here.** Its own header said
*"applies cleanly and compiles locally; the venv build and smoke test have not been run against
it"* — both have now been run. The smoke test reports the decoder as
`torchcodec.decoders._video_decoder.VideoDecoder`, and that decoder reads real corpus video
(`episode_000000.mp4` → `(3, 480, 640) uint8`). **This contradicts the prediction** drawn from
upstream's own `pyproject.toml:38` (*"torchcodec 0.8.0 … does NOT support FFmpeg 8"*, and this box
runs FFmpeg 8.1.2): torchcodec does not link the system FFmpeg here. The shim is present and never
entered, so the executed path is upstream's. It is kept applied so the local and cluster trees stay
byte-identical — on Discoverer+ the fallback **is** load-bearing.

### Both entrypoints, item 6 — read from the pinned tree

| | value | primary source |
|---|---|---|
| `TRAINER_ENTRYPOINT` | `gr00t/experiment/launch_finetune.py` | `examples/finetune.sh` builds `LAUNCH_CMD` around it and `exec`s it |
| `POLICY_ENTRYPOINT` | `gr00t.policy.Gr00tPolicy` | `gr00t/policy/__init__.py` `__all__`; `gr00t_policy.py:83` `__init__(embodiment_tag, model_path, *, device, strict)`; `get_action()` on `BasePolicy` (`policy.py:80`) over the abstract `_get_action` |

`examples/finetune.sh` is worth reading whole — it is upstream's own single-GPU path, and it
settles two things this project had open (§6, questions 2 and 3).

> **A wrong string was already in our tree, and this is exactly what the house rule is for.**
> `scripts/train_t39_baseline.py:5` gives the usage example
> `--trainer-entrypoint scripts/gr00t_finetune.py`. **No such file exists at `1a1837f`** — the tree
> ships `gr00t/experiment/launch_finetune.py`. It is an example rather than a default, so nothing
> could have run under it silently, but it is a plausible-looking path in prose that would not have
> worked, which is the failure mode PR-07 §8 item 6 was written against. **Left unfixed on purpose:**
> that file has uncommitted changes from concurrent work, and a drive-by edit to someone else's
> in-flight file trades one small wrong string for a merge conflict in the T-39 driver.

---

## 2. Python environments

| venv | python | `gr00t` | torch | `flash_attn` | `deepspeed` |
|---|---|---|---|---|---|
| **`/home/humanoid/venvs/t39`** | **3.12.13** | **editable → `third_party/isaac-gr00t` @ `1a1837f`** | **2.9.0+cu128** | **2.8.3 ✓ runs** | **✓** |
| `/home/humanoid/venvs/arena` | — | 0.1.0 (editable → `IsaacLab-Arena/submodules/Isaac-GR00T` @ `e29d8fc`) | 2.11.0+cu130 | absent | — |
| `/home/humanoid/develop/wam/.venv` (WAM) | — | — | 2.13.0+cu130 | absent | — |

**`venvs/t39` closes item 4.** It is upstream's own environment — `uv sync --frozen` against the
`uv.lock` at the pinned commit, so the dependency set is the lock's and not a re-resolution wearing
its name — plus `av`, installed separately and named in `PROVENANCE.json` so the fallback's
dependency is visible as ours. Smoke test, `2026-08-16`:

```json
{"python": "3.12.13", "torch": "2.9.0+cu128", "cuda_available": true,
 "arch_list": ["sm_70","sm_75","sm_80","sm_86","sm_90","sm_100","sm_120"],
 "device": "NVIDIA GeForce RTX 5090",
 "flash_attn": true, "deepspeed": true, "av": true, "gr00t_import": true,
 "video_decoder": "torchcodec.decoders._video_decoder.VideoDecoder"}
```

Two of the first pass's caveats are now answered by measurement rather than by argument:

- **`flash-attn` runs on `sm_120`.** Not merely importable — `flash_attn_func` on a bf16 tensor
  returns on the 5090. The lock installs the official `flash_attn-2.8.3+cu12torch2.9…cp312` wheel,
  nothing was compiled, and the SDPA-fallback question (which *would* have been a recipe change,
  `groot_recipeA_ckpt5000/experiment_cfg/conf.yaml:14` sets `use_flash_attention: true`) does not
  arise.
- **`venvs/arena` must not be used for T-39.** It is an editable install pointing at `e29d8fc`, so
  using it silently selects the wrong recipe. It stays what it was: the Arena environment.

**The modality config now verifies inside upstream's registry at the pin**, which had never been
possible before — `configs/groot/verify_new_embodiment_config.py` under `venvs/t39` reports
`PASS`: identical to the recorded recipe on keys, `delta_indices` and `action_configs`; state slices
tile 0..43 against the dataset's `info.json`; every config key defined by `meta/modality.json`.

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

## 4. The base checkpoint — item 5, now closed on both halves

Item 5 had two halves and the first pass conflated them: **the id had to come from a primary
source, and the weights had to exist.** Both are now done.

### The id, from the vendor tree rather than from our own sbatch

`nvidia/GR00T-N1.7-3B` appears **46 times** at `1a1837f`, and not incidentally — upstream names it
*as the base* in code that would be wrong if it were any other string:

- `gr00t/data/embodiment_tags.py:204` — *"Tags baked into the base model (`nvidia/GR00T-N1.7-3B`) —
  usable without finetuning."*
- `gr00t/model/modules/qwen3_backbone.py:36` — *"Every GR00T checkpoint (including the base
  `nvidia/GR00T-N1.7-3B`)…"*
- `tests/gr00t/experiment/test_experiment_run.py:59` and
  `tests/gr00t/policy/test_gr00t_policy_gpu.py:49` — `MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"`
- `tests/scripts/deployment/test_standalone_inference.py:120` — `DEVICE_BASE_MODEL_REPO`

The four sibling ids (`-DROID`, `-LIBERO`, `-SimplerEnv-Fractal`, `-SimplerEnv-Bridge`) are the
*post-trained* variants, which is what makes the base id unambiguous rather than one of a set.
`cluster/discoverer/73_dryrun_t39_subset.sbatch:74` had the same string as a default — it was
right, and it is now no longer the only witness for it.

### The weights, fetched at a pinned revision

```
nvidia/GR00T-N1.7-3B @ 2fc962b973bccdd5d8ce4f67cc63b264d6886495   6.93 GB, 27 files
  model-00001-of-00002.safetensors   4 990 519 232 B
  model-00002-of-00002.safetensors   1 919 980 184 B
  → ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b9…
```

**That revision is not a choice made today.** `72_build_t39_env.sbatch`'s header records that the
July run's *"staged weights were pulled at HuggingFace commit `2fc962b973bccdd5d8ce4f67cc63b264d6886495`
on 2026-07-06"*, and the download was pinned to it rather than to `main`, so the local copy is the
same artifact the cluster staged. AC-04 gets a revision, not a moving tag.

`config.json` confirms it is the right object: `model_type: Gr00tN1d7`, backbone
`nvidia/Cosmos-Reason2-2B`, `max_state_dim 132`, `max_action_dim 132`, `action_horizon 40` —
**identical to the recipeA/B configs in §3**, which is the direct evidence that those two are
post-trains of this base and therefore unusable as a starting point for a positive control.

The first pass's argument stands and is now the reason the download mattered: T-39 runs NVIDIA's
recipe from NVIDIA's released base on our split, and starting from `groot_recipeA/B` would be a
warm start on a model already trained on this corpus by a different recipe.

### One thing the base already carries, worth knowing before E-02 is re-read

`embodiment_id.json` in the base checkpoint registers **`unitree_g1_full_body_with_waist_height_nav_cmd`
→ 25**, alongside `real_g1_relative_eef_absolute_joints` → 25 and `agibot` → 26. So G1 tags exist in
the base's table. That does **not** make them usable here, and
`70_train_t39_baseline.sbatch:113-120` already gives the reason: upstream files `UNITREE_G1` under
*"pre-registered POSTTRAIN tags (require finetuned checkpoint)"*, so on the **base** checkpoint that
tag names a head carrying no trained weights. `new_embodiment` stays correct.

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

## 6. Where items 4–6 stand: all three closed

| item | first pass | now |
|---|---|---|
| 4 — separate trainer venv | "substantially met by `venvs/arena`" — **wrong env** | **CLOSED.** `~/venvs/t39`, built from the pin's own `uv.lock`, smoke-tested green, `PROVENANCE.json` written |
| 5 — `MODEL_ID` from a primary source | open, and the weights absent | **CLOSED.** `nvidia/GR00T-N1.7-3B` read off the vendor tree in four places; 6.93 GB fetched at revision `2fc962b9` |
| 6 — `TRAINER_ENTRYPOINT` / `POLICY_ENTRYPOINT` | trainer half only | **CLOSED.** `gr00t/experiment/launch_finetune.py` and `gr00t.policy.Gr00tPolicy` |

**PR-07 §8's stated blockers are gone.** The first pass said the blocker was never cluster access;
this pass shows it was never anything that needed a decision either — it was reading, a vendored
clone and a download.

### The three questions from the first pass — two answered, one open

1. **Recipe fidelity on one GPU — ANSWERED, and the premise was wrong.** The concern was that
   `global_batch_size: 128` came from 8 GPUs and would need gradient accumulation to reproduce.
   That figure is **recipeA's**, a Discoverer+ H200 artifact — it is not T-39's recipe.
   `70_train_t39_baseline.sbatch:144` pre-registers `GLOBAL_BATCH_SIZE=32` and passes
   `--num_gpus 1`, matching upstream's own default in `examples/finetune.sh`
   (`GLOBAL_BATCH_SIZE:-32`). **T-39 was always a single-GPU recipe at batch 32.** Upstream even
   special-cases the single-GPU path — *"restrict to a single GPU so HF Trainer doesn't wrap the
   model in DataParallel"* — and `gr00t/experiment/experiment.py:266` sets
   `per_device_train_batch_size = global_batch_size // num_gpus`, so one card at 32 is the
   unmodified recipe rather than a local approximation of it. Were accumulation ever wanted, it is
   upstream's own knob (`training_config.py:143`, `accumulated_batch_size`), not a patch.
2. **Venue — ANSWERED as a recording requirement, not a threshold.** `T39_RULE_V1` does not move if
   this runs here. `PROVENANCE.json` carries `"venue": "workstation"` so a run built from this tree
   cannot silently inherit the sbatch's claim, per §8's own precedent of recording the
   `--wam-dataset` correction rather than quietly patching it.
3. **32 GB vs the quote — STILL OPEN, and now the only thing between here and a local run.**
   `cluster/discoverer/70_train_t39_baseline.sbatch:45` records NVIDIA's quote as *"2–4 h on one
   40 GB GPU"*; this card is 32 607 MiB, **8 GB short**. Everything else now argues it fits — the
   backbone and vision tower are frozen (`tune_llm: false`, `tune_visual: false`), only the
   projector, diffusion head and vlln train, flash-attn is available rather than SDPA, and the
   pre-registered batch is 32 rather than 128. **None of that is a measurement. [?]**

   The measurement is one short run of `launch_finetune.py` — a dozen steps, `--save_steps`
   effectively disabled, peak VRAM and step time recorded and nothing else. **It was written and
   not run: the permission classifier declined it, twice.** It is
   `vram_probe.sh` in this session's scratchpad and needs an explicit go-ahead. Recorded as
   *blocked*, not as *skipped* — the difference matters, because a page that quietly omitted it
   would read as though 32 GB had been checked.

### What is still genuinely missing for a local T-39

Not an item 4–6 blocker, but the next thing that bites: **the converted WAM corpus is not on this
disk.** `70_train_t39_baseline.sbatch` passes two dataset paths — `--dataset` (the raw LeRobot
source the trainer eats, present at `~/wam-t041/raw/GR00T-N1.7-AppleToPlate`) and `--wam-dataset`
(the converted WAM episodes the snapshot hash is taken over). Only the first exists here; there is
no local `gr00t-apple-full`. Without the second, `run_metadata.json` cannot be written in the shape
`eval_t16.verify_split` consumes, and `71_eval_t39_control.sbatch` will refuse to score the result —
correctly. Conversion is explicitly allowed ahead of T-39 and is the obvious next task.

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
