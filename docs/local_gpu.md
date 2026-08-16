# Local GPU runbook — run, test and benchmark a WAM checkpoint

For a single consumer GPU box (written against an **RTX 5090, 32 GB**). Mostly inference and
scoring; §5 adds the one training job that now fits.

**The T-16 fine-tune of record is not in this runbook.** It trained on Discoverer+
(`cluster/discoverer/README.md`), because that is where the resume harness, the 402-episode dataset
and the accounted GPU hours live. This box is primarily for the loop that follows a fine-tune:
**prove the adapter → generate predictions → score them → run the closed loop**. What *has* since
moved here is the remaining I-8 / T-32 data-scaling rungs (§5) — the memory arithmetic that says a
LoRA fine-tune fits in 32 GiB is in `configs/training/joint_wan_gr00t_5090.yaml`'s header.

> **Nothing on this page has ever run on a 5090.** Every measured number below was recorded on an
> H200 or on a ZeroGPU RTX PRO 6000, and is labelled with where. Every number that was *not*
> measured is labelled unmeasured. The first thing to do on the box is §0, which replaces the
> guesses with figures from your own card.

**Units.** Every VRAM figure on this page is **decimal GB** — `max_memory_allocated() / 1e9`, which
is what `scripts/dream.py:634` and `scripts/hf_job_wan_smoke.py:424` write into `runs/`. A "32 GB"
card is 32 **GiB** = **34.36 decimal GB**. Comparing a decimal-GB artifact against a binary-GB card
makes the 5090 look 2.4 GB smaller than it is, which is exactly how an earlier draft of this page
concluded that `dream.py` could not fit. That was a unit error — but correcting it does not make
the run safe either, because *allocated* is not *occupied*. Once the allocator's slack and the CUDA
context are added back, the two archived peaks straddle the board limit. Run `dream.py` with
`--offload-text` (§0c).

---

## Why one GPU is enough

WAM does not generate video at inference. `JointWorldActionModel.predict()`
(`src/wam/training/joint.py:606`) runs **one** backbone pass at the clean end of the flow, reads the
readout blocks, applies `ActionHead`, and throws the video velocity away — no denoising loop, one
pass per control cycle. The video branch's job was to shape the features during *training*.

The sequences are also small. T-16's geometry is 9 frames at 128×160; the VAE strides 16 spatially
and 4 temporally, so latents are `(B, 48, 3, 8, 10)`, and patchifying `[1, 2, 2]` gives
**60 tokens per sample**.

The resident weights are not an estimate. Parameter counts below are read from the **safetensors
headers** of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (per-tensor shapes and dtypes, not file sizes) and
multiplied by the dtype each tower is actually loaded at in `src/wam/backbones/wan_i2v.py:288-296`:

| resident at inference | params | loaded as | GB |
|---|---|---|---|
| Wan DiT (5B), frozen | 4,999,787,712 | bf16 (`config.dtype`) | 10.00 |
| umT5 text tower, frozen | 5,680,910,336 | bf16 (`config.dtype`) | 11.36 |
| VAE (encoder path only — `decode_video` is unused here) | 704,688,668 | **fp32, hard-wired** (`wan_i2v.py:289-291`) | **2.82** |
| LoRA adapters + action branch | 82,519,450 | fp32 (`model.safetensors`, 643 tensors, all F32) | 0.33 |
| activations, 60 tokens, batch 1 | | | negligible |
| **frozen sum** | | | **24.18** |

**Measured: 24.28 GB peak / 25.18 GB reserved** (smoke job `183599` on an H200,
`runs/smoke/183599/wan_smoke_report.json:129-130`; the readout probe independently saw 24.65 GB).
The arithmetic lands **0.42 % under** a measurement it did not use, which is the reason to trust the
weight terms. On a 32 GiB card (34.36 decimal GB) that leaves about **10 GB spare** — an earlier
version of this page said 7 GB, having subtracted a decimal-GB peak from a binary-GB card.

**The VAE is 2.82 GB, not the ~1.4 GB this table used to claim.** 1.41 GB is the bf16 figure; the
Hub stores the VAE F32 and `wan_i2v.py` keeps it F32, with no flag to change that. It is the
third-largest single item in the budget.

An older version put the text tower outside the budget entirely, reasoning that `condition_text` is
cached (`src/wam/backbones/wan_i2v.py:459`) and one task has one instruction, so the tower runs once.
Caching the *output* does not evict the *weights* — 11.36 GB of umT5 stays resident for the whole run
unless something explicitly drops it. The ~12 GB figure was the **offloaded** budget presented as the
default.

**Two levers:**

- **Evict the text tower** — `--offload-text`, now on every entry point (§0b). Expect ~13 GB with it
  (24.28 measured − 11.36 umT5 = 12.92). Unmeasured: the 24.28 GB above was recorded with
  `offload_text: false`, and no run in `runs/` has recorded a peak with it set.
- **Truncate the DiT** — not built. T-16 reads blocks `[2, 10]` of 30
  (`configs/training/joint_wan_gr00t.yaml:63`, the depth the readout probe measured), so **blocks
  11–29 are computed and thrown away — 19 of 30 layers, ~63 % of DiT weight and compute.** Worth
  ~6.3 GB of the DiT's 10.00, so *less* VRAM than the text tower — an earlier version of this page
  said "worth more", which does not survive the corrected weight table. It is the only one of the
  two that buys **latency**, which is what §4 needs. (The smoke script's `[15, 22]` is the backbone
  default, not what T-16 trains.)

> The weight terms are measured; everything derived from them is arithmetic. §0 and §1 measure it on
> your card. Trust the measurement.

---

## 0. Prerequisites — `scripts/preflight_gpu.py` is the gate

Do not run a GPU command on a fresh box before this exits 0. It is stdlib-only, takes seconds, and
checks the seven things that actually go wrong here: interpreter, torch build, **a real kernel
launch**, VRAM and host RAM, dependency completeness, asset paths, and the per-entry-point VRAM
budget against *your* card.

```bash
python3 -m venv .venv && . .venv/bin/activate

# 1. torch FIRST, from the CUDA build you chose (see "Which torch" below).
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 2. then the project. `local` is the one bracket that installs what this runbook runs.
pip install -e '.[local]'

# 3. the gate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/preflight_gpu.py
python scripts/preflight_gpu.py --backbone-source /path/to/Wan2.2-TI2V-5B \
    --dataset datasets/gr00t-apple-full \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000
```

Exit code is 0 iff nothing FAILed; `--json preflight.json` writes the same report machine-readably,
and `--deep-import` actually imports each dependency instead of probing for it (catches the classic
`cv2` installed without `libGL.so.1`).

**Why not the old one-liner.** This page used to say
`python -c "import torch; print(torch.cuda.get_device_capability())"` and expect `(12, 0)`. That
check cannot catch the failure it is aimed at: the capability comes from the *driver*, not from the
compiled cubins, so a wheel with no sm_120 kernels reports `(12, 0)` correctly and then dies at the
first launch. Section 3 of the preflight launches real kernels (bf16 matmul, fp32 matmul, SDPA,
conv2d, layernorm), synchronizes, and checks the outputs are finite. **It is the only check on this
page that can fail for the right reason** — and it is also the only part of the preflight that has
never been executed on sm_120 hardware, because there is no such hardware here.

**`pip install -e '.[dev]'`, which this page said until now, does not work.** `dev` is
`[pytest, ruff, peft]`; `src/wam/data/episode.py` imports `pyarrow` at module level, so every GPU
command below died on import before it ever reached the card. `.[local]` is a flat union of
`wan + data + serve + sim + dev` declared in `pyproject.toml` — the preflight parses that table at
runtime and prints the same bracket, so the two cannot drift apart.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is what every cluster job exports
(`cluster/discoverer/50_train_t16.sbatch:41` and the seven other sbatch files) and what nothing sets
locally. It has to be **exported before the process starts** — PyTorch reads it once, lazily, on the
first CUDA allocation. On a card this close to its ceiling, fragmentation is the difference between
fitting and not. The scripts print a hint when it is unset rather than setting it themselves, on
purpose: `expandable_segments` changes allocator behaviour and therefore the peak-VRAM figures the
scripts record, so a run that set it silently would report numbers not comparable with `runs/`
(`src/wam/runtime/offload.py:advise_alloc_conf`).

**Host RAM, not VRAM, is the failure nobody expects.** `device_map` is reachable only from
`hf_job_wan_smoke.py:70`, so on every other entry point `from_pretrained` materializes each tower in
**host** RAM before it reaches the card: ~24.2 GB of weights plus a shard buffer. **32 GB of system
RAM is the floor.** Below it the load swaps or gets OOM-killed and you never see a CUDA error at
all. The preflight FAILs on this (`host.ram`) — it is the check that fails on the Mac this was
written on, honestly, at 19.3 GB.

Then confirm the repo is healthy before trusting any GPU number from it:

```bash
python -m pytest -q          # the whole suite, all CPU
```

### 0a. Which torch — cu128 or the default index?

**There is a live conflict in this repo, and it will silently undo your install.**

- `docs/local_gpu.md` (this page) says the **cu128** index.
- `uv.lock` resolves **torch 2.13.0 from PyPI**, which on Linux pulls `nvidia-cudnn-cu13 9.20.0.48`
  and `cuda-toolkit 13.0.3.0` — i.e. a **CUDA 13** build. (Verified by reading `uv.lock`: the
  `torch` package entry's `source = { registry = "https://pypi.org/simple" }` and its Linux-marked
  dependencies.)
- The cluster, which produced every measured number in `runs/`, uses neither: conda-forge
  `pytorch=*=cuda12*` under `CONDA_OVERRIDE_CUDA=12.9` (`cluster/discoverer/10_build_env.sbatch:32`).

**Recommendation: use pip and the cu128 index. Do not run `uv sync` in this checkout on the GPU
box.** Two reasons, one strong and one weak:

1. *(strong, mechanical)* `torch` is not in the lockfile's default dependency set — it only appears
   under the `train`/`wan`/`local` extras. `uv sync` is exact by default: it makes the environment
   match the lock, which means a bare `uv sync` **removes** your hand-installed torch, and
   `uv sync --extra local` **replaces** it with the PyPI CUDA-13 wheel. Nothing in `pyproject.toml`
   pins an index for uv, so there is no configuration making it honour cu128. This is uv's
   documented default behaviour, not something measured here.
2. *(weak, and only a preference)* CUDA 12.8 matches the CUDA major the archived numbers were
   produced under. Both toolkits support sm_120; neither has been run on this box. If you prefer
   the CUDA-13 wheel, the argument for it is that it needs no `--index-url` at all — but check
   `nvidia-smi` first, because CUDA 13 wants a newer driver branch than CUDA 12.8 does.

**How not to have it clobbered:**

```bash
# safe: uv's pip interface installs without pruning, and honours the index you give it
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install -e '.[local]'

# NOT safe on this box: reconciles the venv to uv.lock and overwrites/removes torch
uv sync
uv sync --extra local
```

`pip install -e '.[local]'` after torch is present will **not** reinstall it — the requirement is
unpinned, so an installed torch already satisfies it. That ordering is why step 1 above comes first.

### 0b. `--offload-text` — where it is now wired

`WanI2VAdapter.offload("text_encoder")` (`src/wam/backbones/wan_i2v.py:397`) parks the ~11.36 GB umT5
tower on the CPU after the instruction is encoded. Until now `scripts/hf_job_wan_smoke.py` was its
only caller. The shared wiring is `src/wam/runtime/offload.py`, and the flag is now on:

| entry point | `--offload-text` |
|---|---|
| `scripts/hf_job_wan_smoke.py` | yes (was always) |
| `scripts/eval_t16.py` | yes |
| `scripts/dream.py` | yes |
| `scripts/rollout.py` | yes (`--policy joint` only) |
| `scripts/serve_policy.py` | yes (`--joint` only) |
| `scripts/train_t16_lora.py` | yes |

**Off by default**, like `--frame-history`, `--flow-sampler` and `--base-null`, so archived runs stay
bit-reproducible. The preflight reports the count it *reads out of the scripts* rather than trusting
this table (`budget.offload_lever`).

Two things worth knowing before you use it:

- **Order matters.** `WanFlowBackbone._apply` forwards every device move to the towers it holds, so
  an offload issued before the final `.to(device)` is silently undone. Every call site is after
  residency; if you wire a new one, do the same.
- **The cost is one CPU umT5 forward per *distinct instruction*, not per call.** `condition_text`
  memoizes under the prompt string and stores the result on `self.device`, so the cache survives the
  tower moving. `datasets/gr00t-apple-full` and `gr00t-apple-grip` have exactly **one** distinct
  instruction across all 402 episodes, so training pays it once in 20 000 steps.
  `train_t16_lora.py` counts the distinct instructions from the manifests at startup and logs a loud
  `WARNING` above one, so a multi-instruction corpus cannot hit that cliff silently.

### 0c. The VRAM budget, per entry point

Decimal GB, against a 32 GiB card = 34.36 decimal GB. `scripts/preflight_gpu.py` prints this table
computed against *your* card's measured total, with the provenance of every row.

| entry point | peak | headroom | verdict | measured? |
|---|---|---|---|---|
| `scripts/dream.py` | 32.54 | +1.82 → **−0.05 occupied** | **NO MARGIN — use `--offload-text`** | yes — ZeroGPU RTX PRO 6000, `runs/dream/t36-zerogpu-motion-seed0/dream.json:265` (and 32.47 at `t35-zerogpu-seed0/dream.json:172`) |
| diffusers `WanImageToVideoPipeline` | 31.55 | +2.81 | FITS | yes — `runs/presentation/t16_lora_futures/scale0.report.json:66` (31.39 without LoRA) |
| `scripts/hf_job_wan_smoke.py` | 24.28 | +10.08 | FITS | yes — H200, `runs/smoke/183599/wan_smoke_report.json:129` |
| `eval_t16.py` / `rollout.py` / `serve_policy.py` | ~24.28 | +10.08 | FITS (est) | **no** — inferred from the smoke figure at a smaller geometry; no eval artifact in `runs/` carries a `peak_vram_gb` field |
| `train_t16_lora.py`, batch 2 (§5) | ~27.7 | +6.66 | FITS (est) | **no** — measured weights (24.18) + measured training state (1.32) + CPU-profiled activations; see the config header |

Every headroom in that column is `card − peak allocated` against the full 34.36 GB board. **That is
the optimistic bound, not the number to plan with** — it credits you with the CUDA context, the
allocator's reserved-but-unallocated slack, and the compositor, none of which
`max_memory_allocated()` can see. Subtracting them costs roughly 1.7 GB on every row (0.90 measured
+ ~0.80 estimated), plus 0.3–1.5 GB more if the box is not headless.

That correction changes no verdict except `dream.py`'s, which is why only that row carries a second
figure. It is also why the training row disagrees with its own config header: the header quotes
**~4 GB** where the table says `+6.66`, because it works against ~31.8 GB of *allocatable* VRAM
rather than the 34.36 GB board. Both are arithmetically right; `~4` is the one to plan with.

**`dream.py` at archived settings has no headroom on this card. Use `--offload-text`.**

Two wrong answers have been given about this row, and it is worth keeping both visible because the
second is the more seductive one.

The first draft of this page compared the artifact's 32.54 against "32 GB" and reported a deficit.
That was a unit error: the artifact is decimal GB, the card is 32 GiB = 34.36 decimal GB.

The correction to that — "+1.82 GB, it fits" — is also wrong, because the `+1.82` compares two
things that are not comparable. `max_memory_allocated()` counts *allocated* bytes. What has to fit
in the card is *occupied* bytes, which is larger by two terms the allocator never reports:

| term | GB | basis |
|---|---|---|
| peak allocated | 32.54 | measured, `t36-zerogpu-motion-seed0/dream.json:265` |
| caching-allocator slack (`reserved − allocated`) | +0.90 | **measured**, the one run here that logged both — job 183599:129-130 |
| CUDA context | +0.80 | estimate, not measured on any card in this repo |
| **occupied** | **34.24** | |
| board total as `nvidia-smi` reports it | 34.19 | ~32607 MiB after board reserve — **unverified**, no 5090 has been seen here |

That is **0.05 GB over**, on a row where two of the three terms are estimates. The other archived
peak (32.47) works out to 34.17 — 0.02 GB *under*. The two runs land on opposite sides of the line
and the gap between them is smaller than the error bar on the line's position. The correct reading
is not "fits" or "does not fit"; it is **these artifacts cannot settle it, and nothing about the
run leaves room to be wrong.** A desktop compositor (0.3–1.5 GB) settles it against you.

So do not run it at archived settings to find out. `--offload-text` moves the 11.36 GB umT5 tower
to the CPU for the encode, taking the peak to **~21.2 GB allocated / ~22.9 GB occupied** — the
question stops being close. The alternatives are fewer clips (`--episodes` × `--windows-per-episode`;
the archived run recorded `clips: 16`) or fewer Euler `--steps` (that run used 32), but both change
what you measured, and `--offload-text` does not.

Two caveats that apply across the table and that no per-row number can express:

- **The load transient is unmeasured everywhere except `dream.py`.** The smoke and probe peaks were
  recorded *after* `reset_peak_memory_stats()` (`hf_job_wan_smoke.py:385`), so they describe the
  steady-state forward, not the worst moment of the run. `dream.py` never resets, which is why its
  number is the largest one here — it includes the load. On a card this tight the load, not the
  forward, may be what OOMs. Measure it: run §1 and watch
  `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in another shell.
- **Nothing in this table was measured on a 5090.** The two cards it was measured on are an H200
  (141 GB) and a ZeroGPU RTX PRO 6000 (96 GB); neither was under memory pressure, so none of these
  numbers is an *adaptive* peak.

---

## 1. Prove the adapter and measure peak VRAM

`hf_job_wan_smoke.py` is the gate: 13 checks against the real 5B weights, and it reports
`peak_vram_gb`. Run it before anything else — it is minutes long and it replaces every estimate on
this page with a number.

```bash
python scripts/hf_job_wan_smoke.py \
    --source /path/to/Wan2.2-TI2V-5B \
    --episode datasets/gr00t-apple-full/gr00t-apple-000000 \
    --camera ego --frames 5 --height 256 --width 448 \
    --offload-text
```

Expect `[PASS]` on all 13 and, from the geometry dump, `num_layers: 30`, `feature_dim: 3072`,
`feature_blocks: [15, 22]`, `dtype: bfloat16`.

> This is the check that failed on Discoverer+ job 183565 — six checks in, on a two-line bug in its
> own synthetic state (`gripper_dims` defaulting to 1 against a real G1 episode's 2). Fixed in
> `78fc56d`; `tests/test_wan_smoke.py` now covers the path that broke.

Note the smoke geometry (5 frames, 256×448, batch 1) is **not** the training geometry. Its VRAM
figure describes inference at that resolution, nothing else.

---

## 2. Generate predictions on the holdout

`train_t16_lora.py` writes weights and a training log and deliberately does **no** evaluation — the
fine-tune runs in preemptible 4-hour chunks, so an eval welded to the end of one would run on
whichever chunk happened to finish. `eval_t16.py` is that step, standalone:

```bash
python scripts/eval_t16.py \
    --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --train-episodes configs/splits/i8_train_362.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B \
    --device cuda
```

**`--train-episodes` is the split proof's external witness, and it is safe to pass on every
call.** `train_t16_lora.py` records `train_episode_ids` on every run, so any checkpoint trained
after 2026-08-01 is scored under the *disjointness* proof, where the witness is **mandatory** —
without it the recorded ids and the recorded hash are two fields of one self-description and the
check compares the checkpoint against itself. Checkpoints written before the field exists (the
archived `t16-lora-seed0`) are scored under the *complement* proof, where the file is checked as a
redundant cross-check instead of refused. `i8_train_362.txt` is the complement of the holdout above
(402 − 40 = 362), so it is the right file for both. An I-8 rung needs its own
`configs/splits/i8_train_0NN.txt`; passing the wrong one is refused, not scored.

**`--backbone-source` is not optional for a checkpoint trained on Discoverer+**, which is every
checkpoint this runbook is about. The frozen weight location is kept out of the committed config and
folded in at run time from the flag, so the archived config records `/valhalla/projects/...`.
Without the override, loading here goes looking for weights on a filesystem this box does not have.
The flag also exists on `rollout.py` and `serve_policy.py` for the same reason.

**The weight path IS part of `config_hash`, and this page and `policies.py` both used to say it was
not.** `config_hash` (`src/wam/interfaces/versioning.py:65-73`) digests the whole canonicalized
config with **no exclusion list**, and `train_t16_lora.py:920` hashes the config *after* splicing
`--backbone-source` into it. Measured, on the archived checkpoint:

```
runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors
  recorded config_hash                                      45ee9e6035eb…
  recomputed from the embedded config                       45ee9e6035eb…   (reproduces)
  same config, backbone.checkpoint_path -> a local dir      915ca4b0fd32…
  same config, device cuda -> cpu                           92fd74189354…
```

Two consequences, and only the second one bites:

- **Reading a checkpoint here is safe.** `_relocate_backbone` substitutes the path and device and
  deliberately does *not* recompute the hash. The recorded hash keeps describing the run that
  produced the checkpoint, which is what AC-04 wants it to describe.
- **Training here does not reproduce the cluster's hash.** A local retrain with a local
  `--backbone-source` produces a *different* `config_hash` for a bit-identical model, because the
  path is inside the hashed config. So "identical experiment ⇒ identical `config_hash`" is **not**
  true across machines — despite being the stated reason for keeping the path out of the YAML.
  AC-04 traceability itself is intact: every checkpoint still records the hash of the config it was
  actually trained with, and that is what the chain needs. What you cannot do is use the hash to
  argue two runs on two machines are the same experiment. (`batch_size` has the same property and
  bites the same way for §5 — see `configs/training/joint_wan_gr00t_5090.yaml`'s closing note.)

One forward pass per chunk, ~960 chunks over 40 episodes — minutes, well under a GPU-hour. Writes
into the run dir:

| file | what |
|---|---|
| `predictions.jsonl` | the only artifact the scorers need. No GPU ever required again |
| `e1.json` / `e1.md` | E1 action metrics, same format as the baseline's |
| `bench.json` / `bench.md` | the WAM-Bench ladder (`docs/benchmark.md`) |

**It refuses to score an unproven split.** The trainer hashes the manifests of the episodes it
actually trained on into `dataset_snapshot_ref`; this script recomputes that hash over
`dataset − holdout` and stops unless the two match:

```
REFUSING TO SCORE — split not provable.
  checkpoint trained on: sha256:...
  dataset minus holdout: sha256:...
```

That means the holdout is not the complement of the training set, so those episodes may have been
trained on and every number downstream would be meaningless in the one way that matters. Fix the
`--dataset`/`--holdout` arguments rather than reaching for `--skip-split-check` (which scores
anyway and drops an `UNPROVEN_SPLIT` marker next to the artifacts).

The command above is the **complement** proof, which is what `runs/t16-lora-seed0` and every other
pre-I-8 checkpoint is judged by. A checkpoint that records its own `train_episode_ids` (every
fresh run does) is judged by the **disjointness** proof instead, and that one additionally
requires `--train-episodes <the reviewed split file>`: without an external witness the recorded
ids and the recorded hash are two fields of one self-description, and checking them against each
other cannot fail. The evaluator picks the proof from the checkpoint, never from a flag.

The checkpoint knows how to load itself: a Wan-backed one carries no base weights, and
`load_joint_policy` branches on the embedded config's `requires_external_weights` to build the
frozen base and load with `strict=False`.

---

## 3. Score — no GPU, runs anywhere

From here on it is numpy on archived predictions. This runs on the 5090, on the Mac, on CI:

```bash
# The ladder, and the comparison that matters
python scripts/run_bench.py runs/t16-lora-seed0
python scripts/run_bench.py runs/d1-full-gen-seed0 runs/t16-lora-seed0 --compare

# AC-07: does video help? (identical split, same seed/head/threshold)
python scripts/run_ablation.py --baseline-run runs/d1-full-gen-seed0 \
    --dataset datasets/gr00t-apple-full --device cuda
```

**The bar is `skill_vs_repeat_pct > 0`** — WAM-Bench rung **L1, beats-inertia**. Not "beats the
action-only baseline": that baseline (mse 1.10e-5) itself loses to a causal repeat-last-action
heuristic (9.14e-6) by 17 %, so clearing it proves nothing. See `docs/benchmark.md`.

Reference points on the identical 40-episode holdout, including the first T-16 checkpoint:

| | `d1-full-gen-seed0` (action-only) | `t18-real-ablation-seed0` (world-action) | `t16-lora-seed0` (Wan 5B LoRA) |
|---|---|---|---|
| frame mode | tiled ⚠ | tiled ⚠ | tiled ⚠ (re-scored — §3b) |
| level | **L0** beats-doing-nothing | **below L0** | **L0** beats-doing-nothing |
| score | 28.6/100 | 19.9/100 | 48.4/100 |
| `skill_vs_repeat_pct` | −20.9 % | −129.0 % | **−32.4 %** |

⚠ **tiled** = one frame repeated 9×, which is how every number recorded before 2026-08-01 was
measured (§3b). Only `t16-lora-seed0` has since been re-scored in the mode it was trained in, where
it is **−21.80 %**; the other two columns have never been measured that way.

The bar is unbeaten in every column — all three lose to inertia, and the levels say so. What this
table does **not** support is a ranking: reading `skill_vs_repeat_pct` across columns now compares
one in-distribution number against two freeze-frame ones. Read the level. `docs/benchmark.md` has
the full column and the diagnosis.

### 3b. T-29 — re-score with the frames training actually used

> **Ran 2026-08-01 (Slurm job 184648). The verdict survives; the published figure does not.**
> Both arms on one H200 from one checkpoint (`runs/t16-lora-seed0/checkpoints/step-020000`), the
> same proven 40-episode holdout, the same 1 040 chunks, differing in exactly one flag. Artifacts:
> `runs/t16-lora-seed0/eval-t29-{tiled,history}/`.
>
> | metric | tiled (as published) | real window (`--frame-history`) | delta |
> |---|---|---|---|
> | **`skill_vs_repeat_pct`** (the L1 gate) | **−32.45 %** | **−21.80 %** | **+10.65 pp** |
> | `ci_skill_vs_repeat_pct` (L2, critical chunks) | −50.74 % | −23.11 % | +27.64 pp |
> | `skill_vs_zero_pct` (L0) | +25.88 % | +31.83 % | +5.96 pp |
> | `mse` | 1.21027e-05 | 1.11298e-05 | −8.0 % |
> | `horizon_ratio` | 1.30 | 1.32 | |
> | `smoothness_ratio` | 0.29 | 0.32 | |
> | level | L0 | **L0** | unchanged |
> | score, bench spec 0.1.0 | 48.4 | 50.6 | +2.2 |
> | score, bench spec 0.2.0 | 28.4 | 30.6 | +2.2 |
>
> The confound below was real and was worth 10.65 pp — about a third of the 32.45 pp gap — and did
> not come close to closing it. L1 still fails, by 21.80 pp. The tiled arm reproduces the archived
> `eval-latest/bench.json` to every digit, so the published −32.4 % is confirmed to be the
> freeze-frame measurement and the A/B is clean. **Only `t16-lora-seed0` was re-measured** — the
> other two columns in §3 are still tiled-only. The runbook below is unchanged and is still how you
> reproduce this locally.

**Why this was worth a job.** Every number in the §3 reference table was produced with
`predict()` tiling a single camera frame to the backbone's 9-frame context, while training fed the
real 9-frame window ending at the chunk (`docs/improvements.md` I-7). A video backbone trained on a
moving clip was graded on a freeze-frame — and repeat-last-action, the baseline it loses to, is
nothing but motion continuity.

`--frame-history` feeds the window `EpisodeDataset` selected, via the same
`frame_window_indices`. No retraining; the checkpoint is untouched and only the input changes.

```bash
# A: how everything before 2026-08-01 was measured
python scripts/eval_t16.py --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda \
    --out runs/t16-lora-seed0/eval-t29-tiled

# B: what training fed the model — one flag different
python scripts/eval_t16.py --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda \
    --frame-history --out runs/t16-lora-seed0/eval-t29-history

python scripts/run_bench.py runs/t16-lora-seed0/eval-t29-{tiled,history} --compare --no-write
```

Run **both** rather than reusing `eval-latest`: that one came from another day and possibly another
machine, and an A/B whose halves differ in more than the thing under test is not an A/B. Cost is one
extra pass — minutes. On Discoverer+ the whole thing including the verdict is one job:
`sbatch cluster/discoverer/61_eval_t29_frame_history.sbatch`.

The mode is written into `bench.json`'s `run_name` (`…+frame_history`), because two prediction files
from one checkpoint differ only in what the policy was shown, and a report that does not say which
will eventually be compared against the wrong one.

**Decision rule, fixed before the run** — `skill_vs_repeat_pct` moves toward or past 0 → T-16 and
T-18 were measured out of distribution, `docs/benchmark.md` needs a correction rather than an
addendum, and AC-07 reopens (re-run the T-18 ablation the same way before concluding anything).
Essentially unchanged → the model had the motion and still lost to inertia, the negative is about
the model rather than the harness, and I-8 (the data-scaling curve) is next.

**Resolution, 2026-08-01 — the rule text above is left as it was written; this is what it
resolved to.** The result landed between its two branches, and both fired in part. Toward-0: yes,
by 10.65 pp, so `docs/benchmark.md` took a correction rather than an addendum and **AC-07 is back
to OPEN** — undetermined, not answered. The old claim that the fine-tune is worse than the
action-only baseline rested on −32.4 % vs −20.9 %; in distribution it is −21.80 % vs an unknown.
**That unknown was measured the same day** (`scripts/rescore_archived.py`, laptop CPU, zero
allocation): both baselines re-score to −20.88 % and −129.00 %, i.e. neither moved, because the
`tiny` backbone does not use the frame axis. AC-07 is readable again and its answer is unchanged —
*no measurable world-action advantage* — with the 0.92 pp T-16-vs-baseline gap explicitly **not** a
clean ablation (backbone and branch differ at once). `docs/improvements.md` I-7.
Past 0: no, so the second branch also holds — the model had the motion and still lost to inertia
by 21.80 pp, and that part of the negative is about the model rather than the harness.

The parenthetical is the one instruction still **OUTSTANDING**: `t18-real-ablation-seed0` and
`d1-full-gen-seed0` have not been re-run this way, so any table carrying all three is mixed-mode
and is not a comparison. It costs ~0.4 GPU-h and no retraining, and it is item 2 in
`docs/improvements.md`. The rule's closing "I-8 is next" is superseded by that queue: T-30 (§3c)
is item 1 and running, the re-score is item 2, I-8 follows.

`--compare` refuses two runs whose holdouts differ, so the columns always mean the same thing.

Because scoring only reads `predictions.jsonl`, a **new rung costs no retrain** — every past run is
re-scorable.

### 3c. T-30 — re-score through the action branch that was trained but never read

The joint model trains **two** action readouts and deploys one. `ActionHead` regresses the whole
chunk in one shot from the pooled features; a rectified-flow branch (`velocity_head` +
`action_recon`, `weights.action_flow = 1.0`) models the same chunk as a distribution and is never
touched at inference. A single-shot L2 regressor under a one-to-many conditional is mean-seeking,
and `t16-lora-seed0` is **consistent with** one: chunk RMS **0.00226** against the demonstrations'
**0.00404**, `smoothness_ratio` **0.29** tiled and **0.32** in the `--frame-history` mode the arms
below run in (§3b) — the diagnosis does not turn on which. `--flow-sampler` reads the chunk out of
the flow branch instead. No retraining; the checkpoint is untouched and only the decode step changes
(`docs/improvements.md` I-3).

**"Consistent with", not "the signature of"** — an earlier version of this section said the
latter, and only one alternative had actually been ruled out. Ruled out: bounded-output
saturation (max |target| 0.0192 against a `tanh`, and `limit_penalty` bites only outside ±0.95).
Not ruled out, and each sufficient on its own to produce small smooth chunks:

- **the one-sided jerk regulariser.** `configs/training/joint_wan_gr00t.yaml` sets
  `weights.smoothness = 0.01`, and `JointTrainer.compute_losses` applies `smoothness_loss` to
  `decoded_targets` — the *regression* head's output — and to nothing else. The flow branch is
  never charged for jerk, so a `smoothness_ratio` gap between the two readouts is predicted by
  the objective. That is why the smoothness clause of the rule below is **confounded** and reads
  as descriptive rather than decisive.
- **L2 shrinkage.** `action_reg` is an MSE against targets whose own RMS is 0.004, with AdamW
  weight decay on top; a head that under-shoots magnitude uniformly is indistinguishable, on RMS
  and jerk alone, from one that averages modes.

Separating those from mean-seeking needs an intervention (`smoothness = 0` and retrain, or a
multi-modal task), not a readout swap.

**Run the CPU pre-flight first — it is seconds and it can cancel the GPU pass outright:**

```bash
python scripts/check_action_latent.py \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000 \
    --predictions runs/t16-lora-seed0/eval-latest/predictions.jsonl
```

It prints two bounds on everything the A/B can win, both measured on 2026-08-01 for that pair:

| | target MSE | what it is |
|---|---|---|
| ceiling | **8.10e-07** | encode the demonstrated chunks, decode straight back through `action_recon` — a perfect sampler |
| this run | 1.21e-05 | the deployed regression head, tiled (1.11e-05 in `--frame-history`, §3b) |
| L1 bar | 9.14e-06 | repeat-last-action |
| zero-delta | 1.63e-05 | hold still |
| floor | **1.68e-05** | decode the per-step latent *centroids* — a sampler that recovers only *which step* |

The ceiling says the latent carries the chunk almost perfectly, so a poor A/B result is about the
velocity head and not about the representation. The floor is the one to read before booking GPU
time: step index is recoverable from the latent at **100 %** accuracy (within-step std 0.056 vs
between-step 1.42) and `ActionVelocityHead` takes **no step index** — it sees `[z_t | pooled | t]`
and nothing else. So the per-chunk content the sampler has to hit is a small perturbation riding on
a much larger positional signal, and landing near the floor is the *expected* outcome rather than
evidence of a bug.

```bash
# A: the regression head (today's deployed readout)
python scripts/eval_t16.py --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda \
    --frame-history --out runs/t16-lora-seed0/eval-t30-regression

# B: the flow readout — one flag different
python scripts/eval_t16.py --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda \
    --frame-history --flow-sampler --flow-steps 32 \
    --out runs/t16-lora-seed0/eval-t30-flow32

# B_mean: the same readout, 8 draws averaged — the arm the rule keys on (see below)
python scripts/eval_t16.py --run-dir runs/t16-lora-seed0 \
    --dataset datasets/gr00t-apple-full \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda \
    --frame-history --flow-sampler --flow-steps 32 --flow-mean-k 8 \
    --out runs/t16-lora-seed0/eval-t30-flow32-mean8

python scripts/run_bench.py runs/t16-lora-seed0/eval-t30-{regression,flow32-mean8} \
    --compare --no-write
```

Both arms carry `--frame-history`, and that is now fixed rather than conditional: T-29 reported on
2026-08-01 (§3b) and `--frame-history` is the mode left standing, so every T-30 arm runs inside it.
Mixing the two modes here would collapse the two A/Bs into a 2×2 nobody pre-registered. The
readout goes into `bench.json`'s `run_name` as `…+flow32s0k8`, for the same reason the frame mode
does, and **one `--out` holds exactly one readout** — the script refuses to write a second arm's
artifacts over a first one's, because every filename is fixed and `--out` defaults to `--run-dir`.
`--flow-steps` and its companions without `--flow-sampler` are refused rather than silently
scoring the regression head into a flow-named output dir. Sweep `--flow-steps 1 4 16 32 64` as
separate runs to show the verdict does not hinge on the integrator — **n=1 is the control**: one
Euler step from noise is what "rectified flow needs one step" would deploy, and it measures how
far from straight this field actually is. On Discoverer+ the whole sweep plus the verdict is one
job: `sbatch cluster/discoverer/63_eval_t30_flow_head.sbatch` — **ran 2026-08-01 as job 184670 and
the answer is negative**: all nine flow arms land below L0, and the mean-of-8 measurement arm the
rule keys on is 11.1× worse than the regression readout (1.23988e-04 vs 1.11298e-05). The arms sit
at 7.4× `FLOOR_MSE`, well above it, so the flow branch does not merely mis-place the chunk in
time — as trained, it does not carry the chunk's content through the sampler at all.

#### Why the headline arm averages k draws

`skill_vs_repeat_pct` is MSE-derived, and for any calibrated conditional
`E‖a − draw‖² = E‖a − mean‖² + E‖draw − mean‖²`. A single unbiased draw therefore scores **worse**
than the conditional mean by exactly the conditional variance — so comparing one draw against a
head accused of being mean-seeking rewards the defect under test. `--flow-mean-k 8` leaves 1/8 of
that penalty and makes the comparison apples-to-apples. It is a **measurement** arm, never a
deployment candidate: averaging draws *is* the mean-seeking the flow branch exists to avoid. The
single-draw arm stays, as the measurement of what a deployed sampler would emit — magnitude
(`RMS/demo`) and jerk (`smoothness_ratio`), neither of which rewards mean-seeking.

One consequence of the determinism contract belongs here too: the seed is re-drawn per call and
never advanced (that is what keeps T-25's rollouts bit-identical), so **all 1 040 chunks of one
arm are integrated from the same noise vector**. Per chunk that is a fair draw, but the errors
across the holdout are correlated through it, so the arm's aggregate is conditioned on that one
vector. That is measured, not assumed: the second `--flow-seed` arm re-runs the whole holdout from
a different vector and its difference *is* the band.

#### Why there is a warm-start arm

Training always paired *(features from video noised to t, action latent noised to the same t)* —
`co_denoise` shares one `t`. The sampler computes **one** backbone pass at `t=1` on the clean
observation and reuses those features at every `t_k`, so near `t_k = 0` the velocity head is asked
about a combination it never saw. That is the same shape of confound as I-7, and without an arm to
measure it, a negative would be unreadable in exactly the way T-16's was. `--flow-t0 0.6` starts
the integration at 0.6 from the regression chunk re-encoded by `action_encoder` and noised to
`t0` — only in the region where the two timesteps roughly agree. It **inherits the regression
mean** by construction (there is no clean action latent at inference; using the demonstrated chunk
would make it an oracle rather than a readout), so it can never show that the flow branch models
the conditional. What it can do is separate *"the branch is dead"* from *"we sampled it outside
its training region"*. The faithful sampler — n backbone passes on an observation noised to each
`t_k` — is not run at all, so **every negative below is about the flow branch AS SAMPLED THIS WAY,
never about the flow branch as trained.**

**Decision rule `T30_RULE_V2`, fixed before the run** — the version
`cluster/discoverer/63_eval_t30_flow_head.sbatch` actually runs; V1 is superseded **unrun**, no arm
of it was ever executed, and its three defects are kept in that file's header. Keyed on
`skill_vs_repeat_pct` of the
**mean-of-k** arm against A, with `BAND = max(3·σ̂, 10 pp)` and σ̂ the *measured* `|seed0 − seed1|`
spread (a second `--flow-seed` arm, the same construction I-8 uses; a bare literal band would be a
threshold nobody had tested). B_mean beats A by more than BAND *and* clears 0 → the mean-seeking
head was the T-16 negative, the flow readout becomes the deployed path, and `docs/benchmark.md`
needs a correction. Beats A but stays below the bar → the head cost real skill and was not the
whole story; keep the regression head deployed until a closed-loop number exists, then I-8. Inside
the band → **the two readouts score the same on this metric** — that sentence, and not "the flow
branch is decorative", which additionally requires the measured draw-to-draw spread to be under
10 % of the demonstrations' RMS. Worse than A by more than BAND → the mean was doing real work.
Both negative branches then consult the **variance-matched warm arm** (`--flow-t0` with
`--flow-mean-k`): if it clears the band against A or against B_mean, the negative belongs to the
sampler's conditioning mismatch and the next step is the sampler, not I-2. `smoothness_ratio` is
reported for every arm but gates nothing — the regression head carries a jerk penalty the flow
branch does not. Checked *first*, on **every** arm: RMS |targets| above 3× the demos' 0.00404, or
anything non-finite, or MSE above 5× zero-delta, is an integration bug — fix and re-run, record
nothing. An MSE merely a little above zero-delta is **not** that case; that is the floor, and the
floor is a result.

`eval_t16.py` also writes `timing.json` (ms/chunk) now — the sampler's cost is the delta between
the two arms, and it is the only number in the run that cannot be recomputed from the archive.

---

## 4. Run the closed loop

Offline metrics filter candidates; they do not tell you whether the thing runs in a loop. Two ways
in, both using the checkpoint directly:

```bash
# In-process, against the mock robot: safety layer, watchdog, receding horizon
python scripts/rollout.py --robot mock --policy joint \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors \
    --backbone-source /path/to/Wan2.2-TI2V-5B \
    --contract-from-dataset datasets/gr00t-apple-full \
    --instruction "move the apple to the plate" \
    --policy-device cuda --rollouts 5

# MuJoCo G1 + Dex3 with rendered pixels (docs/sim.md)
python scripts/rollout.py --robot mujoco_g1 --policy joint \
    --checkpoint <same> --backbone-source <same-weights> \
    --contract-from-dataset datasets/gr00t-apple-full \
    --instruction "move the apple to the plate" \
    --policy-device cuda --policy-camera head --image-hw 120 160
```

**`--policy checkpoint|joint` refuses to start without a policy contract**, and nothing writes a
`policy_contract.json` today, so contract discovery cannot rescue a command that omits it. The two
flags above are what make these runnable: the contract declares which state groups training used
(our converter wrote `ValidityMask(imu=False)` for all 402 episodes, every adapter here reports
`imu=True`, and the encoder moves 2.011 against an embedding norm of 2.454 between the two), and
the instruction is checked against the ones the run actually trained on. `--no-policy-contract`
opts out and is recorded as `"contract": null` so an unchecked run cannot pass for a clean one.
Full detail and the divergence table: `docs/sim.md`.

Or serve it and drive from elsewhere on the network — this is how the Mac can sit in the loop
without holding the weights:

```bash
# on the 5090
python scripts/serve_policy.py --joint --checkpoint <same> \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda
# anywhere — no weights, no GPU, no backbone source
python scripts/rollout.py --policy remote --server-uri ws://<5090-host>:8765
```

What a `--policy joint` sim run measures: **latency against the deadline**, the safety and watchdog
paths under real model timing, and whether predicted chunks survive the filter at all. What it does
**not** measure: task competence. MuJoCo renderings are not RealSense frames and no backbone here
has seen one. Task success is E3 and needs the robot.

Watch the `min_policy_rate_hz` line. One backbone pass per cycle is essentially the whole latency
budget — `policy_deadline_ms` is 500 ms and `min_policy_rate_hz` is 2 Hz (`ExecutorConfig`), against
a measured 79 ms per window for the Wan pass — so if the loop misses its deadline the fix is the two
levers from the top of this page, not a smaller batch. Note that only one of them buys latency:
truncating the DiT after block 10 skips 19 of 30 layers of compute, while `--offload-text` (already
available here, §0b) buys VRAM and nothing else — the umT5 forward it moves to the CPU runs once per
distinct instruction, not once per cycle.

`--flow-sampler` (§3c) is not wired into `rollout.py` yet, deliberately: it is an offline
question until the A/B answers it. When it is, it adds *n* MLP evaluations per cycle — at T-16's
dimensions one step is roughly 13 M MAC at batch 1, so 32 steps is arithmetic noise next to one
79 ms backbone pass, and `timing.json` from the two eval arms is the measured version of that
claim. The variant that would blow the budget is the one this design rejects: re-running the
backbone at every sampler timestep, where 79 ms/step leaves room for **n ≤ 5** inside 500 ms.

---

## 5. Fine-tune here — the remaining I-8 / T-32 rungs

The one training job that fits. **Everything in this section is unverified on a 5090**: the memory
arithmetic is measured (weights from safetensors headers, activations from a CPU profiler at the
exact shapes), but no step of it has run on this card, and the wall-clock figures rest on an
explicitly-labelled speed assumption.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FIRST TIME: 20 steps into a throwaway directory, to turn the estimated activation term
# into a number measured on your card. Watch nvidia-smi in another shell.
RUNG=040 PREFLIGHT=1 ./scripts/run_i8_rung_local.sh

# then the rung itself
RUNG=040 ./scripts/run_i8_rung_local.sh
RUNG=120 ./scripts/run_i8_rung_local.sh
```

`scripts/run_i8_rung_local.sh` is the local twin of `cluster/discoverer/55_train_i8_rung.sbatch`:
same splits, same 20 000 steps, same effective batch, same resume-until-`DONE` loop. Rung 362 is
**not** run here — it is `runs/t16-lora-seed0`.

**The config differs from `configs/training/joint_wan_gr00t.yaml` in exactly one field:
`batch_size` 8 → 2.** Not the geometry, not `feature_blocks`, not `lora_rank`, not the losses, not
the learning rates, not the step budget. That is deliberate — T-32 is a data-scaling *curve*, and a
rung trained at a different geometry is a different experiment wearing the curve's axis labels.

> **`batch_size: 2` is only comparable to rung 362 with `--grad-accum 4`.** Effective batch stays 8.
> `train_t16_lora.py` normalizes the accumulated loss by `1/len(batches)`, so 2×4 is arithmetically
> the same optimizer update as 1×8, not a 4× larger gradient. The runner pins the pair and refuses
> a resume that changes it; run the config by hand without `--grad-accum 4` and you have silently
> trained at effective batch 2.

Where the memory goes, from the config header (all decimal GB):

| | umT5 resident | `--offload-text` |
|---|---|---|
| frozen weights (measured) | 24.18 | 12.82 |
| trainable + grads + AdamW (measured, from the rung-362 checkpoint) | 1.32 | 1.32 |
| **resident floor** | **25.50** | **14.14** |
| + activations at batch 2 (CPU-profiled, extrapolated) + allocator + CUDA context | ~27.7 | ~15.9 |

Both fit 34.36 GB. **`--offload-text` is helpful here, not required** — and it is nearly free,
because this corpus has one distinct instruction (§0b), so the CPU umT5 forward is paid once in
20 000 steps. Take it anyway: it turns a ~4 GB margin against the two unmeasured terms into a ~16 GB
one. The runner sets `OFFLOAD_TEXT=1` by default.

`BATCH=8 ACCUM=1` fits on paper too (~29.0 GB) and is the H200's own setting; it is not the default
because 2.8 GB of margin is inside the error bar of the two terms nobody has measured — the cuDNN
conv3d workspace for the fp32 VAE, and a desktop compositor if the box is not headless. If your
preflight run measures a peak where the config predicts, `BATCH=4 ACCUM=2` or `BATCH=8 ACCUM=1` is a
pure speedup at the same effective batch.

**Wall clock is an assumption, and the header says so.** Measured: 0.42 s/step on one H200 at
batch 8 (`runs/_slurm_logs/t16.183601.out`). Assumed: the 5090 is 2.0× slower per step, picked
inside a band whose ends are real (1.0× if the step is launch-latency bound — 60 tokens through 30
blocks of small LoRA-wrapped linears — up to 2.68× if it is bound by streaming frozen weights,
4.8 TB/s HBM3e vs 1.79 TB/s GDDR7). That gives ~5.8 h per rung, ~11.7 h for both, inside a band of
4.7 h to 15.7 h. **Nothing here is measured; the first rung measures it.**

`Ctrl-C` is safe: the trainer stops at a step boundary, checkpoints, and exits 0. A crash or a
reboot needs no special handling — run the same command again and it resumes from `latest`.

The `config_hash` caveat from §2 applies with a second cause: `batch_size` is part of
`JointTrainingConfig`, so a rung from this file hashes differently from `runs/t16-lora-seed0` no
matter what `--grad-accum` does (`grad_accum` is a runtime argument and is not in the hash at all).
The comparability claim rests on the effective batch and the loss normalization, not on the hash.

---

## 6. What this box cannot do

| | why |
|---|---|
| The T-16 fine-tune of record | it already happened, on Discoverer+ — rung 362 is `runs/t16-lora-seed0` and is not re-run |
| Task success / real safety | E3 — needs the G1 |
| Optimism-bias scoring | needs failure demonstrations; our data is success-only |
| Video-fidelity metrics | needs stored predicted frames (nothing writes them yet) |

---

## Troubleshooting

**`no kernel image is available for execution on the device`** — the wheel predates sm_120.
Reinstall from the cu128 index (§0a). `torch.cuda.get_device_capability()` will *not* tell you this
in advance; `scripts/preflight_gpu.py` section 3 will, by launching real kernels.

**`ModuleNotFoundError: No module named 'pyarrow'`** (or `diffusers`, `transformers`, `cv2`, `av`,
`websockets`) before the GPU is ever touched — you installed `.[dev]`, which this page used to tell
you to. Install `.[local]` (§0). The preflight prints the exact repair line for whatever is missing.

**Killed / the box swaps during weight load, with no CUDA error at all** — host RAM, not VRAM. The
towers are materialized in system RAM before they reach the card because `device_map` is unreachable
from every entry point except `hf_job_wan_smoke.py`. 32 GB is the floor (§0).

**OOM while loading, then fine afterwards** — the text tower (11.36 GB) and the DiT (10.00 GB) are
both resident during the first `condition_text` call. Pass `--offload-text` (§0b), which moves the
tower to CPU right after conditioning (`src/wam/backbones/wan_i2v.py:397`), and export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the process starts (§0).

**`gripper: expected 1 values, got 2`** — a `StateMLPConfig` left at the default `gripper_dims=1`
against a real G1 episode, which has one gripper value per hand. Derive both dims from the state.

**Weights not found under `/valhalla/projects/...`** on a machine that has no such path — the
checkpoint recorded where its frozen base sat *on the cluster*. Pass `--backbone-source` (§2). Only
the location is substituted; the recorded `config_hash` is left alone on purpose, so it keeps
describing the run that produced the checkpoint. That path **is** inside the hash, though — see §2
for what that costs when you *train* here rather than load.

**`REFUSING TO SCORE`** — see §2. This is the guard working; do not paper over it.

**`holdout mismatch — not comparable`** from `run_bench.py --compare` — the two runs were scored on
different episode sets. Re-run `eval_t16.py` for one of them against the other's holdout.

---

## See also

- `scripts/preflight_gpu.py` — the gate in §0; `--json` for the machine-readable version
- `docs/local_gr00t_assets.md` — what GR00T is **already on this box** (two Isaac-GR00T checkouts, the
  `venvs/arena` gr00t env that sees the 5090, three post-trained checkpoints, the 402-episode corpus)
  and the one thing that is not: the `nvidia/GR00T-N1.7-3B` base. Also records that **`MUJOCO_GL=egl`
  is required here** — without it MuJoCo picks GLFW, which wants an X display these shells lack.
- `configs/training/joint_wan_gr00t_5090.yaml` — the §5 memory arithmetic, in full, with provenance
- `docs/benchmark.md` — the ladder, its KPIs and the external benchmark landscape
- `cluster/discoverer/README.md` — where the T-16 fine-tune of record ran
- `docs/sim.md` — what the MuJoCo loop proves and what it does not
- `docs/discoverer.md` — cluster facts, quotas, and the login-node rules
