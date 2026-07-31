# Local GPU runbook — run, test and benchmark a WAM checkpoint

For a single consumer GPU box (written against an **RTX 5090, 32 GB**). Everything here is
inference and scoring.

**Fine-tuning is not in this runbook.** T-16 trains on Discoverer+ (`cluster/discoverer/README.md`),
because that is where the resume harness, the 402-episode dataset and the accounted GPU hours live.
This box is for the loop that follows a fine-tune: **prove the adapter → generate predictions →
score them → run the closed loop**.

---

## Why one GPU is enough

WAM does not generate video at inference. `JointWorldActionModel.predict()`
(`src/wam/training/joint.py:362`) runs **one** backbone pass at the clean end of the flow, reads the
readout blocks, applies `ActionHead`, and throws the video velocity away — no denoising loop, one
pass per control cycle. The video branch's job was to shape the features during *training*.

The sequences are also small. T-16's geometry is 9 frames at 128×160; the VAE strides 16 spatially
and 4 temporally, so latents are `(B, 48, 3, 8, 10)`, and patchifying `[1, 2, 2]` gives
**60 tokens per sample**.

| resident at inference | bf16 |
|---|---|
| Wan DiT (5B), frozen | ~10 GB |
| umT5 text tower, frozen | ~11 GB |
| VAE (encoder path only — `decode_video` is unused here) | ~1.4 GB |
| LoRA adapters + action branch | <0.1 GB |
| activations, 60 tokens, batch 1 | negligible |
| **total** | **~23 GB** |

**Measured: 24.3 GB peak / 25.2 GB reserved** (smoke job `183599` on an H200; the readout probe
independently saw 24.65 GB). So on a 32 GB card this fits with roughly **7 GB spare**, not the 20 GB
an earlier version of this table implied.

That earlier version put the text tower outside the budget, reasoning that `condition_text` is cached
(`src/wam/backbones/wan_i2v.py:459`) and one task has one instruction, so the tower runs once.
Caching the *output* does not evict the *weights* — 11 GB of umT5 stays resident for the whole run
unless something explicitly drops it. The ~12 GB figure was the **offloaded** budget presented as the
default.

**Two levers, and the first is no longer optional on a 32 GB card:**

- **Evict the text tower** — `adapter.offload("text_encoder")` exists, exposed as `--offload-text` on
  the smoke script, but is *not* wired into `eval_t16.py`, `rollout.py` or `serve_policy.py`. Expect
  ~13 GB with it. Unmeasured: the 24.3 GB above was recorded with `offload_text: false`.
- **Truncate the DiT** — not built. T-16 reads blocks `[2, 10]` of 30
  (`configs/training/joint_wan_gr00t.yaml`, the depth the readout probe measured), so **blocks 11–29
  are computed and thrown away — 19 of 30 layers, ~63 % of DiT weight and compute.** Worth more than
  the text tower and it cuts latency too. (The smoke script's `[15, 22]` is the backbone default, not
  what T-16 trains.)

> Every number above except the two bolded measurements is arithmetic. Step 1 measures it on your
> card. Trust the measurement.

---

## 0. Prerequisites

The 5090 is Blackwell, compute capability **sm_120**. A PyTorch wheel built for older
architectures installs cleanly and then fails at the first kernel launch with
`no kernel image is available for execution on the device`. Use a CUDA 12.8+ build:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e '.[dev]'
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# expect (12, 0) — anything else and the wheel does not match the card
```

Confirm the repo is healthy before trusting any GPU number from it:

```bash
python -m pytest -q          # 666 tests, all CPU
```

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
    --backbone-source /path/to/Wan2.2-TI2V-5B \
    --device cuda
```

**`--backbone-source` is not optional for a checkpoint trained on Discoverer+**, which is every
checkpoint this runbook is about. The frozen weight location is deliberately kept out of the
committed config so `config_hash` matches across machines training the identical model (AC-04) — the
path is folded in at run time instead, so the file records `/valhalla/projects/...`. Without the
override, loading here goes looking for weights on a filesystem this box does not have. The flag
also exists on `rollout.py` and `serve_policy.py` for the same reason.

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
| level | **L0** beats-doing-nothing | **below L0** | **L0** beats-doing-nothing |
| score | 28.6/100 | 19.9/100 | 48.4/100 |
| `skill_vs_repeat_pct` | −20.9 % | −129.0 % | **−32.4 %** |

The bar is still unbeaten: the highest score in that table is the run that loses hardest to inertia
after the tiny one. Read the level. `docs/benchmark.md` has the full column and the diagnosis.

### 3b. T-29 — re-score with the frames training actually used

**Do this before treating the numbers above as settled.** Every one of them was produced with
`predict()` tiling a single camera frame to the backbone's 9-frame context, while training fed the
real 9-frame window ending at the chunk (`docs/improvements.md` I-7). A video backbone trained on a
moving clip was graded on a freeze-frame — and repeat-last-action, the baseline it loses to, is
nothing but motion continuity.

`--frame-history` feeds the window `EpisodeDataset` selected, via the same
`frame_window_indices`. No retraining; the checkpoint is untouched and only the input changes.

```bash
# A: how everything before 2026-07-30 was measured
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

`--compare` refuses two runs whose holdouts differ, so the columns always mean the same thing.

Because scoring only reads `predictions.jsonl`, a **new rung costs no retrain** — every past run is
re-scorable.

---

## 4. Run the closed loop

Offline metrics filter candidates; they do not tell you whether the thing runs in a loop. Two ways
in, both using the checkpoint directly:

```bash
# In-process, against the mock robot: safety layer, watchdog, receding horizon
python scripts/rollout.py --robot mock --policy joint \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors \
    --backbone-source /path/to/Wan2.2-TI2V-5B \
    --policy-device cuda --rollouts 5

# MuJoCo G1 + Dex3 with rendered pixels (docs/sim.md)
python scripts/rollout.py --robot mujoco_g1 --policy joint \
    --checkpoint <same> --backbone-source <same-weights> \
    --policy-device cuda --policy-camera head --image-hw 120 160
```

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

Watch the `min_policy_rate_hz` line. One forward pass per cycle is the whole latency budget, so if
the loop misses its deadline the fix is the two levers from the top of this page (truncate blocks
23–29, evict the text tower), not a smaller batch.

---

## 5. What this box cannot do

| | why |
|---|---|
| The T-16 fine-tune of record | Discoverer+: resume harness, dataset, accounted hours |
| Task success / real safety | E3 — needs the G1 |
| Optimism-bias scoring | needs failure demonstrations; our data is success-only |
| Video-fidelity metrics | needs stored predicted frames (nothing writes them yet) |

---

## Troubleshooting

**`no kernel image is available for execution on the device`** — the wheel predates sm_120.
Reinstall from the cu128 index (§0).

**OOM while loading, then fine afterwards** — the text tower (~11 GB) and the DiT (~10 GB) are both
resident during the first `condition_text` call. Precompute the instruction embedding before the DiT
is built, or move the tower to CPU after it (`src/wam/backbones/wan_i2v.py:401` already has the
machinery).

**`gripper: expected 1 values, got 2`** — a `StateMLPConfig` left at the default `gripper_dims=1`
against a real G1 episode, which has one gripper value per hand. Derive both dims from the state.

**Weights not found under `/valhalla/projects/...`** on a machine that has no such path — the
checkpoint recorded where its frozen base sat *on the cluster*. Pass `--backbone-source` (§2). Only
the location is substituted; the config's `config_hash` is untouched, because where a run happened
was never part of what was trained.

**`REFUSING TO SCORE`** — see §2. This is the guard working; do not paper over it.

**`holdout mismatch — not comparable`** from `run_bench.py --compare` — the two runs were scored on
different episode sets. Re-run `eval_t16.py` for one of them against the other's holdout.

---

## See also

- `docs/benchmark.md` — the ladder, its KPIs and the external benchmark landscape
- `cluster/discoverer/README.md` — where the fine-tune actually runs
- `docs/sim.md` — what the MuJoCo loop proves and what it does not
- `docs/discoverer.md` — cluster facts, quotas, and the login-node rules
