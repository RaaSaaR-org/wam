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
| VAE (encoder path only — `decode_video` is unused here) | ~1.4 GB |
| LoRA adapters + action branch | <0.1 GB |
| activations, 60 tokens, batch 1 | negligible |
| **total** | **~12 GB** |

The umT5 text tower is ~11 GB and is **not** in that budget: `condition_text` is cached
(`src/wam/backbones/wan_i2v.py:459`) and one task has one instruction, so it is called once. Peak
during that first call is real, though — see the ordering note under *Troubleshooting*.

**Two levers.** Evicting the text tower already exists: `adapter.offload("text_encoder")`, exposed
as `--offload-text` on the smoke script — but not yet wired into `eval_t16.py`, `rollout.py` or
`serve_policy.py`. Truncating the DiT is not built: readouts are at blocks `[15, 22]` of 30, so
blocks 23–29 are computed and discarded, worth ~23 % of weight and compute.

> The table above is arithmetic, not a measurement. Step 1 measures it. Trust the measurement.

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
    --device cuda
```

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

Reference points on the identical 40-episode holdout:

| | `d1-full-gen-seed0` (action-only) | `t18-real-ablation-seed0` (world-action) |
|---|---|---|
| level | **L0** beats-doing-nothing | **below L0** |
| score | 28.6/100 | 19.9/100 |
| `skill_vs_repeat_pct` | −20.9 % | −129.0 % |

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
    --policy-device cuda --rollouts 5

# MuJoCo G1 + Dex3 with rendered pixels (docs/sim.md)
python scripts/rollout.py --robot mujoco_g1 --policy joint \
    --checkpoint <same> --policy-device cuda --policy-camera head --image-hw 120 160
```

Or serve it and drive from elsewhere on the network — this is how the Mac can sit in the loop
without holding the weights:

```bash
python scripts/serve_policy.py --joint --checkpoint <same> --device cuda   # on the 5090
python scripts/rollout.py --policy remote --server-uri ws://<5090-host>:8765      # anywhere
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

**`REFUSING TO SCORE`** — see §2. This is the guard working; do not paper over it.

**`holdout mismatch — not comparable`** from `run_bench.py --compare` — the two runs were scored on
different episode sets. Re-run `eval_t16.py` for one of them against the other's holdout.

---

## See also

- `docs/benchmark.md` — the ladder, its KPIs and the external benchmark landscape
- `cluster/discoverer/README.md` — where the fine-tune actually runs
- `docs/sim.md` — what the MuJoCo loop proves and what it does not
- `docs/discoverer.md` — cluster facts, quotas, and the login-node rules
