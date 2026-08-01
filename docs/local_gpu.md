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

### 3c. T-30 — re-score through the action branch that was trained but never read

The joint model trains **two** action readouts and deploys one. `ActionHead` regresses the whole
chunk in one shot from the pooled features; a rectified-flow branch (`velocity_head` +
`action_recon`, `weights.action_flow = 1.0`) models the same chunk as a distribution and is never
touched at inference. A single-shot L2 regressor under a one-to-many conditional is mean-seeking,
and `t16-lora-seed0` is **consistent with** one: chunk RMS **0.00226** against the demonstrations'
**0.00404**, `smoothness_ratio` **0.29**. `--flow-sampler` reads the chunk out of the flow branch
instead. No retraining; the checkpoint is untouched and only the decode step changes
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
| this run | 1.21e-05 | the deployed regression head |
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

Both arms carry `--frame-history` (or neither): run this **after** T-29 reports, inside whichever
frame mode it leaves standing, or the two A/Bs collapse into a 2×2 nobody pre-registered. The
readout goes into `bench.json`'s `run_name` as `…+flow32s0k8`, for the same reason the frame mode
does, and **one `--out` holds exactly one readout** — the script refuses to write a second arm's
artifacts over a first one's, because every filename is fixed and `--out` defaults to `--run-dir`.
`--flow-steps` and its companions without `--flow-sampler` are refused rather than silently
scoring the regression head into a flow-named output dir. Sweep `--flow-steps 1 4 16 32 64` as
separate runs to show the verdict does not hinge on the integrator — **n=1 is the control**: one
Euler step from noise is what "rectified flow needs one step" would deploy, and it measures how
far from straight this field actually is. On Discoverer+ the whole sweep plus the verdict is one
job: `sbatch cluster/discoverer/63_eval_t30_flow_head.sbatch`.

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

**Decision rule `T30_RULE_V1`, fixed before the run** — keyed on `skill_vs_repeat_pct` of the
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

Watch the `min_policy_rate_hz` line. One backbone pass per cycle is essentially the whole latency
budget — `policy_deadline_ms` is 500 ms and `min_policy_rate_hz` is 2 Hz (`ExecutorConfig`), against
a measured 79 ms per window for the Wan pass — so if the loop misses its deadline the fix is the two
levers from the top of this page (truncate blocks 23–29, evict the text tower), not a smaller batch.

`--flow-sampler` (§3c) is not wired into `rollout.py` yet, deliberately: it is an offline
question until the A/B answers it. When it is, it adds *n* MLP evaluations per cycle — at T-16's
dimensions one step is roughly 13 M MAC at batch 1, so 32 steps is arithmetic noise next to one
79 ms backbone pass, and `timing.json` from the two eval arms is the measured version of that
claim. The variant that would blow the budget is the one this design rejects: re-running the
backbone at every sampler timestep, where 79 ms/step leaves room for **n ≤ 5** inside 500 ms.

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
