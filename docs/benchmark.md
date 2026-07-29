# WAM-Bench — the offline ladder (T-27)

We have 402 real G1 episodes, no robot, and no sim for this task. Everything measurable today
therefore has to come out of held-out demonstrations. WAM-Bench is that layer: a fixed ladder of
rungs scored from an archived `predictions.jsonl`, no GPU and no hardware involved.

```
.venv/bin/python scripts/run_bench.py runs/d1-full-gen-seed0
.venv/bin/python scripts/run_bench.py runs/a runs/b --compare
```

Writes `bench.json` + `bench.md` into each run directory. Because it consumes only stored
predictions, **every past run can be re-scored when the metric set changes** — a new rung does not
cost a retrain.

---

## Where it sits in the E-ladder

`E0 unit → E1 offline replay → E2 sim → E3 real robot → E4 generalization`

E3 and E4 need hardware we do not have. E2 needs a sim of this task, which we also do not have.
So the whole real-data question lives inside E1, and E1's headline was a single number: action
MSE on the holdout. WAM-Bench subdivides that one rung into five, all computable on the same data.

---

## Why raw MSE was not enough

Two published results drive the design.

**Task-critical restriction beats whole-trajectory averaging.** Restricting the same action error
to task-critical intervals lifts the rank correlation with real rollout success from Spearman
≈ −0.61 to ≈ −0.87 ([CI-MSE, arXiv:2606.29898](https://arxiv.org/html/2606.29898v1)). Most
timesteps in a demonstration are quiet, so an average over all of them largely measures how well
a policy holds still.

**Fidelity is not controllability.** A world model can produce plausible video while ignoring the
commanded action ([MiraBench, arXiv:2605.29360](https://arxiv.org/pdf/2605.29360)). The two must
be scored on separate axes, or "looks right" gets mistaken for "responds right".

And the field itself is nowhere near saturation, which sets expectations: on
[RoboDojo](https://robodojo-benchmark.com/) — 42 sim + 18 real manipulation tasks — the best of 30
evaluated policies reaches **8.8% average success against 76.0% for human experts**; on the real
leaderboard π0.5 gets 12.8%. A benchmark whose top score is ~12% of human is measuring the right
thing. One where everyone passes is not.

---

## The ladder

Five rungs, each a yes/no gate plus continuous points (20 each, 100 total). The reported **level**
is the highest *contiguous* rung passed — clearing L3 while failing L1 does not earn L3, because
the rungs are premises for each other. All thresholds are module constants in
`src/wam/evaluation/benchmark.py`, fixed before any run is scored.

| rung | question | metric | gate |
|---|---|---|---|
| **L0** beats-doing-nothing | Better than holding still? | `skill_vs_zero_pct` | > 0% |
| **L1** beats-inertia | Better than repeating the last action? | `skill_vs_repeat_pct` | > 0% |
| **L2** acts-when-it-counts | Still better where the task actually happens? | `ci_skill_vs_repeat_pct` | > 0% |
| **L3** holds-the-horizon | Does the chunk hold together to its last step? | `horizon_ratio` | ≤ 4.0 |
| **L4** moves-like-a-demo | As smooth as the demonstrations? | `smoothness_ratio` | ≤ 2.0 |

**The two trivial baselines.** *Zero-delta* predicts no motion. *Repeat-last-action* replays the
action executed one control period before the chunk starts — strictly causal, and derived from the
chunk cadence rather than by taking the previous chunk's last step, which would read the future
whenever chunks overlap (the regime FR-05's receding horizon actually runs in).

**Critical intervals.** CI-MSE annotates them with a VLM. We have one task and no annotations, so
the proxy is the demonstrator's own commanded motion energy: the top 20% of chunks by RMS action
magnitude are the moments the task is being performed. Documented as a proxy, not as the paper's
method.

**Diagnostics** (not gated): `timing_gain_pct` — error removed by allowing a ±1-step shift; a large
gain means the shape is right and the phase is wrong, which is a latency bug rather than a capacity
one. Plus `gripper_accuracy` and its dynamic range, and any `warnings`.

---

## First results (2026-07-29)

Both existing real-data runs, identical 40-episode holdout:

| metric | `d1-full-gen-seed0` (action-only) | `t18-real-ablation-seed0` (world-action) |
|---|---|---|
| **level** | **L0** beats-doing-nothing | **none — below L0** |
| **score** | **28.6 / 100** | **19.9 / 100** |
| mse | 1.10439e-05 | 2.09285e-05 |
| ci_mse | 2.30187e-05 | 5.26316e-05 |
| skill_vs_zero_pct | +32.4% | −28.2% |
| skill_vs_repeat_pct | **−20.9%** | **−129.0%** |
| ci_skill_vs_repeat_pct | −7.0% | −144.6% |
| horizon_ratio | 1.66 | 1.02 |
| smoothness_ratio | 2.35 | 5.10 |

Three things the previous dashboard could not say:

1. **The action-only baseline loses to repeat-last-action.** T-14 recorded "E1 mse 1.10e-5 vs
   zero-delta 1.63e-5 (−32%)" as the first generalization result. True — but the repeat baseline
   scores 9.14e-6, **17% better than the trained model**. The −32% is the demonstration's own
   inertia, not learned behaviour. This does not retract T-14's number; it adds the reference that
   makes it readable.
2. **The world-action candidate is worse than predicting no motion at all** (−28.2% skill).
   "Hurts" was already the T-18 verdict at −89.5% relative to the baseline; against an absolute
   floor it is starker, and `smoothness_ratio` 5.10 says the multi-task tax shows up as jerk.
3. **`gripper_accuracy` is not a grasp proxy on this dataset.** The demonstrated gripper signal
   has peak-to-peak range 0.120 and sits on the 0.5 binarization threshold — it never opens or
   closes. The reported 0.87/0.85 is thresholding noise. The bench now emits this as a warning
   instead of letting the number stand.

Point 1 is the load-bearing one: **the bar T-16 has to clear is `skill_vs_repeat_pct > 0`**, not
"beats the action-only baseline". Beating a model that itself loses to a one-line heuristic is not
evidence that video helps.

---

## What is deliberately not in here yet

- **Action-following / controllability** (MiraBench L2, [WorldArena](https://arxiv.org/pdf/2605.00080)):
  condition the video branch on true vs. perturbed actions and measure whether the prediction
  diverges. If it does not, the video is decorative. Needs model access, not just predictions —
  a separate harness. This is the most WAM-specific rung and the natural L5.
- **Optimism bias** (MiraBench L3): requires failure demonstrations. Our data is success-only, so
  it is not computable, not merely unimplemented.
- **Video fidelity** (PSNR/SSIM/LPIPS under action replay): needs stored predicted frames.
- **Closed-loop autoregressive divergence**: roll the policy on its own predictions and measure
  time-to-divergence. Needs the model in the loop.
- **Real success rate** — that is E3, and it needs the robot. Nothing offline substitutes for it;
  the ladder is a filter that stops bad candidates from reaching hardware, not a replacement.

---

## External landscape (July 2026)

| benchmark | what it scores | relevance here |
|---|---|---|
| [RoboDojo](https://arxiv.org/abs/2607.04434) | 42 sim + 18 real manipulation tasks; generalization, memory, precision, long-horizon, open-vocab | The laddered/leaderboard shape this borrows; sets the "best policy ≈ 8.8%" expectation |
| [CI-MSE](https://arxiv.org/html/2606.29898v1) | offline validation that correlates with rollout success | Directly implemented as L2 |
| [MiraBench](https://arxiv.org/pdf/2605.29360) | action-conditioned world-model reliability: physics, action-following, optimism bias | Source of the fidelity≠controllability split; future L5 |
| [WorldModelBench](https://papers.neurips.cc/paper_files/paper/2025/file/4ec03ed08a3fcb59e1c815b5598beff1-Paper-Datasets_and_Benchmarks_Track.pdf) | instruction following, commonsense, physics violations across 7 domains | Video-side scoring if/when we store predicted frames |
| [VLABench](https://github.com/OpenMOSS/VLABench) | large-scale VLA / embodied-agent leaderboard | Comparison point once a policy is worth submitting |
| [RoboTrustBench](https://arxiv.org/pdf/2606.01600) | trustworthiness of video world models for manipulation | Safety-adjacent, maps onto AC-03/AC-06 rather than here |
| ThorArena | humanoids under direct physical human contact | Hardware-gated; relevant at E3 |

---

## Sources

- [RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/abs/2607.04434)
- [Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies](https://arxiv.org/html/2606.29898v1)
- [MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models](https://arxiv.org/pdf/2605.29360)
- [WorldModelBench: Judging Video Generation Models As World Models](https://papers.neurips.cc/paper_files/paper/2025/file/4ec03ed08a3fcb59e1c815b5598beff1-Paper-Datasets_and_Benchmarks_Track.pdf)
- [VLABench](https://github.com/OpenMOSS/VLABench)
- [RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation](https://arxiv.org/pdf/2606.01600)
- [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/pdf/2605.00080)
