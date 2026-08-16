---
id: E-03
subproject: edge-wam
title: "Establish what edge hardware we actually have, and whether the published rate reproduces"
slug: establish-the-target-edge-hardware-and-reproduce-the-rate
status: todo
priority: 2
owner: ''
tags:
- edge
- hardware
- benchmark
depends_on: []
created: 2026-08-15
updated: 2026-08-16
status_note: "Cannot close: AC-1 is a purchasing fact only the user can answer. Established 2026-08-16 that AGX Orin is absent from the policy variant's test hardware and that the local RTX 5090 fits inference + LoRA (9.14 GB weights, measured)."
---

# Establish the target edge hardware, and whether the published rate reproduces

## Description

The whole sub-project is justified by a number we have not measured, on hardware we may not own.

**The published figures disagree with each other.** NVIDIA's blog and the launch coverage say
**32 actions per inference at 15 Hz on Jetson Thor** at 640×360 [doc]. The `Cosmos3-Edge` model card
instead shows **action chunk size 16** and **20 fps** in its example configs [✓]. Both cannot
describe the same setup, and neither has been reproduced here.

**Step 0 is not a benchmark.** It is: *do we have a Jetson?* Nothing in this repo mentions one —
`cluster/discoverer/` is H200s and the workstation is a single RTX 5090 with 93 GB host RAM. If the
answer is no, the deployment target is a decision, not a fact, and this task's real output is that
decision plus its cost.

Note the 5090 is a *poor* proxy: 32 GB and desktop power envelope tell you little about a Jetson's
memory bandwidth or thermal budget. A 5090 measurement bounds the model's compute, not its
deployability.

## Acceptance

1. **The hardware question answered by the user**, recorded here: which Jetson (Thor / AGX Orin /
   none), or which alternative target.
2. If a device exists: 4B policy loaded, and actions-per-inference, control rate, resolution and
   power measured, against the two published claims, with the discrepancy resolved or recorded.
3. If no device exists: the options costed — buy a Thor, target the 5090, or run the policy off-board
   over the network (and what that does to the closed loop's 0.5–2.0 s chunk budget and to the
   safety layer's watchdog).
4. Whichever branch: a stated **latency budget** for the closed loop, since FR-05's re-observe /
   re-plan cycle is what this model has to fit inside.

## Notes / Report

**2026-08-16 — E-03 CANNOT be closed, and the reason is not research: AC-1 is a purchasing
fact.** What desk work could settle is settled; one finding lands against the sub-project's
stated premise and should be read before E-05 is written.

**(a) There is no Jetson in this project, and no robot.** `docs/ROADMAP.md:162` still reads
"order the G1 EDU4 (+ VR headset) … as soon as it arrives". So AC-2 (load the 4B on the device
and measure) is unrunnable and AC-3 (cost the options) is the live branch.

**(b) The premise "the world modelling stays, on-device, at 15 Hz" is at risk [doc].**
`README.md:25` asserts it. The model cards confirm 15 Hz on exactly **one** part: Jetson AGX
Thor T5000, 128 GB, MAXN, RTF 1.40. The next SKU down (T4000, 64 GB) **misses the 15 Hz budget
by 3.5 %**. And two things cut against an Orin:
  - **`Jetson AGX Orin` is absent from the *policy* variant's Test Hardware line entirely**
    [OK, both cards fetched 2026-08-16]. The base card lists it; `Cosmos3-Edge-Policy-DROID`
    lists B200, H100, H20, RTX PRO 6000, DGX Station, DGX Spark, **Jetson Thor** — no Orin.
    No Cosmos3 action-path number exists for any Orin, on either card.
  - Every reseller spec sheet for the G1 EDU tiers names **Jetson Orin NX 16 GB / 100 TOPS**
    [doc] — Ampere, one generation behind Thor, with 16 GB **unified** memory shared with
    Ubuntu, the Unitree stack, RealSense and DDS, against **9.14 GB of bf16 weights**.

  If the G1 EDU4 ships with Orin NX, "on-device at 15 Hz" is not merely unmeasured — it is
  contradicted by the nearest published data point. **Unitree's own G1 page does not commit to
  a variant:** it sells the NVIDIA module as a configurable accessory ("Orin, etc."), and
  "EDU4" is not Unitree's naming at all — it is reseller tier U4. So the repo's "the G1 EDU4's
  onboard computer is a Jetson Orin" (`docker/dds/README.md:20`) is right about arm64 and
  silent about the thing that decides this task.

**(c) The local RTX 5090 is a real target, not a poor proxy.** Measured from the safetensors
**headers** (HTTP Range, no weights downloaded), `Cosmos3-Edge-Policy-DROID` resident weights:

| component | params | bytes |
|---|---|---|
| transformer (2 shards) | 3 369 657 024 | 6 748 759 424 |
| vision_encoder | 489 342 704 | 978 685 408 |
| vae | 704 688 668 | 1 409 377 336 |
| **total** | **4 563 688 396** | **9 136 822 168 = 9.14 GB** |

Cross-check closes to the byte against the index's `metadata.total_size`. Against 32 607 MiB:
**inference fits comfortably (~12–14 GB estimated peak) and LoRA post-training fits; a full
fine-tune does not** and stays on Discoverer+. NVIDIA publishes a PyTorch Policy-DROID figure
for the RTX PRO 6000 Blackwell — same GB202/sm_120 family as this card — at 1.65 s E2E. Note
**only BF16 is tested** (FP4/FP8/FP16 explicitly unsupported), so the 5090's fp8/fp4 tensor
cores buy nothing: there is no quantization lever.

**(d) The rate conflict, resolved.** The model card's own tables use action chunk 16 and
**denoising steps, not chunk size, dominate on-device latency** — 6.32 s at 30 steps vs 1.528 s
at 4 UniPC steps, a ~4× move. The blog's "32 actions at 15 Hz" and the card's "16 / 20 fps" are
not the same measurement and neither is reproduced by us.

**What would close this task**
1. **AC-1, the blocker, needs a human answer: which G1 SKU is on the order, and which compute
   option?** No amount of research settles it.
2. AC-3 is uncosted. Three live options: (a) buy a Jetson AGX Thor T5000 128 GB — the only part
   measured at 15 Hz; (b) target the 5090 and accept a tethered robot; (c) run the policy off-
   board over the existing `serve_policy.py` / `--policy remote` path already in this repo.
3. AC-4 has no latency budget. `ExecutorConfig` today is `policy_deadline_ms=500`,
   `min_policy_rate_hz=2`; the README promises 15 Hz. **That is a 7.5× gap between what this
   repo executes and what this sub-project advertises**, and it should be reconciled in E-05
   rather than discovered at bring-up.
4. Cheap and now clearly feasible: measure `Cosmos3-Edge-Policy-DROID` E2E latency and peak
   VRAM on this 5090. It would be the first Cosmos3 action number this project owns.
