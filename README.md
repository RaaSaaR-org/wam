# WAM — Modular World Action Model

One shared model predicts short visual futures **and** the matching robot actions, executed as a
safe closed loop on humanoid hardware (target: Unitree G1 EDU4 + Dex3-1 hands).

**Vision → world prediction → action → safe closed loop.**

- PRD: internal document, kept outside this repository (v0.1, 2026-07-25)
- Working notes for agents: `CLAUDE.md` · MVP task breakdown: `TASKS.md`
- Ordered path to real usage: `docs/ROADMAP.md` · How it fits together: `docs/architecture.md`

## Architecture (runtime)

```
cameras + robot state + instruction
        │
   [frozen VAE / text encoder]   [state encoder]
        │                             │
        └──────► video backbone (Wan2.2-TI2V-5B + LoRA) ◄──────┘
                        │
              action latents ──► action decoder ──► action chunk
                        │
                 SAFETY LAYER (deterministic, non-learned)
                        │
              low-level controller (≥100 Hz) ──► robot
                        │
              new observation ──► re-plan (receding horizon)
```

The backbone is swappable by contract (FR-09): anything satisfying `BackboneAdapter` — plus
`FlowBackbone` for joint training — drops in. Wan2.2-TI2V-5B (Apache 2.0) is the decided MVP
backbone; FLUX 3 Dev is deferred to M5. The safety layer is never learned and never bypassed.

## Status (2026-07-29)

**858 tests green.** M0–M4 are code-complete. What that does and does not mean:

| Proven | How |
|--------|-----|
| Action-only policy trains and generalizes on real robot data | 402 real G1 episodes, 40 unseen: E1 mse 1.10e-5 vs. 1.63e-5 zero-delta baseline (−32 %) |
| The Wan adapter works on real 5B weights | ZeroGPU, 13/13 checks, readout blocks label-validated to (2, 10) |
| The closed loop survives contact physics | MuJoCo G1 + Dex3, rendered pixels, 1.19× realtime (`docs/sim.md`) |
| The DDS wire layer is real | `DdsG1Transport` vs. a fake G1 on a CycloneDDS bus, arm64 container, 11/11 (`docker/dds/README.md`) |
| T-16 can run on a cluster | LoRA path + resume harness, all CPU-testable (`cluster/discoverer/README.md`) |
| Frozen video features really do lose — it is not a pooling artefact | Spatial readouts vs. a same-width random control: geometry adds nothing (T-26, `docs/hf_jobs.md`) |

| Not proven | Why it matters |
|------------|----------------|
| **That the action-only policy learned anything a heuristic cannot do** | A causal repeat-last-action baseline scores mse 9.14e-6 — 17 % *better* than the trained model. The −32 % above is real, but it is the demonstration's own inertia. WAM-Bench puts the run at **L0, 28.6/100** (T-27, `docs/benchmark.md`). |
| **"Video helps"** (AC-07) | Frozen features from *both* Wan and Cosmos3 lose to a state-only ridge — including with the spatial readout that could have explained it away (T-26) — and at tiny scale the video branch *hurts*. The T-16 LoRA fine-tune has now run and is **negative** — WAM-Bench L0, `skill_vs_repeat_pct` −32.4 %, losing to repeat-last-action. Three confounds under that verdict are staged and unrun (T-29/T-30/T-32), so it is "negative, measured out of distribution", not "no". |
| Anything on a physical robot | No G1 yet. Vendor conformance, E-stop chain, real limits and the Dex3 mapping are asserted, not measured. |
| Real teleop data (D1/D2) | Everything so far is synthetic or converted from `nvidia/GR00T-N1.7-AppleToPlate`. |

Open decisions: **OD-06** (FLUX 3 access — deferred to M5) and **OD-08** (which safety functions
the vendor controller covers — verify at bring-up). Everything else is decided; see `TASKS.md`.

## The one design idea worth knowing

There is exactly **one** hardware seam, `G1Transport`, with three implementations —
`FakeG1Transport` (unit tests), `MujocoG1Transport` (physics + pixels), `DdsG1Transport` (real
vendor DDS). Above that seam the code is byte-identical in all three cases. There is deliberately
no `MujocoAdapter`: the sim buys coverage of the *production* adapter, not of a sim-specific one.

The same principle runs through the model side. `JointWorldActionModel` depends on the
`FlowBackbone` protocol, never on a concrete backbone, and `WanFlowBackbone` keeps the 5B DiT, the
VAE and the text tower **out of the module tree** — only LoRA parameters are registered. So
"checkpoint the adapter, not the 5B model" is structurally enforced rather than a flag someone has
to remember.

## Repo layout

```
src/wam/        interfaces, backbones, encoders, decoders, safety, robot,
                data, training, runtime, evaluation (each has its own README)
configs/        versioned robot / model / training / sim configs
scripts/        entry points (record, train, probe, replay, rollout)
cluster/        HPC job scripts — discoverer/ = EuroHPC H200, sbatch-ready
docker/dds/     arm64 DDS conformance harness (no robot required)
deploy/         ZeroGPU Spaces for the free-tier Wan / Cosmos3 probes
tests/          E0 unit tests
docs/           runbooks and specs
datasets/       converted episodes (WAM format)  ·  runs/ experiment artifacts
```

## Entry points

```
scripts/run_mock_loop.py        M0: dummy policy + safety + watchdog closed loop
scripts/record_mock_dataset.py  M1: synthetic dataset + validation gates
scripts/convert_lerobot_g1.py   M1: LeRobot/GR00T G1 episodes -> WAM format
scripts/overfit_d1.py           M2: D1 overfit gate + E1 eval
scripts/run_ablation.py         M3: world-action vs. action-only (AC-07)
scripts/run_bench.py            M3: WAM-Bench ladder on archived predictions (T-27)
scripts/train_t16_lora.py       M3: the Wan LoRA fine-tune (resumable, 4 h chunks)
scripts/eval_t16.py             M3: score a T-16 checkpoint on a PROVEN holdout (T-28)
scripts/fetch_g1_model.py       E2: pull the MuJoCo G1 assets (~38 MB)
scripts/rollout.py              M4: E2/E3 rollouts (mock | g1 | mujoco_g1)
scripts/view_sim.sh             E2: watch the MuJoCo closed loop live (wraps mjpython)
scripts/serve_policy.py         M4: WebSocket inference server
scripts/run_acceptance.py       M4: AC-01…07 acceptance report
```

## Docs

| File | What |
|------|------|
| `docs/ROADMAP.md` | Ordered path to real usage — read this first |
| `docs/benchmark.md` | WAM-Bench: the offline ladder, its KPIs, and the external benchmark landscape |
| `docs/local_gpu.md` | Single consumer GPU: run, test and benchmark a checkpoint (no fine-tune) |
| `docs/discoverer.md` | EuroHPC H200 cluster: machine facts, quotas, billing, gotchas |
| `cluster/discoverer/README.md` | The same cluster as a runbook — eleven `sbatch` files in execution order |
| `docs/sim.md` | MuJoCo scene, what it proves and what it does not |
| `docker/dds/README.md` | DDS conformance + the ordered hardware bring-up checklist |
| `docs/hf_jobs.md` | Free-tier GPU work: ZeroGPU Spaces and HF Jobs |
| `docs/teleop.md` | Teleop + calibration workflow |

## Next step

T-16 has run and lost to repeat-last-action. Three confounds under that verdict are staged on
Discoverer+ (`cluster/discoverer/README.md`), in this order, and none of them needs a robot:

1. `61_eval_t29_frame_history.sbatch` — T-29. Training fed the real `num_frames` window; `predict()`
   fed one frame tiled. A backbone trained on a moving clip was graded on a freeze-frame.
2. `63_eval_t30_flow_head.sbatch` — T-30. We train two action readouts and deploy the cheaper one.
   The action *latent* reconstructs the holdout 15× better than the readout that was scored.
3. `55_train_i8_rung.sbatch` ×3, then `62_eval_i8_curve.sbatch` — T-32. "Not enough data" has
   explained every negative in this project and has never been tested.

The bar any of them has to clear is `skill_vs_repeat_pct > 0` on WAM-Bench, not "beats the
action-only baseline" — that baseline itself loses to a one-line heuristic.
