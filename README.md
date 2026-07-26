# WAM — Modular World Action Model

A modular World-Action-System: one shared model predicts short visual futures and matching robot actions, executed as a safe closed loop on humanoid hardware (target: Unitree G1).

**Vision → world prediction → action → safe closed loop.**

- PRD: internal document, kept outside this repository (v0.1, 2026-07-25)
- Working notes for agents: `CLAUDE.md`
- MVP task breakdown: `TASKS.md`

## Architecture (runtime)

```
cameras + robot state + instruction
        │
   [frozen VAE / text encoders]  [state encoder]
        │                             │
        └──────► video backbone (FLUX 3 | open fallback) ◄──────┘
                        │
              action latents ──► action decoder ──► action chunk
                        │
                 SAFETY LAYER (deterministic, non-learned)
                        │
              low-level controller (≥100 Hz) ──► robot
                        │
              new observation ──► re-plan (receding horizon)
```

## Repo layout

```
src/wam/        Python package (interfaces, backbones, encoders, decoders,
                safety, robot, data, training, runtime, evaluation)
configs/        versioned robot / model / training configs
scripts/        entry points (record, train, replay, rollout)
tests/          E0 unit tests
docs/           specs (schema, safety interface, ...)
idea/           PRD
```

## Status

M0–M4 code-complete (mock/sim verified, 445 tests). Pending: hardware (G1 + cameras + teleop),
real datasets D1/D2, open decisions OD-01/02/03/04/06/07. Tasks: `TASKS.md` ·
path to real usage: `docs/ROADMAP.md`.

Key entry points:

```
scripts/run_mock_loop.py        M0: dummy policy + safety + watchdog closed loop
scripts/record_mock_dataset.py  M1: synthetic dataset + validation gates
scripts/overfit_d1.py           M2: D1 overfit gate + E1 eval
scripts/serve_policy.py         M4: WebSocket inference server
scripts/rollout.py              M4: E2/E3 rollouts (mock | g1)
scripts/run_acceptance.py       M4: AC-01…07 acceptance report
```
