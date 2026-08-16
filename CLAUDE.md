# CLAUDE.md — WAM (World Action Model)

Modular World-Action-System: predicts short visual futures **and** robot actions jointly. MVP = safe, repeatable pick-and-place on Unitree G1 (or compatible). Full spec: internal PRD (v0.1, 2026-07-25), kept outside this repository.

## Core decisions (from PRD)

- **No foundation pretraining.** Adopt a video backbone (FLUX 3 Dev preferred, open I2V model as fallback), add robotics adapters, fine-tune on own demonstrations.
- **Backbone is swappable.** All backbones implement the same adapter interface (FR-09). Never hard-wire FLUX 3.
- **Safety is deterministic.** The learned model never outputs motor currents. Every action passes the Safety Layer (limits, collision, watchdog) before the low-level controller (FR-07).
- **Closed loop.** Short action chunks (0.5–2.0 s), execute a prefix, re-observe, re-plan (FR-05).
- **Canonical schema.** Training and policy use a canonical state/action space; robot-specific mapping lives only in the Robot Adapter (FR-06).
- **Data quality gates.** Timestamp sync, calibration and dataset validation are release gates, not nice-to-haves.

## Layout

```
src/wam/
  interfaces/   versioned core contracts (schema, protocols) — change with care
  backbones/    FLUX 3 + open fallback behind one adapter interface
  encoders/     state encoder (trainable), action encoder (training only), frozen text/VAE wrappers
  decoders/     action decoder → normalized joint deltas / EE deltas + gripper
  safety/       deterministic limits, projection, watchdog — no ML in here
  robot/        HAL: canonical schema ↔ robot API (g1, mock)
  data/         episode format (PRD Anhang A), recording, replay, validation
  training/     losses (video/action/alignment), trainers
  runtime/      closed-loop executor, inference server
  evaluation/   E0 unit → E1 replay → E2 sim → E3 real robot → E4 generalization
configs/        versioned robot/model/training configs
subprojects/    the two halves (2026-08-15) — see subprojects/README.md
  edge-wam/     image in, action out, on the robot (Cosmos3-Edge 4B) — tasks E-NN
  data-factory/ more/better training data from real episodes (Super/Nano) — tasks D-NN
```

`src/wam/` is **shared and unforked** by both sub-projects — the canonical schema, the safety layer
and the robot HAL are contracts, and two drifting copies is the expensive failure.

## Conventions

- Python, PyTorch. Tasks are one file each under `.mc/tasks/{todo,done}/` (MissionControl format,
  IDs stay `T-NN`; see `.mc/README.md`). `TASKS.md` is the milestone index over them — milestones
  M0–M4 map to the PRD roadmap. `mc task next` gives the next actionable task, `mc show T-16` the
  full record. Edit the task file, not the index; run `mc index` afterwards.
- **Three task namespaces since 2026-08-15.** `T-NN` (root, `.mc/tasks/`, driven by `mc`), `E-NN`
  (`subprojects/edge-wam/tasks/`) and `D-NN` (`subprojects/data-factory/tasks/`). The sub-project
  files use the same frontmatter shape plus a `subproject:` field and a `## Notes / Report` section
  carrying the result. **`mc task next` only sees `T-NN`** — each sub-project's `TASKS.md` is its
  own hand-maintained index, which is the accepted cost of the split. Existing `T-040`/`T-041`/
  `T-042` were deliberately **not** migrated: their pre-registrations, sbatch files and commit
  subjects all cite them where they are.
- **T-39 has reported, and the answer was `VOID (labels)`** (2026-08-16,
  `docs/preregistration/PR-07-RESULT.md`). The positive control asked whether this corpus's own
  action column clears L1 under our scorer. **It does not — −359.41 pp, while `oracle_state` scored
  a bit-exact `mse 0.0`**, so the failure is our label space, not our plumbing. Therefore no policy
  trained on these labels can clear the bar either, and both sub-projects share that corpus.
  **This is not a discharged gate. Do not read VOID as permission to train** — the rule it replaces
  said "no training run before T-39 reports", and what T-39 reported was that its own premise fails.
  Whether training may start, and against which label space, is the project owner's call. It
  licenses a defect report against the label pipeline and **forbids any statement about GR00T**
  (the policy arm never ran); PR-07 §7's second model is licensed only on outcome N, not on VOID.
- Every rollout must be traceable to checkpoint + dataset snapshot + config hash (AC-04).
- Milestone order is strict: overfit a small task first (D1), only then scale (P6).
