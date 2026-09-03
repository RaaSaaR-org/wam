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
- **T-39 first reported `VOID (labels)` (2026-08-16), PR-13 withdrew that verdict's premise by
  measurement, and T-39 then RE-REPORTED `VERDICT N` on 2026-08-17** under `T39_RULE_V2`, job
  188408 — `docs/preregistration/PR-07-V2-RESULT.md`. **`T40_RULE_V3` §5.3 registers that `N`
  satisfies "T-39 has reported" while `VOID` does not**, and that is what closed PR-08 §8 item 7.
  The record below is the route to that re-report and stands exactly as written.
  The positive control asked whether this corpus's own action column
  clears L1 under our scorer, and got **−359.41 pp** (`docs/preregistration/PR-07-RESULT.md`).
  **That number was produced by a defective instrument, not by the labels.**
  `commanded_to_chunk` built the chunk's step 0 as `command − STATE` while every other step of both
  arms is a homogeneous first difference; that one element carried **~90 % of the summed per-step
  MSE and 143× its neighbours** (PR-12, verdict `C`). Anchor it on the previous command instead and
  the same column scores **+68.10 % on L1 and +75.40 % on L2 across T-39's own 40-episode holdout**,
  reaching **L4** — PR-13, verdict `W`, `docs/preregistration/PR-13-RESULT.md`, with the unmodified
  bridge reproducing `−359.41` to `+0.002 pp` in the same run. **So "no policy trained on these
  labels can clear the bar" is withdrawn: it was a statement about our evaluation adapter.** The
  blast radius is that adapter and the four sweeps importing it — **zero hits in `src/wam/`, zero in
  the training path**, and `relabel_chunks` is homogeneous at every step, so the corpus on disk was
  always clean and **the fourteen negatives in `docs/benchmark.md` stand.**
  **THE GATE ITSELF IS UNCHANGED, AND CORRECTING A CLAIM IS NOT LIFTING ONE. This is not a
  discharged gate. Do not read `C` or `W` as permission to train** — the rule it replaces said "no
  training run before T-39 reports", T-39 reported, and its premise has now been withdrawn rather
  than satisfied. **Whether training may start, and against which label space, is the project
  owner's call**, and no session may edit this file to lift it. PR-07 §6 still **forbids any
  statement about GR00T** (that clause was written when the V1 policy arm had never run — job
  187813 died at 108 s; the V2 re-report's arm did run, and its figures live in
  `PR-07-V2-RESULT.md`, not here); PR-07 §7's second
  model is licensed only on outcome N. What the corrected instrument means for `docs/benchmark.md`'s
  L4 gate is a separate open decision — the repaired cell is L4 under spec 0.1.0 and below spec
  0.2.0's two-sided floor, so the two specs disagree about it.
- **T-040 generation is under way, and it licenses nothing about training.** PR-08 §8's
  seven-item conjunctive gate closed **7/7 on 2026-09-01**; `PARTITION_CEILING_GPU_H = 2013.75`
  (train share `805.50`) was signed the same day; and **one chunk** — `STAGE=1 STYLE_SET=train
  CHUNK_INDEX=1 CHUNK_TOTAL=4`, one quarter of stage 1's train set — was released by the project
  owner on 2026-09-02, `docs/preregistration/PR-08-DET-2026-09-02-the-first-real-chunk-released.md`.
  **The other three chunks, stage 2, the eval set and the identity set are NOT released**; the
  ceiling stays a cap over the whole partition. Everything the run writes carries a
  `NOT_TRAINING_DATA` marker, and **clearing that marker is the owner's decision, not a session's.**
- Every rollout must be traceable to checkpoint + dataset snapshot + config hash (AC-04).
- Milestone order is strict: overfit a small task first (D1), only then scale (P6).
