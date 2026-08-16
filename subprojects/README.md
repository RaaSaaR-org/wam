# `subprojects/` — the two halves of WAM

**Created 2026-08-15.** WAM was one project that did everything: model the world, and act. Those two
jobs have opposite constraints — one wants the biggest model that fits in a datacenter, the other
wants the smallest model that fits on a robot — so they are now two sub-projects, each with its own
entry point, its own task namespace, and its own Claude session.

| sub-project | question it answers | model class | runs on |
|---|---|---|---|
| **[`edge-wam/`](edge-wam/README.md)** | image in, action out, on the robot | Cosmos3-Edge, 4B | Jetson (target) |
| **[`data-factory/`](data-factory/README.md)** | more and better training data from the data we have | Cosmos3-Super / Nano, 64B / 16B | Discoverer+ |

## What is *not* split

`src/wam/` stays shared and unforked. The canonical state/action schema (`interfaces/`), the
deterministic safety layer (`safety/`) and the robot HAL (`robot/`) are the contracts CLAUDE.md
calls load-bearing; two copies of them that drift is the expensive failure. Both sub-projects import
the same ones.

The root project also keeps its own tasks. `T-NN` in `.mc/tasks/` remains the core work — the
positive control (T-39), the data-scaling curve (T-32), backbone screening (T-37) — and the existing
Cosmos tasks T-040 / T-041 / T-042 were **not** migrated. They are cited from here by ID; their
history, their pre-registrations and the paths in `cluster/discoverer/*.sbatch` all point at where
they are.

## Task IDs

Three namespaces, deliberately separate (decided 2026-08-15):

- `T-NN` — root project, in `.mc/tasks/`, driven by `mc`
- `E-NN` — edge WAM, in `subprojects/edge-wam/tasks/`
- `D-NN` — data factory, in `subprojects/data-factory/tasks/`

**`mc task next` only sees `T-NN`.** That is the accepted cost of the split: each sub-project's
`TASKS.md` is its own index and has to be read directly. Task files use the same frontmatter shape
as `.mc/tasks/` so they can be adopted by `mc` later without a rewrite, plus a `subproject:` field
and a `## Notes / Report` section that carries the result once the task runs.

## The gate that binds both

**T-39 HAS REPORTED, 2026-08-16, and the verdict is `VOID (labels)`** —
`docs/preregistration/PR-07-RESULT.md`, commit `8a4728c`, 1.37 of the 12 GPU-h ceiling.

Its `oracle_action` arm asked whether this corpus's own action column can clear L1 under our scorer,
and answered **−359.41 pp**. By `T39_RULE_V1`'s own ordering — `not L1_action` is checked before any
policy branch — that one clause decided the verdict, and no policy number could move it.

**PR-13 has since withdrawn that clause's premise by measurement — `docs/preregistration/PR-13-RESULT.md`,
verdict `W`, 2026-08-16, zero GPU-hours.** The −359.41 was **our evaluation adapter, not the
labels**. `commanded_to_chunk` built the chunk's step 0 as `command − STATE` while every other step
of both arms is a homogeneous first difference, so a steady-state tracking offset cancelled
everywhere else and survived there at full magnitude: **that one element carried ~90 % of the summed
per-step MSE and 143× its neighbours** (PR-12, verdict `C`). Anchored on the previous command
instead — identical to the current definition under perfect tracking — the same column scores
**+68.10 % L1, +75.40 % L2, level L4** on T-39's own holdout, while the unmodified bridge reproduced
`−359.41` to `+0.002 pp` in the same run.

**The blast radius is the instrument, not the track record.** Zero `commanded_to_chunk` in
`src/wam/`, zero in the training path, and `relabel_chunks` is homogeneous at every step — the corpus
on disk was always clean, **the fourteen negatives stand**, and nothing is retro-validated.

**Read what that does and does not license, because it is easy to get backwards.** It licenses
**correcting** the claim that no policy trained on these labels can clear the bar — that claim is
withdrawn — and a defect report against the evaluation adapter. It still **forbids any statement
about GR00T**: the policy arm never ran. It does not license swapping in another model — PR-07 §7's
second candidate is pre-registered only on outcome **N**.

> **The standing rule was "neither sub-project starts a training run before T-39 reports". T-39 has
> reported, and its premise has now been withdrawn rather than satisfied.** That is not the same as
> being unblocked, and **`C` and `W` must not be read as a green light — correcting a claim is not
> lifting a gate.** Whether training may now start, and against which label space, is the project
> owner's decision rather than a formality that has been discharged, and no session may edit this
> file or `CLAUDE.md` to lift it. Probes, staging, reading and pre-registration remain fine, as they
> always were.
