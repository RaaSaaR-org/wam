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

**T-39, the positive control, has still never run.** Its `oracle_action` arm asks whether this
corpus's own action column can clear L1 under our scorer. If it cannot, no policy trained on this
data can — and both sub-projects are built on the same corpus. **Neither sub-project may start a
training run before T-39 reports.** Probes, staging, reading and pre-registration are all fine.
