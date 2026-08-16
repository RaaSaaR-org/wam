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

Its `oracle_action` arm asked whether this corpus's own action column can clear L1 under our scorer.
**It cannot: −359.41 pp on L1, 4.59× worse than repeating the last action.** The companion arm
`oracle_state` scored a bit-exact `mse 0.0` and +100 % on every rung, which is what makes the first
number a measurement rather than broken plumbing. The premise the gate was built on came back
**negative**, and by `T39_RULE_V1`'s own ordering — `not L1_action` is checked before any policy
branch — no policy number can move it.

**Read what that does and does not license, because it is easy to get backwards.** It licenses a
defect report against our label pipeline and **forbids any statement about GR00T**: the policy arm
never ran. It does not license swapping in another model — PR-07 §7's second candidate is
pre-registered only on outcome **N**, so running it now would test a second policy against the same
broken ruler.

> **The standing rule was "neither sub-project starts a training run before T-39 reports". T-39 has
> reported, with its own premise failing.** That is not the same as being unblocked, and it must not
> be read as a green light. Whether training may now start, and against which label space, is the
> project owner's decision rather than a formality that has been discharged. Probes, staging,
> reading and pre-registration remain fine, as they always were.
