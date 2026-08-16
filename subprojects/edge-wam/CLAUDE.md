# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where you are

`subprojects/edge-wam/` — one of the two halves WAM was split into on 2026-08-15. **The root
`/home/humanoid/develop/wam/CLAUDE.md` still applies in full** and is loaded alongside this file;
this one only adds what is specific to the edge sub-project.

**This directory holds no code.** It is a README, a hand-maintained task index (`TASKS.md`), task
files (`tasks/E-NN-*.md`) and a primary-source research pass (`research/`). All code, tests, configs
and scripts live at the repo root in the shared, unforked `src/wam/` — run every command from
`/home/humanoid/develop/wam`, not from here.

## Commands (all from the repo root)

```bash
.venv/bin/python -m pytest -q                  # full suite; ~1 618 tests, ~58 s
.venv/bin/python -m pytest tests/test_safety.py -q
.venv/bin/python -m pytest tests/test_safety.py::test_name -q   # single test
.venv/bin/ruff check .                         # expect: All checks passed! (ruff 0.16.0, pinned)
```

`ruff` is pinned and `[tool.ruff.lint]` names the rule set — a green lint with a floating linter is
a claim about the week it ran. `pyproject.toml` explains what `ignore = ["F405", "E402"]` hides and
the exact override command that reproduces the suppressed count; read it before adding a rule, a
`# noqa` or a per-file-ignore.

Installing: extras are `wan`, `data`, `train`, `serve`, `sim`, `jobs`, `dev`, and `local` (the flat
union that makes `docs/local_gpu.md` runnable end to end). `isaac` is deliberately empty — Isaac Sim
lives in its own Python because `isaacsim-core` pins a torch that would silently downgrade this
venv's. `scripts/preflight_gpu.py` parses the extras table to name the repair command for a missing
import; `scripts/preflight_isaac.py` gates an Isaac box before a rollout.

## Architecture worth knowing before you touch anything

The load-bearing shape is two seams, both of which are contracts rather than conveniences:

- **`G1Transport` is the only hardware seam**, with three implementations — `FakeG1Transport`
  (unit tests), `MujocoG1Transport` (physics + rendered pixels), `DdsG1Transport` (real vendor DDS).
  Above it the code is byte-identical in all three cases, and there is deliberately no
  `MujocoAdapter`: the sim buys coverage of the *production* adapter.
- **`JointWorldActionModel` depends on the `FlowBackbone` protocol, never a concrete backbone**
  (FR-09). `WanFlowBackbone` keeps the 5B DiT, the VAE and the text tower *out of the module tree*,
  registering only LoRA parameters — so "checkpoint the adapter, not the 5B model" is structurally
  enforced instead of being a flag someone remembers.

Everything the learned model emits passes the deterministic `wam/safety/` layer before the
controller. A 15 Hz on-device policy makes that stricter, not looser.

## The premise of this sub-project

A policy small enough to run **on the robot**: image in, action out, closed loop, no datacenter in
the path. Candidate is NVIDIA `Cosmos3-Edge` 4B (OpenMDW-1.1). This sub-project **post-trains an
existing model; it does not build one.**

The single idea that governs every design decision here:

> **Image → action is an *interface* choice, not an architecture choice.** We do not decode the
> video head. The video world model stays — it is what makes this a WAM rather than a VLA, and at
> inference the action and the predicted video come out of the same forward pass. Removing it would
> leave the exact thing this project exists to be an alternative to.

`research/2026-08-15-cosmos3-edge-and-dreamzero.md` is the primary-source pass. Read it before
proposing anything, and preserve its marking convention in anything you write: **[✓]** read off a
primary source, **[doc]** vendor blog or secondary outlet, **[?]** open.

## Tasks: `E-NN`, and `mc` cannot see them

Task files are `tasks/E-NN-slug.md` — same frontmatter shape as `.mc/tasks/` plus a `subproject:`
field and a `## Notes / Report` section that stays empty until the task runs. `TASKS.md` is the
index and is **maintained by hand**; `mc task next` only ever returns root `T-NN` tasks. Edit the
task file and update `TASKS.md` in the same change.

`depends_on:` / `blocks:` are load-bearing, not documentation — keep the ordering in frontmatter,
not only in prose.

## Gates, in the order they bind

1. **T-39, the positive control, has never run, and no training run in this sub-project starts
   before it reports.** It asks whether this corpus's own action column clears L1 under our scorer;
   if it does not, no policy trained on it can. Probes, staging, reading and pre-registration are
   all fine meanwhile.
2. **E-01 and E-02 gate E-05 and E-06.** E-01 asks whether the Edge policy path can run with no
   text (the released `Cosmos3-Edge-Policy-DROID` is language-conditioned); E-02 asks what a 28-dim
   G1/Dex3 embodiment actually costs. Both are cheap and need no GPU, and between them decide
   whether "no VLA" is a config flag or a re-training job.
3. **Pre-register before training** (E-05), in `docs/preregistration/PR-NN-*.md`. These are
   versioned, never edited in place — a gate rewritten after seeing its output is not a gate.

## House rules that have already cost something

- **A load-bearing string comes from code, not from prose or recollection** — model IDs, config
  field names, input contracts. Cite `file:line`.
- **Nothing gets submitted, downloaded at scale, or paid for without asking first.** Training is on
  Discoverer+ (`ehpc-aif-2026pg01-905`, H200, 4 h max walltime per job, ~4 875 of 5 000 GPU-h left);
  `docs/discoverer.md` is the *why* and `cluster/discoverer/README.md` the runbook, opening with
  seven never-do rules.
- Every rollout traces to checkpoint + dataset snapshot + config hash (AC-04).

## Numbers people keep getting wrong here

- **Block order in the 28-dim G1_Dex3 action vector is arm-first — `[0:14]` arm, `[14:28]` hand**,
  measured across all 13 sets on 2026-08-15 (T-043 §1). The hand-first figure belongs to a different
  corpus (`Humanoid-Everyday-G1`). Left/right and intra-hand order are still **[?]**, with three
  conflicting orderings on record — that is where the plausible-wrong-numbers risk lives.
- **3 152 real G1 episodes declaring `action float32[28]`** exist, Apache-2.0, in repos whose video
  we already hold.
- **Our "the world branch costs 108 pp" result bounds our old implementation on our corpus.** It is
  not evidence about the architecture class and must not be cited as such here.
- Cosmos3 embodiments include **AgiBot at 29D, which is a humanoid** — so "no humanoid in the
  supported vocabulary" is wrong; the accurate claim is "no G1 and no Dex3".
