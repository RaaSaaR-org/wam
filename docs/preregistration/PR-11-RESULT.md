# PR-11 result — **R**: the residual is not high-frequency, and the jerk metric was misleading us

Ran 2026-08-16 on the workstation, CPU only, **zero GPU-hours, nothing submitted.** 29 evaluations,
about two minutes. Pre-registration `PR-11-command-lowpass-sweep.md`, rule `T45_RULE_V1`, committed
in `90a0570`; driver `scripts/sweep_command_lowpass.py` and its 22 tests in `bd74e9a`, both before
any cell existed. Task **T-45**. Artifact `runs/t45-lowpass-sweep/sweep.json`.

**Verdict `R`: no cutoff clears L1 on half A at either anchor.** The commanded and executed streams
do not differ by a bandwidth. Post-processing these labels will not reconcile them.

## G0 — three gates, all passed, and the third one earned its place

| gate | requirement | measured |
|---|---|---|
| **G0.1** the four no-op cells reproduce T-44 | ±0.5 pp | drift **≤ 0.003 pp** on all four |
| **G0.2** `oracle_state` unfiltered | ≥ 90 % | **+100.00 %**, 960 chunks |
| **G0.3** the filter reaches the array, harder as `fc` falls | non-zero, monotone | `1.55e-03` at 12 Hz → `2.14e-02` at 1 Hz, monotone |

G0.1's four drifts are `−0.0025`, `+0.0015`, `−0.0017`, `−0.0007` pp — the no-op path is
bit-identical to T-44's, because the filter is applied to the episode's `action` array *before* the
chunk builder and the builder itself is untouched.

**G0.3 is the gate that would have caught the failure this experiment was most exposed to.** A
filter threaded through the call chain but never applied produces a flat grid, and a flat grid reads
as *"the jerk is irreducible"* — which is verdict **R**, the verdict this run actually returned. The
two are indistinguishable in the output. They are distinguished here by a number: at 1 Hz the filter
moves the commanded array by an RMS of `2.14e-02`, and that number is in the artifact.

## The grid — and the finding is how little happens

Half A, primary anchor `d = −2`, spec 0.1.0:

| `fc` | L1 | Δ vs no-op | `skill_vs_zero` Δ | `smoothness_ratio` | `horizon_ratio` | RMS change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 Hz | −220.88 | +4.02 | +2.29 | 5.381 | 0.00620 | 2.14e-02 |
| **2 Hz** | **−220.27** | **+4.63** | +2.63 | 5.380 | 0.00621 | 1.21e-02 |
| 3 Hz | −221.45 | +3.44 | +1.96 | 5.390 | 0.00649 | 8.12e-03 |
| 5 Hz | −223.35 | +1.54 | +0.88 | 5.403 | 0.00698 | 4.75e-03 |
| 8 Hz | −224.64 | +0.25 | +0.14 | 5.422 | 0.00748 | 2.98e-03 |
| 12 Hz | −224.41 | +0.48 | +0.27 | 5.543 | 0.00761 | 1.55e-03 |
| no-op | −224.89 | — | — | 5.675 | 0.00771 | 0 |

The secondary anchor `d = 0` has the same shape: best at 2 Hz, `−249.36` against a no-op of
`−253.70`, a gain of **+4.34 pp**. Half B agrees at both anchors. Both bench specs agree.

**The best cutoff in the grid is worth 4.63 pp**, less than half the borrowed
`MATERIAL_FLOOR_PP = 10.0`, against a deficit of 224.89. **Filtering recovers about 2 % of it.**
For comparison, PR-10's re-anchoring recovered ~11 %, and that was already the small answer.

The optimum is interior (2 Hz, not the 1 Hz edge), so **E** does not apply and the grid is bounded
on both sides. The rule never reached the shrinkage guard, because nothing came close to clearing
L1 — but the guard's diagnostic still reads correctly: `skill_vs_zero_pct` moves the *same*
direction as L1 and about half as far, so what little the filter buys is not pure shrinkage either.

## The number that reframes the problem

**A 1 Hz cutoff on a 30 Hz signal moves `smoothness_ratio` from 5.675 to 5.381 — a 5.2 % change.**

That should be impossible for a signal whose excess jerk lives in its high frequencies. Filtering at
1 Hz discards everything above 1/15th of Nyquist; if the command were the executed trajectory plus
high-frequency noise, the predicted jerk would collapse. It does not move.

`horizon_ratio` says where the error is instead, and says it at **every** cutoff: `0.0062`–`0.0077`
at `d = −2`, `0.0054`–`0.0061` at `d = 0`. **Last-step error is between 1/130th and 1/185th of
first-step error, and filtering does not shift that.**

**Stated as measurement, not as mechanism**, the discipline PR-07-RESULT and both PR-10s used: the
two measurements above are measured. The reading that follows — that the deficit is a **level offset
between the command and the state at the chunk's anchor**, not a spectral property of the command's
trajectory, and that a filter cannot reach it because a filter reshapes steps 1…15 while the error
sits in step 0 — is an interpretation consistent with them and is **not** established here.
Establishing it needs the per-step error profile, which this sweep did not compute and which is
named in §"What comes next" rather than run after the fact.

**If that reading holds, `smoothness_ratio` has been misleading this project.** It is a jerk ratio,
and a chunk that opens with a large step discontinuity is "jerky" by that definition no matter how
smooth the remaining fifteen steps are. Every previous document, this one included, has read
`smoothness_ratio 8.52` as *"the command carries high-frequency content"*. The 1 Hz cell says it
cannot mean that. It is more likely reporting the same first-step offset `horizon_ratio` reports,
in different units.

## What this retires

**"Filter the labels" joins "re-anchor the labels" as a measured non-answer.** Between PR-10 and
PR-11 the two obvious cheap repairs to the label space have now been tried and priced: an anchor
shift is worth ~11 % of the deficit, a low-pass is worth ~2 %, and they are not additive in any
useful way — the best filtered cell at the best anchor is still `−220.27`, which is 3.2× worse than
repeating the last action.

Both were the kind of explanation that gets re-proposed indefinitely until someone measures it.
They are now measured.

## What comes next, and it is not another repair

Per PR-11 §6, **R** is the outcome that most changes direction while starting the least work: it
says the difference between the commanded and executed streams is not something a transformation of
these recordings reaches, and that **PR-04's collection spec — what *kind* of data — is ahead of
processing this data better.**

Two things are named as follow-up rather than answered, and neither should be started by reading
this document alone:

1. **The per-step error profile of `oracle_action`.** Cheap, descriptive, and it converts this
   document's central reading into a measurement. It was deliberately not run after seeing the
   verdict.
2. **Whether `smoothness_ratio` measures what its name and every citation of it claim.** That is a
   question about `src/wam/evaluation/benchmark.py`, not about the corpus, and it bears on numbers
   throughout `docs/benchmark.md` — including the L4 *moves-like-a-demo* gate. It should be
   pre-registered like anything else.

## What this does not license

- **Nothing about GR00T or any policy.** No model was trained, loaded or consulted; this scored
  oracles against oracles. PR-07 §6's prohibition is untouched.
- **No training run, no generation.** The gate is the project owner's to release, and `R` is not a
  release.
- **No relabelling**, and no retro-validation of any of the fourteen negatives.
- **No attribution of the physics.** That a low-pass does *not* help says the residual is not a
  bandwidth mismatch; it does not identify what the residual is.
- **Nothing about the other twelve `G1_Dex3_*` corpora** (T-043), and **nothing about grasping** —
  `gripper_accuracy` stayed withheld by the scorer on every cell.

## Process note

PR-11 was **claimed across live sessions before it was written**, after two sessions independently
ran the delay sweep the same afternoon and both called it PR-10. Both peers confirmed it unclaimed.
The duplicate PR-10 remains unresolved and is the user's to settle.

One design gap in this pre-registration was found by its own tests before any cell existed: §5 fixed
the verdict precedence but not the meaning of "best cutoff" under a **tie**, and a naive `max` would
have reported a perfectly flat grid as verdict **E**, *"over-smoothing wins monotonically"* — the
one reading a flat grid cannot support. Ties break toward less filtering. That is recorded in the
driver's docstring and pinned by a regression test, **not** added to PR-11.
