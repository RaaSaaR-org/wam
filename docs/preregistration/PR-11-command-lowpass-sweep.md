# PR-11 — Is the residual the arm's low-pass filter, or is it content?

Pre-registered **2026-08-16**, after both PR-10 runs and **before any filtering is run**. Task
**T-45**. Rule **`T45_RULE_V1`**, fixed in §5 of this file and nowhere else. Zero GPU-hours: an
offline re-score of artifacts already on disk.

**PR-11 was claimed across sessions before it was written.** Two sessions independently ran the
delay sweep the same afternoon and both called it PR-10; this number was announced to both live
peers and confirmed unclaimed by the author of `PR-10-anchor-delay-sweep.md` before a line of this
file existed. That is the fix for the duplication, and it is recorded here because the duplicate
PR-10 is still unresolved.

## 1. What both PR-10 runs converged on

Two drivers, two chunk sets, two rules, one finding: the commanded column leads the executed state
by about two control steps, that offset is real and replicates out-of-sample, and it is nowhere
near enough.

| | `PR-10-RESULT.md` (peer) | `PR-10-RESULT-T-44.md` (T-44) |
|---|---|---|
| best offset on L1 | `k = −2` | `d = −2`, on both holdout halves |
| gain | +29.75 pp of 342.24 | +28.81 / +30.35 pp of 253.70 / 410.03 |
| does anything clear L1 | no | no |
| `smoothness_ratio` at the optimum | 8.28 → 7.70 | 6.21 → 5.67 |

The fractions differ only by denominator — the same ~29 pp numerator over 992 pooled chunks versus
474/486 in halves — and neither should be quoted without its chunk set.

**What survives every anchoring both runs tried is the jerk.** Optimal alignment leaves the
commanded stream 3–5× jerkier than the trajectory the robot executed, against an L4 gate of 2.0. A
shift re-indexes a signal; it cannot smooth one. Both result documents end at the same sentence:
the live question is *what the commanded stream contains that the executed trajectory does not*.

## 2. The hypothesis, and why it is worth one CPU-hour

A position-controlled arm is a low-pass filter. The controller is told to go somewhere at 30 Hz;
inertia, gearing and the servo loop mean the joint does not follow the command's high-frequency
content, and the executed trajectory is a smoothed version of what was asked for. If that is the
whole of the residual, then **low-pass filtering the commanded column before it becomes a chunk
should recover most of the remaining deficit** — because it would be reconstructing, in software,
the filter the arm applies in hardware.

If it does not, the two streams differ in content that is not high-frequency, and no amount of
post-processing the labels will reconcile them. That is a materially different project state and
the reason this is worth running rather than assuming.

## 3. The manipulation — one variable, and the filter is defined here, not chosen later

The commanded array is low-pass filtered **before** `commanded_to_chunk` sees it. Everything else
— chaining, chunk length, anchor, the delay — is untouched.

**The filter is a zero-phase symmetric FIR: a Hann-windowed sinc, numpy only.**

```
half   = ceil(2.0 * fs / fc)                 # fs = 30 Hz, fc = the swept cutoff
n      = arange(-half, half + 1)
kernel = sinc(2 * fc / fs * n) * hanning(2 * half + 1)
kernel = kernel / kernel.sum()
```

applied per channel by `np.convolve(..., mode="valid")` over the **whole episode** with
**edge-clamped** padding of `half` samples at each end.

Every clause of that is a decision, and each is fixed here rather than discovered during the run:

- **Zero phase, not `filtfilt`, not a causal filter.** A causal filter adds phase lag, which would
  be indistinguishable from the delay PR-10 just measured — the experiment would confound its own
  manipulation with the previous one. A symmetric kernel has exactly zero phase by construction.
- **numpy, not scipy.** `scipy` is not in the WAM venv, and installing it would change the
  dependency set every number in `docs/benchmark.md` was produced under — the same argument
  `72_build_t39_env.sbatch` makes for keeping the trainer's venv separate. A 15-line explicit
  kernel is also auditable in a way `filtfilt`'s default `padtype`/`padlen` is not.
- **Whole episode, not per chunk.** Filtering inside a chunk would create a discontinuity at every
  chunk boundary and put the artifact exactly where `horizon_ratio` looks.
- **Edge-clamped, not reflected.** Reflection invents a symmetry the recording does not have.
  Clamping is the honest choice at an episode boundary and the chunks at both ends are dropped
  anyway (§3, the intersection rule, inherited from PR-10).

**The grid, fixed here:** `fc ∈ {1, 2, 3, 5, 8, 12} Hz`, plus an explicit **no-op** cell (no
filtering at all) which is the control and the bridge. 30 fps means Nyquist is 15 Hz, so 12 Hz is
already a light touch and 1 Hz is aggressive.

**Two anchors, and which one is primary.** The grid runs at `d = −2` (**primary** — the best anchor
both PR-10 runs found, so this asks whether filtering fixes what is left *after* the known fix) and
at `d = 0` (**secondary** — so the two corrections are visible jointly and the result stays
readable against T-39). `d` is **not** re-fitted here. It is taken from T-44 and held fixed;
searching both at once would be a 2-D garden of forking paths.

**Chunk set and halves are inherited unchanged from PR-10**: first and last chunk of every episode
dropped uniformly, holdout split A = even index / B = odd index in
`configs/splits/t18_holdout_episodes.txt`. `fc*` is fitted on **A** and every verdict-bearing
number is read on **B**, at that one `fc*`, with no further search.

## 4. The trap this design exists to avoid — shrinkage, not noise removal

**A low-pass filter shrinks the magnitude of what it filters.** `skill_vs_repeat_pct` is an
MSE-ratio metric, and shrinking a noisy prediction toward its mean improves MSE whether or not the
removed part was noise. A filter aggressive enough to flatten the commanded deltas toward zero
could therefore climb the L1 ladder for a reason that has nothing to do with the arm's dynamics,
and the curve would look exactly like a success.

The registered guard uses a metric already in every bench report: **`skill_vs_zero_pct`**, the
comparison against predicting no motion at all. Shrinkage toward zero cannot improve that number —
it moves the prediction *toward* the zero baseline it is being scored against. So **F additionally
requires `skill_vs_zero_pct` to improve by at least the material floor**, and a cell that clears L1
without it is verdict **S**, recorded as shrinkage rather than as a finding.

## 5. Gates — `T45_RULE_V1`

Ladder unchanged; `MATERIAL_FLOOR_PP = 10.0` borrowed from `I8_RULE_V3` for the third time rather
than coined, so the floor cannot be the finding.

- **L1** `skill_vs_repeat_pct > 0` · **L2** `ci_skill_vs_repeat_pct > 0`

**G0 · INVALID — runs first, can stop everything.**

1. **The no-op cell reproduces T-44.** At `d = −2`: A within ±0.5 pp of `−224.89`, B of `−379.68`.
   At `d = 0`: A of `−253.70`, B of `−410.03`. Outside that, this is not the same measurement and
   nothing may be compared across documents.
2. **`oracle_state` at `d = 0`, unfiltered, still reaches `skill_vs_repeat_pct ≥ 90 %`.**
3. **THE FILTER REACHES THE ARRAY.** Every filtered cell must record a non-zero RMS change against
   the raw commanded array, and that change must be **monotonically larger as `fc` falls**. A
   filter threaded through the call chain but never applied produces a flat grid and a confident
   verdict that the jerk is irreducible — the same shape a real **R** has, and indistinguishable
   from it in the output. This gate is checked at runtime and the per-cell RMS change is written
   into the artifact, so an inert grid is visible in the result rather than only in a test.
   *(This gate exists because the author of `PR-10-anchor-delay-sweep.md` reported paying a
   mutation test to notice the analogous defect in the offset knob.)*

**Which conclusion is expensive.** **F** is, as **T** was in PR-10: it licenses proposing a
relabel of the whole corpus, this time with a filter in the pipeline, and re-reading
`docs/benchmark.md` against a moved ruler. So F carries the material margin, the held-out
confirmation *and* the anti-shrinkage guard. **R** — "the residual is not high-frequency" — changes
nothing and starts no work.

**The verdicts**, read on half **B** at the `fc*` fitted on half **A**, primary anchor `d = −2`:

| | condition | reading |
|---|---|---|
| **F** | `fc*` is not the no-op, **B** clears **L1**, B's L1 gain over its own no-op cell `≥ 10.0 pp`, **and** B's `skill_vs_zero_pct` gain `≥ 10.0 pp` | the residual is the arm's low-pass. Anchor + filter is a label space worth proposing |
| **S** | as F, but the `skill_vs_zero_pct` guard fails | the L1 gain is magnitude shrinkage, not noise removal. Licenses nothing, and is a warning about reading MSE ratios under any smoothing |
| **R** | **no** `fc` clears L1 on **A**, at **either** anchor | the residual is not high-frequency content. Post-processing the labels will not reconcile the two streams, and the question moves to PR-04's collection spec |
| **E** | `fc*` is the **lowest** swept cutoff (1 Hz) | over-smoothing wins monotonically, which is the shrinkage signature at the grid edge. **Nothing is concluded.** One extension, to 0.5 Hz, re-read under the same rule. No second extension |
| **I** | anything else | indeterminate; nothing licensed |

**Precedence is fixed here, and deliberately so** — PR-10's table left it open and the driver had
to decide it in a docstring. Evaluated in order: **E, then R, then S, then F, then I.**

**Recorded regardless of verdict:** the full grid at both anchors on both halves under both bench
specs; `horizon_ratio`, `smoothness_ratio` and the RMS filter change per cell; the no-op bridges;
the retained chunk count; and the wall time.

## 6. Reading the outcome — decided before the numbers exist

- **F** licenses a defect report naming `(d*, fc*)` and licenses *proposing* a relabel. It does not
  relabel anything, does not retro-validate any of the fourteen negatives, and is not a licence to
  train.
- **R** is the outcome that most changes the project's direction while starting the least work: it
  would mean the commanded and executed streams differ in something neither an anchor nor a filter
  reaches, and that **collecting different data (PR-04) is ahead of processing this data better**.
- **S** is a result about our metric, not about the corpus, and it is worth having: it would say
  that any future smoothing of labels anywhere in this project can climb L1 for free, which is a
  standing hazard nobody has written down.
- **No outcome licenses a statement about GR00T or any policy.** No model is trained, loaded or
  consulted; this scores oracles against oracles.
- **No outcome unblocks training** after T-39's `VOID`. That is the project owner's call.
- **No outcome attributes the physics.** A cutoff that helps is consistent with arm dynamics, with
  a controller-side filter, and with the corpus's own recording chain. PR-11 measures a cutoff; it
  does not identify what imposes it.

## 7. Cost

Zero GPU-hours. CPU only: 7 cells × 2 anchors × 2 halves plus gates, on artifacts already on disk.
Expected minutes.

## 8. What must exist before this runs

1. `scripts/sweep_command_lowpass.py`, importing `commanded_to_chunk`, `raw_anchor_indices`,
   `read_raw_episode`, `ChunkLookupPolicy` and the trim rule from the existing drivers rather than
   re-implementing them. Only the filter is new.
2. **A test that fails when the filter is removed.** Not "a test that the filter is called" — one
   that fails if the filtered array equals the raw array, at every swept cutoff. G0.3 is the
   runtime half of the same guard; this is the offline half.
3. A test that the kernel is symmetric and sums to 1, so "zero phase" is a property of the code and
   not of this document's prose.
4. A test that a pure sinusoid above the cutoff is attenuated and one below it is not — the filter
   does the thing its name claims, checked against an analytic signal rather than against itself.

## 9. What this cannot answer

- **Whether a policy could learn the filtered labels.** It scores oracles. Necessary, not
  sufficient.
- **Whether the cutoff is constant across joints.** One scalar `fc` is assumed by construction, as
  one scalar `d` was in PR-10 — and PR-10's L1/L2 disagreement about `d*` is already weak evidence
  against per-joint uniformity. A per-joint fit is a different design and inventing it after seeing
  this grid is what pre-registration exists to prevent.
- **The gripper.** Withheld by the scorer on every arm, for the reason PR-07-RESULT records.
- **Anything about the other twelve `G1_Dex3_*` corpora** (T-043), or about `docs/benchmark.md`'s
  validity beyond the bound PR-07-RESULT already placed on it.
