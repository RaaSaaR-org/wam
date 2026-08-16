# `smoothness_ratio` audit — 96.8 % of it is one term, and that term is the anchor

Measured 2026-08-16 on this workstation, CPU only, **no re-evaluation and no GPU**: the chunks were
written to disk when the PR-10 cells were scored. Tool: `scripts/audit_smoothness_ratio.py`.
Artifacts: the `predictions.jsonl` already under `runs/t39-baseline-seed0/pr10-anchor-sweep/`.

**This is not an experiment and it has no verdict.** It carries no pre-registration because it asks
a question about our own code rather than about the corpus, and nothing in it can move a recorded
result: L4 smoothness has never gated one. PR-07, PR-10 and PR-11 all turned on L1/L2.

## Why it was run

PR-10 and `PR-10-RESULT-T-44` both read `smoothness_ratio ≈ 8` as *"the commanded stream carries
high-frequency content the executed trajectory does not"*. PR-11 then low-passed the commanded
column at **1 Hz** — a fifteenth of Nyquist on a 30 Hz signal, which discards essentially all
high-frequency content — and `smoothness_ratio` moved from 5.675 to 5.381. **Five percent.** Those
two statements cannot both be true, and the second one is a measurement. The peer session running
PR-11 flagged the contradiction rather than explaining it away, which is the only reason it was
caught.

## What the metric computes

`bench_metrics` accumulates, per chunk and per arm:

```python
d2   = x[2:] - 2*x[1:-1] + x[:-2]        # second difference along the WITHIN-CHUNK time axis
jerk = sum(d2**2) / count
```

A second difference of a JOINT_DELTA chunk is a third difference of position, so the arithmetic is
right and the name is defensible — **provided every element of `x` is the same quantity.**

For the target it is. `convert_lerobot_g1.relabel_chunks:365` builds `targets[t] = q[s+t+1] -
q[s+t]` for every `t`: one homogeneous first difference of executed positions.

For a chunk anchored on an observed state it is not. `eval_t39_baseline.commanded_to_chunk:253-255`
builds

```
targets[0]   = q_cmd[0] - q_state[s]     # command MINUS STATE — the standing tracking error
targets[t>0] = q_cmd[t] - q_cmd[t-1]     # command minus command — a per-step increment
```

**Step 0 is the only step in the chunk that subtracts a command from a state.** A steady-state
tracking offset cancels in every homogeneous first difference and survives at full magnitude in
that one. `d2[0] = x[2] - 2*x[1] + x[0]` is the sole jerk term containing it, and the sum is over
**squares**, so a single term a few times larger than the others takes the whole statistic.

## What was measured

Variant A of the PR-10 grid, 992 chunks, 40 holdout episodes. The audit reproduces `bench.json`'s
`smoothness_ratio` to **1.8e-14** before decomposing it — an audit that computes its own slightly
different number audits nothing.

| | `k = 0` | `k = −2` |
|---|---:|---:|
| `smoothness_ratio` as published | **8.2827** | **7.7011** |
| share of the **predicted** jerk sum carried by index 0 | **96.8 %** | **96.7 %** |
| share of the **target** jerk sum carried by index 0 | 6.6 % | 6.6 % |
| `smoothness_ratio` with index 0 dropped from **both** arms | **0.2797** | **0.2747** |

The target's profile is flat across all fourteen indices to within a point — 6.6 % ≈ 1/14 — exactly
as a homogeneous first difference must be. The predicted profile is one spike and thirteen terms of
about 0.2 % each.

Index 0 is dropped from **both** arms, not from the prediction only. The target has no
discontinuity there, so removing its index 0 costs it a legitimate term and biases the comparison
**against** this finding. It collapses anyway.

## What this changes

**The reading in PR-10-RESULT and PR-10-RESULT-T-44 is wrong, and it is wrong in direction, not
just in size.** Over steps 1–15 the commanded stream is about **3.6× smoother** than the executed
trajectory, not 8× jerkier. `smoothness_ratio ≈ 8` is very nearly a restatement of the first-step
discontinuity that `horizon_ratio ≈ 0.005` already reports, in different units — which is also why
it barely responds to re-anchoring, and why a 1 Hz low-pass moved it 5 %. Filtering the command
cannot remove a DC offset between the command and the state.

Note the direction of the corrected number: **0.28 is below the two-sided floor of 0.5** that spec
0.2.0 added. The command does not land inside the L4 band; it crosses it and fails on the *bland*
side. "Below the gate of 2.0" is true and is half the statement.

**This is not a defect in `commanded_to_chunk`.** `action[t] - q[t]` is the correct commanded
displacement over step `t`, its docstring argues it, and `tests/test_t39_baseline.py` kills three
plausible alternatives. Under **perfect tracking** (`action[i] == q[i+1]`) element 0 is a plain
first difference like the others and no discontinuity exists at all. The contamination is
proportional to how badly the arm tracks — which is precisely the quantity T-39 set out to measure.
The L4 gate therefore reads a real number, computed correctly, on a vector with one element that
means something else: a worse failure mode than a wrong formula, and harder to see.

## How the mechanism is pinned

Two tests in `tests/test_benchmark.py` hold the command **byte-identical** and move only the anchor:

- `test_a_tracking_offset_alone_inflates_smoothness_ratio` — two chunks differing in exactly one
  element, the one the anchor sets. Perfect anchor scores 1.0; a 0.03 rad standing offset scores
  past 4× the gate.
- `test_low_passing_the_command_barely_moves_an_offset_dominated_ratio` — every increment replaced
  by their mean, the most violent low-pass available, so steps 1–15 carry *no* high-frequency
  content whatsoever. The ratio moves under 5 %. PR-11's result as a controlled fact rather than an
  inference.

`test_the_smoothness_audit_reproduces_the_metric_it_audits` pins the tool against a `bench.json`
it wrote, which is the same `drift` check the script prints.

## What is deliberately not done here

- **`src/wam/evaluation/benchmark.py` is untouched and no gate moved.** What L4 should do about an
  anchored chunk is a decision that reaches every number in `docs/benchmark.md`, and it is not one
  session's to take. Surfaced, not taken.
- **No archived result is amended.** The `smoothness_ratio` values in `docs/benchmark.md` were
  computed correctly under the shipped metric and stay as recorded; what changes is the sentence
  underneath them.
- **Nothing is claimed about which anchoring is right.** Whether a homogeneous step 0 (`targets[0]
  = q_cmd[0] - q_cmd[-1]`) clears L1 is unmeasured here and is the peer session's PR-12.
- **Nothing about GR00T, no training, no generation.** Unchanged by every number above.
