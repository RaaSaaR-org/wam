# PR-12 result — **C**: one number in sixteen was the entire deficit

Ran 2026-08-16 on the workstation, CPU only, **zero GPU-hours, nothing submitted.** 26 evaluations,
about three minutes. Pre-registration `PR-12-step-zero-anchor-heterogeneity.md`, rule `T46_RULE_V1`,
committed in `6b4f836`; driver `scripts/probe_step_zero_anchor.py` and its 20 tests in `d4f6ec2`,
both **before any cell existed**. Task **T-46**. Artifact `runs/t46-step-zero/probe.json`.

**Verdict `C`: V-chain clears L1 on held-out half B by +69.15 % against the unmodified −379.68 %.**
The registered prediction P2 holds, and it holds by a margin that is not a matter of judgement.

## The whole finding is one line

Per-step MSE, half A, primary anchor `d = −2`, 474 chunks:

```
unmodified   4.79e-04  3.34e-06  3.26e-06  3.37e-06 ... 3.50e-06  3.70e-06
v_chain      3.51e-06  3.34e-06  3.26e-06  3.37e-06 ... 3.50e-06  3.70e-06
             ^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^ bit-identical, all fifteen ^^^^^^^^^
```

**Step 0 carried 143× the error of its neighbours. V-chain changes that one number and nothing
else** — G0.3 measured the difference on rows 1…15 at exactly `0.000e+00`, at both anchors. Nothing
about the corpus, the chunking, the delay, the gripper or the scorer moved.

That single element was **90.10 %** of the summed per-step MSE at `d = −2` and **91.26 %** at
`d = 0`, against the ~92 % §4 derived in advance from `horizon_ratio ≈ 0.006`.

## The grid

| cell | anchor | half | L1 | L2 | vs-zero | `horizon_ratio` | `smoothness_ratio` | level (0.1.0) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| unmodified | −2 | A | −224.89 | −39.14 | −84.97 | 0.0077 | 5.675 | below L0 |
| unmodified | −2 | **B** | **−379.68** | −132.68 | −165.68 | 0.0043 | 9.550 | below L0 |
| v_mask | −2 | A | +67.30 | +72.67 | +80.49 | 1.1049 | 0.274 | L4 |
| **v_chain** | **−2** | **B** | **+69.15** | **+76.30** | **+82.91** | **1.0200** | **0.276** | **L4** |
| unmodified | 0 | B | −410.03 | −139.04 | −182.50 | 0.0037 | 10.181 | below L0 |
| **v_chain** | **0** | **B** | **+69.41** | **+76.88** | **+83.05** | **0.9482** | **0.280** | **L4** |

**Gain on held-out B: +448.82 pp at `d = −2`, +479.44 pp at `d = 0`** — against a borrowed material
floor of 10.0 pp. Both halves, both anchors, both bench specs agree.

**It is not shrinkage.** PR-11 built the `skill_vs_zero_pct` guard because an MSE ratio can be
climbed by predicting less motion. V-chain scores **+82.91 %** against predicting no motion at all,
which shrinkage toward zero cannot do — it moves a prediction *toward* that baseline. And the
repeat and zero baselines are **numerically identical** between the two cells, because the target
chunks never changed: the entire L1 movement is the model arm's step-0 error falling.

**L2 moves with L1** (+76.30), so it is not an artefact of low-motion chunks.

## Gates

| gate | requirement | measured |
|---|---|---|
| **G0.1** unmodified cells reproduce T-44/T-45 | ±0.5 pp, 4 cells | drift **≤ 0.0026 pp**; retained counts identical at 474 / 486 across all twelve cells |
| **G0.2** `oracle_state` unmodified | ≥ 90 % | **+100.00 %** on both halves |
| **G0.3a** V-chain reaches row 0 | non-zero RMS | `2.2442e-02` at `d = −2`, `2.3594e-02` at `d = 0` |
| **G0.3b** V-chain touches nothing else | exactly zero | **`0.000e+00`** on rows 1…15, both anchors |
| **G0.3c** V-mask scores one step fewer | 15 vs 16 | held |
| extra | per-step profile reproduces the scorer's `horizon_ratio` | to **`0.00e+00`** |

## A previously registered prediction, from a different document, that this satisfies

PR-10 registered that a correct anchoring would drive `horizon_ratio` toward 1.0. **For the delay
shift it failed — it moved the wrong way**, and both PR-10 result documents recorded that failure.

Under V-chain at `d = 0`, `horizon_ratio` is **1.0021**. That prediction was written down before
this mechanism was known, by a different experiment, and it was not designed to be satisfied here.
It is the strongest corroboration in this document precisely because nobody was aiming at it.

## What this retires, including one of our own results

**The 67 ms command lead was fitting the discontinuity.** Under the unmodified anchoring, `d = −2`
beat `d = 0` (−224.89 against −253.70) and both PR-10 runs read that as a real controller lag.
Under V-chain the preference **flips**: `d = 0` is at least as good as `d = −2` on both halves
(A: +67.14 vs +65.70; B: +69.41 vs +69.15).

Once step 0 is homogeneous there is no lead left to find. The honest reading — stated as
interpretation, since it was not separately pre-registered — is that the delay sweep was optimising
the one contaminated element, and shifting the command slice was a way of making
`q_cmd[0] − q_state[s]` smaller rather than a measurement of the controller. **PR-10's ~11 % and
PR-11's ~2 % were both fractions of a deficit that was ~90 % a single subtraction.**

## Two driver defects, both found by gates firing, both recorded rather than quietly fixed

The pre-registration's text never changed. Its *implementation* was wrong twice, and each time a
gate caught it and returned a confident, finite, wrong verdict — this project's recurring failure
mode, so it goes in the result rather than in a commit message.

1. **First run returned `INVALID`.** G0.3's chunk-key check compared the **untrimmed** chunk
   dictionaries. V-chain requires `start ≥ 1`, so at `d = 0` it drops each episode's first chunk —
   which `trim_pairs` drops from scoring anyway, for every cell. The gate was comparing chunks that
   enter no number. PR-12 §3 puts the requirement on the **retained** count, so the check now
   compares the scored set, and a **stricter** gate was added that asserts the scored counts are
   equal across all three cells directly. They are: 474 and 486 everywhere.
2. **Second run returned `X` (coherence failure).** `_verdict` read `step_zero_share_pct` off the
   **V-mask** cell — a quantity that cannot exist, because dropping step 0 is what V-mask *is*. It
   was reporting the share of step 1 (~6 %) and failing a floor written for step 0. §4 names the
   registered quantity as a property of the unmodified profile; that is what it now reads.

**The cell values are bit-identical across all three runs.** Only the gate and verdict plumbing
changed; the full grid printed the same numbers each time, and both corrected checks are strictly
tighter than what they replaced. That is checkable in the artifact and is the reason `C` should be
read as `C` rather than as a third attempt at a verdict.

## A design defect in PR-12 itself, found by its own tests before any cell existed

At `d ≠ 0`, V-chain changes **two** things, not one. The homogeneous anchor is the command preceding
the *slice*, and the slice moved by `d`, so the anchor sits `d` steps from the eval timestamp.
"Homogenise step 0" and "keep the eval-timestamp anchor" are contradictory once the slice has moved;
no third definition satisfies both.

`test_at_a_nonzero_delay_v_chain_also_moves_the_anchor_and_that_is_a_confound` pins the confound at
exactly `d` steps and no more. **So the unconfounded test of P2 is the `d = 0` cell**, and the
`d = −2` cell the rule reads is a joint test. PR-12 §6 already required both anchors on both halves
to be recorded, so this needed no amendment — and empirically it does not matter: the two agree to
within 1.5 pp and `d = 0` is the *better* of the two.

## The bland side, registered in advance and duly arrived

PR-12 §5C registered that a repaired label space would be exposed at spec 0.2.0's **two-sided** L4
band on the opposite side from the one this project has worried about. It is:

- **spec 0.1.0** (`r ≤ 2.0`): `smoothness_ratio` **0.276** → **L4, moves-like-a-demo**, the top rung.
- **spec 0.2.0** (`0.5 ≤ r ≤ 2.0`): the same 0.276 is **below the floor** → **L3, holds-the-horizon**.

Over steps 1–15 the commanded stream is ~3.6× **smoother** than the executed trajectory. That
reproduces the peer session's independent decomposition (`docs/smoothness-ratio-audit.md`: 0.2747 at
`k = −2`) from different code on a different chunk set, and it inverts what this project's documents
have said about `smoothness_ratio` for as long as they have cited it.

## The blast radius, checked rather than assumed — and it is small

`C` is a positive number after fourteen negatives, which makes it exactly the kind of result that
gets over-read. So: **which recorded numbers were produced through the defective anchoring?**

```
grep -rn "commanded_to_chunk" scripts/ src/
```

- `scripts/eval_t39_baseline.py` — the definition and its two call sites
- `scripts/sweep_t39_anchor.py`, `sweep_label_anchoring.py`, `sweep_command_lowpass.py`,
  `probe_step_zero_anchor.py` — the four sweeps that import it
- **zero hits in `src/wam/`. Zero hits in `scripts/train_t39_baseline.py`.**

**The defect is confined to the evaluation adapter and everything built on it** — T-39's `G0b`
`oracle_action` arm, both PR-10s, PR-11, and this. It is a defect in the *instrument*, and in every
number that instrument produced.

Three things it therefore does **not** reach, each checkable:

1. **The GR00T training path never used it.** `train_t39_baseline.py` symlinks an episode subset,
   writes stats, and hands `--dataset-path` to the vendored trainer, which reads raw LeRobot through
   GR00T's own modality config. No WAM adapter is in that path.
2. **The corpus on disk is homogeneous by construction.** `relabel_chunks`
   (`convert_lerobot_g1.py:365`) is `q[s+t+1] − q[s+t]` at *every* `t` including 0, and that is what
   built `datasets/gr00t-apple-full`. The targets were never the problem; the comparison against
   them was.
3. **The runtime executor does not difference a command against a state at all** — it emits a
   `JOINT_DELTA` chunk from a decoder. There may be a separate step-0 question in prefix execution,
   but it is a different question and must not be smuggled in under this one's name.

**So the fourteen negatives in `docs/benchmark.md` are, on this evidence, untouched**: T-16, T-18 and
D1 scored WAM-format predictions against WAM-format targets, both homogeneous, with no
`commanded_to_chunk` in the path. This result does not absolve the project's track record, and
nobody should read it that way. *(Blast radius jointly established with the peer session, which
grepped it independently; the greps above were re-run here rather than taken on report.)*

## What this licenses

- **A defect report against `commanded_to_chunk`'s step-0 anchoring**, naming `q_cmd[0] − q_cmd[−1]`
  as the replacement, and **carrying PR-12 §3's stated cost**: the repaired chunk loses its only tie
  to the measured state, so it describes the commanded trajectory purely and relies on FR-05's
  re-observe-and-re-plan loop to correct accumulated tracking drift. That is a design consequence,
  not a footnote, and the closed-loop executor is where it has to be answered.
- **Proposing a relabel.** It relabels nothing here.

## What this does not license

- **Nothing about GR00T or any policy.** No model was trained, loaded or consulted; this scored
  oracles against oracles. PR-07 §6's prohibition is untouched, and `C` is not evidence that any
  policy can learn these labels — only that the labels stopped being self-contradictory.
- **No training run, no generation.** T-39's `VOID` is the project owner's to release. `C` removes
  the *cause* the VOID identified; it does not discharge the gate, and this session does not.
- **No retro-validation of the fourteen negatives.** They were scored against the executed-state
  target, which never changed. Nothing here makes a past run better than it was.
- **No edit to `src/wam/evaluation/benchmark.py` or `docs/benchmark.md`.** Both are untouched. That
  `smoothness_ratio` is dominated by an anchor discontinuity, and that the L4 gate has been read
  backwards, is a decision for the owner and needs its own pre-registration.
- **No attribution of the physics.** A homogeneous anchoring helping is consistent with a
  steady-state tracking offset, a controller-side feedforward term, and a recording-chain timestamp
  convention. This measures an anchoring; it does not identify what imposes the offset.
- **Nothing about the other twelve `G1_Dex3_*` corpora** (T-043) — though the probe is CPU-only and
  takes three minutes, so checking one is cheap.

## What comes next

The cheapest and most load-bearing follow-up is **not** another label experiment. It is that
`commanded_to_chunk` is the *evaluation* adapter, and the same step-0 question has to be asked of
the training path and of the runtime executor before "one line of the adapter" is true of anything
that ships. That is a code question with a known answer shape, and it is the honest next step
toward a policy that can be trained at all.
