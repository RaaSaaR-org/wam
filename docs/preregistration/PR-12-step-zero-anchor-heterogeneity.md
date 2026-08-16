# PR-12 — Step 0 is the whole defect. Does making it homogeneous repair the labels?

Pre-registered **2026-08-16**, after PR-11 returned `R` and after the jerk decomposition in §2
**identified** the defect. This file registers **one** prediction — §4's P2 — and it is the only
thing here that is unmeasured at the time of writing. Task **T-46**. Rule **`T46_RULE_V1`**, fixed
in §6 and nowhere else. Zero GPU-hours.

**PR-12 was claimed across live sessions before it was written.** That claim is why this file says
what it says: the peer session replied that the diagnostic half was **already on disk**, which
converted two of my three intended predictions into prior work. They are cited below as
measurements, with attribution, and they carry no verdict. **A prediction whose answer already
exists is not a prediction**, and the process fix PR-11 introduced is what caught it — before this
file was committed rather than after.

## 1. Why this is not the third repair PR-11 said not to attempt

`PR-11-RESULT.md` closes with *"the next thing is not another repair"*, written about a specific
class of thing: transformations guessed at from the outside of the signal. PR-10 re-indexed the
commanded stream in time. PR-11 reshaped its spectrum. Both were hypotheses about **what the
recording contains**; both were priced, at ~11 % and ~2 %.

This one was **read off the code rather than hypothesised about the data**. Two functions, side by
side:

```
scripts/convert_lerobot_g1.py:365     target[t]      = q[s+t+1] - q[s+t]        state   - state
scripts/eval_t39_baseline.py:254-255  predicted[0]   = q_cmd[0] - q_state[s]    COMMAND - STATE
                                      predicted[t>0] = q_cmd[t] - q_cmd[t-1]    command - command
```

The target side is a homogeneous first difference at all sixteen steps. The prediction side is
homogeneous at fifteen. **Step 0 is the only element in either chunk that subtracts a quantity of
one kind from a quantity of another kind.**

That asymmetry is not a typo — `eval_t39_baseline.py:218-241` derives it deliberately and
`tests/test_t39_baseline.py` kills three plausible alternatives to it. It is correct **under the
premise it states**: `action[i] == q[i+1]`, perfect tracking. The premise is where the exposure
lives. Let `c` be a steady-state tracking offset — the standing distance between where a
position-controlled joint is told to be and where it settles under gravity and load:

- in `q_cmd[t] - q_cmd[t-1]`, `c` appears in both terms and **cancels**;
- in `q[s+t+1] - q[s+t]`, `c` is absent from both and **never enters**;
- in `q_cmd[0] - q_state[s]`, `c` appears in **one term only** and **survives at full magnitude**.

A constant tracking offset — the most ordinary property a position-controlled arm has — contaminates
exactly one element of a sixteen-element chunk.

## 2. What is already measured. Prior work, not prediction.

**Two sessions reached this code reading independently from the same PR-11 cell**, and the peer had
already built and run the decomposition. `scripts/audit_smoothness_ratio.py` reads the PR-10
prediction files already on disk, decomposes the jerk sum by within-chunk index, and reproduces
`bench.json`'s `smoothness_ratio` to `1.8e-14` — which is what makes the decomposition worth
reading at all.

Measured on variant A, 992 chunks, spec 0.1.0 — **the peer's chunk set, not T-44's**:

| | index 0's share of the **predicted** jerk sum | index 0's share of the **target** jerk sum |
|---|---|---|
| `k = 0` | **96.8 %** | 6.6 % (= 1/14, flat) |
| `k = −2` | **96.7 %** | 6.6 % (= 1/14, flat) |

The target's profile is flat across all fourteen indices to within a point, **exactly as a
homogeneous first difference must be**. The predicted profile is one spike and thirteen terms of
~0.2 % each.

**`smoothness_ratio` with index 0 dropped from both arms: 0.2797 at `k = 0`, 0.2747 at `k = −2`**,
against published values of 8.2827 and 7.7011.

Three things follow, and none of them is a prediction of this document:

1. **The mechanism is identified, not hypothesised.** 96.7 % in one of fourteen terms is not a
   tendency.
2. **`benchmark.py:538` is arithmetically correct.** `targets` are already first differences of
   position (`schema.py:178`), so a second difference of them is a true third derivative. PR-11's
   result document reasoned toward the metric being wrong; it is not. **The metric is right and the
   vector it is fed has one corrupted element** — a worse failure mode, because a correct
   computation over a corrupted input survives every unit test the metric has.
3. **The direction of the published statement is inverted.** Over steps 1–15 the command is about
   **3.6× smoother** than the executed trajectory, not 8× jerkier. `PR-10-RESULT.md`, T-44's result
   document and this project's prose generally have read `smoothness_ratio` as *"the command carries
   high-frequency content"*. It does not. It carries an anchor discontinuity.

The peer dropped index 0 from **both** arms rather than from the prediction only — the target has no
discontinuity there, so removing its index 0 costs it a legitimate term and **biases the comparison
against the finding**. It collapses anyway.

**None of the above licenses anything on its own**, and §7 says why.

## 3. The manipulation

```
targets[0] = q_cmd[0] - q_cmd[-1]        # the PREVIOUS COMMAND, not the measured state
```

where `q_cmd[-1]` is the commanded column at index `s-1`. Everything else — chaining for `t>0`,
chunk length, the delay `d`, the gripper path — is untouched. Call it **V-chain**.

**Three properties, all of which are why it is worth running:**

1. **It is identical to the current anchoring under the premise the current one states.** With
   perfect tracking, `q_state[s] = q_cmd[s-1]` by definition, so the two coincide exactly. This is
   not a change of premise; it is the same premise **made robust to the premise being false**.
2. **It is available at inference.** A policy always knows the command it emitted one step ago. It
   does not know the state one step in the future. Nothing here needs information a deployed policy
   would lack.
3. **It has a real cost, stated before the run rather than discovered after.** The chunk loses its
   only tie to the measured state. Every step becomes command-to-command, so the label never
   corrects accumulated tracking drift — it describes the *commanded* trajectory purely and trusts
   the closed loop (FR-05: execute a prefix, re-observe, re-plan) to correct. That is defensible and
   it is also a genuine loss of information. **If V-chain wins, this cost goes into the defect
   report, not into a footnote.**

A second cell, **V-mask** — score steps 1…15 only, changing no label — is run as an **instrument**,
to size step 0's share of the *MSE* sum (§2 measured its share of the *jerk* sum, which is a
different sum). V-mask is not a proposal: a policy that does not predict its first step is not a
policy.

**Grid, fixed here.** Both cells at **`d = −2`** (primary — T-44's and the peer's optimum) and
**`d = 0`** (secondary, so the result stays readable against T-39). `d` is **not** re-fitted;
re-fitting it alongside a new anchoring is a 2-D garden of forking paths.

**Chunk set and halves inherited unchanged from PR-10/PR-11**: first and last chunk of every episode
dropped uniformly, holdout split A = even index / B = odd index in
`configs/splits/t18_holdout_episodes.txt`. **V-chain has no free parameter**, so A and B are both
replications and the verdict is read on **B**.

V-chain needs `commanded[s-1]`, one sample earlier than the current anchoring reads. The inherited
trim already drops each episode's first chunk, so at `d = 0` this is free and at `d = −2` it is
inside the retained window. Any chunk with `s + d - 1 < 0` is dropped, and **the retained count must
match the unmodified cell's** — recorded, and checked by G0.1.

## 4. The one prediction, and the two things that are not predictions

**P2 — V-chain clears L1 on half B.** This is the entire at-risk content of PR-12. It fails if the
tracking offset is not constant: if `c` scales with velocity, load or joint, it does not cancel
between two consecutive commands either, and re-homogenising step 0 buys little.

Explicitly **not** predictions, because §2 already measured them or something that implies them:

- Step 0 dominating the jerk sum — **measured at 96.7 %**.
- `smoothness_ratio` excluding step 0 falling below the L4 ceiling — **measured at 0.2747**.
- Step 0 dominating the **MSE** sum — not directly measured, but `horizon_ratio ≈ 0.006` across
  PR-07/PR-10/PR-11 already implies ~92 % if steps 1–15 resemble the last step. **It is recorded,
  not predicted**, and §6 uses it only as a coherence check that can invalidate, never as
  confirmation.

## 5. Two traps this design exists to avoid

**A. A manipulation that is silently a no-op.** A V-chain implementation that reduced to the current
anchoring would produce a flat grid, and a flat grid reads as a confident negative. This is the same
hazard PR-11's G0.3 was built for, and the peer flagged it again unprompted. G0.3 below is its
successor and is **two-sided**: V-chain must differ from the unmodified cell on row 0 **and be
bit-identical on rows 1…15**. The second half matters more — a V-chain that accidentally re-chained
every step would be a different experiment wearing this one's name.

**B. An instrument that flatters itself.** V-mask removes the largest element of a sum and then
reports the sum got smaller. That is arithmetic, not a finding: `mse`, `zero_mse` and `repeat_mse`
all shrink when step 0 goes. **V-mask is therefore only ever read as a ratio against baselines
masked the same way.** If step 0 is merely the largest step, the baseline loses it too and
`skill_vs_repeat_pct` barely moves; a large change under masking means step 0 is where
`oracle_action` is *differentially* worse than the baseline, which is the actual claim. **A V-mask
cell that improves raw `mse` but not `skill_vs_repeat_pct` is recorded as arithmetic and licenses
nothing.**

**C. The bland side, which §2 just made a live risk.** Over steps 1–15 the command is ~3.6×
**smoother** than the executed trajectory, and spec 0.2.0's L4 is a **two-sided** band
(`0.5 ≤ r ≤ 2.0`, `benchmark.py:130-133`). So a V-chain label space is exposed at the *opposite*
gate from the one this project has been worrying about: it may fail L4 for being too bland. **This
is registered as an expected possibility, not as a surprise to be explained afterwards**, and §6
records L4 under both specs without letting either decide the verdict.

## 6. Gates — `T46_RULE_V1`

Ladder unchanged. `MATERIAL_FLOOR_PP = 10.0`, borrowed from `I8_RULE_V3` for the fourth time rather
than coined, so the floor cannot be the finding.

- **L1** `skill_vs_repeat_pct > 0` · **L2** `ci_skill_vs_repeat_pct > 0` · **L4** spec 0.1.0
  `r ≤ 2.0`, spec 0.2.0 `0.5 ≤ r ≤ 2.0`

**G0 · INVALID — runs first, can stop everything.**

1. **The unmodified cell reproduces T-44/T-45.** At `d = −2`: A within ±0.5 pp of `−224.89`, B of
   `−379.68`. At `d = 0`: A of `−253.70`, B of `−410.03`. Retained chunk counts must match.
2. **`oracle_state` at `d = 0`, unmodified, still reaches `skill_vs_repeat_pct ≥ 90 %`.**
3. **BOTH CELLS REACH THE ARRAY, AND ONLY WHERE THEY SHOULD.** V-chain: non-zero RMS difference on
   `targets[0]`, **exactly zero on rows 1…15**, at every anchor. V-mask: scored-step count exactly
   15 against the unmodified 16, with all three baselines on the same 15. Checked at runtime, both
   written into the artifact.

**Which conclusion is expensive.** **C** is: it licenses a defect report against the adapter naming
a concrete replacement and puts a relabel of the corpus on the table. So C carries the material
margin **and** the held-out reading. **D** changes direction and starts one cheap measurement.

**The verdicts**, read on half **B**, primary anchor `d = −2`:

| | condition | reading |
|---|---|---|
| **C** | V-chain clears **L1** on B with a gain over the unmodified cell `≥ 10.0 pp` | the deficit was an anchoring heterogeneity, and the repair is one line of the adapter |
| **D** | V-chain does **not** clear L1 | the diagnosis holds (§2) and the fix does not reach it: the offset is real but not constant, so it does not cancel between consecutive commands either. Licenses measuring the offset's structure — per-joint, velocity-dependent — and nothing else |
| **X** | V-mask's step-0 share of MSE is `< 50 %` | **coherence failure. Nothing is concluded and nothing is licensed, whatever V-chain did.** §2 measured 96.7 % of the jerk sum in index 0 and `horizon_ratio` implies ~92 % of the MSE sum; a number under half would mean this document has misunderstood its own instrument, and no verdict read through a misunderstood instrument is worth having |
| **I** | anything else | indeterminate; nothing licensed |

**Precedence, fixed here** and evaluated in order: **X, then C, then D, then I.**

**Recorded regardless of verdict:** the full 16-element per-step MSE profile for `oracle_action` and
`oracle_state` at both anchors on both halves; `skill_vs_repeat_pct`, `skill_vs_zero_pct`,
`horizon_ratio` and `smoothness_ratio` for unmodified / V-mask / V-chain; all of it under **both**
bench specs, so §5C's bland-side exposure is visible either way; the G0.3 RMS rows; retained chunk
counts; wall time.

## 7. Reading the outcome — decided before the numbers exist

- **C** licenses a defect report against `commanded_to_chunk`'s step-0 anchoring and licenses
  *proposing* a relabel. It does **not** relabel anything, does **not** retro-validate any of the
  fourteen negatives, and is **not** a licence to train. It obliges §3's stated cost — the loss of
  the chunk's tie to measured state — into that report as a design consequence.
- **D** is the outcome most likely to be misread as C, because §2's diagnosis holds under both. D
  says the mechanism is identified and the cheap fix does not reach it. That is genuinely useful and
  licenses exactly one thing: measuring whether the offset is constant per joint or scales with
  velocity and load.
- **§2's measurements license a correction, not an edit.** That `smoothness_ratio` over steps 1–15
  is 0.27 rather than 8.28 is a claim about `src/wam/evaluation/benchmark.py` reaching
  `docs/benchmark.md` and the L4 *moves-like-a-demo* gate. **This pre-registration touches neither
  file.** What the L4 gate should do about a metric that is dominated by an anchor discontinuity is
  a separate, separately pre-registered decision, and it is not one session's to take. What §2 does
  oblige is that no document in this project keeps asserting the *inverted* direction — the command
  is not jerkier than the executed trajectory once the anchor term is removed.
- **No outcome licenses a statement about GR00T or any policy.** No model is trained, loaded or
  consulted; this scores oracles against oracles. PR-07 §6's prohibition is untouched.
- **No outcome unblocks training** after T-39's `VOID`. That remains the project owner's call.
- **No outcome attributes the physics.** A homogeneous anchoring helping is consistent with a
  steady-state tracking offset, with a controller-side feedforward term, and with a recording-chain
  timestamp convention. PR-12 measures an anchoring; it does not identify what imposes the offset.

## 8. Cost

Zero GPU-hours, nothing submitted. CPU only: 3 cells × 2 anchors × 2 halves × 2 bench specs plus
gates, over artifacts already on disk. Expected minutes.

## 9. What must exist before this runs

1. `scripts/probe_step_zero_anchor.py`, importing `commanded_to_chunk`, `raw_anchor_indices`,
   `read_raw_episode`, `ChunkLookupPolicy` and the trim rule from the existing drivers rather than
   re-implementing them. Only the two cells are new.
2. **A test that V-chain and the current anchoring agree exactly under synthetic perfect tracking**
   (`action[i] == q[i+1]`). This is §3's claim 1 made a property of the code instead of a paragraph
   of this document, and it is the most load-bearing test here: if it fails, V-chain is a change of
   premise and this experiment is asking a different question than it says it is.
3. **A test that V-chain differs from the current anchoring on row 0 and is bit-identical on rows
   1…15**, at every anchor — the offline half of G0.3, and the guard against trap 5A.
4. **A test that V-mask's baselines are masked too**: `repeat_mse` and `zero_mse` computed over the
   same 15 steps as `mse`. Trap 5B as code.
5. A test that the per-step profile has exactly 16 entries and is not silently truncated by the
   shortest chunk in the set.

## 10. What this cannot answer

- **Whether a policy could learn the repaired labels.** It scores oracles. Necessary, not
  sufficient, and PR-07 §6 still forbids the policy statement.
- **Whether the offset is constant across joints.** One scalar story is assumed by construction, as
  one scalar `d` was in PR-10 and one scalar `fc` in PR-11. Verdict **D** exists precisely because
  that assumption can fail, and a per-joint fit invented after seeing this grid is what
  pre-registration prevents.
- **What the offset physically is.** §7 lists three sources it cannot distinguish.
- **The gripper.** Withheld by the scorer on every arm so far, for the reason `PR-07-RESULT.md`
  records.
- **Anything about the other twelve `G1_Dex3_*` corpora** (T-043), or about `docs/benchmark.md`'s
  validity beyond the bound `PR-07-RESULT.md` already placed on it.
