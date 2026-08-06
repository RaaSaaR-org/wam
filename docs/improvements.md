# Improvements — state encoder, action head, backbone readout

**Status 2026-08-01.** M0–M4 code-complete (861 tests green). T-16 has run: the Wan2.2-TI2V-5B
LoRA scores WAM-Bench **L0**, and against the causal repeat-last-action baseline it is
**`skill_vs_repeat_pct` −21.8 %** measured the way training fed it (50.6/100) — the widely quoted
**−32.4 % / 48.4** is the same checkpoint scored on a freeze-frame (I-7 below, run 2026-08-01).
Either way the pretrained prior does not clear the bar (`docs/benchmark.md`, `TASKS.md` T-16), and
every route to "video helps" has returned a negative on the same 402 success-only episodes of one
task.

That makes this document's job different from a week ago. It is no longer a list of things to do
after D1 — **it is the list of reasons the negative might not mean what it looks like**, ordered
by how cheaply each can be ruled out. Two of them (I-7, I-3) are deviations between the trained
path and the deployed path, found by reading the code *after* the result came in. **I-7 is now
measured: it was real, worth 10.65 pp, and not enough** — which is the strongest thing this
document has said, because it is the first time one of these hypotheses has been priced.

Each item carries an evidence label:

- **measured** — we ran it, numbers are in `runs/`
- **literature** — published result on someone else's system
- **hypothesis** — reasoned from our code, not yet tested

---

## The through-line: we mean-pool everything

Three separate places collapse the backbone's token grid to a single vector before anything
learns from it:

| Where | Line | What | Status |
|---|---|---|---|
| Joint training | `src/wam/training/joint.py:337` | `pooled = features.mean(dim=1)` → action branch | open (I-2) |
| Inference | `src/wam/decoders/action_head.py:119` | leading dims mean-pooled in `decode()` | open (I-2) |
| Wan probe | `scripts/hf_job_wan_probe.py:328` | `tokens.float().mean(dim=1)` | ✅ **tested** (I-1): pooling is not the limit here |
| Cosmos3 probe | `scripts/hf_job_cosmos3_probe.py:285` | its own mean-pooled extractor | open — but I-1 removed the reason to chase it |

Mean-pooling destroys *where* things are. For pick-and-place, "the cube is at token (7,12)" is
the single most useful item in the feature map, and the average deletes it. Precision is also
the dimension where the entire field collapses on RoboDojo — no policy exceeds ~12 %.

This was the suspected common cause behind most items below. I-1 tested it on the one row we
could test cheaply — the frozen Wan probe — and found **no gain from keeping the geometry**
(details below). That result is specific: frozen features, linear readout. It does not clear the
training-path rows, where the pooling sits in front of a *learned* head that could exploit
positions a ridge cannot. But it does mean the mean-pool is no longer an assumed culprit
everywhere; each row now has to earn its own evidence.

---

## The second through-line: the deployed path is not the trained path

Found while reading the code after the T-16 verdict. In **two** places, `predict()` does
something training never did — and both deviations point at the same measured failure signature.

| # | Trained on | Deployed / evaluated on | Item | Priced? |
|---|---|---|---|---|
| 1 | a real 9-frame window, `frames[indices]` (`training/datasets.py:156`) | **one** frame tiled 9×, `image.expand(num_frames, …)` (`training/joint.py:388`) | **I-7** | **yes — +10.65 pp, still fails** |
| 2 | both `velocity_head` (flow) and `action_head` (regression) (`joint.py:338-339`) | `action_head` only — the flow branch is never sampled | **I-3** | **no — T-30 ran 2026-08-01, all flow arms below L0** |

The T-16 result, on the freeze-frame it was published from, is `skill_vs_zero` **+25.9 %**,
`skill_vs_repeat` **−32.4 %**, `smoothness_ratio` **0.29**. Read as a symptom rather than a score,
that is: *moves in roughly the right direction, far too smoothly, and loses to motion continuity.*
Deviation 1 removes motion from the visual input, which is what a model would need to beat motion
continuity. Deviation 2 is a mean-seeking regressor, which is what produces a trajectory 3.4×
smoother than a demonstration. This ordering was post-hoc — the symptom was known before the causes
were looked for — so both got their own falsifiable run. **Deviation 1's came back: restoring the
motion recovered 10.65 pp of the 32.45 pp gap, about a third, and left the model losing to inertia
by 21.80 pp.** The symptom was real and partly explained by the harness; the remainder is not.

**Neither deviation affects T-15 / T-24 / T-26.** The frozen-feature probes build their own real
multi-frame windows and use a ridge, not `predict()`. "Frozen features carry no action signal past
a state-only ridge" stands independently. What is under review is only the verdict on *trained*
world-action models — T-18 and T-16, which share the `predict()` path.

---

## I-1 · Re-run the frozen-feature probe with a spatial readout

**✅ Done — ran 2026-07-29 on ZeroGPU, 10/10 checks. Outcome: the confound was real but the
verdict survives it.** `runs/wan_probe/2026-07-29-zerogpu-5b-readouts.json`. Result first, the
reasoning that led here below it.

**What we concluded** (T-15, T-24, *measured*): neither Wan2.2-TI2V-5B nor Cosmos3-Nano frozen
features beat the state-only ridge. Wan best block pair joints R² — Cosmos3 0.359 / gripper
0.708 vs. state-only 0.456 / 0.881. Recorded as: "no frozen features beat state-only yet — LoRA
(T-16) carries the burden."

**The confound** (*hypothesis*): the probe measured the features **through a mean-pool**
(`hf_job_wan_probe.py:269`). We did not show that the spatial information is absent from the
backbone. We showed it does not survive averaging. Those are different claims, and we currently
have the weaker one written down as the stronger one.

**The result** (2026-07-29, *measured*): joints, held-out-episode R², block pair chosen on val —

| Readout | Width | val | **test** |
|---|---:|---:|---:|
| `mean` | 3 072 | 0.404 | **0.310** |
| `grid2x2` | 12 288 | 0.424 | **0.370** |
| `rand4` (control) | 12 288 | 0.417 | **0.376** |
| `state_only` | 52 | 0.547 | **0.456** |

Gripper says the same thing more loudly: `state_only` 0.881 against 0.704 for the best readout.

`grid2x2` scores **below** its own random control (0.338 vs. 0.3657 on the val-selected pair).
The step up from `mean` to `grid2x2` looks like progress and is entirely explained by the 4×
width — shuffling the same tokens into meaningless groups buys the same thing. Both report bits
came back false:

```
any_geometry_gain_over_control: false
any_spatial_beats_state_only:   false
```

**So the mean-pool was not the mistake.** The confound was real — we genuinely had not measured
what we claimed — but testing it changed nothing except our confidence. T-15/T-24's verdict is
now checked against the obvious alternative explanation and holds: 52 dimensions of raw robot
state beat 12 288 dimensions of frozen video feature, and not because of how they were pooled.

Cost: 7.6 s GPU (0.079 s/window, 24.6 GB peak). The three readouts share one set of forward
passes, so the extra two were free. Token geometry verified in-run: `S=96, grid (2, 6, 8)`.

**What it does not license.** This is about *frozen* features under a *linear* readout, on one
task with 96 windows and a 12-episode split. It says nothing about whether a fine-tuned backbone
carries action signal (T-16), nor whether a non-linear head could find what a ridge cannot. It
also does not retire I-2: the training path still mean-pools, and a cross-attention head is
motivated by literature, not by this measurement.

---

**How it was built.** `scripts/hf_job_wan_probe.py` grew a `--readout`
flag; everything else (windows, labels, episode split, ridge code) is untouched, so the output
is directly comparable to `runs/wan_probe/` and `runs/cosmos3_probe/`.

Three readouts, all fitted on the *same* forward passes:

| Readout | What it does | Width (D = 3072) |
|---|---|---|
| `mean` | the historical mean-pool, byte-for-byte | 3 072 |
| `grid<R>x<C>` | average-pool the token grid into R×C cells, keep them separate | R·C · 3 072 |
| `rand<N>` | pool the same tokens into N equally sized **random** groups | N · 3 072 |

`rand<N>` is what makes the comparison worth anything. A coarse grid has more dimensions than
a mean-pool, and more dimensions alone can raise a ridge R². The random control has the
*identical* width and group sizes with geometry removed, so:

- **grid > rand** → position carries action signal, and the mean-pool was hiding it
- **grid ≈ rand** → we only bought dimensions; the recorded verdict stands and gets stronger

The Wan geometry is known: 5 context frames at 192×256 through the TI2V-5B VAE (spatial 16,
temporal 4) and patch (1, 2, 2) give a **(F'=2, H'=6, W'=8)** token grid — 96 tokens, which is
why `grid2x2` (cells of 3×4 tokens) and its `rand4` control are the default pair. `token_grid()`
on the adapter is the single source of truth and the probe asserts S against it before any
reshape, because a silent mismatch would make every spatial number meaningless.

Run it on the free ZeroGPU Space (`scripts/deploy_wan_space.py`, "readout probes" tab, the
readout box is blank = default), or:

```bash
uv run scripts/hf_job_wan_probe.py --source /model --data-dir data/raw/gr00t_apple \
    --readout mean,grid2x2,rand4
```

Read `info.probe.readout_comparison` in the report: `any_spatial_beats_state_only` and
`any_geometry_gain_over_control` are the two bits this experiment exists to produce. The
primary readout (`mean`) still occupies `info.probe` unchanged, so the recorded numbers stay
the anchor rather than being overwritten.

**Why it mattered:** the decision rule was written before the run, and both branches were worth
having. Had the spatial readout beaten state-only, T-16's premise would have changed and OD-04
would have deserved a second look. It lost, so LoRA still carries the burden — and now for a
reason we tested rather than one we assumed.

**Still open:** the Cosmos3 probe (`scripts/hf_job_cosmos3_probe.py`) has its own extractor and
still hands `analyze_probes` a mean-pooled array — which the legacy path accepts, so it keeps
working and keeps reporting a single `mean` readout. Re-running T-24 spatially needs the MoT
token grid worked out the same way. **Not worth doing:** the condition for it was Wan's spatial
readout moving the number, and it did not.

---

## I-7 · Give the policy a frame history at inference — the T-16 eval measured a still

**Evidence: measured, both the mismatch and its effect.** The mismatch is in the code (both sides
quoted below); its effect on the verdict was hypothesis until 2026-08-01 and is now **+10.65 pp of
`skill_vs_repeat_pct`, which does not reach the gate**. Cheapest item in this document, and the
only one that has moved a recorded number.

> **Status 2026-08-01: RUN. The mismatch was worth 10.6 pp and the bar is still not cleared.**
> `cluster/discoverer/61_eval_t29_frame_history.sbatch`, job 184648, both arms on one H200 from
> one checkpoint (`runs/t16-lora-seed0/checkpoints/step-020000`), same 40-episode proven holdout,
> same 1 040 chunks, differing in exactly one flag. Artifacts:
> `runs/t16-lora-seed0/eval-t29-{tiled,history}/`.
>
> | metric | tiled (as published) | real window | delta |
> | --- | --- | --- | --- |
> | **`skill_vs_repeat_pct`** (the L1 gate) | **−32.45 %** | **−21.80 %** | **+10.65 pp** |
> | `ci_skill_vs_repeat_pct` (L2, critical chunks) | −50.74 % | −23.11 % | **+27.64 pp** |
> | `skill_vs_zero_pct` (L0) | +25.88 % | +31.83 % | +5.96 pp |
> | `mse` | 1.21027e-05 | 1.11298e-05 | −8.0 % |
> | level | L0 | **L0** | unchanged |
> | score, spec 0.1.0 | 48.4 | 50.6 | +2.2 |
> | score, spec 0.2.0 | 28.4 | 30.6 | +2.2 |
>
> The tiled arm reproduces `eval-latest` to every digit, so the published −32.4 % is confirmed to
> be the freeze-frame measurement and the A/B is clean. **Everything moved in the direction I-7
> predicted, and nothing crossed a gate.** The largest move is on L2's task-critical chunks
> (+27.6 pp) — where the demonstration is actually moving is exactly where deleting the motion
> hurt most, which is the mechanism I-7 describes rather than a generic improvement.
>
> Local equivalent and the full runbook: `docs/local_gpu.md` §3b. What shipped:
> `Observation.image_history` (optional, `INTERFACES_VERSION` 0.3.0),
> `wam.data.episode.frame_window_indices` as the **one** definition of the window that both
> `EpisodeDataset` and `build_eval_pairs` now call, `resolve_frame_context` shared by both
> `predict()` implementations, and `eval_t16.py --frame-history` — **off by default**, so the
> archived runs stay reproducible and the A/B is an explicit experiment. Verified bit-identical
> to the pre-T-29 path on real chunks from `d1-full-gen-seed0`.

`EpisodeDataset` hands the model the `num_frames` frames *ending at* the chunk timestamp:

```python
frame_idx = max(int(np.searchsorted(frame_ts, ts, side="right")) - 1, 0)
lo = frame_idx - self.num_frames + 1
indices = np.clip(np.arange(lo, frame_idx + 1), 0, frames.shape[0] - 1)   # datasets.py:154-156
```

`JointWorldActionModel.predict` hands it the same frame nine times:

```python
frames = image.unsqueeze(0).expand(self.config.backbone.num_frames, -1, -1, -1)   # joint.py:388
```

So a backbone trained to read a moving clip is evaluated on a freeze-frame repeated nine times.
Both world-action numbers on record (T-18 tiny, T-16 LoRA) were produced this way — `eval_t16.py`
and `run_ablation.py` both go through `build_eval_pairs` → `evaluate_policy` → `policy.predict()`.

**Why this is not a rounding error.** The baseline the model must beat is repeat-last-action,
which is nothing but motion continuity. A static clip carries no velocity at all, so the only
motion evidence left is `dq` from the state vector — the very channel the action-only baseline
already has. Whatever the video branch might contribute about *where things are going*, the
evaluation deleted before asking. This is not a new discovery so much as an unjoined dot: the
`predict()` docstring already calls the tiling "a real limitation … a rolling frame buffer is the
follow-up" (`joint.py:376-379`). Nobody connected it to the number.

**Scope.** `Observation.images` is documented as one `HxWxC` array per camera
(`interfaces/protocols.py:27-30`), so this is a versioned-interface question, not a one-liner.
Two shapes, and they are not exclusive:

- *Offline (does the verdict change?)* — **done.** The frames were already in hand:
  `build_eval_pairs` reads the whole episode, so `num_frames=` now attaches the window
  `EpisodeDataset` selected, through the same `frame_window_indices`. `Observation` grew an
  optional `image_history` so `evaluate_policy` stays policy-agnostic; the invariant that its last
  frame **is** `images[key]` is what makes the field safe to be optional, and it is checked rather
  than trusted. Robot-facing code untouched.
- *Closed loop (does the robot get one?)* — **not done, deliberately.** Needs a rolling buffer,
  because `ClosedLoopExecutor` supplies one render per cycle. That makes the policy stateful, which
  drags in a reset-between-episodes rule and a defined answer for the first N cycles (pad by
  repeating — today's behaviour, but only at startup rather than forever). It changes the deployed
  path, so it gets its own review, and only if the offline run says it matters.

**Cost:** one eval pass over the 40 holdout episodes — minutes, ~0.2 GPU-h, **no retraining**. The
checkpoint is unchanged; only what we show it changes.

**Decision rule, fixed before the run.** Re-score `runs/t16-lora-seed0` with the true frame window,
identical split, identical bench code:

- `skill_vs_repeat_pct` **moves materially toward or past 0** → the T-16 verdict was measured out
  of distribution and must be re-stated; T-18's too, and `docs/benchmark.md` needs a correction
  rather than an addendum. AC-07 goes back to open.
- **essentially unchanged** → the negative gets substantially stronger: the model had the motion
  available and still lost to inertia, which is a claim about the model rather than the harness.
  Then the honest bottleneck really is data, and I-8 is next.

Either way the result is worth more than the run costs, which is why it is item 1.

**The rule above is left exactly as it was written, because it was wrong in a way worth keeping.**
It has two branches. The outcome had three. `skill_vs_repeat_pct` moved 10.65 pp toward 0 and
stayed 21.80 pp short of it — "materially toward 0" *and* "still fails L1" at the same time, which
the prose admits no verdict for. The executable copy in
`cluster/discoverer/61_eval_t29_frame_history.sbatch:183-194` (`:143-154` before that file's own
result block was prepended on 2026-08-01) was the only one of **six** written
copies that anticipated this case: it splits on a numeric `b − a > 5.0` and routes the middle to
"the negative stands but its size does not — re-state both runs with the real window, then go to
I-8". The five prose copies (here, `61_`'s own header, `TASKS.md`, `docs/local_gpu.md`,
and the if/then restatement at the foot of this file) all collapse the middle into the top branch
and say **AC-07 reopens**.

**Picking the executable copy after seeing the data is exactly the move pre-registration exists to
stop**, so it is recorded here as a judgement call and not presented as the rule having spoken.
What makes it safe to make: the two readings differ only in a *label*. Both agree L1 fails, both
send the next GPU-hour to the same place, and the work below is the intersection of both — done
regardless of which copy governs.

**AC-07 is nevertheless back to open, for a reason no copy of the rule anticipated.** AC-07 is a
comparison: does the world-action model beat the action-only baseline? That comparison now reads
T-16 at −21.80 % (real window) against `d1-full-gen-seed0` at −20.9 % and `t18-real-ablation-seed0`
at −129.0 % — **both still freeze-frame numbers.** Only `t16-lora-seed0` was re-measured. A
three-run table with one run scored in a different mode from the other two is not a comparison, and
the 11.5 pp gap that made T-16 look clearly worse than the action-only baseline is now 0.9 pp
*across a mode boundary*, which is to say unknown. So AC-07 is undetermined pending the re-score,
which is materially "open" — arrived at through the mixed-mode problem rather than through either
branch.

**RESOLVED 2026-08-01 — the re-score ran, on a laptop CPU, for zero allocation.** It needed no GPU
slot at all: both archived checkpoints are ~0.9 MB (the 16 MB in that directory is
`predictions.jsonl`, not the weights), and a full 1 040-chunk pass takes about 7 s.
`scripts/rescore_archived.py` does it without retraining. The ladder is single-mode again:

| run | backbone / branch | tiled | real window | delta | level |
|---|---|---|---|---|---|
| `t16-lora-seed0` | Wan2.2 LoRA, world-action | −32.45 % | **−21.80 %** | **+10.65 pp** | L0 |
| `d1-full-gen-seed0` | tiny, action-only | −20.86 % | **−20.88 %** | −0.02 pp | L0 |
| `t18-real-ablation-seed0` | tiny, world-action | −129.04 % | **−129.00 %** | +0.03 pp | below L0 |

**The confound is backbone-specific, and that is the finding.** Showing the model a frozen frame
instead of the real window costs the Wan fine-tune 10.65 pp and costs the two `tiny` runs
**nothing** — 0.02 and 0.03 pp, which is noise. Both directions, so not even a consistent bias. The
`tiny` backbone does not use the frame axis at all; whether it sees motion or a still is invisible
in its output. The Wan backbone does. That is the first *positive* evidence in this project that
the pretrained video prior carries temporal information the action head can reach — arrived at
sideways, as a control for a measurement error, which is where it is worth the least to have
guessed and the most to have measured.

**It also means the I-7 correction never threatened the tiled AC-07 numbers.** They were re-measured
and did not move. The verdict below stands on in-distribution numbers now, not on freeze-frames.

The trustworthy part is the control: run in tiled mode, `rescore_archived.py` reproduces the
archived `predictions.jsonl` at max |Δ| = 0 (bit-identical) for `d1-full-gen-seed0` and 8.94e-08 for
`t18-real-ablation-seed0`, and returns −20.86 % and −129.04 % — the archived values to every printed
digit. A re-score that could not reproduce the old number would not be evidence about the new one.

**What AC-07 now reads, in one mode.** The clean ablation is the *same-backbone* pair, which is what
`run_ablation.py` was built to produce: `t18-real-ablation-seed0` (tiny, world-action) at −129.00 %
against `d1-full-gen-seed0` (tiny, action-only) at −20.88 %. Adding the world branch to the tiny
backbone costs **108 pp** and drops the run below L0 entirely. That verdict is now measured in
distribution and is not a harness artifact.

T-16 against that baseline — −21.80 % vs −20.88 %, a 0.92 pp gap — is **not** a clean ablation and
must not be read as one: it differs from the baseline in the backbone *and* in the world branch at
once. What it does support is a weaker, still useful statement: the pretrained prior recovers
essentially all of the 108 pp the world branch costs on `tiny`, landing back level with the
action-only baseline — and buys no advantage over it. On AC-07's actual question, *no measurable
world-action advantage*, the answer is unchanged. What changed is that it is now an answer about
models rather than about frame windows.

**One diagnostic that is not yet a claim.** `shift_tolerant_mse` is 8.33e-06 for the real-window
arm, below repeat-last-action's 9.14e-06 `mse` — i.e. allowing a ±1-step shift takes the model
past the baseline it otherwise loses to, which is what "right content, wrong placement in time"
would look like and is the same shape as T-30's `FLOOR_MSE` hypothesis. **It is not evidence yet**:
the ±1-step allowance is computed for the model only, and the baselines were never given it, so the
two numbers are not comparable. Giving the baselines the same allowance is a bench-code change
(and therefore a spec version), not a re-read of these artifacts.

---

## I-2 · Replace mean-pool with cross-attention in the action head

**Evidence: literature + hypothesis.** Best value/effort ratio of the architectural changes.

Today `ActionHead` consumes one pooled vector. The standard alternative is action queries that
**attend into** the feature tokens, so the head can look at the region it needs per step.
AHA-WAM does exactly this ("observation-guided context routing"); it is also why its action
expert can be a DiT rather than an MLP.

**Scope:** changes only the head. `ActionDecoder` protocol, `ActionChunk`, the safety layer and
every robot adapter stay untouched — `decode()` would take `[*, F]` and *not* pool it.

**Ordering note:** do this before I-3. Cross-attention also improves the flow branch, since
`ActionVelocityHead` consumes the same `pooled` (`joint.py:338`).

**Caveat:** at D1 scale this may not show up at all, or may hurt — more capacity on a tiny
dataset. Gate it through `scripts/run_ablation.py` on the same 362/40 split as T-18, not on
intuition.

---

## I-3 · Make the flow branch the deployed path

**Evidence: literature — plus, since T-16, a symptom of our own.** Promoted from "not urgent".

> **RAN 2026-08-01, Slurm 184670, 33 min, exit 0 — and returned NO RESULT, by pre-registered
> refusal.** Nine arms on one H200, one checkpoint, the proven 40-episode holdout, 1 040 chunks
> each, all in `--frame-history`. **Every flow arm tripped the sampler-broken guard**, including
> `B_mean` at n=32, the arm `T30_RULE_V2` keys on. The rule's consequence is explicit: *record
> nothing, this is an integration bug and not a result.* So I-3's question — does reading the
> action out of the flow branch beat the regression head — remains **unanswered**. What the job
> established is that the sampler is broken, which is a different and cheaper thing to fix.
>
> | readout | mse | `skill_vs_repeat_pct` | RMS / demo | |
> |---|---|---|---|---|
> | regression (deployed) | 1.113e-05 | −21.80 % | 0.74 | |
> | flow n=1 | 0.1529 | −1 673 733 % | 96.79 | `[SUSPECT]` |
> | flow n=4 | 0.00122 | −13 253 % | 8.64 | `[SUSPECT]` |
> | flow n=16 | 3.154e-04 | −3 352 % | 4.40 | `[SUSPECT]` |
> | flow n=32 | 3.130e-04 | −3 326 % | 4.38 | `[SUSPECT]` |
> | flow n=64 | 3.124e-04 | −3 319 % | 4.38 | `[SUSPECT]` |
> | flow mean8 (**keyed arm**) | 1.240e-04 | −1 257 % | 2.75 | `[SUSPECT]` |
> | flow warm0.6 | 1.454e-04 | −1 492 % | 2.97 | `[SUSPECT]` |
> | flow warm+mean8 | 1.459e-04 | −1 496 % | 2.97 | `[SUSPECT]` |
>
> guard: `mse > 5 × zero-delta (1.633e-05)` or `RMS > 3 × demos' 0.004041`.
>
> **Three things point at the same cause, and none of them is step size.** The sweep has
> *converged* — n=16/32/64 give 3.1542e-04 / 3.1305e-04 / 3.1245e-04, so doubling the steps twice
> moves the fourth significant figure. The converged RMS is **4.38× the demonstrations'**, so it
> lands on the wrong scale, not merely in the wrong place. And the warm start, which begins
> integration at t0 = 0.6 from the regression head's own chunk — a 1.11e-05 estimate — comes out at
> 1.454e-04, i.e. **13× worse than where it started**. An integrator handed a good estimate that
> actively degrades it is applying the velocity field with the wrong sign, scale, or time
> parameterisation. That is a direction/scale bug, and it converges perfectly well to a wrong
> answer, which is exactly why the converged sweep is not the reassurance it first looks like.
>
> **The guard earned its place here.** Without it the honest-looking reading of −1 257 % is "the
> flow branch is catastrophically worse than the regression head", which would have retired I-3 and
> sent the next months to I-2 (a new cross-attention head, days plus a retrain) on the strength of
> a sampler bug. Pre-registering "this pattern means broken, not worse" before seeing the pattern
> is what stopped that.
>
> `tests/test_training.py::TestFlowSampler` and `::TestFlowSamplerControlArms` claim to pin the
> direction, the conditioning and the warm start. They pass. At least one of them therefore pins
> the sampler against itself rather than against the training path — a test that is green while the
> thing it covers is broken is its own finding, and it is being chased down now.
>
> **Cost:** 0.56 GPU-h against ~6 budgeted. The arms are cheap; it was the interpretation that was
> expensive, and the guard is what made it survivable.
>
> ---
>
> **CORRECTION, same day, after the audit: there is no integration bug. The guard's ACTION was
> right and its DIAGNOSIS was wrong, and the two have to be scored separately.**
>
> The block above reasons its way to "direction, scale or time parameterisation" and calls the warm
> start the strongest clue. That inference was wrong, and the mistake is instructive: I compared
> `warm0.6` (1.454e-04) against the regression head's 1.11e-05 and concluded the integrator degrades
> a good estimate 13-fold. But `warm0.6` does not start from a good estimate — it starts from
> `(1-t0)·noise + t0·init`, i.e. the regression chunk with **40 % noise re-injected**. Against the
> arm it should be compared with, the plain single draw at 3.130e-04, the warm start is **better**,
> not worse. The integration was helping the whole time. I read a control arm against the wrong
> reference and got a sign out of it.
>
> All three suspects the verdict block named are provably clean, each checked against the *training
> path* rather than against a restatement of it:
>
> | suspect | check | result |
> |---|---|---|
> | direction | sampler's `z` at each `t_k` vs `make_flow_targets(noise, clean, t_k)` | max abs diff **2.98e-07** (fp32 rounding); final `z` == `x1` |
> | pooled conditioning | tensor reaching `velocity_head` from `predict` vs from `co_denoise` at t=1 | **bit-identical**, max diff 0.0 |
> | warm start | `(1-t0)·noise + t0·init` vs `make_flow_targets(noise, init, t0)` | identical |
> | decode side | `action_recon(action_encoder(chunk))` on the real checkpoint | mse **8.1037e-07** — exactly the pre-registered ceiling 8.104e-07 |
>
> **The real cause is the velocity field, and it is architectural.** `ActionVelocityHead`
> (`joint.py:217-235`) takes the timestep as **one raw scalar** concatenated beside 32 latent and
> 3 072 feature inputs — `in_dim = latent_dim + feature_dim + 1`, no sinusoidal embedding, no step
> index. First-layer weight-block Frobenius norms on the trained checkpoint: **t 1.68, latent 23.3,
> features 51.0**. The measured `−∂v/∂z` is **flat in t** at every feature scale probed, where a
> straight-path flow needs a gain of `1/(1-t)` — 1 at t=0, 33 at t=0.97. A constant-gain linear
> field contracts by a fixed factor per step and **cannot reach zero at any step count**.
>
> That single fact predicts all four observations, which is why it is the explanation and the
> integration story was not: the sweep converges (fixed contraction has a fixed fixed-point), the
> scale is wrong by a constant, the residual is zero-mean (so `mean_of` averages it down ~2.5×), and
> more steps buy nothing. Quantitatively: a chunk latent's *content* has per-element std 0.049 while
> the flow starts from `N(0, I)`, so the sampler must delete ~95 % of a unit-variance vector to
> reach the demonstrations' RMS. It deletes ~90 %. The guard's `RMS > 3 × demos` line sits at ~91.5 %
> removal. **T-30 failed its guard by about one and a half percentage points of noise removal.**
>
> So the guard was right to refuse — a run where the sampler cannot reach the data manifold says
> nothing about whether the flow readout beats the regression head — and wrong about why. Its
> instruction ("check the direction, the conditioning, the warm start") sent the audit at three
> things that were already correct. A guard that names a cause is doing more than a guard should;
> the condition earned its keep, the diagnosis did not, and the fix is to have it say *broken* and
> stop there.
>
> **This is a better outcome than the verdict would have been.** I-3 asked whether the flow branch
> is a better readout. The answer is that, as built, it cannot be *sampled* — the head has no way to
> know what time it is. That is a concrete defect with a concrete fix, found for 0.56 GPU-h.
> One caveat carried forward, because it is the part that is not yet solved: at tiny scale, adding a
> Fourier timestep embedding does **not** fix it — the head still learns the flat, L2-optimal
> time-blind gain. So this is a head *and objective* problem, not a missing-embedding problem, and
> the next move (retrain the head alone on the frozen encoder/recon, with both a timestep embedding
> and a readable step index) is a hypothesis, not a known fix.
>
> **The tests were green and blind, and that is named.** `TestFlowSampler` and
> `TestFlowSamplerControlArms` never call `co_denoise` or `make_flow_targets`; they check the
> sampler against `_StraightPathField`, a copy of the convention re-written as a literal in the test
> file, and against a hand-rolled copy of the sampler's own loop. Verified by mutation:
>
> | mutation | old 28 tests | new class |
> |---|---|---|
> | `co_denoise`'s flow convention **inverted** | **28 passed** | 2 failed |
> | head trained on `1 − t` | **28 passed** | 1 failed |
> | `co_denoise` pools `features[:, 0]` not `.mean(dim=1)` | **28 passed** | 2 failed |
> | sampler integrates backward | 2 failed | 1 failed |
> | warm-start mix flipped | 1 failed | 1 failed |
>
> The three failure modes the verdict block named as suspects are exactly the three the old tests
> cannot see. They passed because the sampler is correct — but they would have passed had it not
> been. `TestFlowSamplerAgainstTheTrainingPath` computes every expectation *from* the training path
> and catches all five.

We already train two action paths:

- `velocity_head` — rectified flow on action latents, co-denoised with video, shared `t`
  (`joint.py:338`)
- `action_head` — direct regression from pooled features (`joint.py:339`)

The **deployed** path is the regression head: `ActionOnlyModel.predict` →
`action_head.decode(features[0])` (`src/wam/training/action_only.py:152`).

Single-shot MSE regression is mean-seeking. Where two valid actions exist (grasp the cube from
the left or the right), the regressor outputs the average — reaching between them, which is
invalid. This is the standard argument for iterative action heads and why π0, π0.5 and
GR00T-N1.x all ship a flow-matching action expert.

**What it needs:** a sampler on the existing velocity head, plus a latency budget check. The
executor floor is ≥2 Hz (`ExecutorConfig.min_policy_rate_hz`, PRD §11.1) and the deadline is
500 ms — n denoising steps must fit. AHA-WAM reaches ~57 Hz with a comparable structure, so the
budget is not obviously a problem, but it is the thing that decides feasibility.

**What changed on 2026-07-30.** The argument above was purely a priori: mean-seeking regression
*should* average over valid alternatives. T-16 then produced the matching symptom — a
`smoothness_ratio` of **0.29**, i.e. predictions 3.4× smoother than the demonstrations they are
scored against, together with a positive `skill_vs_zero` and a negative `skill_vs_repeat`. A
model outputting the blandly-averaged trajectory of the task is what that combination looks like.
The earlier note said multimodality is a D2/scale problem; the evidence now says the *smoothing*
shows up at D1 scale, whether or not the cause is multimodality.

**Cheap version first.** The velocity head is already trained in every T-16 checkpoint and simply
never sampled. Adding a sampler and re-scoring `runs/t16-lora-seed0` needs **no retraining** — the
same "re-score what we already have" move as I-7, and it can share that eval pass. It was gated
behind I-7, since a static-clip conditioning signal would handicap both heads equally and confound
the comparison. **That gate is now cleared** (I-7 ran 2026-08-01), and it settled which mode the
comparison happens in: every T-30 arm runs with `--frame-history`, because that is the mode the
model was trained in and the mode in which its remaining deficit is 21.80 pp rather than 32.45 pp.
Running the readout experiment inside the freeze-frame would have handed the flow branch a
harness defect to overcome as well as a readout one.

**Then the latency question**, which decides whether it can ship: the executor floor is ≥2 Hz
(`ExecutorConfig.min_policy_rate_hz`, PRD §11.1) with a 500 ms deadline, so n denoising steps must
fit alongside the backbone pass. AHA-WAM reaches ~57 Hz with a comparable structure, so this is a
budget to measure, not an obvious wall.

**Measured 2026-08-01 on the real checkpoint, before writing any sampler.** On
`runs/t16-lora-seed0/checkpoints/step-020000` against the 1 040 archived holdout chunks:

| path | chunk MSE | RMS \|targets\| |
|---|---|---|
| deployed regression head | 1.21027e-05 | 0.00226 |
| `action_encoder` → `action_recon` round-trip | **8.10372e-07** | 0.00412 |
| the demonstrations | — | 0.00404 |
| repeat-last-action (the L1 bar) | 9.14e-06 | — |

The action **latent** carries the chunk to within 8.1e-07 — 15× better than the deployed readout
and 11× better than the bar T-16 failed. Whatever T-16 measured, it was not an encoder that cannot
represent these actions. The single-shot readout discards it and under-shoots magnitude by 44 %.
That is what makes I-3 a *readout* experiment rather than a retrain, and it also sets the two
pre-registered anchors in `63_eval_t30_flow_head.sbatch`: `CEILING_MSE` (the round-trip, what a
perfect sampler would reach) and `FLOOR_MSE` **1.68201e-05** — the score of a chunk with the right
content and no position, because step index is recoverable from the latent at ~100 % accuracy and
`ActionVelocityHead` takes no step index. Landing near the floor is the *expected* outcome of an
order-blind sampler, not a bug, which is why the rule reads it as a mechanism only when the number
actually lands there.

**"Consistent with mean-seeking", not "the signature of it."** An earlier version of this entry,
and of `docs/benchmark.md`, claimed the smoothness/magnitude pair diagnosed a mean-seeking head.
It does not. Ruled out: a bounded-output artifact (max `|target|` 0.0192 against a `tanh`;
`limit_penalty` bites only outside ±0.95). **Not** ruled out, each sufficient on its own:

- **the one-sided jerk regulariser.** `configs/training/joint_wan_gr00t.yaml` sets
  `weights.smoothness = 0.01`, and `JointTrainer.compute_losses` applies `smoothness_loss` to
  `decoded_targets` — the *regression* head's output — and to nothing else. The flow branch is
  never charged for jerk, so a smoothness gap between the readouts is the objective's own
  consequence. This confound applies to the **positive** branches of T-30 as much as the negative
  ones, which is a correction to the first draft of that rule: it held the negatives to an
  alternative reading and the positives to none.
- **plain L2 shrinkage** toward zero under a small-target distribution, with weight decay on top.

Separating any of these from mean-seeking needs an intervention — retrain at
`weights.smoothness = 0`, or a genuinely multi-modal task — not a readout swap. T-30 answers the
narrower question a readout swap *can* answer: does the branch trained as a distribution carry
anything the regression head does not?

**The sampler's own confound, named in the rule rather than discovered afterwards.** Training
co-noises video and action at a **shared** `t` (`co_denoise`). `sample_action_chunk` computes
**one** backbone pass at `t=1` on the clean observation and reuses those features at every `t_k`,
so near `t_k=0` the velocity head is asked about (near-pure-noise latent, clean-video features,
t≈0) — a combination it never saw. That is I-7's structure again, one level down. The faithful
sampler (n backbone passes, observation noised to each `t_k`) is deliberately **not** run: it costs
n backbone passes and destroys the observation the readout exists to read. So **no branch of T-30
refutes the flow branch as trained** — it is bounded by a warm-start arm (`--flow-t0`, integrating
only where the head's `t` and the features' `t=1` roughly agree) rather than removed.

**And the metric penalises the thing under test.** `skill_vs_repeat_pct` is MSE-derived, and for
any calibrated conditional `E‖a − draw‖² = E‖a − mean‖² + E‖draw − mean‖²`: a single *unbiased*
draw scores worse than the conditional mean by exactly the conditional variance. Scoring one draw
against a mean-seeking regressor would reward the defect under test. The rule therefore keys on a
mean-of-8 arm — a **measurement**, never a deployment candidate — and gates deployment separately
on the single-draw arm, since averaging draws re-introduces the mean-seeking the flow branch
exists to avoid.

**Ready to run:** `sbatch cluster/discoverer/63_eval_t30_flow_head.sbatch` (T-30). Ten arms, one
GPU, ~6 h; the rule is `T30_RULE_V2`, in git before any arm ran, with V1's three defects recorded
in the file rather than silently edited out.

### DIAGNOSED 2026-08-01 — three defects, not one, and the recorded one is the smallest

All measured on CPU against archived artifacts. No GPU, no allocation. `GAIN_RULE_V1` was written
down before any of it was measured.

**D3 — the head's gain is flat in `t`. CONFIRMED, and this is the defect already in this file.**
`scripts/probe_velocity_head.py` on `step-020000` measures the Jacobian `−∂v/∂z` at 4.8590 (t=0),
4.8584 (t=0.5), 4.8527 (t=0.9375) — **constant to four significant figures.**

> **CORRECTION, same day.** An earlier version of this section — and the `1 at t=0, 33 at t=0.97`
> in `sample_action_chunk`'s docstring (`joint.py:464`) — said the required gain is `1/(1−t)`,
> running 1 → 32 over the grid. **That holds only if `p(x1|c)` is a point mass.** For a residual
> posterior std σ the Bayes gain is `((1−t) − tσ²) / ((1−t)² + t²σ²)`, which is bounded and
> **non-monotone** — it peaks and then falls. At the measured content scale:
>
> | t | `1/(1−t)` (as claimed) | σ=0.049 | σ=0.0454 |
> |---|---|---|---|
> | 0.875 | 8.00 | 7.04 | 7.16 |
> | 0.9375 | 16.00 | 10.01 | 10.59 |
> | 0.96875 | **32.00** | **8.96** | **10.05** |
>
> Max ideal gain is **10.2 at t≈0.949**, not 32, and there is **no pole**. The defect is real —
> 1.4 measured against ~10 ideal — but roughly 3× less extreme than this file claimed, and a
> different shape. σ=0 is also the one assumption that cannot be granted here, since a
> point-mass conditional is precisely the case in which the flow branch has no reason to exist.

Per-column first-layer norms are latent 4.12, **t 1.68**, feats 0.92 — the `t` column carries
*more* weight than an average feature column, so `t` is not underweighted.

**A competing explanation fits the same data better, and it is not an architectural defect.**
The probe holds the features fixed and sweeps the explicit `t` input: the gain does not move. But
sweep the *feature scale* instead and it moves a great deal — 4.859 / 3.683 / 2.501 / 1.434 /
0.605 at scales 0.1 / 0.5 / 1 / 2 / 5. So the head's gain is set by **feature magnitude**, not by
its `t` column.

That is exactly what a head would learn if it read the noise level off the features rather than
off `t` — and during training it could, because `co_denoise` noises video and action at a
**shared** `t`, so the 3 072-dim feature vector carries `t`. At inference `sample_action_chunk`
computes **one** backbone pass at `t=1` on the clean observation and reuses it at every `t_k`
(`joint.py:470-480`), so the head sees "t=1 features" at every step and emits a constant gain.

**This confound was already documented in that docstring as the sampler's known weakness; what is
new is that it now also explains the flat gain, with no architectural defect required.** The two
explanations are not distinguished by anything measured so far, and the probe cannot separate them
by construction — it varies `t` with features held fixed, which is the inference condition, not
the training one. Distinguishing them needs the faithful sampler (features recomputed per `t_k`),
which costs n backbone passes and destroys the observation, or a head trained with features whose
`t` content is ablated.

The flat-gain model then predicts the archived step sweep. One free parameter, fitted on the n=64
arm alone, mapping latent error to action MSE via `k = 1.68201e-05 / 0.05583²`:

| n | predicted | measured | |
|---|---|---|---|
| 1 | 9.01705e-04 | 1.52949e-01 | **missed** |
| 4 | 1.67360e-04 | 1.22018e-03 | **missed** |
| 16 | 2.82611e-04 | 3.15420e-04 | −10 % |
| 32 | 3.02478e-04 | 3.13045e-04 | −3 % |
| 64 | — | 3.12448e-04 | *fitted* |

Gain fitted end-to-end on the holdout: **1.4088**. Gain measured directly from the checkpoint
tensors: **1.434**. Independent routes, agreeing to 2 % — that is the load-bearing agreement, and
it is the reason to believe the mechanism.

**Do not oversell that table.** n=16/32/64 are already converged to one another, so "predicting"
them is largely predicting flatness. And the model misses n=1 and n=4 outright: at dt=1 and
dt=0.25 a single Euler step from unit noise leaves the regime a local linearization describes.

**D1 — the action latent is 99.85 % a constant, by variance. NEW.**
`scripts/check_action_latent.py` on the same checkpoint: between-step centroid std **1.422**,
within-step (content) std **0.05583**, nearest-centroid step accuracy 1.0000. The content is
`0.0558²/1.422² = 0.15 %` of the latent variance, and the flow is trained to transport `N(0, I)`
onto that — so **99.85 % of the transport work is reproducing a fixed pattern** and the 0.15 %
carrying the answer is the remainder. Nothing in WAM normalizes the action latent; SD's 0.18215
scale factor exists for precisely this.

**D2 — the head cannot identify the chunk position at small `t`. Structural.** One MLP is applied
to all 16 chunk positions and takes **no step index** (`joint.py:230-235`). At small `t`, `z_t` is
near-pure noise and identically distributed across positions, so the head cannot know which of the
16 centroids (spread 1.422) to move toward. Position becomes recoverable only once
`t·1.422 > (1−t)·1.0`, i.e. **`t > 0.416`**. Below that the correct field is *not representable*,
and no encoding of `t` helps, because the missing information is not `t`.

This was found analytically and then reproduced independently: a synthetic testbed built to
compare head architectures had *every* arm fail, including the analytically-motivated ones,
because it had faithfully reproduced D2. The arms were at their architecture's ceiling, not
underfitting. An oracle control confirmed the harness itself is exact — fed the true field, the
sampler lands on `x1` to 1e-15.

### Ranked by a synthetic ablation — and it refuted the ranking I had pre-registered

A CPU testbed reproduces the measured latent geometry exactly (positional std 1.422, content std
0.05583, no step index) and ablates the three defects independently. An oracle control confirms
the harness: fed the true field, the sampler lands on `x1` to 1e-15. 4 000 steps, feat 512.

| arm | latent MSE | vs content-free floor |
|---|---|---|
| shipped (raw scalar `t`, no step index) | 3.597 | **1147×** |
| `+pos` | 2.087e-02 | 6.66× |
| `+pos +fourier` | 5.413e-03 | 1.73× |
| `+pos +film` | 5.674e-03 | 1.81× |
| `+pos +film +xpred` | 3.840e-03 | **1.23×** |
| `+pos +film +NORM` | 5.309e-03 | 1.69× |
| `+pos +film +xpred +NORM` | 5.724e-03 | 1.83× |

**`NORM_V1` — pre-registered, and REFUTED.** The written prediction was that arms *without*
normalization would score at or above the content-free floor and arms *with* it would score below.
No arm beat the floor, and normalization changed almost nothing (5.674e-03 → 5.309e-03; it made
the `xpred` arm *worse*). **D1 is not the dominant defect, and the priority order this file carried
for part of 2026-08-01 — normalization first — was wrong.** Recorded, not retrofitted.

**D2 is the dominant lever: 3.597 → 0.0209, a 172× improvement from the step index alone.** That
is the largest single effect anywhere in this file.

**The multiplicative-conditioning argument is not supported here.** Fourier (1.73×) slightly beats
FiLM (1.81×), where the argument predicted the reverse. The *measurement* of a flat gain on the
archived weights stands on its own — it is a direct Jacobian of real tensors — but the mechanism
story attached to it ("additive conditioning cannot express a multiplicative gain") does not
survive this ablation and should not be repeated as established.

**Confound, stated so the null does not read as stronger than it is.** No arm beats the floor, so
in this testbed the binding constraint is predicting the *content* from the conditioning vector at
all, not transporting onto it. A normalization null under those conditions is weak evidence;
the `+pos` result, which moves 172×, is not.

**A per-`(step, dim)` offset is a null in action space. Measured, so it does not get re-proposed.**
A design panel recommended learning one constant per chunk step and joint dim as the *first*
ticket, on the grounds that it beats anything the flow branch can reach. On the real holdout:

    zero-delta               1.632760e-05
    per-(step, dim) offset   1.629561e-05   <- 0.2 % better than emitting nothing
    global scalar offset     1.632733e-05
    model                    1.112983e-05

The recommendation confuses two spaces. Per-step offsets *are* nearly the whole story in the
**latent**, where the positional centroid has std 1.422 against content 0.05583 — which is exactly
why `FLOOR_MSE` exists. In **action** space the mean chunk is ≈ 0 and the offsets buy nothing.
`FLOOR_MSE = 1.68201e-05` is the order-blind anchor and is itself *worse* than zero-delta
(1.632760e-05), so any fix whose whole content is "recover the step index" cannot clear zero.

### Why all three now sit behind I-10

With a t-flat gain the sampler is `z ← z + g(x̂1 − z)dt`, which contracts onto the head's **own**
`x̂1` estimate by `exp(−g)`. So the flow readout is a strictly-worse noisy copy of the regression
head, converging *to* it as `g → ∞` and unable to beat it at any `g`. Implied latent error:
flow-converged 0.2406 (4.3× the content), regression head 0.0454 (0.81× the content).

Fixing the field therefore raises a ceiling set by the regression head — and I-10 measures that
head losing 1.76× to a linear map on proprioception. A separate measurement of the conditional
(near-duplicate observations, 1-NN twin MSE 2.238e-06 vs random-partner 2.431e-05, ratio **0.09**,
monotone across all ten distance deciles; no bimodality signature; prediction sits *inside* the
local ground-truth cloud rather than in a void) found it **near-deterministic on this dataset**.
Under a near-deterministic conditional the regression head is Bayes-optimal and the flow branch
has nothing to win. That verdict is dataset-specific and should be re-measured on any dataset with
genuine branch points before it is generalized.

---

## I-4 · State history window

**Evidence: literature (RoboDojo Memory dimension).**

`StateMLP.encode()` sees one state snapshot (`src/wam/encoders/state_mlp.py:145`). `dq` gives
first-order motion; nothing carries "what was visible three seconds ago."

RoboDojo's Memory dimension is where every policy sits in the low single digits, and the paper's
finding is specifically that *temporal prediction alone is insufficient for sparse-evidence
recall* — i.e. our video branch will not fix this for free either.

**Only worth doing if memory-conditioned tasks become a target.** The MVP pick-and-place is
memoryless. Deliberately parked.

---

## I-5 · State as latent frames rather than global conditioning

**Evidence: literature.** Bigger surgery — post-MVP.

Today: `StateMLP` → `backbone.condition_state(state_emb)` → one global conditioning vector.

Cosmos Policy instead injects proprioception, future states and value estimates as **additional
latent frames inside the video denoising sequence**, preserving the pretrained model's frame
format. Proprioception then flows through the full attention stack and can interact with the
spatial tokens, instead of biasing everything uniformly.

Attractive because it needs no new architecture — it reuses the backbone's own interface. But it
touches `FlowBackbone` and both adapters, so it is an M6-shaped change, not an M5 one.

---

## I-6 · Add FLUX.2 as an image-editing backbone candidate

**Evidence: literature.** Touches OD-04 / OD-06 / M5.

Our backbone assumption is a **video** model — OD-04 settled on Wan2.2-TI2V-5B (Apache 2.0),
verified on real weights, and OD-06 defers FLUX 3 to M5.

**ImageWAM** (arXiv 2606.19531, June 2026) argues world-action models do not need video
generation at all: an **image editing** model is better aligned with manipulation, because
"transform the current observation toward a desired visual state" *is* the manipulation problem.
It extracts intermediate KV caches from the editing denoiser and conditions an action head on
them — no future-frame decoding. With FLUX.2 4B it reports 93.38 % on RoboTwin 2.0, comparable
to Fast-WAM (91.83 %) and LingBot-VA (92.20 %), without extra policy pretraining.

**Read those numbers carefully.** Fast-WAM scores 91.83 % on RoboTwin 2.0 and **2.03 %** on
RoboDojo. RoboTwin-class benchmarks are exactly what RoboDojo was built to expose, and ImageWAM
is not in RoboDojo. Treat this as "worth a probe," not "better."

**Why it is attractive anyway:** FLUX.2 is open and available **today**, unlike FLUX 3 Dev
(API/private early access only, license unpublished as of 2026-07-26). It gives M5 something to
work with that does not depend on a release we do not control. FLUX.2 4B→9B scaling on
LIBERO-Plus was 83.1 % → 85.21 % — modest, so start at 4B.

**Licensing, checked 2026-07-29 — this is the part that bears on OD-06.** BFL ships the FLUX.2
collection under a *split* licence:

| Repo | Licence | Usable |
|---|---|---|
| `black-forest-labs/FLUX.2-dev` (32B) | FLUX Non-Commercial | no |
| `black-forest-labs/FLUX.2-klein-4B` / `-9B` (+ `-base`, FP8/NVFP4 quants) | **Apache 2.0** | yes |

If FLUX 3 follows the same pattern — and FLUX.1 did too — then **"FLUX 3 Dev", the PRD's
preferred backbone, will be non-commercial**, and only a `klein` variant would be usable under
the same criterion that decided OD-04 in Wan's favour. That is worth knowing *before* the
release rather than after: it turns I-6 from "a probe worth running at M5" into the hedge for
OD-06. `klein-4B` also runs in ~13 GB at 4 distilled steps, i.e. on a single 4090 — so it is
the cheap rig for debugging the T-16 training loop against real 4B weights, separately from
whether it ever becomes a backbone.

**What it is not:** a video model. FLUX.2 is image-to-image with no temporal attention and no
temporal VAE compression, so it cannot satisfy `FlowBackbone` in the sense Wan does. ImageWAM's
claim is precisely that a world-action model does not need that — but adopting it means
adopting their reformulation (autoregressive next-frame editing, KV-cache readout), not
dropping FLUX.2 into the existing seam.

**Cheapest form:** a third `hf_job_*_probe.py` reusing the same windows/labels/split as T-15 and
T-24. Same decision rule. If I-1 lands first, run it with the spatial readout.

---

## I-8 · Measure the data-scaling curve before buying more data with months

**Evidence: hypothesis. Uses assets we already own.**

Every negative we have — T-18 (tiny trunk hurts), T-15/T-24/T-26 (frozen features carry nothing),
T-16 (LoRA loses to inertia) — was produced on the *same* 402 success-only GR00T episodes of one
task. "Not enough data" is the standing explanation for all three, and it has never been tested.
It is also the most expensive conclusion in the project: it implies a G1 EDU4, a teleop rig and
months of recording (D1/D2, `docs/ROADMAP.md`).

Test it with what is already staged. Retrain T-16 at **40 / 120 / 362** training episodes,
identical everything else, and score each on the same 40-episode holdout. `run_bench.py` already
compares runs and refuses mismatched holdouts, so the curve costs no new measurement code.

**Cost:** 3 runs; the full one took a fraction of the 5 000 GPU-h allocation, and the two smaller
ones are cheaper still. Nothing here needs a robot.

**Decision rule, fixed before the run:**

- `skill_vs_repeat_pct` **improves monotonically** with episode count → data is the binding
  constraint, the extrapolation says roughly how much is needed, and the case for D1/D2 collection
  is evidence rather than hope.
- **flat or non-monotonic** → more of *this* data will not fix it. Then the question is the kind of
  data (task diversity, failure cases, a live gripper — I-9) or the architecture (I-2), and months
  of recording the same task would have been the wrong move.

Weakness to state up front: three points on one task cannot distinguish "needs more episodes" from
"needs more *tasks*". A useful variant, at the same cost, is holding episode count fixed and
varying task diversity across public LeRobot sets.

**Staged 2026-08-01, rule `I8_RULE_V3` in git before the first rung is submitted**
(`cluster/discoverer/55_train_i8_rung.sbatch` trains, `62_eval_i8_curve.sbatch` scores and prints
the verdict). Splits are nested and committed — `configs/splits/i8_train_{040,120,362}.txt`, with
40 ⊂ 120 ⊂ 362, zero overlap with the 40-episode holdout, and a seeded shuffle rather than a
sorted prefix, so a rung is a random sample of the corpus and not its alphabetical head. Rung 362
is `runs/t16-lora-seed0`, already scored; its `bench.json` is consumed rather than recomputed.

Three things the rule had to be taught, all of them before seeing a number:

- **The confound is symmetric.** Equal STEPS means rung 40 runs 147.5 epochs against rung 362's
  16.9, and that gap manufactures a flat curve exactly as readily as a steep one. The first draft
  held verdict A (*build the dataset*) provisional for it and let verdict C (*not data-limited*)
  through free — so the confound could produce the cheap conclusion at no cost. **All three**
  verdicts are now gated on the same equal-EPOCH control (`EPOCH_INFLATED`, read from the run log's
  final `action_reg`). A rule that holds only the expensive conclusion to a standard is a
  preference, not a decision rule.
- **N\* is an extrapolation, not a measurement.** It is OLS through three points, 2.8 doublings
  past the largest measured N, with one residual degree of freedom. Verdict A now requires the
  whole bootstrapped interval — propagated from the *measured* seed spread, via the
  `i8-rung040-seed1` replicate — to sit inside what a collection campaign could deliver, not just
  the point estimate.
- **"C" was two different worlds sharing one sentence.** C fires on `not (MONOTONE and SPAN ≥
  MATERIAL)`, so three noisy rungs landing out of order while spanning materially printed "more of
  this data did not move the headline" when the headline had moved by more than a seed does. Split
  out as **C-NOISY**, which claims no readable curve, routes to a seed replicate rather than to
  I-9, and licenses no recording campaign either way.

Sequencing: run this **after** T-30. A readout swap changes the headline metric for every rung at
once, and fitting a scaling curve through numbers that a pending decode change may move is fitting
a curve to a moving target.

---

## I-9 · Score on a dataset whose gripper actually opens

**Evidence: measured (T-27).**

The demonstrated gripper channel in `datasets/gr00t-apple-full` has peak-to-peak range **0.120**,
sitting on the 0.5 binarization threshold — it never opens or closes. `gripper_accuracy` 0.89 is
thresholding noise, and WAM-Bench emits it as a warning rather than a metric.

The consequence is larger than one bad number: **the benchmark is structurally blind to grasping**,
which is the entire point of pick-and-place and the thing AC-01/02 are about. Every verdict we have
is a verdict about arm trajectories only.

**The prescription above was wrong, and the correction is the whole point of this entry
(2026-08-01).** "Select a public dataset with real transitions" assumed the grasp was missing from
the data. It is not. Measured on `data/raw/gr00t_apple`, parquet only, no decode:

- the **left** hand's joints span up to **0.826 rad** peak-to-peak, and the commanded
  `action[29:36]` spans the full range;
- the **right** hand is frozen — 0.0007 rad across all 402 episodes;
- `hand_synergy`'s `clip((mean + 1) / 2, 0, 1)` assumes source joints in `[-1, 1]`, a range the
  Dex3 hand never uses, and squashes 0.826 → **0.157**;
- `relabel_chunks` then averages **both** hands into `gripper_target`, so the dead right hand
  halves it again → **0.0785**, which is the 0.120 T-27 measured, centred on the 0.5 threshold.

Rescale the same recordings and **30/30 episodes show exactly two debounced open/close
transitions** — one grasp, one release, which is what "move the apple to the plate" should look
like. The grasp was always on disk. We destroyed it in the converter and then read the flat
channel as a property of the task.

**Fix, as it actually shipped:** `--gripper-mapping active-hand` fits one **dataset-level** affine
over the raw synergy of the hand that moves and takes `gripper_target` from that hand alone.
Dataset-level rather than per-episode on purpose — a per-episode min-max makes the same physical
aperture a different number in every episode, which is unlearnable. Because the fit depends on
which episodes are in the conversion (30 eps → offset −0.39980/span 0.41004; 402 eps →
−0.43865/0.46675), two conversions are comparable only if they share an affine: `--gripper-affine
OFFSET SPAN` pins it, and a pinned affine that would clip **any** sample is refused, because
clipping is silent in the output and moves every admissibility clause in the *passing* direction.
The legacy mapping is now held to the same bar (`legacy_clipped_frac`); until 2026-08-01 the one
mapping that *assumes* a scale was the only one allowed to be silently wrong about it.

`scripts/audit_gripper.py` is the gate: four clauses (dynamic range, debounced transitions per
episode, fraction of episodes with a transition, fraction of episodes with a full
close-**and**-reopen **cycle**). The cycle clause exists because a monotone ramp from 0 to 1 clears
a transition count and is not a grasp. Saturation is reported against
`expected_saturated_frac` — a fitted affine rails exactly the two extremal samples it was fitted
on, so a rate far above that is "clipped, not measured".

**What this does not retract:** T-16 and T-18 are arm-trajectory verdicts and stay exactly as
recorded — `skill_vs_repeat_pct` never touched the gripper channel. What it retracts is the
*explanation*, and the size of the fix: this was hours of converter work on data already on disk,
not a dataset acquisition.

Related and cheap, in `docs/benchmark.md` rather than here: **L4's gate is one-sided.**
`smoothness_ratio ≤ 2` scored T-16's 0.29 a full 20/20 while that value means the prediction is
3.4× *smoother* than a demonstration — a defect, not a virtue. A two-sided band would catch it.
Recorded rather than quietly patched, because changing a pre-registered threshold after seeing the
number it scored is exactly the move the pre-registration exists to prevent; it changes no level
here (L4 is only reachable through L1 and L2, and this run stops at L0).

---

## I-10 · The deployed model loses to a linear map from proprioception

**Measured 2026-08-01. Two independent implementations, same number. No GPU, no allocation.**

Fit a ridge regression from the 32-dim robot state at chunk time (q15 + dq15 + gripper2) to the
flattened `[16, 15]` action chunk. Train on the 362 training episodes, score on the same 40
holdout episodes the model was scored on, taken from its own `predictions.jsonl` via
`load_episode_ids` so the split cannot drift.

| predictor | holdout MSE | trainable params |
|---|---|---|
| zero-delta (hold still) | 1.632760e-05 | — |
| **Wan-5B + LoRA, deployed** | **1.112983e-05** | **82 519 450** |
| ridge, `q` only | 1.348259e-05 | 3 615 |
| ridge, gripper only | 1.550558e-05 | 495 |
| ridge, `dq` only | 6.869239e-06 | 3 615 |
| **ridge, full 32-dim state** | **6.330899e-06** | **7 920** |

**The 7 920-parameter linear map is 1.76× better than the 82.5 M-parameter model.** `dq` alone —
fifteen joint-velocity readings, no vision at all — is 1.62× better. Ridge is flat across λ from
1e-2 to 1e2 (6.3309e-06 → 6.3332e-06), so it is not memorizing.

**Controls, asserted before anything was fitted.** The harness reproduces the archived zero-delta
baseline (1.632760e-05 vs 1.63276e-05) and the model's own archived MSE (1.112983e-05 vs
1.11298e-05) to every digit. A separate agent's independent implementation returned 6.330899e-06
for the ridge — the same value to seven figures from different code.

*Not* reproduced: repeat-last-action came out 8.041488e-06 here against the bench's 9.13766e-06,
i.e. this file's definition of that baseline is not the bench's. It is not load-bearing for
anything above and is recorded rather than quietly dropped.

**The comparison is not "we forgot to feed it `dq`".** That was the obvious way for this result to
be an artifact, so it was checked rather than assumed. `StateMLP` takes all four canonical groups
— `_GROUP_ORDER = ("q", "dq", "imu", "gripper")` (`state_mlp.py:30`), each with its own input
block (`:81-82`) — and the archived config carries `num_joints: 15, gripper_dims: 2,
embedding_dim: 32`, so the state reaches the backbone through `condition_state` with `dq`
included. The model therefore has **strictly more** information than the ridge: the same
proprioception, plus vision. It loses by 1.76× anyway.

### What it does and does not establish

**Establishes:** on this holdout, under this metric, the visual pathway contributes nothing
measurable. A model that loses to proprioception has not earned its backbone. The bench already
knew half of this — `skill_vs_repeat_pct` is negative, i.e. the model loses to repeat-last-action
— but "loses to one trivial baseline" and "loses to *any* linear function of proprioception" are
different claims, and only the second one indicts the whole visual path.

**Does not establish** that the task is trivial or that vision is useless for it. It equally
supports the reading that **this MSE is largely satisfiable by momentum extrapolation**, in which
case the metric was never measuring the thing we care about and *beating* it would not have proven
much either. Both readings are damning for the current state, in different ways, and this
measurement cannot separate them. Separating them needs a metric that a momentum extrapolator
fails — task success, or a subset with genuine branch points.

The parsimonious confound is recorded plainly: 402 episodes of **one** task, ~90 minutes of robot
data, ~9 476 heavily-overlapping training chunks, against 82.5 M trainable parameters — about 36
trainable parameters per scalar action target.

`scripts/bench_ridge_baseline.py` ships this as a permanent bar. It is deliberately blind: it
never reads a frame.

---

## What not to change

Recorded so these do not get "improved" by accident:

- **tanh + identity normalization** (`action_head.py:95`). Targets are physical canonical units
  with no denormalization anywhere; `EpisodeDataset` enforces per-step `|targets| < 1`. This is
  a deliberate MVP simplification with a documented failure mode — it will need revisiting when
  action ranges grow, not before.
- **Validity mask + learned missing embeddings** (`state_mlp.py:142`). Genuinely ahead of most
  published policies, which either concat proprioception raw (breaks on sensor dropout) or drop
  it. `torch.where` keeps it differentiable-safe and stops NaN/Inf in masked groups from
  reaching the backward pass. Keep.
- **`hidden_dims=(64,)` on the action head.** Small on purpose. For "overfit one small task" a
  tiny MLP on pooled features is the *right* choice — fastest to debug, and if it cannot overfit,
  the fault is data or wiring, not capacity. Revisit only with I-2.
- **Deterministic safety layer downstream.** Non-negotiable (FR-07). It is also why soft bounds
  in the head are acceptable.
- **Backbone-agnostic protocol** (FR-09, `FlowBackbone`). It is what makes I-6 a probe instead of
  a rewrite. Most published work hardcodes its backbone; we should not follow.

---

## Suggested order

Re-ordered 2026-07-30 around one principle: **anything that can change the meaning of a number we
have already recorded comes before anything that produces a new number.** Items 1–2 re-score an
existing checkpoint and need no retraining at all.

| # | Item | When | Cost |
|---|---|---|---|
| ~~—~~ | ~~I-1 spatial-readout probe~~ — **✅ ran 2026-07-29, verdict unchanged** | done | 7.6 s GPU, free |
| ~~—~~ | ~~I-9 converter (the gripper we destroyed)~~ — **✅ built 2026-08-01, needs no GPU** | done | hours, no GPU |
| ~~—~~ | ~~I-7 frame history at inference~~ (T-29) — **✅ ran 2026-08-01: +10.65 pp, still fails L1** | done | 0.2 GPU-h, no retrain |
| ~~—~~ | ~~I-7 re-score T-18 + `d1-full-gen-seed0`~~ — **✅ ran 2026-08-01 on a laptop CPU: both moved <0.05 pp, ladder is single-mode, AC-07 readable again** | done | **zero GPU**, no retrain |
| ~~—~~ | ~~I-3 flow branch deployed~~ (T-30) — **ran 2026-08-01, job 184670: every arm SUSPECT, pre-registered refusal, question still open** | ran, no verdict | 0.56 GPU-h spent |
| ~~—~~ | ~~Fix the flow sampler~~ — **audited 2026-08-01: no bug. All three suspects clean against the training path; the old tests were green *and blind* (mutation-proven), now fixed** | done | hours, no GPU |
| ~~1~~ | ~~**Give `ActionVelocityHead` a sense of time**~~ — **superseded 2026-08-01, see the re-order below.** The diagnosis was right and is now confirmed on the real weights, but it is the *third*-largest defect in the flow branch, and the whole branch is downstream of I-10 | demoted | hours + head-only retrain |
| **1** | **I-10 · decide what the ridge result means** — the deployed model loses 1.76× to a 7 920-parameter linear map on proprioception. Either the visual path contributes nothing, or the metric is momentum-satisfiable. Needs a metric a momentum extrapolator fails, not another readout change | now | hours, no GPU |
| **2** | **I-3/D2 · give the head a readable step index** — a correctness fix, not a refinement: with no step index and `z_t` near-pure noise below `t ≈ 0.416`, the head cannot tell which of 16 centroids to move toward and the correct field is **unrepresentable**. Worth **172×** in the ablation, the largest single effect in this file | after 1 says the branch is worth keeping | hours |
| **3** | **I-3/D3 · give the head a real `t` embedding** — the gain is confirmed flat to 4 s.f. on the archived weights. Worth a further ~4× in the ablation. Fourier and FiLM measured equivalent, so take the simpler one; the "must be multiplicative" argument did **not** survive testing | with 2 | hours |
| ~~4~~ | ~~**I-3/D1 · normalize the action latent**~~ — **`NORM_V1` refuted 2026-08-01.** The latent really is 99.85 % a fixed positional pattern, but normalizing it changed almost nothing in the ablation. Kept in the file as a measured negative, not a queued item | — | — |
| 5 | **Re-run T-30** once a flow arm clears the guard on its own — arms and `63_...sbatch` unchanged | after 2–4 | ~0.6 GPU-h |
| 6 | I-9 re-score the ladder on the rescaled gripper (T-31) | after the converter's audit passes | ~0.2 GPU-h, no retrain |
| 7 | I-8 data-scaling curve (T-32) | after 1 — see below, I-10 changes what this curve would mean | 3 runs, existing allocation |
| 8 | I-2 cross-attention head | after 1–5 say whether the readout was the problem | days + retrain |
| 9 | I-6 FLUX.2 probe | M5, alongside the FLUX 3 decision | hours |
| 10 | I-4 state history | only if memory tasks become a target | days |
| 11 | I-5 state as latent frames | M6 | weeks |

**Re-ordered again 2026-08-01, and this one is a demotion of the item that was #1 that morning.**
The principle from 2026-07-30 still holds, but I-10 extends it: *anything that can change the
meaning of a number we have already recorded* now includes the question of whether the headline
metric measures the thing we care about at all. A 7 920-parameter blind baseline beating the
deliverable is that kind of number.

**One of these entries is a retraction of the same table earlier the same day.** For part of
2026-08-01 this table had D1 (latent normalization) at #2 on the strength of a derivation, before
the ablation that tested it had reported. `NORM_V1` was written down first and came back refuted,
so D1 is struck and D2 promoted. The derivation is left in I-3 with its refutation attached rather
than deleted — it was wrong for an interesting reason, and deleting it would make this file look
like it has never mis-ranked anything.

**Why the flow items dropped behind I-10.** On the flat-gain analysis in I-3 the flow readout
contracts onto the regression head's own estimate, so it converges *to* that head and cannot beat
it. Every flow fix is therefore capped by what the regression head knows — and I-10 measures that
head losing to proprioception. Fixing the flow branch first would be optimizing the readout of a
representation that has not been shown to carry anything.

**Why I-8 slipped again, from 4 to 7.** The 2026-07-30 note said fitting a scaling curve through
numbers a pending decode change may move is fitting to a moving target. I-10 is worse than a
moving target: if the metric is momentum-satisfiable, N\* would be extrapolated from a curve
measuring how fast the model learns to imitate `dq`. That is ~125 GPU-h to fit a curve whose
y-axis is in question. **Still held for explicit confirmation, now for a second reason.**

**Why I-9 moved but did not jump the queue.** Its *build* is done and cost no GPU, because the
finding turned out to be a converter bug rather than a missing dataset. Its *re-score* still sits
at 3, behind the two items that can change the meaning of a number already recorded. Nothing about
a rescaled gripper channel changes what `skill_vs_repeat_pct` measured on arm trajectories.

**Why I-8 slipped from 3 to 4.** T-30 changes the headline metric for every rung simultaneously.
Fitting a scaling curve through numbers a pending decode change may move is fitting a curve to a
moving target — and N\*, the number that would buy a robot and months of recording, is the most
extrapolated quantity in the file.

I-1 jumped the queue because it tested a claim we had already written into `TASKS.md` as settled.
It came back negative, which is the cheapest possible outcome: nothing downstream has to change,
and the claim is now one we have earned rather than one we inherited from a pooling choice.

I-7 was in the same position, one level more serious: I-1 questioned how we *probed* the backbone,
I-7 questioned what we *showed* it. If the answer is "nothing changes", the T-16 negative
becomes one of the better-supported results in the project. If the answer is "it moves", then two
recorded verdicts were measured out of distribution and the cost of finding that out was one eval
pass. The asymmetry is why it is item 1 despite being the smallest item in the file.

**Resolution, 2026-08-01 — the sixth copy, restored.** The two sentences above are the
pre-registered text, put back verbatim after a propagation pass had rewritten them in place into
past-tense narration. They are a copy of the I-7 rule and are covered by the same "annotate, never
edit" instruction as the other five; the count of **six** at `:282` is only true if this one still
says what it said before the run. **The answer was neither.** `skill_vs_repeat_pct` moved by a
third of the gap and still lost, so the T-16 negative survives while the number it was published as
does not, and only one of the two affected verdicts was actually re-measured. The asymmetry held
and it paid: 0.2 GPU-h bought a correction to a published figure and the discovery that the ladder
table is mixed-mode.

**What it cost to learn nothing would have been higher.** Had this stayed unrun, every downstream
decision — I-8's N\*, the D1/D2 collection commitment, the "bottleneck is data" claim in
`TASKS.md` — would have rested on a number that was 10.65 pp wrong in the flattering direction for
the harness and the unflattering direction for the model.

---

## External context

Backbones used by the five world-action models in RoboDojo, for the bake-off record:

| Model | RoboDojo rank | Backbone |
|---|---|---|
| X-WAM | 5 | Wan2.2-TI2V-5B + its causal VAE; depth as a light added branch |
| GigaWorld-Policy | 8 | undisclosed pretrained video backbone / GigaWorld-0.5 |
| AHA-WAM | 12 | Wan2.2-5B (video branch only); action DiT 1.02B trained from scratch |
| Fast-WAM | 14 | Wan 5.5B |
| LDA-1B | 19 | latent dynamics, no pixel diffuser |

Wan 2.2 (28.07.2025, Apache 2.0) is not the strongest video model — it is the **newest open**
one. Wan 2.5 / 2.6 / 2.7 are all closed, API-only. That licensing wall, not model quality,
explains the field's backbone monoculture, and it independently confirms OD-04.

Evidence that a stronger backbone helps is thinner than the marketing suggests: ImageWAM's
4B→9B gain was ~2 points and non-uniform, and "Turning Video Models into Generalist Robot
Policies" has no backbone ablation at all — only pretrained-vs-random init (0.40 → 0.13
validation MSE), whose authors explicitly warn it is not a substitute for closed-loop success.

**Read that alongside T-18** (*measured*): our own world-action-vs-action-only ablation came out
at −89.5 % holdout MSE — the video branch *hurt* at tiny scale. Both point the same way: what
matters is **whether** you pretrain, far more than **on which** backbone.

Sources: [RoboDojo](https://arxiv.org/abs/2607.04434) ·
[ImageWAM](https://arxiv.org/html/2606.19531v1) ·
[AHA-WAM](https://arxiv.org/html/2606.09811) ·
[Cosmos Policy](https://arxiv.org/html/2601.16163v1) ·
[NVIDIA WAM overview](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
