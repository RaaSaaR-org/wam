# PR-13 — Does T-39's `VOID` survive its instrument being repaired?

Pre-registered **2026-08-16**, after PR-12 returned `C` and **before the re-derivation is run**. Task
**T-47**. Rule **`T47_RULE_V1`**, fixed in §5 of this file and nowhere else. Zero GPU-hours: an
offline re-score of artifacts already on disk. **Claimed across live sessions before it was
written**, and the scope below is the project owner's narrowing, not this session's.

## 1. The one clause this is about

T-39 reported `VOID (labels)` on 2026-08-16. That verdict was not reached through the policy arm —
the policy arm never ran. It was decided by a single clause of `T39_RULE_V1`
(`PR-07-positive-control.md:137-141`):

> **G0 · VOID (runs first, can stop everything).**
> `oracle_state` must reach `skill_vs_repeat_pct >= 90 %`. […]
> `oracle_action` must reach **L1**. Below that […] T-39 is VOID and the finding is recorded
> against the label pipeline.

`oracle_action` came in at **−359.41 %**, so the clause fired and the project has run on its
consequence ever since: *no policy trained on this corpus's action column can clear our bar.*

**PR-12 showed that number was produced through a defective instrument.** `commanded_to_chunk`
built the chunk's step 0 as `command − STATE` while every other step of both arms is a homogeneous
first difference; that one element carried **~90 % of the summed per-step MSE and 143× its
neighbours**, and homogenising it took a held-out half from **−379.68 to +69.15**.

**PR-13 asks exactly one question: does G0's blocking clause still hold when the instrument is
repaired?** Nothing wider. Not the training path, not the runtime executor, not the fourteen
negatives — PR-12-RESULT's blast-radius section already bounds all three, and this does not revisit
them.

## 2. Why this is worth running even though the answer is largely foreseeable

**§4 registers this as confirmatory by design and says so before it runs.** After +69.15 on a
held-out half, the outcome is not seriously in doubt, and pretending otherwise would repeat the
error PR-12 nearly made — registering a prediction whose answer already exists.

The value is elsewhere and is specific: **PR-12 measured a different chunk set, through a different
driver, in halves.** G0 read the *full, untrimmed* `groot-holdout` — 1 040 chunks — in one pass.
A gate's own condition should be re-derived on the gate's own set before anyone says the gate's
premise is withdrawn. Inferring it across a set boundary and a driver boundary is exactly the kind
of step this project has paid for before.

## 3. The measurement

Three cells, `d = 0` throughout — PR-12 retired the delay, so no delay is applied and none is fitted.

| cell | anchoring | chunk set |
|---|---|---|
| **bridge** | unmodified (`q_cmd[0] − q_state[s]`) | the **full 1 040**, exactly T-39's |
| **control** | unmodified | the **anchorable** set (below) |
| **repaired** | V-chain (`q_cmd[0] − q_cmd[−1]`) | the **anchorable** set |

**The chunk set cannot be 1 040 for the repaired cell, and that is registered rather than
discovered.** V-chain requires `start ≥ 1`, so it drops each episode's first chunk. The
**anchorable set** is therefore the full holdout minus each episode's first chunk, and **both** the
control and repaired cells are scored on exactly it — a difference of chunk set between the two
compared cells would make them not a comparison. The count must equal
`full_count − num_episodes`, and it is recorded.

The bridge exists so that one number in this document is directly comparable to
`PR-07-RESULT.md`'s table, and the control exists so that the repaired cell has a same-set
counterpart. Quoting the repaired cell against the 1 040-chunk bridge would compare two sets.

`oracle_state` is **unaffected by V-chain** — it never calls `commanded_to_chunk` — and is scored
unmodified as G0's other clause.

## 4. What is a prediction here, and what is not

- **Not a prediction:** that the repaired `oracle_action` clears L1. PR-12 measured +69.15 / +69.41
  on held-out halves. This is a **replication on the registered gate's own set**, and it is labelled
  as one.
- **Not a prediction:** the mechanism. PR-12 and `docs/smoothness-ratio-audit.md` established it.
- **Genuinely open, and the reason a rule is still needed:** whether the effect **survives the set
  change**. The full set includes each episode's first chunks and its tail, which the trimmed set
  dropped uniformly and which no cell in PR-10, PR-11 or PR-12 ever scored. If the effect is
  materially smaller there, the correct reading is that the trimmed set was unrepresentative — and
  that would bear on all four documents, not just this one.

## 5. Gates — `T47_RULE_V1`

Ladder unchanged. `MATERIAL_FLOOR_PP = 10.0`, borrowed from `I8_RULE_V3` for the fifth time rather
than coined.

- **L1** `skill_vs_repeat_pct > 0` · **L2** `ci_skill_vs_repeat_pct > 0`

**G0 · INVALID — runs first, can stop everything.**

1. **`oracle_state` ≥ 90 %** on the full set — T-39's own first clause, unchanged. Below it the
   adapter is broken and this is a code fix, not a threshold change.
2. **The bridge reproduces `PR-07-RESULT.md`'s `−359.41`** within ±0.5 pp on the full 1 040. T-44
   already reproduced it to `−359.4078`; if it does not reproduce here, this is not the same
   measurement and nothing may be compared to the archive.
3. **The control and repaired cells scored an identical chunk set**, and that count equals
   `full_count − num_episodes`. Checked at runtime and written into the artifact.
4. **V-chain reached row 0 and only row 0** — non-zero RMS on `targets[0]`, exactly zero on rows
   1…15. Inherited from PR-12 §6 G0.3, unchanged, because a V-chain that silently reduced to the
   unmodified anchoring would produce a confident **S**.

**Which conclusion is expensive.** **W** is: it withdraws the stated premise of a gate the whole
project has been organised around. So W carries the material margin **and** requires L2, not L1
alone.

**The verdicts**, on the repaired cell, anchorable set:

| | condition | reading |
|---|---|---|
| **W** | repaired cell clears **L1** with `skill_vs_repeat_pct ≥ 10.0`, **and** clears **L2** | `T39_RULE_V1`'s G0 blocking clause **does not hold** under a corrected instrument. The premise of `VOID (labels)` is withdrawn by measurement |
| **S** | repaired cell fails **L1** | the `VOID` **survives** its instrument being repaired, which makes it a far stronger result than it was. The label space really is the problem and PR-12's gain did not generalise off the trimmed set |
| **I** | anything else — in particular `0 < skill_vs_repeat_pct < 10.0`, or L1 without L2 | indeterminate; nothing licensed, and the set-representativeness question in §4 becomes the live one |

**Precedence, fixed here:** **S, then W, then I.**

**Recorded regardless of verdict:** all three cells under both bench specs; the full per-step MSE
profile and step-0 share for each; `skill_vs_zero_pct`, `horizon_ratio`, `smoothness_ratio`, level
name and score; the bridge drift; the chunk counts; wall time.

## 6. Reading the outcome — decided before the numbers exist

**W is not a verdict on T-39 and cannot be.** `T39_RULE_V1`'s P/N/M/I verdicts all require the
policy arm, which never ran (job 187813 died at 108 s). The most W can say is that **the clause
which decided `VOID` no longer fires**. T-39 does not become `P`; it becomes a gate whose
blocking condition was measured through a broken instrument and whose re-derivation is now on
record.

**W licenses:**

- **A correction to every document asserting that no policy trained on this corpus's action column
  can clear our bar** — `CLAUDE.md`, `subprojects/README.md`, `subprojects/edge-wam/CLAUDE.md`,
  `PR-07-RESULT.md`'s standing reading. That sentence's basis is withdrawn. **Correcting a claim is
  not lifting a gate**, and the gate text is not to be edited by this task.
- **Handing the training decision back to the project owner with the premise corrected**, which is
  where it already sat.

**W does not license:**

- **Discharging the training gate.** Explicitly, and this is the whole reason the distinction above
  is drawn. `C` removed the cause; `W` removes the stated premise; **neither is permission**, and
  no session may edit `CLAUDE.md` to lift it.
- **Any statement about GR00T or any policy.** No model is trained, loaded or consulted; this
  scores oracles against oracles. PR-07 §6's prohibition is untouched.
- **Retro-validating the fourteen negatives**, which PR-12-RESULT's blast radius already excluded:
  they were scored WAM-format against WAM-format with no `commanded_to_chunk` in the path.
- **Relabelling anything**, or amending `PR-07-positive-control.md`. Rules here are versioned and
  never edited in place. PR-13 is a new document precisely so PR-07 does not have to change.
- **Any claim about `docs/benchmark.md`'s L4 gate**, which remains an open decision for the owner.

**S licenses** a much stronger negative than the project currently holds, and it would mean PR-12's
result is a property of the trimmed chunk set. That is the outcome worth being honest about wanting
to know.

## 7. Cost

Zero GPU-hours, nothing submitted. CPU only: three cells plus gates over artifacts already on disk.
Expected minutes.

## 8. What must exist before this runs

1. `scripts/rederive_t39_g0.py`, importing `commanded_to_chunk`, `oracle_state_chunks`,
   `raw_anchor_indices`, `read_raw_episode` and `ChunkLookupPolicy` from `eval_t39_baseline`, and
   `chained_oracle_action_chunks` from `probe_step_zero_anchor`. **Nothing is re-implemented**; the
   repaired anchoring must be the *same function object* PR-12 scored, or this replicates a copy.
2. **A test that the driver's full-set cell and `eval_t39_baseline`'s own `oracle_action_chunks`
   agree bit for bit**, so the bridge is a bridge and not a lookalike.
3. **A test that the control and repaired cells are scored over the same timestamps**, failing if
   either drops a chunk the other keeps. G0.3 is the runtime half; this is the offline half.

## 9. What this cannot answer

- **Whether a policy can learn these labels.** It scores oracles. Necessary, not sufficient, and
  PR-07 §6 still forbids the policy statement. This is the same limit T-39 had and repairing the
  instrument does not remove it.
- **What the tracking offset physically is.** PR-12 §7 lists three sources it cannot separate.
- **Whether the training path needs any change at all** — PR-12-RESULT's blast radius says it never
  used the defective adapter, and nothing here re-tests that.
- **The gripper**, withheld by the scorer on every arm so far.
- **Anything about the other twelve `G1_Dex3_*` corpora** (T-043).
