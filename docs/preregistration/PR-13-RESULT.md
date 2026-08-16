# PR-13 result — **W**: the clause that decided `VOID` does not fire on a repaired instrument

Ran 2026-08-16 on the workstation, CPU only, **zero GPU-hours, nothing submitted.** Five cells, about
two minutes. Pre-registration `PR-13-t39-g0-rederivation.md`, rule `T47_RULE_V1`, committed in
`7d93ab6`, with §4a's magnitude prediction committed in `85087fd` — **both before the driver produced
a single cell**. Driver `scripts/rederive_t39_g0.py` and its 8 tests in the same commit. Task
**T-47**. Artifact `runs/t47-g0-rederivation/rederivation.json`.

**Verdict `W`. `T39_RULE_V1`'s G0 blocking clause does not hold under a corrected instrument.** On
T-39's own holdout, the corpus's own commanded column reaches **L1 at +68.10 %** and **L2 at
+75.40 %**.

## The table

| cell | anchoring | chunks | L1 | L2 | vs-zero | `horizon_ratio` | `smoothness_ratio` | step-0 share | level |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **bridge** | unmodified | **1040** | **−359.41** | −102.54 | −157.11 | 0.0044 | 8.518 | 93.39 % | below L0 |
| control | unmodified | 1000 | −344.54 | −93.81 | −150.36 | 0.0045 | 8.328 | 93.26 % | below L0 |
| **repaired** | V-chain | **1000** | **+68.10** | **+75.40** | **+82.04** | **0.9717** | 0.280 | 6.03 % | **L4 moves-like-a-demo** |

**The bridge reproduced `PR-07-RESULT.md`'s `−359.41` to a drift of `+0.002 pp`** — so one number in
this document is the archive's number, produced by this driver, and the other two are comparable to
it.

## Gates — all four passed, first run

| gate | requirement | measured |
|---|---|---|
| **G0.1** `oracle_state` on the full set | ≥ 90 % | **+100.00 %**, 1040 chunks |
| **G0.2** bridge reproduces PR-07's `−359.41` | ±0.5 pp | drift **`+0.002 pp`** |
| **G0.3** control and repaired scored one set, of size `full − episodes` | 1000 = 1040 − 40 | **1000 and 1000** |
| **G0.4** V-chain touched row 0 and only row 0 | non-zero, then exactly zero | `2.3607e-02`, then **`0.000e+00`** |

## The registered magnitude, and how it did

§4a predicted **+55 to +75**, and specifically *slightly below* PR-12's half-A values, on the
reasoning that the anchorable set re-adds each episode's **last** chunk — end-of-task, low-motion,
where repeat-last-action is strongest.

**Measured: +68.10.** In band, and below the pooled half-A/half-B `v_chain` figures at `d = 0`
(+67.14 / +69.41, pooling to roughly +68.3). Direction and magnitude both hold.

**One correction to my own addendum**, which changes nothing but should not be quietly left: §4a
quoted "+67.30 / +68.57" as the half-A comparators, and those are the **`v_mask`** cells. The right
comparator for a `v_chain` result is PR-12's `v_chain` row, +67.14 at `d = 0`. The prediction is
unaffected — +68.10 is in band against either — but the numbers named in the addendum were the wrong
two.

## What the set change actually cost, which nobody had measured

Dropping each episode's **first** chunk moves the unmodified arm from **−359.41 to −344.54** —
**14.87 pp from 40 chunks out of 1040**. Those forty are ~4 % of the set and carry a
disproportionate share of the damage, which is what an anchor defect at episode start predicts: the
robot begins at rest and the standing command-versus-state gap is at its widest there.

It is also a caution about the trimmed set every earlier document used. The trim was adopted in
PR-10 for an unrelated reason — making delays comparable — and it silently removed the worst-affected
chunks. **No conclusion in PR-10, PR-11 or PR-12 turns on this**, since all of them compared cells
within one set, but it is the kind of thing that should be on record rather than rediscovered.

## What `W` is, and what it is emphatically not

**It is not a verdict on T-39, and `T47_RULE_V1` registered that before the run.** `T39_RULE_V1`'s
`P`/`N`/`M`/`I` verdicts all require the policy arm, which never ran — job 187813 died at 108 s on a
missing `GROOT_PATCH_MISTRAL` export. T-39 does not become `P`. It becomes **a gate whose blocking
condition was measured through a broken instrument, and whose re-derivation is now on record.**

### `W` licenses

- **Correcting every document that asserts no policy trained on this corpus's action column can
  clear our bar** — the root `CLAUDE.md`, `subprojects/README.md`,
  `subprojects/edge-wam/CLAUDE.md`, and `PR-07-RESULT.md`'s standing reading. That sentence's basis
  is withdrawn by measurement. **Correcting a claim is not lifting a gate**, and the gate text is
  not touched by this task.
- **Handing the training decision back to the project owner with the premise corrected**, which is
  where it already sat.

### `W` does not license

- **Discharging the training gate.** `C` removed the cause; `W` removes the stated premise;
  **neither is permission.** No session may edit `CLAUDE.md` to lift it.
- **Any statement about GR00T or any policy.** No model was trained, loaded or consulted; this
  scored oracles against oracles. PR-07 §6 stands.
- **Retro-validating the fourteen negatives.** PR-12-RESULT's blast radius already excluded them:
  they were scored WAM-format against WAM-format with no `commanded_to_chunk` in the path.
- **Relabelling anything**, or amending `PR-07-positive-control.md`. PR-13 is a separate document
  precisely so PR-07 never has to change.
- **Any claim about `docs/benchmark.md`'s L4 gate**, which stays an open decision for the owner —
  and note the repaired cell reaches **L4 under spec 0.1.0** while `smoothness_ratio 0.280` is below
  spec 0.2.0's two-sided floor, so the two specs disagree about this cell.

## What this still cannot answer

**Whether a policy can learn these labels.** This scores oracles. It is the same limit T-39 had, and
repairing the instrument does not remove it — it only removes the reason to believe the answer was
already no. The honest statement of the project's position is now:

> The corpus's own action column, correctly compared, is **L4** on our own holdout. Nothing has been
> trained on it. Whether a policy can reach that is untested, and testing it is a training run.
