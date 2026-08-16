# PR-07 V2 — `T39_RULE_V2`: T-39 is re-scored under the repaired anchoring

**Registered 2026-08-17, before the run.** Amends `T39_RULE_V1` (`PR-07-positive-control.md` §5)
in exactly one respect: which quantity chunk step 0 is differenced against. It changes no
threshold, no verdict definition, and no arm.

`T39_RULE_V1` and `PR-07-RESULT.md` are **not edited**. This is a further version alongside them,
which is the convention `PR-08-V3` §7 states for this repo.

---

## 1. What changed, and why it is not a threshold move

`commanded_to_chunk` built chunk step 0 as `command − STATE` while every other step of both arms is
a homogeneous first difference `command[t] − command[t−1]`. PR-12 (verdict `C`) measured that one
element at **~90 % of the summed per-step MSE and 143× its neighbours**. PR-13 (verdict `W`)
re-derived G0 on the repair and got **+68.10 L1 / +75.40 L2** on T-39's own holdout, against
`−359.41` for the unmodified bridge — which it reproduced to **+0.002 pp** in the same run.

So the quantity `T39_RULE_V1` §5's G0 clause tests was measured through an instrument with a known
defect. **This amendment changes the instrument, not the bar.** Every threshold below is quoted
from V1 unchanged:

- `oracle_state` ≥ **90 %** `skill_vs_repeat_pct`
- `oracle_action` must reach **L1** (`skill_vs_repeat_pct > 0`)
- `MATERIAL_FLOOR_PP` = **10.0**, still borrowed from `I8_RULE_V3`
- **P / N / M / I** as defined in V1 §5, on `groot-holdout`, with `groot-train40` diagnostic only

## 2. The registered change

**T-39 is scored with `--anchor-kind command` on all four arms.**

All four, not the two that visibly move. `CommandedPolicy`'s own docstring states the constraint:
the policy's `anchor_kind` **must match the oracle's**, or the oracle stops being a ceiling over
the policy and PR-07 §4's "ceiling for any policy trained on that column" claim is void. A mixed
run would be two adapters wearing one rule.

## 3. The scored set changes, and this is the part that needed registering

Under `command`, a chunk whose anchor is the episode's **first** raw index has no preceding
command. It is **skipped**, not silently anchored on the state row — a mixed set would put the
defect back into precisely the chunks that carry the most of it.

| | chunks | note |
|---|---|---|
| `T39_RULE_V1`, archived | **1040** | 40 holdout episodes |
| `T39_RULE_V2`, here | **1000** | 40 dropped: one per episode |

PR-13 measured what those 40 carry: dropping them **alone** moves the unmodified arm from
`−359.41` to `−344.54`, i.e. **14.87 pp from 40 of 1040 chunks**. They are not a random 3.8 %.

**Consequence, registered rather than discovered later: a `T39_RULE_V2` number is comparable only
against other `V2` numbers.** It may not be placed beside the archived `−359.41`, beside any cell
in `docs/benchmark.md`, or beside PR-10's sweep cells, none of which were scored on this set. The
run records `anchor_kind` in its metadata so no future reader has to infer which set they hold.

## 4. What this does NOT do

- **It does not issue a verdict.** `P`/`N`/`M`/`I` all require the policy arm, which has never run
  — job 187813 died at 108 s on a missing `GROOT_PATCH_MISTRAL`, *after* both oracle arms
  reported. This document licenses the run that produces a verdict; it does not anticipate one.
- **It does not lift PR-08 §1.** `T40_RULE_V3` registers that a T-39 **VOID** closes PR-08 rather
  than opening it, and that `P`, `N`, `M` and `I` satisfy §1 while VOID does not. If this run
  returns VOID again, PR-08 stays shut. Nothing here grants `PR08_OVERRIDE_T39_VOID`, and this
  document registers no circumstance under which it may be used.
- **It does not restate PR-07 §7.** The 12 GPU-h ceiling and the one conditional second candidate
  (`lerobot/pi05_base`, on outcome N only) are unchanged and are not re-opened here.
- **It says nothing about GR00T.** PR-07 §6's VOID row still forbids that, and no policy number
  exists yet.

## 5. Why the expected G0 outcome is registered in advance

PR-13 already measured `oracle_action` on the repaired instrument on this exact holdout: **+68.10**,
which clears L1. So **G0 is expected to pass**, and the run's informative content is the policy
arms, not the oracles.

Registering that here is the point. If `oracle_action` comes back materially away from +68.10 on a
set of 1000 chunks, that is a **defect report against this run**, not a new finding about the
labels — the same quantity was measured twice by two drivers and disagreed.

## 6. What is preserved before the run

`${PROJ}/runs/t39-baseline-seed0/_archive-187813-oracles/` holds job 187813's `eval-t39-oracle-state`
and `eval-t39-oracle-action` complete — `bench.json`, `bench_0.2.0.json`, `e1.json`,
`predictions.jsonl`, `timing.json`. Verified in place: `oracle_action −359.4077743907937`,
`oracle_state +100.0`. The eval job overwrites arm directories, and those two are the evidence
`PR-07-RESULT.md` cites, so they were copied before anything was submitted.

## 7. Open, and deliberately not resolved here

**The walltime is unverified.** Two policy arms × 1040 chunks of a 3B diffusion policy, sequential,
against a 02:00:00 wall, and **no ms/chunk figure exists anywhere in this repo** —
`74_probe_t39_policy_shim.sbatch` does two forward passes and records no timing. At ≥3.46 s/chunk
the job does not fit. This document registers the rule, not the schedule; measuring the rate before
committing the allocation is an operational step and is not a term of the rule.
