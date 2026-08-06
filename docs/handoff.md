# Handoff — resume here

Last session: **2026-08-06**. Branch `feat/local-gpu-and-isaac`. Live task: **T-39**, the positive
control (`docs/preregistration/PR-07-positive-control.md`).

This file is the cold-start entry point: what state the tree is in, what the next action is, which
decisions are already made and must not be re-litigated, and which facts were expensive to
establish so they are not re-derived. It is a working note, not a record — when T-39 reports, the
result goes in `.mc/tasks/todo/T-39-*.md` and `PR-07-RESULT.md`, and this file gets rewritten.

**Tasks moved (2026-08-06).** `TASKS.md` is now a milestone index; each task is its own file under
`.mc/tasks/{todo,done}/` in MissionControl format, prose carried over unchanged. `mc task next`
answers "what now" (→ T-39, with T-32 correctly filtered out as blocked). Conventions, and why
`mc validate`'s filename check is expected to fail here: `.mc/README.md`.

---

## 0. Read this before touching anything

**The branch has zero commits. All the work is in the working tree** — 25 modified files, 31
untracked. `git log main..HEAD` is empty. That spans three separate efforts (local-GPU/Isaac, the
backbone eval, T-39), none of them checkpointed.

```bash
git status --short          # 56 entries
git stash list              # must stay empty — nothing here is stashed
```

Nothing is committed, pushed, submitted to Slurm, or deployed. **Committing is the first thing to
ask about tomorrow**, before any new file is added to the pile. Splitting it into three commits
along those three efforts is the obvious shape; it has not been done because commits were never
authorised.

---

## 1. The next action

**T-39 is written and deliberately not runnable.** The gate is in git before the cluster is
touched, which was the point:

| artifact | state |
|---|---|
| `docs/preregistration/PR-07-positive-control.md` | written, 9 sections |
| `cluster/discoverer/70_train_t39_baseline.sbatch` | written, `bash -n` clean |
| `cluster/discoverer/71_eval_t39_control.sbatch` | written — contains the executable rule `T39_RULE_V1` |
| `scripts/train_t39_baseline.py` | ✅ 2026-08-06 — subset view + witness, trainer as a subprocess |
| `scripts/eval_t39_baseline.py` | ✅ 2026-08-06 — the adapter, the four arms, both bench specs |
| `tests/test_t39_baseline.py` | ✅ 2026-08-06 — 31 tests, 12 mutants introduced and killed |

**Three of the six prerequisites are done (PR-07 §8). Three remain, and T-39 is still not
submittable:**

4. `$PROJ/virt_envs/t39` on Discoverer+ — a separate venv. The vendored trainer pins its own torch
   and attention kernels; `70_*.sbatch` exits FATAL if it is missing rather than importing into the
   WAM env. **Needs SSH.**
5. A verified `MODEL_ID`. It has **no default on purpose** — the exact checkpoint id and revision
   were never confirmed from a primary source, and guessing it inside the sbatch would put an
   unverified string into the artifact. **Needs a primary source, not a recollection.**
6. `TRAINER_ENTRYPOINT` and `POLICY_ENTRYPOINT`, added 2026-08-06 for the same reason as (5).
   Writing the drivers made explicit that we do not know the vendored trainer's entrypoint path or
   its inference API from a primary source either. The inference contract the eval needs is small
   and stated in `eval_t39_baseline.load_commanded_policy`; a shim in the t39 venv may adapt to it
   but may **not** convert into canonical units — that happens once, in our code, shared with the
   oracle. Also needs `third_party/isaac-gr00t` vendored (unmodified).

Everything local and free is now done. What is left needs the cluster or a source document — and
**submission is the user's call, every time.**

### What the implementation surfaced

Worth knowing before reading the code, because both are corrections to yesterday's files:

- **`70_*.sbatch` now also passes `--wam-dataset`,** and had to. The trainer eats the LeRobot
  *source*, but `dataset_snapshot_ref` must be taken over the *converted* episodes — that is what
  `eval_t16.verify_split` recomputes it against. As first written, the witness could not have been
  verified at all. No threshold moved; the rule in `71_*.sbatch` is untouched.
- **Two of the twelve mutants survived the first version of the test suite** — a gripper channel
  read one step late, and the wrong hand selected. Both slipped through because the
  `oracle_action` test compared only `targets`. The joint delta spans `t → t+1` while the gripper
  is sampled *at* `t+1`, so the two channels are anchored differently by construction and one
  convention cannot cover both. Recorded in PR-07 §8 rather than quietly fixed.

## 2. Cluster rules that bind

- **The Discoverer+ login node is off limits.** Over SSH: `sbatch`, `squeue`, file management. Never
  anything that computes — it risks the whole allocation.
- Account/qos `ehpc-aif-2026pg01-905`, partition `common`, 4 h max walltime, 5 000 GPU-h budget.
  `$PROJ=/valhalla/projects/ehpc-aif-2026pg01-905`. Every job sources `caches.sh`.
- T-39's ceiling is **12 GPU-h**, enforced by `MAX_RESTARTS=2` (3 × 4 h). T-32 by comparison is
  ~109 GPU-h and is **blocked behind T-39** — that ordering is pre-registered, not a preference.
- Compute nodes have Internet; the login node does not. Weights must be staged, hence `MODEL_DIR`.

## 3. Decisions already made — do not reopen

Each of these was argued out and written down. Reopening one costs a session.

- **The gate invents nothing.** The bar is WAM-Bench's own ladder (L1 `skill_vs_repeat_pct > 0`,
  L2 `ci_skill_vs_repeat_pct > 0`), and the single margin `MATERIAL_FLOOR_PP = 10.0` is *borrowed*
  from `I8_RULE_V3` rather than coined, so the choice of floor cannot become the finding.
- **Rules are versioned, never edited in place.** A gate rewritten after seeing its output is not a
  gate. `T30_RULE_V2` keeps V1's defects visible; PR-05's G2 is recorded VOID rather than patched.
  If `T39_RULE_V1` turns out wrong, the fix is `V2` alongside it.
- **The trainer is vendored unmodified** under `third_party/`. A positive control run through our
  reimplementation of someone else's recipe is not a positive control. Our code appears in exactly
  two places: the episode restriction and the eval adapter.
- **The two oracle arms run first and can veto the experiment.** `oracle_state` is the identity
  check on our own label pipeline (anything but ~perfect is our bug). `oracle_action` is the one
  worth the most — if the dataset's own `action` column cannot clear L1 under our scorer, then no
  policy trained on it can, T-39 is VOID, and every number in `docs/benchmark.md` is bounded by a
  label-space mismatch nobody had measured.
- **Verdict N is gated symmetrically.** Here the *negative* is the expensive conclusion (it licenses
  "stop trying methods on this corpus"), so N needs the material margin *and* the `train40` arm.
- **Exactly one conditional second candidate** (`lerobot/pi05_base`, only on N, "attempt 2 of 2").
  There is no attempt 3 under this pre-registration. This is what closes the p-hacking path.

## 4. Facts established the hard way

Verified with receipts. Do not re-derive, and do not trust older prose that contradicts them.

- **AC-07 is closed, not pending.** Re-scored **2026-08-01** on a laptop (`"device": "mps"`), zero
  allocation — `runs/{t18-real-ablation,d1-full-gen}-seed0/rescore-{history,tiled}/`,
  `docs/improvements.md:1065`. Six docs still claimed "pending a ~0.4 GPU-h re-score"; all six are
  now corrected. *I asserted the stale version last session and was wrong.*
- **`ci_` means task-CRITICAL chunks, not confidence interval** (`src/wam/evaluation/benchmark.py:180`).
  It is a subset metric. Anything that reads it as an interval is a bug.
- **The clean same-backbone ablation** is `t18-real-ablation-seed0` (−129.00 %) vs `d1-full-gen-seed0`
  (−20.88 %): the world branch costs **108 pp**. `t16-lora-seed0` at −21.80 % vs −20.88 % is **not**
  a clean ablation — backbone *and* branch differ.
- **The frame-mode confound is backbone-specific**: 10.65 pp for Wan, ~0.03 pp for `tiny`.
- **Test suite: 1 618 passed, 0 skipped, ~58 s.** Older counts (583/604/617/861/1 091) are left in
  `TASKS.md` as period record — they are history, not current.
- **`ruff check .` passes again, and now says what it checked** (fixed 2026-08-06). The 10 errors
  were never a source regression — the bare `[tool.ruff]` named no rules, so the repo inherited
  whatever ruff's default was that week, and 0.16.0's default is a different set. The fix is the
  two things that were missing, not a fix to `docs/anim/`: `ruff==0.16.0` is pinned in the `dev`
  and `local` extras (`pyproject.toml:62`, `:105`) and the rule set is named in `[tool.ruff.lint]`
  (`pyproject.toml:158-160`), `select = ["E4","E7","E9","F"]`. Clearing the 10 needed no edit in
  `docs/anim/`, no `# noqa` and no per-file-ignore — the later E731/E702 fix in
  `docs/anim/make-poster.py` is a separate, deliberate source change (next bullet), not part of it.
  Meta-lesson stands, now with both halves closed:
  *a lint claim with neither a pinned linter nor a named rule set has an expiry date.*
- **"E4/E7/E9/F" was not the state the repo was clean in** — that was an inference from ruff's
  historical default, and it is wrong. Measured 2026-08-06 under ruff 0.16.0, selecting E4/E7/E9/F
  reported **221** findings, none of them the 10. Plain `.venv/bin/ruff check . --statistics` does
  **not** show them: it reads `pyproject.toml`, whose `ignore` suppresses all of them, so it prints
  nothing and exits 0. The command that reproduces the count is the explicit override

  ```
  .venv/bin/ruff check . --statistics \
      --config 'lint.select=["E4","E7","E9","F"]' --config 'lint.ignore=[]'
  ```

  0.16.0's default does *not* contain E402 or F405 (`ruff check --isolated
  scripts/audit_gripper.py` → "All checks passed!"; `--isolated --select E4` → E402 at line 30).
  The 221 were five codes — F405 158× (all from the star import at
  `docs/anim/make-receding-horizon.py:39`), E402 58× (the `sys.path.insert` script bootstrap),
  E731 2×, E741 2×, E702 1×. The last three (5 sites) were the source-fixable ones and **were fixed
  in source on 2026-08-06** (see below), so the same override command now reports **216** and the
  `ignore` is down to two codes, listed with counts and reasons next to the `select`.
  The select is not vacuous: a probe with an unused import, an undefined name, `== None` and a
  bare `except` still draws F401, F821, E711 and E722 — and E711 is *not* in 0.16.0's default,
  so the named set is stricter than the default it replaces, not weaker.
- `ruff` is not on `PATH`; use `.venv/bin/ruff`.

## 5. Loose ends, ranked

1. **Commit the branch** (§0). Highest value, needs one word of approval.
2. ~~`docs/anim/` lint~~ — **done 2026-08-06**, by pinning *and* naming the rules (§4). It left five
   `ignore` codes behind; the three source-fixable ones (E731/E741/E702, 5 sites) were then **fixed
   in source the same day** and their entries deleted — `def` instead of lambda
   (`docs/anim/make-poster.py:33`, `scripts/hf_job_wan_probe.py:553`), `l` → `lbl`/`lam`
   (`hf_job_wan_probe.py:792`, `tests/test_bench_ridge_baseline.py:967`), semicolon split
   (`make-poster.py:56`). `ignore` is now `["F405", "E402"]` — 216 findings, not lint work.
   `ruff check .` exits 0 and the suite still reports 1 618 passed.
3. ~~Rebuild `wam_03_t24_bakeoff`~~ — **done 2026-08-06**. Repointed at
   `runs/backbone_eval/reports/{wan,cosmos}_ep48.json` and both mp4s rebuilt (22.9 s).
   **The headline inverted**: at 48 episodes Cosmos3-Nano leads Wan on both channels *and* on the
   val-selected pair (0.43 › 0.39), so the old "Wan bleibt" card was a single-corpus-size
   conclusion. What survives at every corpus size is that neither backbone reaches the state-only
   floor. Two defects the repoint would otherwise have created were fixed with it: the left panel's
   episode is now derived from `info.probe.split_episodes.test[0]` (000036, not the hard-coded
   000010, which the 48-episode split no longer holds out), and `check_same_experiment` now also
   compares the two reports' `state_only` rows — the dashed floor is read from the Wan file and
   drawn across both curves. `runs/presentation/README.md` carried the same stale ranking and was
   corrected too.
4. ~~`docs/hf_jobs.md` sweep~~ — **done 2026-08-06**. All 472 lines checked, ~95 claims: 11 fixed,
   5 marked unverified, the rest receipted. Real defects found: a mislabelled table row ("block pair
   chosen on val" was the best single block), a wrong state-only width (52 → 32), a cross-backbone
   comparison mixing a single block against a pair, two GPU wall times contradicted by the archived
   logs (73 s → 16.1 s, ~33 s → 18.3 s), and a sampling time contradicted by the chunk reports
   (~165 s → 145–146 s). The vendor claims (HF quota, billing, flavor specs) have no repo artifact
   and are now labelled as such in one place rather than read as measurements.
5. **Two wrong numbers found outside the swept files, fixed 2026-08-06.** `docs/backbone-eval.md:33`
   compared Cosmos3's best *single block* gripper 0.822 against "Wan's 0.698", which is Wan's
   `suggested_2_10` **pair** — Wan's best single block is **0.734** (block 6), so the gap was
   8.8 pp, not 12.4. And `src/wam/evaluation/dream.py:78` justified `FREEZE_MARGIN` with "96 % of
   frame pairs move less than one grey level", which is a t35 per-clip figure; the corpus statistic
   the margin was actually set against is **68.9 %** (`PR-05:44`, 200 clips over 40 episodes).
   Neither changes a gate — both were load-bearing prose attached to a number from the wrong run.

## 6. Verify the tree is where this note left it

```bash
.venv/bin/python -m pytest -q                       # expect: 1618 passed, ~58s
.venv/bin/ruff check .                              # expect: All checks passed! (ruff 0.16.0)
bash -n cluster/discoverer/7{0,1}_*.sbatch          # expect: silent
```

The split-witness test now globs `*_eval_*.sbatch`, not `6?_eval_*` — the old pattern would have
left `71_eval_t39_control.sbatch` silently exempt from the one check that exists precisely because
eval jobs kept dropping `--train-episodes` (`tests/test_eval_t16.py`).

---

## Standing constraints

**Nothing gets submitted, committed, pushed, deployed, or paid for without asking first.** The
pre-registration is in git ahead of the cluster deliberately; that ordering is the whole method, and
it only holds if the submission decision stays with the user.
