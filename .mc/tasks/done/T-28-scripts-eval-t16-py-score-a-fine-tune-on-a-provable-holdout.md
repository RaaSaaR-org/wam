---
id: T-28
aliases:
- T-28
title: "`scripts/eval_t16.py` — score a fine-tune on a provable holdout"
slug: scripts-eval-t16-py-score-a-fine-tune-on-a-provable-holdout
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- backbone
- eval
- cluster
- sim
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-30
updated: 2026-08-01
---

# `scripts/eval_t16.py` — score a fine-tune on a provable holdout

## Description

`scripts/eval_t16.py` — score a fine-tune on a **provable** holdout, plus the local-GPU runbook
(`docs/local_gpu.md`, 9 + 2 tests) — *`train_t16_lora.py` writes weights and a training log and no
evaluation, on purpose: the fine-tune runs in preemptible 4-hour chunks, so an eval welded to the
end of one would run on whichever chunk happened to finish. That left the measurement ladder with no
input — T-16 could complete and still produce no verdict. This is that step, standalone: one GPU
pass over the holdout writing `predictions.jsonl` + `e1.*` + `bench.*` (~960 chunks, minutes, ~0.2
GPU-h), after which every score is CPU-only and re-runnable forever. **The split is proven, not
asserted:** the trainer hashes the manifests of the episodes it actually trained on into
`dataset_snapshot_ref`, so the evaluator recomputes that hash over `dataset − holdout` and REFUSES
TO SCORE unless they match — a holdout that is not the complement of the training set may have been
trained on, and a number computed on it is meaningless in the one way that matters.
`--skip-split-check` exists as an escape hatch and drops an `UNPROVEN_SPLIT` marker so an unproven
number never looks proven. Three pieces of duplication collapsed on the way: `build_eval_pairs` and
the split-file reader moved into `wam.evaluation` (`run_ablation.py` imported the former **from
another script**), and `load_joint_policy` now centralizes a bug neither `rollout.py` nor
`serve_policy.py` had noticed — both constructed `JointCheckpointPolicy` directly, which cannot load
a Wan-backed checkpoint at all, because `WanFlowBackbone` keeps the 5B DiT out of the module tree so
the file holds only `backbone.lora.*` and `backbone.state_proj.*`. The branch is taken from the
embedded config's new `requires_external_weights`, which travels inside the checkpoint and cannot go
missing like a sidecar. `docs/local_gpu.md` is the runbook for a single 32 GB card: inference needs
**one** backbone pass (`predict()` at the clean flow timestep, video velocity discarded — no
denoising loop) over 60 tokens per sample, so **~24 GB bf16 resident** (measured 24.3 GB peak on an
H200; the ~12 GB this originally said was the *offloaded* budget presented as the default, and
`--offload-text` is not wired into any of the three entry points — `docs/local_gpu.md` records the
correction); fine-tuning stays on Discoverer+*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
