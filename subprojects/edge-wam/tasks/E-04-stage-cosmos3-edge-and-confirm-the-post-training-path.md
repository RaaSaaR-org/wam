---
id: E-04
subproject: edge-wam
title: "Stage Cosmos3-Edge and confirm a post-training path exists"
slug: stage-cosmos3-edge-and-confirm-the-post-training-path
status: todo
priority: 2
owner: ''
tags:
- edge
- cosmos3
- cluster
- staging
depends_on: []
blocks:
- E-06
created: 2026-08-15
updated: 2026-08-15
status_note: "Not started. Needs SSH (available again since 2026-08-15) and a download decision from the user. The sources contradict each other on whether post-training scripts ship at all."
---

# Stage Cosmos3-Edge and confirm a post-training path exists

## Description

Two jobs, both prerequisites for any training:

**1. Does a post-training recipe actually ship?** The NVIDIA blog and the launch coverage say Edge
comes with post-training scripts [doc]. The model card says no explicit fine-tuning scripts are
provided and points users at "the Cosmos Framework" instead [✓]. One of those is wrong, and the
difference is whether E-06 adapts a recipe or writes one. Resolve from the repo, not the prose.

**2. Stage the weights.** 4B under OpenMDW-1.1. Compute nodes on Discoverer+ have Internet, the
login node does not, so this follows the existing `MODEL_DIR` staging pattern — not a download on
the login node, which is a never-do.

**Licence check is part of this task, not an afterthought.** T-041 lost job 187249 to
`nvidia/Cosmos-Guardrail1` being a **gated** repo; accepting a licence is the account holder's act,
not an agent's. Verify up front whether `nvidia/Cosmos3-Edge` is gated, and if it is, stop and ask
rather than discovering it inside a job.

## Acceptance

1. Post-training path resolved: recipe file and entrypoint named with a path, or "none ships" stated
   with the evidence.
2. Gated/ungated status of the HF repo confirmed **before** any job is written.
3. Weights staged under `$PROJ` (never `/home` — 2 GB and ~100 k inodes), with size recorded and
   `caches.sh` sourced.
4. A `sbatch` file that follows `cluster/discoverer/README.md`'s seven never-do rules: explicit
   `--qos`, `--gres=gpu:1` unless justified, ≤26 threads and ≤257 GB per GPU, checkpoints under
   `${PROJ}/runs`.
5. **Submission is the user's call** — this task ends at a written, `bash -n`-clean job file.

## Notes / Report

*(empty — fill in when the task runs)*
