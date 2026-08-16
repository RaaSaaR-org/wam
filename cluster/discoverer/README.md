# Discoverer+ job scripts

Ready-to-`sbatch` files for the EuroHPC H200 partition (PetaSC / Sofia Tech Park), allocation
`ehpc-aif-2026pg01-905`. Machine facts, quotas, billing and gotchas: **`docs/discoverer.md`** —
that is the *why*, this is the *how*.

**Steps 1–8 have run** — the environment build, the weight staging, the Wan smoke job, the
readout probe, T-16 itself (20 000 LoRA steps, `runs/t16-lora-seed0`, negative: WAM-Bench L0,
`skill_vs_repeat_pct` −32.4 %, a *tiled* number), and T-29, which re-measured that same
checkpoint through the real frame window: −21.80 % (+10.65 pp). L1 is still failed by 21.80 pp,
so the verdict survives and the published figure does not. Step 9 (T-30) **ran 2026-08-01 as job
184670 and is negative** — every flow arm below L0. **Steps 10 and 11 ran 2026-08-16** (jobs 187804
and 187813): the verdict is `VOID (labels)`, and PR-12 (`C`) then PR-13 (`W`) traced that VOID to
our own evaluation adapter rather than the corpus — repaired, the corpus's own action column scores
**+68.10 L1, level L4** on T-39's own holdout. **Step 11's POLICY arm still has not run** (G0 fires
before the policy branch), so steps 12–13 remain blocked behind 11 and the training decision that
would unblock them is the project owner's.

**No document here asserts how much of the allocation is left** — that number moves every job and a
stale copy of it is worse than none. Read it live with `accountcheck` (`docs/discoverer.md` §9).

---

## ⛔ NEVER DO THIS

**1 — Never run work on the login node.** `login-plus` is an administrative shell. Forbidden by
name: `conda`, `pip`, compilation, downloads, notebooks, IDEs, and **AI coding agents over
Remote-SSH — the provider names VSCode, Claude Code and OpenHands directly**. Violations "can
result in account restrictions or termination"
([login-node limits](https://docs.discoverer.bg/gpu-login-node-resource-limits.html)).
Permitted there: `sbatch`/`squeue`/`scancel`, allocation checks, and file management under
project storage. **Everything else is a Slurm job.** Enforcement is real: CPUQuota 200 %,
`MemoryHigh` 4 GB — exceeding it puts processes in D-state, which looks like a hung filesystem,
not like an error message.

**2 — Never put anything big in `/home`.** 2 GiB and ~100 k inodes. One HF model or one careless
`pip install --user` fills it, and then nothing works. `caches.sh` redirects every cache
(HF, torch, triton, pip, conda, `TMPDIR`) to project storage — source it in every job.

**3 — Never exceed 26 threads or 257 GB per GPU.** `330 900 billing-h ÷ 5 000 GPU-h = 66.18`,
which is exactly the fair-share rate for **one** GPU. Ask for more and the billing counter
empties before the GPU-hours do, **permanently stranding the rest of the allocation**. Every job
here stays at `--cpus-per-task=26 --mem=192G --gres=gpu:1`
([billing explainer](https://docs.discoverer.bg/slurm-gpu-billing-explainer.html)).

**4 — Never omit `--qos`.** Without it a job lands on `normal`: 1 minute, 0 GPUs. It does not
fall back to something permissive, it falls into a trap. GPU work needs
`--qos=ehpc-aif-2026pg01-905`; free CPU work needs `--qos=2cpu-single-host`. A CPU-only job under
the project QoS is rejected, and a GPU job under `2cpu-single-host` is rejected.

**5 — Never `mv` between storage systems.** Group ownership does not follow, and the quota
accounting silently breaks. Copy, verify, then delete
([moving files](https://docs.discoverer.bg/storage_moving_files_overquota.html),
[disk usage](https://docs.discoverer.bg/calculating_the_disk_usage_basics.html)).

**6 — Never mail the helpdesk from a free provider.** Gmail and friends may go unanswered — use
the institutional address on file ([help](https://docs.discoverer.bg/help.html)). Note the git
identity on this workstation is a gmail address.

**7 — Never leave checkpoints in scratch.** Scratch is reaped after 61 days without access.
Checkpoints belong in `${PROJ}/runs` ([scratch](https://docs.discoverer.bg/scratchfolder.html)).

---

## Order

| # | Job | QoS | Cost | Does |
|---|-----|-----|------|------|
| 0 | `sync.sh --data` | — | — | rsync repo + 81 MB dataset + `caches.sh` (run on the Mac) |
| 1 | `10_build_env.sbatch` | `2cpu-single-host` | **free** | conda prefix env at `$PROJ/virt_envs/wam` |
| 2 | `20_stage_weights.sbatch` | `2cpu-single-host` | **free** | Wan2.2-TI2V-5B (~20 GB) from HF; `--groot` also pulls the raw LeRobot snapshot |
| 3 | `25_requeue_probe.sbatch` | `2cpu-single-host` | **free** | proves `--requeue` + `--signal` before step 6 depends on them |
| 4 | `30_smoke_wan.sbatch` | project | ~0.25 GPU-h | 13/13 adapter checks on one H200, peak VRAM |
| 5 | `40_readout_probe.sbatch` | project | ~1.5 GPU-h | re-measure feature blocks `[2, 10]` on this stack |
| 6 | `50_train_t16.sbatch` | project | the budget | the LoRA fine-tune |
| 7 | `60_eval_t16.sbatch` | project | ~0.2 GPU-h | score the checkpoint on the proven holdout |
| 8 | `61_eval_t29_frame_history.sbatch` | project | ~0.4 GPU-h | T-29 / I-7: tiled frame vs. the real window — **ran** 2026-08-01 (job 184648): −32.45 % → −21.80 %, L1 still failed |
| 9 | `63_eval_t30_flow_head.sbatch` | project | ~4 GPU-h ×1–2 | T-30 / I-3: regression head vs. the flow sampler — **ran** 2026-08-01 (job 184670): all 10 arms below L0, mean-of-8 arm 11.1× worse than the regression readout |
| 10 | `70_train_t39_baseline.sbatch` | project | ≤8 GPU-h | **T-39 / PR-07, the positive control** — NVIDIA's own recipe on our committed split — **ran** 2026-08-16 (job 187804, 1:22:14, 1.37 of the 12 GPU-h ceiling) |
| 11 | `71_eval_t39_control.sbatch` | project | ~0.5 GPU-h | four arms + the pre-registered `T39_RULE_V1` verdict — **verdict `VOID (labels)`**, decided by G0 on the oracle arms. **Its POLICY arm has never run**: job 187813 died at 108 s on a missing `GROOT_PATCH_MISTRAL` export (since fixed), and G0 fires before the policy branch, so the checkpoint step 10 wrote has never been scored. Read `MODEL_DIR`'s guard in the file before resubmitting |
| 12 | `55_train_i8_rung.sbatch` | project | ~36 GPU-h ×3 | I-8 / T-32 rungs 040 / 120 (+ a seed-1 replicate) — **blocked on step 11** |
| 13 | `62_eval_i8_curve.sbatch` | project | ~1 GPU-h | both frame modes × 3 rungs, then the pre-registered verdict |

The `#` column is execution order, not the filename number: `55_` sorts before `60_` because it
is a *training* script, but it runs after step 8.

**Steps 10–11 run before 12–13, and that ordering is pre-registered** (`PR-07` §2). T-32 spends
~109 GPU-h fitting a scaling curve on a method no positive control has ever validated on this
corpus; if the method is simply broken here, every branch of `I8_RULE_V3` describes the scaling of
brokenness. T-39 costs an order of magnitude less and is what makes T-32 readable.

Steps 1–3 cost nothing. Run them first — they catch every environment problem before a single
GPU-hour is spent.

## Before the first sbatch

```bash
ssh-add ~/.ssh/id_ed25519_eu_ai_hub     # key is passphrase-protected, must be in the agent
./cluster/discoverer/sync.sh --data
ssh ffromm@login-plus.discoverer.bg
cd /valhalla/projects/ehpc-aif-2026pg01-905/wam/cluster/discoverer
sbatch 10_build_env.sbatch && sbatch 20_stage_weights.sbatch --groot
```

SSH keys are served from LDAP — you cannot self-install or rotate them; ask the helpdesk.

## Why step 6 looks the way it does

The QoS caps every job at **4 hours** and the partition runs `PreemptMode=REQUEUE` on shared
nodes (`OverSubscribe=FORCE:4`). A LoRA run over 402 episodes does not fit in one job, so a run is
a *chain* of jobs and checkpoint-and-resume is the architecture, not an optimisation:
`--requeue`, `--open-mode=append`, `--signal=B:USR1@300`, a job-id-independent output dir, and a
`DONE` sentinel so the chain terminates. Step 3 exists because `--requeue` and `--signal` are
standard Slurm but documented nowhere on this site — prove them for free before betting a
4-hour job on them.

T-16a (2026-07-27) closed the code side: `FlowBackbone` protocol, backbone-agnostic joint model,
LoRA on the Wan DiT via `WanFlowBackbone` (the 5B DiT stays out of the module tree, so
adapter-only checkpoints are structural), and `scripts/train_t16_lora.py` with
SIGUSR1→checkpoint→exit 0 and `--resume latest`. A test proves bitwise equality between an
uninterrupted run and one interrupted and resumed — that is what protects the chain. Caveat:
proven on the tiny backbone; the Wan `--save-adapter-only` path is covered for key sets and
completion, not bitwise.

The queue is full of 4/7/8-GPU jobs on a 15-GPU machine, so `--gres=gpu:1` is also the biggest
throughput lever we have — and a 5B LoRA needs exactly one H200.

**Step 10 reuses that mechanism unchanged and is deliberately NOT a job array.** `55_train_i8_rung`
is `50_train_t16` with the rung block and two extra trainer flags; the requeue machinery
(`--signal=B:USR1@300`, python in the background with `PY=$!`, `MAX_RESTARTS`, the four exit
branches) is copied verbatim rather than re-derived, because that block is what protects a
20 000-step run. An array would be the obvious "simplification" and is the one thing not to do:
job arrays are an open question on this cluster (`docs/discoverer.md`, open questions item 8 —
"Arrays appear in `squeue`, so probably yes", against a provider whose docs this file already
marks contested), and an array task's interaction with `scontrol requeue` / `SLURM_RESTART_COUNT`
is exactly the unknown that, if wrong, produces the unbounded requeue loop `MAX_RESTARTS` exists
to stop. Three plain submissions sit inside the 4-running cap and give each rung its own `DONE`,
its own restart counter and its own blast radius.

Step 10 also refuses to start until `runs/t16-lora-seed0/eval-t29-history/bench.json` exists:
rung 362 of the I-8 curve *is* step 8's output, so a T-29 verdict nobody has read yet means I-8's
premise is unknown. `SKIP_T29_CHECK=1` overrides it — only after reading that verdict. That
verdict has been read (2026-08-01): the real window did not clear L1, so I-8's premise stands,
the gate passes on its own, and rung 362 contributes −21.80 %, not the published −32.4 %.

Step 11 is the **pre-registration**: every threshold in the I-8 decision rule is a literal in
`62_eval_i8_curve.sbatch`, and the file is committed before the first rung is submitted. If a
constant has to change, add a new rule version next to it; never edit a number in place, or the
runs already scored stop being re-derivable.

## Provider docs

Treat these as *contested*: several pages were wrong when checked against the live machine on
2026-07-27 (`common-gpu` does not exist, the node/GPU counts are inflated, the CUDA module name
is published in three forms and none of them is the installed one). `docs/discoverer.md` §3 marks
what was verified with `[✓]`.

| Topic | Page |
|-------|------|
| Start here | [software](https://docs.discoverer.bg/software.html) · [resource overview](https://docs.discoverer.bg/resource_overview.html) |
| Login + SSH | [login nodes](https://docs.discoverer.bg/login_nodes.html) · [logging in (plus)](https://docs.discoverer.bg/ssh_logging_in_plus.html) · [host fingerprints](https://docs.discoverer.bg/ssh_key_fingeprint_plus.html) · [key generation](https://docs.discoverer.bg/ssh_key_generation.html) · [LDAP keystore](https://docs.discoverer.bg/quantum-resistant-openssh-keystore-in-ldap.html) |
| **Login-node limits** | [gpu-login-node-resource-limits](https://docs.discoverer.bg/gpu-login-node-resource-limits.html) — read rule 1 above |
| Slurm | [writing batch scripts](https://docs.discoverer.bg/writing_slurm_batch.html) · [job control](https://docs.discoverer.bg/job_control.html) · [resource allocation](https://docs.discoverer.bg/computational_resources_allocation.html) · [GPU billing](https://docs.discoverer.bg/slurm-gpu-billing-explainer.html) |
| Storage | [project folder](https://docs.discoverer.bg/projectfolder.html) · [scratch](https://docs.discoverer.bg/scratchfolder.html) · [disk usage](https://docs.discoverer.bg/calculating_the_disk_usage_basics.html) · [moving files / over quota](https://docs.discoverer.bg/storage_moving_files_overquota.html) · [file transfers](https://docs.discoverer.bg/filetransfers.html) |
| Python + GPU | [pytorch](https://docs.discoverer.bg/pytorch_gpu.html) · [conda](https://docs.discoverer.bg/conda_gpu.html) · [virtualenvs](https://docs.discoverer.bg/python_virtual_environments_gpu.html) · [where to compile](https://docs.discoverer.bg/where_to_compile.html) |
| Worked examples | [Kimi-K26 vLLM on H200](https://docs.discoverer.bg/kimi-k26-vllm-h200-guide.html) · [multinode](https://docs.discoverer.bg/kimi-multinode-discoverer.html) |
| 10 | `sync.sh --pull` | — | — | rsync results + Slurm logs BACK to the Mac (run on the Mac) |
| Support | [help](https://docs.discoverer.bg/help.html) — see rule 6 |

## Files here

```
caches.sh              cache/scratch redirection — sourced by every job (rule 2)
sync.sh                push repo + dataset from the Mac; --pull fetches results back
10_build_env.sbatch    conda prefix env                    free
20_stage_weights.sbatch Wan weights (+ --groot dataset)    free
25_requeue_probe.sbatch proves --requeue/--signal          free
30_smoke_wan.sbatch    Wan adapter on one H200             ~0.25 GPU-h
40_readout_probe.sbatch re-measure feature blocks          ~1.5 GPU-h
50_train_t16.sbatch    the LoRA fine-tune                  the budget
55_train_i8_rung.sbatch one I-8 data-scaling rung          ~36 GPU-h each
60_eval_t16.sbatch     score the checkpoint (the verdict)  ~0.2 GPU-h
61_eval_t29_frame_history.sbatch  T-29 frame-mode A/B      ~0.4 GPU-h, ran 2026-08-01
62_eval_i8_curve.sbatch the I-8 curve + its decision rule  ~1 GPU-h
63_eval_t30_flow_head.sbatch      T-30 readout A/B         ~4 GPU-h, resubmit to continue
70_train_t39_baseline.sbatch      T-39 positive control    <=8 GPU-h, own venv virt_envs/t39
71_eval_t39_control.sbatch        T-39 arms + T39_RULE_V1  ~0.5 GPU-h
```

`70_`/`71_` use **`$PROJ/virt_envs/t39`**, not `virt_envs/wam`. The vendored trainer pins its own
torch and attention kernels, and importing it into the WAM env is how one dependency resolution
quietly changes every WAM number afterwards. Neither job is submittable yet — `PR-07` §8 lists
what must exist first.
