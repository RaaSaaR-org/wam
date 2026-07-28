# Discoverer+ job scripts

Ready-to-`sbatch` files for the EuroHPC H200 partition (PetaSC / Sofia Tech Park), allocation
`ehpc-aif-2026pg01-905`. Machine facts, quotas, billing and gotchas: **`docs/discoverer.md`** —
that is the *why*, this is the *how*.

**Nothing here has been executed yet. Allocation untouched: 5 000 GPU-h, 0 % used.**

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
| Support | [help](https://docs.discoverer.bg/help.html) — see rule 6 |

## Files here

```
caches.sh              cache/scratch redirection — sourced by every job (rule 2)
sync.sh                push repo + dataset from the Mac
10_build_env.sbatch    conda prefix env                    free
20_stage_weights.sbatch Wan weights (+ --groot dataset)    free
25_requeue_probe.sbatch proves --requeue/--signal          free
30_smoke_wan.sbatch    Wan adapter on one H200             ~0.25 GPU-h
40_readout_probe.sbatch re-measure feature blocks          ~1.5 GPU-h
50_train_t16.sbatch    the LoRA fine-tune                  the budget
```
