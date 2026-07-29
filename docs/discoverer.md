# Discoverer+ — the GPU cluster for T-16

**What it is:** an NVIDIA H200 partition of the Bulgarian EuroHPC supercomputer *Discoverer*,
operated by PetaSC / Sofia Tech Park. WAM has a personal account on it under the EuroHPC
AI-Factory allocation `ehpc-aif-2026pg01-905`.

**Why we care:** T-16 (the Wan2.2-TI2V-5B LoRA fine-tune) is the one open claim in the whole
project — "video helps" currently rests on it, and it is the only milestone that needs a GPU
that can hold a training run for hours. ZeroGPU cannot; Discoverer+ can. This closes the
compute half of OD-05.

**Runnable form:** the job scripts in this document are checked in as `cluster/discoverer/`
(sync script + six `sbatch` files, in order). This document is the *why* — machine facts,
quotas, billing, gotchas; that directory is the *how*. Its README opens with the seven
**never-do** rules and carries the full link table to the provider's documentation (24 pages,
all checked reachable 2026-07-28). Nothing has run on the cluster yet.

| | |
|---|---|
| Login node | `login-plus.discoverer.bg` (port 22, public Internet, no VPN) |
| Username | `ffromm` · key `~/.ssh/id_ed25519_eu_ai_hub` (passphrase-protected) |
| Slurm account / QoS | `ehpc-aif-2026pg01-905` · partition `common` |
| **Budget** | **5 000 GPU-hours + 330 900 billing-hours, 0 used** |
| **Max walltime** | **4 h per job**, max 4 running / 8 submitted jobs, ≤16 GPU + 416 CPU at once |
| Project storage | `/valhalla/projects/ehpc-aif-2026pg01-905` — 5 TiB, setgid |
| Home | `/home/ffromm` — 2 GB quota, 100 k inodes |
| Support | `helpdesk@discoverer.bg` (institutional address only — §9) |
| Docs | <https://docs.discoverer.bg> |

> **Verified on the machine 2026-07-27.** Everything tagged **[✓]** was read off `login-plus`
> in a live session. **[doc]** = documented but not yet confirmed on the machine. **[?]** = still
> open. Several official doc pages turned out to be wrong — those are called out inline so
> nobody "corrects" this file back to them.

---

## 1. Access

```bash
ssh-add ~/.ssh/id_ed25519_eu_ai_hub     # once per login session; prompts for passphrase
ssh ffromm@login-plus.discoverer.bg
```

That is the whole procedure. The key is registered in Discoverer's LDAP keystore and works [✓].

Two things to know for later:

- sshd resolves authorized keys from a **389 Directory Server via `AuthorizedKeysCommand`**, not
  from `~/.ssh/authorized_keys` [doc]. You cannot install or rotate your own key. A lost key is a
  total lockout only `helpdesk@discoverer.bg` can clear.
- If `ssh` fails with `Permission denied (publickey)`, check the agent first —
  `ssh-add -l` must list `SHA256:1WiG4/oXoh0I94LAAx49evKgazZZdpE/tE3v3AEZLn0`. A verbose trace
  saying `Server accepts key` followed by a denial means exactly this: right key, not unlocked.

Published host key (only one algorithm is published) [doc]:
`ecdsa-sha2-nistp521 SHA256:ceY8MM9O7KB7CipOOcm44wDboE+PyjGJlF5Xv6zM8Tw`

`~/.ssh/config`:

```
Host dplus
    HostName login-plus.discoverer.bg
    User ffromm
    IdentityFile ~/.ssh/id_ed25519_eu_ai_hub
    IdentitiesOnly yes
    AddKeysToAgent yes
    ServerAliveInterval 60
```

---

## 2. What the login node is for

An administrative shell, nothing more. Explicitly **forbidden**: conda, pip, compilation,
downloads, notebooks, IDEs and *"AI coding agents"* over Remote-SSH — the page names VSCode,
Claude Code and OpenHands directly, and warns violations "can result in account restrictions or
termination" [doc, [login-node limits](https://docs.discoverer.bg/gpu-login-node-resource-limits.html)].

Explicitly **permitted**: checking the allocation, and managing files under project storage.
Enforcement: CPUQuota 200 %, `MemoryHigh` 4.0 GB (processes go into D-state, which presents as a
hung filesystem), max 4 concurrent SSH sessions.

Everything else runs as a Slurm job.

---

## 3. Verified machine state (2026-07-27)

### Partition and nodes [✓]

```
PARTITION  AVAIL  TIMELIMIT  NODES  STATE   NODELIST
common*    up     infinite   2      mixed   dgx[1-2]

dgx1  gpu:8                224 CPUs  2063425 MB
dgx2  gpu:7, gpu_biz:1     224 CPUs  2063425 MB
```

- **`common` is the partition. `common-gpu` does not exist** (`Partition common-gpu not found`).
  The one docs page using it is stale.
- **2 nodes, 15 general-purpose GPUs** (dgx2's 8th is a separate `gpu_biz` resource, presumably
  commercial). The "4 nodes / 32 GPUs" prose in `resource_overview.html` is wrong, and the
  "128 × H200" on the operator's marketing page is wrong by a factor of 4 — that page's own
  4.5 TB aggregate GPU memory works out to 32, and Slurm reports 15+1.
- `DefaultTime=00:15:00`, `MaxTime=UNLIMITED` **at the partition level** — the real cap is the QoS.
- **`OverSubscribe=FORCE:4` and `PreemptMode=REQUEUE`** — nodes are *shared* between up to 4 jobs,
  and the partition is configured for requeue-on-preempt. Do not assume exclusive access, and
  write jobs that survive being requeued.
- Billing weights confirmed exactly as documented:
  `TRESBillingWeights=CPU=0.035714286,Mem=0.25G,GRES/gpu=1.0`

### QoS [✓]

| QoS | MaxWall | Limits | Jobs/user |
|---|---|---|---|
| `ehpc-aif-2026pg01-905` | **04:00:00** | `cpu=416, gres/gpu=16` | **4** |
| `2cpu-single-host` | 04:00:00 | `cpu=2, node=1` | 2 |
| `normal` | **00:01:00** | `gres/gpu=0` | 1 |

All carry `DenyOnLimit`; ours also `NoDecay`.

**The walltime question is settled: 4 hours.** Not the 2 h the login-limits page implies, not the
48 h the PyTorch guide's example uses. A T-16 LoRA run therefore *must* be a chain of checkpointed
jobs (§8).

Note `normal` — it is the default association and allows **1 minute and zero GPUs**. Omitting
`--qos` doesn't fall back to something permissive; it falls into a trap.

### Budget [✓]

```
ALLOCATED   billing: 330900.0 hours   gres/gpu: 5000.0 GPU-hours
USED        billing:      0.0 (0.0%)  gres/gpu:    0.0 (0.0%)
```

`330900 / 5000 = 66.18` — which is *exactly* the fair-share billing rate for one GPU
(26 threads × 0.035714 + 257 GB × 0.25 + 1.0 GPU). That is not a coincidence, and it gives a
hard sizing rule:

> **Per GPU requested, stay at or below ~26 CPU threads and ~257 GB RAM.** Go above and the
> billing counter runs out *before* the GPU-hours, and the docs are explicit that
> "over-consuming billing results in permanent loss of the remaining GPU-hours". Slurm bills
> **allocated**, not used, resources.

At 1 GPU per job that is 5 000 hours of runway — ~1 250 four-hour jobs. Not a constraint for T-16.

### Modules [✓]

The three names the docs disagree about resolve to one:

```
nvidia/cuda/12/12.8              # NOT nvidia/cuda/12, NOT cuda/12/12.8.0
nvidia/cudnn/cuda12/9.7.1.26
anaconda3/python3.12             # note: 3.12, not the 3.11 the docs recommend
python/3.12/gcc/base/3.12.9      # plain python modules DO exist, contra the docs
nvidia/cuda/11/11.8  ·  nvidia/hpcsdk/*  ·  ngc/3/3.60.2  ·  openmpi/*  ·  gcc/15/15.1.0
```

No `pytorch` module and no `ffmpeg` module on this cluster — both come from the conda prefix.

### Storage [✓]

```
drwxrws---  nobody  ehpc-aif-2026pg01-905   /valhalla/projects/ehpc-aif-2026pg01-905
umask 0022 · groups: ffromm, plus, ehpc-aif-2026pg01-905

/home       weka-nfs:/tmphome      nfs    39T   (2 GB *quota*, not a 2 GB filesystem)
/valhalla   10.106.1.1:/valhalla   nfs4  5.1P   10% used
```

- **The project directory is setgid** (`rwxrws---`, group `ehpc-aif-2026pg01-905`). Files inherit
  the project group automatically — the undocumented `chgrp`/`setgid` worry is a non-issue.
  Owner shows as `nobody` (NFS root-squash), which is normal.
- **`/valhalla` is mounted NFSv4 on the login node**, not native Lustre — so `lfs quota` /
  `lfs project` genuinely cannot run here, exactly as the docs say. Use a `2cpu-single-host` job.
- `quota -u` returns nothing — the documented NFS quota-reporting failure is real.
- Project dir is currently empty (just `._dbindex_`). Home holds 5 files / 16 K.
- **`/raid` does not exist on the login node** — it is node-local to dgx1/dgx2 [?].
- **`/weka` exists and is mounted**, but `/weka/projects/` contains no directory for us. Other
  projects have one — including a sibling AIF Playground project `ehpc-aif-2026pg01-555`. So it
  is allocated on request, and we do not have it. Ask (§10).

### Containers [✓]

`apptainer` and `singularity` **are installed** at `/usr/bin/` on the login node. The docs claim
otherwise (the term appears nowhere in their search index, and the GPU page argues against
containers). Whether they run on the compute nodes, and whether `--fakeroot` or registry access
work, is untested [?]. `enroot`, `podman` and `docker` are absent.

### Queue contention [✓]

The queue is busy — a long list of pending jobs from other AIF/DEV/BEN projects requesting
`gres/gpu:4`, `:7` and `:8`, plus job arrays. On a 15-GPU machine an 8-GPU request queues behind
everything.

> **Practical consequence: ask for `--gres=gpu:1`.** A 5B LoRA needs one H200 anyway, and small
> jobs schedule dramatically faster here. This is the single biggest throughput lever we have.

### Still open

`/weka` allocation · `/raid` usability from jobs · inode limit on the 5 TiB quota · Apptainer on
compute nodes · compute-node egress to the HF CDN · `ConsumedEnergy` · allocation end date ·
whether `seff`/`sstat`/`--overlap` work. See §10.

---

## 4. The machine

Per DGX H200 node [doc + [✓] where noted]:

- **8 × NVIDIA H200 SXM**, 141 GB HBM3e each, Hopper `sm_90`, NVSwitch intra-node
- 2 × Intel Xeon Platinum 8480C (Sapphire Rapids), 224 threads [✓], ~2 063 425 MB RAM [✓]
- 8 cores / 16 threads reserved for the WEKA storage client
- Node-local NVMe `/raid`, 24.5 TB — documented as "local scratch", no job example anywhere [?]
- Host CUDA toolkit 12.8, matching the `nvidia/cuda/12/12.8` module [✓]

**Sizing for us:** a bf16 LoRA on a 5B model is ~10 GB of frozen base weights plus a few hundred
MB of adapter and optimizer state — **one H200, with room to spare**. What pushes memory is
activation from the video latent sequence, so gradient checkpointing, not sharding. No FSDP, no
DeepSpeed, no multi-node.

---

## 5. Slurm

### The header every job needs

```bash
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905        # GPU work — omitting this lands you on `normal` (1 min, 0 GPUs)
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # mandatory on the GPU cluster
#SBATCH --gres=gpu:1                       # mandatory for anything touching a GPU
#SBATCH --time=04:00:00                    # the QoS cap
```

`--gpus`, `--gpus-per-node` and `--gpus-per-task` are not used here — `--gres=gpu:N` only.

| QoS | Use for | GPU hours |
|---|---|---|
| `ehpc-aif-2026pg01-905` | training, inference, anything with `--gres` | charged |
| `2cpu-single-host` | conda installs, `huggingface-cli download`, dataset prep, `lfs quota` | **free** |

A CPU-only job under the project QoS is rejected; a GPU job under `2cpu-single-host` is rejected.

### Billing

```
billing/min = threads × 0.035714 + memoryGB × 0.25 + GPUs × 1.0
```

Memory dominates. Keep ≤26 threads and ≤257 GB per GPU (§3). Check what a pilot actually used:

```bash
sacct -j <jobid> --format=JobID,JobName%20,AllocTRES%70,Elapsed,MaxRSS,State
accountcheck ehpc-aif-2026pg01-905     # after `module load accountcheck`
```

---

## 6. Storage

| Path | Size | Use |
|---|---|---|
| `/home/ffromm` | 2 GB quota, **100 k inodes** | dotfiles only |
| `/valhalla/projects/ehpc-aif-2026pg01-905` | 5 TiB, setgid | everything: envs, weights, data, runs |
| `/weka/projects/…` | — | **not allocated to us**; ask if the dataloader needs it |
| `/raid` | 24.5 TB node-local | undocumented for jobs; probe first |

**The home directory will hit the inode limit before the byte limit.** 100 000 inodes is one HF
cache. Only `HF_HOME` and `TMPDIR` are documented — the rest are on you:

```bash
# --- caches.sh — source in EVERY job script and interactive shell ---
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
export WORK=${PROJ}/scratch/${USER}

export HF_HOME=${PROJ}/hf_cache          # [doc]
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TORCH_HOME=${WORK}/torch
export XDG_CACHE_HOME=${WORK}/xdg
export TRITON_CACHE_DIR=${WORK}/triton   # bites the moment anything torch.compiles
export CUDA_CACHE_PATH=${WORK}/nv_compute
export MPLCONFIGDIR=${WORK}/mpl
export PIP_CACHE_DIR=${WORK}/pip_cache
export CONDA_PKGS_DIRS=${PROJ}/conda/pkgs
export TMPDIR=${PROJ}/tmp                # [doc]
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
```

**61-day access-time reaper**: *"Discoverer reserves the right to remove any of the files stored
therein that have not been accessed within the last 61 days"* [doc]. atime-based, and Lustre is
commonly `relatime`, so reads may not refresh it. Keep the authoritative copy off-cluster.

**Never `mv` a tree into the project dir from elsewhere on `/valhalla`** — project IDs live on the
inode and survive a rename within a filesystem, so the data stays charged to the wrong project.
Use `rsync`/`cp`.

### Quota

```bash
srun -p common -N 1 -n 1 --account=ehpc-aif-2026pg01-905 \
     --qos=2cpu-single-host --mem-per-cpu=512 --pty bash
PID=$(lfs project -d /valhalla/projects/ehpc-aif-2026pg01-905 | awk '{print $1}')
lfs quota -p "$PID" /valhalla
```

### Transfer

`rsync` over SSH; `scp`/`sftp` are explicitly discouraged [doc]. `-l` (preserve symlinks) matters
for HF caches, whose `snapshots/` are symlinks into `blobs/`.

```bash
rsync -e ssh -vrtl --progress --partial --append \
  datasets/gr00t-apple-full/ \
  ffromm@login-plus.discoverer.bg:/valhalla/projects/ehpc-aif-2026pg01-905/data/gr00t-apple-full/
```

Our converted dataset is **81 MB / 402 episodes** — a non-issue. The base weights are the big
object and should not be uploaded (§7).

---

## 7. Environment setup

### Compute nodes have outbound Internet [doc]

`kimi-k26-vllm-h200-guide.html` publishes a batch job downloading a **594 GB** checkpoint, and
`huggingface_setup_guide.html` shows a dgx1 transcript pulling `model.safetensors` at
211–321 MB/s. So stage weights in a `2cpu-single-host` job, then set `HF_HUB_OFFLINE=1` for
training so a hub hiccup cannot kill a run.

Only `huggingface.co`, PyPI, conda channels and `gitlab.discoverer.bg` are *proven*. The HF CDN
hosts (`cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`) and `github.com` are untested [?] — a partial
allowlist looks like success right up until the first large file.

### Step 1 — check `.condarc` before the first `conda create`

The docs warn the shipped `~/.condarc` points `pkgs_dirs`/`envs_dirs` at `/tmp` ("320 M and 95 %
full"), producing `NoSpaceLeftError` while `/valhalla` sits empty. **There is currently no
`~/.condarc` on our account** [✓] — so either the trap doesn't apply or the file appears on first
conda use. Check, then set it anyway:

```bash
module load anaconda3/python3.12
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
mkdir -p ${PROJ}/conda/pkgs ${PROJ}/conda/envs ${PROJ}/tmp
conda config --add pkgs_dirs ${PROJ}/conda/pkgs
conda config --add envs_dirs ${PROJ}/conda/envs
conda info | grep -E "(package cache|envs directories)"
```

`conda init` and `conda activate` are forbidden. Activate by PATH export:

```bash
export VIRTUAL_ENV=/valhalla/projects/ehpc-aif-2026pg01-905/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
```

### Step 2 — build the env (CPU-only job, free)

```bash
#!/bin/bash
#SBATCH --partition=common
#SBATCH --job-name=wam_env
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -o wam_env.%j.out

set -euo pipefail
module purge || exit 1
module load anaconda3/python3.12 || exit 1

PROJ=/valhalla/projects/${SLURM_JOB_ACCOUNT}
source ${PROJ}/caches.sh
export VIRTUAL_ENV=${PROJ}/virt_envs/wam

conda create --prefix ${VIRTUAL_ENV} python=3.12 --solver=libmamba -y
export PATH=${VIRTUAL_ENV}/bin:${PATH}

# CONDA_OVERRIDE_CUDA is MANDATORY here: loading the CUDA module is explicitly not enough —
# conda's solver probes for a virtual __cuda package that only materialises with a GPU present,
# and this QoS has none.
CONDA_OVERRIDE_CUDA=12.9 conda install --prefix ${VIRTUAL_ENV} -c conda-forge -y \
    'pytorch=*=cuda12*'

echo "pip: $(which pip)"     # must be inside the prefix, never ~/.local
python -m pip install --no-cache-dir \
    'diffusers>=0.35' transformers accelerate peft safetensors \
    imageio imageio-ffmpeg av einops 'huggingface_hub[cli]' \
    --constraint <(python -c "import torch;print('torch=='+torch.__version__.split('+')[0])")

python -m pip install --no-cache-dir -e ${PROJ}/wam
python -c "import torch,diffusers,peft;print(torch.__version__,torch.version.cuda,diffusers.__version__)"
```

Notes:
- **Python 3.12**, matching both `pyproject.toml` and the `anaconda3/python3.12` module. The
  docs' 3.11 recommendation predates the module now installed.
- The docs pin `pytorch=2.8.0=cuda129_generic_py311_h469a2b5_201` — an exact conda-forge build
  hash that will rot. Resolve it live with `conda search -c conda-forge 'pytorch[build=cuda12*]'`.
- pip must run from inside the prefix, or packages land in `~/.local` and blow the home inode
  limit.

### Step 3 — stage the Wan weights (CPU-only job, free)

```bash
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=2 --mem=32G

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir ${PROJ}/models/Wan2.2-TI2V-5B
```

That repo id is what `configs/model/wan22_ti2v_5b.yaml` already pins and what the verified
ZeroGPU runs used — known-good.

---

## 8. Running T-16

### The 4-hour constraint is the design constraint

A LoRA run over 402 episodes will not finish in one 4-hour job, and the partition is configured
`PreemptMode=REQUEUE` on top of that. So checkpoint-and-resume is not an optimisation, it is the
architecture:

- `--requeue`, `--open-mode=append`, `--signal=B:USR1@300`
- a `SIGUSR1` handler in the training loop that saves state at the next step boundary and exits 0
- a **job-id-independent** `--output_dir` so `--resume_from_checkpoint latest` finds it
- `--checkpoints_total_limit`, saving **LoRA adapter weights only** (hundreds of MB), never full
  pipeline states

`--requeue` and `--signal` are standard Slurm but documented nowhere on this site — prove them
with a 5-minute toy job before building on them.

### The job

```bash
#!/bin/bash
#SBATCH --partition=common
#SBATCH --job-name=wam-t16-lora
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # fair share for 1 GPU
#SBATCH --gres=gpu:1                 # 1 GPU schedules fast; 8 queues behind everything
#SBATCH --mem=192G                   # ≤257G keeps billing under the GPU-hour rate
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t16.%j.out

set -euo pipefail
PROJ=/valhalla/projects/${SLURM_JOB_ACCOUNT}
source ${PROJ}/caches.sh

module purge || exit 1
module load anaconda3/python3.12 || exit 1
module load nvidia/cuda/12/12.8 || exit 1

export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi -L

trap 'echo "USR1 -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID' USR1
python -m wam.training.joint ... &
wait
```

### Order of work

Each step below is a checked-in file in `cluster/discoverer/` — see its README for the table.

0. `sync.sh --data` (on the Mac) — repo + 81 MB dataset + `caches.sh`.
1. **Env + weight staging** — `10_build_env.sbatch`, `20_stage_weights.sbatch --groot`. Two
   `2cpu-single-host` jobs, costs no GPU hours.
2. **Requeue probe** — `25_requeue_probe.sbatch`, also free. Proves `--requeue` + `--signal`
   before step 5 is built on them.
3. **Smoke test** — `30_smoke_wan.sbatch` (~15 min, 1 GPU): load Wan2.2-TI2V-5B through
   `WanI2VAdapter`, confirm the 13/13 checks from the ZeroGPU run reproduce, record peak VRAM.
   Validates the environment, not the model.
4. **Re-run the readout probe** — `40_readout_probe.sbatch` so the blocks-`(2, 10)` pick in
   `configs/model/wan22_ti2v_5b.yaml` is re-measured on this stack.
5. **T-16 LoRA** on the 402 converted GR00T episodes — `50_train_t16.sbatch`, the actual
   deliverable. **Blocked on T-16a** (local code work, no cluster needed): the joint model
   hard-wires `TinyVideoBackbone`, `WanI2VAdapter` has no LoRA injection, and
   `scripts/train_t16_lora.py` does not exist. The job refuses to start without it rather than
   burning GPU hours. Details in `cluster/discoverer/README.md`.
6. **Re-run T-18** (`scripts/run_ablation.py`, identical 362/40 split) against the fine-tuned
   backbone. That is the real AC-07 verdict; the current one ("video hurts") was measured with the
   `tiny` backbone and explicitly attributed to the missing pretrained prior.

If we ever pre-extract latents: Lustre punishes many-small-files dataloaders — pack shards, don't
decode MP4s per step.

---

## 9. Allocation and obligations

`ehpc-aif-2026pg01-905` = EuroHPC / **AI Factories** call / **Playground** access mode [inf — from
the EuroHPC ID mask; the `01` and `905` fields are defined nowhere]. Confirmed for our account:
**5 000 GPU-hours** — which matches the published Playground fixed allocation for Discoverer GPU.

Playground terms:

- **1–3 months**, rolling, decision within 2 working days, no peer review
- **No extensions. No additional resources.** The ToR says so twice.
- **Under-usage is policed too** — monitored monthly; unjustified under-usage reduces future
  allocations.
- IP terms are favourable: generated data and models remain ours; SMEs may exploit them
  commercially.

**Mandatory acknowledgement** — verbatim, EuroHPC greps publications for the exact pattern. Put it
in any paper, model card, README or dataset deposit:

> We acknowledge EuroHPC JU for awarding the project ID EHPC-AIF-2026PG01-905 access to
> Discoverer GPU partition hosted by SofiaTech, Bulgaria.

A **Final Report is due within 3 months** of allocation end, including energy use and carbon
footprint. Slurm records may be purged 6 months after expiry — export early:

```bash
sacct -A ehpc-aif-2026pg01-905 --starttime=2026-01-01 \
  --format=JobID,JobName%30,AllocTRES%60,Elapsed,State,ConsumedEnergy --parsable2 \
  > allocation_jobs.csv
```

Project data is deleted 30 days after project end (`scratchfolder.html`) or 45
(`projectfolder.html`) — plan for 30.

**Support:** `helpdesk@discoverer.bg`, single mailbox, no portal, no SLA. Hard requirement:
*"Do not utilize Gmail or other free email services… Always use the institutional email address
linked to your account… Otherwise, you may not receive a response."* The git identity on this
workstation is a gmail address — check which address Discoverer has on file **before** you need
help.

---

## 10. Open questions for helpdesk

The first six from the original list are answered — walltime (4 h), partition (`common`), node
count (2), CUDA module (`nvidia/cuda/12/12.8`), setgid (yes), budget (5 000 GPU-h). What remains:

1. **What is the allocation end date?** Not visible in `accountcheck`, and it sets both the
   experiment schedule and the 30-day data-export cliff. Blocking for planning.
2. **Can we get a `/weka` folder?** `/weka/projects/` has directories for other projects
   (including sibling AIF project `…-555`) but none for us. For a video dataloader this is the
   difference between a fast NVMe tier and hammering Lustre metadata. Path, capacity, inode quota,
   how to check usage, purge policy?
3. **May jobs write to node-local `/raid`?** What path, is `$SLURM_TMPDIR` set, is it purged
   between jobs, is it shared across the ≤4 oversubscribed jobs on a node?
4. **Is preemption actually active?** The partition shows `OverSubscribe=FORCE:4` and
   `PreemptMode=REQUEUE`. Can our jobs be preempted mid-run, and by which QoS? Determines how
   aggressive the checkpoint interval must be.
5. **What is the inode limit on the 5 TiB quota?**
6. **Apptainer is installed on the login node — does it work on compute nodes?** Is `--fakeroot`
   permitted, and is there a sanctioned route to nvcr.io / docker.io images? (Your docs say
   containers are CPU-partition-only, which the login node contradicts.)
7. **What outbound egress do compute nodes really have?** The HF *API* is proven; the CDN hosts
   that actually serve safetensors (`cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`) and `github.com`
   are not. Is `HF_TOKEN` usable for gated repos?
8. **Are `--requeue` and `--signal=B:USR1@N` supported?** With a 4-hour cap they are load-bearing.
   Also: are `seff`, `sstat`, `srun --overlap` and job arrays available? (Arrays appear in
   `squeue`, so probably yes.)
9. **Is `ConsumedEnergy` populated on the DGX nodes?** The EuroHPC final report mandates energy and
   carbon figures.
10. **Which email address do you have on file**, and what is the SSH key rotation process?

---

## 11. Gotchas, condensed

- **4-hour walltime.** Checkpoint/requeue is mandatory, not optional.
- **Omitting `--qos` lands you on `normal`: 1 minute, 0 GPUs.** Always set account *and* QoS.
- **Two QoS names.** GPU work vs. free `2cpu-single-host`. Wrong one = rejection.
- **Ask for `gpu:1`.** The queue is full of 4/7/8-GPU jobs on a 15-GPU machine.
- **Stay ≤26 threads / ≤257 GB per GPU** or billing strands your GPU-hours permanently.
- **`$HOME` is 2 GB *and* 100 k inodes.** Export every cache var (§6) before the first install.
- **`CONDA_OVERRIDE_CUDA=12.9`** is mandatory on the CPU-only install QoS.
- **Never run conda/pip/compiles/agents on `login-plus`.** Termination is on the table.
- **61-day atime reaper** on `/valhalla`. Keep an off-cluster copy.
- **`mv` within `/valhalla` keeps the old project ID.** Use `rsync`/`cp`.
- **pip can silently replace the conda CUDA torch** with a PyPI wheel. Constrain, then verify
  `torch.version.cuda`.
- **Max 4 concurrent SSH sessions**, and max 4 running jobs.
- **No Gmail to helpdesk.**

---

*Written 2026-07-27 from `docs.discoverer.bg` and the EuroHPC AI-Factory Terms of Reference, then
verified in a live session on `login-plus` the same day. No job has been submitted yet — the
allocation is at 0 % used.*
