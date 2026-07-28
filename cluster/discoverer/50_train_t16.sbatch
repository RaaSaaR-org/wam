#!/bin/bash
# Step 6 — T-16 LoRA fine-tune. THE deliverable.
#
# scripts/train_t16_lora.py (T-16a: Wan backbone inside JointWorldActionModel + LoRA
#   injection) exists and satisfies the contract below; the Slurm side is exercised
#   independently by 25_requeue_probe.sbatch.
#
#   Contract this job expects from that entry point:
#     - SIGUSR1 -> checkpoint at the next step boundary, then exit 0;
#     - write ${OUT}/DONE once the configured step budget is finished — that file is the
#       ONLY signal distinguishing "finished" from "ran out of wall clock" (both exit 0);
#     - --resume latest picks the newest checkpoint in --out-dir.
#
#   sbatch 50_train_t16.sbatch
#
#SBATCH --job-name=wam-t16-lora
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1                 # 1 GPU schedules fast; 8 queues behind everything
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t16.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi -L

TRAIN=${PROJ}/wam/scripts/train_t16_lora.py
if [[ ! -f "${TRAIN}" ]]; then
  echo "FATAL: ${TRAIN} missing — T-16a (Wan backbone in JointWorldActionModel + LoRA)"
  echo "       is not implemented yet. Not burning GPU hours on a job that cannot run."
  exit 1
fi

# Run id is job-id-INDEPENDENT on purpose: a requeued job gets a new SLURM_JOB_ID but must
# find the same checkpoint dir. Bump the suffix by hand for a genuinely new experiment.
RUN_ID=t16-lora-seed0
OUT=${PROJ}/runs/${RUN_ID}
mkdir -p "${OUT}"

# The trap only fires while the shell is at `wait` — never run python in the foreground here.
python "${TRAIN}" \
    --backbone-config "${PROJ}/wam/configs/model/wan22_ti2v_5b.yaml" \
    --backbone-source "${PROJ}/models/Wan2.2-TI2V-5B" \
    --training-config "${PROJ}/wam/configs/training/joint_wan_gr00t.yaml" \
    --dataset "${PROJ}/data/gr00t-apple-full" \
    --camera ego \
    --out-dir "${OUT}" \
    --resume latest \
    --checkpoint-every-min 30 \
    --checkpoints-total-limit 3 \
    --save-adapter-only \
    --device cuda &
PY=$!

# --signal=B: delivers USR1 to THIS shell, not to python. Forward it and let the trainer
# stop at a step boundary: requeueing from inside the trap (what this used to do) tears the
# process down mid-step, so every requeue threw away the work since the last checkpoint —
# up to the full 30-minute checkpoint interval, on every single 4-hour wrap-around.
GOT_USR1=0
trap 'GOT_USR1=1; echo "USR1 at $(date -Is) -> forward to pid ${PY}, waiting for checkpoint";
      kill -USR1 "${PY}" 2>/dev/null || true' USR1

# A trap makes `wait` return early (128+signal) with the child still alive, and `set -e`
# would abort the script on that status — hence `|| true` and the loop until the pid is gone.
RC=0
while kill -0 "${PY}" 2>/dev/null; do
  wait "${PY}" && RC=0 || RC=$?
done
echo "trainer pid ${PY} exited ${RC} at $(date -Is) (usr1=${GOT_USR1})"

if [[ -f "${OUT}/DONE" ]]; then
  echo "T-16 COMPLETE — ${OUT}/DONE present, not requeueing"
  exit 0
fi
if [[ "${RC}" -ne 0 && "${GOT_USR1}" -eq 0 ]]; then
  # Died on its own (config error, OOM, ...). Requeueing that only rebuilds the same crash
  # every time the scheduler finds a free H200.
  echo "FATAL: trainer failed with ${RC} and was never signalled — refusing to requeue"
  exit "${RC}"
fi
echo "no ${OUT}/DONE -> requeue ${SLURM_JOB_ID} to continue from the last checkpoint"
scontrol requeue "${SLURM_JOB_ID}"
