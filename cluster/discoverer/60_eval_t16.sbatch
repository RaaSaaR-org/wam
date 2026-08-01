#!/bin/bash
# Step 7 — score the T-16 checkpoint on the PROVEN holdout. THE verdict.
#
# 50_train_t16.sbatch produces weights and a loss curve, and neither answers the only
# question that matters: does the fine-tune beat repeating the last action? That bar exists
# because the action-only baseline (T-13) LOST to causal repeat-last by 17 %, so a falling
# training loss is not evidence of skill. This job turns the checkpoint into
# predictions.jsonl + bench.* and prints skill_vs_repeat_pct.
#
#   sbatch 60_eval_t16.sbatch                     # scores runs/t16-lora-seed0 @ latest
#   RUN_ID=t16-lora-seed1 sbatch 60_eval_t16.sbatch
#   CKPT=step-010696 sbatch 60_eval_t16.sbatch    # an earlier checkpoint
#
# One GPU, minutes: 40 holdout episodes is ~960 chunks and each is ONE forward pass at the
# clean flow timestep (no denoising loop, no video generated). The 4 h cap is startup slack.
#
# No --requeue on purpose. The job is idempotent but writes no DONE marker, so a requeue has
# no way to tell "preempted" from "failed" — and re-scoring from scratch costs minutes, which
# is cheaper than any loop-safety rail would be to maintain.
#
#SBATCH --job-name=wam-eval-t16
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/eval.%j.out

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
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

EVAL=${PROJ}/wam/scripts/eval_t16.py
if [[ ! -f "${EVAL}" ]]; then
  echo "FATAL: ${EVAL} missing — run cluster/discoverer/sync.sh from the Mac first."
  exit 1
fi

RUN_ID=${RUN_ID:-t16-lora-seed0}
CKPT=${CKPT:-latest}
OUT=${PROJ}/runs/${RUN_ID}

if [[ ! -d "${OUT}" ]]; then
  echo "FATAL: ${OUT} does not exist — nothing to score."
  exit 1
fi
# Not fatal: scoring an intermediate checkpoint mid-run is legitimate. But the artifacts would
# then describe a partially trained model, so say so loudly rather than let the numbers pass
# for a finished run's.
if [[ ! -f "${OUT}/DONE" ]]; then
  echo "WARNING: no ${OUT}/DONE — this run has not finished its step budget."
  echo "         The numbers below describe whatever ${CKPT} happens to be."
fi

# --exclude-episodes in 50_train_t16.sbatch and --holdout here MUST be the same file: the
# evaluator recomputes the dataset hash over (dataset - holdout) and refuses to score unless
# it matches the hash the checkpoint recorded. Pointing these at two different files is
# exactly the drift that guard exists to catch.
#
# --train-episodes is passed UNCONDITIONALLY, and i8_train_362.txt is the complement of the
# holdout above (402 - 40 = 362, set-equal — configs/splits/README). Why it is here at all:
# train_t16_lora.py records train_episode_ids on EVERY run, so any checkpoint retrained after
# 2026-08-01 takes eval_t16's disjointness path, which needs an external witness. The archived
# runs/t16-lora-seed0/checkpoints/step-020000 predates the field and takes the complement path,
# where this file is checked as a redundant cross-check instead. One command line, both proofs.
python "${EVAL}" \
    --run-dir "${OUT}" \
    --checkpoint "${CKPT}" \
    --dataset "${PROJ}/data/gr00t-apple-full" \
    --holdout "${PROJ}/wam/configs/splits/t18_holdout_episodes.txt" \
    --train-episodes "${PROJ}/wam/configs/splits/i8_train_362.txt" \
    --camera ego \
    --device cuda \
    --out "${OUT}/eval-${CKPT}"

echo "=== artifacts"
ls -la "${OUT}/eval-${CKPT}"
echo "=== bench.md"
cat "${OUT}/eval-${CKPT}/bench.md"
