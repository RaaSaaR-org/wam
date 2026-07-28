#!/bin/bash
# Step 4 — Wan backbone smoke test on one H200 (~15 min, ~0.25 GPU-h).
# Validates the environment, not the model: does WanI2VAdapter reproduce the 13/13 checks
# from the verified ZeroGPU run on this stack? Also records peak VRAM for sizing T-16.
#
#   sbatch 30_smoke_wan.sbatch
#
#SBATCH --job-name=wam-smoke
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/smoke.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are staged; no hub at runtime

nvidia-smi -L

OUT=${PROJ}/runs/smoke/${SLURM_JOB_ID}
mkdir -p "${OUT}"

# --camera ego: our converted episodes carry a single 160x120 "ego" stream; the script
# resizes to the requested probe geometry (multiples of 32).
python "${PROJ}/wam/scripts/hf_job_wan_smoke.py" \
    --source "${PROJ}/models/Wan2.2-TI2V-5B" \
    --device cuda --dtype bfloat16 \
    --episode "${PROJ}/data/gr00t-apple-full/gr00t-apple-000000" \
    --camera ego --frames 5 --height 256 --width 448 \
    --action-steps 16 --target-dim 7 \
    --out "${OUT}/wan_smoke_report.json"

echo "--- report ---"
cat "${OUT}/wan_smoke_report.json"
