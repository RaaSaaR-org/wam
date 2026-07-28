#!/bin/bash
# Step 5 — re-measure the readout blocks on this stack (~1 h, 1 GPU).
# configs/model/wan22_ti2v_5b.yaml pins feature_blocks [2, 10] from the ZeroGPU ridge
# probes. Confirm the same pick here before spending GPU hours on T-16 with it.
#
# Needs the raw LeRobot snapshot:  sbatch 20_stage_weights.sbatch --groot
#
#SBATCH --job-name=wam-probe
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/probe.%j.out

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

OUT=${PROJ}/runs/wan_probe/${SLURM_JOB_ID}
mkdir -p "${OUT}"

python "${PROJ}/wam/scripts/hf_job_wan_probe.py" \
    --source "${PROJ}/models/Wan2.2-TI2V-5B" \
    --device cuda --dtype bfloat16 \
    --data-dir "${PROJ}/data/groot-apple-raw" \
    --episodes 12 --windows-per-episode 8 --frames 5 \
    --height 192 --width 256 --chunk-steps 16 \
    --measured-blocks 2,10 --heuristic-blocks 15,22 \
    --out "${OUT}/wan_probe_report.json"

cat "${OUT}/wan_probe_report.json"
