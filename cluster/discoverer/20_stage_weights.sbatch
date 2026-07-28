#!/bin/bash
# Step 2 — pull Wan2.2-TI2V-5B (and optionally the raw GR00T LeRobot snapshot) onto
# project storage. CPU-only QoS: costs NO GPU hours. Compute nodes have outbound Internet.
#
#   sbatch 20_stage_weights.sbatch            # Wan weights only (~20 GB)
#   sbatch 20_stage_weights.sbatch --groot    # + first 16 GR00T episodes (for step 40)
#
#SBATCH --job-name=wam-stage
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/stage.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}

# `hf` replaced `huggingface-cli` in huggingface_hub>=0.34; fall back so this job does not
# hinge on which one the env resolved to.
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi
echo "using ${HF}"

# Repo id is what configs/model/wan22_ti2v_5b.yaml pins and what the verified ZeroGPU
# runs used — known-good, do not "upgrade" it casually.
${HF} download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir "${PROJ}/models/Wan2.2-TI2V-5B"

du -sh "${PROJ}/models/Wan2.2-TI2V-5B"

if [[ "${1:-}" == "--groot" ]]; then
  # Raw LeRobot v2.1 snapshot — only scripts/hf_job_wan_probe.py needs this (it reads
  # data/ + videos/ directly). The converted WAM episodes come over via sync.sh --data.
  ${HF} download nvidia/GR00T-N1.7-AppleToPlate --repo-type dataset \
      --local-dir "${PROJ}/data/groot-apple-raw" \
      --include 'meta/*' \
                'data/chunk-000/episode_00000*.parquet' \
                'data/chunk-000/episode_0000?.parquet' \
                'videos/chunk-000/*/episode_0000?.mp4'
  du -sh "${PROJ}/data/groot-apple-raw"
fi

echo "STAGING OK"
