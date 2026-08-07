#!/bin/bash
# Step 15 — T-041 / PR-09: stage Cosmos3-Super at the pinned revision and convert it to DCP.
#
#   T041_FREEZE_LIFTED="<reason>" sbatch 91_stage_cosmos_weights.sbatch
#
# 134.6 GB down, plus a DCP copy. Budget ~300 GB under ${PROJ} (5 TiB, 10% used).
# One GPU, because convert_model_to_dcp materialises the model. PR-09 §7 ceiling: 4 GPU-h.
#
# THE REVISION IS PINNED AND HAS NO FLOATING DEFAULT. PR-09 §3 records
# e0262be9d8f7586bc24c069a2aed2b665bdff266, read off the HF API on 2026-08-07. `main` on a model
# repo is not a checkpoint id; a run that cannot name the bytes it trained from is not
# reproducible, and T-39's PR-07 §8 item 5 exists because this was got wrong once already.
#
#SBATCH --job-name=wam-cosmos-stage
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1
#SBATCH --mem=250G                   # keep <=257G for the same billing reason
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/cosmos-stage.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
source "${PROJ}/cosmos_env.sh"

: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export PATH=${UV_INSTALL_DIR}:${FFMPEG_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=""
source "${FRAMEWORK}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi -L

MODEL_ID=${MODEL_ID:-nvidia/Cosmos3-Super}
MODEL_REVISION=${MODEL_REVISION:-e0262be9d8f7586bc24c069a2aed2b665bdff266}
CKPT_ROOT=${PROJ}/checkpoints
HF_DIR=${CKPT_ROOT}/Cosmos3-Super-hf
DCP_DIR=${CKPT_ROOT}/Cosmos3-Super
VAE_DIR=${CKPT_ROOT}/wan22_vae
VAE_PATH=${VAE_DIR}/Wan2.2_VAE.pth
mkdir -p "${CKPT_ROOT}"

# -- 1. weights ------------------------------------------------------------------------------
if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "=== downloading ${MODEL_ID}@${MODEL_REVISION} (134.6 GB)"
  uvx hf@latest download "${MODEL_ID}" --revision "${MODEL_REVISION}" --local-dir "${HF_DIR}"
else
  echo "=== ${HF_DIR} present, skipping download"
fi

# The recipe's own step 2 pulls the Wan2.2 VAE. Same VAE family as T-16's backbone — noted
# because it is a coincidence of NVIDIA's tokenizer choice, not a shared component with WAM.
if [[ ! -f "${VAE_PATH}" ]]; then
  uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "${VAE_DIR}"
fi

# -- 2. prove the bytes ------------------------------------------------------------------------
# The revision goes into the artifact as a fact, so it is read back off the download rather than
# echoed from the variable that requested it.
python - "${HF_DIR}" "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import json, pathlib, sys
d, model_id, want = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
shards = sorted(d.glob("**/*.safetensors"))
total = sum(p.stat().st_size for p in shards)
print(f"{len(shards)} safetensors shards, {total/2**30:.1f} GiB")
if not shards:
    sys.exit("FATAL: no safetensors in the download — the revision or the repo is wrong.")
print("resolved revision (requested):", want)
(d / "STAGED.json").write_text(json.dumps(
    {"model_id": model_id, "revision": want, "shards": len(shards), "bytes": total}, indent=2))
PY

# -- 3. DCP ------------------------------------------------------------------------------------
if [[ ! -d "${DCP_DIR}" ]]; then
  echo "=== converting to DCP (the slow half)"
  cd "${FRAMEWORK}"
  python -m cosmos_framework.scripts.convert_model_to_dcp -o "${DCP_DIR}" --checkpoint-path "${HF_DIR}"
else
  echo "=== ${DCP_DIR} present, skipping conversion"
fi
[[ -d "${DCP_DIR}" ]] || { echo "FATAL: DCP conversion produced nothing."; exit 1; }

cat >> "${PROJ}/cosmos_env.sh" <<EOF
export MODEL_ID=${MODEL_ID}
export MODEL_REVISION=${MODEL_REVISION}
export BASE_CHECKPOINT_PATH=${DCP_DIR}
export WAN_VAE_PATH=${VAE_PATH}
EOF
du -sh "${HF_DIR}" "${DCP_DIR}" "${VAE_DIR}"
echo "STAGED OK"
echo "next: sbatch 92_fetch_g1_corpus.sbatch"
