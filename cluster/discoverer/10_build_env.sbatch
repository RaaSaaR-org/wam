#!/bin/bash
# Step 1 — build the conda prefix env. CPU-only QoS: costs NO GPU hours.
#   sbatch 10_build_env.sbatch
#SBATCH --job-name=wam-env
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/env.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12

export VIRTUAL_ENV=${PROJ}/virt_envs/wam

# conda init / conda activate are forbidden here — activate by PATH export only.
conda config --add pkgs_dirs "${PROJ}/conda/pkgs" || true
conda config --add envs_dirs "${PROJ}/conda/envs" || true
conda info | grep -E "(package cache|envs directories)"

conda create --prefix "${VIRTUAL_ENV}" python=3.12 --solver=libmamba -y
export PATH=${VIRTUAL_ENV}/bin:${PATH}

# CONDA_OVERRIDE_CUDA is mandatory: this QoS has no GPU, so conda's __cuda virtual
# package never materialises and the solver silently picks the CPU build.
CONDA_OVERRIDE_CUDA=12.9 conda install --prefix "${VIRTUAL_ENV}" -c conda-forge -y \
    'pytorch=*=cuda12*'

# pip must resolve inside the prefix — otherwise packages land in ~/.local and blow
# the 100k inode limit on /home.
which pip | grep -q "${VIRTUAL_ENV}" || { echo "FATAL: pip outside prefix"; exit 1; }

# Pin the conda torch so pip cannot silently swap it for a PyPI wheel.
python -c "import torch;print('torch=='+torch.__version__.split('+')[0])" \
    > "${WORK}/torch-constraint.txt"

python -m pip install --no-cache-dir --constraint "${WORK}/torch-constraint.txt" \
    'diffusers>=0.35' 'transformers>=4.51' accelerate peft safetensors \
    sentencepiece protobuf ftfy einops \
    pyarrow opencv-python-headless imageio imageio-ffmpeg av \
    'huggingface_hub[cli]' pydantic pyyaml numpy

python -m pip install --no-cache-dir --no-deps -e "${PROJ}/wam"

python - <<'PY'
import torch, diffusers, peft, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__, "peft", peft.__version__,
      "transformers", transformers.__version__)
import wam; print("wam ok")
PY
echo "ENV OK -> ${VIRTUAL_ENV}"
