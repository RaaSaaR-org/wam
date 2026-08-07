#!/bin/bash
# Step 14 — T-041 / PR-09: build the cosmos-framework env. Its OWN venv, never virt_envs/wam.
#
#   T041_FREEZE_LIFTED="<reason>" sbatch 90_build_cosmos_env.sbatch
#
# Default QoS is 2cpu-single-host: costs NO GPU hours, but it is 2 CPUs and 3 h, and a large
# torch stack can miss that window. If it times out, resubmit on the GPU QoS instead of trimming
# the install:
#   sbatch --qos=ehpc-aif-2026pg01-905 --gres=gpu:1 --cpus-per-task=26 --mem=192G \
#          --time=04:00:00 90_build_cosmos_env.sbatch
# (CLI flags override the #SBATCH directives below. That path DOES bill GPU hours — PR-09 §7
# budgets zero for this step, so record it if you take it.)
#
# WHY A SEPARATE ENV. cosmos-framework pins its own torch, its own CUDA group and its own
# attention kernels. Installing it into virt_envs/wam is how one dependency resolution quietly
# changes every WAM number measured afterwards — the same reasoning as 70_train_t39_baseline's
# virt_envs/t39. Two separate rules, one reason.
#
# CUDA GROUP: cu128-train, NOT the recommended cu130-train. This cluster's newest CUDA module is
# nvidia/cuda/12/12.8 (docs/discoverer.md §3) and the framework's setup guide says to pin
# --group=cu128-train on a CUDA 12.x driver. Taking the "recommended" line off the README would
# install a CUDA 13 build against a 12.8 driver.
#
#SBATCH --job-name=wam-cosmos-env
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/cosmos-env.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

# -- the freeze ---------------------------------------------------------------------------------
# PR-07 §7: "Frozen until T-39 reports: T-32 (§2), any Cosmos3-Super generation, any Cosmos3-Edge
# work." This is a speed bump with a name attached, not a lock: state the reason and it runs, and
# the reason is written into the artifact so the record says who decided and why.
: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason (e.g. \"T-39 reported P on 2026-08-xx\" or \"user lifted 2026-08-xx, recorded as OD-NN\").}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12

# Pinned 2026-08-07 from the GitHub API, not from a branch name. main moves; a run that says it
# used "the recipe" has to name the commit that was the recipe.
COSMOS_SHA=${COSMOS_SHA:-f76cd8705dc04e5d6fba0ce0c057930b4393ad5d}
FRAMEWORK_SHA=${FRAMEWORK_SHA:-12a9a81e01b6de687139b7dba6e6928ea4994a82}
UV_GROUP=${UV_GROUP:-cu128-train}

COSMOS=${PROJ}/third_party/cosmos
# The cookbook README says to clone the framework to packages/cosmos3 and run torchrun from
# there; the framework's own setup.md says to clone it standalone. Both are satisfied by putting
# it where the cookbook expects it.
FRAMEWORK=${COSMOS}/packages/cosmos3
export UV_CACHE_DIR=${WORK}/uv_cache
export UV_PYTHON_INSTALL_DIR=${PROJ}/uv_python
mkdir -p "${PROJ}/third_party" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

# -- uv ------------------------------------------------------------------------------------------
export UV_INSTALL_DIR=${PROJ}/tools/uv
export PATH=${UV_INSTALL_DIR}:${PATH}
if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv into ${UV_INSTALL_DIR} (compute nodes have Internet; login does not)"
  mkdir -p "${UV_INSTALL_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
fi
uv --version

# -- sources ---------------------------------------------------------------------------------------
clone_at() {  # repo_url dest sha
  local url=$1 dest=$2 sha=$3
  if [[ ! -d "${dest}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --depth=1 origin "${sha}" || git -C "${dest}" fetch origin
  git -C "${dest}" checkout --detach "${sha}"
  echo "=== $(basename "${dest}") @ $(git -C "${dest}" rev-parse HEAD)"
}
clone_at https://github.com/NVIDIA/cosmos.git "${COSMOS}" "${COSMOS_SHA}"
clone_at https://github.com/NVIDIA/cosmos-framework.git "${FRAMEWORK}" "${FRAMEWORK_SHA}"

# The recipe files this whole chain depends on. If the pinned commit moved them, fail here rather
# than three jobs and 100 GPU-hours later.
RECIPE_DIR=${COSMOS}/cookbooks/cosmos3/generator/audiovisual/finetune
for f in "${RECIPE_DIR}/launch_sft_vision_super.sh" \
         "${RECIPE_DIR}/toml/sft_config/vision_sft_super.toml"; do
  [[ -f "${f}" ]] || { echo "FATAL: ${f} missing at the pinned commit."; exit 1; }
done

# -- install ---------------------------------------------------------------------------------------
# ffmpeg: docs/discoverer.md §3 records that this cluster has no ffmpeg module and no root, so
# apt-get from the framework's setup guide is not available. It comes from conda, and the
# captioner and the video loader both need it.
export FFMPEG_PREFIX=${PROJ}/tools/ffmpeg
if [[ ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
  conda create --prefix "${FFMPEG_PREFIX}" -c conda-forge -y --solver=libmamba ffmpeg git-lfs
fi
export PATH=${FFMPEG_PREFIX}/bin:${PATH}
ffmpeg -version | head -1

cd "${FRAMEWORK}"
uv sync --all-extras --group="${UV_GROUP}"

# -- verify ------------------------------------------------------------------------------------------
# LD_LIBRARY_PATH must be empty for the framework's own CUDA libs to win over the host's — the
# setup guide and launch_sft_vision_super.sh both do this, and it is not optional.
export LD_LIBRARY_PATH=""
source "${FRAMEWORK}/.venv/bin/activate"
python - <<'PY'
import importlib, pathlib
import cosmos_framework
root = pathlib.Path(cosmos_framework.__file__).resolve().parents[1]
print("cosmos_framework at", root)
for mod in ("cosmos_framework.scripts.train",
            "cosmos_framework.scripts.convert_model_to_dcp",
            "cosmos_framework.scripts.caption_from_video",
            "cosmos_framework.scripts.captions_to_sft_jsonl",
            "cosmos_framework.scripts.export_model"):
    importlib.import_module(mod)
    print("ok", mod)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

cat > "${PROJ}/cosmos_env.sh" <<EOF
# Written by 90_build_cosmos_env.sbatch on $(date -Is). Sourced by 91/92/93.
export COSMOS=${COSMOS}
export FRAMEWORK=${FRAMEWORK}
export RECIPE_DIR=${RECIPE_DIR}
export COSMOS_SHA=$(git -C "${COSMOS}" rev-parse HEAD)
export FRAMEWORK_SHA=$(git -C "${FRAMEWORK}" rev-parse HEAD)
export UV_GROUP=${UV_GROUP}
export UV_CACHE_DIR=${UV_CACHE_DIR}
export UV_INSTALL_DIR=${UV_INSTALL_DIR}
export FFMPEG_PREFIX=${FFMPEG_PREFIX}
EOF
echo "COSMOS ENV OK -> ${FRAMEWORK}/.venv   (recorded in ${PROJ}/cosmos_env.sh)"
echo "next: sbatch 91_stage_cosmos_weights.sbatch"
