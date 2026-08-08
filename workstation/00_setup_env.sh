#!/bin/bash
# Step 0 of the workstation pipeline — build the cosmos-framework environment.
#
#   T041_FREEZE_LIFTED="<reason>" workstation/00_setup_env.sh
#
# THE SAME PINNED COMMITS AS THE CLUSTER. cluster/discoverer/90_build_cosmos_env.sbatch pins both
# repos by SHA rather than by branch, because main moves and a run that says it "used the recipe"
# has to name the commit that was the recipe. The workstation pipeline produces the captions that a
# cluster job then trains on, so if the two environments drift, the corpus was built by one version
# of the captioner and consumed by another — and nothing in the artifacts would say so.
#
# CUDA GROUP. The cluster pins cu128-train because its newest module is CUDA 12.8. A workstation is
# free to be newer, and a Blackwell card (RTX 5090, sm_120) needs at least cu128 — earlier wheels
# have no kernels for it and fail at the first matmul, not at import. Override UV_GROUP if the
# driver is newer; do not go below cu128.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_freeze_lifted
need git ""
need ffmpeg "the captioner and the video loader both need it"

COSMOS_SHA=${COSMOS_SHA:-f76cd8705dc04e5d6fba0ce0c057930b4393ad5d}
FRAMEWORK_SHA=${FRAMEWORK_SHA:-12a9a81e01b6de687139b7dba6e6928ea4994a82}
UV_GROUP=${UV_GROUP:-cu128-train}

COSMOS=${COSMOS:-${WORK}/third_party/cosmos}
# The cookbook README says to clone the framework to packages/cosmos3 and run torchrun from there;
# the framework's own setup.md says to clone it standalone. Both are satisfied by putting it where
# the cookbook expects it.
FRAMEWORK=${FRAMEWORK:-${COSMOS}/packages/cosmos3}
mkdir -p "${WORK}/third_party"

if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=${HOME}/.local/bin:${PATH}
fi
uv --version

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

cd "${FRAMEWORK}"
uv sync --all-extras --group="${UV_GROUP}"

# LD_LIBRARY_PATH must be empty for the framework's own CUDA libs to win over the host's — the
# setup guide and launch_sft_vision_super.sh both do this, and it is not optional.
export LD_LIBRARY_PATH=""
source "${FRAMEWORK}/.venv/bin/activate"
python - <<'PY'
import importlib, pathlib
import cosmos_framework
print("cosmos_framework at", pathlib.Path(cosmos_framework.__file__).resolve().parents[1])
for mod in ("cosmos_framework.scripts.caption_from_video",
            "cosmos_framework.scripts.captions_to_sft_jsonl"):
    importlib.import_module(mod)
    print("ok", mod)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu{i} {p.name} sm_{p.major}{p.minor} {p.total_memory // 2**30} GiB")
else:
    # Not fatal here — preparation is CPU work. It IS fatal at the captioning step, and saying so
    # now is cheaper than discovering it after a 40-minute transcode.
    print("WARNING: no CUDA device visible. Step 30 (captioning) needs one.")
PY

# cv2 is what actually reads the clips, and its build — not ffmpeg's — decides whether a codec
# works. Recording the version here makes "which decoder did we verify against" answerable later.
python -c "import cv2; print('cv2', cv2.__version__)" 2>/dev/null || echo "note: cv2 not in this env"

cat > "${ENV_FILE}" <<EOF
# Written by workstation/00_setup_env.sh on $(date -Is). Sourced by 20/30.
export WORK=${WORK}
export COSMOS=${COSMOS}
export FRAMEWORK=${FRAMEWORK}
export RECIPE_DIR=${COSMOS}/cookbooks/cosmos3/generator/audiovisual/finetune
export COSMOS_SHA=$(git -C "${COSMOS}" rev-parse HEAD)
export FRAMEWORK_SHA=$(git -C "${FRAMEWORK}" rev-parse HEAD)
export UV_GROUP=${UV_GROUP}
EOF
echo
echo "ENV OK -> ${FRAMEWORK}/.venv   (recorded in ${ENV_FILE})"
echo "next: workstation/10_fetch_corpus.sh"
