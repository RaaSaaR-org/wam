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

# Before any `need`, because on a machine without root this is where ffmpeg, git-lfs and cc live —
# checking for them first and extending PATH afterwards would fail on the very tools we installed.
PATH=${HOME}/.local/bin:${PATH}
export PATH

need git ""
# cosmos-framework LFS-tracks assets/** and every media extension. Without git-lfs on PATH the
# clone still succeeds and then the CHECKOUT dies half-way, leaving a repo whose working tree is
# empty but whose .git is fine — `git log` works, the files are gone, and the error git prints is
# "filter-process: git-lfs: not found" followed by a remote-hangup message that reads like a network
# fault. Fail here instead, where the cause is named.
need git-lfs "cosmos-framework LFS-tracks its assets; the checkout half-fails without it"
need ffmpeg "the captioner and the video loader both need it"
# `uv sync --all-extras` reaches lerobot -> pynput -> evdev, which ships no wheel and so must be
# compiled. The failure arrives ~10 minutes into a multi-GB resolve, as "command 'cc' failed: No
# such file or directory" buried in a build log. Nothing we run needs evdev — it is a keyboard-
# teleop dependency — but --all-extras does not let us decline it, so a compiler is a prerequisite.
need cc "uv sync builds evdev from source; conda create -n build -c conda-forge gcc gxx"

# The framework LFS-tracks demo media we never open; our clips come from our own corpus. Skipping
# the smudge filter keeps checkout honest without fetching the blobs.
export GIT_LFS_SKIP_SMUDGE=${GIT_LFS_SKIP_SMUDGE:-1}

COSMOS_SHA=${COSMOS_SHA:-f76cd8705dc04e5d6fba0ce0c057930b4393ad5d}
FRAMEWORK_SHA=${FRAMEWORK_SHA:-12a9a81e01b6de687139b7dba6e6928ea4994a82}
UV_GROUP=${UV_GROUP:-cu128-train}

COSMOS=${COSMOS:-${WORK}/third_party/cosmos}
# The cookbook README says to clone the framework to packages/cosmos3 and run torchrun from there;
# the framework's own setup.md says to clone it standalone. Both are satisfied by putting it where
# the cookbook expects it.
FRAMEWORK=${FRAMEWORK:-${COSMOS}/packages/cosmos3}
mkdir -p "${WORK}/third_party"

# uv is a prerequisite, not something this script installs. It used to pipe astral.sh's installer
# into a shell, which is the one unpinned, unreviewed thing in a pipeline whose entire point is that
# every component is named by SHA — and it silently changed what `uv sync` would resolve. Install it
# yourself (`pipx install uv`, your package manager, or the vendor installer if you have read it),
# then re-run.
need uv "install it first: pipx install uv"
uv --version

clone_at() {  # repo_url dest sha
  local url=$1 dest=$2 sha=$3
  if [[ ! -d "${dest}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --depth=1 origin "${sha}" || git -C "${dest}" fetch origin
  git -C "${dest}" checkout --force --detach "${sha}"
  # A half-applied checkout leaves HEAD correct and the files missing, so the SHA alone proves
  # nothing. A clean porcelain does: it means every tracked path is present as committed.
  local dirty
  dirty=$(git -C "${dest}" status --porcelain | wc -l)
  [[ ${dirty} -eq 0 ]] || {
    echo "FATAL: ${dest} checked out ${sha} but ${dirty} tracked paths do not match it."
    git -C "${dest}" status --porcelain | head -5
    exit 1
  }
  echo "=== $(basename "${dest}") @ $(git -C "${dest}" rev-parse HEAD) (tree clean)"
}
clone_at https://github.com/NVIDIA/cosmos.git "${COSMOS}" "${COSMOS_SHA}"
clone_at https://github.com/NVIDIA/cosmos-framework.git "${FRAMEWORK}" "${FRAMEWORK_SHA}"

# Same check 90_build_cosmos_env.sbatch makes, for the same reason: ENV_FILE below exports
# RECIPE_DIR and steps 20/30 trust it. If the pinned commit moved these, fail here.
RECIPE_DIR=${COSMOS}/cookbooks/cosmos3/generator/audiovisual/finetune
for f in "${RECIPE_DIR}/launch_sft_vision_super.sh" \
         "${RECIPE_DIR}/toml/sft_config/vision_sft_super.toml"; do
  [[ -f "${f}" ]] || { echo "FATAL: ${f} missing at the pinned commit."; exit 1; }
done

cd "${FRAMEWORK}"
uv sync --all-extras --group="${UV_GROUP}"

# -- transformer_engine cudart shim ----------------------------------------------------------------
# ONLY BITES ON A MACHINE WITH NO SYSTEM CUDA TOOLKIT, which is why the cluster never hit it and this
# workstation did. transformer_engine probes for a system toolkit by trying to load nvrtc and curand;
# if BOTH are missing it concludes there is no CTK and re-loads cublas, cudart and cudnn from the pip
# wheels with strict=True. Discoverer+ has a CUDA 12.8 module, so it never takes that branch. A
# workstation with only a driver has no /usr/local/cuda at all, so it always does.
#
# And that path is broken for CUDA 12: it globs `nvidia/cuda_<name>/lib`, so for "cudart" it looks in
# `nvidia/cuda_cudart` — the CUDA 13 wheel layout. The cu12 wheel is `nvidia-cuda-runtime-cu12` and
# installs to `nvidia/cuda_runtime`. cublas and cudnn resolve because their directories happen to be
# named after the library; only cudart differs, and it fails as "cudart shared object not found"
# eleven frames deep in a megatron import, which reads like a broken install rather than a name.
#
# The alias points TE's expected name at the wheel that is actually installed. Nothing is copied and
# no library is shadowed — TE dlopens the same libcudart.so.12 torch already uses.
SITE_NVIDIA=$("${FRAMEWORK}/.venv/bin/python" -c \
  'import os, sysconfig; print(os.path.join(sysconfig.get_path("purelib"), "nvidia"))')
if [[ -d "${SITE_NVIDIA}/cuda_runtime" && ! -e "${SITE_NVIDIA}/cuda_cudart" ]]; then
  ln -sfn cuda_runtime "${SITE_NVIDIA}/cuda_cudart"
  echo "=== aliased nvidia/cuda_cudart -> cuda_runtime (no system CUDA toolkit on this host)"
fi

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
