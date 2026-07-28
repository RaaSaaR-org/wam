# Cache + scratch redirection for Discoverer+. Sourced by every job.
# Installed target on the cluster: /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
#
# Why: /home is 2 GiB with a 100k inode limit. Every tool below writes to $HOME by
# default, and a single HF model or conda solve blows that quota. Redirect once, here.

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
export PROJ
export WORK=${PROJ}/scratch/${USER}

export HF_HOME=${PROJ}/hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TORCH_HOME=${WORK}/torch
export XDG_CACHE_HOME=${WORK}/xdg
export TRITON_CACHE_DIR=${WORK}/triton
export CUDA_CACHE_PATH=${WORK}/nv_compute
export MPLCONFIGDIR=${WORK}/mpl
export PIP_CACHE_DIR=${WORK}/pip_cache
export CONDA_PKGS_DIRS=${PROJ}/conda/pkgs
export TMPDIR=${PROJ}/tmp

mkdir -p "${WORK}" "${HF_HUB_CACHE}" "${TMPDIR}" "${PROJ}/logs" \
         "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" \
         "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${PIP_CACHE_DIR}" \
         "${CONDA_PKGS_DIRS}" "${PROJ}/conda/envs"

# Scratch under $PROJ is reaped after 61 days of no access — checkpoints belong in
# ${PROJ}/runs, not ${WORK}.
