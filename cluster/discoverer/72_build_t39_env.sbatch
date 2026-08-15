#!/bin/bash
# Step 15 — T-39 / PR-07 §8 item 4: vendor Isaac-GR00T and build its OWN venv.
#
#   GR00T_COMMIT=<sha> sbatch 72_build_t39_env.sbatch
#
# Free on the 2cpu-single-host QoS: no GPU hours. PR-07 §7 budgets 12 GPU-h for T-39 and this
# step must not eat any of it. If the install misses the 4 h window, resubmit on the GPU QoS
# rather than trimming the dependency set — and record that you did, because it bills.
#
# WHY A SEPARATE VENV. 70_train_t39_baseline.sbatch exits FATAL if ${PROJ}/virt_envs/t39 is
# absent, on purpose: the vendored trainer pins torch 2.7.1 / flash-attn 2.7.4 / deepspeed
# 0.17.6, and resolving those into virt_envs/wam would quietly change the dependency set every
# WAM number in docs/benchmark.md was produced under. Same rule, same reason, as the cosmos env.
#
# WHY A PINNED SHA WITH NO DEFAULT. PR-07 §8 items 5 and 6 exist because "the recipe" is not a
# thing you can cite — a branch name moves. The same argument applies to the trainer tree itself,
# so GR00T_COMMIT is required here exactly as MODEL_ID is required in 70_*. A run that cannot
# name the commit it trained under is not a positive control, it is an anecdote.
#
# THE ONE SHA WE CAN JUSTIFY TODAY, and its problem, stated rather than buried:
#
#   e5749287857afd97b78f1147166137de29746392   (2026-06-18)
#
# is the tree that DEMONSTRABLY trained ${PROJ}/runs/apple_pnp_h200_frozen to checkpoint-5000 on
# our own 402-episode apple corpus in July 2026. Established 2026-08-15 by diffing the unpacked
# bundle at ${PROJ}/apple_pnp_h200/Isaac-GR00T against 50 upstream commits and taking the
# minimum: exactly ONE file differs (see the patch below), and pyproject.toml + uv.lock are
# byte-identical to it.
#
# The problem: e5749287 PREDATES upstream's own "GR00T N1.7 General Release" commit
# (1a1837f20538b7d7e21f977a11a5aee14f99803c, 2026-07-06), while the staged weights were pulled at
# HuggingFace commit 2fc962b973bccdd5d8ce4f67cc63b264d6886495 on 2026-07-06 — i.e. pre-release
# code against release weights. It ran, and it trained, so the mismatch is not fatal; but "it ran"
# is not "this is the recipe NVIDIA documents for N1.7", and PR-07 §8 item 6 is precisely the
# item that forbids conflating those two. Resolve it before the T-39 run of record, not after.
#
#SBATCH --job-name=wam-t39-env
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t39-env.%j.out

set -euo pipefail
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
source "${PROJ}/caches.sh"

: "${GR00T_COMMIT:?set GR00T_COMMIT to the upstream NVIDIA/Isaac-GR00T commit to vendor. No default — see the header.}"

module purge
module load anaconda3/python3.12

VENDOR=${PROJ}/wam/third_party/isaac-gr00t
VENV=${PROJ}/virt_envs/t39
PATCH=${PROJ}/wam/third_party/patches/isaac-gr00t-pyav-fallback.patch
PROV=${VENDOR}/PROVENANCE.json
UPSTREAM=https://github.com/NVIDIA/Isaac-GR00T.git

# Compute nodes have outbound network; the login node does not. If this is running anywhere
# without it, fail here rather than half-way through a 6 GB wheel download.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FATAL: not inside a SLURM job. The provider's login-node policy forbids installs there;"
  echo "       submit this with sbatch."
  exit 1
fi

echo "=== vendoring ${UPSTREAM} @ ${GR00T_COMMIT}"

# A rebuild must be reproducible, so a stale tree is removed rather than updated in place. rm -rf
# of a directory we created under project storage, NOT an mv: project ids live on the inode and
# survive a rename (docs/discoverer.md §9).
if [[ -d "${VENDOR}" ]]; then
  if [[ "${T39_ENV_REBUILD:-0}" != "1" ]]; then
    echo "FATAL: ${VENDOR} already exists. Set T39_ENV_REBUILD=1 to discard and rebuild it."
    echo "       Refusing by default: silently reusing a tree of unknown sha is how a run ends up"
    echo "       recording a commit it did not actually train under."
    exit 1
  fi
  echo "=== T39_ENV_REBUILD=1 — discarding ${VENDOR}"
  rm -rf "${VENDOR}"
fi

mkdir -p "$(dirname "${VENDOR}")"
git clone --quiet "${UPSTREAM}" "${VENDOR}"
git -C "${VENDOR}" checkout --quiet --detach "${GR00T_COMMIT}"
CLONED_SHA=$(git -C "${VENDOR}" rev-parse HEAD)
if [[ "${CLONED_SHA}" != "${GR00T_COMMIT}"* && "${GR00T_COMMIT}" != "${CLONED_SHA}"* ]]; then
  echo "FATAL: asked for ${GR00T_COMMIT}, got ${CLONED_SHA}."
  exit 1
fi
echo "=== vendored at ${CLONED_SHA}"

# -- the one patch, applied as a reviewable artifact rather than as an edit ----------------------
# PR-07 §3 requires the trainer vendored UNMODIFIED so that a failure is attributable to NVIDIA's
# recipe and not to our copy of it. This cluster has no FFmpeg, so upstream's torchcodec-only
# decoder cannot import and every dataloader worker dies. The patch restores a PyAV fallback and
# nothing else — no model, no trainer, no loss, no hyperparameter. It is a recorded exception to
# §3, not a loophole, and the patch file carries the full argument. It is applied here so the
# vendored tree is provably "upstream@sha + this exact patch" rather than "a fork someone packed".
if [[ ! -f "${PATCH}" ]]; then
  echo "FATAL: ${PATCH} missing — run cluster/discoverer/sync.sh first."
  exit 1
fi
PATCH_SHA=$(sha256sum "${PATCH}" | cut -d' ' -f1)
if [[ "${T39_SKIP_PATCH:-0}" == "1" ]]; then
  # The cleaner answer to the FFmpeg problem, if it ever works: leave upstream untouched and make
  # torchcodec load. Kept as a switch so that experiment is one variable, not a re-edit.
  echo "=== T39_SKIP_PATCH=1 — vendoring upstream UNPATCHED; torchcodec must load on its own"
  PATCH_SHA="skipped"
else
  echo "=== applying ${PATCH} (sha256 ${PATCH_SHA})"
  ( cd "${VENDOR}" && patch -p1 --forward < "${PATCH}" )
fi

# -- the venv -----------------------------------------------------------------------------------
# uv sync --frozen builds exactly upstream's uv.lock: python 3.10, torch 2.7.1+cu128,
# flash-attn 2.7.4 (prebuilt wheel), deepspeed 0.17.6. --frozen means the lock is obeyed, not
# re-resolved; a re-resolution here would be a different environment wearing the same name.
export UV_PROJECT_ENVIRONMENT=${VENV}
export UV_CACHE_DIR=${PROJ}/caches/uv
export UV_INSTALL_DIR=${PROJ}/tools/uv-bin
mkdir -p "${UV_CACHE_DIR}" "${UV_INSTALL_DIR}" "$(dirname "${VENV}")"
export PATH="${UV_INSTALL_DIR}:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv not found on PATH (${UV_INSTALL_DIR}). It was installed there for the July"
  echo "       apple_pnp_h200 build; if it is gone, reinstall it into project storage, never \$HOME."
  exit 1
fi
echo "=== uv $(uv --version)"

( cd "${VENDOR}" && uv sync --frozen )

# Upstream commit e5749287 deleted the `av` dependency in the same change that made decoding
# torchcodec-only, so the patch's fallback has no package to import unless we add it back. It is
# installed separately, and named in the provenance, so it is visible as OUR addition rather than
# hiding inside the lock.
if [[ "${T39_SKIP_PATCH:-0}" != "1" ]]; then
  ( cd "${VENDOR}" && uv pip install av )
fi

# -- prove it, before anything downstream trusts it ----------------------------------------------
# 70_* exits FATAL on a missing venv but cannot tell a working venv from a broken one, and a
# dependency failure that surfaces inside a GPU job costs GPU hours to discover.
echo "=== smoke test"
"${VENV}/bin/python" - <<'EOF'
import importlib.util, json, sys
import torch
out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    "deepspeed": importlib.util.find_spec("deepspeed") is not None,
    "av": importlib.util.find_spec("av") is not None,
}
import gr00t  # noqa: F401
out["gr00t_import"] = True

# Which decoder will actually run here? This is the whole reason the patch exists, and getting it
# wrong is silent: torchcodec importing on the login node and failing on a compute node would make
# the fallback look untested. Ask the module itself rather than assuming.
from gr00t.utils import video_utils
try:
    cls = video_utils._get_video_decoder_cls()
    out["video_decoder"] = f"{cls.__module__}.{cls.__name__}"
except Exception as exc:  # noqa: BLE001
    out["video_decoder"] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"

print(json.dumps(out, indent=2))
if not out["flash_attn"] or not out["deepspeed"]:
    raise SystemExit("FATAL: flash-attn or deepspeed missing — the trainer will not run.")
if out["video_decoder"].startswith("UNAVAILABLE"):
    raise SystemExit("FATAL: no usable video decoder — every dataloader worker would die.")
EOF

# `cuda available: false` is EXPECTED: this job holds no GPU. It is recorded, not asserted on.
echo "=== NOTE: cuda_available=false is expected here; this is a CPU-only job."

# -- provenance ----------------------------------------------------------------------------------
# AC-04: every rollout traceable to checkpoint + dataset snapshot + config hash. The trainer tree
# is part of that config, and until now it was a tarball nobody could name.
"${VENV}/bin/python" - "$PROV" "$CLONED_SHA" "$PATCH_SHA" "$VENDOR" "$VENV" <<'EOF'
import hashlib, json, subprocess, sys
prov_path, sha, patch_sha, vendor, venv = sys.argv[1:6]

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

freeze = subprocess.run([f"{venv}/bin/python", "-m", "pip", "freeze"],
                        capture_output=True, text=True).stdout
json.dump({
    "upstream": "https://github.com/NVIDIA/Isaac-GR00T.git",
    "commit": sha,
    "patch": "third_party/patches/isaac-gr00t-pyav-fallback.patch",
    "patch_sha256": patch_sha,
    "uv_lock_sha256": digest(f"{vendor}/uv.lock"),
    "pyproject_sha256": digest(f"{vendor}/pyproject.toml"),
    "venv": venv,
    "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    "note": ("Built by 72_build_t39_env.sbatch for PR-07 §8 item 4. The commit is NOT yet "
             "established as the revision NVIDIA documents for GR00T-N1.7 post-training — see "
             "PR-07 §8 items 5 and 6 and this script's header."),
}, open(prov_path, "w"), indent=2, sort_keys=True)
open(f"{vendor}/PIP_FREEZE.txt", "w").write(freeze)
print(open(prov_path).read())
EOF

echo
echo "=== DONE. Vendored ${VENDOR} @ ${CLONED_SHA}; venv ${VENV}"
echo "=== PR-07 §8 item 4 is satisfied. Items 5 (MODEL_ID) and 6 (entrypoints) are NOT."
