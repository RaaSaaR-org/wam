#!/bin/bash
# Step 18 — T-040 / PR-08 §8 item 3: build the Cosmos-Transfer2.5 env and write cosmos_transfer_env.sh.
#
#   sbatch 98_build_transfer25_env.sbatch
#
# THIS FILE IS THE `TODO(human)` AT 97_transfer25_restyle.sbatch:255. That file asserts four names
# and dies with a message naming a job that did not exist. It exists now, and 99 supplies the rest.
#
# WHAT IT DOES NOT DO. It builds an environment. It generates nothing, trains nothing and stages no
# weights. PR-08 §1 licenses "writing the pipeline ... and timing one episode on an H200"
# unconditionally and conditions only *generating a corpus* on every §8 item being closed; building
# the env this file builds is on the licensed side, and 97 keeps its own generation guards
# regardless of what this writes.
#
# WHY A THIRD ENV, AND THE COLLISION THAT MAKES IT NECESSARY.
# 90_build_cosmos_env.sbatch writes ${PROJ}/cosmos_env.sh for the Cosmos3 GENERATOR family
# (T-041/PR-09). Transfer2.5 is a different repository with a different dependency set, and both
# files export the name FRAMEWORK pointing at different checkouts. **Nothing may source both.**
# 97 sources only cosmos_transfer_env.sh, which is why that is safe today; if a future job needs
# ffmpeg from one and the framework from the other, it must take FFMPEG_PREFIX explicitly rather
# than sourcing two files and letting the second win. Recorded here because a silently-overwritten
# FRAMEWORK produces a run that works, uses the wrong checkout, and says so nowhere.
#
# THE FRAMEWORK COMMIT IS PINNED AND HAS NO FLOATING DEFAULT, the same rule 91 states for weights:
# 2ff49d0717af02057ae79bc75c00fbff9da1b4e7, read off the GitHub API on 2026-08-18. `main` is not a
# revision. A run that cannot name the code it ran is not reproducible, and PR-08 §6 requires the
# generator recorded regardless of verdict.
#
# THE CUDA BUILD IS VERIFIED, NOT ASSUMED, AND THE EXTRA IS NOT OPTIONAL.
# 90 hardcodes `--group=cu128-train` because the Cosmos3 setup guide names that group. This is a
# different repository, so the value here was read off ITS pyproject.toml at the pinned sha rather
# than guessed:
#
#     [project.optional-dependencies]
#     cu128 = ["cosmos-oss[cu128_torch27]"]
#     cu130 = ["cosmos-oss[cu130_torch29]"]
#     [tool.uv] conflicts = [[{extra = "cu128"}, {extra = "cu129"}, {extra = "cu130"}]]
#
# **torch appears in NEITHER the base dependencies NOR a default extra.** A plain `uv sync` on this
# project therefore succeeds and produces a venv with no torch in it at all -- not a CPU build, not
# a mismatched build, none. The default below is consequently `--extra cu128`, because this
# cluster's newest CUDA module is 12.8 (docs/discoverer.md §3) and cu130 would import cleanly and
# die on the first kernel launch. The check after the install still verifies the OUTCOME: the
# default is a claim about what upstream declared today, and the check is what notices when the
# pin moves or the extra is overridden.
#
# TWO THINGS §4 NEEDS SHIP IN THIS ENV'S BASE DEPENDENCIES, which is worth knowing before anyone
# prices them as separate work: `sam2>=1.1.0` (the segmenter measure_geom_tol.py refuses to
# substitute a red-pixel heuristic for) and `video-depth-anything` (a monocular depth estimator,
# i.e. the thing whose error EST_DRIFT_P95 is). Neither is installed on this workstation.
#
#SBATCH --job-name=wam-transfer25-env
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# 2cpu-single-host: costs NO GPU hours and caps at 4 h. This job compiles nothing large; if the
# torch download misses the window, resubmit with the same flags 90's header lists rather than
# trimming the install. That path DOES bill, so record it if taken.
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/transfer25-env.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WORK=${PROJ}/scratch/${USER}

TRANSFER_REPO_URL=${TRANSFER_REPO_URL:-https://github.com/nvidia-cosmos/cosmos-transfer2.5}
TRANSFER_FRAMEWORK_SHA=${TRANSFER_FRAMEWORK_SHA:-2ff49d0717af02057ae79bc75c00fbff9da1b4e7}
TRANSFER_ROOT=${TRANSFER_ROOT:-${PROJ}/third_party/cosmos-transfer2.5}
# --extra cu128, not empty: see the header. A plain sync installs no torch on this project.
TRANSFER_UV_ARGS=${TRANSFER_UV_ARGS:---extra cu128}

echo "=== repo      ${TRANSFER_REPO_URL}"
echo "=== pinned at ${TRANSFER_FRAMEWORK_SHA}"
echo "=== into      ${TRANSFER_ROOT}"

# -- uv, installed the way 90 installs it, and idempotently ---------------------------------------
export UV_CACHE_DIR=${UV_CACHE_DIR:-${WORK}/uv_cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${PROJ}/uv_python}
export UV_INSTALL_DIR=${UV_INSTALL_DIR:-${PROJ}/tools/uv}
export PATH=${UV_INSTALL_DIR}:${PATH}
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${UV_INSTALL_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv into ${UV_INSTALL_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
fi
uv --version

# -- ffmpeg: reused, never rebuilt ----------------------------------------------------------------
# 90 already built one under ${PROJ}/tools/ffmpeg and 97 wants FFMPEG_PREFIX. Rebuilding a second
# copy would burn an hour to produce a byte-identical tree, so this asserts rather than builds --
# but it asserts against the PATH, not against the directory existing, because a half-finished
# build leaves the directory behind.
export FFMPEG_PREFIX=${FFMPEG_PREFIX:-${PROJ}/tools/ffmpeg}
if [[ ! -x "${FFMPEG_PREFIX}/bin/ffmpeg" ]]; then
  echo "FATAL: no ffmpeg at ${FFMPEG_PREFIX}/bin/ffmpeg."
  echo "       90_build_cosmos_env.sbatch builds it. Run that first, or point FFMPEG_PREFIX at"
  echo "       another prefix. This job does not build a second copy on purpose."
  exit 1
fi
"${FFMPEG_PREFIX}/bin/ffmpeg" -version | head -1

# -- the checkout, at the pin ---------------------------------------------------------------------
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$(dirname "${TRANSFER_ROOT}")"
if [[ ! -d "${TRANSFER_ROOT}/.git" ]]; then
  git clone "${TRANSFER_REPO_URL}" "${TRANSFER_ROOT}"
fi
git -C "${TRANSFER_ROOT}" fetch --all --tags
git -C "${TRANSFER_ROOT}" checkout --detach "${TRANSFER_FRAMEWORK_SHA}"

GOT_SHA=$(git -C "${TRANSFER_ROOT}" rev-parse HEAD)
if [[ "${GOT_SHA}" != "${TRANSFER_FRAMEWORK_SHA}" ]]; then
  echo "FATAL: asked for ${TRANSFER_FRAMEWORK_SHA}, HEAD is ${GOT_SHA}."
  exit 1
fi
echo "=== checkout is at ${GOT_SHA}"

# What groups/extras this project actually offers, printed rather than guessed. If TRANSFER_UV_ARGS
# has to be set on a later run, this output is where the legal values come from.
echo "=== dependency groups/extras declared by the checkout:"
python3 - "${TRANSFER_ROOT}/pyproject.toml" <<'PY' || echo "    (could not parse pyproject.toml — read it by hand before setting TRANSFER_UV_ARGS)"
import sys, tomllib
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("    (no pyproject.toml at the checkout root)")
    raise SystemExit(0)
d = tomllib.loads(p.read_text())
groups = list((d.get("dependency-groups") or {}).keys())
extras = list(((d.get("project") or {}).get("optional-dependencies") or {}).keys())
print("    groups:", ", ".join(groups) if groups else "(none)")
print("    extras:", ", ".join(extras) if extras else "(none)")
PY

# -- the install ----------------------------------------------------------------------------------
cd "${TRANSFER_ROOT}"
echo "=== uv sync ${TRANSFER_UV_ARGS}"
# Unquoted on purpose: TRANSFER_UV_ARGS is a word list (e.g. "--extra cu128"), not one word.
# shellcheck disable=SC2086
uv sync ${TRANSFER_UV_ARGS}

if [[ ! -x "${TRANSFER_ROOT}/.venv/bin/python" ]]; then
  echo "FATAL: uv sync finished but ${TRANSFER_ROOT}/.venv/bin/python does not exist."
  exit 1
fi

# -- verify the CUDA build against the driver, per the header --------------------------------------
# No GPU is allocated on this QoS, so torch.cuda.is_available() is expected to be False and is NOT
# the check. The check is the CUDA MAJOR the wheel was built against vs. what the cluster provides:
# a cu13x wheel on a 12.x driver imports fine here and dies on the first kernel launch four hours
# into a generation job. nvidia-smi is absent on a CPU node, so the driver figure comes from the
# module tree, and if neither can be read the job says so rather than quietly passing.
"${TRANSFER_ROOT}/.venv/bin/python" - <<'PY'
try:
    import torch
except ModuleNotFoundError:
    raise SystemExit(
        "FATAL: the venv has no torch. On this project torch arrives ONLY through an extra "
        "(cu128 / cu130) -- it is in neither the base dependencies nor a default -- so a sync "
        "with no --extra succeeds and installs none. Re-run with TRANSFER_UV_ARGS='--extra cu128' "
        "(this cluster is CUDA 12.8); the extras the checkout declares were printed above."
    )
built = torch.version.cuda
print(f"    torch {torch.__version__}, built against CUDA {built}")
if built is None:
    raise SystemExit("FATAL: this is a CPU-only torch build. Transfer2.5 inference needs CUDA.")
major = int(str(built).split(".")[0])
if major != 12:
    raise SystemExit(
        f"FATAL: torch is built against CUDA {built} (major {major}). This cluster's newest CUDA "
        f"module is 12.8 (docs/discoverer.md §3), so a major other than 12 will import cleanly and "
        f"then fail on the first kernel launch. Re-run with TRANSFER_UV_ARGS naming this project's "
        f"CUDA 12 group -- the groups/extras it declares were printed above."
    )
print("    CUDA major 12 — matches this cluster's 12.8 driver")
PY

# -- the env file ----------------------------------------------------------------------------------
# FRAMEWORK is the name 97 sources and activates (`${FRAMEWORK}/.venv/bin/activate`), so it points
# at the checkout root that owns the .venv uv just created. TRANSFER_FRAMEWORK is the same path
# under an unambiguous name, for any future job that must hold both frameworks at once without the
# collision this file's header describes.
#
# The three TRANSFER_MODEL_* / TRANSFER_CHECKPOINT_PATH names 97 asserts are deliberately NOT
# written here — 99_stage_transfer25_weights.sbatch appends them once the bytes are on disk at a
# pinned revision. Until then 97's `: ${..:?}` guards fire, which is the correct behaviour: an env
# file that names a checkpoint nobody has staged is worse than one that admits it.
cat > "${PROJ}/cosmos_transfer_env.sh" <<EOF
# Written by 98_build_transfer25_env.sbatch on $(date -Iseconds). Sourced by 97.
# DO NOT source this together with cosmos_env.sh — both export FRAMEWORK, pointing at different
# checkouts, and the second one sourced silently wins. See 98's header.
export TRANSFER_FRAMEWORK=${TRANSFER_ROOT}
export FRAMEWORK=${TRANSFER_ROOT}
export TRANSFER_REPO_URL=${TRANSFER_REPO_URL}
export TRANSFER_FRAMEWORK_SHA=${GOT_SHA}
export UV_CACHE_DIR=${UV_CACHE_DIR}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}
export UV_INSTALL_DIR=${UV_INSTALL_DIR}
export FFMPEG_PREFIX=${FFMPEG_PREFIX}
EOF

echo
echo "=== wrote ${PROJ}/cosmos_transfer_env.sh"
cat "${PROJ}/cosmos_transfer_env.sh"
echo
echo "=== NEXT: 99_stage_transfer25_weights.sbatch appends TRANSFER_MODEL_ID,"
echo "===       TRANSFER_MODEL_REVISION and TRANSFER_CHECKPOINT_PATH. 97 refuses to run until"
echo "===       it does, and 99 refuses to run until the HF licence gate is accepted by the"
echo "===       ACCOUNT HOLDER — that acceptance is not something a job or an agent can do."
