#!/usr/bin/env bash
# Local counterpart of cluster/discoverer/72_build_t39_env.sbatch — PR-07 §8 item 4, on this
# workstation instead of Discoverer+.
#
#   GR00T_COMMIT=1a1837f20538b7d7e21f977a11a5aee14f99803c bash scripts/build_t39_env_local.sh
#
# It vendors NVIDIA/Isaac-GR00T at a pinned sha into third_party/isaac-gr00t, applies the PyAV
# fallback patch registered for that sha, builds the trainer's OWN venv from upstream's uv.lock,
# smoke-tests it and writes PROVENANCE.json. Every decision below is the sbatch's; read its header
# for the arguments, which are not repeated here. Only the deltas are.
#
# DELTA 1 — THE PATCH IS APPLIED HERE AND IS INERT HERE, WHICH IS NOT WHAT WAS PREDICTED. The
# prediction, from upstream's own pyproject.toml at 1a1837f, was that it would be load-bearing on
# this box too:
#
#     # torchcodec 0.8.0 pairs with torch 2.9 and supports FFmpeg 4-7. It does NOT
#     # support FFmpeg 8 (the default on Ubuntu 25.10+/26.04); on those distros
#     # install an FFmpeg<8 runtime.
#
# and this box's system FFmpeg is 8.1.2. The smoke test below says otherwise — it reports
# `torchcodec.decoders._video_decoder.VideoDecoder`, and that decoder reads real corpus video
# (episode_000000.mp4 -> (3, 480, 640) uint8, 2026-08-16). torchcodec does not link the system
# FFmpeg here. So the shim is present and never entered, and the executed code path is upstream's.
#
# The patch is kept applied anyway, deliberately: on Discoverer+ it IS load-bearing (no FFmpeg at
# all), and a local tree that differs from the cluster tree would make the two runs incomparable
# for no gain. PROVENANCE.json names the patch and its sha256 either way. Do not infer from this
# comment which decoder you have — the smoke test is what decides, and it is the reason it asks the
# module for the class rather than assuming.
#
# DELTA 2 — NO SLURM GUARD. The sbatch refuses to run outside a job because the provider forbids
# installs on the login node. That rule is about Discoverer+ and does not travel here.
#
# DELTA 3 — THE VENV LIVES IN $HOME. On the cluster it is ${PROJ}/virt_envs/t39 because /home has
# an inode quota there. Here ~/venvs/t39 sits beside the existing arena/ and onnx/ envs.
#
# WHAT THIS SCRIPT DOES NOT DO. It does not download weights and it does not train. Item 5 wants
# MODEL_ID established from a primary source; that is a reading task, and docs/local_gr00t_assets.md
# §1 records the answer and where it was read.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR=${VENDOR:-${REPO}/third_party/isaac-gr00t}
VENV=${VENV:-${HOME}/venvs/t39}
PATCHES=${REPO}/third_party/patches
PROV=${VENDOR}/PROVENANCE.json
UPSTREAM=${UPSTREAM:-https://github.com/NVIDIA/Isaac-GR00T.git}

: "${GR00T_COMMIT:?set GR00T_COMMIT to the upstream NVIDIA/Isaac-GR00T commit to vendor. No default — see the sbatch header.}"

# The patch is selected BY COMMIT and an unregistered commit is fatal, for the reason spelled out
# in 72_build_t39_env.sbatch: a context diff does not travel between bases, and a rejected hunk
# would read to the operator as a build failure rather than as "wrong patch for this commit".
PATCH=""
case "${GR00T_COMMIT}" in
  e5749287*) PATCH=${PATCHES}/isaac-gr00t-pyav-fallback.patch ;;
  1a1837f*)  PATCH=${PATCHES}/isaac-gr00t-pyav-fallback-1a1837f.patch ;;
  *)
    if [[ "${T39_SKIP_PATCH:-0}" == "1" ]]; then
      echo "=== no patch registered for ${GR00T_COMMIT}, and T39_SKIP_PATCH=1 — continuing unpatched"
    else
      echo "FATAL: no PyAV fallback patch is registered for GR00T_COMMIT=${GR00T_COMMIT}."
      echo "       Re-derive it against this commit and add it to ${PATCHES} plus the case above."
      echo "       Do NOT edit an existing patch file — they are the provenance record for the runs"
      echo "       built on them (docs/handoff.md §3). To vendor unpatched, set T39_SKIP_PATCH=1."
      exit 1
    fi
    ;;
esac

command -v uv >/dev/null 2>&1 || { echo "FATAL: uv not on PATH."; exit 1; }
echo "=== uv $(uv --version)"

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

echo "=== vendoring ${UPSTREAM} @ ${GR00T_COMMIT}"
mkdir -p "$(dirname "${VENDOR}")"
git clone --quiet "${UPSTREAM}" "${VENDOR}"
git -C "${VENDOR}" checkout --quiet --detach "${GR00T_COMMIT}"
CLONED_SHA=$(git -C "${VENDOR}" rev-parse HEAD)
if [[ "${CLONED_SHA}" != "${GR00T_COMMIT}"* && "${GR00T_COMMIT}" != "${CLONED_SHA}"* ]]; then
  echo "FATAL: asked for ${GR00T_COMMIT}, got ${CLONED_SHA}."
  exit 1
fi
echo "=== vendored at ${CLONED_SHA}"

if [[ "${T39_SKIP_PATCH:-0}" == "1" ]]; then
  echo "=== T39_SKIP_PATCH=1 — vendoring upstream UNPATCHED; torchcodec must load on its own"
  PATCH_SHA="skipped"
else
  [[ -f "${PATCH}" ]] || { echo "FATAL: ${PATCH} missing."; exit 1; }
  PATCH_SHA=$(sha256sum "${PATCH}" | cut -d' ' -f1)
  echo "=== applying ${PATCH} (sha256 ${PATCH_SHA})"
  ( cd "${VENDOR}" && patch -p1 --forward < "${PATCH}" )
fi

export UV_PROJECT_ENVIRONMENT=${VENV}
mkdir -p "$(dirname "${VENV}")"
( cd "${VENDOR}" && uv sync --frozen )

# `av` is installed separately, and named in the provenance, so the fallback's dependency is
# visible as OUR addition rather than hiding inside upstream's lock. --python is not optional:
# UV_PROJECT_ENVIRONMENT is honoured by `uv sync` and NOT by `uv pip`.
if [[ "${T39_SKIP_PATCH:-0}" != "1" ]]; then
  ( cd "${VENDOR}" && uv pip install --python "${VENV}/bin/python" av )
fi

echo "=== smoke test"
"${VENV}/bin/python" - <<'EOF'
import importlib.util, json, sys
import torch
out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    "deepspeed": importlib.util.find_spec("deepspeed") is not None,
    "av": importlib.util.find_spec("av") is not None,
}
import gr00t  # noqa: F401
out["gr00t_import"] = True

# Which decoder will actually run here? Getting this wrong is silent, so ask the module rather
# than assuming — on this box FFmpeg 8 makes torchcodec unusable and the answer must be the shim.
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

# -- provenance (AC-04) --------------------------------------------------------------------------
"${VENV}/bin/python" - "$PROV" "$CLONED_SHA" "$PATCH_SHA" "$VENDOR" "$VENV" "${PATCH:-skipped}" <<'EOF'
import hashlib, json, os, subprocess, sys
prov_path, sha, patch_sha, vendor, venv, patch_path = sys.argv[1:7]

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# `uv pip freeze --python`, NOT `python -m pip freeze`: a uv-created venv ships no pip, and the
# empty-string sha256 is a record of nothing that reads like a dependency record.
freeze = subprocess.run(["uv", "pip", "freeze", "--python", f"{venv}/bin/python"],
                        capture_output=True, text=True).stdout
if not freeze.strip():
    raise SystemExit(f"FATAL: `uv pip freeze` produced nothing for {venv}.")
json.dump({
    "upstream": "https://github.com/NVIDIA/Isaac-GR00T.git",
    "commit": sha,
    "patch": os.path.basename(patch_path),
    "patch_sha256": patch_sha,
    "uv_lock_sha256": digest(f"{vendor}/uv.lock"),
    "pyproject_sha256": digest(f"{vendor}/pyproject.toml"),
    "venv": venv,
    "venue": "workstation",
    "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    "note": ("Built by scripts/build_t39_env_local.sh for PR-07 §8 item 4, on the workstation "
             "rather than Discoverer+. PR-07 pre-registers the cluster as the venue; a run built "
             "from this tree must record the deviation rather than inherit the sbatch's claim."),
}, open(prov_path, "w"), indent=2, sort_keys=True)
open(f"{vendor}/PIP_FREEZE.txt", "w").write(freeze)
print(open(prov_path).read())
EOF

echo
echo "=== DONE. Vendored ${VENDOR} @ ${CLONED_SHA}; venv ${VENV}"
