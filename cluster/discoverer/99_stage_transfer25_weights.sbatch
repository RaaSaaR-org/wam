#!/bin/bash
# Step 19 — T-040 / PR-08 §8 item 3: stage Cosmos-Transfer2.5 at the pinned revision.
#
#   sbatch 99_stage_transfer25_weights.sbatch
#
# Completes the `TODO(human)` at 97_transfer25_restyle.sbatch:255 that 98 began: 98 writes the
# framework half of ${PROJ}/cosmos_transfer_env.sh, this appends the three checkpoint names 97
# asserts (TRANSFER_MODEL_ID, TRANSFER_MODEL_REVISION, TRANSFER_CHECKPOINT_PATH).
#
# ================================================================================================
# THE LICENCE GATE IS A HUMAN ACT AND THIS JOB CANNOT PERFORM IT.
#
# nvidia/Cosmos-Transfer2.5-2B is gated ("gated": "auto" on the HF API). Auto-approval still
# requires a person to request access under the NVIDIA Open Model License while signed in. Checked
# 2026-08-18 with the token in this account's HF cache: a HEAD on the repo's config.json returned
# **403**. Accepting a model licence is the account holder's act -- not an agent's, and not a batch
# job's. So this file CHECKS the gate and refuses with instructions; it never tries to work around
# one. If the check fails, the fix is a browser, not a flag:
#
# THE ACCOUNT HOLDER ACCEPTED IT ON 2026-08-19, and the same probe against the PINNED revision now
# returns 200. The check below stays exactly as it was: gate state is a property of an account at a
# moment, not of this file, and a token rotation or a licence revision puts the 403 back. A guard
# deleted because it passed once is a guard that reports nothing the day it matters.
#
#     https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B     <- accept, then re-run this job
#
# ================================================================================================
# WHAT IT DOWNLOADS, AND WHAT IT DELIBERATELY DOES NOT.
# The repo is 55.24 GB over 23 entries. PR-08 §4 uses depth + segmentation + Canny edge, so this
# stages `general/depth`, `general/seg` and `general/edge` -- 30.75 GB. It SKIPS:
#   auto/multiview  (9.56 GB)  driving-domain multiview, a different deliverable
#   general/blur    (10.25 GB) a control PR-08 does not use
#   distilled/      (4.72 GB)  a distilled edge variant; a different generator, and PR-08 §1 fixes
#                              the generator as frozen Transfer2.5, so silently swapping in a
#                              distilled model would change what the experiment measures
# Set TRANSFER_STAGE_ALL=1 to take the whole repo instead; the manifest records which was taken.
#
# EACH CONTROL SHIPS **TWO** CHECKPOINTS (4.72 GB and 5.53 GB) AND THE REPO DOCUMENTS NEITHER.
# There is no config.json, no README mapping and no public UUID->variant table -- checked on the
# HF API and in the GitHub README on 2026-08-18. So this job does NOT choose between them and does
# NOT hardcode a UUID. It stages both, records every file with its size and sha256 in STAGED.json,
# and points TRANSFER_CHECKPOINT_PATH at the staged ROOT. Selection belongs to the driver, reading
# the framework checkout's own config -- picking one here would be a plausible wrong value chosen
# by whoever wrote this line, which is the single failure PR-08's guards exist to prevent.
#
# NO GPU, SO NO GPU-HOURS. 91 needs a GPU because convert_model_to_dcp materialises the model; this
# job only downloads and hashes. It runs on the free 2cpu-single-host QoS and bills zero against
# PR-08's (still unwritten) ceiling. That matters: §8 item 3 says no budget line exists until the
# throughput measurement does, so staging must not spend from a budget that does not yet exist.
#
#SBATCH --job-name=wam-transfer25-stage
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/transfer25-stage.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}

if [[ ! -f "${TRANSFER_ENV}" ]]; then
  echo "FATAL: ${TRANSFER_ENV} does not exist. Run 98_build_transfer25_env.sbatch first."
  exit 1
fi
# shellcheck source=/dev/null
source "${TRANSFER_ENV}"

# THE REVISION IS PINNED AND HAS NO FLOATING DEFAULT -- the rule 91's header states, for the same
# reason. ce8440327c632d8313c3bde69db13b627ba5cae1 was read off the HF API on 2026-08-18 as the
# then-current `main`. `main` is not a checkpoint id.
#
# This is not only a reproducibility nicety here: upstream's own installation guide says
# "checkpoints are automatically downloaded during inference", i.e. the framework will happily
# fetch a FLOATING revision at generation time if it finds nothing staged. PR-08 §6 requires the
# generator's id AND revision recorded regardless of verdict, so pre-staging at a pin -- and
# pointing HF_HOME at it -- is what stops an unpinned auto-download from deciding what we ran.
TRANSFER_MODEL_ID=${TRANSFER_MODEL_ID:-nvidia/Cosmos-Transfer2.5-2B}
TRANSFER_MODEL_REVISION=${TRANSFER_MODEL_REVISION:-ce8440327c632d8313c3bde69db13b627ba5cae1}
STAGE_DIR=${STAGE_DIR:-${PROJ}/checkpoints/Cosmos-Transfer2.5-2B}
TRANSFER_STAGE_ALL=${TRANSFER_STAGE_ALL:-0}

case "${TRANSFER_MODEL_REVISION}" in
  main|master|HEAD|latest|"")
    echo "FATAL: TRANSFER_MODEL_REVISION='${TRANSFER_MODEL_REVISION}' is a moving pointer, not a"
    echo "       revision. PR-08 §6 requires the pinned commit. Read one off the HF API:"
    echo "         curl -s .../api/models/${TRANSFER_MODEL_ID} | python -c 'import json,sys;print(json.load(sys.stdin)[\"sha\"])'"
    exit 1 ;;
esac
[[ "${TRANSFER_MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FATAL: TRANSFER_MODEL_REVISION='${TRANSFER_MODEL_REVISION}' is not a 40-hex commit sha."; exit 1; }

echo "=== model     ${TRANSFER_MODEL_ID}"
echo "=== revision  ${TRANSFER_MODEL_REVISION}"
echo "=== into      ${STAGE_DIR}"

export PATH=${UV_INSTALL_DIR}:${PATH}

# -- the token ------------------------------------------------------------------------------------
# Read from the environment first, then the standard cache. Never echoed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -f "${f}" ]] && { HF_TOKEN=$(<"${f}"); break; }
  done
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: no Hugging Face token. Export HF_TOKEN, or run 'hf auth login' as the account"
  echo "       holder so a token lands in the HF cache. A READ token is enough."
  exit 1
fi
export HF_TOKEN
echo "=== HF token present (${#HF_TOKEN} chars, not echoed)"

# -- THE GATE CHECK, before a single byte ----------------------------------------------------------
# A 403 here is the licence gate, not a network fault, and the remedy is a person accepting it.
# Checked before the download so the failure costs seconds instead of a partial 30 GB tree.
GATE_URL="https://huggingface.co/${TRANSFER_MODEL_ID}/resolve/${TRANSFER_MODEL_REVISION}/README.md"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -I -L -H "Authorization: Bearer ${HF_TOKEN}" "${GATE_URL}" || echo 000)
echo "=== gate probe -> HTTP ${CODE}"
case "${CODE}" in
  200) echo "=== access confirmed" ;;
  401|403)
    echo
    echo "FATAL: HTTP ${CODE} — this account cannot download ${TRANSFER_MODEL_ID}."
    echo
    echo "  ${TRANSFER_MODEL_ID} is gated under the NVIDIA Open Model License. Access is"
    echo "  auto-approved but must still be REQUESTED, while signed in, by the account holder:"
    echo
    echo "      https://huggingface.co/${TRANSFER_MODEL_ID}"
    echo
    echo "  Accepting a model licence is the account holder's act. This job will not attempt to"
    echo "  work around the gate, and no flag here can. Accept it, then resubmit this job."
    echo
    echo "  IF IT WAS ALREADY ACCEPTED, THE LIKELY CAUSE IS THAT THIS IS A DIFFERENT ACCOUNT."
    echo "  The gate is granted to a USER, not to a repository: acceptance in a browser on one"
    echo "  machine does nothing for a token minted by another account. The token this job just"
    echo "  used belongs to:"
    WHO=$(curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
            https://huggingface.co/api/whoami-v2 \
          | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name","<unknown>"))' \
          2>/dev/null || echo "<could not resolve>")
    echo "      ${WHO}"
    echo "  Compare that against the account that accepted the licence. If they differ, copy that"
    echo "  account's read token to \${HF_HOME}/token, or export HF_TOKEN for this submission."
    exit 1 ;;
  *)
    echo "FATAL: unexpected HTTP ${CODE} probing the repo. Not treating this as a gate failure —"
    echo "       check the network and the revision before rerunning."
    exit 1 ;;
esac

# -- the download ----------------------------------------------------------------------------------
mkdir -p "${STAGE_DIR}"
PATTERNS=()
if [[ "${TRANSFER_STAGE_ALL}" != "0" ]]; then
  echo "=== TRANSFER_STAGE_ALL=${TRANSFER_STAGE_ALL} — taking the whole 55.24 GB repo"
else
  PATTERNS=(--include "general/depth/*" --include "general/seg/*" --include "general/edge/*"
            --include "*.md" --include "*.json" --include "*.yaml" --include "*.txt")
  echo "=== staging general/{depth,seg,edge} only (~30.75 GB); set TRANSFER_STAGE_ALL=1 for all"
fi

uvx hf@latest download "${TRANSFER_MODEL_ID}" \
    --revision "${TRANSFER_MODEL_REVISION}" \
    --local-dir "${STAGE_DIR}" \
    "${PATTERNS[@]}"

# -- the manifest, read back off the bytes rather than asserted -------------------------------------
# Every .pt on disk with its size and sha256, plus the requested revision. This is what makes the
# UUID choice auditable later: whoever selects a variant can name the exact file they selected.
python3 - "${STAGE_DIR}" "${TRANSFER_MODEL_ID}" "${TRANSFER_MODEL_REVISION}" "${TRANSFER_STAGE_ALL}" <<'PY'
import hashlib, json, sys
from pathlib import Path

stage, model_id, revision, stage_all = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
files = sorted(p for p in stage.rglob("*.pt") if p.is_file())
if not files:
    sys.exit(f"FATAL: no .pt checkpoints under {stage} — the revision, the patterns or the repo is wrong.")

records, total = [], 0
for p in files:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    size = p.stat().st_size
    total += size
    rel = p.relative_to(stage).as_posix()
    records.append({"path": rel, "control": rel.split("/")[-2], "bytes": size, "sha256": h.hexdigest()})
    print(f"    {size/1e9:6.2f} GB  {h.hexdigest()[:16]}  {rel}")

out = stage / "STAGED.json"
out.write_text(json.dumps({
    "model_id": model_id,
    "revision": revision,
    "staged_all": stage_all != "0",
    "total_bytes": total,
    "checkpoints": records,
    # Recorded as an open choice rather than resolved here: each control ships two checkpoints and
    # upstream documents neither. The driver selects, against the framework's own config.
    "variant_selected": None,
}, indent=2) + "\n")
print(f"\n    {len(records)} checkpoints, {total/1e9:.2f} GB total -> {out}")
PY

# -- append the three names 97 asserts ---------------------------------------------------------------
# TRANSFER_CHECKPOINT_PATH is the staged ROOT, not one .pt: the variant is unresolved by design
# (see the header), and 97 passes this straight to the driver as --checkpoint-path.
#
# Appended rather than rewritten so 98's framework half survives, matching how 91 extends 90's file.
# Guarded so a rerun does not stack duplicate exports whose last-wins order nobody can read.
python3 - "${PROJ}/cosmos_transfer_env.sh" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
kept = [ln for ln in p.read_text().splitlines()
        if not ln.startswith(("export TRANSFER_MODEL_ID=", "export TRANSFER_MODEL_REVISION=",
                              "export TRANSFER_CHECKPOINT_PATH=", "export HF_HOME="))]
p.write_text("\n".join(kept) + "\n")
PY

cat >> "${PROJ}/cosmos_transfer_env.sh" <<EOF
# Appended by 99_stage_transfer25_weights.sbatch on $(date -Iseconds).
export TRANSFER_MODEL_ID=${TRANSFER_MODEL_ID}
export TRANSFER_MODEL_REVISION=${TRANSFER_MODEL_REVISION}
export TRANSFER_CHECKPOINT_PATH=${STAGE_DIR}
EOF

echo
echo "=== ${PROJ}/cosmos_transfer_env.sh now reads:"
cat "${PROJ}/cosmos_transfer_env.sh"
echo
echo "=== 97 can now resolve its four names. IT STILL WILL NOT GENERATE: PR-08 §8 items 2 and 4"
echo "===   are open, and T40_RULE_V1 §1 forbids generation until EVERY item is closed."
echo "===   TIMING=1 is the licensed path (§1: 'timing one episode on an H200')."
