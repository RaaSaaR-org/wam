#!/bin/bash
# Step 19b — T-040 / PR-08 §8 item 3: warm upstream's OWN checkpoint registry into the hub cache,
# so that the timing run in 97 measures generation and not a download.
#
# WHY THIS EXISTS SEPARATELY FROM 99.
# 99 stages our pinned revision into a local directory and hands 97 the root as
# --checkpoint-path. That is the right thing for a SINGLE-control run: upstream keeps the override
# (cosmos_transfer2/inference.py:52-62) and generates from the bytes 99 recorded in STAGED.json.
#
# PR-08's committed control set is not single. CONTROL=depth:0.5,seg:0.5 gives
# Control2WorldInference two hint keys, which takes the multi-branch branch (:64-72). That branch:
#
#   1. NEVER READS args.checkpoint_path. 99's whole staged tree is unused on this path -- the
#      override is not overridden, it is not consulted.
#   2. Loads ALL FOUR of CONTROL_KEYS = ["edge", "vis", "depth", "seg"], not the two the sample
#      actually carries. Upstream's own comment: "load ALL control modalities even if some have
#      control weight = 0".
#   3. Resolves each one through download_checkpoint(), i.e. an HF fetch, at the revision hardcoded
#      per checkpoint in packages/cosmos-oss/cosmos_oss/checkpoints_transfer2.py. Those revisions
#      are NOT ${TRANSFER_MODEL_REVISION} and they are not even equal to each other -- as of
#      framework sha 2ff49d0 they are four different commits.
#
# So on the committed control set, a cold run downloads roughly 22 GB from inside whatever it is
# doing. In the TIMING path that download lands inside the measured window, and the resulting
# "seconds per frame" is mostly network. That is how a throughput gate gets a number that has
# nothing to do with throughput.
#
# NOTHING HERE IS HARDCODED. The UUIDs, revisions and filenames are read out of the framework
# checkout at ${TRANSFER_FRAMEWORK_SHA}, by calling the same download_checkpoint() the inference
# calls, on the same MODEL_CHECKPOINTS[ModelKey(variant=key)] lookup the inference performs. A
# plausible UUID typed into this file by whoever wrote it is exactly the failure 99's header
# refuses to commit, and it would be worse here: it would silently succeed.
#
# IT ALSO ANSWERS A QUESTION 99 LEFT OPEN. 99 staged both .pt files per control and declined to
# choose ("selection belongs to the driver, reading the framework checkout's own config"). The
# framework checkout's own config says: general/depth/626e6618-..., general/edge/61f5694b-...,
# general/seg/5136ef49-... -- the 5.53 GB member of each pair. The fourth, ModelVariant.VIS, is
# backed by general/blur/ba2f44f2-..., the directory 99 deliberately skipped as "a control PR-08
# does not use". PR-08 does not use it. Upstream loads it anyway.
#
#   sbatch 99b_stage_transfer25_multibranch.sbatch
#
# Read-only with respect to the experiment: it writes the hub cache and one manifest, and touches
# neither ${STAGE_DIR} nor cosmos_transfer_env.sh.
#
#SBATCH --job-name=wam-transfer25-mb
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/transfer25-multibranch.%j.out

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

: "${TRANSFER_FRAMEWORK:?${TRANSFER_ENV} must export TRANSFER_FRAMEWORK — 98 writes it.}"
[[ -d "${TRANSFER_FRAMEWORK}/.venv" ]] || {
  echo "FATAL: no venv at ${TRANSFER_FRAMEWORK}/.venv. Run 98_build_transfer25_env.sbatch first."
  exit 1; }

MANIFEST=${PROJ}/checkpoints/MULTIBRANCH_STAGED.json

echo "=== framework ${TRANSFER_FRAMEWORK} @ ${TRANSFER_FRAMEWORK_SHA:-unknown}"
echo "=== hub cache ${HF_HUB_CACHE}"
echo "=== manifest  ${MANIFEST}"

# -- the token ------------------------------------------------------------------------------------
# Same rule as 99: public repo, but anonymous means rate-limited, and a 429 halfway through a 22 GB
# fetch is how 100_fetch_pr08_source failed at 80%. Never echoed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -f "${f}" ]] && { HF_TOKEN=$(<"${f}"); break; }
  done
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  echo "=== hf token  present"
else
  echo "=== hf token  ABSENT — anonymous, and anonymous means rate-limited. Expect a 429."
fi

# shellcheck source=/dev/null
source "${TRANSFER_FRAMEWORK}/.venv/bin/activate"
cd "${TRANSFER_FRAMEWORK}"

python - "${MANIFEST}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

out = Path(sys.argv[1])

# The same three imports Control2WorldInference.__init__ makes, in the same order, so that what is
# resolved here is what is resolved there. CONTROL_KEYS is upstream's list and is NOT filtered down
# to the controls PR-08 uses: the multi-branch loader iterates all of it.
from cosmos_transfer2.config import CONTROL_KEYS, MODEL_CHECKPOINTS, ModelKey
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint

print(f"    CONTROL_KEYS = {CONTROL_KEYS}", flush=True)

records = []
for key in CONTROL_KEYS:
    cfg = MODEL_CHECKPOINTS[ModelKey(variant=key)]
    print(f"--- {key:5s} {cfg.full_name}", flush=True)
    print(f"      repo     {cfg.hf.repository}", flush=True)
    print(f"      revision {cfg.hf.revision}", flush=True)
    print(f"      file     {cfg.hf.filename}", flush=True)
    # download_checkpoint(), not hf download: this is the exact call the inference makes, so a
    # success here is proof that the inference will find it rather than a claim that it should.
    local = download_checkpoint(cfg.s3.uri)
    p = Path(local)
    if not p.is_file():
        sys.exit(f"FATAL: {key}: download_checkpoint returned {local}, which is not a file.")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    rec = {
        "control_key": key,
        "uuid": cfg.uuid,
        "name": cfg.name,
        "experiment": cfg.experiment,
        "repository": cfg.hf.repository,
        "revision": cfg.hf.revision,
        "filename": cfg.hf.filename,
        "local_path": str(p),
        "bytes": p.stat().st_size,
        "sha256": h.hexdigest(),
    }
    records.append(rec)
    print(f"      -> {p}  {rec['bytes']/1e9:.2f} GB  sha256 {rec['sha256'][:16]}", flush=True)

total = sum(r["bytes"] for r in records)
revs = sorted({r["revision"] for r in records})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "purpose": "checkpoints the MULTI-BRANCH inference path resolves for itself; see 99b's header",
    "resolved_by": "cosmos_transfer2._src.imaginaire.utils.checkpoint_db.download_checkpoint",
    "framework_sha": os.environ.get("TRANSFER_FRAMEWORK_SHA", ""),
    "control_keys": list(CONTROL_KEYS),
    "distinct_revisions": revs,
    "our_pin_TRANSFER_MODEL_REVISION": os.environ.get("TRANSFER_MODEL_REVISION", ""),
    "note": (
        "These revisions are upstream's, hardcoded per checkpoint in checkpoints_transfer2.py. "
        "They are NOT our_pin_TRANSFER_MODEL_REVISION. On the multi-branch path the run generates "
        "from THESE bytes and 99's STAGED.json describes bytes that were not loaded."
    ),
    "checkpoints": records,
}, indent=2) + "\n")

print(f"\n    {len(records)} checkpoints, {total/1e9:.2f} GB -> {out}")
print(f"    {len(revs)} distinct upstream revisions: {revs}")
PY

echo
echo "=== 97 TIMING=1 can now run without downloading inside the measured window."
echo "===   It STILL WILL NOT GENERATE: PR-08 §8 items 2 and 4 are open and T40_RULE_V1 §1 forbids"
echo "===   generation until every item is closed. TIMING=1 is the licensed path (PR-08 §1)."
