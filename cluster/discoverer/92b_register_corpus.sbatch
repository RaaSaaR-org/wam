#!/bin/bash
# Step 17b — register the workstation-built T-041 corpus with the cluster.
#
#   sbatch 92b_register_corpus.sbatch
#
# WHY THIS EXISTS. Corpus preparation and captioning moved off the cluster to workstation/20 and
# workstation/30 (PR-09 §4e): they need one local GPU for an evening instead of a Slurm
# reservation, and what ships is a finished dataset. But 94_train_t041_cosmos_super.sbatch reads
# DATASET_PATH and CORPUS_MANIFEST_SHA256 out of ${PROJ}/cosmos_env.sh, and the ONLY thing that
# ever wrote those two exports was 93_caption_corpus.sbatch — which is superseded and must not run
# (it re-captions the untranscoded tree; that is job 186357, 372 clips, 0 captions, no error).
#
# So after the move, nothing wrote DATASET_PATH. 94 would queue behind every other multi-GPU
# request for the ONLY 8-GPU node on the machine, be allocated all of dgx1, and exit in under a
# second at its `: "${DATASET_PATH:?...}"` guard — having spent the scarcest resource here and
# printed an error naming a job you must not run. This job closes that gap, and it verifies the
# corpus rather than trusting it, because the transfer is 14 GB over ssh and a truncated file is
# indistinguishable from a complete one until the trainer reads it.
#
# FREE QOS ON PURPOSE. Verification and a file append need no GPU. --qos=2cpu-single-host is not
# charged; putting this on the GPU QoS would bill an H200 to run sha256sum.
#
#SBATCH --job-name=wam-t041-register
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t041-register.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

CORPUS=${CORPUS_DIR:-${PROJ}/data/cosmos-g1-embodiment}
ENV_OUT=${PROJ}/cosmos_env.sh

echo "=== registering ${CORPUS}"
[[ -d "${CORPUS}" ]] || {
  echo "FATAL: ${CORPUS} does not exist. Ship it first, from the workstation:"
  echo "       ./cluster/discoverer/sync.sh --corpus"
  exit 1; }

# ---- structure -------------------------------------------------------------------------------
for f in manifest.json MANIFEST_SHA256 \
         train/video_dataset_file.jsonl val/video_dataset_file.jsonl \
         train/decode_report.json val/decode_report.json; do
  [[ -f "${CORPUS}/${f}" ]] || { echo "FATAL: ${CORPUS}/${f} missing."; exit 1; }
done
for d in train/videos val/videos; do
  [[ -d "${CORPUS}/${d}" ]] || { echo "FATAL: ${CORPUS}/${d} missing."; exit 1; }
done

# ---- the manifest is the provenance record (AC-04): prove it survived the wire ----------------
RECORDED=$(cat "${CORPUS}/MANIFEST_SHA256")
ACTUAL=$(sha256sum "${CORPUS}/manifest.json" | cut -d' ' -f1)
echo "=== manifest sha256 recorded ${RECORDED}"
echo "===                   actual ${ACTUAL}"
[[ "${RECORDED}" == "${ACTUAL}" ]] || {
  echo "FATAL: manifest.json does not match MANIFEST_SHA256. The corpus was truncated or"
  echo "       modified in transit. Re-run sync.sh --corpus; do not train on this."
  exit 1; }

# ---- counts, decode gate, and that every clip the jsonl names is actually here ----------------
# The jsonl stores vision_path RELATIVE to its own directory, so it is portable — but only if the
# videos came across too. Checking every path is a few thousand stats and costs nothing here;
# discovering it inside the trainer costs an 8-GPU allocation.
python3 - "${CORPUS}" <<'PY'
import json, pathlib, sys
corpus = pathlib.Path(sys.argv[1])
manifest = json.loads((corpus / "manifest.json").read_text())
counts = manifest["counts"]
bad = 0
for split in ("train", "val"):
    report = json.loads((corpus / split / "decode_report.json").read_text())
    if report["failed"]:
        sys.exit(f"FATAL: {split} decode_report has {report['failed']} failures. "
                 "This corpus never passed its own gate.")
    jsonl = corpus / split / "video_dataset_file.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if len(rows) != counts[split]:
        sys.exit(f"FATAL: {split} manifest promises {counts[split]} clips, jsonl has {len(rows)}. "
                 "The captioning run and the manifest disagree.")
    missing = []
    for r in rows:
        vp = r.get("vision_path") or ""
        if "://" in vp:
            continue
        if not (jsonl.parent / vp).is_file():
            missing.append(vp)
    if missing:
        bad += len(missing)
        print(f"  {split}: {len(missing)} missing videos, first 5: {missing[:5]}")
    print(f"  {split}: {len(rows)} clips, decode gate {report['checked']} checked / 0 failed, "
          f"{len(rows) - len(missing)} videos present")
if bad:
    sys.exit(f"FATAL: {bad} clips in the jsonl have no video on this filesystem. The transfer is "
             "incomplete — re-run sync.sh --corpus (rsync --partial resumes it).")
print("corpus verified")
PY

# ---- register ---------------------------------------------------------------------------------
# grep -q first, same reason as 92: this job can run more than once, and a duplicate export would
# still work while hiding a real change to the corpus path.
if ! grep -q "^export DATASET_PATH=${CORPUS}$" "${ENV_OUT}" 2>/dev/null; then
  cat >> "${ENV_OUT}" <<EOF
export DATASET_PATH=${CORPUS}
EOF
  echo "=== appended DATASET_PATH to ${ENV_OUT}"
else
  echo "=== DATASET_PATH already registered"
fi
if ! grep -q "^export CORPUS_MANIFEST_SHA256=${ACTUAL}$" "${ENV_OUT}" 2>/dev/null; then
  # Drop any stale value first: two CORPUS_MANIFEST_SHA256 lines would leave the run metadata
  # claiming whichever one happened to be sourced last, which is exactly the provenance AC-04
  # is supposed to make unambiguous.
  sed -i '/^export CORPUS_MANIFEST_SHA256=/d' "${ENV_OUT}"
  cat >> "${ENV_OUT}" <<EOF
export CORPUS_MANIFEST_SHA256=${ACTUAL}
EOF
  echo "=== appended CORPUS_MANIFEST_SHA256 to ${ENV_OUT}"
else
  echo "=== CORPUS_MANIFEST_SHA256 already registered"
fi

echo
echo "CORPUS REGISTERED -> ${CORPUS}"
echo "next: T041_FREEZE_LIFTED=\"<reason>\" PROBE=1 sbatch 94_train_t041_cosmos_super.sbatch"
