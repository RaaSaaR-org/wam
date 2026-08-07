#!/bin/bash
# Step 17 — T-041 / PR-09: caption the G1 corpus and build the SFT jsonl.
#
#   T041_FREEZE_LIFTED="<reason>" CORPUS=/path/to/cosmos-g1-embodiment \
#       sbatch 93_caption_corpus.sbatch
#
# THE CAPTIONER IS NVIDIA'S, NOT OURS. T-041 recorded captioning as "its own pipeline, plausibly
# Cosmos-Reason2, the largest unpriced item — cost this before anything else". It is a shipped
# script: cosmos_framework.scripts.caption_from_video drives a vLLM server (two phases —
# structured-JSON scene analysis, then a dense narrative rewrite) and captions_to_sft_jsonl turns
# the result into video_dataset_file.jsonl. That item was answered by reading the framework, not
# by building anything. Recorded here because "we built a captioner" would have been a real cost
# and a real place for our own bug.
#
# The corpus tree comes from scripts/prepare_cosmos_corpus.py, which has already applied the
# loader's own silent filters (>61 s, <61 frames), so the counts below are the counts that train.
#
# PR-09 §7 ceiling: 6 GPU-h. One GPU: the reasoner is 8B FP8 and the job is throughput-bound on
# video decode, not on the model.
#
#SBATCH --job-name=wam-cosmos-caption
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/cosmos-caption.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
source "${PROJ}/cosmos_env.sh"

: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8
export PATH=${UV_INSTALL_DIR}:${FFMPEG_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
nvidia-smi -L

CORPUS=${CORPUS:-${PROJ}/data/cosmos-g1-embodiment}
REASONER=${REASONER:-Qwen/Qwen3-VL-8B-Instruct-FP8}
VLLM_VERSION=${VLLM_VERSION:-0.19.0}
PORT=${PORT:-8000}
MAX_WORKERS=${MAX_WORKERS:-16}
# -1 = all frames, matching the recipe's window setting. It also sets the frame floor the
# converter applies, and it must equal what prepare_cosmos_corpus.py used.
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:--1}

for d in "${CORPUS}/train/videos" "${CORPUS}/val/videos"; do
  [[ -d "${d}" ]] || { echo "FATAL: ${d} missing — run scripts/prepare_cosmos_corpus.py first."; exit 1; }
done
[[ -f "${CORPUS}/manifest.json" ]] || { echo "FATAL: ${CORPUS}/manifest.json missing — that file is the provenance record (AC-04), not decoration."; exit 1; }
echo "=== corpus ${CORPUS}  manifest sha256 $(cat "${CORPUS}/MANIFEST_SHA256")"
echo "=== train $(find "${CORPUS}/train/videos" -name '*.mp4' | wc -l) clips, val $(find "${CORPUS}/val/videos" -name '*.mp4' | wc -l) clips"

# -- vLLM server ---------------------------------------------------------------------------------
# --allowed-local-media-path is required: caption_from_video passes file:// URLs, so the server
# has to be allowed to read from the filesystem. Without it every request fails identically and
# the job burns its whole walltime producing zero captions.
LOGDIR=${PROJ}/logs
mkdir -p "${LOGDIR}"
SERVER_LOG=${LOGDIR}/vllm.${SLURM_JOB_ID}.log
echo "=== starting ${REASONER} on :${PORT} (log ${SERVER_LOG})"
uvx --with nvidia-cuda-runtime-cu12 "vllm@${VLLM_VERSION}" serve "${REASONER}" \
    --tensor-parallel-size 1 \
    --port "${PORT}" \
    --allowed-local-media-path / > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== waiting for startup (a couple of minutes is normal)"
for _ in $(seq 1 120); do
  if grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FATAL: vLLM died during startup — tail of ${SERVER_LOG}:"; tail -40 "${SERVER_LOG}"; exit 1
  fi
  sleep 10
done
grep -q "Application startup complete" "${SERVER_LOG}" || {
  echo "FATAL: vLLM not ready after 20 min."; tail -40 "${SERVER_LOG}"; exit 1; }
echo "=== server up"

# -- caption + convert ---------------------------------------------------------------------------
source "${FRAMEWORK}/.venv/bin/activate"
cd "${FRAMEWORK}"
for SPLIT in train val; do
  VIDEOS=${CORPUS}/${SPLIT}/videos
  CAPTIONS=${CORPUS}/${SPLIT}/captions
  JSONL=${CORPUS}/${SPLIT}/video_dataset_file.jsonl
  mkdir -p "${CAPTIONS}"
  echo "=== captioning ${SPLIT}"
  python -m cosmos_framework.scripts.caption_from_video \
      --video "${VIDEOS}" -o "${CAPTIONS}" \
      --server "http://localhost:${PORT}/v1" \
      --max_workers "${MAX_WORKERS}"

  echo "=== building ${JSONL}"
  python -m cosmos_framework.scripts.captions_to_sft_jsonl \
      --captions-dir "${CAPTIONS}" \
      --videos-dir "${VIDEOS}" \
      --num-video-frames "${NUM_VIDEO_FRAMES}" \
      -o "${JSONL}"

  # The converter writes a sibling summary with the kept count and per-reason drops. If it kept
  # fewer than the manifest promised, the corpus and the run disagree — and that disagreement is
  # exactly what silently shrinks a training set. Fail here, not in a results table.
  SUMMARY=${JSONL}.summary.json
  [[ -f "${SUMMARY}" ]] && cat "${SUMMARY}"
  python - "${CORPUS}/manifest.json" "${SPLIT}" "${JSONL}" <<'PY'
import json, sys
manifest, split, jsonl = sys.argv[1], sys.argv[2], sys.argv[3]
promised = json.load(open(manifest))["counts"][split]
kept = sum(1 for line in open(jsonl) if line.strip())
print(f"{split}: manifest promised {promised}, jsonl has {kept}")
if kept != promised:
    sys.exit(f"FATAL: {split} lost {promised - kept} clips between the manifest and the jsonl. "
             "prepare_cosmos_corpus.py mirrors the loader's filters, so a difference here means "
             "captioning dropped clips (check the summary's drop reasons) — not a rounding issue.")
PY
done

# What the trainer will consume. The framework resolves <DATASET_PATH>/train/video_dataset_file.jsonl.
cat >> "${PROJ}/cosmos_env.sh" <<EOF
export DATASET_PATH=${CORPUS}
export CORPUS_MANIFEST_SHA256=$(cat "${CORPUS}/MANIFEST_SHA256")
EOF
echo "CAPTIONED OK -> ${CORPUS}"
echo "next: probe first —  PROBE=1 sbatch 94_train_t041_cosmos_super.sbatch"
