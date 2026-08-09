#!/bin/bash
# Step 3 of the workstation pipeline — caption the corpus and build the SFT jsonl.
#
#   T041_FREEZE_LIFTED="<reason>" workstation/30_caption_corpus.sh
#
# THE CAPTIONER IS NVIDIA'S, NOT OURS. cosmos_framework.scripts.caption_from_video drives a vLLM
# server (structured-JSON scene analysis, then a dense narrative rewrite) and captions_to_sft_jsonl
# turns the result into video_dataset_file.jsonl. T-041 listed captioning as "its own pipeline,
# plausibly Cosmos-Reason2, the largest unpriced item"; the answer came from reading the framework
# rather than building anything.
#
# WHY THIS RUNS HERE AND NOT ON THE CLUSTER. Captions are portable text. Producing them locally
# costs one workstation GPU for an evening instead of a Slurm reservation, keeps the whole
# debug loop off the queue, and means what ships to Discoverer+ is a finished dataset rather than
# an unfinished one plus the hope that the next job works.
#
# The corpus must already have passed workstation/20_prepare_corpus.sh's decode gate. Captioning an
# unverified corpus is precisely what job 186357 did: 372 clips, 372 requests, 0 captions, no error.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_freeze_lifted
[[ -f ${ENV_FILE} ]] || { echo "FATAL: ${ENV_FILE} missing — run workstation/00_setup_env.sh"; exit 1; }
source "${ENV_FILE}"

REASONER=${REASONER:-Qwen/Qwen3-VL-8B-Instruct-FP8}
VLLM_VERSION=${VLLM_VERSION:-0.19.0}
VLLM_PYTHON=${VLLM_PYTHON:-3.12}
# Qwen3-VL declares max_seq_len=262144, which needs 36 GiB of KV cache. No 32 GB card has that, so
# vLLM refuses to start at the default — this is a property of the model, not of a busy GPU.
# 32768 is sized from the corpus: the longest clip is 60.6 s and p95 is 45.7 s (decode_report.json),
# which at Qwen3-VL's video sampling is ~12k tokens for the worst case. That leaves ~2.5x headroom
# and keeps enough KV free for MAX_WORKERS concurrent requests. If a clip ever does overflow, the
# request fails loudly and the manifest-vs-jsonl count check at the end of this script catches it.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
PORT=${PORT:-8000}
MAX_WORKERS=${MAX_WORKERS:-16}
# -1 = all frames, matching the recipe's window setting. It also sets the frame floor the converter
# applies, and it must equal what prepare_cosmos_corpus.py used.
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:--1}
SMOKE=${SMOKE:-0}
[[ ${SMOKE} == 1 ]] && CORPUS=${WORK}/cosmos-smoke

for d in "${CORPUS}/train/videos" "${CORPUS}/val/videos"; do
  [[ -d "${d}" ]] || { echo "FATAL: ${d} missing — run workstation/20_prepare_corpus.sh first."; exit 1; }
done
[[ -f "${CORPUS}/manifest.json" ]] || { echo "FATAL: ${CORPUS}/manifest.json missing — that file is the provenance record (AC-04), not decoration."; exit 1; }

# Refuse to caption what was never verified. The report is written by the decode gate; its absence
# means the gate did not run, and its contents mean it did not pass.
for split in train val; do
  report=${CORPUS}/${split}/decode_report.json
  [[ -f ${report} ]] || { echo "FATAL: ${report} missing — the decode gate never ran on ${split}."; exit 1; }
  python3 - "${report}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if r["failed"]:
    sys.exit(f"FATAL: {r['failed']}/{r['checked']} clips decoded no frames under cv2 "
             f"{r['cv2']}. Captioning these produces empty captions and no error.")
print(f"decode gate: {r['checked']} clips OK under cv2 {r['cv2']}")
PY
done

echo "=== corpus ${CORPUS}  manifest sha256 $(cat "${CORPUS}/MANIFEST_SHA256")"
echo "=== train $(find "${CORPUS}/train/videos" -name '*.mp4' | wc -l) clips, val $(find "${CORPUS}/val/videos" -name '*.mp4' | wc -l) clips"

# -- vLLM server -----------------------------------------------------------------------------
# --allowed-local-media-path is required: caption_from_video passes file:// URLs, so the server has
# to be allowed to read from the filesystem. Without it every request fails identically and the run
# produces zero captions — the same symptom as an unreadable codec, from a different cause.
LOGDIR=${WORK}/logs
mkdir -p "${LOGDIR}"
SERVER_LOG=${LOGDIR}/vllm.$(date +%Y%m%d-%H%M%S).log
echo "=== starting ${REASONER} on :${PORT} (log ${SERVER_LOG})"
# --python and --managed-python are both load-bearing, and they fix two DIFFERENT failures.
#
# --python: uvx otherwise picks the newest interpreter it can find (3.14 here, via anaconda), and
# vllm pulls numba -> llvmlite, which ships no 3.14 wheel. uv falls back to building it from source
# and dies in setuptools with "spawn() got an unexpected keyword argument 'dry_run'" — a traceback
# that names llvmlite and distutils and never mentions the interpreter.
#
# --managed-python: with only --python, uv resolves 3.12 to /usr/bin/python3.12, which has no dev
# headers. vLLM then downloads the model, loads it, and dies 90 seconds later during the KV-cache
# profile run when Triton JIT-compiles cuda_utils.c: "Python.h: No such file or directory". The
# uv-managed build ships its own headers. Failing this late looks like a GPU/model problem and is
# not one.
uvx --managed-python --python "${VLLM_PYTHON}" "vllm@${VLLM_VERSION}" serve "${REASONER}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --allowed-local-media-path / > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== waiting for startup (a couple of minutes is normal)"
for _ in $(seq 1 120); do
  grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null && break
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FATAL: vLLM died during startup — tail of ${SERVER_LOG}:"; tail -40 "${SERVER_LOG}"; exit 1
  fi
  sleep 10
done
grep -q "Application startup complete" "${SERVER_LOG}" || {
  echo "FATAL: vLLM not ready after 20 min."; tail -40 "${SERVER_LOG}"; exit 1; }
echo "=== server up"

# -- caption + convert -----------------------------------------------------------------------
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

  # If the converter kept fewer than the manifest promised, the corpus and the run disagree — and
  # that disagreement is exactly what silently shrinks a training set. Fail here, not in a results
  # table three weeks later.
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

cat >> "${ENV_FILE}" <<EOF
export DATASET_PATH=${CORPUS}
export CORPUS_MANIFEST_SHA256=$(cat "${CORPUS}/MANIFEST_SHA256")
EOF
echo
echo "CAPTIONED OK -> ${CORPUS}"
echo "next: ship it to the cluster, then PROBE=1 sbatch 94_train_t041_cosmos_super.sbatch there"
