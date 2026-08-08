#!/bin/bash
# Step 2 of the workstation pipeline — lay the corpus out as decodable clips, then PROVE they decode.
#
#   T041_FREEZE_LIFTED="<reason>" workstation/20_prepare_corpus.sh
#   ENCODER=h264_nvenc JOBS=4 T041_FREEZE_LIFTED="..." workstation/20_prepare_corpus.sh
#
# TRANSCODING IS NOT OPTIONAL HERE. Every one of the 14 sources is AV1 — LeRobot's default encoder
# is libsvtav1 and nobody overrode it. vLLM's video path is OpenCV-only; on the cluster it opened
# each AV1 file, read the container header correctly (377 frames, 30 fps, 640x480) and then failed
# every cap.grab(). The model received array([], shape=(0, 480, 640, 3)) 372 times, the captioner
# reported "0/372 videos were successfully captioned", and nothing exited non-zero. H.264 yuv420p
# is the format nothing argues with.
#
# It is also the only way to read v3.0 at all: 13 of the 14 sources concatenate their episodes into
# a handful of mp4s, so a clip is a [from_timestamp, to_timestamp) window that has to be cut out.
#
# THE VERIFY STEP IS THE POINT. It runs with the CAPTIONER'S interpreter, not this shell's python,
# because the question is not "is this file valid" — ffprobe answered yes to the AV1 corpus all
# along — but "can the decoder that will actually read it get pixels out". Those turned out to be
# different questions.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_freeze_lifted
need ffmpeg "brew install ffmpeg / apt install ffmpeg"
need ffprobe ""

ENCODER=${ENCODER:-libx264}
JOBS=${JOBS:-4}
VAL_EPISODES=${VAL_EPISODES:-30}
SEED=${SEED:-0}

# SMOKE_SOURCES replaces the corpus with a named subset to rehearse the pipeline. It REPLACES
# rather than filters, and forces a separate output tree, so a rehearsal can never be mistaken for
# — or overwrite — the pre-registered corpus.
SMOKE_SOURCES=${SMOKE_SOURCES:-}

read_corpus
if [[ -n ${SMOKE_SOURCES} ]]; then
  read -r -a wanted <<< "${SMOKE_SOURCES}"
  keep_repos=(); keep_cams=()
  for w in "${wanted[@]}"; do
    found=0
    for i in "${!REPOS[@]}"; do
      if [[ ${REPOS[$i]} == "${w}" || $(basename "${REPOS[$i]}") == "${w}" ]]; then
        keep_repos+=("${REPOS[$i]}"); keep_cams+=("${CAMS[$i]}"); found=1
      fi
    done
    # Selected BY NAME out of the real list, so a rehearsal source keeps its real camera key
    # instead of being paired with whatever happened to be at the same index.
    [[ ${found} == 1 ]] || { echo "FATAL: '${w}' is not in ${CORPUS_TSV}"; exit 1; }
  done
  REPOS=("${keep_repos[@]}"); CAMS=("${keep_cams[@]}")
  CORPUS=${WORK}/cosmos-smoke
  echo "=== REHEARSAL: ${#REPOS[@]} source(s): ${REPOS[*]}"
  echo "=== REHEARSAL: CORPUS=${CORPUS} (the registered corpus is untouched)"
fi

ARGS=()
for i in "${!REPOS[@]}"; do
  dest=${RAW}/$(basename "${REPOS[$i]}")
  [[ -f "${dest}/.download-complete" ]] || {
    echo "FATAL: ${dest} was never fetched to completion — run workstation/10_fetch_corpus.sh"; exit 1; }
  ARGS+=(--source "${dest}" --camera-key "${CAMS[$i]}")
done

echo "=== transcoding to H.264 (${ENCODER}, ${JOBS} parallel) -> ${CORPUS}"
python3 "${WAM_ROOT}/scripts/prepare_cosmos_corpus.py" "${ARGS[@]}" \
    --out "${CORPUS}" --val-episodes "${VAL_EPISODES}" --seed "${SEED}" \
    --mode transcode --encoder "${ENCODER}" --jobs "${JOBS}"

if [[ -n ${SMOKE_SOURCES} ]]; then
  cat > "${CORPUS}/SMOKE" <<EOF
Rehearsal corpus — NOT the PR-09 §2 pre-registered corpus.
sources: ${REPOS[*]}
No verdict under T041_RULE_V1 may be issued from anything trained on this.
EOF
fi

# -- the gate ------------------------------------------------------------------------------------
# Not a smoke test on a sample: every clip, because a corpus is only as readable as its worst file
# and a partial pass here buys nothing that the captioner would not discover more expensively.
CAPTIONER_PYTHON=${CAPTIONER_PYTHON:-}
if [[ -z ${CAPTIONER_PYTHON} && -f ${ENV_FILE} ]]; then
  source "${ENV_FILE}"
  CAPTIONER_PYTHON=${FRAMEWORK:+${FRAMEWORK}/.venv/bin/python}
fi
if [[ -z ${CAPTIONER_PYTHON} || ! -x ${CAPTIONER_PYTHON} ]]; then
  echo
  echo "WARNING: the captioner's interpreter was not found, so the clips are UNVERIFIED."
  echo "         Set CAPTIONER_PYTHON=/path/to/cosmos-framework/.venv/bin/python and re-run, or"
  echo "         run scripts/verify_clip_decode.py with it by hand before captioning. Verifying"
  echo "         with a different cv2 build proves nothing about the one that will read these."
  exit 1
fi

for split in train val; do
  echo "=== verifying ${split} decodes with the captioner's own cv2"
  "${CAPTIONER_PYTHON}" "${WAM_ROOT}/scripts/verify_clip_decode.py" \
      "${CORPUS}/${split}/videos" --jobs "${JOBS}" \
      --report "${CORPUS}/${split}/decode_report.json"
done

echo
echo "CORPUS OK -> ${CORPUS}  sha256 $(cat "${CORPUS}/MANIFEST_SHA256")"
du -sh "${CORPUS}" 2>/dev/null || true
echo "next: workstation/30_caption_corpus.sh"
