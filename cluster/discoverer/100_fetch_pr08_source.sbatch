#!/bin/bash
# Step 17 — T-040 / PR-08 §8 item 3: build ${PROJ}/data/pr08-apple-640x480, the corpus 97 restyles.
#
#   sbatch 100_fetch_pr08_source.sbatch
#
# THE NUMBER IS OUT OF ORDER AND THE STEP IS NOT. This runs BEFORE 97, and before 98/99 if you
# like — it shares nothing with them. The filename keeps climbing because the numbers are a
# submission log, not a dependency graph; the ordering that matters is stated here and in each
# job's epilogue. For T-040 the whole chain is:
#
#     100 (this: the source corpus)  ->  98 (framework)  ->  99 (weights)  ->  97 TIMING=1
#
# WHY IT EXISTS. 97's header says, in as many words: *"SOURCE ... Nothing writes it today —
# workstation/10_fetch_corpus.sh and scripts/convert_lerobot_g1.py produce the 120x160 tree PR-08
# §3 forbids."* That sentence had been true since 97 was written, which means the one measurement
# PR-08 §1 licenses unconditionally — timing one episode on an H200 — had no input to run on. Every
# other §8 item could have closed and the timing run would still have died at its manifest guard.
#
# IT DOES NOT CONVERT ANYTHING. nvidia/GR00T-N1.7-AppleToPlate is ALREADY 640x480 (meta/info.json:
# observation.images.ego_view has shape [480, 640, 3], codec av1, 30 fps, 402 episodes, 171625
# frames). So this job downloads, CHECKS, and writes a manifest. It never rescales or re-encodes:
# a lossy generation between the recorded pixels and the restyle is exactly what §3 refuses, and
# the premise of Transfer2.5-as-augmentation is that the pixels under the carried-over actions are
# the real ones.
#
# THE CHECK WORTH THE JOB'S EXISTENCE is per-episode frame count against meta/episodes.jsonl's
# declared length. Labels are carried over BY INDEX. A video that holds a different number of
# frames than its label column declares pairs every later label with a different instant than its
# pixels -- silently, with no decode error, and with the restyle looking perfect. No later gate
# catches it: G0a checks the actions are unchanged (they are) and G0b checks geometry within a clip
# (it is fine), and neither compares the two lengths. Measured on the workstation copy 2026-08-19:
# all 402 agree, and the sum is 171625, matching info.json's own total.
#
# NO GPU, SO NO GPU-HOURS -- same reasoning as 98 and 99. PR-08 §8 item 3 says no budget line
# exists until the throughput measurement does, so nothing that PRECEDES that measurement may spend
# from it. ~0.94 GB over the wire, ~0.87 GB of it mp4.
#
# THE REVISION IS PINNED, for the reason 91 and 99 state: a corpus that cannot name itself is not
# a corpus a result can cite. The dataset is NOT gated (checked on the HF API 2026-08-19,
# "gated": false), so unlike 99 there is no licence probe here -- if that ever changes, copy 99's.
#
#SBATCH --job-name=wam-pr08-source
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/pr08-source.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${WAM:-${PROJ}/wam}

SOURCE_REPO_ID=${SOURCE_REPO_ID:-nvidia/GR00T-N1.7-AppleToPlate}
SOURCE_REVISION=${SOURCE_REVISION:-d89c126a713c6632432a607c12661546ff4d6ea9}
SOURCE_OUT=${SOURCE_OUT:-${PROJ}/data/pr08-apple-640x480}
# copy, not symlink: this tree is the registered SOURCE of an experiment, and a symlink farm into
# an HF cache is a foundation someone can clear without noticing what depended on it. 0.87 GB is a
# cheap price for a corpus that stands on its own. Set SOURCE_MATERIALIZE=symlink to trade back.
SOURCE_MATERIALIZE=${SOURCE_MATERIALIZE:-copy}

# The same refusal 99 makes, for the same reason. `main` is not a revision.
case "${SOURCE_REVISION}" in
  main|master|HEAD|latest|"")
    echo "FATAL: SOURCE_REVISION=${SOURCE_REVISION:-<empty>} is a moving pointer, not a revision."
    echo "       A corpus that cannot name the snapshot it came from cannot be cited by a result."
    exit 1 ;;
esac
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FATAL: SOURCE_REVISION=${SOURCE_REVISION} is not a 40-hex commit sha."; exit 1; }

echo "=== repo     ${SOURCE_REPO_ID}"
echo "=== revision ${SOURCE_REVISION}"
echo "=== out      ${SOURCE_OUT}  (${SOURCE_MATERIALIZE})"

# -- ffprobe: reused from 90's build, never rebuilt ------------------------------------------------
export FFMPEG_PREFIX=${FFMPEG_PREFIX:-${PROJ}/tools/ffmpeg}
FFPROBE=${FFPROBE:-${FFMPEG_PREFIX}/bin/ffprobe}
if [[ ! -x "${FFPROBE}" ]]; then
  echo "FATAL: no ffprobe at ${FFPROBE}. 90_build_cosmos_env.sbatch builds ffmpeg; run it first,"
  echo "       or set FFPROBE. This job does not build a second copy."
  exit 1
fi
"${FFPROBE}" -version | head -1

BUILDER=${BUILDER:-${WAM}/scripts/build_pr08_source.py}
[[ -f "${BUILDER}" ]] || { echo "FATAL: ${BUILDER} missing — re-sync the repo."; exit 1; }

# -- the download ----------------------------------------------------------------------------------
# Whole repo: 0.94 GB, of which 0.87 GB is the 402 mp4s. There is no useful --include here because
# the builder needs meta/ AND videos/, and data/ (the action parquets) is what the ASSEMBLER carries
# through byte-identically later. Taking the lot once is simpler than discovering the gap in 97.
# `uvx hf@latest download` is 99's idiom: one tool, pinned by uv, resolved the same way in both
# jobs. The path is captured rather than reconstructed from ${HF_HOME} and a naming convention
# that is upstream's to change.
#
# --quiet IS LOAD-BEARING, not noise control. hf 1.28's default output is decorated:
#
#     ✓ Downloaded
#       path: /valhalla/.../snapshots/d89c126a...
#
# so `tail -1` yields "  path: /valhalla/..." — label, leading spaces and all — and the directory
# check below then rejects a download that in fact succeeded (measured, job 189134). `--quiet` is
# documented as "one ID per line" and emits the bare path, which is a contract to depend on rather
# than a banner to parse. Note 99 does NOT need this: it passes --local-dir and never reads stdout.
# -- the token ------------------------------------------------------------------------------------
# REQUIRED, EVEN THOUGH THIS DATASET IS NOT GATED. Anonymity is not the same as access: an
# unauthenticated client gets the anonymous rate limit, and this is an 813-file fetch. Measured,
# job 189023: the download ran happily to 80% and then died on
#
#     ConnectionError: Network error: Request error: HTTP status client error
#     (429 Too Many Requests), domain: .../xet-read-token/d89c126a...
#
# six minutes in, after a queue wait far longer than that. hf itself warns about this on the first
# line it prints ("You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to
# enable higher rate limits") and then proceeds anyway, which is the worst of both: the run looks
# fine until it is not. So this refuses up front instead. A READ token is enough; nothing here
# writes to the Hub.
#
# Search order is 99's, unchanged, so the two jobs cannot disagree about where a token lives.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -f "${f}" ]] && { HF_TOKEN=$(<"${f}"); break; }
  done
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: no Hugging Face token, and an anonymous fetch of 813 files hits the rate limit"
  echo "       long before it finishes (measured: 429 at 80%, job 189023). Export HF_TOKEN, or"
  echo "       put a READ token at \${HF_HOME}/token or ~/.huggingface/token. This dataset is"
  echo "       NOT gated -- the token buys throughput, not permission."
  exit 1
fi
export HF_TOKEN
echo "=== HF token present (${#HF_TOKEN} chars, not echoed)"

echo "=== downloading into ${HF_HOME}"
SNAPSHOT=$(uvx hf@latest download "${SOURCE_REPO_ID}" \
    --repo-type dataset --revision "${SOURCE_REVISION}" --quiet | tail -1)
[[ -d "${SNAPSHOT}" ]] || {
  echo "FATAL: hf download did not print a directory (got: ${SNAPSHOT:-<empty>})."
  echo "       Not guessing the cache layout -- that is upstream's to change."
  exit 1; }
echo "=== snapshot ${SNAPSHOT}"

# -- the build -------------------------------------------------------------------------------------
MAT_FLAG=()
[[ "${SOURCE_MATERIALIZE}" == "copy" ]] && MAT_FLAG=(--copy)

python3 "${BUILDER}" \
    --snapshot "${SNAPSHOT}" \
    --out "${SOURCE_OUT}" \
    --ffprobe "${FFPROBE}" \
    --repo-id "${SOURCE_REPO_ID}" \
    --revision "${SOURCE_REVISION}" \
    "${MAT_FLAG[@]}"

echo
echo "=== manifest:"
python3 - "${SOURCE_OUT}/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
eps = m["episodes"]
print(f"    resolution {m['resolution']}  fps {m['fps']}  codecs {m['source']['codecs']}")
print(f"    {len(eps)} episodes, {sum(e['frames'] for e in eps)} frames")
print(f"    from {m['source']['repo_id']}@{m['source']['revision']}")
PY
cat "${SOURCE_OUT}/manifest.json.sha256"

echo
echo "=== NEXT, in order:"
echo "===   sbatch 98_build_transfer25_env.sbatch      (framework, pinned)"
echo "===   sbatch 99_stage_transfer25_weights.sbatch  (weights, pinned; needs the licence gate)"
echo "===   the timing run, whose exact invocation is in 97_transfer25_restyle.sbatch's own"
echo "===   header — copied nowhere else on purpose, so the two cannot drift apart."
echo "===   That measurement is licensed by PR-08 §1. GENERATION is not, and 97 keeps its"
echo "===   own guards regardless of what this job wrote."
