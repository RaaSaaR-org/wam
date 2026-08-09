#!/usr/bin/env bash
# Move code and artifacts between the Mac and Discoverer+. Runs on the Mac, not on the cluster.
#
#   ./cluster/discoverer/sync.sh              # push repo + jobs + caches.sh
#   ./cluster/discoverer/sync.sh --data       # also push the 81 MB converted dataset
#   ./cluster/discoverer/sync.sh --corpus     # push the 14 GB captioned T-041 corpus (then 92b)
#   ./cluster/discoverer/sync.sh --pull       # pull every run's artifacts back
#   ./cluster/discoverer/sync.sh --pull t16-lora-seed0   # ...or just one run
#
# Push and pull are separate directions on purpose. The push excludes runs/ so a stale local
# copy can never overwrite what the GPU just produced; the pull excludes checkpoints/ so a
# routine artifact fetch can never drag 5B weights over the wire. Every eval job's epilogue
# tells the operator to "copy back with sync.sh" — --pull is what makes that sentence true.
#
# Needs the key in the agent first:  ssh-add ~/.ssh/id_ed25519_eu_ai_hub
set -euo pipefail

HOST=ffromm@login-plus.discoverer.bg
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# No --append/--append-verify: every source here is a file that gets *edited* in place, and
# --append skips a changed file whose size happens to be unchanged. It also does not exist in
# the openrsync that ships as /usr/bin/rsync on macOS. --partial alone gives the resume, and
# the delta algorithm gives the verification. -l because HF-style caches are symlink farms.
RSYNC=(rsync -e ssh -rtl --progress --partial)

if [[ "${1:-}" == "--pull" ]]; then
  # Artifacts live at ${PROJ}/runs/<RUN_ID>/, NOT ${PROJ}/wam/runs/ — the job scripts set
  # OUT=${PROJ}/runs/${RUN_ID} so results survive a repo re-sync. Slurm logs come too: the
  # printed verdict exists nowhere else, and reading it straight off the login node is the
  # only copy until it lands here.
  RUN="${2:-}"
  mkdir -p "${ROOT}/runs" "${ROOT}/runs/_slurm_logs"
  echo "==> runs${RUN:+/${RUN}} <- ${PROJ}/runs"
  # WHITELIST, not blacklist. ${PROJ}/runs is 140 GB and most of it is weights: 5B safetensors
  # shards sitting at the top of a run dir, plus checkpoint-NNNN/ dirs that no single --exclude
  # name catches. Naming what to *skip* is unbounded and gets it wrong the first time a run
  # invents a new layout; naming what to *take* is four extensions that cover every scoring
  # artifact we have ever read (predictions.jsonl, bench/e1/timing.json, the .md reports, DONE).
  # `--include '*/'` lets rsync walk into subdirs; the trailing `--exclude '*'` drops the rest.
  # --max-size is the belt: even a .json can't be a model, so anything above 256M is not ours.
  "${RSYNC[@]}" --max-size=256m \
    --include '*/' \
    --include '*.json' --include '*.jsonl' --include '*.md' --include '*.txt' \
    --include 'DONE' --include 'GIT_COMMIT' \
    --exclude '*' \
    "${HOST}:${PROJ}/runs/${RUN:+${RUN}/}" "${ROOT}/runs/${RUN:+${RUN}/}"
  echo "==> runs/_slurm_logs <- ${PROJ}/logs"
  "${RSYNC[@]}" "${HOST}:${PROJ}/logs/" "${ROOT}/runs/_slurm_logs/"
  echo "done. Re-score on CPU forever:  python scripts/run_bench.py runs/<run>/<arm> --compare"
  exit 0
fi

echo "==> repo -> ${PROJ}/wam"
# The excludes mirror .gitignore: everything gitignored is either machine-local (.venv,
# caches), fetched on the cluster instead (data/raw = 966 MB of raw LeRobot snapshot,
# assets/ = the MuJoCo model, which no cluster job uses), or shipped separately below.
#
# The LEADING SLASHES are load-bearing. An rsync pattern without one matches the last path
# component at ANY depth, so a bare `--exclude 'data'` silently drops src/wam/data/ as well —
# episode.py, validation.py, the lot. There is no --delete either, so the cluster keeps a
# stale copy and the jobs die on `cannot import name frame_window_indices` AFTER the H200 is
# allocated. Anchored, each pattern means the one top-level directory it was written for.
"${RSYNC[@]}" \
  --exclude '.git' --exclude '/datasets' --exclude '/runs' --exclude '/data' \
  --exclude '/assets' --exclude '__pycache__' --exclude '.venv' --exclude '*.egg-info' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude '.mypy_cache' \
  "${ROOT}/" "${HOST}:${PROJ}/wam/"

# The cluster copy has no .git (excluded above), so read_git_commit() there returns "unknown"
# and the run metadata loses the code provenance AC-04 asks for. Stamp it at sync time instead.
# The -dirty suffix is the honest part: it says the tree had uncommitted edits, so the hash
# alone does not reproduce what ran.
COMMIT="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if ! git -C "${ROOT}" diff --quiet HEAD 2>/dev/null; then COMMIT="${COMMIT}-dirty"; fi
echo "==> GIT_COMMIT -> ${PROJ}/wam/GIT_COMMIT (${COMMIT})"
printf '%s\n' "${COMMIT}" | ssh "${HOST}" "cat > ${PROJ}/wam/GIT_COMMIT"

echo "==> caches.sh -> ${PROJ}/caches.sh"
"${RSYNC[@]}" "${ROOT}/cluster/discoverer/caches.sh" "${HOST}:${PROJ}/caches.sh"

if [[ "${1:-}" == "--data" ]]; then
  # rsync creates the last path component, not the ones above it, and --mkpath needs
  # rsync >=3.2.3 on both ends (macOS ships openrsync). One mkdir is file management on
  # the login node, which is permitted — unlike anything that computes.
  ssh "${HOST}" "mkdir -p ${PROJ}/data"
  echo "==> dataset -> ${PROJ}/data/gr00t-apple-full"
  "${RSYNC[@]}" "${ROOT}/datasets/gr00t-apple-full/" \
    "${HOST}:${PROJ}/data/gr00t-apple-full/"
fi

if [[ "${1:-}" == "--corpus" ]]; then
  # The T-041 corpus is built and captioned on the workstation (workstation/20,30), so nothing on
  # the cluster produces it any more — and nothing shipped it either. 94_train reads DATASET_PATH
  # out of cosmos_env.sh, whose only writer was 93_caption_corpus.sbatch, which is superseded.
  # Without this, 94 is allocated all 8 GPUs of dgx1 and then exits in under a second at its
  # DATASET_PATH guard. This is the missing half of that path; 92b_register_corpus.sbatch is the
  # other half and must run after it.
  CORPUS_SRC=${CORPUS_SRC:-${HOME}/wam-t041/cosmos-g1-embodiment}
  DEST=${PROJ}/data/cosmos-g1-embodiment
  [[ -d "${CORPUS_SRC}" ]] || { echo "FATAL: ${CORPUS_SRC} missing. Set CORPUS_SRC."; exit 1; }
  for f in manifest.json MANIFEST_SHA256 train/video_dataset_file.jsonl val/video_dataset_file.jsonl; do
    [[ -f "${CORPUS_SRC}/${f}" ]] || {
      echo "FATAL: ${CORPUS_SRC}/${f} missing — the corpus is not captioned yet."
      echo "       Run workstation/30_caption_corpus.sh to completion first."; exit 1; }
  done
  # Ship straight to the FINAL path. Never stage elsewhere under /valhalla and move it into place:
  # project IDs live on the inode and survive a rename, so an intra-/valhalla mv leaves the data
  # charged to the wrong project with nothing to show that it happened.
  ssh "${HOST}" "mkdir -p ${DEST}"
  echo "==> corpus -> ${DEST}  ($(du -sh "${CORPUS_SRC}" | cut -f1))"
  # vision_path in the jsonl is stored RELATIVE to the jsonl's own directory
  # (captions_to_sft_jsonl.py:_relativize_vision_path), so the tree relocates as-is — provided
  # the train/videos <-> train/video_dataset_file.jsonl layout is preserved. Ship the whole tree.
  "${RSYNC[@]}" "${CORPUS_SRC}/" "${HOST}:${DEST}/"
  echo
  echo "shipped. Now register it (free QoS, no GPU):"
  echo "  ssh ${HOST}"
  echo "  cd ${PROJ}/wam/cluster/discoverer && sbatch 92b_register_corpus.sbatch"
  exit 0
fi

echo "done. Next:  ssh ${HOST}  then  cd ${PROJ}/wam/cluster/discoverer && sbatch 10_build_env.sbatch"
