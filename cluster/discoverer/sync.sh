#!/usr/bin/env bash
# Push repo + dataset + job scripts to Discoverer+. Runs on the Mac, not on the cluster.
#
#   ./cluster/discoverer/sync.sh          # repo + jobs + caches.sh
#   ./cluster/discoverer/sync.sh --data   # also the 81 MB converted dataset
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

echo "done. Next:  ssh ${HOST}  then  cd ${PROJ}/wam/cluster/discoverer && sbatch 10_build_env.sbatch"
