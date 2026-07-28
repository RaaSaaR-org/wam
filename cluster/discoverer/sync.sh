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
RSYNC=(rsync -e ssh -rtl --progress --partial --append-verify)

echo "==> repo -> ${PROJ}/wam"
"${RSYNC[@]}" \
  --exclude '.git' --exclude 'datasets' --exclude 'runs' --exclude '__pycache__' \
  --exclude '.venv' --exclude '*.egg-info' \
  "${ROOT}/" "${HOST}:${PROJ}/wam/"

echo "==> caches.sh -> ${PROJ}/caches.sh"
"${RSYNC[@]}" "${ROOT}/cluster/discoverer/caches.sh" "${HOST}:${PROJ}/caches.sh"

if [[ "${1:-}" == "--data" ]]; then
  echo "==> dataset -> ${PROJ}/data/gr00t-apple-full"
  "${RSYNC[@]}" "${ROOT}/datasets/gr00t-apple-full/" \
    "${HOST}:${PROJ}/data/gr00t-apple-full/"
fi

echo "done. Next:  ssh ${HOST}  then  cd ${PROJ}/wam/cluster/discoverer && sbatch 10_build_env.sbatch"
