#!/bin/bash
# Shared helpers for the workstation pipeline. Sourced, never executed.
#
# WHY THERE IS A WORKSTATION PIPELINE AT ALL. Every T-041 failure so far has been a data-preparation
# failure, not a training failure: the Hub rate-limiting a repo, 13 of 14 sources turning out to be
# LeRobot v3.0, and an AV1 corpus that vLLM's OpenCV opened and decoded zero frames from. Each cost
# hours of Slurm queue to learn something a laptop could have told us in seconds, and the last one
# burned a GPU hour to produce 372 empty caption files without erroring.
#
# So the split is: this machine does everything that is IO, ffmpeg and iteration; Discoverer+ does
# nothing but train. The cluster's GPU allocation is the scarce resource and transcoding video is
# not what it is for.

set -euo pipefail

WAM_ROOT=${WAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK=${WORK:-${HOME}/wam-t041}
CORPUS_TSV=${CORPUS_TSV:-${WAM_ROOT}/configs/cosmos3/corpus_g1_embodiment.tsv}

RAW=${RAW:-${WORK}/raw}
CORPUS=${CORPUS:-${WORK}/cosmos-g1-embodiment}
ENV_FILE=${ENV_FILE:-${WORK}/workstation_env.sh}

# Populates REPOS[] and CAMS[] as index-matched arrays from the one file that defines the corpus.
# They are read from the SAME LINE, so the pairing cannot drift the way two hand-maintained bash
# arrays could — which is the bug this file exists to make impossible.
read_corpus() {
  REPOS=(); CAMS=()
  [[ -f ${CORPUS_TSV} ]] || { echo "FATAL: ${CORPUS_TSV} missing"; exit 1; }
  while IFS=$'\t' read -r repo cam _rest || [[ -n ${repo:-} ]]; do
    repo=${repo%%[[:space:]]}
    [[ -z ${repo} || ${repo} == \#* ]] && continue
    [[ -n ${cam} ]] || { echo "FATAL: ${CORPUS_TSV}: '${repo}' has no camera key"; exit 1; }
    REPOS+=("${repo}"); CAMS+=("${cam}")
  done < "${CORPUS_TSV}"
  [[ ${#REPOS[@]} -gt 0 ]] || { echo "FATAL: ${CORPUS_TSV} lists no sources"; exit 1; }
}

# PR-07 §7 freezes Cosmos3-Super. The gate is a speed bump with a name attached, not a lock: state
# the reason and it runs, and the reason lands in the log so the record says who decided and why.
# It applies here exactly as it does on the cluster — the freeze is about the experiment, not about
# which machine happens to run it.
require_freeze_lifted() {
  : "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
  echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"
}

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "FATAL: $1 not on PATH. $2"; exit 1; }
}
