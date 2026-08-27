#!/usr/bin/env bash
# Submit ONE wave of the PR-08 GEOM_TOL re-run, or the merge. Runs on the workstation.
#
#   ./cluster/discoverer/submit_geom_tol_wave.sh 1     # array 0-3
#   ./cluster/discoverer/submit_geom_tol_wave.sh 2     # array 4-7
#   ./cluster/discoverer/submit_geom_tol_wave.sh 3     # array 8-11
#   ./cluster/discoverer/submit_geom_tol_wave.sh 4     # array 12-15
#   ./cluster/discoverer/submit_geom_tol_wave.sh merge
#
# This file exists for one reason: the submit line is ~250 characters and a terminal paste
# broke it at the wrap point, so `sbatch` read an empty script and `--qos=...` ran as a
# command. A short argv cannot break that way. It changes nothing else — the flags below are
# byte-for-byte the ones in docs/PR-08-RUNBOOK-2026-08-27-geom-tol-re-run.md, which stays the
# authority. Read them here before you run it; that is the point of the echo.
#
# It does NOT decide anything. It prints the command, checks the two things that silently
# waste a wave, and submits. Everything it refuses, it refuses loudly.
set -euo pipefail

HOST=dplus
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
RUN_ID=${RUN_ID:-pr08-geom-tol-v2}
NUM_SHARDS=16

WAVE=${1:-}
case "${WAVE}" in
  1) ARRAY=0-3   ;;
  2) ARRAY=4-7   ;;
  3) ARRAY=8-11  ;;
  4) ARRAY=12-15 ;;
  merge) ARRAY=  ;;
  *)
    echo "usage: $0 {1|2|3|4|merge}" >&2
    echo "  1..4 submit four shards each; run them in order, one at a time." >&2
    exit 64
    ;;
esac

# ---------------------------------------------------------------------------
# Preflight. Both of these have already cost this project a submission once.
# ---------------------------------------------------------------------------

# MaxSubmitJobsPU=8 on the project QoS (verified 2026-08-27 via sacctmgr), and EVERY array
# task counts as one submission. A peer session shares this allocation and its jobs count
# against the same per-user ceiling, so "my queue is empty" is not the question -- "is there
# room for four more" is. Refusing here is free; being rejected by sbatch after the wave is
# assembled is not, and a partial array is worse than none.
echo "==> checking submit-slot budget on ${HOST}"
PENDING=$(ssh "${HOST}" "squeue -u \$USER -r -h -o '%i' | wc -l")
echo "    ${PENDING} job(s) submitted by this user right now (ceiling 8)"
if [[ -z "${ARRAY}" ]]; then
  NEED=1
else
  NEED=4
fi
if (( PENDING + NEED > 8 )); then
  echo "REFUSING: ${PENDING} + ${NEED} > 8 = MaxSubmitJobsPU." >&2
  echo "  Someone else is using the allocation. Wait for their jobs, then re-run." >&2
  ssh "${HOST}" "squeue -u \$USER -r -o '%.14i %.20j %.8T %.10M %.10l'" >&2
  exit 75
fi

# The cluster copy is an rsync, not a clone, and it lags HEAD until sync.sh runs. Submitting
# against a stale adapter is the failure that costs the most and shows the least: all sixteen
# shards exit 3 on the standing gate and 13.64 GPU-h buys nothing. Checking the flag is a
# cheaper question than checking the commit, because it is the flag the shards actually read.
echo "==> checking the adapter's standing flag on the cluster"
FLAG=$(ssh "${HOST}" "grep -m1 '^GATE_QUALIFIED' ${WAM}/scripts/estimators/apple_sam2.py")
echo "    ${FLAG}"
if [[ "${FLAG}" != *"= True"* ]]; then
  echo "REFUSING: the cluster adapter is not gate-qualified. Run ./cluster/discoverer/sync.sh" >&2
  exit 75
fi

# ---------------------------------------------------------------------------
# Submit.
# ---------------------------------------------------------------------------

if [[ -z "${ARRAY}" ]]; then
  # Free QoS: caps at cpu=2 and rejects --gres, while the file asks for 26 threads, 32G and a
  # GPU -- so all four overrides travel together or none of them do. If this is rejected,
  # submit with no overrides at all: it then runs under the project QoS and spends twenty
  # seconds of an H200 doing arithmetic, which is wasteful and correct. Dropping --qos is
  # NOT the fallback; that lands on `normal`, which is one minute and zero GPUs.
  CMD="cd ${WAM} && RUN_ID=${RUN_ID} MERGE=1 NUM_SHARDS=${NUM_SHARDS}"
  CMD="${CMD} sbatch --qos=2cpu-single-host --gres=none --cpus-per-task=2 --mem=8G"
  CMD="${CMD} --time=00:20:00 cluster/discoverer/103_measure_geom_tol.sbatch"
else
  # --time=01:45:00 overrides the file's 00:30:00. Both 01:30 and 01:45 fit the cost model;
  # 01:45 leaves the heaviest shard ~1400s of slack instead of 499s and costs nothing, because
  # Slurm bills runtime, not the request. Do not reach for GEOM_ALLOW_TIGHT_WALL.
  CMD="cd ${WAM} && RUN_ID=${RUN_ID} SHARD=1 NUM_SHARDS=${NUM_SHARDS} GEOM_STEP_FRAMES=1"
  CMD="${CMD} sbatch --qos=ehpc-aif-2026pg01-905 --array=${ARRAY}%4 --time=01:45:00"
  CMD="${CMD} cluster/discoverer/103_measure_geom_tol.sbatch"
fi

echo
echo "==> submitting wave '${WAVE}' as RUN_ID=${RUN_ID}"
echo "    ${CMD}"
echo
ssh "${HOST}" "${CMD}"
echo
ssh "${HOST}" "squeue -u \$USER -r -o '%.14i %.20j %.8T %.10M %.10l %R'"
echo
echo "Watch it with:  ./cluster/discoverer/watch_geom_tol.sh"
