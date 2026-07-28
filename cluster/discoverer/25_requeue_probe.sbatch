#!/bin/bash
# Step 3 — prove --requeue + --signal work here BEFORE building a 4-hour training run on
# them. Both are standard Slurm but documented nowhere on the Discoverer site. Free QoS.
#
#   sbatch 25_requeue_probe.sbatch
#
# Expect in the log: one "PASS 1", a USR1 trap firing, then "PASS 2" appended to the SAME
# file (--open-mode=append) under the same job id. If pass 2 never appears, the whole
# checkpoint/resume architecture in 50_train_t16.sbatch is invalid.
#
#SBATCH --job-name=wam-requeue
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/requeue-probe.%j.out

set -uo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

MARK=${PROJ}/logs/requeue-probe.${SLURM_JOB_ID}.pass
PASS=$(( $(cat "${MARK}" 2>/dev/null || echo 0) + 1 ))
echo "${PASS}" > "${MARK}"
echo "PASS ${PASS} — job ${SLURM_JOB_ID}, restarts=${SLURM_RESTART_COUNT:-0}, host $(hostname)"

if [[ "${PASS}" -ge 2 ]]; then
  echo "REQUEUE PROBE OK — signal + requeue + append all work"
  rm -f "${MARK}"
  exit 0
fi

requeue_me() {
  echo "USR1 received at $(date -Is) -> requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_me USR1

# Sleep in the background and `wait`, so the shell can service the trap. A foreground
# sleep would swallow the signal — this is the pattern 50_train_t16.sbatch relies on.
sleep 600 &
wait
