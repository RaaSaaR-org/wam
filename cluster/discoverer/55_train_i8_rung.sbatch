#!/bin/bash
# Step 10 — I-8 / T-32: one RUNG of the data-scaling curve.
#
# Every negative WAM has (T-18, T-15/T-24/T-26, T-16) came from the same 402 success-only
# episodes of one task. "Not enough data" is the standing explanation for all of them, has
# never been tested, and is the most expensive conclusion in the project — it buys a G1 EDU4,
# a teleop rig and months of recording. This job retrains T-16 on a smaller, COMMITTED subset
# so the three-point curve can be measured on the allocation we already have.
#
#   RUNG=040 SEED=0 sbatch 55_train_i8_rung.sbatch
#   RUNG=120 SEED=0 sbatch 55_train_i8_rung.sbatch
#   RUNG=040 SEED=1 sbatch 55_train_i8_rung.sbatch        # the reproducibility replicate
#   RUNG=040 SEED=0 STEPS=2290 RUN_ID=i8-rung040-e17-seed0 sbatch 55_train_i8_rung.sbatch
#                                                         # stage 2, equal-EPOCH control
#
# Rung 362 is NOT submitted: it is runs/t16-lora-seed0 (same config, same 362 episodes, same
# 20 000 steps, same seed 0). Re-running it would spend 36 GPU-h reproducing a number we have.
#
# This is deliberately NOT a job array. Job arrays are an open question on this cluster
# (docs/discoverer.md §"open questions" item 8, against a provider whose docs the repo marks
# contested and which was wrong about the partition name, the node counts and the CUDA module
# when checked), and an array task's interaction with `scontrol requeue` / SLURM_RESTART_COUNT
# is exactly the unknown that, if wrong, produces the unbounded requeue loop MAX_RESTARTS
# exists to stop — a loop that re-queues onto every free H200 until the allocation is gone.
# Three plain submissions cost nothing (max 4 running jobs), and each rung gets its own DONE,
# its own restart counter and its own failure blast radius.
#
# Everything below the RUNG block is 50_train_t16.sbatch verbatim. The preemption machinery is
# not re-derived here on purpose; it is the thing that protects a 20 000-step run.
#
#SBATCH --job-name=wam-i8-rung
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1                 # 1 GPU schedules fast; 8 queues behind everything
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/i8.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# No .git on the cluster copy — sync.sh stamped the hash so run_metadata records real
# provenance instead of "unknown" (AC-04). Empty/missing file just falls back to that.
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

TRAIN=${PROJ}/wam/scripts/train_t16_lora.py
if [[ ! -f "${TRAIN}" ]]; then
  echo "FATAL: ${TRAIN} missing — run cluster/discoverer/sync.sh from the Mac first."
  exit 1
fi

# -- the rung ---------------------------------------------------------------------------------
RUNG=${RUNG:?set RUNG to the rung size, zero-padded to 3 digits (e.g. RUNG=040)}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
RUN_ID=${RUN_ID:-i8-rung${RUNG}-seed${SEED}}
OUT=${PROJ}/runs/${RUN_ID}
SPLIT=${PROJ}/wam/configs/splits/i8_train_${RUNG}.txt
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt

if [[ ! -f "${SPLIT}" ]]; then
  echo "FATAL: ${SPLIT} missing. The rung files are COMMITTED artifacts —"
  echo "       regenerate with scripts/make_rung_splits.py on the Mac and re-sync."
  echo "       Do not generate them here; a rung chosen on the cluster is not reviewable."
  exit 1
fi

# Queue-order rail. Rung 362's headline number IS 61_eval_t29_frame_history.sbatch's output,
# so a T-29 verdict that has not been read yet means this experiment's premise is unknown —
# and if T-29 comes back "L1 CLEARED" the T-16 negative is retracted and I-8 asks a different
# question. 36 GPU-h is cheap; 36 GPU-h spent answering a question that moved is not.
T29=${PROJ}/runs/t16-lora-seed0/eval-t29-history/bench.json
if [[ ! -f "${T29}" && "${SKIP_T29_CHECK:-0}" != "1" ]]; then
  echo "FATAL: ${T29} missing — 61_eval_t29_frame_history.sbatch has not reported."
  echo "       I-8 consumes that file as rung 362. Run T-29 first, read its verdict, then"
  echo "       re-read docs/improvements.md I-8 before submitting. SKIP_T29_CHECK=1 overrides"
  echo "       ONLY if you have read the verdict and it did not change the premise."
  exit 1
fi

mkdir -p "${OUT}"

# Requeue safety rail. scontrol requeue keeps SLURM_JOB_ID and increments SLURM_RESTART_COUNT
# (verified by 25_requeue_probe.sbatch: restarts=0 -> restarts=1). Without a cap, any failure
# that exits 0 without writing DONE — an early return, a corrupt --resume checkpoint — is an
# unbounded loop that re-queues itself onto every free H200 until the allocation is gone.
# Checked before the model loads so a wedged run costs seconds, not a full staging pass.
MAX_RESTARTS=24                      # ~96 GPU-h ceiling; a healthy 20k-step run needs <10
RESTARTS=${SLURM_RESTART_COUNT:-0}
if (( RESTARTS >= MAX_RESTARTS )); then
  echo "FATAL: ${RESTARTS} restarts and still no ${OUT}/DONE — stopping instead of looping."
  echo "       Inspect ${OUT} and the earlier ${PROJ}/logs/i8.*.out before resubmitting."
  exit 1
fi
echo "=== I-8 rung ${RUNG} seed ${SEED} (${STEPS} steps) pass $((RESTARTS + 1)) (restarts=${RESTARTS})"
echo "=== on $(hostname) at $(date -Is), split ${SPLIT}"

# The trap only fires while the shell is at `wait` — never run python in the foreground here.
START=${SECONDS}
python "${TRAIN}" \
    --backbone-config "${PROJ}/wam/configs/model/wan22_ti2v_5b.yaml" \
    --backbone-source "${PROJ}/models/Wan2.2-TI2V-5B" \
    --training-config "${PROJ}/wam/configs/training/joint_wan_gr00t.yaml" \
    --dataset "${PROJ}/data/gr00t-apple-full" \
    --exclude-episodes "${HOLDOUT}" \
    --train-episodes "${SPLIT}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --camera ego \
    --out-dir "${OUT}" \
    --resume latest \
    --checkpoint-every-min 30 \
    --checkpoints-total-limit 3 \
    --save-adapter-only \
    --device cuda &
PY=$!

# --signal=B: delivers USR1 to THIS shell, not to python. Forward it and let the trainer
# stop at a step boundary: requeueing from inside the trap (what this used to do) tears the
# process down mid-step, so every requeue threw away the work since the last checkpoint —
# up to the full 30-minute checkpoint interval, on every single 4-hour wrap-around.
GOT_USR1=0
trap 'GOT_USR1=1; echo "USR1 at $(date -Is) -> forward to pid ${PY}, waiting for checkpoint";
      kill -USR1 "${PY}" 2>/dev/null || true' USR1

# A trap makes `wait` return early (128+signal) with the child still alive, and `set -e`
# would abort the script on that status — hence `|| true` and the loop until the pid is gone.
RC=0
while kill -0 "${PY}" 2>/dev/null; do
  wait "${PY}" && RC=0 || RC=$?
done
ELAPSED=$((SECONDS - START))
echo "trainer pid ${PY} exited ${RC} after ${ELAPSED}s at $(date -Is) (usr1=${GOT_USR1})"

if [[ -f "${OUT}/DONE" ]]; then
  echo "I-8 rung ${RUNG} seed ${SEED} COMPLETE — ${OUT}/DONE present, not requeueing"
  exit 0
fi
if [[ "${RC}" -eq 0 && "${GOT_USR1}" -eq 0 && "${ELAPSED}" -lt 300 ]]; then
  # Exit 0, no DONE, never signalled, and gone in under 5 minutes: the trainer did nothing.
  # Requeueing a silent no-op is the one path that can spin without ever burning a GPU long
  # enough to look wrong, so treat it as fatal rather than as an interrupted slice.
  echo "FATAL: trainer exited 0 after only ${ELAPSED}s with no ${OUT}/DONE and no signal —"
  echo "       that is a no-op, not an interrupted slice. Refusing to requeue."
  exit 1
fi
if [[ "${RC}" -ne 0 && "${GOT_USR1}" -eq 0 ]]; then
  # Died on its own (config error, OOM, a rung file that overlaps the holdout, ...).
  # Requeueing that only rebuilds the same crash every time the scheduler finds a free H200.
  echo "FATAL: trainer failed with ${RC} and was never signalled — refusing to requeue"
  exit "${RC}"
fi
echo "no ${OUT}/DONE -> requeue ${SLURM_JOB_ID} to continue from the last checkpoint"
scontrol requeue "${SLURM_JOB_ID}"
