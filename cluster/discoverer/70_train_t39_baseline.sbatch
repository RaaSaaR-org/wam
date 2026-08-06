#!/bin/bash
# Step 12 — T-39 / PR-07: the positive control. Post-train NVIDIA's own policy on the corpus
# NVIDIA published to demonstrate it, on OUR committed split, and score it with OUR scorer.
#
# Fourteen experiments in this repo produced negatives. Every one of them compares a WAM variant
# against a trivial baseline, and NOT ONE compares anything against a method known to work — so
# none of them can separate "our approach is wrong" from "nothing clears this bar on this corpus".
# The corpus is nvidia/GR00T-N1.7-AppleToPlate, i.e. the tutorial dataset for exactly this recipe.
# We adopted the data and never ran the recipe. This job runs it.
#
#   MODEL_ID=<id> MODEL_DIR=<staged path> TRAINER_ENTRYPOINT=<path in the vendor tree> \
#       sbatch 70_train_t39_baseline.sbatch
#
# CORRECTED 2026-08-06, while implementing scripts/train_t39_baseline.py and before anything was
# submitted: this job now also passes --wam-dataset and --trainer-entrypoint. The first is
# necessary, not cosmetic — the trainer eats the LeRobot SOURCE, but dataset_snapshot_ref has to be
# taken over the CONVERTED episodes, because that is what eval_t16.verify_split recomputes it
# against and what every other number in this repo was measured on. The first version passed only
# the raw root, so the witness it described could not have been verified. No threshold moved: the
# rule is 71_eval_t39_control.sbatch's and is untouched.
#
# READ docs/preregistration/PR-07-positive-control.md FIRST. The verdict rule lives in
# 71_eval_t39_control.sbatch and is in git before this job is submitted. Nothing below decides
# anything; this file only produces a checkpoint and the witness that makes it scorable.
#
# THE TRAINER IS NOT OURS AND IS NOT MODIFIED. A positive control run through our reimplementation
# of someone else's recipe is not a positive control — a failure would again be ambiguous between
# the recipe and our copy of it. The vendored trainer under third_party/ is called as shipped, in
# its OWN venv (its torch/flash-attn pins are not ours). Our code appears in exactly two places:
# the episode restriction and the witness, both in scripts/train_t39_baseline.py.
#
# Not a job array, and the preemption rails are 55_train_i8_rung.sbatch's, for the reasons stated
# in that file's header. One difference, stated because it changes what a requeue costs: the
# vendored trainer has no SIGUSR1 path, so a preemption loses up to one `save_steps` interval
# rather than stopping at a step boundary. The trap below therefore does NOT promise a clean stop.
#
#SBATCH --job-name=wam-t39-train
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1                 # NVIDIA quote for this post-training: 2-4 h on one 40 GB GPU
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t39.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

# NOT the WAM venv. The vendored trainer pins its own torch and attention kernels; importing it
# into virt_envs/wam is how one dependency resolution quietly changes every WAM number afterwards.
export VIRTUAL_ENV=${PROJ}/virt_envs/t39
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

# -- what must be staged, checked before a GPU-second is spent ---------------------------------
#
# MODEL_ID has NO DEFAULT on purpose. PR-07 §8 records that the exact checkpoint id and revision
# were never verified from a primary source, so a default here would be a guess that ends up in an
# artifact as a fact. The operator states it, and it is written into the witness verbatim.
MODEL_ID=${MODEL_ID:?set MODEL_ID to the exact checkpoint id, e.g. MODEL_ID=nvidia/GR00T-N1.7-3B}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the staged weights (compute nodes have Internet; login does not)}
# Also no default, and for the same reason: the vendored trainer's entrypoint path and flag names
# were never verified from a primary source either, and a plausible guess would either fail late or
# run something adjacent and record it as "NVIDIA's recipe" (PR-07 §8).
TRAINER_ENTRYPOINT=${TRAINER_ENTRYPOINT:?set TRAINER_ENTRYPOINT to the trainer inside the vendored tree, relative to it}
SEED=${SEED:-0}
RUN_ID=${RUN_ID:-t39-baseline-seed${SEED}}
OUT=${PROJ}/runs/${RUN_ID}
SPLIT=${PROJ}/wam/configs/splits/i8_train_362.txt
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt
# The CONVERTED dataset — the same root 71_eval_t39_control.sbatch scores against. The witness is
# expressed in its episode ids and its manifest hashes, or verify_split cannot check it.
WAM_DATASET=${PROJ}/data/gr00t-apple-full
VENDOR=${PROJ}/wam/third_party/isaac-gr00t
DRIVER=${PROJ}/wam/scripts/train_t39_baseline.py
PREREG=${PROJ}/wam/docs/preregistration/PR-07-positive-control.md

if [[ ! -d "${VIRTUAL_ENV}" ]]; then
  echo "FATAL: ${VIRTUAL_ENV} missing. The vendored trainer needs its OWN venv —"
  echo "       building it into virt_envs/wam changes the dependency set every WAM number"
  echo "       was produced under. See PR-07 §8 item 4."
  exit 1
fi
for f in "${DRIVER}" "${PREREG}" "${SPLIT}" "${HOLDOUT}" "${WAM_DATASET}"; do
  if [[ ! -e "${f}" ]]; then
    echo "FATAL: ${f} missing — run cluster/discoverer/sync.sh from the Mac first."
    exit 1
  fi
done
if [[ ! -d "${VENDOR}" ]]; then
  echo "FATAL: ${VENDOR} missing. The trainer is vendored UNMODIFIED and is the whole point of"
  echo "       this experiment; a reimplementation would make a failure unattributable again."
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "FATAL: ${MODEL_DIR} missing — stage the weights first (HF_HUB_OFFLINE=1 here)."
  exit 1
fi

# The split file is a COMMITTED artifact and is the same one T-32's rung 362 trains on. Holding it
# by sharing the file, not by copying a number, is what makes the two runs comparable at all.
echo "=== T-39 positive control, PR-07"
echo "=== model ${MODEL_ID}  (weights ${MODEL_DIR})"
echo "=== train split ${SPLIT} ($(grep -cve '^\s*#' -e '^\s*$' "${SPLIT}") episodes)"
echo "=== holdout     ${HOLDOUT} ($(grep -cve '^\s*#' -e '^\s*$' "${HOLDOUT}") episodes) — NOT seen here"

mkdir -p "${OUT}"

# Requeue safety rail — 55_train_i8_rung.sbatch's, same reasoning: without a cap, any failure that
# exits 0 without writing DONE re-queues itself onto every free H200 until the allocation is gone.
# The number is lower here because PR-07 §7 fixes a 12 GPU-h ceiling for this whole experiment,
# and the ceiling is a pre-registered term, not a convenience.
MAX_RESTARTS=2                       # 3 passes x 4 h = the 12 GPU-h ceiling PR-07 §7 fixes
RESTARTS=${SLURM_RESTART_COUNT:-0}
if (( RESTARTS >= MAX_RESTARTS )); then
  echo "FATAL: ${RESTARTS} restarts and still no ${OUT}/DONE — stopping at the PR-07 §7 ceiling."
  echo "       This is the pre-registered budget, not a transient limit. If the recipe genuinely"
  echo "       needs more than 12 GPU-h, that is a finding to record before raising the number."
  exit 1
fi
echo "=== pass $((RESTARTS + 1)) of $((MAX_RESTARTS + 1)) on $(hostname) at $(date -Is)"

START=${SECONDS}
python "${DRIVER}" \
    --vendor-root "${VENDOR}" \
    --trainer-entrypoint "${TRAINER_ENTRYPOINT}" \
    --model-id "${MODEL_ID}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${PROJ}/data/groot-apple-raw" \
    --wam-dataset "${WAM_DATASET}" \
    --train-episodes "${SPLIT}" \
    --exclude-episodes "${HOLDOUT}" \
    --seed "${SEED}" \
    --out-dir "${OUT}" \
    --resume latest \
    --device cuda
RC=$?
ELAPSED=$((SECONDS - START))
echo "trainer exited ${RC} after ${ELAPSED}s at $(date -Is)"

if [[ -f "${OUT}/DONE" ]]; then
  echo "T-39 baseline COMPLETE — ${OUT}/DONE present"
  # The witness is what makes the checkpoint scorable at all: eval_t16.verify_split takes the
  # DISJOINTNESS path when train_episode_ids is recorded, and refuses without an external file to
  # check it against — because the recorded ids and the recorded hash are two fields of one
  # self-description. A checkpoint with no witness is not a cheaper measurement, it is not one.
  if [[ ! -f "${OUT}/run_metadata.json" ]]; then
    echo "FATAL: DONE but no ${OUT}/run_metadata.json — the split witness was not written."
    echo "       71_eval_t39_control.sbatch will refuse to score this, correctly."
    exit 1
  fi
  echo "next: sbatch 71_eval_t39_control.sbatch   (the pre-registered rule; do NOT read it first)"
  exit 0
fi
if [[ "${RC}" -eq 0 && "${ELAPSED}" -lt 300 ]]; then
  echo "FATAL: exited 0 after only ${ELAPSED}s with no ${OUT}/DONE — that is a no-op, not an"
  echo "       interrupted slice. Refusing to requeue."
  exit 1
fi
if [[ "${RC}" -ne 0 ]]; then
  echo "FATAL: trainer failed with ${RC} — requeueing only rebuilds the same crash."
  exit "${RC}"
fi
echo "no ${OUT}/DONE -> requeue ${SLURM_JOB_ID} (resumes from the last save_steps checkpoint;"
echo "the vendored trainer has no SIGUSR1 path, so up to one save interval is lost)"
scontrol requeue "${SLURM_JOB_ID}"
