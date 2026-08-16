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

# GROOT_PATCH_MISTRAL=1 — upstream's own escape hatch, and it is NOT one of the two forbidden ones.
# transformers' PreTrainedTokenizerBase._patch_mistral_regex calls is_base_mistral(), which calls
# huggingface_hub.model_info() on the model NAME while loading the tokenizer. Offline that raises
# OfflineModeIsEnabled and the run dies ~150 s in, after the weights have already been read (job
# 187802, 2026-08-15); upstream hit the same call as a 600 s HANG in their own CI.
#
# gr00t/__init__.py:_patch_mistral installs a wrapper that returns the tokenizer untouched when the
# name is a directory or does not contain "mistral" — which is every path we take — so no hub call
# happens and no weight-loading behaviour changes. It is gated on its own env var and is a separate
# function from _patch_hf_local_first.
#
# DO NOT reach for GROOT_HF_LOCAL_FIRST or GROOT_SKIP_HF_MODEL_WEIGHTS instead. Those wrap
# PreTrainedModel.from_pretrained generically, so they would also skip the start_from_checkpoint
# load, zero the parameters, and fake the loading-info that setup.py's strict missing-key guard
# reads — training from zeros with no error anywhere.
export GROOT_PATCH_MISTRAL=1
# albumentations phones home for a version check on import (check_for_updates()). It is only a
# warning when it fails, but it is a network round trip inside a GPU job on a node that is not
# supposed to need one, and its timeout is not ours to rely on.
export NO_ALBUMENTATIONS_UPDATE=1
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

# THE EMBODIMENT TAG, and why it is not the one the corpus is named after. Upstream's own enum
# (gr00t/data/embodiment_tags.py) files UNITREE_G1 -> "unitree_g1_full_body_with_waist_height_nav_cmd"
# under "pre-registered POSTTRAIN tags (require finetuned checkpoint)", and NEW_EMBODIMENT under
# "finetuning tag (for custom robots)". nvidia/GR00T-N1.7-3B is the BASE checkpoint, so the unitree
# tag names an embodiment head this checkpoint does not carry trained weights for. new_embodiment is
# the tag upstream's finetuning path is written around, and it is what the modality config below
# registers. Overridable, but changing it changes what is being measured.
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
# The .py that registers ${EMBODIMENT_TAG} in upstream's MODALITY_CONFIGS. Named ONCE and used
# TWICE below — once for the statistics and once for the trainer. They must be the same file: the
# statistics are computed per modality key, and normalising a 43-dim vector sliced one way with
# statistics computed over it sliced another way does not raise, it just trains against the wrong
# target (configs/groot/new_embodiment_config_defaults.py header).
MODALITY_CONFIG=${MODALITY_CONFIG:-${PROJ}/wam/configs/groot/new_embodiment_config_defaults.py}

# HYPERPARAMETERS ARE UPSTREAM'S, from examples/finetune.sh at the vendored commit, so that a
# failure is attributable to NVIDIA's recipe rather than to our tuning of it (PR-07 §3). PR-07
# fixes a budget, not a learning rate, and deliberately names no hyperparameter — so the defensible
# default is the one the recipe ships with. Every value is echoed into the log below.
#
# Two deviations, both throughput/durability rather than science, and both stated rather than
# buried:
#   SAVE_STEPS      500 not 1000 — --requeue can lose up to one save interval, and this job has a
#                   4 h walltime against a 12 GPU-h ceiling. Halving the interval halves the worst
#                   case of a preemption; it changes no gradient.
#   NUM_WORKERS     8 not 4 — this cluster has no FFmpeg, so video decoding runs through the PyAV
#                   fallback patch, which is slower than torchcodec. 8 <= the 26 threads/GPU
#                   billing cap. It changes no gradient either.
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
SHARD_SIZE=${SHARD_SIZE:-1024}
NUM_SHARDS_PER_EPOCH=${NUM_SHARDS_PER_EPOCH:-100000}
EPISODE_SAMPLING_RATE=${EPISODE_SAMPLING_RATE:-0.1}

if [[ ! -d "${VIRTUAL_ENV}" ]]; then
  echo "FATAL: ${VIRTUAL_ENV} missing. The vendored trainer needs its OWN venv —"
  echo "       building it into virt_envs/wam changes the dependency set every WAM number"
  echo "       was produced under. See PR-07 §8 item 4."
  exit 1
fi
for f in "${DRIVER}" "${PREREG}" "${SPLIT}" "${HOLDOUT}" "${WAM_DATASET}" "${MODALITY_CONFIG}"; do
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

# THE SECOND SET OF WEIGHTS, which is easy to forget because nothing on the command line names it.
# launch_finetune.py hardcodes config.model.model_name = "nvidia/Cosmos-Reason2-2B" and the backbone
# is constructed by Qwen3VLForConditionalGeneration.from_pretrained(model_name) — a real 4.5 GB
# read from the HF cache, not a config-only build. The no-weight fast paths in gr00t/__init__.py
# are pytest-only (they need PYTEST_CURRENT_TEST or GROOT_HF_LOCAL_FIRST *and*
# GROOT_SKIP_HF_MODEL_WEIGHTS) and must NOT be used to work around a missing cache: that patch
# wraps PreTrainedModel.from_pretrained generically, so it would skip the start_from_checkpoint
# load too and zero the model while faking the loading-info that the strict key check reads. The
# result trains silently from zeros. Fail here instead.
COSMOS_BACKBONE=${HF_HUB_CACHE}/models--nvidia--Cosmos-Reason2-2B
if [[ ! -d "${COSMOS_BACKBONE}" ]]; then
  echo "FATAL: ${COSMOS_BACKBONE} missing. HF_HUB_OFFLINE=1 is set here, so the Cosmos-Reason2-2B"
  echo "       backbone cannot be fetched at run time and the trainer dies after the GPU is"
  echo "       allocated. Ship it from the workstation cache:"
  echo "         ./cluster/discoverer/sync.sh --backbone"
  echo "       Do NOT set GROOT_HF_LOCAL_FIRST or GROOT_SKIP_HF_MODEL_WEIGHTS to get past this."
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

# RESUME IS CONDITIONAL, not a constant. HF Trainer raises when resume_from_checkpoint=True and
# output_dir holds no checkpoint-*, so passing it unconditionally would make the FIRST pass fail and
# every requeue after it fail identically. Passing it never would make a requeue silently restart
# from step 0 and burn the 12 GPU-h ceiling three times over the same first 4 h of training.
RESUME_FLAG=()
if compgen -G "${OUT}/checkpoints/checkpoint-*" >/dev/null; then
  echo "=== resuming: $(ls -d "${OUT}"/checkpoints/checkpoint-* | tail -1)"
  RESUME_FLAG=(--resume_from_checkpoint)
fi

echo "=== embodiment ${EMBODIMENT_TAG} via ${MODALITY_CONFIG}"
echo "=== recipe     max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} batch=${GLOBAL_BATCH_SIZE}"
echo "===            lr=${LEARNING_RATE} warmup=${WARMUP_RATIO} wd=${WEIGHT_DECAY}"
echo "===            shard=${SHARD_SIZE} shards/epoch=${NUM_SHARDS_PER_EPOCH} sampling=${EPISODE_SAMPLING_RATE}"
echo "=== backbone   ${COSMOS_BACKBONE} (offline)"

START=${SECONDS}
# RC is captured with `|| RC=$?`, not with a bare call followed by `RC=$?`. `set -e` is on at the
# top of this file, so a bare failing call terminates the script immediately and every branch below
# that inspects RC is unreachable — including the one that refuses to requeue a hard crash.
RC=0
# Everything after the bare -- is handed to the vendored trainer VERBATIM and appears in the log
# exactly as invoked. --dataset-path and --output-dir are the driver's (it points them at the
# 362-episode subset it just built); every flag here is upstream's own, spelled as
# examples/finetune.sh spells it.
python "${DRIVER}" \
    --vendor-root "${VENDOR}" \
    --trainer-entrypoint "${TRAINER_ENTRYPOINT}" \
    --model-id "${MODEL_ID}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${PROJ}/data/groot-apple-raw" \
    --wam-dataset "${WAM_DATASET}" \
    --train-episodes "${SPLIT}" \
    --exclude-episodes "${HOLDOUT}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --modality-config "${MODALITY_CONFIG}" \
    --seed "${SEED}" \
    --out-dir "${OUT}" \
    --resume latest \
    --device cuda \
    -- \
    --base_model_path "${MODEL_DIR}" \
    --embodiment_tag "${EMBODIMENT_TAG}" \
    --modality_config_path "${MODALITY_CONFIG}" \
    --num_gpus 1 \
    --max_steps "${MAX_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers "${NUM_WORKERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --shard_size "${SHARD_SIZE}" \
    --num_shards_per_epoch "${NUM_SHARDS_PER_EPOCH}" \
    --episode_sampling_rate "${EPISODE_SAMPLING_RATE}" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"} || RC=$?
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
