#!/bin/bash
# Step 8 — T-29 / I-7: does the verdict survive being measured in distribution?
#
# 60_eval_t16.sbatch scored the checkpoint the way every WAM result before 2026-07-30 was
# scored: predict() tiles ONE camera frame to the backbone's 9-frame context, while training
# fed the real 9-frame window ending at the chunk. A backbone trained on a moving clip was
# therefore graded on a freeze-frame — and the baseline it lost to, repeat-last-action, is
# nothing but motion continuity.
#
# This job runs BOTH modes back to back on the same GPU, same checkpoint, same holdout, and
# prints the one number that decides it: skill_vs_repeat_pct.
#
#   sbatch 61_eval_t29_frame_history.sbatch
#   RUN_ID=t16-lora-seed1 sbatch 61_eval_t29_frame_history.sbatch
#   CKPT=step-010696 sbatch 61_eval_t29_frame_history.sbatch
#
# Both modes rather than reusing runs/<id>/eval-latest on purpose: that file was produced on a
# different day and possibly a different machine, and an A/B whose two halves differ in more
# than the thing under test is not an A/B. Cost is one extra eval pass — minutes.
#
# DECISION RULE, fixed before the run (docs/improvements.md I-7):
#   skill_vs_repeat_pct moves materially toward or past 0  -> T-16 and T-18 were measured out of
#       distribution; docs/benchmark.md needs a correction, not an addendum, and AC-07 reopens.
#   essentially unchanged                                  -> the negative is real and much
#       stronger: the model had the motion and still lost to inertia. Data (I-8) is next.
#
#SBATCH --job-name=wam-eval-t29
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/eval-t29.%j.out

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
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

EVAL=${PROJ}/wam/scripts/eval_t16.py
BENCH=${PROJ}/wam/scripts/run_bench.py
for f in "${EVAL}" "${BENCH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "FATAL: ${f} missing — run cluster/discoverer/sync.sh from the Mac first."
    exit 1
  fi
done

RUN_ID=${RUN_ID:-t16-lora-seed0}
CKPT=${CKPT:-latest}
OUT=${PROJ}/runs/${RUN_ID}
DATASET=${PROJ}/data/gr00t-apple-full
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt
# The external witness for the split proof. i8_train_362.txt IS the complement of the holdout
# above (402 - 40 = 362, set-equal), so it is both the right witness for a run trained on the
# full training set AND the value eval_t16 cross-checks the complement proof against. Pass it
# unconditionally: train_t16_lora.py records train_episode_ids on every run, so a RETRAINED
# t16 takes the disjointness path and REFUSES without a witness, while the archived
# t16-lora-seed0 checkpoint predates the field and takes the complement path.
# Scoring an I-8 RUNG here means overriding this to that rung's own split file:
#   TRAIN_SPLIT=${PROJ}/wam/configs/splits/i8_train_040.txt RUN_ID=i8-rung040-seed0 sbatch ...
TRAIN_SPLIT=${TRAIN_SPLIT:-${PROJ}/wam/configs/splits/i8_train_362.txt}

if [[ ! -d "${OUT}" ]]; then
  echo "FATAL: ${OUT} does not exist — nothing to score."
  exit 1
fi

# Resolve CKPT to an absolute step dir BEFORE it reaches eval_t16.py. resolve_checkpoint does
# `step_dir = Path(which)` for any non-"latest" value, with no join against --run-dir, so the
# `CKPT=step-010696` invocation this file's own header advertises (line 15) would resolve against
# the job's CWD and die on "step-010696/model.safetensors missing" — after the node is allocated.
# CKPT=latest is unaffected either way; this makes the documented override work too.
if [[ "${CKPT}" == "latest" && -e "${OUT}/latest" ]]; then
  STEP_DIR=$(readlink -f "${OUT}/latest")
elif [[ "${CKPT}" == "latest" ]]; then
  STEP_DIR=$(ls -d "${OUT}"/checkpoints/step-* | sort | tail -1)
elif [[ -d "${CKPT}" ]]; then
  STEP_DIR="${CKPT}"
else
  STEP_DIR="${OUT}/checkpoints/${CKPT}"
fi
if [[ ! -f "${STEP_DIR}/model.safetensors" ]]; then
  echo "FATAL: ${STEP_DIR}/model.safetensors missing — not a restorable checkpoint."
  echo "       Available: $(ls -d "${OUT}"/checkpoints/step-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  exit 1
fi
echo "checkpoint dir: ${STEP_DIR}"

# The two runs differ in EXACTLY one flag. Everything else — checkpoint, dataset, holdout,
# camera, device — is held identical by construction below rather than by copy-paste.
common=(
  --run-dir "${OUT}"
  --checkpoint "${STEP_DIR}"
  --dataset "${DATASET}"
  --holdout "${HOLDOUT}"
  --train-episodes "${TRAIN_SPLIT}"
  --camera ego
  --device cuda
)

echo "=== A: tiled single frame (how every result before 2026-07-30 was measured)"
python "${EVAL}" "${common[@]}" --out "${OUT}/eval-t29-tiled"

echo
echo "=== B: real frame window (what training fed the model)"
python "${EVAL}" "${common[@]}" --frame-history --out "${OUT}/eval-t29-history"

echo
echo "=== the comparison"
# --compare refuses two runs whose holdouts differ, so the columns are guaranteed to mean the
# same thing. Both halves came from this job, this GPU, this checkpoint. --no-write keeps the
# bench.json eval_t16.py wrote — it is the one carrying the "+frame_history" run_name, which a
# re-score from the directory name alone would flatten back to the run id.
python "${BENCH}" "${OUT}/eval-t29-tiled" "${OUT}/eval-t29-history" --compare --no-write || true

echo
echo "=== the decision (docs/improvements.md I-7)"
python - "${OUT}/eval-t29-tiled/bench.json" "${OUT}/eval-t29-history/bench.json" <<'PY'
import json, sys

tiled, history = (json.load(open(p)) for p in sys.argv[1:3])
a, b = tiled["skill_vs_repeat_pct"], history["skill_vs_repeat_pct"]
print(f"skill_vs_repeat_pct   tiled {a:+.2f}%   ->   frame history {b:+.2f}%   (delta {b - a:+.2f} pp)")
print(f"level                 {tiled['level_name']}   ->   {history['level_name']}")
print(f"score                 {tiled['score']:.1f}   ->   {history['score']:.1f}")
print()
if b > 0:
    print("VERDICT: L1 CLEARED with the real window. The recorded T-16/T-18 negatives were")
    print("         measured out of distribution. docs/benchmark.md needs a CORRECTION and")
    print("         AC-07 reopens. Re-run the T-18 ablation the same way before concluding.")
elif b - a > 5.0:
    print("VERDICT: moved materially but still below the bar. The tiled eval was understating")
    print("         the model; the negative stands but its size does not. Re-state both runs")
    print("         with the real window, then go to I-8 (data-scaling curve).")
else:
    print("VERDICT: unchanged. The model had the motion available and still lost to inertia —")
    print("         the negative is about the model, not the harness, and is now much stronger.")
    print("         Next: I-8 (data-scaling curve) before committing months to D1/D2 collection.")
PY

echo
echo "Copy back to the Mac (run this ON THE MAC), then re-score on CPU forever:"
echo "  ./cluster/discoverer/sync.sh --pull ${RUN_ID}"
echo "  python scripts/run_bench.py runs/${RUN_ID}/eval-t29-tiled runs/${RUN_ID}/eval-t29-history --compare"
