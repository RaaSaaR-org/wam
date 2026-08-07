#!/bin/bash
# Step 19 — T-041 / PR-09 §6: the verdict. `T041_RULE_V1`.
#
#   T041_FREEZE_LIFTED="<reason>" sbatch 95_eval_t041_embodiment.sbatch
#
# DO NOT READ THIS FILE LOOKING FOR THE ANSWER — the rule it executes was fixed in
# docs/preregistration/PR-09-cosmos-super-finetune.md before anything was generated, and lives in
# scripts/eval_t041_embodiment.py. This job only produces the clips and runs the scorer.
#
# WHAT IS COMPARED. 30 held-out prompts, two arms, identical seed and sampler:
#   base  = Cosmos3-Super at the pinned revision, no adapter
#   lora  = the same model plus the adapter 94_train produced
# The pairing is what makes 30 prompts enough; the discordant pairs carry all the information and
# the exact McNemar test at alpha=0.05 is borrowed, not coined.
#
# THE SCORER NEVER SEES WHICH ARM MADE A CLIP. build-sheet writes a shuffled sheet of neutral item
# ids pointing into a neutral symlink tree, plus a separate key.json. --verdict is the only step
# that opens the key. That separation is also the human fallback: if G0b fails, a person scores
# the SAME sheet by hand and the same rule applies unchanged.
#
# 20 CALIBRATION ITEMS RIDE ALONG (G0b), all REAL footage: 10 held-out G1 + Dex3 clips that must
# score YES and 10 WidowX clips from the recipe's own BridgeData2 subset that must score NO. All
# 20 must be right. It is a necessary check and not a sufficient one — PR-09 §9 says so — but a
# rubric that cannot separate a real Dex3 from a real parallel gripper cannot adjudicate a
# generated frame.
#
#SBATCH --job-name=wam-t041-eval
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:8
#SBATCH --mem=1800G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t041-eval.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
source "${PROJ}/cosmos_env.sh"

: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8
export PATH=${UV_INSTALL_DIR}:${FFMPEG_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=""
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=26
nvidia-smi -L

NPROC=${NPROC:-8}
RUN_ID=${RUN_ID:-t041-super-lora}
OUT=${PROJ}/runs/${RUN_ID}
EVAL=${OUT}/eval
RUN_DIR=${OUT}/train/cosmos3/sft/vision_sft_super
LORA_MODEL=${LORA_MODEL:-${RUN_DIR}/model}
WAM=${PROJ}/wam
CONFIG=${WAM}/configs/cosmos3/t041_eval_selection.toml
PROMPTS=${EVAL}/t041_eval_prompts.jsonl
PORT=${PORT:-8000}

[[ -f "${OUT}/DONE" ]] || { echo "FATAL: ${OUT}/DONE missing — training did not complete."; exit 1; }
[[ -f "${OUT}/run_metadata.json" ]] || { echo "FATAL: no run_metadata.json — the run is not scorable."; exit 1; }
[[ -d "${LORA_MODEL}" ]] || {
  echo "FATAL: ${LORA_MODEL} missing. Export the adapter first:"
  echo "  CKPT=${RUN_DIR}/checkpoints/\$(cat ${RUN_DIR}/checkpoints/latest_checkpoint.txt)"
  echo "  python -m cosmos_framework.scripts.export_model --checkpoint-path \$CKPT \\"
  echo "      --config-file ${RUN_DIR}/config.yaml -o ${LORA_MODEL}"
  exit 1; }
mkdir -p "${EVAL}"
source "${FRAMEWORK}/.venv/bin/activate"

# -- 1. the prompts, materialised from the pre-registered selection -----------------------------
# The rule (sorted-by-uuid, first 30, from the val split) and every sampler setting were committed
# in ${CONFIG} before generation. This step only executes it, and hashes the result.
if [[ ! -f "${PROMPTS}" ]]; then
  python "${WAM}/scripts/make_t041_eval_prompts.py" \
      --corpus "${DATASET_PATH}" --config "${CONFIG}" -o "${PROMPTS}"
fi
echo "=== prompts $(wc -l < "${PROMPTS}")  sha256 $(cat "${PROMPTS}.sha256")"

# -- 2. generate, both arms, same seed ----------------------------------------------------------
cd "${FRAMEWORK}"
for ARM in base lora; do
  CKPT=$([[ "${ARM}" == base ]] && echo "${BASE_CHECKPOINT_PATH}" || echo "${LORA_MODEL}")
  DEST=${EVAL}/clips/${ARM}
  mkdir -p "${DEST}" "${EVAL}/payloads/${ARM}"
  echo "=== generating arm ${ARM} from ${CKPT}"
  # One payload per prompt, built from the committed settings. Identical across arms by
  # construction: the same file writer, the same config, the same seed. A sampler difference
  # between arms would turn the paired test into a comparison of samplers.
  python - "${PROMPTS}" "${CONFIG}" "${EVAL}/payloads/${ARM}" <<'PY'
import json, pathlib, sys, tomllib
prompts, cfg_path, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
g = tomllib.loads(pathlib.Path(cfg_path).read_text())["generation"]
for line in pathlib.Path(prompts).read_text().splitlines():
    if not line.strip():
        continue
    p = json.loads(line)
    payload = {"model_mode": g["model_mode"], "name": p["uuid"], "prompt": p["prompt"],
               "negative_prompt": "", "enable_sound": g["enable_sound"],
               "num_steps": g["num_steps"], "guidance": g["guidance"], "shift": g["shift"],
               "fps": g["fps"], "num_frames": g["num_frames"], "resolution": g["resolution"],
               "aspect_ratio": g["aspect_ratio"], "seed": g["seed"]}
    (dest / f"{p['uuid']}.json").write_text(json.dumps(payload, indent=2) + "\n")
print("payloads:", len(list(dest.glob('*.json'))))
PY
  SEED=$(python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('${CONFIG}').read_text())['generation']['seed'])")
  PRESET=$(python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('${CONFIG}').read_text())['generation']['parallelism_preset'])")
  for PAY in "${EVAL}/payloads/${ARM}"/*.json; do
    UUID=$(basename "${PAY}" .json)
    [[ -f "${DEST}/${UUID}.mp4" ]] && continue
    torchrun --nproc-per-node="${NPROC}" -m cosmos_framework.scripts.inference \
        --parallelism-preset="${PRESET}" -i "${PAY}" -o "${DEST}/_raw/${UUID}" \
        --checkpoint-path "${CKPT}" --seed="${SEED}"
    # The inference script names its output by the payload's "name"; normalise to <uuid>.mp4 so
    # build-sheet's neutral tree has exactly one file per (arm, prompt).
    FOUND=$(find "${DEST}/_raw/${UUID}" -name '*.mp4' | head -1)
    [[ -n "${FOUND}" ]] || { echo "FATAL: no mp4 for ${ARM}/${UUID}"; exit 1; }
    mv "${FOUND}" "${DEST}/${UUID}.mp4"
  done
  echo "=== arm ${ARM}: $(find "${DEST}" -maxdepth 1 -name '*.mp4' | wc -l) clips"
done

# -- 3. blind ------------------------------------------------------------------------------------
CAL=${CAL:-${PROJ}/data/t041-calibration}
[[ -d "${CAL}/positive" && -d "${CAL}/negative" ]] || {
  echo "FATAL: ${CAL} must hold positive/ (real held-out G1+Dex3 clips) and negative/ (real"
  echo "       non-G1 manipulator clips, e.g. BridgeData2's WidowX). Without it G0b cannot run"
  echo "       and PR-09 §6 issues no verdict."; exit 1; }
python "${WAM}/scripts/eval_t041_embodiment.py" --config "${CONFIG}" --out "${EVAL}" \
    build-sheet --prompts "${PROMPTS}" --clips "${EVAL}/clips" --calibration "${CAL}"

# -- 4. judge --------------------------------------------------------------------------------------
JUDGE_MODEL=$(python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('${CONFIG}').read_text())['judge']['model'])")
SERVER_LOG=${PROJ}/logs/vllm-judge.${SLURM_JOB_ID}.log
echo "=== starting judge ${JUDGE_MODEL} on :${PORT}"
uvx --with nvidia-cuda-runtime-cu12 "vllm@${VLLM_VERSION:-0.19.0}" serve "${JUDGE_MODEL}" \
    --tensor-parallel-size 1 --port "${PORT}" --allowed-local-media-path / \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 120); do
  grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "FATAL: judge died:"; tail -40 "${SERVER_LOG}"; exit 1; }
  sleep 10
done
grep -q "Application startup complete" "${SERVER_LOG}" || {
  echo "FATAL: judge not ready after 20 min"; tail -40 "${SERVER_LOG}"; exit 1; }

python "${WAM}/scripts/eval_t041_embodiment.py" --config "${CONFIG}" --out "${EVAL}" \
    judge --server "http://localhost:${PORT}/v1" --resume

# -- 5. verdict --------------------------------------------------------------------------------------
# run_metadata.json is 94_train's own report of where the run got to. The gate reads it rather
# than re-deriving the iteration count here, because the counter lives in the DCP trainer state.
python "${WAM}/scripts/eval_t041_embodiment.py" --config "${CONFIG}" --out "${EVAL}" \
    verdict --prompts "${PROMPTS}" --corpus "${DATASET_PATH}" \
    --run-metadata "${OUT}/run_metadata.json"

echo
echo "=== T-041 done. The scoring sheet is human-rescoreable forever:"
echo "    ${EVAL}/scoring_sheet.jsonl  +  ${EVAL}/items/"
echo "    replace scores.jsonl with hand scores and re-run the verdict step — same rule."
echo "copy back with:  ./cluster/discoverer/sync.sh --pull ${RUN_ID}"
