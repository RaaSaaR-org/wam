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
# 4 h is the QoS ceiling, not a choice — `ehpc-aif-2026pg01-905` has MaxWall 04:00:00
# (docs/discoverer.md:117-119), so "ask for a longer wall" is not on the table and a job that does
# not fit here has to be split across submissions. With generation batched to two checkpoint loads
# (see step 2) the honest estimate is ~1.4 h — ~0.6 h of generation and ~0.8 h of judge — so the
# arms stay in one job. If the judge phase ever grows past ~2 h, split the arms rather than the
# scoring: two submissions of step 2 need no code change, because the missing-list is the resume.
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

# Guarded explicitly rather than left to `set -u`, and here rather than at first use: unset,
# DATASET_PATH dies fifty lines below at the prompt step and BASE_CHECKPOINT_PATH just after it
# inside the arm loop — both with a bare "unbound variable", and both after Slurm has already
# handed this job 8 H200s on the paid QoS. That is precisely the failure 92b_register_corpus.sbatch
# was written to prevent for 94, and 95 reads the same registered corpus and the same staged
# weights while carrying none of 94's guards.
: "${FRAMEWORK:?90_build_cosmos_env.sbatch has not run — no cosmos-framework checkout or venv.}"
: "${UV_INSTALL_DIR:?90_build_cosmos_env.sbatch has not run — no uv, so no judge server.}"
: "${FFMPEG_PREFIX:?90_build_cosmos_env.sbatch has not run — no ffmpeg.}"
: "${BASE_CHECKPOINT_PATH:?91_stage_cosmos_weights.sbatch has not run — no staged weights, no base arm.}"
: "${DATASET_PATH:?no corpus registered — run sync.sh --corpus then sbatch 92b_register_corpus.sbatch}"

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
# THE BASE ARM READS THE HF DOWNLOAD, NOT THE DCP CONVERSION, and the two are the same bytes:
# 91_stage downloaded nvidia/Cosmos3-Super@e0262be9 to Cosmos3-Super-hf and converted *that* to
# the DCP tree at ${BASE_CHECKPOINT_PATH}, which exists because 94 trains from DCP. Inference
# cannot read it. A DCP checkpoint leaves config_file at the default .py module, and the module
# path then needs a Hydra experiment name inferred by regex from the checkpoint path
# (inference/common/args.py:598-606) — `/<experiment>/checkpoints/iter_<N>/`, which a staged tree
# named for its model does not match. Job 187078 died there in 1:42. The experiment the regex
# wants is `cosmos3_ga_64bm32b_v3_midtrain` (named in inference/configs/model/Cosmos3-Super.yaml)
# and it is NOT in the OSS Hydra store, so naming it explicitly would only move the failure.
#
# The HF tree needs none of it: CheckpointType.from_path returns `hf`, config_file falls back to
# the checkpoint's own config.json, and no experiment is consulted. That is also the arm the
# exported adapter takes, so both arms now enter through one code path instead of two — the DCP
# path additionally honours `use_ema_weights` (default True, load_ema_to_reg at
# inference/inference.py:1193) and the HF path has no such switch, an asymmetry between arms that
# no rubric could have caught downstream.
BASE_MODEL=${BASE_MODEL:-${PROJ}/checkpoints/Cosmos3-Super-hf}
WAM=${PROJ}/wam
CONFIG=${WAM}/configs/cosmos3/t041_eval_selection.toml
PROMPTS=${EVAL}/t041_eval_prompts.jsonl
PORT=${PORT:-8000}
CAL=${CAL:-${PROJ}/data/t041-calibration}

[[ -f "${OUT}/DONE" ]] || { echo "FATAL: ${OUT}/DONE missing — training did not complete."; exit 1; }
# Both arms are asserted loadable BEFORE any GPU-second, and by the property that actually failed:
# a config.json at the root is what makes CheckpointType.from_path return `hf` and what the
# config_file fallback reads. A directory that is merely present is not a checkpoint.
[[ -f "${BASE_MODEL}/config.json" ]] || {
  echo "FATAL: ${BASE_MODEL}/config.json missing — that is the HF download from 91_stage, not the"
  echo "       DCP tree at ${BASE_CHECKPOINT_PATH}. Inference cannot read the DCP tree; see the"
  echo "       BASE_MODEL comment above."; exit 1; }
[[ -f "${OUT}/run_metadata.json" ]] || { echo "FATAL: no run_metadata.json — the run is not scorable."; exit 1; }
[[ -d "${LORA_MODEL}" ]] || {
  echo "FATAL: ${LORA_MODEL} missing. Export the adapter first:"
  echo "  CKPT=${RUN_DIR}/checkpoints/\$(cat ${RUN_DIR}/checkpoints/latest_checkpoint.txt)"
  echo "  python -m cosmos_framework.scripts.export_model --checkpoint-path \$CKPT \\"
  echo "      --config-file ${RUN_DIR}/config.yaml -o ${LORA_MODEL}"
  exit 1; }
# Checked here and not at its point of use in step 3: nothing before the blinding needs the
# calibration footage, but the check used to sit AFTER both arms had generated 60 clips on 8 GPUs
# — so a directory that was never shipped cost the entire generation budget to discover, and the
# job then died holding 60 clips and no verdict. The same two stats cost nothing here.
[[ -d "${CAL}/positive" && -d "${CAL}/negative" ]] || {
  echo "FATAL: ${CAL} must hold positive/ (real held-out G1+Dex3 clips) and negative/ (real"
  echo "       non-G1 manipulator clips, e.g. BridgeData2's WidowX). Without it G0b cannot run"
  echo "       and PR-09 §6 issues no verdict."; exit 1; }
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

# ONE torchrun PER ARM, NOT ONE PER CLIP, and the reason is now measured instead of guessed. This
# loop used to launch torchrun once per payload — 30 per arm, 60 in all. The PR-09 §7 probe (job
# 186663, PROBE.json) separated the constant from the per-iteration cost by subtraction across a
# 5-iteration and a 25-iteration run and reported load_seconds = 494.8: 8.25 MINUTES of cold start
# (imports, an 8-rank NCCL world, a 64.6 GB DCP read) before the first denoising step. 60 × 494.8 s
# = 8.25 h of checkpoint loading alone, inside a job whose wall is 4 h, with generation still to
# pay on top. No number of requeues fixes a per-pass cost that is dominated by a constant repeated
# 60 times: each restart only saves the clips the previous one finished, so the job would need
# ~10 resubmissions to grind through well under an hour of actual sampling. 60 loads become 2,
# ≈16.5 min, and that is an UPPER bound — 494.8 s was a *training* load (model + optimiser + EMA
# shards) and inference reads only <ckpt>/model (utils/generator/model_loader.py:380-382).
#
# THE BATCHED PATH IS READ OUT OF THE FRAMEWORK, NOT ASSUMED. `-i` is a list and its docstring
# states the property this depends on — "the model will be loaded once and all the samples will be
# run sequentially" (scripts/inference.py:22-27). Under parallelism_preset = "throughput" the eight
# ranks now hold eight DIFFERENT samples instead of duplicating one: cp = cfgp = 1
# (inference/args.py:1364-1378) makes replica_size = 1, so num_replicas = world_size = 8
# (inference/inference.py:1966-1980) and create_batches mod-shards the sample list across ranks
# (inference/inference.py:1410). Short ranks are padded with a discarded 1-step dummy batch
# (inference/inference.py:1471-1486) and the per-step collectives are held in lockstep by the
# dp_shard all-reduce at inference/inference.py:1648-1676, which the framework wrote for exactly
# this case. 30 clips per arm therefore cost ceil(30/8) = 4 sequential generations per rank, not 30
# — the ~8× the "throughput" preset always promised and that one-payload-per-launch threw away.
# max_num_seqs defaults to 1 (inference/common/args.py:820), so a rank still samples one clip at a
# time: this buys throughput, not per-rank activation memory, and does not change what fits in VRAM.
#
# THE ARMS STAY TWO PROCESSES. Each launch carries its own --checkpoint-path, so no rank ever holds
# base weights while writing a lora clip. That is why this is two loads and not one; merging them
# would save ~8 min and cost the only property the paired test cannot survive losing.

# RESTART-SAFETY, MOVED RATHER THAN LOST. The old loop skipped a clip whose .mp4 was already on
# disk, one clip at a time. A batched launch cannot skip mid-flight, so the skip moves into the
# *input list*: harvest() files everything already finished and leaves the rest in the missing
# list, and only that list is handed to torchrun. A requeue regenerates exactly what is absent.
#
# harvest() also rescues clips the framework finished but the job died before they were filed, and
# it uses the framework's own completion marker to do it: sample_outputs.json is written strictly
# after `assert vision_file.is_file()` (inference/inference.py:1752-1784), so mp4-plus-marker means
# the encode returned, while an mp4 truncated by a SIGKILL has no marker beside it. The marker is
# also written with status "error" by _handle_sample_exception (inference/inference.py:1953), which
# is why the status is read and not merely the file counted.
harvest() {  # payload-dir  raw-dir  dest-dir  missing-list-to-write
python - "$1" "$2" "$3" "$4" <<'PY'
import json, pathlib, shutil, sys
pay, raw, dest, out = (pathlib.Path(a) for a in sys.argv[1:5])
missing = []
for p in sorted(pay.glob("*.json")):
    if (dest / f"{p.stem}.mp4").is_file():
        continue
    # <-o>/<payload "name">/vision.mp4 — output_dir is the sample's own name
    # (scripts/inference.py:52) and the extension follows num_frames (inference/args.py:437-439).
    src, rec = raw / p.stem / "vision.mp4", raw / p.stem / "sample_outputs.json"
    ok = False
    if src.is_file() and rec.is_file():
        try:
            ok = json.loads(rec.read_text()).get("status") == "success"
        except ValueError:
            ok = False
    if ok:
        shutil.move(str(src), str(dest / f"{p.stem}.mp4"))
    else:
        missing.append(p.stem)
out.write_text("".join(f"{u}\n" for u in missing))
print(f"harvest: {len(list(dest.glob('*.mp4')))} clips filed, {len(missing)} to generate")
PY
}

for ARM in base lora; do
  CKPT=$([[ "${ARM}" == base ]] && echo "${BASE_MODEL}" || echo "${LORA_MODEL}")
  DEST=${EVAL}/clips/${ARM}
  RAW=${DEST}/_raw
  MISSING=${EVAL}/payloads/${ARM}.missing
  mkdir -p "${DEST}" "${RAW}" "${EVAL}/payloads/${ARM}"
  echo "=== generating arm ${ARM} from ${CKPT}"
  # One payload per prompt, built from the committed settings. Identical across arms by
  # construction: the same file writer, the same config, the same seed. A sampler difference
  # between arms would turn the paired test into a comparison of samplers.
  python - "${PROMPTS}" "${CONFIG}" "${EVAL}/payloads/${ARM}" <<'PY'
import json, pathlib, sys, tomllib
prompts, cfg_path, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
g = tomllib.loads(pathlib.Path(cfg_path).read_text())["generation"]
n = 0
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
    n += 1
# THE ONE-TO-ONE MAPPING IS ASSERTED HERE BECAUSE THE FRAMEWORK DOES NOT ASSERT IT. Batch mode
# derives each output directory from the payload's "name" (scripts/inference.py:52), so two
# payloads sharing a name would share an output directory and the second would silently overwrite
# the first — a clip filed under the wrong uuid, which is the one failure that produces a plausible
# verdict from shuffled labels. from_files LOOKS like it guards this ("Duplicate name" at
# inference/common/args.py:897) but the `names` set it tests against is never added to (:889), so
# at FRAMEWORK_SHA 12a9a81 the check cannot fire. Here it can: name == filename stem by
# construction, so a repeated uuid loses a file and this count catches it. A stale payload from a
# different prompt set — which harvest() would otherwise hand to the generator — trips it too.
written = len(list(dest.glob('*.json')))
assert written == n, f"{n} prompts but {written} payloads: duplicate uuid, or stale payloads in {dest}"
print("payloads:", written)
PY
  SEED=$(python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('${CONFIG}').read_text())['generation']['seed'])")
  PRESET=$(python -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('${CONFIG}').read_text())['generation']['parallelism_preset'])")
  harvest "${EVAL}/payloads/${ARM}" "${RAW}" "${DEST}" "${MISSING}"
  if [[ -s "${MISSING}" ]]; then
    TODO=()
    while IFS= read -r U; do TODO+=("${EVAL}/payloads/${ARM}/${U}.json"); done < "${MISSING}"
    echo "=== arm ${ARM}: one load, ${#TODO[@]} clips"
    torchrun --nproc-per-node="${NPROC}" -m cosmos_framework.scripts.inference \
        --parallelism-preset="${PRESET}" -i "${TODO[@]}" -o "${RAW}" \
        --checkpoint-path "${CKPT}" --seed="${SEED}"
    # Second pass files what the launch produced. -o is the arm's own tree, so the mapping back to
    # clips/${ARM}/<uuid>.mp4 is a named lookup, not the find|head it replaces — that could return
    # any mp4 under the directory, and "any mp4, renamed to a uuid" is precisely how a correctly
    # named clip ends up holding the wrong prompt's frames.
    harvest "${EVAL}/payloads/${ARM}" "${RAW}" "${DEST}" "${MISSING}"
  else
    echo "=== arm ${ARM}: every clip already on disk — no checkpoint load"
  fi
  if [[ -s "${MISSING}" ]]; then
    echo "FATAL: arm ${ARM} still missing $(wc -l < "${MISSING}") clips after generation:"
    head -5 "${MISSING}"; exit 1
  fi
  echo "=== arm ${ARM}: $(find "${DEST}" -maxdepth 1 -name '*.mp4' | wc -l) clips"
done

# -- 3. blind ------------------------------------------------------------------------------------
# ${CAL} was checked in the preflight above, before a single GPU-second was spent generating.
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
