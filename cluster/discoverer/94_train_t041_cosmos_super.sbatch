#!/bin/bash
# Step 18 — T-041 / PR-09: LoRA SFT of Cosmos3-Super on G1 footage.
#
#   probe first, it is a GATE and not a warm-up:
#       T041_FREEZE_LIFTED="<reason>" PROBE=1 sbatch 94_train_t041_cosmos_super.sbatch
#   then:
#       T041_FREEZE_LIFTED="<reason>" sbatch 94_train_t041_cosmos_super.sbatch
#
# READ docs/preregistration/PR-09-cosmos-super-finetune.md FIRST. The verdict rule is §6 and lives
# in 95_eval_t041_embodiment.sbatch. Nothing in this file decides anything: it produces a LoRA and
# the record that makes it scorable.
#
# THE RECIPE IS NVIDIA'S AND THE COLD START USES ITS TOML BYTE-IDENTICAL. Our deltas are two, both
# printed into this log:
#
#   (a) THE RESUME TRAP — PR-09 §4a. The shipped TOML sets
#           checkpoint.keys_to_skip_loading = ["net_ema.", "lora_"]
#       which is CORRECT cold: the base checkpoint has no LoRA tensors and they must init fresh.
#       On a resume from checkpoints/iter_<N>/ the same line skips the LoRA tensors we just paid
#       for and re-initialises them — while optim/, scheduler/ and the iteration counter in
#       trainer/ ARE restored. The run would carry on from iteration 300 with a fresh adapter and
#       a stale optimiser, log plausible losses, and write a checkpoint that is not what its own
#       metadata says. Finite, plausible, correctly shaped, and invisible to any assertion about
#       shape or range — T-37's transposed-xmat failure mode, one layer up.
#       Handled below by generating a resume TOML with "lora_" removed, and DIFFING IT INTO THIS
#       LOG. A pass whose log has no diff is not a valid resume (PR-09 §6, G0c).
#
#   (b) --nproc_per_node and, in probe mode, max_iter/save_iter. Both recorded in the artifact.
#
# EIGHT GPUs EXIST ON EXACTLY ONE NODE. dgx1 has 8; dgx2 has 7 plus a separate gpu_biz. This job
# can only ever land on dgx1 and queues behind every other 4/7/8-GPU request. NPROC=4 is supported
# by the recipe README but is UNTESTED at this model size; if used, the artifact records it and
# the result is reported as a 4-GPU run, never as "the recipe".
#
#SBATCH --job-name=wam-t041-train
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208          # 8 GPUs x 26 threads — above that, billing runs out first
#SBATCH --gres=gpu:8
#SBATCH --mem=1800G                  # 8 x 257G is the billing-neutral ceiling; the node has ~2 TB
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t041.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
source "${PROJ}/cosmos_env.sh"

: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8
export PATH=${UV_INSTALL_DIR}:${FFMPEG_PREFIX}/bin:${PATH}
export OMP_NUM_THREADS=26
# Both are load-bearing and both come from launch_sft_vision_super.sh: the 32B backbone needs the
# host CUDA libs cleared and the expandable allocator to fit at all.
export LD_LIBRARY_PATH=""
export PYTORCH_ALLOC_CONF=expandable_segments:True
source "${FRAMEWORK}/.venv/bin/activate"
nvidia-smi -L

NPROC=${NPROC:-8}
PROBE=${PROBE:-0}
RUN_ID=${RUN_ID:-t041-super-lora}
OUT=${PROJ}/runs/${RUN_ID}
TOML_SRC=${RECIPE_DIR}/toml/sft_config/vision_sft_super.toml
# From the TOML's [job]: project=cosmos3, group=sft, name=vision_sft_super.
RUN_DIR=${OUT}/train/cosmos3/sft/vision_sft_super
PREREG=${PROJ}/wam/docs/preregistration/PR-09-cosmos-super-finetune.md
# PR-09 §7. 3 passes x 4 h x 8 GPUs = 96 GPU-h. A pre-registered term, not a convenience.
CEILING_GPU_H=${CEILING_GPU_H:-96}
# ONE number, used by both the probe gate and the runtime rail. It used to be MAX_RESTARTS=2
# tested with `>=`, which permits two passes while the log line printed "of 3" and the probe gate
# rejected anything needing more than 3 — three different budgets from one pre-registered term.
MAX_PASSES=${MAX_PASSES:-3}

for f in "${TOML_SRC}" "${PREREG}"; do
  [[ -f "${f}" ]] || { echo "FATAL: ${f} missing (run sync.sh / 90_build_cosmos_env first)."; exit 1; }
done
# The old message here named 93_caption_corpus.sbatch, which is superseded and must NOT be run:
# it would re-caption the untranscoded tree. Preparation now happens on the workstation and the
# corpus is shipped with sync.sh --corpus, then registered by 92b_register_corpus.sbatch.
: "${DATASET_PATH:?no corpus registered — run sync.sh --corpus then sbatch 92b_register_corpus.sbatch}"
# Guarded explicitly rather than left to `set -u`: unset, this dies 150 lines later with a bare
# "unbound variable" inside a heredoc, after the 8 GPUs are already allocated.
: "${CORPUS_MANIFEST_SHA256:?no corpus manifest sha — run sbatch 92b_register_corpus.sbatch}"
: "${BASE_CHECKPOINT_PATH:?91_stage_cosmos_weights.sbatch has not run — no DCP checkpoint.}"
: "${WAN_VAE_PATH:?91_stage_cosmos_weights.sbatch has not run — no VAE.}"
[[ -f "${DATASET_PATH}/train/video_dataset_file.jsonl" ]] || {
  echo "FATAL: ${DATASET_PATH}/train/video_dataset_file.jsonl missing."; exit 1; }

echo "=== T-041 LoRA SFT, PR-09"
echo "=== model      ${MODEL_ID}@${MODEL_REVISION}"
echo "=== base DCP   ${BASE_CHECKPOINT_PATH}"
echo "=== corpus     ${DATASET_PATH}  manifest ${CORPUS_MANIFEST_SHA256}"
echo "=== recipe     cosmos@${COSMOS_SHA}  framework@${FRAMEWORK_SHA}"
echo "=== nproc      ${NPROC}$([[ ${NPROC} -ne 8 ]] && echo '   <-- NOT the tested 8-GPU config')"

mkdir -p "${OUT}"
cd "${FRAMEWORK}"

# -------------------------------------------------------------------------------------------
# PROBE — a gate. Two torchruns at different iteration counts, so the per-iteration cost is
# separated from the (large, one-off) DCP load by subtraction rather than by parsing NVIDIA's
# log format, which we would be guessing at. Each probe writes to a THROWAWAY output root: a
# 10-iteration checkpoint left where the real run looks for latest_checkpoint.txt would be
# resumed from, and the real run would silently start at iteration 10 of 500.
# -------------------------------------------------------------------------------------------
if [[ "${PROBE}" != "0" ]]; then
  # The restart rail below lives in the TRAIN section, which the probe never reaches — so a
  # preempted probe requeued itself onto 8 H200s with no bound at all. It also redid completed
  # work: PROBE.json already written means there is nothing left to measure.
  if [[ -f "${OUT}/PROBE.json" ]]; then
    echo "=== ${OUT}/PROBE.json already exists — probe already passed, nothing to measure:"
    cat "${OUT}/PROBE.json"
    exit 0
  fi
  PROBE_RESTARTS=${SLURM_RESTART_COUNT:-0}
  if (( PROBE_RESTARTS >= 2 )); then
    echo "FATAL: the probe has been requeued ${PROBE_RESTARTS} times without producing"
    echo "       ${OUT}/PROBE.json. Something is wrong with the measurement, not the queue."
    exit 1
  fi
  N1=${N1:-5}; N2=${N2:-25}
  declare -A WALL
  for N in "${N1}" "${N2}"; do
    PTOML=${OUT}/probe_${N}.toml
    sed -e "s/^max_iter .*/max_iter = ${N}/" \
        -e "s/^save_iter .*/save_iter = 1000000/" "${TOML_SRC}" > "${PTOML}"
    grep -E '^(max_iter|save_iter)' "${PTOML}"
    PROOT=${OUT}/probe_root_${N}
    rm -rf "${PROOT}"; mkdir -p "${PROOT}"
    echo "=== probe ${N} iters at $(date -Is)"
    S=${SECONDS}
    IMAGINAIRE_OUTPUT_ROOT="${PROOT}" torchrun --nproc_per_node="${NPROC}" \
        -m cosmos_framework.scripts.train --sft-toml="${PTOML}"
    WALL[${N}]=$((SECONDS - S))
    echo "=== probe ${N} iters: ${WALL[${N}]}s"
    rm -rf "${PROOT}"
  done

  python - "${WALL[${N1}]}" "${WALL[${N2}]}" "${N1}" "${N2}" "${NPROC}" "${CEILING_GPU_H}" \
           "${OUT}/PROBE.json" "${MAX_PASSES}" <<'PY'
import json, math, sys
t1, t2, n1, n2, nproc, ceiling, out, max_passes = (
    float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]), sys.argv[7], int(sys.argv[8]))
step = (t2 - t1) / (n2 - n1)
load = t1 - n1 * step
if step <= 0:
    sys.exit("FATAL: non-positive step time — the two probes did not separate. Rerun with a "
             "larger N2 before trusting any estimate.")
train_s = 500 * step
usable = 4 * 3600 - load          # each pass pays the DCP load again
if usable <= 0:
    sys.exit(f"FATAL: the {load:.0f}s checkpoint load leaves no room in a 4 h job. The recipe "
             "cannot be chained on this QoS; that is the finding.")
passes = math.ceil(train_s / usable)
gpu_h = passes * (load + min(usable, train_s)) / 3600 * nproc
report = {"seconds_per_iter": round(step, 2), "load_seconds": round(load, 1),
          "iters": 500, "passes_needed": passes, "nproc": nproc,
          "estimated_gpu_hours": round(gpu_h, 1), "ceiling_gpu_hours": ceiling,
          "max_passes": max_passes}
print(json.dumps(report, indent=2))
open(out, "w").write(json.dumps(report, indent=2) + "\n")
if passes > max_passes:
    sys.exit(f"PROBE FAILS: 500 iters needs {passes} passes of 4 h; PR-09 §7 allows {max_passes}. "
             "The ceiling is a pre-registered term — do not raise it to fit the recipe. "
             "Record the shortfall and take the pre-registered alternative (Cosmos3-Edge).")
if gpu_h > ceiling:
    sys.exit(f"PROBE FAILS: {gpu_h:.1f} GPU-h > the {ceiling} GPU-h ceiling. Same rule.")
print("PROBE PASSES — the run fits the pre-registered ceiling.")
PY
  echo "next: T041_FREEZE_LIFTED=... sbatch 94_train_t041_cosmos_super.sbatch   (no PROBE=)"
  exit 0
fi

# -------------------------------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------------------------------
[[ -f "${OUT}/PROBE.json" ]] || {
  echo "FATAL: no ${OUT}/PROBE.json. PR-09 §7 makes the probe a gate, not a warm-up: without a"
  echo "       measured step time the 96 GPU-h ceiling cannot be checked before spending it."
  echo "       Run: PROBE=1 sbatch 94_train_t041_cosmos_super.sbatch"
  exit 1
}
cat "${OUT}/PROBE.json"

# DONE is written at the end of a successful run but was never read: a resubmission after the run
# had already finished would load the checkpoint and train past 500. Cheapest possible check first.
if [[ -f "${OUT}/DONE" ]]; then
  echo "T-041 already complete — ${OUT}/DONE exists. Nothing to do."
  exit 0
fi

# Pass accounting is persisted, not inferred from SLURM_RESTART_COUNT alone. scontrol requeue keeps
# the job id and increments that counter, but a MANUAL resubmit after a walltime kill starts a new
# job at restart 0 — and manual resubmission is exactly what a 4 h cap on a multi-pass run
# produces. Counting only requeues means the pre-registered 3-pass ceiling silently never binds.
PASSES_LOG=${OUT}/passes.jsonl
mkdir -p "${OUT}"
PRIOR=$( [[ -f "${PASSES_LOG}" ]] && wc -l < "${PASSES_LOG}" || echo 0 )
PASS_NO=$((PRIOR + 1))
RESTARTS=${SLURM_RESTART_COUNT:-0}
if (( PASS_NO > MAX_PASSES )); then
  echo "FATAL: ${PRIOR} passes already recorded in ${PASSES_LOG} and still no ${OUT}/DONE —"
  echo "       stopping at the PR-09 §7 ceiling of ${CEILING_GPU_H} GPU-h (${MAX_PASSES} passes)."
  echo "       If the recipe genuinely needs more, that is a finding to record before the number"
  echo "       is raised. Inspect ${PASSES_LOG} and the earlier ${PROJ}/logs/t041.*.out first."
  exit 1
fi
echo "=== pass ${PASS_NO} of ${MAX_PASSES} (slurm restarts=${RESTARTS}) on $(hostname) at $(date -Is)"

LATEST=${RUN_DIR}/checkpoints/latest_checkpoint.txt
if [[ -f "${LATEST}" ]]; then
  ITER=$(cat "${LATEST}")
  export BASE_CHECKPOINT_PATH=${RUN_DIR}/checkpoints/${ITER}
  [[ -d "${BASE_CHECKPOINT_PATH}" ]] || {
    echo "FATAL: ${LATEST} names ${ITER}, which does not exist."; exit 1; }
  # --- PR-09 §4a. The one edit, and it is shown. ---
  TOML=${OUT}/vision_sft_super.resume.toml
  # The comment deliberately avoids naming the adapter prefix: the guard below greps this very
  # line for it, and a comment mentioning it would trip the guard on itself.
  sed 's/^keys_to_skip_loading .*/keys_to_skip_loading = ["net_ema."]                # PR-09 §4a: the adapter is NOT skipped on resume/' \
      "${TOML_SRC}" > "${TOML}"
  echo "=== RESUME from ${ITER}; TOML delta against the shipped recipe:"
  diff -u "${TOML_SRC}" "${TOML}" || true
  RESUMED=true
  RESUME_DIFF_LOGGED=true
  RESUMED_FROM=${ITER}
  # Scoped to the one line. "lora_" legitimately appears elsewhere — [optimizer].keys_to_select =
  # ["lora_"] is what restricts training to the adapter, and removing it would train the whole 32B
  # tower. Grepping the file as a whole would either always fire or hide that distinction.
  if grep -q '^keys_to_skip_loading .*lora_' "${TOML}"; then
    echo "FATAL: the resume TOML still skips lora_ — the sed did not match. Refusing to run:"
    echo "       this would reinitialise the adapter while keeping the optimiser state (PR-09 §4a)."
    exit 1
  fi
  if ! grep -q '^keys_to_skip_loading ' "${TOML}"; then
    echo "FATAL: no keys_to_skip_loading line in the resume TOML — the recipe changed shape."
    exit 1
  fi
  # The complement: the optimizer must STILL select only the adapter, or this stops being a LoRA
  # run and becomes a 32B full fine-tune that happens to fit until it does not.
  if ! grep -q '^keys_to_select .*lora_' "${TOML}"; then
    echo "FATAL: [optimizer].keys_to_select no longer restricts training to lora_ tensors."
    exit 1
  fi
else
  TOML=${TOML_SRC}
  echo "=== COLD START from ${BASE_CHECKPOINT_PATH}; TOML is the shipped recipe, unmodified:"
  echo "    $(sha256sum "${TOML_SRC}")"
  RESUMED=false
  # A cold start has nothing to diff: the shipped TOML is used byte-identical, which is the point.
  # G0c asks whether every pass THAT RESUMED logged its diff, so this pass is vacuously fine.
  RESUME_DIFF_LOGGED=true
  RESUMED_FROM=none
fi

# One line per pass, written BEFORE the trainer starts. If this pass is SIGKILLed at the wall the
# record still exists and still counts against MAX_PASSES — the whole point of persisting it.
python - "${PASSES_LOG}" "${PASS_NO}" "${RESTARTS}" "${SLURM_JOB_ID:-none}" "${RESUMED}" \
         "${RESUMED_FROM}" "${RESUME_DIFF_LOGGED}" "${TOML}" "${NPROC}" <<'PY'
import datetime, hashlib, json, pathlib, sys
log, pass_no, restarts, jobid, resumed, resumed_from, diff_logged, toml, nproc = sys.argv[1:10]
sha = hashlib.sha256(pathlib.Path(toml).read_bytes()).hexdigest()
with open(log, "a") as f:
    f.write(json.dumps({
        "pass": int(pass_no), "slurm_restart_count": int(restarts), "job_id": jobid,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": __import__("socket").gethostname(),
        "resumed": resumed == "true", "resumed_from": resumed_from,
        "resume_diff_logged": diff_logged == "true",
        "toml_sha256": sha, "nproc_per_node": int(nproc),
    }) + "\n")
PY
echo "=== recorded pass ${PASS_NO} in ${PASSES_LOG}"

START=${SECONDS}
# NEVER run the trainer in the foreground here. A foreground torchrun makes the 4-hour wall a DEAD
# END rather than a link in the chain: --requeue covers preemption and node failure, but a job that
# simply runs out of walltime is killed outright, so the `scontrol requeue` at the bottom of this
# file is never reached and the run stops one pass in. PR-09 §7 budgets three passes; without this
# the second one only ever happens because a human noticed. --signal=B:USR1@300 buys 5 minutes,
# and the trap only fires while the shell is sitting in `wait`. Same pattern as 50_train_t16.sbatch.
IMAGINAIRE_OUTPUT_ROOT="${OUT}/train" torchrun --nproc_per_node="${NPROC}" \
    -m cosmos_framework.scripts.train --sft-toml="${TOML}" &
PY=$!

# The Cosmos trainer documents no SIGUSR1 handler, so there is no graceful "stop at a step
# boundary" to ask for. SIGTERM is the honest option: it costs at most the iterations since the
# last save_iter=100 checkpoint — the same 99 the requeue message below already accounts for —
# and it buys an orderly requeue instead of a SIGKILL that ends the chain.
GOT_USR1=0
trap 'GOT_USR1=1; echo "=== USR1 at $(date -Is): 5 min to the wall. Terminating pid ${PY} so this
      pass requeues instead of being killed."; kill -TERM "${PY}" 2>/dev/null || true' USR1

# A trap makes `wait` return early (128+signal) with the child still alive, and `set -e` would
# abort on that status — hence the loop and the `&& / ||`.
RC=0
while kill -0 "${PY}" 2>/dev/null; do
  wait "${PY}" && RC=0 || RC=$?
done
ELAPSED=$((SECONDS - START))
echo "trainer pid ${PY} exited ${RC} after ${ELAPSED}s at $(date -Is) (usr1=${GOT_USR1})"

REACHED=$( [[ -f "${LATEST}" ]] && cat "${LATEST}" || echo none )
echo "=== latest checkpoint: ${REACHED}"

# 500 is [trainer].max_iter in the shipped TOML; the counter lives in the DCP's trainer/ state,
# so this is the run's own report of where it got to, not an inference from wall time.
if [[ "${REACHED}" == *000000500 ]]; then
  # The adapter export used to be printed as instructions for a human to run. That left
  # 95_eval's G0c key `export_nonempty` with nobody to write it, so the verdict gate could never
  # pass however good the run was. Do the export here and report what actually happened.
  CKPT=${RUN_DIR}/checkpoints/${REACHED}
  EXPORTED=${RUN_DIR}/model
  echo "=== exporting adapter from ${CKPT}"
  set +e
  python -m cosmos_framework.scripts.export_model \
      --checkpoint-path "${CKPT}" \
      --config-file "${RUN_DIR}/config.yaml" \
      -o "${EXPORTED}"
  EXPORT_RC=$?
  set -e
  if [[ ${EXPORT_RC} -eq 0 ]] && [[ -d "${EXPORTED}" ]] && \
     [[ -n "$(find "${EXPORTED}" -type f -size +0 -print -quit 2>/dev/null)" ]]; then
    EXPORT_NONEMPTY=true
  else
    EXPORT_NONEMPTY=false
  fi
  echo "=== export rc=${EXPORT_RC} nonempty=${EXPORT_NONEMPTY} -> ${EXPORTED}"

  python - "${OUT}" "${MODEL_ID}" "${MODEL_REVISION}" "${COSMOS_SHA}" "${FRAMEWORK_SHA}" \
           "${CORPUS_MANIFEST_SHA256}" "${NPROC}" "${RUN_DIR}" "${T041_FREEZE_LIFTED}" \
           "${REACHED}" "${PASSES_LOG}" "${EXPORT_NONEMPTY}" <<'PY'
import json, re, sys, pathlib
(out, model, rev, csha, fsha, corpus, nproc, run_dir, freeze,
 reached, passes_log, export_nonempty) = sys.argv[1:13]

# 95_eval_t041_embodiment.py's G0c compares iteration_reached to N_MCNEMAR_ITERS as an INT.
# "iter_000000500" is a string and would never equal 500, so parse the digits out.
m = re.search(r"(\d+)\s*$", reached)
iteration_reached = int(m.group(1)) if m else None

# G0c's resume_diffs_logged is only meaningful across the whole chain: one pass that resumed
# without printing its TOML diff is enough to make the adapter suspect (PR-09 §4a), even if
# every other pass was clean. Compute it from the persisted record, not from this pass.
passes = []
p_log = pathlib.Path(passes_log)
if p_log.is_file():
    passes = [json.loads(l) for l in p_log.read_text().splitlines() if l.strip()]
resumes = [p for p in passes if p.get("resumed")]
resume_diffs_logged = bool(passes) and all(p.get("resume_diff_logged") for p in resumes)

p = pathlib.Path(out) / "run_metadata.json"
p.write_text(json.dumps({
    "task": "T-041", "prereg": "docs/preregistration/PR-09-cosmos-super-finetune.md",
    "rule": "T041_RULE_V1", "model_id": model, "model_revision": rev,
    "cosmos_sha": csha, "framework_sha": fsha, "corpus_manifest_sha256": corpus,
    "nproc_per_node": int(nproc), "run_dir": run_dir,
    "freeze_lifted_reason": freeze,
    # --- the three keys 95_eval's G0c reads. Absent, the verdict can never pass. ---
    "iteration_reached": iteration_reached,
    "resume_diffs_logged": resume_diffs_logged,
    "export_nonempty": export_nonempty == "true",
    # --- the per-pass record PR-09 §6 requires regardless of verdict ---
    "passes": passes,
    "pass_count": len(passes),
    "recipe_toml": "vision_sft_super.toml, byte-identical on cold start; resume copy differs "
                   "only in checkpoint.keys_to_skip_loading (PR-09 §4a), diffed into the job log",
}, indent=2) + "\n")
print("wrote", p)
print(f"  iteration_reached={iteration_reached} resume_diffs_logged={resume_diffs_logged} "
      f"export_nonempty={export_nonempty} passes={len(passes)}")
PY
  touch "${OUT}/DONE"
  if [[ "${EXPORT_NONEMPTY}" != "true" ]]; then
    echo "TRAINING COMPLETE but the adapter export FAILED (rc=${EXPORT_RC}). The checkpoint at"
    echo "${CKPT} is intact and DONE is written, so nothing needs re-training — but 95_eval's G0c"
    echo "will refuse this run until the export succeeds. Fix the export and re-run this job;"
    echo "it will exit early on DONE, so re-export by hand:"
    echo "  python -m cosmos_framework.scripts.export_model --checkpoint-path ${CKPT} \\"
    echo "      --config-file ${RUN_DIR}/config.yaml -o ${EXPORTED}"
    exit 1
  fi
  echo "T-041 TRAINING COMPLETE — adapter exported to ${EXPORTED}"
  echo "next: sbatch 95_eval_t041_embodiment.sbatch"
  exit 0
fi

if [[ "${RC}" -eq 0 && "${GOT_USR1}" -eq 0 && "${ELAPSED}" -lt 300 ]]; then
  echo "FATAL: exited 0 after ${ELAPSED}s without reaching iteration 500 — that is a no-op, not"
  echo "       an interrupted slice. Refusing to requeue."
  exit 1
fi
# A non-zero RC means "requeueing only rebuilds the same crash" ONLY when the trainer chose to
# exit. When we sent the SIGTERM ourselves, the non-zero status is OUR doing (143) and says
# nothing about the run's health — treating it as a crash here would turn the walltime handling
# above into an elaborate way of stopping after one pass.
if [[ "${RC}" -ne 0 && "${GOT_USR1}" -eq 0 ]]; then
  echo "FATAL: trainer failed with ${RC} — requeueing only rebuilds the same crash."
  exit "${RC}"
fi
echo "not at 500 yet (${REACHED}) -> requeue ${SLURM_JOB_ID}; resumes from the last save_iter=100"
echo "checkpoint, so at most 99 iterations are lost. The next pass MUST log a TOML diff."
scontrol requeue "${SLURM_JOB_ID}"
