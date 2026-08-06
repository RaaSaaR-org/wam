#!/usr/bin/env bash
# scripts/run_i8_rung_local.sh — one I-8 / T-32 rung on a LOCAL single-GPU box.
#
# The local twin of cluster/discoverer/55_train_i8_rung.sbatch. Same experiment, same splits,
# same step budget, same effective batch — a different machine, and therefore a different
# training config (configs/training/joint_wan_gr00t_5090.yaml, sized for 32 GiB; read its
# header for the memory arithmetic).
#
#   RUNG=040 ./scripts/run_i8_rung_local.sh
#   RUNG=120 ./scripts/run_i8_rung_local.sh
#   RUNG=040 SEED=1 ./scripts/run_i8_rung_local.sh                # reproducibility replicate
#   RUNG=040 STEPS=2290 RUN_ID=i8-rung040-e17-seed0 ./scripts/run_i8_rung_local.sh
#                                                                 # stage 2, equal-EPOCH control
#
# Rung 362 is NOT run here: it is runs/t16-lora-seed0 (same config geometry, same 362
# episodes, same 20 000 steps, same seed 0).
#
# FIRST TIME ON THIS BOX, DO THIS FIRST:
#   RUNG=040 PREFLIGHT=1 ./scripts/run_i8_rung_local.sh
# 20 steps into a throwaway directory. It is the only way to turn the config's ESTIMATED
# activation term into a number measured on your card — watch `nvidia-smi --query-gpu=\
# memory.used --format=csv -l 1` in another shell while it runs. If the peak lands near the
# 27.7 GB the config predicts for BATCH=2, you can go faster with BATCH=4 ACCUM=2 (or
# BATCH=8 ACCUM=1, which is exactly what the H200 ran). Effective batch stays 8 either way.
#
# HOW LONG THIS TAKES — one MEASUREMENT, one ASSUMPTION, and the assumption is the big one.
#
#   MEASURED: 0.42 s/step on one H200 at batch 8, grad_accum 1 (runs/_slurm_logs/t16.183601.out;
#             the log's own step 10 -> step 20 window is 0.40 s/step).
#
#   ASSUMPTION (NOT a measurement — there is no 5090 to measure on): the 5090 is 2.0x slower
#   per step than the H200. That number is picked inside a band whose two ends are real:
#     - lower end 1.0x. At 9 frames of 128x160 the DiT sequence is 60 tokens. A step is 30
#       blocks x 12 LoRA-wrapped linears of almost nothing, twice over (gradient checkpointing
#       recomputes the forward), so it is launch-latency bound, and launch latency tracks
#       clocks — where the 5090 (~2.4 GHz boost) is not behind an H200 (~1.98 GHz).
#     - upper end 2.68x. If instead the step is bound by streaming the frozen weights, it
#       tracks memory bandwidth: 4.8 TB/s HBM3e vs 1.79 TB/s GDDR7.
#     The fp32 VAE encode, which the 5090 has far less TF32/fp32 tensor throughput for, pushes
#     toward the upper end. 2.0x sits between them; nothing here is measured.
#
#   ASSUMPTION: +25 % for splitting one update into 4 micro-batches (per-pass fixed cost is
#   paid 4x instead of 1x, and the step looks launch-bound). -> 0.42 * 2.0 * 1.25 = 1.05 s/step.
#
#   Two remaining stage-1 rungs, 20 000 steps each, at BATCH=2 ACCUM=4:
#       rung 040   5.8 h        rung 120   5.8 h        TOTAL ~11.7 h
#   Band from the two ends of the assumption: 4.7 h (1.0x, no split penalty) to 15.7 h (2.68x).
#   At BATCH=8 ACCUM=1, if the preflight says it fits: ~4.7 h per rung, ~9.3 h total.
#   The stage-2 equal-epoch controls (2 290 + 6 748 steps) add ~2.6 h on top.
#   Model load is ~5 min per process start (H200 log: 23:34:35 -> first step 23:39:43),
#   which matters only if the run keeps getting interrupted.
#
# INTERRUPTION. train_t16_lora.py stops at a STEP BOUNDARY on SIGUSR1/SIGTERM/SIGINT,
# checkpoints, and exits 0; --resume latest picks that checkpoint back up. So:
#   - Ctrl-C is safe and does NOT lose the interval since the last checkpoint. This script
#     will not restart after you interrupt it on purpose.
#   - `kill -USR1 <pid of python>` forces a checkpoint-and-stop without killing the box.
#   - A crash, an OOM or a reboot: just run the same command again. The loop below re-invokes
#     with --resume latest until ${OUT}/DONE exists, capped by MAX_PASSES.
#   - Resume takes the CHECKPOINT's config verbatim, so editing the yaml mid-chain is
#     reported and ignored. --grad-accum is NOT in that config, which is why this script
#     pins it in ${OUT}/local_run.env and refuses to resume with a different value.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${WAM_PYTHON:-${REPO_ROOT}/.venv/bin/python}
TRAIN=${REPO_ROOT}/scripts/train_t16_lora.py

# -- the rung ---------------------------------------------------------------------------------
RUNG=${RUNG:?set RUNG to the rung size, zero-padded to 3 digits (e.g. RUNG=040)}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
RUN_ID=${RUN_ID:-i8-rung${RUNG}-seed${SEED}}

# -- what fits on this box --------------------------------------------------------------------
# BATCH * ACCUM must be 8: that is the effective batch runs/t16-lora-seed0 trained at and the
# STAGE1_BATCH tests/test_splits.py pins the stage-2 epoch budgets against. train_t16_lora.py
# scales the accumulated loss by 1/len(batches) (:774, :779), so 2x4 is arithmetically the
# same update as 1x8 — only the peak activation differs.
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-4}
EFFECTIVE=$((BATCH * ACCUM))

TRAINING_CONFIG=${TRAINING_CONFIG:-${REPO_ROOT}/configs/training/joint_wan_gr00t_5090.yaml}
BACKBONE_CONFIG=${BACKBONE_CONFIG:-${REPO_ROOT}/configs/model/wan22_ti2v_5b.yaml}
BACKBONE_SOURCE=${BACKBONE_SOURCE:-${WAM_MODELS:-${HOME}/models}/Wan2.2-TI2V-5B}
DATASET=${DATASET:-${REPO_ROOT}/datasets/gr00t-apple-full}
SPLIT=${REPO_ROOT}/configs/splits/i8_train_${RUNG}.txt
HOLDOUT=${REPO_ROOT}/configs/splits/t18_holdout_episodes.txt
OUT=${OUT:-${REPO_ROOT}/runs/${RUN_ID}}

MAX_HOURS=${MAX_HOURS:-24}      # no 4 h scheduler cap here; one pass should finish the rung
MAX_PASSES=${MAX_PASSES:-24}    # same runaway rail as the sbatch's MAX_RESTARTS
OFFLOAD_TEXT=${OFFLOAD_TEXT:-1} # helpful, not required — see the config header
PREFLIGHT=${PREFLIGHT:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${PREFLIGHT}" == "1" ]]; then
  STEPS=${PREFLIGHT_STEPS:-20}
  OUT="${OUT}-preflight"
  MAX_PASSES=1
  echo "=== PREFLIGHT: ${STEPS} steps into ${OUT}; nothing here belongs in a curve"
fi

# -- rails --------------------------------------------------------------------------------------
if (( EFFECTIVE != 8 )) && [[ "${ALLOW_BATCH_DRIFT:-0}" != "1" ]]; then
  echo "FATAL: BATCH(${BATCH}) * ACCUM(${ACCUM}) = ${EFFECTIVE}, not 8."
  echo "       Rung 362 (runs/t16-lora-seed0) trained at effective batch 8. A rung trained at"
  echo "       a different effective batch is not a point on the same curve. Pick a pair that"
  echo "       multiplies to 8 (2x4, 4x2, 8x1) or set ALLOW_BATCH_DRIFT=1 and say why."
  exit 1
fi
if [[ ! -x "${PY}" ]]; then
  echo "FATAL: no interpreter at ${PY} — create the venv or set WAM_PYTHON."
  exit 1
fi
if [[ ! -f "${TRAIN}" ]]; then
  echo "FATAL: ${TRAIN} missing."
  exit 1
fi
if [[ ! -f "${SPLIT}" ]]; then
  echo "FATAL: ${SPLIT} missing. The rung files are COMMITTED artifacts —"
  echo "       regenerate with scripts/make_rung_splits.py and commit them."
  echo "       Do not generate them ad hoc; a rung chosen at run time is not reviewable."
  exit 1
fi
if [[ ! -d "${DATASET}" ]]; then
  echo "FATAL: dataset ${DATASET} missing (set DATASET=...)."
  exit 1
fi
if [[ ! -d "${BACKBONE_SOURCE}" && "${DRY_RUN}" != "1" ]]; then
  echo "FATAL: Wan weights not found at ${BACKBONE_SOURCE}."
  echo "       Point BACKBONE_SOURCE (or WAM_MODELS) at a local snapshot of"
  echo "       Wan-AI/Wan2.2-TI2V-5B-Diffusers. ~34 GB on disk, ~24 GB of host RAM to load:"
  echo "       device_map is unreachable from this entry point, so every tower is"
  echo "       materialised in RAM before it moves to the GPU."
  exit 1
fi

# Queue-order rail, carried over verbatim from 55_train_i8_rung.sbatch. Rung 362's headline
# number IS the T-29 frame-history eval; if that verdict has not been read, this experiment's
# premise is unknown. RESOLVED 2026-08-01 (job 184648): NOT "L1 CLEARED" — the real frame
# window moves t16-lora-seed0 from -32.45 % to -21.80 %, so L1 is still failed by 21.80 pp,
# the T-16 negative stands, and I-8 asks the same question. The guard now simply passes.
T29=${REPO_ROOT}/runs/t16-lora-seed0/eval-t29-history/bench.json
if [[ ! -f "${T29}" && "${SKIP_T29_CHECK:-0}" != "1" ]]; then
  echo "FATAL: ${T29} missing — I-8 consumes that file as rung 362. Read the T-29 verdict and"
  echo "       re-read docs/improvements.md I-8 first. SKIP_T29_CHECK=1 overrides ONLY if you"
  echo "       have read the verdict and it did not change the premise."
  exit 1
fi

mkdir -p "${OUT}"

# train_t16_lora.py takes --offload-text as of the local-GPU work. The probe stays anyway: this
# script also has to run against an older checkout (a bisect, a resumed run from a branch), and
# passing a flag argparse does not know is a hard exit before a single step runs.
#
# It runs HERE, above the stamp, because the stamp has to pin what actually reaches the command
# line. OFFLOAD_TEXT is the request; HAVE_OFFLOAD is the effect, and the probe can turn a 1 into
# a 0 without anyone typing anything different.
HAVE_OFFLOAD=0
if [[ "${OFFLOAD_TEXT}" == "1" ]]; then
  if "${PY}" "${TRAIN}" --help 2>/dev/null | grep -q -- "--offload-text"; then
    HAVE_OFFLOAD=1
  else
    echo "NOTE: this checkout's train_t16_lora.py has no --offload-text — running with umT5"
    echo "      resident. That is +11.36 GB of VRAM (~27.7 GB total at BATCH=2 instead of"
    echo "      ~15.9 GB). It still fits on paper. Set OFFLOAD_TEXT=0 to silence this."
  fi
fi

# --grad-accum is a runtime argument: it is NOT in the config the checkpoint stores and NOT in
# config_hash, so a resume with a different value would silently be a different experiment in
# the same directory. Pin it on the first pass, refuse to contradict it afterwards.
#
# OFFLOAD has all three properties too, and it is the easier one to flip by accident: it defaults
# to 1, it is settable from the environment, and the probe above can clear it on its own. A rung
# that trains its first 9k steps against a CPU-encoded text context and its last 11k against a
# GPU-encoded one is one experiment in the record and two in fact. Nothing downstream could
# detect it, so it is refused here.
ENV_FILE=${OUT}/local_run.env
STAMP="RUNG=${RUNG} SEED=${SEED} STEPS=${STEPS} BATCH=${BATCH} ACCUM=${ACCUM} OFFLOAD=${HAVE_OFFLOAD}"
if [[ -f "${ENV_FILE}" ]]; then
  PREV=$(cat "${ENV_FILE}")
  if [[ "${PREV}" != "${STAMP}" ]]; then
    echo "FATAL: ${OUT} was started as:  ${PREV}"
    echo "       this invocation is:     ${STAMP}"
    echo "       Resuming into it would mix two experiments. Use a different RUN_ID/OUT."
    exit 1
  fi
else
  printf '%s' "${STAMP}" > "${ENV_FILE}"
fi

# -- environment ---------------------------------------------------------------------------------
# Every cluster sbatch sets these two and nothing local ever did. expandable_segments keeps the
# allocator from fragmenting a budget that has ~4 GB of slack; PYTHONHASHSEED is part of what
# makes the run reproducible at all.
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# The weights are local; do not let a missing file turn into a silent Hub download.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

ARGS=(
  --backbone-config "${BACKBONE_CONFIG}"
  --backbone-source "${BACKBONE_SOURCE}"
  --training-config "${TRAINING_CONFIG}"
  --dataset "${DATASET}"
  --exclude-episodes "${HOLDOUT}"
  --train-episodes "${SPLIT}"
  --seed "${SEED}"
  --steps "${STEPS}"
  --batch-size "${BATCH}"
  --grad-accum "${ACCUM}"
  --camera ego
  --out-dir "${OUT}"
  --run-id "${RUN_ID}"
  --resume latest
  --checkpoint-every-min "${CHECKPOINT_EVERY_MIN:-30}"
  --checkpoints-total-limit 3
  --save-adapter-only
  --max-hours "${MAX_HOURS}"
  --device cuda
)
# Appended separately, not interpolated as an array: bash 3.2 (still the /bin/bash on macOS)
# treats an empty "${arr[@]}" as an unbound variable under `set -u`.
if (( HAVE_OFFLOAD == 1 )); then
  ARGS+=(--offload-text)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== dry run: resolve configs and splits, touch no weights and no GPU"
  exec "${PY}" "${TRAIN}" "${ARGS[@]}" --dry-run
fi

echo "=== I-8 rung ${RUNG} seed ${SEED} — ${STEPS} steps, batch ${BATCH} x accum ${ACCUM}"
echo "=== (effective ${EFFECTIVE}) on $(hostname) at $(date -Is)"
echo "=== config ${TRAINING_CONFIG}"
echo "=== split  ${SPLIT}"
echo "=== out    ${OUT}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "no nvidia-smi on PATH"

# -- the requeue chain, local edition -------------------------------------------------------------
STOP_REQUESTED=0
PASS=0
while (( PASS < MAX_PASSES )); do
  if [[ -f "${OUT}/DONE" ]]; then
    echo "I-8 rung ${RUNG} seed ${SEED} COMPLETE — ${OUT}/DONE present"
    exit 0
  fi
  PASS=$((PASS + 1))
  echo "=== pass ${PASS}/${MAX_PASSES} at $(date -Is)"
  START=${SECONDS}

  # Background + wait, the same shape 55_train_i8_rung.sbatch uses: a trap only fires while the
  # shell sits in `wait`, so the trainer must never run in the foreground here.
  #
  # The cluster twin's stdout survives as runs/_slurm_logs/*.out — that archive is where this
  # branch reads its own 0.42 s/step figure from. Locally there is no scheduler to capture it,
  # and a rung is hours long: without this the offload line, the instruction count, the step
  # timings and any OOM traceback exist only in terminal scrollback. Section 5 asks the first
  # rung to turn the estimated activation term into a measurement; this is where it lands.
  #
  # Process substitution, not `| tee`: a pipe would make $! the tee's pid and the USR1 below
  # would checkpoint nothing. `-u` because the trainer's stdout is no longer a tty, and a
  # block-buffered 6-hour log is not a log you can watch.
  PASS_LOG="${OUT}/train_pass$(printf '%02d' "${PASS}").log"
  echo "=== log    ${PASS_LOG}"
  "${PY}" -u "${TRAIN}" "${ARGS[@]}" > >(tee -a "${PASS_LOG}") 2>&1 &
  CHILD=$!
  GOT_SIGNAL=0
  # shellcheck disable=SC2064
  trap 'GOT_SIGNAL=1; STOP_REQUESTED=1;
        echo "signal at $(date -Is) -> forward to ${CHILD}, waiting for a checkpoint";
        kill -USR1 "${CHILD}" 2>/dev/null || true' INT TERM USR1

  # A trap makes `wait` return early (128+signal) with the child still alive, and `set -e`
  # would abort on that status — hence the loop until the pid is gone.
  RC=0
  while kill -0 "${CHILD}" 2>/dev/null; do
    wait "${CHILD}" && RC=0 || RC=$?
  done
  trap - INT TERM USR1
  ELAPSED=$((SECONDS - START))
  # Into the log as well as the terminal: the exit status and the wall clock are the two things
  # a post-mortem reads first, and they are produced by this shell, not by the child whose
  # stdout the tee captured.
  echo "trainer pid ${CHILD} exited ${RC} after ${ELAPSED}s at $(date -Is) (signalled=${GOT_SIGNAL})" |
    tee -a "${PASS_LOG}"

  if [[ -f "${OUT}/DONE" ]]; then
    echo "I-8 rung ${RUNG} seed ${SEED} COMPLETE — ${OUT}/DONE present"
    exit 0
  fi
  if (( STOP_REQUESTED == 1 )); then
    echo "stopped on request; state is in ${OUT}/checkpoints. Re-run the same command to"
    echo "continue from the last checkpoint."
    exit 0
  fi
  if (( RC == 0 && ELAPSED < 300 )); then
    # Exit 0, no DONE, never signalled, gone in under 5 minutes: the trainer did nothing.
    # Relaunching a silent no-op is the one path that spins without ever looking wrong.
    echo "FATAL: trainer exited 0 after only ${ELAPSED}s with no ${OUT}/DONE and no signal —"
    echo "       that is a no-op, not an interrupted slice. Refusing to relaunch."
    exit 1
  fi
  if (( RC != 0 )); then
    # Died on its own: config error, CUDA OOM, a rung file that overlaps the holdout.
    # Relaunching only rebuilds the same crash.
    echo "FATAL: trainer failed with ${RC} and was never signalled — refusing to relaunch."
    echo "       If this was a CUDA OOM: lower BATCH and raise ACCUM by the same factor"
    echo "       (BATCH=1 ACCUM=8 is the floor that still keeps effective batch 8), and"
    echo "       start a FRESH ${OUT} — batch_size lives in the checkpointed config, so a"
    echo "       resume will reuse the old one."
    exit "${RC}"
  fi
  echo "no ${OUT}/DONE -> relaunch and continue from the last checkpoint"
done

echo "FATAL: ${MAX_PASSES} passes and still no ${OUT}/DONE — stopping instead of looping."
exit 1
