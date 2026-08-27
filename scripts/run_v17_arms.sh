#!/usr/bin/env bash
# T40_RULE_V17 — every measurement the protocol needs, in the order §4 reads them.
#
#   scripts/run_v17_arms.sh [--min-free-mib N] [--wait-minutes M]
#
# WHY THIS IS A SCRIPT AND NOT A COMMAND LIST IN A DOCUMENT. V17 fixes eight captures, three
# ladder captures, one control and one corpus sample, and every one of them must be measured with
# THE SAME estimator on THE SAME device — `pool_est_drift_arms.py` refuses a pool whose members
# disagree about the instrument, and a device switch halfway through is exactly the kind of
# disagreement that would not show up in the segmenter contract. Running them from one script is
# how that stays true.
#
# WHY IT WAITS FOR THE GPU. Measured on this workstation 2026-08-27: the adapter runs at 8.28 s
# per frame on CPU with 20 threads. Arm B alone is 33 692 frame-inferences, i.e. ~77 hours, so CPU
# is not a fallback for this work — it is a different project. The script therefore waits for
# headroom rather than quietly producing a number on the wrong device, and says what it is waiting
# for. It never kills, suspends or otherwise touches another process's memory.
#
# ONE OPERATIONAL TRAP, RECORDED HERE BECAUSE IT COSTS AN HOUR TO REDISCOVER. SAM 2's
# PositionEmbeddingSine warms a cache on CUDA whenever `torch.cuda.is_available()` is true, BEFORE
# `build_sam2` moves the model to the device it was asked for. So `WAM_PR08_DEVICE=cpu` alone still
# allocates on the GPU and dies with a CUDA OOM when the card is full. A CPU run needs
# `CUDA_VISIBLE_DEVICES=""` as well.
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_FREE_MIB=10000
WAIT_MINUTES=720
while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2;;
    --wait-minutes) WAIT_MINUTES="$2"; shift 2;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

V17=runs/pr08-est-drift/v17
CONTROL_CAPTURE=runs/pr08-est-drift/capture-mujoco-lattice-f60-control
CORPUS=/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless
EST=estimators.apple_sam2

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# THREE CONSECUTIVE SAMPLES, TWENTY SECONDS APART, AND THE REASON IS A FAILURE THIS SCRIPT ALREADY
# HAD. The first version tested `free_mib` once and started; it caught a dip while a neighbouring
# job was between allocations, launched, and died with a CUDA OOM 40 seconds later. A single sample
# of free memory is not a statement about the next ten minutes. Three agreeing samples is not one
# either, but it excludes the transient, and an OOM is treated as "go back to waiting" below rather
# than as a fatal — which is what actually makes this safe.
SAMPLES=3
SAMPLE_GAP=20

headroom_holds() {
  local i free
  for ((i = 0; i < SAMPLES; i++)); do
    free="$(free_mib)"
    if [[ "${free}" -lt "${MIN_FREE_MIB}" ]]; then
      echo "  sample $((i + 1))/${SAMPLES}: ${free} MiB free, need ${MIN_FREE_MIB}"
      return 1
    fi
    [[ $((i + 1)) -lt ${SAMPLES} ]] && sleep "${SAMPLE_GAP}"
  done
  return 0
}

waited=0
wait_for_gpu() {
  while ! headroom_holds; do
    if [[ "${waited}" -ge "${WAIT_MINUTES}" ]]; then
      echo "GIVING UP: ${MIN_FREE_MIB} MiB never held for ${SAMPLES} samples within ${WAIT_MINUTES} min." >&2
      echo "  Nothing further was measured. CPU is not an option: 8.28 s/frame, Arm B ~77 h." >&2
      exit 4
    fi
    echo "waiting for GPU (${waited}/${WAIT_MINUTES} min)"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    holder: /'
    sleep 300
    waited=$((waited + 5))
  done
  echo "=== GPU headroom held across ${SAMPLES} samples ($(free_mib) MiB free); starting. ==="
}

# Fragmentation is the other way a run this size dies on a shared card, and the allocator's own
# advice is the fix. Harmless when the card is empty.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_for_gpu

# AN OOM IS NOT A RESULT AND IS NOT A FATAL. Another process taking the card back mid-run says
# nothing about this measurement, so the step goes back to the wait loop and is retried rather than
# ending the protocol half-measured. Anything else — a missing capture, a refusing contract, an
# estimator that will not load — is a real refusal and stops everything, because retrying it would
# just produce the same refusal more slowly.
OOM_RETRIES=6

run_step() {  # <label> <command...>
  local label="$1"; shift
  local attempt=0 rc=0 log
  log="$(mktemp)"
  while :; do
    set +e
    "$@" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    # exit 3 is "written but not gate-qualified", the expected state while GATE_QUALIFIED is False.
    if [[ ${rc} -eq 0 || ${rc} -eq 3 ]]; then rm -f "${log}"; return 0; fi
    if grep -qiE "OutOfMemoryError|CUDA out of memory" "${log}"; then
      attempt=$((attempt + 1))
      if [[ ${attempt} -gt ${OOM_RETRIES} ]]; then
        echo "GIVING UP on ${label}: ${OOM_RETRIES} OOMs. The card is not free enough." >&2
        rm -f "${log}"; exit 4
      fi
      echo "--- ${label}: CUDA OOM (attempt ${attempt}/${OOM_RETRIES}). Back to waiting."
      wait_for_gpu
      continue
    fi
    echo "FATAL: ${label} exited ${rc}; nothing was written." >&2
    rm -f "${log}"; exit "${rc}"
  done
}

measure() {  # <capture dir> <artifact stem>
  local cap="$1" stem="$2"
  local out="${V17}/EST_DRIFT-${stem}.json"
  if [[ -f "${out}" ]]; then echo "SKIP ${stem} (measured)"; return 0; fi
  echo "--- measuring ${stem} (${cap})"
  run_step "${stem}" .venv/bin/python scripts/measure_est_drift.py measure \
    --capture "${cap}" --estimators "${EST}" --arm both --out "${out}"
}

# --- V17 §5 C1 first: the outcome table reads the control before anything else. ----------------
measure "${CONTROL_CAPTURE}" "C1-lattice"

# --- V17 §2 Arm A. ------------------------------------------------------------------------------
for id in A1 A2 A3 A4 A5 A6 A7 A8; do measure "${V17}/${id}" "${id}"; done

# --- V17 §5 C2, the dose ladder. Reported, never pooled. ----------------------------------------
for id in C2-t20 C2-t40 C2-t80; do measure "${V17}/${id}" "${id}"; done

# --- T40_RULE_V19 §3: C3, the control C1 could not be. -------------------------------------------
# C1 fired (10 runs) but its longest run was 5, which is the lattice's own period rather than a
# measurement of the statistic's sensitivity — a lattice control cannot produce a run longer than
# the lattice repeats. C3 seeds the propagation on the cube distractor's GROUND-TRUTH box on frame
# 0 of A1, a coherent capture that makes exactly one revolution, so there is no periodic return to
# break the run. Its est_drift_p95_px is the cube-to-apple distance and means NOTHING; only
# low_iou_runs is read. The fire condition is V17 §5's, unchanged: n_runs >= 1 and longest_run >= 10.
if [[ ! -f "${V17}/EST_DRIFT-C3-wrongseed.json" ]]; then
  echo "--- V19: C3, propagation held on the wrong object over A1"
  WAM_PR08_CONTROL_SEED_FROM_CAPTURE="${V17}/A1" WAM_PR08_CONTROL_SEED_LABEL=cube \
  run_step "C3-wrongseed" .venv/bin/python scripts/measure_est_drift.py measure \
    --capture "${V17}/A1" --estimators "${EST}" --arm both \
    --propagation-module estimators.apple_sam2_video_wrongseed \
    --out "${V17}/EST_DRIFT-C3-wrongseed.json"
fi

# --- V17 §3 Arm B: the real corpus. -------------------------------------------------------------
if [[ ! -f "${V17}/ARM_DIVERGENCE.json" ]]; then
  echo "--- Arm B: 40 episodes, both arms, cross-arm divergence runs"
  run_step "Arm B" .venv/bin/python scripts/measure_arm_divergence.py \
    --corpus "${CORPUS}" --estimators "${EST}" --out "${V17}/ARM_DIVERGENCE.json"
fi

# --- V17 §4: pool, and read the outcome. --------------------------------------------------------
.venv/bin/python scripts/pool_est_drift_arms.py \
  --artifact "${V17}"/EST_DRIFT-A[1-8].json \
  --control "${V17}/EST_DRIFT-C1-lattice.json" \
  --divergence "${V17}/ARM_DIVERGENCE.json" \
  --out "${V17}/POOLED.json"

# --- T40_RULE_V18: the other precondition on GATE_QUALIFIED, and it is not this blocker's. -------
# Runs here because it needs the same GPU and the same adapter; it decides nothing, and V17
# outcome N would not flip the flag without it any more than this would without V17.
if [[ ! -f runs/pr08-operating-point/EPISODE_094_CENSUS.json ]]; then
  echo "--- V18: every frame of episode_000094, both decodes, which ones the filter refuses"
  run_step "V18 census" .venv/bin/python scripts/census_operating_point_episode.py \
    --episode episode_000094 \
    --corpus "${CORPUS}" \
    --corpus /home/humanoid/wam-t041/pr08-apple-640x480 \
    --out runs/pr08-operating-point/EPISODE_094_CENSUS.json
fi

echo
echo "=== the C2 ladder, reported and not pooled (V17 §5) ==="
.venv/bin/python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("runs/pr08-est-drift/v17").glob("EST_DRIFT-C2-*.json")):
    d = json.loads(p.read_text())
    cap, arm = d["capture"], d["arm_comparison"]["propagation"]
    print(f"{p.stem:>18}: median motion "
          f"{cap['temporal_coherence']['median_interframe_motion_px']:8.3f} px  "
          f"runs {arm['low_iou_runs']['n_runs']:3d}  longest {arm['low_iou_runs']['longest_run']:4d}  "
          f"p95 {arm['est_drift_p95_px']}")
PY
echo "ALL V17 MEASUREMENTS DONE"
