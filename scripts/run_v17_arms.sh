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

waited=0
while [[ "$(free_mib)" -lt "${MIN_FREE_MIB}" ]]; do
  if [[ "${waited}" -ge "${WAIT_MINUTES}" ]]; then
    echo "GIVING UP: ${MIN_FREE_MIB} MiB never became free within ${WAIT_MINUTES} min." >&2
    echo "  Nothing was measured. CPU is not an option here: 8.28 s/frame measured, Arm B ~77 h." >&2
    exit 4
  fi
  echo "waiting for GPU: $(free_mib) MiB free, need ${MIN_FREE_MIB} (${waited}/${WAIT_MINUTES} min)"
  sleep 300
  waited=$((waited + 5))
done
echo "=== GPU has $(free_mib) MiB free; starting. ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

measure() {  # <capture dir> <artifact stem>
  local cap="$1" stem="$2"
  local out="${V17}/EST_DRIFT-${stem}.json"
  if [[ -f "${out}" ]]; then echo "SKIP ${stem} (measured)"; return 0; fi
  echo "--- measuring ${stem} (${cap})"
  # exit 3 is "written but not gate-qualified", which is the expected state while
  # GATE_QUALIFIED is False. Only a 2 (fatal, nothing written) is a failure here.
  set +e
  .venv/bin/python scripts/measure_est_drift.py measure \
    --capture "${cap}" --estimators "${EST}" --arm both --out "${out}"
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 && ${rc} -ne 3 ]]; then
    echo "FATAL: ${stem} exited ${rc}; nothing was written." >&2
    exit "${rc}"
  fi
}

# --- V17 §5 C1 first: the outcome table reads the control before anything else. ----------------
measure "${CONTROL_CAPTURE}" "C1-lattice"

# --- V17 §2 Arm A. ------------------------------------------------------------------------------
for id in A1 A2 A3 A4 A5 A6 A7 A8; do measure "${V17}/${id}" "${id}"; done

# --- V17 §5 C2, the dose ladder. Reported, never pooled. ----------------------------------------
for id in C2-t20 C2-t40 C2-t80; do measure "${V17}/${id}" "${id}"; done

# --- V17 §3 Arm B: the real corpus. -------------------------------------------------------------
if [[ ! -f "${V17}/ARM_DIVERGENCE.json" ]]; then
  echo "--- Arm B: 40 episodes, both arms, cross-arm divergence runs"
  set +e
  .venv/bin/python scripts/measure_arm_divergence.py \
    --corpus "${CORPUS}" --estimators "${EST}" --out "${V17}/ARM_DIVERGENCE.json"
  rc=$?
  set -e
  [[ ${rc} -eq 0 || ${rc} -eq 3 ]] || { echo "FATAL: Arm B exited ${rc}" >&2; exit "${rc}"; }
fi

# --- V17 §4: pool, and read the outcome. --------------------------------------------------------
.venv/bin/python scripts/pool_est_drift_arms.py \
  --artifact "${V17}"/EST_DRIFT-A[1-8].json \
  --control "${V17}/EST_DRIFT-C1-lattice.json" \
  --divergence "${V17}/ARM_DIVERGENCE.json" \
  --out "${V17}/POOLED.json"

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
