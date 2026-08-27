#!/usr/bin/env bash
# T40_RULE_V17 — every measurement the protocol needs, in the order §4 reads them.
#
#   scripts/run_v17_arms.sh [--min-free-mib N] [--wait-minutes M]
#                           [--out-dir DIR] [--reuse-existing | --remeasure]
#
# WHY THERE IS AN --out-dir AND WHY AN EXISTING ARTIFACT IS NOT SILENTLY SKIPPED. Until 2026-08-27
# the output directory was a bare assignment and `measure()` returned 0 for any file that already
# existed, printing `SKIP <stem> (measured)`. Those two facts together have one failure mode, and it
# is the one that will actually happen: the day `apple_sam2.GATE_QUALIFIED` flips, somebody re-runs
# this script to obtain gate-qualified artifacts, all thirteen are skipped in about a second, the
# thirteen `gate_qualified: false` files stay exactly where they were, and the carry downstream then
# refuses with reasons that no longer describe reality. Nothing in the output says a stale artifact
# was preferred to a measurement. The remediation on record was "point it at a fresh directory",
# which was not possible: there was no flag and no env override.
#
# So: the directory is overridable (`--out-dir`, or `V17=` in the environment), and an artifact that
# is already on disk stops the script BEFORE the GPU wait rather than being skipped inside it. Which
# of the three exits to take — reuse what is there, measure over it, or write somewhere fresh — is
# the operator's call and is not guessable from the file system, so this script refuses and names
# the three rather than picking one.
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
# refuse | reuse | remeasure. The default is the only one that cannot silently produce the wrong
# artifact set: see the header. Both alternatives are things the operator says out loud.
ON_EXISTING=refuse
# `${V17:-...}` and not a bare assignment, so a fresh output directory is reachable from the
# environment as well as from the flag. The captures Arm A reads live under the SAME directory
# (`${V17}/A1` ... `${V17}/A8`), so pointing this at an empty directory moves the inputs too — that
# is what `--out-dir` refuses on below, rather than measuring nothing and reporting success.
V17="${V17:-runs/pr08-est-drift/v17}"
# V18's census, written by this script for the reason given at its call site. Overridable for the
# same reason as V17 and skipped under the same policy; it is a different artifact tree, so it gets
# its own variable rather than being wedged under V17.
CENSUS_OUT="${CENSUS_OUT:-runs/pr08-operating-point/EPISODE_094_CENSUS.json}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2;;
    --wait-minutes) WAIT_MINUTES="$2"; shift 2;;
    --out-dir) V17="$2"; shift 2;;
    --reuse-existing) ON_EXISTING=reuse; shift;;
    --remeasure) ON_EXISTING=remeasure; shift;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

CONTROL_CAPTURE=runs/pr08-est-drift/capture-mujoco-lattice-f60-control
CORPUS=/home/humanoid/wam-t041/pr08-apple-640x480-h264-lossless
EST=estimators.apple_sam2

ARM_A_IDS=(A1 A2 A3 A4 A5 A6 A7 A8)
LADDER_IDS=(C2-t20 C2-t40 C2-t80)

# The captures are INPUTS and are not written by this script. A directory chosen with --out-dir that
# does not hold them would measure nothing and still print ALL V17 MEASUREMENTS DONE, so it is
# refused here instead — before the GPU wait, and before anything is written.
for id in "${ARM_A_IDS[@]}" "${LADDER_IDS[@]}"; do
  [[ -d "${V17}/${id}" ]] || {
    echo "FATAL: ${V17}/${id} is not a directory. V17 §2's eight Arm A captures and §5's three" >&2
    echo "       ladder captures live UNDER the output directory, so --out-dir/V17 selects the" >&2
    echo "       INPUTS as well as the outputs. Nothing was measured. Either point this at the" >&2
    echo "       directory holding the captures, or render the captures into the new one first." >&2
    exit 2
  }
done

# Every artifact this script writes, in one list, so the pre-flight below and the steps themselves
# cannot disagree about what "already measured" means. Each entry carries the policy the STEPS apply
# to it, `keep|path` or `rewrite|path`, because the two are not the same question for the pre-flight
# either:
#
#   keep     `--reuse-existing` leaves this file on disk and hands it to whatever reads it next, so
#            whether it was written by the adapter running now is what decides whether reusing it is
#            defensible. This is twelve of the thirteen §4 measurements, C3, ARM_DIVERGENCE and the
#            V18 census.
#   rewrite  this script recomputes the file on every run whatever the mode, so the same comparison
#            decides nothing about it and reporting it as "agrees" or "stale" would be a claim about
#            a file that is about to be replaced. POOLED.json is the only one, and its pooling step
#            has no skip on purpose: a pool kept from a previous run would omit whichever capture
#            this run was just asked to measure.
#
# It stays in the existence list under either policy, because the default refusal is also a warning
# that something on disk is about to be overwritten.
ARTIFACTS=("keep|${V17}/EST_DRIFT-C1-lattice.json")
for id in "${ARM_A_IDS[@]}" "${LADDER_IDS[@]}"; do
  ARTIFACTS+=("keep|${V17}/EST_DRIFT-${id}.json")
done
ARTIFACTS+=(
  "keep|${V17}/EST_DRIFT-C3-wrongseed.json"
  "keep|${V17}/ARM_DIVERGENCE.json"
  "rewrite|${V17}/POOLED.json"
  "keep|${CENSUS_OUT}"
)

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

# -- what is already on disk, and whether it was measured by the adapter running now ---------------
#
# THE CASE THIS EXISTS FOR IS THE FLIP. `apple_sam2.GATE_QUALIFIED` is False today and every artifact
# in the directory was written while it was False; the whole point of re-running this script will be
# that it has become True. An artifact's own `gate_qualified` is therefore not a detail — it is the
# reason the re-run is happening, and a skip that ignores it hands the pool the exact files the
# re-run was meant to replace. So the pre-flight compares each existing artifact against the adapter
# THIS invocation would drive, and reports the comparison whatever the mode, because "the artifacts
# on disk disagree with the flag in the module" is the one fact that decides between reusing and
# re-measuring, and it is invisible from the file names.
#
# It runs BEFORE wait_for_gpu on purpose: the wait is up to twelve hours, and discovering the
# refusal after it is discovering it at the worst possible time. Fail closed, fail fast, fail cheap.
preflight_rc=0
set +e
.venv/bin/python - "${EST}" "${ARTIFACTS[@]}" <<'PY'
import importlib, json, pathlib, sys

spec = sys.argv[1]
entries = [tuple(a.split("|", 1)) for a in sys.argv[2:]]   # ("keep"|"rewrite", path)
paths = [pathlib.Path(p) for _policy, p in entries]
sys.path.insert(0, "scripts")
live = getattr(importlib.import_module(spec), "GATE_QUALIFIED", None)
print(f"=== pre-flight: {spec}.GATE_QUALIFIED is {live!r}; {len(paths)} artifacts expected ===")

# WHERE THE ADAPTER'S OWN FLAG IS RECORDED, AND IT IS NOT ONE PLACE. Each writer this script drives
# embeds `apple_sam2.stats()`'s adapter block under a key of its own, so a probe that reads one of
# them is silently vacuous for every artifact written by the others:
#
#   measure_est_drift.py             estimator_stats.adapter.gate_qualified  (the 13 EST_DRIFT-*)
#   measure_arm_divergence.py        estimators.gate_qualified               (ARM_DIVERGENCE.json)
#   census_operating_point_episode   estimator_stats.gate_qualified          (the V18 census)
#                                    estimator.gate_qualified                (its summary block)
#
# Verified against the files on disk 2026-08-27. Only the first was read until then, which made the
# check vacuous for exactly the V18 census — the OTHER precondition on GATE_QUALIFIED, and the one
# artifact here that is not an EST_DRIFT measurement. It reported "agrees" whatever its recorded
# flag said, so after the flip `--reuse-existing` would have kept a pre-flip census in silence:
# the defect this pre-flight exists to close, surviving inside the pre-flight.
ADAPTER_FLAG_PATHS = (
    ("estimator_stats", "adapter", "gate_qualified"),
    ("estimator_stats", "gate_qualified"),
    ("estimator", "gate_qualified"),
    ("estimators", "gate_qualified"),
)
MISSING = object()


def dig(doc, path):
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return MISSING
        node = node[key]
    return node


present, disagree = [], []
for policy, path in [(pol, pathlib.Path(p)) for pol, p in entries]:
    if not path.exists():
        continue
    present.append(path)
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:                      # a truncated artifact is not a measurement
        disagree.append(path)
        print(f"  UNREADABLE  {path}: {exc}")
        continue
    artifact_flag = doc.get("gate_qualified", "<absent>") if isinstance(doc, dict) else "<absent>"
    recorded = [v for v in (dig(doc, p) for p in ADAPTER_FLAG_PATHS) if v is not MISSING]
    # Two different questions, and both are reported. `recorded` is what the ADAPTER's flag was when
    # the file was written, i.e. the direct comparison against `live`. `artifact_flag` is whether the
    # ARTIFACT came out gate-qualified, which can be False for reasons that have nothing to do with
    # this flag (the committed GEOM_TOL document, for one) — so it is shown and only counted as
    # stale in the direction that bites: the module now says qualified and the file says it is not.
    if policy == "rewrite":
        # Not compared, and said rather than implied: this run replaces the file either way, so any
        # verdict about the instrument that wrote it would be a verdict about a file about to go.
        verdict = "rewritten"
    elif not recorded:
        # NOT "agrees". No flag was found anywhere the writers put one, so nothing was compared, and
        # a file whose instrument cannot be established is not a file --reuse-existing may hand on.
        verdict = "UNCHECKABLE"
        disagree.append(path)
    elif any(value != live for value in recorded) or (live is True and artifact_flag is False):
        verdict = "STALE"
        disagree.append(path)
    else:
        verdict = "agrees"
    shown = recorded if recorded else "<none recorded>"
    print(
        f"  {verdict:<11} {path}  "
        f"artifact.gate_qualified={artifact_flag!r}  adapter.gate_qualified={shown!r}"
    )

print(f"=== {len(present)} of {len(paths)} already exist; {len(disagree)} disagree with the adapter ===")
sys.exit(0 if not present else (11 if disagree else 10))
PY
preflight_rc=$?
set -e

case "${preflight_rc}" in
  0) ;;  # nothing on disk: the ordinary first run, and no decision to make.
  10|11)
    case "${ON_EXISTING}" in
      remeasure)
        echo "=== --remeasure: every artifact above will be MEASURED AGAIN and overwritten. ==="
        ;;
      reuse)
        if [[ "${preflight_rc}" == "11" ]]; then
          echo "REFUSING: --reuse-existing was passed, but the artifacts marked STALE or" >&2
          echo "  UNCHECKABLE above were NOT SHOWN to be written by the adapter this invocation" >&2
          echo "  drives — their recorded gate_qualified disagrees with ${EST}.GATE_QUALIFIED," >&2
          echo "  or they carry a false one the module no longer agrees with, or they record no" >&2
          echo "  adapter flag at all and so establish nothing. Reusing them would pool a" >&2
          echo "  measurement with a document that describes a different instrument, and" >&2
          echo "  pool_est_drift_arms cannot see the difference: its instrument key is the" >&2
          echo "  segmenter contract, the resolution, the object class, the propagator spec and" >&2
          echo "  the IoU threshold — the gate flag is not in it." >&2
          echo "  Nothing was measured. Pass --remeasure to overwrite them, or --out-dir DIR to" >&2
          echo "  write a fresh set beside them." >&2
          exit 5
        fi
        echo "=== --reuse-existing: the artifacts above are kept; only missing ones are measured."
        echo "    They agree with ${EST}.GATE_QUALIFIED, which is what makes reusing them"
        echo "    defensible. Nothing here says they are otherwise current."
        echo "    The one line marked 'rewritten' is the exception and is NOT kept: POOLED.json is"
        echo "    recomputed on every run, because a pool carried over from a previous one would"
        echo "    omit whichever capture this run was asked to measure."
        ;;
      *)
        echo "REFUSING TO SKIP: artifacts this script writes already exist under" >&2
        echo "  ${V17} (and ${CENSUS_OUT}); the ones listed above are on disk." >&2
        if [[ "${preflight_rc}" == "11" ]]; then
          echo "  At least one of them DISAGREES with ${EST}.GATE_QUALIFIED — that is the case" >&2
          echo "  this refusal exists for. A silent skip would keep exactly those files, and the" >&2
          echo "  carry downstream would then refuse for reasons that no longer describe reality." >&2
        fi
        echo "  A skipped artifact is not a measured one, and this script cannot decide for you" >&2
        echo "  which of the two you want. Say it out loud:" >&2
        echo "    --reuse-existing   keep what is on disk, measure only what is missing" >&2
        echo "    --remeasure        measure every step again, overwriting what is there" >&2
        echo "    --out-dir DIR      write a fresh set (the captures must live under DIR)" >&2
        echo "  Nothing was measured and nothing was written." >&2
        exit 5
        ;;
    esac
    ;;
  *)
    echo "FATAL: the pre-flight itself exited ${preflight_rc}, so what is on disk is UNKNOWN." >&2
    echo "  That is not permission to measure over it. Nothing was measured." >&2
    exit 2
    ;;
esac

# Fragmentation is the other way a run this size dies on a shared card, and the allocator's own
# advice is the fix. Harmless when the card is empty.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# WHICH FILES THE OPERATOR'S DECISION IS ABOUT, frozen here where the pre-flight's answer is still
# the current one. The pre-flight asks "what is on disk" ONCE and before the wait, which is right:
# a refusal found after twelve hours of waiting is found at the worst possible moment. But the skip
# below asks `-f` again at the FAR END of that wait, and the two questions have different answers
# whenever a file appears in between — a neighbouring operator's run, or a second copy of this
# script, on a workstation several sessions share. Such a file was never listed, never compared
# against ${EST}.GATE_QUALIFIED, and never decided about, and the skip would have kept it in the
# DEFAULT mode too (`!= remeasure` is true for `refuse`), printing `--reuse-existing` over a flag
# nobody passed. That is the same failure this whole pre-flight exists to stop, arriving through a
# twelve-hour window instead of through the directory's initial state, and it is invisible
# downstream: `pool_est_drift_arms`' instrument key is the segmenter contract, the resolution, the
# object class, the propagator spec and the IoU threshold — the gate flag is not in it.
#
# WHAT THIS IS NOT, so nobody reads a stronger guarantee out of it than it gives: this is not the
# pre-flight's own answer handed forward. It is a SECOND `-f` pass over the same candidate names,
# taken a few lines after the pre-flight returned and still before the wait. The two share the
# LIST, not the existence answer, so a file appearing in the sub-second gap between them is still
# admitted as "seen". Closing that would mean the pre-flight emitting its own listing for this loop
# to read; the gap is milliseconds against a window that can be twelve hours, so it is named here
# rather than engineered away. Existence, not content, either: a listed file REWRITTEN during the
# wait is still kept under `reuse`.
PREFLIGHT_SEEN=$'\n'
for entry in "${ARTIFACTS[@]}"; do
  if [[ -f "${entry#*|}" ]]; then PREFLIGHT_SEEN+="${entry#*|}"$'\n'; fi
done

# Only reached with a decision recorded: either nothing was on disk, or the operator named one.
already_measured() {  # <artifact path> -- true when this step may be skipped
  local out="$1"
  [[ -f "${out}" ]] || return 1
  # --remeasure is the one mode that KEEPS nothing, so a file that turned up in the window is a file
  # about to be overwritten and there is no decision left to make about it. The window bites only on
  # the keep path, and refusing here as well would refuse the mode's whole purpose after a wait that
  # can be twelve hours long.
  if [[ "${ON_EXISTING}" == "remeasure" ]]; then return 1; fi
  if [[ "${PREFLIGHT_SEEN}" != *$'\n'"${out}"$'\n'* ]]; then
    echo "REFUSING: ${out} was not on disk when the pre-flight ran, and it is now." >&2
    echo "  It appeared while this run was waiting for the GPU, so nothing here has compared it" >&2
    echo "  against ${EST}.GATE_QUALIFIED and no decision on this command line is about it. The" >&2
    echo "  modes below are answers to the pre-flight's listing; keeping a file that was not in" >&2
    echo "  that listing would extend one of them to a measurement nobody looked at, and pooling" >&2
    echo "  it is silent — the pool's instrument key does not carry the gate flag." >&2
    echo "  Nothing further was measured. Re-run: the pre-flight will list it, say whether it" >&2
    echo "  agrees with the adapter, and ask for a decision that is actually about it." >&2
    exit 5
  fi
  case "${ON_EXISTING}" in
    reuse)     return 0 ;;
    # Unreachable today and refusing anyway: `refuse` is the default, and with a file the pre-flight
    # LISTED it exits 5 up there before this function is defined. Returning false here instead would
    # silently overwrite that file, which is the one thing the default mode was written not to do,
    # so if the ordering above ever changes this fails closed rather than measuring over it.
    *)
      echo "REFUSING: ${out} exists and no mode was recorded for it (ON_EXISTING=${ON_EXISTING})." >&2
      echo "  The pre-flight must have asked for a decision before this point. Nothing was" >&2
      echo "  measured and nothing was overwritten." >&2
      exit 5 ;;
  esac
}

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
  # The skip is reachable only after the pre-flight above put the operator's decision on the record,
  # and it says which decision it is acting on. `SKIP (measured)` was the old wording and it was
  # false: nothing here measured anything.
  if already_measured "${out}"; then
    echo "KEEPING existing ${stem} (--reuse-existing; NOT measured by this run)"
    return 0
  fi
  echo "--- measuring ${stem} (${cap})"
  run_step "${stem}" .venv/bin/python scripts/measure_est_drift.py measure \
    --capture "${cap}" --estimators "${EST}" --arm both --out "${out}"
}

# --- V17 §5 C1 first: the outcome table reads the control before anything else. ----------------
measure "${CONTROL_CAPTURE}" "C1-lattice"

# --- V17 §2 Arm A. ------------------------------------------------------------------------------
for id in "${ARM_A_IDS[@]}"; do measure "${V17}/${id}" "${id}"; done

# --- V17 §5 C2, the dose ladder. Reported, never pooled. ----------------------------------------
for id in "${LADDER_IDS[@]}"; do measure "${V17}/${id}" "${id}"; done

# --- T40_RULE_V19 §3: C3, the control C1 could not be. -------------------------------------------
# C1 fired (10 runs) but its longest run was 5, which is the lattice's own period rather than a
# measurement of the statistic's sensitivity — a lattice control cannot produce a run longer than
# the lattice repeats. C3 seeds the propagation on the cube distractor's GROUND-TRUTH box on frame
# 0 of A1, a coherent capture that makes exactly one revolution, so there is no periodic return to
# break the run. Its est_drift_p95_px is the cube-to-apple distance and means NOTHING; only
# low_iou_runs is read. The fire condition is V17 §5's, unchanged: n_runs >= 1 and longest_run >= 10.
if already_measured "${V17}/EST_DRIFT-C3-wrongseed.json"; then
  echo "KEEPING existing C3-wrongseed (--reuse-existing; NOT measured by this run)"
else
  echo "--- V19: C3, propagation held on the wrong object over A1"
  WAM_PR08_CONTROL_SEED_FROM_CAPTURE="${V17}/A1" WAM_PR08_CONTROL_SEED_LABEL=cube \
  run_step "C3-wrongseed" .venv/bin/python scripts/measure_est_drift.py measure \
    --capture "${V17}/A1" --estimators "${EST}" --arm both \
    --propagation-module estimators.apple_sam2_video_wrongseed \
    --out "${V17}/EST_DRIFT-C3-wrongseed.json"
fi

# --- V17 §3 Arm B: the real corpus. -------------------------------------------------------------
if already_measured "${V17}/ARM_DIVERGENCE.json"; then
  echo "KEEPING existing ARM_DIVERGENCE (--reuse-existing; NOT measured by this run)"
else
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
if already_measured "${CENSUS_OUT}"; then
  echo "KEEPING existing V18 census (--reuse-existing; NOT measured by this run)"
else
  echo "--- V18: every frame of episode_000094, both decodes, which ones the filter refuses"
  run_step "V18 census" .venv/bin/python scripts/census_operating_point_episode.py \
    --episode episode_000094 \
    --corpus "${CORPUS}" \
    --corpus /home/humanoid/wam-t041/pr08-apple-640x480 \
    --out "${CENSUS_OUT}"
fi

echo
echo "=== the C2 ladder, reported and not pooled (V17 §5) ==="
# The directory comes in as argv: hard-coding it here would report the DEFAULT directory's
# ladder after a run written somewhere else by --out-dir, which is a summary of the wrong files.
.venv/bin/python - "${V17}" <<'PY'
import json, pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).glob("EST_DRIFT-C2-*.json")):
    d = json.loads(p.read_text())
    cap, arm = d["capture"], d["arm_comparison"]["propagation"]
    print(f"{p.stem:>18}: median motion "
          f"{cap['temporal_coherence']['median_interframe_motion_px']:8.3f} px  "
          f"runs {arm['low_iou_runs']['n_runs']:3d}  longest {arm['low_iou_runs']['longest_run']:4d}  "
          f"p95 {arm['est_drift_p95_px']}")
PY
echo "ALL V17 MEASUREMENTS DONE"
