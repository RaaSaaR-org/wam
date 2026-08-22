#!/bin/bash
# Step 23 — T-040 / PR-08 §8 item 4: measure GEOM_TOL over the AppleToPlate source corpus.
#
#   PILOT — measure the cost of the full run before paying it:
#       PILOT=1 sbatch --time=00:30:00 103_measure_geom_tol.sbatch
#
#   FULL — the committed number, walltime taken from the pilot's own derivation:
#       GEOM_STEP_FRAMES=1 sbatch --time=<from PILOT.json> 103_measure_geom_tol.sbatch
#
# WHY A PILOT MODE EXISTS AT ALL, AND WHY IT IS NOT CEREMONY.
# GEOM_TOL is one segmenter forward per decoded frame over 171 625 frames, and NOTHING IN THIS
# REPOSITORY HAS EVER MEASURED THAT RATE. scripts/measure_geom_tol.py calls its mask function once
# per frame, one frame at a time, with no batching (episode_centroids_from_video). Whether that is
# three GPU-hours or thirty decides whether the full run is a single job or needs chunking and
# resume that the script does not currently have. Guessing it, then asking for a walltime built on
# the guess, is how 189402 came to sit an hour in PENDING behind a 4 h request nobody had sized —
# and this project has already published one throughput number that was never measured
# (THROUGHPUT.json, job 189142, "9.56 GPU-h" from a run that generated zero clips). Measure it.
#
# THE PILOT RUNS TWICE, AND THAT IS THE POINT.
# A single short run cannot separate per-frame cost from the fixed cost of loading GroundingDINO
# and SAM2 — roughly 1.8 GB + 1.9 GB of weights — and extrapolating a total that still carries the
# load time inside its slope overstates the full run by the load, multiplied by every frame. Two
# passes over the SAME episodes at two frame budgets give the slope directly:
#
#     seconds_per_frame = (t_long - t_short) / (frames_long - frames_short)
#     load_seconds      = t_short - seconds_per_frame * frames_short
#
# The denominators are read out of each pass's own artifact (`n_frames`), never assumed from the
# --max-frames that was asked for: a clip shorter than the budget contributes fewer frames, and the
# artifact is the only thing that knows how many were actually decoded.
#
# BOTH PILOT PASSES EXIT 3, AND EXIT 3 IS THE CORRECT OUTCOME.
# measure_geom_tol.py forces `gate_qualified: false` and exits 3 on ANY non-zero --limit or
# --max-frames (its header, "It will not let a smoke test become the gate"). The pilot is a smoke
# test — deliberately — so this job treats 3 as success ON THE PILOT PATH ONLY and treats it as a
# failure on the full path, where it would mean the committed number came from a sample.
#
# THE PILOT ALSO ANSWERS A SECOND QUESTION FOR FREE.
# 102 staged the three estimator checkpoints and verified their ids and revisions against
# scripts/estimators/apple_sam2.py, but NOTHING HAS YET RUN THE ADAPTER. Staged bytes and a working
# segmenter are different claims. The first pass here is the first execution of
# estimators.apple_sam2 on this cluster, so a crash in it is a finding worth 20 minutes rather than
# a discovery made four hours into the full run.
#
# THE DECODER IS LEFT ON auto, AND THAT IS NOT THE SAME KIND OF auto AS --method's.
# 189585, the first run of this job, decoded zero frames: the corpus is AV1 and the cv2 in 98's venv
# has no AV1 decoder, and VideoCapture does not raise about it -- it reports 590 frames off the
# container header and reads none of them (job 189586 measured which decoders here can: pyav via
# libdav1d, imageio, torchvision; not cv2, not decord). `--decoder auto` PROBES each decoder against
# a clip of this corpus before choosing one and writes the choice and the whole probe trail into the
# artifact. That is stronger than a flag typed at submission time, which records what the operator
# believed rather than what read the pixels -- and it is why the decoder is left to auto while the
# segmenter is typed: the segmenter is a pre-registered commitment, the decoder is a fact about the
# machine that the artifact has to report.
#
# WHY --method sam2 IS TYPED OUT RATHER THAN LEFT TO auto.
# `auto` would also accept a --masks directory or fall through to whatever adapter happens to be
# importable. PR-08 §4 step 2 requires GEOM_TOL and EST_DRIFT_P95 to come from THE SAME segmenter,
# because §6 subtracts them; naming it here makes that a property of the command line rather than
# of the machine's import path. Same reason 102 stages exactly one segmenter.
#
# WHY GEOM_STEP_FRAMES IS REQUIRED ON THE FULL PATH AND NOT DEFAULTED.
# Identical discipline to 97: GEOM_TOL scales ~linearly with what a step is taken to be, PR-08 §6
# never defines one, and a default here would pick G0b's tolerance silently by a factor of the
# step. PR-08 V3 §4 REGISTERS it at 1 — so the value is known, and it still has to be typed, because
# the registered value and a default that happens to match it are different objects the day V3 is
# amended. The pilot does not ask for it: a disqualified sample sets no tolerance.
#
# WHERE THE ARTIFACT GOES, AND WHY IT IS COPIED TWICE.
# The full path writes the script's tracked default, ${WAM}/configs/transfer25/pr08_geom_tol.json
# plus its .sha256 — §8 item 4 wants GEOM_TOL COMMITTED, and a path under gitignored runs/ can
# never be committed. But sync.sh --pull only walks ${PROJ}/runs, so the tracked copy would be
# invisible from the workstation and the number would live solely on the cluster until someone
# thought to scp it. Both files are therefore copied into ${OUT} as well. The copy is the transport;
# the tracked path is the artifact.

#SBATCH --job-name=wam-geom-tol
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# NEVER OMIT --qos: without it Slurm lands the job on `normal` — 1 minute, 0 GPUs — which is a trap
# and not a fallback. GPU work takes the project QoS; the free `2cpu-single-host` rejects a --gres.
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 26 threads / 192 GB is the fair-share rate for ONE GPU (330 900 billing-h / 5 000 GPU-h = 66.18).
# Asking for more empties the billing counter before the GPU-hours and strands the rest of the
# allocation permanently. See cluster/discoverer/README.md rule 3.
#SBATCH --cpus-per-task=26
#SBATCH --mem=192G
# Sized for the PILOT. The full path needs the --time the pilot derives, passed on the command
# line — the same override 97's TIMING recipe carries, and for the same reason: `common` runs two
# nodes against ~40 pending jobs, so an honest short request backfills and a padded one waits.
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/geom-tol.%j.out

set -euo pipefail

# Before any tool can decide for itself where to write: caches.sh points HF_HOME, TORCH_HOME,
# TRITON_CACHE_DIR and TMPDIR at project storage. /home is 2 GiB and ~100 k inodes and one HF
# snapshot fills it.
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480}
RUN_ID=${RUN_ID:-pr08-geom-tol}
OUT=${PROJ}/runs/${RUN_ID}
mkdir -p "${OUT}"

PILOT_MODE=0
if [[ "${PILOT:-0}" != "0" ]]; then PILOT_MODE=1; fi

# 98's env, then its venv. sam2 1.1.0 and transformers already live there (98:47-50), so this job
# installs nothing — a job that pip-installs is a job whose result depends on the day it ran.
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}
[[ -f "${TRANSFER_ENV}" ]] || {
  echo "FATAL: ${TRANSFER_ENV} missing. Run 98_build_transfer25_env.sbatch first — it writes this"
  echo "       file and owns the .venv that carries sam2 and transformers."
  exit 1
}
source "${TRANSFER_ENV}"
source "${FRAMEWORK}/.venv/bin/activate"

# The corpus is a manifest + a flat videos/ directory, NOT a LeRobot root: find_episodes() takes the
# second branch and reports layout "clip-dir". Pointing at the parent is correct — rglob finds the
# 402 clips under videos/ — and pointing at videos/ would work too and record the same layout.
[[ -d "${SOURCE}" ]] || { echo "FATAL: SOURCE ${SOURCE} does not exist."; exit 1; }
N_CLIPS=$(find "${SOURCE}" -name '*.mp4' | wc -l)
echo "=== source     ${SOURCE}  (${N_CLIPS} clips)"
echo "=== framework  ${FRAMEWORK}"
echo "=== python     $(command -v python)"
echo "=== gpu        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# The estimator weights 102 staged. Absent, the adapter would either refuse (correct) or fetch its
# own at an unpinned revision (silent, and the committed number would then describe an estimator
# nobody staged). Checking here costs nothing and names the fix.
STAGED=${PROJ}/checkpoints/PR08_ESTIMATORS_STAGED.json
[[ -f "${STAGED}" ]] || {
  echo "FATAL: ${STAGED} missing — the segmenter, detector and depth checkpoints are not staged."
  echo "       Run 102_stage_sam2_weights.sbatch (free QoS, ~15 s once the cache is warm)."
  exit 1
}
echo "=== estimators $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(", ".join(c["repo_id"]+"@"+c["revision_resolved"][:9] for c in d["checkpoints"]))' "${STAGED}")"

# The adapter is not on the framework's import path; it lives in this repo beside the two scripts
# that drive it. Both measure_geom_tol.py and measure_est_drift.py resolve it as
# `estimators.apple_sam2`, so scripts/ is what has to be importable — not scripts/estimators/.
export PYTHONPATH="${WAM}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

# ------------------------------------------------------------------------------------------------
if (( PILOT_MODE )); then
  PILOT_EPISODES=${PILOT_EPISODES:-3}
  PILOT_FRAMES_SHORT=${PILOT_FRAMES_SHORT:-60}
  PILOT_FRAMES_LONG=${PILOT_FRAMES_LONG:-180}
  echo "=== PILOT: two passes over ${PILOT_EPISODES} episodes at ${PILOT_FRAMES_SHORT} and"
  echo "===        ${PILOT_FRAMES_LONG} frames/clip. Both are DISQUALIFIED from the gate by"
  echo "===        construction (--limit forces gate_qualified:false and exit 3) and neither"
  echo "===        touches the tracked artifact — they write under ${OUT}."

  run_pass () {  # $1 = tag, $2 = max-frames, $3 = out path -> echoes elapsed seconds
    local tag=$1 frames=$2 out=$3 start rc
    start=${SECONDS}
    set +e
    python "${WAM}/scripts/measure_geom_tol.py" \
      --corpus "${SOURCE}" \
      --method sam2 \
      --limit "${PILOT_EPISODES}" \
      --max-frames "${frames}" \
      --out "${out}" > "${OUT}/pilot-${tag}.log" 2>&1
    rc=$?
    set -e
    local elapsed=$((SECONDS - start))
    # 3 is the disqualified-sample exit and is what a correct pilot returns. Anything else is a
    # real failure: the adapter could not load, the weights are not where 102 put them, cv2 could
    # not decode av1. Print the tail rather than the whole log — the refusal messages this script
    # writes are long by design and the operative line is the last one.
    if (( rc != 3 )); then
      echo "FATAL: pilot pass '${tag}' exited ${rc}, expected 3 (the disqualified-sample exit)." >&2
      echo "       Last 40 lines of ${OUT}/pilot-${tag}.log:" >&2
      tail -n 40 "${OUT}/pilot-${tag}.log" >&2
      return 1
    fi
    echo "${elapsed}"
  }

  T_SHORT=$(run_pass short "${PILOT_FRAMES_SHORT}" "${OUT}/geom_tol_pilot_short.json")
  echo "=== pass short: ${T_SHORT}s"
  T_LONG=$(run_pass long "${PILOT_FRAMES_LONG}" "${OUT}/geom_tol_pilot_long.json")
  echo "=== pass long:  ${T_LONG}s"

  T_SHORT="${T_SHORT}" T_LONG="${T_LONG}" OUT="${OUT}" SOURCE="${SOURCE}" \
  SLURM_JOB_ID="${SLURM_JOB_ID:-}" N_CLIPS="${N_CLIPS}" python - <<'PY'
import json, os, pathlib, subprocess

out = pathlib.Path(os.environ["OUT"])
t_short, t_long = float(os.environ["T_SHORT"]), float(os.environ["T_LONG"])
short = json.loads((out / "geom_tol_pilot_short.json").read_text())
long_ = json.loads((out / "geom_tol_pilot_long.json").read_text())
f_short, f_long = short["n_frames"], long_["n_frames"]

# The corpus's own count, from the manifest the conversion wrote -- not from this pilot's sample and
# not from a number typed into a comment. If the two disagree the extrapolation is about a corpus
# that is not the one on disk.
manifest = json.loads((pathlib.Path(os.environ["SOURCE"]) / "manifest.json").read_text())
total_frames = manifest["source"]["total_frames"]
total_eps = manifest["source"]["total_episodes"]

blockers = []
if f_long <= f_short:
    # Both budgets hit the end of the clips: the long pass decoded no more frames than the short
    # one, so there is no slope to read and the difference is noise. Say so instead of dividing.
    blockers.append(
        f"the long pass decoded {f_long} frames and the short one {f_short} -- no additional "
        "frames, so no slope. Raise PILOT_FRAMES_LONG, or lower PILOT_FRAMES_SHORT."
    )

rec = {
    "schema": "wam.pr08_geom_tol_pilot/1",
    "purpose": "Cost of the full GEOM_TOL measurement, measured rather than assumed, so the full "
               "run can be given an honest --time. PR-08 §8 item 4.",
    "produced_by": "cluster/discoverer/103_measure_geom_tol.sbatch (PILOT=1)",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
    "corpus": os.environ["SOURCE"],
    "corpus_total_frames": total_frames,
    "corpus_total_episodes": total_eps,
    "clips_on_disk": int(os.environ["N_CLIPS"]),
    "passes": [
        {"tag": "short", "wall_seconds": t_short, "n_frames": f_short,
         "n_episodes": short.get("n_episodes"), "coverage": short.get("coverage")},
        {"tag": "long", "wall_seconds": t_long, "n_frames": f_long,
         "n_episodes": long_.get("n_episodes"), "coverage": long_.get("coverage")},
    ],
    "mask_method": short.get("mask_method"),
    "resolution_hw": short.get("resolution_hw"),
    "gate_qualified": False,
    "gate_qualified_note":
        "A pilot is a smoke test. Both passes carry gate_qualified:false and exited 3, which is "
        "measure_geom_tol.py refusing to let a sample become the gate. Nothing here is GEOM_TOL.",
}

if blockers:
    rec["blockers"] = blockers
else:
    per_frame = (t_long - t_short) / (f_long - f_short)
    load_s = t_short - per_frame * f_short
    full_s = per_frame * total_frames + load_s
    rec |= {
        "seconds_per_frame": per_frame,
        "load_seconds": load_s,
        "load_seconds_note":
            "Fixed startup: interpreter, torch, and the GroundingDINO + SAM2 weights. Separated "
            "out by the two-pass slope so it is not multiplied by every frame of the corpus.",
        "full_run_seconds": full_s,
        "full_run_gpu_hours": full_s / 3600.0,
        "recommended_time_limit":
            # +25 % and rounded up to the next 30 minutes. The margin is for the tail of the
            # episode-length distribution (249-749 frames) that a 3-episode sample cannot see, and
            # it is stated here rather than folded silently into the estimate.
            "%02d:%02d:00" % divmod(int(((full_s * 1.25) + 1799) // 1800) * 30, 60),
        "recommended_time_limit_basis": "full_run_seconds x 1.25, rounded up to the next 30 min",
        "single_job_feasible": full_s * 1.25 < 4 * 3600,
        "single_job_feasible_note":
            "False means the full measurement does not fit one comfortable request on `common` and "
            "measure_geom_tol.py needs an episode-range chunking mode with a merge, which it does "
            "not have today.",
    }

path = out / "GEOM_TOL_PILOT.json"
path.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print(f"\n=== wrote {path}")
PY
  echo "=== PILOT done. Read ${OUT}/GEOM_TOL_PILOT.json, then submit the full run with the"
  echo "===   --time it recommends:  GEOM_STEP_FRAMES=1 sbatch --time=<hh:mm:00> $(basename "$0")"
  exit 0
fi

# -- FULL ----------------------------------------------------------------------------------------
# See the header. Registered by PR-08 V3 §4 at 1; still typed, never defaulted.
: "${GEOM_STEP_FRAMES:?PR-08 §6 never defines what a step is, and GEOM_TOL scales ~linearly with it, so a default here would not pick a convention -- it would set the G0b tolerance silently, by a factor of the step. PR-08 V3 §4 REGISTERS the value: GEOM_STEP_FRAMES=1. Type the registered value. It stays undefaulted for the same reason CEILING_GPU_H does: the registered number and a default that happens to agree with it today are different objects the day V3 is amended, and 97 refuses an artifact whose step_frames differs from what it was handed. (NOTE: no apostrophes in this message. Inside ${var:?word} bash still parses quotes, so one lone apostrophe here opens a string that swallows the rest of the file -- which is exactly how this line first shipped broken.)}"
[[ "${GEOM_STEP_FRAMES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: GEOM_STEP_FRAMES=${GEOM_STEP_FRAMES} is not a positive integer."
  exit 1
}

ARTIFACT=${WAM}/configs/transfer25/pr08_geom_tol.json
echo "=== FULL: ${N_CLIPS} clips, step ${GEOM_STEP_FRAMES} frame(s), no --limit and no --max-frames."
echo "===       Either of those would force gate_qualified:false and exit 3, which is the whole"
echo "===       point of them: the committed number is the whole corpus or it is not the number."
START=${SECONDS}
# `set -e` is on, so a bare invocation followed by `RC=$?` NEVER SEES A NON-ZERO STATUS: the shell
# is already gone by the time the assignment runs, and RC would be a variable that can only ever
# hold 0 while looking like it reports the exit code. Put the call in an `if`, where `set -e` does
# not fire, and keep what it actually said.
if python "${WAM}/scripts/measure_geom_tol.py" \
  --corpus "${SOURCE}" \
  --method sam2 \
  --step-frames "${GEOM_STEP_FRAMES}" \
  --out "${ARTIFACT}"; then RC=0; else RC=$?; fi
ELAPSED=$((SECONDS - START))
echo "=== measure_geom_tol.py exited ${RC} after ${ELAPSED}s"
# Exit 3 on THIS path is not the pilot's benign disqualification. It means the run that was supposed
# to produce the committed number produced a sample instead -- coverage below the floor, or a
# --limit that reached it somehow -- and the artifact on disk carries gate_qualified:false. Copying
# that forward would put a disqualified GEOM_TOL where 97 looks for the committed one.
if (( RC != 0 )); then
  echo "FATAL: the FULL measurement exited ${RC}. Whatever is at ${ARTIFACT} is not GEOM_TOL:"
  echo "       exit 3 means the run is disqualified from the gate (partial, or coverage under the"
  echo "       floor) and any other code means it did not finish. Nothing is copied to ${OUT}."
  exit "${RC}"
fi

# The transport copy — see the header. Missing sidecar is fatal: 97 refuses a GEOM_CONSTANTS with no
# .sha256 beside it, so shipping the artifact without it produces a file that cannot be used.
[[ -f "${ARTIFACT}.sha256" ]] || {
  echo "FATAL: ${ARTIFACT}.sha256 was not written. 97 refuses a GEOM_TOL with no committed digest."
  exit 1
}
cp "${ARTIFACT}" "${ARTIFACT}.sha256" "${OUT}/"
echo "=== copied artifact + sidecar to ${OUT}/ for sync.sh --pull"
echo "=== NEXT: commit configs/transfer25/pr08_geom_tol.json AND its .sha256 in the repo, then"
echo "===       re-run 102_stage_sam2_weights.sbatch — its segmenter-vs-pr08_geom_tol.json check"
echo "===       reads 'unverified: not committed' until that file exists."
