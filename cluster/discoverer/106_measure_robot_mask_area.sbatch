#!/bin/bash
# Step 26 — T-040 / PR-08 §6 G0c: measure the ROBOT-MASK AREA DISTRIBUTION over the source corpus.
#
#   PILOT — measure the cost and the empty-mask rate before paying for either:
#       PILOT=1 sbatch --time=00:30:00 106_measure_robot_mask_area.sbatch
#
#   FULL — the distribution in ONE job. The arithmetic below says it does not fit; kept because the
#          sharded path has to be checkable against something, and because it is the right shape on
#          any machine whose wall is longer than the run:
#       sbatch --time=<from PILOT.json> 106_measure_robot_mask_area.sbatch
#
#   SHARD — the distribution as an array of N jobs, each measuring the episodes that hash to it:
#       SHARD=1 NUM_SHARDS=3 sbatch --array=0-2%3 --time=02:00:00 \
#           106_measure_robot_mask_area.sbatch
#
#   MERGE — once every shard has landed, pool them. NO GPU IS USED, so it belongs on the free QoS,
#           which needs its limits on the command line too:
#       MERGE=1 NUM_SHARDS=3 sbatch --qos=2cpu-single-host --gres=none \
#           --cpus-per-task=2 --mem=8G --time=00:20:00 106_measure_robot_mask_area.sbatch
#
# WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT DECIDE.
# PR-08 §6 G0c composites the real robot's pixels back over every generated frame. A robot mask that
# has grounded on the table, or on the whole scene, composites the SOURCE back over everything: the
# restyle becomes a no-op and arms B and C silently become arm A at full GPU cost. Refusing such a
# frame needs a number, and §6 opens with "No threshold is coined." So robot_composite.load_area_bound
# REFUSES to run without a committed bound, and job 189644 — the §8 item 3 throughput measurement —
# stopped on exactly that refusal, correctly, because the corpus statistic that would derive the
# bound had never been measured. This job measures it.
#
# IT DOES NOT SET THE BOUND. The artifact is written with `max_frame_fraction: null` on every path
# here, including the merge, and scripts/robot_composite.py contains no default, no heuristic and no
# suggested value. The reason is not squeamishness: the only bound derivable from the distribution
# is the observed maximum, which can never fire on the frames it was measured over, and any bound
# that CAN fire sits above it by a margin nothing in the corpus determines. A human reads the five
# numbers, writes max_frame_fraction and bound_rationale into the artifact, and commits it. The
# rationale is the record of a decision a script refused to make.
#
# WHY THE H.264 TREE AND NOT THE AV1 ONE.
# SOURCE defaults to data/pr08-apple-640x480-h264-lossless, not data/pr08-apple-640x480. The corpus
# ships AV1 and the cv2 4.11.0 in 98's generation venv is built against avcodec 59.37.100, which has
# no AV1 decoder and DOES NOT RAISE about it: VideoCapture opens the container, believes the header,
# reports 590 frames and fails every read. Three jobs have died on that shape (186357 wrote 0
# captions; 189585 died at 7 s; 189584 watched upstream's own SAM2 helper extract 0 of 590 frames).
# robot_composite._decode_frames has an imageio fallback for exactly this, but the fallback is not
# the point: the H.264 tree is proven BIT-EXACT per clip (scripts/transcode_corpus_lossless.py,
# TRANSCODE_PROOF.json — frame count and max abs channel delta 0 in yuv420p AND rgb24 on every clip)
# and job 189613 verified all 402 clips decode under the generation venv's own cv2. Measuring the
# area distribution on pixels the generation path cannot read would be a distribution of a corpus
# this pipeline never sees. Bit-exactness is what keeps the transcode out from between the two.
#
# WHY THE FULL RUN IS SHARDED, WITH THE ARITHMETIC RATHER THAN A CHOSEN NUMBER.
# The corpus is 171 625 frames (manifest: 402 episodes, 249-749 frames each). One GroundingDINO
# forward plus one SAM 2 forward per frame, one frame at a time, no batching. The per-frame cost was
# MEASURED rather than assumed — see PROVENANCE OF THE ESTIMATE below — at
#
#     p = 0.0765 s/frame        (estimate for one H200; the PILOT re-derives it on the machine)
#     L = 120 s                 (fixed: interpreter, torch, ~3.7 GB of GroundingDINO + SAM 2)
#
#     full_run_seconds     171 625 x 0.0765 + 120                  = 13 249 s = 3.68 GPU-h
#     MaxWall, every QoS                                             14 400 s
#                                                                    ------------------
#                          1.09x the wall. It fits only if the estimate is exactly right, which is
#                          not a plan: the estimate is a scaled workstation measurement and the
#                          pilot exists because this repository has published one throughput number
#                          that was never measured (THROUGHPUT.json, job 189142, "9.56 GPU-h" from
#                          a run that generated zero clips). A 25 % error puts it over the wall.
#
# Sharding pays L once per shard instead of once, and the shards are NOT equal: the partition is
# blake2b(episode_id) % N, which balances episode COUNTS approximately and FRAMES not at all. The
# frames per shard are known exactly from the manifest, so the table below is the heaviest shard's
# real load rather than an even-split floor:
#
#     N   shard frames (max)   heaviest = frames x p + L    x1.25    --time(x1.25)  wall/heaviest
#     2          91 439                 7 115 s = 118.6 m   148.2 m    02:30:00        2.02x
#     3          62 580                 4 907 s =  81.8 m   102.3 m    01:45:00        2.93x
#     4          48 766                 3 851 s =  64.2 m    80.2 m    01:30:00        3.74x
#     5          38 042                 3 030 s =  50.5 m    63.1 m    01:15:00        4.75x
#     6          32 326                 2 593 s =  43.2 m    54.0 m    01:00:00        5.55x
#     7          27 251                 2 205 s =  36.7 m    45.9 m    01:00:00        6.53x
#     8          24 546                 1 998 s =  33.3 m    41.6 m    00:45:00        7.21x
#     total cost is N x L + 171 625 x p either way: N = 3 costs 3.66 GPU-h, N = 8 costs 3.78 GPU-h.
#     The whole price of sharding is the extra model loads: 0.03 GPU-h per shard.
#
# THE `--time` COLUMN IS THE PILOT'S OWN FORMULA (x1.25, rounded up to the next 15 min) AND IT IS
# NOT WHAT TO TYPE UNTIL THE PILOT HAS RUN. p above is a workstation measurement scaled by a
# cross-stack ratio, not a measurement on an H200, and at N = 3 the x1.25 request leaves 2.7 minutes
# of slack against its own estimate — enough for the formula and not enough for the estimate being
# wrong. Until PILOT.json exists, round UP to the next half hour instead (N = 3 -> 02:00:00, which
# is still 2.0x the estimate and 2.9x under the wall) and let the pilot replace the guess: it writes
# per_shard_estimates with this same table computed from a p it measured on the machine.
#
# WHY N = 3 IS THE RECOMMENDATION, AND IT IS A SUBMISSION-LIMIT DECISION, NOT A COST ONE.
# The project QoS carries MaxSubmitJobsPU=8 and MaxJobsPU=4, per user, ACROSS THE QOS, and every
# association carries DenyOnLimit — so an over-limit array is REJECTED at submission with
# QOSMaxSubmitJobPerUserLimit rather than queued. An N-task array is N submissions. With five of my
# jobs already in flight (189637, 189658 x4 tasks, 189661, 189662, 189665 — check with
# `squeue -u $USER -r`), three submission slots are free, so N = 3 is what can be submitted at all
# today, and it happens to run in ONE scheduling round because MaxJobsPU=4 >= 3. N = 8 buys 2.5x the
# headroom on the shard that decides the outcome and costs 0.12 GPU-h more; it is the right choice
# the moment the queue is empty, and the table above is there so the choice can be re-made without
# re-deriving anything. NUM_SHARDS is a parameter of this file for that reason.
#
# `%3` in `--array=0-2%3` is not politeness — it keeps the array inside MaxJobsPU rather than
# relying on a DenyOnLimit QoS to queue the surplus instead of rejecting it.
#
# PROVENANCE OF THE ESTIMATE, WHICH IS A SCALED MEASUREMENT AND SAYS SO.
# The GEOM_TOL pilot (job 189588) measured its estimator stack on an H200 at 0.0833 s/frame with a
# 116 s fixed load. That stack is estimators.apple_sam2.segment: GroundingDINO -> SAM 2, one box,
# NO DEPTH MODEL — the depth checkpoint is staged but segment() never touches it. The robot masker
# is the same two models through the same module, with a three-phrase prompt and every box above
# threshold segmented and OR-ed instead of the best one. So "robot masking skips the depth model" is
# NOT a saving over GEOM_TOL: neither runs it. The two differ only in prompt length and box count.
# Measured on this project's workstation (RTX 5090), interleaved frame by frame over 235 frames of
# four clips of THIS corpus so that clock drift and GPU contention hit both stacks equally:
#
#     robot / apple = 0.918        (robot 0.3091 s/frame, apple 0.3368 s/frame, contended)
#     0.0833 x 0.918 = 0.0765 s/frame on an H200
#
# The ratio is the transferable quantity and the absolute workstation numbers are not: an idle
# repeat of the same robot pass measured 0.102 s/frame, and the 3x spread between the two is two
# other processes that appeared on the card (an ollama server and an Isaac capture), which is
# exactly why the ratio was taken interleaved. Decode is not in either figure and is not material:
# cv2 on 640x480 H.264 is single-digit milliseconds against a 76 ms frame. THE PILOT BELOW MEASURES
# p ON THE MACHINE THAT WILL RUN IT and writes it down; the SHARD branch re-reads it at run time
# rather than trusting the constants in this header.
#
# WHY A PILOT AT ALL, AND WHY IT IS NOT CEREMONY.
# Two questions, neither of which a staged checkpoint answers:
#   1. p on an H200 for THIS masker, which is what the --time above is built on.
#   2. THE EMPTY-MASK RATE. An empty robot mask is clip-fatal in G0c (robot_composite.check_mask:
#      "zero is zero", no threshold), so if ROBOT_TEXT_PROMPT leaves source frames ungrounded, every
#      clip containing one of those frames REFUSES. The prompt's adequacy on this corpus is unproven
#      — robot_composite says so in the constant's own comment — and this is the first time the
#      pinned pair is pointed at these frames with it. Learning that from a pilot costs 20 minutes;
#      learning it from a chunk of refusals costs a generation pass.
# The pilot also runs the adapter for the first time in this venv on this data path, so a crash in
# it is a finding worth 20 minutes rather than an array's worth of GPU.
#
# BOTH PILOT PASSES EXIT 3, AND EXIT 3 IS THE CORRECT OUTCOME THERE.
# robot_composite.py forces `measurement_qualified: false` and exits 3 on ANY --limit or any
# --stride > 1, and load_area_bound refuses such a file by name. The pilot is a smoke test — it uses
# both knobs deliberately — so this job treats 3 as success ON THE PILOT PATH ONLY, and as a failure
# everywhere else, where it would mean the committed distribution came from a sample.
#
# THE PILOT RUNS TWICE, AND THAT IS THE POINT.
# A single short run cannot separate per-frame cost from the fixed cost of loading two models, and
# extrapolating a total that still carries the load inside its slope overstates the full run by the
# load multiplied by every frame. Two passes over the SAME episodes at two strides give the slope:
#
#     seconds_per_frame = (t_dense - t_sparse) / (frames_dense - frames_sparse)
#     load_seconds      = t_sparse - seconds_per_frame * frames_sparse
#
# The frame counts are read out of each pass's own artifact (`measured.frames`), never assumed from
# the --stride that was asked for.
#
# WHY THE SHARDS CANNOT SIMPLY BE AVERAGED, WHICH IS THE ENTIRE REASON --merge EXISTS.
# The artifact reports min, median, p95, p99 and max. THREE OF THOSE FIVE DO NOT DECOMPOSE. The
# median of the shard medians is a different statistic with the same units and a plausible
# magnitude, and the same is true of a p95 and a p99; only min and max recombine. A bound will sit
# above these numbers and its rationale will quote them, and nothing downstream re-derives either,
# so the error would be permanent and invisible. Each shard therefore emits every per-frame area
# fraction it measured and --merge takes the five numbers ONCE over the pooled population, rebuilt
# in the manifest's own enumeration order, so the merged artifact is identical to what a single
# un-sharded run would have written — asserted, not claimed, in
# tests/test_robot_composite_shards.py.
#
# WHAT A MERGE REFUSES AND WHAT IT MERELY DISQUALIFIES.
# Two different outcomes with two different exit codes, because they are two different facts:
#   exit 1  NOTHING IS WRITTEN. The shards cannot be pooled into one distribution at all: a
#           duplicated shard index, two partitions mixed, an episode in a shard it does not hash to,
#           a shard reporting an episode it was not assigned, a shard that kept only its summary.
#           In each of those the pool itself would be wrong — an episode weighted twice moves the
#           median and both percentiles — so there is no honest distribution to write.
#   exit 3  THE ARTIFACT IS WRITTEN AND STAMPED `measurement_qualified: false`, with the failing
#           conditions named under merged_from.qualification: a missing shard, a truncated shard, a
#           shard that calls itself unqualified, or shards that disagree about the estimator, the
#           source manifest or the prompt. The arithmetic is right about the frames it saw and is
#           not the corpus's, which is what that flag has always meant here — and load_area_bound
#           refuses the file by name, so it can never become a bound.
#
# WHY A JOB ARRAY HERE, WHEN cluster/discoverer/README.md SAYS NOT TO USE ONE.
# That rule is specific and its reason does not reach this job. 55, 70 and 97 avoid arrays because
# an array task's interaction with `scontrol requeue` and SLURM_RESTART_COUNT is an open question
# (docs/discoverer.md open questions item 8) and those jobs carry a requeue rail whose failure mode
# is an unbounded loop. THIS JOB HAS NO REQUEUE RAIL: no --requeue, no USR1 trap, no restart
# counter. It reads video and writes one JSON file, and a shard whose artifact already landed and
# validates is SKIPPED, so a preempted task is free to re-run. Same argument 103 makes.
#
# BILLING, WHICH IS WHERE THE MEMORY REQUEST COMES FROM.
# Billing per minute = GPUs x 1.0 + MemGB x 0.25 + Threads x 0.036, so MEMORY DOMINATES. Job 189588
# ran this estimator stack and peaked at MaxRSS 4.75 GiB against a 192 G request. 32 G is 6.7x the
# measured peak and takes the rate from ~50 units/min to 1.0 + 8.0 + 0.936 = 9.94. The 26 threads
# are kept — they are the fair-share thread rate for one GPU and they are 0.94 of those ~10 units,
# so cutting them is a rounding error against the memory. NEVER exceed 26 threads or 257 GB per GPU,
# and NEVER omit --qos: without it Slurm lands the job on `normal`, which is 1 minute and 0 GPUs.

#SBATCH --job-name=wam-robot-mask-area
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# NEVER OMIT --qos: `normal` is 1 minute and 0 GPUs, which is a trap and not a fallback. GPU work
# takes the project QoS; the free `2cpu-single-host` rejects a --gres, which is why the MERGE recipe
# overrides four directives at once.
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
# 32 G, not the 192 G 103 asks for. See BILLING above: 189588 peaked at 4.75 GiB on this stack, and
# memory is 0.25 units/min per GB against a GPU's 1.0. This is the single biggest lever on the
# allocation and it is set from a measurement.
#SBATCH --mem=32G
# Sized for the PILOT. Every other path needs the --time the arithmetic above (or, better, the
# pilot's own artifact) derives, passed on the command line — `common` runs two nodes against a
# deep queue, so an honest short request backfills and a padded one waits.
#SBATCH --time=00:30:00
# THERE IS DELIBERATELY NO `#SBATCH --array` HERE. A directive in this file applies to EVERY
# submission of it, so a hardcoded array would turn the PILOT and the MERGE into arrays as well —
# three pilots racing for one PILOT.json, three merges racing for the artifact. The array is passed
# on the command line, exactly the way --time already is, and the SHARD path REFUSES to run when
# SLURM_ARRAY_TASK_ID is unset rather than quietly measuring shard 0 and calling it the corpus.
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/robot-mask-area.%j.out

set -euo pipefail

# Before any tool can decide for itself where to write: caches.sh points HF_HOME, TORCH_HOME,
# TRITON_CACHE_DIR and TMPDIR at project storage. /home is 2 GiB and ~100 k inodes and one HF
# snapshot fills it.
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
# The H.264 tree, for the reason in the header: it is bit-exact per clip and it is the only tree the
# generation venv's cv2 4.11.0 can read (verified over all 402 clips by job 189613).
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480-h264-lossless}
RUN_ID=${RUN_ID:-pr08-robot-mask-area}
OUT=${PROJ}/runs/${RUN_ID}
mkdir -p "${OUT}"

PILOT_MODE=0
if [[ "${PILOT:-0}" != "0" ]]; then PILOT_MODE=1; fi
SHARD_MODE=0
if [[ "${SHARD:-0}" != "0" ]]; then SHARD_MODE=1; fi
MERGE_MODE=0
if [[ "${MERGE:-0}" != "0" ]]; then MERGE_MODE=1; fi

# Two modes at once is not a preference to resolve, it is a submission that meant one thing and
# would do another. Named individually so the message says which pair was set.
if (( PILOT_MODE + SHARD_MODE + MERGE_MODE > 1 )); then
  echo "FATAL: PILOT=${PILOT:-0}, SHARD=${SHARD:-0} and MERGE=${MERGE:-0} are three different jobs."
  echo "       PILOT sizes the run, SHARD measures one piece of the corpus, MERGE pools the pieces"
  echo "       into the distribution a bound may sit above. Set exactly one, or none for FULL."
  exit 1
fi

# One directory, one file per shard, so MERGE can be pointed at the directory and
# robot_composite.py can skip whatever else has accumulated in ${OUT} (the pilot artifacts, and
# after the first merge the merged artifact itself).
SHARD_DIR=${OUT}/shards

# 98's env, then its venv. sam2 and transformers already live there (98:47-50), so this job installs
# nothing — a job that pip-installs is a job whose result depends on the day it ran.
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}
[[ -f "${TRANSFER_ENV}" ]] || {
  echo "FATAL: ${TRANSFER_ENV} missing. Run 98_build_transfer25_env.sbatch first — it writes this"
  echo "       file and owns the .venv that carries sam2 and transformers."
  exit 1
}
source "${TRANSFER_ENV}"
source "${FRAMEWORK}/.venv/bin/activate"

# The adapter is not on the framework's import path; it lives in this repo beside the scripts that
# drive it. robot_composite.py resolves it as `estimators.apple_sam2`, so scripts/ is what has to be
# importable — not scripts/estimators/.
export PYTHONPATH="${WAM}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

# THE PROMPT AND THE THRESHOLDS ARE NOT SUBMISSION-TIME DECISIONS.
# apple_sam2 reads WAM_PR08_BOX_THRESHOLD / WAM_PR08_TEXT_THRESHOLD / WAM_PR08_OBJECT_PROMPT from
# the environment. A detection threshold typed into a submit script is a per-run decision about
# which pixels the generator is allowed to touch, recorded nowhere anybody would look — the exact
# failure robot_composite's ROBOT_TEXT_PROMPT is a committed constant to avoid. 97 unsets these
# before the driver runs and so does this: the distribution and the bound sitting above it must
# describe the segmenter the code declares.
unset WAM_PR08_BOX_THRESHOLD WAM_PR08_TEXT_THRESHOLD WAM_PR08_OBJECT_PROMPT
unset WAM_PR08_RETRY_BOX_THRESHOLD WAM_PR08_RETRY_TEXT_THRESHOLD

ARTIFACT=${WAM}/configs/transfer25/pr08_robot_mask_area.json

# ------------------------------------------------------------------------------------------------
# MERGE. Placed BEFORE the corpus and estimator checks on purpose: --merge reads shard artifacts and
# never opens a clip, never imports the adapter and never touches a GPU, which is exactly why it can
# run on the free QoS. Requiring the corpus here would tie the cheapest step in the chain to the
# most expensive preconditions for no reason.
if (( MERGE_MODE )); then
  # No apostrophes in this message. Inside ${var:?word} bash still parses quotes, so one lone
  # apostrophe opens a string that swallows the rest of the file.
  : "${NUM_SHARDS:?MERGE needs to know how many shards to insist on. Without it a merge over the shards that happen to have landed is a distribution over part of the corpus wearing the name of the whole. Pass the same NUM_SHARDS the array was submitted with, for example NUM_SHARDS=3.}"
  [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} is not a positive integer."
    exit 1
  }

  [[ -d "${SHARD_DIR}" ]] || {
    echo "FATAL: ${SHARD_DIR} does not exist — no shard has run yet."
    echo "       SHARD=1 NUM_SHARDS=${NUM_SHARDS} \\"
    echo "         sbatch --array=0-$((NUM_SHARDS - 1))%3 --time=02:00:00 $(basename "$0")"
    exit 1
  }

  # Counted here as well as inside the script. The script stamps the artifact and exits 3 on a
  # missing shard with the full argument; this says the same thing before a Python interpreter
  # starts, which is what an operator reading the log wants at the top of it.
  MISSING=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    [[ -f "${SHARD_DIR}/shard-${i}.json" ]] || MISSING+=("${i}")
  done
  if (( ${#MISSING[@]} )); then
    echo "FATAL: shard artifact(s) missing from ${SHARD_DIR}: ${MISSING[*]} of ${NUM_SHARDS}."
    echo "       Re-submit just those array indices — completed shards are skipped, so this costs"
    echo "       only the shards that did not land:"
    echo "         SHARD=1 NUM_SHARDS=${NUM_SHARDS} \\"
    echo "           sbatch --array=$(IFS=,; echo "${MISSING[*]}")%3 --time=02:00:00 $(basename "$0")"
    exit 1
  fi

  echo "=== MERGE: ${NUM_SHARDS} shard(s) from ${SHARD_DIR} -> ${ARTIFACT}"
  echo "===        min/median/p95/p99/max are taken ONCE over the pooled per-frame area fractions."
  echo "===        Shard percentiles are never recombined: three of the five do not decompose, and"
  echo "===        the bound a human writes will sit above them."
  # `set -e` is on, so a bare invocation followed by RC=$? never sees a non-zero status — the shell
  # is already gone. The `if` is where set -e does not fire and the exit code survives.
  if python "${WAM}/scripts/robot_composite.py" measure \
    --merge "${SHARD_DIR}" \
    --out "${ARTIFACT}"; then RC=0; else RC=$?; fi
  echo "=== robot_composite.py measure --merge exited ${RC}"
  if (( RC != 0 )); then
    echo "FATAL: the merge exited ${RC} and whatever is at ${ARTIFACT} is not the distribution."
    echo "       Exit 1 is a refusal and NOTHING WAS WRITTEN — a duplicated shard, two partitions"
    echo "       mixed, an episode in a shard it does not hash to, a shard that kept only its"
    echo "       summary. Exit 3 means the merge completed and the artifact is stamped"
    echo "       measurement_qualified:false: a missing shard, a truncated shard, or shards that"
    echo "       disagree about the estimator, the manifest or the prompt. Read the message above;"
    echo "       merged_from.qualification names which condition failed."
    exit "${RC}"
  fi
  cp "${ARTIFACT}" "${OUT}/"
  echo "=== copied the artifact to ${OUT}/ for sync.sh --pull"
  echo "=== NEXT, AND IT IS NOT A COMMIT YET:"
  echo "===   1. Read the five numbers. max_frame_fraction is null and every path in this job"
  echo "===      leaves it null — the observed maximum cannot fire on the frames it was measured"
  echo "===      over, and any bound above it carries a margin nothing in the corpus derives."
  echo "===   2. Read empty_frames. An empty robot mask REFUSES a clip (no threshold; zero is"
  echo "===      zero), so a non-zero count here is a prompt problem to solve before generation."
  echo "===   3. A human writes max_frame_fraction and bound_rationale into the artifact and"
  echo "===      commits configs/transfer25/pr08_robot_mask_area.json. The rationale is the record"
  echo "===      of a decision this script refused to make."
  exit 0
fi

# ------------------------------------------------------------------------------------------------
[[ -d "${SOURCE}" ]] || { echo "FATAL: SOURCE ${SOURCE} does not exist."; exit 1; }
MANIFEST=${SOURCE}/manifest.json
[[ -f "${MANIFEST}" ]] || {
  echo "FATAL: ${MANIFEST} missing. robot_composite.py measure reads the manifest, not a glob:"
  echo "       the episode ids in it are what the shard partition is keyed on."
  exit 1
}
N_CLIPS=$(find "${SOURCE}" -name '*.mp4' | wc -l)
echo "=== source     ${SOURCE}  (${N_CLIPS} clips)"
echo "=== manifest   $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%d episodes, %s frames, codecs %s" % (len(d["episodes"]), d["source"].get("total_frames"), d["source"].get("codecs")))' "${MANIFEST}")"
echo "=== framework  ${FRAMEWORK}"
echo "=== python     $(command -v python)"
echo "=== gpu        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# The estimator weights 102 staged. Absent, the adapter would either refuse (correct) or fetch its
# own at an unpinned revision (silent, and the distribution would then describe an estimator nobody
# staged, while load_area_bound's cross-check would later reject the bound for disagreeing with the
# masker that made the masks). Checking here costs nothing and names the fix.
STAGED=${PROJ}/checkpoints/PR08_ESTIMATORS_STAGED.json
[[ -f "${STAGED}" ]] || {
  echo "FATAL: ${STAGED} missing — the segmenter and detector checkpoints are not staged."
  echo "       Run 102_stage_sam2_weights.sbatch (free QoS, ~15 s once the cache is warm)."
  exit 1
}
echo "=== estimators $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(", ".join(c["repo_id"]+"@"+c["revision_resolved"][:9] for c in d["checkpoints"]))' "${STAGED}")"

# ------------------------------------------------------------------------------------------------
if (( PILOT_MODE )); then
  PILOT_EPISODES=${PILOT_EPISODES:-3}
  PILOT_STRIDE_SPARSE=${PILOT_STRIDE_SPARSE:-5}
  echo "=== PILOT: two passes over the SAME ${PILOT_EPISODES} episodes, at stride"
  echo "===        ${PILOT_STRIDE_SPARSE} and stride 1. Both are DISQUALIFIED from being a bound's"
  echo "===        distribution by construction (--limit and --stride each force"
  echo "===        measurement_qualified:false and exit 3) and neither touches the tracked"
  echo "===        artifact — they write under ${OUT}."
  echo "===        The strides are over the SAME episodes on purpose: the slope between two frame"
  echo "===        counts on one set of clips is the per-frame cost with the model load removed."

  run_pass () {  # $1 = tag, $2 = stride, $3 = out path -> echoes elapsed seconds
    local tag=$1 stride=$2 out=$3 start rc
    start=${SECONDS}
    set +e
    python "${WAM}/scripts/robot_composite.py" measure \
      --manifest "${MANIFEST}" \
      --limit "${PILOT_EPISODES}" \
      --stride "${stride}" \
      --out "${out}" > "${OUT}/pilot-${tag}.log" 2>&1
    rc=$?
    set -e
    local elapsed=$((SECONDS - start))
    # 3 is the disqualified-sample exit and is what a correct pilot returns. Anything else is a
    # real failure: the adapter could not load, the weights are not where 102 put them, cv2 could
    # not decode. Print the tail — the refusals this script writes are long by design and the
    # operative line is the last one.
    if (( rc != 3 )); then
      echo "FATAL: pilot pass '${tag}' exited ${rc}, expected 3 (the disqualified-sample exit)." >&2
      echo "       Last 40 lines of ${OUT}/pilot-${tag}.log:" >&2
      tail -n 40 "${OUT}/pilot-${tag}.log" >&2
      return 1
    fi
    echo "${elapsed}"
  }

  T_SPARSE=$(run_pass sparse "${PILOT_STRIDE_SPARSE}" "${OUT}/area_pilot_sparse.json")
  echo "=== pass sparse (stride ${PILOT_STRIDE_SPARSE}): ${T_SPARSE}s"
  T_DENSE=$(run_pass dense 1 "${OUT}/area_pilot_dense.json")
  echo "=== pass dense  (stride 1):  ${T_DENSE}s"

  T_SPARSE="${T_SPARSE}" T_DENSE="${T_DENSE}" OUT="${OUT}" SOURCE="${SOURCE}" \
  SLURM_JOB_ID="${SLURM_JOB_ID:-}" N_CLIPS="${N_CLIPS}" python - <<'PY'
import json, os, pathlib

out = pathlib.Path(os.environ["OUT"])
t_sparse, t_dense = float(os.environ["T_SPARSE"]), float(os.environ["T_DENSE"])
sparse = json.loads((out / "area_pilot_sparse.json").read_text())
dense = json.loads((out / "area_pilot_dense.json").read_text())
f_sparse, f_dense = sparse["measured"]["frames"], dense["measured"]["frames"]

# The corpus's own count, from the manifest the conversion wrote -- not from this pilot's sample and
# not from a number typed into a comment. If the two disagree the extrapolation is about a corpus
# that is not the one on disk.
manifest = json.loads((pathlib.Path(os.environ["SOURCE"]) / "manifest.json").read_text())
total_frames = manifest["source"]["total_frames"]
episodes = [(e["id"], e["frames"]) for e in manifest["episodes"]]

import hashlib


def shard_of(key, n):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % n


blockers = []
if f_dense <= f_sparse:
    blockers.append(
        f"the dense pass measured {f_dense} frames and the sparse one {f_sparse} -- no additional "
        "frames, so no slope. Lower PILOT_STRIDE_SPARSE, or raise PILOT_EPISODES."
    )

rec = {
    "schema": "wam.pr08_robot_mask_area_pilot/1",
    "purpose": "Cost and empty-mask rate of the full robot-mask-area measurement, measured rather "
               "than assumed, so the full run can be given an honest --time and the prompt can be "
               "shown to ground the robot before 10 050 clips depend on it. PR-08 6 G0c.",
    "produced_by": "cluster/discoverer/106_measure_robot_mask_area.sbatch (PILOT=1)",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
    "corpus": os.environ["SOURCE"],
    "corpus_total_frames": total_frames,
    "corpus_total_episodes": len(episodes),
    "clips_on_disk": int(os.environ["N_CLIPS"]),
    "passes": [
        {"tag": "sparse", "wall_seconds": t_sparse, "n_frames": f_sparse,
         "stride": sparse["measured"]["stride"], "episodes": sparse["measured"]["episodes"]},
        {"tag": "dense", "wall_seconds": t_dense, "n_frames": f_dense,
         "stride": dense["measured"]["stride"], "episodes": dense["measured"]["episodes"]},
    ],
    "estimator": dense.get("estimator"),
    "prompt": dense.get("prompt"),
    # THE SECOND QUESTION, and on a first run it is the more important one. An empty robot mask is
    # clip-fatal in G0c, so a non-zero rate here is a prompt problem to solve before generation.
    "empty_mask": {
        "frames": dense["measured"]["empty_frames"],
        "fraction": dense["measured"]["empty_frame_fraction"],
        "note": "Measured over this pilot's episodes only. An empty robot mask REFUSES the clip in "
                "robot_composite.check_mask -- there is no threshold in that check -- so any "
                "non-zero fraction here means clips will refuse at generation time.",
    },
    # A SAMPLE, NAMED AS ONE. Printed because a first look at the shape is worth having; it is not
    # the distribution, and no bound may sit above it.
    "area_fraction_sample": {k: dense["measured"][k]
                             for k in ("min", "median", "p95", "p99", "max")},
    "area_fraction_sample_note":
        "THREE EPISODES. measurement_qualified is false in the artifact this came from and "
        "load_area_bound refuses it. Not the corpus's distribution and not a bound's basis.",
    "measurement_qualified": False,
    "measurement_qualified_note":
        "A pilot is a smoke test. Both passes carry measurement_qualified:false and exited 3, "
        "which is robot_composite.py refusing to let a sample become the distribution a bound "
        "sits above. Nothing here is that distribution.",
}

if blockers:
    rec["blockers"] = blockers
else:
    per_frame = (t_dense - t_sparse) / (f_dense - f_sparse)
    load_s = t_sparse - per_frame * f_sparse
    full_s = per_frame * total_frames + load_s
    rec |= {
        "seconds_per_frame": per_frame,
        "load_seconds": load_s,
        "load_seconds_note":
            "Fixed startup: interpreter, torch, and the GroundingDINO + SAM 2 weights. Separated "
            "out by the two-stride slope so it is not multiplied by every frame of the corpus.",
        "full_run_seconds": full_s,
        "full_run_gpu_hours": full_s / 3600.0,
        "recommended_time_limit":
            "%02d:%02d:00" % divmod(int(((full_s * 1.25) + 1799) // 1800) * 30, 60),
        "recommended_time_limit_basis": "full_run_seconds x 1.25, rounded up to the next 30 min",
        "single_job_feasible": full_s * 1.25 < 4 * 3600,
        "single_job_feasible_note":
            "False means the full measurement does not fit one comfortable request on `common` and "
            "has to be sharded: --shard I --num-shards N per array task, then --merge over the "
            "shard artifacts, which pools the raw per-frame area fractions and takes the five "
            "numbers ONCE (a median does not decompose, and neither does a p95). Submit it with "
            "SHARD=1 on this file.",
    }
    # THE SHARD TABLE, ON THE REAL FRAME COUNTS. The partition balances episode COUNTS only
    # approximately and FRAMES not at all (249-749 per episode), so an even split is a floor and
    # not a plan. The manifest knows the frames, so the heaviest shard is computed rather than
    # estimated, for every N an operator might reasonably pick.
    table = []
    for n in range(2, 9):
        frames = [0] * n
        for key, count in episodes:
            frames[shard_of(key, n)] += count
        heaviest = max(frames)
        seconds = heaviest * per_frame + load_s
        table.append({
            "num_shards": n,
            "frames_per_shard": frames,
            "heaviest_shard_frames": heaviest,
            "heaviest_shard_seconds": seconds,
            "heaviest_shard_minutes": seconds / 60.0,
            "with_margin_minutes": seconds * 1.25 / 60.0,
            "suggested_time_limit":
                "%02d:%02d:00" % divmod(int(((seconds * 1.25) + 899) // 900) * 15, 60),
            "wall_over_heaviest": 4 * 3600 / seconds,
            "total_gpu_hours": (n * load_s + per_frame * total_frames) / 3600.0,
            "fits_maxwall": seconds * 1.25 < 4 * 3600,
        })
    rec["per_shard_estimates"] = table
    rec["per_shard_estimates_note"] = (
        "Computed from the manifest's real per-episode frame counts under the same blake2b "
        "partition the shards use, so heaviest_shard_frames is exact rather than an even-split "
        "floor. Choosing N is ALSO a submission-limit decision: MaxSubmitJobsPU=8 and MaxJobsPU=4 "
        "across the QoS, an N-task array is N submissions, and DenyOnLimit rejects an over-limit "
        "array outright. Check `squeue -u $USER -r` before picking N."
    )

path = out / "ROBOT_MASK_AREA_PILOT.json"
path.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print(f"\n=== wrote {path}")
PY
  echo "=== PILOT done. Read ${OUT}/ROBOT_MASK_AREA_PILOT.json:"
  echo "===   - empty_mask.fraction first. Non-zero means clips will refuse at generation time."
  echo "===   - per_shard_estimates, then pick NUM_SHARDS against `squeue -u \$USER -r`."
  exit 0
fi

# -- SHARD ---------------------------------------------------------------------------------------
if (( SHARD_MODE )); then
  # SUBMITTED WITHOUT --array, THIS WOULD MEASURE SHARD 0 AND STOP. Everything downstream would be
  # well formed: one shard artifact, and a merge that stamps the result disqualified because the
  # rest are missing — the right outcome in the wrong place, hours of H200 later. Catch it here.
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    echo "FATAL: SHARD=1 but this job is not an array task (SLURM_ARRAY_TASK_ID is unset)."
    echo "       The shard index IS the array index; there is no default, because the default would"
    echo "       be shard 0 and a run that measured a third of the corpus and exited 0."
    echo "         SHARD=1 NUM_SHARDS=3 sbatch --array=0-2%3 --time=02:00:00 $(basename "$0")"
    exit 1
  }
  # No apostrophes in this message; see the note on the MERGE branch.
  : "${NUM_SHARDS:?SHARD needs the denominator of the partition. It is not derived from the array bounds, because the array can legitimately be a subset (re-running two failed shards is --array=1,2) while the partition it belongs to still has N pieces, and a merge cannot say a shard is missing without knowing N. Pass the same NUM_SHARDS for every submission of one partition, for example NUM_SHARDS=3.}"
  [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} is not a positive integer."
    exit 1
  }
  SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
  # Checked as a STRING before it is used in an arithmetic context. Inside (( )) bash treats a
  # non-numeric word as a variable name and an unset variable as 0, so a malformed
  # SLURM_ARRAY_TASK_ID would sail through the range test below as shard 0.
  [[ "${SHARD_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "FATAL: SLURM_ARRAY_TASK_ID=${SHARD_INDEX} is not a non-negative integer."
    exit 1
  }
  if (( SHARD_INDEX >= NUM_SHARDS )); then
    echo "FATAL: array task ${SHARD_INDEX} is outside the partition 0..$((NUM_SHARDS - 1))."
    echo "       The array indices ARE the shard indices, so --array must be a subset of"
    echo "       0-$((NUM_SHARDS - 1)) whenever NUM_SHARDS=${NUM_SHARDS}. robot_composite.py refuses an"
    echo "       out-of-range index rather than measuring the empty set, and this says so sooner."
    exit 1
  fi
  # Not fatal, and it must not be: re-running two failed shards is --array=1,2 over the same
  # partition, and that is the resumable shape this mode exists for. Said out loud, because a FIRST
  # submission whose array does not span the partition produces a set of shards whose merge is
  # stamped disqualified, and the operator should learn that now and not in the merge job.
  if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "${NUM_SHARDS}" ]]; then
    echo "=== NOTE: this array has ${SLURM_ARRAY_TASK_COUNT:-?} task(s) and the partition has"
    echo "===       ${NUM_SHARDS}. Correct for a re-run of specific shards, wrong for a first"
    echo "===       submission. The merge insists on all ${NUM_SHARDS}."
  fi

  mkdir -p "${SHARD_DIR}"
  SHARD_OUT=${SHARD_DIR}/shard-${SHARD_INDEX}.json

  # RESUMABLE, in the shape 97 gets from its DONE stamps: a shard that already landed is not
  # re-measured. This is what makes "re-submit the two that failed" cost two shards rather than a
  # partition, and it is what makes a preempted array task free to simply re-run. The stamp is the
  # artifact itself and it is VALIDATED, not merely present: a truncated file from a task killed at
  # the wall must not be mistaken for a finished one, and neither must a shard for a different
  # index or a truncated measurement.
  if [[ -f "${SHARD_OUT}" && "${FORCE:-0}" == "0" ]]; then
    if python - "${SHARD_OUT}" "${SHARD_INDEX}" "${NUM_SHARDS}" <<'PY'
import json, sys
path, want_i, want_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    d = json.loads(open(path).read())
except Exception as exc:
    print(f"    not reusable: {path} does not parse ({exc})"); raise SystemExit(1)
sh = d.get("shard") or {}
for label, got, want in (("schema", d.get("schema"), "wam.robot_mask_area_shard/1"),
                         ("shard.index", sh.get("index"), want_i),
                         ("shard.num_shards", sh.get("num_shards"), want_n)):
    if got != want:
        print(f"    not reusable: {label} is {got!r}, wanted {want!r}"); raise SystemExit(1)
if not d.get("measurement_qualified"):
    why = "; ".join(d.get("measurement_disqualified_reasons") or ["(no reason recorded)"])
    print(f"    not reusable: measurement_qualified is false -- {why}")
    raise SystemExit(1)
m = d.get("measured") or {}
print(f"    reusable: {sh.get('n_episodes_in_shard')} episodes, {m.get('frames')} frames, "
      f"max {m.get('max')}, empty {m.get('empty_frames')}")
raise SystemExit(0)
PY
    then
      echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} already landed. Skipping."
      echo "===   FORCE=1 re-measures it."
      exit 0
    fi
    echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} exists and is NOT reusable (above)."
    echo "===   Re-measuring. The old file is overwritten, so no stale shard can reach the merge."
  fi

  echo "=== SHARD ${SHARD_INDEX} of ${NUM_SHARDS} -> ${SHARD_OUT}"
  echo "===   Episodes are assigned by a blake2b digest of the episode ID, not by a slice of the"
  echo "===   episode list: shards are computed by different jobs at different times, and a range"
  echo "===   would silently re-partition the corpus the moment a clip is added or removed."
  echo "===   No --limit and no --stride: either would stamp this shard measurement_qualified:false"
  echo "===   and disqualify the whole merge."

  # The pilot's own numbers, re-read here rather than trusted from the header comment, turned into
  # THIS shard's expected cost. A header comment cannot notice that the pilot was re-run; this can,
  # and unlike 103's version it uses the exact frame count of this shard rather than an even split.
  if [[ -f "${OUT}/ROBOT_MASK_AREA_PILOT.json" ]]; then
    NUM_SHARDS="${NUM_SHARDS}" SHARD_INDEX="${SHARD_INDEX}" \
    PILOT_JSON="${OUT}/ROBOT_MASK_AREA_PILOT.json" python - <<'PY' || true
import json, os, time
d = json.loads(open(os.environ["PILOT_JSON"]).read())
n, idx = int(os.environ["NUM_SHARDS"]), int(os.environ["SHARD_INDEX"])
per, load = d.get("seconds_per_frame"), d.get("load_seconds")
if per is None or load is None:
    print("===   pilot carries no slope (blockers: "
          + "; ".join(d.get("blockers") or ["?"]) + ") -- no per-shard estimate")
else:
    row = next((r for r in d.get("per_shard_estimates") or [] if r["num_shards"] == n), None)
    if row is None:
        print(f"===   pilot has no table row for N={n}; per-frame {per:.4f}s, load {load:.0f}s")
    else:
        frames = row["frames_per_shard"][idx]
        est = frames * per + load
        print(f"===   pilot: {per:.4f} s/frame, load {load:.0f}s. This shard is {frames} frames "
              f"-> ~{est:.0f}s ({est/60:.1f} min), x1.25 = {est*1.25/60:.1f} min")
        end = os.environ.get("SLURM_JOB_END_TIME")
        if end and end.isdigit():
            left = int(end) - int(time.time())
            print(f"===   walltime remaining {left}s ({left/60:.1f} min)")
            if est * 1.25 > left:
                print("===   WARNING: the estimate with the pilot margin exceeds this request. The")
                print("===   run is not stopped -- the estimate extrapolates from 3 of 402 episodes")
                print("===   and the measurement is the better evidence -- but if this task dies at")
                print("===   the wall, that is why. Raise --time (4 h is the cap) or raise")
                print("===   NUM_SHARDS and re-run the partition.")
PY
  fi

  START=${SECONDS}
  # `set -e` is on, so a bare invocation followed by RC=$? never sees a non-zero status: the shell
  # is already gone by the time the assignment runs. The `if` is where set -e does not fire.
  if python "${WAM}/scripts/robot_composite.py" measure \
    --manifest "${MANIFEST}" \
    --shard "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --out "${SHARD_OUT}"; then RC=0; else RC=$?; fi
  ELAPSED=$((SECONDS - START))
  echo "=== shard ${SHARD_INDEX} exited ${RC} after ${ELAPSED}s"
  if (( RC != 0 )); then
    echo "FATAL: shard ${SHARD_INDEX} exited ${RC} and its artifact must not reach the merge."
    echo "       Exit 3 means the shard measured something truncated, which disqualifies the whole"
    echo "       merge -- the merge refuses to let one truncated shard into a distribution a bound"
    echo "       will sit above. Exit 1 means it refused before measuring. Either way, fix the"
    echo "       cause and re-run THIS index:"
    echo "         SHARD=1 NUM_SHARDS=${NUM_SHARDS} \\"
    echo "           sbatch --array=${SHARD_INDEX} --time=02:00:00 $(basename "$0")"
    exit "${RC}"
  fi
  echo "=== NEXT, once every shard has landed (this is NOT the distribution and no shard is):"
  echo "===   MERGE=1 NUM_SHARDS=${NUM_SHARDS} sbatch --qos=2cpu-single-host --gres=none \\"
  echo "===     --cpus-per-task=2 --mem=8G --time=00:20:00 $(basename "$0")"
  exit 0
fi

# -- FULL ----------------------------------------------------------------------------------------
echo "=== FULL: ${N_CLIPS} clips, no --limit and no --stride. Either would stamp"
echo "===       measurement_qualified:false and exit 3, which is the whole point of them: the"
echo "===       distribution a bound sits above is the whole corpus or it is not the distribution."
echo "===       The arithmetic in this file's header says this does not fit a 4 h wall with any"
echo "===       margin. Submit it only with a --time the PILOT recommends."
START=${SECONDS}
if python "${WAM}/scripts/robot_composite.py" measure \
  --manifest "${MANIFEST}" \
  --out "${ARTIFACT}"; then RC=0; else RC=$?; fi
ELAPSED=$((SECONDS - START))
echo "=== robot_composite.py measure exited ${RC} after ${ELAPSED}s"
if (( RC != 0 )); then
  echo "FATAL: the FULL measurement exited ${RC}. Whatever is at ${ARTIFACT} is not the corpus's"
  echo "       distribution: exit 3 means it is stamped measurement_qualified:false and"
  echo "       load_area_bound refuses it; any other code means it did not finish. Nothing is"
  echo "       copied to ${OUT}."
  exit "${RC}"
fi
cp "${ARTIFACT}" "${OUT}/"
echo "=== copied the artifact to ${OUT}/ for sync.sh --pull"
echo "=== NEXT: read empty_frames, then the five numbers, then a human writes max_frame_fraction"
echo "===       and bound_rationale and commits the file. This job leaves the bound null on every"
echo "===       path, deliberately."
