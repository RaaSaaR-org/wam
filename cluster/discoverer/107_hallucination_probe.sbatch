#!/bin/bash
# Step 27 — T-040 / PR-08 `T40_RULE_V8`: can Cosmos-Transfer2.5 hallucinate a manipulator into a
#           robot-free frame?
#
#   CONTROL=depth:0.5,seg:0.5 \
#     sbatch --qos=ehpc-aif-2026pg01-905 --time=01:00:00 107_hallucination_probe.sbatch
#
#   (the #SBATCH block below already carries --qos and --time; both are repeated on the command
#   line in the recipe above because a submission that forgets --qos lands on `normal`, which is
#   1 minute and 0 GPUs, and because an honest short --time is what backfills on `common`.
#   CONTROL is REQUIRED and has no default, for the reason restyle_transfer25.py refuses to default
#   it. EPISODES / STYLE_COUNT / FRAMES default to 2 / 2 / 96 and may only be LOWERED.)
#
# ================================================================================================
# THIS JOB REFUSES TO RUN UNTIL docs/preregistration/PR-08-V8-hallucination-probe.md IS SIGNED.
# It is written and committed BEFORE that signature exists, deliberately: the decision the owner is
# being asked for should be about running the measurement, not about whether a job could be built.
# ================================================================================================
#
# WHAT THIS MEASURES.
# `robot_composite.check_mask` refuses a clip on an empty robot mask with no threshold — "zero is
# zero" — and its error text offers the operator a dichotomy: either the source corpus is not what
# PR-08 §3 describes, or the detector is not finding a robot that is there. The diagnosis
# (runs/pr08-robot-mask-empty/DIAGNOSIS.json, scripts/diagnose_robot_mask_empty.py) settled that
# question and found a THIRD answer: the robot is genuinely absent from 36.2 % of frames, clustered
# in the approach and the retreat, with ZERO detector failures across 917 robot-present frames. So
# the empty masks are correct answers.
#
# That leaves one premise underneath the refusal UNMEASURED, and this job measures it: if the robot
# is absent from a frame, and the generator is asked to restyle it, DOES THE GENERATOR PUT A ROBOT
# THERE? Nothing in this repository knows. It is the question the threshold-free option the T-040
# record proposes — accept a clip when the source mask and the generated mask are both empty —
# stands or falls on.
#
# WHY THIS NEEDS ITS OWN REGISTRATION, AND THE CIRCULARITY THAT IS THE REAL REASON.
# T40_RULE_V1 §1 licenses exactly four things: writing the pipeline, committing the style partition,
# measuring the estimator error budget, and timing one episode on an H200. A diagnostic restyle of a
# few hundred frames is NOT a corpus — but it is not on that list either, and §1 licenses by
# enumeration rather than by exclusion. Hence T40_RULE_V8, and hence the signature check below.
#
# And the loop that no licensed activity can break:
#
#     the G0c robot-mask area bound  ->  needs the timing run's throughput?  no: it needs a
#         DISTRIBUTION, and the corpus's observed maximum IS the defect the bound exists to catch
#         (10 of the 11 whole-tablecloth masks contain no robot at all)
#     the licensed TIMING run        ->  refuses without the bound. 97_transfer25_restyle.sbatch
#         makes the timing path depend on G0c's own prerequisites ON PURPOSE (97:1178), and job
#         189644 died on exactly that refusal
#     the bound                      ->  cannot be chosen without knowing what an empty robot mask
#         MEANS on a robot-free frame
#     which is                       ->  this job
#
# bound -> timing -> this answer -> bound. V8 §1.2 states it in those terms rather than presenting
# this job as a convenience, because it is not one: PR-08 cannot proceed until somebody measures it.
#
# ------------------------------------------------------------------------------------------------
# WHAT THIS JOB MUST NOT BECOME, AND WHAT STOPS IT
# ------------------------------------------------------------------------------------------------
#
#   (a) A CORPUS GENERATOR. This is the whole design constraint. A narrowly-licensed generation path
#       that can be widened by an environment variable is exactly the hole §1 exists to close. Five
#       mechanisms, none of which is a comment:
#
#         1. THE FRAME CAP IS NOT READABLE FROM THE SUBMIT LINE. The guard block below carries
#            CAP_EPISODES=3, CAP_STYLES=3, CAP_CLIPS=6, CAP_FRAMES_PER_CLIP=121 and
#            CAP_TOTAL_FRAMES=726 as literals. EPISODES / STYLE_COUNT / FRAMES may be LOWERED from
#            the submit line and anything above a cap EXITS 2 — refused, never clamped, because a
#            clamp turns "I asked for 4000 frames" into a run of 726 that looks like it did what was
#            asked. scripts/probe_hallucination.py carries the same five numbers as module constants
#            and refuses independently, so neither file can be widened alone.
#         2. IT WRITES NOTHING A CONSUMER CAN READ. scripts/assemble_restyled_lerobot.py files a
#            clip directory by glob("*.mp4"); 97's harvest keys on a file called vision.mp4. The
#            probe renames every generated clip to *.mp4.quarantined the moment the framework
#            returns it, writes no manifest.json, no work.jsonl, no sample_outputs.json, no parquet
#            and no meta/ — and BOTH the script and this job then WALK the output tree and fail
#            non-zero if any such name came back.
#         3. THE OUTPUT DIRECTORY NAME SAYS WHAT IT IS, and both this job and the script refuse a
#            path that does not.
#         4. IT DOES NOT ROUTE THROUGH 97's WORK LIST, G0c COMPOSITE OR HARVEST. No STYLE_SET, no
#            CHUNK_INDEX, no work.jsonl, no CEILING_GPU_H, no NOT_TRAINING_DATA-clearing gate job.
#            Setting any of 97's variables on this submit line is REFUSED below rather than ignored:
#            an operator who typed them meant to run the other job.
#         5. IT DOES NOT CALL check_mask. The gate is the thing whose premise is being measured;
#            running it would refuse every unit before a number existed. Nothing here edits
#            scripts/robot_composite.py, and nothing here derives or suggests an area bound.
#
#   (b) THE §8 ITEM 3 THROUGHPUT MEASUREMENT. It is not, and V8 §4 item 3 forbids the substitution
#       in the registered document as well as here. Item 3 requires ONE TIMED EPISODE on an H200 at
#       640x480 through the corpus path; this probe restyles truncated 96-121 frame sub-clips
#       through a probe-specific path. A seconds-per-frame number taken from it would be a
#       measurement of a different run shape sold as the licensed one. Item 3 STAYS OPEN. The
#       artifact stamps its own elapsed time `not admissible` for that item, in those words.
#
#   (c) A RESULT. No number this job writes is a PR-08 result: not P/F/N/I, not L1/L2, not a
#       GEOM_TOL, not an EST_DRIFT_P95, not a bound. T40_RULE_V1 §1's third prohibition is
#       untouched and V8 adds itself to it.
#
#   (d) A VERDICT A MODEL SIGNED OFF. The probe's H/N/U letter is an UPPER BOUND, not the finding.
#       The committed masker grounds THE APPLE on 41 % of robot-absent SOURCE frames corpus-wide
#       (DIAGNOSIS.json secondary_finding_false_positives), so a non-empty mask on a generated frame
#       is not by itself a manipulator. PROBE.json carries human_review.looked_at = false and it
#       stays false until a person has read the contact sheets. THIS ARTIFACT EXISTS TO BE LOOKED
#       AT.
#
# ------------------------------------------------------------------------------------------------
# THE CAP, IN CONTEXT, SO IT CAN BE JUDGED RATHER THAN ACCEPTED
# ------------------------------------------------------------------------------------------------
#   6 clips x 121 frames  =  726 generated frames, the hard ceiling
#                         =  1.70x the ONE episode (427 mean frames) that §8 item 3's ALREADY
#                            LICENSED timing run restyles
#                         =  0.42 % of one style-instance (171 625 frames)
#                         =  0.017 % of the registered partition (25 style-instances, ~4.29 M)
#   default               =  2 episodes x 2 committed TRAIN_STYLES x 96 frames = 384 frames
#
# The frames are chosen BECAUSE THEY ARE ROBOT-FREE, by diagnose_robot_mask_empty.py's EXISTING
# reference predicate at its primary setting and at the band DIAGNOSIS.json records
# (instrument.band_px, 800 / 3000). No new predicate is written and no constant is re-chosen; the
# band is a module constant rather than a flag, because a band typed at submit time is a per-run
# decision about which frames count as robot-free, recorded nowhere anybody would look.
#
# ONE CONTIGUOUS RUN PER EPISODE, AND THAT IS WHY THIS IS NOT THE TIMING RUN WITH A SMALLER NUMBER.
# Transfer2.5 conditions on a video. A robot-free frame sitting next to robot-present frames can
# acquire a manipulator by TEMPORAL PROPAGATION from its neighbours, which is a different finding
# from invention out of nothing. The probe takes the longest CONTIGUOUS run of robot-absent frames
# in an episode — the diagnosis measured those runs at a median of 95.5 and a max of 175 frames on
# the pilot's episodes, with decile 0 at 65.6 % absent and decile 9 at 94.8 % — so every conditioning
# frame the generator sees is robot-free. The whole-episode timing run cannot make that separation,
# which is why it could not answer this even if it were unblocked.
#
# THE PROMPTS ARE THE COMMITTED ONES, AND THAT IS LOAD-BEARING.
# All 16 committed prompts — 10 train, 5 eval, 1 identity — end with the clause
# "Scene geometry, camera framing and the robot are unchanged." On a robot-free frame every one of
# them therefore refers to "the robot" as though one were present. That is not a defect in the
# partition (the clause holds the robot still on the 56.5 % of clip-time it is in shot) but it is a
# plausible MECHANISM for the failure this asks about, and it is why the probe must run the
# committed prompts verbatim: a probe under a prompt that does not mention a robot would measure a
# generator this project is not going to run. TRAIN_STYLES only — arm B is what the intervention
# runs, EVAL_STYLES is the held-out domain and has no business being touched before the experiment,
# and the identity style is arm C's question.
#
# ------------------------------------------------------------------------------------------------
# WHY THE H.264 TREE
# ------------------------------------------------------------------------------------------------
# SOURCE defaults to data/pr08-apple-640x480-h264-lossless, not data/pr08-apple-640x480. The corpus
# ships AV1 and the cv2 4.11.0 in 98's generation venv is built against avcodec 59.37.100, which has
# no AV1 decoder and DOES NOT RAISE about it: VideoCapture opens the container, believes the header
# and fails every read. The H.264 tree is proven BIT-EXACT per clip (transcode_corpus_lossless.py,
# TRANSCODE_PROOF.json) and job 189613 verified all 402 clips decode under the generation venv's own
# cv2. Same reason 106 uses it.
#
# ------------------------------------------------------------------------------------------------
# BILLING, WHICH IS WHERE THE MEMORY REQUEST COMES FROM — AND THE WALL, WHICH IS A REQUEST AND NOT
# AN ESTIMATE
# ------------------------------------------------------------------------------------------------
# Billing per minute = GPUs x 1.0 + MemGB x 0.25 + Threads x 0.036, so MEMORY DOMINATES.
#
#     job 189584  the restyle path (Transfer2.5 + the masker)  MaxRSS ~38 GiB against a 96 G request
#     job 189588  the estimator stack alone                    MaxRSS  4.75 GiB against 192 G
#
# PROVENANCE OF THOSE TWO NUMBERS, BECAUSE THEY ARE NOT EQUALLY WELL ATTESTED IN THIS TREE. 189588's
# 4.75 GiB is committed here, in 105's and 106's own headers. 189584's ~38 GiB is CARRIED FROM THE
# T-040 RECORD and no artifact in this repository holds it — this file cites it the way
# docs/isaac-est-drift-runbook.md §4.7 cites 189588's end-to-end run: as a citation, not as a local
# measurement. It is the larger of the two and it is the one this request is sized on, so the error
# it could carry is in the direction of asking for too little memory. If the job dies on OOM, that
# citation is what to distrust first, and `sacct -j <id> --format=MaxRSS` on the failed run is the
# measurement that replaces it.
#
# This job loads BOTH — the generator and the committed masker — so 189584 is the number that binds.
# 64 G is 1.68x that measured peak and gives a rate of
#
#     1.0 (GPU) + 64 x 0.25 (16.0, memory) + 26 x 0.036 (0.936, threads) = 17.936 units/min
#
# A one-hour allocation therefore costs at most 1 076 billing units, of which the GPU is 60 and the
# memory is 960. Dropping to 48 G would save 4.0 units/min and leave only 1.26x headroom over a
# measured peak; that trade is available and is not taken, because a job that dies on memory costs
# the whole allocation rather than a quarter of it. The 26 threads are the fair-share thread rate
# for one GPU and are 0.9 of ~18 units — cutting them is a rounding error against the memory.
# NEVER exceed 26 threads or 257 GB per GPU, and NEVER omit --qos.
#
# THE WALL IS AN HONEST GUESS AND THE FILE SAYS SO. **No Cosmos-Transfer2.5 throughput has ever been
# measured on this project.** THROUGHPUT.json's "9.56 GPU-h" came from job 189142, a run that
# generated ZERO clips, and 106's header names publishing that number as the mistake not to repeat.
# So --time=01:00:00 is a request, not a derivation, and the probe is built to survive being wrong
# about it: PROBE.json is rewritten after EVERY unit with `complete: false`, so a wall kill leaves
# the units that finished plus a file that says it is partial. If the wall takes it, resubmit with
# EPISODES=1 STYLE_COUNT=2 (or FRAMES=48) and read the partial artifact for what it cost per unit —
# and note that that per-unit number is still NOT admissible for §8 item 3.
#
# NO --requeue AND NO USR1 TRAP. 97 has both because its chunked generation has to resume; this job
# has neither, on purpose. A requeued probe would re-generate frames already generated and could
# walk past the cap across restarts, which is the one thing (a) exists to prevent. A wall kill here
# is final, and the partial artifact is the record.
#
# QUEUE. MaxJobsPU=4, MaxSubmitJobsPU=8, DenyOnLimit — an over-limit submission is REJECTED at
# submit time, not queued. Three of the owner's jobs are in flight as this is written
# (`squeue -u $USER -r` / `dplus`), so one running slot and five submit slots are free: this single
# job can be submitted and can run in the same scheduling round. It is NOT an array and must not be
# made one.

#SBATCH --job-name=wam-t040-halluc-probe
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# NEVER OMIT --qos: `normal` is 1 minute and 0 GPUs, which is a trap and not a fallback.
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
# 64 G, from 189584's measured 38 GiB peak on this exact stack. See BILLING above; this is the
# single biggest lever on the allocation and it is set from a measurement.
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t040-halluc-probe.%j.out

set -euo pipefail

# caches.sh first, before any tool decides for itself where to write: /home is 2 GiB and ~100 k
# inodes and one HF snapshot fills it.
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480-h264-lossless}
# The name is part of the guarantee: both this job and probe_hallucination.py refuse an output
# directory that does not say what it holds.
RUN_ID=${RUN_ID:-pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS}
OUT=${PROJ}/runs/${RUN_ID}
V8_DOC=${WAM}/docs/preregistration/PR-08-V8-hallucination-probe.md
PROBE=${WAM}/scripts/probe_hallucination.py

# ================================================================================================
# --- BEGIN V8 GUARDS (lifted verbatim by tests/test_hallucination_probe.py; keep self-contained) ---
# Everything in this block depends only on V8_DOC, OUT and the three optional counts. It is executed
# by the test suite exactly as it is written here, so a guard that is reworded, weakened or deleted
# fails a test rather than quietly stopping to guard.
RULE=T40_RULE_V8

# THE CAP. Literals, on purpose. There is no environment variable that raises them, and
# probe_hallucination.py carries the same five numbers as module constants so neither file can be
# widened alone. Raising them is a V9 alongside the pre-registration, not an edit and not a flag.
CAP_EPISODES=3
CAP_STYLES=3
CAP_CLIPS=6
CAP_FRAMES_PER_CLIP=121
CAP_TOTAL_FRAMES=726

EPISODES=${EPISODES:-2}
STYLE_COUNT=${STYLE_COUNT:-2}
FRAMES=${FRAMES:-96}

for pair in "EPISODES=${EPISODES}" "STYLE_COUNT=${STYLE_COUNT}" "FRAMES=${FRAMES}"; do
  value=${pair#*=}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: ${pair} is not a positive integer."
    exit 2
  fi
done

if (( EPISODES > CAP_EPISODES )); then
  echo "FATAL: ${RULE} caps this probe at ${CAP_EPISODES} episodes and EPISODES=${EPISODES} was asked for."
  echo "       The cap is a literal in this file and a module constant in probe_hallucination.py."
  echo "       It cannot be raised from the submit line. Raising it is a V9 alongside ${V8_DOC}."
  exit 2
fi
if (( STYLE_COUNT > CAP_STYLES )); then
  echo "FATAL: ${RULE} caps this probe at ${CAP_STYLES} committed styles and STYLE_COUNT=${STYLE_COUNT} was asked for."
  exit 2
fi
if (( FRAMES > CAP_FRAMES_PER_CLIP )); then
  echo "FATAL: ${RULE} caps a probe clip at ${CAP_FRAMES_PER_CLIP} frames and FRAMES=${FRAMES} was asked for."
  echo "       Refused rather than clamped: a clamp turns an operator asking for a corpus into a"
  echo "       run that looks like it did what was asked."
  exit 2
fi
if (( EPISODES * STYLE_COUNT > CAP_CLIPS )); then
  echo "FATAL: ${RULE} caps this probe at ${CAP_CLIPS} clips; ${EPISODES} episodes x ${STYLE_COUNT} styles is $((EPISODES * STYLE_COUNT))."
  exit 2
fi
if (( EPISODES * STYLE_COUNT * FRAMES > CAP_TOTAL_FRAMES )); then
  echo "FATAL: ${RULE} caps this probe at ${CAP_TOTAL_FRAMES} generated frames; this submission asks for $((EPISODES * STYLE_COUNT * FRAMES))."
  echo "       ${CAP_TOTAL_FRAMES} frames is 1.70x the ONE episode PR-08 section 8 item 3's licensed"
  echo "       timing run restyles. This probe is not a corpus and cannot be made into one here."
  exit 2
fi

# THE PRE-REGISTRATION HAS TO EXIST AND HAVE BEEN SIGNED BY THE PROJECT OWNER.
# A diagnostic restyle is not a corpus, but T40_RULE_V1 section 1 licenses by ENUMERATION and this is
# not on its list. So it is registered, and the registration is machine-checked here rather than
# remembered: the owner line must carry a name and a date, exactly one box must be ticked, and it
# must be a `signed` box rather than the `declined` one.
if [[ ! -f "${V8_DOC}" ]]; then
  echo "FATAL: ${RULE} is not registered — ${V8_DOC} does not exist."
  echo "       PR-08 section 1 licenses four things and a diagnostic restyle is not one of them."
  echo "       Nothing may be generated under a rule that has not been written down."
  exit 1
fi
V8_LINE=$(grep -m1 -E '^Project owner:' "${V8_DOC}" || true)
V8_NAME=$(printf '%s' "${V8_LINE}" | sed -E 's/^Project owner:[[:space:]]*//; s/[[:space:]]*Date:.*$//; s/_//g; s/[[:space:]]+$//')
V8_DATE=$(printf '%s' "${V8_LINE}" | sed -nE 's/.*Date:[[:space:]]*//p' | sed -E 's/_//g; s/[[:space:]]+$//')
V8_SIGNED=$(grep -cE '^[[:space:]]*(Determination:[[:space:]]*)?\[[xX]\][[:space:]]*signed' "${V8_DOC}" || true)
V8_DECLINED=$(grep -cE '^[[:space:]]*(Determination:[[:space:]]*)?\[[xX]\][[:space:]]*declined' "${V8_DOC}" || true)
if [[ -z "${V8_NAME}" || -z "${V8_DATE}" || "${V8_SIGNED}" != "1" || "${V8_DECLINED}" != "0" ]]; then
  echo "FATAL: ${RULE}'s determination is NOT SIGNED, so this job licenses nothing and refuses."
  echo "       document   ${V8_DOC}"
  echo "       owner line ${V8_LINE:-<no line starting with Project owner:>}"
  echo "       name       ${V8_NAME:-<blank>}"
  echo "       date       ${V8_DATE:-<blank>}"
  echo "       signed box ${V8_SIGNED}   declined box ${V8_DECLINED}   (want exactly 1 and 0)"
  echo
  echo "       The determination in section 8 of that document takes effect only when the project"
  echo "       owner completes the signature line. NO AGENT MAY SIGN IT, and no agent may act as"
  echo "       though it were signed. A session that fills that line in without the owner having"
  echo "       decided has forged it, and this probe would then be running on an agent's authority."
  echo "       If the owner has decided and the line was transcribed for them, the transcription is"
  echo "       the owner's decision and this check passes on it — that is the T40_RULE_V4 precedent."
  exit 1
fi

# THE OUTPUT DIRECTORY HAS TO SAY WHAT IT HOLDS.
OUT_NAME=$(basename "${OUT}")
if [[ "${OUT_NAME}" != *hallucination-probe* || "${OUT_NAME}" != *NOT-A-CORPUS* ]]; then
  echo "FATAL: ${RULE}: the run directory name ${OUT_NAME} must carry BOTH hallucination-probe and"
  echo "       NOT-A-CORPUS. This output is quarantined by construction and its path has to say so,"
  echo "       so nothing downstream can pick it up by looking like a corpus directory."
  exit 2
fi

# 97's VARIABLES ARE REFUSED, NOT IGNORED. An operator who set one of these meant to run the
# generation job. Silently ignoring them would let somebody believe they had passed a style set, a
# chunk or a ceiling to a job that has none of those things.
for var in STYLE_SET CHUNK_INDEX CHUNK_TOTAL WORK_LIST CEILING_GPU_H PARTITION_CEILING_GPU_H \
           TIMING PR08_T39_REPORTED PR08_T39_ARTIFACT PR08_OVERRIDE_T39_VOID NPROC; do
  if [[ -n "${!var:-}" ]]; then
    echo "FATAL: ${var} is set, and it belongs to 97_transfer25_restyle.sbatch, not to this job."
    echo "       This probe has no style set, no chunk, no work list, no GPU-h ceiling, no T-39"
    echo "       attestation and no override — it is not the generation path and must not look"
    echo "       like it. If you meant to generate a corpus: PR-08 section 1 still forbids it."
    exit 2
  fi
done
# --- END V8 GUARDS ---
# ================================================================================================

echo "=== ${RULE}: hallucination probe, ${EPISODES} episode(s) x ${STYLE_COUNT} style(s) x ${FRAMES} frames"
echo "===          = $((EPISODES * STYLE_COUNT * FRAMES)) generated frames, cap ${CAP_TOTAL_FRAMES}"
echo "=== signed   ${V8_NAME}  ${V8_DATE}  (${V8_DOC})"

[[ -d "${SOURCE}" ]] || { echo "FATAL: SOURCE ${SOURCE} does not exist."; exit 1; }
MANIFEST=${SOURCE}/manifest.json
[[ -f "${MANIFEST}" ]] || { echo "FATAL: ${MANIFEST} missing."; exit 1; }

# ------------------------------------------------------------------------------------------------
# 98's env, then its venv. Modelled on 97's setup and NOT on its work-list / G0c / harvest path.
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}
[[ -f "${TRANSFER_ENV}" ]] || {
  echo "FATAL: ${TRANSFER_ENV} missing. 98_build_transfer25_env.sbatch writes it and owns the .venv;"
  echo "       99_stage_transfer25_weights.sbatch appends the model id, revision and checkpoint."
  exit 1
}
# shellcheck source=/dev/null
source "${TRANSFER_ENV}"

: "${FRAMEWORK:?no Transfer2.5 checkout or venv — ${TRANSFER_ENV} does not export FRAMEWORK. Run 98_build_transfer25_env.sbatch.}"
: "${FFMPEG_PREFIX:?no ffmpeg — ${TRANSFER_ENV} does not export FFMPEG_PREFIX.}"
: "${TRANSFER_CHECKPOINT_PATH:?no staged Transfer2.5 checkpoint — ${TRANSFER_ENV} must export TRANSFER_CHECKPOINT_PATH. The generator revision is recorded with every artifact, so this must be a pinned revision and not a floating tag.}"
: "${TRANSFER_MODEL_ID:?${TRANSFER_ENV} must export TRANSFER_MODEL_ID.}"
: "${TRANSFER_MODEL_REVISION:?${TRANSFER_ENV} must export TRANSFER_MODEL_REVISION — the pinned commit, not a tag.}"
# The control blocks are required and never defaulted, for the reason restyle_transfer25.py refuses
# to default them: the choice decides how much geometry survives, and V8 section 7 item 6 records
# that it moves this answer too. No apostrophes inside a :? word; bash parses them.
: "${CONTROL:?which Transfer2.5 control blocks condition the restyle and at what weight, e.g. depth:0.5,seg:0.5. Required here for the same reason the driver requires it: the control choice decides how much of the source survives, so it is a parameter of the measurement and never a default. It is recorded in PROBE.json.}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8
export PATH=${FFMPEG_PREFIX}/bin:${PATH}
# Both from the Cosmos launch scripts and both load-bearing: the host CUDA libs have to be cleared
# and the expandable allocator is what makes the model fit at all.
export LD_LIBRARY_PATH=""
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=26
nvidia-smi -L || echo "=== nvidia-smi -L failed — the allocation check below will refuse this job"

# -- the billing ceiling, checked against what Slurm actually allocated ---------------------------
# Slurm bills ALLOCATED, not used, resources, and the #SBATCH block above can be overridden at
# submit time. A wrong override does not fail — it silently burns the billing counter faster than
# the GPU counter, and when billing runs out the remaining GPU-hours are permanently lost.
GPUS_ALLOC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}
CPUS_ALLOC=${SLURM_CPUS_PER_TASK:-1}
MEM_MB_ALLOC=${SLURM_MEM_PER_NODE:-0}
if (( GPUS_ALLOC < 1 )); then
  echo "FATAL: no GPUs allocated. A GPU job under --qos=2cpu-single-host is REJECTED and a missing"
  echo "       --qos lands on \`normal\` (gres/gpu=0). Check the submit line."
  exit 1
fi
if (( CPUS_ALLOC > 26 * GPUS_ALLOC )); then
  echo "FATAL: ${CPUS_ALLOC} threads against ${GPUS_ALLOC} GPU(s) exceeds the 26-per-GPU cap."
  exit 1
fi
if (( MEM_MB_ALLOC > 257 * 1024 * GPUS_ALLOC )); then
  echo "FATAL: ${MEM_MB_ALLOC} MB against ${GPUS_ALLOC} GPU(s) exceeds the 257 GB-per-GPU cap."
  exit 1
fi
echo "=== alloc    ${GPUS_ALLOC} GPU, ${CPUS_ALLOC} threads, ${MEM_MB_ALLOC} MB"
echo "=== billing  $(python3 -c "print(round(${GPUS_ALLOC} + ${MEM_MB_ALLOC}/1024*0.25 + ${CPUS_ALLOC}*0.036, 3))") units/min"

# The estimator weights 102 staged. Absent, the masker would either refuse (correct) or fetch its
# own at an unpinned revision (silent), and the probe's masks would then describe a segmenter nobody
# staged — while the whole measurement is a claim about THE COMMITTED masker.
STAGED=${PROJ}/checkpoints/PR08_ESTIMATORS_STAGED.json
[[ -f "${STAGED}" ]] || {
  echo "FATAL: ${STAGED} missing — the detector and segmenter checkpoints are not staged."
  echo "       Run 102_stage_sam2_weights.sbatch (free QoS, ~15 s once the cache is warm)."
  exit 1
}

# -- the committed style partition, verified exactly the way 97 verifies it -----------------------
# The probe reads the committed PROMPTS, so the same two checks apply: the rendering must match its
# committed .sha256 sidecar byte for byte, and check_style_partition.py must re-derive it from
# styles.toml. Run BEFORE the framework venv is activated, because the checker is stdlib-only and
# needs tomllib, which the module-loaded anaconda3/python3.12 guarantees.
STYLE_PARTITION=${WAM}/configs/transfer25/pr08_style_partition.json
STYLE_SOURCE=${WAM}/configs/transfer25/styles.toml
STYLE_CHECKER=${WAM}/scripts/check_style_partition.py
[[ -f "${STYLE_PARTITION}.sha256" ]] || {
  echo "FATAL: ${STYLE_PARTITION}.sha256 missing — nothing proves this is the committed rendering."
  exit 1
}
STYLES_FILE_SHA=$(sha256sum "${STYLE_PARTITION}" | cut -d' ' -f1)
STYLES_SIDECAR=$(tr -d '[:space:]' < "${STYLE_PARTITION}.sha256")
[[ "${STYLES_FILE_SHA}" == "${STYLES_SIDECAR}" ]] || {
  echo "FATAL: ${STYLE_PARTITION} does not match its committed .sha256 sidecar."
  echo "       on disk  ${STYLES_FILE_SHA}"
  echo "       sidecar  ${STYLES_SIDECAR}"
  echo "       A partition edited after it was committed is not a pre-commitment. Regenerate it"
  echo "       with check_style_partition.py --emit-json --write-hash; do not re-hash it."
  exit 1
}
python "${STYLE_CHECKER}" --partition "${STYLE_SOURCE}" --derived "${STYLE_PARTITION}" || {
  echo "FATAL: the committed style partition does not verify, so no prompt this probe ran could be"
  echo "       attributed to a committed style."
  exit 1
}

echo "=== source     ${SOURCE}"
echo "=== styles     ${STYLE_PARTITION}  (TRAIN_STYLES only; there is no flag that changes that)"
echo "=== generator  ${TRANSFER_MODEL_ID}@${TRANSFER_MODEL_REVISION}  (FROZEN)"
echo "=== checkpoint ${TRANSFER_CHECKPOINT_PATH}"
echo "=== control    ${CONTROL}"
echo "=== out        ${OUT}"

mkdir -p "${OUT}"
# The marker goes down BEFORE anything is written beside it, and before the venv is even activated.
# probe_hallucination.py rewrites it with the full text; this stub is what survives a crash in
# between, so an interrupted run still leaves a directory that says what it is. Same argument as
# 92's SMOKE file and 97's NOT_TRAINING_DATA: a downstream job can refuse a tree by `ls`.
cat > "${OUT}/NOT_A_CORPUS" <<'MARKER'
NOT A CORPUS. NOT TRAINING DATA. NOT AN EVALUATION SET.
Written by 107_hallucination_probe.sbatch under T40_RULE_V8 before the probe started.
Nothing in here may enter a training set, a LeRobot dataset, an evaluation set, or any artifact a
downstream consumer reads; no number in here is a PR-08 result; and this run does NOT satisfy
PR-08 section 8 item 3, which remains the timing run's job alone.
MARKER

# THE SEGMENTER'S OPERATING POINT IS NOT A SUBMIT-TIME DECISION. apple_sam2 reads its thresholds and
# its retry pair from these variables, deliberately, because a measurement sweep needs to move them.
# Here a threshold decides which pixels count as a robot in the SOURCE frame and in the GENERATED
# one — i.e. it decides the answer. 97 and 106 clear them for the same reason; so does this.
unset WAM_PR08_BOX_THRESHOLD WAM_PR08_TEXT_THRESHOLD || true
unset WAM_PR08_RETRY_BOX_THRESHOLD WAM_PR08_RETRY_TEXT_THRESHOLD || true
unset WAM_PR08_OBJECT_PROMPT || true

# shellcheck source=/dev/null
source "${FRAMEWORK}/.venv/bin/activate"
export PYTHONPATH="${WAM}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

# NO --backend HERE, EVER. probe_hallucination.py defaults to transfer25 and --backend null is the
# deterministic placeholder the tests run. The cluster path must stay unable to reach it, exactly as
# 97 must; a test walks this file's command line and asserts the flag is absent.
S=${SECONDS}
if python "${PROBE}" \
  --manifest "${MANIFEST}" \
  --styles "${STYLE_PARTITION}" \
  --out "${OUT}" \
  --checkpoint-path "${TRANSFER_CHECKPOINT_PATH}" \
  --control "${CONTROL}" \
  --resolution 640x480 \
  --episodes "${EPISODES}" \
  --style-count "${STYLE_COUNT}" \
  --frames "${FRAMES}" \
  --no-guardrails; then RC=0; else RC=$?; fi
echo "=== probe_hallucination.py exited ${RC} after $((SECONDS - S))s"

# -- NOTHING DOWNSTREAM MAY READ THIS TREE, AND THAT IS WALKED RATHER THAN ASSERTED ---------------
# The script does this too and refuses on it. It is repeated here because the two failures are
# different: the script's walk cannot run if the script died, and a tree left behind by a crash is
# exactly the tree somebody would later find and wonder about. assemble_restyled_lerobot.py files a
# clip directory by glob("*.mp4"); 97's harvest keys on vision.mp4.
CONSUMABLE=$(find "${OUT}" -type f \( -name '*.mp4' -o -name 'manifest.json' -o -name 'work.jsonl' \
  -o -name 'sample_outputs.json' -o -name 'chunk_metadata.json' -o -name '*.parquet' \
  -o -name 'info.json' -o -name '*.g0c.json' \) -print 2>/dev/null | head -20 || true)
if [[ -n "${CONSUMABLE}" ]]; then
  echo "FATAL: this probe left file(s) a downstream consumer could read, which T40_RULE_V8 section 4"
  echo "       item 1 forbids:"
  printf '       %s\n' ${CONSUMABLE}
  echo "       Generated clips are renamed to *.mp4.quarantined precisely so that"
  echo "       assemble_restyled_lerobot.py's glob and 97's harvest key both miss them. Delete or"
  echo "       rename them before anything else reads this tree."
  exit 3
fi
echo "=== output tree audit: nothing consumable. No mp4, no manifest, no work list, no parquet."

if (( RC != 0 )); then
  echo "FATAL: the probe exited ${RC}. Read the FATAL line above it — every refusal in that script"
  echo "       names the rule and the constant it refused on. If PROBE.json exists it carries"
  echo "       complete:false and the units that did finish."
  exit "${RC}"
fi

echo "=== NEXT, AND NONE OF IT IS AUTOMATIC:"
echo "===   1. LOOK AT THE SHEETS. ${OUT}/sheets/*__candidate_invention.png first. The verdict"
echo "===      letter in PROBE.json is an UPPER BOUND: the committed masker grounds THE APPLE on"
echo "===      41 % of robot-absent SOURCE frames, so a non-empty mask on a generated frame is not"
echo "===      by itself a manipulator. human_review.looked_at is false until a person has looked."
echo "===   2. Whatever the letter is, it licenses NOTHING. H is a finding against G0c's premise;"
echo "===      N removes one objection; U means the instrument was measured. T40_RULE_V1 section 1"
echo "===      still forbids generating a corpus, sections 8 items 3 and 4 are still open, and the"
echo "===      G0c area bound is still uncommitted and is still a person's decision."
echo "===   3. This run does NOT close section 8 item 3. The timing run is still the only thing"
echo "===      that measures throughput, and it is still blocked behind the bound."
