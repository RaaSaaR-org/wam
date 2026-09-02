#!/bin/bash
# Step 21 — T-040 / PR-08 §8 item 3: chunked, resumable Cosmos-Transfer2.5 restyle of the
# AppleToPlate corpus at 640x480.
#
#   measure first, it is a GATE and not a warm-up (PR-08 §8 item 3). PR-08 §1 licenses "timing one
#   episode on an H200" WITHOUT waiting for T-39, so the timing path asks for neither a T-39
#   attestation nor a budget — the budget is what it MEASURES. See (a) and (c) below:
#       TIMING=1 STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=1 CONTROL=depth:0.5,seg:0.5 \
#         RUN_ID=t040-transfer25-restyle-timing-<YYYY-MM-DD> \
#         sbatch --time=01:30:00 97_transfer25_restyle.sbatch
#
#   THE EXPLICIT RUN_ID IS PART OF THE RECIPE. This line passed none until 2026-08-27, so it
#   resolved to the default RUN_ID (see the -- paths -- section), where job 189142's THROUGHPUT.json
#   still sits: 0.2 s/frame, from a run whose own log says "0 success, 1 error". The timing branch's
#   first act was to `cat` that file and exit 0, and the operator read a finished measurement that
#   prices the whole partition ~10x under. That branch now refuses to reuse an artifact it cannot
#   see a successful unit inside, and refuses to reuse ANY pre-existing one on a fresh submission
#   unless REUSE_THROUGHPUT=1 says so — but a dated RUN_ID is the answer that also keeps the two
#   measurements side by side, so it belongs in the recipe rather than in the error message.
#
#   THE --time OVERRIDE IS PART OF THE RECIPE, NOT A TWEAK. The #SBATCH --time=04:00:00 below is
#   sized for a GENERATION chunk and is simply wrong for one episode: `common` churns on 1-2 hour
#   jobs, so a 4 h request cannot fit the gaps the partition actually offers and sits PENDING
#   (Priority) behind them. Measured 2026-08-20 on three jobs: 4 h requests estimated 13-17 hours
#   out, and lowering them to honest values started two within seconds. 189402 was submitted at
#   4 h on 2026-08-22, sat an hour with no start estimate, and moved only after
#   `scontrol update JobId=189402 TimeLimit=01:30:00` — which is also the fix for a job already
#   queued, since it keeps the job's age and can be raised again while it is still pending.
#   Ask for what the run needs. That is the honest number and Slurm rewards it; it is not gaming.
#   then one submission per chunk, at most 4 in flight (MaxJobsPU=4):
#       PR08_T39_REPORTED="<verdict letter> ... <YYYY-MM-DD>" PR08_T39_ARTIFACT=<path> \
#         PARTITION_CEILING_GPU_H=<whole partition> CEILING_GPU_H=<this style set's share> \
#         GEOM_STEP_FRAMES=<registered step> STAGE=1 STYLE_SET=train CONTROL=<key:weight,...> \
#         CHUNK_INDEX=1 CHUNK_TOTAL=4 sbatch 97_transfer25_restyle.sbatch
#
#   STAGE IS REQUIRED AND HAS NO DEFAULT (T40_RULE_V11, adopted 2026-08-24). `1` is the first four
#   TRAIN_STYLES in committed order plus their four matched identity repeats — 8 of 25 style-
#   instances, 3 216 clips, ~442 GPU-h against the whole partition's ~1 380 — with the eval set
#   DEFERRED, not cut. `2` is the complement (six train styles, six identity repeats, all five eval
#   styles) and V11 §3 fixes IN ADVANCE which stage-1 outcome licenses it. `all` is the unstaged
#   committed partition. STAGE=1 with STYLE_SET=eval is refused, because deferring the eval set is
#   the point rather than an oversight. The stage is hashed into WORK_SHA, carried in CHUNK_TAG,
#   stamped in chunk.env and recorded on every clip's sidecar, so two stages can neither resume
#   into one another's directory nor be told apart later by guesswork.
#
#   THE TWO GPU-h NUMBERS, AND WHY THERE ARE TWO (changed 2026-08-15 — see (e) below).
#     PARTITION_CEILING_GPU_H  the ceiling T40_RULE_V2 §3 step 2 requires: derived from the timed
#                              measurement over the WHOLE partition, all 25 style-instances /
#                              10 050 clips (train + eval + identity). It is the SAME number in
#                              all three submissions of a RUN_ID, it is pinned by the first of
#                              them in ${OUT}/run.env, and a later submission that contradicts it
#                              is refused.
#     CEILING_GPU_H            THIS invocation's SHARE of that ceiling, per T40_RULE_V2 §3 step 3
#                              ("split that whole-partition ceiling across the STYLE_SET
#                              invocations ... so the sum of the CEILING_GPU_H values actually
#                              passed is bounded by the whole-partition figure"). It differs per
#                              style set, because the three sets are not the same size: at the
#                              committed partition the shares are 10/25, 5/25 and 10/25 of the
#                              whole. Each share is pinned per style set under
#                              ${OUT}/ceiling_shares/, and the shares are required to SUM within
#                              PARTITION_CEILING_GPU_H — that sum bound is the thing V2 asks for,
#                              and it is checked here rather than trusted.
#   Both are enforced by ONE persisted ledger under ${OUT}/gpu_ledger/: every pass of every chunk
#   of every style set reserves its worst case there before it runs, this style set's reservations
#   must stay within its share, and the run total across all three must stay within
#   PARTITION_CEILING_GPU_H. The same RUN_ID must be used for all three submissions or the ledger
#   sees nothing to add up.
#
# READ docs/preregistration/PR-08-photoreal-augmentation.md FIRST. This job decides nothing. It
# produces restyled clips and the record that makes them gateable. G0a and G0b are checkers over a
# finished corpus and neither runs here. G0c is different in kind and DOES run here, because §6
# solves it by construction rather than by a threshold: video_fidelity cannot see the
# generic-manipulator defect and any IoU threshold on the robot mask "would be a coined number", so
# the real robot's pixels are composited back over every generated frame inside
# scripts/restyle_transfer25.py and the defect cannot enter. A gate solved by construction has to
# live in the construction; there is nothing for a later checker to check.
#
# WHAT THIS IS. Cosmos-Transfer2.5, FROZEN, restyling frames of episodes that already have actions.
# The labels are the recorded teleop trajectory, carried over unchanged — the generator only
# perturbs pixels. That is the entire reason PR-06's negative (generated video priced as
# supervision, 16.656 vs 12.020 for standing still) does not settle this, and it is a narrow
# reason: it does not predict the restyle will be good, only that PR-06's number is about a
# different question. Fine-tuning a generator is T-041 / PR-09 and a different file.
#
# WHAT IT MUST NOT BECOME.
#
#   (a) NOT GATED ON T-39. PR-08 §1: PR-07 §7's freeze does not name Transfer2.5 by letter, and
#       this document binds itself to the freeze's REASON anyway — until T-39 reports whether ANY
#       method clears the bar on this corpus, "the data is wrong" and "the method is wrong" are
#       not separable, and generating data is a bet on the first. So the GENERATION path refuses
#       to start without PR08_T39_REPORTED, and writes the string verbatim into the record. No
#       clip from this pipeline can exist without carrying the decision that allowed it.
#
#       THE TIMING PATH IS EXEMPT, AND THAT IS NOT A LOOPHOLE — IT IS §1 READ LITERALLY. §1's
#       Licenses sentence is: "writing the pipeline, committing the style partition, measuring the
#       estimator error budget (§4), and timing one episode on an H200 (§8 item 3)." Its Forbids
#       sentence, the one conditioned on T-39, names "generating a corpus, training any weight on
#       generated frames, and quoting any number from this document as a result." Timing is on the
#       licensed side, unconditionally. Requiring the attestation for it — as this file did until
#       2026-08-15 — made the only way to perform the licensed measurement typing a T-39 verdict
#       that does not exist yet. That does not protect the gate; it teaches the operator that the
#       attestation is a formality to be filled in, which is precisely the habit the GENERATION
#       path depends on not having. The requirement is therefore enforced where §6's "generation
#       does not start" actually bites, and nowhere else.
#
#       AND IT IS MACHINE-CHECKED, not merely typed. PR08_T39_REPORTED must carry one of PR-07
#       §5's verdict tokens and an ISO date, and PR08_T39_ARTIFACT must be an existing, non-empty
#       file — T-39's own committed result — whose sha256 and verdict-bearing fields are recorded
#       beside the prose. A prose string alone is unfalsifiable; a path is a thing that either
#       exists or does not.
#
#       AND **VOID IS REFUSED, NOT ACCEPTED** (changed 2026-08-15, and this reverses the file's
#       previous behaviour). Until today the token check accepted (P|N|M|I|VOID) and the error
#       message's worked example was literally `VOID (labels) 2026-08-14` — so the gate satisfied
#       §1's letter ("T-39 has reported") while inverting §1's REASON, and taught the operator to
#       type the one verdict that must stop this job. PR-07 §4 defines what a VOID is: if the
#       ground-truth `action` column itself cannot clear L1 under our scorer, then "no policy
#       trained on this dataset can clear our bar, and the finding is about our label pipeline."
#       PR-07 §6's VOID row licenses exactly one thing — a defect report against the adapter or
#       the label pipeline — and forbids "any statement about GR00T". PR-08 §1's stated reason for
#       gating on T-39 is that "until T-39 reports whether ANY method clears the bar on this
#       corpus ... generating data is a bet on the first". A VOID does not resolve that bet; it
#       LOSES it, on the record, in the direction that says the corpus/labels are the problem.
#       Generating 10 050 restyled clips whose labels are copied from that same label pipeline is
#       therefore the single worst use of the allocation the verdict admits.
#
#       So a VOID attestation refuses. There IS an override, because a human may later hold
#       evidence this file cannot see, and it is deliberately NOT a boolean: it is a separate,
#       explicitly named variable, PR08_OVERRIDE_T39_VOID, whose accepted value is one exact
#       sentence stating what is being overridden. A flag reading `=1` is something an operator
#       adds to make an error go away; a sentence that has to be typed out is a claim they made.
#
#   (b) TRAINING DATA. Nothing this job writes may enter any training corpus until G0a (label
#       identity under screen_corpus --expect) and G0b (geometry invariance within
#       GEOM_TOL - EST_DRIFT_P95) have passed. They are VOID gates, not reports. This job drops a
#       NOT_TRAINING_DATA marker beside every clip directory it writes so anything downstream can
#       refuse the tree by `ls`, and the marker is removed by whatever runs those two gates, never
#       by this file.
#
#       G0c IS NOT ON THAT WAITING LIST BECAUSE IT HAS ALREADY HAPPENED, and that is §6's design
#       rather than a relaxation of it. "The real robot's pixels are unconditionally composited
#       back over every generated frame ... The defect cannot enter, and no threshold is needed."
#       The driver does it per clip, between the model writing the mp4 and the unit being called a
#       success; a unit whose composite refused never gets that status and its output is renamed to
#       vision.uncomposited.mp4, so the harvest cannot file it either.
#
#       AND THIS JOB VERIFIES IT RATHER THAN ASSERTING IT. The harvest reads each unit's own
#       sample_outputs.json and refuses to file a clip that cannot prove the composite ran on all
#       of ITS frames — see the long comment on harvest() for the five checks — then writes the
#       proof beside the clip as <unit>.g0c.json. A clip in the directory with no proof beside it
#       is fatal, which is also what covers clips filed by any earlier driver. The claim this
#       record makes about G0c is therefore a count, not a sentence. The robot-mask IoU inside the
#       proof is a DIAGNOSTIC ON THE GENERATOR AND NEVER A GATE, which §6 states twice; nothing
#       here compares it to anything.
#
#   (c) A BUDGET IT INVENTED. PR-08 §8 item 3: "No budget line exists until that measurement
#       does." CEILING_GPU_H and PARTITION_CEILING_GPU_H are therefore REQUIRED WITH NO DEFAULT.
#       A default here would be a number nobody measured, dressed as a pre-registered term.
#       TIMING=1 produces the one measurement PR-08 licenses — a single episode on a single H200
#       at 640x480 — and the full mode refuses to run until THROUGHPUT.json exists, exactly as
#       94's probe gates its train.
#
#       AND THE TIMING PATH DOES NOT ASK FOR EITHER OF THEM (changed 2026-08-15). Until today the
#       `: ${CEILING_GPU_H:?}` guard sat above the TIMING branch, so the measurement that DERIVES
#       the ceiling refused to run without the ceiling. There is no honest value to type there —
#       §8 item 3 says the budget line does not exist yet — so the only way to perform the
#       licensed measurement was to invent one, which is the same defect as a default with an
#       extra step: an operator-invented number instead of an author-invented one. Both ceilings
#       are now required only on the GENERATION path. THROUGHPUT.json records
#       `ceiling_gpu_hours_supplied_at_measurement_time: null` when none was supplied, so the
#       record says the number was absent rather than implying one was measured against.
#
#   (d) 120x160. PR-08 §3: the converted corpus `datasets/gr00t-apple-full/` is 120x160 and a
#       photoreal restyle at that size is worthless to a VLA training at 640x480 (the GR00T N1.7
#       ego_view contract). This path re-derives from the HF source at full resolution and does
#       NOT reuse that tree. Both facts are asserted below rather than trusted: the source
#       manifest must declare 640x480, and a SOURCE path under gr00t-apple-full is refused
#       outright.
#
#   (e) A BUDGET AUTHORISED THREE TIMES, **OR** A RULE THAT CANNOT BE FOLLOWED. Both failures have
#       now been in this file, in that order, and the second is why the ceiling has two names.
#
#       Until 2026-08-15 CEILING_GPU_H was compared against ONE style set's projection, so handing
#       the same number to the train, eval and identity submissions authorised it three times over
#       — and after T40_RULE_V2's arm-C frame-match the identity set is 4 020 clips, not 402, so
#       the under-count was nearer 4x than 3x. The repair made CEILING_GPU_H "ONE number for the
#       WHOLE partition" and gated it against the whole-partition projection.
#
#       That repair contradicted the registered rule it was implementing. T40_RULE_V2 §3 step 3
#       requires the whole-partition ceiling to be SPLIT "across the STYLE_SET invocations and
#       chunks, so the sum of the CEILING_GPU_H values actually passed is bounded by the
#       whole-partition figure", and configs/transfer25/styles.toml [consumer] item 3 says the
#       same. Under the whole-partition-only reading, an operator who follows the rule passes
#       shares (at the committed partition and a 1068.19 GPU-h projection: 427.28 / 213.64 /
#       427.28) and ALL THREE submissions fail the gate, because each share is smaller than the
#       whole-partition projection it was compared against. The experiment was unlaunchable by
#       anyone obeying the rule. V2 is registered and is not editable, so this file moved.
#
#       The fix is not to pick one of the two numbers — both are real and they answer different
#       questions — but to name both and check the relation V2 actually states:
#         * PARTITION_CEILING_GPU_H is V2 §3 step 2's figure, derived over all 25 style-instances.
#           The whole-partition PROJECTION is gated against it, which is the property the
#           2026-08-15 repair was defending, and it is pinned run-wide so three submissions cannot
#           quietly carry three different budgets.
#         * CEILING_GPU_H is V2 §3 step 3's share for THIS invocation. This style set's projection
#           is gated against it, MAX_PASSES is derived from it, and this style set's ledger
#           reservations may not exceed it.
#         * the shares are pinned per style set and their SUM is required to fit
#           PARTITION_CEILING_GPU_H — which is the sentence V2 §3 step 3 is, enforced instead of
#           trusted. A ceiling handed to three submissions is now three shares that add up, and
#           "one budget spent three times" is caught by the sum rather than by forbidding shares.
#       Neither has a default, for (c)'s reason.
#
#   (f) A THRESHOLD COINED IN A DEFAULT'S CLOTHES. GEOM_STEP_FRAMES had a silent default of 1
#       until 2026-08-15, and that default was a pre-registration written by this file.
#       PR-08 §6 opens "No threshold is coined. The ladder is WAM-Bench's own, the margin is
#       borrowed, and the two geometry numbers are derived from the corpus itself." GEOM_TOL is
#       the median per-step object-centroid displacement, so it scales ~LINEARLY with what a
#       "step" is taken to be — and §6 never defines one. A shell default of 1 therefore does not
#       pick a convention, it picks the G0b tolerance: gate at 1 frame and the tolerance is a
#       tenth of what it is at 10, and nothing anywhere would say a threshold had been chosen.
#       That is precisely what CEILING_GPU_H is required-with-no-default to avoid, and the two
#       are the same defect wearing different clothes. So GEOM_STEP_FRAMES is now required
#       explicitly on the GENERATION path, with no default, and the artifact's own step_frames
#       must still equal it. The step is a REGISTERED quantity and not an operator preference:
#       the V3 being written registers it against AppleToPlate's actual action rate, and until it
#       does, refusing is the correct behaviour rather than an inconvenience.
#
# THE SHAPE IS BORROWED, NOT INVENTED. 4 h MaxWall and MaxJobsPU=4 make a long generation a CHAIN
# of jobs, so this file is 96_generate_apple_variations.sbatch's idempotent harvest/missing-list
# plus 94_train_t041_cosmos_super.sbatch's persisted pass rail, USR1 trap, four exit branches and
# DONE re-entry, plus scripts/run_i8_rung_local.sh's stamp refusal. Deliberately NOT a job array
# (cluster/discoverer/README.md, "Step 10 ... is deliberately NOT a job array"): an array task's
# interaction with `scontrol requeue` / SLURM_RESTART_COUNT is the open question the pass rail
# exists to survive. CHUNK_TOTAL plain submissions each get their own DONE, their own pass
# counter and their own blast radius.
#
#SBATCH --job-name=wam-t040-restyle
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
# --qos is not optional and not a preference. Omitted, the job lands on the `normal` association:
# MaxWall 00:01:00, gres/gpu=0, 1 job/user. It does not fall back permissively, it falls into a
# trap (README.md rule 4, docs/discoverer.md §QoS).
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 1 GPU x 26 threads x 192G. 26 threads and 257 GB per GPU is the billing-neutral ceiling
# (330900 billing-h / 5000 GPU-h = 66.18 = 26*0.035714 + 257*0.25 + 1.0); above it the billing
# counter empties before the GPU-hours and the remainder of the allocation is permanently lost.
# One GPU is also the biggest throughput lever on a 15-GPU machine whose queue is full of 4/7/8-GPU
# jobs, and four 1-GPU chunks sit inside both MaxJobsPU=4 and gres/gpu=16.
#
# For a larger allocation, override at submit time AND set NPROC to match — the runtime check
# below refuses a mismatch and refuses anything over the per-GPU ceiling:
#   NPROC=8 sbatch --gres=gpu:8 --cpus-per-task=208 --mem=1800G 97_transfer25_restyle.sbatch
# Note that only dgx1 has gpu:8 (dgx2 advertises gpu:7 plus a separate gpu_biz:1), so an 8-GPU
# request queues behind every other 4/7/8-GPU request on the machine.
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t040-restyle.%j.out

set -euo pipefail

# ------------------------------------------------------------------------------------------------
# THE USR1 TRAP, INSTALLED FIRST — BEFORE ANYTHING ELSE CAN RUN.
#
# `#SBATCH --signal=B:USR1@300` sends SIGUSR1 to THIS shell five minutes before the wall, and the
# default disposition of SIGUSR1 is TERMINATE. Until 2026-08-15 the trap was installed at the
# bottom of the file, after the driver had been backgrounded and only inside the `[[ -s MISSING ]]`
# branch — so every one of: the work-list expansion, the stamp check, the partition verification,
# the G0b geometry gate, the ceiling gate, the pass-rail write and the FIRST HARVEST ran with
# SIGUSR1 lethal. A signal in that window killed the batch shell outright, `scontrol requeue` at
# the bottom was never reached, and the chain that the whole file is shaped around stopped one pass
# in — the exact dead end the trap exists to prevent, arriving through the trap's own absence.
#
# So it is registered here, on the second line, and it is written to be safe with no driver
# running: CHILD is empty until one is backgrounded, and the handler records the signal either way.
# GOT_USR1 is read later by the pre-driver requeue check and by the four exit branches.
# ------------------------------------------------------------------------------------------------
GOT_USR1=0
CHILD=""
usr1_handler() {
  GOT_USR1=1
  echo "=== USR1 at $(date -Is): 5 min to the wall."
  if [[ -n "${CHILD}" ]] && kill -0 "${CHILD}" 2>/dev/null; then
    echo "    terminating driver pid ${CHILD} so this pass requeues instead of being killed."
    # There is no documented graceful "stop at a unit boundary" to ask the driver for, so SIGTERM
    # is the honest option: it costs at most the one work unit in flight — every finished unit is
    # already on disk and the harvest will file it — and it buys an orderly requeue instead of a
    # SIGKILL that ends the chain.
    kill -TERM "${CHILD}" 2>/dev/null || true
  else
    echo "    no driver is running yet. Recorded, and this shell SURVIVES: the setup phase is not"
    echo "    worth a kill, and staying alive is what lets the requeue below be reached at all."
  fi
}
trap usr1_handler USR1

# -- the token ------------------------------------------------------------------------------------
# THIS JOB DOWNLOADS AT RUNTIME WHETHER OR NOT WE WANT IT TO, AND UNTIL 2026-08-22 IT DID SO
# ANONYMOUSLY. 99 and 99b both read a token; this file had no reference to one at all. That looked
# harmless while the only weights in play were the pre-staged Transfer2.5 branches. It is not:
# Control2WorldInference instantiates the BASE model's tokenizer, which lives in
# `nvidia/Cosmos-Predict2.5-2B` -- a DIFFERENT repo from Transfer2.5, not staged by 99 or 99b, and
# `gated: auto`. Job 189402 died there on a bare `uvx hf download ... tokenizer.pth`.
#
# Measured 2026-08-22 rather than assumed: with a VALID token that repo returns 403 on
# tokenizer.pth at the framework's pinned f176dc95, at the current 85f8ae7b and at main, while
# nvidia/Cosmos-Transfer2.5-2B (accepted by the account holder 2026-08-19) returns 206 for the same
# token. So the licence is the blocker and the token is the OTHER half: once accepted, an anonymous
# fetch is still a 401. Both are needed. Never echoed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -f "${f}" ]] && { HF_TOKEN=$(<"${f}"); break; }
  done
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  echo "=== hf token  present"
else
  echo "=== hf token  ABSENT — the tokenizer fetch below is gated and will 401. Write a token to"
  echo "===           \${HOME}/.huggingface/token or export HF_TOKEN for this submission."
fi

source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

# ------------------------------------------------------------------------------------------------
# cosmos_transfer_env.sh — written by 98, appended to by 99. (Was a TODO(human) until 2026-08-20:
# when this file was written nothing in the repo built or staged Transfer2.5, and `grep -ril
# transfer2` over the tree returned only prose.)
#
# 90_build_cosmos_env.sbatch writes ${PROJ}/cosmos_env.sh for the Cosmos3 GENERATOR family
# (FRAMEWORK, FFMPEG_PREFIX, MODEL_ID, BASE_CHECKPOINT_PATH, ...). Transfer2.5 is a different
# checkout with a different entrypoint, so it gets its own file and its own two jobs:
#
#     98_build_transfer25_env.sbatch   writes it: TRANSFER_FRAMEWORK, FRAMEWORK, FFMPEG_PREFIX,
#                                      the UV_* triple, and the pinned framework sha
#     99_stage_transfer25_weights.sbatch  appends TRANSFER_MODEL_ID, TRANSFER_MODEL_REVISION
#                                      and TRANSFER_CHECKPOINT_PATH
#
# DO NOT source this together with cosmos_env.sh — both export FRAMEWORK, pointing at different
# checkouts, and the second one sourced silently wins. 98's own header says the same thing.
#
# It is still sourced only `if present`, and the `: ${..:?}` guards below still fire by name. That
# is not vestigial now that the producers exist: 99 can succeed at staging weights and still leave
# the file half-written if it is killed between the two appends, and the guards are what turn that
# into a message naming the missing job rather than a `set -u` death 200 lines down, after Slurm
# has already handed the job a GPU.
# ------------------------------------------------------------------------------------------------
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}
if [[ -f "${TRANSFER_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${TRANSFER_ENV}"
fi

# -- TIMING_MODE, resolved before any requirement is decided -------------------------------------
# See (a) and (c) in the header. PR-08 §1 licenses "timing one episode on an H200" unconditionally
# and conditions only "generating a corpus" on T-39 having reported; §8 item 3 says the budget line
# does not exist until that measurement does. So BOTH the attestation and the two GPU-h ceilings
# are decided by this flag, and it has to be resolved before the first `: ${..:?}` fires — the
# TIMING branch itself is 300 lines down and by then the job would already have refused.
#
# `set -u` is on, so this reads TIMING defensively; the assignment is a plain command (not a
# compound test) because `[[ ... ]] && X=1` as the LAST statement of a script would exit non-zero
# under `set -e`. It is not last here, but the pattern is one line from being a trap.
TIMING_MODE=0
if [[ "${TIMING:-0}" != "0" ]]; then TIMING_MODE=1; fi

# -- required parameters, no defaults ------------------------------------------------------------
# Each of these is a runtime argument: none of them is inside any checkpoint config or config_hash,
# so nothing downstream could ever detect a wrong value. They are required here and stamped into
# the run directory below.
: "${STYLE_SET:?which half of the committed partition to generate: train | eval | identity. PR-08 §5 splits TRAIN_STYLES from EVAL_STYLES before the first clip so the eval domain cannot be chosen after seeing which restyles came out well; identity is arm C, the generator-fingerprint control.}"
: "${CHUNK_INDEX:?1-based chunk number. Chunking is not optional under 4 h MaxWall (PR-08 §8 item 3).}"
: "${CHUNK_TOTAL:?how many chunks this run is split into. At most 4 may be in flight (MaxJobsPU=4).}"
: "${CONTROL:?which Transfer2.5 control blocks condition the restyle and at what weight, e.g. depth:0.5,seg:0.5. Required on BOTH paths and never defaulted, for the reason restyle_transfer25.py refuses to default it: the choice decides how much geometry survives, which is exactly what PR-08 §6 G0b measures, so picking it after looking at clips is the failure the committed style partition exists to prevent. It also decides the TIMING number — our manifest carries no depth or segmentation maps, so each control block Transfer2.5 has to ESTIMATE is GPU time the measurement must include.}"
: "${STAGE:?which stage of the committed partition to generate: 1 | 2 | all. T40_RULE_V11, adopted 2026-08-24. Stage 1 is the first four TRAIN_STYLES in committed order plus their four matched identity repeats — 8 of 25 style-instances, ~442 GPU-h instead of ~1380 — and the eval set is DEFERRED, so STAGE=1 with STYLE_SET=eval is refused rather than quietly generated. Stage 2 is the complement, and V11 §3 fixes IN ADVANCE which stage-1 outcome licenses it. `all` is the unstaged committed partition and is what a run that is not staging should pass. There is deliberately NO default: defaulting to `all` would spend the full partition for a missing variable, and defaulting to `1` would silently truncate a run that meant the whole thing.}"

case "${STYLE_SET}" in
  train|eval|identity) ;;
  *) echo "FATAL: STYLE_SET=${STYLE_SET} is not one of train|eval|identity."; exit 1 ;;
esac
# THE STAGE IS NOT A STYLE COUNT AND IS NOT OPERATOR-SIZED. T40_RULE_V11 §2 fixes k = 4 as the
# stage-1 prefix length, in committed order, and §2.4 is explicit that the selector is the owed
# change: the alternative — editing configs/transfer25/styles.toml — is forbidden by V11 §0 because
# the partition hash is the provenance of every clip. So the prefix length lives here as a constant
# citing the rule, and no environment variable can move it. An operator who wants a different k is
# asking for a different rule, not a different submission.
case "${STAGE}" in
  1|2|all) ;;
  *) echo "FATAL: STAGE=${STAGE} is not one of 1|2|all (T40_RULE_V11 §2)."; exit 1 ;;
esac
STAGE1_TRAIN_STYLES=4          # T40_RULE_V11 §2. Not a knob.
if [[ "${STAGE}" == "1" && "${STYLE_SET}" == "eval" ]]; then
  echo "FATAL: STAGE=1 with STYLE_SET=eval. T40_RULE_V11 §2 DEFERS the eval set out of stage 1;"
  echo "       it is deferred, not cut, and it is generated in stage 2. Generating it now would"
  echo "       spend the eval budget before the stage-1 result that decides whether stage 2"
  echo "       happens at all, which is the whole point of staging."
  exit 1
fi
[[ "${CHUNK_TOTAL}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: CHUNK_TOTAL=${CHUNK_TOTAL} is not a positive integer."; exit 1; }
[[ "${CHUNK_INDEX}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: CHUNK_INDEX=${CHUNK_INDEX} is not a positive integer."; exit 1; }
(( CHUNK_INDEX <= CHUNK_TOTAL )) || { echo "FATAL: CHUNK_INDEX=${CHUNK_INDEX} > CHUNK_TOTAL=${CHUNK_TOTAL}."; exit 1; }
# Shape only. The KEYS are validated against upstream-s set by parse_controls() in the driver,
# which is where that knowledge lives; this catches an empty or malformed spec here rather than
# after Slurm has handed the job a GPU.
[[ "${CONTROL}" =~ ^[a-z]+:[0-9]*[.]?[0-9]+(,[a-z]+:[0-9]*[.]?[0-9]+)*$ ]] || {
  echo "FATAL: CONTROL=${CONTROL} is not a comma-separated list of key:weight pairs,"
  echo "       e.g. depth:0.5,seg:0.5. The keys are checked by the driver against the"
  echo "       upstream control set; this only checks the shape."
  exit 1; }

# -- the two GPU-h ceilings: required on GENERATION, EXEMPT on TIMING -----------------------------
# See (c) and (e) in the header. TIMING=1 is the measurement the ceilings are DERIVED FROM, so
# demanding them here made the licensed measurement impossible to perform honestly. On the timing
# path both are left empty and recorded as null; on the generation path both are required, both
# with no default, and their relation (share <= whole, and the shares SUM within the whole) is
# checked at the gate below.
if (( TIMING_MODE )); then
  echo "=== TIMING=1: no GPU-h ceiling is asked for. PR-08 §8 item 3 — no budget line exists until"
  echo "    this measurement does, so the measurement cannot be gated on the budget it produces."
  if [[ -n "${CEILING_GPU_H:-}" || -n "${PARTITION_CEILING_GPU_H:-}" ]]; then
    echo "    NOTE: a ceiling was supplied anyway. It is IGNORED and recorded as supplied-but-"
    echo "          unused, so the derivation cannot later be read as having been checked."
  fi
  CEILING_GPU_H=""
  PARTITION_CEILING_GPU_H=""
else
  : "${PARTITION_CEILING_GPU_H:?T40_RULE_V2 §3 step 2: the GPU-h ceiling derived from the MEASURED throughput over the WHOLE partition — all 25 style-instances, 10050 clips, train + eval + identity. There is deliberately NO default: a default would be a budget line nobody measured. Run TIMING=1 first, read THROUGHPUT.json, derive it, record the derivation, then pass PARTITION_CEILING_GPU_H=<n>. It is the SAME number in all three STYLE_SET submissions of a RUN_ID.}"
  : "${CEILING_GPU_H:?T40_RULE_V2 §3 step 3: THIS invocation-s SHARE of PARTITION_CEILING_GPU_H, because the rule splits the whole-partition ceiling across the STYLE_SET invocations and requires the shares to SUM within it. The three sets are not the same size, so the shares are not equal: at the committed partition they are 10/25, 5/25 and 10/25 of the whole. NO default. Passing the whole-partition figure here as well is refused unless it really is this set-s share.}"
  [[ "${PARTITION_CEILING_GPU_H}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "FATAL: PARTITION_CEILING_GPU_H=${PARTITION_CEILING_GPU_H} is not a number."; exit 1; }
  [[ "${CEILING_GPU_H}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "FATAL: CEILING_GPU_H=${CEILING_GPU_H} is not a number."; exit 1; }
  # awk, not python: `module load anaconda3/python3.12` is 40 lines below this point and the
  # module-free login shell python is not something this file may assume.
  if awk -v a="${CEILING_GPU_H}" -v b="${PARTITION_CEILING_GPU_H}" 'BEGIN{exit !(a>b)}'; then
    echo "FATAL: CEILING_GPU_H=${CEILING_GPU_H} (this style set's share) exceeds"
    echo "       PARTITION_CEILING_GPU_H=${PARTITION_CEILING_GPU_H} (the whole partition)."
    echo "       T40_RULE_V2 §3 step 3: the shares are a SPLIT of the whole-partition figure, so"
    echo "       no single share can be larger than it. Re-derive the split."
    exit 1
  fi
fi

# -- the T-39 attestation: required on the GENERATION path, exempt on the TIMING path ------------
if (( TIMING_MODE )); then
  echo "=== TIMING=1: PR-08 §1 licenses this measurement WITHOUT T-39, so no attestation is asked"
  echo "    for and none is recorded. This path may not write a clip into ${CLIPS:-clips/}: it"
  echo "    generates into a throwaway root and deletes it. Generation refuses without T-39."
  PR08_T39_REPORTED="N/A — TIMING=1, exempt under PR-08 §1 (licensed: timing one episode on an H200)"
  PR08_T39_ARTIFACT=""
  PR08_T39_ARTIFACT_SHA256=""
  PR08_T39_VOID_OVERRIDDEN=""
else
  # NOTE for anyone editing these two messages: the word of a ${VAR:?word} expansion is parsed by
  # bash, and an apostrophe or an escaped quote inside one ends the surrounding double quote and
  # breaks the whole file at parse time — `bash -n` catches it 300 lines later, at the next
  # heredoc, with a message that names neither. Keep them free of ' and ".
  : "${PR08_T39_REPORTED:?PR-08 §1 gates every generated clip on T-39 having REPORTED. Set PR08_T39_REPORTED to the verdict and its date, e.g. P (oracle_action clears L1) 2026-08-14; it is written verbatim into the record. Accepted verdicts are PR-07 §5-s P, N, M and I. VOID IS REFUSED HERE, and is not an example to copy: PR-07 §4 defines a VOID as the finding that no policy trained on this dataset can clear our bar and that the defect is our own label pipeline, which is the bet PR-08 §1 gates against losing. (TIMING=1 does NOT need this — §1 licenses the timing measurement outright.)}"
  : "${PR08_T39_ARTIFACT:?PR-08 §1 gates every generated clip on T-39 having REPORTED, and a prose string is unfalsifiable. Set PR08_T39_ARTIFACT to the PATH of the committed T-39 result artifact (e.g. the eval-t39-oracle-action bench.json that 71_eval_t39_control.sbatch rules from). It must exist and be non-empty; its sha256 is recorded beside the prose. (TIMING=1 does NOT need this.)}"
  # A verdict token and a date, so "yes it reported" cannot be the attestation. The tokens are
  # PR-07 §5's, not this file's: P, N, M, I, and the VOID branches 71_eval_t39_control.sbatch can
  # print. This checks the SHAPE of the claim, never its truth — only the artifact can carry that,
  # and only a human can read it. VOID stays in THIS pattern so that a VOID attestation reaches
  # the refusal below and gets its own diagnosis, instead of falling out here as "no verdict
  # token" and sending the operator off to find a different word for the same result.
  if ! [[ "${PR08_T39_REPORTED}" =~ (^|[^A-Za-z])(P|N|M|I|VOID)([^A-Za-z]|$) ]]; then
    echo "FATAL: PR08_T39_REPORTED=\"${PR08_T39_REPORTED}\" carries no PR-07 §5 verdict token."
    echo "       One of P | N | M | I must appear (VOID is a verdict too, and it is REFUSED below"
    echo "       rather than accepted). PR-08 §1 gates on T-39 having REPORTED, and a report is a"
    echo "       verdict, not a status update."
    exit 1
  fi
  # ----------------------------------------------------------------------------------------------
  # VOID REFUSES. See (a) in the header for the argument; this is the enforcement.
  #
  # Until 2026-08-15 VOID was ACCEPTED by the pattern above and the :? message's worked example was
  # the string `VOID (labels) 2026-08-14` — so this gate satisfied PR-08 §1's letter and inverted
  # its reason, while teaching the operator to type the forbidden verdict. PR-07 §4: if the
  # ground-truth action column cannot clear L1 under our scorer, "no policy trained on this dataset
  # can clear our bar, and the finding is about our label pipeline". PR-07 §6's VOID row licenses a
  # defect report and forbids any statement about GR00T. PR-08 §1 gates generation on T-39 because
  # "generating data is a bet on the first" — and a VOID is that bet, lost, in writing.
  #
  # The override is a separate variable and a whole sentence on purpose. A boolean named
  # FORCE=1 is something an operator types to make a message stop; a sentence that says what is
  # being overridden is a claim they have to have read. It is also recorded verbatim in
  # chunk_metadata.json, so no clip generated under it can lose the fact.
  # ----------------------------------------------------------------------------------------------
  T39_VOID_OVERRIDE_ACK="I am generating under a T-39 VOID, which says this corpus cannot clear the bar"
  PR08_T39_VOID_OVERRIDDEN=""
  if [[ "${PR08_T39_REPORTED}" =~ (^|[^A-Za-z])VOID([^A-Za-z]|$) ]]; then
    if [[ "${PR08_OVERRIDE_T39_VOID:-}" != "${T39_VOID_OVERRIDE_ACK}" ]]; then
      echo "FATAL: PR08_T39_REPORTED=\"${PR08_T39_REPORTED}\" is a T-39 VOID, and VOID is the one"
      echo "       verdict that STOPS PR-08 rather than unlocking it."
      echo
      echo "       PR-07 §4: if the ground-truth action column itself cannot clear L1 under our"
      echo "       scorer, then \"no policy trained on this dataset can clear our bar, and the"
      echo "       finding is about our label pipeline, not about GR00T.\" PR-07 §6's VOID row"
      echo "       licenses exactly one thing — a defect report against the adapter or the label"
      echo "       pipeline — and forbids any statement about GR00T."
      echo
      echo "       PR-08 §1 gates generation on T-39 for that exact reason: until T-39 reports"
      echo "       whether ANY method clears the bar on this corpus, \"the data is wrong\" and"
      echo "       \"the method is wrong\" are not separable, and generating data is a bet on the"
      echo "       first. A VOID does not settle that bet — it LOSES it. Restyling 10050 clips"
      echo "       whose action labels are copied unchanged from the pipeline the VOID indicts"
      echo "       would spend the allocation on the one hypothesis T-39 just ruled out."
      echo
      echo "       P, N, M and I are accepted here. VOID is not. If a human holds evidence this"
      echo "       script cannot see, the override is deliberately a sentence and not a flag:"
      echo
      echo "         PR08_OVERRIDE_T39_VOID=\"${T39_VOID_OVERRIDE_ACK}\""
      echo
      echo "       Setting it is a claim, it is recorded verbatim in chunk_metadata.json beside"
      echo "       the verdict, and it belongs in a V3 before it belongs on a submit line."
      exit 1
    fi
    PR08_T39_VOID_OVERRIDDEN="${PR08_OVERRIDE_T39_VOID}"
    echo "=== !! GENERATING UNDER A T-39 VOID. PR08_OVERRIDE_T39_VOID was set to the exact"
    echo "===    acknowledgement, so this run proceeds against PR-07 §4's finding that no policy"
    echo "===    trained on this dataset can clear our bar. Recorded verbatim in the record."
  fi
  if ! [[ "${PR08_T39_REPORTED}" =~ [0-9]{4}-[0-9]{2}-[0-9]{2} ]]; then
    echo "FATAL: PR08_T39_REPORTED=\"${PR08_T39_REPORTED}\" carries no ISO date (YYYY-MM-DD)."
    echo "       The date is what makes the attestation checkable against the run it cites."
    exit 1
  fi
  if [[ ! -s "${PR08_T39_ARTIFACT}" ]]; then
    echo "FATAL: PR08_T39_ARTIFACT=${PR08_T39_ARTIFACT} does not exist or is empty."
    echo "       This is the machine-checkable half of the T-39 gate: the prose says what the"
    echo "       verdict was, the artifact is the thing that either exists or does not. Point it"
    echo "       at T-39's committed result (71_eval_t39_control.sbatch reads"
    echo "       <run>/eval-t39-oracle-action/bench.json and the three sibling arms)."
    exit 1
  fi
  PR08_T39_ARTIFACT_SHA256=$(sha256sum "${PR08_T39_ARTIFACT}" | cut -d' ' -f1)
  echo "=== T-039 artifact: ${PR08_T39_ARTIFACT}"
  echo "===   sha256 ${PR08_T39_ARTIFACT_SHA256}"
fi

: "${FRAMEWORK:?no Transfer2.5 checkout or venv — ${TRANSFER_ENV} is missing or does not export FRAMEWORK. See the TODO above: the build/stage job for Transfer2.5 is not written.}"
: "${FFMPEG_PREFIX:?no ffmpeg — ${TRANSFER_ENV} is missing or does not export FFMPEG_PREFIX.}"
: "${TRANSFER_CHECKPOINT_PATH:?no staged Transfer2.5 checkpoint — ${TRANSFER_ENV} must export TRANSFER_CHECKPOINT_PATH. PR-08 §6 records the generator checkpoint id AND revision regardless of verdict, so this must be a pinned revision, not a floating tag.}"
: "${TRANSFER_MODEL_ID:?${TRANSFER_ENV} must export TRANSFER_MODEL_ID (e.g. the HF repo id of the frozen Transfer2.5 variant). PR-08 §6 requires it in the record.}"
: "${TRANSFER_MODEL_REVISION:?${TRANSFER_ENV} must export TRANSFER_MODEL_REVISION — the pinned commit, not a tag.}"

echo "=== T-039 reported: ${PR08_T39_REPORTED}"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8
export PATH=${FFMPEG_PREFIX}/bin:${PATH}
# Both from the Cosmos launch scripts and both load-bearing: the host CUDA libs have to be cleared
# and the expandable allocator is what makes the model fit at all.
export LD_LIBRARY_PATH=""
export PYTORCH_ALLOC_CONF=expandable_segments:True
# 26 threads per GPU is the hard cap on this allocation; exceeding it strands GPU-hours.
export OMP_NUM_THREADS=26
nvidia-smi -L || echo "=== nvidia-smi -L failed — the allocation check below will refuse this job"

# -- the billing ceiling, checked against what Slurm actually allocated ---------------------------
# Slurm bills ALLOCATED, not used, resources, and the #SBATCH block above can be overridden at
# submit time. A wrong override does not fail — it silently burns the billing counter faster than
# the GPU counter, and when billing runs out the remaining GPU-hours are permanently lost. So the
# ratio is asserted here, on the real allocation, before a single GPU-second is spent.
# `grep -c` prints 0 AND exits 1 when it matches nothing, so `|| echo 0` would make this "0\n0" and
# break every (( )) below. `|| true` keeps the single 0 that grep already printed.
GPUS_ALLOC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}
CPUS_ALLOC=${SLURM_CPUS_PER_TASK:-1}
MEM_MB_ALLOC=${SLURM_MEM_PER_NODE:-0}
NPROC=${NPROC:-${GPUS_ALLOC}}
if (( GPUS_ALLOC < 1 )); then
  echo "FATAL: no GPUs allocated. A GPU job under --qos=2cpu-single-host is REJECTED and a"
  echo "       missing --qos lands on \`normal\` (gres/gpu=0). Check the submit line."
  exit 1
fi
if (( NPROC != GPUS_ALLOC )); then
  echo "FATAL: NPROC=${NPROC} but Slurm allocated ${GPUS_ALLOC} GPUs. Idle allocated GPUs are"
  echo "       billed at full rate; over-subscribed ones fail. Set NPROC to match --gres."
  exit 1
fi
if (( CPUS_ALLOC > GPUS_ALLOC * 26 )); then
  echo "FATAL: ${CPUS_ALLOC} threads for ${GPUS_ALLOC} GPU(s) exceeds the 26-per-GPU billing"
  echo "       ceiling. Above it the billing counter empties before the GPU-hours and the rest"
  echo "       of the 5000 GPU-h allocation is PERMANENTLY LOST (README.md rule 3)."
  exit 1
fi
if (( MEM_MB_ALLOC > GPUS_ALLOC * 257 * 1024 )); then
  echo "FATAL: ${MEM_MB_ALLOC} MB for ${GPUS_ALLOC} GPU(s) exceeds the 257 GB-per-GPU billing"
  echo "       ceiling. Same consequence as the thread ceiling above."
  exit 1
fi
echo "=== allocation ${GPUS_ALLOC} GPU / ${CPUS_ALLOC} threads / ${MEM_MB_ALLOC} MB — within the ceiling"

# -- paths ---------------------------------------------------------------------------------------
WAM=${PROJ}/wam
RUN_ID=${RUN_ID:-t040-transfer25-restyle}
OUT=${PROJ}/runs/${RUN_ID}
# Job-id-independent, so --open-mode=append accumulates one log per chunk across every pass.
# The STAGE is in the tag, not only in the stamp below. Without it stage 1 and stage 2 of the same
# style set and chunk number resolve to ONE directory, and the stamp check would refuse the second
# with a WORK_SHA mismatch — a true diagnostic pointing at the wrong cause. With it they are simply
# different runs. (CLIPS is deliberately NOT staged: the stages generate disjoint styles and
# disjoint identity repeats, so their unit ids cannot collide, and the corpus a consumer reads is
# the union of the stages rather than one of them.)
CHUNK_TAG=$(printf 's%s-%s-%02dof%02d' "${STAGE}" "${STYLE_SET}" "${CHUNK_INDEX}" "${CHUNK_TOTAL}")
CHUNK_DIR=${OUT}/chunks/${CHUNK_TAG}
CLIPS=${OUT}/clips/${STYLE_SET}
RAW=${CHUNK_DIR}/_raw
PAYLOADS=${CHUNK_DIR}/payloads
MISSING=${CHUNK_DIR}/missing
# The units G0c refused on their SOURCE masks, which are DECIDED rather than outstanding. Kept in
# its own file beside the missing list and never merged into it: the missing list is what the chunk
# requeues on, and a permanent refusal on it is an unbounded loop. See permanently_refused() in the
# harvest for the whole argument, and for why this cannot let a unit pass.
REFUSED=${CHUNK_DIR}/refused.json
PASSES_LOG=${CHUNK_DIR}/passes.jsonl
PREREG=${WAM}/docs/preregistration/PR-08-photoreal-augmentation.md

# ------------------------------------------------------------------------------------------------
# TODO(human) — THE PR-08 §8 PRECONDITION THAT DOES NOT EXIST IN THIS REPO YET.
# It is asserted, not assumed, because it is a way for this job to produce clips that look fine
# and are not evidence:
#
#   SOURCE            §8 item 3 / §3. The 402 real demos re-derived from the HF source at 640x480.
#                     Nothing writes it today — workstation/10_fetch_corpus.sh and
#                     scripts/convert_lerobot_g1.py produce the 120x160 tree PR-08 §3 forbids.
#                     Contract assumed here: ${SOURCE}/manifest.json with
#                       {"resolution": [640, 480],
#                        "episodes": [{"id": str, "frames": int, "video": relpath,
#                                      "depth": relpath, "segmentation": relpath}, ...]}
#                     depth/ and segmentation/ are the ESTIMATED conditioning of PR-08 §4 and are
#                     blocked on §8 item 5 (the isaac_binding.py annotators) via the error budget.
# ------------------------------------------------------------------------------------------------
# STYLE_PARTITION and GEOM_CONSTANTS DO exist — they are the outputs of the sibling PR-08 §8
# workstreams, and the defaults below point at the real files rather than at names this file
# invented:
#   configs/transfer25/pr08_style_partition.json   rule T40_STYLES_V1, committed 2026-08-15, with
#                                                  train (10) / eval (5) / identity (1) sets.
#   configs/transfer25/pr08_geom_tol.json          scripts/measure_geom_tol.py's default --out, with
#                                                  a .sha256 sidecar. TRACKED, on purpose: §8 item 4
#                                                  wants GEOM_TOL *committed* before generation, and
#                                                  the old default (runs/pr08/geom_tol.json) sat
#                                                  under a gitignored directory, so it could never
#                                                  have been the pre-commitment the rule asks for.
# Both are read, never written, here.
#
# GEOM_STEP_FRAMES is the step the gate is evaluated under. GEOM_TOL scales ~linearly with it and
# PR-08 §6 does not define "step", so the artifact records which reading produced the number and
# this job asserts it matches rather than inheriting whatever the measurement happened to use.
# IT HAS NO DEFAULT AND IS NOT SET HERE — see (f) in the header. It is required in the GENERATE
# section, beside GEOM_CONSTANTS and for the same reason: PR-08 §1 licenses the timing measurement
# independently of the geometry work, so requiring it on this path would block the one measurement
# the document explicitly permits.
# ------------------------------------------------------------------------------------------------
# THE H.264 TREE, NOT THE AV1 ONE, FOR TWO INDEPENDENT REASONS — either alone decides it.
# (1) DECODABILITY. The corpus ships AV1 and the cv2 4.11.0 in 98's generation venv is built
#     against avcodec 59.37.100, which has no AV1 decoder and DOES NOT RAISE about it:
#     VideoCapture opens the container, believes the header, reports the frame count and fails
#     every read. Three jobs have died on that shape (186357 wrote 0 captions; 189585 died at 7 s;
#     189584 watched upstream's own SAM 2 helper extract 0 of 590 frames). Job 189613 verified all
#     402 clips of the H.264 tree decode under the generation venv's own cv2.
# (2) THE AREA BOUND IS A CLAIM ABOUT A CORPUS. 106 measures the robot-mask area distribution on
#     the H.264 tree, and robot_composite.load_area_bound compares the bound's
#     source_manifest_sha256 against THIS run's SOURCE manifest. The two trees have different
#     manifests, so pointing this at the AV1 tree makes the composite refuse at G0c — correctly,
#     and after Slurm has handed out a GPU.
# The transcode is proven BIT-EXACT per clip (scripts/transcode_corpus_lossless.py,
# TRANSCODE_PROOF.json: frame count and max abs channel delta 0 in yuv420p AND rgb24 on every
# clip), so this is the same pixels by a decodable route and not a different corpus.
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480-h264-lossless}
STYLE_PARTITION=${STYLE_PARTITION:-${WAM}/configs/transfer25/pr08_style_partition.json}
# The SOURCE of the rendering above, and the verifier that relates the two. Both are read (never
# written) by the pre-commitment check below — see FIX 3's block. styles.toml is the file PR-08 §6's
# "partition hash" is a hash OF; pr08_style_partition.json is its generated rendering, and the only
# reason it is safe to have both in git is that check_style_partition.py refuses when they disagree.
STYLE_SOURCE=${STYLE_SOURCE:-${WAM}/configs/transfer25/styles.toml}
STYLE_CHECKER=${STYLE_CHECKER:-${WAM}/scripts/check_style_partition.py}
GEOM_CONSTANTS=${GEOM_CONSTANTS:-${WAM}/configs/transfer25/pr08_geom_tol.json}
# THE DRIVER. This block used to read "TODO(human): there is no Transfer2.5 invocation anywhere in
# this repo to copy, and guessing the module path, the payload schema and the conditioning flags
# would be worse than saying so." That refusal was right, and it has now been discharged by READING
# THE SOURCE rather than by guessing: docs/transfer25-api.md, every claim cited to a file and line
# in nvidia-cosmos/cosmos-transfer2.5 @ main, 2026-08-16. Read that before editing this block.
# The contract this file needs is deliberately narrow, and is UNCHANGED:
#     python ${RESTYLE_DRIVER} --checkpoint-path <ckpt> --no-guardrails \
#            --manifest <SOURCE/manifest.json> --styles <STYLE_PARTITION> --style-set <set> \
#            --work-list <missing-list> --out <raw-dir> --resolution 640x480 --nproc <n>
# and per finished work unit it must write <raw>/<unit>/vision.mp4 plus a
# <raw>/<unit>/sample_outputs.json carrying {"status": "success"|"error"} AFTER the mp4 is
# asserted to exist.
#
# THE SPELLINGS ABOVE ARE OURS AND DO NOT PASS THROUGH. The driver translates at its boundary and
# nothing here may be handed to upstream's examples/inference.py directly (docs/transfer25-api.md §7):
#   --no-guardrails    upstream is --disable-guardrails, a SetupArguments field, CLI-only.
#   --resolution 640x480   upstream `resolution` is a BUCKET KEY, not WxH: "480" + a 4:3 input
#                      resolves to (640, 480) via VIDEO_RES_SIZE_INFO. The driver asserts the
#                      bucket it lands in rather than passing our string through.
#
# CORRECTION, 2026-08-16 — the reason this file used to give for the status file was wrong.
# It said "96's harvest reads the status rather than counting files precisely because the
# framework's _handle_sample_exception writes that file too". THERE IS NO SUCH FUNCTION anywhere in
# cosmos-transfer2.5. The INSTRUCTION stands and is now better supported, by two verified facts:
#   (a) upstream writes its <name>.json args sidecar BEFORE generation and before the guardrail
#       check (inference.py:205-208), so a sidecar proves the job was ATTEMPTED, not that a video
#       exists — a harvest keyed on file presence counts failures as finished work; and
#   (b) generate() loops with NO try/except (inference.py:158-183) and `keep_going` is consulted
#       only for guardrail blocks, so ONE unreadable video kills the whole batch. Per-unit fault
#       isolation is the driver's job, not the framework's, and on a 10 050-clip run that is the
#       difference between losing a clip and losing a chunk.
#
# THE WORK LIST IS THE SEED CHANNEL, AND THE DRIVER MUST HONOUR IT. Each JSONL row is
#     {"unit": "<episode>__<style>__r<NN>", "episode": ..., "frames": int,
#      "style": "<style id>", "repeat": int, "seed": int}
# and `seed` is REQUIRED, not advisory: arm C is ten samples of ONE prompt, so if the driver
# ignores the field the ten identity clips of an episode are either ten identical files or ten
# irreproducible ones, and either way the generator-fingerprint control has measured nothing. A
# driver that does not set its sampler from row["seed"] must fail loudly rather than default.
RESTYLE_DRIVER=${RESTYLE_DRIVER:-${WAM}/scripts/restyle_transfer25.py}

# PR-08 §3, as a refusal rather than a note, and BEFORE the existence loop below. The 120x160 tree
# is the obvious thing to point this at, it exists, it has a manifest.json, it would sail through
# every generic check — and every clip would be worthless to a VLA that trains at 640x480. Checked
# first so that pointing at it gets this diagnosis rather than a generic "missing file".
case "${SOURCE}" in
  *gr00t-apple-full*)
    echo "FATAL: SOURCE=${SOURCE} is the converted 120x160 corpus. PR-08 §3 explicitly forbids"
    echo "       reusing datasets/gr00t-apple-full/ — this path re-derives from the HF source at"
    echo "       full resolution. Restyling 120x160 produces frames a 640x480 VLA cannot use."
    exit 1 ;;
esac

# The AV1 tree, by name, for the same reason and in the same place: it exists, it has a
# manifest.json at the right resolution, and it would sail through every generic check. What it
# would NOT do is decode under the generation venv's cv2, and the failure is SILENT — the reader
# returns no frames rather than raising. Caught here so that pointing at it costs a second and a
# named diagnosis instead of a GPU allocation and a puzzle. See the SOURCE default above.
case "${SOURCE%/}" in
  *pr08-apple-640x480)
    echo "FATAL: SOURCE=${SOURCE} is the AV1 tree. The cv2 in 98's generation venv has no AV1"
    echo "       decoder and does not raise about it — it reports the frame count and reads"
    echo "       nothing (186357, 189585, 189584 all died on that shape)."
    echo "       Use the bit-exact H.264 transcode instead, which is also the tree 106 measures"
    echo "       the G0c area bound over, so it is the only one load_area_bound will accept:"
    echo "         SOURCE=${PROJ}/data/pr08-apple-640x480-h264-lossless"
    exit 1 ;;
esac

# GEOM_CONSTANTS is deliberately NOT in this list. PR-08 §1 licenses "timing one episode on an
# H200" independently of the estimator work, so requiring the geometry budget here would block the
# one measurement the document explicitly permits. It is required in the GENERATE section instead,
# which is where §6's "generation does not start" actually bites.
for f in "${PREREG}" "${STYLE_PARTITION}" "${STYLE_SOURCE}" "${STYLE_CHECKER}" \
         "${RESTYLE_DRIVER}" "${SOURCE}/manifest.json"; do
  [[ -f "${f}" ]] || {
    echo "FATAL: ${f} missing. PR-08 §8 lists what must exist before a single clip is generated;"
    echo "       this is one of them and it is not written yet. See the TODO(human) blocks in"
    echo "       this file. Nothing here can be worked around by generating anyway."
    exit 1; }
done

# ------------------------------------------------------------------------------------------------
# THE PRE-COMMITMENT IS VERIFIED HERE, BEFORE THE WORK LIST IS BUILT.
#
# Until 2026-08-15 this job byte-hashed whatever file it was handed, recorded the digest, and
# re-checked exactly one property of it: that the train and eval halves did not share an ID. That
# is not the pre-commitment. PR-08 §5's claim is that the eval domain is VISUALLY SHIFTED from the
# training domain — and visual shift lives in the four axis slugs (apple / table / background /
# lighting), not in the ids. Two styles can carry different ids and the same lighting slug, and an
# id-only check passes them. So the property the headline B−A depends on was asserted in a repo
# script that nothing on the cluster ran, and any hand-edited or substituted JSON would have
# generated 4 020 clips under a partition nobody committed, with the sbatch faithfully recording
# the digest of the wrong file as evidence.
#
# Three things are checked, in increasing strength, and all of them BEFORE the expansion:
#   1. the rendering matches its own committed .sha256 sidecar (bytes);
#   2. scripts/check_style_partition.py passes against the committed styles.toml — which is what
#      re-derives the rendering, compares it byte-for-byte with what is on disk, re-checks the
#      content hash against styles.toml.sha256, and enforces the axis-slug disjointness of
#      TRAIN/EVAL/IDENTITY plus the arm-C frame-match arithmetic;
#   3. no OPEN blocking_todo names this STYLE_SET (below, in the expansion).
#
# Run BEFORE `source ${FRAMEWORK}/.venv/bin/activate` on purpose: the checker is stdlib-only and
# needs tomllib (3.11+), which the module-loaded anaconda3/python3.12 guarantees and a vendored
# framework venv of unknown vintage does not.
# ------------------------------------------------------------------------------------------------
STYLES_FILE_SHA=$(sha256sum "${STYLE_PARTITION}" | cut -d' ' -f1)
if [[ -f "${STYLE_PARTITION}.sha256" ]]; then
  STYLES_SIDECAR=$(tr -d '[:space:]' < "${STYLE_PARTITION}.sha256")
  [[ "${STYLES_FILE_SHA}" == "${STYLES_SIDECAR}" ]] || {
    echo "FATAL: ${STYLE_PARTITION} does not match its committed .sha256 sidecar."
    echo "       on disk  ${STYLES_FILE_SHA}"
    echo "       sidecar  ${STYLES_SIDECAR}"
    echo "       The partition is a PRE-COMMITMENT (PR-08 §5): a file edited after it was"
    echo "       committed is not one. Regenerate it from styles.toml with"
    echo "       scripts/check_style_partition.py --emit-json --write-hash, do not re-hash it."
    exit 1; }
else
  echo "FATAL: ${STYLE_PARTITION}.sha256 missing — there is nothing to prove this is the"
  echo "       committed rendering. PR-08 §5 requires the partition to be committed BEFORE the"
  echo "       first clip; an uncommitted one cannot have been."
  exit 1
fi
echo "=== verifying the committed partition with ${STYLE_CHECKER}"
if ! python "${STYLE_CHECKER}" --partition "${STYLE_SOURCE}" --derived "${STYLE_PARTITION}"; then
  echo "FATAL: the committed style partition does not verify. The generator reads the RENDERING,"
  echo "       and the failure above says the rendering, its sidecar and styles.toml do not agree"
  echo "       — so nothing this job generated could be attributed to a committed partition."
  echo "       PR-08 §5's disjointness (checked on all four axis slugs, not on ids) is what makes"
  echo "       the eval set visually shifted; §6's headline B−A is unreadable without it."
  exit 1
fi
# The PR-08 §6 "partition hash" is the CONTENT hash — sha256 over the canonical JSON of styles.toml
# minus its [hash] and [consumer] tables — and NOT the byte hash of either file. The rendering
# carries it, the checker above has just re-derived it from the source and confirmed the sidecar,
# so it can be read out here and recorded under a name that says which of the two it is (FIX 9).
STYLES_CONTENT_SHA=$(python -c '
import json, sys
d = json.load(open(sys.argv[1]))
v = d.get("source_content_sha256")
if not v:
    sys.exit("FATAL: the rendering carries no source_content_sha256 — the PR-08 §6 partition hash "
             "would be unrecordable. Regenerate it with check_style_partition.py --emit-json.")
print(v)' "${STYLE_PARTITION}")
echo "=== partition  content ${STYLES_CONTENT_SHA}"
echo "===            file    ${STYLES_FILE_SHA}  (bytes of the rendering)"

echo "=== T-040 Transfer2.5 restyle, PR-08 (rule T40_RULE_V1)"
echo "=== generator  ${TRANSFER_MODEL_ID}@${TRANSFER_MODEL_REVISION}  (FROZEN)"
echo "=== checkpoint ${TRANSFER_CHECKPOINT_PATH}"
echo "=== source     ${SOURCE}"
echo "=== styles     ${STYLE_PARTITION}  set=${STYLE_SET}"
echo "=== chunk      ${CHUNK_INDEX} of ${CHUNK_TOTAL}   run ${RUN_ID}"

mkdir -p "${CHUNK_DIR}" "${CLIPS}" "${RAW}" "${PAYLOADS}"
# Marker first, before anything is written next to it. 92_fetch_g1_corpus.sbatch drops a SMOKE file
# inside a rehearsal corpus for the same reason: a downstream job can refuse a tree by `ls` without
# having to understand what produced it.
cat > "${CLIPS}/NOT_TRAINING_DATA" <<'MARKER'
These clips are Cosmos-Transfer2.5 output under T-040 / PR-08.

NOTHING HERE MAY ENTER A TRAINING CORPUS until all three of PR-08 §6's VOID gates have passed on
this tree:

  G0a  screen_corpus.py --expect against the SOURCE's M1/M2/M3 within EXPECT_TOL (0.02/0.02/0.05).
       A restyle changes no action, so a deviation is not a finding about the corpus — it is proof
       the pipeline corrupted or reordered the action labels.
  G0b  object and plate centroids agree with the source within GEOM_TOL - EST_DRIFT_P95.

G0c IS NOT ON THAT LIST, AND THAT IS NOT AN OMISSION. PR-08 §6 solves it BY CONSTRUCTION rather
than by a threshold: the real robot's pixels are unconditionally composited back over every
generated frame, so the generic-manipulator defect cannot enter and no gate has to detect it. That
composite runs inside scripts/restyle_transfer25.py, between the model writing a clip and the
driver calling the unit a success. A unit whose composite refused never reached `status: success`,
and its output was renamed to vision.uncomposited.mp4, so the harvest cannot file it either.

AND THAT IS CHECKED HERE RATHER THAN TAKEN ON TRUST. Beside every `<unit>.mp4` in this directory
there is a `<unit>.g0c.json`, written by the harvest out of that unit's own sample_outputs.json
BEFORE the clip was moved in, and only after all of these held: status success; backend transfer25
(a placeholder clip is a valid mp4, so it is refused by content and not by habit); g0c.composited
true; frames_composited == frames_total == the frame count that unit's work-list row declares; and
an area bound cross-checked against the segmenter that actually made the masks. A clip in here with
no `.g0c.json` beside it makes the harvest and the chunk record refuse — so if you can `ls` the
pair, the composite ran on that clip, and if you cannot, no part of this tree claims it did. The
robot-mask IoU inside that file is a DIAGNOSTIC ON THE GENERATOR AND NEVER A GATE — §6 says so
twice — and nothing compares it to anything.

Delete this file only from the job that runs G0a and G0b.
MARKER

# THE ROBOT SEGMENTER'S OPERATING POINT IS NOT A SUBMIT-TIME DECISION. scripts/estimators/apple_sam2
# reads BOX_THRESHOLD / TEXT_THRESHOLD and its retry pair from these four variables, deliberately,
# because a measurement sweep needs to move them. PR-08 §6 G0c's robot mask is made through that
# same adapter, and there a threshold decides WHICH PIXELS THE GENERATOR IS ALLOWED TO TOUCH: a
# lower one grows the mask, a higher one shrinks it, and a smaller mask is generated manipulator
# left in the frame — the exact defect G0c exists to exclude, arriving through an inherited
# environment variable that nobody typed into this file. robot_composite refuses a per-run prompt
# for that reason and cannot refuse these, because the adapter owns them. So they are cleared here,
# in the one place both driver invocations below inherit from, and the values that end up in every
# clip's record are the committed ones.
unset WAM_PR08_BOX_THRESHOLD WAM_PR08_TEXT_THRESHOLD || true
unset WAM_PR08_RETRY_BOX_THRESHOLD WAM_PR08_RETRY_TEXT_THRESHOLD || true
# Same argument, one level up: the robot prompt is a committed constant with no override, and the
# APPLE prompt's override must not reach a job whose records quote a prompt.
unset WAM_PR08_OBJECT_PROMPT || true

# shellcheck source=/dev/null
source "${FRAMEWORK}/.venv/bin/activate"

# -- the chunk's work list -----------------------------------------------------------------------
# Deterministic and stamped. Episodes sorted by id, then a stride over CHUNK_TOTAL so the frame
# counts balance across chunks without a packing heuristic nobody can reproduce by hand.
#
# A WORK UNIT IS (EPISODE, STYLE, REPEAT), AND THE REPEAT IS THE 2026-08-15 FIX. It used to be
# (episode, style) with no repetition, which silently made arm C 402 clips against arm B's 4 020 —
# PR-08 §5 requires arm C to be "same generator, same pipeline, SAME FRAME COUNT, no added
# diversity", and a control at a tenth of the treatment's volume cannot separate "diversity helped"
# from "more data helped", which is the one confound arm C exists to remove. The committed
# partition now declares the repetition itself (`repeats` and `seeds` on every style, plus
# `volume` and `seed_schedule`), and this expansion READS it. Inferring it — "there is one identity style, so arm C is one clip
# per episode" — is what produced the defect, so nothing here infers anything: a style with no
# integer `repeats` is a fatal error, not a default of 1.
#
# THE SEEDS ARE COMMITTED, NOT DRAWN, AND THE SCHEDULE IS READ RATHER THAN DERIVED (2026-08-15,
# second amendment to T40_STYLES_V1). Every style in the rendering carries a resolved `seeds` list,
# one entry per repeat in repeat order, so repeat r of style s is generated under s["seeds"][r] and
# this script computes no index of its own. It used to index ONE shared list by the repeat number —
# volume.repeat_seeds[r], under the committed rule "a style with repeats = 1 uses repeat_seeds[0]"
# — and because every TRAIN and every EVAL style has repeats = 1, that put all 4 020 arm-B clips
# AND all 2 010 eval clips on seed 7001 while arm C alone spanned ten. Two defects in one line:
# B − C became a difference of seeds as well as of prompts, and arm B's training clips shared their
# initial latent with the clips the headline is scored on while arm C's did not, which lets a
# seed-specific generator fingerprint inflate the headline. The committed schedule
# ([seed_schedule] in styles.toml) now gives train and identity the SAME ten seeds and the eval set
# a disjoint block, and both invariants are re-checked below — this job is the last gate before a
# clip exists. A seed drawn at submit time, or chosen after seeing which identity samples came out
# well, would make the control a selection. The seed travels in the work unit, is hashed into
# WORK_SHA (so the stamp below refuses a resume under different seeds), and is recorded in
# chunk_metadata.json — which is what makes any single clip reproducible from the record alone.
WORK_LIST=${CHUNK_DIR}/work.jsonl
PARTITION_FACTS=${CHUNK_DIR}/partition_facts.json
python - "${SOURCE}/manifest.json" "${STYLE_PARTITION}" "${STYLE_SET}" \
         "${CHUNK_INDEX}" "${CHUNK_TOTAL}" "${WORK_LIST}" "${TIMING_MODE}" "${PARTITION_FACTS}" \
         "${STAGE}" "${STAGE1_TRAIN_STYLES}" <<'PY'
import hashlib, json, pathlib, re, sys
man_p, styles_p, style_set, idx, total, out, timing, facts_p, stage, k1 = sys.argv[1:11]
idx, total, timing, k1 = int(idx), int(total), int(timing), int(k1)
man = json.loads(pathlib.Path(man_p).read_text())
res = tuple(man.get("resolution") or ())
if res != (640, 480):
    sys.exit(f"FATAL: {man_p} declares resolution {res or 'nothing'}, not (640, 480). PR-08 §3 "
             "fixes 640x480 as the GR00T N1.7 ego_view contract; a restyle at any other size is "
             "not the registered experiment.")
styles = json.loads(pathlib.Path(styles_p).read_text())
SETS = ("train", "eval", "identity")
for half in ("train", "eval"):
    if half not in styles:
        sys.exit(f"FATAL: {styles_p} has no '{half}' half. PR-08 §5 requires a committed DISJOINT "
                 "TRAIN_STYLES / EVAL_STYLES partition.")
tr = {s["id"] for s in styles["train"]}
ev = {s["id"] for s in styles["eval"]}
if tr & ev:
    sys.exit(f"FATAL: {styles_p} train and eval halves overlap on {sorted(tr & ev)}. PR-08 §5's "
             "whole point is that the eval domain cannot be chosen after seeing the restyles.")
# All three sets come from the COMMITTED file, including identity. An earlier draft of this script
# synthesised the identity style here ({"id": "identity", "prompt": None}) on the assumption that
# arm C needed no committed prompt. That was wrong twice over: the partition really does ship an
# identity set (rule T40_STYLES_V1), and arm C is the control that decides whether a gain from arm
# B is diversity or generator fingerprint — so its prompt is exactly as load-bearing as B's, and
# generating it from a prompt this file made up would make the control unattributable.
if style_set not in styles:
    sys.exit(f"FATAL: {styles_p} has no '{style_set}' set. Arm C (identity) needs a COMMITTED "
             "style whose prompt is the source's own appearance; it must not be invented here.")

# -- the committed volume block, which is what makes the expansion readable rather than inferred --
vol = styles.get("volume")
if not isinstance(vol, dict):
    sys.exit(f"FATAL: {styles_p} carries no 'volume' block. It is where the committed repeat "
             "counts, the repeat seeds and the whole-partition clip totals live; without it arm "
             "C's frame-match to arm B is a number this script would have to invent. Regenerate "
             "the rendering with scripts/check_style_partition.py --emit-json.")
# -- the committed SEED SCHEDULE ------------------------------------------------------------------
# The rendering carries `seeds` on every style, already resolved: seeds[r] is the seed for repeat r
# of that style. This script therefore reads a seed and does not compute one. It used to index one
# shared volume.repeat_seeds list by r, and since every TRAIN and every EVAL style has repeats = 1
# that resolved to repeat_seeds[0] for all of them — arm B and the whole eval set on seed 7001,
# arm C alone on ten. Recomputing an index rule here is what let a schedule nobody had looked at
# decide the headline, so there is no fallback and no default: a style without a usable `seeds`
# list is fatal.
sched = styles.get("seed_schedule") or {}
if not sched.get("blocks"):
    sys.exit(f"FATAL: {styles_p} carries no seed_schedule.blocks. The seed VALUES are not the "
             "commitment — the assignment is. Regenerate the rendering with "
             "scripts/check_style_partition.py --emit-json against a styles.toml that has a "
             "[seed_schedule] block.")


def repeats_of(s):
    r = s.get("repeats")
    if not isinstance(r, int) or isinstance(r, bool) or r < 1:
        sys.exit(f"FATAL: style {s.get('id')!r} in {styles_p} has repeats={r!r}. Every style must "
                 "declare a positive integer `repeats`. There is deliberately NO default: "
                 "defaulting it to 1 is exactly how arm C shipped at a tenth of arm B's volume.")
    return r


def seeds_of(s):
    """The committed seed per repeat of this style. Read, never derived."""
    seeds = s.get("seeds")
    r = repeats_of(s)
    if not isinstance(seeds, list) or len(seeds) != r:
        sys.exit(f"FATAL: style {s.get('id')!r} in {styles_p} declares repeats={r} but carries "
                 f"seeds={seeds!r}. Every style-instance is generated under a seed committed "
                 "before the first clip; there is one entry per repeat, in repeat order, and no "
                 "default.")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in seeds):
        sys.exit(f"FATAL: style {s.get('id')!r} in {styles_p} has non-integer seeds: {seeds!r}.")
    if len(set(seeds)) != len(seeds):
        sys.exit(f"FATAL: style {s.get('id')!r} in {styles_p} repeats a seed: {seeds!r}. Two "
                 "repeats under the same seed are one sample of the generator's variance recorded "
                 "twice, so arm C would be frame-matched to arm B on paper and not in fact.")
    return [int(v) for v in seeds]


# THE TWO SCHEDULE INVARIANTS, re-checked here against the sets as expanded rather than trusted
# from the file. check_style_partition.py already enforces both against styles.toml and this job
# runs it above — this is the same property re-proved against the thing that is actually about to
# be generated from, and it is the last gate before a clip exists.
seeds_used = {k: {x for s in styles[k] for x in seeds_of(s)} for k in SETS if k in styles}
if "train" in seeds_used and "identity" in seeds_used:
    if seeds_used["train"] != seeds_used["identity"]:
        sys.exit("FATAL: ARM B AND ARM C DO NOT SPAN THE SAME SEED SET. train "
                 f"{sorted(seeds_used['train'])} vs identity {sorted(seeds_used['identity'])}. "
                 "PR-08 §5's arm C decides whether a gain in arm B is diversity or the generator; "
                 "if the arms are also generated under different seeds, B - C is a difference of "
                 "seeds AND prompts and isolates nothing.")
if "eval" in seeds_used:
    training_seeds = set()
    for k in ("train", "identity"):
        training_seeds |= seeds_used.get(k, set())
    shared = seeds_used["eval"] & training_seeds
    if shared:
        sys.exit(f"FATAL: EVAL SEEDS INTERSECT A TRAINING SEED BLOCK: {sorted(shared)}. The "
                 "headline is scored on EVAL_STYLES clips, so an eval clip sharing its initial "
                 "latent with a training clip hands the arm that owns it a match to the scoring "
                 "distribution that comes from the generator's seed-specific behaviour rather "
                 "than from diversity. That is the confound arm C exists to remove.")


# -- OPEN blocking_todos veto this style set, TIMING INCLUDED --------------------------------------
# The partition ships these as records, and says in as many words that they "block GENERATION, not
# this check" — check_style_partition.py prints them and passes, because committing the file is
# licensed (PR-08 §1) while generating from it is not. This job is the generation, so this job is
# where they bite.
#
# TIMING IS NOT EXEMPT (changed 2026-08-15). It used to WARN and continue, on the argument that
# PR-08 §1 licenses the measurement and the timing branch writes into a throwaway root it deletes.
# Both halves of that are true and the conclusion was still wrong, for the todo that actually
# exists: T40-TODO-01-identity-prompt-provenance blocks STYLE_SET=identity because arm C's prompt
# has unverified provenance — it is a T-041 machine caption of ONE clip, applied unchanged to all
# 402. A timing run with STYLE_SET=identity would put that prompt through the generator and
# produce a clip from it. Deleting the clip afterwards does not un-generate it, and it is not the
# clip that mattered anyway: the number the timing run EXISTS to produce would have been measured
# under a prompt that the open todo says may still change, and PR-08 §8 item 3's budget line is
# then derived from a measurement of a pipeline nobody has committed to. "Writes no clip" was
# never the reason §1 licenses timing; §1 licenses it because it is a measurement, and a
# measurement of an unprovenanced configuration is not the measurement §8 item 3 asks for.
#
# The exemption also read as a hole rather than a rule: TIMING=1 was the one flag that turned a
# refusal into a printed line, on the one style set whose prompt is under review. The cost of
# closing it is one line on the submit form — time STYLE_SET=train, which is what the usage header
# at the top of this file already shows, and which is unblocked.
def blocks_this_set(todo):
    """Which style sets a todo's free-text `blocks` field names.

    Read as `STYLE_SET=<set>` first and only fall back to bare set names, so that a note whose
    prose happens to contain "train" does not veto the train submission. A todo that names no set
    at all is treated as blocking EVERYTHING: the conservative direction is the one where an
    unparseable blocker stops the run rather than passing it.
    """
    blocks = str(todo.get("blocks", "")).lower()
    named = set(re.findall(r"style_set\s*=\s*([a-z]+)", blocks))
    if not named:
        named = set(re.findall(r"\b(train|eval|identity|all)\b", blocks))
    return (not named) or style_set in named or "all" in named


open_todos = [t for t in (styles.get("blocking_todos") or [])
              if str(t.get("status", "")).upper() == "OPEN" and blocks_this_set(t)]
if open_todos:
    lines = "\n".join(f"         - {t.get('id')}: {str(t.get('why_blocking', '')).strip()}"
                      for t in open_todos)
    sys.exit(f"FATAL: {len(open_todos)} OPEN blocking_todo(s) in {styles_p} name "
             f"STYLE_SET={style_set}:\n{lines}\n"
             "       These are recorded as blocking GENERATION, and this is the generation. "
             "Close them in styles.toml (with the evidence the record asks for) and "
             "regenerate the rendering; do not delete them to get past this.\n"
             + ("       TIMING=1 DOES NOT EXEMPT THIS. The timing branch does put the prompt "
                "through the generator, and deleting the output afterwards does not make the "
                "measurement one of a configuration anybody has committed to. Time an unblocked "
                "style set instead: STYLE_SET=train."
                if timing else
                "       TIMING=1 does not exempt it either, so do not reach for it."))

chosen = styles[style_set]
if not chosen:
    sys.exit(f"FATAL: {styles_p}'s '{style_set}' set is empty.")
for s in chosen:
    if not s.get("id") or not s.get("prompt"):
        sys.exit(f"FATAL: a style in '{style_set}' has no id or no prompt: {s!r}")
    repeats_of(s)

# -- T40_RULE_V11's stage selector ----------------------------------------------------------------
# THE STAGE IS A PREFIX OF THE COMMITTED ORDER, NEVER A SELECTION. §2 names stage 1's four styles
# explicitly and in the order styles.toml already carries them, so the selector takes a slice and
# does not sort, filter or choose. Anything else would let the stage be picked after seeing which
# styles rendered well, which is the failure the committed partition exists to prevent.
#
# THE TWO SETS ARE STAGED DIFFERENTLY BECAUSE THEY ARE SHAPED DIFFERENTLY, and this is the part
# §2.4 flags as easy to get wrong. TRAIN is ten styles at repeats=1, so its stage is a slice of the
# STYLE LIST. IDENTITY is ONE style at repeats=10, so its stage is a slice of the REPEAT RANGE —
# slicing its style list would take all ten repeats or none. Arm C stays frame-matched only if both
# sides are cut the same way, and the seeds prove it: train styles 0..3 carry 7001..7004 and
# identity repeats 0..3 carry the same four, so a prefix on each side spans the same seed block.
def repeat_span(s, kset):
    """The half-open repeat range of style ``s`` of set ``kset`` under this stage.

    WHICH AXIS A SET IS CUT ON IS A PROPERTY OF THE SET, NOT OF THE STYLE'S REPEAT COUNT, and
    getting that backwards is a latent bug the committed partition happens to hide: every TRAIN
    style there carries repeats=1, so a rule keyed on "n > 1" behaves identically to the right one
    today and diverges the moment a train style gains a repeat. TRAIN and EVAL are staged by
    slicing the STYLE LIST, so a style that is in this stage is in it WHOLE — cutting its repeats
    as well would drop clips out of both stages and out of the partition. Only IDENTITY, which is
    one style carrying all ten instances, is cut on the repeat axis.
    """
    n = repeats_of(s)
    if stage == "all" or kset != "identity":
        return 0, n
    if n <= k1:
        sys.exit(f"FATAL: style {s['id']!r} has repeats={n}, which is not more than the stage-1 "
                 f"prefix k={k1}, so stage 2 would be empty for it. T40_RULE_V11 §2 sizes the "
                 "stages against the committed partition (identity repeats=10, k=4); this "
                 "rendering is not that partition.")
    return (0, k1) if stage == "1" else (k1, n)


if stage != "all":
    # THE REPEAT-AXIS CUT IS ONLY WELL-DEFINED FOR ONE STYLE. With two identity styles at five
    # repeats each, taking repeats 0..k-1 of EACH yields 2k instances rather than k, so stage 1
    # would quietly draw twice its share and arm C would stop being frame-matched by exactly the
    # factor nobody looked at. The committed partition has one (identity-source, repeats=10); a
    # rendering with more is not that partition and is refused rather than reinterpreted.
    if len(styles.get("identity") or []) > 1:
        sys.exit(f"FATAL: the rendering carries {len(styles['identity'])} identity styles. "
                 "T40_RULE_V11 §2 stages arm C by slicing the REPEAT range of its single style, "
                 "and that cut is not well defined across several. Stage this partition only "
                 "after registering how arm C is to be divided.")
    if style_set == "train":
        if len(chosen) <= k1:
            sys.exit(f"FATAL: '{style_set}' has {len(chosen)} styles, which is not more than the "
                     f"stage-1 prefix k={k1}: stage 2 would be empty. T40_RULE_V11 §2 sizes the "
                     "stages against the committed partition (10 train styles, k=4).")
        chosen = chosen[:k1] if stage == "1" else chosen[k1:]
    elif style_set == "eval":
        # STAGE=1 + eval was already refused in bash, before a GPU was asked for. Stage 2 takes the
        # eval set whole: §2 defers it out of stage 1, it does not split it.
        pass
    # identity is cut on the repeat axis by repeat_span(), not here.

# What THIS invocation will actually emit, per episode — the number the ceiling gate must price and
# the arm-C guard must compare. Computed for every set, not just this one, because a stage-level
# frame-match is a statement about two sets at once and this invocation only generates one of them.
staged = {}
for kset in SETS:
    if kset not in styles:
        continue
    pool = styles[kset]
    if stage != "all" and kset == "train":
        pool = pool[:k1] if stage == "1" else pool[k1:]
    if stage == "1" and kset == "eval":
        staged[kset] = 0          # deferred, not cut. Priced at zero for stage 1.
        continue
    staged[kset] = sum(b - a for s in pool for a, b in [repeat_span(s, kset)])

# -- the whole partition, priced here so the ceiling gate can be about the EXPERIMENT -------------
# The gate below enforces TWO bounds (T40_RULE_V2 §3): the whole-partition projection against
# PARTITION_CEILING_GPU_H, and this style set's projection against its CEILING_GPU_H share. Only
# the first of those can catch "one budget authorised three times", and this invocation can only
# see its own style set — so the frames of the WHOLE partition are computed here, every style of
# every set, times its repeats, times every episode in the manifest, and handed to the gate
# alongside this chunk's own frames. With arm C at ten clips per episode the identity invocation
# is the largest of the three, which is exactly why a per-set number cannot stand in for the whole.
eps = sorted(man["episodes"], key=lambda e: str(e["id"]))
if not eps:
    sys.exit(f"FATAL: {man_p} lists no episodes.")
corpus_frames = sum(int(e["frames"]) for e in eps)
declared_eps = vol.get("episodes")
if isinstance(declared_eps, int) and declared_eps != len(eps):
    sys.exit(f"FATAL: {man_p} has {len(eps)} episodes but the committed partition prices the "
             f"experiment at volume.episodes={declared_eps}. The GPU-h ceiling, arm C's "
             "frame-match and every clip total in the partition are derived from that number, so "
             "generating against a different corpus size spends against a budget for a different "
             "experiment. Re-derive the corpus or amend the partition — do not proceed.")
instances = {k: sum(repeats_of(s) for s in styles[k]) for k in SETS if k in styles}
partition_instances = sum(instances.values())
partition_frames = corpus_frames * partition_instances
partition_clips = len(eps) * partition_instances
declared_instances = vol.get("style_instances")
if isinstance(declared_instances, int) and declared_instances != partition_instances:
    sys.exit(f"FATAL: the rendering expands to {partition_instances} style-instances "
             f"({instances}) but volume.style_instances says {declared_instances}. The ceiling "
             "basis and the expansion disagree; one of them is wrong.")
declared_clips = vol.get("whole_partition_clips")
if isinstance(declared_clips, int) and declared_clips != partition_clips:
    sys.exit(f"FATAL: the expansion yields {partition_clips} clips over the whole partition but "
             f"volume.whole_partition_clips says {declared_clips}.")
# Arm C's frame-match, re-checked against the expansion rather than trusted from the volume block:
# this is the property the repeat loop exists to produce and it costs one line to prove here.
if "train" in instances and "identity" in instances and instances["train"] != instances["identity"]:
    sys.exit(f"FATAL: ARM C IS NOT FRAME-MATCHED. The expansion gives arm B "
             f"{instances['train']} clips per episode and arm C {instances['identity']}. PR-08 §5 "
             "requires the same frame count; unequal, B − C confounds diversity with volume.")
# THE SAME PROPERTY, AT STAGE LEVEL, WHICH IS THE ONE THAT IS ACTUALLY GENERATED.
# T40_RULE_V11 §2.4: the guard above is computed over the WHOLE rendering and is independent of
# `chosen`, so under staging it passes at 10 == 10 while saying NOTHING about the stage. A stage
# that generated four train styles against ten identity repeats would clear every other check in
# this file and produce an arm C with 2.5x arm B's volume — B − C would then measure volume and be
# read as diversity. This is the enforcement behind V11 §0's promise that B and C stay matched "at
# every stage", and until 2026-08-24 that promise had none.
if stage != "all" and "train" in staged and "identity" in staged:
    if staged["train"] != staged["identity"]:
        sys.exit(f"FATAL: ARM C IS NOT FRAME-MATCHED AT STAGE {stage}. This stage gives arm B "
                 f"{staged['train']} clips per episode and arm C {staged['identity']}. The whole-"
                 f"partition guard above passes ({instances.get('train')} == "
                 f"{instances.get('identity')}) and is not the property that matters here: PR-08 §5 "
                 "requires the frame match in what is GENERATED, and this stage is what is "
                 "generated. T40_RULE_V11 §2.4.")
    # Seeds too, for the reason the whole-file check at the top exists: equal COUNTS drawn from
    # different seed blocks is still a confound, and it is the one a count comparison cannot see.
    def staged_seeds(kset):
        pool = styles[kset]
        if kset == "train":
            pool = pool[:k1] if stage == "1" else pool[k1:]
        return {x for s in pool for x in seeds_of(s)[slice(*repeat_span(s, kset))]}
    if staged_seeds("train") != staged_seeds("identity"):
        sys.exit(f"FATAL: ARM B AND ARM C DO NOT SPAN THE SAME SEED SET AT STAGE {stage}: "
                 f"{sorted(staged_seeds('train'))} vs {sorted(staged_seeds('identity'))}. Equal "
                 "clip counts over different seeds is still a confound; arm C exists to isolate "
                 "the generator's fingerprint, and it can only do that seed for seed.")

mine = eps[idx - 1::total]
rows, frames = [], 0
with open(out, "w") as f:
    for e in mine:
        for s in chosen:
            style_seeds = seeds_of(s)
            lo, hi = repeat_span(s, style_set)
            for r in range(lo, hi):
                # The unit id carries the repeat on EVERY style, including the repeats=1 ones. A
                # suffix only where it is needed would make two naming schemes, and the clip files
                # are named after the unit — so `__r00` present everywhere is also the cheapest
                # way to tell a repeat-expanded tree from a pre-fix one by `ls`.
                unit = f'{e["id"]}__{s["id"]}__r{r:02d}'
                # `stage` travels WITH the clip, per T40_RULE_V11 §2.4's "recorded per clip". The
                # unit id deliberately does not encode it: the stages hold disjoint styles and
                # disjoint identity repeats, so the id is already unique, and putting the stage in
                # the filename would make a clip's name depend on a decision about scheduling
                # rather than on what the clip IS.
                row = {"unit": unit, "episode": e["id"], "frames": int(e["frames"]),
                       "style": s["id"], "repeat": r, "seed": style_seeds[r], "stage": stage}
                f.write(json.dumps(row) + "\n")
                # Seed and stage in the hashed material, not just the unit id: a resume under a
                # different seed list — or under a different stage — is a different experiment in
                # the same directory, and the stamp check below is the only thing that could ever
                # notice.
                rows.append(f'{unit}|{style_seeds[r]}|s{stage}')
                frames += int(e["frames"])
sha = hashlib.sha256("\n".join(rows).encode()).hexdigest()
pathlib.Path(out + ".sha256").write_text(sha + "\n")
pathlib.Path(out + ".frames").write_text(str(frames) + "\n")
pathlib.Path(facts_p).write_text(json.dumps({
    "corpus_episodes": len(eps), "corpus_frames": corpus_frames,
    "style_instances_per_set": instances, "style_instances_whole_partition": partition_instances,
    "whole_partition_clips": partition_clips, "whole_partition_frames": partition_frames,
    "seed_blocks": sched.get("blocks"),
    "seed_assignment": sched.get("assignment"),
    "seed_rule": sched.get("rule"),
    "seeds_by_style": {s["id"]: seeds_of(s) for k in SETS if k in styles for s in styles[k]},
    "seeds_used_by_set": {k: sorted(v) for k, v in seeds_used.items()},
    "repeats_by_style": {s["id"]: repeats_of(s) for k in SETS if k in styles for s in styles[k]},
    "this_set": style_set, "this_set_instances_per_episode": instances[style_set],
    # -- T40_RULE_V11 ----------------------------------------------------------------------------
    # The ceiling gate reads THIS file, so the staged numbers have to reach it here or the gate
    # prices a stage against the whole set's share and refuses a run that is well inside budget.
    # `stage_instances_per_set` is what this stage emits; `style_instances_per_set` above is left
    # untouched and still describes the whole committed partition, because the whole-partition
    # ceiling is a bound on the EXPERIMENT and does not move when one invocation generates less.
    "stage": stage,
    "stage_rule": "T40_RULE_V11 §2 (adopted 2026-08-24)",
    "stage1_train_styles": k1,
    "stage_instances_per_set": staged,
    "this_stage_instances_per_episode": staged.get(style_set, instances[style_set]),
    "stage_styles": [s["id"] for s in chosen],
    "stage_clips": len(eps) * staged.get(style_set, instances[style_set]),
    "stage_frames": corpus_frames * staged.get(style_set, instances[style_set]),
}, indent=2) + "\n")
print(f"chunk {idx}/{total}: {len(mine)} episodes, {len(chosen)} styles, "
      f"{staged.get(style_set, instances[style_set])} clips/episode in set '{style_set}' "
      f"at stage {stage}, {len(rows)} units, {frames} frames  sha256 {sha[:16]}")
print(f"stage {stage}: {staged} instances/episode by set, styles {[s['id'] for s in chosen]}")
print(f"whole partition: {partition_instances} style-instances ({instances}), "
      f"{partition_clips} clips, {partition_frames} frames over {len(eps)} episodes")
PY
WORK_SHA=$(cat "${WORK_LIST}.sha256")
CHUNK_FRAMES=$(cat "${WORK_LIST}.frames")
PARTITION_FRAMES=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["whole_partition_frames"])' "${PARTITION_FACTS}")
# One variant of the corpus, in frames, straight off the manifest the expansion above read. The
# TIMING branch below projects `seconds_per_frame x this` into gpu_hours_per_variant, and until
# 2026-08-27 it multiplied by a literal 172_000 instead — see FRAMES_PER_VARIANT there. Read here,
# beside the partition-wide count it is a factor of, because both come out of the same file and a
# reader comparing them should not have to go looking for the second one.
CORPUS_FRAMES=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["corpus_frames"])' "${PARTITION_FACTS}")

# -- stamp: refuse to resume into a directory that is not this invocation's -----------------------
# scripts/run_i8_rung_local.sh:165-195, and the reasoning transfers exactly. Every value below is a
# RUNTIME argument: none is inside a checkpoint config or config_hash, so a resume under a
# different one would silently be a different experiment in the same directory and nothing
# downstream could detect it. Pin on pass 1, refuse to contradict it afterwards.
ENV_FILE=${CHUNK_DIR}/chunk.env
STAMP="RUN_ID=${RUN_ID} STYLE_SET=${STYLE_SET} STAGE=${STAGE} CHUNK=${CHUNK_INDEX}/${CHUNK_TOTAL} \
WORK_SHA=${WORK_SHA} STYLES_CONTENT_SHA=${STYLES_CONTENT_SHA} SOURCE=${SOURCE} \
CKPT=${TRANSFER_CHECKPOINT_PATH} REV=${TRANSFER_MODEL_REVISION} RES=640x480 NPROC=${NPROC}"
if [[ -f "${ENV_FILE}" ]]; then
  PREV=$(cat "${ENV_FILE}")
  if [[ "${PREV}" != "${STAMP}" ]]; then
    echo "FATAL: ${CHUNK_DIR} was started as:  ${PREV}"
    echo "       this invocation is:           ${STAMP}"
    echo "       Resuming into it would mix two experiments. Use a different RUN_ID."
    exit 1
  fi
else
  printf '%s' "${STAMP}" > "${ENV_FILE}"
fi

# ------------------------------------------------------------------------------------------------
# Can THIS THROUGHPUT.json be built on? One definition, called by both paths that read it.
# ------------------------------------------------------------------------------------------------
# Exit 0 = qualified. 2 = unreadable. 3 = disqualified. 4 = unproven (nothing in it says a clip was
# ever produced). Written once and called twice on purpose: the TIMING path asks before REUSING a
# measurement, and the GENERATE path asks before DERIVING a ceiling from one, and those two must
# never be able to answer differently about the same file.
throughput_qualification() {
  python - "$1" <<'PY'
import json, sys
p = sys.argv[1]
try:
    tp = json.loads(open(p).read())
except (OSError, ValueError) as exc:
    print(f"    UNREADABLE: {exc}")
    sys.exit(2)
if not isinstance(tp, dict):
    print(f"    NOT A THROUGHPUT RECORD: the file holds a {type(tp).__name__}, not an object.")
    sys.exit(2)
# A disqualification, however it was written. None of these fields exist in the record this job
# writes today, and that is the point: they are the shapes a LATER version or a hand-annotation
# would use to say "do not build a budget on this", and a reuse check that only understood today's
# schema would silently ignore tomorrow's.
dq = []
if tp.get("disqualified"):
    dq.append(f"disqualified={tp['disqualified']!r}")
reasons = tp.get("gate_disqualified_reasons")
if isinstance(reasons, (list, tuple)) and reasons:
    dq.append(f"gate_disqualified_reasons={list(reasons)!r}")
if tp.get("measurement_qualified") is False:
    dq.append("measurement_qualified=false")
if dq:
    print("    DISQUALIFIED: " + "; ".join(dq))
    sys.exit(3)
# Did a clip actually come out of the run that produced this? The writer below records the count it
# measured, under a schema name, and both landed on 2026-08-27. An artifact without them is not
# necessarily bad — it is UNPROVEN, which for a number that prices 4.29M frames is the same thing.
# Job 189142's artifact is exactly this case: every field it carries is well-formed and its
# seconds_per_frame is a clock around a crash.
if tp.get("schema") != "wam.transfer25_throughput/1":
    print(f"    UNPROVEN: schema is {tp.get('schema')!r}, not 'wam.transfer25_throughput/1'.")
    print("    It was written before this job recorded whether the timed unit produced a clip, so")
    print("    nothing in it can distinguish a timed episode from a timed crash.")
    sys.exit(4)
ok = tp.get("units_succeeded")
if not isinstance(ok, int) or isinstance(ok, bool) or ok < 1:
    print(f"    NO SUCCESSFUL UNIT: units_succeeded={ok!r}. A wall clock around a unit that")
    print("    produced no clip is not a throughput number.")
    sys.exit(4)
print(f"    qualified: {ok} successful unit(s), {tp.get('frames')} frames, "
      f"{tp.get('seconds_per_frame')} s/frame, generator {tp.get('generator')}")
PY
}

# ------------------------------------------------------------------------------------------------
# TIMING — a GATE, not a warm-up. PR-08 §8 item 3 licenses exactly this: "one timed episode on an
# H200 at 640x480". It is the precondition for the full run, and it is where the number in
# CEILING_GPU_H is supposed to come from.
# ------------------------------------------------------------------------------------------------
THROUGHPUT=${OUT}/THROUGHPUT.json
if (( TIMING_MODE )); then
  TIMING_RESTARTS=${SLURM_RESTART_COUNT:-0}
  # 94's probe branch returns BEFORE the train section's pass rail, so a preempted probe requeued
  # itself onto the paid QoS with no bound at all, and redid work it had already finished. This
  # branch carries both guards for the same reason.
  #
  # BUT "the file is there" IS NOT "the measurement is done", AND UNTIL 2026-08-27 THIS READ IT AS
  # IF IT WERE. RUN_ID defaults to t040-transfer25-restyle (see the -- paths -- section above) and
  # the header's own recipe passed no RUN_ID, so a fresh timing submission resolved THROUGHPUT to
  # the path where job 189142's artifact still sits: 0.2 s/frame, produced by a run whose own log
  # says "0 success, 1 error" — the driver crashed in SetupArguments validation, generated nothing,
  # exited 0, and the clock around the crash became the budget. That file `cat`s and `exit 0`s in
  # under a second, the operator reads a completed measurement, and the whole partition prices at
  # 238.4 GPU-h against roughly 2 380 at the artifact's own end-to-end rate. A budget line that is
  # ~10x under is worse than no budget line, because PR-08 §8 item 3 makes this file the gate.
  #
  # So the three cases are now separated, and only one of them reuses anything:
  #   * an artifact that cannot PROVE a clip was produced is never reused, by anyone, for any
  #     reason. That is the classifier below and it is checked first, so no flag and no requeue can
  #     reach past it.
  #   * a REQUEUE of this same job (SLURM_RESTART_COUNT >= 1) finding an artifact it wrote itself
  #     keeps the original guard: the work is done, do not redo it on the paid QoS.
  #   * a FRESH submission finding one refuses and names the two exits — a new RUN_ID, which is the
  #     right answer, or REUSE_THROUGHPUT=1 if the operator really means to re-read this file.
  # There is no way to spend GPU-hours by getting this wrong; every branch here is free. What the
  # refusal protects is the number, not the allocation.
  if [[ -f "${THROUGHPUT}" ]]; then
    echo "=== ${THROUGHPUT} already exists. Reading what it actually records before trusting it:"
    TP_VERDICT=0
    throughput_qualification "${THROUGHPUT}" || TP_VERDICT=$?
    if (( TP_VERDICT != 0 )); then
      cat "${THROUGHPUT}"
      echo "FATAL: ${THROUGHPUT} cannot be reused as the PR-08 §8 item 3 measurement (see above)."
      echo "       This is refused OUTRIGHT — REUSE_THROUGHPUT does not override it, because the"
      echo "       file is not a measurement that failed to be re-read, it is a number with"
      echo "       nothing behind it. PR-08 §8 item 3 derives the whole partition's GPU-h ceiling"
      echo "       from seconds_per_frame; a figure from a run that generated no clip prices the"
      echo "       corpus against a crash."
      echo "       DO: submit with a fresh RUN_ID, e.g."
      echo "           TIMING=1 STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=1 \\"
      echo "             CONTROL=<key:weight,...> RUN_ID=t040-transfer25-restyle-timing-<date> \\"
      echo "             sbatch --time=01:30:00 97_transfer25_restyle.sbatch"
      echo "       Moving or deleting the old artifact is the operator's call and is NOT done here:"
      echo "       ${THROUGHPUT} is the evidence of what job 189142 actually produced."
      exit 1
    fi
    if (( TIMING_RESTARTS >= 1 )); then
      echo "=== requeue #${TIMING_RESTARTS} of this job, and the measurement it wrote is already"
      echo "    here. Nothing to time — this is the guard 94's probe branch was missing:"
      cat "${THROUGHPUT}"
      exit 0
    fi
    if [[ "${REUSE_THROUGHPUT:-0}" != "0" ]]; then
      echo "=== REUSE_THROUGHPUT=${REUSE_THROUGHPUT}: reusing the qualified artifact above as this"
      echo "    submission's measurement, on the operator's explicit instruction."
      cat "${THROUGHPUT}"
      exit 0
    fi
    cat "${THROUGHPUT}"
    echo "FATAL: this is a FRESH timing submission (SLURM_RESTART_COUNT=${TIMING_RESTARTS}) and"
    echo "       ${THROUGHPUT} already holds a measurement from an earlier run."
    echo "       Reusing it silently is how a stale seconds-per-frame becomes a budget line nobody"
    echo "       chose: RUN_ID defaults to t040-transfer25-restyle, so a submission that passes no"
    echo "       RUN_ID lands on whatever the last run left there. Choose, explicitly:"
    echo "         * measure again  ->  pass a fresh RUN_ID (recommended; it also keeps the two"
    echo "                              measurements side by side)"
    echo "         * reuse this one ->  re-submit with REUSE_THROUGHPUT=1"
    exit 1
  fi
  if (( TIMING_RESTARTS >= 2 )); then
    echo "FATAL: the timing run has been requeued ${TIMING_RESTARTS} times without producing"
    echo "       ${THROUGHPUT}. Something is wrong with the measurement, not with the queue."
    exit 1
  fi
  # "on an H200", singular. A multi-GPU number is not the number PR-08 asks for and would not be
  # comparable to the per-GPU billing rate the ceiling is derived against.
  if (( NPROC != 1 )); then
    echo "FATAL: TIMING=1 measures ONE episode on ONE H200 (PR-08 §8 item 3). Got NPROC=${NPROC}."
    echo "       Submit the timing run with the file's default --gres=gpu:1."
    exit 1
  fi
  TIMING_UNIT=$(head -1 "${WORK_LIST}")
  [[ -n "${TIMING_UNIT}" ]] || { echo "FATAL: empty work list — nothing to time."; exit 1; }
  # -- THE TIMED UNIT IS SCREENED BEFORE THE GPU IS ASKED FOR --------------------------------------
  # `head -1` is a DETERMINISTIC choice and, on this corpus, it is a choice this job cannot finish.
  # The work list is the manifest's episodes sorted by id crossed with the staged styles, so its
  # first row is always the lowest-numbered episode under the first style of the set — job 189644's
  # log records which unit that was. The robot mask of that episode is EMPTY on 254 of its 590
  # SOURCE frames (runs/pr08-robot-mask-area/POOLED.json), and robot_composite.check_mask refuses a
  # clip on ONE empty frame ("zero is zero", :1388). So the deterministic choice is refused on
  # every submission that has ever been made and on every one that ever will be. 189644 itself died
  # earlier still, on an area bound that had not been committed yet; the bound exists now, and what
  # the same submission buys today ends in the refusal below and no THROUGHPUT.json — the artifact
  # PR-08 §8 item 3 cannot close without.
  #
  # BE PRECISE ABOUT WHAT THIS SCREEN SAVES, because this file's older prose is not. The header
  # carries `--gres=gpu:1` (:260) and the file contains no `srun` at all, so by the time any line of
  # it executes the H200 is ALREADY ALLOCATED: nothing here can save a queue slot, and the
  # allocation's wall clock is billed either way. The Cosmos checkpoint is not saved either — the
  # driver's own preflight_source_masks (restyle_transfer25.py:775) already refuses ahead of
  # _transfer25_backend (:778), so this path never reached a checkpoint load even before the screen
  # existed. What IS saved is the driver's process start, the torch/cosmos imports, the
  # GroundingDINO + SAM 2 masker load and a segmentation pass over every source frame of the
  # episode: single-digit minutes inside an allocation that is paid for regardless.
  #
  # The minutes are not the reason. The reason is that the verdict is then reached from committed
  # evidence, written down as an artifact, and needs neither a GPU nor a working masker to reach —
  # so it is reproducible off the cluster and quotable in the record.
  #
  # NO EPISODE IS NAMED IN THIS FILE, here or anywhere: the id above is deliberately left to the
  # log and to the evidence artifact, because writing one into the job is how "which episode does
  # §8 item 3 time" gets answered by an sbatch line instead of by a registered rule.
  #
  # The refusal is not the defect — the refusal is correct and G0c stays exactly as it is. The
  # defect is asking Slurm for a GPU to discover something that was measured on 2026-08-25 and has
  # been sitting on disk ever since. check_mask's verdict on an episode is a function of that
  # episode's SOURCE masks, the pooled measurement is those masks' area fractions at stride 1 over
  # every frame, and the committed bound is the same number the composite loads: the verdict is
  # therefore computable here, for free, in the seconds before the job would otherwise start
  # loading a checkpoint.
  #
  # THIS SCREEN ONLY EVER REFUSES. It cannot make a unit PASS — the driver's own preflight
  # (restyle_transfer25.preflight_source_masks) still runs the real masker over the real frames and
  # is still what decides, and admitting a unit here only means that nothing on disk predicts a
  # refusal. Nor may it pick a different unit: which episode PR-08 §8 item 3's "one timed episode"
  # is remains the project owner's decision under a registered rule, exactly as the failure block
  # below already says, and a silent re-pick here would be this job answering that question by
  # choosing the first episode that survives — a selection made after seeing the data. So it
  # refuses, it names the episode and the evidence it read, and it stops.
  # THE EVIDENCE IS THE SHARDS, NOT A POOLED FILE, and that is the fix for job 190981.
  # 190981 asked for an H200 and died six seconds later because POOLED.json is not on the cluster.
  # It is not on the cluster because NO COMMITTED SCRIPT IN THIS REPOSITORY WRITES IT: it is a
  # hand-built file with six keys, no schema and no merge verdicts, which until 0c2b1eb existed on
  # exactly one workstation under gitignored runs/. `106 MERGE=1` does not write it either — that
  # merge DROPS per_episode
  # and emits the five summary numbers and the bound, which cannot answer a question about one
  # episode. So the old default named a file whose producer did not exist.
  #
  # The sixteen wam.robot_mask_area_shard/1 artifacts it was pooled from ARE on the cluster, carry a
  # schema, their own measurement_qualified, the partition and the raw per-frame fractions, and
  # scripts/pool_robot_mask_area.py rebuilds the pool from them. Measured 2026-08-28: that rebuild
  # is IDENTICAL to POOLED.json field for field, every float included, and POOLED.json hashes to the
  # sha256 T40_RULE_V20 §2 registered. Reading the shards is therefore the registered evidence
  # reconstructed from its own inputs, not a different measurement and not a rule change.
  #
  # A pooled FILE is still accepted, so pointing this at POOLED.json keeps working unchanged.
  MASK_AREA_EVIDENCE=${MASK_AREA_EVIDENCE:-${PROJ}/runs/pr08-robot-mask-area/shards}
  # The same artifact robot_composite.load_area_bound reads by default (AREA_BOUND_ARTIFACT), named
  # here rather than imported: this screen must not need torch, transformers or a GPU to run.
  AREA_BOUND_ARTIFACT=${AREA_BOUND_ARTIFACT:-${WAM}/configs/transfer25/pr08_robot_mask_area.json}
  TIMING_ADMISSIBILITY=${CHUNK_DIR}/timing_unit_admissibility.json
  rm -f "${TIMING_ADMISSIBILITY}"
  python - "${TIMING_UNIT}" "${MASK_AREA_EVIDENCE}" "${SOURCE}/manifest.json" \
           "${AREA_BOUND_ARTIFACT}" "${TIMING_ADMISSIBILITY}" "${WAM}" <<'PY' || exit 1
import hashlib, json, pathlib, sys

unit_json, pooled_p, manifest_p, bound_p, out_p, wam = sys.argv[1:7]
# STDLIB ONLY, still. pool_robot_mask_area imports nothing this screen could not already import;
# it is a separate module rather than inlined here so that the sbatch, the submit wrapper and the
# tail-sheet renderer answer "which evidence said so" with one implementation instead of three.
sys.path.insert(0, str(pathlib.Path(wam) / "scripts"))
from pool_robot_mask_area import PoolError, load_area_evidence
u = json.loads(unit_json)
episode, frames = str(u["episode"]), int(u["frames"])
SELECTION_RULE = ("head -1 of the work list, which is the manifest's episodes sorted by id, "
                  "expanded over the staged styles and their repeat span (T40_RULE_V11 §2). "
                  "Deterministic: the same partition and the same manifest name the same unit.")


def refuse(headline, *lines):
    """Every refusal on this path is free, and every one of them names what it read."""
    print(f"FATAL: {headline}")
    for line in lines:
        print(f"       {line}")
    print("       Decided from evidence already on disk, before the driver started and before the")
    print(f"       masker was loaded: {pooled_p}")
    print("       The H200 is ALREADY allocated here (this file's own --gres, no srun), so no queue")
    print("       slot is saved. Nothing was generated and no frame was segmented.")
    sys.exit(1)


# The evidence itself, before anything is asked of it. Absent or unqualified is a REFUSAL and not a
# skip: this screen costs nothing, so there is no case in which proceeding without it is the
# cheaper option, and "we could not check" is not a reason to run a masker over every source frame
# of a unit whose G0c verdict is already written down somewhere.
try:
    pooled = load_area_evidence(pooled_p)
except PoolError as exc:
    refuse(f"the source-mask evidence {pooled_p} could not be read or pooled.",
           *str(exc).splitlines(),
           "PR-08 §8 item 3's timed episode has to be one G0c will let through, and G0c's verdict",
           "on an episode is a function of its SOURCE masks. That distribution is measured by",
           "cluster/discoverer/106_measure_robot_mask_area.sbatch, whose ARRAY writes the sixteen",
           "shard artifacts this reads. NOTE: MERGE=1 does NOT write a per-episode pool — it drops",
           "per_episode and emits the summary and the bound — so re-running the merge does not fix",
           "this. Point MASK_AREA_EVIDENCE at the shards directory, or run the 106 array first.",
           "This job does not time an episode whose refusal it cannot rule out in advance.")
if pooled.get("measurement_qualified") is not True:
    refuse(f"{pooled_p} says measurement_qualified={pooled.get('measurement_qualified')!r}.",
           "Its own merge refused to vouch for it — see measurement_disqualified_reasons:",
           f"{pooled.get('measurement_disqualified_reasons')!r}.",
           "A screen built on a disqualified distribution is a screen that proves nothing.")
# A distribution is a claim ABOUT a corpus, which is why load_area_bound cross-checks the same
# sha256 before it will composite. Applied to other episodes these fractions are numbers with no
# frames behind them, and the episode ids would silently line up anyway.
try:
    want_sha = hashlib.sha256(pathlib.Path(manifest_p).read_bytes()).hexdigest()
except OSError as exc:
    refuse(f"the source manifest {manifest_p} could not be read ({exc}).",
           "Without it the evidence cannot be tied to the corpus this run would restyle.")
got_sha = str(pooled.get("source_manifest_sha256"))
if got_sha != want_sha:
    refuse("the source-mask evidence measured a DIFFERENT corpus.",
           f"evidence source_manifest_sha256: {got_sha}",
           f"{manifest_p}: {want_sha}",
           "Episode ids are reused across trees (the AV1 and the H.264 corpora share them), so",
           "this cannot be waved through on the ids matching.")

by_id = {str(e.get("episode")): e for e in pooled["per_episode"] if isinstance(e, dict)}
ep = by_id.get(episode)
if ep is None:
    refuse(f"{episode} is not in {pooled_p}.",
           f"The measurement covers {len(by_id)} episodes and this is not one of them, so there is",
           "no basis on which to predict G0c's verdict for the unit this job is about to time.")
n_frames = int(ep.get("n_frames", -1))
if n_frames != frames:
    refuse(f"{episode} is {frames} frames in the manifest and {n_frames} in the evidence.",
           "The two are about different footage. G0c refuses on ONE bad frame, so a screen over a",
           "different frame count is not a screen over the frames this job would generate.")
fractions = ep.get("area_fractions")
if not isinstance(fractions, list) or len(fractions) != n_frames:
    refuse(f"{episode}'s entry carries {len(fractions) if isinstance(fractions, list) else 'no'} "
           f"area fractions for {n_frames} frames.",
           "Every frame's fraction, or the maximum below is a maximum over an unknown subset.")

# THE EMPTY HALF, which is the one that fires here. check_mask raises on covered == 0 with no
# threshold in it, so any zero in this list is a clip-fatal refusal waiting to be paid for.
empty = [i for i, f in enumerate(fractions) if not float(f) > 0.0]
declared_empty = ep.get("empty_frames")
if isinstance(declared_empty, int) and declared_empty != len(empty):
    refuse(f"{episode}'s entry says empty_frames={declared_empty} but {len(empty)} of its "
           f"{n_frames} fractions are zero.",
           "The artifact contradicts itself about the one property this screen turns on.")
if empty:
    refuse(f"the timed unit {u['unit']} WILL be refused by PR-08 §6 G0c, before it produces a clip.",
           f"{episode}'s robot mask is EMPTY on {len(empty)} of its {n_frames} source frames, "
           f"first at frame {empty[0]}.",
           "robot_composite.check_mask refuses a clip on one empty frame and there is no number in",
           "it to loosen: an empty mask means the GENERATED manipulator went into the corpus.",
           "",
           "THIS IS NOT A RETRY AND THIS JOB DOES NOT PICK ANOTHER EPISODE. The mask is a property",
           "of the SOURCE frame, so this unit refuses identically on every resubmission; and",
           "choosing a different one because it survives G0c is a selection made after seeing the",
           "data. Two questions are open and both are the project owner's:",
           "  * whether an empty robot mask is acceptable for a timed measurement at all",
           "  * which episode PR-08 §8 item 3's 'one timed episode' is",
           "Both need a registered rule version. CHUNK_INDEX/CHUNK_TOTAL already reach any episode",
           "without a code change, which is exactly why the choice has to be written down rather",
           "than typed on a submit line.",
           "",
           f"For that decision, and as evidence rather than as a shortlist: {len(by_id)} episodes "
           f"were measured, {sum(1 for e in by_id.values() if e.get('empty_frames') == 0)} of them",
           "have no empty frame at all.")

# THE AREA HALF. It needs the committed bound, and a bound that has not been decided is not a
# looser screen, it is no screen — so this refuses rather than checking one half. The driver would
# refuse on the same missing bound anyway, later, with a GPU already allocated.
try:
    bound_doc = json.loads(pathlib.Path(bound_p).read_text())
    bound = bound_doc.get("max_frame_fraction")
except (OSError, ValueError) as exc:
    refuse(f"the committed area bound {bound_p} could not be read ({exc}).",
           "PR-08 §6 G0c has two halves and this screen may not run only one of them.")
if not isinstance(bound, (int, float)) or isinstance(bound, bool):
    refuse(f"{bound_p} has max_frame_fraction={bound!r}.",
           "The artifact's own bound_note: it measures the distribution, it does not choose the",
           "bound. Until a number is committed there with a rationale, G0c's area half is unarmed",
           "and this job cannot tell whether the unit it would time is admissible.")
worst = max(float(f) for f in fractions)
if worst > float(bound):
    refuse(f"the timed unit {u['unit']} WILL be refused by PR-08 §6 G0c's area half.",
           f"{episode}'s largest robot-mask fraction is {worst!r}, above the committed bound "
           f"{bound!r} ({bound_p}).",
           "A mask that large has grounded on the table or the whole scene; compositing it copies",
           "the source back over the frame and the restyle becomes a no-op.",
           "As above: this job refuses and does not choose another episode.")

if pathlib.Path(pooled_p).is_dir():
    _evidence_shards = [
        {"path": part["path"], "index": part["index"], "sha256": part["sha256"]}
        for part in pooled.get("pooled_from") or ()
    ]
    _evidence_digest = hashlib.sha256(
        "".join(part["sha256"] for part in _evidence_shards).encode("utf-8")
    ).hexdigest()
else:
    _evidence_shards = None
    _evidence_digest = hashlib.sha256(pathlib.Path(pooled_p).read_bytes()).hexdigest()

record = {
    "schema": "wam.transfer25_timing_unit_admissibility/1",
    "unit": u["unit"], "episode": episode, "style": u.get("style"), "n_frames": n_frames,
    "empty_frames": 0, "max_area_fraction": worst, "max_frame_fraction_bound": float(bound),
    "evidence": str(pooled_p),
    # A DIRECTORY HAS NO BYTES, and hashing one raises IsADirectoryError. When the evidence is the
    # shards, what identifies it is the sixteen shard artifacts by name and hash, and the digest
    # below is taken over those hashes in index order — so the field still answers "exactly which
    # evidence said so" and still changes if any shard does. `evidence_shards` carries the parts.
    "evidence_sha256": _evidence_digest,
    "evidence_shards": _evidence_shards,
    "evidence_source_manifest_sha256": want_sha,
    "bound_artifact": str(bound_p),
    "bound_artifact_sha256": hashlib.sha256(pathlib.Path(bound_p).read_bytes()).hexdigest(),
    "selection_rule": SELECTION_RULE,
    "screen": ("PR-08 §6 G0c evaluated over this episode's SOURCE mask area fractions before the "
               "generator ran. It PREDICTS that G0c will not refuse this unit; it does not "
               "replace the driver's own preflight, which recomputes the masks and decides."),
    "not_a_pass": ("Admissible is not passed. This is a PREDICTION, from an earlier run of the "
                   "same masker over the same source frames, and the bound's own rationale "
                   "records that per-frame classification near the bound is hardware-dependent "
                   "for one narrow band. So this screen and the driver's preflight can disagree "
                   "in either direction; where they do, this one refuses before the driver "
                   "starts and before the masker is loaded — the allocation is already held by "
                   "then, so what it saves is a segmentation pass, not a queue slot — and the "
                   "preflight is what decides after."),
}
pathlib.Path(out_p).write_text(json.dumps(record, indent=2) + "\n")
print(f"=== timed unit screened against {pooled_p}: {episode} has 0 empty source frames and a "
      f"largest mask fraction of {worst:.6f} <= {bound}. G0c is not expected to refuse it.")
PY
  TIMING_LIST=${CHUNK_DIR}/timing.jsonl
  printf '%s\n' "${TIMING_UNIT}" > "${TIMING_LIST}"
  TROOT=${CHUNK_DIR}/timing_raw
  # Throwaway output root, rm -rf'd before and after: a stray clip left where the real run's
  # harvest looks would be filed as finished work and the episode would never be regenerated.
  rm -rf "${TROOT}"; mkdir -p "${TROOT}"
  echo "=== timing one episode at 640x480 on 1 GPU: ${TIMING_UNIT}"
  S=${SECONDS}
  # --no-guardrails: see the note on the real invocation below. It is mandatory here too, and for
  # the timing number specifically it matters twice over — the guardrail pass is real GPU time
  # that the production run will not spend, so timing with it on measures a pipeline we do not run.
  # --require-success: the chunked run below deliberately exits 0 on a dead unit so the chunk stays
  # resumable, and until 2026-08-21 this path inherited that. It must not. Job 189142 crashed inside
  # the driver on a SetupArguments validation error, produced 0 clips, exited 0 -- and the block
  # below turned the 118 s it took to reach the crash into "9.56 GPU-h per variant" and wrote it to
  # THROUGHPUT.json as the input to the PR-08 §8 item 3 budget. The clock measures the crash unless
  # something asserts the clip exists, so the driver asserts it and this exits here if it does not.
  #
  # AND THE NUMBER THIS MEASURES NOW INCLUDES PR-08 §6 G0c. The composite is inside the driver, so a
  # timed episode pays for decoding the source, segmenting the robot in every source frame, the
  # composite itself and the re-encode — exactly what the generation run pays. That is the point of
  # measuring here rather than around the model call: a throughput number that excluded the gate
  # would under-derive the GPU-h ceiling the generation path is then held to. The source mask is
  # cached per episode and reused across the 25 restyles of it, so the marginal cost of a SECOND
  # variant of the same episode is lower than this first one — the ceiling derived from here is
  # therefore conservative, which is the correct direction for a budget.
  #
  # THIS MAKES THE TIMING PATH DEPEND ON G0c's OWN PREREQUISITES, and that is deliberate. The driver
  # refuses to start without staged GroundingDINO + SAM 2 checkpoints (102_stage_sam2_weights.sbatch)
  # and a committed configs/transfer25/pr08_robot_mask_area.json, so the order is: stage the
  # weights, run `scripts/robot_composite.py measure` over the SOURCE corpus, decide and commit the
  # area bound, then time. PR-08 §1 licenses this measurement unconditionally and nothing here
  # withdraws that licence — it withdraws the option of measuring a DIFFERENT pipeline. Timing the
  # generator without the composite would be the same defect as timing it with the guardrail on, in
  # the opposite direction: a wall clock around a pipeline we do not run, feeding a GPU-h ceiling
  # the pipeline we DO run is then held to. Sub-section (c) above refuses budgets nobody measured;
  # this refuses budgets measured on something else.
  python "${RESTYLE_DRIVER}" \
      --checkpoint-path "${TRANSFER_CHECKPOINT_PATH}" \
      --no-guardrails \
      --require-success \
      --manifest "${SOURCE}/manifest.json" \
      --styles "${STYLE_PARTITION}" \
      --style-set "${STYLE_SET}" \
      --work-list "${TIMING_LIST}" \
      --out "${TROOT}" \
      --resolution 640x480 \
      --control "${CONTROL}" \
      --nproc 1 || {
    echo "FATAL: the timed episode did not produce a clip, so there is no measurement to record."
    echo "       THROUGHPUT.json is deliberately NOT written -- a wall clock around a failed unit"
    echo "       is not a throughput number, and PR-08 §8 item 3 derives the budget from this file."
    echo "       Per-unit reason: ${TROOT}/*/sample_outputs.json"
    # If G0c refused the unit on its SOURCE masks, the driver's preflight caught it before the
    # generator ran and this failure cost seconds rather than an H200 slot. Say so, and say what it
    # is evidence OF — because the operator's next move is not to retry, and it is not a decision
    # this job may make for them. The timed episode is fixed by `head -1` of a deterministically
    # sorted work list, and 385 of the corpus's 402 episodes are refused by one half of check_mask
    # or the other (runs/pr08-robot-mask-area/POOLED.json). WHICH episode PR-08 §8 item 3's
    # measurement runs on, and whether an empty source mask is acceptable at all, are open owner
    # decisions. This job refuses; it does not choose.
    python - "${TROOT}" <<'PY' || true
import json, pathlib, sys
refused = []
for rec in sorted(pathlib.Path(sys.argv[1]).glob("*/sample_outputs.json")):
    try:
        payload = json.loads(rec.read_text())
    except (OSError, ValueError):
        continue
    if (payload.get("g0c") or {}).get("refused_before_generation") is True:
        refused.append(payload)
if not refused:
    sys.exit(0)
print()
print("       G0c REFUSED THIS UNIT ON ITS SOURCE MASKS, BEFORE THE GENERATOR RAN.")
print("       Nothing was generated and no GPU-hours were spent on a clip; the refusal is the same")
print("       one the composite would have raised afterwards, discovered where it is free. The")
print("       episode is:", refused[0].get("episode"), "-", refused[0].get("unit"))
print()
print("       THIS IS NOT A RETRY. The robot mask is a property of the SOURCE frame, so the same")
print("       episode will refuse identically on every resubmission. Two questions are open and")
print("       both are the project owner's, not this job's:")
print("         * whether an empty robot mask is acceptable for a timed measurement at all")
print("         * which episode PR-08 §8 item 3's 'one timed episode' is, given that the work")
print("           list's first row is refused and choosing another because it survives G0c is a")
print("           selection made after seeing the data")
print("       Both need a registered rule version. CHUNK_INDEX/CHUNK_TOTAL can already reach any")
print("       episode without a code change, which is exactly why the choice must be written down")
print("       rather than typed on a submit line.")
PY
    exit 1
  }
  TIMING_WALL=$((SECONDS - S))
  echo "=== timed episode: ${TIMING_WALL}s"

  python - "${TIMING_UNIT}" "${TIMING_WALL}" "${THROUGHPUT}" "${TRANSFER_MODEL_ID}" \
           "${TRANSFER_MODEL_REVISION}" "${CEILING_GPU_H}" "${CONTROL}" \
           "${CORPUS_FRAMES}" "${TROOT}" "${TIMING_ADMISSIBILITY}" <<'PY'
import json, pathlib, sys
unit, wall, out, model, rev, ceiling, control, corpus_frames, troot = sys.argv[1:10]
# The pre-GPU screen's record, tenth and optional. Optional because of WHERE this block sits: the
# GPU-hours are already spent by the time it runs, and refusing to write the artifact here would
# throw away a paid measurement to punish a missing note about it. The place to refuse is the
# screen itself, which runs before the driver starts and exits 1. So this records what the
# screen found, or records that it found nothing — either way the artifact says which.
admissibility = sys.argv[10] if len(sys.argv) > 10 else ""
u = json.loads(unit)
wall = float(wall)
# The ceiling is EXEMPT on this path (header (c)): PR-08 §8 item 3 says no budget line exists
# until this measurement does, so there is normally nothing to record here. An empty string is
# recorded as null rather than as 0.0 — 0.0 is a budget, and a budget of zero is a claim.
ceiling = float(ceiling) if ceiling.strip() else None
frames = int(u["frames"])
if frames <= 0:
    sys.exit("FATAL: the timed episode declares 0 frames — the manifest is wrong, not the clock.")
if wall <= 0:
    sys.exit("FATAL: the timed episode took 0s. Nothing was generated; that is not a measurement.")
sec_per_frame = wall / frames
# WHAT THE TIMED UNIT ACTUALLY PRODUCED, counted off the driver's own per-unit records rather than
# inferred from an exit status. --require-success above already exits 1 when a unit dies, so this is
# the second of two independent assertions that a clip exists — and it is the one that lands IN the
# artifact, which is what matters. Job 189142 wrote a THROUGHPUT.json whose every field was
# well-formed from a run its own log describes as "0 success, 1 error"; nothing inside that file
# could contradict it, so every later reader had to know the job number to know the number was
# fiction. This count is what makes the artifact self-describing, and the reuse check at the top of
# this branch refuses any THROUGHPUT.json that does not carry it.
records = sorted(pathlib.Path(troot).glob("*/sample_outputs.json"))
succeeded = []
for rec in records:
    try:
        payload = json.loads(rec.read_text())
    except (OSError, ValueError) as exc:
        sys.exit(f"FATAL: {rec} is unreadable ({exc}) — the timed unit's own record is the evidence "
                 "that a clip was produced, so an unreadable one is not a measurement.")
    if payload.get("status") == "success":
        succeeded.append(payload)
if len(succeeded) != 1:
    sys.exit(f"FATAL: the timed run left {len(succeeded)} successful unit record(s) in {troot} out "
             f"of {len(records)}. PR-08 §8 item 3 times ONE episode; a clock around zero clips is "
             "not a throughput number and a clock around two is not the number it asks for. "
             "THROUGHPUT.json is deliberately not written.")
# GPU-seconds, not wall-seconds. The measurement is on ONE GPU so the two coincide here, and that
# is exactly why PR-08 asks for one H200: it makes the number directly comparable to the per-GPU
# billing rate. At NPROC>1 the wall drops but the GPU-seconds do not, so a projection built on
# this number is a LOWER BOUND for any multi-GPU run and is recorded as such.
#
# FRAMES_PER_VARIANT IS READ, NOT COINED. It was `172_000  # 402 episodes x ~427 mean frames` until
# 2026-08-27 — a rounded number typed into the one artifact PR-08 §8 item 3 forbids inventing
# numbers in, while the exact figure was already sitting two hundred lines up: partition_facts.json
# carries corpus_frames, summed from the manifest the run actually reads (see the expansion above),
# and the corpus is 171 625 frames, not 172 000. The gap is small (0.2 %) and its size is not the
# argument; the argument is that a budget derived from a measured rate must be derived over a
# counted corpus, or the derivation cannot be checked against anything.
FRAMES_PER_VARIANT = int(corpus_frames)
if FRAMES_PER_VARIANT <= 0:
    sys.exit("FATAL: partition_facts.json reports corpus_frames <= 0. The projection below is a "
             "rate times a frame count; without the count there is no projection.")
gpu_h_per_variant = sec_per_frame * FRAMES_PER_VARIANT / 3600.0
# WHICH UNIT THIS IS A MEASUREMENT OF, AND WHY THAT UNIT. The screen above wrote this record before
# the generator ran; it names the episode, the selection rule that reached it, and the two hashes
# (the pooled source-mask evidence and the committed area bound) that said G0c would not refuse it.
# It travels INTO the artifact because a seconds-per-frame prices 4.29M frames, and a reader who
# can say which episode produced the number but not how that episode came to be the one cannot
# tell a registered selection from a lucky one. `null` here is therefore not a shrug: it says the
# screen left no record, which on this path means it was removed or its output was deleted.
admitted, admitted_note = None, None
try:
    admitted = json.loads(pathlib.Path(admissibility).read_text())
except (OSError, ValueError) as exc:
    admitted_note = (f"NOT RECORDED: {admissibility or '(no path supplied)'} could not be read "
                     f"({exc}). The pre-GPU screen at the top of the TIMING branch writes it and "
                     "exits 1 when the timed unit is one PR-08 §6 G0c will refuse, so a missing "
                     "record means that screen did not run for this measurement — the number "
                     "below is still a measured wall clock, but nothing says the unit it timed "
                     "was screened.")
report = {
    # Named so a reader — and the reuse check at the top of this branch — can tell an artifact that
    # records whether a clip was produced from one that predates the question.
    "schema": "wam.transfer25_throughput/1",
    "measured_on": "1 x H200, 640x480, one episode",
    "units_timed": len(records),
    "units_succeeded": len(succeeded),
    "episode": u["episode"], "style": u["style"], "frames": frames,
    "timed_unit": u["unit"],
    "timed_unit_admissibility": admitted,
    "timed_unit_admissibility_note": admitted_note,
    "wall_seconds": round(wall, 1),
    "seconds_per_frame": round(sec_per_frame, 4),
    "gpu_seconds_per_frame": round(sec_per_frame, 4),
    "frames_per_variant": FRAMES_PER_VARIANT,
    "frames_per_variant_source": (
        "partition_facts.json corpus_frames — summed over this manifest's episodes at "
        "expansion time. Not an estimate and not a round number."),
    "gpu_hours_per_variant": round(gpu_h_per_variant, 2),
    "gpu_hours_per_variant_is_lower_bound_above_1_gpu": True,
    "generator": f"{model}@{rev}",
    # The controls are part of the measurement, not context for it. The source manifest carries no
    # depth or segmentation maps, so every control block named here is one Transfer2.5 ESTIMATED
    # with its own model on this GPU -- real seconds, in this number. A projection built from a
    # timing under one control spec and spent under another is not a projection of that run.
    "control": control,
    "ceiling_gpu_hours_supplied_at_measurement_time": ceiling,
    "ceiling_gpu_hours_supplied_at_measurement_time_note": (
        "null is the expected value. PR-08 §8 item 3: no budget line exists until this "
        "measurement does, so TIMING=1 asks for no ceiling and ignores one if supplied."),
}
print(json.dumps(report, indent=2))
open(out, "w").write(json.dumps(report, indent=2) + "\n")
print()
print(f"ONE variant of the full corpus projects to {gpu_h_per_variant:.1f} GPU-h.")
print("That number is the input to the ceiling, NOT the ceiling. PR-08 §8 item 3: derive the")
print("budget line from it over the WHOLE partition (T40_RULE_V2 §3 step 2 — all 25")
print("style-instances, 10050 clips), record the derivation, and pass it back as")
print("PARTITION_CEILING_GPU_H. Then split it across the three STYLE_SET submissions")
print("(§3 step 3) and pass each one its share as CEILING_GPU_H; the sum of the shares is")
print("checked against PARTITION_CEILING_GPU_H at the gate.")
PY
  rm -rf "${TROOT}"
  echo "next: read ${THROUGHPUT}, derive the ceiling, then submit the chunks without TIMING=1."
  exit 0
fi

# ------------------------------------------------------------------------------------------------
# GENERATE
# ------------------------------------------------------------------------------------------------
[[ -f "${THROUGHPUT}" ]] || {
  echo "FATAL: no ${THROUGHPUT}. PR-08 §8 item 3 makes the measurement a gate, not a warm-up:"
  echo "       \"No budget line exists until that measurement does.\" Without a measured"
  echo "       seconds-per-frame the GPU-h ceiling cannot be checked BEFORE it is spent."
  echo "       Run: TIMING=1 CHUNK_INDEX=1 CHUNK_TOTAL=1 ... sbatch 97_transfer25_restyle.sbatch"
  exit 1
}
# EXISTING IS NOT THE SAME AS USABLE, AND THIS IS THE PATH THAT SPENDS THE ALLOCATION. The ceiling
# gate below reads exactly one field out of this file — seconds_per_frame — and multiplies it by
# whole_partition_frames to decide whether a chunk may run. Job 189142's artifact would answer 0.2,
# from a run that produced no clip, and every projection built on it comes out ~10x under: the gate
# passes, the pass rail is sized against a fiction, and the ceiling T40_RULE_V2 §3 asks for is a
# number nobody measured. So the same qualification the TIMING path applies before REUSING a
# measurement is applied here before DERIVING from one. There is no override: a ceiling is not a
# thing to be overridden on a submit line, and re-measuring costs ~0.5 GPU-h against a partition
# priced in the thousands.
TP_VERDICT=0
throughput_qualification "${THROUGHPUT}" || TP_VERDICT=$?
if (( TP_VERDICT != 0 )); then
  cat "${THROUGHPUT}"
  echo "FATAL: ${THROUGHPUT} exists but cannot be built on (see above), and the GPU-h ceiling this"
  echo "       job enforces is derived from its seconds_per_frame. Generating against it would"
  echo "       spend the allocation under a budget line that measures nothing."
  echo "       Re-measure: TIMING=1 STAGE=1 STYLE_SET=<set> CHUNK_INDEX=1 CHUNK_TOTAL=1 \\"
  echo "                   CONTROL=<key:weight,...> RUN_ID=<fresh> sbatch 97_transfer25_restyle.sbatch"
  exit 1
fi
cat "${THROUGHPUT}"

# -- PR-08 §6 G0b: the geometry budget has to be positive BEFORE anything is generated ------------
# "GEOM_TOL - EST_DRIFT_P95 ... if that is <= 0, the estimator is not good enough and generation
# does not start." That is a precondition, not a post-hoc gate, and §4 says the estimator is
# characterised "before generation, never after" — so it is enforced here, where refusing is free,
# rather than in the scoring job, where the GPU-h are already spent and unrecoverable.
# GEOM_STEP_FRAMES: REQUIRED, NO DEFAULT — see (f) in the header. It used to default to 1, and a
# default that linearly rescales the tolerance the gate is evaluated at is a coined threshold, not
# a convenience. PR-08 §6 opens "No threshold is coined"; CEILING_GPU_H is required-with-no-default
# for exactly this reason and this is the same defect. The word below is free of ' and " for the
# ${VAR:?word} parsing reason noted at the T-39 attestation.
: "${GEOM_STEP_FRAMES:?PR-08 §6 opens: No threshold is coined. GEOM_TOL is the median per-step object-centroid displacement, so it scales ~LINEARLY with what a step is taken to be, and §6 never defines one — a default of 1 here would therefore not pick a convention, it would pick the G0b tolerance, silently, by a factor of the step. There is deliberately NO default, the same discipline CEILING_GPU_H has. The step is a REGISTERED quantity and not an operator preference: the V3 in progress registers it against AppleToPlate-s actual action rate, and until it does, refusing is the correct outcome. Set GEOM_STEP_FRAMES to the registered step; the committed artifact-s own step_frames must equal it.}"
[[ "${GEOM_STEP_FRAMES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: GEOM_STEP_FRAMES=${GEOM_STEP_FRAMES} is not a positive integer. It is a number of"
  echo "       source frames per G0b step, and GEOM_TOL scales ~linearly with it."
  exit 1; }
[[ -f "${GEOM_CONSTANTS}" ]] || {
  echo "FATAL: ${GEOM_CONSTANTS} missing. PR-08 §8 item 4 requires GEOM_TOL and EST_DRIFT_P95"
  echo "       measured and committed before a single clip is generated. Produce it with"
  echo "       scripts/measure_geom_tol.py, whose default --out is the tracked"
  echo "       configs/transfer25/pr08_geom_tol.json (+ .sha256). Do NOT point this at a file"
  echo "       under runs/ — that tree is gitignored and cannot hold a pre-commitment."
  exit 1; }
# The sidecar, same discipline as the style partition: the number this job gates on has to be the
# number that was committed, and "sha256sum -c" is what makes that checkable rather than assumed.
#
# ITS ABSENCE IS FATAL, NOT A WARNING (changed 2026-08-15). A WARN here made the sidecar optional,
# and an optional proof is not a proof: the one thing an operator would do to get a hand-written
# GEOM_TOL past this gate is delete the sidecar, and the job would have printed a line nobody reads
# and generated 4 020 clips against a tolerance nobody measured. §8 item 4 wants GEOM_TOL COMMITTED
# before generation; a file with no committed digest cannot have been.
[[ -f "${GEOM_CONSTANTS}.sha256" ]] || {
  echo "FATAL: ${GEOM_CONSTANTS}.sha256 missing — nothing proves this is the committed artifact."
  echo "       scripts/measure_geom_tol.py writes the sidecar next to the artifact on every run."
  echo "       A GEOM_TOL with no committed digest is a number, not a pre-commitment."
  exit 1; }
GEOM_SHA=$(sha256sum "${GEOM_CONSTANTS}" | cut -d' ' -f1)
[[ "${GEOM_SHA}" == "$(tr -d '[:space:]' < "${GEOM_CONSTANTS}.sha256")" ]] || {
  echo "FATAL: ${GEOM_CONSTANTS} does not match its committed .sha256 sidecar."
  echo "       on disk  ${GEOM_SHA}"
  echo "       sidecar  $(tr -d '[:space:]' < "${GEOM_CONSTANTS}.sha256")"
  echo "       GEOM_TOL is a pre-commitment; an artifact edited after it was committed is not"
  echo "       one. Re-measure, do not re-hash."
  exit 1; }
python - "${GEOM_CONSTANTS}" "${GEOM_STEP_FRAMES}" "${SOURCE}/manifest.json" <<'PY'
import hashlib, json, pathlib, sys
g = json.loads(pathlib.Path(sys.argv[1]).read_text())
want_step = int(sys.argv[2])
manifest_p = pathlib.Path(sys.argv[3])
tol = g.get("GEOM_TOL_px", g.get("GEOM_TOL"))
drift = g.get("est_drift_p95_px", g.get("EST_DRIFT_P95"))
if tol is None:
    sys.exit(f"FATAL: {sys.argv[1]} carries no GEOM_TOL. G0b has no tolerance to hold the "
             "generator to, so nothing generated could be gated.")
# gate_qualified must be PRESENT and TRUE. It used to be read as `g.get("gate_qualified", True)`,
# which is default-permissive in the one direction that matters: an artifact that never made the
# claim — a hand-written file, or one produced before the field existed — passed G0b by saying
# nothing, and saying nothing is exactly what a fabricated artifact does. measure_geom_tol.py
# applies the same rule to its own inputs ("An unstated claim is not a claim: gate_qualified
# defaults to False"), so this is the consumer half of a discipline the producer already keeps.
if "gate_qualified" not in g:
    sys.exit(f"FATAL: {sys.argv[1]} does not record gate_qualified at all. The artifact never "
             "claimed to be usable as a gate, so it is not one — an absent claim is not a pass. "
             "Re-measure with the current scripts/measure_geom_tol.py.")
if g.get("gate_qualified") is not True:
    sys.exit(f"FATAL: {sys.argv[1]} says gate_qualified={g.get('gate_qualified')!r}, so GEOM_TOL "
             "is not usable as a gate. Reasons the measurement itself recorded:\n         "
             + "\n         ".join(g.get("gate_disqualified_reasons")
                                  or ["(none recorded — an older artifact)"])
             + "\n       Fix the measurement first.")
# THE TOLERANCE HAS TO BE A TOLERANCE FOR *THIS* CORPUS, and until 2026-09-02 nothing here asked.
# The two checks above prove the artifact is the committed one and that its own merge vouched for
# it. Neither says it was measured over the tree this job is about to restyle. That gap was found
# by the 2026-08-28 audit and recorded, deliberately unfixed, in
# docs/investigations/2026-08-28-two-generate-path-gaps-recorded-not-fixed.md §1 — the fix had to
# wait until pr08-geom-tol-v2 had merged, because adding a field to a shard shape mid-array is how
# a merge comes to refuse its own shards.
#
# WHY THE DIGEST AND NOT THE EPISODE IDS. This is the identical check
# robot_composite.load_area_bound already makes for the mask-area bound, in its own words: "a bound
# is a statement about a corpus, and holding corpus A's distribution over corpus B is the same
# drift by another route." The AV1 and the H.264 apple trees enumerate the SAME episode ids from
# zero, so matching ids prove nothing; the manifest's bytes are the identity. measure_geom_tol.py
# --bind-source-manifest is what writes the field, and it refuses to write it unless the manifest
# and the measurement agree on every episode id, every per-episode frame count, the pixel grid and
# the fps — so a digest that reaches this line was checked when it was recorded, not asserted.
#
# ABSENT REFUSES, for the same reason gate_qualified absent refuses: an artifact that never named a
# corpus is not an artifact that named this one.
want_corpus = g.get("source_manifest_sha256")
if not want_corpus:
    sys.exit(f"FATAL: {sys.argv[1]} names no corpus (source_manifest_sha256 is "
             f"{want_corpus!r}), so nothing can say whether its GEOM_TOL was measured over the "
             f"tree this job restyles.\n       A tolerance is a statement ABOUT a corpus. Bind it:"
             "\n         .venv/bin/python scripts/measure_geom_tol.py --bind-source-manifest "
             "<SOURCE>/manifest.json\n       That mode checks the episode ids, the per-episode "
             "frame counts, the pixel grid and the fps before it\n       records anything, and "
             "refuses if they disagree. An unstated claim is not a claim.")
try:
    got_corpus = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
except OSError as exc:
    sys.exit(f"FATAL: the SOURCE manifest {manifest_p} could not be read ({exc}), so the committed "
             "GEOM_TOL cannot be tied to the corpus this job would restyle.")
if got_corpus != want_corpus:
    sys.exit(f"FATAL: the committed GEOM_TOL measured a DIFFERENT corpus.\n"
             f"         {sys.argv[1]} source_manifest_sha256: {want_corpus}\n"
             f"         {manifest_p}: {got_corpus}\n"
             "       Episode ids are reused across trees (the AV1 and the H.264 corpora share "
             "them), so this cannot\n       be waved through on the ids matching — that is why "
             "the digest is what is compared. Either point\n       SOURCE at the tree GEOM_TOL "
             "was measured over, or re-measure over this one. Do not re-bind an\n       "
             "unchanged measurement to a new manifest.")
print(f"=== G0b corpus: source_manifest_sha256 {want_corpus[:16]}… matches {manifest_p}")
# A PARTIAL measurement scores coverage over the steps it DECODED, so --limit/--max-frames leave
# coverage at 1.000 and every other field looking finished. Checked explicitly, and belt-and-braces
# against the episode counts, so an artifact from an older script cannot slip through on the flag
# alone.
if g.get("partial_measurement"):
    sys.exit(f"FATAL: {sys.argv[1]} is a PARTIAL measurement (limit={g.get('limit')}, "
             f"max_frames={g.get('max_frames')}) — a smoke test, not the committed number.")
found, done_ = g.get("n_episodes_found"), g.get("n_episodes")
if found is not None and done_ is not None and found != done_:
    sys.exit(f"FATAL: {sys.argv[1]} measured {done_} of {found} episodes. coverage is computed "
             "over the decoded steps only and does not notice this. GEOM_TOL must be measured "
             "over the whole source corpus.")
# "per-step" is undefined in PR-08 §6 and GEOM_TOL scales ~linearly with the reading, so a gate
# quoted against a different step is wrong by that factor and nothing would ever say so.
step = g.get("step_frames")
if step is None:
    sys.exit(f"FATAL: {sys.argv[1]} does not record step_frames, so it cannot be known which "
             "reading of 'per-step' produced GEOM_TOL. Re-measure with the current script.")
if int(step) != want_step:
    sys.exit(f"FATAL: {sys.argv[1]} was measured at step_frames={step} but this job gates at "
             f"GEOM_STEP_FRAMES={want_step}. GEOM_TOL scales ~linearly with the step, so the "
             f"tolerance would be wrong by roughly {step}/{want_step}x. Re-measure at the step "
             "you intend to gate under, or set GEOM_STEP_FRAMES to match and say why.\n"
             f"       step_definition: {g.get('step_definition')}")
# The integer alone does not say what a step IS — overlapping offsets i -> i+n, or disjoint
# windows, are different measurements that both report step_frames=n. The artifact writes the
# prose out; it is required here and carried into chunk_metadata.json so a later reader of a clip
# can tell which reading of "per-step" its tolerance came from without re-deriving it.
if not g.get("step_definition"):
    sys.exit(f"FATAL: {sys.argv[1]} records step_frames={step} but no step_definition, so the "
             "number cannot be interpreted. GEOM_TOL scales with the step and PR-08 §6 does not "
             "define one. Re-measure with the current scripts/measure_geom_tol.py.")
print(f"=== G0b step:   step_frames={step}  {g['step_definition']}")
if drift is None:
    sys.exit(
        "FATAL: EST_DRIFT_P95 is null in " + sys.argv[1] + ".\n"
        "       blocked_by: " + str(g.get("est_drift_p95_blocked_by")) + "\n"
        "       PR-08 §6 G0b holds the generator to GEOM_TOL - EST_DRIFT_P95, and §4 requires the\n"
        "       estimator to be characterised BEFORE generation, never after. Without the drift\n"
        "       number the budget is unknown, G0b cannot be evaluated on anything this job would\n"
        "       write, and the GPU-h would be unrecoverable. This is PR-08 §8 item 5 (the\n"
        "       distance_to_camera / semantic_segmentation annotators in isaac_binding.py)\n"
        "       blocking generation exactly as the document says it does. Do NOT set the drift to\n"
        "       zero to get past this: §4 records it as a LOWER bound on the real error, and a G0b\n"
        "       margin that only clears under a lower bound is not a pass.")
budget = float(tol) - float(drift)
if budget <= 0:
    sys.exit(f"FATAL: GEOM_TOL({tol}) - EST_DRIFT_P95({drift}) = {budget:.3f} <= 0. PR-08 §6: "
             "the estimator is not good enough and generation does not start. That is the "
             "finding — record it. Improving the estimator is the move, not widening the gate.")
print(f"=== G0b budget: GEOM_TOL {tol} px - EST_DRIFT_P95 {drift} px = {budget:.3f} px (> 0)")
PY

# -- the ceiling, the shares, and the pass rail derived from them ---------------------------------
# TWO numbers, and the relation between them is what T40_RULE_V2 §3 actually registers.
# 94's header records the bug this shape avoids: MAX_RESTARTS=2 tested with `>=` permitted two
# passes while the log printed "of 3" and a separate gate allowed 3 — three different budgets out
# of one pre-registered term. So every budget here is derived, never re-typed.
#
# THE HISTORY, BECAUSE BOTH READINGS HAVE BEEN WRONG IN THIS FILE (see (e) in the header).
#
#   Before 2026-08-15  CEILING_GPU_H was compared against ONE style set's projection, and MAX_PASSES
#                      divided it by that one set's chunk count. The same number handed to the
#                      train, eval and identity submissions therefore authorised it three times
#                      over — nearer 4x after T40_RULE_V2 raised arm C from 402 clips to 4 020.
#   Then (same day)    it became "ONE number for the WHOLE partition", gated against
#                      PARTITION_FRAMES. That closed the triple-authorisation and OPENED a worse
#                      hole: T40_RULE_V2 §3 step 3 requires the whole-partition ceiling to be SPLIT
#                      across the STYLE_SET invocations so that "the sum of the CEILING_GPU_H
#                      values actually passed is bounded by the whole-partition figure", and
#                      configs/transfer25/styles.toml [consumer] item 3 says the same. An operator
#                      following the registered rule passes shares — at a 1068.19 GPU-h projection,
#                      427.28 / 213.64 / 427.28 — and ALL THREE submissions fail a gate that
#                      compares each share against 1068.19. The experiment was unlaunchable by
#                      anyone obeying the rule, and V2 is registered and unamendable.
#
# SO BOTH NUMBERS ARE NAMED AND THE RULE'S OWN SENTENCE IS THE GATE:
#
#   1. PARTITION_CEILING_GPU_H (V2 §3 step 2) is pinned run-wide in ${OUT}/run.env. The WHOLE
#      partition's projection — every style of every set, times its repeats, times every episode —
#      must fit it. This is the property the "one number" repair was defending, kept.
#   2. CEILING_GPU_H (V2 §3 step 3) is THIS style set's share. This style set's projection must fit
#      it, and MAX_PASSES is derived from it, so a share too small to buy a pass fails loudly here
#      rather than after the first chunk.
#   3. Each share is pinned per style set in ${OUT}/ceiling_shares/<set>.json, and the SUM of the
#      pinned shares must fit PARTITION_CEILING_GPU_H. That sum is the sentence V2 §3 step 3 is,
#      enforced rather than trusted: three submissions carrying the whole-partition figure as their
#      "share" now fail on the sum, which is the old triple-authorisation caught in the one place
#      that can actually see all three.
#   4. The SPEND RAIL is the persisted run-level ledger under ${OUT}/gpu_ledger/. Every pass of
#      every chunk of every style set reserves its worst case there BEFORE it runs; this style
#      set's reservations may not exceed its share, and the run total may not exceed
#      PARTITION_CEILING_GPU_H. A per-style-set counter cannot see the other two submissions; a
#      shared directory can, and it costs one file per pass.
#
# WALL_H must match --time above. A pass costs AT MOST WALL_H * NPROC GPU-h (Slurm bills the
# allocation for the whole wall, used or not), which is what both the ledger and MAX_PASSES charge.
# MAX_PASSES stays as a per-chunk stop so a single chunk cannot spin through its style set's share
# on its own.
WALL_H=4
GATE=${CHUNK_DIR}/gate.json
LEDGER_DIR=${OUT}/gpu_ledger
SHARES_DIR=${OUT}/ceiling_shares
mkdir -p "${LEDGER_DIR}" "${SHARES_DIR}"

# -- the run-level ceiling stamp, PINNED BEFORE THE GATE READS IT ---------------------------------
# The ledger and the share sum only bound the run if every submission charges against the SAME
# whole-partition ceiling, and that ceiling is a runtime argument nothing downstream could check.
# So the first submission under this RUN_ID pins it and a later one that contradicts it is refused
# rather than silently doubling the budget — the run-level twin of the per-chunk stamp above.
# It is pinned BEFORE the gate now (it used to be stamped after), because the gate's share-sum
# check is meaningless against a figure a later submission could still change.
RUN_ENV=${OUT}/run.env
RUN_STAMP="PARTITION_CEILING_GPU_H=${PARTITION_CEILING_GPU_H} STYLES_CONTENT_SHA=${STYLES_CONTENT_SHA} \
PARTITION_FRAMES=${PARTITION_FRAMES} WALL_H=${WALL_H}"
if [[ -f "${RUN_ENV}" ]]; then
  RUN_PREV=$(cat "${RUN_ENV}")
  if [[ "${RUN_PREV}" != "${RUN_STAMP}" ]]; then
    echo "FATAL: run ${RUN_ID} was started under:  ${RUN_PREV}"
    echo "       this invocation says:             ${RUN_STAMP}"
    echo "       PARTITION_CEILING_GPU_H is ONE number for the whole partition and every style set"
    echo "       charges its share against one ledger, so two submissions under two whole-partition"
    echo "       ceilings is two budgets. Use the ceiling this run was derived with, or start a new"
    echo "       RUN_ID. (The per-style-set SHARE is allowed to differ between submissions and is"
    echo "       pinned separately, in ${SHARES_DIR}/.)"
    exit 1
  fi
else
  printf '%s' "${RUN_STAMP}" > "${RUN_ENV}"
fi

# -- this style set's share, pinned so the sum below is a sum over facts --------------------------
# One file per style set, written on the first submission of that set and refused if a later one
# contradicts it. Without this the "sum of the shares" check would only ever see the shares of the
# submissions that happen to be running, and an operator could raise a share on a resubmit and
# spend past the whole-partition figure one set at a time.
SHARE_FILE=${SHARES_DIR}/${STYLE_SET}.share
if [[ -f "${SHARE_FILE}" ]]; then
  SHARE_PREV=$(tr -d '[:space:]' < "${SHARE_FILE}")
  if [[ "${SHARE_PREV}" != "${CEILING_GPU_H}" ]]; then
    echo "FATAL: STYLE_SET=${STYLE_SET} was first submitted with CEILING_GPU_H=${SHARE_PREV} and"
    echo "       this invocation says ${CEILING_GPU_H}. T40_RULE_V2 §3 step 3 splits ONE"
    echo "       whole-partition ceiling across the style sets; a share that changes between"
    echo "       submissions of the same set is a second split of the same budget."
    echo "       Restore ${SHARE_PREV}, or re-derive the whole split under a new RUN_ID."
    exit 1
  fi
else
  printf '%s' "${CEILING_GPU_H}" > "${SHARE_FILE}"
fi

python - "${THROUGHPUT}" "${CEILING_GPU_H}" "${CHUNK_TOTAL}" "${NPROC}" "${WALL_H}" \
         "${CHUNK_FRAMES}" "${GATE}" "${PARTITION_FACTS}" "${STYLE_SET}" \
         "${PARTITION_CEILING_GPU_H}" "${SHARES_DIR}" <<'PY'
import json, math, pathlib, sys
(tp_p, share, chunks, nproc, wall_h, chunk_frames, gate_p, facts_p, style_set,
 partition_ceiling, shares_dir) = sys.argv[1:12]
tp = json.loads(pathlib.Path(tp_p).read_text())
facts = json.loads(pathlib.Path(facts_p).read_text())
share, chunks, nproc, wall_h = float(share), int(chunks), int(nproc), float(wall_h)
partition_ceiling = float(partition_ceiling)
chunk_frames = int(chunk_frames)
spf = float(tp["seconds_per_frame"])

# NOTE, carried over unchanged from the pre-2026-08-15 formula and flagged rather than silently
# altered: the `* nproc` overstates GPU-hours above one GPU (the wall shortens by roughly the same
# factor the billing widens), so a multi-GPU projection here is conservative. At the file's default
# --gres=gpu:1 it is exact. Fixing it is a budget change and belongs with the throughput
# measurement, not with this repair.
partition_frames = int(facts["whole_partition_frames"])
instances = facts["style_instances_per_set"]
style_sets = len(instances)
projected_chunk = spf * chunk_frames * nproc / 3600.0
# EXACT, not projected_chunk * chunks. The chunking is a stride (eps[idx-1::total]), so at 402
# episodes over 4 chunks the chunks hold 101/101/100/100 and any one of them times 4 misprices the
# set by up to a chunk's imbalance — downward for the short chunks, which is the direction that
# quietly approves a set that does not fit its share. The corpus frame count and this set's
# style-instance count are both in partition_facts.json, so the set can be priced from the corpus
# instead of extrapolated from a sample of it.
# ...and it is the STAGE's instance count, not the whole set's, once T40_RULE_V11 is staging.
# partition_facts.json carries both: `style_instances_per_set` is the committed partition and does
# not move, `stage_instances_per_set` is what this invocation will actually emit. Pricing a stage
# at the whole set's count would refuse a run that is well inside its share — stage 1 draws 4 of
# train's 10 instances — and, worse, would make the share meaningless as a bound on what was spent.
# The fallback to `instances` keeps a partition_facts.json written before the selector existed
# readable; `stage` is recorded beside the number so a reader can tell which was used.
staged_instances = facts.get("stage_instances_per_set") or instances
projected_style_set = (spf * int(facts["corpus_frames"]) * staged_instances[style_set] * nproc / 3600.0)
projected_partition = spf * partition_frames * nproc / 3600.0
# The share buys passes for THIS style set's chunks only — that is what a share is. Dividing the
# whole-partition figure by (style_sets * chunks), as this did while CEILING_GPU_H meant the whole
# partition, silently priced every set at the average set.
max_passes = math.floor(share / (chunks * wall_h * nproc))

# Every share pinned so far under this RUN_ID, read from disk rather than from the submit line:
# the point of the sum is to see the submissions this one cannot.
pinned = {}
for p in sorted(pathlib.Path(shares_dir).glob("*.share")):
    try:
        pinned[p.stem] = float(p.read_text().strip())
    except ValueError:
        sys.exit(f"FATAL: {p} is not a number. The pinned per-style-set shares are the only thing "
                 "relating the three submissions to one whole-partition ceiling; an unreadable "
                 "one must stop the run, not be skipped.")
shares_sum = sum(pinned.values())
# The share each set WOULD get from a projection-proportional split, printed so an operator whose
# split does not fit has the arithmetic in front of them rather than having to redo it.
proportional = {k: round(projected_partition * v / sum(instances.values()), 2)
                for k, v in instances.items()}

report = {"seconds_per_frame": spf, "chunk_frames": chunk_frames, "chunks": chunks,
          "nproc": nproc, "wall_hours": wall_h,
          "style_set": style_set, "style_sets_in_partition": style_sets,
          "style_instances_per_set": instances,
          "whole_partition_frames": partition_frames,
          "whole_partition_clips": facts["whole_partition_clips"],
          "projected_gpu_hours_this_chunk": round(projected_chunk, 2),
          "projected_gpu_hours_this_style_set": round(projected_style_set, 2),
          "projected_gpu_hours_whole_partition": round(projected_partition, 2),
          "partition_ceiling_gpu_hours": partition_ceiling,
          "partition_ceiling_scope": "whole partition (train + eval + identity), T40_RULE_V2 §3 step 2",
          "ceiling_gpu_hours_this_style_set": share,
          "ceiling_scope": "this STYLE_SET's share of the whole-partition ceiling, T40_RULE_V2 §3 step 3",
          "pinned_shares_by_style_set": pinned,
          "pinned_shares_sum": round(shares_sum, 4),
          "proportional_shares_from_projection": proportional,
          "max_passes_per_chunk": max_passes}
print(json.dumps(report, indent=2))
pathlib.Path(gate_p).write_text(json.dumps(report, indent=2) + "\n")

if projected_partition > partition_ceiling:
    sys.exit(f"GATE FAILS: the measured {spf:.3f} s/frame projects {projected_partition:.2f} GPU-h "
             f"for the WHOLE partition ({facts['whole_partition_clips']} clips over "
             f"{facts['style_instances_whole_partition']} style-instances), over the "
             f"{partition_ceiling} GPU-h PARTITION_CEILING_GPU_H. T40_RULE_V2 §3 step 2 derives "
             "that ceiling over all 25 style-instances; this is the experiment not fitting its "
             "budget. That is the finding: record the shortfall, cut the style count or the "
             "episode count, and re-derive. Raising the ceiling to fit the recipe is the one move "
             "PR-08 forbids.")
if shares_sum > partition_ceiling:
    sys.exit(f"GATE FAILS: the per-style-set shares pinned under this RUN_ID sum to "
             f"{shares_sum:.2f} GPU-h, over the {partition_ceiling} GPU-h "
             f"PARTITION_CEILING_GPU_H. Pinned: {pinned}. T40_RULE_V2 §3 step 3: the shares are a "
             "SPLIT of the whole-partition figure and their sum is bounded by it — one budget "
             "spent three times is three budgets. A projection-proportional split of this "
             f"partition is {proportional}.")
if projected_style_set > share:
    sys.exit(f"GATE FAILS: STYLE_SET={style_set} projects {projected_style_set:.2f} GPU-h "
             f"({instances[style_set]} style-instance(s) of {sum(instances.values())}) and its "
             f"share is CEILING_GPU_H={share}. The share is too small for the set it was assigned "
             "to. Re-derive the split — a projection-proportional one for this partition is "
             f"{proportional} — and do NOT raise the whole-partition ceiling to make it fit.")
if max_passes < 1:
    sys.exit(f"GATE FAILS: CEILING_GPU_H={share} (this style set's share) does not buy even one "
             f"{wall_h} h pass on {nproc} GPU(s) for each of {chunks} chunks "
             f"({chunks * wall_h * nproc:.0f} GPU-h minimum). Either the share is wrong or the "
             "chunking is. Do NOT raise the ceiling to fit the run — PR-08 §8 item 3 derives it "
             "from the measurement, and a ceiling raised to fit is not a budget.")
print(f"GATE PASSES — the WHOLE partition ({projected_partition:.2f} GPU-h projected) fits the "
      f"{partition_ceiling} GPU-h ceiling; the shares pinned so far sum to {shares_sum:.2f}; and "
      f"STYLE_SET={style_set} ({projected_style_set:.2f} GPU-h projected) fits its {share} GPU-h "
      "share.")
PY
MAX_PASSES=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['max_passes_per_chunk'])" "${GATE}")
echo "=== partition ceiling ${PARTITION_CEILING_GPU_H} GPU-h; ${STYLE_SET} share ${CEILING_GPU_H}"
echo "===   -> ${MAX_PASSES} passes per chunk"

# -- DONE re-entry -------------------------------------------------------------------------------
# Cheapest possible check, and it comes before any work. Split into two facts the way 94 had to
# learn to split them: "DONE exists" and "the record is written" are not the same, and conflating
# them made a failed record step unrecoverable except by a by-hand command that nothing logs.
if [[ -f "${CHUNK_DIR}/DONE" ]]; then
  if [[ -s "${CHUNK_DIR}/chunk_metadata.json" ]]; then
    echo "chunk ${CHUNK_TAG} already complete — DONE and a written record. Nothing to do."
    exit 0
  fi
  echo "=== DONE exists but no chunk_metadata.json. RE-RECORDING ONLY: no generation is started"
  echo "    and no pass is charged against the ${CEILING_GPU_H} GPU-h ${STYLE_SET} share."
fi

# -- pass rail -----------------------------------------------------------------------------------
# Persisted, not inferred from SLURM_RESTART_COUNT alone. `scontrol requeue` keeps the job id and
# increments that counter, but a MANUAL resubmit after a walltime kill starts a new job at restart
# 0 — and manual resubmission is exactly what a 4 h cap on a multi-pass run produces. Counting only
# requeues means the ceiling silently never binds.
if [[ ! -f "${CHUNK_DIR}/DONE" ]]; then
  PRIOR=$( [[ -f "${PASSES_LOG}" ]] && wc -l < "${PASSES_LOG}" || echo 0 )
  PASS_NO=$((PRIOR + 1))
  RESTARTS=${SLURM_RESTART_COUNT:-0}

  # -- a USR1 that arrives before the driver does --------------------------------------------------
  # The trap at the top of this file keeps the shell alive through the setup phase, which is the
  # whole point of moving it there. But surviving is not the same as proceeding: if there are five
  # minutes left on the wall, a driver started now is a checkpoint load that gets SIGKILLed, and
  # the pass would be charged against the ceiling for producing nothing. So requeue instead, before
  # anything is charged. Bounded by MAX_PASSES rather than by a number invented here — a chunk that
  # keeps being signalled during setup is a queue problem, and it must not be an unbounded loop on
  # the paid QoS.
  EARLY=${CHUNK_DIR}/early_requeues
  if (( GOT_USR1 )); then
    EARLY_N=$( [[ -f "${EARLY}" ]] && wc -l < "${EARLY}" || echo 0 )
    if (( EARLY_N >= MAX_PASSES )); then
      echo "FATAL: USR1 arrived before the driver started for the ${EARLY_N}th time — the wall is"
      echo "       being hit during setup, not during generation. Requeueing again would loop"
      echo "       without ever charging a pass. Read ${EARLY} and ${PROJ}/logs/t040-restyle.*.out."
      exit 1
    fi
    echo "$(date -Is) job=${SLURM_JOB_ID:-none} restarts=${RESTARTS} usr1 before driver" >> "${EARLY}"
    echo "=== USR1 before the driver started: requeueing ${SLURM_JOB_ID} with nothing charged."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi

  if (( PASS_NO > MAX_PASSES )); then
    echo "FATAL: ${PRIOR} passes already recorded in ${PASSES_LOG} and still no ${CHUNK_DIR}/DONE"
    echo "       — stopping at the ${STYLE_SET} share of ${CEILING_GPU_H} GPU-h (max"
    echo "       ${MAX_PASSES} passes per chunk)."
    echo "       If the restyle genuinely needs more, that is a finding to record BEFORE the"
    echo "       number is raised. Read ${PASSES_LOG} and ${PROJ}/logs/t040-restyle.*.out first."
    exit 1
  fi

  # -- the run-level GPU-h ledger ------------------------------------------------------------------
  # RESERVE BEFORE SPENDING, and reserve where every other submission can see it. One file per
  # (style set, chunk, pass) under ${OUT}/gpu_ledger/, so concurrent chunks never write the same
  # path and no lock is needed on a shared filesystem — the sum is a glob. The reservation is
  # written first and the total re-checked afterwards: two chunks starting in the same instant can
  # both see room, and the re-check makes the loser back its own reservation out rather than both
  # proceeding. Both backing out is the acceptable failure here, because stopping is the safe
  # direction for a budget and the operator resubmits.
  #
  # TWO BOUNDS, because T40_RULE_V2 §3 defines two (see the gate above):
  #   * this style set's reservations may not exceed its own share (CEILING_GPU_H). Without this a
  #     runaway train submission could eat identity's share and arm C — the control the whole
  #     experiment turns on — would be the one that runs out of budget.
  #   * the run total across every style set may not exceed PARTITION_CEILING_GPU_H. This is the
  #     bound the per-set counter cannot see, and it is why the ledger is one shared directory.
  RESERVATION=${LEDGER_DIR}/$(printf '%s-pass%03d.json' "${CHUNK_TAG}" "${PASS_NO}")
  PASS_COST=$(python -c "print(${NPROC} * ${WALL_H}.0)")
  ledger_total() {  # [style-set to filter on; empty or absent = the whole run]
    python - "${LEDGER_DIR}" "${1:-}" <<'PY'
import json, pathlib, sys
only = sys.argv[2] if len(sys.argv) > 2 else ""
total = 0.0
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    try:
        e = json.loads(p.read_text())
    except ValueError:
        sys.exit(f"FATAL: {p} is not readable JSON. The GPU-h ledger is the only thing bounding "
                 "this run's spend across style sets; a corrupt entry must stop it, not be "
                 "skipped.")
    if only and e.get("style_set") != only:
        continue
    total += float(e.get("gpu_hours_worst_case", 0.0))
print(f"{total:.4f}")
PY
  }
  SPENT_BEFORE=$(ledger_total)
  SET_SPENT_BEFORE=$(ledger_total "${STYLE_SET}")
  if (( $(python -c "print(1 if ${SET_SPENT_BEFORE} + ${PASS_COST} > ${CEILING_GPU_H} else 0)") )); then
    echo "FATAL: STYLE_SET=${STYLE_SET} has ${SET_SPENT_BEFORE} GPU-h reserved in the ledger of run"
    echo "       ${RUN_ID}, and this pass would add ${PASS_COST} — over its ${CEILING_GPU_H} GPU-h"
    echo "       SHARE. T40_RULE_V2 §3 step 3 splits the whole-partition ceiling across the style"
    echo "       sets; a set that overruns its share is spending another set's budget, and the"
    echo "       other set here is arm C. Re-derive the split, do not raise the share."
    echo "       The ledger is ${LEDGER_DIR}/ — one file per pass, and it is the record."
    exit 1
  fi
  if (( $(python -c "print(1 if ${SPENT_BEFORE} + ${PASS_COST} > ${PARTITION_CEILING_GPU_H} else 0)") )); then
    echo "FATAL: the run-level ledger has ${SPENT_BEFORE} GPU-h reserved across every style set of"
    echo "       run ${RUN_ID}, and this pass would add ${PASS_COST} — over the"
    echo "       ${PARTITION_CEILING_GPU_H} GPU-h whole-partition ceiling. PR-08 §8 item 3 derives"
    echo "       that number from the measurement; raising it to fit the run is the one move the"
    echo "       document forbids. The ledger is ${LEDGER_DIR}/ — one file per pass, and it is the"
    echo "       record."
    exit 1
  fi
  python - "${RESERVATION}" "${STYLE_SET}" "${CHUNK_TAG}" "${PASS_NO}" "${PASS_COST}" \
           "${SLURM_JOB_ID:-none}" "${NPROC}" "${WALL_H}" <<'PY'
import datetime, json, socket, sys
path, style_set, tag, pass_no, cost, jobid, nproc, wall_h = sys.argv[1:9]
open(path, "w").write(json.dumps({
    "style_set": style_set, "chunk_tag": tag, "pass": int(pass_no),
    # The worst case, which is what the ceiling is bounded against: Slurm bills the whole
    # allocation for the whole wall whether or not the pass uses it.
    "gpu_hours_worst_case": float(cost), "nproc": int(nproc), "wall_hours": float(wall_h),
    "job_id": jobid, "hostname": socket.gethostname(),
    "reserved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
  SPENT_AFTER=$(ledger_total)
  SET_SPENT_AFTER=$(ledger_total "${STYLE_SET}")
  if (( $(python -c "print(1 if ${SET_SPENT_AFTER} > ${CEILING_GPU_H} or ${SPENT_AFTER} > ${PARTITION_CEILING_GPU_H} else 0)") )); then
    rm -f "${RESERVATION}"
    echo "FATAL: with this pass reserved the ledger totals ${SET_SPENT_AFTER} GPU-h for"
    echo "       STYLE_SET=${STYLE_SET} (share ${CEILING_GPU_H}) and ${SPENT_AFTER} GPU-h for the"
    echo "       whole run (ceiling ${PARTITION_CEILING_GPU_H}) — one of the two is over, because"
    echo "       another submission reserved between this job's check and its write. This pass has"
    echo "       backed its own reservation out and spent nothing. Resubmit once the other chunks"
    echo "       finish, or re-derive the split."
    exit 1
  fi
  echo "=== ledger: ${STYLE_SET} ${SET_SPENT_BEFORE} -> ${SET_SPENT_AFTER} of its ${CEILING_GPU_H}"
  echo "    GPU-h share; run ${RUN_ID} ${SPENT_BEFORE} -> ${SPENT_AFTER} of the"
  echo "    ${PARTITION_CEILING_GPU_H} GPU-h whole partition (every style set charges here)"
  echo "=== pass ${PASS_NO} of ${MAX_PASSES} (slurm restarts=${RESTARTS}) on $(hostname) at $(date -Is)"

  # One line per pass, written BEFORE the driver starts. A SIGKILL at the wall still leaves the
  # record, and it still counts — the entire point of persisting it.
  python - "${PASSES_LOG}" "${PASS_NO}" "${RESTARTS}" "${SLURM_JOB_ID:-none}" "${WORK_SHA}" \
           "${NPROC}" "${WALL_H}" "${RESERVATION}" <<'PY'
import datetime, json, socket, sys
log, pass_no, restarts, jobid, work_sha, nproc, wall_h, reservation = sys.argv[1:9]
with open(log, "a") as f:
    f.write(json.dumps({
        "pass": int(pass_no), "slurm_restart_count": int(restarts), "job_id": jobid,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "work_sha256": work_sha,
        "nproc": int(nproc),
        # The worst case, which is what the ceiling is bounded against: Slurm bills the whole
        # allocation for the whole wall whether or not the pass uses it.
        "gpu_hours_worst_case": int(nproc) * float(wall_h),
        "ledger_entry": reservation,
    }) + "\n")
PY
  echo "=== recorded pass ${PASS_NO} in ${PASSES_LOG}"

  # -- harvest -----------------------------------------------------------------------------------
  # Lifted from 96_generate_apple_variations.sbatch, which lifted it from 95. Files any clip the
  # driver finished into clips/<style_set>/, and leaves the rest in a missing-list which is the
  # ONLY thing handed to the driver — so a requeue regenerates exactly what is absent and nothing
  # else. sample_outputs.json's status is READ rather than the file merely counted, because the
  # error path writes that file too.
  #
  # AND IT READS THE COMPOSITOR'S OWN EVIDENCE, PER CLIP, BECAUSE THIS FILE MAKES A CLAIM ABOUT IT.
  # chunk_metadata.json and the NOT_TRAINING_DATA marker below both say "the composite ran on every
  # clip filed here". Until 2026-08-22 that was a sentence about a sibling script's behaviour, filed
  # on two conditions that could both be true without it: vision.mp4 exists and status == "success".
  # PR-08 §6 calls G0c "solved by construction", and a construction that is asserted rather than
  # verified is not a construction — it is the same shape of claim as PR-07's -359.41, a number
  # whose instrument nobody checked. So the record is now READ and every field the claim rests on is
  # compared to something independent:
  #
  #   * status == success and backend == transfer25. The second keeps the null placeholder — which
  #     since G0c landed writes a REAL decodable mp4 — out of a corpus by file content rather than
  #     by the sbatch's habit of not passing --backend.
  #   * g0c.composited is true, and g0c.frames_composited == g0c.frames_total == the frame count
  #     THIS WORK-LIST ROW declares. That is the per-clip, specific half: a record copied from
  #     another unit, or a composite that stopped early, disagrees with the row.
  #   * the area bound that refused over-large masks was cross-checked against the segmenter that
  #     made them (g0c.area_bound.cross_checked), and its sha256 is recorded.
  #
  # A clip that fails any of these is NOT filed and is NOT quietly left for the next pass: it is a
  # fatal, because "success without a composite" means the generation path is not the one this job
  # describes and every further pass would produce more of it.
  #
  # THE EVIDENCE THEN TRAVELS WITH THE CLIP. Filing writes <unit>.g0c.json beside <unit>.mp4 holding
  # that unit's own compositor record, and a later pass that finds an mp4 with no sidecar refuses
  # the chunk. That is what covers clips filed by an EARLIER driver: the old harvest skipped any
  # unit already in clips/, so a pre-G0c clip would have survived a requeue untouched and inherited
  # the new claim. It cannot now — it has no sidecar, and the raw record it came from may be long
  # gone, so the sidecar is the only durable per-clip proof and its absence is the refusal.
  harvest() {  # work-list  raw-dir  dest-dir  missing-list-to-write
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, pathlib, shutil, sys
work, raw, dest, out, refused_out = (pathlib.Path(a) for a in sys.argv[1:6])


def permanently_refused(unit):
    """Did G0c refuse this unit on its SOURCE masks, before the generator ran?

    THE BUG THIS EXISTS TO FIX, and it is certain rather than hypothetical on this corpus. A unit
    the source-mask preflight refuses never produces a ``vision.mp4``: the driver quarantines
    whatever the backend wrote as ``vision.uncomposited.mp4``, under a name nothing looks for. The
    loop below then puts it on the MISSING list, and the missing list is what the chunk requeues on.
    So pass 1 files the survivors, pass 2 files nothing, ``MISSING_AFTER == MISSING_BEFORE``, and
    the chunk exits 1 at the no-op guard without ever reaching ``touch DONE`` — which means
    ``chunk_metadata.json`` and the whole PR-08 §6 record are never written. The clips are produced
    and filed; the record that says so is not. At 385 of 402 episodes refused, that fires on the
    first chunk of the first real run.

    WHY THIS IS AN ACCOUNTING FIX AND NOT A WEAKENING OF ANYTHING. It cannot make a unit pass: a
    refused unit is still not filed, still has no clip in ``dest``, still has no sidecar, and is
    still counted nowhere as a success. What changes is only whether the chunk RETRIES it. And
    retrying it cannot ever succeed: ``check_mask``'s verdict is a function of the source bytes, the
    segmenter and the bound — ``robot_composite.MaskCache.key`` hashes the first two and the third
    is the frozen bound object — and all three are fixed for the life of a run. A permanent refusal
    on the missing list is not work outstanding, it is a decision already taken.

    IT IS READ OFF THE NAME THE DRIVER PUTS THERE FOR EXACTLY THIS. ``SourceMaskRefusal`` is a
    subclass rather than a flag, and ``restyle_transfer25.py:493-501`` says why: *"What the subclass
    buys is a NAME in the per-unit record — ``detail`` reads ``SourceMaskRefusal: …``"*. Nothing
    read it until now.

    EVERY OTHER FAILURE STAYS MISSING. A crash, a truncated video, an OOM, a licence error: all of
    those are retried exactly as before, and if they repeat, the no-op guard still fires. The
    unbounded-loop protection is preserved precisely where it belongs — this narrows what counts as
    outstanding work, it does not remove the check.
    """
    record = raw / unit / "sample_outputs.json"
    try:
        payload = json.loads(record.read_text())
    except (OSError, ValueError):
        # No record, or an unreadable one, is NOT a refusal. It is a unit nobody can account for,
        # and the safe reading of that is "missing" — which retries, and which the no-op guard
        # bounds.
        return None
    if payload.get("status") != "error":
        return None
    detail = str(payload.get("detail") or "")
    if not detail.startswith("SourceMaskRefusal:"):
        return None
    g0c = payload.get("g0c")
    reason = None
    if isinstance(g0c, dict):
        preflight = g0c.get("source_mask_preflight")
        if isinstance(preflight, dict):
            reason = preflight.get("refusal") or preflight.get("detail")
    return {"unit": unit, "detail": detail, "reason": reason,
            "episode": payload.get("episode"), "style": payload.get("style")}

def sidecar(unit):
    return dest / f"{unit}.g0c.json"

def read_proof(path):
    """The sidecar's CONTENT, or a reason. Presence is not proof, and this file already knows why.

    The correction of 2026-08-16 above states the rule for the raw record — "a harvest keyed on file
    presence counts failures as finished work" — and a sidecar checked only with is_file() is that
    same check by another name: an empty ``{}`` beside a clip would satisfy the count in
    chunk_metadata.json while proving nothing about whether the composite ran. So every evidence
    file is read, on every pass, including the ones this pass did not write.
    """
    try:
        proof = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"it is unreadable ({exc})"
    if not isinstance(proof, dict):
        return None, "it is not an object"
    g0c = proof.get("g0c")
    if not isinstance(g0c, dict) or g0c.get("composited") is not True:
        return None, "it carries no g0c.composited: true, so it proves nothing about that clip"
    return proof, None

def whole_frames(container, field):
    """``container[field]`` when it is a plain integer, else None. Never raises.

    ``int(None)`` is a TypeError, and a traceback is not a refusal — the operator reads it as a
    crash of this job rather than as a diagnosis of the record. A frame count that is absent, null
    or a string proves nothing about how many frames were composited, which is the same failure as
    a count that disagrees, so it takes the same path.
    """
    value = container.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else None

def refuse(unit, why, remedy):
    sys.exit(
        f"FATAL: unit {unit} cannot be filed into {dest} — {why}\n"
        f"       PR-08 §6 G0c is solved BY CONSTRUCTION, which means this job may only file a clip\n"
        f"       whose own record PROVES the real robot's pixels were composited back over every\n"
        f"       frame of it. chunk_metadata.json and the NOT_TRAINING_DATA marker state that as a\n"
        f"       fact about this directory; filing this clip would make both of them false.\n"
        f"       {remedy}"
    )

def evidence(unit, row):
    """This unit's compositor record, or a fatal. Never a bool: the caller files what it returns."""
    rec = raw / unit / "sample_outputs.json"
    try:
        payload = json.loads(rec.read_text())
    except (OSError, ValueError) as exc:
        return None, f"its record {rec} is unreadable ({exc})"
    if payload.get("status") != "success":
        return None, f"status is {payload.get('status')!r}"
    if payload.get("backend") != "transfer25":
        refuse(unit, f"its record says backend={payload.get('backend')!r}, not 'transfer25'.",
               "A --backend other than transfer25 is a PLACEHOLDER generator. Since G0c landed it "
               "writes a valid mp4, so nothing about the file itself keeps it out of a corpus — "
               "this does. 97 never passes --backend; find out who did.")
    g0c = payload.get("g0c")
    if not isinstance(g0c, dict) or g0c.get("composited") is not True:
        refuse(unit, "its record carries no g0c.composited: true.",
               "A driver that reports success without compositor evidence is not the driver this "
               "job describes (scripts/restyle_transfer25.py). Do not file its output.")
    want = int(row["frames"])
    if whole_frames(g0c, "frames_composited") != want or whole_frames(g0c, "frames_total") != want:
        refuse(unit,
               # repr, because "5" and 5 print the same and only one of them is a frame count.
               f"it composited {g0c.get('frames_composited')!r} of {g0c.get('frames_total')!r} "
               f"frames and this work-list row declares {want}.",
               "Every frame or none: a clip composited on part of its frames carries the generated "
               "manipulator on the rest, and nothing downstream reads a corpus frame by frame.")
    bound = g0c.get("area_bound") or {}
    if bound.get("cross_checked") is not True or not bound.get("artifact_sha256"):
        refuse(unit, "its area bound was not cross-checked against the segmenter that made the masks.",
               "The bound is what refuses an over-large mask, and a bound measured under other "
               "weights or over another corpus is a number with no distribution behind it — see "
               "robot_composite.load_area_bound.")
    return {"unit": unit, "frames": want, "g0c": g0c,
            "seed": payload.get("seed"), "episode": payload.get("episode"),
            "style": payload.get("style"), "backend": payload.get("backend"),
            # T40_RULE_V11 §2.4 asks for the stage "recorded per clip", and this sidecar is the
            # per-clip record. Read off the work-list row rather than off the environment, so a
            # clip says which stage PRODUCED it and not which stage happened to be harvesting.
            "stage": payload.get("stage"),
            "verified_by": "97_transfer25_restyle.sbatch harvest()",
            "verified": ["status==success", "backend==transfer25", "g0c.composited",
                         "frames_composited==frames_total==work_list.frames",
                         "area_bound.cross_checked"]}, None

missing = []
refused = []
for line in work.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    unit = row["unit"]
    want_frames = int(row["frames"])
    if (dest / f"{unit}.mp4").is_file():
        # Already filed — by THIS harvest, which means the sidecar is there. If it is not, the clip
        # was filed by something that did not check, and no later pass can go back and prove it.
        if not sidecar(unit).is_file():
            refuse(unit, f"{dest / (unit + '.mp4')} is already filed with no {sidecar(unit).name} "
                         "beside it, so nothing here can show it was ever composited.",
                   "It was filed by a driver or a harvest that predates PR-08 §6 G0c's evidence "
                   "check. Delete that clip and let this chunk regenerate it; do not delete the "
                   "sidecar requirement.")
        # And the sidecar is READ, not merely counted. This row knows how many frames the unit has,
        # which the directory-wide sweep below cannot know, so the one check that can only happen
        # here happens here: an evidence file that says nothing, or that says a different clip's
        # frame count, is not evidence about this clip.
        proof, why = read_proof(sidecar(unit))
        if proof is None:
            refuse(unit, f"{sidecar(unit).name} is beside it but {why}.",
                   "A sidecar is the only durable proof that the composite ran on that clip. One "
                   "that cannot be read, or that does not say so, leaves the clip in exactly the "
                   "state the sidecar exists to exclude. Delete the clip and its sidecar and let "
                   "this chunk regenerate them.")
        if whole_frames(proof, "frames") != want_frames:
            refuse(unit,
                   f"{sidecar(unit).name} records {proof.get('frames')!r} frames and this "
                   f"work-list row declares {want_frames}.",
                   "The evidence beside that clip is about a different clip — copied, or written "
                   "for another unit. Delete both and let this chunk regenerate them.")
        continue
    src = raw / unit / "vision.mp4"
    if not src.is_file():
        # No output at all, or the driver quarantined an uncomposited one as
        # vision.uncomposited.mp4. Those two are not the same thing and this used to treat them as
        # one: see permanently_refused() for what that cost. A G0c source-mask refusal is a
        # decision, and a decision is not outstanding work.
        decided = permanently_refused(unit)
        if decided is not None:
            refused.append(decided)
        else:
            missing.append(line)
        continue
    proof, why = evidence(unit, row)
    if proof is None:
        print(f"harvest: {unit} not filed — {why}")
        missing.append(line)
        continue
    # The sidecar lands BEFORE the mp4 moves. A crash between the two leaves a sidecar with no clip,
    # which the next pass overwrites harmlessly; the other order would leave a clip with no proof,
    # which is the state this whole check exists to make impossible.
    sidecar(unit).write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    shutil.move(str(src), str(dest / f"{unit}.mp4"))

out.write_text("".join(f"{u}\n" for u in missing))
# WRITTEN, NOT SWALLOWED. This is the whole risk of the change above: a chunk that now reaches DONE
# while most of its units produced nothing must SAY so, or the record lies by omission. It is a file
# the shell reads back into chunk_metadata.json, and it is printed here in full count.
refused_out.write_text(json.dumps(refused, indent=2, sort_keys=True) + "\n")
# The invariant this function maintains, re-checked over the WHOLE directory rather than over the
# units this pass touched: every clip in it has its own evidence file. A unit absent from this
# chunk's work list cannot be checked above, and it is exactly the case a wrong claim would hide in.
filed = {p.stem for p in dest.glob("*.mp4")}
proofs = set()
unreadable = []
for p in sorted(dest.glob("*.g0c.json")):
    # Read, not counted — same rule as the per-row check above, applied to the sidecars this pass
    # never looked at. What the frame count means cannot be checked here (a unit outside this
    # chunk's work list has no row to compare against), but "this file claims the composite ran"
    # can, and a sidecar that does not claim it must not stand in for one that does.
    seen, why = read_proof(p)
    if seen is None:
        unreadable.append(f"{p.name}: {why}")
        continue
    proofs.add(p.name[: -len(".g0c.json")])
if unreadable:
    sys.exit(
        f"FATAL: {dest} holds {len(unreadable)} G0c evidence file(s) that are not evidence:\n       "
        + "\n       ".join(unreadable[:5])
        + (f"\n       … and {len(unreadable) - 5} more" if len(unreadable) > 5 else "")
        + "\n       chunk_metadata.json counts these files as proof that the composite ran on the "
        "clips beside them. A file that says nothing is the file-presence check this harvest "
        "refuses everywhere else. Remove those clips and their sidecars and let the chunk "
        "regenerate them."
    )
orphans = sorted(filed - proofs)
if orphans:
    sys.exit(
        f"FATAL: {dest} holds {len(orphans)} clip(s) with no G0c evidence file beside them: "
        f"{orphans[:5]}{' …' if len(orphans) > 5 else ''}\n"
        "       chunk_metadata.json is about to claim the composite ran on every clip in this "
        "directory. It did not run on those, or nothing here can show that it did. Remove them "
        "and let the chunk regenerate them."
    )
print(f"harvest: {len(filed)} clips filed, each with its own G0c evidence file; "
      f"{len(missing)} to generate; {len(refused)} refused by G0c on their SOURCE masks")
if refused:
    print("harvest: those refusals are FINAL for this run — check_mask's verdict is a function of "
          "the source bytes, the segmenter and the bound, and all three are fixed here. They are "
          "not retried and they are not filed. First few:")
    for entry in refused[:3]:
        print(f"           {entry['unit']}: {entry['detail'][:110]}")
PY
  }

  harvest "${WORK_LIST}" "${RAW}" "${CLIPS}" "${MISSING}" "${REFUSED}"
  MISSING_BEFORE=$(wc -l < "${MISSING}")

  RC=0
  # GOT_USR1 is NOT reset here. It is set by the trap installed at the top of the file, it is
  # already known to be 0 (a USR1 before this point requeued above), and re-initialising it inside
  # the branch is how the old trap came to be installed inside the branch as well.
  START=${SECONDS}
  if [[ -s "${MISSING}" ]]; then
    cd "${FRAMEWORK}"
    # NEVER run the driver in the foreground. A foreground child makes the 4 h wall a DEAD END
    # rather than a link in the chain: --requeue covers preemption and node failure, but a job that
    # simply runs out of walltime is killed outright, so the `scontrol requeue` at the bottom is
    # never reached and the chain stops one pass in. --signal=B:USR1@300 buys 5 minutes, and the
    # trap only fires while the shell is sitting in `wait`.
    #
    # --no-guardrails IS MANDATORY ON COSMOS INFERENCE IN THIS PROJECT, and here it is not merely a
    # licence problem. nvidia/Cosmos-Guardrail1 is a GATED repo with no accepted licence on this
    # allocation (job 187249 died on it), but the disqualifying fact is what the guardrail does to
    # frames it lets through: its RetinaFace post-processor REWRITES the returned frames, and a
    # false positive on robot footage blurs the hand. This pipeline's entire output is evidence
    # about whether a restyle preserves geometry (G0b) and the robot's own pixels (G0c). A
    # post-processor that edits the hand region edits the evidence, silently, after the model has
    # already produced the frame — and no gate downstream could tell that apart from a generator
    # defect. Leave it off.
    python "${RESTYLE_DRIVER}" \
        --checkpoint-path "${TRANSFER_CHECKPOINT_PATH}" \
        --no-guardrails \
        --manifest "${SOURCE}/manifest.json" \
        --styles "${STYLE_PARTITION}" \
        --style-set "${STYLE_SET}" \
        --work-list "${MISSING}" \
        --out "${RAW}" \
        --resolution 640x480 \
        --control "${CONTROL}" \
        --mask-cache "${OUT}/robot_masks" \
        --nproc "${NPROC}" &
    # --mask-cache is ONE directory for the whole RUN_ID, not one per chunk, and that is the point
    # of passing it at all: PR-08 §6 G0c's robot mask is a property of the SOURCE frame, so it is
    # identical across every restyle of an episode, and the committed partition asks for 25 of them.
    # Left to the driver's default it would land beside ${RAW} and be recomputed once per style set;
    # here the train, eval and identity submissions of one RUN_ID share it. Entries are keyed on the
    # source bytes, the prompt and the estimator version, so nothing stale can be served across a
    # rebuilt corpus or a re-pinned segmenter.
    # The trap at the top of the file reads this. It was assigned "" there, so the handler knows
    # there is no driver to terminate before this line and knows there is one after it; there is no
    # second `trap` call here, and there must not be — see FIX 8's note at the top.
    CHILD=$!
    # WAIT EXACTLY ONCE, AND KEEP WHAT IT SAID.
    #
    # The old form was `while kill -0 "${CHILD}"; do wait "${CHILD}" && RC=0 || RC=$?; done`, and it
    # loses the driver's exit status whenever bash has already reaped the child before the first
    # `kill -0` — a fast crash, which is the case that matters most. The body then never runs, RC
    # keeps the 0 it was initialised with, and the pass reports "exited 0 having filed no clip":
    # branch (b) instead of branch (c), a no-op diagnosis for a crash, and the operator sent to look
    # for a driver that silently did nothing instead of at a traceback. `wait` on a known job id
    # returns its remembered status even after the child is gone, so calling it unconditionally is
    # both correct and the only way to see that status.
    #
    # The loop still exists for the OTHER case: a trap makes `wait` return early with 128+signal and
    # the child still alive, so a >128 status is re-waited rather than believed. `set -e` does not
    # fire on either form because both sit in an `if` condition.
    while true; do
      if wait "${CHILD}"; then RC=0; else RC=$?; fi
      if (( RC > 128 )) && kill -0 "${CHILD}" 2>/dev/null; then
        # Interrupted by our own USR1 handler, not by the driver exiting. Go back to waiting.
        continue
      fi
      break
    done
    ELAPSED=$((SECONDS - START))
    echo "driver pid ${CHILD} exited ${RC} after ${ELAPSED}s at $(date -Is) (usr1=${GOT_USR1})"
    harvest "${WORK_LIST}" "${RAW}" "${CLIPS}" "${MISSING}" "${REFUSED}"
  else
    ELAPSED=$((SECONDS - START))
    echo "=== every unit in this chunk is already on disk — no checkpoint load"
  fi

  # -- the four exit branches --------------------------------------------------------------------
  # (a) target reached -> record + DONE. (b) a pass that made NO progress -> refuse to requeue.
  # (c) a real crash -> refuse to requeue, because requeueing only rebuilds it. (d) otherwise
  # requeue. Branch (c) deliberately exempts the USR1 case: the non-zero status is OUR SIGTERM
  # (143), not the run's health, and treating it as a crash would turn the walltime handling into
  # an elaborate way of stopping after one pass.
  #
  # BRANCH (b) IS NOT 94's `ELAPSED < 300` TEST, DELIBERATELY. That heuristic is what 94 could
  # afford: a trainer's progress lives inside a checkpoint it may not have written yet, so elapsed
  # time is the only proxy available. Here the missing-list is an EXACT progress measure, and the
  # proxy is wrong in both directions. It refuses a fast pass that really did file clips — a short
  # chunk, or a resume with three units left, both legitimate — and it permits a pass that burns
  # the whole 4 h wall producing nothing to requeue forever, which is precisely the unbounded loop
  # this branch exists to stop. Progress is the honest test, and it is available, so use it.
  MISSING_AFTER=$(wc -l < "${MISSING}")
  echo "=== units missing: ${MISSING_BEFORE} before this pass, ${MISSING_AFTER} after"
  if [[ -s "${MISSING}" ]]; then
    if [[ "${RC}" -eq 0 && "${GOT_USR1}" -eq 0 && "${MISSING_AFTER}" -ge "${MISSING_BEFORE}" ]]; then
      echo "FATAL: exited 0 after ${ELAPSED}s having filed no clip at all — ${MISSING_AFTER} units"
      echo "       still missing, unchanged. That is a no-op, not an interrupted slice, and"
      echo "       requeueing it would loop on the paid QoS until the allocation is gone."
      echo "       First missing:"
      head -3 "${MISSING}"
      exit 1
    fi
    if [[ "${RC}" -ne 0 && "${GOT_USR1}" -eq 0 ]]; then
      echo "FATAL: driver failed with ${RC} — requeueing only rebuilds the same crash."
      exit "${RC}"
    fi
    echo "$(wc -l < "${MISSING}") units left in ${CHUNK_TAG} -> requeue ${SLURM_JOB_ID}; the"
    echo "missing-list IS the resume, so the next pass regenerates exactly what is absent."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  # A chunk can now legitimately reach DONE with most of its work list refused, which is new and
  # must never be silent. On this corpus 385 of 402 episodes are refused by one half of check_mask
  # or the other, so this line is the normal case rather than an exception.
  if [[ -s "${REFUSED}" ]]; then
    REFUSED_N=$(python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "${REFUSED}")
    if (( REFUSED_N > 0 )); then
      echo "=== ${REFUSED_N} unit(s) in ${CHUNK_TAG} were REFUSED BY G0c ON THEIR SOURCE MASKS and"
      echo "===   produced no clip. They are decided, not outstanding: check_mask-s verdict is a"
      echo "===   function of the source bytes, the segmenter and the bound, all fixed for this"
      echo "===   run, so retrying them cannot change the answer. Nothing was filed for them."
      echo "===   Per-unit reasons: ${REFUSED}"
    fi
  fi
  touch "${CHUNK_DIR}/DONE"
fi

# -- the record ----------------------------------------------------------------------------------
# PR-08 §6 "Recorded regardless of verdict": the style partition hash, GEOM_TOL, EST_DRIFT_P95,
# measured throughput, GPU-h spent, and the generator checkpoint id AND revision. Written here and
# re-writable by the DONE re-entry above, so a failed record step is recoverable BY THIS SCRIPT
# rather than by a by-hand command that nothing logs.
python - "${CHUNK_DIR}" "${CHUNK_TAG}" "${STYLE_SET}" "${CHUNK_INDEX}" "${CHUNK_TOTAL}" \
         "${TRANSFER_MODEL_ID}" "${TRANSFER_MODEL_REVISION}" "${TRANSFER_CHECKPOINT_PATH}" \
         "${STYLES_CONTENT_SHA}" "${WORK_SHA}" "${CEILING_GPU_H}" "${PR08_T39_REPORTED}" \
         "${GEOM_CONSTANTS}" "${THROUGHPUT}" "${CLIPS}" "${STYLES_FILE_SHA}" \
         "${STYLE_PARTITION}" "${STYLE_SOURCE}" "${PARTITION_FACTS}" "${WORK_LIST}" \
         "${LEDGER_DIR}" "${PR08_T39_ARTIFACT}" "${PR08_T39_ARTIFACT_SHA256}" \
         "${PARTITION_CEILING_GPU_H}" "${SHARES_DIR}" "${GEOM_STEP_FRAMES}" \
         "${PR08_T39_VOID_OVERRIDDEN}" <<'PY'
import json, pathlib, sys
(chunk_dir, tag, style_set, idx, total, model, rev, ckpt, styles_content_sha, work_sha,
 ceiling, t39, geom_p, tp_p, clips, styles_file_sha, styles_rendering_p, styles_source_p,
 facts_p, work_list_p, ledger_dir, t39_artifact, t39_artifact_sha,
 partition_ceiling, shares_dir, geom_step_frames, t39_void_override) = sys.argv[1:28]
cd = pathlib.Path(chunk_dir)
passes = []
pl = cd / "passes.jsonl"
if pl.is_file():
    passes = [json.loads(l) for l in pl.read_text().splitlines() if l.strip()]
geom = json.loads(pathlib.Path(geom_p).read_text())
tp = json.loads(pathlib.Path(tp_p).read_text())
facts = json.loads(pathlib.Path(facts_p).read_text())
n_clips = len(list(pathlib.Path(clips).glob("*.mp4")))
# THE UNITS THAT PRODUCED NOTHING, AND WHY, COUNTED FROM THE SAME FILE THE HARVEST WROTE. Without
# this the record states how many clips a chunk filed and is silent about how many its own gate
# refused — and on this corpus the second number is the larger one by an order of magnitude. A §6
# record that says "136 clips" over a work list of 3216 without saying what happened to the other
# 3080 is a record a later reader cannot interpret. Read from the chunk directory rather than passed
# as a nineteenth positional argument, because it is written by the harvest a few lines above and
# argv is already long enough to be its own hazard.
refused_p = pathlib.Path(chunk_dir) / "refused.json"
refused = []
if refused_p.is_file():
    try:
        refused = json.loads(refused_p.read_text())
    except ValueError:
        sys.exit(f"FATAL: {refused_p} is unreadable. It is the harvest's own account of which units "
                 "G0c refused, and a record that cannot read it would undercount the refusals "
                 "rather than report none.")
# The G0c claim below is COUNTED, not asserted. harvest() writes <unit>.g0c.json beside each clip it
# files, only after reading that unit's own compositor record, and refuses to file one without it.
# Recomputing the two counts here means the record cannot claim what the directory does not show —
# and if something removed evidence between the harvest and now, this record refuses to be written
# rather than stating a number it cannot support.
g0c_proofs = sorted(pathlib.Path(clips).glob("*.g0c.json"))
if len(g0c_proofs) != n_clips:
    sys.exit(f"FATAL: {clips} holds {n_clips} clips and {len(g0c_proofs)} G0c evidence files. This "
             "record was about to state that the composite ran on every clip filed here. Do not "
             "write a claim this directory cannot support — find the clips without evidence "
             "(harvest() names them) and remove them.")
ledger = sorted(pathlib.Path(ledger_dir).glob("*.json"))
ledger_entries = [json.loads(p.read_text()) for p in ledger]
rec = {
    "task": "T-040", "prereg": "docs/preregistration/PR-08-photoreal-augmentation.md",
    "rule": "T40_RULE_V1",
    "rule_v2": "T40_RULE_V2 (docs/preregistration/PR-08-V2-arm-c-frame-matching.md) — arm C "
               "frame-matched to arm B at 10 x 402, ceiling derived over the whole partition",
    "rule_v11": "T40_RULE_V11 (docs/preregistration/PR-08-V11-staged-partition.md), adopted "
                "2026-08-24 — the partition is generated in stages; stage 1 is the first four "
                "TRAIN_STYLES in committed order plus their four matched identity repeats, and "
                "the eval set is DEFERRED out of it rather than cut",
    # Off `facts`, not off a shell variable: partition_facts.json is what the expansion actually
    # emitted, so the record and the work list cannot disagree about which stage this chunk is.
    "stage": facts.get("stage"),
    "stage_rule": facts.get("stage_rule"),
    "stage1_train_styles": facts.get("stage1_train_styles"),
    "stage_instances_per_set": facts.get("stage_instances_per_set"),
    "stage_styles": facts.get("stage_styles"),
    "stage_clips": facts.get("stage_clips"),
    "stage_frames": facts.get("stage_frames"),
    "t39_reported": t39,
    # A T-39 VOID refuses (PR-07 §4/§6, PR-08 §1 — see (a) in the sbatch header). If this field is
    # non-null a human typed the whole acknowledgement sentence to generate anyway, and every clip
    # in this chunk exists because of it.
    "t39_void_overridden": t39_void_override or None,
    "t39_result_artifact": t39_artifact or None,
    "t39_result_artifact_sha256": t39_artifact_sha or None,
    "generator_model_id": model, "generator_model_revision": rev,
    "generator_checkpoint_path": ckpt, "generator_frozen": True,
    "resolution": [640, 480],
    "style_set": style_set,
    # TWO HASHES, NAMED SO NEITHER CAN BE MISTAKEN FOR THE OTHER (2026-08-15).
    # This record used to carry one field, "style_partition_sha256", holding `sha256sum` of the
    # generator-facing JSON. That is NOT the value PR-08 §6 lists under "Recorded regardless of
    # verdict": §6's partition hash is the CONTENT hash of styles.toml — canonical JSON of the
    # document minus its [hash] and [consumer] tables — which is what styles.toml.sha256 holds and
    # what the rendering republishes as source_content_sha256. The two differ by construction (one
    # is bytes of a derived file, one is content of the source), so recording the byte hash under
    # §6's name meant a later reader comparing the record against the pre-registration would find a
    # mismatch and have no way to tell "the partition changed" from "these are different hashes".
    # Both are recorded, both are labelled, and the file each belongs to is named beside it.
    "style_partition_content_sha256": styles_content_sha,
    "style_partition_content_sha256_is": (
        "PR-08 §6's partition hash: sha256(canonical_json(styles.toml minus [hash] and "
        "[consumer])). Sidecar: configs/transfer25/styles.toml.sha256. This is the value to quote "
        "with the verdict."),
    "style_partition_source": styles_source_p,
    "style_partition_file_sha256": styles_file_sha,
    "style_partition_file_sha256_is": (
        "sha256 of the BYTES of the generator-facing rendering this job actually read. Sidecar: "
        "<rendering>.sha256. Identifies the file, not the styles."),
    "style_partition_rendering": styles_rendering_p,
    "chunk": f"{idx}/{total}", "chunk_tag": tag, "work_list_sha256": work_sha,
    "work_list": work_list_p,
    "clips_in_style_set_dir": n_clips,
    # Reproducibility of a single clip: the unit id carries episode, style and repeat index, and
    # repeat r of style s was generated under seeds_by_style[s][r]. The resolved per-style seeds,
    # the committed blocks and the assignment rule are all here, so a clip can be regenerated from
    # the record without re-reading the partition — and so a reader can check after the fact that
    # arm B and arm C spanned the same seeds and the eval set spanned none of them.
    "seeds_by_style": facts["seeds_by_style"],
    "seed_blocks": facts["seed_blocks"],
    "seeds_used_by_set": facts["seeds_used_by_set"],
    "seed_assignment": facts["seed_assignment"],
    "seed_rule": facts["seed_rule"],
    "repeats_by_style": facts["repeats_by_style"],
    "clips_per_episode_this_set": facts["this_set_instances_per_episode"],
    "unit_id_format": "<episode>__<style id>__r<NN>, NN = repeat index into that style's seeds",
    "whole_partition_clips": facts["whole_partition_clips"],
    "whole_partition_frames": facts["whole_partition_frames"],
    # PR-08 §6 "Recorded regardless of verdict" applies to the refusals as much as to the clips.
    # These units reached the source-mask preflight and were refused before the generator ran, so
    # they produced no clip and cost a segmentation pass rather than a generation.
    "units_refused_by_g0c_on_source_masks": len(refused),
    "units_refused_detail": refused,
    "units_refused_note": (
        "Refused BEFORE the backend by restyle_transfer25.preflight_source_masks, which runs "
        "robot_composite.check_mask over the SOURCE masks. The verdict is a function of the source "
        "bytes, the segmenter and the bound, all fixed for this run, so these are decisions and not "
        "outstanding work: they are not retried, not filed, and counted in no success total."
    ),
    "geom_tol_px": geom.get("GEOM_TOL_px", geom.get("GEOM_TOL")),
    "est_drift_p95_px": geom.get("est_drift_p95_px", geom.get("EST_DRIFT_P95")),
    "geom_constants_source": geom_p,
    # GEOM_TOL scales ~linearly with the step it was measured at, so the number alone does not say
    # what gate it is. Both travel into the record together or neither is interpretable later.
    "geom_tol_step_frames": geom.get("step_frames"),
    "geom_tol_step_definition": geom.get("step_definition"),
    "geom_tol_n_episodes": geom.get("n_episodes"),
    "geom_tol_n_episodes_found": geom.get("n_episodes_found"),
    "measured_throughput": tp,
    # TWO ceilings, per T40_RULE_V2 §3: the whole-partition figure (step 2) and this style set's
    # share of it (step 3). Recording only one of them would leave a later reader unable to tell
    # whether the run was budgeted per set or for the experiment.
    "partition_ceiling_gpu_hours": float(partition_ceiling),
    "partition_ceiling_scope": "whole partition (train + eval + identity), T40_RULE_V2 §3 step 2",
    "ceiling_gpu_hours": float(ceiling),
    "ceiling_scope": f"the {style_set} share of the whole-partition ceiling, T40_RULE_V2 §3 step 3",
    "ceiling_shares_by_style_set": {
        q.stem: float(q.read_text().strip())
        for q in sorted(pathlib.Path(shares_dir).glob("*.share"))},
    "geom_step_frames_gated_at": int(geom_step_frames),
    "gpu_hours_worst_case_spent": sum(p.get("gpu_hours_worst_case", 0.0) for p in passes),
    # The whole run, across every style set — the number the ceiling is actually about. The
    # per-chunk sum above cannot see the other submissions.
    "gpu_hours_worst_case_run_ledger": sum(e.get("gpu_hours_worst_case", 0.0)
                                           for e in ledger_entries),
    "gpu_ledger_entries": len(ledger_entries),
    "gpu_ledger_dir": ledger_dir,
    "passes": passes, "pass_count": len(passes),
    "guardrails": "disabled (--no-guardrails): gated repo, and its RetinaFace post-processor "
                  "rewrites returned frames and blurs the hand — it edits the evidence G0b and "
                  "G0c are about",
    "gates_passed": None,
    "gates_note": "G0a and G0b have NOT run. Nothing in this tree may be trained on until they do; "
                  "see the NOT_TRAINING_DATA marker beside the clips. G0c is not among them: "
                  "PR-08 §6 solves it BY CONSTRUCTION, the composite runs inside "
                  "restyle_transfer25.py before a unit may be called a success, and this job's "
                  "harvest VERIFIED that per clip before filing it — see g0c_evidence below.",
    "g0c": "enforced at generation time, per clip, in scripts/restyle_transfer25.py, and CHECKED "
           "here before filing. The robot-mask IoU recorded beside each clip is a DIAGNOSTIC ON "
           "THE GENERATOR AND NEVER A GATE (PR-08 §6, repeated in T40_RULE_V2 §0 and "
           "T40_RULE_V3 §1).",
    # The evidence, counted rather than claimed. One file per clip, written by harvest() out of that
    # unit's own sample_outputs.json and only after the checks listed inside it passed. The two
    # counts are recomputed above and disagreeing them refuses to write this record at all.
    "g0c_evidence": {
        "evidence_files": len(g0c_proofs),
        "clips": n_clips,
        "every_clip_has_evidence": len(g0c_proofs) == n_clips,
        "file": "<clips dir>/<unit>.g0c.json",
        "checked_before_filing": [
            "sample_outputs.json status == success",
            "backend == transfer25 (a placeholder clip is a valid mp4 and is refused by content)",
            "g0c.composited is true",
            "g0c.frames_composited == g0c.frames_total == the frames the work-list row declares",
            "g0c.area_bound.cross_checked (the bound belongs to the segmenter that made the masks)",
        ],
        "note": "harvest() refuses to file a clip failing any of these, and refuses the chunk if "
                "any clip in the directory has no evidence file — including one filed by an "
                "earlier driver, which is why absence of the sidecar is fatal rather than "
                "backfilled. A claim asserted rather than verified is not a construction.",
    },
}
(cd / "chunk_metadata.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
PY

echo
echo "=== chunk ${CHUNK_TAG} complete:  ${CLIPS}"
echo "    record: ${CHUNK_DIR}/chunk_metadata.json"
echo
echo "This is generated data and it is NOT training data. Two of PR-08 §6's three VOID gates —"
echo "G0a label identity and G0b geometry invariance — have not run."
echo "The NOT_TRAINING_DATA marker beside the clips stays until they do."
echo "G0c is the third and it is solved BY CONSTRUCTION, so it does not wait for a checker: the"
echo "real robot's pixels were composited back over every frame of every clip filed above, inside"
echo "the driver — and the harvest verified that from each unit's own record before filing it, so"
echo "every <unit>.mp4 there has a <unit>.g0c.json beside it and a clip without one is refused."
echo "The robot-mask IoU inside that file is a DIAGNOSTIC ON THE GENERATOR AND NEVER A GATE."
