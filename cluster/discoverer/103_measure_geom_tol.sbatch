#!/bin/bash
# Step 23 — T-040 / PR-08 §8 item 4: measure GEOM_TOL over the AppleToPlate source corpus.
#
# EVERY RECIPE BELOW CARRIES RUN_ID, AND IT IS NOT OPTIONAL PUNCTUATION. RUN_ID names the directory
# a submission reads and writes -- ${PROJ}/runs/${RUN_ID}, which holds the pilot artifact, the shard
# directory the array fills and the merge pools, and the transport copy sync.sh --pull takes home.
# It defaults to pr08-geom-tol, and THAT DEFAULT NOW POINTS AT THE SIXTEEN PERMANENTLY
# UNCOMMITTABLE SHARDS of the 2026-08-26/27 partition, so a re-run takes a fresh RUN_ID and every
# step of that re-run has to be given the SAME one. `<run>` below is that value, typed out.
# OMITTING IT ON THE MERGE LINE IS THE SINGLE MOST EXPENSIVE COPY-PASTE IN THIS FILE: a merge that
# falls back to the default pools the stale sixteen into configs/transfer25/pr08_geom_tol.json and
# writes a complete, refusal-passing, wrong number over the pre-commitment.
#
#   PILOT — measure the cost of the full run before paying it:
#       RUN_ID=<run> PILOT=1 sbatch --time=00:30:00 103_measure_geom_tol.sbatch
#
#   FULL — the committed number in ONE job. THE PILOT SAYS THIS DOES NOT FIT; see the arithmetic
#          below. Kept because it is the right shape on any machine whose wall is longer than the
#          run, and because the sharded path has to be checkable against something:
#       RUN_ID=<run> GEOM_STEP_FRAMES=1 sbatch --time=<from PILOT.json> \
#           103_measure_geom_tol.sbatch
#
#   SHARD — the committed number as an array of N jobs, each measuring the episodes that hash to it.
#           N = 16 AND --time=01:30:00 SINCE 2026-08-23; see RE-RUN SHAPE below for the arithmetic.
#           MaxSubmitJobsPU = 8 counts EVERY ARRAY TASK, so the sixteen go up in waves of at most
#           the free submit slots — `%4` throttles running tasks and does nothing for the submit
#           limit. One wave, repeated with the next four indices once the previous wave has left
#           the queue:
#       RUN_ID=<run> SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 \
#           sbatch --array=0-3%4 --time=01:30:00 103_measure_geom_tol.sbatch
#
#   MERGE — once every shard has landed, pool them into the committed artifact. NO GPU IS USED, so
#           it belongs on the free QoS, which needs its limits on the command line too:
#       RUN_ID=<run> MERGE=1 NUM_SHARDS=16 sbatch --qos=2cpu-single-host --gres=none \
#           --cpus-per-task=2 --mem=8G --time=00:20:00 103_measure_geom_tol.sbatch
#
# ON THE `%4`, WHICH IS ABOUT THE RUNNING LIMIT AND NOT THE SUBMIT LIMIT, AND ON THE MERGE
# OVERRIDES.
# The project QoS carries TWO limits and they are not the same limit. MaxJobsPU=4 caps RUNNING
# jobs, and `%4` is the throttle that respects it: it holds the array's own tasks to four at a
# time so Slurm does not have to reject the surplus under DenyOnLimit. MaxSubmitJobsPU=8 caps
# SUBMITTED (pending + running) jobs, AN ARRAY TASK COUNTS AS ONE EACH, AND `%4` DOES NOTHING FOR
# IT: the throttle is applied inside an array that has already been accepted, so `--array=0-15%4`
# is sixteen submissions and is rejected outright at sbatch time. Getting these two the wrong way
# round is what cost job 189658. The submit limit is respected by SUBMITTING IN WAVES; see RE-RUN
# SHAPE below. The MERGE line overrides four directives at once
# because they belong to the GPU path: 2cpu-single-host caps at cpu=2 and rejects a --gres, and the
# file asks for 26 threads, 32 GB and one GPU. If the override is rejected for any reason, the
# fallback is to submit MERGE with no overrides at all: it then runs under the project QoS and
# spends roughly twenty seconds of one H200 doing arithmetic, which is wasteful and correct. It is
# NOT correct to drop --qos to get around it; that lands on `normal`, which is 1 minute and 0 GPUs.
#
# WHY THE FULL RUN IS SHARDED, WITH THE ARITHMETIC RATHER THAN A CHOSEN NUMBER.
# The pilot (job 189588, ${OUT}/GEOM_TOL_PILOT.json) measured the full run at 4.005 GPU-h and wrote
# "single_job_feasible": false with "recommended_time_limit": "05:30:00". Every Discoverer+ QoS caps
# at MaxWall 04:00:00 (docs/discoverer.md §QoS) and the project QoS allows MaxJobsPU=4:
#
#     full_run_seconds     4.005 GPU-h x 3600                     = 14 418 s
#     MaxWall, every QoS                                            14 400 s
#                                                                   ------------------
#                          the measurement is over the wall BEFORE any margin, by 18 s. It is not
#                          a request that could be written more tightly; it is a run that does not
#                          fit the machine, which is what "single_job_feasible: false" means.
#
# Sharding pays the fixed startup L (interpreter, torch, and the ~3.7 GB of GroundingDINO + SAM2
# weights) once PER SHARD instead of once. The pilot separated L out precisely so this arithmetic
# can be done, and the SHARD path below re-reads BOTH numbers out of GEOM_TOL_PILOT.json at run time
# rather than trusting the constants written here:
#
#     per shard at N       (14 418 - L)/N + L    <- EXACT ONLY FOR AN EVEN SPLIT OF FRAMES
#     at N = 8, L = 120 s  (14 418 - 120)/8 + 120     = 1 787 + 120 = 1 907 s = 31 m 47 s
#
# THAT FORMULA ASSUMES A BALANCE THE PARTITION DOES NOT PROVIDE, so it is a floor and not the
# number to plan a --time against. Assignment is `blake2b(episode_key) % N`, which balances episode
# COUNTS only approximately and does not balance FRAMES at all — this corpus's episodes run 249-749
# frames. Running measure_geom_tol.shard_of over the 402 keys this corpus actually has
# (`episode_000000`..`episode_000401`) gives, at N = 8, shard sizes [46, 38, 58, 49, 54, 53, 56, 48]:
# the heaviest shard carries 58 of 402 episodes, 1.15x the mean of 50.25. Re-doing the arithmetic
# for THAT shard, which is the one that decides whether the array finishes:
#
#     heaviest at N = 8    (14 418 - 120) x 58/402 + 120            = 2 183 s = 36 m 23 s
#     pilot's own margin   x 1.25                                   = 2 729 s = 45 m 29 s
#     requested --time                                                3 600 s = 01:00:00
#                          -> 1.65x the heaviest-shard estimate, and the 4 h wall is 6.6x it. (The
#                             even-split figures were 1.89x and 7.5x; they were optimistic by ~15 %,
#                             and both are still comfortable, so nothing needs re-sharding.)
#     total cost           unchanged by imbalance, because the sum is N x L + (14 418 - L) either
#                          way: 8 x load + the corpus = 4.24 GPU-h. Sharding costs 0.13 GPU-h over
#                          N = 4, which is the four extra model loads, and that is the whole price.
#
# ------------------------------------------------------------------------------------------------
# MEASURED 2026-08-22 BY JOB 189658, AND EVERY ESTIMATE ABOVE IS ~2x OPTIMISTIC. READ THIS FIRST.
#
# The paragraph below this block says "a 2x error in that extrapolation is entirely plausible."
# It was 2x. Job 189658 ran --array=0-3%4 --time=02:00:00 against the H.264 corpus:
#
#     shard 0 = 42 673 frames (100 episodes) -> did NOT finish in 7 200 s
#                                            => p > (7200 - 116)/42673 = 0.1660 s/frame
#     shard 2 = 48 766 frames (114 episodes) -> cancelled; it could not have finished either
#
# So p >= 0.166 against the pilot's 0.0833: AT LEAST 1.99x. The cause is not contention and not
# the corpus. THE PILOT MEASURED A DIFFERENT SEGMENTER. GEOM_TOL_PILOT.json was produced on
# 2026-08-22 at box_threshold 0.35 with no retry branch; scripts/estimators/apple_sam2.py has
# since moved to Cosmos-Transfer2.5's OWN operating point, 0.15 with a single (0.10, 0.10) retry,
# because PR-08 section 4 step 2 requires our detection point to BE the generator's. A lower box
# threshold means more boxes survive detection, and every surviving box costs a SAM 2 pass.
# apple_sam2.GATE_QUALIFICATION_BLOCKERS[0] says this in as many words -- "it is also evidence
# about a configuration THIS FILE HAS SINCE REPLACED" -- and the array was sized off it anyway.
#
# USE p = 0.18 s/frame UNTIL SOMETHING RE-MEASURES IT AT THE CURRENT OPERATING POINT.
# SOMETHING HAS, AND IT WAS NOT A PILOT: the N = 16 partition itself ran on 2026-08-26/27 and its
# sixteen logs give p = 0.2478 s/frame with L = 410 s. The block under the table below derives that
# from the logs and is what this file's defaults now carry. Everything between here and it is the
# 2026-08-23 planning arithmetic, kept because the SHAPE it chose (N = 16 at 01:30:00) is the shape
# that ran and survived — only its cost model was wrong, and by 1.50x. Then:
#
#     full corpus          171 625 x 0.18                          = 30 893 s = 8.58 GPU-h
#                          (+ N model loads; at N = 8, 8.85 GPU-h)
#     heaviest at N = 8    24 546 x 0.18 + 120                      =  4 538 s = 75 m 38 s
#     that shard x 1.25                                             =  5 673 s = 94 m 33 s
#     requested --time                                              =  02:00:00
#                          -> 1.59x the heaviest-shard estimate, 2.1x to the 4 h wall.
#     heaviest at N = 4    48 766 x 0.18 + 120                      =  8 898 s = 148 m
#                          -> DOES NOT FIT THE 4 h WALL WITH ANY MARGIN. N = 4 IS NOT AN OPTION.
#
# AND THE REASON N = 4 WAS CHOSEN ANYWAY, so the next person does not repeat it: the QoS has
# MaxSubmitJobsPU = 8, an N-task array is N submissions, and two other jobs were already pending,
# so --array=0-7 was REJECTED outright with QOSMaxSubmitJobPerUserLimit. The fix is to submit the
# partition IN WAVES, NOT to shard coarser. Sharding coarser trades a scheduling constraint
# for a wall-clock cliff, and the wall does not negotiate: a shard that overruns loses everything
# it computed. Job 189658 cost roughly six GPU-hours and produced no artifact at all.
# ------------------------------------------------------------------------------------------------
# RE-RUN SHAPE, DERIVED 2026-08-23 FROM THE MANIFEST'S OWN PER-EPISODE FRAME COUNTS. N = 16.
#
# The frame counts are not estimated here. ${SOURCE}/manifest.json carries `episodes[].frames` for
# all 402 keys (sum 171 625, range 249-749, mean 426.93), and running measure_geom_tol.shard_of over
# those keys gives the heaviest shard EXACTLY. The reconstruction is checked against the machine:
# at N = 4 it reproduces job 189658's own shard 0 (100 episodes, 42 673 frames) and shard 2
# (114 episodes, 48 766 frames) to the frame.
#
#   N   heaviest shard   x mean    at p=0.2478 s/frame + L=410 s    x1.25      total GPU-h
#   --------------------------------------------------------------------------------------
#    4      48 766 f      1.137          03:28:14                 04:20:18       12.269
#    5      38 042 f      1.108          02:43:57                 03:24:56       12.383
#    6      32 326 f      1.130          02:20:20                 02:55:25       12.497
#    8      24 546 f      1.144          01:48:13                 02:15:16       12.725
#   10      21 265 f      1.239          01:34:40                 01:58:19       12.953
#   12      17 404 f      1.217          01:18:43                 01:38:24       13.181
#   16      14 162 f      1.320          01:05:19                 01:21:39       13.636
#   20      13 363 f      1.557          01:02:01                 01:17:32       14.092
#   24      10 677 f      1.493          00:50:56                 01:03:40       14.548
#
# THE COST MODEL IN THAT TABLE IS MEASURED (2026-08-27), AND THE (0.18, 120) TABLE IT REPLACES WAS
# 1.50x OPTIMISTIC. The heaviest-shard column and the x1.25 column moved with it; the frame counts
# and the imbalance ratios did not, because those come from the manifest and are unchanged.
#
# WHERE THE PAIR COMES FROM, SO THE NEXT READER CAN RE-DERIVE IT RATHER THAN TRUST IT. The N = 16
# partition RAN — four waves of four, jobs 189935, 189971, 189984 and 190125 — and each of the
# sixteen logs prints BOTH its own shard's exact frame count (the self-check's "N episodes, F of
# 171 625 frames" line) and its wall clock (the "shard i exited 3 after Ts" line). Sixteen (F, t)
# pairs are a cost model, not a planning constant:
#
#     for f in runs/_slurm_logs/geom-tol.1899*_*.out runs/_slurm_logs/geom-tol.190125_*.out; do
#       grep -o 'exited [0-9]* after [0-9]*s' "$f" | head -1; done \
#       | awk -F'[ s]' '{s+=$4} END {print NR, s, s/3600}'
#     -> 16   49 091 s   13.6364 GPU-h          <- the measured total; the old table said 9.115
#
#     least squares over the sixteen (F, t)   p = 0.2478 s/frame, L = 410 s
#     per-shard rate t/F                      0.2640 - 0.3189 s/frame, mean 0.2879
#
# The fit is not an approximation OF that total: least-squares residuals sum to zero, so the
# UNROUNDED fit — p = 0.247793, L = 410.219 — reproduces the observed 49 091 s exactly. The four
# significant figures this file actually carries are a rounding OF that fit and do not: 0.2478 x
# 171 625 + 16 x 410 = 49 089 s, 2 s under, 0.005 %. Said precisely rather than as "exactly",
# because a reader who checks the arithmetic and finds a 2 s gap has no way to tell a rounding from
# a mistake, and this file's whole claim on the operator is that its numbers check out. The old
# model was under by
# 4.52 GPU-h in total, and — which is the part that kills a job — under by 40 % on the shard that
# decides the array: the heaviest (index 5, 14 162 frames) took 3 741 s = 62.4 min against the old
# model's 00:44:29. A --time=01:00:00 request would have passed the self-check and died at the
# wall, which is the exact failure the self-check was written for after 189658.
#
# WHY NOT THE BLENDED 0.29. The mean per-shard rate is 0.2879 s/frame, but every shard's t ALREADY
# CONTAINS one load, so that mean has L folded into it. Pairing 0.2879 with L = 410 charges the
# load twice and over-estimates the heaviest shard by ~21 % — safe, and not honest. The two
# self-consistent pairs are (0.2478, 410) and (0.2879, 0); the fitted one is what this file uses,
# because L is a real fixed cost the sharding arithmetic above subtracts explicitly, and because it
# is the pair that reproduces the total.
#
# THE NUMBER THAT DECIDES THE SHAPE IS STILL THE BREAK-EVEN p -- the rate at which the heaviest
# shard exactly hits the request -- and it is now compared against a measurement rather than a floor:
#
#     N = 16, --time=01:30:00   break-even p = (5400 - 410)/14 162 = 0.3524 s/frame
#                              = 1.42x the fitted rate, and 1.11x the WORST per-shard rate observed
#                                (0.3189, shard 0). All sixteen fitted: the heaviest used 3 741 of
#                                5 400 s, 69 % of the request.
#     N =  8, --time=02:00:00   break-even p = (7200 - 410)/24 546 = 0.2766 = 1.12x the fitted rate
#     N =  5, --time=03:00:00   break-even p = (10800 - 410)/38 042 = 0.2731 = 1.10x
#
# So N = 16 at 01:30:00 is vindicated and the two coarser shapes are not: ~10 % of margin against a
# rate that varies by 21 % across shards is not margin. N = 16 costs 0.911 GPU-h more than N = 8
# (the eight extra model loads, 8 x 410 s) and it caps the loss from ONE dead shard at 65 minutes
# instead of 108.
#
# AND THE SUBMIT LIMIT IS WHY IT GOES UP IN WAVES RATHER THAN AS ONE ARRAY.
# MaxSubmitJobsPU = 8 and MaxJobsPU = 4 on the project QoS, both DenyOnLimit. AN N-TASK ARRAY IS N
# SUBMISSIONS: `--array=0-15%4` is sixteen against the submit limit and is rejected outright with
# QOSMaxSubmitJobPerUserLimit, exactly as `--array=0-7` was in 189658. `%4` CAPS RUNNING TASKS ONLY
# -- it is a throttle inside an array that has already been accepted, and it does nothing for the
# submit limit. That distinction is the whole reason 189658 dropped to N = 4, which traded a
# scheduling constraint for the wall-clock cliff that killed it.
#
# The fix is waves, not coarser shards. With F free submit slots, submit min(F, 4) indices at a
# time -- 4 is the ceiling worth using because MaxJobsPU = 4 caps RUNNING tasks anyway, so a fifth
# queued task buys queue position and no throughput. Four waves of four:
#
#     squeue -u "${USER}" -r -h -o '%i' | wc -l        # must be <= 4 before each wave
#     RUN_ID=<run> SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 \
#         sbatch --array=0-3%4   --time=01:30:00 <this file>
#     ... then 4-7, then 8-11, then 12-15, each after the previous wave has left the queue.
#
# NUM_SHARDS STAYS 16 IN EVERY WAVE. It is the denominator of the partition, not the size of the
# submission; the merge insists on all sixteen. The per-wave NOTE below says so.
#
# WHAT THE RUN BUYS, AND WHAT IT COULD NOT BUY UNTIL 2026-08-27.
# scripts/estimators/apple_sam2.py NOW SETS GATE_QUALIFIED = True. It stood at False from the day
# this file was written until 2026-08-27, when commit 13f0416 flipped it on the project owner's
# recorded decision (573c80d). Everything this paragraph used to assert about that False is kept
# below, in the tense it now belongs in, because the flag can go BACK — apple_sam2's own "WHAT
# WOULD TAKE IT BACK" paragraph names three findings that would return it to False — and because a
# waived run still meets every consequence of it.
#
# WHAT THE BLOCKERS DID. Until 2026-08-26 this paragraph said GATE_QUALIFICATION_BLOCKERS "still
# lead with NOBODY HAS LOOKED AT A MASK"; THAT STOPPED BEING THE REASON THEN. The human look was
# discharged on 2026-08-26 and the propagation difference on 2026-08-27, the tuple went EMPTY, and
# all seven discharged entries are kept verbatim in GATE_QUALIFICATION_DISCHARGED with the evidence
# that retired each one. AN EMPTY TUPLE WAS NOT PERMISSION, and this file said so while that was
# the live question: the module's own comment beside GATE_QUALIFIED states TWO preconditions and
# says an empty blocker tuple satisfies one of them. The other is a RECORDED DECISION on the
# residue those discharges carry forward — in particular blocker 2's residue (i): a failure the
# discharged blocker predicted, occurring by a route it did not predict, on 92 frames. THAT
# DECISION WAS MADE AND RECORDED, by the owner, on 2026-08-26/27, and 13f0416 is the flag moving
# with it. This file did not make it and does not predict the next one.
#
# WHAT THAT CHANGES HERE, CONCRETELY. All three conjuncts measure_geom_tol.sam2_method() ANDs into
# each shard's gate_qualified pass at HEAD — the flag, the weights the adapter names, and the
# committed contract — so a shard measured at HEAD records gate_qualified: true AND EXITS 0, the
# SHARD path's pre-GPU preflight lets the array start, and the merge takes its committable branch.
# These ~13.6 GPU-h therefore buy a measured, committed-contract-checked, COMMITTABLE GEOM_TOL,
# which is exactly what the sixteen shards already sitting under the default RUN_ID are not: they
# were measured under the old flag, gate_qualified is baked into each shard AT MEASUREMENT TIME and
# no merge re-derives it, so they are permanently uncommittable and this run is the re-measurement
# that replaces them. That same arithmetic is why the flag is a decision to take BEFORE an array
# goes in and not after — and since 2026-08-27 it is enforced rather than advised: the SHARD path's
# pre-GPU preflight refuses to start while the flag is False, before a frame is decoded. See THE
# PRE-GPU CONTRACT AND GATE PREFLIGHT in the SHARD branch below.
#
# The SHARD path below no longer treats that standing, corpus-independent disqualification as a
# failed shard: an artifact whose ONLY gate_disqualified_reason is the adapter's own flag LANDS,
# is skipped on re-submission, and the job exits 0. Before 2026-08-23 it exited 3, the operator
# saw FATAL, and the resume check refused to reuse it -- so every re-submission re-measured the
# whole partition and the array could never make progress at all. Nothing else about exit 3 is
# forgiven: a truncating flag, coverage under the floor, or an empty measurement is still FATAL.
# ------------------------------------------------------------------------------------------------
# SUPERSEDED BY THE RE-RUN SHAPE ABOVE, AND KEPT SO THE ERROR IS LEGIBLE. Everything from here to
# the next divider was written against the pilot's p = 0.0833 and is optimistic by at least 1.99x.
# Two of its claims are now false as well as optimistic: the SHARD branch below no longer prints an
# even-split figure re-read from the pilot (it computes THIS shard's exact frames from the manifest,
# and it refuses a pilot whose segmenter does not match the committed contract), and its closing
# sentence -- "N = 16 would double the load cost again for headroom nothing suggests is needed" --
# is exactly the reasoning 189658 disproved. N = 16 is what the corrected arithmetic chooses.
#
# WHY N = 8 AND NOT 4, WHICH WOULD BE ONE SCHEDULING ROUND.
#     even split at N = 4  (14 418 - 120)/4 + 120                   = 3 694 s = 61 m 34 s
#     heaviest at N = 4    (14 418 - 120) x 114/402 + 120           = 4 175 s = 69 m 35 s
#                          (shard sizes [100, 91, 114, 97]; 114 is 1.13x the mean of 100.5)
#     that shard x 1.25                                             = 5 218 s = 87 m
# Which does NOT fit a one-hour request — it would need --time 02:00:00 — and leaves 3.45x to the
# wall, against a pilot that extrapolated from THREE of 402 episodes at at most 180 of 249-749
# frames. A 2x error in that extrapolation is entirely plausible and a 4x one is not absurd. N = 8
# costs one extra scheduling round (MaxJobsPU=4, so 0-7%4 is two rounds of four, ~45 min each) and
# 0.13 GPU-h more than N = 4 (4 x 3 694 s = 14 776 s = 4.105 GPU-h against 8 x 1 907 s = 15 256 s =
# 4.238 GPU-h), and buys nearly twice the headroom on the shard that decides the outcome. N = 16
# would double the load cost again for headroom nothing suggests is needed.
#
# WHY A JOB ARRAY HERE, WHEN cluster/discoverer/README.md SAYS NOT TO USE ONE.
# That rule is specific and its reason does not reach this job. 55_train_i8_rung, 70 and 97 avoid
# arrays because an array task's interaction with `scontrol requeue` and SLURM_RESTART_COUNT is an
# open question (docs/discoverer.md open questions item 8) and those jobs carry a requeue rail whose
# failure mode is an unbounded loop that re-queues onto every free H200 until the allocation is gone.
# THIS JOB HAS NO REQUEUE RAIL: no --requeue, no USR1 trap, no restart counter, no MAX_RESTARTS. It
# reads video and writes one JSON file. If a task is preempted it is simply re-submitted, and it is
# idempotent — a shard whose artifact already exists and is gate-qualified is SKIPPED, which is the
# resumable shape 97 gets from its DONE stamps. The blast radius of the open question here is one
# shard re-running, so the array is bought for what it is good at (one submission, one throttle) at
# a cost the standing rule was not written about.
#
# WHY THE SHARDS CANNOT SIMPLY BE AVERAGED, WHICH IS THE ENTIRE REASON --merge EXISTS.
# GEOM_TOL is "the median per-step object-centroid displacement in the source clips" (PR-08 §6). A
# median does NOT decompose: the median of eight shard medians is a different statistic with the
# same units and a plausible magnitude, and on a park-then-transfer corpus the two differ. Nothing
# downstream re-derives GEOM_TOL, so that error would be permanent and invisible. Each shard
# therefore emits every per-step displacement it measured and --merge takes ONE median over the
# pool, rebuilt in the corpus's own enumeration order so the merged artifact is byte-identical to
# what a single un-sharded run would have written. measure_geom_tol.py refuses to merge a set of
# shards that is missing one, disagrees about the segmenter, the decoder, the grid or the step,
# splits on gate_qualified, or does not cover the corpus exactly once.
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
# 26 threads / 257 GB is the fair-share rate for ONE GPU (330 900 billing-h / 5 000 GPU-h = 66.18).
# Asking for more empties the billing counter before the GPU-hours and strands the rest of the
# allocation permanently. See cluster/discoverer/README.md rule 3. NEVER exceed 26 threads or
# 257 GB per GPU.
#SBATCH --cpus-per-task=26
# 32 G, NOT the 192 G this file asked for until 2026-08-23, and the change is a measurement and not
# a preference. Billing/min = GPUs x 1.0 + MemGB x 0.25 + Threads x 0.035714, so MEMORY DOMINATES:
# 192 G bills 1.0 + 48.0 + 0.93 = 49.93 units/wall-hour, 32 G bills 1.0 + 8.0 + 0.93 = 9.93. Job
# 189588 ran THIS estimator stack (GroundingDINO + SAM 2 + Depth-Anything, one frame at a time) and
# peaked at MaxRSS 4.75 GiB; 32 G is 6.7x that. 106_measure_robot_mask_area.sbatch is already at
# 32 G off the same measurement. Over the ~9.1 GPU-h the N = 16 partition costs, that is 455
# billing-h at 192 G against 91 at 32 G -- 365 billing-h saved, which at the 66.18 fair-share rate
# is 5.5 GPU-h of allocation runway, more than half of what the measurement itself costs. The 26
# threads stay: they are 0.93 of the ~9.93 and cutting them is a rounding error against the memory.
#SBATCH --mem=32G
# Sized for the PILOT. The full path needs the --time the pilot derives, passed on the command
# line — the same override 97's TIMING recipe carries, and for the same reason: `common` runs two
# nodes against ~40 pending jobs, so an honest short request backfills and a padded one waits.
#SBATCH --time=00:30:00
# THERE IS DELIBERATELY NO `#SBATCH --array` HERE, and it is not an oversight. A directive in this
# file applies to EVERY submission of it, so a hardcoded array would turn the PILOT and the MERGE
# into arrays as well — eight pilots racing for one GEOM_TOL_PILOT.json, and eight merges racing for
# the committed artifact. The array is passed on the command line, exactly the way --time already is,
# and the SHARD path below REFUSES to run when SLURM_ARRAY_TASK_ID is unset rather than quietly
# measuring shard 0 and calling it the corpus.
#SBATCH --open-mode=append
# %A_%a, NOT %j. For an array task %j is a per-task job id, but the SHARD path is the
# only one that ever runs as an array and job 189658 proved the cost of getting this
# wrong: four tasks wrote to one geom-tol.189658.out, so three headers were lost and the
# surviving one described a different shard than the progress lines under it. %A_%a keeps
# the array grouped and the tasks separate; a non-array run still gets %A as its own id.
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/geom-tol.%A_%a.out

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

# EVERY COMMAND THIS FILE PRINTS CARRIES RUN_ID=${RUN_ID}, FOR THE REASON THE HEADER GIVES. Until
# 2026-08-27 the NEXT and remedy lines below printed the bare recipe, so a copy-paste fell back to
# the default RUN_ID rather than reproducing the run the operator was reading the log of. A printed
# command that does not name the run it belongs to is worse than no command: it is right about the
# shape and wrong about the world, and on the MERGE line it pools the previous partition.
#
# AND THE NAME THEY PRINT IS NOT $0. Slurm copies the batch script into the job's spool directory
# and execs the copy, so under sbatch $0 is .../slurm_script and $(basename "$0") rendered the
# literal string "slurm_script" -- see runs/_slurm_logs/geom-tol.189935_1.out:277, where it did.
# An operator who pastes that gets "sbatch: error: Unable to open file slurm_script", which is a
# dead end at exactly the moment the log is being read for what to do next. Slurm exports no
# variable carrying the submitted path, so the spool copy is answered with this file's tracked
# name. Run directly, $0 IS a path someone can paste and is kept: printing a fixed name for a file
# that has been renamed or copied elsewhere would be the same lie in the other direction.
THIS_SCRIPT=$0
if [[ "$(basename "${THIS_SCRIPT}")" == "slurm_script" || ! -f "${THIS_SCRIPT}" ]]; then
  THIS_SCRIPT=103_measure_geom_tol.sbatch
fi

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
  echo "       into the committed number. Set exactly one of them, or none for the FULL path."
  exit 1
fi

# Where the shards live. One directory, one file per shard, so MERGE can be pointed at the directory
# and measure_geom_tol.py can skip whatever else has accumulated in ${OUT} (the pilot artifact, the
# decode probe, and after the first merge the merged artifact itself).
SHARD_DIR=${OUT}/shards

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

# ------------------------------------------------------------------------------------------------
# MERGE. Placed BEFORE the corpus and estimator checks on purpose: --merge reads shard artifacts and
# never opens a clip, never imports the adapter and never touches a GPU, which is exactly why it can
# run on the free QoS. Requiring the corpus here would tie the cheapest step in the chain to the
# most expensive preconditions for no reason. The shards each recorded the enumeration they saw and
# measure_geom_tol.py checks them against each other, which is a stronger coverage proof than
# re-globbing the directory would be.
if (( MERGE_MODE )); then
  # No apostrophes in this message. Inside ${var:?word} bash still parses quotes, so one lone
  # apostrophe opens a string that swallows the rest of the file — the bug this file shipped with
  # once already, on the GEOM_STEP_FRAMES line below.
  : "${NUM_SHARDS:?MERGE needs to know how many shards to insist on. Without it a merge over the shards that happen to have landed is a median over part of the corpus wearing the name of the whole, and it is indistinguishable from the real thing: right units, plausible magnitude, wrong number. Pass the same NUM_SHARDS the array was submitted with, for example NUM_SHARDS=8.}"
  [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} is not a positive integer."
    exit 1
  }

  [[ -d "${SHARD_DIR}" ]] || {
    echo "FATAL: ${SHARD_DIR} does not exist — no shard has run yet."
    echo "       RUN_ID=${RUN_ID} SHARD=1 NUM_SHARDS=${NUM_SHARDS} GEOM_STEP_FRAMES=1 \\"
    echo "         sbatch --array=0-$((NUM_SHARDS - 1))%4 --time=01:30:00 ${THIS_SCRIPT}"
    exit 1
  }

  # Counted here as well as inside the script. The script refuses on a missing shard with the full
  # argument; this says the same thing before a Python interpreter starts, which is what an operator
  # reading the log wants at the top of it rather than 40 lines down.
  MISSING=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    [[ -f "${SHARD_DIR}/shard-${i}.json" ]] || MISSING+=("${i}")
  done
  if (( ${#MISSING[@]} )); then
    echo "FATAL: shard artifact(s) missing from ${SHARD_DIR}: ${MISSING[*]} of ${NUM_SHARDS}."
    echo "       Re-submit just those array indices — the completed shards are skipped, so this"
    echo "       costs only the shards that did not land:"
    echo "         RUN_ID=${RUN_ID} SHARD=1 NUM_SHARDS=${NUM_SHARDS} GEOM_STEP_FRAMES=1 \\"
    echo "           sbatch --array=$(IFS=,; echo "${MISSING[*]}")%4 --time=01:30:00 ${THIS_SCRIPT}"
    exit 1
  fi

  ARTIFACT=${WAM}/configs/transfer25/pr08_geom_tol.json

  # THE MERGE WRITES OVER THE COMMITTED SEGMENTER CONTRACT'S OWN PATH, ON PURPOSE: PR-08 section 4
  # step 2 is checked by comparing the contract already in that file against the adapter the shards
  # ran, and measure_geom_tol.py refuses the merge outright if it is absent or if it disagrees.
  # Checked here as well, before an array's worth of GPU time is spent on a merge that will be
  # refused: the tracked file only goes missing when this checkout is stale or someone deleted it.
  if ! CONTRACT_FILE="${ARTIFACT}" python - <<'PY'
import json, os
path = os.environ["CONTRACT_FILE"]
try:
    doc = json.loads(open(path).read())
except Exception as exc:
    print("    " + str(exc))
    raise SystemExit(1)
params = (doc.get("mask_method") or {}).get("params") or {}
block = doc.get("segmenter") or params.get("segmenter")
if not isinstance(block, dict):
    print("    no top-level 'segmenter' block and no mask_method.params.segmenter")
    raise SystemExit(1)
print("    contract present: %s box_thr=%s grid=%s"
      % (block.get("method_name"), block.get("box_threshold"), block.get("pixel_grid_hw")))
PY
  then
    echo "FATAL: ${ARTIFACT} carries no committed segmenter contract (see above)."
    echo "       That file is the PR-08 section 4 step 2 pre-commitment - the detector, the"
    echo "       segmenter, the depth model, their pinned revisions, the prompt, both thresholds"
    echo "       and the pixel grid, written down BEFORE the number. measure_geom_tol.py --merge"
    echo "       refuses to write that path without it, so this merge would fail after reading"
    echo "       every shard."
    echo "       RECOVERY, AND IT IS NOT A GIT COMMAND. This message used to say"
    echo "       'git -C ${WAM} checkout -- configs/...', which CANNOT RUN HERE:"
    echo "       cluster/discoverer/sync.sh:66 rsyncs the repo with --exclude '.git', so"
    echo "       ${WAM} is a copy of a working tree and not a repository — the"
    echo "       command fails with 'not a git repository' at exactly the moment the"
    echo "       pre-commitment has been lost. Re-send the tracked file from the workstation,"
    echo "       which is the only machine that has the history:"
    echo "         ./cluster/discoverer/sync.sh"
    echo "       (Run it BY A HUMAN from the workstation checkout. It also re-stamps"
    echo "       ${WAM}/GIT_COMMIT, which is how the run records what code it ran.)"
    exit 1
  fi

  echo "=== MERGE: ${NUM_SHARDS} shard(s) from ${SHARD_DIR} -> ${ARTIFACT}"
  echo "===        The median is taken ONCE over the pooled per-step displacements. Shard medians"
  echo "===        are never averaged: a median does not decompose, and averaging them would put a"
  echo "===        plausible wrong tolerance behind PR-08 §6 G0b that nothing downstream re-derives."
  # `set -e` is on, so a bare invocation followed by RC=$? never sees a non-zero status — the shell
  # is already gone. The `if` is where set -e does not fire and the exit code survives.
  if python "${WAM}/scripts/measure_geom_tol.py" \
    --merge "${SHARD_DIR}" \
    --out "${ARTIFACT}"; then RC=0; else RC=$?; fi
  echo "=== measure_geom_tol.py --merge exited ${RC}"
  # EXIT 3 IS NOT A FAILED MERGE. It means the pool was complete, every refusal passed, the median
  # was taken over the whole corpus and the RESULT is disqualified from the gate — which is what
  # apple_sam2.GATE_QUALIFIED=False guaranteed for every possible run of this partition, until
  # 2026-08-27, when 13f0416 flipped that flag on. AT HEAD A MERGE OVER SHARDS MEASURED AT HEAD
  # EXITS 0 AND TAKES THE COMMITTABLE BRANCH BELOW; this one is reached by a pool that is older
  # than the flag — the sixteen shards under the default RUN_ID are — or by a waived run, and it
  # stays because such a pool still has to be classified rather than silently copied. The
  # artifact IS written by measure_geom_tol.py in that case, and this branch used to exit before
  # copying it, so nine GPU-hours of measurement stayed on the cluster where sync.sh --pull (which
  # only walks ${PROJ}/runs) could not see it. Transport it and refuse to call it committable; the
  # exit code stays 3 so nothing automated reads this as a pass.
  if (( RC == 3 )) && [[ -f "${ARTIFACT}" && -f "${ARTIFACT}.sha256" ]]; then
    cp "${ARTIFACT}" "${ARTIFACT}.sha256" "${OUT}/"
    echo "=== copied the DISQUALIFIED artifact + sidecar to ${OUT}/ for sync.sh --pull"
    echo "=== DO NOT COMMIT IT AS GEOM_TOL. Exit 3 means the merge completed and the number carries"
    echo "===   gate_qualified:false. Read gate_disqualified_reasons in the artifact: if the only"
    echo "===   entries are 'mask method ... is not gate-qualified', the corpus was measured"
    echo "===   correctly and what was missing is the adapter's own opt-in AT THE TIME THESE"
    echo "===   SHARDS WERE MEASURED. That opt-in has since been given: apple_sam2's flag was"
    echo "===   flipped on on 2026-08-27 by commit 13f0416, on the project owner's recorded"
    echo "===   decision about blocker 2's residue (i) — NOT because 'nobody has looked at a"
    echo "===   mask', which was discharged 2026-08-26 and left GATE_QUALIFICATION_BLOCKERS the"
    echo "===   empty tuple. So reaching this branch at HEAD means the pool is older than the flag"
    echo "===   (the sixteen shards under the default RUN_ID are), or the cluster copy lags that"
    echo "===   commit, or the run was waived. There is no repair either way: gate_qualified is"
    echo "===   baked into each shard at measurement time and no merge re-derives it, so the"
    echo "===   corpus is RE-MEASURED under a fresh RUN_ID and merged again."
    echo "===   Restore the pre-commitment before anything else reads that path. NOT with git:"
    echo "===   sync.sh:66 excludes '.git', so ${WAM} is not a repository and"
    echo "===   'git -C ... checkout --' fails there with 'not a git repository'. From the"
    echo "===   workstation checkout, by a human:"
    echo "===     ./cluster/discoverer/sync.sh"
    exit "${RC}"
  fi
  if (( RC != 0 )); then
    echo "FATAL: the merge exited ${RC} and whatever is at ${ARTIFACT} is not GEOM_TOL."
    echo "       Exit 2 is a refusal — a missing shard, shards that disagree about the segmenter,"
    echo "       the decoder, the pixel grid or the step, a gate_qualified split between shards, or"
    echo "       a union of episodes that is not the corpus exactly once. Nothing was written in"
    echo "       that case. Exit 3 means the merge completed and the number is disqualified from"
    echo "       the gate — handled above when the artifact landed. Read the refusal; it names"
    echo "       which one and why."
    exit "${RC}"
  fi

  [[ -f "${ARTIFACT}.sha256" ]] || {
    echo "FATAL: ${ARTIFACT}.sha256 was not written. 97 refuses a GEOM_TOL with no committed digest."
    exit 1
  }
  cp "${ARTIFACT}" "${ARTIFACT}.sha256" "${OUT}/"
  echo "=== copied artifact + sidecar to ${OUT}/ for sync.sh --pull"
  echo "=== NEXT: commit configs/transfer25/pr08_geom_tol.json AND its .sha256 in the repo. The"
  echo "===       committed SEGMENTER CONTRACT that was already in that file is inside this artifact:"
  echo "===       measure_geom_tol.py compared it field for field against the adapter this run drove"
  echo "===       and would have refused the whole run, writing nothing, on any disagreement. The"
  echo "===       diff you are committing therefore ADDS the measurement to the contract; it must not"
  echo "===       change spec_version, what_this_is or the segmenter block. If it does, do not commit"
  echo "===       it — that is the pre-commitment PR-08 §4 step 2 is checked against."
  echo "===       Then re-run 102_stage_sam2_weights.sbatch: its <role>-vs-pr08_geom_tol.json check"
  echo "===       reads mask_method.params.checkpoints, which only a MEASURED artifact carries."
  exit 0
fi

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
            "False means the full measurement does not fit one comfortable request on `common` "
            "and has to be sharded. measure_geom_tol.py HAS that mode as of 2026-08-22: "
            "--shard I --num-shards N per array task, then --merge over the shard artifacts, "
            "which pools the raw per-step displacements and takes ONE median (a median does not "
            "decompose, so shard medians are never averaged). Submit it with SHARD=1 on this "
            "file; the shard-count arithmetic is in its header.",
    }

path = out / "GEOM_TOL_PILOT.json"
path.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
print(f"\n=== wrote {path}")
PY
  echo "=== PILOT done. Read ${OUT}/GEOM_TOL_PILOT.json, then submit the full run with the"
  echo "===   --time it recommends, and into THIS run, whose pilot artifact that --time came from:"
  echo "===     RUN_ID=${RUN_ID} GEOM_STEP_FRAMES=1 sbatch --time=<hh:mm:00> ${THIS_SCRIPT}"
  exit 0
fi

# -- SHARD ---------------------------------------------------------------------------------------
if (( SHARD_MODE )); then
  # SUBMITTED WITHOUT --array, THIS WOULD MEASURE SHARD 0 AND STOP. Everything downstream would be
  # well formed: one gate-qualified shard artifact, and a merge that refuses because seven are
  # missing — which is the right refusal in the wrong place, an hour of H200 later. Catch it here.
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    echo "FATAL: SHARD=1 but this job is not an array task (SLURM_ARRAY_TASK_ID is unset)."
    echo "       The shard index IS the array index; there is no default, because the default would"
    echo "       be shard 0 and a run that measured a twelfth of the corpus and exited 0."
    echo "       Submit it as an array:"
    echo "         RUN_ID=${RUN_ID} SHARD=1 NUM_SHARDS=16 GEOM_STEP_FRAMES=1 \\"
    echo "           sbatch --array=0-3%4 --time=01:30:00 ${THIS_SCRIPT}"
    exit 1
  }
  # No apostrophes in this message; see the note on the MERGE branch and on GEOM_STEP_FRAMES below.
  : "${NUM_SHARDS:?SHARD needs the denominator of the partition. It is not derived from the array bounds, because the array can legitimately be a subset (re-running two failed shards is --array=2,5) while the partition it belongs to still has N pieces, and a merge cannot say a shard is missing without knowing N. Pass the same NUM_SHARDS for every submission of one partition, for example NUM_SHARDS=8.}"
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
    echo "       0-$((NUM_SHARDS - 1)) whenever NUM_SHARDS=${NUM_SHARDS}. A task above the range would"
    echo "       measure the empty set and report it as a clean run."
    exit 1
  fi
  # Not fatal, and it must not be: an array narrower than the partition is the NORMAL shape here,
  # for two reasons. Re-running two failed shards is --array=2,5 over the same partition. And
  # MaxSubmitJobsPU=8 counts every array task, so a partition wider than the free submit slots has
  # to go up in waves of four — `%4` throttles RUNNING tasks and does nothing for the submit limit.
  # Say it out loud anyway, because the merge insists on all NUM_SHARDS and an operator who submits
  # ONE wave and stops should learn that here and not in the merge job.
  if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "${NUM_SHARDS}" ]]; then
    echo "=== NOTE: this array has ${SLURM_ARRAY_TASK_COUNT:-?} task(s) and the partition has"
    echo "===       ${NUM_SHARDS}. That is EXPECTED for a wave of a wave-submitted partition and for"
    echo "===       a re-run of specific shards; it is wrong only if this is meant to be the whole"
    echo "===       partition. The merge insists on all ${NUM_SHARDS} shard artifacts."
  fi
  # Same discipline as the FULL path, and now load-bearing across jobs: measure_geom_tol.py --merge
  # REFUSES a set of shards that disagree on step_frames, because GEOM_TOL scales ~linearly with it
  # and two steps pooled into one median is a tolerance that is proportionally wrong against either.
  # (NOTE: no apostrophes in this message. Inside ${var:?word} bash still parses quotes, so one lone
  # apostrophe here opens a string that swallows the rest of the file -- which is exactly how the
  # FULL path line first shipped broken.)
  : "${GEOM_STEP_FRAMES:?PR-08 §6 never defines what a step is, and GEOM_TOL scales ~linearly with it, so a default here would not pick a convention -- it would set the G0b tolerance silently, by a factor of the step. PR-08 V3 §4 REGISTERS the value: GEOM_STEP_FRAMES=1. Type the registered value, and type the SAME one for every shard of a partition: the merge refuses shards that disagree about it.}"
  [[ "${GEOM_STEP_FRAMES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: GEOM_STEP_FRAMES=${GEOM_STEP_FRAMES} is not a positive integer."
    exit 1
  }

  mkdir -p "${SHARD_DIR}"
  SHARD_OUT=${SHARD_DIR}/shard-${SHARD_INDEX}.json

  # WHETHER A SHARD ARTIFACT ON DISK COUNTS AS LANDED. One definition, used in two places: before
  # the measurement (skip a shard that is already done) and after it (decide whether exit 3 was a
  # failure or the standing, corpus-independent disqualification described in the header). Two
  # copies of this rule would eventually disagree, and the direction they would disagree in is a
  # re-measured partition or a bad shard reaching the merge.
  #
  # The stamp is the artifact itself and it is VALIDATED, not merely present: a truncated file from
  # a task killed at the wall must not be mistaken for a finished one, and neither must a shard
  # artifact for a different index, a different partition or a different step.
  #
  # ON gate_qualified. Refusing every artifact with gate_qualified:false was correct against a
  # DATA-dependent disqualification and catastrophic against the one this corpus actually had.
  # scripts/estimators/apple_sam2.py set GATE_QUALIFIED=False until 2026-08-27 (13f0416), so EVERY
  # shard of a perfectly good measurement recorded exactly one reason -- "mask method ... is not
  # gate-qualified" -- and the old check therefore refused to reuse ANY shard, ever: each
  # re-submission re-measured the whole partition and the array could never converge. That reason
  # is a property of the ADAPTER, is identical across all N shards, and is re-derived by the merge
  # from the same artifact, which still refuses to write a gate-qualified GEOM_TOL from it. Reusing
  # such a shard therefore launders nothing. Every OTHER reason -- a truncating flag, coverage under
  # the floor, an empty measurement -- is data-dependent, means THIS shard measured the wrong thing,
  # and is still refused. headline_valid and the two truncating flags are checked directly rather
  # than inferred from the reason strings, so a reason this rule does not recognise cannot be waved
  # through.
  #
  # AND THAT WHOLE PARAGRAPH IS TRUE ONLY WHILE THE WORLD AROUND THE SHARD HAS NOT MOVED, which is
  # the defect the two staleness checks inside the heredoc repair. "The adapter's standing flag,
  # not this shard" stops being a reason to reuse the moment the adapter's flag flips: re-measuring
  # would then produce a QUALIFIED shard, so skipping is no longer free, it is the one action that
  # keeps a permanently disqualified artifact in the partition. And nothing here ever compared the
  # shard's own segmenter block against the committed pre-commitment, so a shard measured by a
  # different adapter was silently reused -- which is exactly how sixteen artifacts missing
  # mask_validity_reference_max_frame_fraction survived a resubmission and were discovered at the
  # merge, an array and 13.6 GPU-h later. Both checks only ever cause MORE work to be done.
  shard_artifact_landed () {  # $1 path $2 index $3 num_shards $4 step -> 0 when it counts as landed
    CONTRACT_FILE="${WAM}/configs/transfer25/pr08_geom_tol.json" \
      python - "$1" "$2" "$3" "$4" <<'PY'
import importlib, json, os, re, sys
path, want_i, want_n, want_step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try:
    d = json.loads(open(path).read())
except Exception as exc:
    print(f"    not reusable: {path} does not parse ({exc})"); raise SystemExit(1)
sh = d.get("shard") or {}
for label, got, want in (("schema", d.get("schema"), "wam.geom_tol_shard/1"),
                         ("shard.index", sh.get("index"), want_i),
                         ("shard.num_shards", sh.get("num_shards"), want_n),
                         ("step_frames", d.get("step_frames"), want_step)):
    if got != want:
        print(f"    not reusable: {label} is {got!r}, wanted {want!r}"); raise SystemExit(1)

# -- WHAT THE FOUR CHECKS ABOVE CANNOT SEE -------------------------------------------------------
#
# Every one of them asks the artifact about itself, and a shard can be internally perfect and still
# be one that could never be committed. Both of the ways that happens are properties of the WORLD
# OUTSIDE the file -- the adapter's standing gate flag, and the committed pre-commitment the merge
# will hold the artifact against -- so no field of the shard can answer either of them.
#
# BOTH CHECKS ARE FAIL-CLOSED, AND THAT DIRECTION IS THE WHOLE DESIGN: anything they cannot
# establish is a refusal, so the most either can ever cost is one shard re-measured for nothing,
# while what they prevent is a partition of permanently uncommittable artifacts pooled into a
# median. A resume check that reuses too little is an inconvenience; one that reuses too much is
# 13.6 GPU-h and a number nobody may commit.
#
# THE TWO NAMES ARE RESOLVED THE WAY THE PREFLIGHT BELOW RESOLVES THEM, deliberately and not
# incidentally: measure_geom_tol off the PYTHONPATH this file exported, the adapter through THAT
# module's own SAM2_ADAPTER_SPEC rather than a literal spelled here, and the committed document
# through ${CONTRACT_FILE} on this heredoc's own invocation. One idiom in the file and not two. A
# reader and a writer that each keep their own idea of where the segmenter block lives is precisely
# how this repo has been burned before -- measure_est_drift.py carries the comments that say so.
try:
    mgt = importlib.import_module("measure_geom_tol")
    SPEC = getattr(mgt, "SAM2_ADAPTER_SPEC", None)
    if not isinstance(SPEC, str) or not SPEC:
        raise ImportError("measure_geom_tol.py declares no SAM2_ADAPTER_SPEC to name the adapter")
    adapter = importlib.import_module(SPEC)
except Exception as exc:
    print(f"    not reusable: the adapter and the committed contract CANNOT BE CHECKED from here"
          f" ({type(exc).__name__}: {exc}).")
    print("    That is a broken or half-synced cluster copy and not a verdict about this shard, but")
    print("    an unchecked shard is not a checked one: it is re-measured rather than pooled.")
    raise SystemExit(1)

# CHECK 1 needs this and asks for it here, once, so that the branch below reads as the one sentence
# it is about: is the adapter's flag STILL what it was when this artifact was written?
adapter_gate_qualified = bool(getattr(adapter, "GATE_QUALIFIED", False))

# -- CHECK 2: THE CONTRACT WENT STALE ------------------------------------------------------------
#
# The comparison is measure_geom_tol's own committed_segmenter_contract() and
# contract_disagreements(), IMPORTED. Both sides are looked up by the same function, so this reader
# cannot develop its own idea of where a segmenter block lives -- and absence counts as a
# disagreement there, which is the entire reason this check catches the sixteen shards that carry
# no mask_validity_reference_max_frame_fraction while the committed document does. That field is
# not decoration: it is a frame-refusal predicate (apple_sam2.py reference_is_object_scale), so an
# artifact that does not record it was measured by a segmenter that refused a different set of
# frames than the committed one describes.
CONTRACT_FILE = os.environ.get("CONTRACT_FILE") or ""
try:
    with open(CONTRACT_FILE, encoding="utf-8") as handle:
        committed_doc = json.load(handle)
except Exception as exc:
    print(f"    not reusable: the committed pre-commitment at {CONTRACT_FILE!r} cannot be read"
          f" ({type(exc).__name__}: {exc}), so there is nothing to hold this shard against.")
    print("    Reusing a shard nothing can be compared to is how an unmergeable partition survives")
    print("    to the merge; re-measuring one costs a shard.")
    raise SystemExit(1)
committed, committed_where = mgt.committed_segmenter_contract(committed_doc)
recorded, recorded_where = mgt.committed_segmenter_contract(d)
if committed is None:
    print(f"    not reusable: {CONTRACT_FILE} carries no segmenter block, so PR-08 §4 step 2's")
    print("    'the same segmenter' is uncheckable here -- and an uncheckable requirement reads")
    print("    downstream exactly like a satisfied one.")
    raise SystemExit(1)
if recorded is None:
    print("    not reusable: this shard records no segmenter block of its own, so nothing says")
    print("    which segmenter produced it. The merge compares the same block and refuses.")
    raise SystemExit(1)
drift = mgt.contract_disagreements(recorded, committed)
if drift:
    print("    not reusable: THIS SHARD WAS MEASURED UNDER A DIFFERENT SEGMENTER CONTRACT than the")
    print(f"    committed one ({recorded_where} here, {committed_where} in {CONTRACT_FILE}):")
    for item in drift:
        print(f"      {item['field']}: committed {item['geom_tol']!r},"
              f" this shard {item['this_run']!r}")
    print("    The merge holds every shard against that same document and refuses on the same")
    print("    fields, so keeping this one buys an array of GPU-hours and then a refusal. It is")
    print("    re-measured by the adapter this job actually runs.")
    raise SystemExit(1)

# The exact string measure_geom_tol.main() appends for an adapter that has not opted in. Anchored
# at both ends so a longer sentence that merely starts this way is not swallowed.
ADAPTER_BLOCKER = re.compile(r"^mask method .* is not gate-qualified$")
if not d.get("gate_qualified"):
    reasons = list(d.get("gate_disqualified_reasons") or [])
    unrecognised = [r for r in reasons if not ADAPTER_BLOCKER.match(str(r))]
    blockers = []
    if not reasons:
        blockers.append("gate_qualified is false and no reason is recorded")
    if unrecognised:
        blockers.append("data-dependent disqualification(s): " + "; ".join(unrecognised))
    if not d.get("headline_valid"):
        blockers.append("headline_valid is false")
    if d.get("partial_measurement"):
        blockers.append("partial_measurement is true")
    if d.get("limit") or d.get("max_frames"):
        blockers.append(f"limit={d.get('limit')!r} max_frames={d.get('max_frames')!r}")
    if not d.get("n_steps_measured"):
        blockers.append("n_steps_measured is 0")
    if blockers:
        print("    not reusable: " + " | ".join(blockers))
        raise SystemExit(1)
    # -- CHECK 1: THE GATE FLAG WENT STALE ------------------------------------------------------
    #
    # "That is the adapter's standing flag and not this shard" is a reason to reuse ONLY while the
    # flag still stands where it stood at measurement time. It was correct while apple_sam2 set
    # GATE_QUALIFIED=False: re-measuring would have produced another unqualified shard, so the
    # skip cost nothing. The moment the owner flips that flag the sentence inverts -- re-measuring
    # now produces a QUALIFIED shard, and skipping is the one action that keeps a permanently
    # disqualified artifact in the partition and pools it into a median that can never be
    # committed as GEOM_TOL. gate_qualified is BAKED INTO EACH SHARD AT MEASUREMENT TIME and no
    # merge re-derives it, so there is no later repair: the only way to get a qualified artifact
    # for this index is to measure one.
    if adapter_gate_qualified:
        print("    not reusable: THE ADAPTER'S STANDING FLAG HAS MOVED SINCE THIS SHARD WAS")
        print(f"    MEASURED. {SPEC}.GATE_QUALIFIED is True today, and this artifact permanently")
        print("    records the opposite: " + "; ".join(reasons))
        print("    gate_qualified is baked in at measurement time and no merge re-derives it, so")
        print("    reusing this shard would pool a median that can never be committed as GEOM_TOL")
        print("    -- while RE-MEASURING it now produces a qualified one. Re-measuring.")
        raise SystemExit(1)
    print("    LANDED BUT NOT GATE-QUALIFIED, and that is the adapter's standing flag and not this")
    print("    shard: " + "; ".join(reasons))
    print("    The measurement is complete and reusable. The merge will pool it and will write the")
    print("    pooled median DISQUALIFIED, which is the correct outcome while apple_sam2 sets")
    print("    GATE_QUALIFIED=False. It must not be committed as GEOM_TOL.")
print(f"    reusable: {sh.get('n_episodes_in_shard')} episodes, "
      f"{d.get('n_steps_measured')} steps, median {d.get('shard_median_px')} px")
raise SystemExit(0)
PY
  }

  # RESUMABLE, in the shape 97 gets from its DONE stamps: a shard that already landed is not
  # re-measured. This is what makes "re-submit the two that failed" cost two shards rather than a
  # partition, and it is what makes a preempted array task free to simply re-run.
  if [[ -f "${SHARD_OUT}" && "${FORCE:-0}" == "0" ]]; then
    if shard_artifact_landed "${SHARD_OUT}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${GEOM_STEP_FRAMES}"
    then
      echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} already landed. Skipping."
      echo "===   FORCE=1 re-measures it."
      exit 0
    fi
    echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS}: ${SHARD_OUT} exists and is NOT reusable (above)."
    echo "===   Re-measuring. The old file is overwritten, so no stale shard can reach the merge."
  fi

  echo "=== SHARD ${SHARD_INDEX} of ${NUM_SHARDS}, step ${GEOM_STEP_FRAMES} frame(s) -> ${SHARD_OUT}"
  echo "===   Episodes are assigned by a blake2b digest of the episode KEY, not by a slice of the"
  echo "===   episode list: shards are computed by different jobs at different times, and a range"
  echo "===   would silently re-partition the corpus the moment a clip is added or removed."

  # -- THE PRE-GPU CONTRACT AND GATE PREFLIGHT ----------------------------------------------------
  #
  # ADDED 2026-08-27, AND IT IS THE MOST EXPENSIVE THING THIS FILE HAS EVER GOT WRONG.
  #
  # WHAT WAS MISSING. Nothing on this path compared the adapter this job is about to run against
  # ${WAM}/configs/transfer25/pr08_geom_tol.json — the PR-08 §4 step 2 pre-commitment — BEFORE the
  # GPU work. measure_geom_tol.py does compare, twice, and both comparisons are too late to save
  # anything:
  #
  #   * on the SHARD path its --out is ${SHARD_OUT}, so merge_committed_contract() treats a stale
  #     shard artifact sitting there as "the committed contract". That comparison runs AFTER the
  #     decode. A FORCE=1 re-measure into a populated shard directory therefore decodes this
  #     shard's ~11 000 frames, then exits 2, then writes nothing — and the refusal blames the
  #     operator's segmenter for disagreeing with a file this job produced itself;
  #   * the real cross-check against the committed document happens at the MERGE, which is a
  #     separate submission an array later.
  #
  # WHAT THAT COST, MEASURED. TWELVE of the last sixteen shards ran a pre-6a32143 adapter, because
  # the cluster copy lagged HEAD and nothing on this path could see it; and ALL SIXTEEN lack
  # mask_validity_reference_max_frame_fraction, which the committed document has. The result is a
  # whole array of shards that are internally consistent, mutually mergeable, and unusable: the
  # merge refuses them against the committed contract with one field named
  # (contract_disagreements(landed, committed) -> [{'field':
  # 'mask_validity_reference_max_frame_fraction', 'geom_tol': 0.1, 'this_run': None}]), an array
  # and 13.6 GPU-h later.
  #
  # WHAT IT DOES. Three refusals, all before a frame is decoded, all naming a different remedy:
  #
  #   1. CONTRACT DRIFT — the adapter's live SEGMENTER_CONTRACT against the committed document.
  #      The comparison is measure_geom_tol.contract_disagreements(), IMPORTED, not reimplemented:
  #      a second spelling of "the same segmenter" is exactly the kind of copy that drifts, and the
  #      direction it drifts in is a preflight that passes a run the merge will refuse. Both
  #      imports are resolved EXACTLY THE WAY THE MEASUREMENT RESOLVES THEM: measure_geom_tol.py
  #      off the PYTHONPATH this file exported above, and then estimators.apple_sam2 the way that
  #      module's own _import_sam2_adapter() reaches it (importing measure_geom_tol first puts its
  #      directory at the head of sys.path, which is why the order here is not arbitrary). A
  #      preflight that passes therefore cannot be reading a different adapter than the run — and
  #      it also proves the adapter imports at all, which the measurement would otherwise discover
  #      on frame 0 with the GPU already allocated.
  #   2. A STALE SHARD ARTIFACT AT --out — the same comparison against whatever is already at
  #      ${SHARD_OUT}, because on this path THAT file is what measure_geom_tol.py will read as the
  #      committed contract. This is the FORCE=1 case above, refused for 0 GPU-seconds instead of
  #      after the decode. It is deliberately NOT fixed by deleting the file: that stale artifact is
  #      the only in-situ evidence that this shard was measured by a different adapter, and an
  #      `rm -f` here would silence the one detector this path has.
  #   3. THE SHARD WOULD RECORD gate_qualified: false — refused, and this is the one that is not
  #      obvious. gate_qualified is BAKED INTO EACH SHARD AT MEASUREMENT TIME and no merge
  #      re-derives it, so a partition started before it can be true produces sixteen artifacts
  #      that are permanently disqualified however the decision goes afterwards. The full 13.6
  #      GPU-h buys a number that cannot be committed. The header says "that is a decision to take
  #      BEFORE the array goes in"; this is that sentence, enforced. Note that what the shard
  #      records is not the flag alone: sam2_method() writes `GATE_QUALIFIED and bool(checkpoints)
  #      and contract is not None`, so ALL THREE conjuncts are checked here — the flag, the weights
  #      the adapter declares (through measure_geom_tol's own _adapter_checkpoints()), and the
  #      contract, which is refusal 1. Checking the flag alone would print PREFLIGHT OK over two
  #      thirds of the condition and buy the same disqualified number.
  #
  # THE REFUSAL MUST NOT BE READ AS "EDIT THE FLAG". apple_sam2.GATE_QUALIFIED is the project
  # owner's to move, in its own commit, once the residue blocker 2 carries forward is decided on the
  # record — and it HAS moved: 13f0416 flipped it on on 2026-08-27, after that decision was
  # recorded on 2026-08-26/27. So refusal 3 does not fire at HEAD at all; if it fires, the flag has
  # gone BACK, and apple_sam2's own "WHAT WOULD TAKE IT BACK" paragraph is what to read for which
  # finding did it. The module's own comment and CLAUDE.md both say no session may make that edit,
  # in either direction. The message below names the decision and does not name a diff.
  #
  # THE WAIVER IS ONE ENV VAR AND IT IS LOUD. GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT=1 downgrades
  # all three refusals to warnings — for the case where an operator is knowingly measuring
  # something that is not going to be committed. It waives the CHECKS, never the two imports: a
  # measure_geom_tol.py or an adapter that will not import is a broken cluster copy, the
  # measurement itself cannot run, and there is nothing to waive.
  #
  # WHAT THIS DOES NOT TOUCH, SO THE NEXT READER DOES NOT GO LOOKING. It is on the SHARD path only:
  # the PILOT deliberately produces a disqualified sample and the FULL path is not the shape this
  # corpus is measured in. It also sits AFTER the resume check above, so a partition that has
  # already landed is still skipped for free rather than being refused for a flag its artifacts
  # already record. And one downstream consequence worth naming: while GATE_QUALIFIED is False,
  # the "exit 3 is the adapter's standing flag, so this shard LANDED" branch further down is now
  # unreachable except under the waiver — that branch is kept because it is about the artifact,
  # not about the flag, and a waived run still has to classify what it wrote.
  if ! CONTRACT_FILE="${WAM}/configs/transfer25/pr08_geom_tol.json" \
       SHARD_OUT="${SHARD_OUT}" \
       GIT_COMMIT_FILE="${WAM}/GIT_COMMIT" \
       WAIVED="${GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT:-0}" python - <<'PY'
import importlib, json, os
from collections.abc import Mapping

CONTRACT_FILE = os.environ["CONTRACT_FILE"]
SHARD_OUT = os.environ.get("SHARD_OUT") or ""
WAIVED = os.environ.get("WAIVED", "0") != "0"

# The one recovery command that works on this machine. `git -C ${WAM} checkout -- ...` does not:
# cluster/discoverer/sync.sh:66 rsyncs with --exclude '.git', so the cluster copy is a working tree
# with no repository behind it and git answers "not a git repository".
SYNC = "./cluster/discoverer/sync.sh      # from the WORKSTATION checkout, by a human"


def say(*lines):
    for line in lines:
        print("===   " + line)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# -- the two imports. A failure here is a broken or half-synced cluster copy and is never waivable:
#    the measurement resolves the same names from the same PYTHONPATH and could not run either.
try:
    mgt = importlib.import_module("measure_geom_tol")
except Exception as exc:
    print("FATAL: PREFLIGHT — STALE OR BROKEN CLUSTER COPY.")
    say("scripts/measure_geom_tol.py will not import: %s: %s" % (type(exc).__name__, exc),
        "The measurement resolves the same module from the same PYTHONPATH, so this job cannot",
        "produce anything at all. This is not waivable. Re-send the repo and re-submit:",
        "  " + SYNC)
    raise SystemExit(1)

# WHICH ADAPTER — ASKED, NEVER ASSUMED. The module name is read off measure_geom_tol.py, because
# that is the name its own _import_sam2_adapter() will resolve during the measurement. A literal
# fallback here would invert the one guarantee this block is for: if the measurement's spec is ever
# renamed or repointed, a preflight carrying its own copy of the old name would import the OLD
# adapter, find it agreeing with the committed contract, and print PREFLIGHT OK about a module this
# job does not run. There is no honest default for "which adapter is the measurement's", so absence
# is a refusal — and not a waivable one, for the same reason the imports are not: a cluster copy
# whose measure_geom_tol.py has no SAM2_ADAPTER_SPEC is a copy this preflight cannot speak about at
# all.
SPEC = getattr(mgt, "SAM2_ADAPTER_SPEC", None)
if not isinstance(SPEC, str) or not SPEC:
    print("FATAL: PREFLIGHT — measure_geom_tol.py DECLARES NO SAM2_ADAPTER_SPEC.")
    say("This preflight names the adapter by asking the measurement which module it will import,",
        "so that the two cannot disagree about it. Nothing here may guess that name: a guess that",
        "happens to import would be checked, reported OK, and be the wrong module. Not waivable.",
        "If the cluster copy is stale or half-synced, re-send it and re-submit:",
        "  " + SYNC)
    raise SystemExit(1)
try:
    adapter = importlib.import_module(SPEC)
except Exception as exc:
    print("FATAL: PREFLIGHT — THE ADAPTER WILL NOT IMPORT.")
    say("%s: %s: %s" % (SPEC, type(exc).__name__, exc),
        "--method sam2 would fail on frame 0 with the same error, after the GPU is allocated.",
        "This is not waivable. If the cluster copy is stale, re-send it:",
        "  " + SYNC)
    raise SystemExit(1)

commit = "(no GIT_COMMIT stamp — sync.sh writes one; this copy predates it or was not synced)"
try:
    with open(os.environ["GIT_COMMIT_FILE"], encoding="utf-8") as handle:
        commit = handle.read().strip() or commit
except OSError:
    pass

say("PREFLIGHT: the adapter this job would run, against the committed pre-commitment.",
    "cluster copy at %s" % commit,
    "adapter    %s  (%s)" % (SPEC, getattr(adapter, "__file__", "?")),
    "version    %s" % getattr(adapter, "ESTIMATOR_VERSION", "(none declared)"),
    "committed  %s" % CONTRACT_FILE)

failures = []  # (headline, [detail lines])

live = getattr(adapter, "SEGMENTER_CONTRACT", None)
# Mapping, not dict, because that is the test measure_geom_tol.sam2_method() applies to the same
# attribute ("contract = dict(contract) if isinstance(contract, Mapping) else None"). An adapter
# that froze its contract behind a MappingProxyType — the obvious way to make a pre-registered
# constant unwritable — is accepted by the measurement, and a preflight spelling the test `dict`
# would refuse it with THE ADAPTER EXPORTS NO SEGMENTER_CONTRACT and send the operator off to
# re-sync a cluster copy that was never stale. Refusing a run the real check would accept is as
# much a defect here as passing one it would refuse; this preflight is only worth anything if it
# answers the same question, the same way.
live = dict(live) if isinstance(live, Mapping) else None
if live is None:
    failures.append((
        "THE ADAPTER EXPORTS NO SEGMENTER_CONTRACT.",
        ["%s declares no operating point, so PR-08 §4 step 2's 'the same segmenter' is" % SPEC,
         "uncheckable — and an uncheckable requirement reads downstream exactly like a satisfied",
         "one. measure_geom_tol.py withholds gate qualification for this on its own, after the",
         "decode. Almost certainly a stale cluster copy:",
         "  " + SYNC]))

committed = None
try:
    doc = load(CONTRACT_FILE)
except Exception as exc:
    doc = None
    failures.append((
        "THE COMMITTED PRE-COMMITMENT IS MISSING OR UNREADABLE.",
        ["%s: %s" % (type(exc).__name__, exc),
         "That file IS the PR-08 §4 step 2 pre-commitment. The merge refuses to write over a path",
         "that carries no contract, so this whole partition would be unmergeable. Note that job",
         "190191 overwrote this path once already, with a merged disqualified artifact. Restore",
         "the tracked file from the workstation — NOT with git, this copy has no .git:",
         "  " + SYNC]))
if doc is not None:
    committed, where = mgt.committed_segmenter_contract(doc)
    if committed is None:
        failures.append((
            "THE COMMITTED DOCUMENT CARRIES NO SEGMENTER BLOCK.",
            ["%s parses, but has neither a top-level 'segmenter' nor" % CONTRACT_FILE,
             "mask_method.params.segmenter. 'The file used to say so before it was overwritten' is",
             "not a property anything downstream can read. Restore the tracked file:",
             "  " + SYNC]))
    else:
        say("committed contract found at %r (%d fields)" % (where, len(committed)))

if live is not None and committed is not None:
    drift = mgt.contract_disagreements(live, committed)
    if drift:
        detail = ["%s: committed %r, this adapter %r" % (d["field"], d["geom_tol"], d["this_run"])
                  for d in drift]
        failures.append((
            "CONTRACT DRIFT — the adapter is not the one the committed document describes.",
            detail + [
                "",
                "TWO CAUSES, TWO DIFFERENT REMEDIES, AND THE FIELD NAMES ABOVE TELL THEM APART.",
                "(a) THE CLUSTER COPY LAGS HEAD. This is what happened last time: twelve of the",
                "    last sixteen shards ran a pre-6a32143 adapter and all sixteen were missing",
                "    mask_validity_reference_max_frame_fraction, which the committed document has.",
                "    A field the adapter simply does not have ('this adapter None') is this case.",
                "    Remedy, from the workstation, by a human:",
                "      " + SYNC,
                "(b) THE ADAPTER HAS GENUINELY MOVED PAST ITS PRE-REGISTRATION. Then the number",
                "    this array would produce is not the number PR-08 §4 step 2 registered, and",
                "    the remedy is a NEW pre-registration that says so — a PR-08 V-document and a",
                "    re-committed contract, decided by the project owner. IT IS NOT AN EDIT TO",
                "    THE segmenter BLOCK OF pr08_geom_tol.json: that block is the pre-commitment,",
                "    and changing a value in it silently invalidates every artifact already",
                "    measured against it."]))
    else:
        say("contract agrees field for field on all %d fields." % len(committed))

# THE FORCE=1 TRAP. On this path measure_geom_tol.py's --out IS the shard artifact, and
# merge_committed_contract() reads whatever is already there as "the committed contract" — after
# the decode. Ask the same question here, for nothing.
if live is not None and SHARD_OUT and os.path.exists(SHARD_OUT):
    previous = None
    try:
        previous = load(SHARD_OUT)
    except Exception:
        previous = None  # a truncated file from a task killed at the wall; not a contract at all
    stale, stale_where = (mgt.committed_segmenter_contract(previous)
                          if isinstance(previous, dict) else (None, None))
    if stale is not None:
        stale_drift = mgt.contract_disagreements(live, stale)
        if stale_drift:
            failures.append((
                "A STALE SHARD ARTIFACT AT --out WOULD REFUSE THIS RUN AFTER THE DECODE.",
                ["%s already exists and records a segmenter at %r that" % (SHARD_OUT, stale_where),
                 "disagrees with this adapter on: "
                 + ", ".join(d["field"] for d in stale_drift),
                 "measure_geom_tol.py writes this shard THROUGH merge_committed_contract(), which",
                 "treats the file already at --out as the committed contract. It would decode this",
                 "shard in full and THEN exit 2, writing nothing — the failure that destroys the",
                 "work instead of preventing it.",
                 "",
                 "REMEDY: measure into a FRESH RUN_ID, which is the mitigation that needs no source",
                 "change and keeps the evidence:",
                 "  RUN_ID=<fresh> SHARD=1 NUM_SHARDS=... GEOM_STEP_FRAMES=1 \\",
                 "    sbatch --array=... --time=01:30:00 103_measure_geom_tol.sbatch",
                 "DO NOT simply delete the old shard. It is the only in-situ record that this index",
                 "was measured by a different adapter, and deleting it hides cause (a) above."]))

# THE FLAG. Checked last because it is the one an operator is most likely to want to argue with.
if not bool(getattr(adapter, "GATE_QUALIFIED", False)):
    failures.append((
        "%s.GATE_QUALIFIED IS False — this array would buy a disqualified number." % SPEC,
        ["gate_qualified is BAKED INTO EACH SHARD AT MEASUREMENT TIME and no merge re-derives",
         "it. So every shard measured now records gate_qualified: false permanently, the merge",
         "propagates it, and the pooled median can never be committed as GEOM_TOL however the",
         "decision goes afterwards. Flipping the flag later does not repair these artifacts — it",
         "requires re-measuring the corpus, at the full 13.6 GPU-h, a second time.",
         "",
         "WHAT THIS REFUSAL USED TO MEAN, AND WHY IT DOES NOT MEAN THAT ANY MORE. Until",
         "2026-08-27 it meant the second of the flag's two preconditions was still open: an",
         "empty GATE_QUALIFICATION_BLOCKERS tuple satisfies one, and the other is a RECORDED",
         "DECISION on the residue the 2026-08-26 discharges carry forward — in particular",
         "blocker 2's residue (i), a failure the discharged blocker predicted, occurring by a",
         "route it did not predict, on 92 frames. THAT DECISION WAS MADE AND RECORDED by the",
         "owner, and the flag moved with it in 13f0416 on 2026-08-27. So this refusal firing at",
         "all means the flag has gone BACK, or that this cluster copy predates that commit.",
         "Read apple_sam2's own WHAT WOULD TAKE IT BACK paragraph for which finding did it, and",
         "treat THAT, not residue (i), as what is outstanding.",
         "",
         "IT IS STILL THE PROJECT OWNER'S FLAG AND THIS IS NOT A REQUEST TO EDIT THE FLAG. The",
         "module's own comment and CLAUDE.md both say no session may make that edit; it is the",
         "owner's, on the record, in its own commit. Submit this array AFTER it stands again."]))

# THE FLAG IS ONLY ONE OF THREE CONJUNCTS, AND THE OTHER TWO FAIL THE SAME WAY FOR THE SAME MONEY.
# What lands in the shard is not GATE_QUALIFIED: measure_geom_tol.sam2_method() writes
# `gate_qualified=declared_gate and bool(checkpoints) and contract is not None`. The contract half
# is the refusal above. The WEIGHTS half is checked here, because an adapter that sets the flag and
# names no checkpoints is refused by that AND — quietly, at measurement time, into every one of the
# sixteen artifacts, with the reason parked in gate_qualification_withheld_reason where nothing
# reads it until the merge. A preflight that checked only the flag would print PREFLIGHT OK over
# exactly the 13.6 GPU-h loss it exists to prevent, having looked at two thirds of the condition.
# Read through measure_geom_tol's own _adapter_checkpoints() for the same reason
# contract_disagreements() is imported rather than re-spelled: it is the single implementation of
# "what did this adapter DECLARE it loads" (ESTIMATOR_CHECKPOINTS, then CHECKPOINT_ATTRS), and a
# second copy of that rule here would be a copy that drifts. A private name, and named as such: if
# it moves, this refuses rather than passing, which is the safe direction and is waivable, because
# unlike the imports the measurement itself still runs.
checkpoints_of = getattr(mgt, "_adapter_checkpoints", None)
declared_checkpoints = checkpoints_of(adapter) if callable(checkpoints_of) else None
if not declared_checkpoints:
    failures.append((
        "%s NAMES NO WEIGHTS — the shard would record gate_qualified: false anyway." % SPEC
        if callable(checkpoints_of) else
        "measure_geom_tol._adapter_checkpoints IS GONE — the weights half cannot be checked.",
        (["The adapter declares neither ESTIMATOR_CHECKPOINTS nor any of the single-checkpoint",
          "attributes measure_geom_tol._adapter_checkpoints() reads. sam2_method() writes",
          "gate_qualified = GATE_QUALIFIED and bool(checkpoints) and contract is not None, so the",
          "flag being True is not enough and this array would still buy a disqualified number —",
          "\"an estimator is its weights as much as its code\", in that function's own words: a",
          "tolerance that cannot say which weights produced it cannot be re-run at gate time,",
          "which is the only thing GEOM_TOL is for.",
          "",
          "Almost certainly a stale or partial cluster copy of scripts/estimators/:",
          "  " + SYNC]
         if callable(checkpoints_of) else
         ["scripts/measure_geom_tol.py no longer exports _adapter_checkpoints(), which is where",
          "this preflight reads the weights half of gate_qualified from. It is not re-spelled here",
          "on purpose — a second copy of that rule is a copy that drifts. Refusing rather than",
          "passing, because an unchecked conjunct reads downstream exactly like a satisfied one.",
          "Re-send the repo, or move this preflight to whatever replaced that function:",
          "  " + SYNC])))

if not failures:
    say("PREFLIGHT OK: the contract agrees field for field, the adapter names its weights, and its",
        "GATE_QUALIFIED flag is set — the three conjuncts sam2_method() ANDs into each shard's",
        "gate_qualified. Proceeding to the walltime self-check.")
    raise SystemExit(0)

print("")
for headline, detail in failures:
    print(("WARNING (WAIVED): " if WAIVED else "FATAL: ") + "PREFLIGHT — " + headline)
    say(*detail)
    print("")

if WAIVED:
    print("################################################################################")
    print("#  GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT=1 IS SET. Every refusal above was")
    print("#  downgraded to a warning and this shard is measuring anyway.")
    print("#  WHAT YOU ARE BUYING: an artifact that the merge, or PR-08 §4 step 2, or both,")
    print("#  will refuse — at full GPU cost. There is no code path that repairs it later;")
    print("#  the contract and gate_qualified are baked in at measurement time.")
    print("#  DO NOT COMMIT ANYTHING THIS RUN PRODUCES AS GEOM_TOL.")
    print("################################################################################")
    raise SystemExit(0)

print("       REFUSING TO START, BEFORE ANY DECODE. Nothing was measured and no GPU-seconds were")
print("       spent. Each refusal above names its own remedy; they are different remedies and the")
print("       field names say which one applies.")
print("       If you are knowingly measuring something that will not be committed, and only then:")
print("         GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT=1")
raise SystemExit(1)
PY
  then
    echo "FATAL: the pre-GPU contract/gate preflight refused this shard (see above). Nothing was"
    echo "       decoded, no GPU-seconds were spent, and no artifact was written or overwritten."
    exit 1
  fi

  # -- THE WALLTIME SELF-CHECK -------------------------------------------------------------------
  #
  # WHAT THIS USED TO DO AND WHY IT WAS WORSE THAN NOTHING. It re-read full_run_seconds and
  # load_seconds out of ${OUT}/GEOM_TOL_PILOT.json, divided by NUM_SHARDS, multiplied by a hardcoded
  # 1.155 count-imbalance factor measured at N=8, and warned if the result exceeded the request. The
  # committed pilot is job 189588's, produced at box_threshold 0.35 with no retry branch, and
  # scripts/estimators/apple_sam2.py has since moved to 0.15 with a (0.10, 0.10) retry — so its
  # slope is at least 1.99x optimistic (job 189658 measured p >= 0.166 against the pilot's 0.0833).
  # Against that artifact the old check would have printed a comfortable estimate and NO warning for
  # every shard of an array that was about to die at the wall, which is precisely what a self-check
  # exists to prevent. It also never knew THIS shard's size: it printed an even split scaled by a
  # constant from a different N.
  #
  # WHAT IT DOES NOW. Three changes, in order of how much they matter:
  #
  #   1. THE DENOMINATOR IS EXACT. ${SOURCE}/manifest.json carries episodes[].frames for every key,
  #      and shard_of() is re-derived here from the same rule measure_geom_tol.py uses, so this is
  #      THIS shard's real frame count and not an even split wearing a fudge factor.
  #   2. THE PILOT IS TRUSTED ONLY IF IT MEASURED THE COMMITTED SEGMENTER. The pilot records the
  #      mask_method it ran under; configs/transfer25/pr08_geom_tol.json holds the PR-08 §4 step 2
  #      pre-commitment, and measure_geom_tol.py refuses the whole run if the adapter disagrees with
  #      it. So the committed contract is a free, correct proxy for the adapter's operating point:
  #      a pilot whose segmenter block does not match it is DISCREDITED and its slope is discarded
  #      in favour of GEOM_SECONDS_PER_FRAME. This is the same comparison PR-08 §4 step 2 is checked
  #      on, applied to the cost model instead of to the number.
  #   3. IT REFUSES INSTEAD OF WARNING. A shard that provably cannot finish costs GPU-hours and
  #      produces nothing — 189658 spent six of them proving it. GEOM_ALLOW_TIGHT_WALL=1 overrides,
  #      and the refusal prints the command that would.
  #
  # THE FALLBACK COST MODEL IS MEASURED, AND SINCE 2026-08-27 IT IS NOT (0.18, 120).
  #
  # It was (0.18, 120) — 189658's measured FLOOR of 0.166 rounded up, paired with the discredited
  # pilot's load — and that pair is 1.50x optimistic, which is the same class of error as the one it
  # was written to catch. THE PARTITION HAS SINCE RUN, so the pair is now read off sixteen real
  # shards. Both numbers come out of the previous array's own logs, and both can be re-derived
  # without trusting this comment: runs/_slurm_logs/geom-tol.1899*_*.out and geom-tol.190125_*.out
  # (jobs 189935, 189971, 189984, 190125 — four waves of four), each of which prints its own shard's
  # exact frame count and its wall clock. Summed and fitted:
  #
  #     sum of the sixteen wall clocks           49 091 s = 13.6364 GPU-h
  #     least squares over the (frames, seconds) p = 0.2478 s/frame, L = 410 s
  #     per-shard rate                           0.2640 - 0.3189 s/frame, mean 0.2879
  #
  # Least-squares residuals sum to zero, so the unrounded fit (p = 0.247793, L = 410.219)
  # reproduces the 49 091 s exactly — the pair is the total, not an approximation of it. Rounded to
  # the four figures below it lands at 49 089 s, 2 s under, which is a rounding and not a second
  # model. The header's planning table has
  # been recomputed on it: N = 16 costs 13.636 GPU-h, not the 9.115 the old constants predicted.
  #
  # WHY THIS MATTERS AT THIS LINE AND NOT ONLY IN THE HEADER. The heaviest shard (index 5, 14 162
  # frames) actually took 3 741 s = 62.4 min. At (0.18, 120) THIS CHECK estimated it at 44.5 min and
  # would have waved through a --time=01:00:00 request that dies at the wall — a self-check that
  # reassures is worse than none, which is the sentence this whole block was written under. At
  # (0.2478, 410) it estimates 65.3 min, x1.25 = 81.7 min: it still passes --time=01:30:00 (the
  # shape that ran) and it refuses the one-hour request.
  #
  # NOT THE BLENDED 0.29. Each shard's wall clock already contains one model load, so the 0.2879
  # mean has L folded into it; pairing it with 410 charges the load twice and over-estimates by
  # ~21 %. The two self-consistent pairs are (0.2478, 410) and (0.2879, 0). The fitted one is used
  # because L is a real fixed cost this file's sharding arithmetic subtracts explicitly.
  #
  # Both stay env vars so a re-measured pilot or a different machine does not need an edit here —
  # and a PILOT that matches the committed contract is still preferred over them automatically.
  GEOM_SECONDS_PER_FRAME=${GEOM_SECONDS_PER_FRAME:-0.2478}
  GEOM_LOAD_SECONDS=${GEOM_LOAD_SECONDS:-410}
  if ! SHARD_INDEX="${SHARD_INDEX}" NUM_SHARDS="${NUM_SHARDS}" \
       MANIFEST="${SOURCE}/manifest.json" PILOT_JSON="${OUT}/GEOM_TOL_PILOT.json" \
       CONTRACT_FILE="${WAM}/configs/transfer25/pr08_geom_tol.json" \
       P_FALLBACK="${GEOM_SECONDS_PER_FRAME}" L_FALLBACK="${GEOM_LOAD_SECONDS}" \
       ALLOW_TIGHT="${GEOM_ALLOW_TIGHT_WALL:-0}" python - <<'PY'
import hashlib, json, os, time

def contract_of(doc):
    """The segmenter block, wherever this repo writes it. None when the document makes no claim."""
    if not isinstance(doc, dict):
        return None
    params = (doc.get("mask_method") or {}).get("params") or {}
    for block in (doc.get("segmenter"), params.get("segmenter")):
        if isinstance(block, dict):
            return block
    return None

def load(path):
    try:
        return json.loads(open(path).read())
    except Exception:
        return None

i, n = int(os.environ["SHARD_INDEX"]), int(os.environ["NUM_SHARDS"])
manifest = load(os.environ["MANIFEST"])
episodes = (manifest or {}).get("episodes") or []
if not episodes:
    print("===   NO SELF-CHECK: %s carries no episodes[], so this shard's frame count is unknown."
          % os.environ["MANIFEST"])
    print("===   The run is NOT stopped — the manifest is provenance, not an input to the "
          "measurement — but")
    print("===   nothing here can say whether this request is long enough. Watch the wall.")
    raise SystemExit(0)

# The same rule as measure_geom_tol.shard_of(), re-derived rather than imported: this check runs
# before the interpreter that owns that function is started, and a copy that drifts would size the
# wrong shard. It is eleven bytes of arithmetic and the merge re-derives it too.
def shard_of(key, num):
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big") % num

mine = [e for e in episodes if shard_of(str(e["id"]), n) == i]
frames = sum(int(e["frames"]) for e in mine)
total = sum(int(e["frames"]) for e in episodes)

p, load_s, basis = float(os.environ["P_FALLBACK"]), float(os.environ["L_FALLBACK"]), None
pilot = load(os.environ["PILOT_JSON"])
committed = contract_of(load(os.environ["CONTRACT_FILE"]))
if pilot is None:
    basis = "no pilot artifact"
elif pilot.get("seconds_per_frame") is None or pilot.get("load_seconds") is None:
    basis = "pilot carries no slope (%s)" % "; ".join(pilot.get("blockers") or ["?"])
elif committed is None:
    basis = "no committed segmenter contract to check the pilot against"
else:
    theirs = contract_of(pilot)
    if theirs is None:
        basis = "the pilot records no segmenter block, so its operating point is unknown"
    else:
        diff = sorted(k for k in set(theirs) | set(committed) if theirs.get(k) != committed.get(k))
        if diff:
            basis = ("PILOT DISCREDITED: its segmenter disagrees with the committed contract on "
                     + ", ".join(diff))
        else:
            p, load_s = float(pilot["seconds_per_frame"]), float(pilot["load_seconds"])
            basis = "pilot %s, segmenter matches the committed contract" % (
                pilot.get("slurm_job_id") or "?")

est = frames * p + load_s
print(f"===   shard {i}/{n}: {len(mine)} episodes, {frames} of {total} frames "
      f"({frames / (total / n):.3f}x the even split)")
print(f"===   cost model: p={p:.4f} s/frame, fixed load={load_s:.0f}s  [{basis}]")
if basis and basis.startswith(("PILOT DISCREDITED", "the pilot records no")):
    print("===   -> using GEOM_SECONDS_PER_FRAME instead. A pilot measured at a segmenter this "
          "adapter has")
    print("===      replaced is evidence about a different run; see the header block on job "
          "189658.")
print(f"===   estimate {est:.0f}s ({est/60:.1f} min), x1.25 = {est*1.25/60:.1f} min")

end = os.environ.get("SLURM_JOB_END_TIME")
if not (end and end.isdigit()):
    print("===   SLURM_JOB_END_TIME is unset — cannot compare the estimate against the request.")
    raise SystemExit(0)
left = int(end) - int(time.time())
print(f"===   walltime remaining {left}s ({left/60:.1f} min); "
      f"break-even p for this request = {(left - load_s) / frames:.4f} s/frame")
if est * 1.25 <= left:
    raise SystemExit(0)
if os.environ.get("ALLOW_TIGHT", "0") != "0":
    print("===   WARNING: the estimate with margin exceeds this request, and "
          "GEOM_ALLOW_TIGHT_WALL=1 says run anyway.")
    raise SystemExit(0)
print("FATAL: this shard needs ~%.0f min (x1.25 = %.0f min) and the request leaves %.0f min."
      % (est / 60, est * 1.25 / 60, left / 60))
print("       REFUSING TO START. A shard that dies at the wall writes nothing and the GPU-hours are")
print("       gone: job 189658 spent roughly six of them proving exactly this. The estimate is")
print("       built on THIS shard's real frame count from the manifest, so it is not the "
      "three-episode")
print("       extrapolation the old warning apologised for.")
print("       Raise --time (04:00:00 is the QoS cap), or raise NUM_SHARDS and re-run the whole")
print("       partition, or set GEOM_ALLOW_TIGHT_WALL=1 if you believe the cost model is wrong.")
raise SystemExit(1)
PY
  then
    echo "FATAL: the walltime self-check refused this shard (see above). Nothing was measured."
    exit 1
  fi

  START=${SECONDS}
  # `set -e` is on, so a bare invocation followed by RC=$? never sees a non-zero status: the shell is
  # already gone by the time the assignment runs. The `if` is where set -e does not fire.
  if python "${WAM}/scripts/measure_geom_tol.py" \
    --corpus "${SOURCE}" \
    --method sam2 \
    --step-frames "${GEOM_STEP_FRAMES}" \
    --shard "${SHARD_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --out "${SHARD_OUT}"; then RC=0; else RC=$?; fi
  ELAPSED=$((SECONDS - START))
  echo "=== shard ${SHARD_INDEX} exited ${RC} after ${ELAPSED}s"
  # EXIT 3 IS TWO DIFFERENT EVENTS AND THIS FILE USED TO CONFLATE THEM. It means "the measurement
  # is disqualified from the gate", which was TRUE OF EVERY CORRECT SHARD while
  # apple_sam2.GATE_QUALIFIED was False — nothing about the corpus could change that — and stopped
  # being true on 2026-08-27, when 13f0416 flipped that flag on: a shard measured at HEAD records
  # gate_qualified: true, exits 0, and never reaches this branch. The branch stays because it is
  # about the ARTIFACT and not about the flag, and a waived run, a lagging cluster copy or a
  # re-flip still has to classify what it wrote. Treating exit 3 as FATAL printed sixteen failures
  # over sixteen good measurements and — with the old resume check — guaranteed that every
  # re-submission re-measured the whole partition. So exit 3 is re-classified
  # by asking the SAME question the resume check asks, of the artifact that was actually written:
  # a shard whose only disqualification is the adapter's standing flag has LANDED. Anything else,
  # and every other exit code, is still fatal and still writes nothing to the merge.
  if (( RC == 3 )) && [[ -f "${SHARD_OUT}" ]] \
     && shard_artifact_landed "${SHARD_OUT}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${GEOM_STEP_FRAMES}"
  then
    echo "=== SHARD ${SHARD_INDEX}/${NUM_SHARDS} LANDED. Exit 3 here is the adapter's standing gate"
    echo "===   flag (GATE_QUALIFIED is False in the adapter this job actually ran — NOT 'nobody"
    echo "===   has looked at a mask', which was discharged 2026-08-26), which is identical across"
    echo "===   every shard of this partition and is not a fact about this shard."
    echo "===   READ THIS AS A SURPRISE, NOT AS ROUTINE: 13f0416 flipped that flag on on"
    echo "===   2026-08-27, so a shard measured at HEAD records gate_qualified: true, exits 0 and"
    echo "===   never prints this. Getting here means the cluster copy lags that commit, or the"
    echo "===   flag has gone back, or this run was waived."
    echo "===   The merge will pool it and will write the pooled median DISQUALIFIED. THAT NUMBER"
    echo "===   MUST NOT BE COMMITTED AS GEOM_TOL, and there is no later repair: gate_qualified is"
    echo "===   baked in at measurement time and no merge re-derives it, so the corpus has to be"
    echo "===   re-measured, under a fresh RUN_ID, by an adapter whose flag is set."
    RC=0
  fi
  if (( RC != 0 )); then
    echo "FATAL: shard ${SHARD_INDEX} exited ${RC} and its artifact must not reach the merge."
    echo "       Exit 3 here is NOT the adapter's standing gate blocker — that case is handled"
    echo "       above and lands. It means this shard measured the wrong thing: coverage under the"
    echo "       floor, a truncating flag, or no displacement at all. The merge refuses a partition"
    echo "       where one shard says gate_qualified false and another says true, so this stops the"
    echo "       whole measurement rather than skewing it. Exit 2 means nothing was measured at"
    echo "       all. Either way, fix the cause and re-run THIS index:"
    echo "         RUN_ID=${RUN_ID} SHARD=1 NUM_SHARDS=${NUM_SHARDS} \\"
    echo "           GEOM_STEP_FRAMES=${GEOM_STEP_FRAMES} \\"
    echo "           sbatch --array=${SHARD_INDEX} --time=01:30:00 ${THIS_SCRIPT}"
    exit "${RC}"
  fi
  echo "=== NEXT, once every shard has landed (this is NOT GEOM_TOL and no shard is):"
  echo "===   RUN_ID=${RUN_ID} MERGE=1 NUM_SHARDS=${NUM_SHARDS} sbatch --qos=2cpu-single-host \\"
  echo "===     --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00 ${THIS_SCRIPT}"
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
echo "=== NEXT: commit configs/transfer25/pr08_geom_tol.json AND its .sha256 in the repo. The"
echo "===       committed SEGMENTER CONTRACT that was already in that file is inside this artifact:"
echo "===       measure_geom_tol.py compared it field for field against the adapter this run drove"
echo "===       and would have refused the whole run, writing nothing, on any disagreement. The"
echo "===       diff you are committing therefore ADDS the measurement to the contract; it must not"
echo "===       change spec_version, what_this_is or the segmenter block. If it does, do not commit"
echo "===       it — that is the pre-commitment PR-08 §4 step 2 is checked against."
echo "===       Then re-run 102_stage_sam2_weights.sbatch: its <role>-vs-pr08_geom_tol.json check"
echo "===       reads mask_method.params.checkpoints, which only a MEASURED artifact carries."
