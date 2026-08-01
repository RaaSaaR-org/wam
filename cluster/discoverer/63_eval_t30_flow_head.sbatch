#!/bin/bash
# Step 9 — T-30 / I-3: is the single-shot regression head throwing the chunk away?
#
# THIS FILE IS THE PRE-REGISTRATION. Every constant below is a literal and this file is in git
# before the first arm is submitted. Changing one after seeing the number it scored is the move
# pre-registration exists to prevent (docs/improvements.md I-9). If a constant has to change,
# version it — add T30_RULE_V2 next to T30_RULE_V1 and say which runs scored under which.
#
# The joint model trains TWO action readouts and deploys one. `ActionHead` regresses the whole
# chunk in one shot from the pooled features; a rectified-flow branch (`velocity_head` +
# `action_recon`) models the same chunk as a distribution and is never read at inference. A
# single-shot L2 regressor under a one-to-many conditional is mean-seeking, and on
# runs/t16-lora-seed0 the numbers are CONSISTENT WITH one: chunk RMS 0.00226 against the
# demonstrations' 0.00404 (a 44 % shortfall) and smoothness_ratio 0.29.
#
# "Consistent with", not "the signature of" — an earlier version of this header said the latter
# and it overstated the evidence. What is actually ruled out is a bounded-output artifact: max
# |target| is 0.0192 against a tanh, and the `limit` penalty only bites outside ±0.95, so neither
# bound is active. Two alternatives are NOT ruled out, and each produces small, smooth chunks on
# its own:
#   (a) the one-sided jerk regulariser. configs/training/joint_wan_gr00t.yaml sets
#       weights.smoothness = 0.01, and JointTrainer.compute_losses applies smoothness_loss to
#       decoded_targets — the REGRESSION head's output — and to nothing else. The flow branch is
#       never charged for jerk. A smoothness gap between the readouts is therefore the expected
#       consequence of the objective, so the smoothness clause of the rule below is CONFOUNDED
#       and reads as descriptive, not decisive.
#   (b) L2 shrinkage: action_reg is an MSE against targets whose own RMS is 0.004, with AdamW
#       weight decay on top. A head that under-shoots magnitude uniformly is indistinguishable,
#       on RMS and jerk alone, from one that averages modes.
# Separating (a)/(b) from mean-seeking needs an intervention (smoothness=0 and retrain, or a
# multi-modal task), not a readout swap. This job answers the narrower question it can answer:
# does the branch trained as a distribution carry anything the regression head does not?
#
# WHAT A NEGATIVE HERE IS ABOUT. During training the velocity head always saw (features from
# video noised to t, action latent noised to the SAME t) — co_denoise shares one t. The sampler
# computes ONE backbone pass at t=1 on the clean observation and reuses those features at every
# t_k, so near t_k=0 the head is asked about (near-pure-noise z, clean-video features, t≈0), a
# combination it was never trained on. That is a confound with the same structure as I-7, and it
# is why every negative branch below says "as sampled this way" and why the WARM-START arm exists.
# The faithful sampler (n backbone passes, observation noised to each t_k) is NOT run here — it
# costs n backbone passes and destroys the observation the readout exists to read — so no branch
# of this rule refutes the flow branch as trained.
#
#   sbatch 63_eval_t30_flow_head.sbatch
#   RUN_ID=t16-lora-seed1 sbatch 63_eval_t30_flow_head.sbatch
#   FLOW_STEPS_LIST="1 32" sbatch 63_eval_t30_flow_head.sbatch      # the quick version
#   FRAME_HISTORY= sbatch 63_eval_t30_flow_head.sbatch              # score the tiled mode instead
#
# RUN THIS AFTER T-29 REPORTS, not alongside it. Every arm here runs with --frame-history because
# T-29 establishes the real window as the in-distribution mode; running both A/Bs at once turns
# two 1-D questions into a 2x2 nobody pre-registered (docs/improvements.md, "Suggested order").
#
# PRE-FLIGHT, measured on CPU 2026-08-01 (scripts/check_action_latent.py, no GPU, seconds):
#   ceiling  encode the demonstrated chunks and decode them straight back through action_recon:
#            target MSE 8.10372e-07, jerk ratio 1.375. 14.9x below this run's own 1.21027e-05 and
#            11.3x below repeat-last-action's 9.13766e-06. The latent carries the chunk; whatever
#            this job scores is a statement about the velocity head, not the representation.
#   floor    decode the per-step latent CENTROIDS — "the sampler knew which step, and nothing
#            else": 1.68201e-05, WORSE than zero-delta's 1.63276e-05. Step index is recoverable
#            from the latent at 100 % accuracy (within-step std 0.056 vs between-step 1.42), and
#            ActionVelocityHead takes no step index. So the content the sampler must hit is a
#            small perturbation riding on a much larger positional signal, and landing near the
#            floor is the EXPECTED outcome, not evidence of a bug. Read the guard below first.
#
# THE ARMS. A = regression head (today's deployed readout). Everything else is one readout flag
# different on the same checkpoint, holdout, frames, camera and GPU:
#   B(n)      --flow-sampler --flow-steps n, one draw from noise, seed 0, for n in the sweep.
#             This is what a deployed sampler would emit. It is the DIVERSITY/MAGNITUDE arm.
#   B_mean    --flow-mean-k K at n=FLOW_PRIMARY: K draws averaged. This is the arm the rule keys
#             on, because the headline metric is MSE-derived and MSE structurally penalises
#             sampling. For any calibrated conditional, E||a - draw||^2 = E||a - mean||^2 +
#             E||draw - mean||^2: a single unbiased draw scores WORSE than the conditional mean
#             by exactly the conditional variance. Scoring one draw against a mean-seeking
#             regression head would reward the very defect under test. K draws leave 1/K of that
#             penalty. B_mean is a MEASUREMENT, never a deployment candidate — averaging draws
#             is the mean-seeking the flow branch exists to avoid.
#   B_seed1   --flow-seed 1 at n=FLOW_PRIMARY: the replicate that MEASURES the band below,
#             instead of asserting one. Same shape as I-8's sigma_hat (62_eval_i8_curve.sbatch).
#             It also covers a property of the sampler worth stating out loud: the seed is
#             re-drawn per call and never advanced (T-25's bit-identical rollouts depend on that),
#             so ALL 1 040 chunks of one arm are integrated from the SAME noise vector. Per chunk
#             that is a fair draw, but the errors across the holdout are correlated through it, so
#             an arm's aggregate is conditioned on one vector. sigma_hat is exactly the measured
#             sensitivity of the whole aggregate to that choice — which is why the band is derived
#             from it rather than written down as a literal.
#   B_warm    --flow-t0 T0 at n=FLOW_PRIMARY: integration starts at t0 from the regression chunk
#             re-encoded by action_encoder and noised to t0 — i.e. ONLY in the region where the
#             head's t and the features' t=1 roughly agree. It inherits the regression mean by
#             construction (there is no clean action latent at inference; using the demonstrated
#             chunk would make it an oracle, not a readout), so it CANNOT show that the flow
#             branch models the conditional. What it can do is tell the two readings of a
#             negative apart: "the branch is dead" vs "we sampled it outside its training region".
#   B_warmK   --flow-t0 T0 --flow-mean-k K: the same control, variance-matched to B_mean. This is
#             what the mismatch clause TRIGGERS on, because a warm single draw beats a from-noise
#             single draw partly by carrying less sampling variance — the same rigging the mean
#             arm exists to remove, and it must not be allowed back in through the control.
#
# DECISION RULE T30_RULE_V2, fixed before the run (docs/improvements.md I-3). Keyed on
# skill_vs_repeat_pct of B_mean against A, with BAND = max(3 * sigma_hat, 10 pp) and sigma_hat
# the MEASURED |seed0 - seed1| spread. V1 is superseded UNRUN — no arm of it was ever executed,
# so this is a pre-registration edit, not a post-hoc one; V1's two defects are recorded at the
# bottom of this block:
#   B_mean > A + BAND, B_mean > 0   -> the flow READOUT scores higher and clears L1. Confound (a)
#                                 applies HERE TOO: only the regression head is charged
#                                 smoothness_loss, so "the mean-seeking head was the T-16
#                                 negative" and "the jerk regulariser was" both produce this
#                                 outcome and this job cannot separate them. What IS licensed:
#                                 the negative was not a property of the checkpoint's action
#                                 knowledge, so the T-16 row needs a CORRECTION.
#                                 DEPLOYMENT is gated SEPARATELY on the SINGLE-draw arm B > 0,
#                                 because B_mean is what the rule keys on but mean-of-k is a
#                                 measurement, never a policy: B_mean can clear the bar purely by
#                                 averaging away the conditional variance, which is the
#                                 mean-seeking under test. B_mean > 0 with B <= 0 authorises a
#                                 CORRECTION and no deployment.
#   B_mean > A + BAND, B_mean <= 0  -> the head cost real skill and was still not the whole story.
#                                 Confound (a) applies here as well. Record the delta, keep the
#                                 regression head deployed until a closed-loop number exists, and
#                                 go to I-8 (data-scaling curve).
#   |B_mean - A| <= BAND       -> the two readouts SCORE THE SAME ON THIS METRIC. That sentence
#                                 is the whole finding; "the flow branch is decorative" is a
#                                 stronger claim and is only licensed when the RMS and spread
#                                 columns agree (see the mode-collapse clause below).
#   B_mean < A - BAND          -> the mean was doing real work. The MECHANISM claim — that the
#                                 flow branch AS SAMPLED THIS WAY cannot PLACE a chunk in time
#                                 (each step is an independent draw, ActionVelocityHead takes no
#                                 step index) — is gated SEPARATELY on B_mean's mse landing within
#                                 FLOOR_NEAR_FRAC of FLOOR_MSE, the score of a chunk with the
#                                 right content and no position. Off the floor it is a hypothesis,
#                                 not a reading, and the run records the negative without a cause.
#                                 Only the on-floor case routes to I-2.
#   MISMATCH CLAUSE, applied to BOTH negative branches and printed either way, and thresholded on
#                                 MISMATCH_BAND = 3 * sigma_hat (the MEASURED spread, NOT the
#                                 floored BAND). The floor is conservative for the four headline
#                                 branches — a too-wide band can only push them toward "the same
#                                 on this metric" — but this clause is inverted: it must FIRE to
#                                 withhold the strong claim, so widening it makes "the mismatch
#                                 has been measured and rejected" easier to print. At sigma_hat
#                                 = 1 pp the floored band would swallow a real 9 pp effect.
#                                 If B_warmK beats
#                                 A by more than MISMATCH_BAND, or beats B_mean by more than it —
#                                 both variance-matched comparisons — the negative is attributable to
#                                 the sampler's conditioning mismatch and is NOT a statement about
#                                 the flow branch. Next step is then the sampler (t-conditioned
#                                 features, or a retrain that noises the observation), not I-2.
#                                 B_warm-vs-single-draw is REPORTED but is not a trigger: a warm
#                                 draw carries less sampling variance than a from-noise draw, so
#                                 that comparison would fire on the metric's own bias. If B_warmK
#                                 is also flat, the mismatch has been MEASURED and rejected as the
#                                 explanation — the negative then extends to the branch under
#                                 every protocol runnable offline without a retrain, and still not
#                                 to the faithful one.
#   MODE COLLAPSE, only as a labelled diagnostic: claimed only when the measured draw-to-draw
#                                 spread (from B(primary) vs B_seed1, element-wise) is below 10 %
#                                 of the demonstrations' RMS AND B's own RMS is no closer to the
#                                 demonstrations' than A's. Otherwise the sampler varies and
#                                 "mode-collapsed" is not the word for what it did.
#   smoothness_ratio: DESCRIPTIVE, not a gate — the regression head carries a jerk penalty the
#                                 flow branch does not (see (a) above), so a gap is predicted by
#                                 the objective. Reported in the table for every arm next to
#                                 RMS/demo, because neither of those rewards mean-seeking the way
#                                 an MSE-derived score does.
#   SAMPLER-BROKEN GUARD, checked FIRST and on EVERY arm: RMS |targets| above 3x the
#                                 demonstrations' 0.004041, or any non-finite value, or mse above
#                                 5x zero-delta. Those are integration bugs (direction, scale,
#                                 conditioning) — fix and re-run, record nothing. A rule-keyed arm
#                                 tripping it stops the verdict; a sweep-only arm tripping it
#                                 marks that arm unreadable and the sweep reports around it (n=1
#                                 overshooting is what a non-straight field looks like, not a
#                                 bug). An mse merely a little above zero-delta is NOT this case;
#                                 that is the floor, and it is a result.
#
# WHAT V1 GOT WRONG (superseded before any arm ran; kept because a pre-registration that is
# silently edited is not one):
#   1. Both POSITIVE branches named "the mean-seeking head was the T-16 negative" and no
#      alternative, while the same file's confound (a) predicts flow-beats-regression directly
#      from the objective — only the regression head is charged smoothness_loss. V1 held the
#      negative branches to an alternative reading and the positive ones to none. Fixed: both
#      positive branches now name (a), and neither claims mean-seeking was the cause.
#   2. The positive branch keyed on B_mean but prescribed deploying the SINGLE draw, with no
#      condition on that arm at all — so "L1 CLEARED, ship it" could print with B_mean at +2 %
#      and the arm actually shipping at -25 %. Fixed: deployment is gated on B > 0 separately.
#   3. The mismatch clause was thresholded on the floored BAND. Fixed: MISMATCH_BAND.
#
#SBATCH --job-name=wam-eval-t30
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=06:00:00              # 10 arms, each rebuilding the frozen 5B base from disk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/eval-t30.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/wam
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

EVAL=${PROJ}/wam/scripts/eval_t16.py
BENCH=${PROJ}/wam/scripts/run_bench.py
LATENT=${PROJ}/wam/scripts/check_action_latent.py
for f in "${EVAL}" "${BENCH}" "${LATENT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "FATAL: ${f} missing — run cluster/discoverer/sync.sh from the Mac first."
    exit 1
  fi
done

# t16-lora-seed0 predates train_episode_ids, so eval_t16.py proves its split by COMPLEMENT and
# needs no witness file. Pointing RUN_ID at an I-8 rung instead would need --train-episodes added
# to `common` below (the disjointness proof refuses to score without an external witness).
RUN_ID=${RUN_ID:-t16-lora-seed0}
CKPT=${CKPT:-latest}
OUT=${PROJ}/runs/${RUN_ID}
DATASET=${PROJ}/data/gr00t-apple-full
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt
# The external witness for the split proof — see 61_eval_t29_frame_history.sbatch for the full
# note. i8_train_362.txt is the complement of the holdout above, so it serves the archived
# checkpoint (complement proof, cross-checked) and a retrained one (disjointness proof, where it
# is MANDATORY) with the same command line. The header advertises
# `RUN_ID=t16-lora-seed1 sbatch ...`, which is exactly the retrained case.
#   TRAIN_SPLIT=${PROJ}/wam/configs/splits/i8_train_040.txt RUN_ID=i8-rung040-seed0 sbatch ...
TRAIN_SPLIT=${TRAIN_SPLIT:-${PROJ}/wam/configs/splits/i8_train_362.txt}
# 1 is the control: one Euler step from noise is what a "rectified flow needs one step" reading
# of the literature would deploy, and it shows how far from straight this field actually is.
FLOW_STEPS_LIST=${FLOW_STEPS_LIST:-"1 4 16 32 64"}
FLOW_PRIMARY=${FLOW_PRIMARY:-32}     # eval_t16.py's DEFAULT_FLOW_STEPS — the arm the rule keys on
# 8 draws: the single-draw MSE penalty is the conditional variance, and 8 leaves 1/8 of it, which
# is below the measured band unless the sampler's spread dwarfs the signal (the run prints both,
# so the assumption is checked rather than trusted). Averaging is nearly free — the backbone pass
# is shared across draws and one draw is n MLP evaluations at batch 1.
FLOW_MEAN_K=${FLOW_MEAN_K:-8}
# 0.6: past the halfway point of the path, so most of the integration happens where the features'
# t=1 and the head's t are closest, while still leaving 40 % of the path for the head to move the
# chunk. A t0 near 1 would return the warm start almost unchanged and measure nothing.
FLOW_T0=${FLOW_T0:-0.6}
REPLICATE_SEED=${REPLICATE_SEED:-1}  # the second draw the band is measured from
# Set FRAME_HISTORY="" to score the pre-T-29 tiled mode instead. ${VAR-default}, not ${VAR:-...},
# so an explicitly empty value survives.
FRAME_HISTORY=${FRAME_HISTORY-"--frame-history"}

# The rule keys on the primary arm, so it must exist even if someone trims the sweep.
case " ${FLOW_STEPS_LIST} " in
  *" ${FLOW_PRIMARY} "*) ;;
  *) FLOW_STEPS_LIST="${FLOW_STEPS_LIST} ${FLOW_PRIMARY}" ;;
esac

# The two control arms have to be control arms. FLOW_T0=0 would write a plain from-noise run into
# a directory named "warm0", and FLOW_MEAN_K=1 a single draw into one named "mean1" — an artifact
# whose name and contents disagree, which is the failure the readout tags exist to prevent.
if ! awk -v t="${FLOW_T0}" 'BEGIN { exit !(t > 0 && t < 1) }'; then
  echo "FATAL: FLOW_T0 must be in (0, 1), got ${FLOW_T0} — at 0 the warm arm is the plain"
  echo "       sampler under a warm arm's name; at 1 it integrates nothing."
  exit 1
fi
if [[ "${FLOW_MEAN_K}" -lt 2 ]]; then
  echo "FATAL: FLOW_MEAN_K must be >= 2, got ${FLOW_MEAN_K} — the point of the arm is to remove"
  echo "       the single draw's conditional-variance penalty, and k=1 removes none of it."
  exit 1
fi

if [[ ! -d "${OUT}" ]]; then
  echo "FATAL: ${OUT} does not exist — nothing to score."
  exit 1
fi

# Same resolution eval_t16.py's resolve_checkpoint applies, spelled out here because the CPU
# pre-flight needs the step dir itself rather than a --checkpoint argument.
if [[ "${CKPT}" == "latest" && -e "${OUT}/latest" ]]; then
  STEP_DIR=$(readlink -f "${OUT}/latest")
elif [[ "${CKPT}" == "latest" ]]; then
  STEP_DIR=$(ls -d "${OUT}"/checkpoints/step-* | sort | tail -1)
elif [[ -d "${CKPT}" ]]; then
  STEP_DIR="${CKPT}"
else
  STEP_DIR="${OUT}/checkpoints/${CKPT}"
fi
echo "checkpoint dir: ${STEP_DIR}"

# One --out per arm, and eval_t16.py now REFUSES to write a second readout's artifacts over a
# first one's — the names are built once here and handed to the verdict script so bash and python
# cannot drift about which directory holds which arm.
REG_DIR=eval-t30-regression
MEAN_DIR=eval-t30-flow${FLOW_PRIMARY}-mean${FLOW_MEAN_K}
WARM_DIR=eval-t30-flow${FLOW_PRIMARY}-warm${FLOW_T0}
WARMK_DIR=eval-t30-flow${FLOW_PRIMARY}-warm${FLOW_T0}-mean${FLOW_MEAN_K}
SEED1_DIR=eval-t30-flow${FLOW_PRIMARY}-seed${REPLICATE_SEED}

# Every arm differs in EXACTLY the readout flags. Checkpoint, dataset, holdout, camera, device
# and frame mode are held identical by construction below rather than by copy-paste.
common=(
  --run-dir "${OUT}"
  --checkpoint "${CKPT}"
  --dataset "${DATASET}"
  --holdout "${HOLDOUT}"
  --train-episodes "${TRAIN_SPLIT}"
  --camera ego
  --device cuda
)
if [[ -n "${FRAME_HISTORY}" ]]; then
  common+=("${FRAME_HISTORY}")
fi
echo "frame mode: ${FRAME_HISTORY:-tiled single frame (pre-T-29 default)}"

echo
echo "=== A: the regression head (today's deployed readout)"
python "${EVAL}" "${common[@]}" --out "${OUT}/${REG_DIR}"

echo
echo "=== pre-flight: what the action latent can carry at best (CPU, no GPU)"
# Run against A's own predictions so the ceiling is printed next to the error it has to beat.
# Non-zero exit means the decoder, not the readout, is the constraint — the sweep below is then
# already known to be pointless, but it is minutes and the numbers still belong in the record.
python "${LATENT}" --checkpoint "${STEP_DIR}" \
  --predictions "${OUT}/${REG_DIR}/predictions.jsonl" || true

# Unquoted on purpose: FLOW_STEPS_LIST is a word list, not one argument.
for n in ${FLOW_STEPS_LIST}; do
  echo
  echo "=== B(${n}): the flow readout, ${n} Euler step(s), one draw from noise"
  python "${EVAL}" "${common[@]}" --flow-sampler --flow-steps "${n}" \
    --out "${OUT}/eval-t30-flow${n}"
done

echo
echo "=== B_seed1: the same arm, second draw — this MEASURES the band the rule uses"
python "${EVAL}" "${common[@]}" --flow-sampler --flow-steps "${FLOW_PRIMARY}" \
  --flow-seed "${REPLICATE_SEED}" --out "${OUT}/${SEED1_DIR}"

echo
echo "=== B_mean: ${FLOW_MEAN_K} draws averaged — the MSE-fair comparison against A"
python "${EVAL}" "${common[@]}" --flow-sampler --flow-steps "${FLOW_PRIMARY}" \
  --flow-mean-k "${FLOW_MEAN_K}" --out "${OUT}/${MEAN_DIR}"

echo
echo "=== B_warm: integration starts at t0=${FLOW_T0} — the conditioning-mismatch control"
python "${EVAL}" "${common[@]}" --flow-sampler --flow-steps "${FLOW_PRIMARY}" \
  --flow-t0 "${FLOW_T0}" --out "${OUT}/${WARM_DIR}"

echo
echo "=== B_warmK: the same control, ${FLOW_MEAN_K} draws — variance-matched to B_mean"
python "${EVAL}" "${common[@]}" --flow-sampler --flow-steps "${FLOW_PRIMARY}" \
  --flow-t0 "${FLOW_T0}" --flow-mean-k "${FLOW_MEAN_K}" --out "${OUT}/${WARMK_DIR}"

echo
echo "=== the comparison (regression vs the MSE-fair flow arm)"
# --compare refuses two runs whose holdouts differ, so the columns are guaranteed to mean the
# same thing. Both halves came from this job, this GPU, this checkpoint. The mean arm is the one
# compared here because a single draw is worth the conditional variance MORE MSE than the mean,
# and --compare is an MSE table. --no-write keeps the bench.json eval_t16.py wrote — it is the
# one carrying the "+flow${FLOW_PRIMARY}s0k${FLOW_MEAN_K}" run_name, which a re-score from the
# directory name alone would flatten back to the run id.
python "${BENCH}" "${OUT}/${REG_DIR}" "${OUT}/${MEAN_DIR}" --compare --no-write || true

echo
echo "=== the decision (docs/improvements.md I-3)"
# FLOW_STEPS_LIST unquoted again: one argv entry per swept step count.
python - "${OUT}" "${FLOW_PRIMARY}" "${FLOW_MEAN_K}" "${FLOW_T0}" "${REPLICATE_SEED}" \
        "${REG_DIR}" "${MEAN_DIR}" "${WARM_DIR}" "${WARMK_DIR}" "${SEED1_DIR}" \
        ${FLOW_STEPS_LIST} <<'PY'
import json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
primary = int(sys.argv[2])
mean_k = int(sys.argv[3])
warm_t0 = float(sys.argv[4])
replicate_seed = int(sys.argv[5])
REG, MEAN, WARM, WARMK, SEED1 = sys.argv[6:11]
steps_list = [int(s) for s in sys.argv[11:]]

# ---------------------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS — committed before the first arm was submitted. See the header.
# ---------------------------------------------------------------------------------------
RULE_VERSION = "T30_RULE_V2"
HEADLINE = "skill_vs_repeat_pct"
SIGMA_MULT = 3.0          # BAND = max(SIGMA_MULT * sigma_hat, BAND_FLOOR_PP), as in I8_RULE_V1
BAND_FLOOR_PP = 10.0      # same metric, same holdout as I-8's floor, so the two stay comparable
COLLAPSE_SPREAD_FRAC = 0.10   # labelling threshold for "mode-collapsed", not a verdict gate
SMOOTH_BAND = (0.5, 2.0)
DEMO_RMS = 0.004041
CEILING_MSE = 8.10372e-07
FLOOR_MSE = 1.68201e-05
FLOOR_NEAR_FRAC = 0.20    # "near the floor" = within 20% of FLOOR_MSE; gates verdict 4's MECHANISM
                          # claim, not verdict 4 itself. Worse-than-A by more than BAND is
                          # compatible with landing anywhere below A, including well above the
                          # floor for reasons with nothing to do with step placement (magnitude
                          # scale, the conditioning mismatch the clause below is still deciding).
DEADLINE_MS = 500.0          # ExecutorConfig.policy_deadline_ms
MIN_RATE_HZ = 2.0            # ExecutorConfig.min_policy_rate_hz
# ---------------------------------------------------------------------------------------


def arm(name):
    """bench.json + the chunk RMS + ms/chunk for one arm, or None if it was not scored."""
    bench_path = out / name / "bench.json"
    if not bench_path.is_file():
        return None
    bench = json.load(open(bench_path))
    timing_path = out / name / "timing.json"
    timing = json.load(open(timing_path)) if timing_path.exists() else {}
    total = 0.0
    n = 0
    for line in open(out / name / "predictions.jsonl"):
        if not line.strip():
            continue
        for step in json.loads(line)["predicted"]["targets"]:
            for v in step:
                total += v * v
                n += 1
    bench["rms"] = math.sqrt(total / n) if n else float("nan")
    bench["ms_per_chunk"] = timing.get("ms_per_chunk", float("nan"))
    return bench


def flat_targets(name):
    """Every predicted number of one arm, in file order — the two seed arms are element-wise
    comparable because they were built from the same holdout, the same chunks and the same
    checkpoint, differing only in the noise draw."""
    values = []
    path = out / name / "predictions.jsonl"
    if not path.is_file():
        return values
    for line in open(path):
        if not line.strip():
            continue
        for step in json.loads(line)["predicted"]["targets"]:
            values.extend(step)
    return values


a = arm(REG)
if a is None:
    raise SystemExit("REFUSING TO RULE — the regression arm is missing; there is no A to compare.")
flows = {n: arm(f"eval-t30-flow{n}") for n in steps_list}
b = flows.get(primary)
b_mean = arm(MEAN)
b_warm = arm(WARM)
b_warmk = arm(WARMK)
b_seed1 = arm(SEED1)

zero_mse = a["baselines"]["zero_mse"]


def broken(r):
    """The sampler-broken guard, applied to EVERY arm rather than only the primary.

    An arm that was never scored is not "suspect" — it is absent, which the rule handles by
    name below (a missing mean arm refuses the verdict; a missing warm arm narrows it)."""
    if r is None:
        return []
    reasons = []
    if not math.isfinite(r["mse"]) or not math.isfinite(r["rms"]):
        reasons.append("non-finite predictions")
    if r["rms"] > 3.0 * DEMO_RMS:
        reasons.append(f"RMS {r['rms']:.5f} > 3x the demos' {DEMO_RMS:.5f}")
    if r["mse"] > 5.0 * zero_mse:
        reasons.append(f"mse {r['mse']:.4g} > 5x zero-delta")
    return reasons


labelled = [("regression", a), *((f"flow n={n}", flows[n]) for n in steps_list)]
labelled += [(f"flow mean{mean_k}", b_mean), (f"flow warm{warm_t0:g}", b_warm),
             (f"flow warm+mean{mean_k}", b_warmk), (f"flow seed{replicate_seed}", b_seed1)]
suspect = {label: broken(r) for label, r in labelled if broken(r)}

print(f"{'readout':<16}{'mse':>12}{'skill_vs_repeat':>18}{'smooth':>9}{'RMS':>10}"
      f"{'RMS/demo':>10}{'ms/chunk':>10}")


def row(label, r):
    if r is None:
        print(f"{label:<16}{'(not scored)':>12}")
        return
    flag = "  [SUSPECT]" if label in suspect else ""
    print(f"{label:<16}{r['mse']:>12.4g}{r[HEADLINE]:>17.2f}%"
          f"{r['smoothness_ratio']:>9.3f}{r['rms']:>10.5f}"
          f"{r['rms'] / DEMO_RMS:>10.2f}{r['ms_per_chunk']:>10.1f}{flag}")


for label, r in labelled:
    row(label, r)
print(f"{'demos':<16}{'-':>12}{'-':>18}{1.0:>9.3f}{DEMO_RMS:>10.5f}{1.0:>10.2f}{'-':>10}")
print(f"pre-registered: ceiling {CEILING_MSE:.4g} | floor {FLOOR_MSE:.4g} | "
      f"zero-delta {zero_mse:.4g} | repeat {a['baselines']['repeat_mse']:.4g}")
print()

# Latency: the eval loop's ms/chunk is an UPPER bound on the policy's own latency (it includes
# building the observation), which is the safe direction for a deadline check. The deployable
# candidate is the SINGLE draw — the mean arm is a measurement, so its cost is reported but is
# not what the deadline is checked against.
budget = min(DEADLINE_MS, 1000.0 / MIN_RATE_HZ)
if b is not None:
    overhead = b["ms_per_chunk"] - a["ms_per_chunk"]
    print(f"latency: regression {a['ms_per_chunk']:.1f} ms/chunk, flow n={primary} "
          f"{b['ms_per_chunk']:.1f} ms/chunk (sampler {overhead:+.1f} ms)")
    print(f"         budget {budget:.0f} ms (policy_deadline_ms {DEADLINE_MS:.0f}, "
          f"min_policy_rate_hz {MIN_RATE_HZ:.0f}) -> "
          f"{'FITS' if b['ms_per_chunk'] < budget else 'MISSES THE DEADLINE'}")
    print()

RULE_ARMS = {"regression": a, f"flow n={primary}": b, f"flow mean{mean_k}": b_mean,
             f"flow warm{warm_t0:g}": b_warm, f"flow warm+mean{mean_k}": b_warmk,
             f"flow seed{replicate_seed}": b_seed1}
blocking = {label: suspect[label] for label in RULE_ARMS if label in suspect}
if suspect:
    print("SAMPLER GUARD:")
    for label, reasons in suspect.items():
        print(f"  {label}: {', '.join(reasons)}")
if blocking:
    print()
    print("SAMPLER SUSPECT on a rule-keyed arm — record nothing, this is an integration bug and")
    print("not a result. Check the direction first (t=0 is NOISE, t=1 is CLEAN, z += v*dt with t")
    print("ascending), then the pooled conditioning, then the warm start's (1-t0)*noise + t0*init.")
    print("tests/test_training.py::TestFlowSampler and ::TestFlowSamplerControlArms pin all three.")
    raise SystemExit(0)
if suspect:
    print("  (sweep-only arms; the rule-keyed arms are readable, so the verdict stands and those")
    print("   rows are excluded from the straightness reading.)")
    print()

# -- the band, measured rather than asserted ---------------------------------------------
sigma_hat = None
if b is not None and b_seed1 is not None:
    sigma_hat = abs(b[HEADLINE] - b_seed1[HEADLINE])
BAND = BAND_FLOOR_PP if sigma_hat is None else max(SIGMA_MULT * sigma_hat, BAND_FLOOR_PP)
# The mismatch clause needs its OWN threshold, and it is the one place the floor runs the wrong
# way. For the four headline branches a too-wide band is conservative: it can only push a verdict
# toward "the same on this metric". The mismatch clause is inverted — it is a clause that must FIRE
# to withhold the strong claim, so widening it makes "the mismatch has been MEASURED and rejected"
# EASIER to print. With sigma_hat = 1 pp, a variance-matched warm arm beating A by 9 pp is a real
# seven-sigma effect that the floored 10 pp band would silently swallow. So: use the measured
# 3*sigma_hat when a replicate exists, and fall back to the floor only when nothing was measured.
MISMATCH_BAND = BAND_FLOOR_PP if sigma_hat is None else SIGMA_MULT * sigma_hat

# The draw-to-draw spread, element-wise: E[(X-Y)^2] = 2*Var for two independent draws, so this is
# an unbiased estimate of the sampler's own per-element conditional variance. It is what makes
# "mode-collapsed" a measurement instead of an adjective, and it predicts the single-draw MSE
# penalty the mean arm removes.
xs, ys = flat_targets(f"eval-t30-flow{primary}"), flat_targets(SEED1)
var_hat = None
if xs and len(xs) == len(ys):
    var_hat = sum((x - y) ** 2 for x, y in zip(xs, ys)) / (2.0 * len(xs))

print(f"-- {RULE_VERSION}")
print(f"  sigma_hat |seed0-seed{replicate_seed}| : "
      f"{'UNMEASURED' if sigma_hat is None else f'{sigma_hat:.2f} pp'}")
print(f"  BAND = max({SIGMA_MULT:.0f}*sigma, {BAND_FLOOR_PP:.0f} pp) : {BAND:.2f} pp"
      + ("   [FLOOR ONLY — no replicate]" if sigma_hat is None else ""))
print("  (measured on SINGLE draws, so it over-states the mean-of-k arm's own spread by about")
print(f"   sqrt({mean_k}); for the four HEADLINE branches that is the conservative direction —")
print("   it can only push the verdict toward 'the same on this metric', never toward a")
print("   positive claim. It is NOT conservative for the mismatch clause, which must fire to")
print("   withhold a claim, so that clause uses MISMATCH_BAND instead.)")
print(f"  MISMATCH_BAND (clause only) : {MISMATCH_BAND:.2f} pp"
      + ("   [FLOOR ONLY — see caveat if it does not fire]" if sigma_hat is None else ""))
if var_hat is not None:
    print(f"  draw spread   sqrt(Var) {math.sqrt(var_hat):.5f} = "
          f"{math.sqrt(var_hat) / DEMO_RMS:.2f}x the demos' RMS")
    print(f"  the MSE identity: one draw carries +{var_hat:.4g} over the conditional mean, of "
          f"which mean{mean_k} leaves {var_hat / mean_k:.4g}")
    if b is not None and b_mean is not None:
        print(f"  observed mse(draw) - mse(mean{mean_k}) = {b['mse'] - b_mean['mse']:+.4g}   "
              f"[predicted {var_hat * (1.0 - 1.0 / mean_k):+.4g} if the sampler is unbiased]")
print()

if b_mean is None:
    raise SystemExit("REFUSING TO RULE — the mean-of-k arm is missing, and it is what the rule "
                     "keys on. A single draw against a mean-seeking head is not the comparison "
                     "this rule was written to make.")

delta_mean = b_mean[HEADLINE] - a[HEADLINE]
print(f"skill_vs_repeat_pct   A {a[HEADLINE]:+.2f}%   ->   B_mean {b_mean[HEADLINE]:+.2f}%"
      f"   (delta {delta_mean:+.2f} pp, band {BAND:.2f} pp)")
if b is not None:
    print(f"                      single draw {b[HEADLINE]:+.2f}% "
          f"(delta {b[HEADLINE] - a[HEADLINE]:+.2f} pp — reported, NOT the key: MSE charges a "
          "draw the conditional variance)")
if b_warm is not None:
    print(f"                      warm t0={warm_t0:g} {b_warm[HEADLINE]:+.2f}% "
          f"(vs A {b_warm[HEADLINE] - a[HEADLINE]:+.2f} pp"
          + (f", vs single draw {b_warm[HEADLINE] - b[HEADLINE]:+.2f} pp — reported, NOT a "
             "trigger: a warm draw carries less variance)" if b is not None else ")"))
if b_warmk is not None:
    print(f"                      warm+mean{mean_k} {b_warmk[HEADLINE]:+.2f}% "
          f"(vs A {b_warmk[HEADLINE] - a[HEADLINE]:+.2f} pp, vs B_mean "
          f"{b_warmk[HEADLINE] - b_mean[HEADLINE]:+.2f} pp) <- the mismatch clause's key")
print()

if delta_mean > BAND and b_mean[HEADLINE] > 0:
    print("VERDICT: THE FLOW READOUT SCORES HIGHER, and the mean-of-k arm clears L1.")
    print("         CONFOUND (a) APPLIES TO THIS BRANCH TOO. Only the regression head is charged")
    print("         smoothness_loss (weights.smoothness = 0.01 on decoded_targets); the flow")
    print("         branch is charged nothing. A handicapped readout losing to an unhandicapped")
    print("         one is the objective's own consequence, so 'the mean-seeking head was the")
    print("         T-16 negative' is ONE reading of this outcome and 'the jerk regulariser was")
    print("         the T-16 negative' is another. This job cannot separate them — that needs the")
    print("         weights.smoothness = 0 retrain named in the header. Record the delta as a")
    print("         READOUT result, and CORRECT the T-16 row in docs/benchmark.md to say the")
    print("         negative was not a property of the checkpoint's action knowledge.")
    if b is None:
        print("         DEPLOYMENT: NOT AUTHORISED by this branch — the single-draw arm did not")
        print("         run. mean-of-k is a measurement, never a policy, so nothing here has")
        print("         measured the arm that would actually ship.")
    elif b[HEADLINE] > 0:
        print(f"         DEPLOYMENT AUTHORISED: the SINGLE-draw arm is {b[HEADLINE]:+.2f}% and")
        print("         clears the bar on its own. Make it the deployed path and open a task for")
        print("         the closed-loop half (rolling buffer + measured latency in rollout.py).")
    else:
        print(f"         DEPLOYMENT: NOT AUTHORISED — the single draw is {b[HEADLINE]:+.2f}%, still")
        print("         below repeat-last-action. B_mean cleared the bar by averaging away the")
        print("         conditional variance, which is the mean-seeking this branch exists to")
        print("         avoid; shipping it would re-introduce the defect under test. Keep the")
        print("         regression head deployed and open a task for a sampler whose SINGLE draw")
        print("         clears the bar.")
elif delta_mean > BAND:
    print("VERDICT: better, still below the bar. The head cost real skill and was not the whole")
    print("         story. Same confound as the branch above: the smoothness regulariser charges")
    print("         only the regression head, so part of this gap is the objective and not the")
    print("         readout, and this job cannot say how much. Record the delta, keep the")
    print("         regression head deployed until a closed-loop number exists, then I-8.")
elif delta_mean >= -BAND:
    print("VERDICT: THE TWO READOUTS SCORE THE SAME ON THIS METRIC. That is the finding, stated")
    print("         at the strength the evidence supports: a delta inside a band measured from")
    print("         one seed replicate. It is NOT 'the flow branch is decorative' unless the")
    print("         columns below say so, and it is NOT a statement about the branch as trained")
    print("         — only about it AS SAMPLED THIS WAY (one t=1 backbone pass reused at every")
    print("         t_k). Weights.action_flow = 1.0 bought nothing MEASURABLE ON THIS METRIC in")
    print("         20 000 steps.")
else:
    print("VERDICT: worse. The mean was doing real work.")
    near_floor = abs(b_mean["mse"] - FLOOR_MSE) <= FLOOR_NEAR_FRAC * FLOOR_MSE
    if near_floor:
        print(f"         B_mean's mse {b_mean['mse']:.4g} sits within "
              f"{FLOOR_NEAR_FRAC:.0%} of the pre-registered floor")
        print(f"         {FLOOR_MSE:.4g} — the score of a chunk with the right CONTENT and no")
        print("         position. That is the mechanism: the flow branch AS SAMPLED THIS WAY")
        print("         cannot PLACE a chunk in time — each step is an independent draw and")
        print("         ActionVelocityHead takes no step index. The fix is a retrain")
        print("         (step-conditioned velocity head / cross-step attention) and belongs to")
        print("         I-2, not to a decode change.")
    else:
        print(f"         B_mean's mse {b_mean['mse']:.4g} is NOT near the floor "
              f"{FLOOR_MSE:.4g} (within {FLOOR_NEAR_FRAC:.0%} was the pre-registered test), so")
        print("         the step-placement mechanism is a HYPOTHESIS here, not a reading of the")
        print("         number. Being worse than A by more than the band is compatible with")
        print("         landing anywhere below it — magnitude scale, or the conditioning")
        print("         mismatch the clause below is about. Record the negative; do NOT record a")
        print("         cause, and do not send this to I-2 on this evidence alone.")

# -- the mismatch clause: what a negative is ABOUT ----------------------------------------
if delta_mean <= BAND:
    print()
    if b_warmk is None:
        print("         MISMATCH CLAUSE: the variance-matched warm arm did not run, so nothing")
        print("         here distinguishes 'the branch is dead' from 'we sampled it outside its")
        print("         training region'. Run it (--flow-t0 with --flow-mean-k) before recording")
        print("         any verdict about the flow BRANCH; what is recordable without it is a")
        print("         verdict about this SAMPLER only.")
    elif (b_warmk[HEADLINE] - a[HEADLINE]) > MISMATCH_BAND or (
        b_warmk[HEADLINE] - b_mean[HEADLINE]
    ) > MISMATCH_BAND:
        print("         MISMATCH CLAUSE FIRES: the warm start, which only ever evaluates the head")
        print("         where its t and the features' t=1 roughly agree, clears the band. The")
        print("         negative above is then attributable to the sampler's conditioning")
        print("         mismatch and is NOT a statement about the flow branch. Next step is the")
        print("         SAMPLER (t-conditioned features, or a retrain that noises the observation")
        print("         alongside the action) — not I-2. Note the warm start inherits the")
        print("         regression mean by construction, so this is evidence about WHERE the head")
        print("         works, never that the branch models the conditional.")
    else:
        print(f"         MISMATCH CLAUSE DOES NOT FIRE (threshold {MISMATCH_BAND:.2f} pp = "
              + ("the floor, UNMEASURED" if sigma_hat is None
                 else f"{SIGMA_MULT:.0f}*sigma_hat, measured") + "): the warm start is inside")
        print("         it too, so the conditioning mismatch has been MEASURED and rejected as")
        print("         the explanation. The negative extends to the branch under every protocol")
        print("         runnable offline without a retrain. It still does not extend to the")
        print("         faithful sampler (n backbone passes on a noised observation), which was")
        print("         never run and which no arm here approximates.")
        if sigma_hat is None:
            print("         CAVEAT: no replicate ran, so this clause was thresholded on the bare")
            print("         10 pp floor. Unlike the branches above, a too-wide threshold here is")
            print("         ANTI-conservative — it makes 'rejected as the explanation' easier to")
            print("         print. Treat this line as unproven until B_seed1 has run.")
        if var_hat is not None and b is not None:
            collapsed = (math.sqrt(var_hat) < COLLAPSE_SPREAD_FRAC * DEMO_RMS) and (
                abs(b["rms"] - DEMO_RMS) >= abs(a["rms"] - DEMO_RMS)
            )
            if collapsed:
                print(f"         MODE COLLAPSE: draw spread {math.sqrt(var_hat):.5f} is under "
                      f"{COLLAPSE_SPREAD_FRAC:.0%} of the demos' RMS and the flow arm is no")
                print("         closer to the demonstrated magnitude than the regression head is.")
                print("         The velocity head is on one average; I-2 (cross-attention")
                print("         conditioning, which feeds BOTH heads) is the lever left before I-8.")
            else:
                print(f"         NOT mode collapse: the draws differ by {math.sqrt(var_hat):.5f} "
                      f"({math.sqrt(var_hat) / DEMO_RMS:.2f}x the demos' RMS). The sampler varies;")
                print("         what it varies over just does not score better. Say that, not")
                print("         'mode-collapsed'.")

# -- descriptive columns, deliberately not gates -------------------------------------------
print()
print("smoothness_ratio (DESCRIPTIVE — weights.smoothness = 0.01 penalises the REGRESSION head's")
print("  jerk and nothing else, so a gap between the readouts is predicted by the objective):")
for label, r in labelled:
    if r is None or label in suspect:
        continue
    if SMOOTH_BAND[0] <= r["smoothness_ratio"] <= SMOOTH_BAND[1]:
        note = "inside the demonstrated band"
    elif r["smoothness_ratio"] > SMOOTH_BAND[1]:
        note = "above the band — per-step independence showing up as jerk"
    else:
        note = "below the band — smoother than the demonstrations"
    print(f"  {label:<16}{r['smoothness_ratio']:>7.3f}  {note}")
print()
print("STANDING WEAKNESS, unchanged: one checkpoint, one task, one training seed. sigma_hat here")
print("  is a SAMPLING replicate — it covers the draw, not the retrain. A claim about 'the flow")
print("  branch' across training seeds needs runs/t16-lora-seed1 scored the same way.")
PY

echo
echo "Copy back to the Mac with cluster/discoverer/sync.sh, then re-score on CPU forever:"
echo "  python scripts/run_bench.py runs/${RUN_ID}/${REG_DIR} runs/${RUN_ID}/${MEAN_DIR} --compare"
