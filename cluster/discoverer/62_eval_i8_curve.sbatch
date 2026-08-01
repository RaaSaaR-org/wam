#!/bin/bash
# Step 11 — I-8 / T-32: score every rung and print the pre-registered verdict.
#
# THIS FILE IS THE PRE-REGISTRATION. Every threshold below is a literal, and this file is in
# git before the first rung is submitted. Changing a threshold after seeing the number it
# scored is precisely the move pre-registration exists to prevent — the repo already records
# that lesson about L4's one-sided gate (docs/improvements.md, I-9). If a constant genuinely
# has to change, version it: bump RULE_VERSION, record what moved and why, and say which runs
# scored under which. Do not edit a number in place.
#
# The rule is currently I8_RULE_V3 (2026-08-01). V1 and V2 were both superseded before any rung
# reported — the changelog is next to the constants below. Every verdict this script prints must
# carry RULE_VERSION with it wherever it is recorded.
#
# SYMMETRY IS PART OF THE PRE-REGISTRATION. Verdicts A and C are equally strong claims about
# the world in opposite directions, and the equal-STEPS design's confound (147.5 passes at
# N=40 against 16.9 at N=362) can manufacture either one. Both are therefore gated on the same
# evidence. A rule that holds only the expensive conclusion to a standard is not a decision
# rule, it is a preference.
#
#   sbatch 62_eval_i8_curve.sbatch
#
# One GPU, minutes per pass: 40 holdout episodes is 1 040 chunks and each is ONE forward at
# the clean flow timestep (no denoising loop, no video generated). No --requeue, same reason as
# 60_eval_t16.sbatch: idempotent, writes no DONE, and re-scoring costs minutes.
#
# Two frame modes per rung and nothing else. The readout A/B (T-30, 63_eval_t30_flow_head)
# stays out: running both at once turns two 1-D questions into a 2x2 nobody pre-registered.
#
#SBATCH --job-name=wam-eval-i8
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26           # >26 threads/GPU burns billing faster than GPU-hours
#SBATCH --gres=gpu:1
#SBATCH --mem=192G                   # keep <=257G for the same billing reason
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/eval-i8.%j.out

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
if [[ ! -f "${EVAL}" ]]; then
  echo "FATAL: ${EVAL} missing — run cluster/discoverer/sync.sh from the Mac first."
  exit 1
fi

RUNS=${PROJ}/runs
DATASET=${PROJ}/data/gr00t-apple-full
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt

# Rung 362 is runs/t16-lora-seed0, already scored in both modes by
# 61_eval_t29_frame_history.sbatch. Its bench.json files are consumed, not recomputed.
# That job reported on 2026-08-01 (Slurm 184648), so both files exist and this gate passes; it
# stays because a fresh clone or a partial sync.sh --pull is the case it was written for.
for f in "${RUNS}/t16-lora-seed0/eval-t29-tiled/bench.json" \
         "${RUNS}/t16-lora-seed0/eval-t29-history/bench.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "FATAL: ${f} missing — 61_eval_t29_frame_history.sbatch has not reported."
    echo "       That job IS rung 362. Run it first."
    exit 1
  fi
done

for RUN_ID in i8-rung040-seed0 i8-rung120-seed0; do
  if [[ ! -f "${RUNS}/${RUN_ID}/DONE" ]]; then
    echo "FATAL: ${RUNS}/${RUN_ID}/DONE missing — that rung has not finished its step budget."
    echo "       Scoring a partial rung against a finished one measures wall clock, not data."
    exit 1
  fi
done
if [[ ! -f "${RUNS}/i8-rung040-seed1/DONE" ]]; then
  echo "WARNING: no ${RUNS}/i8-rung040-seed1/DONE — the seed replicate is missing."
  echo "         sigma_hat cannot be measured, so the rule below REFUSES outcome A."
fi

# The runs differ in EXACTLY one flag between the two halves. Everything else — dataset,
# holdout, camera, device — is held identical by construction rather than by copy-paste.
# Each rung needs its OWN witness. 55_train_i8_rung.sbatch trains every rung with
# --train-episodes "${SPLIT}", so each checkpoint records an explicit training set and
# eval_t16.verify_split takes the DISJOINTNESS path, which refuses without an external witness —
# the recorded ids and the recorded hash are two fields of one self-description, so without a
# file the checkpoint did not write the check compares the checkpoint against itself. Passing the
# wrong rung's file is refused loudly rather than scored, which is the point of the anchor.
declare -A TRAIN_SPLIT=(
  [i8-rung040-seed0]=i8_train_040.txt
  [i8-rung120-seed0]=i8_train_120.txt
  [i8-rung040-seed1]=i8_train_040.txt
)

for RUN_ID in i8-rung040-seed0 i8-rung120-seed0 i8-rung040-seed1; do
  OUT=${RUNS}/${RUN_ID}
  # DONE, not just the directory. The two headline rungs already died above without it; this loop
  # also carries i8-rung040-seed1, whose only gate was a WARNING that promised a consequence the
  # code did not implement. An unfinished replicate exists as a directory the moment its first
  # requeue pass starts, so `-d` alone scored it and fed sigma_hat = |s40 - replicate| from a run
  # that had not reached its step budget. That inflates sigma_hat, which inflates MATERIAL, which
  # makes SPAN >= MATERIAL and STEEP harder — routing the verdict to C. The header of this file
  # says a rule that holds only the expensive conclusion to a standard is a preference, not a
  # decision rule; admitting a partial replicate made the CHEAP conclusion the easy one to reach.
  # Unmeasured is the safe state: sigma_hat stays None and the rule refuses outcome A outright.
  [[ -d "${OUT}" && -f "${OUT}/DONE" ]] || { echo "skip ${RUN_ID} (no finished run)"; continue; }
  SPLIT=${PROJ}/wam/configs/splits/${TRAIN_SPLIT[${RUN_ID}]}
  if [[ ! -f "${SPLIT}" ]]; then
    echo "FATAL: ${SPLIT} missing — that is the witness the split proof anchors on."
    exit 1
  fi
  common=(
    --run-dir "${OUT}"
    --checkpoint latest
    --dataset "${DATASET}"
    --holdout "${HOLDOUT}"
    --train-episodes "${SPLIT}"
    --camera ego
    --device cuda
  )
  echo "=== ${RUN_ID} · A: tiled single frame (how everything before 2026-08-01 was measured)"
  python "${EVAL}" "${common[@]}" --out "${OUT}/eval-i8-tiled"
  echo "=== ${RUN_ID} · B: real frame window (what training fed the model) — THE HEADLINE"
  python "${EVAL}" "${common[@]}" --frame-history --out "${OUT}/eval-i8-history"
done

echo
echo "=== the curve and the pre-registered verdict (docs/improvements.md I-8, TASKS.md T-32)"
python - "${RUNS}" <<'PY'
import json, math, random, sys
from pathlib import Path

# ---------------------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS — committed before the first rung was submitted.
#
# I8_RULE_V1 -> I8_RULE_V2 (2026-08-01), versioned rather than edited in place, per this
# file's own standing instruction. V1 was NEVER applied to a measured curve: it is superseded
# before any rung reported, so no recorded verdict is being restated. What changed and why:
#
#   1. STEEP read a POINT estimate of N* — an OLS extrapolation from three points, 2.8
#      doublings past the largest measured N, with one residual degree of freedom and no
#      interval — and it gated verdict A, which buys a G1 EDU4 and months of recording.
#      V2 propagates the measured sigma_hat through the fit and requires the WHOLE one-sigma
#      N* interval to sit below N_REACH. Strictly harder to pass than V1.
#   2. Verdict C ("not data-limited") required nothing, while verdict A was held PROVISIONAL
#      for the equal-STEPS / unequal-EPOCHS confound. That confound is SYMMETRIC: 147.5
#      passes at N=40 against 16.9 at N=362 can manufacture a flat curve exactly as easily as
#      a steep one, so C was the verdict the confound could produce for free. V2 gates C on
#      the same evidence as A. Strictly harder to pass than V1.
#   3. The MONOTONE annotation claimed P=1/3 for ONE specified increasing ordering of three
#      exchangeable values. Three values have 3! = 6 orderings, so one specified ordering is
#      P=1/6; 1/3 is monotone in EITHER direction, which is not what the flag tests.
#      Annotation only — MONOTONE's definition is untouched.
#
# I8_RULE_V2 -> I8_RULE_V3 (2026-08-01), same standing instruction, still before any rung has
# reported. V2 fixed the asymmetry between A and C and then left the same hole in two places:
#
#   4. Verdict B carried NO epoch caveat. "More of this data improves the model" is a causal
#      data-volume claim of exactly A's and C's strength, and the same 147.5-vs-16.9 epoch gap
#      manufactures a RISING curve as readily as a flat one — V2's own item 2 says so. V3 gates
#      B on EPOCH_INFLATED too. The header's stated principle ("a rule that holds only the
#      expensive conclusion to a standard is not a decision rule, it is a preference") applies
#      verbatim to B; V2 applied it to two branches out of three.
#   5. Verdict C's TEXT was unconditional on why C fired. C is `not (MONOTONE and SPAN >=
#      MATERIAL)`, which is two different worlds, and V2 printed "more of this data did not move
#      the headline" for both. A reachable case: s40=-50, s120=-70, s362=-20, sigma_hat=4 — D
#      does not fire, MONOTONE is False, so C prints with SPAN=+30 pp, and both of its printed
#      sentences are false. V3 splits out C-NOISY, which claims no curve rather than a flat one
#      and routes to a seed replicate instead of to I-9.
#   6. The C-branch note on the MATERIAL floor stated its direction BACKWARDS ("underestimating
#      it can only push the verdict toward C"). A smaller MATERIAL makes `SPAN >= MATERIAL`
#      easier, i.e. pushes toward A/B. V2's conclusion was right and its reason inverted; the
#      sibling note in 63_eval_t30_flow_head.sbatch has the direction right, so this was an
#      error rather than a convention. Text only.
#
# No numeric threshold moved in EITHER revision: SIGMA_MULT, MATERIAL_FLOOR_PP, N_REACH,
# N_REACH_HIGH and HEADLINE are V1's values, unchanged. Every rule change so far tightens a gate
# or splits an over-broad one; none was written with a measured rung in view, because there is
# not one yet.
# ---------------------------------------------------------------------------------------
RULE_VERSION = "I8_RULE_V3"
SIGMA_MULT    = 3.0     # MATERIAL = max(SIGMA_MULT * sigma_hat, MATERIAL_FLOOR_PP)
MATERIAL_FLOOR_PP = 10.0
# N_REACH: how many episodes a collection campaign we would actually commit to can produce.
# docs/ROADMAP.md D2 says "MVP set (10-30 h)". Measured on datasets/gr00t-apple-full: 402
# episodes, mean 14.2 s each (manifest t1_ns - t0_ns) = 1.59 h. The LOW end of the ROADMAP
# range is the number a campaign can be counted on to deliver, so the gate uses it; the high
# end is reported next to it so a reader can apply the optimistic bar themselves.
N_REACH      = 2500     # ~10 h of demonstration time at 14.2 s/episode
N_REACH_HIGH = 7600     # ~30 h, reported only
HEADLINE = "skill_vs_repeat_pct"   # L1: better than repeating the last action

# N* interval (V2). Three points with a known per-point sigma is a parametric bootstrap:
# perturb each rung's score by N(0, sigma_hat), refit, and read the spread of N*. Draws and
# seed are literals so the interval is reproducible from the archived scores alone; 16/84 is
# the two-sided ONE-sigma band, chosen to match the one-sigma language of the fix, not to a
# tail probability nobody would defend on three points.
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED  = 20260801
BOOTSTRAP_LO_PCT, BOOTSTRAP_HI_PCT = 16.0, 84.0
# ---------------------------------------------------------------------------------------

runs = Path(sys.argv[1])
# (N, run_id, eval-dir prefix). Rung 362 is t16-lora-seed0 REUSED, not re-trained: identical
# config hash 45ee9e60, identical 362 episodes, identical 20 000 steps, seed 0. It therefore
# also proves its split by COMPLEMENT (it predates train_episode_ids) while the rungs prove
# theirs by DISJOINTNESS. Both are printed by eval_t16.py above; the difference is in what is
# proven about exhaustiveness, not about whether the holdout was seen.
CURVE = [(40, "i8-rung040-seed0", "eval-i8"), (120, "i8-rung120-seed0", "eval-i8"),
         (362, "t16-lora-seed0", "eval-t29")]
REPLICATE = (40, "i8-rung040-seed1", "eval-i8")


def bench(run_id, prefix, mode):
    path = runs / run_id / f"{prefix}-{mode}" / "bench.json"
    return json.loads(path.read_text()) if path.is_file() else None


def final_action_reg(run_id):
    """Last logged action_reg — the regime diagnostic the equal-STEPS design buys.

    Rung 40 runs 147.5 epochs against rung 362's 16.9, so if the curve moves because the small
    rungs were fitted in a different regime rather than because data is short, it shows up here
    as a training loss that collapses at small N. Free: it is already in the run log.

    LOAD-BEARING under I8_RULE_V3, not decoration: it gates whether verdicts B and C are final
    EPOCH_INFLATED). Returns None when training_log.jsonl is absent, and None is treated as
    "confound present" downstream — a diagnostic nobody could read is not a passing one."""
    path = runs / run_id / "training_log.jsonl"
    if not path.is_file():
        return None
    for line in reversed(path.read_text().splitlines()):
        record = json.loads(line) if line.strip() else {}
        if record.get("kind") == "step" and "action_reg" in record:
            return record["action_reg"]
    return None


# Which mode is the headline was an assumption when this file was committed; T-29 measured it
# (2026-08-01). On t16-lora-seed0 — one checkpoint, the same 40-episode holdout, the same 1 040
# chunks, differing in exactly one flag — the real window scores -21.80 % against tiled's
# -32.45 %, and the tiled arm reproduces the archived eval-latest/bench.json to every digit. So
# --frame-history is the in-distribution mode, the one training actually fed the model, and the
# rule below reads it; the tiled column stays as the bridge to every number recorded before
# 2026-08-01, all of which are tiled numbers.
for mode in ("history", "tiled"):
    label = "--frame-history (HEADLINE)" if mode == "history" else "tiled (cross-check only)"
    print(f"\n-- {label}")
    print(f"{'N':>4} {'run':<18} {'chunks':>6} {'skill_vs_rep':>13} {'ci_skill':>9} "
          f"{'mse':>11} {'smooth':>7}  level")
    for n, run_id, prefix in [*CURVE, REPLICATE]:
        b = bench(run_id, prefix, mode)
        if b is None:
            print(f"{n:>4} {run_id:<18} {'—':>6}  (not scored)")
            continue
        print(f"{n:>4} {run_id:<18} {b['num_predictions']:>6} {b[HEADLINE]:>+12.2f}% "
              f"{b['ci_skill_vs_repeat_pct']:>+8.2f}% {b['mse']:>11.4g} "
              f"{b['smoothness_ratio']:>7.3f}  {b['level_name']}")

print("\n-- regime diagnostic (final training action_reg)")
print("   equal STEPS means 147.5 / 50.0 / 16.9 epochs at N = 40 / 120 / 362")
for n, run_id, _ in [*CURVE, REPLICATE]:
    value = final_action_reg(run_id)
    print(f"{n:>4} {run_id:<18} {'n/a' if value is None else f'{value:.6g}'}")

# -- the rule ---------------------------------------------------------------------------
primary = {n: bench(run_id, prefix, "history") for n, run_id, prefix in CURVE}
if any(b is None for b in primary.values()):
    raise SystemExit("REFUSING TO RULE — a rung is missing from the --frame-history curve.")
s40, s120, s362 = (primary[n][HEADLINE] for n in (40, 120, 362))

replicate = bench(REPLICATE[1], REPLICATE[2], "history")
# sigma_hat is |seed0 - seed1| at rung 40, not a standard deviation: for two samples the sd is
# |diff|/sqrt(2), so this OVERSTATES the per-point sigma by ~1.41x. Kept as V1 defined it —
# the overstatement widens the N* interval and therefore only makes STEEP (the gate on the
# expensive verdict) harder to pass. It is also measured at ONE rung and assumed to hold at
# the other two, which is an assumption this design cannot check with three points.
sigma_hat = None if replicate is None else abs(s40 - replicate[HEADLINE])
MATERIAL = MATERIAL_FLOOR_PP if sigma_hat is None else max(SIGMA_MULT * sigma_hat,
                                                           MATERIAL_FLOOR_PP)
MONOTONE = s40 < s120 < s362
SPAN = s362 - s40

# EPOCH_INFLATED — the equal-STEPS design's confound, made testable (V2). Rung 40 sees 147.5
# passes over 1 085 chunks while rung 362 sees 16.9 over 9 476: the two ends of the curve were
# not fitted in the same regime, and this experiment cannot separate "regime" from "data
# volume". A final training action_reg that is LOWER at N=40 than at N=362 despite ~8.7x fewer
# distinct chunks is that regime difference showing itself. Note what this is NOT: it is not a
# claim about which DIRECTION the bias runs (more passes can flatter a small rung by finishing
# its convergence or punish it by memorising), and that is exactly the point — once the ends
# are in different regimes, neither a flat curve nor a steep one is attributable to N.
# It can genuinely come out False: the LoRA rank caps capacity, the LR is constant, and rung
# 40's 1 085 chunks are a different (possibly harder) mix, so its loss floor is not forced to
# sit below rung 362's. Unmeasured (missing training_log.jsonl) counts as inflated: an absent
# diagnostic is not evidence that the confound is quiet.
reg40, reg362 = final_action_reg(CURVE[0][1]), final_action_reg(CURVE[2][1])
EPOCH_INFLATED = True if (reg40 is None or reg362 is None) else (reg40 < reg362)

# OLS of skill on log2(N) over the three rungs; N* is where the fit reaches the L1 bar (0 %).
xs = [math.log2(n) for n in (40, 120, 362)]
ys = [s40, s120, s362]
xbar = sum(xs) / 3
denom = sum((x - xbar) ** 2 for x in xs)


def fit(values):
    """(intercept, slope, N*) for skill-on-log2(N). N* = inf when the slope is not positive."""
    ybar = sum(values) / len(values)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, values)) / denom
    intercept = ybar - slope * xbar
    if slope <= 0:
        return intercept, slope, math.inf
    exponent = -intercept / slope
    # 2**1024 overflows; a fit that needs more doublings than that is "unreachable" anyway.
    return intercept, slope, (math.inf if exponent > 1000.0 else 2.0 ** exponent)


intercept, slope, n_star = fit(ys)

# Parametric bootstrap of N*. Every draw refits BOTH slope and intercept, so the interval
# carries the intercept's uncertainty too — not just "N* at slope +/- one sigma_hat", which
# would hold the centroid fixed and understate the spread.
n_star_lo = n_star_hi = None
if sigma_hat is not None:
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        fit([y + rng.gauss(0.0, sigma_hat) for y in ys])[2] for _ in range(BOOTSTRAP_DRAWS)
    )
    n_star_lo = draws[int(round(BOOTSTRAP_LO_PCT / 100.0 * (BOOTSTRAP_DRAWS - 1)))]
    n_star_hi = draws[int(round(BOOTSTRAP_HI_PCT / 100.0 * (BOOTSTRAP_DRAWS - 1)))]


def steep_at(bar):
    """Does the curve reach L1 inside ``bar`` episodes? Whole interval, not the point estimate.

    V2: the point estimate alone is one number out of a 3-point fit with one residual degree of
    freedom, and it gated the most expensive decision in the project. With no replicate there
    is no interval, and this falls back to the point estimate — safe only because the
    sigma_hat-is-None branch below can reach nothing stronger than "A WITHHELD".
    """
    if slope <= 0 or not 0 < n_star <= bar:
        return False
    if sigma_hat is None:
        return True
    return n_star_lo is not None and 0 < n_star_lo and n_star_hi <= bar


STEEP_POINT = slope > 0 and 0 < n_star <= N_REACH
STEEP = steep_at(N_REACH)
STEEP_HIGH = steep_at(N_REACH_HIGH)


def n_star_num(value):
    """One N* as a bare number. 'unreachable' is a real outcome, not a formatting failure."""
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "unreachable"
    return f"{value:.0f}" if value < 1e6 else f"{value:.3g}"


def n_star_text(value):
    return "unreachable — the fit never rises to L1" if (
        value is not None and not math.isfinite(value)
    ) else f"{n_star_num(value)} episodes"


def interval_text():
    """The one-sigma N* interval, or why there isn't one."""
    if sigma_hat is None:
        return "interval UNAVAILABLE — no replicate, so no sigma_hat to propagate"
    if not math.isfinite(n_star_lo) and not math.isfinite(n_star_hi):
        return "one-sigma interval: unreachable throughout"
    return (f"one-sigma interval {n_star_num(n_star_lo)} .. {n_star_num(n_star_hi)} episodes "
            f"({BOOTSTRAP_DRAWS} draws @ sigma_hat={sigma_hat:.2f} pp, seed {BOOTSTRAP_SEED})")


print(f"\n-- {RULE_VERSION}")
print(f"  s40 {s40:+.2f}%   s120 {s120:+.2f}%   s362 {s362:+.2f}%")
print(f"  MONOTONE (s40<s120<s362) : {MONOTONE}   "
      "[necessary, NEVER sufficient: P=1/6 for ONE specified increasing ordering of three "
      "exchangeable noisy values — 3! = 6 orderings; 1/3 would be monotone in EITHER "
      "direction, which is not what this flag tests]")
print(f"  sigma_hat |seed0-seed1|  : "
      f"{'UNMEASURED' if sigma_hat is None else f'{sigma_hat:.2f} pp'}")
print(f"  MATERIAL = max(3*sigma, {MATERIAL_FLOOR_PP:.0f}) : {MATERIAL:.2f} pp"
      + ("   [FLOOR ONLY — no replicate]" if sigma_hat is None else ""))
print(f"  SPAN = s362 - s40        : {SPAN:+.2f} pp")
print(f"  final action_reg 40/362  : {'n/a' if reg40 is None else f'{reg40:.6g}'} / "
      f"{'n/a' if reg362 is None else f'{reg362:.6g}'}")
print(f"  EPOCH_INFLATED (reg40<reg362, or unmeasured) : {EPOCH_INFLATED}")
print(f"  fit s = {intercept:.2f} + {slope:.2f}*log2(N)  ->  N* = {n_star_text(n_star)}")
print(f"  {interval_text()}")
print(f"  STEEP (whole interval <= {N_REACH}) : {STEEP}"
      + (f"   [V1's point estimate alone would say: {STEEP_POINT}]"
         if sigma_hat is not None else "")
      + f"   [at the optimistic {N_REACH_HIGH}-episode bar: {STEEP_HIGH}]")
print("  CAVEAT the fit is log-linear in N on a metric BOUNDED ABOVE by 100%, extrapolated")
print("         2.8 doublings past the largest measured N. It cannot be the true functional")
print("         form out there. If the real curve saturates before the L1 bar, N* is an")
print("         UNDER-estimate or does not exist at all — so an N* that only just fits inside")
print("         N_REACH is the optimistic reading of the fit, never the safe one.")

print()
if (s40 - s362) >= MATERIAL:
    print("VERDICT D · INVERTED. More data makes it materially worse across a 9x span, which is")
    print("  not a data-volume story at all. It points at the training setup (constant LR, step")
    print("  budget, the multi-task tax T-18 measured at -89.5%). Stop: no scaling claim in")
    print("  either direction until this is explained.")
elif MONOTONE and SPAN >= MATERIAL and STEEP and sigma_hat is None:
    print("VERDICT A WITHHELD · the curve has A's shape, but sigma_hat is UNMEASURED, so MATERIAL")
    print(f"  is the bare {MATERIAL_FLOOR_PP:.0f} pp floor and nothing here rules out a movement a")
    print("  second seed would reproduce as noise. A is the expensive conclusion — it buys a G1")
    print("  EDU4 and months of recording — and it does not get to be reached by an untested")
    print("  threshold. Run RUNG=040 SEED=1 sbatch 55_train_i8_rung.sbatch (~36 GPU-h), re-score.")
elif MONOTONE and SPAN >= MATERIAL and STEEP:
    print("VERDICT A (PROVISIONAL) · BUILD THE DATASET. Data volume is the binding constraint")
    print(f"  and a collectable amount fixes it: the fit reaches L1 at N* = {n_star_text(n_star)},")
    print(f"  {interval_text()},")
    print(f"  and the WHOLE interval is inside the {N_REACH}-episode bar.")
    print("  A IS NOT FINAL until the stage-2 equal-EPOCH control has run at rungs 40 and 120")
    print("  (STEPS=2290 / 6748). If that control is flat or non-monotone, the movement here is")
    print("  the small rungs overfitting, not data, and this downgrades to C — see A-DOWNGRADED.")
elif MONOTONE and SPAN >= MATERIAL:
    print("VERDICT B · DATA HELPS, NOT ENOUGH. More of this data improves the model but cannot be")
    print("  SHOWN to reach L1 inside a campaign we could actually run: the fit reaches L1 at")
    print(f"  N* = {n_star_text(n_star)},")
    print(f"  {interval_text()},")
    print(f"  which does not sit wholly inside the {N_REACH}-episode bar D2 can deliver. Months of")
    print("  recording the SAME task is not justified by this result. I-9 (the kind of data — the")
    print("  gripper that never opens) and I-2 (architecture) come first.")
    if sigma_hat is not None and STEEP_POINT:
        print("  (Record verbatim: the POINT estimate of N* is inside the bar and the interval is")
        print("   not. B is 'we cannot show it reaches L1 in reach', not 'we showed it does not'.)")
    if EPOCH_INFLATED:
        # V3. B's first clause — "more of this data improves the model" — is a causal data-volume
        # claim of the same strength as A's and C's, and 147.5 epochs at N=40 against 16.9 at
        # N=362 manufactures a RISING curve as readily as a flat one. V2 gated A and C on
        # EPOCH_INFLATED and left B out, which is the asymmetry this file's own header calls a
        # preference rather than a rule.
        print("  B IS PROVISIONAL for the same reason A and C are. The rungs were not fitted in")
        print("  the same regime, and an equal-STEPS design cannot separate 'more data helped'")
        print("  from 'the small rungs overfit' any better than it can separate a flat curve.")
        print("  Run the stage-2 equal-EPOCH control at rungs 40 and 120 (STEPS=2290 / 6748)")
        print("  before recording the slope. What is recordable NOW is the ordering, not a cause.")
else:
    provisional = EPOCH_INFLATED
    # V3. C fires on `not (MONOTONE and SPAN >= MATERIAL)`, which is TWO different worlds, and
    # V2 printed the same sentence for both. Concretely reachable: s40=-50, s120=-70, s362=-20,
    # sigma_hat=4 (MATERIAL=12). D does not fire, MONOTONE is False -> C, with SPAN=+30 pp. Then
    # "did not move the headline" and "did not clear the seed spread" are both false. Three noisy
    # points landing out of order while spanning materially is not a corner case: MONOTONE's own
    # annotation puts P = 1/6 on the specified ordering.
    NOISY = (not MONOTONE) and SPAN >= MATERIAL
    if NOISY:
        print("VERDICT C-NOISY (PROVISIONAL) · NO READABLE CURVE. The endpoints span")
        print(f"  {SPAN:+.2f} pp, which clears MATERIAL ({MATERIAL:.2f} pp), but the three rungs")
        print(f"  are OUT OF ORDER (s40 {s40:+.2f} / s120 {s120:+.2f} / s362 {s362:+.2f}), so no")
        print("  monotone trend was measured and N* was fitted through points that do not lie on")
        print("  a curve. This is NOT 'not data-limited' — the headline DID move by more than a")
        print("  seed does. It is 'this design did not resolve the shape'. Neither I-9 nor a")
        print("  recording campaign is licensed. The next run is a SEED REPLICATE at the rung")
        print("  that breaks the order, to find out whether the ordering or the noise is real.")
    else:
        print("VERDICT C" + (" (PROVISIONAL)" if provisional else "")
              + " · NOT DATA-LIMITED. More of this data did not move the")
        print("  headline. The live hypotheses become the KIND of data (I-9) and the architecture")
        print("  (I-2); months of recording the same task would have been the wrong move.")
    if provisional:
        # The symmetric half of A's provisionality (V2). A flat curve is exactly what an
        # inflated-epoch small rung produces when the extra passes finish a convergence the
        # big rung never got, so C cannot be recorded as REFUTED off this design alone. The
        # control is the same one A needs — that is the point of gating both on it.
        why = ("final action_reg 40 < 362 despite ~8.7x fewer distinct chunks"
               if not (reg40 is None or reg362 is None)
               else "action_reg UNMEASURED, and a missing diagnostic is not a quiet one")
        print("  C IS PROVISIONAL for the SAME reason A is. The two ends of the curve were not")
        print(f"  fitted in the same regime ({why}),")
        print("  and an equal-STEPS design cannot separate that from data volume in EITHER")
        print("  direction. Run the stage-2 equal-EPOCH control at rungs 40 and 120")
        print("  (STEPS=2290 / 6748, RUN_ID=i8-rung0NN-e17-seed0) and re-score. Until it reports,")
        if NOISY:
            print("  record this as 'the rungs did not fall in order', NOT as 'the curve is flat'")
            print("  and NOT as \"'not enough data' is REFUTED\" — the span above is material.")
        else:
            print("  record this as 'the curve did not move', NOT as \"'not enough data' is")
            print("  REFUTED\" — the first is a claim about the curve, the second about the world.")
    else:
        print("  The stage-2 equal-EPOCH control is NOT required: rung 40's final action_reg did")
        print("  not undercut rung 362's, so the extra 130 passes did not put the small rung in a")
        print("  different fitting regime, and the flat curve is attributable to N. Record it in")
        print("  docs/benchmark.md as a measured result, not an addendum.")
    if sigma_hat is None:
        # V3: V2 stated this backwards. A SMALLER MATERIAL makes `SPAN >= MATERIAL` EASIER, so
        # underestimating it pushes toward A/B, not toward C. The conclusion V2 drew is still
        # right — landing in C anyway is robust to the threshold — but the reason was inverted,
        # and this text is recorded verbatim next to a pre-registered verdict.
        print("  (sigma_hat unmeasured; MATERIAL was the bare floor. A floor that UNDERSTATES the")
        print("   true seed spread makes `SPAN >= MATERIAL` easier, i.e. it pushes toward A/B —")
        print("   so landing in C despite it is the ROBUST direction, and the FLOOR is not what")
        print("   makes this provisional. The epoch confound is. What the missing replicate does")
        print("   cost: C cannot be strengthened, because a true spread ABOVE the floor is the")
        print("   one thing that would make this curve flat by construction rather than by")
        print("   measurement.)")
    elif SPAN >= MATERIAL_FLOOR_PP and not NOISY:
        print(f"  (Nuance to record verbatim: SPAN {SPAN:+.2f} pp cleared the {MATERIAL_FLOOR_PP:.0f}")
        print(f"   pp floor but not {MATERIAL:.2f} pp of measured seed spread. The honest sentence")
        print("   is 'the curve moved less than one seed does', not 'the curve is flat'. Either")
        print("   way it is not evidence for spending months on more of the same data.)")

print()
print("STANDING WEAKNESS, unchanged: three points on ONE task cannot distinguish 'needs more")
print("  episodes' from 'needs more TASKS'. I-8 answers only the first.")
PY

echo
echo "Copy back to the Mac (run this ON THE MAC), then re-score on CPU forever:"
# This job writes three run dirs, so pull them all — a bare ${RUN_ID} here would be whatever
# the scoring loop happened to exit with (i8-rung040-seed1), which is not what you want.
echo "  ./cluster/discoverer/sync.sh --pull"
echo "  python scripts/run_bench.py runs/i8-rung040-seed0/eval-i8-history \\"
echo "                             runs/i8-rung120-seed0/eval-i8-history --compare"
