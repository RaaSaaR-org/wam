#!/bin/bash
# Step 13 — T-39 / PR-07: score the positive control and print the pre-registered verdict.
#
# THIS FILE IS THE PRE-REGISTRATION, together with docs/preregistration/PR-07-positive-control.md.
# Every threshold below is a literal and this file is in git before 70_train_t39_baseline.sbatch is
# submitted. Changing a threshold after seeing the number it scored is precisely the move
# pre-registration exists to prevent — this repo already records that lesson twice (I-9's one-sided
# L4 gate, and I8_RULE_V1->V3). If a constant genuinely has to change, version it: bump
# RULE_VERSION, record what moved and why, say which runs scored under which. Never edit in place.
#
# The rule is T39_RULE_V1 (2026-08-06). It has not been revised, and no arm has reported.
#
# SYMMETRY IS PART OF THE PRE-REGISTRATION, and here the NEGATIVE is the expensive conclusion:
# verdict N licenses "stop trying methods on this corpus", which retires an entire class of future
# work. It is therefore gated on a material margin AND on a second arm, exactly as the positive is
# gated on the ladder. 62_eval_i8_curve.sbatch's header states the principle; this file inherits
# both the principle and the MATERIAL_FLOOR_PP constant from it rather than coining a new one.
#
#   sbatch 71_eval_t39_control.sbatch
#
# One GPU, minutes per arm: 40 holdout episodes is 1 040 chunks and each is one forward pass.
# No --requeue, same reason as 60_eval_t16.sbatch: idempotent, writes no DONE, re-scoring is cheap.
#
#SBATCH --job-name=wam-eval-t39
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/eval-t39.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/t39
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# GROOT_PATCH_MISTRAL=1 — the same export 70_train_t39_baseline.sbatch carries, and it was missing
# here. NO THRESHOLD MOVED: this is an environment variable, not a constant of T39_RULE_V1, and the
# rule block below is untouched. Recorded rather than quietly patched, per this file's own header.
#
# Why it is needed: the policy arm loads the processor through AutoProcessor.from_pretrained, and
# transformers' _patch_mistral_regex calls huggingface_hub.model_info() on the model NAME during
# tokenizer construction. Under HF_HUB_OFFLINE=1 that raises. Job 187802 died on exactly this at
# 158 s, and job 187813 died on it again here — after both oracle arms had already reported, which
# is why the two G0 numbers in that run are sound and only the policy arm is missing.
#
# It is upstream's own env-gated wrapper and is weight-neutral: its _safe path returns the tokenizer
# untouched for any name that is a directory or lacks "mistral". It is NOT one of the two forbidden
# escape hatches (GROOT_HF_LOCAL_FIRST / GROOT_SKIP_HF_MODEL_WEIGHTS), which zero the model.
export GROOT_PATCH_MISTRAL=1
export NO_ALBUMENTATIONS_UPDATE=1
export WAM_GIT_COMMIT=$(cat "${PROJ}/wam/GIT_COMMIT" 2>/dev/null || true)

nvidia-smi -L

EVAL=${PROJ}/wam/scripts/eval_t39_baseline.py
RUNS=${PROJ}/runs
RUN_ID=${RUN_ID:-t39-baseline-seed0}
OUT=${RUNS}/${RUN_ID}
DATASET=${PROJ}/data/gr00t-apple-full
RAW=${PROJ}/data/groot-apple-raw
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt
TRAIN=${PROJ}/wam/configs/splits/i8_train_362.txt
# The in-distribution diagnostic reuses a COMMITTED, already-reviewed artifact rather than minting
# a fresh 40-episode sample here: i8_train_040.txt is a seeded random subset of the 362, nested
# inside them and disjoint from the holdout by construction (tests/test_splits.py). A sample drawn
# on the cluster after the fact is not reviewable and could be chosen to flatter either verdict.
TRAIN_PROBE=${PROJ}/wam/configs/splits/i8_train_040.txt

if [[ ! -f "${EVAL}" ]]; then
  echo "FATAL: ${EVAL} missing — run cluster/discoverer/sync.sh from the Mac first."
  exit 1
fi
if [[ ! -f "${OUT}/DONE" ]]; then
  echo "FATAL: ${OUT}/DONE missing — 70_train_t39_baseline.sbatch has not finished."
  echo "       Scoring a partial post-training against a finished WAM run measures wall clock."
  exit 1
fi
if [[ ! -f "${OUT}/run_metadata.json" ]]; then
  echo "FATAL: ${OUT}/run_metadata.json missing — no split witness, so the holdout cannot be"
  echo "       PROVEN unseen. The recorded ids and the recorded hash are two fields of one"
  echo "       self-description; without the external file this checks the run against itself."
  exit 1
fi

common=(
  --run-dir "${OUT}"
  --dataset "${DATASET}"
  --raw-dataset "${RAW}"
  --holdout "${HOLDOUT}"
  --train-episodes "${TRAIN}"
  --camera ego
  --device cuda
)

# -- G0 first. The oracle arms can VOID the whole experiment, and they cost no policy forward
# passes at all, so they run before the thing they are able to invalidate. PR-07 §4.
#
# POLICY_ENTRYPOINT/MODEL_DIR ARE RESOLVED BELOW, NOT HERE. They used to be `:?`-required at this
# point, which quietly contradicted the design: with either one unset the job died before G0a, so
# the veto that is supposed to cost nothing could not be obtained at all without first having a
# post-trained checkpoint and a written inference shim. `scripts/eval_t39_baseline.py` already
# gets this right — it only demands them for `--arm policy` (see its argument check). Resolving
# them after G0b means an operator with neither can still run the two oracle arms, read the VOID
# gates, and stop; the job then exits non-zero on the missing variable with both reports on disk.
echo "=== G0a oracle_state — the holdout's future EXECUTED STATES through the adapter."
echo "===     This is the identity of our own label pipeline. Anything but ~perfect is OUR bug."
python "${EVAL}" "${common[@]}" --arm oracle_state --out "${OUT}/eval-t39-oracle-state"

echo "=== G0b oracle_action — the holdout's native ACTION COLUMN through the adapter."
echo "===     This is the CEILING for any policy trained on that column. If it fails L1, no such"
echo "===     policy can clear our bar and the finding is about our labels, not about GR00T."
python "${EVAL}" "${common[@]}" --arm oracle_action --out "${OUT}/eval-t39-oracle-action"

# T39_ORACLE_ONLY=1 stops here, cleanly and on purpose: the two gates above are the cheap veto,
# and reading them before spending a GPU hour is the whole shape of PR-07 §4. The verdict block at
# the end of this file needs all four arms, so it is not reached in this mode.
if [[ "${T39_ORACLE_ONLY:-0}" == "1" ]]; then
  echo "=== T39_ORACLE_ONLY=1 — oracle arms complete. Reports:"
  echo "===   ${OUT}/eval-t39-oracle-state"
  echo "===   ${OUT}/eval-t39-oracle-action"
  echo "=== No policy arm, no verdict. Re-run without T39_ORACLE_ONLY once a policy exists."
  exit 0
fi

# No defaults, the same rule as MODEL_ID (PR-07 §8) — the vendored inference API was never
# verified from a primary source, so the operator states it and it is recorded in every arm's
# timing.json.
#
# MODEL_DIR IS WHAT 70 PRODUCED, NOT WHAT IT CONSUMED. This prompt used to read "the staged weights
# used by 70_train_t39_baseline.sbatch", which names the wrong directory: 70's own MODEL_DIR is the
# BASE checkpoint it fine-tunes FROM (nvidia/GR00T-N1.7-3B), and 70 writes what it PRODUCES to
# ${OUT}/checkpoints/checkpoint-<step>. An operator following the old wording would have scored the
# base model as if it were the headline — and because the embodiment tag is `new_embodiment`, the
# base carries an UNTRAINED action head for that slot, so it loads without complaint and returns
# numbers rather than an error. A plausible wrong number is the single failure this whole file
# exists to prevent, and the old wording asked for one by name.
#
# NO THRESHOLD MOVED. This is an operator-facing path, not a constant of T39_RULE_V1; the rule block
# below is byte-identical and no arm has been re-scored. Recorded here rather than quietly patched,
# per this file's own header and the GROOT_PATCH_MISTRAL note above.
POLICY_ENTRYPOINT=${POLICY_ENTRYPOINT:?set POLICY_ENTRYPOINT to 'module.path:factory' — see scripts/eval_t39_baseline.py}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the POST-TRAINED checkpoint 70 wrote, e.g. ${OUT}/checkpoints/checkpoint-<step> — NOT the base weights 70 trained from}

# The wording alone is not the fix. A path is a path, and the base checkpoint remains one keystroke
# away, so the refusal is mechanical: every checkpoint HF Trainer writes carries trainer_state.json
# (it is what --resume_from_checkpoint reads), and no released base repo does. This distinguishes
# "trained here" from "downloaded here" without trusting the operator to have read the paragraph.
if [[ ! -f "${MODEL_DIR}/trainer_state.json" ]]; then
  echo "FATAL: ${MODEL_DIR} has no trainer_state.json, so it is not a checkpoint this run wrote."
  echo "       If that is the BASE checkpoint, it will load here WITHOUT ERROR and score an"
  echo "       untrained new_embodiment action head — a number, not a failure, and nothing"
  echo "       downstream could tell it apart from the headline."
  echo "       Expected one of:"
  ls -d "${OUT}"/checkpoints/checkpoint-* 2>/dev/null | sed 's/^/         /' || echo "         (none — did 70 finish?)"
  exit 1
fi
policy=(
  --policy-entrypoint "${POLICY_ENTRYPOINT}"
  --model-dir "${MODEL_DIR}"
  --vendor-root "${PROJ}/wam/third_party/isaac-gr00t"
)

echo "=== THE HEADLINE — the post-trained policy on the 40-episode holdout it never saw"
python "${EVAL}" "${common[@]}" "${policy[@]}" --arm policy --out "${OUT}/eval-t39-holdout"

echo "=== diagnostic — the same policy on 40 episodes it DID train on (upper bound, never a headline)"
python "${EVAL}" "${common[@]}" "${policy[@]}" --arm policy --score-episodes "${TRAIN_PROBE}" \
    --out "${OUT}/eval-t39-train40"

echo
echo "=== T39_RULE_V1 (docs/preregistration/PR-07-positive-control.md §5)"
python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path

# ---------------------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS — committed before 70_train_t39_baseline.sbatch was submitted.
#
# Two of the three are not new. The ladder gates (L1 `skill_vs_repeat_pct > 0`, L2
# `ci_skill_vs_repeat_pct > 0`, where ci_ is the task-CRITICAL chunk subset and NOT a confidence
# interval) are WAM-Bench's own, already applied to every number in docs/benchmark.md.
# MATERIAL_FLOOR_PP is I8_RULE_V3's, borrowed deliberately: it is the repo's established floor for
# "material on this headline", and inventing a second one for the same metric in the same week
# would let the choice of floor be the finding.
#
# ORACLE_STATE_FLOOR_PCT is the one new number. It gates a plumbing check whose true value is
# ~100 % by construction — the arm feeds the label pipeline its own output — so it is set well
# below what a correct adapter produces and is not a tuned threshold. A correct adapter does not
# score 91; a broken one does not score 89.
# ---------------------------------------------------------------------------------------
RULE_VERSION = "T39_RULE_V1"
HEADLINE = "skill_vs_repeat_pct"          # L1: better than repeating the last action
CRITICAL = "ci_skill_vs_repeat_pct"       # L2: still better on the task-critical chunks
ORACLE_STATE_FLOOR_PCT = 90.0
MATERIAL_FLOOR_PP = 10.0                  # from I8_RULE_V3, 62_eval_i8_curve.sbatch
# ---------------------------------------------------------------------------------------

out = Path(sys.argv[1])
ARMS = {
    "oracle_state":  "eval-t39-oracle-state",
    "oracle_action": "eval-t39-oracle-action",
    "policy":        "eval-t39-holdout",
    "train40":       "eval-t39-train40",
}


def bench(name):
    path = out / ARMS[name] / "bench.json"
    return json.loads(path.read_text()) if path.is_file() else None


reports = {name: bench(name) for name in ARMS}
missing = [n for n, b in reports.items() if b is None]
if missing:
    raise SystemExit(f"REFUSING TO RULE — arms did not report: {', '.join(missing)}")

print(f"\n{'arm':>14} {'chunks':>6} {'skill_vs_rep':>13} {'ci_skill':>9} {'mse':>11} "
      f"{'horizon':>7} {'smooth':>7}  level")
for name in ARMS:
    b = reports[name]
    print(f"{name:>14} {b['num_predictions']:>6} {b[HEADLINE]:>+12.2f}% {b[CRITICAL]:>+8.2f}% "
          f"{b['mse']:>11.4g} {b['horizon_ratio']:>7.3f} {b['smoothness_ratio']:>7.3f}  "
          f"{b['level_name']}")

s_state  = reports["oracle_state"][HEADLINE]
s_action = reports["oracle_action"][HEADLINE]
s_policy = reports["policy"][HEADLINE]
s_train  = reports["train40"][HEADLINE]

L1_policy = s_policy > 0.0
L2_policy = reports["policy"][CRITICAL] > 0.0
L1_train  = s_train > 0.0
L1_action = s_action > 0.0

print(f"\n-- {RULE_VERSION}")
print(f"  oracle_state  {s_state:+.2f}%   (floor {ORACLE_STATE_FLOOR_PCT:+.2f}%)")
print(f"  oracle_action {s_action:+.2f}%   (must clear L1: {L1_action})")
print(f"  policy        {s_policy:+.2f}%   L1 {L1_policy}   L2 {L2_policy}")
print(f"  train40       {s_train:+.2f}%   L1 {L1_train}   [DIAGNOSTIC — never a headline]")
print(f"  MATERIAL_FLOOR_PP {MATERIAL_FLOOR_PP:.1f} pp   [borrowed from I8_RULE_V3]")
print()

# -- G0. The vetoes run first and are unconditional. --------------------------------------
if s_state < ORACLE_STATE_FLOOR_PCT:
    print("VERDICT VOID (adapter) · The label pipeline fed its own output back through the")
    print(f"  adapter and scored {s_state:+.2f}%, under the {ORACLE_STATE_FLOOR_PCT:.0f}% floor. That")
    print("  is arithmetically close to impossible for a correct mapping, so this is OUR bug —")
    print("  most likely the delta anchor, the joint ordering, or the gripper synergy. NOTHING is")
    print("  claimed about GR00T. Fix the adapter, kill the mutation that let it through, re-run.")
elif not L1_action:
    print("VERDICT VOID (labels) · and this is a FIRST-CLASS FINDING, not a setback.")
    print(f"  The dataset's own ACTION COLUMN — the ground truth GR00T is trained to predict —")
    print(f"  scores {s_action:+.2f}% under our scorer and does not clear L1. Our labels are")
    print("  relabeled from executed STATE; the action column is the COMMANDED target, and the two")
    print("  differ by more than the bar. Consequence: NO policy trained on this dataset's actions")
    print("  can clear our bar, T-39 cannot be run as designed, and every number in")
    print("  docs/benchmark.md is bounded by a label-space mismatch nobody had measured.")
    print("  Record this against the label pipeline before anything else is scored.")

# -- The verdicts. -------------------------------------------------------------------------
elif L1_policy:
    print("VERDICT P · THE BAR IS CLEARABLE. A published method post-trained on this corpus")
    print(f"  clears L1 at {s_policy:+.2f}%" + (f" and L2 at {reports['policy'][CRITICAL]:+.2f}%."
          if L2_policy else ", and does NOT clear L2 (the task-critical chunks)."))
    print("  The fourteen negatives in this repo are therefore statements about WAM, not about the")
    print("  corpus and not about the scorer. 'Not enough data' is refuted at N=362 by existence.")
    print("  Consequences, fixed in PR-07 §6 before this number existed:")
    print("    - T-32 is DESCOPED. ~109 GPU-h not spent; the data-volume question is answered by a")
    print("      method that works at the corpus size we already have.")
    print("    - Adopt the working policy as the NeoDEM capability. WAM's contribution narrows to")
    print("      evaluation, safety and integration AROUND it — which is what PR-07 §6 licenses.")
    print("  FORBIDDEN: reading this as 'WAM would clear it with more compute'. It had 36 GPU-h.")
    print("  FORBIDDEN: quoting this as a success rate. It is offline chunk MSE on 40 episodes.")
elif s_policy <= -MATERIAL_FLOOR_PP and not L1_train:
    print("VERDICT N · NOTHING CLEARS THIS BAR, AND THE POLICY CANNOT EVEN FIT THE DATA.")
    print(f"  {s_policy:+.2f}% on the holdout, past the {MATERIAL_FLOOR_PP:.0f} pp material margin, and")
    print(f"  {s_train:+.2f}% on 40 episodes it TRAINED ON. A method that cannot beat")
    print("  repeat-last-action on its own training data is not being beaten by generalisation.")
    print("  The corpus, the scorer, or both, are the finding — not the method, and not WAM.")
    print("  Consequences, fixed in PR-07 §6:")
    print("    - T-32 is answered for free: more of THIS data is not the story.")
    print("    - The next move is the KIND of data (PR-04's collection spec: failures, randomized")
    print("      placement, an unfrozen right hand), not another method.")
    print("  FORBIDDEN: reading N as refuting any specific WAM design. N says the instrument")
    print("  saturates on this corpus; it does not say which arm was wrong.")
elif not L1_policy and L1_train:
    print("VERDICT M · IT FITS AND DOES NOT GENERALISE. The policy clears L1 on episodes it trained")
    print(f"  on ({s_train:+.2f}%) and fails on the holdout ({s_policy:+.2f}%). Two things follow, and")
    print("  they point in opposite directions, which is why this is its own verdict:")
    print("    - The bar IS reachable under this scorer. N is refuted; the metric is not the wall.")
    print("    - The 362/40 split is now the live question — 402 episodes of one task may not")
    print("      contain 40 episodes' worth of held-out variation. PR-04 measured the corpus as")
    print("      action-side degenerate (right hand frozen, 0 of 402 failures), which is exactly")
    print("      what would produce this.")
    print("  FORBIDDEN: reading M as either P or N. It is neither.")
else:
    print("VERDICT I · INDETERMINATE. Nothing is recorded.")
    print(f"  The holdout headline is {s_policy:+.2f}%: it fails L1 but sits inside the")
    print(f"  {MATERIAL_FLOOR_PP:.0f} pp material margin, or the train40 diagnostic disagrees with it")
    print("  in a way the three verdicts above do not cover. A near-miss on a single seed is not a")
    print("  result in either direction, and this is the branch that exists so that it cannot be")
    print("  written up as one.")
    print("  The next run is SEED=1 sbatch 70_train_t39_baseline.sbatch, then re-score. Nothing")
    print("  else — in particular T-32 stays frozen, because its premise is still unestablished.")

print()
print("STANDING BOUNDS, unchanged by any verdict above (PR-07 §9): offline chunk MSE, one task,")
print("  402 success-only episodes, one holdout, one seed. The gripper channel has peak-to-peak")
print("  0.120 and never opens, so NOTHING measured here can see a grasp — docs/benchmark.md's")
print("  standing rule applies to every number this job printed.")
PY

echo
echo "Copy back to the Mac (run this ON THE MAC), then re-score on CPU forever:"
echo "  ./cluster/discoverer/sync.sh --pull"
echo "  python scripts/run_bench.py runs/${RUN_ID}/eval-t39-holdout \\"
echo "                              runs/t16-lora-seed0/eval-t29-history --compare"
