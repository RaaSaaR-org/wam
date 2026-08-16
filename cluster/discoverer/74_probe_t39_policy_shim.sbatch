#!/bin/bash
# Step 17 — T-39 / PR-07: prove the POLICY_ENTRYPOINT shim works against the REAL checkpoint,
# in twenty GPU-minutes, before 71_eval_t39_control.sbatch spends a full allocation finding out.
#
#   sbatch 74_probe_t39_policy_shim.sbatch
#
# WHY THIS EXISTS AT ALL. The same argument as 73_dryrun_t39_subset.sbatch: this session has burned
# three allocations on defects a smaller job would have caught (187799's tyro enum-name, 187802's
# offline model_info call at 158 s, and 187813's repeat of that same call in the eval). The shim is
# newer than all of them and sits on a contract nobody has executed end to end:
#
#   - a permuted state key order        -> a scrambled joint vector, finite metrics, no error
#   - the wrong embodiment slot         -> random-init weights, a real-looking number, no error
#   - relative actions added twice      -> plausible magnitudes, wrong direction, no error
#   - an action horizon below the bar   -> caught by the eval, but only after the model loads
#
# The first three never raise anywhere. This job runs the actual vendored policy on the actual
# post-trained checkpoint over real holdout observations and asserts the properties that separate
# a correct shim from one that merely scores.
#
# THIS IS NOT THE EVAL. It writes no bench.json, computes no metric and reads no threshold. It
# answers "does the contract hold", not "is the policy any good".
#
# WHY THIS NEEDS A GPU, having been written for the free QoS and failed there (job 187812).
# The checkpoint's own config pins attn_implementation="flash_attention_2", and transformers
# refuses to construct the module at all without CUDA:
#
#   ValueError: FlashAttention2 has been toggled on, but it cannot be used ...
#              Flash Attention 2 is not available on CPU.
#
# That is raised inside Qwen3VLForConditionalGeneration.__init__, before a single line of the shim
# runs, so no amount of care in our code makes a CPU probe possible for THIS checkpoint. The free
# variant is therefore not "not yet working", it is unreachable, and pretending otherwise would
# leave a gate in the tree that always fails for a reason unrelated to what it claims to test.
#
# The cost is small and bounded: one GPU for the model load plus two forward passes, minutes, with
# a 20-minute wall. PR-07 §7's ceiling is 12 GPU-h and job 187804 spent 1.37 of it.
#
#SBATCH --job-name=wam-t39-shimprobe
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t39-shimprobe.%j.out

set -euo pipefail
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
source "${PROJ}/caches.sh"

module purge
module load anaconda3/python3.12
module load nvidia/cuda/12/12.8

export VIRTUAL_ENV=${PROJ}/virt_envs/t39
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
export NO_ALBUMENTATIONS_UPDATE=1

RUN_ID=${RUN_ID:-t39-baseline-seed0}
OUT=${PROJ}/runs/${RUN_ID}
CKPT=${CKPT:-${OUT}/checkpoints/checkpoint-10000}
VENDOR=${PROJ}/wam/third_party/isaac-gr00t
RAW=${PROJ}/data/groot-apple-raw
HOLDOUT=${PROJ}/wam/configs/splits/t18_holdout_episodes.txt

# PYTHONPATH is how the shim becomes importable as a bare module name. `module purge` runs above
# this line on purpose, so whatever the modules do to PYTHONPATH cannot undo it — the ordering is
# the point, not an accident of where the export landed.
export PYTHONPATH=${PROJ}/wam/scripts:${VENDOR}${PYTHONPATH:+:${PYTHONPATH}}

if [[ ! -d "${CKPT}" ]]; then
  echo "FATAL: ${CKPT} missing — 70_train_t39_baseline.sbatch has not produced it."
  exit 1
fi

nvidia-smi -L

echo "=== probing t39_policy_shim against ${CKPT}"

# -u and faulthandler are not decoration. Job 187813 exited 1 with NO traceback and no message,
# which cost a round trip to diagnose; unbuffered streams plus a fault handler mean the next
# failure in this path describes itself the first time.
python -u -X faulthandler - "${CKPT}" "${RAW}" "${HOLDOUT}" "${PROJ}/wam" <<'EOF'
import sys
from pathlib import Path

import numpy as np

ckpt, raw_root, holdout_file, wam_root = (Path(p) for p in sys.argv[1:5])
sys.path.insert(0, str(wam_root / "src"))
sys.path.insert(0, str(wam_root / "scripts"))

import t39_policy_shim  # noqa: E402

print("--- 1. build the policy on GPU from the post-trained checkpoint")
infer = t39_policy_shim.build_policy(ckpt, "cuda")

print("--- 2. take real observations from a holdout episode the policy never saw")
import pyarrow.parquet as pq  # noqa: E402

episode_id = holdout_file.read_text().split()[0]
index = int(episode_id.rsplit("-", 1)[-1])
parquet = raw_root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
table = pq.read_table(parquet, columns=["observation.state"])
states = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
print(f"    {episode_id} -> {parquet.name}, {states.shape[0]} frames of {states.shape[1]} dims")

# A grey frame rather than the episode's video: decoding mp4 here would test PyAV, which
# 73_dryrun already covers, and the properties below are about the shim's plumbing, not the
# pixels. The policy still runs its full forward pass.
video = np.full((1, 480, 640, 3), 128, dtype=np.uint8)
instruction = "pick up the apple and put it on the plate"

failures = []


def check(label, condition, detail=""):
    print(f"    {'OK  ' if condition else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


print("--- 3. the contract's shape and dtype")
first = infer({"video": video, "state": states[0], "instruction": instruction})
check("returns float32", first.dtype == np.float32, str(first.dtype))
check("returns [horizon, 43]", first.ndim == 2 and first.shape[1] == 43, str(first.shape))
check("horizon >= 16 (the scored chunk length)", first.shape[0] >= 16, str(first.shape[0]))

print("--- 4. legs NaN, commanded columns finite")
check("legs [0:12] are NaN", bool(np.isnan(first[:, 0:12]).all()))
check("waist+arms+hands [12:43] all finite", bool(np.isfinite(first[:, 12:43]).all()))

print("--- 5. the output is a PREDICTION, not the input echoed back")
# A shim that silently returned the anchor state would satisfy every shape check above and score
# a suspiciously good MSE. The commanded columns must move away from the state they started at.
delta = np.abs(first[:, 12:43] - states[0][12:43])
check("commanded columns differ from the anchor state", float(delta.max()) > 1e-4,
      f"max |cmd - state| = {float(delta.max()):.6f}")

print("--- 6. the output DEPENDS on the state it was given")
# The single cheapest test for a plumbing break: feed a different, real observation and require a
# different answer. An ignored state argument produces identical chunks for every step of every
# episode, which no metric in the eval would flag as anything but a very consistent policy.
later = infer({"video": video, "state": states[len(states) // 2], "instruction": instruction})
spread = np.abs(first[:, 12:43] - later[:, 12:43])
check("a different anchor gives a different chunk", float(spread.max()) > 1e-4,
      f"max difference = {float(spread.max()):.6f}")

print("--- 7. arms move more than the waist over the chunk, as a pick-and-place should")
# Not a threshold and not a gate: a directional sanity read, printed so a scrambled joint vector
# has one more chance to look wrong to a human before the GPU arm runs.
arm_span = float(np.abs(np.diff(first[:, 15:29], axis=0)).sum())
waist_span = float(np.abs(np.diff(first[:, 12:15], axis=0)).sum())
print(f"    arm travel {arm_span:.4f} vs waist travel {waist_span:.4f}  (informational only)")

if failures:
    raise SystemExit(f"\nSHIM PROBE FAILED: {', '.join(failures)}")
print("\n    all contract properties hold.")
EOF

echo
echo "=== DONE. Nothing was scored. Next: sbatch 71_eval_t39_control.sbatch with"
echo "===   POLICY_ENTRYPOINT=t39_policy_shim:build_policy"
echo "===   MODEL_DIR=${CKPT}"
echo "===   PYTHONPATH must include ${PROJ}/wam/scripts"
