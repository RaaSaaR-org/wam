#!/bin/bash
# Step 16 — T-39 / PR-07: run everything in 70_train_t39_baseline.sbatch that does NOT need a GPU,
# on the free QoS, so that a defect in it costs zero GPU-hours.
#
#   sbatch 73_dryrun_t39_subset.sbatch
#
# WHAT THIS ACTUALLY DE-RISKS. train_t39_baseline.py --dry-run does three things before it stops:
# it resolves the committed split against the converted corpus, it materialises the 362-episode
# LeRobot subset, and it generates that subset's meta/stats.json + meta/relative_stats.json through
# upstream's gr00t/data/stats.py. The third is the reason this file exists. It is the newest code
# in the path, it is the first thing that imports our modality config into upstream's registry, and
# its failure modes are all quiet:
#
#   - a modality key that meta/modality.json does not define    -> KeyError, minutes in
#   - action_configs=None                                       -> generate_rel_stats returns EARLY,
#                                                                  exit 0, no relative_stats.json
#   - a delta_indices horizon the parquet cannot serve          -> IndexError during normalisation
#
# Inside 70_* every one of those surfaces AFTER an H200 has been allocated. Here they surface on
# two CPUs that bill nothing. PR-07 §7 fixes a 12 GPU-h ceiling and this job spends none of it.
#
# THIS IS NOT THE RUN. It writes ${OUT}/lerobot_subset and ${OUT}/run_metadata.json and stops
# before the trainer. 70_* rebuilds the subset from scratch (build_lerobot_subset rmtree's its own
# output), so nothing produced here is silently reused as if it had been trained on.
#
#SBATCH --job-name=wam-t39-dryrun
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t39-dryrun.%j.out

set -euo pipefail
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
source "${PROJ}/caches.sh"

module purge
module load anaconda3/python3.12

# The vendored trainer's venv, not the WAM one — the same interpreter 70_* uses, because the point
# is to exercise the path 70_* will take rather than a lookalike.
export VIRTUAL_ENV=${PROJ}/virt_envs/t39
export PATH=${VIRTUAL_ENV}/bin:${PATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SEED=${SEED:-0}
RUN_ID=${RUN_ID:-t39-baseline-seed${SEED}}
OUT=${PROJ}/runs/${RUN_ID}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
MODALITY_CONFIG=${MODALITY_CONFIG:-${PROJ}/wam/configs/groot/new_embodiment_config_defaults.py}
VENDOR=${PROJ}/wam/third_party/isaac-gr00t
DRIVER=${PROJ}/wam/scripts/train_t39_baseline.py

if [[ ! -d "${VIRTUAL_ENV}" ]]; then
  echo "FATAL: ${VIRTUAL_ENV} missing — run 72_build_t39_env.sbatch first."
  exit 1
fi

mkdir -p "${OUT}"
echo "=== T-39 dry run (no GPU, no trainer) into ${OUT}"
echo "=== embodiment ${EMBODIMENT_TAG} via ${MODALITY_CONFIG}"

# --model-id and --model-dir are required by the driver and are recorded in the witness, so they
# are stated here exactly as 70_* states them even though --dry-run never opens the weights. A
# placeholder would put a false checkpoint id into a run_metadata.json that 70_* then overwrites —
# briefly true, and exactly the kind of artifact PR-07 §8 exists to prevent.
python "${DRIVER}" \
    --vendor-root "${VENDOR}" \
    --trainer-entrypoint gr00t/experiment/launch_finetune.py \
    --model-id "${MODEL_ID:-nvidia/GR00T-N1.7-3B}" \
    --model-dir "${MODEL_DIR:-${PROJ}/apple_pnp_h200/GR00T-N1.7-3B}" \
    --dataset "${PROJ}/data/groot-apple-raw" \
    --wam-dataset "${PROJ}/data/gr00t-apple-full" \
    --train-episodes "${PROJ}/wam/configs/splits/i8_train_362.txt" \
    --exclude-episodes "${PROJ}/wam/configs/splits/t18_holdout_episodes.txt" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --modality-config "${MODALITY_CONFIG}" \
    --seed "${SEED}" \
    --out-dir "${OUT}" \
    --device cpu \
    --dry-run

echo
echo "=== what landed"
ls -la "${OUT}/lerobot_subset/meta"
echo "=== relative_stats.json shape check"
python - "${OUT}/lerobot_subset/meta/relative_stats.json" <<'EOF'
import json, sys
stats = json.load(open(sys.argv[1]))
stats.pop("__fingerprints__", None)
for key, entry in sorted(stats.items()):
    mean = entry.get("mean")
    shape = []
    node = mean
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None
    print(f"  {key:22s} mean shape {shape}")
EOF

# -- the processor probe -------------------------------------------------------------------------
# The other thing 70_* does that needs no GPU, and the thing that killed job 187802 at 158 s:
# Gr00tN1d7DataCollator -> build_processor(model_name) -> Qwen3VLProcessor.from_pretrained(
# "nvidia/Cosmos-Reason2-2B"), inside which transformers' _patch_mistral_regex calls
# huggingface_hub.model_info() on the NAME. Offline, that raises; online, upstream measured it
# hanging for 600 s. GROOT_PATCH_MISTRAL=1 installs upstream's own wrapper, which short-circuits
# for any name that is a directory or lacks "mistral" — i.e. this one — without touching weights.
#
# Probing it here means the next GPU allocation is not spent rediscovering it. This loads the
# 4.5 GB backbone processor config and tokenizer only; no model weights, no CUDA.
export GROOT_PATCH_MISTRAL=1
echo
echo "=== processor probe (offline, CPU): the call that failed job 187802"
python - <<'EOF'
import os
import gr00t  # noqa: F401  — import for its side-effect patches, before transformers is touched
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_processor

assert os.environ.get("HF_HUB_OFFLINE") == "1", "the probe is only meaningful offline"
processor = build_processor("nvidia/Cosmos-Reason2-2B", {})
print(f"  OK — {type(processor).__name__} loaded offline from the staged cache")
EOF

echo
echo "=== DONE. Nothing was trained. Next: sbatch 70_train_t39_baseline.sbatch"
