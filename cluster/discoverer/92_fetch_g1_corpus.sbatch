#!/bin/bash
# Step 16 — T-041 / PR-09: fetch the G1 corpus at SOURCE resolution and lay it out for captioning.
#
#   T041_FREEZE_LIFTED="<reason>" sbatch 92_fetch_g1_corpus.sbatch
#
# CPU-only QoS: no GPU hours. Downloads and symlinks only.
#
# SOURCE RESOLUTION, NOT OURS. datasets/gr00t-apple-full is 120x160 — convert_lerobot_g1.py threw
# the pixels away because a policy does not need them. A generator trains ON pixels, so the
# converted copy is unusable here and the 640x480 originals have to come down again.
#
# THE CORPUS IS PR-09 §2's CHOICE AND IT IS NOT INTERCHANGEABLE. This is the embodiment-fidelity
# corpus: AppleToPlate (CC-BY-4.0) plus the 13 unitreerobotics/G1_Dex3_* sets (Apache-2.0), all of
# them the exact G1 + Dex3 embodiment the priors render as a generic manipulator.
#
# Humanoid Everyday was added on top by operator decision (OD-09, 2026-08-07) — see the note on
# its entry below and the dated amendment in PR-09 §2. It is UNLICENSED, and it was originally
# excluded here as the VARIETY corpus for a goal PR-09 §2 dropped. Nothing else may be swapped in
# on that precedent: a different corpus changes which experiment this is and needs its own
# pre-registration. The amendment records what the addition costs, which is not nothing.
#
# CAMERA KEYS ARE NOT GUESSED. The sets do not agree on names, and a generator trained on a wrist
# camera when we meant the head view is a finite, plausible, wrong run. Step 1 prints every video
# feature key per repo; if a repo has more than one, prepare_cosmos_corpus.py REFUSES until the
# operator names it in CAMERA_KEYS. Read the listing, then set it.
#
#SBATCH --job-name=wam-t041-corpus
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/t041-corpus.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh
source "${PROJ}/cosmos_env.sh"

: "${T041_FREEZE_LIFTED:?PR-07 §7 freezes Cosmos3-Super. Set T041_FREEZE_LIFTED to the reason.}"
echo "=== freeze lifted: ${T041_FREEZE_LIFTED}"

module purge
module load anaconda3/python3.12
export PATH=${UV_INSTALL_DIR}:${PATH}

RAW=${PROJ}/data/raw
CORPUS=${CORPUS:-${PROJ}/data/cosmos-g1-embodiment}
VAL_EPISODES=${VAL_EPISODES:-30}
SEED=${SEED:-0}
PREPARE=${PROJ}/wam/scripts/prepare_cosmos_corpus.py
[[ -f "${PREPARE}" ]] || { echo "FATAL: ${PREPARE} missing — run sync.sh from the Mac."; exit 1; }
mkdir -p "${RAW}"

# PR-09 §2. Apache-2.0 on all 13 Unitree sets and CC-BY-4.0 on AppleToPlate, checked 2026-08-07.
# The 15th entry is the exception and carries OD-09's asterisk on its own — read its note.
REPOS=(
  nvidia/GR00T-N1.7-AppleToPlate
  unitreerobotics/G1_Dex3_BlockStacking_Dataset
  unitreerobotics/G1_Dex3_CameraPackaging_Dataset
  unitreerobotics/G1_Dex3_GraspSquare_Dataset
  unitreerobotics/G1_Dex3_ObjectPlacement_Dataset
  unitreerobotics/G1_Dex3_Pouring_Dataset
  unitreerobotics/G1_Dex3_ToastedBread_Dataset
  unitreerobotics/G1_Dex3_PickApple_Dataset
  unitreerobotics/G1_Dex3_PickBottle_Dataset
  unitreerobotics/G1_Dex3_PickCharger_Dataset
  unitreerobotics/G1_Dex3_PickGum_Dataset
  unitreerobotics/G1_Dex3_PickSnack_Dataset
  unitreerobotics/G1_Dex3_PickTissue_Dataset
  unitreerobotics/G1_Dex3_PickDoll_Dataset
  # Added 2026-08-07 by operator decision, before any job of this chain had started (all four
  # were still PENDING) — a pre-commit edit, not an amendment to a rule that had already run.
  # UNLICENSED. No LICENSE file, no license: field, no card metadata; the GitHub API 404s and
  # the paper's CC BY 4.0 covers the manuscript, not the data. docs/outbound/ holds an unsent
  # request asking USC PSI Lab to fix that. The operator's call is that this is an internal
  # test, so no weight from it is published or redistributed. See PR-09 §2.
  # Shape: LeRobot v2.1, 4064 episodes, chunk size 1000 — it roughly triples the corpus alone.
  # Its single camera key is `egocentric`. That is unambiguous, so step 2 would not refuse over
  # it, but CAMERA_KEYS is positional: once ANY repo needs an entry the array must be full
  # length, so `egocentric` is still the 15th value the operator passes.
  #
  # The viewpoint is the cost. `egocentric` is head-mounted and mostly shows hands and the
  # object, where the 13 Unitree sets' `cam_left_high` shows the arms and torso. For "are these
  # Dex3 hands" that is on target; for "is this a G1" it contributes little, and it now dominates
  # the pool by episode count. PR-09 §2's amendment records this as accepted, not overlooked.
  USC-PSI-Lab/Humanoid-Everyday-G1
)
# Operator-supplied, space-separated, one per repo above and in that order. Leave empty on the
# first run: step 1 prints the keys and step 2 refuses with the list.
CAMERA_KEYS=${CAMERA_KEYS:-}

# -- 1. fetch: meta + videos only. The parquet is state and action, which a generator never sees.
for repo in "${REPOS[@]}"; do
  dest=${RAW}/$(basename "${repo}")
  if [[ ! -f "${dest}/meta/info.json" ]]; then
    echo "=== downloading ${repo}"
    # ONE --include per pattern. The hf CLI is click-based, so `--include a b` binds only `a`
    # to the flag and leaves `b` as a positional *filename*, which silently switches the whole
    # command into explicit-file mode ("Ignoring --include since filenames have been explicitly
    # set") and then 404s on a literal file named `videos/**`. That is what killed job 186283.
    uvx hf@latest download --repo-type dataset "${repo}" \
        --include 'meta/*' --include 'videos/**' --local-dir "${dest}"
  else
    echo "=== ${dest} present"
  fi
done

echo "=== video feature keys per repo (name one per repo in CAMERA_KEYS if any has >1)"
for repo in "${REPOS[@]}"; do
  dest=${RAW}/$(basename "${repo}")
  python - "${dest}" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
info = json.loads((d / "meta" / "info.json").read_text())
keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
print(f"  {d.name}: fps={info.get('fps')} cameras={keys}")
PY
done

# -- 2. lay it out. The script applies cosmos-framework's own silent filters (>61 s, <61 frames)
#       so the manifest count is the count that trains, and writes the seeded train/val split
#       PR-09 §5 requires the eval prompts to be drawn from.
ARGS=()
i=0
read -r -a KEYS <<< "${CAMERA_KEYS}"
for repo in "${REPOS[@]}"; do
  ARGS+=(--source "${RAW}/$(basename "${repo}")")
  if [[ ${#KEYS[@]} -gt 0 ]]; then
    [[ -n "${KEYS[$i]:-}" ]] || { echo "FATAL: CAMERA_KEYS has ${#KEYS[@]} entries for ${#REPOS[@]} repos."; exit 1; }
    ARGS+=(--camera-key "${KEYS[$i]}")
  fi
  i=$((i + 1))
done

python "${PREPARE}" "${ARGS[@]}" \
    --out "${CORPUS}" --val-episodes "${VAL_EPISODES}" --seed "${SEED}"

cat >> "${PROJ}/cosmos_env.sh" <<EOF
export CORPUS=${CORPUS}
EOF
echo "CORPUS OK -> ${CORPUS}"
echo "next: sbatch 93_caption_corpus.sbatch"
