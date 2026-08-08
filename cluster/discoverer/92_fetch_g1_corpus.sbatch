#!/bin/bash
# Step 16 — T-041 / PR-09: fetch the G1 corpus at SOURCE resolution and lay it out for captioning.
#
# SUPERSEDED 2026-08-08 BY workstation/10_fetch_corpus.sh AND workstation/20_prepare_corpus.sh.
# Kept because the download it already performed is a usable cache, not because it should be run.
# Two reasons it cannot finish the job:
#
#   1. It symlinks. 13 of the 14 sources are LeRobot v3.0, where a clip is a timestamp window
#      inside a concatenated mp4 — there is no per-episode file to link to. Extraction needs
#      ffmpeg, and this QoS has 2 CPUs.
#   2. Every source is AV1. The captioner's OpenCV decodes none of it (job 186357: 372 clips,
#      0 captions, exit 0), so the corpus has to be transcoded to H.264 before anything reads it.
#
# Both are ffmpeg-and-IO work with a fast edit loop, which is what a workstation is for and what a
# GPU allocation is not. The corpus definition now lives in configs/cosmos3/corpus_g1_embodiment.tsv
# and is read by both paths, so they cannot disagree about what the corpus is.
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
# Humanoid Everyday was added on top by operator decision (OD-09, 2026-08-07) and REMOVED again on
# 2026-08-08 — see PR-09 §2, which carries both dated notes. The removal is operational, not a
# reversal of OD-09: the HF Hub rate-limited the repo (429) on three consecutive attempts and the
# last made no progress at all, so it could not be fetched without authenticating. The corpus is
# therefore the one PR-09 §2 pre-registered in the first place, which is the happier outcome.
#
# Nothing may be swapped in on the precedent of that addition: a different corpus changes which
# experiment this is and needs its own pre-registration.
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
# Where the CORPUS export lands. Indirected so a rehearsal cannot overwrite the variable the real
# chain reads: 93 and 94 source cosmos_env.sh, and a smoke path written there would be picked up
# silently by a job that believes it is training on the pre-registered corpus.
ENV_OUT=${ENV_OUT:-${PROJ}/cosmos_env.sh}
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
  # USC-PSI-Lab/Humanoid-Everyday-G1 was the 15th entry on 2026-08-07 and is deliberately absent
  # again. Do not re-add it without reading PR-09 §2's second dated note: it is unlicensed, its
  # `egocentric` head-mounted view answers "are these Dex3 hands" but not "is this a G1", and the
  # Hub 429s it without a token. Measured before it was dropped: ~2.5 GB against this corpus's
  # 70 GB, so it added far less signal than its 4064-episode count suggested.
)
# Operator-supplied, space-separated, one per repo above and in that order. Leave empty on the
# first run: step 1 prints the keys and step 2 refuses with the list.
CAMERA_KEYS=${CAMERA_KEYS:-}

# -- rehearsal mode --------------------------------------------------------------------------
# SMOKE_SOURCES REPLACES the corpus with a named subset, to prove the pipeline end to end before
# committing to the full one. It is NOT the pre-registered corpus and must never be mistaken for
# it, so three things are forced rather than defaulted:
#
#   1. a distinct CORPUS directory, so the registered tree cannot be overwritten;
#   2. a distinct ENV_OUT, so `export CORPUS=` for a rehearsal never reaches cosmos_env.sh, which
#      is what 93 and 94 read;
#   3. a SMOKE marker inside the corpus, so anything downstream can see what it is holding.
#
# It replaces rather than filters on purpose: CAMERA_KEYS is positional against REPOS, and quietly
# dropping entries from one list and not the other is precisely how the wrong camera gets trained
# on. Supply both, and prepare_cosmos_corpus.py enforces that the counts agree.
SMOKE_SOURCES=${SMOKE_SOURCES:-}
if [[ -n ${SMOKE_SOURCES} ]]; then
  read -r -a REPOS <<< "${SMOKE_SOURCES}"
  CORPUS=${PROJ}/data/cosmos-smoke
  ENV_OUT=${PROJ}/cosmos_env.smoke.sh
  echo "=== REHEARSAL: corpus replaced by ${#REPOS[@]} source(s): ${REPOS[*]}"
  echo "=== REHEARSAL: CORPUS=${CORPUS}  ENV_OUT=${ENV_OUT}  (registered corpus untouched)"
fi

# -- 1. fetch: meta + videos only. The parquet is state and action, which a generator never sees.
#
# THE GUARD IS A COMPLETION SENTINEL, NOT meta/info.json. `hf download` fetches `meta/*` before
# `videos/**`, so a run killed partway — walltime, preemption, a dropped connection — leaves a
# perfectly valid info.json next to an incomplete video tree. Keying the skip on that file made
# the retry declare the repo "present" and move on, handing step 2 a partial set it has no way to
# detect: episodes.jsonl lists episodes whose mp4s are missing, the manifest counts only what it
# found, and the run trains on a silently truncated corpus that still looks clean in every
# artifact. This file is written only after hf exits 0, so a resume re-enters exactly the repos
# that did not finish, and hf's own resume makes that cheap — completed files are not re-fetched.
for repo in "${REPOS[@]}"; do
  dest=${RAW}/$(basename "${repo}")
  if [[ -f "${dest}/.download-complete" ]]; then
    echo "=== ${dest} complete"
    continue
  fi
  echo "=== downloading ${repo}"
  # ONE --include per pattern. The hf CLI is click-based, so `--include a b` binds only `a`
  # to the flag and leaves `b` as a positional *filename*, which silently switches the whole
  # command into explicit-file mode ("Ignoring --include since filenames have been explicitly
  # set") and then 404s on a literal file named `videos/**`. That is what killed job 186283.
  uvx hf@latest download --repo-type dataset "${repo}" \
      --include 'meta/*' --include 'videos/**' --local-dir "${dest}"
  # set -e means we only reach this line when hf exited 0.
  touch "${dest}/.download-complete"
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

# Skip when a previous job in the chain already laid the corpus out. This exists so this job can
# be chained to itself with `afterany` and run under a short walltime: the first attempt does the
# work, and any follow-up that finds everything finished becomes a fast no-op instead of recopying
# 77 GB. Only a manifest written AFTER the last download counts — an older one was built from a
# corpus that has since grown, and re-running is then the correct thing to do.
# `|| true` because `set -o pipefail` is on: with no sentinels `ls` exits 2, head swallows it, the
# pipeline still reports 2, and the bare assignment would take the whole job down on a condition
# that is merely "nothing downloaded yet".
NEWEST_SENTINEL=$(ls -t "${RAW}"/*/.download-complete 2>/dev/null | head -1 || true)
if [[ -f "${CORPUS}/manifest.json" && -n ${NEWEST_SENTINEL:-} \
      && "${CORPUS}/manifest.json" -nt "${NEWEST_SENTINEL}" ]]; then
  echo "=== ${CORPUS}/manifest.json is newer than every download — layout already done, skipping"
else
  python "${PREPARE}" "${ARGS[@]}" \
      --out "${CORPUS}" --val-episodes "${VAL_EPISODES}" --seed "${SEED}"
fi

# A rehearsal corpus says so on disk. The manifest already records which sources went in, but that
# takes reading JSON; this is visible to `ls` and to any later job that wants to refuse.
if [[ -n ${SMOKE_SOURCES} ]]; then
  cat > "${CORPUS}/SMOKE" <<EOF
Rehearsal corpus — NOT the PR-09 §2 pre-registered corpus.
sources: ${REPOS[*]}
No verdict under T041_RULE_V1 may be issued from anything trained on this.
EOF
fi

# grep -q first: this job can run more than once in a chain, and the env file is sourced by every
# later job. Appending the same export on each pass would leave a file that still works but reads
# like a log, and the duplicate would hide a real change to CORPUS.
if ! grep -q "^export CORPUS=${CORPUS}$" "${ENV_OUT}" 2>/dev/null; then
  cat >> "${ENV_OUT}" <<EOF
export CORPUS=${CORPUS}
EOF
fi
echo "CORPUS OK -> ${CORPUS}"
echo "next: sbatch 93_caption_corpus.sbatch"
