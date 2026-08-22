#!/bin/bash
# Step 25 — PR-08 §4 blockers 1 and 2: produce the overlays somebody has to LOOK at.
#
#   sbatch cluster/discoverer/105_audit_apple_masks.sbatch
#
#   ...and then, from the workstation:
#       ./cluster/discoverer/sync.sh --pull pr08-mask-audit     # JSON only, see below
#       scp -r dplus:/valhalla/projects/ehpc-aif-2026pg01-905/runs/pr08-mask-audit/sheets .
#
# WHY THIS JOB EXISTS
# ===================
# scripts/estimators/apple_sam2.py carries GATE_QUALIFIED = False and three blockers, and the first
# of them opens with the sentence "NOBODY HAS LOOKED AT A MASK." Everything downstream is stuck
# behind it: with the flag false every shard of the GEOM_TOL array exits 3, the merge writes
# gate_qualified: false, the resume logic refuses to treat the shards as landed, and a re-submit
# re-measures the corpus and pays the ~4.23 GPU-h again. PR-08 §8 item 4 cannot close while it is
# false.
#
# Blocker 1 names its own discharge condition and it is not a computation: *a human looking at a
# sample of overlaid masks spanning the corpus (occluded frames, apple-out-of-frame frames, and the
# grasp)*. Blocker 2 adds the recorded detection-score distribution and the retry counts. This job
# makes both of those things exist. IT DOES NOT DISCHARGE EITHER. scripts/audit_apple_masks.py
# cannot write GATE_QUALIFIED — tests/test_audit_apple_masks.py asserts the source contains no
# assignment to it — and the artifact says so in three separate fields. The discharge is a person
# looking at sheets/ and then editing GATE_QUALIFICATION_BLOCKERS by hand.
#
# BLOCKER 3 IS NOT TOUCHED HERE, AND MUST NOT BE READ AS ADDRESSED.
# The remaining difference from upstream is that this adapter re-detects per frame where
# Cosmos-Transfer2.5 propagates one mask with SAM2VideoPredictor. Discharging that needs the same
# Isaac capture measured BOTH ways. Nothing in this job renders anything in Isaac, segments anything
# twice, or produces a second p95. The artifact copies all three blockers in verbatim precisely so
# that a reader cannot mistake a shortened list for a satisfied one.
#
# WHY THE H.264 TREE AND NOT THE AV1 ONE
# ======================================
# ${PROJ}/data/pr08-apple-640x480 is AV1 and the cv2 4.11.0 in 98's venv is built against an
# avcodec with no AV1 decoder. VideoCapture does not raise on that: it opens the container, reports
# 590 frames off the header, and reads none of them (jobs 186357, 189584, 189585). The H.264
# lossless tree was transcoded plane-for-plane with crf=0 and PROVEN bit-exact per clip
# (TRANSCODE_PROOF.json), and job 189613 verified all 402 clips decode in the generation venv. So
# the pixels are the same pixels and this job can use the decoder the generator uses. --decoder auto
# still PROBES before choosing and records the trail: a decoder that imports is not evidence that it
# reads this corpus.
#
# WHY THE CENSUS IS REQUIRED AND NOT OPTIONAL
# ===========================================
# Blocker 1 asks for OCCLUDED frames. On this corpus they are rare and concentrated:
# build_identity_calibration.py probe-scan measured the strict apple mask on all 154 447 frames of
# the 362 non-measured episodes and found 48 below 1 200 px of visible apple, 24 of them eligible,
# ALL IN ONE EPISODE. An episode sample drawn by stride therefore contains, with high probability,
# no occlusion at all — and the artifact would look exactly the same. probe_census.json is the only
# thing that knows which episode to force in, so this job refuses without it and prints the copy
# command. ALLOW_NO_CENSUS=1 says "a sample with no measured occlusion is what I meant", and the
# artifact records that choice.
#
# SIZING: MEASURED, NOT COPIED FROM 103
# =====================================
# 103 asks for 26 threads and 192 GB because that is the fair-share rate for one GPU. sacct says
# job 189588 — this same adapter, over this same corpus — peaked at MaxRSS = 4.75 GiB. A 40x
# over-request costs real allocation: billing per minute is GPUs*1.0 + MemGB*0.25 + Threads*0.036,
# so memory dominates everything else and 192 GB alone is 48 billing-minutes per wall-minute against
# the GPU's 1. This job asks for 24 GB and 8 threads — 7.3 billing-min/min instead of 49.9 — which
# is still five times the measured peak. The rule this respects is the ceiling (never more than 26
# threads or 257 GB per GPU), not a floor: asking for less than the fair-share rate is always
# allowed and is the only way the allocation lasts.
#
# WALLTIME. The work is one segmenter forward per sampled frame — a few hundred frames, not the
# 171 625 the GEOM_TOL run pays for — plus two cheap decode passes over the sampled episodes and a
# few hundred PNGs. At the pilot's measured rate that is a couple of minutes of GPU inside about ten
# minutes of wall. 30 minutes is the honest request, and an honest short request BACKFILLS on
# `common` while a padded 4 h one waits behind ~40 pending jobs at the same priority.

#SBATCH --job-name=wam-mask-audit
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# NEVER OMIT --qos. Without it Slurm lands the job on `normal` — 1 minute, 0 GPUs — which is a trap
# and not a fallback.
#SBATCH --qos=ehpc-aif-2026pg01-905
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# See SIZING above. Measured from job 189588's MaxRSS, not copied from 103's fair-share rate.
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/mask-audit.%j.out

set -euo pipefail

# /home is 2 GiB and ~100k inodes; caches.sh points HF_HOME, TORCH_HOME, TRITON_CACHE_DIR and
# TMPDIR at project storage before any tool can decide for itself.
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
# The LOSSLESS tree. See WHY THE H.264 TREE above; overriding this back to the AV1 corpus will not
# crash, it will decode zero frames and report that as a fact about the corpus.
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480-h264-lossless}
RUN_ID=${RUN_ID:-pr08-mask-audit}
OUT=${PROJ}/runs/${RUN_ID}
mkdir -p "${OUT}"

EPISODES=${EPISODES:-24}
MAX_ANCHORS=${MAX_ANCHORS:-12}
SHEET_TILES=${SHEET_TILES:-12}

# 98's env, then its venv: sam2 1.1.0, transformers and torch already live there, so this job
# installs nothing. A job that pip-installs is a job whose result depends on the day it ran.
TRANSFER_ENV=${TRANSFER_ENV:-${PROJ}/cosmos_transfer_env.sh}
[[ -f "${TRANSFER_ENV}" ]] || {
  echo "FATAL: ${TRANSFER_ENV} missing. Run 98_build_transfer25_env.sbatch first — it writes this"
  echo "       file and owns the .venv that carries sam2 and transformers."
  exit 1
}
source "${TRANSFER_ENV}"
source "${FRAMEWORK}/.venv/bin/activate"

[[ -d "${SOURCE}" ]] || { echo "FATAL: SOURCE ${SOURCE} does not exist."; exit 1; }
N_CLIPS=$(find "${SOURCE}" -name '*.mp4' | wc -l)
echo "=== source     ${SOURCE}  (${N_CLIPS} clips, H.264 lossless — bit-exact, job 189613)"
echo "=== framework  ${FRAMEWORK}"
echo "=== python     $(command -v python)"
echo "=== gpu        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# The estimator weights 102 staged, at the revisions pinned in scripts/estimators/apple_sam2.py.
# Absent, the adapter refuses (correct) — but say so here, before a GPU is held, and name the fix.
STAGED=${PROJ}/checkpoints/PR08_ESTIMATORS_STAGED.json
[[ -f "${STAGED}" ]] || {
  echo "FATAL: ${STAGED} missing — the detector, segmenter and depth checkpoints are not staged."
  echo "       Run 102_stage_sam2_weights.sbatch (free QoS, ~15 s once the cache is warm)."
  exit 1
}
echo "=== estimators $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(", ".join(c["repo_id"]+"@"+c["revision_resolved"][:9] for c in d["checkpoints"]))' "${STAGED}")"

# WAM_PR08_ALLOW_DOWNLOAD STAYS UNSET. Unset is not merely "no permission to fetch": it is what
# switches the adapter's offline enforcement ON, so a hub request cannot leave this node and no
# load can quietly resolve a different revision from the staged, checksum-verified one.
if [[ -n "${WAM_PR08_ALLOW_DOWNLOAD:-}" ]]; then
  echo "FATAL: WAM_PR08_ALLOW_DOWNLOAD is set (${WAM_PR08_ALLOW_DOWNLOAD}). This job reads staged"
  echo "       weights only. Unset it — that is also what turns offline enforcement on."
  exit 1
fi

# The adapter is not on the framework's import path; it lives in this repo beside the scripts that
# drive it, and both resolve it as `estimators.apple_sam2`, so scripts/ is what must be importable.
export PYTHONPATH="${WAM}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

# -- the census ------------------------------------------------------------------------------------
CENSUS=${CENSUS:-}
if [[ -z "${CENSUS}" ]]; then
  for c in "${PROJ}/runs/t040-identity-prompt/calibration-2/probe_census.json" \
           "${WAM}/runs/t040-identity-prompt/calibration-2/probe_census.json"; do
    [[ -f "${c}" ]] && { CENSUS=${c}; break; }
  done
fi
CENSUS_ARGS=()
if [[ -n "${CENSUS}" ]]; then
  echo "=== census     ${CENSUS}"
  echo "===            $(sha256sum "${CENSUS}" | cut -c1-16)... — probe-scan over 154447 frames of"
  echo "===            the 362 non-measured episodes. It is what forces the one episode that"
  echo "===            actually contains an occlusion into the sample."
  CENSUS_ARGS=(--census "${CENSUS}")
elif [[ "${ALLOW_NO_CENSUS:-0}" != "0" ]]; then
  echo "=== census     NONE — ALLOW_NO_CENSUS=1. The sample will contain no MEASURED occlusion and"
  echo "===            the artifact records that. Blocker 1 asks for occluded frames by name, so"
  echo "===            this weakens the evidence rather than merely shortening the run."
  CENSUS_ARGS=(--allow-missing-census)
else
  echo "FATAL: probe_census.json is not on this machine and ALLOW_NO_CENSUS is not set."
  echo "       It is build_identity_calibration.py probe-scan's measurement over all 154447 frames"
  echo "       of the 362 non-measured episodes, and it is the ONLY thing that knows which episode"
  echo "       contains an occlusion — 48 below-threshold frames in one episode out of 362. Without"
  echo "       it a stride sample contains no occluded frame and looks identical to one that does."
  echo "       sync.sh EXCLUDES runs/ in both directions, so it has to go over by hand:"
  echo "         ssh dplus mkdir -p ${PROJ}/runs/t040-identity-prompt/calibration-2"
  echo "         scp runs/t040-identity-prompt/calibration-2/probe_census.json \\"
  echo "             dplus:${PROJ}/runs/t040-identity-prompt/calibration-2/"
  echo "       Or re-run it here: build_identity_calibration.py probe-scan (CPU, free QoS)."
  exit 1
fi

echo "=== OUT        ${OUT}"
echo "=== sample     ${EPISODES} episode(s), <= ${MAX_ANCHORS} anchors each, plus each hard"
echo "===            anchor's successor frame so adjacent-frame centroid steps are measurable."
echo "=== NOT A DISCHARGE. This job writes evidence. GATE_QUALIFIED stays false and all three"
echo "===            blockers are copied into the artifact verbatim, including blocker 3 (per-frame"
echo "===            segmentation vs SAM2VideoPredictor propagation), which NOTHING here addresses."

START=${SECONDS}
# `set -e` is on, so a bare invocation followed by RC=$? never sees a non-zero status — the shell is
# already gone by the time the assignment runs. The `if` is where set -e does not fire.
if python "${WAM}/scripts/audit_apple_masks.py" \
  --corpus "${SOURCE}" \
  --out "${OUT}" \
  "${CENSUS_ARGS[@]}" \
  --episodes "${EPISODES}" \
  --max-anchors-per-episode "${MAX_ANCHORS}" \
  --sheet-tiles "${SHEET_TILES}" \
  --slurm-job-id "${SLURM_JOB_ID:-}"; then RC=0; else RC=$?; fi
ELAPSED=$((SECONDS - START))
echo "=== audit_apple_masks.py exited ${RC} after ${ELAPSED}s"

if (( RC != 0 )); then
  echo "FATAL: the audit exited ${RC} and no evidence was written. Exit 2 is a refusal and names"
  echo "       its reason above: the adapter would not load, the corpus decoded nothing, or the"
  echo "       census is missing. There is no partial artifact to read."
  exit "${RC}"
fi

OUT="${OUT}" python - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["OUT"])
d = json.loads((out / "MASK_AUDIT.json").read_text())
c, b2 = d["counts"], d["blocker_2_numbers"]
print("=== frames segmented      %d (%d anchors + %d neighbours) over %d episode(s)"
      % (c["n_frames_segmented"], c["n_anchor_frames"], c["n_neighbour_frames"],
         d["corpus"]["episodes_audited"]))
print("=== per stratum           " + ", ".join(f"{k}={v}" for k, v in c["per_stratum"].items()))
print("=== strata NOT FOUND      " + (", ".join(d["sampling"]["strata_not_found"]) or "(none)")
      + "   <- the corpus has none, which is itself a finding")
print("=== strata cut by budget  "
      + (", ".join(d["sampling"].get("strata_truncated_by_budget") or []) or "(none)")
      + "   <- raise MAX_ANCHORS to see these")
print("=== n_frames_retry_fired  %d   n_frames_retry_recovered  %d"
      % (b2["n_frames_retry_fired"], b2["n_frames_retry_recovered"]))
print("=== no-detection          %d (%.1f%% of this SAMPLE, which over-weights hard frames)"
      % (b2["n_frames_without_detection"], 100 * b2["no_detection_rate"]))
s = b2["detection_score_distribution"]
if s["n"]:
    print("=== detection score       min %.3f  p05 %.3f  median %.3f  max %.3f  (n=%d)"
          % (s["min"], s["p05"], s["median"], s["max"], s["n"]))
a = b2["mask_area_px_distribution"]
if a["n"]:
    print("=== mask area px          min %d  median %d  p95 %d  max %d"
          % (a["min"], a["median"], a["p95"], a["max"]))
print("=== flagged               %d frame(s): %s"
      % (c["n_frames_flagged"], ", ".join(f"{k}={v}" for k, v in c["flag_counts"].items()) or "-"))
print("=== gate_qualified        %s  (unchanged, and this job cannot change it)"
      % d["estimator"]["gate_qualified_read_from_adapter"])
print("=== blockers still open   %d, copied in verbatim"
      % len(d["estimator"]["gate_qualification_blockers_verbatim"]))
PY

echo "=== NEXT, and it is the whole point of the job:"
echo "===   1. Copy the PICTURES back. sync.sh --pull takes .json/.jsonl/.md/.txt only, so it will"
echo "===      bring MASK_AUDIT.json and OBSERVATIONS.template.json and NONE of the overlays:"
echo "===        scp -r dplus:${OUT}/sheets ./runs/${RUN_ID}/"
echo "===        scp -r dplus:${OUT}/frames ./runs/${RUN_ID}/     # the full-size per-frame PNGs"
echo "===   2. LOOK at sheets/flagged-*.png first, then every other sheet. For each frame decide"
echo "===      whether the green mask is the APPLE — not whether it is a plausible object."
echo "===   3. Write what you saw into OBSERVATIONS.template.json, rename it OBSERVATIONS.json."
echo "===   4. ONLY THEN is blocker 1 in a position to be discharged, and that is an edit to"
echo "===      scripts/estimators/apple_sam2.py's GATE_QUALIFICATION_BLOCKERS by a person, moving"
echo "===      the retired wording into GATE_QUALIFICATION_DISCHARGED with this artifact cited."
echo "===      Blocker 3 remains open regardless: it needs an Isaac capture measured both ways."
