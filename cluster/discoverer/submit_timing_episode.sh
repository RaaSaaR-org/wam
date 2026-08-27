#!/usr/bin/env bash
# Submit the ONE timed episode PR-08 §8 item 3 asks for. Runs on the workstation.
#
#   ./cluster/discoverer/submit_timing_episode.sh
#
# T40_RULE_V20 (docs/preregistration/PR-08-V20-timing-episode-registration.md), decided by the
# project owner on 2026-08-27, registers WHICH episode this measures — and registers it as a
# CRITERION rather than as a name, "so that it can be checked rather than trusted":
#
#     of the episodes both halves of check_mask accept, take the one whose frame count is closest
#     to the corpus median; break ties by lowest episode id.
#
# So this file RE-DERIVES that criterion from the committed evidence every time it runs, and
# refuses if the answer is not the one V20 §3 wrote down. Typing `CHUNK_INDEX=372` into a submit
# line would be the opposite: a name nobody can check, in the one place where a name chosen after
# looking at the data is exactly the failure V20 §4 exists to disclose.
#
# It also exists for the reason submit_geom_tol_wave.sh exists: the recipe is ~200 characters and a
# terminal paste broke the GEOM_TOL one at the wrap point on 2026-08-27, directly after `sbatch`.
# A short argv cannot break that way.
#
# It decides NOTHING. Every flag below is the header recipe of 97_transfer25_restyle.sbatch
# (lines 8-10) with V20's CHUNK_INDEX/CHUNK_TOTAL substituted for the header's placeholder 1-of-1,
# which is the whole difference between "time one episode" and "time the episode that check_mask
# refuses on frame 0".
set -euo pipefail

HOST=dplus
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480-h264-lossless}

# V20 §3's answer, held here only to be CHECKED against the re-derivation below — never to be used
# as the input. If these two disagree, the manifest changed and V20 says the submission does not
# proceed and the rule is re-evaluated. It is explicitly NOT retargeted at a neighbour.
V20_EPISODE=episode_000371
V20_FRAMES=422

# PR-08's committed control set (docs/transfer25-api.md:256). Not a knob: the control blocks decide
# how much geometry survives, which is what §6 G0b measures, and they decide the timing number too,
# because every block Transfer2.5 has to ESTIMATE is GPU time this measurement must include.
CONTROL=${CONTROL:-depth:0.5,seg:0.5}
# Dated, because the DEFAULT RUN_ID still holds job 189142's disqualified THROUGHPUT.json — 0.2
# s/frame from a run whose own log says "0 success, 1 error", which prices the partition ~10x
# under. throughput_qualification() refuses that artifact now, but a dated RUN_ID is the answer
# that also keeps the two measurements side by side.
RUN_ID=${RUN_ID:-t040-transfer25-restyle-timing-$(date -u +%Y-%m-%d)}
# The header calls this override PART OF THE RECIPE, not a tweak: the file's own 04:00:00 is sized
# for a generation chunk, and `common` churns on 1-2 hour jobs, so a 4 h request cannot fit the gaps
# the partition offers. Measured 2026-08-20: 4 h requests estimated 13-17 hours out; honest values
# started within seconds.
WALL=${WALL:-01:30:00}

# ---------------------------------------------------------------------------
# 1. Re-derive V20's criterion from the committed evidence, here, before anything is submitted.
# ---------------------------------------------------------------------------
echo "==> re-deriving T40_RULE_V20 §3 from runs/pr08-robot-mask-area/POOLED.json"
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DERIVED=$(python3 - "${REPO}" <<'PY'
import json, pathlib, statistics, sys

repo = pathlib.Path(sys.argv[1])
pooled = json.loads((repo / "runs/pr08-robot-mask-area/POOLED.json").read_text())
if not pooled.get("measurement_qualified"):
    sys.exit("POOLED.json is not measurement_qualified — V20 §2 computes its population from it.")
bound = json.loads(
    (repo / "configs/transfer25/pr08_robot_mask_area.json").read_text()
)["max_frame_fraction"]

eps = pooled["per_episode"]
# BOTH halves of check_mask, in the order V20 §2 states them: no empty robot mask on any frame, and
# no frame above the committed area bound. Neither half is waived here or anywhere.
survivors = [
    e for e in eps
    if e["empty_frames"] == 0 and max(e["area_fractions"]) <= bound
]
median = statistics.median([e["n_frames"] for e in eps])
pick = min(survivors, key=lambda e: (abs(e["n_frames"] - median), str(e["episode"])))
print(f"{pick['episode']} {pick['n_frames']} {len(survivors)} {median} {len(eps)}")
PY
)
read -r EPISODE FRAMES N_SURVIVORS MEDIAN N_EPISODES <<<"${DERIVED}"
echo "    ${N_SURVIVORS} of ${N_EPISODES} episodes survive both halves of check_mask"
echo "    corpus median ${MEDIAN} frames -> ${EPISODE} (${FRAMES} frames)"

if [[ "${EPISODE}" != "${V20_EPISODE}" || "${FRAMES}" != "${V20_FRAMES}" ]]; then
  echo "REFUSING: the criterion now yields ${EPISODE} (${FRAMES} frames), but T40_RULE_V20 §3" >&2
  echo "  recorded ${V20_EPISODE} (${V20_FRAMES}). The evidence under the rule has moved." >&2
  echo "  V20 §3: the submission does not proceed and the rule is re-evaluated against the" >&2
  echo "  evidence that exists. It is NOT silently retargeted at a neighbour." >&2
  exit 75
fi

# ---------------------------------------------------------------------------
# 2. Resolve the chunk index the same way the sbatch will, from the manifest and not from here.
# ---------------------------------------------------------------------------
# V20 §3 does not assert the index, deliberately: it depends on the manifest rather than on the
# document. The sbatch builds its work list as `sorted(episodes, key=str(id))[idx-1::total]`, so
# CHUNK_TOTAL = the episode count selects exactly one episode. Asking the manifest that will
# actually be read is the only way this is a check rather than a second guess.
echo "==> resolving CHUNK_INDEX from ${SOURCE}/manifest.json on ${HOST}"
RESOLVED=$(ssh "${HOST}" "python3 -c \"
import json
m = json.load(open('${SOURCE}/manifest.json'))
eps = [str(e['id']) for e in sorted(m['episodes'], key=lambda e: str(e['id']))]
res = tuple(m.get('resolution') or ())
print(res == (640, 480), len(eps), eps.index('${EPISODE}') + 1 if '${EPISODE}' in eps else 0)
\"")
read -r RES_OK CHUNK_TOTAL CHUNK_INDEX <<<"${RESOLVED}"
if [[ "${RES_OK}" != "True" ]]; then
  echo "REFUSING: the manifest does not declare 640x480. PR-08 §3 fixes that as the GR00T N1.7" >&2
  echo "  ego_view contract; a restyle at any other size is not the registered experiment." >&2
  exit 75
fi
if (( CHUNK_INDEX == 0 )); then
  echo "REFUSING: ${EPISODE} is not in ${SOURCE}/manifest.json." >&2
  echo "  T40_RULE_V20 §3: the submission does not proceed and the rule is re-evaluated against" >&2
  echo "  the manifest that exists." >&2
  exit 75
fi
if (( CHUNK_TOTAL != N_EPISODES )); then
  echo "REFUSING: the manifest holds ${CHUNK_TOTAL} episodes, the area evidence ${N_EPISODES}." >&2
  echo "  The population V20 §2 selected from is not the population that would be timed." >&2
  exit 75
fi
echo "    CHUNK_TOTAL=${CHUNK_TOTAL} CHUNK_INDEX=${CHUNK_INDEX}  (1-based, ${EPISODE})"

# ---------------------------------------------------------------------------
# 3. Submit-slot budget. MaxSubmitJobsPU=8 counts the WHOLE user's queue, and a peer session shares
#    this allocation. Refusing here is free; being rejected by sbatch is not.
# ---------------------------------------------------------------------------
echo "==> checking submit-slot budget on ${HOST}"
PENDING=$(ssh "${HOST}" "squeue -u \$USER -r -h -o '%i' | wc -l")
echo "    ${PENDING} job(s) submitted by this user right now (ceiling 8)"
if (( PENDING + 1 > 8 )); then
  echo "REFUSING: ${PENDING} + 1 > 8 = MaxSubmitJobsPU. Wait, then re-run." >&2
  exit 75
fi

# ---------------------------------------------------------------------------
# 4. Submit. TIMING=1 asks for no GPU-h ceiling and no T-39 attestation — PR-08 §1 licenses "timing
#    one episode on an H200" outright, and the budget is what this MEASURES.
# ---------------------------------------------------------------------------
CMD="cd ${WAM}/cluster/discoverer && TIMING=1 STAGE=1 STYLE_SET=train"
CMD="${CMD} CHUNK_INDEX=${CHUNK_INDEX} CHUNK_TOTAL=${CHUNK_TOTAL} CONTROL=${CONTROL}"
CMD="${CMD} RUN_ID=${RUN_ID} sbatch --time=${WALL} 97_transfer25_restyle.sbatch"

echo
echo "==> submitting the timed episode ${EPISODE} as RUN_ID=${RUN_ID}"
echo "    ${CMD}"
echo
ssh "${HOST}" "${CMD}"
echo
echo "T40_RULE_V20 §5 fixed the outcomes BEFORE this was submitted:"
echo "  M  THROUGHPUT.json with units_succeeded >= 1  -> item 3's rate exists, and every later"
echo "     quotation of it must carry 'measured on an episode selected for surviving G0c'."
echo "  R  the source-mask preflight refuses ${EPISODE} on the H200 -> the machine-conditionality"
echo "     in §4 is load-bearing, item 3 stays OPEN, and the area pass is owed a re-measurement on"
echo "     the machine that will generate. Walking down the 17 until one passes is REFUSED."
echo "  F  walltime, cold checkpoints, licence, a crash -> nothing was learned; repeat. An F may"
echo "     not be reported as an R."
echo
echo "In all three the run generates at most one clip, which is deleted. T40_RULE_V1 §1 is not"
echo "lifted and forbids everything else."
