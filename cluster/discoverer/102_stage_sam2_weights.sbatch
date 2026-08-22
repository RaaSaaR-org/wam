#!/bin/bash
# Step 22 — T-040 / PR-08 §8 item 4: stage the ESTIMATOR weights that GEOM_TOL and EST_DRIFT_P95
# are measured with, at pinned revisions, into project storage.
#
#   sbatch 102_stage_sam2_weights.sbatch
#
# WHY THIS IS ONE JOB AND NOT THREE.
# §8 item 4 wants two numbers measured and committed before a single clip is generated: GEOM_TOL
# (scripts/measure_geom_tol.py) and EST_DRIFT_P95 (scripts/measure_est_drift.py). Both need an
# object segmenter, and §4 step 2 says it must be **the same one** -- §6 subtracts the two numbers,
# and two segmenters is two different quantities whose difference is not arithmetic. So one
# segmenter closes both, one staging job stages it, and there is no second copy to drift.
#
# WHICH SEGMENTER, AND WHY IT IS NOT A PREFERENCE.
# Both scripts refuse to invent one: `measure_geom_tol.no_segmenter_message()` lists four candidates
# and stages none of them, because a stand-in that produces a number is worse than a crash -- it
# sets G0b's tolerance and every later verdict inherits it silently. The tie is broken by the
# generator rather than by taste. Cosmos-Transfer2.5 drives its own segmentation control from
#
#     cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py
#         SAM2_MODEL_CHECKPOINT           = "facebook/sam2-hiera-large"
#         GROUNDING_DINO_MODEL_CHECKPOINT = "IDEA-Research/grounding-dino-base"
#
# i.e. text prompt -> GroundingDINO box -> SAM2 mask, via transformers
# AutoModelForZeroShotObjectDetection + AutoProcessor. Measuring the tolerance with the estimator
# the generator itself runs is the strongest available reading of "the same segmenter": it makes
# GEOM_TOL a property of the pipeline under test instead of a property of whatever we happened to
# install. `sam2` 1.1.0 and its hiera configs are already in 98's venv
# (98_build_transfer25_env.sbatch:47-50), so nothing here is a new dependency -- only the weights
# are missing, and NO checkpoint is staged for any of the three as of 2026-08-22.
#
# THE DEPTH CHECKPOINT IS **METRIC**, AND THAT IS THE WHOLE OF THE ARGUMENT.
# An earlier revision of this file staged `depth-anything/Video-Depth-Anything-Large` on the
# reasoning that it is what the generator drives for its own depth control (`DepthConfig`,
# docs/transfer25-api.md:161) -- "the identical argument that fixes the segmenter". It is not the
# identical argument, and the difference is the defect:
#
#   * §4 step 2 constrains the SEGMENTER because §6 subtracts our mask error from the generator's
#     mask error, so the two have to be errors of the same instrument.
#   * §4 step 3 asks for the ABSOLUTE DEPTH ERROR IN METRES, and the ground truth it is subtracted
#     from is Isaac's `distance_to_camera`, which is metres. Video-Depth-Anything emits
#     affine-invariant INVERSE depth: arbitrarily scaled per clip, larger = nearer, no metres in it
#     anywhere. Subtracting that from metres does not crash and does not produce an error -- it
#     produces a plausible float, written under a key called `mean_m`, that every later reader
#     inherits. The generator's control map has no obligation to be in metres and PR-08's step 3
#     has no way to do without it.
#
# So the depth id here is the METRIC head the adapter already defaults to, at the revision read off
# the HF API on 2026-08-22 (ungated, pipeline_tag depth-estimation):
#
#     depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf @ d2fc6a93...
#
# It is transformers-native (`AutoModelForDepthEstimation`), i.e. it loads out of 98's pinned
# transformers rather than out of the `video-depth-anything` package, and `Indoor` rather than
# `Outdoor` because the metric heads carry a domain `max_depth` (20 m vs 80 m) and AppleToPlate is a
# tabletop under 2 m. scripts/estimators/apple_sam2.py refuses a relative checkpoint at load time by
# reading `depth_estimation_type` off the loaded config, so staging the relative one would have cost
# a staging run reported as success and a measurement that never starts -- or, worse, one that does.
#
# WHERE THE IDS COME FROM, AND WHY THIS JOB CANNOT DISAGREE WITH THE MEASUREMENT QUIETLY.
#     ** THE ID STAGED HERE AND THE ID THE ESTIMATOR WIRING DRIVES MUST BE THE SAME STRING. **
# If they diverge, nothing crashes: the wiring silently fetches its own checkpoint at generation
# time, at an unpinned revision, and EST_DRIFT_P95 ends up characterising an estimator the run does
# not use -- the same failure docs/transfer25-api.md §"The item this dissolves" flags for the
# control maps, one layer down.
#
# The single source of truth is scripts/estimators/apple_sam2.py, which declares the ids AND their
# revisions as module-level constants. This job restates them below so that a reader of the sbatch
# sees what it fetches, and then EXTRACTS the adapter's own values and compares
# them string for string, BEFORE the first byte is downloaded. A disagreement is exit 1. An earlier
# revision searched a committed artifact for the PRESENCE of an id (a fixed-string grep over the
# whole file), which is structurally incapable of reporting a disagreement -- presence is not
# agreement, and a check whose only two outcomes are "found" and "not found" says nothing on the day
# the two files name different checkpoints. It also cannot tell "recorded as the checkpoint loaded"
# from "listed as a candidate that was rejected", so its hit was weaker than the line it printed.
#
# WHAT IT TAKES FROM EACH REPO, AND WHAT IT DELIBERATELY DOES NOT PRUNE.
# The whole repo, all three times. Both `facebook/sam2-hiera-large` and
# `IDEA-Research/grounding-dino-base` ship their weights TWICE -- safetensors for the transformers
# loader plus, respectively, a native `sam2_hiera_large.pt` (+ `sam2_hiera_l.yaml` for hydra) and a
# legacy `pytorch_model.bin`. Dropping the duplicate would be a bet on which loader path the driver
# takes: the env pins transformers 4.51.3, and whether that build's AutoModel knows the `sam2_video`
# model_type this repo declares is a question for whoever wires the driver, not for this job. Being
# wrong costs an unpinned fetch inside a measured window; being right saves ~1.8 GB on a filesystem
# that holds 55 GB of Transfer2.5 already.
#
# The total is RECORDED, never asserted: a byte count in a comment is a number nobody re-measures.
# What IS asserted is completeness. The reachability probe reads each revision's file list off the
# HF API and the manifest refuses any snapshot missing a file that list names, because a 429 halfway
# through is not a clean failure (job 100 died on one at 80%) -- it leaves a partial snapshot that a
# later loader treats as a cache hit. For the same reason a snapshot entry that cannot be read
# (a dangling blob symlink, a stat that raises) is a fatal error and not a shorter manifest.
#
# NO GPU, SO NO GPU-HOURS -- the same rule 99's header states. This job downloads and hashes; it
# never materialises a model. It runs on the free 2cpu-single-host QoS and bills zero against
# PR-08's still-unwritten ceiling, which matters because §8 item 3 says no budget line exists until
# the throughput measurement does.
#
# IT DOES NOT DEPEND ON 98 HAVING RUN. Unlike 99/99b it does not source cosmos_transfer_env.sh and
# does not import anything from the framework: it needs `caches.sh`, `uv`, python3 and the network.
# That is deliberate -- the estimator measurements are on §8 item 4's critical path and item 3's env
# build is not their prerequisite. If the framework checkout HAPPENS to be present it is used,
# read-only, to cross-check the two segmenter ids against upstream's own constants. The repo
# checkout is NOT optional in the same way: without the adapter there is nothing for the staged ids
# to agree with, and this job refuses rather than staging several GB against an unread contract.
#
# EXIT STATUS
#   0  every checkpoint staged, hashed and complete, and every id this job can check was checked and
#      agreed.
#   1  fatal: no token, no uv, no adapter, a moving-pointer revision, a download that did not land,
#      a snapshot missing files the hub says exist, or an id that DISAGREES with the adapter, the
#      generator's own constant or a committed measurement artifact. Nothing is guessed past.
#   3  staged and hashed, but at least one id could not be VERIFIED against something that DOES
#      exist and should have agreed (no framework checkout, no revision constant in the adapter, an
#      artifact that did not parse, an artifact recording no id in the shape this reads). A
#      measurement artifact that is simply NOT WRITTEN YET is exit 0 instead: it is this job's own
#      precondition, since the measurements that write it need the weights staged here, so treating
#      it as a fault made a correct first run unable to exit 0 at all.
#      The bytes are on disk and usable; what is not established is
#      that they are the right bytes. The manifest is still written with `staged_qualified: false`
#      and the reasons, because "we tried and this is what came out" is a record. READ A 3 AS
#      "STAGED, NOT VERIFIED" -- not as a failed download, and not as permission to skip the
#      verification later.
#
#SBATCH --job-name=wam-sam2-stage
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
# 2cpu-single-host is the free tier: zero GPU-hours, and the ONLY reason this is safe to submit
# against a budget nobody has written yet. Omitting --qos does not fall back to something similar,
# it falls into "normal" = 1 minute of walltime and 0 GPUs, i.e. a job that dies before the first
# file lands.
#SBATCH --qos=2cpu-single-host
# TWENTY MINUTES, AND THE ARITHMETIC IS THE POINT. 99b pulled 22 GB in 2:31 on this partition, so a
# few GB is well under a minute of transfer, plus a sha256 pass over the same bytes. Twenty minutes
# is ~10x that and still short enough to land in a backfill gap. Asking for the 4 h ceiling because
# it is available is how a five-minute job waits thirteen hours: the partition churns on 1-2 h jobs
# and cannot fit a 4 h request into the gaps it keeps offering. If this ever times out, the cause is
# an HF outage and the fix is to resubmit, not to widen the request.
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/sam2-stage.%j.out

set -euo pipefail

# FIRST, AND BEFORE ANY TOOL CAN DECIDE FOR ITSELF WHERE TO WRITE. caches.sh points HF_HOME and
# HF_HUB_CACHE at project storage. /home is 2 GiB with a ~100k inode limit; the three repos below
# are several GB and the hub cache layout is thousands of files. An HF download that lands in $HOME
# does not warn, it fills the quota and takes every other job on this account down with it.
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
REPO_ROOT=${REPO_ROOT:-${PROJ}/wam}
TRANSFER_FRAMEWORK=${TRANSFER_FRAMEWORK:-${PROJ}/third_party/cosmos-transfer2.5}
MANIFEST=${MANIFEST:-${PROJ}/checkpoints/PR08_ESTIMATORS_STAGED.json}
# The single source of truth for the ids. Not optional -- see the header.
ESTIMATOR_ADAPTER=${ESTIMATOR_ADAPTER:-${REPO_ROOT}/scripts/estimators/apple_sam2.py}
TMPDIR=${TMPDIR:-/tmp}

# -- the pins --------------------------------------------------------------------------------------
# Every revision is a 40-hex commit read off the HF API on 2026-08-22, and none of them has a
# floating default. Same rule, same reason as 99: a repo id plus `main` is not a checkpoint id, and
# these particular repos are the ones a loader will happily re-resolve for itself at run time if it
# finds nothing staged. Pinning here plus HF_HOME pointing at the staged cache is what stops an
# unpinned auto-download from deciding what we measured.
#
# These are RESTATEMENTS of scripts/estimators/apple_sam2.py's constants, not a second opinion:
# check_ids.py below extracts that file's values and this job dies if they differ.
SAM2_MODEL_ID=${SAM2_MODEL_ID:-facebook/sam2-hiera-large}
SAM2_MODEL_REVISION=${SAM2_MODEL_REVISION:-e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251}

GDINO_MODEL_ID=${GDINO_MODEL_ID:-IDEA-Research/grounding-dino-base}
GDINO_MODEL_REVISION=${GDINO_MODEL_REVISION:-12bdfa3120f3e7ec7b434d90674b3396eccf88eb}

# METRIC, not relative. See the header: §4 step 3 wants metres, and the relative head has none.
# Override BOTH together -- a new id with the old revision resolves to nothing.
DEPTH_MODEL_ID=${DEPTH_MODEL_ID:-depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf}
DEPTH_MODEL_REVISION=${DEPTH_MODEL_REVISION:-d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f}

echo "=== segmenter  ${SAM2_MODEL_ID} @ ${SAM2_MODEL_REVISION}"
echo "=== detector   ${GDINO_MODEL_ID} @ ${GDINO_MODEL_REVISION}"
echo "=== depth      ${DEPTH_MODEL_ID} @ ${DEPTH_MODEL_REVISION}"
echo "=== hub cache  ${HF_HUB_CACHE}"
echo "=== adapter    ${ESTIMATOR_ADAPTER}"
echo "=== manifest   ${MANIFEST}"

refuse_moving_pointer() {
  # A tag, a branch or an empty string names whatever upstream pushed last, which is a different
  # answer on a different day. PR-08 §6 requires the revision recorded regardless of verdict, and a
  # recorded "main" records nothing.
  local name=$1 value=$2
  case "${value}" in
    main|master|HEAD|latest|"")
      echo "FATAL: ${name}='${value}' is a moving pointer, not a revision. Read a commit off the"
      echo "       HF API and pin it:"
      echo "         curl -s https://huggingface.co/api/models/<repo> \\"
      echo "           | python3 -c 'import json,sys;print(json.load(sys.stdin)[\"sha\"])'"
      exit 1 ;;
  esac
  [[ "${value}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "FATAL: ${name}='${value}' is not a 40-hex commit sha."; exit 1; }
}
refuse_moving_pointer SAM2_MODEL_REVISION  "${SAM2_MODEL_REVISION}"
refuse_moving_pointer GDINO_MODEL_REVISION "${GDINO_MODEL_REVISION}"
refuse_moving_pointer DEPTH_MODEL_REVISION "${DEPTH_MODEL_REVISION}"

# -- uv ----------------------------------------------------------------------------------------------
# `uvx hf@latest` is 99's way of getting an HF client without owning a venv, and it is why this job
# does not need 98's. uv itself is installed by 90/98, not here: installing a toolchain as a
# side-effect of a staging job is how two copies of it end up on PATH in different jobs.
if ! command -v uv >/dev/null 2>&1; then
  for d in "${UV_INSTALL_DIR:-}" "${PROJ}/tools/uv" "${PROJ}/tools/uv-bin"; do
    [[ -n "${d}" && -x "${d}/uv" ]] && { export PATH="${d}:${PATH}"; break; }
  done
fi
command -v uv >/dev/null 2>&1 || {
  echo "FATAL: no uv on PATH. Looked at: \${UV_INSTALL_DIR}='${UV_INSTALL_DIR:-<unset>}',"
  echo "       ${PROJ}/tools/uv, ${PROJ}/tools/uv-bin, and PATH itself."
  echo "       98_build_transfer25_env.sbatch installs it into ${PROJ}/tools/uv; run that, or"
  echo "       export UV_INSTALL_DIR to wherever it already lives. This job will not install a"
  echo "       second copy."
  exit 1; }
echo "=== uv         $(command -v uv)"

# python3 is the interpreter for the id comparison, the file list and the manifest -- all stdlib,
# no venv. Checked here rather than discovered as a 127 from a command substitution, which the
# cross-check would otherwise report as "the adapter disagrees".
command -v python3 >/dev/null 2>&1 || {
  echo "FATAL: no python3 on PATH. This job's checks and its manifest are stdlib python; without"
  echo "       an interpreter it can only download bytes and assert nothing about them."
  exit 1; }
echo "=== python3    $(command -v python3) ($(python3 -c 'import sys;print(sys.version.split()[0])'))"

# -- the token -------------------------------------------------------------------------------------
# All three repos are PUBLIC ("gated": false on the HF API, 2026-08-22) and a token is still
# mandatory, because anonymous does not mean unauthenticated, it means rate-limited: job 100 died on
# a 429 at 80% of its fetch. Read from the environment first, then the standard caches, exactly as
# 99/99b do. Never echoed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  for f in "${HF_HOME:-${HOME}/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
    [[ -f "${f}" ]] && { HF_TOKEN=$(<"${f}"); break; }
  done
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: no Hugging Face token. Looked at \$HF_TOKEN,"
  echo "       ${HF_HOME:-${HOME}/.cache/huggingface}/token and ${HOME}/.huggingface/token."
  echo "       These repos are public, so this is NOT a licence gate -- it is rate limiting, and"
  echo "       proceeding anonymously is how job 100 got a 429 at 80%. Export HF_TOKEN, or run"
  echo "       'hf auth login' as the account holder. A READ token is enough."
  exit 1
fi
export HF_TOKEN
echo "=== HF token   present (${#HF_TOKEN} chars, not echoed)"

# -- the cross-checks, BEFORE a single byte -------------------------------------------------------
# Staging the right bytes for the wrong id is the failure this whole file is arranged against, and it
# does not crash: the driver fetches its own copy and the measurement describes an estimator the run
# never used. So each id is compared, where it can be, against the thing it has to agree with -- and
# it is compared FIRST, because a disagreement discovered after the download is a disagreement that
# cost 4 GB and looks like a success in the log.
UNVERIFIED=()
CHECKS=()

check_framework_constant() {
  # STRICT, because it is an equality on a string literal in upstream's own source. If the framework
  # checkout is here, its constants are the ground truth for the segmenter pair and a disagreement
  # is fatal -- upstream renaming its checkpoint would otherwise leave us measuring with the model
  # it stopped using.
  local const=$1 want=$2 src="${TRANSFER_FRAMEWORK}/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py" got
  if [[ ! -f "${src}" ]]; then
    echo "=== UNVERIFIED ${const}: no framework checkout at ${src}"
    UNVERIFIED+=("${const}: no framework checkout at ${src} — run 98_build_transfer25_env.sbatch and rerun this job to establish agreement")
    CHECKS+=("${const}"$'\t'"unverified"$'\t')
    return 0
  fi
  got=$(sed -n "s/^${const}[[:space:]]*=[[:space:]]*[\"']\\([^\"']*\\)[\"'].*/\\1/p" "${src}" | head -n 1)
  if [[ -z "${got}" ]]; then
    echo "=== UNVERIFIED ${const}: present in ${src} but not in the shape this reads"
    UNVERIFIED+=("${const}: ${src} exists but no '${const} = \"...\"' line parsed — upstream changed the shape; re-read it by hand before trusting the staged id")
    CHECKS+=("${const}"$'\t'"unparsed"$'\t')
    return 0
  fi
  if [[ "${got}" != "${want}" ]]; then
    echo "FATAL: ${const} in the framework checkout is '${got}', this job staged '${want}'."
    echo "       Measuring with a segmenter the generator does not use makes GEOM_TOL and"
    echo "       EST_DRIFT_P95 properties of our install rather than of the pipeline under test."
    echo "       Fix the id here (and its revision), or pin a framework sha where they agree."
    exit 1
  fi
  echo "=== verified   ${const} == ${got}  (framework checkout)"
  CHECKS+=("${const}"$'\t'"agrees"$'\t'"${got}")
}
check_framework_constant SAM2_MODEL_CHECKPOINT           "${SAM2_MODEL_ID}"
check_framework_constant GROUNDING_DINO_MODEL_CHECKPOINT "${GDINO_MODEL_ID}"

# The adapter and the committed artifacts, compared field for field rather than grepped for. Written
# out to a file and run rather than inlined into a pipeline so that a pytest can extract it verbatim
# and feed it a disagreeing adapter -- a check nobody can exercise is a check nobody has seen fail.
CHECK_IDS_PY=$(mktemp "${TMPDIR}/pr08-check-ids.XXXXXX.py")
CHECK_IDS_OUT=$(mktemp "${TMPDIR}/pr08-check-ids.XXXXXX")
STAGED_LIST=$(mktemp "${TMPDIR}/pr08-estimators.XXXXXX")
trap 'rm -f "${CHECK_IDS_PY}" "${CHECK_IDS_OUT}" "${STAGED_LIST}"' EXIT

cat > "${CHECK_IDS_PY}" <<'PY_CHECK_IDS'
#!/usr/bin/env python3
"""Compare the ids/revisions a staging job is about to fetch against the wiring that loads them.

    check_ids.py <adapter.py> [<artifact.json> ...]        PR08_WANT=role\tid\trevision (\x1f-joined)

The adapter is READ, never imported: importing it pulls numpy and torch into a staging job that
deliberately owns no venv, and an ImportError would be reported as "cannot verify" when the file is
sitting right there and perfectly readable. `ast` gets the module-level constants without executing
anything, including the ``os.environ.get("VAR", "default")`` form the adapter uses -- and when VAR
is set in THIS job's environment, that override is what the adapter would see, so that is the value
compared and the source is recorded.

Output, one tab-separated record per line on stdout:

    INFO        <message>
    CHECK       <name>  <agrees|unverified>  <detail>
    UNVERIFIED  <sentence, for the manifest's `unverified` list>
    FATAL       <sentence>

Exit 0: nothing disagreed (some things may be unverified, which the caller turns into exit 3).
Exit 2: something DISAGREED. Not "was not found" -- found, and different. There is no third
outcome where a value is close enough.
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: role -> the adapter constant that must hold this job's repo id.
ROLE_ID_CONSTANT = {
    "segmenter": "SAM2_MODEL_CHECKPOINT",
    "detector": "GROUNDING_DINO_MODEL_CHECKPOINT",
    "depth": "DEPTH_MODEL_CHECKPOINT",
}
#: role -> substrings that identify a revision constant as belonging to it. The adapter may not
#: declare revisions yet; when it does, this finds them by name rather than by position.
ROLE_REVISION_TOKENS = {
    "segmenter": ("SAM2",),
    "detector": ("GROUNDING_DINO", "GDINO", "DINO"),
    "depth": ("DEPTH",),
}
#: role -> the key it appears under in measure_est_drift's `estimators.version`, which the adapter
#: builds as ``det=<id>@<rev>;seg=<id>@<rev>;depth=<id>@<rev>;prompt=...``. `split_pin` recovers
#: the two halves; see `artifact_id`.
ROLE_VERSION_KEY = {"segmenter": "seg", "detector": "det", "depth": "depth"}


def emit(kind, *fields):
    clean = [str(f).replace("\t", " ").replace("\n", " ") for f in fields]
    print("\t".join([kind, *clean]), flush=True)


def call_dotted_name(node):
    parts, func = [], node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def literal_string(node, known=None):
    """(value, source) for a string constant or an ``os.environ.get`` with a string default.

    ``known`` is what the module has defined above this line, because the adapter's shape is
    ``X = os.environ.get("WAM_...", X_DEFAULT)`` -- the default is a NAME, and refusing to follow
    one name would report the single source of truth as unreadable while it sits there readable.
    """
    known = known or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "literal"
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id][0], node.id
    if isinstance(node, ast.Call) and node.args:
        dotted = call_dotted_name(node)
        if dotted in ("os.environ.get", "environ.get", "os.getenv", "getenv"):
            var = node.args[0]
            if isinstance(var, ast.Constant) and isinstance(var.value, str):
                if var.value in os.environ:
                    return os.environ[var.value], "${} in this job's environment".format(var.value)
                if len(node.args) > 1:
                    inner = literal_string(node.args[1], known)
                    if inner is not None:
                        return inner[0], "default of os.environ.get({!r})".format(var.value)
    return None


def module_strings(tree):
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id] if node.value is not None else []
        else:
            continue
        if not targets:
            continue
        got = literal_string(node.value, found)
        if got is None:
            continue
        for name in targets:
            found[name] = got
    return found


def main(argv):
    adapter = Path(argv[1])
    artifacts = [Path(a) for a in argv[2:]]
    wants = [w.split("\t") for w in os.environ.get("PR08_WANT", "").split("\x1f") if w]
    if not wants:
        emit("FATAL", "PR08_WANT is empty: this checker was handed nothing to check, which is not "
                      "the same as everything agreeing.")
        return 2

    if not adapter.is_file():
        emit("FATAL",
             "no estimator adapter at {}. The ids staged here are only right if they equal the "
             "ones measure_geom_tol.py and measure_est_drift.py load, and that file is where they "
             "are declared. Point ESTIMATOR_ADAPTER at the checkout, or check the repo out; this "
             "job will not stage several GB against a contract it has not read.".format(adapter))
        return 2
    try:
        tree = ast.parse(adapter.read_text(encoding="utf-8"), filename=str(adapter))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        emit("FATAL", "cannot read {} ({}: {}). Unreadable is not 'agrees'.".format(
            adapter, type(exc).__name__, exc))
        return 2
    strings = module_strings(tree)
    emit("INFO", "adapter    {} ({} module-level string constants)".format(adapter, len(strings)))

    failed = False
    for want in wants:
        if len(want) != 3:
            emit("FATAL", "malformed PR08_WANT record {!r}".format(want))
            failed = True
            continue
        role, repo, revision = want

        # -- the id, against the adapter ----------------------------------------------------------
        const = ROLE_ID_CONSTANT.get(role)
        if const is None:
            emit("FATAL", "unknown role {!r}: this checker has no constant to compare it against, "
                          "and inventing one is how the wrong file gets blessed.".format(role))
            failed = True
            continue
        got = strings.get(const)
        if got is None:
            emit("FATAL",
                 "{} declares no {} this checker can read. It is the single source of truth for "
                 "the {} id; if it moved or changed shape, move this job with it rather than "
                 "staging {!r} on the strength of a comment.".format(adapter, const, role, repo))
            failed = True
            continue
        value, source = got
        if value != repo:
            emit("FATAL",
                 "{} DISAGREES: {} is {!r} ({}), this job stages {!r}. One of the two is what the "
                 "measurement will load and the other is what this job would have downloaded; "
                 "nothing downstream notices the difference, it just reports a number for the "
                 "wrong estimator."
                 .format(adapter.name, const, value, source, repo))
            failed = True
            continue
        emit("CHECK", "{}-id-vs-adapter".format(role), "agrees", "{}={}".format(const, value))
        emit("INFO", "verified   {} == {}  ({}, {})".format(const, value, adapter.name, source))

        # -- the revision, against the adapter ----------------------------------------------------
        tokens = ROLE_REVISION_TOKENS[role]
        candidates = {
            name: v for name, (v, _src) in strings.items()
            if "REVISION" in name and any(tok in name for tok in tokens) and SHA_RE.match(v)
        }
        if not candidates:
            emit("CHECK", "{}-revision-vs-adapter".format(role), "unverified", "")
            emit("UNVERIFIED",
                 "{}: {} declares no 40-hex revision constant for this role, so the adapter can "
                 "still resolve the id at an unpinned revision while this job staged {}. The bytes "
                 "on disk are pinned; the loader's choice of them is not established."
                 .format(role, adapter.name, revision))
        else:
            bad = {n: v for n, v in candidates.items() if v != revision}
            if bad:
                emit("FATAL",
                     "{} DISAGREES on the {} revision: {}, this job stages {}. The id matching is "
                     "not enough -- two revisions of one repo are two sets of weights."
                     .format(adapter.name, role,
                             ", ".join("{}={}".format(n, v) for n, v in sorted(bad.items())),
                             revision))
                failed = True
                continue
            emit("CHECK", "{}-revision-vs-adapter".format(role), "agrees",
                 ",".join(sorted(candidates)))
            emit("INFO", "verified   {} revision {} ({})".format(
                role, revision, ",".join(sorted(candidates))))

        # -- the id, against whichever measurement artifacts are committed ------------------------
        for art in artifacts:
            label = "{}-vs-{}".format(role, art.name)
            if not art.is_file():
                emit("CHECK", label, "unverified", "not committed")
                emit("UNVERIFIED",
                     "{}: {} is not committed yet, so the id the measurement actually recorded is "
                     "unknown to this job. Rerun it after the measurement lands; do not assume."
                     .format(role, art.name))
                continue
            try:
                doc = json.loads(art.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                emit("CHECK", label, "unverified", "unreadable")
                emit("UNVERIFIED",
                     "{}: {} exists but did not parse as JSON ({}: {}), so no field was compared."
                     .format(role, art.name, type(exc).__name__, exc))
                continue
            recorded, recorded_rev, where = artifact_id(doc, art.name, role)
            if recorded is None:
                emit("CHECK", label, "unverified", where)
                emit("UNVERIFIED",
                     "{}: {} records no {} id in the shape this reads ({}). The artifact changed "
                     "shape or the measurement ran with an adapter that declares nothing; either "
                     "way this job compared nothing."
                     .format(role, art.name, role, where))
                continue
            if recorded != repo:
                emit("FATAL",
                     "{} DISAGREES: it records the {} as {!r} (at {}), this job stages {!r}. The "
                     "committed number therefore describes an estimator this staging run did not "
                     "produce.".format(art.name, role, recorded, where, repo))
                failed = True
                continue
            if recorded_rev is not None and recorded_rev != revision:
                emit("FATAL",
                     "{} DISAGREES ON THE COMMIT: it records the {} at {!r} (at {}), this job "
                     "stages {!r}. Same repo, different weights -- which is the drift this check "
                     "exists to catch, and it is invisible in the id alone."
                     .format(art.name, role, recorded_rev, where, revision))
                failed = True
                continue
            emit("CHECK", label, "agrees", where)
            emit("INFO", "verified   {} id == {}{}  ({} {})".format(
                role, repo, "" if recorded_rev is None else "@" + recorded_rev, art.name, where))

    return 2 if failed else 0


def artifact_id(doc, filename, role):
    """The repo id a committed measurement artifact RECORDS for this role, and where it was read.

    Returns ``(repo_id, revision_or_None, where)``.

    Two shapes, because the two rigs record it differently and neither is a free-text mention:
    measure_geom_tol writes the adapter's declared checkpoints into
    ``mask_method.params.checkpoints`` keyed by the constant name; measure_est_drift writes only
    ``estimators.version``. A whole-file substring search would accept an id that appears as a
    *candidate* or in a refusal message, so it is not used.

    BOTH SHAPES CARRY ``id@revision`` NOW, and this used to assume a bare id. apple_sam2 pins the
    commit alongside the repo (AC-04: the committed gate number has to identify the exact weights),
    so ``seg=facebook/sam2-hiera-large@e6a8e880...`` is what lands in ``estimators.version``. The
    caller compared that whole string against the bare ``SAM2_MODEL_ID`` and would have emitted
    FATAL "DISAGREES" on a correctly configured run -- a refusal of the right answer, which is the
    one failure worse than a missed check here because it teaches the operator to skip the job.
    So the split happens once, in one place, and the revision it recovers is CHECKED rather than
    discarded: the artifact naming the right repo at the wrong commit is exactly the drift this
    job exists to catch.
    """
    if "geom_tol" in filename:
        where = "mask_method.params.checkpoints." + ROLE_ID_CONSTANT[role]
        params = (doc.get("mask_method") or {}).get("params") or {}
        checkpoints = params.get("checkpoints") or {}
        if not isinstance(checkpoints, dict):
            return None, None, where
        value = checkpoints.get(ROLE_ID_CONSTANT[role])
        if not (isinstance(value, str) and value):
            # The role key is the other shape the adapter declares (ESTIMATOR_CHECKPOINTS), and it
            # carries id@revision. Reading it too means a future adapter that stops exporting the
            # bare *_MODEL_CHECKPOINT attributes does not silently become "unverified".
            value = checkpoints.get(role)
        if not (isinstance(value, str) and value):
            return None, None, where
        return split_pin(value) + (where,)
    if "est_drift" in filename:
        key = ROLE_VERSION_KEY[role]
        where = "estimators.version[{}=]".format(key)
        version = (doc.get("estimators") or {}).get("version")
        if not isinstance(version, str):
            return None, None, where
        for part in version.split(";"):
            name, sep, value = part.partition("=")
            if sep and name.strip() == key and value.strip():
                return split_pin(value.strip()) + (where,)
        return None, None, where
    return None, None, "unknown artifact shape"


def split_pin(value):
    """``"repo/name@sha"`` -> ``("repo/name", "sha")``; a bare id -> ``(id, None)``.

    Split on the LAST ``@``: a Hugging Face repo id cannot contain one, but being explicit costs
    nothing and a right-split is correct for both shapes.
    """
    repo_id, sep, revision = value.rpartition("@")
    if not sep:
        return value, None
    return repo_id, (revision or None)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY_CHECK_IDS

join_us() { local IFS=$'\x1f'; printf '%s' "$*"; }
CHECK_RC=0
PR08_WANT=$(join_us \
  "segmenter"$'\t'"${SAM2_MODEL_ID}"$'\t'"${SAM2_MODEL_REVISION}" \
  "detector"$'\t'"${GDINO_MODEL_ID}"$'\t'"${GDINO_MODEL_REVISION}" \
  "depth"$'\t'"${DEPTH_MODEL_ID}"$'\t'"${DEPTH_MODEL_REVISION}") \
  python3 "${CHECK_IDS_PY}" "${ESTIMATOR_ADAPTER}" \
    "${REPO_ROOT}/configs/transfer25/pr08_geom_tol.json" \
    "${REPO_ROOT}/configs/transfer25/pr08_est_drift.json" \
    > "${CHECK_IDS_OUT}" || CHECK_RC=$?

while IFS=$'\t' read -r kind f1 f2 f3; do
  case "${kind}" in
    INFO)       echo "=== ${f1}" ;;
    CHECK)      CHECKS+=("${f1}"$'\t'"${f2}"$'\t'"${f3:-}") ;;
    UNVERIFIED) UNVERIFIED+=("${f1}"); echo "=== UNVERIFIED ${f1}" ;;
    FATAL)      echo "FATAL: ${f1}" ;;
    *)          [[ -n "${kind}" ]] && echo "=== ${kind} ${f1:-}" ;;
  esac
done < "${CHECK_IDS_OUT}"

if [[ "${CHECK_RC}" -ne 0 ]]; then
  echo "       Nothing was downloaded. Fix it in ONE place: scripts/estimators/apple_sam2.py is"
  echo "       the source of truth for the ids and their revisions, and the values above are this"
  echo "       job's restatement of them. Change whichever of the two is wrong, then resubmit."
  exit 1
fi

# -- reachability and the expected file list, before a single byte ---------------------------------
# The API's revision endpoint rather than a HEAD on README.md: a repo with no README at that commit
# answers 404 to the HEAD, and the old FATAL text then asserted "the repo or the revision does not
# exist" on evidence that established no such thing. This endpoint answers exactly the question --
# does this commit exist and can this token see it -- and hands back the file list at the same time,
# which is what turns "the download exited 0" into "the download is complete".
FILES_EXPECTED=()
probe_repo() {
  local role=$1 repo=$2 rev=$3 body list code
  body="${TMPDIR}/pr08-meta-${role}.json"
  list="${TMPDIR}/pr08-files-${role}.txt"
  code=$(curl -s -o "${body}" -w '%{http_code}' -L \
           -H "Authorization: Bearer ${HF_TOKEN}" \
           "https://huggingface.co/api/models/${repo}/revision/${rev}") || code=000
  echo "=== probe      ${repo}@${rev:0:8} -> HTTP ${code}"
  case "${code}" in
    200) : ;;
    401|403)
      echo "FATAL: HTTP ${code} on ${repo}. This account cannot read it at ${rev}."
      echo "       It was ungated on 2026-08-22, so either the repo acquired a licence gate (accept"
      echo "       it in a browser at https://huggingface.co/${repo} -- an account holder's act,"
      echo "       never an agent's) or this token belongs to a different account than you think."
      exit 1 ;;
    404)
      echo "FATAL: HTTP 404 from the model API for ${repo} at revision ${rev}. The repo does not"
      echo "       exist, or that commit is not on it. If the id was overridden, the revision has"
      echo "       to be overridden with it."
      exit 1 ;;
    000)
      echo "FATAL: curl could not reach huggingface.co for ${repo}. This is a transport failure,"
      echo "       not a gate and not a missing repo. Check the network and resubmit."
      exit 1 ;;
    *)
      echo "FATAL: unexpected HTTP ${code} probing ${repo}. NOT treating this as a gate failure --"
      echo "       check the network before rerunning, rather than working around the probe."
      exit 1 ;;
  esac
  python3 - "${body}" "${rev}" "${list}" <<'PY_FILELIST'
import json, sys
from pathlib import Path

meta_path, rev, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
except ValueError as exc:
    sys.exit("FATAL: the model API answered 200 with a body this job cannot parse ({}). Not "
             "guessing at the file list -- a completeness check nobody can run is worse than "
             "none, because it reports a clean run.".format(exc))
sha = meta.get("sha")
if sha != rev:
    sys.exit("FATAL: asked the API for revision {}, it answered with sha {!r}. A pin that resolves "
             "to something else is not a pin.".format(rev, sha))
siblings = meta.get("siblings")
if not isinstance(siblings, list) or not siblings:
    sys.exit("FATAL: the model API listed no files for this revision. Either the API changed shape "
             "or the revision is empty; neither is a repo worth staging.")
names = sorted({s["rfilename"] for s in siblings
                if isinstance(s, dict) and isinstance(s.get("rfilename"), str) and s["rfilename"]})
if len(names) != len(siblings):
    sys.exit("FATAL: {} of {} entries in the API's file list carry no usable rfilename. The "
             "completeness check would be comparing against a list this job does not understand."
             .format(len(siblings) - len(names), len(siblings)))
out.write_text("\n".join(names) + "\n", encoding="utf-8")
print("=== expects    {} files at this revision".format(len(names)))
PY_FILELIST
  FILES_EXPECTED+=("${list}")
}
probe_repo segmenter "${SAM2_MODEL_ID}"  "${SAM2_MODEL_REVISION}"
probe_repo detector  "${GDINO_MODEL_ID}" "${GDINO_MODEL_REVISION}"
probe_repo depth     "${DEPTH_MODEL_ID}" "${DEPTH_MODEL_REVISION}"

# -- the download ------------------------------------------------------------------------------------
# Into the HUB CACHE, not a --local-dir. 99 stages to a local dir because upstream takes a
# --checkpoint-path and reads bytes from it; these three are loaded by REPO ID
# (`from_pretrained("facebook/sam2-hiera-large")`) and a repo id only resolves offline out of a hub
# cache in the blobs/snapshots/refs layout. Writing that layout by hand would be a guess about
# huggingface_hub's internals, so the client writes it.
stage_repo() {
  local role=$1 repo=$2 rev=$3 list=$4 out snap resolved
  echo
  echo "=== fetching   ${role}: ${repo} @ ${rev}"
  # --quiet, AND the path is recovered by pattern rather than by position. THIS EXACT DEFECT HAS
  # ALREADY COST THIS PROJECT A QUEUE CYCLE ONCE: 100_fetch_pr08_source failed on 2026-08-20 with
  # the download SUCCEEDED, because hf >= 1.28 decorates stdout -- a green "✓ Downloaded" line and
  # then an INDENTED "  path: <dir>" -- so `tail -n 1` returns "  path: …", which is not a
  # directory, and the guard correctly rejected a good fetch. It happened again here on 2026-08-22
  # as job 189452, 12 seconds in, after all nine files had landed. `--quiet` is documented as
  # printing one path per line, and the sed strips the decoration if a future client ignores that;
  # taking the last line that IS a directory is what makes this independent of the layout.
  out=$(uvx hf@latest download "${repo}" --revision "${rev}" --quiet)
  snap=$(printf '%s\n' "${out}" \
    | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/^[[:space:]]*path:[[:space:]]*//' -e 's/[[:space:]]*$//' \
    | while IFS= read -r line; do [[ -d "${line}" ]] && printf '%s\n' "${line}"; done \
    | tail -n 1)
  if [[ ! -d "${snap}" ]]; then
    echo "FATAL: ${repo}: 'hf download' did not name a directory. Last stdout line was:"
    echo "         ${snap}"
    echo "       Full stdout:"
    printf '%s\n' "${out}" | sed 's/^/         /'
    exit 1
  fi
  # The snapshot directory is NAMED for the commit the hub actually resolved, so this reads the
  # revision back off the filesystem instead of trusting the one we asked for. They can only differ
  # if the client redirected, and a silent redirect is exactly the thing the pin exists to prevent.
  resolved=$(basename "${snap}")
  if [[ "${resolved}" != "${rev}" ]]; then
    echo "FATAL: ${repo}: asked for ${rev}, the hub resolved ${resolved}."
    exit 1
  fi
  echo "=== staged     ${snap}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${role}" "${repo}" "${rev}" "${snap}" "${list}" >> "${STAGED_LIST}"
}
stage_repo segmenter "${SAM2_MODEL_ID}"  "${SAM2_MODEL_REVISION}" "${FILES_EXPECTED[0]}"
stage_repo detector  "${GDINO_MODEL_ID}" "${GDINO_MODEL_REVISION}" "${FILES_EXPECTED[1]}"
stage_repo depth     "${DEPTH_MODEL_ID}" "${DEPTH_MODEL_REVISION}" "${FILES_EXPECTED[2]}"

# The two lists cross the shell/python boundary as \x1f-joined environment strings rather than as
# arguments, because the unverified reasons are sentences containing spaces, quotes and em dashes:
# passed as words they would be split, and a truncated reason is a record that no longer explains
# why the run is not qualified. \x1f cannot occur in any of them.
PR08_UNVERIFIED=$(join_us ${UNVERIFIED[@]+"${UNVERIFIED[@]}"})
PR08_CHECKS=$(join_us ${CHECKS[@]+"${CHECKS[@]}"})
export PR08_UNVERIFIED PR08_CHECKS REPO_ROOT

# -- the manifest, read back off the bytes rather than asserted ------------------------------------------
# Modelled on 99b's MULTIBRANCH_STAGED.json: every file that landed, with the size and sha256 read
# from the file itself. Symlinks are resolved first, because the hub cache's snapshot entries point
# at blobs/ and hashing the link target is the only way the number describes the bytes a loader will
# read. This is what lets a later run say WHICH weights produced GEOM_TOL rather than which repo it
# meant to use.
python3 - "${MANIFEST}" "${STAGED_LIST}" <<'PY_MANIFEST'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

manifest = Path(sys.argv[1])
rows = [ln.split("\t") for ln in Path(sys.argv[2]).read_text().splitlines() if ln.strip()]
if len(rows) != 3:
    sys.exit(f"FATAL: expected 3 staged repos, the staging list has {len(rows)}. Nothing is guessed.")

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()

records, grand_total = [], 0
for role, repo, revision, snap, expected_list in rows:
    root = Path(snap)
    files, total, broken = [], 0, []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        # A DANGLING ENTRY IS A FAILURE, NOT A SHORTER MANIFEST. The hub cache's snapshot entries
        # are symlinks into blobs/; a 429 or a killed client leaves entries whose blob never
        # arrived. Skipping them silently -- which this loop used to do with `if not p.is_file():
        # continue` -- deletes the evidence of the one failure the header names, and the run
        # reports success over a snapshot a loader will then treat as a cache hit.
        if p.is_symlink() and not p.exists():
            broken.append(f"{rel} -> {os.readlink(p)} (dangling: the blob never landed)")
            continue
        if p.is_dir():
            continue
        if not p.is_file():
            broken.append(f"{rel} (not a regular file: {p.stat().st_mode:#o})")
            continue
        try:
            real = p.resolve()
            size = real.stat().st_size
            digest = sha256_of(real)
        except OSError as exc:
            broken.append(f"{rel} ({type(exc).__name__}: {exc})")
            continue
        total += size
        files.append({
            "path": rel,
            # The blob the snapshot entry points at. Recorded because two snapshots of the same repo
            # share blobs, and "the file is still there" is a question about the blob, not the link.
            "blob": str(real),
            "bytes": size,
            "sha256": digest,
        })
        print(f"    {size/1e6:9.1f} MB  {digest[:16]}  {role}/{rel}", flush=True)
    if broken:
        listed = "\n         ".join(broken)
        sys.exit(f"FATAL: {repo}: {len(broken)} entr{'y' if len(broken) == 1 else 'ies'} under "
                 f"{root} could not be read:\n         {listed}\n"
                 f"       This is what a partial fetch looks like. Delete the snapshot and rerun; "
                 f"do not write a manifest that omits them, because the omission is invisible.")
    if not files:
        sys.exit(f"FATAL: {repo}: nothing under {root}. The download reported success and left no files.")

    # COMPLETENESS, against the file list the hub itself gave for this revision (read by the probe,
    # before the download). `hf download` exiting 0 is not the same claim: job 100's 429 came back
    # through a client that had already written part of the tree.
    expected = sorted(x for x in Path(expected_list).read_text(encoding="utf-8").splitlines() if x)
    staged = {f["path"] for f in files}
    missing = [x for x in expected if x not in staged]
    if missing:
        listed = "\n         ".join(missing[:40])
        more = "" if len(missing) <= 40 else f"\n         ... and {len(missing) - 40} more"
        sys.exit(f"FATAL: {repo} @ {revision}: the hub lists {len(expected)} files at this "
                 f"revision, {len(missing)} of them are not in {root}:\n         {listed}{more}\n"
                 f"       A snapshot missing files is a cache hit for every later loader. Rerun.")

    grand_total += total
    records.append({
        "role": role,
        "repo_id": repo,
        "revision_requested": revision,
        # Read off the snapshot directory name, which the hub writes from the commit it resolved.
        "revision_resolved": root.name,
        "snapshot_path": str(root),
        "total_bytes": total,
        "n_files": len(files),
        "n_files_expected": len(expected),
        "files": files,
    })

unverified = [x for x in os.environ.get("PR08_UNVERIFIED", "").split("\x1f") if x]
checks = [dict(zip(("check", "status", "detail"), c.split("\t")))
          for c in os.environ.get("PR08_CHECKS", "").split("\x1f") if c]

# OPT-IN, and derived from the checks THEMSELVES rather than from a parallel list. The reasons and
# the check rows are written by the same code today, but a later check that appends a row and
# forgets the reason would otherwise flip this field to true -- in the one field the rule says must
# never default to true. No checks at all is also false: nothing checked is not everything agreed.
qualified = bool(checks) and all(c["status"] == "agrees" for c in checks) and not unverified

manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({
    "schema": "wam.pr08_estimators_staged/1",
    "purpose": (
        "Estimator weights for PR-08 §8 item 4: GEOM_TOL (scripts/measure_geom_tol.py) and "
        "EST_DRIFT_P95 (scripts/measure_est_drift.py). §4 step 2 requires ONE segmenter for both; "
        "§4 step 3 requires depth in METRES, which is why the depth head is the metric one."
    ),
    "staged_by": "cluster/discoverer/102_stage_sam2_weights.sbatch",
    "staged_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "repo_commit": (Path(os.environ.get("REPO_ROOT", "")) / "GIT_COMMIT").read_text().strip()
                   if (Path(os.environ.get("REPO_ROOT", "")) / "GIT_COMMIT").is_file() else "",
    "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
    "total_bytes": grand_total,
    "checkpoints": records,
    "id_checks": checks,
    "staged_qualified": qualified,
    "unverified": unverified,
    "note": (
        "The id staged here and the id the measurement wiring loads must be the same string. If "
        "they differ nothing crashes: the wiring fetches its own checkpoint at an unpinned "
        "revision and the committed number describes an estimator the run does not use."
    ),
}, indent=2) + "\n")

print(f"\n    {sum(len(r['files']) for r in records)} files, {grand_total/1e9:.2f} GB -> {manifest}")
PY_MANIFEST

echo
# EXPECTED-ABSENT vs GENUINELY-UNVERIFIABLE, and why the difference is worth code.
#
# The measurement artifacts this job wants to check its ids against are PRODUCED BY MEASUREMENTS
# THAT NEED THESE WEIGHTS. On a first staging run they cannot exist, so the original rule -- any
# unverified reason means exit 3 -- made a correct first run unable to exit 0 BY CONSTRUCTION.
# Job 189453 is the proof: 4.8 GB staged, manifest written, every id and revision agreed against
# the adapter, and Slurm recorded FAILED.
#
# That is worse than untidy. An exit code that is always the same carries no information, and a
# job that reports failure for doing exactly the right thing is how an operator learns to stop
# reading its exit code -- the same reasoning that made the id-vs-artifact comparison FATAL rather
# than advisory a few hundred lines up.
#
# So absence of a not-yet-written artifact is a LIFECYCLE STATE and exits 0 with the reason
# recorded; anything else -- an artifact that exists and did not parse, that records no id in the
# shape this reads, a missing framework checkout, an adapter with no revision constant -- is a real
# gap in the verification and still exits 3. `staged_qualified` stays FALSE in both cases, because
# it says the ids were verified and in neither case were they.
EXPECTED_ABSENT=0
GENUINE_GAPS=()
for reason in ${UNVERIFIED[@]+"${UNVERIFIED[@]}"}; do
  if [[ "${reason}" == *"is not committed yet"* ]]; then
    EXPECTED_ABSENT=$((EXPECTED_ABSENT + 1))
  else
    GENUINE_GAPS+=("${reason}")
  fi
done

if [[ ${#UNVERIFIED[@]} -eq 0 ]]; then
  echo "=== all three checkpoints staged, and every id this job can check agreed."
  echo "===   GEOM_TOL and EST_DRIFT_P95 can now be measured with the SAME segmenter, which is"
  echo "===   what PR-08 §4 step 2 requires and what §6 subtracts."
  exit 0
fi
if [[ ${#GENUINE_GAPS[@]} -eq 0 ]]; then
  echo "=== STAGED — exit 0. The bytes are on disk and the manifest describes them."
  echo "===   ${EXPECTED_ABSENT} id check(s) could not run because the measurement artifact does not"
  echo "===   exist yet. That is this job's own precondition, not a fault: the measurements that"
  echo "===   write those artifacts need the weights this job just staged. Rerun AFTER they land"
  echo "===   and the same checks become real."
  echo "===   staged_qualified is FALSE in the manifest regardless — the ids are not yet verified."
  exit 0
fi
echo "=== STAGED, NOT VERIFIED — exit 3. The bytes are on disk and the manifest describes them."
echo "===   What is NOT established is that they are the ids the measurements will load:"
for reason in "${UNVERIFIED[@]}"; do
  echo "===     - ${reason}"
done
echo "===   This is not a failed download. It is a refusal to record agreement that was never"
echo "===   checked, because an id mismatch here does not crash — it produces a committed number"
echo "===   for an estimator the generation run never uses."
exit 3
