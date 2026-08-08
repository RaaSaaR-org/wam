#!/bin/bash
# Step 1 of the workstation pipeline — fetch PR-09 §2's corpus at source resolution.
#
#   T041_FREEZE_LIFTED="<reason>" workstation/10_fetch_corpus.sh
#
# ONE CAMERA PER SOURCE, NOT ALL OF THEM. The cluster run pulled `videos/**` and spent 69 GB on 2–4
# cameras per repo when the corpus uses exactly one. The camera is known before the download starts
# — it is column 2 of configs/cosmos3/corpus_g1_embodiment.tsv — so the include pattern can name it
# and the wrist cameras never cross the network. Roughly half the bytes, and the half we keep is
# the half we train on.
#
# The pattern differs by format, which is why metadata comes down first:
#   v2.1  videos/chunk-NNN/<camera>/episode_NNNNNN.mp4   -> videos/*/<camera>/**
#   v3.0  videos/<camera>/chunk-NNN/file-NNN.mp4         -> videos/<camera>/**
# Guessing one and hoping is how you get a 4-second job and an empty directory.
#
# SOURCE RESOLUTION. datasets/gr00t-apple-full is 120x160 — convert_lerobot_g1.py threw the pixels
# away because a policy does not need them. A generator trains ON pixels, so the 640x480 originals
# are the only usable copy.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_freeze_lifted
need hf "pip install -U 'huggingface_hub[cli]'"
need python3 ""

read_corpus
mkdir -p "${RAW}"
echo "=== ${#REPOS[@]} sources -> ${RAW}"

for i in "${!REPOS[@]}"; do
  repo=${REPOS[$i]}
  cam=${CAMS[$i]}
  dest=${RAW}/$(basename "${repo}")

  # THE GUARD IS A COMPLETION SENTINEL, NOT meta/info.json. `hf download` fetches metadata before
  # videos, so a run killed partway leaves a perfectly valid info.json next to an incomplete video
  # tree. Keying the skip on that file makes a retry declare the repo "present" and hand the next
  # step a partial set it cannot detect: the episode list names episodes whose mp4s are missing,
  # the manifest counts only what it found, and the run trains on a silently truncated corpus that
  # looks clean in every artifact. This file is written only after hf exits 0.
  if [[ -f "${dest}/.download-complete" ]]; then
    echo "=== ${repo} complete"
    continue
  fi

  echo "=== ${repo}: metadata"
  # ONE --include PER PATTERN. The hf CLI is click-based, so `--include a b` binds only `a` and
  # leaves `b` as a positional filename, which silently switches the command into explicit-file
  # mode and then 404s on a literal file named `videos/**`. That killed job 186283.
  hf download --repo-type dataset "${repo}" --include 'meta/**' --local-dir "${dest}"

  version=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("codebase_version","unknown"))' \
            "${dest}/meta/info.json")
  case "${version}" in
    v2.1) pattern="videos/*/${cam}/**" ;;
    v3.0) pattern="videos/${cam}/**" ;;
    *) echo "FATAL: ${repo} is LeRobot ${version}; prepare_cosmos_corpus.py reads v2.1 and v3.0."; exit 1 ;;
  esac

  # Fail before the download rather than after: an unknown camera yields an include pattern that
  # matches nothing, and `hf download` treats "no files matched" as success.
  python3 - "${dest}/meta/info.json" "${cam}" "${repo}" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
cam, repo = sys.argv[2], sys.argv[3]
keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
if cam not in keys:
    sys.exit(f"FATAL: {repo} has no camera {cam!r}. Available: {keys}\n"
             f"       Fix column 2 of the corpus tsv — an unmatched include pattern downloads\n"
             f"       nothing and reports success.")
PY

  echo "=== ${repo}: videos (${version}, camera ${cam})"
  hf download --repo-type dataset "${repo}" --include "${pattern}" --local-dir "${dest}"
  # set -e means we only reach this line when hf exited 0.
  touch "${dest}/.download-complete"
done

echo
echo "FETCHED OK -> ${RAW}"
du -sh "${RAW}" 2>/dev/null || true
echo "next: workstation/20_prepare_corpus.sh"
