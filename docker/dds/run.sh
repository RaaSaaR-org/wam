#!/usr/bin/env bash
# Build the linux/arm64 DDS conformance image and run the check against the working tree.
#
# Contracts:
# - Idempotent: re-running rebuilds from the layer cache (seconds) unless the Dockerfile or
#   its pinned refs changed.
# - The repo is bind-mounted READ-ONLY at /wam. The check never writes into the repo.
# - Everything happens inside the container: no DDS traffic crosses the Docker-Desktop VM
#   boundary, so the macOS multicast limitation is irrelevant here (see README.md).
# - Exit code is the conformance exit code: 0 = no FAIL, 1 = at least one FAIL.
#
# Usage:  docker/dds/run.sh [extra args for conformance.py]
#         docker/dds/run.sh --interface lo --domain 0
#         NO_CACHE=1 docker/dds/run.sh          # force a clean rebuild
set -euo pipefail

IMAGE="${IMAGE:-wam-dds-conformance:latest}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/../.." && pwd)"

build_args=(--platform linux/arm64 -t "${IMAGE}" "${HERE}")
if [[ -n "${NO_CACHE:-}" ]]; then
  build_args=(--no-cache "${build_args[@]}")
fi

echo "==> building ${IMAGE} (linux/arm64)"
docker build "${build_args[@]}"

echo "==> running conformance check (repo mounted read-only at /wam)"
# --init reaps the fake-peer child process if the check is interrupted.
# No --network flag: the default bridge namespace already provides `lo`, which is all DDS
# discovery needs here.
exec docker run --rm --init \
  --platform linux/arm64 \
  -v "${REPO_ROOT}:/wam:ro" \
  "${IMAGE}" \
  python3 /wam/docker/dds/conformance.py "$@"
