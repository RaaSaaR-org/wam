#!/usr/bin/env bash
# Launch the interactive MuJoCo viewer (scripts/view_sim.py) on macOS.
#
# Why a wrapper: the viewer needs `mjpython` (the main thread must run the native event
# loop), and `mjpython` loads the interpreter with dlopen — so the @rpath lookup uses
# mjpython's own rpaths, not the Python binary's. A uv-managed CPython keeps its
# libpython3.x.dylib in the uv toolchain directory, which is on none of those paths, and
# mjpython dies with "Library not loaded: @rpath/libpython3.12.dylib" before Python starts.
# Nothing inside view_sim.py can fix that, so we export the directory here.
#
# Usage: scripts/view_sim.sh [--fast] [--amplitude-rad 0.2] [--max-cycles 400] ...
# All arguments are forwarded to scripts/view_sim.py verbatim.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
MJPYTHON="$REPO_ROOT/.venv/bin/mjpython"

if [[ ! -x "$MJPYTHON" ]]; then
  echo "mjpython not found at $MJPYTHON — install the sim extra: uv pip install mujoco" >&2
  exit 1
fi

LIBDIR="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [[ -n "$LIBDIR" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$LIBDIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

exec "$MJPYTHON" "$REPO_ROOT/scripts/view_sim.py" "$@"
