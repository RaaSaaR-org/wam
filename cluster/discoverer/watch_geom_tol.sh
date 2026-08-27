#!/usr/bin/env bash
# One read-only snapshot of the PR-08 GEOM_TOL re-run. Runs on the workstation.
#
#   ./cluster/discoverer/watch_geom_tol.sh
#
# Everything here is `squeue`/`sacct`/`ls`/`tail` on the login node, which is the permitted
# administrative set -- no compute, no install, no long-running process. It takes one snapshot
# and exits; it does not poll in a loop, because a loop is a long-running process and the login
# node's enforcement (CPUQuota 200%, MemoryHigh 4GB) puts offenders in D-state, which looks like
# a hung filesystem rather than an error.
set -euo pipefail

HOST=dplus
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
RUN_ID=${RUN_ID:-pr08-geom-tol-v2}

ssh "${HOST}" "bash -s" <<REMOTE
set -u
P=${PROJ}
R=${RUN_ID}
echo "=== \$(date -u '+%Y-%m-%d %H:%M:%S UTC') · RUN_ID=\${R}"
echo
echo "--- queue (ALL of this user's jobs; a peer session shares the ceiling of 8) ---"
squeue -u \$USER -r -o '%.14i %.20j %.9T %.10M %.10l %R' 2>&1
echo
echo "--- shards landed ---"
if [ -d "\${P}/runs/\${R}/shards" ]; then
  ls -1 "\${P}/runs/\${R}/shards" 2>/dev/null | sort -t- -k2 -n | tr '\n' ' '
  echo
  echo "    \$(ls -1 "\${P}/runs/\${R}/shards"/shard-*.json 2>/dev/null | wc -l) of 16"
else
  echo "    (no shard directory yet)"
fi
echo
echo "--- finished tasks, this run only ---"
sacct -u \$USER -S \$(date -u -d '-8 hours' +%Y-%m-%dT%H:%M) \
  --name=wam-geom-tol -X \
  --format=JobID%16,State%12,ExitCode%8,Elapsed%10,Start%20 2>&1 | head -30
echo
echo "--- tail of the newest geom-tol log ---"
L=\$(ls -t "\${P}/logs"/geom-tol.*.out 2>/dev/null | head -1)
if [ -n "\${L}" ]; then
  echo "    \${L}"
  tail -n 25 "\${L}"
else
  echo "    (none)"
fi
REMOTE
