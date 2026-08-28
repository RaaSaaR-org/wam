#!/usr/bin/env bash
# One read-only snapshot of the PR-08 §8 item 3 throughput measurement. Runs on the workstation.
#
#   ./cluster/discoverer/watch_timing_episode.sh
#   RUN_ID=t040-transfer25-restyle-timing-2026-08-28 ./cluster/discoverer/watch_timing_episode.sh
#
# The sibling of watch_geom_tol.sh, and it keeps that file's discipline for the same reasons.
# Everything here is `squeue`/`sacct`/`ls`/`tail` on the login node, which is the permitted
# administrative set — no compute, no install, no long-running process. It takes ONE snapshot and
# exits; it does not poll in a loop, because a loop is a long-running process and the login node's
# enforcement (CPUQuota 200%, MemoryHigh 4GB) puts offenders in D-state, which looks like a hung
# filesystem rather than an error.
#
# WHAT IT IS WATCHING FOR, which is not "did it finish". `T40_RULE_V20` §5 fixed three outcomes
# before the job was submitted, and the difference between two of them is not visible in an exit
# code — so this prints the evidence each one needs and refuses to collapse them:
#
#   M  THROUGHPUT.json exists with units_succeeded >= 1. Item 3's rate exists.
#   R  the source-mask preflight REFUSED episode_000371 on the H200. Item 3 stays open, the area
#      pass is owed a re-measurement on the machine that will generate, and walking down the 17
#      until one passes is refused by the rule.
#   F  anything else — walltime, cold checkpoints, licence, a crash. Nothing was learned; repeat.
#
# V20 §5, verbatim: "An `F` may not be reported as an `R`." The two are told apart by whether the
# log carries the preflight's own refusal, not by the job's state, which is FAILED either way.
set -euo pipefail

HOST=dplus
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
RUN_ID=${RUN_ID:-t040-transfer25-restyle-timing-$(date -u +%Y-%m-%d)}

ssh "${HOST}" "bash -s" <<REMOTE
set -u
P=${PROJ}
R=${RUN_ID}
echo "=== \$(date -u '+%Y-%m-%d %H:%M:%S UTC') · RUN_ID=\${R}"
echo
echo "--- queue (ALL of this user's jobs; a peer session shares the ceiling of 8) ---"
squeue -u \$USER -r -o '%.14i %.24j %.9T %.10M %.10l %R' 2>&1
echo
echo "--- the artifact item 3 cannot close without ---"
T="\${P}/runs/\${R}/THROUGHPUT.json"
if [ -f "\$T" ]; then
  echo "    \$T"
  python3 -c "
import json
d = json.load(open('\$T'))
for k in ('units_succeeded', 'units_failed', 'seconds_per_frame', 'gpu_seconds_per_frame',
          'frames', 'wall_seconds', 'qualified', 'disqualified_reasons'):
    if k in d:
        print(f'      {k}: {d[k]!r}')
" 2>&1
else
  echo "    (none yet at \$T)"
fi
echo
echo "--- the pre-GPU screen's own verdict, if it got that far ---"
A=\$(ls -1t \${P}/runs/\${R}/chunks/*/timing_unit_admissibility.json 2>/dev/null | head -1)
if [ -n "\$A" ]; then
  echo "    \$A"
  python3 -c "
import json
d = json.load(open('\$A'))
for k in ('episode', 'n_frames', 'empty_frames', 'max_area_fraction',
          'max_frame_fraction_bound', 'evidence', 'evidence_sha256'):
    print(f'      {k}: {d.get(k)!r}')
n = d.get('evidence_shards')
print(f'      evidence_shards: {len(n) if n else None}')
" 2>&1
else
  echo "    (no admissibility record — the screen refused, or the job has not started)"
fi
echo
echo "--- finished tasks, last 12 hours ---"
sacct -u \$USER -S \$(date -u -d '-12 hours' +%Y-%m-%dT%H:%M) \
  --name=t040-restyle -X \
  --format=JobID%16,State%12,ExitCode%8,Elapsed%10,Start%20,End%20 2>&1 | head -20
echo
echo "--- tail of the newest t040-restyle log ---"
L=\$(ls -1t \${P}/logs/t040-restyle.*.out 2>/dev/null | head -1)
if [ -n "\$L" ]; then
  echo "    \$L"
  tail -40 "\$L" 2>&1
  echo
  echo "--- outcome discriminator (V20 §5): R is a PREFLIGHT REFUSAL, F is anything else ---"
  if grep -q "WILL be refused by PR-08 §6 G0c" "\$L" 2>/dev/null \
     || grep -q "preflight_source_masks" "\$L" 2>/dev/null; then
    echo "    the log carries a source-mask refusal — read it before calling this R or F"
  else
    echo "    no source-mask refusal in this log"
  fi
else
  echo "    (no log yet)"
fi
REMOTE
