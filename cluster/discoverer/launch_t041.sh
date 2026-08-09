#!/usr/bin/env bash
# Launch the T-041 Cosmos3-Super chain on Discoverer+. Runs on the Mac, not on the cluster.
#
#   ./cluster/discoverer/launch_t041.sh             # sync, then submit 90 -> 91
#   ./cluster/discoverer/launch_t041.sh --status    # show the queue and what has landed
#   ./cluster/discoverer/launch_t041.sh --corpus    # ship the captioned corpus, then register it
#   ./cluster/discoverer/launch_t041.sh --probe     # the 8-GPU gate, after 91 and --corpus
#
# Everything this runs on the login node is sbatch, squeue, and file management — exactly
# docs/discoverer.md §2's permitted set ("checking the allocation, and managing files under
# project storage"; "Everything else runs as a Slurm job"). Nothing computes there. The heavy
# work is inside the jobs, which is the whole point of the machine.
#
# WHAT THIS DELIBERATELY DOES NOT DO: submit 94 for real, or 95. The probe is a gate, not a
# warm-up — PR-09 §7 — and a human reads PROBE.json before eight H200s are committed. 95 needs
# the calibration clips a human has to choose. Both are one sbatch each, printed when due.
set -euo pipefail

HOST=ffromm@login-plus.discoverer.bg
PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
JOBS=${PROJ}/wam/cluster/discoverer
KEY=${HOME}/.ssh/id_ed25519_eu_ai_hub
# Published in docs/discoverer.md §1. Checked, not assumed: a wrong key here does not fail
# clean — sshd resolves authorized keys from a 389 Directory Server, so the denial looks
# identical to a locked agent, and there is no self-service way to rotate a key back.
FPR=SHA256:1WiG4/oXoh0I94LAAx49evKgazZZdpE/tE3v3AEZLn0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID=${RUN_ID:-t041-super-lora}

# PR-07 §7 freezes Cosmos3-Super generation. OD-10 lifts that one clause and nothing else.
# Every job refuses to start without this string and writes it verbatim into run_metadata.json,
# so no artifact can exist without carrying the decision that allowed it.
export T041_FREEZE_LIFTED=${T041_FREEZE_LIFTED:-"OD-10, 2026-08-07: user lifted PR-07 §7 Cosmos3-Super clause"}

need_key() {
  [[ -f "${KEY}.pub" ]] || { echo "FATAL: ${KEY}.pub missing."; exit 1; }
  local got
  got=$(ssh-keygen -lf "${KEY}.pub" | awk '{print $2}')
  [[ "${got}" == "${FPR}" ]] || {
    echo "FATAL: ${KEY} is not the Discoverer key."
    echo "       want ${FPR}"
    echo "       got  ${got}"; exit 1; }
  # ssh-add -l exits 1 when the agent is empty; that is a state, not an error.
  if ! ssh-add -l 2>/dev/null | grep -q "${FPR}"; then
    # ssh-add reads the passphrase from /dev/tty, not stdin. Under a harness that captures
    # output there is no controlling terminal, so it blocks forever on a prompt nobody can
    # answer. Say so and stop, rather than hang looking like a slow network.
    if [[ ! -t 0 ]] || [[ ! -r /dev/tty ]]; then
      echo "FATAL: ${KEY} is not in the agent, and this shell has no terminal to prompt on."
      echo "       Unlock it once in a normal terminal window, then re-run this script:"
      echo
      echo "         ssh-add --apple-use-keychain ${KEY}"
      echo
      echo "       --apple-use-keychain stores it in the login keychain, so it survives reboots"
      echo "       and this is the last time you type it. Plain 'ssh-add' works too, per login."
      exit 1
    fi
    echo "==> unlocking ${KEY} (passphrase prompt follows)"
    ssh-add "${KEY}"
  fi
}

# One SSH round trip, all of it read-only: squeue, sinfo, sacct and tailing a log are
# "checking the allocation" under docs/discoverer.md §2. Passed as a heredoc to `bash -s`
# rather than a quoted one-liner so remote $vars do not have to be escaped past two shells.
queue() {
  ssh "${HOST}" bash -s -- "${PROJ}" "${RUN_ID}" <<'REMOTE'
set -u
PROJ=$1; RUN_ID=$2
echo "== queue"
squeue -u ffromm -o '%.10i %.22j %.9T %.10M %R'
# Slurm only estimates a start time once the backfill scheduler has placed the job; a blank
# column means "not scheduled yet", not "never". With OverSubscribe=FORCE:4 on two nodes this
# can move a lot between polls.
echo
echo "== estimated start"
squeue -u ffromm --start -o '%.10i %.22j %.20S %R' 2>/dev/null
echo
echo "== partition"
sinfo -p common -o '%.10P %.6a %.11l %.6D %.6t %N'
echo
# -X = one row per job, not per step. Catches a job that already failed and left the queue,
# which is otherwise invisible: squeue only shows what is still pending or running.
echo "== today"
sacct -u ffromm -S today -X -o 'JobID%10,JobName%22,State%20,Elapsed%10,ExitCode%8' 2>/dev/null | tail -20
echo
echo "== artifacts"
# t041-corpus/ was never a path any job wrote — the status probe reported "absent" forever and
# meant nothing by it. 94's actual precondition is the registered jsonl, so check that.
for f in "$PROJ/cosmos_env.sh" "$PROJ/data/cosmos-g1-embodiment/manifest.json" \
         "$PROJ/data/cosmos-g1-embodiment/train/video_dataset_file.jsonl" \
         "$PROJ/runs/$RUN_ID/PROBE.json" "$PROJ/runs/$RUN_ID/DONE"; do
  if [[ -e $f ]]; then echo "  present  $f"; else echo "  absent   $f"; fi
done
# The tail of whichever log was touched last is nearly always the thing worth reading: while a
# job runs it is progress, and after it dies it is the error.
LOG=$(ls -t "$PROJ"/logs/*.out 2>/dev/null | head -1)
if [[ -n ${LOG:-} ]]; then echo; echo "== tail $LOG"; tail -20 "$LOG"; fi
REMOTE
}

case "${1:-}" in
  --status)
    need_key; queue; exit 0 ;;

  --corpus)
    # Corpus preparation and captioning moved to the workstation (PR-09 §4e). This used to submit
    # 92 + 93; both are superseded, and 93 in particular runs on the PAID GPU QoS and re-captions
    # the untranscoded tree — the exact run that produced 372 clips and 0 captions. Submitting it
    # from here would spend an H200 reproducing a known failure.
    need_key
    echo "==> shipping the captioned corpus from this machine (14 GB, resumable)"
    "${ROOT}/cluster/discoverer/sync.sh" --corpus
    echo "==> registering it on the cluster (free QoS, no GPU)"
    ssh "${HOST}" "cd ${JOBS} && sbatch --parsable 92b_register_corpus.sbatch"
    queue; exit 0 ;;

  --probe)
    need_key
    echo "==> submitting the PR-09 §7 gate (8 GPUs, measures per-iteration cost, refuses to"
    echo "    let a run start that would exceed the 96 GPU-h ceiling)"
    ssh "${HOST}" "cd ${JOBS} && export T041_FREEZE_LIFTED='${T041_FREEZE_LIFTED}' PROBE=1 && \
      sbatch --parsable 94_train_t041_cosmos_super.sbatch"
    queue; exit 0 ;;

  "") ;;
  *) echo "usage: $0 [--status|--corpus|--probe]"; exit 1 ;;
esac

need_key
echo "==> syncing the working tree (rsync, not git — uncommitted edits go too, stamped -dirty)"
"${ROOT}/cluster/discoverer/sync.sh"

echo
echo "==> submitting 90 -> 91"
# afterok, not afterany. If the env build fails there is nothing to stage weights into. A
# dependency that is never satisfied costs zero GPU-hours and leaves the job visible in the queue
# as DependencyNeverSatisfied, which is a better signal than a job that starts and dies.
#
# 92 and 93 are NOT submitted here any more. Fetching and captioning the corpus moved to the
# workstation (PR-09 §4e, workstation/20 and workstation/30); the corpus arrives via
# `sync.sh --corpus` and is registered by 92b_register_corpus.sbatch. Leaving 93 in this chain
# meant one `launch_t041.sh` with no arguments spent a paid H200 re-captioning the untranscoded
# tree, and it had been superseded for a day by then.
ssh "${HOST}" "cd ${JOBS} && export T041_FREEZE_LIFTED='${T041_FREEZE_LIFTED}' && \
  J90=\$(sbatch --parsable 90_build_cosmos_env.sbatch) && \
  J91=\$(sbatch --parsable --dependency=afterok:\$J90 91_stage_cosmos_weights.sbatch) && \
  echo \"90=\$J90 (env, free)  91=\$J91 (weights, 1 GPU)\""

echo
queue

cat <<EOF

next:
  $0 --status                     # poll
  $0 --corpus                     # ship + register the workstation corpus (needed by 94)
  $0 --probe                      # once 91 and --corpus are done — the gate before any real run

still needs a human, before job 95 can issue a verdict at all:
  ${PROJ}/data/t041-calibration/positive   10 real held-out G1 + Dex3 clips  (must score YES)
  ${PROJ}/data/t041-calibration/negative   10 real non-G1 manipulator clips  (must score NO)
EOF
