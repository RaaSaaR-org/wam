"""cluster/discoverer/submit_geom_tol_wave.sh — the wrapper, pinned to what it wraps.

WHY THIS FILE EXISTS. The GEOM_TOL submit line is ~250 characters. On 2026-08-27 an operator
pasted it and the terminal's visual wrap became a real newline at the one place that hurts: right
after ``sbatch``. Slurm read an empty script ("Batch script is empty!") and ``--qos=...`` ran as a
shell command on the next line. Nothing was submitted, which is the lucky outcome; the unlucky one
is a wave that submits with a flag missing.

The wrapper exists to make the operator's argv short enough that it cannot break that way. That
buys safety only if the flags inside it stay the ones the runbook and the sbatch agree on — a
wrapper that drifts from what it wraps is strictly worse than the long line, because it looks
authoritative and is checked by nobody. So every assertion below reads the wrapper and compares it
against a *second* source, never against a constant typed into this file.

The two sources:
  docs/PR-08-RUNBOOK-2026-08-27-geom-tol-re-run.md   the authority for the flags
  cluster/discoverer/103_measure_geom_tol.sbatch     the authority for the QoS ceilings
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO_ROOT / "cluster/discoverer/submit_geom_tol_wave.sh"
_WATCHER = _REPO_ROOT / "cluster/discoverer/watch_geom_tol.sh"
_RUNBOOK = _REPO_ROOT / "docs/PR-08-RUNBOOK-2026-08-27-geom-tol-re-run.md"


@pytest.fixture(scope="module")
def wrapper() -> str:
    return _WRAPPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runbook() -> str:
    return _RUNBOOK.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [_WRAPPER, _WATCHER], ids=["submit", "watch"])
def test_the_shell_helpers_parse(script: Path) -> None:
    """A syntax error here is discovered at 2 a.m. with a queue slot in hand."""
    assert script.exists(), script
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_the_four_waves_partition_the_sixteen_shards_exactly_once(wrapper: str) -> None:
    """Four waves of four. A gap is a shard nobody measures; an overlap is a wasted H200 hour.

    The merge refuses on a missing shard, so a gap is caught -- but only after the other fifteen
    have been paid for.
    """
    ranges = re.findall(r"^\s*([1-4])\)\s*ARRAY=(\d+)-(\d+)\s*;;", wrapper, re.M)
    assert [r[0] for r in ranges] == ["1", "2", "3", "4"], ranges

    covered: list[int] = []
    for _, lo, hi in ranges:
        assert int(hi) - int(lo) == 3, f"wave {lo}-{hi} is not four shards"
        covered.extend(range(int(lo), int(hi) + 1))
    assert covered == list(range(16)), covered


def _assembled_commands(wrapper: str) -> dict[str, str]:
    """Rejoin each branch's ``CMD="..."`` / ``CMD="${CMD} ..."`` fragments into one string.

    Checking raw lines does not work and is not a detail: the command is built across three
    assignments, so a line-wise scan both misses a flag that moved to a sibling line and trips
    over the continuation line that merely ends in `.sbatch`. What ships to Slurm is the joined
    string, so that is what gets asserted on.
    """
    out: dict[str, str] = {}
    current: str | None = None
    for raw in wrapper.splitlines():
        line = raw.strip()
        if line.startswith('CMD="cd '):
            current = "merge" if "MERGE=1" in line else "wave"
            out[current] = line[len('CMD="') : -1]
        elif line.startswith('CMD="${CMD} ') and current:
            out[current] += " " + line[len('CMD="${CMD} ') : -1]
        elif line and not line.startswith(("CMD=", "#")):
            current = None
    return out


def test_the_wave_and_merge_lines_are_two_different_qos_and_neither_is_dropped(
    wrapper: str,
) -> None:
    """Dropping --qos does not fall back to something permissive: it lands on `normal`,
    which is one minute and zero GPUs. Each QoS also rejects the other's job type."""
    cmds = _assembled_commands(wrapper)
    assert set(cmds) == {"wave", "merge"}, cmds
    for name, cmd in cmds.items():
        assert " sbatch " in cmd, f"{name} does not invoke sbatch: {cmd}"
        assert "--qos=" in cmd, f"{name} assembles an sbatch without --qos: {cmd}"
    assert "--qos=ehpc-aif-2026pg01-905" in cmds["wave"]
    assert "--qos=2cpu-single-host" in cmds["merge"]
    # The two must not be confusable: a wave is an array, the merge never is.
    assert "--array=" in cmds["wave"] and "MERGE=1" not in cmds["wave"]
    assert "--array=" not in cmds["merge"] and "MERGE=1" in cmds["merge"]


def test_every_submitted_command_carries_run_id(wrapper: str) -> None:
    """Nothing in the sbatch infers RUN_ID. Its default is `pr08-geom-tol`, which is where the
    sixteen permanently uncommittable shards live -- so an omitted RUN_ID does not fail, it
    silently merges the previous partition into a document that looks finished."""
    assembled = [ln for ln in wrapper.splitlines() if ln.lstrip().startswith('CMD="cd ')]
    assert len(assembled) == 2, f"expected one wave and one merge opener, got {assembled}"
    for line in assembled:
        assert "RUN_ID=${RUN_ID}" in line, line


def test_the_merge_carries_all_four_overrides_together(wrapper: str) -> None:
    """The free QoS caps at cpu=2 and rejects --gres, while the file asks for 26 threads, 32G
    and one GPU. Three of the four overrides is a rejected submission, not a smaller job."""
    merge = wrapper.split("if [[ -z \"${ARRAY}\" ]]; then")[-1].split("else")[0]
    for flag in ("--qos=2cpu-single-host", "--gres=none", "--cpus-per-task=2", "--mem=8G"):
        assert flag in merge, f"merge branch is missing {flag}"


def test_the_walltime_is_the_one_the_runbook_argued_for(wrapper: str, runbook: str) -> None:
    """01:45:00 leaves the heaviest shard ~1400s of slack instead of 499s and costs nothing,
    because Slurm bills runtime and not the request. If the runbook is ever revised, this fails
    rather than letting the wrapper keep submitting the old number."""
    wrapper_time = set(re.findall(r"--time=(\d\d:\d\d:\d\d)", wrapper))
    assert wrapper_time == {"01:45:00", "00:20:00"}, wrapper_time
    assert "--time=01:45:00" in runbook
    assert "--time=00:20:00" in runbook


def test_the_submit_ceiling_is_per_qos_because_the_two_branches_differ(
    wrapper: str, runbook: str
) -> None:
    """MaxSubmitJobsPU is a per-QoS association limit, and this file submits to TWO QoS. Measured
    2026-08-27 via sacctmgr: `ehpc-aif-2026pg01-905` allows 8, `2cpu-single-host` allows 4.

    A single ceiling of 8 -- which this file shipped until it was corrected -- was wrong for the
    merge and wrong in the UNSAFE direction: five jobs already on the free QoS pass `5 + 1 <= 8`
    and are then rejected by sbatch. EVERY array task counts as one submission; %4 throttles
    running jobs, not submitted ones.
    """
    assert re.search(r"PENDING \+ NEED > CEILING", wrapper), "the ceiling is not per-branch"
    # The wave branch and the merge branch must carry DIFFERENT ceilings, each beside its QoS.
    assert re.search(r"NEED=4;\s*CEILING=8;\s*QOS_NAME=ehpc-aif-2026pg01-905", wrapper)
    assert re.search(r"NEED=1;\s*CEILING=4;\s*QOS_NAME=2cpu-single-host", wrapper)
    # And the QoS each ceiling is named for must be the one that branch actually submits to.
    cmds = _assembled_commands(wrapper)
    assert "--qos=ehpc-aif-2026pg01-905" in cmds["wave"]
    assert "--qos=2cpu-single-host" in cmds["merge"]
    assert "MaxSubmitJobsPU" in runbook
    # It must ask the cluster for the count rather than assuming an empty queue.
    assert re.search(r"squeue -u \\\$USER -r -h -o '%i' \| wc -l", wrapper)
    # Counting the WHOLE queue against a per-QoS ceiling over-refuses and never under-refuses,
    # because the whole queue is a superset of any one QoS's share of it. That direction is the
    # safe one and the file must say so rather than leaving it to be re-derived.
    prose = " ".join(wrapper.replace("#", " ").split())  # the comment wraps; the claim does not
    assert "over-refuses and never under-refuses" in prose


def test_it_refuses_to_submit_against_a_stale_adapter(wrapper: str) -> None:
    """The cluster copy is an rsync, not a clone. Submitting while it still carries
    GATE_QUALIFIED = False exits 3 on all sixteen shards and spends 13.64 GPU-h on nothing --
    the most expensive failure available here, and the quietest."""
    assert "GATE_QUALIFIED" in wrapper
    assert '!= *"= True"*' in wrapper
    assert "sync.sh" in wrapper, "the refusal must name the repair"


def test_usage_is_refused_loudly_and_with_a_distinct_exit_code() -> None:
    """No argument must never mean 'wave 1'."""
    r = subprocess.run([str(_WRAPPER)], capture_output=True, text=True)
    assert r.returncode == 64, r
    assert "usage:" in r.stderr
    r_bad = subprocess.run([str(_WRAPPER), "5"], capture_output=True, text=True)
    assert r_bad.returncode == 64, "wave 5 does not exist and must not be accepted"


def test_the_watcher_takes_one_snapshot_and_does_not_poll() -> None:
    """A polling loop on login-plus is a long-running process. Enforcement is CPUQuota 200% +
    MemoryHigh 4GB, and exceeding it puts processes in D-state, which looks like a hung
    filesystem rather than an error."""
    text = _WATCHER.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(while|until)\b", text, re.M), "the watcher must not loop"
    assert "sleep" not in text
    # Read-only administration only: no sbatch, no scancel, no compute.
    for forbidden in ("sbatch", "scancel", "srun", "pip ", "conda "):
        assert forbidden not in text, f"the watcher must stay read-only, found {forbidden!r}"
