"""cluster/discoverer/submit_timing_episode.sh — the wrapper, pinned to the rule it submits under.

WHY THIS FILE EXISTS. `T40_RULE_V20` registers which episode PR-08 §8 item 3 times, and it registers
it as a CRITERION rather than as a name — §3, verbatim: *"The rule is a criterion, not a name, so
that it can be checked rather than trusted."* A wrapper that types `CHUNK_INDEX=372` into a submit
line converts that criterion back into a name and throws away the whole property. So the wrapper
re-derives the criterion from the committed evidence at submit time, and these tests check that the
derivation is really the rule's — against the rule DOCUMENT and against the committed artifacts,
never against a number typed into this file.

The sources:
  docs/preregistration/PR-08-V20-timing-episode-registration.md  the rule
  runs/pr08-robot-mask-area/POOLED.json                          the population
  configs/transfer25/pr08_robot_mask_area.json                   the area bound
  cluster/discoverer/97_transfer25_restyle.sbatch                the flags it must carry
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO / "cluster/discoverer/submit_timing_episode.sh"
_SBATCH = _REPO / "cluster/discoverer/97_transfer25_restyle.sbatch"
_RULE = _REPO / "docs/preregistration/PR-08-V20-timing-episode-registration.md"
_POOLED = _REPO / "runs/pr08-robot-mask-area/POOLED.json"
_BOUND = _REPO / "configs/transfer25/pr08_robot_mask_area.json"


@pytest.fixture(scope="module")
def wrapper() -> str:
    return _WRAPPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rule() -> str:
    return _RULE.read_text(encoding="utf-8")


def test_the_wrapper_parses_and_is_executable(wrapper: str) -> None:
    """A syntax error here is discovered with a queue slot in hand."""
    assert _WRAPPER.stat().st_mode & 0o111, "not executable"
    subprocess.run(["bash", "-n", str(_WRAPPER)], check=True, capture_output=True)


def test_the_criterion_the_wrapper_computes_is_the_one_v20_registered() -> None:
    """The load-bearing test. Recompute V20 §3 here, independently of the wrapper's own python, and
    require that it lands on the episode the rule wrote down.

    This is not circular: the rule document, the pooled evidence and the committed bound are three
    files the wrapper does not write, and the arithmetic below is spelled out rather than imported
    from it. If the evidence moves under the rule, this goes red — which is the outcome V20 §3 asks
    for ("the submission does not proceed and this rule is re-evaluated"), reached before a slot is
    spent rather than after.
    """
    pooled = json.loads(_POOLED.read_text(encoding="utf-8"))
    assert pooled["measurement_qualified"] is True
    bound = json.loads(_BOUND.read_text(encoding="utf-8"))["max_frame_fraction"]

    eps = pooled["per_episode"]
    # BOTH halves of check_mask, in V20 §2's order. Neither is waived.
    survivors = [e for e in eps if e["empty_frames"] == 0 and max(e["area_fractions"]) <= bound]
    median = statistics.median([e["n_frames"] for e in eps])
    pick = min(survivors, key=lambda e: (abs(e["n_frames"] - median), str(e["episode"])))

    # V20 §2's own counts, quoted back at it.
    assert len(eps) == 402
    assert len(survivors) == 17
    assert median == 421.5
    assert pick["episode"] == "episode_000371"
    assert pick["n_frames"] == 422


def test_the_rule_document_still_says_what_the_wrapper_holds_it_to(rule: str) -> None:
    """The wrapper carries V20's answer as a value to CHECK against, so the rule and the wrapper
    must agree about it. If V20 is ever revised, this fails rather than letting the wrapper keep
    refusing against a superseded name."""
    assert "episode_000371" in rule
    assert "**`episode_000371`, 422 frames" in rule
    # And the three things the wrapper prints back are the rule's, not the wrapper's invention.
    assert "Outcome M — MEASURED" in rule
    assert "Outcome R — REFUSED ON THE CLUSTER" in rule
    assert "Outcome F — FAILED FOR ANY OTHER REASON" in rule
    # Wrapped in the document, so the newline is normalised rather than guessed at.
    assert "outcome R does not license walking down" in " ".join(rule.split())


def test_the_wrapper_derives_rather_than_types_the_index(wrapper: str) -> None:
    """The defect this file exists to prevent: a hard-coded CHUNK_INDEX. V20 §3 explicitly does NOT
    assert the index, because it depends on the manifest rather than on the document, so the wrapper
    must ask the manifest that will actually be read."""
    assert "CHUNK_INDEX=${CHUNK_INDEX}" in wrapper
    # Comments are exempt on purpose: the file's own header explains the defect by NAMING the
    # literal it refuses to use, and screening that out would make the explanation unwritable.
    code = "\n".join(
        line for line in wrapper.splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"CHUNK_INDEX=\d", code), "the index is typed, not resolved"
    assert not re.search(r"CHUNK_TOTAL=\d", code), "the total is typed, not resolved"
    # It resolves from the manifest the sbatch reads, by the sbatch's own ordering.
    assert "manifest.json" in wrapper
    assert "sorted(m['episodes'], key=lambda e: str(e['id']))" in wrapper
    # And it re-derives the criterion rather than trusting the name.
    assert "POOLED.json" in wrapper
    assert "max_frame_fraction" in wrapper
    assert "statistics.median" in wrapper


def test_it_refuses_rather_than_retargets_when_the_criterion_moves(wrapper: str) -> None:
    """V20 §3's blind pre-registration: 'It is not silently retargeted at a neighbour.' The runners-
    up are 2.5, 3.5 and 4.5 frames away, so a neighbour is always available and always wrong."""
    assert "REFUSING: the criterion now yields" in wrapper
    assert "NOT silently retargeted at a neighbour" in wrapper
    # Three separate refusals, all exit 75: criterion moved, episode absent, populations disagree.
    assert wrapper.count("exit 75") >= 4


def test_the_flags_are_the_sbatch_header_recipe(wrapper: str) -> None:
    """Every flag must be the one 97_transfer25_restyle.sbatch's own header asks for. A wrapper that
    drifts from what it wraps is worse than the long line, because it looks authoritative."""
    header = _SBATCH.read_text(encoding="utf-8")
    recipe = header.split("sbatch --time=01:30:00 97_transfer25_restyle.sbatch")[0]
    for flag in ("TIMING=1", "STAGE=1", "STYLE_SET=train"):
        assert flag in wrapper, f"wrapper is missing {flag}"
        assert flag in recipe, f"{flag} is not in the sbatch's own recipe"
    assert "--time=${WALL}" in wrapper and 'WALL=${WALL:-01:30:00}' in wrapper
    assert "sbatch --time=01:30:00" in header, "the header no longer registers this walltime"


def test_the_control_set_is_the_committed_one_and_not_a_choice_made_here(wrapper: str) -> None:
    """CONTROL decides how much geometry survives — which is what §6 G0b measures — and it decides
    the timing number too, because every block Transfer2.5 must ESTIMATE is GPU time this
    measurement includes. Picking it here would be picking it after seeing the data."""
    assert "CONTROL=${CONTROL:-depth:0.5,seg:0.5}" in wrapper
    api = (_REPO / "docs/transfer25-api.md").read_text(encoding="utf-8")
    assert "PR-08's committed control set is `depth:0.5,seg:0.5`" in api


def test_the_run_id_is_dated_and_never_the_sbatch_default(wrapper: str) -> None:
    """The default RUN_ID still holds job 189142's disqualified THROUGHPUT.json — 0.2 s/frame from
    a run whose own log says '0 success, 1 error', pricing the partition ~10x under."""
    assert "RUN_ID=${RUN_ID:-t040-transfer25-restyle-timing-$(date -u +%Y-%m-%d)}" in wrapper
    assert "189142" in wrapper, "the refusal must say what the default holds"
    assert "REUSE_THROUGHPUT" not in wrapper.split("# 4.")[-1], "the submit line must not reuse"


def test_the_submit_ceiling_counts_the_whole_users_queue(wrapper: str) -> None:
    """MaxSubmitJobsPU=8, and a peer session shares this allocation, so 'my queue is empty' is not
    the question."""
    assert "PENDING + 1 > 8" in wrapper
    assert re.search(r"squeue -u \\\$USER -r -h -o '%i' \| wc -l", wrapper)


def test_it_checks_the_resolution_the_registered_experiment_fixes(wrapper: str) -> None:
    """PR-08 §3 fixes 640x480 as the GR00T N1.7 ego_view contract. The sbatch refuses a manifest at
    any other size; refusing here means refusing before the slot rather than inside it."""
    assert "(640, 480)" in wrapper
    assert "ego_view contract" in wrapper
