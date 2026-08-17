"""Tests for the two jobs that close 97_transfer25_restyle.sbatch's ``TODO(human)``.

98 and 99 cannot be exercised here — one clones a repository and the other downloads 30 GB from a
gated host — so what is testable is the set of properties that are cheap to get wrong and
expensive to discover on the cluster four hours in. Each test below is one such property, and each
corresponds to a decision recorded in the two headers rather than to an implementation detail:

1. **The contract with 97 is complete.** 97 asserts four names with ``: ${..:?}``. If 98 and 99
   together export three of them, the failure surfaces only after Slurm has handed out a GPU. The
   contract is read out of 97 rather than restated here, so adding a fifth assertion to 97 fails
   this test instead of silently outgrowing it.
2. **The revision is pinned, and moving pointers are refused.** ``main`` is not a checkpoint id
   (91's header, PR-08 §6). Upstream auto-downloads checkpoints at inference time, so an unpinned
   stage is not a missing nicety — it lets the framework choose what we ran.
3. **No checkpoint UUID is hardcoded.** Each control ships two undocumented variants; 99 stages
   both and records them. A UUID appearing in the job would be a plausible wrong value chosen by
   whoever typed that line, which is the failure PR-08's guards exist to prevent.
4. **The licence gate is checked before the download**, so a 403 costs seconds rather than a
   partial 30 GB tree — and is reported as a human act, never worked around.
5. **Staging bills no GPU-hours.** PR-08 §8 item 3 says no budget line exists until the throughput
   measurement does, so staging must not spend from a budget that has not been written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOBS = _REPO_ROOT / "cluster" / "discoverer"
_BUILD = _JOBS / "98_build_transfer25_env.sbatch"
_STAGE = _JOBS / "99_stage_transfer25_weights.sbatch"
_RESTYLE = _JOBS / "97_transfer25_restyle.sbatch"


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.name} does not exist"
    return path.read_text()


def test_both_jobs_exist_and_are_the_ones_97_points_at() -> None:
    """97's TODO names a job that must exist; these two are it."""
    build, stage = _text(_BUILD), _text(_STAGE)
    assert "cosmos_transfer_env.sh" in build
    assert "cosmos_transfer_env.sh" in stage
    # 97 sources the file both halves write.
    assert "cosmos_transfer_env.sh" in _text(_RESTYLE)


def test_the_four_names_97_asserts_are_all_exported_by_98_or_99() -> None:
    """Read the contract out of 97 rather than restating it.

    97 refuses to start unless each of these resolves. Discovering a missing one on the cluster
    costs a GPU allocation; discovering it here costs nothing.
    """
    restyle = _text(_RESTYLE)
    required = set(re.findall(r'^:\s*"\$\{([A-Z_][A-Z0-9_]*)\s*:\?', restyle, re.MULTILINE))
    # The four this pair is responsible for. STYLE_SET/CHUNK_* are per-run arguments the operator
    # passes to 97 itself, not environment the staging jobs can know.
    ours = {n for n in required if n.startswith("TRANSFER_")}
    assert ours, "97 no longer asserts any TRANSFER_* name — has the contract moved?"

    exported = set(re.findall(r"^export ([A-Z_][A-Z0-9_]*)=", _text(_BUILD), re.MULTILINE))
    exported |= set(re.findall(r"^export ([A-Z_][A-Z0-9_]*)=", _text(_STAGE), re.MULTILINE))

    missing = ours - exported
    assert not missing, f"97 asserts {sorted(missing)} and neither 98 nor 99 exports them"

    # FRAMEWORK is asserted by use, not by `:?` — 97 activates ${FRAMEWORK}/.venv.
    assert "FRAMEWORK" in exported, "97 activates ${FRAMEWORK}/.venv but nothing exports FRAMEWORK"


def test_the_revision_is_a_pinned_sha_not_a_moving_pointer() -> None:
    stage = _text(_STAGE)
    match = re.search(r"TRANSFER_MODEL_REVISION=\$\{TRANSFER_MODEL_REVISION:-([^}]*)\}", stage)
    assert match, "99 no longer sets a default TRANSFER_MODEL_REVISION"
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), (
        f"default revision {match.group(1)!r} is not a 40-hex commit sha"
    )


@pytest.mark.parametrize("pointer", ["main", "master", "HEAD", "latest"])
def test_moving_pointers_are_refused_by_name(pointer: str) -> None:
    """Not merely 'a sha is the default' — passing ``main`` explicitly must be rejected.

    A default is only a default. The guard is what stops someone overriding it on the command line
    with the one value that looks most reasonable and is least reproducible.
    """
    stage = _text(_STAGE)
    case_block = stage[stage.index('case "${TRANSFER_MODEL_REVISION}"') :]
    case_block = case_block[: case_block.index("esac")]
    assert pointer in case_block, f"{pointer!r} is not refused as a revision"


def test_the_hex_guard_backs_up_the_case_statement() -> None:
    """The case list is finite; the regex catches everything else, e.g. a tag like 'v1.0'."""
    assert re.search(
        r"\[\[\s*\"\$\{TRANSFER_MODEL_REVISION\}\"\s*=~\s*\^\[0-9a-f\]\{40\}\$", _text(_STAGE)
    ), "99 does not enforce a 40-hex revision beyond its named-pointer case list"


def test_no_checkpoint_uuid_is_hardcoded() -> None:
    """The variant choice is deliberately unresolved — see 99's header.

    Each control ships two checkpoints and upstream documents neither, so a UUID appearing in the
    job would be a guess wearing the costume of a decision.
    """
    # No \b anchors. Upstream names these files ``<uuid>_ema_bf16.pt``, and ``_`` is a word
    # character — so a trailing \b never matches the exact shape this test exists to catch. Found
    # by mutation: the first version of this test passed while a real UUID sat in the file.
    uuid = re.compile(
        r"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    )
    for path in (_BUILD, _STAGE):
        found = uuid.findall(_text(path))
        assert not found, f"{path.name} hardcodes checkpoint UUID(s) {found}"


def test_the_staged_manifest_leaves_the_variant_unselected() -> None:
    assert '"variant_selected": None' in _text(_STAGE), (
        "STAGED.json must record the variant as an open choice, not silently pick one"
    )


def test_the_gate_is_probed_before_a_single_byte_is_downloaded() -> None:
    """Order matters: a 403 must cost seconds, not a partial 30 GB tree."""
    stage = _text(_STAGE)
    download = stage.index("hf@latest download")
    # Anchor on the control flow, not on a log line. An earlier version of this test looked for the
    # string "gate probe" and passed when the whole block was moved below the download, because the
    # echo travelled with it — the assertion was about a message rather than about order.
    for marker, what in (
        ("GATE_URL=", "the probe URL is built"),
        ('case "${CODE}"', "the response code is branched on"),
        ("403)", "the gate branch"),
        ("exit 1 ;;", "the refusal"),
    ):
        assert marker in stage, f"99 no longer contains {what} ({marker!r})"
        assert stage.index(marker) < download, (
            f"{what} ({marker!r}) happens after the download starts — a 403 would then cost a "
            f"partial 30 GB tree instead of seconds"
        )


def test_a_403_names_the_human_action_and_offers_no_workaround() -> None:
    stage = _text(_STAGE)
    assert "403" in stage
    assert "https://huggingface.co/${TRANSFER_MODEL_ID}" in stage, (
        "the 403 branch must print the URL a person has to visit"
    )
    assert "account holder's act" in stage
    # A gate is not something a flag gets around. If one ever appears, this test is the place the
    # discussion happens.
    for forbidden in ("--token-from", "HF_HUB_DISABLE_", "trust_remote_code", "hf_transfer_bypass"):
        assert forbidden not in stage, f"{forbidden} looks like a gate workaround"


def test_staging_and_building_bill_no_gpu_hours() -> None:
    """PR-08 §8 item 3: no budget line exists yet, so neither job may spend from it."""
    for path in (_BUILD, _STAGE):
        text = _text(path)
        assert "--qos=2cpu-single-host" in text, f"{path.name} is not on the free QoS"
        assert "--gres=gpu" not in text, f"{path.name} requests a GPU"


def test_neither_job_generates_anything() -> None:
    """PR-08 §1 licenses building and timing, never generating. These two must stay on that side."""
    for path in (_BUILD, _STAGE):
        text = _text(path)
        assert "restyle_transfer25.py" not in text, f"{path.name} invokes the generation driver"
        assert "STYLE_SET" not in text, f"{path.name} reaches into the generation partition"


def test_98_refuses_a_cuda_major_that_does_not_match_the_cluster() -> None:
    """The check is on the built-against CUDA major, not on torch importing.

    A cu13x wheel imports fine on a CPU node and dies on the first kernel launch hours into a
    generation job, which is the expensive place to find out.
    """
    build = _text(_BUILD)
    assert "torch.version.cuda" in build
    assert "major != 12" in build, "98 does not check the CUDA major against this cluster's 12.8"


def test_98_warns_that_the_two_env_files_collide_on_framework() -> None:
    """Both cosmos_env.sh and cosmos_transfer_env.sh export FRAMEWORK, at different checkouts."""
    build = _text(_BUILD)
    assert "cosmos_env.sh" in build and "FRAMEWORK" in build
    assert "DO NOT source this together with cosmos_env.sh" in build, (
        "the written env file must carry the collision warning, not only 98's header"
    )
