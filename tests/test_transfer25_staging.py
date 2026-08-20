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
_SOURCE = _JOBS / "100_fetch_pr08_source.sbatch"


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
    """PR-08 §8 item 3: no budget line exists yet, so none of these jobs may spend from it.

    100 is in this list for the same reason as 98 and 99 and one more: it PRECEDES the throughput
    measurement, so there is not even a number it could be checked against.
    """
    for path in (_BUILD, _STAGE, _SOURCE):
        text = _text(path)
        assert "--qos=2cpu-single-host" in text, f"{path.name} is not on the free QoS"
        assert "--gres=gpu" not in text, f"{path.name} requests a GPU"


def test_neither_job_generates_anything() -> None:
    """PR-08 §1 licenses building and timing, never generating. These must stay on that side."""
    for path in (_BUILD, _STAGE, _SOURCE):
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


def test_98_syncs_a_cuda_12_extra_because_torch_is_in_no_default() -> None:
    """A plain ``uv sync`` on this project installs no torch at all.

    Read off the pinned checkout's own pyproject.toml: torch reaches the venv only through the
    ``cu128`` / ``cu130`` extras, and the base dependencies name none of them. So an empty default
    is not the conservative choice it looks like -- it is the one that produces a venv the CUDA
    check below can only report as broken, four hours after the job started.
    """
    build = _text(_BUILD)
    match = re.search(r"TRANSFER_UV_ARGS=\$\{TRANSFER_UV_ARGS:-([^}]*)\}", build)
    assert match, "98 no longer sets a default TRANSFER_UV_ARGS"
    default = match.group(1).strip()
    assert default, (
        "TRANSFER_UV_ARGS defaults to empty, i.e. a plain `uv sync`. On this project that "
        "installs no torch -- torch is only in the cu128/cu130 extras."
    )
    assert "cu128" in default, (
        f"default {default!r} does not name the CUDA 12 extra. This cluster's newest CUDA module "
        f"is 12.8; a cu130 build imports cleanly and dies on the first kernel launch."
    )
    assert "cu130" not in default, f"default {default!r} names cu130 on a 12.8 cluster"


def test_98_reports_a_missing_torch_as_a_missing_extra_not_a_traceback() -> None:
    """The failure mode the default above prevents still has to be legible when it happens.

    Someone overriding TRANSFER_UV_ARGS gets the empty-sync venv back. A bare ImportError
    traceback sends them looking for a broken install; the real answer is one flag.
    """
    build = _text(_BUILD)
    assert "ModuleNotFoundError" in build, (
        "98 does not distinguish 'no torch installed' from 'torch built against the wrong CUDA'"
    )
    guard = build.index("ModuleNotFoundError")
    version_check = build.index("torch.version.cuda")
    assert guard < version_check, (
        "the missing-torch guard must come before torch.version.cuda is read, or the import "
        "error surfaces as a traceback instead of the message"
    )


def test_100_pins_the_source_corpus_revision() -> None:
    """The corpus 97 restyles has to be able to name the snapshot it came from.

    Same rule as 91 and 99 state for weights, and it bites harder here: a result cites the corpus
    it was measured on, and ``main`` is not a corpus.
    """
    source = _text(_SOURCE)
    match = re.search(r"SOURCE_REVISION=\$\{SOURCE_REVISION:-([^}]*)\}", source)
    assert match, "100 no longer sets a default SOURCE_REVISION"
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), (
        f"default revision {match.group(1)!r} is not a 40-hex commit sha"
    )
    case_block = source[source.index('case "${SOURCE_REVISION}"') :]
    case_block = case_block[: case_block.index("esac")]
    for pointer in ("main", "master", "HEAD", "latest"):
        assert pointer in case_block, f"{pointer!r} is not refused as a source revision"


def test_100_writes_the_path_97_reads_by_default() -> None:
    """Two files naming the same tree by two different literals is a bug waiting for a rename."""
    source = _text(_SOURCE)
    restyle = _text(_RESTYLE)
    match = re.search(r"SOURCE=\$\{SOURCE:-\$\{PROJ\}/([^}]*)\}", restyle)
    assert match, "97 no longer defaults SOURCE under ${PROJ}"
    assert match.group(1) in source, (
        f"97 reads ${{PROJ}}/{match.group(1)} and 100 does not write it"
    )


def test_97_passes_control_to_every_driver_invocation() -> None:
    """restyle_transfer25.py declares --control required with no default; 97 has to supply it.

    The driver was written after 97 (14bf784 after 50ab5a4) and made the control spec mandatory, on
    the grounds that the choice decides how much geometry survives and so cannot be picked after
    looking at clips. 97 was not updated to match, so BOTH of its invocations -- the timing run and
    the generation run -- ended at argparse, after Slurm had already handed out an H200. That is
    the exact shape of failure 97's own header exists to prevent, which is why the guard belongs in
    a test rather than in a comment.
    """
    restyle = _text(_RESTYLE)
    driver = (_REPO_ROOT / "scripts" / "restyle_transfer25.py").read_text()

    # Read the requirement off the driver rather than restating it: if --control ever gains a
    # default, this test should stop demanding one instead of failing for the wrong reason.
    idx = driver.index('"--control"')
    assert "required=True" in driver[idx : idx + 200], (
        "restyle_transfer25.py no longer requires --control; this test is now over-strict"
    )

    invocations = restyle.count('python "${RESTYLE_DRIVER}"')
    assert invocations >= 2, f"expected the timing and generation invocations, found {invocations}"
    passes = restyle.count('--control "${CONTROL}"')
    assert passes == invocations, (
        f"{invocations} driver invocations in 97 but only {passes} pass --control. The one that "
        f"does not dies at argparse with a GPU already allocated."
    )
    assert ': "${CONTROL:?' in restyle, (
        "CONTROL must be required with no default, like the driver's own flag -- a default here "
        "would reintroduce exactly the silent choice the driver refuses to make"
    )


def test_the_timing_number_records_which_controls_it_measured() -> None:
    """A throughput figure is only comparable to a run under the same conditioning.

    The source manifest carries no depth or segmentation maps, so each control block is one map
    Transfer2.5 estimates with its own model on the same GPU. Those are real seconds inside the
    measurement, and a ceiling derived under one spec and spent under another is not a ceiling for
    that run.
    """
    restyle = _text(_RESTYLE)
    assert '"control": control' in restyle, "THROUGHPUT.json does not record the control spec"


def test_98_warns_that_the_two_env_files_collide_on_framework() -> None:
    """Both cosmos_env.sh and cosmos_transfer_env.sh export FRAMEWORK, at different checkouts."""
    build = _text(_BUILD)
    assert "cosmos_env.sh" in build and "FRAMEWORK" in build
    assert "DO NOT source this together with cosmos_env.sh" in build, (
        "the written env file must carry the collision warning, not only 98's header"
    )


def test_98_pins_the_interpreter_rather_than_obeying_dot_python_version() -> None:
    """The checkout's own ``.python-version`` cannot install the checkout's own cu128 extra.

    Measured, job 189024, which failed in 12 s: upstream commit ce13887 ("Add Python 3.13 support
    (cu130+torch29 via v1.5.0 index)") set ``.python-version`` to 3.13, but the pinned sha still
    resolves cu128 from the **v1.2.0** index, and that index publishes flash-attn wheels for cp310
    and cp312 only. uv obeys ``.python-version`` unless told otherwise, so the default path is the
    broken one and the failure is a wheel-resolution error with no mention of Python at all.
    """
    build = _text(_BUILD)
    match = re.search(r"TRANSFER_PYTHON=\$\{TRANSFER_PYTHON:-([^}]*)\}", build)
    assert match, (
        "98 no longer pins TRANSFER_PYTHON. Without it uv takes the checkout's .python-version "
        "(3.13 at the pinned sha), for which the cu128 extra has no wheels."
    )
    pinned = match.group(1).strip()
    assert pinned in {"3.10", "3.12"}, (
        f"TRANSFER_PYTHON={pinned!r} is not an ABI tag the cu128 v1.2.0 index carries "
        "(it publishes cp310 and cp312 only)"
    )
    assert re.search(r"uv sync --python \"\$\{TRANSFER_PYTHON\}\"", build), (
        "TRANSFER_PYTHON is set but never reaches `uv sync`, so .python-version still wins"
    )


def test_98_does_not_need_tomllib_to_print_its_own_diagnostic() -> None:
    """The compute nodes run python3 3.9; tomllib arrived in 3.11.

    This block exists to tell the operator which extras are legal when TRANSFER_UV_ARGS has to be
    set by hand. On the one run where that mattered it raised ModuleNotFoundError and printed
    nothing, so the job that needed the hint is exactly the job that did not get it.
    """
    build = _text(_BUILD)
    # Scope this to the heredoc body rather than the whole file. The word survives in the comment
    # that explains the absence -- that comment is the point -- so a file-wide search would either
    # fail on the explanation or, if written as a line pattern, miss `import sys, tomllib`. The
    # body is what actually runs on a 3.9 interpreter, so the body is what gets asserted about.
    lines = build.splitlines()
    starts = [i for i, line in enumerate(lines) if "<<'EOFDIAG'" in line]
    assert len(starts) == 1, "98's pyproject diagnostic heredoc is no longer named EOFDIAG"
    ends = [i for i, line in enumerate(lines) if line.strip() == "EOFDIAG" and i > starts[0]]
    assert ends, "98's EOFDIAG heredoc is never terminated"
    body = "\n".join(lines[starts[0] + 1 : ends[0]])
    assert "tomllib" not in body, (
        "98's diagnostic uses tomllib, which does not exist on the 3.9 python3 these nodes ship"
    )
    assert "tomli" not in body, "98 depends on a third-party toml parser for a diagnostic"


def test_98_explains_an_abi_failure_instead_of_inviting_the_wrong_fix() -> None:
    """The obvious 'fix' for a --python failure is to drop --python, which restores the bug."""
    build = _text(_BUILD)
    assert "ABI tag" in build, "98's sync failure path never mentions the ABI mismatch"
    assert "Do NOT 'fix' this by dropping --python" in build, (
        "98 does not warn against the fix that reinstates .python-version=3.13"
    )


def test_100_authenticates_because_anonymous_hits_the_rate_limit() -> None:
    """Not gated is not the same as not rate-limited.

    Job 189023 fetched 80% of 813 files anonymously and then died on 429 Too Many Requests, six
    minutes in and after a much longer queue wait. hf warns on its first line and proceeds anyway,
    so the run looks healthy right up until it isn't. Refusing up front costs a second.
    """
    source = _text(_SOURCE)
    assert "HF_TOKEN" in source, "100 never looks for a token; an anonymous 813-file fetch 429s"
    assert "429" in source, (
        "100's token requirement no longer records WHY it exists, so the next reader will "
        "reasonably delete it on the grounds that the dataset is public"
    )
    # Line-anchored: a commented-out `# export HF_TOKEN` still contains the substring, and that
    # is exactly the shape a disabling edit takes.
    assert re.search(r"^export HF_TOKEN$", source, re.MULTILINE), (
        "100 resolves a token but never exports it, so the hf CLI still runs anonymous"
    )
    for path in (_SOURCE, _STAGE):
        assert '"${HOME}/.huggingface/token"' in _text(path), (
            f"{path.name} does not search ~/.huggingface/token; the two jobs must agree on where "
            "a token lives or one of them silently runs anonymous"
        )


def test_no_job_ever_echoes_a_token() -> None:
    """A Slurm log is a plain file that outlives the job and gets copied around by sync.sh."""
    for path in (_SOURCE, _STAGE):
        text = _text(path)
        assert not re.search(r"echo[^\n|]*\$\{?HF_TOKEN\}?(?![A-Za-z_])", text), (
            f"{path.name} echoes HF_TOKEN into the job log"
        )
        if "HF_TOKEN" in text:
            assert "not echoed" in text, f"{path.name} handles a token without saying it is hidden"


def test_100_asks_hf_for_a_bare_path_rather_than_parsing_its_banner() -> None:
    """hf 1.28 decorates its output; `tail -1` then captures the label, not the path.

    Measured, job 189134: the download SUCCEEDED and the job failed anyway, because stdout ended
    "  path: /valhalla/..." and the directory guard rejected it. --quiet is documented as "one ID
    per line", which is a contract; the banner is not.
    """
    source = _text(_SOURCE)
    capture = re.search(r"SNAPSHOT=\$\((.*?)\)\n", source, re.DOTALL)
    assert capture, "100 no longer captures a snapshot path from hf"
    assert "--quiet" in capture.group(1), (
        "100 captures hf's stdout without --quiet, so it parses the decorated banner"
    )


def test_99_does_not_capture_stdout_so_it_needs_no_quiet_flag() -> None:
    """Guards the asymmetry, so nobody 'fixes' 99 by symmetry or breaks it by tidying.

    99 passes --local-dir and reads the directory afterwards. That is why the banner change that
    broke 100 left 99 untouched, and why the two jobs legitimately differ here.
    """
    stage = _text(_STAGE)
    assert "--local-dir" in stage, "99 no longer stages via --local-dir"
    assert not re.search(r"\$\(uvx hf@latest download", stage), (
        "99 now captures hf's stdout; if so it needs --quiet for the reason 100 does"
    )
