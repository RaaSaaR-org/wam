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

import json
import pathlib
import re
import subprocess
import sys
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


def _default_source(job: pathlib.Path) -> str:
    """The tree a job defaults SOURCE to, as the literal it writes."""
    match = re.search(r"SOURCE=\$\{SOURCE:-\$\{PROJ\}/([^}]*)\}", _text(job))
    assert match, f"{job.name} no longer defaults SOURCE under ${{PROJ}}"
    return match.group(1)


def test_97_and_106_default_to_the_same_tree() -> None:
    """The area bound is a claim about a corpus, so the two jobs must mean the same one.

    106 measures the robot-mask area distribution and stamps the SOURCE manifest's sha256 into the
    artifact; ``robot_composite.load_area_bound`` then compares that against the sha256 of the tree
    the RUN is restyling, and refuses when they differ — "holding corpus A's distribution over
    corpus B is the same drift by another route". The AV1 and H.264 trees are bit-exact in pixels
    and have DIFFERENT manifests, so two jobs disagreeing here is not a cosmetic mismatch: it is a
    G0c refusal that arrives after Slurm has handed out a GPU. 97 pointed at the AV1 tree while 106
    and 107 pointed at the H.264 one until 2026-08-24.
    """
    assert _default_source(_RESTYLE) == _default_source(_JOBS / "106_measure_robot_mask_area.sbatch")


def test_the_tree_97_reads_by_default_is_one_something_in_the_repo_produces() -> None:
    """Two files naming the same tree by two different literals is a bug waiting for a rename.

    100 builds the AV1 tree from the HF source; ``scripts/transcode_corpus_lossless.py`` builds the
    bit-exact H.264 transcode of it, which is what the generation venv's cv2 can actually decode.
    Whichever 97 defaults to, its PRODUCER has to name it by the same literal.
    """
    # On the BASENAME: the producers write the tree under their own parent (100 under ${PROJ}/data,
    # the transcode wherever it is pointed), so the directory name is the shared literal and the
    # thing a rename would break.
    default = _default_source(_RESTYLE).rstrip("/").rsplit("/", 1)[-1]
    producers = [_text(_SOURCE),
                 (_REPO_ROOT / "scripts" / "transcode_corpus_lossless.py").read_text(encoding="utf-8")]
    assert any(default in text for text in producers), (
        f"97 defaults SOURCE to .../{default} and nothing in the repo writes that name"
    )


def test_97_refuses_the_av1_tree_by_name_rather_than_failing_to_decode_it() -> None:
    """The AV1 failure is SILENT — cv2 opens the container, reports the frame count and reads
    nothing — so three jobs died on it before anyone knew why (186357, 189585, 189584). A job that
    can only discover this on a GPU should refuse it on the login node instead."""
    restyle = _text(_RESTYLE)
    assert "*pr08-apple-640x480)" in restyle, "97 no longer refuses the AV1 tree by name"
    assert "AV1" in restyle


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


# -- 97's TIMING branch: the artifact it reuses, and the number it projects with -------------------
#
# Two defects live here and both are silent. The default RUN_ID still holds job 189142's
# THROUGHPUT.json — 0.2 s/frame from a run whose own log says "0 success, 1 error" — and the timing
# branch's first act was to cat it and exit 0, which prices the whole partition at 238.4 GPU-h
# against roughly 2 380 at the artifact's own end-to-end rate. And the projection multiplied by a
# literal 172_000 while the corpus is 171 625 and partition_facts.json already carried the counted
# figure. Both of these are checked here by RUNNING the embedded python rather than by reading it:
# a heredoc is not covered by anything else in this repository, and its failure mode is a cluster
# job that dies four hundred lines in.

def _py_block(text: str, marker: str) -> str:
    """One `python - ... <<'PY'` body out of a job file, chosen by a string only it contains."""
    # `<<'PY'` can be followed by more of the command line -- the classifier ends `|| TP_VERDICT=$?`
    # so that `set -e` does not kill the job on a refusal -- hence `[^\n]*` before the body.
    blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY\n", text, re.S)
    hits = [b for b in blocks if marker in b]
    assert len(hits) == 1, f"{len(hits)} blocks contain {marker!r}; the marker is no longer unique"
    return hits[0]


def _run_block(block: str, *argv: str) -> "subprocess.CompletedProcess":
    """Exactly how the sbatch runs it: `python - <args>` with the body on stdin."""
    return subprocess.run(
        [sys.executable, "-", *argv], input=block, text=True, capture_output=True, check=False
    )


#: The artifact job 189142 actually left at the default RUN_ID, field for field, as quoted in
#: docs/investigations/2026-08-27-pr08-fronts/F1-item3-throughput.md §3. Every field is well-formed;
#: that is the point of it. Nothing INSIDE this file says the run generated no clip, which is why
#: reading it as a finished measurement was the easiest thing in the world to do.
_JOB_189142 = {
    "measured_on": "1 x H200, 640x480, one episode",
    "episode": "episode_000000",
    "style": "train-01-oak-tungsten",
    "frames": 590,
    "wall_seconds": 118.0,
    "seconds_per_frame": 0.2,
    "gpu_seconds_per_frame": 0.2,
    "frames_per_variant": 172000,
    "gpu_hours_per_variant": 9.56,
    "gpu_hours_per_variant_is_lower_bound_above_1_gpu": True,
    "generator": "nvidia/Cosmos-Transfer2.5-2B@ce8440327c632d8313c3bde69db13b627ba5cae1",
    "control": "depth:0.5,seg:0.5",
    "ceiling_gpu_hours_supplied_at_measurement_time": None,
    "ceiling_gpu_hours_supplied_at_measurement_time_note": (
        "null is the expected value. PR-08 §8 item 3: no budget line exists until this measurement "
        "does, so TIMING=1 asks for no ceiling and ignores one if supplied."),
}


def _classifier() -> str:
    return _py_block(_text(_RESTYLE), "UNPROVEN: schema is")


def _classify(tmp_path: pathlib.Path, payload) -> "subprocess.CompletedProcess":
    target = tmp_path / "THROUGHPUT.json"
    target.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=2))
    return _run_block(_classifier(), str(target))


def test_the_timing_branch_refuses_job_189142s_artifact_outright(tmp_path: pathlib.Path) -> None:
    """The one real THROUGHPUT.json this project has ever produced must never be reused.

    It is a wall clock around a crash: the driver died in SetupArguments validation, produced no
    clip and exited 0, and 118 s became 0.2 s/frame. Nothing in the file contradicts it, so the
    check cannot be "does it look wrong" — it is "can it PROVE a clip came out", and this one
    predates the field that records that. Unproven and wrong are the same thing for a number that
    prices 4 290 625 frames.
    """
    done = _classify(tmp_path, _JOB_189142)
    assert done.returncode == 4, done.stdout + done.stderr
    assert "UNPROVEN" in done.stdout


def test_an_artifact_that_records_no_successful_unit_is_refused(tmp_path: pathlib.Path) -> None:
    payload = dict(_JOB_189142, schema="wam.transfer25_throughput/1", units_timed=1,
                   units_succeeded=0)
    done = _classify(tmp_path, payload)
    assert done.returncode == 4
    assert "NO SUCCESSFUL UNIT" in done.stdout


@pytest.mark.parametrize(
    "marker",
    [
        {"disqualified": "measured under the wrong control spec"},
        {"gate_disqualified_reasons": ["segmenter contract disagreement"]},
        {"measurement_qualified": False},
    ],
)
def test_a_disqualified_artifact_is_refused_however_the_disqualification_is_spelled(
    tmp_path: pathlib.Path, marker: dict
) -> None:
    """None of these fields exist in what 97 writes today, and that is deliberate.

    They are the shapes a later version or a hand-annotation would use to say "do not build a
    budget on this". A reuse check that only understood today's schema would read tomorrow's
    disqualification as an unknown key and reuse the number anyway.
    """
    payload = dict(_JOB_189142, schema="wam.transfer25_throughput/1", units_timed=1,
                   units_succeeded=1, **marker)
    done = _classify(tmp_path, payload)
    assert done.returncode == 3, done.stdout + done.stderr
    assert "DISQUALIFIED" in done.stdout


def test_an_unreadable_artifact_is_refused_rather_than_crashing_the_job(tmp_path: pathlib.Path) -> None:
    done = _classify(tmp_path, "{not json at all")
    assert done.returncode == 2
    assert "UNREADABLE" in done.stdout


def test_a_qualified_artifact_still_passes_the_check(tmp_path: pathlib.Path) -> None:
    """The guard must not become "never reuse anything", or the requeue guard it sits on top of —
    a preempted timing job re-timing work it already finished on the paid QoS — comes back."""
    payload = dict(_JOB_189142, schema="wam.transfer25_throughput/1", units_timed=1,
                   units_succeeded=1)
    done = _classify(tmp_path, payload)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "qualified: 1 successful unit" in done.stdout


def test_a_fresh_timing_submission_refuses_a_pre_existing_artifact_before_it_can_be_reused() -> None:
    """The bash half: the disqualification check runs first, and no flag reaches past it.

    Order is the whole property here. REUSE_THROUGHPUT exists so an operator can deliberately
    re-read a measurement; it must not be able to re-read job 189142's, because that file is not a
    measurement someone forgot to re-read.
    """
    text = _text(_RESTYLE)
    branch = text[text.index("if (( TIMING_MODE )); then"):]
    verdict = branch.index("if (( TP_VERDICT != 0 )); then")
    reuse = branch.index('if [[ "${REUSE_THROUGHPUT:-0}" != "0" ]]; then')
    requeue = branch.index("if (( TIMING_RESTARTS >= 1 )); then")
    assert verdict < requeue < reuse, (
        "the qualification check must precede both the requeue guard and REUSE_THROUGHPUT"
    )
    assert "REUSE_THROUGHPUT does not override it" in branch
    # And a fresh submission that finds a QUALIFIED artifact still refuses, naming both exits.
    assert "this is a FRESH timing submission" in branch
    assert "pass a fresh RUN_ID" in branch


def test_the_headers_own_timing_recipe_passes_an_explicit_run_id() -> None:
    """The recipe is what gets pasted. Without a RUN_ID it lands on the default path, which is
    where the disqualified artifact lives — the trap and the instructions were the same line."""
    header = _text(_RESTYLE)[: _text(_RESTYLE).index("#SBATCH") if "#SBATCH" in _text(_RESTYLE) else 4000]
    recipe = re.search(r"TIMING=1 STAGE=1 [^\n]*\n(?:#[^\n]*\n){0,3}", header)
    assert recipe, "the header no longer carries a TIMING=1 recipe"
    assert "RUN_ID=" in recipe.group(0), (
        "the TIMING=1 recipe passes no RUN_ID, so it resolves to the default one — which still "
        "holds job 189142's disqualified THROUGHPUT.json"
    )


def test_the_projection_multiplies_by_the_counted_corpus_and_not_by_a_coined_number() -> None:
    """PR-08 §8 item 3 forbids inventing numbers in exactly this artifact.

    172_000 was a rounded stand-in for a number the job already had: partition_facts.json carries
    corpus_frames, summed over the manifest at expansion time, and the corpus is 171 625 frames.
    The 0.2 % gap is not the argument — the argument is that a budget derived from a measured rate
    has to be derived over a counted corpus, or nothing downstream can check the derivation.
    """
    text = _text(_RESTYLE)
    assert not re.search(r"FRAMES_PER_VARIANT\s*=\s*172_?000", text), (
        "the coined frame count is back. The prose above it may cite 172_000 as the number that "
        "was there; the assignment may not be it."
    )
    assert 'CORPUS_FRAMES=$(python -c' in text and '"corpus_frames"' in text
    assert "FRAMES_PER_VARIANT = int(corpus_frames)" in text
    assert '"frames_per_variant_source"' in text, (
        "the artifact must say where its frame count came from; it is the field that makes the "
        "projection checkable without the job log"
    )


def _timing_writer() -> str:
    return _py_block(_text(_RESTYLE), "FRAMES_PER_VARIANT = int(corpus_frames)")


def _troot(tmp_path: pathlib.Path, *statuses: str) -> pathlib.Path:
    root = tmp_path / "timing_raw"
    for i, status in enumerate(statuses):
        unit = root / f"unit{i}"
        unit.mkdir(parents=True)
        (unit / "sample_outputs.json").write_text(json.dumps({"status": status, "unit": f"unit{i}"}))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_throughput(tmp_path: pathlib.Path, troot: pathlib.Path, corpus_frames: str = "171625"):
    out = tmp_path / "THROUGHPUT.json"
    unit = json.dumps({"unit": "episode_000093__train-01__r00", "episode": "episode_000093",
                       "frames": 448, "style": "train-01", "repeat": 0, "seed": 7001})
    return out, _run_block(
        _timing_writer(), unit, "900", str(out), "nvidia/Cosmos-Transfer2.5-2B", "deadbeef",
        "", "depth:0.5,seg:0.5", corpus_frames, str(troot),
    )


def test_the_throughput_artifact_records_the_corpus_it_projected_over(tmp_path: pathlib.Path) -> None:
    out, done = _write_throughput(tmp_path, _troot(tmp_path, "success"))
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads(out.read_text())
    assert report["frames_per_variant"] == 171625, "the projection did not use the counted corpus"
    assert report["units_succeeded"] == 1 and report["units_timed"] == 1
    assert report["schema"] == "wam.transfer25_throughput/1"
    # 900 s / 448 frames x 171 625 frames / 3600 -- recomputed here rather than copied, because a
    # projection nobody can reproduce from the artifact's own fields is not a derivation.
    assert report["gpu_hours_per_variant"] == round(900 / 448 * 171625 / 3600.0, 2)


def test_a_timing_run_whose_unit_died_writes_no_artifact_at_all(tmp_path: pathlib.Path) -> None:
    """The second, independent assertion that a clip exists — the first is --require-success.

    Job 189142 had neither, and its wall clock around a crash became the budget input. This one is
    the important half because it lands IN the file: it is what lets every later reader decide
    whether the number is a measurement without knowing the job number.
    """
    out, done = _write_throughput(tmp_path, _troot(tmp_path, "error"))
    assert done.returncode != 0
    assert "0 successful unit record" in (done.stdout + done.stderr)
    assert not out.exists(), "a THROUGHPUT.json was written for a run that generated nothing"


def test_a_timing_run_that_produced_two_clips_is_refused_as_not_the_measurement_asked_for(
    tmp_path: pathlib.Path,
) -> None:
    """"one timed episode on an H200" is the registered measurement, and a wall clock around two
    episodes divided by one episode's frame count is a number about nothing."""
    out, done = _write_throughput(tmp_path, _troot(tmp_path, "success", "success"))
    assert done.returncode != 0
    assert "2 successful unit record" in (done.stdout + done.stderr)
    assert not out.exists()


def _refusal_advice() -> str:
    return _py_block(_text(_RESTYLE), "G0c REFUSED THIS UNIT ON ITS SOURCE MASKS")


def test_a_timing_run_refused_before_generation_names_the_owner_decision_and_makes_none(
    tmp_path: pathlib.Path,
) -> None:
    """The refusal is cheap now; the DECISION behind it is still not this job's to make.

    The driver's source-mask preflight turns "half a GPU-hour and no artifact" into "seconds and no
    artifact", which changes the price of the failure and nothing else. What the operator does next
    — whether an empty robot mask is acceptable for a timed measurement, and which of the 402
    episodes the measurement runs on — is open, and reaching any episode already needs no code
    change (CHUNK_INDEX/CHUNK_TOTAL). So the job says so, and picks nothing.
    """
    root = tmp_path / "timing_raw" / "episode_000000__train-01__r00"
    root.mkdir(parents=True)
    (root / "sample_outputs.json").write_text(json.dumps({
        "status": "error", "unit": "episode_000000__train-01__r00", "episode": "episode_000000",
        "detail": "SourceMaskRefusal: ... the robot mask is EMPTY on frame 0.",
        "g0c": {"composited": False, "refused_before_generation": True},
    }))
    done = _run_block(_refusal_advice(), str(tmp_path / "timing_raw"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "REFUSED THIS UNIT ON ITS SOURCE MASKS" in done.stdout
    assert "episode_000000" in done.stdout, "the operator is not told which episode refused"
    assert "THIS IS NOT A RETRY" in done.stdout
    assert "registered rule version" in done.stdout
    # And it must not choose: no episode is recommended anywhere in the job, by id.
    assert not re.search(r"episode_0*\d+", _text(_RESTYLE)), (
        "97 names a specific episode. Choosing the timed episode because it survives G0c is a "
        "selection made after seeing the data and needs a rule version, not a line in an sbatch."
    )


def test_the_refusal_advice_stays_silent_for_an_ordinary_failed_unit(tmp_path: pathlib.Path) -> None:
    """A unit that died in the generator is a different failure and must not be given G0c's
    explanation — the operator would go and read a gate that had nothing to do with it."""
    root = tmp_path / "timing_raw" / "u0"
    root.mkdir(parents=True)
    (root / "sample_outputs.json").write_text(json.dumps({
        "status": "error", "unit": "u0",
        "g0c": {"composited": False, "refused_before_generation": False},
    }))
    done = _run_block(_refusal_advice(), str(tmp_path / "timing_raw"))
    assert done.returncode == 0
    assert done.stdout.strip() == ""


def test_one_qualification_check_serves_both_paths_that_read_the_throughput_artifact() -> None:
    """The TIMING path asks before REUSING a measurement; GENERATE asks before DERIVING from one.

    Two copies of that question could answer differently about the same file, which is the failure
    mode of every duplicated gate — so it is a shell function defined once. The generation side is
    the one that spends the allocation: its ceiling gate reads exactly `seconds_per_frame` out of
    this file, and job 189142's artifact answers 0.2 from a run that produced no clip. There is no
    override on that side, deliberately: a GPU-h ceiling is not a thing to be waived on a submit
    line.
    """
    text = _text(_RESTYLE)
    assert text.count("throughput_qualification() {") == 1, (
        "two definitions of the qualification check can answer differently about one file"
    )
    calls = [m.start() for m in re.finditer(r'throughput_qualification "\$\{THROUGHPUT\}"', text)]
    assert len(calls) == 2, f"expected the timing and the generation caller, found {len(calls)}"

    generate = text.index("\n# GENERATE\n")
    assert min(calls) < generate < max(calls), (
        "one of the two paths that reads THROUGHPUT.json does not ask whether it can be built on"
    )
    tail = text[generate:]
    assert "REUSE_THROUGHPUT" not in tail, "the generation path must offer no override"
    assert "spend the allocation under a budget line that measures nothing" in tail
