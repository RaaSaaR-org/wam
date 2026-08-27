"""`scripts/run_v17_arms.sh` — the artifact-skip policy, and the override that makes it usable.

T40_RULE_V17's thirteen measurements are driven by one script so that one estimator on one device
produces all of them. That script had two properties which are harmless apart and expensive
together: the output directory was a bare assignment with no override, and any artifact already on
disk was skipped with `SKIP <stem> (measured)`.

THE DAY THAT COSTS SOMETHING IS THE DAY `apple_sam2.GATE_QUALIFIED` FLIPS. Every artifact under
`runs/pr08-est-drift/v17` was measured while the flag was False and carries `gate_qualified: false`;
the re-run exists precisely to replace them. With the old script that re-run skipped all thirteen in
about a second, kept the stale files, printed nothing that said so, and the carry downstream then
refused for reasons that had stopped describing reality. The remediation on record — "point it at a
fresh directory" — was not reachable: there was no flag and no environment override.

So these tests hold three properties, and they are behavioural rather than textual wherever the
behaviour can be reached without a GPU: an existing artifact stops the script instead of being
skipped; the operator's way out is nameable on the command line; and an artifact whose recorded
`gate_qualified` disagrees with the running adapter's is called out by name, because that is the
case that actually bites.

Nothing here measures anything. Every invocation is arranged to refuse, or to give up at the GPU
wait, long before `measure_est_drift` is reached.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "run_v17_arms.sh"

#: Enough to make `headroom_holds` fail on its first sample and `wait_for_gpu` give up immediately,
#: so an invocation that gets PAST the pre-flight still measures nothing. Without this a test that
#: the pre-flight admits a run would start a three-hour GPU job.
_NEVER_ENOUGH_GPU = ["--min-free-mib", "999999999", "--wait-minutes", "0"]


def _text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _captures(root: pathlib.Path) -> pathlib.Path:
    """The capture directories the script reads. They are INPUTS and live under the output dir."""
    for name in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "C2-t20", "C2-t40", "C2-t80"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _artifact(path: pathlib.Path, *, gate_qualified: bool, adapter: bool | None) -> None:
    doc: dict = {"schema": "wam.est_drift/1", "gate_qualified": gate_qualified}
    if adapter is not None:
        doc["estimator_stats"] = {"adapter": {"gate_qualified": adapter}}
    path.write_text(json.dumps(doc))


def _run(out_dir: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(out_dir),
        "V17": str(out_dir),
        "CENSUS_OUT": str(out_dir / "CENSUS.json"),
    }
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(_REPO),
    )


def test_the_script_parses():
    """A refusal that cannot be shell-parsed is not a refusal."""
    assert subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True).returncode == 0


def test_the_output_directory_is_overridable_rather_than_a_bare_assignment():
    """The remediation the defect report proposed has to be REACHABLE.

    `V17=runs/pr08-est-drift/v17` accepts nothing from the environment and the flag parser accepted
    only `--min-free-mib` / `--wait-minutes`, so "point it at a fresh directory" was advice that
    could not be followed without editing the script.
    """
    text = _text()
    assert 'V17="${V17:-runs/pr08-est-drift/v17}"' in text
    assert "\nV17=runs/" not in text, "a bare assignment cannot be overridden"
    assert "--out-dir" in text
    assert 'CENSUS_OUT="${CENSUS_OUT:-' in text, (
        "the V18 census this script also writes is skipped by the same policy and must be "
        "redirectable by the same means, or --out-dir moves twelve of thirteen artifacts"
    )


def test_an_existing_artifact_stops_the_script_instead_of_being_skipped(tmp_path):
    """THE DEFECT, AS A TEST. One artifact on disk, and the script refuses to start.

    It refuses rather than choosing, because whether an artifact on disk is a measurement to keep or
    a stale file to replace is not decidable from the file system — it is the operator's call, and
    the whole failure mode is that the old script answered it silently and always the same way.
    """
    out = _captures(tmp_path)
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=False)
    done = _run(out, *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "REFUSING TO SKIP" in done.stderr
    for named_exit in ("--reuse-existing", "--remeasure", "--out-dir"):
        assert named_exit in done.stderr, "the refusal must name the ways out, not just refuse"
    assert "Nothing was measured" in done.stderr
    assert not (out / "POOLED.json").exists()


def test_the_refusal_happens_before_the_gpu_wait(tmp_path):
    """Up to twelve hours of waiting sits between the start of this script and its first
    measurement. A refusal discovered after that wait is a refusal discovered at the worst possible
    moment, so the pre-flight runs first — fail closed, fail fast, fail cheap."""
    out = _captures(tmp_path)
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=False)
    done = _run(out, *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5
    assert "waiting for GPU" not in done.stdout
    assert "GIVING UP" not in done.stderr


def test_the_skip_can_be_forced_and_then_says_it_did_not_measure(tmp_path):
    """`--reuse-existing` is the old behaviour, now spoken out loud rather than assumed.

    The wording matters as much as the flag: `SKIP <stem> (measured)` claimed a measurement that did
    not happen, which is how a stale artifact travels into a pool without anyone deciding it should.
    """
    out = _captures(tmp_path)
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=False)
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    combined = done.stdout + done.stderr
    assert "REFUSING" not in combined, combined
    assert "--reuse-existing: the artifacts above are kept" in done.stdout
    assert "NOT measured by this run" in _text(), (
        "the per-step message must not claim the skipped artifact was measured"
    )
    assert "SKIP ${stem} (measured)" not in _text()


def test_an_artifact_whose_gate_qualified_disagrees_with_the_adapter_is_named(tmp_path):
    """THE CASE THAT ACTUALLY BITES, and the reason the pre-flight reads the files at all.

    An artifact carries the adapter's `gate_qualified` as it stood when the file was written. When
    that disagrees with the flag in the module this invocation would drive, the file is not a
    measurement by this instrument — and `pool_est_drift_arms._instrument_key` cannot notice, since
    it keys on the segmenter contract, the resolution, the object class, the propagator spec and the
    IoU threshold, and the gate flag is in none of them.

    The direction exercised here is an artifact claiming True against a module reading False, which
    is the one available while `GATE_QUALIFIED` is False. The direction that will fire after the
    flip is the mirror of it — the module True, the artifacts on disk False — and is the same
    comparison in the same expression.
    """
    out = _captures(tmp_path)
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=False)
    _artifact(out / "EST_DRIFT-A2.json", gate_qualified=True, adapter=True)
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "STALE" in done.stdout and "EST_DRIFT-A2.json" in done.stdout
    assert "agrees" in done.stdout and "EST_DRIFT-A1.json" in done.stdout, (
        "an artifact that DOES agree must be reported as agreeing, or the check is just a refusal"
    )
    assert "REFUSING" in done.stderr
    assert "--remeasure" in done.stderr and "--out-dir" in done.stderr


def test_the_pre_flight_reports_the_adapters_flag_even_when_nothing_is_stale(tmp_path):
    """The comparison is printed on every run, not only on the failing one. It is the fact that
    decides between reusing and re-measuring, and it is invisible from the file names."""
    out = _captures(tmp_path)
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=False)
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert "estimators.apple_sam2.GATE_QUALIFIED is" in done.stdout
    assert "16 artifacts expected" in done.stdout, (
        "every artifact the script writes must be pre-flighted, including ARM_DIVERGENCE, POOLED "
        "and the V18 census — a partial list is a silent skip with extra steps"
    )


def test_a_fresh_directory_that_lacks_the_captures_is_refused_not_measured(tmp_path):
    """`--out-dir` moves the INPUTS too: V17 §2's captures live under the same directory. An empty
    one would measure nothing at all and finish by printing ALL V17 MEASUREMENTS DONE."""
    empty = tmp_path / "fresh"
    empty.mkdir()
    done = _run(empty, *_NEVER_ENOUGH_GPU)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "is not a directory" in done.stderr
    assert "Nothing was measured" in done.stderr


def test_an_unreadable_artifact_counts_as_a_disagreement_not_as_a_measurement(tmp_path):
    """A truncated JSON file is the one case where "it exists" and "it is a measurement" come apart
    most obviously. It must not be reusable in silence."""
    out = _captures(tmp_path)
    (out / "EST_DRIFT-A1.json").write_text("{not json")
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5
    assert "UNREADABLE" in done.stdout


@pytest.mark.parametrize("flag", ["--remeasure", "--reuse-existing"])
def test_both_named_exits_are_accepted_as_flags(flag, tmp_path):
    """Neither may fall through to `unknown flag`, which is exit 2 and a different message from the
    one the refusal above tells the operator to expect."""
    out = _captures(tmp_path)
    done = _run(out, flag, *_NEVER_ENOUGH_GPU)
    assert "unknown flag" not in done.stderr, done.stderr
    assert done.returncode != 2, done.stdout + done.stderr


def test_every_step_and_every_summary_reads_the_chosen_directory():
    """--out-dir has to move ALL of it, or it moves the failure instead of the run.

    Four steps skipped on `-f` and one summary that globbed the default path by name: any one of
    them left hard-coded turns `--out-dir` into a run that measures into a fresh directory and then
    reports, or reuses, the old one's files. Read off the source because reaching these lines needs
    the GPU and three hours; the behaviour that CAN be reached cheaply is tested above.
    """
    text = _text()
    assert 'if [[ ! -f "${V17}' not in text, "a raw -f skip bypasses the recorded decision"
    assert 'if [[ ! -f runs/' not in text
    assert text.count("already_measured") == 5, "one definition and four skip sites"
    assert 'pathlib.Path("runs/pr08-est-drift/v17")' not in text, (
        "the C2 ladder summary must read the directory this invocation wrote, not the default one"
    )
    # The two id lists exist once and are used by both the pre-flight list and the steps, so
    # "already measured" cannot mean one set of artifacts in one place and another set elsewhere.
    assert text.count("ARM_A_IDS=(") == 1 and text.count("LADDER_IDS=(") == 1
    assert text.count('"${ARM_A_IDS[@]}"') >= 3 and text.count('"${LADDER_IDS[@]}"') >= 3


# -- the pre-flight has to read the flag WHERE EACH WRITER PUTS IT ---------------------------------
#
# The comparison above is only as good as the place it looks. This script drives three different
# writers and they do not agree on where `apple_sam2.stats()`'s adapter block goes:
#
#   measure_est_drift.py             estimator_stats.adapter.gate_qualified  (the 13 EST_DRIFT-*)
#   measure_arm_divergence.py        estimators.gate_qualified               (ARM_DIVERGENCE.json)
#   census_operating_point_episode   estimator_stats.gate_qualified          (the V18 census)
#                                    estimator.gate_qualified                (its summary block)
#
# Read off the committed artifacts on 2026-08-27, not assumed. A probe that reads only the first is
# not a weaker check on the other two — it is NO check on them, and it reports "agrees" for a file
# it never compared. That matters most for the V18 census, which is the OTHER precondition on
# GATE_QUALIFIED: with the single-path probe it stayed "agrees" across the flip, so
# `--reuse-existing` would have carried a pre-flip census forward in silence, which is the whole
# defect this pre-flight was written to close.
#
# The direction exercised is an artifact claiming True against a module reading False, because that
# is the one reachable while `GATE_QUALIFIED` is False; the post-flip direction is the same
# expression with the operands swapped.


def test_a_census_shaped_artifact_is_compared_at_the_key_the_census_actually_writes(tmp_path):
    """The V18 census records no top-level `gate_qualified` at all, so it is invisible to a probe
    that reads only `estimator_stats.adapter` plus the artifact's own outcome."""
    out = _captures(tmp_path)
    (out / "CENSUS.json").write_text(
        json.dumps(
            {
                "schema": "wam.operating_point_census/1",
                "estimator": {"gate_qualified": True},
                "estimator_stats": {"gate_qualified": True},
            }
        )
    )
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "STALE" in done.stdout and "CENSUS.json" in done.stdout
    assert "REFUSING" in done.stderr


def test_an_arm_divergence_shaped_artifact_is_compared_at_its_own_key(tmp_path):
    """`measure_arm_divergence` writes the adapter block under `estimators`, and this artifact is
    one of the two the pool reads directly."""
    out = _captures(tmp_path)
    (out / "ARM_DIVERGENCE.json").write_text(json.dumps({"estimators": {"gate_qualified": True}}))
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "STALE" in done.stdout and "ARM_DIVERGENCE.json" in done.stdout


def test_an_artifact_recording_no_gate_flag_is_not_reported_as_agreeing(tmp_path):
    """"Nothing was found" is not "the instrument matches".

    A file whose writer recorded no adapter flag anywhere establishes nothing about the instrument
    that produced it, and `pool_est_drift_arms._instrument_key` cannot establish it either. Printing
    `agrees` there asserts a comparison that did not happen, and `--reuse-existing` would then hand
    the file on. Fail closed: it is a disagreement.
    """
    out = _captures(tmp_path)
    (out / "EST_DRIFT-A1.json").write_text(json.dumps({"schema": "wam.est_drift/1"}))
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert "UNCHECKABLE" in done.stdout
    assert "agrees" not in done.stdout


def test_the_pool_is_recomputed_every_run_rather_than_reused(tmp_path):
    """POOLED.json is the one artifact here with no skip, and the pre-flight has to say so.

    The pooling step runs unconditionally on purpose: a pool carried over from an earlier run would
    omit whichever capture THIS run was asked to measure. So the file is listed (it is about to be
    overwritten, and the default refusal is also a warning about that) but it is not compared, and
    it must not be able to block `--reuse-existing` on the ground that it records no adapter flag —
    it never does, and it is not being reused.
    """
    out = _captures(tmp_path)
    (out / "POOLED.json").write_text(json.dumps({"schema": "wam.est_drift_pooled/1"}))
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    combined = done.stdout + done.stderr
    assert "rewritten" in done.stdout, combined
    assert "UNCHECKABLE" not in done.stdout, combined
    assert done.returncode != 5, combined
    assert "is NOT kept" in done.stdout, (
        "the reuse message must not claim POOLED.json is kept when the pooling step rewrites it"
    )
