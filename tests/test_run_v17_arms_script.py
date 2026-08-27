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

**THAT DAY IS 2026-08-27: `GATE_QUALIFIED` IS NOW True.** The blocker tuple reached `()` and the
project owner decided T40_RULE_V18 §3 outcome C for blocker 2's residue (i), in
`docs/preregistration/PR-08-RESULT-2026-08-27-residue-i-is-contained-and-the-flag-flips.md`.
Nothing about this script changed with it and nothing here was relaxed. What changed is which side
of the comparison the artifacts on disk are on: the thirteen pre-flip files now DISAGREE with the
running adapter, the pre-flight says STALE, and `--reuse-existing` refuses. That refusal is the
feature, on the exact day it was written for.

READ THAT NARROWLY. THIS FILE IS NOT EVIDENCE THAT THE GATE IS OPEN AND NOTHING BELOW TESTS THAT
IT IS. The flag is one input to `gate_qualified` and qualifies no artifact by itself:
`measure_geom_tol.sam2_method` still requires the checkpoints and the contract too. T40_RULE_V1 §1
is not lifted. PR-08 §8 items 3 and 4 are open. Blocker 1 was not touched by the determination, so
NOBODY HAS LOOKED at a mask — the area test is a proxy for a wrong-object mask, not an observation
of one. Five frames of the 473-vs-478 gap are unexplained and the decode hypothesis for them is
refuted. Everything below is about the PROVENANCE of a file on disk, which instrument wrote it, and
never about whether that instrument's numbers were right. The tripwires on the flip's legitimacy
live with the module, in `tests/test_apple_sam2_estimator.py`,
`tests/test_apple_sam2_video_propagation.py` and `tests/test_audit_apple_masks.py`.

The fixtures below are therefore built RELATIVE to the flag the script actually reads
(:func:`_live_gate_qualified`) and never pinned to a literal True or False. A fixture pinned to a
literal is a fixture that silently changes meaning the next time the flag moves — which is not
hypothetical in either direction: the module's own comment lists what would take the flip back.
"Agrees with the running adapter" and "was written on the other side of the flip" are the two things
these tests need to say, and each is said as a function of the live flag.

So these tests hold four properties, and they are behavioural rather than textual wherever the
behaviour can be reached without a GPU: an existing artifact stops the script instead of being
skipped; the operator's way out is nameable on the command line; an artifact whose recorded
`gate_qualified` disagrees with the running adapter's is called out BY NAME AND BY VERDICT, because
that is the case that actually bites; and an artifact from the other side of the flip is never
reusable, which is the protection the flip actually bought.

Nothing here measures anything. Every invocation is arranged to refuse, or to give up at the GPU
wait, long before `measure_est_drift` is reached.
"""

from __future__ import annotations

import functools
import json
import pathlib
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "run_v17_arms.sh"
#: The interpreter the script's own pre-flight runs, so the flag read here is the flag it compares.
_VENV_PYTHON = _REPO / ".venv" / "bin" / "python"

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


@functools.lru_cache(maxsize=1)
def _live_gate_qualified() -> bool:
    """`GATE_QUALIFIED` AS THE SCRIPT READS IT — same module, same interpreter, same import path.

    Every fixture below is a function of this and never of a literal. The flag flipped False -> True
    on 2026-08-27 and the module's own comment names what would take it back, so a fixture spelled
    `gate_qualified=False` does not mean "agrees with the adapter" for longer than one flip: it
    meant that before, it means the opposite now, and the test keeps passing while guarding the
    other case. Reading the live value is also itself a check — a `GATE_QUALIFIED` that stopped
    being a bool would make the pre-flight's comparison meaningless, so that is asserted here.
    """
    probe = (
        "import importlib, sys; sys.path.insert(0, 'scripts'); "
        "print(repr(getattr(importlib.import_module('estimators.apple_sam2'), "
        "'GATE_QUALIFIED', None)))"
    )
    done = subprocess.run(
        [str(_VENV_PYTHON), "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(_REPO),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    value = done.stdout.strip()
    assert value in ("True", "False"), (
        f"the pre-flight compares every artifact against GATE_QUALIFIED and read {value!r}; a "
        "non-bool there makes 'agrees' and 'STALE' claims about nothing"
    )
    return value == "True"


def _artifact(path: pathlib.Path, *, gate_qualified: bool, adapter: bool | None) -> None:
    doc: dict = {"schema": "wam.est_drift/1", "gate_qualified": gate_qualified}
    if adapter is not None:
        doc["estimator_stats"] = {"adapter": {"gate_qualified": adapter}}
    path.write_text(json.dumps(doc))


def _artifact_agreeing_with_the_adapter(path: pathlib.Path) -> None:
    """An EST_DRIFT artifact that WAS written by the adapter this invocation drives.

    Both recorded flags follow the live one, so this keeps meaning "agrees" whichever way the flag
    stands: the pre-flight's rule is `recorded != live` in either direction, plus the one-sided
    `live is True and artifact.gate_qualified is False`, and this fixture is on the right side of
    both by construction.
    """
    live = _live_gate_qualified()
    _artifact(path, gate_qualified=live, adapter=live)


def _artifact_from_before_the_flip(path: pathlib.Path) -> None:
    """An artifact written on the OTHER SIDE of the flip, i.e. the thirteen files really on disk.

    Today that is `gate_qualified: false` against a module reading True — what all thirteen
    `EST_DRIFT-*.json` under `runs/pr08-est-drift/v17` carry, and `ARM_DIVERGENCE.json` beside them.
    (`POOLED.json` records no adapter flag at all; it is the pre-flight's `rewritten` case, not this
    one.) Expressed as the negation of the live flag so it still means "written by a different
    instrument" if the flag is ever taken back to False.
    """
    live = _live_gate_qualified()
    _artifact(path, gate_qualified=not live, adapter=not live)


def _verdict(stdout: str, name: str) -> str:
    """The pre-flight's verdict for the listed artifact whose path ends in `name`.

    Asserting `"STALE" in stdout and name in stdout` passes when the two belong to DIFFERENT lines,
    which is exactly what happens when a flip swaps which fixture is stale: the assertion outlives
    the swap and stops testing anything. The verdict is read off the artifact's own line instead.
    """
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].endswith("/" + name):
            return fields[0]
    raise AssertionError(f"the pre-flight never listed {name}:\n{stdout}")


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

    The artifact here AGREES with the running adapter, which is the harder half of the property: the
    default mode refuses on existence alone, not on staleness, so a run that found only agreeing
    artifacts still stops and still makes the operator say which of the three exits they want.
    """
    out = _captures(tmp_path)
    _artifact_agreeing_with_the_adapter(out / "EST_DRIFT-A1.json")
    done = _run(out, *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert _verdict(done.stdout, "EST_DRIFT-A1.json") == "agrees", done.stdout
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
    _artifact_agreeing_with_the_adapter(out / "EST_DRIFT-A1.json")
    done = _run(out, *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5
    assert "waiting for GPU" not in done.stdout
    assert "GIVING UP" not in done.stderr


def test_the_skip_can_be_forced_and_then_says_it_did_not_measure(tmp_path):
    """`--reuse-existing` is the old behaviour, now spoken out loud rather than assumed.

    The wording matters as much as the flag: `SKIP <stem> (measured)` claimed a measurement that did
    not happen, which is how a stale artifact travels into a pool without anyone deciding it should.

    WHAT MOVED ON 2026-08-27. This fixture used to be spelled `gate_qualified=False`, which agreed
    with the adapter only for as long as `GATE_QUALIFIED` was False. After the flip that same
    literal is an artifact from the previous instrument, the pre-flight correctly calls it STALE and
    `--reuse-existing` correctly refuses — the script was right and the fixture was stale. So the
    artifact is now built from the live flag: the override under test is "reuse what AGREES", and
    forcing the skip past a disagreement is a different property, tested (as a refusal) below.
    """
    out = _captures(tmp_path)
    _artifact_agreeing_with_the_adapter(out / "EST_DRIFT-A1.json")
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    combined = done.stdout + done.stderr
    assert "REFUSING" not in combined, combined
    assert _verdict(done.stdout, "EST_DRIFT-A1.json") == "agrees", done.stdout
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

    The direction that fires TODAY is the mirror of the one this test was written with: it used to
    build A2 as `True` against a module reading False, and after the 2026-08-27 flip it is the
    module reading True and the file on disk saying False. Same comparison, same expression, sides
    swapped. That is why the fixtures are the live flag and its negation rather than literals,
    and why the verdict is read off each artifact's OWN line. The old form (`"STALE" in stdout and
    "EST_DRIFT-A2.json" in stdout`) still passed after the flip with the two roles exchanged, which
    is a test that survives the event it was watching for without noticing it.
    """
    out = _captures(tmp_path)
    _artifact_agreeing_with_the_adapter(out / "EST_DRIFT-A1.json")
    _artifact_from_before_the_flip(out / "EST_DRIFT-A2.json")
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert _verdict(done.stdout, "EST_DRIFT-A2.json") == "STALE", done.stdout
    assert _verdict(done.stdout, "EST_DRIFT-A1.json") == "agrees", (
        "an artifact that DOES agree must be reported as agreeing, or the check is just a refusal"
    )
    assert "REFUSING" in done.stderr
    assert "--remeasure" in done.stderr and "--out-dir" in done.stderr


def test_artifacts_from_before_the_flip_are_stale_and_cannot_be_reused(tmp_path):
    """THE SITUATION THE WHOLE PRE-FLIGHT WAS BUILT FOR, NOW THAT IT IS THE REAL ONE.

    `GATE_QUALIFIED` flipped on 2026-08-27 and every EST_DRIFT artifact under
    `runs/pr08-est-drift/v17`, plus the `ARM_DIVERGENCE.json` beside them, was written on the other
    side of that flip. This is the state of the world the header predicted: an
    operator re-runs this script precisely BECAUSE the flag moved, and the old script would have
    skipped all thirteen in about a second and kept exactly the files the re-run existed to replace.
    So the property is pinned directly, across all three writers at once and in the mode where it
    bites: a pre-flip set is STALE, `--reuse-existing` refuses it, nothing is measured, nothing on
    disk is touched, and the refusal names the two exits that do produce post-flip artifacts.

    It is written as the negation of the live flag rather than as a literal `false` so that it keeps
    testing the flip rather than the date: the module's own comment lists what would take the flag
    back to False, and on that day these same three files become the pre-flip set again with the
    values exchanged.

    Note what is NOT claimed here. That the flag moved says nothing about whether these artifacts'
    numbers were right; STALE means "not written by the instrument running now", which is a
    statement about provenance and about `pool_est_drift_arms._instrument_key` being unable to see
    the difference — not a verdict on any measurement.
    """
    out = _captures(tmp_path)
    pre_flip = not _live_gate_qualified()
    _artifact_from_before_the_flip(out / "EST_DRIFT-A1.json")
    (out / "ARM_DIVERGENCE.json").write_text(
        json.dumps({"estimators": {"gate_qualified": pre_flip}})
    )
    _census(out / "CENSUS.json", pre_flip)
    names = ("EST_DRIFT-A1.json", "ARM_DIVERGENCE.json", "CENSUS.json")
    before = {name: (out / name).read_text() for name in names}

    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    for name in before:
        assert _verdict(done.stdout, name) == "STALE", done.stdout
    assert "REFUSING" in done.stderr
    for named_exit in ("--remeasure", "--out-dir"):
        assert named_exit in done.stderr, (
            "a refusal that names no way forward is where a flip stalls"
        )
    assert "Nothing was measured" in done.stderr
    assert not (out / "POOLED.json").exists(), "a refusal must not leave a pool behind"
    assert {name: (out / name).read_text() for name in names} == before, (
        "the refusal must leave the pre-flip artifacts exactly as it found them: a refusal to "
        "reuse them, not a decision to discard them"
    )

    # The default mode refuses too, and SAYS a disagreement is what it found: on the day of a flip
    # "these files exist" and "these files are from the previous instrument" are different warnings,
    # and only the second explains why the operator is standing here.
    default = _run(out, *_NEVER_ENOUGH_GPU)
    assert default.returncode == 5, default.stdout + default.stderr
    assert "REFUSING TO SKIP" in default.stderr
    assert "DISAGREES" in default.stderr, default.stderr

    # And the way forward is a re-measurement, not a reuse: --remeasure gets past the pre-flight
    # (and then dies at the GPU wait, which is what _NEVER_ENOUGH_GPU is for).
    forward = _run(out, "--remeasure", *_NEVER_ENOUGH_GPU)
    assert "MEASURED AGAIN" in forward.stdout, forward.stdout + forward.stderr
    assert forward.returncode == 4, forward.stdout + forward.stderr


def test_an_artifact_that_came_out_not_qualified_is_not_reusable_once_the_module_claims_it_is(
    tmp_path,
):
    """The OTHER half of the STALE rule, and the half that only exists after a flip.

    An artifact records two different things: the adapter's flag when the file was written, and
    whether that measurement itself came out gate-qualified. They can differ — a run can be driven
    by a qualified adapter and still write `gate_qualified: false` for reasons of its own — so the
    pre-flight reports both and treats the mismatch as stale in ONE direction: the module now claims
    qualification and the file on disk says it does not have it. Reusing that file would carry a
    not-qualified measurement into a pool assembled under a flag that says otherwise.

    The expectation is deliberately asymmetric, because the rule is. While the module reads False, a
    `gate_qualified: false` artifact is not in tension with anything and is a perfectly ordinary
    measurement; it is the module's True that turns the same file into a contradiction. Written this
    way the test states the asymmetry instead of hiding it behind a literal.
    """
    out = _captures(tmp_path)
    live = _live_gate_qualified()
    _artifact(out / "EST_DRIFT-A1.json", gate_qualified=False, adapter=live)
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    expected = "STALE" if live else "agrees"
    assert _verdict(done.stdout, "EST_DRIFT-A1.json") == expected, done.stdout
    assert (done.returncode == 5) is live, done.stdout + done.stderr


def test_the_pre_flight_reports_the_adapters_flag_even_when_nothing_is_stale(tmp_path):
    """The comparison is printed on every run, not only on the failing one. It is the fact that
    decides between reusing and re-measuring, and it is invisible from the file names.

    The fixture has to be the agreeing one for the test to be about what its name says. Pinned to
    `False` it kept passing after the flip while quietly becoming a second copy of the STALE case,
    so the run really is checked for having no disagreement in it.
    """
    out = _captures(tmp_path)
    _artifact_agreeing_with_the_adapter(out / "EST_DRIFT-A1.json")
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert "estimators.apple_sam2.GATE_QUALIFIED is" in done.stdout
    assert "STALE" not in done.stdout and "UNCHECKABLE" not in done.stdout, done.stdout
    assert done.returncode != 5, done.stdout + done.stderr
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
# Since the 2026-08-27 flip both directions are reachable, so both are exercised for each writer:
# an artifact recording the live flag must come back `agrees`, and one recording its negation must
# come back `STALE`. One direction alone is not enough: a probe that reads none of these keys
# refuses too, as `UNCHECKABLE`, so a lone "it refused" assertion cannot tell "compared and
# disagreed" from "never found a flag to compare", and the second is the bug.


def _census(path: pathlib.Path, flag: bool) -> None:
    """A V18 census as `census_operating_point_episode` writes one: no top-level `gate_qualified`,
    the adapter's flag under `estimator_stats` and again in the `estimator` summary block."""
    path.write_text(
        json.dumps(
            {
                "schema": "wam.operating_point_census/1",
                "estimator": {"gate_qualified": flag},
                "estimator_stats": {"gate_qualified": flag},
            }
        )
    )


def test_a_census_shaped_artifact_is_compared_at_the_key_the_census_actually_writes(tmp_path):
    """The V18 census records no top-level `gate_qualified` at all, so it is invisible to a probe
    that reads only `estimator_stats.adapter` plus the artifact's own outcome.

    BOTH DIRECTIONS ARE EXERCISED, and that is what carries the intent past the flip. The verdict is
    read off the census's own line, so a probe blind to these keys fails here: it would report
    `UNCHECKABLE` for both censuses, and `UNCHECKABLE` is a refusal too — asserting only "exit 5"
    would let a probe that never found the flag pass as though it had compared it. Only a probe that
    reads `estimator_stats.gate_qualified` / `estimator.gate_qualified` can say `agrees` for one and
    `STALE` for the other. The stale one is now the census a pre-flip run left behind, which is the
    file this check was written for: it is the OTHER precondition on `GATE_QUALIFIED`, and carrying
    it silently across the flip is the defect the pre-flight exists to close.
    """
    out = _captures(tmp_path)

    _census(out / "CENSUS.json", _live_gate_qualified())
    agreeing = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert _verdict(agreeing.stdout, "CENSUS.json") == "agrees", agreeing.stdout
    assert "REFUSING" not in agreeing.stderr, agreeing.stderr

    _census(out / "CENSUS.json", not _live_gate_qualified())
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert _verdict(done.stdout, "CENSUS.json") == "STALE", done.stdout
    assert "REFUSING" in done.stderr


def test_an_arm_divergence_shaped_artifact_is_compared_at_its_own_key(tmp_path):
    """`measure_arm_divergence` writes the adapter block under `estimators`, and this artifact is
    one of the two the pool reads directly.

    Same two directions as the census, for the same reason: a probe that does not read `estimators`
    reports `UNCHECKABLE` either way and would pass a one-directional exit-5 assertion while having
    compared nothing. The flags follow the live one, so the STALE case stays "written by the other
    instrument" whichever way the flag stands.
    """
    out = _captures(tmp_path)
    divergence = out / "ARM_DIVERGENCE.json"

    divergence.write_text(json.dumps({"estimators": {"gate_qualified": _live_gate_qualified()}}))
    agreeing = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert _verdict(agreeing.stdout, "ARM_DIVERGENCE.json") == "agrees", agreeing.stdout
    assert "REFUSING" not in agreeing.stderr, agreeing.stderr

    divergence.write_text(
        json.dumps({"estimators": {"gate_qualified": not _live_gate_qualified()}})
    )
    done = _run(out, "--reuse-existing", *_NEVER_ENOUGH_GPU)
    assert done.returncode == 5, done.stdout + done.stderr
    assert _verdict(done.stdout, "ARM_DIVERGENCE.json") == "STALE", done.stdout


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
