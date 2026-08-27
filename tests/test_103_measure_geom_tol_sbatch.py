"""cluster/discoverer/103_measure_geom_tol.sbatch — the runbook, not the estimator.

WHY THIS IS A SEPARATE FILE FROM ``tests/test_measure_geom_tol.py``. That file already carries a
section over this sbatch (``SBATCH_103``), and its helpers are the pattern followed here: run the
sbatch's OWN heredocs, extracted from the file, never a copy pasted into a test — a copy passes
forever after the sbatch changes. The tests below are kept apart from it only because they landed in
a sprint where ``scripts/measure_geom_tol.py`` and its test file were being edited concurrently by
another author, and two writers in one 3 600-line file is how one of them loses work. The anchors
(``_heredoc_after``) are identical, so the two sections can be joined later with no edits.

WHAT IS UNDER TEST, AND WHY EACH ONE COSTS REAL GPU-HOURS IF IT REGRESSES:

  the cost model      Until 2026-08-27 the walltime self-check defaulted to p = 0.18 s/frame and
  is measured         L = 120 s. The N = 16 partition has since RUN, and its sixteen logs give
                      p = 0.2478 and L = 410 — the old pair is 1.50x optimistic and would pass a
                      --time=01:00:00 request for a shard that measurably takes 62.4 min. A
                      self-check that reassures is worse than none; that sentence is in the sbatch
                      and this file is what keeps it true.

  the pre-GPU         Nothing compared the running adapter's SEGMENTER_CONTRACT against the
  preflight           committed configs/transfer25/pr08_geom_tol.json before the decode. Twelve of
                      the last sixteen shards ran a stale adapter and all sixteen were missing a
                      contract field; the array was discovered unusable at the merge, one array and
                      13.6 GPU-h later. Same block refuses while apple_sam2.GATE_QUALIFIED is
                      False, because gate_qualified is baked into each shard at measurement time.

  the recovery        The refusal messages used to print `git -C ${WAM} checkout -- ...`, which
  command runs        cannot run on the cluster: sync.sh:66 rsyncs with --exclude '.git'.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_103 = _REPO_ROOT / "cluster/discoverer/103_measure_geom_tol.sbatch"
TEXT = SBATCH_103.read_text(encoding="utf-8")

#: The line that opens the pre-GPU preflight heredoc. Anchored on the waiver's env var, because the
#: waiver's NAME is part of what is under test: it has to say what is being waived.
PREFLIGHT = 'WAIVED="${GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT:-0}" python - <<\'PY\''

#: The walltime self-check's opening line, spelled exactly as tests/test_measure_geom_tol.py spells
#: it, so the two files cannot drift about which block they mean.
SELFCHECK = "ALLOW_TIGHT=\"${GEOM_ALLOW_TIGHT_WALL:-0}\" python - <<'PY'"


def _heredoc_after(anchor: str) -> str:
    """The first ``<<'PY' ... PY`` block at or after ``anchor``. Refuses rather than guessing."""
    at = TEXT.index(anchor)
    start = TEXT.index("<<'PY'\n", at) + len("<<'PY'\n")
    end = TEXT.index("\nPY\n", start)
    return TEXT[start:end]


def _run_snippet(source: str, argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", source, *argv],
                          capture_output=True, text=True,
                          env={**os.environ, **env})


# -- the cost model: measured, not planned ---------------------------------------------------------
#
# MEASURED 2026-08-27 from the previous array's own logs, and re-derivable without this file:
#
#   for f in runs/_slurm_logs/geom-tol.1899*_*.out runs/_slurm_logs/geom-tol.190125_*.out; do
#     grep -o 'exited [0-9]* after [0-9]*s' "$f" | head -1; done \
#     | awk -F'[ s]' '{s+=$4} END {print NR, s, s/3600}'      # -> 16  49091  13.6364
#
# Each of those logs also prints its shard's exact frame count, so the sixteen (frames, seconds)
# pairs give the model directly. Least squares: p = 0.247793 s/frame, L = 410.219 s, and because
# least-squares residuals sum to zero, THAT pair reproduces 49 091 s exactly. The four figures the
# sbatch carries are a rounding of it and land 2 s under, at 49 089 s — which is why the assertion
# below has a tolerance at all, and why the sbatch's comment says "rounded" rather than "exactly".

MEASURED_P = 0.2478
MEASURED_L = 410.0
MEASURED_TOTAL_SECONDS = 49_091
CORPUS_FRAMES = 171_625
#: Shard 5 of the N = 16 partition — the heaviest, and the one that decides whether an array
#: finishes. runs/_slurm_logs/geom-tol.189971_5.out: 14 162 frames, "exited 3 after 3741s".
HEAVIEST_SHARD_FRAMES = 14_162
HEAVIEST_SHARD_SECONDS = 3_741


def _default_of(name: str) -> str:
    """The ``${NAME:-default}`` the sbatch assigns to ``NAME``, as written."""
    m = re.search(r"^\s*%s=\$\{%s:-([^}]+)\}\s*$" % (name, name), TEXT, re.M)
    assert m, f"{name} no longer has a `${{{name}:-default}}` assignment in the sbatch"
    return m.group(1)


def test_the_fallback_cost_model_is_the_measured_pair_and_not_the_planning_constants() -> None:
    """THE DEFECT: (0.18, 120) is 1.50x optimistic and the array it sizes has already run.

    0.18 was 189658's measured FLOOR rounded up and 120 was the discredited pilot's load. Sixteen
    real shards now say 0.2478 and 410. Pinning the literals here rather than only the behaviour
    below, because an operator reads these two lines out of the file when writing the ``--time``.
    """
    assert float(_default_of("GEOM_SECONDS_PER_FRAME")) == pytest.approx(MEASURED_P, abs=5e-5)
    assert float(_default_of("GEOM_LOAD_SECONDS")) == pytest.approx(MEASURED_L, abs=0.5)


def test_the_measured_pair_reproduces_the_arrays_own_total() -> None:
    """The pair is the total, not an approximation of it — that is why it is the fitted pair and
    not the blended 0.29 the sprint sheet floated. Checked as arithmetic so a future edit that
    "rounds" one of the two has to explain the 49 091 s it no longer reproduces."""
    p = float(_default_of("GEOM_SECONDS_PER_FRAME"))
    load = float(_default_of("GEOM_LOAD_SECONDS"))
    assert p * CORPUS_FRAMES + 16 * load == pytest.approx(MEASURED_TOTAL_SECONDS, abs=30)
    assert (p * CORPUS_FRAMES + 16 * load) / 3600 == pytest.approx(13.636, abs=0.01)


def test_the_planning_table_states_the_measured_total_and_not_the_old_one() -> None:
    """The header's N-vs-cost table is what an operator prices the submission against.

    Its N = 16 row said 9.115 GPU-h against a measured 13.636 — under by 4.52 GPU-h, and under by
    40 % on the per-shard column, which is the column that kills a job.
    """
    row = re.search(r"^#\s+16\s+14 162 f\s+1\.320\s+(\S+)\s+(\S+)\s+([\d.]+)\s*$", TEXT, re.M)
    assert row, "the N = 16 row of the planning table is gone or reshaped"
    assert float(row.group(3)) == pytest.approx(13.636, abs=0.01), (
        f"the table still prices N = 16 at {row.group(3)} GPU-h; the array measured 13.636")
    assert "p=0.2478 s/frame + L=410 s" in TEXT, (
        "the planning table's column header no longer names the cost model its rows were computed "
        "with, so a reader cannot tell which pair produced them")
    # The old table's own total. It may still be NAMED — this codebase keeps its errors legible —
    # but only on a line that says it is the number being replaced, never as a live figure.
    for line in TEXT.splitlines():
        if "9.115" in line:
            assert "old" in line, (
                f"9.115 GPU-h is still stated as a live figure, not as the number the measurement "
                f"replaced:\n{line}")


# -- the walltime self-check, driven with the file's own defaults ----------------------------------


def _selfcheck(tmp_path: Path, *, frames: int, remaining: int,
               p: str | None = None, load: str | None = None) -> subprocess.CompletedProcess:
    """Run the sbatch's own self-check over a one-episode, one-shard partition of ``frames``.

    NUM_SHARDS=1 puts every episode in shard 0 whatever ``shard_of`` does, so the frame count under
    test is exactly ``frames`` and the test says nothing about the partition function (which
    tests/test_measure_geom_tol.py already pins against measure_geom_tol.shard_of).
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"id": "episode_000000", "frames": frames}]}))
    import time as _t
    return _run_snippet(_heredoc_after(SELFCHECK), [], {
        "SHARD_INDEX": "0", "NUM_SHARDS": "1",
        "MANIFEST": str(manifest),
        "PILOT_JSON": str(tmp_path / "no-pilot.json"),
        "CONTRACT_FILE": str(tmp_path / "no-contract.json"),
        "P_FALLBACK": p if p is not None else _default_of("GEOM_SECONDS_PER_FRAME"),
        "L_FALLBACK": load if load is not None else _default_of("GEOM_LOAD_SECONDS"),
        "ALLOW_TIGHT": "0",
        "SLURM_JOB_END_TIME": str(int(_t.time()) + remaining),
    })


def test_the_heaviest_shard_refuses_a_one_hour_request(tmp_path) -> None:
    """THE FAILURE THE OLD CONSTANTS WOULD HAVE WAVED THROUGH, in its exact shape.

    Shard 5 of the partition took 3 741 s = 62.4 min. At (0.18, 120) the self-check estimated it at
    44.5 min, x1.25 = 55.6 min, and would have PASSED a --time=01:00:00 request that dies at the
    wall — which is the one failure this check exists for. With the file's own defaults it must
    refuse, and it must refuse for a reason the operator can act on.
    """
    r = _selfcheck(tmp_path, frames=HEAVIEST_SHARD_FRAMES, remaining=3600)
    assert r.returncode == 1, (
        "the self-check passed a one-hour request for a shard that measurably takes 62.4 min:\n"
        + r.stdout + r.stderr)
    assert "REFUSING TO START" in r.stdout
    # And the estimate it refuses on is the measured one, not the planning one.
    m = re.search(r"estimate (\d+)s", r.stdout)
    assert m and abs(int(m.group(1)) - HEAVIEST_SHARD_SECONDS) < 400, (
        f"the estimate {m.group(1) if m else '?'}s is not close to the {HEAVIEST_SHARD_SECONDS}s "
        f"this shard actually took:\n{r.stdout}")


def test_the_old_constants_would_have_passed_that_same_request(tmp_path) -> None:
    """The counter-example, so the test above is known to be discriminating and not merely strict.

    This is the regression itself, reproduced: hand the check (0.18, 120) and it waves through the
    request that killed the shard. If this ever starts refusing, the two tests have stopped
    measuring different things and one of them is redundant.
    """
    r = _selfcheck(tmp_path, frames=HEAVIEST_SHARD_FRAMES, remaining=3600, p="0.18", load="120")
    assert r.returncode == 0, r.stdout
    assert "REFUSING TO START" not in r.stdout


def test_the_shape_that_actually_ran_still_passes(tmp_path) -> None:
    """N = 16 at --time=01:30:00 is the shape the partition ran and survived under. The corrected
    constants must not refuse it, or the fix has replaced an optimistic model with a useless one:
    65.3 min x 1.25 = 81.7 min against a 90 min request."""
    r = _selfcheck(tmp_path, frames=HEAVIEST_SHARD_FRAMES, remaining=5400)
    assert r.returncode == 0, r.stdout
    assert "REFUSING" not in r.stdout


# -- the pre-GPU contract and gate preflight -------------------------------------------------------
#
# The stub adapter below is a real module on a real PYTHONPATH, shadowing scripts/estimators/, so
# the preflight imports it exactly the way the measurement would. measure_geom_tol.py itself is NOT
# stubbed: the comparison under test is its own contract_disagreements(), and a test that
# reimplemented it would pass over a preflight that had reimplemented it too.

CONTRACT_NOW = {
    "method_name": "grounding-dino+sam2+depth-anything-v2",
    "box_threshold": 0.15,
    "retry_box_threshold": 0.10,
    "object_text_prompt": "apple.",
    "pixel_grid_hw": [480, 640],
    "mask_validity_reference_max_frame_fraction": 0.1,
}


#: Sentinel for "this parameter was not supplied", so that ``None`` can mean the thing that is
#: actually under test — an adapter with no ``SEGMENTER_CONTRACT``, a committed document that is not
#: there at all.
_UNSET = object()


#: What a gate-qualifiable adapter DECLARES it loads, in the shape
#: ``measure_geom_tol._adapter_checkpoints()`` reads. The stub carries it by default because
#: ``sam2_method()`` writes ``gate_qualified = GATE_QUALIFIED and bool(checkpoints) and contract is
#: not None`` — a stub with the flag set and no weights is not a gate-qualified adapter, it is the
#: third refusal below, and a fixture that omitted it would have made every "this passes" test
#: below a test of the wrong module.
CHECKPOINTS_NOW = {
    "grounding_dino": "IDEA-Research/grounding-dino-base@12bdfa31",
    "sam2": "facebook/sam2-hiera-large@e6a8e880",
}


def _scripts_dir(tmp_path: Path, *, contract, gate: bool, version: str = "test-adapter",
                 checkpoints=_UNSET) -> Path:
    """A throwaway ``scripts/`` holding the REAL comparator and a stub ``estimators.apple_sam2``.

    WHY THE WHOLE DIRECTORY AND NOT JUST A STUB PACKAGE AHEAD OF IT ON PYTHONPATH. Importing
    ``measure_geom_tol`` runs ``sys.path.insert(0, <its own repo>/scripts)`` (measure_geom_tol.py,
    just above its ``prepare_cosmos_corpus`` import), so after that import the real
    ``estimators.apple_sam2`` shadows anything further down PYTHONPATH. THAT IS THE CORRECT
    PRODUCTION BEHAVIOUR and the preflight must keep it: the whole point of the block is that it
    resolves the same adapter the measurement will, from the same path, so a preflight that passed
    could not be describing a different module than the run. The test therefore moves the adapter
    rather than trying to out-rank it.

    ``measure_geom_tol.py`` and its one repo-local import are COPIED, not stubbed and not
    symlinked — copied at test time so they cannot drift from the repo, and not symlinked because
    that module resolves its own root through ``Path(__file__).resolve()``, which follows the link
    straight back to the real tree.
    """
    scripts = tmp_path / "scripts"
    (scripts / "estimators").mkdir(parents=True)
    for name in ("measure_geom_tol.py", "prepare_cosmos_corpus.py"):
        (scripts / name).write_bytes((_REPO_ROOT / "scripts" / name).read_bytes())
    (scripts / "estimators" / "__init__.py").write_text("")
    body = [f"ESTIMATOR_VERSION = {version!r}", f"GATE_QUALIFIED = {gate!r}"]
    if contract is not _UNSET:
        body.append(f"SEGMENTER_CONTRACT = {contract!r}")
    weights = CHECKPOINTS_NOW if checkpoints is _UNSET else checkpoints
    if weights is not None:
        body.append(f"ESTIMATOR_CHECKPOINTS = {weights!r}")
    (scripts / "estimators" / "apple_sam2.py").write_text("\n".join(body) + "\n")
    return scripts


def _preflight(tmp_path: Path, *, contract=_UNSET, gate: bool = True,
               committed=_UNSET, committed_doc=_UNSET, shard_out=_UNSET,
               waived: str = "0", import_path: str | None = None,
               checkpoints=_UNSET) -> subprocess.CompletedProcess:
    """Run the sbatch's preflight heredoc.

    ``contract`` is the adapter's live ``SEGMENTER_CONTRACT`` (pass ``None`` for an adapter that
    exports none); ``committed`` is the segmenter block of the committed document (pass ``None``
    for no document at all); ``committed_doc`` writes a whole document verbatim, for the case where
    the document exists and carries no segmenter block anywhere; ``checkpoints`` is what the
    adapter declares it loads (pass ``None`` for an adapter that names no weights).
    """
    live = CONTRACT_NOW if contract is _UNSET else contract
    scripts = _scripts_dir(tmp_path, contract=_UNSET if live is None else live, gate=gate,
                           checkpoints=checkpoints)

    contract_file = tmp_path / "pr08_geom_tol.json"
    if committed_doc is not _UNSET:
        contract_file.write_text(json.dumps(committed_doc))
    else:
        block = dict(CONTRACT_NOW) if committed is _UNSET else committed
        if block is not None:
            contract_file.write_text(json.dumps({"segmenter": block}))

    shard_path = tmp_path / "shard-0.json"
    if shard_out is not _UNSET:
        shard_path.write_text(json.dumps(
            {"schema": "wam.geom_tol_shard/1",
             "mask_method": {"params": {"segmenter": shard_out}}}))

    return _run_snippet(_heredoc_after(PREFLIGHT), [], {
        "PYTHONPATH": import_path if import_path is not None else str(scripts),
        "CONTRACT_FILE": str(contract_file),
        "SHARD_OUT": str(shard_path),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        "WAIVED": waived,
    })


def test_the_preflight_exists_and_runs_before_the_decode() -> None:
    """The whole value of this block is WHERE it is. A correct check after the decode is the
    situation it was written to replace: measure_geom_tol.py already refuses on contract drift, at
    the end of the shard, having spent the frames."""
    assert PREFLIGHT in TEXT, "the pre-GPU preflight block is gone"
    decode = TEXT.index('--shard "${SHARD_INDEX}"')
    assert TEXT.index(PREFLIGHT) < decode, (
        "the preflight no longer precedes the measure_geom_tol.py --shard invocation, which is the "
        "only property that makes it cheap")
    # ...and before the walltime self-check, so the cheapest and most fundamental refusal is first.
    assert TEXT.index(PREFLIGHT) < TEXT.index(SELFCHECK)


def test_the_preflight_uses_measure_geom_tols_own_comparator() -> None:
    """A second spelling of "the same segmenter" is exactly the copy that drifts, and the direction
    it drifts in is a preflight that passes a run the merge will refuse."""
    src = _heredoc_after(PREFLIGHT)
    assert "mgt.contract_disagreements(" in src
    assert "mgt.committed_segmenter_contract(" in src
    assert "def contract_disagreements" not in src, (
        "the preflight has grown its own copy of the comparison it is supposed to import")


def test_the_two_names_the_preflight_imports_still_exist() -> None:
    """The other half of the same guarantee, and the only cross-file dependency this sbatch has.

    The preflight resolves ``contract_disagreements`` and ``committed_segmenter_contract`` off
    ``scripts/measure_geom_tol.py`` at run time, on the cluster, inside a job. A rename there would
    surface as an AttributeError inside a Slurm log — after the allocation, and only for whoever
    happened to submit next. Pinned here so it surfaces as a test failure on the workstation
    instead. If the names move, the sbatch's heredoc moves with them.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mgt_for_103", _REPO_ROOT / "scripts" / "measure_geom_tol.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mgt_for_103"] = module
    spec.loader.exec_module(module)
    for name in ("contract_disagreements", "committed_segmenter_contract", "SAM2_ADAPTER_SPEC"):
        assert hasattr(module, name), (
            f"scripts/measure_geom_tol.py no longer exports {name}, which "
            "cluster/discoverer/103_measure_geom_tol.sbatch's pre-GPU preflight imports by name")
    assert module.contract_disagreements({"a": 1}, {"a": 2}) == [
        {"field": "a", "geom_tol": 2, "this_run": 1}], (
        "contract_disagreements() changed shape; the preflight formats its refusal from the "
        "'field', 'geom_tol' and 'this_run' keys")


def test_a_matching_contract_and_a_qualified_adapter_pass(tmp_path) -> None:
    r = _preflight(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PREFLIGHT OK" in r.stdout


def test_contract_drift_refuses_and_names_the_field(tmp_path) -> None:
    """THE DEFECT, in the exact shape it already occurred in. All sixteen of the last shards were
    missing mask_validity_reference_max_frame_fraction, which the committed document has; the merge
    named that one field, one array and 13.6 GPU-h later."""
    stale = {k: v for k, v in CONTRACT_NOW.items()
             if k != "mask_validity_reference_max_frame_fraction"}
    r = _preflight(tmp_path, contract=stale)
    assert r.returncode == 1, r.stdout
    assert "CONTRACT DRIFT" in r.stdout
    assert "mask_validity_reference_max_frame_fraction" in r.stdout
    # Both causes, and they have different remedies — that distinction is the point of the message.
    assert "cluster/discoverer/sync.sh" in r.stdout
    assert "V-document" in r.stdout
    assert "REFUSING TO START, BEFORE ANY DECODE" in r.stdout


def test_contract_drift_does_not_tell_the_operator_to_edit_the_committed_block(tmp_path) -> None:
    """configs/transfer25/pr08_geom_tol.json's segmenter block is a pre-registered contract.
    Changing a value in it to make a refusal go away silently invalidates every landed artifact."""
    r = _preflight(tmp_path, contract={**CONTRACT_NOW, "box_threshold": 0.35})
    assert r.returncode == 1
    assert "IT IS NOT AN EDIT TO" in r.stdout
    assert "invalidates every artifact already" in r.stdout


def test_a_false_gate_flag_refuses_before_the_gpu_work(tmp_path) -> None:
    """gate_qualified is baked into each shard AT MEASUREMENT TIME and no merge re-derives it, so a
    partition started before the flag flips is 13.6 GPU-h spent on a permanently disqualified
    number. The header has said "that is a decision to take BEFORE the array goes in" since
    2026-08-23; this is that sentence enforced."""
    r = _preflight(tmp_path, gate=False)
    assert r.returncode == 1, r.stdout
    assert "GATE_QUALIFIED IS False" in r.stdout
    assert "baked into" in r.stdout.lower() or "BAKED INTO" in r.stdout
    assert "re-measuring the corpus" in r.stdout


def test_the_gate_refusal_never_suggests_editing_the_flag(tmp_path) -> None:
    """The flag is the project owner's, in its own commit, after the residue is decided on the
    record. A refusal that reads as "set it to True" is worse than no refusal at all — and it must
    name the decision that is actually outstanding, not the blocker that was discharged."""
    r = _preflight(tmp_path, gate=False)
    out = r.stdout
    assert "NOT A REQUEST TO EDIT THE FLAG" in out
    assert "GATE_QUALIFIED = True" not in out and "GATE_QUALIFIED=True" not in out
    assert "residue (i)" in out, (
        "the refusal does not name the decision that is actually outstanding")
    assert "NOBODY HAS LOOKED" not in out.upper().replace("'", ""), (
        "the refusal still cites the blocker discharged on 2026-08-26")


def test_a_stale_shard_artifact_at_out_is_refused_before_the_decode(tmp_path) -> None:
    """THE FORCE=1 TRAP. On the shard path measure_geom_tol.py's --out IS the shard artifact, and
    merge_committed_contract() reads whatever is already there as the committed contract — after
    the decode. A re-measure into a populated directory therefore decodes ~11 000 frames and THEN
    exits 2, destroying the work rather than refusing it."""
    stale = {k: v for k, v in CONTRACT_NOW.items()
             if k != "mask_validity_reference_max_frame_fraction"}
    r = _preflight(tmp_path, shard_out=stale)
    assert r.returncode == 1, r.stdout
    assert "STALE SHARD ARTIFACT AT --out" in r.stdout
    assert "mask_validity_reference_max_frame_fraction" in r.stdout
    assert "FRESH RUN_ID" in r.stdout
    assert "DO NOT simply delete the old shard" in r.stdout, (
        "the remedy must not be an rm -f: that stale artifact is the only in-situ evidence that "
        "this index was measured by a different adapter")


def test_a_shard_artifact_that_agrees_is_not_a_refusal(tmp_path) -> None:
    """Resumability is the whole reason the partition is sharded. An existing shard measured by
    THIS adapter is a re-run, not a trap."""
    r = _preflight(tmp_path, shard_out=dict(CONTRACT_NOW))
    assert r.returncode == 0, r.stdout
    assert "PREFLIGHT OK" in r.stdout


def test_a_truncated_shard_artifact_is_not_mistaken_for_a_contract(tmp_path) -> None:
    """What a task killed at the wall leaves behind. Re-running over it is normal and must not be
    refused — merge_committed_contract() ignores it too, for the same reason."""
    scripts = _scripts_dir(tmp_path, contract=dict(CONTRACT_NOW), gate=True)
    (tmp_path / "shard-0.json").write_text('{"schema": "wam.geom_tol_shard/1", "mask_me')
    (tmp_path / "pr08_geom_tol.json").write_text(json.dumps({"segmenter": dict(CONTRACT_NOW)}))
    r = _run_snippet(_heredoc_after(PREFLIGHT), [], {
        "PYTHONPATH": str(scripts),
        "CONTRACT_FILE": str(tmp_path / "pr08_geom_tol.json"),
        "SHARD_OUT": str(tmp_path / "shard-0.json"),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        "WAIVED": "0",
    })
    assert r.returncode == 0, r.stdout


def test_a_missing_pre_commitment_is_refused_and_not_ignored(tmp_path) -> None:
    """Job 190191 overwrote that path once already, with a merged disqualified artifact. A preflight
    that shrugged at a missing contract would send an array at a partition the merge cannot write."""
    r = _preflight(tmp_path, committed=None)
    assert r.returncode == 1, r.stdout
    assert "PRE-COMMITMENT IS MISSING" in r.stdout
    assert "cluster/discoverer/sync.sh" in r.stdout


def test_a_committed_document_with_no_segmenter_block_is_refused(tmp_path) -> None:
    r = _preflight(tmp_path, committed_doc={"spec_version": "0.1.0"})
    assert r.returncode == 1, r.stdout
    assert "NO SEGMENTER BLOCK" in r.stdout


def test_an_adapter_that_declares_no_contract_is_refused(tmp_path) -> None:
    r = _preflight(tmp_path, contract=None)
    assert r.returncode == 1, r.stdout
    assert "NO SEGMENTER_CONTRACT" in r.stdout


# -- the preflight has to ask the SAME question the measurement asks, both ways ---------------------
#
# A preflight is only worth its refusal if it agrees with the check it is standing in for. Both
# directions of disagreement are defects and they cost different things: passing a unit the real
# check later refuses buys the 13.6 GPU-h this block exists to save, and refusing a unit the real
# check would accept stops a legitimate array and sends the operator chasing a fault that is not
# there. The three tests below pin one of each, plus the case where the preflight cannot know which
# module the measurement will import.


def test_an_adapter_that_names_no_weights_is_refused(tmp_path) -> None:
    """THE CONJUNCT A FLAG-ONLY CHECK MISSES. What lands in the shard is not GATE_QUALIFIED: it is
    ``sam2_method()``'s ``declared_gate and bool(checkpoints) and contract is not None``. An adapter
    that sets the flag, agrees with the contract field for field, and names NO weights is refused by
    that AND — silently, at measurement time, into all sixteen artifacts, with the reason parked in
    ``gate_qualification_withheld_reason`` where nothing reads it until the merge.

    Before this was checked the preflight printed PREFLIGHT OK for exactly this adapter, which is
    the 13.6 GPU-h loss reached through the one conjunct nobody looked at.
    """
    r = _preflight(tmp_path, checkpoints=None)
    assert r.returncode == 1, r.stdout
    assert "NAMES NO WEIGHTS" in r.stdout
    assert "gate_qualified" in r.stdout
    # And it must not read as "set GATE_QUALIFIED", which is a different (owner's) question.
    assert "GATE_QUALIFIED = True" not in r.stdout


def test_the_weights_check_is_measure_geom_tols_own_and_refuses_if_it_moves(tmp_path) -> None:
    """The rule for "what did this adapter declare it loads" is ``_adapter_checkpoints()`` and it is
    imported, not re-spelled — the same reason ``contract_disagreements()`` is. If that function
    ever moves, an unchecked conjunct reads downstream exactly like a satisfied one, so the
    preflight refuses rather than passing. Waivable, unlike the imports: the measurement still runs.
    """
    src = _heredoc_after(PREFLIGHT)
    assert "_adapter_checkpoints" in src, (
        "the preflight no longer reads the weights half of gate_qualified from measure_geom_tol")
    assert "ESTIMATOR_CHECKPOINTS" not in src.split("failures.append")[0], (
        "the preflight has grown its own copy of the checkpoint-discovery rule")

    scripts = _scripts_dir(tmp_path, contract=dict(CONTRACT_NOW), gate=True)
    mgt = scripts / "measure_geom_tol.py"
    mgt.write_text(mgt.read_text() + "\ndel _adapter_checkpoints\n")
    (tmp_path / "pr08_geom_tol.json").write_text(json.dumps({"segmenter": dict(CONTRACT_NOW)}))
    r = _run_snippet(_heredoc_after(PREFLIGHT), [], {
        "PYTHONPATH": str(scripts),
        "CONTRACT_FILE": str(tmp_path / "pr08_geom_tol.json"),
        "SHARD_OUT": str(tmp_path / "does-not-exist.json"),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        "WAIVED": "0",
    })
    assert r.returncode == 1, r.stdout
    assert "_adapter_checkpoints IS GONE" in r.stdout


def test_the_preflight_does_not_invent_the_adapters_module_name(tmp_path) -> None:
    """THE OTHER WAY A PREFLIGHT LIES: checking the right question about the wrong module.

    The adapter's name is read off ``measure_geom_tol.SAM2_ADAPTER_SPEC``, because that is what its
    own ``_import_sam2_adapter()`` resolves during the measurement. A literal fallback here would
    mean that if the measurement's spec is ever renamed or repointed, the preflight imports the OLD
    adapter, finds it agreeing with the committed contract, and prints PREFLIGHT OK about a module
    this job does not run — the failure this whole block exists to prevent, inverted. There is no
    honest default for "which adapter is the measurement's", so absence is a refusal.
    """
    scripts = _scripts_dir(tmp_path, contract=dict(CONTRACT_NOW), gate=True)
    mgt = scripts / "measure_geom_tol.py"
    mgt.write_text(mgt.read_text() + "\ndel SAM2_ADAPTER_SPEC\n")
    (tmp_path / "pr08_geom_tol.json").write_text(json.dumps({"segmenter": dict(CONTRACT_NOW)}))
    env = {
        "PYTHONPATH": str(scripts),
        "CONTRACT_FILE": str(tmp_path / "pr08_geom_tol.json"),
        "SHARD_OUT": str(tmp_path / "does-not-exist.json"),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        "WAIVED": "0",
    }
    r = _run_snippet(_heredoc_after(PREFLIGHT), [], env)
    assert r.returncode == 1, (
        "the preflight guessed the adapter's module name and reported on a module the measurement "
        "would not import:\n" + r.stdout)
    assert "DECLARES NO SAM2_ADAPTER_SPEC" in r.stdout
    # Not waivable, for the same reason the two imports are not: nothing here can be checked.
    waived = _run_snippet(_heredoc_after(PREFLIGHT), [], {**env, "WAIVED": "1"})
    assert waived.returncode == 1, waived.stdout


def test_a_contract_frozen_behind_a_mappingproxy_is_not_a_missing_contract(tmp_path) -> None:
    """THE FALSE REFUSAL. ``sam2_method()`` tests that attribute with ``isinstance(contract,
    Mapping)``, so an adapter that froze its pre-registered contract behind a ``MappingProxyType``
    — the obvious way to make a constant unwritable — is accepted by the measurement. A preflight
    spelling the same test ``dict`` refuses it with THE ADAPTER EXPORTS NO SEGMENTER_CONTRACT and
    sends the operator off to re-sync a cluster copy that was never stale.

    Refusing a run the real check would accept is as much a defect here as passing one it would
    refuse, and this one is the more expensive to diagnose: the message names the wrong cause.
    """
    scripts = _scripts_dir(tmp_path, contract=dict(CONTRACT_NOW), gate=True)
    (scripts / "estimators" / "apple_sam2.py").write_text(
        "from types import MappingProxyType\n"
        "ESTIMATOR_VERSION = 'test-adapter'\n"
        "GATE_QUALIFIED = True\n"
        f"ESTIMATOR_CHECKPOINTS = {CHECKPOINTS_NOW!r}\n"
        f"SEGMENTER_CONTRACT = MappingProxyType({dict(CONTRACT_NOW)!r})\n")
    (tmp_path / "pr08_geom_tol.json").write_text(json.dumps({"segmenter": dict(CONTRACT_NOW)}))
    r = _run_snippet(_heredoc_after(PREFLIGHT), [], {
        "PYTHONPATH": str(scripts),
        "CONTRACT_FILE": str(tmp_path / "pr08_geom_tol.json"),
        "SHARD_OUT": str(tmp_path / "does-not-exist.json"),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        "WAIVED": "0",
    })
    assert r.returncode == 0, (
        "the preflight refused a contract the measurement accepts, and blamed the adapter for "
        "exporting none:\n" + r.stdout)
    assert "PREFLIGHT OK" in r.stdout


# -- the waiver: explicit, loud, and never the default ---------------------------------------------


def test_the_waiver_is_not_the_default(tmp_path) -> None:
    """A check that is default-off is a comment. Every refusal above ran with WAIVED unset."""
    scripts = _scripts_dir(tmp_path, contract=dict(CONTRACT_NOW), gate=False)
    (tmp_path / "pr08_geom_tol.json").write_text(json.dumps({"segmenter": dict(CONTRACT_NOW)}))
    r = _run_snippet(_heredoc_after(PREFLIGHT), [], {
        "PYTHONPATH": str(scripts),
        "CONTRACT_FILE": str(tmp_path / "pr08_geom_tol.json"),
        "SHARD_OUT": str(tmp_path / "does-not-exist.json"),
        "GIT_COMMIT_FILE": str(tmp_path / "GIT_COMMIT"),
        # WAIVED deliberately absent from the environment, not set to "0".
    })
    assert r.returncode == 1, r.stdout


def test_the_waiver_names_what_it_waives_and_shouts(tmp_path) -> None:
    """One env var, and its name says what is being given up. The warning has to be unmissable in a
    log an operator skims: what it buys is an artifact that will be refused at full GPU cost."""
    assert "GEOM_WAIVE_CONTRACT_AND_GATE_PREFLIGHT" in TEXT
    r = _preflight(tmp_path, gate=False, waived="1")
    assert r.returncode == 0, r.stdout
    assert "WARNING (WAIVED)" in r.stdout
    assert "DO NOT COMMIT ANYTHING THIS RUN PRODUCES AS GEOM_TOL" in r.stdout


def test_the_waiver_cannot_wave_through_a_broken_cluster_copy(tmp_path) -> None:
    """The waiver waives the CHECKS, never the two imports. A measure_geom_tol.py that will not
    import means the measurement itself cannot run; there is nothing there to waive."""
    r = _preflight(tmp_path, waived="1", import_path=str(tmp_path / "empty"))
    assert r.returncode == 1, r.stdout
    assert "STALE OR BROKEN CLUSTER COPY" in r.stdout
    assert "not waivable" in r.stdout.lower()
    assert "cluster/discoverer/sync.sh" in r.stdout


# -- the recovery command has to run on the machine that prints it ---------------------------------

_GIT_IS_REFUTED = ("CANNOT RUN HERE", "not a git repository", "does not:", "NOT with git")


def test_no_message_tells_the_operator_to_run_git_on_the_cluster() -> None:
    """THE DEFECT: sync.sh:66 rsyncs with --exclude '.git', so ${PROJ}/wam is a working tree with no
    repository behind it. `git -C ${WAM} checkout -- configs/...` fails there with "not a git
    repository" — at exactly the moment the pre-commitment has been overwritten and the operator is
    reading the log to find out how to put it back.

    A `git -C` may still APPEAR, because naming the command that does not work is how a reader who
    remembers the old message learns why it is gone. It may not appear as an instruction.
    """
    lines = TEXT.splitlines()
    for i, line in enumerate(lines):
        if "git -C" not in line:
            continue
        window = "\n".join(lines[max(0, i - 4):i + 5])
        assert any(marker in window for marker in _GIT_IS_REFUTED), (
            f"line {i + 1} prints a `git -C` command with nothing nearby saying it cannot run on "
            f"the cluster:\n{window}")


def test_both_recovery_sites_name_the_command_that_does_work() -> None:
    """The real recovery is a re-sync from the workstation, which is the only machine with the
    history. Both places that lose the pre-commitment have to say so."""
    assert TEXT.count("./cluster/discoverer/sync.sh") >= 2, (
        "at least one refusal that destroys or misses the committed contract still fails to name "
        "the re-sync that restores it")
    assert "--exclude '.git'" in TEXT, (
        "the messages no longer say WHY git does not work there, so the next author will put the "
        "git command back")


# -- the operator-facing account of why an artifact would be disqualified --------------------------


def test_the_file_does_not_claim_the_human_look_is_still_the_blocker() -> None:
    """GATE_QUALIFICATION_BLOCKERS is `()` and the human look was discharged 2026-08-26. The page an
    operator reads WHILE DECIDING WHETHER TO SPEND 13.6 GPU-HOURS was wrong about why the artifact
    would be disqualified.

    The phrase may survive where the file is explicitly correcting itself — this codebase keeps its
    errors legible rather than deleting them — so what is asserted is that every occurrence sits
    beside a marker that retracts it.
    """
    lines = TEXT.splitlines()
    for i, line in enumerate(lines):
        if "NOBODY HAS LOOKED AT A MASK" not in line.upper():
            continue
        window = "\n".join(lines[max(0, i - 3):i + 4])
        assert any(m in window for m in ("NO\n", "NO LONGER", "NOT because", "discharged", "used to")), (
            f"line {i + 1} still gives the discharged blocker as the current reason:\n{window}")


def test_the_actual_outstanding_precondition_is_named() -> None:
    """Not overstated in the other direction either: the flag is False, the second precondition is
    a recorded owner decision on blocker 2's residue (i), and nothing here may read as though the
    flag is about to flip."""
    assert "residue (i)" in TEXT
    assert "GATE_QUALIFIED = False" in TEXT or "GATE_QUALIFIED=False" in TEXT
    assert "empty tuple" in TEXT.lower()


# -- the resume check: a shard that could never be committed is not a shard worth keeping ----------
#
# THE DEFECT THIS SECTION PINS, AND IT IS THE MOST EXPENSIVE KIND — a check that saves work by
# skipping exactly the work that had to be redone. ``shard_artifact_landed()`` decides whether the
# artifact already at ${SHARD_OUT} makes this array task unnecessary, and it is consulted BEFORE the
# pre-GPU preflight above. The sixteen shards under runs/pr08-geom-tol/shards/ are permanently
# uncommittable for two independent reasons — each records ``gate_qualified: false`` from an adapter
# flag that has since been flipped, and each lacks mask_validity_reference_max_frame_fraction, which
# the committed pre-commitment carries — and the function called every one of them REUSABLE. A
# default submission therefore printed "already landed. Skipping." sixteen times, exited 0 in
# seconds, never reached the preflight, and left the merge pooling a permanently disqualified
# median. The two staleness checks below are both fail-closed: the most either can cost is a shard
# re-measured for nothing.

#: The line that opens the resume check's heredoc, spelled the way tests/test_measure_geom_tol.py
#: spells it, so the two files cannot drift about which block they mean.
LANDED = "shard_artifact_landed () {"

REAL_SHARD = _REPO_ROOT / "runs/pr08-geom-tol/shards/shard-1.json"
REAL_CONTRACT = _REPO_ROOT / "configs/transfer25/pr08_geom_tol.json"


def _mgt():
    """The REAL ``scripts/measure_geom_tol.py``, loaded by path.

    Used only where a test needs to read the committed document the way the sbatch reads it — the
    same argument as everywhere else in this file: a test that spelled the lookup itself would keep
    passing over a resume check that had spelled it differently.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mgt_for_103", _REPO_ROOT / "scripts" / "measure_geom_tol.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mgt_for_103"] = module
    spec.loader.exec_module(module)
    return module


#: The exact sentence ``measure_geom_tol.main()`` appends when the adapter has not opted in — the
#: one reason ``shard_artifact_landed()`` forgives. Built the way main() builds it.
ADAPTER_BLOCKER_REASON = (
    "mask method 'grounding-dino+sam2+depth-anything-v2' is not gate-qualified")


def _shard_doc(*, segmenter=_UNSET, **over) -> dict:
    """A shard artifact in the shape main() writes for a complete, correctly measured shard.

    ``segmenter`` is the block the shard RECORDS as having produced it, written where
    ``measure_geom_tol.sam2_method()`` writes it (mask_method.params.segmenter) rather than where a
    test would find it convenient: the whole point of the contract check is that the reader and the
    writer agree about where that block lives.
    """
    block = CONTRACT_NOW if segmenter is _UNSET else segmenter
    doc = {
        "schema": "wam.geom_tol_shard/1",
        "shard": {"index": 0, "num_shards": 16, "n_episodes_in_shard": 33},
        "step_frames": 1,
        "gate_qualified": False,
        "gate_disqualified_reasons": [ADAPTER_BLOCKER_REASON],
        "headline_valid": True,
        "partial_measurement": False,
        "limit": 0,
        "max_frames": 0,
        "n_steps_measured": 14129,
        "shard_median_px": 1.5,
        "mask_method": {"name": "grounding-dino+sam2+depth-anything-v2",
                        "params": {"segmenter": block} if block is not None else {}},
    }
    doc.update(over)
    return doc


def _landed(tmp_path: Path, doc, *, index: int = 0, num_shards: int = 16, step: int = 1,
            gate: bool = True, committed=_UNSET, contract_file=None,
            import_path: str | None = None) -> subprocess.CompletedProcess:
    """Run the sbatch's OWN resume-check heredoc against a shard artifact.

    ``gate`` is the adapter's CURRENT ``GATE_QUALIFIED`` — the world outside the artifact, which is
    the whole subject of check 1 — and it is carried by the same stub adapter the preflight tests
    use, on the same throwaway ``scripts/`` holding the real ``measure_geom_tol``. ``committed`` is
    the segmenter block of the committed pre-commitment.
    """
    scripts = _scripts_dir(tmp_path, contract=CONTRACT_NOW, gate=gate)
    if contract_file is None:
        contract_file = tmp_path / "pr08_geom_tol.json"
        block = CONTRACT_NOW if committed is _UNSET else committed
        if block is not None:
            contract_file.write_text(json.dumps({"segmenter": block}))
    path = tmp_path / f"shard-{index}.json"
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return _run_snippet(_heredoc_after(LANDED), [str(path), str(index), str(num_shards), str(step)],
                        {"PYTHONPATH": import_path if import_path is not None else str(scripts),
                         "CONTRACT_FILE": str(contract_file)})


def test_the_resume_check_uses_measure_geom_tols_own_comparator() -> None:
    """Same guarantee as the preflight's, and for the same reason: a reader and a writer that each
    keep their own idea of where the segmenter block lives is how this array was lost once."""
    src = _heredoc_after(LANDED)
    assert "mgt.contract_disagreements(" in src
    assert "mgt.committed_segmenter_contract(" in src
    assert "def contract_disagreements" not in src, (
        "the resume check has grown its own copy of the comparison it is supposed to import")


def test_a_shard_the_adapters_flag_has_moved_past_is_not_reusable(tmp_path) -> None:
    """CHECK 1, AND THE INVERSION IT CATCHES. "That is the adapter's standing flag and not this
    shard" was a correct reason to reuse while the flag was False — re-measuring would have produced
    another unqualified shard, so the skip was free. With the flag True the same skip is the one
    action that keeps a permanently disqualified artifact in the partition, because gate_qualified
    is baked in at measurement time and no merge re-derives it."""
    r = _landed(tmp_path, _shard_doc(), gate=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not reusable" in r.stdout
    assert "STANDING FLAG HAS MOVED" in r.stdout
    assert "GATE_QUALIFIED is True today" in r.stdout


def test_the_same_shard_is_still_reusable_while_the_flag_is_still_false(tmp_path) -> None:
    """The other half, and it is load-bearing for a waived run: while the adapter is still not
    gate-qualified the old branch is CORRECT and its wording is kept verbatim. A repair that refused
    here would take resumability away from the partition again — every re-submission re-measuring
    the whole corpus is the defect that branch was written to fix."""
    r = _landed(tmp_path, _shard_doc(), gate=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LANDED BUT NOT GATE-QUALIFIED, and that is the adapter's standing flag and not this" \
        in r.stdout
    assert "It must not be committed as GEOM_TOL." in r.stdout
    assert "reusable:" in r.stdout


def test_a_shard_missing_a_contract_field_is_not_reusable_and_the_field_is_named(tmp_path) -> None:
    """CHECK 2, in the exact shape blocker (b) has. mask_validity_reference_max_frame_fraction is
    not metadata: it is a frame-refusal predicate, so a shard that does not record it was measured
    by a segmenter that refused a different set of frames than the committed document describes.
    Absence counts as a disagreement in contract_disagreements(), which is why importing it rather
    than re-spelling it is what makes this check work at all."""
    stale = {k: v for k, v in CONTRACT_NOW.items()
             if k != "mask_validity_reference_max_frame_fraction"}
    r = _landed(tmp_path, _shard_doc(segmenter=stale, gate_qualified=True,
                                     gate_disqualified_reasons=[]), gate=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not reusable" in r.stdout
    assert "DIFFERENT SEGMENTER CONTRACT" in r.stdout
    assert "mask_validity_reference_max_frame_fraction" in r.stdout
    # Both values, not merely the field: "they disagree" is not an actionable message.
    assert "committed 0.1" in r.stdout and "this shard None" in r.stdout


@pytest.mark.skipif(not REAL_SHARD.exists() or not REAL_CONTRACT.exists(),
                    reason="the previous array's shards are not in this working tree")
def test_the_real_landed_shard_on_disk_today_is_refused(tmp_path) -> None:
    """THE REGRESSION TEST FOR THE ACTUAL TRAP, run against the real artifact and the real
    committed document, through the real measure_geom_tol and the real adapter.

    runs/pr08-geom-tol/shards/shard-1.json is one of the sixteen a default submission resolves
    ${SHARD_OUT} straight onto (RUN_ID defaults to pr08-geom-tol). Before the repair the function
    answered LANDED=TRUE for it, the task exited 0 in seconds, the preflight below never ran, and
    the merge pooled a permanently disqualified median. It must never answer that again.
    """
    doc = json.loads(REAL_SHARD.read_text())
    r = _run_snippet(_heredoc_after(LANDED),
                     [str(REAL_SHARD), str(doc["shard"]["index"]),
                      str(doc["shard"]["num_shards"]), str(doc["step_frames"])],
                     {"PYTHONPATH": str(_REPO_ROOT / "scripts"),
                      "CONTRACT_FILE": str(REAL_CONTRACT)})
    assert r.returncode == 1, (
        "the resume check would still skip the shard that started this:\n" + r.stdout + r.stderr)
    assert "not reusable" in r.stdout
    # ...and refused for one of the two REAL reasons, not because the extraction fed it garbage.
    assert ("mask_validity_reference_max_frame_fraction" in r.stdout
            or "STANDING FLAG HAS MOVED" in r.stdout), r.stdout
    for structural in ("schema is", "shard.index is", "shard.num_shards is", "step_frames is",
                       "does not parse"):
        assert structural not in r.stdout, f"refused for the wrong reason: {r.stdout}"


@pytest.mark.skipif(not REAL_SHARD.exists() or not REAL_CONTRACT.exists(),
                    reason="the previous array's shards are not in this working tree")
def test_a_shard_with_both_defects_repaired_is_reusable_again(tmp_path) -> None:
    """THE CHECK ON THE CHECK: the repair must refuse the sixteen shards on disk WITHOUT refusing
    what a correct re-measurement will write, or the array loses resumability and every wave
    re-measures the corpus. So the real artifact is copied, its two staleness defects are undone —
    the gate flag it records and the contract field it lacks — and nothing else is touched.
    """
    doc = json.loads(REAL_SHARD.read_text())
    doc["gate_qualified"] = True
    doc["gate_disqualified_reasons"] = []
    committed, _ = _mgt().committed_segmenter_contract(json.loads(REAL_CONTRACT.read_text()))
    doc["mask_method"]["params"]["segmenter"] = dict(committed)
    fixed = tmp_path / "shard-1.json"
    fixed.write_text(json.dumps(doc))
    r = _run_snippet(_heredoc_after(LANDED),
                     [str(fixed), str(doc["shard"]["index"]), str(doc["shard"]["num_shards"]),
                      str(doc["step_frames"])],
                     {"PYTHONPATH": str(_REPO_ROOT / "scripts"),
                      "CONTRACT_FILE": str(REAL_CONTRACT)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reusable:" in r.stdout


def test_an_unreadable_pre_commitment_is_not_a_licence_to_reuse(tmp_path) -> None:
    """FAIL-CLOSED, AND IN THE ONLY DIRECTION THAT IS SAFE. A missing committed document means the
    one comparison that would have caught blocker (b) cannot be made — and an unchecked shard is not
    a checked one. The cost of being wrong here is one shard re-measured; the cost of the other
    default is the whole partition pooled unmergeable."""
    r = _landed(tmp_path, _shard_doc(gate_qualified=True, gate_disqualified_reasons=[]),
                contract_file=tmp_path / "not-there.json")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not reusable" in r.stdout
    assert "not-there.json" in r.stdout


def test_an_adapter_that_will_not_import_is_not_a_licence_to_reuse(tmp_path) -> None:
    """The same rule for the other half of what check 1 and check 2 need. A cluster copy whose
    measure_geom_tol.py or adapter will not import cannot answer either question, and it is also a
    copy that cannot run the measurement — the preflight below says so in the same words."""
    r = _landed(tmp_path, _shard_doc(gate_qualified=True, gate_disqualified_reasons=[]),
                import_path=str(tmp_path / "empty"))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not reusable" in r.stdout
    assert "CANNOT BE CHECKED" in r.stdout
