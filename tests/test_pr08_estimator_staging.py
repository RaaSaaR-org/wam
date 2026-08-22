"""Tests for 102_stage_sam2_weights.sbatch — the job that stages PR-08 §4's estimator weights.

The job itself cannot run here: it downloads several GB from a rate-limited host onto a cluster
filesystem. What CAN run here is every decision it embeds, and the reason to run them is that the
first version of this file shipped two failures of exactly the shape PR-08 exists to prevent:

1. **It staged the wrong depth model.** `depth-anything/Video-Depth-Anything-Large` is RELATIVE —
   affine-invariant inverse depth, no metres in it — and §4 step 3 asks for the absolute depth error
   in metres against Isaac's ``distance_to_camera``. That does not crash; it writes a plausible
   float under a key called ``mean_m``.
2. **Its agreement check could not report a disagreement.** It ran ``grep -qF`` for the PRESENCE of
   an id in an artifact, so the only two outcomes were "found" and "unverified". The one outcome it
   existed to produce — "these two files name different checkpoints" — was unreachable.

So the tests below are in two halves. The first reads the job as text and asserts the decisions.
The second EXTRACTS the three python programs the job embeds, runs them against fabricated adapters,
artifacts and hub-cache trees, and asserts that a disagreement is a non-zero exit. No GPU, no
network, no weights.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_JOB = _REPO_ROOT / "cluster" / "discoverer" / "102_stage_sam2_weights.sbatch"
_ADAPTER = _REPO_ROOT / "scripts" / "estimators" / "apple_sam2.py"

#: The contract, restated once here on purpose. Three files have to agree — the adapter, the job and
#: this test — and a test that reads its expectation out of the file under test asserts nothing.
_SEGMENTER = ("facebook/sam2-hiera-large", "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251")
_DETECTOR = ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
_DEPTH = ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
          "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f")
_RELATIVE_DEPTH = "depth-anything/Video-Depth-Anything-Large"


def _text() -> str:
    assert _JOB.is_file(), f"{_JOB} does not exist"
    return _JOB.read_text()


def _default(name: str, text: str) -> str:
    match = re.search(rf"^{name}=\$\{{{name}:-([^}}]*)\}}", text, re.MULTILINE)
    assert match, f"102 no longer sets a default {name}"
    return match.group(1).strip()


def _heredoc(text: str, marker: str) -> str:
    """The body of a quoted heredoc, so the embedded program can be run instead of read."""
    opener = f"<<'{marker}'\n"
    assert opener in text, f"102 no longer embeds a {marker} heredoc"
    start = text.index(opener) + len(opener)
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


# == half one: the decisions, read off the job =====================================================


def test_the_job_bills_no_gpu_hours() -> None:
    """PR-08 §8 item 3: there is no budget line yet, so staging may not spend from one."""
    text = _text()
    assert "--qos=2cpu-single-host" in text, "102 is not on the free QoS"
    assert "--gres=gpu" not in text, "102 requests a GPU"


def test_the_job_generates_nothing() -> None:
    """PR-08 §1 licenses staging and timing, never generating."""
    text = _text()
    assert "restyle_transfer25.py" not in text
    assert "STYLE_SET" not in text


@pytest.mark.parametrize(
    "name,expected",
    [
        ("SAM2_MODEL_ID", _SEGMENTER[0]),
        ("SAM2_MODEL_REVISION", _SEGMENTER[1]),
        ("GDINO_MODEL_ID", _DETECTOR[0]),
        ("GDINO_MODEL_REVISION", _DETECTOR[1]),
        ("DEPTH_MODEL_ID", _DEPTH[0]),
        ("DEPTH_MODEL_REVISION", _DEPTH[1]),
    ],
)
def test_the_pins_are_the_contracted_ones(name: str, expected: str) -> None:
    assert _default(name, _text()) == expected


def test_the_depth_head_is_metric_and_the_relative_one_is_not_staged() -> None:
    """The high finding, as a regression.

    ``Video-Depth-Anything`` is allowed to appear in the header — the reasoning for NOT using it is
    the most valuable thing in this file — but it may not appear in any line the shell executes.
    """
    text = _text()
    assert "Metric" in _default("DEPTH_MODEL_ID", text), (
        "the depth id is not a metric head; §4 step 3's error is in metres and a relative "
        "checkpoint has no metres in it"
    )
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if _RELATIVE_DEPTH in ln or "Video-Depth-Anything" in ln]
    assert not offenders, f"the relative depth model is still referenced in executable lines: {offenders}"


@pytest.mark.parametrize("pointer", ["main", "master", "HEAD", "latest"])
def test_moving_pointers_are_refused_by_name(pointer: str) -> None:
    text = _text()
    case_block = text[text.index("refuse_moving_pointer() {"):]
    case_block = case_block[: case_block.index("esac")]
    assert pointer in case_block, f"{pointer!r} is not refused as a revision"


def test_every_revision_goes_through_the_pointer_guard() -> None:
    text = _text()
    for name in ("SAM2_MODEL_REVISION", "GDINO_MODEL_REVISION", "DEPTH_MODEL_REVISION"):
        assert f"refuse_moving_pointer {name}" in text.replace("  ", " "), (
            f"{name} is never checked for a moving pointer"
        )


def test_the_ids_are_compared_before_anything_is_downloaded() -> None:
    """A disagreement found after the fetch is a disagreement that cost several GB.

    Anchored on control flow rather than on a log line: an earlier sibling job's ordering test
    passed while the whole block had been moved below the download, because the echo moved with it.
    """
    text = _text()
    download = text.index("hf@latest download")
    for marker, what in (
        ('python3 "${CHECK_IDS_PY}"', "the id comparison runs"),
        ("CHECK_RC", "its exit status is branched on"),
        ("probe_repo segmenter", "the reachability probe"),
    ):
        assert marker in text, f"102 no longer contains {what} ({marker!r})"
        assert text.index(marker) < download, f"{what} happens after the download starts"


def test_no_presence_only_grep_stands_in_for_an_agreement_check() -> None:
    """``grep -qF <id> <artifact>`` has no outcome that means "they differ"."""
    text = _text()
    assert "grep -qF" not in text, (
        "a presence grep is back. Presence is not agreement: it can report 'found' or 'not "
        "found', never 'these two files name different checkpoints'."
    )


def test_the_reachability_probe_asks_the_question_it_reports_on() -> None:
    """A HEAD on README.md 404s for a repo with no README, and the old text called that 'the repo

    or the revision does not exist'. The revision endpoint answers existence directly.
    """
    text = _text()
    assert "api/models/${repo}/revision/${rev}" in text
    assert "resolve/${rev}/README.md" not in text
    assert "|| echo 000" not in text, (
        "curl's -w already prints 000 on a transport failure; `|| echo 000` appends a second one "
        "and the unexpected-code branch then prints a malformed code"
    )


def test_the_adapter_is_the_named_source_of_truth() -> None:
    text = _text()
    assert "ESTIMATOR_ADAPTER" in text
    assert "scripts/estimators/apple_sam2.py" in text


def test_the_job_cites_98s_dependency_paragraph_at_the_line_it_is_on() -> None:
    """The low finding: the citation was already off by one when it was written."""
    text = _text()
    match = re.search(r"98_build_transfer25_env\s*\n?#?\s*\.sbatch:(\d+)-(\d+)", text)
    assert match, "the citation to 98's base-dependency paragraph is gone"
    first, last = int(match.group(1)), int(match.group(2))
    cited = (_REPO_ROOT / "cluster" / "discoverer" / "98_build_transfer25_env.sbatch").read_text()
    lines = cited.splitlines()[first - 1:last]
    assert any("sam2>=1.1.0" in ln for ln in lines), (
        f"98:{first}-{last} does not contain the sam2 dependency claim it is cited for"
    )


# == half two: the embedded programs, run ==========================================================


def _write_adapter(
    tmp_path: Path,
    *,
    segmenter: str = _SEGMENTER[0],
    detector: str = _DETECTOR[0],
    depth: str = _DEPTH[0],
    revisions: dict[str, str] | None = None,
) -> Path:
    """A stand-in for scripts/estimators/apple_sam2.py, in the same ``os.environ.get`` shape."""
    body = [
        "import os",
        f'SAM2_MODEL_CHECKPOINT = os.environ.get("WAM_PR08_SAM2_CHECKPOINT", "{segmenter}")',
        "GROUNDING_DINO_MODEL_CHECKPOINT = os.environ.get(",
        f'    "WAM_PR08_GROUNDING_DINO_CHECKPOINT", "{detector}"',
        ")",
        f'DEPTH_MODEL_CHECKPOINT = os.environ.get("WAM_PR08_DEPTH_CHECKPOINT", "{depth}")',
        'METRIC_DEPTH_CHECKPOINT_SUGGESTIONS = ("a", "b")',
    ]
    for name, value in (revisions or {}).items():
        body.append(f'{name} = "{value}"')
    path = tmp_path / "adapter.py"
    path.write_text("\n".join(body) + "\n")
    return path


def _run_checker(
    tmp_path: Path,
    adapter: Path,
    artifacts: list[Path] | None = None,
    *,
    wants: list[tuple[str, str, str]] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "check_ids.py"
    script.write_text(_heredoc(_text(), "PY_CHECK_IDS"))
    if wants is None:
        wants = [("segmenter", *_SEGMENTER), ("detector", *_DETECTOR), ("depth", *_DEPTH)]
    child = dict(os.environ)
    # The adapter reads its ids through these; leaking one in from the developer's shell would make
    # the comparison depend on who ran the test.
    for key in [k for k in child if k.startswith("WAM_PR08_")]:
        child.pop(key)
    child["PR08_WANT"] = "\x1f".join("\t".join(w) for w in wants)
    child.update(env or {})
    return subprocess.run(
        [sys.executable, str(script), str(adapter), *[str(a) for a in (artifacts or [])]],
        capture_output=True, text=True, env=child,
    )


def _checks(out: str) -> dict[str, str]:
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if parts[0] == "CHECK":
            rows[parts[1]] = parts[2]
    return rows


def test_checker_agrees_with_an_adapter_that_names_the_staged_ids(tmp_path: Path) -> None:
    done = _run_checker(tmp_path, _write_adapter(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    rows = _checks(done.stdout)
    for role in ("segmenter", "detector", "depth"):
        assert rows[f"{role}-id-vs-adapter"] == "agrees"


def test_checker_refuses_when_the_adapter_names_a_different_depth_checkpoint(tmp_path: Path) -> None:
    """The medium finding, as a regression: this is the case a presence grep could not report."""
    adapter = _write_adapter(tmp_path, depth=_RELATIVE_DEPTH)
    done = _run_checker(tmp_path, adapter)
    assert done.returncode == 2, done.stdout + done.stderr
    fatal = [ln for ln in done.stdout.splitlines() if ln.startswith("FATAL")]
    assert fatal, "a disagreement produced no FATAL record"
    assert _RELATIVE_DEPTH in fatal[0] and _DEPTH[0] in fatal[0], (
        "the refusal must name both strings; 'they disagree' without the two values is a message "
        "nobody can act on"
    )


@pytest.mark.parametrize("role,ident", [("segmenter", "facebook/sam2-hiera-tiny"),
                                        ("detector", "IDEA-Research/grounding-dino-tiny")])
def test_checker_refuses_a_different_segmenter_or_detector(
    tmp_path: Path, role: str, ident: str
) -> None:
    kwargs = {"segmenter": ident} if role == "segmenter" else {"detector": ident}
    done = _run_checker(tmp_path, _write_adapter(tmp_path, **kwargs))
    assert done.returncode == 2, done.stdout + done.stderr


def test_checker_refuses_when_there_is_no_adapter_at_all(tmp_path: Path) -> None:
    """Missing is not 'agrees'. Without the adapter there is nothing for the ids to agree with."""
    done = _run_checker(tmp_path, tmp_path / "nope.py")
    assert done.returncode == 2
    assert "FATAL" in done.stdout


def test_checker_refuses_an_adapter_it_cannot_parse(tmp_path: Path) -> None:
    broken = tmp_path / "adapter.py"
    broken.write_text("def (:\n")
    done = _run_checker(tmp_path, broken)
    assert done.returncode == 2
    assert "FATAL" in done.stdout


def test_checker_refuses_an_adapter_that_dropped_the_constant(tmp_path: Path) -> None:
    partial = tmp_path / "adapter.py"
    partial.write_text('SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"\n')
    done = _run_checker(tmp_path, partial)
    assert done.returncode == 2
    assert any("GROUNDING_DINO_MODEL_CHECKPOINT" in ln for ln in done.stdout.splitlines())


def test_checker_reads_an_environment_override_the_adapter_would_see(tmp_path: Path) -> None:
    """``os.environ.get(VAR, default)`` means the default is not always the value."""
    done = _run_checker(tmp_path, _write_adapter(tmp_path),
                        env={"WAM_PR08_DEPTH_CHECKPOINT": _RELATIVE_DEPTH})
    assert done.returncode == 2, done.stdout + done.stderr
    assert _RELATIVE_DEPTH in done.stdout


def test_checker_follows_the_named_default_the_adapter_indirects_through(tmp_path: Path) -> None:
    """The adapter's real shape is ``X = os.environ.get("WAM_...", X_DEFAULT)``.

    A reader that only understands a literal default would report the single source of truth as
    unreadable while it sits there readable — and "unreadable" is a refusal, so the job would never
    run. Both outcomes have to be reachable through the indirection: agreement and disagreement.
    """
    agreeing = tmp_path / "agree.py"
    agreeing.write_text(
        "import os\n"
        f'SAM2_MODEL_ID_DEFAULT = "{_SEGMENTER[0]}"\n'
        f'GROUNDING_DINO_MODEL_ID_DEFAULT = "{_DETECTOR[0]}"\n'
        f'DEPTH_MODEL_ID_DEFAULT = "{_DEPTH[0]}"\n'
        'SAM2_MODEL_CHECKPOINT = os.environ.get("WAM_PR08_SAM2_CHECKPOINT", SAM2_MODEL_ID_DEFAULT)\n'
        'GROUNDING_DINO_MODEL_CHECKPOINT = os.environ.get("A", GROUNDING_DINO_MODEL_ID_DEFAULT)\n'
        'DEPTH_MODEL_CHECKPOINT = os.environ.get("B", DEPTH_MODEL_ID_DEFAULT)\n'
    )
    done = _run_checker(tmp_path, agreeing)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _checks(done.stdout)["depth-id-vs-adapter"] == "agrees"

    disagreeing = tmp_path / "disagree.py"
    disagreeing.write_text(
        agreeing.read_text().replace(f'"{_DEPTH[0]}"', f'"{_RELATIVE_DEPTH}"')
    )
    done = _run_checker(tmp_path, disagreeing)
    assert done.returncode == 2, done.stdout + done.stderr


def test_checker_treats_a_missing_revision_constant_as_unverified_not_as_agreement(
    tmp_path: Path,
) -> None:
    done = _run_checker(tmp_path, _write_adapter(tmp_path))
    assert done.returncode == 0
    rows = _checks(done.stdout)
    assert rows["depth-revision-vs-adapter"] == "unverified"
    assert any(ln.startswith("UNVERIFIED") for ln in done.stdout.splitlines())


def test_checker_agrees_with_revision_constants_when_the_adapter_declares_them(
    tmp_path: Path,
) -> None:
    adapter = _write_adapter(tmp_path, revisions={
        "SAM2_MODEL_REVISION": _SEGMENTER[1],
        "GROUNDING_DINO_MODEL_REVISION": _DETECTOR[1],
        "DEPTH_MODEL_REVISION": _DEPTH[1],
    })
    done = _run_checker(tmp_path, adapter)
    assert done.returncode == 0, done.stdout + done.stderr
    rows = _checks(done.stdout)
    for role in ("segmenter", "detector", "depth"):
        assert rows[f"{role}-revision-vs-adapter"] == "agrees"


def test_checker_refuses_a_revision_constant_that_disagrees(tmp_path: Path) -> None:
    """Same repo, two commits, two sets of weights. The id matching is not enough."""
    adapter = _write_adapter(tmp_path, revisions={"DEPTH_MODEL_REVISION": "0" * 40})
    done = _run_checker(tmp_path, adapter)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "0" * 40 in done.stdout


def _geom_tol(tmp_path: Path, checkpoints: dict[str, str]) -> Path:
    path = tmp_path / "pr08_geom_tol.json"
    path.write_text(json.dumps({
        "mask_method": {"name": "apple_sam2", "params": {"checkpoints": checkpoints}},
        "GEOM_TOL_px": 4.0,
    }))
    return path


def _est_drift(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "pr08_est_drift.json"
    path.write_text(json.dumps({"estimators": {"spec": "estimators.apple_sam2", "version": version}}))
    return path


def test_checker_refuses_a_committed_geom_tol_that_records_another_depth_id(tmp_path: Path) -> None:
    """Precisely the case the presence grep downgraded to 'we could not check'."""
    art = _geom_tol(tmp_path, {
        "SAM2_MODEL_CHECKPOINT": _SEGMENTER[0],
        "GROUNDING_DINO_MODEL_CHECKPOINT": _DETECTOR[0],
        "DEPTH_MODEL_CHECKPOINT": _RELATIVE_DEPTH,
    })
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 2, done.stdout + done.stderr
    assert "pr08_geom_tol.json" in done.stdout and _RELATIVE_DEPTH in done.stdout


def test_checker_accepts_a_committed_geom_tol_that_agrees(tmp_path: Path) -> None:
    art = _geom_tol(tmp_path, {
        "SAM2_MODEL_CHECKPOINT": _SEGMENTER[0],
        "GROUNDING_DINO_MODEL_CHECKPOINT": _DETECTOR[0],
        "DEPTH_MODEL_CHECKPOINT": _DEPTH[0],
    })
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 0, done.stdout + done.stderr
    assert _checks(done.stdout)["depth-vs-pr08_geom_tol.json"] == "agrees"


def test_checker_reads_the_est_drift_version_string_field_by_field(tmp_path: Path) -> None:
    good = _est_drift(tmp_path, f"det={_DETECTOR[0]};seg={_SEGMENTER[0]};depth={_DEPTH[0]};"
                                f"prompt='apple.';box_thr=0.35")
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [good])
    assert done.returncode == 0, done.stdout + done.stderr
    assert _checks(done.stdout)["depth-vs-pr08_est_drift.json"] == "agrees"


def test_checker_refuses_an_est_drift_whose_version_names_another_estimator(tmp_path: Path) -> None:
    bad = _est_drift(tmp_path, f"det={_DETECTOR[0]};seg=facebook/sam2-hiera-tiny;depth={_DEPTH[0]}")
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [bad])
    assert done.returncode == 2, done.stdout + done.stderr
    assert "sam2-hiera-tiny" in done.stdout


def test_an_id_named_only_in_prose_does_not_establish_agreement(tmp_path: Path) -> None:
    """A whole-file substring hit is weaker than the line it used to print.

    Here the staged id appears in the artifact — as a rejected candidate — and the field that says
    what was actually loaded says something else. The old check reported '=== verified'.
    """
    path = tmp_path / "pr08_geom_tol.json"
    path.write_text(json.dumps({
        "mask_method": {
            "name": "red-pixel-heuristic",
            "params": {"checkpoints": {"DEPTH_MODEL_CHECKPOINT": _RELATIVE_DEPTH}},
            "candidates_not_used": [_DEPTH[0], _SEGMENTER[0]],
        },
    }))
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [path])
    assert done.returncode == 2, done.stdout + done.stderr


def test_missing_artifacts_are_unverified_and_never_qualified(tmp_path: Path) -> None:
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [tmp_path / "pr08_geom_tol.json"])
    assert done.returncode == 0
    rows = _checks(done.stdout)
    assert rows["depth-vs-pr08_geom_tol.json"] == "unverified"


def test_the_checker_refuses_an_empty_want_list(tmp_path: Path) -> None:
    """Nothing to check is not everything agreeing."""
    done = _run_checker(tmp_path, _write_adapter(tmp_path), wants=[])
    # PR08_WANT="" — set explicitly by _run_checker's join of an empty list.
    assert done.returncode == 2
    assert "FATAL" in done.stdout


def test_the_real_adapter_and_the_real_job_agree_today(tmp_path: Path) -> None:
    """The drift test the whole mechanism exists for, run against the two real files."""
    if not _ADAPTER.is_file():
        pytest.skip("scripts/estimators/apple_sam2.py is not in this tree")
    text = _text()
    wants = [
        ("segmenter", _default("SAM2_MODEL_ID", text), _default("SAM2_MODEL_REVISION", text)),
        ("detector", _default("GDINO_MODEL_ID", text), _default("GDINO_MODEL_REVISION", text)),
        ("depth", _default("DEPTH_MODEL_ID", text), _default("DEPTH_MODEL_REVISION", text)),
    ]
    done = _run_checker(tmp_path, _ADAPTER, wants=wants)
    assert done.returncode == 0, (
        "102 and scripts/estimators/apple_sam2.py name different checkpoints:\n"
        + done.stdout + done.stderr
    )
    rows = _checks(done.stdout)
    for role in ("segmenter", "detector", "depth"):
        assert rows[f"{role}-id-vs-adapter"] == "agrees"
        # The adapter is the single source of truth for the REVISIONS too, and a revision the job
        # cannot extract is a revision the adapter can resolve for itself at load time.
        assert rows[f"{role}-revision-vs-adapter"] == "agrees", (
            f"apple_sam2.py declares no extractable 40-hex revision constant for the {role}; the "
            f"job then stages a pinned commit that the loader is free to ignore"
        )


# -- the manifest builder --------------------------------------------------------------------------


def _snapshot(tmp_path: Path, name: str, files: dict[str, bytes], dangling: str | None = None) -> Path:
    """A hub-cache snapshot: entries are symlinks into blobs/, which is what the real layout is."""
    root = tmp_path / name
    blobs = root / "blobs"
    snap = root / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for i, (rel, data) in enumerate(files.items()):
        blob = blobs / f"blob{i}"
        blob.write_bytes(data)
        target = snap / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(blob, target.parent))
    if dangling is not None:
        (snap / dangling).symlink_to("../../blobs/never-arrived")
    return snap


def _run_manifest(
    tmp_path: Path,
    snap: Path,
    expected: list[str],
    *,
    checks: list[tuple[str, str, str]],
    unverified: list[str],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    script = tmp_path / "manifest.py"
    script.write_text(_heredoc(_text(), "PY_MANIFEST"))
    listing = tmp_path / "files.txt"
    listing.write_text("\n".join(expected) + "\n")
    staged = tmp_path / "staged.tsv"
    staged.write_text("".join(
        f"{role}\trepo/{role}\t{'a' * 40}\t{snap}\t{listing}\n"
        for role in ("segmenter", "detector", "depth")
    ))
    manifest = tmp_path / "PR08_ESTIMATORS_STAGED.json"
    child = dict(os.environ)
    child["PR08_CHECKS"] = "\x1f".join("\t".join(c) for c in checks)
    child["PR08_UNVERIFIED"] = "\x1f".join(unverified)
    child["REPO_ROOT"] = str(tmp_path)
    done = subprocess.run([sys.executable, str(script), str(manifest), str(staged)],
                          capture_output=True, text=True, env=child)
    return done, manifest


def test_manifest_records_every_file_and_qualifies_when_everything_agreed(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "hub", {"config.json": b"{}", "model.safetensors": b"weights"})
    done, manifest = _run_manifest(
        tmp_path, snap, ["config.json", "model.safetensors"],
        checks=[("segmenter-id-vs-adapter", "agrees", "x")], unverified=[],
    )
    assert done.returncode == 0, done.stdout + done.stderr
    doc = json.loads(manifest.read_text())
    assert doc["staged_qualified"] is True
    assert doc["checkpoints"][0]["n_files"] == 2


def test_manifest_refuses_a_dangling_snapshot_entry(tmp_path: Path) -> None:
    """The medium finding, as a regression.

    ``if not p.is_file(): continue`` skipped exactly the entry a 429 leaves behind, in the one
    branch that would have caught the failure the header names — and the run reported success.
    """
    snap = _snapshot(tmp_path, "hub", {"config.json": b"{}"}, dangling="model.safetensors")
    done, manifest = _run_manifest(
        tmp_path, snap, ["config.json", "model.safetensors"],
        checks=[("x", "agrees", "")], unverified=[],
    )
    assert done.returncode != 0, done.stdout
    assert "dangling" in done.stderr
    assert "model.safetensors" in done.stderr
    assert not manifest.exists(), "a manifest was written over a partial snapshot"


def test_manifest_refuses_a_snapshot_missing_a_file_the_hub_lists(tmp_path: Path) -> None:
    """`hf download` exiting 0 is not the same claim as 'the snapshot is complete'."""
    snap = _snapshot(tmp_path, "hub", {"config.json": b"{}"})
    done, manifest = _run_manifest(
        tmp_path, snap, ["config.json", "model.safetensors"],
        checks=[("x", "agrees", "")], unverified=[],
    )
    assert done.returncode != 0, done.stdout
    assert "model.safetensors" in done.stderr
    assert not manifest.exists()


def test_manifest_does_not_qualify_when_a_check_is_unverified(tmp_path: Path) -> None:
    """The low finding: the flag came from a parallel list, not from the checks themselves.

    Feeding an ``unverified`` row with an empty reason list used to yield ``staged_qualified:
    true`` — in the one field the opt-in rule says must never default to true.
    """
    snap = _snapshot(tmp_path, "hub", {"config.json": b"{}"})
    done, manifest = _run_manifest(
        tmp_path, snap, ["config.json"],
        checks=[("depth-vs-artifact", "unverified", "")], unverified=[],
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(manifest.read_text())["staged_qualified"] is False


def test_manifest_does_not_qualify_when_nothing_was_checked_at_all(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "hub", {"config.json": b"{}"})
    done, manifest = _run_manifest(tmp_path, snap, ["config.json"], checks=[], unverified=[])
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(manifest.read_text())["staged_qualified"] is False


# -- the file list the probe extracts ---------------------------------------------------------------


def _run_filelist(tmp_path: Path, meta: object, rev: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "filelist.py"
    script.write_text(_heredoc(_text(), "PY_FILELIST"))
    body = tmp_path / "meta.json"
    body.write_text(json.dumps(meta) if not isinstance(meta, str) else meta)
    return subprocess.run(
        [sys.executable, str(script), str(body), rev, str(tmp_path / "files.txt")],
        capture_output=True, text=True,
    )


def test_filelist_refuses_when_the_api_resolved_another_commit(tmp_path: Path) -> None:
    done = _run_filelist(tmp_path, {"sha": "b" * 40, "siblings": [{"rfilename": "config.json"}]},
                         "a" * 40)
    assert done.returncode != 0
    assert "not a pin" in done.stderr


def test_filelist_refuses_an_unparseable_body(tmp_path: Path) -> None:
    done = _run_filelist(tmp_path, "<html>rate limited</html>", "a" * 40)
    assert done.returncode != 0
    assert "cannot parse" in done.stderr


def test_filelist_writes_the_names_it_will_be_checked_against(tmp_path: Path) -> None:
    done = _run_filelist(
        tmp_path,
        {"sha": "a" * 40, "siblings": [{"rfilename": "b.json"}, {"rfilename": "a.json"}]},
        "a" * 40,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "files.txt").read_text().split() == ["a.json", "b.json"]


# -- the committed measurement artifacts, which now carry id@revision -----------------------------
#
# The fix pass pinned commits alongside repo ids in the adapter (AC-04), so `estimators.version`
# became `seg=<id>@<sha>;...` while this checker still compared the whole string against a bare id.
# On a correctly configured run that is a FATAL "DISAGREES" -- a refusal of the right answer, which
# is worse than a missed check because it teaches the operator to skip the job. Caught by the
# recheck pass; these are its regressions.


def _est_drift_pinned(tmp_path: Path, *, seg: str, det: str, depth: str) -> Path:
    p = tmp_path / "pr08_est_drift.json"
    p.write_text(json.dumps(
        {"estimators": {"version": f"det={det};seg={seg};depth={depth};prompt='apple.'"}}
    ))
    return p


def _pin(role_id: str, revision: str) -> str:
    return f"{role_id}@{revision}"


def test_an_est_drift_artifact_pinned_to_the_staged_commits_agrees(tmp_path: Path) -> None:
    art = _est_drift_pinned(
        tmp_path,
        seg=_pin(*_SEGMENTER), det=_pin(*_DETECTOR), depth=_pin(*_DEPTH),
    )
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 0, done.stdout + done.stderr
    rows = _checks(done.stdout)
    for role in ("segmenter", "detector", "depth"):
        assert rows[f"{role}-vs-pr08_est_drift.json"] == "agrees"


def test_a_bare_id_with_no_commit_still_agrees_on_the_id(tmp_path: Path) -> None:
    """The older shape must not become a FATAL either -- it is less informative, not wrong."""
    art = _est_drift_pinned(tmp_path, seg=_SEGMENTER[0], det=_DETECTOR[0], depth=_DEPTH[0])
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_right_repo_at_the_wrong_commit_is_fatal(tmp_path: Path) -> None:
    """The drift the id alone cannot see: same segmenter repo, different weights."""
    art = _est_drift_pinned(
        tmp_path,
        seg=_pin(_SEGMENTER[0], "0" * 40), det=_pin(*_DETECTOR), depth=_pin(*_DEPTH),
    )
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 2, done.stdout + done.stderr
    assert "DISAGREES ON THE COMMIT" in done.stdout


def test_a_different_repo_is_still_fatal_when_pinned(tmp_path: Path) -> None:
    art = _est_drift_pinned(
        tmp_path,
        seg=_pin("facebook/sam2-hiera-tiny", _SEGMENTER[1]),
        det=_pin(*_DETECTOR), depth=_pin(*_DEPTH),
    )
    done = _run_checker(tmp_path, _write_adapter(tmp_path), [art])
    assert done.returncode == 2, done.stdout + done.stderr
    assert "DISAGREES" in done.stdout


# -- the defect that has now cost two queue cycles ------------------------------------------------
#
# 100_fetch_pr08_source, 2026-08-20: "failed with the download SUCCEEDED". hf >= 1.28 decorates
# stdout with a green "✓ Downloaded" and an indented "  path: <dir>", so a `tail -n 1` directory
# guard rejects a good fetch. This job reproduced it exactly on 2026-08-22 as job 189452 -- 12
# seconds in, after all nine files had landed. These tests are the regression for the SECOND time.


def test_the_download_asks_for_undecorated_output() -> None:
    text = _text()
    assert "--quiet" in text, (
        "hf decorates stdout without --quiet, which is how job 189452 called a successful "
        "download FATAL"
    )
    for line in text.splitlines():
        if "hf@latest download" in line:
            assert "--quiet" in line, line


def test_the_snapshot_path_is_not_taken_by_position_alone() -> None:
    """--quiet is the fix; not depending on the layout is the belt. A future client that ignores
    its own --quiet contract must not be able to break this again."""
    text = _text()
    i = text.find("hf@latest download")
    assert i != -1
    window = text[i : i + 600]
    assert '[[ -d "${line}" ]]' in window, (
        "the path must be recovered by testing which line IS a directory, not by position"
    )
    assert "path:" in window, "the decoration must be stripped explicitly too"


def test_the_reason_is_recorded_at_the_code_that_carries_it() -> None:
    """A bug that recurs after being written down once was written down in the wrong place. It is
    now in the function, not only in the task file."""
    text = _text()
    i = text.find("hf@latest download")
    window = text[max(0, i - 1200) : i]
    assert "100_fetch_pr08_source" in window
    assert "189452" in window


# -- the exit code has to carry information -------------------------------------------------------
#
# Job 189453 staged 4.8 GB, wrote the manifest, agreed on every id and revision it could check, and
# Slurm recorded FAILED. The measurement artifacts it wanted to check against are produced BY
# measurements that need these weights, so a first run could not exit 0 by construction. An exit
# code that is always the same carries nothing, and a job that reports failure for doing the right
# thing teaches the operator to stop reading it.


def test_a_not_yet_written_measurement_artifact_is_not_a_failure() -> None:
    text = _text()
    assert "EXPECTED_ABSENT" in text and "GENUINE_GAPS" in text, (
        "the two kinds of 'unverified' must be distinguished, or a correct first run cannot exit 0"
    )
    i = text.find("if [[ ${#GENUINE_GAPS[@]} -eq 0 ]]; then")
    assert i != -1, "no branch exits 0 on lifecycle-only gaps"
    assert "exit 0" in text[i : i + 900]


def test_a_real_verification_gap_still_exits_three() -> None:
    """Loosening the lifecycle case must not loosen the case it was carved out of."""
    text = _text()
    i = text.find("STAGED, NOT VERIFIED")
    assert i != -1
    assert "exit 3" in text[i:], "the genuine-gap branch must still exit 3"


def test_staged_qualified_is_false_in_both_cases() -> None:
    """Exiting 0 says the job did its work. It must not start claiming the ids were verified."""
    text = _text()
    assert "qualified = bool(checks) and all(" in text
    assert "and not unverified" in text, (
        "staged_qualified must stay keyed to ALL unverified reasons, lifecycle ones included"
    )
    i = text.find("if [[ ${#GENUINE_GAPS[@]} -eq 0 ]]; then")
    assert "staged_qualified is FALSE" in text[i : i + 900]


def test_the_exit_status_block_documents_the_carve_out() -> None:
    i = _text().find("# EXIT STATUS")
    block = _text()[i : i + 1400]
    assert "NOT WRITTEN YET is exit 0" in block
