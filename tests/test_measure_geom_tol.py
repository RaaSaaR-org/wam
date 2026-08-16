"""Tests for ``scripts/measure_geom_tol.py`` — PR-08 §6 G0b's ``GEOM_TOL``.

GEOM_TOL is a number that gets committed once, before any clip is generated, and then decides for
every subsequent restyle whether its geometry drifted too far. Nothing downstream re-derives it. So
the failures worth pinning are not "it crashed":

  it returns a plausible   The apple is red, the table is not, and thresholding red pixels produces
  number from a            a number that looks exactly like a measurement. A GEOM_TOL that is wrong
  non-segmenter            by 2x is invisible — no gate fires, no run fails, every verdict downstream
                           inherits it. So the default path must FAIL, loudly, naming what is
                           missing, rather than fall back to anything.

  it fills occlusion       The Dex3 hand covers the apple during the grasp and the apple leaves
  with zeros               frame. Those steps have no displacement. Folding them in as 0 px pulls
                           the median down and TIGHTENS the gate, which reads as conservative and is
                           simply a different quantity. They must be dropped and counted.

  it measures the          "per-step" is undefined in PR-08 §6 and GEOM_TOL scales roughly linearly
  wrong step               with the reading. The knob must exist and the artifact must say which
                           reading produced the number.

  it forgets who           The identical estimator has to be re-runnable on the restyled clips at
  measured it              gate time. Masks with no named, versioned producer cannot be recorded,
                           so they are refused rather than measured.

  it writes the gate       The default --out used to be a CWD-relative path. Invoked from anywhere
  artifact wherever        but the repository root it wrote the gate artifact under the caller's
  it was invoked from      directory, exited 0, and every consumer looking under the repository
                           reported it missing. The default is absolute and repo-anchored, and it
                           is a TRACKED path: PR-08 §8 item 4 wants GEOM_TOL *committed*, and
                           runs/ is gitignored, so an artifact there can never be a pre-commitment.

  a smoke test becomes     coverage is a fraction of the steps that were DECODED, so --limit 3 over
  the committed number     402 episodes scores coverage 1.000 and every other field reads like a
                           finished measurement. Either flag forces gate_qualified: false and
                           n_episodes_found is counted before --limit truncates, so a consumer can
                           assert the two counts match.

Every corpus here is synthetic and carries the answer in its construction: a solid square whose
centroid moves by a known (dx, dy) each frame, so the expected median is arithmetic, not a fixture.
Nothing decodes video and nothing needs the real AppleToPlate corpus — these tests pass on a machine
with no data on it at all.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import measure_geom_tol as mgt  # noqa: E402

CANVAS = 128
BLOB = 8  # 8x8 = 64 px, comfortably over the default --min-area-px of 40


def blob_mask(x0: int, y0: int, canvas: int = CANVAS) -> np.ndarray:
    """A solid square whose centroid is exactly (x0 + (BLOB-1)/2, y0 + (BLOB-1)/2)."""
    m = np.zeros((canvas, canvas), dtype=np.uint8)
    m[y0:y0 + BLOB, x0:x0 + BLOB] = 1
    return m


def make_corpus(
    tmp: Path,
    episodes: dict[str, list[tuple[int, int] | None]],
    *,
    gate_qualified: bool = True,
    sidecar: bool = True,
    canvas: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    """A clip directory plus a mask directory keyed by clip stem.

    ``episodes`` maps a clip stem to its per-frame blob top-left corners; ``None`` means the object
    is not visible in that frame (occlusion, or out of frame), written as an all-zero mask, which is
    what a real segmenter emits when it finds nothing.

    The .mp4 files are empty. Under ``--method precomputed`` nothing decodes them, and that they are
    never opened is part of what is being asserted: the mask path must not depend on the pixels.
    """
    corpus = tmp / "corpus"
    masks = tmp / "masks"
    corpus.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    if sidecar:
        meta = {"method": "synthetic-oracle", "version": "1.0.0",
                "gate_qualified": gate_qualified, "prompt": "apple"}
        (masks / "masks.meta.json").write_text(json.dumps(meta, indent=2))
    for stem, corners in episodes.items():
        (corpus / f"{stem}.mp4").write_bytes(b"")
        ep_dir = masks / stem
        ep_dir.mkdir(parents=True, exist_ok=True)
        side = (canvas or {}).get(stem, CANVAS)
        for i, corner in enumerate(corners):
            arr = (np.zeros((side, side), dtype=np.uint8) if corner is None
                   else blob_mask(corner[0], corner[1], side))
            np.save(ep_dir / f"{i:06d}.npy", arr)
    return corpus, masks


def walk(start: tuple[int, int], step: tuple[int, int], n: int) -> list[tuple[int, int]]:
    return [(start[0] + i * step[0], start[1] + i * step[1]) for i in range(n)]


def run(argv: list[str]) -> int:
    return mgt.main(argv)


# -- the measurement itself ----------------------------------------------------------------------


def test_recovers_the_known_median(tmp_path: Path) -> None:
    """Three episodes moving 2, 5 and 8 px per step. The pooled median is 5.0 and nothing else.

    ep1 moves (3, 4) per frame: Euclidean 5.0, Manhattan 7.0. A script that summed the axes instead
    of taking the hypotenuse would report a median of 7.0 here, so this fixture separates the two.
    """
    corpus, masks = make_corpus(tmp_path, {
        "ep0": walk((10, 10), (2, 0), 5),   # 2.0 px per step, x only
        "ep1": walk((10, 40), (3, 4), 5),   # 5.0 px per step, diagonal
        "ep2": walk((60, 10), (0, 8), 5),   # 8.0 px per step, y only
    })
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out)])
    assert rc == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert rec["GEOM_TOL_px"] == pytest.approx(5.0)
    assert rec["n_episodes"] == 3
    assert rec["n_frames"] == 15
    assert rec["n_steps_total"] == 12
    assert rec["n_steps_measured"] == 12
    assert rec["n_steps_dropped_object_not_visible"] == 0
    assert rec["coverage"] == pytest.approx(1.0)
    assert rec["headline_valid"] is True
    assert rec["gate_qualified"] is True

    per = {e["episode"]: e for e in rec["per_episode"]}
    assert per["ep0"]["median_px"] == pytest.approx(2.0)
    assert per["ep1"]["median_px"] == pytest.approx(5.0)
    assert per["ep2"]["median_px"] == pytest.approx(8.0)


def test_records_the_full_distribution_not_only_the_median(tmp_path: Path) -> None:
    """PR-08 §6 records GEOM_TOL; a median alone hides a corpus that is bimodal in exactly the way
    a park-then-transfer teleop episode is."""
    corpus, masks = make_corpus(tmp_path, {
        "ep0": walk((10, 10), (2, 0), 5),
        "ep1": walk((10, 40), (3, 4), 5),
        "ep2": walk((60, 10), (0, 8), 5),
    })
    out = tmp_path / "geom_tol.json"
    dump = tmp_path / "displacements.npy"
    assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out),
                "--dump-displacements", str(dump)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    dist = rec["distribution"]

    assert dist["n"] == 12
    assert dist["min_px"] == pytest.approx(2.0)
    assert dist["max_px"] == pytest.approx(8.0)
    assert dist["mean_px"] == pytest.approx(5.0)
    assert dist["percentiles_px"]["p50"] == pytest.approx(5.0)
    assert dist["percentiles_px"]["p0"] == pytest.approx(2.0)
    assert dist["percentiles_px"]["p100"] == pytest.approx(8.0)
    # Every displacement is in exactly one bin and no displacement was lost on the way.
    assert sum(dist["histogram"]["counts"]) == 12

    raw = np.load(dump)
    assert sorted(np.round(raw, 6)) == pytest.approx([2.0] * 4 + [5.0] * 4 + [8.0] * 4)


def test_step_frames_is_the_knob_pr08_leaves_undefined(tmp_path: Path) -> None:
    """A step of 2 frames over constant 2 px/frame motion is 4.0 px, and the artifact says so."""
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out),
                "--step-frames", "2"]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["GEOM_TOL_px"] == pytest.approx(4.0)
    assert rec["step_frames"] == 2
    assert rec["n_steps_measured"] == 3  # offsets 0->2, 1->3, 2->4
    assert "does not define" in rec["step_definition"]


def test_invisible_object_is_dropped_and_counted_never_zero(tmp_path: Path) -> None:
    """Frames 1 and 2 have an empty mask: the hand is over the apple.

    Three of the four steps are unmeasurable. The one that is measurable moved 6 px, so the median
    is 6.0. A script that folded the occluded steps in as 0 px would report 0.0 — a tighter gate
    that no restyle could pass, from a displacement that was never observed.
    """
    corpus, masks = make_corpus(tmp_path, {
        "ep0": [(10, 10), None, None, (28, 10), (34, 10)],
    })
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out)])

    rec = json.loads(out.read_text())
    assert rec["GEOM_TOL_px"] == pytest.approx(6.0)
    assert rec["n_steps_total"] == 4
    assert rec["n_steps_measured"] == 1
    assert rec["n_steps_dropped_object_not_visible"] == 3
    assert rec["coverage"] == pytest.approx(0.25)
    # Coverage is under the default floor, so the number is recorded (PR-08 §6 requires that) but
    # is not offered as a headline, and the exit status says so.
    assert rec["headline_valid"] is False
    assert rc == mgt.EXIT_NOT_GATE_QUALIFIED


# -- honesty about the method ----------------------------------------------------------------------


def test_auto_fails_loudly_when_no_segmenter_is_wired(tmp_path: Path, capsys) -> None:
    """The default path with no masks must name what is missing and write nothing at all."""
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--out", str(out)])
    assert rc == mgt.EXIT_FATAL
    assert not out.exists(), "a failed measurement must not leave an artifact behind"

    err = capsys.readouterr().err
    assert "no gate-qualified object segmenter is wired" in err
    assert "sam2" in err
    assert "masks.meta.json" in err
    # The download route exists and is not taken silently.
    assert "ASK" in err


def test_the_heuristic_cannot_be_reached_by_accident(tmp_path: Path) -> None:
    """`auto` never resolves to the red-pixel heuristic; it has to be typed out in full."""
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    args = mgt._parse_args(["--corpus", str(corpus)])
    with pytest.raises(mgt.MethodUnavailable):
        mgt.resolve_method(args)

    typed = mgt._parse_args(["--corpus", str(corpus), "--method", "hsv-red-diagnostic"])
    method = mgt.resolve_method(typed)
    assert method.name == "hsv-red-diagnostic"
    assert method.gate_qualified is False


def test_masks_without_provenance_are_refused(tmp_path: Path, capsys) -> None:
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)}, sidecar=False)
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()
    assert "masks.meta.json" in capsys.readouterr().err


def test_an_unstated_gate_claim_is_not_a_claim(tmp_path: Path) -> None:
    """A sidecar that does not assert gate_qualified is treated as not asserting it."""
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    (masks / "masks.meta.json").write_text(json.dumps({"method": "mystery", "version": "0"}))
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out)])
    rec = json.loads(out.read_text())

    assert rc == mgt.EXIT_NOT_GATE_QUALIFIED
    assert rec["mask_method"]["gate_qualified"] is False
    assert rec["gate_qualified"] is False
    # Recorded regardless of verdict — PR-08 §6.
    assert rec["GEOM_TOL_px"] == pytest.approx(2.0)


def test_mixed_geometry_is_fatal(tmp_path: Path, capsys) -> None:
    """§4 subtracts EST_DRIFT_P95 (px) from GEOM_TOL (px). That is arithmetic only on one grid."""
    corpus, masks = make_corpus(
        tmp_path,
        {"ep0": walk((10, 10), (2, 0), 5), "ep1": walk((10, 10), (2, 0), 5)},
        canvas={"ep1": 96},
    )
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_FATAL
    assert "EST_DRIFT_P95" in capsys.readouterr().err


def test_empty_corpus_is_not_a_pass(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(empty), "--masks", str(empty),
                "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()


# -- the artifact ----------------------------------------------------------------------------------


def test_artifact_records_what_pr08_asks_to_be_recorded(tmp_path: Path) -> None:
    corpus, masks = make_corpus(tmp_path, {
        "ep0": walk((10, 10), (2, 0), 5),
        "ep1": walk((10, 40), (3, 4), 5),
    })
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    for key in ("GEOM_TOL_px", "distribution", "n_episodes", "n_frames", "mask_method",
                "measured_date", "measured_utc", "corpus", "units", "step_frames",
                "est_drift_p95_px", "notes"):
        assert key in rec, f"artifact is missing {key}"

    assert rec["schema"] == mgt.SCHEMA
    assert rec["rule"] == "T40_RULE_V1"
    assert rec["mask_method"]["name"] == "synthetic-oracle"
    assert rec["mask_method"]["version"] == "1.0.0"
    assert rec["units"] == f"pixels at {CANVAS}x{CANVAS}"
    assert rec["measured_date"] == __import__("datetime").date.today().isoformat()
    # EST_DRIFT_P95 is not measurable until the Isaac annotators are wired, and the artifact says
    # that rather than assuming zero.
    assert rec["est_drift_p95_px"] is None
    assert rec["geom_tol_minus_est_drift_px"] is None
    assert "isaac_binding.py" in rec["est_drift_p95_blocked_by"]


def test_the_artifact_says_which_step_it_was_gated_under(tmp_path: Path) -> None:
    """The sbatch quotes GEOM_TOL_px; GEOM_TOL scales ~linearly with the step it was measured at.

    A number without its step is a gate that is wrong by an unknown factor and says nothing about
    it, so the consumer has to be able to assert the step — which means the artifact has to carry
    it, name the assertion, and warn that the two are coupled.
    """
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out),
                "--step-frames", "3"]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert rec["step_frames"] == 3
    assert "step_frames" in rec["step_definition"] or "3 source frame" in rec["step_definition"]
    assert any("step_frames" in a for a in rec["consumer_asserts"])
    assert any("scales" in n and "step_frames" in n for n in rec["notes"])


# -- FIX 1: the default output path is anchored to the repo, not to the caller's CWD ---------------


def test_default_out_is_absolute_and_repo_anchored() -> None:
    """A CWD-relative default silently writes the gate artifact wherever the caller happened to be.

    The failure it produces is the worst shape available: exit 0, an artifact on disk, and a
    consumer under the repository reporting it missing. Nobody looks for a path bug in a measurement
    that reported success.
    """
    assert mgt.DEFAULT_OUT.is_absolute(), "DEFAULT_OUT must not depend on the caller's CWD"
    assert mgt.DEFAULT_OUT == mgt._REPO_ROOT / mgt.DEFAULT_OUT_REL
    assert mgt.DEFAULT_OUT.parent == mgt._REPO_ROOT / "configs" / "transfer25"
    # Every other path in the module is repo-anchored; the artifact path is not the exception.
    assert mgt._REPO_ROOT in mgt.DEFAULT_OUT.parents


def test_the_default_artifact_does_not_follow_the_callers_cwd(tmp_path: Path, monkeypatch) -> None:
    """Run from an unrelated directory with no --out: the artifact lands at the anchor, not here."""
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    anchor = tmp_path / "anchor" / "configs" / "transfer25" / "pr08_geom_tol.json"
    monkeypatch.setattr(mgt, "DEFAULT_OUT", anchor)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run(["--corpus", str(corpus), "--masks", str(masks)]) == mgt.EXIT_OK

    assert anchor.is_file(), "the artifact must land at the repo-anchored default"
    assert list(elsewhere.iterdir()) == [], (
        f"the caller's CWD must stay untouched, found {[p.name for p in elsewhere.iterdir()]}")
    assert json.loads(anchor.read_text())["artifact_path"] == str(anchor)


# -- FIX 3: the committed artifact is on a TRACKED path, with a sidecar ----------------------------


def test_the_committed_artifact_path_is_not_gitignored() -> None:
    """PR-08 §8 item 4 wants GEOM_TOL "measured and COMMITTED" before a clip is generated.

    A file under a gitignored directory cannot be committed, so it cannot be a pre-commitment — it
    is scratch with the right shape, and the rule it is supposed to satisfy is unsatisfiable by
    construction. This is checked against git itself rather than against a copy of .gitignore.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:  # pragma: no cover - git is present in this repo's CI
        pytest.skip("git not available")

    def ignored(rel: str) -> bool:
        return subprocess.run(["git", "-C", str(mgt._REPO_ROOT), "check-ignore", "-q", rel],
                              capture_output=True).returncode == 0

    assert ignored("runs/pr08/geom_tol.json"), (
        "the premise of this test has changed: runs/ is no longer gitignored")
    assert not ignored(mgt.DEFAULT_OUT_REL), (
        f"{mgt.DEFAULT_OUT_REL} is gitignored, so GEOM_TOL could never be committed there")
    assert not mgt.DEFAULT_OUT_REL.startswith("runs/")


def test_a_sha256_sidecar_is_written_next_to_the_artifact(tmp_path: Path) -> None:
    """Same discipline as configs/transfer25/pr08_style_partition.json.sha256.

    The digest is what makes "the file the gate read is the file that was committed" checkable with
    sha256sum instead of assumed.
    """
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "pr08_geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out)]) == mgt.EXIT_OK

    side = out.parent / (out.name + ".sha256")
    assert side.is_file(), "the gate artifact must travel with its digest"
    assert side == mgt.sidecar_path(out)
    assert side.read_text().strip() == hashlib.sha256(out.read_bytes()).hexdigest()
    assert json.loads(out.read_text())["artifact_sha256_sidecar"] == str(side)

    # A one-byte edit after the fact is what the sidecar exists to catch.
    out.write_bytes(out.read_bytes().replace(b'"GEOM_TOL_px": 2.0', b'"GEOM_TOL_px": 9.0'))
    assert side.read_text().strip() != hashlib.sha256(out.read_bytes()).hexdigest()


# -- FIX 2: a partial measurement is never the committed number ------------------------------------


def _five_episode_corpus(tmp_path: Path) -> tuple[Path, Path]:
    return make_corpus(tmp_path, {f"ep{i}": walk((10, 10 + 12 * i), (2 + i, 0), 5)
                                  for i in range(5)})


def test_limit_cannot_produce_a_gate_qualified_number(tmp_path: Path) -> None:
    """--limit 2 of 5 scores coverage 1.000 and measures the wrong corpus. gate_qualified is false.

    Nothing in the numbers can catch this: coverage is a fraction of the steps that were DECODED, so
    a sample scores perfectly by construction. The disqualification therefore comes from the flag,
    not from the result.
    """
    corpus, masks = _five_episode_corpus(tmp_path)
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out), "--limit", "2"])
    rec = json.loads(out.read_text())

    assert rec["coverage"] == pytest.approx(1.0), "the trap: coverage cannot see a partial run"
    assert rec["headline_valid"] is True
    assert rec["mask_method"]["gate_qualified"] is True
    # ... and yet:
    assert rec["gate_qualified"] is False
    assert rec["partial_measurement"] is True
    assert rc == mgt.EXIT_NOT_GATE_QUALIFIED
    assert rec["limit"] == 2
    assert any("--limit" in r for r in rec["gate_disqualified_reasons"]), rec[
        "gate_disqualified_reasons"]
    # PR-08 §6 records the number regardless of verdict; it is stamped, not withheld.
    assert rec["GEOM_TOL_px"] is not None


def test_max_frames_cannot_produce_a_gate_qualified_number(tmp_path: Path) -> None:
    corpus, masks = _five_episode_corpus(tmp_path)
    out = tmp_path / "geom_tol.json"
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out),
              "--max-frames", "120"])
    rec = json.loads(out.read_text())

    assert rc == mgt.EXIT_NOT_GATE_QUALIFIED
    assert rec["gate_qualified"] is False
    assert rec["partial_measurement"] is True
    assert rec["max_frames"] == 120
    assert any("--max-frames" in r for r in rec["gate_disqualified_reasons"])


def test_n_episodes_found_counts_the_corpus_not_the_sample(tmp_path: Path) -> None:
    """The pair a consumer asserts on. Counted before --limit truncates, or it would always agree."""
    corpus, masks = _five_episode_corpus(tmp_path)
    out = tmp_path / "geom_tol.json"
    run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out), "--limit", "2"])
    rec = json.loads(out.read_text())

    assert rec["n_episodes"] == 2
    assert rec["n_episodes_found"] == 5
    assert any("n_episodes == n_episodes_found" in a for a in rec["consumer_asserts"])


def test_a_whole_corpus_run_carries_no_disqualification(tmp_path: Path) -> None:
    """The control: without the flags the same corpus is gate-qualified and the counts agree.

    Without this, forcing gate_qualified=false everywhere would pass every test above.
    """
    corpus, masks = _five_episode_corpus(tmp_path)
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert rec["gate_qualified"] is True
    assert rec["partial_measurement"] is False
    assert rec["gate_disqualified_reasons"] == []
    assert rec["n_episodes"] == rec["n_episodes_found"] == 5
    assert rec["limit"] == 0 and rec["max_frames"] == 0


def test_the_partial_run_says_so_on_stderr(tmp_path: Path, capsys) -> None:
    """A flag in a JSON file nobody opens is not a warning. The operator sees it too."""
    corpus, masks = _five_episode_corpus(tmp_path)
    run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(tmp_path / "g.json"),
         "--limit", "2"])
    err = capsys.readouterr().err
    assert "PARTIAL MEASUREMENT" in err
    assert "2 of 5 episodes" in err
    assert "MUST NOT be committed" in err


def test_centroid_helpers_agree_with_arithmetic() -> None:
    """The two lowest-level pieces, checked against numbers computed by hand."""
    assert mgt.centroid_of_mask(blob_mask(10, 20), largest_component=False, min_area=1) == (
        pytest.approx(10 + (BLOB - 1) / 2), pytest.approx(20 + (BLOB - 1) / 2))
    assert mgt.centroid_of_mask(np.zeros((16, 16)), largest_component=False, min_area=1) is None
    # A speck below --min-area-px is "not visible", not a centroid.
    tiny = np.zeros((16, 16), dtype=np.uint8)
    tiny[0, 0] = 1
    assert mgt.centroid_of_mask(tiny, largest_component=False, min_area=40) is None

    cents = [(0.0, 0.0), (3.0, 4.0), None, (3.0, 4.0)]
    d, dropped = mgt.displacements(cents, step=1)
    assert d == pytest.approx([5.0])
    assert dropped == 2
