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


# -- the shared PR-08 §4 segmenter -----------------------------------------------------------------
#
# §4 step 2 requires EST_DRIFT_P95 to be measured with THE SAME segmenter as GEOM_TOL, and §6
# subtracts the two. So the thing under test below is not "can it segment" — it is whether a run can
# end up with a number whose producer cannot be identified, cannot be re-run on the restyled clips,
# or is not the producer the other half of the budget used. Each of those subtracts cleanly to a
# plausible pixel figure and fires no gate.

import types  # noqa: E402

import measure_est_drift as ed  # noqa: E402

SAM2_SPEC = mgt.SAM2_ADAPTER_SPEC


@pytest.fixture(autouse=True)
def no_real_adapter(monkeypatch):
    """Every test in this file runs as if the real adapter were not on this machine.

    This is not tidiness. ``scripts/estimators/apple_sam2.py`` reaches for SAM2 and GroundingDINO
    checkpoints, and these tests are required to pass with no GPU, no network and no weights.
    Without this fixture the result of ``--method auto`` would depend on whether someone had staged
    two gigabytes of checkpoints on the machine running pytest, and the refusal tests above would
    start passing or failing for reasons that have nothing to do with the code under test. ``None``
    in ``sys.modules`` is the documented way to make an import of that name raise ImportError; tests
    that want an adapter install their own stub over the top of it.
    """
    monkeypatch.setitem(sys.modules, SAM2_SPEC, None)


def red_blob_segment(rgb: np.ndarray) -> np.ndarray:
    """A stub segmenter that only finds the object if it was handed RGB.

    The blob is pure red, so in RGB it is (255, 0, 0) and in the BGR array cv2 decodes it is
    (0, 0, 255). Keying on channel 0 makes the channel-order bug a wrong ANSWER rather than a
    crash — which is exactly its shape in production, where GroundingDINO would ground "apple" on
    whatever looks like an apple in a world where red is blue.
    """
    arr = np.asarray(rgb)
    return (arr[..., 0] > 200) & (arr[..., 1] < 50) & (arr[..., 2] < 50)


def install_adapter(
    monkeypatch,
    *,
    name: str = "sam2-hiera-large+gdino-base",
    version: str = "sam2 1.1.0 / transformers 4.51.3",
    gate_qualified: bool | None = True,
    checkpoints: dict[str, str] | None = None,
    available: bool | None = True,
    segment=red_blob_segment,
    with_depth: bool = True,
) -> types.ModuleType:
    """Install a stub of ``estimators.apple_sam2`` satisfying the estimator contract.

    ``None`` for ``gate_qualified`` or ``available`` means the attribute is ABSENT, which is a
    different statement from False and is tested separately: absent is what a module written without
    thinking about the gate looks like.
    """
    mod = types.ModuleType(SAM2_SPEC)
    mod.ESTIMATOR_NAME = name
    mod.ESTIMATOR_VERSION = version
    mod.segment = segment
    if with_depth:
        mod.estimate_depth = lambda rgb: np.zeros(np.asarray(rgb).shape[:2], dtype="float32")
    if gate_qualified is not None:
        mod.GATE_QUALIFIED = gate_qualified
    if checkpoints is None:
        checkpoints = {
            "SAM2_MODEL_CHECKPOINT": "facebook/sam2-hiera-large",
            "GROUNDING_DINO_MODEL_CHECKPOINT": "IDEA-Research/grounding-dino-base",
        }
    for attr, value in checkpoints.items():
        setattr(mod, attr, value)
    if available is not None:
        mod.available = lambda: available
    monkeypatch.setitem(sys.modules, SAM2_SPEC, mod)
    return mod


def bgr_frames(corners: list[tuple[int, int] | None], canvas: int = CANVAS) -> list[np.ndarray]:
    """Frames in cv2's colour order: a pure-red square on black, or an empty frame for ``None``."""
    out = []
    for corner in corners:
        frame = np.zeros((canvas, canvas, 3), dtype=np.uint8)
        if corner is not None:
            x0, y0 = corner
            frame[y0:y0 + BLOB, x0:x0 + BLOB] = (0, 0, 255)  # BGR red
        out.append(frame)
    return out


def install_video_frames(monkeypatch, frames: dict[str, list[np.ndarray]]) -> None:
    """Replace cv2 decoding, and ONLY cv2 decoding.

    The method's own segmenter, the centroid arithmetic, the largest-component rule and the
    drop-and-count of invisible objects all stay real. The single thing a machine with no codecs and
    no corpus cannot do is turn an .mp4 into pixels, so that is the single thing stubbed.
    """

    def fake(clip, method, min_area, max_frames):
        stack = frames[clip.stem]
        if max_frames > 0:
            stack = stack[:max_frames]
        cents = [
            mgt.centroid_of_mask(method.mask_fn(f, method), largest_component=True,
                                 min_area=min_area)
            for f in stack
        ]
        h, w = stack[0].shape[:2]
        return cents, (int(w), int(h)), 30.0

    monkeypatch.setattr(mgt, "episode_centroids_from_video", fake)


def test_the_adapter_is_never_imported_at_module_scope() -> None:
    """A module-scope import would drag transformers, torch and a checkpoint fetch into every test.

    It would also make ``import measure_geom_tol`` fail on any machine without the weights, which
    includes the machine the refusal messages exist to be read on. Checked structurally rather than
    by observing one import, because the failure mode is a line someone adds at the top later.
    """
    import ast

    tree = ast.parse(Path(mgt.__file__).read_text(encoding="utf-8"))
    # MODULE SCOPE only, in both directions. The earlier version of this test walked the whole tree
    # for Import/ImportFrom — which also forbade a legitimate lazy `import estimators.apple_sam2`
    # INSIDE a function, the very thing this module is supposed to do — while checking calls only
    # for an ast.Attribute named import_module, so a module-scope `from importlib import
    # import_module; import_module(SAM2_ADAPTER_SPEC)` or a module-scope `_try_import_sam2_adapter()`
    # walked straight past it. Both directions are pinned here.
    lazy_importers = {"import_module", "_try_import_sam2_adapter", "_import_sam2_adapter",
                      "sam2_method", "auto_sam2_method", "resolve_method"}
    module_level = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for node in module_level:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                assert not any(a.name.startswith("estimators") for a in sub.names)
            if isinstance(sub, ast.ImportFrom):
                assert not (sub.module or "").startswith("estimators")
            if isinstance(sub, ast.Call):
                called = (sub.func.attr if isinstance(sub.func, ast.Attribute)
                          else sub.func.id if isinstance(sub.func, ast.Name) else None)
                assert called not in lazy_importers, (
                    f"{called}() at module scope would import the adapter at import time; "
                    "the adapter must be reached from inside a call")

    # And the lazy import is a real one: it exists, and it is inside a function body.
    inside_a_function = [
        fn.name for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and any(isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "import_module"
                for c in ast.walk(fn))
    ]
    assert inside_a_function, "no function imports the adapter — the lazy import went missing"


def test_method_sam2_records_who_measured_and_with_which_weights(tmp_path, monkeypatch) -> None:
    """name, version, gate_qualified and the checkpoints — the artifact's only job is to make the
    identical estimator re-runnable on the restyled clips at gate time."""
    install_adapter(monkeypatch)
    args = mgt._parse_args(["--corpus", str(tmp_path), "--method", "sam2"])
    method = mgt.resolve_method(args)

    assert method.name == "sam2-hiera-large+gdino-base"
    assert method.version == "sam2 1.1.0 / transformers 4.51.3"
    assert method.gate_qualified is True
    assert method.frames_from == "video"
    assert SAM2_SPEC in method.provenance
    assert "facebook/sam2-hiera-large" in method.provenance
    assert "IDEA-Research/grounding-dino-base" in method.provenance
    assert method.params["checkpoints"]["SAM2_MODEL_CHECKPOINT"] == "facebook/sam2-hiera-large"
    assert method.params["estimator_spec"] == SAM2_SPEC


def test_sam2_hands_the_adapter_rgb_and_not_cv2s_bgr(tmp_path, monkeypatch) -> None:
    """The whole end-to-end run, on frames whose object is only findable in RGB.

    A channel-swapped frame does not crash a segmenter and does not return an error: it returns a
    mask of something else, or of nothing. Here "of nothing" would surface as coverage 0.0 and a
    GEOM_TOL of None, which reads as a fact about the corpus.
    """
    seen: list[tuple[int, int, int]] = []

    def recording_segment(rgb):
        arr = np.asarray(rgb)
        seen.append(tuple(int(v) for v in arr[12, 12]))  # inside the blob at (10, 10)
        return red_blob_segment(arr)

    install_adapter(monkeypatch, segment=recording_segment)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert seen[0] == (255, 0, 0), "the adapter was handed BGR; its contract says segment(rgb)"
    assert rec["GEOM_TOL_px"] == pytest.approx(2.0)
    assert rec["coverage"] == pytest.approx(1.0)
    assert rec["mask_method"]["name"] == "sam2-hiera-large+gdino-base"
    assert rec["mask_method"]["gate_qualified"] is True
    assert rec["mask_method"]["centroid_rule"] == "largest connected component by area"


def test_auto_selects_the_adapter_once_it_declares_its_weights(tmp_path, monkeypatch) -> None:
    """`auto` is allowed to pick the adapter — but only on the adapter's own statement about weights."""
    install_adapter(monkeypatch, available=True)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["mask_method"]["params"]["cli_method"] == "sam2"
    assert rec["gate_qualified"] is True


def test_auto_refuses_an_adapter_that_says_its_weights_are_missing(tmp_path, monkeypatch, capsys) -> None:
    """An unloaded segmenter returns empty masks; it does not raise. That is coverage 0.0 reported
    as a property of the corpus, so `auto` must decline rather than try."""
    install_adapter(monkeypatch, available=False)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()

    err = capsys.readouterr().err
    assert "no gate-qualified object segmenter is wired" in err
    assert "available() returned False" in err
    assert "--method sam2 types it explicitly" in err


def test_auto_refuses_an_adapter_that_will_not_say_whether_it_has_weights(
    tmp_path, monkeypatch, capsys
) -> None:
    """Silence is not consent. Nothing here goes looking for a checkpoint directory on the adapter's
    behalf: "a path called sam2-hiera-large exists" is a different claim from "this can segment"."""
    install_adapter(monkeypatch, available=None)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()

    err = capsys.readouterr().err
    assert "declares neither available() nor WEIGHTS_AVAILABLE" in err


def test_auto_reads_a_plain_weights_available_flag_too(tmp_path, monkeypatch) -> None:
    """The other half of the declaration contract, so an adapter with no probe to run can still
    make the statement."""
    mod = install_adapter(monkeypatch, available=None)
    mod.WEIGHTS_AVAILABLE = True
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--out", str(out)]) == mgt.EXIT_OK


def test_auto_still_refuses_exactly_as_before_when_the_adapter_is_absent(
    tmp_path, monkeypatch, capsys
) -> None:
    """The standing refusal is APPENDED to, never replaced: it names every segmenter package and
    every weight directory this machine was checked for, and narrowing it to one adapter would lose
    all of that."""
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()

    err = capsys.readouterr().err
    for standing in ("no gate-qualified object segmenter is wired", "sam2", "masks.meta.json",
                     "ASK", "isaac_binding.py"):
        assert standing in err
    assert "not importable" in err
    assert SAM2_SPEC in err


def test_masks_still_win_under_auto_when_both_are_available(tmp_path, monkeypatch) -> None:
    """Masks on disk were produced deliberately and carry their own named provenance. Preferring a
    locally importable adapter over the thing the operator pointed at would change which estimator
    produced GEOM_TOL without saying so anywhere."""
    install_adapter(monkeypatch, available=True)
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_OK
    assert json.loads(out.read_text())["mask_method"]["name"] == "synthetic-oracle"


def test_gate_qualification_is_opt_in_for_the_adapter_too(tmp_path, monkeypatch) -> None:
    """No GATE_QUALIFIED in the module means not gate-qualified, so a stub cannot become a gate
    input by being importable. The number is still recorded — PR-08 §6 asks for that."""
    install_adapter(monkeypatch, gate_qualified=None)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2",
                "--out", str(out)]) == mgt.EXIT_NOT_GATE_QUALIFIED
    rec = json.loads(out.read_text())
    assert rec["mask_method"]["gate_qualified"] is False
    assert rec["gate_qualified"] is False
    assert rec["GEOM_TOL_px"] == pytest.approx(2.0)
    assert any("not gate-qualified" in r for r in rec["gate_disqualified_reasons"])


def test_an_adapter_that_names_no_weights_cannot_be_gate_qualified(tmp_path, monkeypatch) -> None:
    """"sam2" without a checkpoint is a family of segmenters, not one: hiera-large and hiera-tiny
    are the same package and two different estimators. A tolerance that cannot say which weights
    produced it cannot be re-run with the same estimator at gate time."""
    install_adapter(monkeypatch, gate_qualified=True, checkpoints={})
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2",
                "--out", str(out)]) == mgt.EXIT_NOT_GATE_QUALIFIED
    rec = json.loads(out.read_text())["mask_method"]

    assert rec["gate_qualified"] is False
    assert rec["params"]["adapter_declares_gate_qualified"] is True
    assert "names no checkpoints" in rec["params"]["gate_qualification_withheld_reason"]
    assert "NONE DECLARED" in rec["provenance"]


def test_half_the_estimator_contract_is_refused(tmp_path, monkeypatch) -> None:
    """estimate_depth is unused by GEOM_TOL and still required: §4 step 2 wants ONE module behind
    both numbers, and a module that cannot measure the budget leaves this tolerance with no
    subtractable partner."""
    install_adapter(monkeypatch, with_depth=False)
    args = mgt._parse_args(["--corpus", str(tmp_path), "--method", "sam2"])
    with pytest.raises(mgt.MethodUnavailable, match="estimate_depth"):
        mgt.resolve_method(args)


def test_a_mask_on_the_wrong_grid_is_refused(tmp_path, monkeypatch) -> None:
    """The same refusal measure_est_drift.py makes: a centroid taken on one grid and reported in
    another is not a displacement."""
    install_adapter(monkeypatch, segment=lambda rgb: np.ones((7, 9), dtype=bool))
    method = mgt.resolve_method(mgt._parse_args(["--corpus", str(tmp_path), "--method", "sam2"]))
    with pytest.raises(mgt.MethodUnavailable, match="mask for a"):
        method.mask_fn(bgr_frames([(10, 10)])[0], method)


def test_a_video_method_with_no_segmenter_attached_never_falls_back(tmp_path) -> None:
    """The dispatch has no default. Papering this over with the red-pixel heuristic would produce a
    number under another method's name, which is the one failure the whole file exists to stop."""
    method = mgt.MaskMethod(name="broken", version="0", gate_qualified=True, frames_from="video")
    clip = tmp_path / "ep0.mp4"
    clip.write_bytes(b"")
    with pytest.raises(mgt.MethodUnavailable, match="carries no segmenter"):
        mgt.episode_centroids_from_video(clip, method, min_area=40, max_frames=0)


def test_the_heuristic_still_carries_its_own_segmenter(tmp_path) -> None:
    """The control for the dispatch change: hsv-red-diagnostic is untouched, still not a segmenter,
    still reachable only by typing its name."""
    method = mgt.hsv_red_method(40)
    assert method.mask_fn is mgt.hsv_red_mask
    assert method.gate_qualified is False
    assert method.name == "hsv-red-diagnostic"


# -- the two halves of §4 have to be talking about the same thing ----------------------------------


def _sam2_artifact(tmp_path, monkeypatch) -> Path:
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = tmp_path / "pr08_geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    return out


def test_the_two_rigs_join_on_mask_method_name(tmp_path, monkeypatch) -> None:
    """PR-08 §4 step 2: the SAME segmenter. The join key is ``mask_method.name``.

    ``measure_est_drift`` reads ESTIMATOR_NAME off the estimator module and records it as
    ``estimators.name``; this script reads it off the same module and records it as
    ``mask_method.name``. Equality has to hold by construction, because a mismatch is invisible:
    §6 subtracts two plausible pixel numbers and gets a plausible pixel number.
    """
    out = _sam2_artifact(tmp_path, monkeypatch)
    # cross_check_geom_tol() reports the artifact path relative to the repository root, so the
    # anchor moves with the artifact here. Nothing about the comparison itself is stubbed.
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", out)
    monkeypatch.setattr(ed, "_REPO_ROOT", out.parent)

    reasons, compare = ed.cross_check_geom_tol([CANVAS, CANVAS])
    theirs = ed.Estimators(sys.modules[SAM2_SPEC], SAM2_SPEC)

    assert reasons == [], reasons
    assert compare["geom_tol_mask_method"]["name"] == theirs.name
    assert compare["geom_tol_mask_method"]["version"] == theirs.version
    assert compare["geom_tol_gate_qualified"] is True


def test_the_grid_cross_check_is_not_a_no_op(tmp_path, monkeypatch) -> None:
    """cross_check_geom_tol() reads ``resolution_hw``. While this artifact did not write that key,
    the check found None, appended no reason, and reported a clean cross-check of two grids it had
    never compared — a no-op check being strictly worse than an absent one."""
    out = _sam2_artifact(tmp_path, monkeypatch)
    rec = json.loads(out.read_text())
    assert rec["resolution_hw"] == [CANVAS, CANVAS], "[height, width], the order the capture writes"
    assert rec["frame_width"] == CANVAS and rec["frame_height"] == CANVAS

    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", out)
    monkeypatch.setattr(ed, "_REPO_ROOT", out.parent)
    reasons, compare = ed.cross_check_geom_tol([64, 64])
    assert "resolution_disagrees_with_geom_tol" in reasons
    assert compare["geom_tol_resolution_hw"] == [CANVAS, CANVAS]


def test_the_artifact_states_the_join_key_for_whoever_reads_the_json(tmp_path, monkeypatch) -> None:
    """The consumer that has to check §4 step 2 reads the artifact, not this repository."""
    rec = json.loads(_sam2_artifact(tmp_path, monkeypatch).read_text())
    assert any("mask_method.name" in n and "estimators.name" in n for n in rec["notes"])
    assert "mask_method.name" in rec["mask_method"]["params"]["cross_check_join_key"]


# -- the adapter saying no is an answer, not a suggestion -------------------------------------------


def refusing_segment(_rgb):
    """What the real adapter's segment() does with no checkpoints staged: raises, on frame 0.

    ``EstimatorDependencyMissing`` subclasses ``ImportError`` (scripts/estimators/apple_sam2.py),
    which is deliberately NOT this module's ``MethodUnavailable``, so it is the shape that used to
    escape ``main``.
    """
    raise ImportError(
        "FATAL: facebook/sam2-hiera-large is not in any local hub cache and "
        "WAM_PR08_ALLOW_DOWNLOAD is unset."
    )


def test_method_sam2_refuses_before_decoding_when_the_adapter_says_the_weights_are_absent(
    tmp_path, monkeypatch, capsys
) -> None:
    """The adapter already answered. Typing the method out in full does not overrule the answer.

    Before this refusal, ``sam2_method()`` recorded ``weights_available: false`` in params and
    returned the method anyway; the decode loop then called ``segment()``, which raised an
    ImportError that ``main`` does not catch — traceback, exit 1, outside the documented EXIT STATUS
    table, no artifact and the decode already spent. ``segment`` here raises exactly that, and the
    assertion that matters is that it is never called at all.
    """
    calls: list[int] = []

    def counting_segment(rgb):
        calls.append(1)
        return refusing_segment(rgb)

    install_adapter(monkeypatch, available=False, segment=counting_segment)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists(), "a refusal must not leave an artifact behind"
    assert calls == [], "the adapter was asked to segment after it had said it cannot"

    err = capsys.readouterr().err
    assert "says its checkpoints are NOT on this machine" in err
    assert "available() returned False" in err
    assert "WAM_PR08_ALLOW_DOWNLOAD=1" in err
    assert "ASK" in err


def test_an_authorised_download_is_the_one_way_past_that_refusal(tmp_path, monkeypatch) -> None:
    """``available()`` stays False while the weights are absent even when a fetch IS permitted — the
    adapter says so in its own docstring — so gating on it alone would make
    ``--method sam2 WAM_PR08_ALLOW_DOWNLOAD=1``, the only honest way to say "fetch them, on
    purpose", unreachable. The permission is read off the adapter, never from this module's own
    environment, and it lands in the artifact because "the weights arrived during the measurement"
    is provenance."""
    mod = install_adapter(monkeypatch, available=False)
    mod.ALLOW_DOWNLOAD = True
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    params = json.loads(out.read_text())["mask_method"]["params"]
    assert params["adapter_download_authorised"] is True
    assert params["weights_available"] is False


def test_an_estimator_that_raises_mid_run_is_a_refusal_and_not_a_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    """Everything the adapter can raise mid-decode — a checkpoint found missing on frame 3, a CUDA
    RuntimeError — is an exception ``main`` does not catch, so all of it used to leave as a
    traceback and an exit 1 that is in no table. It is fatal, it is exit 2, nothing is written, and
    the adapter's own message survives verbatim because that message names its checkpoints."""
    def dies_on_the_third_frame(rgb):
        if len(seen) >= 2:
            raise RuntimeError("CUDA error: out of memory while running SAM 2")
        seen.append(1)
        return red_blob_segment(rgb)

    seen: list[int] = []
    install_adapter(monkeypatch, available=True, segment=dies_on_the_third_frame)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()

    err = capsys.readouterr().err
    assert "raised while segmenting a frame" in err
    assert "CUDA error: out of memory while running SAM 2" in err


def test_the_shape_refusal_is_not_swallowed_by_the_new_catch(tmp_path, monkeypatch) -> None:
    """The control for the try/except around the adapter call: this module's own refusals still
    travel as themselves rather than being re-wrapped as "the estimator raised"."""
    install_adapter(monkeypatch, segment=lambda rgb: np.ones((7, 9), dtype=bool))
    method = mgt.resolve_method(mgt._parse_args(["--corpus", str(tmp_path), "--method", "sam2"]))
    with pytest.raises(mgt.MethodUnavailable, match="mask for a"):
        method.mask_fn(bgr_frames([(10, 10)])[0], method)


def test_two_segmenters_on_one_command_line_are_refused(tmp_path, monkeypatch, capsys) -> None:
    """``--method sam2 --masks DIR`` names two segmenters and used to use one and silently discard
    the other's masks.meta.json provenance. The artifact's only job is to say which estimator
    produced GEOM_TOL, and a dropped --masks makes that record a guess."""
    install_adapter(monkeypatch, available=True)
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()
    assert "name two different segmenters" in capsys.readouterr().err

    # And the diagnostic heuristic is refused the same way, for the same reason.
    with pytest.raises(mgt.MethodUnavailable, match="two different segmenters"):
        mgt.resolve_method(mgt._parse_args(
            ["--corpus", str(corpus), "--method", "hsv-red-diagnostic", "--masks", str(masks)]
        ))


# -- [height, width], in that order, proved on a corpus where the two differ ------------------------

NONSQ_H, NONSQ_W = 96, 160


def nonsquare_bgr_frames(corners: list[tuple[int, int] | None]) -> list[np.ndarray]:
    """The same red blob on a 160x96 canvas — wider than it is tall, so H and W cannot be swapped
    without the artifact saying something false."""
    out = []
    for corner in corners:
        frame = np.zeros((NONSQ_H, NONSQ_W, 3), dtype=np.uint8)
        if corner is not None:
            x0, y0 = corner
            frame[y0:y0 + BLOB, x0:x0 + BLOB] = (0, 0, 255)
        out.append(frame)
    return out


def nonsquare_mask_corpus(tmp: Path) -> tuple[Path, Path]:
    """A precomputed-mask corpus on the same non-square grid: the other frames_from branch."""
    corpus = tmp / "corpus"
    masks = tmp / "masks"
    corpus.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    (masks / "masks.meta.json").write_text(json.dumps(
        {"method": "synthetic-oracle", "version": "1.0.0", "gate_qualified": True}))
    (corpus / "ep0.mp4").write_bytes(b"")
    ep = masks / "ep0"
    ep.mkdir(parents=True, exist_ok=True)
    for i, (x0, y0) in enumerate(walk((10, 10), (2, 0), 5)):
        arr = np.zeros((NONSQ_H, NONSQ_W), dtype=np.uint8)
        arr[y0:y0 + BLOB, x0:x0 + BLOB] = 1
        np.save(ep / f"{i:06d}.npy", arr)
    return corpus, masks


def test_resolution_hw_is_height_then_width_and_a_square_fixture_cannot_say_so(
    tmp_path, monkeypatch
) -> None:
    """``resolution_hw`` is the grid join key and its ORDER is the whole of its meaning.

    Every other fixture in this file is square, so both orders satisfy them and the fix that
    introduced this key was unprotected: transposing it left the suite green. On a 160x96 corpus a
    transposed key makes ``cross_check_geom_tol`` report ``resolution_disagrees_with_geom_tol``
    against a perfectly correct Isaac capture — a refusal earned by a bug in the artifact, which is
    the expensive direction because it looks like a finding.
    """
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": nonsquare_bgr_frames(walk((10, 10), (2, 0), 5))})

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert NONSQ_H != NONSQ_W, "this fixture is pointless if the canvas is square"
    assert rec["resolution_hw"] == [NONSQ_H, NONSQ_W]
    assert rec["resolution_hw"] != [NONSQ_W, NONSQ_H]
    assert rec["frame_width"] == NONSQ_W
    assert rec["frame_height"] == NONSQ_H
    assert rec["units"] == f"pixels at {NONSQ_W}x{NONSQ_H}"

    # And the key means what the consumer thinks it means: the capture's own [H, W] agrees, the
    # transpose does not.
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", out)
    monkeypatch.setattr(ed, "_REPO_ROOT", out.parent)
    assert ed.cross_check_geom_tol([NONSQ_H, NONSQ_W])[0] == []
    assert "resolution_disagrees_with_geom_tol" in ed.cross_check_geom_tol([NONSQ_W, NONSQ_H])[0]


def test_the_masks_path_writes_the_same_order(tmp_path) -> None:
    """The other branch: episode_centroids_from_masks returns (w, h) and the record must still be
    [h, w]. Two producers of the same key are two chances to transpose it."""
    corpus, masks = nonsquare_mask_corpus(tmp_path)
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks),
                "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["resolution_hw"] == [NONSQ_H, NONSQ_W]
    assert rec["frame_width"] == NONSQ_W and rec["frame_height"] == NONSQ_H


# -- the producer and the consumer, made to agree in something other than prose ---------------------


def _est_drift_fields_read_from_source() -> set[str]:
    """Every ``doc.get("...")`` literal inside ``measure_est_drift.cross_check_geom_tol``.

    Parsed out of that function's source rather than copied from it, so the day it grows a field
    this test fails here instead of the field being silently absent from every artifact this module
    writes.
    """
    import ast
    import inspect
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(ed.cross_check_geom_tol)))
    keys: set[str] = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def test_the_fields_the_consumer_reads_are_the_fields_this_module_declares() -> None:
    """The join is only checkable if both ends name the same fields, and prose in two files is not
    a check. ``frame_hw`` is the consumer's legacy fallback for ``resolution_hw``; this module
    writes the modern spelling, which is why it is read-but-not-guaranteed."""
    assert _est_drift_fields_read_from_source() == set(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert set(mgt.CROSS_CHECK_FIELDS_REQUIRED) <= set(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert "frame_hw" not in mgt.CROSS_CHECK_FIELDS_REQUIRED


def test_the_consumers_grid_check_is_absence_permissive_and_the_producer_closes_its_half(
    tmp_path, monkeypatch
) -> None:
    """``if theirs_hw is not None and ...``: an artifact with no ``resolution_hw`` passes the grid
    comparison by saying nothing, and downstream that is indistinguishable from a comparison that
    ran and agreed — the default-permissive shape this repo has already removed once. The reader is
    not this module's to fix. What is: never being the producer of such an artifact."""
    out = _sam2_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", out)
    monkeypatch.setattr(ed, "_REPO_ROOT", out.parent)

    # The consumer, demonstrated on a stripped artifact: two different grids, and it says nothing.
    stripped = json.loads(out.read_text())
    stripped.pop("resolution_hw")
    out.write_text(json.dumps(stripped))
    reasons, compare = ed.cross_check_geom_tol([7, 9])
    assert "resolution_disagrees_with_geom_tol" not in reasons
    assert compare["geom_tol_resolution_hw"] is None

    # The producer's half: that record would never have been written.
    assert mgt.missing_cross_check_fields(stripped) == ["resolution_hw"]
    assert mgt.missing_cross_check_fields(json.loads(_sam2_artifact(tmp_path, monkeypatch)
                                                    .read_text())) == []
    # A null resolution_hw is missing — null is exactly the value that makes the consumer's check
    # say nothing. A gate_qualified of False is NOT missing: it is a stated claim, and the consumer
    # does disqualify on it.
    assert mgt.missing_cross_check_fields({"resolution_hw": None, "gate_qualified": False}) == [
        "resolution_hw", "mask_method"
    ]


def test_a_record_missing_a_cross_check_field_is_never_written(tmp_path, monkeypatch) -> None:
    """The guard is wired into main, not merely defined: a record the consumer would read
    permissively is a fatal refusal with nothing on disk, not an artifact with a caveat."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    monkeypatch.setattr(mgt, "missing_cross_check_fields", lambda rec: ["resolution_hw"])

    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()
    assert not mgt.sidecar_path(out).exists()


def test_consumer_asserts_names_the_join_key_it_is_organised_around(tmp_path, monkeypatch) -> None:
    """The checklist a consumer follows literally must contain the invariant the whole change
    exists for. It did not: a consumer that satisfied every line of it could still subtract two
    numbers measured by two different segmenters and get a plausible pixel figure."""
    rec = json.loads(_sam2_artifact(tmp_path, monkeypatch).read_text())
    asserts = rec["consumer_asserts"]

    join = [a for a in asserts if "mask_method.name" in a and "estimators.name" in a]
    assert join, "the segmenter join key is absent from the artifact's own checklist"
    assert "NOTHING ENFORCES THIS" in join[0], "and it must say that nobody checks it for you"
    assert any("resolution_hw" in a for a in asserts)
    # Every field the consumer's cross-check reads is named somewhere in the checklist it follows.
    for field in mgt.CROSS_CHECK_FIELDS_REQUIRED:
        assert any(field in a for a in asserts), field


def test_the_artifact_records_where_the_consumer_is_weaker_than_it_reads(
    tmp_path, monkeypatch
) -> None:
    """Both limits are in the machine-readable artifact, not only in a docstring in this repo — the
    consumer reads the JSON."""
    rec = json.loads(_sam2_artifact(tmp_path, monkeypatch).read_text())
    limits = rec["cross_check_limits"]

    assert limits["fields_it_reads"] == list(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert limits["fields_this_artifact_guarantees"] == list(mgt.CROSS_CHECK_FIELDS_REQUIRED)
    assert "never asserts mask_method.name == estimators.name" in \
        limits["estimator_name_is_recorded_not_compared"]
    assert "absence" in limits["grid_comparison_is_absence_permissive"] or \
        "WITHOUT it" in limits["grid_comparison_is_absence_permissive"]
    assert "cross_check_geom_tol" in limits["checked_by"]


# -- the file's own claims about itself -------------------------------------------------------------


def test_the_refusal_derives_the_isaac_blocker_instead_of_asserting_it() -> None:
    """``no_segmenter_message()`` used to end with a hardcoded 'isaac_binding.py today wires only
    "rgb"', which commit 5ef3535 made false while the same file's ``_est_drift_blocker()`` was
    re-deriving the true version two hundred lines away. A refusal that prints a stale fact next to
    a computed one teaches its reader to trust neither."""
    msg = mgt.no_segmenter_message()
    assert 'today wires only "rgb"' not in msg
    assert mgt._est_drift_blocker() in msg
    assert "isaac_binding.py" in msg


def test_the_module_docstring_does_not_deny_the_segmenter_it_wires() -> None:
    """The docstring asserted, one paragraph before wiring one, that nothing in this repo, this
    virtualenv or this machine's local weights can find the apple. That absolute is false as of
    --method sam2, and it is the paragraph a reader hits first."""
    doc = mgt.__doc__ or ""
    assert "Nothing in this repo, this virtualenv or this machine's local weights can" not in doc
    assert mgt.SAM2_METHOD_CLI in doc


def test_the_adapter_still_declares_the_two_things_the_sam2_gate_reads() -> None:
    """The gate on ``--method sam2`` is only as real as the declarations it reads.

    ``available()`` says whether the checkpoints are here; ``ALLOW_DOWNLOAD`` says whether a human
    authorised fetching them, and it is the ONLY thing separating "absent, refuse" from "absent, a
    fetch was asked for". If either is renamed in the adapter, this module's refusal silently
    inverts — it would either refuse the authorised download path or stop refusing at all. Checked
    by parsing the adapter's source: importing it needs transformers, torch and 3 GB of weights, and
    these tests run with none of the three.
    """
    import ast

    tree = ast.parse(mgt.SAM2_ADAPTER_FILE.read_text(encoding="utf-8"))
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}

    assert mgt.ADAPTER_DOWNLOAD_ATTR in assigned, (
        f"{mgt.SAM2_ADAPTER_SPEC} no longer declares {mgt.ADAPTER_DOWNLOAD_ATTR}; "
        "--method sam2 would refuse the authorised-download path it is meant to allow through")
    assert "available" in functions
