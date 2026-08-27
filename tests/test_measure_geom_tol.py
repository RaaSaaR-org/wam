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
    """Run from an unrelated directory with no --out: the artifact lands at the anchor, not here.

    The anchor is seeded with the committed contract first, because that is what the real tracked
    path holds and because the run is refused otherwise — see
    ``test_the_tracked_path_may_not_be_measured_onto_when_the_contract_is_not_there``.
    """
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    anchor = tmp_path / "anchor" / "configs" / "transfer25" / "pr08_geom_tol.json"
    write_contract(anchor)
    monkeypatch.setattr(mgt, "DEFAULT_OUT", anchor)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run(["--corpus", str(corpus), "--method", "sam2"]) == mgt.EXIT_OK

    assert anchor.is_file(), "the artifact must land at the repo-anchored default"
    assert list(elsewhere.iterdir()) == [], (
        f"the caller's CWD must stay untouched, found {[p.name for p in elsewhere.iterdir()]}")
    assert json.loads(anchor.read_text())["artifact_path"] == str(anchor)


# -- the committed segmenter contract survives the measurement it constrains ----------------------
#
# The defect these cover, in one sentence: the pre-measurement contract lives at exactly
# DEFAULT_OUT_REL, so the first real GEOM_TOL run wrote its own schema over it, the contract was
# gone, and measure_est_drift refused every later run with geom_tol_does_not_record_segmenter_params
# — the gate was not wrong, it was unreachable, permanently.


def test_a_measurement_carries_the_committed_contract_forward_instead_of_erasing_it(
    tmp_path, monkeypatch
) -> None:
    """The whole point. Measure onto the path that holds the contract and the contract is still
    there afterwards, byte for byte, in the place the consumer looks — and the number is beside it
    rather than instead of it."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json")
    committed = json.loads(out.read_text())

    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    for key in mgt.CONTRACT_SECTION_FIELDS:
        assert rec[key] == committed[key], f"{key} was altered by the measurement"
    assert rec["GEOM_TOL_px"] is not None
    # Both places the reader looks, and they mean different things: what was committed, and what
    # ran. The guard has just proved them equal.
    assert mgt.committed_segmenter_contract(rec) == (committed["segmenter"], "segmenter")
    assert rec["mask_method"]["params"]["segmenter"] == committed["segmenter"]


def test_the_measurement_fills_the_contracts_null_slot_rather_than_leaving_two_spellings(
    tmp_path, monkeypatch
) -> None:
    """``run_g0_gates._first_present`` REFUSES a document stating one quantity under two spellings
    that disagree. A contract slot left null beside a measured ``GEOM_TOL_px`` is exactly that, so
    carrying the contract forward without filling ``geom_tol_px`` would have replaced one
    unreachable gate with another."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json")
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK

    rec = json.loads(out.read_text())
    assert rec["geom_tol_px"] == rec["GEOM_TOL_px"]
    assert rec["geom_tol_source"], "the slot must say who measured it, not merely hold a float"
    # EST_DRIFT_P95 is measured by the other script and this one must not appear to have.
    assert rec["est_drift_p95_px"] is None
    assert rec["gate_margin_px"] is None


def test_a_measurement_does_not_erase_an_est_drift_budget_already_in_the_file(
    tmp_path, monkeypatch
) -> None:
    """A GEOM_TOL run must not delete a measurement of something else.

    The record this module builds carries ``est_drift_p95_px: None`` hardcoded — this script does
    not measure that number — so copying the contract forward without care would null a budget
    somebody had already carried into the committed document, on a re-measure. ``gate_margin_px``
    is the opposite case: it is DERIVED, so carrying the old one forward would leave the artifact
    disagreeing with its own arithmetic, which run_g0_gates refuses outright."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json",
                         est_drift_p95_px=0.5, est_drift_source="pr08_est_drift.json",
                         est_drift_estimator_name="sam2-hiera-large+gdino-base",
                         gate_margin_px=99.0)

    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["est_drift_p95_px"] == 0.5, "a GEOM_TOL run must not erase EST_DRIFT_P95"
    assert rec["est_drift_source"] == "pr08_est_drift.json"
    assert rec[mgt.EST_DRIFT_NAME_FIELD] == "sam2-hiera-large+gdino-base", (
        "the join key travels with the number it names, or the number cannot be joined to this half"
    )
    assert rec["gate_margin_px"] == rec["geom_tol_px"] - 0.5, (
        "the margin is derived from THIS tolerance, not carried from the last one"
    )


def test_a_carried_budget_survives_a_document_that_never_declared_the_name_slot(
    tmp_path, monkeypatch
) -> None:
    """A document committed before ``est_drift_estimator_name`` existed declares the SHORTER
    measurement_fields list. Taking that list alone would drop the name from the artifact written
    over it — carrying the budget forward while losing the only evidence of which segmenter
    produced it, which is refused three lines later. The carry is the union of the two lists."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(
        tmp_path / "pr08_geom_tol.json",
        measurement_fields=["geom_tol_px", "geom_tol_source", "est_drift_p95_px",
                            "est_drift_source", "gate_margin_px"],
        est_drift_p95_px=0.5,
        est_drift_estimator_name="sam2-hiera-large+gdino-base",
    )
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec[mgt.EST_DRIFT_NAME_FIELD] == "sam2-hiera-large+gdino-base"


@pytest.mark.parametrize(
    "extra,needle",
    [
        ({"est_drift_p95_px": 0.5}, "and no est_drift_estimator_name"),
        (
            {"est_drift_p95_px": 0.5, "est_drift_estimator_name": "some-other-segmenter"},
            "would name two segmenters",
        ),
    ],
)
def test_a_budget_whose_segmenter_is_unnamed_or_different_writes_nothing(
    tmp_path, monkeypatch, extra, needle
) -> None:
    """PR-08 §6 subtracts EST_DRIFT_P95 from GEOM_TOL and §4 step 2 requires one segmenter behind
    both. A number in this document with no name beside it leaves run_g0_gates unable to check that
    for ever — a gate that cannot say yes — and a name that disagrees is two quantities being
    subtracted. Both stop the run with nothing written, where the fix is free."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json", **extra)
    before = out.read_bytes()

    code = run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)])
    assert code == mgt.EXIT_FATAL
    assert out.read_bytes() == before, "nothing may be written when the join key is not there"
    assert not mgt.sidecar_path(out).exists()


def test_a_run_whose_segmenter_disagrees_with_the_contract_writes_nothing_at_all(
    tmp_path, monkeypatch
) -> None:
    """The reason the contract is committed FIRST. A threshold moved after seeing the number is
    invisible in the result — the same adapter at two thresholds returns two plausible tolerances
    under one ESTIMATOR_NAME — so the disagreement has to stop the run, and it has to name the
    field."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json",
                         {**STUB_CONTRACT, "box_threshold": 0.35})
    before = out.read_bytes()

    rc = run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)])
    assert rc == mgt.EXIT_FATAL
    assert out.read_bytes() == before, "the committed contract must survive a refused run untouched"
    assert not mgt.sidecar_path(out).exists()


def test_a_disagreeing_run_names_the_field_and_both_values(tmp_path, monkeypatch, capsys) -> None:
    """"The segmenters disagree" is not an actionable message: a moved box_threshold and a moved
    checkpoint revision have different fixes."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = write_contract(tmp_path / "pr08_geom_tol.json",
                         {**STUB_CONTRACT, "box_threshold": 0.35})
    run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)])

    err = capsys.readouterr().err
    assert "box_threshold" in err
    assert "0.35" in err and "0.15" in err
    assert "before the measurement and never after it" in err


def test_a_method_that_cannot_state_its_segmenter_may_not_overwrite_a_contract(
    tmp_path, monkeypatch
) -> None:
    """``--method precomputed`` and ``hsv-red-diagnostic`` produce no segmenter contract, so a
    tolerance they wrote over the committed one would leave §4 step 2 permanently unanswerable —
    and the file would still look like a finished gate artifact."""
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    out = write_contract(tmp_path / "pr08_geom_tol.json")
    before = out.read_bytes()

    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(out)])
    assert rc == mgt.EXIT_FATAL
    assert out.read_bytes() == before


def test_a_scratch_out_that_holds_no_contract_is_overwritten_as_before(
    tmp_path, monkeypatch
) -> None:
    """The guard is about a file that made a pre-commitment, not about every existing file. A
    diagnostic re-run over yesterday's scratch artifact must still work, or the guard becomes a
    thing people route around with rm."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    out = tmp_path / "scratch.json"
    out.write_text(json.dumps({"schema": "something-else", "GEOM_TOL_px": 99.0}))
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    assert json.loads(out.read_text())["GEOM_TOL_px"] == 2.0
    # Nothing was carried forward, because there was no contract to carry.
    assert "segmenter" not in json.loads(out.read_text())


def test_the_tracked_path_may_not_be_measured_onto_when_the_contract_is_not_there(
    tmp_path, monkeypatch
) -> None:
    """The other half. A deleted or never-created contract would let a measurement write the
    tracked path with nothing to have been checked against, and the artifact is then
    indistinguishable from one that WAS checked. ``git checkout`` is cheaper than a re-run of a
    402-episode corpus."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 5))})
    anchor = tmp_path / "configs" / "transfer25" / "pr08_geom_tol.json"
    monkeypatch.setattr(mgt, "DEFAULT_OUT", anchor)

    assert run(["--corpus", str(corpus), "--method", "sam2"]) == mgt.EXIT_FATAL
    assert not anchor.exists()
    # ...and an --out elsewhere is untouched by that rule: it is scratch, and scratch is free.
    assert run(["--corpus", str(corpus), "--method", "sam2",
                "--out", str(tmp_path / "scratch.json")]) == mgt.EXIT_OK


def test_an_adapter_that_declares_no_segmenter_contract_is_not_gate_qualified(
    tmp_path, monkeypatch
) -> None:
    """Gate qualification is the claim that this number may set G0b's tolerance, and PR-08 §4 step 2
    is uncheckable against a module that never said what operating point it ran at. An uncheckable
    requirement reads downstream exactly like a satisfied one, which is why this is withheld here
    rather than left to the consumer."""
    install_adapter(monkeypatch, with_contract=False)
    method = mgt.resolve_method(mgt._parse_args(["--corpus", str(tmp_path), "--method", "sam2"]))
    assert method.gate_qualified is False
    assert "SEGMENTER_CONTRACT" in method.params["gate_qualification_withheld_reason"]
    assert method.params["segmenter"] is None


def test_the_merge_wears_the_same_contract_guard_as_the_measurement(tmp_path) -> None:
    """--merge writes the committed path too — that is what the merge is FOR — so a merged artifact
    that could erase the contract would reopen the hole on the second of the two write paths."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    shards = _sharded(tmp_path, corpus, masks, 3)
    out = write_contract(tmp_path / "pr08_geom_tol.json")
    before = out.read_bytes()

    # These shards were measured with --masks, which states no segmenter contract.
    assert run(["--merge", *shards, "--out", str(out)]) == mgt.EXIT_FATAL
    assert out.read_bytes() == before


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


#: The shape ``scripts/estimators/apple_sam2.SEGMENTER_CONTRACT`` has, small enough to read in a
#: failure message and complete enough that a field can be perturbed one at a time. The real one is
#: not imported: these tests must run with no transformers, no weights and no GPU, and a fixture
#: that drags the adapter in would be asserting that this machine can load SAM 2.
STUB_CONTRACT: dict = {
    "method_name": "sam2-hiera-large+gdino-base",
    "detector": {"repo": "IDEA-Research/grounding-dino-base", "revision": "12bdfa31"},
    "segmenter": {"repo": "facebook/sam2-hiera-large", "revision": "e6a8e880"},
    "depth": {"repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
              "revision": "d2fc6a93"},
    "object_text_prompt": "apple.",
    "box_threshold": 0.15,
    "text_threshold": 0.25,
    "box_selection": "highest_score",
    "propagation": "per_frame",
    "pixel_grid_hw": [480, 640],
}


def write_contract(path: Path, contract: dict | None = None, **extra) -> Path:
    """Put a pre-measurement contract document at ``path``, in the committed file's own shape.

    Mirrors ``configs/transfer25/pr08_geom_tol.json``: a contract section named by
    ``contract_fields``, a measurement section of nulls named by ``measurement_fields``. Written by
    hand rather than copied from the repo so a test can perturb one field and still be a valid
    document — and so these tests do not fail when the real contract is legitimately re-committed.
    """
    doc = {
        "spec_version": "1.0.0",
        "what_this_is": "test fixture standing in for the committed PR-08 contract",
        "contract_fields": list(mgt.CONTRACT_SECTION_FIELDS),
        "measurement_fields": list(mgt.CONTRACT_MEASUREMENT_FIELDS),
        "segmenter": dict(contract or STUB_CONTRACT),
        **{k: None for k in mgt.CONTRACT_MEASUREMENT_FIELDS},
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


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
    contract: dict | None = None,
    with_contract: bool = True,
) -> types.ModuleType:
    """Install a stub of ``estimators.apple_sam2`` satisfying the estimator contract.

    ``None`` for ``gate_qualified`` or ``available`` means the attribute is ABSENT, which is a
    different statement from False and is tested separately: absent is what a module written without
    thinking about the gate looks like. ``with_contract=False`` is the same distinction for
    ``SEGMENTER_CONTRACT``: a module that never declared its operating point, which cannot be
    gate-qualified because PR-08 §4 step 2 would be uncheckable against it.
    """
    mod = types.ModuleType(SAM2_SPEC)
    mod.ESTIMATOR_NAME = name
    mod.ESTIMATOR_VERSION = version
    mod.segment = segment
    if with_contract:
        mod.SEGMENTER_CONTRACT = dict(contract or {**STUB_CONTRACT, "method_name": name})
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


def install_video_frames(monkeypatch, frames: dict[str, list[np.ndarray]], module=None) -> None:
    """Replace decoding, and ONLY decoding.

    The method's own segmenter, the centroid arithmetic, the largest-component rule and the
    drop-and-count of invisible objects all stay real. The single thing a machine with no codecs and
    no corpus cannot do is turn an .mp4 into pixels, so that is the single thing stubbed.

    ``resolve_decoder`` is stubbed alongside it because it is now part of decoding: it PROBES the
    real file before choosing, and these fixtures are empty .mp4 stubs by design. Leaving it real
    here would mean every test in this file asserts, first and mostly, that the machine running it
    has a codec — which is a fact about the machine and not about this script.
    """

    def fake(clip, method, min_area, max_frames, decoder=None):
        stack = frames[clip.stem]
        if max_frames > 0:
            stack = stack[:max_frames]
        cents = [
            (module or mgt).centroid_of_mask(method.mask_fn(f, method), largest_component=True,
                                             min_area=min_area)
            for f in stack
        ]
        h, w = stack[0].shape[:2]
        return cents, (int(w), int(h)), 30.0

    # ``module`` exists for the before/after test below, which drives a SECOND copy of this script
    # (the one at the previous commit) over the same fixture and has to stub that copy's own
    # attributes rather than this one's.
    target = module or mgt
    monkeypatch.setattr(target, "episode_centroids_from_video", fake)
    monkeypatch.setattr(target, "resolve_decoder", lambda name, probe_clip: target.Decoder(
        name="stub", version="0",
        open_fn=lambda clip: (iter(frames[clip.stem]), 30.0),
        note=f"test stub; --decoder was {name!r}",
    ))


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
        mgt.episode_centroids_from_video(clip, method, min_area=40, max_frames=0,
                                         decoder=mgt.DECODERS["cv2"])


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
    """Every ``.get("...")`` literal ``measure_est_drift.cross_check_geom_tol`` reaches.

    Parsed out of source rather than copied from it, so the day the reader grows a field this test
    fails here instead of the field being silently absent from every artifact this module writes.

    IT FOLLOWS THE READER INTO ITS HELPERS, and that is not thoroughness — it is the repair of a
    real hole. The walk used to cover ``cross_check_geom_tol``'s own body only, so when the
    committed-contract lookup moved into ``committed_segmenter_contract()`` the two names it reads
    (``segmenter``, ``params``) vanished from this check AND from
    ``CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT``, and the artifact went on publishing
    ``cross_check_limits.fields_it_reads`` as a list that understated what the reader read. A guard
    that a refactor can disarm is not a guard; this one resolves every call to a module-level
    function of ``measure_est_drift`` and walks that too, transitively.
    """
    import ast
    import inspect
    import textwrap
    import types

    keys: set[str] = set()
    seen: set[str] = set()
    queue = [ed.cross_check_geom_tol]
    while queue:
        fn = queue.pop()
        if fn.__name__ in seen:
            continue
        seen.add(fn.__name__)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
            # A bare name that resolves to a function on the reader's module is a place the read
            # could have moved to. Imported names resolve too: `committed_segmenter_contract` is
            # measure_geom_tol's, reached through measure_est_drift's namespace, which is exactly
            # the refactor that hid it.
            if isinstance(node.func, ast.Name):
                target = getattr(ed, node.func.id, None)
                if isinstance(target, types.FunctionType):
                    queue.append(target)
    return keys


def test_the_guard_over_the_consumers_reads_follows_it_into_its_helpers() -> None:
    """The guard above is only worth having if a read that MOVES cannot escape it.

    ``committed_segmenter_contract`` is where the segmenter lookup lives, it is imported into
    ``measure_est_drift`` from ``measure_geom_tol``, and its two literals must be found. If this
    fails, the parser has stopped following the reader and the declaration test below is passing
    vacuously."""
    reached = _est_drift_fields_read_from_source()
    assert {"segmenter", "params"} <= reached, (
        "the walk no longer reaches committed_segmenter_contract(); the field declaration is "
        "unguarded again"
    )


def test_the_fields_the_consumer_reads_are_the_fields_this_module_declares() -> None:
    """The join is only checkable if both ends name the same fields, and prose in two files is not
    a check. ``frame_hw`` is the consumer's legacy fallback for ``resolution_hw``; this module
    writes the modern spelling, which is why it is read-but-not-guaranteed."""
    assert _est_drift_fields_read_from_source() == set(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert set(mgt.CROSS_CHECK_FIELDS_REQUIRED) <= set(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert "frame_hw" not in mgt.CROSS_CHECK_FIELDS_REQUIRED


def test_a_stripped_artifact_is_disqualified_by_name_and_would_never_have_been_written(
    tmp_path, monkeypatch
) -> None:
    """An artifact with no ``resolution_hw`` used to pass the consumer's grid comparison BY SAYING
    NOTHING (``if theirs_hw is not None and ...``), which downstream is indistinguishable from a
    comparison that ran and agreed. The reader was repaired on 2026-08-22: each field it needs and
    does not find is its own ``geom_tol_does_not_record_<field>``. This asserts BOTH halves — the
    reader refusing, and this module never being the producer of such an artifact — because either
    one alone is a check that can be removed without a test noticing."""
    out = _sam2_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", out)
    monkeypatch.setattr(ed, "_REPO_ROOT", out.parent)

    stripped = json.loads(out.read_text())
    stripped.pop("resolution_hw")
    # The contract carried into the artifact states the grid too, so strip that as well: this test
    # is about a document that says NOTHING about its grid, not about the fallback path.
    stripped.pop("segmenter", None)
    stripped["mask_method"]["params"].pop("segmenter", None)
    out.write_text(json.dumps(stripped))
    reasons, compare = ed.cross_check_geom_tol([7, 9])
    assert "geom_tol_does_not_record_resolution_hw" in reasons, (
        "silence about the grid must disqualify, not pass"
    )
    assert "resolution_disagrees_with_geom_tol" not in reasons, (
        "and it must not be reported as a comparison that ran"
    )
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
    # The entry must tell a consumer WHAT IN THIS DOCUMENT to check the name against. Until
    # 2026-08-22 it named only the other artifact and the measurement-time enforcement inside
    # measure_est_drift, so a consumer holding a finished tolerance had nothing to compare and
    # run_g0_gates could only answer "could not check" — for ever, on every run.
    assert mgt.EST_DRIFT_NAME_FIELD in join[0], (
        "the checklist must name the field this document carries the other half's segmenter in"
    )
    assert "measurement-time" in join[0] or "measurement time" in join[0], (
        "and it must still say which half of the check runs only while the measurement is running"
    )
    assert "NOTHING ENFORCES THIS" not in join[0], (
        "that claim was falsified when cross_check_geom_tol gained the name comparison"
    )
    assert any("segmenter" in a and "revision" in a for a in asserts), (
        "a name is not a segmenter: the checklist must name the block, not only the string"
    )
    assert any("resolution_hw" in a for a in asserts)
    # Every field the consumer's cross-check reads is named somewhere in the checklist it follows.
    for field in mgt.CROSS_CHECK_FIELDS_REQUIRED:
        assert any(field in a for a in asserts), field


def test_the_artifact_records_what_the_consumer_checks_and_does_not_still_claim_the_old_limits(
    tmp_path, monkeypatch
) -> None:
    """``cross_check_limits`` is the one field whose job is to tell a later reader what was and was
    not compared, so a falsified statement in it is worse than an absent one.

    Two of its keys asserted, into every artifact this module wrote, that the consumer never
    compared the estimator name and that its grid check passed on absence. Both were closed on
    2026-08-22, hours before this artifact's shape was last edited. This asserts the replacements
    say the true thing AND that the falsified keys are gone — a stale limit is a reader re-checking
    by hand something the machine already checks, and a reader who finds one wrong stops trusting
    the rest of the block."""
    rec = json.loads(_sam2_artifact(tmp_path, monkeypatch).read_text())
    limits = rec["cross_check_limits"]

    assert limits["fields_it_reads"] == list(mgt.CROSS_CHECK_FIELDS_READ_BY_EST_DRIFT)
    assert limits["fields_this_artifact_guarantees"] == list(mgt.CROSS_CHECK_FIELDS_REQUIRED)
    assert "cross_check_geom_tol" in limits["checked_by"]
    assert "estimator_name_is_recorded_not_compared" not in limits
    assert "grid_comparison_is_absence_permissive" not in limits
    assert "mask_method_disagrees_with_estimator" in limits["what_the_reader_now_enforces"]
    assert "segmenter_params_disagree_with_geom_tol" in limits["what_the_reader_now_enforces"]
    # The limit that is REAL and replaces them: none of it runs for a consumer holding two
    # finished artifacts.
    assert "checked by nobody" in limits["it_only_runs_at_measurement_time"]
    assert "REFUSES" in limits["the_committed_contract_is_write_protected_here"]


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


# -- the decoder is a choice, and it is now made against the corpus rather than assumed -----------
#
# Job 189585, the first cluster run of this script, decoded ZERO frames of the PR-08 corpus: it is
# AV1, and the cv2 in the generator's own venv has no AV1 decoder. VideoCapture did not raise. It
# opened the file, reported 590 frames off the container header, and failed every read. That is the
# shape that already cost this project job 186357 (372 clips captioned, 0 captions written), and
# these tests exist so the third time is caught here instead of on a GPU.


def _dead_decoder(name: str = "dead") -> "mgt.Decoder":
    """Opens fine, decodes nothing. cv2-on-AV1, reduced to its essentials."""
    return mgt.Decoder(name=name, version="0", open_fn=lambda clip: (iter(()), 30.0),
                       note="decodes nothing")


def _live_decoder(name: str = "live", frames: int = 3) -> "mgt.Decoder":
    stack = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(frames)]
    return mgt.Decoder(name=name, version="1", open_fn=lambda clip: (iter(stack), 30.0),
                       note="decodes")


def test_a_named_decoder_that_decodes_nothing_is_refused_rather_than_used(tmp_path, monkeypatch) -> None:
    """The failure is silent at the decoder, so it has to be loud here.

    An empty iterator is indistinguishable from a short clip unless something asks for one frame and
    checks. Nothing downstream can tell them apart afterwards: zero frames yields zero steps yields
    coverage 0.0, which reads as a fact about the corpus rather than about the reader.
    """
    monkeypatch.setitem(mgt.DECODERS, "cv2", _dead_decoder("cv2"))
    clip = tmp_path / "ep0.mp4"
    clip.write_bytes(b"")
    with pytest.raises(mgt.MethodUnavailable) as exc:
        mgt.resolve_decoder("cv2", clip)
    assert "decoded no frames" in str(exc.value)
    assert "ep0.mp4" in str(exc.value)


def test_auto_probes_and_takes_the_first_decoder_that_returns_a_frame(tmp_path, monkeypatch) -> None:
    """'auto' is not 'whatever imports'. An importable decoder is not evidence it can read AV1."""
    monkeypatch.setattr(mgt, "DECODERS", {"cv2": _dead_decoder("cv2"), "pyav": _live_decoder("pyav")})
    clip = tmp_path / "ep0.mp4"
    clip.write_bytes(b"")
    chosen = mgt.resolve_decoder("auto", clip)
    assert chosen.name == "pyav"
    # The one that failed is named in the note, not quietly dropped: the artifact has to be able to
    # say the corpus was unreadable by the decoder a reader would assume was used.
    assert "cv2" in chosen.note and "decoded no frames" in chosen.note


def test_auto_refuses_when_nothing_can_read_the_corpus(tmp_path, monkeypatch) -> None:
    """No decoder is not "use the first one anyway"; it is nothing was measured."""
    monkeypatch.setattr(mgt, "DECODERS", {"cv2": _dead_decoder("cv2"), "pyav": _dead_decoder("pyav")})
    clip = tmp_path / "ep0.mp4"
    clip.write_bytes(b"")
    with pytest.raises(mgt.MethodUnavailable) as exc:
        mgt.resolve_decoder("auto", clip)
    assert "not a pass" in str(exc.value)
    assert "cv2" in str(exc.value) and "pyav" in str(exc.value)


def test_the_artifact_records_which_decoder_read_the_pixels(tmp_path, monkeypatch) -> None:
    """Provenance, beside mask_method and for the same reason: two readers of the same bytes are not
    obviously producing the same quantity, and the artifact is the only place that can say which."""
    install_adapter(monkeypatch)
    corpus, _ = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 6)})
    install_video_frames(monkeypatch, {"ep0": bgr_frames(walk((10, 10), (2, 0), 6))})
    out = tmp_path / "geom_tol.json"
    assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["decoder"]["name"] == "stub"
    assert rec["decoder"]["selected"] == "auto"


def test_the_pyav_reader_yields_bgr_because_everything_downstream_assumes_cv2s_order(tmp_path) -> None:
    """A real clip, encoded and read back, checked on a colour that BGR and RGB disagree about.

    sam2_mask_via flips once with frame[:, :, ::-1] on the way to the adapter's segment(rgb). A
    decoder handing back RGB would not crash — GroundingDINO would ground "apple" on a frame where
    red is blue, and GEOM_TOL would become the median displacement of whatever that found. The
    channel order is therefore a property each decoder owes, not a detail.
    """
    av = pytest.importorskip("av")
    clip = tmp_path / "red.mp4"
    # Pure red in RGB. In BGR that is (0, 0, 255); in RGB it is (255, 0, 0) — the two orders cannot
    # both be right and a grey fixture could not tell.
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    with av.open(str(clip), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for _ in range(3):
            container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
        container.mux(stream.encode())

    frames, fps = mgt.DECODERS["pyav"].open_fn(clip)
    first = np.asarray(next(iter(frames)))
    assert first.dtype == np.uint8 and first.shape == (16, 16, 3)
    b, g, r = (int(v) for v in first[8, 8])
    # Lossy codec: assert the ordering, not the exact values.
    assert r > 200 and b < 60 and g < 60, f"expected BGR of pure red, got (b,g,r)=({b},{g},{r})"
    assert fps == pytest.approx(30.0)


# -- sharding, and the merge that has to put the median back together ------------------------------
#
# The pilot (cluster job 189588) measured the full GEOM_TOL run at 4.005 GPU-h against a 4 h MaxWall
# on every Discoverer+ QoS, so the committed number cannot be produced by one job. It is produced by
# N and merged, and the merge is where the interesting failures live:
#
#   the median is         GEOM_TOL is "the median per-step object-centroid displacement in the
#   averaged instead      source clips" (PR-08 §6). A median does NOT decompose. The median of the
#   of pooled             shard medians has the right units and a plausible magnitude and is a
#                         different number, and nothing downstream re-derives GEOM_TOL, so the error
#                         is permanent and invisible. The fixture below is built so the two ANSWERS
#                         DIFFER — a merge that averaged would fail loudly rather than by luck.
#
#   a shard goes missing  Seven of eight array tasks land, the eighth hits the wall. A merge over
#   and the merge         seven shards is a median over 7/8 of the corpus wearing the name of the
#   proceeds              whole. Every refusal here is fatal with nothing written.
#
#   a shard artifact is   A shard is a partial measurement. If one can be committed as GEOM_TOL then
#   committed as the      the gate is set from a twelfth of the corpus. Three independent statements
#   number                stop it: a different schema, GEOM_TOL_px null, is_shard true.
#
#   the partition moves   An --episode-range is an index into the episode list, so one added clip
#   under the chain       renumbers everything after it — and shards are computed by different jobs
#                         at different times. Assignment is a digest of the episode KEY instead, and
#                         PYTHONHASHSEED makes hash() the wrong digest.


def _sharded(tmp_path: Path, corpus: Path, masks: Path, n: int,
             extra: list[str] | None = None) -> list[str]:
    """Run all ``n`` shards over the same corpus and return their artifact paths."""
    paths = []
    for i in range(n):
        p = tmp_path / f"shard-{i}.json"
        rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(p),
                  "--shard", str(i), "--num-shards", str(n), *(extra or [])])
        assert rc == mgt.EXIT_OK, f"shard {i} exited {rc}"
        paths.append(str(p))
    return paths


def _eleven_episode_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Eleven episodes of differing length and speed, chosen so the shards are UNBALANCED.

    Under the key digest at N=3 this splits 6/2/3, which is the realistic case and is also the case
    where averaging the shard medians is most obviously wrong: the pooled median is 3.1623 px and
    the median of the three shard medians is 2.7361 px. A fixture that split evenly and moved at one
    speed would let a merge that averaged pass.
    """
    return make_corpus(tmp_path, {
        f"ep{i:02d}": walk((10, 10 + 3 * i), (1 + (i % 5), i % 3), 6 + (i % 4))
        for i in range(11)
    })


def test_a_shard_that_measured_less_than_it_was_assigned_stops_the_merge(tmp_path: Path) -> None:
    """THE COVERAGE PROOF WAS WEAKER THAN THE ARTIFACT IT WROTE.

    The union was taken over each shard's ASSIGNED ``episode_keys`` while the pooled displacements
    come from ``per_episode``, so an episode assigned and never measured was in the first list and
    absent from the second, nothing fired, and ``merged_from.refusals_checked`` recorded "the union
    of covered episodes is not the corpus, exactly once each" as a check that had been performed.
    A later reader trusting that line would believe measurement coverage was proved when only
    assignment coverage was — on the one number PR-08 §6 defines over the whole source corpus and
    that nobody re-derives.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    shards = _sharded(tmp_path, corpus, masks, 3)

    victim = Path(shards[0])
    rec = json.loads(victim.read_text())
    dropped = rec["per_episode"].pop()["episode"]
    victim.write_text(json.dumps(rec))

    out = tmp_path / "merged.json"
    assert run(["--merge", *shards, "--out", str(out)]) == mgt.EXIT_FATAL
    assert not out.exists()
    assert dropped  # the episode that vanished is named in the refusal, checked below


def test_that_refusal_names_the_unaccounted_episode_and_the_arithmetic(
    tmp_path: Path, capsys
) -> None:
    """A merge refusal has to say which episode and how the counts fail to add up, because the fix
    is to re-run one shard and the operator has to know which."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    shards = _sharded(tmp_path, corpus, masks, 3)
    victim = Path(shards[0])
    rec = json.loads(victim.read_text())
    dropped = rec["per_episode"].pop()["episode"]
    victim.write_text(json.dumps(rec))

    run(["--merge", *shards, "--out", str(tmp_path / "merged.json")])
    err = capsys.readouterr().err
    assert dropped in err
    assert "UNACCOUNTED FOR" in err
    assert "measured" in err and "skipped" in err


def test_an_episode_a_shard_was_not_assigned_cannot_be_pooled_by_it(tmp_path: Path) -> None:
    """The count identity alone is satisfiable by a shard that measured somebody else's episode and
    skipped one of its own — a double-count in the median and a hole in the corpus at once."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    shards = _sharded(tmp_path, corpus, masks, 3)

    a, b = Path(shards[0]), Path(shards[1])
    rec_a, rec_b = json.loads(a.read_text()), json.loads(b.read_text())
    rec_a["per_episode"].append(dict(rec_b["per_episode"][0]))
    rec_a["n_episodes_skipped_no_masks"] = 1
    a.write_text(json.dumps(rec_a))

    assert run(["--merge", *shards, "--out", str(tmp_path / "merged.json")]) == mgt.EXIT_FATAL


def test_an_episode_a_shard_legitimately_skipped_still_merges_and_is_counted(
    tmp_path: Path
) -> None:
    """PARITY WITH THE UN-SHARDED RUN, which is the property the whole merge design rests on.

    An episode with no mask directory is skipped and named on both paths, so the merge must accept
    it — refusing here would make a merged run stricter than the single-job run it is supposed to be
    identical to. What it must NOT do is call that corpus coverage: ``coverage_proof`` records the
    measured count and the skipped count separately, so the artifact states what was proved."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    import shutil

    victim = sorted(p for p in masks.iterdir() if p.is_dir())[0]
    shutil.rmtree(victim)

    shards = _sharded(tmp_path, corpus, masks, 3)
    out = tmp_path / "merged.json"
    assert run(["--merge", *shards, "--out", str(out)]) in (mgt.EXIT_OK,
                                                            mgt.EXIT_NOT_GATE_QUALIFIED)
    proof = json.loads(out.read_text())["merged_from"]["coverage_proof"]
    assert proof["corpus_episodes"] == 11
    assert proof["assigned_episodes"] == 11
    assert proof["measured_episodes"] == 10
    assert proof["skipped_no_masks"] == 1


def test_the_merge_refuses_two_shards_that_read_two_cameras_or_two_corpora(
    tmp_path: Path
) -> None:
    """``corpus_episode_keys_sha256`` digests episode KEYS, so two roots holding the same episode
    names agree on it while holding different pixels — and ``camera_key`` selects which pixels were
    measured at all. Both are carried into the committed artifact from shard 0, so a disagreement
    means the artifact names one corpus and one camera while half its displacements came from
    somewhere else."""
    corpus, masks = _eleven_episode_corpus(tmp_path)

    for field, value in (("camera_key", "wrist"), ("corpus", "/somewhere/else")):
        shards = _sharded(tmp_path, corpus, masks, 3)
        victim = Path(shards[0])
        rec = json.loads(victim.read_text())
        rec[field] = value
        victim.write_text(json.dumps(rec))
        assert run(["--merge", *shards, "--out", str(tmp_path / f"{field}.json")]) == \
            mgt.EXIT_FATAL, field


def test_the_shards_cover_every_episode_exactly_once(tmp_path: Path) -> None:
    """The property the whole merge rests on. Not "the counts add up" — WHICH episodes.

    Eight shards reporting 50 episodes each sum to 400 whether they covered 400 distinct episodes or
    380 with 20 counted twice, so the artifact records the keys and this asserts on the keys.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)

    seen: list[str] = []
    for p in paths:
        rec = json.loads(Path(p).read_text())
        seen += rec["shard"]["episode_keys"]
        assert rec["shard"]["n_episodes_in_shard"] == len(rec["shard"]["episode_keys"])
        assert rec["n_episodes"] == len(rec["shard"]["episode_keys"])

    expected = json.loads(Path(paths[0]).read_text())["corpus_episode_keys"]
    assert len(expected) == 11
    assert sorted(seen) == sorted(expected), "the union must be the corpus"
    assert len(seen) == len(set(seen)), "no episode may be measured twice"
    # And unbalanced, on purpose: a fixture that happened to split evenly would not exercise the
    # case the assignment rule was chosen for.
    sizes = sorted(len(json.loads(Path(p).read_text())["shard"]["episode_keys"]) for p in paths)
    assert sizes[0] != sizes[-1], "this fixture is meant to be unbalanced"


def test_every_episode_index_is_its_place_in_the_full_enumeration(tmp_path: Path) -> None:
    """The merge's sort key. A serial number within the shard would rebuild the pool in the wrong
    order, which the median would survive and mean_px, std_px and the histogram would not."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)

    got: dict[int, str] = {}
    for p in paths:
        for ep in json.loads(Path(p).read_text())["per_episode"]:
            got[ep["episode_index"]] = ep["episode"]
    assert sorted(got) == list(range(11))
    assert [got[i] for i in range(11)] == [f"ep{i:02d}" for i in range(11)]


def test_adding_an_episode_moves_that_episode_and_no_other(tmp_path: Path) -> None:
    """Why the assignment is a digest of the KEY and not a slice of the LIST.

    A shard chain is resumable: shard 3 may be computed on Tuesday and shard 7 re-run on Wednesday
    after a preemption. With an --episode-range, one clip added in between renumbers every episode
    after it, and the resulting artifacts overlap on some episodes and skip others while each one
    stays internally consistent. Here, the new episode moves and nothing else does.
    """
    before = {k: mgt.shard_of(k, 8) for k in (f"ep{i:02d}" for i in range(11))}
    after = {k: mgt.shard_of(k, 8) for k in
             list(f"ep{i:02d}" for i in range(11)) + ["ep05b"]}
    assert all(after[k] == before[k] for k in before), "no existing episode may move"
    assert "ep05b" in after


def test_the_partition_does_not_depend_on_the_interpreters_hash_seed(tmp_path: Path) -> None:
    """``hash(key) % N`` is the obvious spelling and would silently break every array job.

    ``PYTHONHASHSEED`` is randomised per interpreter, so each Slurm array task would compute a
    different partition of the same corpus — producing shards that cover some episodes twice and
    others never, each internally consistent and each individually well-formed. This runs the
    assignment under two fixed, different seeds and requires the same answer.
    """
    import os
    import subprocess

    prog = (
        "import sys; sys.path.insert(0, %r); import measure_geom_tol as m;"
        "print(','.join(str(m.shard_of('ep%%02d' %% i, 8)) for i in range(11)))"
        % str(_REPO_ROOT / "scripts")
    )
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", prog], env=env, check=True,
                                   capture_output=True, text=True).stdout.strip())
    assert outs[0] == outs[1], f"the partition moved with PYTHONHASHSEED: {outs}"
    assert outs[0] == ",".join(str(mgt.shard_of(f"ep{i:02d}", 8)) for i in range(11))


# -- THE important one: the merged number is the un-sharded number, exactly ------------------------


#: Fields that CANNOT match between a merged artifact and an un-sharded one, because they describe
#: how the artifact was produced rather than what was measured. Everything else must be equal, and
#: the list is short on purpose: an exact whole-record comparison is a far easier property to defend
#: than "equal in the fields we thought to check".
_PROVENANCE_ONLY = {
    "measured_by",            # "... --merge" vs "..."
    "measured_utc",           # a timestamp
    "artifact_path",          # different --out
    "artifact_sha256_sidecar",
    "merged_from",            # exists only on the merged side
}


def test_the_merged_artifact_is_the_unsharded_artifact_exactly(tmp_path: Path) -> None:
    """GEOM_TOL merged from three shards equals GEOM_TOL measured in one pass. Exactly — ``==``.

    Not ``approx``. A merge that pooled correctly but rebuilt the array in shard order would pass an
    approximate check on the median (order-invariant) while quietly changing ``mean_px``, ``std_px``
    and the histogram counts, which are floating-point sums and are not. So the assertion is on the
    WHOLE record minus five provenance fields, and the exactness is not a coincidence: shards emit
    raw float64 displacements, ``float -> JSON -> float`` is the identity (``json.dumps`` renders a
    float with ``repr``, the shortest round-tripping string since Python 3.1), and the merge
    rebuilds the pool in the corpus's own enumeration order via ``episode_index``.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    full = tmp_path / "full.json"
    assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(full)]) == mgt.EXIT_OK
    reference = json.loads(full.read_text())

    merged_path = tmp_path / "merged.json"
    assert run(["--merge", *_sharded(tmp_path, corpus, masks, 3),
                "--out", str(merged_path)]) == mgt.EXIT_OK
    merged = json.loads(merged_path.read_text())

    assert merged["GEOM_TOL_px"] == reference["GEOM_TOL_px"], "the headline, bit for bit"
    differing = sorted(k for k in set(reference) | set(merged)
                       if k not in _PROVENANCE_ONLY and reference.get(k) != merged.get(k))
    assert differing == [], f"merged and un-sharded disagree on {differing}"
    # Named individually as well, so a future change that shrinks the record cannot make the
    # comparison above vacuous for the fields that matter most.
    for key in ("distribution", "per_episode", "coverage", "n_steps_measured", "n_steps_total",
                "n_steps_dropped_object_not_visible", "n_frames", "n_episodes",
                "n_episodes_found", "resolution_hw", "mask_method", "step_frames",
                "gate_qualified", "headline_valid", "schema"):
        assert merged[key] == reference[key], key
    assert merged["schema"] == mgt.SCHEMA


def test_the_median_of_the_shard_medians_is_not_the_answer(tmp_path: Path) -> None:
    """The failure the whole design exists against, made visible on this fixture.

    If the merge averaged (or medianed) the shard medians it would report 2.7361 px where the
    corpus's median is 3.1623 px — a 13 % error in a gate tolerance, in the right units, from an
    artifact that looks finished. This asserts the two are DIFFERENT here, so the equality test
    above cannot be passing by accident on a fixture where every route gives the same answer.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    shard_medians = [json.loads(Path(p).read_text())["shard_median_px"] for p in paths]

    merged_path = tmp_path / "merged.json"
    assert run(["--merge", *paths, "--out", str(merged_path)]) == mgt.EXIT_OK
    pooled = json.loads(merged_path.read_text())["GEOM_TOL_px"]

    assert float(np.median(shard_medians)) != pytest.approx(pooled, rel=1e-3), (
        "this fixture is supposed to separate the two statistics", shard_medians, pooled)
    assert float(np.mean(shard_medians)) != pytest.approx(pooled, rel=1e-3)


#: Fields of a merged artifact that record WHEN and WHERE this particular merge ran, rather than
#: what it measured. Excluded from the two determinism comparisons below, and from nothing else.
_MERGE_RUN_LOCAL = {"measured_utc", "artifact_path", "artifact_sha256_sidecar"}


def test_the_merged_artifact_does_not_depend_on_the_order_the_shards_were_merged_in(
        tmp_path: Path) -> None:
    """Merging the same shards forwards, backwards and shuffled gives ONE artifact.

    This is not a style point. ``--merge`` is pointed at a DIRECTORY by the sbatch, and the order a
    directory scan yields is the filesystem's business — ``shard-10.json`` sorts before
    ``shard-2.json``, and a re-run that rewrites one file can move it. If the record depended on
    that order then GEOM_TOL, or the histogram counts beside it, would depend on which machine
    globbed the directory, and the artifact's sha256 sidecar would stop being reproducible from its
    inputs. The mechanism that makes it not depend on it is that every list in the merged record is
    rebuilt in a key the shards carry — ``episode_index`` for the pool, ``shard.index`` for the
    provenance — and never in the order the arguments arrived.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path / "p", corpus, masks, 3)

    def merged(order: list[str], tag: str) -> dict:
        out = tmp_path / tag / "merged.json"
        assert run(["--merge", *order, "--out", str(out)]) == mgt.EXIT_OK
        return json.loads(out.read_text())

    forward = merged(list(paths), "fwd")
    for tag, order in (("rev", list(reversed(paths))), ("mix", [paths[1], paths[2], paths[0]])):
        other = merged(order, tag)
        differing = sorted(k for k in set(forward) | set(other)
                           if k not in _MERGE_RUN_LOCAL and forward.get(k) != other.get(k))
        assert differing == [], f"merge order {tag} changed {differing}"
    # Including the provenance block, which is the one that names the shards and could plausibly
    # have been built in argument order.
    assert [s["index"] for s in forward["merged_from"]["shards"]] == [0, 1, 2]

    # AND ON A FIXTURE WHERE THE ORDER COULD SHOW. Everything above is measured from precomputed
    # masks, where every field the merge templates forward is identical across the shards by the
    # time the refusals have run — so the comparison passes even if the template were taken from
    # "whichever shard was listed first" rather than from shard 0. `decoder.note` is the ONE
    # templated field the merge deliberately allows to differ between shards (each probes its own
    # first clip), so it is the only lever that can tell the two rules apart. Give the shards
    # different notes and the choice becomes observable.
    def with_notes(i, rec):
        rec["decoder"] = {"name": "pyav", "version": "16.0.1", "selected": "auto",
                          "note": f"Selected by --decoder auto after probing ep{i:02d}.mp4"}

    noted = _shards_then(tmp_path / "noted", 3, with_notes)
    first = merged(list(noted), "n-fwd")
    last = merged(list(reversed(noted)), "n-rev")
    assert first["decoder"]["note"] == last["decoder"]["note"], (
        "the templated decoder block must come from shard 0, not from whichever shard the "
        "directory scan happened to yield first")
    assert "ep00.mp4" in first["decoder"]["note"]


def test_two_different_partitions_of_one_corpus_merge_to_the_same_artifact(tmp_path: Path) -> None:
    """N=2 and N=5 over ONE corpus produce one GEOM_TOL, one distribution, one per_episode.

    The number PR-08 §6 defines is a property of the corpus, so how many jobs happened to measure it
    must not be visible in it. It is worth pinning separately from the un-sharded comparison because
    the two failures are different: that test would catch a merge that pooled in shard order at a
    FIXED N, and this one catches a merge that pooled correctly but let N leak into a count, a
    coverage figure or a histogram — which is what "re-run it at N=16 because N=8 hit the wall"
    would silently do to a committed tolerance.

    ``merged_from`` is the one field that differs, and differing is its job: it names the shards, so
    it is the record OF the partition rather than a number contaminated by it.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)

    def merged(n: int) -> dict:
        out = tmp_path / f"n{n}" / "merged.json"
        assert run(["--merge", *_sharded(tmp_path / f"p{n}", corpus, masks, n),
                    "--out", str(out)]) == mgt.EXIT_OK
        return json.loads(out.read_text())

    two, five = merged(2), merged(5)
    differing = sorted(k for k in set(two) | set(five)
                       if k not in _MERGE_RUN_LOCAL and two.get(k) != five.get(k))
    assert differing == ["merged_from"], f"the partition leaked into {differing}"
    assert two["GEOM_TOL_px"] == five["GEOM_TOL_px"], "the headline, bit for bit, across partitions"
    assert two["merged_from"]["num_shards"] == 2 and five["merged_from"]["num_shards"] == 5


def test_the_decoders_probe_note_is_the_one_field_a_partition_shows_through(
        tmp_path: Path) -> None:
    """A KNOWN, DELIBERATE EXCEPTION, pinned here so it is documented rather than discovered.

    Every fixture above reads precomputed masks, so ``decoder`` is null and the two determinism
    tests never see it. The real run decodes video, and ``--decoder auto`` writes into
    ``decoder.note`` WHICH CLIP it probed — each shard probes its own first clip. The merge takes
    the whole ``decoder`` block from shard 0's template (it compares only name and version across
    shards, deliberately: comparing the note would refuse every correct sharded run), so the merged
    artifact's note names shard 0's first clip and not the corpus's.

    That makes ``decoder.note`` partition-dependent, and it means the merged artifact is NOT quite
    byte-identical to an un-sharded run on the video path. It is prose provenance that no consumer
    reads, and every shard's own note is preserved under ``merged_from.shards[].decoder_note``, so
    nothing is lost — but the claim "identical to what a single un-sharded run would have written"
    is exact only up to this field, and a test that says so is cheaper than a reader discovering it
    in a diff of two committed artifacts.
    """
    def notes_from(n: int, first_clip) -> dict:
        def mutate(i, rec):
            rec["decoder"] = {"name": "pyav", "version": "16.0.1", "selected": "auto",
                              "note": f"Selected by --decoder auto after probing {first_clip(i)}"}
        out = tmp_path / f"m{n}" / "merged.json"
        assert run(["--merge", *_shards_then(tmp_path, n, mutate),
                    "--out", str(out)]) == mgt.EXIT_OK
        return json.loads(out.read_text())

    three = notes_from(3, lambda i: f"ep{i:02d}.mp4")
    # The SAME corpus re-sharded; only shard 0's probe clip moves.
    two = notes_from(2, lambda i: f"ep{i + 5:02d}.mp4")

    assert three["GEOM_TOL_px"] == two["GEOM_TOL_px"], "the number is not affected"
    assert three["distribution"] == two["distribution"]
    assert three["decoder"]["name"] == two["decoder"]["name"] == "pyav"
    assert three["decoder"]["note"] != two["decoder"]["note"], (
        "if this ever becomes equal the exception has been closed and this test should be deleted")
    # Nothing is lost: each partition still names every clip its own shards probed.
    assert len({s["decoder_note"] for s in three["merged_from"]["shards"]}) == 3


def test_json_carries_a_float64_displacement_back_unchanged(tmp_path: Path) -> None:
    """The lossless claim the merge's exactness rests on, asserted rather than assumed.

    If this ever stopped holding, every merged GEOM_TOL would be off by an unstated amount and the
    module docstring's "there is no error bound to prove because there is no error" would be a lie.
    """
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.random(512) * 1e3,
        np.asarray([0.0, np.nextafter(0.0, 1.0), 1 / 3, np.hypot(3.0, 4.0), 1e-300, 1e300]),
    ])
    back = np.asarray(json.loads(json.dumps([float(v) for v in values])), dtype=float)
    assert np.array_equal(back, values), "float -> JSON -> float must be the identity"


# -- a shard artifact must never be mistakable for the committed number -----------------------------


def test_a_shard_artifact_cannot_be_read_as_geom_tol(tmp_path: Path) -> None:
    """Three independent statements, because a single flag is one oversight away from being ignored."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    rec = json.loads(Path(_sharded(tmp_path, corpus, masks, 3)[0]).read_text())

    assert rec["schema"] == mgt.SHARD_SCHEMA != mgt.SCHEMA
    assert rec["GEOM_TOL_px"] is None
    assert rec["is_shard"] is True
    assert rec["shard_median_px"] is not None, "the diagnostic is recorded, just not called GEOM_TOL"
    assert "median does not decompose" in rec["geom_tol_px_is_null_because"]
    # gate_qualified on a shard is "fit to be merged", and the artifact says so in words rather
    # than leaving a reader to infer it from a flag that reads like the gate's.
    assert rec["gate_qualified"] is True
    assert "FIT TO BE MERGED" in rec["gate_qualified_scope"]


def test_a_shard_refuses_to_write_the_committed_artifact_path(tmp_path: Path, capsys) -> None:
    """N array tasks writing one path is a race whose winner is whichever task finished last, and
    what it leaves behind is one shard of the corpus sitting where the gate reads GEOM_TOL."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--shard", "0", "--num-shards", "3",
              "--out", str(mgt.DEFAULT_OUT)])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "refuses to write the tracked default" in err
    assert "race" in err


def test_shard_and_num_shards_go_together(tmp_path: Path, capsys) -> None:
    corpus, masks = _eleven_episode_corpus(tmp_path)
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(tmp_path / "s.json"),
              "--shard", "0"])
    assert rc == mgt.EXIT_FATAL
    assert "go together" in capsys.readouterr().err


def test_a_shard_index_outside_the_partition_is_refused(tmp_path: Path, capsys) -> None:
    """--shard 3 --num-shards 3 measures the empty set and would report it as a clean run."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(tmp_path / "s.json"),
              "--shard", "3", "--num-shards", "3"])
    assert rc == mgt.EXIT_FATAL
    assert "out of range" in capsys.readouterr().err


def test_merging_and_sharding_on_one_command_line_are_refused(tmp_path: Path, capsys) -> None:
    rc = run(["--merge", str(tmp_path / "a.json"), "--shard", "0", "--num-shards", "2",
              "--out", str(tmp_path / "m.json")])
    assert rc == mgt.EXIT_FATAL
    assert "two different jobs on one command line" in capsys.readouterr().err


def test_corpus_is_still_required_when_not_merging(tmp_path: Path, capsys) -> None:
    """--corpus stopped being argparse-required so the merge job can run with no data mounted.
    Every other invocation must still refuse without it, with a reason and not a usage block."""
    rc = run(["--out", str(tmp_path / "g.json")])
    assert rc == mgt.EXIT_FATAL
    assert "--corpus is required" in capsys.readouterr().err


# -- the merge's refusals. Each one its own message, each one fatal with nothing written ------------


def _shards_then(tmp_path: Path, n: int, mutate) -> list[str]:
    """Build ``n`` shards over the standard fixture, then let ``mutate(index, record)`` edit them.

    Hand-editing the JSON is not a contrivance: a stale copy in a scan directory, a shard written by
    an older version of the script, and a job that was pointed at the wrong corpus all reach the
    merge as exactly this — a well-formed artifact that disagrees with its siblings.
    """
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, n)
    for i, p in enumerate(paths):
        rec = json.loads(Path(p).read_text())
        mutate(i, rec)
        Path(p).write_text(json.dumps(rec, indent=2) + "\n")
    return paths


def test_a_missing_shard_is_refused_rather_than_merged(tmp_path: Path, capsys) -> None:
    """Seven of eight array tasks land and the eighth hits the wall. The merge must not proceed."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    out = tmp_path / "merged.json"

    assert run(["--merge", paths[0], paths[2], "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "shard(s) 1 of 3 are missing" in err
    assert "plausible magnitude" in err
    assert not out.exists(), "a refused merge writes nothing"


def test_two_artifacts_claiming_the_same_shard_are_refused(tmp_path: Path, capsys) -> None:
    """A stale copy left in the scan directory would weight its episodes twice in the median."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    stale = tmp_path / "shard-0-rerun.json"
    stale.write_text(Path(paths[0]).read_text())

    rc = run(["--merge", *paths, str(stale), "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "both claim shard index 0" in capsys.readouterr().err


def test_shards_from_two_different_partitions_are_refused(tmp_path: Path, capsys) -> None:
    def mutate(i, rec):
        if i == 1:
            rec["shard"]["num_shards"] = 4
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "disagree on num_shards" in err
    assert "two DIFFERENT partitions" in err


def test_shards_that_did_not_enumerate_the_same_corpus_are_refused(tmp_path: Path, capsys) -> None:
    def mutate(i, rec):
        if i == 2:
            rec["shard"]["corpus_episode_keys_sha256"] = "0" * 64
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "the corpus they enumerated" in capsys.readouterr().err


def test_shards_that_disagree_on_the_mask_method_are_refused(tmp_path: Path, capsys) -> None:
    """PR-08 §4 step 2 requires ONE segmenter behind GEOM_TOL, because §6 subtracts EST_DRIFT_P95
    from it and that subtraction is arithmetic only between two numbers from the same estimator."""
    def mutate(i, rec):
        if i == 1:
            rec["mask_method"]["version"] = "9.9.9"
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "disagree on the mask method" in err
    assert "PR-08 §4 step 2" in err


def test_shards_that_disagree_on_the_decoder_are_refused(tmp_path: Path, capsys) -> None:
    """Two readers of the same AV1 bytes are not obviously the same quantity — job 189585 read 590
    frames off a container header and decoded none of them."""
    def mutate(i, rec):
        rec["decoder"] = {"name": "pyav" if i else "cv2", "version": "1.0", "selected": "auto",
                          "note": f"probed clip {i}"}
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "disagree on the decoder" in capsys.readouterr().err


def test_shards_whose_decoders_only_probed_different_clips_still_merge(tmp_path: Path) -> None:
    """The other half of that refusal, and the reason it compares name and version only.

    ``--decoder auto`` records WHICH clip it probed in ``decoder.note``, and each shard probes its
    own first clip. Comparing the whole block would refuse every correct sharded run there will
    ever be — a refusal that fires on the normal case is not a safeguard, it is an outage.
    """
    def mutate(i, rec):
        rec["decoder"] = {"name": "pyav", "version": "16.0.1", "selected": "auto",
                          "note": f"Selected by --decoder auto after probing ep{i:02d}.mp4"}
    paths = _shards_then(tmp_path, 3, mutate)
    out = tmp_path / "merged.json"
    assert run(["--merge", *paths, "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())
    assert rec["decoder"]["name"] == "pyav"
    # ... and every shard's note is kept, so which clip each one probed is still recoverable.
    notes = [s["decoder_note"] for s in rec["merged_from"]["shards"]]
    assert len(set(notes)) == 3


def test_shards_that_disagree_on_the_pixel_grid_are_refused(tmp_path: Path, capsys) -> None:
    def mutate(i, rec):
        if i == 0:
            rec["resolution_hw"] = [480, 640]
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "the pixel grid" in err
    assert "two units" in err


def test_shards_that_disagree_on_the_step_are_refused(tmp_path: Path, capsys) -> None:
    """GEOM_TOL scales ~linearly with what a step is taken to be, and PR-08 §6 never defines one."""
    def mutate(i, rec):
        if i == 2:
            rec["step_frames"] = 2
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "step_frames" in capsys.readouterr().err


def test_shards_that_disagree_on_the_coverage_floor_are_refused(tmp_path: Path, capsys) -> None:
    def mutate(i, rec):
        if i == 1:
            rec["min_coverage"] = 0.5
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "min_coverage" in capsys.readouterr().err


def test_a_gate_qualification_split_between_shards_is_refused(tmp_path: Path, capsys) -> None:
    """One shard says it is fit to be merged and another says it is not. The pooled artifact would
    be neither, and its gate flag would be a claim about only some of its own inputs."""
    def mutate(i, rec):
        if i == 1:
            rec["gate_qualified"] = False
            rec["gate_disqualified_reasons"] = ["coverage 0.400 < --min-coverage 0.9"]
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "disagree on gate_qualified" in err
    assert "coverage 0.400" in err, "the refusal must say WHY the odd shard was disqualified"


def test_shards_that_all_say_they_are_unfit_merge_into_a_disqualified_artifact(
    tmp_path: Path,
) -> None:
    """Not a disagreement, so not a refusal — PR-08 §6 records GEOM_TOL regardless of verdict. The
    artifact is written, stamped, and exits 3 so no chain treats it as the committed number."""
    def mutate(i, rec):
        rec["gate_qualified"] = False
        rec["gate_disqualified_reasons"] = ["mask method 'synthetic-oracle' is not gate-qualified"]
        rec["mask_method"]["gate_qualified"] = False
    paths = _shards_then(tmp_path, 3, mutate)
    out = tmp_path / "merged.json"
    assert run(["--merge", *paths, "--out", str(out)]) == mgt.EXIT_NOT_GATE_QUALIFIED
    rec = json.loads(out.read_text())
    assert rec["gate_qualified"] is False
    assert rec["GEOM_TOL_px"] is not None
    assert any("shard 0:" in r for r in rec["gate_disqualified_reasons"])


def test_an_episode_in_a_shard_it_does_not_hash_to_is_refused(tmp_path: Path, capsys) -> None:
    """The merge re-derives the assignment rule rather than trusting the artifact. This catches an
    artifact written under a different partition rule — including the PYTHONHASHSEED class of bug
    the rule exists to make impossible."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    victim = Path(paths[0])
    rec = json.loads(victim.read_text())
    stolen = json.loads(Path(paths[1]).read_text())["shard"]["episode_keys"][0]
    rec["shard"]["episode_keys"].append(stolen)
    victim.write_text(json.dumps(rec, indent=2) + "\n")

    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "do not hash to it" in err
    assert stolen in err


def test_a_union_that_is_not_the_whole_corpus_is_refused(tmp_path: Path, capsys) -> None:
    """The shards are all present, all agree, and together they have a hole in them. A merge that
    cannot prove it saw every episode is not a merge."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    victim = Path(paths[2])
    rec = json.loads(victim.read_text())
    dropped = rec["shard"]["episode_keys"].pop()
    rec["shard"]["n_episodes_in_shard"] = len(rec["shard"]["episode_keys"])
    rec["per_episode"] = [e for e in rec["per_episode"] if e["episode"] != dropped]
    victim.write_text(json.dumps(rec, indent=2) + "\n")

    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "do not cover the corpus exactly once" in err
    assert "NEVER MEASURED (1)" in err
    assert dropped in err


def test_a_shard_that_reports_only_its_median_cannot_be_merged(tmp_path: Path, capsys) -> None:
    """The refusal that stands between the merge and averaging medians. A shard that dropped its raw
    displacements would leave the merge nothing to pool, and there is no correct fallback."""
    def mutate(i, rec):
        if i == 1:
            for ep in rec["per_episode"]:
                ep.pop("displacements_px")
    paths = _shards_then(tmp_path, 3, mutate)
    rc = run(["--merge", *paths, "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "a median does not" in err
    assert "averaging medians is the wrong number" in err


def test_a_finished_geom_tol_is_not_an_input_to_a_merge(tmp_path: Path, capsys) -> None:
    """Merging the committed artifact back in would pool the corpus with itself."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    full = tmp_path / "full.json"
    run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(full)])
    rc = run(["--merge", str(full), "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "not an input to a merge" in capsys.readouterr().err


def test_a_shard_named_explicitly_and_absent_is_a_missing_shard_not_a_filter(
    tmp_path: Path, capsys
) -> None:
    rc = run(["--merge", str(tmp_path / "nope.json"), "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "does not exist" in capsys.readouterr().err


def test_a_merge_over_no_shards_is_a_missing_input_and_not_an_empty_result(
    tmp_path: Path, capsys
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = run(["--merge", str(empty), "--out", str(tmp_path / "merged.json")])
    assert rc == mgt.EXIT_FATAL
    assert "found no shard artifacts at all" in capsys.readouterr().err


def test_a_directory_of_shards_merges_and_skips_what_is_not_a_shard(tmp_path: Path, capsys) -> None:
    """The shape the sbatch's MERGE step uses: point it at the run directory. That directory also
    holds the pilot artifact and, after the first merge, the merged one — a scan that refused on
    those would be unusable, so they are skipped with a line saying so."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(3):
        p = shard_dir / f"shard-{i}.json"
        assert run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(p),
                    "--shard", str(i), "--num-shards", "3"]) == mgt.EXIT_OK
    (shard_dir / "GEOM_TOL_PILOT.json").write_text(
        json.dumps({"schema": "wam.pr08_geom_tol_pilot/1"}))

    out = tmp_path / "merged.json"
    assert run(["--merge", str(shard_dir), "--out", str(out)]) == mgt.EXIT_OK
    err = capsys.readouterr().err
    assert "skipping" in err and "GEOM_TOL_PILOT.json" in err
    assert json.loads(out.read_text())["n_episodes"] == 11


# -- what the merged artifact has to carry for the consumers downstream -----------------------------


def test_the_merged_artifact_carries_what_the_gate_consumers_read(tmp_path: Path) -> None:
    """97_transfer25_restyle.sbatch and measure_est_drift.cross_check_geom_tol() read this file and
    nothing else. A merged artifact missing one of their fields passes their checks by saying
    nothing, which is worse than failing them."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    out = tmp_path / "merged.json"
    assert run(["--merge", *_sharded(tmp_path, corpus, masks, 3), "--out", str(out)]) == mgt.EXIT_OK
    rec = json.loads(out.read_text())

    assert mgt.missing_cross_check_fields(rec) == []
    for key in ("mask_method", "decoder", "resolution_hw", "coverage", "min_coverage",
                "GEOM_TOL_px", "gate_qualified", "step_frames", "consumer_asserts",
                "cross_check_limits", "notes", "est_drift_p95_blocked_by"):
        assert key in rec, key
    assert rec["mask_method"]["params"], "params is what makes the estimator re-runnable"
    assert rec["GEOM_TOL_px"] is not None
    # A sidecar, exactly as the un-sharded path writes: the merged file IS the pre-commitment.
    side = mgt.sidecar_path(out)
    assert side.read_text().strip() == hashlib.sha256(out.read_bytes()).hexdigest()
    # And none of the shard-only fields survived into it.
    for key in mgt.SHARD_ONLY_FIELDS:
        assert key not in rec, f"{key} is a shard field and must not reach the committed artifact"
    assert "displacements_px" not in rec["per_episode"][0]


def test_the_merged_artifact_names_the_shards_it_was_built_from(tmp_path: Path) -> None:
    """Traceability, in the shape AC-04 asks for everywhere else: which files, and their digests."""
    corpus, masks = _eleven_episode_corpus(tmp_path)
    paths = _sharded(tmp_path, corpus, masks, 3)
    out = tmp_path / "merged.json"
    run(["--merge", *paths, "--out", str(out)])
    merged = json.loads(out.read_text())["merged_from"]

    assert merged["num_shards"] == 3
    assert [s["index"] for s in merged["shards"]] == [0, 1, 2]
    for s in merged["shards"]:
        assert s["sha256"] == hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest()
    assert sum(s["n_episodes"] for s in merged["shards"]) == 11
    assert "does NOT decompose" in merged["pooling"]


def test_an_empty_shard_names_the_partition_and_not_the_corpus(tmp_path: Path, capsys) -> None:
    """--num-shards near the episode count leaves shards empty by chance, and an empty shard cannot
    be written honestly: it decodes no frame, so it has no pixel grid, and an artifact with a null
    ``resolution_hw`` is one the consumer's cross-check passes by comparing nothing. The refusal has
    to point at the partition — left to fall through, the run dies on "no episode yielded any
    frames", which is a sentence about the clips and sends the operator to the wrong place.
    """
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    empty = next(i for i in range(64) if mgt.shard_of("ep0", 64) != i)
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(tmp_path / "s.json"),
              "--shard", str(empty), "--num-shards", "64"])
    assert rc == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "holds no episodes" in err
    assert "Lower --num-shards" in err
    assert not (tmp_path / "s.json").exists()


# -- --carry-est-drift: the merge that used to be a person with a text editor ----------------------
#
# PR-08 §8 item 4 wants two numbers committed, and they are measured by two scripts into two files.
# Somebody merges them into the one document the gate reads, and the thing that merge can drop is
# the SEGMENTER NAME — at which point run_g0_gates can never establish §4 step 2's "the same
# segmenter" and no G0b run can return 0. These pin that the merge is code, and that every way it
# can produce a document nobody may gate on stops it with nothing written.


def _est_artifact(path: Path, **over) -> Path:
    doc = {
        "schema": "wam.est_drift/1",
        "gate_qualified": True,
        "est_drift_p95_px": 0.5,
        "is_lower_bound": True,
        "measured_utc": "2026-08-22T00:00:00+00:00",
        "resolution_hw": [480, 640],
        "estimators": {"name": "sam2-hiera-large+gdino-base"},
        "geom_tol_cross_check": {"this_segmenter_contract": dict(STUB_CONTRACT)},
        **over,
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def test_carrying_a_budget_writes_the_number_the_name_and_the_margin(tmp_path: Path) -> None:
    """The number alone is not carryable: PR-08 §6 subtracts it and §4 step 2 asks which segmenter
    produced it, so the name travels with it mechanically instead of being retyped."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"},
                         est_drift_p95_blocked_by="steps 1-4 have not been run here")
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["est_drift_p95_px"] == 0.5
    assert doc[mgt.EST_DRIFT_NAME_FIELD] == "sam2-hiera-large+gdino-base"
    assert "pr08_est_drift.json" in doc["est_drift_source"]
    assert "sha256=" in doc["est_drift_source"], "AC-04: which bytes the number came out of"
    assert doc["gate_margin_px"] == pytest.approx(3.0)
    # The document must not go on explaining an absence that is no longer there.
    assert doc["est_drift_p95_blocked_by"] is None
    # The contract section is untouched: this mode carries a measurement, it does not re-commit.
    assert doc["segmenter"] == STUB_CONTRACT
    assert mgt.sidecar_path(out).read_text().strip() == hashlib.sha256(
        out.read_bytes()).hexdigest()


def test_carrying_a_budget_declares_the_name_slot_for_the_next_measurement(tmp_path: Path) -> None:
    """A later measure run copies the measurement slots the document DECLARES. A carry into a
    document whose list predates the name slot must add it, or the next measurement drops the name
    and is then refused for having dropped it."""
    out = write_contract(
        tmp_path / "pr08_geom_tol.json",
        geom_tol_px=3.5, GEOM_TOL_px=3.5, gate_qualified=True,
        measurement_fields=["geom_tol_px", "geom_tol_source", "est_drift_p95_px",
                            "est_drift_source", "gate_margin_px"],
        mask_method={"name": "sam2-hiera-large+gdino-base"},
    )
    est = _est_artifact(tmp_path / "pr08_est_drift.json")
    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_OK
    assert mgt.EST_DRIFT_NAME_FIELD in json.loads(out.read_text())["measurement_fields"]


def test_carrying_a_budget_into_a_document_with_no_tolerance_writes_nothing_at_all(
    tmp_path: Path, capsys
) -> None:
    """THE DEFECT, ON THE NARROW PATH NOTHING ELSE COVERED.

    Until 2026-08-27 this case WROTE and then refused: `tol` was read four lines above
    write_artifact(), so a gate-qualified EST_DRIFT artifact carried onto the pristine committed
    contract — geom_tol_px null, which is exactly how PR-08 §4 step 2 requires that file to sit
    until the corpus is measured — landed est_drift_p95_px, est_drift_estimator_name,
    est_drift_source AND a fresh .sha256 sidecar in the TRACKED document, printed "gate_margin_px
    is null" and exited. Three of four measured slots filled, a digest certifying the half-written
    state, and no margin: a document that reads as a carry somebody performed rather than as one
    that was refused, with the sidecar as the part that makes it look checked.

    The sidecar is asserted separately and not as an afterthought. A .sha256 beside a document that
    did not change is its own corruption — the next reader checks the digest, finds it matches, and
    concludes the file was written by this tool.
    """
    out = write_contract(tmp_path / "pr08_geom_tol.json", gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    assert "states no GEOM_TOL" in capsys.readouterr().err
    assert out.read_bytes() == before, "the half-written gate document is the defect"
    assert not mgt.sidecar_path(out).exists(), "a digest certifying a document that did not change"


def test_a_disqualified_est_drift_artifact_is_refused_before_the_target_is_even_read(
    tmp_path: Path, capsys
) -> None:
    """The regression guard on the half that was ALREADY right, pinned at the one ordering that
    matters. A disqualified EST_DRIFT artifact must stop the carry on its own reasons — not on the
    target document's — so that an operator holding a bad measurement is told about the measurement
    rather than sent to re-measure GEOM_TOL. Here the target is the pristine contract, which the
    2026-08-27 refusals would also reject; the artifact's verdict has to be the one that speaks."""
    out = write_contract(tmp_path / "pr08_geom_tol.json",
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _est_artifact(tmp_path / "pr08_est_drift.json", gate_qualified=False,
                        gate_disqualified_reasons=["estimator_not_gate_qualified"])

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "estimator_not_gate_qualified" in err
    assert "states no GEOM_TOL" not in err, "the artifact's own verdict is the reason to report"
    assert out.read_bytes() == before
    assert not mgt.sidecar_path(out).exists()


@pytest.mark.parametrize("over", [{"gate_qualified": False}, {}])
def test_carrying_onto_a_geom_tol_document_that_is_not_itself_qualified_refuses(
    tmp_path: Path, over: dict, capsys
) -> None:
    """THE HALF THIS MODE NEVER CHECKED. est_drift_measurement refuses a disqualified EST_DRIFT
    artifact, so an exit 0 from this command certified one side of GEOM_TOL - EST_DRIFT_P95 and
    asserted nothing whatever about the side the document is named after. A merge that came back
    gate_qualified: false — shards disagreeing about the pixel grid, a partial pool, an adapter
    whose flag was down when the array ran — still carries a real-looking geom_tol_px, and the
    carry would subtract from it, print a positive margin and exit 0. Both halves then sit in one
    file under one gate_margin_px that a reader cannot attribute.

    ABSENT REFUSES EXACTLY AS FALSE DOES, which is the second parameter. The committed contract
    carries no gate_qualified because it is committed BEFORE the measurement; a document that has
    not been measured is not a qualified one, and saying nothing has never been saying yes here."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"}, **over)
    before = out.read_bytes()
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "gate_qualified" in err, "the refusal must name the reason, not merely refuse"
    assert out.read_bytes() == before
    assert not mgt.sidecar_path(out).exists()


def test_a_qualified_pair_carries_end_to_end_with_the_arm_named(tmp_path: Path) -> None:
    """THE HAPPY PATH, ASSERTED AS ARITHMETIC. A gate-qualified EST_DRIFT artifact, a gate-qualified
    GEOM_TOL document with a real tolerance, and an arm stated on the command line: all four
    measured slots land together, the sidecar matches the bytes on disk, and gate_margin_px is the
    subtraction PR-08 §6 asks for rather than merely a non-null number. Pinned as the arithmetic
    because a margin field that is populated but wrong is the failure this whole file is built
    against — two plausible pixel numbers subtracting to a plausible pixel number."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json", per_frame=0.5, propagation=0.9)

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "per_frame"]) == mgt.EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["est_drift_p95_px"] == 0.5
    assert doc[mgt.EST_DRIFT_NAME_FIELD] == "sam2-hiera-large+gdino-base"
    assert "arm='per_frame'" in doc["est_drift_source"]
    assert doc["gate_margin_px"] == pytest.approx(3.5 - 0.5)
    assert doc["gate_margin_px"] == pytest.approx(doc["geom_tol_px"] - doc["est_drift_p95_px"])
    assert mgt.sidecar_path(out).read_text().strip() == hashlib.sha256(
        out.read_bytes()).hexdigest()


def test_a_non_positive_margin_is_carried_and_reported_rather_than_widened(tmp_path: Path) -> None:
    """PR-08 §6: a margin <= 0 is the FINDING. The number is written — that is the record — and the
    status refuses to call it a gate. The move is a better estimator, never a wider tolerance."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=0.4, GEOM_TOL_px=0.4,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _est_artifact(tmp_path / "pr08_est_drift.json")
    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_NOT_GATE_QUALIFIED
    assert json.loads(out.read_text())["gate_margin_px"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"gate_qualified": False, "gate_disqualified_reasons": ["capture_is_not_from_isaac_sim"]},
         "capture_is_not_from_isaac_sim"),
        ({"est_drift_p95_px": None}, "There is no number to carry"),
        ({"estimators": {}}, "no estimators.name"),
        ({"schema": "something/else"}, "does not carry schema"),
        ({"resolution_hw": [256, 256]}, "two pixel grids"),
        ({"geom_tol_cross_check": {"this_segmenter_contract":
                                   dict(STUB_CONTRACT, box_threshold=0.35)}},
         "box_threshold"),
        ({"estimators": {"name": "some-other-segmenter"}}, "would name two segmenters"),
    ],
)
def test_a_budget_that_cannot_be_joined_to_this_document_is_not_carried(
    tmp_path: Path, over: dict, needle: str, capsys
) -> None:
    """Every one of these is a way GEOM_TOL - EST_DRIFT_P95 stops being arithmetic: a disqualified
    or absent measurement, an unidentifiable file, a different pixel grid, a segmenter whose
    parameters differ field for field, a different segmenter entirely. None of them is visible in
    the result — two plausible pixel numbers subtract to a plausible pixel number — so each stops
    the carry with the document untouched."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _est_artifact(tmp_path / "pr08_est_drift.json", **over)

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    assert needle in capsys.readouterr().err
    assert out.read_bytes() == before, "nothing may be written when the join fails"
    assert not mgt.sidecar_path(out).exists()


def test_carrying_into_a_document_that_does_not_exist_refuses(tmp_path: Path, capsys) -> None:
    """This mode fills two slots in a COMMITTED document; it does not create one. A file it
    invented would carry an EST_DRIFT_P95 and no committed segmenter contract — precisely the
    document PR-08 §4 step 2 cannot be checked against."""
    est = _est_artifact(tmp_path / "pr08_est_drift.json")
    out = tmp_path / "nothing_here.json"
    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    assert "does not exist" in capsys.readouterr().err
    assert not out.exists()


def test_carrying_into_a_document_with_no_segmenter_block_refuses(tmp_path: Path, capsys) -> None:
    """A document with no contract cannot be joined to anything. Restore it from git rather than
    letting a budget land in a file that can never support the claim it is subtracted under."""
    out = tmp_path / "pr08_geom_tol.json"
    out.write_text(json.dumps({"geom_tol_px": 3.5}, indent=2) + "\n")
    est = _est_artifact(tmp_path / "pr08_est_drift.json")
    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    assert "records no segmenter block" in capsys.readouterr().err


def test_carry_is_not_a_measurement_and_refuses_to_share_a_command_line_with_one(
    tmp_path: Path, capsys
) -> None:
    """It copies a number somebody else measured. Combined with --merge or --shard, one of the two
    jobs on the command line is being ignored, and which one is not obvious from the output."""
    est = _est_artifact(tmp_path / "pr08_est_drift.json")
    rc = run(["--carry-est-drift", str(est), "--merge", str(tmp_path), "--out", str(tmp_path / "o")])
    assert rc == mgt.EXIT_FATAL
    assert "MEASURES NOTHING" in capsys.readouterr().err


# -- --carry-est-drift and the ARM: the one thing in that artifact nothing downstream can see -----
#
# scripts/measure_est_drift.py --arm both measures ONE capture two ways: this repository's adapter
# re-detecting on every frame (`per_frame`), and Cosmos-Transfer2.5's own topology of one frame-0
# detection propagated forward (`propagation`). Both are valid measurements, they differ, and the
# artifact's TOP-LEVEL est_drift_p95_px is the per-frame one BY CONSTRUCTION — measure_est_drift
# computes it from the per-frame pairs. Until 2026-08-27 --carry-est-drift read that field and only
# that field, so the arm G0b's budget came from was chosen by a field name; on the V17 grid that is
# 0.16650 px of margin instead of 0.02997 px, a 5.56x wider per-clip tolerance and 28.5 % of
# GEOM_TOL. And it is invisible afterwards: both arms are the same adapter and record the SAME
# segmenter contract, so contract_disagreements() agrees, run_g0_gates' join on the segmenter name
# matches, and the document reads as a finished gate either way.
#
# Which arm PR-08 §6 subtracts is an OPEN OWNER DECISION and none of these tests answers it. What
# they pin is that the code refuses to answer it either — and that when a person does answer it, the
# answer is written down where a later reader can tell it from a default.


def _two_arm_est_artifact(path: Path, *, per_frame=0.5, propagation=0.9,
                          per_frame_coverage=1.0, propagation_coverage=1.0,
                          min_coverage=0.9, **over) -> Path:
    """An artifact in the shape `measure_est_drift.py measure --arm both` writes.

    The headline number is the per-frame arm's, exactly as that script writes it, because the whole
    point of these tests is that the headline being one particular arm is a property of the producer
    and not a decision anybody made.

    EACH ARM CARRIES ITS OWN COVERAGE, and the document carries the floor, because that is what
    measure_est_drift writes and because the two are not interchangeable: `coverage_below_floor`
    and `headline_valid` are computed from the PER-FRAME arm only, so a document can be
    gate_qualified while its propagation arm sits under the same floor. That is not a hypothetical
    shape — runs/pr08-est-drift/v17/EST_DRIFT-C1-lattice.json is exactly it (per-frame 0.95,
    propagation 0.483, min_coverage 0.9) — so a fixture that omitted the field would let a carry
    of a barely-measured arm pass a test.
    """
    def arm(name: str, p95: float | None, coverage: float | None) -> dict:
        return {"arm": name, "measured": p95 is not None, "absent_because": None,
                "est_drift_p95_px": p95, "n_frames": 480, "n_measured": 480,
                "coverage": coverage}

    return _est_artifact(
        path,
        est_drift_p95_px=per_frame,
        est_drift_p95_px_arm="per_frame",
        coverage=per_frame_coverage,
        min_coverage=min_coverage,
        arm_comparison={
            "arms": ["per_frame", "propagation"],
            "per_frame": arm("per_frame", per_frame, per_frame_coverage),
            "propagation": arm("propagation", propagation, propagation_coverage),
        },
        **over,
    )


def test_a_two_arm_budget_is_not_carried_until_somebody_names_the_arm(
    tmp_path: Path, capsys
) -> None:
    """THE DEFECT, stated as a test. `--arm both` used to change nothing about which number reached
    the gate document: the carry read the top-level field, which is the per-frame arm by
    construction. Two arms in one artifact is not a malformed file and not a mismatch — it is two
    valid measurements 28.5 % of the tolerance apart, and picking one of them is a decision. So the
    carry refuses, names both numbers, and says whose decision it is."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "records 2 measured arms" in err
    # Both candidates are printed, because a refusal that hides the alternative is asking the
    # operator to decide without the numbers.
    assert "per_frame" in err and "propagation" in err
    assert "0.5" in err and "0.9" in err
    # And it says why there is no default to fall back on, rather than merely that there is none.
    assert "TWO-SIDED" in err
    assert "CROSS between p95 and p99" in err
    assert "--est-drift-arm" in err
    assert out.read_bytes() == before, "nothing may be written while the arm is undecided"
    assert not mgt.sidecar_path(out).exists()


@pytest.mark.parametrize("arm,expected,margin", [("per_frame", 0.5, 3.0),
                                                 ("propagation", 0.9, 2.6)])
def test_naming_the_arm_carries_that_arms_number_and_records_that_it_was_named(
    tmp_path: Path, arm: str, expected: float, margin: float
) -> None:
    """The number that lands is the named arm's — including when that is NOT the artifact's headline
    field, which is the half that was unreachable before. And the document records which arm it is
    and that a person said so, because a later reader has no other way to tell a decision from a
    default: both arms are the same adapter under the same segmenter contract."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True, mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", arm]) == mgt.EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["est_drift_p95_px"] == expected
    assert doc["gate_margin_px"] == pytest.approx(margin)
    # THE DURABLE COPY. est_drift_source is a declared measurement slot, so merge_committed_contract
    # carries it into every later artifact written over this document; a new top-level key would not
    # be carried, and the number would outlive the record of which arm it is.
    assert f"arm={arm!r}" in doc["est_drift_source"]
    assert "arm_selected_by=" in doc["est_drift_source"]
    assert "--est-drift-arm" in doc["est_drift_source"]
    # The detail beside it, including the arm that was NOT taken and how far away it was.
    prov = doc["est_drift_arm_provenance"]
    assert prov["arm"] == arm
    assert prov["arms_measured_px"] == {"per_frame": 0.5, "propagation": 0.9}
    assert "--est-drift-arm" in prov["selected_by"]


def test_the_arm_a_carry_recorded_survives_a_later_geom_tol_measurement(tmp_path: Path) -> None:
    """The mechanical reason the arm is written into est_drift_source rather than into a slot of its
    own. A later GEOM_TOL run over the same document carries the measurement slots forward and drops
    everything else, so an arm recorded anywhere but in a declared slot would be lost while the
    number it describes survived — the number outliving what says which measurement it is, which is
    the failure refuse_unnamed_est_drift exists to prevent, one field over."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json")
    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "propagation"]) == mgt.EXIT_OK

    # What a later measuring run brings: its own tolerance, its own segmenter, and nulls in the
    # EST_DRIFT slots it did not measure.
    later = {
        "GEOM_TOL_px": 4.0,
        "mask_method": {"name": "sam2-hiera-large+gdino-base",
                        "params": {"segmenter": dict(STUB_CONTRACT)}},
        "est_drift_p95_px": None,
        "est_drift_source": None,
        mgt.EST_DRIFT_NAME_FIELD: None,
    }
    mgt.merge_committed_contract(out, later)

    assert later["est_drift_p95_px"] == 0.9, "the number is a declared slot and is carried"
    assert "arm='propagation'" in later["est_drift_source"], "so is what says which arm it is"


def test_a_single_arm_budget_still_carries_with_no_flag_and_says_why_no_decision_was_needed(
    tmp_path: Path
) -> None:
    """The behaviour that already worked is untouched: an artifact from a default `--arm per_frame`
    run carries a single number, there is nothing to choose between, and demanding a flag there
    would be refusing the case with no ambiguity in it. It still records the arm and how it was
    settled, so the two kinds of document are read the same way."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["est_drift_p95_px"] == 0.5
    assert "arm='per_frame'" in doc["est_drift_source"]
    assert "no decision to make" in doc["est_drift_source"]
    assert doc["est_drift_arm_provenance"]["arms_measured_px"] == {"per_frame": 0.5}


def test_naming_an_arm_the_measurement_never_drove_refuses(tmp_path: Path, capsys) -> None:
    """--est-drift-arm states a decision about a measurement; it cannot manufacture one. An arm the
    artifact does not carry sends the operator back to the measurement rather than quietly to the
    arm that happens to be in the file."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "propagation"]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "did not measure" in err
    assert "--arm both" in err
    assert out.read_bytes() == before


def test_naming_the_one_arm_a_single_arm_artifact_did_measure_is_accepted(tmp_path: Path) -> None:
    """The flag is refused when it names an arm the artifact did not measure — never merely because
    the artifact measured one. Pinned because the flag's own help text claimed the opposite until
    2026-08-27 ("REQUIRED for such an artifact and refused for any other"), and an operator who
    reads that either does not type it where it is harmless or reports a bug when it works. The
    behaviour is the right one: naming per_frame over a one-arm capture states a decision that
    happened to have no alternative, and the artifact records it as stated rather than as inferred,
    which is the distinction the whole flag exists to keep."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         gate_qualified=True,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    est = _est_artifact(tmp_path / "pr08_est_drift.json")

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "per_frame"]) == mgt.EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["est_drift_p95_px"] == 0.5
    assert "arm='per_frame'" in doc["est_drift_source"]
    assert "--est-drift-arm" in doc["est_drift_arm_provenance"]["selected_by"], (
        "a stated decision must not be recorded as 'there was no decision to make'"
    )


def test_an_artifact_disagreeing_with_itself_about_the_per_frame_arm_is_not_carried(
    tmp_path: Path, capsys
) -> None:
    """The headline and arm_comparison.per_frame are one percentile over one set of pairs, so they
    cannot differ unless the file was edited. Whichever of the two is right, neither is carryable
    from a document that states the same quantity twice and differently."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json")
    doc = json.loads(est.read_text())
    doc["est_drift_p95_px"] = 0.4
    est.write_text(json.dumps(doc, indent=2) + "\n")

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "per_frame"]) == mgt.EXIT_FATAL
    assert "twice and differently" in capsys.readouterr().err
    assert out.read_bytes() == before


def test_an_arm_under_the_documents_own_coverage_floor_is_not_carried(
    tmp_path: Path, capsys
) -> None:
    """THE HOLE NAMING AN ARM OPENS, and it is open on a file that exists.

    measure_est_drift derives `coverage` from the PER-FRAME pairs, stamps `coverage_below_floor`
    from it and computes `headline_valid` from it; the propagation arm's coverage is written inside
    its own block and read by nothing. So `gate_qualified` — the flag --carry-est-drift refuses on
    — is a verdict about the per-frame number, and until an arm could be named that was harmless,
    because the per-frame number was the only one carryable. These are C1-lattice's real numbers:
    runs/pr08-est-drift/v17/EST_DRIFT-C1-lattice.json states a propagation p95 of 0.3194 px over 29
    of 60 frames against its own registered min_coverage of 0.9, and its three disqualification
    reasons are all about the estimator flag and the committed GEOM_TOL document — none about
    coverage. The moment those clear, that artifact is carryable, and the arm flag would hand G0b a
    percentile over under half a capture."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _two_arm_est_artifact(
        tmp_path / "pr08_est_drift.json",
        per_frame=0.1855527738074582, propagation=0.3194458657163223,
        per_frame_coverage=0.95, propagation_coverage=0.48333333333333334, min_coverage=0.9,
    )

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "propagation"]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "0.48333333333333334" in err and "0.9" in err
    # It must say why the document's own pass does not cover this number, not merely that it
    # refuses: the operator is holding a file stamped gate_qualified.
    assert "PER-FRAME arm only" in err
    assert out.read_bytes() == before
    assert not mgt.sidecar_path(out).exists()


def test_an_arm_whose_sample_the_artifact_never_states_is_not_carried(
    tmp_path: Path, capsys
) -> None:
    """And absence refuses rather than passes. A p95 whose sample nothing states is the
    default-permissive pattern the reader's side of this cross-check was repaired for on
    2026-08-22 — `if est['resolution_hw'] is not None and ...` one field over. Nothing already on
    disk is disqualified by refusing here, because naming an arm is itself new."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json", propagation_coverage=None)

    assert run(["--carry-est-drift", str(est), "--out", str(out),
                "--est-drift-arm", "propagation"]) == mgt.EXIT_FATAL
    assert "nothing to compare" in capsys.readouterr().err
    assert out.read_bytes() == before


def test_a_qualified_artifact_whose_comparison_block_drops_the_per_frame_arm_is_not_carried(
    tmp_path: Path, capsys
) -> None:
    """THE SAME DEFECT ARRIVING THROUGH THE BACK DOOR, which is why the self-agreement check is not
    conditional on which arm was asked for. An artifact that is gate_qualified and states a
    top-level p95 while its arm_comparison records the per-frame arm as NOT measured cannot have
    been written by measure_est_drift — `per_frame_arm_not_measured` disqualifies exactly that run.
    Read literally, though, it leaves ONE measured arm, so the two-arm refusal does not fire, and
    the carry would take the propagation number with no flag on the command line and no decision
    behind it: an arm chosen by a malformed file instead of by a field name."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    before = out.read_bytes()
    est = _two_arm_est_artifact(tmp_path / "pr08_est_drift.json")
    doc = json.loads(est.read_text())
    doc["arm_comparison"]["per_frame"]["measured"] = False
    doc["arm_comparison"]["per_frame"]["est_drift_p95_px"] = None
    est.write_text(json.dumps(doc, indent=2) + "\n")

    assert run(["--carry-est-drift", str(est), "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "per_frame_arm_not_measured" in err
    assert "contradicts itself" in err
    assert out.read_bytes() == before


def test_an_arm_flag_on_a_command_line_that_carries_nothing_refuses(tmp_path: Path, capsys) -> None:
    """A flag that is silently ignored is worse than no flag: this one exists so that a decision is
    legible afterwards, and an operator who typed it on a measuring command line would believe they
    had stated one."""
    corpus, masks = make_corpus(tmp_path, {"ep0": walk((10, 10), (2, 0), 5)})
    rc = run(["--corpus", str(corpus), "--masks", str(masks), "--out", str(tmp_path / "o.json"),
              "--est-drift-arm", "propagation"])
    assert rc == mgt.EXIT_FATAL
    assert "carries nothing" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_a_pooled_artifact_names_what_is_missing_and_that_the_decision_is_open(
    tmp_path: Path, capsys
) -> None:
    """DEFECT 8, made legible rather than fixed. scripts/pool_est_drift_arms.py writes a different
    schema with a different shape, and no committed tool can carry it into the gate document. That
    is not an oversight to route around: whether G0b's budget is the POOLED p95 or a single
    capture's is a question T40_RULE_V17 §4 explicitly declines to answer, and building the carry
    would answer it by making one of the two the reachable one. So the refusal stands and what it
    owes the operator is the whole list plus the name of the open question — not `does not carry
    schema wam.est_drift/1`, which reads as a conversion problem."""
    out = write_contract(tmp_path / "pr08_geom_tol.json", geom_tol_px=3.5, GEOM_TOL_px=3.5,
                         mask_method={"name": "sam2-hiera-large+gdino-base"})
    pooled = tmp_path / "POOLED.json"
    pooled.write_text(json.dumps({
        "schema": "wam.est_drift_pooled/1",
        "arms": {"per_frame": {"pooled_est_drift_p95_px": 0.31},
                 "propagation": {"pooled_est_drift_p95_px": 0.45}},
    }, indent=2) + "\n")

    assert run(["--carry-est-drift", str(pooled), "--out", str(out)]) == mgt.EXIT_FATAL
    err = capsys.readouterr().err
    assert "POOLED artifact" in err
    # Every one of the four ways it is not carryable, named separately.
    assert "gate_qualified" in err
    assert "est_drift_p95_px — absent" in err
    assert "estimators.name" in err
    assert "resolution_hw" in err
    # And the reason the answer is not a code change.
    assert "T40_RULE_V17" in err
    assert "does not get to do that" in err
    assert json.loads(out.read_text())["est_drift_p95_px"] is None


# -- what the adapter saw, recorded beside what the harness measured -----------------------------
#
# The evidence the adapter's second gate-qualification blocker asks for "from a full pass" — the
# retry counts and the detection-score distribution — had nowhere to land until 2026-08-22: neither
# harness read ``estimators.apple_sam2.stats()``, so the 402-episode, ~171 600-frame GEOM_TOL run
# produced none of it. These tests are about the place it lands, and about the three ways that place
# could be worse than useless: numbers that belong to a previous run in the same interpreter, a
# distribution that does not survive the 8-way shard, and zeros written where nobody looked.


def counting_adapter(monkeypatch, **kw) -> types.ModuleType:
    """An adapter stub that keeps the counters and the score list the real one keeps.

    CUMULATIVE, exactly as ``scripts/estimators/apple_sam2.py`` is: nothing resets them, so a second
    measurement in the same interpreter sees the first one's totals. That is the property the probe
    exists for, and a stub that reset itself per run would let a harness reading the totals straight
    into its artifact pass.

    The score is a function of the FRAME and not of the call order, so the same frame scores the
    same whichever shard segments it — without that the sharded and un-sharded runs would pool
    different values and the exactness test below would be asserting something it could not have.
    """
    mod = install_adapter(monkeypatch, **kw)
    mod.DETECTION_SCORES = []
    mod.SEGMENT_CALLS = 0
    mod.NO_DETECTION_FRAMES = 0
    mod.EMPTY_MASK_FRAMES = 0
    mod.RETRY_FRAMES = 0
    mod.RETRY_RECOVERED_FRAMES = 0
    inner = mod.segment

    def segment(rgb):
        mod.SEGMENT_CALLS += 1
        mask = inner(rgb)
        if not np.asarray(mask).any():
            mod.NO_DETECTION_FRAMES += 1
            return mask
        # Position-sensitive, so two frames of the same blob at two places do NOT score the same:
        # a fixture where every score is equal is permutation-invariant, and the exactness claim
        # below would then be a claim no wrong implementation could fail.
        red = np.asarray(rgb)[:, :, 0].astype(np.int64)
        h, w = red.shape
        key = int((red * (np.arange(w) + 1)).sum() + (red.T * (np.arange(h) + 1)).sum())
        score = round(0.08 + (key % 92) / 100.0, 6)
        if score < 0.15:                      # only the retry can produce one of these
            mod.RETRY_FRAMES += 1
            mod.RETRY_RECOVERED_FRAMES += 1
        mod.DETECTION_SCORES.append(score)
        return mask

    def stats():
        return {
            "estimator_name": mod.ESTIMATOR_NAME,
            "estimator_version": mod.ESTIMATOR_VERSION,
            "gate_qualified": mod.GATE_QUALIFIED,
            "box_threshold": 0.15,
            "retry_box_threshold": 0.1,
            "n_segment_calls": mod.SEGMENT_CALLS,
            "n_frames_without_detection": mod.NO_DETECTION_FRAMES,
            "n_frames_with_empty_mask": mod.EMPTY_MASK_FRAMES,
            "n_frames_retry_fired": mod.RETRY_FRAMES,
            "n_frames_retry_recovered": mod.RETRY_RECOVERED_FRAMES,
            "n_detection_scores": len(mod.DETECTION_SCORES),
        }

    mod.segment = segment
    mod.stats = stats
    return mod


def _sam2_run(tmp_path: Path, monkeypatch, corners: dict, out: Path, extra=None) -> dict:
    """One --method sam2 measurement over a synthetic corpus, and the artifact it wrote."""
    corpus, _ = make_corpus(tmp_path, corners)
    install_video_frames(monkeypatch, {k: bgr_frames(v) for k, v in corners.items()})
    rc = run(["--corpus", str(corpus), "--method", "sam2", "--out", str(out), *(extra or [])])
    assert rc == mgt.EXIT_OK, rc
    return json.loads(out.read_text())


def test_the_artifact_records_the_retry_counts_and_the_score_distribution(
        tmp_path: Path, monkeypatch) -> None:
    """The full-pass evidence blocker 2 asks for, in the artifact the full pass writes.

    Counts AND scores: "the retry fired 41 times" does not say whether it bought 41 confident
    detections or 41 masks of the plate, and the 169-frame local audit found the scores sharply
    bimodal — p25 0.758 where the mask was right, 0.155-0.264 on every flagged frame. The part of
    the distribution below box_threshold is the retry's contribution by construction: the first
    pass discards everything under it.
    """
    counting_adapter(monkeypatch)
    rec = _sam2_run(tmp_path, monkeypatch, {"ep0": walk((10, 10), (2, 0), 5)},
                    tmp_path / "geom_tol.json")

    stats = rec["estimator_stats"]
    assert stats["recorded"] is True
    assert stats["this_run"]["n_segment_calls"] == 5
    assert stats["this_run"]["n_frames_without_detection"] == 0
    for key in ("n_frames_retry_fired", "n_frames_retry_recovered"):
        assert isinstance(stats["this_run"][key], int), key
    dist = stats["detection_scores"]["distribution"]
    assert dist["n"] == 5
    assert stats["detection_scores"]["n"] == 5
    assert dist["box_threshold"] == 0.15
    assert dist["n_below_box_threshold"] == stats["this_run"]["n_frames_retry_recovered"]
    assert 0.0 <= dist["min"] <= dist["percentiles"]["p50"] <= dist["max"] <= 1.0
    assert sum(dist["histogram"]["counts"]) == 5
    # The descriptive half of stats() is carried verbatim, so the artifact says which adapter, at
    # which operating point, produced the counts beside them.
    assert stats["adapter"]["estimator_name"] == rec["mask_method"]["name"]
    assert stats["adapter"]["box_threshold"] == 0.15


def test_the_recorded_counters_belong_to_this_run_and_not_to_the_interpreter(
        tmp_path: Path, monkeypatch) -> None:
    """THE LEAK THIS DESIGN EXISTS AGAINST. The adapter's counters are lifetime totals — nothing in
    it resets them — so a harness that copied ``stats()`` into its artifact would report the first
    measurement's frames in the second measurement's record, and the number would look right.

    Two runs, one interpreter, one adapter instance. The second artifact must count its own five
    frames, and must SAY that the interpreter was not fresh rather than hiding it.
    """
    counting_adapter(monkeypatch)
    first = _sam2_run(tmp_path, monkeypatch, {"ep0": walk((10, 10), (2, 0), 5)},
                      tmp_path / "first.json")
    second = _sam2_run(tmp_path, monkeypatch, {"ep0": walk((10, 10), (2, 0), 5)},
                       tmp_path / "second.json")

    assert first["estimator_stats"]["this_run"]["n_segment_calls"] == 5
    assert second["estimator_stats"]["this_run"]["n_segment_calls"] == 5, (
        "the second run recorded the interpreter's total, not its own frames")
    assert second["estimator_stats"]["counters_at_end_of_run"]["n_segment_calls"] == 10
    assert second["estimator_stats"]["counters_at_start_of_run"]["n_segment_calls"] == 5, (
        "a non-fresh interpreter has to be visible in the artifact, not corrected away silently")
    assert second["estimator_stats"]["detection_scores"]["n"] == 5
    # And the adapter's own state was not touched to achieve it: a harness that reset somebody
    # else's module counters would break the next caller instead of itself.
    assert sys.modules[SAM2_SPEC].SEGMENT_CALLS == 10


def test_an_adapter_that_exports_no_stats_records_an_absence_and_not_zeros(
        tmp_path: Path, monkeypatch) -> None:
    """The contract both harnesses call is segment(rgb)/estimate_depth(rgb). stats() is an extra.

    An adapter without one must measure exactly as before — and "we did not look" must not be
    written down as "the retry never fired", which is what a block of zeros would say to a reader
    holding a full pass and a blocker asking whether the retry bought its coverage.
    """
    install_adapter(monkeypatch)                      # no stats(), no DETECTION_SCORES
    rec = _sam2_run(tmp_path, monkeypatch, {"ep0": walk((10, 10), (2, 0), 5)},
                    tmp_path / "geom_tol.json")

    stats = rec["estimator_stats"]
    assert stats["recorded"] is False
    assert "stats()" in stats["absent_because"]
    assert stats["this_run"] is None and stats["counters_at_end_of_run"] is None
    assert stats["detection_scores"]["recorded"] is False
    assert stats["detection_scores"]["n"] is None
    assert "Absent is not zero" in stats["note"]
    # The measurement is untouched.
    assert rec["GEOM_TOL_px"] == pytest.approx(2.0)
    assert rec["gate_qualified"] is True


def test_a_stats_that_raises_loses_the_evidence_and_not_the_measurement(
        tmp_path: Path, monkeypatch) -> None:
    """stats() is read for the record only. A GEOM_TOL pass costs four GPU-hours and must not die
    on the way to its artifact because an optional accessor threw."""
    mod = install_adapter(monkeypatch)

    def boom():
        raise RuntimeError("no counters here")

    mod.stats = boom
    rec = _sam2_run(tmp_path, monkeypatch, {"ep0": walk((10, 10), (2, 0), 5)},
                    tmp_path / "geom_tol.json")
    assert rec["GEOM_TOL_px"] == pytest.approx(2.0)
    assert rec["estimator_stats"]["recorded"] is False
    assert "RuntimeError: no counters here" in rec["estimator_stats"]["absent_because"]


def _sam2_shards(tmp_path: Path, monkeypatch, corners: dict, n: int) -> list[str]:
    corpus, _ = make_corpus(tmp_path, corners)
    install_video_frames(monkeypatch, {k: bgr_frames(v) for k, v in corners.items()})
    paths = []
    for i in range(n):
        p = tmp_path / f"sam2-shard-{i}.json"
        assert run(["--corpus", str(corpus), "--method", "sam2", "--out", str(p),
                    "--shard", str(i), "--num-shards", str(n)]) == mgt.EXIT_OK
        paths.append(str(p))
    return paths


#: The fields of ``estimator_stats`` that are about the PROCESS rather than about the corpus, and
#: are therefore the only ones a merged block does not reproduce from the un-sharded one. The
#: lifetime totals of eight processes do not sum to anything, so the merge nulls them and lists them
#: per shard; everything else — the counts, the scores, the distribution, the adapter's own
#: description of itself — is the same object either way, which is the same standard
#: ``test_the_merged_artifact_is_the_unsharded_artifact_exactly`` holds the rest of the record to.
_STATS_PROCESS_LOCAL = {"counters_at_start_of_run", "counters_at_end_of_run", "per_shard"}


def test_the_merged_score_distribution_is_the_unsharded_one_exactly(
        tmp_path: Path, monkeypatch) -> None:
    """Eleven episodes, three shards, and a distribution that has to survive the trip.

    ``mean`` and ``std`` are floating-point sums and every histogram count is a count over the same
    values, so pooling the shards in shard order rather than in the corpus's own enumeration order
    would give a merged artifact that is approximately, not exactly, the un-sharded one — the same
    failure the raw per-step displacements exist to prevent, and the reason the scores are carried
    per EPISODE and sorted by ``episode_index`` rather than concatenated per shard.
    """
    corners = {f"ep{i:02d}": walk((10, 10 + 3 * i), (1 + (i % 5), i % 3), 6 + (i % 4))
               for i in range(11)}
    mod = counting_adapter(monkeypatch)

    full = tmp_path / "full.json"
    reference = _sam2_run(tmp_path, monkeypatch, corners, full)
    # The un-sharded run's scores IN THE ORDER IT SEGMENTED THEM, read off the adapter rather than
    # out of the artifact (the artifact keeps the distribution, not the values). This is the
    # sequence the merge has to rebuild, and it is not permutation-invariant: the fixture scores
    # each frame by where its blob is, so a pool assembled in shard order is a different list.
    unsharded_scores = list(mod.DETECTION_SCORES)
    assert len(set(unsharded_scores)) > 1, "the fixture must not score every frame the same"

    shards = _sam2_shards(tmp_path, monkeypatch, corners, 3)
    merged_path = tmp_path / "merged.json"
    assert run(["--merge", *shards, "--out", str(merged_path)]) == mgt.EXIT_OK
    merged = json.loads(merged_path.read_text())

    ref_stats, got_stats = reference["estimator_stats"], merged["estimator_stats"]
    assert got_stats["recorded"] is True
    assert got_stats["detection_scores"]["distribution"] == (
        ref_stats["detection_scores"]["distribution"]), "the pooled distribution is not exact"
    differing = sorted(k for k in set(ref_stats) | set(got_stats)
                       if k not in _STATS_PROCESS_LOCAL and ref_stats.get(k) != got_stats.get(k))
    assert differing == [], f"merged and un-sharded estimator_stats disagree on {differing}"
    assert got_stats["this_run"]["n_segment_calls"] == sum(
        s["this_run"]["n_segment_calls"] for s in got_stats["per_shard"])
    # WHERE THE RAW VALUES LIVE, which is exactly where displacements_px lives: attributed to an
    # episode, in a shard artifact, and nowhere in the committed one. Attributed rather than
    # top-level because the merge sorts by episode_index; absent from the merged artifact because
    # the pooled statistic is what the merge exists to produce and the tracked document stays small.
    assert "values" not in got_stats["detection_scores"]
    assert all("detection_scores" not in ep for ep in merged["per_episode"])
    shard_zero = json.loads(Path(shards[0]).read_text())
    assert "values" not in shard_zero["estimator_stats"]["detection_scores"]
    assert all(ep["detection_scores"] for ep in shard_zero["per_episode"])
    # And the pooled values are all of them, in the corpus's own order and not in the shards'.
    by_episode = sorted(
        (ep["episode_index"], ep["detection_scores"])
        for p in shards for ep in json.loads(Path(p).read_text())["per_episode"])
    pooled = [v for _, scores in by_episode for v in scores]
    assert pooled == unsharded_scores
    shard_order = [v for p in shards
                   for ep in json.loads(Path(p).read_text())["per_episode"]
                   for v in ep["detection_scores"]]
    assert shard_order != unsharded_scores, (
        "this fixture must separate the two orders, or the assertion above is vacuous")


def test_a_shard_that_recorded_no_adapter_stats_is_an_absence_and_not_a_dead_merge(
        tmp_path: Path, monkeypatch) -> None:
    """A shard written by an older version of this script carries no estimator_stats at all.

    The merge must still produce GEOM_TOL — the counts are evidence, nothing subtracts them — and
    must say WHY the evidence is missing rather than pooling the shards that do have it and
    reporting the result as if it covered the corpus.
    """
    corners = {f"ep{i:02d}": walk((10, 10 + 3 * i), (1 + (i % 5), i % 3), 6 + (i % 4))
               for i in range(11)}
    counting_adapter(monkeypatch)
    shards = _sam2_shards(tmp_path, monkeypatch, corners, 3)
    victim = Path(shards[1])
    rec = json.loads(victim.read_text())
    rec.pop("estimator_stats")
    victim.write_text(json.dumps(rec, indent=2) + "\n")

    merged_path = tmp_path / "merged.json"
    assert run(["--merge", *shards, "--out", str(merged_path)]) == mgt.EXIT_OK
    merged = json.loads(merged_path.read_text())
    assert merged["GEOM_TOL_px"] is not None
    assert merged["gate_qualified"] is True
    stats = merged["estimator_stats"]
    assert stats["recorded"] is False
    assert "shard 1" in stats["absent_because"]
    assert stats["this_run"] is None


def test_a_shard_that_kept_only_its_distribution_cannot_be_pooled(
        tmp_path: Path, monkeypatch) -> None:
    """The scores' half of "a median does not decompose". Two binned distributions pool only if
    they were binned identically and even then not exactly, so the merge takes the raw values or
    records that it could not."""
    corners = {f"ep{i:02d}": walk((10, 10 + 3 * i), (1 + (i % 5), i % 3), 6 + (i % 4))
               for i in range(11)}
    counting_adapter(monkeypatch)
    shards = _sam2_shards(tmp_path, monkeypatch, corners, 3)
    victim = Path(shards[2])
    rec = json.loads(victim.read_text())
    for ep in rec["per_episode"]:
        ep.pop("detection_scores")
    victim.write_text(json.dumps(rec, indent=2) + "\n")

    merged_path = tmp_path / "merged.json"
    assert run(["--merge", *shards, "--out", str(merged_path)]) == mgt.EXIT_OK
    stats = json.loads(merged_path.read_text())["estimator_stats"]
    # The counts still pool — they are per-shard totals and none of them went missing.
    assert stats["recorded"] is True
    assert stats["this_run"]["n_segment_calls"] > 0
    assert stats["detection_scores"]["recorded"] is False
    assert "no raw detection_scores" in stats["detection_scores"]["absent_because"]
    assert stats["detection_scores"]["distribution"] is None


# -- the additive claim, checked against the script as it was ------------------------------------


def _previous_version(tmp_path: Path, name: str, commit: str = "d9ac5d1") -> object:
    """Import ``scripts/<name>.py`` AS IT WAS AT ``commit``, under its own module name.

    ONE substitution is made in the source: ``_REPO_ROOT``, which both scripts derive from
    ``__file__`` and use to find the repository, its git commit and the file the est_drift blocker
    is read out of. A copy living under tmp_path would otherwise describe a repository that is not
    this one, and the comparison would be against fields that differ for a reason having nothing to
    do with the change under test. Every other byte is the previous version's.
    """
    import importlib.util
    import subprocess

    src = subprocess.run(["git", "show", f"{commit}:scripts/{name}.py"],
                         cwd=str(Path(mgt.__file__).resolve().parents[1]),
                         capture_output=True, text=True, check=True).stdout
    anchor = "_REPO_ROOT = Path(__file__).resolve().parent.parent"
    assert src.count(anchor) == 1, f"{name} at {commit} no longer derives _REPO_ROOT that way"
    src = src.replace(anchor, f"_REPO_ROOT = Path({str(Path(mgt.__file__).resolve().parents[1])!r})")
    path = tmp_path / f"{name}_at_{commit}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"{name}_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Provenance of the individual run rather than of the measurement: a timestamp, and the two paths
#: that name the file being written. Everything else in the record is the measurement and must come
#: out of both versions identically.
_RUN_LOCAL = {"measured_utc", "artifact_path", "artifact_sha256_sidecar"}


def test_recording_the_adapter_stats_changed_no_number_this_script_already_produced(
        tmp_path: Path, monkeypatch) -> None:
    """THE WHOLE CLAIM OF THIS CHANGE, checked against the script as it was rather than asserted.

    ``estimator_stats`` is additive: it is written beside the measurement, nothing reads it back,
    and no refusal, exit code or gate flag depends on it. The way to be sure of that is not to read
    the diff — it is to run the previous version of the script over the same fixture and compare
    the two artifacts key for key. The only permitted differences are the new keys themselves and
    the per-run provenance.

    ``git_commit_source`` is the SECOND key added since the pinned baseline, and it is additive on
    the same terms. It names which of two sources answered ``_git_commit()`` — a live
    ``git rev-parse`` or the ``GIT_COMMIT`` file ``cluster/discoverer/sync.sh`` writes beside the
    rsynced tree. The fallback exists because the cluster copy is not a git repository, so
    ``rev-parse`` failed there and ``git_commit`` came out ``null`` on all sixteen GEOM_TOL shards
    of 2026-08-23/24 and on their merge. The set below is the declaration: a key that appears here
    without being added to it fails this test, which is the point.
    """
    corners = {"ep0": walk((10, 10), (2, 0), 5),
               "ep1": walk((10, 40), (3, 4), 6),
               "ep2": walk((60, 10), (0, 8), 4)}
    old = _previous_version(tmp_path, "measure_geom_tol")
    counting_adapter(monkeypatch)
    corpus, masks = make_corpus(tmp_path, corners)
    frames = {k: bgr_frames(v) for k, v in corners.items()}
    install_video_frames(monkeypatch, frames)
    install_video_frames(monkeypatch, frames, module=old)

    argv = ["--corpus", str(corpus), "--method", "sam2", "--out"]
    assert old.main([*argv, str(tmp_path / "before.json")]) == mgt.EXIT_OK
    assert run([*argv, str(tmp_path / "after.json")]) == mgt.EXIT_OK
    before = json.loads((tmp_path / "before.json").read_text())
    after = json.loads((tmp_path / "after.json").read_text())

    assert set(after) - set(before) == {"estimator_stats", "git_commit_source"}, \
        "a key appeared that was not declared"
    assert set(before) - set(after) == set(), "a key the previous version wrote went missing"
    differing = sorted(k for k in before if k not in _RUN_LOCAL and before[k] != after[k])
    assert differing == [], f"recording the adapter's stats changed {differing}"

    # The same, on the path that reads masks off disk and involves no estimator at all — where the
    # new block is an ABSENCE, which must also change nothing.
    argv = ["--corpus", str(corpus), "--masks", str(masks), "--out"]
    assert old.main([*argv, str(tmp_path / "before_masks.json")]) == mgt.EXIT_OK
    assert run([*argv, str(tmp_path / "after_masks.json")]) == mgt.EXIT_OK
    before = json.loads((tmp_path / "before_masks.json").read_text())
    after = json.loads((tmp_path / "after_masks.json").read_text())
    assert after["estimator_stats"]["recorded"] is False
    differing = sorted(k for k in before if k not in _RUN_LOCAL and before[k] != after[k])
    assert differing == [], f"recording the adapter's stats changed {differing}"


def test_the_merge_is_unchanged_by_it_too(tmp_path: Path, monkeypatch) -> None:
    """The merge has its own record-building path, and the same claim has to hold there.

    Shards written by the previous version merge under the previous version; shards written by this
    one merge under this one; the two merged artifacts agree on everything the gate reads.
    """
    old = _previous_version(tmp_path, "measure_geom_tol")
    corpus, masks = _eleven_episode_corpus(tmp_path)

    def shard_with(module, tag: str) -> list[str]:
        paths = []
        for i in range(3):
            p = tmp_path / f"{tag}-{i}.json"
            assert module.main(["--corpus", str(corpus), "--masks", str(masks), "--out", str(p),
                                "--shard", str(i), "--num-shards", "3"]) == mgt.EXIT_OK
            paths.append(str(p))
        return paths

    assert old.main(["--merge", *shard_with(old, "old"), "--out",
                     str(tmp_path / "before.json")]) == mgt.EXIT_OK
    assert run(["--merge", *shard_with(mgt, "new"), "--out",
                str(tmp_path / "after.json")]) == mgt.EXIT_OK
    before = json.loads((tmp_path / "before.json").read_text())
    after = json.loads((tmp_path / "after.json").read_text())

    assert set(after) - set(before) == {"estimator_stats", "git_commit_source"}
    ignore = _RUN_LOCAL | {"merged_from"}          # names the shard paths and their digests
    differing = sorted(k for k in before if k not in ignore and before[k] != after[k])
    assert differing == [], f"the merge changed {differing}"
    assert before["merged_from"]["shards"][0]["n_steps_measured"] == (
        after["merged_from"]["shards"][0]["n_steps_measured"])


def test_the_counter_names_this_module_differences_are_the_ones_the_adapter_exports() -> None:
    """A renamed counter would record ``null`` for every frame instead of failing.

    ``ADAPTER_RUN_COUNTERS`` and ``ADAPTER_SCORES_ATTR`` are read off the adapter by name. The
    failure mode of a rename is silent and it is the worst-shaped one available here: the run
    completes, the artifact is written, ``this_run`` says ``null`` for the retry counts, and it
    reads to a later human exactly like an adapter that was asked and had nothing to say. Checked by
    PARSING the adapter, because importing it needs transformers, torch and 3 GB of weights, and
    these tests run with none of the three.
    """
    import ast

    tree = ast.parse(mgt.SAM2_ADAPTER_FILE.read_text(encoding="utf-8"))
    assigned = {t.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                for t in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
                if isinstance(t, ast.Name)}
    assert mgt.ADAPTER_SCORES_ATTR in assigned, (
        f"{mgt.SAM2_ADAPTER_SPEC} no longer declares {mgt.ADAPTER_SCORES_ATTR}; the recorded "
        "detection-score distribution would be an absence on every run, silently")

    stats_fn = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "stats")
    returned = next(n for n in ast.walk(stats_fn) if isinstance(n, ast.Dict))
    keys = {k.value for k in returned.keys if isinstance(k, ast.Constant)}
    missing = [k for k in mgt.ADAPTER_RUN_COUNTERS if k not in keys]
    assert missing == [], (
        f"{mgt.SAM2_ADAPTER_SPEC}.stats() no longer reports {missing}. Those counters would be "
        "recorded as null — 'we asked and it said nothing' — on every full pass, which is what "
        "the adapter's second gate-qualification blocker asks for and would not get.")


# -- 103_measure_geom_tol.sbatch: the plumbing that decides whether the measurement survives -------
#
# Job 189658 lost roughly six GPU-hours and produced zero artifacts. Two of the three reasons are
# not in measure_geom_tol.py at all — they are in the sbatch that drives it — and both are the kind
# of defect that costs the hours BEFORE anyone can read a Python traceback:
#
#   the walltime self-check   It sized the shard from GEOM_TOL_PILOT.json, an artifact produced at
#   trusted a discredited     box_threshold 0.35 with no retry branch, while the adapter had moved
#   pilot                     to 0.15 with a (0.10, 0.10) retry. Against that file it printed a
#                             comfortable estimate and NO warning for an array that was about to
#                             die at the wall. A self-check that reassures is worse than none.
#
#   exit 3 was fatal, so no   apple_sam2.GATE_QUALIFIED is False, so EVERY correct shard exits 3
#   shard could ever land     with the single reason "mask method ... is not gate-qualified". The
#                             job called that FATAL and the resume check refused to reuse the
#                             artifact, so every re-submission re-measured the whole partition and
#                             the array could never converge. Sharding buys resumability; this gave
#                             it back.
#
# These tests RUN THE SBATCH'S OWN HEREDOCS, extracted from the file, rather than a copy of them
# here. A copy would pass forever after the sbatch changed.

import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

SBATCH_103 = _REPO_ROOT / "cluster/discoverer/103_measure_geom_tol.sbatch"


def _heredoc_after(anchor: str) -> str:
    """The first ``<<'PY' ... PY`` block at or after ``anchor``. Refuses rather than guessing."""
    text = SBATCH_103.read_text(encoding="utf-8")
    at = text.index(anchor)
    start = text.index("<<'PY'\n", at) + len("<<'PY'\n")
    end = text.index("\nPY\n", start)
    return text[start:end]


def _run_snippet(source: str, argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", source, *argv],
                          capture_output=True, text=True,
                          env={**os.environ, **env})


def _landed(tmp_path: Path, doc: dict, *, index: int = 0, num_shards: int = 16,
            step: int = 1) -> subprocess.CompletedProcess:
    """Run the sbatch's ``shard_artifact_landed()`` against the REAL adapter and the REAL contract.

    THE ENVIRONMENT IS PART OF THE CALL AND USED NOT TO BE. This helper used to pass ``{}``, which
    was harmless only while the heredoc imported nothing: the function looked at the shard and at
    nothing else. It now resolves ``measure_geom_tol`` and the adapter named by
    ``SAM2_ADAPTER_SPEC``, and reads the committed document, because a shard measured under a
    different contract or under a different standing gate flag is stale and must be re-measured
    rather than reused. Those imports are FAIL-CLOSED — an empty environment does not mean "skip
    the check", it means "cannot check, therefore not reusable" — so a helper that supplies no
    ``PYTHONPATH`` and no ``CONTRACT_FILE`` stops testing the classification at all and starts
    testing the refusal it falls back to.

    The real ``scripts/`` and the real ``configs/transfer25/pr08_geom_tol.json`` are handed over
    deliberately. The parametrised cases below are all about the shard's OWN fields, so pinning
    them against the live adapter is what keeps them honest: if the adapter or the committed block
    moves, these tests move with it rather than passing against a stub that froze in 2026.
    ``tests/test_103_measure_geom_tol_sbatch.py`` is the file that varies the adapter and the
    contract themselves, with a throwaway ``scripts/`` per case; this file does not duplicate that
    machinery.
    """
    path = tmp_path / "shard-0.json"
    path.write_text(json.dumps(doc))
    return _run_snippet(_heredoc_after("shard_artifact_landed () {"),
                        [str(path), str(index), str(num_shards), str(step)],
                        {"PYTHONPATH": str(_REPO_ROOT / "scripts"),
                         "CONTRACT_FILE": str(_REPO_ROOT / "configs/transfer25/pr08_geom_tol.json")})


#: The committed segmenter block, read off the document rather than restated. A copy here would
#: agree with the contract on the day it was written and silently stop agreeing afterwards, which
#: is the exact failure ``contract_disagreements()`` exists to catch — so the fixture that has to
#: AGREE with the contract takes it from the contract.
COMMITTED_SEGMENTER = json.loads(
    (_REPO_ROOT / "configs/transfer25/pr08_geom_tol.json").read_text(encoding="utf-8")
)["segmenter"]


def _shard_doc(**over) -> dict:
    """A shard artifact in the shape main() writes for a complete, correctly measured shard.

    It carries ``mask_method.params.segmenter`` because a real one does: that block is what the
    shard actually ran with, and ``shard_artifact_landed()`` compares it field for field against
    the committed document. A fixture without it is not a shard main() would write, and it would
    be refused for a reason none of these tests is about.
    """
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
                        "params": {"segmenter": dict(COMMITTED_SEGMENTER)}},
    }
    doc.update(over)
    return doc


#: The exact sentence ``measure_geom_tol.main()`` appends when the adapter has not opted in. Built
#: the way main() builds it, so a reword there fails this file rather than silently turning the
#: sbatch's re-classification into a rule that matches nothing.
ADAPTER_BLOCKER_REASON = f"mask method {'grounding-dino+sam2+depth-anything-v2'!r} is not gate-qualified"


def test_the_adapter_blocker_sentence_is_the_one_main_actually_writes() -> None:
    """The sbatch re-classifies exit 3 by matching a reason string. Pin the string to its producer.

    If ``main()`` rewords that reason, the sbatch stops recognising it, every shard goes back to
    FATAL, and the array goes back to being unable to make progress — silently, and only on the
    cluster. So the sentence is read out of the source of the branch that writes it.
    """
    source = mgt.__file__ and Path(mgt.__file__).read_text(encoding="utf-8")
    assert 'f"mask method {method.name!r} is not gate-qualified"' in source, (
        "measure_geom_tol.main() no longer writes the reason 103's shard_artifact_landed() "
        "matches. Update BOTH, or every shard of the partition becomes FATAL again.")
    pattern = re.compile(r"^mask method .* is not gate-qualified$")
    assert pattern.match(ADAPTER_BLOCKER_REASON)


def test_a_shard_the_standing_flag_has_moved_past_is_re_measured(tmp_path) -> None:
    """THE FORGIVENESS EXPIRES WHEN THE THING IT FORGAVE STOPS BEING TRUE.

    This test used to assert the opposite, and it was right to. ``GATE_QUALIFIED = False`` was a
    property of scripts/estimators/apple_sam2.py rather than of any shard: identical across every
    shard of the partition, re-derived by the merge from the same artifact, and unchanged by
    re-measuring. Refusing to reuse a shard over it meant each re-submission re-measured the whole
    corpus, which made sharding pointless — so the sbatch forgave exactly that one reason.

    THE FLAG MOVED ON 2026-08-27 (``13f0416``) AND THE FORGIVENESS INVERTED. Re-measuring such a
    shard today does NOT reproduce it: it produces a gate-qualified one. Reusing it instead skips
    exactly the work that has to be redone, and — because the resume check runs BEFORE the
    contract-and-gate preflight — it does so without the preflight ever executing. Every task of a
    default submission would have printed "already landed. Skipping.", exited 0 in seconds, and
    left the merge pooling a permanently-disqualified median. That is what this test now pins.

    The other half of the rule — that the shard IS still reusable while the flag is still False —
    cannot be checked here, because this file runs against the LIVE adapter. It is covered against
    a stub adapter in ``tests/test_103_measure_geom_tol_sbatch.py``, which is the file that owns
    varying the flag.
    """
    r = _landed(tmp_path, _shard_doc())
    assert r.returncode == 1, (
        "a shard disqualified only by a standing flag that has since moved was reused. The "
        "re-measurement it skips is the one that would clear the disqualification.\n"
        + r.stdout + r.stderr)
    assert "not reusable" in r.stdout
    assert "GATE_QUALIFIED" in r.stdout, (
        "the refusal must name the flag that moved, or the operator reading the array's log "
        "cannot tell this apart from a data-dependent refusal:\n" + r.stdout)


def test_a_gate_qualified_shard_is_reusable_without_any_of_that(tmp_path) -> None:
    r = _landed(tmp_path, _shard_doc(gate_qualified=True, gate_disqualified_reasons=[]))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reusable:" in r.stdout


@pytest.mark.parametrize("over, why", [
    ({"gate_disqualified_reasons": [ADAPTER_BLOCKER_REASON, "coverage 0.412 < --min-coverage 0.9"]},
     "a data-dependent reason rides along with the standing one"),
    ({"gate_disqualified_reasons": ["no step yielded a displacement"]},
     "the shard measured nothing"),
    ({"gate_disqualified_reasons": []},
     "gate_qualified is false and nothing says why"),
    ({"headline_valid": False}, "headline_valid is false"),
    ({"partial_measurement": True}, "a partial measurement"),
    ({"limit": 3}, "--limit truncated the episode list"),
    ({"max_frames": 180}, "--max-frames truncated every clip"),
    ({"n_steps_measured": 0}, "no steps were measured"),
    ({"schema": "wam.geom_tol/1"}, "a finished GEOM_TOL is not a shard"),
    ({"shard": {"index": 1, "num_shards": 16, "n_episodes_in_shard": 33}}, "a different index"),
    ({"shard": {"index": 0, "num_shards": 8, "n_episodes_in_shard": 33}}, "a different partition"),
    ({"step_frames": 2}, "a different step"),
])
def test_everything_else_is_still_refused(tmp_path, over, why) -> None:
    """The forgiveness is exactly one reason wide. Everything data-dependent still re-measures."""
    r = _landed(tmp_path, _shard_doc(**over))
    assert r.returncode == 1, f"{why} was accepted as landed:\n{r.stdout}{r.stderr}"
    assert "not reusable" in r.stdout


def test_a_truncated_shard_artifact_is_not_landed(tmp_path) -> None:
    """What a task killed at the wall leaves behind. It must never be mistaken for a finished one."""
    path = tmp_path / "shard-0.json"
    path.write_text('{"schema": "wam.geom_tol_shard/1", "shard": {"ind')
    r = _run_snippet(_heredoc_after("shard_artifact_landed () {"), [str(path), "0", "16", "1"], {})
    assert r.returncode == 1
    assert "does not parse" in r.stdout


# -- the walltime self-check -----------------------------------------------------------------------

SELFCHECK = "ALLOW_TIGHT=\"${GEOM_ALLOW_TIGHT_WALL:-0}\" python - <<'PY'"

#: A segmenter block in the shape apple_sam2.SEGMENTER_CONTRACT has. Only equality matters here.
CONTRACT_NOW = {"method_name": "grounding-dino+sam2+depth-anything-v2", "box_threshold": 0.15,
                "retry_box_threshold": 0.10, "pixel_grid_hw": [480, 640]}
CONTRACT_PILOT_ERA = {"method_name": "grounding-dino+sam2+depth-anything-v2", "box_threshold": 0.35,
                      "retry_box_threshold": None, "pixel_grid_hw": [480, 640]}


def _selfcheck(tmp_path, *, pilot: dict | None, contract: dict | None, num_shards: int = 16,
               index: int = 0, frames_per_episode: int = 400, n_episodes: int = 402,
               remaining: int | None = 5400, allow_tight: str = "0",
               p_fallback: str = "0.18") -> subprocess.CompletedProcess:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [
        {"id": f"episode_{i:06d}", "frames": frames_per_episode} for i in range(n_episodes)]}))
    pilot_path = tmp_path / "GEOM_TOL_PILOT.json"
    if pilot is not None:
        pilot_path.write_text(json.dumps(pilot))
    contract_path = tmp_path / "pr08_geom_tol.json"
    if contract is not None:
        contract_path.write_text(json.dumps({"segmenter": contract}))
    env = {
        "SHARD_INDEX": str(index), "NUM_SHARDS": str(num_shards),
        "MANIFEST": str(manifest), "PILOT_JSON": str(pilot_path),
        "CONTRACT_FILE": str(contract_path),
        "P_FALLBACK": p_fallback, "L_FALLBACK": "120", "ALLOW_TIGHT": allow_tight,
    }
    if remaining is None:
        env["SLURM_JOB_END_TIME"] = ""
    else:
        import time as _t
        env["SLURM_JOB_END_TIME"] = str(int(_t.time()) + remaining)
    return _run_snippet(_heredoc_after(SELFCHECK), [], env)


def _pilot(contract: dict | None, *, per_frame: float = 0.0833) -> dict:
    rec = {"seconds_per_frame": per_frame, "load_seconds": 120.0, "slurm_job_id": "189588"}
    if contract is not None:
        rec["mask_method"] = {"params": {"segmenter": contract}}
    return rec


def test_a_pilot_measured_at_a_replaced_operating_point_is_discredited(tmp_path) -> None:
    """THE DEFECT THIS CHECK EXISTS FOR, in the exact shape that cost job 189658 six GPU-hours.

    ``GEOM_TOL_PILOT.json`` on the cluster was produced at box_threshold 0.35 with no retry branch;
    apple_sam2 has since moved to 0.15 with a (0.10, 0.10) retry because PR-08 §4 step 2 requires
    our detection point to BE the generator's. Its slope is at least 1.99x optimistic. The check
    must not use it, and it must say why rather than quietly substituting a number.
    """
    r = _selfcheck(tmp_path, pilot=_pilot(CONTRACT_PILOT_ERA), contract=CONTRACT_NOW)
    assert r.returncode == 0, r.stdout
    assert "PILOT DISCREDITED" in r.stdout
    assert "box_threshold" in r.stdout and "retry_box_threshold" in r.stdout
    assert "p=0.1800" in r.stdout, f"the discredited slope was used anyway:\n{r.stdout}"


def test_a_pilot_that_matches_the_committed_contract_is_used(tmp_path) -> None:
    """The check is not "ignore the pilot" — a re-measured pilot at the current point is better
    evidence than a planning constant, and the committed contract is what tells the two apart."""
    r = _selfcheck(tmp_path, pilot=_pilot(CONTRACT_NOW, per_frame=0.19), contract=CONTRACT_NOW)
    assert r.returncode == 0, r.stdout
    assert "PILOT DISCREDITED" not in r.stdout
    assert "p=0.1900" in r.stdout
    assert "189588" in r.stdout


def test_a_pilot_with_no_segmenter_block_states_nothing_and_is_not_trusted(tmp_path) -> None:
    r = _selfcheck(tmp_path, pilot=_pilot(None), contract=CONTRACT_NOW)
    assert r.returncode == 0, r.stdout
    assert "records no segmenter block" in r.stdout
    assert "p=0.1800" in r.stdout


def test_with_no_pilot_at_all_the_corrected_constant_is_used(tmp_path) -> None:
    r = _selfcheck(tmp_path, pilot=None, contract=CONTRACT_NOW)
    assert r.returncode == 0, r.stdout
    assert "no pilot artifact" in r.stdout
    assert "p=0.1800" in r.stdout


def test_the_estimate_is_this_shards_real_frames_and_not_an_even_split(tmp_path) -> None:
    """The old check divided the corpus by N and multiplied by a constant measured at a different N.

    Here every episode is the same length, so the partition's frame imbalance is exactly its episode
    imbalance and the expected number is arithmetic: the shard's own episode count times 400.
    """
    n_eps, per_ep, n = 402, 400, 16
    mine = sum(1 for i in range(n_eps) if mgt.shard_of(f"episode_{i:06d}", n) == 0)
    r = _selfcheck(tmp_path, pilot=None, contract=CONTRACT_NOW,
                   num_shards=n, index=0, frames_per_episode=per_ep, n_episodes=n_eps,
                   remaining=4 * 3600)
    assert r.returncode == 0, r.stdout
    assert f"{mine} episodes, {mine * per_ep} of {n_eps * per_ep} frames" in r.stdout, r.stdout
    # ... and it is NOT the even split, which is the number the old check printed.
    assert mine * per_ep != n_eps * per_ep // n


def test_a_shard_that_cannot_finish_refuses_to_start(tmp_path) -> None:
    """189658's shards ran for two hours and wrote nothing. Refusing costs zero and says why.

    The old code warned and ran anyway, on the argument that the estimate came from three of 402
    episodes. That argument is gone: the frame count is exact and p is either contract-checked or
    the measured floor.
    """
    r = _selfcheck(tmp_path, pilot=None, contract=CONTRACT_NOW, remaining=600)
    assert r.returncode == 1, r.stdout
    assert "REFUSING TO START" in r.stdout
    assert "GEOM_ALLOW_TIGHT_WALL=1" in r.stdout


def test_the_operator_can_override_the_refusal_deliberately(tmp_path) -> None:
    r = _selfcheck(tmp_path, pilot=None, contract=CONTRACT_NOW, remaining=600, allow_tight="1")
    assert r.returncode == 0, r.stdout
    assert "WARNING" in r.stdout


def test_a_comfortable_request_passes_quietly(tmp_path) -> None:
    r = _selfcheck(tmp_path, pilot=None, contract=CONTRACT_NOW, remaining=5400)
    assert r.returncode == 0, r.stdout
    assert "REFUSING" not in r.stdout
    assert "break-even p for this request" in r.stdout


def test_a_missing_manifest_says_so_instead_of_guessing(tmp_path) -> None:
    """No frame counts means no estimate. It must not fall back to an even split and it must not
    refuse the run either — the manifest is provenance, not an input to the measurement."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": []}))
    r = _run_snippet(_heredoc_after(SELFCHECK), [], {
        "SHARD_INDEX": "0", "NUM_SHARDS": "16", "MANIFEST": str(manifest),
        "PILOT_JSON": str(tmp_path / "nope.json"), "CONTRACT_FILE": str(tmp_path / "nope2.json"),
        "P_FALLBACK": "0.18", "L_FALLBACK": "120", "ALLOW_TIGHT": "0",
        "SLURM_JOB_END_TIME": "0",
    })
    assert r.returncode == 0, r.stdout
    assert "NO SELF-CHECK" in r.stdout


def test_the_shard_rule_in_the_sbatch_is_the_rule_measure_geom_tol_uses() -> None:
    """Two spellings of the partition would size the wrong shard. Run the sbatch's copy and compare.

    It is re-derived rather than imported because the check runs before the interpreter that owns
    ``shard_of`` is started — so the guarantee has to come from a test, not from an import.
    """
    src = _heredoc_after(SELFCHECK)
    body = src[src.index("def shard_of("):]
    body = body[:body.index("\n\n")] if "\n\n" in body else body
    ns: dict = {}
    exec("import hashlib\n" + body, ns)
    keys = [f"episode_{i:06d}" for i in range(402)]
    for n in (4, 8, 16):
        assert [ns["shard_of"](k, n) for k in keys] == [mgt.shard_of(k, n) for k in keys], (
            f"the sbatch and measure_geom_tol.py disagree about the partition at N={n}")


# -- the two directives job 189658 got wrong --------------------------------------------------------


def test_the_array_log_path_is_per_task() -> None:
    """189658's four tasks all wrote to one geom-tol.189658.out: three headers lost, and the
    surviving one described a different shard than the progress lines under it."""
    text = SBATCH_103.read_text(encoding="utf-8")
    out = re.search(r"^#SBATCH -o (\S+)$", text, re.M)
    assert out is not None, "103 declares no -o path"
    assert "%A_%a" in out.group(1), f"log path {out.group(1)} is not per-array-task"
    assert "%j" not in out.group(1)


def test_the_request_stays_inside_the_fair_share_rate() -> None:
    """Billing/min = GPUs x 1.0 + MemGB x 0.25 + Threads x 0.035714, and 26 threads / 257 GB / 1 GPU
    is exactly the 66.18 fair-share rate. Above it the billing counter empties before the GPU-hours
    and the rest of the allocation is lost permanently (docs/discoverer.md §3)."""
    text = SBATCH_103.read_text(encoding="utf-8")
    cpus = int(re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M).group(1))
    mem = int(re.search(r"^#SBATCH --mem=(\d+)G$", text, re.M).group(1))
    assert cpus <= 26, f"{cpus} threads exceeds the fair-share rate for one GPU"
    assert mem <= 257, f"{mem} GB exceeds the fair-share rate for one GPU"
    rate = cpus * 0.035714286 + mem * 0.25 + 1.0
    assert rate <= 66.18
    # And it is right-sized, not merely legal: job 189588 peaked at MaxRSS 4.75 GiB on this exact
    # estimator stack, so 192 G was 40x the measured peak and five times the billing it needed.
    assert mem <= 64, (
        f"--mem={mem}G bills {rate:.2f} units/wall-hour against 9.93 at 32 G. The stack peaks at "
        "4.75 GiB (job 189588, cited in 106_measure_robot_mask_area.sbatch); memory is the single "
        "biggest lever on this allocation.")
