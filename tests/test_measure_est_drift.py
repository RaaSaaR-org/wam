"""PR-08 §4's calibration rig: the refusals, and the arithmetic G0b subtracts.

These run without Isaac and without an estimator — the capture half against ``FakeIsaacBinding``,
the measure half against a stub estimator module written per-test. That split is the point of the
two subcommands: the number itself needs a GPU and weights nobody has fetched, but every way of
getting it *wrong* is reachable here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import measure_est_drift as ed  # noqa: E402

from wam.robot.isaac_binding import FakeIsaacBinding  # noqa: E402


def _stub_module(tmp_path: pathlib.Path, name: str, body: str) -> str:
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    return name


@pytest.fixture()
def capture(tmp_path):
    out = tmp_path / "cap"
    binding = FakeIsaacBinding(cameras=("persp",), ground_truth=("depth", "segmentation"))
    ed.capture_frames(binding, "persp", 6, out, steps_per_frame=1)
    return out


# -- capture -------------------------------------------------------------------------------------


def test_capture_writes_all_three_channels_per_frame(capture):
    header = json.loads((capture / "capture.json").read_text())
    assert header["n_frames"] == 6
    for i in range(6):
        d = capture / "frames" / f"{i:06d}"
        assert np.load(d / "rgb.npy").ndim == 3
        assert np.load(d / "depth.npy").ndim == 2
        assert np.load(d / "seg_ids.npy").ndim == 2
        assert json.loads((d / "seg_labels.json").read_text()) != {}


def test_a_fake_capture_says_so_in_the_header(capture):
    """The whole rig is exercisable on a laptop, which is exactly why the artifact has to record
    that the ground truth was not ground truth."""
    assert json.loads((capture / "capture.json").read_text())["is_simulated_binding"] is True


def test_capture_refuses_a_binding_with_no_ground_truth_attached(tmp_path):
    binding = FakeIsaacBinding(cameras=("persp",))
    with pytest.raises(ed.EstimatorUnavailable, match="no 'depth' channel attached"):
        ed.capture_frames(binding, "persp", 2, tmp_path / "x", steps_per_frame=1)


def test_a_warmup_frame_is_not_written_as_a_partial_frame(tmp_path):
    """A frame whose depth is one tick and whose segmentation is another is worse than no frame."""
    binding = FakeIsaacBinding(
        cameras=("persp",), ground_truth=("depth", "segmentation"), warmup_frames=3
    )
    header = ed.capture_frames(binding, "persp", 4, tmp_path / "w", steps_per_frame=1)
    assert header["n_frames"] == 4
    assert header["warmup_returns"] >= 1


# -- the refusals --------------------------------------------------------------------------------


def test_auto_names_every_estimator_it_looked_for_and_writes_nothing(capsys, capture, tmp_path):
    out = tmp_path / "should_not_exist.json"
    assert ed.main(["measure", "--capture", str(capture), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "no gate-qualified object segmenter is wired" in err
    assert "no gate-qualified monocular depth estimator is wired" in err
    for name, _ in ed.CANDIDATE_SEGMENTERS:
        assert name in err
    # §4 step 2 says "the SAME segmenter", so wiring one closes both halves of §8 item 4 at once.
    assert "one decision for both measurements" in err
    assert not out.exists()


def test_an_estimator_module_must_define_both_halves(tmp_path):
    name = _stub_module(tmp_path, "half_estimator", "def segment(rgb):\n    return rgb\n")
    with pytest.raises(ed.EstimatorUnavailable, match="estimate_depth"):
        ed.resolve_estimators(name)


def test_gate_qualification_is_opt_in_not_assumed(tmp_path):
    """A stub that forgot to say it is not a gate must not become the gate by omission."""
    name = _stub_module(
        tmp_path,
        "silent_estimator",
        "def segment(rgb):\n    return rgb\n\n\ndef estimate_depth(rgb):\n    return rgb\n",
    )
    assert ed.resolve_estimators(name).gate_qualified is False


# -- the arithmetic ------------------------------------------------------------------------------


def test_an_unmeasurable_frame_is_dropped_and_not_folded_in_as_zero():
    """Zeros would pull the p95 down, which WIDENS G0b — conservative-looking and backwards."""
    pairs = [((0.0, 0.0), (3.0, 4.0)), (None, (1.0, 1.0)), ((2.0, 2.0), None)]
    values, dropped = ed.paired_displacements(pairs)
    assert dropped == 2
    assert values.tolist() == [5.0]


def test_depth_error_excludes_the_infinite_background_and_counts_it():
    """distance_to_camera reports a ray that hit nothing as inf and the binding passes it through,
    so including it would make the mean a function of how much sky is in frame."""
    true = np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    est = np.array([[1.5, 0.0], [2.0, 3.5]], dtype=np.float32)
    stats = ed.depth_error(est, true, mask=np.ones_like(true, dtype=bool))
    assert stats["n"] == 3
    assert stats["n_non_finite_px"] == 1
    assert stats["mean_m"] == pytest.approx((0.5 + 0.0 + 0.5) / 3)


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"class": "apple"}, "apple"),
        ("apple", "apple"),
        ({"semanticLabel": "apple"}, "apple"),
        ({"unrecognised": "apple"}, None),
        (42, None),
    ],
)
def test_label_text_refuses_to_guess_at_an_unrecognised_shape(entry, expected):
    """Replicator's idToLabels shape is UNVERIFIED. Guessing inside it is how the rig ends up
    tracking the plate and still producing a number."""
    assert ed.label_text(entry) == expected


def test_object_ids_reports_the_vocabulary_it_saw():
    """'the apple is absent' and 'the apple is called something else' look identical to the caller
    and are fixed differently."""
    matched, seen = ed.object_ids({1: {"class": "plate"}, 2: {"class": "Apple"}}, "apple")
    assert matched == [2]
    assert seen == ["Apple", "plate"]


def test_mask_from_ids_selects_every_matching_id():
    ids = np.array([[1, 2], [3, 2]], dtype=np.uint32)
    assert ed.mask_from_ids(ids, [2]).tolist() == [[False, True], [False, True]]
    assert ed.mask_from_ids(ids, []).any() is np.False_


# -- the end-to-end measure path, with a stub in the estimator's place ---------------------------


def _naive_estimator(tmp_path) -> str:
    """A red-channel threshold — deliberately NOT the true mask.

    It stands in for a real segmenter only in shape: it returns a plausible binary mask of the
    right size on every frame, so the pipeline runs end to end and produces a non-zero drift. The
    number it yields is meaningless, which is the point of every assertion below being about
    disqualification rather than about the value.
    """
    return _stub_module(
        tmp_path,
        "naive_estimator",
        "import numpy as np\n"
        "ESTIMATOR_NAME = 'naive-red-threshold-stub'\n"
        "ESTIMATOR_VERSION = '0'\n"
        "\n"
        "def segment(rgb):\n"
        "    return np.asarray(rgb)[:, :, 0] > 0\n"
        "\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype='float32')\n",
    )


def test_a_stub_runs_the_whole_path_and_is_refused_as_a_gate(capture, tmp_path):
    out = tmp_path / "est_drift.json"
    code = ed.main(
        [
            "measure",
            "--capture", str(capture),
            "--estimators", _naive_estimator(tmp_path),
            "--object-class", "apple",
            "--min-coverage", "0.0",
            "--out", str(out),
        ]
    )
    assert code == 3, "a stub estimator over a fake capture must never exit 0"
    doc = json.loads(out.read_text())
    assert doc["schema"] == ed.SCHEMA
    assert doc["gate_qualified"] is False
    assert "estimator_not_gate_qualified" in doc["gate_disqualified_reasons"]
    assert "capture_is_not_from_isaac_sim" in doc["gate_disqualified_reasons"]
    # Unconditional, and not a flag: Isaac frames are not RealSense frames (PR-08 §4).
    assert doc["is_lower_bound"] is True


def test_the_artifact_is_written_even_when_disqualified(capture, tmp_path):
    """PR-08 §6 records the number regardless of verdict — 'we tried and this is what came out'."""
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
         "--min-coverage", "0.0", "--out", str(out)]
    )
    assert out.is_file()
    assert json.loads(out.read_text())["est_drift_p95_px"] is not None


def test_a_partial_run_can_never_be_the_gate(capture, tmp_path):
    """--limit is exactly the shape of a silent corruption: coverage is a fraction of what was
    actually decoded, so a 2-frame run over a 6-frame capture reports a perfect score."""
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
         "--limit", "2", "--min-coverage", "0.0", "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    assert "partial_run_limit" in doc["gate_disqualified_reasons"]
    assert doc["n_frames"] == 2
    assert doc["n_frames_found"] == 6


def test_coverage_below_the_floor_disqualifies_without_hiding_the_number(capture, tmp_path):
    name = _stub_module(
        tmp_path,
        "blind_estimator",
        "import numpy as np\n"
        "def segment(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype=bool)\n"
        "\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype='float32')\n",
    )
    out = tmp_path / "d.json"
    assert ed.main(
        ["measure", "--capture", str(capture), "--estimators", name, "--out", str(out)]
    ) == 3
    doc = json.loads(out.read_text())
    assert "coverage_below_floor" in doc["gate_disqualified_reasons"]
    assert doc["headline_valid"] is False
    assert doc["n_dropped"] == doc["n_frames"]


def test_an_estimator_that_returns_a_differently_shaped_mask_is_fatal(capture, tmp_path, capsys):
    name = _stub_module(
        tmp_path,
        "wrong_grid_estimator",
        "import numpy as np\n"
        "def segment(rgb):\n"
        "    return np.ones((7, 9), dtype=bool)\n"
        "\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype='float32')\n",
    )
    out = tmp_path / "d.json"
    assert ed.main(
        ["measure", "--capture", str(capture), "--estimators", name, "--out", str(out)]
    ) == 2
    assert "is not a" in capsys.readouterr().err
    assert not out.exists()


def test_the_geom_tol_cross_check_records_that_it_is_not_committed_yet(capture, tmp_path):
    """§6 computes GEOM_TOL - EST_DRIFT_P95. Nothing else in the pipeline checks that the two were
    measured on the same grid with the same segmenter, and a mismatch subtracts cleanly to a
    plausible wrong number."""
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
         "--min-coverage", "0.0", "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    assert "geom_tol_not_committed" in doc["gate_disqualified_reasons"]
    assert doc["geom_tol_cross_check"]["this_resolution_hw"] == [64, 64]


# -- the consumer half of the cross-check ---------------------------------------------------------
#
# measure_geom_tol.py names the join key in its module docstring: its `mask_method.name` must equal
# this artifact's `estimators.name`. Until these tests existed both artifacts RECORDED the two names
# and nothing compared them, so two different segmenters produced two plausible pixel numbers that
# subtracted cleanly to a plausible wrong tolerance. Found by the adversarial pass, twice, from both
# sides of the join.


@pytest.fixture()
def committed_geom_tol(tmp_path, monkeypatch):
    """Point the cross-check at a synthetic committed GEOM_TOL artifact."""

    def _write(**over):
        doc = {
            "resolution_hw": [64, 64],
            "gate_qualified": True,
            "mask_method": {"name": "grounding-dino+sam2+depth-anything-v2"},
        }
        doc.update(over)
        p = tmp_path / "pr08_geom_tol.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", p)
        return p

    return _write


def test_a_matching_artifact_raises_no_cross_check_reason(committed_geom_tol):
    committed_geom_tol()
    reasons, _ = ed.cross_check_geom_tol([64, 64], "grounding-dino+sam2+depth-anything-v2")
    assert reasons == []


def test_a_different_segmenter_disqualifies_rather_than_being_noted(committed_geom_tol):
    """§4 step 2 says 'the SAME segmenter'. Two segmenters is two different quantities and §6
    subtracts them."""
    committed_geom_tol()
    reasons, cmp = ed.cross_check_geom_tol([64, 64], "hsv-red-diagnostic")
    assert "mask_method_disagrees_with_estimator" in reasons
    assert cmp["geom_tol_mask_method_name"] == "grounding-dino+sam2+depth-anything-v2"
    assert cmp["this_estimator_name"] == "hsv-red-diagnostic"


@pytest.mark.parametrize("missing", ["resolution_hw", "gate_qualified", "mask_method"])
def test_absence_is_not_agreement(committed_geom_tol, missing):
    """The first version read `if theirs is not None and theirs != ours`, so an artifact that
    simply did not record its grid passed the grid check BY SAYING NOTHING. A missing field means
    the check could not be made, which is a reason to disqualify and not a reason to proceed."""
    committed_geom_tol(**{missing: None})
    reasons, _ = ed.cross_check_geom_tol([64, 64], "grounding-dino+sam2+depth-anything-v2")
    assert f"geom_tol_does_not_record_{missing}" in reasons


def test_a_disagreeing_grid_still_disqualifies(committed_geom_tol):
    committed_geom_tol(resolution_hw=[480, 640])
    reasons, _ = ed.cross_check_geom_tol([64, 64], "grounding-dino+sam2+depth-anything-v2")
    assert "resolution_disagrees_with_geom_tol" in reasons


def test_the_join_key_is_named_in_the_artifact_not_only_in_a_docstring(committed_geom_tol):
    """A reader acting on these artifacts reads the JSON, not the module docstring."""
    committed_geom_tol()
    _, cmp = ed.cross_check_geom_tol([64, 64], "grounding-dino+sam2+depth-anything-v2")
    assert "estimators.name" in cmp["join_key"]
    assert "mask_method.name" in cmp["join_key"]
