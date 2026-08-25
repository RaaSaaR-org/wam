"""PR-08 §4's calibration rig: the refusals, and the arithmetic G0b subtracts.

These run without Isaac and without an estimator — the capture half against ``FakeIsaacBinding``,
the measure half against a stub estimator module written per-test. That split is the point of the
two subcommands: the number itself needs a GPU and weights nobody has fetched, but every way of
getting it *wrong* is reachable here.
"""

from __future__ import annotations

import importlib.util
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


def test_the_geom_tol_cross_check_refuses_a_stub_against_the_really_committed_contract(
    capture, tmp_path
):
    """§6 computes GEOM_TOL - EST_DRIFT_P95. Nothing else in the pipeline checks that the two were
    measured on the same grid with the same segmenter, and a mismatch subtracts cleanly to a
    plausible wrong number.

    This runs against the REAL ``configs/transfer25/pr08_geom_tol.json`` — the segmenter contract
    committed 2026-08-22, before either number was measured — rather than a fixture, which is the
    only way to catch the contract and the reader drifting apart. Its earlier form asserted
    ``geom_tol_not_committed``; that file now exists, so the assertion that means the same thing is
    that a 64x64 fake capture segmented by a red-channel stub is refused for every reason it should
    be: wrong grid, wrong (unnamed) segmenter, and a GEOM_TOL that is not measured yet.
    """
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
         "--min-coverage", "0.0", "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    reasons = doc["gate_disqualified_reasons"]
    assert "resolution_disagrees_with_geom_tol" in reasons
    assert "mask_method_disagrees_with_estimator" in reasons
    assert "geom_tol_is_not_gate_qualified" in reasons
    assert "estimator_does_not_declare_segmenter_contract" in reasons
    assert doc["geom_tol_cross_check"]["this_resolution_hw"] == [64, 64]
    # The committed grid is the corpus's 640x480, read out of the contract's pixel_grid_hw because
    # the pre-measurement shape has no resolution_hw of its own to read.
    assert doc["geom_tol_cross_check"]["geom_tol_resolution_hw"] == [480, 640]


def test_an_absent_geom_tol_artifact_is_still_reported_as_absent(capture, tmp_path, monkeypatch):
    """The committed contract exists today; it did not always, and a reader that only handles the
    file being there would report a clean cross-check on a machine where it is missing."""
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", tmp_path / "nothing_here.json")
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


# -- capture's command line, which is what makes a capture a gate input at all ---------------------
#
# Until 2026-08-22 `capture` declared --out/--camera/--frames/--steps-per-frame/--fake and nothing
# else, and each of the three gaps cost something different: the render grid could not be set and
# took a constructor default that disagreed with the committed contract (so EVERY Isaac capture was
# disqualified, at any resolution), no flag named a stage (so only the bare g1.usd — which has no
# apple in it — could be captured), and --camera DEFAULTED to a name no Isaac stage carries, which
# raises after a full Isaac boot. These pin the fixes, one refusal at a time.


@pytest.fixture()
def contract(tmp_path, monkeypatch):
    """A committed GEOM_TOL contract at a grid that is deliberately NOT the real [480, 640].

    The point of the default is that it is READ from the committed document. A test written against
    480x640 would pass just as well if the number had been hard-coded into measure_est_drift.py
    beside the one in the contract, which is the second copy this arrangement exists to prevent.
    """

    def _write(grid=(12, 20), **over):
        doc = {"segmenter": {"method_name": "stub", "pixel_grid_hw": list(grid)}, **over}
        p = tmp_path / "pr08_geom_tol.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", p)
        return p

    return _write


def _capture_argv(tmp_path, *extra):
    return ["capture", "--fake", "--out", str(tmp_path / "cap"), "--frames", "1", *extra]


def test_the_render_grid_defaults_to_the_committed_contract_and_not_to_a_literal(
    contract, tmp_path
):
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path)) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["resolution_hw"] == [12, 20], "the frames themselves are on the committed grid"
    assert header["render_hw_requested"] == [12, 20]
    assert "pixel_grid_hw" in header["render_hw_source"]


def test_a_render_grid_that_disagrees_with_the_contract_refuses_before_anything_renders(
    contract, tmp_path, capsys
):
    """§6 subtracts GEOM_TOL and EST_DRIFT_P95, which is arithmetic on one grid only. The capture
    this would produce could never be a gate input, and finding that out from `measure` — after the
    render, on another machine — is the shape this refusal exists to end."""
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path, "--render-hw", "256", "256")) == 2
    err = capsys.readouterr().err
    assert "resolution_disagrees_with_geom_tol" in err
    assert "12x20" in err
    assert not (tmp_path / "cap").exists(), "nothing may be rendered when the grid is wrong"


def test_a_render_grid_equal_to_the_contract_is_accepted_and_says_where_it_came_from(
    contract, tmp_path
):
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path, "--render-hw", "12", "20")) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["render_hw_source"].startswith("--render-hw")


def test_no_committed_contract_means_no_default_grid_rather_than_a_guessed_one(
    tmp_path, monkeypatch, capsys
):
    """The contract is the pre-commitment 'the same segmenter' is checked against. A capture
    rendered at a grid nobody committed to is not a gate input at any resolution, so the missing
    document is a refusal that names git rather than a number picked here."""
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", tmp_path / "nothing_here.json")
    assert ed.main(_capture_argv(tmp_path)) == 2
    assert "states no pixel grid" in capsys.readouterr().err
    assert not (tmp_path / "cap").exists()


def test_the_default_camera_is_one_the_binding_actually_has(contract, tmp_path):
    """It defaulted to 'ego' while DEFAULT_CAMERA_PRIMS is {'persp': ...}, so the DEFAULT VALUE
    raised `unknown camera 'ego'` — after SimulationApp had started, the stage had loaded and 43
    DOFs had resolved. The default now comes out of that dict rather than being typed twice."""
    contract()
    assert ed.main(_capture_argv(tmp_path)) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["camera"] in ed.DEFAULT_CAMERA_PRIMS
    assert header["camera_prim"] == ed.DEFAULT_CAMERA_PRIMS[header["camera"]]


def test_an_unknown_camera_is_refused_before_the_binding_is_constructed(contract, tmp_path, capsys):
    """WITHOUT --fake, so the next statement after this check is the Isaac boot. The refusal must
    be the camera's and not `Isaac Sim is not importable`: a typo may not cost a GPU boot."""
    contract()
    argv = ["capture", "--out", str(tmp_path / "cap"), "--frames", "1", "--camera", "ego"]
    assert ed.main(argv) == 2
    assert "unknown camera 'ego'" in capsys.readouterr().err
    assert not (tmp_path / "cap").exists()


def test_a_stage_camera_can_be_named_and_is_recorded_with_its_prim(contract, tmp_path):
    """A scene authored for §4 step 1 carries an ego-like camera prim. Validating --camera against
    a fixed dict would otherwise make that scene unusable, so the dict is extended by the operator
    and the check stays exact."""
    contract()
    assert ed.main(_capture_argv(
        tmp_path, "--camera", "ego", "--camera-prim", "ego=/World/Scene/EgoCam")) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["camera"] == "ego"
    assert header["camera_prim"] == "/World/Scene/EgoCam"
    assert header["camera_prims_declared"] == {"ego": "/World/Scene/EgoCam"}


@pytest.mark.parametrize("bad", ["ego", "ego=World/EgoCam", "=/World/EgoCam"])
def test_a_malformed_camera_prim_is_refused(contract, tmp_path, capsys, bad):
    contract()
    assert ed.main(_capture_argv(tmp_path, "--camera-prim", bad)) == 2
    assert "not NAME=/Prim/Path" in capsys.readouterr().err


def test_the_stage_is_named_once_and_recorded(contract, tmp_path):
    """--asset and --scene are one knob, spelled two ways because configs/robot/isaac_g1.yaml and
    IsaacG1Robot already spell it both ways. What matters downstream is that the capture says which
    stage it was the ground truth OF: a p95 over the bare g1.usd and one over an apple-to-plate
    scene are otherwise indistinguishable in the artifact."""
    contract()
    assert ed.main(_capture_argv(tmp_path, "--scene", "/abs/apple_to_plate.usd")) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["asset"] == "/abs/apple_to_plate.usd"
    assert header["asset_source"] == "--scene"


def test_naming_the_stage_twice_is_refused(contract, tmp_path, capsys):
    contract()
    assert ed.main(_capture_argv(tmp_path, "--scene", "/a.usd", "--asset", "/b.usd")) == 2
    assert "same knob" in capsys.readouterr().err


def test_a_capture_with_no_stage_says_so_rather_than_saying_nothing(contract, tmp_path):
    """`asset: null` is the honest record of "Isaac's own default asset root", which is the bare
    G1 — no table, no plate, no apple. A reader of the p95 needs to be able to tell that case from
    a scene that was there."""
    contract()
    assert ed.main(_capture_argv(tmp_path)) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["asset"] is None
    assert "get_assets_root_path" in header["asset_source"]


def test_the_measured_artifact_carries_the_scene_the_capture_named(contract, tmp_path):
    """The provenance has to reach the file the gate reads. §4 step 1 renders "N Isaac episodes";
    an EST_DRIFT_P95 that does not say what was in the scene cannot be audited afterwards."""
    contract()
    assert ed.main(_capture_argv(tmp_path, "--asset", "/abs/apple_to_plate.usd")) == 0
    out = tmp_path / "d.json"
    ed.main(["measure", "--capture", str(tmp_path / "cap"), "--estimators",
             _naive_estimator(tmp_path), "--min-coverage", "0.0", "--out", str(out)])
    block = json.loads(out.read_text())["capture"]
    assert block["asset"] == "/abs/apple_to_plate.usd"
    assert block["asset_source"] == "--asset"
    assert block["camera_prim"] == ed.DEFAULT_CAMERA_PRIMS["persp"]
    assert block["render_hw_requested"] == [12, 20]


# -- what the estimator saw, recorded beside the budget it produced ------------------------------
#
# EST_DRIFT_P95 is measured with the same adapter GEOM_TOL is, and that adapter's second
# gate-qualification blocker asks for its retry counts and its detection-score distribution from a
# real pass. Until 2026-08-22 this harness recorded the estimator's NAME, its version and its gate
# flag and threw the rest away. These tests are about the block that now carries it — and about the
# fact that an estimator which exports none of it must still measure exactly as it did before.


def _counting_estimator(tmp_path) -> str:
    """A stub that keeps the counters and the score list ``estimators.apple_sam2`` keeps.

    Cumulative, because the real one is: nothing in it resets them, so a harness that copied
    ``stats()`` straight into its artifact would report a total shared with every other run in the
    same interpreter.
    """
    return _stub_module(
        tmp_path,
        "counting_estimator",
        "import numpy as np\n"
        "ESTIMATOR_NAME = 'counting-stub'\n"
        "ESTIMATOR_VERSION = '0'\n"
        "SEGMENT_CALLS = 0\n"
        "NO_DETECTION_FRAMES = 0\n"
        "RETRY_FRAMES = 0\n"
        "RETRY_RECOVERED_FRAMES = 0\n"
        "DETECTION_SCORES = []\n"
        "\n"
        "def segment(rgb):\n"
        "    global SEGMENT_CALLS\n"
        "    SEGMENT_CALLS += 1\n"
        "    DETECTION_SCORES.append(round(0.1 + 0.01 * (SEGMENT_CALLS % 50), 6))\n"
        "    return np.asarray(rgb)[:, :, 0] > 0\n"
        "\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype='float32')\n"
        "\n"
        "def stats():\n"
        "    return {\n"
        "        'estimator_name': ESTIMATOR_NAME,\n"
        "        'box_threshold': 0.15,\n"
        "        'n_segment_calls': SEGMENT_CALLS,\n"
        "        'n_frames_without_detection': NO_DETECTION_FRAMES,\n"
        "        'n_frames_retry_fired': RETRY_FRAMES,\n"
        "        'n_frames_retry_recovered': RETRY_RECOVERED_FRAMES,\n"
        "        'n_detection_scores': len(DETECTION_SCORES),\n"
        "    }\n",
    )


def test_the_budget_artifact_records_what_the_estimator_saw(capture, tmp_path):
    """The counts and the scores of THIS run, beside the p95 they were produced with."""
    out = tmp_path / "d.json"
    ed.main(["measure", "--capture", str(capture), "--estimators", _counting_estimator(tmp_path),
             "--object-class", "apple", "--min-coverage", "0.0", "--out", str(out)])
    doc = json.loads(out.read_text())

    stats = doc["estimator_stats"]
    assert stats["recorded"] is True
    assert stats["this_run"]["n_segment_calls"] == doc["n_frames"] - (
        doc["n_frames_without_object_label"])
    assert stats["detection_scores"]["n"] == stats["this_run"]["n_detection_scores"]
    assert stats["detection_scores"]["distribution"]["n"] == stats["detection_scores"]["n"]
    assert stats["detection_scores"]["distribution"]["box_threshold"] == 0.15
    # The raw values are kept here — a capture is a few hundred frames, not the 171 600 of a
    # GEOM_TOL pass — so the distribution beside them is re-derivable rather than merely quoted.
    assert stats["detection_scores"]["values"] == pytest.approx(
        [round(0.1 + 0.01 * ((i + 1) % 50), 6) for i in range(stats["detection_scores"]["n"])])
    assert stats["adapter"]["estimator_name"] == "counting-stub"
    # Recording evidence is not accepting it: nothing here reaches the verdict.
    assert "estimator_stats" not in " ".join(doc["gate_disqualified_reasons"])


def test_the_recorded_counts_are_this_runs_and_not_the_interpreters(capture, tmp_path):
    """Two measurements, one interpreter, one imported estimator. The second one's counts are its
    own — the adapter's totals are cumulative and are snapshotted and differenced, not copied."""
    name = _counting_estimator(tmp_path)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        ed.main(["measure", "--capture", str(capture), "--estimators", name,
                 "--object-class", "apple", "--min-coverage", "0.0", "--out", str(out)])
    a = json.loads(first.read_text())["estimator_stats"]
    b = json.loads(second.read_text())["estimator_stats"]

    assert b["this_run"] == a["this_run"], "the second run recorded the interpreter's total"
    assert b["counters_at_start_of_run"]["n_segment_calls"] == (
        a["counters_at_end_of_run"]["n_segment_calls"])
    # Stated as arithmetic rather than as absolute numbers: this module stays imported for the
    # whole session, so what the counters START at depends on which tests ran before this one —
    # which is the leak, reproduced. What must hold is that the difference is this run's.
    assert b["counters_at_start_of_run"]["n_segment_calls"] > 0
    assert (b["counters_at_end_of_run"]["n_segment_calls"]
            - b["counters_at_start_of_run"]["n_segment_calls"]) == b["this_run"]["n_segment_calls"]
    assert b["counters_at_end_of_run"]["n_segment_calls"] > b["this_run"]["n_segment_calls"]


def test_an_estimator_with_no_stats_records_an_absence_and_measures_the_same(capture, tmp_path):
    """The contract is segment(rgb)/estimate_depth(rgb). stats() is an extra, and "we did not look"
    must not be written down as "the retry never fired"."""
    out = tmp_path / "d.json"
    ed.main(["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
             "--object-class", "apple", "--min-coverage", "0.0", "--out", str(out)])
    doc = json.loads(out.read_text())

    stats = doc["estimator_stats"]
    assert stats["recorded"] is False
    assert "stats()" in stats["absent_because"]
    assert stats["this_run"] is None
    assert stats["detection_scores"]["n"] is None
    assert doc["est_drift_p95_px"] is not None


def _previous_version(tmp_path, name: str, commit: str = "d9ac5d1"):
    """``scripts/<name>.py`` as it was at ``commit``, importable under its own module name.

    One substitution, ``_REPO_ROOT``: the copy lives under tmp_path and would otherwise resolve the
    committed GEOM_TOL contract, the git commit and the repository itself somewhere else, so the
    comparison below would be against differences that have nothing to do with the change. Every
    other byte is the previous version's.
    """
    import importlib.util
    import subprocess

    repo = pathlib.Path(ed.__file__).resolve().parents[1]
    src = subprocess.run(["git", "show", f"{commit}:scripts/{name}.py"], cwd=str(repo),
                         capture_output=True, text=True, check=True).stdout
    anchor = "_REPO_ROOT = Path(__file__).resolve().parent.parent"
    assert src.count(anchor) == 1
    src = src.replace(anchor, f"_REPO_ROOT = Path({str(repo)!r})")
    path = tmp_path / f"{name}_at_{commit}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"{name}_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recording_the_estimator_stats_changed_no_number_this_script_already_produced(
        capture, tmp_path):
    """THE ADDITIVE CLAIM, checked against the script as it was rather than asserted.

    The previous version of measure_est_drift and this one, over the same capture with the same
    estimator. The only permitted difference is the new key and the timestamp — not the p95, not
    the coverage, not the disqualification reasons, not the cross-check.
    """
    old = _previous_version(tmp_path, "measure_est_drift")
    name = _counting_estimator(tmp_path)
    argv = ["measure", "--capture", str(capture), "--estimators", name,
            "--object-class", "apple", "--min-coverage", "0.0", "--out"]
    old.main([*argv, str(tmp_path / "before.json")])
    ed.main([*argv, str(tmp_path / "after.json")])
    before = json.loads((tmp_path / "before.json").read_text())
    after = json.loads((tmp_path / "after.json").read_text())

    # `estimator_stats` is the addition this test was written for. PR-08-V5 (2026-08-22) added
    # three more and they are additive in the SAME sense: they record which ground-truth route a
    # capture came from and which way that route's error points. Over a FakeIsaacBinding capture
    # — which is what `capture` is — there is no route, the pair below falls back to the old
    # unconditional stamp verbatim, and the `differing == []` assertion is what proves it.
    # `independent_samples` (2026-08-23) is additive in the same sense again: it records how many
    # scene configurations the measured frames came from, and it is read by nothing. Over a
    # FakeIsaacBinding capture the header carries no `steps_per_state`, so it records an absence
    # with a reason — which is itself the check that it invents no grouping.
    # `mask_vs_ground_truth_iou` (2026-08-25) is additive in the same sense once more: it is the
    # per-frame overlap between the estimated mask and the renderer's exact one, recorded beside
    # the centroid displacement because a displacement cannot see a plausible mask on the wrong
    # object. Nothing reads it, no disqualification reason depends on it, and the block says in
    # its own text that it discharges nothing.
    assert set(after) - set(before) == {
        "estimator_stats",
        "error_direction",
        "error_direction_measured",
        "ground_truth_route",
        "independent_samples",
        "mask_vs_ground_truth_iou",
    }
    assert after["independent_samples"]["recorded"] is False
    assert set(before) - set(after) == set()
    assert after["is_lower_bound"] is before["is_lower_bound"] is True
    assert after["is_lower_bound_reason"] == before["is_lower_bound_reason"]
    differing = sorted(k for k in before if k != "measured_utc" and before[k] != after[k])
    # The `capture` block gained five keys under PR-08-V5 — which backend rendered it, which
    # ground-truth route that is, and the object-substitution limitations a budget measured on a
    # stand-in object must not be readable without. Over a fake capture every one of them is
    # null and every key that was already there is unchanged, which is the whole claim.
    assert differing == ["capture"], f"recording the estimator's stats changed {differing}"
    added = set(after["capture"]) - set(before["capture"])
    assert added == {
        "backend",
        "ground_truth_route",
        "object_limitations",
        "n_scene_states_scheduled",
        "n_scene_states_visited",
        "scene_schedule",
        "scene_schedule_source",
        "temporal_coherence",
    }
    # Every one of them null over a fake capture EXCEPT the coherence block, which is written by
    # `capture_frames` for every binding and which — over this capture, made with no object class
    # — records an absence with a reason rather than the zero that would read as "still life".
    assert all(after["capture"][k] is None for k in added - {"temporal_coherence"})
    assert after["capture"]["temporal_coherence"]["measured"] is False
    assert after["capture"]["temporal_coherence"]["object_moved_during_capture"] is None
    assert {k: after["capture"][k] for k in before["capture"]} == before["capture"]


# -- PR-08-V5: the second ground-truth route, and what widening the allow-list may not do --------
#
# `T40_RULE_V5` (docs/preregistration/PR-08-V5-ground-truth-route.md, registered 2026-08-22 before
# any capture was run) generalises §4's ground-truth source from Isaac specifically to "a simulator
# with ground-truth segmentation". Two things in this file move with it and nothing else does: the
# `capture_is_not_from_isaac_sim` check becomes an ALLOW-LIST, and the `is_lower_bound` stamp
# becomes per-route. Both are the kind of edit that could quietly let a laptop capture become G0b's
# budget, so every test below is about what must still be refused.


class _NotAGroundTruthBinding(FakeIsaacBinding):
    """A binding whose class name is not in the allow-list. Same behaviour, different name."""


class MuJoCoGroundTruthBinding(FakeIsaacBinding):  # noqa: N801 - the NAME is the thing under test
    """A stand-in whose CLASS NAME is the one the allow-list carries.

    Deliberately a rename of the fake and nothing more: the allow-list is a lookup on
    ``type(binding).__name__``, so this is exactly what it sees, and testing it this way needs no
    GL context, no mesh and no MuJoCo. The real binding's own behaviour is
    ``tests/test_mujoco_binding.py``'s subject.
    """


def test_the_fake_is_not_in_the_allow_list_and_must_not_be(tmp_path):
    """Every capture anyone has run so far came from FakeIsaacBinding, whose "ground truth" is a
    moving square. It is the thing `capture_is_not_from_isaac_sim` exists to catch."""
    assert "FakeIsaacBinding" not in ed.GROUND_TRUTH_BINDINGS
    assert ed.ground_truth_route("FakeIsaacBinding") is None
    assert ed.ground_truth_route(None) is None
    assert ed.ground_truth_route("SomethingSomebodyWroteLastNight") is None


def test_an_unlisted_binding_is_still_stamped_simulated(tmp_path):
    binding = _NotAGroundTruthBinding(cameras=("persp",), ground_truth=("depth", "segmentation"))
    header = ed.capture_frames(binding, "persp", 1, tmp_path / "cap", steps_per_frame=1)
    assert header["is_simulated_binding"] is True
    assert header["ground_truth_route"] is None


def test_a_listed_binding_is_ground_truth_and_carries_its_route(tmp_path):
    binding = MuJoCoGroundTruthBinding(cameras=("persp",), ground_truth=("depth", "segmentation"))
    header = ed.capture_frames(binding, "persp", 1, tmp_path / "cap", steps_per_frame=1)
    assert header["is_simulated_binding"] is False
    assert header["ground_truth_route"] == "mujoco"


def _reroute(capture: pathlib.Path, binding_name: str) -> pathlib.Path:
    """Rewrite a capture header's binding name, leaving the frames alone.

    The frames are the fake's and stay the fake's — this exercises the ROUTING, which is a
    property of the header, and only the routing."""
    header = json.loads((capture / "capture.json").read_text())
    header["binding"] = binding_name
    route = ed.ground_truth_route(binding_name)
    header["is_simulated_binding"] = route is None
    header["ground_truth_route"] = (route or {}).get("route")
    (capture / "capture.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    return capture


def _measure(capture, tmp_path, name: str = "out"):
    out = tmp_path / f"{name}.json"
    ed.main(
        ["measure", "--capture", str(capture), "--estimators", _naive_estimator(tmp_path),
         "--object-class", "apple", "--min-coverage", "0.0", "--out", str(out)]
    )
    return json.loads(out.read_text())


def test_a_mujoco_capture_is_not_disqualified_for_not_being_isaac(capture, tmp_path):
    doc = _measure(_reroute(capture, "MuJoCoGroundTruthBinding"), tmp_path)
    assert "capture_is_not_from_isaac_sim" not in doc["gate_disqualified_reasons"]
    # And the reasons that have nothing to do with the route are untouched: widening the
    # ground-truth source does not gate-qualify an estimator nobody has looked at a mask from.
    assert "estimator_not_gate_qualified" in doc["gate_disqualified_reasons"]
    assert doc["gate_qualified"] is False


def test_the_mujoco_route_does_not_inherit_isaacs_lower_bound_sentence(capture, tmp_path):
    """§6 SUBTRACTS this number from GEOM_TOL, so which way its error points is the property that
    decides whether an error in the budget lands in the generator's favour or against it. The
    Isaac sentence says "plausibly optimistic"; saying that about a route whose argued direction
    is the opposite would be worse than saying nothing."""
    doc = _measure(_reroute(capture, "MuJoCoGroundTruthBinding"), tmp_path)
    assert doc["is_lower_bound"] is False
    assert "Isaac renders" not in doc["is_lower_bound_reason"]
    assert "T40_RULE_V5" in doc["is_lower_bound_reason"]
    assert doc["ground_truth_route"] == "mujoco"


def test_the_conservative_direction_is_recorded_as_argued_and_not_as_measured(capture, tmp_path):
    """`is_lower_bound: false` on its own reads as "so it is an upper bound", which is a claim no
    route here has earned. The direction is a separate field and it says who established it."""
    doc = _measure(_reroute(capture, "MuJoCoGroundTruthBinding"), tmp_path)
    assert "conservative" in doc["error_direction"]
    assert doc["error_direction_measured"] is False
    assert "ARGUMENT and not a measurement" in doc["is_lower_bound_reason"]


def test_the_isaac_route_is_stamped_exactly_as_it_was_before_v5(capture, tmp_path):
    """THE NON-REGRESSION. Whatever V5 did, an Isaac capture's two fields are the strings the
    unconditional stamp wrote, Humanoid-Everyday sentence and all — correcting that one is the
    runbook's §7 defect 4 and a judgement for whoever owns PR-08, not a side effect of this."""
    doc = _measure(_reroute(capture, "IsaacSimBinding"), tmp_path)
    assert doc["is_lower_bound"] is True
    assert doc["is_lower_bound_reason"] == (
        "measured on Isaac renders, not RealSense footage (PR-08 §4). The confirmatory "
        "measurement against Humanoid Everyday is blocked on that corpus's licence and is "
        "deliberately off the critical path."
    )
    assert doc["error_direction_measured"] is False


def test_a_capture_from_nothing_recognised_keeps_the_old_unconditional_stamp(capture, tmp_path):
    """The fallback is the old behaviour on purpose, so widening the route moved no string on the
    path that already existed. What such a capture gets instead is the new direction field and
    the disqualifier — neither of which existed to be changed."""
    doc = _measure(capture, tmp_path)
    assert doc["is_lower_bound"] is True
    assert "capture_is_not_from_isaac_sim" in doc["gate_disqualified_reasons"]
    assert doc["error_direction"] == "unknown — not a ground-truth capture"
    assert doc["ground_truth_route"] is None


def test_a_header_cannot_buy_a_route_it_did_not_come_from(capture, tmp_path):
    """`ground_truth_route` is resolved from the BINDING CLASS NAME, not from the header's own
    route string, so hand-editing the string does not make a fake capture ground truth."""
    header = json.loads((capture / "capture.json").read_text())
    header["ground_truth_route"] = "mujoco"
    header["is_simulated_binding"] = False
    (capture / "capture.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    doc = _measure(capture, tmp_path)
    assert "capture_is_not_from_isaac_sim" in doc["gate_disqualified_reasons"]
    assert doc["is_lower_bound"] is True


# -- the mujoco backend's own command-line refusals ----------------------------------------------


def test_the_backend_defaults_to_isaac_so_nothing_existing_moved(contract, tmp_path):
    """--backend is new, its default is the old behaviour, and the capture header proves it."""
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path)) == 0
    header = json.loads((tmp_path / "cap" / "capture.json").read_text())
    assert header["binding"] == "FakeIsaacBinding"
    assert header["camera"] == "persp"
    assert header["asset_source"].startswith("isaacsim.storage.native")


def test_the_mujoco_backend_refuses_the_isaac_fake(contract, tmp_path, capsys):
    """--fake is FakeIsaacBinding. There is nothing for it to stand in for here: the mujoco
    backend already runs on CPU with no install, which is why it was chosen."""
    contract(grid=(12, 20))
    argv = ["capture", "--backend", "mujoco", "--fake", "--out", str(tmp_path / "c"),
            "--frames", "1"]
    assert ed.main(argv) == 2
    assert "FakeIsaacBinding" in capsys.readouterr().err


def test_the_mujoco_backend_refuses_a_usd_prim_path(contract, tmp_path, capsys):
    """--camera-prim names a USD prim and an MJCF has no prims. Accepting it silently would
    record a camera_prim in the header that nothing ever read."""
    contract(grid=(12, 20))
    argv = ["capture", "--backend", "mujoco", "--out", str(tmp_path / "c"), "--frames", "1",
            "--camera-prim", "ego=/World/Scene/EgoCam"]
    assert ed.main(argv) == 2
    err = capsys.readouterr().err
    assert "MJCF" in err and "head" in err


def test_the_mujoco_backend_refuses_a_grid_that_disagrees_with_the_contract(
    contract, tmp_path, capsys
):
    """Same refusal, same reason, same place as the Isaac path: PR-08 §6 computes
    GEOM_TOL - EST_DRIFT_P95 and that is arithmetic on one pixel grid only. Refused before
    anything is compiled or rendered."""
    contract(grid=(12, 20))
    argv = ["capture", "--backend", "mujoco", "--out", str(tmp_path / "c"), "--frames", "1",
            "--render-hw", "480", "640"]
    assert ed.main(argv) == 2
    assert "resolution_disagrees_with_geom_tol" in capsys.readouterr().err


def test_the_mujoco_backend_refuses_when_no_contract_states_a_grid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ed, "GEOM_TOL_ARTIFACT", tmp_path / "absent.json")
    argv = ["capture", "--backend", "mujoco", "--out", str(tmp_path / "c"), "--frames", "1"]
    assert ed.main(argv) == 2
    assert "states no pixel grid" in capsys.readouterr().err


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="reaches MuJoCoGroundTruthBinding.__init__, which imports mujoco (optional 'sim' extra)",
)
def test_the_mujoco_backend_refuses_a_mesh_that_does_not_exist(contract, tmp_path, capsys):
    """It does not fall back to the scene's orange cube, and it does not fetch one."""
    contract(grid=(12, 20))
    argv = ["capture", "--backend", "mujoco", "--out", str(tmp_path / "c"), "--frames", "1",
            "--object-mesh", str(tmp_path / "no-such-apple.obj")]
    assert ed.main(argv) == 2
    assert "no-such-apple.obj" in capsys.readouterr().err


def test_the_mujoco_backend_refuses_zero_scene_states(contract, tmp_path, capsys):
    contract(grid=(12, 20))
    argv = ["capture", "--backend", "mujoco", "--out", str(tmp_path / "c"), "--frames", "1",
            "--scene-states", "0"]
    assert ed.main(argv) == 2
    assert "--scene-states" in capsys.readouterr().err


def test_the_mujoco_backends_default_stage_is_the_committed_scene():
    """Named once, and not a second literal: the Isaac answer is unchanged."""
    from wam.robot.mujoco_binding import DEFAULT_SCENE

    assert ed.resolve_stage(None, None, "mujoco") == (
        str(DEFAULT_SCENE), "wam.robot.mujoco_binding.DEFAULT_SCENE"
    )
    assert ed.resolve_stage(None, None) == (
        None, "isaacsim.storage.native.get_assets_root_path() + DEFAULT_ASSET_SUBPATH"
    )
    assert ed.resolve_stage(None, "/tmp/x.xml", "mujoco") == ("/tmp/x.xml", "--scene")


# -- the sample size behind the p95, which a frame count does not give -------------------------
#
# MEASURED 2026-08-23 on a real 20-state / 240-frame MuJoCo capture: the displacement spread
# INSIDE each scene state was 0.05-0.28 px while the spread BETWEEN states was 0.06-39.9 px, and
# the one state whose mask went to the left hand failed on all twelve of its frames. So
# `T40_RULE_V5` §5's two floors — >= 20 states AND >= 200 measured frames — were both met by a
# capture carrying 19 independent observations. This block records that; it gates nothing.


def _sched_header(steps_per_state: int, n_frames: int, **extra) -> dict:
    return {"steps_per_state": steps_per_state, "ticks": list(range(1, n_frames + 1)), **extra}


def test_frames_are_grouped_by_the_headers_own_ticks_and_not_by_division():
    """A run that ended early must not have its frames spread over states it never visited."""
    states, absent = ed.scene_state_per_frame(_sched_header(4, 10))
    assert absent is None
    assert states == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2], "ticks are 1-based; state 0 is ticks 1..4"


@pytest.mark.parametrize(
    "header",
    [
        {"ticks": [1, 2, 3]},                          # Isaac: no schedule at all
        {"steps_per_state": 4},                        # no ticks
        {"steps_per_state": 0, "ticks": [1, 2]},       # nonsense stride
        {"steps_per_state": "twelve", "ticks": [1]},   # nonsense type
    ],
)
def test_a_capture_that_cannot_be_grouped_records_an_absence_with_a_reason(header):
    """An absence with a reason, never a fabricated grouping — the module's rule everywhere."""
    states, absent = ed.scene_state_per_frame(header)
    assert states is None
    assert absent and isinstance(absent, str)
    block = ed.independent_sample_block(header, [0.1, 0.2])
    assert block["recorded"] is False
    assert block["absent_because"] == absent or block["absent_because"]


def test_the_block_counts_configurations_where_n_measured_counts_duplicates():
    """THE MEASUREMENT THIS BLOCK EXISTS FOR, as arithmetic rather than as a claim.

    Twenty configurations, twelve near-identical frames each, one configuration bad in all
    twelve. ``n_measured`` says 240 and ``T40_RULE_V5`` §5's second floor is met twelvefold; the
    number of independent observations is 20. Both are now in the artifact, so a reader is not
    asked to divide one by the other to find out.

    Note what is NOT asserted: that the two p95s differ. When every configuration carries the
    same number of frames and fails as a whole, they mostly agree — the duplication does not
    move the percentile, it moves how many observations the percentile is over, which is a
    property of the SAMPLE and not of the statistic. Both are recorded because a capture with
    uneven frames per configuration is the case where they do separate.
    """
    per_frame = [0.1] * 12 * 19 + [35.0] * 12
    block = ed.independent_sample_block(_sched_header(12, 240), per_frame)
    assert block["recorded"] is True
    assert block["n_measured_frames"] == 240
    assert block["n_scene_states_with_a_measured_frame"] == 20, "the real sample size"
    assert block["frames_per_scene_state"] == {"min": 12, "max": 12, "median": 12.0}
    # Zero spread inside a configuration, the full range between them: that ratio is what says
    # the twelve frames of a configuration are one observation and not twelve.
    assert block["within_state_displacement_spread_px"]["max"] == pytest.approx(0.0)
    medians = block["scene_state_median_displacement_px"]
    assert min(medians.values()) == pytest.approx(0.1)
    assert max(medians.values()) == pytest.approx(35.0)
    assert block["p95_over_frames_px"] is not None
    assert block["p95_over_scene_state_medians_px"] is not None


def test_dropped_frames_are_absent_from_the_block_rather_than_folded_in_as_zero():
    """Same rule as ``paired_displacements``: an unmeasurable frame is dropped and counted.

    Folding it in as 0 px would pull every statistic here down, which is the direction that
    widens G0b — the failure the whole harness is careful about.
    """
    per_frame = [None, None, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1]
    block = ed.independent_sample_block(_sched_header(4, 8), per_frame)
    assert block["n_measured_frames"] == 6
    assert block["measured_frames_per_scene_state"] == {"0": 2, "1": 4}
    assert block["scene_state_median_displacement_px"]["0"] == pytest.approx(5.0)
    assert 0.0 not in block["scene_state_median_displacement_px"].values()


def test_a_state_with_no_measurable_frame_at_all_does_not_appear_as_a_zero():
    per_frame = [None, None, 0.4, 0.6]
    block = ed.independent_sample_block(_sched_header(2, 4), per_frame)
    assert block["n_scene_states_with_a_measured_frame"] == 1
    assert block["measured_frames_per_scene_state"] == {"1": 2}


def test_a_header_and_a_frame_list_that_disagree_refuse_to_group():
    """--limit shortens the frame list while the header still lists every tick."""
    block = ed.independent_sample_block(_sched_header(12, 240), [0.1] * 24)
    assert block["recorded"] is False
    assert "240" in block["absent_because"] and "24" in block["absent_because"]


def test_the_block_is_additive_and_moves_no_gate(capture, tmp_path):
    """It appears in the artifact, and it changes no number and no reason that was there before."""
    est = _stub_module(
        tmp_path,
        "stub_independent",
        "import numpy as np\n"
        "OBJECT_TEXT_PROMPT = 'apple.'\n"
        "def segment(rgb):\n"
        "    m = np.zeros(rgb.shape[:2], bool); m[1:, 1:] = True; return m\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(rgb.shape[:2], np.float32)\n",
    )
    out = tmp_path / "drift.json"
    rc = ed.main(["measure", "--capture", str(capture), "--estimators", est, "--out", str(out)])
    doc = json.loads(out.read_text())
    assert "independent_samples" in doc
    assert "independent_samples" not in str(doc["gate_disqualified_reasons"])
    # The gated number is still the frame-wise percentile of centroid_displacement, untouched.
    assert doc["est_drift_p95_px"] == doc["centroid_displacement"]["percentiles_px"]["p95"]
    assert rc in (0, 3)


# -- what makes a propagation arm POSSIBLE, and the two fields that keep it honest ---------------
#
# `apple_sam2`'s third gate-qualification blocker names its own discharge condition: measure "the
# same capture BOTH ways - this adapter per frame, and the video predictor propagating from frame
# 0 - and recording the two p95s". Against the committed lattice that experiment is not merely
# unrun, it is UNRUNNABLE: consecutive frames teleport the object, so propagating from frame 0
# crosses a jump cut and measures the cut. `--schedule trajectory` renders a temporally coherent
# capture instead.
#
# NOTHING BELOW DISCHARGES ANYTHING. The propagation arm is out of scope here and the capture is a
# SIMULATOR capture; both facts are fields in the artifact, asserted as such.


class _MovingMaskBinding(FakeIsaacBinding):
    """A ground-truth binding whose object mask marches a fixed number of pixels per frame.

    The stock fake paints a square whose centre depends on ``sum(|q|)``, and with the default zero
    gains ``q`` never moves - so its capture is genuinely static, which is what the sibling test
    uses it for. This one is the other half: a capture that really is temporally coherent, so
    "the harness measured motion" and "the harness reported the number it was told" can be told
    apart.
    """

    #: px per frame. Small and constant on purpose: the assertion is on the MEASUREMENT, so a
    #: value the test can predict exactly is worth more than a plausible one.
    STEP_PX = 3

    def render_segmentation(self, camera):  # noqa: D102 - contract documented on the base class
        frame = super().render_segmentation(camera)
        if frame is None:
            return None
        height, width = frame.ids.shape
        ids = np.zeros_like(frame.ids)
        x = 8 + (self.get_physics_step_count() - 1) * self.STEP_PX
        ids[height // 4 : height // 4 + 6, x : x + 6] = 2
        return type(frame)(ids=ids, id_to_labels=dict(frame.id_to_labels))


def _gt_binding(**kw):
    kw.setdefault("cameras", ("persp",))
    kw.setdefault("ground_truth", ("depth", "segmentation"))
    kw.setdefault("render_hw", (64, 128))
    return kw


def test_a_capture_measures_whether_the_object_moved_rather_than_trusting_the_schedules_name(
    tmp_path,
):
    """The point of the field. `--schedule trajectory` is a NAME; a capture whose prop never
    moved would carry that name just as well, and a reader six months later cannot re-render it
    to check. The stock fake's square is static, and the header says so in a number."""
    out = tmp_path / "cap"
    binding = FakeIsaacBinding(**_gt_binding())
    ed.capture_frames(binding, "persp", 5, out, steps_per_frame=1, object_class="apple")
    block = json.loads((out / "capture.json").read_text())["temporal_coherence"]
    assert block["measured"] is True
    assert block["object_moved_during_capture"] is False
    assert block["max_interframe_motion_px"] == 0.0
    assert block["n_interframe_steps"] == 4


def test_a_moving_object_is_measured_as_moving_and_its_step_size_recorded(tmp_path):
    out = tmp_path / "cap"
    ed.capture_frames(
        _MovingMaskBinding(**_gt_binding()), "persp", 5, out, steps_per_frame=1,
        object_class="apple",
    )
    block = json.loads((out / "capture.json").read_text())["temporal_coherence"]
    assert block["object_moved_during_capture"] is True
    assert block["max_interframe_motion_px"] == pytest.approx(_MovingMaskBinding.STEP_PX)
    assert block["interframe_motion_px"] == pytest.approx([3.0, 3.0, 3.0, 3.0])


def test_a_capture_that_cannot_name_its_object_records_an_absence_and_not_a_zero(tmp_path):
    """Zero motion and unknown motion are different facts, and one of them would read as "this
    capture is a still life" in an artifact nobody can re-render."""
    out = tmp_path / "cap"
    ed.capture_frames(FakeIsaacBinding(**_gt_binding()), "persp", 3, out, steps_per_frame=1)
    block = json.loads((out / "capture.json").read_text())["temporal_coherence"]
    assert block["measured"] is False
    assert block["object_moved_during_capture"] is None
    assert block["max_interframe_motion_px"] is None
    assert block["absent_because"]


def test_the_coherence_block_does_not_overload_the_registered_static_prop_field(tmp_path):
    """`object_limitations.object_is_static_prop` is registered by PR-08-V5 §4.4 and means
    "teleported, not dropped or grasped" - which stays true on a trajectory capture. The measured
    field answers a different question and must not be read as a correction to it."""
    out = tmp_path / "cap"
    ed.capture_frames(
        _MovingMaskBinding(**_gt_binding()), "persp", 3, out, steps_per_frame=1,
        object_class="apple",
    )
    block = json.loads((out / "capture.json").read_text())["temporal_coherence"]
    assert "object_is_static_prop" in block["not_the_same_claim_as"]
    assert "grasp" in block["not_the_same_claim_as"] or "drop" in block["not_the_same_claim_as"]


def test_the_schedule_flag_is_refused_on_a_backend_that_drives_no_schedule(
    contract, tmp_path, capsys
):
    """The isaac path is unchanged in every flag and refusal (PR-08-V5 §0). A schedule name it
    cannot honour is refused rather than accepted and ignored, which is how a capture comes to
    carry a header field that is a lie."""
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path, "--schedule", "trajectory")) == 2
    err = capsys.readouterr().err
    assert "--schedule" in err and "mujoco" in err
    assert not (tmp_path / "cap").exists()


def test_the_default_schedule_is_the_lattice_so_every_existing_capture_still_means_what_it_meant(
    contract, tmp_path
):
    """Not a claim about the mujoco branch - a claim about the DEFAULT. With no --schedule the
    isaac/fake path must not refuse, because the flag it never passed still has its old value."""
    contract(grid=(12, 20))
    assert ed.main(_capture_argv(tmp_path)) == 0


# -- the mask against the ground truth, which is the half nobody has measured --------------------


def _iou_capture(tmp_path, ids_per_frame) -> pathlib.Path:
    """A hand-built capture whose true mask is exactly what the test says it is."""
    root = tmp_path / "iou_cap"
    (root / "frames").mkdir(parents=True)
    for i, ids in enumerate(ids_per_frame):
        d = root / "frames" / f"{i:06d}"
        d.mkdir()
        rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (ids == 2) * 255
        np.save(d / "rgb.npy", rgb)
        np.save(d / "depth.npy", np.ones(ids.shape, dtype="float32"))
        np.save(d / "seg_ids.npy", ids.astype("uint32"))
        (d / "seg_labels.json").write_text(
            json.dumps({"1": {"class": "table"}, "2": {"class": "apple"}}), encoding="utf-8"
        )
    (root / "capture.json").write_text(
        json.dumps(
            {
                "schema": "wam.est_drift_capture/1",
                "binding": "MuJoCoGroundTruthBinding",
                "camera": "head",
                "n_frames": len(ids_per_frame),
                "steps_per_frame": 1,
                "resolution_hw": list(ids_per_frame[0].shape),
                "ticks": list(range(1, len(ids_per_frame) + 1)),
                "steps_per_state": 1,
                "is_simulated_binding": False,
                "ground_truth_route": "mujoco",
                "scene_schedule": "trajectory",
            }
        ),
        encoding="utf-8",
    )
    return root


def _square(shape, x, y, half=4):
    ids = np.zeros(shape, dtype="uint32")
    ids[y - half : y + half, x - half : x + half] = 2
    return ids


def test_the_ground_truth_iou_is_recorded_and_is_not_the_colour_reference_check(tmp_path):
    """THE SECOND LIMB OF BLOCKER 1's DISCHARGE CONDITION - *"a mask-vs-ground-truth IoU
    distribution from the Isaac capture recorded beside the centroid displacement"*.

    Named apart from `mask_validity_iou` deliberately: that one is the adapter's own check of a
    mask against a warm-and-saturated COLOUR predicate, which is a second opinion and not the
    truth. Two things called "the IoU" in one artifact is how one gets read as the other."""
    cap = _iou_capture(tmp_path, [_square((32, 48), 12 + i, 16) for i in range(6)])
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(cap), "--estimators", _naive_estimator(tmp_path),
         "--object-class", "apple", "--min-area-px", "1", "--min-coverage", "0.0",
         "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    block = doc["mask_vs_ground_truth_iou"]
    assert block["recorded"] is True
    assert block["n"] == 6
    # The stub segments `rgb[:, :, 0] > 0`, which is painted from the true mask, so the two agree
    # exactly. A perfect score is the right assertion for a rig test: it says the two masks were
    # compared frame for frame rather than that the estimator is good.
    assert block["min"] == pytest.approx(1.0)
    assert block["percentiles"]["p95"] == pytest.approx(1.0)
    assert len(block["values"]) == 6
    assert "mask_validity_iou" in block["not_to_be_confused_with"]
    assert "colour" in block["not_to_be_confused_with"].lower()


def test_an_estimator_that_found_nothing_scores_zero_iou_rather_than_being_dropped(tmp_path):
    """A missed detection is a real event and `coverage` already counts it. It is also an IoU of
    exactly zero against a ground truth that HAS the object, and folding it out of this
    distribution instead would report the estimator's error only on the frames it got right."""
    cap = _iou_capture(tmp_path, [_square((32, 48), 12 + i, 16) for i in range(4)])
    blind = _stub_module(
        tmp_path,
        "blind_estimator",
        "import numpy as np\n"
        "ESTIMATOR_NAME = 'finds-nothing'\n"
        "ESTIMATOR_VERSION = '0'\n"
        "def segment(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype=bool)\n"
        "def estimate_depth(rgb):\n"
        "    return np.zeros(np.asarray(rgb).shape[:2], dtype='float32')\n",
    )
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(cap), "--estimators", blind, "--object-class", "apple",
         "--min-area-px", "1", "--min-coverage", "0.0", "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    assert doc["centroid_displacement"]["n"] == 0, "no centroid pairs at all - every frame dropped"
    assert doc["mask_vs_ground_truth_iou"]["n"] == 4, "and yet four frames of measured error"
    assert doc["mask_vs_ground_truth_iou"]["max"] == 0.0
    assert doc["mask_vs_ground_truth_iou"]["n_frames_zero_iou"] == 4


def test_the_ground_truth_iou_block_refuses_to_be_read_as_a_discharge(tmp_path):
    """Producing evidence and accepting it are two different acts (the same rule
    `estimator_stats` is written under). `GATE_QUALIFIED` is untouched by this measurement and
    the artifact has to say so where the number is, not in a commit message."""
    cap = _iou_capture(tmp_path, [_square((32, 48), 12, 16) for _ in range(3)])
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(cap), "--estimators", _naive_estimator(tmp_path),
         "--object-class", "apple", "--min-area-px", "1", "--min-coverage", "0.0",
         "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    block = doc["mask_vs_ground_truth_iou"]
    assert "NOTHING" in block["discharges"]
    assert "GATE_QUALIFICATION_BLOCKERS" in block["discharges"]
    assert doc["gate_qualified"] is False
    assert "estimator_not_gate_qualified" in doc["gate_disqualified_reasons"]


def test_the_iou_block_says_this_was_a_simulator_and_leaves_the_isaac_question_open(tmp_path):
    """Blocker 3 says "the same **Isaac** capture" in its own words; PR-08-V5 rerouted §4 to
    MuJoCo for a different purpose. Whether MuJoCo may stand where the blocker says Isaac is an
    OPEN RULE QUESTION for the project owner, and this artifact states it rather than deciding
    it - and states, unconditionally, that a simulator capture says nothing on its own about the
    real AppleToPlate corpus."""
    cap = _iou_capture(tmp_path, [_square((32, 48), 12, 16) for _ in range(3)])
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(cap), "--estimators", _naive_estimator(tmp_path),
         "--object-class", "apple", "--min-area-px", "1", "--min-coverage", "0.0",
         "--out", str(out)]
    )
    block = json.loads(out.read_text())["mask_vs_ground_truth_iou"]
    assert "AppleToPlate" in block["simulator_caveat"]
    assert "Isaac" in block["open_rule_question"]
    assert "owner" in block["open_rule_question"]


def test_the_measured_artifact_carries_the_schedule_and_the_coherence_the_capture_recorded(
    tmp_path,
):
    """The schedule name and the measured motion are properties of the CAPTURE, so they are
    copied into the budget artifact the same way the stage and the object limitations are: a
    reader of the number must not have to go and find the capture directory."""
    cap = _iou_capture(tmp_path, [_square((32, 48), 12 + i, 16) for i in range(4)])
    header = json.loads((cap / "capture.json").read_text())
    header["temporal_coherence"] = {"measured": True, "max_interframe_motion_px": 1.0}
    (cap / "capture.json").write_text(json.dumps(header), encoding="utf-8")
    out = tmp_path / "d.json"
    ed.main(
        ["measure", "--capture", str(cap), "--estimators", _naive_estimator(tmp_path),
         "--object-class", "apple", "--min-area-px", "1", "--min-coverage", "0.0",
         "--out", str(out)]
    )
    doc = json.loads(out.read_text())
    assert doc["capture"]["scene_schedule"] == "trajectory"
    assert doc["capture"]["temporal_coherence"]["max_interframe_motion_px"] == 1.0
