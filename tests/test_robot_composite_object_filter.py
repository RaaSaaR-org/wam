"""Tests for PR-08 §6 G0c's robot-mask object-grounding filter — ``T40_RULE_V9``.

The failure these pin is not "it crashed". It is the one shape of PR-08 defect that no gate in the
pre-registration can see:

  the masker grounds     ``"robot arm. robotic hand. robotic gripper."`` against a frame with no
  the APPLE              robot in it still returns its best-scoring box above 0.15, and on this
                         corpus that box lands on the fruit. Measured, not supposed:
                         ``runs/pr08-robot-mask-empty/`` (verdict ABSENT, ~36 % of frames are
                         robot-absent) and a re-segmentation of every box of its 710-frame plan.

  and G0c composites     the robot mask is the region copied back from the SOURCE, so the generated
  the source apple       apple is overwritten by the real one. G0a measures labels. G0b measures
  over the generated     geometry, and a pixel-identical apple has moved zero pixels — it does not
  one                    merely pass G0b, it passes perfectly. §6 says the robot-mask IoU is "a
                         diagnostic on the generator, never a gate". An apple-sized mask is ~0.02 of
                         the frame, far under any plausible area bound. And the mask is not empty,
                         so "zero is zero" never fires. **The defect manufactures a pass.**

So the tests below are about an asymmetry and about a number:

  * an apple-coloured detection must be DROPPED, and if it was the only one the frame's mask is
    empty and :func:`robot_composite.check_mask` refuses the clip — the loud path, with no fallback
    that keeps the best-scoring reject to avoid the refusal;
  * a robot-shaped detection must be KEPT, byte for byte, filter or no filter;
  * both of those in ONE frame must keep the robot and drop only the apple, which is why the filter
    is per candidate and not on the finished union;
  * and :data:`robot_composite.ROBOT_MASK_OBJECT_MAX_IOU` must be a value read off a measured gap
    rather than a tuned one, which is checked by sweeping the gap rather than asserted in prose.

The colour predicate is the REAL ``apple_sam2.object_color_reference`` and the IoU is the REAL
``apple_sam2.mask_validity_iou``: the adapter is imported with ``transformers`` and ``sam2`` stubbed
out of ``sys.modules``, exactly as ``tests/test_apple_sam2_estimator.py`` does it, so no checkpoint,
no GPU and no video are needed and the discriminator under test is the one that ships. Only the
detector's boxes and SAM 2's masks are faked, because those are the two things a test cannot
produce.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import robot_composite as rc  # noqa: E402

CANVAS = 64


# -- the measurement the threshold was read off ---------------------------------------------------
#
# ``runs/`` is gitignored, so a test that loads the artifact is a test that skips. These are the
# numbers themselves.
#
# Every GroundingDINO box above the committed operating point (0.15 / 0.25, prompt unchanged) on the
# 710 frames of ``runs/pr08-robot-mask-empty/plan_corpus.json`` — the stratified 40-episode plan the
# ABSENT diagnosis was produced from — was segmented by the pinned SAM 2 and scored against
# ``apple_sam2.object_color_reference``. 2 845 detections. 2 509 of them score below 0.01 and are
# recorded as a COUNT rather than one by one, because every cut anywhere near the gap treats them
# identically and 2 509 zeros in a file teach a reader nothing; the 336 that could conceivably
# matter are listed in full.
MEASURED_BELOW_0_01 = 2509

#: The count a person confirmed as the apple from the contact sheets
#: (``runs/pr08-robot-mask-apple/sheet_absent_now_empty.png`` and the present/shrunk sheets beside
#: it). This is the identity the sweep below is asserted against — fixed by the flag, not recomputed
#: from the threshold, which is what makes the sweep a check rather than a tautology.
MEASURED_APPLE_DETECTIONS = 146

#: The gap's two edges, as measured. Nothing lies between them.
MEASURED_HIGHEST_NON_APPLE = 0.513101
MEASURED_LOWEST_APPLE = 0.936351

MEASURED_IOUS_ABOVE_0_01 = [
    0.010056, 0.010339, 0.010960, 0.011035, 0.011291, 0.011292, 0.011327, 0.011429, 0.011775,
    0.011996, 0.012036, 0.012067, 0.012298, 0.012406, 0.012476, 0.012778, 0.013486, 0.013784,
    0.014164, 0.014196, 0.014874, 0.018590, 0.018983, 0.021472, 0.022221, 0.022500, 0.025430,
    0.025764, 0.026164, 0.042665, 0.044616, 0.046386, 0.046935, 0.048488, 0.048828, 0.050058,
    0.053169, 0.058471, 0.060597, 0.062784, 0.064207, 0.066585, 0.067438, 0.067682, 0.069013,
    0.069467, 0.071273, 0.071645, 0.072505, 0.072828, 0.074385, 0.076382, 0.076814, 0.078433,
    0.079026, 0.079363, 0.079446, 0.079464, 0.079471, 0.079767, 0.080732, 0.080995, 0.081048,
    0.081057, 0.081716, 0.081883, 0.082348, 0.082432, 0.082875, 0.083518, 0.083828, 0.084090,
    0.084293, 0.084451, 0.084704, 0.084747, 0.085213, 0.085942, 0.086262, 0.086354, 0.086971,
    0.087069, 0.087250, 0.087272, 0.088010, 0.088660, 0.089034, 0.089448, 0.089455, 0.089847,
    0.089984, 0.090377, 0.091972, 0.092210, 0.092212, 0.092261, 0.092275, 0.092387, 0.092441,
    0.092591, 0.093781, 0.093940, 0.094036, 0.095927, 0.096630, 0.096775, 0.097086, 0.097169,
    0.097170, 0.097173, 0.097472, 0.097569, 0.097594, 0.097895, 0.097954, 0.098032, 0.098154,
    0.098157, 0.098170, 0.098362, 0.098467, 0.098812, 0.099776, 0.100347, 0.101332, 0.102481,
    0.103714, 0.104473, 0.108029, 0.109277, 0.110005, 0.110855, 0.111009, 0.111276, 0.111285,
    0.111842, 0.112853, 0.114639, 0.116018, 0.116246, 0.116258, 0.116276, 0.116934, 0.119447,
    0.119711, 0.119740, 0.120398, 0.120801, 0.120952, 0.121306, 0.122900, 0.124556, 0.124718,
    0.125307, 0.125364, 0.125968, 0.130306, 0.131487, 0.133102, 0.135327, 0.137010, 0.138968,
    0.139846, 0.141058, 0.141273, 0.142048, 0.143861, 0.144293, 0.147794, 0.149950, 0.152625,
    0.162577, 0.162960, 0.163404, 0.179073, 0.179170, 0.179337, 0.182758, 0.189828, 0.225790,
    0.228708, 0.246445, 0.252346, 0.311629, 0.311977, 0.312322, 0.364628, 0.510858, 0.512638,
    0.513101, 0.936351, 0.944513, 0.947542, 0.948159, 0.948621, 0.949122, 0.949219, 0.949962,
    0.950331, 0.956339, 0.957626, 0.957848, 0.958005, 0.958340, 0.958391, 0.958703, 0.958846,
    0.958853, 0.958983, 0.959290, 0.959391, 0.960101, 0.960219, 0.960245, 0.960371, 0.960604,
    0.960646, 0.961094, 0.961458, 0.961475, 0.961490, 0.961596, 0.962424, 0.962476, 0.962542,
    0.962615, 0.962706, 0.962779, 0.963210, 0.963303, 0.963346, 0.963604, 0.963696, 0.963812,
    0.963873, 0.963889, 0.964030, 0.964252, 0.964349, 0.965106, 0.965200, 0.965367, 0.965581,
    0.965615, 0.965818, 0.965903, 0.966150, 0.966160, 0.966428, 0.967026, 0.967134, 0.967723,
    0.967742, 0.968190, 0.968468, 0.968474, 0.968558, 0.969027, 0.969054, 0.969240, 0.969246,
    0.969272, 0.969374, 0.969524, 0.969733, 0.970136, 0.970153, 0.970489, 0.970686, 0.970747,
    0.970849, 0.971023, 0.971042, 0.971079, 0.971166, 0.971479, 0.971541, 0.971616, 0.971741,
    0.971757, 0.971767, 0.971785, 0.971898, 0.972171, 0.972185, 0.972280, 0.972454, 0.972514,
    0.972712, 0.972804, 0.973407, 0.973483, 0.973505, 0.973603, 0.973843, 0.974017, 0.974026,
    0.974128, 0.974338, 0.974779, 0.974918, 0.974932, 0.974964, 0.975060, 0.975139, 0.975213,
    0.975310, 0.975366, 0.975625, 0.975706, 0.976483, 0.976717, 0.976718, 0.976802, 0.977394,
    0.977402, 0.977461, 0.977814, 0.978025, 0.978039, 0.978264, 0.978580, 0.978626, 0.978680,
    0.978759, 0.979700, 0.979989, 0.980235, 0.980238, 0.980843, 0.980911, 0.981143, 0.981405,
    0.982206, 0.983359, 0.984690]


# -- the adapter, with transformers and sam2 stubbed out of the import graph ----------------------


class _StubPredictor:
    """SAM 2's image predictor, faked. Every prompted box comes back as its own filled rectangle.

    Faithful to the one property the filter depends on: ``predict(box=[N, 4])`` returns ``N`` masks,
    in the order the boxes were given. Nothing here decides what a mask CONTAINS — the fixture's
    frame does — so a test can put an apple under one box and a robot under another and the filter
    is the only thing that tells them apart.
    """

    def __init__(self) -> None:
        self.image_hw: tuple[int, int] | None = None
        self.boxes_seen: list[np.ndarray] = []

    def set_image(self, image) -> None:
        self.image_hw = np.asarray(image).shape[:2]

    def predict(self, box=None, multimask_output=False):
        assert self.image_hw is not None, "set_image must precede predict"
        boxes = np.asarray(box, dtype=np.float64).reshape(-1, 4)
        self.boxes_seen.append(boxes.copy())
        h, w = self.image_hw
        out = np.zeros((boxes.shape[0], h, w), dtype=np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            out[i, int(y0):int(y1), int(x0):int(x1)] = 1.0
        return out, np.ones(boxes.shape[0]), out


@pytest.fixture()
def adapter(monkeypatch):
    """The REAL ``estimators.apple_sam2``, importable without transformers, sam2 or a checkpoint."""
    import importlib

    for name in [n for n in sys.modules if n == "estimators" or n.startswith("estimators.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("transformers", "sam2", "sam2.build_sam", "sam2.sam2_image_predictor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    tf = sys.modules["transformers"]
    tf.AutoProcessor = object
    tf.AutoModelForZeroShotObjectDetection = object
    tf.pipeline = lambda *a, **k: None
    module = importlib.import_module("estimators.apple_sam2")
    yield module
    for name in [n for n in sys.modules if n == "estimators" or n.startswith("estimators.")]:
        del sys.modules[name]


@pytest.fixture()
def masker(adapter, monkeypatch):
    """A real :class:`Sam2RobotMasker` on the real adapter, with only the two models faked."""
    instance = rc.Sam2RobotMasker()
    instance._module = adapter
    predictor = _StubPredictor()
    monkeypatch.setattr(adapter, "_predictor", lambda: predictor)
    monkeypatch.setattr(adapter, "_detector", lambda: (None, None))
    instance.stub_predictor = predictor
    return instance


def _boxes(masker, monkeypatch, boxes):
    """What the detector grounded on this frame, above the committed threshold."""
    monkeypatch.setattr(
        type(masker), "_boxes", lambda self, frame: np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    )


def _blank() -> np.ndarray:
    """The corpus's tablecloth: neutral, unsaturated, and invisible to the colour reference."""
    return np.full((CANVAS, CANVAS, 3), 70, dtype=np.uint8)


def _paint_apple(frame: np.ndarray, box) -> np.ndarray:
    """Warm and saturated inside ``box``: r=220, g=40, b=30 clears r>90, r-b>50 and sat>0.35."""
    x0, y0, x1, y1 = [int(v) for v in box]
    frame[y0:y1, x0:x1] = (220, 40, 30)
    return frame


def _paint_robot(frame: np.ndarray, box) -> np.ndarray:
    """Near-black, which is what the Dex3 hand and the arm are in this view."""
    x0, y0, x1, y1 = [int(v) for v in box]
    frame[y0:y1, x0:x1] = (18, 18, 20)
    return frame


APPLE_BOX = (8, 8, 20, 20)
ROBOT_BOX = (34, 6, 58, 46)


# -- the asymmetry, which is the whole rule --------------------------------------------------------


def test_a_frame_whose_only_detection_is_the_apple_yields_an_empty_mask(masker, monkeypatch):
    """The defect, refused. This is the frame ``sheet_absent_now_empty.png`` is nine copies of.

    Before the filter this frame produced a confident 144-pixel mask of the fruit, ``check_mask``
    passed it, and G0c composited the SOURCE apple over the generated one on every frame of the
    clip — invisibly to G0a, G0b and the area bound alike.
    """
    frame = _paint_apple(_blank(), APPLE_BOX)
    _boxes(masker, monkeypatch, [APPLE_BOX])

    mask = masker.mask(frame)

    assert mask.shape == (CANVAS, CANVAS)
    assert not mask.any(), "a detection that IS the apple is not a robot and must not be composited"
    assert masker.filter_counters["detections_dropped_as_object"] == 1
    assert masker.filter_counters["frames_emptied_by_the_filter"] == 1


def test_an_emptied_frame_reaches_the_existing_zero_is_zero_refusal(masker, monkeypatch, tmp_path):
    """Fail LOUD, in the direction this file already fails — the filter adds no new quiet path.

    ``check_mask`` refuses an empty mask with no threshold and no number to loosen. The filter is
    built to hand it an empty mask rather than to invent a second, softer refusal beside it, so the
    operator sees the message that already exists and the clip dies where clips already die.
    """
    frame = _paint_apple(_blank(), APPLE_BOX)
    _boxes(masker, monkeypatch, [APPLE_BOX])
    bound = rc.AreaBound(
        max_frame_fraction=0.6, artifact=tmp_path / "bound.json", artifact_sha256="0" * 64,
        rationale="a test bound; the empty-mask refusal fires before any bound is consulted",
    )

    with pytest.raises(rc.CompositeError, match="the robot mask is EMPTY"):
        rc.check_mask(masker.mask(frame), frame_index=0, bound=bound, source="clip")


def test_a_genuine_robot_detection_passes_through_untouched(masker, monkeypatch):
    """The other half of the asymmetry, and the one a careless fix breaks.

    A filter that refused a robot mask would leave the generated manipulator in the frame — the
    exact defect G0c exists to exclude, arriving through G0c's own repair. The mask must be
    identical, pixel for pixel, to what the union rule produced before the filter existed.
    """
    frame = _paint_robot(_blank(), ROBOT_BOX)
    _boxes(masker, monkeypatch, [ROBOT_BOX])

    mask = masker.mask(frame)

    expected = np.zeros((CANVAS, CANVAS), dtype=bool)
    expected[ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True
    assert np.array_equal(mask, expected), "the robot's pixels must come back unchanged"
    assert masker.filter_counters["detections_dropped_as_object"] == 0
    assert masker.filter_counters["frames_emptied_by_the_filter"] == 0


def test_the_filter_is_asymmetric_and_the_asymmetry_is_the_rule(masker, monkeypatch):
    """ONE assertion covering both directions, so an edit cannot satisfy half of it.

    A future change that inverts the comparison, or that scores containment rather than overlap, or
    that simply deletes the filter, makes one of these two lines false. Splitting them across two
    tests would let a refactor keep one green and read as mostly passing.
    """
    apple = _paint_apple(_blank(), APPLE_BOX)
    robot = _paint_robot(_blank(), ROBOT_BOX)
    _boxes(masker, monkeypatch, [APPLE_BOX])
    apple_mask = masker.mask(apple)
    _boxes(masker, monkeypatch, [ROBOT_BOX])
    robot_mask = masker.mask(robot)

    assert not apple_mask.any() and robot_mask.any(), (
        "the apple is dropped and the robot is kept. Both directions, one assertion: a filter that "
        "kept the apple manufactures a silent G0c pass, and one that dropped the robot lets the "
        "generated manipulator through."
    )


def test_the_filter_is_per_detection_and_not_on_the_finished_union(masker, monkeypatch):
    """The grasp frame. Filtering the union would have to choose; filtering candidates need not.

    ``runs/pr08-robot-mask-apple/DETECTIONS.json`` frame ``episode_000392`` f102 is this frame for
    real: seven detections, one of them the fruit at IoU 0.959 and three of them robot boxes that
    also swallow it at 0.12. Dropping the apple box there removes 140 px of a 31 710 px mask — the
    gripper's own pixels are covered by the gripper's own detections and stay.
    """
    frame = _paint_robot(_paint_apple(_blank(), APPLE_BOX), ROBOT_BOX)
    _boxes(masker, monkeypatch, [APPLE_BOX, ROBOT_BOX])

    mask = masker.mask(frame)

    assert mask.any(), "the robot detection must survive its neighbour being dropped"
    assert not mask[APPLE_BOX[1]:APPLE_BOX[3], APPLE_BOX[0]:APPLE_BOX[2]].any(), (
        "the apple's pixels must not be composited back")
    assert mask[ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]].all()
    assert masker.filter_counters["detections_dropped_as_object"] == 1
    assert masker.filter_counters["frames_emptied_by_the_filter"] == 0
    assert masker.filter_counters["frames_with_a_dropped_detection"] == 1


def test_dropping_everything_does_not_fall_back_to_the_best_reject(masker, monkeypatch):
    """No "keep the highest-scoring one so the clip survives" path, ever.

    The module docstring refuses upstream's ``(0.10, 0.10)`` retry on exactly this ground: here a
    weak or wrong detection does not recover a frame, it suppresses a refusal. A fallback inside the
    filter would be that same trade made twice.

    Two boxes on ONE fruit, not two fruits, because that is what the corpus does: the recorded
    detections carry near-duplicate boxes on the same object routinely (``DETECTIONS.json``,
    ``episode_000372`` f52 has two at 3 094 px and two at 1 427 px).
    """
    frame = _paint_apple(_blank(), APPLE_BOX)
    _boxes(masker, monkeypatch, [APPLE_BOX, (7, 7, 21, 21)])

    assert not masker.mask(frame).any()
    assert masker.filter_counters["detections_dropped_as_object"] == 2
    assert masker.filter_counters["frames_emptied_by_the_filter"] == 1


def test_a_frame_the_detector_grounds_nothing_on_is_unchanged_by_the_filter(masker, monkeypatch):
    """The ABSENT verdict's own frames. An empty mask was already correct and already refused.

    ``runs/pr08-robot-mask-empty/DIAGNOSIS.json`` settled that these are right answers, and V9 does
    not revisit it: with no boxes there is nothing to filter, no counter moves, and ``check_mask``
    refuses for the reason it always did.
    """
    _boxes(masker, monkeypatch, np.zeros((0, 4)))

    assert not masker.mask(_blank()).any()
    assert masker.filter_counters["detections_segmented"] == 0
    assert masker.filter_counters["frames_emptied_by_the_filter"] == 0, (
        "a frame with no detection was not emptied BY THE FILTER, and conflating the two would "
        "misattribute ~36 % of this corpus")


def test_a_frame_with_no_fruit_in_it_is_counted_as_such(masker, monkeypatch):
    """``frames_with_no_object_reference`` — the case where the second opinion had nothing to say.

    Its being non-zero is not by itself a defect: the apple leaves frame too. Its being non-zero and
    UNREAD would be, and on the restyled frames of arms B/C/D a high value is a finding about the
    reference rather than about the masker. Same reasoning as ``T40_RULE_V6`` §5.3.
    """
    _boxes(masker, monkeypatch, [ROBOT_BOX])
    masker.mask(_paint_robot(_blank(), ROBOT_BOX))

    assert masker.filter_counters["frames_with_no_object_reference"] == 1
    assert masker.filter_counters["detections_dropped_as_object"] == 0


# -- the number ------------------------------------------------------------------------------------


def _measured() -> list[float]:
    return MEASURED_IOUS_ABOVE_0_01 + [0.0] * MEASURED_BELOW_0_01


def test_the_measured_populations_do_not_overlap() -> None:
    """The premise of everything below: there is a gap, and these are its edges."""
    values = _measured()
    assert len(values) == 2845
    assert len([v for v in values if v > MEASURED_HIGHEST_NON_APPLE]) == MEASURED_APPLE_DETECTIONS
    assert min(v for v in values if v > MEASURED_HIGHEST_NON_APPLE) == MEASURED_LOWEST_APPLE
    assert max(v for v in values if v < MEASURED_LOWEST_APPLE) == MEASURED_HIGHEST_NON_APPLE
    assert not [v for v in values if MEASURED_HIGHEST_NON_APPLE < v < MEASURED_LOWEST_APPLE]


def test_every_threshold_in_the_gap_partitions_the_measured_detections_identically() -> None:
    """The defence of admitting a number into a gate path at all — swept, not asserted.

    ``ROBOT_MASK_OBJECT_MAX_IOU``'s exact value is irrelevant over an interval 0.42 wide, so it
    cannot have been tuned against an outcome. The identity checked against is the count a person
    confirmed from the contact sheets, not a recomputation of the threshold against itself.
    """
    values = _measured()
    cuts = [round(0.52 + 0.01 * i, 2) for i in range(42)]
    assert cuts[0] == 0.52 and cuts[-1] == 0.93
    for cut in cuts:
        dropped = len([v for v in values if v > cut])
        assert dropped == MEASURED_APPLE_DETECTIONS, f"the partition moved at {cut}"


def test_the_shipped_threshold_lies_strictly_inside_the_measured_gap() -> None:
    """And is not sitting on either edge, where one borderline frame of a new corpus moves it."""
    assert MEASURED_HIGHEST_NON_APPLE < rc.ROBOT_MASK_OBJECT_MAX_IOU < MEASURED_LOWEST_APPLE
    assert rc.ROBOT_MASK_OBJECT_MAX_IOU - MEASURED_HIGHEST_NON_APPLE > 0.1
    assert MEASURED_LOWEST_APPLE - rc.ROBOT_MASK_OBJECT_MAX_IOU > 0.1


def test_the_threshold_cannot_be_moved_from_the_environment_or_a_flag() -> None:
    """Same rule as ``ROBOT_TEXT_PROMPT``, and ``apple_sam2.MASK_VALIDITY_MIN_IOU`` before it.

    A per-run value here is a per-run decision about which pixels the generator may touch, taken on
    a submit line, recorded nowhere a reader would look, and invisible in the output.
    """
    source = (_REPO / "scripts" / "robot_composite.py").read_text(encoding="utf-8")
    assignment = [
        line for line in source.splitlines()
        if re.match(r"^ROBOT_MASK_OBJECT_MAX_IOU\s*=", line)
    ]
    assert assignment == ["ROBOT_MASK_OBJECT_MAX_IOU = 0.70"], assignment
    assert "WAM_PR08_ROBOT_MASK" not in source
    assert not re.search(r"ROBOT_MASK_OBJECT_MAX_IOU.*environ", source)
    assert not any(
        "object" in (action.option_strings and action.option_strings[0] or "")
        and "iou" in (action.option_strings and action.option_strings[0] or "")
        for action in rc.build_parser()._actions
    )


# -- the record ------------------------------------------------------------------------------------


def test_the_filter_is_part_of_the_segmenter_identity(masker) -> None:
    """So a cached mask and a committed area bound cannot survive it changing.

    ``SEGMENTER_IDENTITY_FIELDS`` is the one definition of "everything that changes which pixels
    come back for a given frame", read by :meth:`MaskCache.key` and by ``load_area_bound``'s
    cross-check. A filter that changed the mask but not the identity is the drift PR-13 is about.
    """
    assert "object_grounding_filter" in rc.SEGMENTER_IDENTITY_FIELDS

    prov = masker.provenance()
    assert "0.7" in prov["object_grounding_filter"]
    assert rc.segmenter_identity(prov)["object_grounding_filter"] == prov["object_grounding_filter"]

    without = dict(prov)
    without.pop("object_grounding_filter")
    assert rc.segmenter_identity(without) != rc.segmenter_identity(prov), (
        "a provenance that stopped declaring the filter must compare UNEQUAL, not equal by absence")


def test_a_cached_mask_does_not_survive_the_filter_changing(masker, tmp_path) -> None:
    """The cache key is the provenance, and the provenance now carries the filter."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not really a video, but it hashes")
    prov = masker.provenance()
    unfiltered = dict(prov, object_grounding_filter="none")
    assert rc.MaskCache.key(video, prov) != rc.MaskCache.key(video, unfiltered)


def test_an_adapter_that_stops_declaring_the_predicate_is_a_refusal(masker, monkeypatch) -> None:
    """Not a silently unfiltered mask. Every G0c record claims this filter ran.

    Same treatment as ``upstream_retry_not_run``: an unverifiable sentence in 10 050 records is
    worse than a refusal here, and an unfiltered mask is worse than both.
    """
    monkeypatch.delattr(masker._module, "object_color_reference")
    _boxes(masker, monkeypatch, [APPLE_BOX])
    with pytest.raises(rc.CompositeError, match="no longer declares object_color_reference"):
        masker.mask(_paint_apple(_blank(), APPLE_BOX))

    monkeypatch.delattr(masker._module, "MASK_VALIDITY_REFERENCE")
    with pytest.raises(rc.CompositeError, match="no longer declares MASK_VALIDITY_REFERENCE"):
        masker.provenance()


def test_the_discriminator_is_the_adapter_s_and_is_not_restated_here(masker) -> None:
    """One definition of "this region is the apple" in the repository, reached rather than copied.

    ``T40_RULE_V6`` already runs ``object_color_reference`` on every ``segment()`` call. Two copies
    of a discriminator drift, and this pair would drift silently because the two callers never
    compare their answers.
    """
    source = (_REPO / "scripts" / "robot_composite.py").read_text(encoding="utf-8")
    assert "saturation" not in source, "the colour predicate must not be restated in this module"
    assert "object_color_reference" in source and "mask_validity_iou" in source

    frame = _paint_apple(_blank(), APPLE_BOX)
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[APPLE_BOX[1]:APPLE_BOX[3], APPLE_BOX[0]:APPLE_BOX[2]] = True
    direct = masker._module.mask_validity_iou(mask, masker._module.object_color_reference(frame))
    assert masker.object_grounding_iou(frame, mask[None, :, :])[0] == pytest.approx(direct)


def test_the_filter_record_carries_its_constants_beside_its_counts(masker) -> None:
    """Zeros that mean "no such mechanism" must not read as "it never fired" — V6 §6's rule."""
    record = masker.filter_record()
    assert record["max_iou"] == rc.ROBOT_MASK_OBJECT_MAX_IOU
    assert "r>90" in record["reference"]
    assert record["frames_masked"] == 0
    for name in rc.Sam2RobotMasker._COUNTERS:
        assert name in record


# -- where the filter is allowed to be asked the question ------------------------------------------


def _composite_clip_body() -> list[str]:
    source = (_REPO / "scripts" / "robot_composite.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("def composite_clip("))
    end = next(i for i in range(start + 1, len(source)) if source[i].startswith("def "))
    return [line.strip() for line in source[start:end]]


def test_the_composite_takes_its_mask_from_the_source_frame_and_only_from_there() -> None:
    """The reason the colour reference's SOURCE-corpus assumption is sound where it is used.

    ``object_color_reference``'s docstring justifies itself with "the only saturated warm thing in
    any of these frames is the fruit", which is a claim about AppleToPlate's real pixels and is
    FALSE on a restyle by construction — ``configs/transfer25/styles.toml`` varies apple colour and
    variety on purpose (green Granny Smith, pale-green waxy, pale-yellow Golden Delicious, brown
    russet, mottled Pink Lady) and varies the table with it. Measured on job 189926's own contact
    sheet, ``train-01-oak-tungsten``'s generated frame returns 34 632 warm-saturated pixels of OAK
    TABLE and about a thousand of green apple: the predicate does not merely go quiet there, it
    moves to a different object.

    G0c is unexposed to that, and by placement rather than by luck: the mask that decides which
    pixels are composited is made from the SOURCE frame, which is where the predicate's claim is
    true. The one call on a generated frame is the robot-mask IoU that PR-08 §6 calls "a diagnostic
    on the generator, never a gate", twice. This pins the placement, because a future edit that
    masked the generated frame and composited THAT would move the filter onto pixels its reference
    does not describe, silently.
    """
    body = _composite_clip_body()

    assert "masks, from_cache = source_masks(source_video, src, context)" in body
    assert "out[index] = composite_frame(src[index], gen[index], mask)" in body

    generated_masks = [line for line in body if "masker.mask(" in line]
    assert generated_masks == [
        "ious.append(mask_iou(mask, np.asarray(context.masker.mask(gen[index]), dtype=bool)))"
    ], generated_masks
    assert any("THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE" in line for line in body)


def test_the_recorded_counts_are_the_source_pass_and_say_so() -> None:
    """Differenced before the diagnostic, so the two populations are never pooled.

    A block that added the generated frames' counts to the source frames' would report a filter
    firing rate over a mixture of pixels the reference describes and pixels it does not, which is a
    number that answers neither question.
    """
    body = _composite_clip_body()
    before = body.index("before_filter = dict(getattr(context.masker, \"filter_counters\", {}) or {})")
    after = body.index("after_filter = dict(getattr(context.masker, \"filter_counters\", {}) or {})")
    loop = body.index("for index in range(src.shape[0]):")
    assert before < after < loop, "the delta must close before the generated frames are masked"


# -- the rule has exactly one implementation, and it is callable ------------------------------------
#
# ``scripts/diagnose_robot_mask_empty.py`` reimplements the mask path so that it can report WHY a
# mask was empty (the masker returns all-False for "no boxes" and for "SAM 2 segmented nothing"
# alike). That reimplementation predates V9, so it now disagrees with :meth:`Sam2RobotMasker.mask`
# and the diagnose module's own ``--verify`` guard would fire. The repair is NOT a second copy of
# the V9 rule over there — two implementations of one rule is the drift these gates exist to catch,
# and ``test_the_discriminator_is_the_adapter_s_and_is_not_restated_here`` above already refuses it
# one level down. So the rule is factored into :meth:`Sam2RobotMasker.object_grounding_keep` and
# both callers reach it. The tests below pin that the extraction changed nothing.


def test_object_grounding_keep_is_the_rule_and_mask_is_its_union(masker, monkeypatch):
    """The extracted unit decides, and :meth:`mask` only ORs what it kept.

    Checked on the frame that has both — filtering the union could not produce this answer and
    neither could a mask that re-derived the keep flags differently.
    """
    frame = _paint_robot(_paint_apple(_blank(), APPLE_BOX), ROBOT_BOX)
    stacked = np.zeros((2, CANVAS, CANVAS), dtype=bool)
    stacked[0, APPLE_BOX[1]:APPLE_BOX[3], APPLE_BOX[0]:APPLE_BOX[2]] = True
    stacked[1, ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True

    keep = masker.object_grounding_keep(frame, stacked)

    assert keep.tolist() == [False, True], "the apple is dropped, the robot is kept, per candidate"
    _boxes(masker, monkeypatch, [APPLE_BOX, ROBOT_BOX])
    fresh = rc.Sam2RobotMasker()
    fresh._module = masker._module
    assert np.array_equal(fresh.mask(frame), np.any(stacked[keep], axis=0))


@pytest.mark.parametrize(
    "boxes,painted,expected_boxes,dropped,emptied",
    [
        ([APPLE_BOX], [APPLE_BOX], [], 1, 1),
        ([ROBOT_BOX], [ROBOT_BOX], [ROBOT_BOX], 0, 0),
        ([APPLE_BOX, ROBOT_BOX], [APPLE_BOX, ROBOT_BOX], [ROBOT_BOX], 1, 0),
        ([], [], [], 0, 0),
    ],
)
def test_extracting_the_filter_changed_nothing_mask_returns(
    masker, monkeypatch, boxes, painted, expected_boxes, dropped, emptied
):
    """The behaviour :meth:`mask` had before the extraction, enumerated rather than asserted.

    Four cases and their counters, written down independently of the implementation: every
    detection dropped, none dropped, one of two dropped, and nothing detected at all. A refactor
    that moved a counter increment across the ``if dropped:`` boundary, or that returned the union
    of an all-False keep instead of a fresh zeros array, breaks one of these rows.
    """
    frame = _blank()
    for box in painted:
        (_paint_apple if box == APPLE_BOX else _paint_robot)(frame, box)
    _boxes(masker, monkeypatch, np.asarray(boxes, dtype=np.float64).reshape(-1, 4))

    mask = masker.mask(frame)

    expected = np.zeros((CANVAS, CANVAS), dtype=bool)
    for x0, y0, x1, y1 in expected_boxes:
        expected[y0:y1, x0:x1] = True
    assert np.array_equal(mask, expected)
    assert mask.dtype == np.dtype(bool) and mask.shape == (CANVAS, CANVAS)
    assert masker.filter_counters["detections_dropped_as_object"] == dropped
    assert masker.filter_counters["frames_emptied_by_the_filter"] == emptied
    assert masker.filter_counters["frames_with_a_dropped_detection"] == (1 if dropped else 0)
    assert masker.filter_counters["frames_masked"] == 1


def test_the_threshold_comparison_appears_exactly_once_in_the_repository() -> None:
    """One implementation of the V9 rule, reachable by every caller that needs it.

    ``scripts/diagnose_robot_mask_empty.py`` must reach it rather than grow its own
    ``<= ROBOT_MASK_OBJECT_MAX_IOU``, which is what this refuses. ``provenance`` and
    ``filter_record`` quote the constant into a STRING and are excluded by that spelling.
    """
    hits = []
    for path in sorted((_REPO / "scripts").rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"[<>=]=?\s*ROBOT_MASK_OBJECT_MAX_IOU|ROBOT_MASK_OBJECT_MAX_IOU\s*[<>]", line):
                hits.append((str(path.relative_to(_REPO)), i, line.strip()))
    assert len(hits) == 1, hits
    assert hits[0][0] == "scripts/robot_composite.py"

    body = _method_body("object_grounding_keep")
    assert any("ROBOT_MASK_OBJECT_MAX_IOU" in line for line in body), (
        "the one comparison must live in the unit both callers reach")

    diagnose = (_REPO / "scripts" / "diagnose_robot_mask_empty.py").read_text(encoding="utf-8")
    assert "object_grounding_keep" in diagnose, (
        "the diagnosis must call the masker's filter, not re-type its rule")


def _method_body(name: str) -> list[str]:
    source = (_REPO / "scripts" / "robot_composite.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source) if line.strip().startswith(f"def {name}("))
    indent = len(source[start]) - len(source[start].lstrip())
    end = next(
        (i for i in range(start + 1, len(source))
         if source[i].strip() and (len(source[i]) - len(source[i].lstrip())) <= indent),
        len(source),
    )
    return [line.strip() for line in source[start:end]]


# -- V10: the colour reference over GENERATED pixels, recorded rather than assumed ------------------
#
# ``object_grounding_iou`` runs ``apple_sam2.object_color_reference`` on whatever frame it is given,
# and ``composite_clip``'s IoU diagnostic gives it GENERATED frames. There the predicate's own
# justification — "the only saturated warm thing in any of these frames is the fruit" — is false by
# construction, because PR-08's committed prompts change the table and the fruit's colour on purpose.
# Measured on job 189926's ``train-01-oak-tungsten``: 37.18-56.40 % of the frame comes back warm, all
# of it oak table, against ``MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION`` = 0.10.
#
# PR-08 §6 FORBIDS gating on that number, and nothing below turns it into a gate. What the tests
# require is that the artifact SAYS the instrument was inapplicable to the pixels it ran on, so that
# a reader of one clip's record can tell a diagnostic produced by a working reference from one
# produced by a reference that had moved to the table.

OAK_TABLE = (150, 90, 45)


def _paint_oak_table(frame: np.ndarray) -> np.ndarray:
    """Warm, saturated and covering the frame: r=150>90, r-b=105>50, sat=0.70>0.35.

    The shape of ``train-01-oak-tungsten``'s generated frames, painted at 100 % rather than the 37-56
    % that was measured, because the bound this is read against is 0.10 and the point is the side of
    it, not the exact value.
    """
    frame[:, :] = OAK_TABLE
    return frame


def test_a_scene_scale_reference_on_a_generated_frame_is_counted(masker, monkeypatch):
    """The record that does not exist yet: the reference covered the scene and nothing said so."""
    frame = _paint_robot(_paint_oak_table(_blank()), ROBOT_BOX)
    adapter = masker._module
    fraction = adapter.reference_frame_fraction(adapter.object_color_reference(frame))
    assert fraction > adapter.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION, (
        "the fixture must reproduce the measured failure, not merely a warm pixel")

    _boxes(masker, monkeypatch, [ROBOT_BOX])
    mask = masker.mask(frame)

    expected = np.zeros((CANVAS, CANVAS), dtype=bool)
    expected[ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True
    assert np.array_equal(mask, expected), (
        "PR-08 §6 forbids gating on this number: the mask must be exactly what V9's comparison "
        "returned, with or without the applicability record")
    assert masker.filter_counters["frames_with_reference_not_object_scale"] == 1


def test_the_applicability_record_decides_nothing_and_v9_still_decides_everything(masker):
    """Not a gate, asserted against V9's own comparison rather than against a hand-written answer.

    ``object_grounding_keep`` must remain exactly ``iou <= ROBOT_MASK_OBJECT_MAX_IOU``, elementwise,
    on a frame whose reference has moved to the table. A V10 predicate that short-circuited the keep
    vector here would be the coined threshold §6 refuses, arriving as a repair.
    """
    frame = _paint_robot(_paint_apple(_paint_oak_table(_blank()), APPLE_BOX), ROBOT_BOX)
    stacked = np.zeros((2, CANVAS, CANVAS), dtype=bool)
    stacked[0, APPLE_BOX[1]:APPLE_BOX[3], APPLE_BOX[0]:APPLE_BOX[2]] = True
    stacked[1, ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True

    overlaps = masker.object_grounding_iou(frame, stacked)
    keep = masker.object_grounding_keep(frame, stacked)

    assert keep.tolist() == (overlaps <= rc.ROBOT_MASK_OBJECT_MAX_IOU).tolist()
    assert masker.filter_counters["frames_with_reference_not_object_scale"] == 2, (
        "the filter was asked twice about a scene-scale reference and both askings are facts")


def test_an_adapter_that_stops_declaring_the_applicability_predicate_is_a_refusal(
    masker, monkeypatch
):
    """T40_RULE_V10 exported it for this call site. Losing it must not silently lose the record."""
    monkeypatch.delattr(masker._module, "reference_is_object_scale")
    _boxes(masker, monkeypatch, [ROBOT_BOX])
    with pytest.raises(rc.CompositeError, match="no longer declares reference_is_object_scale"):
        masker.mask(_paint_robot(_paint_oak_table(_blank()), ROBOT_BOX))


def _composite_context(masker, tmp_path, *, iou_stride: int) -> "rc.CompositeContext":
    bound = rc.AreaBound(
        max_frame_fraction=0.9,
        artifact=tmp_path / "bound.json",
        artifact_sha256="0" * 64,
        rationale="a test bound; this test is about the diagnostic's record, not about the bound",
        cross_checked=True,
        cross_checked_against={"test": True},
    )
    return rc.CompositeContext(masker=masker, bound=bound, iou_stride=iou_stride, cache=None)


def test_the_iou_diagnostic_records_that_its_own_instrument_was_inapplicable(
    masker, monkeypatch, tmp_path
):
    """The artifact, which is the thing that has to say it.

    The IoU block already carries its stride, its measurand and its never-a-gate flag. What it does
    not carry is whether the colour reference the masker consulted on those generated frames was the
    size of an object or the size of the scene — and on arms B/C/D it is the scene. A reader holding
    one ``sample_outputs.json`` cannot otherwise tell the two apart.
    """
    frames = 4
    src = np.stack([_paint_robot(_blank(), ROBOT_BOX) for _ in range(frames)])
    gen = np.stack([_paint_robot(_paint_oak_table(_blank()), ROBOT_BOX) for _ in range(frames)])
    source_video = tmp_path / "source.mp4"
    generated_video = tmp_path / "vision.mp4"
    source_video.write_bytes(b"source")
    generated_video.write_bytes(b"generated")

    masks = np.zeros((frames, CANVAS, CANVAS), dtype=bool)
    masks[:, ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True

    monkeypatch.setattr(rc, "decode_clip", lambda path: src if path == source_video else gen)
    monkeypatch.setattr(rc, "source_masks", lambda video, frames_, context: (masks, False))
    monkeypatch.setattr(rc, "container_fps", lambda path: 30.0)
    monkeypatch.setattr(rc, "encode_clip", lambda arr, path, fps: path.write_bytes(b"composited"))
    _boxes(masker, monkeypatch, [ROBOT_BOX])

    record = rc.composite_clip(
        source_video=source_video,
        generated_video=generated_video,
        context=_composite_context(masker, tmp_path, iou_stride=2),
        expected_frames=frames,
    )

    assert record["composited"] is True, "nothing here may change a refusal outcome"
    diagnostic = record["robot_mask_iou_source_vs_generated"]
    assert diagnostic["frames_sampled"] == 2
    applicability = diagnostic["object_reference_applicability"]
    assert applicability["frames_masked"] == 2, "counted over the sampled GENERATED frames only"
    assert applicability["frames_with_reference_not_object_scale"] == 2
    assert applicability["reference_max_frame_fraction"] == pytest.approx(
        masker._module.MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION
    )
    assert applicability["reference"] == masker._module.MASK_VALIDITY_REFERENCE
    assert "THIS_IS_A_DIAGNOSTIC_ON_THE_GENERATOR_AND_NEVER_A_GATE" in diagnostic

    # And the source-pass block is not polluted by the generated frames' counts, which is the
    # property test_the_recorded_counts_are_the_source_pass_and_say_so pins one level up.
    assert record["robot_mask_object_filter"]["frames_masked"] == 0


def test_the_applicability_record_says_when_it_was_never_asked(masker, monkeypatch, tmp_path):
    """The honest hole in the count, pinned so the note keeps matching the arithmetic.

    The colour reference is consulted inside ``object_grounding_iou``, which ``mask`` only reaches
    when the detector grounded at least one box. On a generated frame it grounded nothing on, the
    reference is never built and ``frames_with_reference_not_object_scale`` cannot move — which
    would read as "the instrument applied" to anyone who did not also look at
    ``detections_segmented``. So the count is emitted beside that one, and the block's note points
    at it rather than at frames_masked.
    """
    frames = 2
    src = np.stack([_paint_robot(_blank(), ROBOT_BOX) for _ in range(frames)])
    gen = np.stack([_paint_oak_table(_blank()) for _ in range(frames)])
    source_video = tmp_path / "source.mp4"
    generated_video = tmp_path / "vision.mp4"
    source_video.write_bytes(b"source")
    generated_video.write_bytes(b"generated")

    masks = np.zeros((frames, CANVAS, CANVAS), dtype=bool)
    masks[:, ROBOT_BOX[1]:ROBOT_BOX[3], ROBOT_BOX[0]:ROBOT_BOX[2]] = True

    monkeypatch.setattr(rc, "decode_clip", lambda path: src if path == source_video else gen)
    monkeypatch.setattr(rc, "source_masks", lambda video, frames_, context: (masks, False))
    monkeypatch.setattr(rc, "container_fps", lambda path: 30.0)
    monkeypatch.setattr(rc, "encode_clip", lambda arr, path, fps: path.write_bytes(b"composited"))
    _boxes(masker, monkeypatch, np.zeros((0, 4)))

    record = rc.composite_clip(
        source_video=source_video,
        generated_video=generated_video,
        context=_composite_context(masker, tmp_path, iou_stride=1),
        expected_frames=frames,
    )

    applicability = record["robot_mask_iou_source_vs_generated"]["object_reference_applicability"]
    assert applicability["frames_masked"] == frames
    assert applicability["detections_segmented"] == 0
    assert applicability["frames_with_reference_not_object_scale"] == 0
    assert "detections_segmented" in applicability["note"], (
        "a zero that means 'never asked' must be readable as such from this block alone")
