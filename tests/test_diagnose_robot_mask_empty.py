"""Tests for scripts/diagnose_robot_mask_empty.py — the pure parts, on synthetic frames.

The two GPU commands are not exercised here (no checkpoints in CI); what IS exercised is every
function whose being wrong would make the diagnosis wrong: the reference predicate's three clauses,
the band that refuses to force a middling frame into a bucket, the run/decile arithmetic that
distinguishes "spread" from "clustered", and the contingency table that is the verdict.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import diagnose_robot_mask_empty as diag  # noqa: E402


CLOTH = 90


def scene(h: int = 64, w: int = 64) -> np.ndarray:
    """A uniform mid-grey cloth, which is what this corpus mostly is."""
    return np.full((h, w, 3), CLOTH, dtype=np.uint8)


def clip(frames: list[np.ndarray]) -> np.ndarray:
    return np.stack(frames)


# -- the reference predicate ------------------------------------------------------------------


def test_dark_moving_blob_is_seen():
    static = scene()
    moving = scene()
    moving[10:30, 10:30] = 10          # matte black, neutral, and not in the background
    frames = clip([static] * 8 + [moving] * 2)
    background = diag.background_median(frames)
    mask = diag.robot_dark_mask(moving, background)
    assert mask.sum() == 400
    assert diag.robot_dark_mask(static, background).sum() == 0


def test_static_dark_region_is_not_seen():
    """The cloth's fold shadows and the dark band along the top edge are dark and STATIC."""
    frame = scene()
    frame[0:8, :] = 20                  # a permanent dark band
    frames = clip([frame] * 10)
    background = diag.background_median(frames)
    assert diag.robot_dark_mask(frame, background).sum() == 0


def test_moving_saturated_apple_is_not_seen():
    """The apple's dark stem and shadow move with it; the saturation clause is what removes them."""
    static = scene()
    moved = scene()
    moved[20:28, 20:28] = (60, 12, 8)   # dark AND strongly warm — an apple shadow, not a robot
    frames = clip([static] * 8 + [moved] * 2)
    background = diag.background_median(frames)
    assert diag.robot_dark_mask(moved, background).sum() == 0


def test_bright_wrist_is_the_documented_blind_spot():
    """Stated in robot_dark_mask's docstring, asserted here so it cannot quietly stop being true."""
    static = scene()
    wrist = scene()
    wrist[10:30, 10:30] = 240           # the white forearm cuff: moving, neutral, NOT dark
    frames = clip([static] * 8 + [wrist] * 2)
    background = diag.background_median(frames)
    assert diag.robot_dark_mask(wrist, background).sum() == 0


def test_background_median_survives_a_minority_arm():
    static = scene()
    arm = scene()
    arm[10:30, 10:30] = 10
    frames = clip([static] * 7 + [arm] * 3)
    background = diag.background_median(frames, stride=1)
    assert np.allclose(background, CLOTH)


def test_background_median_refuses_bad_shapes():
    with pytest.raises(diag.DiagnosisError):
        diag.background_median(np.zeros((4, 8, 8), dtype=np.uint8))
    with pytest.raises(diag.DiagnosisError):
        diag.background_median(np.zeros((0, 8, 8, 3), dtype=np.uint8))


def test_robot_dark_mask_refuses_mismatched_background():
    with pytest.raises(diag.DiagnosisError):
        diag.robot_dark_mask(scene(64, 64), np.zeros((32, 32, 3), dtype=np.float32))


def test_largest_component_ignores_scatter():
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:10, 0:10] = True             # one blob of 100
    mask[30, ::4] = True                # a scatter of isolated pixels
    assert mask.sum() > 100
    assert diag.largest_component(mask) == 100
    assert diag.largest_component(np.zeros((4, 4), dtype=bool)) == 0


# -- the band ---------------------------------------------------------------------------------


def test_classify_has_three_outcomes():
    assert diag.classify(10, absent_below=100, present_above=1000) == "absent"
    assert diag.classify(500, absent_below=100, present_above=1000) == "ambiguous"
    assert diag.classify(5000, absent_below=100, present_above=1000) == "present"


def test_classify_boundaries_are_exclusive_both_ways():
    """Exactly at either edge is 'ambiguous': the band never claims a frame it does not clear."""
    assert diag.classify(100, absent_below=100, present_above=1000) == "ambiguous"
    assert diag.classify(1000, absent_below=100, present_above=1000) == "ambiguous"


def test_classify_refuses_an_inverted_band():
    with pytest.raises(diag.DiagnosisError):
        diag.classify(5, absent_below=1000, present_above=100)


# -- clustering arithmetic --------------------------------------------------------------------


def test_runs_of_finds_both_ends():
    flags = [True, True, False, False, True, True, True]
    assert diag.runs_of(flags) == [(0, 2), (4, 3)]


def test_runs_of_handles_all_and_nothing():
    assert diag.runs_of([True] * 5) == [(0, 5)]
    assert diag.runs_of([False] * 5) == []
    assert diag.runs_of([]) == []


def test_phase_bucket_spans_the_timeline_and_clamps():
    assert diag.phase_bucket(0, 100) == 0
    assert diag.phase_bucket(99, 100) == 9
    assert diag.phase_bucket(50, 100) == 5
    assert diag.phase_bucket(100, 100) == 9        # clamped rather than out of range
    with pytest.raises(diag.DiagnosisError):
        diag.phase_bucket(0, 0)


def test_contingency_counts_the_cell_that_decides():
    table = diag.contingency([
        ("present", True), ("present", False), ("present", False),
        ("absent", True), ("absent", True),
        ("ambiguous", True),
    ])
    assert table["present_empty"] == 1
    assert table["present_nonempty"] == 2
    assert table["absent_empty"] == 2
    assert table["absent_nonempty"] == 0
    assert table["ambiguous_empty"] == 1
    assert table["ambiguous_nonempty"] == 0


# -- settings and selection -------------------------------------------------------------------


def test_all_settings_puts_the_primary_first_and_sweeps_each_constant():
    settings = diag.all_settings()
    assert settings[0] == {
        "dark_offset": diag.DARK_OFFSETS[0],
        "sat_max": diag.SAT_MAXES[0],
        "change_min": diag.CHANGE_MINS[0],
    }
    assert len(settings) == len(diag.DARK_OFFSETS) + len(diag.CHANGE_MINS) + len(diag.SAT_MAXES) - 2
    assert len({diag.setting_key(s) for s in settings}) == len(settings)


def test_select_episodes_keeps_manifest_positions():
    episodes = [{"id": f"e{i}"} for i in range(10)]
    picked = diag.select_episodes(episodes, limit=3, stride=2)
    assert [p for p, _ in picked] == [0, 2, 4]


def test_read_manifest_refuses_an_empty_one(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"episodes": []}))
    with pytest.raises(diag.DiagnosisError):
        diag.read_manifest(path)


# -- end to end over the report ---------------------------------------------------------------


def test_report_joins_visibility_and_detection(tmp_path):
    """A synthetic episode whose robot is absent for the first and last third."""
    areas = [10] * 10 + [8000] * 10 + [10] * 10
    visible = {
        "schema": diag.SCHEMA_VISIBLE,
        "primary_setting": "d45_s0.25_c25",
        "settings": {"d45_s0.25_c25": {"dark_offset": 45, "sat_max": 0.25, "change_min": 25}},
        "per_episode": {
            "episode_000000": {
                "n_frames": 30,
                "frame_indices": list(range(30)),
                "areas": {"d45_s0.25_c25": areas},
                "largest_component": areas,
            }
        },
    }
    detect = {
        "schema": diag.SCHEMA_DETECT,
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "estimator": {"box_threshold": 0.15, "text_threshold": 0.25},
        "per_episode": {
            "episode_000000": [
                {"frame_index": 0, "mask_px": 0, "raw_max": 0.02,
                 "empty_reason": "no_boxes_above_threshold"},
                {"frame_index": 15, "mask_px": 40000, "raw_max": 0.6, "empty_reason": None},
                {"frame_index": 29, "mask_px": 0, "raw_max": 0.03,
                 "empty_reason": "no_boxes_above_threshold"},
            ]
        },
    }
    vis_path, det_path, out = tmp_path / "v.json", tmp_path / "d.json", tmp_path / "r.json"
    vis_path.write_text(json.dumps(visible))
    det_path.write_text(json.dumps(detect))

    rc = diag.main([
        "report", "--visible", str(vis_path), "--detect", str(det_path),
        "--out", str(out), "--absent-below", "1000", "--present-above", "3000",
    ])
    assert rc == 0
    report = json.loads(out.read_text())

    setting = report["visibility_by_setting"]["d45_s0.25_c25"]
    assert setting["absent"] == 20 and setting["present"] == 10 and setting["ambiguous"] == 0

    episode = report["per_episode"]["episode_000000"]
    assert episode["n_absent_runs"] == 2
    assert episode["first_run_starts_at_0"] and episode["last_run_ends_at_end"]

    # clustered, not spread: the first and last thirds are all-absent, the middle third none.
    deciles = report["absent_by_decile"]
    assert deciles[0]["fraction"] == 1.0
    assert deciles[5]["fraction"] == 0.0
    assert deciles[9]["fraction"] == 1.0

    table = report["detector"]["contingency"]
    assert table == {
        "present_empty": 0, "present_nonempty": 1,
        "absent_empty": 2, "absent_nonempty": 0,
        "ambiguous_empty": 0, "ambiguous_nonempty": 0,
    }
    assert report["detector"]["empty_reasons"] == {"no_boxes_above_threshold": 2}


def test_report_counts_detected_frames_with_no_reference_measurement(tmp_path):
    """A frame the visibility pass never measured must be counted, never silently paired."""
    visible = {
        "primary_setting": "p",
        "settings": {"p": {"dark_offset": 45, "sat_max": 0.25, "change_min": 25}},
        "per_episode": {"episode_000000": {
            "n_frames": 2, "frame_indices": [0, 10],
            "areas": {"p": [10, 10]}, "largest_component": [10, 10]}},
    }
    detect = {
        "prompt": "p", "estimator": {"box_threshold": 0.15, "text_threshold": 0.25},
        "per_episode": {
            "episode_000000": [{"frame_index": 7, "mask_px": 0, "raw_max": 0.0,
                                "empty_reason": "no_boxes_above_threshold"}],
            "episode_999999": [{"frame_index": 0, "mask_px": 0, "raw_max": 0.0,
                                "empty_reason": "no_boxes_above_threshold"}],
        },
    }
    vis_path, det_path, out = tmp_path / "v.json", tmp_path / "d.json", tmp_path / "r.json"
    vis_path.write_text(json.dumps(visible))
    det_path.write_text(json.dumps(detect))
    assert diag.main([
        "report", "--visible", str(vis_path), "--detect", str(det_path),
        "--out", str(out), "--absent-below", "1000", "--present-above", "3000",
    ]) == 0
    report = json.loads(out.read_text())
    assert report["detector"]["frames_without_a_reference_measurement"] == 2
    assert report["detector"]["n_frames"] == 0


def test_report_refuses_an_inverted_band(tmp_path, capsys):
    visible = {
        "primary_setting": "p",
        "settings": {"p": {"dark_offset": 45, "sat_max": 0.25, "change_min": 25}},
        "per_episode": {"episode_000000": {
            "n_frames": 1, "frame_indices": [0], "areas": {"p": [10]}, "largest_component": [10]}},
    }
    vis_path, out = tmp_path / "v.json", tmp_path / "r.json"
    vis_path.write_text(json.dumps(visible))
    assert diag.main([
        "report", "--visible", str(vis_path), "--out", str(out),
        "--absent-below", "3000", "--present-above", "1000",
    ]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_episode_visibility_traces_a_synthetic_entry_and_exit():
    static = scene()
    arm = scene()
    arm[10:40, 10:40] = 10
    frames = clip([static] * 6 + [arm] * 4 + [static] * 6)
    record = diag.episode_visibility(frames, settings=diag.all_settings())
    trace = record["areas"][diag.setting_key(diag.all_settings()[0])]
    assert record["n_frames"] == 16
    assert trace[:6] == [0] * 6
    assert trace[6:10] == [900] * 4
    assert trace[10:] == [0] * 6
    assert record["largest_component"][6] == 900


# -- the strata the sheets are drawn from ------------------------------------------------------


def test_stratify_names_the_cell_that_decides():
    visible = {
        "primary_setting": "p",
        "per_episode": {"episode_000000": {
            "frame_indices": [0, 1, 2, 3],
            "areas": {"p": [10, 10, 9000, 9000]},
        }},
    }
    detect = {"per_episode": {"episode_000000": [
        {"frame_index": 0, "mask_px": 0, "raw_max": 0.01, "empty_reason": "no_boxes_above_threshold"},
        {"frame_index": 1, "mask_px": 500, "raw_max": 0.2, "empty_reason": None},
        {"frame_index": 2, "mask_px": 0, "raw_max": 0.04, "empty_reason": "no_boxes_above_threshold"},
        {"frame_index": 3, "mask_px": 40000, "raw_max": 0.7, "empty_reason": None},
    ]}}
    strata = diag.stratify(visible, detect, absent_below=1000, present_above=3000)
    assert set(strata) == {
        "empty_and_reference_absent",
        "nonempty_but_reference_absent",
        "empty_and_reference_present",
        "nonempty_but_reference_present",
    }
    key, index, row = strata["empty_and_reference_present"][0]
    assert (key, index) == ("episode_000000", 2)
    assert row["reference_px"] == 9000


def test_stratify_skips_frames_with_no_reference_measurement():
    visible = {"primary_setting": "p",
               "per_episode": {"episode_000000": {"frame_indices": [0], "areas": {"p": [10]}}}}
    detect = {"per_episode": {
        "episode_000000": [{"frame_index": 5, "mask_px": 0, "raw_max": 0.0, "empty_reason": "x"}],
        "episode_000009": [{"frame_index": 0, "mask_px": 0, "raw_max": 0.0, "empty_reason": "x"}],
    }}
    assert diag.stratify(visible, detect, absent_below=1000, present_above=3000) == {}


def test_evenly_spreads_rather_than_taking_a_prefix():
    items = list(range(100))
    picked = diag.evenly(items, 5)
    assert len(picked) == 5
    assert picked[0] == 0 and picked[-1] >= 80
    assert diag.evenly([1, 2], 10) == [1, 2]
    assert diag.evenly([1, 2], 0) == [1, 2]


def test_frame_fields_and_apply_setting_match_robot_dark_mask():
    """The optimisation must not be a second definition of the predicate."""
    static = scene()
    arm = scene()
    arm[10:30, 10:30] = 10
    background = diag.background_median(clip([static] * 8 + [arm] * 2))
    for setting in diag.all_settings():
        direct = diag.robot_dark_mask(arm, background, **setting)
        staged = diag.apply_setting(diag.frame_fields(arm, background), **setting)
        assert np.array_equal(direct, staged)


def test_verification_frames_are_spread_not_a_prefix():
    """The prefix of this corpus's plan is all robot-absent, where any two maskers agree."""
    plan = {"episode_000000": list(range(590)), "episode_000001": list(range(535))}
    ordered = [(k, i) for k, idxs in plan.items() for i in idxs]
    picked = diag.evenly(ordered, 24)
    assert len(picked) == 24
    assert len({k for k, _ in picked}) == 2                  # both episodes are reached
    assert max(i for _k, i in picked) > 400                  # and the far end of one of them


def test_stratified_plan_gives_every_bucket_a_quota():
    areas = [10] * 100 + [1500] * 4 + [9000] * 100        # ambiguous is 2 % of the episode
    visible = {
        "primary_setting": "p",
        "per_episode": {"episode_000000": {
            "frame_indices": list(range(len(areas))), "areas": {"p": areas}}},
    }
    plan = diag.stratified_plan(
        visible, absent_below=800, present_above=3000, per_bucket_per_episode=4)
    picked = plan["episode_000000"]
    assert sum(1 for i in picked if areas[i] < 800) == 4
    assert sum(1 for i in picked if 800 <= areas[i] <= 3000) == 4
    assert sum(1 for i in picked if areas[i] > 3000) == 4
    assert picked == sorted(set(picked))


def test_stratified_plan_spreads_over_episodes():
    visible = {
        "primary_setting": "p",
        "per_episode": {
            f"episode_{i:06d}": {"frame_indices": [0, 1], "areas": {"p": [10, 9000]}}
            for i in range(20)
        },
    }
    plan = diag.stratified_plan(
        visible, absent_below=800, present_above=3000, per_bucket_per_episode=1, episodes=5)
    assert len(plan) == 5
    assert "episode_000000" in plan and max(plan) != "episode_000001"


# ==============================================================================================
# `detect` — the readout must be the masker G0c runs, V9 filter included
# ==============================================================================================
#
# ``detector_readout`` reimplements the mask path so it can say WHY a mask was empty; the masker
# returns all-False for "no boxes" and for "SAM 2 segmented nothing" alike. That reimplementation
# predates V9's object-grounding filter, so ``cmd_detect --verify`` would now raise its own refusal.
# The repair is to CALL ``Sam2RobotMasker.object_grounding_keep`` rather than to re-type its rule:
# ``tests/test_robot_composite_object_filter.py`` refuses a second copy of that comparison anywhere
# under ``scripts/``. The adapter below is the REAL ``estimators.apple_sam2`` with ``transformers``
# and ``sam2`` stubbed out of the import graph, exactly as that file does it, so the discriminator
# under test is the one that ships and no checkpoint, GPU or video is needed.

import types  # noqa: E402

import robot_composite as rc  # noqa: E402

CANVAS = 64
APPLE_BOX = (8, 8, 20, 20)
ROBOT_BOX = (34, 6, 58, 46)


class _Inputs(dict):
    """What the processor hands back. ``.to(device)`` is a no-op; ``["input_ids"]`` is a key."""

    def to(self, device):  # noqa: ANN001, ANN201
        return self


class _StubProcessor:
    """GroundingDINO's processor, faked at exactly the two seams both mask paths use.

    ONE source of boxes for BOTH paths, deliberately: ``Sam2RobotMasker._boxes`` and
    ``detector_readout`` each post-process this stub, so a pixel-for-pixel agreement between them
    is an agreement about the filter and the union rule rather than an artifact of the test handing
    each of them a different answer.
    """

    def __init__(self) -> None:
        self.boxes: list = []
        self.kept_scores: list[float] = []
        self.raw_scores: list[float] = [0.31, 0.08, 0.02]
        self.texts: list[str] = []

    def __call__(self, images=None, text=None, return_tensors=None):  # noqa: ANN001, ANN201
        import torch

        self.texts.append(text)
        return _Inputs(input_ids=torch.zeros((1, 6), dtype=torch.long))

    def post_process_grounded_object_detection(
        self, outputs, input_ids, threshold, text_threshold, target_sizes
    ):  # noqa: ANN001, ANN201
        import torch

        if float(threshold) <= 0.0:
            return [{
                "boxes": torch.zeros((len(self.raw_scores), 4), dtype=torch.float64),
                "scores": torch.tensor(self.raw_scores, dtype=torch.float64),
            }]
        boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        scores = self.kept_scores or [0.5] * boxes.shape[0]
        return [{
            "boxes": torch.tensor(boxes, dtype=torch.float64),
            "scores": torch.tensor(scores[: boxes.shape[0]], dtype=torch.float64),
        }]


class _StubPredictor:
    """SAM 2's image predictor. Every prompted box comes back as its own filled rectangle.

    ``segments_nothing`` is the third empty-mask cause: boxes above threshold that SAM 2 resolves
    to no pixels at all. It is a different finding from both of the others and the readout has to
    keep saying so.
    """

    def __init__(self, segments_nothing: bool = False) -> None:
        self.image_hw: tuple[int, int] | None = None
        self.segments_nothing = segments_nothing

    def set_image(self, image) -> None:  # noqa: ANN001
        self.image_hw = np.asarray(image).shape[:2]

    def predict(self, box=None, multimask_output=False):  # noqa: ANN001, ANN201
        boxes = np.asarray(box, dtype=np.float64).reshape(-1, 4)
        h, w = self.image_hw
        out = np.zeros((boxes.shape[0], h, w), dtype=np.float32)
        if not self.segments_nothing:
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
    processor = _StubProcessor()
    predictor = _StubPredictor()
    monkeypatch.setattr(adapter, "_detector", lambda: (processor, lambda **kw: object()))
    monkeypatch.setattr(adapter, "_predictor", lambda: predictor)
    monkeypatch.setattr(adapter, "_device", lambda: "cpu")
    instance.stub_processor = processor
    instance.stub_predictor = predictor
    return instance


def _blank() -> np.ndarray:
    return np.full((CANVAS, CANVAS, 3), 70, dtype=np.uint8)


def _paint_apple(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    frame[y0:y1, x0:x1] = (220, 40, 30)
    return frame


def _paint_robot(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    frame[y0:y1, x0:x1] = (18, 18, 20)
    return frame


def test_readout_agrees_with_the_masker_when_the_filter_drops_one_of_two(masker):
    """The grasp frame: one apple box, one robot box. Pixel for pixel, or the diagnosis is void."""
    frame = _paint_robot(_paint_apple(_blank(), APPLE_BOX), ROBOT_BOX)
    masker.stub_processor.boxes = [APPLE_BOX, ROBOT_BOX]

    record = diag.detector_readout(masker, frame)
    mine = record.pop("_mask")

    assert np.array_equal(mine, np.asarray(masker.mask(frame), dtype=bool))
    assert record["empty_reason"] is None
    assert record["n_boxes_kept"] == 2
    assert record["n_dropped_as_object"] == 1
    assert record["mask_px"] == (ROBOT_BOX[2] - ROBOT_BOX[0]) * (ROBOT_BOX[3] - ROBOT_BOX[1])


def test_readout_names_the_filter_when_it_is_what_emptied_the_mask(masker):
    """The new ``empty_reason``, and the entire point of the change.

    "V9 removed the robot detection" and "the detector never found one" are opposite findings about
    ~36 % of this corpus, and before this the readout reported them with the same word.
    """
    frame = _paint_apple(_blank(), APPLE_BOX)
    masker.stub_processor.boxes = [APPLE_BOX]

    record = diag.detector_readout(masker, frame)

    assert record["mask_px"] == 0
    assert record["empty_reason"] == "all_boxes_dropped_as_object"
    assert record["n_boxes_kept"] == 1 and record["n_dropped_as_object"] == 1
    assert not np.asarray(masker.mask(frame), dtype=bool).any()


def test_readout_still_tells_the_other_two_empty_causes_apart(masker):
    """No boxes at all, and boxes that SAM 2 resolved to nothing. Three causes, three words."""
    masker.stub_processor.boxes = []
    nothing = diag.detector_readout(masker, _blank())
    assert nothing["mask_px"] == 0
    assert nothing["empty_reason"] == "no_boxes_above_threshold"
    assert nothing["n_dropped_as_object"] == 0

    masker.stub_predictor.segments_nothing = True
    masker.stub_processor.boxes = [ROBOT_BOX]
    blank_masks = diag.detector_readout(masker, _paint_robot(_blank(), ROBOT_BOX))
    assert blank_masks["mask_px"] == 0
    assert blank_masks["empty_reason"] == "sam2_segmented_nothing"
    assert blank_masks["n_dropped_as_object"] == 0


def test_readout_keeps_reading_the_raw_pre_threshold_scores(masker):
    """The second post-processing pass, at threshold 0, is what the filter change must not cost."""
    masker.stub_processor.boxes = [ROBOT_BOX]
    record = diag.detector_readout(masker, _paint_robot(_blank(), ROBOT_BOX))
    assert record["raw_n"] == 3
    assert record["raw_max"] == pytest.approx(0.31)
    assert record["raw_top5"][:3] == [0.31, 0.08, 0.02]


def test_readout_reports_the_reason_the_masker_itself_cannot(masker):
    """Why this module reimplements the path at all, asserted rather than argued in a comment."""
    apple = _paint_apple(_blank(), APPLE_BOX)
    masker.stub_processor.boxes = [APPLE_BOX]
    dropped = diag.detector_readout(masker, apple)
    masker.stub_processor.boxes = []
    none = diag.detector_readout(masker, _blank())

    assert dropped["mask_px"] == none["mask_px"] == 0
    assert dropped["empty_reason"] != none["empty_reason"], (
        "the masker returns all-False for both; the readout exists to separate them")


def test_cmd_detect_verify_no_longer_refuses(tmp_path, monkeypatch, capsys):
    """End to end through the ``--verify`` guard that PR-08's v12-preconditions §2 says would fire.

    Every planned frame is verified, over a plan whose frames span all three causes, so an
    unfiltered readout cannot pass this by landing only on frames both paths call empty.
    """
    frames = np.stack([
        _paint_robot(_paint_apple(_blank(), APPLE_BOX), ROBOT_BOX),
        _paint_apple(_blank(), APPLE_BOX),
        _blank(),
    ])
    monkeypatch.setattr(diag, "decode", lambda path: frames)

    processor = _StubProcessor()
    predictor = _StubPredictor()
    import estimators.apple_sam2 as adapter_module  # noqa: PLC0415

    class _PlannedMasker(rc.Sam2RobotMasker):
        """The committed masker, whose only fake is which boxes the detector grounded per frame."""

        per_frame = {0: [APPLE_BOX, ROBOT_BOX], 1: [APPLE_BOX], 2: []}

        def _boxes(self, frame):  # noqa: ANN001, ANN202
            processor.boxes = self.per_frame[self.current]
            return super()._boxes(frame)

    instance = _PlannedMasker()
    instance._module = adapter_module
    monkeypatch.setattr(adapter_module, "_detector", lambda: (processor, lambda **kw: object()))
    monkeypatch.setattr(adapter_module, "_predictor", lambda: predictor)
    monkeypatch.setattr(adapter_module, "_device", lambda: "cpu")
    monkeypatch.setattr(rc, "build_masker", lambda: instance)

    original = diag.detector_readout

    def _readout(masker_arg, frame):
        instance.current = int(np.argmax([
            int(np.array_equal(frame, frames[i])) for i in range(frames.shape[0])
        ]))
        processor.boxes = _PlannedMasker.per_frame[instance.current]
        return original(masker_arg, frame)

    monkeypatch.setattr(diag, "detector_readout", _readout)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"id": "episode_000000", "video": "e0.mp4"}]}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"episode_000000": [0, 1, 2]}))
    out = tmp_path / "detect.json"

    rc_code = diag.main([
        "detect", "--manifest", str(manifest), "--plan", str(plan),
        "--out", str(out), "--verify", "3",
    ])
    assert rc_code == 0, capsys.readouterr()

    payload = json.loads(out.read_text())
    assert payload["verified_against_masker_frames"] == 3
    reasons = [r["empty_reason"] for r in payload["per_episode"]["episode_000000"]]
    assert reasons == [None, "all_boxes_dropped_as_object", "no_boxes_above_threshold"]


# ==============================================================================================
# `blind-sheet` / `blind-score` — adjudication whose label is assigned BEFORE the masker's answer
# ==============================================================================================
#
# ``docs/preregistration/PR-08-RESULT-2026-08-25-v12-preconditions.md`` §2: "no frame anywhere
# carries a human label assigned *before* seeing the masker's answer: every human inspection on
# record was of frames nominated by a disagreement with the masker", and the ``absent_empty`` cell
# — where a masker failure would hide — was adjudicated by a predicate whose own docstring says it
# "scores none of" the white/silver forearm and therefore "UNDERSTATES robot presence".
#
# So: three arms, one of them uniform over the whole empty-mask population, one of them deliberately
# aimed at the predicate's blind spot; tiles that reveal neither the arm nor the masker's answer;
# and a scorer that refuses an incomplete sheet and reports the biased arms as biased.


def _wrist(frame: np.ndarray, box=(10, 30, 30, 50)) -> np.ndarray:
    """The white/silver forearm cuff: moving, near-neutral, and BRIGHTER than the cloth."""
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = 240
    return frame


def _arm(frame: np.ndarray, box=(10, 30, 30, 50)) -> np.ndarray:
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = 10
    return frame


def test_blind_spot_score_measures_exactly_what_the_predicate_cannot():
    """Same fields, opposite clause. The wrist the predicate scores at zero is what this scores."""
    static = scene()
    wrist = _wrist(scene())
    arm = _arm(scene())
    background = diag.background_median(clip([static] * 8 + [wrist, arm]))

    assert diag.robot_dark_mask(wrist, background).sum() == 0
    assert diag.blind_spot_score(diag.frame_fields(wrist, background)) == 400

    assert diag.robot_dark_mask(arm, background).sum() == 400
    assert diag.blind_spot_score(diag.frame_fields(arm, background)) == 0

    assert diag.blind_spot_score(diag.frame_fields(static, background)) == 0


def test_blind_spot_score_and_the_predicate_partition_the_moving_neutral_pixels():
    """No third detector: the two are complementary halves of one clause of ``robot_dark_mask``."""
    static = scene()
    both = _arm(_wrist(scene(), (10, 10, 30, 30)), (40, 40, 55, 55))
    background = diag.background_median(clip([static] * 8 + [both] * 2))
    fields = diag.frame_fields(both, background)

    moving_and_neutral = np.count_nonzero(
        (fields["change"] > diag.CHANGE_MINS[0]) & (fields["saturation"] < diag.SAT_MAXES[0])
    )
    assert diag.robot_dark_mask(both, background).sum() + diag.blind_spot_score(fields) == (
        moving_and_neutral)


# -- the population and the three arms -----------------------------------------------------------


def _visible_for(n_frames: int, present_from: int, episodes=("episode_000000",)) -> dict:
    areas = [10] * present_from + [9000] * (n_frames - present_from)
    return {
        "primary_setting": "p",
        "settings": {"p": {"dark_offset": 45, "sat_max": 0.25, "change_min": 25}},
        "per_episode": {
            key: {"n_frames": n_frames, "frame_indices": list(range(n_frames)),
                  "areas": {"p": areas}, "largest_component": areas}
            for key in episodes
        },
    }


def _detect_for(n_frames: int, empty, episodes=("episode_000000",)) -> dict:
    return {
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "estimator": {"box_threshold": 0.15, "text_threshold": 0.25},
        "per_episode": {
            key: [
                {"frame_index": i, "mask_px": 0 if i in empty else 12000,
                 "raw_max": 0.4, "n_boxes_kept": 0 if i in empty else 1,
                 "empty_reason": "no_boxes_above_threshold" if i in empty else None}
                for i in range(n_frames)
            ]
            for key in episodes
        },
    }


def test_blind_population_is_every_empty_mask_frame_and_only_those():
    empty = {0, 3, 7, 11}
    population = diag.blind_population(
        _visible_for(12, 6), _detect_for(12, empty), absent_below=1000, present_above=3000)

    assert {r["frame_index"] for r in population} == empty
    assert all(r["mask_px"] == 0 for r in population)
    assert {r["predicate_verdict"] for r in population} == {"absent", "present"}
    assert [r["frame_index"] for r in population] == sorted(empty), "deterministic order"


def test_blind_population_keeps_frames_the_reference_pass_never_measured():
    """The uniform arm is over ALL empty-mask frames; dropping the unmeasured ones would bias it."""
    visible = _visible_for(4, 2)
    detect = _detect_for(8, set(range(8)))
    population = diag.blind_population(visible, detect, absent_below=1000, present_above=3000)

    assert len(population) == 8
    unmeasured = [r for r in population if r["frame_index"] >= 4]
    assert len(unmeasured) == 4
    assert {r["predicate_verdict"] for r in unmeasured} == {"unmeasured"}
    assert all(r["predicate_px"] is None for r in unmeasured)


def _population(n: int = 30, present_every: int = 5) -> list[dict]:
    return [
        {"episode": "episode_%06d" % (i % 3), "frame_index": i, "mask_px": 0,
         "empty_reason": "no_boxes_above_threshold",
         "predicate_verdict": "present" if i % present_every == 0 else "absent",
         "predicate_px": 9000 if i % present_every == 0 else 10,
         "blind_spot_px": i * 10}
        for i in range(n)
    ]


def test_draw_blind_arms_is_reproducible_under_the_seed():
    first = diag.draw_blind_arms(_population(), seed=4242, n_uniform=6, n_predicate=3, n_blind_spot=3)
    again = diag.draw_blind_arms(_population(), seed=4242, n_uniform=6, n_predicate=3, n_blind_spot=3)
    assert [(r["tile"], r["episode"], r["frame_index"], r["arm"]) for r in first] == [
        (r["tile"], r["episode"], r["frame_index"], r["arm"]) for r in again]
    assert [r["tile"] for r in first] == sorted(r["tile"] for r in first)


def test_the_uniform_arm_is_drawn_independently_of_the_predicate():
    """The whole reason this arm exists: it must not know what the predicate or the blind spot said.

    Scrambling both fields must leave the uniform draw untouched. An arm that consulted either
    would be a third nominated sample and would bound nothing.
    """
    population = _population()
    scrambled = [
        row | {"predicate_verdict": "absent", "predicate_px": 0, "blind_spot_px": 0}
        for row in population
    ]
    kw = {"seed": 11, "n_uniform": 8, "n_predicate": 0, "n_blind_spot": 0}

    def drawn(rows):
        return {(r["episode"], r["frame_index"]) for r in diag.draw_blind_arms(rows, **kw)}

    assert drawn(population) == drawn(scrambled)
    assert len(drawn(population)) == 8


def test_the_blind_spot_arm_over_weights_the_frames_the_predicate_cannot_score():
    drawn = diag.draw_blind_arms(
        _population(), seed=1, n_uniform=0, n_predicate=0, n_blind_spot=3)
    assert {r["arm"] for r in drawn} == {"blind_spot_targeted"}
    assert sorted(r["frame_index"] for r in drawn) == [27, 28, 29]


def test_the_predicate_arm_is_the_cell_the_existing_sheets_already_nominate():
    drawn = diag.draw_blind_arms(
        _population(), seed=1, n_uniform=0, n_predicate=4, n_blind_spot=0)
    assert {r["arm"] for r in drawn} == {"predicate_nominated"}
    assert all(r["predicate_verdict"] == "present" for r in drawn)


def test_no_frame_is_drawn_into_two_arms():
    drawn = diag.draw_blind_arms(
        _population(), seed=99, n_uniform=6, n_predicate=4, n_blind_spot=4)
    keys = [(r["episode"], r["frame_index"]) for r in drawn]
    assert len(keys) == len(set(keys)) == 14
    assert len({r["tile"] for r in drawn}) == 14


def test_a_frame_with_no_blind_spot_measurement_is_a_refusal():
    population = [{k: v for k, v in row.items() if k != "blind_spot_px"} for row in _population(4)]
    with pytest.raises(diag.DiagnosisError):
        diag.draw_blind_arms(population, seed=1, n_uniform=1, n_predicate=0, n_blind_spot=1)


# -- the sheets, and what they must not show -------------------------------------------------------


@pytest.fixture()
def blind_run(tmp_path, monkeypatch):
    """``blind-sheet`` over two synthetic episodes, with the decoder faked and no GPU anywhere."""
    n = 24
    empty = set(range(n)) - {0, 6, 12, 18}

    def _frames(_path):
        out = []
        for i in range(n):
            frame = scene()
            if i % 3 == 1:
                _arm(frame)
            elif i % 3 == 2:
                _wrist(frame)
            out.append(frame)
        return np.stack(out)

    monkeypatch.setattr(diag, "decode", _frames)
    episodes = ("episode_000000", "episode_000001")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [
        {"id": key, "video": f"{key}.mp4"} for key in episodes]}))
    vis = tmp_path / "v.json"
    vis.write_text(json.dumps(_visible_for(n, 12, episodes)))
    det = tmp_path / "d.json"
    det.write_text(json.dumps(_detect_for(n, empty, episodes)))
    out_dir = tmp_path / "blind"

    def run(*extra, seed="4242"):
        return diag.main([
            "blind-sheet", "--manifest", str(manifest), "--visible", str(vis),
            "--detect", str(det), "--out-dir", str(out_dir), "--seed", seed,
            "--absent-below", "1000", "--present-above", "3000",
            "--n-uniform", "6", "--n-predicate", "3", "--n-blind-spot", "3",
            "--per-sheet", "4", *extra,
        ])

    return types.SimpleNamespace(run=run, out_dir=out_dir, manifest=manifest, visible=vis,
                                 detect=det, n_frames=n, empty=empty, episodes=episodes)


def test_blind_sheet_writes_the_sheets_the_key_and_a_blank_template(blind_run):
    assert blind_run.run() == 0

    key = json.loads((blind_run.out_dir / "BLIND_KEY.json").read_text())
    labels = json.loads((blind_run.out_dir / "BLIND_LABELS.template.json").read_text())
    sheets = sorted(blind_run.out_dir.glob("blind_sheet_*.png"))

    assert len(sheets) == 3, [p.name for p in sheets]
    assert key["seed"] == 4242
    assert set(key["tiles"]) == set(labels["tiles"]) and len(key["tiles"]) == 12
    assert {v["arm"] for v in key["tiles"].values()} == {
        "uniform_random", "predicate_nominated", "blind_spot_targeted"}
    assert all(v["mask_px"] == 0 for v in key["tiles"].values())
    assert all(v["frame_index"] in blind_run.empty for v in key["tiles"].values())
    assert all(row["label"] is None and row["note"] == "" for row in labels["tiles"].values())
    assert labels["label_values"] == list(diag.BLIND_LABEL_VALUES)


def test_the_same_seed_redraws_the_same_sheet(blind_run):
    assert blind_run.run() == 0
    first = json.loads((blind_run.out_dir / "BLIND_KEY.json").read_text())["tiles"]
    assert blind_run.run() == 0
    assert json.loads((blind_run.out_dir / "BLIND_KEY.json").read_text())["tiles"] == first


def test_the_tiles_carry_the_pixels_and_the_tile_id_and_nothing_else(blind_run, monkeypatch):
    """BLIND, asserted at the seam where a leak would be drawn.

    Every tile is the SOURCE frame unmodified plus its opaque id: no mask overlay, no box, no
    score, no arm, no episode and no frame number. A reviewer who could tell the arms apart would
    be back to labelling frames the masker nominated, which is the thing this instrument exists to
    stop doing.
    """
    import audit_apple_masks  # noqa: PLC0415

    seen = []
    original = audit_apple_masks.captioned
    monkeypatch.setattr(
        audit_apple_masks, "captioned",
        lambda arr, lines, **kw: seen.append((np.array(arr), list(lines))) or original(arr, lines, **kw),
    )
    assert blind_run.run() == 0

    key = json.loads((blind_run.out_dir / "BLIND_KEY.json").read_text())
    assert len(seen) == 12
    for arr, lines in seen:
        assert lines in [[tile] for tile in key["tiles"]], lines
        tile = lines[0]
        assert np.array_equal(arr, np.asarray(diag.decode(None))[key["tiles"][tile]["frame_index"]])
    assert [lines[0] for _arr, lines in seen] == sorted(key["tiles"])


def test_the_sheet_title_names_no_arm_and_no_frame(blind_run):
    title = diag.blind_sheet_title(1, 3)
    lowered = title.lower()
    for leak in ("uniform", "predicate", "blind_spot", "episode", "mask_px", "empty", "seed"):
        assert leak not in lowered, title
    assert "arm_present" in title and "arm_absent" in title and "undecidable" in title


def test_the_template_reveals_nothing_and_carries_the_correlated_observer_warning(blind_run):
    assert blind_run.run() == 0
    raw = (blind_run.out_dir / "BLIND_LABELS.template.json").read_text()
    labels = json.loads(raw)

    for leak in ("episode_", "uniform_random", "predicate_nominated", "blind_spot_targeted",
                 "mask_px", "empty_reason", "frame_index"):
        assert leak not in raw, leak

    warning = labels["established_by_note"]
    assert "MODEL FILLING THIS IN IS NOT THE MEASUREMENT" in warning
    assert "CORRELATED OBSERVER" in warning
    assert "audit_apple_masks.py" in warning, "the wording is copied; say where from"
    assert labels["human_review"]["looked_at"] is False


def test_blind_sheet_refuses_a_corpus_with_no_empty_masks(blind_run, tmp_path, capsys):
    det = tmp_path / "none.json"
    det.write_text(json.dumps(_detect_for(24, set(), blind_run.episodes)))
    rc_code = diag.main([
        "blind-sheet", "--manifest", str(blind_run.manifest), "--visible", str(blind_run.visible),
        "--detect", str(det), "--out-dir", str(tmp_path / "x"), "--seed", "1",
        "--absent-below", "1000", "--present-above", "3000",
    ])
    assert rc_code == 1
    assert "REFUSED" in capsys.readouterr().err


# -- scoring ---------------------------------------------------------------------------------------


def _fill(out_dir: pathlib.Path, decide, path_name: str = "BLIND_LABELS.json") -> pathlib.Path:
    key = json.loads((out_dir / "BLIND_KEY.json").read_text())
    labels = json.loads((out_dir / "BLIND_LABELS.template.json").read_text())
    for tile in labels["tiles"]:
        labels["tiles"][tile]["label"] = decide(tile, key["tiles"][tile])
    labels["human_review"] = dict(labels["human_review"], looked_at=True, established_by="a test")
    path = out_dir / path_name
    path.write_text(json.dumps(labels))
    return path


def test_blind_score_reports_the_uniform_arm_as_the_unbiased_estimate(blind_run, tmp_path, capsys):
    assert blind_run.run() == 0
    labels = _fill(blind_run.out_dir, lambda tile, row: "arm_present" if row["arm"] == "blind_spot_targeted" else "arm_absent")
    out = tmp_path / "score.json"

    assert diag.main([
        "blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
        "--labels", str(labels), "--out", str(out)]) == 0
    report = json.loads(out.read_text())

    uniform = report["per_arm"]["uniform_random"]
    assert uniform["n"] == 6
    assert uniform["a_robot_absent"] == 6 and uniform["b_robot_present_mask_empty"] == 0
    assert "UNBIASED" in uniform["estimate"]
    lo, hi = uniform["b_rate_ci95_wilson"]
    assert lo == 0.0 and 0.0 < hi < 0.5

    targeted = report["per_arm"]["blind_spot_targeted"]
    assert targeted["b_robot_present_mask_empty"] == 3
    assert "BIASED" in targeted["estimate"]
    assert "BIASED" in report["per_arm"]["predicate_nominated"]["estimate"]
    capsys.readouterr()


def test_blind_score_counts_undecidable_separately_and_bounds_it(blind_run, tmp_path):
    assert blind_run.run() == 0
    labels = _fill(blind_run.out_dir, lambda tile, row: "undecidable" if row["arm"] == "uniform_random" else "arm_absent")
    out = tmp_path / "score.json"
    assert diag.main(["blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
                      "--labels", str(labels), "--out", str(out)]) == 0
    uniform = json.loads(out.read_text())["per_arm"]["uniform_random"]

    assert uniform["undecidable"] == 6
    assert uniform["b_rate_of_decided"] is None, "no decided frames means no rate, not zero"
    assert uniform["b_rate_upper_bound_counting_undecidable_as_b"] == 1.0


def test_blind_score_refuses_a_labels_file_that_is_still_blank(blind_run, tmp_path, capsys):
    assert blind_run.run() == 0
    labels = _fill(blind_run.out_dir, lambda tile, row: None if tile == min(
        json.loads((blind_run.out_dir / "BLIND_KEY.json").read_text())["tiles"]) else "arm_absent")
    assert diag.main(["blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
                      "--labels", str(labels), "--out", str(tmp_path / "s.json")]) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and "blank" in err.lower()


def test_blind_score_refuses_an_unknown_label(blind_run, tmp_path, capsys):
    assert blind_run.run() == 0
    labels = _fill(blind_run.out_dir, lambda tile, row: "probably_an_arm")
    assert diag.main(["blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
                      "--labels", str(labels), "--out", str(tmp_path / "s.json")]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_blind_score_refuses_when_the_key_and_the_labels_disagree_on_tiles(blind_run, tmp_path, capsys):
    assert blind_run.run() == 0
    path = _fill(blind_run.out_dir, lambda tile, row: "arm_absent")
    payload = json.loads(path.read_text())
    payload["tiles"].pop(sorted(payload["tiles"])[0])
    payload["tiles"]["t9999"] = {"label": "arm_absent", "note": ""}
    path.write_text(json.dumps(payload))

    assert diag.main(["blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
                      "--labels", str(path), "--out", str(tmp_path / "s.json")]) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and "t9999" in err


def test_blind_score_writes_a_measurement_and_never_a_verdict(blind_run, tmp_path):
    assert blind_run.run() == 0
    labels = _fill(blind_run.out_dir, lambda tile, row: "arm_absent")
    out = tmp_path / "score.json"
    assert diag.main(["blind-score", "--key", str(blind_run.out_dir / "BLIND_KEY.json"),
                      "--labels", str(labels), "--out", str(out)]) == 0
    raw = out.read_text()

    assert "measurement" in raw.lower()
    for forbidden in ("discharged?", "GATE_QUALIFIED", "verdicts?", "signed", "licensed",
                      "qualified", "passes", "cleared"):
        assert not re.search(rf"\b{forbidden}\b", raw, re.IGNORECASE), forbidden


def test_wilson_interval_brackets_the_point_estimate_and_never_collapses_at_zero():
    lo, hi = diag.wilson_interval(0, 40)
    assert lo == 0.0 and 0.05 < hi < 0.12, (lo, hi)
    lo, hi = diag.wilson_interval(5, 20)
    assert lo < 0.25 < hi
    assert diag.wilson_interval(0, 0) == (0.0, 1.0)
