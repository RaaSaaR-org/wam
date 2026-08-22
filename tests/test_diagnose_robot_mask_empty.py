"""Tests for scripts/diagnose_robot_mask_empty.py — the pure parts, on synthetic frames.

The two GPU commands are not exercised here (no checkpoints in CI); what IS exercised is every
function whose being wrong would make the diagnosis wrong: the reference predicate's three clauses,
the band that refuses to force a middling frame into a bucket, the run/decile arithmetic that
distinguishes "spread" from "clustered", and the contingency table that is the verdict.
"""

from __future__ import annotations

import json
import pathlib
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
