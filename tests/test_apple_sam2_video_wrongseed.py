"""T40_RULE_V19 §3 — the wrong-seed control: what it changes, and what it must leave alone.

No weights: the base module's ``propagate`` is stubbed, because the thing under test is the seed
swap and its restoration, not SAM 2.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from estimators import apple_sam2_video as base  # noqa: E402
from estimators import apple_sam2_video_wrongseed as ws  # noqa: E402


@pytest.fixture
def capture(tmp_path):
    frame0 = tmp_path / "frames" / "000000"
    frame0.mkdir(parents=True)
    seg = np.zeros((32, 32), dtype=np.uint32)
    seg[10:20, 4:14] = 107  # cube
    seg[2:6, 24:30] = 108  # apple
    np.save(frame0 / "seg_ids.npy", seg)
    (frame0 / "seg_labels.json").write_text(
        json.dumps({"107": {"class": "cube"}, "108": {"class": "apple"}})
    )
    return tmp_path


def test_the_seed_is_the_ground_truth_bbox_and_not_a_detection(capture):
    """THE PROPERTY THAT MAKES IT A CONTROL. A seed from GroundingDINO could fail because the
    detector had a bad day, which is uninformative in exactly the direction that matters."""
    box = ws.seed_box_from_capture(capture, "cube")
    assert list(box) == [4.0, 10.0, 13.0, 19.0]


def test_a_label_the_render_does_not_have_is_refused(capture):
    with pytest.raises(ValueError, match="no geom labelled 'plate'"):
        ws.seed_box_from_capture(capture, "plate")


def test_a_named_but_invisible_geom_is_refused_rather_than_seeded_on_an_empty_box(capture, tmp_path):
    frame0 = capture / "frames" / "000000"
    (frame0 / "seg_labels.json").write_text(
        json.dumps({"107": {"class": "cube"}, "109": {"class": "ghost"}})
    )
    with pytest.raises(ValueError, match="covers no pixel"):
        ws.seed_box_from_capture(capture, "ghost")


def test_without_the_capture_env_it_refuses_rather_than_guessing(monkeypatch):
    monkeypatch.delenv(ws.CAPTURE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="is not set"):
        ws._resolve_seed()


def test_the_real_propagate_is_called_with_the_swapped_seed(capture, monkeypatch):
    """A REIMPLEMENTATION HERE WOULD BE THE BUG. If this module copied propagate, a control that
    fired would be evidence about a different propagator than the one under test."""
    monkeypatch.setenv(ws.CAPTURE_ENV, str(capture))
    monkeypatch.setenv(ws.LABEL_ENV, "cube")
    seen = {}

    def _fake_propagate(rgbs):
        seen["seed"] = list(base.seed_box(rgbs[0]))
        return [np.zeros((4, 4), dtype=bool) for _ in rgbs]

    monkeypatch.setattr(base, "propagate", _fake_propagate)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    ws.propagate(frames)
    assert seen["seed"] == [4.0, 10.0, 13.0, 19.0]


def test_the_seed_is_put_back_even_when_propagation_raises(capture, monkeypatch):
    """A module left permanently patched would turn the MEASURED arm into a control the next time
    anything imported it."""
    monkeypatch.setenv(ws.CAPTURE_ENV, str(capture))
    original = base.seed_box

    def _boom(rgbs):
        raise RuntimeError("boom")

    monkeypatch.setattr(base, "propagate", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        ws.propagate([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert base.seed_box is original


def test_the_measurement_path_grows_no_seed_override():
    """V19 §3: apple_sam2_video's own seed frame and its recorded reason are untouched."""
    assert base.SEED_FRAME_INDEX == 0
    src = pathlib.Path(base.__file__).read_text()
    assert "a sweep over seed frames would be a different experiment" in src
    assert ws.CAPTURE_ENV not in src, "the control's env must not leak into the measured arm"


def test_the_artifact_says_its_est_drift_is_meaningless(capture, monkeypatch):
    """measure_est_drift will happily write est_drift_p95_px from a control run. That number is the
    distance from the cube to the apple, and the sentence saying so has to travel with the file."""
    monkeypatch.setenv(ws.CAPTURE_ENV, str(capture))
    contract = ws.PROPAGATION_CONTRACT
    assert contract["role"] == "POSITIVE CONTROL, NOT AN ARM"
    assert "never be quoted as one" in contract["est_drift_is_meaningless_here"]
    assert "cannot show" in " ".join(contract).lower() or "what_it_cannot_show" in contract
    assert "subtle" in contract["what_it_cannot_show"].lower()


def test_stats_carries_the_seed_that_was_actually_used(capture, monkeypatch):
    monkeypatch.setenv(ws.CAPTURE_ENV, str(capture))
    monkeypatch.setenv(ws.LABEL_ENV, "cube")
    monkeypatch.setattr(base, "propagate", lambda rgbs: [np.zeros((4, 4), dtype=bool)])
    ws.propagate([np.zeros((4, 4, 3), dtype=np.uint8)])
    got = ws.stats()
    assert got["control_seed_box_xyxy"] == [4.0, 10.0, 13.0, 19.0]
    assert got["control_seed_label"] == "cube"
    assert got["control"]["rule"] == "T40_RULE_V19 §3"


def test_reset_clears_the_recorded_seed(capture, monkeypatch):
    monkeypatch.setenv(ws.CAPTURE_ENV, str(capture))
    monkeypatch.setattr(base, "propagate", lambda rgbs: [np.zeros((4, 4), dtype=bool)])
    ws.propagate([np.zeros((4, 4, 3), dtype=np.uint8)])
    ws.reset_counters()
    assert ws.stats()["control_seed_box_xyxy"] is None
