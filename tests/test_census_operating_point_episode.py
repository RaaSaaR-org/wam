"""T40_RULE_V18 — the census that tells three all-False masks apart.

No weights and no corpus: a stub adapter whose counters move exactly the way ``apple_sam2``'s do.
The thing under test is the delta-reading, because a total says how MANY frames were refused and
residue (i) turns on WHICH.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import census_operating_point_episode as cen  # noqa: E402


class _StubAdapter:
    """Counters that move like the real module's, and nothing else."""

    def __init__(self, script):
        self.script = script  # per frame: "mask" | "refused" | "no_detection" | "empty"
        self.i = 0
        self._counts = dict.fromkeys(cen.EVENT_COUNTERS, 0)

    def segment(self, rgb):
        what = self.script[self.i]
        self.i += 1
        if what == "refused":
            self._counts["n_frames_mask_refused"] += 1
            return np.zeros((8, 8), dtype=bool)
        if what == "no_detection":
            self._counts["n_frames_without_detection"] += 1
            return np.zeros((8, 8), dtype=bool)
        if what == "empty":
            self._counts["n_frames_with_empty_mask"] += 1
            return np.zeros((8, 8), dtype=bool)
        m = np.zeros((8, 8), dtype=bool)
        m[:2, :2] = True
        return m

    def stats(self):
        return dict(self._counts)

    def object_color_reference(self, rgb):
        r = np.zeros((8, 8), dtype=bool)
        r[:2, :2] = True
        return r

    def reference_frame_fraction(self, reference):
        return float(reference.sum()) / float(reference.size)

    def reference_is_object_scale(self, reference):
        return True

    def mask_validity_iou(self, mask, reference):
        union = int(np.count_nonzero(mask | reference))
        return 0.0 if union == 0 else float(np.count_nonzero(mask & reference)) / union


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "episode_000094.mp4").write_bytes(b"stub")
    return tmp_path


def _decode_n(n):
    def _fake(path):
        for i in range(n):
            yield np.full((8, 8, 3), i % 255, dtype=np.uint8)

    return _fake


def test_the_three_all_false_events_are_told_apart(corpus, monkeypatch):
    """THE POINT OF THE SCRIPT. segment() returns the same all-False array for no detection, an
    empty mask from a real box, and a mask refused as the wrong object — and residue (i) is about
    exactly the third."""
    script = ["mask", "no_detection", "refused", "empty", "mask"]
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(len(script)))
    got = cen.census("episode_000094", corpus, _StubAdapter(script))
    assert got["refused_frames"] == [2]
    assert got["no_detection_frames"] == [1]
    assert got["empty_mask_frames"] == [3]
    assert got["n_frames_with_mask"] == 2


def test_a_contiguous_run_of_refusals_is_reported_as_contiguous(corpus, monkeypatch):
    script = ["mask"] * 5 + ["refused"] * 6 + ["mask"] * 4
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(len(script)))
    got = cen.census("episode_000094", corpus, _StubAdapter(script))
    assert got["n_refused"] == 6
    assert got["refused_span"] == [5, 10]
    assert got["refused_is_contiguous"] is True


def test_scattered_refusals_are_not_reported_as_contiguous(corpus, monkeypatch):
    script = ["refused", "mask", "refused", "mask", "refused"]
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(len(script)))
    got = cen.census("episode_000094", corpus, _StubAdapter(script))
    assert got["refused_frames"] == [0, 2, 4]
    assert got["refused_is_contiguous"] is False


def test_an_episode_with_no_refusals_reports_none_rather_than_a_span(corpus, monkeypatch):
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(4))
    got = cen.census("episode_000094", corpus, _StubAdapter(["mask"] * 4))
    assert got["n_refused"] == 0
    assert got["refused_span"] is None
    assert got["refused_is_contiguous"] is False


def test_a_refused_frame_carries_no_validity_iou_because_it_carries_no_mask(corpus, monkeypatch):
    script = ["refused", "mask"]
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(2))
    got = cen.census("episode_000094", corpus, _StubAdapter(script))
    assert got["frames"][0]["mask_validity_iou"] is None
    assert got["frames"][1]["mask_validity_iou"] == 1.0


def test_a_missing_episode_is_refused_rather_than_measured_as_empty(tmp_path):
    (tmp_path / "videos").mkdir()
    with pytest.raises(SystemExit, match="does not exist"):
        cen.census("episode_999999", tmp_path, _StubAdapter([]))


def test_the_counters_watched_are_the_adapters_own():
    """If apple_sam2 ever grows a fourth way to return all-False, this census would silently stop
    classifying it — so the names are pinned against the real module."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from estimators import apple_sam2 as est

    stats = est.stats()
    for counter in cen.EVENT_COUNTERS:
        assert counter in stats, counter


def test_the_script_says_it_decides_nothing():
    src = pathlib.Path(cen.__file__).read_text()
    assert "It does not decide anything" in src
    assert "V18" in cen.WRITEUP
    assert (pathlib.Path(cen.__file__).resolve().parents[1] / cen.WRITEUP).is_file()


def test_json_round_trips(corpus, monkeypatch):
    monkeypatch.setattr(cen, "_decode_frames", _decode_n(3))
    got = cen.census("episode_000094", corpus, _StubAdapter(["mask", "refused", "mask"]))
    assert json.loads(json.dumps(got))["n_refused"] == 1
