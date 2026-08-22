"""Tests for ``scripts/audit_apple_masks.py`` — the evidence PR-08 §4's blockers 1 and 2 ask for.

The thing being built here is a picture a person looks at, so the failures worth pinning are not
"it crashed":

  it discharges the        Producing evidence and deciding that the evidence is sufficient are two
  blocker by running       different acts, and only the second one is a judgement. A script that
                           could flip ``GATE_QUALIFIED`` would discharge blocker 1 by being
                           executed. So the source is asserted to contain no assignment to that
                           name and none to ``GATE_QUALIFICATION_BLOCKERS``, and the artifact is
                           asserted to say, in its own fields, that it discharges nothing.

  it samples uniformly     A uniform sample of this corpus contains, in expectation, none of the
  and calls it spanning    frames blocker 1 names: probe-scan measured 48 frames below 1200 px of
                           visible apple in 154447, all in one episode. So the selection rule is
                           tested on a synthetic episode that HAS an occlusion, a lift-off and a
                           border contact, and it must find all three — and it must be
                           deterministic, because a sample nobody can rebuild is not a sample
                           anybody can argue with.

  it re-implements the     The detection score and the retry are computed inside
  detector to read its     ``apple_sam2._best_box`` and thrown away. Re-running the detector here
  score                    to recover them is a second implementation of upstream's rule that can
                           drift from the one under audit. The recorder WATCHES instead, and the
                           test proves it delegates everything else untouched and that its view
                           agrees with the adapter's own counters.

  a flag reads as a        Every flag is a rule of thumb that cannot tell an apple from a plate.
  verdict                  The tests fix what each one fires on, so "flagged" keeps meaning "look
                           at this" rather than "this is wrong".

Nothing here loads a model, decodes a video or needs the AppleToPlate corpus: the adapter is a stub
whose scripted detections are the answer key, and the corpus is drawn with numpy.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import build_identity_calibration as bic  # noqa: E402
import measure_geom_tol as mgt  # noqa: E402

import audit_apple_masks as aud  # noqa: E402

H, W = 480, 640


# -- synthetic pixels ------------------------------------------------------------------------------


def disc(cx: float, cy: float, r: float, h: int = H, w: int = W) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def scene(apple_xy: tuple[float, float], apple_r: float, *, plate: bool = True) -> np.ndarray:
    """A dark neutral cloth, a bright neutral plate, and one warm saturated apple. RGB uint8."""
    img = np.full((H, W, 3), 90, dtype=np.uint8)          # cloth: neutral, mid luminance
    img[:40, :] = 200                                      # the pale top band
    if plate:
        img[disc(430, 300, 70)] = (235, 235, 235)          # plate: bright, near-neutral
    ax, ay = apple_xy
    img[disc(ax, ay, apple_r)] = (220, 60, 40)             # apple: warm, saturated
    return img


def episode_frames(n: int = 30, *, occlude_at: int = 12, lift_at: int = 8,
                   border_at: int = 26) -> list[np.ndarray]:
    """One synthetic episode carrying, by construction, each hard case the sampler must find."""
    out = []
    for i in range(n):
        x = 200.0 + (0.0 if i < lift_at else 14.0 * (i - lift_at))
        r = 26.0
        if i == occlude_at:
            r = 5.0                                        # ~78 px: an occlusion
        if i == border_at:
            x = 4.0                                        # the apple against the left edge
        out.append(scene((x, 240.0), r))
    return out


def bgr(frames):
    return [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]


# -- the strict apple discriminator is the census's, not a second one ------------------------------


def test_warm_apple_mask_is_the_calibration_discriminator():
    """A frame with one solid warm blob: the audit's predicate and ``apple_mask`` must agree.

    ``apple_mask`` grows one connected component and fills its holes; on a solid disc that is the
    predicate itself. If someone edits either threshold on either side, this fails — which is the
    point, because "occluded" in this artifact has to mean what "occluded" meant in probe_census.
    """
    frame = scene((300.0, 240.0), 30.0)
    ours = aud.warm_apple_mask(frame)
    theirs = bic.apple_mask(frame)
    assert ours.sum() > 1500
    assert np.array_equal(ours, theirs)


def test_warm_apple_mask_does_not_raise_on_a_frame_with_no_apple():
    """``apple_mask`` refuses a frame with no fruit. Here that frame is the case of interest."""
    frame = scene((300.0, 240.0), 1.0)
    with pytest.raises(bic.CalibrationError):
        bic.apple_mask(frame)
    assert aud.warm_apple_mask(frame).sum() < 1500


# -- the sampling rule -----------------------------------------------------------------------------


def test_episode_selection_forces_the_census_episodes_and_spans_the_rest():
    keys = [f"episode_{i:06d}" for i in range(100)]
    chosen, meta = aud.select_episodes(keys, ["episode_000094"], budget=6)
    assert "episode_000094" in chosen
    assert meta["forced"] == ["episode_000094"]
    assert len(chosen) == 6
    # Spanning means both ends, not the head of the list.
    assert chosen[0] == "episode_000000"
    assert chosen[-1] == "episode_000099"


def test_episode_selection_is_deterministic_and_uses_no_rng():
    keys = [f"episode_{i:06d}" for i in range(57)]
    a, _ = aud.select_episodes(keys, ["episode_000013"], budget=9)
    b, _ = aud.select_episodes(keys, ["episode_000013"], budget=9)
    assert a == b
    src = (_REPO_ROOT / "scripts" / "audit_apple_masks.py").read_text()
    assert "import random" not in src
    assert "np.random" not in src


def test_frame_selection_finds_every_hard_case_the_blocker_names():
    frames = episode_frames()
    sc = aud.scan_episode(bgr(frames), "episode_000000")
    assert sc.lift_index is not None, sc.lift_note
    anchors, meta = aud.select_frames(sc, census_frames=(3,), max_anchors=12)
    strata = {a.stratum for a in anchors if a.role == "anchor"}
    assert aud.S_CENSUS in strata
    assert aud.S_OCCLUDED in strata, meta["notes"]
    assert aud.S_BORDER in strata, meta["notes"]
    assert aud.S_GRASP in strata, meta["notes"]
    assert aud.S_SPANNING in strata
    # The occlusion frame itself, not merely something near it.
    assert 12 in [a.frame_index for a in anchors]
    # The grasp window reaches BACKWARDS into the approach, where the box is most ambiguous.
    grasp = sorted(a.frame_index for a in anchors if a.stratum == aud.S_GRASP)
    assert min(grasp) < sc.lift_index


def test_every_hard_anchor_pulls_in_its_neighbour_and_spanning_does_not():
    frames = episode_frames()
    sc = aud.scan_episode(bgr(frames), "episode_000000")
    anchors, _ = aud.select_frames(sc, census_frames=(), max_anchors=12)
    by_index = {a.frame_index: a for a in anchors}
    for a in list(anchors):
        if a.role != "anchor":
            continue
        nxt = a.frame_index + 1
        if a.stratum in aud.PAIRED_STRATA and nxt < sc.n_frames:
            assert nxt in by_index, f"{a.stratum} anchor {a.frame_index} has no neighbour"
        if a.stratum == aud.S_SPANNING and nxt in by_index:
            assert by_index[nxt].pair_of != a.frame_index


def test_a_stratum_the_corpus_does_not_contain_is_reported_and_not_faked():
    """No border contact anywhere -> the stratum is empty, and the notes say how many were found."""
    frames = [scene((300.0, 240.0), 26.0) for _ in range(20)]
    sc = aud.scan_episode(bgr(frames), "episode_000001")
    anchors, meta = aud.select_frames(sc, census_frames=(), max_anchors=12)
    assert meta["notes"][aud.S_BORDER].startswith("0 frame(s)")
    assert not [a for a in anchors if a.stratum == aud.S_BORDER]
    assert sc.lift_index is None
    assert "never left its resting position" in meta["notes"][aud.S_GRASP]


def test_a_stratum_cut_by_the_budget_is_not_reported_as_one_the_corpus_lacks():
    """Opposite findings, identical empty list. The scan's candidates are what separate them."""
    frames = episode_frames()
    sc = aud.scan_episode(bgr(frames), "episode_000000")
    _anchors, meta = aud.select_frames(sc, census_frames=(3,), max_anchors=2, neighbour_offset=0)
    assert meta["dropped_by_budget"], "the budget cut strata and nothing recorded it"
    # The strata that were cut still have candidates, which is what stops them being read as absent.
    for stratum in meta["dropped_by_budget"]:
        assert meta["picked_per_stratum"][stratum]


def test_priority_order_keeps_the_hard_cases_when_the_budget_binds():
    frames = episode_frames()
    sc = aud.scan_episode(bgr(frames), "episode_000000")
    anchors, _ = aud.select_frames(sc, census_frames=(3,), max_anchors=2, neighbour_offset=0)
    assert [a.stratum for a in anchors] == [aud.S_CENSUS, aud.S_OCCLUDED]


# -- triage ----------------------------------------------------------------------------------------


def _rec(**over):
    base = {
        "no_detection": False, "empty_mask": False, "retry_fired": False,
        "retry_recovered": False, "detection_score": 0.6, "mask_area_px": 2000,
        "centroid_step_px": None, "plate_overlap_fraction": 0.0, "warm_apple_px": 2000,
        "warm_apple_iou": 0.9, "recorder_inconsistent": False,
    }
    base.update(over)
    return base


def test_a_plate_sized_mask_is_flagged_and_an_apple_sized_one_is_not():
    t = aud.TriageThresholds()
    assert aud.flag_frame(_rec(), median_warm_px=2000, frame_px=H * W, thresholds=t) == []
    flags = aud.flag_frame(_rec(mask_area_px=40000, plate_overlap_fraction=0.9, warm_apple_iou=0.0),
                           median_warm_px=2000, frame_px=H * W, thresholds=t)
    assert "mask_area_above_band" in flags
    assert "plate_overlap" in flags
    assert "disagrees_with_warm_apple" in flags


def test_the_tabletop_ceiling_fires_independently_of_the_band():
    t = aud.TriageThresholds()
    flags = aud.flag_frame(_rec(mask_area_px=int(0.5 * H * W)),
                           median_warm_px=100000, frame_px=H * W, thresholds=t)
    assert "mask_covers_frame" in flags
    assert "mask_area_above_band" not in flags


def test_the_retry_and_a_weak_score_are_always_visible():
    t = aud.TriageThresholds()
    flags = aud.flag_frame(_rec(retry_fired=True, retry_recovered=True, detection_score=0.11),
                           median_warm_px=2000, frame_px=H * W, thresholds=t)
    assert {"retry_fired", "retry_recovered", "low_score"} <= set(flags)


def test_a_centroid_that_teleports_between_adjacent_frames_is_flagged():
    t = aud.TriageThresholds()
    assert "centroid_jump" in aud.flag_frame(
        _rec(centroid_step_px=90.0), median_warm_px=2000, frame_px=H * W, thresholds=t)
    assert "centroid_jump" not in aud.flag_frame(
        _rec(centroid_step_px=4.0), median_warm_px=2000, frame_px=H * W, thresholds=t)


def test_distribution_reports_the_shape_and_not_only_a_mean():
    d = aud.distribution([0.1, 0.2, 0.3, 0.4, 0.9], bins=4)
    assert d["n"] == 5 and d["median"] == 0.3 and d["max"] == 0.9
    assert sum(d["histogram"]["counts"]) == 5
    assert aud.distribution([])["n"] == 0


# -- the recorder ----------------------------------------------------------------------------------


class FakeProcessor:
    """A GroundingDINO processor stand-in whose post-processing is scripted by the test."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.other_attribute_reads = 0

    def set_script(self, script):
        """Called through the recorder, which delegates method lookups to this object."""
        self.script = list(script)

    def __call__(self, **kwargs):
        return {"input_ids": None}

    @property
    def marker(self):
        self.other_attribute_reads += 1
        return "delegated"

    def post_process_grounded_object_detection(self, outputs, input_ids, *, threshold,
                                               text_threshold, target_sizes=None):
        self.calls += 1
        scores, boxes = self.script.pop(0) if self.script else ([], [])
        return [{"scores": np.asarray(scores, dtype=np.float32),
                 "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4)}]


def test_the_recorder_delegates_everything_it_does_not_intercept():
    inner = FakeProcessor([([0.5], [[1, 2, 3, 4]])])
    rec = aud.RecordingProcessor(inner)
    assert rec.marker == "delegated"
    assert inner.other_attribute_reads == 1
    assert rec(images=None, text="apple.") == {"input_ids": None}


def test_the_recorder_captures_scores_and_thresholds_of_every_post_processing_call():
    inner = FakeProcessor([([], []), ([0.31, 0.12], [[1, 2, 3, 4], [5, 6, 7, 8]])])
    rec = aud.RecordingProcessor(inner)
    rec.post_process_grounded_object_detection(None, None, threshold=0.15, text_threshold=0.25)
    rec.post_process_grounded_object_detection(None, None, threshold=0.10, text_threshold=0.10)
    assert [c["threshold"] for c in rec.calls] == [0.15, 0.10]
    assert rec.calls[0]["scores"] == []
    assert rec.calls[1]["scores"] == pytest.approx([0.31, 0.12], abs=1e-5)


# -- the stub adapter, and the frame driver ----------------------------------------------------------


def make_stub_adapter(script_for_frame):
    """A module shaped like ``estimators.apple_sam2``, whose detections are the test's answer key.

    ``script_for_frame`` is called with the frame counter and returns the list of post-processing
    results the fake detector will produce, most-recent-call-last. Two entries mean the retry fires.
    """
    mod = types.ModuleType("estimators.apple_sam2")
    mod.ESTIMATOR_NAME = "stub-detector+stub-segmenter"
    mod.ESTIMATOR_VERSION = "stub/1"
    mod.ESTIMATOR_CHECKPOINTS = {"detector": "stub/detector@" + "0" * 40}
    mod.SEGMENTER_CONTRACT = {"method_name": mod.ESTIMATOR_NAME, "box_threshold": 0.15}
    mod.GATE_QUALIFIED = False
    mod.GATE_QUALIFICATION_BLOCKERS = ("NOBODY HAS LOOKED AT A MASK.", "the retry is unmeasured")
    mod.GATE_QUALIFICATION_DISCHARGED = ()
    for name in aud.COUNTER_NAMES:
        setattr(mod, name, 0)
    mod._DETECTOR = None
    mod.frame_counter = 0

    def available():
        return True

    def _detector():
        if mod._DETECTOR is None:
            mod._DETECTOR = (FakeProcessor([]), object())
        return mod._DETECTOR

    def estimate_depth(rgb):
        return np.zeros(np.asarray(rgb).shape[:2], dtype=np.float32)

    def segment(rgb):
        """Upstream's control flow, in miniature: one pass, one retry, highest score wins."""
        arr = np.asarray(rgb)
        h, w = arr.shape[:2]
        mod.SEGMENT_CALLS += 1
        processor, _model = _detector()
        processor.set_script(script_for_frame(mod.frame_counter))
        mod.frame_counter += 1
        res = processor.post_process_grounded_object_detection(
            None, None, threshold=0.15, text_threshold=0.25)[0]
        if len(res["scores"]) == 0:
            mod.RETRY_FRAMES += 1
            res = processor.post_process_grounded_object_detection(
                None, None, threshold=0.10, text_threshold=0.10)[0]
            if len(res["scores"]) == 0:
                mod.NO_DETECTION_FRAMES += 1
                return np.zeros((h, w), dtype=bool)
            mod.RETRY_RECOVERED_FRAMES += 1
        box = np.asarray(res["boxes"])[int(np.argmax(np.asarray(res["scores"])))]
        mask = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        if not mask.any():
            mod.EMPTY_MASK_FRAMES += 1
        return mask

    def stats():
        return {
            "estimator_name": mod.ESTIMATOR_NAME,
            "gate_qualified": mod.GATE_QUALIFIED,
            "n_frames_retry_fired": mod.RETRY_FRAMES,
            "n_frames_retry_recovered": mod.RETRY_RECOVERED_FRAMES,
        }

    mod.available = available
    mod._detector = _detector
    mod.segment = segment
    mod.estimate_depth = estimate_depth
    mod.stats = stats
    return mod


def _install(monkeypatch, mod):
    monkeypatch.setitem(sys.modules, "estimators.apple_sam2", mod)
    monkeypatch.setitem(sys.modules, "estimators", types.ModuleType("estimators"))


def test_audit_one_frame_reads_the_score_the_retry_and_the_counters(monkeypatch):
    scripts = {
        0: [([0.72, 0.31], [[100, 100, 140, 140], [0, 0, 10, 10]])],   # clean detection
        1: [([], []), ([0.11], [[300, 200, 340, 240]])],               # retry buys a box
        2: [([], []), ([], [])],                                       # honest all-False
    }
    mod = make_stub_adapter(lambda i: scripts[i])
    _install(monkeypatch, mod)
    method = mgt.sam2_method(40)
    recorder = aud.attach_recorder(mod)
    mask_fn = lambda f: method.mask_fn(f, method)  # noqa: E731
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    clean = aud.audit_one_frame(mod, mask_fn, frame, recorder)
    assert clean.score == pytest.approx(0.72, abs=1e-5)
    assert clean.box == pytest.approx([100, 100, 140, 140])
    assert clean.retry_fired is False and clean.no_detection is False
    assert clean.mask.sum() > 0
    assert clean.recorder_inconsistent is False

    bought = aud.audit_one_frame(mod, mask_fn, frame, recorder)
    assert bought.retry_fired is True and bought.retry_recovered is True
    assert bought.score == pytest.approx(0.11, abs=1e-5)
    assert bought.recorder_inconsistent is False

    honest = aud.audit_one_frame(mod, mask_fn, frame, recorder)
    assert honest.retry_fired is True and honest.retry_recovered is False
    assert honest.no_detection is True and not honest.mask.any()
    assert honest.score is None

    assert mod.RETRY_FRAMES == 2 and mod.RETRY_RECOVERED_FRAMES == 1
    assert mod.NO_DETECTION_FRAMES == 1


def test_the_recorder_is_cross_checked_against_the_adapters_own_counters(monkeypatch):
    """A recorder that saw a box on a frame the adapter counted as no-detection is a DEFECT HERE.

    The counters are the adapter's own account of what it did; the recorder is this file's. When
    they disagree the artifact says so per frame rather than quietly reporting the prettier one.
    """
    mod = make_stub_adapter(lambda i: [([0.5], [[10, 10, 50, 50]])])
    _install(monkeypatch, mod)
    method = mgt.sam2_method(40)
    recorder = aud.attach_recorder(mod)
    mod.NO_DETECTION_FRAMES = 0

    real_segment = mod.segment

    def lying_segment(rgb):
        out = real_segment(rgb)
        mod.NO_DETECTION_FRAMES += 1       # the counter says one thing, the recorder saw another
        return out

    mod.segment = lying_segment
    res = aud.audit_one_frame(mod, lambda f: method.mask_fn(f, method),
                              np.zeros((H, W, 3), dtype=np.uint8), recorder)
    assert res.recorder_inconsistent is True
    assert "no-detection" in res.recorder_note


# -- overlays --------------------------------------------------------------------------------------


def test_the_overlay_actually_draws_the_mask_the_box_and_the_heuristic_apple():
    frame = scene((300.0, 240.0), 30.0)
    mask = disc(300.0, 240.0, 30.0)
    warm = aud.warm_apple_mask(frame)
    shot = aud.composite(frame, mask, warm, [260, 200, 340, 280])
    assert shot.shape == frame.shape and shot.dtype == np.uint8
    # The tint moved the pixels inside the mask, and the outline is on the boundary.
    assert not np.array_equal(shot[mask], frame[mask])
    assert (shot == np.asarray(aud.COLOR_MASK)).all(axis=2).any()
    assert (shot == np.asarray(aud.COLOR_BOX)).all(axis=2).any()


def test_an_empty_mask_still_produces_a_readable_overlay():
    frame = scene((300.0, 240.0), 30.0)
    shot = aud.composite(frame, np.zeros((H, W), dtype=bool), None, None)
    assert np.array_equal(shot, frame)


def test_the_contact_sheet_holds_every_tile_it_was_given():
    tiles = [aud.captioned(np.zeros((60, 80, 3), dtype=np.uint8), ["a", "b"], font_size=10)
             for _ in range(5)]
    sheet = aud.contact_sheet(tiles, "title", cols=3)
    assert sheet.width >= 3 * tiles[0].width
    assert sheet.height >= 2 * tiles[0].height


# -- end to end ------------------------------------------------------------------------------------


def _fake_decoder(clips: dict[str, list[np.ndarray]]):
    def open_fn(path):
        frames = clips[Path(path).stem]
        return (f.copy() for f in frames), 30.0

    return mgt.Decoder(name="fixture", version="test", open_fn=open_fn, note="synthetic")


def _census(tmp: Path, episode: str, frames: list[int]) -> Path:
    doc = {
        "built_utc": "2026-08-22T00:00:00+00:00",
        "rule": {"census_apple_warm_px": bic.NATURAL_PROBE_CENSUS_PX},
        "corpus": {"episodes_scanned": 3, "frames_scanned": 90},
        "eligible_frames": [{"episode": episode, "frame_index": f} for f in frames],
        "per_episode": {episode: {"n_below_census": len(frames)}},
    }
    p = tmp / "probe_census.json"
    p.write_text(json.dumps(doc))
    return p


def test_end_to_end_writes_looking_evidence_and_discharges_nothing(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus" / "videos"
    corpus.mkdir(parents=True)
    clips: dict[str, list[np.ndarray]] = {}
    for i in range(3):
        key = f"episode_{i:06d}"
        (corpus / f"{key}.mp4").write_bytes(b"")
        # BGR, because every decoder in measure_geom_tol.DECODERS yields BGR and the audit's one
        # flip to RGB is what a fixture handing back RGB would silently defeat.
        clips[key] = bgr(episode_frames())
    monkeypatch.setattr(mgt, "DECODERS", {"fixture": _fake_decoder(clips)})

    # Frame 4 of every episode is the one where the retry buys a box on the PLATE. Everything else
    # lands on the apple. The audit must find that frame and flag it without being told.
    plate_box = [360, 230, 500, 370]

    def script(i):
        if i % 7 == 4:
            return [([], []), ([0.12], [plate_box])]
        return [([0.66], [[180, 214, 240, 266]])]

    mod = make_stub_adapter(script)
    _install(monkeypatch, mod)

    out = tmp_path / "out"
    rc = aud.main([
        "--corpus", str(tmp_path / "corpus"),
        "--out", str(out),
        "--census", str(_census(tmp_path, "episode_000001", [12])),
        "--episodes", "3",
        "--max-anchors-per-episode", "6",
        "--decoder", "fixture",
        "--sheet-tiles", "4",
    ])
    assert rc == 0

    doc = json.loads((out / "MASK_AUDIT.json").read_text())
    assert doc["schema"] == aud.SCHEMA
    # It says, in its own fields, that it decides nothing.
    assert doc["estimator"]["blockers_discharged_by_this_run"] == []
    assert doc["estimator"]["gate_qualified_read_from_adapter"] is False
    assert doc["estimator"]["gate_qualification_blockers_verbatim"] == list(
        mod.GATE_QUALIFICATION_BLOCKERS)
    assert "DOES NOT DISCHARGE" in doc["not_a_discharge"]
    assert doc["human_review"]["looked_at"] is False
    assert "CORRELATED OBSERVER" in doc["human_review"]["correlated_observer_warning"]

    # The numbers blocker 2 names, under the names blocker 2 uses.
    b2 = doc["blocker_2_numbers"]
    assert b2["n_frames_retry_fired"] >= 1
    assert b2["n_frames_retry_recovered"] == b2["n_frames_retry_fired"]
    assert b2["detection_score_distribution"]["n"] == doc["counts"]["n_frames_segmented"] - \
        b2["n_frames_without_detection"]
    assert b2["mask_area_px_distribution"]["n"] >= 1
    assert "Not a corpus rate" in b2["over"]

    # The sample is stated, not implied.
    assert doc["sampling"]["rng_used"] is False
    assert "over-weights" in doc["sampling"]["bias"]
    assert doc["sampling"]["census"]["used"] is True
    assert "episode_000001" in doc["sampling"]["episode_selection"]["forced"]

    # The plate-sized mask the retry bought was found and flagged, by the rule and not by a fixture.
    flagged = [f for f in doc["frames"] if f["flags"]]
    assert flagged, "the planted wrong-object frame was not flagged"
    assert any("retry_fired" in f["flags"] and
               ("mask_area_above_band" in f["flags"] or "plate_overlap" in f["flags"] or
                "disagrees_with_warm_apple" in f["flags"])
               for f in flagged)

    # And the pictures a person is supposed to look at exist.
    pngs = sorted((out / "frames").glob("*.png"))
    assert len(pngs) == doc["counts"]["n_frames_segmented"]
    assert doc["human_review"]["contact_sheets"]
    assert any(s.startswith("sheets/flagged-") for s in doc["human_review"]["contact_sheets"])
    for rel in doc["human_review"]["contact_sheets"]:
        assert (out / rel).is_file()

    template = json.loads((out / "OBSERVATIONS.template.json").read_text())
    assert len(template["frames"]) == doc["counts"]["n_frames_segmented"]
    assert all(f["looked_at"] is False and f["verdict"] is None for f in template["frames"])


def test_no_census_and_no_waiver_is_a_refusal_not_a_quieter_sample(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus" / "videos"
    corpus.mkdir(parents=True)
    (corpus / "episode_000000.mp4").write_bytes(b"")
    monkeypatch.setattr(mgt, "DECODERS", {"fixture": _fake_decoder({"episode_000000": []})})
    rc = aud.main(["--corpus", str(tmp_path / "corpus"), "--out", str(tmp_path / "out"),
                   "--decoder", "fixture"])
    assert rc == 2
    assert not (tmp_path / "out" / "MASK_AUDIT.json").exists()


def test_a_missing_census_file_is_refused_with_the_reason(tmp_path):
    with pytest.raises(aud.AuditError) as exc:
        aud.load_census(tmp_path / "nope.json")
    assert "probe-scan" in str(exc.value)
    assert "--allow-missing-census" in str(exc.value)


# -- the guard that makes the rest of it mean anything ------------------------------------------------


def test_this_script_can_never_flip_the_gate_or_edit_the_blockers():
    """Blocker 1 is discharged by a person looking, not by a script running.

    Read as a source-text assertion rather than a behavioural one on purpose: a behavioural test
    proves that today's code path does not write the flag, and the thing worth preventing is the
    edit that adds one tomorrow.
    """
    src = (_REPO_ROOT / "scripts" / "audit_apple_masks.py").read_text()
    # An ASSIGNMENT — at the head of a line, or through an attribute. Prose in the docstring that
    # quotes the adapter's own ``GATE_QUALIFIED = False`` is not one, and must stay quotable.
    for name in ("GATE_QUALIFIED", "GATE_QUALIFICATION_BLOCKERS", "GATE_QUALIFICATION_DISCHARGED"):
        assert re.search(rf"^\s*[\w.]*{name}\s*=[^=]", src, re.MULTILINE) is None, name
    assert "setattr(" not in src
    # It reads the adapter; it never opens the adapter's file.
    assert "apple_sam2.py" not in src.replace("scripts/estimators/apple_sam2.py", "")


def test_the_adapter_itself_still_declares_the_blockers_this_audit_addresses():
    """If blocker 1's wording moves, this artifact's `addresses` line is stale and must move too."""
    src = (_REPO_ROOT / "scripts" / "estimators" / "apple_sam2.py").read_text()
    assert "GATE_QUALIFIED = False" in src
    assert "NOBODY HAS LOOKED AT A MASK" in src
    assert "n_frames_retry_fired / n_frames_retry_recovered" in src
