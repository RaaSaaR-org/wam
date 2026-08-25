"""Tests for ``scripts/render_area_tail_sheet.py`` — the PR-08 V13 §3.2 tail look.

The thing being built is a picture a person looks at in order to make a decision this script must
not make, so the failures worth pinning are not "it crashed":

  it decides the bound      V13 §3.2 asks a rationale to say whether the tail frames were looked
  by running               at. A script that could also write the number, or record that a review
                           happened, would turn "produce the evidence" into "make the decision" —
                           and it would do it by being executed. So the source is asserted to
                           contain no assignment of a bound and no reviewer-confirmation flag of
                           any name, and the artifact is asserted to say in its own fields that it
                           writes no bound and discharges nothing.

  the sample cannot be     A sample nobody can rebuild is not a sample anybody can argue with. The
  rebuilt                  selection is asserted to be a pure function of (pooled artifact,
                           threshold, budget): same input twice, same frames, in the same order,
                           with no RNG anywhere in the module.

  the sheets are one       Twelve tiles from the worst episode would show the tail of that episode
  episode                  and be labelled the tail of the corpus. The round-robin is asserted to
                           spread the budget across every episode before it takes a second frame
                           from any of them.

  a truncated measurement  ``measurement_qualified: false`` is the stamp a --limit/--stride
  becomes the picture      shakedown carries and ``load_area_bound`` refuses it by name. Rendering
                           sheets from one would put a picture of three episodes in front of the
                           person deciding a bound over 402. Asserted to refuse, non-zero.

  a caption hides half     Each tile exists to compare an H200 measurement against an RTX 5090
  the comparison           re-render. A caption carrying only one of the two fractions would look
                           identical and prove nothing, so the caption is asserted to carry both.

  the band silently        --max-fraction closes the top of the selection window. If it were
  becomes a bound          recorded as anything other than a selection window, or if the artifact
                           named only the lower edge, a reader would be looking at a band and
                           reading a threshold. The band is asserted to appear in the artifact with
                           BOTH edges and with the statement that neither is a bound, and the
                           default is asserted to leave the open upper tail exactly as it was.

Nothing here loads a model, decodes a video or needs the AppleToPlate corpus: the masker is a stub
whose masks are the answer key and the "clips" are drawn with numpy.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import render_area_tail_sheet as rats  # noqa: E402

SOURCE = (REPO / "scripts" / "render_area_tail_sheet.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------
# fixtures — a pooled artifact and a corpus, both synthetic
# --------------------------------------------------------------------------------------------


def _pooled(episodes: dict[str, list[float]], *, qualified: bool = True) -> dict:
    return {
        "git_commit": "0" * 40,
        "source_manifest_sha256": "f" * 64,
        "prompt": "robot arm. robotic hand. robotic gripper.",
        "estimator": {"adapter": "estimators.apple_sam2"},
        "measurement_qualified": qualified,
        "per_episode": [
            {
                "episode_index": i,
                "episode": name,
                "n_frames": len(fractions),
                "empty_frames": sum(1 for f in fractions if f == 0.0),
                "area_fractions": list(fractions),
            }
            for i, (name, fractions) in enumerate(sorted(episodes.items()))
        ],
    }


def _write_pooled(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "POOLED.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _corpus(tmp_path: Path, pooled: dict) -> Path:
    """A manifest matching the pooled artifact, plus a stub clip per episode. No video is written.

    ``render`` is handed the decoder, so the "video" only has to exist as a path the manifest names.
    """
    root = tmp_path / "corpus"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    episodes = []
    for record in pooled["per_episode"]:
        name = record["episode"]
        (root / "videos" / f"{name}.mp4").write_bytes(b"")
        episodes.append({"id": name, "frames": record["n_frames"],
                         "video": f"videos/{name}.mp4"})
    (root / "manifest.json").write_text(
        json.dumps({"resolution": [640, 480], "fps": 30, "episodes": episodes}), encoding="utf-8")
    return root


class StubMasker:
    """Masks a fixed fraction of the frame, taken from the frame's own top-left pixel.

    The recomputed fraction is then something the test controls exactly, which is what lets the
    mismatch flag be tested without a GPU and without pretending a stub is a segmenter.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        wanted = float(rgb[0, 0, 0]) / 255.0
        flat = np.zeros(h * w, dtype=bool)
        flat[: int(round(wanted * h * w))] = True
        return flat.reshape(h, w)

    def provenance(self) -> dict:
        return {"version": "stub"}

    def filter_record(self) -> dict:
        return {"rule": "stub"}


def _decoder(fraction_by_frame: dict[str, list[float]], *, shape=(20, 20)):
    """A ``decode_clip`` stand-in: frame ``i`` is a flat image encoding its own target fraction."""

    def decode(path) -> np.ndarray:
        name = Path(path).stem
        fractions = fraction_by_frame[name]
        h, w = shape
        clip = np.zeros((len(fractions), h, w, 3), dtype=np.uint8)
        for i, f in enumerate(fractions):
            clip[i, :, :, :] = int(round(float(f) * 255.0))
        return clip

    return decode


# --------------------------------------------------------------------------------------------
# 1. the refusal on an unqualified measurement
# --------------------------------------------------------------------------------------------


def test_refuses_unqualified_measurement(tmp_path):
    path = _write_pooled(tmp_path, _pooled({"episode_000000": [0.9]}, qualified=False))
    with pytest.raises(rats.TailLookError) as excinfo:
        rats.load_pooled(path)
    assert "measurement_qualified" in str(excinfo.value)


@pytest.mark.parametrize("value", [1, "true", "True", None, 0, [], {}])
def test_only_the_boolean_true_qualifies(tmp_path, value):
    """``1`` and ``"true"`` are not the stamp ``measure_source_mask_area`` writes."""
    payload = _pooled({"episode_000000": [0.9]})
    payload["measurement_qualified"] = value
    path = _write_pooled(tmp_path, payload)
    with pytest.raises(rats.TailLookError):
        rats.load_pooled(path)


def test_main_exits_non_zero_on_an_unqualified_measurement(tmp_path, capsys):
    pooled = _pooled({"episode_000000": [0.9, 0.9]}, qualified=False)
    path = _write_pooled(tmp_path, pooled)
    corpus = _corpus(tmp_path, pooled)
    code = rats.main(["--pooled", str(path), "--corpus", str(corpus),
                      "--out", str(tmp_path / "out")])
    assert code != 0
    assert "measurement_qualified" in capsys.readouterr().err
    # And nothing was written: a refusal that leaves an artifact behind is not a refusal.
    assert not (tmp_path / "out" / "TAIL_SAMPLE.json").exists()


# --------------------------------------------------------------------------------------------
# 2. selection: deterministic, reproducible, spread
# --------------------------------------------------------------------------------------------


def _demo_pooled() -> dict:
    return _pooled({
        "episode_000001": [0.1, 0.9, 0.2, 0.8, 0.75, 0.7, 0.99],
        "episode_000002": [0.9, 0.0, 0.95],
        "episode_000003": [0.5, 0.5, 0.5],          # nothing in the tail
        "episode_000004": [0.71, 0.72, 0.73, 0.74],
    })


def test_selection_is_deterministic():
    pooled = _demo_pooled()
    first = rats.select_frames(rats.tail_candidates(pooled, 0.7), 6)
    second = rats.select_frames(rats.tail_candidates(pooled, 0.7), 6)
    assert first == second
    # And a fresh parse of the same bytes gives the same answer, so nothing leaks from dict order.
    reparsed = json.loads(json.dumps(pooled))
    assert rats.select_frames(rats.tail_candidates(reparsed, 0.7), 6) == first


def test_selection_uses_no_rng():
    """No RNG is imported or called anywhere in the module. Prose about randomness is fine."""
    for pattern in (r"\bimport\s+random\b", r"\brandom\.", r"\bnp\.random\b",
                    r"\bnumpy\.random\b", r"\bsecrets\.", r"\.shuffle\(", r"\buuid\."):
        assert not re.search(pattern, SOURCE), pattern


def test_threshold_is_inclusive_and_indexes_are_frame_numbers():
    pooled = _pooled({"episode_000000": [0.5, 0.7, 0.69999, 0.8]})
    candidates = rats.tail_candidates(pooled, 0.7)
    assert candidates["episode_000000"] == [(1, 0.7), (3, 0.8)]


def test_episodes_with_no_tail_frame_are_absent():
    candidates = rats.tail_candidates(_demo_pooled(), 0.7)
    assert "episode_000003" not in candidates
    assert list(candidates) == ["episode_000001", "episode_000002", "episode_000004"]


def test_budget_is_spread_across_episodes_before_any_second_frame():
    """Three episodes, budget three: one frame each, never three from the biggest."""
    selected = rats.select_frames(rats.tail_candidates(_demo_pooled(), 0.7), 3)
    assert len(selected) == 3
    assert sorted(r["episode"] for r in selected) == [
        "episode_000001", "episode_000002", "episode_000004"]


def test_budget_smaller_than_episode_count_takes_one_from_each_of_the_first():
    pooled = _pooled({f"episode_{i:06d}": [0.9, 0.91, 0.92] for i in range(10)})
    selected = rats.select_frames(rats.tail_candidates(pooled, 0.7), 4)
    assert [r["episode"] for r in selected] == [f"episode_{i:06d}" for i in range(4)]
    assert {r["frame_index"] for r in selected} == {0}


def test_larger_budget_keeps_the_spread_even():
    pooled = _pooled({
        "episode_a": [0.9] * 100,
        "episode_b": [0.9] * 100,
        "episode_c": [0.9] * 100,
    })
    selected = rats.select_frames(rats.tail_candidates(pooled, 0.7), 12)
    counts = {name: sum(1 for r in selected if r["episode"] == name)
              for name in ("episode_a", "episode_b", "episode_c")}
    assert counts == {"episode_a": 4, "episode_b": 4, "episode_c": 4}


def test_a_short_episode_does_not_starve_the_budget():
    """episode_000002 has two tail frames; the budget rolls on to the others rather than stalling."""
    selected = rats.select_frames(rats.tail_candidates(_demo_pooled(), 0.7), 9)
    counts = {name: sum(1 for r in selected if r["episode"] == name) for name in
              ("episode_000001", "episode_000002", "episode_000004")}
    assert counts["episode_000002"] == 2
    assert sum(counts.values()) == 9


def test_budget_larger_than_the_tail_takes_the_whole_tail_once():
    candidates = rats.tail_candidates(_demo_pooled(), 0.7)
    total = sum(len(v) for v in candidates.values())
    selected = rats.select_frames(candidates, 500)
    assert len(selected) == total
    assert len({(r["episode"], r["frame_index"]) for r in selected}) == total


def test_even_stride_includes_both_endpoints():
    assert rats.even_stride_indices(10, 3) == [0, 5, 9]
    assert rats.even_stride_indices(7, 4) == [0, 2, 4, 6]
    assert rats.even_stride_indices(4, 4) == [0, 1, 2, 3]
    assert rats.even_stride_indices(3, 9) == [0, 1, 2]
    assert rats.even_stride_indices(9, 1) == [0]
    assert rats.even_stride_indices(0, 5) == []


def test_selected_frames_carry_the_recorded_fraction():
    selected = rats.select_frames(rats.tail_candidates(_demo_pooled(), 0.7), 3)
    for record in selected:
        assert set(record) == {"episode", "frame_index", "recorded_fraction"}
        assert record["recorded_fraction"] >= 0.7


def test_zero_budget_is_refused():
    with pytest.raises(rats.TailLookError):
        rats.select_frames(rats.tail_candidates(_demo_pooled(), 0.7), 0)


# --------------------------------------------------------------------------------------------
# 3. the caption carries BOTH fractions
# --------------------------------------------------------------------------------------------


def test_caption_carries_both_fractions_and_the_frame():
    record = {"episode": "episode_000338", "frame_index": 201,
              "recorded_fraction": 0.681234, "recomputed_fraction": 0.680977,
              "delta": -0.000257, "mismatch": False}
    lines = rats.caption_lines(record, tolerance=0.01)
    text = " ".join(lines)
    assert "episode_000338" in text
    assert "00201" in text
    assert "recorded=0.681234" in text
    assert "recomputed=0.680977" in text
    assert "0.681234" in text and "0.680977" in text
    assert not any("MISMATCH" in line for line in lines)


def test_caption_marks_a_mismatch_in_words_as_well_as_in_colour():
    record = {"episode": "episode_000001", "frame_index": 3,
              "recorded_fraction": 0.90, "recomputed_fraction": 0.10,
              "delta": -0.80, "mismatch": True}
    lines = rats.caption_lines(record, tolerance=0.01)
    assert any("MISMATCH" in line for line in lines)
    assert "recorded=0.900000" in " ".join(lines)
    assert "recomputed=0.100000" in " ".join(lines)


# --------------------------------------------------------------------------------------------
# 4. the render: recompute, compare, refuse a mis-decoded clip
# --------------------------------------------------------------------------------------------


def test_render_recomputes_and_flags_only_real_disagreement(tmp_path):
    # episode_a frame 1 is recorded 0.90 and the stub clip encodes 0.90 -> agreement.
    # episode_b frame 0 is recorded 0.90 but the clip encodes 0.20 -> a mismatch.
    pooled = _pooled({"episode_a": [0.1, 0.9], "episode_b": [0.9, 0.1]})
    corpus = _corpus(tmp_path, pooled)
    decode = _decoder({"episode_a": [0.1, 0.9], "episode_b": [0.2, 0.1]})
    selected = [
        {"episode": "episode_a", "frame_index": 1, "recorded_fraction": 0.9},
        {"episode": "episode_b", "frame_index": 0, "recorded_fraction": 0.9},
    ]
    rendered = rats.render(selected, manifest=corpus / "manifest.json", pooled=pooled,
                           masker=StubMasker(), decode_clip=decode, tolerance=0.01,
                           full_frames_dir=None)
    by_episode = {r["episode"]: r for r in rendered}
    assert by_episode["episode_a"]["mismatch"] is False
    assert by_episode["episode_b"]["mismatch"] is True
    assert by_episode["episode_b"]["recomputed_fraction"] == pytest.approx(0.2, abs=0.01)
    assert by_episode["episode_a"]["delta"] == pytest.approx(0.0, abs=0.01)


def test_render_refuses_a_clip_whose_decode_length_differs_from_the_measurement(tmp_path):
    pooled = _pooled({"episode_a": [0.9, 0.9, 0.9]})
    corpus = _corpus(tmp_path, pooled)
    decode = _decoder({"episode_a": [0.9, 0.9]})       # one frame short of the measurement
    with pytest.raises(rats.TailLookError) as excinfo:
        rats.render([{"episode": "episode_a", "frame_index": 0, "recorded_fraction": 0.9}],
                    manifest=corpus / "manifest.json", pooled=pooled, masker=StubMasker(),
                    decode_clip=decode, tolerance=0.01, full_frames_dir=None)
    assert "decodes" in str(excinfo.value)


def test_render_refuses_an_episode_the_corpus_manifest_does_not_list(tmp_path):
    pooled = _pooled({"episode_a": [0.9]})
    corpus = _corpus(tmp_path, pooled)
    manifest = corpus / "manifest.json"
    manifest.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    with pytest.raises(rats.TailLookError):
        rats.render([{"episode": "episode_a", "frame_index": 0, "recorded_fraction": 0.9}],
                    manifest=manifest, pooled=pooled, masker=StubMasker(), decode_clip=_decoder(
                        {"episode_a": [0.9]}), tolerance=0.01, full_frames_dir=None)


def test_overlay_paints_the_mask_and_leaves_the_rest_alone():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    out = rats.overlay(rgb, mask)
    assert out.dtype == np.uint8
    assert out[:4].any()                     # the mask is visible
    assert not out[4:].any()                 # and nothing else was touched


# --------------------------------------------------------------------------------------------
# 5. end to end, on a stub masker and a stub decoder: the artifact and the sheets
# --------------------------------------------------------------------------------------------


def _run_end_to_end(tmp_path, monkeypatch, *, max_frames=6, tiles=4, max_fraction=None):
    fractions = {f"episode_{i:06d}": [0.1, 0.9, 0.95, 0.2] for i in range(5)}
    pooled = _pooled(fractions)
    pooled_path = _write_pooled(tmp_path, pooled)
    corpus = _corpus(tmp_path, pooled)

    class _RC:
        @staticmethod
        def build_masker():
            return StubMasker()

        decode_clip = staticmethod(_decoder(fractions))

    masker = StubMasker()
    masker.preflight = lambda: None  # type: ignore[attr-defined]
    _RC.build_masker = staticmethod(lambda: masker)  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "robot_composite", _RC)

    out = tmp_path / "out"
    argv = ["--pooled", str(pooled_path), "--corpus", str(corpus), "--out", str(out),
            "--threshold", "0.7", "--max-frames", str(max_frames),
            "--sheet-tiles", str(tiles), "--sheet-cols", "2"]
    if max_fraction is not None:
        argv += ["--max-fraction", str(max_fraction)]
    code = rats.main(argv)
    return code, out, json.loads((out / "TAIL_SAMPLE.json").read_text(encoding="utf-8"))


def test_end_to_end_writes_sheets_and_an_artifact(tmp_path, monkeypatch):
    code, out, artifact = _run_end_to_end(tmp_path, monkeypatch)
    assert code == 0
    assert len(artifact["frames"]) == 6
    assert artifact["sheets"] == ["sheets/area-tail-00.png", "sheets/area-tail-01.png"]
    for name in artifact["sheets"]:
        assert (out / name).is_file()
    assert artifact["population"]["frames_at_or_above_threshold"] == 10
    assert artifact["population"]["episodes_with_tail_frames"] == 5
    assert artifact["population"]["episodes_sampled"] == 5
    assert artifact["population"]["episodes_not_sampled"] == 0


def test_artifact_carries_provenance_and_both_fractions_per_frame(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    assert artifact["source"]["git_commit"] == "0" * 40
    assert artifact["source"]["source_manifest_sha256"] == "f" * 64
    for record in artifact["frames"]:
        assert "recorded_fraction" in record and "recomputed_fraction" in record
        assert "delta" in record and "mismatch" in record
        assert record["sheet"].startswith("area-tail-")
    assert artifact["mismatch"]["count"] == len(artifact["mismatch"]["frames"])


def test_artifact_carries_the_selection_rule_verbatim(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    rule = artifact["selection_rule"]["rule"]
    assert rule == rats.SELECTION_RULE_TEXT
    assert "round-robin" in rule.lower() or "cycling" in rule.lower()
    assert artifact["selection_rule"]["max_frames"] == 6
    assert "none" in artifact["selection_rule"]["rng"].lower()
    # The artifact must be enough to rebuild the sample without reading the sheets.
    assert artifact["threshold"]["value"] == 0.7


def test_artifact_carries_the_hardware_caveat(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    caveat = artifact["hardware_caveat"]
    assert "H200" in caveat and "5090" in caveat
    assert "detector-noise-floor" in caveat
    assert "mismatch count" in caveat


def test_artifact_states_it_writes_no_bound_and_discharges_nothing(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    assert artifact["writes_a_bound"] is False
    assert isinstance(artifact["not_a_discharge"], str) and artifact["not_a_discharge"]
    lowered = artifact["not_a_discharge"].lower()
    assert "discharges no blocker" in lowered
    assert "no bound" in lowered


# --------------------------------------------------------------------------------------------
# 6. the two things no code path here may do
# --------------------------------------------------------------------------------------------


def _walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]")


def test_no_code_path_writes_a_bound(tmp_path, monkeypatch):
    """Neither the source nor the artifact ever assigns ``max_frame_fraction`` a value."""
    assert not re.search(r"max_frame_fraction\s*[=:]\s*[-+0-9.\"']", SOURCE), (
        "this script may name the bound but never set it")
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    for _, key, value in _walk(artifact):
        assert key != "max_frame_fraction", f"the artifact assigned a bound: {key}={value!r}"
        if isinstance(value, str):
            assert not re.search(r"max_frame_fraction\s*[=:]\s*[-+0-9]", value), value


def test_no_code_path_records_a_review_flag(tmp_path, monkeypatch):
    """No reviewer-confirmation flag exists here, under that name or any other.

    ``audit_apple_masks`` carries a ``human_review`` block whose flag a PERSON fills in. This script
    does not carry one at all, which is the stronger statement: there is no field a run could set.
    """
    assert "looked_at" not in SOURCE
    assert "human_review" not in SOURCE
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    for _, key, value in _walk(artifact):
        assert "looked_at" not in key
        assert key != "human_review"
        assert not (isinstance(value, bool) and value and "review" in key and key != "records_no_"
                    "reviewer_confirmation")


def test_the_script_does_not_touch_the_masker_configuration():
    """No threshold, prompt or operating point is settable from here."""
    for forbidden in ("ROBOT_TEXT_PROMPT =", "box_thr", "text_threshold =", "ROBOT_MASK_OBJECT_MAX"):
        assert forbidden not in SOURCE, forbidden
    assert "build_masker()" in SOURCE, "the masker comes from robot_composite, unmodified"


def test_the_decode_path_is_the_measurement_s_own():
    assert "rc.decode_clip" in SOURCE
    for pattern in (r"\bimport\s+cv2\b", r"\bcv2\.VideoCapture\b", r"\bimport\s+imageio\b",
                    r"\biio\.", r"\bimageio\.v3\b"):
        assert not re.search(pattern, SOURCE), (
            f"{pattern}: a second decoder here would make the tile a picture of a different frame")


# --------------------------------------------------------------------------------------------
# 7. --max-fraction: the band, its default, and what the artifact has to say about it
# --------------------------------------------------------------------------------------------


def _band_pooled() -> dict:
    """Two episodes straddling a band whose edges land exactly on measured values."""
    return _pooled({
        "episode_000001": [0.10, 0.36, 0.50, 0.6015462239583333, 0.68, 0.95],
        "episode_000002": [0.35, 0.42, 0.7469694010416666, 0.99],
    })


def test_the_band_excludes_frames_above_max_fraction():
    """The whole point: a frame above the upper edge is not a candidate, however large it is."""
    candidates = rats.tail_candidates(_band_pooled(), 0.36, 0.6015462239583333)
    assert candidates["episode_000001"] == [
        (1, 0.36), (2, 0.50), (3, 0.6015462239583333)]
    assert candidates["episode_000002"] == [(1, 0.42)]
    every = [f for hits in candidates.values() for _, f in hits]
    assert max(every) <= 0.6015462239583333
    assert min(every) >= 0.36


def test_both_edges_of_the_band_are_inclusive():
    pooled = _pooled({"episode_000000": [0.359999, 0.36, 0.5, 0.6015462239583333, 0.6015463]})
    candidates = rats.tail_candidates(pooled, 0.36, 0.6015462239583333)
    assert candidates["episode_000000"] == [(1, 0.36), (2, 0.5), (3, 0.6015462239583333)]


def test_the_default_is_no_upper_limit_and_keeps_todays_open_tail():
    """Omitted, None, and DEFAULT_MAX_FRACTION are the same thing: the open upper tail."""
    pooled = _band_pooled()
    assert rats.DEFAULT_MAX_FRACTION is None
    omitted = rats.tail_candidates(pooled, 0.36)
    explicit_none = rats.tail_candidates(pooled, 0.36, None)
    defaulted = rats.tail_candidates(pooled, 0.36, rats.DEFAULT_MAX_FRACTION)
    assert omitted == explicit_none == defaulted
    every = [f for hits in omitted.values() for _, f in hits]
    assert max(every) == 0.99                       # nothing at the top was dropped
    assert len(every) == 8


def test_the_default_argparse_value_is_none():
    args = rats.build_argparser().parse_args([])
    assert args.max_fraction is None
    assert args.threshold == rats.DEFAULT_TAIL_EDGE


def test_a_band_narrower_than_a_single_value_is_still_honoured():
    pooled = _pooled({"episode_000000": [0.4, 0.5, 0.6]})
    assert rats.tail_candidates(pooled, 0.5, 0.5)["episode_000000"] == [(1, 0.5)]


def test_an_upper_edge_below_the_lower_edge_is_refused():
    """An empty-by-construction band is a typo, and neither number is a bound either way."""
    with pytest.raises(rats.TailLookError) as excinfo:
        rats.tail_candidates(_band_pooled(), 0.6, 0.36)
    assert "max-fraction" in str(excinfo.value)


def test_main_exits_non_zero_when_the_band_holds_no_frame(tmp_path, monkeypatch, capsys):
    pooled = _pooled({"episode_000000": [0.1, 0.9]})
    path = _write_pooled(tmp_path, pooled)
    corpus = _corpus(tmp_path, pooled)
    code = rats.main(["--pooled", str(path), "--corpus", str(corpus),
                      "--out", str(tmp_path / "out"),
                      "--threshold", "0.3", "--max-fraction", "0.4"])
    assert code != 0
    assert "band" in capsys.readouterr().err.lower()
    assert not (tmp_path / "out" / "TAIL_SAMPLE.json").exists()


def test_the_band_is_deterministic_like_the_open_tail():
    pooled = _band_pooled()
    first = rats.select_frames(rats.tail_candidates(pooled, 0.36, 0.6015462239583333), 3)
    second = rats.select_frames(
        rats.tail_candidates(json.loads(json.dumps(pooled)), 0.36, 0.6015462239583333), 3)
    assert first == second


def test_the_sheet_header_carries_both_edges_of_the_band():
    closed = rats.sheet_title(0, 0.36, 12, 0.6015462239583333)
    assert "0.360000" in closed and "0.601546" in closed
    assert "<=" in closed
    open_tail = rats.sheet_title(0, 0.36, 12)
    assert ">=" in open_tail and "0.360000" in open_tail
    assert rats.band_text(0.36, None) == "fraction >= 0.360000"


def test_the_artifact_records_the_band_and_not_only_the_threshold(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch, max_fraction=0.9)
    band = artifact["band"]
    assert band["lower_inclusive"] == 0.7
    assert band["upper_inclusive"] == 0.9
    assert band["open_at_the_top"] is False
    assert "0.700000" in band["selection"] and "0.900000" in band["selection"]
    # ...and it keeps saying, plainly, that neither number is a bound.
    text = " ".join(str(v) for v in band.values()).lower()
    assert "not a bound" in text or "neither edge is a bound" in text
    assert "candidate bound" in text
    assert "selection window" in band["note"].lower()
    assert artifact["threshold"]["max_fraction"] == 0.9
    assert artifact["selection_rule"]["band"] == band["selection"]
    assert artifact["population"]["max_fraction"] == 0.9


def test_the_artifact_says_a_look_below_the_gap_is_the_control_for_a_look_above(
        tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch, max_fraction=0.9)
    why = artifact["band"]["why_a_band"].lower()
    assert "control" in why
    assert "separable only when read from above" in why


def test_the_artifact_records_an_open_band_as_open(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    band = artifact["band"]
    assert band["upper_inclusive"] is None
    assert band["open_at_the_top"] is True
    assert band["selection"] == "fraction >= 0.700000"
    assert artifact["population"]["max_fraction"] is None


def test_the_band_run_renders_only_frames_inside_the_band(tmp_path, monkeypatch):
    code, _, artifact = _run_end_to_end(tmp_path, monkeypatch, max_frames=10, max_fraction=0.9)
    assert code == 0
    # Five episodes of [0.1, 0.9, 0.95, 0.2]: ten frames are >= 0.7, five are inside [0.7, 0.9].
    assert artifact["population"]["frames_in_band"] == 5
    assert artifact["population"]["frames_at_or_above_threshold"] == 10
    assert len(artifact["frames"]) == 5
    for record in artifact["frames"]:
        assert 0.7 <= record["recorded_fraction"] <= 0.9


def test_the_open_tail_run_still_reports_the_same_population(tmp_path, monkeypatch):
    """The default's population block is byte-for-byte the statement it made before the band."""
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch)
    assert artifact["population"]["frames_at_or_above_threshold"] == 10
    assert artifact["population"]["frames_in_band"] == 10


def test_the_band_still_writes_no_bound(tmp_path, monkeypatch):
    _, _, artifact = _run_end_to_end(tmp_path, monkeypatch, max_fraction=0.9)
    assert artifact["writes_a_bound"] is False
    for _, key, value in _walk(artifact):
        assert key != "max_frame_fraction"
        if isinstance(value, str):
            assert not re.search(r"max_frame_fraction\s*[=:]\s*[-+0-9]", value), value
